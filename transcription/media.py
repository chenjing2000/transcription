"""Media probing and original-codec audio extraction (Phase 3).

Extracted files are written to a caller-supplied staging directory, NOT directly
published. A later phase is responsible for group numbering and atomic publishing.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import TranscriptionError

SUPPORTED_INPUTS = frozenset({".mp4", ".mkv", ".mp3", ".wav"})
VIDEO_INPUTS = frozenset({".mp4", ".mkv"})
# Codec -> a stream-copy-compatible container/extension. Unsupported codecs fail
# rather than silently transcoding the user's source audio.
COPY_EXTENSIONS = {
    "aac": ".m4a",
    "mp3": ".mp3",
    "opus": ".opus",
    "vorbis": ".ogg",
    "flac": ".flac",
    "ac3": ".ac3",
    "eac3": ".eac3",
    "alac": ".m4a",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s32le": ".wav",
    "pcm_f32le": ".wav",
    "pcm_f64le": ".wav",
    "pcm_u8": ".wav",
}


@dataclass(frozen=True, slots=True)
class AudioStream:
    index: int                  # absolute FFprobe stream index
    codec: str
    sample_rate: int | None
    channels: int | None
    bit_rate: int | None
    start_time: float | None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    source_path: Path
    duration: float | None
    audio_streams: tuple[AudioStream, ...]
    is_video: bool


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    source_path: Path
    audio_path: Path
    stream: AudioStream
    extracted: bool
    # Both time bases are kept. A constant offset is only a candidate for
    # simple continuous tracks; it must be validated before video highlighting.
    original_stream_start: float | None
    extracted_stream_start: float | None
    warnings: tuple[str, ...] = ()

    @property
    def candidate_video_offset(self) -> float | None:
        if self.original_stream_start is None or self.extracted_stream_start is None:
            return None
        return self.original_stream_start - self.extracted_stream_start


def _tool_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise TranscriptionError("MEDIA_TOOL_NOT_FOUND", f"{name} was not found on PATH")
    return path


def _run(command: list[str], description: str) -> str:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise TranscriptionError("MEDIA_TOOL_FAILED", f"{description}: {exc}") from exc
    if proc.returncode:
        message = proc.stderr.strip()[-1500:] or f"exit code {proc.returncode}"
        raise TranscriptionError("MEDIA_TOOL_FAILED", f"{description}: {message}")
    return proc.stdout


def _optional_float(value: object) -> float | None:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_media(file_path: str | Path, *, ffprobe: str | None = None) -> MediaInfo:
    """Inspect an MP4/MKV/MP3/WAV and its audio streams without modifying it."""
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_INPUTS:
        raise TranscriptionError("UNSUPPORTED_FORMAT", f"Unsupported media extension: {path.suffix}")
    if not path.is_file():
        raise TranscriptionError("INVALID_INPUT", f"Input file does not exist: {path}")
    executable = ffprobe or _tool_path("ffprobe")
    payload = _run(
        [executable, "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        "FFprobe inspection failed",
    )
    try:
        info = json.loads(payload)
        streams = info["streams"]
        if not isinstance(streams, list):
            raise ValueError("streams is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        raise TranscriptionError("INVALID_MEDIA_METADATA", f"Invalid FFprobe JSON: {exc}") from exc
    audio = tuple(
        AudioStream(
            index=int(stream["index"]),
            codec=str(stream.get("codec_name", "unknown")),
            sample_rate=_optional_int(stream.get("sample_rate")),
            channels=_optional_int(stream.get("channels")),
            bit_rate=_optional_int(stream.get("bit_rate")),
            start_time=_optional_float(stream.get("start_time")),
        )
        for stream in streams if stream.get("codec_type") == "audio"
    )
    if not audio:
        raise TranscriptionError("NO_AUDIO_STREAM", f"No audio stream found: {path}")
    duration = _optional_float((info.get("format") or {}).get("duration"))
    return MediaInfo(path, duration, audio, path.suffix.lower() in VIDEO_INPUTS)


def prepare_audio(
    file_path: str | Path,
    staging_dir: str | Path,
    *,
    audio_track: int = 0,
    ffprobe: str | None = None,
    ffmpeg: str | None = None,
    output_stem: str | None = None,
) -> PreparedAudio:
    """Stream-copy a selected video audio track; audio inputs are reused as-is.

    `audio_track` is a zero-based index among AUDIO streams, not a global
    FFmpeg stream index. Existing staging files are never overwritten.
    """
    if isinstance(audio_track, bool) or not isinstance(audio_track, int) or audio_track < 0:
        raise TranscriptionError("INVALID_AUDIO_TRACK", "audio_track must be a nonnegative integer")
    media = probe_media(file_path, ffprobe=ffprobe)
    if audio_track >= len(media.audio_streams):
        raise TranscriptionError(
            "AUDIO_TRACK_NOT_FOUND",
            f"Audio track {audio_track} unavailable (found {len(media.audio_streams)})",
        )
    stream = media.audio_streams[audio_track]
    if not media.is_video:
        if audio_track != 0:
            raise TranscriptionError("AUDIO_TRACK_NOT_FOUND", "Audio inputs support audio_track=0 only")
        return PreparedAudio(media.source_path, media.source_path, stream, False,
                             stream.start_time, stream.start_time)
    extension = COPY_EXTENSIONS.get(stream.codec)
    if extension is None:
        raise TranscriptionError(
            "UNSUPPORTED_AUDIO_CODEC",
            f"No verified stream-copy container for audio codec {stream.codec!r}",
        )
    folder = Path(staging_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = output_stem or media.source_path.stem
    if not stem or Path(stem).name != stem or stem in {".", ".."}:
        raise TranscriptionError("INVALID_OUTPUT_NAME", "output_stem must be a plain filename stem")
    destination = folder / (stem + extension)
    if destination.exists():
        raise TranscriptionError("OUTPUT_EXISTS", f"Refusing to overwrite: {destination}")

    executable = ffmpeg or _tool_path("ffmpeg")
    try:
        _run([
            executable, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(media.source_path),
            "-map", f"0:{stream.index}", "-vn", "-c:a", "copy",
            "-n", str(destination),
        ], "FFmpeg stream-copy failed")
        if not destination.is_file() or destination.stat().st_size == 0:
            raise TranscriptionError("AUDIO_EXTRACTION_FAILED", "No audio file was produced")
        extracted = probe_media(destination, ffprobe=ffprobe)
        copied = extracted.audio_streams[0]
        if (copied.codec != stream.codec or copied.sample_rate != stream.sample_rate
                or copied.channels != stream.channels):
            raise TranscriptionError(
                "AUDIO_EXTRACTION_FAILED",
                "Stream-copy validation failed: codec, sample rate or channel count changed",
            )
    except Exception:
        # Never leave an incomplete staging artifact behind.
        destination.unlink(missing_ok=True)
        raise
    warnings = []
    if stream.start_time is None or copied.start_time is None:
        warnings.append("Stream start time missing; original video time mapping is unverified")
    else:
        warnings.append(
            "Stream start times recorded; constant-offset mapping must be verified "
            "for edit lists or discontinuous timestamps before video highlighting"
        )
    return PreparedAudio(
        media.source_path, destination, stream, True,
        stream.start_time, copied.start_time, tuple(warnings),
    )
