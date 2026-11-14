"""Conservative packet and decoded-sample timeline checks for video."""
from __future__ import annotations
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class PacketTimelineCheck:
    verified: bool
    offset_seconds: float | None
    reason: str

@dataclass(frozen=True, slots=True)
class PlaybackMapping:
    verified: bool
    offset_seconds: float | None
    reason: str

def _packets(path: Path, ffprobe: str, stream_index: int | None,
             *, max_packets: int) -> tuple[list[tuple[float, float, int]], str | None]:
    cmd = [ffprobe, "-v", "error", "-show_packets",
           "-show_entries", "packet=stream_index,pts_time,duration_time,size,side_data_list",
           "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=False, stdin=subprocess.DEVNULL)
    except OSError as exc:
        return [], f"FFprobe packet inspection failed: {exc}"
    if proc.returncode:
        return [], f"FFprobe packet inspection failed (exit {proc.returncode})"
    try:
        records = json.loads(proc.stdout)["packets"]
        if not isinstance(records, list):
            return [], "FFprobe packet list is malformed"
    except (ValueError, KeyError, TypeError):
        return [], "FFprobe returned invalid packet metadata"
    selected = [p for p in records if stream_index is None
                or p.get("stream_index") == stream_index]
    if not selected:
        return [], "No matching audio packets"
    if len(selected) > max_packets:
        return [], "Too many audio packets for bounded verification"
    packets = []
    for packet in selected:
        if packet.get("side_data_list"):
            return [], "Packet side data requires decoder-time validation"
        try:
            pts, duration, size = (float(packet["pts_time"]),
                                   float(packet["duration_time"]), int(packet["size"]))
        except (KeyError, ValueError, TypeError):
            return [], "Incomplete packet timestamp, duration or size"
        if not (math.isfinite(pts) and math.isfinite(duration) and duration > 0 and size > 0):
            return [], "Invalid audio packet metadata"
        packets.append((pts, duration, size))
    return packets, None

def verify_stream_copy_packet_timeline(
    source: Path, extracted: Path, original_stream_index: int,
    *, ffprobe: str = "ffprobe", max_packets: int = 100000,
    tolerance_seconds: float = .002,
) -> PacketTimelineCheck:
    before, error = _packets(source, ffprobe, original_stream_index, max_packets=max_packets)
    if error:
        return PacketTimelineCheck(False, None, "Source: " + error)
    after, error = _packets(extracted, ffprobe, None, max_packets=max_packets)
    if error:
        return PacketTimelineCheck(False, None, "Extracted: " + error)
    if len(before) != len(after):
        return PacketTimelineCheck(False, None, "Source/extracted packet counts differ")
    offset = before[0][0] - after[0][0]
    for i, (original, copied) in enumerate(zip(before, after)):
        src_pts, src_duration, src_size = original
        dst_pts, dst_duration, dst_size = copied
        if src_size != dst_size or abs(src_duration - dst_duration) > tolerance_seconds:
            return PacketTimelineCheck(False, None, f"Packet {i} content/duration mismatch")
        if abs((src_pts - dst_pts) - offset) > tolerance_seconds:
            return PacketTimelineCheck(False, None, f"Packet {i} time offset is inconsistent")
        if i:
            prev_src, prev_duration, _ = before[i - 1]
            prev_dst, prev_dst_duration, _ = after[i - 1]
            if (abs(src_pts - (prev_src + prev_duration)) > tolerance_seconds
                    or abs(dst_pts - (prev_dst + prev_dst_duration)) > tolerance_seconds):
                return PacketTimelineCheck(False, None, f"Packet {i} has a gap or overlap")
    return PacketTimelineCheck(
        True, offset,
        "Continuous matching packet timestamps; decoded-sample and video-playback alignment remain unverified",
    )

def _probe_frames(path: Path, ffprobe: str, selector: str, fields: str) -> list[dict]:
    command = [ffprobe, "-v", "error", "-select_streams", selector,
               "-show_frames", "-show_entries", "frame=" + fields, "-of", "json", str(path)]
    proc = subprocess.run(command, capture_output=True, check=False, stdin=subprocess.DEVNULL)
    if proc.returncode:
        raise ValueError("FFprobe decoded-frame inspection failed")
    data = json.loads(proc.stdout)["frames"]
    if not isinstance(data, list) or not data:
        raise ValueError("No decoded frames")
    return data

def _decoded_audio_frames(path: Path, selector: str, ffprobe: str):
    frames = _probe_frames(path, ffprobe, selector, "best_effort_timestamp_time,nb_samples")
    result = []
    for frame in frames:
        timestamp = float(frame["best_effort_timestamp_time"])
        samples = int(frame["nb_samples"])
        if not math.isfinite(timestamp) or samples <= 0:
            raise ValueError("Invalid decoded-audio frame")
        result.append((timestamp, samples))
    return result

def _hash_pcm(path: Path, stream: str, ffmpeg: str) -> str:
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", stream,
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
               "-f", "hash", "-hash", "sha256", "-"]
    proc = subprocess.run(command, capture_output=True, check=False, stdin=subprocess.DEVNULL)
    if proc.returncode:
        raise ValueError("FFmpeg decoded PCM checksum failed")
    digest = proc.stdout.decode("ascii").strip()
    if not digest.startswith("SHA256=") or len(digest) != 71:
        raise ValueError("Invalid decoded PCM checksum")
    return digest

def verify_playback_mapping(
    source: Path, extracted: Path, recognition_wav: Path,
    original_stream_index: int, sample_rate: int,
    *, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg",
    tolerance_seconds: float = .002, max_frames: int = 1000000,
) -> PlaybackMapping:
    """Validate continuous decoded audio and exact ASR-input PCM identity.

    Audio sample zero of the 16kHz mono recognition WAV maps to the first
    decoded original audio frame PTS, relative to original video first-frame
    presentation PTS. Reject ambiguous video gaps and inconsistent audio.
    """
    try:
        if not sample_rate or sample_rate <= 0:
            raise ValueError("Audio sample rate unknown")
        video = _probe_frames(source, ffprobe, "v:0", "best_effort_timestamp_time")
        if len(video) > max_frames:
            raise ValueError("Too many video frames to verify")
        video_pts = [float(f["best_effort_timestamp_time"]) for f in video]
        if not all(math.isfinite(p) for p in video_pts):
            raise ValueError("Invalid video presentation timestamps")
        if any(not .0 < video_pts[i] - video_pts[i - 1] <= .2
               for i in range(1, len(video_pts))):
            raise ValueError("Discontinuous or nonmonotonic video presentation timestamps")
        video_t0 = video_pts[0]
        before = _decoded_audio_frames(source, str(original_stream_index), ffprobe)
        after = _decoded_audio_frames(extracted, "a:0", ffprobe)
        if len(before) != len(after) or len(before) > max_frames:
            raise ValueError("Decoded audio frame count mismatch or too large")
        shift = before[0][0] - after[0][0]
        for i, ((a, n), (b, m)) in enumerate(zip(before, after)):
            if n != m or abs(a - b - shift) > tolerance_seconds:
                raise ValueError(f"Decoded audio frame mismatch at {i}")
            if i:
                dt = before[i - 1][1] / sample_rate
                if abs(a - before[i - 1][0] - dt) > tolerance_seconds:
                    raise ValueError(f"Original audio discontinuity at frame {i}")
                if abs(b - after[i - 1][0] - dt) > tolerance_seconds:
                    raise ValueError(f"Extracted audio discontinuity at frame {i}")
        if before[0][0] < video_t0 - tolerance_seconds:
            raise ValueError("Audio starts before first video frame")
        hashes = (
            _hash_pcm(source, f"0:{original_stream_index}", ffmpeg),
            _hash_pcm(extracted, "0:a:0", ffmpeg),
            _hash_pcm(recognition_wav, "0:a:0", ffmpeg),
        )
        if len(set(hashes)) != 1:
            raise ValueError("Decoded PCM mismatch between original, extracted and ASR audio")
        return PlaybackMapping(True, before[0][0] - video_t0,
                               "Decoded samples and continuous presentation times verified")
    except (ValueError, KeyError, OSError, TypeError, UnicodeDecodeError) as exc:
        return PlaybackMapping(False, None, str(exc))
