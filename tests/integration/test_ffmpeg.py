"""Actual local FFmpeg smoke tests: runnable on Windows and Linux."""
from pathlib import Path
import shutil
import subprocess

import pytest
from transcription.media import prepare_audio, probe_media


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg and FFprobe not installed")
def test_aac_video_stream_copy(tmp_path: Path):
    source = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=0.65",
         "-c:a", "aac", "-b:a", "96k", str(source)],
        check=True,
    )
    original = probe_media(source)
    prepared = prepare_audio(source, tmp_path / "stage")
    assert prepared.extracted
    assert prepared.audio_path.suffix == ".m4a"
    assert prepared.audio_path.exists()
    copied = probe_media(prepared.audio_path, allow_extracted=True)
    assert original.audio_streams[0].codec == copied.audio_streams[0].codec == "aac"
    assert original.audio_streams[0].sample_rate == copied.audio_streams[0].sample_rate
    assert original.audio_streams[0].channels == copied.audio_streams[0].channels


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg and FFprobe not installed")
def test_video_does_not_overwrite_existing_audio(tmp_path: Path):
    source = tmp_path / "sample.mkv"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=0.45",
         "-c:a", "flac", str(source)],
        check=True,
    )
    first = prepare_audio(source, tmp_path / "stage")
    assert first.audio_path.suffix == ".flac"
    assert first.audio_path.is_file()
    from transcription.errors import TranscriptionError
    with pytest.raises(TranscriptionError) as error:
        prepare_audio(source, tmp_path / "stage")
    assert error.value.code == "OUTPUT_EXISTS"
