"""Real-video timing integration and safe fallbacks."""
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from transcription.media import prepare_audio
from transcription.output import publish_result
from transcription.result import TranscriptionResult, Word

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and FFprobe required",
)


def _video(path, *, audio_offset=0.0):
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=2",
    ]
    if audio_offset:
        command += ["-itsoffset", str(audio_offset)]
    command += [
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
        "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4",
        "-c:a", "aac", str(path),
    ]
    subprocess.run(command, check=True)


@pytest.mark.parametrize("audio_offset", [0.0, 0.3])
def test_real_video_playback_mapping_and_english_export(tmp_path, audio_offset):
    src = tmp_path / "lecture.mp4"
    _video(src, audio_offset=audio_offset)
    prepared = prepare_audio(src, tmp_path / "working")
    assert prepared.recognition_audio_path.is_file()
    assert prepared.recognition_audio_path.suffix == ".wav"
    assert prepared.audio_path.suffix == ".m4a"
    assert prepared.video_playback_offset is not None, prepared.video_mapping_reason
    assert abs(prepared.video_playback_offset - audio_offset) < .05

    result = TranscriptionResult(
        status="partial", info="mock ASR", engine="whisperx_large_v3",
        source_path=src, text="Hello.",
        words=[Word("Hello.", .1, .4, 0, 6, "aligned")],
    )
    result.warnings.extend(prepared.warnings)
    published = publish_result(result, prepared)
    data = json.loads(published.files["timed_json"].read_text(encoding="utf8"))
    assert data["timeline"]["mapping_verified"] is True
    assert data["timeline"]["reference"] == "source_video_playback"
    assert data["word_segments"][0]["start"] == round(.1 + prepared.video_playback_offset, 3)
    assert data["word_segments"][0]["end"] == round(.4 + prepared.video_playback_offset, 3)
    assert published.files["audio"].suffix == ".m4a"
    assert set(published.files) == {"audio", "timed_json", "txt"}


def test_non_video_mp4_retains_text_but_not_unverified_video_timestamps(tmp_path):
    src = tmp_path / "audio_only.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
        "-c:a", "aac", str(src),
    ], check=True)
    prepared = prepare_audio(src, tmp_path / "working")
    assert prepared.video_playback_offset is None
    result = TranscriptionResult(
        "partial", "ASR", "whisperx_large_v3", src,
        text="Hello.", words=[Word("Hello.", .1, .3, 0, 6, "aligned")],
    )
    result.warnings.extend(prepared.warnings)
    published = publish_result(result, prepared)
    data = json.loads(published.files["timed_json"].read_text(encoding="utf8"))
    assert published.status == "partial"
    assert data["timeline"]["mapping_verified"] is False
    assert data["word_segments"][0]["start"] is None
