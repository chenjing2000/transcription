"""End-to-end media + output on Windows runner with a simulated recognition engine."""
import json
import shutil
import subprocess

import pytest

from transcription import release_models, transcribe
from transcription.media import probe_media
from transcription.result import TranscriptionResult, Word


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg and FFprobe required")
def test_video_stream_copy_and_three_outputs(monkeypatch, tmp_path):
    import transcription.api as api
    import transcription.engines.whisperx as module

    source = tmp_path / "中文 lecture.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
        "-c:a", "aac", "-b:a", "96k", str(source),
    ], check=True)
    original_bytes = source.read_bytes()
    original_audio = probe_media(source).audio_streams[0]

    class FakeWhisperXEngine:
        def __init__(self, **kwargs):
            pass
        def transcribe(self, audio):
            assert audio.suffix == ".wav"
            assert audio.is_file()
            return TranscriptionResult(
                status="partial", info="recognition", engine="whisperx_large_v3",
                source_path=audio, text="Hello world.",
                words=[
                    Word("Hello", .1, .25, 0, 5, "aligned"),
                    Word("world.", .26, .45, 6, 12, "aligned"),
                ],
            )
        def close(self):
            pass

    monkeypatch.setattr(api, "_device", lambda requested: "cpu")
    monkeypatch.setattr(module, "WhisperXEngine", FakeWhisperXEngine)
    try:
        first = transcribe(source, "whisperx_large_v3")
        assert first.status == "partial"  # video timeline not verified
        assert set(first.files) == {"audio", "timed_json", "txt"}
        assert first.audio_path == first.files["audio"]
        assert first.files["audio"].suffix == ".m4a"
        assert source.read_bytes() == original_bytes
        copied = probe_media(first.files["audio"], allow_extracted=True).audio_streams[0]
        assert original_audio.codec == copied.codec
        assert original_audio.sample_rate == copied.sample_rate
        assert original_audio.channels == copied.channels
        timed = json.loads(first.files["timed_json"].read_text(encoding="utf8"))
        assert [word["word"] for word in timed["word_segments"]] == ["Hello", "world."]
        assert timed["segments"][0]["text"] == "Hello world."
        assert timed["timeline"]["video_time_mapping_verified"] is False
        assert timed["timeline"]["reference"] == "source_video_playback"
        assert timed["segments"][0]["start"] is None
        assert timed["segments"][0]["alignment_status"] == "missing"
        assert all(w["start"] is None and w["end"] is None for w in timed["word_segments"])
        second = transcribe(source, "whisperx_large_v3")
        assert second.status == "partial"
        assert second.files["audio"].name == "中文 lecture (1).m4a"
        assert second.files["timed_json"].name == "中文 lecture (1).timed.json"
        assert first.files["timed_json"].exists()
    finally:
        release_models()
