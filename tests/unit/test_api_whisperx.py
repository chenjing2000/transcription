from types import SimpleNamespace

from transcription import transcribe
from transcription.model_manager import release_models
from transcription.result import TranscriptionResult, Word


def test_api_english_audio_path(monkeypatch, tmp_path):
    import transcription.api as api
    import transcription.engines.whisperx as module
    source = tmp_path / "test.wav"
    source.write_bytes(b"mock")
    monkeypatch.setattr(api, "_device", lambda x: "cpu")
    monkeypatch.setattr(api, "prepare_audio",
                        lambda path, directory, **k: SimpleNamespace(
                            audio_path=source, extracted=False, warnings=(),
                            original_stream_start=0.0, extracted_stream_start=0.0,
                            candidate_video_offset=0.0))
    class FakeEngine:
        initialized = 0
        def __init__(self, **kwargs):
            type(self).initialized += 1
        def transcribe(self, file):
            return TranscriptionResult(
                "partial", "recognized", "whisperx_large_v3",
                source_path=source, text="Hello",
                words=[Word("Hello", .1, .4, 0, 5, "aligned")])
        def close(self): pass
    monkeypatch.setattr(module, "WhisperXEngine", FakeEngine)
    try:
        result = transcribe(source, "whisperx_large_v3")
        assert result.status == "success"
        assert result.text == "Hello"
        assert len(result.files) == 2
        assert result.files["timed_json"].is_file()
        assert result.audio_path == source
        result2 = transcribe(source, "whisperx_large_v3")
        assert result2.files["timed_json"].name == "test (1).timed.json"
        assert FakeEngine.initialized == 1
    finally:
        release_models()
