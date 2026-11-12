from types import SimpleNamespace
import sys

from transcription import transcribe
from transcription.model_manager import release_models
from transcription.result import TranscriptionResult


def test_api_english_audio_path(monkeypatch, tmp_path):
    import transcription.api as api
    import transcription.engines.whisperx as module
    source = tmp_path / "test.wav"
    source.write_bytes(b"mock")
    monkeypatch.setattr(api, "_device", lambda x: "cpu")
    monkeypatch.setattr(api, "prepare_audio",
                        lambda path, directory, **k: SimpleNamespace(audio_path=source, extracted=False, warnings=()))
    class FakeEngine:
        initialized = 0
        def __init__(self, **kwargs):
            type(self).initialized += 1
        def transcribe(self, file):
            return TranscriptionResult("partial", "recognized, exports pending", "whisperx_large_v3",
                                       source_path=source, text="Hello")
        def close(self): pass
    monkeypatch.setattr(module, "WhisperXEngine", FakeEngine)
    try:
        result = transcribe(source, "whisperx_large_v3")
        assert result.status == "partial"
        assert result.text == "Hello"
        assert result.files == {}
        assert result.audio_path == source
        result2 = transcribe(source, "whisperx_large_v3")
        assert result2.text == "Hello"
        assert FakeEngine.initialized == 1
    finally:
        release_models()
