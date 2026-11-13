from types import SimpleNamespace

from transcription import transcribe, release_models
from transcription.result import TranscriptionResult, Word


def test_chinese_api_dispatch_cache_and_export(monkeypatch, tmp_path):
    import transcription.api as api
    import transcription.engines.paraformer as module
    source = tmp_path / "sample.wav"
    source.write_bytes(b"test")
    monkeypatch.setattr(api, "_device", lambda dev: "cpu")
    monkeypatch.setattr(api, "prepare_audio",
                        lambda path, staging, **kw: SimpleNamespace(
                            audio_path=source, extracted=False, warnings=(),
                            original_stream_start=0.0, extracted_stream_start=0.0,
                            candidate_video_offset=0.0))
    events = {"init": 0, "close": 0}
    class FakeEngine:
        def __init__(self, **kwargs):
            events["init"] += 1
        def transcribe(self, file):
            return TranscriptionResult(
                "partial", "Chinese recognition complete", "paraformer_zh",
                source_path=source, text="你好",
                words=[Word("你", .0, .2, 0, 1, "aligned"),
                       Word("好", .2, .4, 1, 2, "aligned")],
            )
        def close(self):
            events["close"] += 1
    monkeypatch.setattr(module, "ParaformerEngine", FakeEngine)
    try:
        first = transcribe(source, "paraformer_zh")
        second = transcribe(source, "paraformer_zh")
        assert first.text == second.text == "你好"
        assert first.status == "success"
        assert len(first.files) == 4
        assert second.files["json"].name == "sample (1).json"
        assert events["init"] == 1
    finally:
        release_models()
    assert events["close"] == 1
