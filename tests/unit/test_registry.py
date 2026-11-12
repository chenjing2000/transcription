from transcription import list_engines
from transcription.errors import TranscriptionError
from transcription.registry import get_engine


def test_initial_engines():
    items = list_engines()
    assert [x.name for x in items] == ["paraformer_zh", "whisperx_large_v3"]
    assert get_engine("paraformer_zh").language == "zh"
    assert get_engine("whisperx_large_v3").model_id == "large-v3"


def test_unknown_engine():
    try:
        get_engine("not_real")
    except TranscriptionError as e:
        assert e.code == "ENGINE_NOT_FOUND"
    else:
        assert False, "expected engine discovery failure"
