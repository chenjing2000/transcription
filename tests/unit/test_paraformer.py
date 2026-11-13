"""No FunASR/model downloads required; mocked adapter and normalization tests."""
import sys
from pathlib import Path
from types import SimpleNamespace

from transcription.engines.paraformer import ParaformerEngine, normalize_paraformer_result


def test_verified_character_alignment_and_punctuation():
    data = {"text": "今天，我们学习！",
            "raw_text": "今 天 我 们 学 习",
            "timestamp": [[0, 150], [150, 250], [280, 400], [400, 560],
                          [600, 750], [750, 900]]}
    text, words, chars, warnings = normalize_paraformer_result(data)
    assert text == "今天，我们学习！"
    assert [c.text for c in chars] == list("今天我们学习")
    assert [c.char_index for c in chars] == [0, 1, 3, 4, 5, 6]
    assert text[words[0].char_start:words[0].char_end] == words[0].text
    assert all(w.start is not None for w in words)
    assert not warnings


def test_no_timestamp_mapping_when_counts_disagree():
    text, words, chars, warnings = normalize_paraformer_result(
        {"text": "我们学习。", "raw_text": "我 们 学 习", "timestamp": [[0, 100]]}
    )
    assert text == "我们学习。"
    assert len(chars) == 4
    assert all(w.start is None for w in words)
    assert warnings


def test_word_level_units_not_pretended_to_be_characters():
    text, words, chars, warnings = normalize_paraformer_result(
        {"text": "人工智能。", "raw_text": "人工智能",
         "timestamp": [[100, 900]]}
    )
    assert all(w.start is None for w in words)
    assert warnings


def test_mocked_funasr_load_and_reuse(monkeypatch, tmp_path):
    record = {"init": 0, "generate": 0}
    class FakePipeline:
        def generate(self, **kw):
            record["generate"] += 1
            assert kw["return_raw_text"] is True
            assert kw["output_timestamp"] is True
            return [{"text": "你好。", "raw_text": "你 好",
                     "timestamp": [[100, 220], [220, 380]]}]
    def fake_auto_model(**kwargs):
        record["init"] += 1
        assert kwargs["model"] == "paraformer-zh"
        assert kwargs["vad_model"] == "fsmn-vad"
        assert kwargs["punc_model"] == "ct-punc"
        return FakePipeline()
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=fake_auto_model))
    engine = ParaformerEngine(model_id="paraformer-zh", name="paraformer_zh", device="cpu")
    result = engine.transcribe(tmp_path / "sample.wav")
    assert result.text == "你好。"
    assert len(result.characters) == 2
    assert [w.start for w in result.words] == [.1, .22]
    engine.transcribe(tmp_path / "sample.wav")
    assert record == {"init": 1, "generate": 2}
    engine.close()
