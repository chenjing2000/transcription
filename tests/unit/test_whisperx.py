"""Mocked tests: no network, model downloads, GPU or ASR installation required."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from transcription.engines.whisperx import WhisperXEngine, _word_entries
from transcription.errors import TranscriptionError


def test_word_offsets_and_missing_timestamps():
    text, words, warnings = _word_entries([
        {"text": "Hello hello!", "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "hello!", "start": None, "end": None},
        ]},
        {"text": "Again.", "words": [{"word": "Again.", "start": 1.2, "end": 1.5}]},
    ])
    assert text == "Hello hello! Again."
    assert [text[x.char_start:x.char_end] for x in words] == ["Hello", "hello!", "Again."]
    assert words[1].alignment_status == "missing"
    assert len(warnings) == 1


def test_english_model_lazily_loads_and_aligns(monkeypatch, tmp_path):
    calls = {"load": 0, "align": 0}
    class FakeASR:
        def transcribe(self, audio, batch_size):
            assert batch_size == 4
            return {"language": "en", "segments": [{"text": "Hello world."}]}
    def load_model(*args, **kwargs):
        calls["load"] += 1
        assert kwargs["language"] == "en"
        return FakeASR()
    def align(segments, model, metadata, audio, device, **kwargs):
        calls["align"] += 1
        return {"segments": [{
            "text": "Hello world.",
            "words": [{"word": "Hello", "start": .0, "end": .3},
                      {"word": "world.", "start": .4, "end": .8}],
        }]}
    fake = SimpleNamespace(load_model=load_model, load_audio=lambda p: "audio",
                           load_align_model=lambda **k: ("alignmodel", {"language": "en"}),
                           align=align)
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    engine = WhisperXEngine(model_id="large-v3", name="whisperx_large_v3", device="cpu")
    path = tmp_path / "a.wav"
    path.touch()
    result = engine.transcribe(path)
    assert result.status == "partial"
    assert result.text == "Hello world."
    assert len(result.words) == 2
    assert result.words[1].char_start == 6
    engine.transcribe(path)
    assert calls == {"load": 1, "align": 2}
    engine.close()


def test_recognition_error(monkeypatch, tmp_path):
    class BadASR:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("broken audio")
    fake = SimpleNamespace(load_model=lambda *a, **k: BadASR(), load_audio=lambda p: p)
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    engine = WhisperXEngine(model_id="large-v3", name="whisperx_large_v3", device="cpu")
    with pytest.raises(TranscriptionError) as err:
        engine.transcribe(tmp_path / "file.wav")
    assert err.value.code == "TRANSCRIPTION_FAILED"


def test_no_speech(monkeypatch, tmp_path):
    fake = SimpleNamespace(load_model=lambda *a, **k: SimpleNamespace(transcribe=lambda *a, **k: {"segments": []}),
                           load_audio=lambda p: p)
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    engine = WhisperXEngine(model_id="large-v3", name="whisperx_large_v3", device="cpu")
    res = engine.transcribe(tmp_path / "silence.wav")
    assert res.status == "partial"
    assert res.words == []
