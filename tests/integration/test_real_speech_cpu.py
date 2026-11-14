"""Opt-in public-speech real-model acceptance on Windows CPU.

Official sources:
  English: https://github.com/openai/whisper/blob/main/tests/jfk.flac
  Mandarin: FunASR sample asr_example_zh.wav

These tests download checkpoints and run real inference ONLY with the
TRANSCRIPTION_ENABLE_REAL_SPEECH=1 flag. There are no GPU assumptions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from transcription import release_models, transcribe
from transcription.engines.whisperx import WhisperXEngine
from transcription.media import prepare_audio
from transcription.output import publish_result


def _fixture(name: str) -> Path:
    if os.getenv("TRANSCRIPTION_ENABLE_REAL_SPEECH") != "1":
        pytest.skip("Real speech checkpoint downloads are opt-in")
    parent = os.getenv("TRANSCRIPTION_SPEECH_DIR")
    if not parent:
        pytest.skip("TRANSCRIPTION_SPEECH_DIR not set")
    source = Path(parent) / name
    assert source.is_file() and source.stat().st_size > 1000, source
    return source


def _record(name: str, payload: dict) -> None:
    folder = Path(os.environ["TRANSCRIPTION_SPEECH_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + "-report.json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _validate_outputs(result):
    assert result.status in {"success", "partial"}, (
        result.error_code, result.info
    )
    assert result.text.strip(), "Empty recognition"
    assert set(result.files) == {"timed_json", "txt"}
    assert all(path.is_file() and path.stat().st_size for path in result.files.values())
    timed = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
    assert len(timed["word_segments"]) == len(result.words)
    assert timed["language"] in {"en", "zh"}
    for word in result.words:
        assert result.text[word.char_start:word.char_end] == word.text
        if word.start is not None:
            assert 0 <= word.start <= word.end
    return timed


@pytest.mark.real_speech
def test_real_english_speech_recognition_and_word_alignment():
    source = _fixture("jfk.wav")
    # tiny.en isolates the real WhisperX decoding + English alignment path;
    # it is NOT a large-v3 accuracy/compatibility acceptance.
    engine = WhisperXEngine(
        name="whisperx_tiny_real_speech", model_id="tiny.en",
        device="cpu", batch_size=1,
    )
    try:
        result = engine.transcribe(source)
        result.source_path = source
        result.device = "cpu"
        prepared = prepare_audio(source, source.parent / "unused", audio_track=0)
        result = publish_result(result, prepared)
        _validate_outputs(result)
        _record("english", {
            "status": result.status,
            "text": result.text,
            "word_count": len(result.words),
            "timed_word_count": sum(w.start is not None for w in result.words),
            "warnings": result.warnings,
        })
        lower = result.text.lower()
        assert "american" in lower or "country" in lower, result.text
        assert len(result.words) >= 6, result.text
        assert sum(w.start is not None for w in result.words) >= 4, result.warnings
    finally:
        engine.close()


@pytest.mark.real_speech
def test_real_chinese_speech_recognition_and_timestamp_reporting():
    source = _fixture("asr_example_zh.wav")
    try:
        result = transcribe(source, engine="paraformer_zh", device="cpu")
        _record("chinese", {
            "status": result.status,
            "error_code": result.error_code,
            "info": result.info,
            "text": result.text,
            "word_count": len(result.words),
            "timed_word_count": sum(w.start is not None for w in result.words),
            "timed_character_count": sum(c.start is not None for c in (result.characters or [])),
            "warnings": result.warnings,
        })
        _validate_outputs(result)
        assert sum("\u3400" <= c <= "\u9fff" for c in result.text) >= 2
        assert len(result.words) > 0, "No word entries from Chinese recognition"
        # Missing timestamps are reported rather than fabricated. We record
        # precise coverage for later tuning, but do not claim exact alignment
        # until actual token units are proven to map one-to-one.
        assert all(
            word.alignment_status == "missing" if word.start is None else True
            for word in result.words
        )
    finally:
        release_models()
