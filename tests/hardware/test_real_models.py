"""Explicit opt-in real-model acceptance. Never run model downloads by default.

Set TRANSCRIPTION_EN_SAMPLE and TRANSCRIPTION_ZH_SAMPLE to known short recordings
and TRANSCRIPTION_ENABLE_HARDWARE=1 to enable. Intended for a Windows CUDA host.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from transcription import release_models, transcribe


@pytest.mark.hardware
def test_real_english_chinese_english_sequence():
    if os.getenv("TRANSCRIPTION_ENABLE_HARDWARE") != "1":
        pytest.skip("Real-model hardware tests are opt-in")
    en = os.getenv("TRANSCRIPTION_EN_SAMPLE")
    zh = os.getenv("TRANSCRIPTION_ZH_SAMPLE")
    if not en or not zh or not Path(en).is_file() or not Path(zh).is_file():
        pytest.skip("Provide existing EN/ZH sample recordings")
    device = os.getenv("TRANSCRIPTION_HARDWARE_DEVICE", "gpu")
    try:
        for path, engine in [
            (en, "whisperx_large_v3"), (zh, "paraformer_zh"),
            (en, "whisperx_large_v3"),
        ]:
            result = transcribe(path, engine=engine, device=device)
            assert result.status in {"success", "partial"}, (
                engine, result.error_code, result.info
            )
            assert result.text.strip(), engine
            assert result.files.keys() >= {"timed_json", "timed_txt", "json", "txt"}
            assert all(p.is_file() for p in result.files.values())
            timed = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
            plain = json.loads(result.files["json"].read_text(encoding="utf-8"))
            assert timed["text"] == plain["text"] == result.text
            for word in result.words:
                assert result.text[word.char_start:word.char_end] == word.text
    finally:
        release_models()
