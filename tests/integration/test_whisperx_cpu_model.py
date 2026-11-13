"""Opt-in real CPU model-loading smoke test.

This test downloads the public tiny.en model and runs real inference on a
synthetic WAV. It is not a quality or large-v3 acceptance test.
"""
from __future__ import annotations

import math
import os
import struct
import wave

import pytest

from transcription.engines.whisperx import WhisperXEngine


@pytest.mark.real_model
def test_tiny_english_cpu_inference(tmp_path):
    if os.getenv("TRANSCRIPTION_ENABLE_CPU_MODEL") != "1":
        pytest.skip("Requires explicit opt-in to download a speech model")
    path = tmp_path / "synthetic.wav"
    rate = 16000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        samples = (
            int(300 * math.sin(2 * math.pi * 440 * i / rate))
            for i in range(2 * rate)
        )
        handle.writeframes(b"".join(struct.pack("<h", v) for v in samples))
    engine = WhisperXEngine(
        model_id="tiny.en", name="whisperx_tiny_smoke", device="cpu", batch_size=1,
    )
    try:
        assert engine.asr is not None
        result = engine.transcribe(path)
        assert result.status == "partial"
        assert isinstance(result.text, str)
        assert isinstance(result.warnings, list)
    finally:
        engine.close()
