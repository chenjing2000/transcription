"""Opt-in bilingual GPU acceptance with structured diagnostics.

No model weights are fetched by normal Windows CI. Invoke explicitly with
TRANSCRIPTION_ENABLE_HARDWARE=1 and two real, known speech recordings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from transcription import release_models, transcribe
from transcription.gpu_diagnostics import cuda_compute_probe, cuda_snapshot, nvidia_smi_snapshot


def _record(step, result, memory_before, memory_after):
    return {
        "step": step,
        "status": result.status,
        "error_code": result.error_code,
        "info": result.info,
        "text": result.text,
        "word_count": len(result.words),
        "timed_words": sum(w.start is not None for w in result.words),
        "warnings": result.warnings,
        "cuda_before": memory_before,
        "cuda_after": memory_after,
    }


@pytest.mark.hardware
def test_real_english_chinese_english_sequence():
    if os.getenv("TRANSCRIPTION_ENABLE_HARDWARE") != "1":
        pytest.skip("Real-model hardware tests are opt-in")
    en = os.getenv("TRANSCRIPTION_EN_SAMPLE")
    zh = os.getenv("TRANSCRIPTION_ZH_SAMPLE")
    assert en and zh and Path(en).is_file() and Path(zh).is_file(), (
        "Provide TRANSCRIPTION_EN_SAMPLE and TRANSCRIPTION_ZH_SAMPLE pointing to existing audio"
    )
    assert os.getenv("TRANSCRIPTION_HARDWARE_DEVICE", "gpu") == "gpu", (
        "This GPU acceptance test must run explicitly on GPU, never silently on CPU"
    )
    report = {"preflight": cuda_compute_probe(), "steps": [], "final_cuda": None,
              "nvidia_smi_before": nvidia_smi_snapshot(), "nvidia_smi_after": None}
    path = Path(os.getenv("TRANSCRIPTION_GPU_REPORT", "transcription-gpu-report.json"))
    try:
        assert report["preflight"]["compute_ok"], report["preflight"]
        for index, (sample, engine) in enumerate([
            (en, "whisperx_large_v3"), (zh, "paraformer_zh"),
            (en, "whisperx_large_v3"),
        ], 1):
            before = cuda_snapshot()
            result = transcribe(sample, engine=engine, device="gpu")
            after = cuda_snapshot()
            report["steps"].append(_record(index, result, before, after))
            assert result.status in {"success", "partial"}, (
                engine, result.error_code, result.info
            )
            assert result.text.strip(), engine
            assert result.device == "gpu"
            assert result.files.keys() >= {"timed_json", "txt"}
            assert all(p.is_file() for p in result.files.values())
            timed = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
            assert len(timed["word_segments"]) == len(result.words)
            for word in result.words:
                assert result.text[word.char_start:word.char_end] == word.text
    finally:
        release_models()
        report["final_cuda"] = cuda_snapshot()
        report["nvidia_smi_after"] = nvidia_smi_snapshot()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
