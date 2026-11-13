"""Reporting behavior is testable even without an NVIDIA GPU."""
import json

import transcription.gpu_diagnostics as gpu


def test_cli_without_cuda_returns_nonzero_and_writes_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "report", lambda: {
        "cuda": {"cuda_available": False, "compute_ok": False, "error": "no GPU"},
        "nvidia_smi": {"available": False},
    })
    monkeypatch.setattr(gpu.sys, "argv", ["gpu_diagnostics", str(tmp_path / "report.json")])
    assert gpu.main() == 2
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf8"))
    assert data["cuda"]["error"] == "no GPU"


def test_cli_with_success_writes_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "report", lambda: {
        "cuda": {"cuda_available": True, "compute_ok": True, "compute_capability": [12, 0]},
        "nvidia_smi": {"available": True},
    })
    monkeypatch.setattr(gpu.sys, "argv", ["gpu_diagnostics", str(tmp_path / "ok.json")])
    assert gpu.main() == 0
    data = json.loads((tmp_path / "ok.json").read_text(encoding="utf8"))
    assert data["cuda"]["compute_capability"] == [12, 0]
