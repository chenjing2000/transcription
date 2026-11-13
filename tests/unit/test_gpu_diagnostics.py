"""GPU prerequisite tests run on Windows CPU-only CI with injected fake torch."""
from types import SimpleNamespace

from transcription.gpu_diagnostics import cuda_compute_probe, cuda_snapshot, nvidia_smi_snapshot


def test_no_cuda_returns_explicit_failure():
    fake = SimpleNamespace(
        __version__="2.8.0+cpu",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    result = cuda_compute_probe(fake)
    assert result["cuda_available"] is False
    assert result["compute_ok"] is False
    assert "error" in result


def test_cuda_arithmetic_and_metadata():
    class FakeTensor:
        def __init__(self, nums): self.nums = nums
        def __mul__(self, other): return FakeTensor([x * y for x, y in zip(self.nums, other.nums)])
        def sum(self): return FakeTensor([sum(self.nums)])
        def item(self): return self.nums[0]
    fake = SimpleNamespace(
        __version__="2.12.0",
        version=SimpleNamespace(cuda="13.0"),
        arange=lambda count, **kwargs: FakeTensor(list(range(count))),
        float32="float32",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda i: SimpleNamespace(name="Mock Blackwell", total_memory=16000),
            get_device_capability=lambda i: (12, 0),
            memory_allocated=lambda i: 512,
            memory_reserved=lambda i: 1024,
            max_memory_allocated=lambda i: 2048,
            synchronize=lambda: None,
        ),
    )
    result = cuda_compute_probe(fake)
    assert result["compute_ok"] is True
    assert result["compute_capability"] == [12, 0]
    assert result["sum_of_squares"] == 1240.0
    assert result["reserved_bytes"] == 1024


def test_cuda_probe_does_not_fallback_when_compute_fails():
    fake = SimpleNamespace(
        __version__="2.12.0", version=SimpleNamespace(cuda="13.0"),
        float32="float32",
        arange=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("missing kernel for sm_120")),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda i: SimpleNamespace(name="Mock", total_memory=1),
            get_device_capability=lambda i: (12, 0),
            memory_allocated=lambda i: 0, memory_reserved=lambda i: 0,
            max_memory_allocated=lambda i: 0,
        ),
    )
    result = cuda_compute_probe(fake)
    assert result["compute_ok"] is False
    assert "sm_120" in result["error"]


def test_nvidia_smi_absence_is_not_crash(monkeypatch):
    import transcription.gpu_diagnostics as diag
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
    assert nvidia_smi_snapshot()["available"] is False
