"""CUDA preflight and memory snapshots without loading speech model weights."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def cuda_snapshot(torch_module=None) -> dict:
    """Report one process's allocator statistics; never claim to measure all VRAM."""
    torch = torch_module if torch_module is not None else _torch()
    if torch is None:
        return {"torch_installed": False, "cuda_available": False, "error": "PyTorch not installed"}
    info = {
        "torch_installed": True,
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not info["cuda_available"]:
        return info
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update({
            "device_index": device,
            "device_name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(props.total_memory),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def cuda_compute_probe(torch_module=None) -> dict:
    """Perform a tiny actual CUDA operation, with no CPU fallback on failure."""
    torch = torch_module if torch_module is not None else _torch()
    snapshot = cuda_snapshot(torch)
    if not snapshot.get("cuda_available"):
        snapshot["compute_ok"] = False
        snapshot["error"] = snapshot.get("error", "CUDA is unavailable")
        return snapshot
    try:
        a = torch.arange(16, dtype=torch.float32, device="cuda")
        b = (a * a).sum()
        torch.cuda.synchronize()
        observed = float(b.item())
        if observed != 1240.0:
            raise RuntimeError(f"Unexpected CUDA compute result: {observed}")
        snapshot["compute_ok"] = True
        snapshot["sum_of_squares"] = observed
    except Exception as exc:
        snapshot["compute_ok"] = False
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Keep tensors local and avoid retaining probe allocations.
        if "a" in locals():
            del a
        if "b" in locals():
            del b
    return snapshot


def nvidia_smi_snapshot() -> dict:
    """Optional driver-level reading, not a requirement for running ASR."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if completed.returncode:
        return {"available": False, "error": completed.stderr.strip()[-500:]}
    return {"available": True, "raw": completed.stdout.strip()}


def report() -> dict:
    return {"cuda": cuda_compute_probe(), "nvidia_smi": nvidia_smi_snapshot()}


def main() -> int:
    payload = report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if len(sys.argv) > 1:
        dest = Path(sys.argv[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if payload["cuda"].get("compute_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
