# GPU preflight — Windows RTX 50 series

RTX 5060 Ti is a Blackwell (compute capability 12.0) GPU. A working NVIDIA
driver and a CUDA-enabled PyTorch **wheel with sm_120 support** are required.
The locally installed CUDA Toolkit is not necessarily required to run a
PyTorch wheel. Check the current official install selector rather than assuming
that the CPU-only PyTorch wheel installed by CI provides CUDA.

Official sources:
- https://developer.nvidia.com/cuda/gpus
- https://pytorch.org/get-started/locally/
- https://github.com/pytorch/pytorch/releases

## Safe checks before any ASR model download

In a clean Python 3.11 virtual environment, install a compatible CUDA PyTorch
wheel from the official PyTorch index. For recent stable versions, CUDA 13.0
wheels are available; check compatibility with your currently installed driver
using the official selector before choosing CUDA 13.0 or an older build.

Then run (no model weights downloaded):

```powershell
python -m transcription.doctor
python -m transcription.gpu_diagnostics gpu-preflight.json
```

The second command exits 0 **only** if actual CUDA tensor arithmetic succeeds.
It reports the PyTorch build, detected GPU, compute capability, and
per-process allocator memory. If no CUDA GPU is available or a CUDA kernel
fails, it exits nonzero; no silent CPU fallback occurs.

Note: `torch.cuda.memory_reserved` measures PyTorch's allocator, not the
entire process or physical VRAM. `nvidia-smi` is recorded separately when
available, and can differ due to other CUDA libraries and processes.

## After preflight succeeds

Use two short EN/ZH samples and the opt-in test in `docs/hardware_acceptance.md`
for English -> Mandarin -> English switching. Only after these pass consider
long-form inference and stability measurement. Do not install/change global
CUDA runtimes or overwrite an existing research environment merely for this
project; use a dedicated virtual environment.
