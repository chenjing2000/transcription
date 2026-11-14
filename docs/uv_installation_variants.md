# uv installation variants: CPU/GPU and online/local wheels

This guide is for a **separate Windows x64, Python 3.12 inference workspace**,
such as `E:\pywork\whisperx`. It is **not** a replacement for this repository's
lightweight development `pyproject.toml`: normal Windows CI intentionally does
not download real ASR engines.

## 1. What changes?

| Target | PyTorch source | Torch wheel location | GPU requirements |
| --- | --- | --- | --- |
| NVIDIA GPU + online | `pytorch-cu128` | Official CUDA 12.8 index | Suitable NVIDIA driver and CUDA-compatible wheel |
| CPU only + online | `pytorch-cpu` | Official CPU index | No NVIDIA driver / CUDA installation |
| NVIDIA GPU + local | `path = "wheels/...+cu128-...whl"` | Local CUDA wheels | As in GPU online |
| CPU only + local | `path = "wheels/...+cpu-...whl"` | Local CPU wheels | None |

Use **exactly one** source mode for each of `torch`, `torchaudio`,
`torchvision` in `[tool.uv.sources]`. The versions in `[project].dependencies`
do not have to change when switching between the 2.8.0 CPU and cu128 builds.
Ordinary Python dependencies (`whisperx`, `funasr`, `numpy`,
`transformers`, `pytest`) are identical for these four cases.
`torchvision` is retained here for parity with the example workspace; the
transcription library itself does not import it directly.

**Version scope:** These examples target Python 3.12 on Windows x64 with
`torch==2.8.0`, `torchaudio==2.8.0`, `torchvision==0.23.0`.
Do not apply the `cp312-win_amd64` wheels to Python 3.11, Linux or macOS.
Model and indirect dependency compatibility still needs `uv lock` and
real inference validation.

## 2. Complete CPU + online pyproject.toml

Copy this file into the **dedicated inference workspace**, not over the
`transcription-toolkit` repository's packaging metadata:

```toml
[project]
name = "whisperx-workspace"
version = "0.1.0"
description = "CPU English and Mandarin transcription environment"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "torchvision==0.23.0",
    "whisperx==3.8.6",
    "funasr==1.4.26",
    "numpy>=2.1,<3",
    "transformers>=4.48,<5",
]

[dependency-groups]
dev = ["pytest>=8,<10"]

[tool.uv]
package = false

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "hardware: real model and GPU hardware acceptance",
    "real_model: opt-in real ASR model smoke",
    "real_speech: opt-in real English/Mandarin speech acceptance",
]
```

The PyTorch source pin prevents `uv add numpy` or `uv sync` from silently
re-resolving `torch` to a PyPI CPU build on a CUDA workspace (or vice versa).
The named index is explicit so unrelated dependencies still use the ordinary
package index. Source selection is uv-specific; regular pip does not apply it.

## 3. GPU + online: replace only the source configuration

Keep `[project]`, `[dependency-groups]` and pytest settings from section 2,
but **replace** the CPU `[tool.uv.sources]` and `[[tool.uv.index]]`
sections with:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

RTX 5060 Ti requires a Blackwell-compatible build. CUDA 12.8 wheel selection
does not imply the local CUDA Toolkit must be installed system-wide. Actual
ASR/native runtime dependencies may need additional DLLs; see
[`gpu_preflight.md`](gpu_preflight.md).

## 4. Local wheel installation (CPU or GPU)

Download **all three** wheel files corresponding to the selected variant and
Python/Windows tag. Example names for Python 3.12 Windows x64:

- GPU: `torch-2.8.0+cu128-cp312-cp312-win_amd64.whl`,
  `torchaudio-2.8.0+cu128-cp312-cp312-win_amd64.whl`,
  `torchvision-0.23.0+cu128-cp312-cp312-win_amd64.whl`
- CPU: `torch-2.8.0+cpu-cp312-cp312-win_amd64.whl`,
  `torchaudio-2.8.0+cpu-cp312-cp312-win_amd64.whl`,
  `torchvision-0.23.0+cpu-cp312-cp312-win_amd64.whl`

Download only from the matching **official PyTorch** index:
https://download.pytorch.org/whl/cu128 or
https://download.pytorch.org/whl/cpu. Check that each file actually exists
for your interpreter/platform before trying to lock. The wheel filenames
above are examples of the expected tags, not portable cross-platform names.

Place downloaded files in a project-local `wheels/` directory, then **replace**
the online `[tool.uv.sources]` with one of the following.

GPU + local:

```toml
[tool.uv.sources]
torch = { path = "wheels/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl" }
torchaudio = { path = "wheels/torchaudio-2.8.0+cu128-cp312-cp312-win_amd64.whl" }
torchvision = { path = "wheels/torchvision-0.23.0+cu128-cp312-cp312-win_amd64.whl" }
```

CPU + local:

```toml
[tool.uv.sources]
torch = { path = "wheels/torch-2.8.0+cpu-cp312-cp312-win_amd64.whl" }
torchaudio = { path = "wheels/torchaudio-2.8.0+cpu-cp312-cp312-win_amd64.whl" }
torchvision = { path = "wheels/torchvision-0.23.0+cpu-cp312-cp312-win_amd64.whl" }
```

For local wheel mode, you may remove the no-longer-used
`[[tool.uv.index]]` block. A local-wheel source means the project depends
on those files remaining at their relative paths; it does **not** guarantee
offline installation of the other Python dependencies or ASR model weights.
Use the normal online source again before moving the project to a new platform
or publishing it without bundled wheels.

For any variant, generate an environment-specific `uv.lock`. Do not carry a
lockfile from GPU mode into CPU mode (or vice versa) without re-locking.
Switching source configuration changes the dependency lock and installed
PyTorch binary; ideally keep separate workspace folders/lockfiles for each
machine variant.

## 5. Installation, verification, and FFmpeg

From the matching workspace root:

```powershell
uv lock
uv sync
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run --no-sync python -c "import whisperx; from funasr import AutoModel; print('ASR imports OK')"
uv pip check
```

CPU mode should print `torch.version.cuda == None` and
`torch.cuda.is_available() == False`; GPU cu128 mode should report a CUDA
build and, on a working NVIDIA host, `True`. For actual CUDA execution:

```powershell
uv run --no-sync python -c "import torch; print(torch.arange(10, device='cuda').sum().item())"
```

Expected: `45` (GPU mode only). A successful import is not full ASR acceptance.

**FFmpeg and FFprobe are external executables**, not ordinary Python package
dependencies. Install them separately and ensure both commands are on
`PATH`:

```powershell
ffmpeg -version
ffprobe -version
```

The library probes all supported inputs with FFprobe and uses FFmpeg to
stream-copy video audio. The official Windows distribution is available via
https://ffmpeg.org/download.html.

When running the repository's transcription API on a CPU-only machine,
pass `device="cpu"`; for GPU use `device="gpu"` and follow
[`hardware_acceptance.md`](hardware_acceptance.md).

## Sources

- uv index pins: https://docs.astral.sh/uv/configuration/indexes/
- uv PyTorch integration: https://docs.astral.sh/uv/guides/integration/pytorch/
- uv local sources: https://docs.astral.sh/uv/concepts/projects/dependencies/
- PyTorch wheel indexes: https://download.pytorch.org/whl/cpu and
  https://download.pytorch.org/whl/cu128
