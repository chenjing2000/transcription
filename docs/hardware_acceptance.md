# Optional real-model GPU acceptance

Normal GitHub Windows runners are CPU-only. The separate real English/Mandarin
CPU tests have exercised recognition and timestamp exports, but **not** the
WhisperX large-v3 GPU model or the GPU model-switching lifecycle.

## Stage 1 — prerequisites without downloading models

Create a fresh Python 3.11 environment; do not modify an existing research
environment. Install a CUDA-enabled PyTorch build compatible with the NVIDIA
driver and RTX 5060 Ti's sm_120 architecture. Prefer the current official
PyTorch Windows install selector: https://pytorch.org/get-started/locally/

Install the project, WhisperX, FunASR and FFmpeg/FFprobe, and run:

```powershell
python -m transcription.doctor
python -m transcription.gpu_diagnostics gpu-preflight.json
```

The second command must exit 0 and report `compute_ok: true`. If it fails,
do not download huge ASR models until driver/CUDA compatibility is resolved.

## Stage 2 — one bilingual hardware acceptance

Only when Stage 1 passes and two short audio files are available, use:

```powershell
$env:TRANSCRIPTION_ENABLE_HARDWARE = "1"
$env:TRANSCRIPTION_EN_SAMPLE = "D:\\Samples\\english.wav"
$env:TRANSCRIPTION_ZH_SAMPLE = "D:\\Samples\\mandarin.wav"
$env:TRANSCRIPTION_HARDWARE_DEVICE = "gpu"
$env:TRANSCRIPTION_GPU_REPORT = "gpu-acceptance-report.json"
python -m pytest tests/hardware/test_real_models.py -m hardware -v
```

This tests **English large-v3 -> Chinese Paraformer -> English large-v3**
and writes a JSON report even if an assertion or model loading fails.
It records GPU model, PyTorch CUDA build, peak/allocated/reserved memory,
driver-level readings when nvidia-smi is available, result status,
timed-word coverage and transcript warnings.

These are observations rather than strict VRAM thresholds: PyTorch cached
allocator memory and native CTranslate2/FunASR memory are not necessarily
visible in the same counters. A device-level `nvidia-smi` comparison is also
included.

This test publishes output files under the source's `transcript results`,
using numbered groups on reruns, and does not overwrite previous outputs.
Inspect the generated report and review transcription/timestamp quality
separately. No exact transcription accuracy guarantee follows from a clean run.

Do not use these commands until CUDA preflight succeeds, and do not use
`--force-reinstall` blindly because ML frameworks may resolve to incompatible
PyTorch builds.

## One-command Windows shortcut

```powershell
# Preflight only. No speech model weights loaded:
.\\scripts\\run-gpu-acceptance.ps1

# Later, when both test recordings are ready:
.\\scripts\\run-gpu-acceptance.ps1 -EnglishSample 'D:\\Samples\\english.wav' -ChineseSample 'D:\\Samples\\mandarin.wav'
```

The script stops on a failed CUDA compute probe and saves diagnostic JSON.
It neither installs PyTorch nor updates the Windows driver.
