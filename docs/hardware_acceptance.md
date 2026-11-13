# Optional real-model GPU acceptance

This project does not ask end users to test every development milestone. Real
model testing is deferred until both dependencies are installable.

A future single local acceptance run with 2 short files verifies model loading,
output publication and the switching order English → Chinese → English.

## Preconditions

- Windows workstation with supported CUDA-enabled PyTorch and correct GPU driver
- Compatible WhisperX and FunASR installations
- FFmpeg and FFprobe on PATH
- Two short known English/Mandarin recordings, ideally 10–30 seconds
- Enough free SSD for model caches; downloads can be large

PowerShell example, only when ready to conduct the real GPU acceptance:

```powershell
$env:TRANSCRIPTION_ENABLE_HARDWARE = "1"
$env:TRANSCRIPTION_EN_SAMPLE = "D:\\Samples\\english.wav"
$env:TRANSCRIPTION_ZH_SAMPLE = "D:\\Samples\\mandarin.wav"
$env:TRANSCRIPTION_HARDWARE_DEVICE = "gpu"
python -m pytest tests/hardware/test_real_models.py -m hardware -v
```

Test generates output under each sample's `transcript results` and never
overwrites pre-existing file groups. After the test, visually inspect
transcript quality and alignment using the sample audio; unit tests alone do
not establish acoustic accuracy.

A successful import-only GitHub job is insufficient for this GPU test. The
hardware test is intentionally skipped by normal CI unless explicitly enabled.
