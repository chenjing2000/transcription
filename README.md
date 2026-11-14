# Transcription

Python audio/video transcription library with two lazily loaded speech engines:
English WhisperX (`whisperx_large_v3`) and Mandarin FunASR Paraformer (`paraformer_zh`).

## Public API

```python
from transcription import list_engines, transcribe, release_models

print([item.name for item in list_engines()])
try:
    result = transcribe(
        r"D:\Videos\Lecture.mp4", engine="whisperx_large_v3",
        device="auto", audio_track=0,
    )
    print(result.status, result.info, result.error_code, result.files)
finally:
    release_models()
```

The public `transcribe()` API takes MP4, MKV, MP3 or WAV files and returns
`success`, `partial` (usable transcript with warnings), or `failed` with
a machine-readable error code. `device` is `auto`, `cpu` or `gpu`;
`audio_track` is zero-based among audio tracks.

## Development

Use Python 3.11 or 3.12, FFmpeg and FFprobe on PATH. From the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python main.py
```

Heavy model dependencies and weights are intentionally excluded from the base
installation. To run real speech models, install a compatible PyTorch,
[WhisperX](https://github.com/m-bain/whisperX) and/or
[FunASR](https://github.com/modelscope/FunASR) stack in a dedicated environment.

## Output

After successful recognition, `transcript results` is created beside the
input file. Each run publishes `.timed.json` and plain `.txt`: English JSON retains
word timestamps; Mandarin JSON contains sentence timestamps only. For videos,
all unverified original-video playback times are `null` (status `partial`)
until full video mapping is validated. Video additionally publishes its extracted
original-codec audio (no re-encoding). A shared numeric suffix ` (1)`, ` (2)`, ... prevents
overwriting previous output groups. Times that cannot be verified are
explicitly missing rather than synthesized.

## Validation status

- Windows Python 3.11/3.12 unit and FFmpeg integration checks run on pushes.
- Real short **English tiny.en** and **Mandarin Paraformer** speech inference
  succeeded on CPU CI, including transcript publishing and timestamp coverage
  for those samples.
- English **large-v3 GPU inference**, bilingual GPU model switching, physical
  VRAM release, and complex video time-base synchronization remain unverified.
- Publishing rolls back ordinary failures and avoids overwriting other files.
  It does **not** promise crash-atomic multi-file transactions.

For CPU/GPU online/local uv environment variants, see the\n[uv environment guide](docs/uv_installation_variants.md).\n\nSee [roadmap](docs/roadmap.md), [validation boundaries](docs/validation.md),
[output format](docs/output_schema.md), [architecture](docs/architecture.md),
and [GPU acceptance](docs/hardware_acceptance.md).
