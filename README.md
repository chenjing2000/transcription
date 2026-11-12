# Transcription

A Python-first, extensible Chinese/English audio/video transcription project.

**Current status: Phase 1 + Phase 2 foundation only. Real audio/video transcription, FFmpeg extraction, time alignment, JSON/TXT exports and model installation are NOT implemented yet.** The two engines are registered as placeholders and deliberately return `NOT_IMPLEMENTED` when invoked.

## Public API

```python
from transcription import list_engines, transcribe, release_models

for item in list_engines():
    print(item.name, item.language, item.model_id)

result = transcribe("lecture.mp4", engine="whisperx_large_v3", device="auto")
print(result.status, result.error_code, result.info)
release_models()
```

Running `python main.py` lists registered engines and shows a safe placeholder example. It does **not** run actual ASR.

## Install and test

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python main.py
```

Python 3.11/3.12 are proposed for future ASR compatibility checks; neither CUDA nor ASR frameworks are installed by this foundational package.

## Agreed behavior for future phases

- Inputs: MP4, MKV, MP3, WAV.
- Engines: WhisperX large-v3 (English) and FunASR Paraformer-zh (Chinese), expandable registry.
- `device`: `gpu`, `cpu`, `auto`. A GPU requested but unavailable must return an error; no silent CPU fallback after GPU inference failure.
- Stream-copy first video audio track, preserving encoded stream parameters, and use a compatible output container. No transcoding of the saved extracted audio.
- Always produce four files per successful run: `.timed.json`, `.timed.txt`, `.json`, `.txt` under the source directory's `transcript results` folder. Video inputs additionally save the extracted original-codec audio.
- If any target exists, choose a common suffix `(1)`, `(2)`, ... for the entire output set; never overwrite existing files.
- Preserve original transcript content and offsets; partial alignments must be reported.
- Only one active cached engine; release it before switching engine/device and support `release_models()`.

See `docs/architecture.md`, `docs/roadmap.md`, and `docs/output_schema.md`.
