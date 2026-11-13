# Transcription

Python audio/video speech transcription framework with extensible engine adapters.

## Current stage

Phase 6: document export pipeline and media stream copy implemented. **Real WhisperX/FunASR model execution, their actual dependency versions, CUDA compatibility and video timebase alignment have not yet been validated on hardware.** Windows CI runs FFmpeg and mocked ASR adapters; it does not download large speech models.

## Public API

```python
from transcription import list_engines, transcribe, release_models

print([engine.name for engine in list_engines()])
result = transcribe(r"D:\Videos\Lecture.mp4", engine="whisperx_large_v3", device="auto")
print(result.status, result.info, result.error_code, result.files)
release_models()
```

Supported inputs: MP4, MKV, MP3, WAV. Engines: `whisperx_large_v3` (English) and `paraformer_zh` (Chinese), with a registry designed for more models. `device` accepts `gpu`, `cpu`, `auto`; `audio_track` is a zero-based audio stream selector.

## Development environment

Python 3.11/3.12, FFmpeg and FFprobe on PATH. For Windows local development:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python main.py
```

Real recognition needs separately installed compatible WhisperX or FunASR, PyTorch, and model weights; see their respective upstream installation guidance: https://github.com/m-bain/whisperX and https://github.com/modelscope/FunASR. These heavy dependencies are intentionally not required by CI.

## Output behavior

After actual recognition, files are saved under the source directory's `transcript results` folder. Every run creates `.timed.json`, `.timed.txt`, `.json` and `.txt`. Video additionally saves extracted audio using the **original encoded packets** in a compatible audio container; no re-encoding. Existing files are never overwritten: the entire batch uses a common ` (1)`, ` (2)`, ... suffix.

Returns `success` for completed output without warnings, `partial` for valid output with alignment/timeline warnings, and `failed` for errors. Incomplete timing is explicit rather than guessed.

See `docs/architecture.md`, `docs/output_schema.md`, `docs/media.md`, `docs/whisperx.md`, `docs/paraformer.md`, `docs/roadmap.md`.
