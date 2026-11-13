# Architecture and contracts

The library exposes `list_engines()`, `transcribe(file_path, engine, *, device="auto", audio_track=0)` and `release_models()`.

- **Registry:** trusted adapter modules publish `ENGINE_SPECS` (`EngineInfo`) and the lightweight `ENGINE_CLASS` class name; the registry resolves both metadata and factory without loading weights. An adapter may declare multiple engine IDs.
- **Adapters:** each model family implements the same inference interface; model identifiers remain independent of module filenames. First planned engines are `whisperx_large_v3` (English) and `paraformer_zh` (Chinese).
- **Model manager:** a process-local single active engine keyed by `(engine_id, resolved_device)`. Inference uses the sole lock-protected `run()` entry, which prevents switching/release during inference. Cache reuse, deterministic release, GPU cache cleanup where appropriate. Calls are serialized rather than executed concurrently.
- **Media:** MP4/MKV -> FFprobe stream selection -> FFmpeg `-c:a copy` into codec-compatible audio container; MP3/WAV -> original source. Recognition decoding is separate from the saved stream copy. Preserve video/audio timeline mapping.
- **Processing:** normalize recognition units, preserve source text and provenance, derive sentences and paragraphs, English word timestamps or Chinese words plus reliable character timestamps.
- **Publishing:** four artifacts derived from one normalized transcript; output into sibling `transcript results`, use common non-colliding numeric suffix. Stage files and publish as a set; never overwrite. Internally, `_stage_files` prepares and verifies payloads and `_publish_files` assigns a collision-free shared suffix and publishes under a directory lock.

## Status semantics

- `success`: transcription, alignment validation and publication complete.
- `partial`: usable text with alignment gaps or explicit nonfatal warnings; files may be published with quality flags.
- `failed`: no completed output set; result includes stable `error_code` and human-readable `info`.

Both English WhisperX and Chinese Paraformer CPU recognition have passed short real-speech CI tests. The CUDA large-v3 bilingual model-switching acceptance is pending.
