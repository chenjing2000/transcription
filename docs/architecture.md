# Architecture and contracts

The library exposes `list_engines()`, `transcribe(file_path, engine, *, device="auto", audio_track=0)` and `release_models()`.

- **Registry:** trusted adapter modules publish `ENGINE_SPECS` (`EngineInfo`) and are discovered without loading weights; an adapter may declare multiple engine IDs.
- **Adapters:** each model family implements the same inference interface; model identifiers remain independent of module filenames. First planned engines are `whisperx_large_v3` (English) and `paraformer_zh` (Chinese).
- **Model manager:** a process-local single active engine keyed by `(engine_id, resolved_device)`. Cache reuse, deterministic release, GPU cache cleanup where appropriate. Concurrent inference is not supported by the initial design.
- **Media:** MP4/MKV -> FFprobe stream selection -> FFmpeg `-c:a copy` into codec-compatible audio container; MP3/WAV -> original source. Recognition decoding is separate from the saved stream copy. Preserve video/audio timeline mapping.
- **Processing:** normalize recognition units, preserve source text and provenance, derive sentences and paragraphs, English word timestamps or Chinese words plus reliable character timestamps.
- **Publishing:** four artifacts derived from one normalized transcript; output into sibling `transcript results`, use common non-colliding numeric suffix. Stage files and publish as a set; never overwrite.

## Status semantics

- `success`: transcription, alignment validation and publication complete.
- `partial`: usable text with alignment gaps or explicit nonfatal warnings; files may be published with quality flags.
- `failed`: no completed output set; result includes stable `error_code` and human-readable `info`.

Current Phase 1/2 scaffold: all actual inference calls return `NOT_IMPLEMENTED`; this is intentional.
