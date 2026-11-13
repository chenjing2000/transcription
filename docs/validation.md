# Phase 7 validation boundaries

## Automated Windows CPU-only checks

- Python 3.11 and 3.12 `pytest`: registry, data contracts, audio stream-copy FFmpeg integration, mock WhisperX/FunASR invocation, four-file output, collision numbering, rollback and character/time validation.
- The separate **Optional ASR dependency import smoke** workflow installs one optional ASR dependency stack per job on Windows Python 3.11 and imports its API. It intentionally does not download any checkpoints or perform real transcription. See `.github/workflows/asr-dependency-smoke.yml`.

## Environment inspection

Run `python -m transcription.doctor` to print installed optional packages and FFmpeg/FFprobe versions without loading model weights. Missing package is not necessarily broken; actual ASR requires explicit optional installation.

## Not yet proven

1. Real `whisperx_large_v3` English audio transcription and word alignment (small known-audio sample).
2. Real `paraformer_zh` Chinese audio transcription with VAD/punctuation and timestamp output.
3. CUDA PyTorch and ASR model compatibility with RTX 5060 Ti; memory usage after `release_models()`.
4. Source-video clock mapping for edit lists, discontinuities, variable timebases or nonzero start offsets.
5. Recovery after sudden process termination or power loss mid multi-file publish.

A successful dependency import is **not** equivalent to passing any of these real-model or GPU checks.

## Future minimal GPU acceptance

Use one short known English recording and one short Mandarin recording. Verify returned `status`, `files`, `words`, missing alignments, timing in a media player, model switching order English -> Chinese -> English, and VRAM before/after release. Do not claim accuracy thresholds until there are reference transcripts.
