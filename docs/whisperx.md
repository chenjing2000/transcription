# WhisperX adapter — Phase 4

- `engine="whisperx_large_v3"` selects the English `large-v3` model. The adapter loads `whisperx` only on first inference; listing engines needs no ASR installation.
- CPU selects `int8`, GPU selects `float16`, default batch size is 4. Device `auto` prefers CUDA; device `gpu` fails clearly without CUDA. Model loading and alignment models are cached by the single-active model manager.
- The adapter uses `load_model`, `load_audio`, `model.transcribe`, `load_align_model`, `align` per WhisperX's Python interface.
- Word entries have word text, seconds, a half-open text character span, and explicit alignment status. Missing timestamps remain missing instead of being estimated.
- Real WhisperX recognition returns in-memory `TranscriptionResult(status="partial")` until Phase 6 publishes the required four files. Extracted audio from videos is temporary, not returned as an existing output. Video timeline transforms still need validation.
- Windows CI exercises mock adapters and media integration without downloading large language models. CPU/GPU hardware inference needs subsequent functional verification.

Reference: https://github.com/m-bain/whisperX
