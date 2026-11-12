# Implementation milestones

1. **Foundation:** project scaffold, result dataclasses, public API, validation, error codes. [Implemented]
2. **Registry/cache framework:** metadata-driven engine discovery, single-active-engine manager, release API, unit tests. [Implemented for mock engines; no framework-specific loading]
3. **Media:** FFprobe validation, stream-copy extraction, container selection and basic time offset metadata. [Implemented as an independent module; complex video timeline synchronization remains future validation]
4. **WhisperX:** large-v3 English ASR and forced word alignment integrated in-memory with mock tests. [Implemented; real CPU/GPU, full video timeline and JSON/TXT exports pending]
5. **FunASR:** Paraformer-zh, FSMN VAD, punctuation, word/character alignment and hardware tests. [Pending]
6. **Export:** single transcript authority, sentence/paragraph segmentation, timed/plain JSON/TXT, non-overwriting group reservation, staged publication and failure handling. [Pending]
7. **Integration:** MP4/MKV/MP3/WAV, both languages, CPU/GPU/auto, alternation tests, long audio, memory monitoring and docs. [Pending]

A milestone is considered accepted only after its own tests have run. This repository should not claim real transcription support until milestones 4–6 are implemented.


## Phase 5: Chinese recognition (optional dependencies)

Install a compatible FunASR stack and its model dependencies per the upstream
[FunASR documentation](https://github.com/modelscope/FunASR).
`engine="paraformer_zh"` loads a combined Paraformer + FSMN-VAD + CT punctuation
pipeline. No model is installed or downloaded by Windows CI.
Timestamp mapping is deliberately conservative: only raw tokens that match
punctuated text and time intervals exactly receive aligned status; unmatched
tokens remain in the transcript without fabricated timestamps.
Chinese segmentation uses Jieba if installed, otherwise individual Han characters
are fallback units. Real model compatibility and timing accuracy still need
validation before audio highlighting.
