# Transcription implementation milestones

1. Foundation: public API, result dataclasses and validation. [Implemented]
2. Registry and model cache: metadata-driven engine discovery, single-active-engine manager. [Framework implemented]
3. Media: FFprobe metadata, FFmpeg stream-copy and conservative codec mapping. [Basic Windows CI passed]
4. WhisperX: large-v3 English recognition, forced word alignment. [Mock CI passed; real ASR/GPU unverified]
5. FunASR: Paraformer Chinese, VAD, punctuation and conservative word/character alignment. [Mock CI passed; real ASR/GPU unverified]
6. Export: canonical text segmentation, four JSON/TXT files, video original-codec audio, grouped collision numbering, staged publishing and ordinary-error rollback. [Implemented; CI pending]
7. Complete real-model CPU/GPU integrations, extended audio/video codecs and complex video timebases, model switching and memory profiling. [Pending]

Unverified model interfaces and time alignment quality must not be mistaken for completed end-to-end model acceptance.
