# Transcript output schema

Input `D:/Media/Lecture.mp4` saves a group to `D:/Media/transcript results/`:

- `Lecture.m4a` (example: extension depends on original audio codec and compatible container; stream copy only)
- `Lecture.timed.json`: schema_version=1, source_file, engine, device, status, text, sentence/paragraph spans, words with start/end (seconds), char_start/char_end (half-open), alignment_status; Chinese optional characters, and timeline.
- `Lecture.timed.txt`: one line per word, `[HH:MM:SS.mmm - HH:MM:SS.mmm] word`; missing times display `--:--:--.---`.
- `Lecture.json`: same canonical text and sentence/paragraph spans, no word or time fields.
- `Lecture.txt`: punctuated paragraphs separated by blank lines.

Audio input MP3/WAV creates the four text artifacts only. Missing timestamps must never be invented.

## Naming and publication

The next run uses `Lecture (1).*`, then `Lecture (2).*`, and so forth. Every file in a batch receives one shared suffix. The publisher checks for collisions with *any* target filename. It also takes a directory-wide exclusive lock, opens final files in exclusive `xb` mode, and rolls back files created on ordinary exceptions. No existing output is overwritten.

`output_dir` is fixed to the source file's `transcript results` directory; no additional public argument is needed.

A concurrent publisher returns `OUTPUT_BUSY` while the directory lock is held; stale locks must not be deleted while another publisher is active. A crash/power failure mid-copy is not a fully atomic filesystem transaction. The directory may contain incomplete files requiring manual cleanup; in-process exceptions trigger rollback. The lock directory `.transcription-publish.lock` can require manual removal after a process crash, after verifying that no publisher is active. A separate manifest/transaction recovery mechanism is a future reliability improvement.

## Timeline semantics

Internal word/character times refer to the audio decoded for recognition. On video, `timeline` records original_stream_start, extracted_stream_start, and a **candidate** offset, but sets `video_time_mapping_verified=false`. Do not treat that offset as verified video playback synchronization for edit lists or discontinuities.

A complete alignment returns `success` when no warnings occur. Missing or unverified alignment returns `partial` and retains the same textual content.

## Input text invariants

`timed.json.text == plain.json.text == result.text`. Sentence spans refer to character positions in that string. Paragraph `text` is a slice of canonical text. The TXT format is human-readable and may introduce paragraph breaks instead of preserving exactly identical whitespace.
