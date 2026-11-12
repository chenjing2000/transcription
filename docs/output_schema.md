# Output contract (planned; not yet implemented)

For input `Lecture.mp4`, write under `<source directory>/transcript results/`:

- `Lecture.m4a` (example only: output suffix depends on actual codec and compatible container)
- `Lecture.timed.json`: canonical text, words with `start`, `end`, half-open `char_start`, `char_end`, quality flags; Chinese may also have character-level alignment.
- `Lecture.timed.txt`: individual word + start/end timing lines for diagnostics.
- `Lecture.json`: same canonical text segmented into sentences and paragraphs, without time fields.
- `Lecture.txt`: paragraphs for reading, with natural punctuation and blank lines.

Use `Lecture (1).*` for **all files in the same batch** if any target would collide. Audio-only MP3/WAV inputs generate four outputs and reuse their source audio (not duplicated). Record audio extraction time mapping for video synchronization. A failure must not publish a completed-looking partial set. Never fabricate timestamps: null plus alignment flags indicate missing values.
