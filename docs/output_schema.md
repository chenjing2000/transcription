# Transcript output schema

A video input `D:/Media/Lecture.mp4` publishes exactly three new files in
`D:/Media/transcript results/`: `Lecture.timed.json`, `Lecture.txt`, and
`Lecture.m4a` (or another original-codec-compatible extension). Audio is
stream-copied, not re-encoded. MP3/WAV inputs publish the two text files;
the original audio remains the third usable item without duplicating it.

`Lecture.json` and `Lecture.timed.txt` are no longer generated.

## Time contract

**Every numeric `start`/`end` is elapsed seconds on the specified source
media playback timeline**, never a character index or offset metadata.
For video inputs the reference is `source_video_playback`; for audio input
`source_audio_playback`. A time is only numeric when its mapping to that
reference is verified. Missing times are paired `null` values marked
`alignment_status: "missing"`, not zero or an estimate.

**Current limitation:** video packet-level continuity checks have been
implemented, but decoded audio sample-to-original-video playback mapping
is still unverified. Therefore all video `start/end` values remain
`null` and the transcription result is `partial`. The `candidate_video_offset`
field is diagnostic ONLY; it is not used to manufacture public timestamps.
See [timeline verification](timeline_verification.md).

## English (WhisperX)

English exports Transcript Studio-style `segments[].words` plus flattened
`word_segments`. Example for *audio input*:

```json
{
  "language": "en",
  "segments": [
    {
      "text": "Hello there.",
      "start": 0.12,
      "end": 0.78,
      "alignment_status": "aligned",
      "words": [
        {"word": "Hello", "start": 0.12, "end": 0.42, "alignment_status": "aligned"},
        {"word": "there.", "start": 0.50, "end": 0.78, "alignment_status": "aligned", "breakParagraph": true}
      ]
    }
  ],
  "word_segments": [
    {"word": "Hello", "start": 0.12, "end": 0.42, "alignment_status": "aligned"},
    {"word": "there.", "start": 0.50, "end": 0.78, "alignment_status": "aligned", "breakParagraph": true}
  ],
  "transcript_studio": {"format": "transcript-studio", "schemaVersion": 2},
  "timeline": {"unit": "seconds", "reference": "source_audio_playback", "mapping_verified": true}
}
```

Only genuine model-provided confidence `score` values are emitted.
`avg_logprob` is not fabricated. Sentence status is `partial` when
some words lack verified time, `missing` when none have it.

## Chinese (Paraformer)

Chinese exports **sentence-only** `segments` and does not export
`words`, `word_segments` or `characters`:

```json
{
  "schema_version": 2,
  "language": "zh",
  "timeline": {"unit": "seconds", "reference": "source_audio_playback", "mapping_verified": true},
  "segments": [
    {"text": "你好。", "start": 0.1, "end": 0.4, "alignment_status": "aligned"},
    {"text": "今天学习。", "start": null, "end": null, "alignment_status": "missing"}
  ]
}
```

Sentence bounds derive from verified internal token timing. Any missing or
inconsistent token makes the **whole sentence** missing; the complete text is
still kept. Internal word/character timing data remain available for
validation even though they are not exported.

## Publishing

Repeated runs share one group suffix: `Lecture (1).timed.json`,
`Lecture (1).txt`, `Lecture (1).m4a`. Existing files are never
overwritten. A per-directory lock and exclusive file creation protect
cooperative publishers; ordinary failures roll back newly created files.
A crash may leave partial artifacts or a stale lock (not crash-atomic).
