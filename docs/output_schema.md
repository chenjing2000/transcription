# Transcript output schema

For video input `D:/Media/Lecture.mp4`, publish **three files** in
`D:/Media/transcript results/`:

- `Lecture.timed.json`: word-level timestamps in Transcript Studio-style
  `segments` and `word_segments`; `language` and `transcript_studio` metadata.
- `Lecture.txt`: punctuated paragraphs separated by blank lines.
- `Lecture.m4a` (example): extracted video audio, copying the original codec
  without re-encoding, using a compatible container extension.

For MP3/WAV inputs, the source audio already exists, so **two new files**
(`.timed.json` and `.txt`) are published; the source audio is not duplicated.
The publisher no longer creates `.json` or `.timed.txt` files.

## Timed JSON structure

The word-level layout follows the uploaded Transcript Studio v2 example.
`segments` groups words into sentence-derived segments; `word_segments`
is a flattened copy of the same word entries:

```json
{
  "segments": [
    {
      "start": 7.293,
      "end": 7.613,
      "text": "Hello.",
      "words": [
        {"word": "Hello.", "start": 7.293, "end": 7.613, "score": 0.816}
      ]
    }
  ],
  "word_segments": [
    {"word": "Hello.", "start": 7.293, "end": 7.613, "score": 0.816}
  ],
  "language": "en",
  "transcript_studio": {"format": "transcript-studio", "schemaVersion": 2},
  "timeline": {
    "unit": "seconds",
    "clock": "extracted_audio",
    "original_stream_start": 0.0,
    "extracted_stream_start": 0.0,
    "candidate_video_offset": 0.0,
    "video_time_mapping_verified": false
  }
}
```

The values above are illustrative. All word times are in seconds. The
`score` property is **optional**: only real recognition/alignment confidence
reported by the engine is exported; FunASR word confidence is not currently
available. Neither confidence scores nor `avg_logprob` are invented.
Missing word timestamps are `null`; segments with no timed words also
have `null` start/end. Paragraph ends may be marked by
`breakParagraph: true` on the last word of a derived paragraph.

Segments are created from the library's sentence/paragraph segmentation,
which **may differ from the original ASR chunks**. The `transcript_studio`
schema marker mirrors the supplied reference; full compatibility with every
Transcript Studio consumer requires an import test. The added `timeline`
extension retains the prior cautious time-base metadata.

## Naming, publication and timeline

Later runs receive `Lecture (1).timed.json`, `Lecture (1).txt`,
`Lecture (1).m4a`, etc. The suffix is shared by all outputs; no existing
file is overwritten. The publisher uses a directory lock and exclusive file
creation and rolls back ordinary failures. Process crashes can leave partial
artifacts or stale locks, so publishing is not crash-atomic.

For video, `timeline.clock` reports the **extracted audio** time base.
`candidate_video_offset` is not a verified video playback offset when
`video_time_mapping_verified=false`. Do not infer frame-accurate original
video sync from it.

Alignment warnings still determine `TranscriptionResult.status == "partial"`
when recognition produces usable text but lacks verified timestamps.
