# Original-video playback timeline verification

The export contract defines every numeric `start` and `end` as **seconds
relative to the original video's first presented video frame**. Audio-only
inputs retain their own playback clock. If any required mapping evidence is
absent, exports retain the transcript but use `null` timestamps.

## Verification pipeline

For a video input, the pipeline now:

1. Stream-copies the selected source audio into its original-codec container
   for publication, and separately decodes a temporary 16kHz mono PCM WAV
   as the actual recognition input for WhisperX and Paraformer.
2. Checks source versus extracted decoded audio frame counts, sample counts,
   timestamps and continuity using FFprobe.
3. Checks video frame presentation timestamps are monotonic and have no
   gaps greater than 200 milliseconds. Non-video MP4s cannot establish
   `source_video_playback` and fail closed.
4. Compares FFmpeg SHA-256 hashes of the **canonical decoded PCM** from the
   source audio stream, extracted original-codec audio and actual recognition
   WAV. The three must match exactly.
5. Only after all checks pass, maps recognition time `t` to
   `t + (first_original_audio_decoded_frame_PTS - first_video_frame_PTS)`.
   Negative/out-of-duration intervals are withheld (`null`).

The PCM WAV is a temporary inference input, not an extra delivered file.
The published video group remains three files: original-codec audio,
human-readable TXT and timed JSON. A separate packet-check remains available
for diagnostics. The decoder and video mapping checks are independent of
packet-copy timestamp normalization.

The video mapping is conservative: discontinuities, timestamp gaps, malformed
metadata, sample mismatch or unsupported time origins result in
`mapping_verified=false`; no guessed offset is applied. Long recordings
require FFprobe frame scans and three PCM checksum passes, trading speed for
confidence in time-axis identity.

## Limitations

This is a strict FFmpeg presentation-time mapping for continuous media; it
does not promise that every player interprets arbitrary edit lists or
nonstandard container timelines identically. Variable-frame-rate videos
containing presentation gaps over 200 ms are intentionally left unverified.
Model-provided word/sentence boundaries still depend on ASR alignment quality.
For local real-model acceptance, inspect synchronized timestamps at beginning,
middle and end of an actual TED video and Mandarin video.

## Regression checks

- Canonical PCM recognition input and retained original-codec extracted audio.
- Normal video and a delayed-audio video verify playback offsets.
- Audio-only MP4 fails closed when a video playback clock is unavailable.
- English word-level and Mandarin sentence-level JSON, including missing data.
- Full Windows Python 3.11/3.12 CI and the user's physical RTX 5060 Ti test.
