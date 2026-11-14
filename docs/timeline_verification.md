# Source-video time-axis verification — batch 1

The target output contract requires **every numeric** `start`/`end`
in a video `.timed.json` to refer to the original video's playback timeline,
not merely elapsed seconds in extracted audio. Unknown locations must be
`null`, never an assumed zero or a guessed offset.

## What is implemented in batch 1

`transcription.processing.timeline.verify_stream_copy_packet_timeline()`
compares FFprobe audio packet PTS, duration, and byte sizes for original and
stream-copied audio. It accepts a constant packet timestamp shift only when:
- source and copied packet counts and sizes agree;
- packet durations agree;
- both packet sequences have continuous, ordered presentation timestamps;
- the shift is constant across the entire inspected sequence;
- no packet has unhandled side-data, incomplete timestamps or malformed values.

The packet count is bounded. Any error or ambiguity fails closed, reporting a
reason instead of a fabricated shift. `PreparedAudio` retains three
diagnostic values: `packet_timeline_verified`,
`packet_timeline_offset`, `packet_timeline_reason`.

**Important limitation:** Even a verified packet time shift is *not* proof
that decoded ASR samples align to compressed packet PTS, nor that the container
timestamp origin matches the video player's zero point. Encoder priming,
edit lists, skip-sample metadata, discontinuities and video presentation
offsets need additional evidence. This batch therefore does **not** mark
`video_time_mapping_verified=true` and does **not** apply the packet offset
to JSON word timestamps.

The existing exported JSON still uses the extracted-audio clock and therefore
**must not yet be treated as source-video-aligned**. Batch 2 must integrate
fully validated video-playback mapping or emit null timestamps when this
cannot be proven. Chinese sentence-level exports are also deferred to batch 2.

## Tests and acceptance

Unit tests cover matching packet timelines, discontinuities, offset drift,
size mismatches, missing metadata and packet count mismatches. Integration
testing should additionally compare real FFmpeg stream-copies with non-zero
starts and discontinuities. The final original-video timestamp acceptance
remains pending until decoder and playback-origin validation are implemented.
