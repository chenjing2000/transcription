# Media handling — Phase 3

The `transcription.media` module is now implemented independently of the unimplemented ASR API.

```python
from transcription.media import probe_media, prepare_audio

metadata = probe_media("lecture.mp4")
print(metadata.duration, metadata.audio_streams)
audio = prepare_audio("lecture.mp4", staging_dir="temporary-work")
print(audio.audio_path, audio.extracted, audio.candidate_video_offset)
```

- Supported input suffixes: MP4, MKV, MP3, WAV. FFprobe inspects actual codec/stream metadata, including absolute stream index.
- Video: choose zero-based audio track (default 0), select via its global stream index and stream-copy only the encoded audio into a compatible container, without sample-rate/channel changes or transcoding. Unsupported codecs fail; the caller must handle failures.
- MP3/WAV: use original file; no copied audio artifact.
- FFmpeg/FFprobe must be on PATH (or explicitly passed into functions).
- `prepare_audio` writes video extraction only into a **caller-supplied staging directory**; does not yet create `transcript results`, determine group suffixes or publish files. Those are Phase 6 tasks.
- It will not overwrite an existing staging target.
- It retains original/extracted stream `start_time` and exposes a *candidate* offset. Complex edit lists, discontinuous packet timestamps and transport/container timeline differences require more validation before the produced word timestamps may drive exact video highlighting.
- MP4/MKV stream-copy output extension depends on codec; codec-to-container mapping is intentionally conservative.
- Do not treat `bit_rate` metadata as exact independent validation: muxers may estimate or omit it even when packets are copied unchanged.
- For inference, models may independently decode/resample audio internally, without touching the preserved stream-copy artifact.
