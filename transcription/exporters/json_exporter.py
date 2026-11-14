"""Language-specific timed JSON: English words, Mandarin sentences.

Never publish extracted-audio timestamps as source-video playback timestamps
until decoder and video-playback origin mapping have been independently proven.
"""
from __future__ import annotations

import json
from transcription.result import TranscriptionResult, Word
from transcription.media import PreparedAudio


def _timeline(prepared: PreparedAudio) -> dict:
    video = prepared.extracted
    verified = not video or getattr(prepared, "video_playback_offset", None) is not None
    return {
        "unit": "seconds",
        "reference": "source_video_playback" if video else "source_audio_playback",
        "mapping_verified": verified,
        "video_time_mapping_verified": verified if video else False,
        "original_stream_start": prepared.original_stream_start,
        "extracted_stream_start": prepared.extracted_stream_start,
        "candidate_video_offset": prepared.candidate_video_offset if video else None,
    }


def _group_words(result: TranscriptionResult):
    """Group aligned text entries by canonical sentence character spans."""
    index = 0
    for sentence in result.sentences:
        entries = []
        while index < len(result.words) and result.words[index].char_start < sentence.char_end:
            word = result.words[index]
            index += 1
            if word.char_start >= sentence.char_start and word.char_end <= sentence.char_end:
                entries.append(word)
        yield sentence, entries

def _mapped_interval(start, end, prepared: PreparedAudio):
    if start is None or end is None:
        return None, None
    if prepared.extracted:
        offset = getattr(prepared, "video_playback_offset", None)
        if offset is None:
            return None, None
        start, end = start + offset, end + offset
        duration = getattr(prepared, "video_duration", None)
        if duration is not None and end > duration + .05:
            return None, None
    if start < 0 or end < start:
        return None, None
    return round(start, 3), round(end, 3)


def _word_entry(word: Word, prepared: PreparedAudio) -> dict:
    start, end = _mapped_interval(word.start, word.end, prepared)
    aligned = word.alignment_status == "aligned" and start is not None
    entry = {
        "word": word.text,
        "start": start if aligned else None,
        "end": end if aligned else None,
        "alignment_status": "aligned" if aligned else "missing",
    }
    if word.score is not None:
        entry["score"] = word.score
    return entry


def _english_segments(result: TranscriptionResult, prepared: PreparedAudio):
    paragraph_ends = {p.sentences[-1].char_end for p in result.paragraphs if p.sentences}
    segments, flattened = [], []
    for sentence, words in _group_words(result):
        entries = []
        for word in words:
            entry = _word_entry(word, prepared)
            if word.char_end == sentence.char_end and sentence.char_end in paragraph_ends:
                entry["breakParagraph"] = True
            entries.append(entry)
            flattened.append(dict(entry))
        aligned = [e for e in entries if e["start"] is not None]
        segments.append({
            "text": sentence.text,
            "start": min(e["start"] for e in aligned) if aligned else None,
            "end": max(e["end"] for e in aligned) if aligned else None,
            "alignment_status": "aligned" if len(aligned) == len(entries) and entries
                else "partial" if aligned else "missing",
            "words": entries,
        })
    return segments, flattened


def _chinese_segments(result: TranscriptionResult, prepared: PreparedAudio) -> list[dict]:
    segments = []
    for sentence, words in _group_words(result):
        complete = bool(words) and all(
            w.alignment_status == "aligned" and w.start is not None and w.end is not None
            for w in words
        )
        if complete:
            complete = all(words[i].start >= words[i - 1].end for i in range(1, len(words)))
        start, end = _mapped_interval(words[0].start, words[-1].end, prepared) if complete else (None, None)
        complete = complete and start is not None
        segments.append({
            "text": sentence.text,
            "start": start if complete else None,
            "end": end if complete else None,
            "alignment_status": "aligned" if complete else "missing",
        })
    return segments


def has_missing_chinese_sentence_timing(result: TranscriptionResult,
                                        prepared: PreparedAudio) -> bool:
    """Use the same sentence logic as JSON to determine partial status."""
    return result.engine == "paraformer_zh" and any(
        segment["alignment_status"] == "missing"
        for segment in _chinese_segments(result, prepared)
    )


def serialize_json(result: TranscriptionResult, prepared: PreparedAudio,
                   *, timed: bool = True) -> str:
    if not timed:
        raise ValueError("Only timed JSON is published")
    timeline = _timeline(prepared)
    if result.engine == "paraformer_zh":
        output = {
            "schema_version": 2,
            "language": "zh",
            "timeline": timeline,
            "segments": _chinese_segments(result, prepared),
        }
    else:
        segments, flat = _english_segments(result, prepared)
        output = {
            "segments": segments,
            "word_segments": flat,
            "language": "en",
            "transcript_studio": {"format": "transcript-studio", "schemaVersion": 2},
            "timeline": timeline,
        }
    return json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
