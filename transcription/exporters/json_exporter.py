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
    return {
        "unit": "seconds",
        "reference": "source_video_playback" if video else "source_audio_playback",
        "mapping_verified": not video,
        "video_time_mapping_verified": not video,
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


def _word_entry(word: Word, verified: bool) -> dict:
    aligned = verified and word.alignment_status == "aligned" and word.start is not None
    entry = {
        "word": word.text,
        "start": round(word.start, 3) if aligned else None,
        "end": round(word.end, 3) if aligned else None,
        "alignment_status": "aligned" if aligned else "missing",
    }
    if word.score is not None:
        entry["score"] = word.score
    return entry


def _english_segments(result: TranscriptionResult, verified: bool):
    paragraph_ends = {p.sentences[-1].char_end for p in result.paragraphs if p.sentences}
    segments, flattened = [], []
    for sentence, words in _group_words(result):
        entries = []
        for word in words:
            entry = _word_entry(word, verified)
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


def _chinese_segments(result: TranscriptionResult, verified: bool) -> list[dict]:
    segments = []
    for sentence, words in _group_words(result):
        complete = verified and bool(words) and all(
            w.alignment_status == "aligned" and w.start is not None and w.end is not None
            for w in words
        )
        if complete:
            complete = all(words[i].start >= words[i - 1].end for i in range(1, len(words)))
        segments.append({
            "text": sentence.text,
            "start": round(words[0].start, 3) if complete else None,
            "end": round(words[-1].end, 3) if complete else None,
            "alignment_status": "aligned" if complete else "missing",
        })
    return segments


def has_missing_chinese_sentence_timing(result: TranscriptionResult,
                                        prepared: PreparedAudio) -> bool:
    """Use the same sentence logic as JSON to determine partial status."""
    return result.engine == "paraformer_zh" and any(
        segment["alignment_status"] == "missing"
        for segment in _chinese_segments(result, not prepared.extracted)
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
            "segments": _chinese_segments(result, timeline["mapping_verified"]),
        }
    else:
        segments, flat = _english_segments(result, timeline["mapping_verified"])
        output = {
            "segments": segments,
            "word_segments": flat,
            "language": "en",
            "transcript_studio": {"format": "transcript-studio", "schemaVersion": 2},
            "timeline": timeline,
        }
    return json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
