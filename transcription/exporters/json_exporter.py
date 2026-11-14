"""Export word-timed Transcript Studio-style segments without invented scores or times."""
from __future__ import annotations

import json

from transcription.result import TranscriptionResult
from transcription.media import PreparedAudio


def serialize_json(result: TranscriptionResult, prepared: PreparedAudio,
                   *, timed: bool = True) -> str:
    """Export words in segments and as a flattened sequence.

    Segments are sentence-derived rather than WhisperX's original ASR chunks.
    Confidence scores and avg_logprob are not synthesized when unavailable.
    Missing timestamps are represented by null, not guesses. The timeline
    extension preserves whether source-video synchronization is verified.
    """
    if not timed:
        raise ValueError("Only word-timed JSON is published")
    language = "zh" if result.engine == "paraformer_zh" else "en"
    paragraph_ends = {
        paragraph.sentences[-1].char_end for paragraph in result.paragraphs
        if paragraph.sentences
    }
    segments = []
    flat_words = []
    word_index = 0
    for sentence in result.sentences:
        entries = []
        while word_index < len(result.words):
            word = result.words[word_index]
            if word.char_start >= sentence.char_end:
                break
            word_index += 1
            if word.char_start < sentence.char_start:
                continue
            entry = {"word": word.text, "start": word.start, "end": word.end}
            if word.score is not None:
                entry["score"] = word.score
            if word.char_end <= sentence.char_end and word.char_end == sentence.char_end \
                    and sentence.char_end in paragraph_ends:
                entry["breakParagraph"] = True
            entries.append(entry)
            flat_words.append(dict(entry))
        if not entries:
            continue
        timed_words = [word for word in entries if word["start"] is not None and word["end"] is not None]
        segments.append({
            "start": timed_words[0]["start"] if timed_words else None,
            "end": timed_words[-1]["end"] if timed_words else None,
            "text": sentence.text,
            "words": entries,
        })
    # Defensively preserve aligned words that were not assignable to a sentence.
    # (The normal character-span invariants make this path unreachable.)
    for word in result.words[word_index:]:
        entry = {"word": word.text, "start": word.start, "end": word.end}
        if word.score is not None:
            entry["score"] = word.score
        flat_words.append(entry)
        segments.append({"start": word.start, "end": word.end, "text": word.text, "words": [entry]})
    output = {
        "segments": segments,
        "word_segments": flat_words,
        "language": language,
        "transcript_studio": {"format": "transcript-studio", "schemaVersion": 2},
        "timeline": {
            "unit": "seconds",
            "clock": "extracted_audio" if prepared.extracted else "source_audio",
            "original_stream_start": prepared.original_stream_start,
            "extracted_stream_start": prepared.extracted_stream_start,
            "candidate_video_offset": prepared.candidate_video_offset if prepared.extracted else None,
            "video_time_mapping_verified": not prepared.extracted,
        },
    }
    return json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
