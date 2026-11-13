"""JSON output schemas, derived from one authoritative transcript text."""
from __future__ import annotations

import json
from transcription.result import TranscriptionResult
from transcription.media import PreparedAudio


def _structure(result: TranscriptionResult) -> dict:
    return {
        "paragraphs": [
            {
                "text": paragraph.text,
                "sentences": [
                    {"text": item.text, "char_start": item.char_start,
                     "char_end": item.char_end}
                    for item in paragraph.sentences
                ],
            }
            for paragraph in result.paragraphs
        ],
        "sentences": [
            {"text": item.text, "char_start": item.char_start,
             "char_end": item.char_end}
            for item in result.sentences
        ],
    }


def serialize_json(result: TranscriptionResult, prepared: PreparedAudio,
                   *, timed: bool) -> str:
    """Untimed output never contains seconds, audio offsets or timing metadata."""
    output = {
        "schema_version": 1,
        "engine": result.engine,
        "device": result.device,
        "source_file": result.source_path.name,
        "status": result.status,
        "text": result.text,
        **_structure(result),
        "warnings": result.warnings,
    }
    if timed:
        output["words"] = [
            {"text": w.text, "start": w.start, "end": w.end,
             "char_start": w.char_start, "char_end": w.char_end,
             "alignment_status": w.alignment_status}
            for w in result.words
        ]
        output["characters"] = None if result.characters is None else [
            {"text": c.text, "char_index": c.char_index,
             "start": c.start, "end": c.end,
             "alignment_status": c.alignment_status}
            for c in result.characters
        ]
        output["timeline"] = {
            "unit": "seconds",
            "clock": "extracted_audio" if prepared.extracted else "source_audio",
            "original_stream_start": prepared.original_stream_start,
            "extracted_stream_start": prepared.extracted_stream_start,
            "candidate_video_offset": prepared.candidate_video_offset if prepared.extracted else None,
            "video_time_mapping_verified": not prepared.extracted,
        }
    return json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
