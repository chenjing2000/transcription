"""Lazy WhisperX English ASR adapter with forced word alignment.

Heavy dependencies and model weights are loaded only when an engine is used.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

from transcription.engines.base import BaseEngine
from transcription.errors import TranscriptionError
from transcription.result import EngineInfo, TranscriptionResult, Word

ENGINE_SPECS = (
    EngineInfo(
        name="whisperx_large_v3",
        model_id="large-v3",
        language="en",
        description="WhisperX large-v3 English with forced word alignment",
    ),
)


ENGINE_CLASS = "WhisperXEngine"


def _word_entries(segments: list[dict[str, Any]]) -> tuple[str, list[Word], list[str]]:
    """Preserve the aligned segment text, mapping aligned words to its spans.

    Failing to match a token is reported instead of fabricating offsets.
    """
    chunks: list[str] = []
    spans: list[Word] = []
    warnings: list[str] = []
    length = 0
    for segment in segments:
        piece = str(segment.get("text") or "")
        if not piece:
            continue
        prefix = " " if chunks and not piece[0].isspace() and not chunks[-1].endswith((" ", "\n")) else ""
        offset = length + len(prefix)
        chunks.extend((prefix, piece))
        length = offset + len(piece)
        cursor = 0
        for item in segment.get("words") or []:
            raw = str(item.get("word") or "")
            token = raw.strip()
            if not token:
                continue
            pos = piece.find(token, cursor)
            if pos < 0:
                # Transcript takes precedence; an unmatched token cannot have a valid
                # character span and is explicitly reported.
                warnings.append(f"Aligned token not located in transcript text: {token!r}")
                continue
            start = item.get("start")
            end = item.get("end")
            valid = (
                isinstance(start, (int, float)) and not isinstance(start, bool)
                and isinstance(end, (int, float)) and not isinstance(end, bool)
                and 0 <= start <= end
            )
            raw_score = item.get("score")
            score = float(raw_score) if (isinstance(raw_score, (int, float))
                and not isinstance(raw_score, bool) and isfinite(raw_score)
                and 0 <= raw_score <= 1) else None
            spans.append(Word(
                text=token, start=float(start) if valid else None,
                end=float(end) if valid else None,
                char_start=offset + pos, char_end=offset + pos + len(token),
                alignment_status="aligned" if valid else "missing", score=score,
            ))
            if not valid:
                warnings.append(f"No reliable timestamp for word: {token!r}")
            cursor = pos + len(token)
    complete = "".join(chunks)
    removed_prefix = len(complete) - len(complete.lstrip())
    if removed_prefix:
        for word in spans:
            word.char_start -= removed_prefix
            word.char_end -= removed_prefix
    return complete.strip(), spans, warnings


class WhisperXEngine(BaseEngine):
    """Adapter instance owns both ASR and alignment models for cache release."""

    def __init__(self, *, model_id: str, name: str, device: str, batch_size: int = 4):
        try:
            import whisperx
        except ImportError as exc:
            raise TranscriptionError(
                "DEPENDENCY_MISSING", "WhisperX is not installed. Install the optional ASR dependencies."
            ) from exc
        self._whisperx = whisperx
        self.model_id = model_id
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.asr = None
        self.align_model = None
        self.align_metadata = None
        try:
            self.asr = whisperx.load_model(
                model_id, device, compute_type="float16" if device == "cuda" else "int8",
                language="en", task="transcribe",
            )
        except Exception as exc:
            raise TranscriptionError("MODEL_LOAD_FAILED", f"WhisperX model loading failed: {exc}") from exc

    def transcribe(self, file_path: Path) -> TranscriptionResult:
        try:
            audio = self._whisperx.load_audio(str(file_path))
            raw = self.asr.transcribe(audio, batch_size=self.batch_size)
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"WhisperX recognition failed: {exc}") from exc
        segments = list(raw.get("segments") or [])
        if not segments:
            return TranscriptionResult(
                status="partial", info="No speech detected", engine=self.name,
                source_path=file_path, device=self.device, text="",
                warnings=["Empty speech result"],
            )
        # Recognition is authoritative. If the alignment model fails or
        # unexpectedly changes the transcript, preserve all recognized words
        # rather than throwing away usable ASR output.
        canonical_text, original_words, _ = _word_entries(segments)
        # Original ASR segments may carry coarse times, but they do not
        # demonstrate successful forced alignment.
        for word in original_words:
            word.start = None
            word.end = None
            word.alignment_status = "missing"
        try:
            if self.align_model is None:
                self.align_model, self.align_metadata = self._whisperx.load_align_model(
                    language_code="en", device=self.device,
                )
            aligned = self._whisperx.align(
                segments, self.align_model, self.align_metadata, audio, self.device,
                return_char_alignments=False,
            )
            text, words, warnings = _word_entries(list(aligned.get("segments") or []))
            if text != canonical_text:
                text, words = canonical_text, original_words
                warnings = ["Alignment changed recognized text; original transcript retained without word timing"]
        except Exception as exc:
            text, words = canonical_text, original_words
            warnings = [f"Word alignment unavailable; original recognized text retained: {type(exc).__name__}: {exc}"]
        if not text:
            raise TranscriptionError("TRANSCRIPTION_FAILED", "Recognition produced no text")
        if not words:
            warnings.append("No aligned word entries; timestamps unavailable")
        return TranscriptionResult(
            status="partial", info="English recognition complete; export handled by public API",
            engine=self.name, source_path=file_path, device=self.device,
            text=text, words=words,
            warnings=warnings,
        )

    def close(self) -> None:
        self.asr = None
        self.align_model = None
        self.align_metadata = None
