"""Human-readable transcript and word timing text exports."""
from __future__ import annotations

from transcription.result import TranscriptionResult


def plain_txt(result: TranscriptionResult) -> str:
    return "\n\n".join(p.text.strip() for p in result.paragraphs if p.text.strip()) + "\n"


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--.---"
    millis = round(seconds * 1000)
    hh, millis = divmod(millis, 3600000)
    mm, millis = divmod(millis, 60000)
    ss, millis = divmod(millis, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{millis:03d}"


def timed_txt(result: TranscriptionResult) -> str:
    lines = []
    for word in result.words:
        lines.append(f"[{_clock(word.start)} - {_clock(word.end)}] {word.text}")
    if not lines:
        lines = ["[--:--:--.--- - --:--:--.---] " + result.text]
    return "\n".join(lines) + "\n"
