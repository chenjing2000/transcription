"""Stable public data contracts; times are seconds and character spans are half-open."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

Status = Literal["success", "partial", "failed"]
AlignmentStatus = Literal["aligned", "estimated", "missing"]


@dataclass(frozen=True, slots=True)
class EngineInfo:
    name: str
    model_id: str
    language: str
    description: str
    devices: tuple[str, ...] = ("cpu", "gpu")


@dataclass(slots=True)
class Word:
    text: str
    start: float | None
    end: float | None
    char_start: int
    char_end: int
    alignment_status: AlignmentStatus = "missing"

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError("Invalid half-open character span")
        if (self.start is None) != (self.end is None):
            raise ValueError("Word timestamps must both exist or both be absent")
        if self.start is not None and (not isfinite(self.start) or not isfinite(self.end) or self.start < 0 or self.end < self.start):
            raise ValueError("Invalid word timestamps")
        if self.start is None and self.alignment_status != "missing":
            raise ValueError("Missing timestamps must be marked missing")


@dataclass(slots=True)
class Character:
    text: str
    char_index: int
    start: float | None = None
    end: float | None = None
    alignment_status: AlignmentStatus = "missing"

    def __post_init__(self) -> None:
        if len(self.text) != 1 or self.char_index < 0:
            raise ValueError("Invalid character or index")
        if (self.start is None) != (self.end is None):
            raise ValueError("Character timestamps must both exist or both be absent")
        if self.start is not None and (
            not isfinite(self.start) or not isfinite(self.end)
            or self.start < 0 or self.end < self.start
        ):
            raise ValueError("Invalid character timestamps")
        if self.start is None and self.alignment_status != "missing":
            raise ValueError("Missing character timestamps must be marked missing")

@dataclass(slots=True)
class Sentence:
    text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class Paragraph:
    text: str
    sentences: list[Sentence] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionResult:
    status: Status
    info: str
    engine: str
    source_path: Path
    device: str | None = None
    audio_path: Path | None = None
    output_dir: Path | None = None
    text: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    characters: list[Character] | None = None
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_code: str | None = None
