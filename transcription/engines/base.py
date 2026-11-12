"""Protocol implemented by inference adapters in later phases."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from transcription.result import TranscriptionResult


class BaseEngine(ABC):
    @abstractmethod
    def transcribe(self, file_path: Path) -> TranscriptionResult:
        """Run recognition and alignment, without exporting files."""

    @abstractmethod
    def close(self) -> None:
        """Release ASR, VAD, punctuation and alignment model resources."""
