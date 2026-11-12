"""Public transcription entry point; Phase 4 returns in-memory recognition results.

The agreed four-file publication is Phase 6 and is not silently simulated.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .errors import TranscriptionError
from .media import prepare_audio, SUPPORTED_INPUTS
from .model_manager import _MANAGER
from .registry import get_engine
from .result import TranscriptionResult

DEVICES = frozenset({"cpu", "gpu", "auto"})


def _device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise TranscriptionError("DEPENDENCY_MISSING", "PyTorch is required to select a WhisperX device") from exc
    available = bool(torch.cuda.is_available())
    if requested == "gpu" and not available:
        raise TranscriptionError("DEVICE_UNAVAILABLE", "GPU requested but CUDA is unavailable")
    return "cuda" if available else "cpu"


def transcribe(file_path: str | Path, engine: str, *, device: str = "auto",
               audio_track: int = 0) -> TranscriptionResult:
    """Recognize English with WhisperX; publishing and Chinese ASR remain pending.

    All expected failures return a status='failed' result. Successful English
    recognition is 'partial' until the Phase 6 output contract is implemented.
    No files are written into 'transcript results' in this phase.
    """
    source = Path(file_path)
    result = TranscriptionResult(
        status="failed", info="", engine=engine, source_path=source, device=device,
    )
    try:
        if device not in DEVICES:
            raise TranscriptionError("INVALID_DEVICE", "device must be gpu, cpu or auto")
        if isinstance(audio_track, bool) or not isinstance(audio_track, int) or audio_track < 0:
            raise TranscriptionError("INVALID_AUDIO_TRACK", "audio_track must be a nonnegative integer")
        spec = get_engine(engine)
        if source.suffix.lower() not in SUPPORTED_INPUTS:
            raise TranscriptionError("UNSUPPORTED_FORMAT", "Supported inputs: mp4, mkv, mp3, wav")
        if not source.is_file():
            raise TranscriptionError("INVALID_INPUT", f"Input file does not exist: {source}")
        if spec.name != "whisperx_large_v3":
            raise TranscriptionError("NOT_IMPLEMENTED", f"Engine {spec.name} is registered but not implemented")
        selected = _device(device)
        from .engines.whisperx import WhisperXEngine
        with TemporaryDirectory(prefix="transcription-") as workdir:
            prepared = prepare_audio(source, workdir, audio_track=audio_track)
            result = _MANAGER.run(
                spec.name, selected,
                lambda: WhisperXEngine(model_id=spec.model_id, name=spec.name, device=selected),
                lambda adapter: adapter.transcribe(prepared.audio_path),
            )
            result.source_path = source
            result.device = "gpu" if selected == "cuda" else "cpu"
            # Extracted audio is staged and removed after this function exits.
            # Never return a stale temp path as though it were a published artifact.
            result.audio_path = source if not prepared.extracted else None
            result.warnings.extend(prepared.warnings)
        return result
    except TranscriptionError as exc:
        result.error_code = exc.code
        result.info = str(exc)
        return result
    except Exception as exc:
        result.error_code = "UNEXPECTED_ERROR"
        result.info = f"Unexpected transcription failure: {type(exc).__name__}: {exc}"
        return result
