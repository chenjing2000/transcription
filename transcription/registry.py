"""Discover trusted engine declarations without importing heavy ASR frameworks."""
from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from threading import RLock
from .errors import TranscriptionError
from .result import EngineInfo

_registry_lock = RLock()


def _discover() -> dict[str, EngineInfo]:
    package = import_module("transcription.engines")
    result: dict[str, EngineInfo] = {}
    for item in sorted(iter_modules(package.__path__), key=lambda x: x.name):
        if item.name.startswith("_") or item.name == "base" or item.ispkg:
            continue
        module = import_module(f"transcription.engines.{item.name}")
        for spec in getattr(module, "ENGINE_SPECS", ()):
            if not isinstance(spec, EngineInfo):
                raise TranscriptionError("INVALID_ENGINE", f"Invalid declaration in {item.name}")
            if not spec.name or spec.name in result:
                raise TranscriptionError("DUPLICATE_ENGINE", f"Duplicate or empty engine ID: {spec.name}")
            result[spec.name] = spec
    return result


def list_engines() -> list[EngineInfo]:
    """Return declared engine choices (NOT an inference-ready health check)."""
    with _registry_lock:
        return list(_discover().values())


def get_engine(name: str) -> EngineInfo:
    with _registry_lock:
        engines = _discover()
    try:
        return engines[name]
    except KeyError as exc:
        raise TranscriptionError("ENGINE_NOT_FOUND", f"Unknown engine: {name}") from exc
