"""Discover trusted engine declarations and their lazy adapter factories."""
from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from threading import RLock

from .errors import TranscriptionError
from .result import EngineInfo

_registry_lock = RLock()


def _discover() -> dict[str, tuple[EngineInfo, str, str]]:
    """Map engine ID to metadata, module path and adapter class name.

    Importing an adapter module must not load model weights or ASR dependencies.
    The class is resolved when actually running an engine, so test doubles and
    new engine modules do not require changes to the public API.
    """
    package = import_module("transcription.engines")
    result: dict[str, tuple[EngineInfo, str, str]] = {}
    for item in sorted(iter_modules(package.__path__), key=lambda x: x.name):
        if item.name.startswith("_") or item.name == "base" or item.ispkg:
            continue
        path = f"transcription.engines.{item.name}"
        module = import_module(path)
        specs = getattr(module, "ENGINE_SPECS", ())
        if not specs:
            continue
        class_name = getattr(module, "ENGINE_CLASS", None)
        if not isinstance(class_name, str) or not class_name:
            raise TranscriptionError("INVALID_ENGINE", f"Missing ENGINE_CLASS in {item.name}")
        for spec in specs:
            if not isinstance(spec, EngineInfo):
                raise TranscriptionError("INVALID_ENGINE", f"Invalid declaration in {item.name}")
            if not spec.name or spec.name in result:
                raise TranscriptionError("DUPLICATE_ENGINE", f"Duplicate or empty engine ID: {spec.name}")
            result[spec.name] = (spec, path, class_name)
    return result


def list_engines() -> list[EngineInfo]:
    """Return declared engine choices, without loading model weights."""
    with _registry_lock:
        return [entry[0] for entry in _discover().values()]


def _entry(name: str) -> tuple[EngineInfo, str, str]:
    with _registry_lock:
        entries = _discover()
    try:
        return entries[name]
    except KeyError as exc:
        raise TranscriptionError("ENGINE_NOT_FOUND", f"Unknown engine: {name}") from exc


def get_engine(name: str) -> EngineInfo:
    return _entry(name)[0]


def get_engine_adapter(name: str):
    """Resolve a registered engine's adapter without any API-side engine switch."""
    spec, module_name, class_name = _entry(name)
    module = import_module(module_name)
    adapter = getattr(module, class_name, None)
    if not isinstance(adapter, type):
        raise TranscriptionError("INVALID_ENGINE", f"Adapter {class_name} missing in {module_name}")
    return spec, adapter
