"""Single-active-engine cache and explicit resource cleanup.

Framework-specific loading is intentionally deferred to Phase 4/5.
"""
from __future__ import annotations

import gc
from threading import RLock
from typing import Callable
from .engines.base import BaseEngine


class ModelManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._key: tuple[str, str] | None = None
        self._engine: BaseEngine | None = None

    def get_or_load(
        self, name: str, device: str, factory: Callable[[], BaseEngine]
    ) -> BaseEngine:
        with self._lock:
            key = (name, device)
            if self._engine is not None and self._key == key:
                return self._engine
            self._release_unlocked()
            instance = factory()
            self._engine = instance
            self._key = key
            return instance

    def _release_unlocked(self) -> None:
        old = self._engine
        self._engine = None
        self._key = None
        if old is not None:
            try:
                old.close()
            finally:
                del old
                gc.collect()
                self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        # No hard dependency on torch for engine discovery and unit tests.
        import sys
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass

    def release(self) -> None:
        with self._lock:
            self._release_unlocked()


_MANAGER = ModelManager()


def release_models() -> None:
    """Release any cached engine loaded via the shared model manager."""
    _MANAGER.release()
