from pathlib import Path
from transcription.engines.base import BaseEngine
from transcription.model_manager import ModelManager


class FakeEngine(BaseEngine):
    def __init__(self):
        self.closed = False

    def transcribe(self, file_path: Path):
        raise NotImplementedError

    def close(self):
        self.closed = True


def test_reuse_and_release_on_switch():
    manager = ModelManager()
    first = manager.get_or_load("a", "cpu", FakeEngine)
    assert manager.get_or_load("a", "cpu", FakeEngine) is first
    second = manager.get_or_load("b", "cpu", FakeEngine)
    assert second is not first
    assert first.closed
    manager.release()
    assert second.closed


def test_device_change_reloads():
    manager = ModelManager()
    old = manager.get_or_load("a", "cpu", FakeEngine)
    new = manager.get_or_load("a", "gpu", FakeEngine)
    assert old.closed
    assert new is not old
    manager.release()
