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


def use(manager, key="a", device="cpu"):
    return manager.run(key, device, FakeEngine, lambda adapter: adapter)


def test_reuse_and_release_on_switch():
    manager = ModelManager()
    first = use(manager)
    assert use(manager) is first
    second = use(manager, "b")
    assert second is not first
    assert first.closed
    manager.release()
    assert second.closed


def test_device_change_reloads():
    manager = ModelManager()
    old = use(manager)
    new = use(manager, "a", "gpu")
    assert old.closed
    assert new is not old
    manager.release()


def test_switch_does_not_release_engine_during_inference():
    manager = ModelManager()
    first = use(manager)
    def operation(adapter):
        assert adapter is first
        assert not adapter.closed
        return "completed"
    assert manager.run("a", "cpu", FakeEngine, operation) == "completed"
    assert first.closed is False
    manager.release()


def test_failed_factory_does_not_retain_old_model():
    manager = ModelManager()
    old = use(manager)
    def fail():
        raise RuntimeError("load failed")
    try:
        manager.run("b", "cpu", fail, lambda adapter: None)
    except RuntimeError:
        pass
    else:
        assert False, "Expected load failure"
    assert old.closed
    newer = use(manager)
    assert newer is not old
    manager.release()
