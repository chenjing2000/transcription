"""Dependency diagnostics should work with or without ML frameworks."""
from transcription.doctor import inspect_environment, _binary_status, _module_status


def test_inspect_environment_is_serializable():
    import json
    inspection = inspect_environment()
    assert set(inspection["binaries"]) == {"ffmpeg", "ffprobe"}
    assert set(inspection["python_packages"]) == {"torch", "whisperx", "funasr"}
    assert json.loads(json.dumps(inspection)) == inspection


def test_missing_binary(monkeypatch):
    import transcription.doctor as doctor
    monkeypatch.setattr(doctor.shutil, "which", lambda p: None)
    assert _binary_status("missing") == {"available": False, "path": None, "version": None}


def test_missing_module(monkeypatch):
    import transcription.doctor as doctor
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda p: None)
    assert _module_status("unknown", "unknown") == {"installed": False, "version": None}
