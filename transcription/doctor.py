"""Non-invasive runtime prerequisite inspection (never downloads model weights)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from importlib import metadata


def _module_status(module: str, distribution: str) -> dict:
    try:
        installed = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        installed = False
    try:
        version = metadata.version(distribution) if installed else None
    except metadata.PackageNotFoundError:
        version = None
    return {"installed": installed, "version": version}


def _binary_status(name: str) -> dict:
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": None, "version": None}
    try:
        result = subprocess.run([path, "-version"], capture_output=True,
                                text=True, timeout=10, check=False)
        version = (result.stdout or result.stderr).splitlines()[0] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        version = None
    return {"available": version is not None, "path": path, "version": version}


def inspect_environment() -> dict:
    """Report local dependencies without importing heavy ASR or torch frameworks."""
    return {
        "binaries": {name: _binary_status(name) for name in ("ffmpeg", "ffprobe")},
        "python_packages": {
            "torch": _module_status("torch", "torch"),
            "whisperx": _module_status("whisperx", "whisperx"),
            "funasr": _module_status("funasr", "funasr"),
        },
        "note": "Package detection is not a real-model or GPU execution test.",
    }


def main() -> None:
    print(json.dumps(inspect_environment(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
