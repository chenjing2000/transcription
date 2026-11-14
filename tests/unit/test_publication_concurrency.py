"""Publication locking, racing writers, and failure recovery regressions."""
from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

import transcription.output as publisher
from transcription.errors import TranscriptionError
from transcription.result import TranscriptionResult, Word


def _result(source: Path) -> TranscriptionResult:
    return TranscriptionResult(
        status="partial", info="recognized", engine="test", device="cpu",
        source_path=source, text="Hello.",
        words=[Word("Hello.", 0.1, 0.4, 0, 6, "aligned")],
    )


def _audio(source: Path):
    return SimpleNamespace(
        source_path=source, audio_path=source, extracted=False,
        original_stream_start=0.0, extracted_stream_start=0.0,
        candidate_video_offset=0.0,
    )


def test_busy_publisher_never_damages_existing_group(tmp_path):
    source = tmp_path / "clip.wav"
    source.write_bytes(b"original")
    first = publisher.publish_result(_result(source), _audio(source))
    prior = {key: path.read_bytes() for key, path in first.files.items()}
    folder = first.output_dir
    occupied = Event()
    release = Event()

    def hold_lock():
        with publisher._directory_lock(folder):
            occupied.set()
            assert release.wait(5), "Timed out waiting for release"

    worker = Thread(target=hold_lock)
    worker.start()
    try:
        assert occupied.wait(5), "Publishing lock not acquired"
        with pytest.raises(TranscriptionError) as error:
            publisher.publish_result(_result(source), _audio(source))
        assert error.value.code == "OUTPUT_BUSY"
        assert {key: path.read_bytes() for key, path in first.files.items()} == prior
        assert set(folder.glob("clip (1).*")) == set()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    second = publisher.publish_result(_result(source), _audio(source))
    assert second.files["timed_json"].name == "clip (1).timed.json"


def test_collision_after_suffix_check_preserves_foreign_file(monkeypatch, tmp_path):
    source = tmp_path / "clip.wav"
    source.write_bytes(b"original")
    folder = tmp_path / "transcript results"
    original = publisher._unique_names
    foreign = folder / "clip.timed.json"

    def concurrent_creation(folder_arg, stem, ext):
        result = original(folder_arg, stem, ext)
        foreign.write_text("other process", encoding="utf-8")
        return result

    monkeypatch.setattr(publisher, "_unique_names", concurrent_creation)
    with pytest.raises(TranscriptionError) as error:
        publisher.publish_result(_result(source), _audio(source))
    assert error.value.code == "OUTPUT_FAILED"
    assert foreign.read_text(encoding="utf-8") == "other process"
    assert sorted(path.name for path in folder.iterdir()) == ["clip.timed.json"]
