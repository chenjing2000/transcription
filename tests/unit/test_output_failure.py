from types import SimpleNamespace
from pathlib import Path

import pytest

import transcription.output as output
from transcription.errors import TranscriptionError
from transcription.result import TranscriptionResult, Word


def _media(source):
    return SimpleNamespace(
        source_path=source, audio_path=source, extracted=False,
        original_stream_start=0.0, extracted_stream_start=0.0,
        candidate_video_offset=0.0,
    )


def test_mid_publish_failure_rolls_back_all_its_files(monkeypatch, tmp_path):
    src = tmp_path / "source.wav"
    src.write_bytes(b"source")
    result = TranscriptionResult(
        "partial", "", "mock", src, text="Hello.",
        words=[Word("Hello.", 0.0, 0.3, 0, 6, "aligned")],
    )
    original = output._safe_copy
    count = 0
    def fail_on_second(src_path, dest_path):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated disk error")
        return original(src_path, dest_path)
    monkeypatch.setattr(output, "_safe_copy", fail_on_second)
    with pytest.raises(TranscriptionError) as err:
        output.publish_result(result, _media(src))
    assert err.value.code == "OUTPUT_FAILED"
    folder = tmp_path / "transcript results"
    assert list(folder.iterdir()) == []


def test_output_lock_does_not_delete_existing_files(tmp_path):
    folder = tmp_path / "transcript results"
    folder.mkdir()
    (folder / "source.json").write_text("keep", encoding="utf8")
    with output._directory_lock(folder):
        assert (folder / ".transcription-publish.lock").is_dir()
        with pytest.raises(TranscriptionError) as e:
            with output._directory_lock(folder):
                pass
        assert e.value.code == "OUTPUT_BUSY"
    assert (folder / "source.json").read_text(encoding="utf8") == "keep"


def test_failed_result_cannot_be_published(tmp_path):
    src = tmp_path / "source.wav"
    src.touch()
    obj = TranscriptionResult("failed", "invalid", "mock", src, text="Hello.")
    with pytest.raises(TranscriptionError):
        output.publish_result(obj, _media(src))
