"""Staged transcript publishing with collision-safe grouped numbering.

An exclusive per-output-directory lock serializes cooperative publisher processes.
Each final path is opened with 'xb' to prevent overwrites, even on collision.
If publishing raises an ordinary exception, all files created by the attempt
are removed. Process crashes cannot guarantee all-or-none filesystem visibility.
"""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from .errors import TranscriptionError
from .media import PreparedAudio
from .processing.text_structure import structure_result
from .exporters.json_exporter import serialize_json
from .exporters.txt_exporter import plain_txt, timed_txt
from .result import TranscriptionResult


@contextmanager
def _directory_lock(folder: Path) -> Iterator[None]:
    lock = folder / ".transcription-publish.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise TranscriptionError(
            "OUTPUT_BUSY", "Another publication is running or a stale publication lock exists: " + str(lock),
        ) from exc
    try:
        yield
    finally:
        lock.rmdir()


def _names(stem: str, extension: str | None, number: int) -> dict[str, str]:
    base = stem + (f" ({number})" if number else "")
    names = {
        "timed_json": base + ".timed.json",
        "timed_txt": base + ".timed.txt",
        "json": base + ".json",
        "txt": base + ".txt",
    }
    if extension is not None:
        names["audio"] = base + extension
    return names


def _unique_names(folder: Path, stem: str, ext: str | None) -> dict[str, str]:
    for number in range(100000):
        names = _names(stem, ext, number)
        if all(not (folder / filename).exists() for filename in names.values()):
            return names
    raise TranscriptionError("OUTPUT_FAILED", "No available output suffix found")


def _safe_copy(src: Path, dest: Path) -> None:
    with src.open("rb") as input_file, dest.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)


def publish_result(result: TranscriptionResult, prepared: PreparedAudio) -> TranscriptionResult:
    """Generate and publish four files plus original-codec audio for video input."""
    structure_result(result)
    if result.status == "failed":
        raise TranscriptionError("TRANSCRIPTION_FAILED", "Cannot publish a failed recognition result")
    result.status = "partial" if result.warnings else "success"
    result.info = ("Published with warnings" if result.warnings else "Transcription and export completed")
    folder = result.source_path.parent / "transcript results"
    try:
        folder.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".transcription-stage-", dir=folder) as work:
            stage = Path(work)
            staging_files = {
                "timed_json": stage / "timed.json",
                "timed_txt": stage / "timed.txt",
                "json": stage / "plain.json",
                "txt": stage / "plain.txt",
            }
            staging_files["timed_json"].write_text(
                serialize_json(result, prepared, timed=True), encoding="utf-8")
            staging_files["timed_txt"].write_text(timed_txt(result), encoding="utf-8")
            staging_files["json"].write_text(
                serialize_json(result, prepared, timed=False), encoding="utf-8")
            staging_files["txt"].write_text(plain_txt(result), encoding="utf-8")
            if prepared.extracted:
                staging_files["audio"] = stage / ("audio" + prepared.audio_path.suffix)
                shutil.copyfile(prepared.audio_path, staging_files["audio"])

            for file in staging_files.values():
                if not file.is_file() or file.stat().st_size == 0:
                    raise TranscriptionError("OUTPUT_FAILED", "Missing or empty staging artifact: " + str(file))

            with _directory_lock(folder):
                filenames = _unique_names(
                    folder, result.source_path.stem,
                    prepared.audio_path.suffix if prepared.extracted else None,
                )
                created: list[Path] = []
                try:
                    for key, name in filenames.items():
                        destination = folder / name
                        try:
                            _safe_copy(staging_files[key], destination)
                        except Exception:
                            # The exclusive-open file may have been partially copied.
                            if destination.exists() and destination not in created:
                                # Do not delete an independently created file on collision.
                                # If this task created it, the helper adds it to created
                                # before subsequent operations.
                                pass
                            raise
                        created.append(destination)
                except Exception:
                    for path in reversed(created):
                        path.unlink(missing_ok=True)
                    raise
        result.output_dir = folder
        result.files = {key: folder / name for key, name in filenames.items()}
        result.audio_path = result.files["audio"] if prepared.extracted else result.source_path
        return result
    except TranscriptionError:
        raise
    except OSError as exc:
        raise TranscriptionError("OUTPUT_FAILED", f"Failed to publish transcript files: {exc}") from exc
