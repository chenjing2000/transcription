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
from .exporters.json_exporter import serialize_json, has_missing_chinese_sentence_timing
from .exporters.txt_exporter import plain_txt
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
    # Opening with xb is the ownership boundary: only clean up files opened
    # by this attempt, never an existing path that caused FileExistsError.
    with src.open("rb") as input_file:
        with dest.open("xb") as output_file:
            try:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            except BaseException:
                output_file.close()
                dest.unlink(missing_ok=True)
                raise


def _stage_files(stage: Path, result: TranscriptionResult,
                 prepared: PreparedAudio) -> dict[str, Path]:
    """Render each payload once, then verify all required staged artifacts."""
    staged = {
        "timed_json": stage / "timed.json",
        "txt": stage / "plain.txt",
    }
    staged["timed_json"].write_text(
        serialize_json(result, prepared, timed=True), encoding="utf-8")
    staged["txt"].write_text(plain_txt(result), encoding="utf-8")
    if prepared.extracted:
        staged["audio"] = stage / ("audio" + prepared.audio_path.suffix)
        shutil.copyfile(prepared.audio_path, staged["audio"])
    for artifact in staged.values():
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise TranscriptionError("OUTPUT_FAILED", f"Missing or empty staged artifact: {artifact}")
    return staged


def _publish_files(folder: Path, stem: str, audio_extension: str | None,
                   staged: dict[str, Path]) -> dict[str, Path]:
    """Reserve a shared suffix, exclusively create all files, rollback on errors."""
    with _directory_lock(folder):
        names = _unique_names(folder, stem, audio_extension)
        created: list[Path] = []
        try:
            for key, filename in names.items():
                target = folder / filename
                _safe_copy(staged[key], target)
                created.append(target)
        except Exception:
            for target in reversed(created):
                target.unlink(missing_ok=True)
            raise
        return {key: folder / filename for key, filename in names.items()}


def publish_result(result: TranscriptionResult, prepared: PreparedAudio) -> TranscriptionResult:
    """Publish word-timed JSON, plain TXT, and original-codec video audio."""
    if result.status == "failed":
        raise TranscriptionError("TRANSCRIPTION_FAILED", "Cannot publish a failed recognition result")
    structure_result(result)
    if prepared.extracted and getattr(prepared, 'video_playback_offset', None) is None and not any('video playback timeline' in w.lower() for w in result.warnings):
        result.warnings.append('Original video playback timeline is not yet verified; all exported timestamps are marked missing')
    if has_missing_chinese_sentence_timing(result, prepared) and not any(
        'Chinese sentences lack verified timestamps' in w for w in result.warnings
    ):
        result.warnings.append('Some Chinese sentences lack verified timestamps')
    result.status = "partial" if result.warnings else "success"
    result.info = "Published with warnings" if result.warnings else "Transcription and export completed"
    folder = result.source_path.parent / "transcript results"
    try:
        folder.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=".transcription-stage-", dir=folder) as work:
            staged = _stage_files(Path(work), result, prepared)
            paths = _publish_files(
                folder, result.source_path.stem,
                prepared.audio_path.suffix if prepared.extracted else None, staged,
            )
        result.output_dir = folder
        result.files = paths
        result.audio_path = paths["audio"] if prepared.extracted else result.source_path
        return result
    except TranscriptionError:
        raise
    except OSError as exc:
        raise TranscriptionError("OUTPUT_FAILED", f"Failed to publish transcript files: {exc}") from exc
