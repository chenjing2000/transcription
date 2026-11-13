import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcription.errors import TranscriptionError
from transcription.output import publish_result, _names
from transcription.processing.text_structure import split_sentences
from transcription.result import Character, TranscriptionResult, Word


def prepared(source, *, extracted=False, audio=None, warnings=()):
    return SimpleNamespace(
        source_path=source, extracted=extracted,
        audio_path=audio if audio is not None else source,
        original_stream_start=1.0, extracted_stream_start=0.0,
        candidate_video_offset=1.0,
        warnings=warnings,
    )


def sample(source, *, missing=False):
    return TranscriptionResult(
        status="partial", info="ASR in memory", engine="fake", device="cpu",
        source_path=source, text="Hello world. Again!",
        words=[
            Word("Hello", 0.1, 0.3, 0, 5, "aligned"),
            Word("world.", None if missing else 0.4, None if missing else 0.8, 6, 12,
                 "missing" if missing else "aligned"),
            Word("Again!", 1.1, 1.4, 13, 19, "aligned"),
        ],
    )


def test_text_segmentation_preserves_character_positions():
    text = "Hello world.  Again! 今天很好。"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert all(text[s.char_start:s.char_end] == s.text for s in sentences)


def test_four_files_and_consistent_json(tmp_path):
    src = tmp_path / "Hello.wav"
    src.write_bytes(b"fake")
    result = publish_result(sample(src), prepared(src))
    assert result.status == "success"
    assert result.output_dir == tmp_path / "transcript results"
    assert set(result.files) == {"timed_json", "timed_txt", "json", "txt"}
    assert all(p.is_file() for p in result.files.values())
    t = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
    plain = json.loads(result.files["json"].read_text(encoding="utf-8"))
    assert t["text"] == plain["text"] == result.text
    assert t["words"][0]["start"] == .1
    assert "words" not in plain
    assert "timeline" not in plain
    assert "start" not in plain["sentences"][0]
    assert result.files["txt"].read_text(encoding="utf-8").startswith("Hello world.")
    assert "[00:00:00.100 - 00:00:00.300] Hello" in result.files["timed_txt"].read_text(encoding="utf-8")


def test_group_numbering_does_not_overwrite_previous(tmp_path):
    src = tmp_path / "Lecture.mp3"
    src.write_bytes(b"original")
    first = publish_result(sample(src), prepared(src))
    previous = first.files["json"].read_bytes()
    second = publish_result(sample(src), prepared(src))
    assert second.files["json"].name == "Lecture (1).json"
    assert second.files["txt"].name == "Lecture (1).txt"
    assert second.files["timed_json"].name == "Lecture (1).timed.json"
    assert first.files["json"].read_bytes() == previous
    # A collision involving only one member reserves the next complete group.
    (tmp_path / "transcript results" / "Lecture (2).txt").write_text("other", encoding="utf-8")
    third = publish_result(sample(src), prepared(src))
    assert third.files["json"].name == "Lecture (3).json"


def test_video_publishes_original_audio_and_timeline(tmp_path):
    src = tmp_path / "Lecture.mp4"
    src.write_bytes(b"source")
    audio = tmp_path / "stage.m4a"
    audio.write_bytes(b"untouched-encoded-packets")
    result = publish_result(sample(src), prepared(src, extracted=True, audio=audio))
    assert result.files["audio"].name == "Lecture.m4a"
    assert result.audio_path == result.files["audio"]
    assert result.audio_path.read_bytes() == audio.read_bytes()
    timed = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
    assert timed["timeline"]["candidate_video_offset"] == 1.0
    assert timed["timeline"]["video_time_mapping_verified"] is False


def test_partial_alignment_retained(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    result = publish_result(sample(src, missing=True), prepared(src))
    assert result.status == "partial"
    timed = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
    assert timed["words"][1]["start"] is None
    assert "--:--:--.---" in result.files["timed_txt"].read_text(encoding="utf-8")


def test_alignment_inconsistency_rejected_without_outputs(tmp_path):
    src = tmp_path / "bad.wav"
    src.write_bytes(b"x")
    result = sample(src)
    result.words[0].char_start = 1
    with pytest.raises(TranscriptionError) as e:
        publish_result(result, prepared(src))
    assert e.value.code == "INVALID_ALIGNMENT"
    assert not (tmp_path / "transcript results").exists()


def test_no_speech_is_not_published(tmp_path):
    src = tmp_path / "quiet.wav"
    src.touch()
    result = TranscriptionResult("partial", "No speech", "fake", src)
    with pytest.raises(TranscriptionError) as e:
        publish_result(result, prepared(src))
    assert e.value.code == "NO_SPEECH"


def test_output_name_mapping():
    assert _names("sample", ".m4a", 2)["audio"] == "sample (2).m4a"
    assert _names("sample", None, 0)["json"] == "sample.json"


def test_rejects_nonfinite_character_timestamp(tmp_path):
    src = tmp_path / "broken.wav"
    src.write_bytes(b"audio")
    result = sample(src)
    invalid = Character("H", 0, 0.2, 1.0, "aligned")
    invalid.start = float("nan")  # Mutation after construction is still caught at publication.
    result.characters = [invalid]
    with pytest.raises(TranscriptionError) as err:
        publish_result(result, prepared(src))
    assert err.value.code == "INVALID_ALIGNMENT"
    assert not (tmp_path / "transcript results").exists()
