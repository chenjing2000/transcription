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


def test_two_outputs_and_word_level_json(tmp_path):
    src = tmp_path / "Hello.wav"
    src.write_bytes(b"fake")
    result = publish_result(sample(src), prepared(src))
    assert result.status == "success"
    assert result.output_dir == tmp_path / "transcript results"
    assert set(result.files) == {"timed_json", "txt"}
    assert all(p.is_file() for p in result.files.values())
    payload = json.loads(result.files["timed_json"].read_text(encoding="utf-8"))
    assert payload["language"] == "en"
    assert payload["transcript_studio"] == {"format": "transcript-studio", "schemaVersion": 2}
    assert [w["word"] for w in payload["word_segments"]] == ["Hello", "world.", "Again!"]
    assert [w["start"] for w in payload["word_segments"]] == [0.1, 0.4, 1.1]
    assert sum(len(seg["words"]) for seg in payload["segments"]) == 3
    assert payload["segments"][0]["text"] == "Hello world."
    assert result.files["txt"].read_text(encoding="utf-8").startswith("Hello world.")
    assert not (result.output_dir / "Hello.json").exists()
    assert not (result.output_dir / "Hello.timed.txt").exists()


def test_group_numbering_does_not_overwrite_previous(tmp_path):
    src = tmp_path / "Lecture.mp3"
    src.write_bytes(b"original")
    first = publish_result(sample(src), prepared(src))
    previous = first.files["timed_json"].read_bytes()
    second = publish_result(sample(src), prepared(src))
    assert second.files["timed_json"].name == "Lecture (1).timed.json"
    assert second.files["txt"].name == "Lecture (1).txt"
    assert second.files["timed_json"].name == "Lecture (1).timed.json"
    assert first.files["timed_json"].read_bytes() == previous
    # A collision involving only one member reserves the next complete group.
    (tmp_path / "transcript results" / "Lecture (2).txt").write_text("other", encoding="utf-8")
    third = publish_result(sample(src), prepared(src))
    assert third.files["timed_json"].name == "Lecture (3).timed.json"


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
    assert timed["word_segments"][1]["start"] is None
    assert "score" not in timed["word_segments"][1]


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
    assert _names("sample", None, 0)["timed_json"] == "sample.timed.json"


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

def test_real_score_is_preserved_without_guessing_missing_values(tmp_path):
    src = tmp_path / "score.wav"
    src.write_bytes(b"x")
    item = sample(src)
    item.words[0].score = 0.816
    output = publish_result(item, prepared(src))
    data = json.loads(output.files["timed_json"].read_text(encoding="utf-8"))
    assert data["word_segments"][0]["score"] == 0.816
    assert "score" not in data["word_segments"][1]
    assert data["segments"][0]["words"][0]["score"] == 0.816


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        Word("hi", 0.1, 0.2, 0, 2, "aligned", 1.2)
