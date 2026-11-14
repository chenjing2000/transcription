"""English word and Mandarin sentence timing contracts."""
import json
from types import SimpleNamespace

from transcription.output import publish_result
from transcription.result import TranscriptionResult, Word


def prepared(src, video=False, audio=None):
    return SimpleNamespace(source_path=src, audio_path=audio or src,
        extracted=video, original_stream_start=1.0,
        extracted_stream_start=0.0, candidate_video_offset=1.0,
        warnings=())


def zh(src, missing=False):
    return TranscriptionResult(
        "partial", "recognized", "paraformer_zh", src,
        text="你好。我们学习！",
        words=[
            Word("你", .1, .2, 0, 1, "aligned"),
            Word("好", .2, .4, 1, 2, "aligned"),
            Word("我", None if missing else .8,
                 None if missing else .9, 3, 4,
                 "missing" if missing else "aligned"),
            Word("们", .9, 1.0, 4, 5, "aligned"),
            Word("学习", 1.0, 1.4, 5, 7, "aligned"),
        ],
    )


def test_zh_audio_sentence_timing_only(tmp_path):
    src = tmp_path / "zh.wav"
    src.write_bytes(b"audio")
    result = publish_result(zh(src), prepared(src))
    data = json.loads(result.files["timed_json"].read_text(encoding="utf8"))
    assert result.status == "success"
    assert set(data) == {"schema_version", "language", "timeline", "segments"}
    assert data["language"] == "zh"
    assert data["timeline"]["reference"] == "source_audio_playback"
    assert [s["text"] for s in data["segments"]] == ["你好。", "我们学习！"]
    assert [(s["start"], s["end"]) for s in data["segments"]] == [(.1, .4), (.8, 1.4)]
    assert all(s["alignment_status"] == "aligned" for s in data["segments"])
    assert "word_segments" not in data


def test_zh_partial_sentence_is_missing_without_invented_timing(tmp_path):
    src = tmp_path / "zh.wav"
    src.write_bytes(b"audio")
    result = publish_result(zh(src, missing=True), prepared(src))
    data = json.loads(result.files["timed_json"].read_text(encoding="utf8"))
    assert result.status == "partial"
    assert data["segments"][0]["alignment_status"] == "aligned"
    assert data["segments"][1] == {
        "text": "我们学习！", "start": None, "end": None,
        "alignment_status": "missing",
    }


def test_zh_video_three_files_and_no_unverified_timestamps(tmp_path):
    src = tmp_path / "Lecture.mp4"
    src.write_bytes(b"video")
    audio = tmp_path / "track.m4a"
    audio.write_bytes(b"stream-copy")
    result = publish_result(zh(src), prepared(src, video=True, audio=audio))
    data = json.loads(result.files["timed_json"].read_text(encoding="utf8"))
    assert set(result.files) == {"audio", "txt", "timed_json"}
    assert result.status == "partial"
    assert data["timeline"]["reference"] == "source_video_playback"
    assert data["timeline"]["mapping_verified"] is False
    assert all(s["start"] is None and s["end"] is None for s in data["segments"])
    assert all(s["alignment_status"] == "missing" for s in data["segments"])
    assert "word_segments" not in data
    assert result.files["audio"].read_bytes() == b"stream-copy"


def test_en_partially_aligned_sentence_keeps_real_word_time(tmp_path):
    src = tmp_path / "talk.wav"
    src.write_bytes(b"audio")
    result = TranscriptionResult(
        "partial", "recognized", "whisperx_large_v3", src,
        text="One two.",
        words=[
            Word("One", .2, .4, 0, 3, "aligned"),
            Word("two.", None, None, 4, 8, "missing"),
        ],
    )
    output = publish_result(result, prepared(src))
    data = json.loads(output.files["timed_json"].read_text(encoding="utf8"))
    assert data["segments"][0]["alignment_status"] == "partial"
    assert data["segments"][0]["start"] == .2
    assert data["segments"][0]["end"] == .4
    assert data["word_segments"][1]["start"] is None


def test_zh_inconsistent_word_times_do_not_become_sentence_times(tmp_path):
    src = tmp_path / "zh.wav"
    src.write_bytes(b"audio")
    result = zh(src)
    result.words[1].start = .15
    result.words[1].end = .19
    output = publish_result(result, prepared(src))
    data = json.loads(output.files["timed_json"].read_text(encoding="utf8"))
    assert output.status == "partial"
    assert data["segments"][0]["alignment_status"] == "missing"
    assert data["segments"][0]["start"] is None
