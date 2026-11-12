import json
import pytest

from transcription.errors import TranscriptionError
from transcription.media import COPY_EXTENSIONS, prepare_audio, probe_media


def _response(*, codec="aac", start_time="0.5", audio=True):
    return json.dumps({
        "streams": ([{"index": 1, "codec_type": "audio", "codec_name": codec,
                      "sample_rate": "48000", "channels": 2,
                      "bit_rate": "192000", "start_time": start_time}]
                    if audio else [{"index": 0, "codec_type": "video"}]),
        "format": {"duration": "10.0"},
    })


def test_probe_audio_metadata(monkeypatch, tmp_path):
    source = tmp_path / "lecture.mp4"
    source.touch()
    monkeypatch.setattr("transcription.media._run", lambda *a: _response())
    info = probe_media(source, ffprobe="ffprobe")
    assert info.is_video
    assert info.audio_streams[0].index == 1
    assert info.audio_streams[0].sample_rate == 48000
    assert info.audio_streams[0].start_time == 0.5
    assert info.duration == 10.0


def test_no_audio_stream(monkeypatch, tmp_path):
    source = tmp_path / "silent.mkv"
    source.touch()
    monkeypatch.setattr("transcription.media._run", lambda *a: _response(audio=False))
    with pytest.raises(TranscriptionError, match="No audio stream"):
        probe_media(source, ffprobe="ffprobe")


def test_audio_input_not_copied(monkeypatch, tmp_path):
    source = tmp_path / "original.mp3"
    source.touch()
    monkeypatch.setattr("transcription.media._run", lambda *a: _response(codec="mp3"))
    info = prepare_audio(source, tmp_path / "staging", ffprobe="ffprobe")
    assert info.audio_path == source
    assert info.extracted is False
    assert not (tmp_path / "staging").exists()


def test_invalid_track(monkeypatch, tmp_path):
    source = tmp_path / "x.mp4"
    source.touch()
    monkeypatch.setattr("transcription.media._run", lambda *a: _response())
    with pytest.raises(TranscriptionError) as error:
        prepare_audio(source, tmp_path / "s", audio_track=1, ffprobe="ffprobe")
    assert error.value.code == "AUDIO_TRACK_NOT_FOUND"


def test_unsupported_codec_fails_before_copy(monkeypatch, tmp_path):
    source = tmp_path / "x.mkv"
    source.touch()
    monkeypatch.setattr("transcription.media._run", lambda *a: _response(codec="unknown_codec"))
    with pytest.raises(TranscriptionError) as error:
        prepare_audio(source, tmp_path / "s", ffprobe="ffprobe")
    assert error.value.code == "UNSUPPORTED_AUDIO_CODEC"


def test_supported_codec_mapping():
    assert COPY_EXTENSIONS["aac"] == ".m4a"
    assert COPY_EXTENSIONS["opus"] == ".opus"
    assert COPY_EXTENSIONS["pcm_s16le"] == ".wav"


def test_probing_input_checks(tmp_path):
    with pytest.raises(TranscriptionError) as error:
        probe_media(tmp_path / "missing.wav", ffprobe="ffprobe")
    assert error.value.code == "INVALID_INPUT"
    source = tmp_path / "x.mov"
    source.touch()
    with pytest.raises(TranscriptionError) as error:
        probe_media(source, ffprobe="ffprobe")
    assert error.value.code == "UNSUPPORTED_FORMAT"


def test_extracted_media_validation_allows_m4a_and_flac_only_internally(monkeypatch, tmp_path):
    monkeypatch.setattr("transcription.media._run", lambda *a: _response())
    for ext in (".m4a", ".flac"):
        path = tmp_path / ("sample" + ext)
        path.touch()
        with pytest.raises(TranscriptionError) as err:
            probe_media(path, ffprobe="ffprobe")
        assert err.value.code == "UNSUPPORTED_FORMAT"
        info = probe_media(path, ffprobe="ffprobe", allow_extracted=True)
        assert info.audio_streams[0].codec == "aac"
