from transcription import transcribe


def test_missing_file():
    result = transcribe("missing.mp4", engine="whisperx_large_v3")
    assert result.status == "failed"
    assert result.error_code == "INVALID_INPUT"
    assert not result.files


def test_unsupported_file(tmp_path):
    path = tmp_path / "abc.mov"
    path.write_bytes(b"sample")
    result = transcribe(path, engine="paraformer_zh")
    assert result.error_code == "UNSUPPORTED_FORMAT"


def test_missing_optional_funasr_dependency_is_explicit(tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"sample")
    result = transcribe(path, engine="paraformer_zh")
    assert result.error_code == "DEPENDENCY_MISSING"
    assert result.status == "failed"


def test_device_and_track_validation():
    assert transcribe("a.mp4", "whisperx_large_v3", device="tpu").error_code == "INVALID_DEVICE"
    assert transcribe("a.mp4", "whisperx_large_v3", audio_track=-1).error_code == "INVALID_AUDIO_TRACK"
    assert transcribe("a.mp4", "unknown").error_code == "ENGINE_NOT_FOUND"
