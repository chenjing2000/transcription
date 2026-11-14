"""Real FFmpeg stream copies: packet mapping is verified or fails closed."""
import shutil
import subprocess

import pytest

from transcription.media import prepare_audio
from transcription.processing.timeline import verify_stream_copy_packet_timeline


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg and FFprobe are required")
def test_real_video_stream_copy_packet_diagnostics(tmp_path):
    source = tmp_path / "delayed.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.6",
        "-filter_complex", "[1:a]adelay=200|200[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    prepared = prepare_audio(source, tmp_path / "stage")
    assert prepared.extracted
    assert prepared.audio_path.is_file()
    # Verify the diagnostics return values and do not silently upgrade
    # packet agreement into proven decoder/video-playback agreement.
    assert isinstance(prepared.packet_timeline_verified, bool)
    assert prepared.packet_timeline_reason
    assert not any("video playback verified" in w.lower() for w in prepared.warnings)
    again = verify_stream_copy_packet_timeline(source, prepared.audio_path, prepared.stream.index)
    assert again.verified == prepared.packet_timeline_verified
    assert again.offset_seconds == prepared.packet_timeline_offset


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg and FFprobe are required")
def test_real_audio_copy_with_nonzero_video_origin_is_not_claimed_verified(tmp_path):
    source = tmp_path / "shifted.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x180:r=25:d=1",
        "-itsoffset", "0.35",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "mpeg4", "-c:a", "aac", str(source),
    ], check=True)
    prepared = prepare_audio(source, tmp_path / "stage")
    assert prepared.audio_path.exists()
    # Packet-level check does not yet represent a verified *video* timeline.
    assert "constant-offset mapping must be verified" in " ".join(prepared.warnings)
