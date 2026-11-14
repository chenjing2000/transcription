"""Packet-timeline validation must fail closed on uncertain time mapping."""
from pathlib import Path
from unittest.mock import patch

from transcription.processing.timeline import (
    PacketTimelineCheck,
    verify_stream_copy_packet_timeline,
)

SRC = [(1.0, .02, 100), (1.02, .02, 120), (1.04, .02, 110)]
COPIED = [(0.0, .02, 100), (.02, .02, 120), (.04, .02, 110)]


def run(src=SRC, extracted=COPIED, source_error=None, extracted_error=None):
    def fake_packets(path, ffprobe, stream_index, *, max_packets):
        if path.name == "source.mp4":
            return src, source_error
        return extracted, extracted_error

    with patch("transcription.processing.timeline._packets", side_effect=fake_packets):
        return verify_stream_copy_packet_timeline(
            Path("source.mp4"), Path("audio.m4a"), 1,
        )


def test_matching_continuous_packets_have_only_packet_level_verification():
    result = run()
    assert result == PacketTimelineCheck(
        True, 1.0,
        "Continuous matching packet timestamps; decoded-sample and video-playback "
        "alignment remain unverified",
    )


def test_packet_gap_or_overlap_is_unverified():
    result = run(src=[SRC[0], (1.05, .02, 120), SRC[2]])
    assert not result.verified
    assert result.offset_seconds is None


def test_drifting_packet_offsets_are_unverified():
    result = run(extracted=[COPIED[0], (.025, .02, 120), COPIED[2]])
    assert not result.verified


def test_different_packet_payload_sizes_are_unverified():
    result = run(extracted=[COPIED[0], (.02, .02, 999), COPIED[2]])
    assert not result.verified


def test_different_packet_counts_are_unverified():
    assert not run(extracted=COPIED[:2]).verified


def test_missing_packet_metadata_is_unverified():
    result = run(source_error="No packet timestamps")
    assert not result.verified and "No packet timestamps" in result.reason
