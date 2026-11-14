"""Conservative packet-timeline checks for stream-copied video audio.

A matching packet sequence establishes *packet* timing correspondence only.
It does not prove alignment between decoded ASR sample offsets and original
video playback time; those require independent decoder/origin validation.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PacketTimelineCheck:
    verified: bool
    offset_seconds: float | None
    reason: str


def _packets(path: Path, ffprobe: str, stream_index: int | None,
             *, max_packets: int) -> tuple[list[tuple[float, float, int]], str | None]:
    cmd = [
        ffprobe, "-v", "error", "-show_packets",
        "-show_entries", "packet=stream_index,pts_time,duration_time,size,side_data_list",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False, stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        return [], f"FFprobe packet inspection failed: {exc}"
    if proc.returncode:
        return [], f"FFprobe packet inspection failed (exit {proc.returncode})"
    try:
        records = json.loads(proc.stdout)["packets"]
        if not isinstance(records, list):
            return [], "FFprobe packet list is malformed"
    except (ValueError, KeyError, TypeError):
        return [], "FFprobe returned invalid packet metadata"

    selected = [p for p in records if stream_index is None
                or p.get("stream_index") == stream_index]
    if not selected:
        return [], "No matching audio packets"
    if len(selected) > max_packets:
        return [], "Too many audio packets for bounded verification"

    packets = []
    for packet in selected:
        # Skip/discard metadata can change the relationship of decoded sample
        # time to compressed-packet PTS, even with an unchanged codec.
        if packet.get("side_data_list"):
            return [], "Packet side data requires decoder-time validation"
        try:
            pts = float(packet["pts_time"])
            duration = float(packet["duration_time"])
            size = int(packet["size"])
        except (KeyError, ValueError, TypeError):
            return [], "Incomplete packet timestamp, duration or size"
        if (not math.isfinite(pts) or not math.isfinite(duration)
                or duration <= 0 or size <= 0):
            return [], "Invalid audio packet metadata"
        packets.append((pts, duration, size))
    return packets, None


def verify_stream_copy_packet_timeline(
    source: Path, extracted: Path, original_stream_index: int,
    *, ffprobe: str = "ffprobe", max_packets: int = 100000,
    tolerance_seconds: float = 0.002,
) -> PacketTimelineCheck:
    """Verify constant packet-PTS shift of a stream copy, not video sync.

    Fail closed on stream mismatch, non-monotonic/discontinuous presentation
    timestamps, missing metadata or mismatched packet sizes/durations.
    """
    before, error = _packets(source, ffprobe, original_stream_index,
                             max_packets=max_packets)
    if error:
        return PacketTimelineCheck(False, None, "Source: " + error)
    after, error = _packets(extracted, ffprobe, None, max_packets=max_packets)
    if error:
        return PacketTimelineCheck(False, None, "Extracted: " + error)
    if len(before) != len(after):
        return PacketTimelineCheck(False, None, "Source/extracted packet counts differ")

    offset = before[0][0] - after[0][0]
    for i, (original, copied) in enumerate(zip(before, after)):
        src_pts, src_duration, src_size = original
        dst_pts, dst_duration, dst_size = copied
        if src_size != dst_size or abs(src_duration - dst_duration) > tolerance_seconds:
            return PacketTimelineCheck(False, None, f"Packet {i} content/duration mismatch")
        if abs((src_pts - dst_pts) - offset) > tolerance_seconds:
            return PacketTimelineCheck(False, None, f"Packet {i} time offset is inconsistent")
        if i:
            prev_src, prev_duration, _ = before[i - 1]
            prev_dst, prev_dst_duration, _ = after[i - 1]
            if (abs(src_pts - (prev_src + prev_duration)) > tolerance_seconds
                    or abs(dst_pts - (prev_dst + prev_dst_duration)) > tolerance_seconds):
                return PacketTimelineCheck(False, None, f"Packet {i} has a gap or overlap")
    return PacketTimelineCheck(
        True, offset,
        "Continuous matching packet timestamps; decoded-sample and video-playback "
        "alignment remain unverified",
    )
