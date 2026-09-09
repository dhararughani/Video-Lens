"""Pointer/cursor intelligence tests: contract validation, ground-truth
motion detection, false-positive resistance, tracking, and frame-extractor
integration.

Generates synthetic media with ffmpeg (no committed binaries). Run:
    python tests/test_pointer.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.frame_extractor import FrameExtractor
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import (
    detect_pointer, detect_pointer_at, detect_pointer_for_frames, track_pointer,
)
from core.contracts import Frame, PointerEvent


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")


def _moving_cursor_video(path: str, duration: float = 5.0, size: str = "320x240"):
    """12x12 white square on gray, center at (20+40t+6, 20+20t+6)."""
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=gray:size={size}:rate=10:duration={duration}",
        "-f", "lavfi", "-i", f"color=c=white:size=12x12:rate=10:duration={duration}",
        "-filter_complex", "[0][1]overlay=x='20+t*40':y='20+t*20':shortest=1",
        "-pix_fmt", "yuv420p", path,
    ])


def _static_video(path: str, duration: float = 3.0, size: str = "320x240"):
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=gray:size={size}:rate=10:duration={duration}",
        "-pix_fmt", "yuv420p", path,
    ])


def _blinking_ui_video(path: str, duration: float = 3.0, size: str = "320x240"):
    """Large (80x30) fixed-position element toggling every frame -- too big
    to be a cursor; a naive frame-differ would otherwise flag it as motion."""
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=gray:size={size}:rate=10:duration={duration}",
        "-f", "lavfi", "-i", f"color=c=white:size=80x30:rate=10:duration={duration}",
        "-filter_complex", "[0][1]overlay=x=150:y=100:enable='mod(floor(t*10)\\,2)'",
        "-pix_fmt", "yuv420p", path,
    ])


def _scene_transition_video(path: str, size: str = "320x240"):
    """Full-frame color swap -- scene-wide motion, not a pointer."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4")
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=red:size={size}:rate=10:duration=1.5",
              "-pix_fmt", "yuv420p", a])
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:size={size}:rate=10:duration=1.5",
              "-pix_fmt", "yuv420p", b])
        listfile = os.path.join(tmp, "list.txt")
        with open(listfile, "w") as f:
            f.write(f"file '{a}'\nfile '{b}'\n")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
              "-pix_fmt", "yuv420p", path])


# ------------------------------- contract --------------------------------

def test_pointer_event_status_validated():
    PointerEvent(timestamp_sec=1.0, status="not_detected")
    try:
        PointerEvent(timestamp_sec=1.0, status="maybe")
        assert False, "invalid status should be rejected"
    except ValueError:
        pass


def test_pointer_event_requires_xy_unless_not_detected():
    try:
        PointerEvent(timestamp_sec=1.0, status="detected")
        assert False, "'detected' without x/y should be rejected"
    except ValueError:
        pass
    PointerEvent(timestamp_sec=1.0, status="uncertain", x=1, y=1)  # ok


def test_pointer_event_normalized_coordinates():
    e = PointerEvent(timestamp_sec=1.0, status="detected", x=960, y=540,
                      frame_width=1920, frame_height=1080, confidence=0.9)
    assert e.normalized_x == 0.5
    assert e.normalized_y == 0.5
    not_detected = PointerEvent(timestamp_sec=1.0, status="not_detected")
    assert not_detected.normalized_x is None
    assert not_detected.normalized_y is None


# ----------------------------- ground truth -------------------------------

def test_detect_pointer_for_frames_matches_known_trajectory():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _moving_cursor_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        frames = ext.extract_window(video, 0.0, 4.9, 0.5)
        events = detect_pointer_for_frames(frames)

        assert len(events) == len(frames)
        assert events[0].status == "not_detected"  # no triple to diff against
        assert events[-1].status == "not_detected"

        detected = [e for e in events if e.status == "detected"]
        assert len(detected) >= 6, "should confidently detect most of the trajectory"
        for e in detected:
            expected_x = 20 + 40 * e.timestamp_sec + 6
            expected_y = 20 + 20 * e.timestamp_sec + 6
            assert abs(e.x - expected_x) <= 2
            assert abs(e.y - expected_y) <= 2


def test_detect_pointer_two_frame_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _moving_cursor_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        a = ext.get_frame(video, 1.0)
        b = ext.get_frame(video, 1.5)
        e = detect_pointer(b, prev_frame=a)
        assert e.status in ("detected", "uncertain")
        assert e.x is not None and e.y is not None


def test_detect_pointer_without_prev_frame_is_honest():
    f = Frame(timestamp_sec=1.0, path="whatever.jpg", width=100, height=100)
    e = detect_pointer(f, prev_frame=None)
    assert e.status == "not_detected"
    assert e.x is None and e.y is None


# ------------------------- false-positive resistance -----------------------

def test_no_motion_produces_no_detections():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _static_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        frames = ext.extract_window(video, 0.0, 2.9, 0.3)
        events = detect_pointer_for_frames(frames)
        assert all(e.status == "not_detected" for e in events)


def test_large_blinking_ui_element_is_not_mistaken_for_pointer():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _blinking_ui_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        frames = ext.extract_window(video, 0.0, 2.9, 0.3)
        events = detect_pointer_for_frames(frames)
        assert not any(e.status == "detected" for e in events), \
            "an 80x30 fixed-position blinking block should fail the cursor size filter"


def test_scene_transition_is_not_mistaken_for_pointer():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _scene_transition_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        frames = ext.extract_window(video, 0.0, 2.9, 0.3)
        events = detect_pointer_for_frames(frames)
        assert not any(e.status == "detected" for e in events), \
            "a full-frame color swap should be vetoed as scene-wide motion, not tracked as a pointer"


# --------------------------------- tracking --------------------------------

def test_track_pointer_orders_events_and_reuses_frame_extractor():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _moving_cursor_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        track = track_pointer(video, 0.0, 4.9, ext, interval_sec=0.5)
        assert track.start_sec == 0.0 and track.end_sec == 4.9
        timestamps = [e.timestamp_sec for e in track.events]
        assert timestamps == sorted(timestamps)
        assert 0.0 < track.confidence <= 1.0


def test_detect_pointer_at_single_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _moving_cursor_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        e = detect_pointer_at(video, 2.0, ext, step_sec=0.5)
        assert e.status == "detected"
        assert abs(e.x - (20 + 40 * 2.0 + 6)) <= 2
        assert abs(e.y - (20 + 20 * 2.0 + 6)) <= 2


def test_empty_and_single_frame_lists_are_handled():
    assert detect_pointer_for_frames([]) == []
    f = Frame(timestamp_sec=0.0, path="x.jpg", width=10, height=10)
    events = detect_pointer_for_frames([f])
    assert len(events) == 1 and events[0].status == "not_detected"


if __name__ == "__main__":
    test_pointer_event_status_validated()
    test_pointer_event_requires_xy_unless_not_detected()
    test_pointer_event_normalized_coordinates()
    test_detect_pointer_for_frames_matches_known_trajectory()
    test_detect_pointer_two_frame_fallback()
    test_detect_pointer_without_prev_frame_is_honest()
    test_no_motion_produces_no_detections()
    test_large_blinking_ui_element_is_not_mistaken_for_pointer()
    test_scene_transition_is_not_mistaken_for_pointer()
    test_track_pointer_orders_events_and_reuses_frame_extractor()
    test_detect_pointer_at_single_timestamp()
    test_empty_and_single_frame_lists_are_handled()
    print("All pointer tests passed.")
