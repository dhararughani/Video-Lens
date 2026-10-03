"""P0-C cursor intelligence v2 tests: what a PointerTrack establishes about
pointer movement -- direction, speed, confidence, honest uncertainty -- and the
rule that "stationary" is never inferred from an absence of motion.

Hand-built tracks use a 300x400 frame, whose diagonal is exactly 500px, so
every expected speed below is exact. Run: python tests/test_cursor_intelligence.py
"""
from __future__ import annotations

import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.frame_extractor import FrameExtractor
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import track_pointer
from core.contracts import (
    CURSOR_MOTION_STATES, KNOWLEDGE_SCHEMA_VERSION, CursorSegment, Evidence, PointerEvent, PointerTrack,
)
from core.cursor_intelligence import analyze_track
from video_lens import analyze_cursor_motion

W, H, DIAG = 300, 400, 500.0


def _ev(t, x=None, y=None, status="detected", conf=0.9, w=W, h=H) -> PointerEvent:
    if status == "not_detected":
        return PointerEvent(timestamp_sec=t, status=status, frame_width=w, frame_height=h)
    return PointerEvent(timestamp_sec=t, status=status, x=x, y=y, frame_width=w, frame_height=h,
                        confidence=conf)


def _track(events, start=None, end=None) -> PointerTrack:
    start = events[0].timestamp_sec if start is None else start
    end = events[-1].timestamp_sec if end is None else end
    return PointerTrack(start_sec=start, end_sec=end, events=list(events))


def _line(x0, y0, dx, dy, n=5, dt=0.5, t0=0.0):
    return [_ev(t0 + i * dt, x0 + i * dx, y0 + i * dy) for i in range(n)]


def _only(segments) -> CursorSegment:
    assert len(segments) == 1, segments
    return segments[0]


# ----------------------------- 1-3. direction & speed -----------------------------

def test_horizontal_rightward_movement():
    s = _only(analyze_track(_track(_line(100, 200, 30, 0))))
    assert s.motion_state == "moving" and s.direction_deg == 0.0
    assert s.mean_speed_norm == round(30 / DIAG / 0.5, 4) == 0.12
    assert (s.start_sec, s.end_sec) == (0.0, 2.0)


def test_horizontal_leftward_movement():
    assert _only(analyze_track(_track(_line(250, 200, -30, 0)))).direction_deg == 180.0


def test_vertical_movement_uses_image_convention_down_is_90():
    assert _only(analyze_track(_track(_line(150, 50, 0, 40)))).direction_deg == 90.0
    assert _only(analyze_track(_track(_line(150, 350, 0, -40)))).direction_deg == 270.0


def test_diagonal_angle_is_deterministic_and_wraps_into_0_360():
    assert _only(analyze_track(_track(_line(50, 50, 30, 30)))).direction_deg == 45.0
    s = _only(analyze_track(_track(_line(50, 300, 40, -30))))
    assert s.direction_deg == round(math.degrees(math.atan2(-30, 40)) % 360, 1) == 323.1
    assert s.mean_speed_norm == round(50 / DIAG / 0.5, 4)  # 3-4-5 triangle: 50px per step


def test_direction_uses_pixel_geometry_not_per_axis_normalization():
    # On a 300x400 frame a visually 45-degree move is 0.1 of the width but only
    # 0.075 of the height; per-axis normalized deltas would report ~36.9 degrees.
    s = _only(analyze_track(_track(_line(50, 50, 30, 30))))
    assert s.direction_deg == 45.0
    assert round(math.degrees(math.atan2(30 / H, 30 / W)), 1) == 36.9  # the distortion avoided


# ----------------------------- 4. zero displacement is never "stationary" -----------------------------

def test_zero_displacement_is_uncertain_never_stationary():
    events = [_ev(i * 0.5, 150, 200) for i in range(6)]  # confident detections, never displaced
    s = _only(analyze_track(_track(events)))
    assert s.motion_state == "uncertain" and s.motion_state != "stationary"
    assert s.basis == "displacement_within_jitter"
    assert s.direction_deg is None and s.mean_speed_norm is None and s.confidence == 0.0


def test_jitter_sized_moves_are_not_movement():
    events = [_ev(i * 0.5, 150 + (i % 2) * 5, 200) for i in range(6)]  # 5px = 0.01 of the diagonal
    assert _only(analyze_track(_track(events))).basis == "displacement_within_jitter"


def test_stationary_is_not_a_representable_state():
    assert "stationary" not in CURSOR_MOTION_STATES
    try:
        CursorSegment(start_sec=0.0, end_sec=1.0, motion_state="stationary", basis="x")
        assert False, "'stationary' must be rejected by the contract itself"
    except ValueError:
        pass


def test_no_track_shape_ever_yields_stationary_or_unsupported_motion():
    rng = random.Random(11)
    for _ in range(300):
        t, events = 0.0, []
        for _ in range(rng.randint(0, 12)):
            t += rng.choice([0.0, 0.25, 0.5, 0.5, 1.0, 3.0])
            status = rng.choice(["detected", "detected", "uncertain", "not_detected"])
            events.append(_ev(t, rng.randint(0, W), rng.randint(0, H), status=status))
        segments = analyze_track(_track(events, start=0.0, end=t + rng.choice([0.0, 0.7])))
        for s in segments:
            assert s.motion_state in ("moving", "uncertain")
            if s.motion_state == "uncertain":
                assert s.direction_deg is None and s.mean_speed_norm is None and s.confidence == 0.0


# ----------------------------- 5. sparse observations -----------------------------

def test_detections_separated_by_misses_are_uncertain():
    events = [_ev(0.0, 50, 50), _ev(0.5, status="not_detected"), _ev(1.0, 150, 50),
              _ev(1.5, status="not_detected"), _ev(2.0, 250, 50)]
    s = _only(analyze_track(_track(events)))
    assert s.motion_state == "uncertain" and s.basis == "pointer_not_observed"


def test_ambiguous_positions_do_not_establish_movement():
    events = [_ev(0.0, 50, 50), _ev(0.5, 100, 50, status="uncertain", conf=0.6), _ev(1.0, 150, 50)]
    s = _only(analyze_track(_track(events)))
    assert s.motion_state == "uncertain" and s.basis == "ambiguous_position"


def test_a_typical_track_with_detector_misses_at_both_ends():
    # three-frame differencing can never detect the first or last sample
    events = ([_ev(0.0, status="not_detected")] + _line(60, 100, 30, 0, n=4, t0=0.5)
              + [_ev(2.5, status="not_detected")])
    segments = analyze_track(_track(events))
    assert [s.motion_state for s in segments] == ["uncertain", "moving", "uncertain"]
    assert (segments[1].start_sec, segments[1].end_sec) == (0.5, 2.0)


# ----------------------------- 6. timing -----------------------------

def test_large_gap_between_detections_is_not_called_one_movement():
    events = [_ev(0.0, 50, 50), _ev(5.0, 250, 50)]
    s = _only(analyze_track(_track(events)))
    assert s.motion_state == "uncertain" and s.basis == "sampling_gap"


def test_a_wider_allowed_gap_uses_the_real_elapsed_time():
    s = _only(analyze_track(_track([_ev(0.0, 50, 50), _ev(5.0, 250, 50)]), max_interval_sec=10.0))
    assert s.motion_state == "moving" and s.mean_speed_norm == round(200 / DIAG / 5.0, 4)


def test_speed_is_path_length_over_real_elapsed_time_with_uneven_sampling():
    events = [_ev(0.0, 50, 50), _ev(0.25, 80, 50), _ev(1.25, 230, 50)]  # 30px in 0.25s, 150px in 1s
    s = _only(analyze_track(_track(events)))
    assert s.mean_speed_norm == round(180 / DIAG / 1.25, 4)


def test_samples_at_the_same_instant_are_handled_safely():
    s = _only(analyze_track(_track([_ev(1.0, 50, 50), _ev(1.0, 200, 50)])))
    assert s.motion_state == "uncertain" and s.basis == "zero_elapsed_time"


# ----------------------------- 7-8. degenerate tracks -----------------------------

def test_single_observation_yields_no_direction_or_speed():
    s = _only(analyze_track(PointerTrack(start_sec=0.0, end_sec=2.0, events=[_ev(1.0, 50, 50)])))
    assert s.motion_state == "uncertain" and s.basis == "too_few_samples"
    assert (s.start_sec, s.end_sec) == (0.0, 2.0)
    assert s.direction_deg is None and s.mean_speed_norm is None


def test_empty_track_is_one_explicit_uncertain_segment():
    s = _only(analyze_track(PointerTrack(start_sec=3.0, end_sec=7.5, events=[])))
    assert s.motion_state == "uncertain" and s.basis == "no_pointer_samples"
    assert (s.start_sec, s.end_sec) == (3.0, 7.5)


def test_out_of_order_events_are_rejected():
    try:
        analyze_track(_track([_ev(1.0, 50, 50), _ev(0.5, 60, 50)], start=0.0, end=1.0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_max_interval_is_rejected():
    for bad in (0.0, -1.0, float("nan")):
        try:
            analyze_track(_track(_line(50, 50, 30, 0)), max_interval_sec=bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_frame_size_change_breaks_comparability():
    events = [_ev(0.0, 50, 50), _ev(0.5, 150, 50, w=600, h=800)]
    assert _only(analyze_track(_track(events))).basis == "frame_size_changed"


# ----------------------------- segmentation -----------------------------

def test_segments_tile_the_whole_range_in_order():
    events = (_line(50, 50, 30, 0, n=3) + [_ev(1.5, status="not_detected")]
              + _line(100, 300, 0, -40, n=3, t0=2.0))
    segments = analyze_track(_track(events, start=0.0, end=4.0))
    assert segments[0].start_sec == 0.0 and segments[-1].end_sec == 4.0
    for a, b in zip(segments, segments[1:]):
        assert a.end_sec == b.start_sec, "segments must leave no gaps and no overlaps"
    assert [s.motion_state for s in segments] == ["moving", "uncertain", "moving", "uncertain"]
    assert [s.direction_deg for s in segments if s.motion_state == "moving"] == [0.0, 270.0]
    assert segments[-1].basis == "not_sampled"  # 3.0 -> 4.0 was never sampled


def test_a_turning_path_is_moving_without_a_single_direction():
    events = [_ev(0.0, 50, 50), _ev(0.5, 110, 50), _ev(1.0, 110, 110)]  # right, then down
    s = _only(analyze_track(_track(events)))
    assert s.motion_state == "moving" and s.direction_deg is None
    assert s.basis == "consecutive_detections_displaced_heading_varies"
    assert s.mean_speed_norm == round(120 / DIAG / 1.0, 4), "speed still follows the observed path"


# ----------------------------- confidence -----------------------------

def test_confidence_reflects_detector_quality_and_observation_count():
    assert _only(analyze_track(_track(_line(50, 50, 30, 0, n=5)))).confidence == 0.9
    one_interval = _only(analyze_track(_track(_line(50, 50, 30, 0, n=2))))
    assert one_interval.confidence == 0.45, "a single displaced interval is half-supported"
    two_frame = [_ev(i * 0.5, 50 + 30 * i, 50, conf=0.75) for i in range(4)]
    assert _only(analyze_track(_track(two_frame))).confidence == 0.75


# ----------------------------- 9. determinism -----------------------------

def test_identical_input_gives_identical_output():
    events = (_line(50, 50, 30, 10, n=4) + [_ev(2.0, status="not_detected")]
              + _line(200, 300, -20, 0, n=3, t0=2.5))
    track = _track(events, start=0.0, end=5.0)
    assert analyze_track(track) == analyze_track(track)


# ----------------------------- contract -----------------------------

def test_cursor_segment_contract_rejects_unsupported_claims():
    CursorSegment(0.0, 1.0, "moving", direction_deg=0.0, mean_speed_norm=0.1, confidence=0.9, basis="b")
    CursorSegment(0.0, 1.0, "uncertain", basis="b")
    bad = [
        dict(start_sec=0.0, end_sec=1.0, motion_state="uncertain", direction_deg=90.0, basis="b"),
        dict(start_sec=0.0, end_sec=1.0, motion_state="uncertain", mean_speed_norm=0.1, basis="b"),
        dict(start_sec=0.0, end_sec=1.0, motion_state="uncertain", confidence=0.4, basis="b"),
        dict(start_sec=0.0, end_sec=1.0, motion_state="moving", basis="b"),  # no speed
        dict(start_sec=0.0, end_sec=1.0, motion_state="moving", mean_speed_norm=0.0, basis="b"),
        dict(start_sec=0.0, end_sec=1.0, motion_state="moving", mean_speed_norm=0.1,
             direction_deg=360.0, basis="b"),
        dict(start_sec=2.0, end_sec=1.0, motion_state="uncertain", basis="b"),
        dict(start_sec=0.0, end_sec=1.0, motion_state="uncertain", basis=""),
        dict(start_sec=0.0, end_sec=1.0, motion_state="clicking", basis="b"),
    ]
    for kwargs in bad:
        try:
            CursorSegment(**kwargs)
            assert False, f"expected ValueError for {kwargs}"
        except ValueError:
            pass


# ----------------------------- layering -----------------------------

def test_the_analyzer_stays_below_the_evidence_layer():
    """Schema 1.2 made "cursor_track" an Evidence kind; the mapping lives in
    core/evidence.py (tests/test_evidence_schema.py). The analyzer itself must
    stay a pure function of a PointerTrack that knows nothing of evidence,
    knowledge or synthesis -- detector result -> Evidence is one-way."""
    root = Path(__file__).resolve().parent.parent
    src = (root / "core/cursor_intelligence.py").read_text(encoding="utf-8")
    for upper in ("core.evidence", "core.knowledge", "core.synthesis", "Evidence("):
        assert upper not in src, upper


# ----------------------------- end to end: real detector, real frames -----------------------------

def _moving_cursor_video(path: str):
    """Same ground truth as tests/test_pointer.py: a 12x12 white square on a
    320x240 gray field, top-left at (20+40t, 20+20t) -- 44.7 px/s along 26.6 deg."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=gray:size=320x240:rate=10:duration=5",
         "-f", "lavfi", "-i", "color=c=white:size=12x12:rate=10:duration=5",
         "-filter_complex", "[0][1]overlay=x='20+t*40':y='20+t*20':shortest=1",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )


def test_end_to_end_motion_matches_ground_truth():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _moving_cursor_video(p)
        video = ingest(p)
        ext = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        track, segments = analyze_cursor_motion(video, 0.0, 4.9, ext, interval_sec=0.5)

        assert track == track_pointer(video, 0.0, 4.9, ext, interval_sec=0.5), "raw track is the existing one"
        moving = [s for s in segments if s.motion_state == "moving"]
        assert moving, segments
        expected_speed = math.hypot(40, 20) / math.hypot(320, 240)  # 0.1118 diagonals/s
        for s in moving:
            assert abs(s.direction_deg - 26.6) <= 3.0, s
            assert abs(s.mean_speed_norm - expected_speed) <= 0.01, s
        assert segments[0].start_sec == 0.0 and segments[-1].end_sec == 4.9


def test_end_to_end_video_without_motion_never_claims_stationary():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "still.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "color=c=gray:size=320x240:rate=10:duration=3", "-pix_fmt", "yuv420p", p],
                       capture_output=True, check=True)
        video = ingest(p)
        _track_, segments = analyze_cursor_motion(video, 0.0, 2.5, FrameExtractor(
            cache_dir=os.path.join(tmp, "cache")))
        assert [s.motion_state for s in segments] == ["uncertain"]
        assert segments[0].basis == "pointer_not_observed"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"  ok: {t.__name__}")
    print("All cursor-intelligence tests passed.")
