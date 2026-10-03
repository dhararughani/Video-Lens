"""Cursor intelligence v2 (P0-C): what an existing PointerTrack establishes about
pointer MOVEMENT over time -- when it moved, in which direction, how fast, and
how sure that is. A temporal analysis layer over detections that already
exist; it detects nothing itself and reads no pixels.

The one rule everything here is built around: the pointer detector
(adapters/pointer/cursor_detector.py) finds a cursor by its MOTION -- three-frame
differencing -- so a cursor that holds still is invisible to it. Therefore:

    no movement observed  !=  cursor was stationary

Every stretch the evidence cannot speak to is "uncertain", with a `basis`
saying why. "stationary" is not a state this module can produce, and
CursorSegment rejects it outright. Clicks, drags, hovering, selecting or
"pointing at" anything are interpretation and are not attempted.

Conventions (all deterministic, documented in docs/pointer.md):

- Geometry is computed in PIXELS, from `PointerEvent.x/y`, then normalized by
  the frame DIAGONAL. Normalizing x and y separately would distort angles on any
  non-square frame (on 16:9 a visually 45-degree move would read ~29 degrees);
  the diagonal keeps distance and direction isotropic while staying resolution
  independent -- the same convention the detector uses for its step limit.
- direction_deg: 0 = right, 90 = down, 180 = left, 270 = up (image y grows
  downward), in [0, 360), rounded to 0.1.
- mean_speed_norm: observed path length / elapsed time, in frame diagonals per
  second. The path is the straight lines between samples, so it is a LOWER
  bound if the cursor curved between them.

Not (yet) an `Evidence` kind: nothing in knowledge/synthesis/evidence reads
this. Promoting it is the separate, atomic schema step.
"""
from __future__ import annotations

import math

from core.contracts import CursorSegment, PointerEvent, PointerTrack

# Two detections closer than this (fraction of the frame diagonal) are within
# detector jitter, so they do not establish movement -- and, per the module
# rule, not stillness either. Same magnitude as core/evidence.py's
# _MOTION_THRESHOLD, expressed in diagonal units so it is aspect-independent.
_MIN_DISPLACEMENT_DIAG = 0.03

# A moving segment only reports one heading when its path is nearly straight:
# net displacement / path length >= this. A path that turns has no single
# direction, so it is reported as moving with direction None.
_MIN_STRAIGHTNESS = 0.8

# How many consecutive displaced intervals a moving segment needs for full
# observation-count support; fewer scale its confidence down proportionally.
_FULL_SUPPORT_INTERVALS = 2

DEFAULT_MAX_INTERVAL_SEC = 2.0


def analyze_track(track: PointerTrack, *,
                  max_interval_sec: float = DEFAULT_MAX_INTERVAL_SEC) -> tuple[CursorSegment, ...]:
    """Segment a PointerTrack into "moving" and "uncertain" stretches.

    Each pair of consecutive samples is one interval. It is "moving" only when
    BOTH ends are confident detections (status "detected", same frame size), at
    most `max_interval_sec` apart, displaced by at least the jitter threshold.
    Anything else is "uncertain", with the reason as its basis. Adjacent
    moving intervals merge into one segment; adjacent uncertain intervals merge
    when they share a reason. The result tiles the track's whole time range in
    order -- including any unsampled lead-in/tail -- so nothing is silently
    dropped. Never empty, never raises for sparse evidence."""
    if not math.isfinite(max_interval_sec) or max_interval_sec <= 0:
        raise ValueError(f"max_interval_sec must be a finite number > 0, got {max_interval_sec}")
    events = list(track.events)
    for a, b in zip(events, events[1:]):
        if b.timestamp_sec < a.timestamp_sec:
            raise ValueError(f"track events must be ordered by timestamp: {b.timestamp_sec} "
                             f"follows {a.timestamp_sec}")

    start = min([track.start_sec] + [e.timestamp_sec for e in events[:1]])
    end = max([track.end_sec] + [e.timestamp_sec for e in events[-1:]])
    if len(events) < 2:
        basis = "no_pointer_samples" if not events else "too_few_samples"
        return (CursorSegment(start_sec=start, end_sec=end, motion_state="uncertain", basis=basis),)

    # (t0, t1, basis-or-None, points) -- basis None marks a "moving" interval
    intervals: list[tuple[float, float, str | None, list[PointerEvent]]] = []
    if start < events[0].timestamp_sec:
        intervals.append((start, events[0].timestamp_sec, "not_sampled", []))
    for a, b in zip(events, events[1:]):
        intervals.append((a.timestamp_sec, b.timestamp_sec, _why_not_moving(a, b, max_interval_sec), [a, b]))
    if events[-1].timestamp_sec < end:
        intervals.append((events[-1].timestamp_sec, end, "not_sampled", []))

    segments: list[CursorSegment] = []
    run: list = []
    for interval in intervals:
        if run and run[-1][2] != interval[2]:  # basis changed -> close the run
            segments.append(_segment(run))
            run = []
        run.append(interval)
    segments.append(_segment(run))
    return tuple(segments)


# ------------------------------- internals -------------------------------

def _why_not_moving(a: PointerEvent, b: PointerEvent, max_interval_sec: float) -> str | None:
    """None if this interval is evidence of movement, else the reason it is not."""
    dt = b.timestamp_sec - a.timestamp_sec
    if dt <= 0:
        return "zero_elapsed_time"
    if dt > max_interval_sec:
        return "sampling_gap"
    if not (_positioned(a) and _positioned(b)):
        return "pointer_not_observed"
    if a.status != "detected" or b.status != "detected":
        return "ambiguous_position"  # detector picked among several candidates
    if (a.frame_width, a.frame_height) != (b.frame_width, b.frame_height):
        return "frame_size_changed"
    if _distance_diag(a, b) < _MIN_DISPLACEMENT_DIAG:
        return "displacement_within_jitter"  # NOT stillness -- see module docstring
    return None


def _positioned(e: PointerEvent) -> bool:
    return e.status != "not_detected" and e.frame_width > 0 and e.frame_height > 0


def _distance_diag(a: PointerEvent, b: PointerEvent) -> float:
    return math.hypot(b.x - a.x, b.y - a.y) / math.hypot(a.frame_width, a.frame_height)


def _segment(run: list) -> CursorSegment:
    t0, t1, basis = run[0][0], run[-1][1], run[0][2]
    if basis is not None:
        return CursorSegment(start_sec=t0, end_sec=t1, motion_state="uncertain", basis=basis)

    points = [run[0][3][0]] + [interval[3][1] for interval in run]
    path = sum(_distance_diag(p, q) for p, q in zip(points, points[1:]))
    net = _distance_diag(points[0], points[-1])
    first, last = points[0], points[-1]

    direction = None
    basis = "consecutive_detections_displaced"
    if net >= _MIN_DISPLACEMENT_DIAG and net / path >= _MIN_STRAIGHTNESS:
        angle = math.degrees(math.atan2(last.y - first.y, last.x - first.x)) % 360.0
        direction = round(angle, 1) % 360.0  # 359.96 rounds to 360.0 -> 0.0
    else:
        basis = "consecutive_detections_displaced_heading_varies"

    # Confidence = the detector's own confidence in these positions, scaled by
    # how many consecutive displaced intervals support the movement.
    detector = sum(p.confidence for p in points) / len(points)
    support = min(1.0, len(run) / _FULL_SUPPORT_INTERVALS)
    return CursorSegment(
        start_sec=t0, end_sec=t1, motion_state="moving",
        direction_deg=direction,
        mean_speed_norm=round(path / (t1 - t0), 4),
        confidence=round(detector * support, 4),
        basis=basis,
    )
