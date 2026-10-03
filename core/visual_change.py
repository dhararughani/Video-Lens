"""Temporal visual change: how much did the picture change between two sampled
frames, and where. Deterministic measurement only -- no model, no semantics.

Two layers (P0-B L1/L2):

  L1  magnitude  -- the fraction of frame area whose luma changed by more than
                    PIXEL_DIFF_THRESHOLD (0-255). The same per-pixel rule the
                    pointer detector already uses, now shared from here.
  L2  regions    -- when magnitude >= threshold, the changed pixels are grouped
                    into a few normalized `Region`s, and the event is called
                    "localized" or "global" by how much of the frame those
                    regions cover.

What this deliberately does NOT do: name the change. "Scroll", "zoom",
"redraw", "annotation", "page transition" are interpretations of a change, not
measurements of one; they are future classification work and appear nowhere
here. A `VisualChangeEvent` is also not (yet) an `Evidence` kind -- promoting it
is the atomic schema step (P1-B), so nothing in knowledge/synthesis reads it.

OpenCV/NumPy are imported lazily, as in core/visual_evidence.py: `core/` stays
importable without an imaging library, and losing it degrades to
status="unavailable" rather than breaking anything -- or worse, to "no change".
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from core.contracts import Frame, Region, VisualChangeEvent

# 0-255 luma difference above which ONE pixel counts as changed. Tuned for the
# pointer detector (docs/pointer.md) and shared with it, so the two never drift
# apart. Measured on a real 1080p screen recording: 55% of consecutive-frame
# pairs score exactly 0 under this rule -- codec noise sits below it.
PIXEL_DIFF_THRESHOLD = 25

# Fraction of frame area that must change for a pair to count as a meaningful
# visual change (compared with >=). 0.08 is measured, not guessed: on a real
# talking-head video sampled 0.5s apart, natural presenter motion has p95 = 0.064,
# so 0.08 sits just above continuous motion and fires on genuine state
# transitions. It therefore MISSES small localized changes (a tooltip is ~0.5%
# of a frame); for clean screen recordings, where measured noise is ~0, pass a
# lower threshold such as 0.005. The right value depends on the content and on
# the sampling interval -- which is why it is a parameter, not a constant.
DEFAULT_VISUAL_CHANGE_THRESHOLD = 0.08

# Changed blobs within ~2 x this fraction of the longer frame side of each
# other are one region (a line of text is many glyphs, but one change).
_MERGE_RADIUS_FRAC = 0.005
# A region must contain at least this fraction of the frame area in actually
# changed pixels; smaller specks are noise, not places.
_MIN_REGION_FRAC = 0.0005
# Report at most this many regions (largest first) -- a change scattered into
# dozens of places is better described as broad than listed exhaustively.
_MAX_REGIONS = 8
# Regions covering at least this fraction of the frame make the change global.
_GLOBAL_COVERAGE = 0.5


def changed_pixel_mask(img_a, img_b):
    """uint8 mask (255 = changed) of pixels whose difference exceeds
    PIXEL_DIFF_THRESHOLD, or None if the images differ in shape. Shared with
    adapters/pointer/cursor_detector.py -- one per-pixel rule, one place."""
    import cv2
    if img_a.shape != img_b.shape:
        return None
    diff = cv2.absdiff(img_a, img_b)
    _, mask = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def detect_visual_changes(frames: Sequence[Frame],
                          threshold: float = DEFAULT_VISUAL_CHANGE_THRESHOLD) -> list[VisualChangeEvent]:
    """One event per consecutive pair: frames[i] -> frames[i+1]. Linear in the
    number of frames, and each image is read from disk exactly once.

    `frames` must already be ordered by timestamp (as `extract_window`,
    `select_keyframes` and `inspect_visual_evidence` produce them). Out-of-order
    input raises rather than being silently re-sorted: comparing in an order
    other than the caller's is a different measurement."""
    _check_threshold(threshold)
    frames = list(frames)
    for a, b in zip(frames, frames[1:]):
        if b.timestamp_sec < a.timestamp_sec:
            raise ValueError(f"frames must be ordered by timestamp: {b.timestamp_sec} "
                             f"follows {a.timestamp_sec}")
    if len(frames) < 2:
        return []

    events = []
    previous = _load(frames[0])
    for a, b in zip(frames, frames[1:]):
        current = _load(b)
        events.append(_measure(previous, current, a.timestamp_sec, b.timestamp_sec, threshold))
        previous = current
    return events


def compare_frames(earlier: Frame, later: Frame,
                   threshold: float = DEFAULT_VISUAL_CHANGE_THRESHOLD) -> VisualChangeEvent:
    """The single-pair form of `detect_visual_changes`."""
    return detect_visual_changes([earlier, later], threshold)[0]


# ------------------------------- internals -------------------------------

def _check_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")


def _load(frame: Frame):
    """(grayscale image, None) or (None, why it could not be read). Grayscale
    handles RGB, RGBA and gray sources alike; the cost is that a pure hue
    change with equal luma is invisible -- the same documented trade-off as
    frame de-duplication (docs/frames.md)."""
    try:
        import cv2
    except ImportError:
        return None, "OpenCV (cv2) is not installed"
    img = cv2.imread(frame.path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None, f"frame at {frame.timestamp_sec}s could not be read: {frame.path}"
    return img, None


def _unavailable(t0: float, t1: float, method: str, why: str) -> VisualChangeEvent:
    return VisualChangeEvent(timestamp_sec=t1, kind="visual_change", status="unavailable",
                             magnitude=None, detection_method=method,
                             compared_timestamps=(t0, t1), detail=why)


def _measure(a, b, t0: float, t1: float, threshold: float) -> VisualChangeEvent:
    method = f"luma_absdiff>{PIXEL_DIFF_THRESHOLD}/area>={threshold:g}"
    (img_a, why_a), (img_b, why_b) = a, b
    if img_a is None or img_b is None:
        return _unavailable(t0, t1, method, why_a or why_b)
    if img_a.shape != img_b.shape:
        # Resampling one to match the other would itself manufacture differences.
        return _unavailable(t0, t1, method,
                            f"frame sizes differ ({img_a.shape[1]}x{img_a.shape[0]} vs "
                            f"{img_b.shape[1]}x{img_b.shape[0]}); they cannot be compared pixel-for-pixel")

    changed = changed_pixel_mask(img_a, img_b) > 0
    magnitude = int(changed.sum()) / changed.size
    # Not a probability: how far the measurement sits from the decision
    # boundary, in units of the threshold, capped at 1. Identical frames -> 1.0;
    # exactly at the threshold -> 0.0 (detected, but only just).
    confidence = min(1.0, abs(magnitude - threshold) / threshold)

    if magnitude < threshold:
        return VisualChangeEvent(timestamp_sec=t1, kind="visual_change", status="not_detected",
                                 magnitude=magnitude, confidence=confidence,
                                 detection_method=method, compared_timestamps=(t0, t1))

    boxes = _localize(changed)
    h, w = changed.shape
    coverage = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes) / (w * h)
    regions = tuple(Region(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in boxes[:_MAX_REGIONS])
    detail = (f"{len(boxes)} separate changed areas; the {_MAX_REGIONS} largest are reported"
              if len(boxes) > _MAX_REGIONS else "")
    return VisualChangeEvent(
        timestamp_sec=t1,
        kind="visual_change_global" if coverage >= _GLOBAL_COVERAGE else "visual_change_localized",
        status="detected", magnitude=magnitude, regions=regions, confidence=confidence,
        detection_method=method, compared_timestamps=(t0, t1), detail=detail,
    )


def _localize(changed) -> list[tuple[int, int, int, int]]:
    """Changed areas as pixel boxes (x1, y1, x2, y2), half-open, non-overlapping,
    largest first. Nearby blobs are grouped by dilation, but each box is the
    tight bound of the ACTUALLY changed pixels in its group -- dilation decides
    what belongs together, never how big a region is."""
    import cv2
    import numpy as np

    h, w = changed.shape
    radius = max(1, round(_MERGE_RADIUS_FRAC * max(h, w)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    grouped = cv2.dilate(changed.astype(np.uint8), kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    counts = np.bincount(labels[changed], minlength=n)  # changed pixels per group
    min_pixels = max(1, _MIN_REGION_FRAC * h * w)
    boxes = []
    for label in range(1, n):
        if counts[label] < min_pixels:
            continue
        x, y, bw, bh = stats[label, :4]
        ys, xs = np.nonzero(changed[y:y + bh, x:x + bw] & (labels[y:y + bh, x:x + bw] == label))
        boxes.append((x + int(xs.min()), y + int(ys.min()), x + int(xs.max()) + 1, y + int(ys.max()) + 1))

    if not boxes:
        # Every changed pixel is an isolated speck, yet together they cleared the
        # threshold: the change is diffuse. One box bounding all of it says so,
        # instead of dozens of meaningless specks.
        ys, xs = np.nonzero(changed)
        boxes = [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]

    boxes = _merge_overlapping(boxes)
    return sorted(boxes, key=lambda b: (-(b[2] - b[0]) * (b[3] - b[1]), b[1], b[0]))


def _merge_overlapping(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Union boxes that intersect, until none do. O(k^2) per pass over the few
    boxes that survive filtering -- not over pixels."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes
