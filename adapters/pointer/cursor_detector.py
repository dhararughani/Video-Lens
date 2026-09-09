"""Classical, motion-based detection of a cursor/pointer BAKED INTO video
pixels (screen recordings) -- not live OS mouse tracking. See docs/pointer.md
for the investigation writeup (why three-frame differencing + contour
filtering was chosen over optical flow / template matching / ML, and its
measured limits).

Uses OpenCV (already an installed dependency of `scenedetect`, see
requirements.txt) for `absdiff`/`threshold`/`findContours` -- mature,
well-tested primitives, not hand-rolled pixel loops.
"""
from __future__ import annotations

import cv2
import numpy as np

from adapters.frames.frame_extractor import FrameExtractor
from core.contracts import Frame, PointerEvent, PointerTrack, VideoInput

# Cursor-shape heuristics (tuned against synthetic + real test videos, see
# docs/pointer.md). Expressed relative to frame diagonal/area so they scale
# across resolutions instead of hardcoding pixel counts.
_MIN_AREA_FRAC = 0.00005       # smaller than this is noise
_MAX_AREA_FRAC = 0.01          # bigger than ~1% of the frame is not a cursor
_MAX_SCENE_MOTION_FRAC = 0.15  # a scroll/transition/redraw floods most of the frame
_MAX_STEP_FRAC = 0.15          # max plausible pointer displacement between samples
_MIN_SOLIDITY = 0.4            # contour_area / bbox_area -- rejects thin chart lines
_DIFF_THRESHOLD = 25           # 0-255 grayscale diff to count as "changed"


def _load_gray(path: str) -> np.ndarray | None:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _motion_mask(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray | None:
    if img_a.shape != img_b.shape:
        return None
    diff = cv2.absdiff(img_a, img_b)
    _, mask = cv2.threshold(diff, _DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def _blobs_from_mask(mask: np.ndarray, frame_area: int) -> list[tuple[float, float, float]]:
    """Cursor-plausible blobs in a motion mask, as (cx, cy, area_frac)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        area_frac = area / frame_area
        if not (_MIN_AREA_FRAC <= area_frac <= _MAX_AREA_FRAC):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        bbox_area = bw * bh
        if bbox_area == 0 or area / bbox_area < _MIN_SOLIDITY:
            continue
        out.append((x + bw / 2, y + bh / 2, area_frac))
    return out


def _two_frame_candidates(frame_a: Frame, frame_b: Frame) -> list[tuple[float, float, float]]:
    """Motion blobs between two frames. A translating object shows up as TWO
    blobs here (the position it left, and the position it arrived at) --
    this pairwise form can't tell which is which without other context, so
    `detect_pointer_for_frames` prefers three-frame differencing instead (see
    `_triple_frame_candidates`). Kept for the two-frame `detect_pointer`
    convenience wrapper, where no third frame is available."""
    img_a, img_b = _load_gray(frame_a.path), _load_gray(frame_b.path)
    if img_a is None or img_b is None:
        return []
    mask = _motion_mask(img_a, img_b)
    if mask is None:
        return []
    frame_area = img_a.shape[0] * img_a.shape[1]
    if frame_area == 0 or np.count_nonzero(mask) / frame_area > _MAX_SCENE_MOTION_FRAC:
        return []
    return _blobs_from_mask(mask, frame_area)


def _triple_frame_candidates(prev_img: np.ndarray, cur_img: np.ndarray,
                              next_img: np.ndarray) -> list[tuple[float, float, float]]:
    """Candidates AT `cur_img`'s timestamp via three-frame differencing: the
    INTERSECTION of motion(prev,cur) and motion(cur,next) isolates pixels
    that changed on both sides of `cur` -- i.e. where a moving object
    actually sits at `cur`, without the two-frame method's vacated/arrived
    ambiguity. A cursor that pauses at `cur` (no motion on one side) will not
    show up here -- documented limitation, see docs/pointer.md."""
    if prev_img.shape != cur_img.shape or cur_img.shape != next_img.shape:
        return []
    frame_area = cur_img.shape[0] * cur_img.shape[1]
    if frame_area == 0:
        return []
    m1, m2 = _motion_mask(prev_img, cur_img), _motion_mask(cur_img, next_img)
    if m1 is None or m2 is None:
        return []
    if (np.count_nonzero(m1) / frame_area > _MAX_SCENE_MOTION_FRAC or
            np.count_nonzero(m2) / frame_area > _MAX_SCENE_MOTION_FRAC):
        return []  # scene-wide change (scroll/transition/redraw) on either side, not a pointer
    combined = cv2.bitwise_and(m1, m2)
    return _blobs_from_mask(combined, frame_area)


def detect_pointer(frame: Frame, prev_frame: Frame | None = None) -> PointerEvent:
    """Single-timestamp convenience wrapper using two-frame differencing.
    Without a `prev_frame` there is no motion signal to work with, so this
    honestly returns `not_detected` rather than guessing from one static
    image. Prefer `detect_pointer_for_frames`/`track_pointer` (three-frame
    differencing) when 3+ frames are available -- it's the more reliable
    path; see docs/pointer.md."""
    if prev_frame is None:
        return PointerEvent(timestamp_sec=frame.timestamp_sec, status="not_detected",
                             frame_width=frame.width, frame_height=frame.height,
                             detection_method="motion_diff_2frame")
    cands = _two_frame_candidates(prev_frame, frame)
    if not cands:
        return PointerEvent(timestamp_sec=frame.timestamp_sec, status="not_detected",
                             frame_width=frame.width, frame_height=frame.height,
                             detection_method="motion_diff_2frame")
    # ambiguous vacated/arrived pair with no trajectory context to disambiguate --
    # cap confidence below the "detected" threshold rather than picking blind
    cx, cy, _ = min(cands, key=lambda c: c[2])
    confidence = 0.55 if len(cands) > 1 else 0.75
    status = "detected" if confidence >= 0.7 else "uncertain"
    return PointerEvent(timestamp_sec=frame.timestamp_sec, status=status,
                         x=round(cx), y=round(cy), frame_width=frame.width,
                         frame_height=frame.height, confidence=confidence,
                         detection_method="motion_diff_2frame")


def detect_pointer_for_frames(frames: list[Frame]) -> list[PointerEvent]:
    """Detect pointer positions across an ORDERED frame sequence using
    three-frame differencing, with each detection preferring the candidate
    nearest the last confirmed position (temporal coherence) within a
    plausible displacement. The first and last frame have no triple to
    difference against and are `not_detected`. A gap in evidence stays
    `not_detected` -- this never bridges missing frames with an invented
    trajectory."""
    n = len(frames)
    if n == 0:
        return []
    if n < 3:
        return [PointerEvent(timestamp_sec=f.timestamp_sec, status="not_detected",
                              frame_width=f.width, frame_height=f.height,
                              detection_method="motion_diff_3frame") for f in frames]

    imgs = [_load_gray(f.path) for f in frames]
    events: list[PointerEvent] = [
        PointerEvent(timestamp_sec=frames[0].timestamp_sec, status="not_detected",
                     frame_width=frames[0].width, frame_height=frames[0].height,
                     detection_method="motion_diff_3frame")
    ]
    last_xy = None
    diag = (frames[0].width ** 2 + frames[0].height ** 2) ** 0.5

    for i in range(1, n - 1):
        f = frames[i]
        cands = []
        if imgs[i - 1] is not None and imgs[i] is not None and imgs[i + 1] is not None:
            cands = _triple_frame_candidates(imgs[i - 1], imgs[i], imgs[i + 1])

        chosen = None
        if cands:
            pool = cands
            if last_xy is not None and diag > 0:
                in_range = [c for c in cands if
                            ((c[0] - last_xy[0]) ** 2 + (c[1] - last_xy[1]) ** 2) ** 0.5
                            <= _MAX_STEP_FRAC * diag]
                if in_range:
                    pool = in_range
            chosen = min(pool, key=lambda c: c[2])  # smallest/most compact -- least likely noise

        if chosen is None:
            events.append(PointerEvent(timestamp_sec=f.timestamp_sec, status="not_detected",
                                        frame_width=f.width, frame_height=f.height,
                                        detection_method="motion_diff_3frame"))
            last_xy = None
            continue

        cx, cy, _area_frac = chosen
        confidence = 0.9 if len(cands) == 1 else 0.6  # one unambiguous candidate vs. picked among several
        status = "detected" if confidence >= 0.7 else "uncertain"
        events.append(PointerEvent(timestamp_sec=f.timestamp_sec, status=status,
                                    x=round(cx), y=round(cy),
                                    frame_width=f.width, frame_height=f.height,
                                    confidence=confidence, detection_method="motion_diff_3frame"))
        last_xy = (cx, cy)

    events.append(PointerEvent(timestamp_sec=frames[-1].timestamp_sec, status="not_detected",
                                frame_width=frames[-1].width, frame_height=frames[-1].height,
                                detection_method="motion_diff_3frame"))
    return events


def detect_pointer_at(video: VideoInput, timestamp_sec: float, extractor: FrameExtractor,
                       step_sec: float = 0.1) -> PointerEvent:
    """Pointer position at one timestamp, using the two adjacent frames
    `step_sec` before/after for three-frame-differencing context."""
    prev = extractor.get_frame(video, max(0.0, timestamp_sec - step_sec))
    cur = extractor.get_frame(video, timestamp_sec)
    nxt = extractor.get_frame(video, timestamp_sec + step_sec)
    return detect_pointer_for_frames([prev, cur, nxt])[1]


def track_pointer(video: VideoInput, start_sec: float, end_sec: float,
                   extractor: FrameExtractor, interval_sec: float = 0.5) -> PointerTrack:
    """Pointer track across a time range, reusing Step 4's frame extraction
    rather than duplicating it."""
    frames = extractor.extract_window(video, start_sec, end_sec, interval_sec)
    events = detect_pointer_for_frames(frames)
    n_detected = sum(1 for e in events if e.status == "detected")
    confidence = n_detected / len(events) if events else 0.0
    return PointerTrack(start_sec=start_sec, end_sec=end_sec, events=events, confidence=confidence)
