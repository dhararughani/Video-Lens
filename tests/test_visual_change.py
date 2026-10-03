"""P0-B L1/L2 tests: deterministic temporal visual change -- magnitude,
threshold rule, localization into normalized Regions, honest unavailability,
and the shared per-pixel rule with the pointer detector.

Synthetic frames are written as lossless PNGs, so every expected magnitude and
region below is exact, not approximate. Run: python tests/test_visual_change.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import KNOWLEDGE_SCHEMA_VERSION, Evidence, Frame, Region, VisualChangeEvent
from core.visual_change import (
    DEFAULT_VISUAL_CHANGE_THRESHOLD, PIXEL_DIFF_THRESHOLD, changed_pixel_mask, compare_frames,
    detect_visual_changes,
)

W, H = 640, 360


def _frame(tmp: str, name: str, img: np.ndarray, ts: float) -> Frame:
    path = os.path.join(tmp, f"{name}.png")
    assert cv2.imwrite(path, img)
    return Frame(timestamp_sec=ts, path=path, width=img.shape[1], height=img.shape[0])


def _textured(seed: int = 0, h: int = H, w: int = W) -> np.ndarray:
    """Smooth, natural-looking texture (blurred noise), values well inside 0-255."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w)).astype(np.float32), (0, 0), 4)
    return cv2.normalize(img, None, 70, 185, cv2.NORM_MINMAX).astype(np.uint8)


def _shift(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """A copy where every pixel in [x1:x2, y1:y2] changes by exactly 128 luma."""
    out = img.copy()
    out[y1:y2, x1:x2] = (out[y1:y2, x1:x2].astype(np.int16) + 128) % 256
    return out


def _pair(tmp, before, after, t0=1.0, t1=1.5, threshold=DEFAULT_VISUAL_CHANGE_THRESHOLD):
    return compare_frames(_frame(tmp, "a", before, t0), _frame(tmp, "b", after, t1), threshold)


def _speckle(img: np.ndarray, seed: int, count: int) -> np.ndarray:
    """`count` isolated pixels, each changed by exactly 128 luma (so every one
    genuinely clears the per-pixel rule)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    ys, xs = rng.integers(0, img.shape[0], count), rng.integers(0, img.shape[1], count)
    out[ys, xs] = (out[ys, xs].astype(np.int16) + 128) % 256
    return out


def _region_of(x1, y1, x2, y2, w=W, h=H) -> Region:
    return Region(x1 / w, y1 / h, x2 / w, y2 / h)


# ----------------------------- 1. identical -----------------------------

def test_identical_frames_are_not_a_change():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, img.copy())
        assert ev.status == "not_detected" and ev.kind == "visual_change"
        assert ev.magnitude == 0.0 and ev.regions == ()
        assert ev.confidence == 1.0, "as far from the threshold as a measurement can be"


def test_identical_frames_at_different_timestamps_are_not_a_change():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        frames = [_frame(tmp, f"f{i}", img, t) for i, t in enumerate((0.0, 3.0, 60.0))]
        assert [e.status for e in detect_visual_changes(frames)] == ["not_detected"] * 2


# ----------------------------- 2. harmless perturbation -----------------------------

def test_mild_sensor_noise_is_not_a_change():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        noise = np.random.default_rng(1).normal(0, 3, img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        ev = _pair(tmp, img, noisy)
        assert ev.status == "not_detected" and ev.magnitude < 0.001


def test_jpeg_recompression_is_not_a_change():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        ev = _pair(tmp, img, cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE))
        assert ev.status == "not_detected" and ev.magnitude < 0.001


def test_scattered_specks_do_not_clear_the_default_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, _speckle(img, seed=2, count=460))  # ~0.2% of pixels
        assert ev.status == "not_detected" and ev.magnitude > 0


def test_uniform_brightness_drift_below_the_per_pixel_rule_is_not_a_change():
    with tempfile.TemporaryDirectory() as tmp:
        ev = _pair(tmp, np.full((H, W), 100, np.uint8), np.full((H, W), 110, np.uint8))
        assert ev.status == "not_detected" and ev.magnitude == 0.0


# ----------------------------- 3. localized -----------------------------

def test_localized_change_is_detected_and_located_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, _shift(img, 100, 60, 260, 180))  # 160x120 = 8.3% of the frame
        assert ev.status == "detected" and ev.kind == "visual_change_localized"
        assert ev.magnitude == 160 * 120 / (W * H)
        assert ev.regions == (_region_of(100, 60, 260, 180),)


def test_small_change_needs_a_lower_threshold_and_is_then_located_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        after = _shift(img, 500, 300, 550, 340)  # 50x40: a tooltip-sized change, ~0.9%
        assert _pair(tmp, img, after).status == "not_detected"  # by design at the default
        ev = _pair(tmp, img, after, threshold=0.005)
        assert ev.kind == "visual_change_localized"
        assert ev.regions == (_region_of(500, 300, 550, 340),)


def test_multiple_separate_changes_are_separate_regions_largest_first():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        after = _shift(_shift(img, 20, 20, 120, 110), 450, 230, 590, 320)  # 100x90, 140x90
        ev = _pair(tmp, img, after)
        assert ev.kind == "visual_change_localized"
        assert ev.regions == (_region_of(450, 230, 590, 320), _region_of(20, 20, 120, 110))


def test_nearby_blobs_like_glyphs_merge_into_one_region():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        after = img
        for i in range(8):  # eight 10x16 "glyphs" with 4px gaps: one line of text
            x = 200 + i * 14
            after = _shift(after, x, 100, x + 10, 116)
        ev = _pair(tmp, img, after, threshold=0.001)
        assert ev.regions == (_region_of(200, 100, 200 + 7 * 14 + 10, 116),), \
            "a region is the tight bound of the changed pixels, never inflated by grouping"


def test_scattered_noise_never_becomes_dozens_of_regions():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, _speckle(img, seed=3, count=460), threshold=0.001)
        assert ev.status == "detected"
        assert len(ev.regions) == 1, "diffuse specks are one broad change, not hundreds of places"
        assert ev.kind == "visual_change_global"


def test_region_count_is_capped_and_the_truncation_is_stated():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        after = img
        for row in range(3):
            for col in range(4):  # 12 separate 20x20 changes
                after = _shift(after, 40 + col * 150, 40 + row * 110, 60 + col * 150, 60 + row * 110)
        ev = _pair(tmp, img, after, threshold=0.001)
        assert len(ev.regions) == 8
        assert "12 separate changed areas" in ev.detail


# ----------------------------- 4. global -----------------------------

def test_full_frame_change_is_global_with_one_full_region():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, _shift(img, 0, 0, W, H))
        assert ev.status == "detected" and ev.kind == "visual_change_global"
        assert ev.magnitude == 1.0 and ev.regions == (Region(0.0, 0.0, 1.0, 1.0),)
        assert ev.confidence == 1.0


def test_large_uniform_brightness_change_is_global():
    with tempfile.TemporaryDirectory() as tmp:
        ev = _pair(tmp, np.full((H, W), 90, np.uint8), np.full((H, W), 160, np.uint8))
        assert ev.kind == "visual_change_global" and ev.magnitude == 1.0


def test_letterbox_bars_that_do_not_change_are_excluded_from_the_region():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        img[:45], img[315:] = 0, 0  # black bars
        ev = _pair(tmp, img, _shift(img, 0, 45, W, 315))
        assert ev.kind == "visual_change_global"  # 75% of the frame changed
        assert ev.regions == (_region_of(0, 45, W, 315),)


def test_extent_not_amount_decides_global_vs_localized():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        panel = _pair(tmp, img, _shift(img, 0, 0, 260, H))  # a 40%-wide side panel
        assert panel.kind == "visual_change_localized"
        assert panel.magnitude > DEFAULT_VISUAL_CHANGE_THRESHOLD


# ----------------------------- 5. threshold boundary -----------------------------

def _boundary_pair(tmp, changed_pixels: int, threshold: float):
    before = np.zeros((100, 100), np.uint8)
    after = before.copy()
    after.flat[:changed_pixels] = 255
    return _pair(tmp, before, after, threshold=threshold)


def test_threshold_rule_is_greater_than_or_equal():
    with tempfile.TemporaryDirectory() as tmp:
        exact = _boundary_pair(tmp, 800, 0.08)  # 800 / 10,000 == 0.08 exactly
        assert exact.magnitude == 0.08
        assert exact.status == "detected", "magnitude == threshold counts (>=)"
        assert exact.confidence == 0.0, "detected, but only just"
        assert _boundary_pair(tmp, 799, 0.08).status == "not_detected"
        assert _boundary_pair(tmp, 801, 0.08).status == "detected"
        assert _boundary_pair(tmp, 800, 0.0801).status == "not_detected"
        assert _boundary_pair(tmp, 10_000, 0.08).confidence == 1.0


def test_per_pixel_rule_is_strictly_greater_than_25():
    with tempfile.TemporaryDirectory() as tmp:
        base = np.full((50, 50), 100, np.uint8)
        assert _pair(tmp, base, base + PIXEL_DIFF_THRESHOLD).magnitude == 0.0
        assert _pair(tmp, base, base + PIXEL_DIFF_THRESHOLD + 1).magnitude == 1.0


def test_invalid_thresholds_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        f = _frame(tmp, "a", _textured(), 0.0)
        for bad in (0.0, -0.1, 1.5, float("nan"), float("inf")):
            try:
                detect_visual_changes([f, f], threshold=bad)
                assert False, f"expected ValueError for threshold={bad}"
            except ValueError:
                pass


# ----------------------------- 6. timestamps -----------------------------

def test_event_timestamp_is_the_later_frame_and_both_are_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, _shift(img, 0, 0, W, H), t0=12.25, t1=12.75)
        assert ev.compared_timestamps == (12.25, 12.75)
        assert ev.timestamp_sec == 12.75


def test_sequence_compares_each_consecutive_pair_only():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        imgs = [img, img, _shift(img, 0, 0, W, H), _shift(img, 0, 0, W, H)]
        frames = [_frame(tmp, f"f{i}", im, i * 0.5) for i, im in enumerate(imgs)]
        events = detect_visual_changes(frames)
        assert [e.compared_timestamps for e in events] == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]
        assert [e.status for e in events] == ["not_detected", "detected", "not_detected"]


def test_out_of_order_frames_are_rejected_not_silently_resorted():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        frames = [_frame(tmp, "a", img, 2.0), _frame(tmp, "b", img, 1.0)]
        try:
            detect_visual_changes(frames)
            assert False, "expected ValueError for out-of-order frames"
        except ValueError:
            pass


def test_fewer_than_two_frames_yield_no_events():
    with tempfile.TemporaryDirectory() as tmp:
        assert detect_visual_changes([]) == []
        assert detect_visual_changes([_frame(tmp, "a", _textured(), 0.0)]) == []


# ----------------------------- 7. determinism -----------------------------

def test_identical_input_gives_identical_output():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        imgs = [img, _shift(img, 100, 60, 260, 180), _shift(img, 0, 0, W, H), img]
        frames = [_frame(tmp, f"f{i}", im, float(i)) for i, im in enumerate(imgs)]
        assert detect_visual_changes(frames, 0.01) == detect_visual_changes(frames, 0.01)


# ----------------------------- 8. region bounds -----------------------------

def test_regions_touching_the_edges_stay_within_unit_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        ev = _pair(tmp, img, _shift(_shift(img, 0, 0, 120, 100), 500, 250, W, H), threshold=0.01)
        assert _region_of(500, 250, W, H) in ev.regions and _region_of(0, 0, 120, 100) in ev.regions
        for seed in range(5):  # random rectangles, anywhere
            rng = np.random.default_rng(seed)
            after = img
            for _ in range(4):
                x1, y1 = int(rng.integers(0, W - 2)), int(rng.integers(0, H - 2))
                after = _shift(after, x1, y1, int(rng.integers(x1 + 1, W + 1)), int(rng.integers(y1 + 1, H + 1)))
            for r in _pair(tmp, img, after, threshold=0.001).regions:
                assert 0.0 <= r.x1 < r.x2 <= 1.0 and 0.0 <= r.y1 < r.y2 <= 1.0


def test_very_small_frames_are_handled():
    with tempfile.TemporaryDirectory() as tmp:
        for h, w in ((4, 4), (1, 1), (2, 7)):
            ev = _pair(tmp, np.zeros((h, w), np.uint8), np.full((h, w), 200, np.uint8))
            assert ev.kind == "visual_change_global" and ev.regions == (Region(0.0, 0.0, 1.0, 1.0),)


def test_color_and_alpha_sources_are_compared_on_luma():
    with tempfile.TemporaryDirectory() as tmp:
        red = np.zeros((H, W, 4), np.uint8)
        red[..., 2], red[..., 3] = 255, 255  # BGRA red, opaque
        blue = red.copy()
        blue[100:200, 100:300, :3] = (255, 0, 0)  # BGR blue block
        ev = _pair(tmp, red, blue, threshold=0.05)
        assert ev.kind == "visual_change_localized" and ev.regions == (_region_of(100, 100, 300, 200),)


# ----------------------------- unavailable is never "no change" -----------------------------

def _assert_unavailable(ev: VisualChangeEvent, why: str):
    assert ev.status == "unavailable" and ev.magnitude is None
    assert ev.confidence == 0.0 and ev.regions == () and ev.kind == "visual_change"
    assert why in ev.detail, ev.detail


def test_frames_of_different_sizes_are_unavailable_not_resampled():
    with tempfile.TemporaryDirectory() as tmp:
        ev = _pair(tmp, _textured(), _textured(h=180, w=320))
        _assert_unavailable(ev, "frame sizes differ (640x360 vs 320x180)")


def test_missing_and_corrupt_frames_are_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        good = _frame(tmp, "good", _textured(), 0.0)
        missing = Frame(timestamp_sec=1.0, path=os.path.join(tmp, "nope.png"))
        corrupt_path = os.path.join(tmp, "corrupt.png")
        Path(corrupt_path).write_bytes(b"not an image")
        corrupt = Frame(timestamp_sec=1.0, path=corrupt_path)
        _assert_unavailable(compare_frames(good, missing), "could not be read")
        _assert_unavailable(compare_frames(good, corrupt), "could not be read")


def test_one_unreadable_frame_only_affects_the_pairs_that_include_it():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        frames = [_frame(tmp, "a", img, 0.0), _frame(tmp, "b", img, 1.0),
                  Frame(timestamp_sec=2.0, path=os.path.join(tmp, "gone.png")),
                  _frame(tmp, "d", img, 3.0)]
        assert [e.status for e in detect_visual_changes(frames)] == \
            ["not_detected", "unavailable", "unavailable"]


def test_missing_opencv_is_unavailable_not_no_change():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        a, b = _frame(tmp, "a", img, 0.0), _frame(tmp, "b", img, 1.0)
        with mock.patch.dict(sys.modules, {"cv2": None}):
            _assert_unavailable(compare_frames(a, b), "OpenCV (cv2) is not installed")


# ----------------------------- performance shape -----------------------------

def test_each_frame_is_read_once_and_only_neighbours_are_compared():
    with tempfile.TemporaryDirectory() as tmp:
        img = _textured()
        frames = [_frame(tmp, f"f{i}", img, float(i)) for i in range(6)]
        with mock.patch("cv2.imread", side_effect=cv2.imread) as reads:
            events = detect_visual_changes(frames)
        assert reads.call_count == 6 and len(events) == 5


# ----------------------------- contract -----------------------------

def test_visual_change_event_contract_rejects_inconsistent_states():
    region = (Region(0.0, 0.0, 0.5, 0.5),)
    ok = dict(timestamp_sec=2.0, compared_timestamps=(1.0, 2.0))
    VisualChangeEvent(kind="visual_change_localized", status="detected", magnitude=0.1, regions=region, **ok)
    VisualChangeEvent(kind="visual_change", status="not_detected", magnitude=0.01, **ok)
    VisualChangeEvent(kind="visual_change", status="unavailable", magnitude=None, **ok)
    bad = [
        dict(kind="visual_change", status="unavailable", magnitude=0.0, **ok),         # failure dressed as "no change"
        dict(kind="visual_change", status="not_detected", magnitude=None, **ok),
        dict(kind="visual_change", status="unavailable", magnitude=None, confidence=0.5, **ok),
        dict(kind="visual_change", status="detected", magnitude=0.2, regions=region, **ok),  # no extent
        dict(kind="visual_change_global", status="detected", magnitude=0.9, **ok),     # no region
        dict(kind="visual_change_global", status="not_detected", magnitude=0.01, **ok),
        dict(kind="visual_change", status="not_detected", magnitude=0.01, regions=region, **ok),
        dict(kind="scroll", status="detected", magnitude=0.5, regions=region, **ok),   # semantic label
        dict(kind="visual_change", status="maybe", magnitude=0.1, **ok),
        dict(kind="visual_change", status="not_detected", magnitude=1.5, **ok),
        dict(kind="visual_change", status="not_detected", magnitude=0.1,
             timestamp_sec=1.0, compared_timestamps=(1.0, 2.0)),                     # not the later frame
        dict(kind="visual_change", status="not_detected", magnitude=0.1,
             timestamp_sec=1.0, compared_timestamps=(2.0, 1.0)),                     # reversed
    ]
    for kwargs in bad:
        try:
            VisualChangeEvent(**kwargs)
            assert False, f"expected ValueError for {kwargs}"
        except ValueError:
            pass


# ----------------------------- 9. layering -----------------------------

def test_the_detector_stays_below_the_evidence_layer():
    """Schema 1.2 made "visual_change" an Evidence kind; the mapping lives in
    core/evidence.py (tests/test_evidence_schema.py). The detector itself must
    know nothing of evidence, knowledge or synthesis -- detector result ->
    Evidence is one-way. The original four stages are still always expected."""
    from core.knowledge import _ALL_STAGES
    assert _ALL_STAGES == ("transcript", "frame", "pointer", "vision")
    root = Path(__file__).resolve().parent.parent
    src = (root / "core/visual_change.py").read_text(encoding="utf-8")
    for upper in ("core.evidence", "core.knowledge", "core.synthesis", "Evidence("):
        assert upper not in src, upper


# ----------------------------- shared per-pixel rule -----------------------------

def test_pointer_detector_uses_the_shared_rule_with_identical_behaviour():
    from adapters.pointer import cursor_detector
    assert cursor_detector._motion_mask is changed_pixel_mask

    def legacy(a, b):  # cursor_detector._motion_mask before the extraction, verbatim
        if a.shape != b.shape:
            return None
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return mask

    rng = np.random.default_rng(7)
    for _ in range(20):
        a = rng.integers(0, 256, (37, 53), dtype=np.uint8)
        b = np.clip(a.astype(np.int16) + rng.integers(-40, 41, a.shape), 0, 255).astype(np.uint8)
        got, want = changed_pixel_mask(a, b), legacy(a, b)
        assert got.dtype == want.dtype and np.array_equal(got, want)
    assert changed_pixel_mask(np.zeros((2, 2), np.uint8), np.zeros((3, 2), np.uint8)) is None


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"  ok: {t.__name__}")
    print("All visual-change tests passed.")
