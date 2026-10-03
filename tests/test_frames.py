"""Frame intelligence tests: direct extraction, windows, speech->frame
bridge, scene-aware selection, dedup, and cache behavior.

Generates synthetic media with ffmpeg (no committed binaries). Run:
    python tests/test_frames.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from adapters.frames.frame_extractor import JPEG_COLOR_FILTER, FrameExtractor, _even_span, _plan_transform
from adapters.frames.keyframe_selector import dedupe_frames, frames_for_segment, select_keyframes
from adapters.ingestion import ingest
from core.contracts import InspectionRequest, Region, TranscriptSegment
from core.errors import FrameExtractionError
from video_lens import inspect_visual_evidence


def _color_video(path: str, colors: list[str], seconds_each: float = 1.0, size: str = "64x64"):
    """A video whose color changes every `seconds_each` seconds, in order."""
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, c in enumerate(colors):
            cf = os.path.join(tmp, f"c{i}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi",
                 "-i", f"color=c={c}:size={size}:rate=5:duration={seconds_each}",
                 "-pix_fmt", "yuv420p", cf],
                capture_output=True, check=True,
            )
            clips.append(cf)
        listfile = os.path.join(tmp, "list.txt")
        with open(listfile, "w") as f:
            for cf in clips:
                f.write(f"file '{cf}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
             "-pix_fmt", "yuv420p", path],
            capture_output=True, check=True,
        )


def _simple_video(path: str, duration: float = 5.0, rate: int = 30):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=64x64:rate={rate}",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )


# --------------------------- direct extraction ---------------------------

def test_get_frame_valid_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=5.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))

        for ts in (0.0, 2.5, 4.9):
            f = ex.get_frame(video, ts)
            assert os.path.exists(f.path) and os.path.getsize(f.path) > 0
            assert abs(f.timestamp_sec - ts) < 0.01
            assert f.width == 64 and f.height == 64
            assert f.source_video == video.path
            assert f.reason == "direct_timestamp"


def test_get_frame_beyond_duration_is_clamped():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=3.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))

        f = ex.get_frame(video, 999.0)
        assert f.timestamp_sec < video.duration_sec
        assert os.path.exists(f.path)


def test_get_frame_near_end_low_fps_clamps_to_last_real_frame():
    # A low-fps video has no frame between its last one and its reported
    # duration (5fps -> frames every 0.2s, last at 4.8s on a 5.0s video).
    # Asking for a timestamp in that gap must clamp to the last real frame,
    # not fail -- see the frame_interval comment in FrameExtractor.get_frame.
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=5.0, rate=5)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))

        f = ex.get_frame(video, 4.9)
        assert f.timestamp_sec <= 4.8 + 0.01
        assert os.path.exists(f.path) and os.path.getsize(f.path) > 0


def test_get_frame_negative_timestamp_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        try:
            ex.get_frame(video, -1.0)
            assert False, "expected ValueError for negative timestamp"
        except ValueError:
            pass


def test_get_frame_invalid_video_raises():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        os.remove(video.path)
        try:
            ex.get_frame(video, 0.5)
            assert False, "expected FrameExtractionError for a missing video file"
        except FrameExtractionError:
            pass


# --------------------------- frame windows ---------------------------

def test_extract_window_valid_range():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=5.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))

        frames = ex.extract_window(video, 1.0, 3.0, 0.5)
        starts = [f.timestamp_sec for f in frames]
        assert starts == sorted(starts)
        assert starts == [1.0, 1.5, 2.0, 2.5, 3.0]
        assert len(set(starts)) == len(starts)  # no duplicate timestamps
        assert all(f.reason == "window_sample" for f in frames)


def test_extract_window_invalid_range_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        try:
            ex.extract_window(video, 3.0, 1.0, 0.5)
            assert False, "expected ValueError for end <= start"
        except ValueError:
            pass


def test_extract_window_invalid_interval_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        try:
            ex.extract_window(video, 0.0, 2.0, 0.0)
            assert False, "expected ValueError for zero interval"
        except ValueError:
            pass


def test_extract_window_near_end_deduplicates_clamped_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=3.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        # requests several timestamps past the end -- they all clamp to the
        # same instant and must collapse to one frame, not duplicates.
        frames = ex.extract_window(video, 2.0, 10.0, 1.0)
        starts = [f.timestamp_sec for f in frames]
        assert len(set(starts)) == len(starts)


# --------------------------- speech -> frame bridge ---------------------------

def test_frames_for_segment_stay_within_and_around_segment():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=10.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        segment = TranscriptSegment(start_sec=3.0, end_sec=6.0, text="look at this")

        frames = frames_for_segment(video, segment, ex, interval_sec=1.0, scene_timestamps=[])
        starts = [f.timestamp_sec for f in frames]
        assert starts == sorted(starts)
        assert all(segment.start_sec <= s <= segment.end_sec for s in starts)
        assert segment.start_sec in starts and segment.end_sec in starts


def test_frames_for_segment_rejects_zero_duration_segment():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        segment = TranscriptSegment(start_sec=2.0, end_sec=2.0, text="x")
        try:
            frames_for_segment(video, segment, ex, scene_timestamps=[])
            assert False, "expected ValueError for zero-duration segment"
        except ValueError:
            pass


# --------------------------- scene-aware selection ---------------------------

def test_frames_for_segment_includes_known_scene_change():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        # solid red for 3s then solid blue for 3s -- one unambiguous scene change at 3.0s
        _color_video(p, ["red", "blue"], seconds_each=3.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        segment = TranscriptSegment(start_sec=1.0, end_sec=5.0, text="watch the color change")

        frames = frames_for_segment(video, segment, ex, interval_sec=5.0)
        reasons = {f.timestamp_sec: f.reason for f in frames}
        scene_frames = [ts for ts, r in reasons.items() if r == "scene_change"]
        assert scene_frames, f"expected a scene_change frame near 3.0s, got {reasons}"
        assert any(abs(ts - 3.0) < 0.5 for ts in scene_frames)


# --------------------------- dedup ---------------------------

def test_dedupe_frames_reduces_static_stretch():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        # static, static, visual change, static. Blue, not green: ffmpeg's
        # named "red" and "green" have nearly identical grayscale luma
        # (measured ~76 vs ~75), which a luma-based diff genuinely can't
        # separate -- see the hue-blindness limitation in docs/frames.md.
        # Blue's luma (~29) is unambiguously different.
        _color_video(p, ["red", "red", "blue", "red"], seconds_each=1.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))
        # 0.5s offset lands mid-clip, away from the ambiguous exact-second
        # boundaries where the concat transition itself sits.
        frames = ex.extract_window(video, 0.5, 3.5, 1.0)
        assert len(frames) == 4

        deduped = dedupe_frames(frames)
        assert len(deduped) < len(frames), "static stretch should be reduced"
        assert len(deduped) >= 2  # at least the red->blue transition survives


def test_select_keyframes_reduces_and_covers_scene_change():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _color_video(p, ["red", "red", "red", "blue", "blue", "blue"], seconds_each=1.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))

        keyframes = select_keyframes(video, ex, interval_sec=0.9)  # avoids exact-second clip boundaries
        assert len(keyframes) < 6, "6 near-identical seconds should collapse to fewer keyframes"
        assert any(2.5 <= f.timestamp_sec <= 3.5 for f in keyframes), \
            "a keyframe should survive near the red->blue transition at 3.0s"


# --------------------------- cache ---------------------------

def test_cache_reuses_identical_request():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=5.0)
        video = ingest(p)
        cache_dir = os.path.join(tmp, "cache")
        ex = FrameExtractor(cache_dir=cache_dir)

        f1 = ex.get_frame(video, 2.0)
        mtime1 = os.path.getmtime(f1.path)
        f2 = ex.get_frame(video, 2.0)
        assert f1.path == f2.path
        assert os.path.getmtime(f2.path) == mtime1, "second request must not re-extract"


def test_cache_keys_dont_collide_across_videos():
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "a.mp4")
        p2 = os.path.join(tmp, "b.mp4")
        _color_video(p1, ["red"], seconds_each=2.0)
        _color_video(p2, ["blue"], seconds_each=2.0)
        cache_dir = os.path.join(tmp, "cache")
        ex = FrameExtractor(cache_dir=cache_dir)

        fa = ex.get_frame(ingest(p1), 1.0)
        fb = ex.get_frame(ingest(p2), 1.0)
        assert fa.path != fb.path


def test_cache_regenerates_empty_cached_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=5.0)
        video = ingest(p)
        cache_dir = os.path.join(tmp, "cache")
        ex = FrameExtractor(cache_dir=cache_dir)

        f1 = ex.get_frame(video, 1.0)
        with open(f1.path, "wb"):  # simulate a killed-mid-write 0-byte cache entry
            pass
        assert os.path.getsize(f1.path) == 0

        f2 = ex.get_frame(video, 1.0)
        assert os.path.getsize(f2.path) > 0, "a 0-byte cache entry must be regenerated, not trusted"


# --------------------------- default-behaviour regression baseline ---------------------------

def test_default_extraction_command_and_pixels_are_unchanged():
    """Targeted inspection (scale/region) must be purely additive: a caller that
    passes neither gets the default ffmpeg invocation and, for an ordinary
    (untagged / BT.601) source, the exact bytes the extractor produced before
    any of this existed. Step 8 added one option to that invocation, the JPEG
    colour-matrix conversion; on this source it is a byte-for-byte no-op (BT.709
    sources are what it changes -- tests/test_benchmark_findings.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _simple_video(p, duration=5.0)
        video = ingest(p)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "cache"))

        legacy = ["ffmpeg", "-y", "-ss", "2.0", "-i", video.path,
                  "-frames:v", "1", "-q:v", "2", "-pix_fmt", "yuvj420p"]
        default = legacy[:-2] + ["-vf", f"scale={JPEG_COLOR_FILTER}"] + legacy[-2:]
        captured = []
        real_run = subprocess.run

        def spy(cmd, *args, **kwargs):
            captured.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        with mock.patch("adapters.frames.frame_extractor.subprocess.run", spy):
            frame = ex.get_frame(video, 2.0)

        assert len(captured) == 1
        assert captured[0][:-1] == default, "default ffmpeg argv must gain nothing but the colour conversion"
        assert captured[0][-1] == frame.path

        reference = os.path.join(tmp, "legacy.jpg")
        real_run(legacy + [reference], capture_output=True, check=True)
        assert Path(reference).read_bytes() == Path(frame.path).read_bytes()
        assert (frame.width, frame.height) == (video.width, video.height)
        assert frame.reason == "direct_timestamp"


# --------------------------- targeted inspection (P0-A) ---------------------------

def _quadrant_video(path: str, duration: float = 4.0, rate: int = 10):
    """160x80 with four unmistakable quadrants: top-left red, top-right blue,
    bottom-left red, bottom-right lime. Every boundary falls on an even pixel
    (x=80, y=40), so a correctly mapped crop contains exactly one colour."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c=red:size=160x80:rate={rate}:duration={duration}",
         "-vf", "drawbox=x=80:y=0:w=80:h=80:color=blue:t=fill,"
                "drawbox=x=80:y=40:w=80:h=40:color=lime:t=fill",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )


def _dims(path: str) -> tuple[int, int]:
    img = cv2.imread(path)
    return img.shape[1], img.shape[0]


def _mean_bgr(path: str):
    return cv2.imread(path).reshape(-1, 3).mean(axis=0)


class _FfmpegSpy:
    """Records every ffmpeg argv the extractor runs, still running them for real."""
    def __enter__(self):
        self.calls: list[list[str]] = []
        real_run = subprocess.run

        def spy(cmd, *args, **kwargs):
            self.calls.append(list(cmd))
            return real_run(cmd, *args, **kwargs)

        self._patch = mock.patch("adapters.frames.frame_extractor.subprocess.run", spy)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def _quadrant(tmp: str):
    p = os.path.join(tmp, "q.mp4")
    _quadrant_video(p)
    video = ingest(p)
    assert (video.width, video.height) == (160, 80)
    return video, FrameExtractor(cache_dir=os.path.join(tmp, "cache"))


def test_even_span_maps_normalized_span_outward_to_even_pixels():
    assert _even_span(0.0, 1.0, 160) == (0, 160)
    assert _even_span(0.5, 1.0, 160) == (80, 80)
    assert _even_span(0.0, 0.5, 90) == (0, 46)       # 45 rounds OUTWARD, never cuts the region
    assert _even_span(0.7, 0.9, 10) == (6, 4)        # 0.7*10 == 7.000000000000001 must not ceil to 8
    assert _even_span(0.5, 0.5001, 100) == (50, 2)   # a sliver still yields a valid crop
    assert _even_span(0.9, 1.0, 5) == (2, 2)         # odd size, at the edge: stays inside the frame
    assert _even_span(0.0, 1.0, 161) == (0, 160)     # odd size: even-aligned, never past the edge


def test_plan_transform_default_and_noop_requests_resolve_to_the_default():
    default = _plan_transform(160, 90, None, None)
    assert default.crop is None and default.scale is None and default.size == (160, 90)
    assert default.filter_chain() == f"scale={JPEG_COLOR_FILTER}"  # colour conversion only
    assert _plan_transform(160, 90, None, Region(0.0, 0.0, 1.0, 1.0)) == default  # full-frame crop
    assert _plan_transform(160, 90, 160, None) == default                          # scale to own width


def test_plan_transform_rejects_invalid_scale_and_uncroppable_frames():
    for bad in (0, -5, 1.5, True):
        try:
            _plan_transform(160, 90, bad, None)
            assert False, f"expected ValueError for scale_width={bad!r}"
        except ValueError:
            pass
    try:
        _plan_transform(1, 90, None, Region(0.0, 0.0, 0.5, 0.5))
        assert False, "a 1px-wide frame cannot be cropped"
    except ValueError:
        pass


def test_crop_is_mapped_from_normalized_region_to_the_right_pixels():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        expect = {  # region -> (output size, dominant channel index in BGR)
            Region(0.0, 0.0, 0.5, 0.5): ((80, 40), 2),   # top-left: red
            Region(0.5, 0.0, 1.0, 0.5): ((80, 40), 0),   # top-right: blue
            Region(0.5, 0.5, 1.0, 1.0): ((80, 40), 1),   # bottom-right: lime
        }
        for region, (size, channel) in expect.items():
            f = ex.get_frame(video, 1.0, region=region)
            assert (f.width, f.height) == size == _dims(f.path), "Frame size must match the real pixels"
            mean = _mean_bgr(f.path)
            assert mean[channel] > 200 and sum(mean) - mean[channel] < 120, (region, mean)


def test_scale_preserves_aspect_ratio_and_reports_real_dimensions():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        for width, expected in ((80, (80, 40)), (40, (40, 20)), (320, (320, 160)), (101, (101, 51))):
            f = ex.get_frame(video, 1.0, scale_width=width)
            assert (f.width, f.height) == expected == _dims(f.path)
        # a non-trivial ratio: 160x80 cropped to its right half is 80x80, so it stays square
        f = ex.get_frame(video, 1.0, region=Region(0.5, 0.0, 1.0, 1.0), scale_width=30)
        assert (f.width, f.height) == (30, 30) == _dims(f.path)


def test_region_and_scale_use_one_ffmpeg_pass_with_crop_before_scale():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        with _FfmpegSpy() as spy:
            f = ex.get_frame(video, 1.0, region=Region(0.5, 0.0, 1.0, 0.5), scale_width=40)
        assert len(spy.calls) == 1, "crop+scale must be a single ffmpeg process"
        argv = spy.calls[0]
        assert argv[argv.index("-vf") + 1] == f"crop=80:40:80:0,scale=40:20:{JPEG_COLOR_FILTER}"
        assert (f.width, f.height) == (40, 20) == _dims(f.path)
        mean = _mean_bgr(f.path)
        assert mean[0] > 200, "cropped-then-scaled top-right quadrant must still be blue"


def test_different_scale_widths_never_share_a_cache_entry():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        paths = {w: ex.get_frame(video, 1.0, scale_width=w).path for w in (None, 80, 120, 40)}
        assert len(set(paths.values())) == 4
        # and each still holds ITS pixels, regardless of request order
        for w, expected in ((None, 160), (80, 80), (120, 120), (40, 40)):
            assert _dims(paths[w])[0] == expected
            assert ex.get_frame(video, 1.0, scale_width=w).path == paths[w]


def test_different_regions_never_share_a_cache_entry():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        regions = [Region(0.0, 0.0, 0.5, 0.5), Region(0.5, 0.0, 1.0, 0.5),
                   Region(0.0, 0.5, 0.5, 1.0), Region(0.5, 0.5, 1.0, 1.0)]
        paths = [ex.get_frame(video, 1.0, region=r).path for r in regions]
        assert len(set(paths)) == 4
        means = [_mean_bgr(p) for p in paths]
        assert [int(m.argmax()) for m in means] == [2, 0, 2, 1]  # red, blue, red, lime


def test_full_frame_and_cropped_frame_do_not_collide_in_either_order():
    for crop_first in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            video, ex = _quadrant(tmp)
            crop = Region(0.5, 0.0, 1.0, 0.5)
            if crop_first:
                cropped = ex.get_frame(video, 1.0, region=crop)
                full = ex.get_frame(video, 1.0)
            else:
                full = ex.get_frame(video, 1.0)
                cropped = ex.get_frame(video, 1.0, region=crop)
            assert full.path != cropped.path
            assert _dims(full.path) == (160, 80) and _dims(cropped.path) == (80, 40)


def test_extraction_version_is_part_of_the_cache_identity():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        before = ex.get_frame(video, 1.0).path
        with mock.patch("adapters.frames.frame_extractor.FRAME_EXTRACTION_VERSION", "next"):
            after = ex.get_frame(video, 1.0).path
        assert before != after, "bumping the extraction version must invalidate old entries"


def test_equivalent_requests_share_one_deterministic_cache_identity():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex_a = _quadrant(tmp)
        ex_b = FrameExtractor(cache_dir=os.path.join(tmp, "other_cache"))
        name = lambda f: os.path.basename(f.path)  # noqa: E731 -- digest only, cache dirs differ

        a = ex_a.get_frame(video, 1.0, region=Region(0.5, 0.0, 1.0, 0.5), scale_width=40)
        b = ex_b.get_frame(video, 1.0, region=Region(0.5, 0.0, 1.0, 0.5), scale_width=40)
        assert name(a) == name(b), "same request must hash identically across extractors/runs"

        default = ex_a.get_frame(video, 1.0)
        assert name(ex_a.get_frame(video, 1.0, region=Region(0.0, 0.0, 1.0, 1.0))) == name(default)
        assert name(ex_a.get_frame(video, 1.0, scale_width=160)) == name(default)
        # two regions that resolve to the same pixel rectangle ARE the same image
        assert name(ex_a.get_frame(video, 1.0, region=Region(0.5, 0.0, 1.0, 0.5))) == \
            name(ex_a.get_frame(video, 1.0, region=Region(0.5001, 0.0, 1.0, 0.5)))


def test_transformed_request_beyond_duration_clamps_like_a_default_one():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        plain = ex.get_frame(video, 999.0)
        scaled = ex.get_frame(video, 999.0, scale_width=40, region=Region(0.5, 0.0, 1.0, 0.5))
        assert scaled.timestamp_sec == plain.timestamp_sec < video.duration_sec
        assert os.path.getsize(scaled.path) > 0 and _dims(scaled.path) == (40, 20)


def test_inspect_single_frame_returns_the_requested_moment():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        request = InspectionRequest(timestamp_sec=2.0)
        result = inspect_visual_evidence(video, request, ex)
        assert len(result.frames) == 1
        frame = result.frames[0]
        assert abs(frame.timestamp_sec - 2.0) < 0.01 and frame.reason == "direct_timestamp"
        assert (frame.width, frame.height) == (160, 80)
        # same frame the extractor itself would hand back -- no parallel path
        assert frame.path == ex.get_frame(video, 2.0).path


def test_inspect_single_frame_past_the_end_is_clamped_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        result = inspect_visual_evidence(video, InspectionRequest(timestamp_sec=999.0), ex)
        assert len(result.frames) == 1 and result.frames[0].timestamp_sec < video.duration_sec


def test_inspect_window_samples_before_through_after():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        request = InspectionRequest(timestamp_sec=2.0, window_before_sec=0.5,
                                    window_after_sec=0.5, fps=4.0)
        frames = inspect_visual_evidence(video, request, ex).frames
        assert [round(f.timestamp_sec, 2) for f in frames] == [1.5, 1.75, 2.0, 2.25, 2.5]
        assert all(f.reason == "window_sample" for f in frames)


def test_inspect_fps_controls_frame_density():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        def count(fps):
            req = InspectionRequest(timestamp_sec=2.0, window_before_sec=1.0,
                                    window_after_sec=1.0, fps=fps)
            return len(inspect_visual_evidence(video, req, ex).frames)
        assert (count(1.0), count(2.0), count(4.0)) == (3, 5, 9)


def test_inspect_window_is_clamped_to_the_video():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        early = inspect_visual_evidence(video, InspectionRequest(
            timestamp_sec=0.2, window_before_sec=1.0, window_after_sec=0.2, fps=5.0), ex).frames
        assert early[0].timestamp_sec == 0.0, "a window starting before 0 clamps to 0"
        late = inspect_visual_evidence(video, InspectionRequest(
            timestamp_sec=3.9, window_after_sec=50.0, fps=2.0), ex).frames
        assert all(f.timestamp_sec < video.duration_sec for f in late)
        beyond = inspect_visual_evidence(video, InspectionRequest(
            timestamp_sec=999.0, window_before_sec=1.0, fps=2.0), ex).frames
        assert len(beyond) == 1 and beyond[0].timestamp_sec < video.duration_sec


def test_inspect_applies_region_and_scale_to_every_frame_of_a_window():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        request = InspectionRequest(timestamp_sec=2.0, window_before_sec=0.5, window_after_sec=0.5,
                                    fps=2.0, scale_width=40, region=Region(0.5, 0.0, 1.0, 0.5))
        frames = inspect_visual_evidence(video, request, ex).frames
        assert len(frames) == 3 and len({f.path for f in frames}) == 3
        for f in frames:
            assert (f.width, f.height) == (40, 20) == _dims(f.path)
            assert _mean_bgr(f.path)[0] > 200


def test_inspection_result_preserves_request_and_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        request = InspectionRequest(timestamp_sec=2.0, window_after_sec=0.5, fps=2.0,
                                    scale_width=80, region=Region(0.0, 0.0, 0.5, 1.0))
        result = inspect_visual_evidence(video, request, ex)
        assert result.request is request
        assert result.video_source == video.path and result.extraction_method == "ffmpeg_seek"
        assert all(f.source_video == video.path for f in result.frames)
        assert isinstance(result.frames, tuple)


def test_repeated_and_overlapping_inspection_reuses_the_cache():
    with tempfile.TemporaryDirectory() as tmp:
        video, ex = _quadrant(tmp)
        request = InspectionRequest(timestamp_sec=2.0, window_before_sec=0.5,
                                    window_after_sec=0.5, fps=4.0, scale_width=80)
        with _FfmpegSpy() as spy:
            first = inspect_visual_evidence(video, request, ex)
            assert len(spy.calls) == len(first.frames) == 5
            again = inspect_visual_evidence(video, request, ex)
            assert len(spy.calls) == 5, "an identical request must not run ffmpeg again"
            overlap = inspect_visual_evidence(video, InspectionRequest(
                timestamp_sec=2.0, window_after_sec=1.0, fps=4.0, scale_width=80), ex)
            assert len(spy.calls) == 5 + 2, "only the two frames not already cached are extracted"
        assert [f.path for f in again.frames] == [f.path for f in first.frames]
        assert len(overlap.frames) == 5


def test_inspect_without_an_extractor_uses_the_default_temp_cache():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "q.mp4")
        _quadrant_video(p)
        video = ingest(p)
        result = inspect_visual_evidence(video, InspectionRequest(timestamp_sec=1.0))
        assert os.path.exists(result.frames[0].path)
        assert not result.frames[0].path.startswith(tmp), "default cache must not live beside the video"


CONTRACT_TESTS = [
    test_default_extraction_command_and_pixels_are_unchanged,
    test_even_span_maps_normalized_span_outward_to_even_pixels,
    test_plan_transform_default_and_noop_requests_resolve_to_the_default,
    test_plan_transform_rejects_invalid_scale_and_uncroppable_frames,
    test_crop_is_mapped_from_normalized_region_to_the_right_pixels,
    test_scale_preserves_aspect_ratio_and_reports_real_dimensions,
    test_region_and_scale_use_one_ffmpeg_pass_with_crop_before_scale,
    test_different_scale_widths_never_share_a_cache_entry,
    test_different_regions_never_share_a_cache_entry,
    test_full_frame_and_cropped_frame_do_not_collide_in_either_order,
    test_extraction_version_is_part_of_the_cache_identity,
    test_equivalent_requests_share_one_deterministic_cache_identity,
    test_transformed_request_beyond_duration_clamps_like_a_default_one,
    test_inspect_single_frame_returns_the_requested_moment,
    test_inspect_single_frame_past_the_end_is_clamped_not_an_error,
    test_inspect_window_samples_before_through_after,
    test_inspect_fps_controls_frame_density,
    test_inspect_window_is_clamped_to_the_video,
    test_inspect_applies_region_and_scale_to_every_frame_of_a_window,
    test_inspection_result_preserves_request_and_provenance,
    test_repeated_and_overlapping_inspection_reuses_the_cache,
    test_inspect_without_an_extractor_uses_the_default_temp_cache,
    test_get_frame_valid_timestamps, test_get_frame_beyond_duration_is_clamped,
    test_get_frame_near_end_low_fps_clamps_to_last_real_frame,
    test_get_frame_negative_timestamp_rejected, test_get_frame_invalid_video_raises,
    test_extract_window_valid_range, test_extract_window_invalid_range_rejected,
    test_extract_window_invalid_interval_rejected,
    test_extract_window_near_end_deduplicates_clamped_timestamps,
    test_frames_for_segment_stay_within_and_around_segment,
    test_frames_for_segment_rejects_zero_duration_segment,
    test_frames_for_segment_includes_known_scene_change,
    test_dedupe_frames_reduces_static_stretch, test_select_keyframes_reduces_and_covers_scene_change,
    test_cache_reuses_identical_request, test_cache_keys_dont_collide_across_videos,
    test_cache_regenerates_empty_cached_file,
]

if __name__ == "__main__":
    for t in CONTRACT_TESTS:
        t()
        print(f"  ok: {t.__name__}")
    print("All frame tests passed.")
