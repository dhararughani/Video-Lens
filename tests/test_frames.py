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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import dedupe_frames, frames_for_segment, select_keyframes
from adapters.ingestion import ingest
from core.contracts import TranscriptSegment
from core.errors import FrameExtractionError


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


CONTRACT_TESTS = [
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
