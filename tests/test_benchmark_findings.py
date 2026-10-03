"""Regression tests for the four Step 7 benchmark findings (docs/benchmark.md),
fixed in Step 8. Each was first pinned here as a characterization test of the
broken behavior; each is now pinned the other way, on a small synthetic
fixture, so the fix cannot silently regress.

F1 end of video  - the clamp trusted the container duration, not the video stream
F2 max_frames    - kept the first N keyframes instead of spreading N over the video
F3 colour        - BT.709 sources were written to JPEG without a matrix conversion
F4 speech+pointer - "stationary" from uncertain readings / across undetected gaps

Run: python -m pytest tests/test_benchmark_findings.py
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
import numpy as np

import video_lens
from adapters.frames import frame_extractor as fe
from adapters.frames.frame_extractor import FrameExtractor
from adapters.ingestion import ingest
from core.contracts import Frame, PointerEvent, Region, Transcript, TranscriptSegment, VideoInput
from core.errors import FrameExtractionError
from core.evidence import build_structured_observation


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True)


class _Spy:
    """Every subprocess argv the extractor runs (ffmpeg and ffprobe), still run for real."""
    def __enter__(self):
        self.calls, real = [], subprocess.run

        def spy(cmd, *a, **kw):
            self.calls.append(list(cmd))
            return real(cmd, *a, **kw)
        self._p = mock.patch.object(fe.subprocess, "run", spy)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()

    def count(self, tool):
        return sum(c[0] == tool for c in self.calls)


# ---- F1. end of video ----

LAST = 1.96  # 2.0s at 25 fps: the last frame's pts


def _video_ending_before_audio(tmp):
    """Video 2.0s, audio 2.5s: the container duration (2.5s) runs past the last frame (1.96s)."""
    path = os.path.join(tmp, "v.mp4")
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=2.0",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2.5",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", path)
    video = ingest(path)
    assert video.duration_sec > 2.4, "fixture: ingestion reports the container duration"
    return video


def test_end_of_video_requests_at_and_past_the_end_return_the_last_decodable_frame():
    with tempfile.TemporaryDirectory() as tmp:
        video = _video_ending_before_audio(tmp)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        last = ex.get_frame(video, LAST)
        assert last.timestamp_sec == LAST, "the exact last-frame timestamp is served as itself"
        for ts in (LAST + 0.001, 1.99, 2.3, video.duration_sec, video.duration_sec + 10):
            f = ex.get_frame(video, ts)
            assert f.timestamp_sec == LAST and f.path == last.path, ts
        assert os.path.getsize(last.path) > 0
        # deterministic: a fresh extractor (empty memo) resolves the same way, to the same bytes
        again = FrameExtractor(cache_dir=os.path.join(tmp, "c2")).get_frame(video, video.duration_sec)
        assert again.timestamp_sec == LAST
        assert Path(again.path).read_bytes() == Path(last.path).read_bytes()


def test_end_of_video_a_last_frame_between_milliseconds_is_floored_onto_itself():
    """At 30 fps the last frame sits at 1.96667s: rounded to the nearest ms (1.967)
    the seek would land past it and fail again; floored (1.966) it finds it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v30.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30:duration=2.0",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2.5",
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", path)
        video = ingest(path)
        f = FrameExtractor(cache_dir=os.path.join(tmp, "c")).get_frame(video, video.duration_sec)
        assert f.timestamp_sec == 1.966 and os.path.getsize(f.path) > 0


def test_end_of_video_timestamps_before_the_end_are_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        video = _video_ending_before_audio(tmp)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        for ts in (0.0, 1.0, 1.9, 1.95):
            assert ex.get_frame(video, ts).timestamp_sec == ts
        ex.get_frame(video, video.duration_sec)  # learns the real end ...
        assert ex.get_frame(video, 1.95).timestamp_sec == 1.95, "... which never moves earlier requests"


def test_end_of_video_probes_once_and_only_when_a_request_actually_fails():
    with tempfile.TemporaryDirectory() as tmp:
        video = _video_ending_before_audio(tmp)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        with _Spy() as spy:
            for ts in (0.0, 0.5, 1.0, 1.5):
                ex.get_frame(video, ts)
        assert spy.count("ffprobe") == 0, "a normal request never probes"
        with _Spy() as spy:
            for ts in (2.4, 2.45, 3.0, 2.2, 9.0):
                ex.get_frame(video, ts)
        assert spy.count("ffprobe") == 1, "the real end is learned once per video"
        assert spy.count("ffmpeg") == 2, "one failed seek, one extraction; then served from cache"


def test_end_of_video_window_crossing_the_end_has_only_real_increasing_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        video = _video_ending_before_audio(tmp)
        frames = FrameExtractor(cache_dir=os.path.join(tmp, "c")).extract_window(video, 1.5, 3.0, 0.2)
        ts = [f.timestamp_sec for f in frames]
        assert ts == [1.5, 1.7, 1.9, LAST], ts
        assert all(os.path.getsize(f.path) > 0 for f in frames)


def test_end_of_video_when_the_video_stream_outlasts_the_audio_the_clamp_alone_suffices():
    """The other direction: container == video stream, so the duration - 1/fps
    clamp is already exact and nothing is probed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=2.0",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1.0",
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", path)
        video = ingest(path)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        with _Spy() as spy:
            assert ex.get_frame(video, video.duration_sec).timestamp_sec == LAST
            assert ex.get_frame(video, video.duration_sec + 5).timestamp_sec == LAST
        assert spy.count("ffprobe") == 0


def test_end_of_video_an_unexplained_empty_extraction_still_raises():
    """The fallback only applies when the stream really ends before the request;
    otherwise the original error stands (and nothing loops)."""
    with tempfile.TemporaryDirectory() as tmp:
        video = _video_ending_before_audio(tmp)
        for probed in (None, 2.3, 2.46):  # unreadable, or "ends" at/after the failing request
            ex = FrameExtractor(cache_dir=os.path.join(tmp, f"c{probed}"))
            with mock.patch.object(fe, "_last_frame_position", return_value=probed):
                try:
                    ex.get_frame(video, video.duration_sec)
                    assert False, f"must raise when the probe says {probed}"
                except FrameExtractionError as e:
                    assert "no frame" in str(e)


# ---- F2. max_frames spreads over the candidates ----

def _select(n, limit):
    video = VideoInput(path="v.mp4", duration_sec=2.0 * n, width=10, height=10, fps=5.0, has_audio=False)
    candidates = [Frame(timestamp_sec=2.0 * i, path=f"{i}.jpg") for i in range(n)]
    with mock.patch.object(video_lens, "select_keyframes", return_value=candidates):
        kept, _ = video_lens._try_select_frames(video, video_lens.PipelineConfig(max_frames=limit))
    return candidates, kept


def test_max_frames_under_and_at_budget_keeps_every_candidate():
    for n in (0, 1, 5, 20):
        candidates, kept = _select(n, 20)
        assert kept == candidates


def test_max_frames_over_budget_spreads_across_the_whole_candidate_range():
    candidates, kept = _select(50, 20)  # 0..98s
    ts = [f.timestamp_sec for f in kept]
    assert len(kept) == 20 and all(f in candidates for f in kept), "only real candidates"
    assert ts == sorted(set(ts)), "chronological, no duplicates"
    assert ts[0] == 0.0 and ts[-1] == 98.0, "first and last candidates kept"
    for q in range(4):  # every quarter of the video is represented
        assert any(q * 25 <= t < (q + 1) * 25 for t in ts), q
    assert max(b - a for a, b in zip(ts, ts[1:])) <= 6.0, "no gap wider than the even step allows"
    assert _select(50, 20)[1] == kept, "deterministic"


def test_max_frames_tiny_budgets():
    candidates, kept = _select(50, 2)
    assert kept == [candidates[0], candidates[-1]]
    assert _select(50, 1)[1] == [candidates[0]]
    assert _select(50, 0)[1] == []
    assert len(_select(21, 20)[1]) == 20 and len(_select(1000, 19)[1]) == 19


# ---- F3. JPEG colour matrix ----

_TAGS = {"bt709": "bt709", "bt601": "smpte170m"}


def _encode(tmp, name, source, matrix):
    """A tiny clip from a lavfi `source`, encoded with `matrix` ("bt709"/"bt601")
    and tagged with it -- or untagged (encoded BT.601) when matrix is None."""
    path = os.path.join(tmp, f"{name}.mp4")
    tag = _TAGS.get(matrix)
    tags = ["-colorspace", tag, "-color_primaries", tag, "-color_trc", tag, "-color_range", "tv"] if tag else []
    _ffmpeg("-f", "lavfi", "-i", f"{source}:size=160x120:rate=5:duration=1",
            "-vf", f"scale=out_color_matrix={matrix or 'bt601'}:out_range=tv,format=yuv420p",
            *tags, "-c:v", "libx264", path)
    return ingest(path)


SATURATED = "color=c=0xC030E0"
NEUTRAL_RAMP = "gradients=c0=0x101010:c1=0xe0e0e0:nb_colors=2"
_KR_KB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114), None: (0.299, 0.114)}


def _oracle(video, matrix, t=0.2):
    """Independent of ffmpeg's colour code: the decoder's raw yuv420p, converted
    to BGR in float with the matrix the clip is tagged with."""
    w, h = video.width, video.height
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", video.path, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
    planes = np.frombuffer(raw, np.uint8).astype(float)
    y = planes[:w * h].reshape(h, w)
    up = lambda p: cv2.resize(p.reshape(h // 2, w // 2), (w, h), interpolation=cv2.INTER_LINEAR)
    u, v = up(planes[w * h:w * h * 5 // 4]), up(planes[w * h * 5 // 4:w * h * 3 // 2])
    kr, kb = _KR_KB[matrix]
    Y, Pb, Pr = (y - 16) / 219, (u - 128) / 224, (v - 128) / 224
    R, B = Y + 2 * (1 - kr) * Pr, Y + 2 * (1 - kb) * Pb
    G = (Y - kr * R - kb * B) / (1 - kr - kb)
    return np.clip(np.stack([B, G, R], -1) * 255, 0, 255)


def _legacy_jpeg(tmp, video, name, t=0.2):
    """The pre-fix extraction command: yuvj420p with no matrix conversion."""
    out = os.path.join(tmp, f"{name}_legacy.jpg")
    _ffmpeg("-ss", str(t), "-i", video.path, "-frames:v", "1", "-q:v", "2", "-pix_fmt", "yuvj420p", out)
    return out


def _err(path, truth):
    d = cv2.imread(path).astype(float) - truth
    return float(np.abs(d).mean()), float(np.abs(d.reshape(-1, 3).mean(axis=0)).max())  # MAD, worst channel bias


# Tolerances, against the exact float decode above. An extracted frame goes
# through two 8-bit roundings (YUV->BT.601 full-range YUV) and JPEG q=2, so it
# can't be exact: measured MAD 0.97 on the flat saturated colour and channel
# bias 0.1 on the neutral ramp. The broken behaviours this guards against sit far
# outside: no matrix conversion, MAD 15.6 on the saturated colour; the matrix
# with swscale's default fast rounding, bias 2.35 on the ramp (1.05 with
# accurate_rnd but no full-chroma flags).
MAD_TOLERANCE = 2.0
BIAS_TOLERANCE = 0.75


def test_bt709_saturated_colour_matches_an_exact_decode_and_the_fixture_exposes_the_wrong_matrix():
    with tempfile.TemporaryDirectory() as tmp:
        video = _encode(tmp, "sat709", SATURATED, "bt709")
        truth = _oracle(video, "bt709")
        assert _err(_legacy_jpeg(tmp, video, "sat709"), truth)[0] > 8, "fixture must expose the wrong matrix"
        ours = FrameExtractor(cache_dir=os.path.join(tmp, "c")).get_frame(video, 0.2).path
        assert _err(ours, truth)[0] < MAD_TOLERANCE


def test_bt709_neutral_content_gains_no_systematic_bias():
    """Neutral content was (almost) right before the fix, so the fix must not make
    it worse -- which the matrix conversion alone, at default rounding, did."""
    with tempfile.TemporaryDirectory() as tmp:
        video = _encode(tmp, "ramp709", NEUTRAL_RAMP, "bt709")
        ours = FrameExtractor(cache_dir=os.path.join(tmp, "c")).get_frame(video, 0.2).path
        mad, bias = _err(ours, _oracle(video, "bt709"))
        assert bias < BIAS_TOLERANCE and mad < MAD_TOLERANCE, (mad, bias)


def test_bt601_and_untagged_sources_are_byte_identical_to_before():
    """An ordinary source is untouched: the conversion is 601->601, a no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, matrix in (("sat601", "bt601"), ("untagged", None)):
            video = _encode(tmp, name, SATURATED, matrix)
            ours = FrameExtractor(cache_dir=os.path.join(tmp, name)).get_frame(video, 0.2).path
            assert Path(ours).read_bytes() == Path(_legacy_jpeg(tmp, video, name)).read_bytes(), name
            assert _err(ours, _oracle(video, matrix))[0] < MAD_TOLERANCE, name


def test_bt709_crop_and_scale_keep_the_corrected_colour_and_cache_identity():
    with tempfile.TemporaryDirectory() as tmp:
        video = _encode(tmp, "sat709", SATURATED, "bt709")
        truth = _oracle(video, "bt709").reshape(-1, 3).mean(axis=0)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        region = Region(0.25, 0.25, 0.75, 0.75)
        variants = {"full": {}, "crop": {"region": region}, "scale": {"scale_width": 40},
                    "both": {"region": region, "scale_width": 40}}
        paths = {k: ex.get_frame(video, 0.2, **kw).path for k, kw in variants.items()}
        assert len(set(paths.values())) == 4, "each variant is its own cache entry"
        for k, p in paths.items():
            assert np.abs(cv2.imread(p).reshape(-1, 3).mean(axis=0) - truth).max() < MAD_TOLERANCE, k
        with _Spy() as spy:
            again = {k: ex.get_frame(video, 0.2, **kw).path for k, kw in variants.items()}
        assert again == paths and spy.calls == [], "repeats are served from cache"
        # a frame cached under the old (uncorrected) rule is never served as a new one
        with mock.patch.object(fe, "FRAME_EXTRACTION_VERSION", "2"):
            assert ex._cache_path(video, 0.2) != paths["full"]


# ---- F4. speech + pointer: motion from detected readings only, never stationary ----

def _speech(t0=0.0, t1=10.0):
    return Transcript(segments=[TranscriptSegment(t0, t1, "some speech")], language="en")


def _pe(t, status, x=None, y=None):
    if status == "not_detected":
        return PointerEvent(timestamp_sec=t, status=status, frame_width=1000, frame_height=1000)
    return PointerEvent(timestamp_sec=t, status=status, x=x, y=y, frame_width=1000, frame_height=1000,
                        confidence=0.9 if status == "detected" else 0.6)


def _inferences(events):
    so = build_structured_observation(5.0, transcript=_speech(), pointer_events=events, tolerance_sec=1.0)
    return [i for i in so.inferences if i.basis.startswith("speech_pointer")]


def test_uncertain_readings_never_create_a_speech_pointer_inference():
    assert _inferences([_pe(4.5, "uncertain", 500, 500), _pe(5.5, "uncertain", 502, 501)]) == []
    jumpy = [_pe(4.5, "uncertain", 100, 100), _pe(5.0, "uncertain", 900, 900), _pe(5.5, "uncertain", 700, 100)]
    assert _inferences(jumpy) == []


def test_a_reading_that_is_not_detected_between_detected_ones_blocks_the_inference():
    assert _inferences([_pe(4.5, "detected", 100, 100), _pe(5.0, "not_detected"), _pe(5.5, "detected", 900, 900)]) == []
    assert _inferences([_pe(4.5, "detected", 100, 100), _pe(5.0, "uncertain", 500, 500), _pe(5.5, "detected", 900, 900)]) == []


def test_small_displacement_between_detected_readings_is_never_called_stationary():
    assert _inferences([_pe(4.5, "detected", 500, 500), _pe(5.5, "detected", 505, 500)]) == []
    assert _inferences([_pe(4.5, "detected", 500, 500), _pe(5.5, "detected", 500, 500)]) == []


def test_absence_of_evidence_establishes_nothing():
    assert _inferences([_pe(4.5, "not_detected"), _pe(5.0, "not_detected"), _pe(5.5, "not_detected")]) == []
    assert _inferences([_pe(5.0, "detected", 500, 500)]) == [], "one reading cannot show anything"
    assert _inferences([]) == []


def test_motion_needs_consecutive_detected_readings_and_cites_only_them():
    events = [_pe(4.2, "uncertain", 10, 10), _pe(4.5, "detected", 100, 100), _pe(5.0, "detected", 500, 500),
              _pe(5.5, "detected", 900, 900), _pe(5.8, "not_detected")]
    [inf] = _inferences(events)
    assert inf.basis == "speech_pointer_motion"
    cited = [e.timestamp_sec for e in inf.supporting_evidence if e.kind == "pointer"]
    assert cited == [4.5, 5.0, 5.5], "uncertain / not_detected readings outside the run are not cited"
    assert inf.confidence == round(3 / 5, 4), "completeness: detected run over all readings in the window"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
