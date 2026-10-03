"""Step 8: the visual-change measurement stream is independent of keyframe
selection -- sampled on its own bounded time grid, carried whole on
AnalysisResult.visual_changes, and from there into the KnowledgePackage, the
session file and retrieval, even where no keyframe window is near.

Run: python -m pytest tests/test_visual_change_stream.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_lens
from adapters.frames.frame_extractor import FrameExtractor
from adapters.ingestion import ingest
from core.contracts import VisionObservation, VisualChangeEvent
from core.errors import FrameExtractionError
from core.evidence import build_analysis_result
from core.knowledge import build_knowledge_package, validate_knowledge_package
from core.retrieval import evidence_between, evidence_near
from core.session import SESSION_FORMAT_VERSION, build_session, load_session


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True)


def _red_then_blue(tmp, seconds=10):
    """Red for the first half, blue for the second: exactly one change, at the midpoint."""
    path = os.path.join(tmp, "rb.mp4")
    half = seconds / 2
    _ffmpeg("-f", "lavfi", "-i", f"color=c=red:s=160x120:r=10:d={half}",
            "-f", "lavfi", "-i", f"color=c=blue:s=160x120:r=10:d={half}",
            "-filter_complex", "concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", path)
    return path


def _keyframes_at(path, cache, *timestamps):
    """Stand-in keyframe selection that keeps only these moments."""
    video, ex = ingest(path), FrameExtractor(cache_dir=cache)
    return [ex.get_frame(video, t) for t in timestamps]


def _run(path, tmp, keyframes=None, **overrides):
    """process_video with the given keyframes (None = real selection); returns (package, session)."""
    settings = dict(vision_enabled=False, pointer_enabled=False, visual_change_enabled=True,
                    visual_change_interval_sec=1.0, output_dir=os.path.join(tmp, "out"),
                    session_dir=os.path.join(tmp, "sessions"), frame_cache_dir=os.path.join(tmp, "cache"))
    settings.update(overrides)
    patch = (mock.patch.object(video_lens, "select_keyframes", return_value=keyframes)
             if keyframes is not None else contextlib.nullcontext())
    with patch, contextlib.redirect_stderr(io.StringIO()):
        package = video_lens.process_video(path, video_lens.PipelineConfig(**settings))
    sessions = list(Path(tmp, "sessions").glob("*.session.json")) if settings["session_dir"] else []
    return package, (load_session(str(sessions[0])) if sessions else None)


def _pairs(items):
    return [tuple(json.loads(e.ref)["compared"]) for e in items if e.kind == "visual_change"]


GRID_1S = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0),
           (7.0, 8.0), (8.0, 9.0), (9.0, 9.9)]  # ...and the last decodable frame (10s at 10 fps)


# ---- the architectural separation ----

def test_a_change_between_two_keyframes_reaches_package_session_and_retrieval():
    """Keyframes at 0s and 9s only: their windows (+-1s) never see 2..8s. The
    red->blue change at 5s is measured anyway, and survives end to end."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        package, session = _run(path, tmp, keyframes=_keyframes_at(path, os.path.join(tmp, "k"), 0.0, 9.0))
    assert _pairs(package.visual_changes) == GRID_1S, "the whole grid, not the keyframe neighbourhoods"
    assert _pairs(session.evidence) == GRID_1S
    detected = [e for e in package.visual_changes if json.loads(e.ref)["status"] == "detected"]
    assert _pairs(detected) == [(4.0, 5.0)], "exactly the one real change"
    [near] = evidence_near(session, 5.0, 0.0, kinds={"visual_change"})
    assert json.loads(near.ref)["compared"] == [4.0, 5.0]
    assert len(evidence_between(session, 2.0, 8.0, kinds={"visual_change"})) == 7  # inclusive: 2..8s
    assert "visual_change" in package.processing.stages_available
    validate_knowledge_package(package)  # schema 1.2, unchanged
    assert package.processing.knowledge_schema_version == "1.2"
    assert session.format_version == SESSION_FORMAT_VERSION == 1
    assert session.analysis["visual_change_interval_sec"] == 1.0, "the generating interval is recorded"


def test_measurements_survive_even_when_keyframe_selection_fails():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        with mock.patch.object(video_lens, "select_keyframes", side_effect=FrameExtractionError("boom")):
            package, session = _run(path, tmp)
    assert _pairs(package.visual_changes) == GRID_1S and _pairs(session.evidence) == GRID_1S
    assert "visual_change" in package.processing.stages_available, "no window saw it; the stream did"


def test_the_stream_is_carried_whole_on_analysis_result_independent_of_query_timestamps():
    events = [VisualChangeEvent(timestamp_sec=float(t), kind="visual_change", status="not_detected",
                                magnitude=0.0, compared_timestamps=(t - 1.0, float(t)), detection_method="m",
                                confidence=0.9) for t in range(1, 10)]
    unavailable = VisualChangeEvent(timestamp_sec=9.5, kind="visual_change", status="unavailable", magnitude=None,
                                    compared_timestamps=(9.0, 9.5), detection_method="m")
    video = ingest_free_video()
    result = build_analysis_result(video, [0.0], visual_changes=[*events, unavailable], tolerance_sec=1.0)
    assert [e.timestamp_sec for e in result.visual_changes] == [float(t) for t in range(1, 10)]
    assert len([e for so in result.structured_observations for e in so.observed if e.kind == "visual_change"]) == 1
    package = build_knowledge_package(result)
    assert len(package.visual_changes) == 9, "window copy + stream copy = one item; unavailable never evidence"
    assert len([e for e in build_session(result).evidence if e.kind == "visual_change"]) == 9


def ingest_free_video():
    from core.contracts import VideoInput
    return VideoInput(path="v.mp4", duration_sec=10.0, width=160, height=120, fps=10.0, has_audio=False)


# ---- sampling semantics ----

def test_the_interval_is_configurable():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        package, _ = _run(path, tmp, visual_change_interval_sec=2.0, session_dir=None)
    assert _pairs(package.visual_changes) == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 9.9)]


def test_the_sample_budget_widens_the_step_never_exceeds_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        video, ex = ingest(path), FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        for budget in (2, 3, 4, 7, 150):
            config = video_lens.PipelineConfig(visual_change_interval_sec=1.0, visual_change_max_samples=budget)
            frames = video_lens._temporal_samples(video, ex, config)
            ts = [f.timestamp_sec for f in frames]
            assert 2 <= len(frames) <= budget and ts == sorted(set(ts)), (budget, ts)
            assert ts[0] == 0.0 and ts[-1] == 9.9, "always the whole video, first to last frame"


def test_short_videos_and_a_single_frame():
    with tempfile.TemporaryDirectory() as tmp:
        short = os.path.join(tmp, "short.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=0.5", "-pix_fmt", "yuv420p", short)
        package, _ = _run(short, tmp, visual_change_interval_sec=2.0, session_dir=None)
        assert _pairs(package.visual_changes) == [(0.0, 0.4)], "shorter than one step: first vs last frame"

        one = os.path.join(tmp, "one.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=0.1", "-pix_fmt", "yuv420p", one)
        package, _ = _run(one, os.path.join(tmp, "o"), session_dir=None)
    assert package.visual_changes == (), "one frame cannot show a change -- and no fake zero-change item"
    assert "visual_change" in package.processing.stages_unavailable


def test_the_end_of_the_video_is_the_extractors_last_decodable_frame():
    """Video stream 2.0s, audio 2.5s: the grid's end comes from the Step 7.5
    end-of-stream rule (1.96s), not from the container duration."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v.mp4")
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=2.0",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2.5",
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", path)
        package, _ = _run(path, tmp, session_dir=None)
    assert _pairs(package.visual_changes) == [(0.0, 1.0), (1.0, 1.96)]


def test_the_order_is_deterministic_and_nothing_is_duplicated():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        kf = _keyframes_at(path, os.path.join(tmp, "k"), 0.0, 4.5, 5.5, 9.0)  # windows overlapping the stream
        a, sa = _run(path, os.path.join(tmp, "a"), keyframes=kf)
        b, sb = _run(path, os.path.join(tmp, "b"), keyframes=kf)
    assert a.visual_changes == b.visual_changes and _pairs(sa.evidence) == _pairs(sb.evidence) == GRID_1S
    assert len(set(a.visual_changes)) == len(a.visual_changes) == len(GRID_1S)


# ---- degradation, defaults, the keyframe path ----

def test_a_failing_temporal_sample_degrades_to_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        with mock.patch.object(FrameExtractor, "extract_window", side_effect=FrameExtractionError("boom")):
            package, session = _run(path, tmp)
        assert package.visual_changes == () and "visual_change" in package.processing.stages_unavailable
        assert _pairs(session.evidence) == []
        package, _ = _run(path, os.path.join(tmp, "z"), visual_change_interval_sec=0.0, session_dir=None)
    assert package.visual_changes == () and "visual_change" in package.processing.stages_unavailable
    validate_knowledge_package(package)


def test_disabled_by_default_and_then_no_grid_is_sampled():
    assert video_lens.PipelineConfig().visual_change_enabled is False
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        with mock.patch.object(video_lens, "_temporal_samples", side_effect=AssertionError("sampled")):
            package, session = _run(path, tmp, visual_change_enabled=False)
    assert package.visual_changes == () and _pairs(session.evidence) == []
    assert not any("pixels, not meaning" in lim for lim in package.limitations)


class _RecordingVision:
    def __init__(self):
        self.seen = []

    def analyze_frame(self, frame, transcript_context=None, pointer=None):
        self.seen.append(frame.timestamp_sec)
        return VisionObservation(timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status="ok",
                                 description="a frame", confidence=0.5, model="fake")


def test_vision_sees_exactly_the_keyframes_never_the_measurement_grid():
    seen = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        kf = _keyframes_at(path, os.path.join(tmp, "k"), 0.0, 5.0, 9.0)
        for enabled in (False, True):
            vision = _RecordingVision()
            package, _ = _run(path, os.path.join(tmp, str(enabled)), keyframes=kf, vision_enabled=True,
                              vision_provider=vision, visual_change_enabled=enabled, session_dir=None)
            seen[enabled] = vision.seen
            assert package.processing.frame_count == 3, "the keyframe path is unchanged"
    assert seen[False] == seen[True] == [0.0, 5.0, 9.0]


def test_the_measurement_path_opens_no_network_connection():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        video, ex = ingest(path), FrameExtractor(cache_dir=os.path.join(tmp, "c"))
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network")):
            events = video_lens._measure_visual_changes(
                video, ex, video_lens.PipelineConfig(visual_change_interval_sec=1.0))
    assert [e.compared_timestamps for e in events] == GRID_1S


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
