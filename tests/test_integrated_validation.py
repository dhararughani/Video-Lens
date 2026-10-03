"""Step 9 integrated validation: cases the Step 7.5/8 suites did not already pin.

Run: python -m pytest tests/test_integrated_validation.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_lens
from adapters.frames import frame_extractor as fe
from adapters.frames.frame_extractor import FrameExtractor
from adapters.ingestion import ingest
from core.contracts import Transcript, TranscriptSegment, VideoInput, VisionObservation
from core.knowledge import validate_knowledge_package
from core.workspace import JOBS_ROOT


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", *args], capture_output=True, check=True)


def _clip(tmp, name, source, seconds, rate=10, size="160x120"):
    path = os.path.join(tmp, name)
    _ffmpeg("-f", "lavfi", "-i", f"{source}=s={size}:r={rate}:d={seconds}", "-pix_fmt", "yuv420p", path)
    return path


def _red_then_blue(tmp):
    path = os.path.join(tmp, "rb.mp4")
    _ffmpeg("-f", "lavfi", "-i", "color=c=red:s=160x120:r=10:d=5", "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=5",
            "-filter_complex", "concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", path)
    return path


class _Vision:
    def __init__(self, status="ok"):
        self.status, self.seen = status, []

    def analyze_frame(self, frame, transcript_context=None, pointer=None):
        self.seen.append(frame.timestamp_sec)
        return VisionObservation(timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status=self.status,
                                 description="a screen" if self.status == "ok" else "",
                                 confidence=0.8 if self.status == "ok" else 0.0, model="fake")


_SPEECH = Transcript(segments=[TranscriptSegment(0.0, 9.0, "the key point is that the screen turns blue")], language="en")


def _run(path, tmp, **overrides):
    settings = dict(pointer_enabled=False, vision_provider=_Vision(), output_dir=os.path.join(tmp, "out"),
                    session_dir=os.path.join(tmp, "s"), visual_change_interval_sec=1.0)
    settings.update(overrides)
    with mock.patch.object(video_lens, "_try_transcribe", return_value=_SPEECH), \
         contextlib.redirect_stderr(io.StringIO()):
        return video_lens.process_video(path, video_lens.PipelineConfig(**settings))


def _pairs(items):
    return [tuple(json.loads(e.ref)["compared"]) for e in items if e.kind == "visual_change"]


def test_disabling_the_stream_removes_the_between_keyframes_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        ex = FrameExtractor(cache_dir=os.path.join(tmp, "k"))
        keyframes = [ex.get_frame(ingest(path), t) for t in (0.0, 9.0)]
        with mock.patch.object(video_lens, "select_keyframes", return_value=keyframes):
            on = _run(path, os.path.join(tmp, "on"), visual_change_enabled=True)
            off = _run(path, os.path.join(tmp, "off"), visual_change_enabled=False)
    detected = [e for e in on.visual_changes if json.loads(e.ref)["status"] == "detected"]
    assert _pairs(detected) == [(4.0, 5.0)]
    assert off.visual_changes == () and not any(e.kind == "visual_change" for e in off.evidence)


def test_a_video_exactly_one_interval_long_is_one_pair_ending_on_its_last_frame():
    with tempfile.TemporaryDirectory() as tmp:
        path = _clip(tmp, "two.mp4", "testsrc2", 2.0)
        frames = video_lens._temporal_samples(ingest(path), FrameExtractor(cache_dir=tmp),
                                              video_lens.PipelineConfig(visual_change_interval_sec=2.0))
    assert [f.timestamp_sec for f in frames] == [0.0, 1.9]


def test_a_long_video_never_exceeds_the_cap_and_never_scans_frame_by_frame():
    """300s at 10 fps is 3000 frames; with a cap of 20 the grid is 20 seeks, not 3000."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _clip(tmp, "long.mp4", "testsrc2", 300, size="64x48")
        video = ingest(path)
        config = video_lens.PipelineConfig(visual_change_interval_sec=1.0, visual_change_max_samples=20)
        runs = []
        for cache in ("a", "b"):
            with mock.patch.object(fe.subprocess, "run", wraps=subprocess.run) as spy:
                frames = video_lens._temporal_samples(video, FrameExtractor(cache_dir=os.path.join(tmp, cache)), config)
            runs.append([f.timestamp_sec for f in frames])
            assert spy.call_count <= len(frames) + 2, "one seek per sample (+ at most the end-of-stream probe)"
    ts = runs[0]
    assert runs[0] == runs[1], "deterministic"
    assert 2 <= len(ts) <= 20 and ts == sorted(set(ts))
    assert ts[0] == 0.0 and ts[-1] == 299.9, "the final decodable frame is represented"
    step = 300 / 18
    assert all(abs((b - a) - step) < 0.01 for a, b in zip(ts[:-2], ts[1:-1])), "the step widened evenly"


def test_an_unusable_duration_degrades_without_raising():
    bad = VideoInput(path=__file__, duration_sec=0.0, width=160, height=120, fps=10.0, has_audio=False)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()) as err:
        events = video_lens._measure_visual_changes(bad, FrameExtractor(cache_dir=tmp), video_lens.PipelineConfig())
    assert events == [] and "visual change detection unavailable" in err.getvalue()


def test_a_visual_change_failure_leaves_transcript_vision_and_cleanup_intact():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        jobs_before = set(os.listdir(JOBS_ROOT)) if os.path.isdir(JOBS_ROOT) else set()
        off = _run(path, os.path.join(tmp, "off"), visual_change_enabled=False)
        with mock.patch.object(video_lens, "_temporal_samples", side_effect=RuntimeError("grid failed")):
            broken = _run(path, os.path.join(tmp, "broken"), visual_change_enabled=True)
        jobs_after = set(os.listdir(JOBS_ROOT)) if os.path.isdir(JOBS_ROOT) else set()
    validate_knowledge_package(broken)
    assert broken.visual_changes == () and "visual_change" in broken.processing.stages_unavailable
    for field in ("key_lessons", "important_observations", "evidence", "topics", "summary"):
        assert getattr(broken, field) == getattr(off, field), field
    assert {"transcript", "vision", "frame"} <= set(broken.processing.stages_available)
    assert jobs_after == jobs_before, "both jobs succeeded, so both workspaces were removed"


def test_an_unavailable_vision_provider_does_not_affect_the_measurement_stream():
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        package = _run(path, tmp, visual_change_enabled=True, vision_provider=_Vision(status="unavailable"))
    assert "vision" in package.processing.stages_unavailable
    assert len(package.visual_changes) == 10


def _exported(tmp):
    d = json.load(open(next(Path(tmp, "out").glob("*.json")), encoding="utf-8"))
    d["processing"].pop("generated_at")
    return d


def test_enabling_the_stream_changes_only_the_measurement_fields_of_the_package():
    """Same video, same deterministic providers, no synthesizer: on vs off differ
    in exactly visual_changes, one limitation note and stage coverage."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _red_then_blue(tmp)
        _run(path, os.path.join(tmp, "off"), visual_change_enabled=False)
        _run(path, os.path.join(tmp, "on"), visual_change_enabled=True)
        off, on = _exported(os.path.join(tmp, "off")), _exported(os.path.join(tmp, "on"))
    assert sorted(k for k in set(off) | set(on) if off.get(k) != on.get(k)) == ["limitations", "processing", "visual_changes"]
    assert [k for k in off["processing"] if off["processing"][k] != on["processing"][k]] == ["stages_available"]
    assert set(on["limitations"]) - set(off["limitations"]) == {
        next(lim for lim in on["limitations"] if "pixels, not meaning" in lim)}
    assert set(off["limitations"]) <= set(on["limitations"])


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
