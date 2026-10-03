"""P0-D tests: the optional native video-understanding seam
(`core.interfaces.VideoUnderstandingAdapter`) -- injection, timestamp
grounding, merge with per-frame vision, graceful failure, and the guarantee
that with no provider nothing changes.

The only provider here is a deterministic fake defined in this file; Video-Lens
ships none. Run: python tests/test_video_understanding.py
"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_lens
from core.contracts import (
    KNOWLEDGE_SCHEMA_VERSION, Frame, PointerEvent, Region, VideoInput, VisionObservation, VisualElement,
)
from core.evidence import build_analysis_result, build_structured_observation
from core.interfaces import VideoUnderstandingAdapter, VisionAdapter
from video_lens import PipelineConfig, analyze_video, process_video

ROOT = Path(__file__).resolve().parent.parent
VIDEO = VideoInput(path="v.mp4", duration_sec=10.0, width=100, height=100, fps=5.0, has_audio=False)


def _obs(t, text, status="ok", model="fake-native-v1", elements=(), **meta) -> VisionObservation:
    return VisionObservation(timestamp_sec=t, frame_path="", status=status,
                             description=text if status in ("ok", "low_information") else "",
                             elements=elements, confidence=0.8 if status == "ok" else 0.0,
                             model=model, analysis_metadata=dict(meta))


def _frame_obs(t, text, status="ok", elements=()) -> VisionObservation:
    return VisionObservation(timestamp_sec=t, frame_path=f"f{t}.jpg", status=status,
                             description=text if status == "ok" else "", elements=elements,
                             confidence=0.9 if status == "ok" else 0.0, model="frame-model")


_DEFAULT = object()


class FakeVideoProvider:
    """Deterministic native-video provider: records every call, returns what it
    was given (or raises it)."""

    def __init__(self, returns=_DEFAULT, raises: BaseException | None = None):
        self.returns = returns if returns is not _DEFAULT else (
            _obs(1.0, "native: a chart is on screen", scene=0),
            _obs(3.5, "native: a line is drawn across the chart", scene=1),
        )
        self.raises = raises
        self.calls: list[dict] = []

    def analyze_video(self, video, *, transcript=None, start_sec=0.0, end_sec=None, prompt=None):
        self.calls.append(dict(video=video, transcript=transcript, start_sec=start_sec,
                               end_sec=end_sec, prompt=prompt))
        if self.raises is not None:
            raise self.raises
        return self.returns


def _vision_refs(so) -> list[str]:
    return [e.ref for e in so.observed if e.kind == "vision"]


def _so_at(result, t):
    return next(so for so in result.structured_observations if so.timestamp_sec == t)


# ============================== correlation (exact) ==============================

def test_no_video_stream_means_byte_identical_correlation():
    kwargs = dict(frames=[Frame(timestamp_sec=2.0, path="f.jpg")],
                  vision_observations=[_frame_obs(2.0, "frame view")], tolerance_sec=1.0)
    assert build_analysis_result(VIDEO, [2.0], **kwargs) == \
        build_analysis_result(VIDEO, [2.0], video_observations=None, **kwargs)
    so = build_structured_observation(5.0, tolerance_sec=1.0)
    assert so.unavailable == ("frame", "transcript", "pointer", "vision"), \
        "with no provider configured, no new stream name may appear"


def test_same_moment_keeps_both_per_frame_and_native_evidence():
    result = build_analysis_result(VIDEO, [2.0], vision_observations=[_frame_obs(2.0, "frame view")],
                                   video_observations=[_obs(2.0, "native view")])
    so = result.structured_observations[0]
    assert _vision_refs(so) == ["frame view", "native view"], "neither displaces the other"
    mo = result.observations[0]
    assert mo.vision.model == "frame-model" and mo.description == "frame view"  # slot: per-frame first


def test_native_fills_the_vision_slot_when_per_frame_vision_is_absent_or_failed():
    for frame_vision in ([], [_frame_obs(2.0, "", status="failed")]):
        result = build_analysis_result(VIDEO, [2.0], vision_observations=frame_vision,
                                       video_observations=[_obs(2.0, "native view")])
        so, mo = result.structured_observations[0], result.observations[0]
        assert _vision_refs(so) == ["native view"]
        assert "vision" not in so.unavailable and "video_understanding" not in so.unavailable
        assert mo.vision.model == "fake-native-v1" and mo.description == "native view"


def test_configured_but_nothing_usable_is_recorded_as_unavailable():
    for natives in ([], [_obs(2.0, "", status="failed")], [_obs(9.0, "far away")]):
        so = build_structured_observation(2.0, video_observations=natives, tolerance_sec=1.0)
        assert "video_understanding" in so.unavailable and "vision" in so.unavailable
        assert _vision_refs(so) == []


def test_native_correlation_reuses_the_inclusive_tolerance_rule():
    def refs(native_t):
        return _vision_refs(build_structured_observation(
            2.0, video_observations=[_obs(native_t, f"at {native_t}")], tolerance_sec=1.0))
    assert refs(2.6) == ["at 2.6"]
    assert refs(3.0) == ["at 3.0"], "exactly at tolerance is inside (inclusive, as for every stream)"
    assert refs(1.0) == ["at 1.0"]
    assert refs(3.01) == []
    nearest = build_structured_observation(2.0, video_observations=[_obs(1.5, "a"), _obs(2.3, "b")],
                                           tolerance_sec=1.0)
    assert _vision_refs(nearest) == ["b"], "nearest wins, as for per-frame vision"


def test_native_element_descriptions_become_vision_evidence():
    el = VisualElement(kind="text", description="SUPPORT 4200", region=Region(0.1, 0.1, 0.4, 0.2),
                       region_confidence="approximate")
    so = build_structured_observation(2.0, video_observations=[_obs(2.0, "a chart", elements=(el,))])
    assert _vision_refs(so) == ["a chart", "text: SUPPORT 4200"]


def test_inference_rules_still_read_per_frame_vision_only():
    region = Region(0.4, 0.4, 0.6, 0.6)
    el = VisualElement(kind="button", description="OK", region=region, region_confidence="detected")
    pointer = [PointerEvent(timestamp_sec=2.0, status="detected", x=50, y=50, frame_width=100,
                            frame_height=100, confidence=0.9)]
    native_only = build_structured_observation(2.0, pointer_events=pointer,
                                               video_observations=[_obs(2.0, "a dialog", elements=(el,))])
    assert not any(i.basis == "pointer_in_vision_region" for i in native_only.inferences)
    per_frame = build_structured_observation(2.0, pointer_events=pointer,
                                             vision_observations=[_frame_obs(2.0, "a dialog", elements=(el,))])
    assert any(i.basis == "pointer_in_vision_region" for i in per_frame.inferences)


# ============================== the seam itself ==============================

def test_vision_adapter_is_unchanged_and_the_new_seam_is_a_sibling():
    params = list(inspect.signature(VisionAdapter.analyze_frame).parameters)
    assert params == ["self", "frame", "transcript_context", "pointer"]
    sig = inspect.signature(VideoUnderstandingAdapter.analyze_video)
    assert list(sig.parameters) == ["self", "video", "transcript", "start_sec", "end_sec", "prompt"]
    assert all(sig.parameters[n].kind is inspect.Parameter.KEYWORD_ONLY
               for n in ("transcript", "start_sec", "end_sec", "prompt"))
    assert PipelineConfig().video_understanding_provider is None


def test_core_stays_free_of_any_concrete_video_provider_sdk():
    banned = ("google", "genai", "vertexai", "openai", "ollama", "anthropic", "requests", "httpx")
    for relative in ("core/interfaces.py", "core/evidence.py", "core/contracts.py", "video_lens.py"):
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert not any(b in stripped.split()[1] for b in banned), f"{relative}: {stripped}"


# ============================== pipeline (real ffmpeg, fake provider) ==============================

def _make_video(path: str, duration: float = 5.0):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"color=c=blue:size=160x120:rate=5:duration={duration}", "-pix_fmt", "yuv420p", path],
                   capture_output=True, check=True)


def _run(provider, tmp: str | None = None, **overrides):
    if tmp is None:
        with tempfile.TemporaryDirectory() as fresh:
            return _run(provider, fresh, **overrides)
    p = os.path.join(tmp, "v.mp4")
    if not os.path.exists(p):
        _make_video(p)
    settings = dict(vision_enabled=False, pointer_enabled=False,
                    frame_cache_dir=os.path.join(tmp, "fc"), video_understanding_provider=provider)
    settings.update(overrides)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        result, transcript = video_lens._run_pipeline(p, PipelineConfig(**settings))
    return result, transcript, err.getvalue()


def test_no_provider_adds_no_call_no_stream_and_no_query_points():
    with mock.patch.object(video_lens, "_analyze_video_understanding") as spy:
        result, _t, _e = _run(None)
    spy.assert_not_called()
    frame_ts = [so.frame.timestamp_sec for so in result.structured_observations if so.frame]
    assert [so.timestamp_sec for so in result.structured_observations] == frame_ts
    assert all("video_understanding" not in so.unavailable for so in result.structured_observations)


def test_provider_is_called_once_with_video_level_input():
    fake = FakeVideoProvider()
    result, transcript, _e = _run(fake)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["video"] == result.video and call["transcript"] is transcript
    assert (call["start_sec"], call["end_sec"], call["prompt"]) == (0.0, None, None)


def test_native_observations_reach_the_result_at_their_own_timestamps():
    result, _t, _e = _run(FakeVideoProvider())
    times = [so.timestamp_sec for so in result.structured_observations]
    assert 1.0 in times and 3.5 in times, "moments no frame was sampled at still get a window"
    assert "native: a line is drawn across the chart" in _vision_refs(_so_at(result, 3.5))
    mo = next(o for o in result.observations if o.timestamp_sec == 3.5)
    assert mo.vision.timestamp_sec == 3.5 and mo.vision.model == "fake-native-v1"
    assert mo.vision.analysis_metadata == {"scene": 1, "evidence_source": "video_understanding"}


def test_provider_runs_even_when_per_frame_vision_is_disabled():
    fake = FakeVideoProvider()
    _run(fake, vision_enabled=False)
    assert len(fake.calls) == 1


def test_provider_failure_degrades_without_leaking_the_error_message():
    for exc in (RuntimeError("403 for https://bucket.example/v.mp4?X-Signature=SECRET"),
                TimeoutError("timed out"), ValueError("bad"), OSError("cannot read video")):
        result, _t, err = _run(FakeVideoProvider(raises=exc))
        assert result.structured_observations, "the rest of the analysis still ran"
        assert f"the provider raised {type(exc).__name__}" in err
        assert "SECRET" not in err and "https://" not in err
        for so in result.structured_observations:
            assert "video_understanding" in so.unavailable and _vision_refs(so) == []


def test_interrupts_are_not_swallowed():
    try:
        _run(FakeVideoProvider(raises=KeyboardInterrupt()))
        assert False, "KeyboardInterrupt must propagate -- only Exception is caught"
    except KeyboardInterrupt:
        pass


def test_malformed_return_values_mean_unavailable_not_failure():
    for bad in (None, "a whole-video summary", {"scene": 1}, 42, VisionObservation(
            timestamp_sec=1.0, frame_path="", status="ok", description="one, not a sequence")):
        result, _t, err = _run(FakeVideoProvider(returns=bad))
        assert "not a sequence of VisionObservation" in err
        assert all(_vision_refs(so) == [] for so in result.structured_observations)


def test_items_without_a_valid_timestamp_are_dropped_never_guessed():
    good = _obs(2.0, "grounded")
    bad = [{"timestamp_sec": 1.0, "description": "dict"}, "text", None,
           _obs(float("nan"), "WHOLE-VIDEO SUMMARY"), _obs(float("inf"), "inf"),
           _obs(-1.0, "before start"), _obs(5.0 + 1.5, "past the end")]
    result, _t, err = _run(FakeVideoProvider(returns=[bad[0], good] + bad[1:]))
    assert "dropped 7 observation(s)" in err
    every_ref = [e.ref for e in result.evidence]
    assert "grounded" in every_ref
    for text in ("WHOLE-VIDEO SUMMARY", "inf", "before start", "past the end", "dict"):
        assert text not in every_ref, f"{text!r} must not be attached to any moment"


def test_ungrounded_whole_video_summary_attaches_to_no_frame():
    summary = _obs(float("nan"), "The video explains a breakout strategy.")
    result, _t, _e = _run(FakeVideoProvider(returns=[summary]))
    assert all(_vision_refs(so) == [] for so in result.structured_observations)
    assert all("video_understanding" in so.unavailable for so in result.structured_observations)


def test_unordered_provider_output_is_ordered_and_deterministic():
    with tempfile.TemporaryDirectory() as tmp:  # same dir, so frame cache paths match too
        first, _t, _e = _run(FakeVideoProvider(returns=[_obs(3.5, "later"), _obs(1.0, "earlier")]), tmp)
        second, _t, _e = _run(FakeVideoProvider(returns=[_obs(3.5, "later"), _obs(1.0, "earlier")]), tmp)
    times = [so.timestamp_sec for so in first.structured_observations]
    assert times == sorted(times)
    assert first.structured_observations == second.structured_observations


def test_native_evidence_flows_through_process_video_as_ordinary_vision():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        config = PipelineConfig(vision_enabled=False, pointer_enabled=False, output_dir=os.path.join(tmp, "out"),
                                video_understanding_provider=FakeVideoProvider())
        with contextlib.redirect_stderr(io.StringIO()):
            package = process_video(p, config)
        assert package.processing.knowledge_schema_version == KNOWLEDGE_SCHEMA_VERSION == "1.2"
        assert "vision" in package.processing.stages_available, "native output is ordinary vision evidence"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"  ok: {t.__name__}")
    print("All video-understanding tests passed.")
