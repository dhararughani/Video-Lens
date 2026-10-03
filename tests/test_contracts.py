"""Minimal self-check for core contracts. Run: python tests/test_contracts.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    AnalysisResult, Evidence, Frame, InspectionRequest, InspectionResult, MultimodalObservation,
    PointerEvent, Region, Transcript, TranscriptSegment, VideoInput,
)


def test_video_input_is_frozen():
    v = VideoInput(path="x.mp4", duration_sec=1.0, width=10, height=10, fps=30.0, has_audio=True)
    try:
        v.path = "y.mp4"  # type: ignore[misc]
        assert False, "VideoInput should be immutable"
    except AttributeError:
        pass


def test_transcript_segment_ordering_is_representable():
    t = Transcript(segments=[
        TranscriptSegment(start_sec=0.0, end_sec=1.5, text="hello"),
        TranscriptSegment(start_sec=1.5, end_sec=3.0, text="world"),
    ])
    assert t.segments[0].end_sec == t.segments[1].start_sec


def test_analysis_result_composes_observations_and_evidence():
    video = VideoInput(path="x.mp4", duration_sec=5.0, width=10, height=10, fps=30.0, has_audio=False)
    frame = Frame(timestamp_sec=1.0, path="frame.jpg")
    pointer = PointerEvent(timestamp_sec=1.0, status="detected", x=5, y=5,
                            frame_width=10, frame_height=10, confidence=0.9)
    obs = MultimodalObservation(
        timestamp_sec=1.0, frame=frame, transcript_context=None,
        description="cursor over button", pointer=pointer,
    )
    result = AnalysisResult(
        video=video,
        observations=[obs],
        evidence=[Evidence(timestamp_sec=1.0, kind="frame", ref=frame.path)],
    )
    assert result.observations[0].pointer.x == 5
    assert result.evidence[0].kind == "frame"


def test_inspection_request_defaults_to_one_full_frame():
    r = InspectionRequest(timestamp_sec=2.5)
    assert (r.window_before_sec, r.window_after_sec) == (0.0, 0.0)
    assert r.fps is None and r.scale_width is None and r.region is None
    try:
        r.fps = 5.0  # type: ignore[misc]
        assert False, "InspectionRequest should be immutable"
    except AttributeError:
        pass


def test_inspection_request_accepts_a_fully_specified_request():
    r = InspectionRequest(timestamp_sec=10.0, window_before_sec=1.0, window_after_sec=2.0,
                          fps=4.0, scale_width=640, region=Region(0.1, 0.2, 0.9, 0.8))
    assert r.region.x2 == 0.9 and r.scale_width == 640


def test_inspection_request_rejects_obviously_invalid_values():
    bad = [
        dict(timestamp_sec=-1.0),
        dict(timestamp_sec=float("nan")),
        dict(timestamp_sec=float("inf")),
        dict(timestamp_sec=1.0, window_before_sec=-0.5, fps=2.0),
        dict(timestamp_sec=1.0, window_after_sec=float("nan"), fps=2.0),
        dict(timestamp_sec=1.0, fps=0.0),
        dict(timestamp_sec=1.0, fps=-3.0),
        dict(timestamp_sec=1.0, fps=float("inf")),
        dict(timestamp_sec=1.0, scale_width=0),
        dict(timestamp_sec=1.0, scale_width=-640),
        dict(timestamp_sec=1.0, scale_width=640.5),
        dict(timestamp_sec=1.0, scale_width=True),
        # a time window with no sampling density is ambiguous -- must be explicit
        dict(timestamp_sec=1.0, window_before_sec=1.0),
        dict(timestamp_sec=1.0, window_after_sec=1.0),
    ]
    for kwargs in bad:
        try:
            InspectionRequest(**kwargs)
            assert False, f"expected ValueError for {kwargs}"
        except ValueError:
            pass


def test_inspection_result_carries_request_and_provenance():
    req = InspectionRequest(timestamp_sec=1.0)
    frame = Frame(timestamp_sec=1.0, path="f.jpg", source_video="v.mp4", width=10, height=10)
    res = InspectionResult(request=req, frames=(frame,), video_source="v.mp4")
    assert res.request is req and res.frames[0].source_video == res.video_source
    assert res.extraction_method == "ffmpeg_seek"


if __name__ == "__main__":
    test_video_input_is_frozen()
    test_transcript_segment_ordering_is_representable()
    test_analysis_result_composes_observations_and_evidence()
    test_inspection_request_defaults_to_one_full_frame()
    test_inspection_request_accepts_a_fully_specified_request()
    test_inspection_request_rejects_obviously_invalid_values()
    test_inspection_result_carries_request_and_provenance()
    print("All contract tests passed.")
