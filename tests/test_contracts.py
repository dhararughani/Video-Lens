"""Minimal self-check for core contracts. Run: python tests/test_contracts.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    AnalysisResult, Evidence, Frame, MultimodalObservation,
    PointerEvent, Transcript, TranscriptSegment, VideoInput,
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


if __name__ == "__main__":
    test_video_input_is_frozen()
    test_transcript_segment_ordering_is_representable()
    test_analysis_result_composes_observations_and_evidence()
    print("All contract tests passed.")
