"""Evidence correlation + structured understanding tests: timestamp
correlation, temporal tolerance, observation/inference separation,
provenance, confidence, disagreement recording, and graceful degradation
when evidence streams are missing.

Fully synthetic/deterministic -- no video files, no network, no API key.
Run: python tests/test_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    Evidence, Frame, Inference, PointerEvent, Transcript, TranscriptSegment,
    VideoInput, VisionObservation, VisualElement, Region,
)
from core.evidence import (
    all_within_tolerance, build_analysis_result, build_structured_observation,
    nearest_within_tolerance,
)


def _video():
    return VideoInput(path="x.mp4", duration_sec=60.0, width=1920, height=1080, fps=30.0, has_audio=True)


def _frame(t=5.0):
    return Frame(timestamp_sec=t, path=f"f{t}.jpg", width=1920, height=1080)


def _pointer(t, x, y, status="detected", confidence=0.9):
    return PointerEvent(timestamp_sec=t, status=status, x=x, y=y,
                         frame_width=1920, frame_height=1080, confidence=confidence)


def _vision(t=5.0, elements=(), confidence=0.85, status="ok", description="a screen"):
    return VisionObservation(timestamp_sec=t, frame_path=f"f{t}.jpg", status=status,
                              description=description, elements=elements, confidence=confidence,
                              model="claude-sonnet-5")


# ----------------------------- correlation primitives -----------------------------

def test_nearest_within_tolerance_picks_closest():
    items = [_frame(1.0), _frame(4.0), _frame(4.9)]
    got = nearest_within_tolerance(items, 5.0, tolerance_sec=1.0)
    assert got.timestamp_sec == 4.9


def test_nearest_within_tolerance_excludes_out_of_range():
    items = [_frame(1.0)]
    assert nearest_within_tolerance(items, 5.0, tolerance_sec=1.0) is None


def test_nearest_within_tolerance_ties_prefer_earlier():
    items = [_frame(4.0), _frame(6.0)]  # both exactly 1.0s from 5.0
    got = nearest_within_tolerance(items, 5.0, tolerance_sec=1.0)
    assert got.timestamp_sec == 4.0


def test_all_within_tolerance_orders_by_time():
    items = [_pointer(5.9, 1, 1), _pointer(4.1, 2, 2), _pointer(5.0, 3, 3)]
    got = all_within_tolerance(items, 5.0, tolerance_sec=1.0)
    assert [i.timestamp_sec for i in got] == [4.1, 5.0, 5.9]


def test_boundary_exactly_at_tolerance_is_included_beyond_is_excluded():
    items = [_frame(4.0), _frame(3.999)]
    assert nearest_within_tolerance(items, 5.0, tolerance_sec=1.0) is not None
    assert nearest_within_tolerance([_frame(3.999)], 5.0, tolerance_sec=1.0) is None


# ------------------------------- basic structuring -------------------------------

def test_tolerance_must_be_positive():
    try:
        build_structured_observation(5.0, tolerance_sec=0)
        assert False
    except ValueError:
        pass
    try:
        build_structured_observation(5.0, tolerance_sec=-1)
        assert False
    except ValueError:
        pass


def test_empty_evidence_is_fully_unavailable_never_fabricated():
    so = build_structured_observation(5.0, tolerance_sec=1.0)
    assert so.observed == ()
    assert so.inferences == ()
    assert set(so.unavailable) == {"frame", "transcript", "pointer", "vision"}


def test_speech_frame_correlation():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.6, end_sec=5.4, text="hello")])
    frame = _frame(5.0)
    so = build_structured_observation(5.0, transcript=transcript, frames=[frame], tolerance_sec=1.0)
    kinds = {e.kind for e in so.observed}
    assert kinds == {"frame", "transcript"}
    assert "pointer" in so.unavailable and "vision" in so.unavailable


def test_pointer_frame_correlation():
    frame = _frame(5.0)
    pointer = _pointer(5.2, 100, 200)
    so = build_structured_observation(5.0, frames=[frame], pointer_events=[pointer], tolerance_sec=1.0)
    pointer_evidence = [e for e in so.observed if e.kind == "pointer"]
    assert len(pointer_evidence) == 1
    assert "x=100, y=200" in pointer_evidence[0].ref
    assert pointer_evidence[0].confidence == 0.9


def test_vision_frame_correlation():
    frame = _frame(5.0)
    vision = _vision(5.0, description="a code editor")
    so = build_structured_observation(5.0, frames=[frame], vision_observations=[vision], tolerance_sec=1.0)
    vision_evidence = [e for e in so.observed if e.kind == "vision"]
    assert any("code editor" in e.ref for e in vision_evidence)


def test_all_four_streams_together():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.8, end_sec=5.2, text="click here")])
    frame = _frame(5.0)
    pointer = _pointer(5.0, 960, 540)
    vision = _vision(5.0)
    so = build_structured_observation(5.0, transcript=transcript, frames=[frame],
                                       pointer_events=[pointer], vision_observations=[vision],
                                       tolerance_sec=1.0)
    assert {e.kind for e in so.observed} == {"frame", "transcript", "pointer", "vision"}
    assert so.unavailable == ()


# ------------------------------ missing evidence ------------------------------

def test_missing_pointer_degrades_gracefully():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.8, end_sec=5.2, text="x")])
    so = build_structured_observation(5.0, transcript=transcript, frames=[_frame(5.0)], tolerance_sec=1.0)
    assert "pointer" in so.unavailable
    assert not any(e.kind == "pointer" for e in so.observed)


def test_missing_vision_degrades_gracefully():
    so = build_structured_observation(5.0, frames=[_frame(5.0)], pointer_events=[_pointer(5.0, 1, 1)],
                                       tolerance_sec=1.0)
    assert "vision" in so.unavailable
    assert not any(e.kind == "vision" for e in so.observed)


def test_missing_transcript_degrades_gracefully():
    so = build_structured_observation(5.0, frames=[_frame(5.0)], tolerance_sec=1.0)
    assert "transcript" in so.unavailable


def test_not_detected_pointer_never_becomes_a_claimed_position():
    pointer = _pointer(5.0, None, None, status="not_detected")
    # constructing not_detected with x/y=None mirrors real detector output
    so = build_structured_observation(5.0, pointer_events=[pointer], tolerance_sec=1.0)
    assert not any(e.kind == "pointer" for e in so.observed)
    assert "pointer" in so.unavailable


def test_vision_unavailable_status_is_not_treated_as_observed():
    vision = VisionObservation(timestamp_sec=5.0, frame_path="f.jpg", status="unavailable")
    so = build_structured_observation(5.0, vision_observations=[vision], tolerance_sec=1.0)
    assert not any(e.kind == "vision" for e in so.observed)
    assert "vision" in so.unavailable


def test_vision_failed_status_is_not_treated_as_observed():
    vision = VisionObservation(timestamp_sec=5.0, frame_path="f.jpg", status="failed")
    so = build_structured_observation(5.0, vision_observations=[vision], tolerance_sec=1.0)
    assert not any(e.kind == "vision" for e in so.observed)
    assert "vision" in so.unavailable


# ------------------------- inference vs observation, disagreement --------------

def test_pointer_in_vision_region_produces_inference_with_citations():
    frame = _frame(5.0)
    el = VisualElement(kind="button", description="submit", region=Region(0.4, 0.4, 0.6, 0.6),
                        region_confidence="detected")
    vision = _vision(5.0, elements=(el,))
    pointer = _pointer(5.0, 960, 540)  # normalized (0.5, 0.5) -- inside the region
    so = build_structured_observation(5.0, frames=[frame], pointer_events=[pointer],
                                       vision_observations=[vision], tolerance_sec=1.0)
    assert len(so.inferences) == 1
    inf = so.inferences[0]
    assert inf.basis == "pointer_in_vision_region"
    assert "submit" in inf.text
    assert len(inf.supporting_evidence) == 2
    assert inf.confidence == round(pointer.confidence * vision.confidence, 4)


def test_pointer_outside_all_regions_is_a_disagreement_not_an_inference():
    frame = _frame(5.0)
    el = VisualElement(kind="button", description="submit", region=Region(0.4, 0.4, 0.6, 0.6),
                        region_confidence="detected")
    vision = _vision(5.0, elements=(el,))
    pointer = _pointer(5.0, 10, 10)  # normalized ~(0.005, 0.009) -- outside
    so = build_structured_observation(5.0, frames=[frame], pointer_events=[pointer],
                                       vision_observations=[vision], tolerance_sec=1.0)
    assert so.inferences == ()
    assert len(so.disagreements) == 1
    assert "does not fall within" in so.disagreements[0]


def test_vision_region_without_pointer_confirmation_is_a_disagreement():
    frame = _frame(5.0)
    el = VisualElement(kind="chart", description="a chart", region=Region(0.1, 0.1, 0.9, 0.9),
                        region_confidence="detected")
    vision = _vision(5.0, elements=(el,))
    so = build_structured_observation(5.0, frames=[frame], vision_observations=[vision], tolerance_sec=1.0)
    assert so.inferences == ()
    assert any("no pointer evidence" in d for d in so.disagreements)


def test_speech_while_pointer_stationary():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.6, end_sec=5.4, text="see this")])
    pointers = [_pointer(4.7, 960, 540), _pointer(5.3, 962, 541)]  # tiny movement
    so = build_structured_observation(5.0, transcript=transcript, pointer_events=pointers, tolerance_sec=1.0)
    motion_inf = [i for i in so.inferences if i.basis.startswith("speech_pointer")]
    assert len(motion_inf) == 1
    assert motion_inf[0].basis == "speech_pointer_stationary"


def test_speech_while_pointer_moving():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.6, end_sec=5.4, text="watch it move")])
    pointers = [_pointer(4.7, 100, 100), _pointer(5.3, 1800, 1000)]  # large movement
    so = build_structured_observation(5.0, transcript=transcript, pointer_events=pointers, tolerance_sec=1.0)
    motion_inf = [i for i in so.inferences if i.basis.startswith("speech_pointer")]
    assert len(motion_inf) == 1
    assert motion_inf[0].basis == "speech_pointer_motion"


def test_single_pointer_reading_does_not_claim_motion_or_stationary():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.6, end_sec=5.4, text="x")])
    so = build_structured_observation(5.0, transcript=transcript, pointer_events=[_pointer(5.0, 1, 1)],
                                       tolerance_sec=1.0)
    assert not any(i.basis.startswith("speech_pointer") for i in so.inferences), \
        "one reading is not evidence of motion OR of being stationary -- must not guess either way"


def test_inference_never_appears_in_observed():
    frame = _frame(5.0)
    el = VisualElement(kind="button", description="submit", region=Region(0.4, 0.4, 0.6, 0.6),
                        region_confidence="detected")
    vision = _vision(5.0, elements=(el,))
    pointer = _pointer(5.0, 960, 540)
    so = build_structured_observation(5.0, frames=[frame], pointer_events=[pointer],
                                       vision_observations=[vision], tolerance_sec=1.0)
    observed_texts = " ".join(e.ref for e in so.observed)
    assert "may have been referring" not in observed_texts, \
        "inferred language must never leak into the observed/evidence list"


# ---------------------------------- contracts ----------------------------------

def test_inference_requires_evidence_and_valid_confidence():
    ev = Evidence(timestamp_sec=1.0, kind="pointer", ref="x=1,y=1")
    Inference(text="x", supporting_evidence=(ev,), confidence=0.5, basis="test_rule")  # ok
    try:
        Inference(text="x", supporting_evidence=(), confidence=0.5, basis="test_rule")
        assert False, "an inference with no supporting evidence should be rejected"
    except ValueError:
        pass
    try:
        Inference(text="x", supporting_evidence=(ev,), confidence=1.5, basis="test_rule")
        assert False
    except ValueError:
        pass


def test_evidence_kind_validated():
    Evidence(timestamp_sec=1.0, kind="vision", ref="x")  # ok
    try:
        Evidence(timestamp_sec=1.0, kind="nonsense", ref="x")
        assert False
    except ValueError:
        pass


# --------------------------- ordering / boundaries / determinism --------------------------

def test_observations_near_video_start_and_end():
    so_start = build_structured_observation(0.0, frames=[_frame(0.0)], tolerance_sec=1.0)
    so_end = build_structured_observation(59.9, frames=[_frame(59.9)], tolerance_sec=1.0)
    assert so_start.observed and so_end.observed
    assert so_start.timestamp_sec == 0.0 and so_end.timestamp_sec == 59.9


def test_evidence_outside_tolerance_is_not_incorrectly_joined():
    frame_far = _frame(100.0)
    pointer_far = _pointer(200.0, 1, 1)
    so = build_structured_observation(5.0, frames=[frame_far], pointer_events=[pointer_far], tolerance_sec=1.0)
    assert so.observed == ()
    assert "frame" in so.unavailable and "pointer" in so.unavailable


def test_deterministic_repeated_calls_produce_identical_result():
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.8, end_sec=5.2, text="x")])
    frame, pointer = _frame(5.0), _pointer(5.0, 1, 1)
    kwargs = dict(transcript=transcript, frames=[frame], pointer_events=[pointer], tolerance_sec=1.0)
    a = build_structured_observation(5.0, **kwargs)
    b = build_structured_observation(5.0, **kwargs)
    assert a == b  # same inputs -> byte-identical structured observation, no LLM randomness involved


def test_build_analysis_result_preserves_timestamp_order_and_provenance():
    video = _video()
    transcript = Transcript(segments=[TranscriptSegment(start_sec=4.8, end_sec=5.2, text="a"),
                                       TranscriptSegment(start_sec=9.8, end_sec=10.2, text="b")])
    frames = [_frame(5.0), _frame(10.0)]
    result = build_analysis_result(video, [5.0, 10.0], transcript=transcript, frames=frames, tolerance_sec=1.0)
    assert [so.timestamp_sec for so in result.structured_observations] == [5.0, 10.0]
    assert len(result.observations) == 2
    assert all(isinstance(e, Evidence) for e in result.evidence)
    assert result.video is video


if __name__ == "__main__":
    test_nearest_within_tolerance_picks_closest()
    test_nearest_within_tolerance_excludes_out_of_range()
    test_nearest_within_tolerance_ties_prefer_earlier()
    test_all_within_tolerance_orders_by_time()
    test_boundary_exactly_at_tolerance_is_included_beyond_is_excluded()
    test_tolerance_must_be_positive()
    test_empty_evidence_is_fully_unavailable_never_fabricated()
    test_speech_frame_correlation()
    test_pointer_frame_correlation()
    test_vision_frame_correlation()
    test_all_four_streams_together()
    test_missing_pointer_degrades_gracefully()
    test_missing_vision_degrades_gracefully()
    test_missing_transcript_degrades_gracefully()
    test_not_detected_pointer_never_becomes_a_claimed_position()
    test_vision_unavailable_status_is_not_treated_as_observed()
    test_vision_failed_status_is_not_treated_as_observed()
    test_pointer_in_vision_region_produces_inference_with_citations()
    test_pointer_outside_all_regions_is_a_disagreement_not_an_inference()
    test_vision_region_without_pointer_confirmation_is_a_disagreement()
    test_speech_while_pointer_stationary()
    test_speech_while_pointer_moving()
    test_single_pointer_reading_does_not_claim_motion_or_stationary()
    test_inference_never_appears_in_observed()
    test_inference_requires_evidence_and_valid_confidence()
    test_evidence_kind_validated()
    test_observations_near_video_start_and_end()
    test_evidence_outside_tolerance_is_not_incorrectly_joined()
    test_deterministic_repeated_calls_produce_identical_result()
    test_build_analysis_result_preserves_timestamp_order_and_provenance()
    print("All evidence tests passed.")
