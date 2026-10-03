"""P1-B tests: schema 1.2 -- `visual_change` and `cursor_track` as first-class
evidence, carried from the real P0-B / P0-C outputs through correlation, the
KnowledgePackage, JSON handoff and claim verification.

Detector results are produced by the real detectors (real frames, real tracks),
never hand-assembled Evidence, wherever the mapping is what's under test. The
only fakes are the synthesizer and the native-video provider, which Video-Lens
ships none of. Run: python -m pytest tests/test_evidence_schema.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    EVIDENCE_KINDS, KNOWLEDGE_SCHEMA_VERSION, CursorSegment, Evidence, KnowledgePackage, PointerEvent,
    PointerTrack, VideoInput, VisionObservation, VisualChangeEvent,
)
from core.cursor_intelligence import analyze_track
from core.errors import KnowledgePackageError
from core.evidence import (
    NATIVE_VIDEO_SOURCE, build_analysis_result, build_structured_observation,
    cursor_segment_to_evidence, visual_change_to_evidence,
)
from core.knowledge import (
    build_knowledge_package, export_knowledge_package, to_dict, to_json, validate_knowledge_package,
)
from core.synthesis import build_synthesis_brief, parse_and_verify
from core.visual_change import compare_frames, detect_visual_changes

from tests.test_visual_change import _frame, _pair, _region_of, _shift, _textured
from tests.test_video_understanding import FakeVideoProvider, _frame_obs, _obs
from tests.test_handoff_contract import _transcript, _seg

VIDEO = VideoInput(path="v.mp4", duration_sec=30.0, width=640, height=360, fps=5.0, has_audio=False,
                   title="t")
W, H, DIAG = 300, 400, 500.0


def _ref(ev: Evidence) -> dict:
    return json.loads(ev.ref)


def _pointer(t, x=None, y=None, status="detected", conf=0.9) -> PointerEvent:
    if status == "not_detected":
        return PointerEvent(timestamp_sec=t, status=status, frame_width=W, frame_height=H)
    return PointerEvent(timestamp_sec=t, status=status, x=x, y=y, frame_width=W, frame_height=H,
                        confidence=conf)


def _moving_segments(t0=0.0):
    events = [_pointer(t0 + i * 0.5, 100 + 30 * i, 200) for i in range(5)]
    return analyze_track(PointerTrack(start_sec=t0, end_sec=t0 + 2.0, events=events))


# ============================== 1. the contract ==============================

def test_all_six_kinds_are_accepted_and_nothing_else():
    assert EVIDENCE_KINDS == ("frame", "transcript", "pointer", "vision", "visual_change", "cursor_track")
    for kind in EVIDENCE_KINDS:
        Evidence(timestamp_sec=0.0, kind=kind, ref="x")
    for bad in ("", "video_understanding", "stationary", "Vision", "visual_changes", "cursor", None):
        try:
            Evidence(timestamp_sec=0.0, kind=bad, ref="x")
            assert False, f"{bad!r} must be rejected"
        except ValueError:
            pass


def test_schema_version_is_exactly_1_2():
    assert KNOWLEDGE_SCHEMA_VERSION == "1.2"


def test_old_style_evidence_is_unchanged():
    ev = Evidence(timestamp_sec=1.0, kind="pointer", ref="x=1, y=2", confidence=0.5)
    assert ev.timestamp_end_sec is None and ev.source == ""
    assert Evidence(1.0, "frame", "p.jpg") == Evidence(timestamp_sec=1.0, kind="frame", ref="p.jpg")


def test_a_span_cannot_end_before_it_starts():
    Evidence(timestamp_sec=2.0, kind="cursor_track", ref="{}", timestamp_end_sec=2.0)
    try:
        Evidence(timestamp_sec=2.0, kind="cursor_track", ref="{}", timestamp_end_sec=1.9)
        assert False
    except ValueError:
        pass


# ============================== 2. visual_change mapping ==============================

def test_localized_change_becomes_visual_change_evidence_with_everything_preserved():
    with tempfile.TemporaryDirectory() as tmp:
        base = _textured(1)
        event = _pair(tmp, base, _shift(base, 100, 60, 300, 200), t0=2.0, t1=4.0)
    assert event.kind == "visual_change_localized" and event.status == "detected"

    ev = visual_change_to_evidence(event)
    assert ev.kind == "visual_change"
    assert ev.timestamp_sec == event.timestamp_sec == 4.0
    assert ev.confidence == event.confidence
    assert ev.source == event.detection_method and ev.source
    p = _ref(ev)
    assert p["status"] == "detected" and p["kind"] == "visual_change_localized"
    assert p["magnitude"] == round(event.magnitude, 4)
    assert p["compared"] == [2.0, 4.0], "the temporal relationship of the two frames survives"
    assert len(p["regions"]) == len(event.regions) >= 1
    for got, want in zip(p["regions"], event.regions):
        assert all(abs(g - w) < 1e-4 for g, w in zip(got, (want.x1, want.y1, want.x2, want.y2)))
    assert abs(p["regions"][0][0] - _region_of(100, 60, 300, 200).x1) < 0.03


def test_global_change_becomes_visual_change_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        event = _pair(tmp, _textured(1), _textured(2), t0=1.0, t1=3.0)
    assert event.kind == "visual_change_global"
    ev = visual_change_to_evidence(event)
    assert ev.kind == "visual_change" and _ref(ev)["kind"] == "visual_change_global"
    assert _ref(ev)["magnitude"] == round(event.magnitude, 4)


def test_not_detected_and_unavailable_keep_their_status_and_never_invent_a_magnitude():
    with tempfile.TemporaryDirectory() as tmp:
        same = _textured(3)
        quiet = _pair(tmp, same, same.copy(), t0=1.0, t1=2.0)
        a = _frame(tmp, "a", same, 1.0)
        b = _frame(tmp, "b", _textured(3, h=100, w=100), 2.0)  # different size: cannot be compared
        broken = compare_frames(a, b)
    assert quiet.status == "not_detected" and broken.status == "unavailable"
    q, u = _ref(visual_change_to_evidence(quiet)), _ref(visual_change_to_evidence(broken))
    assert q["status"] == "not_detected" and q["regions"] == []
    assert u["status"] == "unavailable" and u["magnitude"] is None, "a failed comparison is not 0.0"


def test_a_pathological_number_of_regions_stays_within_the_excerpt_limit_without_corrupting_json():
    from core.contracts import Region
    regions = tuple(Region(i / 20, 0.0, i / 20 + 0.04, 0.5) for i in range(8))
    event = VisualChangeEvent(timestamp_sec=3.0, kind="visual_change_localized", status="detected",
                              magnitude=0.1, regions=regions, confidence=0.5,
                              detection_method="m" * 60, compared_timestamps=(1.0, 3.0), detail="d" * 200)
    ev = visual_change_to_evidence(event)
    assert len(ev.ref) <= 400
    json.loads(ev.ref)  # still valid JSON


# ============================== 3. cursor_track mapping ==============================

def test_moving_segment_becomes_cursor_track_evidence_with_everything_preserved():
    (seg,) = _moving_segments()
    ev = cursor_segment_to_evidence(seg)
    assert ev.kind == "cursor_track" and ev.source == "cursor_intelligence"
    assert (ev.timestamp_sec, ev.timestamp_end_sec) == (seg.start_sec, seg.end_sec) == (0.0, 2.0)
    assert ev.confidence == seg.confidence > 0
    p = _ref(ev)
    assert p["state"] == "moving" and p["basis"] == seg.basis
    assert p["direction_deg"] == seg.direction_deg == 0.0
    assert p["speed_norm"] == round(seg.mean_speed_norm, 4) == 0.12


def test_uncertain_segment_stays_uncertain_and_carries_no_direction_or_speed():
    track = PointerTrack(start_sec=0.0, end_sec=5.0, events=[_pointer(0.0, 10, 10), _pointer(5.0, 200, 300)])
    (seg,) = analyze_track(track)  # 5s apart: too far to assert movement
    assert seg.motion_state == "uncertain"
    ev = cursor_segment_to_evidence(seg)
    p = _ref(ev)
    assert p["state"] == "uncertain" and p["direction_deg"] is None and p["speed_norm"] is None
    assert ev.confidence == 0.0 and p["basis"] == seg.basis


def test_stationary_is_never_introduced():
    states = set()
    for events in ([], [_pointer(0.0, 5, 5)], [_pointer(0.0, 5, 5), _pointer(0.5, 5, 5)],
                   [_pointer(0.0, 5, 5), _pointer(0.5, status="not_detected")],
                   [_pointer(i * 0.5, 100 + 30 * i, 200) for i in range(5)]):
        track = PointerTrack(start_sec=0.0, end_sec=3.0, events=events)
        for seg in analyze_track(track):
            ev = cursor_segment_to_evidence(seg)
            states.add(_ref(ev)["state"])
            assert "stationary" not in ev.ref
    assert states <= {"moving", "uncertain"}
    try:
        Evidence(timestamp_sec=0.0, kind="stationary", ref="x")
        assert False
    except ValueError:
        pass


# ============================== 4. correlation ==============================

def test_correlation_attaches_measurements_to_the_right_windows_and_marks_the_rest_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        base = _textured(1)
        events = detect_visual_changes([_frame(tmp, "f0", base, 2.0),
                                        _frame(tmp, "f1", _shift(base, 100, 60, 300, 200), 4.0)])
    segments = _moving_segments()
    result = build_analysis_result(VIDEO, [0.5, 4.0, 20.0], visual_changes=events, cursor_segments=segments,
                                   tolerance_sec=1.0)
    early, at_change, late = result.structured_observations
    assert [e.kind for e in at_change.observed if e.kind == "visual_change"] == ["visual_change"]
    assert [e for e in early.observed if e.kind == "visual_change"] == []
    assert "visual_change" in early.unavailable and "visual_change" not in at_change.unavailable
    assert any(e.kind == "cursor_track" for e in early.observed), "segment [0,2] overlaps 0.5 +/- 1"
    assert "cursor_track" in late.unavailable and "cursor_track" in at_change.unavailable


def test_not_configured_means_no_trace_at_all():
    so = build_structured_observation(5.0, tolerance_sec=1.0)
    assert so.unavailable == ("frame", "transcript", "pointer", "vision")
    assert build_analysis_result(VIDEO, [2.0]) == build_analysis_result(
        VIDEO, [2.0], visual_changes=None, cursor_segments=None)


def test_an_uncomparable_change_is_unavailable_not_observed():
    with tempfile.TemporaryDirectory() as tmp:
        broken = compare_frames(_frame(tmp, "a", _textured(1), 1.0),
                                _frame(tmp, "b", _textured(1, h=100, w=100), 2.0))
    assert broken.status == "unavailable"
    so = build_structured_observation(2.0, visual_changes=[broken], tolerance_sec=1.0)
    assert not [e for e in so.observed if e.kind == "visual_change"]
    assert "visual_change" in so.unavailable


def test_measurements_never_feed_an_inference():
    with tempfile.TemporaryDirectory() as tmp:
        base = _textured(1)
        events = detect_visual_changes([_frame(tmp, "f0", base, 2.0),
                                        _frame(tmp, "f1", _shift(base, 100, 60, 300, 200), 4.0)])
    so = build_structured_observation(4.0, visual_changes=events, cursor_segments=_moving_segments(),
                                      tolerance_sec=1.0)
    assert so.inferences == ()


# ============================== 5. native-video provenance ==============================

def test_native_vision_is_distinguishable_from_frame_vision_without_a_new_kind():
    result = build_analysis_result(VIDEO, [2.0], vision_observations=[_frame_obs(2.0, "frame view")],
                                   video_observations=[_obs(2.0, "native view")])
    vision = [e for e in result.structured_observations[0].observed if e.kind == "vision"]
    assert [(e.ref, e.source) for e in vision] == [("frame view", ""), ("native view", NATIVE_VIDEO_SOURCE)]
    assert NATIVE_VIDEO_SOURCE not in EVIDENCE_KINDS
    # and the provenance survives into the package, where it is NOT collapsed
    package = build_knowledge_package(result, transcript=None)
    assert package.processing.stages_available == ("vision",) or "vision" in package.processing.stages_available


def test_identical_text_from_the_two_sources_is_not_merged_in_the_brief_or_package():
    result = build_analysis_result(VIDEO, [2.0], vision_observations=[_frame_obs(2.0, "same words")],
                                   video_observations=[_obs(2.0, "same words")])
    brief = build_synthesis_brief(result, None, build_knowledge_package(result).source)
    assert sorted(i.evidence.source for i in brief.items if i.evidence.kind == "vision") == \
        ["", NATIVE_VIDEO_SOURCE]


def test_native_element_evidence_carries_the_same_provenance():
    from core.contracts import VisualElement
    el = VisualElement(kind="chart", description="a candlestick chart")
    result = build_analysis_result(VIDEO, [2.0], video_observations=[_obs(2.0, "native view", elements=(el,))])
    vision = [e for e in result.structured_observations[0].observed if e.kind == "vision"]
    assert [e.ref for e in vision] == ["native view", "chart: a candlestick chart"]
    assert {e.source for e in vision} == {NATIVE_VIDEO_SOURCE}


def test_package_evidence_keeps_both_sources_when_a_claim_cites_identical_text_from_each():
    result = build_analysis_result(VIDEO, [2.0], vision_observations=[_frame_obs(2.0, "same words")],
                                   video_observations=[_obs(2.0, "same words")])
    brief = build_synthesis_brief(result, None, build_knowledge_package(result).source)
    ids = [i.evidence_id for i in brief.items if i.evidence.kind == "vision"]
    out = parse_and_verify({"claims": [{"text": "x", "nature": "observed", "timestamp_sec": 2.0,
                                        "evidence_ids": ids, "confidence": 0.7}]}, brief)
    package = build_knowledge_package(result, synthesis=out)
    assert sorted(e.source for e in package.evidence if e.kind == "vision") == ["", NATIVE_VIDEO_SOURCE]


def test_native_provenance_survives_claim_verification():
    result = build_analysis_result(VIDEO, [2.0], video_observations=[_obs(2.0, "native view")])
    brief = build_synthesis_brief(result, None, build_knowledge_package(result).source)
    eid = next(i.evidence_id for i in brief.items if i.evidence.kind == "vision")
    out = parse_and_verify({"claims": [{"text": "a chart is shown", "nature": "observed",
                                        "timestamp_sec": 2.0, "evidence_ids": [eid], "confidence": 0.7}]},
                           brief)
    assert out.claims[0].supporting_evidence[0].source == NATIVE_VIDEO_SOURCE


def test_native_vision_does_not_pull_in_a_frame_as_its_source_image():
    from core.visual_evidence import select_and_bundle
    from core.contracts import Claim
    native = Evidence(timestamp_sec=2.0, kind="vision", ref="native view", confidence=0.7,
                      source=NATIVE_VIDEO_SOURCE)
    frame_vision = Evidence(timestamp_sec=2.0, kind="vision", ref="frame view", confidence=0.7)
    frame = Evidence(timestamp_sec=2.0, kind="frame", ref="whatever.jpg")
    with tempfile.TemporaryDirectory() as tmp:
        import cv2, numpy as np
        path = os.path.join(tmp, "whatever.jpg")
        cv2.imwrite(path, np.full((64, 96), 90, dtype=np.uint8))
        frame = Evidence(timestamp_sec=2.0, kind="frame", ref=path)
        for ev, expect_image in ((native, False), (frame_vision, True)):
            claim = Claim(text="x", kind="fact", status="observed", timestamp_sec=2.0,
                          supporting_evidence=(ev,), confidence=0.7, verification="v")
            visual, _ = select_and_bundle(candidates={"e0": frame}, claims=(claim,), key_points=(),
                                          output_dir=os.path.join(tmp, "out"), stem="s",
                                          vision_descriptions={})
            assert bool(visual) is expect_image, ev.source


# ============================== 6. KnowledgePackage ==============================

def _enabled_result(tmp):
    base = _textured(1)
    events = detect_visual_changes([_frame(tmp, "f0", base, 2.0),
                                    _frame(tmp, "f1", _shift(base, 100, 60, 300, 200), 4.0),
                                    _frame(tmp, "f2", _shift(base, 100, 60, 300, 200), 6.0)])
    return build_analysis_result(VIDEO, [2.0, 4.0, 6.0], visual_changes=events,
                                 cursor_segments=_moving_segments(), tolerance_sec=1.0)


def test_package_exposes_the_new_categories_deduplicated_and_in_time_order():
    with tempfile.TemporaryDirectory() as tmp:
        result = _enabled_result(tmp)
    package = build_knowledge_package(result, transcript=_transcript([_seg(0.0, "Hello there.")]))
    assert all(e.kind == "visual_change" for e in package.visual_changes) and package.visual_changes
    assert all(e.kind == "cursor_track" for e in package.cursor_intelligence) and package.cursor_intelligence
    assert len(set((e.timestamp_sec, e.ref) for e in package.visual_changes)) == len(package.visual_changes)
    assert [e.timestamp_sec for e in package.visual_changes] == sorted(e.timestamp_sec
                                                                         for e in package.visual_changes)
    validate_knowledge_package(package)
    assert {"visual_change", "cursor_track"} <= set(package.processing.stages_available)
    assert package.processing.knowledge_schema_version == "1.2"


def test_default_package_is_valid_has_empty_new_fields_and_does_not_list_them_as_missing():
    result = build_analysis_result(VIDEO, [0.0], transcript=_transcript([_seg(0.0, "Hello.")]))
    package = build_knowledge_package(result, transcript=_transcript([_seg(0.0, "Hello.")]))
    validate_knowledge_package(package)
    assert package.visual_changes == () and package.cursor_intelligence == ()
    assert "visual_change" not in package.processing.stages_unavailable
    assert "cursor_track" not in package.processing.stages_unavailable
    assert not any("Visual-change" in n or "Cursor-movement" in n for n in package.limitations)


def test_configured_but_empty_is_reported_as_unavailable_with_an_honest_limitation():
    result = build_analysis_result(VIDEO, [2.0], visual_changes=[], cursor_segments=())
    package = build_knowledge_package(result)
    assert {"visual_change", "cursor_track"} <= set(package.processing.stages_unavailable)
    text = " ".join(package.limitations)
    assert "not proof the picture did not change" in text and "not proof the cursor was still" in text
    validate_knowledge_package(package)


def test_constructing_a_package_without_the_new_fields_still_works():
    package = build_knowledge_package(build_analysis_result(VIDEO, [0.0]))
    from dataclasses import replace
    bare = KnowledgePackage(source=package.source, summary=package.summary, topics=(), key_lessons=(),
                            important_observations=(), evidence=(), limitations=(),
                            processing=package.processing)
    assert bare.visual_changes == () and bare.cursor_intelligence == ()
    assert replace(bare).visual_changes == ()


def test_validation_rejects_a_misfiled_or_out_of_range_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        package = build_knowledge_package(_enabled_result(tmp))
    from dataclasses import replace
    wrong = replace(package, visual_changes=(Evidence(timestamp_sec=2.0, kind="cursor_track", ref="{}"),))
    late = replace(package, cursor_intelligence=(Evidence(timestamp_sec=999.0, kind="cursor_track", ref="{}"),))
    huge = replace(package, visual_changes=(Evidence(timestamp_sec=2.0, kind="visual_change", ref="x" * 401),))
    for bad in (wrong, late, huge):
        try:
            validate_knowledge_package(bad)
            assert False
        except KnowledgePackageError:
            pass


# ============================== 7. serialization / handoff ==============================

def test_json_round_trip_preserves_the_new_evidence_and_fields_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        package = build_knowledge_package(_enabled_result(tmp))
        data = json.loads(to_json(package))
        assert data["processing"]["knowledge_schema_version"] == "1.2"
        for field, kind in (("visual_changes", "visual_change"), ("cursor_intelligence", "cursor_track")):
            original = getattr(package, field)
            assert len(data[field]) == len(original) > 0
            for raw, ev in zip(data[field], original):
                assert Evidence(**raw) == ev, "rebuilding from JSON gives back the same Evidence"
                assert raw["kind"] == kind
                json.loads(raw["ref"])  # the measurement itself is parseable
        # and through the real export path
        path = export_knowledge_package(package, os.path.join(tmp, "out"))
        assert json.load(open(path, encoding="utf-8")) == data


def test_default_package_serializes_the_new_fields_as_empty_lists():
    package = build_knowledge_package(build_analysis_result(VIDEO, [0.0]))
    assert to_dict(package)["visual_changes"] == ()  # the handoff dict, as the dataclass holds it
    data = json.loads(to_json(package))  # ...and on the wire
    assert data["visual_changes"] == [] and data["cursor_intelligence"] == []
    assert data["evidence"] == []
    # every old top-level key is still present (a 1.1 consumer loses nothing)
    for key in ("source", "summary", "topics", "key_lessons", "important_observations", "evidence",
                "limitations", "processing", "semantic_summary", "claims", "visual_evidence", "synthesis"):
        assert key in data


def test_old_evidence_dicts_without_the_new_keys_still_load():
    old = {"timestamp_sec": 1.0, "kind": "vision", "ref": "x", "confidence": 0.5}  # a 1.1 package's evidence
    ev = Evidence(**old)
    assert ev.timestamp_end_sec is None and ev.source == ""
    try:
        Evidence(**{**old, "kind": "made_up"})
        assert False
    except ValueError:
        pass


# ============================== 8. claim verification ==============================

class _Scene:
    """A brief with one item of each interesting kind, so a claim can cite any mix."""

    def __init__(self, tmp):
        base = _textured(1)
        events = detect_visual_changes([_frame(tmp, "f0", base, 2.0),
                                        _frame(tmp, "f1", _shift(base, 100, 60, 300, 200), 4.0)])
        transcript = _transcript([_seg(4.0, "Now I am changing the timeframe.")])
        result = build_analysis_result(
            VIDEO, [4.0], transcript=transcript, visual_changes=events,
            cursor_segments=_moving_segments(3.5),
            vision_observations=[_frame_obs(4.0, "a candlestick chart with a 1h label")],
            tolerance_sec=1.0)
        self.brief = build_synthesis_brief(result, transcript, build_knowledge_package(result).source)

    def id_of(self, kind):
        return next(i.evidence_id for i in self.brief.items if i.evidence.kind == kind)

    def verify(self, nature, *kinds, text="something happened"):
        out = parse_and_verify({"claims": [{"text": text, "nature": nature, "timestamp_sec": 4.0,
                                            "evidence_ids": [self.id_of(k) for k in kinds],
                                            "confidence": 0.8}]}, self.brief)
        assert len(out.claims) == 1, out.metadata
        return out.claims[0]


def test_the_brief_carries_real_measurements_and_explains_them_to_the_synthesizer():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    kinds = {i.evidence.kind for i in scene.brief.items}
    assert {"visual_change", "cursor_track", "vision", "transcript"} <= kinds
    assert "never what changed or why" in scene.brief.system_prompt
    assert "never what the user did" in scene.brief.system_prompt


def test_quiet_intervals_are_kept_out_of_the_brief():
    with tempfile.TemporaryDirectory() as tmp:
        same = _textured(4)
        events = detect_visual_changes([_frame(tmp, "a", same, 1.0), _frame(tmp, "b", same.copy(), 2.0)])
    uncertain = analyze_track(PointerTrack(start_sec=0.0, end_sec=2.0, events=[]))
    result = build_analysis_result(VIDEO, [2.0], visual_changes=events, cursor_segments=uncertain)
    assert any(e.kind == "visual_change" for e in result.structured_observations[0].observed)
    brief = build_synthesis_brief(result, None, build_knowledge_package(result).source)
    assert not [i for i in brief.items if i.evidence.kind in ("visual_change", "cursor_track")]


def test_claims_can_cite_each_new_kind_and_vision_without_being_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    for kinds in (("visual_change",), ("cursor_track",), ("vision",), ("visual_change", "cursor_track")):
        claim = scene.verify("inferred", *kinds)
        assert {e.kind for e in claim.supporting_evidence} == set(kinds)


def test_evidence_provenance_survives_verification_for_measurements():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    claim = scene.verify("inferred", "cursor_track")
    ev = claim.supporting_evidence[0]
    assert ev.source == "cursor_intelligence" and ev.timestamp_end_sec is not None
    assert _ref(ev)["state"] == "moving"


def test_a_measurement_alone_never_makes_an_observed_claim():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    for kinds in (("visual_change",), ("cursor_track",), ("visual_change", "cursor_track")):
        claim = scene.verify("observed", *kinds, text="The user changed the chart timeframe")
        assert claim.status == "inferred", "deterministic evidence cannot make an interpretation 'observed'"
        assert claim.verification == "measurement_evidence_only"
        assert any("not what changed, why, or what the user did" in lim for lim in claim.limitations)


def test_speech_or_a_visual_description_can_still_support_an_observed_claim_alongside_a_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    spoken = scene.verify("observed", "transcript", "visual_change")
    assert spoken.status == "observed" and spoken.verification == "speech_evidence_with_measured_change"
    seen = scene.verify("observed", "vision", "visual_change")
    assert seen.status == "observed" and seen.verification == "visual_evidence_only"
    both = scene.verify("observed", "transcript", "vision", "visual_change")
    assert both.verification == "speech_corroborated_by_visual_evidence"
    for claim in (spoken, seen, both):  # ...but the measurement's limit is always stated
        assert any("not what changed" in lim for lim in claim.limitations)


def test_existing_verification_outcomes_are_exactly_as_before():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    assert scene.verify("observed", "transcript").verification == "speech_evidence_only"
    assert scene.verify("observed", "vision").verification == "visual_evidence_only"
    assert scene.verify("observed", "transcript", "vision").verification == \
        "speech_corroborated_by_visual_evidence"
    assert scene.verify("observed", "vision").status == "observed"
    for kind in ("transcript", "vision"):
        assert not any("Measured" in lim for lim in scene.verify("observed", kind).limitations)


def test_a_fabricated_measurement_id_is_still_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _Scene(tmp)
    out = parse_and_verify({"claims": [{"text": "x", "nature": "inferred", "timestamp_sec": 4.0,
                                        "evidence_ids": ["e999"], "confidence": 0.5}]}, scene.brief)
    assert out.claims == () and out.metadata.claims_rejected == 1


# ============================== 9. the pipeline ==============================

def _make_moving_video(path: str, duration: float = 7.0):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=5:duration={duration}",
                    "-pix_fmt", "yuv420p", path], capture_output=True, check=True)


def _process(tmp, **overrides):
    import video_lens
    p = os.path.join(tmp, "v.mp4")
    _make_moving_video(p)
    settings = dict(vision_enabled=False, pointer_enabled=True, keyframe_interval_sec=2.0,
                    output_dir=os.path.join(tmp, "out"))
    settings.update(overrides)
    with contextlib.redirect_stderr(io.StringIO()):
        return video_lens.process_video(p, video_lens.PipelineConfig(**settings))


def test_pipeline_defaults_do_not_run_the_new_measurements():
    import video_lens
    config = video_lens.PipelineConfig()
    assert config.visual_change_enabled is False and config.cursor_intelligence_enabled is False
    with tempfile.TemporaryDirectory() as tmp:
        package = _process(tmp)
    assert package.visual_changes == () and package.cursor_intelligence == ()
    assert package.processing.knowledge_schema_version == "1.2"
    assert not {"visual_change", "cursor_track"} & set(package.processing.stages_unavailable)


def test_pipeline_with_the_features_enabled_produces_a_valid_package_with_real_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        package = _process(tmp, visual_change_enabled=True, cursor_intelligence_enabled=True)
        exported = json.load(open(next(Path(tmp, "out").glob("*.json")), encoding="utf-8"))
    validate_knowledge_package(package)
    assert package.visual_changes, "testsrc2 changes every frame, so the keyframes differ"
    assert all(_ref(e)["compared"][0] < _ref(e)["compared"][1] for e in package.visual_changes)
    assert package.cursor_intelligence
    assert all(_ref(e)["state"] in ("moving", "uncertain") for e in package.cursor_intelligence)
    assert exported["processing"]["knowledge_schema_version"] == "1.2"
    assert len(exported["visual_changes"]) == len(package.visual_changes)


def test_pipeline_with_a_native_provider_keeps_its_provenance_in_the_package():
    with tempfile.TemporaryDirectory() as tmp:
        package = _process(tmp, pointer_enabled=False,
                           video_understanding_provider=FakeVideoProvider())
    validate_knowledge_package(package)
    assert "vision" in package.processing.stages_available


def test_pipeline_visual_change_limitation_says_what_is_measured():
    """The package says what its visual_change items are -- a pixel measurement
    between the two compared frames, not a meaning -- and only when there are any."""
    with tempfile.TemporaryDirectory() as tmp:
        with_vc = _process(tmp, visual_change_enabled=True)
    with tempfile.TemporaryDirectory() as tmp:
        without = _process(tmp)
    note = [lim for lim in with_vc.limitations if "pixels, not meaning" in lim]
    assert with_vc.visual_changes and len(note) == 1 and "`compared` interval" in note[0]
    assert not any("pixels, not meaning" in lim for lim in without.limitations)


def test_pipeline_survives_a_failing_measurement_stage():
    import video_lens
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(video_lens, "detect_visual_changes", side_effect=RuntimeError("boom")):
            package = _process(tmp, visual_change_enabled=True)
    validate_knowledge_package(package)
    assert package.visual_changes == ()
    assert "visual_change" in package.processing.stages_unavailable


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
