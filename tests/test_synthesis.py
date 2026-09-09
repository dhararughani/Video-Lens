"""Step 11 semantic-synthesis tests: the provider-neutral seam, and -- more
importantly -- the VERIFICATION that stands between a provider's output and a
KnowledgePackage.

Every test here uses a deterministic fake synthesizer. No external API, no
network, no credentials, no paid service: that is the point of the seam, and a
test suite that needed a key would disprove the property it's testing.

Run: python -m pytest tests/test_synthesis.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.contracts import (
    AnalysisResult, Evidence, Frame, Inference, KnowledgeSource, StructuredObservation,
    Transcript, TranscriptSegment, VideoInput,
)
from core.evidence import build_analysis_result
from core.errors import KnowledgePackageError
from core.knowledge import (
    build_knowledge_package, knowledge_source_for, to_json, validate_knowledge_package,
)
from core.synthesis import build_synthesis_brief, parse_and_verify, synthesize, unavailable


# ------------------------------- fixtures -------------------------------

def _video(duration: float = 60.0) -> VideoInput:
    return VideoInput(path="lesson.mp4", duration_sec=duration, width=1920, height=1080,
                       fps=30.0, has_audio=True, source_type="local", title="A Lesson")


def _transcript(pairs) -> Transcript:
    return Transcript(segments=[TranscriptSegment(start_sec=s, end_sec=s + 4.0, text=t,
                                                   confidence=0.9)
                                 for s, t in pairs], status="ok")


def _write_image(path: str, shade: int) -> None:
    """A flat grey frame. Two frames of the same shade are visually identical
    (so dHash collapses them); different shades are genuinely different."""
    import cv2
    cv2.imwrite(path, np.full((64, 96), shade, dtype=np.uint8))


def _frames(tmp: str, spec: list[tuple[float, int]]) -> list[Frame]:
    frames = []
    for i, (ts, shade) in enumerate(spec):
        p = os.path.join(tmp, f"f{i}.jpg")
        _write_image(p, shade)
        frames.append(Frame(timestamp_sec=ts, path=p, width=96, height=64))
    return frames


def _scene(tmp: str):
    """A small but realistic correlated result: speech across the whole video,
    three frames, and a matching source."""
    transcript = _transcript([
        (0.0, "Welcome. Today we look at how a cache actually works."),
        (10.0, "A cache stores a result so the next request avoids the work."),
        (20.0, "Warning: never cache a response that depends on the user."),
        (30.0, "In conclusion, caching trades freshness for speed."),
    ])
    frames = _frames(tmp, [(0.0, 30), (10.0, 140), (20.0, 220)])
    result = build_analysis_result(_video(), [0.0, 10.0, 20.0], transcript=transcript,
                                    frames=frames, tolerance_sec=1.5)
    return result, transcript, frames


class FakeSynthesizer:
    """Deterministic stand-in implementing exactly the KnowledgeSynthesizer
    shape. Cites REAL ids taken from the brief it is given, which is what a
    correct provider does."""

    def __init__(self, response=None, raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.brief = None

    def synthesize(self, brief) -> dict:
        self.brief = brief
        if self.raises is not None:
            raise self.raises
        if self.response is not None:
            return self.response
        speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
        frames = [i.evidence_id for i in brief.items if i.evidence.kind == "frame"]
        return {
            "provider": "fake-synth-v1",
            "summary": "The video explains what a cache is, how it trades freshness for "
                        "speed, and warns against caching user-specific responses.",
            "claims": [
                {"text": "A cache stores a result so later requests can skip the work.",
                 "kind": "concept", "nature": "observed", "timestamp_sec": 10.0,
                 "evidence_ids": [speech[1], frames[1]] if len(speech) > 1 and len(frames) > 1
                                  else speech[:1],
                 "confidence": 0.85},
                {"text": "Caching user-specific responses is unsafe.",
                 "kind": "warning", "nature": "observed", "timestamp_sec": 20.0,
                 "evidence_ids": [speech[2]] if len(speech) > 2 else speech[:1],
                 "confidence": 0.8},
                {"text": "The presenter is an expert on distributed systems.",
                 "kind": "fact", "nature": "unavailable", "timestamp_sec": 0.0,
                 "evidence_ids": [], "confidence": 0.0},
            ],
        }


# ------------------------------- 1. the seam itself -------------------------------

def test_synthesizer_conforms_to_the_protocol_shape():
    from core.interfaces import KnowledgeSynthesizer
    fake = FakeSynthesizer()
    assert hasattr(fake, "synthesize")
    assert set(getattr(KnowledgeSynthesizer, "__protocol_attrs__", {"synthesize"})) \
        .issubset(dir(fake))


def test_video_lens_ships_no_synthesizer_implementation():
    """The seam must stay a seam. If Video-Lens ever ships its own synthesizer
    it has quietly acquired a model dependency -- exactly what Step 11
    forbids."""
    repo = Path(__file__).resolve().parent.parent
    for py in repo.glob("adapters/**/*.py"):
        assert "def synthesize(" not in py.read_text(encoding="utf-8"), \
            f"{py} ships a KnowledgeSynthesizer implementation"


def test_brief_is_built_from_correlated_evidence_not_a_raw_transcript_dump():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        kinds = {i.evidence.kind for i in brief.items}
        assert "transcript" in kinds and "frame" in kinds
        # ids are unique and stable
        ids = [i.evidence_id for i in brief.items]
        assert len(ids) == len(set(ids))
        # speech is chunked, not one item per segment
        speech = [i for i in brief.items if i.evidence.kind == "transcript"]
        assert len(speech) < len(transcript.segments)


# ------------------------------- 2. fallback: no provider -------------------------------

def test_no_synthesizer_degrades_honestly_and_never_fabricates():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        package = build_knowledge_package(result, transcript=transcript,
                                           synthesis=unavailable("no synthesizer supplied"))
        assert package.semantic_summary is None
        assert package.claims == ()
        assert package.synthesis.status == "unavailable"
        # the deterministic layer still works
        assert package.summary and package.key_lessons
        # and the package SAYS semantic synthesis didn't happen
        assert any("no semantic knowledge synthesis" in n.lower() for n in package.limitations)
        validate_knowledge_package(package)


def test_deterministic_summary_is_never_presented_as_semantic():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        package = build_knowledge_package(result, transcript=transcript,
                                           synthesis=unavailable("none"))
        # the one honest tell: semantic_summary stays empty rather than being
        # backfilled from the deterministic sentence
        assert package.semantic_summary is None
        assert package.summary != package.semantic_summary


# ------------------------------- 3. a real (fake) provider end to end -------------------------------

def test_fake_provider_produces_verified_claims_and_a_whole_video_summary():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        fake = FakeSynthesizer()
        out = synthesize(fake, brief)

        assert fake.brief is brief  # the provider really was given the brief
        assert out.metadata.status == "ok"
        assert out.metadata.provider == "fake-synth-v1"
        assert out.summary and "cache" in out.summary.lower()
        assert len(out.claims) == 3
        for claim in out.claims:
            assert claim.verification  # every claim records how it was cross-checked


def test_claims_are_associated_with_the_evidence_they_cite():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = synthesize(FakeSynthesizer(), brief)
        concept = next(c for c in out.claims if c.kind == "concept")
        assert concept.supporting_evidence
        # the cited evidence is REAL evidence from the brief, not text the
        # provider made up
        real = {(e.timestamp_sec, e.kind) for e in brief.evidence_by_id().values()}
        for e in concept.supporting_evidence:
            assert (e.timestamp_sec, e.kind) in real


def test_speech_and_visual_evidence_association_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = synthesize(FakeSynthesizer(), brief)
        concept = next(c for c in out.claims if c.kind == "concept")
        kinds = {e.kind for e in concept.supporting_evidence}
        assert "transcript" in kinds and "frame" in kinds
        # A bare frame is an UNINTERPRETED image -- vision never ran on it, so
        # it illustrates rather than corroborates. Claiming corroboration here
        # would assert a cross-check that never happened.
        assert concept.verification == "speech_evidence_with_uninterpreted_frame"
        assert any("never visually analyzed" in n for n in concept.limitations)
        # The frame it wants is recorded as evidence, but `visual_evidence`
        # stays empty until a bundle actually exists -- a claim must never name
        # an image the package doesn't ship.
        assert concept.visual_evidence == ()
        assert any(e.kind == "frame" for e in concept.supporting_evidence)


def test_only_interpreted_vision_counts_as_visual_corroboration():
    """The distinction that keeps the visual cross-check honest: `vision`
    evidence was actually looked at, a `frame` was not."""
    vision_ev = Evidence(timestamp_sec=5.0, kind="vision",
                          ref="a diagram showing four labelled boxes", confidence=0.8)
    so = StructuredObservation(
        timestamp_sec=5.0, tolerance_sec=1.0,
        observed=(Evidence(timestamp_sec=5.0, kind="transcript", ref="four locations"), vision_ev),
    )
    result = AnalysisResult(video=_video(), structured_observations=[so])
    transcript = _transcript([(5.0, "four locations")])
    brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
    by_kind = {}
    for i in brief.items:
        by_kind.setdefault(i.evidence.kind, []).append(i.evidence_id)

    out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
        {"text": "The diagram shows four locations.", "kind": "concept", "nature": "observed",
         "timestamp_sec": 5.0,
         "evidence_ids": by_kind["transcript"][:1] + by_kind["vision"][:1],
         "confidence": 0.8},
    ]}, brief)
    assert out.claims[0].verification == "speech_corroborated_by_visual_evidence"
    assert not any("never visually analyzed" in n for n in out.claims[0].limitations)


def test_a_speech_only_claim_says_it_has_no_visual_support():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = synthesize(FakeSynthesizer(), brief)
        warning = next(c for c in out.claims if c.kind == "warning")
        assert warning.verification == "speech_evidence_only"
        assert any("no visual evidence" in n.lower() for n in warning.limitations)


def test_observed_inferred_and_unavailable_are_distinct_states():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
        response = {"provider": "p", "summary": "s", "claims": [
            {"text": "The speaker defines a cache.", "kind": "fact", "nature": "observed",
             "timestamp_sec": 10.0, "evidence_ids": speech[:1], "confidence": 0.9},
            {"text": "The video is aimed at beginners.", "kind": "conclusion",
             "nature": "inferred", "timestamp_sec": 10.0, "evidence_ids": speech[:1],
             "confidence": 0.6},
            {"text": "The speaker's employer is unknown.", "kind": "fact",
             "nature": "unavailable", "timestamp_sec": 0.0, "evidence_ids": [],
             "confidence": 0.9},
        ]}
        out = parse_and_verify(response, brief)
        statuses = {c.kind: c.status for c in out.claims}
        assert statuses == {"fact": "unavailable", "conclusion": "inferred"} or \
               [c.status for c in out.claims] == ["observed", "inferred", "unavailable"]
        # an "unavailable" claim establishes nothing, so it cannot carry confidence
        unavailable_claim = next(c for c in out.claims if c.status == "unavailable")
        assert unavailable_claim.confidence == 0.0


def test_conflicting_status_is_decided_by_video_lens_not_the_provider():
    """The provider is never asked to self-declare a conflict -- Video-Lens
    already knows where its own evidence streams disagreed, and applies that
    independently."""
    ev = Evidence(timestamp_sec=12.0, kind="transcript", ref="this is the important part",
                   confidence=0.9)
    so = StructuredObservation(
        timestamp_sec=12.0, tolerance_sec=1.0, observed=(ev,),
        disagreements=("pointer position does not fall within any located region",),
    )
    result = AnalysisResult(video=_video(), structured_observations=[so])
    transcript = _transcript([(12.0, "this is the important part")])
    brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
    speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
    out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
        {"text": "The presenter pointed at the important element.", "kind": "fact",
         "nature": "observed", "timestamp_sec": 12.0, "evidence_ids": speech[:1],
         "confidence": 0.9},
    ]}, brief)
    claim = out.claims[0]
    assert claim.status == "conflicting"
    assert any("disagree" in n.lower() for n in claim.limitations)


def test_pointer_evidence_enters_the_brief_only_when_it_contributed():
    """Pointer detection is limited by design; raw readings would drown the
    real evidence. Only pointer evidence Step 7 actually drew an inference
    from is shown to a synthesizer."""
    pointer_ev = Evidence(timestamp_sec=5.0, kind="pointer", ref="x=10, y=20", confidence=0.5)
    noise = StructuredObservation(timestamp_sec=5.0, tolerance_sec=1.0, observed=(pointer_ev,))
    contributing = StructuredObservation(
        timestamp_sec=8.0, tolerance_sec=1.0,
        observed=(Evidence(timestamp_sec=8.0, kind="pointer", ref="x=99, y=99", confidence=0.6),),
        inferences=(Inference(text="pointer was inside a located region",
                               supporting_evidence=(pointer_ev,), confidence=0.5,
                               basis="pointer_in_vision_region"),),
    )
    result = AnalysisResult(video=_video(), structured_observations=[noise, contributing])
    brief = build_synthesis_brief(result, None, knowledge_source_for(result.video))
    pointer_refs = [i.evidence.ref for i in brief.items if i.evidence.kind == "pointer"]
    assert pointer_refs == ["x=99, y=99"]


# ------------------------------- 4. verification rejects bad output -------------------------------

def test_fabricated_evidence_ids_are_rejected_not_downgraded():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
            {"text": "Something the video never said.", "kind": "fact", "nature": "observed",
             "timestamp_sec": 5.0, "evidence_ids": ["e9999"], "confidence": 0.99},
        ]}, brief)
        assert out.claims == ()
        assert out.metadata.claims_rejected == 1
        assert any("do not exist" in r for r in out.metadata.rejection_reasons)


def test_claims_with_timestamps_outside_the_video_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
        out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
            {"text": "Claimed at a time this video never reaches.", "kind": "fact",
             "nature": "observed", "timestamp_sec": 99999.0, "evidence_ids": speech[:1],
             "confidence": 0.5},
        ]}, brief)
        assert out.claims == ()
        assert any("outside the source duration" in r for r in out.metadata.rejection_reasons)


def test_unsupported_claims_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
            {"text": "A confident claim with nothing behind it.", "kind": "fact",
             "nature": "observed", "timestamp_sec": 5.0, "evidence_ids": [], "confidence": 1.0},
        ]}, brief)
        assert out.claims == ()
        assert any("cites no evidence" in r for r in out.metadata.rejection_reasons)


def test_confidence_is_bounded_by_the_cited_evidences_own_quality():
    """A provider cannot assert its way to a high score on weak evidence."""
    weak = Evidence(timestamp_sec=5.0, kind="vision", ref="a blurry shape", confidence=0.2)
    so = StructuredObservation(timestamp_sec=5.0, tolerance_sec=1.0, observed=(weak,))
    result = AnalysisResult(video=_video(), structured_observations=[so])
    brief = build_synthesis_brief(result, None, knowledge_source_for(result.video))
    vision_ids = [i.evidence_id for i in brief.items if i.evidence.kind == "vision"]
    out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
        {"text": "The screen clearly shows a chart.", "kind": "fact", "nature": "observed",
         "timestamp_sec": 5.0, "evidence_ids": vision_ids[:1], "confidence": 0.99},
    ]}, brief)
    assert out.claims[0].confidence == 0.2


def test_embedded_media_in_provider_output_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
        out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
            {"text": "Here is the frame: data:image/png;base64,iVBORw0KGgoAAAA",
             "kind": "fact", "nature": "observed", "timestamp_sec": 5.0,
             "evidence_ids": speech[:1], "confidence": 0.5},
        ]}, brief)
        assert out.claims == ()


def test_malformed_provider_responses_never_raise():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        for bad in ("not json at all", "", 42, [1, 2, 3], {"claims": "not a list"}):
            out = parse_and_verify(bad, brief)
            assert out.metadata.status == "failed"
            assert out.claims == ()


def test_garbage_claims_are_dropped_while_a_valid_summary_survives():
    """Partial garbage is not total failure: a usable summary is kept, the
    unusable claims are dropped, and the count says so."""
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = parse_and_verify('{"provider": "p", "summary": "a real summary", '
                                '"claims": [null, 7]}', brief)
        assert out.summary == "a real summary"
        assert out.claims == ()
        assert out.metadata.claims_rejected == 2


def test_provider_that_raises_is_recorded_not_propagated():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = synthesize(FakeSynthesizer(raises=RuntimeError("model is down")), brief)
        assert out.metadata.status == "failed"
        assert "model is down" in out.metadata.detail
        assert out.claims == ()


def test_a_provider_returning_json_text_is_accepted():
    """Real providers hand back a string; accepting both that and a parsed
    dict keeps a provider implementation genuinely thin."""
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
        text = "```json\n" + json.dumps({"provider": "p", "summary": "a summary", "claims": [
            {"text": "A claim.", "kind": "fact", "nature": "observed", "timestamp_sec": 10.0,
             "evidence_ids": speech[:1], "confidence": 0.7}]}) + "\n```"
        out = parse_and_verify(text, brief)
        assert out.metadata.status == "ok" and len(out.claims) == 1


# ------------------------------- 5. package-level guarantees -------------------------------

def test_rejections_are_reported_in_the_package_limitations():
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        speech = [i.evidence_id for i in brief.items if i.evidence.kind == "transcript"]
        out = parse_and_verify({"provider": "p", "summary": "s", "claims": [
            {"text": "Real.", "kind": "fact", "nature": "observed", "timestamp_sec": 10.0,
             "evidence_ids": speech[:1], "confidence": 0.7},
            {"text": "Fabricated.", "kind": "fact", "nature": "observed", "timestamp_sec": 10.0,
             "evidence_ids": ["e4242"], "confidence": 0.9},
        ]}, brief)
        package = build_knowledge_package(result, transcript=transcript, synthesis=out)
        assert package.synthesis.claims_rejected == 1
        assert any("failed evidence verification" in n for n in package.limitations)


def test_package_with_claims_is_compact_and_holds_no_full_transcript():
    with tempfile.TemporaryDirectory() as tmp:
        long_speech = [(float(i * 5), f"Sentence number {i} with a fair amount of filler text "
                                       f"so the transcript is genuinely long." * 3)
                        for i in range(60)]
        transcript = _transcript(long_speech)
        frames = _frames(tmp, [(0.0, 30), (10.0, 140)])
        result = build_analysis_result(_video(duration=320.0), [0.0, 10.0],
                                        transcript=transcript, frames=frames, tolerance_sec=1.5)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = synthesize(FakeSynthesizer(), brief)
        package = build_knowledge_package(result, transcript=transcript, synthesis=out)
        validate_knowledge_package(package)

        blob = to_json(package)
        full_transcript = " ".join(t for _, t in long_speech)
        assert len(blob) < len(full_transcript), "package must not approach transcript size"
        for ev in package.evidence:
            if ev.kind != "frame":
                assert len(ev.ref) <= 400
        assert "base64," not in blob


def test_validation_rejects_a_claim_pointing_at_absent_visual_evidence():
    """The invariant that makes a bundle trustworthy: no claim may reference an
    image the package doesn't actually ship."""
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as tmp:
        result, transcript, _ = _scene(tmp)
        brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
        out = synthesize(FakeSynthesizer(), brief)
        package = build_knowledge_package(result, transcript=transcript, synthesis=out)
        validate_knowledge_package(package)  # clean as built

        dangling = replace(package.claims[0], visual_evidence=("e_does_not_exist",))
        broken = replace(package, claims=(dangling,) + package.claims[1:])
        try:
            validate_knowledge_package(broken)
            assert False, "dangling visual evidence reference must be rejected"
        except KnowledgePackageError as e:
            assert "visual evidence" in str(e)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
