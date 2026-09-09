"""Knowledge extraction tests (Step 9): KnowledgePackage creation, compactness,
observed/inferred separation, honest degradation when streams are missing,
provenance/timestamps, and domain neutrality.

Fully synthetic/deterministic -- no video files, no network, no API key.
Run: python tests/test_knowledge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    Evidence, Frame, Inference, KeyPoint, PointerEvent, Transcript, TranscriptSegment,
    VideoInput, VisionObservation,
)
from core.evidence import build_analysis_result
from core.knowledge import build_knowledge_package, render_markdown, to_dict, to_json


def _video(source_type="local", source="lesson.mp4", title=None, duration=120.0):
    return VideoInput(path="lesson.mp4", duration_sec=duration, width=1920, height=1080,
                       fps=30.0, has_audio=True, source_type=source_type, source=source, title=title)


def _transcript(segments):
    return Transcript(segments=segments, source="test", status="ok")


def _seg(t, text):
    return TranscriptSegment(start_sec=t, end_sec=t + 3.0, text=text, confidence=0.9)


# ------------------------------ basic creation ------------------------------

def test_knowledge_package_created_from_analysis_result():
    transcript = _transcript([
        _seg(0.0, "First, open the settings panel."),
        _seg(5.0, "Warning: never click that button while recording."),
    ])
    result = build_analysis_result(_video(), [0.0, 5.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    assert pkg.source.source == "lesson.mp4"
    assert pkg.source.duration_sec == 120.0
    assert pkg.summary


def test_key_lessons_extracted_via_cue_phrases():
    transcript = _transcript([
        _seg(0.0, "First, open the settings panel."),
        _seg(5.0, "Warning: never click that button while recording."),
        _seg(10.0, "This is just some ordinary narration with no cue phrase."),
    ])
    result = build_analysis_result(_video(), [0.0, 5.0, 10.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    kinds = {kp.kind for kp in pkg.key_lessons}
    assert "procedure" in kinds
    assert "warning" in kinds
    assert len(pkg.key_lessons) == 2  # the plain narration segment must NOT be flagged


def test_key_lessons_cite_evidence():
    transcript = _transcript([_seg(0.0, "Warning: never do this.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    assert len(pkg.key_lessons) == 1
    kp = pkg.key_lessons[0]
    assert len(kp.supporting_evidence) >= 1
    assert kp.supporting_evidence[0].kind == "transcript"


def test_important_observations_reuse_step7_inferences():
    ev = Evidence(timestamp_sec=1.0, kind="pointer", ref="x=1, y=1")
    inf = Inference(text="pointer moved during narration", supporting_evidence=(ev,),
                     confidence=0.7, basis="speech_pointer_motion")
    from core.contracts import StructuredObservation
    so = StructuredObservation(timestamp_sec=1.0, tolerance_sec=1.0, inferences=(inf,))
    from core.contracts import AnalysisResult
    result = AnalysisResult(video=_video(), structured_observations=[so])
    pkg = build_knowledge_package(result, transcript=None)
    assert len(pkg.important_observations) == 1
    assert pkg.important_observations[0].text == "pointer moved during narration"
    assert pkg.important_observations[0].kind == "observation"


# ------------------------------ compactness ------------------------------

def test_package_does_not_embed_raw_video_or_image_bytes():
    transcript = _transcript([_seg(0.0, "Warning: never do this.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    blob = to_json(pkg)
    assert "lesson.mp4" in blob  # a path reference is fine (provenance)...
    assert len(blob.encode("utf-8")) < 20_000  # ...but the whole package stays small
    # no field on the contract can hold raw bytes -- structurally verified by
    # the dataclass shape itself (str/tuple/float only), not just this instance


def test_package_does_not_dump_entire_raw_transcript():
    # 50 plain narration segments with no cue phrase -- none should survive
    segments = [_seg(float(i) * 10, f"Just some plain narration sentence number {i}.")
                for i in range(50)]
    transcript = _transcript(segments)
    result = build_analysis_result(_video(), [s.start_sec for s in segments],
                                    transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    assert len(pkg.key_lessons) == 0  # nothing cue-worthy -- not blindly copied in


# ------------------------------ provenance / timestamps ------------------------------

def test_provenance_and_timestamps_preserved():
    transcript = _transcript([_seg(42.5, "For example, this is how it works.")])
    result = build_analysis_result(_video(), [42.5], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    kp = pkg.key_lessons[0]
    assert kp.timestamp_sec == 42.5
    assert kp.supporting_evidence[0].timestamp_sec == 42.5


# ------------------------------ honest degradation ------------------------------

def test_missing_vision_recorded_honestly():
    transcript = _transcript([_seg(0.0, "First, open the panel.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    assert "vision" in pkg.processing.stages_unavailable
    assert any("vision" in note.lower() for note in pkg.limitations)


def test_missing_pointer_recorded_honestly():
    transcript = _transcript([_seg(0.0, "First, open the panel.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    assert "pointer" in pkg.processing.stages_unavailable


def test_missing_transcript_handled_gracefully():
    from core.contracts import AnalysisResult
    result = AnalysisResult(video=_video(), structured_observations=[])
    pkg = build_knowledge_package(result, transcript=None)
    assert pkg.key_lessons == ()
    assert pkg.topics == ()
    assert pkg.processing.transcript_status == "unavailable"
    assert any("no usable speech" in note.lower() for note in pkg.limitations)
    assert pkg.source.duration_sec > 0  # package is still valid, just honestly limited


def test_no_audio_video_transcript_status_surfaced():
    transcript = Transcript(segments=[], source="test", status="no_audio")
    result = build_analysis_result(_video(), [], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    assert pkg.processing.transcript_status == "no_audio"
    assert pkg.key_lessons == ()


# ------------------------------ domain neutrality ------------------------------

def test_no_domain_vocabulary_hardcoded():
    transcript = _transcript([
        _seg(0.0, "For example, ICT fair value gaps and SMT divergence matter here."),
    ])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    # matched purely on the domain-neutral cue phrase "for example" -- not on
    # any trading-specific term, which never appears in the extraction code
    assert pkg.key_lessons[0].kind == "example"
    import core.knowledge as k
    src = Path(k.__file__).read_text(encoding="utf-8").lower()
    for banned in ("fair value gap", "smt divergence", "trading strategy", "ict concept"):
        assert banned not in src


# ------------------------------ serialization / rendering ------------------------------

def test_to_dict_and_to_json_round_trip_shape():
    transcript = _transcript([_seg(0.0, "Warning: never do this.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    d = to_dict(pkg)
    assert d["source"]["source"] == "lesson.mp4"
    assert len(d["key_lessons"]) == 1  # asdict preserves tuple/list container type as-is
    import json
    json.loads(to_json(pkg))  # must be valid JSON


def test_markdown_is_derived_presentation_not_source_of_truth():
    transcript = _transcript([_seg(0.0, "Warning: never do this.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    pkg = build_knowledge_package(result, transcript=transcript)
    md = render_markdown(pkg)
    assert md.startswith("#")
    assert "Warning: never do this." in md
    assert isinstance(md, str)
    # the same package always renders identically -- markdown has no state of its own
    assert render_markdown(pkg) == md


if __name__ == "__main__":
    test_knowledge_package_created_from_analysis_result()
    test_key_lessons_extracted_via_cue_phrases()
    test_key_lessons_cite_evidence()
    test_important_observations_reuse_step7_inferences()
    test_package_does_not_embed_raw_video_or_image_bytes()
    test_package_does_not_dump_entire_raw_transcript()
    test_provenance_and_timestamps_preserved()
    test_missing_vision_recorded_honestly()
    test_missing_pointer_recorded_honestly()
    test_missing_transcript_handled_gracefully()
    test_no_audio_video_transcript_status_surfaced()
    test_no_domain_vocabulary_hardcoded()
    test_to_dict_and_to_json_round_trip_shape()
    test_markdown_is_derived_presentation_not_source_of_truth()
    print("All knowledge tests passed.")
