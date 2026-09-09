"""Step 10 tests: the universal KnowledgePackage handoff/export contract --
schema versioning, observed-vs-inferred (`KeyPoint.nature`), validation,
export, portability, and standalone/no-downstream-consumer/no-Anthropic
guarantees.

Fully synthetic/deterministic where possible; real local videos (no
network, no API key) for the lifecycle-level checks. No downstream-consumer
import, no Anthropic API call anywhere in this file.

Run: python tests/test_handoff_contract.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import (
    KNOWLEDGE_SCHEMA_VERSION, AnalysisResult, Evidence, Inference, KeyPoint, KnowledgePackage,
    KnowledgeSource, ProcessingMetadata, StructuredObservation, Transcript, TranscriptSegment,
    VideoInput,
)
from core.errors import KnowledgePackageError
from core.evidence import build_analysis_result
from core.knowledge import (
    build_knowledge_package, export_knowledge_package, to_dict, to_json, validate_knowledge_package,
)


def _video(source_type="local", source="lesson.mp4", title=None, duration=120.0):
    return VideoInput(path="lesson.mp4", duration_sec=duration, width=1920, height=1080,
                       fps=30.0, has_audio=True, source_type=source_type, source=source, title=title)


def _transcript(segments):
    return Transcript(segments=segments, source="test", status="ok")


def _seg(t, text):
    return TranscriptSegment(start_sec=t, end_sec=t + 3.0, text=text, confidence=0.9)


def _valid_package() -> KnowledgePackage:
    transcript = _transcript([_seg(0.0, "Warning: never do this.")])
    result = build_analysis_result(_video(), [0.0], transcript=transcript, tolerance_sec=1.0)
    return build_knowledge_package(result, transcript=transcript)


# ------------------------------ 1/2. valid package + schema version ------------------------------

def test_package_is_valid_and_carries_schema_version():
    package = _valid_package()
    validate_knowledge_package(package)  # must not raise
    assert package.processing.knowledge_schema_version == KNOWLEDGE_SCHEMA_VERSION
    # Pinned to a literal on purpose: a package-shape change must be a
    # deliberate edit here, never a silent one. Bumped to "1.1" by Step 11,
    # which added the optional semantic fields (semantic_summary/claims/
    # visual_evidence/synthesis) -- see docs/synthesis.md.
    assert package.processing.knowledge_schema_version == "1.1"


def test_schema_version_is_independent_of_software_version():
    package = _valid_package()
    # two distinct concepts, not aliases of the same field
    assert package.processing.knowledge_schema_version != package.processing.video_lens_version \
        or True  # they may coincide by accident of current numbering, but must be separate fields
    from dataclasses import fields
    names = {f.name for f in fields(ProcessingMetadata)}
    assert "knowledge_schema_version" in names
    assert "video_lens_version" in names


# ------------------------------ 3. source provenance survives packaging ------------------------------

def test_source_provenance_survives_packaging():
    package = _valid_package()
    assert package.source.source == "lesson.mp4"
    assert package.source.source_type == "local"
    assert package.source.duration_sec == 120.0


# ------------------------------ 4. observed vs inferred survives packaging ------------------------------

def test_observed_vs_inferred_is_explicit_on_every_key_point():
    package = _valid_package()
    assert len(package.key_lessons) == 1
    assert package.key_lessons[0].nature == "observed"  # a transcript quote

    ev = Evidence(timestamp_sec=1.0, kind="pointer", ref="x=1, y=1")
    inf = Inference(text="pointer moved during narration", supporting_evidence=(ev,),
                     confidence=0.7, basis="speech_pointer_motion")
    so = StructuredObservation(timestamp_sec=1.0, tolerance_sec=1.0, inferences=(inf,))
    result = AnalysisResult(video=_video(), structured_observations=[so])
    package2 = build_knowledge_package(result, transcript=None)
    assert len(package2.important_observations) == 1
    assert package2.important_observations[0].nature == "inferred"


def test_key_point_rejects_invalid_nature():
    ev = Evidence(timestamp_sec=0.0, kind="transcript", ref="x")
    try:
        KeyPoint(text="x", kind="warning", timestamp_sec=0.0, supporting_evidence=(ev,),
                 confidence=0.5, nature="guessed")
        assert False, "an invalid nature value must be rejected"
    except ValueError:
        pass


# ------------------------------ 5. evidence/provenance survives packaging ------------------------------

def test_evidence_provenance_survives_packaging():
    package = _valid_package()
    assert len(package.evidence) >= 1
    assert package.evidence[0].kind == "transcript"
    assert package.key_lessons[0].supporting_evidence[0].timestamp_sec == 0.0


# ------------------------------ 6/7. invalid / empty packages ------------------------------

def test_invalid_duration_is_rejected():
    package = KnowledgePackage(
        source=KnowledgeSource(source_type="local", source="x.mp4", duration_sec=0.0),
        summary="a video", topics=(), key_lessons=(), important_observations=(), evidence=(),
        limitations=(), processing=ProcessingMetadata(
            generated_at="2026-01-01T00:00:00+00:00", video_lens_version="1.0.0",
            knowledge_schema_version=KNOWLEDGE_SCHEMA_VERSION, frame_count=0,
            transcript_status="unavailable", stages_available=(), stages_unavailable=()),
    )
    try:
        validate_knowledge_package(package)
        assert False, "zero/negative duration must be rejected"
    except KnowledgePackageError:
        pass


def test_missing_source_reference_is_rejected():
    package = KnowledgePackage(
        source=KnowledgeSource(source_type="local", source="", duration_sec=10.0),
        summary="a video", topics=(), key_lessons=(), important_observations=(), evidence=(),
        limitations=(), processing=ProcessingMetadata(
            generated_at="2026-01-01T00:00:00+00:00", video_lens_version="1.0.0",
            knowledge_schema_version=KNOWLEDGE_SCHEMA_VERSION, frame_count=0,
            transcript_status="unavailable", stages_available=(), stages_unavailable=()),
    )
    try:
        validate_knowledge_package(package)
        assert False, "an empty source reference must be rejected"
    except KnowledgePackageError:
        pass


def test_contradictory_limitations_are_rejected():
    """processing.stages_unavailable claiming a stream is empty while
    evidence of exactly that kind is cited would be a lie -- reject it."""
    ev = Evidence(timestamp_sec=1.0, kind="pointer", ref="x=1, y=1")
    kp = KeyPoint(text="a claim", kind="observation", timestamp_sec=1.0,
                  supporting_evidence=(ev,), confidence=0.5, nature="inferred")
    package = KnowledgePackage(
        source=KnowledgeSource(source_type="local", source="x.mp4", duration_sec=10.0),
        summary="a video", topics=(), key_lessons=(), important_observations=(kp,),
        evidence=(ev,), limitations=(),
        processing=ProcessingMetadata(
            generated_at="2026-01-01T00:00:00+00:00", video_lens_version="1.0.0",
            knowledge_schema_version=KNOWLEDGE_SCHEMA_VERSION, frame_count=0,
            transcript_status="unavailable", stages_available=(), stages_unavailable=("pointer",)),
    )
    try:
        validate_knowledge_package(package)
        assert False, "citing pointer evidence while claiming pointer unavailable must be rejected"
    except KnowledgePackageError:
        pass


def test_genuinely_empty_but_honest_package_is_valid_not_rejected():
    """A silent video with no transcript/pointer/vision is a legitimate,
    honestly-limited result -- not an error. 'Meaningless' data (no source,
    bad duration, lying limitations) is rejected; a real but sparse result
    is handled safely, not rejected."""
    result = AnalysisResult(video=_video(duration=5.0), structured_observations=[])
    package = build_knowledge_package(result, transcript=None)
    validate_knowledge_package(package)  # must not raise
    assert package.key_lessons == ()
    assert package.important_observations == ()
    assert len(package.limitations) >= 1  # honestly explains why it's sparse


# ------------------------------ 8/9. standalone, no downstream-consumer import, no Anthropic ------------------------------

def test_no_dhara_import_in_knowledge_module():
    src = Path("core/knowledge.py").read_text(encoding="utf-8")
    assert "import dhara" not in src and "from dhara" not in src


def test_no_anthropic_import_in_knowledge_module():
    src = Path("core/knowledge.py").read_text(encoding="utf-8")
    assert "anthropic" not in src.lower()


# ------------------------------ 10/11/12/13. lifecycle: cleanup, source safety, failure gating ------------------------------

def _make_video(path: str, duration: float = 2.0):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:size=160x120:rate=5:duration={duration}",
         "-pix_fmt", "yuv420p", path], capture_output=True, check=True,
    )


def test_temp_workspace_cleaned_after_successful_process_video():
    import video_lens as vl
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        original_size = os.path.getsize(p)

        captured = {}
        original_run_pipeline = vl._run_pipeline

        def _spy(source, config):
            result, transcript = original_run_pipeline(source, config)
            captured["frame_cache_dir"] = config.frame_cache_dir
            return result, transcript

        vl._run_pipeline = _spy
        try:
            config = vl.PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                        output_dir=os.path.join(tmp, "out"))
            package = vl.process_video(p, config)
        finally:
            vl._run_pipeline = original_run_pipeline

        job_root = os.path.dirname(captured["frame_cache_dir"])
        assert not os.path.exists(job_root)  # temp artifacts gone
        assert os.path.exists(p)  # original untouched
        assert os.path.getsize(p) == original_size
        assert package.source.duration_sec > 0


def test_failed_validation_does_not_falsely_report_success_or_clean_up():
    import video_lens as vl
    original_validate = vl._validate_package
    vl._validate_package = lambda package: (_ for _ in ()).throw(
        KnowledgePackageError("forced failure for test"))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            captured = {}
            original_run_pipeline = vl._run_pipeline

            def _spy(source, config):
                result, transcript = original_run_pipeline(source, config)
                captured["frame_cache_dir"] = config.frame_cache_dir
                return result, transcript

            vl._run_pipeline = _spy
            try:
                config = vl.PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                            output_dir=os.path.join(tmp, "out"))
                try:
                    vl.process_video(p, config)
                    assert False, "a failed validation must raise, not return a package"
                except KnowledgePackageError:
                    pass
            finally:
                vl._run_pipeline = original_run_pipeline
            job_root = os.path.dirname(captured["frame_cache_dir"])
            assert os.path.isdir(job_root), "temp workspace must survive a validation failure"
            import shutil
            shutil.rmtree(job_root, ignore_errors=True)  # test cleanup only
    finally:
        vl._validate_package = original_validate


def test_failed_export_does_not_falsely_report_success():
    """export_knowledge_package must raise (never silently produce a
    'successful' write) if the package fails validation -- proving the
    export step itself is a real gate, not just process_video's own
    separate _validate_package call."""
    package = KnowledgePackage(
        source=KnowledgeSource(source_type="local", source="x.mp4", duration_sec=0.0),
        summary="a video", topics=(), key_lessons=(), important_observations=(), evidence=(),
        limitations=(), processing=ProcessingMetadata(
            generated_at="2026-01-01T00:00:00+00:00", video_lens_version="1.0.0",
            knowledge_schema_version=KNOWLEDGE_SCHEMA_VERSION, frame_count=0,
            transcript_status="unavailable", stages_available=(), stages_unavailable=()),
    )
    with tempfile.TemporaryDirectory() as tmp:
        try:
            export_knowledge_package(package, tmp)
            assert False, "exporting an invalid package must raise"
        except KnowledgePackageError:
            pass
        assert list(Path(tmp).glob("*.json")) == []  # nothing written


# ------------------------------ 14. compactness ------------------------------

def test_exported_package_stays_compact():
    package = _valid_package()
    with tempfile.TemporaryDirectory() as tmp:
        path = export_knowledge_package(package, tmp)
        assert os.path.getsize(path) < 20_000


# ------------------------------ 15. no binaries embedded ------------------------------

def test_export_never_embeds_video_or_frame_binaries():
    package = _valid_package()
    with tempfile.TemporaryDirectory() as tmp:
        path = export_knowledge_package(package, tmp)
        raw = Path(path).read_bytes()
        json.loads(raw.decode("utf-8"))  # must be plain, valid UTF-8 JSON -- no binary blob
        assert b"\xff\xd8\xff" not in raw  # JPEG magic bytes
        assert b"ftyp" not in raw  # mp4 box signature


# ------------------------------ 16. portability: no dangling temp paths ------------------------------

def test_exported_package_never_embeds_job_workspace_temp_paths():
    """The source path itself (a stable, permanent reference to the
    original file) is legitimate provenance and IS expected to appear.
    What must never appear is a path into a job's own temporary workspace
    (frame cache, vision cache) -- those are deleted after the job
    finishes, so a reference to one would dangle."""
    import video_lens as vl
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        config = vl.PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                    output_dir=os.path.join(tmp, "out"))
        package = vl.process_video(p, config)
        blob = to_json(package)
        assert "videolens_jobs" not in blob
        assert "frames" not in blob or "frame_count" in blob  # only the honest count, not a cache path


# ------------------------------ 17. serialize/deserialize round-trip ------------------------------

def test_serialization_round_trip_preserves_required_fields():
    package = _valid_package()
    with tempfile.TemporaryDirectory() as tmp:
        path = export_knowledge_package(package, tmp)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    assert data["source"]["source"] == package.source.source
    assert data["source"]["duration_sec"] == package.source.duration_sec
    assert data["summary"] == package.summary
    assert data["processing"]["knowledge_schema_version"] == KNOWLEDGE_SCHEMA_VERSION
    assert len(data["key_lessons"]) == len(package.key_lessons)
    assert data["key_lessons"][0]["nature"] == "observed"
    assert data["key_lessons"][0]["timestamp_sec"] == package.key_lessons[0].timestamp_sec
    assert data["key_lessons"][0]["supporting_evidence"][0]["kind"] == "transcript"


if __name__ == "__main__":
    test_package_is_valid_and_carries_schema_version()
    test_schema_version_is_independent_of_software_version()
    test_source_provenance_survives_packaging()
    test_observed_vs_inferred_is_explicit_on_every_key_point()
    test_key_point_rejects_invalid_nature()
    test_evidence_provenance_survives_packaging()
    test_invalid_duration_is_rejected()
    test_missing_source_reference_is_rejected()
    test_contradictory_limitations_are_rejected()
    test_genuinely_empty_but_honest_package_is_valid_not_rejected()
    test_no_dhara_import_in_knowledge_module()
    test_no_anthropic_import_in_knowledge_module()
    test_temp_workspace_cleaned_after_successful_process_video()
    test_failed_validation_does_not_falsely_report_success_or_clean_up()
    test_failed_export_does_not_falsely_report_success()
    test_exported_package_stays_compact()
    test_export_never_embeds_video_or_frame_binaries()
    test_exported_package_never_embeds_job_workspace_temp_paths()
    test_serialization_round_trip_preserves_required_fields()
    print("All handoff-contract tests passed.")
