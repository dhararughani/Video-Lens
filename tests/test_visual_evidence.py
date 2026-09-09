"""Step 11 visual-evidence tests: selecting the few frames worth keeping,
collapsing near-duplicates, and getting them out of the job workspace before
cleanup destroys it.

The storage rules are the point here. A knowledge package that quietly grew an
image archive, or that base64'd frames into its JSON, would defeat the whole
"artifacts are temporary, knowledge is permanent" contract -- so those are
asserted directly, not assumed.

Run: python -m pytest tests/test_visual_evidence.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.contracts import Claim, Evidence, KeyPoint
from core.knowledge import to_json
from core.visual_evidence import select_and_bundle


def _write_image(path: str, shade: int) -> None:
    import cv2
    cv2.imwrite(path, np.full((64, 96), shade, dtype=np.uint8))


def _candidates(tmp: str, spec: list[tuple[str, float, int]]) -> dict[str, Evidence]:
    out = {}
    for evidence_id, ts, shade in spec:
        p = os.path.join(tmp, f"{evidence_id}.jpg")
        _write_image(p, shade)
        out[evidence_id] = Evidence(timestamp_sec=ts, kind="frame", ref=p)
    return out


def _claim(text: str, ts: float, frame_evidence: list[Evidence], confidence: float = 0.8) -> Claim:
    return Claim(text=text, kind="fact", status="observed", timestamp_sec=ts,
                  supporting_evidence=tuple(frame_evidence), confidence=confidence,
                  verification="visual_evidence_only")


# ------------------------------- selection -------------------------------

def test_only_frames_actual_knowledge_references_are_retained():
    """Selection is driven by demand: an unreferenced frame is not evidence of
    anything, however nice it looks."""
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 30), ("e2", 2.0, 140), ("e3", 3.0, 220)])
        claims = (_claim("about the first frame", 1.0, [cands["e1"]]),)
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out, stem="v")
        assert [v.evidence_id for v in visual] == ["e1"]
        assert len(list(Path(out).rglob("*.jpg"))) == 1


def test_near_duplicate_frames_are_collapsed():
    """A screencast where nothing changes must not yield one image per
    keyframe."""
    with tempfile.TemporaryDirectory() as tmp:
        # e1/e2/e3 are visually identical; e4 genuinely differs
        cands = _candidates(tmp, [("e1", 1.0, 100), ("e2", 2.0, 100),
                                   ("e3", 3.0, 100), ("e4", 4.0, 15)])
        claims = tuple(_claim(f"claim {i}", float(i), [cands[f"e{i}"]]) for i in range(1, 5))
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out, stem="v")
        ids = {v.evidence_id for v in visual}
        assert len(visual) == 2, f"expected the duplicates to collapse, kept {ids}"
        assert "e4" in ids


def test_no_fixed_quota_selection_scales_with_distinct_content():
    """The retained count follows the content, not a hardcoded number."""
    with tempfile.TemporaryDirectory() as tmp:
        spec = [(f"e{i}", float(i), i * 25) for i in range(1, 8)]  # 7 distinct shades
        cands = _candidates(tmp, spec)
        claims = tuple(_claim(f"c{i}", float(i), [cands[f"e{i}"]]) for i in range(1, 8))
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out, stem="v")
        assert len(visual) == 7


def test_max_items_is_a_ceiling_not_the_mechanism():
    with tempfile.TemporaryDirectory() as tmp:
        spec = [(f"e{i}", float(i), i * 25) for i in range(1, 8)]
        cands = _candidates(tmp, spec)
        claims = tuple(_claim(f"c{i}", float(i), [cands[f"e{i}"]]) for i in range(1, 8))
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out,
                                       stem="v", max_items=3)
        assert len(visual) == 3


def test_claims_are_linked_only_to_images_that_exist():
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 100), ("e2", 2.0, 100)])  # e2 duplicates e1
        claims = (_claim("first", 1.0, [cands["e1"]]), _claim("second", 2.0, [cands["e2"]]))
        out = os.path.join(tmp, "out")
        visual, rewritten = select_and_bundle(candidates=cands, claims=claims,
                                               output_dir=out, stem="v")
        kept = {v.evidence_id for v in visual}
        for claim in rewritten:
            for ref in claim.visual_evidence:
                assert ref in kept
        assert all(os.path.exists(os.path.join(out, v.image_path)) for v in visual)


def test_a_claim_whose_frame_was_deduped_links_to_the_stand_in():
    """Dropping a duplicate must not cost the claim its illustration -- the
    identical content is still in the bundle under the retained id."""
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 100), ("e2", 2.0, 100)])  # visually identical
        second = _claim("about the second frame", 2.0, [cands["e2"]])
        out = os.path.join(tmp, "out")
        visual, rewritten = select_and_bundle(candidates=cands, claims=(second,),
                                               output_dir=out, stem="v")
        assert len(visual) == 1
        stand_in = visual[0].evidence_id
        assert rewritten[0].visual_evidence == (stand_in,)
        assert os.path.exists(os.path.join(out, visual[0].image_path))


def test_missing_frame_files_are_skipped_never_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 30)])
        cands["ghost"] = Evidence(timestamp_sec=2.0, kind="frame",
                                   ref=os.path.join(tmp, "never_written.jpg"))
        claims = (_claim("real", 1.0, [cands["e1"]]), _claim("ghost", 2.0, [cands["ghost"]]))
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out, stem="v")
        assert [v.evidence_id for v in visual] == ["e1"]


def test_a_claim_corroborated_by_vision_evidence_still_gets_its_image():
    """Step 12 regression: `vision` Evidence.ref is the model's description
    text, not a frame path -- a claim citing vision (genuine corroboration,
    per core/synthesis.py) must still resolve back to the frame that vision
    interpreted, or a 'corroborated' claim would ship with no picture at
    all. Found via a real live-vision run in Step 12."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "e1.jpg")
        _write_image(p, 80)
        frame_ev = Evidence(timestamp_sec=5.0, kind="frame", ref=p)
        vision_ev = Evidence(timestamp_sec=5.0, kind="vision",
                              ref="a diagram of four labelled boxes", confidence=0.7)
        claim = Claim(text="the diagram shows four locations", kind="concept", status="observed",
                       timestamp_sec=5.0, supporting_evidence=(vision_ev,), confidence=0.7,
                       verification="speech_corroborated_by_visual_evidence")
        out = os.path.join(tmp, "out")
        visual, rewritten = select_and_bundle(candidates={"e1": frame_ev}, claims=(claim,),
                                               output_dir=out, stem="v")
        assert len(visual) == 1
        assert rewritten[0].visual_evidence == (visual[0].evidence_id,)
        assert os.path.exists(os.path.join(out, visual[0].image_path))


def test_deterministic_key_points_also_get_visual_evidence():
    """Without a synthesizer there are no claims -- visual evidence must still
    work, anchored to the nearest frame in time."""
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 30), ("e2", 9.0, 200)])
        point = KeyPoint(text="a lesson", kind="warning", timestamp_sec=1.2,
                          supporting_evidence=(Evidence(timestamp_sec=1.2, kind="transcript",
                                                         ref="a lesson"),),
                          confidence=0.4)
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, key_points=(point,),
                                       output_dir=out, stem="v")
        assert [v.evidence_id for v in visual] == ["e1"]


# ------------------------------- storage discipline -------------------------------

def test_image_paths_are_relative_and_not_machine_specific():
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 30)])
        claims = (_claim("x", 1.0, [cands["e1"]]),)
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out, stem="myvid")
        path = visual[0].image_path
        assert not os.path.isabs(path)
        assert path.startswith("myvid_evidence/")
        assert tmp not in path and "\\" not in path
        # and it resolves from the package's own directory
        assert os.path.exists(os.path.join(out, path))


def test_no_image_bytes_are_embedded_in_the_serialized_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        cands = _candidates(tmp, [("e1", 1.0, 30)])
        claims = (_claim("x", 1.0, [cands["e1"]]),)
        out = os.path.join(tmp, "out")
        visual, _ = select_and_bundle(candidates=cands, claims=claims, output_dir=out, stem="v")
        blob = json.dumps([v.__dict__ for v in visual])
        assert "base64" not in blob
        assert len(blob) < 2000, "visual evidence must be references, not payloads"


# ------------------------------- lifecycle -------------------------------

class _FrameCitingSynthesizer:
    """Cites every frame the brief offers -- exercises evidence selection on a
    real pipeline run without needing speech."""

    def synthesize(self, brief) -> dict:
        frames = [i for i in brief.items if i.evidence.kind == "frame"]
        return {"provider": "fake-frame-synth", "summary": "A synthetic test pattern video.",
                "claims": [
                    {"text": f"The screen shows test content at {i.evidence.timestamp_sec:.1f}s.",
                     "kind": "observation", "nature": "observed",
                     "timestamp_sec": i.evidence.timestamp_sec,
                     "evidence_ids": [i.evidence_id], "confidence": 0.7}
                    for i in frames
                ]}


def _make_video(path: str, duration: float = 4.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=size=160x120:rate=5:duration={duration}", "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )


def test_evidence_is_copied_out_before_the_workspace_is_destroyed():
    """The whole ordering risk of Step 11 in one test: selected frames live in
    the job workspace, and cleanup deletes it. If the bundle were written after
    cleanup, this would find an empty directory."""
    import video_lens as vl
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "v.mp4")
        _make_video(video)
        out = os.path.join(tmp, "out")
        config = vl.PipelineConfig(pointer_enabled=False, vision_enabled=False,
                                    keyframe_interval_sec=1.0, output_dir=out,
                                    knowledge_synthesizer=_FrameCitingSynthesizer())

        captured = {}
        original = vl._run_pipeline

        def _spy(source, cfg):
            result, transcript = original(source, cfg)
            captured["frame_cache_dir"] = cfg.frame_cache_dir
            return result, transcript

        vl._run_pipeline = _spy
        try:
            package = vl.process_video(video, config)
        finally:
            vl._run_pipeline = original

        job_root = os.path.dirname(captured["frame_cache_dir"])
        assert not os.path.exists(job_root), "workspace must still be cleaned up"

        assert package.visual_evidence, "evidence frames must survive cleanup"
        for v in package.visual_evidence:
            resolved = os.path.join(out, v.image_path)
            assert os.path.exists(resolved), f"{v.image_path} did not survive"
            assert os.path.getsize(resolved) > 0
        # every claim's image reference resolves
        kept = {v.evidence_id for v in package.visual_evidence}
        for claim in package.claims:
            assert set(claim.visual_evidence).issubset(kept)


def test_durable_output_is_knowledge_plus_a_small_bundle_not_a_media_archive():
    import video_lens as vl
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "v.mp4")
        _make_video(video, duration=6.0)
        source_size = os.path.getsize(video)
        out = os.path.join(tmp, "out")
        config = vl.PipelineConfig(pointer_enabled=False, vision_enabled=False,
                                    keyframe_interval_sec=1.0, output_dir=out,
                                    knowledge_synthesizer=_FrameCitingSynthesizer())
        package = vl.process_video(video, config)

        total = sum(f.stat().st_size for f in Path(out).rglob("*") if f.is_file())
        assert total < source_size, "durable output must be smaller than the source video"
        # the JSON itself stays tiny -- images are files beside it, not inside it
        blob = to_json(package)
        assert "base64," not in blob
        assert len(blob) < 60_000


def test_visual_evidence_can_be_disabled():
    import video_lens as vl
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "v.mp4")
        _make_video(video)
        out = os.path.join(tmp, "out")
        config = vl.PipelineConfig(pointer_enabled=False, vision_enabled=False,
                                    output_dir=out, visual_evidence_enabled=False,
                                    knowledge_synthesizer=_FrameCitingSynthesizer())
        package = vl.process_video(video, config)
        assert package.visual_evidence == ()
        assert not list(Path(out).rglob("*.jpg"))
        assert all(c.visual_evidence == () for c in package.claims)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
