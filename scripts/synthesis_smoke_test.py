"""Step 11 real-video validation: runs the full lifecycle on a real video and
reports what the semantic layer actually produced -- including what it
REJECTED, which is the part worth watching.

Two phases, because Video-Lens ships no synthesizer and this script must not
require an API key, an account, or a paid service:

    # 1. run the pipeline and write the brief a synthesizer would receive
    python scripts/synthesis_smoke_test.py <video> --brief-out brief.json

    # 2. author/produce response.json from that brief with ANY model, then:
    python scripts/synthesis_smoke_test.py <video> --brief-out brief.json \
        --response response.json

Phase 2 re-runs the pipeline; evidence ids are assigned deterministically from
the evidence order, so ids authored against phase 1's brief still resolve.

The same `--response` mechanism is how you would wire a real provider: a
KnowledgeSynthesizer is just "given this brief, return this JSON".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge import render_markdown, to_json
from video_lens import PipelineConfig, process_video


class FileSynthesizer:
    """Writes the brief it is handed, and replays a previously-authored
    response if one exists. Raising when there is no response yet is a
    supported outcome -- the job still completes with deterministic knowledge
    and records `synthesis.status == "failed"`, which is exactly the honest
    fallback Step 11 requires."""

    def __init__(self, brief_path: str, response_path: str | None):
        self.brief_path = brief_path
        self.response_path = response_path

    def synthesize(self, brief) -> dict:
        with open(self.brief_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": asdict(brief.source),
                "system_prompt": brief.system_prompt,
                "user_prompt": brief.user_prompt,
                "evidence": [{"id": i.evidence_id, "kind": i.evidence.kind,
                              "timestamp_sec": i.evidence.timestamp_sec,
                              "ref": i.evidence.ref, "confidence": i.evidence.confidence}
                             for i in brief.items],
                "disagreements": list(brief.disagreements),
                "stages_unavailable": list(brief.stages_unavailable),
            }, f, indent=2)
        print(f"[brief] {len(brief.items)} evidence items -> {self.brief_path}")
        if self.response_path and os.path.exists(self.response_path):
            with open(self.response_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError("no synthesizer response available yet (brief written for authoring)")


def _dir_size(path: str) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--brief-out", default="synthesis_brief.json")
    ap.add_argument("--response", default=None)
    ap.add_argument("--output-dir", default="synthesis_smoke_test_output")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--keyframe-interval", type=float, default=8.0)
    args = ap.parse_args()

    source_size = os.path.getsize(args.source) if os.path.exists(args.source) else None
    print(f"source: {args.source}")
    if source_size:
        print(f"source size: {source_size / 1_000_000:.2f} MB")

    config = PipelineConfig(
        max_frames=args.max_frames, keyframe_interval_sec=args.keyframe_interval,
        vision_enabled=False,  # no API key is used anywhere in this script
        output_dir=args.output_dir,
        knowledge_synthesizer=FileSynthesizer(args.brief_out, args.response),
    )

    t0 = time.time()
    package = process_video(args.source, config)
    elapsed = time.time() - t0

    print(f"\n--- RESULT ({elapsed:.1f}s) ---")
    print(f"title: {package.source.title}")
    print(f"duration: {package.source.duration_sec:.1f}s")
    print(f"synthesis: {package.synthesis}")
    print(f"\nDETERMINISTIC summary: {package.summary}")
    print(f"\nSEMANTIC summary: {package.semantic_summary}")

    print(f"\n--- CLAIMS ({len(package.claims)}) ---")
    for c in package.claims:
        span = (f"{c.timestamp_sec:.1f}s" if c.timestamp_end_sec is None
                else f"{c.timestamp_sec:.1f}-{c.timestamp_end_sec:.1f}s")
        print(f"[{c.status}/{c.kind}] ({span}, conf {c.confidence:.2f}, {c.verification})")
        print(f"    {c.text}")
        print(f"    evidence: {[(e.kind, round(e.timestamp_sec, 1)) for e in c.supporting_evidence]}")
        if c.visual_evidence:
            print(f"    visual: {list(c.visual_evidence)}")
        for note in c.limitations:
            print(f"    limitation: {note}")

    print(f"\n--- VISUAL EVIDENCE ({len(package.visual_evidence)}) ---")
    for v in package.visual_evidence:
        print(f"{v.evidence_id} @ {v.timestamp_sec:.1f}s  {v.image_path}  "
              f"{v.width}x{v.height}  {v.byte_size / 1024:.1f} KB")
        print(f"    {v.selection_reason}")

    print(f"\n--- LIMITATIONS ---")
    for note in package.limitations:
        print(f"  - {note}")

    print(f"\n--- STORAGE ---")
    json_files = list(Path(args.output_dir).glob("*.json"))
    json_size = json_files[0].stat().st_size if json_files else 0
    total = _dir_size(args.output_dir)
    print(f"knowledge package JSON: {json_size} bytes")
    print(f"total durable output (package + evidence bundle): {total} bytes")
    if source_size:
        print(f"vs source video: {total / source_size * 100:.3f}% of the original")
    print(f"source still present and untouched: {os.path.exists(args.source)}")

    md = Path(args.output_dir) / "preview.md"
    md.write_text(render_markdown(package), encoding="utf-8")
    print(f"markdown preview: {md}")
    assert "base64," not in to_json(package), "package must never embed image data"


if __name__ == "__main__":
    main()
