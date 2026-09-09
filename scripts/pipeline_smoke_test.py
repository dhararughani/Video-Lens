"""Final end-to-end smoke test: the canonical `video_lens.analyze_video`
pipeline against a real video, local file or URL, measuring what actually
happened at every stage. This is the Step 8 release-readiness check --
prints exactly what ran, what didn't, and how long it took. No fabricated
numbers: an unavailable/skipped stage is reported as such, not hidden.

    python scripts/pipeline_smoke_test.py <video_path_or_url> [max_frames]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_lens import PipelineConfig, analyze_video


def main():
    source = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    config = PipelineConfig(
        max_frames=max_frames,
        frame_cache_dir="pipeline_smoke_test_output/frame_cache",
        vision_cache_dir="pipeline_smoke_test_output/vision_cache",
    )

    print(f"Running canonical pipeline on: {source}")
    print(f"ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT set'}")
    t0 = time.time()
    result = analyze_video(source, config)
    elapsed = time.time() - t0

    v = result.video
    print(f"\n--- INGESTION / METADATA ---")
    print(f"duration={v.duration_sec:.1f}s  resolution={v.width}x{v.height}  fps={v.fps}  "
          f"has_audio={v.has_audio}  source_type={v.source_type}")

    print(f"\n--- STAGES ---")
    print(f"structured observations: {len(result.structured_observations)}")
    frame_count = sum(1 for o in result.observations if o.frame is not None)
    transcript_evidence = sum(1 for e in result.evidence if e.kind == "transcript")
    pointer_evidence = sum(1 for e in result.evidence if e.kind == "pointer")
    vision_evidence = sum(1 for e in result.evidence if e.kind == "vision")
    print(f"frames referenced: {frame_count}")
    print(f"transcript evidence items: {transcript_evidence}")
    print(f"pointer evidence items: {pointer_evidence}")
    print(f"vision evidence items: {vision_evidence}")

    unavailable_counts: dict[str, int] = {}
    for so in result.structured_observations:
        for stream in so.unavailable:
            unavailable_counts[stream] = unavailable_counts.get(stream, 0) + 1
    print(f"unavailable-stream counts across all observations: {unavailable_counts}")

    n_inferences = sum(len(so.inferences) for so in result.structured_observations)
    n_disagreements = sum(len(so.disagreements) for so in result.structured_observations)
    print(f"total inferences drawn: {n_inferences}")
    print(f"total disagreements recorded: {n_disagreements}")

    timestamps = [so.timestamp_sec for so in result.structured_observations]
    coherent = timestamps == sorted(timestamps) and all(0.0 <= t <= v.duration_sec + 1.0 for t in timestamps)
    print(f"\ntimestamps coherent (ordered, within video duration): {coherent}")

    print(f"\ntotal processing time: {elapsed:.1f}s")

    print(f"\n--- PER-TIMESTAMP DETAIL ---")
    for so in result.structured_observations:
        print(f"\nt={so.timestamp_sec:.2f}s  observed={len(so.observed)}  "
              f"unavailable={list(so.unavailable)}")
        for inf in so.inferences:
            print(f"  inference [{inf.basis}] conf={inf.confidence:.2f}: {inf.text[:100]}")
        for d in so.disagreements:
            print(f"  disagreement: {d}")

    print("\nPipeline smoke test finished.")


if __name__ == "__main__":
    main()
