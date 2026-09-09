"""Step 9 end-to-end real-video validation: runs process_video() on a real
local video, reports what actually happened at each lifecycle stage, and
measures the before/after storage footprint (Part 11/12 of the Step 9
task). No Anthropic API key is used or required -- vision honestly degrades
to unavailable without one, and this script never sets or reads
ANTHROPIC_API_KEY.

    python scripts/lifecycle_smoke_test.py <video_path> [output_dir]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.knowledge import to_json
from video_lens import PipelineConfig, process_video


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def main():
    source = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "lifecycle_smoke_test_output"

    source_size = os.path.getsize(source) if os.path.exists(source) else None
    print(f"Running Step 9 lifecycle on: {source}")
    print(f"source video size: {source_size / 1_000_000:.2f} MB" if source_size else "source is a URL")

    before_out = _dir_size(output_dir) if os.path.exists(output_dir) else 0

    config = PipelineConfig(max_frames=8, output_dir=output_dir, keyframe_interval_sec=5.0)
    t0 = time.time()
    package = process_video(source, config)
    elapsed = time.time() - t0

    print(f"\n--- LIFECYCLE RESULT ---")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"title: {package.source.title}")
    print(f"duration: {package.source.duration_sec:.1f}s  source_type: {package.source.source_type}")
    print(f"topics: {list(package.topics)}")
    print(f"key_lessons: {len(package.key_lessons)}")
    print(f"important_observations: {len(package.important_observations)}")
    print(f"evidence citations: {len(package.evidence)}")
    print(f"limitations:")
    for note in package.limitations:
        print(f"  - {note}")
    print(f"processing: {package.processing}")

    print(f"\n--- ORIGINAL SOURCE UNTOUCHED ---")
    still_there = os.path.exists(source)
    same_size = os.path.getsize(source) == source_size if still_there and source_size else None
    print(f"source still exists: {still_there}  unchanged size: {same_size}")

    print(f"\n--- STORAGE MEASUREMENT ---")
    after_out = _dir_size(output_dir)
    out_files = list(Path(output_dir).glob("*.json"))
    package_size = out_files[0].stat().st_size if out_files else 0
    print(f"durable output directory size (before this run): {before_out} bytes")
    print(f"durable output directory size (after this run):  {after_out} bytes")
    print(f"knowledge package file size: {package_size} bytes")
    print(f"knowledge package JSON preview (first 500 chars):")
    print(to_json(package)[:500])

    print("\nLifecycle smoke test finished -- no video/frame/cache artifacts "
          "should remain under any videolens_jobs/ temp directory for this job.")


if __name__ == "__main__":
    main()
