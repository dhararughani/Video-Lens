"""End-to-end frame-intelligence smoke test on a real video: ingest ->
transcribe -> pick a real speech segment -> select candidate frames for it
-> select whole-video keyframes. Prints a performance/storage summary.

    python scripts/frame_smoke_test.py <video_path_or_url> [out_dir]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import frames_for_segment, select_keyframes
from adapters.ingestion import ingest
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter


def main():
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "frame_smoke_test_output"

    print(f"[1/5] Ingesting: {source}")
    video = ingest(source)
    print(f"      {video.duration_sec:.1f}s  {video.width}x{video.height}  fps={video.fps}")

    print("[2/5] Transcribing (CPU)")
    transcript = FasterWhisperAdapter().transcribe(video)
    print(f"      status={transcript.status} segments={len(transcript.segments)}")

    extractor = FrameExtractor(cache_dir=os.path.join(out_dir, "cache"))

    if transcript.segments:
        mid = transcript.segments[len(transcript.segments) // 2]
        print(f"[3/5] Speech -> frame bridge for segment "
              f"[{mid.start_sec:.2f}s - {mid.end_sec:.2f}s] \"{mid.text[:60]}\"")
        t0 = time.time()
        candidates = frames_for_segment(video, mid, extractor, interval_sec=1.0)
        elapsed = time.time() - t0
        for f in candidates:
            print(f"      frame @ {f.timestamp_sec:6.2f}s  reason={f.reason:16s} {f.path}")
        assert all(mid.start_sec - 1e-6 <= f.timestamp_sec <= mid.end_sec + 1e-6 for f in candidates)
        starts = [f.timestamp_sec for f in candidates]
        assert starts == sorted(starts)
        print(f"      {len(candidates)} candidate frames in {elapsed:.2f}s")
    else:
        print("[3/5] No speech segments -- skipping speech->frame bridge")

    print("[4/5] Whole-video keyframe selection")
    t0 = time.time()
    keyframes = select_keyframes(video, extractor, interval_sec=2.0)
    elapsed = time.time() - t0
    naive_frame_count = int(video.duration_sec * video.fps) if video.fps else 0
    uniform_2s_count = int(video.duration_sec / 2.0) + 1
    print(f"      {len(keyframes)} keyframes kept out of {uniform_2s_count} sampled "
          f"(naive full decode would be ~{naive_frame_count} frames)")
    print(f"      {elapsed:.2f}s for whole-video keyframe selection")

    print("[5/5] Storage")
    cache_dir = os.path.join(out_dir, "cache")
    if os.path.isdir(cache_dir):
        files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)]
        total = sum(os.path.getsize(f) for f in files)
        print(f"      {len(files)} cached frame files, {total / 1024:.1f} KB total, "
              f"{total / max(len(files), 1):.0f} bytes/frame average")

    print("Frame smoke test passed.")


if __name__ == "__main__":
    main()
