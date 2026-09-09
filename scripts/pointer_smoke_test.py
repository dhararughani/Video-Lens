"""End-to-end pointer-intelligence smoke test on a real video: ingest ->
track the pointer across a window -> (if speech exists) bridge one speech
segment to candidate frames -> detect the pointer across them. Prints a
measurable detection-rate/performance/storage report -- no fabricated
precision, only what was actually observed.

    python scripts/pointer_smoke_test.py <video_path_or_url> [start_sec] [duration_sec] [out_dir]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import frames_for_segment
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import detect_pointer_for_frames, track_pointer
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter


def main():
    source = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "pointer_smoke_test_output"

    print(f"[1/4] Ingesting: {source}")
    video = ingest(source)
    print(f"      {video.duration_sec:.1f}s  {video.width}x{video.height}  fps={video.fps}")

    end_sec = min(start_sec + duration_sec, video.duration_sec)
    extractor = FrameExtractor(cache_dir=os.path.join(out_dir, "cache"))

    print(f"[2/4] Tracking pointer over [{start_sec:.1f}s - {end_sec:.1f}s], 0.5s interval")
    t0 = time.time()
    track = track_pointer(video, start_sec, end_sec, extractor, interval_sec=0.5)
    elapsed = time.time() - t0
    n = len(track.events)
    n_detected = sum(1 for e in track.events if e.status == "detected")
    n_uncertain = sum(1 for e in track.events if e.status == "uncertain")
    n_missing = sum(1 for e in track.events if e.status == "not_detected")
    for e in track.events:
        loc = f"x={e.x:4d} y={e.y:4d}" if e.x is not None else "  (no position)  "
        print(f"      t={e.timestamp_sec:6.2f}s  {e.status:12s} {loc}  conf={e.confidence:.2f}")
    print(f"      frames analyzed: {n}  detected: {n_detected}  uncertain: {n_uncertain}  "
          f"not_detected: {n_missing}")
    print(f"      detection rate (detected/total): {n_detected / n:.1%}" if n else "      no frames")
    print(f"      track.confidence: {track.confidence:.2f}")
    print(f"      processing time: {elapsed:.2f}s ({n / elapsed:.1f} frames/s)" if elapsed else "")

    print("[3/4] Speech -> frame -> pointer bridge")
    transcript = FasterWhisperAdapter().transcribe(video)
    seg = transcript.segment_at(start_sec) or (transcript.segments[0] if transcript.segments else None)
    if seg:
        print(f"      segment [{seg.start_sec:.2f}s - {seg.end_sec:.2f}s] \"{seg.text[:60]}\"")
        frames = frames_for_segment(video, seg, extractor, interval_sec=0.5)
        events = detect_pointer_for_frames(frames)
        for e in events:
            print(f"      t={e.timestamp_sec:6.2f}s  {e.status}")
    else:
        print("      no speech segment available -- skipping")

    print("[4/4] Storage")
    cache_dir = os.path.join(out_dir, "cache")
    if os.path.isdir(cache_dir):
        files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)]
        total = sum(os.path.getsize(f) for f in files)
        print(f"      {len(files)} cached frame files, {total / 1024:.1f} KB total, "
              f"{total / max(len(files), 1):.0f} bytes/frame average")

    print("Pointer smoke test finished.")


if __name__ == "__main__":
    main()
