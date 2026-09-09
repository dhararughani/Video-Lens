"""End-to-end evidence/structured-understanding smoke test on a real video:
ingest -> transcribe -> select frames -> pointer evidence -> vision analysis
-> correlate everything into StructuredObservations -> print the "what
happened at this point in the video" answer for each, with observed facts,
inferences, disagreements, and unavailable streams shown separately.

Works with or without ANTHROPIC_API_KEY -- without one, vision degrades to
status="unavailable" and this script demonstrates that the rest of the
pipeline (and evidence correlation) still produces a coherent, honest
result rather than failing.

    python scripts/evidence_smoke_test.py <video_path_or_url> [start_sec] [duration_sec] [out_dir]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import frames_for_segment
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import detect_pointer_for_frames
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
from adapters.vision.claude_vision import ClaudeVisionAdapter
from core.evidence import build_structured_observation


def main():
    source = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "evidence_smoke_test_output"

    print(f"[1/6] Ingesting: {source}")
    video = ingest(source)
    print(f"      {video.duration_sec:.1f}s  {video.width}x{video.height}  fps={video.fps}")

    print("[2/6] Transcribing (CPU)")
    transcript = FasterWhisperAdapter().transcribe(video)
    seg = transcript.segment_at(start_sec) or (transcript.segments[0] if transcript.segments else None)
    if not seg:
        print("      no speech segment available -- aborting smoke test")
        return
    print(f"      segment [{seg.start_sec:.2f}s - {seg.end_sec:.2f}s] \"{seg.text[:70]}\"")

    extractor = FrameExtractor(cache_dir=os.path.join(out_dir, "frame_cache"))
    print("[3/6] Selecting candidate frames")
    frames = frames_for_segment(video, seg, extractor, interval_sec=1.0)[:5]
    print(f"      {len(frames)} frames selected")

    print("[4/6] Pointer evidence")
    pointer_events = detect_pointer_for_frames(frames)

    print("[5/6] Vision analysis (ANTHROPIC_API_KEY "
          + ("found" if os.environ.get("ANTHROPIC_API_KEY") else "NOT set -- results will be 'unavailable'"))
    vision = ClaudeVisionAdapter(cache_dir=os.path.join(out_dir, "vision_cache"))
    vision_observations = []
    for frame, pointer in zip(frames, pointer_events):
        context = transcript.text_around(frame.timestamp_sec, window_sec=5.0)
        vision_observations.append(vision.analyze_frame(frame, transcript_context=context, pointer=pointer))

    print("[6/6] Structured understanding (evidence correlation, tolerance=1.0s)")
    query_timestamps = [f.timestamp_sec for f in frames]
    for t in query_timestamps:
        so = build_structured_observation(
            t, transcript=transcript, frames=frames, pointer_events=pointer_events,
            vision_observations=vision_observations, tolerance_sec=1.0,
        )
        print(f"\n      === t={so.timestamp_sec:.2f}s (window +/-{so.tolerance_sec}s) ===")
        print(f"      observed ({len(so.observed)}):")
        for e in so.observed:
            conf = f" conf={e.confidence:.2f}" if e.confidence is not None else ""
            print(f"        [{e.kind}]{conf} {e.ref[:90]}")
        print(f"      unavailable: {list(so.unavailable) or 'none'}")
        print(f"      inferences ({len(so.inferences)}):")
        for inf in so.inferences:
            print(f"        [{inf.basis}] conf={inf.confidence:.2f}: {inf.text}")
        if so.disagreements:
            print(f"      disagreements: {list(so.disagreements)}")

    n_unavailable_vision = sum(1 for v in vision_observations if v.status == "unavailable")
    print(f"\nVision analysis: {len(vision_observations) - n_unavailable_vision}/{len(vision_observations)} "
          f"produced real observations ({n_unavailable_vision} unavailable -- no credential).")
    print("Evidence smoke test finished.")


if __name__ == "__main__":
    main()
