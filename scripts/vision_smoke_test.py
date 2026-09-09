"""End-to-end multimodal-evidence smoke test on a real video: ingest ->
transcribe -> select frames for a real speech segment -> get pointer
evidence for those frames -> send each (frame, transcript context, pointer)
to the Vision Adapter -> join everything into MultimodalObservation
("VideoEvidence") records ordered by timestamp. Prints a measurable
report -- no fabricated precision.

Needs ANTHROPIC_API_KEY (or another credential the `anthropic` SDK can
resolve) to get real "ok" analyses; without one every observation will
honestly report status="unavailable" -- the rest of the pipeline (frame
selection, pointer evidence, evidence joining) is still exercised and
verified either way.

    python scripts/vision_smoke_test.py <video_path_or_url> [start_sec] [duration_sec] [out_dir]
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
from adapters.pointer.cursor_detector import detect_pointer_for_frames
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
from adapters.vision.claude_vision import ClaudeVisionAdapter
from core.contracts import MultimodalObservation


def main():
    source = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "vision_smoke_test_output"

    print(f"[1/5] Ingesting: {source}")
    video = ingest(source)
    print(f"      {video.duration_sec:.1f}s  {video.width}x{video.height}  fps={video.fps}")

    print("[2/5] Transcribing (CPU)")
    transcript = FasterWhisperAdapter().transcribe(video)
    seg = transcript.segment_at(start_sec) or (transcript.segments[0] if transcript.segments else None)
    if not seg:
        print("      no speech segment available -- aborting smoke test")
        return
    print(f"      segment [{seg.start_sec:.2f}s - {seg.end_sec:.2f}s] \"{seg.text[:70]}\"")

    extractor = FrameExtractor(cache_dir=os.path.join(out_dir, "frame_cache"))
    print("[3/5] Selecting candidate frames for the segment")
    frames = frames_for_segment(video, seg, extractor, interval_sec=1.0)
    frames = frames[:5]  # keep the smoke test small and cheap
    print(f"      {len(frames)} frames selected")

    print("[4/5] Pointer evidence for the selected frames")
    pointer_events = detect_pointer_for_frames(frames)
    for p in pointer_events:
        print(f"      t={p.timestamp_sec:6.2f}s  pointer={p.status}"
              + (f" ({p.normalized_x:.2f},{p.normalized_y:.2f})" if p.x is not None else ""))

    print("[5/5] Vision analysis -> multimodal evidence")
    vision = ClaudeVisionAdapter(cache_dir=os.path.join(out_dir, "vision_cache"))
    evidence: list[MultimodalObservation] = []
    t0 = time.time()
    for frame, pointer in zip(frames, pointer_events):
        context = transcript.text_around(frame.timestamp_sec, window_sec=5.0)
        obs = vision.analyze_frame(frame, transcript_context=context, pointer=pointer)
        evidence.append(MultimodalObservation(
            timestamp_sec=frame.timestamp_sec, frame=frame, transcript_context=context,
            description=obs.description, pointer=pointer, vision=obs,
        ))
    elapsed = time.time() - t0

    n_ok = sum(1 for e in evidence if e.vision.status == "ok")
    n_low = sum(1 for e in evidence if e.vision.status == "low_information")
    n_failed = sum(1 for e in evidence if e.vision.status == "failed")
    n_unavail = sum(1 for e in evidence if e.vision.status == "unavailable")

    for e in evidence:
        v = e.vision
        print(f"\n      t={e.timestamp_sec:6.2f}s  vision={v.status}  confidence={v.confidence:.2f}")
        if v.status == "ok" or v.status == "low_information":
            print(f"        description: {v.description}")
            if v.visible_text:
                print(f"        visible_text: {list(v.visible_text)}")
            for el in v.elements:
                region = f"region=({el.region.x1:.2f},{el.region.y1:.2f})-({el.region.x2:.2f},{el.region.y2:.2f})" \
                    if el.region else "region=unknown"
                print(f"        element: {el.kind} -- {el.description} [{region}, {el.region_confidence}]")
        elif v.status == "unavailable":
            print(f"        reason: {v.analysis_metadata.get('reason')}")
        else:
            print(f"        error: {v.analysis_metadata.get('error')}")

    print(f"\n      analyzed: {len(evidence)}  ok: {n_ok}  low_information: {n_low}  "
          f"failed: {n_failed}  unavailable: {n_unavail}")
    if elapsed and len(evidence):
        print(f"      total time: {elapsed:.2f}s ({elapsed / len(evidence):.2f}s/frame)")

    cache_dir = os.path.join(out_dir, "vision_cache")
    if os.path.isdir(cache_dir):
        files = os.listdir(cache_dir)
        print(f"      vision cache: {len(files)} entries")

    print("\nVision smoke test finished.")


if __name__ == "__main__":
    main()
