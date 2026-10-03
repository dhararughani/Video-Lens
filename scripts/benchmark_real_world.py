"""Real-footage benchmark for Video-Lens (Step 7). Measures; changes nothing.

    python scripts/benchmark_real_world.py end-of-video  VIDEO
    python scripts/benchmark_real_world.py inspection    VIDEO
    python scripts/benchmark_real_world.py visual-change VIDEO [--label-th 10:30,46:57]
    python scripts/benchmark_real_world.py cursor        VIDEO

Takes any local video; no network, no API key, no model. Frames are extracted
into a temp directory that is removed on exit. Results are printed as JSON.
Findings from the corpus this was written against are in docs/benchmark.md.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import video_lens
from adapters.frames import frame_extractor as fe
from adapters.frames.frame_extractor import FrameExtractor
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import detect_pointer_for_frames
from core.contracts import InspectionRequest, PointerTrack, Region
from core.cursor_intelligence import analyze_track
from core.errors import FrameExtractionError
from core.visual_change import changed_pixel_mask, detect_visual_changes


def _reference_frame(path: str, index: int, out: str):
    """Ground truth: ffmpeg's own decode of frame number `index`, as PNG."""
    subprocess.run(["ffmpeg", "-y", "-i", path, "-vf", f"select=eq(n\\,{index})", "-fps_mode", "passthrough",
                    "-frames:v", "1", out], capture_output=True, check=True)
    return cv2.imread(out)


def _mad(a, b) -> float:
    return float("nan") if a.shape != b.shape else float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def end_of_video(path: str, tmp: str) -> dict:
    """Which timestamps near the end can `get_frame` serve, vs. what the
    container and the video stream each claim the duration is."""
    video = ingest(path)
    stream = float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout)
    ex, rows = FrameExtractor(cache_dir=tmp), {}
    for label, ts in {"dur-5": video.duration_sec - 5, "dur-1": video.duration_sec - 1, "dur-0.5": video.duration_sec - 0.5,
                      "dur-0.1": video.duration_sec - 0.1, "dur": video.duration_sec, "dur+50": video.duration_sec + 50}.items():
        try:
            rows[label] = f"ok @{ex.get_frame(video, ts).timestamp_sec}"
        except FrameExtractionError as e:
            rows[label] = f"FrameExtractionError: {str(e)[:60]}"
    return {"container_duration": video.duration_sec, "video_stream_duration": stream,
            "gap_sec": round(video.duration_sec - stream, 4), "gap_frames": round((video.duration_sec - stream) * video.fps, 2),
            "get_frame": rows}


def inspection(path: str, tmp: str) -> dict:
    """P0-A: timestamp, window, crop, scale, determinism and cache identity, each checked against an
    independent decode of the same video."""
    video = ingest(path)
    ex = FrameExtractor(cache_dir=os.path.join(tmp, "c1"))
    out = {"exact": [], "windows": {}}
    for t in (video.duration_sec * f for f in (0.05, 0.3, 0.55, 0.8)):
        frame = ex.get_frame(video, t)
        index = round(frame.timestamp_sec * video.fps)
        mine = cv2.imread(frame.path)
        by_offset = {k: _mad(mine, _reference_frame(path, index + k, os.path.join(tmp, f"r{k}.png"))) for k in (-2, -1, 0, 1, 2)}
        out["exact"].append({"t": round(t, 2), "returned": frame.timestamp_sec, "nearest_reference_offset_frames": min(by_offset, key=by_offset.get),
                             "mad_by_offset": {k: round(v, 2) for k, v in by_offset.items()}})
    mid = video.duration_sec / 2
    for label, (before, after, fps) in {"4s@2fps": (2, 2, 2.0), "2s@5fps": (1, 1, 5.0), "1s@1fps": (.5, .5, 1.0)}.items():
        r = video_lens.inspect_visual_evidence(video, InspectionRequest(timestamp_sec=mid, window_before_sec=before,
                                                                         window_after_sec=after, fps=fps), extractor=ex)
        ts = [f.timestamp_sec for f in r.frames]
        out["windows"][label] = {"n": len(ts), "first": ts[0], "last": ts[-1], "steps": sorted({round(b - a, 3) for a, b in zip(ts, ts[1:])})}

    full = cv2.imread(ex.get_frame(video, mid).path)
    h, w = full.shape[:2]
    region = Region(0.5, 0.5, 1.0, 1.0)
    crop = cv2.imread(ex.get_frame(video, mid, region=region).path)
    out["crop"] = {"size": [crop.shape[1], crop.shape[0]], "expected": [w - w // 2, h - h // 2], "mad_vs_numpy_crop": round(_mad(crop, full[h // 2:, w // 2:]), 2)}
    odd = cv2.imread(ex.get_frame(video, mid, region=Region(0.123, 0.231, 0.777, 0.652)).path)
    out["crop_unaligned_region"] = {"size": [odd.shape[1], odd.shape[0]], "both_even": odd.shape[1] % 2 == 0 and odd.shape[0] % 2 == 0}
    small = cv2.imread(ex.get_frame(video, mid, scale_width=320).path)
    out["scale"] = {"size": [small.shape[1], small.shape[0]], "aspect_in": round(w / h, 4), "aspect_out": round(small.shape[1] / small.shape[0], 4)}

    def hashes(extractor):
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
        variants = {"full": {}, "crop": {"region": region}, "scale": {"scale_width": 320}, "both": {"region": region, "scale_width": 200}}
        return {k: digest(extractor.get_frame(video, mid, **kw).path) for k, kw in variants.items()}
    first, again = hashes(ex), hashes(FrameExtractor(cache_dir=os.path.join(tmp, "c2")))
    reversed_order = {k: first[k] for k in reversed(list(first))}
    out["cache"] = {"distinct_per_request": len(set(first.values())) == len(first), "identical_across_fresh_caches": first == again,
                    "stable_after_other_requests": hashes(ex) == first and reversed_order == {k: first[k] for k in reversed(list(first))}}
    with mock.patch.object(fe.subprocess, "run", wraps=subprocess.run) as spy:
        for kw in ({}, {"region": region}, {"scale_width": 320}):
            ex.get_frame(video, mid, **kw)
        out["cache"]["ffmpeg_calls_on_repeated_requests"] = spy.call_count
    return out


def _grid(path: str, tmp: str, step: float):
    video = ingest(path)
    return video, FrameExtractor(cache_dir=tmp).extract_window(video, 0.0, video.duration_sec - 1.0, step)  # clear of the end-of-video gap


def visual_change(path: str, tmp: str, label_th: list[tuple[float, float]]) -> dict:
    """P0-B on a uniform 0.5s grid, re-sampled to 0.5/1/2/4s, at the default threshold and two lower ones.
    `--label-th a:b,...` marks intervals you know contain NO real change (e.g. a talking head), to count
    detections inside them."""
    video, frames = _grid(path, tmp, 0.5)
    imgs = [cv2.imread(f.path, cv2.IMREAD_GRAYSCALE) for f in frames]
    mags = np.array([float((changed_pixel_mask(a, b) > 0).mean()) for a, b in zip(imgs, imgs[1:])])
    out = {"frames": len(frames), "pair_magnitude_percentiles_0.5s": {p: round(float(np.percentile(mags, p)), 4) for p in (5, 25, 50, 75, 90, 95, 99)},
           "sweep": {}}
    inside = lambda a, b: any(lo <= a and b <= hi for lo, hi in label_th)
    for stride in (1, 2, 4, 8):
        for thr in (0.08, 0.02, 0.005):
            events = detect_visual_changes(frames[::stride], threshold=thr)
            row = {"pairs": len(events), "detected": sum(e.status == "detected" for e in events),
                   "global": sum(e.kind == "visual_change_global" for e in events)}
            if label_th:
                known_quiet = [e for e in events if inside(*e.compared_timestamps)]
                row["in_labelled_quiet_intervals"] = {"pairs": len(known_quiet), "detected": sum(e.status == "detected" for e in known_quiet)}
            out["sweep"][f"{stride * 0.5}s@{thr}"] = row
    return out


def cursor(path: str, tmp: str) -> dict:
    """P0-C on a 0.5s grid: detector status counts, moving/uncertain segments, and a check that the
    output vocabulary never contains stillness, clicks, drags or intent."""
    video, frames = _grid(path, tmp, 0.5)
    events = detect_pointer_for_frames(frames)
    status = collections.Counter(e.status for e in events)
    track = PointerTrack(start_sec=frames[0].timestamp_sec, end_sec=frames[-1].timestamp_sec, events=events,
                         confidence=status["detected"] / len(events))
    segments = analyze_track(track)
    moving = [s for s in segments if s.motion_state == "moving"]
    return {"events": dict(status), "segments": len(segments), "moving": len(moving), "moving_seconds": round(sum(s.end_sec - s.start_sec for s in moving), 1),
            "uncertain_bases": dict(collections.Counter(s.basis for s in segments if s.motion_state == "uncertain")),
            "states": sorted({s.motion_state for s in segments}),
            "forbidden_vocabulary_present": any(w in repr(segments).lower() for w in ("stationary", "click", "drag", "intent")),
            "moving_segments": [{"t": [s.start_sec, s.end_sec], "direction_deg": s.direction_deg, "speed_diag_per_s": s.mean_speed_norm,
                                 "confidence": round(s.confidence, 2)} for s in moving],
            "note": "Verify detections visually: the detector reports a moving blob, not necessarily a cursor (docs/benchmark.md)."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("section", choices=["end-of-video", "inspection", "visual-change", "cursor"])
    parser.add_argument("video")
    parser.add_argument("--label-th", default="", help="visual-change: intervals known to hold no real change, e.g. 10:30,46:57")
    args = parser.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="vl_bench_")
    try:
        if args.section == "end-of-video":
            result = end_of_video(args.video, tmp)
        elif args.section == "inspection":
            result = inspection(args.video, tmp)
        elif args.section == "visual-change":
            spans = [tuple(map(float, s.split(":"))) for s in args.label_th.split(",") if s]
            result = visual_change(args.video, tmp, spans)
        else:
            result = cursor(args.video, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
