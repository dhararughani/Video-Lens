"""Speech -> frame bridge, scene-aware keyframe selection, and lightweight
near-duplicate reduction. All selection here is deterministic temporal
logic -- no LLM/VLM decides which frame matters (that's a later step).
"""
from __future__ import annotations

import subprocess
from dataclasses import replace

from adapters.frames.ffmpeg_scenedetect import detect_scene_timestamps
from adapters.frames.frame_extractor import FrameExtractor
from core.contracts import Frame, TranscriptSegment, VideoInput


def frames_for_segment(
    video: VideoInput,
    segment: TranscriptSegment,
    extractor: FrameExtractor,
    interval_sec: float = 1.0,
    scene_timestamps: list[float] | None = None,
) -> list[Frame]:
    """Candidate frames for one speech segment: start, middle, end, uniform
    samples across the segment, plus any scene change that falls inside it
    (a transition during narration like "look at this" is often the single
    most useful frame). Pass `scene_timestamps` in when calling this for many
    segments of the same video, to detect scenes once instead of per call.
    """
    if segment.end_sec <= segment.start_sec:
        raise ValueError("segment has no duration")
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be > 0, got {interval_sec}")

    candidates = {segment.start_sec, (segment.start_sec + segment.end_sec) / 2, segment.end_sec}
    n_steps = int((segment.end_sec - segment.start_sec) / interval_sec + 1e-9)
    candidates.update(segment.start_sec + i * interval_sec for i in range(n_steps + 1))

    if scene_timestamps is None:
        scene_timestamps = detect_scene_timestamps(video)
    scene_in_segment = {sc for sc in scene_timestamps if segment.start_sec <= sc <= segment.end_sec}
    candidates.update(scene_in_segment)
    scene_set = {round(s, 3) for s in scene_in_segment}

    frames = []
    for ts in sorted(candidates):
        f = extractor.get_frame(video, ts)
        reason = "scene_change" if round(f.timestamp_sec, 3) in scene_set else "window_sample"
        frames.append(replace(f, reason=reason))
    return frames


def select_keyframes(
    video: VideoInput,
    extractor: FrameExtractor,
    interval_sec: float = 2.0,
    diff_threshold: float = 10.0,
) -> list[Frame]:
    """Whole-video keyframe candidates: scene boundaries + uniform sampling,
    deduplicated. Reduces "every frame" down to a manageable set that still
    covers both narrative structure (scenes) and steady coverage (interval).
    """
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be > 0, got {interval_sec}")

    scenes = detect_scene_timestamps(video)
    n_steps = int(video.duration_sec / interval_sec)
    uniform = [i * interval_sec for i in range(n_steps + 1)]
    all_ts = sorted(set(scenes) | set(uniform) | {0.0})

    frames = [extractor.get_frame(video, ts) for ts in all_ts]
    return dedupe_frames(frames, diff_threshold=diff_threshold)


def _thumbnail(frame_path: str) -> bytes:
    """8x8 grayscale thumbnail via ffmpeg -- no extra CV dependency, just the
    ffmpeg already on PATH piping raw bytes to stdout.

    An average-hash (threshold-against-the-image's-own-mean, then compare
    Hamming distance) was tried first and rejected: on a perfectly flat
    frame every pixel equals the mean, so red-red-green-red measured
    IDENTICAL all-zero hashes for every color -- solid/near-solid frames
    (slides, simple UI) are common exactly in this project's target domain,
    so that failure mode isn't a corner case. Returning raw thumbnail bytes
    and diffing them directly (below) does not have this blind spot.
    """
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", frame_path, "-vf", "scale=8:8,format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    return r.stdout[:64]


def dedupe_frames(frames: list[Frame], diff_threshold: float = 10.0) -> list[Frame]:
    """Drop frames that are near-duplicates of the most recently kept frame:
    mean absolute difference of their 8x8 grayscale thumbnails <=
    `diff_threshold` (0-255 scale). Screen recordings often hold a static
    screen for seconds; this keeps the first frame of each visually-stable
    stretch instead of every sample. Measured in docs/frames.md: threshold=10
    separates a genuine solid-color change from re-encode noise on an
    identical frame (measured MAD ~0-2 for a re-decoded identical frame)."""
    if not frames:
        return []
    kept = [frames[0]]
    kept_thumb = _thumbnail(frames[0].path)
    for f in frames[1:]:
        thumb = _thumbnail(f.path)
        mad = sum(abs(a - b) for a, b in zip(thumb, kept_thumb)) / len(thumb)
        if mad > diff_threshold:
            kept.append(f)
            kept_thumb = thumb
    return kept
