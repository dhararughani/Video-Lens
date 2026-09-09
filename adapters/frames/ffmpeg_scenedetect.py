"""Frame adapter: PySceneDetect picks timestamps, ffmpeg extracts the pixels.

PySceneDetect only detects scene boundaries; it does not need to decode/own
the whole extraction pipeline, so we let ffmpeg do the actual frame grab via
subprocess (same tool already used for metadata/audio elsewhere).
"""
from __future__ import annotations

import os
import subprocess

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from core.contracts import Frame, VideoInput


def detect_scene_timestamps(video: VideoInput, threshold: float = 27.0) -> list[float]:
    """Scene-change boundary timestamps only, no extraction. Shared by
    SceneDetectFrameAdapter and the frame-intelligence layer so scene
    detection logic (and its cost) isn't duplicated."""
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video=open_video(video.path), show_progress=False)
    scenes = sm.get_scene_list()
    return [s[0].seconds for s in scenes]


class SceneDetectFrameAdapter:
    def __init__(self, threshold: float = 27.0):
        self.threshold = threshold

    def extract_frames(self, video: VideoInput, out_dir: str) -> list[Frame]:
        os.makedirs(out_dir, exist_ok=True)

        # No scene changes (e.g. static screen recording) -> sample the start.
        timestamps = detect_scene_timestamps(video, self.threshold) or [0.0]

        frames = []
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video.path,
                 "-frames:v", "1", "-q:v", "2", "-pix_fmt", "yuvj420p", out_path],
                capture_output=True, check=True,
            )
            frames.append(Frame(
                timestamp_sec=ts, path=out_path, source_video=video.path,
                width=video.width, height=video.height,
                frame_index=round(ts * video.fps) if video.fps else None,
                reason="scene_change",
            ))
        return frames
