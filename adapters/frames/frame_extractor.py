"""Direct timestamp-based frame extraction, with a filesystem cache.

Uses ffmpeg input seeking (`-ss` before `-i`). Measured on this machine
(docs/frames.md): ~4x faster than output seeking (0.17s vs 0.74s to reach
55s into a 60s 640x480 video) and, on a video deliberately encoded with a
single GOP to expose fast-seek inaccuracy, produced identical pixel-accuracy
to output seeking. Exact frame-level seeking is still not a hard guarantee
across every container/codec, so `Frame.frame_index` is always a
best-effort estimate, never authoritative.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile

from core.contracts import Frame, VideoInput
from core.errors import FrameExtractionError

_MS = 3  # canonical time precision: seconds, rounded to milliseconds
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "videolens_frame_cache")


def _floor_ms(t: float) -> float:
    """Round DOWN to millisecond precision -- unlike round(), never crosses
    past a boundary it was clamped to (see get_frame)."""
    return math.floor(t * 1000) / 1000


class FrameExtractor:
    """`get_frame`/`extract_window` are the only two operations: everything
    downstream (speech bridge, keyframe selection) is built from these."""

    def __init__(self, cache_dir: str | None = None):
        # ponytail: no eviction -- the cache dir grows unbounded. Treat it as
        # disposable (a temp dir by default) and add size/LRU cleanup if a
        # long-running process ever makes that a real problem.
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_frame(self, video: VideoInput, timestamp_sec: float) -> Frame:
        if not math.isfinite(timestamp_sec) or timestamp_sec < 0:
            raise ValueError(f"timestamp_sec must be a finite number >= 0, got {timestamp_sec}")
        if not os.path.exists(video.path):
            raise FrameExtractionError(f"video file does not exist: {video.path}")
        if video.duration_sec <= 0:
            raise FrameExtractionError(f"video has no usable duration: {video.path}")

        # A caller asking for a timestamp past the end (e.g. a segment's
        # end_sec that lands exactly on duration_sec) is a normal rounding
        # case, not an error -- clamp to the last actual frame instead of
        # failing. The last *frame* sits at ~duration - 1/fps, not at
        # duration itself: a video has no frame between its last one and its
        # reported duration, and ffmpeg errors ("produced no frame") if asked
        # to seek into that gap -- measured on a 30fps 3.0s video, where the
        # true last frame sits at 2.96667s. Round DOWN (never up) so the
        # clamp can't land a hair past that last frame from float rounding.
        frame_interval = (1.0 / video.fps) if video.fps else 0.001
        last_frame_time = max(0.0, video.duration_sec - frame_interval)
        ts = _floor_ms(min(timestamp_sec, last_frame_time))

        out_path = self._cache_path(video, ts)
        # size check, not just existence -- a prior extraction killed
        # mid-write would otherwise leave a 0-byte "cached" file that's
        # trusted forever instead of being regenerated.
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            self._extract(video.path, ts, out_path)

        return Frame(
            timestamp_sec=ts, path=out_path, source_video=video.path,
            width=video.width, height=video.height,
            frame_index=round(ts * video.fps) if video.fps else None,
            reason="direct_timestamp",
        )

    def extract_window(
        self, video: VideoInput, start_sec: float, end_sec: float, interval_sec: float,
    ) -> list[Frame]:
        if end_sec <= start_sec:
            raise ValueError(f"end_sec ({end_sec}) must be > start_sec ({start_sec})")
        if interval_sec <= 0:
            raise ValueError(f"interval_sec must be > 0, got {interval_sec}")

        n_steps = int((end_sec - start_sec) / interval_sec + 1e-9)
        timestamps = [start_sec + i * interval_sec for i in range(n_steps + 1)]

        frames = []
        seen: set[float] = set()
        for ts in timestamps:
            f = self.get_frame(video, ts)
            if f.timestamp_sec in seen:  # two requested timestamps both clamp to the same instant
                continue
            seen.add(f.timestamp_sec)
            frames.append(Frame(timestamp_sec=f.timestamp_sec, path=f.path,
                                 source_video=f.source_video, width=f.width, height=f.height,
                                 frame_index=f.frame_index, reason="window_sample"))
        return frames

    def _cache_path(self, video: VideoInput, ts: float) -> str:
        # mtime+size in the key means a modified/replaced video file
        # produces a new key automatically -- no manual invalidation needed.
        st = os.stat(video.path)
        digest = hashlib.sha1(
            f"{os.path.abspath(video.path)}:{st.st_mtime_ns}:{st.st_size}:{ts:.3f}".encode()
        ).hexdigest()[:20]
        return os.path.join(self.cache_dir, f"{digest}.jpg")

    @staticmethod
    def _extract(video_path: str, ts: float, out_path: str) -> None:
        try:
            subprocess.run(
                # -pix_fmt yuvj420p: some sources (e.g. lavfi-generated
                # yuv420p test video) hit "ff_frame_thread_encoder_init
                # failed" in the mjpeg encoder without it -- see docs/frames.md.
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", "-pix_fmt", "yuvj420p", out_path],
                capture_output=True, check=True,
            )
        except FileNotFoundError as e:
            raise FrameExtractionError("ffmpeg is not installed or not on PATH") from e
        except subprocess.CalledProcessError as e:
            raise FrameExtractionError(
                f"ffmpeg could not extract a frame at {ts}s from '{video_path}': "
                f"{e.stderr.decode(errors='replace').strip()}"
            ) from e
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise FrameExtractionError(f"ffmpeg produced no frame at {ts}s from '{video_path}'")
