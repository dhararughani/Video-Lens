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
import json
import math
import os
import subprocess
import tempfile
from typing import NamedTuple

from core.contracts import Frame, Region, VideoInput
from core.errors import FrameExtractionError

_MS = 3  # canonical time precision: seconds, rounded to milliseconds
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "videolens_frame_cache")

# Part of every cache key. A cache entry's identity is (video, timestamp, WHAT
# was done to the pixels), and this versions the last part: bump it whenever the
# meaning of an existing key would change (e.g. a different crop/scale rule), so
# frames cached under the old rule are never served as if they followed the new
# one. "2" is the first version to key on scale/region at all; entries written
# before it carried no version, so they can never match and are simply orphaned
# in the (disposable) cache directory. "3": JPEGs are colour-matrix converted
# (JPEG_COLOR_FILTER) -- a BT.709 source's cached "2" frames have other pixels.
FRAME_EXTRACTION_VERSION = "3"

# The last step of every extraction. A JPEG (JFIF) is BT.601 full-range by
# definition -- every decoder reads it that way -- but ffmpeg only converts the
# *range* when writing yuvj420p, not the *matrix*: a BT.709 source's YUV was
# stored as-is and decoded back with BT.601 coefficients, shifting saturated
# colours (measured: MAD 9.6/255, up to 48 on a channel, docs/benchmark.md).
# This states the OUTPUT matrix only; the INPUT matrix stays swscale's default,
# read from each frame's own colour metadata, so a BT.709-tagged source is
# converted 709->601, a BT.601 one is untouched, and an untagged one is read
# exactly as ffmpeg's own RGB decode reads it (BT.601).
# The flags matter as much as the matrix: with swscale's default fast rounding
# the 709->601 conversion carries a systematic bias (blue -2, red -1 levels),
# which on the neutral benchmark video made the error WORSE than no conversion
# (MAD 1.5 vs 0.5-0.8 against an exact float decode of the raw YUV);
# accurate_rnd + full chroma interpolation removes it (0.4-0.6, and 1.0-1.2 vs
# 5.5-8.0 on the saturated one). `bicubic` is the scale filter's own default,
# stated because naming any flag replaces the default set.
JPEG_COLOR_FILTER = ("out_color_matrix=bt601:out_range=full"
                     ":flags=bicubic+accurate_rnd+full_chroma_int+full_chroma_inp")


def _floor_ms(t: float) -> float:
    """Round DOWN to millisecond precision -- unlike round(), never crosses
    past a boundary it was clamped to (see get_frame)."""
    return math.floor(t * 1000) / 1000


class _Transform(NamedTuple):
    """What extraction does to the pixels, fully resolved to integers so it can
    be both handed to ffmpeg and used as cache identity. `crop` is
    (x, y, w, h) in source pixels, `scale` is (w, h) of the final image; each is
    None when that step is a no-op. `size` is the output (w, h) either way."""
    crop: tuple[int, int, int, int] | None
    scale: tuple[int, int] | None
    size: tuple[int, int]

    def filter_chain(self) -> str:
        """One ffmpeg `-vf` chain: crop FIRST, then a single scale that resizes
        (when requested) and converts to the JPEG colour matrix -- always
        present, see JPEG_COLOR_FILTER."""
        parts = []
        if self.crop is not None:
            x, y, w, h = self.crop
            parts.append(f"crop={w}:{h}:{x}:{y}")
        size = f"{self.scale[0]}:{self.scale[1]}:" if self.scale is not None else ""
        parts.append(f"scale={size}{JPEG_COLOR_FILTER}")
        return ",".join(parts)

    def cache_token(self) -> str:
        crop = ",".join(map(str, self.crop)) if self.crop else "-"
        scale = ",".join(map(str, self.scale)) if self.scale else "-"
        return f"c{crop}:s{scale}"


def _even_span(lo: float, hi: float, size: int) -> tuple[int, int]:
    """(start, length) in pixels of the normalized span [lo, hi] along an axis
    of `size` pixels, rounded OUTWARD so a crop never cuts off part of the
    requested region, and aligned to even pixels.

    Even alignment is not cosmetic: ffmpeg silently rounds a crop on a
    chroma-subsampled format (yuv420p, the common case) down to even
    dimensions -- `crop=41:23` really produces 40x22 -- so an odd rectangle
    would make the dimensions reported on the Frame disagree with the pixels.
    Doing the alignment here keeps them identical by construction. `round(_, 6)`
    absorbs float noise like 0.7 * 10 == 7.000000000000001, which would
    otherwise ceil to 8."""
    start = math.floor(round(lo * size, 6)) & ~1
    end = min(size, math.ceil(round(hi * size, 6)))
    length = (end - start + 1) & ~1
    if start + length > size:  # only an odd `size` can land here
        length -= 2
    length = max(2, length)
    if start + length > size:
        start = (size - length) & ~1
    return start, length


def _plan_transform(src_w: int, src_h: int, scale_width: int | None,
                     region: Region | None) -> _Transform:
    """Resolve a (region, scale_width) request against the real source frame.
    Requests that change nothing resolve to the all-None default, so equivalent
    requests share one cache identity (a full-frame region, or a scale_width
    equal to the current width, is the default request)."""
    if scale_width is not None and (isinstance(scale_width, bool)
                                    or not isinstance(scale_width, int) or scale_width <= 0):
        raise ValueError(f"scale_width must be a positive integer, got {scale_width!r}")

    crop = None
    w, h = src_w, src_h
    if region is not None:
        if src_w < 2 or src_h < 2:
            raise ValueError(f"a {src_w}x{src_h} frame is too small to crop")
        x, cw = _even_span(region.x1, region.x2, src_w)
        y, ch = _even_span(region.y1, region.y2, src_h)
        if (x, y, cw, ch) != (0, 0, src_w, src_h):
            crop = (x, y, cw, ch)
            w, h = cw, ch

    scale = None
    if scale_width is not None and scale_width != w:
        # Explicit height computed from the (cropped) aspect ratio, rather than
        # ffmpeg's `-2`, so the Frame's reported size is exact by construction.
        # Half-up, not round(): banker's rounding would send 50.5 to 50.
        scale = (scale_width, max(1, int(h * scale_width / w + 0.5)))
        w, h = scale
    return _Transform(crop=crop, scale=scale, size=(w, h))


class _NoFrameProduced(FrameExtractionError):
    """ffmpeg exited cleanly but wrote nothing: the seek landed after the last
    decodable frame. Private -- callers still just see FrameExtractionError."""


def _last_frame_position(video_path: str, near_sec: float) -> float | None:
    """The `-ss` position of the video stream's LAST frame, from the stream's
    own packet timestamps -- not the container duration, which also covers
    audio (and so can run past the last frame, by any amount).

    Reads packet headers only (no decoding), from the keyframe before
    `near_sec` to the end: ~0.1s on the 4-minute benchmark videos. ffmpeg's
    `-ss` is relative to the container's start_time, so that is subtracted.
    None if the stream has no packet timestamps to read."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{near_sec}%",
             "-show_entries", "packet=pts_time:format=start_time", "-of", "json", video_path],
            capture_output=True, check=True, text=True,
        )
        data = json.loads(r.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    pts = [float(p["pts_time"]) for p in data.get("packets", []) if p.get("pts_time") not in (None, "N/A")]
    if not pts:
        return None
    start = data.get("format", {}).get("start_time")
    offset = float(start) if start not in (None, "N/A") else 0.0
    # floor, never round: a position a hair BEFORE the frame's pts still yields
    # that frame, a hair after yields nothing -- the very bug this exists for.
    return max(0.0, _floor_ms(max(pts) - offset))


class FrameExtractor:
    """`get_frame`/`extract_window` are the only two operations: everything
    downstream (speech bridge, keyframe selection) is built from these."""

    def __init__(self, cache_dir: str | None = None):
        # ponytail: no eviction -- the cache dir grows unbounded. Treat it as
        # disposable (a temp dir by default) and add size/LRU cleanup if a
        # long-running process ever makes that a real problem.
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        # video identity -> its real last-frame position, learned the first
        # time a request lands past it (see get_frame); later requests clamp
        # straight to it without a failed ffmpeg run.
        self._stream_end: dict[tuple, float] = {}

    def get_frame(self, video: VideoInput, timestamp_sec: float, *,
                   scale_width: int | None = None, region: Region | None = None) -> Frame:
        """One frame at `timestamp_sec`. `region` (normalized crop) and
        `scale_width` (target width, aspect ratio preserved) are optional and
        applied in that order in a single ffmpeg pass; with neither, behavior
        is exactly what it was before they existed. The returned Frame's
        width/height are those of the OUTPUT image."""
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
        # That clamp trusts the CONTAINER duration, which covers every stream:
        # when the video stream ends first (audio outlasts it -- by 0.35 and
        # 1.6 frames on the benchmark videos, by seconds in some recordings)
        # it is still past the last frame. The real end, once learned, wins.
        video_key = self._video_key(video)
        if video_key in self._stream_end:
            ts = min(ts, self._stream_end[video_key])

        plan = _plan_transform(video.width, video.height, scale_width, region)
        out_path = self._cache_path(video, ts, plan)
        # size check, not just existence -- a prior extraction killed
        # mid-write would otherwise leave a 0-byte "cached" file that's
        # trusted forever instead of being regenerated.
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            try:
                self._extract(video.path, ts, out_path, plan.filter_chain())
            except _NoFrameProduced:
                # Only now is the stream probed -- never for a request that
                # simply succeeds -- and once per video per extractor. A request
                # past the end gets the LAST frame, stamped with that frame's own
                # position (the clamp's rule above, against the real end). An
                # "end" not before ts means the failure is something else: raise.
                end = _last_frame_position(video.path, ts)
                if end is None or end >= ts:
                    raise
                self._stream_end[video_key] = end
                ts = end
                out_path = self._cache_path(video, ts, plan)
                if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                    self._extract(video.path, ts, out_path, plan.filter_chain())

        return Frame(
            timestamp_sec=ts, path=out_path, source_video=video.path,
            width=plan.size[0], height=plan.size[1],
            frame_index=round(ts * video.fps) if video.fps else None,
            reason="direct_timestamp",
        )

    def extract_window(
        self, video: VideoInput, start_sec: float, end_sec: float, interval_sec: float,
        *, scale_width: int | None = None, region: Region | None = None,
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
            f = self.get_frame(video, ts, scale_width=scale_width, region=region)
            if f.timestamp_sec in seen:  # two requested timestamps both clamp to the same instant
                continue
            seen.add(f.timestamp_sec)
            frames.append(Frame(timestamp_sec=f.timestamp_sec, path=f.path,
                                 source_video=f.source_video, width=f.width, height=f.height,
                                 frame_index=f.frame_index, reason="window_sample"))
        return frames

    @staticmethod
    def _video_key(video: VideoInput) -> tuple:
        st = os.stat(video.path)
        return os.path.abspath(video.path), st.st_mtime_ns, st.st_size

    def _cache_path(self, video: VideoInput, ts: float, plan: _Transform | None = None) -> str:
        # mtime+size in the key means a modified/replaced video file
        # produces a new key automatically -- no manual invalidation needed.
        # The version and the resolved crop/scale are in the key too: a cached
        # full-resolution frame must never be returned for a request that asks
        # for different pixels. Keying on the RESOLVED integers (not the raw
        # request) means two requests that produce the same image share an
        # entry, and two that don't can never collide.
        plan = plan or _plan_transform(video.width, video.height, None, None)
        st = os.stat(video.path)
        digest = hashlib.sha1(
            (f"{os.path.abspath(video.path)}:{st.st_mtime_ns}:{st.st_size}:{ts:.3f}"
             f":v{FRAME_EXTRACTION_VERSION}:{plan.cache_token()}").encode()
        ).hexdigest()[:20]
        return os.path.join(self.cache_dir, f"{digest}.jpg")

    @staticmethod
    def _extract(video_path: str, ts: float, out_path: str, vf: str = f"scale={JPEG_COLOR_FILTER}") -> None:
        try:
            subprocess.run(
                # -pix_fmt yuvj420p: some sources (e.g. lavfi-generated
                # yuv420p test video) hit "ff_frame_thread_encoder_init
                # failed" in the mjpeg encoder without it -- see docs/frames.md.
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", "-vf", vf, "-pix_fmt", "yuvj420p", out_path],
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
            raise _NoFrameProduced(f"ffmpeg produced no frame at {ts}s from '{video_path}'")
