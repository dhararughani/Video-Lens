"""Builds a VideoInput from a local file via ffprobe. Not an adapter -- every
adapter needs this metadata, so it lives in core rather than being duplicated
per adapter.
"""
from __future__ import annotations

import json
import os
import subprocess

from core.contracts import VideoInput
from core.errors import VideoIngestionError


def inspect_video(path: str, source_type: str = "local", source: str | None = None,
                   title: str | None = None) -> VideoInput:
    """Validate a local media file and build its VideoInput. Never loads the
    video into memory -- ffprobe reads the container/stream headers only.

    `title` lets a caller (e.g. URLIngestionAdapter, which has yt-dlp's own
    reported title) override the container's own metadata tag; when omitted,
    the ffprobe `format.tags.title` is used if present, else None -- never
    fabricated from the filename."""
    if not os.path.exists(path):
        raise VideoIngestionError(f"Video file does not exist: {path}")
    if not os.path.isfile(path):
        raise VideoIngestionError(f"Video path is not a file: {path}")
    if os.path.getsize(path) == 0:
        raise VideoIngestionError(f"Video file is empty: {path}")

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, check=True, text=True,
        )
    except FileNotFoundError:
        raise VideoIngestionError("ffprobe is not installed or not on PATH")
    except subprocess.CalledProcessError as e:
        raise VideoIngestionError(
            f"ffprobe could not read '{path}' -- file may be corrupt or an "
            f"unsupported/unreadable format: {e.stderr.strip()}"
        ) from e

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VideoIngestionError(f"ffprobe returned unparseable output for '{path}'") from e

    video_stream = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
    if video_stream is None:
        raise VideoIngestionError(f"'{path}' has no video stream -- not a usable video file")
    audio_streams = [s for s in data["streams"] if s["codec_type"] == "audio"]

    num, den = video_stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0

    duration_raw = data.get("format", {}).get("duration") or video_stream.get("duration")
    if duration_raw is None:
        raise VideoIngestionError(f"'{path}' has no readable duration")
    duration_sec = float(duration_raw)
    if duration_sec <= 0:
        raise VideoIngestionError(f"'{path}' has invalid duration: {duration_sec}s")

    width, height = video_stream.get("width", 0), video_stream.get("height", 0)
    if width <= 0 or height <= 0:
        raise VideoIngestionError(f"'{path}' has invalid resolution: {width}x{height}")

    return VideoInput(
        path=os.path.abspath(path),
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        has_audio=len(audio_streams) > 0,
        source_type=source_type,
        source=source if source is not None else os.path.abspath(path),
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_streams[0].get("codec_name") if audio_streams else None,
        format_name=data.get("format", {}).get("format_name"),
        title=title or data.get("format", {}).get("tags", {}).get("title"),
    )
