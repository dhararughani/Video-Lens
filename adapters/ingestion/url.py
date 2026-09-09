"""URL ingestion via yt-dlp's own Python API (not the CLI/shell) -- no
shell-injection surface, and the rest of Video-Lens never needs to know
yt-dlp is involved, only that it gets a VideoInput back.

Downloads live under a dedicated cache directory, not the user's own files,
and are never deleted automatically -- see docs/ingestion.md.
"""
from __future__ import annotations

import os
import tempfile

import yt_dlp

from core.contracts import VideoInput
from core.errors import VideoIngestionError
from core.video_metadata import inspect_video

DEFAULT_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "video_lens_downloads")


class URLIngestionAdapter:
    """Resolves a URL to a VideoInput, downloading through yt-dlp into
    `download_dir`. Reuses an already-downloaded file for the same URL
    instead of re-downloading it."""

    def __init__(self, download_dir: str = DEFAULT_DOWNLOAD_DIR):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def _ydl_opts(self) -> dict:
        return {
            # yt-dlp's own default selector (best video+audio, merged) rather
            # than a narrower ext filter -- some videos don't expose a single
            # progressive mp4 stream, and merging needs ffmpeg, which we
            # already require.
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": os.path.join(self.download_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "restrictfilenames": True,  # safe, predictable Windows-friendly filenames
            "quiet": True,
            "no_warnings": True,
        }

    def resolve(self, source: str) -> VideoInput:
        if not source or not source.strip():
            raise VideoIngestionError("URL is empty")

        with yt_dlp.YoutubeDL(self._ydl_opts()) as ydl:
            try:
                info = ydl.extract_info(source, download=False)
            except yt_dlp.utils.DownloadError as e:
                raise VideoIngestionError(f"URL is invalid or unsupported: '{source}': {e}") from e

            # Merged downloads end up as "<id>.mp4" regardless of the source
            # formats' own extensions (merge_output_format), so guess by id
            # rather than trusting prepare_filename()'s pre-download extension.
            expected_path = os.path.join(self.download_dir, f"{info['id']}.mp4")
            if os.path.exists(expected_path) and os.path.getsize(expected_path) > 0:
                downloaded_path = expected_path  # already downloaded -- reuse, don't re-fetch
            else:
                try:
                    info = ydl.extract_info(source, download=True)
                except yt_dlp.utils.DownloadError as e:
                    raise VideoIngestionError(f"Failed to download '{source}': {e}") from e
                downloads = info.get("requested_downloads") or []
                downloaded_path = downloads[0]["filepath"] if downloads else expected_path

        if not os.path.exists(downloaded_path):
            raise VideoIngestionError(
                f"yt-dlp reported success but output file is missing: {downloaded_path}"
            )

        return inspect_video(downloaded_path, source_type="url", source=source, title=info.get("title"))
