"""Single entry point for ingestion: callers pass a local path or a URL and
get back a normalized VideoInput -- they never touch yt-dlp or the local
adapter directly, and downstream code (frame extraction, transcription)
doesn't care which one ran.
"""
from __future__ import annotations

from urllib.parse import urlparse

from adapters.ingestion.local import LocalIngestionAdapter
from adapters.ingestion.url import URLIngestionAdapter
from core.contracts import VideoInput


def is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def ingest(source: str, download_dir: str | None = None) -> VideoInput:
    if is_url(source):
        adapter = URLIngestionAdapter(download_dir) if download_dir else URLIngestionAdapter()
        return adapter.resolve(source)
    return LocalIngestionAdapter().resolve(source)
