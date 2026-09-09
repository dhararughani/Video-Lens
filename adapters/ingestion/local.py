"""Local-file ingestion. Never copies or loads the video -- ffprobe only
reads container/stream headers, and the resolved VideoInput.path points
straight at the user's own file."""
from __future__ import annotations

import os

from core.contracts import VideoInput
from core.errors import VideoIngestionError
from core.video_metadata import inspect_video


class LocalIngestionAdapter:
    def resolve(self, source: str) -> VideoInput:
        if not source or not source.strip():
            raise VideoIngestionError("Local video path is empty")
        normalized = os.path.abspath(os.path.expanduser(source))
        return inspect_video(normalized, source_type="local", source=normalized)
