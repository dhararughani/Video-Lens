"""Ingestion-time failures. Raised with an actionable message -- never
swallowed -- so callers see exactly what was wrong with the input."""


class VideoIngestionError(Exception):
    pass


class TranscriptionError(Exception):
    """The transcription engine failed. A video that simply has no audio, or
    audio with no speech in it, is NOT this -- that's a Transcript.status."""
    pass


class FrameExtractionError(Exception):
    """ffmpeg failed to produce a requested frame (missing/corrupt video,
    ffmpeg not on PATH). An out-of-range timestamp is NOT this -- it's
    clamped to the video's duration instead."""
    pass


class KnowledgePackageError(Exception):
    """A KnowledgePackage failed validation before handoff (Step 9 lifecycle,
    see docs/lifecycle.md). Raising this -- rather than writing a broken
    package -- is what keeps the cleanup safety gate closed: process_video
    never calls JobWorkspace.mark_success() if this is raised."""
    pass
