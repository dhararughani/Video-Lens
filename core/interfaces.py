"""Adapter contracts. Each adapter type wraps exactly one external
capability. Core code depends only on these Protocols, never on a concrete
library (Whisper, ffmpeg, PySceneDetect, a VLM, ...).
"""
from __future__ import annotations

from typing import Protocol

from core.contracts import (
    Frame, PointerEvent, PointerTrack, SynthesisBrief, Transcript, VideoInput, VisionObservation,
)


class IngestionAdapter(Protocol):
    def resolve(self, source: str) -> VideoInput: ...


class TranscriptionAdapter(Protocol):
    def transcribe(self, video: VideoInput) -> Transcript: ...


class FrameAdapter(Protocol):
    def extract_frames(self, video: VideoInput, out_dir: str) -> list[Frame]: ...


# adapters/pointer/cursor_detector.py implements this shape as module-level
# functions of the same names (detect_pointer/detect_pointer_for_frames), not
# a class instance -- pointer detection has no per-call state (no model to
# load, no client to hold open) unlike TranscriptionAdapter/VisionAdapter, so
# there was nothing for a class to usefully own. Kept as a Protocol here for
# documentation/consistency with every other adapter shape.
class PointerAdapter(Protocol):
    def detect_pointer(self, frame: Frame, prev_frame: Frame | None = None) -> PointerEvent: ...
    def detect_pointer_for_frames(self, frames: list[Frame]) -> list[PointerEvent]: ...


class VisionAdapter(Protocol):
    def analyze_frame(self, frame: Frame, transcript_context: str | None = None,
                       pointer: PointerEvent | None = None) -> VisionObservation: ...


class KnowledgeSynthesizer(Protocol):
    """Step 11's semantic seam: turns a `SynthesisBrief` (already-correlated
    evidence, plus a rendered prompt pair) into the raw structured response a
    model produced -- a plain dict, exactly as parsed from the provider's
    JSON.

    Returning raw provider output rather than finished `Claim` objects is
    deliberate: Video-Lens does not trust this dict. `core.synthesis` parses,
    validates and CROSS-CHECKS it against the real evidence before anything
    reaches a KnowledgePackage, so a provider cannot inject an unverified
    claim by constructing a well-typed object. It also keeps a provider
    implementation trivially thin -- send the brief's prompts, parse JSON,
    return it -- which is what makes "supply your own model" real rather
    than aspirational.

    Video-Lens ships NO implementation of this Protocol and requires none:
    with no synthesizer supplied, `process_video()` degrades honestly to the
    deterministic package (see docs/synthesis.md, "Fallback behavior").
    An implementation may be backed by a local model (e.g. an Ollama install
    the caller already runs), a hosted API the caller already pays for, or a
    deterministic fake in tests -- Video-Lens neither knows nor cares.

    Raising from `synthesize` is a supported outcome, not a bug: it is
    recorded as `SynthesisMetadata.status == "failed"` and never fails the
    surrounding job.
    """
    def synthesize(self, brief: SynthesisBrief) -> dict: ...
