"""Video-Lens public API.

A downstream project should only ever need:

    from video_lens import analyze_video, PipelineConfig
    result = analyze_video("path/to/video.mp4")        # or a URL
    for so in result.structured_observations:
        ...

Everything under `core/`/`adapters/` is implementation detail this module
composes -- import from those packages directly only for advanced use
(e.g. calling one adapter in isolation, as the `scripts/*_smoke_test.py`
files do for focused testing).

Pipeline: ingestion -> transcript -> keyframe selection -> pointer evidence
-> vision analysis -> evidence correlation -> AnalysisResult. Ingestion is
the one required stage (no video, nothing else is possible); every stage
after it is optional and independently degradable -- a disabled or failed
optional stage is recorded as missing evidence (`unavailable`, see
docs/evidence.md), never fabricated, and never prevents the other stages
from running (see docs/architecture.md, "Partial pipeline").

Video-Lens is not built on any one AI provider. Vision analysis is behind
`core.interfaces.VisionAdapter` -- the built-in Claude-backed adapter
(`adapters/vision/claude_vision.py`) is the default implementation, used
only when `PipelineConfig.vision_provider` is left unset, and this module
never imports it except lazily, right where it's constructed. Pass your own
`vision_provider` (or `vision_enabled=False`) to run Video-Lens without
Claude/Anthropic at all -- every other stage (ingestion, transcript,
frames, pointer, evidence correlation, knowledge extraction) has no AI
provider dependency whatsoever. See docs/vision.md, "Provider model".
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import select_keyframes
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import detect_pointer_at
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
from core.contracts import AnalysisResult, Frame, KnowledgePackage, PointerEvent, Transcript, VisionObservation
from core.errors import FrameExtractionError, TranscriptionError
from core.evidence import build_analysis_result
from core.interfaces import KnowledgeSynthesizer, VisionAdapter
from core.knowledge import (
    build_knowledge_package, default_filename_stem, export_knowledge_package,
    knowledge_source_for, validate_knowledge_package,
)
from core.synthesis import build_synthesis_brief, synthesize, unavailable
from core.visual_evidence import select_and_bundle
from core.workspace import JobWorkspace

# The Claude-backed vision provider is imported lazily, inside
# `_analyze_vision`, only when it's actually needed (vision_enabled and no
# `vision_provider` supplied) -- this file, the canonical pipeline, never
# hard-depends on Claude/Anthropic. See docs/vision.md "Provider model".


@dataclass
class PipelineConfig:
    """Every field has a default that works with zero configuration --
    CPU-only, no API key required (vision degrades to `status="unavailable"`
    without one rather than failing the pipeline). Only the knobs a caller
    is actually likely to want to change live here; low-level CV tuning
    constants (e.g. the pointer detector's contour-size/solidity
    thresholds) stay internal to their module -- see docs/architecture.md's
    Configuration section for why that line was drawn where it was.
    """
    # transcription (adapters/transcription/faster_whisper_adapter.py)
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # frame selection (adapters/frames/keyframe_selector.py)
    keyframe_interval_sec: float = 2.0
    keyframe_diff_threshold: float = 10.0
    max_frames: int = 20  # ceiling on frames sent through pointer/vision -- keeps cost bounded on long videos

    # optional stages
    pointer_enabled: bool = True
    # keyframes are typically seconds apart (scene changes / interval
    # sampling) -- too far apart for three-frame differencing to see any
    # motion between them directly. detect_pointer_at extracts two EXTRA,
    # closely-spaced frames around each keyframe timestamp instead (see
    # adapters/pointer/cursor_detector.py); this is how far apart.
    pointer_step_sec: float = 0.2
    vision_enabled: bool = True
    # None -> the default provider's own default model (currently Claude's,
    # see adapters/vision/claude_vision.py's DEFAULT_MODEL) -- this file
    # deliberately does not hardcode or duplicate a provider-specific model
    # name. Ignored entirely when `vision_provider` is supplied.
    vision_model: str | None = None
    # Inject any object implementing core.interfaces.VisionAdapter
    # (`analyze_frame(frame, transcript_context, pointer) -> VisionObservation`)
    # to use a different vision backend -- OpenAI, Gemini, a local VLM, a
    # test fake, whatever. None (default) -> the built-in Claude provider,
    # itself just one more implementation of this same interface. See
    # docs/vision.md "Provider model".
    vision_provider: VisionAdapter | None = None

    # evidence correlation (core/evidence.py)
    tolerance_sec: float = 1.0

    # cache/download locations -- None means "each component's own default
    # temp-dir location" (see docs/frames.md, docs/vision.md, docs/ingestion.md)
    frame_cache_dir: str | None = None
    vision_cache_dir: str | None = None
    download_dir: str | None = None

    # Step 9 lifecycle (process_video only -- see docs/lifecycle.md)
    output_dir: str | None = None  # None -> ./videolens_output
    retain_temp_artifacts: bool = False  # True -> skip cleanup even on success (debugging)

    # Step 11 semantic synthesis (process_video only -- see docs/synthesis.md).
    # None (default) -> no semantic synthesis; the package is deterministic and
    # says so in `limitations`. Inject any object implementing
    # core.interfaces.KnowledgeSynthesizer (`synthesize(brief) -> dict`) to add
    # semantic understanding -- a local model, a hosted API you already use, or
    # a test fake. Video-Lens ships no implementation and requires none.
    knowledge_synthesizer: KnowledgeSynthesizer | None = None
    visual_evidence_enabled: bool = True  # copy selected evidence frames beside the package
    # Safety ceiling only. Selection is driven by which frames retained
    # knowledge actually references, then de-duplicated -- None means "let the
    # content decide", not "unbounded". See core/visual_evidence.py.
    max_visual_evidence: int | None = None


def analyze_video(source: str, config: PipelineConfig | None = None) -> AnalysisResult:
    """Run the canonical Video-Lens pipeline on a local file path or a URL.

    Raises only for the one stage that can't degrade -- ingestion itself
    (an unreadable/missing/corrupt video, per `core.errors.VideoIngestionError`).
    Every later stage that fails or is disabled prints one warning line to
    stderr and continues with that evidence stream simply absent -- see the
    module docstring and docs/architecture.md.
    """
    result, _transcript = _run_pipeline(source, config or PipelineConfig())
    return result


def _run_pipeline(source: str, config: PipelineConfig) -> tuple[AnalysisResult, Transcript | None]:
    """The actual pipeline body, shared by `analyze_video` (result only) and
    `process_video` (needs the raw Transcript too, to extract knowledge from
    -- see docs/lifecycle.md). Not part of the public API; call
    `analyze_video`/`process_video` instead."""
    video = ingest(source, download_dir=config.download_dir)

    transcript = _try_transcribe(video, config)
    frames, extractor = _try_select_frames(video, config)

    pointer_events: list[PointerEvent] = []
    if config.pointer_enabled and frames:
        pointer_events = _detect_pointer_evidence(video, frames, extractor, config)

    vision_observations: list[VisionObservation] = []
    if config.vision_enabled and frames:
        vision_observations = _analyze_vision(frames, transcript, pointer_events, config)

    query_timestamps = [f.timestamp_sec for f in frames]
    if not query_timestamps and transcript is not None:
        query_timestamps = [s.start_sec for s in transcript.segments]

    result = build_analysis_result(
        video, query_timestamps, transcript=transcript, frames=frames,
        pointer_events=pointer_events, vision_observations=vision_observations,
        tolerance_sec=config.tolerance_sec,
    )
    return result, transcript


def process_video(source: str, config: PipelineConfig | None = None) -> KnowledgePackage:
    """The Step 9 lifecycle: run the pipeline in a job-scoped temporary
    workspace, extract a compact `KnowledgePackage`, validate and write it
    to durable storage, and ONLY THEN delete the temporary artifacts this
    job owns (job-downloaded video, extracted frames, vision cache) --
    never the caller's own local source file, which always lives outside
    the job workspace. If anything fails before the package is written,
    nothing is cleaned up -- see docs/lifecycle.md.

    Use this for standalone "give me a video, get me knowledge" usage.
    Use `analyze_video` directly if you want the full evidence-level
    `AnalysisResult` and want to manage storage/caching yourself.
    """
    config = config or PipelineConfig()
    workspace = JobWorkspace()
    job_config = replace(config, frame_cache_dir=workspace.frame_cache_dir,
                          vision_cache_dir=workspace.vision_cache_dir,
                          download_dir=workspace.download_dir)

    result, transcript = _run_pipeline(source, job_config)

    # The brief is built either way: with a synthesizer it's what the provider
    # sees, and without one it still supplies the id'd frame candidates that
    # visual-evidence selection draws from (see docs/synthesis.md).
    brief = build_synthesis_brief(result, transcript, knowledge_source_for(result.video))
    synthesis = (synthesize(config.knowledge_synthesizer, brief)
                 if config.knowledge_synthesizer is not None
                 else unavailable("no knowledge_synthesizer was supplied"))

    package = build_knowledge_package(result, transcript=transcript, synthesis=synthesis)
    _validate_package(package)  # explicit step, kept separate from export so
    # a failure here reads clearly as "the package itself was bad" in a
    # traceback, distinct from an export/write failure (see docs/lifecycle.md)

    output_dir = config.output_dir or os.path.join(os.getcwd(), "videolens_output")
    stem = default_filename_stem(package)  # bundle and JSON must share it

    if config.visual_evidence_enabled:
        # MUST happen before cleanup: the selected frames still live in the job
        # workspace, and cleanup() deletes that whole tree. Copying evidence out
        # first is what lets the workspace be discarded without destroying the
        # evidence a consumer still needs (see docs/synthesis.md).
        visual, claims = select_and_bundle(
            candidates={i.evidence_id: i.evidence for i in brief.items
                        if i.evidence.kind == "frame"},
            claims=package.claims, key_points=package.key_lessons + package.important_observations,
            output_dir=output_dir, stem=stem,
            vision_descriptions=_vision_descriptions(result),
            max_items=config.max_visual_evidence,
        )
        package = replace(package, visual_evidence=visual, claims=claims)

    export_knowledge_package(package, output_dir, filename_stem=stem)  # validates again
    # internally, writes atomically, and raises (never silently "succeeds") on
    # failure -- see core/knowledge.py. Only after this returns is the durable
    # handoff artifact known to exist, which is what mark_success()/cleanup()
    # below are gated on.

    workspace.mark_success()
    if not config.retain_temp_artifacts:
        workspace.cleanup()

    return package


def _vision_descriptions(result: AnalysisResult) -> dict[float, str]:
    """Vision's own description per analyzed timestamp, when vision ran at all
    -- used to label retained evidence frames. Empty when it didn't; a frame
    is never given a fabricated description."""
    return {o.vision.timestamp_sec: o.vision.description
            for o in result.observations
            if o.vision is not None and o.vision.description}


def _validate_package(package: KnowledgePackage) -> None:
    """Thin delegate to `core.knowledge.validate_knowledge_package` -- kept
    as its own named step in `process_video`'s flow (see above) rather than
    folded silently into `export_knowledge_package`."""
    validate_knowledge_package(package)


def _warn(stage: str, exc: Exception) -> None:
    print(f"[video_lens] {stage} unavailable -- continuing without it: {exc}", file=sys.stderr)


def _try_transcribe(video, config: PipelineConfig) -> Transcript | None:
    try:
        adapter = FasterWhisperAdapter(model_size=config.whisper_model_size,
                                        device=config.whisper_device,
                                        compute_type=config.whisper_compute_type)
        return adapter.transcribe(video)
    except TranscriptionError as e:
        _warn("transcription", e)
        return None


def _try_select_frames(video, config: PipelineConfig) -> tuple[list[Frame], FrameExtractor]:
    extractor = FrameExtractor(cache_dir=config.frame_cache_dir)
    try:
        frames = select_keyframes(video, extractor, interval_sec=config.keyframe_interval_sec,
                                   diff_threshold=config.keyframe_diff_threshold)
        return frames[:config.max_frames], extractor
    except FrameExtractionError as e:
        _warn("frame selection", e)
        return [], extractor


def _detect_pointer_evidence(video, frames: list[Frame], extractor: FrameExtractor,
                              config: PipelineConfig) -> list[PointerEvent]:
    """One `detect_pointer_at` call per selected frame -- NOT
    `detect_pointer_for_frames(frames)` directly. Keyframes are typically
    seconds apart (scene changes / interval sampling), far too sparse for
    three-frame differencing to see motion between consecutive keyframes;
    `detect_pointer_at` extracts its own closely-spaced adjacent frames
    around each keyframe's timestamp instead, giving pointer detection an
    actual chance to find motion evidence near that moment."""
    return [detect_pointer_at(video, f.timestamp_sec, extractor, step_sec=config.pointer_step_sec)
            for f in frames]


def _analyze_vision(frames: list[Frame], transcript: Transcript | None,
                     pointer_events: list[PointerEvent], config: PipelineConfig) -> list[VisionObservation]:
    vision = config.vision_provider or _default_vision_provider(config)
    pointer_by_ts = {p.timestamp_sec: p for p in pointer_events}
    observations = []
    for frame in frames:
        context = transcript.text_around(frame.timestamp_sec) if transcript is not None else None
        pointer = pointer_by_ts.get(frame.timestamp_sec)
        observations.append(vision.analyze_frame(frame, transcript_context=context, pointer=pointer))
    return observations


def _default_vision_provider(config: PipelineConfig) -> VisionAdapter:
    """The built-in default when no `vision_provider` is supplied -- Claude,
    imported here rather than at module level so this file never hard-
    depends on the `anthropic` package (it degrades to
    `status="unavailable"` on its own if unconfigured, same as before)."""
    from adapters.vision.claude_vision import ClaudeVisionAdapter
    kwargs = {"cache_dir": config.vision_cache_dir}
    if config.vision_model is not None:
        kwargs["model"] = config.vision_model
    return ClaudeVisionAdapter(**kwargs)
