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

import math
import os
import sys
from dataclasses import dataclass, replace

from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import select_keyframes
from adapters.ingestion import ingest
from adapters.pointer.cursor_detector import detect_pointer_at, track_pointer
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
from core.contracts import (
    AnalysisResult, CursorSegment, Frame, InspectionRequest, InspectionResult, KnowledgePackage,
    PointerEvent, PointerTrack, Transcript, VideoInput, VisionObservation, VisualChangeEvent,
)
from core.cursor_intelligence import DEFAULT_MAX_INTERVAL_SEC, analyze_track
from core.errors import FrameExtractionError, TranscriptionError
from core.evidence import build_analysis_result
from core.interfaces import KnowledgeSynthesizer, VideoUnderstandingAdapter, VisionAdapter
from core.knowledge import (
    build_knowledge_package, default_filename_stem, export_knowledge_package,
    knowledge_source_for, validate_knowledge_package,
)
from core.session import build_session, save_session, session_filename
from core.synthesis import build_synthesis_brief, synthesize, unavailable
from core.visual_change import detect_visual_changes
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
    max_frames: int = 20  # ceiling on frames sent through pointer/vision -- keeps cost bounded on long videos;
    # over budget, the kept frames are spread evenly across the candidates (_spread), never the first N

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
    # Optional SIBLING to vision_provider, not a replacement: any object
    # implementing core.interfaces.VideoUnderstandingAdapter
    # (`analyze_video(video, *, transcript, start_sec, end_sec, prompt)`) -- a
    # model that reasons over the video itself and returns timestamped
    # observations, including moments Video-Lens never sampled as frames.
    # None (default) -> never called: no calls, no network, no added cost.
    # Supplying one is the opt-in; it runs independently of vision_enabled.
    # Video-Lens ships no implementation (see docs/vision.md).
    video_understanding_provider: VideoUnderstandingAdapter | None = None

    # Schema 1.2 deterministic measurements, both OFF by default (a default run
    # is unchanged). Each becomes `visual_change` / `cursor_track` evidence on
    # the package. Measurements, not interpretations.
    # visual_change_enabled: P0-B on its OWN bounded temporal grid -- every
    # visual_change_interval_sec from 0 to the last decodable frame, never the
    # selected keyframes (chosen because they differ, at gaps up to 170s on a
    # benchmark video). The step widens so a video never exceeds
    # visual_change_max_samples frames. Grid frames are only measured, never
    # sent to vision. At the defaults the grid coincides with keyframe
    # selection's own interval samples, already in the frame cache.
    # cursor_intelligence_enabled: segment the pipeline's own pointer events
    # (needs pointer_enabled) into moving/uncertain stretches (P0-C); keyframes
    # are seconds apart, so expect mostly "uncertain" -- for dense tracking call
    # analyze_cursor_motion directly.
    visual_change_enabled: bool = False
    visual_change_interval_sec: float = 2.0
    visual_change_max_samples: int = 150  # 2s covers ~5 min; longer videos get a wider step
    cursor_intelligence_enabled: bool = False

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

    # Opt-in evidence session (process_video only -- see docs/session.md). None
    # (default) -> nothing is persisted and the run is exactly as before. A
    # directory -> after the package is written, the run's OBSERVATIONS
    # (transcript + evidence, no media, no frames, no conclusions) are saved
    # there as `<name>-<hash>.session.json`. Temp-artifact cleanup is unchanged.
    session_dir: str | None = None


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


def inspect_visual_evidence(video: VideoInput, request: InspectionRequest,
                             extractor: FrameExtractor | None = None) -> InspectionResult:
    """Look closely at one moment, or a window around it, of an already-ingested
    video -- optionally cropped to a `Region` and downscaled -- without running
    the pipeline. This is the targeted counterpart to the whole-video survey
    `analyze_video` does, for when something specific needs a second look.

    A request with no window yields the single frame at `request.timestamp_sec`;
    one with a window yields frames from `timestamp - window_before_sec` through
    `timestamp + window_after_sec` at `request.fps`, clamped to the video's
    duration. Everything goes through the same `FrameExtractor` (and cache) the
    pipeline uses, so a repeated or overlapping inspection never re-extracts a
    frame it already has. Pass your own `extractor` to control where frames are
    cached; by default that is the extractor's own temp directory, which is
    disposable like every other Video-Lens cache -- nothing here creates
    durable media storage.

    Raises `FrameExtractionError` (the missing-file / ffmpeg-failure error the
    pipeline's own frame stage raises) if frames cannot be extracted."""
    extractor = extractor or FrameExtractor()
    options = {"scale_width": request.scale_width, "region": request.region}

    start = max(0.0, request.timestamp_sec - request.window_before_sec)
    end = min(request.timestamp_sec + request.window_after_sec, video.duration_sec)
    if request.fps is None or end <= start:
        # No window requested, or the window collapsed against the end of the
        # video: the one frame at the requested moment (get_frame clamps it).
        frames = [extractor.get_frame(video, request.timestamp_sec, **options)]
    else:
        frames = extractor.extract_window(video, start, end, 1.0 / request.fps, **options)
    return InspectionResult(request=request, frames=tuple(frames), video_source=video.path)


def analyze_cursor_motion(video: VideoInput, start_sec: float, end_sec: float,
                          extractor: FrameExtractor | None = None, interval_sec: float = 0.5,
                          max_interval_sec: float = DEFAULT_MAX_INTERVAL_SEC,
                          ) -> tuple[PointerTrack, tuple[CursorSegment, ...]]:
    """Track the pointer across `[start_sec, end_sec]` and say what that
    establishes about its movement: the existing `track_pointer` (frames every
    `interval_sec`, three-frame differencing), then
    `core.cursor_intelligence.analyze_track`.

    Returns the raw `PointerTrack` (every sample, for provenance) and its
    `CursorSegment`s, which tile the whole range as "moving" or "uncertain".
    There is no "stationary": the detector sees a cursor only while it moves,
    so stillness is never something the evidence can establish (see
    docs/pointer.md). Opt-in and standalone -- `analyze_video` does not call
    this, so default pipeline behavior and cost are unchanged."""
    extractor = extractor or FrameExtractor()
    track = track_pointer(video, start_sec, end_sec, extractor, interval_sec=interval_sec)
    return track, analyze_track(track, max_interval_sec=max_interval_sec)


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

    video_observations: list[VisionObservation] | None = None
    if config.video_understanding_provider is not None:
        video_observations = _analyze_video_understanding(video, transcript, config)

    visual_changes: list[VisualChangeEvent] | None = None
    if config.visual_change_enabled:
        visual_changes = _measure_visual_changes(video, extractor, config)

    cursor_segments: tuple[CursorSegment, ...] | None = None
    if config.cursor_intelligence_enabled:
        cursor_segments = _measure_cursor_motion(pointer_events)

    query_timestamps = [f.timestamp_sec for f in frames]
    if not query_timestamps and transcript is not None:
        query_timestamps = [s.start_sec for s in transcript.segments]
    if video_observations:
        # Each usable native observation gets a correlation window at its own
        # moment, so evidence about a moment no frame was sampled at is kept,
        # not dropped for lack of a nearby frame.
        query_timestamps = sorted(set(query_timestamps) | {
            o.timestamp_sec for o in video_observations if o.status in ("ok", "low_information")})

    result = build_analysis_result(
        video, query_timestamps, transcript=transcript, frames=frames,
        pointer_events=pointer_events, vision_observations=vision_observations,
        video_observations=video_observations, visual_changes=visual_changes,
        cursor_segments=cursor_segments, tolerance_sec=config.tolerance_sec,
    )
    return result, transcript


def process_video(source: str, config: PipelineConfig | None = None) -> KnowledgePackage:
    """The Step 9 lifecycle: run the pipeline in a job-scoped temporary
    workspace, extract a compact `KnowledgePackage`, validate and write it
    to durable storage, and ONLY THEN delete the temporary artifacts this
    job owns (job-downloaded video, extracted frames, vision cache) --
    never the caller's own local source file, which always lives outside
    the job workspace. If anything fails before the package is written,
    nothing is cleaned up -- see docs/lifecycle.md -- and the exception
    carries the preserved workspace's path as `workspace_root`.

    Use this for standalone "give me a video, get me knowledge" usage.
    Use `analyze_video` directly if you want the full evidence-level
    `AnalysisResult` and want to manage storage/caching yourself.
    """
    config = config or PipelineConfig()
    workspace = JobWorkspace()
    try:
        return _process_in(source, config, workspace)
    except BaseException as e:
        # cleanup() never ran, so the temp tree is still there for inspection --
        # say where, so a caller (e.g. video_lens_worker) can report it.
        e.workspace_root = workspace.root
        raise


def _process_in(source: str, config: PipelineConfig, workspace: JobWorkspace) -> KnowledgePackage:
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

    if config.session_dir is not None:
        session = build_session(result, transcript, analysis=_session_analysis(config))
        # the file name is derived from the source, so re-analysing the same
        # video replaces its session (atomically) and a different video can't
        save_session(session, os.path.join(config.session_dir, session_filename(session)), overwrite=True)

    workspace.mark_success()
    if not config.retain_temp_artifacts:
        workspace.cleanup()

    return package


def _session_analysis(config: PipelineConfig) -> dict:
    """The configuration facts needed to read a session's evidence -- which
    streams ran and the correlation tolerance -- as plain booleans and numbers.
    Never a model name, credential, path or provider object."""
    return {
        "tolerance_sec": config.tolerance_sec, "keyframe_interval_sec": config.keyframe_interval_sec,
        "max_frames": config.max_frames, "pointer_enabled": config.pointer_enabled,
        "vision_enabled": config.vision_enabled, "visual_change_enabled": config.visual_change_enabled,
        "visual_change_interval_sec": config.visual_change_interval_sec,
        "visual_change_max_samples": config.visual_change_max_samples,
        "cursor_intelligence_enabled": config.cursor_intelligence_enabled,
        "video_understanding": config.video_understanding_provider is not None,
    }


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


def _analyze_video_understanding(video, transcript: Transcript | None,
                                  config: PipelineConfig) -> list[VisionObservation]:
    """Call the optional native-video provider and keep only what is grounded.

    This is a trust boundary, like semantic synthesis: the provider is
    supplied code. Anything it raises (an `Exception` -- never BaseException,
    so Ctrl+C still stops the job) or returns that isn't a sequence means this
    stream is unavailable, and the job continues without it. Each returned
    item must be a `VisionObservation` with a finite timestamp inside the
    video (+1.0s, the same clamping tolerance used everywhere); anything else
    -- e.g. a whole-video summary with no real timestamp -- is dropped rather
    than attached to some arbitrary frame. Survivors are marked
    `analysis_metadata["evidence_source"] = "video_understanding"`; their order
    doesn't matter, since correlation orders everything by time itself.

    Warnings name the exception TYPE only: provider/SDK errors routinely embed
    request URLs, which may be signed, and Video-Lens must not print them."""
    provider = config.video_understanding_provider
    try:
        returned = provider.analyze_video(video, transcript=transcript, start_sec=0.0,
                                          end_sec=None, prompt=None)
    except Exception as e:  # noqa: BLE001 -- supplied code; any failure degrades, never fails the job
        _warn("video understanding", f"the provider raised {type(e).__name__}")
        return []
    if not isinstance(returned, (list, tuple)):
        _warn("video understanding",
              f"the provider returned {type(returned).__name__}, not a sequence of VisionObservation")
        return []

    kept, dropped = [], 0
    for o in returned:
        ts = getattr(o, "timestamp_sec", None)
        if not (isinstance(o, VisionObservation) and isinstance(ts, (int, float))
                and not isinstance(ts, bool) and math.isfinite(ts)
                and 0.0 <= ts <= video.duration_sec + 1.0):
            dropped += 1
            continue
        metadata = o.analysis_metadata if isinstance(o.analysis_metadata, dict) else {}
        kept.append(replace(o, analysis_metadata={**metadata, "evidence_source": "video_understanding"}))
    if dropped:
        print(f"[video_lens] video understanding: dropped {dropped} observation(s) without a "
              f"valid timestamp inside the video -- never attached to a guessed moment", file=sys.stderr)
    return kept


def _temporal_samples(video, extractor: FrameExtractor, config: PipelineConfig) -> list[Frame]:
    """The visual-change grid: 0, step, 2*step, ... plus the video's last
    decodable frame, so the final stretch is measured too. Every timestamp goes
    through the extractor, whose clamp/end-of-stream rule (docs/frames.md) is the
    only one -- requests past the end collapse onto the last frame, never fail.
    At most `visual_change_max_samples` frames: the step is widened, never the
    budget exceeded."""
    if config.visual_change_interval_sec <= 0 or config.visual_change_max_samples < 2:
        raise ValueError("visual_change_interval_sec must be > 0 and visual_change_max_samples >= 2")
    step = max(config.visual_change_interval_sec,
               video.duration_sec / max(1, config.visual_change_max_samples - 2))
    frames = extractor.extract_window(video, 0.0, video.duration_sec, step)
    last = extractor.get_frame(video, video.duration_sec)
    if last.timestamp_sec > frames[-1].timestamp_sec:
        frames.append(last)
    return frames


def _measure_visual_changes(video, extractor: FrameExtractor,
                            config: PipelineConfig) -> list[VisualChangeEvent]:
    """P0-B over the bounded temporal grid -- independent of keyframe selection
    (consecutive pairs t0->t1, t1->t2, ...). A failure anywhere degrades to
    "configured, nothing measured" (recorded as unavailable), never to a
    zero-magnitude change."""
    try:
        return detect_visual_changes(_temporal_samples(video, extractor, config))
    except Exception as e:
        _warn("visual change detection", type(e).__name__)
        return []


def _measure_cursor_motion(pointer_events: list[PointerEvent]) -> tuple[CursorSegment, ...]:
    """P0-C over the pointer events the pipeline already detected -- no new
    extraction. Fewer than two events cannot show movement: nothing is claimed
    (an empty result is recorded as unavailable, never as a still cursor)."""
    events = sorted(pointer_events, key=lambda e: e.timestamp_sec)
    if len(events) < 2:
        return ()
    detected = sum(1 for e in events if e.status == "detected")
    track = PointerTrack(start_sec=events[0].timestamp_sec, end_sec=events[-1].timestamp_sec,
                         events=events, confidence=detected / len(events))
    try:
        return analyze_track(track)
    except Exception as e:
        _warn("cursor intelligence", type(e).__name__)
        return ()


def _warn(stage: str, exc: Exception | str) -> None:
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
        return _spread(frames, config.max_frames), extractor
    except FrameExtractionError as e:
        _warn("frame selection", e)
        return [], extractor


def _spread(frames: list[Frame], limit: int) -> list[Frame]:
    """At most `limit` of the selected keyframes, evenly spaced by position in
    the (chronological) candidate list, first and last always kept -- not the
    first `limit`, which analyzed only 0-64s of a 270s benchmark video. Under
    budget: unchanged. Exact integer arithmetic, so it is deterministic.
    ponytail: even by candidate index, not by time -- candidates are denser
    where the video changes more (dedupe drops static stretches), so a long
    static stretch can get no pick of its own; switch to time-targeted picks if
    that coverage is ever needed."""
    n = len(frames)
    if n <= limit:
        return list(frames)
    if limit <= 1:
        return list(frames[:limit])
    # round(i * (n-1) / (limit-1)), half-up; strictly increasing since the step is >= 1
    return [frames[(2 * i * (n - 1) + (limit - 1)) // (2 * (limit - 1))] for i in range(limit)]


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
