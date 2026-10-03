"""Core data contracts for Video-Lens.

These are plain data shapes, not implementations. Nothing here imports
Whisper, ffmpeg, PySceneDetect, or any specific vision/LLM library — adapters
translate to/from these shapes so the core stays swappable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VideoInput:
    """A video to analyze, already resolved to a local file path.

    `source_type`/`source` record where the video came from ("local" file
    path or "url"); `path` is always the actual local media ffmpeg can read
    (== `source` for local files, the downloaded copy for URLs).
    """
    path: str
    duration_sec: float
    width: int
    height: int
    fps: float
    has_audio: bool
    source_type: str = "local"  # "local" | "url"
    source: str | None = None  # original path or URL; None => same as `path`
    video_codec: str | None = None
    audio_codec: str | None = None
    format_name: str | None = None
    title: str | None = None  # real metadata title only (ffprobe format tag /
    # yt-dlp info) -- never fabricated from a filename; None when the source
    # genuinely carries no title metadata.


@dataclass(frozen=True)
class Word:
    """Word-level timing. Only populated by engines that genuinely provide it
    -- never interpolated or fabricated from segment bounds."""
    text: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    confidence: float | None = None
    words: tuple[Word, ...] | None = None  # None => engine didn't provide them


@dataclass(frozen=True)
class Transcript:
    """Timestamped speech for one video.

    Times are seconds from the start of the video (see TIME MODEL in
    docs/speech.md). `status` distinguishes "transcribed fine" from "there
    was nothing to transcribe" so an empty `segments` is never ambiguous.
    """
    segments: list[TranscriptSegment]
    language: str | None = None
    source: str = "unknown"  # e.g. "faster_whisper", "native_captions"
    duration_sec: float | None = None  # of the source video, when known
    status: str = "ok"  # "ok" | "no_audio" | "no_speech"

    def __post_init__(self):
        prev_start = None
        for s in self.segments:
            if s.end_sec < s.start_sec:
                raise ValueError(f"segment ends before it starts: {s.start_sec} > {s.end_sec}")
            if prev_start is not None and s.start_sec < prev_start:
                raise ValueError(
                    f"segments must be ordered by start time: {s.start_sec} follows {prev_start}"
                )
            prev_start = s.start_sec
        if self.status == "ok" and not self.segments:
            raise ValueError("status 'ok' with no segments -- use 'no_speech' or 'no_audio'")

    # --- time access (see docs/speech.md). Linear scans: a 3h video is only a
    # few thousand segments, so an index would be premature.
    # ponytail: linear scan, add bisect if segment counts ever reach 100k+

    def segment_at(self, t_sec: float) -> TranscriptSegment | None:
        """The segment covering `t_sec`, or None if nobody is speaking then.
        Whisper can emit slightly overlapping segments; the first match wins."""
        for s in self.segments:
            if s.start_sec <= t_sec <= s.end_sec:
                return s
        return None

    def segments_between(self, start_sec: float, end_sec: float) -> list[TranscriptSegment]:
        """Every segment overlapping the window (not just fully contained)."""
        return [s for s in self.segments
                if s.start_sec < end_sec and s.end_sec > start_sec]

    def text_around(self, t_sec: float, window_sec: float = 5.0) -> str:
        """Speech spoken near `t_sec` -- the context for a frame at that time."""
        segs = self.segments_between(t_sec - window_sec, t_sec + window_sec)
        return " ".join(s.text.strip() for s in segs).strip()


@dataclass(frozen=True)
class Frame:
    """One extracted video frame, with enough provenance to join it back to
    its source and to future PointerEvent/vision data by timestamp."""
    timestamp_sec: float
    path: str  # path to the extracted frame image on disk
    source_video: str = ""  # VideoInput.path this frame came from
    width: int = 0
    height: int = 0
    format: str = "jpg"
    frame_index: int | None = None  # best-effort round(timestamp_sec * fps); ffmpeg
    # seek is not guaranteed frame-exact (see docs/frames.md), so this is an
    # estimate for future correlation, never authoritative.
    reason: str = "sampled"  # "scene_change" | "direct_timestamp" | "window_sample"


@dataclass(frozen=True)
class PointerEvent:
    """A detected cursor/pointer position baked into the video pixels -- or
    the deliberate absence of one. `status` is the honest signal: never
    fabricate x/y just because a caller wants a position.
    """
    timestamp_sec: float
    status: str  # "detected" | "not_detected" | "uncertain"
    x: int | None = None
    y: int | None = None
    frame_width: int = 0  # for normalized_x/y -- not duplicated as stored floats
    frame_height: int = 0
    confidence: float = 0.0
    detection_method: str = ""

    def __post_init__(self):
        if self.status not in ("detected", "not_detected", "uncertain"):
            raise ValueError(f"invalid PointerEvent status: {self.status!r}")
        if self.status != "not_detected" and (self.x is None or self.y is None):
            raise ValueError("x/y are required when status is not 'not_detected'")

    @property
    def normalized_x(self) -> float | None:
        if self.x is None or self.frame_width <= 0:
            return None
        return self.x / self.frame_width

    @property
    def normalized_y(self) -> float | None:
        if self.y is None or self.frame_height <= 0:
            return None
        return self.y / self.frame_height


@dataclass(frozen=True)
class PointerTrack:
    """An ordered run of PointerEvents across a time range, so a caller can
    ask "where was the pointer around timestamp X" without re-running
    detection over the whole video."""
    start_sec: float
    end_sec: float
    events: list[PointerEvent] = field(default_factory=list)
    confidence: float = 0.0  # fraction of events with status == "detected"


# CursorSegment.motion_state. There is deliberately NO "stationary": the pointer
# detector finds a cursor by its motion, so a cursor that holds still is
# invisible to it -- "no movement observed" is absence of evidence, never
# evidence of stillness. See core/cursor_intelligence.py.
CURSOR_MOTION_STATES = ("moving", "uncertain")


@dataclass(frozen=True)
class CursorSegment:
    """One stretch of a PointerTrack's time range and what the pointer evidence
    establishes about it. Segments from `analyze_track` tile the track's range
    with no gaps, so every moment is accounted for -- "uncertain" included.

    `direction_deg`: image convention, 0 = right, 90 = down, 180 = left,
    270 = up, in [0, 360). `mean_speed_norm`: frame diagonals per second.
    Both are only ever set for "moving", and only when the evidence supports
    them; `basis` names the deterministic rule that produced the segment."""
    start_sec: float
    end_sec: float
    motion_state: str  # one of CURSOR_MOTION_STATES
    direction_deg: float | None = None
    mean_speed_norm: float | None = None
    confidence: float = 0.0
    basis: str = ""

    def __post_init__(self):
        if self.motion_state not in CURSOR_MOTION_STATES:
            raise ValueError(f"invalid CursorSegment motion_state: {self.motion_state!r}")
        if self.end_sec < self.start_sec:
            raise ValueError(f"segment ends before it starts: {self.start_sec} > {self.end_sec}")
        if not self.basis:
            raise ValueError("a CursorSegment must state its basis")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.direction_deg is not None and not (0.0 <= self.direction_deg < 360.0):
            raise ValueError(f"direction_deg must be in [0, 360), got {self.direction_deg}")
        if self.motion_state == "uncertain":
            if self.direction_deg is not None or self.mean_speed_norm is not None:
                raise ValueError("an 'uncertain' segment establishes no direction or speed")
            if self.confidence != 0.0:
                raise ValueError("an 'uncertain' segment establishes nothing -- confidence must be 0.0")
        elif self.mean_speed_norm is None or self.mean_speed_norm <= 0:
            raise ValueError("a 'moving' segment must report a positive speed")


@dataclass(frozen=True)
class Region:
    """A bounding box normalized to 0.0-1.0 in both axes, so the same region
    is meaningful across frames of any resolution."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        for name, v in (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Region.{name} must be in [0, 1], got {v}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Region must have x2>x1 and y2>y1, got ({self.x1},{self.y1})-({self.x2},{self.y2})"
            )


@dataclass(frozen=True)
class InspectionRequest:
    """A precise "look closely here" request: one moment, or a time window
    around it, optionally cropped to a `Region` and downscaled. Everything
    except `timestamp_sec` is optional and defaults to "the whole frame at
    source resolution".

    `fps=None` means a single frame at `timestamp_sec`. A request with a time
    window MUST say how densely to sample it -- there is no sensible implicit
    default, and silently picking one would make the cost of a request
    unpredictable. `scale_width` is a target width in pixels; height follows
    from the (cropped) aspect ratio."""
    timestamp_sec: float
    window_before_sec: float = 0.0
    window_after_sec: float = 0.0
    fps: float | None = None
    scale_width: int | None = None
    region: Region | None = None

    def __post_init__(self):
        for name, v in (("timestamp_sec", self.timestamp_sec),
                        ("window_before_sec", self.window_before_sec),
                        ("window_after_sec", self.window_after_sec)):
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"InspectionRequest.{name} must be a finite number >= 0, got {v}")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError(f"InspectionRequest.fps must be a finite number > 0, got {self.fps}")
        if self.scale_width is not None and (
                isinstance(self.scale_width, bool) or not isinstance(self.scale_width, int)
                or self.scale_width <= 0):
            raise ValueError(f"InspectionRequest.scale_width must be a positive integer, "
                             f"got {self.scale_width!r}")
        if (self.window_before_sec or self.window_after_sec) and self.fps is None:
            raise ValueError("InspectionRequest with a time window requires fps -- "
                             "say how densely to sample it")


@dataclass(frozen=True)
class InspectionResult:
    """What a targeted inspection produced, with the request that produced it
    so a result is self-describing. `frames` are ordered by time, each with the
    ordinary `Frame` provenance (`source_video`, `timestamp_sec`); for a cropped
    or scaled request their `width`/`height` are the OUTPUT dimensions, not the
    source's."""
    request: InspectionRequest
    frames: tuple[Frame, ...]
    video_source: str  # VideoInput.path the frames were extracted from
    extraction_method: str = "ffmpeg_seek"


# VisualChangeEvent vocabularies. `kind` names the DETECTION LEVEL only -- how
# far the change spread -- never what it means. Semantic labels ("scroll",
# "zoom", "redraw", ...) are interpretation and deliberately absent; see
# core/visual_change.py.
VISUAL_CHANGE_STATUSES = ("detected", "not_detected", "unavailable")
VISUAL_CHANGE_KINDS = ("visual_change", "visual_change_localized", "visual_change_global")


@dataclass(frozen=True)
class VisualChangeEvent:
    """A deterministic measurement of how much the picture changed between two
    sampled frames -- a raw observation, not an interpretation. Not (yet) an
    `Evidence` kind: see core/visual_change.py.

    `status` is the honest signal, as on `PointerEvent`:
      "detected"     -- compared, and `magnitude >= threshold`
      "not_detected" -- compared, and below threshold
      "unavailable"  -- could NOT be compared (unreadable frame, different
                        sizes, no OpenCV). `magnitude` is then None, never 0.0:
                        a failed comparison is not evidence the frames matched.

    `timestamp_sec` is always the LATER frame's (`compared_timestamps[1]`): the
    first moment the new state is observed. The change itself happened
    somewhere in `(compared_timestamps[0], compared_timestamps[1]]`; no finer
    precision than the sampling interval is claimed."""
    timestamp_sec: float
    kind: str  # one of VISUAL_CHANGE_KINDS
    status: str  # one of VISUAL_CHANGE_STATUSES
    magnitude: float | None  # changed fraction of frame area, [0, 1]; None iff unavailable
    regions: tuple[Region, ...] = ()  # where it changed; non-empty iff detected
    confidence: float = 0.0  # distance from the decision threshold -- see core/visual_change.py
    detection_method: str = ""  # algorithm + parameters, so an event is self-describing
    compared_timestamps: tuple[float, float] = (0.0, 0.0)  # (earlier, later)
    detail: str = ""  # why unavailable, or what was truncated -- plain language

    def __post_init__(self):
        if self.status not in VISUAL_CHANGE_STATUSES:
            raise ValueError(f"invalid VisualChangeEvent status: {self.status!r}")
        if self.kind not in VISUAL_CHANGE_KINDS:
            raise ValueError(f"invalid VisualChangeEvent kind: {self.kind!r}")
        earlier, later = self.compared_timestamps
        if later < earlier:
            raise ValueError(f"compared_timestamps must be (earlier, later), got {self.compared_timestamps}")
        if self.timestamp_sec != later:
            raise ValueError("VisualChangeEvent.timestamp_sec must be the later compared timestamp")
        if (self.magnitude is None) != (self.status == "unavailable"):
            raise ValueError("magnitude must be None exactly when status is 'unavailable'")
        if self.magnitude is not None and not (0.0 <= self.magnitude <= 1.0):
            raise ValueError(f"magnitude must be in [0, 1], got {self.magnitude}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.status == "unavailable" and self.confidence != 0.0:
            raise ValueError("an 'unavailable' VisualChangeEvent establishes nothing -- confidence must be 0.0")
        if self.status == "detected":
            if self.kind == "visual_change":
                raise ValueError("a detected change must say whether it is localized or global")
            if not self.regions:
                raise ValueError("a detected change must locate at least one region")
        else:
            if self.kind != "visual_change":
                raise ValueError(f"kind {self.kind!r} asserts an extent; only valid when detected")
            if self.regions:
                raise ValueError("regions are only reported for a detected change")


@dataclass(frozen=True)
class VisualElement:
    """One thing the vision model called out inside a frame. `kind` is free
    text ("text", "chart", "button", "cursor_target", ...), not a fixed enum
    -- Video-Lens stays domain-neutral, so this never hardcodes a vocabulary
    tied to trading, UI frameworks, or any other single content type."""
    kind: str
    description: str
    region: Region | None = None
    region_confidence: str = "unknown"  # "detected" | "approximate" | "unknown"

    def __post_init__(self):
        if self.region_confidence not in ("detected", "approximate", "unknown"):
            raise ValueError(f"invalid region_confidence: {self.region_confidence!r}")
        if self.region is None and self.region_confidence != "unknown":
            raise ValueError("region_confidence must be 'unknown' when region is None")
        if self.region is not None and self.region_confidence == "unknown":
            raise ValueError("a located region must carry 'detected' or 'approximate' confidence")


@dataclass(frozen=True)
class VisionObservation:
    """One vision backend's analysis of a single frame -- honest about every
    way this can fail to produce real content. `status` distinguishes a
    genuine result from every failure mode; fields other than `status` are
    only ever populated for `status == "ok"`/`"low_information"`, never
    fabricated to fill in for `"failed"`/`"unavailable"`."""
    timestamp_sec: float
    frame_path: str
    status: str  # "ok" | "low_information" | "failed" | "unavailable"
    description: str = ""
    visible_text: tuple[str, ...] = ()
    elements: tuple[VisualElement, ...] = ()
    confidence: float = 0.0
    model: str = ""
    analysis_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("ok", "low_information", "failed", "unavailable"):
            raise ValueError(f"invalid VisionObservation status: {self.status!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class MultimodalObservation:
    """One unit of understanding grounded in a frame + nearby transcript +
    (when available) pointer/vision evidence -- Video-Lens's central
    "what was said, what was visible, where the pointer was" join, all keyed
    by the same `timestamp_sec` every adapter shares."""
    timestamp_sec: float
    frame: Frame | None
    transcript_context: str | None
    description: str
    pointer: PointerEvent | None = None
    vision: VisionObservation | None = None


# Every Evidence.kind. The six are the whole vocabulary: the first four are the
# original streams; "visual_change" and "cursor_track" (schema 1.2) are the
# deterministic temporal measurements from core/visual_change.py and
# core/cursor_intelligence.py. Native-video vision is NOT a kind of its own --
# it is "vision", told apart by `Evidence.source`.
EVIDENCE_KINDS = ("frame", "transcript", "pointer", "vision", "visual_change", "cursor_track")


@dataclass(frozen=True)
class Evidence:
    """A pointer back to ONE actual piece of source material -- a raw fact,
    never a derived conclusion. `confidence` (when the underlying source has
    one) preserves the originating evidence's own quality signal, e.g. a
    `PointerEvent.confidence` or `VisionObservation.confidence` -- this is
    what lets a later `Inference`'s confidence reflect evidence quality
    rather than being invented fresh.

    Schema 1.2 adds two optional fields, both defaulting to the old behavior:
    `timestamp_end_sec` (the end of a span, for evidence that covers a stretch
    of time rather than a moment: the two compared frames of a visual change, a
    cursor segment) and `source` (who produced it -- a detector's method, or
    "video_understanding" for a native-video provider's `vision` evidence, so
    it is never mistaken for per-frame vision). For `visual_change` and
    `cursor_track` the structured measurement lives in `ref` as compact,
    deterministic JSON (see core/evidence.py)."""
    timestamp_sec: float
    kind: str  # one of EVIDENCE_KINDS
    ref: str  # frame path, transcript segment text, pointer summary, vision description,
    # or (visual_change / cursor_track) a compact JSON measurement
    confidence: float | None = None  # None -- the source kind carries no confidence of its own
    timestamp_end_sec: float | None = None  # None -- a single moment
    source: str = ""  # provenance label; "" -- nothing beyond `kind` to say

    def __post_init__(self):
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(f"invalid Evidence kind: {self.kind!r}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.timestamp_end_sec is not None and self.timestamp_end_sec < self.timestamp_sec:
            raise ValueError(f"evidence span ends before it starts: "
                             f"{self.timestamp_sec} > {self.timestamp_end_sec}")


@dataclass(frozen=True)
class Inference:
    """A conclusion Video-Lens drew FROM observed evidence -- itself never
    evidence, and never to be presented as an observed fact. Every inference
    must cite the `Evidence` it was drawn from and state a `basis` (a short,
    deterministic rule label -- see core/evidence.py) so a caller can see
    exactly which rule produced it and why, not just trust a bare claim."""
    text: str
    supporting_evidence: tuple[Evidence, ...]
    confidence: float  # reflects the CITED evidence's quality/agreement -- see core/evidence.py
    basis: str  # e.g. "pointer_in_vision_region", "speech_pointer_motion"

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.supporting_evidence:
            raise ValueError("an Inference must cite at least one piece of supporting Evidence")
        if not self.basis:
            raise ValueError("an Inference must state its basis")


@dataclass(frozen=True)
class StructuredObservation:
    """Video-Lens's structured answer to "what happened at this point in the
    video" -- built by core/evidence.py from evidence correlated within
    `tolerance_sec` of `timestamp_sec` (the correlation window is exactly
    `[timestamp_sec - tolerance_sec, timestamp_sec + tolerance_sec]`; no
    separate start/end fields are kept since they're fully determined by
    these two). Strictly separates `observed` (evidence-backed facts) from
    `inferences` (Video-Lens's own conclusions) -- an inference is never
    folded into `observed`, and `observed` never contains anything without
    a citable `Evidence` item."""
    timestamp_sec: float
    tolerance_sec: float
    observed: tuple[Evidence, ...] = ()
    inferences: tuple[Inference, ...] = ()
    disagreements: tuple[str, ...] = ()  # conflicting/incomplete evidence, recorded not resolved
    unavailable: tuple[str, ...] = ()  # evidence stream names with nothing found in this window
    frame: Frame | None = None


@dataclass(frozen=True)
class AnalysisResult:
    video: VideoInput
    observations: list[MultimodalObservation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    structured_observations: list[StructuredObservation] = field(default_factory=list)
    summary: str | None = None
    # Every visual-change measurement supplied to the analysis, as Evidence --
    # the whole stream, NOT only what fell inside a query window above. The
    # windows are per-keyframe; a measurement stream sampled on its own time
    # grid must reach the package and session even where no keyframe is near.
    # `visual_change` evidence in the windows is a subset of this (same
    # Evidence values), so consumers merge the two and drop exact duplicates.
    visual_changes: tuple[Evidence, ...] = ()


# --------------------------------------------------------------------------
# Step 9: compact, domain-neutral knowledge output. "Artifacts are
# temporary. Knowledge is permanent." -- see docs/lifecycle.md.
#
# KNOWLEDGE_SCHEMA_VERSION describes the *shape* of KnowledgePackage/KeyPoint/
# KnowledgeSource/ProcessingMetadata below -- independent of
# core.knowledge.VERSION (the software release). A downstream consumer
# gates its parsing logic on this, not on the software version, so a
# patch release that doesn't touch these shapes never forces a consumer
# migration, and a real field-shape change always bumps this (see
# docs/lifecycle.md, "Schema version").
#
# 1.1 (Step 11) added the optional semantic-synthesis fields to
# KnowledgePackage -- `semantic_summary`, `claims`, `visual_evidence`,
# `synthesis` -- plus the Claim/VisualEvidence/SynthesisMetadata shapes they
# use. All four default to empty/None, so a 1.0 consumer reading a 1.1
# package still finds every field it already knew, unchanged.
#
# 1.2 (P1-B) made "visual_change" and "cursor_track" first-class Evidence kinds
# (EVIDENCE_KINDS), gave Evidence the optional `timestamp_end_sec` / `source`
# fields, and added the optional `visual_changes` / `cursor_intelligence`
# fields to KnowledgePackage. All default to empty, so a 1.1 consumer reading
# a 1.2 package finds every field it already knew, unchanged.
# --------------------------------------------------------------------------
KNOWLEDGE_SCHEMA_VERSION = "1.2"


@dataclass(frozen=True)
class KeyPoint:
    """One deterministically-extracted unit of information -- grounded in
    the `Evidence` it was found in, never invented. `kind` is free text
    ("explanation" | "definition" | "procedure" | "warning" | "example" |
    "conclusion" | "observation" | ...), matching `VisualElement.kind`'s
    domain-neutral, non-enum design. `nature` makes the Step 7 observed-vs-
    inferred distinction explicit and machine-readable at the package level
    too, not just recoverable by knowing which package field a KeyPoint came
    from: `"observed"` for a point that IS the cited evidence (e.g. a
    transcript quote), `"inferred"` for a point Video-Lens concluded FROM
    evidence (e.g. a reused Step 7 `Inference`)."""
    text: str
    kind: str
    timestamp_sec: float
    supporting_evidence: tuple[Evidence, ...]
    confidence: float
    nature: str = "observed"  # "observed" | "inferred"

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.supporting_evidence:
            raise ValueError("a KeyPoint must cite at least one piece of supporting Evidence")
        if self.nature not in ("observed", "inferred"):
            raise ValueError(f"invalid KeyPoint nature: {self.nature!r}")


@dataclass(frozen=True)
class KnowledgeSource:
    """Provenance for a KnowledgePackage -- enough to point back at the
    original video without retaining the video itself."""
    source_type: str  # "local" | "url"
    source: str
    duration_sec: float
    title: str | None = None


@dataclass(frozen=True)
class ProcessingMetadata:
    """Honest bookkeeping about how this package was produced -- which
    evidence streams actually contributed vs. were unavailable, so a
    downstream consumer never mistakes "we didn't check" for "we checked and
    found nothing"."""
    generated_at: str  # ISO 8601 UTC
    video_lens_version: str
    knowledge_schema_version: str  # see KNOWLEDGE_SCHEMA_VERSION above
    frame_count: int
    transcript_status: str  # Transcript.status, or "unavailable"
    stages_available: tuple[str, ...]
    stages_unavailable: tuple[str, ...]


# --------------------------------------------------------------------------
# Step 11: semantic knowledge synthesis + verified visual evidence.
# See docs/synthesis.md. Everything below is OPTIONAL on a KnowledgePackage --
# when no semantic synthesizer is supplied, Video-Lens produces exactly the
# Step 9/10 deterministic package it always did, and says so honestly in
# `synthesis.status` rather than manufacturing an AI-looking summary.
# --------------------------------------------------------------------------

# Claim.status -- the four distinct epistemic states Step 11 requires, kept
# deliberately separate from `confidence` (a claim can be OBSERVED with low
# confidence, or CONFLICTING with high confidence in each disagreeing
# stream). Never collapse these into a single score.
CLAIM_STATUSES = ("observed", "inferred", "unavailable", "conflicting")


@dataclass(frozen=True)
class VisualEvidence:
    """One selected evidence frame that accompanies a KnowledgePackage --
    a *reference* to a small image file written beside the package JSON, never
    the image bytes themselves (no base64, no embedded media; see
    docs/synthesis.md, "Storage"). `image_path` is relative to the package
    JSON's own directory, so a bundle stays portable when moved or copied."""
    evidence_id: str
    timestamp_sec: float
    image_path: str  # relative to the package JSON's directory
    width: int
    height: int
    byte_size: int
    selection_reason: str  # why THIS frame was retained (see core/visual_evidence.py)
    description: str = ""  # vision's own description, when vision actually ran

    def __post_init__(self):
        if not self.evidence_id:
            raise ValueError("VisualEvidence requires an evidence_id")
        if self.timestamp_sec < 0:
            raise ValueError(f"VisualEvidence timestamp must be >= 0, got {self.timestamp_sec}")


@dataclass(frozen=True)
class Claim:
    """One semantically-synthesized unit of knowledge, verified by Video-Lens
    against the evidence it cites before being retained.

    A Claim differs from a `KeyPoint` in origin, not in rigor: a KeyPoint is
    deterministically extracted (a cue-phrase transcript quote or a Step 7
    inference), while a Claim comes from a semantic synthesizer -- and is
    therefore *cross-checked* (`verification`) rather than trusted. A claim
    citing evidence that does not exist is rejected outright, never retained
    with a lowered score (see core/synthesis.py).

    `kind` is free text ("lesson", "fact", "concept", "procedure",
    "conclusion", "warning", "example", "distinction", "relationship", ...),
    matching the domain-neutral, non-enum design of `VisualElement.kind` and
    `KeyPoint.kind` -- no domain vocabulary is ever hardcoded here.
    """
    text: str
    kind: str
    status: str  # one of CLAIM_STATUSES
    timestamp_sec: float
    supporting_evidence: tuple[Evidence, ...]
    confidence: float
    verification: str  # short label of how this claim was cross-checked
    timestamp_end_sec: float | None = None  # set when the claim spans a range
    visual_evidence: tuple[str, ...] = ()  # VisualEvidence.evidence_id references
    limitations: tuple[str, ...] = ()  # what this claim's evidence could NOT establish

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("a Claim must have text")
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"invalid Claim status: {self.status!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        # "unavailable" is the one status that MAY carry no evidence -- it
        # exists precisely to record "the system could not establish this".
        # Every other status asserts something about real evidence and must
        # cite it, exactly as Inference/KeyPoint already require.
        if self.status != "unavailable" and not self.supporting_evidence:
            raise ValueError(f"a Claim with status {self.status!r} must cite supporting Evidence")
        if self.status == "unavailable" and self.confidence != 0.0:
            raise ValueError("an 'unavailable' Claim must have confidence 0.0 -- it establishes nothing")
        if self.timestamp_end_sec is not None and self.timestamp_end_sec < self.timestamp_sec:
            raise ValueError(f"claim range ends before it starts: "
                             f"{self.timestamp_sec} > {self.timestamp_end_sec}")
        if not self.verification:
            raise ValueError("a Claim must record how it was verified")


@dataclass(frozen=True)
class SynthesisMetadata:
    """Honest bookkeeping about the semantic-synthesis stage -- including
    what was REJECTED, so a consumer can see that verification actually ran
    and did something, rather than having to trust that it did.

    `status`: "ok" (a provider ran and produced retained claims) |
    "unavailable" (no synthesizer was supplied -- the default) | "failed"
    (the provider raised) | "rejected" (the provider answered, but nothing it
    returned survived verification)."""
    status: str  # "ok" | "unavailable" | "failed" | "rejected"
    provider: str = ""  # provider-reported name; never a hardcoded vendor
    claims_returned: int = 0
    claims_retained: int = 0
    claims_rejected: int = 0
    rejection_reasons: tuple[str, ...] = ()
    detail: str = ""  # why status is unavailable/failed, in plain language

    def __post_init__(self):
        if self.status not in ("ok", "unavailable", "failed", "rejected"):
            raise ValueError(f"invalid SynthesisMetadata status: {self.status!r}")


@dataclass(frozen=True)
class BriefEvidenceItem:
    """One piece of real evidence, given a short stable id the synthesizer
    cites in its response. The id is what makes fabricated provenance
    detectable: a claim citing an id that was never in the brief cannot be
    resolved back to real evidence, and is rejected."""
    evidence_id: str
    evidence: Evidence


@dataclass(frozen=True)
class SynthesisBrief:
    """Everything a semantic synthesizer is given -- built from ALREADY
    CORRELATED evidence (Step 7), never a raw transcript dump.

    Carries both a rendered prompt pair (`system_prompt`/`user_prompt`, for
    the common case of a text LLM behind the seam) and the structured
    `items` those prompts were rendered from, so a provider that is not a
    text LLM can work from the structure directly and ignore the prose."""
    source: KnowledgeSource
    items: tuple[BriefEvidenceItem, ...]
    system_prompt: str
    user_prompt: str
    disagreements: tuple[str, ...] = ()
    stages_unavailable: tuple[str, ...] = ()

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {i.evidence_id: i.evidence for i in self.items}


@dataclass(frozen=True)
class KnowledgePackage:
    """The durable, compact output of the Step 9 lifecycle -- what survives
    after temporary artifacts (video, frames, caches) are deleted. Small by
    construction: no raw transcript dump, no embedded images/video, no
    per-frame entries -- just the source, a deterministic summary, topics,
    key lessons, important observations, their provenance, and honestly
    stated limitations. See docs/lifecycle.md.

    The last four fields are Step 11's optional semantic layer (see
    docs/synthesis.md); all default to empty/None so a package built without
    a semantic synthesizer is exactly the Step 9/10 package it always was.
    `summary` remains the deterministic sentence in every case --
    `semantic_summary` is populated ONLY by a real synthesizer and is never
    a dressed-up copy of the deterministic one."""
    source: KnowledgeSource
    summary: str
    topics: tuple[str, ...]
    key_lessons: tuple[KeyPoint, ...]
    important_observations: tuple[KeyPoint, ...]
    evidence: tuple[Evidence, ...]
    limitations: tuple[str, ...]
    processing: ProcessingMetadata
    semantic_summary: str | None = None
    claims: tuple[Claim, ...] = ()
    visual_evidence: tuple[VisualEvidence, ...] = ()
    synthesis: SynthesisMetadata | None = None
    # Schema 1.2: the deterministic temporal measurements, exposed by category
    # so a caller need not filter `evidence` by kind. Evidence of kind
    # "visual_change" / "cursor_track" respectively; empty unless the run
    # enabled them (see PipelineConfig). Measurements, not claims.
    visual_changes: tuple[Evidence, ...] = ()
    cursor_intelligence: tuple[Evidence, ...] = ()
