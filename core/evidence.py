"""Evidence correlation and structured understanding.

RAW EVIDENCE (Transcript, Frame, PointerEvent, VisionObservation -- Steps
3-6) -> CORRELATED EVIDENCE (grouped by timestamp proximity, this module) ->
STRUCTURED OBSERVATION (observed facts + inferences, this module) ->
INTERPRETATION (a future consumer's own domain reasoning -- not built here).

Deterministic and rule-based throughout -- no LLM call is made or needed to
build a StructuredObservation. Every inference cites the Evidence it came
from; nothing here fabricates a fact or a position that wasn't already
present in the raw evidence streams. See docs/evidence.md.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from core.contracts import (
    AnalysisResult, CursorSegment, Evidence, Frame, Inference, MultimodalObservation,
    PointerEvent, StructuredObservation, Transcript, VideoInput, VisionObservation,
    VisualChangeEvent,
)

# `Evidence.source` for vision evidence that came from a native-video provider
# (core.interfaces.VideoUnderstandingAdapter) rather than from a per-frame
# VisionAdapter -- same kind ("vision"), told apart by provenance, never by a
# kind of its own.
NATIVE_VIDEO_SOURCE = "video_understanding"
# Matches the per-excerpt limit core.knowledge enforces on a package's evidence.
_MAX_REF = 400

# Normalized-distance threshold (0-1 scale, same units as PointerEvent's
# normalized_x/y) above which two pointer positions count as "moved" rather
# than jitter/measurement noise. Chosen to be well above the ~pixel-level
# noise a real detector shows on a stationary cursor, well below a
# deliberate cursor move across meaningful screen distance.
# ponytail: fixed threshold, make configurable if callers need per-video tuning
_MOTION_THRESHOLD = 0.03


def _within_tolerance(items, timestamp_sec: float, tolerance_sec: float) -> list:
    """Every item whose `timestamp_sec` falls within `tolerance_sec` of
    `timestamp_sec`, ordered by time. The one correlation primitive every
    evidence stream in this module goes through -- same tolerance, same
    inclusive-boundary rule, everywhere."""
    return sorted(
        (i for i in items if abs(i.timestamp_sec - timestamp_sec) <= tolerance_sec),
        key=lambda i: i.timestamp_sec,
    )


def nearest_within_tolerance(items, timestamp_sec: float, tolerance_sec: float):
    """The single closest item within tolerance, or None. Ties (equal
    distance on both sides) resolve to the earlier item -- deterministic,
    not "whichever the input order happened to put first"."""
    candidates = _within_tolerance(items, timestamp_sec, tolerance_sec)
    if not candidates:
        return None
    return min(candidates, key=lambda i: (abs(i.timestamp_sec - timestamp_sec), i.timestamp_sec))


def all_within_tolerance(items, timestamp_sec: float, tolerance_sec: float) -> list:
    """Every item within tolerance, ordered by time -- used where more than
    one reading matters (pointer motion needs >=2 points, a single nearest
    reading can't show movement)."""
    return _within_tolerance(items, timestamp_sec, tolerance_sec)


def _overlapping(spans, timestamp_sec: float, tolerance_sec: float) -> list:
    """Items with a `start_sec`/`end_sec` span that touches the correlation
    window -- the span-shaped counterpart of `_within_tolerance`, same
    inclusive boundary, for evidence that covers a stretch of time."""
    lo, hi = timestamp_sec - tolerance_sec, timestamp_sec + tolerance_sec
    return sorted((x for x in spans if x.start_sec <= hi and x.end_sec >= lo),
                  key=lambda x: (x.start_sec, x.end_sec))


def _compact(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def visual_change_to_evidence(event: VisualChangeEvent) -> Evidence:
    """The normalized `Evidence` for one P0-B measurement. Lossless for what the
    event asserts: `ref` is deterministic JSON carrying status, kind, magnitude,
    regions (4 decimals) and BOTH compared timestamps -- so the temporal
    relationship between the two frames survives -- while `timestamp_sec` stays
    the event's own (the later frame's), `confidence` its confidence, and
    `source` its `detection_method`. A measurement, never an interpretation:
    nothing here says what the change MEANS."""
    payload = {
        "kind": event.kind, "status": event.status,
        "magnitude": None if event.magnitude is None else round(event.magnitude, 4),
        "compared": list(event.compared_timestamps),
        "regions": [[round(v, 4) for v in (r.x1, r.y1, r.x2, r.y2)] for r in event.regions],
    }
    if event.detail:
        payload["detail"] = event.detail[:80]
    omitted = 0
    while len(_compact(payload)) > _MAX_REF and payload["regions"]:
        payload["regions"].pop()  # a package's excerpts are bounded; say so rather than truncate JSON
        omitted += 1
        payload["regions_omitted"] = omitted
    return Evidence(timestamp_sec=event.timestamp_sec, kind="visual_change", ref=_compact(payload),
                    confidence=event.confidence, source=event.detection_method)


def cursor_segment_to_evidence(segment: CursorSegment) -> Evidence:
    """The normalized `Evidence` for one P0-C segment: span = the segment's
    own `[start_sec, end_sec]`, `confidence` its confidence, and `ref` JSON with
    motion state, direction, speed (frame diagonals/sec) and basis. An
    "uncertain" segment stays uncertain -- it carries no direction or speed --
    and there is no "stationary" to map to."""
    payload = {
        "state": segment.motion_state, "basis": segment.basis,
        "direction_deg": None if segment.direction_deg is None else round(segment.direction_deg, 2),
        "speed_norm": None if segment.mean_speed_norm is None else round(segment.mean_speed_norm, 4),
    }
    return Evidence(timestamp_sec=segment.start_sec, timestamp_end_sec=segment.end_sec,
                    kind="cursor_track", ref=_compact(payload), confidence=segment.confidence,
                    source="cursor_intelligence")


def _usable(observation: VisionObservation | None) -> bool:
    return observation is not None and observation.status in ("ok", "low_information")


def build_structured_observation(
    timestamp_sec: float,
    *,
    transcript: Transcript | None = None,
    frames: list[Frame] | None = None,
    pointer_events: list[PointerEvent] | None = None,
    vision_observations: list[VisionObservation] | None = None,
    video_observations: list[VisionObservation] | None = None,
    visual_changes: Sequence[VisualChangeEvent] | None = None,
    cursor_segments: Sequence[CursorSegment] | None = None,
    tolerance_sec: float = 1.0,
) -> StructuredObservation:
    """Answer "what happened at this point in the video" from whatever
    evidence streams are actually supplied -- any subset may be omitted or
    empty, and the result degrades gracefully (missing streams show up in
    `unavailable`, never as a fabricated fact). Every evidence stream is
    optional and independent; there is no required combination.

    `video_observations` come from an optional `VideoUnderstandingAdapter` (a
    model that watched the video itself). They are correlated exactly like
    per-frame vision -- nearest within tolerance -- but as their own input, so a
    native observation and a per-frame one at the same moment BOTH become
    `vision` evidence rather than one silently displacing the other. The
    deterministic inference rules below still read per-frame vision only.
    `None` (not configured) leaves the result exactly as before; an empty list
    (configured, nothing usable here) records "video_understanding" as
    unavailable.

    `visual_changes` (P0-B events, matched on the event's own timestamp) and
    `cursor_segments` (P0-C segments, matched on span overlap) work the same
    way and become `visual_change` / `cursor_track` evidence. They are
    measurements only: no inference rule reads them. A change that could not be
    compared ("unavailable") is not evidence -- it is recorded as unavailable."""
    if tolerance_sec <= 0:
        raise ValueError(f"tolerance_sec must be > 0, got {tolerance_sec}")

    frame = nearest_within_tolerance(frames or [], timestamp_sec, tolerance_sec)
    vision = nearest_within_tolerance(vision_observations or [], timestamp_sec, tolerance_sec)
    native = nearest_within_tolerance(video_observations or [], timestamp_sec, tolerance_sec)
    segments = (transcript.segments_between(timestamp_sec - tolerance_sec, timestamp_sec + tolerance_sec)
                if transcript is not None else [])
    window_pointers = all_within_tolerance(pointer_events or [], timestamp_sec, tolerance_sec)
    pointer = nearest_within_tolerance(window_pointers, timestamp_sec, tolerance_sec)
    pointer_has_position = pointer is not None and pointer.status != "not_detected"

    observed: list[Evidence] = []
    unavailable: list[str] = []

    if frame is not None:
        observed.append(Evidence(timestamp_sec=frame.timestamp_sec, kind="frame", ref=frame.path))
    else:
        unavailable.append("frame")

    if segments:
        for s in segments:
            observed.append(Evidence(timestamp_sec=s.start_sec, kind="transcript", ref=s.text,
                                      confidence=s.confidence))
    else:
        unavailable.append("transcript")

    if pointer_has_position:
        observed.append(Evidence(
            timestamp_sec=pointer.timestamp_sec, kind="pointer",
            ref=f"x={pointer.x}, y={pointer.y} (status={pointer.status})",
            confidence=pointer.confidence,
        ))
    else:
        unavailable.append("pointer")  # covers: no events supplied, none in window, or all not_detected

    located_elements = []
    if vision is not None and vision.status in ("ok", "low_information"):
        observed.append(Evidence(timestamp_sec=vision.timestamp_sec, kind="vision",
                                  ref=vision.description or "(no description)", confidence=vision.confidence))
        for el in vision.elements:
            observed.append(Evidence(timestamp_sec=vision.timestamp_sec, kind="vision",
                                      ref=f"{el.kind}: {el.description}", confidence=vision.confidence))
            if el.region is not None:
                located_elements.append(el)
    elif not _usable(native):
        unavailable.append("vision")  # neither source saw anything usable here

    if _usable(native):
        observed.append(Evidence(timestamp_sec=native.timestamp_sec, kind="vision",
                                  ref=native.description or "(no description)", confidence=native.confidence,
                                  source=NATIVE_VIDEO_SOURCE))
        for el in native.elements:
            observed.append(Evidence(timestamp_sec=native.timestamp_sec, kind="vision",
                                      ref=f"{el.kind}: {el.description}", confidence=native.confidence,
                                      source=NATIVE_VIDEO_SOURCE))
    elif video_observations is not None:
        unavailable.append("video_understanding")

    if visual_changes is not None:
        changes = [e for e in all_within_tolerance(visual_changes, timestamp_sec, tolerance_sec)
                   if e.status != "unavailable"]
        observed.extend(visual_change_to_evidence(e) for e in changes)
        if not changes:
            unavailable.append("visual_change")

    if cursor_segments is not None:
        segs = _overlapping(cursor_segments, timestamp_sec, tolerance_sec)
        observed.extend(cursor_segment_to_evidence(g) for g in segs)
        if not segs:
            unavailable.append("cursor_track")

    inferences: list[Inference] = []
    disagreements: list[str] = []

    # Rule: pointer position falls within (or outside) a vision-located region.
    if pointer_has_position and vision is not None and vision.status == "ok" and located_elements:
        px, py = pointer.normalized_x, pointer.normalized_y
        if px is not None and py is not None:
            contained = [el for el in located_elements
                         if el.region.x1 <= px <= el.region.x2 and el.region.y1 <= py <= el.region.y2]
            if contained:
                el = contained[0]
                inferences.append(Inference(
                    text=(f"The pointer was detected within the region vision identified as "
                          f"'{el.description}' ({el.kind}). This supports -- but does not prove -- "
                          f"that the speaker or presenter may have been referring to or interacting "
                          f"with that element."),
                    supporting_evidence=(
                        Evidence(timestamp_sec=pointer.timestamp_sec, kind="pointer",
                                 ref=f"x={pointer.x}, y={pointer.y}", confidence=pointer.confidence),
                        Evidence(timestamp_sec=vision.timestamp_sec, kind="vision",
                                 ref=f"{el.kind}: {el.description}", confidence=vision.confidence),
                    ),
                    confidence=round(pointer.confidence * vision.confidence, 4),
                    basis="pointer_in_vision_region",
                ))
            else:
                disagreements.append(
                    f"pointer position ({px:.3f}, {py:.3f}) does not fall within any of the "
                    f"{len(located_elements)} visually located region(s) vision identified"
                )

    # Rule: vision found located elements but no pointer evidence confirms interaction.
    if vision is not None and vision.status == "ok" and located_elements and not pointer_has_position:
        disagreements.append(
            "vision identified visually located element(s), but no pointer evidence is "
            "available in this window to confirm interaction with any of them"
        )

    # Rule: speech present + pointer motion. Needs >=2 `detected` readings in
    # the window with nothing but `detected` readings between them: an
    # `uncertain` position is not trusted, and a `not_detected` one in between
    # means the pointer's path is unknown. Never "stationary": the detector
    # finds the pointer BY its motion, so a small displacement (or none) does
    # not establish stillness -- core.cursor_intelligence's rule, which the
    # benchmark showed this older rule contradicting (docs/benchmark.md).
    detected_at = [i for i, e in enumerate(window_pointers) if e.status == "detected"]
    positioned = window_pointers[detected_at[0]:detected_at[-1] + 1] if detected_at else []
    if (segments and len(positioned) >= 2
            and all(e.status == "detected" for e in positioned)):
        first, last = positioned[0], positioned[-1]
        fx, fy, lx, ly = (first.normalized_x, first.normalized_y,
                          last.normalized_x, last.normalized_y)
        dist = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5 if None not in (fx, fy, lx, ly) else 0.0
        if dist > _MOTION_THRESHOLD:
            completeness = len(positioned) / len(window_pointers)  # data quality, not model confidence
            inferences.append(Inference(
                text=(f"Speech was present while the pointer was moving across the frame "
                      f"(normalized displacement {dist:.3f})."),
                supporting_evidence=(
                    tuple(Evidence(timestamp_sec=s.start_sec, kind="transcript", ref=s.text,
                                    confidence=s.confidence) for s in segments)
                    + tuple(Evidence(timestamp_sec=e.timestamp_sec, kind="pointer",
                                      ref=f"x={e.x}, y={e.y}", confidence=e.confidence)
                            for e in positioned)
                ),
                confidence=round(completeness, 4),
                basis="speech_pointer_motion",
            ))

    return StructuredObservation(
        timestamp_sec=timestamp_sec, tolerance_sec=tolerance_sec,
        observed=tuple(observed), inferences=tuple(inferences),
        disagreements=tuple(disagreements), unavailable=tuple(unavailable), frame=frame,
    )


def build_analysis_result(
    video: VideoInput,
    timestamps: list[float],
    *,
    transcript: Transcript | None = None,
    frames: list[Frame] | None = None,
    pointer_events: list[PointerEvent] | None = None,
    vision_observations: list[VisionObservation] | None = None,
    video_observations: list[VisionObservation] | None = None,
    visual_changes: Sequence[VisualChangeEvent] | None = None,
    cursor_segments: Sequence[CursorSegment] | None = None,
    tolerance_sec: float = 1.0,
) -> AnalysisResult:
    """Build a full AnalysisResult across several query timestamps -- the
    top-level entry point a downstream consumer calls. Each timestamp gets
    its own independent StructuredObservation (this module never merges
    evidence across timestamps); `observations`/`evidence` are also
    populated for compatibility with the Step 1-6 AnalysisResult shape."""
    structured = [
        build_structured_observation(
            t, transcript=transcript, frames=frames, pointer_events=pointer_events,
            vision_observations=vision_observations, video_observations=video_observations,
            visual_changes=visual_changes, cursor_segments=cursor_segments,
            tolerance_sec=tolerance_sec,
        )
        for t in timestamps
    ]

    observations = []
    all_evidence: list[Evidence] = []
    for so in structured:
        all_evidence.extend(so.observed)
        transcript_text = " ".join(e.ref for e in so.observed if e.kind == "transcript") or None
        vision_evidence = next((e for e in so.observed if e.kind == "vision"), None)
        pointer_evidence = next((e for e in so.observed if e.kind == "pointer"), None)
        # One `vision` slot, kept consistent with `description` (the first vision
        # evidence): per-frame vision when usable here, else the native-video
        # observation. Both remain in `so.observed` either way.
        frame_vision = nearest_within_tolerance(vision_observations or [], so.timestamp_sec, tolerance_sec)
        native = nearest_within_tolerance(video_observations or [], so.timestamp_sec, tolerance_sec)
        observations.append(MultimodalObservation(
            timestamp_sec=so.timestamp_sec, frame=so.frame, transcript_context=transcript_text,
            description=vision_evidence.ref if vision_evidence else "",
            pointer=(nearest_within_tolerance(pointer_events or [], so.timestamp_sec, tolerance_sec)
                     if pointer_evidence else None),
            vision=(frame_vision if _usable(frame_vision) else native if _usable(native) else None),
        ))

    # The whole measurement stream, independent of `timestamps`: the same
    # conversion and the same "unavailable is not evidence" rule as the windows.
    stream = tuple(visual_change_to_evidence(e) for e in (visual_changes or ()) if e.status != "unavailable")
    return AnalysisResult(video=video, observations=observations, evidence=all_evidence,
                           structured_observations=structured, visual_changes=stream)
