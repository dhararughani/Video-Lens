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

from core.contracts import (
    AnalysisResult, Evidence, Frame, Inference, MultimodalObservation,
    PointerEvent, StructuredObservation, Transcript, VideoInput, VisionObservation,
)

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


def build_structured_observation(
    timestamp_sec: float,
    *,
    transcript: Transcript | None = None,
    frames: list[Frame] | None = None,
    pointer_events: list[PointerEvent] | None = None,
    vision_observations: list[VisionObservation] | None = None,
    tolerance_sec: float = 1.0,
) -> StructuredObservation:
    """Answer "what happened at this point in the video" from whatever
    evidence streams are actually supplied -- any subset may be omitted or
    empty, and the result degrades gracefully (missing streams show up in
    `unavailable`, never as a fabricated fact). Every evidence stream is
    optional and independent; there is no required combination."""
    if tolerance_sec <= 0:
        raise ValueError(f"tolerance_sec must be > 0, got {tolerance_sec}")

    frame = nearest_within_tolerance(frames or [], timestamp_sec, tolerance_sec)
    vision = nearest_within_tolerance(vision_observations or [], timestamp_sec, tolerance_sec)
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
    else:
        unavailable.append("vision")

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

    # Rule: speech present + pointer motion/stationary (needs >=2 positioned
    # readings in the window -- a single point can't show movement).
    positioned = [e for e in window_pointers if e.status != "not_detected"]
    if segments and len(positioned) >= 2:
        first, last = positioned[0], positioned[-1]
        fx, fy, lx, ly = (first.normalized_x, first.normalized_y,
                          last.normalized_x, last.normalized_y)
        if None not in (fx, fy, lx, ly):
            dist = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
            moving = dist > _MOTION_THRESHOLD
            completeness = len(positioned) / len(window_pointers)  # data quality, not model confidence
            inferences.append(Inference(
                text=(f"Speech was present while the pointer was "
                      f"{'moving across the frame' if moving else 'roughly stationary'} "
                      f"(normalized displacement {dist:.3f})."),
                supporting_evidence=(
                    tuple(Evidence(timestamp_sec=s.start_sec, kind="transcript", ref=s.text,
                                    confidence=s.confidence) for s in segments)
                    + tuple(Evidence(timestamp_sec=e.timestamp_sec, kind="pointer",
                                      ref=f"x={e.x}, y={e.y}", confidence=e.confidence)
                            for e in positioned)
                ),
                confidence=round(completeness, 4),
                basis="speech_pointer_motion" if moving else "speech_pointer_stationary",
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
            vision_observations=vision_observations, tolerance_sec=tolerance_sec,
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
        observations.append(MultimodalObservation(
            timestamp_sec=so.timestamp_sec, frame=so.frame, transcript_context=transcript_text,
            description=vision_evidence.ref if vision_evidence else "",
            pointer=(nearest_within_tolerance(pointer_events or [], so.timestamp_sec, tolerance_sec)
                     if pointer_evidence else None),
            vision=(nearest_within_tolerance(vision_observations or [], so.timestamp_sec, tolerance_sec)
                    if vision_evidence else None),
        ))

    return AnalysisResult(video=video, observations=observations, evidence=all_evidence,
                           structured_observations=structured)
