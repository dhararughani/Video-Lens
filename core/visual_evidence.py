"""Step 11: intelligent selection of the few frames worth KEEPING.

A job may extract hundreds or thousands of candidate frames. Almost none of
them are worth retaining: the durable output is a knowledge artifact, not a
media archive. This module decides which frames materially help a reader
understand or verify retained knowledge, discards near-duplicates, and copies
only the survivors into a small evidence bundle beside the package JSON.

Selection is driven by DEMAND, not by a quota. A frame is a candidate only
because some retained claim (or, without a synthesizer, some retained key
point) actually points at it; near-duplicates are then collapsed. There is no
"keep N frames" constant -- a video whose knowledge rests on three distinct
visuals keeps three, and a redundant screencast that never changes keeps one.
`max_items` exists only as a caller-supplied safety ceiling, not as the
mechanism. See docs/synthesis.md.

Images are written as ordinary files and referenced by relative path. Nothing
here ever base64s an image into the package -- see `Claim`/`VisualEvidence` in
core/contracts.py.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import replace

from core.contracts import Claim, Evidence, KeyPoint, VisualEvidence

# Hamming distance between two 64-bit dHashes below which two frames are
# treated as the same visual. 6/64 bits tolerates JPEG noise and a moving
# cursor while still separating genuinely different screens.
_DUPLICATE_DISTANCE = 6

# dHash encodes gradient STRUCTURE, so every featureless frame hashes to the
# same value -- a blank white slide and a blank black slide would otherwise
# collapse into one. Mean brightness separates them; two frames must match on
# both to count as duplicates. 4/255 is well inside JPEG noise.
_DUPLICATE_BRIGHTNESS = 4.0

# Floor for how close a frame must be to a knowledge unit's timestamp to count
# as illustrating it, when the unit doesn't cite a frame explicitly. The real
# window is derived per-run from actual frame spacing (see `_anchor_tolerance`)
# -- key points sit on the transcript's time grid while frames sit on the
# keyframe grid, so a fixed tolerance would leave the deterministic path with
# no visual evidence at all whenever keyframes are sampled further apart.
_MIN_ANCHOR_TOLERANCE_SEC = 1.0


def _anchor_tolerance(candidates: dict[str, Evidence]) -> float:
    """Half the median gap between candidate frames: the span a sampled frame
    can fairly be said to cover. Self-tuning, so a densely-sampled video stays
    strict and a sparsely-sampled one still gets evidence."""
    times = sorted(ev.timestamp_sec for ev in candidates.values())
    if len(times) < 2:
        return _MIN_ANCHOR_TOLERANCE_SEC
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    median = gaps[len(gaps) // 2]
    return max(_MIN_ANCHOR_TOLERANCE_SEC, median / 2)


def _visual_signature(path: str) -> tuple[int, float] | None:
    """A frame's (64-bit dHash, mean brightness), or None if it can't be read.

    OpenCV is imported lazily and its absence is tolerated: `core/` stays
    importable without any imaging library, and losing the signature degrades
    de-duplication to exact-size matching rather than breaking selection.
    """
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small[:, 1:] > small[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value, float(img.mean())


def _duplicate_of(signature: tuple[int, float] | None, size: int,
                   kept: list[tuple[str, tuple[int, float] | None, int]]) -> str | None:
    """The id of an already-retained frame this one duplicates, or None.

    Returning the id rather than a bool lets a claim whose frame was dropped
    still link to the visually-identical frame that stood in for it -- losing
    the illustration entirely would help nobody, since the same content is in
    the bundle under another id."""
    for kept_id, kept_signature, kept_size in kept:
        if signature is None or kept_signature is None:
            if size == kept_size:  # no signature -- byte-identical files only
                return kept_id
            continue
        if bin(signature[0] ^ kept_signature[0]).count("1") <= _DUPLICATE_DISTANCE \
                and abs(signature[1] - kept_signature[1]) <= _DUPLICATE_BRIGHTNESS:
            return kept_id
    return None


def _image_dimensions(path: str) -> tuple[int, int]:
    try:
        import cv2
    except ImportError:
        return 0, 0
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (int(img.shape[1]), int(img.shape[0])) if img is not None else (0, 0)


# A vision Evidence's timestamp is set from the SAME frame's
# VisionObservation.timestamp_sec it was derived from (core/evidence.py), so
# matching by timestamp is exact, not a fuzzy nearest-in-time guess -- this
# epsilon exists only for float round-tripping, not for genuine ambiguity.
_VISION_FRAME_EPSILON_SEC = 0.01


def _frame_id_at(timestamp_sec: float, candidates: dict[str, Evidence]) -> str | None:
    for evidence_id, ev in candidates.items():
        if abs(ev.timestamp_sec - timestamp_sec) <= _VISION_FRAME_EPSILON_SEC:
            return evidence_id
    return None


def _demand(claims: tuple[Claim, ...], key_points: tuple[KeyPoint, ...],
            candidates: dict[str, Evidence],
            id_by_path: dict[str, str]) -> dict[str, tuple[int, float, str]]:
    """Which candidate frames are actually asked for, by how many knowledge
    units, at what confidence, and why. Explicit citations win; a key point
    with no citation falls back to the nearest frame in time, which is how the
    deterministic (no-synthesizer) path still gets visual evidence."""
    demand: dict[str, list] = {}

    def want(evidence_id: str, confidence: float, reason: str) -> None:
        entry = demand.setdefault(evidence_id, [0, 0.0, reason])
        entry[0] += 1
        entry[1] = max(entry[1], confidence)

    for claim in claims:
        for ev in claim.supporting_evidence:
            if ev.kind == "frame" and ev.ref in id_by_path:
                want(id_by_path[ev.ref], claim.confidence, "cited by a verified semantic claim")
            elif ev.kind == "vision":
                # `vision` Evidence.ref is the model's description text, not a
                # frame path -- the interpreted frame is recovered by
                # timestamp so a genuinely corroborated claim still ships
                # with the image it was corroborated against, not nothing.
                frame_id = _frame_id_at(ev.timestamp_sec, candidates)
                if frame_id is not None:
                    want(frame_id, claim.confidence,
                         "the frame a verified semantic claim's cited visual interpretation came from")

    tolerance = _anchor_tolerance(candidates)
    for point in key_points:
        nearest, best = None, tolerance
        for evidence_id, ev in candidates.items():
            distance = abs(ev.timestamp_sec - point.timestamp_sec)
            if distance <= best:
                nearest, best = evidence_id, distance
        if nearest is not None:
            want(nearest, point.confidence, "nearest frame to a retained key point")

    return {k: (v[0], v[1], v[2]) for k, v in demand.items()}


def select_and_bundle(
    *,
    candidates: dict[str, Evidence],
    claims: tuple[Claim, ...] = (),
    key_points: tuple[KeyPoint, ...] = (),
    output_dir: str,
    stem: str,
    vision_descriptions: dict[float, str] | None = None,
    max_items: int | None = None,
) -> tuple[tuple[VisualEvidence, ...], tuple[Claim, ...]]:
    """Select, de-duplicate and copy evidence frames into
    `<output_dir>/<stem>_evidence/`.

    Returns the retained `VisualEvidence` and the claims rewritten so their
    `visual_evidence` references only frames that actually survived -- a claim
    must never point at an image the bundle doesn't contain.

    Called BEFORE the job workspace is cleaned up: the source frames live in
    that workspace, so evidence must be copied out while it still exists.
    """
    vision_descriptions = vision_descriptions or {}
    # Frames are matched back to their candidate id by path -- the same
    # Evidence.ref the pipeline produced. A claim never carries ids of its own
    # (see core/synthesis.py): `visual_evidence` is only ever written once the
    # image is really in the bundle.
    id_by_path = {ev.ref: evidence_id for evidence_id, ev in candidates.items()}
    demand = _demand(claims, key_points, candidates, id_by_path)

    ranked = sorted(
        demand.items(),
        # most-referenced first, then best-supported, then earliest -- fully
        # deterministic, so the same video always yields the same bundle
        key=lambda kv: (-kv[1][0], -kv[1][1], candidates[kv[0]].timestamp_sec, kv[0]),
    )

    bundle_dir = os.path.join(output_dir, f"{stem}_evidence")
    retained: list[VisualEvidence] = []
    kept: list[tuple[str, tuple[int, float] | None, int]] = []
    superseded: dict[str, str] = {}  # dropped id -> the frame that stood in for it

    for evidence_id, (references, confidence, reason) in ranked:
        if max_items is not None and len(retained) >= max_items:
            break
        source_path = candidates[evidence_id].ref
        if not os.path.exists(source_path) or os.path.getsize(source_path) == 0:
            continue  # frame already gone or never written -- skip, never fabricate
        size = os.path.getsize(source_path)
        signature = _visual_signature(source_path)
        duplicate = _duplicate_of(signature, size, kept)
        if duplicate is not None:
            superseded[evidence_id] = duplicate  # adds no new visual information
            continue
        kept.append((evidence_id, signature, size))

        os.makedirs(bundle_dir, exist_ok=True)
        extension = os.path.splitext(source_path)[1] or ".jpg"
        destination = os.path.join(bundle_dir, f"{evidence_id}{extension}")
        shutil.copy2(source_path, destination)

        timestamp = candidates[evidence_id].timestamp_sec
        width, height = _image_dimensions(destination)
        retained.append(VisualEvidence(
            evidence_id=evidence_id,
            timestamp_sec=timestamp,
            image_path=os.path.join(f"{stem}_evidence", f"{evidence_id}{extension}").replace("\\", "/"),
            width=width, height=height, byte_size=os.path.getsize(destination),
            selection_reason=(f"{reason}; referenced by {references} knowledge unit(s)"),
            description=_nearest_description(timestamp, vision_descriptions),
        ))

    # Now -- and only now -- claims may name their images: every id below is a
    # file that exists in the bundle this call just wrote.
    kept_ids = {v.evidence_id for v in retained}

    def _link(evidence_id: str | None) -> str | None:
        """The bundled image standing for this candidate: itself if retained,
        otherwise the near-identical frame that superseded it."""
        if evidence_id in kept_ids:
            return evidence_id
        stand_in = superseded.get(evidence_id)
        return stand_in if stand_in in kept_ids else None

    def _candidate_id(ev: Evidence) -> str | None:
        if ev.kind == "frame":
            return id_by_path.get(ev.ref)
        if ev.kind == "vision":
            return _frame_id_at(ev.timestamp_sec, candidates)
        return None

    rewritten = []
    for claim in claims:
        linked = tuple(dict.fromkeys(
            link for ev in claim.supporting_evidence
            if (candidate_id := _candidate_id(ev)) is not None
            and (link := _link(candidate_id)) is not None
        ))
        rewritten.append(replace(claim, visual_evidence=linked) if linked else claim)
    return tuple(retained), tuple(rewritten)


def _nearest_description(timestamp: float, descriptions: dict[float, str]) -> str:
    """Vision's own words for this frame, when vision actually ran. Empty
    string when it didn't -- never a placeholder pretending to be a
    description."""
    if not descriptions:
        return ""
    nearest = min(descriptions, key=lambda t: abs(t - timestamp))
    # A description must belong to THIS frame, so the strict floor applies here
    # regardless of how sparsely frames were sampled -- a frame is never
    # labelled with a description of a different moment.
    return (descriptions[nearest][:300]
            if abs(nearest - timestamp) <= _MIN_ANCHOR_TOLERANCE_SEC else "")
