"""Deterministic evidence retrieval -- by time, kind and source. No ranking, no
embeddings, no semantic matching, no model call: every result is there because
its timestamps satisfy an explicit rule, and the same input always yields the
same output in the same order. See docs/session.md.

Two operations, over any iterable of `Evidence` (or an `EvidenceSession`):

    evidence_near(evidence, t, tolerance_sec)   -- around a moment
    evidence_between(evidence, start, end)      -- inside a window

SEMANTICS (all boundaries inclusive):
  * point evidence (`timestamp_end_sec is None`)
      - `near`: |timestamp - t| <= tolerance -- literally
        `core.evidence._within_tolerance`, the one rule the correlation layer
        itself uses, so retrieval and correlation never disagree.
      - `between`: start <= timestamp <= end.
  * span evidence (`timestamp_end_sec` set, e.g. cursor_track, transcript)
      - matches when its interval OVERLAPS the window (the window being
        [t - tolerance, t + tolerance] for `near`), touching counts. A span is
        never reduced to its start point: a segment running 17-18s is returned
        for t=17.5 even though it starts 0.5s away.
  * `kinds` / `sources` filter by exact membership; `None` means no filter.
    A source of "" matches evidence with no provenance label (including
    per-frame vision), distinct from "video_understanding".
  * results are ordered by (start, end, kind, source, ref, confidence), exact
    duplicates are returned once, and "nothing matched" is an empty tuple.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

from core.contracts import EVIDENCE_KINDS, Evidence
from core.evidence import _within_tolerance
from core.session import EvidenceSession, canonical_evidence


def _items(evidence) -> list[Evidence]:
    return list(evidence.evidence if isinstance(evidence, EvidenceSession) else evidence)


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _filters(kinds, sources) -> tuple[frozenset | None, frozenset | None]:
    def as_set(values, name):
        if values is None:
            return None
        values = frozenset([values] if isinstance(values, str) else values)
        if not values:
            raise ValueError(f"{name} must not be empty (use None for no filter)")
        return values

    kind_set, source_set = as_set(kinds, "kinds"), as_set(sources, "sources")
    unknown = sorted(kind_set - set(EVIDENCE_KINDS)) if kind_set else []
    if unknown:
        raise ValueError(f"unknown evidence kind(s): {', '.join(map(repr, unknown))}")
    return kind_set, source_set


def _select(items, kind_set, source_set):
    return [e for e in items if (kind_set is None or e.kind in kind_set)
            and (source_set is None or e.source in source_set)]


def evidence_near(evidence: Iterable[Evidence] | EvidenceSession, timestamp_sec: float,
                  tolerance_sec: float = 1.0, *, kinds=None, sources=None) -> tuple[Evidence, ...]:
    """Evidence at or around `timestamp_sec`, within `tolerance_sec` (>= 0; 0 is
    "exactly at this moment", and the default matches the pipeline's own)."""
    t = _finite(timestamp_sec, "timestamp_sec", minimum=0.0)
    tol = _finite(tolerance_sec, "tolerance_sec", minimum=0.0)
    items = _select(_items(evidence), *_filters(kinds, sources))
    points = _within_tolerance([e for e in items if e.timestamp_end_sec is None], t, tol)
    lo, hi = t - tol, t + tol
    spans = [e for e in items if e.timestamp_end_sec is not None
             and e.timestamp_sec <= hi and e.timestamp_end_sec >= lo]
    return canonical_evidence(points + spans)


def evidence_between(evidence: Iterable[Evidence] | EvidenceSession, start_sec: float, end_sec: float,
                     *, kinds=None, sources=None) -> tuple[Evidence, ...]:
    """Evidence inside the closed window `[start_sec, end_sec]` (points inside it,
    spans overlapping it)."""
    lo = _finite(start_sec, "start_sec", minimum=0.0)
    hi = _finite(end_sec, "end_sec", minimum=0.0)
    if hi < lo:
        raise ValueError(f"window ends before it starts: {lo} > {hi}")
    items = _select(_items(evidence), *_filters(kinds, sources))
    return canonical_evidence(
        e for e in items
        if (lo <= e.timestamp_sec <= hi if e.timestamp_end_sec is None
            else e.timestamp_sec <= hi and e.timestamp_end_sec >= lo))
