"""Step 11: provider-neutral SEMANTIC knowledge synthesis, with verification.

    AnalysisResult (Step 7 correlated evidence)
        -> SynthesisBrief          (compact, id'd evidence -- build_synthesis_brief)
        -> [ caller's KnowledgeSynthesizer ]        <- the only model-shaped step
        -> raw provider dict
        -> VERIFY against the real evidence         (parse_and_verify)
        -> SynthesisOutput (summary + Claims + honest metadata)

The important half of this module is the second one. Anyone can send evidence
to a model and keep what comes back; this module keeps only what it can still
tie to evidence Video-Lens itself produced. A claim citing an evidence id that
was never in the brief is FABRICATED PROVENANCE and is rejected outright --
not retained with a lowered confidence -- and the rejection is counted in
`SynthesisMetadata` so a consumer can see verification actually ran.

Video-Lens ships no synthesizer and requires none. Without one, nothing here
runs and the deterministic Step 9/10 package is produced unchanged, with
`synthesis.status == "unavailable"` stating so. Deterministic keyword
extraction is never presented as semantic understanding -- see
docs/synthesis.md.
"""
from __future__ import annotations

import json
import re

from core.contracts import (
    AnalysisResult, BriefEvidenceItem, Claim, Evidence, KnowledgeSource,
    SynthesisBrief, SynthesisMetadata, Transcript,
)

# Budgets. These bound what a synthesizer is shown and what may be retained;
# none of them is a knowledge-quality judgment, they exist so a 3-hour video
# can't produce an unbounded brief or an unbounded package.
_TRANSCRIPT_CHUNK_SEC = 20.0  # speech is grouped into ~20s chunks: enough context to
# be meaningful, coarse enough that a long video doesn't become one item per sentence
_MAX_BRIEF_CHARS = 24_000  # total speech text shown to a synthesizer, condensed past this
_MAX_CLAIM_TEXT = 600  # a claim is a knowledge unit, not an essay
_MAX_EVIDENCE_REF = 400  # per stored evidence excerpt -- keeps excerpts as citations,
# never as a back-door full-transcript dump (see docs/synthesis.md, "Storage")
_MAX_CLAIMS = 40  # safety ceiling on retained claims

# A claim whose window overlaps a recorded Step 7 disagreement by this much
# (seconds) is treated as CONFLICTING rather than observed/inferred.
_CONFLICT_WINDOW_SEC = 2.0

# Signatures of media accidentally (or deliberately) embedded in model output.
# The package is a knowledge artifact -- image bytes belong in the evidence
# bundle as files, never inline.
_EMBEDDED_MEDIA = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,|/9j/4[A-Za-z0-9+/]{40,}|iVBORw0KGgo",
                              re.IGNORECASE)

_SYSTEM_PROMPT = """\
You are the knowledge-synthesis component of a general-purpose video-\
intelligence pipeline. You are NOT a transcript summarizer. Your job is to \
decide what a knowledge system should durably REMEMBER about this video, and \
to tie every retained claim to the evidence that supports it.

You are given numbered evidence items drawn from the video: speech (with \
timestamps), visual frame references, visual descriptions, and pointer/cursor \
observations. Each item has an id like "e7". These ids are the ONLY evidence \
that exists. You may not cite any other id.

Work in this order:
1. Understand what the video as a whole is about and what it is teaching or showing.
2. Identify the information that has durable knowledge value -- concepts, \
lessons, facts, procedures, relationships, conclusions, meaningful examples, \
warnings, and important distinctions.
3. For each retained claim, cite the evidence ids that actually support it.
4. Cross-check speech against visual evidence where both exist. Do not assume \
speech proves what is on screen, or that a visual proves what was said.
5. Mark each claim's nature: "observed" if the evidence directly states or \
shows it, "inferred" if you concluded it from the evidence, "unavailable" if \
you believe it matters but the evidence cannot establish it.
6. Reject anything the evidence does not support. Omit it entirely.
7. Keep only genuinely useful knowledge. Sparse and correct beats full and invented.
8. Reference the frame evidence ids that a reader would actually need to SEE \
to understand or verify a claim. Reference frames sparingly -- only where the \
visual genuinely matters. A "frame" item is an uninterpreted image identified \
only by its timestamp: unless a "vision" item describes that moment, nobody \
has looked at it, so reference it as an illustration and never assert what it \
depicts.
9. Preserve timestamps and provenance exactly as given.
10. Produce compact structured output.

Respond with ONLY a single JSON object (no markdown fences, no other text), \
matching exactly this schema:

{
  "provider": "short name of the model/system answering",
  "summary": "a plain-English summary of the WHOLE video: what it is about, \
what it teaches or shows, its major subjects, and its important conclusions",
  "claims": [
    {
      "text": "one self-contained unit of knowledge worth remembering",
      "kind": "lesson | fact | concept | procedure | conclusion | warning | \
example | distinction | relationship",
      "nature": "observed | inferred | unavailable",
      "timestamp_sec": 12.5,
      "timestamp_end_sec": 30.0,
      "evidence_ids": ["e3", "e7"],
      "confidence": 0.8,
      "limitations": ["what this claim's evidence does NOT establish"]
    }
  ]
}

Rules:
- "evidence_ids" must reference ids present in the evidence list. A claim \
citing an unknown id is discarded. A claim with no evidence is discarded \
unless its "nature" is "unavailable".
- "timestamp_sec" must fall inside the video's duration.
- "confidence" (0.0-1.0) reflects how strongly the cited evidence supports the \
claim -- not how fluent the claim sounds.
- "timestamp_end_sec" and "limitations" are optional; omit rather than invent.
- Do not perform domain-specific interpretation beyond what the evidence \
states. Describe the knowledge, do not editorialize.
- Never include image data, base64, or file contents in your response.
"""


# ------------------------------- brief building -------------------------------

def _chunk_transcript(transcript: Transcript, chunk_sec: float) -> list[tuple[float, str]]:
    """Group consecutive speech segments into ~`chunk_sec` blocks. Whole-video
    speech coverage matters here -- a synthesizer asked "what is this video
    about" cannot answer from the handful of keyframe windows Step 7
    correlated -- but one evidence item per sentence would be pure noise."""
    chunks: list[tuple[float, str]] = []
    start: float | None = None
    parts: list[str] = []
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text:
            continue
        if start is None:
            start = seg.start_sec
        parts.append(text)
        if seg.end_sec - start >= chunk_sec:
            chunks.append((start, " ".join(parts)))
            start, parts = None, []
    if parts and start is not None:
        chunks.append((start, " ".join(parts)))
    return chunks


def _fit_budget(chunks: list[tuple[float, str]], max_chars: int) -> tuple[list[tuple[float, str]], bool]:
    """Keep the brief bounded on very long videos. Returns the (possibly
    condensed) chunks and whether condensing happened -- the caller records
    that honestly rather than silently showing a synthesizer less than it
    thinks it has."""
    total = sum(len(t) for _, t in chunks)
    if total <= max_chars or not chunks:
        return chunks, False
    per_chunk = max(120, max_chars // len(chunks))
    return ([(ts, t if len(t) <= per_chunk else t[:per_chunk].rstrip() + " ...[condensed]")
             for ts, t in chunks], True)


def _pointer_contributed(structured_observation) -> bool:
    """Pointer detection is functional but limited (docs/pointer.md), so
    pointer evidence enters the brief only where it ACTUALLY CONTRIBUTED --
    i.e. Step 7 already drew an inference from it. Dumping every raw pointer
    reading in would bury the real evidence in noise and invite a synthesizer
    to over-read a weak signal."""
    return any("pointer" in inf.basis for inf in structured_observation.inferences)


def build_synthesis_brief(result: AnalysisResult, transcript: Transcript | None,
                           source: KnowledgeSource) -> SynthesisBrief:
    """Build the compact, id'd evidence brief a `KnowledgeSynthesizer` is
    given. Operates on already-correlated evidence plus (for whole-video
    coverage) time-chunked speech -- never a raw per-segment transcript dump."""
    items: list[BriefEvidenceItem] = []
    seen: set[tuple[float, str, str]] = set()
    condensed = False

    def add(ev: Evidence) -> None:
        key = (round(ev.timestamp_sec, 3), ev.kind, ev.ref)
        if key in seen:
            return
        seen.add(key)
        items.append(BriefEvidenceItem(evidence_id=f"e{len(items)}", evidence=ev))

    # 1. speech, chunked across the WHOLE video
    if transcript is not None and transcript.status == "ok":
        chunks, condensed = _fit_budget(_chunk_transcript(transcript, _TRANSCRIPT_CHUNK_SEC),
                                         _MAX_BRIEF_CHARS)
        for ts, text in chunks:
            add(Evidence(timestamp_sec=ts, kind="transcript", ref=text))

    # 2. correlated frame / vision / (contributing) pointer evidence
    disagreements: list[str] = []
    for so in result.structured_observations:
        pointer_ok = _pointer_contributed(so)
        for ev in so.observed:
            if ev.kind == "pointer" and not pointer_ok:
                continue
            if ev.kind == "transcript":
                continue  # already covered, at whole-video resolution, above
            add(ev)
        for note in so.disagreements:
            disagreements.append(f"[{so.timestamp_sec:.1f}s] {note}")

    stages_unavailable = sorted({name for so in result.structured_observations
                                 for name in so.unavailable})

    return SynthesisBrief(
        source=source, items=tuple(items), system_prompt=_SYSTEM_PROMPT,
        user_prompt=_render_user_prompt(source, items, disagreements, stages_unavailable, condensed),
        disagreements=tuple(disagreements), stages_unavailable=tuple(stages_unavailable),
    )


def _render_user_prompt(source: KnowledgeSource, items: list[BriefEvidenceItem],
                         disagreements: list[str], stages_unavailable: list[str],
                         condensed: bool) -> str:
    lines = [
        f"VIDEO: {source.title or source.source}",
        f"DURATION: {source.duration_sec:.1f}s",
        "",
        "EVIDENCE:",
    ]
    for item in items:
        ev = item.evidence
        conf = f", confidence {ev.confidence:.2f}" if ev.confidence is not None else ""
        lines.append(f"[{item.evidence_id}] {ev.kind} @ {ev.timestamp_sec:.1f}s{conf}: {ev.ref}")
    if disagreements:
        lines += ["", "EVIDENCE CONFLICTS ALREADY DETECTED (do not resolve these silently -- "
                       "a claim in one of these windows should say so):"]
        lines += [f"- {d}" for d in disagreements]
    if stages_unavailable:
        lines += ["", f"EVIDENCE STREAMS UNAVAILABLE FOR PARTS OF THIS VIDEO: "
                       f"{', '.join(stages_unavailable)}. Absence of a stream is not "
                       f"evidence of absence -- do not claim anything from it."]
    if condensed:
        lines += ["", "NOTE: speech evidence was condensed to fit a size budget; some "
                       "wording is truncated. Do not quote truncated text as verbatim."]
    lines += ["", "Produce the JSON object now."]
    return "\n".join(lines)


# ------------------------------- verification -------------------------------

def _strip_code_fence(text: str) -> str:
    m = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def _coerce_response(raw) -> dict:
    """A provider may hand back an already-parsed dict or the model's raw
    text. Accept both; anything else is a malformed response."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)):
        return json.loads(_strip_code_fence(raw.decode() if isinstance(raw, bytes) else raw))
    raise ValueError(f"expected a JSON object or JSON text, got {type(raw).__name__}")


def _clean_text(value, limit: int) -> str:
    text = str(value).strip()
    if _EMBEDDED_MEDIA.search(text):
        raise ValueError("response contains embedded media/base64 data")
    return text[:limit]


class _Rejected(Exception):
    """One claim failed verification. Never fails the run -- the claim is
    dropped and the reason recorded in SynthesisMetadata."""


def _verify_claim(raw: dict, brief: SynthesisBrief, duration_sec: float,
                   conflict_windows: list[float]) -> Claim:
    """Turn one raw claim dict into a verified `Claim`, or raise `_Rejected`.

    This is the trust boundary. Everything a provider asserts is checked
    against evidence Video-Lens itself produced: ids must resolve, timestamps
    must be real, and confidence is bounded by the quality of the evidence
    actually cited -- a model cannot talk its way to a high score on weak
    evidence."""
    if not isinstance(raw, dict):
        raise _Rejected("claim is not a JSON object")

    text = _clean_text(raw.get("text", ""), _MAX_CLAIM_TEXT)
    if not text:
        raise _Rejected("claim has no text")

    nature = str(raw.get("nature", "")).strip().lower()
    if nature not in ("observed", "inferred", "unavailable"):
        raise _Rejected(f"invalid claim nature {nature!r}")

    # --- provenance: every cited id must be one we actually issued
    index = brief.evidence_by_id()
    cited_ids = [str(i) for i in (raw.get("evidence_ids") or [])]
    unknown = [i for i in cited_ids if i not in index]
    if unknown:
        raise _Rejected(f"cites evidence id(s) that do not exist: {', '.join(sorted(set(unknown))[:5])}")
    cited = [index[i] for i in cited_ids]
    if not cited and nature != "unavailable":
        raise _Rejected("claim cites no evidence")

    # --- timestamps must be real positions in this video
    try:
        timestamp = float(raw.get("timestamp_sec", cited[0].timestamp_sec if cited else 0.0))
    except (TypeError, ValueError):
        raise _Rejected("claim timestamp is not a number")
    if not (0.0 <= timestamp <= duration_sec + 1.0):  # +1.0s: the same clamping tolerance
        raise _Rejected(f"timestamp {timestamp:.2f}s falls outside the source duration "
                         f"{duration_sec:.2f}s")      # frames/pointer/vision already use
    end = raw.get("timestamp_end_sec")
    timestamp_end = None
    if end is not None:
        try:
            timestamp_end = float(end)
        except (TypeError, ValueError):
            timestamp_end = None
        if timestamp_end is not None and not (timestamp <= timestamp_end <= duration_sec + 1.0):
            timestamp_end = None  # a bad range is dropped, not fatal -- the point still stands

    # --- confidence, bounded by the evidence actually cited
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        raise _Rejected("claim confidence is not a number")
    if not (0.0 <= confidence <= 1.0):
        raise _Rejected(f"claim confidence out of range: {confidence}")
    evidence_conf = [e.confidence for e in cited if e.confidence is not None]
    if evidence_conf:
        confidence = min(confidence, max(evidence_conf))

    # --- status: Video-Lens decides "conflicting", never the provider
    limitations = [_clean_text(n, 300) for n in (raw.get("limitations") or [])][:5]
    kinds = {e.kind for e in cited}
    if nature == "unavailable":
        status, confidence = "unavailable", 0.0
    elif any(abs(timestamp - c) <= _CONFLICT_WINDOW_SEC for c in conflict_windows):
        status = "conflicting"
        limitations.append("Evidence streams materially disagree near this timestamp; "
                            "Video-Lens recorded the conflict rather than resolving it.")
    else:
        status = nature

    # --- the visual cross-check
    #
    # A `frame` and a `vision` observation are NOT interchangeable evidence.
    # `vision` is interpreted content -- something actually looked at the image
    # and said what is in it -- and can corroborate a claim. A bare `frame` is
    # an uninterpreted picture: when vision hasn't run, a synthesizer picking a
    # frame id is guessing from a timestamp, having never seen it. Treating
    # that as corroboration would let the package assert a cross-check that
    # never happened, so it is labelled and limited as illustration instead.
    has_speech = "transcript" in kinds
    has_interpreted_visual = "vision" in kinds
    has_raw_frame = "frame" in kinds
    uninterpreted_note = ("The referenced frame was never visually analyzed, so it "
                           "illustrates this claim but does not corroborate it.")

    if has_speech and has_interpreted_visual:
        verification = "speech_corroborated_by_visual_evidence"
    elif has_speech and has_raw_frame:
        verification = "speech_evidence_with_uninterpreted_frame"
        limitations.append(uninterpreted_note)
    elif has_speech:
        verification = "speech_evidence_only"
        limitations.append("No visual evidence was cited for this claim -- it rests on speech alone.")
    elif has_interpreted_visual:
        verification = "visual_evidence_only"
        limitations.append("No speech evidence was cited for this claim -- it rests on visuals alone.")
    elif has_raw_frame:
        verification = "uninterpreted_frame_only"
        limitations.append(uninterpreted_note)
    elif "pointer" in kinds:
        verification = "pointer_evidence_only"
    else:
        verification = "unverifiable_no_evidence_cited"

    return Claim(
        text=text,
        kind=_clean_text(raw.get("kind", "fact"), 40) or "fact",
        status=status,
        timestamp_sec=timestamp,
        timestamp_end_sec=timestamp_end,
        supporting_evidence=tuple(
            Evidence(timestamp_sec=e.timestamp_sec, kind=e.kind,
                     ref=e.ref[:_MAX_EVIDENCE_REF], confidence=e.confidence)
            for e in cited
        ),
        confidence=confidence,
        verification=verification,
        # Deliberately left empty here. `visual_evidence` may only ever name
        # images the package actually ships, so it is populated by
        # core/visual_evidence.py AFTER the bundle is written -- never
        # provisionally. Which frames this claim wants is already recorded in
        # `supporting_evidence` (kind="frame"), which is what selection reads.
        limitations=tuple(dict.fromkeys(limitations)),
    )


class SynthesisOutput:
    """What survived verification: a whole-video summary, verified claims, and
    metadata that records what did NOT survive."""

    __slots__ = ("summary", "claims", "metadata")

    def __init__(self, summary: str | None, claims: tuple[Claim, ...],
                 metadata: SynthesisMetadata):
        self.summary = summary
        self.claims = claims
        self.metadata = metadata


def unavailable(detail: str) -> SynthesisOutput:
    """The honest no-synthesizer result -- no summary, no claims, and a
    metadata record saying plainly that semantic synthesis did not run."""
    return SynthesisOutput(None, (), SynthesisMetadata(status="unavailable", detail=detail))


def parse_and_verify(raw_response, brief: SynthesisBrief) -> SynthesisOutput:
    """Validate and cross-check a provider's raw response. Never raises for
    bad model output -- a malformed response yields status="failed" and an
    empty result, because a broken synthesizer must not fail a job whose
    deterministic knowledge is already valid."""
    try:
        data = _coerce_response(raw_response)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return SynthesisOutput(None, (), SynthesisMetadata(
            status="failed", detail=f"malformed provider response: {e}"))

    provider = str(data.get("provider", ""))[:80]
    duration = brief.source.duration_sec
    conflict_windows = [float(m.group(1)) for d in brief.disagreements
                        if (m := re.match(r"\[(\d+(?:\.\d+)?)s\]", d))]

    raw_claims = data.get("claims") or []
    if not isinstance(raw_claims, list):
        return SynthesisOutput(None, (), SynthesisMetadata(
            status="failed", provider=provider, detail="'claims' is not a list"))

    claims: list[Claim] = []
    reasons: list[str] = []
    for raw in raw_claims[:_MAX_CLAIMS * 4]:  # bound the work, not just the output
        try:
            claims.append(_verify_claim(raw, brief, duration, conflict_windows))
        except _Rejected as e:
            reasons.append(str(e))
        except (ValueError, TypeError) as e:
            reasons.append(f"unusable claim: {e}")
        if len(claims) >= _MAX_CLAIMS:
            break

    try:
        summary = _clean_text(data.get("summary", ""), 4000) or None
    except ValueError as e:
        summary, reasons = None, reasons + [f"summary rejected: {e}"]

    returned = len(raw_claims)
    rejected = returned - len(claims)
    if not claims and not summary:
        status = "rejected" if returned else "failed"
    else:
        status = "ok"

    return SynthesisOutput(summary, tuple(claims), SynthesisMetadata(
        status=status, provider=provider, claims_returned=returned,
        claims_retained=len(claims), claims_rejected=max(0, rejected),
        rejection_reasons=tuple(dict.fromkeys(reasons))[:10],
        detail="" if status == "ok" else "no claim or summary survived verification",
    ))


def synthesize(provider, brief: SynthesisBrief) -> SynthesisOutput:
    """Call a `KnowledgeSynthesizer` and verify what it returns. A provider
    that raises is recorded as "failed" and never propagates -- semantic
    synthesis is an enhancement, and its absence must never destroy a job
    whose deterministic knowledge is already valid."""
    try:
        raw = provider.synthesize(brief)
    except Exception as e:  # any provider, any failure mode -- see docstring
        return SynthesisOutput(None, (), SynthesisMetadata(
            status="failed", detail=f"synthesizer raised {type(e).__name__}: {e}"))
    return parse_and_verify(raw, brief)
