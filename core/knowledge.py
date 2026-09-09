"""Deterministic, domain-neutral knowledge extraction: turns an already-built
`AnalysisResult` (Step 7's evidence-correlated pipeline output) into a
compact, durable `KnowledgePackage` -- the thing that survives after
temporary artifacts (video, frames, caches) are deleted. See docs/lifecycle.md.

No LLM call is made or required here -- every KeyPoint is a cue-phrase match
against real transcript text, cited back to the Evidence it came from.
This is intentionally not full semantic understanding (see Limitations in
the built package and docs/lifecycle.md); if a smarter summarizer is ever
wanted, `build_knowledge_package` is the one seam to swap it in behind --
nothing else in this module or its callers needs to change.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

from core.contracts import (
    KNOWLEDGE_SCHEMA_VERSION, AnalysisResult, Evidence, KeyPoint, KnowledgePackage,
    KnowledgeSource, ProcessingMetadata,
)
from core.errors import KnowledgePackageError

# Mirrors core/synthesis.py's own per-excerpt cap. Enforced here as a package
# invariant too: excerpts are citations, and a package must never become a
# back-door full-transcript dump no matter which code path built it.
_MAX_EVIDENCE_REF = 400
_EMBEDDED_MEDIA = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,|/9j/4[A-Za-z0-9+/]{40,}|iVBORw0KGgo",
                              re.IGNORECASE)

VERSION = "1.0.0"  # software release -- see KNOWLEDGE_SCHEMA_VERSION in core/contracts.py
                    # for the independently-versioned package *shape*

_ALL_STAGES = ("transcript", "frame", "pointer", "vision")

# (kind, cue phrases) -- checked in order, first match wins per segment.
# Free text, no domain vocabulary (see docs/architecture.md's "What was
# deliberately excluded" -- no trading/UI-framework-specific terms here
# either, on the same principle).
_CUE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("warning", ("warning", "be careful", "watch out", "don't ", "do not ", "never ", "avoid ")),
    ("procedure", ("step ", "first, ", "then, ", "next, ", "you need to", "make sure to", "how to ")),
    ("definition", ("is defined as", "means that", "refers to", "in other words", "is called")),
    ("example", ("for example", "for instance", "such as", "like this")),
    ("conclusion", ("in summary", "to summarize", "the key takeaway", "in conclusion", "the point is")),
    ("explanation", ("because ", "this is why", "the reason", "so that ")),
)

_MAX_KEY_LESSONS = 15
_MAX_TOPICS = 8
_MIN_TOPIC_LEN = 4
_MIN_TOPIC_COUNT = 2

_STOPWORDS = frozenset("""
the a an is are was were be been being to of and or but in on at for with
by from as this that these those it its it's you your i we they he she
what when where why how not no do does did can could will would should
just so if then than very really about into over under again more most
""".split())


def _cue_kind(text: str) -> str | None:
    lowered = text.lower()
    for kind, phrases in _CUE_PATTERNS:
        if any(p in lowered for p in phrases):
            return kind
    return None


def _extract_key_lessons(transcript) -> list[KeyPoint]:
    if transcript is None or transcript.status != "ok":
        return []
    points: list[KeyPoint] = []
    seen_text = set()
    for seg in transcript.segments:
        text = seg.text.strip()
        if not text or text in seen_text:
            continue
        kind = _cue_kind(text)
        if kind is None:
            continue
        seen_text.add(text)
        points.append(KeyPoint(
            text=text, kind=kind, timestamp_sec=seg.start_sec,
            supporting_evidence=(Evidence(timestamp_sec=seg.start_sec, kind="transcript",
                                           ref=text, confidence=seg.confidence),),
            # Heuristic keyword match, not a model judgment -- moderate,
            # constant confidence rather than a fabricated precise score.
            confidence=0.4,
            # the point IS the cited transcript text (a quote), not a
            # conclusion drawn from it -- see KeyPoint.nature.
            nature="observed",
        ))
        if len(points) >= _MAX_KEY_LESSONS:
            break
    return points


def _extract_topics(transcript) -> tuple[str, ...]:
    if transcript is None or transcript.status != "ok":
        return ()
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", " ".join(s.text for s in transcript.segments))
    counts = Counter(w.lower() for w in words
                      if len(w) >= _MIN_TOPIC_LEN and w.lower() not in _STOPWORDS)
    ranked = [w for w, c in counts.most_common(_MAX_TOPICS * 2) if c >= _MIN_TOPIC_COUNT]
    return tuple(w.capitalize() for w in ranked[:_MAX_TOPICS])


def _extract_important_observations(structured_observations) -> list[KeyPoint]:
    """Video-Lens's own Step 7 inferences are already cited conclusions --
    reuse them directly rather than re-deriving anything."""
    points: list[KeyPoint] = []
    seen_text = set()
    for so in structured_observations:
        for inf in so.inferences:
            if inf.text in seen_text:
                continue
            seen_text.add(inf.text)
            points.append(KeyPoint(
                text=inf.text, kind="observation", timestamp_sec=so.timestamp_sec,
                supporting_evidence=inf.supporting_evidence, confidence=inf.confidence,
                # a Step 7 Inference is, by definition, a conclusion DRAWN
                # FROM evidence, never the evidence itself -- see KeyPoint.nature.
                nature="inferred",
            ))
    return points


def _stage_coverage(structured_observations) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = set()
    unavailable = set(_ALL_STAGES)
    for so in structured_observations:
        for e in so.observed:
            available.add(e.kind)
            unavailable.discard(e.kind)
    return tuple(sorted(available)), tuple(sorted(unavailable))


def _synthesis_limitations(synthesis) -> list[str]:
    """State plainly what the semantic layer did or didn't do. The whole point
    of Step 11's fallback contract is that a consumer can tell semantic
    understanding from keyword matching by reading the package -- never by
    guessing from how fluent the text looks."""
    if synthesis is None or synthesis.metadata.status == "unavailable":
        detail = getattr(synthesis.metadata, "detail", "") if synthesis is not None else ""
        return ["No semantic knowledge synthesis was performed" + (f" ({detail})" if detail else "")
                + " -- this package contains deterministic extraction only. `semantic_summary` "
                  "and `claims` are empty; nothing here is a model's understanding of the video."]
    status = synthesis.metadata.status
    if status == "failed":
        return [f"A semantic synthesizer was supplied but did not produce usable output "
                f"({synthesis.metadata.detail}) -- no semantic knowledge is included, and the "
                f"deterministic extraction below was NOT substituted for it."]
    if status == "rejected":
        return [f"A semantic synthesizer answered, but none of its "
                f"{synthesis.metadata.claims_returned} claim(s) could be verified against the "
                f"evidence and all were rejected -- no semantic knowledge is included."]
    notes = []
    if synthesis.metadata.claims_rejected:
        notes.append(f"{synthesis.metadata.claims_rejected} of "
                      f"{synthesis.metadata.claims_returned} claim(s) returned by the semantic "
                      f"synthesizer failed evidence verification and were discarded.")
    if not synthesis.summary:
        notes.append("The semantic synthesizer returned no whole-video summary.")
    return notes


def _limitations(transcript, structured_observations, stages_unavailable: tuple[str, ...],
                  synthesis=None) -> tuple[str, ...]:
    notes = [
        "Topics, key lessons and `summary` are extracted via deterministic keyword/cue-phrase "
        "matching against the transcript, not semantic language understanding -- "
        "they may miss content that doesn't use a recognized cue phrase, or flag "
        "something as notable that isn't. See docs/lifecycle.md.",
    ]
    notes += _synthesis_limitations(synthesis)
    if transcript is None or transcript.status != "ok":
        notes.append(f"No usable speech was available (transcript status: "
                      f"{transcript.status if transcript is not None else 'unavailable'}) -- "
                      f"topics and key lessons could not be extracted.")
    if "vision" in stages_unavailable:
        notes.append("Vision analysis was not available for this run -- visual content "
                      "beyond raw frame timestamps was not analyzed.")
    if "pointer" in stages_unavailable:
        notes.append("No pointer/cursor evidence was found -- this may mean no cursor was "
                      "on screen, or a stationary cursor produced no motion signal (see "
                      "docs/pointer.md); absence of pointer evidence is not proof of absence.")
    return tuple(notes)


def _summary(source: KnowledgeSource, key_lessons: list[KeyPoint], topics: tuple[str, ...],
             transcript) -> str:
    name = source.title or source.source
    base = f"{name}: a {source.duration_sec:.0f}s {source.source_type} video."
    if transcript is None or transcript.status != "ok":
        return base + " No speech was available to analyze."
    topic_str = f" Recurring topics: {', '.join(topics)}." if topics else ""
    return (base + f" {len(key_lessons)} notable point(s) identified via deterministic "
            f"transcript analysis.{topic_str}")


def knowledge_source_for(video) -> KnowledgeSource:
    """Provenance for one `VideoInput`. Shared so the synthesis brief and the
    finished package describe the source identically -- they must never
    disagree about what video this knowledge came from."""
    return KnowledgeSource(source_type=video.source_type, source=video.source or video.path,
                            duration_sec=video.duration_sec, title=video.title)


def build_knowledge_package(result: AnalysisResult, transcript=None,
                             synthesis=None) -> KnowledgePackage:
    """Build the compact, durable KnowledgePackage for a completed
    AnalysisResult. `transcript` is optional -- pass it when the caller
    still has it (AnalysisResult itself doesn't carry the raw Transcript,
    only the Evidence derived from it); without it, topics/key lessons
    degrade to empty and that's recorded honestly in `limitations`.

    `synthesis` is an optional `core.synthesis.SynthesisOutput` -- already
    verified against the evidence by the time it gets here. Omitting it (the
    default) produces exactly the deterministic Step 9/10 package, with
    `limitations` stating that no semantic synthesis ran. `summary` stays
    deterministic in BOTH cases; a synthesizer's whole-video summary lands in
    `semantic_summary`, never overwriting the deterministic one, so a consumer
    can always tell the two apart. See docs/synthesis.md."""
    source = knowledge_source_for(result.video)

    key_lessons = _extract_key_lessons(transcript)
    topics = _extract_topics(transcript)
    important_observations = _extract_important_observations(result.structured_observations)
    stages_available, stages_unavailable = _stage_coverage(result.structured_observations)

    claims = synthesis.claims if synthesis is not None else ()

    cited_evidence: list[Evidence] = []
    seen = set()
    for kp in (*key_lessons, *important_observations, *claims):
        for e in kp.supporting_evidence:
            key = (e.timestamp_sec, e.kind, e.ref)
            if key not in seen:
                seen.add(key)
                cited_evidence.append(e)

    frame_count = len({e.ref for so in result.structured_observations
                        for e in so.observed if e.kind == "frame"})

    return KnowledgePackage(
        source=source,
        summary=_summary(source, key_lessons, topics, transcript),
        topics=topics,
        key_lessons=tuple(key_lessons),
        important_observations=tuple(important_observations),
        evidence=tuple(cited_evidence),
        limitations=_limitations(transcript, result.structured_observations, stages_unavailable,
                                  synthesis),
        semantic_summary=synthesis.summary if synthesis is not None else None,
        claims=claims,
        synthesis=synthesis.metadata if synthesis is not None else None,
        processing=ProcessingMetadata(
            generated_at=datetime.now(timezone.utc).isoformat(),
            video_lens_version=VERSION,
            knowledge_schema_version=KNOWLEDGE_SCHEMA_VERSION,
            frame_count=frame_count,
            transcript_status=transcript.status if transcript is not None else "unavailable",
            stages_available=stages_available,
            stages_unavailable=stages_unavailable,
        ),
    )


def validate_knowledge_package(package: KnowledgePackage) -> None:
    """Structural + honesty validation before a package is treated as real,
    handoff-ready output. Raises `KnowledgePackageError` on the first
    violation found. This is a sanity/honesty check, not a semantic-quality
    judgment -- a package with zero key lessons (e.g. a silent video) is
    perfectly valid as long as it says so in `limitations`; only genuinely
    broken or dishonest packages are rejected here. Most per-field
    constraints (confidence bounds, non-empty evidence, valid Evidence.kind)
    are already enforced by the dataclasses' own `__post_init__` at
    construction time (see core/contracts.py) and are not re-checked --
    this function only checks what construction alone cannot: cross-field
    consistency and the fields with no natural per-field invariant."""
    source = package.source
    if not source.source or not source.source.strip():
        raise KnowledgePackageError("knowledge package has no source reference")
    if source.duration_sec <= 0:
        raise KnowledgePackageError(f"invalid duration in knowledge package: {source.duration_sec}")
    if not package.summary or not package.summary.strip():
        raise KnowledgePackageError("knowledge package summary is empty")

    processing = package.processing
    if not processing.generated_at:
        raise KnowledgePackageError("knowledge package is missing processing metadata (generated_at)")
    if not processing.knowledge_schema_version:
        raise KnowledgePackageError("knowledge package is missing knowledge_schema_version")

    # Timestamps must fall within the source video's own duration -- a
    # KeyPoint timestamped past the end of the video would misdirect anyone
    # who follows it back to the source.
    for kp in (*package.key_lessons, *package.important_observations, *package.claims):
        if not (0.0 <= kp.timestamp_sec <= source.duration_sec + 1.0):  # +1.0s: same clamping
            # tolerance frames/pointer/vision already use (see docs/frames.md)
            raise KnowledgePackageError(
                f"{type(kp).__name__} timestamp {kp.timestamp_sec} falls outside source duration "
                f"{source.duration_sec}")

    # --- Step 11: the semantic layer must not smuggle in bulk or dangling refs.
    known_visuals = {v.evidence_id for v in package.visual_evidence}
    for claim in package.claims:
        dangling = [i for i in claim.visual_evidence if i not in known_visuals]
        if dangling:
            raise KnowledgePackageError(
                f"claim references visual evidence not present in the package: "
                f"{', '.join(sorted(dangling))}")

    for visual in package.visual_evidence:
        if os.path.isabs(visual.image_path) or visual.image_path.startswith("..") \
                or ":" in visual.image_path:
            # An absolute path would leak this machine's layout into the durable
            # artifact and break the moment the bundle is moved or handed off.
            raise KnowledgePackageError(
                f"visual evidence image_path must be relative to the package, got "
                f"{visual.image_path!r}")
        if not (0.0 <= visual.timestamp_sec <= source.duration_sec + 1.0):
            raise KnowledgePackageError(
                f"visual evidence timestamp {visual.timestamp_sec} falls outside source "
                f"duration {source.duration_sec}")

    # A package is a knowledge artifact: excerpts cite, they don't archive, and
    # image bytes live in the bundle as files rather than inline.
    for ev in package.evidence:
        if ev.kind != "frame" and len(ev.ref) > _MAX_EVIDENCE_REF:
            raise KnowledgePackageError(
                f"evidence excerpt of {len(ev.ref)} chars exceeds the {_MAX_EVIDENCE_REF}-char "
                f"limit -- a package must not become a transcript dump")
    if _EMBEDDED_MEDIA.search(package.semantic_summary or "") or \
            any(_EMBEDDED_MEDIA.search(c.text) for c in package.claims):
        raise KnowledgePackageError("package contains embedded media/base64 data -- visual "
                                     "evidence belongs in the bundle as files, never inline")

    # Limitations must accurately describe unavailable evidence streams: a
    # stage listed as unavailable must never also be the kind of evidence a
    # key lesson/observation actually cites -- that would be a contradiction
    # (the package claiming a stream was empty while also citing it).
    cited_kinds = {e.kind for kp in (*package.key_lessons, *package.important_observations,
                                     *package.claims)
                   for e in kp.supporting_evidence}
    contradictions = cited_kinds & set(processing.stages_unavailable)
    if contradictions:
        raise KnowledgePackageError(
            f"processing.stages_unavailable claims {sorted(contradictions)} unavailable, "
            f"but evidence of that kind is cited in the package")


def export_knowledge_package(package: KnowledgePackage, output_dir: str,
                              filename_stem: str | None = None) -> str:
    """The generic handoff/export mechanism: validates `package`, then
    writes it to `output_dir` as a single small JSON file, atomically (write
    to a `.tmp` path, then `os.replace()` into place, so a reader never sees
    a partial file and a crash mid-write can't corrupt the previous output).
    Returns the final path. Raises `KnowledgePackageError` -- and writes
    nothing -- if `package` fails validation; a caller (e.g.
    `video_lens.process_video`) should treat that as export failure, not
    success (see docs/lifecycle.md, "Handoff/export safety"). This function
    has no knowledge of any particular consumer -- it
    just produces a portable JSON file any consumer can read."""
    validate_knowledge_package(package)
    os.makedirs(output_dir, exist_ok=True)
    stem = filename_stem or default_filename_stem(package)
    out_path = os.path.join(output_dir, f"{stem}.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(to_json(package))
    os.replace(tmp_path, out_path)
    return out_path


def default_filename_stem(package: KnowledgePackage) -> str:
    """The package's output basename, derived from its own title/source. Public
    because the Step 11 evidence bundle must sit beside the JSON under the same
    stem -- `process_video` computes it once and passes it to both."""
    name = package.source.title or os.path.splitext(os.path.basename(package.source.source))[0]
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "video"
    return safe[:80]


def to_dict(package: KnowledgePackage) -> dict:
    """Plain-dict form -- the standard handoff contract (docs/lifecycle.md).
    stdlib `dataclasses.asdict` already recurses through nested dataclasses
    and tuples, so this needs no hand-written conversion."""
    return asdict(package)


def to_json(package: KnowledgePackage) -> str:
    return json.dumps(to_dict(package), indent=2)


def render_markdown(package: KnowledgePackage) -> str:
    """A presentation rendering of `package` -- derived, not canonical.
    `to_dict`/`to_json` remain the source of truth (see Part 4)."""
    s = package.source
    lines = [
        f"# {s.title or s.source}",
        "",
        f"- **Source**: {s.source_type} -- `{s.source}`",
        f"- **Duration**: {s.duration_sec:.1f}s",
        f"- **Generated**: {package.processing.generated_at}",
        "",
        "## Summary",
        "",
        package.summary,
    ]
    if package.semantic_summary:
        # Labelled distinctly from the deterministic summary above -- a reader
        # must never have to guess which one a model wrote.
        lines += ["", "## Whole-video summary (semantic synthesis)", "",
                  package.semantic_summary]
    if package.claims:
        lines += ["", "## Verified claims", ""]
        for c in package.claims:
            span = f"{c.timestamp_sec:.1f}s" if c.timestamp_end_sec is None \
                else f"{c.timestamp_sec:.1f}-{c.timestamp_end_sec:.1f}s"
            visuals = f" [visual: {', '.join(c.visual_evidence)}]" if c.visual_evidence else ""
            lines.append(f"- **[{c.kind}/{c.status}]** ({span}, confidence {c.confidence:.2f}) "
                          f"{c.text}{visuals}")
    if package.visual_evidence:
        lines += ["", "## Visual evidence", ""]
        for v in package.visual_evidence:
            lines.append(f"- `{v.evidence_id}` ({v.timestamp_sec:.1f}s) `{v.image_path}` "
                          f"-- {v.selection_reason}")
    if package.topics:
        lines += ["", "## Topics", "", ", ".join(package.topics)]
    if package.key_lessons:
        lines += ["", "## Key lessons", ""]
        for kp in package.key_lessons:
            lines.append(f"- **[{kp.kind}]** ({kp.timestamp_sec:.1f}s) {kp.text}")
    if package.important_observations:
        lines += ["", "## Important observations", ""]
        for kp in package.important_observations:
            lines.append(f"- ({kp.timestamp_sec:.1f}s, confidence {kp.confidence:.2f}) {kp.text}")
    if package.limitations:
        lines += ["", "## Limitations", ""]
        for note in package.limitations:
            lines.append(f"- {note}")
    lines += ["", "## Processing", "",
              f"- Evidence streams available: {', '.join(package.processing.stages_available) or 'none'}",
              f"- Evidence streams unavailable: {', '.join(package.processing.stages_unavailable) or 'none'}",
              f"- Frames referenced: {package.processing.frame_count}",
              f"- Transcript status: {package.processing.transcript_status}",
              f"- Semantic synthesis: "
              f"{package.synthesis.status if package.synthesis else 'unavailable'}"
              + (f" (provider: {package.synthesis.provider}, "
                 f"{package.synthesis.claims_retained} retained / "
                 f"{package.synthesis.claims_rejected} rejected)"
                 if package.synthesis and package.synthesis.status == "ok" else ""),
              f"- Video-Lens version: {package.processing.video_lens_version}"]
    return "\n".join(lines) + "\n"
