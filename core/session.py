"""Evidence sessions: a durable, versioned record of the OBSERVATIONS one
analysis produced -- not conversational memory, not conclusions. See
docs/session.md.

    AnalysisResult + Transcript  ->  EvidenceSession  ->  <stem>.session.json
                                                              |
                                              load_session -> core.retrieval

A session keeps what the evidence layer observed (transcript, the six
`Evidence` kinds with their timestamps, spans and provenance, which streams were
unavailable per window, and the facts about how the run was configured that are
needed to read it). It deliberately does NOT keep: the video, extracted frames,
Video-Lens's own inferences, summaries, claims, or anything about a user or
conversation. Persisting is opt-in (`PipelineConfig.session_dir`); nothing here
runs in a default pipeline.

Its format version (`SESSION_FORMAT_VERSION`) is independent of
`KNOWLEDGE_SCHEMA_VERSION`: the two describe different files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from core.contracts import (
    EVIDENCE_KINDS, AnalysisResult, Evidence, KnowledgeSource, Transcript, TranscriptSegment, Word,
)
from core.errors import SessionError
from core.knowledge import VERSION, knowledge_source_for

SESSION_FORMAT_VERSION = 1
SESSION_SUFFIX = ".session.json"
_MAX_SESSION_BYTES = 64 * 1024 * 1024  # a session is a few hundred KB; this only stops a hostile/corrupt file

# Query parameters that identify a video rather than authorize access to it.
# Everything else (tokens, signatures, expiries) is dropped from a persisted URL.
_IDENTIFYING_QUERY = ("v", "list", "id")


@dataclass(frozen=True)
class SessionVideo:
    """Facts about the source video's media, minus its (temporary) local path."""
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass(frozen=True)
class SessionWindow:
    """One correlation window the run evaluated, and which evidence streams had
    nothing in it -- so "unavailable" is never mistaken for "not checked"."""
    timestamp_sec: float
    tolerance_sec: float
    unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceSession:
    format_version: int
    created_at: str  # ISO 8601 UTC
    video_lens_version: str
    source: KnowledgeSource  # URL credentials/tokens removed -- see _public_source
    video: SessionVideo
    evidence: tuple[Evidence, ...]  # canonical order, no exact duplicates -- see _canonical
    analysis: dict = field(default_factory=dict)  # bool/number/str facts about the run's configuration
    transcript: Transcript | None = None
    windows: tuple[SessionWindow, ...] = ()


# ------------------------------- building -------------------------------

def _order_key(e: Evidence) -> tuple:
    """The one ordering for evidence, shared with core.retrieval: by start, end
    (a point sorts as its own end), kind, source, ref, confidence. Pure data --
    never identity, hash or input order."""
    end = e.timestamp_end_sec if e.timestamp_end_sec is not None else e.timestamp_sec
    return (e.timestamp_sec, end, e.kind, e.source, e.ref, -1.0 if e.confidence is None else e.confidence)


def canonical_evidence(items) -> tuple[Evidence, ...]:
    """Sorted by `_order_key`, exact duplicates (equal in every field) removed."""
    return tuple(dict.fromkeys(sorted(items, key=_order_key)))


def _public_source(source: KnowledgeSource) -> KnowledgeSource:
    """The source as it may be persisted: for a URL, no credentials, fragment or
    access tokens -- only the video-identifying query parameters. Signed URLs are
    never written to disk. (A local path is kept: it is what the package carries.)"""
    if source.source_type != "url":
        return source
    parts = urlsplit(source.source)
    host = parts.hostname or ""
    netloc = host + (f":{parts.port}" if parts.port else "") if host else parts.netloc.rpartition("@")[2]
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k in _IDENTIFYING_QUERY])
    clean = urlunsplit((parts.scheme, netloc, parts.path, query, ""))
    return KnowledgeSource(source_type=source.source_type, source=clean,
                           duration_sec=source.duration_sec, title=source.title)


def _frame_marker(ref: str) -> str:
    """A frame's evidence ref is a temporary file path that will not exist when
    the session is read, and would leak the machine's layout. Keep the file name
    only: a marker that a frame was sampled, not a resolvable location."""
    return re.split(r"[\\/]", ref)[-1]


def build_session(result: AnalysisResult, transcript: Transcript | None = None, *,
                  analysis: dict | None = None, created_at: str | None = None) -> EvidenceSession:
    """The session for a finished analysis. Persists observations only:
    `result.structured_observations[*].inferences`, `.disagreements`, `.summary`
    and the per-stream raw objects (frames, `VisionObservation` detail beyond
    its evidence, undetected pointer events) are intentionally not kept.

    Transcript evidence comes from the Transcript itself, whole-video and as
    spans, rather than from the few windows correlation happened to visit -- so
    the same speech is never stored twice. When no transcript is supplied, any
    transcript evidence in `result` is kept as it is."""
    items = []
    for so in result.structured_observations:
        for e in so.observed:
            if e.kind == "transcript" and transcript is not None:
                continue
            if e.kind == "frame":
                e = Evidence(timestamp_sec=e.timestamp_sec, kind="frame", ref=_frame_marker(e.ref),
                             confidence=e.confidence, timestamp_end_sec=e.timestamp_end_sec, source=e.source)
            items.append(e)
    items.extend(result.visual_changes)  # the whole measurement stream; window copies dedupe below
    if transcript is not None:
        items.extend(Evidence(timestamp_sec=s.start_sec, timestamp_end_sec=s.end_sec, kind="transcript",
                              ref=s.text, confidence=s.confidence)
                     for s in transcript.segments if s.text.strip())

    v = result.video
    return EvidenceSession(
        format_version=SESSION_FORMAT_VERSION,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        video_lens_version=VERSION,
        source=_public_source(knowledge_source_for(v)),
        video=SessionVideo(width=v.width, height=v.height, fps=v.fps, has_audio=v.has_audio),
        analysis=dict(sorted((analysis or {}).items())),
        transcript=transcript,
        windows=tuple(sorted((SessionWindow(so.timestamp_sec, so.tolerance_sec, tuple(so.unavailable))
                              for so in result.structured_observations),
                             key=lambda w: (w.timestamp_sec, w.tolerance_sec))),
        evidence=canonical_evidence(items),
    )


# ------------------------------- serialization -------------------------------

def session_to_dict(session: EvidenceSession) -> dict:
    return asdict(session)


def session_to_json(session: EvidenceSession) -> str:
    """Deterministic: sorted keys, fixed indent, no NaN/inf, UTF-8-safe text
    kept readable. The same session always produces the same bytes."""
    try:
        return json.dumps(session_to_dict(session), indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n"
    except (ValueError, TypeError) as e:
        raise SessionError(f"session is not JSON-safe: {e}") from e


def session_filename(session: EvidenceSession) -> str:
    """`<readable-stem>-<8 hex of the source>.session.json`. The stem is
    restricted to [A-Za-z0-9_-] (no separators, no traversal); the hash keeps two
    different videos with the same title from sharing a file, while the same
    source always maps to the same one."""
    name = session.source.title or os.path.splitext(os.path.basename(session.source.source))[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:60] or "video"
    digest = hashlib.sha256(session.source.source.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}{SESSION_SUFFIX}"


def save_session(session: EvidenceSession, path: str, *, overwrite: bool = False) -> str:
    """Write `session` to `path` atomically (temp file, then `os.replace`), so a
    reader never sees a partial file and a crash cannot damage an existing one.
    Refuses to replace an existing file unless `overwrite=True`. Returns `path`."""
    text = session_to_json(session)  # serialize first: a bad session writes nothing
    if os.path.exists(path) and not overwrite:
        raise SessionError(f"refusing to overwrite existing session file: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


# ------------------------------- loading -------------------------------

def _fail(what: str, why: str):
    raise SessionError(f"malformed session: {what} {why}")


def _obj(value, what: str, required: tuple, optional: tuple = ()) -> dict:
    if not isinstance(value, dict):
        _fail(what, f"must be an object, got {type(value).__name__}")
    missing = [k for k in required if k not in value]
    if missing:
        _fail(what, f"is missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(value) - set(required) - set(optional))
    if unknown:
        _fail(what, f"has unknown field(s): {', '.join(unknown)}")
    return value


def _list(value, what: str) -> list:
    if not isinstance(value, list):
        _fail(what, f"must be a list, got {type(value).__name__}")
    return value


def _str(value, what: str) -> str:
    if not isinstance(value, str):
        _fail(what, f"must be a string, got {type(value).__name__}")
    return value


def _num(value, what: str, *, optional: bool = False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(what, f"must be a finite number, got {value!r}")
    return float(value)


def _time(value, what: str, *, optional: bool = False):
    t = _num(value, what, optional=optional)
    if t is not None and t < 0:
        _fail(what, f"must be >= 0, got {t}")
    return t


def _evidence(raw, what: str) -> Evidence:
    d = _obj(raw, what, ("timestamp_sec", "kind", "ref"), ("confidence", "timestamp_end_sec", "source"))
    kind = _str(d["kind"], f"{what}.kind")
    if kind not in EVIDENCE_KINDS:
        _fail(f"{what}.kind", f"is not an evidence kind: {kind!r}")
    try:
        return Evidence(timestamp_sec=_time(d["timestamp_sec"], f"{what}.timestamp_sec"), kind=kind,
                        ref=_str(d["ref"], f"{what}.ref"),
                        confidence=_num(d.get("confidence"), f"{what}.confidence", optional=True),
                        timestamp_end_sec=_time(d.get("timestamp_end_sec"), f"{what}.timestamp_end_sec",
                                                optional=True),
                        source=_str(d.get("source", ""), f"{what}.source"))
    except ValueError as e:
        _fail(what, str(e))


def _transcript(raw) -> Transcript | None:
    if raw is None:
        return None
    d = _obj(raw, "transcript", ("segments",), ("language", "source", "duration_sec", "status"))
    segments = []
    for i, s in enumerate(_list(d["segments"], "transcript.segments")):
        w = f"transcript.segments[{i}]"
        sd = _obj(s, w, ("start_sec", "end_sec", "text"), ("confidence", "words"))
        words = None
        if sd.get("words") is not None:
            words = tuple(
                Word(text=_str(x["text"], f"{w}.words.text"), start_sec=_time(x["start_sec"], f"{w}.words.start_sec"),
                     end_sec=_time(x["end_sec"], f"{w}.words.end_sec"))
                for x in (_obj(x, f"{w}.words[]", ("text", "start_sec", "end_sec"))
                          for x in _list(sd["words"], f"{w}.words")))
        segments.append(TranscriptSegment(
            start_sec=_time(sd["start_sec"], f"{w}.start_sec"), end_sec=_time(sd["end_sec"], f"{w}.end_sec"),
            text=_str(sd["text"], f"{w}.text"), confidence=_num(sd.get("confidence"), f"{w}.confidence", optional=True),
            words=words))
    lang = d.get("language")
    try:
        return Transcript(segments=segments, language=None if lang is None else _str(lang, "transcript.language"),
                          source=_str(d.get("source", "unknown"), "transcript.source"),
                          duration_sec=_time(d.get("duration_sec"), "transcript.duration_sec", optional=True),
                          status=_str(d.get("status", "ok"), "transcript.status"))
    except ValueError as e:
        _fail("transcript", str(e))


def parse_session(text: str) -> EvidenceSession:
    """Strictly parse session JSON. Raises `SessionError` -- never anything
    else, never a silently-accepted partial structure -- for malformed JSON,
    missing/unknown/mistyped fields, an invalid evidence kind or value, and any
    format version other than the one this code writes."""
    try:
        raw = json.loads(text)
    except ValueError as e:  # JSONDecodeError and bad-UTF-8 surface here
        raise SessionError(f"session file is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        _fail("session", f"must be an object, got {type(raw).__name__}")
    version = raw.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int):
        _fail("format_version", f"must be an integer, got {version!r}")
    if version != SESSION_FORMAT_VERSION:
        raise SessionError(f"unsupported session format version {version} (this Video-Lens reads "
                           f"version {SESSION_FORMAT_VERSION}"
                           + ("; it was written by a newer Video-Lens" if version > SESSION_FORMAT_VERSION else "") + ")")
    d = _obj(raw, "session", ("format_version", "created_at", "video_lens_version", "source", "video", "evidence"),
             ("analysis", "transcript", "windows"))

    s = _obj(d["source"], "source", ("source_type", "source", "duration_sec"), ("title",))
    source_type = _str(s["source_type"], "source.source_type")
    if source_type not in ("local", "url"):
        _fail("source.source_type", f"must be 'local' or 'url', got {source_type!r}")
    title = s.get("title")
    source = KnowledgeSource(source_type=source_type, source=_str(s["source"], "source.source"),
                             duration_sec=_time(s["duration_sec"], "source.duration_sec"),
                             title=None if title is None else _str(title, "source.title"))
    v = _obj(d["video"], "video", ("width", "height", "fps", "has_audio"))
    if not isinstance(v["has_audio"], bool):
        _fail("video.has_audio", "must be a boolean")
    video = SessionVideo(width=int(_time(v["width"], "video.width")), height=int(_time(v["height"], "video.height")),
                         fps=_time(v["fps"], "video.fps"), has_audio=v["has_audio"])

    analysis = d.get("analysis", {})
    if not isinstance(analysis, dict) or not all(
            isinstance(k, str) and isinstance(x, (bool, int, float, str)) and
            (not isinstance(x, float) or math.isfinite(x)) for k, x in analysis.items()):
        _fail("analysis", "must be an object of strings mapped to booleans, finite numbers or strings")

    windows = []
    for i, w in enumerate(_list(d.get("windows", []), "windows")):
        wd = _obj(w, f"windows[{i}]", ("timestamp_sec", "tolerance_sec"), ("unavailable",))
        windows.append(SessionWindow(
            timestamp_sec=_time(wd["timestamp_sec"], f"windows[{i}].timestamp_sec"),
            tolerance_sec=_time(wd["tolerance_sec"], f"windows[{i}].tolerance_sec"),
            unavailable=tuple(_str(n, f"windows[{i}].unavailable[]")
                              for n in _list(wd.get("unavailable", []), f"windows[{i}].unavailable"))))

    return EvidenceSession(
        format_version=version, created_at=_str(d["created_at"], "created_at"),
        video_lens_version=_str(d["video_lens_version"], "video_lens_version"),
        source=source, video=video, analysis=dict(sorted(analysis.items())),
        transcript=_transcript(d.get("transcript")),
        windows=tuple(sorted(windows, key=lambda w: (w.timestamp_sec, w.tolerance_sec))),
        evidence=canonical_evidence(_evidence(e, f"evidence[{i}]")
                                    for i, e in enumerate(_list(d["evidence"], "evidence"))),
    )


def load_session(path: str) -> EvidenceSession:
    """Read and strictly validate a session file. See `parse_session`."""
    try:
        size = os.path.getsize(path)
        if size > _MAX_SESSION_BYTES:
            raise SessionError(f"session file is {size} bytes, over the {_MAX_SESSION_BYTES}-byte limit")
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise SessionError(f"cannot read session file {path}: {e.strerror or e}") from e
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SessionError(f"session file is not valid UTF-8: {e}") from e
    return parse_session(text)
