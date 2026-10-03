"""P1-A tests: evidence sessions -- a versioned, deterministic, opt-in record of
the observations an analysis produced. Persistence format, loading strictness,
the evidence boundary, and the lifecycle guarantees (default pipeline unchanged,
cleanup unchanged, no media retained).

Run: python -m pytest tests/test_session.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_lens as vl
from core.contracts import (
    EVIDENCE_KINDS, KNOWLEDGE_SCHEMA_VERSION, Evidence, Frame, Inference, PointerEvent, Transcript,
    TranscriptSegment, VideoInput, Word,
)
from core.errors import SessionError
from core.evidence import NATIVE_VIDEO_SOURCE, build_analysis_result
from core.session import (
    SESSION_FORMAT_VERSION, build_session, load_session, parse_session, save_session, session_filename,
    session_to_dict, session_to_json,
)
from core.visual_change import detect_visual_changes
from tests.test_evidence_schema import _moving_segments, _pointer
from tests.test_lifecycle import _fast_config, _make_video, _track_job_roots
from tests.test_video_understanding import _frame_obs, _obs
from tests.test_visual_change import _frame, _shift, _textured

VIDEO = VideoInput(path="/tmp/job123/dl/v.mp4", duration_sec=30.0, width=640, height=360, fps=5.0,
                   has_audio=True, source_type="local", source="/home/me/talk.mp4", title="Talk")
FIXED = "2026-01-01T00:00:00+00:00"


def _transcript() -> Transcript:
    return Transcript(segments=[
        TranscriptSegment(0.0, 3.0, "Welcome to the talk.", confidence=0.9,
                          words=(Word("Welcome", 0.0, 0.5), Word("to", 0.5, 0.6))),
        TranscriptSegment(4.0, 6.5, "Now I change the timeframe.", confidence=0.8),
    ], language="en", source="faster_whisper", duration_sec=30.0)


def _result(tmp: str):
    """One result carrying all six evidence kinds, produced by the real
    detectors and correlation (no hand-built Evidence)."""
    base = _textured(1)
    frames = [_frame(tmp, "f0", base, 2.0), _frame(tmp, "f1", _shift(base, 100, 60, 300, 200), 4.0)]
    return build_analysis_result(
        VIDEO, [2.0, 4.0], transcript=_transcript(), frames=frames,
        pointer_events=[_pointer(4.0, 120, 90), _pointer(4.5, 150, 90)],
        vision_observations=[_frame_obs(4.0, "a candlestick chart")],
        video_observations=[_obs(4.0, "native: a chart with a 1h label")],
        visual_changes=detect_visual_changes(frames), cursor_segments=_moving_segments(3.5),
        tolerance_sec=1.0)


def _session(tmp: str, **kw):
    return build_session(_result(tmp), _transcript(), created_at=FIXED, **kw)


# ============================== 1. round trip ==============================

def test_save_load_round_trip_is_exact():
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(tmp, analysis={"tolerance_sec": 1.0, "vision_enabled": True})
        path = save_session(session, os.path.join(tmp, "s", "x.session.json"))
        assert load_session(path) == session


def test_all_six_evidence_kinds_survive():
    with tempfile.TemporaryDirectory() as tmp:
        loaded = load_session(save_session(_session(tmp), os.path.join(tmp, "s.session.json")))
    assert {e.kind for e in loaded.evidence} == set(EVIDENCE_KINDS)


def test_transcript_is_preserved_completely():
    with tempfile.TemporaryDirectory() as tmp:
        loaded = load_session(save_session(_session(tmp), os.path.join(tmp, "s.session.json")))
    assert loaded.transcript == _transcript()
    assert loaded.transcript.segments[0].words[1] == Word("to", 0.5, 0.6)


def test_span_source_and_native_provenance_survive_the_file():
    with tempfile.TemporaryDirectory() as tmp:
        loaded = load_session(save_session(_session(tmp), os.path.join(tmp, "s.session.json")))
    cursor = [e for e in loaded.evidence if e.kind == "cursor_track"]
    assert cursor and all(e.timestamp_end_sec is not None and e.source == "cursor_intelligence" for e in cursor)
    vision = {(e.ref, e.source) for e in loaded.evidence if e.kind == "vision"}
    assert ("a candlestick chart", "") in vision, "per-frame vision keeps the default source"
    assert ("native: a chart with a 1h label", NATIVE_VIDEO_SOURCE) in vision, "native provenance not lost"
    change = [e for e in loaded.evidence if e.kind == "visual_change"]
    assert change and json.loads(change[0].ref)["compared"] == [2.0, 4.0] and change[0].source
    spoken = [e for e in loaded.evidence if e.kind == "transcript"]
    assert [(e.timestamp_sec, e.timestamp_end_sec) for e in spoken] == [(0.0, 3.0), (4.0, 6.5)]


def test_windows_record_which_streams_were_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(tmp)
    by_time = {w.timestamp_sec: w for w in session.windows}
    assert "visual_change" not in by_time[4.0].unavailable
    assert "visual_change" in by_time[2.0].unavailable, "no change event within 1s of t=2.0"
    assert "cursor_track" in by_time[2.0].unavailable and "cursor_track" not in by_time[4.0].unavailable
    assert [w.timestamp_sec for w in session.windows] == [2.0, 4.0]


# ============================== 2. evidence boundary ==============================

def test_session_holds_observations_not_conclusions():
    with tempfile.TemporaryDirectory() as tmp:
        result = _result(tmp)
        assert any(so.inferences for so in result.structured_observations), "fixture must produce inferences"
        text = session_to_json(build_session(result, _transcript(), created_at=FIXED))
    data = json.loads(text)
    assert set(data) <= {"format_version", "created_at", "video_lens_version", "source", "video", "evidence",
                         "analysis", "transcript", "windows"}
    inference_texts = [i.text for so in result.structured_observations for i in so.inferences]
    assert not any(t in text for t in inference_texts), "a Video-Lens conclusion was persisted"
    assert "pointer_in_vision_region" not in text and "speech_pointer" not in text


def test_frame_evidence_keeps_only_a_marker_never_a_temp_path():
    with tempfile.TemporaryDirectory() as tmp:
        text = session_to_json(_session(tmp))
        session = _session(tmp)
    frames = [e for e in session.evidence if e.kind == "frame"]
    assert [e.ref for e in frames] == ["f0.png", "f1.png"]
    assert tmp not in text and "\\" not in "".join(e.ref for e in frames)


def test_video_temp_path_is_not_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        text = session_to_json(_session(tmp))
    assert "/tmp/job123" not in text


def test_url_sources_lose_credentials_and_access_tokens():
    url = VideoInput(path="/tmp/x.mp4", duration_sec=10.0, width=1, height=1, fps=1.0, has_audio=False,
                     source_type="url", title="T",
                     source="https://user:pw@host.example:8443/watch?v=abc123&X-Amz-Signature=SECRET&token=t0k#frag")
    session = build_session(build_analysis_result(url, [0.0]), None, created_at=FIXED)
    assert session.source.source == "https://host.example:8443/watch?v=abc123"
    text = session_to_json(session)
    for leak in ("SECRET", "t0k", "pw", "user", "frag"):
        assert leak not in text


def test_analysis_metadata_is_plain_facts():
    with tempfile.TemporaryDirectory() as tmp:
        config = vl.PipelineConfig(vision_model="some-model", output_dir=tmp)
    facts = vl._session_analysis(config)
    assert all(isinstance(v, (bool, int, float)) for v in facts.values())
    assert "some-model" not in json.dumps(facts) and "dir" not in json.dumps(facts)


# ============================== 3. determinism ==============================

def test_serialization_is_deterministic_and_canonically_ordered():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = _session(tmp), _session(tmp)
        assert session_to_json(a) == session_to_json(b)
        shuffled = build_session(_result(tmp), _transcript(), created_at=FIXED)
    raw = session_to_json(a)
    assert raw.endswith("\n") and raw == session_to_json(parse_session(raw)), "round trip is byte-stable"
    times = [e.timestamp_sec for e in a.evidence]
    assert times == sorted(times) and shuffled.evidence == a.evidence
    keys = list(json.loads(raw))
    assert keys == sorted(keys)


def test_file_is_utf8_with_non_ascii_text_kept_readable():
    t = Transcript(segments=[TranscriptSegment(0.0, 1.0, "café 中文 — ok")], language="fr")
    session = build_session(build_analysis_result(VIDEO, [0.0], transcript=t), t, created_at=FIXED)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_session(session, os.path.join(tmp, "u.session.json"))
        raw = Path(path).read_bytes()
        assert "café 中文".encode("utf-8") in raw
        assert load_session(path).transcript == t


def test_duplicate_evidence_is_stored_once():
    ev = Evidence(timestamp_sec=1.0, kind="pointer", ref="x=1, y=2", confidence=0.5)
    data = session_to_dict(build_session(build_analysis_result(VIDEO, [1.0]), None, created_at=FIXED))
    data["evidence"] = [session_to_dict_ev(ev), session_to_dict_ev(ev)]
    assert len(parse_session(json.dumps(data)).evidence) == 1


def session_to_dict_ev(e: Evidence) -> dict:
    from dataclasses import asdict
    return asdict(e)


# ============================== 4. defaults / optional fields ==============================

def _minimal() -> dict:
    return {"format_version": 1, "created_at": FIXED, "video_lens_version": "1.0.0",
            "source": {"source_type": "local", "source": "a.mp4", "duration_sec": 5.0},
            "video": {"width": 10, "height": 10, "fps": 1.0, "has_audio": False},
            "evidence": [{"timestamp_sec": 1.0, "kind": "vision", "ref": "x"}]}


def test_missing_optional_fields_take_their_defaults():
    s = parse_session(json.dumps(_minimal()))
    e = s.evidence[0]
    assert (e.confidence, e.timestamp_end_sec, e.source) == (None, None, "")
    assert s.transcript is None and s.windows == () and s.analysis == {} and s.source.title is None


def test_old_style_evidence_without_the_step_5_fields_still_loads():
    for kind in ("frame", "transcript", "pointer", "vision"):
        d = _minimal()
        d["evidence"] = [{"timestamp_sec": 1.0, "kind": kind, "ref": "x", "confidence": 0.5}]
        assert parse_session(json.dumps(d)).evidence[0].kind == kind


def test_session_format_version_is_independent_of_the_knowledge_schema():
    assert SESSION_FORMAT_VERSION == 1 and KNOWLEDGE_SCHEMA_VERSION == "1.2"
    assert not isinstance(SESSION_FORMAT_VERSION, str)
    src = Path(__file__).resolve().parent.parent.joinpath("core/session.py").read_text(encoding="utf-8")
    assert "import KNOWLEDGE_SCHEMA_VERSION" not in src and "KNOWLEDGE_SCHEMA_VERSION," not in src


# ============================== 5. rejection ==============================

def _rejects(data) -> str:
    try:
        parse_session(data if isinstance(data, str) else json.dumps(data))
    except SessionError as e:
        return str(e)
    raise AssertionError(f"accepted a session it must reject: {str(data)[:120]}")


def test_unsupported_versions_are_rejected_clearly():
    for v in (2, 99, 0, -1):
        d = _minimal(); d["format_version"] = v
        assert "unsupported session format version" in _rejects(d)
    d = _minimal(); d["format_version"] = 2
    assert "newer Video-Lens" in _rejects(d)
    for bad in ("1", 1.0, True, None, [1]):
        d = _minimal(); d["format_version"] = bad
        _rejects(d)
    d = _minimal(); del d["format_version"]
    _rejects(d)


def test_malformed_json_and_wrong_top_level_types_are_rejected():
    for text in ("", "{", "not json", "[]", "null", "42", '"s"', '{"format_version": 1,}', "﻿{",
                 "NaN"):
        _rejects(text)
    assert "valid JSON" in _rejects("{")


def test_malformed_required_fields_are_rejected():
    mutations = [
        lambda d: d.pop("source"), lambda d: d.pop("video"), lambda d: d.pop("evidence"),
        lambda d: d.pop("created_at"), lambda d: d.update(evidence="x"), lambda d: d.update(evidence={}),
        lambda d: d.update(source="a.mp4"), lambda d: d["source"].pop("duration_sec"),
        lambda d: d["source"].update(duration_sec="5"), lambda d: d["source"].update(duration_sec=-1),
        lambda d: d["source"].update(source_type="ftp"), lambda d: d["video"].update(has_audio=1),
        lambda d: d["video"].update(width=None), lambda d: d.update(extra_field=1),
        lambda d: d["evidence"][0].update(kind="made_up"), lambda d: d["evidence"][0].update(kind="stationary"),
        lambda d: d["evidence"][0].update(kind=None), lambda d: d["evidence"][0].pop("ref"),
        lambda d: d["evidence"][0].pop("timestamp_sec"), lambda d: d["evidence"][0].update(timestamp_sec="1"),
        lambda d: d["evidence"][0].update(timestamp_sec=True), lambda d: d["evidence"][0].update(timestamp_sec=-1),
        lambda d: d["evidence"][0].update(timestamp_sec=1e999), lambda d: d["evidence"][0].update(confidence=2.0),
        lambda d: d["evidence"][0].update(confidence="hi"), lambda d: d["evidence"][0].update(timestamp_end_sec=0.5),
        lambda d: d["evidence"][0].update(source=5), lambda d: d["evidence"][0].update(mystery=1),
        lambda d: d["evidence"].append(7), lambda d: d.update(analysis={"k": [1]}),
        lambda d: d.update(analysis=[1]), lambda d: d.update(windows=[{"timestamp_sec": 1}]),
        lambda d: d.update(windows=[{"timestamp_sec": 1, "tolerance_sec": 1, "unavailable": "vision"}]),
        lambda d: d.update(transcript={"segments": "no"}),
        lambda d: d.update(transcript={"segments": [{"start_sec": 2, "end_sec": 1, "text": "x"}]}),
        lambda d: d.update(transcript={"segments": [{"start_sec": 0, "end_sec": 1}]}),
        lambda d: d.update(transcript={"segments": []}),  # status 'ok' with no segments
        lambda d: d.update(transcript={"segments": [{"start_sec": 0, "end_sec": 1, "text": "x",
                                                     "words": [{"text": "x"}]}]}),
    ]
    for mutate in mutations:
        d = copy.deepcopy(_minimal())
        mutate(d)
        _rejects(d)


def test_unreadable_oversized_and_non_utf8_files_raise_session_error():
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in (("bin.json", b"\xff\xfe\x00bad"), ("empty.json", b"")):
            p = os.path.join(tmp, name)
            Path(p).write_bytes(content)
            try:
                load_session(p)
                assert False
            except SessionError:
                pass
        for target in (os.path.join(tmp, "missing.json"), tmp):  # absent file; a directory
            try:
                load_session(target)
                assert False
            except SessionError:
                pass
        from unittest import mock
        big = os.path.join(tmp, "big.json")
        Path(big).write_text("{}")
        with mock.patch.dict(load_session.__globals__, {"_MAX_SESSION_BYTES": 1}):  # the function under test's own module
            try:
                load_session(big)
                assert False
            except SessionError as e:
                assert "limit" in str(e)


# ============================== 6. writing safely ==============================

def test_saving_never_overwrites_by_default_and_never_leaves_a_temp_file():
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(tmp)
        path = os.path.join(tmp, "out", "s.session.json")
        save_session(session, path)
        original = Path(path).read_bytes()
        try:
            save_session(session, path)
            assert False, "silent overwrite"
        except SessionError:
            pass
        assert Path(path).read_bytes() == original
        save_session(session, path, overwrite=True)
        assert os.listdir(os.path.dirname(path)) == ["s.session.json"], "no stray .tmp"


def test_a_failed_write_leaves_the_existing_session_intact():
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(tmp)
        path = os.path.join(tmp, "s.session.json")
        save_session(session, path)
        before = Path(path).read_bytes()
        with mock.patch("core.session.os.replace", side_effect=OSError("disk full")):
            try:
                save_session(session, path, overwrite=True)
                assert False
            except OSError:
                pass
        assert Path(path).read_bytes() == before and not [f for f in os.listdir(tmp) if f.endswith(".tmp")]


def test_a_non_json_safe_session_writes_nothing():
    from dataclasses import replace
    with tempfile.TemporaryDirectory() as tmp:
        bad = replace(_session(tmp), analysis={"x": float("nan")})
        target = os.path.join(tmp, "n.session.json")
        try:
            save_session(bad, target)
            assert False
        except SessionError:
            pass
        assert not os.path.exists(target) and not os.path.exists(target + ".tmp")


def test_filenames_cannot_traverse_and_distinguish_sources():
    def name(title, source):
        v = VideoInput(path="p", duration_sec=1.0, width=1, height=1, fps=1.0, has_audio=False,
                       source=source, title=title)
        return session_filename(build_session(build_analysis_result(v, [0.0]), None, created_at=FIXED))

    hostile = name("../../etc/passwd\\..\\x", "a.mp4")
    assert os.path.basename(hostile) == hostile and ".." not in hostile and "/" not in hostile
    assert hostile.endswith(".session.json")
    assert name("Same Title", "a.mp4") != name("Same Title", "b.mp4"), "different videos must not collide"
    assert name("Same Title", "a.mp4") == name("Same Title", "a.mp4")
    assert name(None, "/x/y/talk.mp4").startswith("talk-")


# ============================== 7. lifecycle ==============================

def test_default_config_persists_nothing_and_has_no_session_option_set():
    assert vl.PipelineConfig().session_dir is None
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        with _track_job_roots():
            vl.process_video(p, _fast_config(tmp))
        written = sorted(str(f.relative_to(tmp)) for f in Path(tmp).rglob("*") if f.is_file())
    assert not [w for w in written if "session" in w], written


def test_enabled_session_is_written_alongside_an_unchanged_lifecycle():
    with _track_job_roots() as roots:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            sessions = os.path.join(tmp, "sessions")
            package = vl.process_video(p, _fast_config(tmp, session_dir=sessions))
            files = os.listdir(sessions)
            assert len(files) == 1 and files[0].endswith(".session.json")
            loaded = load_session(os.path.join(sessions, files[0]))
            assert loaded.source.duration_sec == package.source.duration_sec
            assert loaded.analysis["tolerance_sec"] == 1.0 and loaded.analysis["video_understanding"] is False
            assert os.path.exists(p), "the caller's own video is never touched"
        assert len(roots) == 1 and not os.path.exists(roots[0]), "temp workspace still cleaned up on success"


def test_session_directory_holds_only_the_session_never_media():
    with _track_job_roots():
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            sessions = os.path.join(tmp, "sessions")
            vl.process_video(p, _fast_config(tmp, session_dir=sessions))
            kept = [f for f in Path(sessions).rglob("*") if f.is_file()]
            assert [f.suffix for f in kept] == [".json"]
            assert sum(f.stat().st_size for f in kept) < 100_000
            blob = kept[0].read_bytes()
            assert b"\xff\xd8\xff" not in blob and b"ftyp" not in blob, "no JPEG/MP4 bytes in a session"


def test_reanalysing_the_same_video_replaces_its_session_atomically():
    with _track_job_roots():
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            sessions = os.path.join(tmp, "sessions")
            vl.process_video(p, _fast_config(tmp, session_dir=sessions))
            vl.process_video(p, _fast_config(tmp, session_dir=sessions))
            assert len(os.listdir(sessions)) == 1


def test_failures_behave_exactly_as_without_a_session():
    """The existing lifecycle leaves the temp workspace in place on failure
    (docs/lifecycle.md) -- unchanged, with or without a session."""
    from core.errors import KnowledgePackageError
    for extra in ({}, {"session_dir": "SESS"}):
        original = vl._validate_package
        vl._validate_package = lambda package: (_ for _ in ()).throw(KnowledgePackageError("forced"))
        try:
            with _track_job_roots() as roots:
                with tempfile.TemporaryDirectory() as tmp:
                    p = os.path.join(tmp, "v.mp4")
                    _make_video(p)
                    cfg = dict(extra)
                    if "session_dir" in cfg:
                        cfg["session_dir"] = os.path.join(tmp, "sessions")
                    try:
                        vl.process_video(p, _fast_config(tmp, **cfg))
                        assert False
                    except KnowledgePackageError:
                        pass
                    assert not os.path.exists(os.path.join(tmp, "sessions")), "no session for a failed job"
                assert len(roots) == 1 and os.path.isdir(roots[0])
        finally:
            vl._validate_package = original


def test_a_session_write_failure_fails_the_job_and_keeps_the_workspace():
    with _track_job_roots() as roots:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            blocked = os.path.join(tmp, "blocked")
            Path(blocked).write_text("a file, not a directory")
            try:
                vl.process_video(p, _fast_config(tmp, session_dir=blocked))
                assert False
            except (OSError, SessionError):
                pass
        assert len(roots) == 1 and os.path.isdir(roots[0]), "marked successful despite a failed write"


def test_retain_temp_artifacts_is_independent_of_the_session():
    with _track_job_roots() as roots:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            vl.process_video(p, _fast_config(tmp, session_dir=os.path.join(tmp, "s"), retain_temp_artifacts=True))
        assert os.path.isdir(roots[0])


def test_package_is_unchanged_by_enabling_a_session():
    with _track_job_roots():
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.mp4")
            _make_video(p)
            plain = vl.process_video(p, _fast_config(tmp))
            with_session = vl.process_video(p, _fast_config(tmp, session_dir=os.path.join(tmp, "s")))
    from dataclasses import replace
    stamp = lambda pk: replace(pk, processing=replace(pk.processing, generated_at=""))
    assert stamp(plain) == stamp(with_session)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
