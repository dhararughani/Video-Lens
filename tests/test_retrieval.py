"""P1-A tests: deterministic evidence retrieval -- exact boundary, tolerance,
point-vs-span, filter and ordering semantics. Run: python -m pytest tests/test_retrieval.py
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import evidence as evidence_module
from core import retrieval
from core.contracts import EVIDENCE_KINDS, Evidence
from core.evidence import NATIVE_VIDEO_SOURCE
from core.retrieval import evidence_between, evidence_near
from core.session import build_session, load_session, save_session
from tests.test_session import FIXED, _result, _transcript


def pt(t, kind="vision", ref="x", source="", conf=None) -> Evidence:
    return Evidence(timestamp_sec=t, kind=kind, ref=ref, source=source, confidence=conf)


def span(a, b, kind="cursor_track", ref="{}", source="cursor_intelligence") -> Evidence:
    return Evidence(timestamp_sec=a, timestamp_end_sec=b, kind=kind, ref=ref, source=source)


def times(result):
    return [e.timestamp_sec for e in result]


# ============================== near: points ==============================

def test_exact_timestamp_with_zero_tolerance():
    data = [pt(4.0), pt(4.001), pt(3.999)]
    assert times(evidence_near(data, 4.0, 0.0)) == [4.0]


def test_tolerance_boundaries_are_inclusive_on_both_sides():
    data = [pt(2.9), pt(3.0), pt(4.0), pt(5.0), pt(5.1)]
    assert times(evidence_near(data, 4.0, 1.0)) == [3.0, 4.0, 5.0]
    assert times(evidence_near(data, 4.0, 0.99)) == [4.0]


def test_before_and_after_the_window_are_excluded():
    data = [pt(0.0), pt(10.0)]
    assert evidence_near(data, 5.0, 1.0) == ()


def test_default_tolerance_matches_the_pipelines_own():
    assert times(evidence_near([pt(3.0), pt(5.0), pt(5.5)], 4.0)) == [3.0, 5.0]


def test_point_matching_is_literally_the_correlation_primitive():
    """Retrieval must never disagree with correlation, including at float edges
    where `abs(a - t) <= tol` and `t - tol <= a` round differently."""
    rng = random.Random(7)
    for _ in range(300):
        t, tol = round(rng.uniform(0, 100), 3), round(rng.uniform(0, 3), 3)
        data = [pt(round(t + rng.choice((-1, 1)) * tol * rng.choice((0.5, 1.0, 1.0000001, 0.9999999)), 6) % 120)
                for _ in range(6)]
        want = evidence_module._within_tolerance(data, t, tol)
        assert set(evidence_near(data, t, tol)) == set(want)  # (retrieval also de-duplicates)


def test_the_primitive_is_actually_used():
    from unittest import mock
    with mock.patch.object(retrieval, "_within_tolerance", return_value=[]) as m:
        assert evidence_near([pt(4.0)], 4.0, 1.0) == ()
    assert m.called


# ============================== near / between: spans ==============================

def test_a_span_is_returned_when_it_overlaps_even_if_it_starts_outside_the_window():
    data = [span(17.0, 18.0)]
    assert evidence_near(data, 17.5, 0.0) == tuple(data), "inside the span, 0.5s from its start"
    assert evidence_near(data, 18.5, 0.5) == tuple(data), "touching at the end counts"
    assert evidence_near(data, 16.5, 0.5) == tuple(data), "touching at the start counts"
    assert evidence_near(data, 19.0, 0.5) == () and evidence_near(data, 16.0, 0.5) == ()


def test_span_overlap_is_not_point_matching():
    """A long span whose START is far from t must still match: reducing it to
    its start point would return nothing here."""
    long_span = span(10.0, 30.0)
    assert evidence_near([long_span], 20.0, 1.0) == (long_span,)
    assert evidence_between([long_span], 15.0, 16.0) == (long_span,)
    assert evidence_between([long_span], 31.0, 40.0) == ()


def test_between_is_closed_for_points_and_overlapping_for_spans():
    data = [pt(1.0), pt(2.0), pt(3.0), span(3.0, 4.0), span(0.0, 1.0), span(4.0, 5.0)]
    got = evidence_between(data, 1.0, 3.0)
    assert pt(1.0) in got and pt(3.0) in got and pt(2.0) in got
    assert span(0.0, 1.0) in got and span(3.0, 4.0) in got, "touching spans overlap a closed window"
    assert span(4.0, 5.0) not in got
    assert evidence_between(data, 3.0, 3.0) == (pt(3.0), span(3.0, 4.0))


def test_zero_length_spans_behave_like_a_point_at_that_moment():
    z = span(5.0, 5.0)
    assert evidence_near([z], 5.0, 0.0) == (z,)
    assert evidence_near([z], 5.1, 0.0) == ()


def test_a_mixed_stream_returns_points_and_spans_together_in_order():
    data = [span(3.5, 6.0), pt(4.0, ref="a"), pt(9.0)]
    got = evidence_near(data, 4.0, 1.0)
    assert got == (span(3.5, 6.0), pt(4.0, ref="a"))


# ============================== filters ==============================

def _mixed():
    return [pt(4.0, "vision", "frame view"), pt(4.0, "vision", "native view", source=NATIVE_VIDEO_SOURCE),
            pt(4.0, "pointer", "x=1"), pt(4.0, "frame", "f.png"), span(3.5, 4.5)]


def test_kind_filter_accepts_one_kind_or_several_and_none_means_all():
    data = _mixed()
    assert {e.kind for e in evidence_near(data, 4.0, 1.0)} == {"vision", "pointer", "frame", "cursor_track"}
    assert {e.kind for e in evidence_near(data, 4.0, 1.0, kinds="vision")} == {"vision"}
    assert {e.kind for e in evidence_near(data, 4.0, 1.0, kinds=["pointer", "cursor_track"])} == \
        {"pointer", "cursor_track"}
    assert evidence_near(data, 4.0, 1.0, kinds="transcript") == ()


def test_source_filter_separates_native_from_frame_vision_and_the_default_label():
    data = _mixed()
    native = evidence_near(data, 4.0, 1.0, kinds="vision", sources=NATIVE_VIDEO_SOURCE)
    assert [e.ref for e in native] == ["native view"]
    default = evidence_near(data, 4.0, 1.0, kinds="vision", sources="")
    assert [e.ref for e in default] == ["frame view"], "'' is the per-frame / unlabelled source"
    both = evidence_near(data, 4.0, 1.0, kinds="vision", sources=["", NATIVE_VIDEO_SOURCE])
    assert len(both) == 2
    assert evidence_near(data, 4.0, 1.0, sources="nobody") == ()


def test_filters_combine_with_spans():
    assert [e.kind for e in evidence_near(_mixed(), 4.0, 0.0, kinds="cursor_track",
                                          sources="cursor_intelligence")] == ["cursor_track"]
    assert evidence_between(_mixed(), 0.0, 10.0, kinds="cursor_track", sources=NATIVE_VIDEO_SOURCE) == ()


def test_every_valid_kind_filter_is_accepted_and_unknown_ones_are_not():
    for kind in EVIDENCE_KINDS:
        evidence_near([], 0.0, 1.0, kinds=kind)
    for bad in ("video_understanding", "stationary", "Vision", ""):
        try:
            evidence_near([], 0.0, 1.0, kinds=bad)
            assert False, bad
        except ValueError:
            pass


# ============================== argument validation ==============================

def test_invalid_arguments_raise_instead_of_returning_something_plausible():
    bad_calls = [
        lambda: evidence_near([], -1.0, 1.0), lambda: evidence_near([], float("nan"), 1.0),
        lambda: evidence_near([], 1.0, -0.1), lambda: evidence_near([], 1.0, float("inf")),
        lambda: evidence_near([], True, 1.0), lambda: evidence_near([], "4", 1.0),
        lambda: evidence_between([], 5.0, 4.0), lambda: evidence_between([], -1.0, 4.0),
        lambda: evidence_between([], 0.0, float("nan")),
        lambda: evidence_near([], 1.0, 1.0, kinds=[]), lambda: evidence_near([], 1.0, 1.0, sources=[]),
    ]
    for call in bad_calls:
        try:
            call()
            assert False
        except ValueError:
            pass


# ============================== ordering / duplicates / empty ==============================

def test_empty_results_are_an_empty_tuple_not_none():
    assert evidence_near([], 1.0, 1.0) == () and evidence_between([pt(5.0)], 0.0, 1.0) == ()
    assert isinstance(evidence_near([pt(1.0)], 1.0, 0.0), tuple)


def test_ordering_is_by_start_end_kind_source_ref_confidence():
    a = pt(1.0, "pointer", "b")
    b = pt(1.0, "pointer", "a")
    c = pt(1.0, "frame", "z")
    d = span(1.0, 2.0)
    e = span(1.0, 3.0)
    f = pt(0.5, "vision", "early")
    g = pt(1.0, "pointer", "a", conf=0.4)
    got = evidence_between([a, b, c, d, e, f, g], 0.0, 5.0)
    assert [(x.timestamp_sec, x.kind, x.ref) for x in got] == [
        (0.5, "vision", "early"),
        (1.0, "frame", "z"), (1.0, "pointer", "a"), (1.0, "pointer", "a"), (1.0, "pointer", "b"),
        (1.0, "cursor_track", "{}"), (1.0, "cursor_track", "{}")]
    assert got[2].confidence is None and got[3].confidence == 0.4, "no-confidence sorts before 0.4"
    assert (got[5].timestamp_end_sec, got[6].timestamp_end_sec) == (2.0, 3.0)


def test_ordering_does_not_depend_on_input_order():
    data = _mixed() + [pt(4.2, "transcript", "hello", conf=0.9), span(4.0, 4.4, "transcript", "hi", "")]
    want = evidence_near(data, 4.0, 1.0)
    for perm in itertools.islice(itertools.permutations(data), 200):
        assert evidence_near(perm, 4.0, 1.0) == want
    rng = random.Random(3)
    for _ in range(50):
        shuffled = data[:]
        rng.shuffle(shuffled)
        assert evidence_between(shuffled, 0.0, 9.0) == evidence_between(data, 0.0, 9.0)


def test_exact_duplicates_are_returned_once_but_near_duplicates_are_not_merged():
    one = pt(4.0, ref="same")
    assert evidence_near([one, one, pt(4.0, ref="same")], 4.0, 0.0) == (one,)
    different_source = pt(4.0, ref="same", source=NATIVE_VIDEO_SOURCE)
    assert len(evidence_near([one, different_source], 4.0, 0.0)) == 2
    assert len(evidence_near([pt(4.0, conf=0.5), pt(4.0, conf=0.6)], 4.0, 0.0)) == 2


# ============================== over a real session ==============================

def test_retrieval_over_a_saved_and_reloaded_session():
    with tempfile.TemporaryDirectory() as tmp:
        session = build_session(_result(tmp), _transcript(), created_at=FIXED)
        loaded = load_session(save_session(session, os.path.join(tmp, "s.session.json")))
    assert evidence_near(loaded, 4.0, 1.0) == evidence_near(session, 4.0, 1.0)
    assert evidence_near(loaded, 4.0, 1.0) == evidence_near(loaded.evidence, 4.0, 1.0), "session or iterable"

    said = evidence_near(loaded, 5.0, 0.0, kinds="transcript")
    assert [e.ref for e in said] == ["Now I change the timeframe."], "span 4.0-6.5 covers t=5.0"

    moving = evidence_near(loaded, 4.2, 0.0, kinds="cursor_track")
    assert moving and json.loads(moving[0].ref)["state"] == "moving" and moving[0].timestamp_end_sec == 5.5

    native = evidence_near(loaded, 4.0, 1.0, kinds="vision", sources=NATIVE_VIDEO_SOURCE)
    assert [e.ref for e in native] == ["native: a chart with a 1h label"]

    change = evidence_between(loaded, 3.0, 5.0, kinds="visual_change")
    assert len(change) == 1 and json.loads(change[0].ref)["compared"] == [2.0, 4.0]
    assert evidence_near(loaded, 20.0, 1.0) == ()


def test_retrieval_never_calls_a_model_or_the_network():
    src = Path(__file__).resolve().parent.parent.joinpath("core/retrieval.py").read_text(encoding="utf-8")
    for banned in ("import requests", "urllib", "socket", "anthropic", "openai", "embedding", "numpy", "random"):
        assert banned not in src.replace("embeddings", ""), banned


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
