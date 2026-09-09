"""Vision intelligence tests: contract validation, mocked-model analysis
(success/low-information/failure), pointer/transcript integration, cache
behavior, and multimodal evidence construction.

Uses a fake Anthropic client (no network, no API key needed) -- run:
    python tests/test_vision.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.vision.claude_vision import ClaudeVisionAdapter
from core.contracts import Frame, MultimodalObservation, PointerEvent, Region, VisionObservation, VisualElement


# --------------------------- fake Anthropic client ---------------------------

class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responder(kwargs)


class _FakeClient:
    def __init__(self, responder):
        self.messages = _FakeMessages(responder)


def _adapter_with_fake_client(responder, cache_dir, model="claude-sonnet-5"):
    a = ClaudeVisionAdapter(model=model, cache_dir=cache_dir, api_key="fake-key-for-test")
    a._client = _FakeClient(responder)
    a._unavailable_reason = None
    return a


def _ok_response(**overrides):
    payload = {
        "description": "A code editor with a file tree on the left and an open Python file.",
        "visible_text": ["main.py", "def run():"],
        "low_information": False,
        "elements": [
            {"kind": "text", "description": "file name tab", "region": [0.1, 0.0, 0.3, 0.05],
             "region_confidence": "detected"},
        ],
        "confidence": 0.87,
    }
    payload.update(overrides)
    return lambda kwargs: _FakeResponse(json.dumps(payload))


def _frame(tmp, timestamp=1.0, name="f.jpg"):
    p = os.path.join(tmp, name)
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake-jpeg-bytes")  # content is irrelevant, never decoded here
    return Frame(timestamp_sec=timestamp, path=p, width=1920, height=1080)


# ------------------------------- contract --------------------------------

def test_region_validates_bounds_and_ordering():
    Region(x1=0.1, y1=0.1, x2=0.5, y2=0.5)  # ok
    for bad in [dict(x1=-0.1, y1=0, x2=0.5, y2=0.5), dict(x1=0, y1=0, x2=1.5, y2=0.5),
                dict(x1=0.5, y1=0, x2=0.4, y2=0.5), dict(x1=0, y1=0.5, x2=0.5, y2=0.5)]:
        try:
            Region(**bad)
            assert False, f"should have rejected {bad}"
        except ValueError:
            pass


def test_visual_element_region_confidence_consistency():
    VisualElement(kind="text", description="x", region=None, region_confidence="unknown")  # ok
    VisualElement(kind="text", description="x", region=Region(0, 0, 0.5, 0.5),
                   region_confidence="detected")  # ok
    try:
        VisualElement(kind="text", description="x", region=None, region_confidence="detected")
        assert False, "region_confidence must be 'unknown' when region is None"
    except ValueError:
        pass
    try:
        VisualElement(kind="text", description="x", region=Region(0, 0, 0.5, 0.5),
                       region_confidence="unknown")
        assert False, "a located region must not carry 'unknown' confidence"
    except ValueError:
        pass


def test_vision_observation_status_and_confidence_validated():
    VisionObservation(timestamp_sec=1.0, frame_path="f.jpg", status="unavailable")  # ok
    try:
        VisionObservation(timestamp_sec=1.0, frame_path="f.jpg", status="nonsense")
        assert False
    except ValueError:
        pass
    try:
        VisionObservation(timestamp_sec=1.0, frame_path="f.jpg", status="ok", confidence=1.5)
        assert False
    except ValueError:
        pass


def test_vision_observation_accepts_any_finite_timestamp():
    # Range/validity of timestamp_sec is the extraction boundary's job (Step 4's
    # FrameExtractor), not this contract's -- consistent with Frame/PointerEvent.
    VisionObservation(timestamp_sec=0.0, frame_path="f.jpg", status="unavailable")
    VisionObservation(timestamp_sec=99999.999, frame_path="f.jpg", status="unavailable")


# ------------------------------ successful analysis ------------------------------

def test_successful_analysis_returns_structured_observation():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=os.path.join(tmp, "cache"))
        obs = a.analyze_frame(frame)
        assert obs.status == "ok"
        assert "code editor" in obs.description
        assert obs.visible_text == ("main.py", "def run():")
        assert len(obs.elements) == 1
        assert obs.elements[0].region.x1 == 0.1
        assert obs.confidence == 0.87
        assert obs.model == "claude-sonnet-5"


def test_low_information_frame_is_distinguished():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        resp = _ok_response(description="A blank gray screen.", elements=[],
                             visible_text=[], low_information=True, confidence=0.95)
        a = _adapter_with_fake_client(resp, cache_dir=os.path.join(tmp, "cache"))
        obs = a.analyze_frame(frame)
        assert obs.status == "low_information"
        assert obs.elements == ()


def test_analysis_failure_never_hallucinates():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)

        def boom(kwargs):
            raise RuntimeError("simulated API error")

        a = _adapter_with_fake_client(boom, cache_dir=os.path.join(tmp, "cache"))
        obs = a.analyze_frame(frame)
        assert obs.status == "failed"
        assert obs.description == ""
        assert obs.visible_text == ()
        assert obs.elements == ()
        assert "simulated API error" in obs.analysis_metadata["error"]


def test_malformed_model_response_is_a_failure_not_a_crash():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(lambda kwargs: _FakeResponse("not json at all"),
                                       cache_dir=os.path.join(tmp, "cache"))
        obs = a.analyze_frame(frame)
        assert obs.status == "failed"


def test_missing_required_field_is_a_failure():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        # no "confidence" key
        resp = lambda kwargs: _FakeResponse(json.dumps({"description": "x"}))
        a = _adapter_with_fake_client(resp, cache_dir=os.path.join(tmp, "cache"))
        obs = a.analyze_frame(frame)
        assert obs.status == "failed"


def test_unavailable_when_no_credentials_configured():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            a = ClaudeVisionAdapter(cache_dir=os.path.join(tmp, "cache"))
            obs = a.analyze_frame(frame)
            assert obs.status == "unavailable"
            assert obs.description == ""
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old


# ------------------------- pointer / transcript integration -------------------

def test_pointer_evidence_is_included_but_marked_as_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=os.path.join(tmp, "cache"))
        pointer = PointerEvent(timestamp_sec=1.0, status="uncertain", x=960, y=540,
                                frame_width=1920, frame_height=1080, confidence=0.6)
        a.analyze_frame(frame, pointer=pointer)
        sent_text = a._client.messages.calls[0]["messages"][0]["content"][1]["text"]
        assert "0.500" in sent_text  # normalized coordinates present
        assert "evidence" not in sent_text.lower() or "Detected cursor position" in sent_text
        assert "status=uncertain" in sent_text


def test_not_detected_pointer_is_never_sent_as_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=os.path.join(tmp, "cache"))
        pointer = PointerEvent(timestamp_sec=1.0, status="not_detected")
        a.analyze_frame(frame, pointer=pointer)
        sent_text = a._client.messages.calls[0]["messages"][0]["content"][1]["text"]
        assert "cursor" not in sent_text.lower(), \
            "a not_detected pointer must not be presented as evidence"


def test_transcript_context_is_included_as_context_only():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=os.path.join(tmp, "cache"))
        a.analyze_frame(frame, transcript_context="let's look at this function")
        sent_text = a._client.messages.calls[0]["messages"][0]["content"][1]["text"]
        assert "let's look at this function" in sent_text
        assert "do not transcribe" in sent_text.lower()


# --------------------------------- caching ---------------------------------

def test_cache_hit_avoids_a_second_model_call():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=os.path.join(tmp, "cache"))
        obs1 = a.analyze_frame(frame)
        obs2 = a.analyze_frame(frame)
        assert len(a._client.messages.calls) == 1
        assert obs2.analysis_metadata.get("cache") == "hit"
        assert obs1.description == obs2.description


def test_different_timestamp_is_a_cache_miss():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        frame_a = _frame(tmp, timestamp=1.0, name="a.jpg")
        frame_b = _frame(tmp, timestamp=2.0, name="a.jpg")  # same path, different timestamp
        a = _adapter_with_fake_client(_ok_response(), cache_dir=cache_dir)
        a.analyze_frame(frame_a)
        a.analyze_frame(frame_b)
        assert len(a._client.messages.calls) == 2


def test_different_model_is_a_cache_miss():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        frame = _frame(tmp)
        a1 = _adapter_with_fake_client(_ok_response(), cache_dir=cache_dir, model="claude-sonnet-5")
        a2 = _adapter_with_fake_client(_ok_response(), cache_dir=cache_dir, model="claude-opus-5")
        a1.analyze_frame(frame)
        a2.analyze_frame(frame)
        assert len(a1._client.messages.calls) == 1
        assert len(a2._client.messages.calls) == 1  # a2 did NOT reuse a1's cache entry


def test_corrupted_cache_entry_is_regenerated_not_trusted():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=cache_dir)
        a.analyze_frame(frame)
        assert len(a._client.messages.calls) == 1
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1
        # corrupt it
        with open(os.path.join(cache_dir, cache_files[0]), "w") as f:
            f.write("{not valid json")
        obs = a.analyze_frame(frame)
        assert len(a._client.messages.calls) == 2  # regenerated, not served corrupted
        assert obs.status == "ok"


def test_failed_analysis_is_not_cached():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        frame = _frame(tmp)
        calls = {"n": 0}

        def flaky(kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient error")
            return _FakeResponse(json.dumps({
                "description": "ok now", "visible_text": [], "low_information": False,
                "elements": [], "confidence": 0.8,
            }))

        a = _adapter_with_fake_client(flaky, cache_dir=cache_dir)
        obs1 = a.analyze_frame(frame)
        assert obs1.status == "failed"
        obs2 = a.analyze_frame(frame)  # should retry, not serve a cached failure
        assert obs2.status == "ok"
        assert calls["n"] == 2


# ------------------------ ordering / multimodal evidence -----------------------

def test_multiple_observations_preserve_timestamp_order():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = os.path.join(tmp, "cache")
        a = _adapter_with_fake_client(_ok_response(), cache_dir=cache_dir)
        frames = [_frame(tmp, timestamp=t, name=f"f{t}.jpg") for t in (5.0, 1.0, 3.0)]
        observations = [a.analyze_frame(f) for f in frames]
        assert [o.timestamp_sec for o in observations] == [5.0, 1.0, 3.0]  # order follows input


def test_multimodal_observation_joins_frame_transcript_pointer_and_vision():
    with tempfile.TemporaryDirectory() as tmp:
        frame = _frame(tmp)
        a = _adapter_with_fake_client(_ok_response(), cache_dir=os.path.join(tmp, "cache"))
        pointer = PointerEvent(timestamp_sec=1.0, status="detected", x=100, y=100,
                                frame_width=1920, frame_height=1080, confidence=0.9)
        vision = a.analyze_frame(frame, transcript_context="here", pointer=pointer)
        evidence = MultimodalObservation(
            timestamp_sec=1.0, frame=frame, transcript_context="here",
            description=vision.description, pointer=pointer, vision=vision,
        )
        assert evidence.frame is frame
        assert evidence.pointer is pointer
        assert evidence.vision is vision
        assert evidence.vision.status == "ok"


if __name__ == "__main__":
    test_region_validates_bounds_and_ordering()
    test_visual_element_region_confidence_consistency()
    test_vision_observation_status_and_confidence_validated()
    test_vision_observation_accepts_any_finite_timestamp()
    test_successful_analysis_returns_structured_observation()
    test_low_information_frame_is_distinguished()
    test_analysis_failure_never_hallucinates()
    test_malformed_model_response_is_a_failure_not_a_crash()
    test_missing_required_field_is_a_failure()
    test_unavailable_when_no_credentials_configured()
    test_pointer_evidence_is_included_but_marked_as_evidence()
    test_not_detected_pointer_is_never_sent_as_evidence()
    test_transcript_context_is_included_as_context_only()
    test_cache_hit_avoids_a_second_model_call()
    test_different_timestamp_is_a_cache_miss()
    test_different_model_is_a_cache_miss()
    test_corrupted_cache_entry_is_regenerated_not_trusted()
    test_failed_analysis_is_not_cached()
    test_multiple_observations_preserve_timestamp_order()
    test_multimodal_observation_joins_frame_transcript_pointer_and_vision()
    print("All vision tests passed.")
