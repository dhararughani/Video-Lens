"""Step 12 LIVE VALIDATION ONLY -- not a Video-Lens adapter, not imported by
video_lens.py or anything under adapters/, and never installed by
requirements*.txt.

This is a `core.interfaces.VisionAdapter` implementation backed by a local
Ollama vision model, used ONCE to prove the seam genuinely accepts a
real, non-Claude visual provider and produces real `vision`-kind evidence
(interpreted content), as distinct from a bare, uninterpreted `frame`. It
lives in scripts/ specifically so it is obvious this is validation tooling,
not a second vision subsystem Video-Lens ships or depends on.

Requires a local Ollama daemon with a vision-capable model already pulled
(e.g. `ollama pull moondream`) -- entirely optional, external to Video-Lens,
and already available on this machine.
"""
from __future__ import annotations

import base64
import json
import urllib.request

from core.contracts import Frame, PointerEvent, VisionObservation


class OllamaVisionAdapter:
    """Implements core.interfaces.VisionAdapter via a local Ollama model.
    Minimal by design: no caching, no retries, no JSON-schema enforcement --
    those are ClaudeVisionAdapter's concerns (adapters/vision/claude_vision.py),
    not this script's. This exists only to prove the seam accepts a real
    non-Claude provider."""

    def __init__(self, model: str = "moondream", host: str = "http://localhost:11434",
                 timeout: float = 90.0):
        self.model = model
        self.host = host
        self.timeout = timeout

    def analyze_frame(self, frame: Frame, transcript_context: str | None = None,
                       pointer: PointerEvent | None = None) -> VisionObservation:
        # A small local model (moondream, 1B) was observed during Step 12
        # validation to ignore "do not transcribe" when transcript context was
        # appended to the prompt, and simply echo the context back as if it
        # were a visual description -- a real failure mode a stronger hosted
        # model (Claude) did not exhibit. Since this script's only purpose is
        # to prove the seam accepts a genuine independent visual read, context
        # is withheld entirely here rather than trusting a weak model to keep
        # "context" and "content" separate.
        prompt = ("Describe only what is visually present in this image in 1-3 factual "
                  "sentences. Do not guess at anything outside the frame. Do not describe "
                  "anything other than what you can actually see in this specific image.")
        try:
            with open(frame.path, "rb") as f:
                image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps({"model": self.model, "prompt": prompt, "images": [image_b64],
                                  "stream": False, "options": {"temperature": 0}}).encode(),
                headers={"Content-Type": "application/json"},
            )
            data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            description = data.get("response", "").strip()
        except Exception as e:
            return VisionObservation(
                timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status="failed",
                model=self.model, analysis_metadata={"error": f"{type(e).__name__}: {e}"},
            )

        if not description:
            return VisionObservation(timestamp_sec=frame.timestamp_sec, frame_path=frame.path,
                                      status="low_information", model=self.model)
        return VisionObservation(
            timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status="ok",
            description=description,
            # moondream's plain-text output carries no calibrated confidence signal
            # of its own -- 0.5 records "a real model genuinely ran" without
            # pretending to a precision the model never reported.
            confidence=0.5, model=self.model,
        )
