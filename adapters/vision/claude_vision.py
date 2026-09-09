"""Vision adapter backed by the Claude API (multimodal Messages endpoint).
See docs/vision.md for why a hosted multimodal LLM was chosen over a local
VLM/OCR stack, and for the full prompt/cache/failure-mode design.

Domain-neutral by construction: the prompt never mentions trading, charts,
or any other specific content type -- `VisualElement.kind` is free text the
model chooses per frame, not a fixed vocabulary.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile

from core.contracts import Frame, PointerEvent, Region, VisionObservation, VisualElement

DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "videolens_vision_cache")
DEFAULT_MODEL = "claude-sonnet-5"
# Cache-key/prompt version -- bump when the prompt or parsing changes so old
# cached results (from a different prompt contract) are never served as if
# they came from the current one.
_PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = """\
You are the visual-analysis component of a general-purpose video-intelligence \
pipeline. You will be shown one still frame from a video -- it could be a \
screen recording, tutorial, presentation, meeting, coding session, or any \
other content. Describe only what is actually visible.

Respond with ONLY a single JSON object (no markdown fences, no other text), \
matching exactly this schema:

{
  "description": "one or two factual sentences describing the frame",
  "visible_text": ["any legible on-screen text, verbatim", "..."],
  "low_information": false,
  "elements": [
    {
      "kind": "short free-text label, e.g. chart, button, text, window, cursor_target",
      "description": "what this element is",
      "region": [0.1, 0.2, 0.4, 0.5],
      "region_confidence": "detected"
    }
  ],
  "confidence": 0.9
}

Rules:
- "region" is a normalized [x1, y1, x2, y2] bounding box (0.0-1.0, top-left \
origin) or null if you cannot localize the element. "region_confidence" must \
be "detected" (confident, tight box), "approximate" (rough area), or \
"unknown" (only valid when region is null).
- Set "low_information" to true if the frame is blank, near-blank, or \
otherwise carries no useful visual content -- then "elements" may be empty.
- "confidence" (0.0-1.0) is your overall confidence in this analysis. This \
field is required.
- Describe only what is visually present. Do not guess at anything outside \
the frame, and do not perform domain-specific interpretation (e.g. do not \
diagnose trading setups, medical findings, or similarly specialized \
judgments) -- describe the visuals, nothing more.
- If pointer coordinates are given, they are approximate evidence of where a \
cursor was detected, not proof of what is being referred to -- you may note \
what is near that location but should say so is uncertain when it's ambiguous.
- If nearby speech is given, use it only as context for interpreting the \
frame -- do not transcribe or repeat it back.
"""


def _load_anthropic_client(api_key: str | None):
    try:
        import anthropic
    except ImportError:
        return None, "the 'anthropic' package is not installed"

    # Best-effort credential precheck so a plain "no key configured" surfaces
    # as status="unavailable" (we never even tried) rather than "failed" (we
    # tried and it broke) -- the SDK itself only discovers "no credentials"
    # when a request is actually made. This covers the common cases
    # (explicit api_key, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN) but not an
    # `ant auth login` OAuth profile or Workload Identity Federation env
    # vars; those still work, they just report as "unavailable" here too if
    # nothing else is set, then genuinely fail (a clearer "failed") if they
    # turn out not to work at call time.
    # ponytail: env-var precheck only, not full SDK credential-resolution parity
    if not (api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None, ("no Claude API credentials configured -- set ANTHROPIC_API_KEY "
                       "(or ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile)")
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        return None, f"could not construct Anthropic client: {e}"
    return client, None


def _strip_code_fence(text: str) -> str:
    m = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def _build_user_text(frame: Frame, transcript_context: str | None,
                      pointer: PointerEvent | None) -> str:
    parts = [f"Frame timestamp: {frame.timestamp_sec:.3f}s"]
    if transcript_context:
        parts.append(f"Nearby speech (context only, do not transcribe): \"{transcript_context}\"")
    if pointer is not None and pointer.status != "not_detected":
        parts.append(
            f"Detected cursor position (evidence only, status={pointer.status}, "
            f"confidence={pointer.confidence:.2f}): "
            f"normalized ({pointer.normalized_x:.3f}, {pointer.normalized_y:.3f})"
        )
    parts.append("Analyze this frame and respond with the JSON object only.")
    return "\n\n".join(parts)


def _parse_region(raw) -> Region | None:
    if raw is None:
        return None
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
        return Region(x1=x1, y1=y1, x2=x2, y2=y2)
    except (TypeError, ValueError):
        return None


def _parse_response_json(data: dict, frame: Frame, model: str) -> VisionObservation:
    if "description" not in data or "confidence" not in data:
        raise ValueError("response JSON missing required field 'description' or 'confidence'")
    confidence = float(data["confidence"])
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence out of range: {confidence}")

    elements = []
    for raw_el in data.get("elements", []) or []:
        region = _parse_region(raw_el.get("region"))
        region_confidence = raw_el.get("region_confidence", "unknown")
        if region is None:
            region_confidence = "unknown"  # model can't claim a localized region without one
        elif region_confidence not in ("detected", "approximate"):
            region_confidence = "approximate"  # model gave a box but an invalid/missing label
        elements.append(VisualElement(
            kind=str(raw_el.get("kind", "unknown")),
            description=str(raw_el.get("description", "")),
            region=region,
            region_confidence=region_confidence,
        ))

    status = "low_information" if data.get("low_information") else "ok"
    return VisionObservation(
        timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status=status,
        description=str(data["description"]),
        visible_text=tuple(str(t) for t in data.get("visible_text", []) or []),
        elements=tuple(elements), confidence=confidence, model=model,
    )


class ClaudeVisionAdapter:
    """Implements core.interfaces.VisionAdapter using the Claude API."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None,
                 api_key: str | None = None, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._client, self._unavailable_reason = _load_anthropic_client(api_key)

    def analyze_frame(self, frame: Frame, transcript_context: str | None = None,
                       pointer: PointerEvent | None = None) -> VisionObservation:
        cache_key = self._cache_key(frame, transcript_context, pointer)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        if self._client is None:
            return VisionObservation(
                timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status="unavailable",
                model=self.model, analysis_metadata={"reason": self._unavailable_reason},
            )

        try:
            obs = self._call_and_parse(frame, transcript_context, pointer)
        except Exception as e:
            return VisionObservation(
                timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status="failed",
                model=self.model, analysis_metadata={"error": f"{type(e).__name__}: {e}"},
            )

        if obs.status in ("ok", "low_information"):
            self._write_cache(cache_key, obs)  # only cache genuine results, not transient failures
        return obs

    def _call_and_parse(self, frame: Frame, transcript_context, pointer) -> VisionObservation:
        with open(frame.path, "rb") as f:
            image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        media_type = "image/jpeg" if frame.format in ("jpg", "jpeg") else f"image/{frame.format}"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                                  "data": image_b64}},
                    {"type": "text", "text": _build_user_text(frame, transcript_context, pointer)},
                ],
            }],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        data = json.loads(_strip_code_fence(text))
        return _parse_response_json(data, frame, self.model)

    # --------------------------------- cache ---------------------------------

    def _cache_key(self, frame: Frame, transcript_context: str | None,
                    pointer: PointerEvent | None) -> str:
        pointer_sig = "none"
        if pointer is not None:
            pointer_sig = f"{pointer.status}:{pointer.x}:{pointer.y}:{pointer.confidence:.3f}"
        raw = (f"{os.path.abspath(frame.path)}:{frame.timestamp_sec:.3f}:{self.model}:"
               f"{_PROMPT_VERSION}:{transcript_context or ''}:{pointer_sig}")
        return hashlib.sha1(raw.encode()).hexdigest()[:24]

    def _cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _read_cache(self, cache_key: str) -> VisionObservation | None:
        path = self._cache_path(cache_key)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            elements = tuple(
                VisualElement(
                    kind=e["kind"], description=e["description"],
                    region=Region(**e["region"]) if e["region"] else None,
                    region_confidence=e["region_confidence"],
                )
                for e in data["elements"]
            )
            return VisionObservation(
                timestamp_sec=data["timestamp_sec"], frame_path=data["frame_path"],
                status=data["status"], description=data["description"],
                visible_text=tuple(data["visible_text"]), elements=elements,
                confidence=data["confidence"], model=data["model"],
                analysis_metadata={**data.get("analysis_metadata", {}), "cache": "hit"},
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None  # corrupted cache entry -- treat as a miss, regenerate

    def _write_cache(self, cache_key: str, obs: VisionObservation) -> None:
        data = {
            "timestamp_sec": obs.timestamp_sec, "frame_path": obs.frame_path,
            "status": obs.status, "description": obs.description,
            "visible_text": list(obs.visible_text),
            "elements": [
                {
                    "kind": e.kind, "description": e.description,
                    "region": (None if e.region is None else
                               {"x1": e.region.x1, "y1": e.region.y1,
                                "x2": e.region.x2, "y2": e.region.y2}),
                    "region_confidence": e.region_confidence,
                }
                for e in obs.elements
            ],
            "confidence": obs.confidence, "model": obs.model,
            "analysis_metadata": obs.analysis_metadata,
        }
        path = self._cache_path(cache_key)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)  # atomic -- a crash mid-write can't leave a corrupt cache file
