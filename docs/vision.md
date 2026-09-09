# Vision Intelligence

Interprets selected video frames -- description, on-screen text, and
located visual elements -- and joins the result with speech and pointer
evidence on the shared timeline. Domain-neutral by construction: nothing
here knows about trading, UI frameworks, or any other specific content type.

Most callers should use `video_lens.analyze_video()` (repository root),
which wires this adapter into the full pipeline automatically
(`vision_enabled`/`vision_model`/`vision_cache_dir` in `PipelineConfig`).
Read on for the adapter's own API if you're calling it directly.

## Provider model

Video-Lens is not built on Claude, or on any one AI provider -- vision
analysis lives entirely behind `core.interfaces.VisionAdapter`, a two-method
Protocol (`analyze_frame(frame, transcript_context, pointer) ->
VisionObservation`) that `core/` depends on and never resolves to a
concrete implementation. `ClaudeVisionAdapter` (this file) is simply the
**default** implementation, used only when `PipelineConfig.vision_provider`
is left unset; `video_lens.py` -- the canonical pipeline -- imports it
lazily, only at the moment it's actually needed, so the pipeline module
itself never hard-depends on `anthropic`.

```python
from video_lens import PipelineConfig, analyze_video

# default: Claude, if configured; gracefully unavailable otherwise
result = analyze_video("video.mp4")

# no vision provider at all
result = analyze_video("video.mp4", PipelineConfig(vision_enabled=False))

# a different provider -- anything implementing VisionAdapter
result = analyze_video("video.mp4", PipelineConfig(vision_provider=my_provider))
```

A future `OpenAIVisionProvider`, `GeminiVisionProvider`, `LocalVLMProvider`,
or a test fake all just need to implement the same two-method shape --
`core/contracts.py`/`core/interfaces.py` never change, and no other part of
the pipeline needs to know which one is running.

```python
from adapters.ingestion import ingest
from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import frames_for_segment
from adapters.pointer.cursor_detector import detect_pointer_for_frames
from adapters.vision.claude_vision import ClaudeVisionAdapter
from core.contracts import MultimodalObservation

video = ingest("tutorial.mp4")
extractor = FrameExtractor()
vision = ClaudeVisionAdapter()  # needs ANTHROPIC_API_KEY (or another credential source)

frames = frames_for_segment(video, transcript.segments[3], extractor)
pointers = detect_pointer_for_frames(frames)

for frame, pointer in zip(frames, pointers):
    context = transcript.text_around(frame.timestamp_sec)
    obs = vision.analyze_frame(frame, transcript_context=context, pointer=pointer)
    evidence = MultimodalObservation(
        timestamp_sec=frame.timestamp_sec, frame=frame, transcript_context=context,
        description=obs.description, pointer=pointer, vision=obs,
    )
```

## Vision backend selected: the Claude API

A hosted multimodal LLM (Claude, via the official `anthropic` Python SDK),
not a local VLM. See the investigation table below for the alternatives
considered. The short version: this machine has no usable GPU (ARM64
Snapdragon, no CUDA path -- same constraint already documented in
`docs/architecture.md`'s Hardware philosophy for transcription), and the
target content (UI text, chart labels, code, small on-screen detail) needs
real OCR-grade reading, not just coarse scene description. A frontier
hosted model reads small screen text and reasons about spatial relationships
far more reliably than a CPU-feasible local model at this project's actual
scale (a handful of selected frames per video, not every frame of a 3-hour
recording), so "smallest mature component that solves the actual problem"
pointed at the API, not a local model.

## Alternatives investigated

| approach | local? | GPU | approx. size | OCR quality | verdict |
|---|---|---|---|---|---|
| **Claude API** (chosen) | no (hosted) | none (client-side) | n/a -- API call | strong; reads small UI/chart text reliably | **USE** |
| Local small VLM (moondream2, ~1.6B) | yes | recommended, CPU possible but slow | ~1.5-3GB weights | weak-to-moderate on dense small text; not built for OCR-heavy screen content | REJECT -- see below |
| Local mid VLM (LLaVA-1.6, Qwen2-VL-7B) | yes | effectively required for usable latency | 7B+ params, ~15GB+ | good, but needs the GPU this project deliberately doesn't require | REJECT -- violates CPU-first philosophy |
| Dedicated OCR library (Tesseract) + separate scene-description model | yes | CPU-fine for OCR | Tesseract ~50MB | good OCR, but no relationship/spatial reasoning, no scene understanding -- would need a second model anyway, and still misses non-text elements (charts, buttons, cursor targets) | REJECT -- solves only half the problem, adds a second dependency for the other half |
| A general object-detection model (YOLO-class) for `elements` | yes | CPU-feasible, GPU faster | 10s of MB | good at generic object boxes, no text reading, no free-text relationship description, fixed class vocabulary (violates domain-neutral requirement) | REJECT -- wrong shape of output for this project |

**Why the local options lost specifically:** the project's own stated
constraint is CPU-first with GPU strictly optional (`docs/architecture.md`).
A local VLM small enough to run acceptably on CPU (moondream-class) is not
strong enough at OCR/spatial reasoning for the target content (screen
recordings full of small text and UI structure); a local VLM strong enough
to be reliable at that (7B+ parameter class) effectively requires a GPU to
run at a usable speed, which this project explicitly avoids requiring.
Splitting OCR and scene-description into two separate local models solves
neither the GPU problem nor the "one coherent structured observation per
frame" goal, and adds a second dependency stack for no net simplicity gain
-- the opposite of "take the piece, not the whole cake." The Claude API
needs zero local compute and zero GPU, at the cost of a network dependency
and per-call cost -- an explicit, documented tradeoff, not a hidden one.

**Cost/latency tradeoff, documented, not hidden:** Video-Lens's frame
selection (Step 4) already exists specifically to avoid sending every frame
anywhere expensive -- `select_keyframes`/`frames_for_segment` typically
reduce a video to single-digit-to-low-double-digit frames per segment or
scene, not thousands. Vision analysis is only ever run on that reduced set,
and results are cached (see below), so the per-call cost of a hosted API is
paid once per unique (frame, context, model) combination, not once per
video-second.

## Keeping the vision layer domain-neutral

The system prompt (`adapters/vision/claude_vision.py`, `_SYSTEM_PROMPT`)
never mentions trading, ICT/SMT/FVG terminology, or any other specific
content type. `VisualElement.kind` is free text the model chooses per frame
("chart", "button", "text", "window", "cursor_target", ...), not a fixed
enum -- there is no vocabulary anywhere in this layer that assumes the
content is financial, educational, or any other single domain. The prompt
explicitly instructs the model not to perform domain-specific
interpretation ("do not diagnose trading setups, medical findings, or
similarly specialized judgments -- describe the visuals, nothing more").
Domain reasoning (e.g. "this is an ICT FVG") is left entirely to a future,
higher-level consumer outside Video-Lens.

## Vision contract

`VisionObservation` (`core/contracts.py`): `timestamp_sec`, `frame_path`,
`status`, `description`, `visible_text`, `elements`, `confidence`, `model`,
`analysis_metadata`.

- `status` -- `"ok"` (real analysis), `"low_information"` (analyzed
  successfully but the frame is blank/near-blank, model said so explicitly),
  `"failed"` (the call/parse broke), `"unavailable"` (no credentials
  configured -- we never even tried). Every field besides `status` is only
  ever populated for `"ok"`/`"low_information"` -- `__post_init__` doesn't
  enforce this the way `PointerEvent` enforces `x`/`y`, because a
  `"failed"`/`"unavailable"` result legitimately carries diagnostic info in
  `analysis_metadata` (the error message) while leaving every content field
  at its empty default; `adapters/vision/claude_vision.py` never populates
  `description`/`visible_text`/`elements` outside `"ok"`/`"low_information"`.
- `confidence` -- validated to `[0, 1]` in `__post_init__` (mirrors
  `PointerEvent`'s validation pattern). The model is required to state a
  confidence; a response missing it is treated as `"failed"`, never
  defaulted to a made-up number.
- `VisualElement`: `kind` (free text), `description`, `region`
  (`Region | None`), `region_confidence` (`"detected"` | `"approximate"` |
  `"unknown"`). `__post_init__` enforces the pairing: `region_confidence`
  must be `"unknown"` exactly when `region is None`, and a located region
  must carry `"detected"` or `"approximate"` -- no state where a region
  exists but its confidence is unstated, or vice versa.
- `Region`: `x1`, `y1`, `x2`, `y2`, normalized `0.0-1.0`, `__post_init__`
  enforces both the range and `x2 > x1`/`y2 > y1`. Never claims pixel-exact
  localization -- Claude's vision responses are box estimates, not a
  detector with measured IoU accuracy, and this project does not claim
  otherwise (see Accuracy honesty below).

## Multimodal evidence: `MultimodalObservation`

The task asked for a "VideoEvidence"-shaped object joining timestamp, frame,
speech context, pointer evidence, and visual observation. `core/contracts.py`
already had exactly this shape since Step 1 -- `MultimodalObservation`
(`timestamp_sec`, `frame`, `transcript_context`, `description`, `pointer`)
-- so it was **extended** with a `vision: VisionObservation | None` field
rather than duplicated under a new name. Per the project's own instruction
("Preserve the existing architecture and contracts unless a genuine
extension is required"), inventing a parallel `VideoEvidence` dataclass with
the same shape would have been redundant, not additive.

## Pointer integration: evidence, not truth

When a `PointerEvent` is passed to `analyze_frame`, its coordinates and
`status`/`confidence` are included in the prompt as one explicit,
labeled fact: `"Detected cursor position (evidence only, status=..., ...)"`.
The system prompt separately instructs the model: pointer coordinates are
approximate evidence of where a cursor was detected, not proof of what is
being referred to; the model may say what's near that location but should
flag ambiguity. A `PointerEvent` with `status == "not_detected"` is **never**
included in the prompt at all -- there is no coordinate to report, so
nothing is sent (verified in `tests/test_vision.py`,
`test_not_detected_pointer_is_never_sent_as_evidence`). The pointer
detector itself is untouched -- this step only ever reads its output.

## Transcript integration: local context only

`transcript_context` (typically `Transcript.text_around(timestamp, window_sec)`
from `docs/speech.md`) is included in the prompt as one line, explicitly
labeled "context only, do not transcribe." The vision layer never sees or
processes the full video transcript for one frame's analysis -- only the
caller-supplied local window, keeping each request small and keeping
Video-Lens's frame-vs-segment relationship (Step 4) intact rather than
re-deriving it inside the vision layer.

## Frame selection: reused, not duplicated

`analyze_frame` takes a `Frame` (Step 4's contract) -- it never extracts
frames itself. A caller drives which frames get analyzed via the existing
selection mechanisms: `frames_for_segment` (speech-linked), `select_keyframes`
(scene-aware, whole-video), or a direct `FrameExtractor.get_frame` call.
Nothing in this step re-implements frame extraction or sampling.

## Cache

`ClaudeVisionAdapter(cache_dir=...)` (default: a `videolens_vision_cache`
folder under the OS temp dir, mirroring `FrameExtractor`'s pattern from
Step 4). Each cache entry is a small JSON file, not an image.

- **Key**: `sha1(abs_frame_path : timestamp_ms : model : prompt_version : transcript_context : pointer_signature)`,
  truncated to 24 hex chars, filename `<hash>.json`. A different model, a
  different transcript context, a different pointer reading, or a bumped
  `_PROMPT_VERSION` constant (incremented whenever the prompt/parsing
  contract changes) all produce a different key -- a cached result never
  silently gets reused for a different configuration (tested:
  `test_different_model_is_a_cache_miss`, `test_different_timestamp_is_a_cache_miss`).
- **Only genuine results are cached** -- `status in ("ok", "low_information")`.
  `"failed"`/`"unavailable"` are deliberately never written to cache, so a
  transient API error or a temporarily-missing credential doesn't poison
  the cache for a retry a moment later (tested: `test_failed_analysis_is_not_cached`).
- **Corrupted entries are never trusted** -- a 0-byte or unparseable cache
  file is treated as a miss and regenerated (same pattern Step 4 established
  for the frame cache), and writes are atomic (`os.replace` from a `.tmp`
  file) so a killed-mid-write process can't leave a corrupt entry in the
  first place -- this step fixes that failure mode proactively rather than
  discovering it the way Step 4 did (tested: `test_corrupted_cache_entry_is_regenerated_not_trusted`).
- **Cleanup**: none, matching Step 4's frame cache. `# ponytail: no
  eviction, cache grows unbounded -- treat the dir as disposable`.

## Model availability

`ClaudeVisionAdapter` degrades gracefully rather than making Video-Lens
depend on a live vision backend to function at all:

- `anthropic` package not installed -> `status="unavailable"`.
- No credentials resolvable (`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` not
  set and no explicit `api_key` passed) -> `status="unavailable"`. This is a
  best-effort precheck covering the common cases, not full parity with the
  SDK's credential-resolution order (an `ant auth login` OAuth profile or
  Workload Identity Federation still work if actually configured -- they
  just report `"unavailable"` here too when no env var is set, and are only
  discovered to work at the real API call).
- A live API error (auth failure, rate limit, network error, malformed
  response) -> `status="failed"`, with the exception folded into
  `analysis_metadata["error"]`, never raised out of `analyze_frame`.

`core/interfaces.py`'s `VisionAdapter` Protocol only names `Frame`/
`PointerEvent`/`VisionObservation` -- no OpenAI/Anthropic-specific type
leaks into core, matching every other adapter boundary in this project. A
future local-VLM or different-API adapter can implement the same Protocol
without touching any caller.

## CPU/GPU requirements

**CPU-only on this machine, no GPU involved at any point.** The vision
computation itself runs entirely on Anthropic's infrastructure; the local
process only reads a JPEG off disk, base64-encodes it, and makes an HTTPS
request. No model weights are downloaded or loaded locally. This preserves
the CPU-first philosophy stated in `docs/architecture.md` exactly as well
as the transcription and frame/pointer layers do, just via a different
mechanism (offloading compute entirely rather than running a small model
locally).

## Real-world smoke test

`scripts/vision_smoke_test.py` was run against the same real 1920x1080
screen-recording tutorial used for Step 5's pointer validation
(`youtube.com/watch?v=upibaDHY36Y`). **This development environment has no
`ANTHROPIC_API_KEY` configured** (confirmed: `ant` CLI is not installed
here, no credential env vars are set, and `api.anthropic.com` returns 401
without one) -- offered to the user as a choice (provide a key now, set one
later, or proceed without live model output); the user chose to proceed
without a live key. The full pipeline was still exercised end to end and
is reported honestly:

```
[1/5] Ingested: 270.0s, 1920x1080, 60fps
[2/5] Transcribed -- segment [26.30s-28.30s] "But before we get to it,"
[3/5] frames_for_segment -> 3 candidate frames
[4/5] Pointer evidence: not_detected at all 3 sampled timestamps (real
      result, per Step 5's documented limitation -- a genuinely static or
      slow-moving cursor at 1s sampling intervals produces no motion signal)
[5/5] Vision analysis: 3/3 status="unavailable" (no credentials) --
      analyzed: 3, ok: 0, low_information: 0, failed: 0, unavailable: 3
      vision cache: 0 entries (unavailable results are correctly never cached)
```

This verifies the entire integration (ingestion -> transcript -> frame
selection -> pointer evidence -> vision call -> joined
`MultimodalObservation`) end to end, including the graceful-degradation
path, without fabricating what a real model response would have said. What
this run does **not** demonstrate is real model output quality (description
accuracy, OCR correctness, region localization) -- that requires an actual
API credential. The script is ready to run as-is (`python
scripts/vision_smoke_test.py <video> [start] [duration] [out_dir]`) the
moment `ANTHROPIC_API_KEY` is set; `tests/test_vision.py`'s mocked-client
tests (`test_successful_analysis_returns_structured_observation`,
`test_low_information_frame_is_distinguished`, etc.) are what verify the
response-parsing and structured-output logic in the meantime, using
synthetic but schema-realistic model responses.

## Accuracy honesty

No object-localization accuracy, OCR accuracy, or pointer-target
association accuracy is claimed or measured in this step -- none of those
were objectively measurable here (no live model output was obtained, and no
ground-truth-labeled real screen recording was available to score against
even if it had been). This is stated as **unmeasured**, not estimated or
assumed. What *is* verified: the contract's structural guarantees (region
bounds, confidence range, status/field consistency, evidence-not-truth
prompt framing for pointer data) via the automated test suite, and the
integration's correctness (real ingestion, real transcript, real frame
selection, real pointer detection, correct joining) via the live smoke run
above.

## Performance

Per-frame cost is dominated by network round-trip + model inference time
for a hosted API call, not measurable in this environment without a live
credential (see above) -- reporting a number here would be fabricated
precision. What's measured: the **rest** of the pipeline leading up to the
vision call (frame selection ~1-2s for a handful of frames, pointer
detection well under a second per frame, both already measured in
`docs/frames.md`/`docs/pointer.md`). Once a credential is available,
`scripts/vision_smoke_test.py` reports total elapsed time and
seconds/frame for the vision step specifically.

**RAM**: each `analyze_frame` call reads one JPEG (typically well under 1MB
at Step 4's `-q:v 2` mjpeg setting) into memory for base64 encoding, then
releases it -- no persistent model weights, no growing memory footprint
with video length or frame count.

## Dependencies

`anthropic` (official Python SDK) -- lives in `requirements-vision.txt`,
**not** `requirements.txt`: it's optional both at install time (core
Video-Lens works without ever installing it) and at runtime
(`ClaudeVisionAdapter` degrades to `status="unavailable"` if it's missing or
unconfigured rather than making the whole library depend on it being
installed and credentialed). No other new dependency; no OCR library, no
local model weights, no additional CV framework.

## Limitations

- **No live model output was validated in this development environment**
  (no API credential available) -- see Real-world smoke test above. This is
  the most significant open item before trusting this adapter's actual
  output quality.
- **Region estimates are not claimed to be pixel-accurate** -- `region`
  reflects the model's own estimate, distinguished only as `"detected"`
  (model is confident) vs `"approximate"` (rough area) vs `"unknown"` (no
  attempt), never validated against ground-truth bounding boxes.
- **The credential precheck is best-effort**, not full parity with the
  `anthropic` SDK's actual resolution order (see Model availability above).
- **No batching** -- each `analyze_frame` call is one API request; a
  caller analyzing many frames pays per-frame network latency serially
  unless it parallelizes calls itself (not built in this step).
- Vision cache has no eviction, same as Step 4's frame cache.
- No click-indicator, chart, or other domain-specific detection was added
  anywhere in this layer -- by design, per Step 6 scope.
