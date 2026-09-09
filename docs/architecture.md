# Video-Lens Architecture

## Purpose

A standalone, domain-agnostic video-intelligence capability: given a video,
produce evidence-grounded structured understanding (speech + relevant frames
+ pointer/timing context). Not built for any specific consumer — some
downstream application consumes it as an external library later.

## Scope

**Step 1** established contracts, adapter boundaries, and proof that the
chosen mature components work together on a real video with real
timestamps. **Step 2** added ingestion (local file + URL → normalized
`VideoInput`). **Step 3** made speech a first-class, queryable temporal
layer (see `docs/speech.md`). **Step 4** added frame intelligence: direct
timestamp extraction, speech-to-frame bridging, scene-aware keyframe
selection, and near-duplicate reduction (see `docs/frames.md`). **Step 5**
added pointer/cursor intelligence: classical, motion-based detection of a
cursor baked into video pixels via three-frame differencing, with an
honest `detected`/`uncertain`/`not_detected` confidence model (see
`docs/pointer.md`). **Step 6** added multimodal vision intelligence: a
Claude-API-backed `VisionAdapter` that turns a selected `Frame` (+ optional
nearby transcript and pointer evidence) into a structured
`VisionObservation` (description, visible text, located elements),
joined with speech and pointer evidence into `MultimodalObservation`
(see `docs/vision.md`). **Step 7** added evidence correlation and
structured understanding: deterministic, timestamp-tolerance-based
correlation of all four evidence streams into `StructuredObservation`
records that strictly separate evidence-backed `observed` facts from
Video-Lens's own `inferences` (always cited, always confidence-scored),
record `disagreements` between streams instead of silently resolving them,
and degrade to `unavailable` rather than guessing when a stream is missing
(see `docs/evidence.md`). **Step 8** is release hardening, not a new
capability: a canonical pipeline entry point (`video_lens.analyze_video`)
composing every existing adapter with no duplicated logic, a centralized
`PipelineConfig`, a full architecture/portability/dependency/git-hygiene
audit, and a concrete bug fix (see Configuration below) discovered during
that audit. No new architecture phase follows this one for the standalone
Video-Lens bundle.

The canonical time representation across every contract is **float seconds
from the start of the video**, so speech, frames, pointer events, vision
observations, and now correlated structured observations all join by plain
numeric comparison. See the TIME MODEL section of `docs/speech.md`, the
Canonical timestamp section of `docs/frames.md`, the Canonical timestamp
section of `docs/pointer.md`, `docs/vision.md`, and the Timestamp tolerance
section of `docs/evidence.md`.

## Non-goals

Not a general AI agent, not a computer-use/screen-control system, not a
vector DB or RAG platform, not a video knowledge base, not a trading system,
not a model-training platform, not a streaming platform, not a generic
automation framework. If a future step starts pulling in any of these
shapes, that's scope creep — stop and reconsider.

## Architecture

```
Video
 ↓
Ingestion         (URL/file -> local VideoInput)      -- IngestionAdapter (local, url)
 ↓
Speech            (VideoInput -> Transcript)           -- TranscriptionAdapter
 ↓
Frames            (VideoInput -> [Frame])               -- FrameAdapter
 ↓
Pointer           (Frame(s) -> PointerEvent)             -- PointerAdapter
 ↓
Vision            (Frame + context + pointer -> VisionObservation) -- VisionAdapter
 ↓
Correlation       (all streams, timestamp + tolerance -> StructuredObservation) -- core/evidence.py
 ↓
Evidence          (AnalysisResult: structured_observations + observations + evidence, all timestamped)
```

`core/contracts.py` defines the data shapes above. `core/interfaces.py`
defines the four adapter Protocols. Core code never imports a concrete
library directly — only these contracts and interfaces. This is what lets
Whisper be swapped for a cloud STT API, or PySceneDetect for a different
frame-selection strategy, without touching core.

## Component boundaries

- `video_lens.py` (repository root) — the public API (`analyze_video`,
  `PipelineConfig`). The only module a downstream integration should import
  from; see Canonical pipeline entry point below.
- `core/` — contracts, interfaces, `errors.py`, and two truly shared
  mechanisms: `video_metadata.py` (every adapter needs ffprobe metadata) and
  `evidence.py` (Step 7's deterministic evidence correlation/structuring --
  pure functions over contracts, no external library, so it belongs in
  `core/` rather than a new `adapters/` package; see `docs/evidence.md`). No
  external ML/CV/download library imports here.
- `adapters/ingestion/`, `adapters/transcription/`, `adapters/frames/`,
  `adapters/pointer/`, `adapters/vision/` — one implementation per external
  capability. Each adapter wraps exactly one mechanism (see
  `docs/component-decisions.md`) and implements the matching Protocol.
  `adapters/ingestion/__init__.py` is the one exception to "adapters don't
  compose adapters" — it's a thin dispatch facade (local vs. URL), not a
  new capability of its own.
- `tests/` — contract- and ingestion-level self-checks (unit only, no network).
- `scripts/` — one-off verification scripts (local smoke test, live URL
  smoke test), not application code.

## Adapter strategy

Every adapter is swappable because core depends on `Protocol`s, not classes.
Concretely in this step:

- `FasterWhisperAdapter` implements `TranscriptionAdapter` using
  faster-whisper on CPU (Step 3 replaced the ffmpeg whisper filter used in
  Steps 1–2, which had unbounded timestamp drift — see
  `docs/component-decisions.md` and `docs/speech.md`).
- `SceneDetectFrameAdapter` implements `FrameAdapter` using PySceneDetect for
  boundary detection + ffmpeg for the actual pixel extraction. Its scene
  detection is exposed separately as `detect_scene_timestamps()` so Step 4's
  frame-intelligence layer reuses it rather than duplicating it.
- `FrameExtractor` (`adapters/frames/frame_extractor.py`) does direct
  timestamp-based extraction via ffmpeg input seeking, with a filesystem
  cache. `adapters/frames/keyframe_selector.py` builds on both of the above
  for the speech→frame bridge, whole-video keyframe selection, and
  near-duplicate reduction (see `docs/frames.md`).
- `adapters/pointer/cursor_detector.py` implements `PointerAdapter` using
  classical three-frame differencing (OpenCV) for motion-based cursor
  detection, with size/solidity/scene-motion/trajectory-continuity
  filtering and an honest `detected`/`uncertain`/`not_detected` status (see
  `docs/pointer.md`).
- `adapters/vision/claude_vision.py`'s `ClaudeVisionAdapter` implements
  `VisionAdapter` using the Claude API (hosted multimodal LLM, no local
  model weights or GPU) -- structured JSON output parsed into
  `VisionObservation`, with pointer/transcript context folded into the
  prompt as labeled evidence, a filesystem JSON cache, and graceful
  `unavailable`/`failed` degradation when no credential is configured or a
  call errors (see `docs/vision.md`).
- `LocalIngestionAdapter` and `URLIngestionAdapter` implement
  `IngestionAdapter` (added in Step 2 — see `docs/ingestion.md`);
  `URLIngestionAdapter` wraps yt-dlp's Python API, never its CLI.
- `core/evidence.py`'s `build_structured_observation`/`build_analysis_result`
  are not adapters (nothing external to wrap) -- they correlate whatever
  `Transcript`/`Frame`/`PointerEvent`/`VisionObservation` data a caller
  already has, entirely deterministically, into `StructuredObservation`/
  `AnalysisResult` (see `docs/evidence.md`).

## Hardware philosophy

CPU-first. No architecture decision in Step 1 requires a GPU:

- FFmpeg decode/encode: CPU, verified.
- Frame extraction: CPU, verified. Each extraction is a short-lived ffmpeg
  subprocess (~30MB working set, measured) that exits after writing one
  frame — memory-safe by construction, independent of video length.
- PySceneDetect: pure Python + numpy, CPU, verified.
- Pointer detection: OpenCV classical CV (`absdiff`/`threshold`/
  `findContours`), CPU-only, no training data or GPU path, verified against
  a synthetic ground-truth trajectory and a real 1920x1080 screen recording.
- Vision: no local computation at all -- the Claude API runs inference on
  Anthropic's infrastructure; the local process only reads/encodes a JPEG
  and makes an HTTPS call, so this layer is CPU-only (and GPU-free) by
  construction, not by tuning. See `docs/vision.md`.
- Evidence correlation: pure Python over already-computed dataclasses, no
  model inference of any kind, no I/O beyond what Steps 3-6 already did --
  the cheapest layer in the whole pipeline. See `docs/evidence.md`.
- Transcription: faster-whisper on `device="cpu"` with int8 (verified
  CPU-only on this machine, which has no usable GPU — ARM64 Qualcomm with an
  Adreno integrated GPU, no CUDA path). 2.4x realtime, 451MB peak RAM, flat
  with video length. `device="cuda"` is available to anyone who has a GPU.

Optional acceleration paths remain available without being required: the
whisper filter's `use_gpu` flag, or (as of Step 6) the vision layer's cloud
API adapter, which offloads compute entirely rather than needing local
acceleration at all. Local GPU models are an optional adapter choice later,
never a baseline requirement.

## Canonical pipeline entry point

`video_lens.py` (repository root) is the public API a downstream project
should actually import from -- `analyze_video(source, config=None) ->
AnalysisResult`. It composes the existing adapters in sequence (ingestion
-> transcription -> keyframe selection -> pointer -> vision -> `core/evidence.py`
correlation) without reimplementing any of their logic; every adapter call
is the exact same call a script in `scripts/` makes when exercising that
adapter alone. Ingestion is the only stage that raises on failure -- every
later stage catches its own adapter's documented failure type
(`TranscriptionError`, `FrameExtractionError`), prints one warning to
stderr, and continues with that evidence stream simply absent (see
`docs/evidence.md`'s Graceful degradation section, which this module is the
concrete realization of).

### Configuration

`PipelineConfig` (in `video_lens.py`) centralizes the knobs a caller is
actually likely to want to change -- model sizes/devices, frame-selection
density, the evidence-correlation tolerance, cache locations, per-stage
enable/disable flags. It deliberately does **not** expose every constant in
the codebase: the pointer detector's low-level CV thresholds
(`adapters/pointer/cursor_detector.py` -- contour size/solidity/scene-motion
fractions) stay internal module constants, already grouped and documented
at the top of that one file, not scattered. Making six raw geometry
thresholds part of a public config surface with no concrete caller need for
per-video tuning would be the "new research project" Step 8 was explicitly
told not to start -- so that line was deliberately not crossed. Every
`PipelineConfig` field has a default that works with zero configuration.

**Bug found and fixed during the Step 8 audit:** the first version of
`analyze_video` fed `select_keyframes()`'s output (frames typically seconds
apart -- scene changes / interval sampling) directly into
`detect_pointer_for_frames()`, which needs temporally *adjacent* frames for
three-frame differencing to see any motion at all. Pointer evidence was
therefore silently near-empty in the full pipeline regardless of how much
real cursor motion the source video had -- confirmed by running the
pipeline against a real video (0 pointer detections across 5 keyframes) and
then against the Step 5 ground-truth moving-cursor video with the same
sparse-frame-list call pattern. Fixed by calling `detect_pointer_at()`
(Step 5's own per-timestamp API, which extracts its own closely-spaced
adjacent frames) once per selected keyframe instead -- verified against the
same ground-truth video (pixel-exact detections recovered) and covered by
`tests/test_pipeline.py`'s `test_pointer_stage_finds_real_motion_against_known_trajectory`
regression test.

## Why each component was selected

See `docs/component-decisions.md` for the full per-component breakdown
(maturity, license, what's used vs. rejected, decision, reasoning).

## What was deliberately excluded

- `openai-whisper` — same model weights as faster-whisper but with a
  multi-GB PyTorch dependency and slower CPU inference, for no accuracy gain.
- Native captions/subtitles as a timestamp source — evaluated in Step 3 and
  deferred: they only exist for URL sources, and their *display* timings
  would corrupt the single consistent timeline (`docs/speech.md`).
- Any live-cursor-tracking library (`pynput`, etc.) — wrong problem entirely;
  those track a live OS cursor, not a cursor baked into recorded pixels.
- The full `bradautomates/claude-video` repository — only worth revisiting
  for a design pattern when the Vision adapter is actually built, never as a
  dependency.
- A trained ML cursor-detection model — classical three-frame differencing
  reached pixel-exact synthetic accuracy and zero false positives on every
  tested trap without one; see `docs/pointer.md`.
- Template matching against known cursor icon shapes, and optical flow —
  both investigated for pointer detection and rejected (icon templates don't
  generalize across OS/theme/scale; optical flow is denser/costlier than the
  three-frame-differencing approach already gives for this step's scope).
  See `docs/pointer.md`.
- A dedicated click-indicator (ripple/ring) shape classifier — a
  cursor-highlight ring in the real test video was caught incidentally by
  ordinary motion detection; a general ring detector was not built.
- Any VLM/vision model for frame *selection* — Step 4's speech→frame bridge
  and keyframe selection are deterministic temporal/pixel-diff logic only.
  Vision models are for a later step to consume the frames this step
  selects, not to help select them.
- An image-hashing library (`imagehash`, OpenCV) for near-duplicate
  detection — an average-hash was hand-implemented first and rejected for a
  real blind spot (flat/solid-color frames all hash identically regardless
  of color); a raw 8x8 grayscale thumbnail diff replaced it, still with no
  new dependency. See `docs/frames.md`.
- A local VLM (moondream-class or larger) for vision analysis — either too
  weak at OCR/spatial reasoning for this project's screen-recording-heavy
  target content, or (7B+ parameter class) effectively requiring a GPU this
  project deliberately doesn't require. See `docs/vision.md`.
- A separate OCR library (e.g. Tesseract) plus a separate local
  scene-description model — solves only half the "structured observation"
  problem each, and adds a second local dependency stack for no net
  simplicity gain over a single hosted multimodal call. See `docs/vision.md`.
- Any domain vocabulary (trading terms, UI-framework-specific element
  types, etc.) in the vision prompt or `VisualElement.kind` — kept free-text
  and domain-neutral; the vision layer describes what's visible, it never
  interprets it. See `docs/vision.md`.
- An LLM call for evidence correlation/structuring — all Step 7 correlation
  rules are deterministic, testable offline, and reproducible byte-for-byte
  given the same input; adding a model call here would make "what happened
  at this point" non-reproducible for no benefit. See `docs/evidence.md`.
- A vector database, RAG pipeline, or agent/orchestration framework for
  Step 7 — evidence correlation is a handful of pure functions over
  in-memory dataclasses; none of that infrastructure is needed at this
  project's actual scale (a handful of evidence items per query timestamp).
- A new `VideoEvidence`-named parallel result type — the existing
  `AnalysisResult`/`MultimodalObservation` scaffold (present since Step 1)
  already represented the right concept; extended with
  `structured_observations` rather than duplicated. See `docs/evidence.md`.
