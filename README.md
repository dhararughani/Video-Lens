# Video-Lens

**Give your AI the ability to watch videos.**

[![tests](https://img.shields.io/badge/tests-209%20passing-brightgreen)](#verification)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)

A standalone, domain-agnostic video-intelligence capability bundle: give it
a video (local file or URL), get evidence-grounded structured understanding
back — what was said, what was visible, where the pointer was, and what
Video-Lens itself can honestly infer from that, all traceable back to the
exact timestamps and frames it came from.

```
video  ->  speech  ->  frames  ->  vision  ->  pointer  ->  evidence  ->  knowledge
```

Not a giant video database, not a monolithic AI framework, and not a wrapper
around one specific model — timestamped speech, intelligent frame selection,
scene awareness, pointer context, provider-independent vision analysis,
evidence correlation, and verified semantic synthesis, producing one compact
`KnowledgePackage` your application can actually use.

**Video-Lens is a universal video-intelligence capability bundle. It is
NOT a trading strategy engine, NOT an autonomous trading system, and it
does NOT determine whether a trading strategy is profitable.** It has no
knowledge of trading terminology at all — it describes what's on screen and
what was said, nothing more. Those judgments belong to whatever downstream
system consumes Video-Lens's output; Video-Lens itself stays usable for
trading-mentorship videos, software tutorials, screen recordings, product
demos, coding walkthroughs, presentations, meetings, or any other video
content.

**Video-Lens is also not a Claude wrapper, and not tied to any particular
downstream consumer.** Ingestion, transcription, frame extraction, pointer
detection, evidence correlation, and knowledge packaging have no AI-provider
dependency at all. Vision analysis is the one optional capability that
benefits from a multimodal model, and it lives entirely behind a small
`VisionAdapter` interface (`core/interfaces.py`) — Claude is the built-in
default implementation, not the architecture. See `docs/vision.md`,
"Provider model." Video-Lens runs and is tested standalone; nothing here
imports any downstream consumer's code.

## What problem this solves

Given a video, most "AI video understanding" either (a) sends every frame
to a vision model, which is slow and expensive, or (b) produces a single
freeform summary that can't be traced back to *when* or *where* something
happened, or *whether it was actually observed vs. guessed*. Video-Lens
instead:

- extracts only the frames that matter (scene changes, speech-linked
  moments, uniform sampling — never "every frame")
- keeps everything on one canonical timestamp axis (float seconds from the
  start of the video) so speech, frames, pointer position, and vision
  observations join by plain numeric comparison
- strictly separates **observed** facts (with citable evidence) from
  **inferred** conclusions (always confidence-scored, always cited, never
  presented as fact)
- degrades gracefully — a missing or failed evidence stream never breaks
  the rest of the pipeline, and nothing is ever fabricated to fill a gap
- is temporary by design — a video, its downloaded copy, extracted frames,
  and processing caches are all deleted once a compact, durable
  `KnowledgePackage` has been produced and verified; only that package (and
  the user's own original file, always untouched) survives (see
  `docs/lifecycle.md`)

## Architecture at a glance

```
VIDEO INPUT (local path or URL)
      |
INGESTION            -- normalize to VideoInput (ffprobe metadata)
      |
TRANSCRIPT           -- faster-whisper, CPU, timestamped speech
      |
FRAME SELECTION       -- scene-aware keyframes + speech-linked candidates
      |
POINTER EVIDENCE      -- classical CV cursor detection (three-frame diff)
      |
VISION                -- Claude API: description, visible text, located elements
      |
EVIDENCE CORRELATION  -- deterministic, timestamp-tolerance based
      |
STRUCTURED OBSERVATIONS -- observed / inferred / disagreements / unavailable
      |
ANALYSIS RESULT
      |
SEMANTIC SYNTHESIS    -- optional, provider-neutral; verified against the evidence
      |
KNOWLEDGE PACKAGE     -- compact, evidence-backed, + a small visual-evidence bundle
```

Every stage after ingestion is optional and independently degradable — see
[Partial pipeline / graceful degradation](#partial-pipeline--graceful-degradation)
below. See `docs/architecture.md` for the full component-boundary
breakdown, and each capability's own doc (`docs/speech.md`,
`docs/frames.md`, `docs/pointer.md`, `docs/vision.md`, `docs/evidence.md`)
for how that layer actually works, what was measured, and what its
limitations are.

## Capabilities

| capability | mechanism |
|---|---|
| Local file ingestion | ffprobe validation, no copy into memory |
| URL ingestion | yt-dlp's Python API (hundreds of sites), download-and-cache |
| Speech transcription | faster-whisper (CTranslate2), CPU, word-accurate timestamps |
| Frame extraction | direct ffmpeg seeking + scene-aware keyframe selection + near-duplicate reduction |
| Pointer/cursor detection | classical CV (three-frame differencing), no ML model |
| Vision analysis | Claude API — description, visible text, located elements, confidence |
| Evidence correlation | deterministic timestamp-tolerance joining of all of the above |
| Structured understanding | observed-vs-inferred, provenance, confidence, disagreement recording |
| Knowledge extraction | deterministic cue-phrase/keyword extraction into a compact `KnowledgePackage` |
| Semantic synthesis | **optional**, provider-neutral seam — every returned claim is verified against the evidence before it is retained |
| Visual evidence | demand-driven selection of the few frames worth keeping, near-duplicate collapsing, files beside the package (never base64) |
| Lifecycle / cleanup | job-scoped temp workspace, safety-gated cleanup, never touches the user's own file |

## Installation

```
pip install -r requirements.txt              # core -- no AI provider dependency
pip install -r requirements-vision.txt        # optional: enables the built-in Claude vision provider
```

### Prerequisites

- **Python 3.11+**
- **`ffmpeg`/`ffprobe` on `PATH`** — required for all video/audio
  processing and frame extraction. Not a Python dependency; install it
  separately (e.g. from [ffmpeg.org](https://ffmpeg.org) or your platform's
  package manager).
- A Whisper model (`base`, ~150MB) — downloaded automatically on first
  transcription, cached by `huggingface-hub`. No model file is committed to
  this repository.
- `ANTHROPIC_API_KEY` (or another credential the `anthropic` SDK can
  resolve) — **optional, and only relevant if you installed
  `requirements-vision.txt`**. Without one, the built-in Claude vision
  provider honestly reports `status="unavailable"` for every frame rather
  than failing the rest of the pipeline; without `requirements-vision.txt`
  installed at all, vision still degrades the same way. Set it as an
  environment variable; never commit it. Video-Lens is not tied to
  Claude/Anthropic specifically — see `docs/vision.md`, "Provider model."

### CPU/GPU requirements

**No GPU is required anywhere.** Transcription runs on CPU by default
(`device="cpu"`, int8 quantization); frame extraction and pointer detection
are CPU-only classical CV/subprocess work; vision analysis runs on
Anthropic's infrastructure via API call, not locally. `device="cuda"` is
available to anyone who has a GPU for transcription, but nothing in this
project requires one, and no GPU-only dependency is ever installed.

## Usage

### Standalone knowledge extraction (recommended for most callers)

```python
from video_lens import process_video

package = process_video("path/to/video.mp4")   # or a URL
print(package.summary)
for kp in package.key_lessons:
    print(kp.kind, kp.timestamp_sec, kp.text)
```

`process_video(source, config=None) -> KnowledgePackage` runs the full
lifecycle: pipeline -> knowledge extraction -> validation -> durable,
schema-versioned JSON output (the handoff artifact) -> temporary-artifact
cleanup (only after successful export). It's the right entry point when you
want compact, durable knowledge rather than the full evidence-level result
and your own storage management. **This works completely standalone — no
particular downstream consumer, no database, no HTTP service, and no
specific AI model are ever required.** The video itself, extracted frames,
and any URL-downloaded copy are not permanently stored — only the compact
`KnowledgePackage` survives. See `docs/lifecycle.md` for the full lifecycle,
cleanup safety gate, schema-version/validation contract, and storage
measurements.

`KnowledgePackage` is a generic handoff artifact, not shaped around any one
consumer — some downstream memory/knowledge system is one possible reader
of it, not a dependency: Video-Lens never imports a downstream consumer's
code, and nothing here knows how any consumer stores, indexes, or retrieves
knowledge. A consumer reads the plain JSON (`core.knowledge.to_dict`/
`to_json`) and decides what to do with it.

### The canonical pipeline (advanced / full evidence)

```python
from video_lens import analyze_video, PipelineConfig

result = analyze_video("path/to/video.mp4")          # local file
result = analyze_video("https://youtube.com/watch?v=...")  # URL

for so in result.structured_observations:
    print(so.timestamp_sec, [e.ref for e in so.observed])
    for inf in so.inferences:
        print(" inferred:", inf.text, inf.confidence, inf.basis)
    print(" missing:", so.unavailable)
```

`analyze_video(source, config=None) -> AnalysisResult` is the single public
entry point — `video_lens.py` at the repository root. A caller never needs
to know about `core/`/`adapters/`'s internal layout; import from
`video_lens` only.

`PipelineConfig` centralizes every knob a caller is actually likely to
want to change — every field has a default that works with zero
configuration:

```python
from video_lens import PipelineConfig

config = PipelineConfig(
    whisper_model_size="base",       # or "small"/"medium"/"large-v3" for more accuracy
    keyframe_interval_sec=2.0,       # frame-selection density
    max_frames=20,                   # ceiling on frames sent through pointer/vision
    pointer_enabled=True,
    vision_enabled=True,             # set False to skip vision-provider calls entirely
    vision_model=None,               # None -> the provider's own default model
    vision_provider=None,            # None -> built-in Claude provider; or pass your own VisionAdapter
    tolerance_sec=1.0,               # evidence-correlation window
    frame_cache_dir=None,            # None -> OS temp dir default
    vision_cache_dir=None,
    download_dir=None,
)
result = analyze_video("video.mp4", config)
```

`vision_provider` is the seam for swapping in a different backend (OpenAI,
Gemini, a local VLM, a test fake) without changing any core contract — see
`docs/vision.md`, "Provider model."

### Individual adapters (advanced use)

Each capability is independently usable — see the corresponding doc for
its own API:

```python
from adapters.ingestion import ingest
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import frames_for_segment, select_keyframes
from adapters.pointer.cursor_detector import detect_pointer_for_frames, track_pointer
from adapters.vision.claude_vision import ClaudeVisionAdapter
from core.evidence import build_structured_observation, build_analysis_result
```

## Output structure

`AnalysisResult` (`core/contracts.py`):

- `video` — the resolved `VideoInput` (metadata).
- `structured_observations` — the Step 7 answer to "what happened at this
  point in the video": each `StructuredObservation` has `observed` (a tuple
  of citable `Evidence`), `inferences` (Video-Lens's own conclusions,
  always cited and confidence-scored), `disagreements` (conflicts/gaps
  recorded, never silently resolved), and `unavailable` (which evidence
  streams had nothing in this window).
- `observations`/`evidence` — the flatter Step 1-6 shape, kept for
  compatibility; every field of `structured_observations` is also
  reachable through these.

See `docs/evidence.md` for the full contract reference, the correlation
tolerance model, and worked examples.

## Partial pipeline / graceful degradation

Every combination below is a supported, tested pipeline state — a missing
capability is recorded as `unavailable` in the result, never fabricated,
and never breaks an unrelated capability:

```
video + speech + frames + pointer + vision   (full pipeline)
video + speech + frames                       (vision_enabled=False, or no API key)
video + frames                                (silent video, or no speech detected)
video + speech                                (frame extraction failed/disabled)
video + frames + vision                       (no speech in the audio track)
video only                                    (every optional stage disabled/unavailable)
```

Ingestion is the one stage that is *not* optional — a missing/corrupt/
unreadable video raises `VideoIngestionError` immediately, since nothing
downstream is possible without it. Every later stage failure prints one
warning line to stderr and continues (see `video_lens.py`).

## Cache behavior

Every cache follows the same pattern: filesystem-based (no database), keyed
so a source/config change automatically invalidates old entries, corrupted
or empty entries are never trusted (regenerated instead), and there is
**no automatic eviction** — caches default to the OS temp directory and are
treated as disposable; clear them manually if disk usage matters for a
long-running process.

| cache | default location | keyed on |
|---|---|---|
| Frame cache | `%TEMP%/videolens_frame_cache` | video path + mtime + size + timestamp |
| Vision cache | `%TEMP%/videolens_vision_cache` | frame + timestamp + model + prompt version + context + pointer |
| URL download cache | `%TEMP%/video_lens_downloads` | yt-dlp video id (reused across runs, never re-downloaded) |

Only genuine results are cached — a transient vision API failure or a
temporarily missing credential is retried on the next call, never
permanently remembered as a failure (see `docs/vision.md`).

## Limitations

- **`ClaudeVisionAdapter` (the built-in Claude-backed provider) has still not
  been validated against a real Claude API call** (no `ANTHROPIC_API_KEY` has
  ever been available in this development environment). Its pipeline wiring —
  prompt construction, response parsing, caching, evidence correlation — is
  verified with 20+ tests against a mocked client and one real end-to-end
  pipeline run producing honest `status="unavailable"` results. Real
  `ClaudeVisionAdapter` quality is a pending external validation step for
  whoever first runs this with a live key.
- **The `VisionAdapter` seam itself HAS been validated live** (Step 12):
  a real, independent, non-Claude, non-mocked local vision model (Ollama +
  `moondream`) was run against real extracted frames from a real 258s
  tutorial through the exact same seam a caller would use. It produced
  genuinely independent visual reads — some accurate, some wrong — proving
  the architecture accepts a real provider and correctly distinguishes
  interpreted `vision` evidence from a bare, uninterpreted `frame` (see
  `docs/synthesis.md`, "What verification cannot do"). This is evidence the
  *seam* works, not evidence `ClaudeVisionAdapter` specifically works.
- **Pointer/cursor detection has real, measured limits**: a stationary
  cursor produces no motion signal and detects as `not_detected` (never
  interpret `not_detected` as "no cursor present" — it means "no motion
  evidence in this window," see `docs/pointer.md`); real-screen-recording
  detection rate (~15% measured) is substantially lower than the
  synthetic-ground-truth rate (~80%); no dedicated click-indicator/ripple
  classifier exists.
- Frame/vision caches have no eviction — long-running processes should
  periodically clear them or pass a scoped `cache_dir`.
- **Knowledge extraction (`core/knowledge.py`) is deterministic cue-phrase
  and word-frequency matching, not semantic language understanding** — it
  can miss real content that doesn't use a recognized phrase, or flag
  something as notable that isn't. No LLM call is made or required; see
  `docs/lifecycle.md`. Semantic understanding is available only when you
  supply a `KnowledgeSynthesizer` (`docs/synthesis.md`); Video-Lens ships
  none, and without one `semantic_summary`/`claims` stay empty rather than
  being backfilled from the deterministic output.
- **Semantic claim quality is bounded by the provider you supply, and
  verification cannot catch a provider that is confidently wrong.**
  Video-Lens verifies provenance, timestamps, evidence existence and
  confidence bounds — it cannot verify that a well-cited claim, or the
  `vision` description it cites, is an *accurate* reading of the evidence.
  Confirmed directly in Step 12: a live local vision model correctly
  described some real frames and confidently misdescribed others (a
  screenshot of two Git GUI clients as "a webpage about HTML"), and
  verification correctly retained a claim citing the wrong description —
  because the citation, timestamp, and image were all genuinely real. Doing
  better would require a second model to grade the first, which Video-Lens
  does not do. A claim citing a bare (uninterpreted) frame is still marked
  as illustrated-but-not-corroborated, distinct from this limitation.
- An orphaned job workspace from a genuinely failed `process_video()` run
  is not swept automatically (same no-eviction policy as the other
  caches) — see `docs/lifecycle.md`'s Cleanup safety gate.
- No trading-domain interpretation exists anywhere in this codebase, by
  design — see the notice at the top of this document.

## Release classification (v1.0)

Honest, per-capability status as of the Step 12 final validation pass —
not "tests pass," but what was actually inspected against real output.

| capability | status | basis |
|---|---|---|
| Local file ingestion | **READY** | real local videos processed end-to-end, source untouched |
| URL ingestion | **READY** | real YouTube URL processed end-to-end via `process_video`, job-owned download cleaned |
| Speech transcription | **READY** | real CPU transcription inspected against real speech, correct segmentation/timestamps |
| Frame intelligence | **READY** | real scene-aware selection inspected against real video content |
| Pointer detection | **FUNCTIONAL BUT LIMITED** | real detection run on real screen recording; ~15% real-world hit rate, honestly disclosed, unchanged by design |
| Vision — seam/architecture | **READY** | live, independent, non-Claude model run through the real seam against real frames |
| Vision — `ClaudeVisionAdapter` | **NOT VALIDATED** | never run against a live Claude API key in this environment |
| Evidence correlation | **READY** | real observed/inferred/disagreement/unavailable output inspected |
| Semantic synthesis (seam + verification) | **READY** | real adversarial claims rejected against real evidence; real corroborated claims correctly labelled |
| Visual corroboration | **READY WITH LIMITATIONS** | genuinely tested live; verification cannot detect a provider that is confidently wrong, disclosed above |
| KnowledgePackage / schema | **READY** | validated, compact, evidence-backed, no embedded media, schema-versioned |
| Lifecycle / cleanup | **READY** | real jobs measured: source untouched, job workspace fully removed after success, nothing removed on failure |
| Provider independence | **READY** | `core/` contains zero Claude/Anthropic/OpenAI/Ollama references outside documentation prose; tests block-import each provider package and still pass |
| Downstream-consumer independence | **READY** | zero `import dhara` anywhere; nothing here writes to any downstream consumer's storage |

## What Video-Lens deliberately does NOT do

- No trading strategy interpretation, ICT/FVG/SMT detection, or any other
  domain-specific reasoning.
- No autonomous trading, no trade recommendations, no profitability
  judgments.
- No HTTP server — this is a Python library, integrated by import.
- No vector database, RAG pipeline, or agent/orchestration framework —
  unjustified at this project's actual scale (see `docs/evidence.md`,
  `docs/component-decisions.md`).
- No local vision model / GPU requirement — vision runs via the Claude API
  specifically to avoid needing one (see `docs/vision.md`).
- No bundled language model and no Ollama (or other runtime) subsystem —
  semantic synthesis is a seam you plug into, not machinery Video-Lens owns.
  If your system already runs Ollama, the adapter belongs on your side of
  that seam (see `docs/synthesis.md`).
- No permanent media archive — the durable output is a knowledge package
  plus, at most, a handful of selected evidence frames as ordinary files.
- No live OS mouse tracking (`pynput`-style) — pointer detection reads
  cursor pixels already baked into recorded video, a different problem
  entirely (see `docs/pointer.md`).

## Documentation

- `docs/architecture.md` — scope, non-goals, component boundaries, hardware
  philosophy, what was deliberately excluded.
- `docs/component-decisions.md` — every mature external component
  considered, what was adopted vs. rejected, and why.
- `docs/ingestion.md`, `docs/speech.md`, `docs/frames.md`, `docs/pointer.md`,
  `docs/vision.md`, `docs/evidence.md` — one doc per capability layer,
  each with its own contract reference, measured performance, real-video
  test results, and honestly-stated limitations.
- `docs/synthesis.md` — the optional semantic-synthesis seam: how a provider
  is supplied, exactly what is verified before a claim is retained, the four
  claim states, the visual cross-check, how evidence frames are selected and
  de-duplicated, and the honest fallback when no synthesizer exists.
- `docs/lifecycle.md` — the temporary-workspace / knowledge-package
  lifecycle: what's kept vs. deleted, the cleanup safety gate, the
  schema-version and validation contract, storage measurements, and the
  generic handoff/export contract (any downstream system is a possible
  consumer of it, not a dependency).

## Verification

```
python tests/test_contracts.py
python tests/test_ingestion.py           # unit tests, no network
python tests/test_speech.py              # speech contracts + transcription
python tests/test_frames.py              # frame extraction, selection, cache
python tests/test_pointer.py             # pointer detection, tracking, false-positive checks
python tests/test_vision.py              # vision contracts, mocked-model analysis, cache, evidence joining
python tests/test_evidence.py            # evidence correlation, tolerance, observation/inference, degradation
python tests/test_pipeline.py            # canonical pipeline: config, degradation, one full local run
python tests/test_knowledge.py           # knowledge extraction: compactness, provenance, degradation, domain-neutrality
python tests/test_lifecycle.py           # job workspace, cleanup safety gate, local-source protection, failure paths
python tests/test_provider_independence.py  # no Claude/Anthropic/downstream-consumer coupling, fake provider injection
python tests/test_handoff_contract.py    # schema version, observed/inferred, validation, export, portability

python scripts/smoke_test.py <video_path> [out_dir]
python scripts/speech_smoke_test.py <video_path_or_url> [model_size]
python scripts/frame_smoke_test.py <video_path_or_url> [out_dir]
python scripts/pointer_smoke_test.py <video_path_or_url> [start_sec] [duration_sec] [out_dir]
python scripts/vision_smoke_test.py <video_path_or_url> [start_sec] [duration_sec] [out_dir]  # needs ANTHROPIC_API_KEY for real results
python scripts/evidence_smoke_test.py <video_path_or_url> [start_sec] [duration_sec] [out_dir]  # works with or without ANTHROPIC_API_KEY
python scripts/pipeline_smoke_test.py <video_path_or_url> [max_frames]  # the canonical end-to-end pipeline
python scripts/lifecycle_smoke_test.py <video_path> [output_dir]  # Step 9: process_video() + storage measurement
python scripts/url_smoke_test.py [url]   # needs network
```

No HTTP interface exists, and none is planned — Video-Lens is a Python
library, not a service.

## Layout

- `video_lens.py` — **the public API**: `process_video()` (Step 9
  lifecycle, recommended), `analyze_video()` (raw evidence-level pipeline),
  `PipelineConfig`. Import from here for normal use.
- `core/` — data contracts (`contracts.py`), adapter interfaces
  (`interfaces.py`, `Protocol`s), `errors.py`, and shared mechanisms:
  `video_metadata.py` (ffprobe metadata every adapter needs),
  `evidence.py` (deterministic evidence correlation/structuring),
  `knowledge.py` (Step 9 deterministic knowledge extraction/rendering), and
  `workspace.py` (Step 9 job-scoped temp workspace + cleanup safety gate).
  No concrete ML/CV/download library imports here.
- `adapters/` — one implementation per external capability (ingestion,
  transcription, frames, pointer, vision), each behind its core interface
  and independently swappable.
- `tests/` — unit and integration tests (no network required, except
  `test_ingestion.py`'s URL-parsing checks and the optional live smoke
  scripts under `scripts/`).
- `scripts/` — verification/smoke-test scripts, including the canonical
  end-to-end pipeline check (`pipeline_smoke_test.py`) and a live URL
  smoke test.
- `docs/` — architecture, component decisions, and one doc per capability.
- `config/` — reserved for future adapter configuration files; empty by
  design (`PipelineConfig` in `video_lens.py` is the actual configuration
  surface today).

## Portability

No machine-specific paths are hard-coded anywhere in this repository —
every cache/download location defaults to the OS temp directory
(`tempfile.gettempdir()`) and every filesystem operation uses relative or
caller-supplied paths. Cloning this repository to a different machine or
directory requires only the documented external prerequisites (Python,
ffmpeg, optionally an Anthropic API key) — no repository-specific setup.
