# Lifecycle

"Artifacts are temporary. Knowledge is permanent."

Video-Lens is a temporary multimodal-understanding pipeline, not a video
archive. This document explains what happens when you give it a video,
in plain language, and what survives afterward.

## What happens when I give Video-Lens a video?

```python
from video_lens import process_video

package = process_video("lesson.mp4")          # or a URL
```

1. Video-Lens reads (local file) or downloads (URL) the video into a
   **job-scoped temporary workspace** — a fresh, uniquely-named directory
   under the OS temp dir, owned entirely by this one job.
2. It transcribes speech, extracts a handful of relevant frames, and looks
   for pointer/cursor evidence — all within that temporary workspace.
3. It correlates everything it found into evidence-grounded
   `StructuredObservation`s (Step 7 — see `docs/evidence.md`), exactly as
   `analyze_video()` already does.
4. It extracts **compact knowledge** from that evidence — topics, key
   lessons, important observations — deterministically (see
   [Knowledge extraction](#knowledge-extraction) below).
5. It **preserves provenance**: every extracted point cites the evidence
   (transcript text, timestamp) it came from.
6. It **verifies** the resulting package (structurally sane, has processing
   metadata) before treating it as real output.
7. It writes that package to durable storage (a small JSON file) and
   **only then** deletes the temporary workspace it owns — the downloaded
   video (if any), extracted frames, and processing caches.
8. It leaves the user's own local video file **completely untouched** —
   Video-Lens never owns a local source, so nothing ever deletes it.
9. It does not become a video archive: nothing frame- or video-sized is
   ever kept once a job finishes successfully.

If anything fails at any point before step 7 finishes, **nothing is
cleaned up** — the temporary workspace stays on disk, so a failed run can
be inspected. See [Cleanup safety gate](#cleanup-safety-gate).

## The lifecycle, concretely

```
SOURCE VIDEO / URL
       |
JOB-SCOPED TEMP WORKSPACE   (core/workspace.py: JobWorkspace)
       |
INGEST -> TRANSCRIBE -> EXTRACT FRAMES -> DETECT POINTER -> [VISION IF ENABLED]
       |                                        (video_lens._run_pipeline --
       |                                         the exact same pipeline body
       |                                         analyze_video() runs)
CORRELATE EVIDENCE          (core/evidence.py -- unchanged from Step 7)
       |
[SEMANTIC SYNTHESIS]        (core/synthesis.py -- ONLY if a KnowledgeSynthesizer
       |                      was supplied; provider output is verified against
       |                      the evidence before any of it is retained.
       |                      See docs/synthesis.md)
       |
EXTRACT KNOWLEDGE           (core/knowledge.py: build_knowledge_package)
       |
VALIDATE PACKAGE            (core/knowledge.py: validate_knowledge_package)
       |
[BUNDLE VISUAL EVIDENCE]    (core/visual_evidence.py -- copies the selected
       |                      frames OUT of the workspace, which cleanup is
       |                      about to delete. Order is load-bearing.)
       |
EXPORT / HANDOFF            (core/knowledge.py: export_knowledge_package --
       |                      validates again, then writes atomically:
       |                      .tmp file, then os.replace() into place)
   [only on success]
       |
MARK SUCCESS -> CLEAN UP TEMP WORKSPACE
```

`validate_knowledge_package` and `export_knowledge_package` (both in
`core/knowledge.py`) are plain functions, not consumer- or process_video-
specific — any caller with a `KnowledgePackage` (from `process_video`,
`build_knowledge_package`, or reconstructed from a saved JSON file) can
call them directly. `process_video()` composes them; it doesn't own
validation/export logic exclusively.

`process_video()` (in `video_lens.py`) is the one function that runs this
whole lifecycle. `analyze_video()` still exists unchanged, for callers who
want the full evidence-level `AnalysisResult` and want to manage their own
storage/caching — `process_video()` is the recommended entry point for
standalone "give it a video, get compact knowledge back" use.

## Knowledge extraction

`core/knowledge.py` builds a `KnowledgePackage` from an already-correlated
`AnalysisResult` (+ the raw `Transcript`, for text access). It is
deterministic and rule-based — **no LLM call is made or required**:

- **Key lessons**: transcript segments matched against a fixed set of
  domain-neutral cue phrases (`"warning"`, `"first, "`, `"for example"`,
  `"in conclusion"`, `"because "`, ...), each tagged with a `kind`
  (`"warning"`, `"procedure"`, `"definition"`, `"example"`, `"conclusion"`,
  `"explanation"`) and cited back to the exact transcript `Evidence` it
  came from. A plain sentence with no recognized cue phrase is not
  included — nothing is invented, and nothing is blindly copied in either.
- **Topics**: word-frequency over the transcript (stopwords excluded,
  words appearing at least twice), capped at 8 — a naive but fully
  deterministic and domain-neutral signal, not semantic topic modeling.
- **Important observations**: Step 7's own `Inference`s (already
  evidence-cited Video-Lens conclusions) reused directly, tagged
  `kind="observation"`.
- **Limitations**: built honestly from what was actually available —
  e.g. `"Vision analysis was not available for this run"` when
  `vision_enabled=False` or no API key was configured, `"No usable speech
  was available"` when the transcript status isn't `"ok"`.

This is intentionally not full semantic summarization. `build_knowledge_package`
is the one seam where a smarter reasoning backend (an LLM-based summarizer)
could be plugged in later without touching anything else — but Video-Lens
does not depend on one, and none is used here. **No `ANTHROPIC_API_KEY` is
read, required, or used anywhere in this module or in `process_video()`.**
The existing, separate `ClaudeVisionAdapter` (Step 6) remains fully
optional and untouched — see `docs/vision.md`.

## KnowledgePackage format

Canonical form is a small JSON object (`core/knowledge.to_json`); Markdown
(`core/knowledge.render_markdown`) is a **derived presentation format**,
never the source of truth.

```
KnowledgePackage
├── source              (source_type, source, duration_sec, title)
├── summary              a short, deterministically-composed sentence
├── topics                tuple of strings
├── key_lessons           tuple of KeyPoint (text, kind, nature, timestamp_sec, evidence, confidence)
├── important_observations  tuple of KeyPoint (kind="observation", nature="inferred")
├── evidence               the Evidence actually cited by the lists above (deduped)
├── limitations             tuple of honestly-stated caveats
├── processing              generated_at, video_lens_version, knowledge_schema_version,
│                            frame_count, transcript_status, stages_available, stages_unavailable
├── semantic_summary        str | None -- whole-video summary, from a synthesizer ONLY
├── claims                  tuple of Claim -- verified semantic knowledge units
├── visual_evidence         tuple of VisualEvidence -- references to bundled image files
└── synthesis               SynthesisMetadata | None -- status, provider, retained/rejected counts
```

The last four fields are Step 11's optional semantic layer (schema `1.1` —
see `docs/synthesis.md`). All default to empty/`None`, so a package built
without a synthesizer is exactly the package it always was, and a 1.0
consumer reading a 1.1 package finds every field it already knew.
`summary` stays deterministic in **both** cases — a synthesizer's summary
lands in `semantic_summary` and never overwrites it, so the two can never be
confused for one another.

`KeyPoint.nature` (`"observed"` | `"inferred"`) makes the Step 7 observed-
vs-inferred distinction explicit on every point, not just recoverable from
which list it's in: `key_lessons` are transcript quotes (`nature=
"observed"` — the point IS the cited evidence), `important_observations`
are Step 7 `Inference`s (`nature="inferred"` — a conclusion drawn FROM
evidence). Every `KeyPoint` cites at least one `Evidence` item either way
(enforced by `KeyPoint.__post_init__`) — an inference with no citation is
rejected at construction, not just discouraged by convention.

Nothing frame/video-sized is ever in this shape: no embedded images, no
embedded video, no raw per-segment transcript dump — see
`tests/test_knowledge.py`'s and `tests/test_handoff_contract.py`'s
compactness tests. On the real 270s tutorial video used for validation, the
resulting package was **5.1 KB** (see Storage below).

## Schema version

`ProcessingMetadata.knowledge_schema_version` (currently `"1.1"`,
`core.contracts.KNOWLEDGE_SCHEMA_VERSION`) versions the *shape* of
`KnowledgePackage`/`KeyPoint`/`KnowledgeSource`/`ProcessingMetadata` —
independent of `video_lens_version` (the software release, e.g. `"1.0.0"`
today). A consumer parsing a saved package gates its parsing logic on this
field, not on the software version: a patch release that touches no field
shapes never forces a consumer migration, and a genuine field-shape change
always bumps `KNOWLEDGE_SCHEMA_VERSION`. This is a single string bump, not
semantic versioning machinery — Video-Lens doesn't need more than "which
shape is this."

## Validation

`core.knowledge.validate_knowledge_package(package)` raises
`KnowledgePackageError` on the first structural or honesty violation found:
an empty/missing source reference, a non-positive duration, a missing
summary or processing metadata, a `KeyPoint` timestamped outside the source
video's own duration, or `processing.stages_unavailable` contradicting
evidence actually cited in the package (claiming, say, "no pointer
evidence" while a `KeyPoint` cites pointer evidence anyway). Most per-field
constraints (confidence bounds, non-empty evidence, valid `Evidence.kind`,
valid `KeyPoint.nature`) are already enforced by the dataclasses'
`__post_init__` at construction time and aren't re-checked here.

This is **not** a semantic-quality judgment — a package with zero key
lessons because the video was silent is valid, as long as `limitations`
honestly says so (see `tests/test_handoff_contract.py`,
`test_genuinely_empty_but_honest_package_is_valid_not_rejected`). Only
genuinely broken or self-contradictory packages are rejected.

## Evidence retention policy

| | keep | temporary (deleted on success) |
|---|---|---|
| Knowledge package (JSON) | ✓ | |
| Provenance / timestamps (inside the package) | ✓ | |
| Source reference (path/URL, inside the package) | ✓ | |
| Selected evidence frames (`<stem>_evidence/`, a handful of files) | ✓ | |
| Downloaded video (URL source, job-owned) | | ✓ |
| Extracted frames | | ✓ |
| Vision cache | | ✓ |
| The user's own local source file | **never touched, never deleted** | |

A **local** source is read, never owned — Video-Lens's temp workspace never
contains it, so cleanup can never reach it, by construction (ownership is
by directory containment: only paths under the job's own root are ever
deleted).

A **URL** source is downloaded into the job's own `download_dir` (inside
its temp workspace, via the same `download_dir=` parameter
`ingest()`/`URLIngestionAdapter` already supported before Step 9 — no new
download mechanism was added). Because it's inside the workspace, it is
deleted along with everything else on successful cleanup, and kept
alongside everything else if the job fails or `retain_temp_artifacts=True`.

## Cleanup safety gate

`core/workspace.py`'s `JobWorkspace` is the only thing that ever deletes a
job's temporary files:

- `cleanup()` **refuses** (returns `False`, deletes nothing) unless
  `mark_success()` was already called, or `force=True` is passed
  explicitly by a caller that has its own reason to know deletion is safe.
- `process_video()` only calls `mark_success()` after the knowledge package
  has been built, validated, and **atomically** written to durable storage
  (write to a `.tmp` file, then `os.replace()` into place — so a reader
  never sees a half-written file, and a crash mid-write leaves the old
  output, if any, intact).
- `cleanup()` only ever deletes paths under the workspace's own `root` —
  never a broad/recursive delete of anything else.

Consequences, all covered by `tests/test_lifecycle.py`:

- **Processing fails** (e.g. ingestion error) → workspace created, nothing
  written, exception propagates, workspace left on disk.
- **Output validation fails** (`KnowledgePackageError`) → same: nothing
  cleaned up, exception propagates.
- **Writing the durable output fails** (e.g. output directory is blocked)
  → `mark_success()` is never reached, workspace stays.
- **Everything succeeds** → workspace is deleted immediately after the
  durable file lands.
- **`retain_temp_artifacts=True`** → workspace is kept even on success, for
  debugging.
- **Re-running the same job** → the output filename is derived from the
  video's title/basename, and the write is atomic, so a second run cleanly
  overwrites the first without ever leaving a corrupt/partial file.

An orphaned workspace from a genuinely failed run is not swept
automatically — the same "no eviction, treat as disposable" policy the
frame/vision caches already document (`docs/frames.md`, `docs/vision.md`)
applies here: `%TEMP%/videolens_jobs/<job-id>/` is safe to delete manually
at any time once you've inspected (or given up on) a failed run.

## Standalone usage

`process_video()` requires nothing from any particular downstream consumer —
give it a local path or URL, get a `KnowledgePackage` back:

```python
from video_lens import process_video, PipelineConfig
from core.knowledge import render_markdown

package = process_video("lesson.mp4")
print(render_markdown(package))   # optional Markdown presentation

config = PipelineConfig(output_dir="./my_knowledge", retain_temp_artifacts=False)
package = process_video("https://example.com/video", config)
```

## Configuration

New `PipelineConfig` fields (all optional, sensible defaults, favoring low
storage):

| field | default | meaning |
|---|---|---|
| `output_dir` | `None` -> `./videolens_output` | where the durable `KnowledgePackage` JSON is written |
| `retain_temp_artifacts` | `False` | `True` skips cleanup even after success, for debugging |

Every other `PipelineConfig` field (model sizes, `vision_enabled`,
`max_frames`, etc.) behaves exactly as documented in the main README — Step
9 does not change `analyze_video()`'s behavior at all.

## Handoff / export contract

`process_video()`'s durable JSON file (or `core.knowledge.to_dict(package)`
/ `to_json(package)` directly) **is** the handoff contract — a plain dict /
JSON string, built with `dataclasses.asdict`, no external-consumer-specific
imports, no coupling to any particular consumer's filesystem or storage
layout. Video-Lens exposes structured knowledge; it has no opinion about
how a consumer stores, indexes, retrieves, embeds, or organizes it:

```
Video-Lens  ->  KnowledgePackage (JSON, schema-versioned)  ->  external consumer
```

That consumer could be a memory/knowledge system, another
application, a CLI, a script, or a human reading the JSON (or the derived
Markdown from `render_markdown`) directly — Video-Lens is the same either
way, and **imports nothing from any of them**. Concretely, a consumer:

1. Reads the JSON file `process_video()` wrote (or calls
   `build_knowledge_package`/`export_knowledge_package` itself, for a
   caller managing its own pipeline run).
2. Checks `processing.knowledge_schema_version` against the shape it knows
   how to parse.
3. Reads `source` to identify what video the knowledge came from.
4. Reads `summary`/`topics`/`key_lessons`/`important_observations` for the
   knowledge itself, using `nature` to tell an observed quote from an
   inferred conclusion, and `supporting_evidence`/`evidence` for
   provenance back to a timestamp in the source video.
5. Reads `limitations` to know what wasn't available for this run.

Nothing here requires calling back into Video-Lens, a database, or a
network service — it's one JSON file, readable by `json.load()` in any
language, not just Python.

## Storage

Measured on the real 270s, 1920x1080 tutorial video already used for Step
9's earlier visual validation (`upibaDHY36Y.mp4`, 82.79 MB):

| | size |
|---|---|
| source video | 82.79 MB |
| durable knowledge package (after cleanup) | **5.1 KB** |
| temporary workspace during processing | frames + caches, deleted on success |
| temporary workspace after successful completion | **0 bytes** (directory removed) |

`DURABLE STORAGE ≈ KNOWLEDGE PACKAGE + MINIMAL METADATA` — not
`VIDEO + FRAMES + TRANSCRIPT + CACHE`. See `scripts/lifecycle_smoke_test.py`
for the script that produced these numbers; run it again on any video to
reproduce.

## Failure behavior

See [Cleanup safety gate](#cleanup-safety-gate) above — in every failure
case, `process_video()` raises the underlying exception
(`VideoIngestionError`, `KnowledgePackageError`, or an `OSError` from a
blocked output path) and leaves the job's temporary workspace on disk.
Nothing is ever silently swallowed into an empty/fake `KnowledgePackage`.
