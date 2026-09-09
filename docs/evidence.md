# Evidence Correlation and Structured Understanding

Turns the four raw, timestamped evidence streams Video-Lens already
produces -- transcript (`docs/speech.md`), frames (`docs/frames.md`),
pointer events (`docs/pointer.md`), vision observations (`docs/vision.md`)
-- into a structured, auditable answer to "what happened at this point in
the video," without ever destroying or hiding the original evidence.

Most callers should use `video_lens.analyze_video()` (repository root)
rather than calling this module directly -- it runs the whole pipeline
(ingestion through this correlation step) and hands back the
`AnalysisResult` this document describes. Read on if you're calling
`core/evidence.py` directly (e.g. correlating evidence you gathered some
other way) or want to understand what `analyze_video()` does internally.

```
RAW EVIDENCE -> CORRELATED EVIDENCE -> STRUCTURED OBSERVATION -> INTERPRETATION
(Steps 3-6)      (core/evidence.py)     (core/evidence.py)        (a future,
                                                                    higher-level
                                                                    consumer --
                                                                    not built here)
```

```python
from core.evidence import build_structured_observation

so = build_structured_observation(
    132.4,
    transcript=transcript, frames=frames,
    pointer_events=pointer_events, vision_observations=vision_observations,
    tolerance_sec=1.0,
)
for e in so.observed:      # evidence-backed facts, never fabricated
    print(e.kind, e.ref)
for inf in so.inferences:  # Video-Lens's OWN conclusions, always cited
    print(inf.basis, inf.confidence, inf.text)
print(so.disagreements)    # conflicts/gaps, recorded rather than resolved
print(so.unavailable)      # evidence streams with nothing in this window
```

## Why no new subsystem

Step 7 adds exactly one new module (`core/evidence.py`) and three small,
immutable dataclasses (`Inference`, `StructuredObservation`, plus a
`confidence` field added to the existing `Evidence`). No vector database, no
RAG pipeline, no agent framework, no orchestration layer, and no LLM call is
introduced -- correlation and structuring are pure, deterministic functions
over data Steps 3-6 already produce. `AnalysisResult` (existing since Step
1) gained one new field, `structured_observations`, rather than being
replaced by a redundant parallel result type -- it already was "the correct
concept" the task asked to reuse.

## Evidence correlation

Every evidence item Video-Lens produces already carries `timestamp_sec` in
the same canonical time model (`docs/speech.md`'s TIME MODEL) -- Step 7
introduces no second time system, just two small generic primitives in
`core/evidence.py`:

- `nearest_within_tolerance(items, timestamp_sec, tolerance_sec)` -- the
  single closest item within `tolerance_sec`, or `None`. Ties resolve to the
  earlier item (deterministic, not input-order-dependent). Used for frames
  and vision observations, which are naturally sparse -- one selected frame
  or one analysis per interesting moment, so "the nearest one" is the right
  question.
- `all_within_tolerance(items, timestamp_sec, tolerance_sec)` -- every item
  within tolerance, ordered by time. Used for pointer events, since motion
  (moving vs. stationary) needs more than one reading to determine at all --
  a single nearest pointer event can't show movement (see Observation vs
  inference below).

Transcript correlation reuses `Transcript.segments_between()`
(`docs/speech.md`, unchanged since Step 3) directly -- no duplicate window
logic was written for it.

## Timestamp tolerance

**Default: `tolerance_sec=1.0`**, always an explicit, required-to-be-positive
parameter (`build_structured_observation` raises `ValueError` for
`tolerance_sec <= 0`) -- never a hidden constant. Chosen because it's the
same order of magnitude as:

- Step 4's `frames_for_segment` default sampling interval (1.0s) -- so a
  frame selected for a speech segment and a query timestamp near that
  segment's boundary correlate naturally.
- Step 5's `track_pointer` default sampling interval (0.5s) -- two pointer
  samples land inside a 1.0s window on either side of a query timestamp,
  which is exactly what the motion-vs-stationary rule needs.

A caller analyzing content with faster cuts or a different frame-selection
interval should pass a tighter `tolerance_sec`; one is not hardcoded into
the correlation primitives themselves, only used as `build_structured_observation`'s
default. Evidence outside the window is never joined, no matter how
"close" it might seem by some other measure --
`test_evidence_outside_tolerance_is_not_incorrectly_joined` verifies this
directly with evidence 95-195 seconds away from a 5.0s query.

## Observation vs. inference

This is the central discipline of this step, enforced by contract shape,
not just convention:

- **`Evidence`** (existing since Step 1, now validated: `kind` must be one
  of `"frame"|"transcript"|"pointer"|"vision"`, and a new optional
  `confidence` field) is a raw, citable fact -- a frame path, a transcript
  segment's actual text, a pointer's actual `x`/`y`, a vision observation's
  actual description or element. `StructuredObservation.observed` is a
  tuple of these and *only* these -- nothing interpretive ever gets added
  to it.
- **`Inference`** (new) is a conclusion Video-Lens drew *from* observed
  evidence. It cannot exist without citing at least one `Evidence`
  (`__post_init__` raises otherwise) and must state a `basis` -- a short,
  fixed rule label (`"pointer_in_vision_region"`,
  `"speech_pointer_stationary"`, `"speech_pointer_motion"`) naming exactly
  which deterministic rule produced it. There is no code path that can
  write inferred language into `observed`, and no code path that writes an
  `Inference` without a citation -- `tests/test_evidence.py`'s
  `test_inference_never_appears_in_observed` and
  `test_inference_requires_evidence_and_valid_confidence` check both
  directions.

Concretely:

```
OBSERVED: transcript says "this is the entry point"          (kind=transcript)
OBSERVED: pointer detector reports x=742, y=391                (kind=pointer)
OBSERVED: vision identifies a highlighted rectangular region   (kind=vision)

INFERENCE (basis=pointer_in_vision_region, confidence=0.68):
  "The pointer was detected within the region vision identified as
  '<description>'. This supports -- but does not prove -- that the speaker
  or presenter may have been referring to or interacting with that element."
```

Note the inference text itself hedges ("supports -- but does not prove")
rather than asserting intent -- this is deliberate wording baked into the
rule, not something a caller has to remember to add. Video-Lens never
produces the equivalent of "the speaker pointed at the entry" as a flat
claim.

### The three deterministic correlation rules

All three run in `core/evidence.py`, entirely rule-based -- **no LLM call is
made or needed to build a `StructuredObservation`**, which is what makes the
whole layer deterministic and testable offline (`test_deterministic_repeated_calls_produce_identical_result`
checks byte-identical output for identical input, twice, no live model
involved):

1. **`pointer_in_vision_region`** -- pointer has a real position, vision
   found a located element, and the pointer's normalized `(x, y)` falls
   inside that element's `Region`. Confidence = `pointer.confidence *
   vision.confidence` (see Confidence below). If the pointer position does
   **not** fall inside any located region, that's recorded as a
   **disagreement** instead (see Disagreement recording), never silently
   dropped or forced into a false-positive inference.
2. **`speech_pointer_stationary` / `speech_pointer_motion`** -- a
   transcript segment overlaps the window AND at least two positioned
   pointer readings exist in it (one reading cannot show motion --
   `test_single_pointer_reading_does_not_claim_motion_or_stationary` checks
   this directly). Normalized displacement between the first and last
   positioned reading is compared against a fixed threshold
   (`_MOTION_THRESHOLD = 0.03`, documented in `core/evidence.py` as a
   `ponytail:` simplification -- make configurable if a caller needs
   per-video tuning). Confidence = fraction of pointer readings in the
   window that were actually positioned (data completeness), not a made-up
   number.
3. **Vision-without-pointer-confirmation** -- vision located element(s) but
   no positioned pointer evidence exists in the window at all. This is
   *not* scored as an inference (there's no positive claim to make
   confidence about) -- it's a disagreement/incompleteness note, per rule
   15's example: "vision identifies a region but pointer evidence does not
   confirm interaction."

## Pointer + speech + vision relationships (generic example)

A software-tutorial frame, not a trading one:

```
Speech:     "click the save icon in the toolbar"      [4.80s - 5.60s]
Frame:      @5.00s
Pointer:    detected, x=812, y=44 (normalized 0.42, 0.04)   confidence=0.9
Vision:     element: button -- "save icon" region=(0.40,0.02)-(0.45,0.06)  confidence=0.85

OBSERVED: 4 items above, each with its own Evidence entry
INFERENCE (pointer_in_vision_region, confidence=0.77):
  "The pointer was detected within the region vision identified as
  'save icon' (button). This supports -- but does not prove -- that the
  speaker or presenter may have been referring to or interacting with that
  element."
```

Nothing here says "the user clicked save" -- pointer proximity is evidence
of possible reference, not of an action, per rule 6 ("never treat pointer
coordinates as proof of what the user intended").

## Provenance

Every `Evidence` item carries its own `timestamp_sec` (the *source* item's
actual timestamp, not the query timestamp it was correlated against) and
`ref` (the actual content -- a frame path, the real transcript text, a
formatted pointer position, a vision description/element). An `Inference`'s
`supporting_evidence` tuple is the literal `Evidence` objects it was built
from -- a caller can trace any inference back to the exact raw facts, not
just a textual claim about them. Nothing is summarized away.

## Confidence

Confidence at every level reflects **evidence quality/agreement**, never a
bare model score treated as ground truth (rule 12):

- An `Evidence` item's `confidence` is the *originating* source's own
  confidence when it has one (a `PointerEvent.confidence`, a
  `VisionObservation.confidence`, a `TranscriptSegment.confidence`) --
  `None` when the source kind doesn't carry one (a `Frame` has no
  confidence concept).
- `pointer_in_vision_region`'s confidence is the **product** of the two
  contributing evidence confidences -- deliberately conservative: it can
  never exceed either input, and drops fast when either signal is weak.
- `speech_pointer_*`'s confidence is the **fraction of pointer readings in
  the window that were actually positioned** -- a data-completeness
  measure, not a claim about how "sure" Video-Lens is about anything
  semantic.
- `Inference.confidence` and `VisionObservation`/`PointerEvent.confidence`
  are all validated to `[0, 1]` (`__post_init__`, consistent with every
  other confidence field in this project).

## Disagreement recording (rule 11)

When evidence streams conflict or one is silently missing where another
implies it should matter, `StructuredObservation.disagreements` records a
plain-text note -- **never** resolved by picking a side silently:

- pointer detected but outside every vision-located region.
- vision located element(s) but no pointer evidence available to confirm
  interaction.

These are deliberately *not* `Inference` objects -- there's no positive
claim being made, so there's no meaningful confidence to assign; they're
flags for a downstream consumer to weigh, not conclusions.

## Graceful degradation

Every evidence stream parameter to `build_structured_observation` is
optional and independent -- any subset may be `None`/empty/missing, and the
function never raises for a missing stream, only records it in
`unavailable`:

```
speech + frames + pointer   (vision unavailable/omitted)
speech + frames              (pointer and vision both unavailable/omitted)
frames + vision               (no transcript for this video/segment)
speech only                   (no frames requested)
nothing at all                 (every field of `unavailable` populated, `observed=()`)
```

Verified with the real vision adapter too, not just synthetic fixtures:
`scripts/evidence_smoke_test.py` run against a real video with no
`ANTHROPIC_API_KEY` configured produces `VisionObservation(status="unavailable")`
for every frame, and `build_structured_observation` correctly folds that
into `unavailable=(..., "vision")` rather than treating it as a missing
piece of information to guess at -- see Live vision validation below for
exact results.

## Live vision validation

**No `ANTHROPIC_API_KEY` was available in this development environment**
(confirmed again this step, same as Step 6: no `ant` CLI, no credential env
vars set). Per this step's explicit instruction, the user was **not** asked
for a key. No live Claude Sonnet vision call was made in this step -- real
multimodal model output quality remains unvalidated in this environment,
exactly as documented in `docs/vision.md`'s Step 6 limitations, unchanged
by this step.

What **was** verified instead, honestly:

- The entire pipeline end to end on a real video
  (`scripts/evidence_smoke_test.py`, same 1920x1080 real tutorial used in
  Steps 5-6): ingestion, transcription, frame selection, pointer detection,
  a real call into `ClaudeVisionAdapter.analyze_frame` (which honestly
  returns `status="unavailable"` here), and evidence correlation into
  `StructuredObservation`s -- all real code paths, all real data, except
  the vision model's own output.
- 20 mocked-client tests already exist in `tests/test_vision.py` (Step 6,
  unchanged) covering malformed/ambiguous model responses -- Step 7 adds no
  new mocked-vision tests of its own since `build_structured_observation`
  only ever consumes an already-validated `VisionObservation`, and Step 6's
  tests already prove that contract is honestly populated regardless of
  what the model returns.
- 29 new deterministic tests in `tests/test_evidence.py` cover every
  correlation/structuring behavior without needing a model at all.

## How a downstream consumer should use `AnalysisResult`

```python
from core.evidence import build_analysis_result

result = build_analysis_result(
    video, query_timestamps,
    transcript=transcript, frames=frames,
    pointer_events=pointer_events, vision_observations=vision_observations,
    tolerance_sec=1.0,
)

for so in result.structured_observations:
    # "what happened at this point in the video" -- the real Step 7 answer
    ...

for obs in result.observations:      # unchanged Step 1-6 shape, for existing callers
    ...
for e in result.evidence:            # every Evidence cited across all timestamps, flattened
    ...
```

`result.structured_observations` is the Step 7 answer; `observations`/
`evidence` are populated too, unchanged in shape, so nothing built against
`AnalysisResult` in Steps 1-6 breaks. A consumer that wants domain-specific
interpretation (e.g. "this is a trading setup," "this is a code refactor
being demonstrated") builds that entirely outside Video-Lens, reading
`observed`/`inferences`/`disagreements` as its evidence base -- Video-Lens
itself never performs that interpretation (rule 14, and this project's
domain-neutrality requirement, reiterated every step).

## Performance

No new extraction, caching, or network behavior. `build_structured_observation`
and `build_analysis_result` are pure in-memory functions over already-computed
evidence lists -- no frame re-extraction (Step 4's cache is untouched), no
repeat vision calls (Step 6's cache is untouched), no video loaded into RAM.
Cost scales with the number of query timestamps times the (typically small,
already-bounded) number of evidence items near each one -- linear, no new
infrastructure.

## Limitations

- Real vision model output quality remains unvalidated in this environment
  (see Live vision validation above) -- inherited from Step 6, unchanged.
- Only three correlation rules exist; more relationship types (e.g. sustained
  gaze-equivalent dwell time, multi-region comparison) are plausible future
  additions but weren't required or built here.
- `_MOTION_THRESHOLD` is a fixed constant, not adaptive to a given video's
  resolution/content style beyond the normalization it already gets from
  `PointerEvent.normalized_x/y`.
- Disagreement notes are plain text, not a structured taxonomy -- sufficient
  for this step's scope, not designed for programmatic disagreement-type
  filtering beyond substring matching.
- No mechanism resolves a disagreement automatically, by design (rule 11) --
  a downstream consumer must decide how to weigh it.
