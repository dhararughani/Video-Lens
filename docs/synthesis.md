# Semantic knowledge synthesis

Step 11 adds the one thing deterministic extraction could never do: an actual
understanding of what a video is *about*, written down as evidence-backed
claims. It adds it without adding a model.

Video-Lens ships **no synthesizer** and requires none. Semantic synthesis is a
capability you supply; when you don't, the pipeline produces exactly the
deterministic package it always did and says so plainly.

## The shape of it

```
AnalysisResult (Step 7 correlated evidence)
      |
build_synthesis_brief()        compact, id'd evidence -- not a transcript dump
      |
[ your KnowledgeSynthesizer ]  <-- the ONLY model-shaped step in Video-Lens
      |
raw provider dict
      |
parse_and_verify()             cross-check against the real evidence
      |
Claims + summary  ->  KnowledgePackage  ->  visual evidence bundle
```

## The seam

`core.interfaces.KnowledgeSynthesizer` is one method:

```python
class KnowledgeSynthesizer(Protocol):
    def synthesize(self, brief: SynthesisBrief) -> dict: ...
```

It returns the provider's **raw** output, not finished `Claim` objects. That
is deliberate. Video-Lens does not trust this dict, and a provider that could
hand back well-typed `Claim`s would be handing back conclusions that skipped
verification. Returning raw JSON also keeps an implementation trivially thin:

```python
class MyLocalModel:
    def synthesize(self, brief):
        reply = my_model.chat(system=brief.system_prompt, user=brief.user_prompt)
        return json.loads(reply)          # that's the whole adapter

package = process_video("lesson.mp4", PipelineConfig(knowledge_synthesizer=MyLocalModel()))
```

The brief carries a rendered `system_prompt`/`user_prompt` pair for the common
case of a text LLM, **and** the structured `items` those prompts were rendered
from, so a provider that isn't a text LLM can work from the structure and
ignore the prose.

A provider that raises is recorded as `synthesis.status == "failed"` and never
fails the job — semantic synthesis is an enhancement, and its absence must not
destroy knowledge that is already valid.

### Ollama and other model runtimes

Video-Lens contains no Ollama integration and must not grow one. If your
system already runs Ollama, that lives on *your* side of the seam — a ~5
line adapter posting to `localhost:11434` and returning the parsed JSON. The
same is true of a hosted API, a local VLM, or anything else. Nothing in
`core/` knows which you chose, and no test needs one.

## What the synthesizer is given

Not the raw transcript. The brief is built from already-correlated evidence,
with each item assigned a short stable id (`e0`, `e7`, ...) that the provider
cites:

| stream | what goes in the brief |
|---|---|
| speech | grouped into ~20s chunks across the **whole** video, so "what is this about" is answerable |
| frames | one item per selected keyframe (timestamp + path) |
| vision | descriptions and located elements, when a vision provider ran |
| pointer | **only** where Step 7 actually drew an inference from it |
| conflicts | Step 7's recorded disagreements, so a claim in that window can say so |

Pointer evidence is filtered on purpose. Pointer detection is functional but
limited (`docs/pointer.md`); dumping every raw reading in would bury real
evidence in noise and invite a synthesizer to over-read a weak signal. Speech
is chunked rather than passed per-segment for the same reason, and the whole
brief is capped (`_MAX_BRIEF_CHARS`) — when a very long video exceeds it, the
chunks are condensed and the brief *says* they were condensed.

The ids are the mechanism that makes fabricated provenance detectable: a claim
citing an id that was never issued cannot be resolved to real evidence.

## Verification — the important half

Anyone can send evidence to a model and keep what comes back.
`core/synthesis.py` keeps only what it can still tie to evidence Video-Lens
itself produced. Every claim is checked before it is retained:

| check | failure |
|---|---|
| every cited id was actually issued | **rejected** — fabricated provenance |
| a claim cites at least one item (unless `unavailable`) | **rejected** — unsupported |
| `timestamp_sec` falls inside the video | **rejected** |
| `confidence` is a number in [0, 1] | **rejected** |
| no embedded media / base64 in the text | **rejected** |
| `confidence` is capped by the cited evidence's own confidence | silently bounded |
| a bad `timestamp_end_sec` | dropped; the claim stands |

Rejected claims are **dropped, never retained with a lowered score**, and the
count plus reasons land in `synthesis.rejection_reasons` — so a consumer can
see that verification ran and did something, rather than trusting that it did.

### The four states

`Claim.status` is deliberately separate from `confidence`. Collapsing them
would destroy the distinction that makes a claim auditable.

- **observed** — the cited evidence directly states or shows it
- **inferred** — a conclusion drawn from cited evidence
- **conflicting** — evidence exists, but streams materially disagree in that window
- **unavailable** — the system could not establish it; always `confidence == 0.0`

`conflicting` is decided **by Video-Lens, not by the provider**. Video-Lens
already knows where its own evidence streams disagreed (Step 7), and applies
that independently of what the model claimed.

### The visual cross-check

A `frame` and a `vision` observation are **not** interchangeable evidence:

- **`vision`** is interpreted content — something actually looked at the image
  and said what was in it. It can corroborate a claim.
- **`frame`** is an uninterpreted picture. When vision hasn't run, a
  synthesizer picking a frame id is guessing from a timestamp, having never
  seen it.

So a claim citing speech + a bare frame is labelled
`speech_evidence_with_uninterpreted_frame`, and carries the limitation *"the
referenced frame was never visually analyzed, so it illustrates this claim but
does not corroborate it."* Only speech + `vision` earns
`speech_corroborated_by_visual_evidence`.

This distinction was found by real-world testing, not designed in advance: a
synthesizer given only timestamps confidently attached a frame to a claim, and
the frame turned out to show the presenter's book covers. Video-Lens had
called that "corroborated". It no longer does.

### What verification cannot do

Verification is provenance-and-structure checking, not fact-checking.
`core/synthesis.py` confirms a cited `vision` item genuinely exists, genuinely
has that timestamp, and genuinely came from an image analyzed for this run —
it has no way to confirm that a vision provider's *description itself* is an
accurate reading of the frame. That would need a second model to grade the
first, which Video-Lens does not do and does not need to: quality of
interpretation is the supplied provider's responsibility, and its absence is
disclosed, not hidden.

This was confirmed with a real, independent local vision model (Ollama +
`moondream`, a 1B-parameter model — validation-only, see
`scripts/ollama_vision_test_adapter.py`, never a Video-Lens dependency) run
against real extracted frames from a real video. Given the frame's own pixels
with no transcript context, it produced descriptions that were sometimes
accurate (correctly noticing on-screen book-cover text and a presenter's
face) and sometimes wrong (describing a screenshot of two Git GUI clients,
GitHub Desktop and SourceTree, as "a webpage... about... HTML"). Verification
correctly accepted a claim citing that wrong description — the citation was
real, the timestamp was real, the image existed — because verification checks
*that the model looked*, not *that it looked correctly*.

Two things were fixed as a direct result of this test, both real defects, not
hypothetical ones:

1. **Passing transcript context to a weak model caused it to echo the speech
   back verbatim instead of describing the image at all**, which would have
   been silently accepted as "vision interpretation" despite containing zero
   independent visual information. The test adapter now withholds context
   from a provider not proven able to keep "reference" and "content"
   separate — a caller-side prompting choice, not a Video-Lens change, since
   `ClaudeVisionAdapter` (a stronger model) does not exhibit this.
2. **A claim citing `vision` evidence had no way to recover the frame that
   vision interpreted**, because `Evidence(kind="vision").ref` holds the
   description text, not a file path — so a genuinely `speech_corroborated_by_
   visual_evidence` claim shipped with zero bundled images. Fixed in
   `core/visual_evidence.py`: a vision citation is now resolved back to its
   originating frame by timestamp (exact, since a `VisionObservation` and the
   `Frame` it analyzed always share one), so a corroborated claim ships with
   the actual picture it was corroborated against.

## Visual evidence selection

`core/visual_evidence.py` decides which of hundreds of candidate frames are
worth keeping. Selection is driven by **demand, not a quota**:

1. A frame is a candidate only because some retained claim cites it — or,
   with no synthesizer, because it's the nearest frame to a retained key point.
2. Candidates are ranked by how many knowledge units want them, then by
   confidence, then by time (fully deterministic — the same video always
   produces the same bundle).
3. Near-duplicates are collapsed: a 64-bit dHash **plus mean brightness**.
   dHash alone encodes gradient structure, so every featureless frame hashes
   identically and a blank white slide would collapse into a blank black one;
   brightness separates them.
4. Survivors are copied into `<stem>_evidence/` beside the package JSON.

There is no "keep N frames" constant. A video whose knowledge rests on three
distinct visuals keeps three; a screencast that never changes keeps one.
`max_visual_evidence` exists only as a caller-supplied safety ceiling.

When a cited frame is dropped as a duplicate, the claim is linked to the frame
that **superseded** it rather than losing its illustration — the same content
is in the bundle under another id.

The anchor window for key points is derived per-run from the median gap
between frames, because key points sit on the transcript's time grid while
frames sit on the keyframe grid; a fixed tolerance leaves the deterministic
path with no visual evidence whenever keyframes are sampled further apart.

## Storage

The durable output is a knowledge artifact plus a small bundle of files:

```
videolens_output/
├── lesson.json              the KnowledgePackage
└── lesson_evidence/
    ├── e13.jpg
    └── e17.jpg
```

Images are **files referenced by relative path**, never base64 inside the
JSON, and `validate_knowledge_package` rejects both embedded media and
absolute paths (an absolute path would leak the producing machine's layout and
break the moment the bundle is moved). Evidence excerpts are capped at 400
characters each so a package can never become a back-door transcript dump.

Measured on a real 258s, 8.81 MB tutorial with 16 verified claims:

| | size |
|---|---|
| source video | 8.81 MB |
| knowledge package JSON | 33 KB |
| evidence bundle (3 frames) | 161 KB |
| **total durable output** | **197 KB — 2.2% of the source** |

Evidence is copied out of the job workspace **before** cleanup runs. The
frames live in the workspace and cleanup deletes that whole tree, so ordering
here is load-bearing: bundle, then export, then `mark_success()`, then clean.
If export fails after bundling, the bundle is left behind with the rest of the
failed run's artifacts — which is what you'd want when diagnosing it.

## Fallback behavior

With no synthesizer supplied:

- the pipeline runs exactly as before,
- `semantic_summary` is `None` and `claims` is empty — never backfilled from
  the deterministic sentence,
- `summary` remains the deterministic one (it always does, even when synthesis
  succeeds, so the two can never be confused),
- `synthesis.status` is `"unavailable"`, and `limitations` says in plain
  language that no semantic synthesis was performed.

Honesty over completeness: deterministic keyword extraction is never presented
as semantic understanding.

## Package additions (schema 1.1)

```
KnowledgePackage
├── ... every 1.0 field, unchanged ...
├── semantic_summary   str | None      the whole-video summary, from a synthesizer only
├── claims             tuple[Claim]    verified knowledge units
├── visual_evidence    tuple[VisualEvidence]   references to bundled images
└── synthesis          SynthesisMetadata | None
```

All four default to empty/`None`, so a 1.0 consumer reading a 1.1 package
finds every field it already knew, unchanged.

```
Claim
├── text, kind                     free text; no domain vocabulary, ever
├── status                         observed | inferred | unavailable | conflicting
├── timestamp_sec, timestamp_end_sec
├── supporting_evidence            real Evidence, resolved from cited ids
├── confidence                     bounded by the evidence's own confidence
├── verification                   how it was cross-checked
├── visual_evidence                ids of images actually in the bundle
└── limitations                    what this claim's evidence does NOT establish
```

`Claim.visual_evidence` is populated **only after** the bundle is written, so
it can never name an image the package doesn't ship — an invariant
`validate_knowledge_package` enforces.

## Trying it

`scripts/synthesis_smoke_test.py` runs the whole thing on a real video and
prints what was retained, what was rejected, and what the bundle cost. It
works in two phases so it needs no API key, no account, and no paid service:

```bash
python scripts/synthesis_smoke_test.py video.mp4 --brief-out brief.json
# produce response.json from brief.json with any model, then:
python scripts/synthesis_smoke_test.py video.mp4 --brief-out brief.json --response response.json
```
