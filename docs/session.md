# Evidence sessions

A **session** is a durable record of the *observations* one analysis produced, so
a downstream application can look at them again without re-running the
pipeline or keeping the video. It is an evidence layer, **not memory**:
Video-Lens does not decide what matters, remember users, or hold conversations.

## Opt-in

Off by default. A default run creates no session, no extra file, no extra
extraction and no extra provider call; cleanup is unchanged.

```python
config = PipelineConfig(session_dir="sessions")      # the only switch
package = process_video("talk.mp4", config)          # also writes sessions/<name>-<hash>.session.json
```

Outside `process_video` use `build_session(result, transcript)` +
`save_session(...)` (`core/session.py`). The file name is derived from the
source, so re-analysing the same video replaces its session (atomically) while a
different video never collides; `save_session` itself refuses to overwrite
unless `overwrite=True`.

## What it stores

| field | content |
|---|---|
| `format_version` | session format, currently `1` — **independent of** `KNOWLEDGE_SCHEMA_VERSION`; any other value is rejected |
| `created_at`, `video_lens_version` | when / by what |
| `source`, `video` | source type, source (URL credentials, fragment and all query parameters except `v`/`list`/`id` removed), duration, title; width/height/fps/has_audio. No temp path |
| `analysis` | plain booleans/numbers saying how the run was configured (tolerance, which streams ran). Never a model name, credential or path |
| `transcript` | the whole transcript, incl. word timings when present |
| `evidence` | every `Evidence` (all six kinds, `timestamp_end_sec`, `source`), canonically ordered, exact duplicates removed. Transcript evidence is the transcript's own segments as spans; `visual_change` evidence is the whole measurement stream (`AnalysisResult.visual_changes`), not only what fell inside a window |
| `windows` | each correlation window and which streams were `unavailable` in it |

## What it does not store

The video; extracted frames (a `frame` evidence keeps only a file-name marker —
frames are not retained); Video-Lens's own **inferences**, disagreements, the
summary and claims (conclusions, not observations); per-stream raw objects
beyond their evidence (e.g. a vision observation's element regions, undetected
pointer events); anything about a user, preferences, conversation or intent.

## Loading

`load_session(path)` / `parse_session(text)` validate strictly and raise
`SessionError` — never a partial result — for malformed JSON, missing, unknown or
mistyped fields, an invalid evidence kind or value, an oversized file, or a
format version other than the one this code reads (a newer one says so).
Optional fields take their defaults.

## Retrieval

`core/retrieval.py` — deterministic, no model, no ranking:

```python
evidence_near(session, 42.0, 1.0, kinds="cursor_track")        # around a moment
evidence_between(session, 17.0, 18.0, sources="video_understanding")
```

- Boundaries are **inclusive**. Point evidence: `|ts − t| ≤ tolerance`, literally
  `core.evidence._within_tolerance`, the correlation layer's own rule.
- **Span** evidence (`cursor_track`, transcript) matches when its interval
  **overlaps** the window; touching counts. It is never reduced to its start.
- `kinds`/`sources` filter by exact membership (`None` = no filter; source `""`
  is per-frame/unlabelled vision, distinct from `"video_understanding"`).
- Results are a tuple ordered by (start, end, kind, source, ref, confidence);
  exact duplicates appear once; no match is `()`. Bad arguments raise `ValueError`.

## Lifecycle

Persisting adds one small JSON file (no media). Temp artifacts are cleaned up
exactly as before: after success, and **kept for inspection on failure** — a
failed job writes no session. A session that cannot be written fails the job
and leaves the workspace in place, like any other failed handoff.
