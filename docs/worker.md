# Filesystem worker

`video_lens_worker.py` is a thin adapter between job files on disk and
`video_lens.process_video()`. It discovers, validates, claims, invokes,
verifies and publishes a status, and nothing else. Every pipeline stage stays
in `video_lens.py`. It needs no database, queue, broker, server or extra
dependency. It is pure stdlib.

## Layout and lifecycle

```
ROOT/
  inbox/<job_id>.json      queued     -- you put it here
        │  claim: mkdir running/<job_id> (exclusive), then move the file in
        ▼
  running/<job_id>/        running    -- job.json, status.json, output/, session/
        │  process_video(...) -> verify artifacts -> write final status.json
        ▼  then rename the whole directory, in one step
  done/<job_id>/           succeeded  -- job.json, status.json, output/, session/
  failed/<job_id>/         failed     -- the same, plus traceback.txt
```

The directory a job is in is its state. `done/` only ever receives a job
whose requested artifacts were verified, and it arrives complete because one
`rename` publishes it. The worker creates all four directories.

## Submitting a job

Write `inbox/<job_id>.json` **atomically**: write `<job_id>.json.tmp`, then
rename it. The worker ignores names that don't end in `.json` or that start
with `.`.

`job_id` is the file name: `[A-Za-z0-9][A-Za-z0-9_-]*`, up to 100 characters.
Jobs run in lexical order of id, so prefix ids with a timestamp if you want
first-in, first-out.

```json
{
  "format": 1,
  "source": "C:/videos/demo.mp4",
  "session": true,
  "config": {
    "vision_enabled": false,
    "visual_change_enabled": true,
    "pointer_enabled": true,
    "whisper_model_size": "base",
    "max_frames": 20
  }
}
```

The fields:

- **`format`**: optional, and must be `1` if present.
- **`source`**: a local path or a URL, exactly as `process_video` takes it.
- **`session`**: optional, `false` by default. When true, the job also saves an
  evidence session (see `docs/session.md`), and the job only succeeds if that
  session file exists and loads.
- **`config`**: plain values, each mapping 1:1 to a `PipelineConfig` field:
  - `whisper_model_size`, `whisper_device`, `whisper_compute_type`
  - `keyframe_interval_sec`, `keyframe_diff_threshold`, `max_frames`
  - `pointer_enabled`, `pointer_step_sec`
  - `vision_enabled`, `vision_model`
  - `visual_change_enabled`, `visual_change_interval_sec`, `visual_change_max_samples`
  - `cursor_intelligence_enabled`, `tolerance_sec`
  - `visual_evidence_enabled`, `max_visual_evidence`

  Unknown keys, wrong types, and negative or non-finite numbers are rejected.
  Anything not set takes the `PipelineConfig` default, and the status file
  records the full effective configuration.
- **`config.vision_enabled` is required.** It is the one setting with an effect
  outside your machine: `true` with no injected provider uses the built-in
  Claude provider, which calls the API when `ANTHROPIC_API_KEY` is set.

Some things a job cannot set:

- **Paths.** Output goes in the job's own directory (`output/`, `session/`),
  never the current directory. Caches and downloads live in the pipeline's own
  temporary workspace.
- **Provider objects.** An operator injects these in code through
  `base_config`:

  ```python
  run_once(root, PipelineConfig(vision_provider=MyVLM()))
  ```

  The job's plain values are applied on top of `base_config`.
- **`retain_temp_artifacts`.** It is forced off, so a successful job always
  cleans up.

## Running

```
python video_lens_worker.py ROOT --once      # at most one job, then exit (1 if it failed)
python video_lens_worker.py ROOT             # until Ctrl+C; --poll-sec N when idle (default 5)
```

From Python, `run_once(root, base_config=None)` returns the job's final status
dict, or `None` if nothing was claimable. The worker uses no shell and runs on
Windows and Linux. The two filesystems differ in ways that matter here; see
[Windows and POSIX](#windows-and-posix).

## Status (`status.json`)

A succeeded job:

```json
{
  "worker_format": 1, "job_id": "demo", "status": "succeeded",
  "claimed_at": "2026-10-03T10:00:00+00:00", "finished_at": "2026-10-03T10:03:12+00:00",
  "elapsed_sec": 192.4, "worker": {"host": "box", "pid": 4242},
  "source": "C:/videos/demo.mp4", "session_requested": true,
  "config": {"vision_enabled": false, "visual_change_enabled": true, "...": "...",
             "providers": {"vision": null, "video_understanding": null, "knowledge_synthesizer": null}},
  "outputs": {"output_dir": "output", "package": "output/demo.json",
              "session": "session/demo-1a2b3c4d.session.json"},
  "stages_unavailable": ["vision"]
}
```

A failed job:

```json
{
  "worker_format": 1, "job_id": "demo", "status": "failed", "...": "...",
  "workspace": "C:/Users/me/AppData/Local/Temp/videolens_jobs/5f0c...",
  "error": {"type": "RuntimeError", "message": "model crashed", "traceback": "traceback.txt"}
}
```

How the status fields are written:

- **`outputs` paths** are relative to the job directory, which moves on
  publish.
- **`stages_unavailable`** is the structured form of the pipeline's stderr
  warnings: the job succeeded, but these stages degraded.
- **`workspace`** is the pipeline's preserved temporary tree (frames,
  downloaded video), taken from the exception's `workspace_root`. It is
  `null` when the pipeline never started (an invalid job), and absent on
  success, because a successful job's workspace is deleted.
- **Secrets.** No API key or environment value is ever written. URL
  credentials and query strings, where signed URLs carry their tokens, are
  redacted from `source`, the error message and `traceback.txt`. The message
  is capped at 1,000 characters. `job.json` is your own submission and is
  kept as written.
- **Redaction covers the status and traceback only.** For a URL source, the
  pipeline stores the URL **verbatim** as the package's `source.source` and in
  the session file. This is provenance, and plain `process_video()` calls do
  the same. A signed URL's token therefore sits in
  `done/<id>/output/*.json`, the session file and `job.json`. Treat signed URLs
  as sensitive: prefer short-lived ones, and protect the worker root
  accordingly.

## Failures

What causes a failed status, versus a succeeded one:

- **A degraded stage still succeeds.** The pipeline already degrades
  transcription, keyframe and vision *unavailability*, native video, synthesis
  and visual change. Those jobs succeed, with `stages_unavailable` saying what
  was missing.
- **An exception from `process_video` fails the job, never silently.** This
  covers ingestion errors, a supplied `VisionAdapter` that raises, a
  bug-shaped exception from any stage, and package or session write
  failures. The error type, message, traceback and preserved workspace are
  recorded. The worker then continues with the next job.
- **An invalid job file fails** with `InvalidJob`, before the pipeline starts.
- **A missing artifact fails** with `IncompleteOutput`: `process_video`
  returned, but the package or the requested session is missing or
  unreadable. In particular, a package without its requested session is
  never success.
- **Ctrl+C or `SystemExit`** is recorded as failed (`KeyboardInterrupt`, with
  the workspace) and then re-raised, so the worker stops.
- **A failure in the worker's own bookkeeping** (e.g. writing `status.json`)
  is *not* caught. The job stays in `running/` and is never reported as done.

To inspect a failure, open `failed/<id>/status.json`, `traceback.txt`, any
partial `output/`, and the `workspace` directory. Delete the workspace when
you're finished with it; nothing sweeps it automatically (see
`docs/lifecycle.md`).

## Duplicates, retries and restarts

- **A finished id is never re-run or overwritten.** If `done/<id>` or
  `failed/<id>` already exists, a new `inbox/<id>.json` is renamed to
  `<id>.json.duplicate`. To retry, delete or rename the finished directory and
  resubmit, for example by moving `failed/<id>/job.json` back to
  `inbox/<id>.json`.
- **An id that is still running** stays queued. It is not claimed. Once the
  first run finishes, it becomes a duplicate.
- **An invalid id** is set aside as `<id>.json.invalid-id`.
- **The same content under two ids** is two independent jobs, with separate
  outputs.
- **A restarted worker** takes jobs from `inbox/` and never touches
  `running/`.

## Crashes and stale jobs

The state after the process is killed (power loss, `kill -9`, a crash) at
each point:

| Killed... | Left on disk |
|---|---|
| before claiming | the job is still in `inbox/` and runs next time |
| mid-claim (between creating `running/<id>/` and moving the job file in) | an **empty** `running/<id>/`, with the job still in `inbox/`. While that directory exists, the job counts as running and is not claimed |
| after claiming / during processing | `running/<id>/` with status `running`; the pipeline workspace in `%TEMP%/videolens_jobs/` |
| after the package write, or after the session write | the same, plus a complete package (and session) in `running/<id>/output/` |
| after the final status write, before the rename | `running/<id>/` whose `status.json` says `succeeded` |

**Stale-job recovery is deferred.** The worker cannot tell a dead worker's
job from a live one. The status records `worker.host` and `worker.pid`, but
checking whether a process is alive is platform-specific, and on Windows,
`os.kill` *terminates* the process. To recover a stale job by hand once no
worker is running:

- **Empty `running/<id>/` with `inbox/<id>.json` present (mid-claim):** delete
  the empty directory. The next scan claims the job.
- **Status `running`:** move `running/<id>/job.json` to `inbox/<id>.json`, then
  delete `running/<id>/`.
- **Status `succeeded`:** move `running/<id>/` to `done/<id>/`.

Automating this needs a heartbeat or lease: a timestamp the worker refreshes,
plus a rule for when a lease expires. That belongs in a later step.

## Concurrency scope

The supported setup is **one worker per ROOT, on a local disk.**

The claim itself is atomic. `os.mkdir` is exclusive on Windows and POSIX, so
two workers racing for the same job claim it exactly once (this is tested).
Even so, several workers on one root is not a supported setup:

- **Stale recovery assumes one worker:** with several, nothing can tell a dead
  worker's job from a live one.
- **Network filesystems are not covered:** the atomic guarantees of `mkdir`
  and `rename` on SMB or NFS are outside this design.

## Windows and POSIX

Atomic claiming (`os.mkdir`) and atomic publishing (`os.replace`, a single
directory `rename`) work on both. The differences:

- **Held files (Windows).** Windows refuses to move, replace or rename while
  another process has a file open: antivirus scanning fresh output, a
  submitter that hasn't closed the job file, a script reading `status.json`.
  POSIX has no such lock. The worker handles each case:
  - **Claim or set-aside of a held inbox file:** the claim is undone, the job
    stays in `inbox/` with a stderr note, and the next scan tries again. The
    job is not failed and not lost.
  - **Publish rename or status write:** retried on the "in use" errors (Windows
    error 5, 32 or 33) for about 2 seconds in total, then raised as before.
    The job stays in `running/` (recovery above) and never reaches `done/`.
    Other errors are never retried.
- **Job ids.** Windows filenames are case-insensitive, so `Job` and `job` are
  one id there: the second is set aside as a duplicate. On POSIX they are two
  jobs. `NUL` is a Windows device name. A `NUL.json` job is set aside as
  `.duplicate` on Windows, because Windows reports `done/NUL` as existing,
  while on POSIX it runs.
- **Encoding.** A job file must be UTF-8 without a byte-order mark. A BOM
  (PowerShell 5's `Set-Content -Encoding UTF8` writes one) is rejected as
  `InvalidJob`, with the reason in the status.
- **Termination.** A forced kill (`kill -9`, `taskkill /F`, Task Manager) or a
  POSIX `SIGTERM` ends the process without the worker recording anything,
  because it installs no signal handlers. That is the crash table above. Only
  Ctrl+C is recorded, as a failed job.
- **Durability.** Nothing is `fsync`ed. A *process* crash never publishes a
  partial job, but a power loss or OS crash can leave recently written files
  empty or missing.
- **Temporary storage.** A failed job's preserved `workspace` lives in the
  system temp directory, which a reboot or a temp cleaner (e.g. systemd's
  tmpfiles, Windows Storage Sense) may delete. Copy anything you need from it
  promptly.
