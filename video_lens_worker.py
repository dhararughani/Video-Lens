"""Thin filesystem worker around `video_lens.process_video` (see docs/worker.md).

    ROOT/inbox/<job_id>.json        queued: a job file, written atomically by the submitter
    ROOT/running/<job_id>/          claimed: job.json + status.json, output/, session/ fill in
    ROOT/done/<job_id>/             succeeded: the whole job directory, renamed in one step
    ROOT/failed/<job_id>/           failed: same, plus traceback.txt; the pipeline's temp
                                    workspace is preserved and named in status.json

The worker only discovers, validates, claims, invokes `process_video`, verifies
the artifacts it asked for, and publishes a status. Every pipeline stage stays
in video_lens.py. Filesystem only; one worker per ROOT is the supported scope
(claiming is atomic, but nothing recovers a job whose worker died -- it stays
in running/, see docs/worker.md "Stale jobs").

    python video_lens_worker.py ROOT            # run until Ctrl+C
    python video_lens_worker.py ROOT --once     # process at most one job, then exit
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import replace

import video_lens
from core.knowledge import default_filename_stem
from core.session import SESSION_SUFFIX, load_session

JOB_FORMAT = 1
STATES = ("inbox", "running", "done", "failed")
_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")

# The PipelineConfig knobs a job file may set, with the JSON types each accepts.
# Paths are the worker's (output/session go in the job directory; caches in the
# pipeline's own job workspace) and provider objects can't come from JSON -- an
# operator injects those through `base_config`.
_NUM, _INT = (int, float), int
_JOB_FIELDS = {
    "whisper_model_size": str, "whisper_device": str, "whisper_compute_type": str,
    "keyframe_interval_sec": _NUM, "keyframe_diff_threshold": _NUM, "max_frames": _INT,
    "pointer_enabled": bool, "pointer_step_sec": _NUM,
    "vision_enabled": bool, "vision_model": (str, type(None)),
    "visual_change_enabled": bool, "visual_change_interval_sec": _NUM, "visual_change_max_samples": _INT,
    "cursor_intelligence_enabled": bool, "tolerance_sec": _NUM,
    "visual_evidence_enabled": bool, "max_visual_evidence": (int, type(None)),
}
# The one setting with an outside effect (the built-in vision provider calls the
# Claude API when ANTHROPIC_API_KEY is set), so a job must state it rather than
# inherit PipelineConfig's default of True.
_REQUIRED = ("vision_enabled",)


class InvalidJob(ValueError):
    """The job file is malformed or asks for something the worker can't do."""


class IncompleteOutput(RuntimeError):
    """process_video returned, but an artifact the job asked for is missing or unreadable."""


def run_once(root: str, base_config: video_lens.PipelineConfig | None = None) -> dict | None:
    """Claim and run the first queued job (lexical order of job id). Returns its
    final status, or None if nothing was claimable. A failed job is recorded and
    returned, not raised -- except Ctrl+C/SystemExit, which are recorded and
    re-raised. Errors in the worker's own bookkeeping (e.g. a status write) are
    not caught: the job then stays in running/, never reported as done."""
    for d in STATES:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for name in sorted(os.listdir(os.path.join(root, "inbox"))):
        if not name.endswith(".json") or name.startswith("."):
            continue  # e.g. a submitter's `<id>.json.tmp` still being written
        job_dir = _claim(root, name[:-len(".json")])
        if job_dir is not None:
            return _run(root, job_dir, base_config or video_lens.PipelineConfig())
    return None


def run_forever(root: str, base_config: video_lens.PipelineConfig | None = None,
                poll_sec: float = 5.0) -> None:
    while True:
        if run_once(root, base_config) is None:
            time.sleep(poll_sec)


def _claim(root: str, job_id: str) -> str | None:
    inbox = os.path.join(root, "inbox", f"{job_id}.json")
    if not _JOB_ID.fullmatch(job_id) or any(
            os.path.exists(os.path.join(root, d, job_id)) for d in ("done", "failed")):
        # A finished record is never overwritten: set the submission aside. To
        # re-run, remove (or rename) the finished directory and resubmit.
        reason = "invalid-id" if not _JOB_ID.fullmatch(job_id) else "duplicate"
        _set_aside(inbox, reason)
        return None
    job_dir = os.path.join(root, "running", job_id)
    try:
        os.mkdir(job_dir)  # the claim: exclusive on Windows and POSIX
    except FileExistsError:
        return None  # already running (or claimed by another worker): leave it queued
    try:
        os.replace(inbox, os.path.join(job_dir, "job.json"))
    except OSError as e:
        os.rmdir(job_dir)  # undo the claim: an empty running/<id> would strand the job
        if not isinstance(e, FileNotFoundError):  # FileNotFoundError: withdrawn since listing
            # e.g. Windows: another process (antivirus, the submitter) has the file open
            print(f"[video_lens_worker] {job_id}: cannot claim yet ({type(e).__name__}: "
                  f"{e.strerror or e}); left queued for the next scan", file=sys.stderr)
        return None
    return job_dir


def _set_aside(path: str, reason: str) -> None:
    try:
        os.replace(path, f"{path}.{reason}")
        print(f"[video_lens_worker] {os.path.basename(path)}: {reason}, set aside as "
              f"{os.path.basename(path)}.{reason}", file=sys.stderr)
    except FileNotFoundError:
        pass  # another worker already took it
    except OSError as e:  # e.g. held open on Windows: stays in the inbox, retried next scan
        print(f"[video_lens_worker] {os.path.basename(path)}: {reason}, but cannot set it aside yet "
              f"({type(e).__name__}: {e.strerror or e}); left for the next scan", file=sys.stderr)


def _run(root: str, job_dir: str, base: video_lens.PipelineConfig) -> dict:
    job_id = os.path.basename(job_dir)
    started = time.monotonic()
    status = {"worker_format": JOB_FORMAT, "job_id": job_id, "status": "running",
              "claimed_at": _now(), "worker": {"host": platform.node(), "pid": os.getpid()}}
    _write_json(os.path.join(job_dir, "status.json"), status)
    try:
        job = _load_job(os.path.join(job_dir, "job.json"))
        status["source"] = _redact(job["source"])
        config = replace(base, **job["config"], output_dir=os.path.join(job_dir, "output"),
                         session_dir=os.path.join(job_dir, "session") if job["session"] else None,
                         retain_temp_artifacts=False)
        status["session_requested"] = job["session"]
        status["config"] = _effective(config)
        package = video_lens.process_video(job["source"], config)
        status["outputs"] = _verify(job_dir, package, job["session"])
        status["stages_unavailable"] = list(package.processing.stages_unavailable)
        status["status"] = "succeeded"
    except BaseException as e:
        tb = _redact("".join(traceback.format_exception(e)))
        with open(os.path.join(job_dir, "traceback.txt"), "w", encoding="utf-8") as f:
            f.write(tb)
        status.update(status="failed", workspace=getattr(e, "workspace_root", None),
                      error={"type": type(e).__name__, "message": _redact(str(e))[:1000],
                             "traceback": "traceback.txt"})
        _finish(root, job_dir, status, started)
        if not isinstance(e, Exception):
            raise  # Ctrl+C / SystemExit: recorded above, never swallowed
        return status
    _finish(root, job_dir, status, started)
    return status


def _load_job(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            job = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InvalidJob(f"job file is not readable JSON: {e}") from e
    if not isinstance(job, dict):
        raise InvalidJob("job file must be a JSON object")
    unknown = set(job) - {"format", "source", "session", "config"}
    if unknown:
        raise InvalidJob(f"unknown job keys: {sorted(unknown)}")
    if job.get("format", JOB_FORMAT) != JOB_FORMAT:
        raise InvalidJob(f"unsupported job format {job['format']!r} (this worker reads {JOB_FORMAT})")
    if not isinstance(job.get("source"), str) or not job["source"].strip():
        raise InvalidJob("`source` must be a non-empty string (a local path or URL)")
    session = job.get("session", False)
    if not isinstance(session, bool):
        raise InvalidJob("`session` must be true or false")
    config = job.get("config")
    if not isinstance(config, dict):
        raise InvalidJob("`config` must be an object")
    for key in _REQUIRED:
        if key not in config:
            raise InvalidJob(f"`config.{key}` must be set explicitly (true uses the built-in Claude "
                             f"vision provider, which calls the API when ANTHROPIC_API_KEY is set)")
    for key, value in config.items():
        if key not in _JOB_FIELDS:
            raise InvalidJob(f"`config.{key}` is not a job setting (allowed: {sorted(_JOB_FIELDS)})")
        types = _JOB_FIELDS[key]
        if (isinstance(value, bool) and types is not bool) or not isinstance(value, types):
            raise InvalidJob(f"`config.{key}` has the wrong type: {value!r}")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and (
                not math.isfinite(value) or value < 0):
            raise InvalidJob(f"`config.{key}` must be a finite, non-negative number: {value!r}")
    return {"source": job["source"], "session": session, "config": config}


def _verify(job_dir: str, package, session_requested: bool) -> dict:
    """The worker's success gate: every artifact the job asked for exists and
    reads back. Paths are relative to the job directory, which moves on publish."""
    package_path = os.path.join("output", default_filename_stem(package) + ".json")
    try:
        with open(os.path.join(job_dir, package_path), encoding="utf-8") as f:
            json.load(f)
    except (OSError, ValueError) as e:
        raise IncompleteOutput(f"package {package_path} missing or unreadable: {e}") from e
    outputs = {"output_dir": "output", "package": package_path.replace("\\", "/"), "session": None}
    if session_requested:
        session_dir = os.path.join(job_dir, "session")
        found = sorted(n for n in os.listdir(session_dir) if n.endswith(SESSION_SUFFIX)) \
            if os.path.isdir(session_dir) else []
        if len(found) != 1:
            raise IncompleteOutput(f"expected one session file, found {found}")
        load_session(os.path.join(session_dir, found[0]))  # raises SessionError if unreadable
        outputs["session"] = f"session/{found[0]}"
    return outputs


def _finish(root: str, job_dir: str, status: dict, started: float) -> None:
    """Final status first (atomically), then the whole directory renamed into
    done/ or failed/ in one step -- so those directories only ever hold
    finished jobs, and done/ only verified ones."""
    status["finished_at"] = _now()
    status["elapsed_sec"] = round(time.monotonic() - started, 3)
    _write_json(os.path.join(job_dir, "status.json"), status)
    target = os.path.join(root, "done" if status["status"] == "succeeded" else "failed",
                          os.path.basename(job_dir))
    if os.path.exists(target):
        raise FileExistsError(f"refusing to replace finished job {target}")
    _retry_in_use(os.rename, job_dir, target)


# Windows refuses to rename or replace while another process holds a file open
# (antivirus scanning fresh output, someone reading status.json) -- usually for
# a moment. Retry those error codes only (access denied, sharing and lock
# violations), for ~2s in total, then raise as before. POSIX never sets
# `winerror`, so nothing is retried there.
_IN_USE_WINERRORS = (5, 32, 33)
_IN_USE_RETRY_SEC = (0.05, 0.1, 0.2, 0.4, 0.5, 0.75)


def _retry_in_use(op, *args):
    for delay in (*_IN_USE_RETRY_SEC, None):
        try:
            return op(*args)
        except PermissionError as e:
            if delay is None or getattr(e, "winerror", None) not in _IN_USE_WINERRORS:
                raise
            time.sleep(delay)


def _effective(config: video_lens.PipelineConfig) -> dict:
    """What actually ran: every job-settable value after defaults, plus which
    provider objects were injected (class name only -- never their state)."""
    out = {k: getattr(config, k) for k in _JOB_FIELDS}
    out["providers"] = {
        "vision": (type(config.vision_provider).__name__ if config.vision_provider is not None
                   else "built-in Claude" if config.vision_enabled else None),
        "video_understanding": type(config.video_understanding_provider).__name__
        if config.video_understanding_provider is not None else None,
        "knowledge_synthesizer": type(config.knowledge_synthesizer).__name__
        if config.knowledge_synthesizer is not None else None,
    }
    return out


def _redact(text: str) -> str:
    """Drop URL credentials and query strings (signed URLs carry tokens there)."""
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1", text)
    return re.sub(r"(https?://[^\s?#'\"]+)[?#][^\s'\"]*", r"\1?<redacted>", text)


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
    _retry_in_use(os.replace, tmp, path)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="worker directory (inbox/, running/, done/, failed/ are created)")
    ap.add_argument("--once", action="store_true", help="process at most one job, then exit")
    ap.add_argument("--poll-sec", type=float, default=5.0, help="idle wait between inbox scans")
    args = ap.parse_args(argv)
    try:
        if args.once:
            status = run_once(args.root)
            return 1 if status and status["status"] == "failed" else 0
        run_forever(args.root, poll_sec=args.poll_sec)
    except KeyboardInterrupt:
        print("[video_lens_worker] stopped", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
