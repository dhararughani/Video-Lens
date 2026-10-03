"""Step 10 filesystem worker: claim, run process_video, verify, publish status.
Real pipeline runs on a tiny generated clip (transcription stubbed, fake vision,
no network); worker-only logic uses a fake process_video. See docs/worker.md.

Run: python -m pytest tests/test_worker.py
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_lens
import video_lens_worker as worker
from core.contracts import Transcript, TranscriptSegment, VisionObservation
from core.errors import FrameExtractionError, TranscriptionError
from core.session import load_session
from core.workspace import JOBS_ROOT

REPO = str(Path(__file__).resolve().parent.parent)
# Windows refuses to move or replace a file another process holds open; POSIX
# doesn't lock. Tests touching that assert each platform's real behaviour.
WINDOWS = os.name == "nt"


@functools.lru_cache(maxsize=None)
def _clip() -> str:
    d = tempfile.mkdtemp(prefix="vl_worker_clip_")
    atexit.register(shutil.rmtree, d, True)
    path = os.path.join(d, "clip.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x120:r=10:d=3",
                    "-pix_fmt", "yuv420p", path], capture_output=True, check=True)
    return path


class _Vision:
    def __init__(self, status="ok", raises=None):
        self.status, self.raises = status, raises

    def analyze_frame(self, frame, transcript_context=None, pointer=None):
        if self.raises:
            raise self.raises
        return VisionObservation(timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status=self.status,
                                 description="a test pattern" if self.status == "ok" else "",
                                 confidence=0.8 if self.status == "ok" else 0.0, model="fake")


class _RaisingSynth:
    def synthesize(self, brief):
        raise RuntimeError("synthesizer down")


_SPEECH = Transcript(segments=[TranscriptSegment(0.0, 3.0, "this is a test pattern")], language="en")


def _job(**over):
    job = {"source": _clip(), "session": True, "config": {"vision_enabled": True, "pointer_enabled": False}}
    job.update(over)
    return job


def _submit(root, job_id, job):
    inbox = os.path.join(root, "inbox")
    os.makedirs(inbox, exist_ok=True)
    tmp = os.path.join(inbox, f"{job_id}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(job if isinstance(job, str) else json.dumps(job))
    os.replace(tmp, os.path.join(inbox, f"{job_id}.json"))


def _work(root, vision=None, **config):
    base = video_lens.PipelineConfig(vision_provider=vision or _Vision(), **config)
    with mock.patch.object(video_lens, "_try_transcribe", return_value=_SPEECH), \
            contextlib.redirect_stderr(io.StringIO()):
        return worker.run_once(root, base)


def _status(root, state, job_id):
    with open(os.path.join(root, state, job_id, "status.json"), encoding="utf-8") as f:
        return json.load(f)


def _ls(root, state):
    return sorted(os.listdir(os.path.join(root, state)))


def _drop_workspace(status):
    if status.get("workspace"):
        shutil.rmtree(status["workspace"], ignore_errors=True)


def _fake_process(source, config):
    """Stands in for process_video: writes the package the worker verifies."""
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "t.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    return SimpleNamespace(source=SimpleNamespace(title="t", source=source),
                           processing=SimpleNamespace(stages_unavailable=()))


def _hold(path, seconds=None):
    """Another process holding `path` open (antivirus, an editor, a reader):
    until _release(), or for `seconds`."""
    wait = f"time.sleep({seconds})" if seconds else "sys.stdin.read()"
    proc = subprocess.Popen([sys.executable, "-c", f"import sys, time; f = open(sys.argv[1], 'rb'); "
                             f"print('held', flush=True); {wait}", path],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def _release(proc):
    proc.stdin.close()
    proc.wait(timeout=30)
    proc.stdout.close()


def _jobs():
    return set(os.listdir(JOBS_ROOT)) if os.path.isdir(JOBS_ROOT) else set()


# ------------------------------------------------------------- happy path

def test_a_valid_job_runs_process_video_and_publishes_verified_outputs_to_done():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "job-1", _job())
        before = _jobs()
        status = _work(root)
        assert _jobs() == before, "success cleaned the pipeline workspace"
        assert status == _status(root, "done", "job-1")
        assert (_ls(root, "inbox"), _ls(root, "running"), _ls(root, "failed")) == ([], [], [])
        done = os.path.join(root, "done", "job-1")
        assert status["status"] == "succeeded" and "workspace" not in status and "error" not in status
        assert status["outputs"]["package"].startswith("output/") and status["outputs"]["session"].startswith("session/")
        assert json.load(open(os.path.join(done, status["outputs"]["package"]), encoding="utf-8"))["processing"]["knowledge_schema_version"] == "1.2"
        assert load_session(os.path.join(done, status["outputs"]["session"])).format_version == 1
        assert os.path.isfile(os.path.join(done, "job.json"))
        # explicit, recorded configuration: the job's choices and the defaults it didn't touch
        assert status["config"]["vision_enabled"] is True and status["config"]["pointer_enabled"] is False
        assert status["config"]["visual_change_enabled"] is False, "the default the job left alone"
        assert status["config"]["providers"]["vision"] == "_Vision"
        assert status["session_requested"] is True and status["source"] == _clip()
        assert not any(n.endswith(".tmp") for _, _, ns in os.walk(root) for n in ns)


def test_the_job_owns_its_settings_and_the_worker_owns_the_paths():
    seen = {}

    def spy(source, config):
        seen["state"] = (_ls(root, "inbox"), _ls(root, "running"),
                         _status(root, "running", "j")["status"])
        seen["call"] = (source, config)
        return _fake_process(source, config)

    with tempfile.TemporaryDirectory() as root:
        _submit(root, "j", _job(session=False, config={"vision_enabled": False, "visual_change_enabled": True,
                                                        "max_frames": 7}))
        with mock.patch.object(video_lens, "process_video", side_effect=spy) as pv:
            status = worker.run_once(root, video_lens.PipelineConfig(output_dir="elsewhere", retain_temp_artifacts=True))
        assert pv.call_count == 1, "the worker invokes process_video, once"
        assert seen["state"] == ([], ["j"], "running"), "claimed out of the inbox before processing"
        source, config = seen["call"]
        assert source == _clip()
        assert (config.vision_enabled, config.visual_change_enabled, config.max_frames) == (False, True, 7)
        assert config.output_dir == os.path.join(root, "running", "j", "output")
        assert config.session_dir is None and config.retain_temp_artifacts is False
        assert status["outputs"] == {"output_dir": "output", "package": "output/t.json", "session": None}
        assert status["config"]["providers"]["vision"] is None


# --------------------------------------------------------------- failures

def test_a_raising_vision_provider_fails_the_job_diagnosably_and_keeps_its_workspace():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "v", _job())
        status = _work(root, vision=_Vision(raises=RuntimeError("model crashed")))
        try:
            assert status == _status(root, "failed", "v") and _ls(root, "done") == []
            assert status["error"]["type"] == "RuntimeError" and status["error"]["message"] == "model crashed"
            assert os.path.isdir(status["workspace"]) and status["workspace"].startswith(JOBS_ROOT)
            assert os.listdir(os.path.join(status["workspace"], "frames")), "the frames are there to inspect"
            tb = open(os.path.join(root, "failed", "v", "traceback.txt"), encoding="utf-8").read()
            assert "analyze_frame" in tb and "model crashed" in tb, "points at the provider"
        finally:
            _drop_workspace(status)


def test_provider_degradation_is_success_and_is_reported():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "d", _job())
        status = _work(root, vision=_Vision(status="unavailable"), knowledge_synthesizer=_RaisingSynth())
    assert status["status"] == "succeeded"
    assert "vision" in status["stages_unavailable"]
    assert status["config"]["providers"]["knowledge_synthesizer"] == "_RaisingSynth"


def test_a_session_that_was_requested_but_not_written_is_never_success():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "s", _job())
        with mock.patch.object(video_lens, "save_session"):  # "writes" nothing
            status = _work(root)
        assert status["status"] == "failed" and status["error"]["type"] == "IncompleteOutput"
        assert _ls(root, "done") == []
        assert os.listdir(os.path.join(root, "failed", "s", "output")), "the package is kept for inspection"


def test_a_session_write_failure_fails_the_job_and_leaves_the_package_and_workspace():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "s", _job())
        with mock.patch.object(video_lens, "save_session", side_effect=OSError("disk full")):
            status = _work(root)
        try:
            assert status["status"] == "failed" and status["error"] == {
                "type": "OSError", "message": "disk full", "traceback": "traceback.txt"}
            assert os.path.isdir(status["workspace"])
            assert any(n.endswith(".json") for n in os.listdir(os.path.join(root, "failed", "s", "output")))
        finally:
            _drop_workspace(status)


_BAD_JOBS = {
    "not-json": "{not json",
    "a-list": "[1, 2]",
    "unknown-key": {**_job(), "priority": 1},
    "no-source": {"config": {"vision_enabled": False}},
    "empty-source": {"source": " ", "config": {"vision_enabled": False}},
    "no-config": {"source": "x.mp4"},
    "implicit-vision": {"source": "x.mp4", "config": {}},
    "worker-owned-path": {"source": "x.mp4", "config": {"vision_enabled": False, "output_dir": "/tmp"}},
    "provider-object": {"source": "x.mp4", "config": {"vision_enabled": False, "vision_provider": "openai"}},
    "string-number": {"source": "x.mp4", "config": {"vision_enabled": False, "max_frames": "5"}},
    "bool-number": {"source": "x.mp4", "config": {"vision_enabled": False, "max_frames": True}},
    "float-int": {"source": "x.mp4", "config": {"vision_enabled": False, "max_frames": 2.5}},
    "negative": {"source": "x.mp4", "config": {"vision_enabled": False, "tolerance_sec": -1}},
    "infinite": '{"source": "x.mp4", "config": {"vision_enabled": false, "tolerance_sec": Infinity}}',
    "session-string": {"source": "x.mp4", "session": "yes", "config": {"vision_enabled": False}},
    "future-format": {"format": 2, "source": "x.mp4", "config": {"vision_enabled": False}},
}


def test_malformed_and_invalid_jobs_fail_cleanly_without_running_the_pipeline():
    with tempfile.TemporaryDirectory() as root:
        for job_id, job in _BAD_JOBS.items():
            _submit(root, job_id, job)
        with mock.patch.object(video_lens, "process_video") as pv:
            statuses = [worker.run_once(root) for _ in _BAD_JOBS]
        assert worker.run_once(root) is None and pv.call_count == 0
        assert _ls(root, "failed") == sorted(_BAD_JOBS) and _ls(root, "done") == []
        for s in statuses:
            assert s["status"] == "failed" and s["error"]["type"] == "InvalidJob", s
            assert s["workspace"] is None, "the pipeline never started, so there is no workspace"


def test_a_job_id_that_is_not_a_safe_name_is_set_aside():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "bad name", _job())
        with mock.patch.object(video_lens, "process_video") as pv, contextlib.redirect_stderr(io.StringIO()):
            assert worker.run_once(root) is None
        assert pv.call_count == 0 and _ls(root, "inbox") == ["bad name.json.invalid-id"]


# ------------------------------------------------------------ idempotency

def test_a_finished_job_id_is_never_rerun_or_overwritten():
    with tempfile.TemporaryDirectory() as root:
        for state in ("done", "failed"):
            os.makedirs(os.path.join(root, state, f"{state}-job"))
            Path(root, state, f"{state}-job", "status.json").write_text("original")
            _submit(root, f"{state}-job", _job())
        with mock.patch.object(video_lens, "process_video") as pv, contextlib.redirect_stderr(io.StringIO()):
            assert worker.run_once(root) is None
        assert pv.call_count == 0
        assert _ls(root, "inbox") == ["done-job.json.duplicate", "failed-job.json.duplicate"]
        for state in ("done", "failed"):
            assert Path(root, state, f"{state}-job", "status.json").read_text() == "original"


def test_a_running_job_id_stays_queued_until_it_finishes():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "running", "r"))
        _submit(root, "r", _job())
        with mock.patch.object(video_lens, "process_video") as pv:
            assert worker.run_once(root) is None
        assert pv.call_count == 0 and _ls(root, "inbox") == ["r.json"]


def test_the_same_job_content_under_two_ids_is_two_independent_jobs():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "a", _job(session=False))
        _submit(root, "b", _job(session=False))
        with mock.patch.object(video_lens, "process_video", side_effect=_fake_process):
            first, second = worker.run_once(root), worker.run_once(root)
        assert (first["job_id"], second["job_id"]) == ("a", "b"), "lexical order"
        assert _ls(root, "done") == ["a", "b"]


def test_two_workers_racing_for_one_job_claim_it_exactly_once():
    for _ in range(25):
        with tempfile.TemporaryDirectory() as root:
            for d in worker.STATES:
                os.makedirs(os.path.join(root, d))
            _submit(root, "race", _job())
            barrier, won = threading.Barrier(4), []

            def claim():
                barrier.wait()
                if worker._claim(root, "race"):
                    won.append(1)

            threads = [threading.Thread(target=claim) for _ in range(4)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            assert len(won) == 1 and _ls(root, "inbox") == []
            assert os.listdir(os.path.join(root, "running", "race")) == ["job.json"]


def test_a_claim_blocked_by_a_held_inbox_file_leaves_the_job_queued_not_stranded():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "held", _job(session=False))
        holder = _hold(os.path.join(root, "inbox", "held.json"))
        try:
            with mock.patch.object(video_lens, "process_video", side_effect=_fake_process) as pv,                     contextlib.redirect_stderr(io.StringIO()) as err:
                first = worker.run_once(root)
        finally:
            _release(holder)
        if WINDOWS:  # WinError 32: the claim is undone and retried later
            assert first is None and pv.call_count == 0 and "cannot claim yet" in err.getvalue()
            assert _ls(root, "inbox") == ["held.json"] and _ls(root, "running") == [], "no stray claim"
            with mock.patch.object(video_lens, "process_video", side_effect=_fake_process) as pv:
                second = worker.run_once(root)
                assert worker.run_once(root) is None
            assert second["status"] == "succeeded" and pv.call_count == 1, "processed exactly once"
        else:  # POSIX: an open file doesn't block a rename, so the claim just succeeds
            assert first["status"] == "succeeded" and pv.call_count == 1
        assert (_ls(root, "inbox"), _ls(root, "running"), _ls(root, "done")) == ([], [], ["held"])


def test_a_set_aside_blocked_by_a_held_inbox_file_does_not_stop_the_worker():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "done", "dup"))
        _submit(root, "dup", _job())
        holder = _hold(os.path.join(root, "inbox", "dup.json"))
        try:
            with mock.patch.object(video_lens, "process_video") as pv,                     contextlib.redirect_stderr(io.StringIO()) as err:
                assert worker.run_once(root) is None
        finally:
            _release(holder)
        assert pv.call_count == 0
        if WINDOWS:
            assert _ls(root, "inbox") == ["dup.json"] and "cannot set it aside yet" in err.getvalue()
            with contextlib.redirect_stderr(io.StringIO()):
                assert worker.run_once(root) is None
        assert _ls(root, "inbox") == ["dup.json.duplicate"] and _ls(root, "running") == []


def _process_then_hold(name, seconds, holders):
    """A fake process_video that leaves `name` (relative to the job directory)
    held open by another process as it returns -- just before the worker publishes."""
    def run(source, config):
        package = _fake_process(source, config)
        holders.append(_hold(os.path.join(os.path.dirname(config.output_dir), name), seconds))
        return package
    return run


def test_publication_retries_while_a_file_is_briefly_held_then_reaches_done():
    for name in ("output/t.json", "status.json"):  # the directory rename; the final status replace
        with tempfile.TemporaryDirectory() as root:
            _submit(root, "pub", _job(session=False))
            holders = []
            try:
                with mock.patch.object(video_lens, "process_video", side_effect=_process_then_hold(name, 0.6, holders)):
                    status = worker.run_once(root)
            finally:
                [_release(h) for h in holders]
            assert status["status"] == "succeeded", name
            assert (_ls(root, "running"), _ls(root, "done")) == ([], ["pub"]), name
            assert _status(root, "done", "pub") == status, name


def test_publication_held_past_the_retry_window_raises_and_never_reaches_done():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "pub", _job(session=False))
        holders = []
        try:
            with mock.patch.object(video_lens, "process_video", side_effect=_process_then_hold("output/t.json", None, holders)):
                started = time.monotonic()
                if WINDOWS:
                    try:
                        worker.run_once(root)
                        raise AssertionError("publication succeeded while the output was held")
                    except PermissionError:
                        waited = time.monotonic() - started
                    assert 2.0 <= waited < 15, f"bounded retry, waited {waited:.2f}s"
                    assert (_ls(root, "running"), _ls(root, "done")) == (["pub"], [])
                    assert _status(root, "running", "pub")["status"] == "succeeded", "kept for manual recovery"
                else:  # POSIX: an open file doesn't block a directory rename
                    assert worker.run_once(root)["status"] == "succeeded"
                    assert (_ls(root, "running"), _ls(root, "done")) == ([], ["pub"])
        finally:
            [_release(h) for h in holders]


def test_a_permission_error_that_is_not_a_file_in_use_is_never_retried():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "pub", _job(session=False))
        calls = []

        def denied(src, dst):
            calls.append(1)
            raise PermissionError(13, "Permission denied")  # no `winerror`: not a held file

        with mock.patch.object(video_lens, "process_video", side_effect=_fake_process),                 mock.patch.object(worker.os, "rename", side_effect=denied):
            try:
                worker.run_once(root)
                raise AssertionError("swallowed")
            except PermissionError:
                pass
        assert len(calls) == 1 and _ls(root, "running") == ["pub"] and _ls(root, "done") == []


# ------------------------------------------------------- interrupts, crashes

def test_ctrl_c_is_recorded_then_re_raised_with_the_workspace_named():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "k", _job())
        with mock.patch.object(video_lens, "_try_select_frames", side_effect=KeyboardInterrupt):
            try:
                _work(root)
                raise AssertionError("KeyboardInterrupt was swallowed")
            except KeyboardInterrupt:
                pass
        status = _status(root, "failed", "k")
        try:
            assert status["error"]["type"] == "KeyboardInterrupt" and os.path.isdir(status["workspace"])
        finally:
            _drop_workspace(status)


def test_a_failed_status_write_never_publishes_and_keeps_the_previous_status():
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "w", _job(session=False))
        real, calls = json.dump, []

        def dump(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")  # the final status write
            return real(*a, **k)

        with mock.patch.object(video_lens, "process_video", side_effect=_fake_process), \
                mock.patch.object(worker.json, "dump", side_effect=dump):
            try:
                worker.run_once(root)
                raise AssertionError("a bookkeeping failure must not be swallowed")
            except OSError:
                pass
        assert _ls(root, "done") == [] and _ls(root, "failed") == [] and _ls(root, "running") == ["w"]
        assert _status(root, "running", "w")["status"] == "running", "the last good status survives"


_CRASH = textwrap.dedent("""
    import os, sys
    from unittest import mock
    sys.path.insert(0, {repo!r})
    import video_lens, video_lens_worker as worker
    from core.contracts import Transcript, TranscriptSegment
    point = sys.argv[2]
    die = lambda *a, **k: os._exit(9)
    real_save, real_mark = video_lens.save_session, video_lens.JobWorkspace.mark_success
    def save_then_die(*a, **k):
        real_save(*a, **k); os._exit(9)
    real_replace = os.replace
    def die_on_claim_move(src, dst):
        if dst.endswith("job.json"):
            os._exit(9)
        return real_replace(src, dst)
    patches = {{
        "mid claim": mock.patch.object(worker.os, "replace", die_on_claim_move),
        "during processing": mock.patch.object(video_lens, "_try_select_frames", die),
        "after package write": mock.patch.object(video_lens, "save_session", die),
        "after session write": mock.patch.object(video_lens, "save_session", save_then_die),
        "before success status": mock.patch.object(worker, "_finish", die),
        "after success status": mock.patch.object(worker.os, "rename", die),
    }}
    speech = Transcript(segments=[TranscriptSegment(0.0, 3.0, "test")], language="en")
    with patches[point], mock.patch.object(video_lens, "_try_transcribe", return_value=speech):
        worker.run_once(sys.argv[1], video_lens.PipelineConfig(vision_enabled=False))
""")


def test_an_unexpected_termination_leaves_the_job_in_running_never_done():
    script = _CRASH.format(repo=REPO)
    with tempfile.TemporaryDirectory() as root:  # killed between mkdir and the move: documented manual recovery
        _submit(root, "c", _job())
        r = subprocess.run([sys.executable, "-c", script, root, "mid claim"], capture_output=True, text=True)
        assert r.returncode == 9, r.stderr[-2000:]
        assert _ls(root, "inbox") == ["c.json"] and os.listdir(os.path.join(root, "running", "c")) == []
    for point in ("during processing", "after package write", "after session write",
                  "before success status", "after success status"):
        with tempfile.TemporaryDirectory() as root:
            _submit(root, "c", _job(config={"vision_enabled": False, "pointer_enabled": False}))
            before = _jobs()
            r = subprocess.run([sys.executable, "-c", script, root, point], capture_output=True, text=True)
            assert r.returncode == 9, (point, r.stderr[-2000:])
            for leaked in _jobs() - before:  # the dead job's workspace (test hygiene)
                shutil.rmtree(os.path.join(JOBS_ROOT, leaked), ignore_errors=True)
            assert (_ls(root, "done"), _ls(root, "failed"), _ls(root, "running")) == ([], [], ["c"]), point
            status = _status(root, "running", "c")["status"]
            assert status == ("succeeded" if point == "after success status" else "running"), point
            out = os.path.join(root, "running", "c", "output")
            has_package = os.path.isdir(out) and any(n.endswith(".json") for n in os.listdir(out))
            assert has_package == (point != "during processing"), point


def test_a_restarted_worker_leaves_stale_running_jobs_alone_and_takes_new_ones():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "running", "stale"))
        Path(root, "running", "stale", "status.json").write_text('{"status": "running"}')
        _submit(root, "fresh", _job(session=False))
        with mock.patch.object(video_lens, "process_video", side_effect=_fake_process):
            status = worker.run_once(root)
            assert worker.run_once(root) is None
        assert status["job_id"] == "fresh" and _ls(root, "done") == ["fresh"]
        assert _ls(root, "running") == ["stale"] and os.listdir(os.path.join(root, "running", "stale")) == ["status.json"]


# ---------------------------------------------------- secrets, failure matrix

def test_no_secrets_reach_the_status_or_traceback():
    signed = "https://alice:hunter2@cdn.example.com/v.mp4?X-Amz-Signature=SECRETSIG&token=SECRETTOK"
    with tempfile.TemporaryDirectory() as root, \
            mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-SECRETKEY"}):
        _submit(root, "u", _job(source=signed))
        with mock.patch.object(video_lens, "process_video", side_effect=OSError(f"could not download {signed}")):
            status = worker.run_once(root)
        written = "".join(Path(root, "failed", "u", n).read_text(encoding="utf-8")
                          for n in ("status.json", "traceback.txt"))
    assert status["source"] == "https://cdn.example.com/v.mp4?<redacted>"
    for secret in ("SECRETSIG", "SECRETTOK", "hunter2", "SECRETKEY"):
        assert secret not in written, secret


def test_failure_injection_never_reports_success_without_its_artifacts():
    """Each stage broken in turn, real pipeline. Degradable stages degrade (the
    job succeeds and says what was unavailable); everything else fails with its
    own error type and, once the pipeline has started, its workspace."""

    class FailingWhisper:
        def __init__(self, exc):
            self.exc = exc

        def __call__(self, **kwargs):
            return self

        def transcribe(self, video):
            raise self.exc

    cases = {
        "ingestion": ({"source": os.path.join(tempfile.gettempdir(), "no-such-video.mp4")}, {}, "VideoIngestionError"),
        "transcription degrades": ({}, {"FasterWhisperAdapter": FailingWhisper(TranscriptionError("no audio"))}, None),
        "transcription bug": ({}, {"FasterWhisperAdapter": FailingWhisper(RuntimeError("bug"))}, "RuntimeError"),
        "keyframes degrade": ({}, {"select_keyframes": mock.Mock(side_effect=FrameExtractionError("ffmpeg"))}, None),
        "keyframes bug": ({}, {"select_keyframes": mock.Mock(side_effect=ValueError("bug"))}, "ValueError"),
        "package export": ({}, {"export_knowledge_package": mock.Mock(side_effect=OSError("blocked"))}, "OSError"),
        "session export": ({}, {"save_session": mock.Mock(side_effect=OSError("blocked"))}, "OSError"),
    }
    for name, (job_over, patches, error) in cases.items():
        with tempfile.TemporaryDirectory() as root:
            _submit(root, "f", _job(**job_over))
            base = video_lens.PipelineConfig(vision_provider=_Vision())
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                if "FasterWhisperAdapter" not in patches:
                    stack.enter_context(mock.patch.object(video_lens, "_try_transcribe", return_value=_SPEECH))
                for attr, value in patches.items():
                    stack.enter_context(mock.patch.object(video_lens, attr, value))
                status = worker.run_once(root, base)
            try:
                if error is None:
                    assert status["status"] == "succeeded" and status["stages_unavailable"], name
                else:
                    assert status["status"] == "failed" and status["error"]["type"] == error, (name, status)
                    assert os.path.isdir(status["workspace"]), name
                    assert _ls(root, "done") == [], name
            finally:
                _drop_workspace(status)

    # cleanup itself failing: outputs are complete, but process_video raised, so not success
    with tempfile.TemporaryDirectory() as root:
        _submit(root, "f", _job())
        with mock.patch("core.workspace.shutil.rmtree", side_effect=OSError("locked")):
            status = _work(root)
        try:
            assert status["status"] == "failed" and status["error"]["type"] == "OSError"
            assert _ls(root, "done") == []
        finally:
            _drop_workspace(status)


# ------------------------------------------------------------------- CLI

def test_the_once_cli_exits_0_when_idle_or_succeeded_and_1_when_failed():
    with tempfile.TemporaryDirectory() as root:
        assert worker.main([root, "--once"]) == 0
        _submit(root, "ok", _job(session=False))
        _submit(root, "zz-bad", "{")
        with mock.patch.object(video_lens, "process_video", side_effect=_fake_process):
            assert worker.main([root, "--once"]) == 0
            assert worker.main([root, "--once"]) == 1
        assert _ls(root, "done") == ["ok"] and _ls(root, "failed") == ["zz-bad"]


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
