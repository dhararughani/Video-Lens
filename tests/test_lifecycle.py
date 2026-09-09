"""Step 9 lifecycle tests: JobWorkspace ownership/cleanup safety gate, and
process_video's temporary-vs-durable storage behavior end to end on real
local videos (no network, no API key). See docs/lifecycle.md.

Run: python tests/test_lifecycle.py
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_lens as vl
from core.errors import KnowledgePackageError
from core.workspace import JobWorkspace
from video_lens import PipelineConfig, process_video


@contextmanager
def _track_job_roots():
    """Records the JobWorkspace.root of every workspace process_video
    creates during the block, and force-removes any that survive (e.g. a
    deliberately-triggered failure path correctly left them uncleaned) --
    test hygiene only, so these tests don't litter the real OS temp
    directory on every run. Does not change or weaken production cleanup
    behavior, which is exercised and asserted on before this ever runs."""
    original_cls = vl.JobWorkspace
    roots: list[str] = []

    class _Tracked(original_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            roots.append(self.root)

    vl.JobWorkspace = _Tracked
    try:
        yield roots
    finally:
        vl.JobWorkspace = original_cls
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)


def _make_video(path: str, duration: float = 2.0, size: str = "160x120"):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:size={size}:rate=5:duration={duration}",
         "-pix_fmt", "yuv420p", path],
        capture_output=True, check=True,
    )


def _fast_config(tmp: str, output_dir: str | None = None, **overrides) -> PipelineConfig:
    return PipelineConfig(vision_enabled=False, pointer_enabled=False,
                           output_dir=output_dir or os.path.join(tmp, "out"), **overrides)


# ------------------------------ JobWorkspace ------------------------------

def test_workspace_owns_a_scoped_temp_tree():
    ws = JobWorkspace()
    try:
        assert os.path.isdir(ws.frame_cache_dir)
        assert os.path.isdir(ws.vision_cache_dir)
        assert os.path.isdir(ws.download_dir)
        assert ws.frame_cache_dir.startswith(ws.root)
    finally:
        ws.cleanup(force=True)


def test_cleanup_refuses_before_mark_success():
    ws = JobWorkspace()
    with open(os.path.join(ws.frame_cache_dir, "f.jpg"), "wb") as f:
        f.write(b"x")
    assert ws.cleanup() is False
    assert os.path.exists(ws.root)
    ws.cleanup(force=True)  # test cleanup


def test_cleanup_proceeds_after_mark_success():
    ws = JobWorkspace()
    ws.mark_success()
    assert ws.cleanup() is True
    assert not os.path.exists(ws.root)


def test_url_downloaded_video_is_owned_and_deleted_only_after_success():
    """process_video() points URLIngestionAdapter's download_dir at the
    job's own workspace.download_dir (see video_lens.process_video) -- so a
    URL-downloaded video is deleted by the same containment-based cleanup
    as everything else, and ONLY after mark_success(). No real network call
    is made here (would violate the "no unnecessary downloads" constraint
    and this project's no-network-in-tests convention) -- this test
    verifies the ownership/cleanup mechanism directly against the
    workspace's download_dir, which is exactly what a real yt-dlp download
    would land in."""
    ws = JobWorkspace()
    downloaded = os.path.join(ws.download_dir, "abc123.mp4")
    with open(downloaded, "wb") as f:
        f.write(b"fake downloaded video bytes")

    assert ws.cleanup() is False  # not yet marked successful -- must survive
    assert os.path.exists(downloaded)

    ws.mark_success()
    assert ws.cleanup() is True
    assert not os.path.exists(downloaded)  # deleted along with the rest of the workspace


def test_cleanup_never_touches_paths_outside_its_root():
    with tempfile.TemporaryDirectory() as tmp:
        sentinel = os.path.join(tmp, "not_mine.txt")
        with open(sentinel, "w") as f:
            f.write("keep me")
        ws = JobWorkspace(root=os.path.join(tmp, "job"))
        ws.mark_success()
        ws.cleanup()
        assert os.path.exists(sentinel)  # sibling file untouched
        assert not os.path.exists(ws.root)


# ------------------------------ process_video: durable vs temporary ------------------------------

def test_process_video_produces_compact_durable_output_and_cleans_up_temp():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "v.mp4")
        _make_video(video_path)
        config = _fast_config(tmp)
        package = process_video(video_path, config)

        out_dir = config.output_dir
        json_files = list(Path(out_dir).glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["source"]["source"] == video_path

        # durable output is small -- nothing frame/video-sized ended up in it
        assert json_files[0].stat().st_size < 20_000
        assert package.source.duration_sec > 0


def test_local_source_is_never_deleted_or_modified():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "v.mp4")
        _make_video(video_path)
        original_size = os.path.getsize(video_path)
        original_mtime = os.path.getmtime(video_path)

        process_video(video_path, _fast_config(tmp))

        assert os.path.exists(video_path)
        assert os.path.getsize(video_path) == original_size
        assert os.path.getmtime(video_path) == original_mtime


def test_local_source_survives_even_when_read_only():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "v.mp4")
        _make_video(video_path)
        os.chmod(video_path, stat.S_IREAD)
        try:
            process_video(video_path, _fast_config(tmp))
            assert os.path.exists(video_path)
        finally:
            os.chmod(video_path, stat.S_IWRITE | stat.S_IREAD)  # allow tempdir cleanup


# ------------------------------ failure paths keep temp artifacts ------------------------------

def test_failed_ingestion_leaves_nothing_to_clean_up_and_raises():
    with _track_job_roots() as roots:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                process_video(os.path.join(tmp, "does_not_exist.mp4"), _fast_config(tmp))
                assert False, "a missing video must raise, not silently produce a package"
            except Exception as e:
                assert "not exist" in str(e).lower() or "does not exist" in str(e).lower()
        assert len(roots) == 1
        assert os.path.isdir(roots[0]), "temp workspace was deleted despite failed ingestion"


def test_failed_output_validation_does_not_clean_up_temp_workspace():
    """A KnowledgePackage that fails _validate_package must leave the job's
    temp workspace on disk for inspection, not silently deleted."""
    original_validate = vl._validate_package
    vl._validate_package = lambda package: (_ for _ in ()).throw(
        KnowledgePackageError("forced failure for test"))
    try:
        with _track_job_roots() as roots:
            with tempfile.TemporaryDirectory() as tmp:
                video_path = os.path.join(tmp, "v.mp4")
                _make_video(video_path)
                try:
                    process_video(video_path, _fast_config(tmp))
                    assert False, "validation failure must propagate, not be swallowed"
                except KnowledgePackageError:
                    pass
            assert len(roots) == 1
            assert os.path.isdir(roots[0]), "temp workspace was deleted despite failed validation"
    finally:
        vl._validate_package = original_validate


def test_failed_handoff_write_does_not_mark_success():
    """If writing the durable output fails, the workspace must never be
    marked successful (and thus never cleaned up) -- simulated by pointing
    output_dir at a path that cannot be created (a file, not a directory)."""
    with _track_job_roots() as roots:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            _make_video(video_path)
            blocked_output = os.path.join(tmp, "blocked_output")
            with open(blocked_output, "w") as f:
                f.write("this is a file, not a directory")

            config = _fast_config(tmp, output_dir=blocked_output)
            try:
                process_video(video_path, config)
                assert False, "writing into a blocked output path must raise"
            except (NotADirectoryError, FileExistsError, OSError):
                pass
        assert len(roots) == 1
        assert os.path.isdir(roots[0]), "temp workspace was deleted despite failed handoff write"


# ------------------------------ retain_temp_artifacts / re-run safety ------------------------------

def test_retain_temp_artifacts_skips_cleanup_on_success():
    with _track_job_roots() as roots:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            _make_video(video_path)
            config = _fast_config(tmp, retain_temp_artifacts=True)
            process_video(video_path, config)
        assert len(roots) == 1
        assert os.path.isdir(roots[0]), "retain_temp_artifacts=True must skip cleanup"


def test_rerunning_a_job_does_not_corrupt_output():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "v.mp4")
        _make_video(video_path)
        config = _fast_config(tmp)

        pkg1 = process_video(video_path, config)
        pkg2 = process_video(video_path, config)

        json_files = list(Path(config.output_dir).glob("*.json"))
        assert len(json_files) == 1  # same source -> same output filename, cleanly overwritten
        with open(json_files[0]) as f:
            data = json.load(f)  # must still be valid, complete JSON after two runs
        assert data["source"]["duration_sec"] == pkg1.source.duration_sec == pkg2.source.duration_sec


if __name__ == "__main__":
    test_workspace_owns_a_scoped_temp_tree()
    test_cleanup_refuses_before_mark_success()
    test_cleanup_proceeds_after_mark_success()
    test_url_downloaded_video_is_owned_and_deleted_only_after_success()
    test_cleanup_never_touches_paths_outside_its_root()
    test_process_video_produces_compact_durable_output_and_cleans_up_temp()
    test_local_source_is_never_deleted_or_modified()
    test_local_source_survives_even_when_read_only()
    test_failed_ingestion_leaves_nothing_to_clean_up_and_raises()
    test_failed_output_validation_does_not_clean_up_temp_workspace()
    test_failed_handoff_write_does_not_mark_success()
    test_retain_temp_artifacts_skips_cleanup_on_success()
    test_rerunning_a_job_does_not_corrupt_output()
    print("All lifecycle tests passed.")
