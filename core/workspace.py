"""Job-scoped temporary workspace + cleanup safety gate for the Step 9
processing lifecycle (see docs/lifecycle.md):

    PROCESSING -> OUTPUT CREATED -> OUTPUT VALIDATED -> HANDOFF SUCCESS
                                                              |
                                                       mark_success()
                                                              |
                                                          cleanup()

A JobWorkspace only ever deletes files under its own `root` -- it never
touches a caller-supplied local source file, which always lives outside the
workspace. `cleanup()` is a no-op unless `mark_success()` was already called
(or `force=True` is passed by a caller that has its own reason to know
deletion is safe), so a crash, a failed validation, or a failed handoff
simply leaves the temp tree in place for inspection -- never silently
cleaned up. Pure stdlib, no external dependency, same "no eviction, treat as
disposable" philosophy as the existing frame/vision caches (docs/frames.md,
docs/vision.md) -- an orphaned job directory from an interrupted run needs
the same manual/periodic cleanup those already document.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid

JOBS_ROOT = os.path.join(tempfile.gettempdir(), "videolens_jobs")


class JobWorkspace:
    """One job's owned temp directory tree: frames/, vision/, downloads/."""

    def __init__(self, root: str | None = None):
        self.root = root or os.path.join(JOBS_ROOT, uuid.uuid4().hex)
        self.frame_cache_dir = os.path.join(self.root, "frames")
        self.vision_cache_dir = os.path.join(self.root, "vision")
        self.download_dir = os.path.join(self.root, "downloads")
        for d in (self.frame_cache_dir, self.vision_cache_dir, self.download_dir):
            os.makedirs(d, exist_ok=True)
        self._succeeded = False

    def mark_success(self) -> None:
        """Arm cleanup(). Call only after the durable output has been
        created, validated, and written -- never before."""
        self._succeeded = True

    def cleanup(self, force: bool = False) -> bool:
        """Delete this job's entire temp tree and everything under it
        (downloaded video, extracted frames, vision cache). Returns False
        and deletes nothing unless mark_success() already ran or
        force=True is passed explicitly. Never touches any path outside
        self.root."""
        if not (self._succeeded or force):
            return False
        shutil.rmtree(self.root, ignore_errors=True)
        return True

    def size_bytes(self) -> int:
        """Current on-disk size of this job's workspace -- for storage
        measurement/reporting (see docs/lifecycle.md), not used internally."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in filenames:
                fp = os.path.join(dirpath, name)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
        return total


def _demo():
    ws = JobWorkspace()
    assert os.path.isdir(ws.frame_cache_dir)
    with open(os.path.join(ws.frame_cache_dir, "f.jpg"), "wb") as f:
        f.write(b"x" * 100)
    assert ws.size_bytes() == 100
    assert ws.cleanup() is False and os.path.isdir(ws.root)  # not marked -> refuses
    ws.mark_success()
    assert ws.cleanup() is True and not os.path.exists(ws.root)  # marked -> deletes
    print("core/workspace.py self-check passed.")


if __name__ == "__main__":
    _demo()
