"""Unit tests for ingestion -- local files only, no network. Generates a
tiny synthetic video on the fly (ffmpeg lavfi testsrc) so no binary media is
committed to the repo. Run: python tests/test_ingestion.py

Live URL ingestion is exercised separately by scripts/url_smoke_test.py,
which needs network access and is not part of this suite.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.ingestion import ingest, is_url
from adapters.ingestion.local import LocalIngestionAdapter
from core.errors import VideoIngestionError


def _make_tiny_video(path: str, with_audio: bool = False):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, capture_output=True, check=True)


def test_valid_local_video():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "tiny.mp4")
        _make_tiny_video(video_path, with_audio=True)

        video = ingest(video_path)
        assert video.source_type == "local"
        assert video.source == os.path.abspath(video_path)
        assert video.path == os.path.abspath(video_path)
        assert video.width == 64 and video.height == 64
        assert video.duration_sec > 0
        assert video.has_audio is True
        assert video.video_codec is not None


def test_valid_local_video_no_audio():
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "tiny.mp4")
        _make_tiny_video(video_path, with_audio=False)

        video = LocalIngestionAdapter().resolve(video_path)
        assert video.has_audio is False
        assert video.audio_codec is None


def test_missing_file_raises():
    try:
        ingest("A:/definitely/does/not/exist.mp4")
        assert False, "expected VideoIngestionError"
    except VideoIngestionError:
        pass


def test_invalid_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = os.path.join(tmp, "not_a_video.mp4")
        with open(bad_path, "w") as f:
            f.write("this is not a video file")
        try:
            ingest(bad_path)
            assert False, "expected VideoIngestionError"
        except VideoIngestionError:
            pass


def test_empty_source_raises():
    try:
        ingest("")
        assert False, "expected VideoIngestionError"
    except VideoIngestionError:
        pass


def test_empty_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        empty_path = os.path.join(tmp, "empty.mp4")
        open(empty_path, "w").close()
        try:
            ingest(empty_path)
            assert False, "expected VideoIngestionError"
        except VideoIngestionError:
            pass


def test_is_url_dispatch():
    assert is_url("https://example.com/video") is True
    assert is_url("http://example.com/video") is True
    assert is_url("A:/videos/clip.mp4") is False
    assert is_url("clip.mp4") is False
    assert is_url("/home/user/clip.mp4") is False


if __name__ == "__main__":
    test_valid_local_video()
    test_valid_local_video_no_audio()
    test_missing_file_raises()
    test_invalid_file_raises()
    test_empty_source_raises()
    test_empty_file_raises()
    test_is_url_dispatch()
    print("All ingestion unit tests passed.")
