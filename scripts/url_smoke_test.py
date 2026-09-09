"""Live URL ingestion smoke test -- needs network access, not part of the
unit test suite. Downloads a short real video through yt-dlp and normalizes
it into VideoInput, then extracts one frame to prove downstream compatibility.

Run: python scripts/url_smoke_test.py [url]
Defaults to a short public-domain/CC clip if no URL is given.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.ffmpeg_scenedetect import SceneDetectFrameAdapter
from adapters.ingestion import ingest

DEFAULT_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", ~19s, first YouTube video


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    print(f"[1/3] Resolving URL: {url}")
    video = ingest(url)
    print(f"      source_type={video.source_type} source={video.source}")
    print(f"      resolved path={video.path}")
    print(f"      duration={video.duration_sec}s size={video.width}x{video.height} "
          f"fps={video.fps} has_audio={video.has_audio} "
          f"video_codec={video.video_codec} audio_codec={video.audio_codec}")
    assert video.duration_sec > 0
    assert video.width > 0 and video.height > 0

    print("[2/3] Re-ingesting same URL (should reuse download, not re-fetch)")
    video2 = ingest(url)
    assert video2.path == video.path

    print("[3/3] Extracting one frame from downloaded video")
    frames = SceneDetectFrameAdapter().extract_frames(video, "url_smoke_test_output/frames")
    print(f"      extracted {len(frames)} frame(s), first at {frames[0].timestamp_sec:.2f}s -> {frames[0].path}")
    assert len(frames) > 0

    print("URL smoke test passed.")


if __name__ == "__main__":
    main()
