"""End-to-end smoke test: input video -> metadata -> frames -> transcript,
timestamps preserved throughout. Run from the repo root:

    python scripts/smoke_test.py <video_path> [out_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.frames.ffmpeg_scenedetect import SceneDetectFrameAdapter
from adapters.ingestion import ingest
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter


def main():
    video_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "smoke_test_output"

    print(f"[1/4] Ingesting: {video_path}")
    video = ingest(video_path)
    print(f"      source_type={video.source_type} duration={video.duration_sec}s "
          f"size={video.width}x{video.height} fps={video.fps} has_audio={video.has_audio} "
          f"video_codec={video.video_codec} audio_codec={video.audio_codec}")
    assert video.duration_sec > 0
    assert video.width > 0 and video.height > 0

    print("[2/4] Extracting frames at detected scene changes")
    frames = SceneDetectFrameAdapter().extract_frames(video, f"{out_dir}/frames")
    for f in frames:
        print(f"      frame @ {f.timestamp_sec:.2f}s -> {f.path}")
    assert len(frames) > 0
    assert all(Path(f.path).exists() for f in frames)

    print("[3/4] Transcribing audio (CPU-only)")
    transcript = FasterWhisperAdapter().transcribe(video)
    print(f"      status={transcript.status} language={transcript.language}")
    for seg in transcript.segments:
        print(f"      [{seg.start_sec:.2f}s - {seg.end_sec:.2f}s] {seg.text}")
    if video.has_audio:
        assert transcript.status in ("ok", "no_speech")
        assert all(s.end_sec >= s.start_sec for s in transcript.segments)

    print("[4/4] Smoke test passed.")


if __name__ == "__main__":
    main()
