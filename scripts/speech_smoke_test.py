"""Speech smoke test on a real video: ingest -> transcribe -> timestamped
segments -> query by time. Also prints a performance baseline.

    python scripts/speech_smoke_test.py <video_path_or_url> [model_size]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.ingestion import ingest
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter


def main():
    source = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else None  # None => adapter default

    print(f"[1/4] Ingesting: {source}")
    video = ingest(source)
    print(f"      {video.duration_sec:.1f}s  {video.width}x{video.height}  "
          f"has_audio={video.has_audio}  codec={video.audio_codec}")

    adapter = FasterWhisperAdapter(model_size=model_size) if model_size else FasterWhisperAdapter()
    print(f"[2/4] Transcribing with faster-whisper '{adapter.model_size}' (CPU)")
    t0 = time.time()
    transcript = adapter.transcribe(video)
    elapsed = time.time() - t0

    print(f"      status={transcript.status} language={transcript.language} "
          f"segments={len(transcript.segments)}")
    print(f"      {elapsed:.1f}s wall for {video.duration_sec:.0f}s of video "
          f"({video.duration_sec / elapsed:.1f}x realtime, model load included)")
    assert transcript.status in ("ok", "no_speech", "no_audio")

    print("[3/4] Timestamped segments")
    for s in transcript.segments[:8]:
        print(f"      [{s.start_sec:7.2f} - {s.end_sec:7.2f}]  {s.text[:66]}")
    if len(transcript.segments) > 8:
        print(f"      ... {len(transcript.segments) - 8} more")

    print("[4/4] Querying speech by time")
    if transcript.segments:
        mid = transcript.segments[len(transcript.segments) // 2]
        probe = (mid.start_sec + mid.end_sec) / 2
        hit = transcript.segment_at(probe)
        print(f"      segment_at({probe:.2f}s) -> {hit.text[:56] if hit else None}")
        window = transcript.segments_between(probe - 30, probe + 30)
        print(f"      segments_between({probe-30:.0f}s, {probe+30:.0f}s) -> {len(window)} segments")
        print(f"      text_around({probe:.2f}s, 5s) -> {transcript.text_around(probe, 5.0)[:66]}")
        assert hit is not None, "segment_at failed to find a segment at its own midpoint"

        # timestamps must be ordered and inside the media
        assert all(s.start_sec <= s.end_sec for s in transcript.segments)
        assert transcript.segments[-1].end_sec <= video.duration_sec + 2.0
        starts = [s.start_sec for s in transcript.segments]
        assert starts == sorted(starts), "segments must be ordered by start time"

    print("Speech smoke test passed.")


if __name__ == "__main__":
    main()
