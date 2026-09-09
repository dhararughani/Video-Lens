"""Speech contract + transcription tests.

Contract tests are pure and fast. Transcription tests generate synthetic
media with ffmpeg (no committed binaries, no network beyond the one-time
Whisper model download) and are skipped if faster-whisper isn't installed.

Run: python tests/test_speech.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.ingestion import ingest
from core.contracts import Transcript, TranscriptSegment, Word
from core.errors import TranscriptionError


def _t(*pairs, **kw):
    segs = [TranscriptSegment(start_sec=a, end_sec=b, text=txt) for a, b, txt in pairs]
    return Transcript(segments=segs, **kw)


# --------------------------- contract tests ---------------------------

def test_valid_segment_and_words():
    w = (Word(text="hello", start_sec=1.0, end_sec=1.4),
         Word(text="there", start_sec=1.4, end_sec=1.9))
    s = TranscriptSegment(start_sec=1.0, end_sec=2.0, text="hello there", words=w)
    assert s.words[0].text == "hello"
    assert s.words[-1].end_sec <= s.end_sec
    # words stay optional -- engines that don't provide them leave None
    assert TranscriptSegment(start_sec=0.0, end_sec=1.0, text="x").words is None


def test_segment_end_before_start_rejected():
    try:
        _t((5.0, 2.0, "backwards"))
        assert False, "expected ValueError for end < start"
    except ValueError:
        pass


def test_out_of_order_segments_rejected():
    try:
        _t((10.0, 12.0, "second"), (1.0, 2.0, "first"))
        assert False, "expected ValueError for unordered segments"
    except ValueError:
        pass


def test_overlapping_segments_allowed():
    # Whisper genuinely emits slightly overlapping segments; the contract
    # must tolerate that rather than reject real engine output.
    t = _t((8.8, 11.9, "one"), (11.7, 12.8, "two"))
    assert len(t.segments) == 2


def test_ok_status_with_no_segments_rejected():
    try:
        Transcript(segments=[], status="ok")
        assert False, "expected ValueError -- empty 'ok' transcript is ambiguous"
    except ValueError:
        pass
    # the explicit statuses are fine with no segments
    assert Transcript(segments=[], status="no_audio").status == "no_audio"
    assert Transcript(segments=[], status="no_speech").status == "no_speech"


# --------------------------- time access ---------------------------

def _sample():
    return _t((0.0, 2.0, "Look at this chart."),
              (10.0, 13.0, "Now I am clicking the button."),
              (20.0, 23.0, "Notice how the price moves down."))


def test_segment_at():
    t = _sample()
    assert t.segment_at(11.5).text == "Now I am clicking the button."
    assert t.segment_at(10.0).text == "Now I am clicking the button."  # inclusive start
    assert t.segment_at(13.0).text == "Now I am clicking the button."  # inclusive end
    assert t.segment_at(5.0) is None  # silence between segments
    assert t.segment_at(999.0) is None


def test_segments_between():
    t = _sample()
    assert len(t.segments_between(9.0, 24.0)) == 2
    assert len(t.segments_between(0.0, 100.0)) == 3
    assert t.segments_between(3.0, 9.0) == []
    # partial overlap counts -- a segment straddling the window edge is included
    assert len(t.segments_between(12.0, 15.0)) == 1


def test_text_around():
    t = _sample()
    assert "clicking" in t.text_around(11.0, window_sec=2.0)
    assert t.text_around(6.0, window_sec=0.5) == ""
    # a wide window joins neighbouring speech in time order
    assert t.text_around(11.5, window_sec=10.0).startswith("Look at this chart.")


# --------------------------- transcription tests ---------------------------

def _make_video(path: str, audio: str | None, duration: float = 4.0):
    """audio: None (no stream) | 'silence' | path to a wav."""
    cmd = ["ffmpeg", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={duration}:size=128x96:rate=5"]
    if audio == "silence":
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={duration}"]
    elif audio:
        cmd += ["-i", audio]
    cmd += ["-shortest", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, capture_output=True, check=True)


def _speech_wav(path: str, text: str) -> bool:
    """Windows SAPI TTS. Returns False if unavailable (test then skips)."""
    ps = (f"Add-Type -AssemblyName System.Speech; "
          f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{text}'); $s.Dispose()")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True)
    return r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0


def test_no_audio_reports_status_not_empty_success():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "silent.mp4")
        _make_video(p, audio=None)
        video = ingest(p)
        assert video.has_audio is False
        t = _adapter().transcribe(video)
        assert t.status == "no_audio"
        assert t.segments == []
        assert t.duration_sec is not None


def test_silent_audio_reports_no_speech():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "quiet.mp4")
        _make_video(p, audio="silence", duration=6.0)
        t = _adapter().transcribe(ingest(p))
        # VAD must suppress Whisper's tendency to invent text over silence
        assert t.status == "no_speech", f"expected no_speech, got {t.status}: {t.segments}"
        assert t.segments == []


def test_real_speech_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "s.wav")
        if not _speech_wav(wav, "Look at this chart on the left side of the screen."):
            print("  (skipped test_real_speech_timestamps -- no TTS available)")
            return
        p = os.path.join(tmp, "speech.mp4")
        _make_video(p, audio=wav, duration=6.0)
        t = _adapter().transcribe(ingest(p))
        assert t.status == "ok", f"expected ok, got {t.status}"
        assert t.language is not None
        assert "chart" in " ".join(s.text for s in t.segments).lower()
        # timestamps must be real and within the media, not placeholders
        for s in t.segments:
            assert 0.0 <= s.start_sec <= s.end_sec <= t.duration_sec + 1.0
        assert t.segments[0].start_sec < 3.0
        # and must be queryable by time
        assert t.segment_at(t.segments[0].start_sec + 0.1) is not None


def test_timestamps_never_exceed_media_duration():
    # Regression: Whisper pads short audio to its 30s window and reported a
    # 7.0s end on a 3.76s video, which would break frame lookup.
    from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "s.wav")
        if not _speech_wav(wav, "Click the button in the top left corner."):
            print("  (skipped test_timestamps_never_exceed_media_duration -- no TTS)")
            return
        p = os.path.join(tmp, "short.mp4")
        _make_video(p, audio=wav, duration=3.0)
        video = ingest(p)
        t = FasterWhisperAdapter(model_size="tiny.en").transcribe(video)
        for s in t.segments:
            assert s.end_sec <= video.duration_sec, (
                f"segment ends at {s.end_sec}s past media end {video.duration_sec}s")
            assert s.start_sec >= 0.0
            for w in (s.words or ()):
                assert 0.0 <= w.start_sec <= video.duration_sec


def test_clamping_drops_segments_outside_media():
    from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
    clamp = FasterWhisperAdapter._clamp
    inside = TranscriptSegment(start_sec=1.0, end_sec=2.0, text="in")
    assert clamp(inside, 10.0) is inside  # untouched when already valid
    over = TranscriptSegment(start_sec=1.0, end_sec=7.0, text="over")
    assert clamp(over, 3.76).end_sec == 3.76
    outside = TranscriptSegment(start_sec=9.0, end_sec=12.0, text="past the end")
    assert clamp(outside, 3.76) is None


def test_missing_model_raises_transcription_error():
    from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p, audio="silence")
        bad = FasterWhisperAdapter(model_size="definitely-not-a-real-model-xyz")
        try:
            bad.transcribe(ingest(p))
            assert False, "expected TranscriptionError"
        except TranscriptionError:
            pass


def _adapter():
    from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter
    return FasterWhisperAdapter(model_size="tiny.en")


CONTRACT_TESTS = [test_valid_segment_and_words, test_segment_end_before_start_rejected,
                  test_out_of_order_segments_rejected, test_overlapping_segments_allowed,
                  test_ok_status_with_no_segments_rejected, test_segment_at,
                  test_segments_between, test_text_around]
CONTRACT_TESTS.append(test_clamping_drops_segments_outside_media)
ENGINE_TESTS = [test_no_audio_reports_status_not_empty_success,
                test_silent_audio_reports_no_speech, test_real_speech_timestamps,
                test_timestamps_never_exceed_media_duration,
                test_missing_model_raises_transcription_error]

if __name__ == "__main__":
    for t in CONTRACT_TESTS:
        t()
        print(f"  ok: {t.__name__}")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("All speech CONTRACT tests passed (engine tests skipped: no faster-whisper).")
        sys.exit(0)
    for t in ENGINE_TESTS:
        t()
        print(f"  ok: {t.__name__}")
    print("All speech tests passed.")
