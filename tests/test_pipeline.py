"""Canonical pipeline tests: config defaults, per-stage opt-out, frame-count
ceiling, and one full local (no network, no API key) end-to-end run proving
every stage actually wires together into a coherent AnalysisResult.

Run: python tests/test_pipeline.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_lens import PipelineConfig, analyze_video


def _make_video(path: str, duration: float = 3.0, size: str = "160x120", with_audio: bool = False):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:size={size}:rate=5:duration={duration}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={duration}", "-shortest"]
    cmd += ["-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, capture_output=True, check=True)


def _speech_wav(path: str, text: str) -> bool:
    """Windows SAPI TTS -- same helper as tests/test_speech.py. Returns
    False if unavailable, so a test can skip rather than fail on a machine
    without it."""
    ps = (f"Add-Type -AssemblyName System.Speech; "
          f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{text}'); $s.Dispose()")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True)
    return r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0


# ------------------------------- config -------------------------------

def test_config_defaults_require_no_setup():
    config = PipelineConfig()
    assert config.whisper_device == "cpu"
    assert config.vision_enabled is True and config.pointer_enabled is True
    assert config.tolerance_sec > 0
    assert config.max_frames > 0


def test_config_is_mutable_per_call_not_global_state():
    a = PipelineConfig(max_frames=5)
    b = PipelineConfig(max_frames=50)
    assert a.max_frames == 5 and b.max_frames == 50  # independent instances, no shared mutable default


# ------------------------------ degradation ------------------------------

def test_vision_disabled_produces_no_vision_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p, duration=2.0)
        config = PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)
        assert all(o.vision is None for o in result.observations)
        assert all("vision" in so.unavailable for so in result.structured_observations)


def test_pointer_disabled_produces_no_pointer_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p, duration=2.0)
        config = PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)
        assert all(o.pointer is None for o in result.observations)
        assert all("pointer" in so.unavailable for so in result.structured_observations)


def test_no_audio_video_still_produces_frame_based_observations():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p, duration=2.0, with_audio=False)
        config = PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)
        assert len(result.structured_observations) >= 1
        assert all("transcript" in so.unavailable for so in result.structured_observations)
        assert all("frame" not in so.unavailable for so in result.structured_observations)


def test_pointer_stage_finds_real_motion_against_known_trajectory():
    """Regression test: an earlier version of this pipeline fed sparse
    keyframes (seconds apart) directly into `detect_pointer_for_frames`,
    which needs temporally ADJACENT frames -- so pointer evidence was
    always empty in practice, no matter how much real cursor motion was in
    the video. Fixed by using `detect_pointer_at` per keyframe (extracts
    its own closely-spaced adjacent frames). This checks the fix against
    the Step 5 ground-truth video: a 12x12 square moving from
    (20+40t+6, 20+20t+6)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=gray:size=320x240:rate=10:duration=3",
            "-f", "lavfi", "-i", "color=c=white:size=12x12:rate=10:duration=3",
            "-filter_complex", "[0][1]overlay=x='20+t*40':y='20+t*20':shortest=1",
            "-pix_fmt", "yuv420p", p,
        ], capture_output=True, check=True)

        config = PipelineConfig(vision_enabled=False, keyframe_interval_sec=0.5,
                                 keyframe_diff_threshold=0.0,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)

        detected = [so for so in result.structured_observations
                    if any(e.kind == "pointer" for e in so.observed)]
        assert len(detected) >= 3, "real cursor motion should produce multiple pointer detections"
        for so in detected:
            pointer_evidence = next(e for e in so.observed if e.kind == "pointer")
            expected_x = 20 + 40 * so.timestamp_sec + 6
            expected_y = 20 + 20 * so.timestamp_sec + 6
            assert f"x={round(expected_x)}" in pointer_evidence.ref
            assert f"y={round(expected_y)}" in pointer_evidence.ref


def test_max_frames_ceiling_is_respected():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p, duration=6.0)  # several 2s-interval keyframe candidates
        config = PipelineConfig(vision_enabled=False, pointer_enabled=False, max_frames=2,
                                 keyframe_interval_sec=1.0,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)
        assert len(result.structured_observations) <= 2


# ---------------------------- error propagation ----------------------------

def test_ingestion_failure_is_not_swallowed():
    try:
        analyze_video("this/path/does/not/exist.mp4")
        assert False, "a missing video must raise, not silently produce an empty result"
    except Exception as e:
        assert "does not exist" in str(e) or "not exist" in str(e).lower()


# ------------------------------ full pipeline ------------------------------

def test_full_pipeline_local_video_no_network_no_api_key():
    """Every stage exercised for real (ingestion, transcription, frame
    selection, pointer detection, vision [honestly unavailable without a
    credential], evidence correlation) and wired together correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "s.wav")
        if not _speech_wav(wav, "This is a short test recording for the pipeline."):
            print("  (skipped test_full_pipeline_local_video_no_network_no_api_key -- no TTS available)")
            return

        p = os.path.join(tmp, "v.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:size=320x240:rate=10:duration=6",
             "-i", wav, "-shortest", "-pix_fmt", "yuv420p", p],
            capture_output=True, check=True,
        )

        config = PipelineConfig(
            frame_cache_dir=os.path.join(tmp, "fc"), vision_cache_dir=os.path.join(tmp, "vc"),
            keyframe_interval_sec=2.0, tolerance_sec=1.5,
        )
        result = analyze_video(p, config)

        assert result.video.duration_sec > 0
        assert len(result.structured_observations) >= 1
        # timestamps stay ordered and within the video
        timestamps = [so.timestamp_sec for so in result.structured_observations]
        assert timestamps == sorted(timestamps)
        assert all(0.0 <= t <= result.video.duration_sec + 1.0 for t in timestamps)
        # vision is real code running (honestly unavailable without a credential),
        # never silently dropped from the result shape
        assert any("vision" in so.unavailable for so in result.structured_observations) or \
               any(o.vision is not None for o in result.observations)
        # no inference exists without at least one cited piece of evidence (contract-enforced,
        # re-checked here at the whole-pipeline level, not just the unit level)
        for so in result.structured_observations:
            for inf in so.inferences:
                assert len(inf.supporting_evidence) >= 1


if __name__ == "__main__":
    test_config_defaults_require_no_setup()
    test_config_is_mutable_per_call_not_global_state()
    test_vision_disabled_produces_no_vision_evidence()
    test_pointer_disabled_produces_no_pointer_evidence()
    test_no_audio_video_still_produces_frame_based_observations()
    test_pointer_stage_finds_real_motion_against_known_trajectory()
    test_max_frames_ceiling_is_respected()
    test_ingestion_failure_is_not_swallowed()
    test_full_pipeline_local_video_no_network_no_api_key()
    print("All pipeline tests passed.")
