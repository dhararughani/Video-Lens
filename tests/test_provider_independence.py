"""Model/provider-independence tests: proves Video-Lens's core pipeline and
contracts do not architecturally depend on Claude, Anthropic, or any other
specific AI provider -- vision is one optional, swappable capability behind
`core.interfaces.VisionAdapter`, not the foundation of the pipeline.

Fully synthetic/deterministic -- no video files needed for most tests, no
network, no real Anthropic API call anywhere (a fake provider is injected;
the real `anthropic` import is actively blocked in some tests to prove
optionality, not just observed to be unused).

Run: python tests/test_provider_independence.py
"""
from __future__ import annotations

import builtins
import glob
import importlib
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import Frame, PointerEvent, VideoInput, VisionObservation
from core.knowledge import to_json


@contextmanager
def _block_import(blocked_name: str):
    """Makes `import <blocked_name>` (and any submodule of it) raise
    ImportError for the duration of the block -- a real proof that code
    under test doesn't need the package, not just an observation that it
    happens not to import it in this environment."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == blocked_name or name.startswith(blocked_name + "."):
            raise ImportError(f"{blocked_name} is not installed (blocked for test)")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def _fresh_reimport(*module_names: str):
    """Drops cached modules so a subsequent `import` actually re-executes
    their top-level code (needed to test import-time behavior under
    _block_import, since a prior test/import may have already cached them)."""
    for name in module_names:
        sys.modules.pop(name, None)


class FakeVisionProvider:
    """A minimal stand-in implementing exactly the VisionAdapter shape --
    proves the pipeline only needs the interface, not Claude."""

    def __init__(self):
        self.calls = 0

    def analyze_frame(self, frame: Frame, transcript_context: str | None = None,
                       pointer: PointerEvent | None = None) -> VisionObservation:
        self.calls += 1
        return VisionObservation(
            timestamp_sec=frame.timestamp_sec, frame_path=frame.path, status="ok",
            description="a fake analysis from a non-Claude provider",
            visible_text=("FAKE",), confidence=0.99, model="fake-provider-v1",
        )


# ------------------------- 1. core imports without Anthropic -------------------------

def test_core_package_imports_without_anthropic_installed():
    """Blocks `import anthropic` entirely and re-imports every core module
    plus video_lens from scratch -- if any of them imported anthropic at
    module level (directly or transitively), this raises.

    Deliberately does NOT reload `core.contracts`: reloading it would create
    a second, distinct set of dataclass objects (e.g. a new `VisionObservation`
    class) diverging from the one already imported at this test file's top
    level, breaking every later `isinstance` check in this suite via normal
    Python module-identity semantics -- a test-harness pitfall, not something
    that needs proving here (its own zero-Anthropic-dependency is already
    checked by source inspection in test_claude_specific_code_does_not_leak_into_core_contracts)."""
    reloadable = ["core.interfaces", "core.errors", "core.evidence", "core.knowledge",
                  "core.workspace", "core.synthesis", "core.visual_evidence",
                  "video_lens", "adapters.vision.claude_vision"]
    _fresh_reimport(*reloadable)
    with _block_import("anthropic"):
        import core.interfaces  # noqa: F401
        import core.errors  # noqa: F401
        import core.evidence  # noqa: F401
        import core.knowledge  # noqa: F401
        import core.workspace  # noqa: F401
        import core.synthesis  # noqa: F401
        import core.visual_evidence  # noqa: F401
        import video_lens  # noqa: F401
        # even the Claude adapter module itself must import cleanly -- it
        # only imports anthropic lazily, inside a function, when actually used
        import adapters.vision.claude_vision  # noqa: F401
    _fresh_reimport(*reloadable)  # leave a clean slate for later tests


# ------------------------- 2. pipeline operates with no vision provider -------------------------

def _make_video(path: str, duration: float = 2.0):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:size=160x120:rate=5:duration={duration}",
         "-pix_fmt", "yuv420p", path], capture_output=True, check=True,
    )


def test_pipeline_operates_with_vision_disabled():
    from video_lens import PipelineConfig, analyze_video
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        config = PipelineConfig(vision_enabled=False, pointer_enabled=False,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)
        assert len(result.structured_observations) >= 1
        assert all(o.vision is None for o in result.observations)
        assert all("vision" in so.unavailable for so in result.structured_observations)


def test_default_vision_provider_degrades_gracefully_when_anthropic_unavailable():
    """Even with vision_enabled=True and no vision_provider supplied, the
    pipeline must not crash or require anthropic when it's genuinely
    unimportable -- this is the real graceful-degradation contract, not
    just "we happened not to configure a key"."""
    import video_lens as vl
    frame = Frame(timestamp_sec=1.0, path=__file__, width=10, height=10)  # any real file path
    with _block_import("anthropic"):
        _fresh_reimport("adapters.vision.claude_vision")
        provider = vl._default_vision_provider(vl.PipelineConfig())
        obs = provider.analyze_frame(frame)
    assert obs.status == "unavailable"
    _fresh_reimport("adapters.vision.claude_vision")


# ------------------------- 3. a fake VisionProvider works end to end -------------------------

def test_fake_vision_provider_can_be_injected_and_used():
    from video_lens import PipelineConfig, analyze_video
    fake = FakeVisionProvider()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        config = PipelineConfig(pointer_enabled=False, vision_provider=fake,
                                 frame_cache_dir=os.path.join(tmp, "fc"))
        result = analyze_video(p, config)
        assert fake.calls >= 1
        vision_evidence = [e for e in result.evidence if e.kind == "vision"]
        assert len(vision_evidence) >= 1
        assert any(o.vision is not None and o.vision.model == "fake-provider-v1"
                   for o in result.observations)


def test_fake_vision_provider_works_through_process_video_and_cleanup_still_happens():
    """Provider injection composes with the Step 9 lifecycle -- and storage
    cleanup behavior is unaffected by which provider ran."""
    import video_lens as vl
    fake = FakeVisionProvider()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        config = vl.PipelineConfig(pointer_enabled=False, vision_provider=fake,
                                    output_dir=os.path.join(tmp, "out"))
        captured = {}
        original_run_pipeline = vl._run_pipeline

        def _spy(source, cfg):
            result, transcript = original_run_pipeline(source, cfg)
            captured["frame_cache_dir"] = cfg.frame_cache_dir
            return result, transcript

        vl._run_pipeline = _spy
        try:
            package = vl.process_video(p, config)
        finally:
            vl._run_pipeline = original_run_pipeline

        assert fake.calls >= 1
        job_root = os.path.dirname(captured["frame_cache_dir"])
        assert not os.path.exists(job_root), "temp workspace must still be cleaned up on success"
        assert len(list(Path(config.output_dir).glob("*.json"))) == 1
        assert package.source.duration_sec > 0


# ------------------------- 4. Claude provider conforms to the generic contract -------------------------

def test_claude_provider_conforms_to_vision_adapter_shape():
    from adapters.vision.claude_vision import ClaudeVisionAdapter
    from core.interfaces import VisionAdapter
    adapter = ClaudeVisionAdapter(api_key=None)  # no credential -- fine, just checking shape/behavior
    assert hasattr(adapter, "analyze_frame")
    assert set(VisionAdapter.__protocol_attrs__ if hasattr(VisionAdapter, "__protocol_attrs__")
               else ["analyze_frame"]).issubset(dir(adapter))
    frame = Frame(timestamp_sec=0.0, path=__file__, width=10, height=10)
    obs = adapter.analyze_frame(frame)  # no credentials configured -> graceful, not a crash
    assert isinstance(obs, VisionObservation)
    assert obs.status == "unavailable"


# ------------------------- 5. no Claude/Anthropic leakage into core contracts -------------------------

def test_claude_specific_code_does_not_leak_into_core_contracts():
    repo_root = Path(__file__).resolve().parent.parent
    for relative in ("core/contracts.py", "core/interfaces.py", "core/errors.py",
                      "core/evidence.py", "core/workspace.py", "core/synthesis.py",
                      "core/visual_evidence.py"):
        src = (repo_root / relative).read_text(encoding="utf-8").lower()
        assert "anthropic" not in src, f"{relative} references anthropic"
        assert "claude" not in src, f"{relative} references claude"


def test_video_lens_module_only_imports_claude_lazily():
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "video_lens.py").read_text(encoding="utf-8")
    # no top-level "from adapters.vision.claude_vision import ..." / "import anthropic"
    for line in src.splitlines():
        if line.startswith(("import ", "from ")):  # unindented -> module (top) level
            assert "claude_vision" not in line, f"top-level Claude import found: {line!r}"
            assert "anthropic" not in line, f"top-level anthropic import found: {line!r}"
    assert "claude_vision" in src  # it's still used -- just not at module level


# ------------------------- 6. no downstream-consumer coupling -------------------------

def test_no_dhara_os_imports_anywhere():
    """No source file imports a specific downstream consumer's package --
    Video-Lens stays a standalone library regardless of who reads its output."""
    repo_root = Path(__file__).resolve().parent.parent
    py_files = [f for f in glob.glob(str(repo_root / "**" / "*.py"), recursive=True)
                if ".git" not in f]
    for f in py_files:
        src = Path(f).read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("import dhara") or stripped.startswith("from dhara"):
                assert False, f"downstream-consumer import found in {f}: {stripped!r}"


# ------------------------- 7. KnowledgePackage stays provider-neutral -------------------------

def test_knowledge_package_is_provider_neutral():
    from core.contracts import (
        AnalysisResult, Evidence, Inference, StructuredObservation,
    )
    from core.knowledge import build_knowledge_package
    ev = Evidence(timestamp_sec=1.0, kind="vision", ref="a fake provider's description")
    inf = Inference(text="the pointer was near a described element",
                     supporting_evidence=(ev,), confidence=0.5, basis="pointer_in_vision_region")
    so = StructuredObservation(timestamp_sec=1.0, tolerance_sec=1.0, inferences=(inf,))
    video = VideoInput(path="x.mp4", duration_sec=10.0, width=100, height=100,
                        fps=30.0, has_audio=False)
    result = AnalysisResult(video=video, structured_observations=[so])
    package = build_knowledge_package(result, transcript=None)
    blob = to_json(package).lower()
    assert "claude" not in blob
    assert "anthropic" not in blob
    # the schema itself carries no provider-specific field names
    from dataclasses import fields
    from core.contracts import KnowledgePackage, ProcessingMetadata
    field_names = {f.name for f in fields(KnowledgePackage)} | {f.name for f in fields(ProcessingMetadata)}
    assert not any("claude" in n.lower() or "anthropic" in n.lower() for n in field_names)


# ------------------------- 8. graceful degradation intact -------------------------

def test_missing_vision_credentials_do_not_fail_the_pipeline():
    from video_lens import PipelineConfig, analyze_video
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "v.mp4")
        _make_video(p)
        config = PipelineConfig(pointer_enabled=False, frame_cache_dir=os.path.join(tmp, "fc"))
        # default provider, real ANTHROPIC_API_KEY not required to be set here --
        # this must not raise regardless of whether one happens to be configured
        result = analyze_video(p, config)
        assert result.video.duration_sec > 0


# ------------------------- dependency split -------------------------

def test_anthropic_is_not_in_core_requirements():
    repo_root = Path(__file__).resolve().parent.parent
    core_reqs = (repo_root / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "anthropic" not in core_reqs
    vision_reqs = (repo_root / "requirements-vision.txt").read_text(encoding="utf-8").lower()
    assert "anthropic" in vision_reqs


if __name__ == "__main__":
    test_core_package_imports_without_anthropic_installed()
    test_pipeline_operates_with_vision_disabled()
    test_default_vision_provider_degrades_gracefully_when_anthropic_unavailable()
    test_fake_vision_provider_can_be_injected_and_used()
    test_fake_vision_provider_works_through_process_video_and_cleanup_still_happens()
    test_claude_provider_conforms_to_vision_adapter_shape()
    test_claude_specific_code_does_not_leak_into_core_contracts()
    test_video_lens_module_only_imports_claude_lazily()
    test_no_dhara_os_imports_anywhere()
    test_knowledge_package_is_provider_neutral()
    test_missing_vision_credentials_do_not_fail_the_pipeline()
    test_anthropic_is_not_in_core_requirements()
    print("All provider-independence tests passed.")
