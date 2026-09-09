# Component Decisions

Verified in this environment (Windows, ARM/Snapdragon, no dedicated GPU) on 2026-09-02.

---

Component: **FFmpeg** (9.0-full_build, gyan.dev Windows build)
Purpose: video/audio decode, metadata, frame extraction, format handling
Maturity: extremely mature, industry standard
License: GPL (this build; LGPL builds also exist)
What we need from it: `ffprobe` for metadata, `-frames:v 1` for frame grabs
What we do NOT need: encoding/streaming features, most of the 300+ filters
Integration method: subprocess (`core/video_metadata.py`, `adapters/frames/`)
GPU requirement: none for our usage (CPU decode is fine at this scale)
Decision: **USE**
Reason: already installed, zero-install baseline, no viable alternative for this role.

---

Component: **ffmpeg's built-in `whisper` audio filter** (whisper.cpp integration, ships in ffmpeg 7.1+)
Purpose: local speech-to-text with timestamps
Maturity: whisper.cpp is mature; the ffmpeg filter wrapper is newer
License: MIT (whisper.cpp) / GPL (ffmpeg build)
Timestamp quality: **fails** — systematic cumulative drift, see below
CPU behavior: excellent, 10.3x realtime, 240MB peak RAM (fastest option measured)
Long-video behavior: **unusable** — drift grows without bound with duration
Integration complexity: very low (subprocess, no Python ML dependency)
Decision: **REJECT** (was USE in Steps 1–2; removed in Step 3)
Reason: measured on a 6-minute file with independently verified speech onsets
(ffmpeg `silencedetect`), the filter's error grows linearly with absolute time:
corr(|error|, time) = **+0.97**, vs corr(|error|, position-within-chunk) = +0.02.
The filter processes 479232-sample chunks (29.952s) and its reported offsets fall
~48ms short per chunk, accumulating to ~0.73s at 350s and extrapolating to **~19s
on a 3-hour video**. Two independent measurements on the same file (silencedetect
and faster-whisper) confirmed the file itself is drift-free, isolating the fault to
the filter. Attempted a configuration fix first — aligning `queue` to an exact
multiple of 1024 samples (29.952s) — which produced **identical** output, so the
drift is not tunable. Since Video-Lens's core promise is speech↔frame alignment,
unbounded drift is disqualifying regardless of its speed advantage.
`adapters/transcription/ffmpeg_whisper.py` was deleted.

---

Component: **faster-whisper** (CTranslate2 Whisper) — `faster-whisper>=1.2,<2`
Purpose: local speech-to-text with timestamps and optional word-level timings
Maturity: mature, the de-facto standard for CPU Whisper inference; actively maintained
License: MIT
Timestamp quality: **good** — corr(|error|, time) = −0.04 (no systematic drift); 62ms mean onset error with the `base` model
CPU behavior: 2.4–3.5x realtime on this machine (ARM64 under x64 emulation, no GPU), 451MB peak RAM, flat with duration
Long-video behavior: streams from file; RAM independent of length; no drift accumulation
Integration complexity: low — Python API, one adapter (`adapters/transcription/faster_whisper_adapter.py`)
GPU requirement: none (`device="cpu"`, `compute_type="int8"`); `device="cuda"` optional
Decision: **USE**
Reason: correct where the incumbent was wrong, in the one dimension Step 3 declares
core. Also supplies genuine word-level timestamps (off by default, ~1.5x cost) and
built-in VAD, which prevents Whisper from inventing text over silence — measured:
20s of pure silence yields a fabricated "You" without VAD, zero segments with it.
Costs ~2-3x the runtime of the ffmpeg filter (74 min for a 3-hour video, offline,
one-time) and adds ctranslate2/tokenizers/huggingface-hub, which is accepted as the
price of a temporal layer that can actually be trusted.

---

Component: **openai-whisper** (reference implementation)
Purpose: local speech-to-text
Maturity: mature (reference implementation)
License: MIT
Timestamp quality: comparable to faster-whisper (same model weights)
CPU behavior: substantially slower than CTranslate2; requires PyTorch
Long-video behavior: adequate
Integration complexity: high — pulls in PyTorch (multi-GB) on an ARM64 machine running x64-emulated Python
Decision: **REJECT**
Reason: same models as faster-whisper but with a multi-GB PyTorch dependency and
slower CPU inference, for no accuracy gain. Not installed or benchmarked — the
dependency weight alone disqualifies it against an equivalent-quality alternative
that needs none of it.

---

Component: **Native captions / subtitles via yt-dlp** (manual + auto-generated)
Purpose: pre-existing timestamped text for URL sources
Maturity: mature source (YouTube); would need a VTT/JSON3 parser on our side
License: n/a (content)
Timestamp quality: millisecond precision, but **display** timings, not speech timings — a measured caption held one phrase ~3s past when it was spoken
CPU behavior: free (no inference)
Long-video behavior: free
Integration complexity: moderate — availability probing, format selection, parser, non-speech-annotation filtering
Decision: **DEFER**
Reason: evaluated against Whisper on the same real video. Human captions did win on
text accuracy ("long trunks" vs `tiny.en`'s "warm pumps"). Deferred anyway because:
they exist only for URL sources (local files, the primary case, still need Whisper —
so this is a second path, not a replacement); their display-oriented timings would
corrupt the single consistent timeline this layer exists to guarantee; and
auto-captions are themselves ASR, offering no accuracy edge over local Whisper while
still requiring a parser. If revisited, the sound use is as a *text* improvement
aligned onto Whisper's timings, never as a timestamp source. See `docs/speech.md`.

---

Component: **PySceneDetect** (`scenedetect`, PyPI 0.7.1)
Purpose: detect scene-change timestamps to pick which frames matter
Maturity: mature, actively maintained, widely used
License: BSD-3-Clause
What we need from it: `ContentDetector` scene-boundary timestamps only
What we do NOT need: its CLI, its own video export/save-images pipeline, HSV/adaptive detectors (for now)
Integration method: direct Python import (`adapters/frames/ffmpeg_scenedetect.py`); ffmpeg does the actual frame extraction at the timestamps it returns
GPU requirement: none
Decision: **USE**
Reason: pure-Python + numpy, installs cleanly with no OpenCV/heavy CV stack forced on us, does exactly the one thing we need (scene boundaries) well. Verified against a real video in the smoke test.

---

Component: **yt-dlp**
Purpose: resolve a URL to a local video file (ingestion)
Maturity: mature, very actively maintained
License: Unlicense
What we need from it: download-to-file for a given URL
What we do NOT need: its extraction of most of its ~1800 supported sites beyond what's actually used, post-processing/embedding features
Integration method: DEFERRED — not wired in this step (ingestion from URLs is out of scope for Step 1, which only verifies local-file processing)
GPU requirement: none
Decision: **DEFER**
Reason: confirmed installable (`pip install yt-dlp`, resolves cleanly), no reason to reject, but URL ingestion isn't part of Step 1's scope. Add as an ingestion adapter when that capability is actually built.

---

Component: **bradautomates/claude-video**
Purpose: an existing video-understanding workflow combining video + Claude
Maturity: unknown/small project, not independently verifiable from this environment (no local clone, and importing an entire external agent workflow was explicitly out of scope to attempt blindly)
License: unknown
What we need from it: potentially its approach to combining frames + transcript context for a multimodal prompt (a pattern, not code)
What we do NOT need: the entire repo, any agent/orchestration layer it may have
Integration method: N/A
GPU requirement: unknown
Decision: **DEFER**
Reason: no mechanism from it is needed yet — Step 1 doesn't reach the multimodal-understanding stage (that's the Vision adapter, deferred). Revisit for design inspiration only, when the Vision adapter is actually built in a later step. Will not be imported as a dependency regardless — "one mechanism, not the whole cake."

---

Component: **Live cursor/pointer tracking libraries** (e.g. `pynput`, `pyautogui`-style OS cursor APIs)
Purpose: track a live OS mouse cursor
Maturity: mature
License: varies (mostly permissive)
What we need from it: nothing
What we do NOT need: all of it
Integration method: N/A
GPU requirement: N/A
Decision: **REJECT**
Reason: these read the live operating-system cursor position via OS APIs. Video-Lens's actual requirement is the opposite problem: recovering a cursor's pixel location from a video that's *already recorded*, where no OS-level signal exists anymore — only pixels. These libraries are the wrong mechanism entirely, not just heavier than needed.

---

Component: **Cursor-in-video detection** (recovering a baked-in pointer from recorded frames)
Purpose: detect/track a screen-recording cursor from pixels alone
Maturity: N/A — no single mature, general-purpose off-the-shelf library found for this exact problem (arbitrary OS cursor icon, arbitrary recording, no live signal); confirmed again by re-investigating in Step 5
License: N/A
What we need from it: pointer x/y + confidence per frame (see `PointerAdapter` in `core/interfaces.py`)
What we do NOT need: N/A
Integration method: `adapters/pointer/cursor_detector.py` -- three-frame differencing (OpenCV `absdiff`/`threshold`/`findContours`) + size/solidity/scene-motion-veto/trajectory-continuity filtering
GPU requirement: none -- CPU-only, no training data or model weights involved
Decision: **USE** (was DEFER in Step 1; implemented in Step 5)
Reason: classical motion-based detection reached pixel-exact accuracy on a synthetic ground-truth trajectory and correctly rejected every tested false-positive trap (large blinking UI block, full-frame scene transition, no motion) with zero false positives, without training data or a GPU. See `docs/pointer.md` for the full investigation (why three-frame beat two-frame differencing, what else was considered and rejected: template matching, optical flow, single-frame heuristics, ML). Still flagged as the least mature Video-Lens component -- real-video detection rate (~19% on a real 1080p tutorial) is honestly much lower than the synthetic case (~80%), by design (uncertain > wrong).

---

Component: **OpenCV (`opencv-python`)** for pointer motion detection
Purpose: `absdiff`/`threshold`/`findContours` for cursor-blob detection
Maturity: extremely mature, industry standard
License: Apache 2.0
What we need from it: three primitives (frame differencing, thresholding, contour extraction) -- not its DNN/ML modules, not its GUI modules
What we do NOT need: everything else in the package (~90MB wheel)
Integration method: direct import in `adapters/pointer/cursor_detector.py`
GPU requirement: none -- CPU build, no CUDA
Decision: **USE**
Reason: rung 5 of the dependency ladder ("already-installed dependency solves it") -- `opencv-python` was already present as a transitive dependency of `scenedetect` (confirmed via `pip show scenedetect`), so using it for pointer detection adds zero new install weight to the environment. Hand-rolling contour detection over raw numpy arrays would be strictly worse: more code, slower, and reinventing a mature, battle-tested primitive for no benefit. Now pinned explicitly in `requirements.txt` since it's imported directly, not just relied on transitively.

---

Component: **Three-frame differencing** (vs. two-frame differencing, optical flow, template matching, ML) for pointer motion detection
Purpose: isolate a moving cursor's position from consecutive video frames
Maturity: standard classical CV technique
License: n/a (technique, not a library)
Timestamp/positional quality: measured **pixel-exact** (0px error) on 8/10 samples of a synthetic ground-truth video with a known linear trajectory; the remaining 2 are the sequence's first/last frame, honestly `not_detected` (no adjacent frame to complete a triple)
CPU behavior: three OpenCV calls per candidate frame (`absdiff` x2, `bitwise_and`, `findContours`) -- cheap, no measurable GPU benefit at this scale
Integration complexity: low -- `adapters/pointer/cursor_detector.py`, no new dependency (see OpenCV entry above)
Decision: **USE**
Reason: two-frame differencing was tried first and produces an inherent vacated/arrived blob ambiguity for any translating object -- confirmed by testing (confidence capped at "uncertain" for every correct detection, since the algorithm itself couldn't tell the two blobs apart). Three-frame differencing's `diff(a,b) AND diff(b,c)` resolves this directly: only the object's true position at `b` survives the intersection. Optical flow and template matching were also investigated and rejected as unnecessary/non-generalizing for this step's scope -- see `docs/pointer.md` for the full comparison table. No ML model was introduced; classical methods were not demonstrably insufficient, per this step's explicit "document why before adding ML" requirement.

---

Component: **ffmpeg input seeking (`-ss` before `-i`) for direct timestamp extraction**
Purpose: extract one frame at an arbitrary timestamp, fast
Maturity: standard ffmpeg usage
License: n/a (already-used tool)
Timestamp quality: measured identical accuracy to output seeking (`-ss` after `-i`) on a
video deliberately built with a single GOP to expose fast-seek keyframe inaccuracy;
this ffmpeg build (9.0) apparently decodes forward to the exact frame either way.
CPU behavior: ~4x faster than output seeking (0.17s vs 0.74s to reach 55s into a 60s
640x480 video) -- avoids decoding all preceding frames.
Integration complexity: low -- one subprocess call (`adapters/frames/frame_extractor.py`)
Decision: **USE**
Reason: faster with no measured accuracy cost. Documented rather than assumed, per Step 4
scope -- see `docs/frames.md`. Frame-exact seeking is not treated as a hard guarantee
across every container/codec, so `Frame.frame_index` is labeled a best-effort estimate.

---

Component: **Average-hash (aHash) image similarity** (hand-implemented, no library)
Purpose: detect near-duplicate frames to reduce redundant candidates
Maturity: well-known classical CV technique
License: n/a
What we need from it: a cheap way to skip re-decoding near-identical frames
What we do NOT need: a full perceptual-hashing library (`imagehash`, OpenCV)
Integration complexity: trivial -- one ffmpeg thumbnail call + threshold-against-mean
Decision: **REJECT** (implemented, tested, then replaced within this step)
Reason: aHash thresholds each pixel against the image's *own* mean. On a perfectly flat
frame every pixel equals the mean, so the hash is all-zero bits regardless of color --
measured IDENTICAL hashes for solid red and solid green frames. Flat/near-flat frames
(slides, simple UI) are common in exactly Video-Lens's target domain (screen recordings),
so this isn't a corner case. Replaced with a direct 8x8 grayscale thumbnail diff
(mean absolute difference, threshold 10/255), which has no such blind spot and still
needs no new dependency. See `docs/frames.md`.

---

Component: **Raw 8x8 grayscale thumbnail diff** (hand-implemented, no library)
Purpose: near-duplicate frame reduction (replaces the rejected aHash above)
Maturity: simple, well-understood technique ("low-resolution pixel comparison")
License: n/a
Timestamp/accuracy quality: correctly distinguishes solid-color frames that aHash could
not; measured MAD ~0-2 for a re-decoded identical frame vs. far higher for a real change
CPU behavior: one ffmpeg subprocess per frame (`scale=8:8,format=gray` -> raw bytes), trivial
Known limitation: grayscale is hue-blind -- two colors of near-identical luma (ffmpeg's
named "red" and "green" measured ~76 vs ~75) are indistinguishable. Documented rather
than fixed with full RGB diffing, since no target content (tutorials/IDEs/browsers/
charts) exercises that failure mode.
Decision: **USE**
Reason: correctly handles the flat-frame case that disqualified aHash, no new dependency,
measured effective at separating a genuine color change from re-encode noise.

---

Component: **A separate Python Whisper package** (`openai-whisper`, `faster-whisper`)
Decision: **SUPERSEDED in Step 3** — this Step 1 entry rejected both packages on the
grounds that ffmpeg's filter "already does timestamped local transcription with no
extra Python ML stack", and said to revisit "if the ffmpeg filter proves
insufficient". It did: Step 3 measured unbounded timestamp drift in that filter.
See the faster-whisper (USE) and openai-whisper (REJECT) entries above, which
replace this one.

---

Component: **Claude API (`anthropic` Python SDK)** for multimodal vision analysis
Purpose: turn a selected frame (+ optional transcript/pointer context) into a
structured visual observation -- description, visible text, located elements
Maturity: official Anthropic SDK, mature, actively maintained
License: MIT (SDK); API usage is metered/paid
What we need from it: one multimodal `messages.create` call per selected frame,
returning JSON we parse into `VisionObservation`
What we do NOT need: tool use, agents, batching, streaming (single-frame requests
are small and fast enough for a synchronous call)
Integration method: `adapters/vision/claude_vision.py`, `ClaudeVisionAdapter`
GPU requirement: none -- inference runs on Anthropic's infrastructure; the local
process only reads/base64-encodes a JPEG and makes an HTTPS request
Decision: **USE**
Reason: this machine has no usable GPU (documented since Step 1/3), and the target
content -- screen recordings full of small UI/chart text -- needs real OCR-grade
reading and spatial reasoning that a CPU-feasible local VLM (moondream-class) isn't
strong enough at, while a local VLM strong enough to be reliable (7B+ params) would
effectively require a GPU this project deliberately avoids requiring. A hosted call
needs zero local compute/weights, at the honest cost of a network dependency and
per-call price -- justified because Step 4's frame selection already keeps the
number of analyzed frames small (not every frame of a video), and results are
cached. See `docs/vision.md` for the full alternatives comparison (local VLMs,
Tesseract+separate model, a generic object detector).

---

Component: **Local VLM** (moondream2-class small model, or LLaVA/Qwen2-VL-class larger model)
Purpose: same as above, run locally instead of via API
Maturity: moondream2 is a real, actively-used small VLM; LLaVA/Qwen2-VL are mature
mid-size VLMs
License: varies by model (mostly Apache 2.0/MIT-family)
What we need from it: reliable OCR + spatial reasoning on screen-recording content
What we do NOT need: general chat capability, most of the pretraining domain
Integration method: none built
GPU requirement: moondream-class runs on CPU but slowly and with weaker
OCR/spatial-reasoning quality for this project's small-on-screen-text content;
LLaVA/Qwen2-VL-class (7B+ params) needs a GPU for usable latency
Decision: **REJECT**
Reason: the two size classes fail in opposite directions -- small enough to run
acceptably on CPU means not strong enough at this project's actual content (dense
small UI/chart text), and strong enough to be reliable means requiring the GPU
this project's hardware philosophy explicitly avoids. Neither class was
demonstrably sufficient enough to justify a multi-GB local dependency and a new
inference stack over a hosted call that needs neither. See `docs/vision.md`.

---

Component: **Tesseract (OCR) + a separate local scene-description model**
Purpose: split "read the text" and "describe the scene" into two specialized local
tools instead of one multimodal call
Maturity: Tesseract is extremely mature; scene-description models vary
License: Tesseract is Apache 2.0
What we need from it: text extraction + a separate description/element list
What we do NOT need: n/a
Integration method: none built
GPU requirement: Tesseract is CPU-fine; the paired description model reintroduces
the local-VLM GPU/CPU tradeoff above
Decision: **REJECT**
Reason: solves only half of "one structured observation per frame" each, doesn't
produce spatial/relationship reasoning between text and other visual elements
(Tesseract has no scene understanding), and still needs a second model for the
non-text half -- two local dependency stacks for a worse-integrated result than one
hosted multimodal call. Not "take the piece, not the whole cake" -- this would have
been taking two separate cakes for one dish.

---

Component: **LLM-based evidence correlation/structuring** (calling Claude to build StructuredObservations)
Purpose: turn correlated raw evidence into the observed/inferred structure Step 7 needs
Maturity: n/a -- would be an application of an existing mature API (Claude), not a
component itself
License: n/a
What we need from it: n/a
What we do NOT need: n/a
Integration method: none built
GPU requirement: n/a
Decision: **REJECT**
Reason: three deterministic rules (pointer-in-region geometry, pointer-motion-vs-
stationary displacement, missing-stream bookkeeping) already produce the required
observed/inferred/disagreement/unavailable structure directly from data Steps 3-6
already computed, with no ambiguity to resolve that needs a model's judgment. An
LLM call here would make correlation non-deterministic and non-reproducible
(`test_deterministic_repeated_calls_produce_identical_result` in
`tests/test_evidence.py` would be impossible to write meaningfully), slower, and
metered, for a task that is really just geometry and arithmetic over timestamps and
normalized coordinates. See `docs/evidence.md`.

---

Component: **A vector database / RAG pipeline for evidence retrieval**
Purpose: let a downstream consumer query "what happened near timestamp X" via
similarity search instead of direct timestamp correlation
Maturity: many mature options exist (Chroma, FAISS, pgvector, ...)
License: varies
What we need from it: nothing -- Video-Lens's evidence streams are already
timestamp-indexed and small in count per video (a handful of frames/pointer
events/vision observations per selected segment, not millions of embedded chunks)
What we do NOT need: semantic similarity search, embeddings, an index to maintain
Integration method: none built
GPU requirement: n/a
Decision: **REJECT**
Reason: explicitly out of scope per this project's own non-goals ("not a vector DB
or RAG platform" -- `docs/architecture.md`) and unjustified at Video-Lens's actual
scale: `nearest_within_tolerance`/`all_within_tolerance` (`core/evidence.py`) answer
every correlation question this step needed with two small linear scans, no index,
no embeddings, no new infrastructure.

---

Component: **A new `VideoEvidence`-named parallel result contract**
Purpose: represent timestamp + frame + speech + pointer + vision as one joined object
Maturity: n/a -- would be new project code
License: n/a
What we need from it: exactly what `MultimodalObservation` (Step 1, extended Step 6)
already provides
What we do NOT need: a second contract with the same shape
Integration method: none built
GPU requirement: n/a
Decision: **REJECT** (in favor of extending `AnalysisResult`/adding `StructuredObservation`)
Reason: `MultimodalObservation` already existed since Step 1 with exactly the
"frame + speech + pointer + vision, joined by timestamp" shape the Step 7 prompt
asked for under the name "VideoEvidence" -- inventing a same-shaped parallel type
would have been pure duplication. What Step 7 genuinely needed and didn't have yet
was the observed/inferred/disagreement/unavailable structure (`StructuredObservation`,
`Inference`) and the correlation logic to build it (`core/evidence.py`) -- both new,
neither redundant with anything existing. See `docs/evidence.md`.
