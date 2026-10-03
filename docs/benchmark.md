# Real-world benchmark (Step 7)

What Steps 1–6 actually do on real footage. Nothing in the pipeline was changed
for this; findings are classified, not fixed. Reproduce with
`scripts/benchmark_real_world.py`; synthetic pins of the reproducible defects are
in `tests/test_benchmark_findings.py`.

## Environment

Python 3.11.0, Windows 11 (10.0.26200), OpenCV 5.0.0, numpy 2.4.6, faster-whisper
1.2.1 (`base`, int8), ffmpeg 9.0. **CPU only, no GPU, no cloud calls, no cost.**
No vision-capable local model (Ollama holds only an embedding model) and no API
key, so real per-frame vision and a real native provider were **not** run.

## Corpus (both pre-existing local files; nothing downloaded)

| | Git explainer | DemoCreator tutorial |
|---|---|---|
| duration / res / fps | 257.65 s, 1280×720, 25 | 269.99 s, 1920×1080, 60 |
| content | animated diagrams, always-on webcam overlay (bottom-right, 4.8% of frame) | alternates **full-frame talking head** with **screen recordings** (real cursor, menus, dialogs, panel changes), webcam circle in a corner |
| why | webcam-overlay motion; slide/diagram transitions; **no cursor at all** (cursor false positives) | natural presenter motion; UI changes; real cursor |

No pure webcam-only clip exists locally; the full-frame talking-head shots of the
second video stand in for it.

## Results

| | Git | Tutorial |
|---|---|---|
| transcript | ok, 55 seg, en | ok, 113 seg, en |
| frame selection (default `max_frames=20`) | 16 frames, 0–251 s | **20 frames, 0–64 s of 270 s** |
| frame selection (`max_frames` raised) | 16 | 80 |
| targeted inspection | ok (see P0-A) | ok |
| pointer detections @0.5 s | 89 detected / 232 uncertain / 193 none of 514 | 37 / 281 / 220 of 538 |
| cursor segments | 10 moving (6.5 s), 70 uncertain | 6 moving (3.5 s), 102 uncertain |
| visual change @2 s, default | 23 of 128 pairs | 84 of 134 pairs |
| per-frame vision, native provider | **fakes only** (see gaps) | fakes only |
| session persist / load / retrieve | ok | ok |
| timings (CPU) | whisper 109 s; keyframes 126 s | whisper 85 s; keyframes 148–200 s |

## P0-A targeted inspection — confirmed working

- Returned timestamp equals the request; the frame matches an independent
  `ffmpeg select=eq(n,…)` decode to within ±1 frame (nearest reference offset 0 or ±1).
- Windows: `4s@2fps`→9 frames/0.5 s steps, `2s@5fps`→11, `1s@1fps`→2. Steps at 5 fps
  are 0.199/0.200/0.201 s (timestamps floor to ms) — cosmetic.
- Crop `(0.5,0.5,1,1)` → exactly 640×360 / 960×540, MAD vs numpy crop 0.15–0.31. An
  unaligned region yields even-sized output. Scale → 320×180, aspect preserved.
- Four request variants of the same moment give four distinct images; identical
  across fresh caches and after interleaving; **0 ffmpeg calls** on repeats.
- **Bug (reproducible, F3): colors are wrong for BT.709 sources.** MAD against a correct decode is
  9.6/255 (max 48 on one channel) on the saturated tutorial, ~0.5 on dark neutral
  content, exactly 0 for gray. Cause: yuvj420p JPEG without a matrix conversion.
  `-vf scale=out_color_matrix=bt601` brings it to 1.1. Relative measures
  (visual change) are unaffected; absolute color (what a vision model is shown) is.

## P0-B visual change

Dense 0.5 s grid, re-sampled to 0.5/1/2/4 s; default threshold 0.08.

- **True positives (Git, 2 s, 23 events, viewed):** essentially all are real slide /
  diagram / window transitions with tight, sensible regions. **Webcam overlay:
  0 detections from webcam motion alone** (its max contribution is 0.026 of frame).
- **Miss (acceptable limitation, calibration finding): sparse dark slides.** 52 of 128
  Git pairs fall in 0.03–0.08; the top ones (0.054–0.074) are real transitions
  (title → diagram, new diagram elements). Changed *area* is small even when
  the whole slide changes. Lowering to 0.005 flags 497 of 513 pairs: this video
  has a noise floor (5th percentile 0.0055), so there is no usable lower value.
- **False positives (reproducible, calibration finding): full-frame talking head.**
  Inside labelled talking-head intervals the default flags 67/98 pairs at 0.5 s,
  46/48 at 1 s, 22/22 at 2 s; median magnitude 0.103 (screen content: 0.013).
  This contradicts the documented calibration ("presenter motion p95 = 0.064");
  these shots contain gestures and shot edits. Magnitude alone cannot separate
  presenter motion from a UI change here.
- Of 20 hand-picked cut/transition times (talking head ↔ screen, title cards), 16 fall inside a
  detected 2 s pair; the 4 misses are low-magnitude (0.01–0.06, title fades) — and my
  times are approximate, so treat this as indicative.
- Brightness shift: +10/+20 → 0.0 (invisible), +30 → 0.97–0.99 global: a cliff at the
  per-pixel threshold of 25. A 0.5 % patch or one added text line (0.2–0.5 %) is
  missed at the default and found at 0.005.
- **Transients:** none found at 0.5 s (no blip appears and disappears within a 2 s step in this corpus) — a benchmark gap, not proof.
- **Architectural finding: pipeline-level `visual_change` is selection-biased.** It compares the
  *de-duplicated keyframes*, which were chosen because they differ. Git: 16 keyframes
  over 258 s, gaps up to **170 s** (44 s → 214 s), and every consecutive pair "detected". The event's
  `compared` interval is honest, but "a change happened" over 170 s is
  nearly vacuous. The detector is sound; its pipeline input is the problem.
  Requires an approved design decision. No threshold was changed.

## P0-C cursor intelligence

- Output vocabulary is only `moving` / `uncertain`; nothing says stationary, click,
  drag or intent (checked on every segment).
- **Verified true positive** (tutorial 110.75–111.25 s, viewed): highlighted cursor moves
  (437,798)→(782,792)→(1118,537)→(1194,381); reported direction 339° (measured 339°), speed 0.70
  diag/s (measured ≈0.66).
- **Precision is low (reproducible, detector limitation):** of 16 random "detected" positions on the
  tutorial, 7 sit on a real cursor or its highlight ring, 9 on presenter motion or
  UI. Of its 6 "moving" segments, 2 are the real highlighted cursor, 4 are not
  (webcam edge, presenter's hand ×2, a slider handle). On the Git video (no
  cursor) **0 of 16** sampled detections are a cursor and 10 "moving" segments were reported.
  The detector finds a moving *blob*, not a cursor; confidence (0.45–0.9) does not reflect that.
- **Recall is low** on slow/small movement. Window 100–112 s @0.25 s (template-matched oracle,
  19 visible frames): 13 detections, 5 within 60 px of the real cursor (median error 5 px);
  2 real moves of ≥3 % of the diagonal, 0 reported.
- Pipeline-supplied cursor evidence (keyframes mostly ≥2 s apart) is predominantly `uncertain`
  (`sampling_gap`, `pointer_not_observed`) — honest, and rarely informative.

## P0-D native provider seam — confirmed working (fakes only)

On both videos through `process_video`: native observations keep their timestamps
(26.0/129.0 s and 82.0/166.4 s); native evidence has `source="video_understanding"`
and per-frame vision `source=""`; identical text from both stays two items;
retrieval separates them with `sources=`; an invalid-timestamp observation is
dropped with a warning; a claim citing only native vision gets **no** bundled frame
while the per-frame-vision claim bundles one. No real provider was exercised.

## Step 5 evidence integrity — confirmed working

All six kinds appear in real output with correct spans, `source`, JSON `ref`s and
confidences; ordering is deterministic. A fake synthesizer's claims "The user changed
the chart timeframe" (cites `visual_change`) and "The user clicked the button"
(cites `cursor_track`) are both retained as **`inferred`**, verification
`measurement_evidence_only`, limitation attached — a measurement never becomes an
observed action.

## Step 6 session and retrieval — confirmed working

Persistence disabled: no session file, same package. Enabled: one file (45 KB / 134 KB), no JPEG/MP4 bytes, no job-workspace path
(the video's own source path is kept, as the package does), loads equal to the in-memory
session, every non-transcript evidence item present, no inference text. Packages
with and without a session are identical. The temp workspace is removed after
success. 400 random near/between/kind/source queries per video match an
independent reference implementation with 0 mismatches; ordering is stable under
shuffling; span queries return the covering span (a 26-second cursor segment at
its midpoint; a transcript span at its midpoint).

## End-of-video — reproducible bug (deferred)

Both videos: the video stream ends before the container duration (0.014 s / 0.35 frames and
0.027 s / 1.6 frames). `get_frame` clamps to `duration − 1/fps`, which is past the last
decodable frame, so **any request in the last frame interval, and any request at or
beyond the end, raises `FrameExtractionError: ffmpeg produced no frame`** — Git from 257.604 s, tutorial from
269.954 s. This defeats the clamp's own stated purpose. It also breaks
`inspect_visual_evidence` at/near the end. The full default pipeline did **not** hit it here:
keyframes stop at 250.8/266.4 s and the pointer step reaches only 0.2 s past a keyframe,
but a keyframe within ~0.25 s of the end would hit it. Synthetic repro: `tests/test_benchmark_findings.py`.

## `speech_pointer_stationary` — evidence-quality issue (deferred)

- In the default pipeline it **never fires** (0 inferences on both videos): it needs ≥2 pointer readings within ±1 s and
  keyframes are further apart.
- Fed dense pointer events (as `track_pointer` provides) it fires constantly: **180 "motion" and 11 "stationary"
  inferences on the Git video, which has no cursor**; 196 / 14 on the tutorial.
- It counts `uncertain` positions (4 of 11 and 9 of 14 stationary inferences rest only on uncertain readings), ignores
  `not_detected` readings between the first and last position (8 of 11, 8 of 14), and decides from first-vs-last displacement.
  P0-C reports `uncertain` for 10/11 and 11/14 of those same windows. It cannot fire from an
  undetected cursor alone (needs ≥2 positions).
- **Smallest correction, for a later approved step:** use only `status == "detected"` readings,
  require no `not_detected` reading between them, and drop the `stationary` basis (stillness
  isn't establishable — P0-C's rule); keep `motion` only under those gates.

## Other finding

- **`max_frames` is a head cut (reproducible bug / design decision).** `frames[:max_frames]`
  keeps the *first* 20 keyframes, so with defaults the tutorial is analyzed for 0–64 s of 270 s (pointer,
  vision, visual change all stop there) — and the cost of selecting every keyframe is paid first
  (148–200 s). The Git video escaped only because dedupe left 16 frames.

## Classification

- **Confirmed working:** P0-A (timestamps, crop, scale, determinism, cache identity); P0-D seam and provenance; Step 5
  mapping and semantic boundary; Step 6 persistence, retrieval and lifecycle; P0-C vocabulary; the true cursor
  movement above.
- **Acceptable limitation:** missing low-area changes at the default threshold; brightness cliff; low cursor recall;
  mostly-`uncertain` pipeline cursor evidence.
- **Reproducible bug:** end-of-video clamp; BT.709 colors; `max_frames` head cut; `speech_pointer_*` gates.
- **Requires architectural / calibration decision:** pipeline `visual_change` input (selection bias, 170 s gaps);
  talking-head false positives vs. the 0.08 default; the pointer detector reporting cursors where none exist.
- **Benchmark gap:** real per-frame vision; a real native provider; webcam-only footage; transients; human-labelled
  UI-change ground truth; footage longer than ~4.5 min; non-English or silent video.
- **Deferred (at Step 7):** all of the above fixes, by decision. Step 8 fixed the four reproducible bugs; see below.

## Step 8 remediation — before / after on the same two videos

| | before (Step 7) | after (Step 8) |
|---|---|---|
| `get_frame` at duration / duration+50 s | `FrameExtractionError` (Git from 257.604 s, tutorial from 269.954 s) | last frame: Git @257.600, tutorial @269.950 (= last packet pts) |
| `get_frame` at duration−0.1 s | ok | ok, same timestamp |
| JPEG colour vs exact float decode of raw YUV, MAD (4 timestamps) | tutorial 5.45–7.96; Git 0.48–0.83 | tutorial 0.98–1.18; Git 0.39–0.63 |
| colour on BT.601 / untagged fixtures | — | byte-identical to before |
| frames kept by `max_frames=20` (tutorial, 80 candidates) | 20, span 0–64 s | 20, span 0–266.4 s, max gap 28 s |
| `max_frames=20` (Git, 16 candidates) | 16 | 16 (unchanged, under budget) |
| `speech_pointer_*`, dense 0.5 s pointer readings, one query per reading | Git 343 motion / 28 stationary (15 from windows with no `detected` reading); tutorial 397 / 33 (19) | Git 38 motion / 0; tutorial 28 / 0 |
| visual change, Git, 2 s grid @0.08 | 23 of 128 detected | 22 of 128 (luma now converted correctly; noise floor unchanged) |

The `speech_pointer` rows re-apply the old rule verbatim to the same windows; the
query layout differs from Step 7's run, so the counts are larger than the 180/11 and
196/14 above. Git's remaining 38 motion inferences rest on `detected` readings in a video with no
cursor: the gate removes stillness claims and uncertain readings, not the detector's
false positives (P0-C).

Visual change in the pipeline was **unchanged and deferred** at Step 7.5 (resolved in Step 8, below). A uniform sampling path was
prototyped, but the package collects every evidence stream through the per-keyframe
observation windows (±tolerance), so grid measurements away from keyframes never
reach the package or session. Making them do so changes the Step 5/6 evidence flow. The
package now states in its `limitations` that visual-change items compare consecutive
selected keyframes and are not a continuous scan.

## Step 8 — visual-change measurement stream, before / after

Same videos, `visual_change_enabled=True`, defaults otherwise (2 s grid, cap 150
samples, threshold 0.08 unchanged). One `process_video` run per video, with a counting fake vision
provider, transcription stubbed and pointer off (neither touches visual change). "Before" is the
pre-Step-8 path on the **same** keyframes: consecutive keyframe pairs, collected through the
keyframe windows.

| | Git before | Git after | Tutorial before | Tutorial after |
|---|---|---|---|---|
| pairs measured | 15 | 129 | 19 | 135 |
| range covered | 0–250.8 s | 0–257.6 s (last frame) | 0–266.4 s | 0–269.95 s (last frame) |
| largest compared gap | **170 s** | 2 s | 28 s | 2 s |
| detected | 15 (every pair) | 22 (14 localized, 8 global) | 19 (all global) | 84 (79 global, 5 localized) |
| magnitude p10 / p50 / p90 | 0.087 / 0.144 / 0.293 | 0.017 / 0.034 / 0.093 | 0.210 / 0.579 / 0.900 | 0.018 / 0.188 / 0.742 |
| `KnowledgePackage.visual_changes` | 15 | 129 | 19 | 135 |
| session `visual_change` items | — | 129 | — | 135 |
| retrieved inside the largest keyframe gap | 0 (Git 44–214 s) | 84 (1 detected) | 0 (150–178 s) | 13 (2 detected) |
| keyframes / vision calls | 16 / 16 | 16 / 16, same timestamps | 20 / 20 | 20 / 20, same timestamps |

Cost of the measurement stage: inside the pipeline the grid reuses keyframe selection's
cached interval samples, so it took **2 ffmpeg + 1 ffprobe calls** (the end-of-stream fallback
for the last frame), 0.46 / 0.61 s of extraction and 1.35 / 5.35 s in total (Git / tutorial).
Cold cache, called alone: 131 / 137 ffmpeg seeks, 20.4 / 37.8 s, plus 0.75 / 4.81 s of detection.
A single sequential decode was not adopted: it decodes the whole video once (this AV1/1080p60
footage) for a saving that disappears inside the pipeline, where the cache already holds the grid.

Reading: before, "changed" was true of every keyframe pair because keyframes are picked for
differing; after, Git's 22 detections are its slide and diagram transitions on an evenly spaced
grid, and the 84 items in its 170 s keyframe gap are now measured (1 detected). The tutorial's
84 detections are mostly its full-frame talking head: genuine pixel change, not UI change. They are
reported as measured. Telling them apart is semantic classification, and is future work.
