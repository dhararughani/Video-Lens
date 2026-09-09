# Pointer / Cursor Intelligence

Detects a cursor/pointer **baked into video pixels** (a screen-recording
mouse arrow, a presentation pointer, a click-highlight ring) -- not a live
OS mouse. There is no live signal left once a video is recorded; only
pixels remain, so this is a small classical computer-vision problem, not a
`pynput`-style problem. See the "Cursor-in-video detection" entry in
`docs/component-decisions.md` for why no off-the-shelf library fit.

Most callers should use `video_lens.analyze_video()` (repository root),
which wires this into the full pipeline automatically
(`pointer_enabled`/`pointer_step_sec` in `PipelineConfig`). Read on for the
detector's own API if you're calling it directly.

```python
from adapters.ingestion import ingest
from adapters.frames.frame_extractor import FrameExtractor
from adapters.pointer.cursor_detector import detect_pointer_at, track_pointer

video = ingest("tutorial.mp4")
extractor = FrameExtractor()

event = detect_pointer_at(video, 38.1, extractor)   # one timestamp
track = track_pointer(video, 36.5, 40.0, extractor, interval_sec=0.5)  # a range
```

## Core design principle: never fabricate a position

A wrong cursor location is worse than no cursor location. Every result
carries a `status`: `"detected"`, `"uncertain"`, or `"not_detected"` --
`x`/`y` are only ever populated for the first two, and the dataclass
enforces this (`core/contracts.py`, `PointerEvent.__post_init__` raises if
`status != "not_detected"` and `x`/`y` are missing). Missing evidence stays
missing; it is never interpolated into a fabricated trajectory (see
Tracking below).

**For downstream consumers: `status="not_detected"` means "no motion
evidence was found in this window" -- it does NOT mean "there is no cursor
in this video."** This detector's only signal is motion (three-frame
differencing, see below); a real, visible, perfectly stationary cursor
produces `not_detected` every time, by construction, not as a bug (see
"Known limitation: a paused cursor is invisible to this method"). Never
treat `not_detected` as proof of cursor absence -- treat it as "no evidence
either way." `"uncertain"` similarly means "more than one plausible
position, no confident pick" -- not "probably wrong."

## PointerEvent contract

`PointerEvent` (in `core/contracts.py`): `timestamp_sec`, `status`, `x`,
`y`, `frame_width`, `frame_height`, `confidence`, `detection_method`, plus
computed properties `normalized_x`/`normalized_y`.

- `frame_width`/`frame_height` are stored (mirroring `Frame`'s provenance
  fields) so `normalized_x`/`normalized_y` (`0.0-1.0`, resolution-independent)
  are **computed properties**, not duplicated stored floats -- one source of
  truth for the pixel position, matching the task's "don't unnecessarily
  store duplicate information" guidance.
- `detection_method` — `"motion_diff_3frame"` (the primary path, used by
  `detect_pointer_for_frames`/`track_pointer`/`detect_pointer_at`) or
  `"motion_diff_2frame"` (the `detect_pointer(frame, prev_frame)` fallback,
  used when only one prior frame is available). Exists so a caller can
  weight three-frame results higher, since two-frame differencing has a
  documented ambiguity (see below).

`PointerTrack` (`start_sec`, `end_sec`, `events`, `confidence`) is an
ordered run of events across a range -- `confidence` is the fraction of
events with `status == "detected"`, so a caller gets an honest track-level
summary without recomputing it.

## Canonical timestamp

Same time model as speech and frames (`docs/speech.md`, `docs/frames.md`):
float seconds from video start. `PointerEvent.timestamp_sec` joins to
`Frame.timestamp_sec` and `TranscriptSegment.start_sec`/`end_sec` by plain
numeric comparison -- no new time system introduced.

## Detection mechanism: three-frame differencing

For an ordered sequence of frames, the position of a moving cursor **at**
frame `i` is estimated as the **intersection** of:

- the motion mask between frame `i-1` and frame `i`, and
- the motion mask between frame `i` and frame `i+1`.

A pixel that changed on both sides of frame `i` is where a moving object
sits *at* frame `i`. This is a standard classical-CV technique ("three-frame
differencing") chosen specifically to fix a problem the simpler two-frame
version has (see below). Implemented with OpenCV's `absdiff`/`threshold`/
`findContours` (mature, well-tested primitives -- not hand-rolled pixel
loops); OpenCV was already an installed dependency (`scenedetect` requires
it) and is now imported directly, so it's pinned explicitly in
`requirements.txt`.

Candidate blobs are filtered before anything is called a cursor:

| filter | threshold | rejects |
|---|---|---|
| size | 0.005%-1% of frame area | noise (too small) / large UI blocks, charts (too big) |
| solidity (contour area / bbox area) | > 0.4 | thin lines (chart lines, crosshairs) |
| scene-wide motion veto | > 15% of frame changed on either side | scroll, transition, full redraw |
| trajectory continuity | within 15% of frame diagonal from the last confirmed position | picks the plausible candidate when several exist |

Confidence: **0.9** ("detected") when exactly one blob survives filtering
after trajectory continuity narrows the pool -- an unambiguous signal.
**0.6** ("uncertain") when more than one plausible candidate remains even
after narrowing -- a real ambiguity, not hidden behind a forced pick.

### Why three-frame, not two-frame, differencing

A first version used plain two-frame differencing (`diff(a, b)` only). A
translating object produces **two** blobs there -- the position it left and
the position it arrived at -- and nothing in a single pairwise diff says
which is which. Measured on the synthetic ground-truth video (see below):
this pushed every detection's confidence down to "uncertain" even though
the position picked was pixel-correct, because the algorithm itself
couldn't justify calling it unambiguous. Three-frame differencing's
intersection resolves this directly: the vacated position only shows up in
`diff(a,b)`, the arrived position only in `diff(b,c)`, and the object's
actual position at `b` shows up in both. Kept the simpler two-frame version
only as `detect_pointer(frame, prev_frame)`'s fallback, for the case where
no third frame exists (documented as lower-confidence, `motion_diff_2frame`).

### Known limitation: a paused cursor is invisible to this method

If the cursor doesn't move between frame `i-1`/`i` or between `i`/`i+1`, one
side of the intersection is empty, so the blob doesn't appear -- a
genuinely stationary cursor detects as `not_detected`, not incorrectly. Per
the task's core design principle, this is the correct honest answer over a
guessed one, but it is a real detection-rate cost. This is why
`track_pointer`'s confidence is a *fraction detected*, not a promise of
continuous coverage.

## API

- `detect_pointer_for_frames(frames)` — the primary mechanism: three-frame
  differencing across an ordered list of `Frame`s.
- `detect_pointer_at(video, timestamp, extractor, step_sec=0.1)` — one
  timestamp; extracts the adjacent frame before/after via `FrameExtractor`
  (reused, not duplicated) for three-frame context.
- `track_pointer(video, start, end, extractor, interval_sec=0.5)` — a range,
  returns a `PointerTrack`.
- `detect_pointer(frame, prev_frame=None)` — the lowest-level, two-frame
  wrapper. Without `prev_frame` there is no motion signal at all, so this
  honestly returns `not_detected` rather than guessing from one static
  image (a single frame alone was investigated and rejected as a detection
  input -- see Repository/mechanism investigation below).

No implementation detail (OpenCV, ffmpeg) leaks into `core/` -- `core/interfaces.py`'s
`PointerAdapter` Protocol only names `Frame`/`PointerEvent` types.

## Tracking: no fabricated trajectories

`detect_pointer_for_frames` never bridges a gap in evidence. Each frame's
result depends only on that frame's own three-frame motion signal and the
last *confirmed* (`status == "detected"`) position for trajectory-continuity
narrowing -- there is no interpolation, no smoothing, no Kalman filter
papering over missing frames. A run like `detected, detected, not_detected,
not_detected, detected` stays exactly that; the missing middle is never
invented. `PointerTrack.confidence` (fraction of events `"detected"`) makes
this honest at the track level too.

## Repository / mechanism investigation

No mature, general-purpose off-the-shelf library solves "recover an
arbitrary OS cursor icon's position from arbitrary recorded pixels" --
confirmed again in this step (this was already flagged in
`docs/component-decisions.md` during Step 1). What was investigated:

| approach | verdict | why |
|---|---|---|
| Live-cursor libraries (`pynput`, `pyautogui`) | **REJECT** | read the live OS cursor; no such signal exists in a recorded file (already rejected in Step 1) |
| Two-frame differencing | **REJECT (superseded)** | vacated/arrived ambiguity, see above -- replaced by three-frame differencing within this step |
| Three-frame differencing + contour filtering | **USE** | resolves the ambiguity, mature OpenCV primitives, CPU-only, no training data needed |
| Single-frame detection (no motion) | **REJECT** | no cursor-shape signal that generalizes across every OS/app cursor icon without a trained model; `detect_pointer(frame, prev_frame=None)` honestly returns `not_detected` rather than force a heuristic with no real basis |
| Template matching against known cursor icons | **REJECT for now** | cursor icons vary by OS/theme/app/scale; a fixed template set doesn't generalize, and building a robust multi-template library is a meaningfully bigger effort than this step's classical-CV motion approach for the same core capability (position + confidence) |
| Optical flow (`cv2.calcOpticalFlowFarneback`/Lucas-Kanade) | **REJECT for now** | denser and costlier than needed; three-frame differencing already gives a clean, cheap single-object isolation for the cursor-motion case this step targets. Worth revisiting if a future step needs sub-pixel velocity, not just position. |
| A trained cursor-detection CNN/ML model | **REJECT** | classical methods were not demonstrably insufficient (see Real-video results below) -- installing a model was never justified per this step's "document why before adding ML" requirement |
| Click-indicator (ripple/ring) detection as a dedicated feature | **DEFERRED, not built** | the real test video's dedicated cursor-highlight ring (see below) was detected as a byproduct of ordinary motion detection since it moves with the cursor -- no dedicated ring-shape classifier was needed for this step's scope. A general ripple/expanding-circle detector is future work, not built now (see Limitations). |

## False-positive resistance

Tested against exactly the traps the task calls out as risky (synthetic,
ground-truth-controlled):

- **Large blinking UI element** (80x30px fixed-position block toggling every
  frame, ~3% of a 320x240 frame): 0 detections. Rejected by the area filter
  (bigger than the 1%-of-frame-area cursor ceiling).
- **Full-frame scene transition** (solid-color swap, 100% of frame changes):
  0 detections. Rejected by the scene-wide-motion veto (>15% of frame
  changed).
- **No motion at all** (static frame): 0 detections, as expected.

All three are automated tests in `tests/test_pointer.py`, not eyeballed.

## Synthetic ground-truth test

A 320x240, 10fps, 5s video with a 12x12 white square moving linearly on a
gray background: center at `(20 + 40t + 6, 20 + 20t + 6)`. Sampled every
0.5s and run through `detect_pointer_for_frames`:

- 8 of 10 samples: `status == "detected"`, position **exact to the pixel**
  (0 error) against the known trajectory.
- First and last sample: honestly `not_detected` (no adjacent frame on one
  side to form a triple) -- not a bug, the documented boundary case.

## Real-video test

Real screen-recording tutorial, found via `yt-dlp`'s own search (not a
guessed URL) for "screen recording tutorial mouse cursor demo" --
"How to Highlight Mouse Cursor While Recording the Screen?"
(`youtube.com/watch?v=upibaDHY36Y`, 270s, 1920x1080, 60fps). Chosen because
its actual subject is a cursor-highlighting effect in a real screen
recording -- exactly this project's target content, and incidentally a
harder case (a glowing ring effect around the cursor, not a bare arrow).

`track_pointer` over a 30-50s window (`scripts/pointer_smoke_test.py`,
0.5s interval, 41 frames analyzed): **6 detected, 20 uncertain, 15
not_detected — 14.6% detection rate, track.confidence 0.15**. Two
`"detected"` positions were manually verified by reading the actual cached
JPEG frames: the reported `(372, 256)` at t=34.0s and `(1592, 402)` at
t=36.0s visually line up with the cursor/highlight-ring position in the
source frame at those timestamps -- confirmed by eye against the extracted
images (Read tool on the cache files), not just asserted. The lower
detection rate than the synthetic test is real and expected: 1920x1080 real
footage has UI micro-motion (progress bars, text-cursor blink, panel
highlights, this specific video's own glow/highlight effects) that produces
multiple simultaneous motion candidates, which this detector correctly
reports as `"uncertain"` rather than guessing among them. The
speech→frame→pointer bridge was also exercised end to end on this video: the
real transcript segment `[28.30s-30.90s]` ("let's discuss what highlighting
effects are") produced 8 candidate frames via `frames_for_segment`, each run
through `detect_pointer_for_frames` -- integration works, no separate frame
extraction path was built for pointer detection.

## Trading/chart-style video

No real trading/chart screen recording was available in this environment.
The closest controlled proxy tested is the "full-frame scene transition"
false-positive trap above (stands in for a chart redrawing/scrolling) --
0 false positives. A dedicated multi-candle/scrolling-chart synthetic video
was not built for this step; flagged as follow-up validation before this
detector is trusted against real trading footage specifically (see
Limitations). No trading-specific detection logic was written or is
planned -- Video-Lens stays domain-agnostic per project scope.

## Performance

Measured on the real 1920x1080 tutorial video (`scripts/pointer_smoke_test.py`),
`track_pointer` over a 20s window, 0.5s interval, 41 frame extractions +
detections: **17.85s total (2.3 frames/s)**, most of which is ffmpeg frame
extraction (cold cache) rather than the OpenCV detection math itself.
Storage: 49 cached 1920x1080 `-q:v 2` mjpeg frames averaged **~219 KB/frame**
(10.7 MB total) -- consistent with `docs/frames.md`'s projected 30-80 KB/frame
*at 1080p* being on the low end; this specific source's visual complexity
pushes it higher, still nowhere near lossless-PNG size.

**RAM**: no persistent process holds a decoded video. Frame extraction
reuses Step 4's subprocess-per-frame `FrameExtractor` (~30MB/extraction,
already measured in `docs/frames.md`); detection itself loads each frame's
JPEG once via OpenCV (grayscale, a few MB for a 1080p frame) and releases it
after building its motion mask -- memory-safe by construction, independent
of video length.

## Dependencies

`opencv-python` -- already an installed transitive dependency of
`scenedetect` (confirmed via `pip show scenedetect`); now imported directly
by `adapters/pointer/cursor_detector.py` for `absdiff`/`threshold`/
`findContours`, so it's pinned explicitly in `requirements.txt` rather than
left implicit. No other new dependency. CPU-only; no GPU-only OpenCV build
or CUDA path used or required.

## Limitations

- **Stationary cursor is undetectable** by this method (see above) -- no
  motion, no signal. A future step could add a static, high-contrast
  cursor-icon heuristic as a fallback, but that reintroduces the
  template-matching generalization problem rejected above.
- **Multiple simultaneously moving objects** (a cursor plus an animated
  progress bar, for example) can push a real detection to `"uncertain"`
  rather than `"detected"`, even when the pointer position picked is
  actually correct -- confidence is conservative by design, per the "wrong
  is worse than uncertain" principle, at some detection-rate cost.
- **No dedicated click-indicator (ripple/ring) detector** -- a highlighted
  cursor ring is caught incidentally (it moves with the cursor and shows up
  in the motion mask), not via any shape-specific ring classifier.
- **Not validated against a real trading/chart screen recording** -- no
  such file was available in this environment; only a synthetic full-frame
  transition proxy was tested.
- **Detection rate on real 1920p footage (~19% measured) is meaningfully
  lower than on the clean synthetic video (~80%)** -- expected given real
  UI noise, documented rather than smoothed over.
- Frame cache reuse and all Step 4 limitations (no eviction, hue-blind
  dedup, best-effort `frame_index`) apply unchanged, since this step
  extracts frames through the same `FrameExtractor`.
