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

## Cursor intelligence: movement over time (`core.cursor_intelligence`)

A temporal analysis layer over a `PointerTrack` that already exists — it
detects nothing itself and reads no pixels. It answers only: was pointer
movement observed, when, in which direction, at roughly what speed, and how
sure is that.

```python
from video_lens import analyze_cursor_motion        # track_pointer + analyze_track
track, segments = analyze_cursor_motion(video, 30.0, 50.0, interval_sec=0.5)
# or, given any PointerTrack:
from core.cursor_intelligence import analyze_track
segments = analyze_track(track)
```

Opt-in: a default run does not call it, so default pipeline behavior and cost
are unchanged. Since knowledge schema `1.2` its segments are first-class
evidence (`core.evidence.cursor_segment_to_evidence` ->
`Evidence(kind="cursor_track")`, see `docs/evidence.md`, "Measured evidence").
`PipelineConfig(cursor_intelligence_enabled=True)` segments the pointer events
the pipeline already detected -- keyframes are seconds apart, so expect mostly
`"uncertain"` there; call `analyze_cursor_motion` for dense tracking.

**There is no `"stationary"`.** The detector finds a cursor by its *motion*
(see "a paused cursor is invisible" above), so "no movement observed" is
absence of evidence, never evidence of stillness. `CursorSegment.motion_state`
is only `"moving"` or `"uncertain"`, and the contract rejects anything else.
Two confident detections at the same spot are `uncertain` too
(`displacement_within_jitter`): the cursor may have paused, or moved away and
back between samples.

- **Segmentation**: each pair of consecutive samples is one interval. It is
  `moving` only when both ends are `detected` (not `uncertain` — the detector
  picked among several candidates there), at most `max_interval_sec` (default
  2.0s) apart, in frames of the same size, and displaced by at least 0.03 of
  the frame diagonal (detector jitter). Every other interval is `uncertain`,
  and its `basis` says why: `pointer_not_observed`, `ambiguous_position`,
  `sampling_gap`, `displacement_within_jitter`, `frame_size_changed`,
  `zero_elapsed_time`, `not_sampled` (outside the sampled range),
  `too_few_samples`, `no_pointer_samples`. Adjacent moving intervals merge, as
  do adjacent uncertain ones with the same reason. Segments **tile the track's
  whole range** with no gaps, so every moment is accounted for.
- **Geometry** uses pixel coordinates normalized by the **frame diagonal**, not
  x and y separately: per-axis normalization would distort angles on any
  non-square frame (on 16:9, a visually 45° move would read ~29°). Distance in
  diagonals is isotropic and resolution-independent — the same convention the
  detector uses for its step limit.
- **Direction** (`direction_deg`): image convention — 0° right, 90° down, 180°
  left, 270° up, in [0, 360), from the first to the last position of the
  segment, rounded to 0.1°. Reported only when the path is nearly straight (net
  displacement ≥ 0.8 × path length); a path that turns is `moving` with
  `direction_deg=None` (`basis` ends `_heading_varies`).
- **Speed** (`mean_speed_norm`): observed path length ÷ elapsed time, in **frame
  diagonals per second** — not pixels, not physical units. The path is straight
  lines between samples, so it is a *lower bound* if the cursor curved between
  them. The real elapsed time is used, so uneven sampling is handled exactly.
- **Confidence** (moving only; uncertain is always 0.0): the mean detector
  confidence of the segment's positions × observation support, where support
  is `min(1, displaced_intervals / 2)` — one displaced interval alone counts
  half. A description of evidence quality, not a probability.

**Not detected, deliberately**: clicks, ripples, drags, hovering, selecting, or
what the pointer is "pointing at". Those are interpretation for a later layer.

Real-video result (the screen recording above, 30–50s at 0.5s): of 41 samples,
6 `detected`, 20 `uncertain`, 15 `not_detected`. Two short intervals were
`moving`; the rest is `uncertain` with reasons. One was checked by eye: 34.0s →
34.5s, (372,256) → (252,244), reported as 185.7° (left, slightly up) — the
cursor really did travel from the panel divider to a colour swatch. This layer
adds structure to the existing signal; it does **not** raise the detection
rate.

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
