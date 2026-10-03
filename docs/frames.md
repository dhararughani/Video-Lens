# Frames

Turns a video into a manageable set of relevant, timestamped frames instead
of every decoded frame — the bridge between a speech segment (from
`docs/speech.md`) and the pixels a future vision step will look at.

```python
from adapters.ingestion import ingest
from adapters.frames.frame_extractor import FrameExtractor
from adapters.frames.keyframe_selector import frames_for_segment, select_keyframes

video = ingest("lesson.mp4")
extractor = FrameExtractor()

frame = extractor.get_frame(video, 37.6)                       # one timestamp
window = extractor.extract_window(video, 36.5, 40.0, 0.5)      # uniform sampling
candidates = frames_for_segment(video, transcript.segments[3], extractor)  # speech -> frames
keyframes = select_keyframes(video, extractor, interval_sec=2.0)           # whole-video, deduped
```

## Frame contract

`Frame` (in `core/contracts.py`): `timestamp_sec`, `path`, `source_video`,
`width`, `height`, `format`, `frame_index`, `reason`.

- `source_video` / `width` / `height` — provenance: which video this frame
  came from and its real pixel dimensions, needed once frames from several
  videos or extraction passes get mixed together downstream.
- `frame_index` is `round(timestamp_sec * video.fps)` — a **best-effort
  estimate**, never authoritative. ffmpeg's `-ss` seek is not guaranteed
  frame-exact on every container/codec (see below), so this field exists for
  rough correlation with a future `PointerEvent`, not as a precise index.
- `reason` — `"direct_timestamp"` (single `get_frame` call), `"window_sample"`
  (uniform interval), or `"scene_change"` (landed on a detected scene
  boundary). Every field earns its place: this is what a caller actually
  needs to decide whether to trust or re-weight a candidate frame.

Frames are **not downscaled or aggressively compressed** — `-q:v 2` (mjpeg,
near-lossless) is the same setting Step 1's scene-based extractor already
used, kept because a future vision step needs to read small text/UI detail,
not just recognize a scene.

## Canonical timestamp

Same time model as speech (`docs/speech.md`): float seconds from video
start, rounded to milliseconds, converted once at the adapter boundary.
`Frame.timestamp_sec` and `TranscriptSegment.start_sec`/`end_sec` are
directly comparable with no unit conversion:

```text
SpeechSegment:  37.120 -> 39.840
Frame:          37.600                      # falls inside the segment
```

## Direct timestamp extraction (`FrameExtractor.get_frame`)

Uses ffmpeg **input seeking** (`-ss` before `-i`), not output seeking.
Measured on this machine (`scripts/frame_smoke_test.py` and ad-hoc probes,
not committed — see Performance below):

- **~4x faster**: 0.17s vs 0.74s to reach 55s into a 60s 640x480 video.
- **Equally accurate**: on a video deliberately encoded with a single GOP
  (no keyframes after frame 0, the worst case for "fast" seeking landing on
  the wrong keyframe), color-probing the extracted frame at each second gave
  identical error to output seeking. Both seek modes decode forward from the
  keyframe to the exact target internally in this ffmpeg build (9.0), so the
  speed/accuracy tradeoff textbooks describe did not actually appear here.

That said, **exact frame-level seeking is not a hard cross-platform
guarantee** — behavior can differ by container/codec/ffmpeg build. Hence
`frame_index` is documented as an estimate, not a promise.

**A video has no frame between its last one and its reported duration.**
Requesting a timestamp in that gap made ffmpeg fail outright ("produced no
frame") — reproduced with a 30fps 3.0s video whose true last frame sits at
2.96667s. Fixed by clamping to `duration_sec - 1/fps` (not `duration_sec`
itself), rounding **down** so float rounding can't push a hair past that
boundary. Timestamps outside the video's range are clamped this way rather
than rejected; only a negative or non-finite timestamp is a `ValueError`. A
missing/unreadable video or an ffmpeg failure raises `FrameExtractionError`.

That clamp uses the **container** duration, which covers every stream. When
the video stream ends first (audio outlasts it — by 0.35 and 1.6 frames on the
Step 7 benchmark videos, by seconds in some recordings) the clamped timestamp is
still past the last frame. So when ffmpeg exits cleanly but writes nothing, the
extractor reads the video stream's packet timestamps from the last keyframe
onward (`ffprobe -read_intervals`, headers only, ~0.1 s on a 4-minute video),
floors the last one to milliseconds, and serves that frame, stamped with that
timestamp. No constant is subtracted: the position is the frame's own pts,
floored because a seek a hair *before* a frame returns it and a hair *after*
returns nothing. The result is remembered per video, so later requests clamp
straight to it; a request that succeeds never probes. If the probe can't
explain the failure (no timestamps, or an end not before the request), the
original error is raised. Pinned in `tests/test_benchmark_findings.py`
(both mismatch directions, a 30 fps last frame between milliseconds, windows
crossing the end).

## Frame windows (`FrameExtractor.extract_window`)

`extract_window(video, start_sec, end_sec, interval_sec)` samples uniformly
across a range. Validates `end_sec > start_sec` and `interval_sec > 0`
(`ValueError` otherwise). Requested timestamps that clamp to the same
instant near the video's end collapse to one frame — no duplicates.

## Targeted inspection (`video_lens.inspect_visual_evidence`)

A first-class "look closely here" request, built on the same `FrameExtractor`
and cache rather than a second extraction path:

```python
from core.contracts import InspectionRequest, Region
from video_lens import inspect_visual_evidence

result = inspect_visual_evidence(video, InspectionRequest(
    timestamp_sec=42.0, window_before_sec=1.0, window_after_sec=1.0, fps=4.0,
    scale_width=640, region=Region(0.5, 0.0, 1.0, 0.5)))
result.frames   # tuple[Frame, ...], ordered by time, each with its own provenance
```

- **No window** (`window_before_sec == window_after_sec == 0`): the single
  frame at `timestamp_sec`. **Window**: frames from `timestamp - before`
  through `timestamp + after` at `fps`, clamped to `[0, duration]` — a window
  that collapses against the end of the video returns the one clamped frame.
  A window **requires** `fps` (`InspectionRequest` rejects it otherwise): there
  is no sensible implicit density, and guessing one would make a request's cost
  unpredictable.
- **`region`** is the existing normalized `Region`, converted to pixels against
  the source frame and rounded *outward* to even pixels so the crop never cuts
  off part of the requested area. Even alignment is required, not cosmetic:
  ffmpeg silently rounds a crop on a chroma-subsampled format (e.g. yuv420p)
  down to even dimensions (`crop=41:23` really yields 40x22), which would make
  the size reported on the `Frame` disagree with its pixels.
- **`scale_width`** sets the output width; height follows the (cropped) aspect
  ratio, rounded half-up, and is passed to ffmpeg explicitly so the reported
  size is exact.
- **One ffmpeg pass**: `-vf crop=...,scale=...`, crop first; the scale also
  does the colour conversion below, so every request has exactly one `-vf`.
  A request with neither option is the original command plus that conversion
  only, and on an ordinary (BT.601/untagged) source its output is byte-for-byte
  what it was before (pinned by a regression test).
- **Colour**: a JPEG is BT.601 full-range by definition, and ffmpeg converts
  only the range, not the matrix, when writing one, so a BT.709 source's colours
  were shifted (MAD 9.6/255, up to 48 on a channel, on the Step 7 tutorial).
  The chain ends in `scale=out_color_matrix=bt601:out_range=full` with
  `accurate_rnd+full_chroma_int+full_chroma_inp`; the input matrix is read from
  each frame's own colour metadata (BT.709 is converted, BT.601 untouched,
  untagged read as BT.601 exactly like ffmpeg's own RGB decode). The flags
  matter: at default rounding the conversion has a 1–2 level bias that made
  neutral BT.709 content *worse* than before. Against an exact float decode of
  the raw YUV, mean error on the benchmark videos went 5.5–8.0 → 1.0–1.2
  (saturated) and 0.5–0.8 → 0.4–0.6 (neutral).
- **`Frame.width`/`height` are the OUTPUT dimensions** for a cropped/scaled
  frame, not the source's.
- Default storage is the extractor's own temp-dir cache — disposable like every
  other Video-Lens cache. Pass an `extractor` to scope it.

Known limitations: crop coordinates use `VideoInput.width/height` (the stream's
coded size, as ingestion reports it), so a source carrying rotation metadata —
which ffmpeg auto-rotates on decode — would map a region against swapped axes;
ingestion does not handle rotation today, so neither does this. There is no
ceiling on `fps × window`; a very dense request is many ffmpeg invocations.

## Temporal visual change (`core.visual_change`)

Deterministic measurement of how much the picture changed between two sampled
frames, and where. **Not an interpretation.** Since knowledge schema `1.2` its
output is first-class evidence: `core.evidence.visual_change_to_evidence` maps
each event to `Evidence(kind="visual_change")` (see `docs/evidence.md`,
"Measured evidence"). The detector itself still knows nothing of evidence,
knowledge or synthesis. Opt in for a whole run with
`PipelineConfig(visual_change_enabled=True)`.

**In the pipeline it has its own bounded temporal path** (Step 8), separate from
keyframe selection, and its frames never go to vision:

```
video ─┬─ keyframes (scene + dedupe) ── vision / pointer / correlation windows
       └─ grid 0, s, 2s, … + last decodable frame ── detect_visual_changes ── AnalysisResult.visual_changes
```

- **Grid.** `s = max(visual_change_interval_sec, duration / (visual_change_max_samples − 2))`:
  2.0 s by default, widened so at most `visual_change_max_samples` (150) frames
  are sampled — a 4.5-minute video takes ~136, a one-hour video gets a ~24 s step.
  Pairs are consecutive: t0→t1, t1→t2, …, last grid point→last decodable frame.
- **End of video.** Every timestamp goes through `FrameExtractor`, whose clamp
  and end-of-stream rule (above) is the only one: the final sample is
  `get_frame(duration)`, i.e. the real last frame even when the container
  outlasts the video stream. A video shorter than one step yields one pair
  (first vs. last frame); a one-frame video yields none and the stage is
  recorded unavailable, never as a zero-magnitude change.
- **Cost.** At the defaults the grid coincides with keyframe selection's own
  interval samples, which are already in the frame cache, so a pipeline run
  adds about one extraction (the last frame). Called alone (cold cache), it is
  one ffmpeg seek per sample.
- **Failure.** Any error sampling or measuring degrades the whole stage to
  "requested, nothing measured" (`stages_unavailable`), with a warning.

Why it is not the keyframes: keyframes are chosen *because* they differ, at
irregular gaps (up to 170 s on a Step 7 benchmark video), so nearly every
keyframe pair "changed" and the change was not localized. See docs/benchmark.md.

```python
from core.visual_change import detect_visual_changes, compare_frames

frames = inspect_visual_evidence(video, InspectionRequest(
    timestamp_sec=60, window_before_sec=30, window_after_sec=30, fps=2)).frames
events = detect_visual_changes(frames)   # one VisualChangeEvent per consecutive pair
```

- **Magnitude (L1)**: the fraction of frame area whose luma changed by more than
  25/255 — the same per-pixel rule as the pointer detector, now shared from
  `core.visual_change.changed_pixel_mask` so the two can't drift. In `[0, 1]`.
- **Threshold**: detected when `magnitude >= threshold` (**`>=`**, inclusive).
  Default `0.08`, measured rather than assumed: on a real talking-head video
  sampled 0.5s apart, natural presenter motion has p95 = 0.064, so 0.08 sits
  just above continuous motion (on 60s of that video: 0 of 120 pairs flagged),
  while on a real screen recording it flags 46 of 120 genuine transitions. It
  therefore **misses small localized changes** — a tooltip is ~0.5% of a frame.
  For clean screen recordings, where measured codec noise is ~0 (55% of
  consecutive 1080p pairs score exactly 0), pass a lower threshold such as
  `0.005`. What counts as meaningful also depends on the sampling interval:
  the same content changes more between frames 2s apart than 0.5s apart.
- **Regions (L2)**: when detected, changed pixels are grouped (blobs within
  ~1% of the longer side merge — a line of text is one change) and each group
  becomes a normalized `Region` that tightly bounds the *actually changed*
  pixels. Groups under 0.05% of the frame are dropped as specks; if only specks
  exist, one region bounds them all (a diffuse change). At most 8 regions are
  reported, largest first, and `detail` says when more existed.
- **Localized vs global**: decided by *extent*, not amount — `global` when the
  regions cover >= 50% of the frame. A side panel changing 40% of the width is
  localized; a scroll touching scattered pixels everywhere is global.
- **Timestamps**: `timestamp_sec` is the **later** frame's time (the first
  moment the new state is observed); `compared_timestamps` records both. The
  change happened somewhere in `(earlier, later]` — no finer precision than
  the sampling interval is claimed.
- **Unavailable is never "no change"**: an unreadable frame, frames of
  different sizes (never resampled — that would manufacture differences), or a
  missing OpenCV give `status="unavailable"` with `magnitude=None` and a
  plain-language `detail`. Out-of-order frames raise rather than being
  re-sorted.
- **Cost**: linear in frames, each read once; ~70 ms/pair at 1080p and ~9 ms at
  720p on this machine (CPU only).

**Deliberately not detected**: what a change *means*. "Scroll", "zoom",
"redraw", "annotation", "page transition" are future classification work. Also
not detected: subtle brightness drift below the per-pixel rule, pure hue
changes at equal luma (grayscale comparison — the same trade-off as
de-duplication), and anything between two samples that reverts before the
next. A webcam overlay or presenter in frame is genuine visual change and is
reported as such; telling it apart from content change is interpretation.

## Speech -> frame bridge (`frames_for_segment`)

For one `TranscriptSegment`: candidate frames at its start, middle, end, a
uniform interval across it, and any scene change that falls inside it (a
transition during narration like "look at this" is often the single most
useful frame). Selection is entirely deterministic temporal logic — **no
LLM/VLM decides relevance**, per Step 4 scope.

Pass `scene_timestamps=` when calling this across many segments of the same
video, so scene detection runs once instead of once per segment.

## Scene-aware selection

Built on the existing PySceneDetect integration (`docs/component-decisions.md`),
not duplicated: `adapters/frames/ffmpeg_scenedetect.py` now exposes
`detect_scene_timestamps(video)` — boundaries only, no extraction — shared by
`SceneDetectFrameAdapter` (Step 1, unchanged) and the new frame-intelligence
layer.

## Keyframe selection (`select_keyframes`)

Whole-video candidates = scene boundaries ∪ uniform interval samples,
deduplicated. On a 36s, 4-slide synthetic tutorial video: 19 sampled frames
(2s interval + scene boundaries) reduced to **4 keyframes** — one per
visually distinct slide — versus ~363 frames a naive full decode would
produce at this video's 10fps.

## Deduplication

**Mean absolute difference of an 8x8 grayscale thumbnail** against the most
recently *kept* frame (not the previous sample — a slow fade should still
collapse into one kept frame, not oscillate). No new CV dependency: the
thumbnail comes from ffmpeg's own `scale=8:8,format=gray` piped as raw bytes.

An **average-hash** (threshold each pixel against the image's own mean, then
compare Hamming distance) was tried first and rejected: on a perfectly flat
frame every pixel equals the mean, so every solid-color frame hashes to the
same all-zero bit pattern regardless of color — measured identical hashes
for solid red and solid green frames. Flat/near-flat frames (slides, simple
UI) are common in exactly this project's target domain, so that blind spot
isn't a corner case. Diffing raw thumbnail bytes directly does not have it.

Threshold: MAD > 10 (0-255 scale) counts as a real change. Measured: a
re-decoded identical frame has MAD ~0-2 (JPEG re-encoding noise); a genuine
color change measures far higher.

**Known limitation — hue-blindness:** grayscale luma cannot distinguish two
colors of near-identical luma. ffmpeg's named "red" and "green" measured
luma ~76 vs ~75 — indistinguishable by this method. A real screen recording
changing hue while holding luma constant (rare, but not impossible — e.g. a
UI theme swap) would not be caught. Documented rather than "fixed" by adding
full RGB diffing, because doing so would roughly triple the thumbnail size
for a failure mode not observed in any target content (tutorials, IDEs,
browsers, charts) during this step's testing.

## Cache

`FrameExtractor(cache_dir=...)` (default: a `videolens_frame_cache` folder
under the OS temp dir). Every `get_frame`/`extract_window` call is cached —
no separate "enable caching" flag, because re-decoding the same instant
twice has no value.

- **Key**: `sha1(abs_path : mtime_ns : size : timestamp_ms :
  FRAME_EXTRACTION_VERSION : crop : scale)`, filename `<hash>.jpg`. Path +
  mtime + size means a modified or replaced video file produces a new key
  automatically — no manual invalidation logic needed. The last three parts
  are what was done to the pixels (see "Targeted inspection" below): a cached
  full-resolution frame is never returned for a request that asks for a
  different crop or width. `crop`/`scale` are the *resolved integer*
  rectangle and size, not the raw request, so two requests that produce the
  same image share one entry and two that don't can't collide.
  `FRAME_EXTRACTION_VERSION` (currently `"3"`, since the colour conversion) is bumped whenever the meaning
  of a key would change; entries written before it existed carried no version,
  so they can never match and are simply orphaned in the disposable cache.
- **Reuse**: if the cache file exists **and is non-empty**, it's returned
  as-is. A cached file left at 0 bytes by a killed-mid-write process is
  **not** trusted — it gets regenerated. (Found and fixed during this step:
  the first version only checked existence, so a corrupt cache entry would
  have been served forever.)
- **Collisions**: different videos never share a key (path is part of the
  hash) — verified with two same-timestamp requests against two different
  videos.
- **Cleanup**: none. `# ponytail: no eviction, cache grows unbounded — treat
  the dir as disposable (it defaults to a temp dir); add size/LRU cleanup if
  a long-running process makes this a real problem.`

## Storage

`-q:v 2` mjpeg frames on the 320x240 test video averaged **~672 bytes/frame**
(31 cached frames, 20.3 KB total). At 1080p that scales roughly with pixel
count (~30-80 KB/frame is typical for `-q:v 2` mjpeg), so keyframe selection
mattering for storage is a real effect at realistic resolutions and video
lengths, not just frame *count*.

## Performance

Measured on the 36.4s, 320x240, 10fps synthetic tutorial video
(`scripts/frame_smoke_test.py`):

| operation | count | time |
|---|---|---|
| speech->frame bridge (one ~9.5s segment, 1s interval) | 13 candidate frames | 2.32s |
| whole-video keyframe selection (2s interval + scenes) | 19 sampled -> 4 kept | 5.49s |
| naive full decode (`ffmpeg` extract-every-frame) | 361 frames | 0.38s |

At this toy resolution/duration, naive extraction is already cheap in wall
time — the case for targeted extraction here is **frame count and storage
at scale**, not raw speed: a 3-hour 1080p30 video naively decodes to
~324,000 frames, while `select_keyframes` scales with scene-change count and
`duration / interval_sec`, independent of fps.

**RAM**: no persistent process ever holds a decoded frame. Each extraction
is a fresh, short-lived ffmpeg subprocess (~30MB working set, measured for a
single `get_frame` call) that exits immediately after writing one JPEG to
disk. Video length therefore does not change peak memory — the pipeline is
memory-safe by construction (subprocess-per-frame), not because of a
specific measurement that happens to hold on this machine.

## Real-video test

`scripts/frame_smoke_test.py` end to end on the synthetic tutorial video (4
distinct-color "slides", 10s each, with TTS narration matching each slide,
same construction technique as Step 3's ground-truth speech video):

- Transcript recovered 3 real speech segments.
- `frames_for_segment` on the "blue section" segment `[6.74s - 16.19s]`
  returned 13 frames, all inside the segment window, correctly ordered, and
  including a `scene_change` frame at **exactly 10.00s** — the true
  red-to-blue transition.
- Manually verified pixel color: the frame at 6.74s decoded to pure red
  `(255,0,0)`, the frame at 10.00s and 15.74s decoded to pure blue
  `(0,0,255)` — matching what the video actually shows at those instants,
  not just what was expected.
- `select_keyframes` reduced 19 sampled candidates to 4 keyframes, one per
  slide.

## Role in future synchronization

`Frame.timestamp_sec` joins to `TranscriptSegment`/`Word` by simple
comparison today, and will join to a future `PointerEvent.timestamp_sec` the
same way — no adapter-specific glue code needed, per the time model shared
across `docs/speech.md` and this document.

## Limitations

- Dedup is grayscale-luma-based and hue-blind (see above).
- No vision model, cursor detection, or agent logic exists yet — selection
  is purely temporal/visual-difference, per Step 4 scope.
- The frame cache has no eviction; long-running processes should periodically
  clear it or pass a scoped `cache_dir`.
- `frame_index` is an estimate, not verified against every container/codec
  ffmpeg supports — only against the mp4/h264 files this step tested.
