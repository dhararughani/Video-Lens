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

## Frame windows (`FrameExtractor.extract_window`)

`extract_window(video, start_sec, end_sec, interval_sec)` samples uniformly
across a range. Validates `end_sec > start_sec` and `interval_sec > 0`
(`ValueError` otherwise). Requested timestamps that clamp to the same
instant near the video's end collapse to one frame — no duplicates.

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

- **Key**: `sha1(abs_path : mtime_ns : size : timestamp_ms)`, filename
  `<hash>.jpg`. Path + mtime + size means a modified or replaced video file
  produces a new key automatically — no manual invalidation logic needed.
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
