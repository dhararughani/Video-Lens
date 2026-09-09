# Speech

Turns a video's audio into timestamped, queryable speech that later steps can
line up against frames and pointer events.

```python
from adapters.ingestion import ingest
from adapters.transcription.faster_whisper_adapter import FasterWhisperAdapter

video = ingest("lesson.mp4")
transcript = FasterWhisperAdapter().transcribe(video)

transcript.segment_at(2234.0)             # what was said at 37:14
transcript.segments_between(1800, 2100)   # everything from 30:00 to 35:00
transcript.text_around(2234.0, 5.0)       # speech context for a frame at 37:14
```

## TIME MODEL

**The canonical representation is `float` seconds from the start of the
video, rounded to milliseconds.** Everything in Video-Lens uses it —
`TranscriptSegment.start_sec`, `Word.start_sec`, `Frame.timestamp_sec`,
`PointerEvent.timestamp_sec` — so speech, frames and pointer events are
joinable by simple numeric comparison, with no unit conversion at the join.

Rules:
- Engines that report milliseconds are converted **once, at the adapter
  boundary**, never in core.
- Rounding to 3 decimal places keeps values like `9.940000000000003` out of
  the contracts; sub-millisecond precision is meaningless here anyway (the
  measured accuracy floor is ~60ms).
- Frame numbers are never stored as timestamps. Convert explicitly via
  `VideoInput.fps` at the point of use.
- Segment times are guaranteed to lie within `[0, video.duration_sec]` — the
  adapter clamps them (Whisper pads short audio to a 30s window and will
  otherwise report times past the end of the media).

## Contract

`Transcript` (in `core/contracts.py`) holds `segments`, `language`,
`source`, `duration_sec` and `status`. Each `TranscriptSegment` has
`start_sec`, `end_sec`, `text`, optional `confidence`, and optional `words`
(a tuple of `Word`, each with its own start/end).

Word timings are populated **only** when the engine genuinely provides them
(`FasterWhisperAdapter(word_timestamps=True)`) — never interpolated from
segment bounds. They cost roughly 1.5x the runtime, so they are off by
default; turn them on when you need to pin a phrase like "right *here*" to a
precise moment.

The contract validates on construction: segments must be ordered by start
time and no segment may end before it starts. Slightly **overlapping**
segments are permitted, because Whisper genuinely emits them.

## Status vs. error

An empty transcript is never ambiguous:

| outcome | result |
|---|---|
| speech transcribed | `status="ok"` (guaranteed non-empty — `"ok"` with no segments is rejected by the contract) |
| video has no audio stream | `status="no_audio"`, no model is even loaded |
| audio exists but holds no speech | `status="no_speech"` |
| engine/model actually failed | raises `TranscriptionError` |

So a caller can distinguish "nothing was said" from "we failed to find out",
which a bare empty list cannot express.

## Transcription mechanism

**faster-whisper** (CTranslate2 Whisper), CPU by default (`device="cpu"`,
`compute_type="int8"`). `device="cuda"` is available but nothing requires a
GPU — all figures below are CPU-only, on a machine with no usable GPU.

It replaced ffmpeg's built-in `whisper` filter, which was used in Steps 1–2.
That filter accumulates a systematic ~48ms timestamp error per 30s chunk —
measured at corr(error, time) = **+0.97**, extrapolating to ~19s of drift on
a 3-hour video, which would make speech-to-frame alignment meaningless. The
full measurements are in `docs/component-decisions.md`.

## Model choice

Measured on a 187s continuous-speech recording containing 48 utterances at
independently known onsets:

| model | segments found | mean onset error | median | max | speed | 3h video |
|---|---|---|---|---|---|---|
| `tiny.en` | 49 (one spurious) | 856 ms | 936 ms | 3086 ms | 3.5x realtime | ~51 min |
| `base.en` | 48 ✓ | 52 ms | — | — | 2.6x realtime | ~69 min |
| **`base`** (default) | **48 ✓** | **62 ms** | 54 ms | 187 ms | 2.4x realtime | ~74 min |

`base` recovered the utterance count exactly; `tiny.en` invented an extra
segment and its timings were roughly an order of magnitude worse. That is a
~14x alignment improvement for ~1.5x the runtime, without hard-coding
English — so `base` is the default. Use `base.en` if you know the content is
English and want the last few ms; use `small`/`medium` if text accuracy
matters more than runtime.

(Speeds include model load, so short clips look slower — a 4s clip runs at
0.6x realtime because ~5s of that is loading the model. The adapter loads
lazily and caches, so the cost is paid once per adapter instance.)

## Language

`language=None` (the default) auto-detects and reports the result in
`Transcript.language` — verified detecting `en` correctly. Pass an explicit
code (`language="de"`) to skip detection. `.en` models force English and
cannot detect. No translation is performed; the transcript is always in the
language spoken.

## Long videos

Whisper streams from the file; nothing loads the whole video into RAM.
Measured on a 6-minute, 117MB video: **peak RAM 451MB**, flat and
independent of video length (it is a function of model size, not duration).
Projected memory for a 3-hour video is the same.

Crucially, **timestamp error does not grow with video length**:
corr(error, absolute time) = **−0.04** over a 187s recording, and drift
across the file was −0.04s. No chunking logic was added on our side — it was
benchmarked first and proved unnecessary, unlike the ffmpeg filter it
replaced.

## Silence and bad audio

`vad_filter=True` (default) runs voice-activity detection first. This
matters for evidence integrity: on 20s of pure silence, Whisper **without**
VAD invents the word "You"; with VAD it correctly returns zero segments.
Fabricating speech that was never spoken is worse than any segmentation
cost.

On realistic continuous speech the flag is measurably free — VAD on and off
produced identical output. Its only cost appears with long silences
(pathological test: 7.5s gaps), where it merges utterances across removed
gaps, yielding coarser segments. Set `vad_filter=False` if you have such
material and prefer granularity over hallucination safety.

Pure noise correctly produces zero segments either way.

## Native captions (not implemented — evaluated and deferred)

For URL sources yt-dlp exposes both manual and automatic captions. These
were fetched and compared against Whisper on the same video before deciding.
Manual captions won on text accuracy (the human caption read "long trunks"
where `tiny.en` produced "warm pumps") and carried millisecond timings.

They were still deferred, for reasons that are about correctness, not
effort:

- **They only exist for URL sources.** Local files — the primary Video-Lens
  case — can never use them, so Whisper is required regardless. Captions
  would be a second path, not a replacement.
- **Their timestamps mean something different.** Caption times are *display*
  timings, not speech timings: one measured caption held "really really long
  trunks" from 7.974s to 12.616s, ~3s past when the phrase was actually
  spoken. Mixing display timing into a temporal model built for frame
  alignment would corrupt exactly what this layer guarantees.
- **Auto-captions are themselves ASR**, so they offer no accuracy advantage
  over local Whisper while adding a VTT/JSON3 parser.
- They contain non-speech annotations (`(baaaaaaaaaaahhh!!)`) that would
  need filtering before being treated as evidence.

If revisited, the sane use is as a *text* quality improvement aligned onto
Whisper's timings — not as a timestamp source.

## Role in future synchronization

Speech, frames and pointer events all carry `float` seconds on the same
timeline, so joining them needs no adapter-specific logic:

```text
transcript.segment_at(frame.timestamp_sec)     # what was said over this frame
transcript.text_around(pointer.timestamp_sec)  # what was said while pointing
```

`text_around()` exists specifically to build the transcript context a future
vision/multimodal step will attach to a frame.
