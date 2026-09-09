# Ingestion

## Supported input types

- **Local file path** — any file ffprobe can read.
- **URL** — anything yt-dlp supports (YouTube and hundreds of other sites).

Both normalize to the same `VideoInput` contract (`core/contracts.py`).
Downstream code (frame extraction, transcription) only ever sees
`VideoInput` — it doesn't know or care whether the video came from disk or
a URL.

## Entry point

```python
from adapters.ingestion import ingest

video = ingest("C:/clips/demo.mp4")          # local
video = ingest("https://youtube.com/watch?v=...")  # URL
```

`ingest()` dispatches on the source string (`urlparse(source).scheme in
("http", "https")` → URL adapter, else local adapter). Call
`LocalIngestionAdapter`/`URLIngestionAdapter` directly if you already know
which one you need.

## Local ingestion (`adapters/ingestion/local.py`)

Validates and normalizes a path — never copies or reads the video's frame
data into memory. `core/video_metadata.inspect_video()` runs `ffprobe`
(header-only read) and does the validation:

- path must exist, be a file, be non-empty
- ffprobe must be able to read it (bad/corrupt files fail here with the
  ffprobe stderr message attached)
- must have at least one video stream
- duration and resolution must be positive

All of these raise `core.errors.VideoIngestionError` with a specific
message — nothing is swallowed.

## URL ingestion (`adapters/ingestion/url.py`)

Wraps yt-dlp's own Python API (`yt_dlp.YoutubeDL`), not its CLI — no shell
string-building, no shell-injection surface. The rest of Video-Lens never
imports yt-dlp directly.

Flow:
1. `extract_info(url, download=False)` first — fails fast on an
   invalid/unsupported URL before downloading anything.
2. If a file for that video id already exists in the download cache, reuse
   it instead of re-downloading.
3. Otherwise download (`format: "bv*+ba/b"`, merged to mp4 — yt-dlp's
   default selector, not an aggressively narrow one) and locate the actual
   output path from yt-dlp's own `requested_downloads` metadata.
4. Run the same `inspect_video()` validation as local files.

Format selection is deliberately simple/default, not tuned — see
`docs/component-decisions.md`.

## Temporary files

Downloads land in `%TEMP%/video_lens_downloads/<video_id>.mp4` (or a custom
`download_dir` passed to `URLIngestionAdapter`/`ingest()`). This directory
is:

- **never** the location of a user's own local files — local ingestion
  never copies anything there
- **not** auto-deleted after each run, so repeat calls for the same URL
  reuse the download instead of re-fetching (see `docs/architecture.md`
  hardware/bandwidth notes) — this is a lightweight reuse cache, not an
  asset-management system, and cleaning it up is a manual/ops concern for
  now (`rm -rf` the directory when needed)
- filenames are keyed by yt-dlp's video id with `restrictfilenames: True`,
  so paths stay ASCII/Windows-safe and never round-trip through a
  filtergraph string (the `A:` drive-colon issue from Step 1 was specific to
  ffmpeg's `-af whisper=...` option string; ingestion never builds one)

## Metadata schema

`VideoInput` (see `core/contracts.py`):

| field | meaning |
|---|---|
| `path` | resolved local file ffmpeg/ffprobe can read (== `source` for local, the download for URLs) |
| `source_type` | `"local"` or `"url"` |
| `source` | the original path or URL passed in |
| `duration_sec`, `width`, `height`, `fps` | from ffprobe |
| `has_audio` | at least one audio stream present |
| `video_codec`, `audio_codec` | ffprobe codec names, `None` if not applicable |
| `format_name` | ffprobe container format string |

## Failure handling

Every failure path raises `VideoIngestionError` with a message naming the
actual problem (missing file, unreadable file, no video stream, invalid
duration/resolution, invalid/unsupported URL, download failure, missing
output after a reported-successful download). Nothing fails silently.

## What's deliberately not built here

No media-asset database, no automatic cache eviction, no aggressive
format/quality tuning, no playlist support (`noplaylist: True`) — all out of
scope for this lightweight foundation.
