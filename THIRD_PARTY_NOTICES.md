# Third-party notices

Video-Lens's own code is MIT-licensed (see `LICENSE`). It depends on the
following third-party software, installed separately via `pip`/your OS
package manager — none of their source is copied into this repository.

| Dependency | Used for | License | Distribution |
|---|---|---|---|
| [FFmpeg](https://ffmpeg.org) | frame extraction, format merging | LGPL/GPL (build-dependent) | external binary, invoked via subprocess, not linked or redistributed |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | scene-change detection | BSD-3-Clause | pip dependency |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | URL video ingestion | Unlicense | pip dependency |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | speech transcription | MIT | pip dependency |
| [OpenCV](https://opencv.org) (opencv-python) | pointer/cursor detection | Apache 2.0 | pip dependency |
| [NumPy](https://numpy.org) | array operations | BSD-3-Clause | pip dependency |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (optional) | Claude vision provider only, `requirements-vision.txt` | MIT | pip dependency, not installed by default |

Video-Lens invokes FFmpeg as an external process (never links against it),
so no GPL/LGPL obligations attach to Video-Lens's own MIT-licensed code.

This file lists dependency names and licenses for attribution purposes; it
is not a legal opinion. Check each project's own license file for the
authoritative terms.
