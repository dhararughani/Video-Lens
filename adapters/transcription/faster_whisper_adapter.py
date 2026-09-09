"""Transcription adapter: faster-whisper (CTranslate2 Whisper).

Chosen over ffmpeg's built-in whisper filter because that filter accumulates
a systematic ~48ms timestamp error per 30s chunk (~19s of drift on a 3-hour
video), which breaks the speech-to-frame alignment Video-Lens is built on.
Measured evidence in docs/component-decisions.md.

CPU-only by default (`device="cpu"`, int8). `device="cuda"` is available for
anyone who has a GPU, but nothing here requires one.
"""
from __future__ import annotations

from core.contracts import Transcript, TranscriptSegment, VideoInput, Word
from core.errors import TranscriptionError

_MS = 3  # canonical time precision: seconds, rounded to milliseconds


class FasterWhisperAdapter:
    def __init__(
        self,
        # "base" measured 62ms mean onset error vs "tiny.en"'s 810ms -- a 13x
        # alignment gain for ~1.5x the time, and multilingual so language
        # auto-detection works. See docs/speech.md.
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,  # None => auto-detect (multilingual models)
        vad_filter: bool = True,  # drops silence; without it Whisper invents text
        word_timestamps: bool = False,  # ~1.5x slower; opt in when you need them
        download_root: str | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.download_root = download_root
        self._model = None  # loaded lazily -- costly, and no-audio videos skip it

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise TranscriptionError(
                    "faster-whisper is not installed -- run: pip install -r requirements.txt"
                ) from e
            try:
                self._model = WhisperModel(
                    self.model_size, device=self.device,
                    compute_type=self.compute_type, download_root=self.download_root,
                )
            except Exception as e:
                raise TranscriptionError(
                    f"Could not load Whisper model '{self.model_size}' on device "
                    f"'{self.device}': {e}"
                ) from e
        return self._model

    def transcribe(self, video: VideoInput) -> Transcript:
        if not video.has_audio:
            return Transcript(
                segments=[], source="faster_whisper", status="no_audio",
                duration_sec=video.duration_sec,
            )

        model = self._load_model()
        try:
            raw_segments, info = model.transcribe(
                video.path,
                language=self.language,
                vad_filter=self.vad_filter,
                word_timestamps=self.word_timestamps,
            )
            segments = [self._to_segment(s) for s in raw_segments]  # generator: consumed here
        except Exception as e:
            raise TranscriptionError(f"Transcription failed for '{video.path}': {e}") from e

        # Whisper pads short audio to its 30s window and can report times past
        # the end of the media (seen: a 7.0s end on a 3.76s video). Times
        # outside the video would break frame/pointer lookup, so clamp them
        # to the media and drop anything starting beyond it.
        segments = [c for c in (self._clamp(s, video.duration_sec) for s in segments)
                    if c is not None]

        # Whisper can emit segments a few ms out of order; the contract requires
        # ordering, and sorting is honest (we're not inventing times).
        segments.sort(key=lambda s: s.start_sec)

        return Transcript(
            segments=segments,
            language=getattr(info, "language", None),
            source="faster_whisper",
            duration_sec=video.duration_sec,
            status="ok" if segments else "no_speech",
        )

    @staticmethod
    def _clamp(s: TranscriptSegment, duration_sec: float) -> TranscriptSegment | None:
        """Confine a segment to [0, duration]. None if it lies outside entirely."""
        start = max(0.0, min(s.start_sec, duration_sec))
        end = max(0.0, min(s.end_sec, duration_sec))
        if start >= duration_sec or end <= start:
            return None
        if start == s.start_sec and end == s.end_sec:
            return s
        words = None
        if s.words:
            words = tuple(w for w in s.words if w.start_sec < duration_sec)
            words = tuple(
                Word(text=w.text, start_sec=max(0.0, min(w.start_sec, duration_sec)),
                     end_sec=max(0.0, min(w.end_sec, duration_sec)))
                for w in words
            ) or None
        return TranscriptSegment(start_sec=round(start, _MS), end_sec=round(end, _MS),
                                 text=s.text, confidence=s.confidence, words=words)

    def _to_segment(self, s) -> TranscriptSegment:
        words = None
        if self.word_timestamps and getattr(s, "words", None):
            words = tuple(
                Word(text=w.word, start_sec=round(float(w.start), _MS),
                     end_sec=round(float(w.end), _MS))
                for w in s.words
            )
        return TranscriptSegment(
            start_sec=round(float(s.start), _MS),
            end_sec=round(float(s.end), _MS),
            text=s.text.strip(),
            confidence=None,  # avg_logprob is not a probability; don't mislabel it
            words=words,
        )
