"""Real-time speech transcription powered by faster-whisper.

Feeds 16 kHz mono float32 numpy chunks (e.g. from a live microphone or a
streamed rally feed) through a single cached WhisperModel and returns text
for near-real-time downstream analysis.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel

# Chunks shorter than this hold no usable speech; skip the model call.
MIN_CHUNK_SECONDS = 0.5

# Warmup sample: half a second of silence at 16 kHz.
_WARMUP_SAMPLES = 8000


class Transcriber:
    """Transcribe streaming audio chunks with a persistent Whisper model."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        # Loaded once here and reused for every chunk — model load is heavy.
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        """Normalize an incoming chunk to contiguous mono float32 in [-1, 1]."""
        samples = np.asarray(audio)
        if samples.ndim > 1:
            # Multi-channel input: average down to mono.
            samples = samples.mean(axis=-1)
        samples = samples.astype(np.float32, copy=False)
        return np.ascontiguousarray(samples)

    def transcribe_chunk(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str:
        """Transcribe one audio chunk and return the text (empty if none)."""
        samples = self._to_float32(audio)
        if samples.size == 0 or samples.size < int(sample_rate * MIN_CHUNK_SECONDS):
            return ""

        segments, _info = self.model.transcribe(
            samples,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    def transcribe_stream(
        self,
        audio_generator: Iterable[np.ndarray],
        sample_rate: int = 16000,
    ) -> Iterator[str]:
        """Yield a transcription for each non-empty chunk from an audio source."""
        for chunk in audio_generator:
            text = self.transcribe_chunk(chunk, sample_rate=sample_rate)
            if text:
                yield text

    def detect_language(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[str, float]:
        """Detect the language of a chunk as (language_code, probability)."""
        samples = self._to_float32(audio)
        if samples.size == 0:
            return ("", 0.0)

        # Whisper's language id works best on ~a few seconds of audio.
        segments, info = self.model.transcribe(
            samples,
            beam_size=5,
        )
        language = info.language
        probability = float(info.language_probability)

        if not language:
            for segment in segments:
                if getattr(segment, "language", None):
                    language = segment.language
                    probability = float(
                        getattr(segment, "language_probability", probability)
                    )
                    break

        return (language or "", probability)

    def warmup(self) -> None:
        """Run a silent chunk through the model so the first real chunk is fast."""
        silence = np.zeros(_WARMUP_SAMPLES, dtype=np.float32)
        self.model.transcribe(
            silence,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )


if __name__ == "__main__":
    # Demo: synthetic sine wave has no speech, so transcription is empty.
    demo_rate = 16000
    t = np.arange(demo_rate, dtype=np.float32) / demo_rate
    sine = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    print("Loading model 'base' ...")
    transcriber = Transcriber(model_size="base")
    transcriber.warmup()
    print("Warmup done.")

    result = transcriber.transcribe_chunk(sine, sample_rate=demo_rate)
    print(f"Sine-wave chunk transcription: {result!r} (expected empty)")

    lang = transcriber.detect_language(sine, sample_rate=demo_rate)
    print(f"Sine-wave language detection: {lang}")
