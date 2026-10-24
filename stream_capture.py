"""Real-time audio stream capture for transcription pipelines.

Captures audio from a YouTube live stream, a local audio file, or the
microphone and yields fixed-size chunks as numpy float32 arrays normalized
to [-1, 1] at a target sample rate (16 kHz mono by default, matching
Whisper's expected input).

Typical usage::

    capture = StreamCapture("https://www.youtube.com/watch?v=...", chunk_seconds=10)
    capture.start()
    while capture.is_active:
        chunk = capture.get_chunk(timeout=5)
        if chunk is not None:
            feed_to_whisper(chunk)
    capture.stop()
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel pushed into the queue when the source ends (or capture stops).
_SENTINEL = object()


class StreamCaptureError(RuntimeError):
    """Raised when a capture source cannot be opened or read."""


class StreamCapture:
    """Capture fixed-size audio chunks from a YouTube URL, file, or microphone.

    Parameters
    ----------
    source:
        ``'http...'``/``'https...'`` — YouTube (or any yt-dlp-supported) URL,
        a local filesystem path to an audio/video file, or ``'mic'`` for the
        default microphone.
    sample_rate:
        Target sample rate of the produced chunks (default 16000 for Whisper).
    chunk_seconds:
        Duration of each chunk in seconds (default 10).
    """

    #: Kill ffmpeg this long after a polite terminate, then escalate.
    _FFMPEG_KILL_TIMEOUT = 5.0

    def __init__(
        self,
        source: str,
        sample_rate: int = 16000,
        chunk_seconds: float = 10.0,
    ) -> None:
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string (URL, file path, or 'mic')")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")

        self.source = source
        self.sample_rate = int(sample_rate)
        self.chunk_seconds = float(chunk_seconds)

        # Exact byte count for one chunk of s16le PCM: samples * 2 bytes.
        self._samples_per_chunk = int(round(self.sample_rate * self.chunk_seconds))
        self._bytes_per_chunk = self._samples_per_chunk * 2

        self._queue: "queue.Queue[object]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._mic_stream = None  # sounddevice.InputStream, when source == 'mic'
        self._mic_buffer = bytearray()
        self._mic_condition = threading.Condition()
        self._active = False
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def is_active(self) -> bool:
        """True while the capture is running (between start() and stop())."""
        return self._active

    def start(self) -> None:
        """Open the source and start the background reader thread."""
        if self._active:
            raise StreamCaptureError("StreamCapture is already running")

        self._stop_event.clear()
        self._queue = queue.Queue()
        self._mic_buffer = bytearray()

        if self.source == "mic":
            self._start_microphone()
        elif self.source.startswith(("http://", "https://")):
            self._start_ffmpeg(self._resolve_stream_url(self.source))
        else:
            self._start_ffmpeg_from_file(self.source)

        self._active = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="StreamCapture-reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("StreamCapture started (source=%s, %d Hz, chunk=%.2fs)",
                    self.source, self.sample_rate, self.chunk_seconds)

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the next audio chunk as float32 samples in [-1, 1].

        Shape is ``(samples,)`` where ``samples = sample_rate * chunk_seconds``.
        Returns ``None`` on timeout or when the stream has ended.
        """
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        if item is _SENTINEL:
            # Put it back so subsequent calls also see the end of stream.
            self._queue.put(_SENTINEL)
            return None

        raw = item  # bytes of s16le PCM
        samples = np.frombuffer(raw, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def stop(self) -> None:
        """Stop capture and release the underlying process / device."""
        if not self._active and self._proc is None and self._mic_stream is None:
            return

        self._stop_event.set()
        self._active = False

        self._stop_ffmpeg()
        self._stop_microphone()

        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        # Wake anyone blocked in get_chunk().
        self._queue.put(_SENTINEL)
        logger.info("StreamCapture stopped")

    def __enter__(self) -> "StreamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # YouTube / network sources
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_stream_url(url: str) -> str:
        """Use yt-dlp to resolve a YouTube URL to a direct audio stream URL."""
        try:
            import yt_dlp
        except ImportError as exc:
            raise StreamCaptureError(
                "yt-dlp is required for YouTube URLs: pip install yt-dlp"
            ) from exc

        ydl_opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:  # yt-dlp raises many exception types
            raise StreamCaptureError(f"Failed to extract stream URL from {url!r}: {exc}") from exc

        if info is None:
            raise StreamCaptureError(f"yt-dlp returned no metadata for {url!r}")

        # Flat playlist entries sometimes nest the real info dict.
        if info.get("_type") == "url" and "url" in info:
            url = info["url"]
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
            except Exception as exc:
                raise StreamCaptureError(f"Failed to resolve nested stream URL: {exc}") from exc

        stream_url = StreamCapture._extract_audio_url(info)
        if not stream_url:
            raise StreamCaptureError(
                f"Could not find an audio stream URL in yt-dlp metadata for {url!r}"
            )
        logger.debug("Resolved audio stream URL: %.120s...", stream_url)
        return stream_url

    @staticmethod
    def _extract_audio_url(info: dict) -> Optional[str]:
        """Pull the audio stream URL out of a yt-dlp info dict."""
        # Single-format response: the audio URL is simply `url`.
        if info.get("url") and info.get("acodec", "none") != "none":
            return info["url"]

        # Multi-format response (bestaudio may expand to video+audio).
        for fmt in info.get("requested_formats") or []:
            if fmt.get("acodec", "none") != "none" and fmt.get("url"):
                return fmt["url"]

        # Fallback: pick the last audio-only format from the full list.
        formats = info.get("formats") or []
        audio_formats = [
            f for f in formats
            if f.get("url") and f.get("acodec", "none") != "none"
        ]
        if audio_formats:
            # yt-dlp orders formats worst->best; take the last one.
            return audio_formats[-1]["url"]

        return info.get("url")

    # ------------------------------------------------------------------ #
    # ffmpeg backend (YouTube URLs and local files)
    # ------------------------------------------------------------------ #

    def _ffmpeg_base_cmd(self) -> list:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
        ]

    def _start_ffmpeg(self, stream_url: str) -> None:
        """Spawn ffmpeg reading a network stream, emitting raw s16le PCM."""
        cmd = self._ffmpeg_base_cmd() + [
            "-i", stream_url,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "pipe:1",
        ]
        self._spawn_ffmpeg(cmd)

    def _start_ffmpeg_from_file(self, path: str) -> None:
        """Spawn ffmpeg decoding a local file to raw s16le PCM."""
        import os

        if not os.path.isfile(path):
            raise StreamCaptureError(f"Local audio file not found: {path!r}")

        cmd = self._ffmpeg_base_cmd() + [
            "-i", path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "pipe:1",
        ]
        self._spawn_ffmpeg(cmd)

    def _spawn_ffmpeg(self, cmd: list) -> None:
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._bytes_per_chunk * 2,
            )
        except FileNotFoundError as exc:
            raise StreamCaptureError(
                "ffmpeg not found on PATH. Install ffmpeg and make sure it is "
                "available as `ffmpeg` in a terminal."
            ) from exc
        except OSError as exc:
            raise StreamCaptureError(f"Failed to launch ffmpeg: {exc}") from exc

    def _stop_ffmpeg(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return

        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self._FFMPEG_KILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=self._FFMPEG_KILL_TIMEOUT)

        # Drain stderr for diagnostics without blocking forever.
        if proc.stderr is not None:
            try:
                err = proc.stderr.read()  # process already reaped: EOF is immediate
                if err:
                    logger.debug("ffmpeg stderr: %s", err.decode(errors="replace").strip())
                proc.stderr.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # Microphone backend
    # ------------------------------------------------------------------ #

    def _start_microphone(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise StreamCaptureError(
                "Microphone capture requires the `sounddevice` package: "
                "pip install sounddevice"
            ) from exc

        def _callback(indata, frames, time_info, status) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
            with self._mic_condition:
                self._mic_buffer.extend(indata.tobytes())
                self._mic_condition.notify_all()

        try:
            self._mic_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
                callback=_callback,
            )
            self._mic_stream.start()
        except Exception as exc:
            self._mic_stream = None
            raise StreamCaptureError(f"Failed to open microphone: {exc}") from exc

    def _stop_microphone(self) -> None:
        stream, self._mic_stream = self._mic_stream, None
        if stream is None:
            return
        with self._mic_condition:
            self._mic_condition.notify_all()
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # device may already be gone
            logger.warning("Error closing microphone stream: %s", exc)

    # ------------------------------------------------------------------ #
    # Reader thread
    # ------------------------------------------------------------------ #

    def _reader_loop(self) -> None:
        try:
            if self.source == "mic":
                self._read_from_microphone()
            else:
                self._read_from_pipe()
        except Exception:
            logger.exception("StreamCapture reader thread crashed")
        finally:
            self._active = False
            self._queue.put(_SENTINEL)

    def _read_from_pipe(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return

        stdout = proc.stdout
        leftover = b""
        try:
            while not self._stop_event.is_set():
                # Read exactly one chunk worth of bytes (plus any remainder).
                needed = self._bytes_per_chunk - len(leftover)
                data = stdout.read(needed) if needed > 0 else b""
                if data is None or data == b"":
                    break  # EOF or stream closed
                leftover += data
                if len(leftover) < self._bytes_per_chunk:
                    continue

                chunk, leftover = leftover[: self._bytes_per_chunk], leftover[self._bytes_per_chunk:]
                self._queue.put(chunk)
        finally:
            # Leftover partial chunk (< one full block) is discarded: too short
            # for a fixed-size Whisper window and not worth padding silently.
            if leftover:
                logger.debug("Discarding %d trailing bytes at end of stream", len(leftover))

    def _read_from_microphone(self) -> None:
        while not self._stop_event.is_set():
            with self._mic_condition:
                while (
                    not self._stop_event.is_set()
                    and len(self._mic_buffer) < self._bytes_per_chunk
                ):
                    self._mic_condition.wait(timeout=0.2)

                if self._stop_event.is_set():
                    return

                if len(self._mic_buffer) < self._bytes_per_chunk:
                    continue

                chunk_bytes = bytes(self._mic_buffer[: self._bytes_per_chunk])
                del self._mic_buffer[: self._bytes_per_chunk]

            self._queue.put(chunk_bytes)


# ---------------------------------------------------------------------- #
# Demo
# ---------------------------------------------------------------------- #

def _demo() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        # Local test file is the safest default; pass a YouTube URL as argv[1].
        source = "test.wav"
        print(f"No source given — trying {source!r}. "
              f"Usage: python stream_capture.py <youtube-url|file-path>")

    capture = StreamCapture(source, sample_rate=16000, chunk_seconds=10)
    try:
        capture.start()
    except StreamCaptureError as exc:
        print(f"Failed to start capture: {exc}")
        raise SystemExit(1)

    try:
        for i in range(3):
            chunk = capture.get_chunk(timeout=15)
            if chunk is None:
                print(f"chunk {i + 1}: (no data — end of stream or timeout)")
                break
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            print(
                f"chunk {i + 1}: samples={chunk.shape[0]}, "
                f"duration={chunk.shape[0] / capture.sample_rate:.2f}s, "
                f"rms={rms:.5f}"
            )
    finally:
        capture.stop()

    print("done")


if __name__ == "__main__":
    _demo()
