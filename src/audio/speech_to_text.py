"""Microphone speech-to-text built on top of VOSK."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional runtime dependency
    sd = None

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
except Exception:  # pragma: no cover - optional runtime dependency
    KaldiRecognizer = None
    Model = None

    def SetLogLevel(_: int) -> None:
        return


TextCallback = Callable[[str], None]
ErrorCallback = Callable[[str], None]


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def extract_audio_track(video_path: str, output_wav_path: str, sample_rate: int = 16000) -> None:
    """Extract mono PCM WAV audio from a video file using ffmpeg."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

    if not ffmpeg_exe:
        raise RuntimeError(
            "ffmpeg is required for video audio extraction. "
            "Install system ffmpeg (PATH) or run: pip install imageio-ffmpeg"
        )

    command = [
        str(ffmpeg_exe),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_wav_path),
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"Audio extraction failed. Ensure the video has an audio track. ffmpeg: {stderr}")


class VoskAudioFileRecognizer:
    """Offline VOSK recognizer for WAV files with timestamped segments."""

    def __init__(self, model_path: str, sample_rate: int = 16000) -> None:
        self.model_path = str(model_path)
        self.sample_rate = int(sample_rate)

    def transcribe_wav(self, wav_path: str) -> list[TranscriptSegment]:
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("VOSK is not installed. Run: pip install vosk")

        if not self.model_path or not Path(self.model_path).exists():
            raise RuntimeError("VOSK model path is invalid.")

        path = Path(wav_path)
        if not path.exists():
            raise RuntimeError(f"WAV file does not exist: {wav_path}")

        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise RuntimeError("WAV must be mono channel for VOSK processing.")
            if wf.getsampwidth() != 2:
                raise RuntimeError("WAV must be 16-bit PCM for VOSK processing.")

            model = Model(self.model_path)
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognizer.SetWords(True)

            segments: list[TranscriptSegment] = []
            while True:
                chunk = wf.readframes(4000)
                if len(chunk) == 0:
                    break
                if recognizer.AcceptWaveform(chunk):
                    self._append_segment(segments, recognizer.Result())

            self._append_segment(segments, recognizer.FinalResult())

        merged: list[TranscriptSegment] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if merged and (seg.start - merged[-1].end) <= 0.2:
                merged[-1] = TranscriptSegment(
                    start=merged[-1].start,
                    end=max(merged[-1].end, seg.end),
                    text=f"{merged[-1].text} {text}".strip(),
                )
            else:
                merged.append(seg)
        return merged

    def _append_segment(self, segments: list[TranscriptSegment], raw_json: str) -> None:
        try:
            payload = json.loads(raw_json or "{}")
        except Exception:
            return

        text = str(payload.get("text", "")).strip()
        words = payload.get("result") or []
        if not text:
            return

        if words:
            start = float(words[0].get("start", 0.0))
            end = float(words[-1].get("end", start + 1.0))
        else:
            start = 0.0
            end = start + 1.0

        segments.append(TranscriptSegment(start=start, end=max(end, start + 0.1), text=text))


class VoskMicrophoneRecognizer:
    """Background microphone recognizer that emits partial and final text."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        blocksize: int = 8000,
    ) -> None:
        self.model_path = str(model_path)
        self.sample_rate = int(sample_rate)
        self.device = device
        self.blocksize = int(blocksize)

        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._stream: Optional[Any] = None

        self._on_partial: Optional[TextCallback] = None
        self._on_final: Optional[TextCallback] = None
        self._on_error: Optional[ErrorCallback] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(
        self,
        on_partial: Optional[TextCallback] = None,
        on_final: Optional[TextCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> None:
        if self.is_running:
            return

        if sd is None or Model is None or KaldiRecognizer is None:
            raise RuntimeError(
                "VOSK dependencies are missing. "
                "Install 'vosk' and 'sounddevice'."
            )

        if not self.model_path or not Path(self.model_path).exists():
            raise RuntimeError(
                "VOSK model path is invalid. "
                "Download a model and set its folder path."
            )

        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        self._stop_event.clear()
        self._audio_q = queue.Queue(maxsize=64)

        SetLogLevel(-1)

        def _callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                self._emit_error(str(status))
            payload = bytes(indata)
            try:
                self._audio_q.put_nowait(payload)
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_q.put_nowait(payload)
                except queue.Full:
                    pass

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=1,
            device=self.device,
            callback=_callback,
        )
        self._stream.start()

        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._worker is not None:
            self._worker.join(timeout=2)
            self._worker = None

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def _run_worker(self) -> None:
        assert Model is not None and KaldiRecognizer is not None
        try:
            model = Model(self.model_path)
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognizer.SetWords(False)
        except Exception as exc:
            self._emit_error(f"Failed to initialize VOSK model: {exc}")
            return

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if recognizer.AcceptWaveform(chunk):
                    text = self._extract_text(recognizer.Result(), "text")
                    if text:
                        self._emit_final(text)
                else:
                    partial = self._extract_text(
                        recognizer.PartialResult(), "partial"
                    )
                    if partial:
                        self._emit_partial(partial)
            except Exception as exc:
                self._emit_error(f"VOSK recognition error: {exc}")
                break

    def _extract_text(self, raw_json: str, key: str) -> str:
        try:
            payload = json.loads(raw_json)
        except Exception:
            return ""
        return str(payload.get(key, "")).strip()

    def _emit_partial(self, text: str) -> None:
        if self._on_partial is not None:
            self._on_partial(text)

    def _emit_final(self, text: str) -> None:
        if self._on_final is not None:
            self._on_final(text)

    def _emit_error(self, text: str) -> None:
        if self._on_error is not None:
            self._on_error(text)
