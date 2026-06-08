# -*- coding: utf-8 -*-
"""
VAD-based voice listener using sounddevice + OpenAI Whisper API.
Continuously records in background, detects speech via RMS,
auto-stops after silence, fires on_utterance callback with raw text.

Usage:
    def on_utterance(text):
        # handle raw transcription text
        ...

    listener = VoiceListener(on_utterance=on_utterance)  # starts background stream
    # ... runs forever in background, calls on_utterance whenever speech detected
    listener.close()             # call when simulation ends
"""

import io
import time
import wave
import queue
import threading
import numpy as np
import sounddevice as sd
from openai import OpenAI

# ── Tunable parameters ────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Hz, Whisper expects 16k
CHUNK_DURATION    = 0.1     # seconds per chunk (smaller = more responsive)
CHUNK_SAMPLES     = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 500    # RMS below this = silence
SILENCE_DURATION  = 0.4     # seconds of silence before cutting off
SILENCE_CHUNKS    = int(SILENCE_DURATION / CHUNK_DURATION)
MAX_DURATION      = 4.0     # hard upper limit seconds
MAX_CHUNKS        = int(MAX_DURATION / CHUNK_DURATION)
# ─────────────────────────────────────────────────────────────────────────────


class VoiceListener:
    """
    Always-on VAD listener. Uses OpenAI Whisper-1 API for transcription.
    Keeps a single sd.InputStream open for the lifetime of the object to
    avoid repeated open/close errors (Windows MME). Fires on_utterance(text)
    callback whenever speech is detected and transcribed.
    """

    def __init__(self, on_utterance=None):
        self.client       = OpenAI()
        self.on_utterance = on_utterance

        self._active      = [True]
        self._user_dec    = [None]
        self._lock        = threading.Lock()
        self._closed      = False
        self._audio_queue = queue.Queue()

        # Open stream once and keep it alive
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._thread.start()
        # print("[Voice] Always-on listening started (OpenAI Whisper API).")

    def _audio_callback(self, indata, _frames, _time_info, status):
        if status:
            print(f"[VAD] stream status: {status}")
        self._audio_queue.put(indata[:, 0].copy())

    def start_listening(self, user_decision_ref):
        with self._lock:
            self._user_dec = user_decision_ref

    def stop_listening(self):
        pass

    def close(self):
        self._closed = True
        self._stream.stop()
        self._stream.close()

    def _recording_loop(self):
        """
        Reads chunks from the audio queue (filled by sd.InputStream callback),
        detects speech via RMS VAD, transcribes on silence.
        """
        while not self._closed:
            # ── Collect one utterance ─────────────────────────────────────
            buffer         = []
            silent_chunks  = 0
            speech_started = False

            for _ in range(MAX_CHUNKS):
                if self._closed:
                    return

                try:
                    chunk = self._audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    silent_chunks  = 0
                    buffer.append(chunk)
                elif speech_started:
                    silent_chunks += 1
                    buffer.append(chunk)
                    if silent_chunks >= SILENCE_CHUNKS:
                        break
                # if speech hasn't started, keep polling without buffering

            if not buffer:
                continue

            # ── Transcribe via OpenAI Whisper API ─────────────────────────
            audio_int16 = np.concatenate(buffer)
            audio_secs  = len(audio_int16) / SAMPLE_RATE
            print(f"[Voice] Transcribing {audio_secs:.2f}s of audio...")
            t0 = time.time()
            try:
                # Pack int16 samples into an in-memory WAV so the API can read it
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)          # int16 = 2 bytes
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())
                wav_buf.seek(0)
                wav_buf.name = 'audio.wav'      # API requires a filename hint

                transcript = self.client.audio.transcriptions.create(
                    model='whisper-1',
                    file=wav_buf,
                    language='en',
                    prompt=(
                        "obstacle 1, obstacle 2, obstacle 3, obstacle 4, "
                        "goal 1, goal 2, goal 3, "
                        "yes, no, accept, reject"
                    ),
                )
                text = transcript.text.strip().lower()
                print(f"[Voice] Transcription took {time.time()-t0:.2f}s → '{text}'")
                if not text:
                    continue
                print(f"[Voice] You said: '{text}'")
            except Exception as e:
                print(f"[Voice] Transcription error: {e}")
                continue

            if not text:
                continue

            # ── Fire callback ─────────────────────────────────────────────
            if self.on_utterance is not None:
                try:
                    self.on_utterance(text)
                except Exception as e:
                    print(f"[Voice] Callback error: {e}")