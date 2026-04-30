# -*- coding: utf-8 -*-
"""
VAD-based voice listener using sounddevice + local Whisper.
Continuously records in background, detects speech via RMS,
auto-stops after silence, outputs yes/no decision.

Usage:
    listener = VoiceListener()   # loads model, starts background stream
    listener.start_listening(user_decision)  # pass in the [None] flag
    # ... user says yes/no, user_decision[0] gets set automatically
    listener.close()             # call when simulation ends
"""

import threading
import numpy as np
import sounddevice as sd
import whisper

# ── Tunable parameters ────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Hz, Whisper expects 16k
CHUNK_DURATION    = 1     # seconds per chunk
CHUNK_SAMPLES     = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 80     # RMS below this = silence
SILENCE_DURATION  = 0.5     # seconds of silence before cutting off
SILENCE_CHUNKS    = int(SILENCE_DURATION / CHUNK_DURATION)  # = 1-2 chunks
MAX_DURATION      = 5.0     # hard upper limit seconds
MAX_CHUNKS        = int(MAX_DURATION / CHUNK_DURATION)

YES_WORDS = ['yes', 'yeah', 'sure', 'okay', 'yep', 'accept']
NO_WORDS  = ['no',  'nope', 'reject', 'cancel']
# ─────────────────────────────────────────────────────────────────────────────


class VoiceListener:
    """
    Loads Whisper model once at init, keeps sounddevice stream open.
    Call start_listening(user_decision) when a decision is needed.
    The background thread sets user_decision[0] = 'accepted'/'rejected'.
    """

    def __init__(self, model_size='base'):
        print(f"[Voice] Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("[Voice] Whisper ready.")

        self._active      = [False]   # True when we're collecting audio
        self._user_dec    = [None]    # reference to current user_decision list
        self._lock        = threading.Lock()
        self._closed      = False

        # Start the background recording thread immediately
        self._thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._thread.start()

    def start_listening(self, user_decision_ref):
        """
        Tell the background loop to start collecting audio for a decision.
        user_decision_ref is the [None] list from simulate_deployment.
        """
        with self._lock:
            self._user_dec = user_decision_ref
            self._active[0] = True
        print("[Voice] 🎤 Listening for YES or NO...")

    def stop_listening(self):
        with self._lock:
            self._active[0] = False

    def close(self):
        self._closed = True

    def _recording_loop(self):
        """
        Continuously runs in background.
        When _active is True, collects chunks, detects speech,
        transcribes on silence, sets user_decision.
        """
        while not self._closed:
            with self._lock:
                active = self._active[0]

            if not active:
                # idle — sleep briefly and poll again
                threading.Event().wait(0.05)
                continue

            # ── Collect one utterance ─────────────────────────────────────
            buffer        = []
            silent_chunks = 0
            speech_started = False

            for _ in range(MAX_CHUNKS):
                with self._lock:
                    if not self._active[0]:
                        break   # decision already made by keyboard

                # Record one chunk via sounddevice
                chunk = sd.rec(CHUNK_SAMPLES, samplerate=SAMPLE_RATE,
                               channels=1, dtype='int16', blocking=True)
                chunk = chunk.flatten()

                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                print(f"[VAD] chunk rms={rms:.1f} speech_started={speech_started} silent_chunks={silent_chunks}")

                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    silent_chunks  = 0
                    buffer.append(chunk)
                elif speech_started:
                    silent_chunks += 1
                    buffer.append(chunk)  # include trailing silence for natural cut
                    if silent_chunks >= SILENCE_CHUNKS:
                        break  # enough silence after speech → stop
                # if speech hasn't started yet, just keep polling (don't buffer)

            if not buffer:
                continue  # nothing recorded, loop again

            # ── Transcribe ────────────────────────────────────────────────
            audio_np = np.concatenate(buffer).astype(np.float32) / 32768.0
            try:
                result = self.model.transcribe(audio_np, language='en', fp16=False)
                text   = result['text'].strip().lower()
                print(f"[Voice] You said: '{text}'")
            except Exception as e:
                print(f"[Voice] Transcription error: {e}")
                continue

            # ── Classify ──────────────────────────────────────────────────
            with self._lock:
                dec_ref = self._user_dec
                still_waiting = self._active[0] and dec_ref[0] == 'waiting'

            if still_waiting:
                if any(w in text for w in YES_WORDS):
                    with self._lock:
                        dec_ref[0]      = 'accepted'
                        self._active[0] = False
                    print("[Voice] Accepted.")
                elif any(w in text for w in NO_WORDS):
                    with self._lock:
                        dec_ref[0]      = 'rejected'
                        self._active[0] = False
                    print("[Voice] Rejected.")
                else:
                    print("[Voice] Please say YES or NO.")
                    # loop continues, collect another utterance