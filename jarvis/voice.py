"""Optional, local microphone/STT/TTS adapter.

Imports for third-party audio packages are deliberately delayed so text mode stays
dependency-free. Audio never leaves the machine in this adapter.
"""
from __future__ import annotations

import platform
import subprocess


class LocalVoiceLoop:
    SAMPLE_RATE = 16_000
    FRAME_SAMPLES = 1_280  # openWakeWord's standard 80 ms frame

    def __init__(self, assistant):
        self.assistant = assistant
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel
            from openwakeword.model import Model
        except ImportError as error:
            raise RuntimeError("Voice dependencies are missing. Run: pip install -r requirements-voice.txt") from error
        self.np, self.sd = np, sd
        self.wake_model = Model()
        # Use a small local model; faster-whisper downloads it once then caches it locally.
        self.stt_model = WhisperModel("base.en", device="auto", compute_type="int8")

    def speak(self, message: str) -> None:
        print(f"Jarvis: {message}")
        if platform.system() == "Windows":
            # System.Speech is built into full Windows PowerShell; pass text as an argument,
            # not as executable PowerShell code.
            script = "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak($args[0])"
            subprocess.run(["powershell", "-NoProfile", "-Command", script, message], check=False)
        else:
            print("Install/configure Piper for audible speech on this platform.")

    def _transcribe(self, audio):
        segments, _ = self.stt_model.transcribe(audio, language="en", vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def run(self) -> None:
        self.speak("Jarvis is ready.")
        listening, captured = False, []
        silence_frames = 0
        with self.sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype="float32", blocksize=self.FRAME_SAMPLES) as stream:
            while True:
                frame, _overflowed = stream.read(self.FRAME_SAMPLES)
                mono = frame[:, 0]
                if not listening:
                    scores = self.wake_model.predict((mono * 32767).astype(self.np.int16))
                    if max(scores.values(), default=0.0) >= 0.5:
                        listening, captured, silence_frames = True, [], 0
                        self.speak("Yes?")
                    continue
                captured.append(mono.copy())
                # End an utterance after about 1.2 seconds of quiet, or at 20 seconds.
                silence_frames = silence_frames + 1 if float(self.np.max(self.np.abs(mono))) < 0.015 else 0
                if silence_frames >= 15 or len(captured) >= 250:
                    transcript = self._transcribe(self.np.concatenate(captured))
                    reply = self.assistant.handle_transcript("hey jarvis " + transcript)
                    if reply:
                        self.speak(reply)
                    listening, captured = False, []
