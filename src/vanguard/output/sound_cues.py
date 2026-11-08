"""Mechanical and Acoustic Sound Cue Dispatcher for V.A.N.G.U.A.R.D.

Synthesizes and dispatches hardware sound cues (servo whirs, descending tones,
camera shutter clicks, and soft chimes) and optionally plays them on system audio.
"""

from __future__ import annotations

import array
import math
import random
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SoundCueEvent:
    """Dispatched non-verbal audio cue."""

    cue_id: str
    source_persona: str
    room_id: str
    timestamp: float = 0.0
    played_on_hardware: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


def synthesize_cue_wav(cue_id: str, output_path: Path, sample_rate: int = 24000) -> Path:
    """Synthesize a real PCM WAV audio file for the specified hardware sound cue."""
    samples = array.array("h")

    if cue_id == "servo_whir":
        # 0.35s hydraulic servo sweep 180Hz -> 440Hz -> 310Hz
        n = int(0.35 * sample_rate)
        phase = 0.0
        for i in range(n):
            t = i / n
            freq = 180.0 + 260.0 * math.sin(math.pi * t) + 25.0 * math.sin(2 * math.pi * 18 * t)
            phase += 2.0 * math.pi * freq / sample_rate
            env = math.sin(math.pi * t)
            val = 0.45 * env * (0.7 * math.sin(phase) + 0.3 * math.sin(2 * phase))
            samples.append(int(max(-1.0, min(1.0, val)) * 32767))

    elif cue_id == "descending_tone":
        # 0.40s descending mechanical alert 520Hz -> 180Hz
        n = int(0.40 * sample_rate)
        phase = 0.0
        for i in range(n):
            t = i / n
            freq = 520.0 - 340.0 * t
            phase += 2.0 * math.pi * freq / sample_rate
            env = 1.0 - 0.5 * t
            val = 0.5 * env * math.sin(phase)
            samples.append(int(max(-1.0, min(1.0, val)) * 32767))

    elif cue_id == "camera_shutter":
        # 0.12s precision mechanical dual shutter click
        n = int(0.12 * sample_rate)
        rng = random.Random(42)
        for i in range(n):
            t = i / sample_rate
            click1 = math.exp(-t * 180.0) if t < 0.05 else 0.0
            click2 = math.exp(-(t - 0.055) * 200.0) if t >= 0.055 else 0.0
            noise = rng.uniform(-1.0, 1.0)
            tone = math.sin(2.0 * math.pi * 1800.0 * t)
            val = 0.65 * (click1 + click2) * (0.6 * noise + 0.4 * tone)
            samples.append(int(max(-1.0, min(1.0, val)) * 32767))

    else:  # soft_chime
        # 0.45s harmonic chime (528Hz + 792Hz)
        n = int(0.45 * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.exp(-t * 6.5)
            val = 0.45 * env * (0.6 * math.sin(2 * math.pi * 528.0 * t) + 0.4 * math.sin(2 * math.pi * 792.0 * t))
            samples.append(int(max(-1.0, min(1.0, val)) * 32767))

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    return output_path


def play_wav_linux(wav_path: Path) -> bool:
    """Play a WAV file on Linux (aplay, paplay, pw-play, ffplay) or macOS (afplay for local workstation testing)."""
    import sys

    commands: list[list[str]] = []
    if sys.platform == "darwin":
        commands.append(["/usr/bin/afplay", str(wav_path)])
    commands.extend(
        [
            ["aplay", "-q", str(wav_path)],
            ["paplay", str(wav_path)],
            ["pw-play", str(wav_path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav_path)],
        ]
    )
    for cmd in commands:
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, timeout=10.0)
            if res.returncode == 0:
                return True
        except Exception:
            continue
    return False


class SoundCueDispatcher:
    """Manages playback of hardware sound cues (servo whirs, shutter clicks) on Linux."""

    AVAILABLE_CUES = {
        "servo_whir": "Mechanical hydraulic servo whir (D.U.M.-E status confirmation)",
        "descending_tone": "Descending mechanical warning tone (D.U.M.-E alert)",
        "camera_shutter": "Precision lens shutter click (U sensor frame capture)",
        "soft_chime": "Refined arrival chime (J.A.R.V.I.S. presence)",
    }

    def __init__(self, enable_hardware_playback: bool = False) -> None:
        self.enable_hardware_playback = enable_hardware_playback
        self.played_cues: list[SoundCueEvent] = []

    def trigger_cue(self, cue_id: str, source_persona: str, room_id: str = "living_room") -> SoundCueEvent:
        """Trigger an acoustic sound cue and optionally play it through the Linux server audio sink."""
        if cue_id not in self.AVAILABLE_CUES:
            raise ValueError(f"Unknown sound cue ID: {cue_id}")

        played_hw = False
        if self.enable_hardware_playback:
            tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix=f"vanguard_cue_{cue_id}_"))
            try:
                synthesize_cue_wav(cue_id, tmp_wav)
                played_hw = play_wav_linux(tmp_wav)
            except Exception:
                played_hw = False
            finally:
                tmp_wav.unlink(missing_ok=True)

        event = SoundCueEvent(
            cue_id=cue_id,
            source_persona=source_persona,
            room_id=room_id,
            played_on_hardware=played_hw,
        )
        self.played_cues.append(event)
        return event
