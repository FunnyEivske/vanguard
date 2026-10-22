"""Mechanical and Acoustic Sound Cue Dispatcher for V.A.N.G.U.A.R.D."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class SoundCueEvent:
    """Dispatched non-verbal audio cue."""

    cue_id: str
    source_persona: str
    room_id: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


class SoundCueDispatcher:
    """Manages playback of hardware sound cues (servo whirs, shutter clicks)."""

    AVAILABLE_CUES = {
        "servo_whir": "Mechanical hydraulic servo whir (D.U.M.-E status confirmation)",
        "descending_tone": "Descending mechanical warning tone (D.U.M.-E alert)",
        "camera_shutter": "Precision lens shutter click (U sensor frame capture)",
        "soft_chime": "Refined arrival chime (J.A.R.V.I.S. presence)",
    }

    def __init__(self) -> None:
        self.played_cues: list[SoundCueEvent] = []

    def trigger_cue(self, cue_id: str, source_persona: str, room_id: str = "living_room") -> SoundCueEvent:
        """Trigger an acoustic sound cue."""
        if cue_id not in self.AVAILABLE_CUES:
            raise ValueError(f"Unknown sound cue ID: {cue_id}")

        event = SoundCueEvent(cue_id=cue_id, source_persona=source_persona, room_id=room_id)
        self.played_cues.append(event)
        return event
