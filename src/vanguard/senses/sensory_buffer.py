"""Rolling sensory buffer for V.A.N.G.U.A.R.D.

Maintains a sliding temporal window of sensory audio frames and environmental telemetry.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensoryFrame:
    """Individual acoustic or environmental frame."""

    timestamp: float = field(default_factory=time.time)
    room_id: str = "living_room"
    audio_energy_db: float = 0.0
    spectral_centroid: float = 0.0
    detected_language: str = "nb-NO"
    raw_text: str | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)


class RollingSensoryBuffer:
    """Bounded ring buffer for continuous ambient perception."""

    def __init__(self, capacity_seconds: float = 30.0, frame_duration_seconds: float = 0.1) -> None:
        self.capacity_frames = max(1, int(capacity_seconds / frame_duration_seconds))
        self.frame_duration = frame_duration_seconds
        self._buffer: collections.deque[SensoryFrame] = collections.deque(maxlen=self.capacity_frames)

    def append(self, frame: SensoryFrame) -> None:
        """Add a frame to the rolling sensory stream."""
        self._buffer.append(frame)

    def get_recent_frames(self, seconds: float = 5.0) -> list[SensoryFrame]:
        """Retrieve frames from the last N seconds."""
        cutoff = time.time() - seconds
        return [f for f in self._buffer if f.timestamp >= cutoff]

    def get_room_frames(self, room_id: str, seconds: float = 5.0) -> list[SensoryFrame]:
        """Retrieve frames from a specific room over recent seconds."""
        cutoff = time.time() - seconds
        return [f for f in self._buffer if f.room_id == room_id and f.timestamp >= cutoff]

    def clear(self) -> None:
        """Flush the sensory buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
