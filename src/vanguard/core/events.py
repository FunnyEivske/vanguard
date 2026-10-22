"""Core event models and type definitions for V.A.N.G.U.A.R.D."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventCategory(str, Enum):
    """Event classification categories."""

    LORE = "lore"
    SPEECH = "speech"
    WHISPER = "whisper"
    SENSOR = "sensor"
    DIAGNOSTIC = "diagnostic"
    FLOOR = "floor"
    COMMAND = "command"
    AUDIT = "audit"
    ALERT = "alert"


@dataclass(frozen=True)
class Event:
    """Immutable substrate event."""

    category: EventCategory
    event_type: str
    source: str
    payload: dict[str, Any]
    target: str | None = None  # None indicates broadcast; set to recipient ID for directed whisper
    room_id: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def is_whisper(self) -> bool:
        """Check if event is a confidential point-to-point whisper."""
        return self.category == EventCategory.WHISPER or self.target is not None


@dataclass(frozen=True)
class SpeechRequest:
    """Request by an agent or human to speak."""

    speaker_id: str
    room_id: str
    text: str
    language: str = "nb-NO"
    target_listener: str | None = None
    is_inter_agent: bool = False
    priority: int = 0  # 10 for human, 5 for Jarvis LeadMode, 0 for routine
