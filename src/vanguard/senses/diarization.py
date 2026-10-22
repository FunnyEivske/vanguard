"""Streaming Diarization and Audio Stream Management for V.A.N.G.U.A.R.D."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiarizedTurn:
    """An acoustic speech turn identified by diarization."""

    speaker_id: str
    room_id: str
    start_time: float
    end_time: float
    text: str
    language: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingDiarizer:
    """Segments continuous audio into discrete speaker turns across household rooms."""

    def __init__(self, default_room: str = "living_room") -> None:
        self.default_room = default_room
        self._active_turns: dict[str, DiarizedTurn] = {}

    def process_utterance(
        self,
        speaker_id: str,
        text: str,
        room_id: str | None = None,
        language: str = "nb-NO",
        confidence: float = 0.95,
    ) -> DiarizedTurn:
        """Create a completed speaker turn from processed stream."""
        target_room = room_id or self.default_room
        now = time.time()
        turn = DiarizedTurn(
            speaker_id=speaker_id,
            room_id=target_room,
            start_time=now - 1.0,
            end_time=now,
            text=text,
            language=language,
            confidence=confidence,
        )
        self._active_turns[target_room] = turn
        return turn

    def get_latest_turn(self, room_id: str) -> DiarizedTurn | None:
        """Get the most recent turn in a room."""
        return self._active_turns.get(room_id)
