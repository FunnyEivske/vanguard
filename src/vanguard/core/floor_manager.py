"""Floor Manager for V.A.N.G.U.A.R.D.

Enforces human-led dialogue governance, human barge-in priority, room presence exclusivity,
and clean multi-room session migration.
"""

from __future__ import annotations

import asyncio

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event, EventCategory, SpeechRequest


class FloorManagerError(Exception):
    """Exception raised for floor policy violations."""


class FloorManager:
    """Manages audio channel ownership, floor arbitration, and room presence."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        # Map: room_id -> active speaker id (None if floor is free)
        self._room_active_speakers: dict[str, str | None] = {}
        # Map: persona_id -> current room_id (ensures persona cannot be in 2 rooms at once)
        self._persona_current_rooms: dict[str, str] = {}
        # Lead mode tracking
        self.lead_mode_active: bool = False
        self.lead_mode_holder: str | None = None
        self._lock = asyncio.Lock()

    def get_active_speaker(self, room_id: str) -> str | None:
        """Get the current active speaker in a specific room."""
        return self._room_active_speakers.get(room_id)

    def get_persona_room(self, persona_id: str) -> str | None:
        """Get the room where a persona is currently active."""
        return self._persona_current_rooms.get(persona_id)

    async def request_floor(self, request: SpeechRequest) -> bool:
        """Arbitrate a request to take the conversational floor in a room.

        Enforces:
        1. V.A.N.G.U.A.R.D. itself can never speak.
        2. No AI can override a human speaker.
        3. Human barge-in immediately interrupts active AI.
        4. Room presence exclusivity (an AI cannot be in two rooms at once).
        """
        async with self._lock:
            # Rule 1: Substrate cannot speak
            if request.speaker_id.lower() in ("vanguard", "v.a.n.g.u.a.r.d.", "system", "substrate"):
                await self.event_bus.publish(
                    Event(
                        category=EventCategory.ALERT,
                        event_type="floor_denied_substrate_mute",
                        source="floor_manager",
                        payload={"reason": "V.A.N.G.U.A.R.D. core is neutral substrate and cannot speak."},
                    )
                )
                return False

            room_id = request.room_id
            current_speaker = self._room_active_speakers.get(room_id)
            is_human = request.priority >= 10 or request.speaker_id.lower() in ("human", "eivind", "wife", "child_elder", "child_younger")

            # Human barge-in: interrupts any AI currently holding the floor
            if is_human:
                if current_speaker and current_speaker != request.speaker_id:
                    # Interrupt AI speaker
                    await self._handle_barge_in(room_id, evicted_speaker=current_speaker, by_speaker=request.speaker_id)
                self._room_active_speakers[room_id] = request.speaker_id
                await self.event_bus.publish(
                    Event(
                        category=EventCategory.FLOOR,
                        event_type="floor_granted_human",
                        source="floor_manager",
                        room_id=room_id,
                        payload={"speaker_id": request.speaker_id, "priority": request.priority},
                    )
                )
                return True

            # If current speaker is human, AI cannot take the floor or interrupt
            if current_speaker and current_speaker.lower() in ("human", "eivind", "wife", "child_elder", "child_younger"):
                return False

            # Room Presence Exclusivity: An AI cannot be active in two rooms simultaneously
            existing_room = self._persona_current_rooms.get(request.speaker_id)
            if existing_room and existing_room != room_id:
                # Migrate persona to the new room cleanly
                await self._migrate_persona_room(request.speaker_id, from_room=existing_room, to_room=room_id)

            # Check if another AI is currently speaking in this room
            if current_speaker and current_speaker != request.speaker_id:
                # If Jarvis is in Lead Mode and emergency priority
                if self.lead_mode_active and request.speaker_id == "jarvis" and request.priority >= 5:
                    await self._evict_ai_speaker(room_id, current_speaker, "jarvis_lead_mode_preemption")
                else:
                    return False

            self._room_active_speakers[room_id] = request.speaker_id
            self._persona_current_rooms[request.speaker_id] = room_id

            await self.event_bus.publish(
                Event(
                    category=EventCategory.FLOOR,
                    event_type="floor_granted",
                    source="floor_manager",
                    room_id=room_id,
                    payload={"speaker_id": request.speaker_id},
                )
            )
            return True

    async def release_floor(self, speaker_id: str, room_id: str) -> None:
        """Release the floor in a room when an utterance finishes."""
        async with self._lock:
            if self._room_active_speakers.get(room_id) == speaker_id:
                self._room_active_speakers[room_id] = None
                await self.event_bus.publish(
                    Event(
                        category=EventCategory.FLOOR,
                        event_type="floor_released",
                        source="floor_manager",
                        room_id=room_id,
                        payload={"speaker_id": speaker_id},
                    )
                )

    async def _handle_barge_in(self, room_id: str, evicted_speaker: str, by_speaker: str) -> None:
        """Handle human barge-in interruption."""
        self._room_active_speakers[room_id] = None
        await self.event_bus.publish(
            Event(
                category=EventCategory.FLOOR,
                event_type="barge_in_interruption",
                source="floor_manager",
                room_id=room_id,
                payload={"interrupted_speaker": evicted_speaker, "interrupted_by": by_speaker},
            )
        )

    async def _evict_ai_speaker(self, room_id: str, evicted_speaker: str, reason: str) -> None:
        """Evict an AI speaker in favor of higher priority."""
        self._room_active_speakers[room_id] = None
        await self.event_bus.publish(
            Event(
                category=EventCategory.FLOOR,
                event_type="ai_speaker_evicted",
                source="floor_manager",
                room_id=room_id,
                payload={"evicted_speaker": evicted_speaker, "reason": reason},
            )
        )

    async def _migrate_persona_room(self, persona_id: str, from_room: str, to_room: str) -> None:
        """Cleanly migrate a persona from one room to another without duplicate voices."""
        if self._room_active_speakers.get(from_room) == persona_id:
            self._room_active_speakers[from_room] = None

        self._persona_current_rooms[persona_id] = to_room
        await self.event_bus.publish(
            Event(
                category=EventCategory.FLOOR,
                event_type="persona_room_migrated",
                source="floor_manager",
                payload={"persona_id": persona_id, "from_room": from_room, "to_room": to_room},
            )
        )

    def enable_lead_mode(self, holder: str = "jarvis") -> bool:
        """Enable temporary crisis Lead Mode."""
        self.lead_mode_active = True
        self.lead_mode_holder = holder
        return True

    def override_lead_mode(self, overridden_by: str) -> bool:
        """Allow a human operator (e.g. Eivind or Wife) to override Lead Mode."""
        if overridden_by.lower() in ("eivind", "wife", "human"):
            self.lead_mode_active = False
            self.lead_mode_holder = None
            return True
        return False
