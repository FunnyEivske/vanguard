"""Directed Peer Sharing (Selective Whispers) Pipeline for V.A.N.G.U.A.R.D.

Enables confidential point-to-point communication between personas over the internal bus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event, EventCategory
from vanguard.memory.private_store import MemoryAccessError


@dataclass
class PeerWhisper:
    """Confidential point-to-point message between two personas."""

    sender_id: str
    recipient_id: str
    subject: str
    body: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class DirectedWhisperPipeline:
    """Manages confidential peer mailboxes and whisper delivery."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        # recipient_id -> list of PeerWhisper
        self._mailboxes: dict[str, list[PeerWhisper]] = {}

    async def send_whisper(
        self,
        sender_id: str,
        recipient_id: str,
        subject: str,
        body: str,
        metadata: dict[str, Any] | None = None,
    ) -> PeerWhisper:
        """Send a confidential whisper from one persona to another."""
        whisper = PeerWhisper(
            sender_id=sender_id.lower(),
            recipient_id=recipient_id.lower(),
            subject=subject,
            body=body,
            metadata=metadata or {},
        )

        if whisper.recipient_id not in self._mailboxes:
            self._mailboxes[whisper.recipient_id] = []
        self._mailboxes[whisper.recipient_id].append(whisper)

        # Publish whisper event with explicit target so the bus only routes it to target
        await self.event_bus.publish(
            Event(
                category=EventCategory.WHISPER,
                event_type="peer_whisper_delivered",
                source=whisper.sender_id,
                target=whisper.recipient_id,
                payload={"subject": subject, "body": body, "sender": whisper.sender_id},
            )
        )

        return whisper

    def get_whispers(self, recipient_id: str, requester_id: str) -> list[PeerWhisper]:
        """Retrieve whispers for a recipient. Enforces requester identity check."""
        if requester_id.lower() != recipient_id.lower():
            raise MemoryAccessError(
                f"Unauthorized access: {requester_id} cannot read confidential whispers directed to {recipient_id}."
            )
        return list(self._mailboxes.get(recipient_id.lower(), []))
