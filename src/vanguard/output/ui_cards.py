"""UI Status and Telemetry Card Emitter for V.A.N.G.U.A.R.D."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UICard:
    """Screen card emitted alongside or in place of voice."""

    card_id: str
    source_persona: str
    card_type: str
    title: str
    content: dict[str, Any]
    room_id: str = "living_room"
    timestamp: float = field(default_factory=time.time)


class UICardEmitter:
    """Emits UI telemetry cards to household displays."""

    def __init__(self) -> None:
        self.cards: list[UICard] = []

    def emit_card(
        self,
        source_persona: str,
        card_type: str,
        title: str,
        content: dict[str, Any],
        room_id: str = "living_room",
    ) -> UICard:
        """Create and emit a UI status card."""
        card_id = f"card_{int(time.time() * 1000)}"
        card = UICard(
            card_id=card_id,
            source_persona=source_persona,
            card_type=card_type,
            title=title,
            content=content,
            room_id=room_id,
        )
        self.cards.append(card)
        return card

    def get_cards(self, source_persona: str | None = None) -> list[UICard]:
        """Retrieve emitted UI cards."""
        if source_persona:
            return [c for c in self.cards if c.source_persona.lower() == source_persona.lower()]
        return list(self.cards)
