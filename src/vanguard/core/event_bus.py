"""Asynchronous Event Bus for V.A.N.G.U.A.R.D.

Coordinates silent inter-agent bus traffic, point-to-point whispers, and system telemetry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from vanguard.core.events import Event, EventCategory

EventHandler = Callable[[Event], Coroutine[Any, Any, None] | None]


class EventBus:
    """Core asynchronous event bus with strict point-to-point whisper isolation."""

    def __init__(self, inter_agent_speech_aloud: bool = False) -> None:
        self.inter_agent_speech_aloud: bool = inter_agent_speech_aloud
        self._subscribers: dict[EventCategory, list[tuple[str | None, EventHandler]]] = {
            cat: [] for cat in EventCategory
        }
        self._global_subscribers: list[tuple[str | None, EventHandler]] = []
        self._history: list[Event] = []
        self._lock = asyncio.Lock()

    def set_inter_agent_speech_aloud(self, enabled: bool) -> None:
        """Dynamically toggle the vocalization mode for inter-agent dialogue."""
        self.inter_agent_speech_aloud = enabled

    def subscribe(
        self,
        category: EventCategory | None,
        handler: EventHandler,
        subscriber_id: str | None = None,
    ) -> None:
        """Subscribe to a category or all events.

        If subscriber_id is provided, directed whispers targeting this ID will be routed to it.
        """
        if category is None:
            self._global_subscribers.append((subscriber_id, handler))
        else:
            self._subscribers[category].append((subscriber_id, handler))

    async def publish(self, event: Event) -> None:
        """Publish an event across the bus, respecting whisper confidentiality."""
        async with self._lock:
            self._history.append(event)

        # Collect eligible handlers
        handlers_to_invoke: list[EventHandler] = []

        is_directed_whisper = event.is_whisper()

        # Check category-specific subscribers
        for sub_id, handler in self._subscribers.get(event.category, []):
            if is_directed_whisper:
                # Only deliver to the explicit target or audit logger
                if sub_id == event.target or sub_id == "audit_logger":
                    handlers_to_invoke.append(handler)
            else:
                handlers_to_invoke.append(handler)

        # Check global subscribers
        for sub_id, handler in self._global_subscribers:
            if is_directed_whisper:
                if sub_id == event.target or sub_id == "audit_logger":
                    handlers_to_invoke.append(handler)
            else:
                handlers_to_invoke.append(handler)

        for handler in handlers_to_invoke:
            res = handler(event)
            if asyncio.iscoroutine(res):
                await res

    def get_history(self, category: EventCategory | None = None) -> list[Event]:
        """Return bus history, optionally filtered by category."""
        if category is None:
            return list(self._history)
        return [e for e in self._history if e.category == category]

    def clear_history(self) -> None:
        """Clear memory event history."""
        self._history.clear()
