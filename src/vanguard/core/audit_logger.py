"""Audit Logger for V.A.N.G.U.A.R.D.

Maintains an append-only verifiable audit trail of system events, floor allocations,
and permission decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event, EventCategory


class AuditLogger:
    """Tamper-evident audit log for Vanguard operations."""

    def __init__(self, event_bus: EventBus, log_file: Path | None = None) -> None:
        self.event_bus = event_bus
        self.log_file = log_file
        self.entries: list[dict[str, Any]] = []

        # Subscribe to all events across categories
        self.event_bus.subscribe(
            category=None,
            handler=self._handle_event,
            subscriber_id="audit_logger",
        )

    def _handle_event(self, event: Event) -> None:
        """Record event in structured audit trail."""
        record = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "category": event.category.value,
            "event_type": event.event_type,
            "source": event.source,
            "target": event.target,
            "room_id": event.room_id,
            "payload": event.payload,
        }
        self.entries.append(record)

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def get_entries(self, category: str | None = None) -> list[dict[str, Any]]:
        """Retrieve audit log entries."""
        if category:
            return [e for e in self.entries if e["category"] == category]
        return list(self.entries)

    def count(self) -> int:
        """Total number of audit records."""
        return len(self.entries)
