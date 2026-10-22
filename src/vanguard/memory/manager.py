"""Unified 3-Tier Memory Manager for V.A.N.G.U.A.R.D.

Coordinates:
Tier 1: Private isolated SQLite tables per persona.
Tier 2: Directed peer whisper pipeline.
Tier 3: Shared household knowledge graph.
"""

from __future__ import annotations

from vanguard.core.event_bus import EventBus
from vanguard.memory.private_store import PrivateMemoryPartition
from vanguard.memory.shared_graph import SharedHouseholdGraph
from vanguard.memory.whisper_pipeline import DirectedWhisperPipeline


class ThreeTierMemoryManager:
    """Orchestrator for Vanguard's 3-tier memory model."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.private_partitions: dict[str, PrivateMemoryPartition] = {}
        self.whisper_pipeline = DirectedWhisperPipeline(event_bus=event_bus)
        self.shared_graph = SharedHouseholdGraph()

    def get_or_create_private_partition(self, persona_id: str) -> PrivateMemoryPartition:
        """Get or initialize the private SQLite partition for a persona."""
        pid = persona_id.lower()
        if pid not in self.private_partitions:
            self.private_partitions[pid] = PrivateMemoryPartition(persona_id=pid)
        return self.private_partitions[pid]

    def close(self) -> None:
        """Close all private database connections."""
        for partition in self.private_partitions.values():
            partition.close()
        self.private_partitions.clear()
