"""Unified 3-Tier Memory Manager for V.A.N.G.U.A.R.D.

Coordinates:
Tier 1: Private isolated SQLite tables per persona.
Tier 2: Directed peer whisper pipeline.
Tier 3: Shared household knowledge graph.
"""

from __future__ import annotations

from pathlib import Path

from vanguard.core.event_bus import EventBus
from vanguard.memory.private_store import PrivateMemoryPartition
from vanguard.memory.shared_graph import SharedHouseholdGraph
from vanguard.memory.whisper_pipeline import DirectedWhisperPipeline


class ThreeTierMemoryManager:
    """Orchestrator for Vanguard's 3-tier memory model."""

    def __init__(self, event_bus: EventBus, storage_dir: Path | None = None) -> None:
        self.event_bus = event_bus
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.private_partitions: dict[str, PrivateMemoryPartition] = {}
        self.whisper_pipeline = DirectedWhisperPipeline(event_bus=event_bus)
        shared_db: Path | str = (
            self.storage_dir / "shared_household_graph.db" if self.storage_dir else ":memory:"
        )
        self.shared_graph = SharedHouseholdGraph(db_path=shared_db)

    def get_or_create_private_partition(self, persona_id: str) -> PrivateMemoryPartition:
        """Get or initialize the private SQLite partition for a persona."""
        pid = persona_id.lower()
        if pid not in self.private_partitions:
            db_path: Path | str = (
                self.storage_dir / f"private_{pid}.db" if self.storage_dir else ":memory:"
            )
            self.private_partitions[pid] = PrivateMemoryPartition(persona_id=pid, db_path=db_path)
        return self.private_partitions[pid]

    def close(self) -> None:
        """Close all private and shared database connections."""
        for partition in self.private_partitions.values():
            partition.close()
        self.private_partitions.clear()
        self.shared_graph.close()
