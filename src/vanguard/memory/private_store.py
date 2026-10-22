"""Private Memory Partition for V.A.N.G.U.A.R.D. Personas.

Provides cryptographically separate or table-isolated SQLite storage per persona.
Prevents cross-sibling inspection of private internal reasoning and musings.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class MemoryAccessError(Exception):
    """Raised when a persona attempts unauthorized access to a sibling's private partition."""


class PrivateMemoryPartition:
    """Isolated storage partition for a single persona."""

    def __init__(self, persona_id: str, db_path: Path | str = ":memory:") -> None:
        self.persona_id = persona_id.lower()
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Create private thoughts and scratchpad schema."""
        with self.conn:
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS private_thoughts_{self.persona_id} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT
                )
                """
            )

    def store_thought(self, requester_id: str, topic: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        """Store a private internal thought or intermediate reasoning trace."""
        if requester_id.lower() != self.persona_id:
            raise MemoryAccessError(
                f"Security violation: {requester_id} is prohibited from writing to {self.persona_id}'s private memory."
            )

        with self.conn:
            cur = self.conn.execute(
                f"""
                INSERT INTO private_thoughts_{self.persona_id} (topic, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (topic, content, time.time(), json.dumps(metadata or {})),
            )
            return cur.lastrowid or 0

    def read_thoughts(self, requester_id: str, topic: str | None = None) -> list[dict[str, Any]]:
        """Read thoughts from this partition. Strictly verifies requester is the owner."""
        if requester_id.lower() != self.persona_id:
            raise MemoryAccessError(
                f"Security violation: {requester_id} is prohibited from reading {self.persona_id}'s private memory partition."
            )

        query = f"SELECT id, topic, content, timestamp, metadata_json FROM private_thoughts_{self.persona_id}"
        params: tuple = ()
        if topic:
            query += " WHERE topic = ?"
            params = (topic,)
        query += " ORDER BY timestamp DESC"

        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "topic": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                }
            )
        return results

    def close(self) -> None:
        """Close SQLite database connection."""
        self.conn.close()
