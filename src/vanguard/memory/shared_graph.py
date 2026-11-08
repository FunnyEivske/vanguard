"""Shared Household Graph for V.A.N.G.U.A.R.D.

Maintains consolidated household knowledge, family preferences, routines, and verified facts
in a SQLite-backed semantic graph accessible by all personas.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeFact:
    """Fact or relational assertion in the shared household graph."""

    subject: str
    predicate: str
    object_value: Any
    confidence: float = 1.0
    added_by: str = "system"
    timestamp: float = field(default_factory=time.time)
    visibility: str = "household"  # "household" or "public"


class SharedHouseholdGraph:
    """Consolidated semantic graph accessible by all personas."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS household_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    added_by TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    visibility TEXT NOT NULL
                )
                """
            )

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        object_value: Any,
        added_by: str,
        confidence: float = 1.0,
        visibility: str = "household",
    ) -> KnowledgeFact:
        """Add or update a fact in the household knowledge graph."""
        now = time.time()
        fact = KnowledgeFact(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            confidence=confidence,
            added_by=added_by,
            timestamp=now,
            visibility=visibility,
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO household_facts (subject, predicate, object_json, confidence, added_by, timestamp, visibility)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject,
                    predicate,
                    json.dumps(object_value),
                    confidence,
                    added_by,
                    now,
                    visibility,
                ),
            )
        return fact

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        requester_role: str = "family_member",
    ) -> list[KnowledgeFact]:
        """Query facts from the shared graph, filtering for visitor privacy."""
        sql = "SELECT subject, predicate, object_json, confidence, added_by, timestamp, visibility FROM household_facts WHERE 1=1"
        params: list[Any] = []
        if requester_role == "visitor":
            sql += " AND visibility = 'public'"
        if subject:
            sql += " AND LOWER(subject) = LOWER(?)"
            params.append(subject)
        if predicate:
            sql += " AND LOWER(predicate) = LOWER(?)"
            params.append(predicate)

        with self.conn:
            cur = self.conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[KnowledgeFact] = []
        for r in rows:
            results.append(
                KnowledgeFact(
                    subject=r[0],
                    predicate=r[1],
                    object_value=json.loads(r[2]),
                    confidence=r[3],
                    added_by=r[4],
                    timestamp=r[5],
                    visibility=r[6],
                )
            )
        return results

    def clear(self) -> None:
        """Flush the graph."""
        with self.conn:
            self.conn.execute("DELETE FROM household_facts")

    def close(self) -> None:
        """Close the SQLite database connection."""
        self.conn.close()
