"""Shared Household Graph for V.A.N.G.U.A.R.D.

Maintains consolidated household knowledge, family preferences, routines, and verified facts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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

    def __init__(self) -> None:
        self._facts: list[KnowledgeFact] = []

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
        fact = KnowledgeFact(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            confidence=confidence,
            added_by=added_by,
            visibility=visibility,
        )
        self._facts.append(fact)
        return fact

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        requester_role: str = "family_member",
    ) -> list[KnowledgeFact]:
        """Query facts from the shared graph, filtering for visitor privacy."""
        results = []
        is_visitor = requester_role == "visitor"

        for fact in self._facts:
            if is_visitor and fact.visibility != "public":
                continue
            if subject and fact.subject.lower() != subject.lower():
                continue
            if predicate and fact.predicate.lower() != predicate.lower():
                continue
            results.append(fact)

        return results

    def clear(self) -> None:
        """Flush the graph (e.g. for testing)."""
        self._facts.clear()
