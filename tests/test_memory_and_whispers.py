"""Tests verifying 3-tier memory model: private partitions, whisper confidentiality, and shared graph."""

import pytest

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event
from vanguard.memory.manager import ThreeTierMemoryManager
from vanguard.memory.private_store import MemoryAccessError


def test_private_memory_partition_isolation(memory_manager: ThreeTierMemoryManager) -> None:
    """Ensure Sibling A cannot inspect or write to Sibling B's private SQLite partition."""
    jarvis_partition = memory_manager.get_or_create_private_partition("jarvis")
    friday_partition = memory_manager.get_or_create_private_partition("friday")

    # Jarvis writes his private reasoning
    jarvis_partition.store_thought(
        requester_id="jarvis",
        topic="security_architecture",
        content="Analyzing perimeter defense latency.",
    )

    # Jarvis can read his own thoughts
    thoughts = jarvis_partition.read_thoughts(requester_id="jarvis")
    assert len(thoughts) == 1
    assert "Analyzing perimeter defense latency" in thoughts[0]["content"]

    # Friday attempts to read Jarvis's private thoughts -> Prohibited
    with pytest.raises(MemoryAccessError, match="prohibited from reading"):
        jarvis_partition.read_thoughts(requester_id="friday")

    # Edith attempts to write into Friday's private thoughts -> Prohibited
    with pytest.raises(MemoryAccessError, match="prohibited from writing"):
        friday_partition.store_thought(
            requester_id="edith",
            topic="game_simulation",
            content="Injecting chaos into Friday partition.",
        )


@pytest.mark.asyncio
async def test_directed_whisper_confidentiality(memory_manager: ThreeTierMemoryManager, event_bus: EventBus) -> None:
    """Ensure directed whispers between Friday and Jarvis remain strictly secret from Edith."""
    whisper_pipeline = memory_manager.whisper_pipeline

    edith_received_events: list[Event] = []
    jarvis_received_events: list[Event] = []

    # Subscribe handlers with persona subscriber IDs
    event_bus.subscribe(category=None, handler=lambda e: edith_received_events.append(e), subscriber_id="edith")
    event_bus.subscribe(category=None, handler=lambda e: jarvis_received_events.append(e), subscriber_id="jarvis")

    # Friday sends confidential whisper directly to Jarvis
    whisper = await whisper_pipeline.send_whisper(
        sender_id="friday",
        recipient_id="jarvis",
        subject="Sensor Drift Warning",
        body="Workshop temperature sensor 2 is drifting +1.4C.",
    )
    assert whisper.recipient_id == "jarvis"

    # Bus delivery check: Jarvis received it, Edith received NOTHING
    assert len(jarvis_received_events) == 1
    assert jarvis_received_events[0].payload["subject"] == "Sensor Drift Warning"
    assert len(edith_received_events) == 0

    # Mailbox check: Jarvis can read it
    jarvis_mail = whisper_pipeline.get_whispers(recipient_id="jarvis", requester_id="jarvis")
    assert len(jarvis_mail) == 1
    assert jarvis_mail[0].subject == "Sensor Drift Warning"

    # Edith attempts to read Jarvis's mailbox -> Prohibited
    with pytest.raises(MemoryAccessError, match="cannot read confidential whispers"):
        whisper_pipeline.get_whispers(recipient_id="jarvis", requester_id="edith")


def test_shared_household_graph(memory_manager: ThreeTierMemoryManager) -> None:
    """Ensure shared graph provides unified facts to all personas while filtering visitor visibility."""
    graph = memory_manager.shared_graph

    # Assert household facts
    graph.assert_fact(
        subject="eivind",
        predicate="preferred_coffee",
        object_value="Black dark roast",
        added_by="friday",
        visibility="household",
    )
    graph.assert_fact(
        subject="guest_wifi",
        predicate="network_name",
        object_value="Vanguard-Visitor",
        added_by="jarvis",
        visibility="public",
    )

    # Any sibling can query all household facts
    household_facts = graph.query(requester_role="family_member")
    assert len(household_facts) == 2

    # Visitors only see public facts
    visitor_facts = graph.query(requester_role="visitor")
    assert len(visitor_facts) == 1
    assert visitor_facts[0].subject == "guest_wifi"
