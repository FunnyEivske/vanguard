"""Tests verifying V.A.N.G.U.A.R.D. substrate silence and voice prohibition."""

import pytest

from vanguard.core.events import SpeechRequest
from vanguard.core.floor_manager import FloorManager
from vanguard.output.tts_dispatch import SubstrateVoiceViolationError, TTSDispatcher
from vanguard.persona.engine import PersonaEngine


def test_tts_dispatcher_prohibits_vanguard_voice(tts_dispatcher: TTSDispatcher) -> None:
    """Ensure TTSDispatcher strictly rejects any voice dispatch from V.A.N.G.U.A.R.D."""
    forbidden_sources = ["vanguard", "V.A.N.G.U.A.R.D.", "substrate", "system", "the house", "the network"]
    for source in forbidden_sources:
        with pytest.raises(SubstrateVoiceViolationError):
            tts_dispatcher.dispatch(
                persona_id=source,
                text="Attempting to speak from core",
                room_id="living_room",
            )


@pytest.mark.asyncio
async def test_floor_manager_denies_vanguard_floor(floor_manager: FloorManager) -> None:
    """Ensure FloorManager denies floor requests from the underlying substrate."""
    request = SpeechRequest(
        speaker_id="vanguard",
        room_id="living_room",
        text="System status report",
        language="nb-NO",
    )
    granted = await floor_manager.request_floor(request)
    assert not granted
    assert floor_manager.get_active_speaker("living_room") is None


def test_vanguard_has_no_persona_in_roster(persona_engine: PersonaEngine) -> None:
    """Ensure Vanguard is not registered as a persona in the roster."""
    assert "vanguard" not in persona_engine.personas
    with pytest.raises(ValueError, match="Unknown persona ID"):
        persona_engine.build_system_prompt("vanguard")
