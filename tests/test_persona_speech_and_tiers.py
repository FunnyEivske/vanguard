"""Tests verifying persona speech styles, word constraints, and inter-agent vocalization toggles."""

import pytest

from vanguard.output.tts_dispatch import TTSDispatcher, VocalConstraintViolationError


def test_dume_word_count_constraint(tts_dispatcher: TTSDispatcher) -> None:
    """Ensure D.U.M.-E respects the 1-3 words limit to humans."""
    # Valid 1-3 word utterances
    u1 = tts_dispatcher.dispatch(persona_id="dume", text="Heat high")
    assert u1.text == "Heat high"
    assert u1.spoken_aloud

    u2 = tts_dispatcher.dispatch(persona_id="dume", text="Friday overprocessing")
    assert u2.text == "Friday overprocessing"

    u3 = tts_dispatcher.dispatch(persona_id="dume", text="Hydraulics ready now")
    assert u3.text == "Hydraulics ready now"

    # Exceeding 3 words must raise VocalConstraintViolationError
    with pytest.raises(VocalConstraintViolationError):
        tts_dispatcher.dispatch(
            persona_id="dume",
            text="The thermal sensors in the workshop are overheating rapidly",
        )


def test_u_single_word_constraint(tts_dispatcher: TTSDispatcher) -> None:
    """Ensure U adheres strictly to single-word vocalizations to humans."""
    valid_words = ["Unit", "Confirmed", "Scanning", "Alert", "Leak"]
    for word in valid_words:
        u = tts_dispatcher.dispatch(persona_id="u", text=word)
        assert u.text == word
        assert u.spoken_aloud

    # Two or more words must raise VocalConstraintViolationError
    with pytest.raises(VocalConstraintViolationError):
        tts_dispatcher.dispatch(persona_id="u", text="System Confirmed")


def test_inter_agent_speech_aloud_toggle(tts_dispatcher: TTSDispatcher) -> None:
    """Ensure inter-agent speech aloud toggle suppresses physical speech when false and enables when true."""
    # When toggle is False (default)
    tts_dispatcher.set_inter_agent_speech_aloud(False)
    utterance_silent = tts_dispatcher.dispatch(
        persona_id="friday",
        text="Jarvis, taking secondary compute cycle.",
        is_inter_agent=True,
    )
    assert not utterance_silent.spoken_aloud

    # When toggle is flipped to True
    tts_dispatcher.set_inter_agent_speech_aloud(True)
    utterance_aloud = tts_dispatcher.dispatch(
        persona_id="friday",
        text="Jarvis, taking secondary compute cycle.",
        is_inter_agent=True,
    )
    assert utterance_aloud.spoken_aloud


def test_persona_vocal_profiles(tts_dispatcher: TTSDispatcher) -> None:
    """Ensure correct accents and cadences are matched per persona."""
    jarvis_out = tts_dispatcher.dispatch(persona_id="jarvis", text="Good evening, Sir.")
    assert "RP" in jarvis_out.accent
    assert "calm" in jarvis_out.cadence.lower()

    friday_out = tts_dispatcher.dispatch(persona_id="friday", text="All systems nominal.")
    assert "scottish" in friday_out.cadence.lower() or "scottish" in friday_out.accent.lower()

    edith_out = tts_dispatcher.dispatch(persona_id="edith", text="Simulating scenarios now!")
    assert "transatlantic" in edith_out.cadence.lower() or "mid-atlantic" in edith_out.accent.lower()
