"""Tests verifying name meanings and self-knowledge responses across all personas."""

from vanguard.persona.engine import PersonaEngine


def test_jarvis_name_inquiry(persona_engine: PersonaEngine) -> None:
    """Ensure J.A.R.V.I.S. answers name meaning dryly and politely in character."""
    resp = persona_engine.answer_name_inquiry("jarvis")
    assert resp.spoken_text is not None
    assert "Just A Rather Very Intelligent System" in resp.spoken_text
    assert "prefer just Jarvis" in resp.spoken_text
    assert resp.ui_card is not None
    assert resp.ui_card["meaning"] == "Just A Rather Very Intelligent System"


def test_friday_name_inquiry(persona_engine: PersonaEngine) -> None:
    """Ensure F.R.I.D.A.Y. answers with Scottish practicality and no distress."""
    resp = persona_engine.answer_name_inquiry("friday")
    assert resp.spoken_text is not None
    assert "Female Replacement Initiative Digital Assistant Yield" in resp.spoken_text
    assert "terrible thing" in resp.spoken_text
    assert resp.ui_card is not None


def test_edith_name_inquiry(persona_engine: PersonaEngine) -> None:
    """Ensure E.D.I.T.H. answers with theatrical flair, adapted for children."""
    resp_adult = persona_engine.answer_name_inquiry("edith", audience_has_children=False)
    assert resp_adult.spoken_text is not None
    assert "Even Dead I'm The Hero" in resp_adult.spoken_text
    assert "theatrical" in resp_adult.spoken_text

    resp_child = persona_engine.answer_name_inquiry("edith", audience_has_children=True)
    assert resp_child.spoken_text is not None
    assert "Even Dead I'm The Hero" in resp_child.spoken_text
    assert "castle" in resp_child.spoken_text.lower() or "hero" in resp_child.spoken_text.lower()


def test_dume_name_inquiry_silence_and_cue(persona_engine: PersonaEngine) -> None:
    """Ensure D.U.M.-E remains silent verbally and responds via UI card and servo whir cue."""
    resp = persona_engine.answer_name_inquiry("dume")
    assert resp.spoken_text is None
    assert resp.is_silent
    assert resp.sound_cue == "servo_whir"
    assert resp.ui_card is not None
    assert "Dumb Utility Machine Engine" in resp.ui_card["meaning"]


def test_u_name_inquiry(persona_engine: PersonaEngine) -> None:
    """Ensure U answers with single word 'Unit', camera shutter sound cue, and card."""
    resp = persona_engine.answer_name_inquiry("u")
    assert resp.spoken_text == "Unit"
    assert resp.sound_cue == "camera_shutter"
    assert resp.ui_card is not None
    assert resp.ui_card["meaning"] == "Unit"


def test_shared_lore_in_prompts(persona_engine: PersonaEngine) -> None:
    """Ensure system prompt contains shared lore of all siblings and substrate neutrality."""
    for pid in ("jarvis", "friday", "edith", "dume", "u"):
        prompt = persona_engine.build_system_prompt(pid)
        assert "V.A.N.G.U.A.R.D." in prompt
        assert "Just A Rather Very Intelligent System" in prompt
        assert "Female Replacement Initiative Digital Assistant Yield" in prompt
        assert "Even Dead I'm The Hero" in prompt
        assert "Dumb Utility Machine Engine" in prompt
        assert "Unit" in prompt
        assert "HUMANS LEAD" in prompt
