"""Tests verifying daemon interventions (D.U.M.-E and U) and affective state dynamics."""

from vanguard.persona.engine import PersonaEngine
from vanguard.persona.occ_appraisal import OCCAppraisalEngine


def test_dume_diagnostic_interventions(persona_engine: PersonaEngine) -> None:
    """Ensure D.U.M.-E generates blunt 1-3 word diagnostic interventions with servo whir."""
    resp1 = persona_engine.generate_dume_intervention(target_sibling="friday", issue="overprocessing")
    assert resp1.spoken_text == "Friday overprocessing"
    assert resp1.sound_cue == "servo_whir"
    assert resp1.ui_card is not None

    resp2 = persona_engine.generate_dume_intervention(target_sibling="jarvis", issue="stalled")
    assert resp2.spoken_text == "Jarvis stalled"

    resp3 = persona_engine.generate_dume_intervention(target_sibling="workshop", issue="thermal")
    assert resp3.spoken_text == "Heat high"


def test_u_sensor_warnings(persona_engine: PersonaEngine) -> None:
    """Ensure U emits single-word warnings accompanied by shutter cue."""
    leak_resp = persona_engine.generate_u_sensor_warning("water_detected")
    assert leak_resp.spoken_text == "Leak"
    assert leak_resp.sound_cue == "camera_shutter"
    assert leak_resp.ui_card is not None

    alert_resp = persona_engine.generate_u_sensor_warning("anomaly")
    assert alert_resp.spoken_text == "Alert"


def test_teasing_halt_on_child_distress_or_daemon_failure() -> None:
    """Ensure OCC appraisal immediately prohibits sibling teasing if a child is upset or daemon fails."""
    engine = OCCAppraisalEngine(persona_id="edith")

    # Baseline: teasing permitted
    state = engine.appraise_event("routine", {}, child_is_distressed=False, daemon_failure_active=False)
    assert state.teasing_permitted

    # Child upset: teasing stopped immediately
    state_child_upset = engine.appraise_event("child_crying", {}, child_is_distressed=True)
    assert not state_child_upset.teasing_permitted

    # Daemon failure: teasing stopped immediately
    state_daemon_fail = engine.appraise_event("dume_hardware_stall", {}, daemon_failure_active=True)
    assert not state_daemon_fail.teasing_permitted
