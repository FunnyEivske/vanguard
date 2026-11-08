"""Tests verifying voice & face biometric recognition, multi-person room presence, and persona adaptation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vanguard.runtime import VanguardRuntime
from vanguard.senses.biometric_store import BiometricStore
from vanguard.senses.face_id import FaceIdentifier
from vanguard.senses.identity_fusion import IdentityFusionEngine
from vanguard.senses.name_spotting import NameSpotter
from vanguard.senses.voice_id import VoiceIdentifier


def _generate_synthetic_voice(f0_hz: float, formant1_hz: float, formant2_hz: float, sample_rate: int = 16000) -> list[float]:
    """Generate a distinct, realistic multi-harmonic synthetic vocal waveform for testing acoustic speaker ID."""
    duration_sec = 0.6
    n_samples = int(duration_sec * sample_rate)
    samples: list[float] = []
    for i in range(n_samples):
        t = i / sample_rate
        sig = (
            0.50 * math.sin(2.0 * math.pi * f0_hz * t)
            + 0.30 * math.sin(2.0 * math.pi * formant1_hz * t)
            + 0.20 * math.sin(2.0 * math.pi * formant2_hz * t)
        )
        samples.append(sig)
    return samples


def _generate_synthetic_face_image(pattern_type: str, size: int = 32) -> list[list[float]]:
    """Generate distinct 2D facial luminance patterns for testing facial biometric extraction."""
    grid: list[list[float]] = []
    for r in range(size):
        row: list[float] = []
        for c in range(size):
            ny = (r - size / 2.0) / (size / 2.0)
            nx = (c - size / 2.0) / (size / 2.0)
            if pattern_type == "eivind":
                # Strong horizontal brow + angular jawline gradient
                val = 120.0 + 60.0 * math.cos(3.0 * ny) - 40.0 * abs(nx) + 25.0 * (nx * ny)
            elif pattern_type == "wife":
                # Circular radial symmetry + soft cheek contours
                dist = math.sqrt(nx * nx + ny * ny)
                val = 150.0 + 70.0 * math.cos(4.0 * dist) + 20.0 * math.sin(2.0 * ny)
            elif pattern_type == "child_elder":
                # High-frequency upper-face features + narrow chin
                val = 130.0 + 50.0 * math.sin(5.0 * nx) * math.cos(4.0 * ny) - 30.0 * ny
            else:
                # Distinct diagonal gradient for unknown visitor
                val = 100.0 + 80.0 * (nx + ny)
            row.append(val)
        grid.append(row)
    return grid


def test_acoustic_voice_enrollment_and_identification(people_config: dict) -> None:
    """Ensure VoiceIdentifier distinguishes different people from raw acoustic waveforms."""
    store = BiometricStore(db_path=":memory:")
    voice_id = VoiceIdentifier(people_profiles=people_config, biometric_store=store, similarity_threshold=0.80)

    eivind_audio = _generate_synthetic_voice(f0_hz=110.0, formant1_hz=520.0, formant2_hz=1480.0)
    wife_audio = _generate_synthetic_voice(f0_hz=215.0, formant1_hz=760.0, formant2_hz=2350.0)
    child_audio = _generate_synthetic_voice(f0_hz=295.0, formant1_hz=980.0, formant2_hz=2950.0)
    stranger_audio = _generate_synthetic_voice(f0_hz=155.0, formant1_hz=3100.0, formant2_hz=3850.0)

    voice_id.enroll_voice("eivind", audio_samples=eivind_audio)
    voice_id.enroll_voice("wife", audio_samples=wife_audio)
    voice_id.enroll_voice("child_elder", audio_samples=child_audio)

    # Identify Eivind with slight amplitude variation
    eivind_query = [s * 0.92 for s in eivind_audio]
    res_eivind = voice_id.identify(audio_samples=eivind_query)
    assert res_eivind.speaker_id == "eivind"
    assert res_eivind.is_known_family_member
    assert res_eivind.role == "family_head"
    assert res_eivind.confidence >= 0.90

    # Identify Wife
    res_wife = voice_id.identify(audio_samples=wife_audio)
    assert res_wife.speaker_id == "wife"
    assert res_wife.role == "family_head"

    # Identify Child
    res_child = voice_id.identify(audio_samples=child_audio)
    assert res_child.speaker_id == "child_elder"
    assert res_child.role == "child"

    # Unenrolled voice falls back to visitor
    res_stranger = voice_id.identify(audio_samples=stranger_audio)
    assert res_stranger.speaker_id == "visitor"
    assert not res_stranger.is_known_family_member
    store.close()


def test_facial_enrollment_and_multi_person_frame_identification(people_config: dict) -> None:
    """Ensure FaceIdentifier enrolls and recognizes multiple distinct faces in a room."""
    store = BiometricStore(db_path=":memory:")
    face_id = FaceIdentifier(people_profiles=people_config, biometric_store=store, similarity_threshold=0.80)

    eivind_face = _generate_synthetic_face_image("eivind")
    wife_face = _generate_synthetic_face_image("wife")
    child_face = _generate_synthetic_face_image("child_elder")
    visitor_face = _generate_synthetic_face_image("visitor")

    face_id.enroll_face("eivind", image=eivind_face)
    face_id.enroll_face("wife", image=wife_face)
    face_id.enroll_face("child_elder", image=child_face)

    # Single face identification
    res_eivind = face_id.identify(image=eivind_face)
    assert res_eivind.person_id == "eivind"
    assert res_eivind.is_known_family_member
    assert res_eivind.confidence >= 0.95

    res_wife = face_id.identify(image=wife_face)
    assert res_wife.person_id == "wife"

    res_visitor = face_id.identify(image=visitor_face)
    assert res_visitor.person_id == "visitor"
    assert not res_visitor.is_known_family_member

    # Multi-face room frame identification
    frame_results = face_id.identify_faces_in_frame(
        face_crops=[(eivind_face, None), (child_face, None)]
    )
    assert [r.person_id for r in frame_results] == ["eivind", "child_elder"]
    store.close()


@pytest.mark.asyncio
async def test_multimodal_identity_fusion_and_room_presence(people_config: dict) -> None:
    """Ensure IdentityFusionEngine combines voice + face biometrics and tracks room occupants."""
    store = BiometricStore(db_path=":memory:")
    voice_id = VoiceIdentifier(people_profiles=people_config, biometric_store=store)
    face_id = FaceIdentifier(people_profiles=people_config, biometric_store=store)
    fusion = IdentityFusionEngine(voice_identifier=voice_id, face_identifier=face_id)

    eivind_audio = _generate_synthetic_voice(110.0, 520.0, 1480.0)
    eivind_face = _generate_synthetic_face_image("eivind")
    child_face = _generate_synthetic_face_image("child_elder")

    voice_id.enroll_voice("eivind", audio_samples=eivind_audio)
    face_id.enroll_face("eivind", image=eivind_face)
    face_id.enroll_face("child_elder", image=child_face)

    # Update visual presence: Eivind and Child are seen in living_room
    occupants = await fusion.update_room_visual_presence(
        room_id="living_room",
        face_crops=[(eivind_face, None), (child_face, None)],
    )
    assert len(occupants) == 2
    assert fusion.room_has_children("living_room")

    # Eivind speaks in living_room -> both voice and face are matched
    resolved = await fusion.resolve_identity(
        room_id="living_room",
        audio_samples=eivind_audio,
    )
    assert resolved.person_id == "eivind"
    assert resolved.voice_matched
    assert resolved.face_matched
    assert resolved.confidence >= 0.95
    store.close()


@pytest.mark.asyncio
async def test_all_five_personas_recognize_people_in_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure Jarvis, Friday, Edith, Dum-E, and U all recognize different people by voice & face in VanguardRuntime."""
    monkeypatch.setenv("VANGUARD_OFFLINE_MODE", "1")
    runtime = VanguardRuntime(data_dir=tmp_path / "data", enable_hardware_audio=False)

    eivind_audio = _generate_synthetic_voice(110.0, 520.0, 1480.0)
    eivind_face = _generate_synthetic_face_image("eivind")
    child_audio = _generate_synthetic_voice(295.0, 980.0, 2950.0)
    child_face = _generate_synthetic_face_image("child_elder")

    runtime.enrollment_service.enroll_person(
        person_id="eivind",
        voice_samples=eivind_audio,
        face_image=eivind_face,
        persist_yaml=False,
    )
    runtime.enrollment_service.enroll_person(
        person_id="child_elder",
        voice_samples=child_audio,
        face_image=child_face,
        persist_yaml=False,
    )

    # 1. J.A.R.V.I.S. recognizes Eivind by voice + face
    t_jarvis = await runtime.process_turn(
        text="Jarvis, check the perimeter.",
        room_id="office",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert t_jarvis.speaker.person_id == "eivind"
    assert t_jarvis.speaker.voice_matched and t_jarvis.speaker.face_matched
    assert t_jarvis.persona_response is not None
    assert "Sir" in (t_jarvis.persona_response.spoken_text or "")

    # 2. F.R.I.D.A.Y. (also matched via phonetic 'firday') recognizes Eivind
    t_friday = await runtime.process_turn(
        text="Firday, give me telemetry.",
        room_id="office",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert t_friday.target_persona_id == "friday"
    assert t_friday.persona_response is not None
    assert "Eivind" in (t_friday.persona_response.spoken_text or "")

    # 3. E.D.I.T.H. (also matched via 'eath') recognizes Child and enables child-safe mode
    t_edith = await runtime.process_turn(
        text="Eath, let's build a castle simulation!",
        room_id="living_room",
        audio_samples=child_audio,
        face_image=child_face,
    )
    assert t_edith.target_persona_id == "edith"
    assert t_edith.speaker.person_id == "child_elder"
    assert t_edith.persona_response is not None
    assert "Child-safe" in (t_edith.persona_response.spoken_text or "")

    # 4. D.U.M.-E (matched via 'dummy') verifies speaker in <= 3 words + servo_whir
    t_dume = await runtime.process_turn(
        text="Dummy, who do you see?",
        room_id="workshop",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert t_dume.target_persona_id == "dume"
    assert t_dume.persona_response is not None
    assert t_dume.persona_response.spoken_text == "Eivind verified"
    assert t_dume.persona_response.sound_cue == "servo_whir"

    # 5. U (matched via 'u' / 'unit') scans and logs speaker biometrics in 1 word + camera_shutter
    t_u = await runtime.process_turn(
        text="U, scan the room.",
        room_id="workshop",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert t_u.target_persona_id == "u"
    assert t_u.persona_response is not None
    assert t_u.persona_response.spoken_text == "Scanning"
    assert t_u.persona_response.sound_cue == "camera_shutter"
    assert t_u.persona_response.ui_card["recognized_speaker"] == "Eivind"

    runtime.close()


def test_phonetic_name_spotting_variants() -> None:
    """Ensure NameSpotter handles common phonetic spellings (firday, eath, dummy, unit)."""
    spotter = NameSpotter()
    assert spotter.spot("Jarvis, status").target_id == "jarvis"
    assert spotter.spot("Firday, report").target_id == "friday"
    assert spotter.spot("Eath, run sim").target_id == "edith"
    assert spotter.spot("Dummy, check heat").target_id == "dume"
    assert spotter.spot("Unit, log this").target_id == "u"


def test_online_first_local_backup_llm_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LLMClient tries Online API first and falls back to Local Ollama when online fails."""
    from vanguard.persona.llm_client import LLMClient

    monkeypatch.delenv("VANGUARD_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-online-key")

    call_order: list[str] = []

    client = LLMClient(online_provider="openai_compatible", backup_provider="ollama")

    # Case 1: Online succeeds -> returns online response immediately without calling local backup
    monkeypatch.setattr(client, "_call_openai", lambda sys_p, usr_m: (call_order.append("online"), "Online response")[1])
    monkeypatch.setattr(client, "_call_ollama", lambda sys_p, usr_m: (call_order.append("local_ollama"), "Backup response")[1])

    res1 = client.generate("sys", "hello", "jarvis")
    assert res1 == "Online response"
    assert call_order == ["online"]

    # Case 2: Online fails (returns None) -> automatically falls back to local Ollama
    call_order.clear()
    monkeypatch.setattr(client, "_call_openai", lambda sys_p, usr_m: (call_order.append("online_failed"), None)[1])
    res2 = client.generate("sys", "hello", "jarvis")
    assert res2 == "Backup response"
    assert call_order == ["online_failed", "local_ollama"]


@pytest.mark.asyncio
async def test_multi_room_roaming_cameras_and_door_actuation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify AIs roam freely between rooms, read house-wide cameras, and enforce door actuation permissions."""
    monkeypatch.setenv("VANGUARD_OFFLINE_MODE", "1")
    runtime = VanguardRuntime(data_dir=tmp_path / "data", enable_hardware_audio=False)

    eivind_audio = _generate_synthetic_voice(110.0, 520.0, 1480.0)
    eivind_face = _generate_synthetic_face_image("eivind")
    wife_face = _generate_synthetic_face_image("wife")

    runtime.enrollment_service.enroll_person("eivind", voice_samples=eivind_audio, face_image=eivind_face, persist_yaml=False)
    runtime.enrollment_service.enroll_person("wife", face_image=wife_face, persist_yaml=False)

    # 1. Kitchen camera spots Wife
    await runtime.identity_fusion.update_room_visual_presence(
        room_id="kitchen",
        face_crops=[(wife_face, None)],
    )

    # 2. Eivind speaks to Jarvis in Office
    t1 = await runtime.process_turn(
        text="Jarvis, check the house cameras.",
        room_id="office",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert runtime.floor_manager.get_persona_room("jarvis") == "office"
    assert "Wife in Kjøkken (kitchen)" in (t1.persona_response.spoken_text or "")

    # 3. Eivind sends Jarvis from Office to Kitchen -> Jarvis roams to Kitchen and speaks on Kitchen speakers
    t2 = await runtime.process_turn(
        text="Jarvis, go to the kitchen and unlock the kitchen door.",
        room_id="office",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert t2.roamed_from_room == "office"
    assert t2.output_room_id == "kitchen"
    assert runtime.floor_manager.get_persona_room("jarvis") == "kitchen"
    assert t2.actuation_report is not None and t2.actuation_report["executed"]
    assert not runtime.house_controller.rooms["kitchen"].doors["kitchen_back_door"].is_locked

    # 4. Eivind asks Edith to unlock Workshop door -> strictly blocked by PermissionRouter (Edith has zero door authority)
    t3 = await runtime.process_turn(
        text="Edith, unlock the workshop door.",
        room_id="workshop",
        audio_samples=eivind_audio,
        face_image=eivind_face,
    )
    assert t3.actuation_report is not None and not t3.actuation_report["executed"]
    assert runtime.house_controller.rooms["workshop"].doors["workshop_heavy_door"].is_locked

    runtime.close()


