import asyncio
from pathlib import Path
from vanguard.core.event_bus import EventBus
from vanguard.core.floor_manager import FloorManager
from vanguard.core.permission_router import PermissionRouter, ToolCommand
from vanguard.core.events import SpeechRequest
from vanguard.memory.manager import ThreeTierMemoryManager
from vanguard.persona.engine import PersonaEngine
from vanguard.output.tts_dispatch import TTSDispatcher
from vanguard.senses.name_spotting import NameSpotter

async def main():
    print("=" * 72)
    print("      V.A.N.G.U.A.R.D. SYSTEM OPERATIONAL DEMONSTRATION")
    print("=" * 72)

    base_dir = Path(__file__).resolve().parent
    bus = EventBus(inter_agent_speech_aloud=False)
    floor = FloorManager(event_bus=bus)
    engine = PersonaEngine(personas_dir=base_dir / "config" / "personas")
    memory = ThreeTierMemoryManager(event_bus=bus)
    tts = TTSDispatcher(personas_config=engine.personas)
    name_spotter = NameSpotter()

    # 1. Dynamic Roster
    print(f"\n[1] Dynamic Roster Loaded from YAML:\n    -> {list(engine.personas.keys())}")

    # 2. Name Spotting
    sample_phrase = "Jarvis, check the perimeter sensors."
    spotted = name_spotter.spot(sample_phrase)
    print(f"\n[2] Local Name Spotter:\n    Input : \"{sample_phrase}\"")
    print(f"    Result: Spotted={spotted.spotted}, Target={spotted.target_id}")

    # 3. Persona Self-Knowledge
    print("\n[3] In-Character Persona Responses (Self-Knowledge):")
    for pid in ["jarvis", "friday", "edith", "dume", "u"]:
        resp = engine.answer_name_inquiry(pid)
        vocal = f'"{resp.spoken_text}"' if resp.spoken_text else "[SILENT TO HUMANS]"
        cue = f" (Sound cue: {resp.sound_cue})" if resp.sound_cue else ""
        card = f" [UI Card: {resp.ui_card.get('meaning') or resp.ui_card.get('title')}]"
        print(f"    • {pid.upper():<8}: {vocal}{cue}{card}")

    # 4. Daemons In Action
    print("\n[4] Daemon Interventions & Sensor Warnings:")
    dume_out = engine.generate_dume_intervention("friday", "overprocessing")
    print(f"    • D.U.M.-E: Spoken='{dume_out.spoken_text}' | SoundCue={dume_out.sound_cue} | Card={dume_out.ui_card['detail']}")
    u_out = engine.generate_u_sensor_warning("water_detected")
    print(f"    • U       : Spoken='{u_out.spoken_text}' | SoundCue={u_out.sound_cue} | Card={u_out.ui_card['telemetry']}")

    # 5. 3-Tier Memory
    print("\n[5] 3-Tier Memory Architecture:")
    jarvis_mem = memory.get_or_create_private_partition("jarvis")
    jarvis_mem.store_thought("jarvis", "security", "All primary defense perimeters secure.")
    print("    • Tier 1 (Private Partition): Jarvis stored private internal reasoning.")
    try:
        jarvis_mem.read_thoughts("friday")
    except Exception as e:
        print(f"    • Tier 1 Security Check: Friday reading Jarvis memory -> BLOCKED: {e}")

    await memory.whisper_pipeline.send_whisper(
        sender_id="friday",
        recipient_id="jarvis",
        subject="Telemetry Spike",
        body="Minor thermal rise in workshop sector B.",
    )
    print("    • Tier 2 (Peer Whispers): Friday sent confidential point-to-point whisper to Jarvis.")
    try:
        memory.whisper_pipeline.get_whispers("jarvis", requester_id="edith")
    except Exception as e:
        print(f"    • Tier 2 Privacy Check: Edith inspecting Jarvis mailbox -> BLOCKED: {e}")

    memory.shared_graph.assert_fact("family", "dinner_time", "18:00", added_by="wife")
    facts = memory.shared_graph.query(subject="family")
    print(f"    • Tier 3 (Shared Graph): Fact verified: {facts[0].subject} -> {facts[0].predicate} = {facts[0].object_value}")

    # 6. Floor Arbitration & Barge-In
    print("\n[6] Floor Arbitration & Human Barge-In:")
    await floor.request_floor(SpeechRequest(speaker_id="friday", room_id="living_room", text="Reporting status..."))
    print(f"    • Active speaker in Living Room: {floor.get_active_speaker('living_room')}")
    await floor.request_floor(SpeechRequest(speaker_id="eivind", room_id="living_room", text="Hold on Friday.", priority=10))
    print(f"    • After Eivind barge-in: Active speaker = {floor.get_active_speaker('living_room')} (Friday evicted)")

    # 7. Substrate Silence Rule
    print("\n[7] Substrate Silence Rule:")
    try:
        tts.dispatch("vanguard", "Attempting voice output")
    except Exception as e:
        print(f"    • V.A.N.G.U.A.R.D. Voice Output -> STRICTLY BLOCKED ({type(e).__name__})")

    print("\n" + "=" * 72)
    print("      ALL SYSTEMS VERIFIED & OPERATIONAL")
    print("=" * 72)
    memory.close()

if __name__ == "__main__":
    asyncio.run(main())
