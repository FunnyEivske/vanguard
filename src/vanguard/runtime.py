"""V.A.N.G.U.A.R.D. Live Operational Runtime Orchestrator.

Integrates the EventBus, FloorManager, PermissionRouter, KillSwitch, AuditLogger,
ThreeTierMemoryManager, BiometricStore, VoiceIdentifier, FaceIdentifier, IdentityFusionEngine,
NameSpotter, PersonaEngine, and TTSDispatcher into a unified household runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vanguard.core.audit_logger import AuditLogger
from vanguard.core.event_bus import EventBus
from vanguard.core.events import SpeechRequest
from vanguard.core.floor_manager import FloorManager
from vanguard.core.house_controller import HouseController
from vanguard.core.kill_switch import KillSwitch
from vanguard.core.permission_router import PermissionResult, PermissionRouter, ToolCommand
from vanguard.memory.manager import ThreeTierMemoryManager
from vanguard.output.sound_cues import SoundCueDispatcher
from vanguard.output.tts_dispatch import DispatchedUtterance, TTSDispatcher
from vanguard.output.ui_cards import UICardEmitter
from vanguard.persona.engine import PersonaEngine, PersonaResponse
from vanguard.senses.biometric_store import BiometricStore, load_people_profiles
from vanguard.senses.diarization import StreamingDiarizer
from vanguard.senses.enrollment import BiometricEnrollmentService
from vanguard.senses.face_id import FaceIdentifier
from vanguard.senses.identity_fusion import IdentityFusionEngine, ResolvedIdentity
from vanguard.senses.name_spotting import NameSpotter
from vanguard.senses.sensory_buffer import RollingSensoryBuffer, SensoryFrame
from vanguard.senses.voice_id import VoiceIdentifier


@dataclass
class TurnExecutionResult:
    """Complete result of processing a live household interaction turn."""

    speaker: ResolvedIdentity
    room_occupants: list[ResolvedIdentity]
    target_persona_id: str | None
    persona_response: PersonaResponse | None
    dispatched_utterance: DispatchedUtterance | None
    output_room_id: str = "living_room"
    roamed_from_room: str | None = None
    permission_result: PermissionResult | None = None
    actuation_report: dict[str, Any] | None = None
    halted_by_kill_switch: bool = False


class VanguardRuntime:
    """Complete operational runtime for the V.A.N.G.U.A.R.D. household substrate."""

    def __init__(
        self,
        repo_root: Path | None = None,
        data_dir: Path | None = None,
        enable_hardware_audio: bool | None = None,
    ) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
        self.config_dir = self.repo_root / "config"
        self.data_dir = data_dir or (self.repo_root / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_dir / "system_settings.yaml", encoding="utf-8") as f:
            self.system_settings: dict[str, Any] = yaml.safe_load(f) or {}

        speech_aloud = bool(self.system_settings.get("inter_agent_speech_aloud", False))
        hw_audio_cfg = (
            self.system_settings.get("hardware", {})
            .get("audio_output", {})
            .get("enable_hardware_playback", False)
        )
        self.enable_hardware_audio = (
            enable_hardware_audio if enable_hardware_audio is not None else bool(hw_audio_cfg)
        )

        # Core infrastructure
        self.event_bus = EventBus(inter_agent_speech_aloud=speech_aloud)
        self.audit_logger = AuditLogger(
            event_bus=self.event_bus,
            log_file=self.data_dir / "logs" / "audit.jsonl",
        )
        self.floor_manager = FloorManager(event_bus=self.event_bus)
        self.kill_switch = KillSwitch(event_bus=self.event_bus)

        # Persona engine & memory
        self.persona_engine = PersonaEngine(
            personas_dir=self.config_dir / "personas",
            system_settings=self.system_settings,
        )
        self.memory = ThreeTierMemoryManager(
            event_bus=self.event_bus,
            storage_dir=self.data_dir / "memory",
        )

        # People & Biometrics (Voice + Face ID)
        bio_cfg = self.system_settings.get("biometrics", {})
        bio_db_path = self.data_dir / "biometrics.db"
        self.biometric_store = BiometricStore(db_path=bio_db_path)
        self.people_profiles = load_people_profiles(people_dir=self.config_dir / "people")

        self.voice_identifier = VoiceIdentifier(
            people_profiles=self.people_profiles,
            biometric_store=self.biometric_store,
            similarity_threshold=float(bio_cfg.get("voice_similarity_threshold", 0.78)),
        )
        self.face_identifier = FaceIdentifier(
            people_profiles=self.people_profiles,
            biometric_store=self.biometric_store,
            similarity_threshold=float(bio_cfg.get("face_similarity_threshold", 0.75)),
        )
        self.identity_fusion = IdentityFusionEngine(
            voice_identifier=self.voice_identifier,
            face_identifier=self.face_identifier,
            event_bus=self.event_bus,
        )
        self.enrollment_service = BiometricEnrollmentService(
            people_dir=self.config_dir / "people",
            biometric_store=self.biometric_store,
            voice_identifier=self.voice_identifier,
            face_identifier=self.face_identifier,
        )

        # Permissions, House Controller & Senses
        self.permission_router = PermissionRouter(
            personas_config=self.persona_engine.personas,
            people_config=self.people_profiles,
        )
        rooms_cfg = self.system_settings.get("hardware", {}).get("rooms")
        self.house_controller = HouseController(
            event_bus=self.event_bus,
            permission_router=self.permission_router,
            rooms_config=rooms_cfg,
        )
        self.name_spotter = NameSpotter()
        self.diarizer = StreamingDiarizer(voice_identifier=self.voice_identifier)
        self.sensory_buffer = RollingSensoryBuffer()

        # Output dispatchers
        self.sound_dispatcher = SoundCueDispatcher(
            enable_hardware_playback=self.enable_hardware_audio
        )
        self.ui_card_emitter = UICardEmitter()
        persona_voices = self.system_settings.get("ai_backend", {}).get("persona_voices")
        self.tts_dispatcher = TTSDispatcher(
            personas_config=self.persona_engine.personas,
            inter_agent_speech_aloud=speech_aloud,
            sound_dispatcher=self.sound_dispatcher,
            ui_card_emitter=self.ui_card_emitter,
            enable_hardware_playback=self.enable_hardware_audio,
            persona_voices=persona_voices,
        )

    async def process_turn(
        self,
        text: str,
        room_id: str = "living_room",
        audio_samples: list[float] | bytes | None = None,
        wav_path: str | None = None,
        voice_embedding: list[float] | None = None,
        voice_print_id: str | None = None,
        claimed_name: str | None = None,
        face_image: Any | None = None,
        face_image_path: str | None = None,
        face_embedding: list[float] | None = None,
        face_print_id: str | None = None,
        tool_command: ToolCommand | None = None,
        human_consent_granted: bool = False,
    ) -> TurnExecutionResult:
        """Process a complete live turn with voice/face biometric resolution, floor arbitration, and persona dispatch."""
        # 1. Resolve who is speaking and who is in the room via Voice & Face Biometrics
        speaker = await self.identity_fusion.resolve_identity(
            room_id=room_id,
            audio_samples=audio_samples,
            wav_path=wav_path,
            voice_embedding=voice_embedding,
            voice_print_id=voice_print_id,
            claimed_name=claimed_name,
            face_image=face_image,
            face_image_path=face_image_path,
            face_embedding=face_embedding,
            face_print_id=face_print_id,
        )
        occupants = self.identity_fusion.get_room_occupants(room_id)

        # 2. Append to rolling sensory buffer & diarizer
        self.sensory_buffer.append(
            SensoryFrame(
                room_id=room_id,
                detected_language=speaker.preferred_language,
                raw_text=text,
                telemetry={
                    "speaker_id": speaker.person_id,
                    "voice_matched": speaker.voice_matched,
                    "face_matched": speaker.face_matched,
                    "confidence": speaker.confidence,
                },
            )
        )
        self.diarizer.process_utterance(
            speaker_id=speaker.person_id,
            text=text,
            room_id=room_id,
            language=speaker.preferred_language,
            confidence=speaker.confidence,
        )

        # 3. Check Emergency Kill Switch
        if self.kill_switch.is_engaged:
            return TurnExecutionResult(
                speaker=speaker,
                room_occupants=occupants,
                target_persona_id=None,
                persona_response=None,
                dispatched_utterance=None,
                halted_by_kill_switch=True,
            )

        # 4. Human takes the conversational floor (preempting any active AI speaker)
        await self.floor_manager.request_floor(
            SpeechRequest(
                speaker_id=speaker.person_id,
                room_id=room_id,
                text=text,
                language=speaker.preferred_language,
                priority=10,
            )
        )
        await self.floor_manager.release_floor(speaker_id=speaker.person_id, room_id=room_id)

        # 5. Spot addressed persona
        spotted = self.name_spotter.spot(text)
        target_id = spotted.target_id

        # Substrate Silence Rule: V.A.N.G.U.A.R.D. itself never speaks
        if spotted.is_substrate_reference or target_id == "vanguard":
            return TurnExecutionResult(
                speaker=speaker,
                room_occupants=occupants,
                target_persona_id="vanguard",
                persona_response=None,
                dispatched_utterance=None,
            )

        if not target_id:
            # No default speaker unless the recognized person has an assigned confidante or visitor greeter
            if speaker.confidante_persona and speaker.confidante_persona in self.persona_engine.personas:
                target_id = speaker.confidante_persona
            elif speaker.role == "visitor":
                target_id = (
                    self.people_profiles.get("visitor", {}).get("greeting_persona", "jarvis")
                )
            else:
                return TurnExecutionResult(
                    speaker=speaker,
                    room_occupants=occupants,
                    target_persona_id=None,
                    persona_response=None,
                    dispatched_utterance=None,
                )

        # 6. Determine if the persona is being sent to roam/speak in another room
        previous_persona_room = self.floor_manager.get_persona_room(target_id)
        roam_target = self.house_controller.detect_roam_target_room(text)
        output_room_id = roam_target or room_id
        roamed_from = previous_persona_room or (room_id if output_room_id != room_id else None)

        # 7. Evaluate tool or house actuation command (doors, locks, lights, climate)
        perm_result: PermissionResult | None = None
        actuation_report: dict[str, Any] | None = None
        if tool_command is None:
            inferred_cmd, _, _ = self.house_controller.detect_house_actuation_intent(
                text=text,
                persona_id=target_id,
                default_room_id=output_room_id,
            )
            tool_command = inferred_cmd

        if tool_command is not None:
            perm_result, actuation_report = await self.house_controller.execute_actuation(
                command=tool_command,
                initiating_human_id=speaker.person_id,
                human_consent_granted=human_consent_granted,
            )

        # 8. Build house-wide camera & sensor snapshot across all rooms
        all_room_occupants_map: dict[str, list[dict[str, Any]]] = {}
        for rid in self.house_controller.rooms:
            occ_in_rid = self.identity_fusion.get_room_occupants(rid)
            all_room_occupants_map[rid] = [
                {
                    "person_id": o.person_id,
                    "display_name": o.display_name,
                    "role": o.role,
                    "voice_matched": o.voice_matched,
                    "face_matched": o.face_matched,
                    "confidence": o.confidence,
                }
                for o in occ_in_rid
            ]
        house_snapshot = self.house_controller.get_house_snapshot(all_room_occupants_map)

        # 9. Retrieve shared household facts (filtered by speaker's role: family vs visitor)
        shared_facts_objs = self.memory.shared_graph.query(requester_role=speaker.role)
        shared_facts_dicts = [
            {
                "subject": f.subject,
                "predicate": f.predicate,
                "object_value": f.object_value,
            }
            for f in shared_facts_objs
        ]

        speaker_ctx = {
            "person_id": speaker.person_id,
            "display_name": speaker.display_name,
            "role": speaker.role,
            "preferred_language": speaker.preferred_language,
            "voice_matched": speaker.voice_matched,
            "face_matched": speaker.face_matched,
            "voice_confidence": speaker.voice_confidence,
            "face_confidence": speaker.face_confidence,
            "confidence": speaker.confidence,
        }
        target_room_occupants = all_room_occupants_map.get(output_room_id) or [speaker_ctx]

        # 10. Generate persona response
        response = self.persona_engine.generate_response(
            persona_id=target_id,
            user_text=text,
            speaker_context=speaker_ctx,
            room_occupants=target_room_occupants,
            shared_facts=shared_facts_dicts,
            house_snapshot=house_snapshot,
            actuation_report=actuation_report,
            roamed_from_room=roamed_from,
            active_room_id=output_room_id,
        )

        # 11. Record private thought in persona's isolated Tier 1 SQLite partition
        partition = self.memory.get_or_create_private_partition(target_id)
        partition.store_thought(
            requester_id=target_id,
            topic="interaction_trace",
            content=(
                f"Responded to {speaker.display_name} ({speaker.person_id}, conf={speaker.confidence}) "
                f"from {room_id} -> output in {output_room_id}: '{text}'"
            ),
            metadata={"source_room": room_id, "output_room": output_room_id, "speaker_id": speaker.person_id},
        )

        # 12. Deliver confidential whisper if persona produced one
        if response.inter_agent_whisper:
            w_target = response.inter_agent_whisper.get("target")
            if w_target and w_target in self.persona_engine.personas:
                await self.memory.whisper_pipeline.send_whisper(
                    sender_id=target_id,
                    recipient_id=w_target,
                    subject="daemon_diagnostic",
                    body=str(response.inter_agent_whisper),
                )

        # 13. Request floor for persona in the target output room (migrating presence cleanly) and dispatch TTS
        floor_granted = await self.floor_manager.request_floor(
            SpeechRequest(
                speaker_id=target_id,
                room_id=output_room_id,
                text=response.spoken_text or "",
                language=speaker.preferred_language,
                target_listener=speaker.person_id,
                priority=0,
            )
        )
        dispatched: DispatchedUtterance | None = None
        if floor_granted:
            dispatched = self.tts_dispatcher.dispatch(
                persona_id=target_id,
                text=response.spoken_text,
                room_id=output_room_id,
                language=speaker.preferred_language,
                is_inter_agent=False,
                sound_cue=response.sound_cue,
                ui_card_content=response.ui_card,
            )
            await self.floor_manager.release_floor(speaker_id=target_id, room_id=output_room_id)

        return TurnExecutionResult(
            speaker=speaker,
            room_occupants=occupants,
            target_persona_id=target_id,
            persona_response=response,
            dispatched_utterance=dispatched,
            output_room_id=output_room_id,
            roamed_from_room=roamed_from,
            permission_result=perm_result,
            actuation_report=actuation_report,
        )

    def close(self) -> None:
        """Cleanly close all database connections."""
        self.memory.close()
        self.biometric_store.close()
