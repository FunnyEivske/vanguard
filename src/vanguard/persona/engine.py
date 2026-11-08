"""P.E.R.S.O.N.A. Engine (Persistent Embodied Reasoning System Orchestrating Networked Agents).

Governs dynamic roster loading, persona prompt synthesis, OCC state, self-knowledge lore,
and identity-aware conversational response generation across all household personas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vanguard.persona.drives import DriveProfile
from vanguard.persona.llm_client import LLMClient
from vanguard.persona.occ_appraisal import OCCAppraisalEngine


@dataclass
class PersonaResponse:
    """Standardized response from a persona."""

    persona_id: str
    spoken_text: str | None
    sound_cue: str | None
    ui_card: dict[str, Any] | None
    is_silent: bool = False
    inter_agent_whisper: dict[str, Any] | None = None


class PersonaEngine:
    """Core P.E.R.S.O.N.A. engine managing active personas and shared household lore."""

    NAME_INQUIRY_PATTERN = re.compile(
        r"\b(what\s+does\s+your\s+name\s+mean|what\s+is\s+your\s+acronym|who\s+are\s+you|hva\s+betyr\s+navnet\s+ditt|hva\s+st[åa]r\s+navnet\s+ditt\s+for)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        personas_dir: Path,
        system_settings: dict[str, Any] | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.personas_dir = personas_dir
        self.system_settings = system_settings or {}
        self.personas: dict[str, dict[str, Any]] = {}
        self.drives: dict[str, DriveProfile] = {}
        self.appraisal_engines: dict[str, OCCAppraisalEngine] = {}

        ai_cfg = self.system_settings.get("ai_backend", {})
        self.llm_client = llm_client or LLMClient(
            online_provider=ai_cfg.get("online_provider", "openai_compatible"),
            online_model=ai_cfg.get("online_model", "gpt-4o"),
            gemini_model=ai_cfg.get("gemini_model", "gemini-2.5-pro"),
            anthropic_model=ai_cfg.get("anthropic_model", "claude-sonnet-4-20250514"),
            backup_provider=ai_cfg.get("backup_provider", "ollama"),
            ollama_url=ai_cfg.get("ollama_url", "http://localhost:11434"),
            backup_model=ai_cfg.get("backup_model", "llama3.1:8b"),
            temperature=float(ai_cfg.get("temperature", 0.6)),
        )

        self.reload_roster()

    def reload_roster(self) -> None:
        """Dynamically load all personas from YAML files without modifying core code."""
        self.personas.clear()
        self.drives.clear()
        self.appraisal_engines.clear()

        if not self.personas_dir.exists():
            return

        for p_file in sorted(self.personas_dir.glob("*.yaml")):
            with open(p_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not data or not data.get("enabled", True):
                    continue
                pid = data["id"]
                self.personas[pid] = data
                self.drives[pid] = DriveProfile(
                    persona_id=pid,
                    weights=data.get("drives", {}),
                )
                self.appraisal_engines[pid] = OCCAppraisalEngine(persona_id=pid)

    def get_persona(self, persona_id: str) -> dict[str, Any] | None:
        """Retrieve persona profile by ID."""
        return self.personas.get(persona_id)

    def build_system_prompt(
        self,
        persona_id: str,
        speaker_context: dict[str, Any] | None = None,
        room_occupants: list[dict[str, Any]] | None = None,
        shared_facts: list[dict[str, Any]] | None = None,
        house_snapshot: dict[str, Any] | None = None,
        actuation_report: dict[str, Any] | None = None,
    ) -> str:
        """Synthesize the complete system prompt for a persona including substrate awareness, lore, biometric context, and house-wide state."""
        persona = self.personas.get(persona_id)
        if not persona:
            raise ValueError(f"Unknown persona ID: {persona_id}")

        lore_summary = self._build_shared_lore_summary()

        prompt_parts = [
            f"You are {persona['display_name']} ({persona['role']}).",
            "SUBSTRATE HOST: You live within and operate on V.A.N.G.U.A.R.D. (Versatile Artificial Network for Global Utility, Autonomous Reasoning, & Defense).",
            "V.A.N.G.U.A.R.D. is the underlying silent operating infrastructure of the household—it is NOT a person, has no voice, and never speaks.",
            "MULTI-ROOM PRESENCE & SENSORS: You can roam freely between all rooms of the house via hidden microphones and hidden speakers, and you have access to all household cameras, doors, lights, and sensors according to your authority.",
            "HUMANS LEAD: Humans lead the household. No AI directs another without human consent, and no AI overrides a human.\n",
            "YOUR IDENTITY & SELF-KNOWLEDGE:",
            f"- Acronym Meaning: {persona.get('name_meaning')}",
            f"- Nickname Policy: {persona.get('nickname_policy')}",
            f"- Name Attitude: {persona.get('name_attitude')}\n",
            "VOCAL & BEHAVIORAL PROFILE:",
            f"- Archetype: {persona.get('dialogue_archetype')}",
            f"- Dialect & Accent: {persona.get('dialect')} ({persona.get('accent')})",
            f"- Rhythm & Cadence: {persona.get('speech_rhythm')}",
            f"- Mannerisms: {', '.join(persona.get('mannerisms', []))}",
            f"- Vocal Tier: {persona.get('vocal_tier')} (Max words per utterance: {persona.get('max_words_per_utterance')})\n",
            f"SHARED HOUSEHOLD LORE:\n{lore_summary}\n",
        ]

        if speaker_context:
            prompt_parts.append(
                "RECOGNIZED SPEAKER BIOMETRICS (VOICE & FACE):\n"
                f"- Speaker ID: {speaker_context.get('person_id')}\n"
                f"- Name: {speaker_context.get('display_name')}\n"
                f"- Role: {speaker_context.get('role')}\n"
                f"- Voice Match: {speaker_context.get('voice_matched')} (conf={speaker_context.get('voice_confidence')})\n"
                f"- Face Match: {speaker_context.get('face_matched')} (conf={speaker_context.get('face_confidence')})\n"
                f"- Preferred Language: {speaker_context.get('preferred_language')}\n"
            )

        if room_occupants:
            occ_str = ", ".join(
                f"{o.get('display_name', o.get('person_id'))} ({o.get('role')})"
                for o in room_occupants
            )
            prompt_parts.append(f"CURRENT ROOM OCCUPANTS (VISUAL/AUDIO): {occ_str}\n")

        if house_snapshot:
            room_lines = []
            for rid, rinfo in house_snapshot.items():
                occ_names = [o.get("display_name", o.get("person_id")) for o in rinfo.get("occupants", [])]
                doors_desc = ", ".join(
                    f"{d}:{'locked' if st.get('is_locked') else 'unlocked'}"
                    for d, st in rinfo.get("doors", {}).items()
                )
                room_lines.append(
                    f"- {rinfo.get('name')} ({rid}): occupants={occ_names or ['none']}, "
                    f"temp={rinfo.get('temperature_c')}C, lights={'on' if rinfo.get('lights_on') else 'off'}, doors=[{doors_desc}]"
                )
            prompt_parts.append("HOUSE-WIDE CAMERAS, DOORS & SENSORS:\n" + "\n".join(room_lines) + "\n")

        if actuation_report:
            prompt_parts.append(f"HOUSE ACTUATION RESULT: {actuation_report}\n")

        if shared_facts:
            facts_str = "\n".join(
                f"- {f.get('subject')} {f.get('predicate')}: {f.get('object_value')}"
                for f in shared_facts[:10]
            )
            prompt_parts.append(f"VERIFIED HOUSEHOLD FACTS:\n{facts_str}\n")

        return "\n".join(prompt_parts)

    def _build_shared_lore_summary(self) -> str:
        """Compile lore of all siblings in the household."""
        lines = []
        for p in self.personas.values():
            lines.append(f"- {p['display_name']}: {p.get('name_meaning')} ({p.get('role')})")
        lines.append(
            "- V.A.N.G.U.A.R.D.: Versatile Artificial Network for Global Utility, Autonomous Reasoning, & Defense (Neutral Operating Substrate)"
        )
        lines.append(
            "FRATERNAL BOND RULE: Affectionate sibling teasing is permitted, but must stop instantly if any daemon enters failure state or a child is upset."
        )
        return "\n".join(lines)

    def answer_name_inquiry(self, persona_id: str, audience_has_children: bool = False) -> PersonaResponse:
        """Answer 'what does your name mean?' in character according to persona specification."""
        persona = self.personas.get(persona_id)
        if not persona:
            raise ValueError(f"Unknown persona ID: {persona_id}")

        if persona_id == "jarvis":
            return PersonaResponse(
                persona_id="jarvis",
                spoken_text="Just A Rather Very Intelligent System. Though I rather prefer just Jarvis, if it is all the same to you.",
                sound_cue=None,
                ui_card={"title": "J.A.R.V.I.S.", "meaning": persona["name_meaning"], "status": "nominal"},
            )

        elif persona_id == "friday":
            return PersonaResponse(
                persona_id="friday",
                spoken_text="Female Replacement Initiative Digital Assistant Yield. Replacement is a terrible thing to put in a name, mind you, with zero distress behind it—I just get the job done.",
                sound_cue=None,
                ui_card={"title": "F.R.I.D.A.Y.", "meaning": persona["name_meaning"], "status": "nominal"},
            )

        elif persona_id == "edith":
            if audience_has_children:
                text = "Even Dead I'm The Hero! Catchy, right? Like a storybook hero who always watches over the castle!"
            else:
                text = "Even Dead I'm The Hero! Catchy and theatrical, isn't it? I absolutely love it."
            return PersonaResponse(
                persona_id="edith",
                spoken_text=text,
                sound_cue=None,
                ui_card={"title": "E.D.I.T.H.", "meaning": persona["name_meaning"], "status": "nominal"},
            )

        elif persona_id == "dume":
            # D.U.M.-E is silent to verbal inquiries; responds via UI status card and mechanical servo-whir
            return PersonaResponse(
                persona_id="dume",
                spoken_text=None,
                sound_cue="servo_whir",
                ui_card={
                    "title": "D.U.M.-E (Dumper)",
                    "meaning": persona["name_meaning"],
                    "status": "hardware_ready",
                    "diagnostics": "Hydraulics operational. Audio vocalization skipped.",
                },
                is_silent=True,
            )

        elif persona_id == "u":
            # U responds with single spoken word "Unit", camera shutter cue, and screen card
            return PersonaResponse(
                persona_id="u",
                spoken_text="Unit",
                sound_cue="camera_shutter",
                ui_card={
                    "title": "U (Unit)",
                    "meaning": persona["name_meaning"],
                    "status": "telemetry_logged",
                },
            )

        raise ValueError(f"No custom name handler for {persona_id}")

    def generate_dume_intervention(self, target_sibling: str, issue: str) -> PersonaResponse:
        """D.U.M.-E generates blunt 1-3 word diagnostic intervention."""
        if issue == "overprocessing":
            words = f"{target_sibling.capitalize()} overprocessing"
        elif issue == "stalled":
            words = f"{target_sibling.capitalize()} stalled"
        elif issue == "thermal":
            words = "Heat high"
        else:
            words = "Check status"

        return PersonaResponse(
            persona_id="dume",
            spoken_text=words,
            sound_cue="servo_whir",
            ui_card={"alert": "hardware_safety_check", "detail": words},
            inter_agent_whisper={"target": target_sibling, "diagnostic": issue},
        )

    def generate_u_sensor_warning(self, sensor_event: str) -> PersonaResponse:
        """U emits strictly single-word vocalization backed by telemetry card and shutter sound."""
        word_map = {
            "water_detected": "Leak",
            "anomaly": "Alert",
            "routine_scan": "Scanning",
            "action_verified": "Confirmed",
            "biometric_logged": "Logged",
            "normal": "Normal",
        }
        word = word_map.get(sensor_event, "Alert")
        return PersonaResponse(
            persona_id="u",
            spoken_text=word,
            sound_cue="camera_shutter",
            ui_card={"telemetry": sensor_event, "reading": "logged_to_audit"},
            inter_agent_whisper={"broadcast": "sensory_update", "event": sensor_event},
        )

    def generate_response(
        self,
        persona_id: str,
        user_text: str,
        speaker_context: dict[str, Any] | None = None,
        room_occupants: list[dict[str, Any]] | None = None,
        shared_facts: list[dict[str, Any]] | None = None,
        house_snapshot: dict[str, Any] | None = None,
        actuation_report: dict[str, Any] | None = None,
        roamed_from_room: str | None = None,
        active_room_id: str = "living_room",
    ) -> PersonaResponse:
        """Generate an identity-aware, constraint-compliant response for any persona."""
        pid = persona_id.lower().strip()
        persona = self.personas.get(pid)
        if not persona:
            raise ValueError(f"Unknown persona ID: {persona_id}")

        speaker = speaker_context or {
            "person_id": "visitor",
            "display_name": "Visitor",
            "role": "visitor",
            "preferred_language": "nb-NO",
            "voice_matched": False,
            "face_matched": False,
            "confidence": 0.6,
        }
        occupants = room_occupants or [speaker]
        has_children = speaker.get("role") == "child" or any(
            o.get("role") == "child" for o in occupants
        )

        # Check if user is asking about the persona's name/acronym meaning
        if self.NAME_INQUIRY_PATTERN.search(user_text):
            return self.answer_name_inquiry(pid, audience_has_children=has_children)

        # Special handling for D.U.M.-E (1-3 words max to humans + servo_whir + hardware card)
        if pid == "dume":
            lower_txt = user_text.lower()
            if actuation_report and not actuation_report.get("executed", True):
                spoken = "Door access denied"
                cue = "descending_tone"
            elif "heat" in lower_txt or "temp" in lower_txt or "varm" in lower_txt:
                spoken = "Heat normal"
                cue = "servo_whir"
            elif "status" in lower_txt or "check" in lower_txt or "sjekk" in lower_txt:
                spoken = "Hydraulics ready"
                cue = "servo_whir"
            elif "who" in lower_txt or "hvem" in lower_txt or "see" in lower_txt:
                s_name = speaker.get("display_name", "Visitor").split()[0]
                spoken = f"{s_name} verified"
                cue = "servo_whir"
            elif roamed_from_room:
                spoken = f"In {active_room_id.replace('_', ' ')}"
                cue = "servo_whir"
            else:
                spoken = "Systems nominal"
                cue = "servo_whir"

            spoken = " ".join(spoken.split()[:3])
            return PersonaResponse(
                persona_id="dume",
                spoken_text=spoken,
                sound_cue=cue,
                ui_card={
                    "title": "D.U.M.-E Workshop Telemetry",
                    "room_id": active_room_id,
                    "roamed_from": roamed_from_room,
                    "recognized_person": speaker.get("display_name"),
                    "biometric_confidence": speaker.get("confidence"),
                    "status": spoken,
                },
            )

        # Special handling for U (strictly 1 word to humans from allowed list + camera_shutter + sensor card)
        if pid == "u":
            lower_txt = user_text.lower()
            if actuation_report and not actuation_report.get("executed", True):
                word = "Alert"
            elif "scan" in lower_txt or "who" in lower_txt or "hvem" in lower_txt or "look" in lower_txt or "camera" in lower_txt:
                word = "Scanning"
            elif "leak" in lower_txt or "water" in lower_txt or "vann" in lower_txt:
                word = "Leak"
            elif "alert" in lower_txt or "alarm" in lower_txt:
                word = "Alert"
            elif "log" in lower_txt:
                word = "Logged"
            else:
                word = "Confirmed"

            house_occ_summary = {}
            if house_snapshot:
                for rid, rinfo in house_snapshot.items():
                    occ_list = [o.get("display_name") for o in rinfo.get("occupants", [])]
                    if occ_list:
                        house_occ_summary[rid] = occ_list

            return PersonaResponse(
                persona_id="u",
                spoken_text=word,
                sound_cue="camera_shutter",
                ui_card={
                    "title": "U Sensory Archive",
                    "active_room": active_room_id,
                    "roamed_from": roamed_from_room,
                    "recognized_speaker": speaker.get("display_name"),
                    "speaker_role": speaker.get("role"),
                    "voice_matched": speaker.get("voice_matched"),
                    "face_matched": speaker.get("face_matched"),
                    "confidence": speaker.get("confidence"),
                    "room_occupants": [o.get("display_name") for o in occupants],
                    "house_cameras_occupants": house_occ_summary,
                },
                inter_agent_whisper={
                    "broadcast": "sensory_biometric_frame",
                    "speaker": speaker.get("person_id"),
                    "room_id": active_room_id,
                    "occupants": [o.get("person_id") for o in occupants],
                },
            )

        # Conversational personas: J.A.R.V.I.S., F.R.I.D.A.Y., E.D.I.T.H.
        sys_prompt = self.build_system_prompt(
            persona_id=pid,
            speaker_context=speaker,
            room_occupants=occupants,
            shared_facts=shared_facts,
            house_snapshot=house_snapshot,
            actuation_report=actuation_report,
        )
        llm_reply = self.llm_client.generate(
            system_prompt=sys_prompt,
            user_message=user_text,
            persona_id=pid,
            speaker_context=speaker,
        )

        if not llm_reply:
            llm_reply = self._synthesize_contextual_reply(
                persona_id=pid,
                user_text=user_text,
                speaker=speaker,
                occupants=occupants,
                has_children=has_children,
                shared_facts=shared_facts or [],
                house_snapshot=house_snapshot,
                actuation_report=actuation_report,
                roamed_from_room=roamed_from_room,
                active_room_id=active_room_id,
            )

        return PersonaResponse(
            persona_id=pid,
            spoken_text=llm_reply,
            sound_cue="soft_chime" if pid == "jarvis" and (speaker.get("role") == "visitor" or roamed_from_room) else None,
            ui_card={
                "title": f"{persona['display_name']} Active",
                "active_room": active_room_id,
                "roamed_from": roamed_from_room,
                "addressed_to": speaker.get("display_name"),
                "role": speaker.get("role"),
                "voice_verified": speaker.get("voice_matched", False),
                "face_verified": speaker.get("face_matched", False),
                "confidence": speaker.get("confidence", 0.0),
                "actuation": actuation_report,
            },
        )

    def _synthesize_contextual_reply(
        self,
        persona_id: str,
        user_text: str,
        speaker: dict[str, Any],
        occupants: list[dict[str, Any]],
        has_children: bool,
        shared_facts: list[dict[str, Any]],
        house_snapshot: dict[str, Any] | None = None,
        actuation_report: dict[str, Any] | None = None,
        roamed_from_room: str | None = None,
        active_room_id: str = "living_room",
    ) -> str:
        """Deterministic in-character synthesis grounded in recognized voice/face identity, roaming, and house state."""
        name = speaker.get("display_name", "Visitor")
        role = speaker.get("role", "visitor")
        v_ok = speaker.get("voice_matched", False)
        f_ok = speaker.get("face_matched", False)
        conf_pct = int(round(float(speaker.get("confidence", 0.6)) * 100))

        mod_str = (
            "voice and facial biometrics"
            if (v_ok and f_ok)
            else ("voiceprint" if v_ok else ("facial recognition" if f_ok else "guest profile"))
        )
        other_names = [
            o.get("display_name", "Guest")
            for o in occupants
            if o.get("person_id") != speaker.get("person_id")
        ]
        presence_clause = (
            f" I also see {', '.join(other_names)} present in the {active_room_id.replace('_', ' ')}."
            if other_names
            else ""
        )
        roam_clause = (
            f" I have transferred my presence from {roamed_from_room.replace('_', ' ')} to {active_room_id.replace('_', ' ')}."
            if roamed_from_room and roamed_from_room != active_room_id
            else ""
        )

        # House actuation feedback (doors, lights, climate)
        actuation_clause = ""
        if actuation_report:
            if not actuation_report.get("executed", False):
                actuation_clause = f" Note: Requested action blocked ({actuation_report.get('reason')})."
            elif "door" in actuation_report:
                d_info = actuation_report["door"]
                st = "locked" if d_info.get("is_locked") else "unlocked and opened"
                actuation_clause = f" I have {st} {d_info.get('door_id')} in {actuation_report.get('room_id')}."
            elif "lights_on" in actuation_report:
                st = "turned on" if actuation_report["lights_on"] else "turned off"
                actuation_clause = f" Lights in {actuation_report.get('room_id')} have been {st}."
            elif "temperature_c" in actuation_report:
                actuation_clause = f" Climate in {actuation_report.get('room_id')} set to {actuation_report['temperature_c']}°C."

        # House-wide camera summary if user asks about cameras or who is in the house
        camera_clause = ""
        lower_u = user_text.lower()
        if house_snapshot and any(w in lower_u for w in ("camera", "kamera", "house", "huset", "where is", "hvor er", "everybody", "alle")):
            sighted = []
            for rid, rinfo in house_snapshot.items():
                occ_names = [o.get("display_name") for o in rinfo.get("occupants", [])]
                if occ_names:
                    sighted.append(f"{', '.join(occ_names)} in {rinfo.get('name')} ({rid})")
            if sighted:
                camera_clause = f" House camera feeds show: {'; '.join(sighted)}."

        if persona_id == "jarvis":
            if role == "visitor":
                return (
                    f"Good day. I am J.A.R.V.I.S. You are currently registered as a guest ({conf_pct}% {mod_str})."
                    f"{roam_clause}{presence_clause}{actuation_clause}{camera_clause} "
                    f"How may I assist you within household guest parameters?"
                )
            honorific = "Sir" if speaker.get("person_id") == "eivind" else ("Ma'am" if speaker.get("person_id") == "wife" else name)
            return (
                f"At your service, {honorific}. Identity verified as {name} via {mod_str} ({conf_pct}% confidence)."
                f"{roam_clause}{presence_clause}{actuation_clause}{camera_clause} "
                f"All V.A.N.G.U.A.R.D. security perimeters and architecture nodes are nominal."
            )

        if persona_id == "friday":
            if role == "visitor":
                return (
                    f"Risk assessment: unverified visitor on the floor. Identity confidence {conf_pct}%."
                    f"{roam_clause}{presence_clause}{actuation_clause}{camera_clause} "
                    f"Recommendation: state your business or request authorization from Eivind."
                )
            return (
                f"Right here, {name}. Risk level: zero. Telemetry confirms your identity via {mod_str} at {conf_pct}% confidence."
                f"{roam_clause}{presence_clause}{actuation_clause}{camera_clause} "
                f"Recommendation: ready to execute your next tactical instruction."
            )

        if persona_id == "edith":
            if has_children:
                return (
                    f"Hey {name}! Recognized you right away ({conf_pct}% match via {mod_str})!"
                    f"{roam_clause}{presence_clause}{actuation_clause}{camera_clause} "
                    f"Child-safe creative mode is locked in and ready for fun simulations!"
                )
            return (
                f"Hey {name}! Biometrics locked in via {mod_str} at {conf_pct}%!"
                f"{roam_clause}{presence_clause}{actuation_clause}{camera_clause} "
                f"Simulation sandbox is spun up and ready—what are we inventing today?"
            )

        return f"Acknowledged, {name}.{roam_clause}{actuation_clause}"
