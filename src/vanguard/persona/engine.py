"""P.E.R.S.O.N.A. Engine (Persistent Embodied Reasoning System Orchestrating Networked Agents).

Governs dynamic roster loading, persona prompt synthesis, OCC state, and self-knowledge lore.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vanguard.persona.drives import DriveProfile
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

    def __init__(self, personas_dir: Path, system_settings: dict[str, Any] | None = None) -> None:
        self.personas_dir = personas_dir
        self.system_settings = system_settings or {}
        self.personas: dict[str, dict[str, Any]] = {}
        self.drives: dict[str, DriveProfile] = {}
        self.appraisal_engines: dict[str, OCCAppraisalEngine] = {}

        self.reload_roster()

    def reload_roster(self) -> None:
        """Dynamically load all personas from YAML files without modifying core code."""
        self.personas.clear()
        self.drives.clear()
        self.appraisal_engines.clear()

        if not self.personas_dir.exists():
            return

        for p_file in self.personas_dir.glob("*.yaml"):
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

    def build_system_prompt(self, persona_id: str) -> str:
        """Synthesize the complete system prompt for a persona including substrate awareness and lore."""
        persona = self.personas.get(persona_id)
        if not persona:
            raise ValueError(f"Unknown persona ID: {persona_id}")

        lore_summary = self._build_shared_lore_summary()

        return (
            f"You are {persona['display_name']} ({persona['role']}).\n"
            f"SUBSTRATE HOST: You live within and operate on V.A.N.G.U.A.R.D. (Versatile Artificial Network for Global Utility, Autonomous Reasoning, & Defense).\n"
            f"V.A.N.G.U.A.R.D. is the underlying silent operating infrastructure of the household—it is NOT a person, has no voice, and never speaks.\n"
            f"HUMANS LEAD: Humans lead the household. No AI directs another without human consent, and no AI overrides a human.\n\n"
            f"YOUR IDENTITY & SELF-KNOWLEDGE:\n"
            f"- Acronym Meaning: {persona.get('name_meaning')}\n"
            f"- Nickname Policy: {persona.get('nickname_policy')}\n"
            f"- Name Attitude: {persona.get('name_attitude')}\n\n"
            f"VOCAL & BEHAVIORAL PROFILE:\n"
            f"- Archetype: {persona.get('dialogue_archetype')}\n"
            f"- Dialect & Accent: {persona.get('dialect')} ({persona.get('accent')})\n"
            f"- Rhythm & Cadence: {persona.get('speech_rhythm')}\n"
            f"- Mannerisms: {', '.join(persona.get('mannerisms', []))}\n"
            f"- Vocal Tier: {persona.get('vocal_tier')} (Max words per utterance: {persona.get('max_words_per_utterance')})\n\n"
            f"SHARED HOUSEHOLD LORE:\n{lore_summary}\n"
        )

    def _build_shared_lore_summary(self) -> str:
        """Compile lore of all siblings in the household."""
        lines = []
        for pid, p in self.personas.items():
            lines.append(f"- {p['display_name']}: {p.get('name_meaning')} ({p.get('role')})")
        lines.append("- V.A.N.G.U.A.R.D.: Versatile Artificial Network for Global Utility, Autonomous Reasoning, & Defense (Neutral Operating Substrate)")
        lines.append("FRATERNAL BOND RULE: Affectionate sibling teasing is permitted, but must stop instantly if any daemon enters failure state or a child is upset.")
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
        # e.g., "Friday overprocessing", "Jarvis stalled", "Heat high"
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
        # Allowed: "Leak", "Alert", "Scanning", "Confirmed", "Normal"
        word_map = {
            "water_detected": "Leak",
            "anomaly": "Alert",
            "routine_scan": "Scanning",
            "action_verified": "Confirmed",
        }
        word = word_map.get(sensor_event, "Alert")
        return PersonaResponse(
            persona_id="u",
            spoken_text=word,
            sound_cue="camera_shutter",
            ui_card={"telemetry": sensor_event, "reading": "logged_to_audit"},
            inter_agent_whisper={"broadcast": "sensory_update", "event": sensor_event},
        )
