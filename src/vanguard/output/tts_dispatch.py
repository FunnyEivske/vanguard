"""Persona-Specific TTS Dispatcher for V.A.N.G.U.A.R.D.

Enforces vocal constraints, persona dialects, substrate mute rules, and inter-agent speech toggles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vanguard.output.sound_cues import SoundCueDispatcher
from vanguard.output.ui_cards import UICardEmitter


class VocalConstraintViolationError(Exception):
    """Raised when an utterance violates a persona's vocal tier constraints."""


class SubstrateVoiceViolationError(Exception):
    """Raised when the V.A.N.G.U.A.R.D. substrate attempts to emit voice."""


@dataclass
class DispatchedUtterance:
    """Dispatched speech utterance."""

    persona_id: str
    text: str
    room_id: str
    language: str
    accent: str
    cadence: str
    spoken_aloud: bool
    sound_cue: str | None
    ui_card_emitted: bool


class TTSDispatcher:
    """Manages audio output dispatch for all personas."""

    def __init__(
        self,
        personas_config: dict[str, Any],
        inter_agent_speech_aloud: bool = False,
        sound_dispatcher: SoundCueDispatcher | None = None,
        ui_card_emitter: UICardEmitter | None = None,
    ) -> None:
        self.personas_config = personas_config
        self.inter_agent_speech_aloud = inter_agent_speech_aloud
        self.sound_dispatcher = sound_dispatcher or SoundCueDispatcher()
        self.ui_card_emitter = ui_card_emitter or UICardEmitter()
        self.dispatched_history: list[DispatchedUtterance] = []

    def set_inter_agent_speech_aloud(self, enabled: bool) -> None:
        """Update toggle for inter-agent physical vocalization."""
        self.inter_agent_speech_aloud = enabled

    def dispatch(
        self,
        persona_id: str,
        text: str | None,
        room_id: str = "living_room",
        language: str = "nb-NO",
        is_inter_agent: bool = False,
        sound_cue: str | None = None,
        ui_card_content: dict[str, Any] | None = None,
    ) -> DispatchedUtterance:
        """Dispatch speech utterance through persona acoustic profile."""
        pid = persona_id.lower().strip()

        # Rule 1: V.A.N.G.U.A.R.D. substrate has NO voice and cannot speak
        if pid in ("vanguard", "v.a.n.g.u.a.r.d.", "substrate", "system", "the house", "the network"):
            raise SubstrateVoiceViolationError(
                "V.A.N.G.U.A.R.D. is the neutral operating substrate and cannot emit voice."
            )

        config = self.personas_config.get(pid)
        if not config:
            raise ValueError(f"Unknown persona ID: {persona_id}")

        accent = config.get("accent", "neutral")
        cadence = config.get("speech_rhythm", "standard")

        # Handle UI card emission if provided or required by persona
        card_emitted = False
        if ui_card_content:
            self.ui_card_emitter.emit_card(
                source_persona=pid,
                card_type=config.get("ui_card_type", "status"),
                title=f"{config.get('display_name', pid)} Status",
                content=ui_card_content,
                room_id=room_id,
            )
            card_emitted = True

        # Handle hardware sound cues
        if sound_cue:
            self.sound_dispatcher.trigger_cue(cue_id=sound_cue, source_persona=pid, room_id=room_id)

        # If text is None or empty (e.g. D.U.M.-E responding with servo whir only)
        if not text or not text.strip():
            utterance = DispatchedUtterance(
                persona_id=pid,
                text="",
                room_id=room_id,
                language=language,
                accent=accent,
                cadence=cadence,
                spoken_aloud=False,
                sound_cue=sound_cue,
                ui_card_emitted=card_emitted,
            )
            self.dispatched_history.append(utterance)
            return utterance

        words = text.strip().split()
        word_count = len(words)

        # Rule 2: Enforce vocal tier word limits
        # D.U.M.-E: 1-3 words max to humans
        if pid == "dume" and not is_inter_agent:
            if word_count > 3:
                raise VocalConstraintViolationError(
                    f"D.U.M.-E vocal limit exceeded: expected 1-3 words, got {word_count} ('{text}')"
                )

        # U: exactly 1 word max to humans
        if pid == "u" and not is_inter_agent:
            if word_count > 1:
                raise VocalConstraintViolationError(
                    f"U vocal limit exceeded: expected 1 word, got {word_count} ('{text}')"
                )

        # Rule 3: Inter-agent vocalization toggle
        # If inter-agent and inter_agent_speech_aloud is false, utterance is NOT spoken aloud in the room
        spoken_aloud = True
        if is_inter_agent and not self.inter_agent_speech_aloud:
            spoken_aloud = False

        utterance = DispatchedUtterance(
            persona_id=pid,
            text=text,
            room_id=room_id,
            language=language,
            accent=accent,
            cadence=cadence,
            spoken_aloud=spoken_aloud,
            sound_cue=sound_cue,
            ui_card_emitted=card_emitted,
        )
        self.dispatched_history.append(utterance)
        return utterance
