"""Persona-Specific TTS Dispatcher for V.A.N.G.U.A.R.D.

Enforces vocal constraints, persona dialects, substrate mute rules, and inter-agent speech toggles.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vanguard.output.sound_cues import SoundCueDispatcher, play_wav_linux
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
        enable_hardware_playback: bool = False,
        persona_voices: dict[str, str] | None = None,
    ) -> None:
        self.personas_config = personas_config
        self.inter_agent_speech_aloud = inter_agent_speech_aloud
        self.enable_hardware_playback = enable_hardware_playback
        self.sound_dispatcher = sound_dispatcher or SoundCueDispatcher(
            enable_hardware_playback=enable_hardware_playback
        )
        self.ui_card_emitter = ui_card_emitter or UICardEmitter()
        self.persona_voices = persona_voices or {
            "jarvis": {
                "online_voice": "onyx",
                "elevenlabs_voice_id": "jarvis_rp_british",
                "piper_model": "en_GB-alan-medium",
                "espeak_voice": "en-gb",
                "macos_voice": "Daniel",
            },
            "friday": {
                "online_voice": "nova",
                "elevenlabs_voice_id": "friday_scottish",
                "piper_model": "en_GB-alba-medium",
                "espeak_voice": "en-sc",
                "macos_voice": "Fiona",
            },
            "edith": {
                "online_voice": "shimmer",
                "elevenlabs_voice_id": "edith_transatlantic",
                "piper_model": "en_US-amy-medium",
                "espeak_voice": "en-us",
                "macos_voice": "Samantha",
            },
            "dume": {
                "online_voice": "echo",
                "elevenlabs_voice_id": "dume_monotone",
                "piper_model": "en_US-Ryan-low",
                "espeak_voice": "en+m1",
                "macos_voice": "Fred",
            },
            "u": {
                "online_voice": "alloy",
                "elevenlabs_voice_id": "u_binary_unit",
                "piper_model": "en_GB-jenny_dioco-medium",
                "espeak_voice": "en+f2",
                "macos_voice": "Victoria",
            },
        }
        self.dispatched_history: list[DispatchedUtterance] = []

    def _speak_hardware(self, persona_id: str, text: str) -> None:
        """Synthesize speech: Online-First (ElevenLabs / OpenAI TTS), Local Backup (Linux Piper/espeak-ng or macOS say)."""
        import sys

        v_cfg = self.persona_voices.get(persona_id, {})
        if isinstance(v_cfg, str):
            v_cfg = {"online_voice": v_cfg, "espeak_voice": "en", "macos_voice": v_cfg}

        default_mac_voices = {
            "jarvis": "Daniel",
            "friday": "Fiona",
            "edith": "Samantha",
            "dume": "Fred",
            "u": "Victoria",
        }

        tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix=f"vanguard_tts_{persona_id}_"))
        try:
            # 1. ONLINE PRIMARY: ElevenLabs API or OpenAI Audio Speech API
            if os.environ.get("ELEVENLABS_API_KEY"):
                el_voice = v_cfg.get("elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{el_voice}?output_format=pcm_24000"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=6.0) as resp:
                        tmp_wav.write_bytes(resp.read())
                        if play_wav_linux(tmp_wav):
                            return
                except Exception:
                    pass

            if os.environ.get("OPENAI_API_KEY"):
                oa_voice = v_cfg.get("online_voice", "onyx")
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                req = urllib.request.Request(
                    f"{base_url}/audio/speech",
                    data=json.dumps(
                        {
                            "model": "tts-1",
                            "voice": oa_voice,
                            "input": text,
                            "response_format": "wav",
                        }
                    ).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=6.0) as resp:
                        tmp_wav.write_bytes(resp.read())
                        if play_wav_linux(tmp_wav):
                            return
                except Exception:
                    pass

            # 2. LOCAL BACKUP (Linux Piper / espeak-ng, or macOS say for local Mac testing)
            if sys.platform == "darwin":
                mac_voice = v_cfg.get("macos_voice") or default_mac_voices.get(persona_id, "Daniel")
                res = subprocess.run(
                    ["/usr/bin/say", "-v", mac_voice, text],
                    check=False,
                    capture_output=True,
                    timeout=30.0,
                )
                if res.returncode != 0:
                    # Fallback if specific accent voice isn't downloaded on this Mac
                    subprocess.run(["/usr/bin/say", text], check=False, capture_output=True, timeout=30.0)
                return

            piper_model = v_cfg.get("piper_model", "en_GB-alan-medium")
            try:
                proc = subprocess.run(
                    ["piper", "--model", piper_model, "--output_file", str(tmp_wav)],
                    input=text.encode("utf-8"),
                    check=False,
                    capture_output=True,
                    timeout=10.0,
                )
                if proc.returncode == 0 and tmp_wav.exists():
                    if play_wav_linux(tmp_wav):
                        return
            except Exception:
                pass

            espeak_voice = v_cfg.get("espeak_voice", "en")
            try:
                subprocess.run(
                    ["espeak-ng", "-v", espeak_voice, text],
                    check=False,
                    capture_output=True,
                    timeout=15.0,
                )
            except Exception:
                pass
        finally:
            tmp_wav.unlink(missing_ok=True)

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

        if spoken_aloud and self.enable_hardware_playback:
            self._speak_hardware(pid, text)

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
