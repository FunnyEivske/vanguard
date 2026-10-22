"""Voice Identification and Speaker Resolution for V.A.N.G.U.A.R.D."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VoiceIdResult:
    """Result of acoustic voice print matching."""

    speaker_id: str
    is_known_family_member: bool
    confidence: float
    preferred_language: str
    role: str


class VoiceIdentifier:
    """Matches acoustic signatures to registered family members or visitors."""

    def __init__(self, people_profiles: dict[str, dict]) -> None:
        self.profiles = people_profiles

    def identify(self, voice_print_id: str | None = None, claimed_name: str | None = None) -> VoiceIdResult:
        """Resolve a speaker identity based on acoustic print or claimed name."""
        if claimed_name:
            norm_name = claimed_name.lower().strip()
            for pid, profile in self.profiles.items():
                if norm_name == pid.lower() or norm_name == profile.get("name", "").lower():
                    langs = profile.get("preferred_languages", ["nb-NO"])
                    return VoiceIdResult(
                        speaker_id=pid,
                        is_known_family_member=profile.get("role") != "visitor",
                        confidence=0.98,
                        preferred_language=langs[0] if langs else "nb-NO",
                        role=profile.get("role", "family_member"),
                    )

        if voice_print_id:
            for pid, profile in self.profiles.items():
                if profile.get("voice_print_id") == voice_print_id:
                    langs = profile.get("preferred_languages", ["nb-NO"])
                    return VoiceIdResult(
                        speaker_id=pid,
                        is_known_family_member=profile.get("role") != "visitor",
                        confidence=0.95,
                        preferred_language=langs[0] if langs else "nb-NO",
                        role=profile.get("role", "family_member"),
                    )

        # Fallback to visitor
        return VoiceIdResult(
            speaker_id="visitor",
            is_known_family_member=False,
            confidence=0.60,
            preferred_language="nb-NO",
            role="visitor",
        )
