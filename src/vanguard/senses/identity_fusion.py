"""Multimodal Identity Fusion & Room Presence Tracker for V.A.N.G.U.A.R.D.

Fuses acoustic Voice ID, visual Face ID, spoken language detection, and room affinity
to resolve speaker identity, tracks active occupants and cross-room movement trajectories,
detects when children or visitors are present, and broadcasts sensory presence updates
through U (Sensory Daemon).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event, EventCategory
from vanguard.senses.diarization import detect_spoken_language
from vanguard.senses.face_id import FaceIdentifier, FaceIdResult
from vanguard.senses.voice_id import VoiceIdentifier, VoiceIdResult


@dataclass
class ResolvedIdentity:
    """Unified person identity resolved from voice, face, language, and room context."""

    person_id: str
    display_name: str
    role: str
    is_known_family_member: bool
    confidence: float
    preferred_language: str
    detected_language: str = "nb-NO"
    voice_matched: bool = False
    face_matched: bool = False
    voice_confidence: float = 0.0
    face_confidence: float = 0.0
    room_id: str = "living_room"
    previous_room_id: str | None = None
    permissions: dict[str, Any] = field(default_factory=dict)
    confidante_persona: str | None = None
    timestamp: float = field(default_factory=time.time)


class IdentityFusionEngine:
    """Combines VoiceIdentifier and FaceIdentifier with cross-room occupant tracking."""

    def __init__(
        self,
        voice_identifier: VoiceIdentifier,
        face_identifier: FaceIdentifier,
        event_bus: EventBus | None = None,
        presence_ttl_seconds: float = 120.0,
    ) -> None:
        self.voice_id = voice_identifier
        self.face_id = face_identifier
        self.event_bus = event_bus
        self.presence_ttl_seconds = presence_ttl_seconds
        # room_id -> dict[person_id, ResolvedIdentity]
        self._room_occupants: dict[str, dict[str, ResolvedIdentity]] = {}
        # person_id -> last known room_id
        self._person_locations: dict[str, str] = {}

    def _record_person_in_room(self, resolved: ResolvedIdentity) -> str | None:
        """Record a person in `resolved.room_id` and vacate any previous room if they moved."""
        pid = resolved.person_id
        room_id = resolved.room_id
        prev_room: str | None = None

        if pid != "visitor":
            prev_room = self._person_locations.get(pid)
            if prev_room and prev_room != room_id:
                if prev_room in self._room_occupants:
                    self._room_occupants[prev_room].pop(pid, None)
            self._person_locations[pid] = room_id

        resolved.previous_room_id = prev_room
        if room_id not in self._room_occupants:
            self._room_occupants[room_id] = {}
        self._room_occupants[room_id][pid] = resolved
        return prev_room

    def get_person_location(self, person_id: str) -> str | None:
        """Return the room where a known person was most recently identified."""
        pid = person_id.lower().strip()
        room_id = self._person_locations.get(pid)
        if not room_id:
            return None
        occupants = self.get_room_occupants(room_id)
        if any(o.person_id == pid for o in occupants):
            return room_id
        return None

    def _build_resolved(
        self,
        person_id: str,
        room_id: str,
        voice_res: VoiceIdResult | None = None,
        face_res: FaceIdResult | None = None,
        spoken_text: str | None = None,
    ) -> ResolvedIdentity:
        profile = self.voice_id.profiles.get(person_id) or self.face_id.profiles.get(person_id) or {}
        langs = profile.get("preferred_languages", ["nb-NO"])
        pref_lang = langs[0] if langs else "nb-NO"
        det_lang = detect_spoken_language(spoken_text or "", default_language=pref_lang)

        role = profile.get(
            "role",
            (voice_res.role if voice_res else (face_res.role if face_res else "visitor")),
        )
        display_name = profile.get(
            "name",
            (voice_res.display_name if voice_res else (face_res.display_name if face_res else person_id.capitalize())),
        )

        v_matched = bool(voice_res and voice_res.speaker_id == person_id and person_id != "visitor")
        f_matched = bool(face_res and face_res.person_id == person_id and person_id != "visitor")
        v_conf = voice_res.confidence if voice_res and voice_res.speaker_id == person_id else 0.0
        f_conf = face_res.confidence if face_res and face_res.person_id == person_id else 0.0

        if v_matched and f_matched:
            combined_conf = min(0.999, (0.55 * v_conf + 0.45 * f_conf) * 1.08)
        elif v_matched:
            combined_conf = v_conf
        elif f_matched:
            combined_conf = f_conf
        else:
            combined_conf = max(v_conf, f_conf, 0.55)

        return ResolvedIdentity(
            person_id=person_id,
            display_name=display_name,
            role=role,
            is_known_family_member=role != "visitor",
            confidence=round(float(combined_conf), 4),
            preferred_language=pref_lang,
            detected_language=det_lang,
            voice_matched=v_matched,
            face_matched=f_matched,
            voice_confidence=round(float(v_conf), 4),
            face_confidence=round(float(f_conf), 4),
            room_id=room_id,
            permissions=dict(profile.get("permissions", {})),
            confidante_persona=profile.get("confidante_persona"),
        )

    async def resolve_identity(
        self,
        room_id: str = "living_room",
        spoken_text: str | None = None,
        audio_samples: list[float] | bytes | None = None,
        wav_path: str | None = None,
        voice_embedding: list[float] | None = None,
        voice_print_id: str | None = None,
        claimed_name: str | None = None,
        face_image: Any | None = None,
        face_image_path: str | None = None,
        face_embedding: list[float] | None = None,
        face_print_id: str | None = None,
        sample_rate: int = 16000,
    ) -> ResolvedIdentity:
        """Fuse voice, face, language, and room presence biometrics to identify the active person."""
        has_voice_input = any(
            x is not None
            for x in (audio_samples, wav_path, voice_embedding, voice_print_id, claimed_name)
        )
        has_face_input = any(
            x is not None
            for x in (face_image, face_image_path, face_embedding, face_print_id)
        )

        voice_res: VoiceIdResult | None = None
        face_res: FaceIdResult | None = None

        if has_voice_input:
            voice_res = self.voice_id.identify(
                voice_print_id=voice_print_id,
                claimed_name=claimed_name,
                audio_samples=audio_samples,
                wav_path=wav_path,
                voice_embedding=voice_embedding,
                sample_rate=sample_rate,
            )

        if has_face_input:
            face_res = self.face_id.identify(
                face_print_id=face_print_id,
                image=face_image,
                image_path=face_image_path,
                face_embedding=face_embedding,
            )

        person_id = "visitor"
        if voice_res and face_res:
            if voice_res.speaker_id == face_res.person_id:
                person_id = voice_res.speaker_id
            elif voice_res.speaker_id != "visitor" and face_res.person_id == "visitor":
                person_id = voice_res.speaker_id
            elif face_res.person_id != "visitor" and voice_res.speaker_id == "visitor":
                person_id = face_res.person_id
            else:
                person_id = (
                    voice_res.speaker_id
                    if voice_res.confidence >= face_res.confidence
                    else face_res.person_id
                )
        elif voice_res:
            person_id = voice_res.speaker_id
            existing = self.get_room_occupants(room_id)
            for occ in existing:
                if occ.person_id == person_id and occ.face_matched:
                    face_res = FaceIdResult(
                        person_id=person_id,
                        display_name=occ.display_name,
                        is_known_family_member=occ.is_known_family_member,
                        confidence=occ.face_confidence,
                        preferred_language=occ.preferred_language,
                        role=occ.role,
                    )
                    break
        elif face_res:
            person_id = face_res.person_id
        else:
            # If neither voice nor face was passed on this turn, check if there is a single verified occupant in the room
            existing = [o for o in self.get_room_occupants(room_id) if o.person_id != "visitor"]
            if len(existing) == 1:
                single = existing[0]
                person_id = single.person_id
                if single.voice_matched:
                    voice_res = VoiceIdResult(
                        speaker_id=single.person_id,
                        is_known_family_member=single.is_known_family_member,
                        confidence=single.voice_confidence,
                        preferred_language=single.preferred_language,
                        role=single.role,
                        display_name=single.display_name,
                    )
                if single.face_matched:
                    face_res = FaceIdResult(
                        person_id=single.person_id,
                        display_name=single.display_name,
                        is_known_family_member=single.is_known_family_member,
                        confidence=single.face_confidence,
                        preferred_language=single.preferred_language,
                        role=single.role,
                    )

        resolved = self._build_resolved(
            person_id=person_id,
            room_id=room_id,
            voice_res=voice_res,
            face_res=face_res,
            spoken_text=spoken_text,
        )
        prev_room = self._record_person_in_room(resolved)

        if self.event_bus is not None:
            await self.event_bus.publish(
                Event(
                    category=EventCategory.SENSOR,
                    event_type="identity_resolved",
                    source="u",
                    room_id=room_id,
                    payload={
                        "person_id": resolved.person_id,
                        "display_name": resolved.display_name,
                        "role": resolved.role,
                        "confidence": resolved.confidence,
                        "voice_matched": resolved.voice_matched,
                        "face_matched": resolved.face_matched,
                        "preferred_language": resolved.preferred_language,
                        "detected_language": resolved.detected_language,
                        "previous_room_id": prev_room,
                    },
                )
            )

        return resolved

    async def update_room_visual_presence(
        self,
        room_id: str,
        frame: Any | None = None,
        face_crops: list[tuple[Any, Any]] | None = None,
        face_embeddings: list[list[float]] | None = None,
    ) -> list[ResolvedIdentity]:
        """Update everyone visually present in a room from a camera frame."""
        face_results = self.face_id.identify_faces_in_frame(
            frame=frame,
            face_crops=face_crops,
            face_embeddings=face_embeddings,
        )
        resolved_list: list[ResolvedIdentity] = []
        if room_id not in self._room_occupants:
            self._room_occupants[room_id] = {}

        for fres in face_results:
            existing = self._room_occupants[room_id].get(fres.person_id)
            v_res = None
            if existing and existing.voice_matched and (time.time() - existing.timestamp) < self.presence_ttl_seconds:
                v_res = VoiceIdResult(
                    speaker_id=existing.person_id,
                    is_known_family_member=existing.is_known_family_member,
                    confidence=existing.voice_confidence,
                    preferred_language=existing.preferred_language,
                    role=existing.role,
                    display_name=existing.display_name,
                )
            resolved = self._build_resolved(
                person_id=fres.person_id,
                room_id=room_id,
                voice_res=v_res,
                face_res=fres,
            )
            self._record_person_in_room(resolved)
            resolved_list.append(resolved)

        return resolved_list

    def get_room_occupants(self, room_id: str) -> list[ResolvedIdentity]:
        """Return all active people detected in a room within the presence TTL window."""
        now = time.time()
        room_map = self._room_occupants.get(room_id, {})
        active: list[ResolvedIdentity] = []
        expired_keys: list[str] = []
        for pid, ident in room_map.items():
            if (now - ident.timestamp) <= self.presence_ttl_seconds:
                active.append(ident)
            else:
                expired_keys.append(pid)
        for k in expired_keys:
            room_map.pop(k, None)
        return active

    def room_has_children(self, room_id: str) -> bool:
        """Check if any child is currently detected in the specified room."""
        return any(occ.role == "child" for occ in self.get_room_occupants(room_id))

    def room_has_visitors(self, room_id: str) -> bool:
        """Check if any visitor is currently detected in the specified room."""
        return any(occ.role == "visitor" for occ in self.get_room_occupants(room_id))
