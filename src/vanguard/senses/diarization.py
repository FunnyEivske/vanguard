"""Streaming Diarization, VAD Segmentation, and Language Detection for V.A.N.G.U.A.R.D.

Segments continuous audio into discrete speaker turns across household rooms, detects
speaker changes via acoustic voiceprint embeddings, and identifies spoken language
(Norwegian `nb-NO` vs English `en-US`).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from vanguard.senses.biometric_store import cosine_similarity
from vanguard.senses.voice_id import VoiceIdentifier, extract_voice_embedding_from_pcm

NORWEGIAN_MARKERS = re.compile(
    r"(\b(hei|hva|hvem|hvor|hvordan|sjekk|status|stue|kjøkken|verksted|kontor|vann|varmt|barna|takk|er|det|på|i|og|ikke|for)\b|[æøåÆØÅ])",
    re.IGNORECASE,
)


def detect_spoken_language(text: str, default_language: str = "nb-NO") -> str:
    """Detect whether an utterance is Norwegian (`nb-NO`) or English (`en-US`)."""
    if not text or not text.strip():
        return default_language
    if NORWEGIAN_MARKERS.search(text):
        return "nb-NO"
    ascii_words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    english_markers = {
        "the", "is", "are", "what", "who", "where", "check", "report",
        "scan", "status", "perimeter", "room", "please", "stop", "hold",
        "good", "hello", "hey", "let", "build", "run", "sim", "simulation",
    }
    if any(w in english_markers for w in ascii_words):
        return "en-US"
    return default_language


@dataclass
class DiarizedTurn:
    """An acoustic speech turn identified by diarization."""

    speaker_id: str
    room_id: str
    start_time: float
    end_time: float
    text: str
    language: str
    confidence: float
    voice_embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingDiarizer:
    """Segments continuous audio into discrete speaker turns across household rooms."""

    def __init__(
        self,
        default_room: str = "living_room",
        voice_identifier: VoiceIdentifier | None = None,
        energy_threshold: float = 0.008,
    ) -> None:
        self.default_room = default_room
        self.voice_identifier = voice_identifier
        self.energy_threshold = energy_threshold
        self._active_turns: dict[str, DiarizedTurn] = {}
        self._turn_history: list[DiarizedTurn] = []

    def process_utterance(
        self,
        speaker_id: str | None = None,
        text: str = "",
        room_id: str | None = None,
        language: str | None = None,
        confidence: float = 0.95,
        audio_samples: list[float] | bytes | None = None,
        sample_rate: int = 16000,
    ) -> DiarizedTurn:
        """Create a completed speaker turn from processed stream, resolving speaker & language if needed."""
        target_room = room_id or self.default_room
        now = time.time()

        emb: list[float] | None = None
        resolved_speaker = speaker_id or "visitor"
        resolved_conf = confidence
        preferred_lang = "nb-NO"

        if audio_samples is not None:
            emb = extract_voice_embedding_from_pcm(audio_samples, sample_rate=sample_rate)
            if self.voice_identifier is not None and not speaker_id:
                vid = self.voice_identifier.identify(voice_embedding=emb)
                resolved_speaker = vid.speaker_id
                resolved_conf = vid.confidence
                preferred_lang = vid.preferred_language

        resolved_lang = language or detect_spoken_language(text, default_language=preferred_lang)

        turn = DiarizedTurn(
            speaker_id=resolved_speaker,
            room_id=target_room,
            start_time=now - 1.0,
            end_time=now,
            text=text,
            language=resolved_lang,
            confidence=resolved_conf,
            voice_embedding=emb,
        )
        self._active_turns[target_room] = turn
        self._turn_history.append(turn)
        return turn

    def segment_multi_speaker_stream(
        self,
        audio_samples: list[float],
        sample_rate: int = 16000,
        room_id: str | None = None,
        window_seconds: float = 0.5,
        speaker_change_threshold: float = 0.78,
    ) -> list[DiarizedTurn]:
        """Segment a continuous audio buffer into speaker-homogeneous turns via VAD & voice embeddings."""
        target_room = room_id or self.default_room
        window_len = max(256, int(window_seconds * sample_rate))
        if not audio_samples:
            return []

        segments: list[tuple[float, float, list[float], list[float]]] = []
        now = time.time()
        total_duration = len(audio_samples) / sample_rate
        stream_start = now - total_duration

        idx = 0
        while idx < len(audio_samples):
            chunk = audio_samples[idx : idx + window_len]
            if len(chunk) < sample_rate // 10:
                break
            rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
            t_start = stream_start + (idx / sample_rate)
            t_end = stream_start + (min(len(audio_samples), idx + window_len) / sample_rate)
            idx += window_len

            if rms < self.energy_threshold:
                continue

            emb = extract_voice_embedding_from_pcm(chunk, sample_rate=sample_rate)
            if segments:
                prev_start, _, prev_samples, prev_emb = segments[-1]
                # Merge adjacent windows if same speaker embedding
                if cosine_similarity(prev_emb, emb) >= speaker_change_threshold:
                    merged_samples = prev_samples + chunk
                    merged_emb = extract_voice_embedding_from_pcm(merged_samples, sample_rate=sample_rate)
                    segments[-1] = (prev_start, t_end, merged_samples, merged_emb)
                    continue

            segments.append((t_start, t_end, list(chunk), emb))

        turns: list[DiarizedTurn] = []
        for s_start, s_end, _s_samples, s_emb in segments:
            spk_id = "visitor"
            conf = 0.60
            lang = "nb-NO"
            if self.voice_identifier is not None:
                res = self.voice_identifier.identify(voice_embedding=s_emb)
                spk_id = res.speaker_id
                conf = res.confidence
                lang = res.preferred_language

            turn = DiarizedTurn(
                speaker_id=spk_id,
                room_id=target_room,
                start_time=s_start,
                end_time=s_end,
                text="",
                language=lang,
                confidence=conf,
                voice_embedding=s_emb,
            )
            self._active_turns[target_room] = turn
            self._turn_history.append(turn)
            turns.append(turn)

        return turns

    def get_latest_turn(self, room_id: str) -> DiarizedTurn | None:
        """Get the most recent turn in a room."""
        return self._active_turns.get(room_id)

    def get_history(self, room_id: str | None = None) -> list[DiarizedTurn]:
        """Return diarized turn history, optionally filtered by room."""
        if room_id:
            return [t for t in self._turn_history if t.room_id == room_id]
        return list(self._turn_history)
