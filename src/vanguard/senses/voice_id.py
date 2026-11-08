"""Voice Identification and Speaker Biometric Resolution for V.A.N.G.U.A.R.D.

Extracts 64-dimensional acoustic voiceprint embeddings from raw PCM/WAV audio
(spectral sub-band energies, DCT cepstral coefficients, pitch/fundamental frequency
contours, spectral centroid, and zero-crossing statistics) and matches speakers
against enrolled family voiceprints in BiometricStore or profile definitions.
"""

from __future__ import annotations

import array
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vanguard.senses.biometric_store import (
    BiometricStore,
    load_people_profiles,
    normalize_vector,
)

VOICE_EMBEDDING_DIM = 64


@dataclass
class VoiceIdResult:
    """Result of acoustic voice print matching."""

    speaker_id: str
    is_known_family_member: bool
    confidence: float
    preferred_language: str
    role: str
    display_name: str = "Visitor"
    matched_print_id: str | None = None


def extract_voice_embedding_from_pcm(
    samples: list[float] | bytes,
    sample_rate: int = 16000,
) -> list[float]:
    """Compute a 64-dimensional normalized acoustic voiceprint embedding from mono audio samples.

    Combines frame-level log-spectral sub-band energies, DCT cepstral coefficients (MFCC-like),
    autocorrelation pitch periodicity, zero-crossing rate, and temporal delta statistics.
    """
    if isinstance(samples, (bytes, bytearray)):
        pcm16 = array.array("h")
        usable = len(samples) - (len(samples) % 2)
        pcm16.frombytes(samples[:usable])
        float_samples = [s / 32768.0 for s in pcm16]
    else:
        float_samples = [float(s) for s in samples]

    if not float_samples:
        return [0.0] * VOICE_EMBEDDING_DIM

    # Remove DC offset and pre-emphasize high frequencies
    mean_val = sum(float_samples) / len(float_samples)
    centered = [x - mean_val for x in float_samples]
    emphasized = [centered[0]] + [
        centered[i] - 0.97 * centered[i - 1] for i in range(1, len(centered))
    ]

    frame_size = max(128, int(sample_rate * 0.025))  # 25ms window
    hop_size = max(64, int(sample_rate * 0.010))     # 10ms hop
    num_bands = 24
    num_cepstra = 16

    # Precompute Hamming window
    window = [0.54 - 0.46 * math.cos(2.0 * math.pi * n / (frame_size - 1)) for n in range(frame_size)]

    # Precompute DCT-II basis for cepstral projection
    dct_basis = [
        [
            math.cos(math.pi * k * (2 * m + 1) / (2 * num_bands))
            for m in range(num_bands)
        ]
        for k in range(num_cepstra)
    ]

    cepstral_frames: list[list[float]] = []
    band_means = [0.0] * num_bands
    zcr_values: list[float] = []
    rms_values: list[float] = []
    pitch_lags: list[float] = []

    idx = 0
    n_total = len(emphasized)
    while idx + frame_size <= n_total and len(cepstral_frames) < 150:
        raw_frame = emphasized[idx : idx + frame_size]
        idx += hop_size

        energy = sum(x * x for x in raw_frame) / frame_size
        rms = math.sqrt(energy + 1e-12)
        if rms < 0.002:
            # Skip silent frames for speaker voiceprint stability
            continue

        rms_values.append(rms)
        frame = [raw_frame[i] * window[i] for i in range(frame_size)]

        # Zero-crossing rate
        zcr = sum(
            1 for i in range(1, frame_size) if (frame[i] >= 0) != (frame[i - 1] >= 0)
        ) / frame_size
        zcr_values.append(zcr)

        # Pitch / fundamental periodicity estimate via normalized autocorrelation in vocal range (75Hz - 400Hz)
        min_lag = max(1, int(sample_rate / 400))
        max_lag = min(frame_size // 2, int(sample_rate / 75))
        best_corr = 0.0
        best_lag = min_lag
        denom0 = sum(x * x for x in frame) + 1e-9
        for lag in range(min_lag, max_lag + 1, 2):
            corr = sum(frame[i] * frame[i + lag] for i in range(frame_size - lag)) / denom0
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        f0_norm = (sample_rate / best_lag) / 500.0 if best_corr > 0.25 else 0.0
        pitch_lags.append(f0_norm)

        # Compute sub-band log spectral energies across 24 Mel-spaced filterbanks (80Hz - 4000Hz)
        mel_low = 2595.0 * math.log10(1.0 + 80.0 / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + min(4000.0, sample_rate * 0.45) / 700.0)
        raw_powers: list[float] = []
        for b in range(num_bands):
            # Sample 3 frequencies inside each Mel sub-band to capture formant bandwidth
            band_power = 0.0
            for sub_offset in (0.2, 0.5, 0.8):
                mel_f = mel_low + (mel_high - mel_low) * (b + sub_offset) / num_bands
                freq_hz = 700.0 * (10.0 ** (mel_f / 2595.0) - 1.0)
                omega = 2.0 * math.pi * freq_hz / sample_rate
                coeff = 2.0 * math.cos(omega)
                s1 = 0.0
                s2 = 0.0
                for sample in frame:
                    s0 = sample + coeff * s1 - s2
                    s2 = s1
                    s1 = s0
                band_power += max(0.0, s1 * s1 + s2 * s2 - coeff * s1 * s2)
            raw_powers.append(band_power / 3.0)

        max_p = max(raw_powers) + 1e-9
        log_bands: list[float] = []
        for b, p in enumerate(raw_powers):
            # Dynamic-range bounded log energy (-6.0 to 0.0) so silent bins don't dominate cosine similarity
            rel_log = max(-6.0, math.log((p + 1e-9) / max_p))
            log_bands.append(rel_log)
            band_means[b] += rel_log

        # Project mean-centered log-bands to 16 cepstral coefficients (exclude c0 DC component)
        lb_mean = sum(log_bands) / num_bands
        centered_lb = [lb - lb_mean for lb in log_bands]
        cep = [
            sum(dct_basis[k][m] * centered_lb[m] for m in range(num_bands)) / math.sqrt(num_bands)
            for k in range(num_cepstra)
        ]
        cepstral_frames.append(cep)

    if not cepstral_frames:
        # Fallback if audio was near-silent: compute basic energy distribution
        fallback = [float_samples[i % len(float_samples)] for i in range(VOICE_EMBEDDING_DIM)]
        return normalize_vector(fallback)

    n_frames = len(cepstral_frames)
    band_means = [b / n_frames for b in band_means]

    # Mean and standard deviation of the 16 cepstral coefficients (32 dims)
    cep_means = [
        sum(frame[k] for frame in cepstral_frames) / n_frames
        for k in range(num_cepstra)
    ]
    cep_stds = [
        math.sqrt(
            sum((frame[k] - cep_means[k]) ** 2 for frame in cepstral_frames) / n_frames + 1e-9
        )
        for k in range(num_cepstra)
    ]

    # 24 normalized log-mel sub-band means
    band_mean_center = sum(band_means) / num_bands
    norm_bands = [b - band_mean_center for b in band_means]

    # 8 prosodic & voice quality features (pitch mean, pitch std, ZCR mean, ZCR std, RMS mean, RMS std, spectral tilt,voiced ratio)
    voiced = [p for p in pitch_lags if p > 0]
    pitch_mean = sum(voiced) / len(voiced) if voiced else 0.0
    pitch_std = (
        math.sqrt(sum((p - pitch_mean) ** 2 for p in voiced) / len(voiced))
        if len(voiced) > 1
        else 0.0
    )
    voiced_ratio = len(voiced) / n_frames
    zcr_mean = sum(zcr_values) / n_frames
    zcr_std = math.sqrt(sum((z - zcr_mean) ** 2 for z in zcr_values) / n_frames)
    rms_mean = sum(rms_values) / n_frames
    rms_std = math.sqrt(sum((r - rms_mean) ** 2 for r in rms_values) / n_frames)
    spectral_tilt = (band_means[-1] - band_means[0]) / max(1.0, abs(band_mean_center))

    prosody = [
        pitch_mean,
        pitch_std,
        voiced_ratio,
        zcr_mean,
        zcr_std,
        rms_mean,
        rms_std,
        spectral_tilt,
    ]

    # Combine: 16 cep_means + 16 cep_stds + 24 norm_bands + 8 prosody = 64 dims
    embedding = cep_means + cep_stds + norm_bands + prosody
    return normalize_vector(embedding[:VOICE_EMBEDDING_DIM])


def extract_voice_embedding_from_wav(wav_path: Path | str) -> list[float]:
    """Read a WAV file from disk and extract its 64-dimensional speaker voiceprint embedding."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw_frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        pcm = array.array("h")
        pcm.frombytes(raw_frames)
        if n_channels > 1:
            mono = [pcm[i] / 32768.0 for i in range(0, len(pcm), n_channels)]
        else:
            mono = [s / 32768.0 for s in pcm]
    elif sampwidth == 1:
        mono = [((b - 128) / 128.0) for b in raw_frames[::n_channels]]
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    return extract_voice_embedding_from_pcm(mono, sample_rate=framerate)


class VoiceIdentifier:
    """Matches acoustic signatures and voiceprint embeddings to registered people or visitors."""

    def __init__(
        self,
        people_profiles: dict[str, dict[str, Any]] | None = None,
        people_dir: Path | None = None,
        biometric_store: BiometricStore | None = None,
        similarity_threshold: float = 0.78,
    ) -> None:
        self.profiles = load_people_profiles(people_dir=people_dir, raw_profiles=people_profiles)
        self.biometric_store = biometric_store
        self.similarity_threshold = similarity_threshold

    def register_person_profile(self, person_id: str, profile: dict[str, Any]) -> None:
        """Dynamically add or update a person profile at runtime."""
        pid = person_id.lower().strip()
        profile_copy = dict(profile)
        profile_copy["id"] = pid
        self.profiles[pid] = profile_copy

    def enroll_voice(
        self,
        person_id: str,
        audio_samples: list[float] | bytes | None = None,
        wav_path: Path | str | None = None,
        embedding: list[float] | None = None,
        sample_rate: int = 16000,
    ) -> str:
        """Extract and enroll a voice biometric embedding for a person."""
        if self.biometric_store is None:
            self.biometric_store = BiometricStore()

        if embedding is not None:
            vec = normalize_vector(embedding)
        elif wav_path is not None:
            vec = extract_voice_embedding_from_wav(wav_path)
        elif audio_samples is not None:
            vec = extract_voice_embedding_from_pcm(audio_samples, sample_rate=sample_rate)
        else:
            raise ValueError("Must provide audio_samples, wav_path, or embedding to enroll_voice.")

        pid = person_id.lower().strip()
        profile = self.profiles.get(pid, {})
        print_id = profile.get("voice_print_id") or f"vp_{pid}_01"
        rec = self.biometric_store.enroll_embedding(
            person_id=pid,
            modality="voice",
            embedding=vec,
            print_id=print_id,
            metadata={"sample_rate": sample_rate},
        )
        return rec.print_id

    def _build_result_from_profile(
        self,
        pid: str,
        profile: dict[str, Any],
        confidence: float,
        matched_print_id: str | None = None,
    ) -> VoiceIdResult:
        langs = profile.get("preferred_languages", ["nb-NO"])
        role = profile.get("role", "family_member")
        return VoiceIdResult(
            speaker_id=pid,
            is_known_family_member=role != "visitor",
            confidence=round(float(confidence), 4),
            preferred_language=langs[0] if langs else "nb-NO",
            role=role,
            display_name=profile.get("name", pid.capitalize()),
            matched_print_id=matched_print_id or profile.get("voice_print_id"),
        )

    def identify(
        self,
        voice_print_id: str | None = None,
        claimed_name: str | None = None,
        audio_samples: list[float] | bytes | None = None,
        wav_path: Path | str | None = None,
        voice_embedding: list[float] | None = None,
        sample_rate: int = 16000,
    ) -> VoiceIdResult:
        """Resolve a speaker identity using acoustic biometric matching, print ID, or claimed name."""
        # 1. Acoustic Biometric Embedding Matching (highest authenticity when audio/embedding is supplied)
        query_emb: list[float] | None = None
        if voice_embedding is not None:
            query_emb = normalize_vector(voice_embedding)
        elif wav_path is not None:
            query_emb = extract_voice_embedding_from_wav(wav_path)
        elif audio_samples is not None:
            query_emb = extract_voice_embedding_from_pcm(audio_samples, sample_rate=sample_rate)

        if query_emb is not None and self.biometric_store is not None:
            match = self.biometric_store.match_embedding(
                modality="voice",
                query_embedding=query_emb,
                threshold=self.similarity_threshold,
            )
            if match.matched:
                profile = self.profiles.get(
                    match.person_id,
                    {"id": match.person_id, "name": match.person_id.capitalize(), "role": "family_member"},
                )
                return self._build_result_from_profile(
                    pid=match.person_id,
                    profile=profile,
                    confidence=match.similarity,
                    matched_print_id=match.print_id,
                )

        # 2. Explicit voice_print_id lookup
        if voice_print_id:
            for pid, profile in self.profiles.items():
                if profile.get("voice_print_id") == voice_print_id:
                    return self._build_result_from_profile(
                        pid=pid,
                        profile=profile,
                        confidence=0.95,
                        matched_print_id=voice_print_id,
                    )

        # 3. Claimed name resolution
        if claimed_name:
            norm_name = claimed_name.lower().strip()
            for pid, profile in self.profiles.items():
                if norm_name == pid.lower() or norm_name == profile.get("name", "").lower():
                    return self._build_result_from_profile(
                        pid=pid,
                        profile=profile,
                        confidence=0.98,
                    )

        # 4. Fallback to visitor
        visitor_profile = self.profiles.get("visitor", {})
        v_langs = visitor_profile.get("preferred_languages", ["nb-NO"])
        return VoiceIdResult(
            speaker_id="visitor",
            is_known_family_member=False,
            confidence=0.60,
            preferred_language=v_langs[0] if v_langs else "nb-NO",
            role="visitor",
            display_name="Visitor",
            matched_print_id=None,
        )
