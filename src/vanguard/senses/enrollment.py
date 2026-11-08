"""Person Biometric Enrollment and Linux Server Sensory Capture for V.A.N.G.U.A.R.D.

Enables registering household members or known guests with real voiceprint and facial
embeddings (from Linux ALSA/PulseAudio/V4L2 devices or WAV/image files) and transcribing
speech via Online-First Whisper API with Local faster-whisper Backup.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vanguard.senses.biometric_store import BiometricStore
from vanguard.senses.face_id import FaceIdentifier
from vanguard.senses.voice_id import VoiceIdentifier


@dataclass
class EnrollmentSummary:
    """Result of enrolling a person's voice and/or face."""

    person_id: str
    display_name: str
    role: str
    voice_print_id: str | None
    face_print_id: str | None
    profile_path: Path | None


def capture_microphone_wav(duration_seconds: float = 4.0, sample_rate: int = 16000) -> Path:
    """Record mono PCM WAV audio on a Linux server via sounddevice, ALSA `arecord`, ffmpeg ALSA/Pulse, or sox."""
    tmp_wav = Path(tempfile.mktemp(suffix=".wav", prefix="vanguard_mic_"))

    # 1. Try sounddevice if installed
    try:
        import wave
        import sounddevice as sd  # type: ignore

        frames = int(duration_seconds * sample_rate)
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        with wave.open(str(tmp_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        return tmp_wav
    except Exception:
        pass

    # 2. Try Linux ALSA `arecord`
    try:
        subprocess.run(
            [
                "arecord",
                "-q",
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                "1",
                "-d",
                str(max(1, int(round(duration_seconds)))),
                str(tmp_wav),
            ],
            check=True,
            capture_output=True,
            timeout=duration_seconds + 5.0,
        )
        if tmp_wav.exists():
            return tmp_wav
    except Exception:
        pass

    # 3. Try ffmpeg with Linux ALSA, PulseAudio, or macOS AVFoundation input
    for fmt, dev in (("alsa", "default"), ("pulse", "default"), ("avfoundation", ":0")):
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    fmt,
                    "-i",
                    dev,
                    "-t",
                    str(duration_seconds),
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    str(tmp_wav),
                ],
                check=True,
                capture_output=True,
                timeout=duration_seconds + 5.0,
            )
            if tmp_wav.exists():
                return tmp_wav
        except Exception:
            continue

    # 4. Try sox / rec
    try:
        subprocess.run(
            [
                "rec",
                "-q",
                "-r",
                str(sample_rate),
                "-c",
                "1",
                "-b",
                "16",
                str(tmp_wav),
                "trim",
                "0",
                str(duration_seconds),
            ],
            check=True,
            capture_output=True,
            timeout=duration_seconds + 5.0,
        )
        if tmp_wav.exists():
            return tmp_wav
    except Exception as exc:
        raise RuntimeError(
            "No live Linux microphone backend available (install `alsa-utils` for `arecord`, `sounddevice`, or `ffmpeg`, or pass `--voice-wav`)."
        ) from exc

    raise RuntimeError("Microphone capture failed to produce a WAV file.")


def capture_webcam_frame(camera_index: int = 0) -> Any:
    """Capture a single frame from a Linux V4L2 camera (`/dev/video<N>`)."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
    except Exception:
        pass

    # Fallback to Linux V4L2 via ffmpeg or fswebcam
    tmp_img = Path(tempfile.mktemp(suffix=".jpg", prefix="vanguard_cam_"))
    device_node = f"/dev/video{camera_index}"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "v4l2",
                "-i",
                device_node,
                "-frames:v",
                "1",
                str(tmp_img),
            ],
            check=True,
            capture_output=True,
            timeout=6.0,
        )
        if tmp_img.exists():
            return tmp_img
    except Exception:
        pass

    try:
        subprocess.run(
            ["fswebcam", "-q", "-d", device_node, "--no-banner", str(tmp_img)],
            check=True,
            capture_output=True,
            timeout=6.0,
        )
        if tmp_img.exists():
            return tmp_img
    except Exception as exc:
        raise RuntimeError(
            f"No live Linux V4L2 camera backend available on {device_node} (install `opencv-python`, `ffmpeg`, or `fswebcam`, or pass `--face-image`)."
        ) from exc

    raise RuntimeError("Camera capture failed.")


def transcribe_audio_wav(wav_path: Path | str, language: str = "nb") -> str:
    """Transcribe a WAV file: Online-First (OpenAI Whisper API), Local Backup (faster-whisper / whisper)."""
    lang_code = language.split("-")[0].lower()
    path = Path(wav_path)

    # 1. ONLINE PRIMARY: OpenAI Audio Transcriptions API
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("VANGUARD_OFFLINE_MODE") != "1":
        try:
            api_key = os.environ["OPENAI_API_KEY"]
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            boundary = f"----VanguardBoundary{uuid.uuid4().hex}"
            file_bytes = path.read_bytes()

            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="language"\r\n\r\n{lang_code}\r\n'
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

            req = urllib.request.Request(
                f"{base_url}/audio/transcriptions",
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                txt = str(data.get("text", "")).strip()
                if txt:
                    return txt
        except Exception:
            pass

    # 2. LOCAL BACKUP: faster-whisper or openai-whisper on the Linux host
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path), language=lang_code)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception:
        pass

    try:
        import whisper  # type: ignore

        model = whisper.load_model("base")
        res = model.transcribe(str(path), language=lang_code)
        return str(res.get("text", "")).strip()
    except Exception:
        return ""


class BiometricEnrollmentService:
    """Registers people and enrolls their voice and face biometrics."""

    def __init__(
        self,
        people_dir: Path,
        biometric_store: BiometricStore,
        voice_identifier: VoiceIdentifier,
        face_identifier: FaceIdentifier,
    ) -> None:
        self.people_dir = people_dir
        self.biometric_store = biometric_store
        self.voice_id = voice_identifier
        self.face_id = face_identifier

    def enroll_person(
        self,
        person_id: str,
        display_name: str | None = None,
        role: str = "family_member",
        preferred_languages: list[str] | None = None,
        relationship: str = "Household Member",
        primary_room: str = "living_room",
        voice_samples: list[float] | bytes | None = None,
        voice_wav_path: Path | str | None = None,
        voice_embedding: list[float] | None = None,
        face_image: Any | None = None,
        face_image_path: Path | str | None = None,
        face_embedding: list[float] | None = None,
        persist_yaml: bool = True,
    ) -> EnrollmentSummary:
        """Enroll a person's voice and/or face biometrics and update their YAML profile."""
        pid = person_id.lower().strip()
        existing = self.voice_id.profiles.get(pid, {})

        resolved_name = display_name or existing.get("name") or pid.capitalize()
        resolved_role = existing.get("role") or role
        resolved_langs = preferred_languages or existing.get("preferred_languages") or ["nb-NO", "en-US"]
        is_head = resolved_role == "family_head"

        profile: dict[str, Any] = {
            "id": pid,
            "name": resolved_name,
            "relationship": existing.get("relationship", relationship),
            "role": resolved_role,
            "preferred_languages": resolved_langs,
            "voice_print_id": existing.get("voice_print_id", f"vp_{pid}_01"),
            "face_print_id": existing.get("face_print_id", f"fp_{pid}_01"),
            "permissions": existing.get(
                "permissions",
                {
                    "full_administrative_access": is_head,
                    "lead_mode_override": is_head,
                    "door_actuation_authorization": is_head,
                    "sandbox_override": is_head,
                },
            ),
            "primary_room": existing.get("primary_room", primary_room),
        }
        if "confidante_persona" in existing:
            profile["confidante_persona"] = existing["confidante_persona"]
        if "content_filter" in existing or resolved_role == "child":
            profile["content_filter"] = existing.get("content_filter", "child_friendly")

        self.voice_id.register_person_profile(pid, profile)
        self.face_id.register_person_profile(pid, profile)

        enrolled_vp: str | None = None
        enrolled_fp: str | None = None

        if any(x is not None for x in (voice_samples, voice_wav_path, voice_embedding)):
            enrolled_vp = self.voice_id.enroll_voice(
                person_id=pid,
                audio_samples=voice_samples,
                wav_path=voice_wav_path,
                embedding=voice_embedding,
            )

        if any(x is not None for x in (face_image, face_image_path, face_embedding)):
            enrolled_fp = self.face_id.enroll_face(
                person_id=pid,
                image=face_image,
                image_path=face_image_path,
                embedding=face_embedding,
            )

        yaml_path: Path | None = None
        if persist_yaml and self.people_dir:
            self.people_dir.mkdir(parents=True, exist_ok=True)
            if pid not in ("child_elder", "child_younger"):
                yaml_path = self.people_dir / f"{pid}.yaml"
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(profile, f, sort_keys=False, allow_unicode=True)

        return EnrollmentSummary(
            person_id=pid,
            display_name=resolved_name,
            role=resolved_role,
            voice_print_id=enrolled_vp,
            face_print_id=enrolled_fp,
            profile_path=yaml_path,
        )
