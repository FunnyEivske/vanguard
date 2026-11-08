"""Senses & Perception subsystem for V.A.N.G.U.A.R.D."""

from vanguard.senses.biometric_store import (
    BiometricMatch,
    BiometricRecord,
    BiometricStore,
    load_people_profiles,
)
from vanguard.senses.diarization import DiarizedTurn, StreamingDiarizer
from vanguard.senses.enrollment import BiometricEnrollmentService, EnrollmentSummary
from vanguard.senses.face_id import BoundingBox, FaceIdentifier, FaceIdResult
from vanguard.senses.identity_fusion import IdentityFusionEngine, ResolvedIdentity
from vanguard.senses.name_spotting import NameSpotter, NameSpottingResult
from vanguard.senses.sensory_buffer import RollingSensoryBuffer, SensoryFrame
from vanguard.senses.voice_id import VoiceIdentifier, VoiceIdResult

__all__ = [
    "BiometricEnrollmentService",
    "BiometricMatch",
    "BiometricRecord",
    "BiometricStore",
    "BoundingBox",
    "DiarizedTurn",
    "EnrollmentSummary",
    "FaceIdResult",
    "FaceIdentifier",
    "IdentityFusionEngine",
    "NameSpotter",
    "NameSpottingResult",
    "ResolvedIdentity",
    "RollingSensoryBuffer",
    "SensoryFrame",
    "StreamingDiarizer",
    "VoiceIdResult",
    "VoiceIdentifier",
    "load_people_profiles",
]
