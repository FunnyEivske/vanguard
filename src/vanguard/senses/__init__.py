"""Senses & Perception subsystem for V.A.N.G.U.A.R.D."""

from vanguard.senses.diarization import DiarizedTurn, StreamingDiarizer
from vanguard.senses.name_spotting import NameSpotter, NameSpottingResult
from vanguard.senses.sensory_buffer import RollingSensoryBuffer, SensoryFrame
from vanguard.senses.voice_id import VoiceIdentifier, VoiceIdResult

__all__ = [
    "DiarizedTurn",
    "NameSpotter",
    "NameSpottingResult",
    "RollingSensoryBuffer",
    "SensoryFrame",
    "StreamingDiarizer",
    "VoiceIdResult",
    "VoiceIdentifier",
]
