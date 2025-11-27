"""
Audio utilities for the character voice pipeline.

This package provides utilities for:
- Audio resampling
- Voice Activity Detection (VAD)
- Audio framing
- Format conversions
"""

from .resample import AudioResampler
from .vad import VoiceActivityDetector
from .framing import AudioFramer
from .utils import (
    bytes_to_float32,
    float32_to_bytes,
    normalize_audio,
    compute_rms,
)

__all__ = [
    "AudioResampler",
    "VoiceActivityDetector",
    "AudioFramer",
    "bytes_to_float32",
    "float32_to_bytes",
    "normalize_audio",
    "compute_rms",
]
