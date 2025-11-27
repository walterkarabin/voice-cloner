"""
Basic audio utility functions.
"""

import numpy as np
from typing import Optional


def bytes_to_float32(
    audio_bytes: bytes,
    sample_width: int = 2,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert raw PCM bytes to float32 numpy array.

    Args:
        audio_bytes: Raw PCM audio data
        sample_width: Bytes per sample (2 for 16-bit, 4 for 32-bit)
        normalize: If True, normalize to [-1.0, 1.0] range

    Returns:
        Float32 numpy array
    """
    if sample_width == 2:
        dtype = np.int16
        max_val = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(audio_bytes, dtype=dtype)

    if normalize:
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    return audio


def float32_to_bytes(
    audio: np.ndarray,
    sample_width: int = 2
) -> bytes:
    """
    Convert float32 numpy array to raw PCM bytes.

    Args:
        audio: Float32 numpy array (range -1.0 to 1.0)
        sample_width: Bytes per sample (2 for 16-bit, 4 for 32-bit)

    Returns:
        Raw PCM bytes
    """
    if sample_width == 2:
        dtype = np.int16
        max_val = 32767
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483647
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to integer type
    audio_int = (audio * max_val).astype(dtype)

    return audio_int.tobytes()


def normalize_audio(
    audio: np.ndarray,
    target_level: float = 0.95
) -> np.ndarray:
    """
    Normalize audio to a target peak level.

    Args:
        audio: Float32 audio array
        target_level: Target peak level (0.0 to 1.0)

    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio).max()

    if max_val > 0:
        return audio * (target_level / max_val)
    else:
        return audio


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS (Root Mean Square) energy of audio.

    Args:
        audio: Float32 audio array

    Returns:
        RMS energy value
    """
    return np.sqrt(np.mean(audio ** 2))
