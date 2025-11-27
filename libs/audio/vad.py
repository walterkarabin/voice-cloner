"""
Voice Activity Detection (VAD) using energy-based approach.

For production use, consider integrating WebRTC VAD or Silero VAD.
"""

import numpy as np
from typing import Optional
from .utils import compute_rms


class VoiceActivityDetector:
    """
    Simple energy-based Voice Activity Detection.

    Detects speech vs silence based on audio energy levels.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        speech_pad_ms: int = 300
    ):
        """
        Initialize VAD.

        Args:
            sample_rate: Audio sample rate (Hz)
            frame_duration_ms: Frame duration in milliseconds
            energy_threshold: RMS energy threshold for speech detection
            speech_pad_ms: Padding to add before/after speech (ms)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.speech_pad_ms = speech_pad_ms

        # Calculate frame size
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.speech_pad_frames = int(speech_pad_ms / frame_duration_ms)

        # State tracking
        self.speech_frames_count = 0
        self.silence_frames_count = 0

    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Determine if a single frame contains speech.

        Args:
            frame: Audio frame (float32 array)

        Returns:
            True if speech detected, False otherwise
        """
        energy = compute_rms(frame)
        return energy > self.energy_threshold

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single audio frame and return VAD decision.

        Args:
            frame: Audio frame (float32 array)

        Returns:
            Dictionary with:
                - is_speech: bool
                - energy: float
                - confidence: float (0.0 to 1.0)
        """
        energy = compute_rms(frame)
        is_speech = energy > self.energy_threshold

        # Update state
        if is_speech:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
        else:
            self.silence_frames_count += 1
            if self.silence_frames_count > self.speech_pad_frames:
                self.speech_frames_count = 0

        # Compute confidence based on energy relative to threshold
        confidence = min(1.0, energy / (self.energy_threshold * 2))

        return {
            "is_speech": is_speech or self.speech_frames_count > 0,
            "energy": float(energy),
            "confidence": float(confidence)
        }

    def set_threshold(self, threshold: float):
        """
        Update energy threshold.

        Args:
            threshold: New RMS energy threshold
        """
        self.energy_threshold = threshold

    def calibrate(self, noise_sample: np.ndarray, margin: float = 2.0):
        """
        Calibrate threshold based on background noise sample.

        Args:
            noise_sample: Audio sample of background noise
            margin: Multiplier above noise level for threshold
        """
        noise_energy = compute_rms(noise_sample)
        self.energy_threshold = noise_energy * margin

    def reset(self):
        """Reset internal state."""
        self.speech_frames_count = 0
        self.silence_frames_count = 0
