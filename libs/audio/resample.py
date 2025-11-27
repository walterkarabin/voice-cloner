"""
Audio resampling utilities using scipy.
"""

import numpy as np
from scipy import signal
from typing import Optional


class AudioResampler:
    """
    Resample audio between different sample rates.

    Uses high-quality polyphase filtering for resampling.
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1
    ):
        """
        Initialize resampler.

        Args:
            input_rate: Input sample rate (Hz)
            output_rate: Output sample rate (Hz)
            channels: Number of audio channels
        """
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels

        # Compute resampling ratio
        from math import gcd
        common = gcd(input_rate, output_rate)
        self.up = output_rate // common
        self.down = input_rate // common

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio (shape: [samples] or [samples, channels])

        Returns:
            Resampled audio
        """
        if self.input_rate == self.output_rate:
            return audio

        # Handle multi-channel audio
        if audio.ndim == 2:
            resampled_channels = []
            for ch in range(audio.shape[1]):
                resampled = signal.resample_poly(
                    audio[:, ch],
                    self.up,
                    self.down
                )
                resampled_channels.append(resampled)
            return np.stack(resampled_channels, axis=1)
        else:
            return signal.resample_poly(audio, self.up, self.down)

    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output length for given input length.

        Args:
            input_length: Number of input samples

        Returns:
            Expected number of output samples
        """
        return int(input_length * self.output_rate / self.input_rate)
