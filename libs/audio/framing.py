"""
Audio framing utilities for chunking audio into fixed-size frames.
"""

import numpy as np
from typing import Iterator, Optional


class AudioFramer:
    """
    Split audio into fixed-size frames with optional overlap.

    Useful for streaming audio processing where data arrives in chunks.
    """

    def __init__(
        self,
        frame_size: int,
        hop_size: Optional[int] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize audio framer.

        Args:
            frame_size: Size of each frame in samples
            hop_size: Number of samples to advance between frames
                     (defaults to frame_size for no overlap)
            sample_rate: Audio sample rate (Hz)
        """
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size
        self.sample_rate = sample_rate

        # Buffer for incomplete frames
        self.buffer = np.array([], dtype=np.float32)

    @classmethod
    def from_duration(
        cls,
        frame_duration_ms: int,
        sample_rate: int = 16000,
        overlap_percent: float = 0.0
    ):
        """
        Create framer from frame duration in milliseconds.

        Args:
            frame_duration_ms: Frame duration in milliseconds
            sample_rate: Audio sample rate (Hz)
            overlap_percent: Overlap between frames (0.0 to 1.0)

        Returns:
            AudioFramer instance
        """
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        hop_size = int(frame_size * (1.0 - overlap_percent))
        return cls(frame_size, hop_size, sample_rate)

    def add_samples(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        """
        Add new audio samples and yield complete frames.

        Args:
            samples: New audio samples to add

        Yields:
            Complete frames of size frame_size
        """
        # Add new samples to buffer
        self.buffer = np.concatenate([self.buffer, samples])

        # Yield complete frames
        while len(self.buffer) >= self.frame_size:
            frame = self.buffer[:self.frame_size]
            yield frame

            # Advance by hop size
            self.buffer = self.buffer[self.hop_size:]

    def get_remaining(self) -> Optional[np.ndarray]:
        """
        Get remaining samples in buffer (incomplete frame).

        Returns:
            Remaining samples or None if buffer is empty
        """
        if len(self.buffer) > 0:
            return self.buffer.copy()
        return None

    def flush(self, pad: bool = True) -> Optional[np.ndarray]:
        """
        Flush remaining samples as a final frame.

        Args:
            pad: If True, pad to frame_size with zeros

        Returns:
            Final frame or None if buffer is empty
        """
        if len(self.buffer) == 0:
            return None

        if pad and len(self.buffer) < self.frame_size:
            # Pad with zeros to reach frame_size
            padding = np.zeros(
                self.frame_size - len(self.buffer),
                dtype=np.float32
            )
            frame = np.concatenate([self.buffer, padding])
        else:
            frame = self.buffer.copy()

        self.buffer = np.array([], dtype=np.float32)
        return frame

    def reset(self):
        """Clear the internal buffer."""
        self.buffer = np.array([], dtype=np.float32)

    @property
    def buffer_size(self) -> int:
        """Get current buffer size in samples."""
        return len(self.buffer)

    @property
    def frame_duration_ms(self) -> float:
        """Get frame duration in milliseconds."""
        return (self.frame_size / self.sample_rate) * 1000

    @property
    def hop_duration_ms(self) -> float:
        """Get hop duration in milliseconds."""
        return (self.hop_size / self.sample_rate) * 1000
