#!/usr/bin/env python3
"""
Audio-Out Service Client

Test client for the audio playback service.
"""

import sys
import grpc
import numpy as np
import time
from typing import Iterator

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import audio_out_pb2, audio_out_pb2_grpc
from libs.proto.generated.python import audio_pb2, common_pb2


class AudioOutClient:
    """Client for the Audio-Out gRPC service."""

    def __init__(self, host: str = 'localhost', port: int = 50057):
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = audio_out_pb2_grpc.AudioOutStub(self.channel)

    def play_stream(self, pcm_chunks: list[np.ndarray], sample_rate: int = 22050):
        """
        Send PCM chunks for playback.

        Args:
            pcm_chunks: List of PCM arrays (float32, range [-1, 1])
            sample_rate: Sample rate in Hz
        """
        def chunk_generator():
            for chunk in pcm_chunks:
                # Convert to int16 for transmission
                pcm_int16 = (chunk * 32767.0).astype(np.int16)

                yield audio_pb2.PCMChunk(
                    data=pcm_int16.tobytes(),
                    sample_rate=sample_rate,
                    channels=1,
                    sample_width=2
                )

        try:
            self.stub.PlayStream(chunk_generator())
        except grpc.RpcError as e:
            print(f"RPC error: {e.code()} - {e.details()}")
            raise

    def control(
        self,
        command: audio_out_pb2.PlaybackCommand,
        volume: float = 0.0
    ) -> dict:
        """
        Send playback control command.

        Args:
            command: Playback command (PLAY, PAUSE, MUTE, etc.)
            volume: Volume level (0.0 - 1.0), ignored if 0.0

        Returns:
            Status dict with is_playing, is_muted, current_volume, buffer_size_ms
        """
        try:
            response = self.stub.Control(
                audio_out_pb2.PlaybackControl(
                    command=command,
                    volume=volume
                )
            )
            return {
                'is_playing': response.is_playing,
                'is_muted': response.is_muted,
                'current_volume': response.current_volume,
                'buffer_size_ms': response.buffer_size_ms
            }
        except grpc.RpcError as e:
            print(f"RPC error: {e.code()} - {e.details()}")
            raise

    def get_status(self) -> dict:
        """
        Get current playback status.

        Returns:
            Status dict with is_playing, is_muted, current_volume, buffer_size_ms
        """
        try:
            response = self.stub.GetStatus(common_pb2.Empty())
            return {
                'is_playing': response.is_playing,
                'is_muted': response.is_muted,
                'current_volume': response.current_volume,
                'buffer_size_ms': response.buffer_size_ms
            }
        except grpc.RpcError as e:
            print(f"RPC error: {e.code()} - {e.details()}")
            raise

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def main():
    """Test the Audio-Out service."""
    print("=== Audio-Out Client Test ===\n")

    client = AudioOutClient(host='localhost', port=50057)

    # Generate test audio (simple tones)
    sample_rate = 22050
    duration_per_chunk = 0.2  # 200ms per chunk
    samples_per_chunk = int(duration_per_chunk * sample_rate)

    # Generate 5 chunks with different frequencies
    frequencies = [220, 247, 262, 294, 330]  # A3, B3, C4, D4, E4
    pcm_chunks = []

    print("Generating test audio chunks...\n")
    for freq in frequencies:
        t = np.arange(samples_per_chunk) / sample_rate
        chunk = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        pcm_chunks.append(chunk)

    # Test 1: Check status
    print("1. Getting initial status...")
    status = client.get_status()
    print(f"   Status: playing={status['is_playing']}, muted={status['is_muted']}, "
          f"volume={status['current_volume']:.2f}\n")

    # Test 2: Play audio stream
    print("2. Streaming audio chunks...")
    try:
        client.play_stream(pcm_chunks, sample_rate=sample_rate)
        print(f"   ✓ Sent {len(pcm_chunks)} chunks\n")

        # Wait for playback to finish
        total_duration = len(pcm_chunks) * duration_per_chunk
        print(f"   Waiting {total_duration:.1f}s for playback to complete...")
        time.sleep(total_duration + 0.5)

    except Exception as e:
        print(f"   ✗ Error: {e}\n")

    # Test 3: Control commands
    print("3. Testing control commands...")

    try:
        # Set volume
        status = client.control(audio_out_pb2.PLAY, volume=0.8)
        print(f"   Set volume to 0.8: volume={status['current_volume']:.2f}")

        # Mute
        status = client.control(audio_out_pb2.MUTE)
        print(f"   Muted: is_muted={status['is_muted']}")

        # Unmute
        status = client.control(audio_out_pb2.UNMUTE)
        print(f"   Unmuted: is_muted={status['is_muted']}")

        print("\n   ✓ Control commands successful")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: Final status
    print("\n4. Final status...")
    status = client.get_status()
    print(f"   Status: playing={status['is_playing']}, muted={status['is_muted']}, "
          f"volume={status['current_volume']:.2f}, buffer={status['buffer_size_ms']}ms")

    print("\n✓ Audio-Out client test completed")
    client.close()


if __name__ == '__main__':
    main()
