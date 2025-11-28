#!/usr/bin/env python3
"""
Vocoder Service Client

Test client for the mel-to-PCM vocoder service.
"""

import sys
import grpc
import numpy as np
from typing import Iterator, Dict, Any

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import vocoder_pb2, vocoder_pb2_grpc
from libs.proto.generated.python import audio_pb2


class VocoderClient:
    """Client for the Vocoder gRPC service."""

    def __init__(self, host: str = 'localhost', port: int = 50056):
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = vocoder_pb2_grpc.VocoderStub(self.channel)

    def generate_pcm_stream(
        self,
        mel_chunks: list[np.ndarray]
    ) -> Iterator[Dict[str, Any]]:
        """
        Send mel chunks and receive PCM audio.

        Args:
            mel_chunks: List of mel spectrograms (n_mels x n_frames)

        Yields:
            Dict with keys: data (np.ndarray), sample_rate, channels, sample_width
        """
        def mel_generator():
            for mel in mel_chunks:
                yield audio_pb2.MelChunk(
                    data=mel.astype(np.float32).tobytes(),
                    n_mels=mel.shape[0],
                    n_frames=mel.shape[1]
                )

        try:
            for pcm_chunk in self.stub.GeneratePCM(mel_generator()):
                pcm_data = np.frombuffer(pcm_chunk.data, dtype=np.int16).astype(np.float32) / 32767.0
                yield {
                    'data': pcm_data,
                    'sample_rate': pcm_chunk.sample_rate,
                    'channels': pcm_chunk.channels,
                    'sample_width': pcm_chunk.sample_width
                }
        except grpc.RpcError as e:
            print(f"RPC error: {e.code()} - {e.details()}")
            raise

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def main():
    """Test the Vocoder service."""
    print("=== Vocoder Client Test ===\n")

    client = VocoderClient(host='localhost', port=50056)

    # Generate synthetic mel spectrograms
    n_mels = 80
    mel_chunks = [
        np.random.randn(n_mels, 50).astype(np.float32),  # 50 frames
        np.random.randn(n_mels, 50).astype(np.float32),  # 50 frames
        np.random.randn(n_mels, 30).astype(np.float32),  # 30 frames
    ]

    print(f"Sending {len(mel_chunks)} mel chunks...\n")

    try:
        pcm_chunks = list(client.generate_pcm_stream(mel_chunks))

        print(f"Received {len(pcm_chunks)} PCM chunks:\n")

        total_samples = 0
        for i, chunk in enumerate(pcm_chunks):
            print(
                f"[{i:2d}] PCM chunk: {len(chunk['data'])} samples, "
                f"{chunk['sample_rate']}Hz, {chunk['channels']}ch, "
                f"{chunk['sample_width']*8}-bit"
            )
            total_samples += len(chunk['data'])

        # Calculate duration
        if pcm_chunks:
            duration = total_samples / pcm_chunks[0]['sample_rate']
            print(f"\n✓ Received {len(pcm_chunks)} PCM chunk(s)")
            print(f"  Total samples: {total_samples}")
            print(f"  Duration: {duration:.2f}s")
        else:
            print("\n✗ No PCM chunks received")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client.close()


if __name__ == '__main__':
    main()
