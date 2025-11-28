#!/usr/bin/env python3
"""
TTS-Streamer Service Client

Test client for the TTS mel generation service.
"""

import sys
import grpc
import numpy as np
from typing import Iterator, Dict, Any

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import tts_pb2, tts_pb2_grpc
from libs.proto.generated.python import audio_pb2


class TTSStreamerClient:
    """Client for the TTS-Streamer gRPC service."""

    def __init__(self, host: str = 'localhost', port: int = 50055):
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = tts_pb2_grpc.TTSStreamerStub(self.channel)

    def generate_mel_stream(
        self,
        requests: list[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """
        Send TTS requests and receive mel chunks.

        Args:
            requests: List of request dicts with keys:
                - text: str
                - embedding: np.ndarray (optional)
                - character_id: str (optional)
                - chunk_index: int (optional)

        Yields:
            Dict with keys: data (np.ndarray), n_mels, n_frames
        """
        def request_generator():
            for req in requests:
                embedding = req.get('embedding')
                embedding_data = b''
                embedding_dim = 0

                if embedding is not None:
                    if isinstance(embedding, np.ndarray):
                        embedding_data = embedding.astype(np.float32).tobytes()
                        embedding_dim = len(embedding)

                yield tts_pb2.TTSRequest(
                    text=req['text'],
                    embedding_data=embedding_data,
                    embedding_dim=embedding_dim,
                    character_id=req.get('character_id', 'unknown'),
                    chunk_index=req.get('chunk_index', 0)
                )

        try:
            for mel_chunk in self.stub.GenerateMel(request_generator()):
                mel_data = np.frombuffer(mel_chunk.data, dtype=np.float32).reshape(
                    mel_chunk.n_mels, mel_chunk.n_frames
                )
                yield {
                    'data': mel_data,
                    'n_mels': mel_chunk.n_mels,
                    'n_frames': mel_chunk.n_frames
                }
        except grpc.RpcError as e:
            print(f"RPC error: {e.code()} - {e.details()}")
            raise

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def main():
    """Test the TTS-Streamer service."""
    print("=== TTS-Streamer Client Test ===\n")

    client = TTSStreamerClient(host='localhost', port=50055)

    # Generate a synthetic embedding
    embedding = np.random.randn(256).astype(np.float32)

    # Test requests
    test_requests = [
        {
            'text': "The force is strong with you.",
            'embedding': embedding,
            'character_id': 'yoda',
            'chunk_index': 0
        },
        {
            'text': "You must learn to control it.",
            'embedding': embedding,
            'character_id': 'yoda',
            'chunk_index': 1
        }
    ]

    print("Sending TTS requests...\n")

    try:
        mel_chunks = list(client.generate_mel_stream(test_requests))

        print(f"Received {len(mel_chunks)} mel chunks:\n")

        total_frames = 0
        for i, chunk in enumerate(mel_chunks):
            print(
                f"[{i:2d}] Mel chunk: {chunk['n_mels']}x{chunk['n_frames']} "
                f"(min={chunk['data'].min():.2f}, max={chunk['data'].max():.2f})"
            )
            total_frames += chunk['n_frames']

        # Estimate audio duration
        hop_length = 256
        sample_rate = 22050
        duration = total_frames * hop_length / sample_rate

        print(f"\n✓ Received {len(mel_chunks)} mel chunk(s)")
        print(f"  Total frames: {total_frames}")
        print(f"  Estimated duration: {duration:.2f}s")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client.close()


if __name__ == '__main__':
    main()
