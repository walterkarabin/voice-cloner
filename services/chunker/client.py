#!/usr/bin/env python3
"""
Chunker Service Client

Test client for the text chunking service.
"""

import sys
import grpc
from typing import Iterator, Dict, Any

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import chunker_pb2, chunker_pb2_grpc
from libs.proto.generated.python import rewriter_pb2


class ChunkerClient:
    """Client for the Chunker gRPC service."""

    def __init__(self, host: str = 'localhost', port: int = 50054):
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = chunker_pb2_grpc.ChunkerStub(self.channel)

    def chunk_stream(
        self,
        fragments: list[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """
        Send character fragments and receive text chunks.

        Args:
            fragments: List of fragment dicts with keys:
                - text: str
                - character_id: str
                - is_provisional: bool (optional)
                - original_start_ms: int (optional)
                - original_end_ms: int (optional)

        Yields:
            Dict with keys: text, character_id, chunk_index, is_clause_boundary
        """
        def fragment_generator():
            for frag in fragments:
                yield rewriter_pb2.CharacterFragment(
                    text=frag['text'],
                    character_id=frag.get('character_id', 'unknown'),
                    is_provisional=frag.get('is_provisional', False),
                    original_start_ms=frag.get('original_start_ms', 0),
                    original_end_ms=frag.get('original_end_ms', 0)
                )

        try:
            for chunk in self.stub.ChunkTextStream(fragment_generator()):
                yield {
                    'text': chunk.text,
                    'character_id': chunk.character_id,
                    'chunk_index': chunk.chunk_index,
                    'is_clause_boundary': chunk.is_clause_boundary
                }
        except grpc.RpcError as e:
            print(f"RPC error: {e.code()} - {e.details()}")
            raise

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def main():
    """Test the Chunker service."""
    print("=== Chunker Client Test ===\n")

    client = ChunkerClient(host='localhost', port=50054)

    # Test fragments with various text lengths and structures
    test_fragments = [
        {
            'text': "The force is strong with you, young Skywalker, and you must learn to control it.",
            'character_id': 'yoda',
            'is_provisional': False
        },
        {
            'text': "This is a short sentence. And this is another one, with a comma in it.",
            'character_id': 'vader',
            'is_provisional': False
        },
        {
            'text': "A very long sentence that goes on and on without any punctuation to guide the chunker in splitting it properly",
            'character_id': 'obi-wan',
            'is_provisional': False
        }
    ]

    print("Sending character fragments for chunking...\n")

    try:
        chunks = list(client.chunk_stream(test_fragments))

        print(f"Received {len(chunks)} chunks:\n")

        for chunk in chunks:
            boundary_marker = "│" if chunk['is_clause_boundary'] else "┆"
            print(
                f"[{chunk['chunk_index']:2d}] {boundary_marker} "
                f"{chunk['character_id']:10s} │ {chunk['text']}"
            )

        print(f"\n✓ Received {len(chunks)} text chunk(s)")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client.close()


if __name__ == '__main__':
    main()
