"""
LLM Rewriter Client

Example client for testing the character voice rewriter service.
"""

import sys
import grpc
import logging
from typing import Iterator

sys.path.insert(0, '/workspace')

from libs.proto.generated.python import rewriter_pb2, rewriter_pb2_grpc, stt_pb2, common_pb2


class LLMRewriterClient:
    """Client for the LLM rewriter service."""

    def __init__(self, host: str = 'localhost', port: int = 50053):
        """
        Initialize client.

        Args:
            host: Service host
            port: Service port
        """
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = rewriter_pb2_grpc.LLMRewriterStub(self.channel)
        self.logger = logging.getLogger(__name__)

    def rewrite_stream(
        self,
        transcripts: Iterator[dict],
        character_id: str = "yoda"
    ) -> Iterator[dict]:
        """
        Rewrite streaming transcripts in character voice.

        Args:
            transcripts: Iterator of transcript dictionaries
            character_id: Character to emulate

        Yields:
            Dictionary with rewritten text
        """
        def transcript_generator():
            for t in transcripts:
                # Create PartialTranscript message
                transcript = stt_pb2.PartialTranscript(
                    text=t.get('text', ''),
                    confidence=t.get('confidence', 1.0),
                    is_final=t.get('is_final', False),
                    start_time_ms=t.get('start_time_ms', 0),
                    end_time_ms=t.get('end_time_ms', 0),
                    metadata=common_pb2.Metadata(
                        sequence_number=t.get('sequence', 0)
                    )
                )
                yield transcript

        try:
            # Call streaming RPC
            responses = self.stub.RewriteStream(transcript_generator())

            # Process responses
            for response in responses:
                fragment = {
                    "text": response.text,
                    "character_id": response.character_id,
                    "is_provisional": response.is_provisional,
                    "original_start_ms": response.original_start_ms,
                    "original_end_ms": response.original_end_ms,
                    "sequence": response.metadata.sequence_number
                }

                self.logger.info(
                    f"[{fragment['character_id']}] "
                    f"{'PROVISIONAL' if fragment['is_provisional'] else 'FINAL'}: "
                    f"{fragment['text']}"
                )

                yield fragment

        except grpc.RpcError as e:
            self.logger.error(f"RPC failed: {e.code()} - {e.details()}")
            raise

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def main():
    """Test the LLM rewriter client."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== LLM Rewriter Client Test ===\n")

    # Create client
    client = LLMRewriterClient()

    # Test transcripts
    test_transcripts = [
        {
            "text": "Hello, how are you today?",
            "confidence": 0.95,
            "is_final": False,
            "start_time_ms": 0,
            "end_time_ms": 1000,
            "sequence": 0
        },
        {
            "text": "I am doing great, thank you.",
            "confidence": 0.92,
            "is_final": False,
            "start_time_ms": 1000,
            "end_time_ms": 2000,
            "sequence": 1
        },
        {
            "text": "The force is strong with you.",
            "confidence": 0.98,
            "is_final": True,
            "start_time_ms": 2000,
            "end_time_ms": 3500,
            "sequence": 2
        },
    ]

    print("Testing character rewriting...")
    print("Original → Yoda style\n")

    # Rewrite in Yoda style
    fragments = []
    try:
        for fragment in client.rewrite_stream(test_transcripts, character_id="yoda"):
            fragments.append(fragment)
            print(f"Original: {test_transcripts[fragment['sequence']]['text']}")
            print(f"Yoda:     {fragment['text']}")
            print()

    except grpc.RpcError as e:
        print(f"Error: {e.details()}")
    except Exception as e:
        print(f"Error: {e}")

    print(f"✓ Received {len(fragments)} rewritten fragment(s)")

    client.close()


if __name__ == '__main__':
    main()
