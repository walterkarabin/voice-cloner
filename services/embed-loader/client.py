"""
Embedding Loader Client

Example client for testing the embedding loader service.
"""

import sys
import grpc
import logging
import numpy as np

sys.path.insert(0, '/workspace')

from libs.proto.generated.python import embed_pb2, embed_pb2_grpc, common_pb2


class EmbedLoaderClient:
    """Client for the embedding loader service."""

    def __init__(self, host: str = 'localhost', port: int = 50051):
        """
        Initialize client.

        Args:
            host: Service host
            port: Service port
        """
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = embed_pb2_grpc.EmbedLoaderStub(self.channel)
        self.logger = logging.getLogger(__name__)

    def get_embedding(self, character_id: str) -> np.ndarray:
        """
        Get embedding for a character.

        Args:
            character_id: Character identifier

        Returns:
            Embedding as numpy array

        Raises:
            grpc.RpcError: If character not found or service error
        """
        request = embed_pb2.EmbeddingRequest(
            character_id=character_id,
            metadata=common_pb2.Metadata(request_id=f"req_{character_id}")
        )

        try:
            response = self.stub.GetEmbedding(request)

            # Convert bytes back to numpy array
            embedding = np.frombuffer(
                response.embedding_data,
                dtype=np.float32
            )

            self.logger.info(
                f"Received embedding for '{character_id}': "
                f"dim={response.embedding_dim}"
            )

            return embedding

        except grpc.RpcError as e:
            self.logger.error(f"RPC failed: {e.code()} - {e.details()}")
            raise

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def main():
    """Test the embedding loader client."""
    logging.basicConfig(level=logging.INFO)

    # Create client
    client = EmbedLoaderClient()

    # Test characters
    test_characters = ["yoda", "vader", "obi-wan", "leia", "unknown"]

    for character_id in test_characters:
        try:
            embedding = client.get_embedding(character_id)
            print(f"✓ {character_id}: embedding shape {embedding.shape}")
        except grpc.RpcError as e:
            print(f"✗ {character_id}: {e.details()}")

    client.close()


if __name__ == '__main__':
    main()
