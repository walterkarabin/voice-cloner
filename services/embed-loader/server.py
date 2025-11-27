"""
Embedding Loader Service

Provides preloaded voice embeddings for character voices.
Embeddings are loaded at startup for fast retrieval.
"""

import os
import sys
import grpc
import logging
import numpy as np
from concurrent import futures
from typing import Dict, Optional

# Add libs to path
sys.path.insert(0, '/workspace')

# Import generated protobuf code
from libs.proto.generated.python import embed_pb2, embed_pb2_grpc, common_pb2


class EmbedLoaderService(embed_pb2_grpc.EmbedLoaderServicer):
    """
    gRPC service for loading character voice embeddings.
    """

    def __init__(self, embeddings_dir: str = "/data/embeddings"):
        """
        Initialize the embedding loader service.

        Args:
            embeddings_dir: Directory containing embedding files
        """
        self.embeddings_dir = embeddings_dir
        self.embeddings: Dict[str, np.ndarray] = {}
        self.logger = logging.getLogger(__name__)

        # Load all embeddings at startup
        self._load_embeddings()

    def _load_embeddings(self):
        """Load all character embeddings from disk."""
        self.logger.info(f"Loading embeddings from: {self.embeddings_dir}")

        if not os.path.exists(self.embeddings_dir):
            self.logger.warning(
                f"Embeddings directory not found: {self.embeddings_dir}"
            )
            # Create sample embeddings for testing
            self._create_sample_embeddings()
            return

        # Load .npy files from embeddings directory
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith('.npy'):
                character_id = filename[:-4]  # Remove .npy extension
                filepath = os.path.join(self.embeddings_dir, filename)

                try:
                    embedding = np.load(filepath)
                    self.embeddings[character_id] = embedding
                    self.logger.info(
                        f"Loaded embedding for '{character_id}': "
                        f"shape {embedding.shape}, dtype {embedding.dtype}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to load embedding {filename}: {e}"
                    )

        self.logger.info(
            f"Loaded {len(self.embeddings)} character embeddings"
        )

    def _create_sample_embeddings(self):
        """Create sample embeddings for testing."""
        self.logger.info("Creating sample embeddings for testing")

        # Create sample embeddings (256-dimensional)
        sample_characters = ["yoda", "vader", "obi-wan", "leia"]

        for character_id in sample_characters:
            # Generate random embedding (would be real voice embedding in production)
            embedding = np.random.randn(256).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            self.embeddings[character_id] = embedding

            self.logger.info(
                f"Created sample embedding for '{character_id}': "
                f"shape {embedding.shape}"
            )

    def GetEmbedding(
        self,
        request: embed_pb2.EmbeddingRequest,
        context: grpc.ServicerContext
    ) -> embed_pb2.EmbeddingResponse:
        """
        Get embedding for a specific character.

        Args:
            request: EmbeddingRequest with character_id
            context: gRPC context

        Returns:
            EmbeddingResponse with embedding data
        """
        character_id = request.character_id

        self.logger.debug(f"Embedding requested for: {character_id}")

        # Check if embedding exists
        if character_id not in self.embeddings:
            self.logger.error(f"Embedding not found for: {character_id}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(
                f"Embedding not found for character: {character_id}"
            )
            return embed_pb2.EmbeddingResponse()

        # Get embedding
        embedding = self.embeddings[character_id]

        # Convert to bytes
        embedding_bytes = embedding.tobytes()

        # Create response
        response = embed_pb2.EmbeddingResponse(
            character_id=character_id,
            embedding_data=embedding_bytes,
            embedding_dim=len(embedding),
            metadata=common_pb2.Metadata(
                request_id=request.metadata.request_id
                if request.metadata else ""
            )
        )

        return response

    def list_characters(self) -> list:
        """Get list of available character IDs."""
        return list(self.embeddings.keys())


def serve(port: int = 50051, embeddings_dir: str = "/data/embeddings"):
    """
    Start the embedding loader gRPC server.

    Args:
        port: Port to listen on
        embeddings_dir: Directory containing embeddings
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add service
    embed_loader = EmbedLoaderService(embeddings_dir)
    embed_pb2_grpc.add_EmbedLoaderServicer_to_server(embed_loader, server)

    # Start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f"Embed-loader service started on port {port}")
    logger.info(f"Available characters: {embed_loader.list_characters()}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down embed-loader service")
        server.stop(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Character Voice Embedding Loader Service'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=50051,
        help='Port to listen on (default: 50051)'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='/data/embeddings',
        help='Directory containing embedding files'
    )

    args = parser.parse_args()
    serve(port=args.port, embeddings_dir=args.embeddings_dir)
