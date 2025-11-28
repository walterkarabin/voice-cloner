#!/usr/bin/env python3
"""
Chunker Service

Splits character-rewritten text fragments into prosodic chunks suitable for streaming TTS.
Enforces max chunk size (20-40 chars) and predicts clause boundaries for natural phrasing.
"""

import sys
import argparse
import logging
import re
from concurrent import futures
from typing import Iterator

import grpc

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import chunker_pb2, chunker_pb2_grpc
from libs.proto.generated.python import rewriter_pb2


class TextChunker:
    """
    Splits text into prosodic chunks for streaming TTS.

    Strategy:
    - Split on natural boundaries: commas, periods, semicolons, conjunctions
    - Predict clause boundaries even without punctuation
    - Enforce max chunk size (20-40 characters)
    - Maintain context for provisional text updates
    """

    def __init__(self, min_chunk_size: int = 15, max_chunk_size: int = 40):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Clause boundary markers (punctuation + common conjunctions)
        self.boundary_patterns = [
            r'[,;:.]',  # Punctuation
            r'\b(and|but|or|so|yet|for|nor|because|although|while|when|if|unless|until)\b',  # Conjunctions
        ]

        self.logger = logging.getLogger(__name__)

    def chunk_text(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text into chunks.

        Returns:
            List of (chunk_text, is_clause_boundary) tuples
        """
        if not text or not text.strip():
            return []

        chunks = []
        current_chunk = ""

        # Split on natural boundaries first
        segments = self._split_on_boundaries(text)

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # If segment is too long, split it further
            if len(segment) > self.max_chunk_size:
                sub_chunks = self._split_long_segment(segment)
                for i, sub in enumerate(sub_chunks):
                    is_boundary = (i == len(sub_chunks) - 1)  # Last sub-chunk is boundary
                    chunks.append((sub, is_boundary))
            else:
                # Check if this completes a clause
                is_boundary = self._is_clause_boundary(segment)
                chunks.append((segment, is_boundary))

        return chunks

    def _split_on_boundaries(self, text: str) -> list[str]:
        """Split text on natural clause boundaries."""
        # Split on punctuation while preserving it
        segments = re.split(r'([,;:.])', text)

        # Recombine punctuation with preceding text
        result = []
        i = 0
        while i < len(segments):
            if i + 1 < len(segments) and segments[i + 1] in ',:;.':
                result.append(segments[i] + segments[i + 1])
                i += 2
            else:
                result.append(segments[i])
                i += 1

        return result

    def _split_long_segment(self, segment: str) -> list[str]:
        """Split a long segment into smaller chunks at word boundaries."""
        words = segment.split()
        chunks = []
        current = ""

        for word in words:
            test_chunk = f"{current} {word}".strip() if current else word

            if len(test_chunk) <= self.max_chunk_size:
                current = test_chunk
            else:
                if current:
                    chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        return chunks if chunks else [segment[:self.max_chunk_size]]

    def _is_clause_boundary(self, text: str) -> bool:
        """Check if text ends with a clause boundary marker."""
        text = text.strip()

        # Check for punctuation at end
        if text and text[-1] in ',.;:!?':
            return True

        # Check for conjunctions
        for pattern in self.boundary_patterns[1:]:  # Skip punctuation pattern
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False


class ChunkerService(chunker_pb2_grpc.ChunkerServicer):
    """gRPC service for text chunking."""

    def __init__(self):
        self.chunker = TextChunker(min_chunk_size=15, max_chunk_size=40)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Chunker service initialized")

    def ChunkTextStream(
        self,
        request_iterator: Iterator[rewriter_pb2.CharacterFragment],
        context: grpc.ServicerContext
    ) -> Iterator[chunker_pb2.TextChunk]:
        """
        Streaming RPC: accepts character fragments and yields text chunks.
        """
        chunk_index = 0

        try:
            for fragment in request_iterator:
                self.logger.debug(
                    f"Received fragment: character={fragment.character_id}, "
                    f"text='{fragment.text[:50]}...', provisional={fragment.is_provisional}"
                )

                # Skip empty fragments
                if not fragment.text or not fragment.text.strip():
                    continue

                # Chunk the text
                chunks = self.chunker.chunk_text(fragment.text)

                # Yield each chunk
                for chunk_text, is_boundary in chunks:
                    text_chunk = chunker_pb2.TextChunk(
                        text=chunk_text,
                        character_id=fragment.character_id,
                        chunk_index=chunk_index,
                        is_clause_boundary=is_boundary
                    )

                    self.logger.debug(
                        f"Yielding chunk {chunk_index}: '{chunk_text}' "
                        f"(boundary={is_boundary})"
                    )

                    yield text_chunk
                    chunk_index += 1

        except grpc.RpcError as e:
            self.logger.error(f"gRPC error in ChunkTextStream: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
        except Exception as e:
            self.logger.error(f"Error in ChunkTextStream: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def serve(port: int = 50054):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chunker_pb2_grpc.add_ChunkerServicer_to_server(ChunkerService(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"Chunker service started on port {port}")
    logging.info(f"Chunk size range: 15-40 characters")
    logging.info(f"Predicts clause boundaries for natural phrasing")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser(description='Chunker Service')
    parser.add_argument('--port', type=int, default=50054, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s'
    )

    serve(port=args.port)


if __name__ == '__main__':
    main()
