#!/usr/bin/env python3
"""
TTS-Streamer Service

Generates mel spectrograms from text using OpenVoice acoustic model.
Streams mel chunks as they are computed for low-latency synthesis.
"""

import sys
import argparse
import logging
import time
from concurrent import futures
from typing import Iterator

import grpc
import numpy as np

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import tts_pb2, tts_pb2_grpc
from libs.proto.generated.python import audio_pb2


class MockTTSEngine:
    """
    Mock TTS engine for testing pipeline flow.

    In production, this would be replaced with OpenVoice acoustic model.
    Generates synthetic mel spectrograms to simulate real TTS behavior.
    """

    def __init__(self, n_mels: int = 80, sample_rate: int = 22050):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = 256  # Standard hop length for mel spectrograms
        self.logger = logging.getLogger(__name__)

    def generate_mel_stream(
        self,
        text: str,
        embedding: np.ndarray,
        chunk_size_frames: int = 50
    ) -> Iterator[np.ndarray]:
        """
        Generate mel spectrogram chunks for text.

        Args:
            text: Text to synthesize
            embedding: Voice embedding vector
            chunk_size_frames: Number of frames per chunk

        Yields:
            Mel spectrogram chunks (n_mels x n_frames)
        """
        # Estimate duration: ~0.1s per character (typical for TTS)
        char_duration = 0.1
        total_duration = len(text) * char_duration

        # Calculate total frames needed
        total_frames = int(total_duration * self.sample_rate / self.hop_length)

        # Generate mel chunks
        frames_generated = 0
        while frames_generated < total_frames:
            frames_this_chunk = min(chunk_size_frames, total_frames - frames_generated)

            # Generate synthetic mel spectrogram
            # In production: actual TTS model inference here
            mel_chunk = self._generate_synthetic_mel(frames_this_chunk, embedding)

            yield mel_chunk

            frames_generated += frames_this_chunk

            # Simulate processing time (~50ms for 50 frames)
            time.sleep(0.05)

    def _generate_synthetic_mel(
        self,
        n_frames: int,
        embedding: np.ndarray
    ) -> np.ndarray:
        """
        Generate synthetic mel spectrogram for testing.

        In production, this would be OpenVoice model forward pass.
        """
        # Create base mel with some structure
        mel = np.random.randn(self.n_mels, n_frames).astype(np.float32) * 0.5

        # Add some harmonic structure based on embedding
        if embedding is not None and len(embedding) > 0:
            # Use embedding to modulate the mel
            embed_mean = np.mean(embedding)
            mel = mel * (0.5 + abs(embed_mean))

        # Normalize
        mel = np.clip(mel, -4.0, 4.0)

        return mel


class RealTTSEngine:
    """
    Real OpenVoice TTS engine (placeholder for future implementation).

    This would load the actual OpenVoice model and perform inference.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.logger = logging.getLogger(__name__)
        # TODO: Load OpenVoice model
        # self.model = load_openvoice_model(model_path, device)
        raise NotImplementedError("Real OpenVoice TTS not yet implemented")


class TTSStreamerService(tts_pb2_grpc.TTSStreamerServicer):
    """gRPC service for TTS mel generation."""

    def __init__(self, model_path: str = None, use_mock: bool = True):
        self.logger = logging.getLogger(__name__)

        if use_mock or model_path is None:
            self.logger.warning("Using mock TTS engine for testing")
            self.engine = MockTTSEngine(n_mels=80, sample_rate=22050)
            self.use_mock = True
        else:
            self.logger.info(f"Loading OpenVoice model from: {model_path}")
            try:
                self.engine = RealTTSEngine(model_path)
                self.use_mock = False
            except Exception as e:
                self.logger.error(f"Failed to load OpenVoice model: {e}")
                self.logger.warning("Falling back to mock TTS engine")
                self.engine = MockTTSEngine(n_mels=80, sample_rate=22050)
                self.use_mock = True

        self.logger.info("TTS-Streamer service initialized")

    def GenerateMel(
        self,
        request_iterator: Iterator[tts_pb2.TTSRequest],
        context: grpc.ServicerContext
    ) -> Iterator[audio_pb2.MelChunk]:
        """
        Streaming RPC: accepts TTS requests and yields mel chunks.
        """
        try:
            for request in request_iterator:
                self.logger.debug(
                    f"Received TTS request: character={request.character_id}, "
                    f"text='{request.text[:30]}...', chunk_index={request.chunk_index}"
                )

                # Parse embedding
                embedding = None
                if request.embedding_data:
                    embedding = np.frombuffer(
                        request.embedding_data,
                        dtype=np.float32
                    ).reshape(-1)
                    self.logger.debug(f"Embedding shape: {embedding.shape}")

                # Generate mel spectrograms
                for mel_chunk in self.engine.generate_mel_stream(
                    text=request.text,
                    embedding=embedding,
                    chunk_size_frames=50
                ):
                    # Convert to protobuf message
                    mel_msg = audio_pb2.MelChunk(
                        data=mel_chunk.tobytes(),
                        n_mels=mel_chunk.shape[0],
                        n_frames=mel_chunk.shape[1]
                    )

                    self.logger.debug(
                        f"Yielding mel chunk: {mel_chunk.shape[0]}x{mel_chunk.shape[1]}"
                    )

                    yield mel_msg

        except grpc.RpcError as e:
            self.logger.error(f"gRPC error in GenerateMel: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
        except Exception as e:
            self.logger.error(f"Error in GenerateMel: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def serve(port: int = 50055, model_path: str = None, use_mock: bool = True):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tts_pb2_grpc.add_TTSStreamerServicer_to_server(
        TTSStreamerService(model_path=model_path, use_mock=use_mock),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"TTS-Streamer service started on port {port}")
    if use_mock:
        logging.info("Using mock TTS engine - generates synthetic mel spectrograms")
    else:
        logging.info(f"Using OpenVoice model from: {model_path}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser(description='TTS-Streamer Service')
    parser.add_argument('--port', type=int, default=50055, help='Port to listen on')
    parser.add_argument('--model-path', type=str, help='Path to OpenVoice model')
    parser.add_argument('--use-mock', action='store_true', default=True,
                        help='Use mock TTS engine (default: True)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s'
    )

    serve(port=args.port, model_path=args.model_path, use_mock=args.use_mock)


if __name__ == '__main__':
    main()
