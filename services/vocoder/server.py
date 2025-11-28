#!/usr/bin/env python3
"""
Vocoder Service

Converts mel spectrograms to PCM audio using HiFi-GAN.
Streams PCM chunks as they are generated for low-latency playback.
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

from libs.proto.generated.python import vocoder_pb2, vocoder_pb2_grpc
from libs.proto.generated.python import audio_pb2


class MockVocoder:
    """
    Mock vocoder for testing pipeline flow.

    In production, this would be replaced with HiFi-GAN model.
    Generates synthetic PCM audio to simulate real vocoder behavior.
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.logger = logging.getLogger(__name__)

    def mel_to_pcm_stream(
        self,
        mel: np.ndarray,
        chunk_size_samples: int = 4096
    ) -> Iterator[np.ndarray]:
        """
        Convert mel spectrogram to PCM audio chunks.

        Args:
            mel: Mel spectrogram (n_mels x n_frames)
            chunk_size_samples: Number of PCM samples per chunk

        Yields:
            PCM audio chunks (float32 array)
        """
        # Calculate total audio samples from mel frames
        total_samples = mel.shape[1] * self.hop_length

        # Generate PCM chunks
        samples_generated = 0
        while samples_generated < total_samples:
            samples_this_chunk = min(chunk_size_samples, total_samples - samples_generated)

            # Generate synthetic PCM audio
            # In production: actual HiFi-GAN inference here
            pcm_chunk = self._generate_synthetic_pcm(samples_this_chunk, mel)

            yield pcm_chunk

            samples_generated += samples_this_chunk

            # Simulate processing time (~20ms per 4096 samples at 22050Hz)
            duration = samples_this_chunk / self.sample_rate
            time.sleep(duration * 0.1)  # 10% of real-time for processing

    def _generate_synthetic_pcm(
        self,
        n_samples: int,
        mel: np.ndarray
    ) -> np.ndarray:
        """
        Generate synthetic PCM audio for testing.

        In production, this would be HiFi-GAN model forward pass.
        """
        # Create synthetic audio with some structure
        # Use mel statistics to modulate the audio
        mel_mean = np.mean(mel)
        mel_std = np.std(mel)

        # Generate base waveform
        # Create a simple tone with some noise
        t = np.arange(n_samples) / self.sample_rate
        frequency = 200 + abs(mel_mean) * 50  # Vary frequency based on mel
        amplitude = 0.1 + abs(mel_std) * 0.05

        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        noise = np.random.randn(n_samples) * 0.01

        pcm = (tone + noise).astype(np.float32)

        # Normalize to [-1, 1]
        pcm = np.clip(pcm, -1.0, 1.0)

        return pcm


class RealVocoder:
    """
    Real HiFi-GAN vocoder (placeholder for future implementation).

    This would load the actual HiFi-GAN model and perform inference.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.logger = logging.getLogger(__name__)
        # TODO: Load HiFi-GAN model
        # self.model = load_hifigan_model(model_path, device)
        raise NotImplementedError("Real HiFi-GAN vocoder not yet implemented")


class VocoderService(vocoder_pb2_grpc.VocoderServicer):
    """gRPC service for mel-to-PCM conversion."""

    def __init__(self, model_path: str = None, use_mock: bool = True):
        self.logger = logging.getLogger(__name__)

        if use_mock or model_path is None:
            self.logger.warning("Using mock vocoder for testing")
            self.vocoder = MockVocoder(sample_rate=22050, hop_length=256)
            self.use_mock = True
        else:
            self.logger.info(f"Loading HiFi-GAN model from: {model_path}")
            try:
                self.vocoder = RealVocoder(model_path)
                self.use_mock = False
            except Exception as e:
                self.logger.error(f"Failed to load HiFi-GAN model: {e}")
                self.logger.warning("Falling back to mock vocoder")
                self.vocoder = MockVocoder(sample_rate=22050, hop_length=256)
                self.use_mock = True

        self.logger.info("Vocoder service initialized")

    def GeneratePCM(
        self,
        request_iterator: Iterator[audio_pb2.MelChunk],
        context: grpc.ServicerContext
    ) -> Iterator[audio_pb2.PCMChunk]:
        """
        Streaming RPC: accepts mel chunks and yields PCM chunks.
        """
        try:
            for mel_msg in request_iterator:
                self.logger.debug(
                    f"Received mel chunk: {mel_msg.n_mels}x{mel_msg.n_frames}"
                )

                # Parse mel data
                mel_data = np.frombuffer(mel_msg.data, dtype=np.float32).reshape(
                    mel_msg.n_mels, mel_msg.n_frames
                )

                # Generate PCM audio
                for pcm_chunk in self.vocoder.mel_to_pcm_stream(
                    mel=mel_data,
                    chunk_size_samples=4096
                ):
                    # Convert to int16 for transmission
                    pcm_int16 = (pcm_chunk * 32767.0).astype(np.int16)

                    # Convert to protobuf message
                    pcm_msg = audio_pb2.PCMChunk(
                        data=pcm_int16.tobytes(),
                        sample_rate=self.vocoder.sample_rate,
                        channels=1,  # Mono
                        sample_width=2  # 16-bit
                    )

                    self.logger.debug(
                        f"Yielding PCM chunk: {len(pcm_chunk)} samples"
                    )

                    yield pcm_msg

        except grpc.RpcError as e:
            self.logger.error(f"gRPC error in GeneratePCM: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
        except Exception as e:
            self.logger.error(f"Error in GeneratePCM: {e}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def serve(port: int = 50056, model_path: str = None, use_mock: bool = True):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vocoder_pb2_grpc.add_VocoderServicer_to_server(
        VocoderService(model_path=model_path, use_mock=use_mock),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"Vocoder service started on port {port}")
    if use_mock:
        logging.info("Using mock vocoder - generates synthetic PCM audio")
    else:
        logging.info(f"Using HiFi-GAN model from: {model_path}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser(description='Vocoder Service')
    parser.add_argument('--port', type=int, default=50056, help='Port to listen on')
    parser.add_argument('--model-path', type=str, help='Path to HiFi-GAN model')
    parser.add_argument('--use-mock', action='store_true', default=True,
                        help='Use mock vocoder (default: True)')
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
