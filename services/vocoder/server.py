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
    Real vocoder using Griffin-Lim algorithm or pre-trained HiFi-GAN.

    Supports two modes:
    1. Griffin-Lim: Fast, CPU-based, no model needed (lower quality)
    2. HiFi-GAN: High quality, GPU-accelerated (requires model checkpoint)
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        sample_rate: int = 22050,
        hop_length: int = 256,
        use_griffin_lim: bool = False
    ):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.logger = logging.getLogger(__name__)

        if use_griffin_lim or model_path is None:
            # Use Griffin-Lim algorithm (no model needed)
            self._init_griffin_lim()
        else:
            # Try to load HiFi-GAN model
            try:
                self._init_hifigan()
            except Exception as e:
                self.logger.warning(f"Failed to load HiFi-GAN: {e}")
                self.logger.info("Falling back to Griffin-Lim")
                self._init_griffin_lim()

    def _init_griffin_lim(self):
        """Initialize Griffin-Lim vocoder."""
        try:
            import torch
            import torchaudio

            self.use_hifigan = False
            self.logger.info("✓ Initialized Griffin-Lim vocoder")

            # Create Griffin-Lim transform
            self.griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=1024,
                n_iter=32,
                win_length=1024,
                hop_length=self.hop_length,
                power=1.0
            )

        except ImportError as e:
            self.logger.error(f"torch/torchaudio not available: {e}")
            raise NotImplementedError("torch/torchaudio required. Run: pip install torch torchaudio")

    def _init_hifigan(self):
        """Initialize HiFi-GAN vocoder."""
        try:
            import torch

            self.logger.info(f"Loading HiFi-GAN model from: {self.model_path}")

            # Load HiFi-GAN checkpoint
            # This is a placeholder - in production, you'd load the actual model
            # For now, we'll fall back to Griffin-Lim
            raise NotImplementedError("HiFi-GAN model loading not yet implemented")

        except ImportError as e:
            self.logger.error(f"torch not available: {e}")
            raise

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
        try:
            # Convert mel to audio
            audio = self._mel_to_audio(mel)

            # Stream audio in chunks
            total_samples = len(audio)
            samples_sent = 0

            while samples_sent < total_samples:
                samples_this_chunk = min(chunk_size_samples, total_samples - samples_sent)
                pcm_chunk = audio[samples_sent:samples_sent + samples_this_chunk]

                yield pcm_chunk
                samples_sent += samples_this_chunk

                # Small delay to simulate streaming
                duration = samples_this_chunk / self.sample_rate
                time.sleep(duration * 0.05)  # 5% of real-time for processing

        except Exception as e:
            self.logger.error(f"Vocoder error: {e}", exc_info=True)
            # Return silence on error
            yield np.zeros(chunk_size_samples, dtype=np.float32)

    def _mel_to_audio(self, mel: np.ndarray) -> np.ndarray:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel: Mel spectrogram (n_mels x n_frames)

        Returns:
            Audio waveform (float32 array)
        """
        try:
            import torch

            if not self.use_hifigan:
                # Use Griffin-Lim
                # Convert mel from log scale back to linear
                mel_linear = np.exp(mel)

                # Convert to torch tensor
                mel_tensor = torch.FloatTensor(mel_linear).unsqueeze(0)

                # Apply Griffin-Lim
                audio_tensor = self.griffin_lim(mel_tensor)

                # Convert to numpy
                audio = audio_tensor.squeeze(0).cpu().numpy()

                # Normalize to [-1, 1]
                audio = audio / np.max(np.abs(audio) + 1e-8)

                return audio.astype(np.float32)

            else:
                # Use HiFi-GAN (not implemented yet)
                raise NotImplementedError("HiFi-GAN inference not implemented")

        except Exception as e:
            self.logger.error(f"Mel conversion error: {e}")
            # Return silence
            estimated_samples = mel.shape[1] * self.hop_length
            return np.zeros(estimated_samples, dtype=np.float32)


class VocoderService(vocoder_pb2_grpc.VocoderServicer):
    """gRPC service for mel-to-PCM conversion."""

    def __init__(
        self,
        model_path: str = None,
        use_mock: bool = True,
        use_griffin_lim: bool = False,
        device: str = 'cuda'
    ):
        self.logger = logging.getLogger(__name__)

        if use_mock:
            self.logger.warning("Using mock vocoder for testing")
            self.vocoder = MockVocoder(sample_rate=22050, hop_length=256)
            self.use_mock = True
        elif use_griffin_lim:
            self.logger.info("Using Griffin-Lim vocoder")
            try:
                self.vocoder = RealVocoder(
                    model_path=None,
                    use_griffin_lim=True,
                    device=device,
                    sample_rate=22050,
                    hop_length=256
                )
                self.use_mock = False
            except (ImportError, NotImplementedError) as e:
                self.logger.error(f"Failed to initialize Griffin-Lim: {e}")
                self.logger.warning("Falling back to mock vocoder")
                self.vocoder = MockVocoder(sample_rate=22050, hop_length=256)
                self.use_mock = True
        else:
            self.logger.info(f"Loading HiFi-GAN model from: {model_path}")
            try:
                self.vocoder = RealVocoder(
                    model_path=model_path,
                    device=device,
                    sample_rate=22050,
                    hop_length=256,
                    use_griffin_lim=False
                )
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


def serve(
    port: int = 50056,
    model_path: str = None,
    use_mock: bool = True,
    use_griffin_lim: bool = False,
    device: str = 'cuda'
):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vocoder_pb2_grpc.add_VocoderServicer_to_server(
        VocoderService(
            model_path=model_path,
            use_mock=use_mock,
            use_griffin_lim=use_griffin_lim,
            device=device
        ),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"Vocoder service started on port {port}")
    if use_mock:
        logging.info("Using mock vocoder - generates synthetic PCM audio")
    elif use_griffin_lim:
        logging.info("Using Griffin-Lim vocoder (CPU-based)")
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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--use-mock', action='store_true', default=True,
                        help='Use mock vocoder (default: True)')
    parser.add_argument('--use-griffin-lim', action='store_true',
                        help='Use Griffin-Lim vocoder (no model needed)')
    parser.add_argument('--use-real', dest='use_mock', action='store_false',
                        help='Use real vocoder (Griffin-Lim or HiFi-GAN)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s'
    )

    serve(
        port=args.port,
        model_path=args.model_path,
        use_mock=args.use_mock,
        use_griffin_lim=args.use_griffin_lim,
        device=args.device
    )


if __name__ == '__main__':
    main()
