"""
Whisper STT Client

Example client for testing the streaming STT service.
"""

import sys
import grpc
import time
import logging
import numpy as np
from typing import Iterator

sys.path.insert(0, '/workspace')

from libs.proto.generated.python import stt_pb2, stt_pb2_grpc, audio_pb2, common_pb2
from libs.audio import float32_to_bytes


class WhisperSTTClient:
    """Client for the Whisper STT service."""

    def __init__(self, host: str = 'localhost', port: int = 50052):
        """
        Initialize client.

        Args:
            host: Service host
            port: Service port
        """
        self.address = f'{host}:{port}'
        self.channel = grpc.insecure_channel(self.address)
        self.stub = stt_pb2_grpc.WhisperSTTStub(self.channel)
        self.logger = logging.getLogger(__name__)

    def transcribe_stream(
        self,
        audio_generator: Iterator[np.ndarray],
        sample_rate: int = 16000,
        sample_width: int = 2
    ) -> Iterator[dict]:
        """
        Transcribe streaming audio.

        Args:
            audio_generator: Iterator yielding audio chunks (float32)
            sample_rate: Audio sample rate
            sample_width: Bytes per sample

        Yields:
            Dictionary with transcript information
        """
        def audio_request_generator():
            for audio_chunk in audio_generator:
                # Convert to bytes
                audio_bytes = float32_to_bytes(audio_chunk, sample_width)

                # Create AudioChunk message
                chunk = audio_pb2.AudioChunk(
                    data=audio_bytes,
                    sample_rate=sample_rate,
                    channels=1,
                    sample_width=sample_width,
                    metadata=common_pb2.Metadata()
                )

                yield chunk

        try:
            # Call streaming RPC
            responses = self.stub.StreamAudio(audio_request_generator())

            # Process responses
            for response in responses:
                transcript = {
                    "text": response.text,
                    "confidence": response.confidence,
                    "is_final": response.is_final,
                    "start_time_ms": response.start_time_ms,
                    "end_time_ms": response.end_time_ms,
                    "sequence": response.metadata.sequence_number
                }

                self.logger.info(
                    f"[{transcript['sequence']}] "
                    f"{'FINAL' if transcript['is_final'] else 'PARTIAL'}: "
                    f"{transcript['text']} (conf: {transcript['confidence']:.2f})"
                )

                yield transcript

        except grpc.RpcError as e:
            self.logger.error(f"RPC failed: {e.code()} - {e.details()}")
            raise

    def transcribe_file(self, audio_path: str) -> list:
        """
        Transcribe audio from a file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of transcript dictionaries
        """
        import soundfile as sf

        # Read audio file
        audio, sample_rate = sf.read(audio_path, dtype='float32')

        # Split into chunks (simulate streaming)
        chunk_size = int(sample_rate * 0.5)  # 500ms chunks

        def audio_generator():
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                yield chunk
                time.sleep(0.05)  # Simulate real-time streaming

        # Transcribe
        transcripts = list(self.transcribe_stream(
            audio_generator(),
            sample_rate=sample_rate
        ))

        return transcripts

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()


def generate_test_audio(duration_sec: float = 5.0, sample_rate: int = 16000):
    """
    Generate test audio for demonstration.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Sample rate

    Yields:
        Audio chunks
    """
    # Generate sine wave tone
    freq = 440.0  # A4 note
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.3

    # Split into chunks
    chunk_size = int(sample_rate * 0.5)  # 500ms chunks
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i + chunk_size]
        time.sleep(0.5)  # Real-time simulation


def main():
    """Test the Whisper STT client."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== Whisper STT Client Test ===\n")

    # Create client
    client = WhisperSTTClient()

    print("Streaming test audio (5 seconds)...")
    print("Note: This is a tone, actual speech will produce real transcripts\n")

    # Generate and transcribe test audio
    transcripts = []
    try:
        for transcript in client.transcribe_stream(generate_test_audio()):
            transcripts.append(transcript)

    except grpc.RpcError as e:
        print(f"Error: {e.details()}")
    except Exception as e:
        print(f"Error: {e}")

    print(f"\n✓ Received {len(transcripts)} transcript(s)")

    if transcripts:
        print("\nFinal transcripts:")
        for t in transcripts:
            status = "FINAL" if t['is_final'] else "PARTIAL"
            print(f"  [{status}] {t['text']}")

    client.close()


if __name__ == '__main__':
    main()
