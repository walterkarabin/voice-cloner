"""
STT-Whisper Service

Streaming speech-to-text service using Whisper for real-time transcription.
Accepts audio chunks and returns partial transcripts as they become available.
"""

import os
import sys
import grpc
import time
import logging
import numpy as np
from concurrent import futures
from typing import Iterator, Optional
from queue import Queue, Empty
from threading import Thread, Lock

sys.path.insert(0, '/workspace')

from libs.proto.generated.python import stt_pb2, stt_pb2_grpc, audio_pb2, common_pb2
from libs.audio import bytes_to_float32, AudioFramer


class WhisperSTTService(stt_pb2_grpc.WhisperSTTServicer):
    """
    gRPC service for streaming speech-to-text using Whisper.
    """

    def __init__(
        self,
        model_path: str = "/models/whisper-small",
        model_size: str = "small",
        sample_rate: int = 16000
    ):
        """
        Initialize the Whisper STT service.

        Args:
            model_path: Path to Whisper model
            model_size: Model size (tiny, base, small, medium, large)
            sample_rate: Expected audio sample rate
        """
        self.model_path = model_path
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

        # Load Whisper model
        self._load_model()

        # Statistics
        self.total_requests = 0
        self.active_streams = 0
        self.stats_lock = Lock()

    def _load_model(self):
        """Load the Whisper model."""
        self.logger.info(f"Loading Whisper model: {self.model_size}")

        try:
            # Try to import faster-whisper (optimized Whisper implementation)
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_size,
                device="cuda",  # Use GPU if available
                compute_type="float16"
            )
            self.logger.info(f"✓ Loaded faster-whisper model: {self.model_size}")
            self.use_faster_whisper = True

        except ImportError:
            self.logger.warning("faster-whisper not available, trying openai-whisper")

            try:
                import whisper
                self.model = whisper.load_model(self.model_size)
                self.logger.info(f"✓ Loaded openai-whisper model: {self.model_size}")
                self.use_faster_whisper = False

            except ImportError:
                self.logger.error(
                    "openai-whisper not available. "
                    "Using mock transcription for testing."
                )
                self.model = None
                self.use_faster_whisper = False

            except Exception as e:
                # Catch network errors, permission errors, etc.
                self.logger.warning(
                    f"Failed to load openai-whisper model: {type(e).__name__}: {str(e)[:100]}. "
                    "Using mock transcription for testing."
                )
                self.model = None
                self.use_faster_whisper = False

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> tuple[str, float]:
        """
        Transcribe a single audio chunk.

        Args:
            audio: Audio samples (float32)
            language: Language code

        Returns:
            Tuple of (text, confidence)
        """
        if self.model is None:
            # Mock transcription for testing
            duration = len(audio) / self.sample_rate
            text = f"[Mock transcript for {duration:.2f}s audio]"
            return text, 0.85

        if self.use_faster_whisper:
            # Using faster-whisper
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=1,  # Faster decoding
                best_of=1,
                vad_filter=True
            )

            text = " ".join([segment.text for segment in segments])
            confidence = info.language_probability if hasattr(info, 'language_probability') else 0.9

        else:
            # Using openai-whisper
            result = self.model.transcribe(
                audio,
                language=language,
                fp16=True
            )
            text = result["text"]
            confidence = 0.9  # OpenAI Whisper doesn't provide confidence scores

        return text.strip(), confidence

    def StreamAudio(
        self,
        request_iterator: Iterator[audio_pb2.AudioChunk],
        context: grpc.ServicerContext
    ) -> Iterator[stt_pb2.PartialTranscript]:
        """
        Bidirectional streaming RPC for audio transcription.

        Args:
            request_iterator: Stream of audio chunks
            context: gRPC context

        Yields:
            Partial transcripts as they become available
        """
        with self.stats_lock:
            self.total_requests += 1
            self.active_streams += 1
            stream_id = self.total_requests

        self.logger.info(f"Stream {stream_id}: Started")

        try:
            # Buffer for accumulating audio
            audio_buffer = []
            sequence_num = 0
            start_time = time.time()

            # Create framer for processing audio in chunks
            # Process in ~1 second chunks for better transcription quality
            framer = AudioFramer.from_duration(
                frame_duration_ms=1000,
                sample_rate=self.sample_rate
            )

            for chunk in request_iterator:
                # Convert bytes to float32
                audio_samples = bytes_to_float32(
                    chunk.data,
                    sample_width=chunk.sample_width
                )

                # Resample if needed
                if chunk.sample_rate != self.sample_rate:
                    from libs.audio import AudioResampler
                    resampler = AudioResampler(
                        input_rate=chunk.sample_rate,
                        output_rate=self.sample_rate
                    )
                    audio_samples = resampler.resample(audio_samples)

                # Add to framer
                for frame in framer.add_samples(audio_samples):
                    # Transcribe frame
                    text, confidence = self._transcribe_chunk(frame)

                    if text:
                        # Calculate timestamps
                        current_time = time.time()
                        elapsed_ms = int((current_time - start_time) * 1000)

                        # Create partial transcript
                        transcript = stt_pb2.PartialTranscript(
                            text=text,
                            confidence=confidence,
                            is_final=False,
                            start_time_ms=elapsed_ms - 1000,
                            end_time_ms=elapsed_ms,
                            metadata=common_pb2.Metadata(
                                sequence_number=sequence_num
                            )
                        )

                        sequence_num += 1
                        yield transcript

                        self.logger.debug(
                            f"Stream {stream_id}: Transcript [{confidence:.2f}]: {text}"
                        )

            # Process any remaining audio
            remaining = framer.flush(pad=True)
            if remaining is not None and len(remaining) > 0:
                text, confidence = self._transcribe_chunk(remaining)

                if text:
                    current_time = time.time()
                    elapsed_ms = int((current_time - start_time) * 1000)

                    transcript = stt_pb2.PartialTranscript(
                        text=text,
                        confidence=confidence,
                        is_final=True,  # Mark final transcript
                        start_time_ms=elapsed_ms - 1000,
                        end_time_ms=elapsed_ms,
                        metadata=common_pb2.Metadata(
                            sequence_number=sequence_num
                        )
                    )

                    yield transcript

            self.logger.info(
                f"Stream {stream_id}: Completed. "
                f"Generated {sequence_num + 1} transcripts"
            )

        except Exception as e:
            self.logger.error(f"Stream {stream_id}: Error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Transcription error: {str(e)}")

        finally:
            with self.stats_lock:
                self.active_streams -= 1


def serve(
    port: int = 50052,
    model_size: str = "small",
    model_path: str = "/models/whisper"
):
    """
    Start the Whisper STT gRPC server.

    Args:
        port: Port to listen on
        model_size: Whisper model size
        model_path: Path to model files
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    # Add service
    whisper_service = WhisperSTTService(
        model_path=model_path,
        model_size=model_size
    )
    stt_pb2_grpc.add_WhisperSTTServicer_to_server(whisper_service, server)

    # Start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f"Whisper STT service started on port {port}")
    logger.info(f"Model: {model_size}, Sample rate: 16000 Hz")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down Whisper STT service")
        server.stop(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Whisper STT Streaming Service'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=50052,
        help='Port to listen on (default: 50052)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: small)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/models/whisper',
        help='Path to Whisper model files'
    )

    args = parser.parse_args()
    serve(
        port=args.port,
        model_size=args.model_size,
        model_path=args.model_path
    )
