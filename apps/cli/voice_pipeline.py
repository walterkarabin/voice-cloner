#!/usr/bin/env python3
"""
Voice Pipeline CLI

Orchestrates the character voice pipeline with live audio input/output.
Coordinates all services and manages audio streaming.
"""

import sys
import argparse
import logging
import time
import threading
import queue
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd

# Add workspace to path
sys.path.insert(0, '/workspace')

from services_loader import (
    EmbedLoaderClient,
    WhisperSTTClient,
    LLMRewriterClient,
    ChunkerClient,
    TTSStreamerClient,
    VocoderClient,
    AudioOutClient
)


class VoicePipelineOrchestrator:
    """
    Coordinates the full voice pipeline from microphone input to speaker output.

    Pipeline flow:
    Microphone → STT → Rewriter → Chunker → TTS → Vocoder → Audio Out → Speaker
    """

    def __init__(
        self,
        character_id: str = 'yoda',
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100
    ):
        self.character_id = character_id
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        self.logger = logging.getLogger(__name__)
        self.running = False

        # Audio queues
        self.audio_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()

        # Service clients
        self.embed_client = None
        self.stt_client = None
        self.rewriter_client = None
        self.chunker_client = None
        self.tts_client = None
        self.vocoder_client = None
        self.audio_out_client = None

        # Character embedding (preload)
        self.character_embedding = None

    def connect_services(
        self,
        embed_host: str = 'localhost',
        stt_host: str = 'localhost',
        rewriter_host: str = 'localhost',
        chunker_host: str = 'localhost',
        tts_host: str = 'localhost',
        vocoder_host: str = 'localhost',
        audio_out_host: str = 'localhost'
    ):
        """Connect to all pipeline services."""
        self.logger.info("Connecting to services...")

        try:
            self.embed_client = EmbedLoaderClient(host=embed_host, port=50051)
            self.stt_client = WhisperSTTClient(host=stt_host, port=50052)
            self.rewriter_client = LLMRewriterClient(host=rewriter_host, port=50053)
            self.chunker_client = ChunkerClient(host=chunker_host, port=50054)
            self.tts_client = TTSStreamerClient(host=tts_host, port=50055)
            self.vocoder_client = VocoderClient(host=vocoder_host, port=50056)
            self.audio_out_client = AudioOutClient(host=audio_out_host, port=50057)

            self.logger.info("✓ Connected to all services")

            # Preload character embedding
            self.logger.info(f"Loading embedding for character: {self.character_id}")
            self.character_embedding = self.embed_client.get_embedding(self.character_id)
            self.logger.info(f"✓ Loaded embedding: shape={self.character_embedding.shape}")

        except Exception as e:
            self.logger.error(f"Failed to connect to services: {e}")
            raise

    def list_audio_devices(self):
        """List available audio input and output devices."""
        devices = sd.query_devices()

        print("\n=== Audio Input Devices ===")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"[{i}] {device['name']}{default}")
                print(f"    Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']} Hz")

        print("\n=== Audio Output Devices ===")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[1] else ""
                print(f"[{i}] {device['name']}{default}")
                print(f"    Channels: {device['max_output_channels']}, "
                      f"Sample Rate: {device['default_samplerate']} Hz")
        print()

    def audio_input_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            self.logger.warning(f"Input status: {status}")

        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata[:, 0]

        # Add to queue
        self.audio_input_queue.put(audio.copy())

    def audio_output_callback(self, outdata, frames, time_info, status):
        """Callback for audio output stream."""
        if status:
            self.logger.warning(f"Output status: {status}")

        try:
            # Get audio from queue
            audio = self.audio_output_queue.get_nowait()

            # Ensure correct size
            if len(audio) < frames:
                audio = np.pad(audio, (0, frames - len(audio)))
            elif len(audio) > frames:
                audio = audio[:frames]

            # Write to output (mono to stereo if needed)
            outdata[:] = audio.reshape(-1, 1)

        except queue.Empty:
            # No audio available, output silence
            outdata.fill(0)

    def process_audio_pipeline(self):
        """
        Main processing loop: reads audio from queue and processes through pipeline.
        """
        self.logger.info("Starting audio pipeline processing...")

        audio_buffer = []
        buffer_duration = 1.0  # Process in 1-second chunks

        try:
            while self.running:
                try:
                    # Get audio chunk from input queue
                    audio_chunk = self.audio_input_queue.get(timeout=0.1)
                    audio_buffer.extend(audio_chunk)

                    # Process when buffer is full
                    buffer_samples = int(self.sample_rate * buffer_duration)
                    if len(audio_buffer) >= buffer_samples:
                        audio_data = np.array(audio_buffer[:buffer_samples], dtype=np.float32)
                        audio_buffer = audio_buffer[buffer_samples:]

                        # Process through pipeline
                        self._process_audio_chunk(audio_data)

                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing audio: {e}", exc_info=True)

        finally:
            self.logger.info("Audio pipeline processing stopped")

    def _process_audio_chunk(self, audio: np.ndarray):
        """Process a single audio chunk through the full pipeline."""
        try:
            # 1. STT: Audio → Text
            def audio_generator():
                yield audio

            transcripts = list(self.stt_client.transcribe_stream(audio_generator()))

            if not transcripts:
                return

            # Get the latest transcript
            transcript = transcripts[-1]
            self.logger.info(f"STT: '{transcript['text']}'")

            # 2. Rewriter: Text → Character style
            rewrite_input = [{
                'text': transcript['text'],
                'confidence': transcript.get('confidence', 0.9),
                'is_final': transcript.get('is_final', True),
                'start_time_ms': 0,
                'end_time_ms': int(len(audio) / self.sample_rate * 1000),
                'sequence': 0
            }]

            rewritten = list(self.rewriter_client.rewrite_stream(
                rewrite_input,
                character_id=self.character_id
            ))

            if not rewritten:
                return

            rewritten_text = rewritten[-1]['text']
            self.logger.info(f"Rewriter ({self.character_id}): '{rewritten_text}'")

            # 3. Chunker: Split into TTS chunks
            chunk_input = [{
                'text': rewritten_text,
                'character_id': self.character_id,
                'is_provisional': False
            }]

            text_chunks = list(self.chunker_client.chunk_stream(chunk_input))
            self.logger.info(f"Chunker: {len(text_chunks)} chunks")

            # 4-6. TTS → Vocoder → Audio Out
            for chunk in text_chunks:
                # TTS: Text → Mel
                tts_requests = [{
                    'text': chunk['text'],
                    'embedding': self.character_embedding,
                    'character_id': self.character_id,
                    'chunk_index': chunk['chunk_index']
                }]

                mel_chunks = list(self.tts_client.generate_mel_stream(tts_requests))

                # Vocoder: Mel → PCM
                mel_arrays = [m['data'] for m in mel_chunks]
                pcm_chunks = list(self.vocoder_client.generate_pcm_stream(mel_arrays))

                # Audio Out: Play PCM
                pcm_arrays = [p['data'] for p in pcm_chunks]
                if pcm_arrays:
                    self.audio_out_client.play_stream(pcm_arrays, sample_rate=22050)
                    self.logger.info(f"Playing chunk: '{chunk['text']}'")

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)

    def start(self):
        """Start the voice pipeline."""
        self.logger.info("Starting voice pipeline...")
        self.running = True

        # Start pipeline processing thread
        self.pipeline_thread = threading.Thread(target=self.process_audio_pipeline, daemon=True)
        self.pipeline_thread.start()

        # Start audio input stream
        self.logger.info(f"Opening audio input (device={self.input_device}, rate={self.sample_rate}Hz)")
        self.input_stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_input_callback
        )
        self.input_stream.start()

        self.logger.info("✓ Voice pipeline started")
        self.logger.info(f"  Character: {self.character_id}")
        self.logger.info(f"  Input device: {self.input_device or 'default'}")
        self.logger.info(f"  Speak into your microphone...")

    def stop(self):
        """Stop the voice pipeline."""
        self.logger.info("Stopping voice pipeline...")
        self.running = False

        if hasattr(self, 'input_stream'):
            self.input_stream.stop()
            self.input_stream.close()

        if hasattr(self, 'pipeline_thread'):
            self.pipeline_thread.join(timeout=5)

        # Close all clients
        if self.embed_client:
            self.embed_client.close()
        if self.stt_client:
            self.stt_client.close()
        if self.rewriter_client:
            self.rewriter_client.close()
        if self.chunker_client:
            self.chunker_client.close()
        if self.tts_client:
            self.tts_client.close()
        if self.vocoder_client:
            self.vocoder_client.close()
        if self.audio_out_client:
            self.audio_out_client.close()

        self.logger.info("✓ Voice pipeline stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Character Voice Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List audio devices
  python3 voice_pipeline.py --list-devices

  # Run with default devices
  python3 voice_pipeline.py --character yoda

  # Run with specific input device
  python3 voice_pipeline.py --character vader --input-device 2

  # Connect to remote services
  python3 voice_pipeline.py --character obi-wan --host 192.168.1.100
"""
    )

    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--character', type=str, default='yoda',
                        choices=['yoda', 'vader', 'obi-wan', 'leia'],
                        help='Character voice to use')
    parser.add_argument('--input-device', type=int, default=None,
                        help='Audio input device index')
    parser.add_argument('--output-device', type=int, default=None,
                        help='Audio output device index')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate (Hz)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host for all services (default: localhost)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # List devices and exit
    if args.list_devices:
        orchestrator = VoicePipelineOrchestrator()
        orchestrator.list_audio_devices()
        return

    # Create orchestrator
    orchestrator = VoicePipelineOrchestrator(
        character_id=args.character,
        input_device=args.input_device,
        output_device=args.output_device,
        sample_rate=args.sample_rate
    )

    try:
        # Connect to services
        orchestrator.connect_services(
            embed_host=args.host,
            stt_host=args.host,
            rewriter_host=args.host,
            chunker_host=args.host,
            tts_host=args.host,
            vocoder_host=args.host,
            audio_out_host=args.host
        )

        # Start pipeline
        orchestrator.start()

        # Run until interrupted
        print("\n" + "="*60)
        print(f"  Voice Pipeline Active - Character: {args.character}")
        print("  Press Ctrl+C to stop")
        print("="*60 + "\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        orchestrator.stop()


if __name__ == '__main__':
    main()
