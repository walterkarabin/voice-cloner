#!/usr/bin/env python3
"""
Audio-Out Service

Plays PCM audio chunks with crossfading to prevent clicks/pops.
Manages playback buffer and provides volume/playback controls.
"""

import sys
import argparse
import logging
import threading
import queue
import time
from concurrent import futures
from typing import Iterator

import grpc
import numpy as np

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

from libs.proto.generated.python import audio_out_pb2, audio_out_pb2_grpc
from libs.proto.generated.python import audio_pb2, common_pb2


class AudioPlayer:
    """
    Audio playback engine with crossfading and buffering.

    Manages a jitter buffer and applies crossfading between chunks
    to prevent clicks and pops.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        crossfade_samples: int = 441,  # ~20ms at 22050Hz
        max_buffer_ms: int = 60
    ):
        self.sample_rate = sample_rate
        self.crossfade_samples = crossfade_samples
        self.max_buffer_samples = int(max_buffer_ms * sample_rate / 1000)

        self.buffer = queue.Queue()
        self.is_playing = False
        self.is_muted = False
        self.volume = 1.0

        self.playback_thread = None
        self.stop_event = threading.Event()
        self.last_chunk = None

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the playback thread."""
        if self.playback_thread is not None and self.playback_thread.is_alive():
            return

        self.stop_event.clear()
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        self.is_playing = True
        self.logger.info("Playback started")

    def stop(self):
        """Stop the playback thread."""
        self.stop_event.set()
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        self.is_playing = False
        self.logger.info("Playback stopped")

    def add_chunk(self, pcm_data: np.ndarray):
        """Add PCM chunk to playback buffer."""
        if self.buffer.qsize() * len(pcm_data) > self.max_buffer_samples:
            self.logger.warning("Buffer full, dropping chunk")
            return

        # Apply crossfade with previous chunk
        if self.last_chunk is not None and len(self.last_chunk) > 0:
            pcm_data = self._apply_crossfade(self.last_chunk, pcm_data)

        self.buffer.put(pcm_data)
        self.last_chunk = pcm_data[-self.crossfade_samples:] if len(pcm_data) >= self.crossfade_samples else pcm_data

    def _apply_crossfade(self, prev_chunk: np.ndarray, curr_chunk: np.ndarray) -> np.ndarray:
        """
        Apply crossfading between chunks.

        Takes the last crossfade_samples from prev_chunk and first crossfade_samples
        from curr_chunk, blending them smoothly.
        """
        fade_len = min(self.crossfade_samples, len(prev_chunk), len(curr_chunk))

        if fade_len == 0:
            return curr_chunk

        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_len).astype(np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len).astype(np.float32)

        # Apply crossfade to the beginning of current chunk
        prev_tail = prev_chunk[-fade_len:]
        curr_head = curr_chunk[:fade_len]

        crossfaded = prev_tail * fade_out + curr_head * fade_in

        # Combine: crossfaded beginning + rest of current chunk
        result = np.concatenate([crossfaded, curr_chunk[fade_len:]])

        return result

    def _playback_loop(self):
        """
        Main playback loop (runs in separate thread).

        In a real implementation, this would use PyAudio or sounddevice
        to output to speakers. For now, it just simulates playback timing.
        """
        self.logger.info("Playback loop started")

        try:
            while not self.stop_event.is_set():
                try:
                    # Get chunk from buffer (with timeout)
                    chunk = self.buffer.get(timeout=0.1)

                    if not self.is_muted:
                        # Apply volume
                        chunk = chunk * self.volume

                        # Simulate playback
                        # In production: output to audio device
                        # e.g., stream.write(chunk.astype(np.int16).tobytes())
                        duration = len(chunk) / self.sample_rate
                        self.logger.debug(f"Playing chunk: {len(chunk)} samples ({duration:.3f}s)")

                        # Simulate real-time playback
                        time.sleep(duration)
                    else:
                        self.logger.debug("Chunk skipped (muted)")

                except queue.Empty:
                    continue

        except Exception as e:
            self.logger.error(f"Error in playback loop: {e}", exc_info=True)
        finally:
            self.logger.info("Playback loop ended")

    def get_status(self) -> dict:
        """Get current playback status."""
        buffer_samples = self.buffer.qsize() * 4096  # Approximate
        buffer_ms = int(buffer_samples / self.sample_rate * 1000)

        return {
            'is_playing': self.is_playing,
            'is_muted': self.is_muted,
            'volume': self.volume,
            'buffer_size_ms': buffer_ms
        }

    def set_volume(self, volume: float):
        """Set playback volume (0.0 - 1.0)."""
        self.volume = np.clip(volume, 0.0, 1.0)
        self.logger.info(f"Volume set to {self.volume:.2f}")

    def mute(self):
        """Mute playback."""
        self.is_muted = True
        self.logger.info("Muted")

    def unmute(self):
        """Unmute playback."""
        self.is_muted = False
        self.logger.info("Unmuted")


class AudioOutService(audio_out_pb2_grpc.AudioOutServicer):
    """gRPC service for audio playback."""

    def __init__(self, sample_rate: int = 22050):
        self.player = AudioPlayer(sample_rate=sample_rate)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Audio-Out service initialized")

    def PlayStream(
        self,
        request_iterator: Iterator[audio_pb2.PCMChunk],
        context: grpc.ServicerContext
    ) -> common_pb2.Empty:
        """
        Streaming RPC: accepts PCM chunks for playback.
        """
        # Start playback if not already running
        if not self.player.is_playing:
            self.player.start()

        try:
            chunk_count = 0
            for pcm_chunk in request_iterator:
                # Parse PCM data
                if pcm_chunk.sample_width == 2:  # 16-bit
                    pcm_data = np.frombuffer(pcm_chunk.data, dtype=np.int16).astype(np.float32) / 32767.0
                elif pcm_chunk.sample_width == 4:  # 32-bit float
                    pcm_data = np.frombuffer(pcm_chunk.data, dtype=np.float32)
                else:
                    self.logger.error(f"Unsupported sample width: {pcm_chunk.sample_width}")
                    continue

                self.logger.debug(
                    f"Received PCM chunk: {len(pcm_data)} samples, "
                    f"{pcm_chunk.sample_rate}Hz, {pcm_chunk.channels}ch"
                )

                # Add to playback queue
                self.player.add_chunk(pcm_data)
                chunk_count += 1

            self.logger.info(f"Playback stream ended: {chunk_count} chunks received")

        except grpc.RpcError as e:
            self.logger.error(f"gRPC error in PlayStream: {e}")
        except Exception as e:
            self.logger.error(f"Error in PlayStream: {e}", exc_info=True)

        return common_pb2.Empty()

    def Control(
        self,
        request: audio_out_pb2.PlaybackControl,
        context: grpc.ServicerContext
    ) -> audio_out_pb2.PlaybackStatus:
        """
        Control playback (play, pause, mute, volume, etc.).
        """
        command = request.command
        self.logger.info(f"Control command: {audio_out_pb2.PlaybackCommand.Name(command)}")

        if command == audio_out_pb2.PLAY:
            self.player.start()
        elif command == audio_out_pb2.PAUSE:
            self.player.stop()
        elif command == audio_out_pb2.RESUME:
            self.player.start()
        elif command == audio_out_pb2.STOP:
            self.player.stop()
        elif command == audio_out_pb2.MUTE:
            self.player.mute()
        elif command == audio_out_pb2.UNMUTE:
            self.player.unmute()

        # Set volume if provided
        if request.volume > 0:
            self.player.set_volume(request.volume)

        # Return current status
        status = self.player.get_status()
        return audio_out_pb2.PlaybackStatus(
            is_playing=status['is_playing'],
            is_muted=status['is_muted'],
            current_volume=status['volume'],
            buffer_size_ms=status['buffer_size_ms']
        )

    def GetStatus(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext
    ) -> audio_out_pb2.PlaybackStatus:
        """
        Get current playback status.
        """
        status = self.player.get_status()
        return audio_out_pb2.PlaybackStatus(
            is_playing=status['is_playing'],
            is_muted=status['is_muted'],
            current_volume=status['volume'],
            buffer_size_ms=status['buffer_size_ms']
        )


def serve(port: int = 50057, sample_rate: int = 22050):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_out_pb2_grpc.add_AudioOutServicer_to_server(
        AudioOutService(sample_rate=sample_rate),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f"Audio-Out service started on port {port}")
    logging.info(f"Sample rate: {sample_rate}Hz")
    logging.info(f"Crossfade: ~20ms, Buffer: ~60ms max")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser(description='Audio-Out Service')
    parser.add_argument('--port', type=int, default=50057, help='Port to listen on')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate (Hz)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s'
    )

    serve(port=args.port, sample_rate=args.sample_rate)


if __name__ == '__main__':
    main()
