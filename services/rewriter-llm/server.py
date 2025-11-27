"""
Rewriter-LLM Service

Transforms text into character-specific speech patterns using LLM.
Accepts partial transcripts and returns character-styled text fragments.
"""

import os
import sys
import grpc
import time
import logging
from concurrent import futures
from typing import Iterator, Optional, Dict

sys.path.insert(0, '/workspace')

from libs.proto.generated.python import rewriter_pb2, rewriter_pb2_grpc, stt_pb2, common_pb2


# Character prompts for different voices
CHARACTER_PROMPTS = {
    "yoda": """You are Yoda from Star Wars. Rewrite the following text in Yoda's speaking style.
Rules:
- Invert subject-verb-object order (e.g., "I am" → "Am I", "You will go" → "Go you will")
- Use wise, philosophical tone
- Keep meaning clear
- Be concise

Text: {text}
Yoda version:""",

    "vader": """You are Darth Vader from Star Wars. Rewrite the following text in Vader's speaking style.
Rules:
- Use commanding, authoritative tone
- Dramatic and ominous
- Short, powerful statements
- Occasionally threatening

Text: {text}
Vader version:""",

    "obi-wan": """You are Obi-Wan Kenobi from Star Wars. Rewrite the following text in Obi-Wan's speaking style.
Rules:
- Wise and diplomatic
- Formal but warm
- Often begins with "Well..." or "Indeed..."
- Philosophical observations

Text: {text}
Obi-Wan version:""",

    "leia": """You are Princess Leia from Star Wars. Rewrite the following text in Leia's speaking style.
Rules:
- Strong and confident
- Direct and assertive
- Leadership qualities
- No-nonsense attitude

Text: {text}
Leia version:""",
}


class LLMRewriterService(rewriter_pb2_grpc.LLMRewriterServicer):
    """
    gRPC service for rewriting text in character voices using LLM.
    """

    def __init__(
        self,
        model_path: str = "/models/llama-3-8b",
        context_size: int = 2048,
        gpu_layers: int = 35
    ):
        """
        Initialize the LLM rewriter service.

        Args:
            model_path: Path to GGUF model file
            context_size: Context window size
            gpu_layers: Number of layers to offload to GPU
        """
        self.model_path = model_path
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.logger = logging.getLogger(__name__)

        # Load LLM model
        self._load_model()

        # Statistics
        self.total_requests = 0
        self.active_streams = 0

    def _load_model(self):
        """Load the LLM model."""
        self.logger.info(f"Loading LLM model from: {self.model_path}")

        try:
            # Try to import llama-cpp-python
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.gpu_layers,
                verbose=False
            )
            self.logger.info(f"✓ Loaded llama-cpp-python model")
            self.use_llama_cpp = True

        except (ImportError, Exception) as e:
            self.logger.warning(
                f"Failed to load llama-cpp-python: {e}. "
                "Using mock rewriter for testing."
            )
            self.model = None
            self.use_llama_cpp = False

    def _rewrite_text(
        self,
        text: str,
        character_id: str,
        max_tokens: int = 100
    ) -> str:
        """
        Rewrite text in character style.

        Args:
            text: Input text to rewrite
            character_id: Character identifier
            max_tokens: Maximum tokens to generate

        Returns:
            Rewritten text
        """
        # Get character prompt
        if character_id not in CHARACTER_PROMPTS:
            self.logger.warning(f"Unknown character: {character_id}, using default")
            character_id = "yoda"

        prompt = CHARACTER_PROMPTS[character_id].format(text=text)

        if self.model is None:
            # Mock rewriter for testing
            return self._mock_rewrite(text, character_id)

        if self.use_llama_cpp:
            # Use llama-cpp-python
            try:
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["\n", "Text:", "version:"],
                    echo=False
                )

                rewritten = response['choices'][0]['text'].strip()
                return rewritten

            except Exception as e:
                self.logger.error(f"LLM generation error: {e}")
                return self._mock_rewrite(text, character_id)

        return text

    def _mock_rewrite(self, text: str, character_id: str) -> str:
        """
        Mock rewriter for testing without real LLM.

        Args:
            text: Input text
            character_id: Character identifier

        Returns:
            Mock rewritten text
        """
        if character_id == "yoda":
            # Simple Yoda-style transformation
            words = text.split()
            if len(words) >= 3:
                # Swap first and last few words
                return f"{words[-1]} {' '.join(words[1:-1])} {words[0]}"
            return text

        elif character_id == "vader":
            return f"{text}... as you wish."

        elif character_id == "obi-wan":
            return f"Well, {text.lower()}"

        elif character_id == "leia":
            return text.upper()

        return text

    def RewriteStream(
        self,
        request_iterator: Iterator[stt_pb2.PartialTranscript],
        context: grpc.ServicerContext
    ) -> Iterator[rewriter_pb2.CharacterFragment]:
        """
        Streaming RPC for text rewriting.

        Args:
            request_iterator: Stream of partial transcripts
            context: gRPC context

        Yields:
            Character-styled text fragments
        """
        self.total_requests += 1
        self.active_streams += 1
        stream_id = self.total_requests

        self.logger.info(f"Stream {stream_id}: Started rewrite stream")

        # Default character (should be set by first request metadata)
        character_id = "yoda"

        try:
            for transcript in request_iterator:
                start_time = time.time()

                # Extract text
                text = transcript.text.strip()

                if not text:
                    continue

                # Rewrite in character style
                rewritten = self._rewrite_text(text, character_id)

                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)

                # Create response
                fragment = rewriter_pb2.CharacterFragment(
                    text=rewritten,
                    character_id=character_id,
                    is_provisional=not transcript.is_final,
                    original_start_ms=transcript.start_time_ms,
                    original_end_ms=transcript.end_time_ms,
                    metadata=common_pb2.Metadata(
                        sequence_number=transcript.metadata.sequence_number
                    )
                )

                yield fragment

                self.logger.debug(
                    f"Stream {stream_id}: [{character_id}] "
                    f"'{text}' → '{rewritten}' ({latency_ms}ms)"
                )

            self.logger.info(f"Stream {stream_id}: Completed")

        except Exception as e:
            self.logger.error(f"Stream {stream_id}: Error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Rewriting error: {str(e)}")

        finally:
            self.active_streams -= 1


def serve(
    port: int = 50053,
    model_path: str = "/models/llama-3-8b.gguf"
):
    """
    Start the LLM rewriter gRPC server.

    Args:
        port: Port to listen on
        model_path: Path to GGUF model file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    # Add service
    rewriter_service = LLMRewriterService(model_path=model_path)
    rewriter_pb2_grpc.add_LLMRewriterServicer_to_server(rewriter_service, server)

    # Start server
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f"LLM Rewriter service started on port {port}")
    logger.info(f"Available characters: {list(CHARACTER_PROMPTS.keys())}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down LLM Rewriter service")
        server.stop(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='LLM Character Voice Rewriter Service'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=50053,
        help='Port to listen on (default: 50053)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/models/llama-3-8b.gguf',
        help='Path to GGUF model file'
    )

    args = parser.parse_args()
    serve(port=args.port, model_path=args.model_path)
