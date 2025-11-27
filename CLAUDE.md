# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **real-time streaming character voice pipeline** that converts live audio input into character-voiced output with minimal latency (300-600ms target). The pipeline transforms: **Audio In → Whisper STT → LLM Rewriter → Chunker → OpenVoice TTS → Vocoder → Audio Out**.

All stages stream data incrementally—no buffering of complete sentences. This enables near-real-time character voice transformation suitable for live conversation or gaming applications.

## Architecture

### Monorepo Structure

The project follows a services-based monorepo architecture:

```
/character-voice-pipeline/
├─ apps/
│  ├─ cli/              # CLI application for testing pipeline
│  └─ web-ui/           # Web interface for configuration/monitoring
├─ services/
│  ├─ stt-whisper/      # Streaming Whisper STT (whisper.cpp)
│  ├─ rewriter-llm/     # LLM text-to-character rewriter (Llama 3)
│  ├─ chunker/          # Prosodic text chunking for streaming TTS
│  ├─ embed-loader/     # Preloaded voice embeddings per character
│  ├─ tts-streamer/     # OpenVoice acoustic model (mel generation)
│  ├─ vocoder/          # HiFi-GAN mel-to-PCM conversion
│  └─ audio-out/        # PCM playback with crossfading
├─ libs/
│  ├─ proto/            # Shared protobuf schemas for all IPC
│  ├─ audio/            # Audio utilities (resample, VAD, framing)
│  └─ models/           # Model wrappers (Whisper, LLM, TTS)
├─ infra/
│  ├─ docker/           # Docker images per service
│  └─ compose/          # Docker Compose orchestration
└─ experiments/         # Experimental features and benchmarks
```

### Service Communication

Services communicate via **gRPC bidirectional streaming**:
- `stt-whisper`: `StreamAudio(AudioChunk) → StreamText(PartialTranscript)`
- `rewriter-llm`: `RewriteStream(PartialTranscript) → StreamCorrectedText(YodaFragments)`
- `chunker`: `ChunkTextStream(YodaFragments) → StreamSegments(TextChunk)`
- `embed-loader`: `GetEmbedding(character_id) → EmbeddingBytes` (unary)
- `tts-streamer`: `GenerateMel(Chunk{text, embedding}) → StreamMel(MelChunk)`
- `vocoder`: `GeneratePCM(MelChunk) → StreamPCM(PCMChunk)`
- `audio-out`: `PlayStream(PCMChunk)` (unary streaming)

### Data Flow

1. **Microphone** → 20-40ms audio frames → `stt-whisper`
2. **Whisper** → Partial transcripts every 150-250ms → `rewriter-llm`
3. **LLM** → Character-rewritten text fragments → `chunker`
4. **Chunker** → Prosodic chunks (0.5-1.2s) → `tts-streamer`
5. **TTS** → Mel spectrogram frames → `vocoder`
6. **Vocoder** → PCM audio frames → `audio-out`
7. **Audio-out** → Playback with 10-20ms crossfade

### Key Design Constraints

- **Target latency**: 300-600ms end-to-end
- **GPU**: RTX 3070 for TTS/Vocoder inference
- **Streaming**: All stages must stream incrementally (no sentence buffering)
- **Chunk size**: Text chunks 20-40 chars, audio chunks 20-50ms
- **Models**:
  - Whisper: small/medium via whisper.cpp
  - LLM: Llama 3 8B/3B quantized GGUF via llama.cpp
  - TTS: OpenVoice acoustic model
  - Vocoder: HiFi-GAN (TorchScript/ONNX)

## Development Environment

### Prerequisites

The devcontainer is configured with:
- **CUDA 12.1** (cudnn8-runtime)
- **Node.js 20** (for JavaScript/TypeScript services)
- **Python 3** with PyTorch (CUDA 12.1), Transformers, TTS
- **Audio libraries**: libsndfile1
- **Tools**: git, gh, vim, nano, jq

### Container Setup

The environment runs as user `node` with sudo access. The devcontainer includes:
- GPU passthrough for CUDA workloads
- Network capabilities (NET_ADMIN, NET_RAW) for firewall control
- Persistent volumes for bash history and Claude config

### Python Environment

Python packages are installed globally:
```bash
# Already installed in container:
# - torch, torchvision, torchaudio (CUDA 12.1)
# - transformers, TTS, numpy, scipy, soundfile

# For service-specific dependencies:
cd services/<service-name>
pip install -r requirements.txt  # if exists
```

### Node.js Environment

Each Node.js service/app should have its own package.json:
```bash
cd apps/cli  # or services/<service-name>
npm install
npm run dev
```

## Common Development Workflows

### Creating a New Service

When implementing services from PROJECT_FLOW.md:

1. Create service directory under `services/`
2. Choose language (Python for ML services, Node.js for coordination)
3. Define protobuf schema in `libs/proto/` if new messages needed
4. Implement gRPC streaming server
5. Add Dockerfile in `infra/docker/<service-name>/`
6. Update docker-compose.yaml in `infra/compose/`

### Testing Services

Each service should be testable independently:
- Unit tests for core logic
- Integration tests with mock gRPC clients/servers
- Latency benchmarks to verify streaming performance

### Running the Pipeline

(To be implemented - this will likely involve docker-compose):
```bash
# Start all services
cd infra/compose
docker-compose up

# Or run individual services for development
cd services/stt-whisper
python server.py  # or npm start
```

### Protobuf Schema Changes

1. Edit `.proto` files in `libs/proto/`
2. Regenerate code for all languages used:
   ```bash
   # Python
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. libs/proto/*.proto

   # Node.js/TypeScript
   protoc -I. --plugin=protoc-gen-ts_proto=./node_modules/.bin/protoc-gen-ts_proto \
     --ts_proto_out=. libs/proto/*.proto
   ```
3. Update all affected services

## Model Considerations

### Whisper STT
- Use whisper.cpp for efficient CPU/GPU inference
- Sliding window streaming mode for partial transcripts
- Expected latency: 150-250ms per partial result

### LLM Rewriter
- Llama 3 8B or 3B quantized GGUF
- Run via llama.cpp for fast inference (~20-40ms/fragment)
- Few-shot prompt defines character voice (e.g., Yoda syntax)
- Must handle partial/incomplete input gracefully

### OpenVoice TTS
- Generates mel spectrograms in streaming fashion
- Requires preloaded voice embeddings per character
- Target: first audio in 150-300ms
- Supports overlapping generation to mask chunk boundaries

### HiFi-GAN Vocoder
- Convert mel frames to PCM
- TorchScript or ONNX for low latency
- Target: ~20ms per 200ms of audio on RTX 3070

## Important Notes

- **No sentence buffering**: All components must process and forward data as soon as possible
- **Chunking strategy**: The chunker predicts clause boundaries even without punctuation
- **Audio crossfading**: Audio-out applies 10-20ms crossfade to prevent clicks/pops
- **Jitter buffer**: Keep audio-out buffer minimal (40-60ms max)
- **Character embeddings**: Preload on startup to avoid lookup latency
- **Error handling**: Services must handle partial/updated text gracefully (LLM may revise earlier outputs)

## Project Status

**Current State**: Repository initialized with architecture documentation. Core services not yet implemented.

**Next Steps** (as per PROJECT_FLOW.md):
1. Set up monorepo structure (apps/, services/, libs/, infra/)
2. Define protobuf schemas in libs/proto/
3. Implement services in priority order:
   - embed-loader (no dependencies)
   - audio utilities (libs/audio/)
   - stt-whisper
   - rewriter-llm
   - chunker
   - tts-streamer
   - vocoder
   - audio-out
4. Create integration tests
5. Build CLI and web-ui for testing
