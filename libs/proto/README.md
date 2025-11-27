# Protocol Buffer Schemas

This directory contains all protobuf definitions for inter-service communication in the character voice pipeline.

## Schema Files

### Core Types
- **common.proto** - Shared types (Timestamp, Metadata, Empty)
- **audio.proto** - Audio data types (AudioChunk, PCMChunk, MelChunk)

### Service Definitions

1. **stt.proto** - Whisper STT Service
   - `StreamAudio`: AudioChunk → PartialTranscript (bidirectional streaming)

2. **rewriter.proto** - LLM Rewriter Service
   - `RewriteStream`: PartialTranscript → CharacterFragment (streaming)

3. **chunker.proto** - Text Chunker Service
   - `ChunkTextStream`: CharacterFragment → TextChunk (streaming)

4. **embed.proto** - Embedding Loader Service
   - `GetEmbedding`: EmbeddingRequest → EmbeddingResponse (unary)

5. **tts.proto** - TTS Streamer Service (OpenVoice)
   - `GenerateMel`: TTSRequest → MelChunk (streaming)

6. **vocoder.proto** - Vocoder Service (HiFi-GAN)
   - `GeneratePCM`: MelChunk → PCMChunk (streaming)

7. **audio_out.proto** - Audio Output Service
   - `PlayStream`: PCMChunk → Empty (client streaming)
   - `Control`: PlaybackControl → PlaybackStatus (unary)
   - `GetStatus`: Empty → PlaybackStatus (unary)

## Data Flow

```
Microphone
    ↓ AudioChunk
WhisperSTT
    ↓ PartialTranscript
LLMRewriter
    ↓ CharacterFragment
Chunker
    ↓ TextChunk → (combine with embedding)
    ↓ TTSRequest
TTSStreamer
    ↓ MelChunk
Vocoder
    ↓ PCMChunk
AudioOut
    ↓
Speaker
```

## Code Generation

### Python
```bash
python -m grpc_tools.protoc \
  -I/workspace/libs/proto \
  --python_out=/workspace/libs/proto/generated/python \
  --grpc_python_out=/workspace/libs/proto/generated/python \
  /workspace/libs/proto/*.proto
```

### Node.js/TypeScript
```bash
protoc \
  -I/workspace/libs/proto \
  --plugin=protoc-gen-ts_proto=./node_modules/.bin/protoc-gen-ts_proto \
  --ts_proto_out=/workspace/libs/proto/generated/typescript \
  /workspace/libs/proto/*.proto
```

## Message Design Principles

1. **Streaming-first**: All services support streaming to minimize latency
2. **Metadata tracking**: Each message includes metadata for request tracking
3. **Flexibility**: Messages include both required data and optional fields for future extension
4. **Type safety**: Strong typing for sample rates, dimensions, and formats
