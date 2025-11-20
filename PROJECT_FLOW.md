## High-Level Flow (Streaming Path)

**Audio In → Whisper (STT) → LLM Rewriter → Chunker → OpenVoice TTS → Vocoder → Audio Out**

All stages stream. No buffering whole sentences.

## Updated Monorepo Structure

```
/character-voice-pipeline/
├─ apps/
│  ├─ cli/
│  └─ web-ui/
├─ services/
│  ├─ stt-whisper/           # Streaming Whisper STT
│  ├─ rewriter-llm/          # Converts text → character phrasing
│  ├─ chunker/               # Splits rewritten text into prosodic chunks
│  ├─ embed-loader/          # Preloaded voice embeddings for characters
│  ├─ tts-streamer/          # OpenVoice acoustic model generating mel chunks
│  ├─ vocoder/               # Converts mel chunks → PCM
│  └─ audio-out/             # Plays PCM chunks
├─ libs/
│  ├─ proto/                 # Shared protobuf for all IPC
│  ├─ audio/                 # Resample, VAD, framing utils
│  └─ models/                # Whisper wrapper, LLM wrapper, TTS utilities
├─ infra/
│  ├─ docker/
│  └─ compose/
└─ experiments/
```

## Detailed Component Design

### 1. Whisper STT (streaming)

**Goal:** Accept live microphone chunks → output partial transcripts every 100–200ms.

**Recommended model:**

- Whisper-small or Whisper-medium using whisper.cpp for fast CPU/GPU decoding.

- Streaming mode: uses sliding window inference.

**API:**

- gRPC bidirectional streaming `StreamAudio(in AudioChunk) → StreamText(out PartialTranscript)`

**Output:**

- Emits partial text fragments as soon as confidence is high enough.

- Can optionally include timestamps for chunker sync.

### 2. LLM Rewriter (character cadence converter)

**Goal:** transform partial utterances into Yoda-style (or any character) phrasing, fast enough to keep pace with speaking.

**Model:**

- Llama 3 8B or 3B, quantized GGUF, running via llama.cpp.

- Few-shot prompt pinned at startup to enforce character voice.

- Executes in ~20–40ms per short fragment on CPU/GPU.

**API:**

`RewriteStream(in PartialTranscript) → StreamCorrectedText(out YodaFragments)`

**Behavior:**

- Accepts partial text from Whisper.

- Rewrites incrementally.

- If text is incomplete, produces provisional rewrites (updated in next revision).

### 3. Chunker

**Goal:** Break rewritten fragments into short prosodic units (0.5–1.2s) so TTS can stream audio.

**API:**

`ChunkTextStream(in YodaFragments) → StreamSegments(out TextChunk)`

**Tasks:**

- Clause splitting (commas, pauses).

- Enforce max characters per chunk (20–40).

- Don’t wait for punctuation—predict clause boundaries.

### 4. Embed-loader

**Goal:** Provide preloaded voice embedding per character.

**Behavior:**

- On startup, loads embeddings from disk for all characters.

- When receiving a chunk: returns the correct embedding instantly.

**API:**

`GetEmbedding(character_id) → EmbeddingBytes`

### 5. TTS-streamer (OpenVoice acoustic model)

**Goal:** Convert one TextChunk + Embedding into Mel spectrogram frames in real time, streaming.

**Behavior:**

- Runs on RTX 3070.

- Generates mel windows as they are computed (20–50ms chunks).

- Supports overlapping generation to mask boundaries.

**API:**

`GenerateMel(in Chunk{ text, embedding }) → StreamMel(out MelChunk)`

Latency: ~150–300ms for first audio.

### 6. Vocoder

**Goal:** Convert MelChunk → PCM quickly.

**Model:**

- HiFi-GAN, TorchScript or ONNX for low latency.

**API:**

`GeneratePCM(in MelChunk) → StreamPCM(out PCMChunk)`

Latency: ≈ 20ms / 200ms audio on a 3070.

### 7. Audio-out

**Goal:** Play PCMChunk instantly with minimal pop/click.

**Features:**

- 10–20ms crossfade between chunks

- Jitter buffer (~40–60ms max)

- Volume control, mute, pause

**API:**

`PlayStream(PCMChunk)`

---

## Full Streaming Data Flow (Step-By-Step)

### A. Input

- Mic captures 20–40ms audio frames.

- Send frames to stt-whisper.

### B. Whisper → partial text

- Every ~150–250ms, Whisper emits updated partial transcripts.

### C. LLM rewriter

- Receives partial text.

- Rewrites it to character cadence.

- Streams out updated rewritten fragments.

### D. Chunker

- Receives rewritten fragments.

- Splits into TTS-friendly micro-chunks.

- Streams them out ASAP.

### E. TTS-streamer

- Takes each chunk + embedding.

- Starts generating mel frames.

- Streams mels as they’re computed.

### F. Vocoder

- Converts mel frames to PCM audio frames.

- Streams PCM frames.

### G. Audio-out

- Plays PCM with minimal buffering.

**Total end-to-end latency:**
~300–600ms, depending on aggressiveness.
