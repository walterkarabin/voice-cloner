# Testing Guide

This document provides instructions for testing each component of the character voice pipeline as they are implemented.

## Prerequisites

- Python 3.x with required packages
- Node.js 20+ (for TypeScript services)
- gRPC tools installed
- CUDA 12.1 (for GPU-accelerated services)

## Component Testing Status

### ✅ Completed Components

#### 1. Protobuf Code Generation

**Test the code generation:**
```bash
cd /workspace/libs/proto
./generate.sh
```

**Expected output:**
- Python files in `generated/python/`
- TypeScript files in `generated/typescript/`
- No compilation errors

**Verify imports:**
```bash
python3 -c "import sys; sys.path.insert(0, '/workspace'); from libs.proto.generated.python import embed_pb2, stt_pb2; print('✓ Proto imports successful')"
```

---

#### 2. Audio Utilities Library

**Test audio utilities:**
```bash
cd /workspace
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')
import numpy as np
from libs.audio import AudioResampler, VoiceActivityDetector, AudioFramer

# Test resampler
print("Testing AudioResampler...")
resampler = AudioResampler(input_rate=48000, output_rate=16000)
audio_48k = np.random.randn(48000).astype(np.float32)
audio_16k = resampler.resample(audio_48k)
print(f"  ✓ Resampled {len(audio_48k)} samples @ 48kHz to {len(audio_16k)} samples @ 16kHz")

# Test VAD
print("Testing VoiceActivityDetector...")
vad = VoiceActivityDetector(sample_rate=16000, energy_threshold=0.01)
frame = np.random.randn(480).astype(np.float32) * 0.5
result = vad.process_frame(frame)
print(f"  ✓ VAD result: speech={result['is_speech']}, energy={result['energy']:.4f}")

# Test framer
print("Testing AudioFramer...")
framer = AudioFramer.from_duration(frame_duration_ms=30, sample_rate=16000)
samples = np.random.randn(1000).astype(np.float32)
frames = list(framer.add_samples(samples))
print(f"  ✓ Generated {len(frames)} frames from {len(samples)} samples")

print("\n✓ All audio utilities tests passed!")
EOF
```

---

#### 3. Embed-Loader Service

**Install dependencies:**
```bash
cd /workspace/services/embed-loader
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/embed-loader
python3 server.py --port 50051
```

**Expected output:**
```
INFO - Embed-loader service started on port 50051
INFO - Available characters: ['yoda', 'vader', 'obi-wan', 'leia']
```

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/embed-loader
python3 client.py
```

**Expected output:**
```
✓ yoda: embedding shape (256,)
✓ vader: embedding shape (256,)
✓ obi-wan: embedding shape (256,)
✓ leia: embedding shape (256,)
✗ unknown: Embedding not found for character: unknown
```

**Manual gRPC test:**
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')
from services_loader import EmbedLoaderClient

client = EmbedLoaderClient(host='localhost', port=50051)
embedding = client.get_embedding('yoda')
print(f"✓ Retrieved yoda embedding: shape={embedding.shape}, dtype={embedding.dtype}")
print(f"  First 5 values: {embedding[:5]}")
client.close()
EOF
```

---

#### 4. STT-Whisper Service

**Install dependencies:**
```bash
cd /workspace/services/stt-whisper
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/stt-whisper
python3 server.py --port 50052 --model-size small
```

**Expected output:**

If running in container with firewall (no model downloads available):
```
INFO - Loading Whisper model: small
WARNING - faster-whisper not available, trying openai-whisper
ERROR - Neither faster-whisper nor openai-whisper available. Using mock transcription for testing.
INFO - Whisper STT service started on port 50052
```

If running with internet access and openai-whisper installed:
```
INFO - Loading Whisper model: small
WARNING - faster-whisper not available, trying openai-whisper
INFO - ✓ Loaded openai-whisper model: small
INFO - Whisper STT service started on port 50052
```

Note: The mock mode will return placeholder transcripts for testing the pipeline flow. For real transcription, the Whisper model needs to be downloaded (requires internet access).

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/stt-whisper
python3 client.py
```

**Expected output:**
```
=== Whisper STT Client Test ===

Streaming test audio (5 seconds)...
Note: This is a tone, actual speech will produce real transcripts

[0] PARTIAL: [Mock transcript for 0.50s audio] (conf: 0.85)
[1] PARTIAL: [Mock transcript for 0.50s audio] (conf: 0.85)
...
✓ Received N transcript(s)
```

**Test with real audio file:**
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')
from services_loader import WhisperSTTClient

client = WhisperSTTClient(host='localhost', port=50052)

# Test with audio file (if available)
# transcripts = client.transcribe_file('/path/to/speech.wav')
# Or generate test audio
import numpy as np

def test_audio():
    # Generate 3 seconds of test audio
    for _ in range(6):  # 6 x 0.5s chunks
        yield np.random.randn(8000).astype(np.float32) * 0.1

results = list(client.transcribe_stream(test_audio()))
print(f"✓ Transcription test: received {len(results)} partial transcripts")
client.close()
EOF
```

**Verify latency (if using real Whisper model):**
- Check log timestamps between audio input and transcript output
- Target: < 250ms per partial result
- Final transcript should have `is_final=True`

---

### ⏳ Pending Components

#### 5. Rewriter-LLM Service

**Install dependencies:**
```bash
cd /workspace/services/rewriter-llm
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/rewriter-llm
python3 server.py --port 50053
```

**Expected output:**
```
INFO - Loading LLM model from: /models/llama-3-8b.gguf
WARNING - Failed to load llama-cpp-python. Using mock rewriter for testing.
INFO - LLM Rewriter service started on port 50053
INFO - Available characters: ['yoda', 'vader', 'obi-wan', 'leia']
```

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/rewriter-llm
python3 client.py
```

**Expected output:**
```
=== LLM Rewriter Client Test ===

Testing character rewriting...
Original → Yoda style

Original: Hello, how are you today?
Yoda:     today? you are how Hello,

Original: I am doing great, thank you.
Yoda:     you. thank great, doing am I

Original: The force is strong with you.
Yoda:     you. with strong is force The

✓ Received 3 rewritten fragment(s)
```

**Test all characters:**
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')
from services_loader import LLMRewriterClient

client = LLMRewriterClient(host='localhost', port=50053)

test_text = {
    "text": "The force is strong with you",
    "confidence": 0.95,
    "is_final": True,
    "start_time_ms": 0,
    "end_time_ms": 2000,
    "sequence": 0
}

for character in ["yoda", "vader", "obi-wan", "leia"]:
    print(f"\n{character.upper()}:")
    for fragment in client.rewrite_stream([test_text], character_id=character):
        print(f"  → {fragment['text']}")

client.close()
EOF
```

**Verify latency (with real LLM):**
- Check log timestamps between input and output
- Target: < 40ms per fragment
- Mock mode provides instant response for testing

---

#### 6. Chunker Service
**Status:** Not yet implemented

#### 7. TTS-Streamer Service
**Status:** Not yet implemented

#### 8. Vocoder Service
**Status:** Not yet implemented

#### 9. Audio-Out Service
**Status:** Not yet implemented

---

## Integration Testing

### End-to-End Pipeline Test

**Once all services are implemented:**

**Terminal 1-7: Start all services**
```bash
# Terminal 1
cd /workspace/services/embed-loader && python3 server.py --port 50051

# Terminal 2
cd /workspace/services/stt-whisper && python3 server.py --port 50052

# Terminal 3
cd /workspace/services/rewriter-llm && python3 server.py --port 50053

# Terminal 4
cd /workspace/services/chunker && python3 server.py --port 50054

# Terminal 5
cd /workspace/services/tts-streamer && python3 server.py --port 50055

# Terminal 6
cd /workspace/services/vocoder && python3 server.py --port 50056

# Terminal 7
cd /workspace/services/audio-out && python3 server.py --port 50057
```

**Terminal 8: Run integration test**
```bash
cd /workspace
python3 test_integration.py --character yoda --input test_audio.wav
```

**Expected flow:**
1. Audio chunked and sent to STT-Whisper
2. Partial transcripts sent to LLM Rewriter
3. Character-styled text sent to Chunker
4. Text chunks combined with embeddings for TTS
5. Mel spectrograms sent to Vocoder
6. PCM audio played through Audio-Out

**Target latency:** 300-600ms end-to-end

---

## Docker Compose Testing

**Once Docker images are built:**

```bash
cd /workspace/infra/compose
docker-compose up
```

**Test connectivity:**
```bash
# Check all services are healthy
docker-compose ps

# Test embed-loader through Docker
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')
from services_loader import EmbedLoaderClient
client = EmbedLoaderClient(host='localhost', port=50051)
embedding = client.get_embedding('yoda')
print(f"✓ Docker service test passed: {embedding.shape}")
client.close()
EOF
```

---

## Performance Benchmarks

### Latency Targets

| Component | Target Latency | Notes |
|-----------|---------------|-------|
| Embed-Loader | < 1ms | Preloaded in memory |
| STT-Whisper | 150-250ms | Per partial result |
| LLM Rewriter | 20-40ms | Per fragment |
| Chunker | < 5ms | Text processing |
| TTS-Streamer | 150-300ms | First audio out |
| Vocoder | ~20ms | Per 200ms audio |
| Audio-Out | < 10ms | Playback buffering |
| **Total Pipeline** | **300-600ms** | End-to-end |

### Running Benchmarks

```bash
cd /workspace
python3 benchmark.py --iterations 100 --character yoda
```

---

## Troubleshooting

### Common Issues

**gRPC connection refused:**
- Check service is running: `ps aux | grep python3`
- Verify port is correct
- Check firewall rules

**Import errors:**
- Ensure `/workspace` is in Python path: `sys.path.insert(0, '/workspace')`
- Regenerate protobuf code: `cd libs/proto && ./generate.sh`

**Model loading failures:**
- Check model files exist in expected locations
- Verify CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`

**Audio quality issues:**
- Check sample rate conversions
- Verify crossfading is working in audio-out
- Monitor buffer sizes

---

## Notes

- This document will be updated as each component is completed
- All tests should pass before moving to the next component
- Integration tests require all services to be running
- Performance benchmarks should be run on the target hardware (RTX 3070)
