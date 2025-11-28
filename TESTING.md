# Testing Guide

This document provides instructions for testing each component of the character voice pipeline as they are implemented.

## Prerequisites

- Python 3.x with required packages
- Node.js 20+ (for TypeScript services)
- gRPC tools installed
- CUDA 12.1 (for GPU-accelerated services)

## Using Real Models vs Mock Implementations

All services support both **mock mode** (for testing without models) and **real model mode** (for production use):

### Mock Mode (Default)
- No model downloads required
- No GPU needed
- Fast startup
- Useful for testing pipeline flow
- Use `--use-mock` flag (default for most services)

### Real Model Mode
- Requires model downloads (internet access needed once)
- GPU recommended for best performance
- Slower startup (model loading)
- Production-quality output
- Use `--use-real` or specific model flags

### Service-Specific Real Model Usage

**STT-Whisper**: Uses faster-whisper or openai-whisper automatically
```bash
python3 server.py --port 50052 --model-size small
# Models auto-download from openaipublic.azureedge.net
```

**Rewriter-LLM**: Requires GGUF model file
```bash
python3 server.py --port 50053 --model-path /path/to/llama-model.gguf
# Download models from huggingface.co
```

**TTS-Streamer**: Uses Coqui TTS models
```bash
python3 server.py --port 50055 --use-real --model-name "tts_models/en/vctk/vits"
# Models auto-download from Coqui TTS
```

**Vocoder**: Uses Griffin-Lim (no model) or HiFi-GAN
```bash
python3 server.py --port 50056 --use-griffin-lim  # No model needed
# OR with HiFi-GAN model:
python3 server.py --port 50056 --use-real --model-path /path/to/hifigan
```

**Note**: Ensure firewall allows access to model download domains (huggingface.co, openaipublic.azureedge.net)

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

**Install dependencies:**
```bash
cd /workspace/services/chunker
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/chunker
python3 server.py --port 50054
```

**Expected output:**
```
INFO - Chunker service initialized
INFO - Chunker service started on port 50054
INFO - Chunk size range: 15-40 characters
INFO - Predicts clause boundaries for natural phrasing
```

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/chunker
python3 client.py
```

**Expected output:**
```
=== Chunker Client Test ===

Sending character fragments for chunking...

Received N chunks:

[ 0] │ yoda       │ The force is strong with you,
[ 1] │ yoda       │ young Skywalker,
[ 2] │ yoda       │ and you must learn to control it.
...

✓ Received N text chunk(s)
```

**Test with inline Python:**
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')
from services_loader import ChunkerClient

client = ChunkerClient(host='localhost', port=50054)

fragments = [
    {
        'text': "The force is strong with you, young Skywalker, and you must learn to control it.",
        'character_id': 'yoda',
        'is_provisional': False
    }
]

chunks = list(client.chunk_stream(fragments))
print(f"✓ Chunking test: received {len(chunks)} chunks")
for chunk in chunks[:3]:  # Show first 3
    print(f"  [{chunk['chunk_index']}] {chunk['text']} (boundary={chunk['is_clause_boundary']})")
client.close()
EOF
```

**Verify chunking quality:**
- Chunks should be 15-40 characters
- Clause boundaries should be detected (commas, periods)
- Long text should be split at word boundaries
- Target latency: < 5ms per fragment

---

#### 7. TTS-Streamer Service

**Install dependencies:**
```bash
cd /workspace/services/tts-streamer
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/tts-streamer
python3 server.py --port 50055 --use-mock
```

**Expected output:**
```
WARNING - Using mock TTS engine for testing
INFO - TTS-Streamer service initialized
INFO - TTS-Streamer service started on port 50055
INFO - Using mock TTS engine - generates synthetic mel spectrograms
```

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/tts-streamer
python3 client.py
```

**Expected output:**
```
=== TTS-Streamer Client Test ===

Sending TTS requests...

Received N mel chunks:

[ 0] Mel chunk: 80x50 (min=-X.XX, max=X.XX)
[ 1] Mel chunk: 80x50 (min=-X.XX, max=X.XX)
...

✓ Received N mel chunk(s)
  Total frames: XXX
  Estimated duration: X.XXs
```

**Test with inline Python:**
```bash
python3 << 'EOF'
import sys
import numpy as np
sys.path.insert(0, '/workspace')
from services_loader import TTSStreamerClient

client = TTSStreamerClient(host='localhost', port=50055)

# Generate test embedding
embedding = np.random.randn(256).astype(np.float32)

requests = [
    {
        'text': "The force is strong with you.",
        'embedding': embedding,
        'character_id': 'yoda',
        'chunk_index': 0
    }
]

mel_chunks = list(client.generate_mel_stream(requests))
print(f"✓ TTS test: received {len(mel_chunks)} mel chunks")
print(f"  First chunk shape: {mel_chunks[0]['n_mels']}x{mel_chunks[0]['n_frames']}")
client.close()
EOF
```

**Verify TTS output:**
- Mel spectrograms should be 80 x N frames
- Streaming should start within ~50ms (mock mode)
- With real OpenVoice: target 150-300ms for first audio

**Note:** Mock mode generates synthetic mel spectrograms. For real TTS, install OpenVoice model and use `--model-path` flag.

---

#### 8. Vocoder Service

**Install dependencies:**
```bash
cd /workspace/services/vocoder
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/vocoder
python3 server.py --port 50056 --use-mock
```

**Expected output:**
```
WARNING - Using mock vocoder for testing
INFO - Vocoder service initialized
INFO - Vocoder service started on port 50056
INFO - Using mock vocoder - generates synthetic PCM audio
```

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/vocoder
python3 client.py
```

**Expected output:**
```
=== Vocoder Client Test ===

Sending 3 mel chunks...

Received N PCM chunks:

[ 0] PCM chunk: XXXX samples, 22050Hz, 1ch, 16-bit
[ 1] PCM chunk: XXXX samples, 22050Hz, 1ch, 16-bit
...

✓ Received N PCM chunk(s)
  Total samples: XXXXX
  Duration: X.XXs
```

**Test with inline Python:**
```bash
python3 << 'EOF'
import sys
import numpy as np
sys.path.insert(0, '/workspace')
from services_loader import VocoderClient

client = VocoderClient(host='localhost', port=50056)

# Generate test mel spectrogram
mel_chunks = [
    np.random.randn(80, 50).astype(np.float32)
]

pcm_chunks = list(client.generate_pcm_stream(mel_chunks))
print(f"✓ Vocoder test: received {len(pcm_chunks)} PCM chunks")
if pcm_chunks:
    total_samples = sum(len(c['data']) for c in pcm_chunks)
    duration = total_samples / pcm_chunks[0]['sample_rate']
    print(f"  Total duration: {duration:.2f}s")
client.close()
EOF
```

**Verify vocoder output:**
- PCM audio should be 22050Hz, mono, 16-bit
- Processing should be ~10% of real-time (mock mode)
- With real HiFi-GAN: target ~20ms per 200ms of audio

**Note:** Mock mode generates synthetic PCM. For real vocoding, install HiFi-GAN model and use `--model-path` flag.

---

#### 9. Audio-Out Service

**Install dependencies:**
```bash
cd /workspace/services/audio-out
pip3 install -r requirements.txt
```

**Terminal 1 - Start the server:**
```bash
cd /workspace/services/audio-out
python3 server.py --port 50057
```

**Expected output:**
```
INFO - Audio-Out service initialized
INFO - Audio-Out service started on port 50057
INFO - Sample rate: 22050Hz
INFO - Crossfade: ~20ms, Buffer: ~60ms max
```

**Terminal 2 - Test with client:**
```bash
cd /workspace/services/audio-out
python3 client.py
```

**Expected output:**
```
=== Audio-Out Client Test ===

1. Getting initial status...
   Status: playing=False, muted=False, volume=1.00

2. Streaming audio chunks...
   ✓ Sent 5 chunks

   Waiting X.Xs for playback to complete...

3. Testing control commands...
   Set volume to 0.8: volume=0.80
   Muted: is_muted=True
   Unmuted: is_muted=False

   ✓ Control commands successful

4. Final status...
   Status: playing=True, muted=False, volume=0.80, buffer=XXms

✓ Audio-Out client test completed
```

**Test with inline Python:**
```bash
python3 << 'EOF'
import sys
import numpy as np
sys.path.insert(0, '/workspace')
from services_loader import AudioOutClient
from libs.proto.generated.python import audio_out_pb2

client = AudioOutClient(host='localhost', port=50057)

# Generate test audio
sample_rate = 22050
t = np.arange(8820) / sample_rate  # 0.4s
tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Play audio
client.play_stream([tone], sample_rate=sample_rate)
print("✓ Audio playback started")

# Get status
status = client.get_status()
print(f"  Status: playing={status['is_playing']}, buffer={status['buffer_size_ms']}ms")

client.close()
EOF
```

**Verify audio-out:**
- Playback should start immediately
- Crossfading prevents clicks/pops between chunks
- Buffer should stay under 60ms
- Control commands (mute, volume) work correctly

**Note:** Current implementation simulates playback timing. For real audio output, install PyAudio or sounddevice and enable in server.py.

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

### Building and Starting Services

**Build all service images:**
```bash
cd /workspace/infra/compose
docker-compose build
```

**Start all services (mock mode):**
```bash
docker-compose up -d
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f stt-whisper
```

**Check service health:**
```bash
docker-compose ps
```

All services should show status as "Up" with healthy status.

### Test Individual Services

**Test embed-loader through Docker:**
```bash
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

### Using Real Models with Docker

Edit `/workspace/infra/compose/.env` (copy from `.env.example`):

```bash
# Enable real models
STT_MODEL_SIZE=small
TTS_USE_MOCK=false
VOCODER_USE_MOCK=false
VOCODER_USE_GRIFFIN_LIM=true
```

Then rebuild and restart:
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

### Stopping Services

```bash
# Stop but keep containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove everything including volumes
docker-compose down -v
```

---

## Live Audio Pipeline (CLI)

### Prerequisites

```bash
cd /workspace/apps/cli
pip3 install -r requirements.txt
```

### List Audio Devices

```bash
python3 voice_pipeline.py --list-devices
```

This shows all available microphones and speakers with their device indices.

### Run Full Pipeline

**With Docker services:**
```bash
# 1. Start services with Docker
cd /workspace/infra/compose
docker-compose up -d

# 2. Run CLI orchestrator
cd /workspace/apps/cli
python3 voice_pipeline.py --character yoda --host localhost
```

**With local services:**
```bash
# Start each service in separate terminals (see individual service tests above)
# Then run CLI:
cd /workspace/apps/cli
python3 voice_pipeline.py --character yoda
```

**With specific audio devices:**
```bash
# List devices first to get indices
python3 voice_pipeline.py --list-devices

# Use specific devices
python3 voice_pipeline.py --character vader --input-device 2 --output-device 3
```

### Available Characters

- `yoda` - Yoda's inverted word order speech
- `vader` - Darth Vader's commanding tone
- `obi-wan` - Obi-Wan's wise, diplomatic style
- `leia` - Princess Leia's assertive voice

### Pipeline Flow

```
Microphone Input
    ↓
STT-Whisper (Speech → Text)
    ↓
Rewriter-LLM (Text → Character Style)
    ↓
Chunker (Split into chunks)
    ↓
Embed-Loader (Get voice embedding)
    ↓
TTS-Streamer (Text + Embedding → Mel)
    ↓
Vocoder (Mel → PCM Audio)
    ↓
Audio-Out (Play through speakers)
    ↓
Speaker Output
```

### Quick Start Script

Use the convenience script:

```bash
# Start with Docker (mock mode)
/workspace/start-pipeline.sh docker yoda

# Show local commands
/workspace/start-pipeline.sh local vader

# Start mock mode
/workspace/start-pipeline.sh mock
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

---

## Docker Infrastructure

### Building Images

**Build all service images:**
```bash
cd /workspace/infra/compose
docker-compose build
```

**Build specific service:**
```bash
docker-compose build stt-whisper
```

### Running Services

**Start all services (detached):**
```bash
docker-compose up -d
```

**Start with logs visible:**
```bash
docker-compose up
```

**View service logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f stt-whisper
```

**Check service health:**
```bash
docker-compose ps
```

### Configuration

Copy `.env.example` to `.env` and customize:

```bash
cd /workspace/infra/compose
cp .env.example .env
nano .env  # Edit configuration
```

**Example configurations:**

Mock mode (fast testing):
```env
STT_MODEL_SIZE=small
TTS_USE_MOCK=true
VOCODER_USE_MOCK=true
```

Real models:
```env
STT_MODEL_SIZE=small
TTS_USE_MOCK=false
TTS_MODEL_NAME=tts_models/en/vctk/vits
VOCODER_USE_GRIFFIN_LIM=true
```

### Stopping Services

```bash
# Stop (keep containers)
docker-compose stop

# Stop and remove
docker-compose down

# Remove everything
docker-compose down -v
```

---

## Live Audio Pipeline

### Setup

**Install CLI dependencies:**
```bash
cd /workspace/apps/cli
pip3 install -r requirements.txt
```

### List Audio Devices

```bash
python3 voice_pipeline.py --list-devices
```

Output shows input/output devices:
```
=== Audio Input Devices ===
[0] Built-in Microphone (DEFAULT)
    Channels: 1, Sample Rate: 48000.0 Hz
[2] USB Microphone
    Channels: 2, Sample Rate: 44100.0 Hz

=== Audio Output Devices ===
[1] Built-in Speaker (DEFAULT)
    Channels: 2, Sample Rate: 48000.0 Hz
```

### Running the Pipeline

**Basic usage:**
```bash
# Start Docker services first
cd /workspace/infra/compose && docker-compose up -d

# Run CLI orchestrator
cd /workspace/apps/cli
python3 voice_pipeline.py --character yoda
```

**With specific devices:**
```bash
python3 voice_pipeline.py --character vader --input-device 2 --output-device 1
```

**With debug logging:**
```bash
python3 voice_pipeline.py --character obi-wan --debug
```

**Connect to remote services:**
```bash
python3 voice_pipeline.py --character leia --host 192.168.1.100
```

### Character Options

- `yoda` - Inverted word order ("Strong with you, the Force is")
- `vader` - Commanding, ominous tone
- `obi-wan` - Wise, diplomatic speech
- `leia` - Assertive, leadership voice

### How It Works

The CLI orchestrator:
1. Captures audio from your microphone (streaming)
2. Sends to STT service for transcription
3. Rewrites text in character's voice (LLM)
4. Chunks text for TTS
5. Generates speech with character voice
6. Plays through speakers in real-time

Target end-to-end latency: 300-600ms

### Quick Start

Use the startup script:

```bash
# Start Docker services and show CLI command
./start-pipeline.sh docker yoda

# Show commands for local development
./start-pipeline.sh local vader

# Start mock mode for testing
./start-pipeline.sh mock
```

### Troubleshooting Audio

**No audio devices:**
```bash
# Install PortAudio
sudo apt-get install portaudio19-dev

# Test with sounddevice
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Microphone not working:**
- Check system audio settings
- Test with `arecord -l` (Linux) or system preferences
- Verify microphone permissions
- Try different `--input-device` index

**No audio output:**
- Check speaker volume
- Verify services are using real models (not mock)
- Check `docker-compose logs audio-out`
- Test with different `--output-device`

**High latency:**
- Use mock mode to test pipeline flow
- Enable GPU for STT, LLM, TTS services
- Reduce audio chunk size
- Use faster-whisper instead of openai-whisper

**Connection refused:**
- Ensure services are running: `docker-compose ps`
- Check service logs: `docker-compose logs`
- Verify ports 50051-50057 are accessible
- Use `--host localhost` for Docker services

---

## Complete End-to-End Example

### 1. Start Services

```bash
cd /workspace/infra/compose
docker-compose up -d
```

### 2. Verify All Services

```bash
docker-compose ps
# All should show "Up" and "healthy"
```

### 3. List Audio Devices

```bash
cd /workspace/apps/cli
python3 voice_pipeline.py --list-devices
```

### 4. Run Pipeline

```bash
python3 voice_pipeline.py --character yoda --debug
```

### 5. Speak and Listen

Speak into your microphone and hear your voice transformed into Yoda's speech pattern in real-time!

### 6. Stop When Done

```bash
# Stop CLI with Ctrl+C

# Stop services
cd /workspace/infra/compose
docker-compose down
```

---
