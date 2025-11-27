# STT-Whisper Service

Streaming speech-to-text service using OpenAI Whisper for real-time transcription.

## Overview

This service:
- Accepts streaming audio chunks via gRPC
- Returns partial transcripts as they become available
- Uses Whisper (small/medium) for high-quality transcription
- Target latency: 150-250ms per partial result
- Supports both CPU and GPU inference

## Architecture

```
Audio Stream (16kHz PCM) → Frame Accumulator (1s chunks) → Whisper Model → Partial Transcripts
```

## Whisper Implementation Options

### 1. faster-whisper (Recommended)

Uses CTranslate2 for optimized inference:

```bash
pip install faster-whisper
```

**Advantages:**
- 4x faster than openai-whisper
- Lower memory usage
- Better for streaming use cases
- GPU support via CUDA

### 2. openai-whisper (Original)

OpenAI's official implementation:

```bash
pip install openai-whisper
```

**Advantages:**
- Original reference implementation
- Well-documented

**Note:** Service will try faster-whisper first, fall back to openai-whisper, then use mock transcription if neither is available.

## Running the Service

### Start Server

```bash
cd /workspace/services/stt-whisper
python3 server.py --port 50052 --model-size small
```

**Model sizes:**
- `tiny`: Fastest, lowest quality (~1GB)
- `base`: Fast, good quality (~1GB)
- `small`: Balanced (~2GB) **← Recommended**
- `medium`: High quality (~5GB)
- `large`: Best quality (~10GB)

### Test with Client

```bash
python3 client.py
```

## gRPC API

### StreamAudio (Bidirectional Streaming)

**Request stream:**
```protobuf
message AudioChunk {
  bytes data = 1;              // Raw PCM audio
  int32 sample_rate = 2;       // e.g., 16000
  int32 channels = 3;          // 1 = mono
  int32 sample_width = 4;      // 2 = 16-bit
  Metadata metadata = 5;
}
```

**Response stream:**
```protobuf
message PartialTranscript {
  string text = 1;             // Transcribed text
  float confidence = 2;        // 0.0 - 1.0
  bool is_final = 3;           // True for last transcript
  int64 start_time_ms = 4;
  int64 end_time_ms = 5;
  Metadata metadata = 6;
}
```

## Audio Requirements

- **Sample rate:** 16000 Hz (will resample if different)
- **Channels:** 1 (mono)
- **Format:** 16-bit PCM (int16)
- **Chunk size:** 20-50ms recommended for low latency

## Processing Flow

1. **Audio Buffering:** Accumulates audio into 1-second chunks
2. **Resampling:** Converts to 16kHz if needed
3. **Transcription:** Runs Whisper inference
4. **Streaming:** Emits partial transcripts immediately
5. **Final Transcript:** Marked with `is_final=true`

## Performance Tuning

### For Lower Latency

```python
# Use smaller model
python3 server.py --model-size tiny

# In code: reduce beam search
segments, info = model.transcribe(
    audio,
    beam_size=1,      # Faster decoding
    best_of=1,
    vad_filter=True   # Skip silence
)
```

### For Higher Quality

```python
# Use larger model
python3 server.py --model-size medium

# In code: increase beam search
segments, info = model.transcribe(
    audio,
    beam_size=5,      # Better quality
    best_of=5
)
```

## Whisper.cpp Integration

For production, consider using whisper.cpp for even better performance:

```bash
# Clone whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp

# Build with CUDA support
make WHISPER_CUDA=1

# Download model
bash ./models/download-ggml-model.sh small

# Run server mode
./server -m models/ggml-small.bin -p 50052
```

Then modify this service to call whisper.cpp via subprocess or socket.

## Example Usage

### Python Client

```python
from services.stt_whisper.client import WhisperSTTClient
import numpy as np

client = WhisperSTTClient(host='localhost', port=50052)

# Generate audio chunks (from microphone, file, etc.)
def audio_stream():
    for chunk in get_microphone_audio():
        yield chunk

# Transcribe
for transcript in client.transcribe_stream(audio_stream()):
    print(f"{transcript['text']} (confidence: {transcript['confidence']:.2f})")

client.close()
```

### From Audio File

```python
client = WhisperSTTClient()
transcripts = client.transcribe_file('speech.wav')

for t in transcripts:
    if t['is_final']:
        print(f"Final: {t['text']}")
```

## Integration with Pipeline

### Connect to Rewriter-LLM

```python
# STT output feeds into LLM rewriter
for transcript in stt_client.transcribe_stream(audio):
    # Send to rewriter
    llm_client.rewrite(transcript['text'])
```

## Monitoring

The service logs:
- Active stream count
- Transcription latency
- Model load time
- Error rates

## Troubleshooting

**Model not loading:**
- Check model path exists
- Verify CUDA is available: `torch.cuda.is_available()`
- Try smaller model size

**High latency:**
- Use smaller model (tiny/base)
- Reduce beam_size
- Enable VAD filter
- Use GPU instead of CPU

**Poor quality:**
- Use larger model
- Check audio quality (16kHz, clear speech)
- Reduce background noise

**Memory issues:**
- Use smaller model
- Reduce max_workers in server
- Use CPU instead of GPU
