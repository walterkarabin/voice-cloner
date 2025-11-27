# Embedding Loader Service

Provides preloaded voice embeddings for character voices. Embeddings are loaded at startup for instant retrieval.

## Overview

This service:
- Loads character voice embeddings from disk at startup
- Provides fast embedding retrieval via gRPC
- Supports multiple character voices
- Uses unary RPC for instant response

## Architecture

```
Client Request → GetEmbedding(character_id) → Return embedding bytes
```

## Running the Service

### Start Server

```bash
cd /workspace/services/embed-loader
python server.py --port 50051 --embeddings-dir /data/embeddings
```

### Test with Client

```bash
python client.py
```

## Embedding Format

Embeddings are stored as `.npy` files in the embeddings directory:

```
/data/embeddings/
├── yoda.npy          # 256-dimensional float32 array
├── vader.npy
├── obi-wan.npy
└── leia.npy
```

Each embedding file contains a numpy array:
- **Shape**: `(256,)` or `(512,)` depending on the TTS model
- **Dtype**: `float32`
- **Normalization**: L2-normalized (unit length)

## Creating Embeddings

Voice embeddings are typically extracted using:

1. **OpenVoice**: Extract speaker embedding from reference audio
2. **YourTTS**: Generate speaker embedding from audio sample
3. **Pre-trained models**: Use embeddings from voice conversion models

Example (using OpenVoice):

```python
import numpy as np
from openvoice import se_extractor

# Extract embedding from reference audio
embedding = se_extractor.get_se(
    audio_path="yoda_voice_sample.wav",
    tone_color_converter=converter
)

# Save as .npy file
np.save("/data/embeddings/yoda.npy", embedding)
```

## gRPC API

### GetEmbedding

**Request:**
```protobuf
message EmbeddingRequest {
  string character_id = 1;
  Metadata metadata = 2;
}
```

**Response:**
```protobuf
message EmbeddingResponse {
  string character_id = 1;
  bytes embedding_data = 2;
  int32 embedding_dim = 3;
  Metadata metadata = 4;
}
```

## Testing Mode

If no embeddings directory exists, the service creates sample random embeddings for testing:
- `yoda`, `vader`, `obi-wan`, `leia`
- 256-dimensional float32 arrays
- L2-normalized

## Integration

Used by:
- **TTS-streamer**: Retrieves embeddings for character voice synthesis
- **CLI/Web-UI**: Lists available characters

## Performance

- **Latency**: < 1ms (embeddings are preloaded in memory)
- **Memory**: ~1KB per embedding (256 float32 values)
- **Concurrency**: Thread-safe, handles multiple requests simultaneously
