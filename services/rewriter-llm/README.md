# Rewriter-LLM Service

Transforms text into character-specific speech patterns using LLM (Llama 3).

## Overview

This service:
- Accepts streaming partial transcripts from STT
- Rewrites text in character voice (Yoda, Vader, Obi-Wan, Leia)
- Returns character-styled fragments in real-time
- Target latency: 20-40ms per fragment
- Uses few-shot prompting for character consistency

## Architecture

```
Partial Transcripts → Character Prompt + LLM → Character-styled Fragments
```

## Supported Characters

### Yoda
- Inverts subject-verb-object order
- Wise, philosophical tone
- Example: "You will learn" → "Learn you will"

### Darth Vader
- Commanding, authoritative
- Dramatic and ominous
- Example: "Come here" → "Come here... as you wish"

### Obi-Wan Kenobi
- Wise and diplomatic
- Formal but warm
- Example: "I understand" → "Well, i understand"

### Princess Leia
- Strong and confident
- Direct and assertive
- Example: "listen to me" → "LISTEN TO ME"

## LLM Implementation Options

### 1. llama-cpp-python (Recommended)

Uses GGUF quantized models for fast CPU/GPU inference:

```bash
pip install llama-cpp-python
```

**Download Llama 3 GGUF:**
```bash
# From HuggingFace
wget https://huggingface.co/...llama-3-8b-instruct-q4_k_m.gguf
```

**Advantages:**
- Fast inference (20-40ms per fragment)
- Low memory usage with quantization
- Runs on CPU or GPU
- No Python/Torch dependencies

### 2. Transformers + PyTorch

Uses HuggingFace models:

```bash
pip install torch transformers
```

**Note:** Service will use mock rewriter if no LLM backend is available.

## Running the Service

### Start Server

```bash
cd /workspace/services/rewriter-llm
python3 server.py --port 50053 --model-path /models/llama-3-8b.gguf
```

**With llama-cpp-python:**
```bash
# CPU inference
python3 server.py

# GPU inference (faster)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
python3 server.py
```

### Test with Client

```bash
python3 client.py
```

## gRPC API

### RewriteStream (Streaming)

**Request stream:**
```protobuf
message PartialTranscript {
  string text = 1;
  float confidence = 2;
  bool is_final = 3;
  int64 start_time_ms = 4;
  int64 end_time_ms = 5;
  Metadata metadata = 6;
}
```

**Response stream:**
```protobuf
message CharacterFragment {
  string text = 1;              // Rewritten text
  string character_id = 2;      // e.g., "yoda"
  bool is_provisional = 3;      // True if may be revised
  int64 original_start_ms = 4;
  int64 original_end_ms = 5;
  Metadata metadata = 6;
}
```

## Character Prompt Engineering

Each character has a few-shot prompt template:

```python
CHARACTER_PROMPTS = {
    "yoda": """You are Yoda from Star Wars. Rewrite the following text in Yoda's speaking style.
Rules:
- Invert subject-verb-object order
- Use wise, philosophical tone
- Keep meaning clear
- Be concise

Text: {text}
Yoda version:""",
    # ... more characters
}
```

## Performance Tuning

### For Lower Latency

```python
# Use smaller quantized model
model = Llama(
    model_path="llama-3-3b-q4_k_m.gguf",
    n_ctx=512,           # Smaller context
    n_gpu_layers=35      # Offload to GPU
)

# Generate with lower max_tokens
response = model(
    prompt,
    max_tokens=50,       # Limit generation
    temperature=0.7
)
```

### For Higher Quality

```python
# Use larger model
model = Llama(
    model_path="llama-3-8b-q8_0.gguf",  # Higher precision
    n_ctx=2048
)

# Generate with higher temperature
response = model(
    prompt,
    max_tokens=100,
    temperature=0.9,     # More creative
    top_p=0.95
)
```

## Mock Rewriter Mode

When no LLM is available, uses rule-based transformations:

- **Yoda:** Simple word order inversion
- **Vader:** Appends "...as you wish"
- **Obi-Wan:** Prepends "Well, "
- **Leia:** Converts to uppercase

This allows testing without model downloads.

## Example Usage

### Python Client

```python
from services.rewriter_llm.client import LLMRewriterClient

client = LLMRewriterClient(host='localhost', port=50053)

# Transcripts from STT service
transcripts = [
    {
        "text": "Hello there",
        "confidence": 0.95,
        "is_final": False,
        "start_time_ms": 0,
        "end_time_ms": 1000,
        "sequence": 0
    }
]

# Rewrite in Yoda style
for fragment in client.rewrite_stream(transcripts, character_id="yoda"):
    print(f"Yoda says: {fragment['text']}")

client.close()
```

### Integration with STT

```python
# Connect STT → Rewriter
for transcript in stt_client.transcribe_stream(audio):
    # Send to rewriter
    for fragment in rewriter_client.rewrite_stream([transcript], "yoda"):
        # Send to chunker
        chunker_client.chunk_text(fragment['text'])
```

## Adding New Characters

1. **Add prompt template:**
```python
CHARACTER_PROMPTS["gandalf"] = """You are Gandalf from Lord of the Rings...
Rules:
- Wise and mystical
- Often cryptic
...
"""
```

2. **Add mock transformation:**
```python
def _mock_rewrite(self, text, character_id):
    if character_id == "gandalf":
        return f"You shall {text.lower()}"
```

3. **Test:**
```python
client.rewrite_stream(transcripts, character_id="gandalf")
```

## Model Recommendations

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Llama 3 3B Q4 | ~2GB | Very Fast | Good | Real-time, CPU |
| Llama 3 8B Q4 | ~5GB | Fast | Better | Real-time, GPU |
| Llama 3 8B Q8 | ~9GB | Medium | Best | Quality priority |

## Troubleshooting

**Slow inference:**
- Use smaller/quantized model
- Enable GPU layers: `n_gpu_layers=35`
- Reduce `max_tokens`

**Out of memory:**
- Use more aggressive quantization (Q4_K_M)
- Reduce context size: `n_ctx=512`
- Use smaller model (3B instead of 8B)

**Poor character accuracy:**
- Improve prompt engineering
- Use larger model
- Increase temperature for creativity
- Add more examples to prompt

**Import errors:**
- Install llama-cpp-python: `pip install llama-cpp-python`
- For GPU: rebuild with CUDA support
- Service falls back to mock mode if unavailable
