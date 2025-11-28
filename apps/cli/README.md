# Voice Pipeline CLI

Command-line interface for the character voice pipeline. Captures audio from your microphone and outputs character-voiced audio in real-time.

## Installation

```bash
cd /workspace/apps/cli
pip3 install -r requirements.txt
```

## Usage

### List Audio Devices

```bash
python3 voice_pipeline.py --list-devices
```

This will show all available input and output devices with their indices.

### Run Pipeline

```bash
# With default devices
python3 voice_pipeline.py --character yoda

# With specific input device
python3 voice_pipeline.py --character vader --input-device 2

# Connect to remote services (Docker)
python3 voice_pipeline.py --character obi-wan --host localhost

# Enable debug logging
python3 voice_pipeline.py --character leia --debug
```

### Available Characters

- `yoda` - Yoda's inverted speech pattern
- `vader` - Darth Vader's commanding tone
- `obi-wan` - Obi-Wan Kenobi's wise demeanor
- `leia` - Princess Leia's assertive style

## How It Works

The CLI orchestrates the full pipeline:

1. **Audio Input**: Captures audio from your microphone
2. **STT**: Transcribes speech to text (Whisper)
3. **Rewriter**: Converts to character's speech pattern (LLM)
4. **Chunker**: Splits text into prosodic chunks
5. **TTS**: Generates mel spectrograms
6. **Vocoder**: Converts mel to audio
7. **Audio Out**: Plays through speakers

## Requirements

- All pipeline services must be running (see TESTING.md or use Docker Compose)
- Microphone access
- Speaker/headphone output
- Python 3.8+

## Troubleshooting

**No audio devices found:**
- Ensure PortAudio is installed: `sudo apt-get install portaudio19-dev`
- Check system audio devices are working

**Connection refused:**
- Ensure all services are running
- Check service ports (50051-50057)
- Use `--host` flag if services are in Docker

**Poor audio quality:**
- Check microphone is not muted
- Adjust input device sensitivity
- Ensure services are using real models (not mock mode)
