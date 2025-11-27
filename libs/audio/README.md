# Audio Utilities Library

Core audio processing utilities for the character voice pipeline.

## Components

### AudioResampler (`resample.py`)
High-quality audio resampling between different sample rates.

```python
from libs.audio import AudioResampler

# Resample from 48kHz to 16kHz
resampler = AudioResampler(input_rate=48000, output_rate=16000)
resampled = resampler.resample(audio_data)
```

### VoiceActivityDetector (`vad.py`)
Energy-based voice activity detection for identifying speech segments.

```python
from libs.audio import VoiceActivityDetector

vad = VoiceActivityDetector(
    sample_rate=16000,
    energy_threshold=0.01
)

# Process frame-by-frame
result = vad.process_frame(audio_frame)
if result['is_speech']:
    print(f"Speech detected with confidence: {result['confidence']}")
```

### AudioFramer (`framing.py`)
Split streaming audio into fixed-size frames with optional overlap.

```python
from libs.audio import AudioFramer

# Create framer for 30ms frames
framer = AudioFramer.from_duration(
    frame_duration_ms=30,
    sample_rate=16000,
    overlap_percent=0.25
)

# Add samples and get frames
for frame in framer.add_samples(new_audio_samples):
    process_frame(frame)
```

### Utility Functions (`utils.py`)
- `bytes_to_float32()`: Convert PCM bytes to float32 numpy array
- `float32_to_bytes()`: Convert float32 array to PCM bytes
- `normalize_audio()`: Normalize audio to target peak level
- `compute_rms()`: Calculate RMS energy

## Installation

```bash
pip install -r requirements.txt
```

## Usage in Services

This library is designed to be used by:
- **stt-whisper**: Frame audio for Whisper input, detect voice activity
- **audio-out**: Resample output audio to target sample rate
- **All services**: Format conversions between bytes and float arrays

## Future Enhancements

For production use, consider integrating:
- **WebRTC VAD**: More robust voice activity detection
- **Silero VAD**: Neural network-based VAD
- **librosa**: Advanced audio processing features
