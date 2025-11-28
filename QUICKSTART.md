# Voice Pipeline Quickstart

Transform your voice into a Star Wars character in real-time.

## Setup (One-time)

```bash
# 1. Install CLI dependencies
cd /workspace/apps/cli
pip3 install -r requirements.txt

# 2. Start services with Docker
cd /workspace/infra/compose
docker-compose up -d
```

Wait 30 seconds for services to start.

## Run

```bash
cd /workspace/apps/cli
python3 voice_pipeline.py --character yoda
```

Speak into your microphone. Your voice comes out as Yoda!

## Characters

- `yoda` - Inverted speech ("Strong with you, the Force is")
- `vader` - Commanding tone
- `obi-wan` - Wise diplomat
- `leia` - Assertive leader

Change character: `python3 voice_pipeline.py --character vader`

## Audio Devices

```bash
# List devices
python3 voice_pipeline.py --list-devices

# Use specific microphone/speaker
python3 voice_pipeline.py --character yoda --input-device 2
```

## Stop

- Press `Ctrl+C` to stop the CLI
- Stop services: `docker-compose down`

## Troubleshooting

**No audio**: Check microphone isn't muted, verify with `--list-devices`

**Connection error**: Ensure services running with `docker-compose ps`

**Slow/laggy**: Services use mock mode by default. For better quality, see TESTING.md for real model setup.

---

That's it! Speak and hear yourself as a character.
