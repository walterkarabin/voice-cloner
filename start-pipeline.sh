#!/bin/bash
# Start the character voice pipeline

set -e

echo "=== Character Voice Pipeline Startup ==="
echo

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose not found"
    echo "Please install docker-compose to use this script"
    exit 1
fi

# Parse arguments
MODE=${1:-"mock"}
CHARACTER=${2:-"yoda"}

case "$MODE" in
    docker)
        echo "Starting services with Docker Compose..."
        cd /workspace/infra/compose
        docker-compose up -d
        echo
        echo "✓ Services started"
        echo "  View logs: docker-compose logs -f"
        echo "  Stop: docker-compose down"
        echo
        echo "To run the CLI:"
        echo "  cd /workspace/apps/cli"
        echo "  python3 voice_pipeline.py --character $CHARACTER --host localhost"
        ;;

    local)
        echo "Starting services locally..."
        echo "NOTE: Run each service in a separate terminal:"
        echo
        echo "  Terminal 1: cd /workspace/services/embed-loader && python3 server.py --port 50051"
        echo "  Terminal 2: cd /workspace/services/stt-whisper && python3 server.py --port 50052"
        echo "  Terminal 3: cd /workspace/services/rewriter-llm && python3 server.py --port 50053"
        echo "  Terminal 4: cd /workspace/services/chunker && python3 server.py --port 50054"
        echo "  Terminal 5: cd /workspace/services/tts-streamer && python3 server.py --port 50055"
        echo "  Terminal 6: cd /workspace/services/vocoder && python3 server.py --port 50056"
        echo "  Terminal 7: cd /workspace/services/audio-out && python3 server.py --port 50057"
        echo "  Terminal 8: cd /workspace/apps/cli && python3 voice_pipeline.py --character $CHARACTER"
        ;;

    mock)
        echo "Starting mock mode (testing without models)..."
        echo "This will start all services in mock mode for testing"
        echo
        cd /workspace/infra/compose
        docker-compose up -d
        echo
        echo "✓ Services started in mock mode"
        ;;

    *)
        echo "Usage: $0 [docker|local|mock] [character]"
        echo
        echo "Modes:"
        echo "  docker  - Start all services with Docker Compose"
        echo "  local   - Show commands to start services locally"
        echo "  mock    - Start services in mock mode (default)"
        echo
        echo "Characters:"
        echo "  yoda, vader, obi-wan, leia (default: yoda)"
        echo
        echo "Examples:"
        echo "  $0 docker yoda"
        echo "  $0 local vader"
        echo "  $0 mock"
        exit 1
        ;;
esac
