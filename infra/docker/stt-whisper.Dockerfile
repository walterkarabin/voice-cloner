FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /app

# Copy proto library
COPY libs/proto /app/libs/proto
COPY libs/audio /app/libs/audio

# Generate proto code
RUN cd /app/libs/proto && \
    pip3 install grpcio==1.48.2 grpcio-tools==1.48.2 protobuf>=3.19.0,<3.20 && \
    ./generate.sh

# Copy service
COPY services/stt-whisper /app/services/stt-whisper

# Install dependencies
RUN cd /app/services/stt-whisper && pip3 install -r requirements.txt

ENV PYTHONPATH=/app
ENV SERVICE_PORT=50052

EXPOSE 50052

# Default to mock mode, override with --model-size for real
CMD ["python3", "/app/services/stt-whisper/server.py", "--port", "50052"]
