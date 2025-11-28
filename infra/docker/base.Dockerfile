# Base image for Python services
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy proto library
COPY libs/proto /app/libs/proto

# Copy audio library
COPY libs/audio /app/libs/audio

# Generate proto code
RUN cd /app/libs/proto && \
    pip3 install grpcio==1.48.2 grpcio-tools==1.48.2 protobuf>=3.19.0,<3.20 && \
    ./generate.sh

# Set Python path
ENV PYTHONPATH=/app

# Health check helper
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import grpc; grpc.insecure_channel('localhost:${SERVICE_PORT}').channel_ready()" || exit 1
