FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /app

COPY libs/proto /app/libs/proto
COPY libs/audio /app/libs/audio

RUN cd /app/libs/proto && \
    pip3 install grpcio==1.48.2 grpcio-tools==1.48.2 protobuf>=3.19.0,<3.20 && \
    ./generate.sh

COPY services/tts-streamer /app/services/tts-streamer

RUN cd /app/services/tts-streamer && pip3 install -r requirements.txt

ENV PYTHONPATH=/app
ENV SERVICE_PORT=50055

EXPOSE 50055

# Default to mock mode
CMD ["python3", "/app/services/tts-streamer/server.py", "--port", "50055", "--use-mock"]
