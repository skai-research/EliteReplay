FROM docker.io/vllm/vllm-openai@sha256:68b773151407ca28c05479c02c0c08a573285ba197c8ff35d2103acd97aff78c
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*
COPY requirements/training.txt /tmp/elitereplay-training.txt
RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python && \
    python -m pip install --no-cache-dir -r /tmp/elitereplay-training.txt && \
    python -m pip install --no-cache-dir --no-deps TransferQueue==0.1.7
ENV PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
