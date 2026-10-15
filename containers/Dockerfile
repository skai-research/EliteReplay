FROM vllm/vllm-openai@sha256:1acd26aae64b2816e2692f74f6d4b8d4efef412d7491d33d02b3537f5b804376
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
