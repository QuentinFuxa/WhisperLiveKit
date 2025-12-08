FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG EXTRAS
ARG HF_PRECACHE_DIR
ARG HF_TKN_FILE

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        git \
        build-essential \
        python3-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# timeout/retries for large torch wheels
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 --disable-pip-version-check install --timeout=120 --retries=5 \
        --index-url https://download.pytorch.org/whl/cu129 \
        torch torchaudio \
    || (echo "Initial install failed — retrying with extended timeout..." && \
        pip3 --disable-pip-version-check install --timeout=300 --retries=3 \
            --index-url https://download.pytorch.org/whl/cu129 \
            torch torchvision torchaudio)

COPY . .

# Install WhisperLiveKit directly, allowing for optional dependencies
# Example: --build-arg EXTRAS="translation"
RUN if [ -n "$EXTRAS" ]; then \
      echo "Installing with extras: [$EXTRAS]"; \
      pip install --no-cache-dir "whisperlivekit[$EXTRAS]"; \
    else \
      echo "Installing base package only"; \
      pip install --no-cache-dir whisperlivekit; \
    fi

# In-container caching for Hugging Face models by: 
# A) Make the cache directory persistent via an anonymous volume.
#    Note: This only persists for a single, named container. This is 
#          only for convenience at de/test stage. 
#          For prod, it is better to use a named volume via host mount/k8s.
VOLUME ["/root/.cache/huggingface/hub"]


# or
# B) Conditionally copy a local pre-cache from the build context to the 
#    container's cache via the HF_PRECACHE_DIR build-arg.
#    WARNING: This will copy ALL files in the pre-cache location.

# Conditionally copy a cache directory if provided
RUN if [ -n "$HF_PRECACHE_DIR" ]; then \
      echo "Copying Hugging Face cache from $HF_PRECACHE_DIR"; \
      mkdir -p /root/.cache/huggingface/hub && \
      cp -r $HF_PRECACHE_DIR/* /root/.cache/huggingface/hub; \
    else \
      echo "No local Hugging Face cache specified, skipping copy"; \
    fi

# Conditionally copy a Hugging Face token if provided. Useful for Diart backend (pyannote audio models)
RUN if [ -n "$HF_TKN_FILE" ]; then \
      echo "Copying Hugging Face token from $HF_TKN_FILE"; \
      mkdir -p /root/.cache/huggingface && \
      cp $HF_TKN_FILE /root/.cache/huggingface/token; \
    else \
      echo "No Hugging Face token file specified, skipping token setup"; \
    fi

EXPOSE 8000

ENTRYPOINT ["whisperlivekit-server", "--host", "0.0.0.0"]

CMD ["--model", "medium"]
