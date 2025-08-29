FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        git \
        build-essential \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app


COPY . .

# --- THE KEY FIX ---
# Install PyTorch and the local WhisperLiveKit package in a SINGLE command.
# This allows pip's dependency resolver to find a compatible set of all
# packages at once, avoiding the downgrade conflicts.
RUN pip install torch torchvision torchaudio . --extra-index-url https://download.pytorch.org/whl/cu129

# Enable in-container caching for Hugging Face models by:
# Note: If running multiple containers, better to map a shared
# bucket.
#
# A) Make the cache directory persistent via an anonymous volume.
#    Note: This only persists for a single, named container. This is
#          only for convenience at de/test stage.
#          For prod, it is better to use a named volume via host mount/k8s.
VOLUME ["/root/.cache/huggingface/hub"]
ARG HF_PRECACHE_DIR
RUN if [ -n "$HF_PRECACHE_DIR" ]; then \
      echo "Copying Hugging Face cache from $HF_PRECACHE_DIR"; \
      mkdir -p /root/.cache/huggingface/hub && \
      cp -r $HF_PRECACHE_DIR/* /root/.cache/huggingface/hub; \
    fi

# Conditionally copy a Hugging Face token if provided
ARG HF_TKN_FILE
RUN if [ -n "$HF_TKN_FILE" ]; then \
      echo "Copying Hugging Face token from $HF_TKN_FILE"; \
      mkdir -p /root/.cache/huggingface && \
      cp $HF_TKN_FILE /root/.cache/huggingface/token; \
    fi

# Expose the application port
EXPOSE 8000

ENTRYPOINT ["whisperlivekit-server", "--host", "0.0.0.0"]

# Default args
CMD ["--model", "medium"]