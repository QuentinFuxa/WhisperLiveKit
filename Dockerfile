FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin

# --- MARK: Builder Stage
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder-gpu
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  python3-dev && \
  rm -rf /var/lib/apt/lists/*

# Install UV and set up the environment 
COPY --from=uvbin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_PYTHON_INSTALL_DIR=/python

RUN uv python install 3.12

# Install dependencies first to leverage caching
ARG EXTRAS=cu129
COPY pyproject.toml uv.lock /app/
RUN set -eux; \
  set --; \
  for extra in $(echo "${EXTRAS:-}" | tr ',' ' '); do \
  set -- "$@" --extra "$extra"; \
  done; \
  uv sync --frozen --no-install-project --no-editable --no-cache "$@"

# Copy the source code and install the package only
COPY whisperlivekit /app/whisperlivekit
RUN set -eux; \
  set --; \
  for extra in $(echo "${EXTRAS:-}" | tr ',' ' '); do \
  set -- "$@" --extra "$extra"; \
  done; \
  uv sync --frozen --no-editable --no-cache "$@"

# --- MARK: Runtime Stage 
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  ffmpeg &&\
  rm -rf /var/lib/apt/lists/*

# Copy UV binaries
COPY --from=uvbin /uv /uvx /bin/

# Copy the Python version
COPY --from=builder-gpu --chown=python:python /python /python

# Copy the virtual environment with all dependencies installed
COPY --from=builder-gpu /app/.venv /app/.venv

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_PYTHON_DOWNLOADS=0

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

ENTRYPOINT ["whisperlivekit-server", "--host", "0.0.0.0"]

CMD ["--model", "medium"]
