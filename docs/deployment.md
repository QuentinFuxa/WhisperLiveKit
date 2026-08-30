# Deployment

## Local server and remote clients

```bash
wlk --model base --language en --host 0.0.0.0 --port 8000
```

For remote browser microphone access, serve the page over HTTPS and connect over WSS. TLS can terminate at a reverse proxy or in WLK using both `--ssl-certfile` and `--ssl-keyfile`.

Set `WLK_API_TOKEN` or `--api-token` to require authentication for transcription requests. REST clients use an `Authorization: Bearer` header; native WebSocket clients can also use `?token=...`. See [the API reference](API.md).

Start with one worker. Each worker loads its own model weights, and inference locks serialize some backend operations within a process. Adding four workers can multiply model memory by four; measure your backend with concurrent sessions before increasing the process count.

A reverse proxy must forward WebSocket upgrades. For example, inside an Nginx server block:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 3600s;
}
```

Configure `--forwarded-allow-ips` for the proxy addresses you trust. Use `--cors-origins` when a separate web origin calls the HTTP API.

## GPU environments

For source installs, initialize the submodule before using uv:

```bash
git clone --recurse-submodules https://github.com/QuentinFuxa/WhisperLiveKit.git
cd WhisperLiveKit

# Choose one profile in each environment:
uv sync --extra cu129 --extra diarization-sortformer
# or
uv sync --extra cu129 --extra voxtral-hf --extra translation
# or
uv sync --extra qwen3-vllm
```

The `cpu` and `cu129` extras select PyTorch indexes through uv's source configuration. Plain pip does not use `[tool.uv.sources]`; choose the PyTorch build separately for pip-based environments.

`qwen3-vllm` uses its own CUDA dependency stack and conflicts with `cu129`. Other backend combinations also conflict; [pyproject.toml](../pyproject.toml) declares the full list. Keep the [AlignAtt4LLM translation server](translation-alignatt.md) in its own environment.

The [dependency audit](dependency-audit.md) records security fixes, affected
backend environments and upgrades blocked by upstream compatibility constraints.

## Docker

Initialize submodules before building either image:

```bash
git submodule update --init --recursive
```

CPU:

```bash
docker build -f Dockerfile.cpu -t wlk-cpu --build-arg EXTRAS=cpu .
docker run --rm -p 8000:8000 wlk-cpu
```

NVIDIA GPU, with the NVIDIA Container Toolkit available on the host:

```bash
docker build -t wlk-gpu .
docker run --rm --gpus all -p 8000:8000 wlk-gpu --model medium
```

Models download on first use. Persist Hugging Face downloads by mounting a cache volume:

```bash
docker run --rm --gpus all -p 8000:8000 \
    -v hf-cache:/root/.cache/huggingface/hub \
    -e HF_TOKEN -e WLK_API_TOKEN \
    wlk-gpu --model base --language en
```

For an image with diarization, build with `--build-arg EXTRAS=cu129,diarization-sortformer` and start it with `--diarization`. Available Compose services are `wlk-cpu`, `wlk-gpu-sortformer`, and `wlk-gpu-voxtral`; run one explicitly, for example:

```bash
docker compose up --build wlk-cpu
```

The CPU and GPU Sortformer services both bind host port 8000. Change a port mapping before starting them together. See [compose.yml](../compose.yml) for each service's installed extras and cache setup.
