<h1 align="center">WLK</h1>
<p align="center"><b>WhisperLiveKit: Ultra-low-latency, self-hosted speech-to-text with speaker identification</b></p>


<p align="center">
<img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/demo.png" alt="WhisperLiveKit Demo" width="730">
</p>


<p align="center">
<a href="https://pypi.org/project/whisperlivekit/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/whisperlivekit?color=g"></a>
<a href="https://pepy.tech/project/whisperlivekit"><img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/whisperlivekit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=installations"></a>
<a href="https://pypi.org/project/whisperlivekit/"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.9--3.15-dark_green"></a>
<a href="https://huggingface.co/qfuxa/whisper-base-french-lora">
  <img alt="Hugging Face Weights" src="https://img.shields.io/badge/🤗-Hugging%20Face%20Weights-yellow" />
</a>
<a href="https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-dark_green"></a>
</p>


#### Powered by Leading Research:

- Simul-[Whisper](https://arxiv.org/pdf/2406.10052)/[Streaming](https://arxiv.org/abs/2506.17077) (SOTA 2025) - Ultra-low latency transcription using [AlignAtt policy](https://arxiv.org/pdf/2305.11408)
- [NLLW](https://github.com/QuentinFuxa/NoLanguageLeftWaiting) (2025), based on [distilled](https://huggingface.co/entai2965/nllb-200-distilled-600M-ctranslate2) [NLLB](https://arxiv.org/abs/2207.04672) (2022, 2024) - Simulatenous translation from & to 200 languages.
- [WhisperStreaming](https://github.com/ufal/whisper_streaming) (SOTA 2023) - Low latency transcription using [LocalAgreement policy](https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.pdf)
- [Streaming Sortformer](https://arxiv.org/abs/2507.18446) (SOTA 2025) - Advanced real-time speaker diarization
- [Diart](https://github.com/juanmc2005/diart) (SOTA 2021) - Real-time speaker diarization
- [Silero VAD](https://github.com/snakers4/silero-vad) (2024) - Enterprise-grade Voice Activity Detection


> **Why not just run a simple Whisper model on every audio batch?** Whisper is designed for complete utterances, not real-time chunks. Processing small segments loses context, cuts off words mid-syllable, and produces poor transcription. WhisperLiveKit uses state-of-the-art simultaneous speech research for intelligent buffering and incremental processing.


### Architecture

<img alt="Architecture" src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/architecture.png" />

*The backend supports multiple concurrent users. Voice Activity Detection reduces overhead when no voice is detected.*

### Installation & Quick Start

```bash
pip install whisperlivekit
```
> You can also clone the repo and `pip install -e .` for the latest version.

#### Quick Start
1. **Start the transcription server:**
   ```bash
   wlk --model base --language en
   ```

2. **Open your browser** and navigate to `http://localhost:8000`. Start speaking and watch your words appear in real-time!


> - See [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/simul_whisper/whisper/tokenizer.py) for the list of all available languages.
> - Check the [troubleshooting guide](docs/troubleshooting.md) for step-by-step fixes collected from recent GPU setup/env issues.
> - The CLI entry point is exposed as both `wlk` and `whisperlivekit-server`; they are equivalent.
> - For HTTPS requirements, see the **Parameters** section for SSL configuration options.


#### Use it to capture audio from web pages.

Go to `chrome-extension` for instructions.

<p align="center">
<img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/chrome-extension/demo-extension.png" alt="WhisperLiveKit Demo" width="600">
</p>



#### Optional Dependencies

| Optional | `pip install` |
|-----------|-------------|
| **Windows/Linux optimizations** | `faster-whisper` |
| **Apple Silicon optimizations** | `mlx-whisper` |
| **Translation** | `nllw` |
| **Speaker diarization** | `git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]` |
| OpenAI API | `openai` |
| *[Not recommanded]*  Speaker diarization with Diart | `diart` |

See  **Parameters & Configuration** below on how to use them.



### Usage Examples

**Command-line Interface**: Start the transcription server with various options:

```bash
# Large model and translate from french to danish
wlk --model large-v3 --language fr --target-language da

# Diarization and server listening on */80 
wlk --host 0.0.0.0 --port 80 --model medium --diarization --language fr
```


**Python API Integration**: Check [basic_server](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/basic_server.py) for a more complete example of how to use the functions and classes.

```python
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args

transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(model="medium", diarization=True, lan="en")
    yield

app = FastAPI(lifespan=lifespan)

async def handle_websocket_results(websocket: WebSocket, results_generator):
    async for response in results_generator:
        await websocket.send_json(response)
    await websocket.send_json({"type": "ready_to_stop"})

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine

    # Create a new AudioProcessor for each connection, passing the shared engine
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)    
    results_generator = await audio_processor.create_tasks()
    results_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))
    await websocket.accept()
    while True:
        message = await websocket.receive_bytes()
        await audio_processor.process_audio(message)        
```

**Frontend Implementation**: The package includes an HTML/JavaScript implementation [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/web/live_transcription.html). You can also import it using `from whisperlivekit import get_inline_ui_html` & `page = get_inline_ui_html()`


## Parameters & Configuration


| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Whisper model size. List and recommandations [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/default_and_custom_models.md) | `small` |
| `--model-path` | Local .pt file/directory **or** Hugging Face repo ID containing the Whisper model. Overrides `--model`. Recommandations [here](https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/docs/default_and_custom_models.md) | `None` |
| `--language` | List [here](docs/supported_languages.md). If you use `auto`, the model attempts to detect the language automatically, but it tends to bias towards English. | `auto` |
| `--target-language` | If sets, translates using [NLLW](https://github.com/QuentinFuxa/NoLanguageLeftWaiting). [200 languages available](docs/supported_languages.md). If you want to translate to english, you can also use `--direct-english-translation`. The STT model will try to directly output the translation. | `None` |
| `--diarization` | Enable speaker identification | `False` |
| `--backend-policy` | Streaming strategy: `1`/`simulstreaming` uses AlignAtt SimulStreaming, `2`/`localagreement` uses the LocalAgreement policy | `simulstreaming` |
| `--backend` | Whisper implementation selector. `auto` picks MLX on macOS (if installed), otherwise Faster-Whisper, otherwise vanilla Whisper. You can also force `mlx-whisper`, `faster-whisper`, `whisper`, or `openai-api` (LocalAgreement only) | `auto` |
| `--no-vac` | Disable Voice Activity Controller. NOT ADVISED | `False` |
| `--no-vad` | Disable Voice Activity Detection. NOT ADVISED | `False` |
| `--warmup-file` | Audio file path for model warmup | `jfk.wav` |
| `--host` | Server host address | `localhost` |
| `--port` | Server port | `8000` |
| `--ssl-certfile` | Path to the SSL certificate file (for HTTPS support) | `None` |
| `--ssl-keyfile` | Path to the SSL private key file (for HTTPS support) | `None` |
| `--forwarded-allow-ips` | Ip or Ips allowed to reverse proxy the whisperlivekit-server. Supported types are  IP Addresses (e.g. 127.0.0.1), IP Networks (e.g. 10.100.0.0/16), or Literals (e.g. /path/to/socket.sock) | `None` |
| `--pcm-input` | raw PCM (s16le) data is expected as input and FFmpeg will be bypassed. Frontend will use AudioWorklet instead of MediaRecorder | `False` |
| `--lora-path` | Path or Hugging Face repo ID for LoRA adapter weights (e.g., `qfuxa/whisper-base-french-lora`). Only works with native Whisper backend (`--backend whisper`) | `None` |

| Translation options | Description | Default |
|-----------|-------------|---------|
| `--nllb-backend` | `transformers` or `ctranslate2` | `ctranslate2` |
| `--nllb-size` | `600M` or `1.3B` | `600M` |

| Diarization options | Description | Default |
|-----------|-------------|---------|
| `--diarization-backend` |  `diart` or `sortformer` | `sortformer` |
| `--disable-punctuation-split` | [NOT FUNCTIONAL IN 0.2.15 / 0.2.16] Disable punctuation based splits. See #214 | `False` |
| `--segmentation-model` | Hugging Face model ID for Diart segmentation model. [Available models](https://github.com/juanmc2005/diart/tree/main?tab=readme-ov-file#pre-trained-models) | `pyannote/segmentation-3.0` |
| `--embedding-model` | Hugging Face model ID for Diart embedding model. [Available models](https://github.com/juanmc2005/diart/tree/main?tab=readme-ov-file#pre-trained-models) | `speechbrain/spkrec-ecapa-voxceleb` |

| SimulStreaming backend options | Description | Default |
|-----------|-------------|---------|
| `--disable-fast-encoder` | Disable Faster Whisper or MLX Whisper backends for the encoder (if installed). Inference can be slower but helpful when GPU memory is limited | `False` |
| `--custom-alignment-heads` | Use your own alignment heads, useful when `--model-dir` is used. Use `scripts/determine_alignment_heads.py` to extract them. <img src="scripts/alignment_heads.png" alt="WhisperLiveKit Demo" width="300">
 | `None` |
| `--frame-threshold` | AlignAtt frame threshold (lower = faster, higher = more accurate) | `25` |
| `--beams` | Number of beams for beam search (1 = greedy decoding) | `1` |
| `--decoder` | Force decoder type (`beam` or `greedy`) | `auto` |
| `--audio-max-len` | Maximum audio buffer length (seconds) | `30.0` |
| `--audio-min-len` | Minimum audio length to process (seconds) | `0.0` |
| `--cif-ckpt-path` | Path to CIF model for word boundary detection | `None` |
| `--never-fire` | Never truncate incomplete words | `False` |
| `--init-prompt` | Initial prompt for the model | `None` |
| `--static-init-prompt` | Static prompt that doesn't scroll | `None` |
| `--max-context-tokens` | Maximum context tokens | Depends on model used, but usually 448. |



| WhisperStreaming backend options | Description | Default |
|-----------|-------------|---------|
| `--confidence-validation` | Use confidence scores for faster validation | `False` |
| `--buffer_trimming` | Buffer trimming strategy (`sentence` or `segment`) | `segment` |




> For diarization using Diart, you need to accept user conditions [here](https://huggingface.co/pyannote/segmentation) for the `pyannote/segmentation` model, [here](https://huggingface.co/pyannote/segmentation-3.0) for the `pyannote/segmentation-3.0` model and [here](https://huggingface.co/pyannote/embedding) for the `pyannote/embedding` model. **Then**, login to HuggingFace: `huggingface-cli login`

### 🚀 Deployment Guide

To deploy WhisperLiveKit in production:
 
1. **Server Setup**: Install production ASGI server & launch with multiple workers
   ```bash
   pip install uvicorn gunicorn
   gunicorn -k uvicorn.workers.UvicornWorker -w 4 your_app:app
   ```

2. **Frontend**: Host your customized version of the `html` example & ensure WebSocket connection points correctly

3. **Nginx Configuration** (recommended for production):
    ```nginx    
   server {
       listen 80;
       server_name your-domain.com;
        location / {
            proxy_pass http://localhost:8000;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
    }}
    ```

4. **HTTPS Support**: For secure deployments, use "wss://" instead of "ws://" in WebSocket URL

## 🐋 Docker

Deploy the application easily using Docker with GPU or CPU support.

### Prerequisites
- Docker installed on your system
- For GPU support: NVIDIA Docker runtime installed

### Quick Start

**With GPU acceleration (recommended):**
```bash
docker build -t wlk .
docker run --gpus all -p 8000:8000 --name wlk wlk
```

**CPU only:**
```bash
docker build -f Dockerfile.cpu -t wlk .
docker run -p 8000:8000 --name wlk wlk
```

### Advanced Usage

**Custom configuration:**
```bash
# Example with custom model and language
docker run --gpus all -p 8000:8000 --name wlk wlk --model large-v3 --language fr
```

### Memory Requirements
- **Large models**: Ensure your Docker runtime has sufficient memory allocated


#### Customization

- `--build-arg` Options:
  - `EXTRAS="translation"` - Add extras to the image's installation (no spaces). Remember to set necessary container options!
  - `HF_PRECACHE_DIR="./.cache/"` - Pre-load a model cache for faster first-time start
  - `HF_TKN_FILE="./token"` - Add your Hugging Face Hub access token to download gated models

## 🔮 Use Cases
Capture discussions in real-time for meeting transcription, help hearing-impaired users follow conversations through accessibility tools, transcribe podcasts or videos automatically for content creation, transcribe support calls with speaker identification for customer service...
