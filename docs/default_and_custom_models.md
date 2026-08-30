# Models and custom checkpoints

## Defaults

The default Whisper model is `base`. `--backend auto` selects an available encoder/backend; the selected runtime is reported at startup. Models are downloaded on first use unless local paths are supplied.

Whisper checkpoints use the Whisper cache; Faster-Whisper and MLX model repositories use Hugging Face caching. There is no single cache directory shared by every backend. Use `wlk models` to inspect locally available models and `--model_cache_dir` where supported by the selected backend.

Whisper sizes include `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, and `large-v3-turbo`. The `.en` variants of tiny/base/small/medium are English-only. Check quality and processing cost on your own audio with [the benchmark tools](../benchmarks/README.md); this repository does not establish a universal accuracy or speed ranking.

## Custom Whisper models

The SimulStreaming loader accepts `--model-path` as a local checkpoint file, a directory, or a Hugging Face repository ID. Explicit `--encoder-model-path` and `--decoder-model-path` let you supply separate formats for a hybrid encoder/decoder setup.

```bash
wlk --backend faster-whisper \
    --encoder-model-path /models/whisper-ct2 \
    --decoder-model-path /models/whisper-pytorch \
    --language en
```

| Format | Files inspected by WLK |
|---|---|
| PyTorch / safetensors decoder | `.pt`, `model.safetensors`, `pytorch_model.bin`, and recognized shard/index files |
| CTranslate2 encoder | `model.bin` or encoder/decoder binaries with vocabulary files distinguishing them from PyTorch weights |
| MLX encoder | `weights.npz` or `weights.safetensors` |

A CTranslate2-only directory cannot supply the PyTorch decoder used by SimulStreaming. Point the decoder at compatible Whisper weights. Format detection is implemented in [model_paths.py](../whisperlivekit/model_paths.py); loading and compatibility checks live in [simul_whisper/backend.py](../whisperlivekit/simul_whisper/backend.py).

For fine-tuned models, [determine_alignment_heads.py](../scripts/determine_alignment_heads.py) can help select heads for `--custom-alignment-heads`. `--lora-path` is available for the native Whisper path.

`--model_dir` is the older model-directory option. Its interpretation depends on the backend. In particular, with FunASR it points to a SenseVoiceSmall snapshot; it is not a general promise that every backend accepts every model format. See [backend setup](backends.md).

## Translation models

`--target-language` enables translation. The default in-process backend uses NLLW with NLLB-200-distilled-600M and `--nllb-backend transformers`. `--nllb-size 1.3B` and `--nllb-backend ctranslate2` select the other exposed options.

```bash
pip install "whisperlivekit[translation]"
wlk --model base --language en --target-language de
```

Translation throughput and memory depend on the language pair, source length, hardware, and runtime. Compare the actual configurations you intend to deploy.

For decoder-only LLM translation, use the separate [AlignAtt4LLM backend](translation-alignatt.md). Its supported language directions depend on the calibrated alignment heads installed in the translation server.
