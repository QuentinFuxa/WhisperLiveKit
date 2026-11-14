# Local translation on Apple Silicon

The `mlx-llm-mt` backend runs a translation model in the WhisperLiveKit process.
It translates completed sentences and flushes an unfinished sentence at a pause,
speaker change, or end of stream. It does not translate every partial ASR update.
The optional [simultaneous mode](simul_mt_calibration.md) can release partial
translations when a compatible MLX calibration is available.

```bash
pip install 'whisperlivekit[mlx-whisper,mlx-llm-mt]'
wlk --backend mlx-whisper --model small --language zh --target-language en \
    --translation-backend mlx-llm-mt --mlx-llm-mt-model hy-mt2-1.8b-8bit
```

This extra requires macOS on Apple Silicon. Model weights download on first use;
startup loads the model and runs a short generation before accepting sessions.
Model loading and generation are serialized because sessions share the MLX model
and tokenizer. Each session has separate source/target languages and buffers.
Concurrent sessions therefore share inference capacity.

`--mlx-llm-mt-model` selects a profile in
[`translation_profiles.py`](../whisperlivekit/translation_profiles.py), including
Hy-MT2, Hunyuan-MT and TranslateGemma. Profiles specify the checkpoint, prompt and
decoding parameters; their presence is not a quality or latency ranking. Use an
explicit source language for TranslateGemma's structured translation prompt.
`hunyuan-mlx` remains an alias of `mlx-llm-mt`.

The MLX translation extra uses Transformers 5. Install it separately from the
Qwen3 `qwen3-streaming`, `qwen3-vllm` and `qwen3-vllm-metal` extras, which use a
different dependency stack. The [remote AlignAtt4LLM backend](translation-alignatt.md)
can keep ASR and translation dependencies in separate processes.

If generation fails, `translation_error` reports the incomplete translation and
ASR can continue. A failed sentence stays queued for retry; successful sentences
are published once. An empty model response is an error, and untranslated source
words are never shown as translated output. Disconnecting a client interrupts
generation at the next generated token, including when inference workers are
fully occupied by other sessions.
