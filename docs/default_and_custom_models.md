# Models and Model Paths

## Defaults

**Default Whisper Model**: `base`  
When no model is specified, WhisperLiveKit uses the `base` model, which provides a good balance of speed and accuracy for most use cases.

**Default Model Cache Directory**: `~/.cache/whisper`  
Models are automatically downloaded from OpenAI's model hub and cached in this directory. You can override this with `--model_cache_dir`.

**Default Translation Model**: `600M` (NLLB-200-distilled)  
When translation is enabled, the 600M distilled NLLB model is used by default. This provides good quality with minimal resource usage.

**Default Translation Backend**: `transformers`  
The translation backend defaults to Transformers. On Apple Silicon, this automatically uses MPS acceleration for better performance.

---


## Available Whisper model sizes:

| Available Model    | Speed    | Accuracy  | Multilingual | Translation | Hardware Requirements | Best Use Case                   |
|--------------------|----------|-----------|--------------|-------------|----------------------|----------------------------------|
| tiny(.en)          | Fastest  | Basic     | Yes/No       | Yes/No      | ~1GB VRAM            | Real-time, low resources         |
| base(.en)          | Fast     | Good      | Yes/No       | Yes/No      | ~1GB VRAM            | Balanced performance             |
| small(.en)         | Medium   | Better    | Yes/No       | Yes/No      | ~2GB VRAM            | Quality on limited hardware      |
| medium(.en)        | Slow     | High      | Yes/No       | Yes/No      | ~5GB VRAM            | High quality, moderate resources |
| large-v2           | Slowest  | Excellent | Yes          | Yes         | ~10GB VRAM           | Good overall accuracy & language support          |
| large-v3           | Slowest  | Excellent | Yes          | Yes         | ~10GB VRAM           | Best overall accuracy & language support                |
| large-v3-turbo     | Fast     | Excellent | Yes          | No          | ~6GB VRAM            | Fast, high-quality transcription |


### How to choose?

#### Language Support
- **English only**: Use `.en` (ex: `base.en`) models for better accuracy and faster processing when you only need English transcription
- **Multilingual**: Do not use `.en` models.
      
#### Special Cases
- **No translation needed**: Use `large-v3-turbo`
  - Same transcription quality as `large-v2` but significantly faster
  - **Important**: Does not translate correctly, only transcribes

### Additional Considerations

**Model Performance**:
- Accuracy improves significantly from tiny to large models
- English-only models are ~10-15% more accurate for English audio
- Newer versions (v2, v3) have better punctuation and formatting

**Audio Quality Impact**:
- Clean, clear audio: smaller models may suffice
- Noisy, accented, or technical audio: larger models recommended
- Phone/low-quality audio: use at least `small` model

_______________________


# Custom Models:

The `--model-path` parameter accepts:

## File Path
- **`.pt` / `.bin` / `.safetensor` formats** Should be openable by pytorch/safetensor.

## Directory Path (recommended)
Must contain:
- **`.pt` / `.bin` / `.safetensor` file** (required for decoder)

May optionally contain:
- **`.bin` file** - faster-whisper model for encoder (requires faster-whisper)
- **`weights.npz`** or **`weights.safetensors`** - for encoder (requires whisper-mlx)

## Hugging Face Repo ID
- Provide the repo ID (e.g. `openai/whisper-large-v3`) and WhisperLiveKit will download and cache the snapshot automatically. For gated repos, authenticate via `huggingface-cli login` first.

To improve speed/reduce hallucinations, you may want to use `scripts/determine_alignment_heads.py` to determine the alignment heads to use for your model, and use the `--custom-alignment-heads` to pass them to WLK. If not, alignment heads are set to be all the heads of the last half layer of decoder.


_______________________

# Translation Models and Backend

**Language Support**: ~200 languages

## Distilled Model Sizes Available

| Model | Size | Parameters | VRAM (FP16) | VRAM (INT8) | Quality |
|-------|------|------------|-------------|-------------|---------|
| 600M | 2.46 GB | 600M | ~1.5GB | ~800MB | Good, understandable |
| 1.3B | 5.48 GB | 1.3B | ~3GB | ~1.5GB | Better accuracy, context |

**Quality Impact**: 1.3B has ~15-25% better BLEU scores vs 600M across language pairs.

## Backend Performance

| Backend | Speed vs Base | Memory Usage | Quality Loss |
|---------|---------------|--------------|--------------|
| CTranslate2 | 6-10x faster | 40-60% less | ~5% BLEU drop |
| Transformers | Baseline | High | None |
| Transformers + MPS (on Apple Silicon) | 2x faster | Medium | None |

**Metrics**:
- CTranslate2: 50-100+ tokens/sec
- Transformers: 10-30 tokens/sec
- Apple Silicon with MPS: Up to 2x faster than CTranslate2
