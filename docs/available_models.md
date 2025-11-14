# Available Whisper model sizes:

- tiny.en (english only)
- tiny
- base.en (english only)
- base
- small.en (english only)
- small
- medium.en (english only)
- medium
- large-v1
- large-v2
- large-v3
- large-v3-turbo

## How to choose?

### Language Support
- **English only**: Use `.en` models for better accuracy and faster processing when you only need English transcription
- **Multilingual**: Do not use `.en` models.

### Resource Constraints
- **Limited GPU/CPU or need for very low latency**: Choose `small` or smaller models
  - `tiny`: Fastest, lowest resource usage, acceptable quality for simple audio
  - `base`: Good balance of speed and accuracy for basic use cases
  - `small`: Better accuracy while still being resource-efficient
- **Good resources available**: Use `large` models for best accuracy
  - `large-v2`: Excellent accuracy, good multilingual support
  - `large-v3`: Best overall accuracy and language support

### Special Cases
- **No translation needed**: Use `large-v3-turbo`
  - Same transcription quality as `large-v2` but significantly faster
  - **Important**: Does not translate correctly, only transcribes

### Model Comparison Table

| Model | Speed | Accuracy | Multilingual | Translation | Best Use Case |
|-------|--------|----------|--------------|-------------|---------------|
| tiny(.en) | Fastest | Basic | Yes/No | Yes/No | Real-time, low resources |
| base(.en) | Fast | Good | Yes/No | Yes/No | Balanced performance |
| small(.en) | Medium | Better | Yes/No | Yes/No | Quality on limited hardware |
| medium(.en) | Slow | High | Yes/No | Yes/No | High quality, moderate resources |
| large-v2 | Slowest | Excellent | Yes | Yes | Best overall quality |
| large-v3 | Slowest | Excellent | Yes | Yes | Maximum accuracy |
| large-v3-turbo | Fast | Excellent | Yes | No | Fast, high-quality transcription |

### Additional Considerations

**Model Performance**:
- Accuracy improves significantly from tiny to large models
- English-only models are ~10-15% more accurate for English audio
- Newer versions (v2, v3) have better punctuation and formatting

**Hardware Requirements**:
- `tiny`: ~1GB VRAM
- `base`: ~1GB VRAM  
- `small`: ~2GB VRAM
- `medium`: ~5GB VRAM
- `large`: ~10GB VRAM
- `large‑v3‑turbo`: ~6GB VRAM

**Audio Quality Impact**:
- Clean, clear audio: smaller models may suffice
- Noisy, accented, or technical audio: larger models recommended
- Phone/low-quality audio: use at least `small` model

### Quick Decision Tree
1. English only? → Add `.en` to your choice
2. Limited resources or need speed? → `small` or smaller
3. Good hardware and want best quality? → `large-v3`
4. Need fast, high-quality transcription without translation? → `large-v3-turbo`
5. Need translation capabilities? → `large-v2` or `large-v3` (avoid turbo)


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

## Quick Decision Matrix

**Choose 600M**: Limited resources, close to 0 lag
**Choose 1.3B**: Quality matters
**Choose Transformers**: On Apple Silicon

