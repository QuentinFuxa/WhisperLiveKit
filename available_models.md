# Available model sizes:

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