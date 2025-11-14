# 1. Simulstreaming: Decouple the encoder for faster inference

Simulstreaming encoder time (whisperlivekit/simul_whisper/simul_whisper.py l. 397) experimentations :

On macOS Apple Silicon M4 :

| Encoder | base.en | small |
|--------|---------|-------|
| WHISPER (no modification) | 0.35s | 1.09s |
| FASTER_WHISPER | 0.4s | 1.20s |
| MLX_WHISPER | 0.07s | 0.20s |

Memory saved by only loading encoder for optimized framework:

For tiny.en, mlx whisper:
Sizes MLX whisper:
Decoder weights: 59110771 bytes
Encoder weights: 15268874 bytes


# 2. Translation: Faster model for each system

## Benchmark Results

Testing on MacBook M3 with NLLB-200-distilled-600M model:

### Standard Transformers vs CTranslate2

| Test Text | Standard Inference Time | CTranslate2 Inference Time | Speedup |
|-----------|-------------------------|---------------------------|---------|
| UN Chief says there is no military solution in Syria | 0.9395s | 2.0472s | 0.5x |
| The rapid advancement of AI technology is transforming various industries | 0.7171s | 1.7516s | 0.4x |
| Climate change poses a significant threat to global ecosystems | 0.8533s | 1.8323s | 0.5x |
| International cooperation is essential for addressing global challenges | 0.7209s | 1.3575s | 0.5x |
| The development of renewable energy sources is crucial for a sustainable future | 0.8760s | 1.5589s | 0.6x |

**Results:**
- Total Standard time: 4.1068s
- Total CTranslate2 time: 8.5476s
- CTranslate2 is slower on this system --> Use Transformers, and ideally we would have an mlx implementation.


# 3. SortFormer Diarization: 4-to-2 Speaker Constraint Algorithm

Transform a diarization model that predicts up to 4 speakers into one that predicts up to 2 speakers by mapping the output predictions.

## Problem Statement
- Input: `self.total_preds` with shape `(x, x, 4)` - predictions for 4 speakers
- Output: Constrained predictions with shape `(x, x, 2)` - predictions for 2 speakers

#
### Initial Setup
For each time step `i`, we have a ranking of 4 speaker predictions (1-4). When only 2 speakers are present, the model will have close predictions for the 2 active speaker positions.

Instead of `np.argmax(preds_np, axis=1)`, we take the top 2 predictions and build a dynamic 4→2 mapping that can evolve over time.

### Algorithm

```python
top_2_speakers = np.argsort(preds_np, axis=1)[:, -2:]
```

- `DS_a_{i}`: Top detected speaker for prediction i
- `DS_b_{i}`: Second detected speaker for prediction i  
- `AS_{i}`: Attributed speaker for prediction i
- `GTS_A`: Ground truth speaker A
- `GTS_B`: Ground truth speaker B
- `DIST(a, b)`: Distance between detected speakers a and b

3. **Attribution Logic**

```
AS_0 ← A

AS_1 ← B

IF DIST(DS_a_0, DS_a_1) < DIST(DS_a_0, DS_a_2) AND 
    DIST(DS_a_0, DS_a_1) < DIST(DS_a_1, DS_a_2):
    # Likely that DS_a_0 = DS_a_1 (same speaker)
    AS_1 ← A
    AS_2 ← B

ELIF DIST(DS_a_0, DS_a_2) < DIST(DS_a_0, DS_a_1) AND 
    DIST(DS_a_0, DS_a_2) < DIST(DS_a_1, DS_a_2):
    AS_2 ← A

ELSE:
    AS_2 ← B

to finish
```
