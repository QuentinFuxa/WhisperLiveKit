# Simultaneous-MT calibration heads

The in-process simultaneous-MT variant (`MlxLlmTranslationSimul`, enabled
with `--mlx-llm-mt-simultaneous`) drafts a translation over the unstable
ASR tail and releases only the target prefix whose attention aligns to
source the ASR has committed. This is the AlignAtt commit policy, ported
to MLX — see [Alignatt4LLM](https://github.com/QuentinFuxa/Alignatt4LLM)
for the method, the attention-head detection tooling, and the promotion
gate (TS > 0.1 for ≥8 heads, stability splits).

## Calibrated alignment heads are required per (model, source, target)

The commit policy reads the cross-attention of a small set of
**translation alignment heads** — the (layer, head) indices whose
attention tracks the source words being translated. These are
direction-specific and must be detected per (model, source, target)
tuple. The registry lives in `whisperlivekit/simul_mt_capture.py`
(`CALIBRATION_REGISTRY`). Existing entries (tencent/Hy-MT2-1.8B):

| direction | top head | TS | heads | disabled quants |
|---|---|---|---|---|
| zh→en | L9/H5 | 0.79 | 8 | 4bit |
| en→zh | L9/H5 | 0.86 | 8 | 4bit |
| ja→zh | L9/H5 | 0.89 | 8 | 4bit |

All three share the top head (L9/H5) — strong evidence these are general
alignment heads for the `hunyuan_v1_dense` architecture, not
direction-specific.

## Adding a new (model, source, target) direction

Detection is done with the AlignAtt4LLM tooling — run
`detect_translation_heads.py` (see the Alignatt4LLM repo for the
annotation + detection pipeline and the promotion gate). This produces
a `translation_heads_<model>_<direction>.json` with a
`token_alignment_heads` array sorted by TS. Transcribe the top-8 heads
into `whisperlivekit/simul_mt_capture.py`:

1. Add three module-level constants near the existing ones:
   - `<DIR>_ALIGNMENT_HEADS: List[Tuple[int,int]]` (top 8)
   - `<DIR>_HEAD_TS_SCORES: Dict[Tuple[int,int], float]`
   - `<DIR>_TOP_HEAD: Tuple[int,int]` (the top entry)
2. Add an entry to `CALIBRATION_REGISTRY` keyed by the NORMALIZED model
   id (org prefix and quant suffix stripped — `hy-mt2-1.8b`, not
   `mlx-community/Hy-MT2-1.8B-8bit`):
   ```python
   ("hy-mt2-1.8b", "<src>", "<tgt>"): CalibrationEntry(
       heads=<DIR>_ALIGNMENT_HEADS,
       ts_scores=<DIR>_HEAD_TS_SCORES,
       top_head=<DIR>_TOP_HEAD,
       disabled_quants={"4bit"},  # only if probed and found divergent
   ),
   ```

The 8bit heads are calibrated on the bf16 base model and transfer to the
8bit MLX quant. The 4bit quant was probed (48.9% argmax match vs 8bit —
attention diverges) and is in `disabled_quants` (silently deactivates).

## Fallback when no calibration exists

If the (model, source, target) tuple has no registry entry — or the
requested quant is in `disabled_quants` — the variant silently degrades
to the base `MlxLlmTranslation` (translate-on-close, no provisional, no
capture). At init it emits:

```
WARNING: MlxLlmTranslationSimul: no calibration for
(model=mlx-community/Hy-MT2-1.8B-8bit, src=en, tgt=zh) — deactivating
simultaneous mode (translation works via base; no provisional)
```

Translation is correct, just not simultaneous. At runtime,
`mt._simul_active` (`True`/`False`) indicates which path is active;
`mt.wants_hypothesis_tail` mirrors it.
