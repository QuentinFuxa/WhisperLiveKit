# AlignAtt calibration on MLX

The local simultaneous translator uses selected **self-attention heads** to
relate target tokens to source words. It shares the head-file convention and
Translation Score calculation with [AlignAtt4LLM](https://github.com/QuentinFuxa/Alignatt4LLM).
The [IWSLT paper](https://arxiv.org/abs/2606.03967) remains the reference for the
method. These MLX measurements are a separate experiment with a different
runtime and prompt; they do not reproduce the paper's system-level results.

## Use a calibration

```bash
wlk --backend mlx-whisper --model small --language en --target-language zh \
    --translation-backend mlx-llm-mt --mlx-llm-mt-model hy-mt2-1.8b-8bit \
    --simultaneous
```

The bundled calibration covers **Hy-MT2-1.8B-8bit, English → Chinese**, at revision
`f54bb3b8885363fd8b83d63d50b50a12a138321f`. It scored 862 of 880 annotated pairs;
18 unusable annotations are listed in the [full calibration report](../whisperlivekit/calibrations/translation_heads_mlx-community_Hy-MT2-1_8B-8bit_en-zh.json).
All three stability checks retained the same eight heads, with a maximum TS
change of 0.0131. Use `--mlx-llm-mt-calibration /path/to/heads.json` for an external file.

The file must match the exact repository, model revision, quantization, language
direction and chat prompt. A directory can hold files named
`translation_heads_<model>_<direction>.json`, using underscores for repository
slashes and model-name dots. Sessions select their own direction from that
directory. Unsupported directions raise a configuration error. Cantonese is not
silently treated as Mandarin, and a 4-bit checkpoint cannot reuse an 8-bit entry.

The JSON retains AlignAtt4LLM fields such as `model`, `direction`,
`token_alignment_heads`, `stability_checks` and `promotion_gate`. Its `runtime`
metadata additionally records MLX versions, the exact model revision,
quantization, prompt and alignment-file SHA-256. A PyTorch head file without
these MLX measurements must be recalibrated before local use.

## Reproduce head selection

Install the MLX translation extra in a development checkout. Download the
word-alignment JSON from a pinned AlignAtt4LLM commit, then run:

```bash
python scripts/calibrate_ts_mlx.py \
    --model-id hy-mt2-1.8b-8bit \
    --revision f54bb3b8885363fd8b83d63d50b50a12a138321f \
    --direction en-zh --alignments word_alignments_en-zh.json \
    --source-url https://github.com/QuentinFuxa/Alignatt4LLM/blob/2a76daf3f2e522a3f08948b72b8a27b612981991/data/alignatt_heads/word_alignments_en-zh.json \
    --output heads.json
```

Each aligned target token is scored by whether the head's full-sequence
attention argmax reaches a gold-aligned source token. Scores are averaged per
pair, matching the research implementation. Three half-sample stability checks
use seed 13. Promotion requires at least 100 scored pairs, eight heads above
TS 0.1, the same top-eight set in every check and a maximum score change of 0.03.
Unusable word annotations remain listed in the JSON; model or capture failures
abort the run. An unsuccessful promotion gate saves its report and exits nonzero.

## Runtime behavior and limits

Capture runs only while holding the shared model lock. The original attention
modules are restored after every generation, including failures, and each draft
owns an immutable attention snapshot. Releasing held tokens after another
session runs therefore uses the original draft's evidence. Missing attention
raises `translation_error` and preserves the last accepted text.

Only `hunyuan_v1_dense` is currently instrumented. Model output uses mlx-lm's
native fused attention; selected head probabilities are computed separately.
The regression checks compare full prefill with cached continuation and verify
that the causal mask includes the cached prefix.

Accepted partials grow within an utterance. A final translation at punctuation,
a pause, a speaker change or EOF can replace the partial at the line level.
Quality, first-text delay and finalization delay must be measured separately
before making performance claims about an ASR/MT combination.
