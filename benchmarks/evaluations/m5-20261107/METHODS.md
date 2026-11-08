# M5 streaming backend screening: 7 November 2026

This series repeats the [September screening](../m5-20260907/METHODS.md) with
all sixteen reports measured in one serial session on AC power. It replaces
the partial September series as the evidence for
[#425](https://github.com/QuentinFuxa/WhisperLiveKit/pull/425) and
[#426](https://github.com/QuentinFuxa/WhisperLiveKit/pull/426). The September
directory is kept unchanged as history, including its power transition.

## Unchanged from September

The corpus, procedure, run order, report schema, text normalization, reading
of the measurements and selection rule are those of the
[September method](../m5-20260907/METHODS.md). In short: the
[fixed FLEURS manifest](../../corpora/fleurs-90.json) (SHA-256
`95586eb74f53e665f35983f53da31ea510867b54c7fe5f481466e2d6af2619e5`), one fresh
process per backend and language, one warmup clip, three passes of 30 clips at
real-time feed speed, then one ten-minute continuous Chinese stream per backend.
Order: Whisper, Nemotron, native Qwen, Qwen Metal; English, French, Chinese,
continuous Chinese within each backend.

The measured source revisions and model revisions are also the same:

| Backend | Source commit | Model revision |
|---|---|---|
| Whisper small MLX / LocalAgreement | `b2baee9` | `mlx-community/whisper-small-mlx@45f3915` |
| Qwen3 0.6B / Metal | `b2baee9`, `qwen3-asr-causal` submodule `8975258` | `Qwen/Qwen3-ASR-0.6B@5eb1441` |
| Qwen3 0.6B / native MLX | `24082a0` (#425) | `Qwen/Qwen3-ASR-0.6B@5eb1441` |
| Nemotron 0.6B / native MLX | `1842f4c` (#426) | `mlx-community/nemotron-3.5-asr-streaming-0.6b@e550040` |

Model files were downloaded again at these revisions; the hashes recorded in the
reports match September. These results describe the commits above, not the
current main branch.

## Power and host

All runs took place on the same Apple M5 host (32 GB) between 18:46 and 22:58
UTC. `pmset` was sampled every 60 seconds: all 253 samples report AC power,
battery between 74% and 80%, and no recorded thermal or performance warning
([power log](power-log.jsonl.gz)). The runner only starts a run on AC power and
flags a run that observed another source; no run was paused or flagged
([runs.jsonl](runs.jsonl)). Ordinary desktop activity was not controlled.

## Environments

Each candidate checkout was synchronized from its own lock with
`uv sync --frozen`, the backend extra and `test`, on Python 3.12.12.
[Package versions](environment-packages.json) are recorded next to those
stored in each report.

- Nemotron and native Qwen: the recorded packages match September.
- Whisper: numpy 2.3.5 instead of 2.4.4 and huggingface-hub 1.14.0 instead of
  1.30.0. mlx-lm, Transformers, safetensors and sentencepiece were installed in
  September but not here; the mlx-whisper path does not import them.
- Qwen Metal: the existing vllm-metal 0.2.0 environment with vLLM 0.21.0+cpu
  and Transformers 5.8.1, left unchanged. WLK and `qwen3-asr-causal` are loaded
  from source through `PYTHONPATH`, with missing audio and benchmark packages
  installed into a separate `--target --no-deps` overlay. September's overlay
  also contained mlx-audio 0.5.1, which requires Transformers 5.14 or later and
  is not imported on the vllm-metal path; it is omitted here.

The Metal backend still builds an automatic-language prompt even when `lan` is
set, while native Qwen and Nemotron receive the session language.

## Additional measurements

The native Qwen continuous stream failed during the series. It was run once
more after the series under the same conditions to check reproducibility; that
report is kept separately as
[qwen-native-zh-continuous-rerun.json.gz](qwen-native-zh-continuous-rerun.json.gz)
and is not part of the summary.

After the series, [`measure_startup.py`](../m5-20260907/measure_startup.py)
recorded fresh-process import and engine initialization for each backend with
the cached model ([startup/](startup/)). It includes constructor warmup and
excludes HTTP server readiness; it is not a cold filesystem-cache measurement.

## Reproduction

```bash
python run_series.py --workspace WORKSPACE --manifest benchmarks/corpora/fleurs-90.json \
  --metal-overlay WORKSPACE/metal-overlay --output OUTPUT
python analyze.py OUTPUT/reports --output OUTPUT
python ../m5-20260907/plot.py OUTPUT/summary.json --output OUTPUT/figure.png
```

`run_series.py` documents the expected workspace layout. Each configuration file
contains only the pinned snapshot path option listed in the September method,
plus `"backend_policy": "localagreement"` for Whisper. `analyze.py` is the
September script with the power note updated; the gate computation is
unchanged.
