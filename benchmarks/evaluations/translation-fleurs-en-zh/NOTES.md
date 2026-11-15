# Audio-to-translation demonstration run

Two schema 3.2 reports from the same machine (Apple Silicon, M-series, macOS),
made to show the `sentence_id` join, the chrF score and the translation latency
fields on a real cascade, not to measure either backend.

| File | Run | Samples |
|---|---|---|
| `report.json` | nemotron-mlx-asr + mlx-llm-mt hy-mt2-1.8b-8bit, no drafting | 3 |
| `report-simultaneous.json` | the same cascade with `mlx_llm_mt_simultaneous` | 1 |

Both were produced at commit `0721830` at `--speed 1`, three and one samples
respectively, from models already in the local HuggingFace cache
(`HF_HUB_OFFLINE=1`, nothing downloaded). `report.json` reports `git_dirty:
false`. `report-simultaneous.json` reports true because `report.json` was
already on disk untracked when it ran; the code was identical.

```bash
wlk bench --backend nemotron-mlx-asr --languages en --target-language zh \
  --translation-backend mlx-llm-mt --mt-model hy-mt2-1.8b-8bit \
  --manifest <subset manifest> --speed 1 --json report.json
wlk bench ... --config '{"mlx_llm_mt_simultaneous": true}' \
  --manifest <one-sample subset manifest> --speed 1 --json report-simultaneous.json
```

## Samples and manifest

The run uses three English rows of `benchmarks/corpora/fleurs-90.json`
(`fleurs_en_1700`, `fleurs_en_1921`, `fleurs_en_1993`) plus their Chinese
siblings. The checked-in corpus is unchanged; the subset manifest was written to
`/tmp` only to bound each run and is reproducible with:

```python
import json
data = json.load(open("benchmarks/corpora/fleurs-90.json"))
names, ids = ["fleurs_en_1700", "fleurs_en_1921", "fleurs_en_1993"], {1700, 1921, 1993}
data["samples"] = [r for r in data["samples"] if r["name"] in names
                   or (r["language"] == "zh" and r["sentence_id"] in ids)]
json.dump(data, open("/tmp/simulst_subset_manifest.json", "w"))
```

The Chinese rows supply the `translation_reference`; only the English rows are
fed to the pipeline. The audio is the cached FLEURS PCM16 file per row, checked
against the manifest `audio_sha256` by the runner.

## What the numbers do and do not say

`report.json` scores 5.30, 10.48 and 5.23 chrF (mean 7.01) on three samples and
`report-simultaneous.json` scores 12.30 on one. Three samples, one repetition
and an uncontrolled thermal state establish no backend quality and no
comparison with any other run; the low scores reflect this cascade, which the
row data shows directly: `fleurs_en_1700` transcribes word-for-word (WER 0.00)
while the MT output diverges, and `fleurs_en_1993` already loses words in ASR
("the tenth named storm" -> "The tensioning storm"). Read them as evidence that
the harness scores what the pipeline emitted.

`provisional_lead_s` is null in `report.json` and 5.67 s in
`report-simultaneous.json`: the field needs a backend that drafts provisional
text, so a null there is a property of the run, not a missing measurement. The
drafting run shows the intended shape — first translation at 2.43 s, final at
8.10 s, both inside a 9.72 s sample.
