# Audio-to-translation demonstration run

Two schema 3.2 reports from the same machine (Apple Silicon, M-series, macOS),
made to show the `sentence_id` join, the chrF score and the translation latency
fields on a real cascade, not to measure either backend.

| File | Run | Samples | Commit |
|---|---|---|---|
| `report.json` | nemotron-mlx-asr + mlx-llm-mt hy-mt2-1.8b-8bit, no drafting | 3 | `0721830` |
| `report-simultaneous.json` | the same cascade with `mlx_llm_mt_simultaneous` | 3 | `0ffb22d` |

Both were taken at `--speed 1` over the same three English samples, from models
already in the local HuggingFace cache (`HF_HUB_OFFLINE=1`, nothing downloaded).
Both report `git_dirty: false`.

`--config` takes a **path** to a JSON file, not an inline object: `cli.py` reads
it with `Path(parsed.config).read_text()`. Writing the drafting option inline
(`--config '{"mlx_llm_mt_simultaneous": true}'`) fails with
`FileNotFoundError: '{"mlx_llm_mt_simultaneous": true}'`. So the drafting run
needs its option in a file first:

```bash
printf '{"mlx_llm_mt_simultaneous": true}\n' > /tmp/simulst_simultaneous_config.json

wlk bench --backend nemotron-mlx-asr --languages en --target-language zh \
  --translation-backend mlx-llm-mt --mt-model hy-mt2-1.8b-8bit \
  --manifest /tmp/simulst_subset_manifest.json --speed 1 \
  --json report.json

wlk bench --backend nemotron-mlx-asr --languages en --target-language zh \
  --translation-backend mlx-llm-mt --mt-model hy-mt2-1.8b-8bit \
  --config /tmp/simulst_simultaneous_config.json \
  --manifest /tmp/simulst_subset_manifest.json --speed 1 \
  --json report-simultaneous.json
```

The two commands differ only in the `--config` file. `report-simultaneous.json`
was written to `/tmp` and copied in afterwards, so the worktree was clean when
the run started.

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

The two reports carry different `corpus_sha256` values because their subset
manifests were serialized separately. They select the same six rows: the same
three English sample names, the same three durations (7.2, 9.72, 8.66 s) and the
same three Chinese references.

## What the numbers do and do not say

`report.json` scores 5.30, 10.48 and 5.23 chrF (mean 7.01); with drafting,
`report-simultaneous.json` scores 5.30, 10.48 and 5.60 (mean 7.13). Three
samples, one repetition and an uncontrolled thermal state establish no backend
quality and no comparison with any other run; the low scores reflect this
cascade, which the row data shows directly: `fleurs_en_1700` transcribes
word-for-word (WER 0.00) while the MT output diverges, and `fleurs_en_1993`
already loses words in ASR ("the tenth named storm" -> "The tensioning storm").
Read them as evidence that the harness scores what the pipeline emitted.

The two runs are not a drafting A/B either. ASR is identical on both (WER 0.00,
13.64 and 22.22) and chrF agrees to six decimals on `fleurs_en_1700` and
`fleurs_en_1921`, while `fleurs_en_1993` moved 5.23 -> 5.60: the incremental MT
input depends on chunk timing, so a repeat of one configuration is not
bit-reproducible on the translated text.

`provisional_lead_s` is null on all three rows of `report.json` and 1.19, 6.02
and 5.03 s in `report-simultaneous.json`: the field needs a backend that drafts
provisional text, so a null there is a property of the run, not a missing
measurement. The drafting run shows the intended shape — on `fleurs_en_1921`
first translation at 2.44 s and final at 8.45 s inside a 9.72 s sample, with
`final_translation_time_s >= first_translation_time_s` on every row.
