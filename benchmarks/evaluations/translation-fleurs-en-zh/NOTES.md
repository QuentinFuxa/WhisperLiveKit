# Audio-to-translation demonstration run

Two schema 3.2 reports from the same machine (Apple Silicon, M-series, macOS),
made to show the `sentence_id` join, the chrF score and the translation latency
fields on a real cascade, not to measure either backend.

| File | Run | Samples | Commit |
|---|---|---|---|
| `report.json` | nemotron-mlx-asr + mlx-llm-mt hy-mt2-1.8b-8bit, no drafting | 3 | `06a902c` |
| `report-simultaneous.json` | the same cascade with `mlx_llm_mt_simultaneous` | 3 | `06a902c` |

Both were taken at `--speed 1` over the same three English samples, from models
already in the local HuggingFace cache (`HF_HUB_OFFLINE=1`, nothing downloaded).
Both report `git_dirty: false` and the same `corpus_sha256`, because this pair
used one subset manifest for both runs.

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

The two commands differ only in the `--config` file. Both reports were written
to `/tmp` and copied in afterwards, so the worktree was clean when each run
started.

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
against the manifest `audio_sha256` by the runner. Six rows select: three English
samples of 7.2, 9.72 and 8.66 s, and the three Chinese references.

## What the numbers do and do not say

chrF here is sacrebleu's, `CHRF(char_order=6, word_order=0, beta=2)`, and the
figure to quote is the summary's `corpus_chrf`: the n-gram counts summed over
all three rows and scored once, as sacrebleu scores a corpus.

| Report | `corpus_chrf` | Per-row `chrf` | `avg_chrf` |
|---|---|---|---|
| `report.json` | 8.59 | 5.58, 14.67, 5.46 | 8.57 |
| `report-simultaneous.json` | 9.32 | 8.76, 14.76, 5.49 | 9.67 |

The two aggregations are different numbers, which is the reason the summary
carries both: `avg_chrf` is the unweighted mean of the per-row scores and stays
for inspection only.

**Three samples, one repetition each, an uncontrolled thermal state and a
single machine establish no quality figure.** Nothing here is a backend
comparison, a drafting A/B, or a claim about this cascade's translation quality.
The rows show why: `fleurs_en_1700` transcribes word-for-word (WER 0.00) while
the MT output diverges from the reference, and `fleurs_en_1993` already loses
words in ASR ("the tenth named storm" -> "The tensioning storm"). Read them as
evidence that the harness scores what the pipeline emitted.

The two runs are not a drafting A/B either. ASR is identical on both (WER 0.00,
13.64, 22.22), while the translated text differs on all three rows: the
incremental MT input depends on chunk timing and on whether the backend drafts,
so repeating one configuration is not bit-reproducible on the MT side. This pair
replaces the one checked in before the chrF correction; its values are not
comparable with the superseded pair, whose per-row scores came from a variant of
chrF that also lowercased and stripped punctuation.

`provisional_lead_s` is null on all three rows of `report.json` and 1.36, 8.80
and 6.84 s in `report-simultaneous.json`: the field needs a backend that drafts
provisional text, so a null there is a property of the run, not a missing
measurement. The drafting run shows the intended shape — on `fleurs_en_1921`
first translation at 2.45 s and final at 11.25 s inside a 9.72 s sample, with
`final_translation_time_s >= first_translation_time_s` on every row.
