# Benchmarks

The README's published scatter plots remain unchanged. Their coordinates match
the archived JSON, but the retained data cannot establish a ranking of current
backends. New runs use the same report as `wlk bench` and write to a separate
`benchmarks/runs/` directory.

## Reproducible corpus

```bash
pip install 'whisperlivekit[test]'
python scripts/prepare_fleurs_benchmark.py
wlk bench --backend faster-whisper --model base --languages en \
  --manifest benchmarks/corpora/fleurs-90.json --warmup --repeats 3 \
  --speed 1 --json results.json
```

The [fixed manifest](corpora/fleurs-90.json) contains 30 English, 30 French and
30 Mandarin Chinese recordings from the FLEURS test split at a pinned Hub
revision. Seed 0 selects 30 shared sentence IDs; each language uses the first
recording by filename for that ID. The original and normalized references,
recording IDs, extraction boundaries and audio hashes are retained. Audio is
cached locally, not committed. The source license is
[CC BY 4.0](https://huggingface.co/datasets/google/fleurs).

FLEURS recordings use floating-point WAV. Preparation converts them to 16 kHz
mono PCM16 and records both original and converted hashes. The runner rejects
files that differ from the manifest. Re-running preparation uses the pinned
source files; it does not select new samples.

The manifest also describes a continuous ten-minute stream per language,
concatenated from its clips, with exact component boundaries. Use
`--continuous --repeats 1` for resource and boundary checks. The last component
may be truncated, so these streams have no aggregate quality score. Their
component references remain available for inspection.

`--config options.json` supplies additional engine settings, including pinned
`model_path`/decoder/encoder paths. Unknown fields are rejected. Explicit local
model files are hashed after timing and snapshot revisions are recorded. An
alias such as `base` alone does not pin a model; resolve it to a snapshot before
publishing a comparison. Keep downloads outside measured runs.

## Measurements

JSON schema 3.2 includes hypotheses, references, WER/CER edit counts, errors,
audio hashes, effective configuration, dependency versions, hardware, source
commit and dirty state. It adds the translation half of an
audio-to-translation run: `target_language`, `translation`,
`translation_reference`, `chrf`, `final_translation_time_s` and
`provisional_lead_s` per row, `corpus_chrf` in the summary, and both halves of
the configuration — `config.asr` with the backend and model that recognized the
audio, `config.translation` with the source languages, target language, backend
and model that translated it. Analyzer scripts under
`benchmarks/evaluations/` remain pinned to the 3.1 reports they were written for.
Separate startup/warmup records do not contribute to measured quality or
latency. All measured repetitions retain their sample IDs.

Paced audio packets are sent when their final sample becomes available, using
absolute deadlines. Processing and scheduler delays do not accumulate into
every later deadline; backpressure can still delay delivery. EOF follows the
last write directly, including when the last packet is short. Older 3.0 reports
used relative waits after packets, which gave decoding an artificial interval
before EOF; do not compare their latency values with 3.1 runs.

- **ASR RTF:** successful inference-call time divided by audio duration.
- **Startup:** harness/model initialization. For a fresh process, the separate
  warmup record includes the initial load and model warmup; this is not a cold
  filesystem-cache measurement.
- **First visible text:** delay from feed start to the first provisional or
  committed ASR text. First committed text and first translation are separate.
  These fields are measured only at speed 1 and are not per-word latencies.
- **Finalization:** time between actual feeding completion and complete EOF
  drainage. **Source-end lag** instead uses nominal audio duration and includes
  delays from a slow producer or backpressure. Both are needed when feeding
  falls behind. No artificial sleep precedes EOF.
- **Process RSS:** largest sample taken every 50 ms, including startup. This is
  a sampled process peak, not a model-size estimate or an exact OS high-water
  mark. It includes other models retained in that process.
- **MLX memory:** allocator peak since the per-sample reset, plus active
  allocations at completion. It is separate from RSS; do not add them on unified
  memory. Missing memory support is represented by null fields.

Summary p95 values pool the retained per-excerpt measurements using linear
interpolation. ASR-call p95 remains a different measure: the last 4096 inference
calls per excerpt, never an average presented as a global percentile. English
and French use normalized word error rate; Chinese uses character error rate
with whitespace removed. Text normalization is recorded in the report.

Failures and timeouts remain visible, keep their partial hypothesis, and have
null quality scores. Translation errors also invalidate a measured sample.
The CLI exits nonzero on failure. A successful `finish()` is distinct from the
intentionally partial `cut()` operation. Compare successful sample identities,
failures, timestamp validity and repeat-level results before comparing scores.

## Compare configurations

```bash
python scripts/run_scatter_benchmark.py --aware --lang en \
  --manifest benchmarks/corpora/fleurs-90.json --repeats 3
python scripts/run_scatter_benchmark.py --plot-only benchmarks/runs/RUN/en_aware.json
```

The existing `--aware`, `--unaware`, `--output`, `--json-output` and `--plot-only`
commands remain available. Without `--manifest`, the script reads the previous
local `long_samples.json`. `--combos configurations.json` accepts an explicit
array of backend/model/policy/label/kwargs configurations. Each configuration
runs in a fresh subprocess; a worker crash remains an error record. Reports are
saved after each configuration. All new figures default to their run directory;
published root figures cannot be overwritten.

Immediate feeding (`--unaware`, speed 0) changes queue batching and scheduling.
Paced feeding (`--aware`, speed 1) uses backpressure rather than intentionally
losing audio. Neither mode's ASR RTF alone establishes user-visible latency.
Plots show fixed-size markers and complete/failing sample counts, without an
unmeasured “sweet spot” or a purported real-time throughput threshold.

For the first M5 backend selection, require a reproducible improvement of at
least 20% in finalization p95 or measured memory, at most one absolute percentage
point worse WER/CER, and no streaming defect. Inspect each of three warmed
passes, not only the pooled result. An ambiguous result leaves the candidate PR
open. A new backend must also justify its adapter and dependency cost.

## Published historical figures

The [July 2026 H100 archive](archive/h100_20260711/) retains the original figures,
aggregate JSON and plotting script. The root copies and README references were
restored and remain published. Their limitations are unchanged:

- Four recordings per language total 390 seconds, rounded to “6min” in the
  figures. References, audio hashes and extraction recipes were not retained.
- The sample-generation helper's LibriSpeech/MLS concatenations do not recreate
  the LibriVox/Gutenberg material named in those results.
- Only aggregate WER/RTF survive. Failed samples were dropped by the script;
  hypotheses, edit counts, completion states, revisions and effective settings
  cannot be recovered from those aggregates.
- RTF measures ASR calls. The “real-time limit” is not a pipeline latency bound;
  the shaded “sweet spot” changes with panel scaling. Marker-size labels do not
  accurately describe all plotted models, and some labels overlap.
- English SimulStreaming turbo WER of 18.1% versus 7.0% for small needs an actual
  rerun to explain; replotting cannot resolve it.

The [March H100/M5 archive](archive/README.md) uses chapter-grouped audio on H100
and per-utterance audio on M5. It does not show that a device changes recognition
quality. Replotting either archive cannot recover missing evidence.

For simultaneous translation, see the [MLX calibration notes](../docs/simul_mt_calibration.md),
[AlignAtt4LLM evaluation code](https://github.com/QuentinFuxa/Alignatt4LLM) and
[paper](https://arxiv.org/abs/2606.03967).

## Audio-to-translation runs

`--target-language` engages the translation half of the same pipeline and turns
a run into an audio-to-translation experiment. `--translation-backend` selects
the translator and `--mt-model` its model; both require `--target-language`, and
a flag wins over the same key in `--config`. The run records **both halves**:
each row keeps its `effective_config`, and `config.translation` states the
target language, the translation backend, the loaded model and whether the
backend drafted. Where the model runs somewhere the client cannot see — the
alignatt server, or the `nllw` default that resolves NLLB's checkpoint from its
size and backend — the record carries a `model_note` and the endpoint or
settings it called instead of claiming a model id.

```bash
wlk bench --backend nemotron-mlx-asr --languages en --target-language zh \
  --translation-backend mlx-llm-mt --mt-model hy-mt2-1.8b-8bit \
  --manifest benchmarks/corpora/fleurs-90.json --speed 1 --json results.json
```

FLEURS is parallel, so the manifest load joins rows on `sentence_id`: an English
recording's translation reference is the row of the same ID in the target
language. No second corpus and no download. A row whose ID has no
target-language sibling scores `null` instead of borrowing a reference.

- **Translation:** the translation text of each committed line, joined. When no
  committed line carries one, the row falls back to the in-flight buffer, and on
  a drafting backend that buffer holds a **provisional** translation rather than
  the settled one, so the scored text can be a draft. The run's
  `config.translation.simultaneous` or its alignatt backend, and a non-null
  `provisional_lead_s`, say when that run could draft; a fallback draft is
  scored as it stood at end of stream.
- **chrF:** sacrebleu's chrF, `CHRF(char_order=6, word_order=0, beta=2)`: the
  character n-gram F-beta of the translation against the same `sentence_id`'s
  reference in the target language, with whitespace removed and case and
  punctuation kept. It scores ASR and MT together, so ASR errors flow into it.
  The summary's `corpus_chrf` is the number to quote: sacrebleu's corpus score
  for the pair, from the n-gram counts summed over every scored row. The per-row
  `chrf` and their unweighted mean `avg_chrf` stay for inspection; a mean of
  per-sentence scores is not a corpus score. The per-row value is null on a row
  with no translation hypothesis or no translation reference.
- **First translation:** delay from feed start to the first provisional or
  committed translation. Paced runs only, like first visible text.
- **Final translation:** the last committed line that first showed a
  translation, also from feed start. A translation that keeps growing on one
  line does not move it.
- **Provisional lead:** final minus first translation time, so how long drafting
  ran before the translation settled. It is null unless the **configured
  backend drafts** — the alignatt server, or MLX with
  `mlx_llm_mt_simultaneous` — which is the same convention as a null CER for
  English: a property of the run, not a missing measurement. A non-empty buffer
  is not evidence of drafting: a backend not known to draft counts as one, and
  whatever sits in its buffer is not read as a draft, so the field stays null.

An audio-to-translation run measures the whole cascade on the emitted output; it
does not isolate the translation commit policy. The **realizable commit
fraction** is that other measurement's number: it comes from
`bench_commit_fraction.py`, the calibrated-attention coverage script owned by
the `km` simultaneous-frontier work. When that script is not on the base, the
run reports the row as `not run` rather than reimplementing a second capture
path.

A demonstration report from this machine is checked in at
[evaluations/translation-fleurs-en-zh](evaluations/translation-fleurs-en-zh/),
with its provenance and its limits in `NOTES.md`: three samples without drafting
and the same three with it. They exist to show the join, the score and the
latency fields on a real cascade run; three samples establish no quality figure
and are not a backend comparison.

