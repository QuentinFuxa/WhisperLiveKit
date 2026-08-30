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

JSON schema 3 includes hypotheses, references, WER/CER edit counts, errors,
audio hashes, effective configuration, dependency versions, hardware, source
commit and dirty state. Separate startup/warmup records do not contribute to
measured quality or latency. All measured repetitions retain their sample IDs.

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
[paper](https://arxiv.org/abs/2606.03967). A WLK audio-to-translation benchmark is a
separate experiment; it must record both ASR and MT configurations.
