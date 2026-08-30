# Benchmarks

The repository currently has historical results, not a validated ranking of the current backends. Use them to investigate past behavior. For a comparison on your hardware, run the same audio through each configuration and retain the transcripts, references, and run settings.

## Run a local evaluation

```bash
pip install "whisperlivekit[test]"
wlk bench --backend faster-whisper --model base --speed 1 --json results.json
```

`wlk bench --help` lists the available language, category, and playback-speed options. Model weights and audio may be downloaded. The JSON export includes per-sample hypotheses, references, error counts, and timing fields.

The runner is a diagnostic tool, not a published ranking. JSON schema 2 records
completion status, errors, audio SHA-256, effective configuration, source commit
and dirty state when run from a checkout, dependency versions, hypotheses and
references. Failures remain in the report with null scores and cause the CLI to
exit with a nonzero status. Missing references produce null WER, not 0% WER.

`TestHarness.finish()` raises on timeout or pipeline error. The intentionally
partial `cut()` operation remains separate. No artificial sleep precedes EOF.

## What the metrics mean

- **ASR RTF** is successful inference-call time divided by audio duration.
- **Wall time** covers feeding and EOF drainage, excluding startup and cleanup.
- **Startup time** records harness/model initialization separately. Cached engines
  can make later startup times shorter.
- **First text time** is the delay from feeding to the first committed text,
  recorded only at `--speed 1`. It is not a per-word latency distribution.
- **ASR call mean/p95** describe inference-call durations for each sample; the
  p95 uses the last 4096 calls, while the mean and RTF use cumulative counters. The
  report does not average those p95s into a purported global percentile.
- **Memory** is not measured. A delta of process high-water marks is not a model
  memory measurement, so that field has been removed.

Only successful samples contribute to aggregate scores. Compare their identities
and failure counts before comparing aggregates. `--speed 0` changes queue batching
and inference scheduling relative to paced audio; retain the speed with results.
Model revisions are not automatically resolved: pin and record the actual model
artifacts before using a run in a published comparison.

## Why the README scatter plots were retired

The [July 2026 H100 archive](archive/h100_20260711/) contains the original figures, aggregate JSON, and plotting script. The plotted coordinates agree with those JSON files. The interpretation and provenance are insufficient for a current comparison:

- Each language has four samples totalling **390 seconds (6 min 30 s)**. The figures round that to “6min”. The old README named 30/60/120/180-second clips, but the archive does not contain their references, audio hashes, or extraction recipe.
- The checked-in sample-generation helper creates roughly 90-second LibriSpeech/MLS concatenations. It does not recreate the LibriVox/Gutenberg samples named in these results.
- Only aggregate WER and RTF survive. No hypotheses, word-error counts, per-sample scores, model revisions, effective configuration, dependency versions, or run commit are recorded. The numbers cannot be recomputed from the archive alone.
- The script silently drops failed samples before averaging. All recorded rows report four successes, but completion and timeout status were not saved.
- The RTF axis measures time in ASR calls. The “real-time limit” at 1 does not certify the whole pipeline's throughput or the latency seen by a user.
- “Sweet spot” has no measured criterion. Its height changes with the largest WER in each figure, so the shaded areas do not even represent a fixed threshold across panels.
- Marker sizes are manually assigned. The legend says “small / 4B” while the plotted Qwen models are 0.6B, and it omits the turbo size.
- The forced axis range leaves the unpaced results crowded at the left; several labels overlap. The English SimulStreaming turbo WER of 18.1% versus 7.0% for small deserves investigation, not omission or an assumed explanation.

The original figures remain unchanged in the archive as a record of what was published. Root-level duplicate PNGs were removed. Replotting the same aggregates would improve the layout but would not recover the missing evidence.

The [March H100/M5 archive](archive/README.md) also predates removed backends. Its H100/M5 comparison uses chapter-grouped audio on H100 and per-utterance audio on M5; the plot itself acknowledges that the WERs are not directly comparable. It should not be used to claim that one device improves transcription quality.

## Requirements for the next published comparison

Use one explicit corpus manifest with audio hashes, references, languages, durations, and extraction boundaries. Run a fixed matrix and save the effective config, model revisions, WLK commit, dependency versions, hardware, and feeding schedule with every result. Save each sample's hypothesis, edit counts, completion status, and any error.

Measure stable word commit latency separately from processing time, using reference word boundaries and the wall clock at emission. Include end-of-stream flush time. Compare quality and latency across repeated runs on the same samples; retain failures and report sample counts. Diarization and translation need their own metrics and test material.

For AlignAtt4LLM, link to the [paper](https://arxiv.org/abs/2606.03967) and [its evaluation code](https://github.com/QuentinFuxa/Alignatt4LLM). An audio-to-translation WLK benchmark would be a separate experiment, with the ASR and MT configurations both recorded.
