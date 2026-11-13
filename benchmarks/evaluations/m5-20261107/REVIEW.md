# Review: sixteen of sixteen reports

All runs were measured on AC power ([method](METHODS.md)). Every short-clip
report has 90 completed sessions with valid line timestamp ordering, and each
backend produced identical edit counts in all three passes. The
[summary](summary.md) and [figure](m5-screening-20261107.png) are generated
from the compressed reports in `reports/`.

Under the agreed rule, neither candidate passes against both existing backends.

| Comparison | English | French | Chinese |
|---|---|---|---|
| Native Qwen vs Whisper small | fail | fail | fail |
| Native Qwen vs Qwen Metal | fail | fail | fail |
| Nemotron vs Whisper small | fail | pass | pass |
| Nemotron vs Qwen Metal | pass | pass | fail |

## Native Qwen ([#425](https://github.com/QuentinFuxa/WhisperLiveKit/pull/425))

WER is 27.65% in English and 44.68% in French, and Chinese CER is 28.46%. That is
16.0, 27.0 and 2.7 points worse than Whisper small, and 6.3, 20.0 and 16.6
points worse than Qwen Metal, which uses the same model revision. Pooled EOF
p95 is 1.24, 1.89 and 0.89 seconds, slower than both baselines in every pass.
Peak RSS is 67% to 86% lower than either baseline, but the MLX allocator peak is
about 3.05 GiB, 166% to 168% above Whisper and 74% to 88% above Metal. On unified
memory this RSS difference does not establish lower memory use.

The gap with Metal is mostly insertions and deletions (French: 441 insertions
and 330 deletions against 81 and 84). Hypotheses drop whole spans at segment
boundaries and repeat others. For example, `fleurs_fr_1810` ends after
"Arnold Schwarzenegger" and loses the rest of the sentence, while
`fleurs_fr_1735` repeats several segments. With identical weights, this points
to the adapter or the incremental finalizer rather than the model.

The ten-minute stream fails after 101.5 seconds. At a pause,
`start_silence()` calls `finish_streaming()` with a 160-sample tail, and
mlx-qwen3-asr 0.3.5 raises `Audio too short for log-mel extraction`. The
separate rerun fails at the same point with the same partial text, so the
failure is reproducible.

## Nemotron ([#426](https://github.com/QuentinFuxa/WhisperLiveKit/pull/426))

Quality is identical to September: 19.27% English WER, 14.97% French WER and
22.62% Chinese CER. Pooled EOF p95 is 0.12, 0.11 and 0.05 seconds; in every
pass EOF p95 is 71% to 87% lower than Whisper and 74% to 94% lower than Metal.
Peak RSS is 22% to 28% lower than Metal. Against Whisper, the MLX allocator peak
is about 64% higher and RSS is higher in most passes.

English fails against Whisper (7.6 points worse) and Chinese fails against Metal
(10.8 points worse than Metal's 11.85%). French passes against both, as do
Chinese against Whisper and English against Metal. Accepting the adapter for
some languages only would need a decision outside this rule.

The ten-minute stream completes with valid timestamps, 0.03 s EOF delay and no
repetition loop. Its output contains one artifact: twenty Thai `า` characters
in the middle of a sentence at character 1543.

## Baselines

Whisper small's English WER is unchanged from September (11.66%). French WER is
17.66% instead of 17.78%, and Chinese CER is 25.77% instead of 25.31%, with the
same commit and model. Power state and numpy differ between the two series; the
cause of the difference is not isolated. The ten-minute Chinese stream again
repeats `的學者和衛星` 31 times, starting at character 1453 (1473 in September).

Qwen Metal completes all runs, including the ten-minute stream without a
repetition loop. Its hypotheses join some segments without a space and repeat
some words, which inflates WER; this was not investigated further.

## Startup

Fresh-process import and engine initialization, measured after the series
([startup/](startup/)):

| Backend | Seconds to engine ready |
|---|---:|
| Whisper small MLX | 4.56 |
| Qwen3 Metal | 4.16 |
| Qwen3 native MLX | 1.76 |
| Nemotron native MLX | 6.26 |

These include each constructor's warmup, so they are not a measurement of model
loading alone, and they do not enter the selection rule.
