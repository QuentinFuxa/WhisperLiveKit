# Archived benchmarks

These are historical records, not current backend recommendations. The original data and images are retained unchanged.

- [h100_20260315](h100_20260315/): LibriSpeech and ACL 6060 runs, including Qwen batch and SimulStream+KV paths that have since been removed.
- [m5_20260315](m5_20260315/): Apple M5 companion run. Its per-utterance WER is not directly comparable to the H100 chapter-grouped WER.
- [h100_20260711](h100_20260711/): English/French H100 scatter results added in commit `1389d45` on 2026-07-11. The aggregate values match the plots, but per-sample records and enough provenance to recreate the comparison are missing. Includes the original runner, which expects an external `long_samples.json` manifest.

See [the benchmark review](../README.md) for the reasons the July figures were removed from the main README and the requirements for a new comparison.
