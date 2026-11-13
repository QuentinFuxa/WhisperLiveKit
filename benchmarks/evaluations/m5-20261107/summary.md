# M5 backend screening

Generated from the retained schema-3.1 reports. Incomplete runs are not ranked.

All sixteen runs were observed on AC power in one serial session; ordinary desktop activity was not controlled.

| Backend | Language | Completed | WER/CER % | EOF p95 s | First visible p95 s | Source-end lag p95 s | RSS GiB | MLX GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Whisper small MLX / LocalAgreement | en | 90/90 | 11.664 | 0.465 | 4.034 | 0.469 | 1.704 | 1.146 |
| Whisper small MLX / LocalAgreement | fr | 90/90 | 17.661 | 0.736 | 6.906 | 0.738 | 1.755 | 1.146 |
| Whisper small MLX / LocalAgreement | zh | 90/90 | 25.769 | 0.406 | 6.681 | 0.408 | 1.924 | 1.144 |
| Qwen3 0.6B / Metal | en | 90/90 | 21.363 | 0.487 | 2.644 | 0.490 | 2.361 | 1.622 |
| Qwen3 0.6B / Metal | fr | 90/90 | 24.678 | 1.107 | 3.688 | 1.110 | 2.441 | 1.668 |
| Qwen3 0.6B / Metal | zh | 90/90 | 11.846 | 1.009 | 3.683 | 1.012 | 2.377 | 1.758 |
| Qwen3 0.6B / native MLX | en | 90/90 | 27.654 | 1.244 | 4.424 | 1.246 | 0.335 | 3.051 |
| Qwen3 0.6B / native MLX | fr | 90/90 | 44.678 | 1.889 | 6.022 | 1.892 | 0.503 | 3.076 |
| Qwen3 0.6B / native MLX | zh | 90/90 | 28.462 | 0.893 | 6.121 | 0.896 | 0.504 | 3.057 |
| Nemotron 0.6B / native MLX | en | 90/90 | 19.266 | 0.122 | 3.113 | 0.128 | 1.758 | 1.883 |
| Nemotron 0.6B / native MLX | fr | 90/90 | 14.971 | 0.108 | 4.139 | 0.112 | 1.757 | 1.889 |
| Nemotron 0.6B / native MLX | zh | 90/90 | 22.615 | 0.052 | 4.142 | 0.056 | 1.756 | 1.883 |

Quality pools edit counts; latency p95 pools successful clips. Memory columns describe separate measured quantities and must not be added.

## Per-pass numerical gate

The same latency or memory metric must improve by at least 20% in each pass, with no more than one percentage point worse quality in any pass. Streaming review is separate.

- qwen-native / whisper-small-la, en: **fail**.
- qwen-native / whisper-small-la, fr: **fail**.
- qwen-native / whisper-small-la, zh: **fail**.
- qwen-native / qwen-metal, en: **fail**.
- qwen-native / qwen-metal, fr: **fail**.
- qwen-native / qwen-metal, zh: **fail**.
- nemotron-native / whisper-small-la, en: **fail**.
- nemotron-native / whisper-small-la, fr: **pass**.
- nemotron-native / whisper-small-la, zh: **pass**.
- nemotron-native / qwen-metal, en: **pass**.
- nemotron-native / qwen-metal, fr: **pass**.
- nemotron-native / qwen-metal, zh: **fail**.

The original JSON retains failed hypotheses, configuration, package versions and model/audio hashes. Passing this gate alone does not establish a backend's suitability.
