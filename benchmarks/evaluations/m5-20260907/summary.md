# M5 backend screening

Generated from the retained schema-3.1 reports. Incomplete runs are not ranked.

Power conditions changed during this series; see METHODS.md before interpreting latency gates.

| Backend | Language | Completed | WER/CER % | EOF p95 s | First visible p95 s | Source-end lag p95 s | RSS GiB | MLX GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Whisper small MLX / LocalAgreement | en | 90/90 | 11.664 | 0.702 | 3.096 | 0.706 | 1.125 | 1.146 |
| Whisper small MLX / LocalAgreement | fr | 90/90 | 17.778 | 0.700 | 6.265 | 0.706 | 1.814 | 1.146 |
| Whisper small MLX / LocalAgreement | zh | 90/90 | 25.308 | 1.001 | 8.784 | 1.004 | 1.337 | 1.146 |
| Nemotron 0.6B / native MLX | en | 90/90 | 19.266 | 0.109 | 3.121 | 0.116 | 1.729 | 1.883 |
| Nemotron 0.6B / native MLX | fr | 90/90 | 14.971 | 0.113 | 4.141 | 0.117 | 1.732 | 1.889 |
| Nemotron 0.6B / native MLX | zh | 90/90 | 22.615 | 0.052 | 4.144 | 0.055 | 1.727 | 1.883 |

Quality pools edit counts; latency p95 pools successful clips. Memory columns describe separate measured quantities and must not be added.

## Per-pass numerical gate

The same latency or memory metric must improve by at least 20% in each pass, with no more than one percentage point worse quality in any pass. Streaming review is separate.

- nemotron-native / whisper-small-la, en: **fail**.
- nemotron-native / whisper-small-la, fr: **pass**.
- nemotron-native / whisper-small-la, zh: **pass**.

Pending: qwen-metal-en, qwen-metal-fr, qwen-metal-zh, qwen-metal-zh-continuous, qwen-native-en, qwen-native-fr, qwen-native-zh, qwen-native-zh-continuous, nemotron-native-zh-continuous.

The original JSON retains failed hypotheses, configuration, package versions and model/audio hashes. Passing this gate alone does not establish a backend's suitability.
