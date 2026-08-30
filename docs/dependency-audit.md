# Dependency audit — September 2026

The lock contains several mutually exclusive backend environments. An advisory
against a version in that lock is not proof that every installation uses it.
It also must not be dismissed merely because the default installation is safe.
This review distinguishes installed code, reachable operations and blocked
upgrades. No NVIDIA inference validation was available on the M5 host.

## Changes verified locally

Whisper checkpoint, shard and LoRA `.bin` loading now explicitly requests
`weights_only=True`. The minimum supported PyTorch is 2.6, which fixes
[CVE-2025-32434](https://github.com/advisories/GHSA-53q9-r3pm-6pq6), a bypass of
that mode in earlier versions. Tensor state dictionaries remain supported;
checkpoints that require arbitrary pickle callables are rejected. Tests exercise
file, in-memory, shard and adapter paths with tensors and a rejected callable.
The existing locked Torch versions already exceed this minimum and were not
upgraded. This does not claim to fix the separate Torch tensor-operation
advisories below.

Package builds already require setuptools 83 or later, which fixes the
[Unicode manifest exclusion issue](https://github.com/advisories/GHSA-h35f-9h28-mq5c).
Keep build isolation enabled when producing distributions; a runtime environment
containing an older transitive setuptools is not the package build environment.
The runtime upgrade is blocked separately: the Qwen-compatible vLLM graph
requires setuptools below 81 on Python 3.12 and later. vLLM 0.24 removes that
graph constraint but requires Transformers 5, conflicting with Qwen ASR's
4.57.6 pin. Keep this pending with the Qwen/vLLM migration.

## Backend-specific findings

| Environment | Finding | Disposition |
|---|---|---|
| Default Whisper and local MLX MT | Model paths are startup configuration; HTTP uploads are decoded as audio. The server does not load a checkpoint supplied by a session. | Explicit weights-only loading and minimum Torch above apply. Use trusted model artifacts. |
| Qwen HF streaming | `qwen-asr==0.0.6` pins Transformers 4.57.6. The lock also contains Transformers 5.15.0 for compatible extras. | The 4.x branch remains affected by advisories for Trainer, model loading and chat-template saving. WLK uses ASR inference and does not expose Trainer or `save_pretrained` to clients. A Transformers 5 migration needs the upstream Qwen pin and runtime tests, not a forced lock override. |
| Qwen CUDA/vLLM | vLLM 0.22.1 is behind fixes in 0.24 and 0.26. | WLK calls the in-process engine, not vLLM's public upload, derender, Anthropic or regex-output endpoints. Do not infer that the library is fully safe: engine/kernel issues need a compatible vLLM upgrade and NVIDIA validation. Keep this upgrade pending. |
| Qwen Metal | The stable `vllm-metal==0.2.0` wheel is preserved for Python 3.12/macOS arm64. Its separately installed CPU vLLM package is outside the WLK lock. | Record that package too when validating an installation. The default extra alone is not a complete vLLM CPU installation. Do not replace the stable wheel with a nightly to silence dependency resolution errors. |
| NeMo Canary/Sortformer | NeMo 3.0.0 pins `hydra-core>1.3,<=1.3.2` and Lightning `>2.2.1,<=2.4.0`. | Hydra's fix starts at 1.3.4. Updating it violates the upstream pin; forcing it requires a tested NeMo compatibility change. Keep the NVIDIA-dependent runtime update pending. |
| FunASR | Its Hydra requirement is `>=1.3.2`, but the shared locked graph can retain 1.3.2 alongside NeMo-compatible combinations. | A targeted lock refresh does not fix all those combinations. Do not present a partially upgraded lock as a general remedy or pass untrusted model metadata to Hydra. |
| Diart | Its supported graph requires Torch `<2.9` and older Pyannote dependencies. | That conflicts with the fixes for the `unpack_sequence` and `lstm_cell` advisories. Keep the affected graph explicit; changing this bound needs Diart runtime validation rather than a global Torch upgrade. |

Hydra's [advisory](https://github.com/advisories/GHSA-2cp2-2r3c-7p7r)
requires attacker-controlled config or model metadata reaching `instantiate()`.
Version 1.3.4 adds defense in depth; its blacklist is not a complete boundary for
untrusted configuration. WLK has no public Hydra configuration endpoint, but
operator-selected third-party model metadata remains relevant.

The [Lightning advisory](https://github.com/advisories/GHSA-qqmf-gpg7-g8gw)
describes releases through 2.6.5 and links a source fix, but its machine-readable
patched version is `2022.6.15`. That mismatch is not a usable upgrade instruction.
The current upstream release was still 2.6.5 when checked. Track the linked fix
and a corrected advisory/release; do not claim a hypothetical patched version.

The remaining Torch advisories concern specific tensor operations, including
`torch.jit.script` (fixed in 2.13). WLK does not expose those Python callables to
clients. Dynamic shapes and downstream model code still require validation on
the affected runtime before declaring an environment unaffected.

DiskCache's [pickle advisory](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v)
has no patched version in the advisory. WLK does not directly use DiskCache;
transitive model/runtime caches must not be shared with untrusted writers. Do
not silently dismiss the alert or invent a version bump.

## Validation needed before closing the remaining alerts

Retain separate environments for Qwen HF, CUDA, Metal, MLX MT, NeMo and Diart.
Verify the selected dependency graph, imports and actual model startup, then
exercise a complete audio stream, pauses, speaker boundaries and EOF. Re-run
only the affected runtime matrix. CUDA image construction and import checks on
AMD64 emulation do not establish NVIDIA inference correctness.
