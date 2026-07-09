# JarvisLab Usage For Agents

This note is for agents working from this laptop/repo and needing to run GPU work
on JarvisLab.

## Local Context

- Repo root on this laptop: `/Users/quentin/Documents/repos/WhisperLiveKit`
- Experimental Qwen causal workspace: `experiments/qwen3-causal`
- JarvisLab CLI binary observed locally: `/Users/quentin/.local/bin/jl`
- Current JarvisLab VM used for this work may have a changing ID after resume.
  Always run `jl list` and use the current `Running` or `Paused` ID.

## Basic Checks

```bash
cd /Users/quentin/Documents/repos/WhisperLiveKit
command -v jl
jl status
jl list
jl gpus
```

`jl status` shows account balance. Do not start or resume an H100 if the balance
looks insufficient for the intended run.

## Resume A Paused Instance

Use `--yes` so commands do not block on an interactive confirmation.

```bash
jl resume <instance_id> --yes
jl list
```

Important: JarvisLab may change the instance ID on resume, e.g. `420638 ->
420777`. Use the new ID from the resume output for all later commands.

Wait until SSH is ready:

```bash
jl exec <new_instance_id> bash -lc 'echo ready && hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

If SSH is refused or times out immediately after resume, wait 10-30 seconds and
retry.

## Running Remote Commands

For compound commands, pass an explicit shell. Do not pass a single quoted command
without `bash -lc`; `jl exec` can treat it as a binary name.

Good:

```bash
jl exec <id> bash -lc 'cd /home/ubuntu/qwen3-causal-export-root/qwen3-causal && pwd && nvidia-smi'
```

Bad:

```bash
jl exec <id> 'cd /home/ubuntu/project && nvidia-smi'
```

## Sync Code To The VM

Prefer uploading a small tarball containing code, scripts, tests, and configs.
Avoid uploading `runs/`, `data/`, model checkpoints, audio corpora, and
`__pycache__`.

```bash
cd /Users/quentin/Documents/repos/WhisperLiveKit
find experiments/qwen3-causal -name '__pycache__' -type d -prune -exec rm -rf {} +
rm -f /tmp/qwen3-causal-code.tgz
tar -czf /tmp/qwen3-causal-code.tgz \
  -C experiments/qwen3-causal \
  qwen3_streaming scripts tests configs pyproject.toml README.md README_WLK_IMPORT.md

jl upload <id> /tmp/qwen3-causal-code.tgz /home/ubuntu/qwen3-causal-code.tgz
jl exec <id> bash -lc 'tar -xzf /home/ubuntu/qwen3-causal-code.tgz -C /home/ubuntu/qwen3-causal-export-root/qwen3-causal'
```

macOS tar may emit `LIBARCHIVE.xattr.com.apple.provenance` warnings on the VM.
Those warnings are harmless.

## Python Environment On The Existing H100 VM

The previously used venv is:

```bash
/home/ubuntu/qwen3-asr-streaming-h100/.venv/bin/python
```

Because that venv may have an older editable package installed, force the synced
workspace to the front of imports:

```bash
PYTHONPATH=/home/ubuntu/qwen3-causal-export-root/qwen3-causal \
/home/ubuntu/qwen3-asr-streaming-h100/.venv/bin/python -m pytest -q tests
```

Without `PYTHONPATH`, Python may import
`/home/ubuntu/qwen3-asr-streaming-h100/qwen3_streaming` instead of the synced
`qwen3-causal` code.

## Quick Validation Commands

Remote compile:

```bash
jl exec <id> bash -lc 'cd /home/ubuntu/qwen3-causal-export-root/qwen3-causal && PYTHONPATH=/home/ubuntu/qwen3-causal-export-root/qwen3-causal /home/ubuntu/qwen3-asr-streaming-h100/.venv/bin/python -m py_compile qwen3_streaming/native_realtime_model.py qwen3_streaming/cached_full_hypothesis.py scripts/train_realtime_tiny_asr.py scripts/infer_cached_full_hypothesis.py scripts/eval_cached_full_hypothesis.py'
```

Remote tests:

```bash
jl exec <id> bash -lc 'cd /home/ubuntu/qwen3-causal-export-root/qwen3-causal && PYTHONPATH=/home/ubuntu/qwen3-causal-export-root/qwen3-causal /home/ubuntu/qwen3-asr-streaming-h100/.venv/bin/python -m pytest -q tests'
```

Smoke GPU availability:

```bash
jl exec <id> bash -lc 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv'
```

## Qwen3-Causal Smoke Training

Use this only as a mechanical smoke test. It is not a quality run.

```bash
jl exec <id> bash -lc 'rm -rf /tmp/qwen_causal_ar_ce_smoke && cd /home/ubuntu/qwen3-causal-export-root/qwen3-causal && PYTHONPATH=/home/ubuntu/qwen3-causal-export-root/qwen3-causal /home/ubuntu/qwen3-asr-streaming-h100/.venv/bin/python scripts/train_realtime_tiny_asr.py --output-dir /tmp/qwen_causal_ar_ce_smoke --train-manifest-jsonl data/qwen_aligned_fleurs_tiny/train_manifest.jsonl --eval-manifest-jsonl data/qwen_aligned_fleurs_tiny/eval_manifest.jsonl --alignment-loss qwen_causal_ar_ce --decoder-backend qwen_audio_causal_kv --qwen-decoder-model Qwen/Qwen3-ASR-0.6B --qwen-dtype bfloat16 --device cuda --steps 1 --batch-size 1 --lr 1e-5 --freeze-qwen-all --freeze-qwen-audio --qwen-audio-lora-rank 4 --qwen-audio-lora-alpha 8 --qwen-audio-lora-dropout 0.0 --qwen-causal-ar-kl-weight 0.1 --qwen-causal-ar-z-loss-weight 1e-5 --qwen-ar-max-target-tokens 32 --no-word-start-token --max-audio-sec 16 --num-workers 0 --log-every 1'
```

Expected mechanical signs:

- command exits `0`
- `alignment_loss` is `qwen_causal_ar_ce`
- `decoder_backend` is `qwen_audio_causal_kv`
- `qwen_audio_lora_modules` is non-empty
- `trainable_params` is small relative to total params

Clean up temporary smoke artifacts:

```bash
jl exec <id> bash -lc 'rm -rf /tmp/qwen_causal_ar_ce_smoke /home/ubuntu/qwen3-causal-code.tgz && find /home/ubuntu/qwen3-causal-export-root/qwen3-causal -name __pycache__ -type d -prune -exec rm -rf {} +'
```

## Download Artifacts

Download only lightweight outputs unless explicitly asked otherwise.

```bash
jl download <id> /remote/path/to/metrics.json /local/path/metrics.json
jl download <id> /remote/path/to/summary.json /local/path/summary.json
```

For larger result folders, compress remotely first:

```bash
jl exec <id> bash -lc 'cd /home/ubuntu/qwen3-causal-export-root/qwen3-causal && tar -czf /tmp/qwen-results.tgz runs/some_run/*.json runs/some_run/*.jsonl'
jl download <id> /tmp/qwen-results.tgz /tmp/qwen-results.tgz
```

Avoid downloading or committing large `model.pt`, audio, or dataset files unless
the user explicitly requests them.

## Pause The Instance

Always pause the VM at the end of GPU work.

```bash
jl pause <id> --yes
jl list
```

Verify the target instance status is `Paused`. Do not leave a running H100 behind.

## Common Failure Modes

- `ssh: connect ... refused`: VM is still booting. Wait and retry.
- `No module named pytest`: use the existing venv Python, not `/usr/bin/python3`.
- Importing old `qwen3_streaming`: set `PYTHONPATH` to the synced
  `qwen3-causal` workspace.
- `jl exec <id> 'cmd && cmd'` reports command not found: use
  `jl exec <id> bash -lc 'cmd && cmd'`.
- Resume output says the ID changed: switch to the new ID immediately.
