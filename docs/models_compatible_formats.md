# Model Path Formats

The `--model-path` parameter accepts:

## File Path
- **`.pt` / `.bin` / `.safetensor` formats** Should be openable by pytorch/safetensor.

## Directory Path (recommended)
Must contain:
- **`.pt` / `.bin` / `.safetensor` file** (required for decoder)

May optionally contain:
- **`.bin` file** - faster-whisper model for encoder (requires faster-whisper)
- **`weights.npz`** or **`weights.safetensors`** - for encoder (requires whisper-mlx)

To improve speed/reduce allucinations, you may want to use `scripts/determine_alignment_heads.py` to determine the alignment heads to use for your model, and use the `--custom-alignment-heads` to pass them to WLK. If not, alignement heads are set to be all the heads of the last half layer of decoder.