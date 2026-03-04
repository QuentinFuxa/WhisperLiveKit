"""
Model weight loading for the MLX Voxtral Realtime backend.

Supports two on-disk formats:
  1. **Converted** (``config.json`` + ``model.safetensors``): ready-to-load,
     with optional quantisation metadata.
  2. **Original Mistral** (``params.json`` + ``consolidated.safetensors``):
     requires weight renaming and conv-weight transposition.

The public entry point is :func:`load_voxtral_model` which returns the
model, tokenizer, and raw config dict.
"""

import json
import logging
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .model import VoxtralMLXModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"

# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

_ALLOWED_PATTERNS = [
    "consolidated.safetensors",
    "model*.safetensors",
    "model.safetensors.index.json",
    "params.json",
    "config.json",
    "tekken.json",
]


def download_weights(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """Download model files from HuggingFace Hub and return the local path."""
    return Path(snapshot_download(model_id, allow_patterns=_ALLOWED_PATTERNS))


# ---------------------------------------------------------------------------
# Weight name remapping (Mistral → our naming)
# ---------------------------------------------------------------------------

_NAME_RULES: list[tuple[str, str]] = [
    # Encoder convolutions
    (r"whisper_encoder\.conv_layers\.0\.conv\.(.*)", r"encoder.conv1.\1"),
    (r"whisper_encoder\.conv_layers\.1\.conv\.(.*)", r"encoder.conv2.\1"),
    # Encoder transformer blocks
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wq\.(.*)",
     r"encoder.blocks.\1.self_attn.q_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wk\.(.*)",
     r"encoder.blocks.\1.self_attn.k_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wv\.(.*)",
     r"encoder.blocks.\1.self_attn.v_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(.*)",
     r"encoder.blocks.\1.self_attn.out_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(.*)",
     r"encoder.blocks.\1.pre_attn_norm.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(.*)",
     r"encoder.blocks.\1.ffn.gate.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(.*)",
     r"encoder.blocks.\1.ffn.down.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(.*)",
     r"encoder.blocks.\1.ffn.up.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(.*)",
     r"encoder.blocks.\1.pre_ffn_norm.\2"),
    (r"whisper_encoder\.transformer\.norm\.(.*)", r"encoder.final_norm.\1"),
    # Adapter
    (r"audio_language_projection\.0\.weight", r"adapter.linear1.weight"),
    (r"audio_language_projection\.2\.weight", r"adapter.linear2.weight"),
    # Decoder embedding
    (r"tok_embeddings\.weight", r"decoder.token_embedding.weight"),
    # Decoder blocks
    (r"layers\.(\d+)\.attention\.wq\.weight",
     r"decoder.blocks.\1.self_attn.q_proj.weight"),
    (r"layers\.(\d+)\.attention\.wk\.weight",
     r"decoder.blocks.\1.self_attn.k_proj.weight"),
    (r"layers\.(\d+)\.attention\.wv\.weight",
     r"decoder.blocks.\1.self_attn.v_proj.weight"),
    (r"layers\.(\d+)\.attention\.wo\.weight",
     r"decoder.blocks.\1.self_attn.out_proj.weight"),
    (r"layers\.(\d+)\.attention_norm\.weight",
     r"decoder.blocks.\1.pre_attn_norm.weight"),
    (r"layers\.(\d+)\.feed_forward\.w1\.weight",
     r"decoder.blocks.\1.ffn.gate.weight"),
    (r"layers\.(\d+)\.feed_forward\.w2\.weight",
     r"decoder.blocks.\1.ffn.down.weight"),
    (r"layers\.(\d+)\.feed_forward\.w3\.weight",
     r"decoder.blocks.\1.ffn.up.weight"),
    (r"layers\.(\d+)\.ffn_norm\.weight",
     r"decoder.blocks.\1.pre_ffn_norm.weight"),
    (r"layers\.(\d+)\.ada_rms_norm_t_cond\.0\.weight",
     r"decoder.blocks.\1.adaptive_scale.proj_in.weight"),
    (r"layers\.(\d+)\.ada_rms_norm_t_cond\.2\.weight",
     r"decoder.blocks.\1.adaptive_scale.proj_out.weight"),
    # Decoder final norm
    (r"norm\.weight", r"decoder.final_norm.weight"),
]

_PREFIX_STRIP = re.compile(
    r"^(mm_streams_embeddings\.embedding_module|mm_whisper_embeddings)\."
)


def _translate_weight_name(name: str) -> str | None:
    name = _PREFIX_STRIP.sub("", name)
    for pattern, replacement in _NAME_RULES:
        result, n = re.subn(f"^{pattern}$", replacement, name)
        if n:
            return result
    return None


def _is_conv_weight(name: str) -> bool:
    return ("conv1.weight" in name or "conv2.weight" in name) and "bias" not in name


# ---------------------------------------------------------------------------
# Converted-format weight remapping (voxmlx names → our names)
# ---------------------------------------------------------------------------

_CONVERTED_RULES: list[tuple[str, str]] = [
    # Adapter
    (r"adapter\.w_in\.(.*)", r"adapter.linear1.\1"),
    (r"adapter\.w_out\.(.*)", r"adapter.linear2.\1"),
    # Encoder transformer blocks
    (r"encoder\.layers\.(\d+)\.attention\.(.*)", r"encoder.blocks.\1.self_attn.\2"),
    (r"encoder\.layers\.(\d+)\.attn_norm\.(.*)", r"encoder.blocks.\1.pre_attn_norm.\2"),
    (r"encoder\.layers\.(\d+)\.mlp\.gate_proj\.(.*)", r"encoder.blocks.\1.ffn.gate.\2"),
    (r"encoder\.layers\.(\d+)\.mlp\.down_proj\.(.*)", r"encoder.blocks.\1.ffn.down.\2"),
    (r"encoder\.layers\.(\d+)\.mlp\.up_proj\.(.*)", r"encoder.blocks.\1.ffn.up.\2"),
    (r"encoder\.layers\.(\d+)\.ffn_norm\.(.*)", r"encoder.blocks.\1.pre_ffn_norm.\2"),
    (r"encoder\.norm\.(.*)", r"encoder.final_norm.\1"),
    # Decoder embedding
    (r"language_model\.embed_tokens\.(.*)", r"decoder.token_embedding.\1"),
    # Decoder blocks
    (r"language_model\.layers\.(\d+)\.attention\.(.*)", r"decoder.blocks.\1.self_attn.\2"),
    (r"language_model\.layers\.(\d+)\.attn_norm\.(.*)", r"decoder.blocks.\1.pre_attn_norm.\2"),
    (r"language_model\.layers\.(\d+)\.mlp\.gate_proj\.(.*)", r"decoder.blocks.\1.ffn.gate.\2"),
    (r"language_model\.layers\.(\d+)\.mlp\.down_proj\.(.*)", r"decoder.blocks.\1.ffn.down.\2"),
    (r"language_model\.layers\.(\d+)\.mlp\.up_proj\.(.*)", r"decoder.blocks.\1.ffn.up.\2"),
    (r"language_model\.layers\.(\d+)\.ffn_norm\.(.*)", r"decoder.blocks.\1.pre_ffn_norm.\2"),
    (r"language_model\.layers\.(\d+)\.ada_norm\.linear_in\.(.*)",
     r"decoder.blocks.\1.adaptive_scale.proj_in.\2"),
    (r"language_model\.layers\.(\d+)\.ada_norm\.linear_out\.(.*)",
     r"decoder.blocks.\1.adaptive_scale.proj_out.\2"),
    (r"language_model\.norm\.(.*)", r"decoder.final_norm.\1"),
]

# Also remap o_proj → out_proj in both encoder and decoder
_POST_RENAME = [
    (r"\.o_proj\.", r".out_proj."),
]


def _remap_converted_name(name: str) -> str:
    """Translate a converted-format weight name to our naming convention."""
    for pattern, replacement in _CONVERTED_RULES:
        result, n = re.subn(f"^{pattern}$", replacement, name)
        if n:
            name = result
            break
    for pattern, replacement in _POST_RENAME:
        name = re.sub(pattern, replacement, name)
    return name


# ---------------------------------------------------------------------------
# Loading strategies
# ---------------------------------------------------------------------------

def _has_converted_layout(path: Path) -> bool:
    return (path / "config.json").exists() and not (path / "consolidated.safetensors").exists()


def _load_converted_weights(path: Path):
    with open(path / "config.json") as f:
        config = json.load(f)

    model = VoxtralMLXModel(config)

    quant = config.get("quantization")
    if quant is not None:
        gs = quant["group_size"]
        nn.quantize(
            model,
            group_size=gs,
            bits=quant["bits"],
            class_predicate=lambda _p, m: (
                hasattr(m, "to_quantized") and m.weight.shape[-1] % gs == 0
            ),
        )

    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            shard_map = json.load(f)
        shard_files = sorted(set(shard_map["weight_map"].values()))
        weights = {}
        for sf in shard_files:
            weights.update(mx.load(str(path / sf)))
    else:
        weights = mx.load(str(path / "model.safetensors"))

    remapped = {_remap_converted_name(k): v for k, v in weights.items()}
    model.load_weights(list(remapped.items()))
    mx.eval(model.parameters())
    return model, config


def _load_original_weights(path: Path):
    with open(path / "params.json") as f:
        config = json.load(f)

    model = VoxtralMLXModel(config)

    raw = mx.load(str(path / "consolidated.safetensors"))
    mapped: dict[str, mx.array] = {}
    skipped: list[str] = []

    for name, tensor in raw.items():
        if name == "output.weight":
            continue
        new_name = _translate_weight_name(name)
        if new_name is None:
            skipped.append(name)
            continue
        # Conv weights: PyTorch [C_out, C_in, K] → MLX [C_out, K, C_in]
        if _is_conv_weight(new_name):
            tensor = mx.swapaxes(tensor, 1, 2)
        mapped[new_name] = tensor

    if skipped:
        logger.warning("Skipped %d unrecognised weight keys (first 5: %s)", len(skipped), skipped[:5])

    model.load_weights(list(mapped.items()))
    mx.eval(model.parameters())
    return model, config


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _load_tokenizer(model_dir: Path):
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    return Tekkenizer.from_file(str(model_dir / "tekken.json"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_voxtral_model(path_or_id: str = DEFAULT_MODEL_ID):
    """Load a Voxtral Realtime model and its tokenizer.

    Args:
        path_or_id: Local directory path **or** a HuggingFace model ID.

    Returns:
        ``(model, tokenizer, config)``
    """
    p = Path(path_or_id)
    if not p.exists():
        p = download_weights(path_or_id)

    if _has_converted_layout(p):
        model, config = _load_converted_weights(p)
    else:
        model, config = _load_original_weights(p)

    tokenizer = _load_tokenizer(p)
    logger.info("Voxtral MLX model loaded from %s", p)
    return model, tokenizer, config
