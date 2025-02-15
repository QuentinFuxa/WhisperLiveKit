import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from mlx_whisper import whisper

from whisperlivekit.model_mapping import MLX_MODEL_MAPPING

mlx_model_mapping = MLX_MODEL_MAPPING

def load_mlx_encoder(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = tree_unflatten(list(weights.items()))
    
    # we only want to load the encoder weights here.
    # Size examples: for tiny.en, 
    # Decoder weights: 59110771 bytes
    # Encoder weights: 15268874 bytes

    
    encoder_weights = {}
    encoder_weights['encoder'] = weights['encoder']
    del(weights)
    


    model.update(encoder_weights)
    mx.eval(model.parameters())
    return model


def load_mlx_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = tree_unflatten(list(weights.items()))
    
    model.update(weights)
    mx.eval(model.parameters())
    return model