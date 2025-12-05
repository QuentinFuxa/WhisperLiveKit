import hashlib
import io
import json
import os
import urllib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from tqdm import tqdm

from whisperlivekit.whisper.audio import (load_audio, log_mel_spectrogram,
                                          pad_or_trim)
from whisperlivekit.whisper.decoding import (DecodingOptions, DecodingResult,
                                             decode, detect_language)
from whisperlivekit.whisper.model import ModelDimensions, Whisper
from whisperlivekit.whisper.transcribe import transcribe
from whisperlivekit.whisper.version import __version__

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
}


def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def _infer_dims_from_config(path: str) -> Optional[ModelDimensions]:
    """
    attempt to infer ModelDimensions from a HF style config.json located
    next to the given checkpoint, usefull for distilled models/MLX models.
    """
    candidates = []
    if os.path.isdir(path):
        candidates.append(os.path.join(path, "config.json"))
    else:
        candidates.append(os.path.join(os.path.dirname(path), "config.json"))

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as f:
            config = json.load(f)

        # native Whisper format
        native_keys = ["n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head",
                       "n_audio_layer", "n_vocab", "n_text_ctx", "n_text_state",
                       "n_text_head", "n_text_layer"]
        if all(k in config for k in native_keys):
            return ModelDimensions(
                n_mels=config["n_mels"],
                n_audio_ctx=config["n_audio_ctx"],
                n_audio_state=config["n_audio_state"],
                n_audio_head=config["n_audio_head"],
                n_audio_layer=config["n_audio_layer"],
                n_vocab=config["n_vocab"],
                n_text_ctx=config["n_text_ctx"],
                n_text_state=config["n_text_state"],
                n_text_head=config["n_text_head"],
                n_text_layer=config["n_text_layer"],
            )

        # HuggingFace format
        try:
            return ModelDimensions(
                n_mels=config["num_mel_bins"],
                n_audio_ctx=config["max_source_positions"],
                n_audio_state=config["d_model"],
                n_audio_head=config["encoder_attention_heads"],
                n_audio_layer=config.get("encoder_layers")
                or config["num_hidden_layers"],
                n_vocab=config["vocab_size"],
                n_text_ctx=config["max_target_positions"],
                n_text_state=config["d_model"],
                n_text_head=config["decoder_attention_heads"],
                n_text_layer=config["decoder_layers"],
            )
        except KeyError as err:
            warnings.warn(f"Missing key {err} in HuggingFace config {candidate}")
            return None

    return None


def _convert_hf_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    converts a HF checkpoint state_dict into the naming convention used by
    default whisper
    """

    if not any(k.startswith("model.") for k in state_dict):
        return state_dict

    def map_block(prefix: str, target_prefix: str, remainder: str) -> Optional[str]:
        if remainder.startswith("self_attn."):
            suffix = remainder.split(".", 1)[1]
            mapping = {
                "q_proj": "attn.query",
                "k_proj": "attn.key",
                "v_proj": "attn.value",
                "out_proj": "attn.out",
            }
            stem = mapping.get(suffix.split(".")[0])
            if stem:
                rest = suffix.split(".", 1)[1] if "." in suffix else ""
                return f"{target_prefix}.{stem}" + (f".{rest}" if rest else "")
        elif remainder == "self_attn_layer_norm.weight":
            return f"{target_prefix}.attn_ln.weight"
        elif remainder == "self_attn_layer_norm.bias":
            return f"{target_prefix}.attn_ln.bias"
        elif remainder.startswith("encoder_attn."):
            suffix = remainder.split(".", 1)[1]
            mapping = {
                "q_proj": "cross_attn.query",
                "k_proj": "cross_attn.key",
                "v_proj": "cross_attn.value",
                "out_proj": "cross_attn.out",
            }
            stem = mapping.get(suffix.split(".", 1)[0])
            if stem:
                rest = suffix.split(".", 1)[1] if "." in suffix else ""
                return f"{target_prefix}.{stem}" + (f".{rest}" if rest else "")
        elif remainder == "encoder_attn_layer_norm.weight":
            return f"{target_prefix}.cross_attn_ln.weight"
        elif remainder == "encoder_attn_layer_norm.bias":
            return f"{target_prefix}.cross_attn_ln.bias"
        elif remainder.startswith("fc1."):
            return f"{target_prefix}.mlp.0.{remainder.split('.',1)[1]}"
        elif remainder.startswith("fc2."):
            return f"{target_prefix}.mlp.2.{remainder.split('.',1)[1]}"
        elif remainder == "final_layer_norm.weight":
            return f"{target_prefix}.mlp_ln.weight"
        elif remainder == "final_layer_norm.bias":
            return f"{target_prefix}.mlp_ln.bias"
        return None

    converted = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue
        subkey = key[len("model.") :]

        if subkey.startswith("encoder.layers."):
            parts = subkey.split(".")
            layer_idx = parts[2]
            remainder = ".".join(parts[3:])
            mapped = map_block(subkey, f"encoder.blocks.{layer_idx}", remainder)
        elif subkey.startswith("decoder.layers."):
            parts = subkey.split(".")
            layer_idx = parts[2]
            remainder = ".".join(parts[3:])
            mapped = map_block(subkey, f"decoder.blocks.{layer_idx}", remainder)
        elif subkey.startswith("encoder.conv") or subkey.startswith("decoder.conv"):
            mapped = subkey
        elif subkey == "encoder.embed_positions.weight":
            mapped = "encoder.positional_embedding"
        elif subkey == "decoder.embed_positions.weight":
            mapped = "decoder.positional_embedding"
        elif subkey == "encoder.layer_norm.weight":
            mapped = "encoder.ln_post.weight"
        elif subkey == "encoder.layer_norm.bias":
            mapped = "encoder.ln_post.bias"
        elif subkey.startswith("decoder.embed_tokens."):
            mapped = subkey.replace("embed_tokens", "token_embedding", 1)
        elif subkey == "decoder.layer_norm.weight":
            mapped = "decoder.ln.weight"
        elif subkey == "decoder.layer_norm.bias":
            mapped = "decoder.ln.bias"
        else:
            mapped = None

        if mapped:
            converted[mapped] = value

    return converted if converted else state_dict


def _convert_mlx_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Converts an mlx whisper checkpoint to a default openai whisper one
    """
    if not any("mlp1" in k or "mlp2" in k for k in state_dict):
        return state_dict

    converted = {}
    for key, value in state_dict.items():
        if key == "alignment_heads":
            continue
        
        new_key = key.replace(".mlp1.", ".mlp.0.").replace(".mlp2.", ".mlp.2.")
        converted[new_key] = value

    return converted


def _load_lora_state(lora_path: str):
    safe_path = os.path.join(lora_path, "adapter_model.safetensors")
    bin_path = os.path.join(lora_path, "adapter_model.bin")
    if os.path.isfile(safe_path):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading LoRA adapters stored as .safetensors requires the `safetensors` package."
            ) from exc
        return load_file(safe_path)
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(
        f"No adapter weights found under {lora_path}. Expected adapter_model.safetensors or adapter_model.bin."
    )


def _collapse_hf_module_name(module: str):
    if module.startswith("base_model."):
        module = module[len("base_model.") :]
    if module.startswith("model.model."):
        module = module[len("model.") :]
    if not module.startswith("model."):
        module = f"model.{module}"
    return module


def _resolve_lora_path(lora_path: Optional[str]) -> Optional[str]:
    """
    Resolve LoRA adapter path - handles both local paths and HuggingFace repo IDs.
    
    If lora_path is a local directory containing adapter files, returns it as-is.
    If lora_path looks like a HuggingFace repo ID (contains '/'), downloads and caches it.
    """
    if not lora_path:
        return None
    
    # Check if it's already a valid local path
    if os.path.isdir(lora_path):
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.isfile(config_path):
            return lora_path
    
    # Try to download from HuggingFace Hub
    if "/" in lora_path:
        try:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(
                repo_id=lora_path,
                allow_patterns=["adapter_config.json", "adapter_model.*"],
            )
            return local_path
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find LoRA adapter at local path or HuggingFace Hub: {lora_path}. Error: {e}"
            )
    
    raise FileNotFoundError(
        f"LoRA path '{lora_path}' is not a valid local directory or HuggingFace repo ID."
    )


def _apply_lora_adapter(state_dict: Dict[str, Tensor], lora_path: Optional[str]):
    if not lora_path:
        return
    
    # Resolve path (handles HuggingFace Hub download)
    lora_path = _resolve_lora_path(lora_path)
    if not lora_path:
        return

    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing adapter_config.json inside {lora_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("peft_type") != "LORA":
        raise ValueError("Only LoRA adapters are supported.")

    r = config.get("r")
    alpha = config.get("lora_alpha") or config.get("alpha")
    if not r or not alpha:
        raise ValueError("LoRA config must include `r` and `lora_alpha`.")
    scaling = alpha / r

    adapter_state = _load_lora_state(lora_path)
    lora_layers: Dict[str, Dict[str, Tensor]] = {}
    for key, tensor in adapter_state.items():
        if key.endswith("lora_A.weight"):
            module = key[: -len(".lora_A.weight")]
            lora_layers.setdefault(module, {})["A"] = tensor
        elif key.endswith("lora_B.weight"):
            module = key[: -len(".lora_B.weight")]
            lora_layers.setdefault(module, {})["B"] = tensor

    if not lora_layers:
        raise ValueError(f"No LoRA tensors found in {lora_path}")

    for module, parts in lora_layers.items():
        if "A" not in parts or "B" not in parts:
            raise ValueError(f"Incomplete LoRA tensors for module '{module}'")

        hf_module = _collapse_hf_module_name(module)
        hf_weight_key = f"{hf_module}.weight"

        delta = parts["B"] @ parts["A"]
        delta = delta * scaling

        converted = _convert_hf_state_dict({hf_weight_key: delta})
        if not converted:
            raise KeyError(f"Failed to map LoRA module '{module}' into Whisper state dict.")
        target_name, delta_tensor = next(iter(converted.items()))
        if target_name not in state_dict:
            raise KeyError(
                f"LoRA module '{module}' mapped to '{target_name}', but the base model has no such parameter."
            )

        state_dict[target_name] = state_dict[target_name] + delta_tensor.to(
            dtype=state_dict[target_name].dtype, device=state_dict[target_name].device
        )


def _load_checkpoint(
    file_path: Union[str, Path],
    device: str,
    in_memory: bool = False,
    checkpoint_bytes: Optional[bytes] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint from a single file.
    
    Handles .pt, .bin, and .safetensors formats.
    """
    if checkpoint_bytes is not None:
        with io.BytesIO(checkpoint_bytes) as fp:
            return torch.load(fp, map_location=device)
    
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    if suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "Please install safetensors to load .safetensors model files: `pip install safetensors`"
            )
        return load_file(str(file_path), device=device)
    else:
        if in_memory:
            with open(file_path, "rb") as f:
                checkpoint_bytes = f.read()
            with io.BytesIO(checkpoint_bytes) as fp:
                return torch.load(fp, map_location=device)
        else:
            with open(file_path, "rb") as fp:
                return torch.load(fp, map_location=device)


def _load_sharded_checkpoint(
    shard_files: List[Path],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Load a sharded checkpoint (multiple .safetensors or .bin files).
    
    Merges all shards into a single state dict.
    """
    merged_state_dict = {}
    first_suffix = shard_files[0].suffix.lower()
    
    if first_suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "Please install safetensors to load sharded .safetensors model: `pip install safetensors`"
            )
        for shard_path in shard_files:
            shard_dict = load_file(str(shard_path), device=device)
            merged_state_dict.update(shard_dict)
    else:
        for shard_path in shard_files:
            with open(shard_path, "rb") as fp:
                shard_dict = torch.load(fp, map_location=device)
            if isinstance(shard_dict, dict):
                merged_state_dict.update(shard_dict)
    
    return merged_state_dict


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
    decoder_only: bool = False,
    custom_alignment_heads: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
        Can be a single file (.pt, .bin, .safetensors), a directory containing model files,
        or a sharded model directory with files like model-00001-of-00002.safetensors.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory
    lora_path: str
        optional directory containing PEFT LoRA adapter weights (adapter_config + adapter_model)

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """
    from whisperlivekit.model_paths import detect_model_format

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")
    
    checkpoint = None
    model_path_for_config = name  # Used to find config.json for dims inference
    
    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        if in_memory:
            checkpoint = _load_checkpoint(None, device, checkpoint_bytes=checkpoint_file)
        else:
            checkpoint = _load_checkpoint(checkpoint_file, device)
    elif os.path.isfile(name):
        if in_memory:
            with open(name, "rb") as f:
                checkpoint_bytes = f.read()
            checkpoint = _load_checkpoint(None, device, checkpoint_bytes=checkpoint_bytes)
        else:
            checkpoint = _load_checkpoint(name, device)
        model_path_for_config = name
    elif os.path.isdir(name):
        model_info = detect_model_format(name)
        
        if not model_info.has_pytorch:
            raise RuntimeError(
                f"No PyTorch checkpoint found in directory {name}. "
                f"Expected .pt, .bin, or .safetensors file(s)."
            )
        
        if model_info.is_sharded:
            checkpoint = _load_sharded_checkpoint(model_info.pytorch_files, device)
        else:
            single_file = model_info.pytorch_files[0]
            if in_memory:
                with open(single_file, "rb") as f:
                    checkpoint_bytes = f.read()
                checkpoint = _load_checkpoint(None, device, checkpoint_bytes=checkpoint_bytes)
            else:
                checkpoint = _load_checkpoint(single_file, device)
        model_path_for_config = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )
        
    alignment_heads = _ALIGNMENT_HEADS.get(name, None)
    if custom_alignment_heads:
        alignment_heads = custom_alignment_heads.encode()

    dims_cfg = checkpoint.get("dims") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    if alignment_heads is None and "alignment_heads" in state_dict:
        alignment_heads = state_dict["alignment_heads"]
    
    state_dict = _convert_hf_state_dict(state_dict)
    state_dict = _convert_mlx_state_dict(state_dict)
    _apply_lora_adapter(state_dict, lora_path)

    if dims_cfg is not None:
        dims = ModelDimensions(**dims_cfg)
    else:
        dims = _infer_dims_from_config(model_path_for_config)
        if dims is None:
            raise RuntimeError(
                "Could not determine model dimensions. "
                "Ensure the checkpoint includes 'dims' or a HuggingFace config.json is present."
            )
        if not isinstance(state_dict, dict):
            state_dict = checkpoint

    model = Whisper(dims, decoder_only=decoder_only)
    
    if decoder_only:
        state_dict = {
            k: v for k, v in state_dict.items() 
            if 'encoder' not in k
        }

    model.load_state_dict(state_dict)

    if alignment_heads is not None:
        if isinstance(alignment_heads, bytes):
            model.set_alignment_heads(alignment_heads)
        elif isinstance(alignment_heads, torch.Tensor): #for mlx whisper
            mask = torch.zeros(dims.n_text_layer, dims.n_text_head, dtype=torch.bool)
            for layer, head in alignment_heads.tolist():
                mask[layer, head] = True
            model.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)
    return model.to(device)


def convert_encoder_to_coreml(
    model_name = "base",
    output_path= "whisper_encoder.mlpackage",
    dummy_frames = 3000, #Number of time frames to use for the dummy mel input during tracing
    precision = "float16",
):
   
    import coremltools as ct
    model = load_model(model_name, device="cpu", decoder_only=False)
    encoder = model.encoder.eval().cpu()

    dummy_input = torch.randn(
        1,
        model.dims.n_mels,
        dummy_frames,
        dtype=next(encoder.parameters()).dtype,
    )

    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder, dummy_input)

    precision_map = {
        "float16": ct.precision.FLOAT16,
        "fp16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
        "fp32": ct.precision.FLOAT32,
    }
    coreml_precision = precision_map[precision.lower()]

    mlmodel = ct.convert(
        traced_encoder,
        inputs=[ct.TensorType(name="mel", shape=dummy_input.shape)],
        convert_to= "mlprogram",
        compute_precision=coreml_precision,
    )

    output_path = Path(output_path)
    mlmodel.save(str(output_path))
    return output_path

# if __name__ == "__main__":
#     convert_encoder_to_coreml(model_name="tiny", output_path="whisper_encoder.mlpackage", dummy_frames=3000, precision="float16", convert_to="mlprogram")