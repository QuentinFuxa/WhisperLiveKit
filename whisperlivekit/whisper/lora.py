"""
Dynamic LoRA adapter support for Whisper models.

This module enables loading a single base Whisper model and dynamically swapping
between multiple LoRA adapters at runtime, saving GPU memory when working with
multiple language-specific fine-tuned models.

Usage:
    from whisperlivekit.whisper import load_model
    from whisperlivekit.whisper.lora import LoRAAdapterManager
    
    # Load base model without any LoRA baked in
    model = load_model("large-v3", device="cuda")
    
    # Create adapter manager
    manager = LoRAAdapterManager(model)
    
    # Load multiple adapters (small memory footprint each)
    manager.load_adapter("french", "path/to/french-lora")
    manager.load_adapter("spanish", "path/to/spanish-lora")
    
    # Switch between adapters at runtime
    manager.set_adapter("french")
    result_fr = model.transcribe(audio_fr)
    
    manager.set_adapter("spanish")
    result_es = model.transcribe(audio_es)
    
    # Disable LoRA (use base model only)
    manager.set_adapter(None)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .model import Linear


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""
    r: int  # LoRA rank
    alpha: float  # LoRA alpha (scaling factor)
    target_modules: List[str] = field(default_factory=list)
    
    @property
    def scaling(self) -> float:
        return self.alpha / self.r


@dataclass 
class LoRAAdapter:
    """Holds the LoRA A/B weight matrices for a single adapter."""
    name: str
    config: LoRAConfig
    # Maps target module name -> (A matrix, B matrix)
    weights: Dict[str, Tuple[Tensor, Tensor]] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = field(default=torch.float32)
    
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        """Move adapter weights to specified device/dtype."""
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weights = {
            name: (a.to(device=device, dtype=dtype or self.dtype), 
                   b.to(device=device, dtype=dtype or self.dtype))
            for name, (a, b) in self.weights.items()
        }
        return self
    
    def memory_footprint_mb(self) -> float:
        """Return approximate memory usage in MB."""
        total_bytes = 0
        for a, b in self.weights.values():
            total_bytes += a.numel() * a.element_size()
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 * 1024)


class LoRALinear(nn.Module):
    """
    A Linear layer wrapper that supports dynamic LoRA injection.
    
    The base weights remain unchanged. LoRA is applied additively during forward:
        output = base_linear(x) + (x @ A @ B) * scaling
    """
    
    def __init__(self, base_linear: Linear):
        super().__init__()
        self.base_linear = base_linear
        self.lora_A: Optional[Tensor] = None
        self.lora_B: Optional[Tensor] = None
        self.scaling: float = 1.0
        self._lora_enabled: bool = False
    
    def set_lora(self, A: Optional[Tensor], B: Optional[Tensor], scaling: float = 1.0):
        """Set the LoRA matrices for this layer."""
        self.lora_A = A
        self.lora_B = B
        self.scaling = scaling
        self._lora_enabled = A is not None and B is not None
    
    def clear_lora(self):
        """Remove LoRA from this layer."""
        self.lora_A = None
        self.lora_B = None
        self._lora_enabled = False
    
    def forward(self, x: Tensor) -> Tensor:
        # Base linear output
        out = self.base_linear(x)
        
        # Add LoRA contribution if enabled
        if self._lora_enabled and self.lora_A is not None and self.lora_B is not None:
            # x: (..., in_features)
            # A: (in_features, r)
            # B: (r, out_features)
            # lora_out: (..., out_features)
            lora_out = (x @ self.lora_A.to(x.dtype)) @ self.lora_B.to(x.dtype)
            out = out + lora_out * self.scaling
        
        return out
    
    # Delegate attribute access to base_linear for compatibility
    @property
    def weight(self):
        return self.base_linear.weight
    
    @property
    def bias(self):
        return self.base_linear.bias
    
    @property
    def in_features(self):
        return self.base_linear.in_features
    
    @property
    def out_features(self):
        return self.base_linear.out_features


# Mapping from HuggingFace LoRA module names to Whisper module paths
_HF_TO_WHISPER_MODULE_MAP = {
    # Encoder attention
    "model.encoder.layers.{}.self_attn.q_proj": "encoder.blocks.{}.attn.query",
    "model.encoder.layers.{}.self_attn.k_proj": "encoder.blocks.{}.attn.key",
    "model.encoder.layers.{}.self_attn.v_proj": "encoder.blocks.{}.attn.value",
    "model.encoder.layers.{}.self_attn.out_proj": "encoder.blocks.{}.attn.out",
    # Encoder MLP
    "model.encoder.layers.{}.fc1": "encoder.blocks.{}.mlp.0",
    "model.encoder.layers.{}.fc2": "encoder.blocks.{}.mlp.2",
    
    # Decoder self-attention
    "model.decoder.layers.{}.self_attn.q_proj": "decoder.blocks.{}.attn.query",
    "model.decoder.layers.{}.self_attn.k_proj": "decoder.blocks.{}.attn.key",
    "model.decoder.layers.{}.self_attn.v_proj": "decoder.blocks.{}.attn.value",
    "model.decoder.layers.{}.self_attn.out_proj": "decoder.blocks.{}.attn.out",
    # Decoder cross-attention
    "model.decoder.layers.{}.encoder_attn.q_proj": "decoder.blocks.{}.cross_attn.query",
    "model.decoder.layers.{}.encoder_attn.k_proj": "decoder.blocks.{}.cross_attn.key",
    "model.decoder.layers.{}.encoder_attn.v_proj": "decoder.blocks.{}.cross_attn.value",
    "model.decoder.layers.{}.encoder_attn.out_proj": "decoder.blocks.{}.cross_attn.out",
    # Decoder MLP
    "model.decoder.layers.{}.fc1": "decoder.blocks.{}.mlp.0",
    "model.decoder.layers.{}.fc2": "decoder.blocks.{}.mlp.2",
}


def _normalize_hf_module_name(name: str) -> str:
    """Normalize HF-style LoRA module names."""
    if name.startswith("base_model."):
        name = name[len("base_model."):]
    if name.startswith("model.model."):
        name = name[len("model."):]
    if not name.startswith("model."):
        name = f"model.{name}"
    return name


def _map_hf_to_whisper_module(hf_name: str) -> Optional[str]:
    """Map a HuggingFace LoRA module name to Whisper module path."""
    hf_name = _normalize_hf_module_name(hf_name)
    
    # Try to match with layer index patterns
    import re
    
    # Match patterns like model.encoder.layers.5.self_attn.q_proj
    for pattern, target_pattern in _HF_TO_WHISPER_MODULE_MAP.items():
        # Create regex from pattern (replace {} with capture group)
        regex = pattern.replace(".", r"\.").replace("{}", r"(\d+)")
        match = re.fullmatch(regex, hf_name)
        if match:
            layer_idx = match.group(1)
            return target_pattern.format(layer_idx)
    
    return None


def _get_module_by_path(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Get a submodule by dot-separated path."""
    parts = path.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "__getitem__"):
            try:
                current = current[int(part)]
            except (ValueError, IndexError, KeyError):
                return None
        else:
            return None
    return current


def _set_module_by_path(model: nn.Module, path: str, module: nn.Module):
    """Set a submodule by dot-separated path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if hasattr(parent, part):
            parent = getattr(parent, part)
        elif hasattr(parent, "__getitem__"):
            parent = parent[int(part)]
    setattr(parent, parts[-1], module)


class LoRAAdapterManager:
    """
    Manages multiple LoRA adapters for a Whisper model.
    
    Enables loading multiple adapters and switching between them at runtime
    without reloading the full model.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the adapter manager.
        
        Args:
            model: A Whisper model instance
        """
        self.model = model
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.current_adapter: Optional[str] = None
        self._lora_layers: Dict[str, LoRALinear] = {}
        self._original_layers: Dict[str, Linear] = {}
        self._initialized = False
    
    def _initialize_lora_layers(self, target_modules: List[str]):
        """
        Replace target Linear layers with LoRALinear wrappers.
        
        This is done lazily on first adapter load.
        """
        if self._initialized:
            return
        
        # Find and wrap all potential LoRA target modules
        for whisper_path in target_modules:
            module = _get_module_by_path(self.model, whisper_path)
            if module is None:
                continue
            if isinstance(module, Linear) and not isinstance(module, LoRALinear):
                # Wrap the Linear layer
                lora_linear = LoRALinear(module)
                _set_module_by_path(self.model, whisper_path, lora_linear)
                self._lora_layers[whisper_path] = lora_linear
                self._original_layers[whisper_path] = module
        
        self._initialized = True
    
    def _resolve_lora_path(self, lora_path: str) -> str:
        """Resolve LoRA path, downloading from HuggingFace Hub if needed."""
        if os.path.isdir(lora_path):
            return lora_path
        
        # Try HuggingFace Hub
        if "/" in lora_path:
            try:
                from huggingface_hub import snapshot_download
                return snapshot_download(
                    repo_id=lora_path,
                    allow_patterns=["adapter_config.json", "adapter_model.*"],
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find LoRA adapter at local path or HuggingFace Hub: {lora_path}. Error: {e}"
                )
        
        raise FileNotFoundError(f"LoRA path '{lora_path}' not found.")
    
    def _load_adapter_weights(self, lora_path: str) -> Dict[str, Tensor]:
        """Load adapter weights from safetensors or bin file."""
        safe_path = os.path.join(lora_path, "adapter_model.safetensors")
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        
        if os.path.isfile(safe_path):
            from safetensors.torch import load_file
            return load_file(safe_path)
        elif os.path.isfile(bin_path):
            return torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No adapter weights found in {lora_path}. "
                "Expected adapter_model.safetensors or adapter_model.bin."
            )
    
    def load_adapter(
        self, 
        name: str, 
        lora_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LoRAAdapter:
        """
        Load a LoRA adapter from disk or HuggingFace Hub.
        
        Args:
            name: Unique name for this adapter (e.g., "french", "spanish")
            lora_path: Local path or HuggingFace repo ID
            device: Device to load weights to (default: model's device)
            dtype: Data type for weights (default: model's dtype)
            
        Returns:
            The loaded LoRAAdapter
        """
        if device is None:
            device = next(self.model.parameters()).device
        if dtype is None:
            dtype = next(self.model.parameters()).dtype
        
        # Resolve path
        lora_path = self._resolve_lora_path(lora_path)
        
        # Load config
        config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing adapter_config.json in {lora_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        if config_dict.get("peft_type") != "LORA":
            raise ValueError("Only LoRA adapters are supported.")
        
        config = LoRAConfig(
            r=config_dict["r"],
            alpha=config_dict.get("lora_alpha") or config_dict.get("alpha"),
            target_modules=config_dict.get("target_modules", []),
        )
        
        # Load weights
        adapter_state = self._load_adapter_weights(lora_path)
        
        # Parse LoRA A/B matrices and map to Whisper module paths
        lora_layers: Dict[str, Dict[str, Tensor]] = {}
        for key, tensor in adapter_state.items():
            if key.endswith("lora_A.weight"):
                module = key[:-len(".lora_A.weight")]
                lora_layers.setdefault(module, {})["A"] = tensor
            elif key.endswith("lora_B.weight"):
                module = key[:-len(".lora_B.weight")]
                lora_layers.setdefault(module, {})["B"] = tensor
        
        # Map to Whisper module paths and collect weights
        weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        whisper_paths = set()
        
        for hf_module, parts in lora_layers.items():
            if "A" not in parts or "B" not in parts:
                continue
            
            whisper_path = _map_hf_to_whisper_module(hf_module)
            if whisper_path is None:
                # Try direct mapping (module might already be in Whisper format)
                whisper_path = hf_module
            
            # A: (r, in_features) -> transpose to (in_features, r)
            # B: (out_features, r) -> transpose to (r, out_features)
            A = parts["A"].T  # (in_features, r)
            B = parts["B"].T  # (r, out_features)
            
            weights[whisper_path] = (A, B)
            whisper_paths.add(whisper_path)
        
        # Create adapter
        adapter = LoRAAdapter(
            name=name,
            config=config,
            weights=weights,
            device=device,
            dtype=dtype,
        )
        adapter.to(device, dtype)
        
        # Initialize LoRA layers if not done yet
        self._initialize_lora_layers(list(whisper_paths))
        
        # Store adapter
        self.adapters[name] = adapter
        
        return adapter
    
    def set_adapter(self, name: Optional[str]):
        """
        Switch to a different adapter or disable LoRA.
        
        Args:
            name: Adapter name to activate, or None to disable all LoRA
        """
        if name is not None and name not in self.adapters:
            raise KeyError(f"Adapter '{name}' not loaded. Available: {list(self.adapters.keys())}")
        
        # Clear all LoRA from layers
        for lora_linear in self._lora_layers.values():
            lora_linear.clear_lora()
        
        self.current_adapter = name
        
        if name is None:
            return
        
        # Apply the selected adapter
        adapter = self.adapters[name]
        for module_path, (A, B) in adapter.weights.items():
            if module_path in self._lora_layers:
                self._lora_layers[module_path].set_lora(A, B, adapter.config.scaling)
    
    def unload_adapter(self, name: str):
        """
        Unload an adapter from memory.
        
        Args:
            name: Name of adapter to unload
        """
        if name not in self.adapters:
            return
        
        if self.current_adapter == name:
            self.set_adapter(None)
        
        del self.adapters[name]
    
    def list_adapters(self) -> List[str]:
        """Return list of loaded adapter names."""
        return list(self.adapters.keys())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Return memory usage in MB for each loaded adapter."""
        return {name: adapter.memory_footprint_mb() for name, adapter in self.adapters.items()}
    
    def restore_original_layers(self):
        """
        Restore the original Linear layers, removing LoRA wrappers.
        
        Call this if you want to go back to the original model structure.
        """
        for path, original in self._original_layers.items():
            _set_module_by_path(self.model, path, original)
        
        self._lora_layers.clear()
        self._original_layers.clear()
        self._initialized = False
        self.current_adapter = None

