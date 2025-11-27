import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


@dataclass
class ModelInfo:
    """Information about detected model format and files in a directory."""    
    path: Optional[Path] = None    
    pytorch_files: List[Path] = field(default_factory=list)
    compatible_whisper_mlx: bool = False
    compatible_faster_whisper: bool = False
    
    @property
    def has_pytorch(self) -> bool:
        return len(self.pytorch_files) > 0
    
    @property
    def is_sharded(self) -> bool:
        return len(self.pytorch_files) > 1
    
    @property
    def primary_pytorch_file(self) -> Optional[Path]:
        """Return the primary PyTorch file (or first shard for sharded models)."""
        if not self.pytorch_files:
            return None
        return self.pytorch_files[0]


#regex pattern for sharded model files such as: model-00001-of-00002.safetensors or pytorch_model-00001-of-00002.bin
SHARDED_PATTERN = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.(safetensors|bin)$")

FASTER_WHISPER_MARKERS = {"model.bin", "encoder.bin", "decoder.bin"}
MLX_WHISPER_MARKERS = {"weights.npz", "weights.safetensors"}
CT2_INDICATOR_FILES = {"vocabulary.json", "vocabulary.txt", "shared_vocabulary.json"}


def _is_ct2_model_bin(directory: Path, filename: str) -> bool:
    """
    Determine if model.bin/encoder.bin/decoder.bin is a CTranslate2 model.
    
    CTranslate2 models have specific companion files that distinguish them
    from PyTorch .bin files.
    """
    n_indicators = 0
    for indicator in CT2_INDICATOR_FILES: #test 1
        if (directory / indicator).exists(): 
            n_indicators += 1
        
    if n_indicators == 0:
        return False

    config_path = directory / "config.json" #test 2
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("model_type") == "whisper": #test 2
                return False
        except (json.JSONDecodeError, IOError):
            pass
    
    return True


def _collect_pytorch_files(directory: Path) -> List[Path]:
    """
    Collect all PyTorch checkpoint files from a directory.
    
    Handles:
    - Single files: model.safetensors, pytorch_model.bin, *.pt
    - Sharded files: model-00001-of-00002.safetensors, pytorch_model-00001-of-00002.bin
    - Index-based sharded models (reads index file to find shards)
    
    Returns files sorted appropriately (shards in order, or single file).
    """
    for index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        index_path = directory / index_name
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                if weight_map:
                    shard_names = sorted(set(weight_map.values()))
                    shards = [directory / name for name in shard_names if (directory / name).exists()]
                    if shards:
                        return shards
            except (json.JSONDecodeError, IOError):
                pass
    
    sharded_groups = {}
    single_files = {}
    
    for file in directory.iterdir():
        if not file.is_file():
            continue
        
        filename = file.name
        suffix = file.suffix.lower()
        
        if filename.startswith("adapter_"):
            continue
        
        match = SHARDED_PATTERN.match(filename)
        if match:
            base_name, shard_idx, total_shards, ext = match.groups()
            key = (base_name, ext, int(total_shards))
            if key not in sharded_groups:
                sharded_groups[key] = []
            sharded_groups[key].append((int(shard_idx), file))
            continue
        
        if filename == "model.safetensors":
            single_files[0] = file  # Highest priority
        elif filename == "pytorch_model.bin":
            single_files[1] = file
        elif suffix == ".pt":
            single_files[2] = file
        elif suffix == ".safetensors" and not filename.startswith("adapter"):
            single_files[3] = file
    
    for (base_name, ext, total_shards), shards in sharded_groups.items():
        if len(shards) == total_shards:
            return [path for _, path in sorted(shards)]
    
    for priority in sorted(single_files.keys()):
        return [single_files[priority]]
    
    return []


def detect_model_format(model_path: Union[str, Path]) -> ModelInfo:
    """
    Detect the model format in a given path.
    
    This function analyzes a file or directory to determine:
    - What PyTorch checkpoint files are available (including sharded models)
    - Whether the directory contains MLX Whisper weights
    - Whether the directory contains Faster-Whisper (CTranslate2) weights
    
    Args:
        model_path: Path to a model file or directory
        
    Returns:
        ModelInfo with detected format information
    """
    path = Path(model_path)
    info = ModelInfo(path=path)
    
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".pt", ".safetensors", ".bin"}:
            info.pytorch_files = [path]
        return info
    
    if not path.is_dir():
        return info
    
    for file in path.iterdir():
        if not file.is_file():
            continue
        
        filename = file.name.lower()
        
        if filename in MLX_WHISPER_MARKERS:
            info.compatible_whisper_mlx = True
        
        if filename in FASTER_WHISPER_MARKERS:
            if _is_ct2_model_bin(path, filename):
                info.compatible_faster_whisper = True
    
    info.pytorch_files = _collect_pytorch_files(path)
    
    return info


def model_path_and_type(model_path: Union[str, Path]) -> Tuple[Optional[Path], bool, bool]:
    """
    Inspect the provided path and determine which model formats are available.
    
    This is a compatibility wrapper around detect_model_format().
    
    Returns:
        pytorch_path: Path to a PyTorch checkpoint (first shard for sharded models, or None).
        compatible_whisper_mlx: True if MLX weights exist in this folder.
        compatible_faster_whisper: True if Faster-Whisper (CTranslate2) weights exist.
    """
    info = detect_model_format(model_path)
    return info.primary_pytorch_file, info.compatible_whisper_mlx, info.compatible_faster_whisper


def resolve_model_path(model_path: Union[str, Path]) -> Path:
    """
    Return a local path for the provided model reference.

    If the path does not exist locally, it is treated as a Hugging Face repo id
    and downloaded via snapshot_download.
    """
    path = Path(model_path).expanduser()
    if path.exists():
        return path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist locally and huggingface_hub "
            "is not installed to download it."
        ) from exc

    downloaded_path = Path(snapshot_download(repo_id=str(model_path)))
    return downloaded_path
