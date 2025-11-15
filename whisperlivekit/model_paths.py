from pathlib import Path
from typing import Optional, Tuple, Union


def model_path_and_type(model_path: Union[str, Path]) -> Tuple[Optional[Path], bool, bool]:
    """
    Inspect the provided path and determine which model formats are available.

    Returns:
        pytorch_path: Path to a PyTorch checkpoint (if present).
        compatible_whisper_mlx: True if MLX weights exist in this folder.
        compatible_faster_whisper: True if Faster-Whisper (ctranslate2) weights exist.
    """
    path = Path(model_path)

    compatible_whisper_mlx = False
    compatible_faster_whisper = False
    pytorch_path: Optional[Path] = None

    if path.is_file() and path.suffix.lower() in [".pt", ".safetensors", ".bin"]:
        pytorch_path = path
        return pytorch_path, compatible_whisper_mlx, compatible_faster_whisper

    if path.is_dir():
        for file in path.iterdir():
            if not file.is_file():
                continue

            filename = file.name.lower()
            suffix = file.suffix.lower()

            if filename in {"weights.npz", "weights.safetensors"}:
                compatible_whisper_mlx = True
            elif filename in {"model.bin", "encoder.bin", "decoder.bin"}:
                compatible_faster_whisper = True
            elif suffix in {".pt", ".safetensors"}:
                pytorch_path = file
            elif filename == "pytorch_model.bin":
                pytorch_path = file

        if pytorch_path is None:
            fallback = path / "pytorch_model.bin"
            if fallback.exists():
                pytorch_path = fallback

    return pytorch_path, compatible_whisper_mlx, compatible_faster_whisper


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
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise FileNotFoundError(
            f"Model path '{model_path}' does not exist locally and huggingface_hub "
            "is not installed to download it."
        ) from exc

    downloaded_path = Path(snapshot_download(repo_id=str(model_path)))
    return downloaded_path
