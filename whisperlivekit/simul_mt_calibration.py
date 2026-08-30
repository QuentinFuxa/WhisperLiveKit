"""Read AlignAtt4LLM head files with provenance for their MLX representation."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    model: str
    revision: str
    direction: str
    heads: tuple[tuple[int, int], ...]
    num_layers: int
    num_heads: int
    quantization: dict
    source: str
    prompt: dict


def load_calibration(path, model_repo, source_language, target_language):
    if not path:
        path = Path(__file__).with_name("calibrations")
    direction = f"{source_language.lower()}-{target_language.lower()}"
    path = Path(path)
    if path.is_dir():
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_repo)
        path = path / f"translation_heads_{safe_model}_{direction}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read calibration for {model_repo} {direction}: {path}") from exc
    if data.get("model") != model_repo or data.get("direction") != direction:
        raise ValueError(f"Calibration must match the exact model {model_repo} and direction {direction}")
    runtime = data.get("runtime", {})
    revision = runtime.get("model_revision", "")
    if runtime.get("backend") != "mlx-lm" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Calibration requires MLX measurements at a pinned model revision")
    if not isinstance(runtime.get("quantization"), dict) or not re.fullmatch(r"[0-9a-f]{64}", runtime.get("source_sha256", "")):
        raise ValueError("Calibration is missing quantization or source provenance")
    gate = data.get("promotion_gate", {})
    if not gate.get("eligible_for_promotion"):
        raise ValueError("Calibration has not passed its stability checks")
    layers, n_heads = data.get("num_layers", 0), data.get("num_heads", 0)
    rows = data.get("token_alignment_heads", [])[:8]
    heads = tuple((row["layer"], row["head"]) for row in rows)
    if len(heads) != 8 or len(set(heads)) != 8 or any(
        not isinstance(layer, int) or not isinstance(head, int)
        or not 0 <= layer < layers or not 0 <= head < n_heads
        or not math.isfinite(float(row.get("ts", 0))) or float(row.get("ts", 0)) <= 0.1
        for (layer, head), row in zip(heads, rows)
    ):
        raise ValueError("Calibration requires eight distinct, scored alignment heads within model dimensions")
    checks = data.get("stability_checks", [])
    if len(checks) < 2 or any(
        not check.get("stable_vs_full")
        or not isinstance(check.get("max_abs_ts_delta_vs_full"), (int, float))
        or not 0 <= check["max_abs_ts_delta_vs_full"] <= 0.03
        for check in checks
    ):
        raise ValueError("Calibration requires at least two successful stability checks")
    if data.get("used_pairs", 0) < 100 or any(f.get("kind") != "invalid_annotation" for f in data.get("failures", [])):
        raise ValueError("Calibration requires at least 100 scored pairs and no inference failures")
    return Calibration(model_repo, revision, direction, heads, layers, n_heads,
                       runtime["quantization"], str(path.resolve()), runtime.get("prompt", {}))
