"""Minimal local LoRA for the Qwen text decoder (no PEFT dependency).

Restored from the pre-cleanup training code (git a76f2d0~1). ``lora_b`` is
zero-initialized, so freshly wrapped modules are an exact no-op — the D2
trainer relies on this for its step-0 sanity gate.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DECODER_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        lora_input = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
        update = self.lora_b(self.lora_a(lora_input)) * self.scaling
        return output + update.to(dtype=output.dtype)


def _set_child_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def add_lora_to_linear_modules(
    root: nn.Module,
    *,
    target_names: tuple[str, ...] = DECODER_LORA_TARGETS,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Wrap matching ``nn.Linear`` leaves in-place; returns wrapped names."""
    replaced: list[str] = []
    for name, module in list(root.named_modules()):
        if not name or isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in target_names:
            continue
        _set_child_module(
            root,
            name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        replaced.append(name)
    return replaced


def lora_parameters(root: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in root.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.lora_a.parameters())
            params.extend(module.lora_b.parameters())
    return params


def lora_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    """State dict restricted to LoRA weights (small, checkpoint-friendly)."""
    return {
        key: value
        for key, value in root.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }
