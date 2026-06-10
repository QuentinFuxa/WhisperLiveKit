"""Teacher-forced CE objective for decoder co-adaptation (D2).

The student audio path produces ``frame_hidden`` (cached audio embeddings,
adapter-projected). The decoder is teacher-forced on the Qwen ASR prompt with
audio placeholders scattered to those embeddings, followed by the reference
transcript; cross-entropy applies only to transcript (+EOS) positions.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .cached_full_hypothesis import (
    expand_audio_prompt_placeholders,
    qwen_asr_prompt_text,
)
from .native_realtime_model import _cached_audio_prefix_embeds


def build_ce_inputs(
    tokenizer,
    *,
    audio_steps: int,
    language: str,
    target_text: str,
    audio_placeholder_token_id: int,
    context: str = "",
    max_target_tokens: int = 384,
    add_eos: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Returns (prompt_ids_with_expanded_audio, target_ids, labels).

    ``labels`` covers the full sequence (prompt + targets): ``-100`` on every
    prompt/audio position, token ids on transcript positions.
    """
    prompt_template = tokenizer.encode(
        qwen_asr_prompt_text(context=context, language=language),
        add_special_tokens=False,
    )
    prompt_ids = expand_audio_prompt_placeholders(
        prompt_template,
        audio_placeholder_token_id=audio_placeholder_token_id,
        audio_steps=int(audio_steps),
    )
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_ids = target_ids[:max_target_tokens]
    if add_eos and tokenizer.eos_token_id is not None:
        target_ids = target_ids + [int(tokenizer.eos_token_id)]
    labels = [-100] * len(prompt_ids) + [int(t) for t in target_ids]
    return prompt_ids, [int(t) for t in target_ids], labels


def ce_forward(
    model,
    frame_hidden: torch.Tensor,
    *,
    prompt_ids: list[int],
    target_ids: list[int],
    audio_placeholder_token_id: int,
) -> tuple[torch.Tensor, dict]:
    """One teacher-forced decoder pass; CE on transcript positions only.

    frame_hidden: [1, steps, d_model] adapter-projected audio embeddings.
    """
    device = frame_hidden.device
    prefix_embeds = _cached_audio_prefix_embeds(
        model,
        frame_hidden,
        prefix_token_ids=prompt_ids,
        audio_placeholder_token_id=int(audio_placeholder_token_id),
    )
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
    target_embeds = model.embed_tokens(target_tensor)
    inputs_embeds = torch.cat(
        [prefix_embeds, target_embeds.to(dtype=prefix_embeds.dtype)], dim=1
    )
    outputs = model.text_model(inputs_embeds=inputs_embeds, use_cache=False)
    logits = model.lm_head(outputs.last_hidden_state)

    # Predict position i+1 from position i: CE over the target span.
    prompt_len = len(prompt_ids)
    pred_logits = logits[:, prompt_len - 1 : prompt_len - 1 + len(target_ids), :]
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]).float(),
        target_tensor.reshape(-1),
    )
    with torch.no_grad():
        accuracy = float(
            (pred_logits.argmax(dim=-1) == target_tensor).float().mean()
        )
    return loss, {"token_accuracy": accuracy, "target_tokens": len(target_ids)}
