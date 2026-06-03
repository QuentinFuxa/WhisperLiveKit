from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RNNTGreedyDecodeResult:
    compact_token_ids: list[int]
    blank_count: int
    decision_count: int
    forced_advance_count: int
    last_prediction_index: int


def rnnt_prefix_targets(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_index: int,
) -> torch.Tensor:
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, max_target_length]")
    batch, max_target_length = targets.shape
    prefixes = targets.new_full((batch, max_target_length + 1), int(blank_index))
    if max_target_length:
        prefixes[:, 1:] = targets
    for batch_idx, length in enumerate(target_lengths.detach().cpu().tolist()):
        length = int(length)
        if length < 0 or length > max_target_length:
            raise ValueError(f"target length out of range: {length}")
        if length + 1 < prefixes.shape[1]:
            prefixes[batch_idx, length + 1 :] = int(blank_index)
    return prefixes


def _rnnt_single_log_likelihood(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    input_length: int,
    target_length: int,
    blank_index: int,
    label_frame_targets: torch.Tensor | None = None,
    duration_prior_weight: float = 0.0,
    duration_prior_sigma_frames: float = 6.0,
    duration_prior_max_penalty: float = 8.0,
) -> torch.Tensor:
    if input_length <= 0:
        raise ValueError("RNNT input_length must be > 0")
    if target_length < 0:
        raise ValueError("RNNT target_length must be >= 0")
    if log_probs.shape[0] < input_length:
        raise ValueError("RNNT log_probs shorter than input_length")
    if log_probs.shape[1] < target_length + 1:
        raise ValueError("RNNT log_probs must include U + 1 prediction states")

    lp = log_probs[:input_length, : target_length + 1, :]
    y = targets[:target_length].to(dtype=torch.long)
    if target_length and torch.any(y == int(blank_index)):
        raise ValueError("RNNT targets must not contain blank")
    if label_frame_targets is not None and label_frame_targets.shape[0] < target_length:
        raise ValueError("RNNT label_frame_targets shorter than target_length")

    use_duration_prior = label_frame_targets is not None and duration_prior_weight > 0.0
    if use_duration_prior and duration_prior_sigma_frames <= 0.0:
        raise ValueError("duration_prior_sigma_frames must be > 0")
    if use_duration_prior and duration_prior_max_penalty < 0.0:
        raise ValueError("duration_prior_max_penalty must be >= 0")

    def label_prior(frame_idx: int, target_idx: int) -> torch.Tensor:
        if not use_duration_prior:
            return lp.new_zeros(())
        target_frame = label_frame_targets[target_idx].to(device=lp.device, dtype=lp.dtype)
        delta = (lp.new_tensor(float(frame_idx)) - target_frame) / float(
            duration_prior_sigma_frames
        )
        penalty = torch.minimum(
            delta.pow(2),
            lp.new_tensor(float(duration_prior_max_penalty)),
        )
        return -float(duration_prior_weight) * penalty

    zero = lp.new_zeros(())
    first_row: list[torch.Tensor] = [zero]
    for target_idx in range(target_length):
        token = int(y[target_idx].item())
        first_row.append(
            first_row[-1] + lp[0, target_idx, token] + label_prior(0, target_idx)
        )

    previous_row = first_row
    for frame_idx in range(1, input_length):
        row: list[torch.Tensor] = [
            previous_row[0] + lp[frame_idx - 1, 0, int(blank_index)]
        ]
        for target_idx in range(1, target_length + 1):
            token = int(y[target_idx - 1].item())
            blank_score = (
                previous_row[target_idx]
                + lp[frame_idx - 1, target_idx, int(blank_index)]
            )
            label_score = (
                row[target_idx - 1]
                + lp[frame_idx, target_idx - 1, token]
                + label_prior(frame_idx, target_idx - 1)
            )
            row.append(torch.logaddexp(blank_score, label_score))
        previous_row = row

    return previous_row[target_length] + lp[
        input_length - 1,
        target_length,
        int(blank_index),
    ]


def rnnt_forward_backward_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_index: int,
    label_frame_targets: torch.Tensor | None = None,
    duration_prior_weight: float = 0.0,
    duration_prior_sigma_frames: float = 6.0,
    duration_prior_max_penalty: float = 8.0,
    reduction: str = "mean",
    zero_infinity: bool = True,
    normalize_by_length: bool = False,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, T, U + 1, vocab]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, U]")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets batch sizes differ")
    if logits.shape[0] != input_lengths.numel() or logits.shape[0] != target_lengths.numel():
        raise ValueError("length tensors must have one item per batch element")
    if label_frame_targets is not None and label_frame_targets.shape != targets.shape:
        raise ValueError("label_frame_targets must have the same shape as targets")
    if not 0 <= int(blank_index) < logits.shape[-1]:
        raise ValueError("blank_index out of range")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown RNNT reduction: {reduction}")
    if duration_prior_weight < 0.0:
        raise ValueError("duration_prior_weight must be >= 0")
    if duration_prior_weight > 0.0 and label_frame_targets is None:
        raise ValueError("label_frame_targets are required when duration_prior_weight > 0")

    log_probs = logits.float().log_softmax(dim=-1)
    losses: list[torch.Tensor] = []
    for batch_idx in range(logits.shape[0]):
        input_length = int(input_lengths[batch_idx].detach().cpu().item())
        target_length = int(target_lengths[batch_idx].detach().cpu().item())
        log_likelihood = _rnnt_single_log_likelihood(
            log_probs[batch_idx],
            targets[batch_idx],
            input_length=input_length,
            target_length=target_length,
            blank_index=int(blank_index),
            label_frame_targets=(
                None
                if label_frame_targets is None
                else label_frame_targets[batch_idx]
            ),
            duration_prior_weight=duration_prior_weight,
            duration_prior_sigma_frames=duration_prior_sigma_frames,
            duration_prior_max_penalty=duration_prior_max_penalty,
        )
        loss = -log_likelihood
        if normalize_by_length:
            normalizer = max(1, input_length + target_length)
            loss = loss / float(normalizer)
        if zero_infinity and not torch.isfinite(loss):
            loss = log_probs.new_zeros(())
        losses.append(loss)

    stacked = torch.stack(losses)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def rnnt_nonblank_rate_loss(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_index: int,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, T, U + 1, vocab]")
    if logits.shape[0] != input_lengths.numel() or logits.shape[0] != target_lengths.numel():
        raise ValueError("length tensors must have one item per batch element")
    if not 0 <= int(blank_index) < logits.shape[-1]:
        raise ValueError("blank_index out of range")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown RNNT rate reduction: {reduction}")

    log_probs = logits.float().log_softmax(dim=-1)
    blank_probs = log_probs[..., int(blank_index)].exp()
    losses: list[torch.Tensor] = []
    pred_rates: list[torch.Tensor] = []
    target_rates: list[torch.Tensor] = []
    max_time = logits.shape[1]
    max_prefix = logits.shape[2]
    for batch_idx in range(logits.shape[0]):
        input_length = int(input_lengths[batch_idx].detach().cpu().item())
        target_length = int(target_lengths[batch_idx].detach().cpu().item())
        if input_length <= 0:
            raise ValueError("RNNT input_length must be > 0")
        if target_length < 0:
            raise ValueError("RNNT target_length must be >= 0")
        if input_length > max_time:
            raise ValueError("RNNT input_length exceeds logits time dimension")
        if target_length + 1 > max_prefix:
            raise ValueError("RNNT target_length exceeds logits prediction dimension")

        valid_blank_probs = blank_probs[
            batch_idx,
            :input_length,
            : target_length + 1,
        ]
        pred_rate = (1.0 - valid_blank_probs).mean()
        target_rate = logits.new_tensor(
            float(target_length) / float(input_length + target_length),
            dtype=torch.float32,
        )
        losses.append((pred_rate - target_rate).pow(2))
        pred_rates.append(pred_rate.detach())
        target_rates.append(target_rate.detach())

    stacked = torch.stack(losses)
    pred_rate = torch.stack(pred_rates).mean()
    target_rate = torch.stack(target_rates).mean()
    if reduction == "none":
        loss = stacked
    elif reduction == "sum":
        loss = stacked.sum()
    else:
        loss = stacked.mean()
    return loss, pred_rate, target_rate


def rnnt_aligned_token_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    label_frame_targets: torch.Tensor,
    *,
    blank_index: int,
    window_frames: int = 2,
    blank_margin: float = 1.0,
    other_margin: float = 0.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError("logits must have shape [batch, T, U + 1, vocab]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, U]")
    if label_frame_targets.shape != targets.shape:
        raise ValueError("label_frame_targets must have the same shape as targets")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets batch sizes differ")
    if logits.shape[0] != input_lengths.numel() or logits.shape[0] != target_lengths.numel():
        raise ValueError("length tensors must have one item per batch element")
    if not 0 <= int(blank_index) < logits.shape[-1]:
        raise ValueError("blank_index out of range")
    if window_frames < 0:
        raise ValueError("window_frames must be >= 0")
    if blank_margin < 0.0:
        raise ValueError("blank_margin must be >= 0")
    if other_margin < 0.0:
        raise ValueError("other_margin must be >= 0")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown RNNT margin reduction: {reduction}")

    blank_losses: list[torch.Tensor] = []
    other_losses: list[torch.Tensor] = []
    blank_margins: list[torch.Tensor] = []
    other_margins: list[torch.Tensor] = []
    max_time = logits.shape[1]
    max_prefix = logits.shape[2]
    vocab_size = logits.shape[-1]
    for batch_idx in range(logits.shape[0]):
        input_length = int(input_lengths[batch_idx].detach().cpu().item())
        target_length = int(target_lengths[batch_idx].detach().cpu().item())
        if input_length <= 0:
            raise ValueError("RNNT input_length must be > 0")
        if target_length < 0:
            raise ValueError("RNNT target_length must be >= 0")
        if input_length > max_time:
            raise ValueError("RNNT input_length exceeds logits time dimension")
        if target_length + 1 > max_prefix:
            raise ValueError("RNNT target_length exceeds logits prediction dimension")

        for target_idx in range(target_length):
            token = int(targets[batch_idx, target_idx].detach().cpu().item())
            if token == int(blank_index):
                raise ValueError("RNNT targets must not contain blank")
            if not 0 <= token < vocab_size:
                raise ValueError("RNNT target token out of range")
            target_frame = int(
                label_frame_targets[batch_idx, target_idx].detach().cpu().item()
            )
            if target_frame < 0:
                raise ValueError("label_frame_targets must be >= 0 for valid targets")
            target_frame = min(target_frame, input_length - 1)
            start = max(0, target_frame - int(window_frames))
            end = min(input_length, target_frame + int(window_frames) + 1)
            state_logits = logits[batch_idx, start:end, target_idx, :].float()
            target_logits = state_logits[:, token]
            blank_logits = state_logits[:, int(blank_index)]

            blank_frame_losses = torch.nn.functional.softplus(
                blank_logits - target_logits + float(blank_margin)
            )
            blank_frame_margins = target_logits - blank_logits
            blank_loss_idx = int(blank_frame_losses.detach().argmin().item())
            blank_losses.append(blank_frame_losses[blank_loss_idx])
            blank_margins.append(blank_frame_margins.detach().max())

            competitor_logits = state_logits.clone()
            competitor_logits[:, int(blank_index)] = -torch.inf
            competitor_logits[:, token] = -torch.inf
            other_logits = competitor_logits.max(dim=-1).values
            other_frame_losses = torch.nn.functional.softplus(
                other_logits - target_logits + float(other_margin)
            )
            other_frame_margins = target_logits - other_logits
            other_loss_idx = int(other_frame_losses.detach().argmin().item())
            other_losses.append(other_frame_losses[other_loss_idx])
            other_margins.append(other_frame_margins.detach().max())

    if not blank_losses:
        zero = logits.new_zeros(())
        return zero, zero, zero, zero

    blank_stacked = torch.stack(blank_losses)
    other_stacked = torch.stack(other_losses)
    blank_margin = torch.stack(blank_margins).mean()
    other_margin = torch.stack(other_margins).mean()
    if reduction == "none":
        blank_loss = blank_stacked
        other_loss = other_stacked
    elif reduction == "sum":
        blank_loss = blank_stacked.sum()
        other_loss = other_stacked.sum()
    else:
        blank_loss = blank_stacked.mean()
        other_loss = other_stacked.mean()
    return blank_loss, other_loss, blank_margin, other_margin


def _aligned_window_target_loss(
    state_logits: torch.Tensor,
    *,
    token: int,
    blank_index: int,
    sampled_negative_count: int,
    frequent_negative_indices: list[int] | None = None,
) -> torch.Tensor:
    if state_logits.ndim != 2:
        raise ValueError("state_logits must have shape [window, vocab]")
    if state_logits.shape[0] <= 0:
        raise ValueError("state_logits window must not be empty")
    vocab_size = state_logits.shape[-1]
    if not 0 <= int(token) < vocab_size:
        raise ValueError("target token out of range")
    if int(token) == int(blank_index):
        raise ValueError("aligned-window targets must not contain blank")
    if not 0 <= int(blank_index) < vocab_size:
        raise ValueError("blank_index out of range")

    logits = state_logits.float()
    frequent = [
        int(index)
        for index in (frequent_negative_indices or [])
        if 0 <= int(index) < vocab_size
        and int(index) not in {int(token), int(blank_index)}
    ]
    if sampled_negative_count > 0 or frequent:
        selected_ids: list[int] = [int(token), int(blank_index)]
        seen = set(selected_ids)
        for index in frequent:
            if index not in seen:
                selected_ids.append(index)
                seen.add(index)
        hard_count = min(int(sampled_negative_count), max(0, vocab_size - len(seen)))
        if hard_count:
            scores = logits.detach().max(dim=0).values.clone()
            for index in seen:
                scores[index] = -torch.inf
            hard_ids = [int(index) for index in scores.topk(hard_count).indices.tolist()]
            for index in hard_ids:
                if index not in seen:
                    selected_ids.append(index)
                    seen.add(index)
        selected = torch.tensor(selected_ids, device=logits.device, dtype=torch.long)
        class_logits = logits.index_select(dim=-1, index=selected)
        target_log_probs = class_logits.log_softmax(dim=-1)[:, 0]
    else:
        target_log_probs = logits.log_softmax(dim=-1)[:, int(token)]

    # Normalize by window size so a wide window does not make the loss negative
    # just because the same token is plausible on multiple adjacent frames.
    return -torch.logsumexp(target_log_probs, dim=0) + logits.new_tensor(
        float(logits.shape[0])
    ).log()


def _aligned_window_blank_loss(
    blank_logits: torch.Tensor,
    *,
    blank_index: int,
    sampled_negative_count: int,
) -> torch.Tensor:
    if blank_logits.ndim != 2:
        raise ValueError("blank_logits must have shape [frames, vocab]")
    if blank_logits.shape[0] == 0:
        return blank_logits.new_zeros(())
    vocab_size = blank_logits.shape[-1]
    if not 0 <= int(blank_index) < vocab_size:
        raise ValueError("blank_index out of range")

    logits = blank_logits.float()
    if sampled_negative_count > 0:
        hard_count = min(int(sampled_negative_count), max(0, vocab_size - 1))
        blank_col = logits[:, int(blank_index)].unsqueeze(-1)
        if hard_count:
            scores = logits.detach().clone()
            scores[:, int(blank_index)] = -torch.inf
            hard_logits = logits.gather(dim=-1, index=scores.topk(hard_count, dim=-1).indices)
            class_logits = torch.cat([blank_col, hard_logits], dim=-1)
        else:
            class_logits = blank_col
        labels = torch.zeros(class_logits.shape[0], device=logits.device, dtype=torch.long)
        return F.cross_entropy(class_logits, labels)

    labels = torch.full(
        (logits.shape[0],),
        int(blank_index),
        device=logits.device,
        dtype=torch.long,
    )
    return F.cross_entropy(logits, labels)


def aligned_window_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    label_frame_targets: torch.Tensor,
    *,
    blank_index: int,
    window_frames: int = 3,
    blank_loss_weight: float = 0.05,
    sampled_negative_count: int = 0,
    token_class_weights: torch.Tensor | None = None,
    frequent_negative_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, T, vocab]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, U]")
    if label_frame_targets.shape != targets.shape:
        raise ValueError("label_frame_targets must have the same shape as targets")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets batch sizes differ")
    if logits.shape[0] != input_lengths.numel() or logits.shape[0] != target_lengths.numel():
        raise ValueError("length tensors must have one item per batch element")
    if not 0 <= int(blank_index) < logits.shape[-1]:
        raise ValueError("blank_index out of range")
    if window_frames < 0:
        raise ValueError("window_frames must be >= 0")
    if blank_loss_weight < 0.0:
        raise ValueError("blank_loss_weight must be >= 0")
    if sampled_negative_count < 0:
        raise ValueError("sampled_negative_count must be >= 0")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unknown aligned-window CE reduction: {reduction}")
    if token_class_weights is not None:
        if token_class_weights.ndim != 1 or token_class_weights.numel() != logits.shape[-1]:
            raise ValueError("token_class_weights must have shape [vocab]")
        if torch.any(token_class_weights < 0):
            raise ValueError("token_class_weights must be >= 0")

    token_losses: list[torch.Tensor] = []
    blank_losses: list[torch.Tensor] = []
    blank_margins: list[torch.Tensor] = []
    other_margins: list[torch.Tensor] = []
    max_time = logits.shape[1]
    vocab_size = logits.shape[-1]
    class_weights = (
        None
        if token_class_weights is None
        else token_class_weights.to(device=logits.device, dtype=logits.float().dtype)
    )
    if isinstance(frequent_negative_indices, torch.Tensor):
        frequent_indices = [
            int(index)
            for index in frequent_negative_indices.detach().cpu().tolist()
        ]
    elif frequent_negative_indices is None:
        frequent_indices = []
    else:
        frequent_indices = [int(index) for index in frequent_negative_indices]
    for batch_idx in range(logits.shape[0]):
        input_length = int(input_lengths[batch_idx].detach().cpu().item())
        target_length = int(target_lengths[batch_idx].detach().cpu().item())
        if input_length <= 0:
            raise ValueError("aligned-window input_length must be > 0")
        if target_length < 0:
            raise ValueError("aligned-window target_length must be >= 0")
        if input_length > max_time:
            raise ValueError("aligned-window input_length exceeds logits time dimension")
        if target_length > targets.shape[1]:
            raise ValueError("aligned-window target_length exceeds targets dimension")

        covered = torch.zeros(input_length, device=logits.device, dtype=torch.bool)
        for target_idx in range(target_length):
            token = int(targets[batch_idx, target_idx].detach().cpu().item())
            if token == int(blank_index):
                raise ValueError("aligned-window targets must not contain blank")
            if not 0 <= token < vocab_size:
                raise ValueError("aligned-window target token out of range")
            target_frame = int(
                label_frame_targets[batch_idx, target_idx].detach().cpu().item()
            )
            if target_frame < 0:
                raise ValueError("label_frame_targets must be >= 0 for valid targets")
            target_frame = min(target_frame, input_length - 1)
            start = max(0, target_frame - int(window_frames))
            end = min(input_length, target_frame + int(window_frames) + 1)
            covered[start:end] = True

            state_logits = logits[batch_idx, start:end, :].float()
            target_loss = _aligned_window_target_loss(
                state_logits,
                token=token,
                blank_index=int(blank_index),
                sampled_negative_count=int(sampled_negative_count),
                frequent_negative_indices=frequent_indices,
            )
            if class_weights is not None:
                target_loss = target_loss * class_weights[token]
            token_losses.append(target_loss)

            target_logits = state_logits[:, token]
            blank_logits = state_logits[:, int(blank_index)]
            competitor_logits = state_logits.clone()
            competitor_logits[:, int(blank_index)] = -torch.inf
            competitor_logits[:, token] = -torch.inf
            other_logits = competitor_logits.max(dim=-1).values
            blank_margins.append((target_logits - blank_logits).detach().max())
            other_margins.append((target_logits - other_logits).detach().max())

        outside = ~covered
        if outside.any():
            blank_losses.append(
                _aligned_window_blank_loss(
                    logits[batch_idx, :input_length, :][outside],
                    blank_index=int(blank_index),
                    sampled_negative_count=int(sampled_negative_count),
                )
            )

    if not token_losses:
        zero = logits.new_zeros(())
        return zero, zero, zero, zero, zero

    token_stacked = torch.stack(token_losses)
    token_loss = token_stacked.sum() if reduction == "sum" else token_stacked.mean()
    if blank_losses:
        blank_stacked = torch.stack(blank_losses)
        blank_loss = blank_stacked.sum() if reduction == "sum" else blank_stacked.mean()
    else:
        blank_loss = logits.new_zeros(())
    total_loss = token_loss + float(blank_loss_weight) * blank_loss
    blank_margin = (
        torch.stack(blank_margins).mean() if blank_margins else logits.new_zeros(())
    )
    other_margin = (
        torch.stack(other_margins).mean() if other_margins else logits.new_zeros(())
    )
    return total_loss, token_loss, blank_loss, blank_margin, other_margin


def rnnt_greedy_decode(
    *,
    frame_count: int,
    step_fn: Callable[[int, int, int], int],
    blank_index: int,
    start_prediction_index: int | None = None,
    max_symbols_per_frame: int = 8,
) -> RNNTGreedyDecodeResult:
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")
    if max_symbols_per_frame <= 0:
        raise ValueError("max_symbols_per_frame must be > 0")

    previous = int(blank_index if start_prediction_index is None else start_prediction_index)
    emitted: list[int] = []
    blank_count = 0
    decision_count = 0
    forced_advance_count = 0
    for frame_index in range(int(frame_count)):
        symbols_this_frame = 0
        while True:
            compact_id = int(step_fn(frame_index, previous, symbols_this_frame))
            decision_count += 1
            if compact_id == int(blank_index):
                blank_count += 1
                break
            emitted.append(compact_id)
            previous = compact_id
            symbols_this_frame += 1
            if symbols_this_frame >= int(max_symbols_per_frame):
                forced_advance_count += 1
                break

    return RNNTGreedyDecodeResult(
        compact_token_ids=emitted,
        blank_count=blank_count,
        decision_count=decision_count,
        forced_advance_count=forced_advance_count,
        last_prediction_index=previous,
    )
