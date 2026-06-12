#!/usr/bin/env python3
"""Measured per-update FLOPs for the qwen3-streaming backends.

Ground truth via ``torch.utils.flop_counter.FlopCounterMode`` on the real
modules (real tower weights; random-init text model — FLOPs are value-
independent). Position/step counts per update are the code-exact per-segment
averages: windowed prefill grows 44->212 positions (avg 128) with 7->49
sequential steps (avg 28); causal forwards ~62 positions (24 audio + 8 tail
+ ~30 draft) over a persistent cache with ~9 sequential steps.

Run: python scripts/measure_flops.py --tower-state-dict <tower.pt|.safetensors>
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from torch.utils.flop_counter import FlopCounterMode

from whisperlivekit.qwen3_streaming.causal import (
    QwenAudioCausalKVEncoder,
    load_tower_checkpoint,
)
from whisperlivekit.qwen3_streaming.model import (
    QwenAudioSurgeryEncoder,
    _register_qwen3_asr_transformers,
)
from whisperlivekit.qwen3_streaming.model_config import RealtimeAudioConfig


def gflops(fn) -> float:
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        fn()
    return counter.get_total_flops() / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tower-state-dict", type=Path, required=True)
    args = parser.parse_args()

    _register_qwen3_asr_transformers()
    from transformers import AutoConfig, AutoModel

    hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-ASR-0.6B")
    model = AutoModel.from_config(hf_config)  # random init, no weight download
    thinker = getattr(model, "thinker", model)
    tower = thinker.audio_tower

    class _Holder:
        class audio_encoder:
            audio_tower = tower

    load_tower_checkpoint(_Holder, args.tower_state_dict)
    tower.eval()
    text_model = thinker.model.eval()
    text_dtype = next(text_model.parameters()).dtype
    d_text = int(hf_config.thinker_config.text_config.hidden_size)

    # ---- encoder, causal steady state (15 s KV window filled) ----
    enc_c = QwenAudioCausalKVEncoder(
        tower,
        RealtimeAudioConfig(
            d_model=d_text,
            qwen_audio_left_context_sec=15.0,
            qwen_audio_block_bidirectional=True,
            qwen_audio_block_frames=192,
        ),
    ).eval()
    state = enc_c.init_state()
    with torch.no_grad():
        for _ in range(9):
            _, state = enc_c.forward_chunk(torch.randn(1, 192, 128), state)
    g_enc_causal = gflops(
        lambda: enc_c.forward_chunk(torch.randn(1, 192, 128), state)
    )

    # ---- encoder, windowed steady state (12 s left + 640 ms right) ----
    enc_w = QwenAudioSurgeryEncoder(tower, RealtimeAudioConfig(d_model=d_text)).eval()
    wstate = enc_w.init_state()
    with torch.no_grad():
        for _ in range(10):
            _, wstate = enc_w.forward_chunk(torch.randn(1, 200, 128), wstate)
    g_enc_win = gflops(lambda: enc_w.forward_chunk(torch.randn(1, 200, 128), wstate))

    # ---- decoder ----
    def run_positions(n_new: int, n_past: int) -> float:
        past = None
        if n_past:
            with torch.no_grad():
                out = text_model(
                    inputs_embeds=torch.randn(1, n_past, d_text, dtype=text_dtype),
                    use_cache=True,
                )
            past = out.past_key_values
        return gflops(
            lambda: text_model(
                inputs_embeds=torch.randn(1, n_new, d_text, dtype=text_dtype),
                past_key_values=past,
                use_cache=True,
            )
        )

    g_prefill_avg = run_positions(128, 0)
    g_prefill_end = run_positions(212, 0)
    g_roll = run_positions(62, 120)
    g_step = run_positions(1, 250)

    win_avg = g_enc_win + g_prefill_avg + 28 * g_step
    win_peak = g_enc_win + g_prefill_end + 49 * g_step
    causal = g_enc_causal + g_roll + 9 * g_step

    print(f"encoder  windowed 2.0s update @steady : {g_enc_win:8.2f} GFLOPs")
    print(f"encoder  causal  1.92s block @steady  : {g_enc_causal:8.2f} GFLOPs")
    print(f"decoder  prefill 128 / 212 pos        : {g_prefill_avg:8.2f} / {g_prefill_end:.2f} GFLOPs")
    print(f"decoder  rolling 62 pos @past120      : {g_roll:8.2f} GFLOPs")
    print(f"decoder  1 step @past250              : {g_step:8.4f} GFLOPs")
    print()
    print(f"WINDOWED avg : {win_avg:7.1f} GFLOPs/update -> {win_avg / 2.0:6.1f} per audio-second")
    print(f"WINDOWED peak: {win_peak:7.1f} GFLOPs/update -> {win_peak / 2.0:6.1f} per audio-second")
    print(f"CAUSAL       : {causal:7.1f} GFLOPs/block  -> {causal / 1.92:6.1f} per audio-second (constant)")


if __name__ == "__main__":
    main()
