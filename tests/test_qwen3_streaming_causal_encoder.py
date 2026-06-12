"""Causal-KV encoder with fixed attention blocks: pacing invariance.

The trained regime is fixed 96/192-frame bidirectional blocks. Production
pacing delivers variable-size mel chunks, so the encoder buffers and consumes
exact multiples of ``block_frames``. These tests pin the invariant: the
output stream is identical no matter how the input is chopped.
"""

import pytest

torch = pytest.importorskip("torch")

from whisperlivekit.qwen3_streaming.causal import (  # noqa: E402
    QwenAudioCausalKVEncoder,
)

from qwen3_streaming_fakes import N_MELS, TinyQwenAudioTower, tiny_config  # noqa: E402

BLOCK = 32  # 4 output steps per block on the tiny tower (8 mel frames -> 1 step)


def make_encoder(**kwargs):
    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_block_bidirectional=True)
    return QwenAudioCausalKVEncoder(tower, config, block_frames=BLOCK).eval()


def feed(encoder, mels, chunk_sizes):
    state = encoder.init_state()
    outputs = []
    cursor = 0
    i = 0
    while cursor < mels.shape[1]:
        size = chunk_sizes[i % len(chunk_sizes)]
        hidden, state = encoder.forward_chunk(
            mels[:, cursor : cursor + size, :], state
        )
        outputs.append(hidden)
        cursor += size
        i += 1
    return torch.cat(outputs, dim=1), state


@pytest.mark.parametrize(
    "chunk_sizes",
    [[7, 50, 100, 3, 96], [1], [200], [33, 5, 190], [BLOCK]],
    ids=["irregular", "tiny", "burst", "mixed", "exact"],
)
def test_pacing_invariance(chunk_sizes):
    torch.manual_seed(1)
    mels = torch.randn(1, 4 * BLOCK, N_MELS)

    encoder = make_encoder()
    reference, _ = feed(encoder, mels, [BLOCK])

    encoder2 = make_encoder()
    paced, state = feed(encoder2, mels, chunk_sizes)

    torch.testing.assert_close(paced, reference, rtol=1e-4, atol=1e-4)
    pending = 0 if state.mel_buffer is None else int(state.mel_buffer.shape[1])
    assert pending < BLOCK


def test_blocked_encoder_matches_external_block_feeding():
    """block_frames inside the encoder == the eval harness feeding exact
    blocks to an unblocked encoder (the configuration the WER numbers come
    from)."""
    torch.manual_seed(2)
    mels = torch.randn(1, 3 * BLOCK, N_MELS)

    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_block_bidirectional=True)
    unblocked = QwenAudioCausalKVEncoder(tower, config, block_frames=0).eval()
    state = unblocked.init_state()
    reference = []
    for start in range(0, mels.shape[1], BLOCK):
        hidden, state = unblocked.forward_chunk(
            mels[:, start : start + BLOCK, :], state
        )
        reference.append(hidden)
    reference = torch.cat(reference, dim=1)

    blocked = make_encoder()
    out, _ = feed(blocked, mels, [70, 11, 200])
    torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-4)


def test_flush_pending_encodes_partial_block_once():
    encoder = make_encoder()
    torch.manual_seed(3)
    mels = torch.randn(1, BLOCK + 17, N_MELS)

    state = encoder.init_state()
    hidden, state = encoder.forward_chunk(mels, state)
    assert hidden.shape[1] == BLOCK // 8  # one full block emitted
    assert int(state.mel_buffer.shape[1]) == 17

    flushed, state = encoder.flush_pending(state)
    # 17 pending frames -> 2 whole 8-frame conv blocks, 1-frame sub-block
    # remainder dropped.
    assert flushed.shape[1] == 2
    again, state = encoder.flush_pending(state)
    assert again.shape[1] == 0  # idempotent


def test_flush_pending_drops_subblock_remainder_only():
    encoder = make_encoder()
    torch.manual_seed(4)
    state = encoder.init_state()
    _, state = encoder.forward_chunk(torch.randn(1, 5, N_MELS), state)
    flushed, state = encoder.flush_pending(state)
    assert flushed.shape[1] == 0  # < one conv block: nothing decodable


def test_block_frames_must_be_multiple_of_chunk_frames():
    tower = TinyQwenAudioTower()
    config = tiny_config()
    with pytest.raises(ValueError):
        QwenAudioCausalKVEncoder(tower, config, block_frames=30)


def test_block_frames_excludes_mutable_tail():
    tower = TinyQwenAudioTower()
    config = tiny_config()
    with pytest.raises(ValueError):
        QwenAudioCausalKVEncoder(
            tower, config, block_frames=BLOCK, mutable_tail_sec=1.0
        )


def test_config_level_block_frames_validation():
    with pytest.raises(ValueError):
        tiny_config(qwen_audio_block_frames=12)  # not a multiple of 8
    with pytest.raises(ValueError):
        tiny_config(qwen_audio_block_frames=96, qwen_audio_mutable_tail_sec=1.0)
    config = tiny_config(qwen_audio_block_frames=96)
    tower = TinyQwenAudioTower()
    encoder = QwenAudioCausalKVEncoder(tower, config)
    assert encoder.block_frames == 96


def test_tower_checkpoint_loading_pt_and_safetensors(tmp_path):
    from whisperlivekit.qwen3_streaming.causal import (
        Qwen3ASRRealtimeQwenAudioCausalModel,
        load_tower_checkpoint,
        resolve_tower_checkpoint,
    )
    from qwen3_streaming_fakes import D_MODEL, FakeCachingQwenTextModel

    torch.manual_seed(0)
    model = Qwen3ASRRealtimeQwenAudioCausalModel(
        tiny_config(),
        qwen_model_id="fake",
        audio_tower=TinyQwenAudioTower(),
        text_model=FakeCachingQwenTextModel(),
        lm_head=torch.nn.Linear(D_MODEL, 16, bias=False),
        bos_token_id=1,
    )
    torch.manual_seed(7)
    donor = TinyQwenAudioTower()

    pt_path = tmp_path / "tower.pt"
    torch.save(
        {"tower_state_dict": donor.state_dict(), "step": 60000, "gate_wer": 0.2},
        pt_path,
    )
    meta = load_tower_checkpoint(model, resolve_tower_checkpoint(str(pt_path)))
    assert meta["step"] == 60000
    torch.testing.assert_close(
        model.audio_encoder.audio_tower.proj1.weight, donor.proj1.weight
    )

    from safetensors.torch import save_file

    st_dir = tmp_path / "hub_repo"
    st_dir.mkdir()
    donor2 = TinyQwenAudioTower()
    save_file(donor2.state_dict(), str(st_dir / "model.safetensors"))
    resolved = resolve_tower_checkpoint(str(st_dir))
    assert resolved.suffix == ".safetensors"
    load_tower_checkpoint(model, resolved)
    torch.testing.assert_close(
        model.audio_encoder.audio_tower.proj1.weight, donor2.proj1.weight
    )

    torch.save({"tower_state_dict": {"bogus": torch.zeros(1)}}, pt_path)
    with pytest.raises(RuntimeError):
        load_tower_checkpoint(model, pt_path)
