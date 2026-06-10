"""Parity tests: the differentiable training forward must equal the streaming
inference execution, step for step. If this holds, D1 training optimizes
exactly what the backend serves."""

import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.native_realtime_model import QwenAudioCausalKVEncoder  # noqa: E402
from qwen3_streaming.tower_distill import (  # noqa: E402
    block_bidirectional_forward,
    distill_loss,
    teacher_forward,
)
from test_mutable_tail import N_MELS, TinyQwenAudioTower, tiny_config  # noqa: E402


def test_training_forward_matches_streaming_inference():
    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_block_bidirectional=True)
    encoder = QwenAudioCausalKVEncoder(tower, config).eval()

    torch.manual_seed(1)
    mels = torch.randn(1, 96, N_MELS)
    block_frames = 32  # 4 steps per block on the tiny tower

    # Streaming: feed block-sized chunks through the causal-KV encoder.
    state = encoder.init_state()
    streamed = []
    for start in range(0, 96, block_frames):
        hidden, state = encoder.forward_chunk(mels[:, start : start + block_frames, :], state)
        streamed.append(hidden)
    streamed = torch.cat(streamed, dim=1)

    # Training: one parallel differentiable forward under the same mask.
    with torch.no_grad():
        trained = block_bidirectional_forward(
            tower,
            mels,
            chunk_frames=encoder.chunk_frames,
            block_frames=block_frames,
            left_context_steps=encoder.left_context_steps,
        )

    torch.testing.assert_close(trained, streamed, rtol=1e-4, atol=1e-4)


def test_training_forward_matches_streaming_with_bounded_left_context():
    torch.manual_seed(0)
    tower = TinyQwenAudioTower().eval()
    config = tiny_config(qwen_audio_block_bidirectional=True)
    encoder = QwenAudioCausalKVEncoder(
        tower, config, left_context_frames=32
    ).eval()  # 4-step window: cross-block masking actually bites

    torch.manual_seed(2)
    mels = torch.randn(1, 128, N_MELS)
    block_frames = 16

    state = encoder.init_state()
    streamed = []
    for start in range(0, 128, block_frames):
        hidden, state = encoder.forward_chunk(mels[:, start : start + block_frames, :], state)
        streamed.append(hidden)
    streamed = torch.cat(streamed, dim=1)

    with torch.no_grad():
        trained = block_bidirectional_forward(
            tower,
            mels,
            chunk_frames=encoder.chunk_frames,
            block_frames=block_frames,
            left_context_steps=encoder.left_context_steps,
        )

    torch.testing.assert_close(trained, streamed, rtol=1e-4, atol=1e-4)


def test_gradients_flow_through_training_forward():
    torch.manual_seed(0)
    tower = TinyQwenAudioTower()
    torch.manual_seed(3)
    mels = torch.randn(2, 64, N_MELS)
    lengths = torch.tensor([64, 32])

    student = block_bidirectional_forward(
        tower,
        mels,
        block_frames=32,
        left_context_steps=100,
        lengths=lengths,
    )
    teacher = torch.randn_like(student)
    loss, stats = distill_loss(
        student, teacher, lengths_steps=torch.tensor([8, 4])
    )
    loss.backward()

    grads = [p.grad for p in tower.parameters() if p.grad is not None]
    assert grads, "no gradients reached the tower"
    assert all(torch.isfinite(g).all() for g in grads)
    assert stats["mse"] > 0


class TinyTowerWithForward(TinyQwenAudioTower):
    """Adds the native full-bidirectional forward the real tower exposes."""

    def forward(self, input_features, feature_lens=None):
        mels = input_features.transpose(0, 1).unsqueeze(0)  # [1, T, n_mels]
        return block_bidirectional_forward(
            self,
            mels,
            block_frames=int(mels.shape[1]),  # one block = fully bidirectional
            left_context_steps=10_000,
        )


def test_teacher_forward_handles_variable_lengths():
    torch.manual_seed(0)
    tower = TinyTowerWithForward().eval()
    torch.manual_seed(4)
    mels = torch.randn(2, 64, N_MELS)
    lengths = torch.tensor([64, 32])

    teacher = teacher_forward(tower, mels, lengths)
    assert teacher.shape[0] == 2
    assert teacher.shape[1] == 8  # longest sample: 64 frames -> 8 steps
    # Padded tail of the short sample is zero.
    assert torch.all(teacher[1, 4:, :] == 0)
