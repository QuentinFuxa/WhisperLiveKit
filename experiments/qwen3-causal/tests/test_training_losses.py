import pytest

torch = pytest.importorskip("torch")

from argparse import Namespace  # noqa: E402

from scripts.train_realtime_tiny_asr import (  # noqa: E402
    build_qwen_ar_prompt_token_ids,
    build_qwen_ar_target_batch,
    build_rnnt_token_frame_targets,
    compact_frame_label_tensors,
    emit_gate_cross_entropy,
    load_manifest_examples,
    qwen_audio_preserve_loss,
    qwen_ar_frame_distill_loss,
    qwen_ar_kl_distill_loss,
    qwen_ar_cross_entropy,
    qwen_ar_streaming_audio_frames_for_training,
    qwen_ar_z_loss,
    prepend_qwen_ar_audio_padding,
    resolve_alignment_loss,
    set_qwen_audio_left_context_sec,
    zero_init_qwen_audio_adapter_blocks,
)
from qwen3_streaming.ctc import CompactCTCVocab  # noqa: E402
from qwen3_streaming.realtime_targets import WordAlignment  # noqa: E402
from qwen3_streaming.rnnt import (  # noqa: E402
    aligned_window_ce_loss,
    rnnt_aligned_token_margin_loss,
    rnnt_greedy_decode,
    rnnt_forward_backward_loss,
    rnnt_nonblank_rate_loss,
    rnnt_prefix_targets,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "hello": [10, 11],
            " world": [12, 13],
        }
        return mapping[text]


def test_emit_rate_loss_weight_zero_preserves_emit_gate_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    emit_logits = torch.randn(2, 4)
    labels = torch.tensor(
        [
            [0, 3, 4, -100],
            [5, 0, 6, 7],
        ],
        dtype=torch.long,
    )

    baseline = emit_gate_cross_entropy(
        logits,
        emit_logits,
        labels,
        vocab_size=8,
        wait_token_id=0,
        gate_loss_weight=0.3,
        gate_wait_weight=0.7,
    )
    with_zero_rate = emit_gate_cross_entropy(
        logits,
        emit_logits,
        labels,
        vocab_size=8,
        wait_token_id=0,
        gate_loss_weight=0.3,
        gate_wait_weight=0.7,
        emit_rate_loss_weight=0.0,
        text_label_smoothing=0.0,
    )

    torch.testing.assert_close(with_zero_rate, baseline)


def test_emit_rate_loss_adds_regularization_term():
    logits = torch.zeros(1, 4, 8)
    emit_logits = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
    labels = torch.tensor([[0, 0, 3, 4]], dtype=torch.long)

    baseline = emit_gate_cross_entropy(
        logits,
        emit_logits,
        labels,
        vocab_size=8,
        wait_token_id=0,
        gate_loss_weight=0.0,
        gate_wait_weight=1.0,
        emit_rate_loss_weight=0.0,
    )
    regularized = emit_gate_cross_entropy(
        logits,
        emit_logits,
        labels,
        vocab_size=8,
        wait_token_id=0,
        gate_loss_weight=0.0,
        gate_wait_weight=1.0,
        emit_rate_loss_weight=0.2,
    )

    assert regularized > baseline


def test_resolve_alignment_loss_preserves_previous_auto_behavior():
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="auto",
                emit_gate_loss_weight=0.0,
                emit_rate_loss_weight=0.0,
            )
        )
        == "frame_ce"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="auto",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "emit_gate"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="ctc",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "ctc"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="compact_ctc",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "compact_ctc"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="qwen_ar_ce",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "qwen_ar_ce"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="qwen_ar_context_distill",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "qwen_ar_context_distill"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="aligned_window_ce",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "aligned_window_ce"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="aligned_window_sampled_ce",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "aligned_window_sampled_ce"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="rnnt_lite",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "rnnt_lite"
    )
    assert (
        resolve_alignment_loss(
            Namespace(
                alignment_loss="rnnt_fb",
                emit_gate_loss_weight=0.2,
                emit_rate_loss_weight=0.0,
            )
        )
        == "rnnt_fb"
    )


class QwenARTinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "hello": [10, 11],
            "world": [12],
            "<|im_start|>system\nctx<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>": [1, 77, 2],
            "<|im_start|>system\nctx<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage French<asr_text>": [1, 77, 3],
        }
        return mapping[text]


def test_build_qwen_ar_target_batch_masks_padding_and_appends_eos():
    input_ids, labels = build_qwen_ar_target_batch(
        ["hello", "world"],
        QwenARTinyTokenizer(),
        eos_token_id=99,
        max_target_tokens=3,
        add_eos=True,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[10, 11, 99], [12, 99, 0]]
    assert labels.tolist() == [[10, 11, 99], [12, 99, -100]]


def test_build_qwen_ar_prompt_token_ids_uses_per_sample_language():
    prompts = build_qwen_ar_prompt_token_ids(
        ["English", "French"],
        QwenARTinyTokenizer(),
        context="ctx",
        default_language="English",
        audio_placeholder_token_id=77,
    )

    assert prompts == [[1, 77, 2], [1, 77, 3]]


def test_qwen_ar_cross_entropy_reports_token_accuracy():
    logits = torch.zeros(1, 2, 5)
    logits[0, 0, 2] = 4.0
    logits[0, 1, 3] = 4.0
    labels = torch.tensor([[2, 3]], dtype=torch.long)

    loss, stats = qwen_ar_cross_entropy(logits, labels)

    assert loss < 0.1
    assert stats["qwen_ar_token_accuracy"] == pytest.approx(1.0)
    assert stats["qwen_ar_target_tokens"] == 2.0


def test_qwen_ar_kl_distill_masks_padding_tokens():
    student = torch.zeros(1, 2, 4)
    teacher = torch.zeros(1, 2, 4)
    teacher[0, 0, 1] = 5.0
    teacher[0, 1, 3] = 100.0
    labels = torch.tensor([[1, -100]], dtype=torch.long)

    masked = qwen_ar_kl_distill_loss(student, teacher, labels)
    unmasked = qwen_ar_kl_distill_loss(student, teacher, torch.tensor([[1, 3]]))

    assert masked < unmasked
    assert masked > 0


def test_qwen_ar_frame_distill_loss_masks_padded_frames():
    student = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    teacher = torch.zeros_like(student)
    lengths = torch.tensor([2], dtype=torch.long)

    loss, stats = qwen_ar_frame_distill_loss(student, teacher, lengths)

    expected = torch.tensor((1.0 + 9.0) / 2.0)
    torch.testing.assert_close(loss, expected)
    assert stats["qwen_context_frame_mse"] == pytest.approx(float(expected))


def test_qwen_ar_z_loss_zero_when_disabled_by_weight():
    logits = torch.randn(1, 2, 5)
    labels = torch.tensor([[1, -100]], dtype=torch.long)

    baseline = torch.tensor(3.0)
    z_loss = qwen_ar_z_loss(logits, labels)

    torch.testing.assert_close(baseline + 0.0 * z_loss, baseline)
    assert z_loss >= 0


def test_prepend_qwen_ar_audio_padding_extends_lengths_with_zero_frames():
    frames = torch.ones(2, 3, 4)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    padded, padded_lengths = prepend_qwen_ar_audio_padding(
        frames,
        lengths,
        padding_frames=2,
    )

    assert padded.shape == (2, 5, 4)
    torch.testing.assert_close(padded[:, :2, :], torch.zeros(2, 2, 4))
    torch.testing.assert_close(padded[:, 2:, :], frames)
    assert padded_lengths.tolist() == [5, 4]


def test_set_qwen_audio_left_context_sec_updates_frames_and_config():
    class AudioEncoder:
        left_context_frames = 0

    class Config:
        mel_hop_ms = 10
        qwen_audio_left_context_sec = 0.0

    class Model:
        audio_encoder = AudioEncoder()
        config = Config()

    frames = set_qwen_audio_left_context_sec(Model(), 4.0)

    assert frames == 400
    assert Model.audio_encoder.left_context_frames == 400
    assert Model.config.qwen_audio_left_context_sec == 4.0


def test_qwen_audio_preserve_loss_masks_padded_frames():
    frame_hidden = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
            [[2.0, 2.0], [100.0, 100.0], [100.0, 100.0]],
        ]
    )
    reference_hidden = torch.zeros_like(frame_hidden)
    lengths = torch.tensor([2, 1], dtype=torch.long)

    loss = qwen_audio_preserve_loss(frame_hidden, reference_hidden, lengths)

    expected = torch.tensor((1.0 + 9.0 + 4.0) / 3.0)
    torch.testing.assert_close(loss, expected)


def test_qwen_ar_streaming_audio_frames_flushes_right_context():
    class FakeAudioEncoder:
        right_context_frames = 2

        def init_state(self):
            return {
                "buffer": None,
                "seen": 0,
                "emitted": 0,
                "chunk_sizes": [],
            }

        def forward_chunk(self, chunk, state):
            state["chunk_sizes"].append(int(chunk.shape[1]))
            state["seen"] += int(chunk.shape[1])
            state["buffer"] = (
                chunk
                if state["buffer"] is None
                else torch.cat([state["buffer"], chunk], dim=1)
            )
            final_steps = max(0, state["seen"] - self.right_context_frames)
            start = int(state["emitted"])
            state["emitted"] = final_steps
            return state["buffer"][:, start:final_steps, :], state

    class FakeAdapter:
        def init_state(self):
            return {}

        def forward_chunk(self, audio_hidden, state):
            return audio_hidden + 10.0, state

    class FakeModel:
        audio_encoder = FakeAudioEncoder()
        adapter = FakeAdapter()

        class config:
            d_model = 2

    mels = torch.zeros(2, 5, 2)
    mels[0] = torch.arange(10, dtype=torch.float32).view(5, 2)
    mels[1, :3] = torch.arange(6, dtype=torch.float32).view(3, 2)
    mel_lengths = torch.tensor([5, 3], dtype=torch.long)

    frames = qwen_ar_streaming_audio_frames_for_training(
        FakeModel(),
        mels,
        mel_lengths,
        chunk_frames=2,
    )

    assert frames.shape == (2, 5, 2)
    torch.testing.assert_close(frames[0], mels[0] + 10.0)
    torch.testing.assert_close(frames[1, :3], mels[1, :3] + 10.0)
    torch.testing.assert_close(frames[1, 3:], torch.zeros(2, 2))


def test_zero_init_qwen_audio_adapter_blocks_zeros_down_projection():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down = torch.nn.Linear(3, 2, bias=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Module()
            self.adapter.blocks = torch.nn.ModuleList([Block(), Block()])

    model = Model()
    for block in model.adapter.blocks:
        torch.nn.init.ones_(block.mlp.down.weight)

    count = zero_init_qwen_audio_adapter_blocks(model)

    assert count == 2
    for block in model.adapter.blocks:
        torch.testing.assert_close(block.mlp.down.weight, torch.zeros_like(block.mlp.down.weight))


def test_compact_frame_label_tensors_use_last_nonblank_prediction_state():
    labels = torch.tensor([0, 5, 0, 7, 99, -100, 7], dtype=torch.long)
    vocab = CompactCTCVocab(token_ids=[0, 5, 7], blank_index=0)

    compact_labels, previous = compact_frame_label_tensors(
        labels,
        vocab,
        wait_token_id=0,
        word_start_token_id=99,
    )

    assert compact_labels.tolist() == [0, 1, 0, 2, 0, -100, 2]
    assert previous.tolist() == [0, 0, 1, 1, 2, 2, 2]


def test_rnnt_prefix_targets_shift_targets_and_pad_after_length():
    targets = torch.tensor([[3, 4, 5], [6, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 1], dtype=torch.long)

    prefixes = rnnt_prefix_targets(targets, lengths, blank_index=0)

    assert prefixes.tolist() == [[0, 3, 4, 5], [0, 6, 0, 0]]


def test_rnnt_forward_backward_loss_matches_single_frame_path():
    logits = torch.zeros(1, 1, 2, 3)
    logits[0, 0, 0, 1] = 2.0
    logits[0, 0, 1, 0] = 1.5
    targets = torch.tensor([[1]], dtype=torch.long)
    input_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)

    loss = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
    )

    log_probs = logits.log_softmax(dim=-1)
    expected = -(log_probs[0, 0, 0, 1] + log_probs[0, 0, 1, 0])
    torch.testing.assert_close(loss, expected)


def test_rnnt_forward_backward_loss_marginalizes_two_frame_paths():
    logits = torch.zeros(1, 2, 2, 3)
    logits[0, 0, 0, 1] = 1.0
    logits[0, 0, 1, 0] = 0.5
    logits[0, 1, 0, 1] = 0.25
    logits[0, 1, 1, 0] = 0.75
    targets = torch.tensor([[1]], dtype=torch.long)
    input_lengths = torch.tensor([2], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)

    loss = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
    )

    log_probs = logits.log_softmax(dim=-1)
    path_emit_first = (
        log_probs[0, 0, 0, 1]
        + log_probs[0, 0, 1, 0]
        + log_probs[0, 1, 1, 0]
    )
    path_emit_second = (
        log_probs[0, 0, 0, 0]
        + log_probs[0, 1, 0, 1]
        + log_probs[0, 1, 1, 0]
    )
    expected = -torch.logaddexp(path_emit_first, path_emit_second)
    torch.testing.assert_close(loss, expected)


def test_rnnt_token_frames_preserve_ctc_token_order():
    token_ids, frames = build_rnnt_token_frame_targets(
        words=[
            WordAlignment(text="hello", start_sec=0.0, end_sec=0.2),
            WordAlignment(text="world", start_sec=0.2, end_sec=0.4),
        ],
        tokenizer=TinyTokenizer(),
        duration_sec=0.5,
        wait_token_id=0,
        word_start_token_id=99,
        bos_token_id=None,
        frame_sec=0.1,
        delay_sec=0.1,
        include_word_start=False,
    )

    assert token_ids == [10, 11, 12, 13]
    assert frames == [1, 2, 3, 4]


def test_manifest_missing_alignment_fails_only_when_required(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"text": "hello", "audio": "/does/not/exist.wav"}\n',
        encoding="utf-8",
    )

    assert (
        load_manifest_examples(
            manifest_jsonl=manifest,
            tokenizer=None,
            wait_token_id=0,
            word_start_token_id=99,
            bos_token_id=None,
            config=None,
            target_delay_sec=0.1,
            max_audio_sec=1.0,
            include_word_start=False,
            require_word_alignments=False,
        )
        == []
    )
    with pytest.raises(ValueError, match="Missing word_alignments"):
        load_manifest_examples(
            manifest_jsonl=manifest,
            tokenizer=None,
            wait_token_id=0,
            word_start_token_id=99,
            bos_token_id=None,
            config=None,
            target_delay_sec=0.1,
            max_audio_sec=1.0,
            include_word_start=False,
            require_word_alignments=True,
        )


def test_rnnt_duration_prior_weight_zero_preserves_loss():
    logits = torch.randn(1, 2, 2, 3)
    targets = torch.tensor([[1]], dtype=torch.long)
    input_lengths = torch.tensor([2], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    label_frames = torch.tensor([[1]], dtype=torch.long)

    baseline = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
    )
    disabled = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
        label_frame_targets=label_frames,
        duration_prior_weight=0.0,
        duration_prior_sigma_frames=1.0,
    )

    torch.testing.assert_close(disabled, baseline)


def test_rnnt_duration_prior_penalizes_far_emission():
    logits = torch.zeros(1, 2, 2, 3)
    logits[0, 0, 0, 1] = 3.0
    logits[0, 0, 1, 0] = 3.0
    logits[0, 1, 1, 0] = 3.0
    targets = torch.tensor([[1]], dtype=torch.long)
    input_lengths = torch.tensor([2], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)

    near = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
        label_frame_targets=torch.tensor([[0]], dtype=torch.long),
        duration_prior_weight=1.0,
        duration_prior_sigma_frames=1.0,
    )
    far = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
        label_frame_targets=torch.tensor([[1]], dtype=torch.long),
        duration_prior_weight=1.0,
        duration_prior_sigma_frames=1.0,
    )

    assert far > near


def test_rnnt_nonblank_rate_loss_zero_weight_preserves_rnnt_loss():
    logits = torch.randn(1, 2, 2, 3)
    targets = torch.tensor([[1]], dtype=torch.long)
    input_lengths = torch.tensor([2], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)

    baseline = rnnt_forward_backward_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        blank_index=0,
    )
    rate_loss, _, _ = rnnt_nonblank_rate_loss(
        logits,
        input_lengths,
        target_lengths,
        blank_index=0,
    )

    torch.testing.assert_close(baseline + 0.0 * rate_loss, baseline)


def test_rnnt_nonblank_rate_target_uses_tokens_over_frames_plus_tokens():
    logits = torch.zeros(1, 3, 2, 3)
    input_lengths = torch.tensor([3], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)

    loss, pred_rate, target_rate = rnnt_nonblank_rate_loss(
        logits,
        input_lengths,
        target_lengths,
        blank_index=0,
    )

    torch.testing.assert_close(pred_rate, torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(target_rate, torch.tensor(1.0 / 4.0))
    torch.testing.assert_close(loss, torch.tensor((2.0 / 3.0 - 1.0 / 4.0) ** 2))


def test_rnnt_nonblank_rate_loss_penalizes_all_blank_attractor():
    input_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    balanced_logits = torch.zeros(1, 1, 2, 2)
    blank_heavy_logits = torch.zeros(1, 1, 2, 2)
    blank_heavy_logits[..., 0] = 6.0

    balanced_loss, _, target_rate = rnnt_nonblank_rate_loss(
        balanced_logits,
        input_lengths,
        target_lengths,
        blank_index=0,
    )
    blank_heavy_loss, pred_rate, _ = rnnt_nonblank_rate_loss(
        blank_heavy_logits,
        input_lengths,
        target_lengths,
        blank_index=0,
    )

    torch.testing.assert_close(target_rate, torch.tensor(0.5))
    assert pred_rate < target_rate
    assert blank_heavy_loss > balanced_loss


def test_rnnt_aligned_token_margin_loss_penalizes_blank_over_target():
    input_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[1]], dtype=torch.long)
    frames = torch.tensor([[0]], dtype=torch.long)
    good_logits = torch.tensor([[[[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]]])
    blank_heavy_logits = torch.tensor([[[[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])

    good_blank_loss, _, good_margin, _ = rnnt_aligned_token_margin_loss(
        good_logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        blank_margin=1.0,
    )
    bad_blank_loss, _, bad_margin, _ = rnnt_aligned_token_margin_loss(
        blank_heavy_logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        blank_margin=1.0,
    )

    assert bad_blank_loss > good_blank_loss
    assert good_margin > bad_margin


def test_rnnt_aligned_token_margin_loss_uses_best_frame_in_window():
    input_lengths = torch.tensor([2], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[1]], dtype=torch.long)
    frames = torch.tensor([[0]], dtype=torch.long)
    logits = torch.zeros(1, 2, 2, 3)
    logits[0, 0, 0, 0] = 4.0
    logits[0, 0, 0, 1] = 0.0
    logits[0, 1, 0, 0] = 0.0
    logits[0, 1, 0, 1] = 4.0

    no_window_loss, _, _, _ = rnnt_aligned_token_margin_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
    )
    window_loss, _, _, _ = rnnt_aligned_token_margin_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=1,
    )

    assert window_loss < no_window_loss


def test_rnnt_aligned_token_margin_loss_penalizes_better_other_token():
    input_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[1]], dtype=torch.long)
    frames = torch.tensor([[0]], dtype=torch.long)
    good_logits = torch.tensor([[[[0.0, 4.0, 1.0], [0.0, 0.0, 0.0]]]])
    other_heavy_logits = torch.tensor([[[[0.0, 1.0, 4.0], [0.0, 0.0, 0.0]]]])

    _, good_other_loss, _, good_margin = rnnt_aligned_token_margin_loss(
        good_logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        other_margin=0.5,
    )
    _, bad_other_loss, _, bad_margin = rnnt_aligned_token_margin_loss(
        other_heavy_logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        other_margin=0.5,
    )

    assert bad_other_loss > good_other_loss
    assert good_margin > bad_margin


def test_aligned_window_ce_prefers_target_inside_window():
    input_lengths = torch.tensor([3], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[2]], dtype=torch.long)
    frames = torch.tensor([[1]], dtype=torch.long)
    good_logits = torch.zeros(1, 3, 4)
    bad_logits = torch.zeros(1, 3, 4)
    good_logits[0, 1, 2] = 5.0
    bad_logits[0, 1, 3] = 5.0

    good_loss, good_token_loss, _, good_blank_margin, good_other_margin = (
        aligned_window_ce_loss(
            good_logits,
            targets,
            input_lengths,
            target_lengths,
            frames,
            blank_index=0,
            window_frames=0,
            blank_loss_weight=0.0,
        )
    )
    bad_loss, bad_token_loss, _, bad_blank_margin, bad_other_margin = (
        aligned_window_ce_loss(
            bad_logits,
            targets,
            input_lengths,
            target_lengths,
            frames,
            blank_index=0,
            window_frames=0,
            blank_loss_weight=0.0,
        )
    )

    assert good_loss < bad_loss
    assert good_token_loss < bad_token_loss
    assert good_blank_margin > bad_blank_margin
    assert good_other_margin > bad_other_margin


def test_aligned_window_ce_ignores_blank_loss_inside_token_window():
    input_lengths = torch.tensor([3], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[2]], dtype=torch.long)
    frames = torch.tensor([[1]], dtype=torch.long)
    logits = torch.zeros(1, 3, 4)
    logits[0, :, 3] = 5.0
    logits[0, 1, 2] = 6.0

    _, _, blank_loss, _, _ = aligned_window_ce_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=1,
        blank_loss_weight=1.0,
    )

    torch.testing.assert_close(blank_loss, torch.tensor(0.0))


def test_aligned_window_ce_penalizes_nonblank_outside_windows():
    input_lengths = torch.tensor([3], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[2]], dtype=torch.long)
    frames = torch.tensor([[1]], dtype=torch.long)
    blank_good = torch.zeros(1, 3, 4)
    blank_bad = torch.zeros(1, 3, 4)
    blank_good[0, 0, 0] = 5.0
    blank_good[0, 2, 0] = 5.0
    blank_bad[0, 0, 3] = 5.0
    blank_bad[0, 2, 3] = 5.0

    _, _, good_blank_loss, _, _ = aligned_window_ce_loss(
        blank_good,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=1.0,
    )
    _, _, bad_blank_loss, _, _ = aligned_window_ce_loss(
        blank_bad,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=1.0,
    )

    assert bad_blank_loss > good_blank_loss


def test_aligned_window_sampled_ce_prefers_target_inside_window():
    input_lengths = torch.tensor([2], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[2]], dtype=torch.long)
    frames = torch.tensor([[0]], dtype=torch.long)
    good_logits = torch.zeros(1, 2, 6)
    bad_logits = torch.zeros(1, 2, 6)
    good_logits[0, 0, 2] = 5.0
    bad_logits[0, 0, 4] = 5.0

    good_loss, _, _, _, _ = aligned_window_ce_loss(
        good_logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=0.0,
        sampled_negative_count=2,
    )
    bad_loss, _, _, _, _ = aligned_window_ce_loss(
        bad_logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=0.0,
        sampled_negative_count=2,
    )

    assert good_loss < bad_loss


def test_aligned_window_ce_applies_target_class_weight():
    input_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[2]], dtype=torch.long)
    frames = torch.tensor([[0]], dtype=torch.long)
    logits = torch.zeros(1, 1, 4)

    baseline, baseline_token, _, _, _ = aligned_window_ce_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=0.0,
    )
    weighted, weighted_token, _, _, _ = aligned_window_ce_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=0.0,
        token_class_weights=torch.tensor([1.0, 1.0, 3.0, 1.0]),
    )

    torch.testing.assert_close(weighted, baseline * 3.0)
    torch.testing.assert_close(weighted_token, baseline_token * 3.0)


def test_aligned_window_sampled_ce_includes_frequent_negatives():
    input_lengths = torch.tensor([1], dtype=torch.long)
    target_lengths = torch.tensor([1], dtype=torch.long)
    targets = torch.tensor([[2]], dtype=torch.long)
    frames = torch.tensor([[0]], dtype=torch.long)
    logits = torch.zeros(1, 1, 6)
    logits[0, 0, 3] = 5.0
    logits[0, 0, 4] = 4.0

    without_frequent, _, _, _, _ = aligned_window_ce_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=0.0,
        sampled_negative_count=1,
    )
    with_frequent, _, _, _, _ = aligned_window_ce_loss(
        logits,
        targets,
        input_lengths,
        target_lengths,
        frames,
        blank_index=0,
        window_frames=0,
        blank_loss_weight=0.0,
        sampled_negative_count=1,
        frequent_negative_indices=[4],
    )

    assert with_frequent > without_frequent


def test_rnnt_greedy_decode_emits_multiple_symbols_before_blank():
    decisions = {
        (0, 0, 0): 1,
        (0, 1, 1): 2,
        (0, 2, 2): 0,
        (1, 2, 0): 3,
        (1, 3, 1): 0,
    }

    result = rnnt_greedy_decode(
        frame_count=2,
        step_fn=lambda frame, previous, emitted: decisions[(frame, previous, emitted)],
        blank_index=0,
        max_symbols_per_frame=4,
    )

    assert result.compact_token_ids == [1, 2, 3]
    assert result.blank_count == 2
    assert result.decision_count == 5
    assert result.forced_advance_count == 0
    assert result.last_prediction_index == 3


def test_rnnt_greedy_decode_forces_frame_advance_after_limit():
    result = rnnt_greedy_decode(
        frame_count=1,
        step_fn=lambda frame, previous, emitted: previous + 1,
        blank_index=0,
        max_symbols_per_frame=3,
    )

    assert result.compact_token_ids == [1, 2, 3]
    assert result.blank_count == 0
    assert result.decision_count == 3
    assert result.forced_advance_count == 1
    assert result.last_prediction_index == 3
