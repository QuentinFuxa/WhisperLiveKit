"""KV-cached decode parity for the promoted WLK model (twin of experiments tests)."""

import pytest

torch = pytest.importorskip("torch")

from whisperlivekit.qwen3_streaming.model import (  # noqa: E402
    Qwen3ASRRealtimeQwenAudioSurgeryModel,
)
from whisperlivekit.qwen3_streaming.model_config import RealtimeAudioConfig  # noqa: E402

D_MODEL = 32
VOCAB = 16


class FakeCache:
    def __init__(self):
        self.running_sum = None
        self.count = 0

    def get_seq_length(self):
        return self.count


class _Out:
    def __init__(self, last_hidden_state, past_key_values=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class FakeCachingQwenTextModel(torch.nn.Module):
    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = D_MODEL):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList()
        self.norm = torch.nn.Identity()
        self.forwarded_positions = 0

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **_kwargs):
        self.forwarded_positions += int(inputs_embeds.shape[1])
        if past_key_values is None:
            cache = FakeCache() if use_cache else None
            running = torch.zeros(
                inputs_embeds.shape[0], inputs_embeds.shape[2],
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )
            count = 0
        else:
            cache = past_key_values
            running = cache.running_sum.clone()
            count = cache.count
        hidden = []
        for t in range(inputs_embeds.shape[1]):
            running = running + inputs_embeds[:, t, :]
            count += 1
            hidden.append(running / count)
        out = torch.stack(hidden, dim=1)
        if cache is not None:
            cache.running_sum = running
            cache.count = count
        return _Out(out, past_key_values=cache if use_cache else None)


class FakeAudioTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(128, D_MODEL, bias=False)

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths // 8


def make_model():
    torch.manual_seed(0)
    return Qwen3ASRRealtimeQwenAudioSurgeryModel(
        RealtimeAudioConfig(d_model=D_MODEL),
        qwen_model_id="fake",
        audio_tower=FakeAudioTower(),
        text_model=FakeCachingQwenTextModel(),
        lm_head=torch.nn.Linear(D_MODEL, VOCAB, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()


def test_cached_matches_uncached_with_controls():
    model = make_model()
    torch.manual_seed(2)
    frame_hidden = torch.randn(2, 5, D_MODEL)
    kwargs = dict(
        max_new_tokens=10,
        eos_token_id=5,
        suppress_token_ids=[2],
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
    )
    a = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden, use_decoder_kv_cache=False, **kwargs
    )
    b = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden, use_decoder_kv_cache=True, **kwargs
    )
    assert torch.equal(a, b)


def test_cached_matches_uncached_with_prefix():
    model = make_model()
    torch.manual_seed(3)
    frame_hidden = torch.randn(1, 3, D_MODEL)
    kwargs = dict(
        prefix_token_ids=[10, 7, 7, 7, 11],
        audio_placeholder_token_id=7,
        max_new_tokens=6,
        eos_token_id=5,
    )
    a = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden, use_decoder_kv_cache=False, **kwargs
    )
    b = model.generate_full_hypothesis_from_cached_audio(
        frame_hidden, use_decoder_kv_cache=True, **kwargs
    )
    assert torch.equal(a, b)


def test_cached_path_is_cheaper():
    model = make_model()
    torch.manual_seed(4)
    frame_hidden = torch.randn(1, 24, D_MODEL)
    model.generate_full_hypothesis_from_cached_audio(
        frame_hidden, use_decoder_kv_cache=False, max_new_tokens=12
    )
    uncached = model.text_model.forwarded_positions
    model.text_model.forwarded_positions = 0
    model.generate_full_hypothesis_from_cached_audio(
        frame_hidden, use_decoder_kv_cache=True, max_new_tokens=12
    )
    cached = model.text_model.forwarded_positions
    assert cached == 24 + 1 + 11
    assert cached < uncached / 4
