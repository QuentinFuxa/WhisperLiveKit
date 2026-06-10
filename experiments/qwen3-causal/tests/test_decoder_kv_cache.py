"""KV-cached decode parity tests.

`FakeCachingQwenTextModel` produces history-dependent hidden states (causal
cumulative mean), so any cache mis-wiring (dropped history, reset positions)
changes the greedy sequence. Parity = cached and uncached paths must produce
identical tokens and logits.
"""

import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.native_realtime_model import (  # noqa: E402
    Qwen3ASRRealtimeQwenAudioCausalModel,
)
from qwen3_streaming.realtime_config import RealtimeAudioConfig  # noqa: E402
from test_mutable_tail import D_MODEL, N_MELS, TinyQwenAudioTower, tiny_config  # noqa: E402

VOCAB = 16


class FakeCache:
    def __init__(self):
        self.running_sum = None  # [B, D]
        self.count = 0

    def get_seq_length(self):
        return self.count


class _Out:
    def __init__(self, last_hidden_state, past_key_values=None):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class FakeCachingQwenTextModel(torch.nn.Module):
    """hidden[t] = mean(inputs[0..t]) — strictly history-dependent."""

    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = D_MODEL):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList()
        self.norm = torch.nn.Identity()
        self.forwarded_positions = 0
        self.calls = 0

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **_kwargs):
        self.calls += 1
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


def make_model(text_model) -> Qwen3ASRRealtimeQwenAudioCausalModel:
    torch.manual_seed(0)
    return Qwen3ASRRealtimeQwenAudioCausalModel(
        tiny_config(),
        qwen_model_id="fake",
        audio_tower=TinyQwenAudioTower(),
        text_model=text_model,
        lm_head=torch.nn.Linear(D_MODEL, VOCAB, bias=False),
        bos_token_id=1,
        wait_token_id=0,
    ).eval()


def run_generate(model, frame_hidden, *, use_cache, **kwargs):
    torch.manual_seed(1)
    return model.generate_full_hypothesis_from_cached_audio(
        frame_hidden,
        use_decoder_kv_cache=use_cache,
        **kwargs,
    )


@pytest.mark.parametrize("batch", [1, 3])
def test_cached_matches_uncached_plain(batch):
    model = make_model(FakeCachingQwenTextModel())
    torch.manual_seed(2)
    frame_hidden = torch.randn(batch, 5, D_MODEL)
    a = run_generate(model, frame_hidden, use_cache=False, max_new_tokens=8)
    b = run_generate(model, frame_hidden, use_cache=True, max_new_tokens=8)
    assert torch.equal(a, b)


def test_cached_matches_uncached_with_controls_and_stop():
    model = make_model(FakeCachingQwenTextModel())
    torch.manual_seed(3)
    frame_hidden = torch.randn(2, 4, D_MODEL)
    kwargs = dict(
        max_new_tokens=10,
        eos_token_id=5,
        stop_token_ids=[6],
        suppress_token_ids=[2, 3],
        repetition_penalty=1.3,
        no_repeat_ngram_size=2,
        max_consecutive_text_tokens=3,
    )
    a = run_generate(model, frame_hidden, use_cache=False, **kwargs)
    b = run_generate(model, frame_hidden, use_cache=True, **kwargs)
    assert torch.equal(a, b)


def test_cached_matches_uncached_with_prefix_placeholders():
    model = make_model(FakeCachingQwenTextModel())
    torch.manual_seed(4)
    frame_hidden = torch.randn(1, 3, D_MODEL)
    kwargs = dict(
        prefix_token_ids=[10, 7, 7, 7, 11, 12],
        audio_placeholder_token_id=7,
        max_new_tokens=6,
        eos_token_id=5,
    )
    a = run_generate(model, frame_hidden, use_cache=False, **kwargs)
    b = run_generate(model, frame_hidden, use_cache=True, **kwargs)
    assert torch.equal(a, b)


def test_cached_matches_uncached_with_prompt_tokens():
    model = make_model(FakeCachingQwenTextModel())
    torch.manual_seed(5)
    frame_hidden = torch.randn(1, 4, D_MODEL)
    kwargs = dict(prompt_token_ids=[9, 8], max_new_tokens=5)
    a = run_generate(model, frame_hidden, use_cache=False, **kwargs)
    b = run_generate(model, frame_hidden, use_cache=True, **kwargs)
    assert torch.equal(a, b)


def test_degenerate_inputs_match():
    model = make_model(FakeCachingQwenTextModel())
    frame_hidden = torch.randn(1, 0, D_MODEL)
    empty_prompt = torch.empty(1, 0, dtype=torch.long)
    a = run_generate(
        model, frame_hidden, use_cache=False, prompt_token_ids=empty_prompt, max_new_tokens=4
    )
    b = run_generate(
        model, frame_hidden, use_cache=True, prompt_token_ids=empty_prompt, max_new_tokens=4
    )
    assert a.shape == b.shape == (1, 0)
    z = run_generate(model, torch.randn(1, 3, D_MODEL), use_cache=True, max_new_tokens=0)
    assert z.shape == (1, 0)


def test_falls_back_when_model_returns_no_cache():
    class NoCacheModel(FakeCachingQwenTextModel):
        def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **kw):
            out = super().forward(inputs_embeds, past_key_values, use_cache=False)
            out.past_key_values = None
            return out

    model = make_model(NoCacheModel())
    torch.manual_seed(6)
    frame_hidden = torch.randn(1, 4, D_MODEL)
    tokens = run_generate(model, frame_hidden, use_cache=True, max_new_tokens=5)
    assert tokens.shape[1] == 5  # legacy path produced a full hypothesis


def test_cached_path_forwards_far_fewer_positions():
    P, T = 24, 12
    uncached_model = make_model(FakeCachingQwenTextModel())
    torch.manual_seed(7)
    frame_hidden = torch.randn(1, P, D_MODEL)
    run_generate(uncached_model, frame_hidden, use_cache=False, max_new_tokens=T)
    uncached_positions = uncached_model.text_model.forwarded_positions

    cached_model = make_model(FakeCachingQwenTextModel())
    torch.manual_seed(7)
    frame_hidden = torch.randn(1, P, D_MODEL)
    run_generate(cached_model, frame_hidden, use_cache=True, max_new_tokens=T)
    cached_positions = cached_model.text_model.forwarded_positions

    # audio P + bos 1 prefilled, then T-1 incremental tokens
    assert cached_positions == P + 1 + (T - 1)
    assert uncached_positions == sum(P + 1 + k for k in range(T))
    assert cached_positions < uncached_positions / 4
