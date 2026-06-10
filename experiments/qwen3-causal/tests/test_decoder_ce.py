"""D2 objective tests: LoRA no-op init, CE label layout, gradient isolation."""

import pytest

torch = pytest.importorskip("torch")

from qwen3_streaming.decoder_ce import build_ce_inputs, ce_forward  # noqa: E402
from qwen3_streaming.lora import (  # noqa: E402
    DECODER_LORA_TARGETS,
    add_lora_to_linear_modules,
    lora_parameters,
    lora_state_dict,
)

D_MODEL = 32


class FakeTokenizer:
    eos_token_id = 5

    def encode(self, text, add_special_tokens=False):
        if "<|audio_pad|>" in text:
            # prompt template: [system, audio_start, AUDIO, audio_end, lang]
            return [10, 11, 7, 12, 13]
        return [20 + (ord(c) % 4) for c in text.replace(" ", "")][:8]

    def convert_tokens_to_ids(self, token):
        return 7 if token == "<|audio_pad|>" else -1


class FakeTextModel(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden=D_MODEL):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden)
        self.layers = torch.nn.ModuleList(
            [torch.nn.ModuleDict({
                "q_proj": torch.nn.Linear(hidden, hidden),
                "down_proj": torch.nn.Linear(hidden, hidden),
            }) for _ in range(2)]
        )
        self.norm = torch.nn.Identity()

    def forward(self, inputs_embeds, **_kwargs):
        h = inputs_embeds
        for layer in self.layers:
            h = h + layer["down_proj"](torch.tanh(layer["q_proj"](h)))

        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return Output(h)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = FakeTextModel()
        self.embed_tokens = self.text_model.embed_tokens
        self.lm_head = torch.nn.Linear(D_MODEL, 32, bias=False)


def test_build_ce_inputs_label_layout():
    tok = FakeTokenizer()
    prompt_ids, target_ids, labels = build_ce_inputs(
        tok,
        audio_steps=3,
        language="English",
        target_text="hello world",
        audio_placeholder_token_id=7,
    )
    assert prompt_ids.count(7) == 3  # placeholder expanded to audio steps
    assert target_ids[-1] == tok.eos_token_id
    assert labels[: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids) :] == target_ids


def test_lora_zero_init_is_noop_and_isolated():
    torch.manual_seed(0)
    model = FakeModel().eval()
    x = torch.randn(1, 4, D_MODEL)
    target = [1, 2, 3]
    prompt = [10, 7, 7, 12]

    with torch.no_grad():
        loss_before, _ = ce_forward(
            model, x[:, :2, :], prompt_ids=prompt, target_ids=target,
            audio_placeholder_token_id=7,
        )
    wrapped = add_lora_to_linear_modules(
        model.text_model, target_names=DECODER_LORA_TARGETS, rank=4, alpha=8.0
    )
    assert wrapped, "no decoder linears wrapped"
    with torch.no_grad():
        loss_after, _ = ce_forward(
            model, x[:, :2, :], prompt_ids=prompt, target_ids=target,
            audio_placeholder_token_id=7,
        )
    torch.testing.assert_close(loss_after, loss_before)  # zero-init no-op

    # Gradients flow into LoRA params only.
    for param in model.parameters():
        param.requires_grad_(False)
    for param in lora_parameters(model.text_model):
        param.requires_grad_(True)
    loss, stats = ce_forward(
        model, x[:, :2, :], prompt_ids=prompt, target_ids=target,
        audio_placeholder_token_id=7,
    )
    loss.backward()
    lora_grads = [p.grad for p in lora_parameters(model.text_model)]
    assert any(g is not None and g.abs().sum() > 0 for g in lora_grads)
    assert model.lm_head.weight.grad is None
    assert 0.0 <= stats["token_accuracy"] <= 1.0

    sd = lora_state_dict(model.text_model)
    assert sd and all(".lora_" in k for k in sd)
