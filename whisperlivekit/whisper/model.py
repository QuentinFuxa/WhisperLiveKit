import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    use_sdpa = False  # Disable SDPA to ensure qk is always computed when needed

    def __init__(self, n_state: int, n_head: int, cache_id: str = "", n_text_ctx: int = 448):
        super().__init__()
        self.n_head = n_head
        self.n_text_ctx = n_text_ctx
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.cache_id = cache_id
        # Cache IDs for key and value (used with dict-based kv_cache)
        self.key_cache_id = f"{cache_id}_key"
        self.value_cache_id = f"{cache_id}_value"
        # Keep these for backward compatibility with hook-based caching
        self.key.cache_id = self.key_cache_id
        self.value.cache_id = self.value_cache_id

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if xa is None:
            # Self-attention
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k, v = self._update_self_attn_cache(k, v, kv_cache)
        else:
            # Cross-attention: compute once and cache, or reuse from cache
            if kv_cache is not None and self.key_cache_id in kv_cache:
                k = kv_cache[self.key_cache_id]
                v = kv_cache[self.value_cache_id]
            else:
                k = self.key(xa)
                v = self.value(xa)
                if kv_cache is not None:
                    kv_cache[self.key_cache_id] = k
                    kv_cache[self.value_cache_id] = v

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def _update_self_attn_cache(
        self, k: Tensor, v: Tensor, kv_cache: dict
    ) -> Tuple[Tensor, Tensor]:
        """Update self-attention kv cache by concatenating new k,v with cached values."""
        if self.key_cache_id not in kv_cache or k.shape[1] > self.n_text_ctx:
            # First token or context overflow: save as-is
            kv_cache[self.key_cache_id] = k.detach()
            kv_cache[self.value_cache_id] = v.detach()
        else:
            # Concatenate with existing cache
            cached_k = kv_cache[self.key_cache_id]
            cached_v = kv_cache[self.value_cache_id]
            k = torch.cat([cached_k, k], dim=1).detach()
            v = torch.cat([cached_v, v], dim=1).detach()
            kv_cache[self.key_cache_id] = k
            kv_cache[self.value_cache_id] = v
        return k, v

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, n_state: int, n_head: int, cross_attention: bool = False, 
        cache_id: str = "", n_text_ctx: int = 448
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            n_state, n_head, cache_id=f"{cache_id}_self_attn", n_text_ctx=n_text_ctx
        )
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(
                n_state, n_head, cache_id=f"{cache_id}_cross_attn", n_text_ctx=n_text_ctx
            ) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            x: The output tensor
            cross_attn_qk: Cross-attention weights (if cross_attn exists), else None
        """
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        cross_attn_qk = None
        if self.cross_attn:
            cross_out, cross_attn_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, kv_cache=kv_cache
            )
            x = x + cross_out
        x = x + self.mlp(self.mlp_ln(x))
        return x, cross_attn_qk


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cache_id=f"enc_layer{i}") for i in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x, _ = block(x)  # Encoder blocks don't have cross-attention

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.n_ctx = n_ctx

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_state, n_head, cross_attention=True, 
                    cache_id=f"dec_layer{i}", n_text_ctx=n_ctx
                )
                for i in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self, 
        x: Tensor, 
        xa: Tensor, 
        kv_cache: Optional[dict] = None,
        return_cross_attn: bool = False,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        kv_cache : Optional[dict]
            Dictionary to store/retrieve key-value cache for efficient decoding
        return_cross_attn : bool
            If True, return cross-attention weights from all decoder layers
            
        Returns
        -------
        logits : Tensor
            The output logits
        cross_attns : Optional[List[Tensor]]
            List of cross-attention weights per layer (only if return_cross_attn=True)
        """
        # Calculate offset from self-attention cache (not cross-attention which has audio length)
        offset = 0
        if kv_cache:
            # Use the first decoder block's self-attention key cache to get token position
            first_self_attn_key = self.blocks[0].attn.key_cache_id
            if first_self_attn_key in kv_cache:
                offset = kv_cache[first_self_attn_key].shape[1]
        
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        cross_attns = [] if return_cross_attn else None
        for block in self.blocks:
            x, cross_attn_qk = block(x, xa, mask=self.mask, kv_cache=kv_cache)
            if return_cross_attn and cross_attn_qk is not None:
                cross_attns.append(cross_attn_qk)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        if return_cross_attn:
            return logits, cross_attns
        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, decoder_only: bool = False):
        super().__init__()
        self.dims = dims
        
        if not decoder_only:
            self.encoder = AudioEncoder(
                self.dims.n_mels,
                self.dims.n_audio_ctx,
                self.dims.n_audio_state,
                self.dims.n_audio_head,
                self.dims.n_audio_layer,
            )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(
        self, 
        tokens: torch.Tensor, 
        audio_features: torch.Tensor,
        kv_cache: Optional[dict] = None,
        return_cross_attn: bool = False,
    ):
        return self.decoder(
            tokens, audio_features, 
            kv_cache=kv_cache, 
            return_cross_attn=return_cross_attn
        )

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
