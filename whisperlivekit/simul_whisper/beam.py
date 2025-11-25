from torch import Tensor

from whisperlivekit.whisper.decoding import PyTorchInference


class BeamPyTorchInference(PyTorchInference):
    """Extension of PyTorchInference for beam search with cross-attention support."""

    def _kv_cache_ids(self):
        """Get cache_id strings for self-attention key/value modules."""
        key_ids = [block.attn.key_cache_id for block in self.model.decoder.blocks]
        value_ids = [block.attn.value_cache_id for block in self.model.decoder.blocks]
        return key_ids + value_ids

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for cache_id in self._kv_cache_ids():
                if cache_id in self.kv_cache:
                    self.kv_cache[cache_id] = self.kv_cache[cache_id][source_indices].detach()

    def logits(
        self, 
        tokens: Tensor, 
        audio_features: Tensor,
        return_cross_attn: bool = False,
    ):
        """Get logits, optionally returning cross-attention weights."""
        return self.model.decoder(
            tokens, audio_features, 
            kv_cache=self.kv_cache,
            return_cross_attn=return_cross_attn,
        )