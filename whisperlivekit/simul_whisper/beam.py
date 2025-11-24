from typing import Optional

from whisperlivekit.whisper.decoding import PyTorchInference

from .decoder_state import DecoderState


# extention of PyTorchInference for beam search
class BeamPyTorchInference(PyTorchInference):

    def __init__(
        self,
        model,
        initial_token_length: int,
        decoder_state: Optional[DecoderState] = None,
    ):
        super().__init__(model, initial_token_length)
        self.decoder_state = decoder_state
        if decoder_state is not None:
            self.kv_cache = decoder_state.kv_cache

    def attach_state(self, decoder_state: DecoderState):
        """Attach a DecoderState so kv_cache references stay in sync."""
        self.decoder_state = decoder_state
        self.kv_cache = decoder_state.kv_cache

    def _kv_modules(self):
        key_modules = [block.attn.key.cache_id for block in self.model.decoder.blocks]
        value_modules = [block.attn.value.cache_id for block in self.model.decoder.blocks]
        return key_modules + value_modules

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module_cache_id in self._kv_modules():
                self.kv_cache[module_cache_id] = self.kv_cache[module_cache_id][source_indices].detach()
    from torch import Tensor
    def logits(self, tokens: Tensor, audio_features: Tensor, return_attn: bool = False) -> Tensor:
        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache, return_attn=return_attn)
