from whisperlivekit.whisper.decoding import PyTorchInference

# extention of PyTorchInference for beam search
class BeamPyTorchInference(PyTorchInference):

    def _kv_modules(self):
        key_modules = [block.attn.key.cache_id for block in self.model.decoder.blocks]
        value_modules = [block.attn.value.cache_id for block in self.model.decoder.blocks]
        return key_modules + value_modules

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module_cache_id in self._kv_modules():
                self.kv_cache[module_cache_id] = self.kv_cache[module_cache_id][source_indices].detach()
    from torch import Tensor
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)