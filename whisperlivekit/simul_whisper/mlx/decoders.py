"""
MLX-native token decoders for streaming ASR.
"""
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np


class MLXGreedyDecoder:
    """Greedy decoder using MLX operations."""
    
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool]:
        """
        Update tokens with next predicted token.
        
        Args:
            tokens: Current token sequence, shape (batch, seq_len)
            logits: Logits for next token, shape (batch, vocab_size)
            sum_logprobs: Cumulative log probabilities, shape (batch,)
            
        Returns:
            Updated tokens and completion flag
        """
        if self.temperature == 0:
            next_tokens = mx.argmax(logits, axis=-1)
        else:
            probs = mx.softmax(logits / self.temperature, axis=-1)
            next_tokens = mx.random.categorical(mx.log(probs + 1e-10))
        
        logprobs = mx.softmax(logits, axis=-1)
        logprobs = mx.log(logprobs + 1e-10)        
        batch_size = logprobs.shape[0]
        current_logprobs = logprobs[mx.arange(batch_size), next_tokens]        
        mask = (tokens[:, -1] != self.eot).astype(mx.float32)
        sum_logprobs = sum_logprobs + current_logprobs * mask        
        eot_mask = (tokens[:, -1] == self.eot)
        next_tokens = mx.where(eot_mask, mx.array(self.eot), next_tokens)        
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=1)        
        completed = bool(mx.all(tokens[:, -1] == self.eot))
        
        return tokens, completed

    def finalize(self, tokens: mx.array, sum_logprobs: mx.array):
        """Finalize decoding by ensuring EOT at end."""
        eot_column = mx.full((tokens.shape[0], 1), self.eot, dtype=tokens.dtype)
        tokens = mx.concatenate([tokens, eot_column], axis=1)
        return tokens, sum_logprobs.tolist()


class MLXBeamSearchDecoder:
    """Beam search decoder using MLX operations."""
    
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Any,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences: Optional[List[Dict]] = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        """Reset finished sequences for new segment."""
        self.finished_sequences = None

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool]:
        """
        Update tokens using beam search.
        
        Args:
            tokens: Current token sequences, shape (batch * beam_size, seq_len)
            logits: Logits for next token, shape (batch * beam_size, vocab_size)
            sum_logprobs: Cumulative log probabilities, shape (batch * beam_size,)
            
        Returns:
            Updated tokens and completion flag
        """
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]
        logprobs = mx.softmax(logits, axis=-1)
        logprobs = mx.log(logprobs + 1e-10)        
        logprobs_np = np.array(logprobs)
        tokens_np = np.array(tokens)
        sum_logprobs_np = np.array(sum_logprobs)
        
        next_tokens, source_indices, finished_sequences = [], [], []
        new_sum_logprobs = []
        
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens_np[idx].tolist()                
                top_k_indices = np.argsort(logprobs_np[idx])[-self.beam_size - 1:][::-1]
                
                for token_idx in top_k_indices:
                    logprob = logprobs_np[idx, token_idx]
                    new_logprob = sum_logprobs_np[idx] + logprob
                    sequence = tuple(prefix + [int(token_idx)])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    new_sum_logprobs.append(scores[sequence])
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)
        tokens = mx.array(np.array(next_tokens, dtype=np.int32))
        sum_logprobs = mx.array(np.array(new_sum_logprobs, dtype=np.float32))        
        self.inference.rearrange_kv_cache(source_indices)
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break
                previously_finished[seq] = newly_finished[seq]
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        
        return tokens, completed

    def finalize(self, preceding_tokens: mx.array, sum_logprobs: mx.array):
        """Finalize beam search by selecting best sequences."""
        preceding_tokens_np = np.array(preceding_tokens)
        sum_logprobs_np = np.array(sum_logprobs)
        
        n_audio = preceding_tokens_np.shape[0] // self.beam_size
        tokens_list: List[List[int]] = [[] for _ in range(n_audio)]
        sum_logprobs_list: List[float] = [0.0] * n_audio

        for i, sequences in enumerate(self.finished_sequences):
            if sequences:
                best_seq = max(sequences, key=sequences.get)
                tokens_list[i] = list(best_seq)
                sum_logprobs_list[i] = sequences[best_seq]
            else:
                idx = i * self.beam_size
                tokens_list[i] = preceding_tokens_np[idx].tolist() + [self.eot]
                sum_logprobs_list[i] = float(sum_logprobs_np[idx])
        max_len = max(len(t) for t in tokens_list)
        for i, t in enumerate(tokens_list):
            tokens_list[i] = t + [self.eot] * (max_len - len(t))

        tokens = mx.array(np.array(tokens_list, dtype=np.int32))
        return tokens, sum_logprobs_list


class MLXInference:
    """MLX inference wrapper for beam search KV cache management."""
    
    def __init__(self, model, initial_token_length: int):
        self.model = model
        self.initial_token_length = initial_token_length
        self.kv_cache = None
    
    def rearrange_kv_cache(self, source_indices: List[int]):
        """Rearrange KV cache based on beam search source indices."""
        if self.kv_cache is None:
            return
            
        if source_indices == list(range(len(source_indices))):
            return
        
        source_indices_mx = mx.array(source_indices, dtype=mx.int32)
        
        new_cache = []
        for layer_cache in self.kv_cache:
            (k, v), (cross_k, cross_v) = layer_cache            
            new_k = k[source_indices_mx]
            new_v = v[source_indices_mx]
            new_cache.append(((new_k, new_v), (cross_k, cross_v)))
        
        self.kv_cache = new_cache
    
    def logits(
        self, 
        tokens: mx.array, 
        audio_features: mx.array,
    ) -> Tuple[mx.array, List]:
        """Get logits from decoder with KV cache."""
        logits, self.kv_cache, cross_qk = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        return logits, cross_qk

