from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np


@dataclass
class MLXDecoderState:
    """
    mlx kv cache format: List of ((k, v), (cross_k, cross_v)) tuples per layer,
    where each element is a tuple of mx.arrays.
    """

    kv_cache: Optional[List[Tuple[Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]]] = None
    
    tokenizer: Any = None
    detected_language: Optional[str] = None
    reset_tokenizer_to_auto_next_call: bool = False
    
    tokens: List[mx.array] = field(default_factory=list)
    initial_tokens: Optional[mx.array] = None
    initial_token_length: int = 0
    sot_index: int = 0    
    align_source: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    num_align_heads: int = 0    
    segments: List[np.ndarray] = field(default_factory=list)
    
    context: Any = None
    
    pending_incomplete_tokens: List[int] = field(default_factory=list)
    
    global_time_offset: float = 0.0
    cumulative_time_offset: float = 0.0
    first_timestamp: Optional[float] = None
    last_attend_frame: int = 0
    
    speaker: int = -1
    log_segments: int = 0    
    cif_weights: Optional[mx.array] = None
    always_fire: bool = False
    never_fire: bool = False
    
    suppress_tokens: Optional[Tuple[int, ...]] = None
    
    token_decoder: Any = None
    decoder_type: str = "greedy"
    
    inference: Any = None
    
    def clean_cache(self):
        self.kv_cache = None
        if self.decoder_type == "beam" and self.inference is not None:
            self.inference.kv_cache = None
            if self.token_decoder is not None:
                self.token_decoder.reset()
    
    def reset(self, rewind_threshold: int = 200):
        self.last_attend_frame = -rewind_threshold
        self.cumulative_time_offset = 0.0
        self.pending_incomplete_tokens = []
        self.log_segments += 1
    
    def full_reset(self, rewind_threshold: int = 200):
        """
        Full reset including audio segments and tokens.
        
        Args:
            rewind_threshold: Value for resetting last_attend_frame
        """
        self.reset(rewind_threshold)
        self.segments = []
        self.tokens = []
        self.kv_cache = None
        self.first_timestamp = None

