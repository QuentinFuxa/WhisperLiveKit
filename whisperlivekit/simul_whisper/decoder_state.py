from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class DecoderState:

    kv_cache: Dict[str, torch.Tensor] = field(default_factory=dict)

    tokenizer: Any = None
    detected_language: Optional[str] = None
    reset_tokenizer_to_auto_next_call: bool = False

    tokens: List[torch.Tensor] = field(default_factory=list)
    initial_tokens: Optional[torch.Tensor] = None
    initial_token_length: int = 0
    sot_index: int = 0

    align_source: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    num_align_heads: int = 0

    segments: List[torch.Tensor] = field(default_factory=list)

    context: Any = None

    pending_incomplete_tokens: List[int] = field(default_factory=list)
    pending_incomplete_token_timestamps: List[float] = field(default_factory=list)
    pending_retries: int = 0

    global_time_offset: float = 0.0
    cumulative_time_offset: float = 0.0
    first_timestamp: Optional[float] = None
    last_attend_frame: int = 0

    speaker: int = -1
    log_segments: int = 0

    CIFLinear: Optional[torch.nn.Module] = None
    always_fire: bool = False
    never_fire: bool = False

    suppress_tokens_fn: Any = None

    token_decoder: Any = None
    decoder_type: str = "greedy"

    inference: Any = None

    def clean_cache(self):
        """Drop per-step kv_cache references without synchronizing CUDA."""
        self.kv_cache.clear()

        if self.decoder_type == "beam" and self.inference is not None:
            # Create NEW dict instead of sharing reference
            self.inference.kv_cache = {}
            if self.token_decoder is not None:
                self.token_decoder.reset()

    def release_gpu_memory(self):
        """Return unused CUDA allocator blocks to the driver at coarse boundaries."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self, rewind_threshold: int = 200):
        """
        Reset transient state for a new segment.

        Args:
            rewind_threshold: Value for resetting last_attend_frame
        """
        self.last_attend_frame = -rewind_threshold
        self.cumulative_time_offset = 0.0
        self.pending_incomplete_tokens = []
        self.pending_incomplete_token_timestamps = []
        self.pending_retries = 0
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
        self.kv_cache = {}
        self.first_timestamp = None
