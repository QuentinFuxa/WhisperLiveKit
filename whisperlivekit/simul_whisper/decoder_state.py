from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DecoderState:
    """
    Container for mutable decoder state.

    We start by moving kv_cache and per-session tokenizer/language flags
    here so they can be owned by the session rather than the decoder
    modules themselves.
    """
    kv_cache: Dict[Any, Any] = field(default_factory=dict)
    tokenizer: Optional[Any] = None
    tokenizer_is_multilingual: bool = True
    detected_language: Optional[str] = None
    reset_tokenizer_to_auto_next_call: bool = False
    token_decoder: Optional[Any] = None
    inference: Optional[Any] = None
    speaker: int = -1
    log_segments: int = 0
    max_context_tokens: Optional[int] = None
    max_text_len: Optional[int] = None
    align_source: Dict[int, list] = field(default_factory=dict)
    num_align_heads: int = 0
    cif_linear: Optional[Any] = None
    always_fire: bool = False
    never_fire: bool = False
    last_attend_frame: float = 0.0
    cumulative_time_offset: float = 0.0
    first_timestamp: Optional[float] = None
    global_time_offset: float = 0.0
    segments: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    initial_tokens: Optional[Any] = None
    context: Optional[Any] = None
    pending_incomplete_tokens: list = field(default_factory=list)
    initial_token_length: Optional[int] = None
    sot_index: Optional[int] = None

    def reset_kv_cache(self) -> None:
        # Preserve the dict reference (used by inference) and clear content.
        self.kv_cache.clear()

    def reset_session(self) -> None:
        """Clear all mutable session state (except hook handles)."""
        self.reset_kv_cache()

        self.segments.clear()
        self.tokens.clear()
        self.pending_incomplete_tokens.clear()
        self.context = None
        self.initial_tokens = None
        self.initial_token_length = None
        self.sot_index = None

        self.last_attend_frame = 0.0
        self.cumulative_time_offset = 0.0
        self.first_timestamp = None
        self.global_time_offset = 0.0
