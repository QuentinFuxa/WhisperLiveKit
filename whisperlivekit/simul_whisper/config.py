from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AlignAttConfig():
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 200
    audio_max_len: float = 20.0
    cif_ckpt_path: str = ""
    never_fire: bool = False
    language: str = field(default="zh")
    nonspeech_prob: float = 0.5
    audio_min_len: float = 1.0
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 5
    task: Literal["transcribe","translate"] = "transcribe"
    tokenizer_is_multilingual: bool = False
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)
    