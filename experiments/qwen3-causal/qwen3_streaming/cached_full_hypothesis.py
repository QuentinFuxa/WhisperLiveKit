from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .realtime import post_process_realtime_text
from .stable_commit import (
    StablePrefixCommitState,
    StableTextCommitState,
    update_stable_prefix_commit,
    update_stable_text_commit,
)


def trim_at_stop(tokens: list[int], stop_token_id: int | None) -> list[int]:
    if stop_token_id is None:
        return tokens
    try:
        stop_idx = tokens.index(int(stop_token_id))
    except ValueError:
        return tokens
    return tokens[:stop_idx]


def qwen_asr_prompt_text(*, context: str = "", language: str | None = None) -> str:
    prompt = (
        f"<|im_start|>system\n{context or ''}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language:
        prompt += f"language {language}<asr_text>"
    return prompt


def expand_audio_prompt_placeholders(
    token_ids: Sequence[int],
    *,
    audio_placeholder_token_id: int,
    audio_steps: int,
) -> list[int]:
    expanded: list[int] = []
    replaced = False
    for token_id in token_ids:
        if int(token_id) == int(audio_placeholder_token_id):
            expanded.extend([int(audio_placeholder_token_id)] * int(audio_steps))
            replaced = True
        else:
            expanded.append(int(token_id))
    if not replaced:
        raise ValueError("prompt does not contain the Qwen ASR audio placeholder token")
    return expanded


def added_token_id(tokenizer, token: str, default: int = -1) -> int:
    if token not in tokenizer.get_added_vocab():
        return int(default)
    token_id = tokenizer.convert_tokens_to_ids(token)
    return int(token_id) if isinstance(token_id, int) else int(default)


def decode_clean_token_ids(
    tokenizer,
    token_ids: Sequence[int],
    *,
    wait_token_id: int,
    word_start_token_id: int,
) -> str:
    ignored = {int(wait_token_id), int(word_start_token_id), -100}
    filtered = [int(token_id) for token_id in token_ids if int(token_id) not in ignored]
    if not filtered:
        return ""
    return post_process_realtime_text(
        tokenizer.decode(filtered, skip_special_tokens=True)
    )


def join_text_segments(*segments: str) -> str:
    kept = [segment.strip() for segment in segments if segment and segment.strip()]
    return " ".join(kept).strip()


def trailing_text_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = (text or "").split()
    return " ".join(words[-max_words:])


def _tensor_to_int_list(values: Any) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


@dataclass(frozen=True)
class CachedFullHypothesisConfig:
    wait_token_id: int
    word_start_token_id: int
    eos_token_id: int | None = None
    max_new_tokens: int = 256
    hold_back_tokens: int = 4
    hold_back_words: int = 6
    stable_iterations: int = 2
    min_commit_audio_sec: float = 0.0
    commit_mode: str = "word"
    normalize_commit_match: bool = False
    suppress_token_ids: tuple[int, ...] = ()
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    max_consecutive_text_tokens: int = 0
    prompt_token_ids: Sequence[int] | None = None
    prompt_prefix_template: Sequence[int] | None = None
    audio_placeholder_token_id: int | None = None
    use_decoder_kv_cache: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if self.hold_back_tokens < 0 or self.hold_back_words < 0:
            raise ValueError("hold-back values must be >= 0")
        if self.stable_iterations <= 0:
            raise ValueError("stable_iterations must be > 0")
        if self.min_commit_audio_sec < 0.0:
            raise ValueError("min_commit_audio_sec must be >= 0")
        if self.commit_mode not in {"word", "token"}:
            raise ValueError("commit_mode must be 'word' or 'token'")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be >= 0")
        if self.max_consecutive_text_tokens < 0:
            raise ValueError("max_consecutive_text_tokens must be >= 0")


@dataclass(frozen=True)
class CachedFullHypothesisFinal:
    final_tokens: list[int]
    final_text: str
    final_display_text: str
    stable_committed_text: str
    last_hypothesis_text: str
    final_committed_units: int


@dataclass
class CachedFullHypothesisStreamer:
    model: Any
    tokenizer: Any
    config: CachedFullHypothesisConfig
    state: Any = None
    token_commit_state: StablePrefixCommitState = field(
        default_factory=StablePrefixCommitState
    )
    text_commit_state: StableTextCommitState = field(
        default_factory=StableTextCommitState
    )
    events: list[dict[str, Any]] = field(default_factory=list)
    last_hypothesis_tokens: list[int] = field(default_factory=list)
    last_hypothesis_text: str = ""
    last_display_text: str = ""
    last_committed_text: str = ""

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = self.model.init_cached_audio_decode_state()

    def prompt_prefix_token_ids(self, *, audio_steps: int) -> list[int] | None:
        if self.config.prompt_prefix_template is None:
            return None
        if self.config.audio_placeholder_token_id is None:
            raise ValueError("missing audio placeholder token id")
        return expand_audio_prompt_placeholders(
            self.config.prompt_prefix_template,
            audio_placeholder_token_id=self.config.audio_placeholder_token_id,
            audio_steps=int(audio_steps),
        )

    def append_mel_chunk(self, chunk: Any, *, is_flush: bool = False) -> dict[str, Any]:
        cached, delta, self.state = self.model.append_audio_to_cache(chunk, self.state)
        if int(cached.shape[1]) == 0:
            hypothesis_tokens: list[int] = []
        else:
            prefix_token_ids = self.prompt_prefix_token_ids(
                audio_steps=int(cached.shape[1])
            )
            generated = self.model.generate_full_hypothesis_from_cached_audio(
                cached,
                prefix_token_ids=prefix_token_ids,
                audio_placeholder_token_id=self.config.audio_placeholder_token_id,
                prompt_token_ids=self.config.prompt_token_ids,
                max_new_tokens=self.config.max_new_tokens,
                eos_token_id=self.config.eos_token_id,
                suppress_token_ids=list(self.config.suppress_token_ids),
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                max_consecutive_text_tokens=self.config.max_consecutive_text_tokens,
                use_decoder_kv_cache=self.config.use_decoder_kv_cache,
            )
            hypothesis_tokens = trim_at_stop(
                _tensor_to_int_list(generated),
                self.config.eos_token_id,
            )

        audio = getattr(self.state, "audio", None)
        audio_sec = float(
            getattr(audio, "frames_seen", 0)
            * self.model.config.mel_hop_ms
            / 1000.0
        )
        event = self.update_from_hypothesis(
            hypothesis_tokens,
            audio_sec=audio_sec,
            is_flush=is_flush,
            input_mel_frames=int(chunk.shape[1]),
            new_cached_steps=int(delta.shape[1]),
            cached_steps=int(cached.shape[1]),
            audio_frames_seen=int(getattr(audio, "frames_seen", 0)),
            input_frames=int(getattr(audio, "last_input_frames", 0)),
            last_recomputed_frames=int(getattr(audio, "last_recomputed_frames", 0)),
            last_recomputed_context_frames=int(
                getattr(audio, "last_recomputed_context_frames", 0)
            ),
        )
        return event

    def update_from_hypothesis(
        self,
        hypothesis_tokens: Sequence[int],
        *,
        audio_sec: float,
        is_flush: bool = False,
        input_mel_frames: int = 0,
        new_cached_steps: int = 0,
        cached_steps: int = 0,
        audio_frames_seen: int = 0,
        input_frames: int = 0,
        last_recomputed_frames: int = 0,
        last_recomputed_context_frames: int = 0,
    ) -> dict[str, Any]:
        hypothesis_tokens = [int(token_id) for token_id in hypothesis_tokens]
        hypothesis_text = decode_clean_token_ids(
            self.tokenizer,
            hypothesis_tokens,
            wait_token_id=self.config.wait_token_id,
            word_start_token_id=self.config.word_start_token_id,
        )
        self.last_hypothesis_tokens = hypothesis_tokens
        self.last_hypothesis_text = hypothesis_text
        allow_commit = float(audio_sec) >= self.config.min_commit_audio_sec

        if self.config.commit_mode == "token":
            update = update_stable_prefix_commit(
                self.token_commit_state,
                hypothesis_tokens,
                hold_back_tokens=self.config.hold_back_tokens,
                stable_iterations=self.config.stable_iterations,
                allow_commit=allow_commit,
            )
            committed_text = decode_clean_token_ids(
                self.tokenizer,
                update.committed_tokens,
                wait_token_id=self.config.wait_token_id,
                word_start_token_id=self.config.word_start_token_id,
            )
            delta_text = decode_clean_token_ids(
                self.tokenizer,
                update.delta_tokens,
                wait_token_id=self.config.wait_token_id,
                word_start_token_id=self.config.word_start_token_id,
            )
            display_text = hypothesis_text
            unstable_text = hypothesis_text[len(committed_text) :].strip()
            committed_count = len(update.committed_tokens)
            delta_count = len(update.delta_tokens)
            candidate_text = decode_clean_token_ids(
                self.tokenizer,
                update.candidate_tokens,
                wait_token_id=self.config.wait_token_id,
                word_start_token_id=self.config.word_start_token_id,
            )
        else:
            text_update = update_stable_text_commit(
                self.text_commit_state,
                hypothesis_text,
                hold_back_units=self.config.hold_back_words,
                stable_iterations=self.config.stable_iterations,
                normalize_for_match=self.config.normalize_commit_match,
                allow_commit=allow_commit,
            )
            committed_text = text_update.committed_text
            delta_text = text_update.delta_text
            display_text = text_update.display_text
            unstable_text = text_update.unstable_text
            committed_count = text_update.committed_unit_count
            delta_count = len(delta_text.split()) if delta_text else 0
            candidate_text = text_update.candidate_text

        self.last_display_text = display_text
        self.last_committed_text = committed_text
        event = {
            "is_flush": is_flush,
            "input_mel_frames": int(input_mel_frames),
            "new_cached_steps": int(new_cached_steps),
            "cached_steps": int(cached_steps),
            "audio_sec": float(audio_sec),
            "audio_frames_seen": int(audio_frames_seen),
            "input_frames": int(input_frames),
            "last_recomputed_frames": int(last_recomputed_frames),
            "last_recomputed_context_frames": int(last_recomputed_context_frames),
            "hypothesis_tokens": len(hypothesis_tokens),
            "committed_units": int(committed_count),
            "delta_units": int(delta_count),
            "hypothesis": hypothesis_text,
            "committed": committed_text,
            "display": display_text,
            "unstable": unstable_text,
            "delta": delta_text,
            "candidate": candidate_text,
        }
        self.events.append(event)
        return event

    def finalize(self, *, finalize_mode: str = "latest") -> CachedFullHypothesisFinal:
        if finalize_mode not in {"latest", "stable"}:
            raise ValueError("finalize_mode must be 'latest' or 'stable'")
        if self.config.commit_mode == "token":
            final_update = update_stable_prefix_commit(
                self.token_commit_state,
                self.last_hypothesis_tokens,
                final=finalize_mode == "latest",
            )
            final_tokens = final_update.committed_tokens
            final_text = decode_clean_token_ids(
                self.tokenizer,
                final_tokens,
                wait_token_id=self.config.wait_token_id,
                word_start_token_id=self.config.word_start_token_id,
            )
            return CachedFullHypothesisFinal(
                final_tokens=final_tokens,
                final_text=final_text,
                final_display_text=self.last_hypothesis_text,
                stable_committed_text=final_text,
                last_hypothesis_text=self.last_hypothesis_text,
                final_committed_units=len(final_tokens),
            )

        if finalize_mode == "latest":
            text_final = update_stable_text_commit(
                self.text_commit_state,
                self.last_hypothesis_text,
                normalize_for_match=self.config.normalize_commit_match,
                final=True,
                final_revises_committed=True,
            )
        else:
            text_final = update_stable_text_commit(
                self.text_commit_state,
                self.last_hypothesis_text,
                hold_back_units=self.config.hold_back_words,
                stable_iterations=self.config.stable_iterations,
                normalize_for_match=self.config.normalize_commit_match,
            )
        return CachedFullHypothesisFinal(
            final_tokens=self.last_hypothesis_tokens.copy(),
            final_text=text_final.committed_text,
            final_display_text=text_final.display_text,
            stable_committed_text=self.last_committed_text,
            last_hypothesis_text=self.last_hypothesis_text,
            final_committed_units=text_final.committed_unit_count,
        )


@dataclass
class SegmentedCachedFullHypothesisStreamer(CachedFullHypothesisStreamer):
    """Full-hypothesis streamer that periodically rolls the active audio window.

    The base streamer keeps every finalized audio embedding and regenerates one
    growing transcript. That is useful as a correctness probe, but expensive for
    long sessions. This variant commits a completed segment, trims the decoded
    audio embedding cache to a small optional tail, and starts a fresh stable
    prefix state for the active segment while preserving global display text.
    """

    segment_max_cached_steps: int = 0
    segment_keep_tail_steps: int = 0
    segment_finalize_mode: str = "latest"
    segment_prompt_context_words: int = 0
    segment_prompt_base_context: str = ""
    segment_prompt_language: str | None = None
    segment_prompt_context_prefix: str = "Previous transcript context:"
    # Reset the audio encoder state (positions + per-layer KV) when a segment
    # rolls. Bounds the encoder's chain length to one segment: long sessions
    # become sequences of segment-length encodes, trading cross-segment
    # acoustic context for trained-regime fidelity.
    reset_encoder_on_rollover: bool = False
    completed_text: str = ""
    completed_tokens: list[int] = field(default_factory=list)
    segments_finalized: int = 0
    dropped_cached_steps_total: int = 0
    last_global_hypothesis_text: str = ""
    last_global_display_text: str = ""
    last_global_committed_text: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.segment_max_cached_steps < 0:
            raise ValueError("segment_max_cached_steps must be >= 0")
        if self.segment_keep_tail_steps < 0:
            raise ValueError("segment_keep_tail_steps must be >= 0")
        if (
            self.segment_max_cached_steps > 0
            and self.segment_keep_tail_steps >= self.segment_max_cached_steps
        ):
            raise ValueError(
                "segment_keep_tail_steps must be smaller than segment_max_cached_steps"
            )
        if self.segment_finalize_mode not in {"latest", "stable"}:
            raise ValueError("segment_finalize_mode must be 'latest' or 'stable'")
        if self.segment_prompt_context_words < 0:
            raise ValueError("segment_prompt_context_words must be >= 0")

    def prompt_prefix_token_ids(self, *, audio_steps: int) -> list[int] | None:
        if self.segment_prompt_context_words <= 0:
            return super().prompt_prefix_token_ids(audio_steps=audio_steps)
        if self.config.audio_placeholder_token_id is None:
            raise ValueError("missing audio placeholder token id")
        previous = trailing_text_words(
            self.completed_text,
            self.segment_prompt_context_words,
        )
        context_parts = []
        if self.segment_prompt_base_context.strip():
            context_parts.append(self.segment_prompt_base_context.strip())
        if previous:
            context_parts.append(
                f"{self.segment_prompt_context_prefix.strip()}\n{previous}"
            )
        prompt_template = self.tokenizer.encode(
            qwen_asr_prompt_text(
                context="\n\n".join(context_parts),
                language=self.segment_prompt_language,
            ),
            add_special_tokens=False,
        )
        return expand_audio_prompt_placeholders(
            prompt_template,
            audio_placeholder_token_id=self.config.audio_placeholder_token_id,
            audio_steps=int(audio_steps),
        )

    def update_from_hypothesis(
        self,
        hypothesis_tokens: Sequence[int],
        *,
        audio_sec: float,
        is_flush: bool = False,
        input_mel_frames: int = 0,
        new_cached_steps: int = 0,
        cached_steps: int = 0,
        audio_frames_seen: int = 0,
        input_frames: int = 0,
        last_recomputed_frames: int = 0,
        last_recomputed_context_frames: int = 0,
    ) -> dict[str, Any]:
        event = super().update_from_hypothesis(
            hypothesis_tokens,
            audio_sec=audio_sec,
            is_flush=is_flush,
            input_mel_frames=input_mel_frames,
            new_cached_steps=new_cached_steps,
            cached_steps=cached_steps,
            audio_frames_seen=audio_frames_seen,
            input_frames=input_frames,
            last_recomputed_frames=last_recomputed_frames,
            last_recomputed_context_frames=last_recomputed_context_frames,
        )

        segment_hypothesis = event["hypothesis"]
        segment_committed = event["committed"]
        segment_display = event["display"]
        segment_unstable = event["unstable"]
        segment_delta = event["delta"]
        segment_candidate = event["candidate"]

        self.last_global_hypothesis_text = join_text_segments(
            self.completed_text,
            segment_hypothesis,
        )
        self.last_global_committed_text = join_text_segments(
            self.completed_text,
            segment_committed,
        )
        self.last_global_display_text = join_text_segments(
            self.completed_text,
            segment_display,
        )

        event.update(
            {
                "segment_index": int(self.segments_finalized),
                "segments_finalized": int(self.segments_finalized),
                "dropped_cached_steps_total": int(self.dropped_cached_steps_total),
                "segment_prompt_context_words": int(
                    self.segment_prompt_context_words
                ),
                "segment_prompt_context": trailing_text_words(
                    self.completed_text,
                    self.segment_prompt_context_words,
                ),
                "segment_hypothesis": segment_hypothesis,
                "segment_committed": segment_committed,
                "segment_display": segment_display,
                "segment_unstable": segment_unstable,
                "segment_delta": segment_delta,
                "segment_candidate": segment_candidate,
                "completed_text": self.completed_text,
                "hypothesis": self.last_global_hypothesis_text,
                "committed": self.last_global_committed_text,
                "display": self.last_global_display_text,
                "unstable": segment_unstable,
                "delta": segment_delta,
                "candidate": join_text_segments(
                    self.completed_text,
                    segment_candidate,
                ),
                "segment_rollover": False,
            }
        )

        if (
            self.segment_max_cached_steps > 0
            and int(cached_steps) > self.segment_max_cached_steps
        ):
            segment_final = self.roll_segment()
            event.update(
                {
                    "segment_rollover": True,
                    "segment_final_text": segment_final.final_text,
                    "segments_finalized": int(self.segments_finalized),
                    "dropped_cached_steps_total": int(
                        self.dropped_cached_steps_total
                    ),
                    "completed_text_after_roll": self.completed_text,
                    "active_cached_steps_after_roll": self._active_cached_steps(),
                }
            )

        return event

    def roll_segment(self) -> CachedFullHypothesisFinal:
        segment_final = super().finalize(finalize_mode=self.segment_finalize_mode)
        self.completed_text = join_text_segments(
            self.completed_text,
            segment_final.final_text,
        )
        self.completed_tokens.extend(segment_final.final_tokens)
        self.segments_finalized += 1
        self._trim_cached_audio_window()
        if self.reset_encoder_on_rollover:
            frames_seen = int(getattr(self.state.audio, "frames_seen", 0))
            pending = getattr(self.state.audio, "mel_buffer", None)
            self.state.audio = self.model.audio_encoder.init_state()
            self.state.audio.frames_seen = frames_seen
            if pending is not None:
                self.state.audio.mel_buffer = pending
        self._reset_active_segment_state()
        return segment_final

    def _active_cached_steps(self) -> int:
        frame_hidden = getattr(self.state, "frame_hidden", None)
        if frame_hidden is None:
            return 0
        return int(frame_hidden.shape[1])

    def _trim_cached_audio_window(self) -> None:
        frame_hidden = getattr(self.state, "frame_hidden", None)
        if frame_hidden is None:
            return
        old_steps = int(frame_hidden.shape[1])
        keep_steps = min(int(self.segment_keep_tail_steps), old_steps)
        dropped = old_steps - keep_steps
        if keep_steps == 0:
            self.state.frame_hidden = frame_hidden[:, 0:0, :]
        else:
            self.state.frame_hidden = frame_hidden[:, -keep_steps:, :]
        self.dropped_cached_steps_total += int(dropped)

    def _reset_active_segment_state(self) -> None:
        self.token_commit_state = StablePrefixCommitState()
        self.text_commit_state = StableTextCommitState()
        self.last_hypothesis_tokens = []
        self.last_hypothesis_text = ""
        self.last_display_text = ""
        self.last_committed_text = ""

    def finalize(self, *, finalize_mode: str = "latest") -> CachedFullHypothesisFinal:
        if finalize_mode not in {"latest", "stable"}:
            raise ValueError("finalize_mode must be 'latest' or 'stable'")
        active_final = super().finalize(finalize_mode=finalize_mode)
        final_tokens = self.completed_tokens + active_final.final_tokens
        final_text = join_text_segments(self.completed_text, active_final.final_text)
        final_display_text = join_text_segments(
            self.completed_text,
            active_final.final_display_text,
        )
        stable_committed_text = join_text_segments(
            self.completed_text,
            active_final.stable_committed_text,
        )
        last_hypothesis_text = join_text_segments(
            self.completed_text,
            active_final.last_hypothesis_text,
        )
        return CachedFullHypothesisFinal(
            final_tokens=final_tokens,
            final_text=final_text,
            final_display_text=final_display_text,
            stable_committed_text=stable_committed_text,
            last_hypothesis_text=last_hypothesis_text,
            final_committed_units=len(final_text.split()) if final_text else 0,
        )
