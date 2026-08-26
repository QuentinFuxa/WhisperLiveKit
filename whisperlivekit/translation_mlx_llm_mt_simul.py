"""Simultaneous-MT variant of the mlx-llm-mt translation backend.

A ``MlxLlmTranslationSimul`` subclass of ``MlxLlmTranslation``
that drafts translation over the unstable ASR tail (``HypothesisTail``)
and applies the AlignAtt commit policy: commit only target tokens whose
attention argmax (top calibrated zh→en head) lands on a source token the
ASR has committed; hold the rest. When the ASR later commits the tail,
held tokens release from the cached attention WITHOUT a new MT call —
that is the latency win.

The base ``MlxLlmTranslation`` is unchanged; this variant only
adds the simultaneous behaviour. The base's ``self._tail`` seam
(``HypothesisTail`` storage) is the opt-in point.

Duck-typed contract (same shape as ``MlxLlmTranslation``):
  - ``insert_tokens(items)``: committed ASRTokens + ``HypothesisTail``.
  - ``process()`` -> ``(Translation|None, TimedText)``: provisional EN
    during speech (buffer), validated Translation at segment close.
  - ``validate_buffer_and_reset()``: flush at silence / speaker change.
  - ``insert_silence(duration)``: no-op.
"""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

from whisperlivekit.simul_mt_capture import (
    ALIGNMENT_HEADS,
    TOP_HEAD,
    apply_commit_policy,
    committed_src_end_from_text,
    install_capture,
    source_span,
)
from whisperlivekit.timed_objects import ASRToken, HypothesisTail, TimedText, Translation
from whisperlivekit.translation_mlx_llm_mt import MlxLlmTranslation

logger = logging.getLogger(__name__)


class MlxLlmTranslationSimul(MlxLlmTranslation):
    """In-process simultaneous-MT backend: AlignAtt commit policy over the
    unstable ASR tail, with calibrated zh→en Hunyuan alignment heads.

    Sets ``wants_hypothesis_tail = True`` so the audio processor forwards
    the unstable ASR tail; drafts translation over (committed + tail) and
    commits only the target prefix aligning to committed source.
    """

    def __init__(
        self,
        model_id: str = "hy-mt2-1.8b-8bit",
        target_language: str = "en",
        source_language: str = "",
        warmup: bool = True,
    ):
        super().__init__(
            model_id=model_id,
            target_language=target_language,
            source_language=source_language,
            warmup=warmup,
        )
        self.wants_hypothesis_tail = True
        # Per-instance simultaneous state.
        self._tail: Optional[HypothesisTail] = None
        self._committed_simul: List[ASRToken] = []  # committed tokens (open utterance)
        self._committed_start: Optional[float] = None
        # Cached draft from the last MT call (for the release-without-call path).
        self._last_source_text: str = ""
        self._last_draft: Optional[dict] = None  # {tokens, src_start, src_end}
        # Minimum source char growth to warrant a fresh draft. Below this,
        # the release path re-applies the commit policy on the cached
        # attention (no MT call). For CJK each char ≈ 1 token; 15 chars ≈ one
        # sentence, keeping provisional calls to ~1 per sentence instead of
        # one per tail update.
        self._MIN_SOURCE_DELTA: int = 15
        # Stable, append-only provisional target emitted so far this utterance.
        self._emitted_partial: str = ""
        self._capture: Optional[dict] = None
        self._capture_installed = False
        logger.info(
            "MlxLlmTranslationSimul: alignment heads=%s top=%s",
            ALIGNMENT_HEADS, TOP_HEAD,
        )

    # ------------------------------------------------------------------
    # Model load + capture installation
    # ------------------------------------------------------------------

    def _ensure_simul_model(self):
        """Ensure the model is loaded and the Q/K capture is installed.

        The capture is installed on the shared cached model (idempotent via
        ``install_capture``). The capture dict is shared across simul
        instances that use the same repo; it is cleared before each
        generate and read after.
        """
        model, tokenizer = self._ensure_model(self._config)
        if not self._capture_installed:
            self._capture = install_capture(model, ALIGNMENT_HEADS)
            self._capture_installed = True
        return model, tokenizer

    # ------------------------------------------------------------------
    # Simul MT generation + commit policy
    # ------------------------------------------------------------------

    def _build_prompt_content(self, text: str):
        """Build the chat-message content for the MT prompt, reusing the
        base class's resolved prompt (``self._prompt``).

        Branches on ``kind`` exactly like ``MlxLlmTranslation._translate_text``.
        Returns the content value (string or structured list) for the
        ``user`` message.
        """
        if self._prompt["kind"] == "structured_chat":
            return [{
                "type": "text",
                "source_lang_code": self._prompt["src"],
                "target_lang_code": self._prompt["tgt"],
                "text": text,
            }]
        return self._prompt["template"].format(
            target_lang=self._prompt["target_name"], text=text
        )

    def _translate_simul(self, source_text: str, committed_text: str) -> str:
        """Generate a translation draft over the full source (committed +
        tail) with Q/K capture, apply the commit policy, and return the
        committed target prefix. Stashes the draft for the
        release-without-call path.

        ``source_text`` is the full source the MT conditions on (committed
        prefix + unstable tail). ``committed_text`` is the stable prefix
        whose source tokens count as committed for the policy.
        """
        from mlx_lm import stream_generate  # lazy

        model, tokenizer = self._ensure_simul_model()
        if self._eos_token is None:
            self._eos_token = getattr(tokenizer, "eos_token", "") or ""
        content = self._build_prompt_content(source_text)
        messages = [{"role": "user", "content": content}]
        prompt_str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        src_start, src_end = source_span(tokenizer, prompt_str, source_text)
        src_ids = prompt_ids[src_start:src_end]
        cend = committed_src_end_from_text(tokenizer, src_ids, committed_text)

        assert self._capture is not None
        self._capture.clear()
        gen = stream_generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=self._config.max_tokens,
            sampler=self._make_sampler(),
            logits_processors=self._make_logits_processors(),
        )
        tokens: List[int] = []
        eos = self._eos_token
        for chunk in gen:
            tokens.append(chunk.token)
            # Stop at EOS for efficiency; the policy commits a prefix anyway.
            if eos:
                det = tokenizer.decode([chunk.token])
                if eos in det:
                    tokens.pop()
                    break
        committed_len = apply_commit_policy(
            self._capture, TOP_HEAD, len(tokens), src_start, src_end, cend
        )
        committed_tokens = tokens[:committed_len]
        committed_text_out = tokenizer.decode(committed_tokens).strip()
        # Stash the draft for the release path (same source, bigger boundary).
        self._last_draft = {
            "tokens": tokens,
            "src_start": src_start,
            "src_end": src_end,
        }
        return committed_text_out

    def _release_held(self, committed_text: str) -> str:
        """Re-apply the commit policy to the CACHED attention from the last MT
        call with a larger committed-source boundary (the ASR committed more
        of the tail). Releases held target tokens WITHOUT a new MT call.

        Only valid when the total source text is unchanged from the last
        call (caller guarantees this by comparing ``_last_source_text``).
        """
        if self._last_draft is None or self._capture is None:
            return self._emitted_partial
        model, tokenizer = self._ensure_simul_model()
        # Re-derive the committed boundary from the stashed source span. The
        # source text is unchanged, so the prompt/source span is the same;
        # recompute cend from committed_text against the stashed span.
        draft = self._last_draft
        src_start, src_end = draft["src_start"], draft["src_end"]
        # Rebuild the prompt to get src_ids (source unchanged → same ids).
        content = self._build_prompt_content(self._last_source_text)
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        src_ids = prompt_ids[src_start:src_end]
        cend = committed_src_end_from_text(tokenizer, src_ids, committed_text)
        # The capture still holds the last call's attentions (not cleared).
        committed_len = apply_commit_policy(
            self._capture, TOP_HEAD, len(draft["tokens"]), src_start, src_end, cend
        )
        committed_tokens = draft["tokens"][:committed_len]
        return tokenizer.decode(committed_tokens).strip()

    def _make_sampler(self):
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=self._config.temp,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
        )

    def _make_logits_processors(self):
        from mlx_lm.sample_utils import make_logits_processors

        return make_logits_processors(
            repetition_penalty=self._config.repetition_penalty
        )

    # ------------------------------------------------------------------
    # Source construction helpers
    # ------------------------------------------------------------------

    def _committed_text(self) -> str:
        return "".join(t.text for t in self._committed_simul).strip()

    def _source_text(self) -> str:
        """Full source the MT conditions on: committed prefix + unstable tail.

        Concatenated WITHOUT a separator so the source is invariant to the
        committed/tail split (the release mechanism relies on this: when the
        ASR commits part of the tail, the total source text is unchanged, so
        the cached attention is still valid). For CJK (the zh→en use case) the
        ``""`` join is natural; the base class joins tokens the same way.
        """
        committed = self._committed_text()
        tail = (self._tail.text or "").strip() if self._tail else ""
        return committed + tail

    def _utterance_start(self) -> Optional[float]:
        if self._committed_start is not None:
            return self._committed_start
        if self._committed_simul:
            return self._committed_simul[0].start
        if self._tail is not None:
            return self._tail.start
        return None

    def _utterance_end(self) -> Optional[float]:
        if self._committed_simul:
            return self._committed_simul[-1].end
        if self._tail is not None:
            return self._tail.end
        return None

    def _segment_start(self, fallback: Optional[float]) -> float:
        return fallback if fallback is not None else 0.0

    # ------------------------------------------------------------------
    # WLK contract
    # ------------------------------------------------------------------

    def insert_tokens(self, items: List[Any]) -> None:
        for item in items:
            if isinstance(item, HypothesisTail):
                self._tail = item
                continue
            if not isinstance(item, ASRToken):
                continue
            if not item.text or not item.text.strip():
                continue
            if self._committed_start is None:
                self._committed_start = item.start
            self._committed_simul.append(item)
            # Punctuation closes the segment → queue a final.
            if item.has_punctuation():
                text = self._committed_text()
                self._pending_finals.append(
                    (text, self._committed_start, item.end)
                )
                self._committed_simul = []
                self._committed_start = None
                self._tail = None

    def process(self) -> Tuple[Optional[Translation], TimedText]:
        # 1. Finals first (punctuation-closed segments): full base-class translation.
        if self._pending_finals:
            text, start, end = self._pending_finals.pop(0)
            _t0 = time.perf_counter()
            try:
                mt = self._translate_text(text)
            except Exception as exc:
                logger.warning("mlx-llm-mt-simul translate failed: %s", exc)
                return None, self._buffer()
            finally:
                self._mt_total_time_s += time.perf_counter() - _t0
            self._reset_simul_draft()
            tr = Translation(start=start, end=end, text=mt)
            self._last_buffer = TimedText(start=start, end=end, text=mt)
            return tr, self._last_buffer

        # 2. Open utterance: simultaneous provisional over committed + tail.
        source = self._source_text()
        committed = self._committed_text()
        has_content = bool(committed) or bool(
            self._tail and (self._tail.text or "").strip()
        )
        if not has_content:
            return None, self._buffer()

        # Decide new MT call vs release (no call).
        # A new draft is warranted only when the total source (committed +
        # tail) grew by >= MIN_SOURCE_DELTA chars since the last draft, or
        # no draft exists yet. Below that, the release path re-applies the
        # commit policy on the cached attention (no MT call) — the committed
        # prefix is still valid in the cached draft.
        #
        # _last_source_text is only updated when a new draft is made, so the
        # hysteresis accumulates across releases within one utterance.
        if self._last_draft is not None:
            source_delta = len(source) - len(self._last_source_text)
            if source_delta < self._MIN_SOURCE_DELTA:
                # Source unchanged or grew but not enough: release held
                # tokens from the cached draft without a new MT call.
                if committed:
                    released = self._release_held(committed)
                    if released and len(released) > len(self._emitted_partial):
                        self._emitted_partial = released
            else:
                # Source grew enough: fresh MT call with capture.
                self._mt_call_count += 1
                _t0 = time.perf_counter()
                try:
                    committed_out = self._translate_simul(source, committed)
                except Exception as exc:
                    logger.warning("mlx-llm-mt-simul simul draft failed: %s", exc)
                    return None, self._buffer()
                finally:
                    self._mt_total_time_s += time.perf_counter() - _t0
                self._last_source_text = source
                self._emitted_partial = committed_out
        else:
            # No draft yet: must make a new call.
            self._mt_call_count += 1
            _t0 = time.perf_counter()
            try:
                committed_out = self._translate_simul(source, committed)
            except Exception as exc:
                logger.warning("mlx-llm-mt-simul simul draft failed: %s", exc)
                return None, self._buffer()
            finally:
                self._mt_total_time_s += time.perf_counter() - _t0
            self._last_source_text = source
            self._emitted_partial = committed_out
        return None, self._buffer()

    def validate_buffer_and_reset(self) -> Tuple[Optional[Translation], TimedText]:
        """Silence / speaker-change boundary: flush the open utterance.

        The provisional (if any) is kept as the buffer — it is NOT committed
        as a validated Translation. The open utterance is queued as a final
        so the next ``process()`` produces the quality pass (full base-class
        translation). This avoids duplication: the only committed Translation
        for each utterance is the final quality pass.
        """
        start = self._segment_start(self._utterance_start())
        end = self._utterance_end() or self._segment_start(None)
        if self._committed_simul:
            text = self._committed_text()
            self._pending_finals.append(
                (text, self._committed_start, self._committed_simul[-1].end)
            )
            self._committed_simul = []
            self._committed_start = None
        self._tail = None
        self._reset_simul_draft()
        emitted = self._emitted_partial
        self._emitted_partial = ""
        if emitted:
            self._last_buffer = TimedText(
                start=start, end=end, text=emitted
            )
            # Keep the provisional as the buffer (shown on screen) but do
            # NOT commit it as a Translation — the pending final (quality
            # pass) will be the only committed Translation for this utterance.
            return None, self._last_buffer
        # Nothing was emitted; fall back to a base-class flush of any buffered
        # tokens (mirrors the base class behaviour for a non-simul flush).
        if self._pending_finals:
            text, start, end = self._pending_finals.pop(0)
            try:
                mt = self._translate_text(text)
            except Exception as exc:
                logger.warning("mlx-llm-mt-simul validate translate failed: %s", exc)
                mt = ""
            tr = Translation(start=start, end=end, text=mt)
            self._last_buffer = TimedText(start=start, end=end, text=mt)
            return tr, self._last_buffer
        return TimedText(), TimedText()

    def insert_silence(self, duration: float = None) -> None:
        pass

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _buffer(self) -> TimedText:
        if not self._emitted_partial:
            return self._last_buffer if self._last_buffer.text else TimedText()
        return TimedText(
            start=self._segment_start(self._utterance_start()),
            end=self._utterance_end(),
            text=self._emitted_partial,
        )

    def _reset_simul_draft(self) -> None:
        self._last_source_text = ""
        self._last_draft = None
        if self._capture is not None:
            self._capture.clear()
