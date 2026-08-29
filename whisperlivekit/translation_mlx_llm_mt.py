"""Generic in-process MLX translation backend with a config-driven model registry.

Any decoder-LLM translation model (Hunyuan-MT, TranslateGemma, Aya, Qwen-MT)
runs via mlx-lm on Apple Silicon. The model-specific parts — prompt template,
EOS token, and sampling params — live in a config dict
(``MTX_MODEL_CONFIGS``). Add a model by adding a config entry, not by
writing a subclass.

Duck-typed contract (same shape as nllw.OnlineTranslation):
  - ``insert_tokens(items)``: receive committed ASRTokens; punctuation closes a segment.
  - ``process()`` -> ``(Translation|None, TimedText buffer)``: translate closed segments.
  - ``validate_buffer_and_reset()`` -> ``(Translation, TimedText)``: flush at silence or speaker change.
  - ``insert_silence(duration)``: no-op.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from whisperlivekit.timed_objects import ASRToken, TimedText, Translation

# The profile dataclass + registry live in the neutral ``translation_profiles"
# module so a future peer backend (vLLM, etc.) can share them without importing
# this mlx-specific module.
from whisperlivekit.translation_profiles import (  # noqa: E402
    MT_MODEL_PROFILES as MTX_MODEL_CONFIGS,
)
from whisperlivekit.translation_profiles import (  # noqa: E402
    MtModelProfile as MlxLlmMtModelConfig,
)

logger = logging.getLogger(__name__)


_HY_PLACEHOLDER_TEXT = "<｜hy_place▁holder▁no▁2｜>"
_HY_PLACEHOLDER_RE = re.compile(r"<[\|｜][^\|｜]*[\|｜]>")


def _placeholder_stop_check(tokenizer):
    """Build a per-call predicate that fires when the placeholder was just emitted.

    Returns ``None`` when the placeholder can't be matched as a token sequence
    (encode failed or fragmented into too many BPE pieces to window reliably);
    the caller then relies on the post-hoc string strip alone.

    Hy-MT2-1.8B encodes the placeholder as a single id — an exact, zero-cost
    match. Hunyuan-MT-7B fragments it into ~13 byte-level BPE ids, so a rolling
    id window is used instead: it fires only when the model emits the exact id
    sequence consecutively, which is still far cheaper than decoding the
    hallucinated tail and letting ``_strip_hy_placeholder`` cut it afterwards.
    """
    try:
        try:
            ids = tokenizer.encode(_HY_PLACEHOLDER_TEXT, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(_HY_PLACEHOLDER_TEXT)
    except Exception:
        return None
    ids = tuple(ids)
    if not ids or len(ids) > 16:
        return None
    if len(ids) == 1:
        target = ids[0]
        return lambda chunk: getattr(chunk, "token", None) == target
    window: deque = deque(maxlen=len(ids))

    def _check(chunk) -> bool:
        window.append(getattr(chunk, "token", None))
        return len(window) == len(ids) and tuple(window) == ids

    return _check


def _strip_hy_placeholder(text: str) -> str:
    """Truncate at the first Hunyuan placeholder token and everything after it.

    The model occasionally emits ``<｜hy_place▁holder▁no▁2｜>`` (fullwidth pipes,
    U+FF5C) and then hallucinates free text after it. Stripping just the token
    leaves the hallucination in the output, so cut from the placeholder onward.
    Applied at source (in the translation engine) so every consumer — terminal,
    overlay, transcript file, simul commit policy — sees clean text; the display
    layers strip defensively as well.

    The decode loop additionally stops in-loop at the placeholder when the
    tokenizer encodes it to a short id sequence (see ``_placeholder_stop_check``),
    so the hallucinated tail is never decoded at all. This strip stays as the
    defensive fallback for tokenizers/pathologies the in-loop stop can't match.
    """
    if not text:
        return text
    m = _HY_PLACEHOLDER_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


class MlxLlmTranslation:
    """In-process MLX translation backend via mlx-lm.

    One shared model instance per repo per process (load is expensive);
    the contract is stateless across sessions except for the per-instance
    segment buffer.
    """

    _MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}  # repo → (model, tokenizer)

    def __init__(
        self,
        model_id: str = "hy-mt2-1.8b-8bit",
        target_language: str = "en",
        source_language: str = "",
        warmup: bool = True,
    ):
        self._model_id = model_id
        config = MTX_MODEL_CONFIGS.get(model_id)
        if config is None:
            available = ", ".join(sorted(MTX_MODEL_CONFIGS))
            raise ValueError(
                f"Unknown mlx-llm-mt model '{model_id}'. "
                f"Available: {available}"
            )
        self._config = config
        self._target_language = target_language
        self._source_language = source_language
        self._prompt = self._resolve_prompt()
        # Per-instance segment state.
        self._buffer_tokens: List[ASRToken] = []
        self._buffer_start: Optional[float] = None
        self._pending_finals: List[Tuple[str, float, float]] = []  # (text, start, end)
        self._last_buffer = TimedText()
        self._eos_token = config.eos_token  # may be None → resolved at load
        # Benchmark instrumentation: cumulative wall-time spent generating MT
        # output (excludes warmup, model load, and ASR).
        self._mt_total_time_s = 0.0
        self._mt_call_count: int = 0
        if warmup:
            self._warmup()

    def new_session(self, target_language: str = "") -> "MlxLlmTranslation":
        """Create a per-session translation client sharing the loaded model/cache
        but with fresh per-instance state (buffer, pending finals, metrics).

        Mirrors AlignAtt's ``new_session`` contract: the server-wide
        ``MlxLlmTranslation`` holds the expensive model; each session gets its
        own ``new_session()`` client so ``_buffer_tokens``, ``_pending_finals``,
        and ``_last_buffer`` don't cross session boundaries.
        """
        return MlxLlmTranslation(
            model_id=self._model_id,
            target_language=target_language or self._target_language,
            source_language=self._source_language,
            warmup=False,  # model already loaded in the cache
        )

    # ------------------------------------------------------------------
    # Model load + decode (generic; config-driven)
    # ------------------------------------------------------------------

    def _resolve_prompt(self) -> dict:
        """Resolve the prompt specification for the source→target pair.

        Returns a dict (see ``resolve_prompt``). The backend branches on
        ``kind`` in ``_translate_text``.
        """
        from whisperlivekit.translation_profiles import resolve_prompt
        return resolve_prompt(self._config, self._source_language, self._target_language)

    def _warmup(self) -> None:
        """Run one short decode to absorb Metal kernel compilation now, so the
        first real sentence's translation doesn't stall for ~10s."""
        try:
            self._translate_text("warmup。")
        except Exception as exc:  # warmup is non-fatal
            logger.debug("mlx-llm-mt warmup failed (non-fatal): %s", exc)

    @classmethod
    def _ensure_model(cls, config: MlxLlmMtModelConfig):
        repo = config.repo
        if repo not in cls._MODEL_CACHE:
            from mlx_lm import load  # lazy; mlx-lm is an extra
            logger.info("Loading MT model %s ...", repo)
            cls._MODEL_CACHE[repo] = load(repo)  # (model, tokenizer) tuple
        return cls._MODEL_CACHE[repo]

    def _translate_text(self, text: str) -> str:
        from mlx_lm import stream_generate  # lazy import in method scope
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        self._mt_call_count += 1

        model, tokenizer = self._ensure_model(self._config)
        # Resolve the EOS token lazily (may need the tokenizer).
        if self._eos_token is None:
            self._eos_token = getattr(tokenizer, "eos_token", "") or ""
        eos = self._eos_token
        if self._prompt["kind"] == "structured_chat":
            content = [{
                "type": "text",
                "source_lang_code": self._prompt["src"],
                "target_lang_code": self._prompt["tgt"],
                "text": text,
            }]
        else:
            content = self._prompt["template"].format(
                target_lang=self._prompt["target_name"], text=text
            )
        # Decoder-LLM MT models are chat models — apply the chat template (a
        # bare prompt hallucinates and rambles past EOS).
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        sampler = make_sampler(
            temp=self._config.temp,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
        )
        processors = make_logits_processors(
            repetition_penalty=self._config.repetition_penalty
        )
        out = ""
        # In-loop early stop: end decode the moment the placeholder is emitted
        # instead of paying for the hallucinated tail and cutting it afterwards.
        stop_at_placeholder = _placeholder_stop_check(tokenizer)
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self._config.max_tokens,
            sampler=sampler,
            logits_processors=processors,
        ):
            out += chunk.text if hasattr(chunk, "text") else str(chunk)
            if eos and out.endswith(eos):
                break
            if stop_at_placeholder is not None and stop_at_placeholder(chunk):
                break
        if eos:
            out = out.replace(eos, "")
        return _strip_hy_placeholder(out)

    # WLK 5-method contract (model-agnostic)

    def insert_tokens(self, items: List[Any]) -> None:
        for item in items:
            if not isinstance(item, ASRToken):
                continue
            if not item.text or not item.text.strip():
                continue
            if self._buffer_start is None:
                self._buffer_start = item.start
            self._buffer_tokens.append(item)
            # Punctuation closes the segment.
            if item.has_punctuation():
                text = "".join(t.text for t in self._buffer_tokens).strip()
                self._pending_finals.append((text, self._buffer_start, item.end))
                self._buffer_tokens = []
                self._buffer_start = None

    def process(self) -> Tuple[Optional[Translation], TimedText]:
        # Translate any closed (punctuated) segments; emit the first.
        if self._pending_finals:
            text, start, end = self._pending_finals.pop(0)
            _t0 = time.perf_counter()
            try:
                mt = self._translate_text(text)
            except Exception as exc:
                logger.warning("mlx-llm-mt translate failed: %s", exc)
                return None, self._last_buffer
            finally:
                self._mt_total_time_s += time.perf_counter() - _t0
            tr = Translation(start=start, end=end, text=mt)
            self._last_buffer = TimedText(start=start, end=end, text=mt)
            return tr, self._last_buffer
        # No closed segment: emit the running partial as the buffer (untranslated).
        if self._buffer_tokens:
            text = "".join(t.text for t in self._buffer_tokens).strip()
            start = self._buffer_start or 0.0
            end = self._buffer_tokens[-1].end
            self._last_buffer = TimedText(start=start, end=end, text=text)
        return None, self._last_buffer

    def validate_buffer_and_reset(self) -> Tuple[Translation, TimedText]:
        """Silence / speaker-change boundary: flush the open segment now."""
        if self._buffer_tokens:
            text = "".join(t.text for t in self._buffer_tokens).strip()
            start = self._buffer_start or 0.0
            end = self._buffer_tokens[-1].end
            self._pending_finals.append((text, start, end))
            self._buffer_tokens = []
            self._buffer_start = None
        # The open utterance is closed.
        if self._pending_finals:
            text, start, end = self._pending_finals.pop(0)
            _t0 = time.perf_counter()
            try:
                mt = self._translate_text(text)
            except Exception as exc:
                logger.warning("mlx-llm-mt validate translate failed: %s", exc)
                mt = ""
            finally:
                self._mt_total_time_s += time.perf_counter() - _t0
            tr = Translation(start=start, end=end, text=mt)
            self._last_buffer = TimedText(start=start, end=end, text=mt)
            return tr, self._last_buffer
        # Nothing to flush: return empty. Do not return the stale
        # _last_buffer; that would duplicate the previous translation.
        return TimedText(), TimedText()

    def insert_silence(self, duration: float = None) -> None:
        pass


# The profiles module is imported above (it populates MT_MODEL_PROFILES at
# load time). The old profiles-specific module is gone — profiles live in
# ``translation_profiles`` now.
