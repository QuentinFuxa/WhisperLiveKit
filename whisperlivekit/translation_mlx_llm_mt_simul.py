"""MLX translation drafts gated by calibrated AlignAtt attention heads."""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

from whisperlivekit.simul_mt_calibration import load_calibration
from whisperlivekit.simul_mt_capture import (
    apply_commit_policy,
    capture_attention,
    committed_src_end_from_text,
    snapshot_capture,
    source_span,
)
from whisperlivekit.timed_objects import ASRToken, HypothesisTail, TimedText
from whisperlivekit.translation_mlx_llm_mt import (
    MlxLlmTranslation,
    _placeholder_stop_check,
    _strip_hy_placeholder,
)


class MlxLlmTranslationSimul(MlxLlmTranslation):
    wants_hypothesis_tail = True

    def __init__(self, model_id="hy-mt2-1.8b-8bit", target_language="en",
                 source_language="", warmup=True, commit_mode="paper",
                 mass_threshold=0.5, simul_soft_max_s=4.0, simul_hard_max_s=20.0,
                 calibration_file=None):
        super().__init__(model_id, target_language, source_language, warmup=False)
        self._calibration_file = calibration_file
        self._calibration = load_calibration(calibration_file, self._config.repo, source_language, target_language)
        if self._calibration.prompt != self._prompt:
            raise ValueError("Calibration prompt differs from the translation model profile")
        self._config = replace(self._config, revision=self._calibration.revision)
        if commit_mode not in {"paper", "argmax", "mass"}:
            raise ValueError(f"Unknown alignment policy: {commit_mode}")
        if not all(math.isfinite(v) for v in (mass_threshold, simul_soft_max_s, simul_hard_max_s)):
            raise ValueError("Simultaneous translation limits must be finite")
        if not 0 < mass_threshold <= 1 or not 0 < simul_soft_max_s <= simul_hard_max_s:
            raise ValueError("Require 0 < mass threshold <= 1 and 0 < soft duration <= hard duration")
        self._commit_mode, self._mass_threshold = commit_mode, mass_threshold
        self._simul_soft_max_s, self._simul_hard_max_s = simul_soft_max_s, simul_hard_max_s
        self._committed_simul = []
        self._tail = None
        self._last_draft = None
        self._emitted_partial = ""
        self._verified_model = False
        if warmup:
            with self._MODEL_LOCK:
                self._ensure_simul_model()
            self._warmup()
            self._mt_call_count = 0

    def new_session(self, target_language="", source_language=None):
        return MlxLlmTranslationSimul(
            model_id=self._model_id, target_language=target_language or self._target_language,
            source_language=self._source_language if source_language is None else source_language,
            warmup=False, commit_mode=self._commit_mode, mass_threshold=self._mass_threshold,
            simul_soft_max_s=self._simul_soft_max_s, simul_hard_max_s=self._simul_hard_max_s,
            calibration_file=self._calibration_file,
        )

    def _ensure_simul_model(self):
        model, tokenizer = self._ensure_model(self._config)
        if not self._verified_model:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self._config.repo, "config.json", revision=self._config.revision, local_files_only=True)
            config = json.loads(Path(path).read_text())
            quantization = config.get("quantization", config.get("quantization_config", {}))
            if quantization != self._calibration.quantization:
                raise ValueError("Model quantization differs from the MLX calibration")
            layers = model.model.layers
            if len(layers) != self._calibration.num_layers or any(
                layer.self_attn.n_heads != self._calibration.num_heads for layer in layers
            ):
                raise ValueError("Model dimensions differ from the MLX calibration")
            self._verified_model = True
        return model, tokenizer

    def _prompt_content(self, text):
        if self._prompt["kind"] == "structured_chat":
            return [{"type": "text", "source_lang_code": self._prompt["src"],
                     "target_lang_code": self._prompt["tgt"], "text": text}]
        return self._prompt["template"].format(target_lang=self._prompt["target_name"], text=text)

    def _translate_simul(self, source, committed):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        with self._MODEL_LOCK:
            if self._closed.is_set():
                raise RuntimeError("Translation session is closed")
            model, tokenizer = self._ensure_simul_model()
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": self._prompt_content(source)}],
                add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            start, end = source_span(tokenizer, prompt, source)
            stop = _placeholder_stop_check(tokenizer)
            tokens = []
            with capture_attention(model, self._calibration.heads, target_start=len(prompt_ids)) as capture:
                for chunk in stream_generate(
                    model, tokenizer, prompt=prompt_ids, max_tokens=self._config.max_tokens,
                    sampler=make_sampler(temp=self._config.temp, top_p=self._config.top_p, top_k=self._config.top_k),
                    logits_processors=make_logits_processors(repetition_penalty=self._config.repetition_penalty),
                ):
                    if self._closed.is_set():
                        raise RuntimeError("Translation session closed during generation")
                    tokens.append(chunk.token)
                    if stop is not None and stop(chunk):
                        break
                attention = snapshot_capture(capture)
            draft = dict(source=source, tokens=tuple(tokens), prompt_length=len(prompt_ids),
                         source_ids=tuple(prompt_ids[start:end]), src_start=start, src_end=end,
                         attention=attention)
            accepted = self._apply_draft(tokenizer, draft, committed)
            self._last_draft = draft
            return accepted

    def _apply_draft(self, tokenizer, draft, committed):
        frontier = committed_src_end_from_text(tokenizer, list(draft["source_ids"]), committed)
        count = apply_commit_policy(
            draft["attention"], self._calibration.heads, len(draft["tokens"]),
            draft["prompt_length"], draft["src_start"], draft["src_end"], frontier,
            mode=self._commit_mode, mass_threshold=self._mass_threshold,
        )
        # An incomplete byte token must remain held, never display U+FFFD.
        while count and tokenizer.decode(list(draft["tokens"][:count])).endswith("\ufffd"):
            count -= 1
        return _strip_hy_placeholder(tokenizer.decode(list(draft["tokens"][:count])))

    def _release_held(self, committed):
        with self._MODEL_LOCK:
            _, tokenizer = self._ensure_simul_model()
            return self._apply_draft(tokenizer, self._last_draft, committed)

    def _committed_text(self):
        separator = "" if self._source_language.startswith(("zh", "ja", "cmn", "yue")) else " "
        return separator.join(t.text.strip() for t in self._committed_simul)

    def _source_text(self):
        committed = self._committed_text()
        tail = self._tail.text.strip() if self._tail else ""
        if tail and self._committed_simul and self._tail.end <= self._committed_simul[-1].end:
            tail = ""
        separator = "" if self._source_language.startswith(("zh", "ja", "cmn", "yue")) else " "
        return separator.join(part for part in (committed, tail) if part)

    def _queue_final(self):
        if self._committed_simul:
            self._pending_finals.append((self._committed_text(), self._committed_simul[0].start,
                                         self._committed_simul[-1].end))
            self._committed_simul = []
        self._last_draft = None
        self._emitted_partial = ""
        self._tail = None

    def insert_tokens(self, items):
        for item in items:
            if isinstance(item, HypothesisTail):
                self._tail = item
            elif isinstance(item, ASRToken) and item.text.strip():
                self._committed_simul.append(item)
                duration = item.end - self._committed_simul[0].start
                if duration >= self._simul_hard_max_s or (item.has_punctuation() and duration >= self._simul_soft_max_s):
                    self._queue_final()

    def _buffer(self):
        if self._emitted_partial and self._committed_simul:
            return TimedText(start=self._committed_simul[0].start, end=self._committed_simul[-1].end,
                             text=self._emitted_partial)
        return self._last_buffer

    def _drain_finals(self):
        previous = self._buffer()
        result, buffer = super().process()
        if self.error:
            self._last_buffer = previous
            buffer = previous
        return result, buffer

    def process(self):
        if self._pending_finals:
            return self._drain_finals()
        source, committed = self._source_text(), self._committed_text()
        if not source or not committed:
            return None, self._buffer()
        started = time.perf_counter()
        try:
            if self._last_draft is not None and source == self._last_draft["source"]:
                accepted = self._release_held(committed)
            else:
                self._mt_call_count += 1
                accepted = self._translate_simul(source, committed)
            if accepted.startswith(self._emitted_partial):
                self._emitted_partial = accepted
            self.error = ""
        except Exception as exc:
            self.error = f"MLX simultaneous translation incomplete: {exc}"
        finally:
            self._mt_total_time_s += time.perf_counter() - started
        self._last_buffer = self._buffer()
        return None, self._last_buffer

    def validate_buffer_and_reset(self):
        self._last_buffer = self._buffer()
        self._queue_final()
        return self._drain_finals()
