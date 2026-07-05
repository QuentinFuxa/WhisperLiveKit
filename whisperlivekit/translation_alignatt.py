"""AlignAtt translation backend: WebSocket client for alignatt-mt-server.

The sidecar (an `Alignatt4LLM <https://github.com/QuentinFuxa/Alignatt4LLM>`_
``alignatt-mt-server`` process, vLLM on CUDA) owns the MT model and the
AlignAtt commit policy; this module is a thin client implementing the same
duck-typed contract as ``nllw.OnlineTranslation``:

- ``insert_tokens(items)``: committed ASRTokens (plus, opted-in,
  ``HypothesisTail`` snapshots of the unstable ASR tail);
- ``process()`` -> ``(Translation | None, TimedText buffer)``: called from a
  worker thread, sends the current utterance state and returns finalized
  utterance translations as validated segments and the streamed AlignAtt
  acceptance as the (stable, append-only) translation buffer;
- ``validate_buffer_and_reset()``: silence/speaker boundary;
- ``insert_silence(duration)``: no-op.

The smart part is upstream commitment mapping: committed words are sent with
their timestamps (accessible to AlignAtt), the unstable tail without
(inaccessible: the model drafts over it but cannot commit against it). When
the ASR later commits tail words, the sidecar releases the held target
tokens from its cached draft without a new MT call, so translation commit
latency tracks ASR commit latency instead of adding to it.

If the sidecar is unreachable the session keeps transcribing: translation
output stays empty, one warning is logged, and the client retries with
backoff. Reconnects resume append-only via ``accepted_target_prefix``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from whisperlivekit.timed_objects import ASRToken, HypothesisTail, TimedText, Translation

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# Latency presets: client behavior + whitelisted per-session server overrides.
# "low" pairs naturally with a low-holdback ASR preset (e.g.
# --qwen3-streaming-hold-back-words 2); see docs/translation-alignatt.md.
LATENCY_PRESETS = {
    "quality": {"tail": False, "overrides": {"translation_alignatt_hold_back_target_units": 1}},
    "balanced": {"tail": True, "overrides": {}},
    "low": {"tail": True, "overrides": {"translation_alignatt_hold_back_target_units": 0,
                                        "translation_alignatt_min_emit_target_units": 0}},
}

_MIN_PARTIAL_INTERVAL_SECONDS = 0.5
_PACING = 1.2
_RECONNECT_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 10.0, 30.0)
_CALL_TIMEOUT_SECONDS = 15.0
_FIRST_CALL_TIMEOUT_SECONDS = 120.0  # sidecar warmup / model load


class AlignAttRemoteEngine:
    """Server-wide handle: connection settings shared by all sessions."""

    def __init__(self, *, url: str, source_language: str, preset: str | None,
                 latency: str, context_text: str = ""):
        self.url = url
        self.source_language = source_language
        self.preset = preset
        self.latency = latency if latency in LATENCY_PRESETS else "balanced"
        self.context_text = context_text

    def new_session(self, target_language: str) -> "AlignAttTranslationClient":
        return AlignAttTranslationClient(self, target_language)


@dataclass
class _Utterance:
    tokens: List[ASRToken] = field(default_factory=list)

    @property
    def start(self) -> Optional[float]:
        return self.tokens[0].start if self.tokens else None

    @property
    def end(self) -> Optional[float]:
        return self.tokens[-1].end if self.tokens else None

    def words_payload(self) -> List[List[Any]]:
        return [
            [
                token.text.strip(),
                None if token.start is None else float(token.start) * 1000.0,
                None if token.end is None else float(token.end) * 1000.0,
            ]
            for token in self.tokens
            if token.text and token.text.strip()
        ]


class AlignAttTranslationClient:
    """One WebSocket session against alignatt-mt-server (sync, thread-driven)."""

    def __init__(self, engine: AlignAttRemoteEngine, target_language: str):
        self.engine = engine
        self.target_language = target_language
        preset = LATENCY_PRESETS[engine.latency]
        self.wants_hypothesis_tail: bool = bool(preset["tail"])
        self._overrides = dict(preset["overrides"])

        self._ws = None
        self._connected_once = False
        self._down_warned = False
        self._retry_at = 0.0
        self._retry_stage = 0

        self._utterance = _Utterance()
        self._pending_finals: List[_Utterance] = []
        self._tail: Optional[HypothesisTail] = None
        self._clock_ms = 0.0
        self._seq = 0
        self._utterance_id = 0
        self._emitted_partial = ""
        self._last_translation_end: Optional[float] = None
        self._history: List[List[str]] = []

        self._last_call_started = 0.0
        self._last_call_duration = 0.0
        self._committed_since_last_send = False

    # ------------------------------------------------------------------
    # duck-typed contract (mirrors nllw.OnlineTranslation)
    # ------------------------------------------------------------------

    def insert_tokens(self, items: List[Any]) -> None:
        for item in items:
            if isinstance(item, HypothesisTail):
                self._tail = item
                if item.end is not None:
                    self._clock_ms = max(self._clock_ms, float(item.end) * 1000.0)
                continue
            if not isinstance(item, ASRToken):
                continue
            self._utterance.tokens.append(item)
            self._committed_since_last_send = True
            if item.end is not None:
                self._clock_ms = max(self._clock_ms, float(item.end) * 1000.0)
            if item.has_punctuation():
                self._pending_finals.append(self._utterance)
                self._utterance = _Utterance()
                self._tail = None

    def process(self) -> Tuple[Optional[Translation], TimedText]:
        try:
            return self._process_inner()
        except Exception as exc:  # translation must never kill the session
            self._mark_down(exc)
            return None, self._buffer()

    def validate_buffer_and_reset(self) -> Tuple[Translation, TimedText]:
        """Silence / speaker-change boundary: close the open utterance now.

        Returns the already-streamed partial acceptance as the validated
        segment (it was on screen; retracting it would violate append-only).
        The utterance is queued as a final so the sidecar rolls its context;
        the quality pass replaces nothing at this boundary in v1.
        """
        validated = Translation(
            start=self._segment_start(self._utterance.start),
            end=self._utterance.end or self._segment_start(None),
            text=self._emitted_partial,
        )
        if self._utterance.tokens:
            self._pending_finals.append(self._utterance)
            self._utterance = _Utterance()
        self._tail = None
        self._emitted_partial = ""
        if validated.text:
            self._last_translation_end = validated.end
        return validated, TimedText()

    def insert_silence(self, duration: float) -> None:
        return None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _segment_start(self, fallback: Optional[float]) -> float:
        if self._last_translation_end is not None:
            return self._last_translation_end
        return fallback if fallback is not None else 0.0

    def _buffer(self) -> TimedText:
        if not self._emitted_partial:
            return TimedText()
        return TimedText(
            start=self._segment_start(self._utterance.start),
            end=self._utterance.end,
            text=self._emitted_partial,
        )

    def _mark_down(self, exc: Exception) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        delay = _RECONNECT_BACKOFF_SECONDS[
            min(self._retry_stage, len(_RECONNECT_BACKOFF_SECONDS) - 1)
        ]
        self._retry_stage += 1
        self._retry_at = time.monotonic() + delay
        if not self._down_warned:
            logger.warning(
                "AlignAtt MT sidecar unavailable (%s: %s); transcription continues "
                "without translation, retrying every few seconds.",
                type(exc).__name__, exc,
            )
            self._down_warned = True
        else:
            logger.debug("AlignAtt sidecar still down: %s", exc)

    def _ensure_connection(self) -> bool:
        if self._ws is not None:
            return True
        if time.monotonic() < self._retry_at:
            return False
        try:
            from websockets.sync.client import connect

            ws = connect(self.engine.url, open_timeout=5.0)
            init = {
                "type": "init",
                "protocol_version": PROTOCOL_VERSION,
                "source_lang": self.engine.source_language,
                "target_lang": self.target_language,
                "overrides": self._overrides,
            }
            if self.engine.preset:
                init["preset"] = self.engine.preset
            if self.engine.context_text:
                init["context_text"] = self.engine.context_text
            if self._history:
                init["history"] = self._history[-8:]
            if self._emitted_partial:
                init["accepted_target_prefix"] = self._emitted_partial
            ws.send(json.dumps(init))
            response = json.loads(ws.recv(timeout=_FIRST_CALL_TIMEOUT_SECONDS
                                          if not self._connected_once
                                          else _CALL_TIMEOUT_SECONDS))
            if response.get("type") != "init_ok":
                ws.close()
                raise RuntimeError(
                    f"{response.get('code', 'error')}: {response.get('message', response)}"
                    + (f" (supported: {', '.join(response['supported'])})"
                       if response.get("supported") else "")
                )
            self._ws = ws
            self._connected_once = True
            self._retry_stage = 0
            if self._down_warned:
                logger.info("AlignAtt MT sidecar reconnected.")
                self._down_warned = False
            return True
        except Exception as exc:
            self._mark_down(exc)
            return False

    def _exchange(self, update: dict, *, timeout: float) -> Optional[dict]:
        assert self._ws is not None
        self._seq += 1
        update["seq"] = self._seq
        self._ws.send(json.dumps(update))
        # Coalescing on the server means some seqs are skipped; read until
        # ours (or newer) arrives.
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no response to seq {self._seq}")
            response = json.loads(self._ws.recv(timeout=remaining))
            if response.get("type") == "error":
                if response.get("code") == "processing_failed":
                    logger.warning("AlignAtt sidecar dropped an update: %s",
                                   response.get("message"))
                    return None
                raise RuntimeError(f"{response.get('code')}: {response.get('message')}")
            if response.get("type") != "translation":
                continue
            if int(response.get("seq") or 0) >= self._seq:
                return response

    def _process_inner(self) -> Tuple[Optional[Translation], TimedText]:
        if not self._ensure_connection():
            return None, self._buffer()

        validated: Optional[Translation] = None

        # 1. Finals first (punctuation boundaries and silence rollovers).
        while self._pending_finals:
            utterance = self._pending_finals[0]
            response = self._call(
                utterance,
                tail=None,
                is_final=True,
                timeout=_CALL_TIMEOUT_SECONDS,
            )
            if response is None:
                return validated, self._buffer()
            self._pending_finals.pop(0)
            self._utterance_id += 1
            final_text = (response.get("committed_text") or "").strip()
            source_text = " ".join(
                token.text.strip() for token in utterance.tokens if token.text
            ).strip()
            if final_text:
                self._history.append([source_text, final_text])
                self._history = self._history[-8:]
                segment = Translation(
                    start=self._segment_start(utterance.start),
                    end=utterance.end,
                    text=final_text,
                )
                self._last_translation_end = utterance.end
                self._emitted_partial = ""
                # One validated segment per process() call keeps the contract
                # simple; remaining finals go out on the next call.
                return segment, self._buffer()
            self._emitted_partial = ""

        # 2. Partial update for the open utterance (paced).
        has_content = bool(self._utterance.tokens) or bool(
            self.wants_hypothesis_tail and self._tail and (self._tail.text or "").strip()
        )
        if not has_content:
            return validated, self._buffer()
        now = time.monotonic()
        due_after = max(_MIN_PARTIAL_INTERVAL_SECONDS, _PACING * self._last_call_duration)
        if not self._committed_since_last_send and (now - self._last_call_started) < due_after:
            return validated, self._buffer()

        tail = self._tail if self.wants_hypothesis_tail else None
        response = self._call(
            self._utterance,
            tail=tail,
            is_final=False,
            timeout=_CALL_TIMEOUT_SECONDS if self._connected_once else _FIRST_CALL_TIMEOUT_SECONDS,
        )
        if response is None:
            return validated, self._buffer()
        if response.get("final"):
            # Server-forced rollover (--max-utterance-words).
            self._utterance_id += 1
            final_text = (response.get("committed_text") or "").strip()
            utterance = self._utterance
            self._utterance = _Utterance()
            self._emitted_partial = ""
            if final_text:
                segment = Translation(
                    start=self._segment_start(utterance.start),
                    end=utterance.end,
                    text=final_text,
                )
                self._last_translation_end = utterance.end
                return segment, self._buffer()
            return validated, self._buffer()

        committed = (response.get("committed_text") or "").strip()
        if committed.startswith(self._emitted_partial):
            self._emitted_partial = committed
        return validated, self._buffer()

    def _call(self, utterance: _Utterance, *, tail: Optional[HypothesisTail],
              is_final: bool, timeout: float) -> Optional[dict]:
        update = {
            "type": "update",
            "utterance_id": self._utterance_id,
            "words": utterance.words_payload(),
            "clock_ms": self._clock_ms,
            "is_final": is_final,
        }
        if tail is not None and (tail.text or "").strip():
            update["tail"] = {
                "words": [[word, None, None] for word in tail.text.split()],
            }
        started = time.monotonic()
        self._last_call_started = started
        try:
            response = self._exchange(update, timeout=timeout)
        finally:
            self._last_call_duration = time.monotonic() - started
        self._committed_since_last_send = False
        return response
