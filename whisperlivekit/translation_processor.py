"""Consume translation boundaries and publish results into session state."""

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from whisperlivekit.processing_queue import SENTINEL, PipelineClosed, PipelineOverloaded, get_all_from_queue
from whisperlivekit.timed_objects import ChangeSpeaker, Silence, TranslationProgress

logger = logging.getLogger(__name__)

# Closing a translation interrupts inference. It must not queue behind the
# inference jobs it needs to stop. Remote socket closes have a bounded timeout.
_CLOSE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wlk-translation-close")


async def close_translation(translation) -> None:
    close = getattr(translation, "close", None)
    if close is not None:
        await asyncio.get_running_loop().run_in_executor(_CLOSE_EXECUTOR, close)


class _ProgressReader:
    """Per-session reader for the translation-progress display contract.

    MlxLlm backends declare ``provides_drafts`` (class attribute) and
    implement ``progress()``. Third-party backends (the nllw
    OnlineTranslation, the alignatt sidecar) predate the contract: they
    report "no drafts, no source text" honestly instead of being probed
    for private state. A ``progress()`` that RAISES is a backend bug:
    log once per session and disable provisional draft display for the
    session — a visible degradation, never a silent empty string.
    """

    def __init__(self, translation) -> None:
        self._translation = translation
        self._warned = False

    def provides_drafts(self) -> bool:
        # Public contract attribute; the default covers third-party backends.
        return getattr(self._translation, "provides_drafts", False)

    def read(self) -> Optional[TranslationProgress]:
        """The backend's progress, or None when it cannot be read (the
        one-shot warning has already fired). Finals do not need progress
        and still emit; provisional drafts are skipped."""
        progress = getattr(self._translation, "progress", None)
        if progress is None:
            return TranslationProgress()
        try:
            return progress()
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "Translation backend %s raised reading progress(); "
                    "provisional draft display is disabled for this session: %s",
                    type(self._translation).__name__, exc,
                )
            return None


async def run_translation(queue, translation, state, lock, event_tap=None) -> None:
    # dedupe: emit MT provisional only when the draft text changes — the release
    # path re-emits the cached draft every process() until the source grows;
    # identical provisionals in a row are display noise (5x repeats in the
    # zh-en capture).
    last_mt_prov = ""
    last_seen_count = 0
    progress_reader = _ProgressReader(translation)
    while True:
        item = None
        try:
            item = await get_all_from_queue(queue)
            new_translation = None
            new_translation_buffer = None

            if item is SENTINEL:
                finalize = getattr(translation, "finish", translation.validate_buffer_and_reset)
                new_translation, new_translation_buffer = await asyncio.to_thread(finalize)
            elif isinstance(item, Silence):
                if item.is_starting:
                    new_translation, new_translation_buffer = await asyncio.to_thread(
                        translation.validate_buffer_and_reset
                    )
                if item.has_ended:
                    translation.insert_silence(item.duration)
            elif isinstance(item, ChangeSpeaker):
                new_translation, new_translation_buffer = await asyncio.to_thread(
                    translation.validate_buffer_and_reset
                )
            else:
                translation.insert_tokens(item)
                new_translation, new_translation_buffer = await asyncio.to_thread(translation.process)

            if new_translation is not None or new_translation_buffer is not None:
                async with lock:
                    if new_translation is not None:
                        state.new_translation.append(new_translation)
                    if new_translation_buffer is not None:
                        state.new_translation_buffer = new_translation_buffer

            # caption events: finalized translation(s) + provisional draft.
            # The fresh flag and the provisional payload read the backend's
            # progress contract (see _ProgressReader) — never private state.
            if event_tap is not None:
                d = progress_reader.read()
                if d is not None:
                    advanced = d.mt_call_count > last_seen_count
                    last_seen_count = d.mt_call_count
                else:
                    advanced = False
                if new_translation is not None:
                    _items = new_translation if isinstance(new_translation, (list, tuple)) else [new_translation]
                    for _tr in _items:
                        _text = (getattr(_tr, "text", "") or "").strip()
                        if _text:
                            event_tap.translation_final(
                                getattr(_tr, "end", None) or state.end_buffer,
                                _text,
                            )
                elif new_translation_buffer is not None and d is not None:
                    # A backend can return a provisional buffer with no finalized
                    # translation (None). Forward the buffer so the display shows
                    # the provisional draft before the final arrives — but ONLY
                    # when the backend actually produces drafts (provides_drafts).
                    # Backends without the capability hold UNTRANSLATED source in
                    # their buffer; showing that as a "provisional translation"
                    # flashed raw source text in the MT row until the real final
                    # replaced it. (d is None — progress unreadable — also skips:
                    # the one-shot warning already fired.)
                    _prov_text = (new_translation_buffer.text or "").strip()
                    _boundary = isinstance(item, (Silence, ChangeSpeaker))
                    fresh_mt = False if _boundary else advanced
                    if _prov_text and progress_reader.provides_drafts() \
                            and _prov_text != last_mt_prov:
                        last_mt_prov = _prov_text
                        event_tap.translation_provisional(
                            d.source_end if d.source_end is not None else state.end_buffer,
                            _prov_text,
                            d.committed_text,
                            d.source_text,
                            bool(fresh_mt),
                        )
            if item is SENTINEL:
                break
        except (PipelineClosed, PipelineOverloaded):
            return
        except Exception as e:
            logger.warning(f"Exception in translation_processor: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            translation.error = f"Translation incomplete: {e}"
            if item is SENTINEL:
                break
    logger.info("Translation processor task finished.")
