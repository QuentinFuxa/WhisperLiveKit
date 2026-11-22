"""Consume translation boundaries and publish results into session state."""

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

from whisperlivekit.processing_queue import SENTINEL, PipelineClosed, PipelineOverloaded, get_all_from_queue
from whisperlivekit.timed_objects import ChangeSpeaker, Silence

logger = logging.getLogger(__name__)

# Closing a translation interrupts inference. It must not queue behind the
# inference jobs it needs to stop. Remote socket closes have a bounded timeout.
_CLOSE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wlk-translation-close")


async def close_translation(translation) -> None:
    close = getattr(translation, "close", None)
    if close is not None:
        await asyncio.get_running_loop().run_in_executor(_CLOSE_EXECUTOR, close)


async def run_translation(queue, translation, state, lock, event_tap=None) -> None:
    # dedupe: emit MT provisional only when the draft text changes — the release
    # path re-emits the cached draft every process() until the source grows;
    # identical provisionals in a row are display noise (5x repeats in the
    # zh-en capture).
    #
    # ``event_tap`` is None when nothing consumes the caption stream.
    #
    # ``provides_drafts`` is the capability the provisional gate keys off: a
    # backend whose buffer holds UNTRANSLATED source (any backend predating the
    # contract too — the default is False) must never have that buffer shown as
    # a draft. Backends without drafts need no change and never enter the path.
    provides_drafts = getattr(translation, "provides_drafts", False)
    last_mt_prov = ""
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
            if event_tap is not None:
                if new_translation is not None:
                    _items = new_translation if isinstance(new_translation, (list, tuple)) else [new_translation]
                    for _tr in _items:
                        _text = (getattr(_tr, "text", "") or "").strip()
                        if _text:
                            event_tap.translation_final(
                                getattr(_tr, "end", None) or state.end_buffer,
                                _text,
                            )
                elif new_translation_buffer is not None:
                    # A backend can return a provisional buffer with no finalized
                    # translation (None). Forward the buffer so the display shows
                    # the provisional draft before the final arrives — but ONLY
                    # when the backend actually produces drafts (provides_drafts),
                    # for the reason in the gate above.
                    _prov_text = (new_translation_buffer.text or "").strip()
                    if _prov_text and provides_drafts and _prov_text != last_mt_prov:
                        last_mt_prov = _prov_text
                        # same audio position the final falls back to: how far
                        # the ASR has processed, which is the source position
                        # this draft was produced against.
                        event_tap.translation_provisional(state.end_buffer, _prov_text)
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
