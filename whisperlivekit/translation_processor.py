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


async def run_translation(queue, translation, state, lock) -> None:
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
