"""Bound pending work and wake blocked producers when a session closes."""

import asyncio
from time import perf_counter
from typing import Any, List, Union

import numpy as np

from whisperlivekit.timed_objects import ChangeSpeaker, Silence

SENTINEL = object()


class PipelineClosed(RuntimeError):
    pass


class PipelineOverloaded(RuntimeError):
    pass


class ProcessingQueue(asyncio.Queue):
    def __init__(self, name, *, max_samples=0, maxsize=256, timeout=30.0, on_overload=None):
        super().__init__(maxsize=maxsize)
        self.name = name
        self.max_samples = max_samples
        self.queued_samples = 0
        self.peak_samples = 0
        self.wait_seconds = 0.0
        self.timeout = timeout
        self.on_overload = on_overload
        self.closed = False

    @staticmethod
    def _samples(item):
        return item.size if isinstance(item, np.ndarray) else 0

    def _has_space(self, item):
        if self.closed:
            raise PipelineClosed("Audio session is closed.")
        return not super().full() and (
            not self.max_samples or self.queued_samples + self._samples(item) <= self.max_samples
        )

    def full(self):
        return super().full() or bool(self.max_samples and self.queued_samples >= self.max_samples)

    async def put(self, item):
        if self.max_samples and self._samples(item) > self.max_samples:
            raise ValueError("Audio chunk exceeds the processing queue capacity.")
        if self._has_space(item):
            self.put_nowait(item)
            return
        started = perf_counter()
        try:
            async with asyncio.timeout(self.timeout):
                while not self._has_space(item):
                    # Use Queue's producer waiters so get()/get_nowait() wake
                    # producers without polling or an extra task per audio chunk.
                    waiter = self._get_loop().create_future()
                    self._putters.append(waiter)
                    try:
                        await waiter
                    except BaseException:
                        waiter.cancel()
                        if waiter in self._putters:
                            self._putters.remove(waiter)
                        self._wakeup_next(self._putters)
                        raise
                self.put_nowait(item)
        except TimeoutError as exc:
            message = f"{self.name} backlog did not clear within {self.timeout:g} seconds."
            if self.on_overload:
                self.on_overload(message)
            raise PipelineOverloaded(message) from exc
        finally:
            self.wait_seconds += perf_counter() - started

    def put_nowait(self, item):
        if not self._has_space(item):
            raise asyncio.QueueFull
        # The weighted limit allows a zero-sized boundary after the last audio
        # chunk. Queue.put_nowait uses full(), so do the ordinary insertion here.
        self._put(item)
        self._unfinished_tasks += 1
        self._finished.clear()
        self._wakeup_next(self._getters)

    def _put(self, item):
        super()._put(item)
        self.queued_samples += self._samples(item)
        self.peak_samples = max(self.peak_samples, self.queued_samples)

    def _get(self):
        item = super()._get()
        self.queued_samples -= self._samples(item)
        # Producers can have different chunk sizes. A large waiting chunk must
        # not leave a smaller one asleep when enough room for it is available.
        while self._putters:
            self._wakeup_next(self._putters)
        return item

    async def get(self):
        if self.closed and self.empty():
            raise PipelineClosed("Audio session is closed.")
        return await super().get()

    def close(self):
        self.closed = True
        while not self.empty():
            self.get_nowait()
            self.task_done()
        for waiters in (self._putters, self._getters):
            while waiters:
                waiter = waiters.popleft()
                if not waiter.done():
                    waiter.set_exception(PipelineClosed("Audio session is closed."))


async def get_all_from_queue(
    queue: asyncio.Queue,
) -> Union[object, Silence, ChangeSpeaker, np.ndarray, List[Any]]:
    items: List[Any] = []

    first_item = await queue.get()
    queue.task_done()
    if first_item is SENTINEL:
        return first_item
    if isinstance(first_item, (Silence, ChangeSpeaker)):
        return first_item
    items.append(first_item)

    while True:
        if not queue._queue:
            break
        next_item = queue._queue[0]
        if next_item is SENTINEL:
            break
        if isinstance(next_item, (Silence, ChangeSpeaker)):
            break
        items.append(await queue.get())
        queue.task_done()
    if isinstance(items[0], np.ndarray):
        return np.concatenate(items)
    else: #translation
        return items
