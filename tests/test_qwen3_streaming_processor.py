"""Qwen3StreamingOnlineProcessor contract tests with a scripted fake backend."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from whisperlivekit.core import online_factory  # noqa: E402
from whisperlivekit.qwen3_streaming.online import Qwen3StreamingOnlineProcessor  # noqa: E402
from whisperlivekit.timed_objects import Transcript  # noqa: E402

SR = 16_000


class FakeMelExtractor:
    """One mel frame per 160 samples, no holdback (keeps tests simple)."""

    def __init__(self):
        self.reset_calls = 0

    def append(self, audio):
        frames = len(audio) // 160
        if frames == 0:
            return None
        return torch.zeros(1, frames, 128)

    def flush(self):
        return None

    def reset(self):
        self.reset_calls += 1


class ScriptedStreamer:
    """Replays canned committed/unstable text per append_mel_chunk call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.events = []
        self.finalized = False

    def append_mel_chunk(self, chunk, is_flush=False):
        committed, unstable = (
            self.script[min(self.calls, len(self.script) - 1)]
            if self.script
            else ("", "")
        )
        self.calls += 1
        event = {"committed": committed, "unstable": unstable}
        self.events.append(event)
        return event

    def finalize(self, finalize_mode="latest"):
        self.finalized = True
        committed = self.script[-1][0] if self.script else ""
        unstable = self.script[-1][1] if self.script else ""
        final_text = (committed + " " + unstable).strip()
        return SimpleNamespace(final_text=final_text)


class FakeASR:
    sep = ""
    SAMPLING_RATE = SR

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.streamers = []
        self.original_language = "en"
        self.chunk_sec = 2.0
        self.device = torch.device("cpu")
        self.right_context_frames = 0
        self.n_mels = 128
        self.decode_lock = threading.Lock()
        self.mel_extractors = []

    def build_streamer(self, language=None):
        script = self._scripts.pop(0) if self._scripts else []
        streamer = ScriptedStreamer(script)
        self.streamers.append(streamer)
        return streamer

    def new_mel_extractor(self):
        extractor = FakeMelExtractor()
        self.mel_extractors.append(extractor)
        return extractor


def feed_seconds(processor, seconds, end_time):
    processor.insert_audio_chunk(np.zeros(int(seconds * SR), dtype=np.float32), end_time)


def test_committed_words_become_timestamped_tokens():
    asr = FakeASR([[("hello world", "and more")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 3.0, 3.0)
    tokens, end = processor.process_iter()

    assert [t.text for t in tokens] == ["hello", " world"]
    for token in tokens:
        assert token.start is not None and token.end is not None
        assert 0.0 <= token.start <= token.end <= 3.0
        assert token.detected_language == "en"
    assert end == 3.0


def test_timestamps_are_monotonic_across_updates():
    asr = FakeASR([[("one", ""), ("one two three", ""), ("one two three four five", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    all_tokens = []
    for i in range(1, 4):
        feed_seconds(processor, 3.0, 3.0 * i)
        tokens, _ = processor.process_iter()
        all_tokens.extend(tokens)

    assert [t.text for t in all_tokens] == ["one", " two", " three", " four", " five"]
    starts = [t.start for t in all_tokens]
    ends = [t.end for t in all_tokens]
    assert starts == sorted(starts)
    assert all(s <= e for s, e in zip(starts, ends))
    assert ends[-1] <= 9.0


def test_no_decode_until_enough_audio_pending():
    asr = FakeASR([[("hello", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 0.5, 0.5)  # below chunk_sec=2.0
    tokens, _ = processor.process_iter()
    assert tokens == []
    assert asr.streamers[0].calls == 0

    feed_seconds(processor, 2.0, 2.5)
    processor.process_iter()
    assert asr.streamers[0].calls == 1


def test_self_pacing_waits_longer_after_slow_decode():
    asr = FakeASR([[("hello", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)
    processor._last_decode_duration = 10.0  # pretend the last decode took 10s

    feed_seconds(processor, 3.0, 3.0)  # above chunk_sec but below 1.2 * 10s
    tokens, _ = processor.process_iter()
    assert tokens == []
    assert asr.streamers[0].calls == 0

    feed_seconds(processor, 10.0, 13.0)
    processor.process_iter()
    assert asr.streamers[0].calls == 1


def test_rollover_revision_never_retracts_emitted_words():
    # Second update REVISES the first committed word ("hello" -> "hullo").
    asr = FakeASR([[("hello world", ""), ("hullo world again", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 3.0, 3.0)
    first = processor.process_iter()[0]
    feed_seconds(processor, 3.0, 6.0)
    second = processor.process_iter()[0]

    assert [t.text for t in first] == ["hello", " world"]
    # The revision of word 0 is dropped; only the new word is emitted.
    assert [t.text for t in second] == [" again"]


def test_get_buffer_returns_unstable_tail():
    asr = FakeASR([[("hello", "uncertain tail")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    assert isinstance(processor.get_buffer(), Transcript)
    assert processor.get_buffer().text == ""

    feed_seconds(processor, 3.0, 3.0)
    processor.process_iter()
    buffer = processor.get_buffer()
    assert buffer.text == "uncertain tail"
    assert buffer.start is not None and buffer.end == 3.0


def test_start_silence_flushes_and_resets():
    asr = FakeASR([[("hello world", "tail")], [("next utterance", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 3.0, 3.0)
    processor.process_iter()
    tokens, _ = processor.start_silence()

    # finalize emits the remaining words ("tail" beyond "hello world")
    assert [t.text for t in tokens] == [" tail"]
    assert asr.streamers[0].finalized
    assert len(asr.streamers) == 2  # fresh streamer for the next utterance
    assert asr.mel_extractors[0].reset_calls == 1
    assert processor.get_buffer().text == ""


def test_end_silence_shifts_subsequent_timestamps():
    asr = FakeASR([[("hello", ""), ("hello world", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 3.0, 3.0)
    first = processor.process_iter()[0]
    processor.end_silence(10.0, offset=3.0)
    feed_seconds(processor, 3.0, 16.0)
    second = processor.process_iter()[0]

    assert second[0].start >= first[-1].end + 10.0 - 3.0  # shifted past the silence


def test_finish_flushes_without_reset():
    asr = FakeASR([[("hello world", "tail")]])
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 3.0, 3.0)
    processor.process_iter()
    tokens, end = processor.finish()

    assert [t.text for t in tokens] == [" tail"]
    assert len(asr.streamers) == 1  # no new streamer
    assert end == 3.0


def test_process_iter_error_returns_empty():
    asr = FakeASR([[("hello", "")]])
    processor = Qwen3StreamingOnlineProcessor(asr)
    processor.mel = None  # force an exception inside the decode path

    feed_seconds(processor, 3.0, 3.0)
    tokens, end = processor.process_iter()
    assert tokens == []
    assert end == 3.0


def test_session_language_used_for_streamer_and_tokens():
    asr = FakeASR([[("bonjour", "")]])
    asr._session_language = "fr"
    processor = Qwen3StreamingOnlineProcessor(asr)

    feed_seconds(processor, 3.0, 3.0)
    tokens, _ = processor.process_iter()
    assert tokens[0].detected_language == "fr"


def test_online_factory_routing():
    args = SimpleNamespace(backend="qwen3-streaming")
    asr = FakeASR([[]])

    processor = online_factory(args, asr)

    assert isinstance(processor, Qwen3StreamingOnlineProcessor)
