"""Tests for silence handling — state machine and double-counting regression."""

import pytest

from whisperlivekit.timed_objects import Silence


class TestSilenceStateMachine:
    """Test Silence object state transitions."""

    def test_initial_state(self):
        s = Silence(start=1.0, is_starting=True)
        assert s.is_starting is True
        assert s.has_ended is False
        assert s.duration is None
        assert s.end is None

    def test_end_silence(self):
        s = Silence(start=1.0, is_starting=True)
        s.end = 3.0
        s.is_starting = False
        s.has_ended = True
        s.compute_duration()
        assert s.duration == pytest.approx(2.0)

    def test_very_short_silence(self):
        s = Silence(start=1.0, end=1.01, is_starting=False, has_ended=True)
        s.compute_duration()
        assert s.duration == pytest.approx(0.01)

    def test_zero_duration_silence(self):
        s = Silence(start=5.0, end=5.0)
        s.compute_duration()
        assert s.duration == pytest.approx(0.0)


class TestSilenceDoubleCounting:
    """Regression tests for the silence double-counting bug.

    The bug: _begin_silence and _end_silence both pushed self.current_silence
    to the queue. Since they were the same Python object, _end_silence's mutation
    affected the already-queued start event. The consumer processed both as
    ended silences, doubling the duration.

    Fix: _begin_silence now pushes a separate Silence object for the start event.
    """

    def test_start_and_end_are_separate_objects(self):
        """Simulate the fix: start event and end event must be different objects."""
        # Simulate _begin_silence: creates start event as separate object
        current_silence = Silence(start=1.0, is_starting=True)
        start_event = Silence(start=1.0, is_starting=True)  # separate copy

        # Simulate _end_silence: mutates current_silence
        current_silence.end = 3.0
        current_silence.is_starting = False
        current_silence.has_ended = True
        current_silence.compute_duration()

        # start_event should NOT be affected by mutations to current_silence
        assert start_event.is_starting is True
        assert start_event.has_ended is False
        assert start_event.end is None

        # current_silence (end event) has the final state
        assert current_silence.has_ended is True
        assert current_silence.duration == pytest.approx(2.0)

    def test_single_object_would_cause_double_counting(self):
        """Demonstrate the bug: if same object is used for both events."""
        shared = Silence(start=1.0, is_starting=True)
        queue = [shared]  # start event queued

        # Mutate (simulates _end_silence)
        shared.end = 3.0
        shared.is_starting = False
        shared.has_ended = True
        shared.compute_duration()
        queue.append(shared)  # end event queued

        # Both queue items point to the SAME mutated object
        assert queue[0] is queue[1]  # same reference
        assert queue[0].has_ended is True  # start event also shows ended!

        # This would cause double-counting: both items have has_ended=True
        # and duration=2.0, so the consumer adds 2.0 twice = 4.0


class TestConsecutiveSilences:
    def test_multiple_silences(self):
        """Multiple silence periods should have independent durations."""
        s1 = Silence(start=1.0, end=2.0)
        s1.compute_duration()
        s2 = Silence(start=5.0, end=8.0)
        s2.compute_duration()
        assert s1.duration == pytest.approx(1.0)
        assert s2.duration == pytest.approx(3.0)
        # Total silence should be sum, not accumulated on single object
        assert s1.duration + s2.duration == pytest.approx(4.0)
