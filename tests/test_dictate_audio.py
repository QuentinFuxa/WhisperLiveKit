"""Tests for dictation audio helpers."""

import numpy as np

from whisperlivekit.dictate.audio import has_speech, is_hallucination, to_pcm16


def test_to_pcm16_converts_and_clips():
    pcm = to_pcm16(np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32))

    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples.tolist() == [0, 16383, -32767, 32767]


def test_has_speech_false_for_silence():
    assert not has_speech(np.zeros(16000, dtype=np.float32))


def test_has_speech_true_for_short_loud_burst():
    audio = np.zeros(16000 * 3, dtype=np.float32)
    audio[16000:16000 + 1600] = 0.2

    assert has_speech(audio)


def test_is_hallucination_filler():
    assert is_hallucination("Thank you.")
    assert is_hallucination("you")
    assert is_hallucination(" ")


def test_is_hallucination_keeps_real_text():
    assert not is_hallucination("Please send the report by Friday.")


def test_is_hallucination_repeated_filler():
    assert is_hallucination("Thank you for watching. Thank you for watching. Thank you for watching.")


def test_is_hallucination_keeps_long_text_with_filler_phrase():
    assert not is_hallucination("Thank you for watching the kids yesterday, it really helped us out.")


def test_is_hallucination_keeps_short_real_dictation():
    for text in ["Yes.", "Okay.", "Hi.", "Meet me at the end of the hall.", "Take care of the bug.",
                 "Thanksgiving plans?", "Cheers, Bob.", "Please unsubscribe me."]:
        assert not is_hallucination(text), text
