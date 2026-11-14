"""Exercise PCM buffering, RNNT commits and session boundaries on the MLX CPU device."""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
pytest.importorskip('mlx_audio')

from whisperlivekit.asr_nemotron_mlx import NemotronMLXOnlineProcessor  # noqa: E402
from whisperlivekit.core import online_factory  # noqa: E402


@pytest.fixture
def shared(monkeypatch):
    from mlx_audio.stt.models.nemotron_asr import audio, streaming

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)

    class Mel:
        def __init__(self, config):
            self.samples = np.empty(0)

        def push(self, samples, *, final):
            self.samples = np.concatenate([self.samples, np.array(samples)])
            frames = len(self.samples) // 1280
            if final and len(self.samples) % 1280:
                frames += 1
            emitted = [self.samples[i*1280] for i in range(frames)]
            self.samples = self.samples[frames*1280:]
            return mx.array(emitted).reshape(1, frames, 1)

    class Encoder:
        def __init__(self, encoder, **kwargs):
            pass

        def push(self, mel, *, final):
            return [mel] if mel.shape[1] else []

        def materialize(self, *arrays):
            mx.eval(*arrays)

    class Model:
        thread_ids = set()
        encoder = None
        encoder_config = SimpleNamespace(subsampling_factor=8)
        preprocessor_config = SimpleNamespace(hop_length=160)
        prompt_dictionary = {'en-US': 0, 'zh-CN': 1, 'fr-FR': 2}
        vocabulary = ['<unk>', '▁Bonjour', '▁monde', '.', '你好', '世界']
        blank_id = 6
        max_symbols = 10

        def decoder(self, current, hidden):
            self.thread_ids.add(threading.get_ident())
            last = -1 if current is None else int(current.item())
            count = 1 if hidden is None else int(hidden[0].item()) + 1
            return mx.array([[[last]]]), (mx.array([count]), mx.array([0]))

        def joint(self, feature, decoded):
            token = int(feature.item())
            prediction = self.blank_id if int(decoded.item()) == token else token
            return mx.array([int(i == prediction) for i in range(self.blank_id + 1)])

        def apply_prompt(self, encoded, language):
            return encoded

    monkeypatch.setattr(audio, 'StreamingLogMelSpectrogram', Mel)
    monkeypatch.setattr(streaming, 'ConformerStreamingState', Encoder)
    yield SimpleNamespace(model=Model(), original_language='fr-FR', att_context=[56, 6],
                          backend_choice='nemotron-mlx-asr')
    mx.set_default_device(previous_device)


def test_stream_drains_partial_frame_at_pause_and_eof_without_repeating_words(shared):
    online = NemotronMLXOnlineProcessor(shared)
    online.insert_audio_chunk(np.ones(1280), 0.08)
    assert online._encoder is None  # No decoding on the PCM insertion path.
    first, _ = online.process_iter()
    online.insert_audio_chunk(np.full(640, 2), 0.12)
    assert online.process_iter()[0] == []
    tail, _ = online.start_silence()
    assert online.finish()[0] == []
    online.end_silence(1.0, 0.12)
    online.insert_audio_chunk(np.full(800, 3), 1.17)
    last, _ = online.finish()
    assert ''.join(t.text for t in first + tail + last) == ' Bonjour monde.'
    np.testing.assert_allclose([(t.start, t.end) for t in first + tail + last],
                               [(0, 0.08), (0.08, 0.12), (1.12, 1.17)])
    assert online.get_buffer().text == ''
    assert online.finish()[0] == []


def test_token_starts_are_populated_and_monotonic_before_finish(shared):
    # The transducer times come out of the greedy decode, so they exist while the
    # utterance is still open.
    shared.model.vocabulary = ['<unk>', '▁un', '▁deux', '▁trois', '▁quatre']
    shared.model.blank_id = 5
    online = NemotronMLXOnlineProcessor(shared)
    online.insert_audio_chunk(
        np.concatenate([np.full(1280, value) for value in (1, 2, 3, 4)]), 0.32)
    tokens, _ = online.process_iter()
    assert online._encoder is not None  # Mid-utterance: finish() never ran.
    assert [token.text for token in tokens] == [' un', ' deux', ' trois', ' quatre']
    starts = [token.start for token in tokens]
    ends = [token.end for token in tokens]
    assert all(start is not None for start in starts + ends)
    assert starts == sorted(starts), 'token starts must not go backwards'
    assert len(set(starts)) == len(starts), 'each frame must carry its own start time'
    assert ends == sorted(ends)
    np.testing.assert_allclose(starts, [0.0, 0.08, 0.16, 0.24])
    np.testing.assert_allclose(ends, [0.08, 0.16, 0.24, 0.32])


def test_sentence_boundary_keeps_one_space(shared):
    # Reproduces the real token sequence: the model closes an utterance with a
    # standalone space token and opens the next one with a space-prefixed word.
    # Joined verbatim that reads 'We.  We'.
    shared.model.vocabulary = ['<unk>', '▁', '▁We', '.']
    shared.model.blank_id = 4
    online = NemotronMLXOnlineProcessor(shared)
    online.insert_audio_chunk(
        np.concatenate([np.full(1280, 2), np.full(1280, 3), np.full(1280, 1)]), 0.24)
    first, _ = online.finish()
    online.insert_audio_chunk(np.concatenate([np.full(1280, 2), np.full(1280, 3)]), 0.40)
    second, _ = online.finish()
    assert ''.join(t.text for t in first) == ' We.'
    assert ''.join(t.text for t in second) == ' We.'
    assert ''.join(t.text for t in first + second) == ' We. We.'


def test_space_ending_line_survives_a_tokenless_finish(shared):
    # A trailing-silence chunk finishes with no tokens, so the line already
    # committed still ends in a space. Reading the trailing-space flag off that
    # empty chunk drops it, the next utterance keeps its leading space, and the
    # joined text reads 'We.  We'.
    shared.model.vocabulary = ['<unk>', '▁', '▁We', '.']
    shared.model.blank_id = 4
    online = NemotronMLXOnlineProcessor(shared)
    online.insert_audio_chunk(
        np.concatenate([np.full(1280, 2), np.full(1280, 3), np.full(1280, 1)]), 0.24)
    head, _ = online.process_iter()
    assert ''.join(t.text for t in head) == ' We. '
    assert online.finish()[0] == []  # Trailing-silence chunk commits no token.
    online.insert_audio_chunk(
        np.concatenate([np.full(1280, 1), np.full(1280, 2), np.full(1280, 3)]), 0.48)
    tail, _ = online.finish()
    assert [t.text for t in tail] == ['We', '.']
    joined = ''.join(t.text for t in head + tail)
    assert joined == ' We. We.'
    assert '  ' not in joined


def test_space_only_final_chunk_does_not_glue_the_next_utterance(shared):
    # A final chunk whose only committed token is the closing space returns nothing
    # after the trim. That space was never handed to the caller, so the next
    # utterance must keep its leading space; clearing the flag instead glues the two
    # lines into 'WeWe.'.
    shared.model.vocabulary = ['<unk>', '▁', 'We', '.']
    shared.model.blank_id = 4
    scripted = []

    def joint(feature, decoded):
        prediction = scripted.pop(0)
        return mx.array([int(i == prediction) for i in range(shared.model.blank_id + 1)])

    shared.model.joint = joint
    online = NemotronMLXOnlineProcessor(shared)
    scripted.extend(value for piece in [2] for value in (piece, shared.model.blank_id))
    online.insert_audio_chunk(np.zeros(1280, dtype=np.float32), 0.16)
    head, _ = online.process_iter()
    scripted.extend(value for piece in [1] for value in (piece, shared.model.blank_id))
    online.insert_audio_chunk(np.zeros(1280, dtype=np.float32), 0.32)
    tail, _ = online.finish()
    assert [t.text for t in head] == ['We']
    assert tail == []  # Only the closing space token arrived, and it was trimmed.
    scripted.extend(value for piece in [1, 2, 3] for value in (piece, shared.model.blank_id))
    online.insert_audio_chunk(np.zeros(1280 * 3, dtype=np.float32), 0.80)
    second, _ = online.finish()
    joined = ''.join(t.text for t in head + tail + second)
    assert joined == 'We We.'
    assert 'WeWe' not in joined


def test_utterance_start_collapses_repeated_space_tokens(shared):
    # The next utterance's first decode can emit two space-only pieces before its
    # first word (`first_tokens=[' ', ' ', 'I']`), a seam inside one call that the
    # previous utterance's trailing trim cannot reach: joined verbatim that reads
    # 'We.  We'. The separator space must survive as exactly one.
    shared.model.vocabulary = ['<unk>', '▁', 'We', '.']
    shared.model.blank_id = 4
    scripted = []

    def joint(feature, decoded):
        prediction = scripted.pop(0)
        return mx.array([int(i == prediction) for i in range(shared.model.blank_id + 1)])

    shared.model.joint = joint
    online = NemotronMLXOnlineProcessor(shared)
    utterances = []
    for pieces in ([2, 3], [1, 1, 2, 3], [1, 1, 2, 3]):
        # One mel frame per scripted piece, plus a blank frame to advance past it.
        scripted.extend(value for piece in pieces for value in (piece, shared.model.blank_id))
        online.insert_audio_chunk(np.zeros(1280 * len(pieces), dtype=np.float32), 0.16 * len(pieces))
        utterances.append(online.finish()[0])
    first, second, third = utterances
    assert [t.text for t in first] == ['We', '.']
    assert [t.text for t in second] == [' ', 'We', '.']
    assert [t.text for t in third] == [' ', 'We', '.']
    assert [t.text for t in first + second + third] == ['We', '.', ' ', 'We', '.', ' ', 'We', '.']
    joined = ''.join(t.text for t in first + second + third)
    assert joined == 'We. We. We.'
    assert '  ' not in joined


def test_intra_utterance_space_run_collapses_before_the_next_word(shared):
    # Mid-utterance the decoder emits a standalone space token and then a
    # space-prefixed word in the SAME batch ('the.  The'), a seam an utterance-start
    # rule never reaches. The run collapses to one space; the word keeps its letters.
    shared.model.vocabulary = ['<unk>', '▁', 'We', '.', '▁The']
    shared.model.blank_id = 5
    scripted = []

    def joint(feature, decoded):
        prediction = scripted.pop(0)
        return mx.array([int(i == prediction) for i in range(shared.model.blank_id + 1)])

    shared.model.joint = joint
    online = NemotronMLXOnlineProcessor(shared)
    pieces = [2, 3, 1, 4]
    scripted.extend(value for piece in pieces for value in (piece, shared.model.blank_id))
    online.insert_audio_chunk(np.zeros(1280 * len(pieces), dtype=np.float32), 0.16 * len(pieces))
    tokens, _ = online.process_iter()
    assert online._encoder is not None  # One decode batch, nothing finalised.
    assert [t.text for t in tokens] == ['We', '.', ' ', 'The']
    joined = ''.join(t.text for t in tokens)
    assert joined == 'We. The'
    assert '  ' not in joined


def test_space_run_across_decode_batches_collapses(shared):
    # The run can also straddle two batches of one utterance: a batch ends on a
    # standalone space token and the next opens with a space-prefixed word, which the
    # utterance-flush trim never sees. Joined verbatim that read 'We.  The'.
    shared.model.vocabulary = ['<unk>', '▁', 'We', '.', '▁The']
    shared.model.blank_id = 5
    scripted = []

    def joint(feature, decoded):
        prediction = scripted.pop(0)
        return mx.array([int(i == prediction) for i in range(shared.model.blank_id + 1)])

    shared.model.joint = joint
    online = NemotronMLXOnlineProcessor(shared)
    scripted.extend(value for piece in [2, 3, 1] for value in (piece, shared.model.blank_id))
    online.insert_audio_chunk(np.zeros(1280 * 3, dtype=np.float32), 0.48)
    head, _ = online.process_iter()
    assert ''.join(t.text for t in head) == 'We. '
    scripted.extend(value for piece in [1, 4] for value in (piece, shared.model.blank_id))
    online.insert_audio_chunk(np.zeros(1280 * 2, dtype=np.float32), 0.80)
    tail, _ = online.finish()
    assert ''.join(t.text for t in tail) == 'The'
    joined = ''.join(t.text for t in head + tail)
    assert joined == 'We. The'
    assert '  ' not in joined


def test_interleaved_sessions_have_separate_decoder_state_and_language(shared):
    args = SimpleNamespace(backend='nemotron-mlx-asr', backend_policy='simulstreaming')
    chinese = online_factory(args, shared, language='zh')
    french = online_factory(args, shared, language='fr')
    automatic = online_factory(args, shared, language='auto')
    chinese.insert_audio_chunk(np.full(1280, 4), 0.08)
    fr_audio = np.concatenate([np.ones(1280), np.full(1280, 2)])
    french.insert_audio_chunk(fr_audio, 0.16)
    with ThreadPoolExecutor(max_workers=2) as callers:
        zh_result = callers.submit(chinese.process_iter)
        fr_result = callers.submit(french.process_iter)
        zh_head, _ = zh_result.result()
        fr_tokens, _ = fr_result.result()
    chinese.insert_audio_chunk(np.full(800, 5), 0.13)
    zh_tail, _ = chinese.new_speaker()
    assert ''.join(t.text for t in zh_head + zh_tail) == '你好世界'
    assert ''.join(t.text for t in fr_tokens) == ' Bonjour monde'
    assert all(t.detected_language == 'zh-CN' for t in zh_head + zh_tail)
    assert all(t.detected_language == 'fr-FR' for t in fr_tokens)
    assert automatic.language is None and shared.original_language == 'fr-FR'
    assert french.finish()[0] == []
    # A lock alone allows calls to migrate between caller threads, which breaks
    # cached MLX streams. Both sessions must keep the same model thread.
    assert len(shared.model.thread_ids) == 1
    with pytest.raises(ValueError, match='not supported'):
        online_factory(args, shared, language='not-a-language')
    with pytest.raises(ValueError, match='context'):
        online_factory(args, shared, context='unsupported terminology prompt')
