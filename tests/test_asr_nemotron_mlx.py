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
