"""Streaming adapter scenarios without a model download."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.asr_mlx_qwen3 import MlxQwen3AsrOnlineProcessor
from whisperlivekit.core import online_factory


@pytest.fixture
def native(monkeypatch):
    module = ModuleType('mlx_qwen3_asr.streaming')
    module.states = []
    module.calls = []

    def init_streaming(**kwargs):
        state = SimpleNamespace(text='', stable_text='', language=kwargs['language'],
                                context=kwargs['context'], final='', updates=[])
        module.states.append(state)
        return state

    def feed_audio(audio, state, *, model):
        module.calls.append(('feed', state.language, len(audio)))
        if state.updates:
            state.text, state.stable_text = state.updates.pop(0)
        return state

    def finish_streaming(state, *, model):
        module.calls.append(('finish', state.language))
        state.text = state.final or state.text
        state.stable_text = state.text
        return state

    module.init_streaming = init_streaming
    module.feed_audio = feed_audio
    module.finish_streaming = finish_streaming
    monkeypatch.setitem(sys.modules, 'mlx_qwen3_asr', ModuleType('mlx_qwen3_asr'))
    monkeypatch.setitem(sys.modules, 'mlx_qwen3_asr.streaming', module)
    return module


def shared_asr():
    return SimpleNamespace(model=object(), model_id='test', language='French', hotwords='WLK',
                           chunk_size_sec=2, max_context_sec=30, finalization_mode='accuracy',
                           backend_choice='mlx-qwen3-asr')


def text(tokens):
    return ''.join(token.text for token in tokens)


def test_pause_eof_and_new_speaker_preserve_spaces_and_commit_once(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(32000), 2.0)
    assert native.calls == []  # Feeding PCM on the event loop does no inference.
    online.process_iter()
    native.states[-1].updates = [('Bonjour le monde', 'Bonjour')]
    native.states[-1].final = 'Bonjour le monde.'
    online.insert_audio_chunk(np.zeros(32000), 4.0)
    committed, _ = online.process_iter()
    assert text(committed) == 'Bonjour'
    assert online.get_buffer().text == ' le monde'
    tokens, _ = online.start_silence()
    committed += tokens
    assert online.finish()[0] == []
    assert online.get_buffer().text == ''
    online.end_silence(1, 4)
    online.insert_audio_chunk(np.zeros(16000), 6.0)
    online.process_iter()
    native.states[-1].final = 'Dernière phrase sans ponctuation'
    tokens, _ = online.new_speaker()
    committed += tokens
    assert text(committed) == 'Bonjour le monde. Dernière phrase sans ponctuation'
    assert [(t.start, t.end) for t in committed] == [(0, 4), (4, 4), (5, 6)]
    assert online.finish()[0] == []


def test_sessions_keep_their_language_context_and_cjk_boundaries(native):
    args = SimpleNamespace(backend='mlx-qwen3-asr', backend_policy='simulstreaming')
    shared = shared_asr()
    chinese = online_factory(args, shared, language='zh', context='术语')
    automatic = online_factory(args, shared, language='auto', context='terms')
    for session in (chinese, automatic):
        session.insert_audio_chunk(np.zeros(16000), 1)
        session.process_iter()
    assert [(s.language, s.context) for s in native.states] == [
        ('Chinese', 'WLK\n术语'), (None, 'WLK\nterms')]
    native.states[0].final = '你好'
    committed, _ = chinese.start_silence()
    chinese.insert_audio_chunk(np.zeros(16000), 2)
    chinese.process_iter()
    native.states[-1].final = '世界'
    final, _ = chinese.finish()
    assert text(committed + final) == '你好世界'
    assert automatic.get_buffer().text == ''
    assert shared.language == 'French' and shared.hotwords == 'WLK'


def test_native_revision_fails_without_reemitting_a_confirmed_prefix(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(16000), 1)
    online.process_iter()
    native.states[-1].updates = [('Bonjour le monde', 'Bonjour')]
    online.insert_audio_chunk(np.zeros(16000), 2)
    confirmed, _ = online.process_iter()
    native.states[-1].final = 'Bonsoir le monde'
    with pytest.raises(RuntimeError, match='already committed'):
        online.finish()
    assert text(confirmed) == 'Bonjour'
    assert online._committed == 'Bonjour'
