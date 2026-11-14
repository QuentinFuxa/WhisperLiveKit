"""Streaming adapter scenarios without a model download.

The ``native`` fixture stands in for ``mlx_qwen3_asr.streaming`` and keeps the
0.4.x contract: a rolling window text that is re-decoded per chunk, a
``stable_text`` prefix that stays empty for the first ``unfixed_chunk_num``
chunks, and a finalize path that re-decodes the window with the pending tail.
Handing it a window below the mel floor raises, as the real
``compute_features()`` does.
"""
import asyncio
import sys
from time import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from whisperlivekit.asr_mlx_qwen3 import (
    MlxQwen3AsrOnlineProcessor,
    _append_only_delta,
)
from whisperlivekit.audio_processor import SENTINEL, AudioProcessor
from whisperlivekit.core import online_factory
from whisperlivekit.metrics_collector import SessionMetrics
from whisperlivekit.timed_objects import Silence, State

# compute_features() needs more than n_fft // 2 samples; the adapter keeps a
# margin over that floor.
MEL_FLOOR_SAMPLES = 400


@pytest.fixture
def native(monkeypatch):
    module = ModuleType('mlx_qwen3_asr.streaming')
    module.states = []
    module.calls = []
    module.script = []   # (window_text, stable_text) per successive decode
    module.finals = []   # window text each finish_streaming call re-decodes

    def init_streaming(**kwargs):
        state = SimpleNamespace(
            text='', stable_text='', window_text='', committed_text='',
            language=kwargs['language'], context=kwargs['context'],
            chunk_size_samples=int(kwargs['chunk_size_sec'] * 16_000),
            audio_accum=np.array([], dtype=np.float32),
            buffer=np.array([], dtype=np.float32),
        )
        module.states.append(state)
        return state

    def feed_audio(audio, state, *, model):
        module.calls.append(('feed', state.language, len(audio)))
        state.buffer = np.concatenate([state.buffer, np.asarray(audio, dtype=np.float32)])
        while len(state.buffer) >= state.chunk_size_samples:
            chunk = state.buffer[:state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples:]
            state.audio_accum = np.concatenate([state.audio_accum, chunk])
            if module.script:
                state.window_text, state.stable_text = module.script.pop(0)
                state.text = state.committed_text + state.window_text
        return state

    def finish_streaming(state, *, model):
        module.calls.append(('finish', state.language))
        decodable = len(state.audio_accum) + len(state.buffer)
        if decodable < MEL_FLOOR_SAMPLES:
            raise ValueError(
                f'Audio too short for STFT: {decodable} samples, need more than 200.'
            )
        state.audio_accum = np.concatenate([state.audio_accum, state.buffer])
        state.buffer = np.array([], dtype=np.float32)
        if module.finals:
            state.window_text = module.finals.pop(0)
        state.text = state.committed_text + state.window_text
        state.stable_text = state.text
        return state

    module.init_streaming = init_streaming
    module.feed_audio = feed_audio
    module.finish_streaming = finish_streaming
    monkeypatch.setitem(sys.modules, 'mlx_qwen3_asr', ModuleType('mlx_qwen3_asr'))
    monkeypatch.setitem(sys.modules, 'mlx_qwen3_asr.streaming', module)
    tokenizer = ModuleType('mlx_qwen3_asr.tokenizer')
    languages = {'fr': 'French', 'en': 'English', 'zh': 'Chinese'}
    tokenizer.canonicalize_language = lambda value: languages.get(value, value)
    tokenizer.language_is_known = lambda value: value in languages or value in languages.values()
    monkeypatch.setitem(sys.modules, tokenizer.__name__, tokenizer)
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
    native.script.append(('Bonjour le monde', 'Bonjour'))
    native.finals.append('Bonjour le monde.')
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
    native.finals.append('Dernière phrase sans ponctuation')
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
    native.finals += ['你好', '世界']
    committed, _ = chinese.start_silence()
    chinese.insert_audio_chunk(np.zeros(16000), 2)
    chinese.process_iter()
    final, _ = chinese.finish()
    assert text(committed + final) == '你好世界'
    assert automatic.get_buffer().text == ''
    with pytest.raises(ValueError, match='Unsupported Qwen3 ASR language'):
        online_factory(args, shared, language='sw')
    assert shared.language == 'French' and shared.hotwords == 'WLK'


def test_native_stable_prefix_is_committed_verbatim(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(32000), 2.0)
    online.process_iter()
    native.script.append(('fish and chips are ready', 'fish and chips'))
    online.insert_audio_chunk(np.zeros(32000), 4.0)
    committed, _ = online.process_iter()
    # Two decodes are needed for the agreed-prefix fallback, so only the native
    # stable prefix can produce this token.
    assert text(committed) == 'fish and chips'
    assert online.get_buffer().text == ' are ready'


def test_agreed_prefix_commits_before_the_native_split_starts(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(32000), 2.0)
    online.process_iter()
    native.script.append(('the quick brown fox jumps over', ''))
    online.insert_audio_chunk(np.zeros(32000), 4.0)
    assert text(online.process_iter()[0]) == ''
    native.script.append(('the quick brown fox jumps over the lazy dog', ''))
    online.insert_audio_chunk(np.zeros(32000), 6.0)
    committed, _ = online.process_iter()
    # Six units agreed across two decodes; the five-unit hold-back keeps one.
    assert text(committed) == 'the'
    assert online.get_buffer().text == ' quick brown fox jumps over the lazy dog'


# Upstream issue #31's pure-string reproduction pairs, in the order the library
# reports them: the stable_text of the previous chunk, the new full text, and the
# stable_text mlx-qwen3-asr 0.4.4 returns for it.
ISSUE_31_SPLIT = [
    # Case 1: the reported prefix is the previous one and is not a prefix of the
    # new text, so the adapter may only append what follows the shared part.
    ('我们去了Co', '我们去了科蒂纳。', '我们去了Co'),
    # Case 2: already-stable text is rewritten; the length fallback keeps the
    # longer value, which the new text no longer contains.
    ('甲乙', '甲丙丁戊己庚辛壬癸', '甲丙丁戊'),
    # Case 3: stable + unstable need not reproduce the text (the boundary
    # duplicates a character).
    ('甲乙丙', '甲乙丙丁戊己庚', '甲乙丙'),
    # Case 4: the joiner between stable and unstable is dropped.
    ('', 'one two three four five six', 'one'),
]


@pytest.mark.parametrize(('previous_stable', 'new_text', 'candidate', 'expected'), [
    # Only the part the client does not already hold; the revision's tail comes
    # with the next window instead of repeating the fragment.
    ('我们去了Co', '我们去了科蒂纳。', '我们去了Co', '科蒂纳。'),
    # The rewrite reaches the client once: '丙丁戊' is not appended twice.
    ('甲乙', '甲丙丁戊己庚辛壬癸', '甲丙丁戊', '丙丁戊己庚辛壬癸'),
    ('甲乙丙', '甲乙丙丁戊己庚', '甲乙丙', '丁戊己庚'),
    ('', 'one two three four five six', 'one', 'one two three four five six'),
])
def test_issue_31_candidates_are_absorbed_as_appends(
    previous_stable, new_text, candidate, expected
):
    source, first = _append_only_delta(previous_stable, candidate)
    _, second = _append_only_delta(source, new_text)
    assert first + second == expected


def test_appending_never_retracts_committed_text():
    # Clients already hold the longer text, so a shorter candidate adds nothing
    # and the source stays put; the text can come back without being repeated.
    source, delta = _append_only_delta('fish and chips', 'fish and')
    assert (source, delta) == ('fish and chips', '')
    assert _append_only_delta(source, 'fish and chips.') == ('fish and chips.', '.')


def test_issue_31_revision_neither_raises_nor_repeats_at_a_pause(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(32000), 2.0)
    online.process_iter()
    native.script.append(('我们去了科蒂纳', '我们去了Co'))
    online.insert_audio_chunk(np.zeros(32000), 4.0)
    committed, _ = online.process_iter()
    assert text(committed) == '我们去了Co'
    assert online.get_buffer().text == '科蒂纳'
    native.finals.append('我们去了科蒂纳。')
    tokens, _ = online.start_silence()
    assert text(committed + tokens) == '我们去了Co科蒂纳。'
    assert online.get_buffer().text == ''


def test_issue_31_rewrite_is_absorbed_once(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(32000), 2.0)
    online.process_iter()
    native.script.append(('甲乙丙丁戊己庚', '甲乙'))
    online.insert_audio_chunk(np.zeros(32000), 4.0)
    committed, _ = online.process_iter()
    assert text(committed) == '甲乙'
    # The library rewrites the already-stable prefix (issue #31, case 2).
    native.script.append(('甲丙丁戊己庚辛壬癸', '甲丙丁戊'))
    online.insert_audio_chunk(np.zeros(32000), 6.0)
    committed += online.process_iter()[0]
    assert text(committed) == '甲乙丙丁戊'
    assert online.get_buffer().text == '己庚辛壬癸'
    native.finals.append('甲丙丁戊己庚辛壬癸')
    tokens, _ = online.start_silence()
    # '丙丁戊' reaches the client once even though two decodes report it.
    assert text(committed + tokens) == '甲乙丙丁戊己庚辛壬癸'


@pytest.mark.parametrize(('previous_stable', 'new_text', 'candidate'), ISSUE_31_SPLIT)
def test_issue_31_split_values_match_the_installed_library(
    previous_stable, new_text, candidate
):
    # The recorded candidates are the upstream reproduction; this fails if a
    # dependency bump changes the contract the repair is written against.
    streaming = pytest.importorskip('mlx_qwen3_asr.streaming')
    split = getattr(streaming, '_split_stable_unstable', None)
    if split is None:
        pytest.skip('the installed mlx-qwen3-asr has no _split_stable_unstable')
    stable, _ = split(previous_stable, new_text)
    assert stable == candidate


def test_pause_before_a_full_chunk_does_not_raise(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(160), 0.01)
    tokens, end_time = online.start_silence()
    assert tokens == [] and end_time == pytest.approx(0.01)
    assert online.finish()[0] == []


def test_short_tail_pause_against_the_installed_library():
    # The real library buffers a sub-chunk tail and finish_streaming() would try
    # to decode it. The sentinel model in shared_asr() cannot load weights, so a
    # missing guard fails here instead of downloading anything.
    pytest.importorskip('mlx_qwen3_asr.streaming')
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(160, dtype=np.float32), 0.01)
    assert online.start_silence() == ([], pytest.approx(0.01))
    assert online._state is None


@pytest.mark.asyncio
async def test_pause_with_a_short_tail_does_not_raise(native):
    processor = _audio_processor(MlxQwen3AsrOnlineProcessor(shared_asr()))
    await _drive(processor, [
        np.zeros(160, dtype=np.float32),
        Silence(start=0.01, is_starting=True),
    ])
    assert processor.processing_error is None
    assert processor.state.tokens == []


@pytest.mark.asyncio
async def test_pause_in_audio_commits_each_sentence_once(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    processor = _audio_processor(online)
    native.script += [
        ('Bonjour le monde', ''),
        ('Bonjour le monde.', 'Bonjour'),
        ('Bonjour le monde. Il fait beau.', 'Bonjour le monde.'),
        ('Comment allez-vous?', ''),
        ('Comment allez-vous? Très bien.', 'Comment allez-vous?'),
    ]
    native.finals += ['Bonjour le monde. Il fait beau.', 'Comment allez-vous? Très bien.']
    await _drive(processor, [
        np.zeros(2 * 16000, dtype=np.float32),
        np.zeros(2 * 16000, dtype=np.float32),
        np.zeros(2 * 16000, dtype=np.float32),
        Silence(start=6.0, is_starting=True),
        Silence(start=6.0, end=7.5, duration=1.5, has_ended=True),
        np.zeros(2 * 16000, dtype=np.float32),
        np.zeros(2 * 16000, dtype=np.float32),
    ])

    assert processor.processing_error is None
    assert text(processor.state.tokens) == (
        'Bonjour le monde. Il fait beau. Comment allez-vous? Très bien.'
    )
    assert processor.state.buffer_transcription.text == ''


def _audio_processor(transcription) -> AudioProcessor:
    processor = object.__new__(AudioProcessor)
    processor.args = SimpleNamespace(asr_coalesce_min_s=0.0, diarization=False)
    processor.pause_segmentation_seconds = 0.0
    processor.transcription = transcription
    processor.transcription_queue = asyncio.Queue()
    processor.translation_queue = None
    processor.translation = None
    processor.translate_on_complete = False
    processor._pending_translation_tokens = []
    processor.diarization_queue = None
    processor.metrics = SessionMetrics()
    processor.state = State()
    processor.lock = asyncio.Lock()
    processor.sample_rate = 16_000
    processor.beg_loop = time()
    processor.sep = transcription.sep
    processor.is_stopping = False
    processor._any_asr_output = False
    processor._silent_backend_warned = False
    processor._last_asr_prov = ''
    processor.processing_error = None
    processor._prune_state_tokens = lambda: None
    return processor


async def _drive(processor, items):
    for item in items:
        await processor.transcription_queue.put(item)
    await processor.transcription_queue.put(SENTINEL)
    await processor.transcription_processor()


def test_case_only_revision_at_segment_start_does_not_duplicate():
    """The maintainer's trace: only the first letter changed case.

    source_before = 'là il y a eu tel'
    candidate     = 'Là il y a eu tel second de silence et je repars'
    """
    source, delta = _append_only_delta(
        "là il y a eu tel", "Là il y a eu tel second de silence et je repars"
    )
    assert source == "Là il y a eu tel second de silence et je repars"
    assert delta.strip() == "second de silence et je repars"
    assert "Là" not in delta


def test_case_only_revision_keeps_no_duplicate_on_fleurs_fr_1935():
    source, delta = _append_only_delta(
        "cependant ce n'est pas vrai bien",
        "Cependant ce n'est pas vrai bien qu'il y ait une raison",
    )
    assert delta.strip() == "qu'il y ait une raison"


def test_rewrite_inside_a_word_reconstructs_instead_of_duplicating():
    # The source is cut inside "petit"; the tail completes it, so what clients
    # hold plus the delta must equal the candidate, with nothing repeated.
    held = "c'est le plus pet"
    source, delta = _append_only_delta(held, "C'est le plus petit pays du monde")
    assert (held + delta).casefold() == "c'est le plus petit pays du monde".casefold()
