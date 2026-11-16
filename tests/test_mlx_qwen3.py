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
    MlxQwen3ASR,
    MlxQwen3AsrOnlineProcessor,
    _append_only_delta,
)
from whisperlivekit.audio_processor import SENTINEL, AudioProcessor
from whisperlivekit.config import WhisperLiveKitConfig
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


def _install_fake_loader(monkeypatch):
    """Replace the native loader so MlxQwen3ASR records the id it would load."""
    loaded = []

    def load_model(path_or_hf_repo, *args, **kwargs):
        loaded.append(path_or_hf_repo)
        return object(), None

    monkeypatch.setattr(sys.modules['mlx_qwen3_asr'], 'load_model', load_model, raising=False)
    return loaded


def test_model_size_selects_the_weights(native, monkeypatch):
    """`--model`/`model_size` picks the weights for this backend too.

    The benchmark runner passes it in kwargs while forbidding the model id in
    --config, so ignoring it runs the configured default under the requested
    model's name; a 1.7B arm scored byte-identically to the 0.6B default.
    """
    for model_size, expected in [
        ('mlx-community/Qwen3-ASR-1.7B-8bit', 'mlx-community/Qwen3-ASR-1.7B-8bit'),
        ('/models/qwen3-asr-1.7b', '/models/qwen3-asr-1.7b'),
        ('1.7B', 'Qwen/Qwen3-ASR-1.7B'),
        ('base', 'Qwen/Qwen3-ASR-0.6B'),
        ('', 'Qwen/Qwen3-ASR-0.6B'),
    ]:
        loaded = _install_fake_loader(monkeypatch)
        config = WhisperLiveKitConfig(backend='mlx-qwen3-asr', lan='en', model_size=model_size)
        asr = MlxQwen3ASR(config)
        assert asr.model_id == expected
        assert loaded == [expected]


def test_unknown_model_size_fails_instead_of_running_the_default(native, monkeypatch):
    loaded = _install_fake_loader(monkeypatch)
    with pytest.raises(ValueError, match='Unknown mlx-qwen3-asr model_size'):
        MlxQwen3ASR(WhisperLiveKitConfig(backend='mlx-qwen3-asr', lan='en',
                                        model_size='medium-en'))
    assert loaded == []


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


# The maintainer's four glued-join traces from the 30-clip French FLEURS run on
# 91d2b01 (`/tmp/wlk18/91-fr.json`), each a full hypothesis of that run: the text
# clients held, the window text that rewrote it, the text the client must end up
# with, and the glued string the run actually delivered. All four reproduce on
# 91d2b01 and none of them comes from a capitalised rewrite only: `1836`'s seam
# is lowercase and `1913` keeps the word it revises.
REWRITTEN_SPAN_TRACES = [
    pytest.param(
        'Situé au sommet Une des montagnes',
        "Situé au sommet D'une des montagnes au nord de la Meck, la grotte est "
        'totalement isolée du reste du monde.',
        'Situé au sommet Une des montagnes au nord de la Meck, la grotte est '
        'totalement isolée du reste du monde.',
        "montagnesD'une",
        id='fleurs_fr_1739',
    ),
    pytest.param(
        "Les Sans d'arbon sont la",
        "Les Sans d'arbon C'est la plus grande ceinture de mangrove côtière du "
        "monde, s'étendant sur quatre-vingt kilomètres cinquante miles depuis la "
        "côte jusqu'à l'arrière pays bangladais et indien.",
        "Les Sans d'arbon sont la plus grande ceinture de mangrove côtière du "
        "monde, s'étendant sur quatre-vingt kilomètres cinquante miles depuis la "
        "côte jusqu'à l'arrière pays bangladais et indien.",
        "laC'est",
        id='fleurs_fr_1893',
    ),
    pytest.param(
        'Le mouvement du',
        'Le mouvement Roussement du ting ne ressemble pas roussement en pluie de '
        'liant mais plutôt à une phrase dans les mots serait des cris et des '
        'grondements',
        'Le mouvement du ting ne ressemble pas roussement en pluie de liant mais '
        'plutôt à une phrase dans les mots serait des cris et des grondements',
        'duRoussement',
        id='fleurs_fr_1913',
    ),
    pytest.param(
        "Depuis l'Antiquité. Les peuples sont familiers des éléments chimiques de "
        "base comme l'or. L'argent et le cuivre. car il se trouve dans",
        "Depuis l'Antiquité. Les peuples sont familiers des éléments chimiques de "
        "base comme l'or. L'argent et le cuivre. car il se trouve ds se trouvent "
        "dans la nature sous forme brute et sont assez simples à miner à l'aide "
        "d'outils primitifs",
        "Depuis l'Antiquité. Les peuples sont familiers des éléments chimiques de "
        "base comme l'or. L'argent et le cuivre. car il se trouve dans se "
        "trouvent dans la nature sous forme brute et sont assez simples à miner "
        "à l'aide d'outils primitifs",
        'danss se trouvent',
        id='fleurs_fr_1836',
    ),
]


@pytest.mark.parametrize(('held', 'candidate', 'expected', 'glued'), REWRITTEN_SPAN_TRACES)
def test_a_rewritten_span_is_appended_at_a_unit_boundary(
    held, candidate, expected, glued
):
    """A candidate that rewrites held text appends from the next unit.

    Aligning the append on a shared character prefix cuts inside a word, which
    is what delivered `montagnesD'une`, `laC'est`, `duRoussement` and
    `danss se trouvent` to clients; a unit cutoff cannot. Cutting the append on
    units instead of characters is the whole change, so restoring the character
    cutoff fails every one of these four.
    """
    delta = _append_only_delta(held, candidate)
    assert held + delta == expected
    # The wording the client already renders is never taken back...
    assert expected.startswith(held)
    # ...and the seam is a separator, not two words run together.
    assert delta.startswith(' ') and not held.endswith(' ')
    assert glued not in held + delta


def test_a_rewrite_that_keeps_the_last_held_word_does_not_repeat_it():
    # 1913's candidate keeps 'du' across the rewrite. Appending from the unit
    # count alone would send 'du' twice; a candidate unit equal to the last held
    # unit is skipped, so the client reads 'du ting'.
    delta = _append_only_delta(
        'Le mouvement du', 'Le mouvement Roussement du ting ne ressemble pas'
    )
    assert delta == ' ting ne ressemble pas'


def test_cjk_revisions_advance_one_character_at_a_time():
    # CJK runs without separators, so a unit is one character and the append
    # starts at the character clients hold; the fragment before it stays put.
    assert _append_only_delta('甲乙', '甲丙丁戊') == '丁戊'
    # A Latin token is one unit, so the four characters plus 'Co' put the
    # boundary at '蒂'.
    assert _append_only_delta('我们去了Co', '我们去了科蒂纳。') == '蒂纳。'


def test_a_rewrite_of_the_last_held_unit_is_dropped_not_glued():
    # The case that motivated committing only at boundaries: provisional "the
    # number is eight", finalize "the number is 802.11a". Under a character
    # cutoff the correction was glued on (`the number is eight802.11a`); under a
    # unit cutoff it cannot reach a client that already renders "eight", so the
    # held wording stays and nothing is concatenated.
    held = 'the number is eight'
    assert _append_only_delta(held, 'the number is 802.11a') == ''
    assert 'eight802' not in held


def test_a_candidate_that_retracts_sends_nothing():
    # Clients already render the longer text, so a shorter candidate adds
    # nothing. A candidate that only changes the last held unit cannot reach
    # them either: `fish and chips.` keeps the units they hold, punctuation
    # included, and the next unit is what gets appended.
    assert _append_only_delta('fish and chips', 'fish and') == ''
    assert _append_only_delta('fish and chips', 'fish and chips.') == ''
    assert _append_only_delta('fish and chips', 'fish and chips. Today') == ' Today'


@pytest.mark.parametrize(('previous_stable', 'new_text', 'candidate', 'expected'), [
    # Only the part the client does not already hold; the revision's tail comes
    # with the next window instead of repeating the fragment.
    ('我们去了Co', '我们去了科蒂纳。', '我们去了Co', '我们去了Co蒂纳。'),
    # '丁戊' follows the two characters clients hold; '丙' is inside their span.
    ('甲乙', '甲丙丁戊己庚辛壬癸', '甲丙丁戊', '甲乙丁戊己庚辛壬癸'),
    ('甲乙丙', '甲乙丙丁戊己庚', '甲乙丙', '甲乙丙丁戊己庚'),
    ('', 'one two three four five six', 'one', 'one two three four five six'),
])
def test_issue_31_candidates_are_absorbed_as_appends(
    previous_stable, new_text, candidate, expected
):
    held = previous_stable + _append_only_delta(previous_stable, candidate)
    held += _append_only_delta(held, new_text)
    assert held == expected


def test_issue_31_revision_neither_raises_nor_repeats_at_a_pause(native):
    online = MlxQwen3AsrOnlineProcessor(shared_asr())
    online.insert_audio_chunk(np.zeros(32000), 2.0)
    online.process_iter()
    native.script.append(('我们去了科蒂纳', '我们去了Co'))
    online.insert_audio_chunk(np.zeros(32000), 4.0)
    committed, _ = online.process_iter()
    assert text(committed) == '我们去了Co'
    # Past the units clients hold, not the whole fragment again.
    assert online.get_buffer().text == '蒂纳'
    native.finals.append('我们去了科蒂纳。')
    tokens, _ = online.start_silence()
    assert text(committed + tokens) == '我们去了Co蒂纳。'
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
    assert text(committed) == '甲乙丁戊'
    assert online.get_buffer().text == '己庚辛壬癸'
    native.finals.append('甲丙丁戊己庚辛壬癸')
    tokens, _ = online.start_silence()
    # '丁戊' reaches the client once even though two decodes report it, and '丙'
    # inside the span they already hold stays as it was sent.
    assert text(committed + tokens) == '甲乙丁戊己庚辛壬癸'


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

    held      = 'là il y a eu tel'
    candidate = 'Là il y a eu tel second de silence et je repars'
    """
    held = "là il y a eu tel"
    delta = _append_only_delta(held, "Là il y a eu tel second de silence et je repars")
    assert delta == " second de silence et je repars"
    assert held + delta == "là il y a eu tel second de silence et je repars"


def test_case_only_revision_keeps_no_duplicate_on_fleurs_fr_1935():
    delta = _append_only_delta(
        "cependant ce n'est pas vrai bien",
        "Cependant ce n'est pas vrai bien qu'il y ait une raison",
    )
    assert delta == " qu'il y ait une raison"
