import asyncio
from types import SimpleNamespace

import numpy as np
import pytest


def _base_simul_kwargs(**overrides):
    kwargs = {
        "warmup_file": "",
        "min_chunk_size": 0.1,
        "model_size": "tiny",
        "model_cache_dir": None,
        "model_dir": None,
        "model_path": None,
        "encoder_model_path": None,
        "decoder_model_path": None,
        "lora_path": None,
        "lan": "en",
        "direct_english_translation": False,
        "disable_fast_encoder": False,
        "custom_alignment_heads": None,
        "frame_threshold": 25,
        "beams": 1,
        "decoder_type": None,
        "audio_max_len": 30.0,
        "audio_min_len": 0.0,
        "cif_ckpt_path": None,
        "never_fire": False,
        "init_prompt": None,
        "static_init_prompt": None,
        "max_context_tokens": None,
        "backend": "faster-whisper",
    }
    kwargs.update(overrides)
    return kwargs


def _make_ct2_dir(tmp_path):
    model_dir = tmp_path / "ct2"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"ct2")
    (model_dir / "vocabulary.json").write_text("{}", encoding="utf-8")
    return model_dir


def _make_pytorch_dir(tmp_path):
    model_dir = tmp_path / "pytorch"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"torch")
    return model_dir


def _make_mlx_dir(tmp_path):
    model_dir = tmp_path / "mlx"
    model_dir.mkdir()
    (model_dir / "weights.npz").write_bytes(b"mlx")
    return model_dir


def _patch_simul_loaders(monkeypatch):
    import whisperlivekit.simul_whisper.backend as backend

    load_calls = []

    class FakeWhisperModel:
        def __init__(self, model_ref, **kwargs):
            self.model_ref = model_ref
            self.kwargs = kwargs

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return SimpleNamespace(model_ref=name)

    monkeypatch.setattr(backend, "HAS_FASTER_WHISPER", True)
    monkeypatch.setattr(backend, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(backend, "load_model", fake_load_model)
    return backend, load_calls


def test_simulstreaming_uses_explicit_ct2_encoder_path(tmp_path, monkeypatch):
    backend, load_calls = _patch_simul_loaders(monkeypatch)
    ct2_dir = _make_ct2_dir(tmp_path)

    asr = backend.SimulStreamingASR(
        **_base_simul_kwargs(encoder_model_path=str(ct2_dir))
    )

    assert asr.encoder_backend == "faster-whisper"
    assert asr.fw_encoder.model_ref == str(ct2_dir)
    assert load_calls[0][0] == "tiny"
    assert load_calls[0][1]["decoder_only"] is True


def test_simulstreaming_uses_separate_encoder_and_decoder_paths(tmp_path, monkeypatch):
    backend, load_calls = _patch_simul_loaders(monkeypatch)
    ct2_dir = _make_ct2_dir(tmp_path)
    pytorch_dir = _make_pytorch_dir(tmp_path)

    asr = backend.SimulStreamingASR(
        **_base_simul_kwargs(
            encoder_model_path=str(ct2_dir),
            decoder_model_path=str(pytorch_dir),
        )
    )

    assert asr.fw_encoder.model_ref == str(ct2_dir)
    assert load_calls[0][0] == str(pytorch_dir)
    assert load_calls[0][1]["decoder_only"] is True


def test_simulstreaming_legacy_model_path_rejects_ct2_only_dir(tmp_path, monkeypatch):
    backend, _ = _patch_simul_loaders(monkeypatch)
    ct2_dir = _make_ct2_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="--encoder-model-path"):
        backend.SimulStreamingASR(
            **_base_simul_kwargs(model_path=str(ct2_dir))
        )


def test_simulstreaming_uses_explicit_mlx_encoder_path(tmp_path, monkeypatch):
    import whisperlivekit.simul_whisper.backend as backend

    mlx_dir = _make_mlx_dir(tmp_path)
    pytorch_dir = _make_pytorch_dir(tmp_path)
    load_calls = []
    mlx_calls = []

    def fake_load_model(name, **kwargs):
        load_calls.append((name, kwargs))
        return SimpleNamespace(model_ref=name)

    def fake_load_mlx_encoder(path_or_hf_repo):
        mlx_calls.append(path_or_hf_repo)
        return SimpleNamespace(model_ref=path_or_hf_repo)

    monkeypatch.setattr(backend, "HAS_MLX_WHISPER", True)
    monkeypatch.setattr(backend, "load_mlx_encoder", fake_load_mlx_encoder, raising=False)
    monkeypatch.setattr(backend, "load_model", fake_load_model)

    asr = backend.SimulStreamingASR(
        **_base_simul_kwargs(
            backend="mlx-whisper",
            encoder_model_path=str(mlx_dir),
            decoder_model_path=str(pytorch_dir),
        )
    )

    assert asr.encoder_backend == "mlx-whisper"
    assert mlx_calls == [str(mlx_dir)]
    assert load_calls[0][0] == str(pytorch_dir)


class FakeSimulStreamingModel:
    def __init__(self, batches):
        self.batches = list(batches)
        self.cfg = SimpleNamespace(language="en")
        self.refresh_calls = []
        self.global_time_offset = 0.0

    def infer(self, is_last=False):
        return self.batches.pop(0) if self.batches else []

    def refresh_segment(self, complete=False):
        self.refresh_calls.append(complete)


def _make_simul_processor(model, end=0.0):
    from whisperlivekit.simul_whisper.backend import SimulStreamingOnlineProcessor

    processor = object.__new__(SimulStreamingOnlineProcessor)
    processor.asr = SimpleNamespace(use_full_mlx=True)
    processor.model = model
    processor.end = end
    processor.buffer = []
    processor._last_committed_end = 0.0
    processor._recent_words = []
    return processor


def test_simulstreaming_filters_rewound_words_after_committed_time():
    from whisperlivekit.timed_objects import ASRToken

    model = FakeSimulStreamingModel(
        [
            [
                ASRToken(10.00, 10.10, " hello"),
                ASRToken(10.20, 10.30, " world"),
            ],
            [
                ASRToken(9.50, 9.60, " stale"),
                ASRToken(10.35, 10.45, " again"),
                ASRToken(10.25, 10.35, " stale2"),
                ASRToken(10.50, 10.60, " now"),
            ],
        ]
    )
    processor = _make_simul_processor(model, end=11.0)

    first, _ = processor.process_iter()
    second, _ = processor.process_iter()

    assert [token.text for token in first] == [" hello", " world"]
    assert [token.text for token in second] == [" again", " now"]
    assert processor._last_committed_end == pytest.approx(10.60)
    assert model.refresh_calls == []


def test_simulstreaming_keeps_minor_intra_batch_timestamp_jitter():
    from whisperlivekit.timed_objects import ASRToken

    model = FakeSimulStreamingModel(
        [
            [
                ASRToken(1.00, 1.10, " concord"),
                ASRToken(1.60, 1.70, " returned"),
                ASRToken(2.20, 2.30, " its"),
                ASRToken(2.18, 2.28, " place"),
                ASRToken(2.50, 2.60, " amidst"),
            ]
        ]
    )
    processor = _make_simul_processor(model, end=3.0)

    tokens, _ = processor.process_iter()

    assert [token.text for token in tokens] == [
        " concord",
        " returned",
        " its",
        " place",
        " amidst",
    ]
    assert model.refresh_calls == []


def test_simulstreaming_resets_when_all_words_rewind_far_behind():
    from whisperlivekit.timed_objects import ASRToken

    model = FakeSimulStreamingModel(
        [[ASRToken(186.0, 186.1, " old"), ASRToken(187.0, 187.1, " text")]]
    )
    processor = _make_simul_processor(model, end=195.0)
    processor._last_committed_end = 191.2

    tokens, processed_upto = processor.process_iter()

    assert tokens == []
    assert processed_upto == 195.0
    assert model.refresh_calls == [True]
    assert model.global_time_offset == 195.0
    assert processor.buffer == []


def test_simulstreaming_resets_repetition_loop_before_emitting_words():
    from whisperlivekit.timed_objects import ASRToken

    phrase = [" Det", " ar", " en", " ny", " kriska", " klimat"]
    tokens = [
        ASRToken(idx * 0.2, idx * 0.2 + 0.1, word)
        for idx, word in enumerate(phrase * 3)
    ]
    model = FakeSimulStreamingModel([tokens])
    processor = _make_simul_processor(model, end=42.0)

    emitted, processed_upto = processor.process_iter()

    assert emitted == []
    assert processed_upto == 42.0
    assert model.refresh_calls == [True]
    assert model.global_time_offset == 42.0
    assert processor._last_committed_end == 0.0


def test_simulstreaming_detects_repetition_across_small_batches():
    from whisperlivekit.timed_objects import ASRToken

    batches = []
    phrase = [" Det", " ar", " en", " ny", " kriska", " klimat"]
    for repeat in range(3):
        batches.append(
            [
                ASRToken(repeat * 2.0 + idx * 0.2, repeat * 2.0 + idx * 0.2 + 0.1, word)
                for idx, word in enumerate(phrase)
            ]
        )
    model = FakeSimulStreamingModel(batches)
    processor = _make_simul_processor(model, end=10.0)

    first, _ = processor.process_iter()
    second, _ = processor.process_iter()
    third, _ = processor.process_iter()

    assert len(first) == 6
    assert len(second) == 6
    assert third == []
    assert model.refresh_calls == [True]


def _make_alignatt_timestamp_tester(offset=0.0):
    from whisperlivekit.simul_whisper.align_att_base import AlignAttBase

    class TimestampAligner(AlignAttBase):
        pass

    TimestampAligner.__abstractmethods__ = frozenset()
    aligner = TimestampAligner()
    aligner.state = SimpleNamespace(
        speaker=2,
        detected_language="en",
        global_time_offset=offset,
        pending_incomplete_tokens=[],
        pending_incomplete_token_timestamps=[],
        pending_retries=0,
    )
    return aligner


def test_alignatt_word_end_uses_next_word_timestamp_with_offset():
    aligner = _make_alignatt_timestamp_tester(offset=10.0)

    tokens = aligner._build_timestamped_words(
        split_words=[" hello", " world"],
        split_tokens=[[101, 102], [103]],
        token_timestamps=[0.50, 0.70, 1.20],
    )

    assert tokens[0].start == pytest.approx(10.50)
    assert tokens[0].end == pytest.approx(11.20)
    assert tokens[1].start == pytest.approx(11.20)
    assert tokens[1].end == pytest.approx(11.30)
    assert tokens[0].end <= tokens[1].start


def test_alignatt_final_multitoken_word_uses_last_token_timestamp_fallback():
    aligner = _make_alignatt_timestamp_tester()

    tokens = aligner._build_timestamped_words(
        split_words=[" longer"],
        split_tokens=[[201, 202]],
        token_timestamps=[2.00, 2.34],
    )

    assert tokens[0].start == pytest.approx(2.00)
    assert tokens[0].end == pytest.approx(2.44)


def test_alignatt_single_token_word_keeps_short_numeric_end():
    aligner = _make_alignatt_timestamp_tester()

    tokens = aligner._build_timestamped_words(
        split_words=[" word"],
        split_tokens=[[301]],
        token_timestamps=[4.00],
    )

    assert tokens[0].start == pytest.approx(4.00)
    assert tokens[0].end == pytest.approx(4.10)


def test_alignatt_pending_utf8_tokens_preserve_original_timestamps():
    aligner = _make_alignatt_timestamp_tester()

    aligner._handle_pending_tokens(
        split_words=[" caf\ufffd"],
        split_tokens=[[401, 402]],
        token_timestamps=[3.20, 3.24],
    )
    assert aligner.state.pending_incomplete_tokens == [401, 402]
    assert aligner.state.pending_incomplete_token_timestamps == [3.20, 3.24]

    merged_tokens, merged_timestamps = aligner._prepend_pending_tokens(
        tokens_to_split=[403, 404],
        token_timestamps=[4.00, 4.12],
    )
    assert merged_tokens == [401, 402, 403, 404]
    assert merged_timestamps == [3.20, 3.24, 4.00, 4.12]

    words = aligner._build_timestamped_words(
        split_words=[" cafe", " next"],
        split_tokens=[[401, 402, 403], [404]],
        token_timestamps=merged_timestamps,
    )

    assert words[0].start == pytest.approx(3.20)
    assert words[0].end == pytest.approx(4.12)


def test_decoder_state_clean_cache_does_not_empty_cuda_cache(monkeypatch):
    import torch

    from whisperlivekit.simul_whisper.decoder_state import DecoderState

    calls = {"empty_cache": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )

    state = DecoderState()
    state.kv_cache["tensor"] = torch.zeros(1)

    state.clean_cache()

    assert state.kv_cache == {}
    assert calls["empty_cache"] == 0


def test_decoder_state_release_gpu_memory_empties_cuda_cache(monkeypatch):
    import torch

    from whisperlivekit.simul_whisper.decoder_state import DecoderState

    calls = {"empty_cache": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )

    DecoderState().release_gpu_memory()

    assert calls["empty_cache"] == 1


def test_load_cif_without_checkpoint_skips_unused_linear(caplog):
    from whisperlivekit.simul_whisper.eow_detection import load_cif

    cfg = SimpleNamespace(cif_ckpt_path=None, never_fire=False)

    cif_linear, always_fire, never_fire = load_cif(
        cfg, n_audio_state=16, device="cpu"
    )

    assert cif_linear is None
    assert always_fire is True
    assert never_fire is False
    assert "CIF end-of-word detection is disabled" in caplog.text


def test_load_cif_without_checkpoint_preserves_never_fire():
    from whisperlivekit.simul_whisper.eow_detection import load_cif

    cfg = SimpleNamespace(cif_ckpt_path="", never_fire=True)

    cif_linear, always_fire, never_fire = load_cif(
        cfg, n_audio_state=16, device="cpu"
    )

    assert cif_linear is None
    assert always_fire is False
    assert never_fire is True


def test_nllw_language_code_maps_whisper_chinese_aliases():
    from whisperlivekit.core import _nllw_language_code

    assert _nllw_language_code("zh") == "zh-CN"
    assert _nllw_language_code("zh_Hans") == "zh-CN"
    assert _nllw_language_code("zh-Hant") == "zh-TW"
    assert _nllw_language_code("en") == "en"
    assert _nllw_language_code("eng_Latn") == "eng_Latn"
    assert _nllw_language_code("auto") == "auto"


def test_transcription_engine_uses_nllw_source_language_alias(monkeypatch):
    import sys

    from whisperlivekit.config import WhisperLiveKitConfig
    from whisperlivekit.core import TranscriptionEngine

    load_calls = []

    def fake_load_model(src_langs, **kwargs):
        load_calls.append((src_langs, kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "nllw",
        SimpleNamespace(load_model=fake_load_model),
    )

    TranscriptionEngine.reset()
    try:
        engine = TranscriptionEngine(
            config=WhisperLiveKitConfig(
                transcription=False,
                vac=False,
                lan="zh",
                target_language="eng_Latn",
            )
        )
    finally:
        TranscriptionEngine.reset()

    assert load_calls == [
        (["zh-CN"], {"nllb_backend": "transformers", "nllb_size": "600M"})
    ]
    assert engine.args.lan == "zh"


def test_online_translation_factory_uses_nllw_language_aliases(monkeypatch):
    import sys

    from whisperlivekit.core import online_translation_factory

    calls = []

    class FakeOnlineTranslation:
        def __init__(self, translation_model, input_languages, output_languages):
            calls.append((translation_model, input_languages, output_languages))

    monkeypatch.setitem(
        sys.modules,
        "nllw",
        SimpleNamespace(OnlineTranslation=FakeOnlineTranslation),
    )

    translation_model = object()
    result = online_translation_factory(
        SimpleNamespace(lan="zh", target_language="zh"),
        translation_model,
    )

    assert isinstance(result, FakeOnlineTranslation)
    assert calls == [(translation_model, ["zh-CN"], ["zh-CN"])]


def test_ctranslate2_storage_view_converts_to_tensor():
    ctranslate2 = pytest.importorskip("ctranslate2")
    from whisperlivekit.simul_whisper.simul_whisper import _encoder_features_to_tensor

    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    view = ctranslate2.StorageView.from_array(data)

    tensor = _encoder_features_to_tensor(view, "cpu")

    assert tuple(tensor.shape) == data.shape
    np.testing.assert_allclose(tensor.numpy(), data)


def test_ctranslate2_storage_view_list_converts_to_tensor():
    ctranslate2 = pytest.importorskip("ctranslate2")
    from whisperlivekit.simul_whisper.simul_whisper import _encoder_features_to_tensor

    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first + 10
    views = [
        ctranslate2.StorageView.from_array(first),
        ctranslate2.StorageView.from_array(second),
    ]

    tensor = _encoder_features_to_tensor(views, "cpu")

    assert tuple(tensor.shape) == (2, 2, 3)
    np.testing.assert_allclose(tensor.numpy(), np.stack([first, second]))


def test_openai_api_asr_initializes_transcribe_kwargs_and_routes_tasks(monkeypatch):
    from whisperlivekit.local_agreement.backends import OpenaiApiASR

    class FakeEndpoint:
        def __init__(self):
            self.calls = []

        def create(self, **params):
            self.calls.append(params)
            return SimpleNamespace(words=[], segments=[])

    transcriptions = FakeEndpoint()
    translations = FakeEndpoint()

    def fake_load_model(self):
        self.client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=transcriptions,
                translations=translations,
            )
        )
        self.transcribed_seconds = 0

    monkeypatch.setattr(OpenaiApiASR, "load_model", fake_load_model)

    asr = OpenaiApiASR(lan="en")
    audio = np.zeros(16_000, dtype=np.float32)

    asr.transcribe(audio)
    asr.transcribe_kargs["task"] = "translate"
    asr.transcribe(audio, init_prompt="previous context")

    assert len(transcriptions.calls) == 1
    assert len(translations.calls) == 1
    assert translations.calls[0]["prompt"] == "previous context"


def test_tokens_alignment_prunes_long_running_history():
    from whisperlivekit.timed_objects import ASRToken, Segment, SpeakerSegment, State, TimedText
    from whisperlivekit.tokens_alignment import TokensAlignment

    state = State()
    alignment = TokensAlignment(state, SimpleNamespace(diarization=False), " ")
    alignment._retention_seconds = 10.0

    tokens = [ASRToken(i, i + 0.5, f" w{i}") for i in range(30)]
    state.new_tokens = tokens[:]
    state.new_diarization = [
        SpeakerSegment(start=i, end=i + 0.5, speaker=0)
        for i in range(30)
    ]
    state.new_translation = [
        TimedText(start=i, end=i + 0.5, text=f"t{i}")
        for i in range(30)
    ]
    alignment.validated_segments = [
        Segment(start=i, end=i + 0.5, text=f"s{i}", speaker=1)
        for i in range(30)
    ]
    alignment.unvalidated_tokens = tokens[:]

    alignment.update()
    alignment.get_lines(audio_time=30.0)

    assert alignment.all_tokens[0].end >= 19.5
    assert alignment.all_diarization_segments[0].end >= 19.5
    assert alignment.all_translation_segments[0].end >= 19.5
    assert alignment.validated_segments[0].end >= 19.5
    assert alignment.current_line_tokens[0].end >= 19.5
    assert alignment.unvalidated_tokens[0].end >= 19.5


def test_audio_processor_prunes_persistent_state_tokens():
    from whisperlivekit.audio_processor import AudioProcessor
    from whisperlivekit.timed_objects import ASRToken, State

    processor = object.__new__(AudioProcessor)
    processor.state = State()
    processor.tokens_alignment = SimpleNamespace(_retention_seconds=10.0)
    processor.state.end_buffer = 30.0
    processor.state.tokens = [
        ASRToken(i, i + 0.5, f" w{i}")
        for i in range(30)
    ]

    processor._prune_state_tokens()

    assert processor.state.tokens[0].end >= 20.0
    assert processor.state.tokens[-1].text == " w29"


@pytest.mark.asyncio
async def test_audio_processor_finish_commits_pending_buffer_when_backend_flush_is_empty():
    from whisperlivekit.audio_processor import AudioProcessor
    from whisperlivekit.metrics_collector import SessionMetrics
    from whisperlivekit.timed_objects import State, Transcript

    class EmptyFinishBackend:
        def finish(self):
            return [], 3.5

        def get_buffer(self):
            return Transcript()

    processor = object.__new__(AudioProcessor)
    processor.transcription = EmptyFinishBackend()
    processor.state = State()
    processor.state.buffer_transcription = Transcript(
        start=0.55,
        end=3.05,
        text="Concord returned to its place amidst the tents",
    )
    processor.state.end_buffer = 3.5
    processor.lock = asyncio.Lock()
    processor.metrics = SessionMetrics()
    processor.translation_queue = None
    processor._prune_state_tokens = lambda: None

    await processor._finish_transcription()

    assert len(processor.state.new_tokens) == 1
    assert processor.state.new_tokens[0].text == "Concord returned to its place amidst the tents"
    assert processor.state.buffer_transcription.text == ""
    assert processor.metrics.n_tokens_produced == 1


def test_verbose_json_fallback_creates_segment_when_text_has_no_lines():
    from whisperlivekit.cli import _format_verbose_json_result

    result = SimpleNamespace(
        committed_text="",
        text="hello world",
        lines=[],
    )

    payload = _format_verbose_json_result(result, duration=12.5, language="en")

    assert payload["text"] == "hello world"
    assert payload["segments"] == [
        {
            "text": "hello world",
            "start": "0:00:00.00",
            "end": "0:00:12.50",
            "speaker": 1,
        }
    ]


def test_parse_cors_origins_defaults_to_disabled():
    from whisperlivekit.config import parse_cors_origins

    assert parse_cors_origins(None) == []
    assert parse_cors_origins("") == []
    assert parse_cors_origins("   ") == []


def test_parse_cors_origins_accepts_comma_separated_values():
    from whisperlivekit.config import parse_cors_origins

    assert parse_cors_origins("https://app.example, http://localhost:3000") == [
        "https://app.example",
        "http://localhost:3000",
    ]
    assert parse_cors_origins("*") == ["*"]


def test_parse_args_accepts_cors_origins(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["wlk", "--cors-origins", "https://app.example,http://localhost:3000"],
    )

    from whisperlivekit.parse_args import parse_args

    config = parse_args()

    assert config.cors_origins == "https://app.example,http://localhost:3000"


class FakeFFmpegManager:
    def __init__(self, chunks=None):
        from whisperlivekit.ffmpeg_manager import FFmpegState

        self.chunks = list(chunks or [])
        self.closed = False
        self.stopped = False
        self.state = FFmpegState.RUNNING

    async def get_state(self):
        return self.state

    async def read_data(self, size):
        await asyncio.sleep(0)
        return self.chunks.pop(0)

    async def close_stdin(self):
        self.closed = True

    async def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_process_audio_non_pcm_closes_ffmpeg_stdin_without_sentinel():
    from whisperlivekit.audio_processor import AudioProcessor

    processor = object.__new__(AudioProcessor)
    processor.beg_loop = 1.0
    processor.is_stopping = False
    processor.is_pcm_input = False
    processor.ffmpeg_manager = FakeFFmpegManager()
    processor.transcription_queue = asyncio.Queue()
    processor.pcm_buffer = bytearray()

    await processor.process_audio(b"")

    assert processor.is_stopping is True
    assert processor.ffmpeg_manager.closed is True
    assert processor.transcription_queue.empty()


@pytest.mark.asyncio
async def test_ffmpeg_reader_drains_stdout_after_stop_before_sentinel():
    from whisperlivekit.audio_processor import SENTINEL, AudioProcessor

    processor = object.__new__(AudioProcessor)
    processor.is_stopping = True
    processor.ffmpeg_manager = FakeFFmpegManager([b"aaaa", None, b"bbbb", b""])
    processor.pcm_buffer = bytearray()
    processor.transcription_queue = asyncio.Queue()
    processor.diarization_queue = None
    processor.translation_queue = None
    processor.diarization = None
    processor.translation = None
    processor.bytes_per_sample = 2
    seen = []

    async def fake_handle_pcm_data():
        seen.append(bytes(processor.pcm_buffer))
        processor.pcm_buffer.clear()

    processor.handle_pcm_data = fake_handle_pcm_data

    await processor.ffmpeg_stdout_reader()

    assert seen == [b"aaaa", b"bbbb"]
    assert processor.ffmpeg_manager.stopped is True
    assert await processor.transcription_queue.get() is SENTINEL
    assert processor.transcription_queue.empty()
