import logging

import pytest

from whisperlivekit.config import WhisperLiveKitConfig


def test_funasr_config_normalizes_to_localagreement(caplog):
    with caplog.at_level(logging.WARNING, logger="whisperlivekit.config"):
        config = WhisperLiveKitConfig(backend="funasr")

    assert config.backend_policy == "localagreement"
    assert "requires LocalAgreement" in caplog.text


@pytest.mark.parametrize("language", ["auto", "zh", "yue", "en", "ja", "ko"])
def test_funasr_config_accepts_supported_languages(language):
    config = WhisperLiveKitConfig(
        backend="funasr", backend_policy="localagreement", lan=language
    )
    assert config.lan == language


def test_funasr_config_rejects_unsupported_language():
    with pytest.raises(ValueError, match="supports only"):
        WhisperLiveKitConfig(
            backend="funasr", backend_policy="localagreement", lan="fr"
        )


def test_funasr_config_rejects_direct_translation():
    with pytest.raises(ValueError, match="direct English translation"):
        WhisperLiveKitConfig(
            backend="funasr",
            backend_policy="localagreement",
            direct_english_translation=True,
        )


def test_funasr_sentence_trimming_requires_explicit_language():
    with pytest.raises(ValueError, match="explicit language"):
        WhisperLiveKitConfig(
            backend="funasr",
            backend_policy="localagreement",
            lan="auto",
            buffer_trimming="sentence",
        )


def _adapter_for_tokens(postprocessed_text, language=None):
    from whisperlivekit.funasr_backend import FunASRASR

    asr = FunASRASR.__new__(FunASRASR)
    asr.original_language = language
    asr._postprocess = lambda _raw: postprocessed_text
    return asr


def test_funasr_backend_declares_sampling_rate():
    from whisperlivekit.funasr_backend import FunASRASR

    assert FunASRASR.SAMPLING_RATE == 16000


def test_funasr_does_not_extend_whisper_language_table():
    from whisperlivekit.local_agreement.whisper_online import WHISPER_LANG_CODES

    assert "yue" not in WHISPER_LANG_CODES


@pytest.mark.parametrize(
    ("text", "words", "expected_parts"),
    [
        (
            "The tribal chief called for 50 pieces.",
            ["The", "tribal", "chief", "called", "for", "5", "0", "pieces"],
            [
                "The",
                " tribal",
                " chief",
                " called",
                " for",
                " 5",
                "0",
                " pieces.",
            ],
        ),
        (
            "开饭时间早上9点至下午5点。",
            ["开", "饭", "时", "间", "早", "上", "9", "点", "至", "下", "午", "5", "点"],
            ["开", "饭", "时", "间", "早", "上", "9", "点", "至", "下", "午", "5", "点。"],
        ),
        (
            "呢几个字，都表达唔到我想讲嘅意思。",
            ["呢", "几", "个", "字", "都", "表", "达", "唔", "到", "我", "想", "讲", "嘅", "意", "思"],
            ["呢", "几", "个", "字，", "都", "表", "达", "唔", "到", "我", "想", "讲", "嘅", "意", "思。"],
        ),
        (
            "조금만 생각을 하면서 살면 편할 거야.",
            ["조금만", "생각을", "하면서", "살면", "편할", "거야"],
            ["조금만", " 생각을", " 하면서", " 살면", " 편할", " 거야."],
        ),
    ],
)
def test_funasr_tokens_reconstruct_postprocessed_text_exactly(
    text, words, expected_parts
):
    asr = _adapter_for_tokens(text)
    timestamps = [
        [index * 100, (index + 1) * 100] for index in range(len(words))
    ]
    result = [{"text": "raw", "words": words, "timestamp": timestamps}]

    tokens = asr.ts_words(result)

    assert [token.text for token in tokens] == expected_parts
    assert "".join(token.text for token in tokens) == text
    assert [(token.start, token.end) for token in tokens] == [
        (index / 10, (index + 1) / 10) for index in range(len(words))
    ]


def test_funasr_tokens_propagate_detected_language_from_control_tag():
    asr = _adapter_for_tokens("hello")
    tokens = asr.ts_words(
        [
            {
                "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|>hello",
                "words": ["hello"],
                "timestamp": [[0, 500]],
            }
        ]
    )
    assert tokens[0].detected_language == "en"


def test_funasr_explicit_language_wins_over_control_tag():
    asr = _adapter_for_tokens("hello", language="ko")
    tokens = asr.ts_words(
        [
            {
                "text": "<|en|>hello",
                "words": ["hello"],
                "timestamp": [[0, 500]],
            }
        ]
    )
    assert tokens[0].detected_language == "ko"


@pytest.mark.parametrize(
    "result",
    [
        [{}],
        [{"text": "", "words": []}],
        [{"text": "", "timestamp": []}],
        [{"words": [], "timestamp": []}],
    ],
)
def test_funasr_missing_result_fields_fail_closed(result):
    asr = _adapter_for_tokens("")
    with pytest.raises(ValueError, match="missing required field"):
        asr.ts_words(result)


@pytest.mark.parametrize(
    "result",
    [
        [{"text": "", "words": None, "timestamp": []}],
        [{"text": "", "words": [], "timestamp": None}],
    ],
)
def test_funasr_invalid_empty_result_containers_fail_closed(result):
    asr = _adapter_for_tokens("")
    with pytest.raises(ValueError, match="must be a list"):
        asr.ts_words(result)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([{"text": "hello", "words": ["hello"], "timestamp": []}], "length"),
        (
            [{"text": "hello", "words": ["hello"], "timestamp": [[500, 100]]}],
            "span",
        ),
        (
            [
                {
                    "text": "hello world",
                    "words": ["world", "hello"],
                    "timestamp": [[0, 100], [100, 200]],
                }
            ],
            "word index",
        ),
        (
            [
                {
                    "text": "hello",
                    "words": ["hello"],
                    "timestamp": [[float("nan"), 100]],
                }
            ],
            "finite",
        ),
        (
            [
                {
                    "text": "hello world",
                    "words": ["hello", "world"],
                    "timestamp": [[200, 400], [100, 500]],
                }
            ],
            "non-monotonic",
        ),
        (
            [
                {
                    "text": "hello missing world",
                    "words": ["hello", "world"],
                    "timestamp": [[0, 100], [100, 200]],
                }
            ],
            "unassigned text",
        ),
    ],
)
def test_funasr_malformed_results_fail_closed(result, message):
    asr = _adapter_for_tokens(result[0]["text"])
    with pytest.raises(ValueError, match=message):
        asr.ts_words(result)


@pytest.mark.parametrize(
    "result", [None, [], [{"text": "", "words": [], "timestamp": []}]]
)
def test_funasr_empty_result_stays_empty(result):
    asr = _adapter_for_tokens("")
    assert asr.ts_words(result) == []
    assert asr.segments_end_ts(result) == []


def test_funasr_decoration_only_output_stays_empty():
    asr = _adapter_for_tokens("")
    result = [{"text": "<|en|><|NEUTRAL|><|Music|><|woitn|>", "words": [], "timestamp": []}]

    assert asr.ts_words(result) == []
    assert asr.segments_end_ts(result) == []


def test_funasr_session_rejects_unsupported_language():
    from types import SimpleNamespace

    from whisperlivekit.core import online_factory

    asr = SimpleNamespace(backend_choice="funasr")
    args = SimpleNamespace(backend="funasr")

    with pytest.raises(ValueError, match="supports only"):
        online_factory(args, asr, language="fr")


def test_funasr_segment_end_uses_last_word_timestamp():
    asr = _adapter_for_tokens("hello world")
    result = [
        {
            "text": "raw",
            "words": ["hello", "world"],
            "timestamp": [[0, 250], [300, 900]],
        }
    ]
    assert asr.segments_end_ts(result) == [0.9]


def test_wlk_cli_registers_funasr_backend():
    from whisperlivekit.cli import BACKENDS

    backend = next(entry for entry in BACKENDS if entry["id"] == "funasr")

    assert backend["module"] == "funasr"
    assert backend["policy"] == "localagreement"


def test_wlk_cli_runs_funasr_without_fake_huggingface_pull(capsys):
    from whisperlivekit.cli import _resolve_pull_target, _resolve_run_spec

    assert _resolve_run_spec("funasr") == ("funasr", None)
    assert _resolve_pull_target("funasr") == []
    assert capsys.readouterr().out == ""
