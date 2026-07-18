def test_parse_args_accepts_canary_options(monkeypatch):
    from whisperlivekit.parse_args import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "whisperlivekit-server",
            "--backend", "canary",
            "--canary-model", "nvidia/canary-1b-v2",
            "--canary-default-lang", "de",
            "--canary-lid-model", "langid_ambernet",
            "--canary-lid-min-sec", "3.0",
            "--canary-lid-min-conf", "0.6",
        ],
    )
    cfg = parse_args()
    assert cfg.backend == "canary"
    assert cfg.canary_model == "nvidia/canary-1b-v2"
    assert cfg.canary_default_lang == "de"
    assert cfg.canary_lid_model == "langid_ambernet"
    assert cfg.canary_lid_min_sec == 3.0
    assert cfg.canary_lid_min_conf == 0.6


def test_canary_config_defaults():
    from whisperlivekit.config import WhisperLiveKitConfig

    cfg = WhisperLiveKitConfig.from_kwargs(backend="canary")
    assert cfg.canary_model == "nvidia/canary-1b-v2"
    assert cfg.canary_default_lang == "en"
    assert cfg.canary_lid_model == "langid_ambernet"
    assert cfg.canary_lid_min_sec == 2.0
    assert cfg.canary_lid_min_conf == 0.5


def test_canary_words_to_tokens_maps_word_timestamps():
    from whisperlivekit.canary_backend import canary_words_to_tokens

    word_stamps = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.9},
    ]
    tokens = canary_words_to_tokens(word_stamps)
    assert [t.text for t in tokens] == ["hello", "world"]
    assert tokens[0].start == 0.0 and tokens[0].end == 0.4
    assert tokens[1].start == 0.4 and tokens[1].end == 0.9


def test_canary_words_to_tokens_handles_missing_stamps():
    from whisperlivekit.canary_backend import canary_words_to_tokens

    assert canary_words_to_tokens(None) == []
    assert canary_words_to_tokens([]) == []


def test_canary_segment_end_ts():
    from whisperlivekit.canary_backend import canary_segment_end_ts

    seg_stamps = [
        {"segment": "hello world.", "start": 0.0, "end": 0.9},
        {"segment": "bye.", "start": 1.0, "end": 1.5},
    ]
    assert canary_segment_end_ts(seg_stamps) == [0.9, 1.5]
    assert canary_segment_end_ts(None) == []
