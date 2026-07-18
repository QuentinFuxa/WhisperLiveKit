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
