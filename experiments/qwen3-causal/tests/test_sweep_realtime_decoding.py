import argparse

from scripts.sweep_realtime_decoding import config_name, iter_configs


def test_iter_configs_builds_cartesian_product():
    args = argparse.Namespace(
        emit_thresholds="0.5,0.75",
        repetition_penalties="1.0,1.2",
        no_repeat_ngram_sizes="0,3",
        max_consecutive_text_tokens="0",
    )

    configs = iter_configs(args)

    assert len(configs) == 8
    assert configs[0] == {
        "emit_threshold": 0.5,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "max_consecutive_text_tokens": 0,
    }
    assert configs[-1] == {
        "emit_threshold": 0.75,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "max_consecutive_text_tokens": 0,
    }


def test_config_name_is_stable_for_paths():
    name = config_name(
        {
            "emit_threshold": 0.65,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "max_consecutive_text_tokens": 12,
        }
    )

    assert name == "th0p65_rp1p20_ng3_mx12"
