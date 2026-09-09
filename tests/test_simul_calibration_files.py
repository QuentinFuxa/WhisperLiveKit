"""The bundled calibration files must cover zh→en and ja→zh for the Hunyuan
8bit model. On upstream main only en→zh shipped; `--simultaneous` with zh→en
raised at engine construction. These tests load every bundled calibration
through the real loader and compare its prompt against the resolved model
profile — a wrong or missing file fails here."""
from whisperlivekit.simul_mt_calibration import load_calibration
from whisperlivekit.translation_profiles import MT_MODEL_PROFILES, resolve_prompt

_REPO = "mlx-community/Hy-MT2-1.8B-8bit"
_PROFILE = MT_MODEL_PROFILES["hy-mt2-1.8b-8bit"]

_DIRECTIONS = [("zh", "en"), ("ja", "zh"), ("en", "zh")]


def test_bundled_calibrations_cover_all_directions():
    for src, tgt in _DIRECTIONS:
        cal = load_calibration(None, _REPO, src, tgt)
        assert cal.heads and len(cal.heads) == 8


def test_bundled_calibration_prompt_matches_model_profile():
    for src, tgt in _DIRECTIONS:
        cal = load_calibration(None, _REPO, src, tgt)
        expected = resolve_prompt(_PROFILE, src, tgt)
        assert cal.prompt == expected, f"{src}->{tgt}: {cal.prompt} != {expected}"


def test_top_head_is_the_cross_direction_consensus_head():
    # L9/H5 is the primary alignment head in all three calibrated directions
    # (hunyuan_v1_dense consensus); a re-calibration that demotes it should
    # be a conscious choice, not an accident.
    for src, tgt in _DIRECTIONS:
        cal = load_calibration(None, _REPO, src, tgt)
        assert cal.heads[0] == (9, 5), f"{src}->{tgt}: top head {cal.heads[0]}"
