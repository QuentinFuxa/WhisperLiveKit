"""The bundled zh→en calibration file must reproduce the commit decisions the
calibrated engine actually made.

Three layers of proof, in increasing strength:
1. The file loads through the real loader and covers the direction that
   previously crashed at engine construction (zh→en).
2. Its prompt matches the resolved model profile byte-for-byte.
3. Its head indices — applied via upstream's paper commit policy to REAL
   captured attention slices from the calibrated zh→en run — reproduce the
   commit decisions recorded during that run. A corrupted, reordered, or
   wrong-direction file fails here even though it still parses.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from whisperlivekit.simul_mt_calibration import load_calibration
from whisperlivekit.simul_mt_capture import AttentionRows, apply_commit_policy
from whisperlivekit.translation_profiles import MT_MODEL_PROFILES, resolve_prompt

_REPO = "mlx-community/Hy-MT2-1.8B-8bit"
_PROFILE = MT_MODEL_PROFILES["hy-mt2-1.8b-8bit"]
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "zh_en_attention_golden.npz"


def _load_cal():
    return load_calibration(None, _REPO, "zh", "en")


def test_bundled_zh_en_calibration_loads():
    cal = _load_cal()
    assert len(cal.heads) == 8 and len(set(cal.heads)) == 8


def test_bundled_calibration_prompt_matches_model_profile():
    cal = _load_cal()
    assert cal.prompt == resolve_prompt(_PROFILE, "zh", "en")


def test_top_head_is_the_cross_direction_consensus_head():
    # L9/H5 is the primary alignment head in every calibrated direction
    # (hunyuan_v1_dense consensus); a re-calibration that demotes it should
    # be a conscious choice, not an accident.
    assert _load_cal().heads[0] == (9, 5)


# --- golden-attention replay -------------------------------------------------

@pytest.fixture(scope="module")
def golden():
    data = np.load(_FIXTURE, allow_pickle=False)
    meta = json.loads(bytes(data["meta"]).decode())
    return data, meta


def _reconstruct_capture(data, rec_idx, meta):
    """Build an upstream-format capture dict from the recorded raw slices."""
    pl = meta["prompt_length"]
    cap = {}
    for (l, h) in [tuple(x) for x in meta["heads"]]:
        rows = data[f"rec{rec_idx}_head_{l}_{h}"]  # (T, pl)
        cap[(l, h)] = tuple(
            AttentionRows(query_start=pl + t, weights=rows[t:t + 1])
            for t in range(rows.shape[0])
        )
    return cap


def test_bundled_heads_reproduce_the_calibrated_run(golden):
    cal = _load_cal()
    data, meta = golden
    # the file's head set must equal the heads the measured engine used
    assert [list(h) for h in cal.heads] == meta[0]["heads"]
    for i, m in enumerate(meta):
        cap = _reconstruct_capture(data, i, m)
        got = apply_commit_policy(
            cap, list(cal.heads), m["n_tokens"], m["prompt_length"],
            m["src_start"], m["src_end"], m["cend"],
            mode="paper", mass_threshold=m["mass_threshold"],
        )
        assert got == m["result"], (
            f"record {i}: bundled heads commit {got} tokens, "
            f"the calibrated engine committed {m['result']}"
        )


def test_wrong_calibration_is_detected(golden):
    """Sensitivity control: the paper policy is deliberately robust to head
    noise (per-head z-scores, head-averaged), so corruption is detected at
    two tiers — a small perturbation (top-1 head) must move the stabilized
    argmax trajectory, and a larger one (top-3 heads) must flip at least one
    recorded commit decision. A head file that survives both is behaviorally
    equivalent on these slices, which is the only claim the golden test makes."""
    data, meta = golden
    alt_indices = sorted({
        int(k.split("_")[2]) for k in data.files
        if k.startswith("rec0_alt_")
    })
    assert len(alt_indices) >= 3, "fixture lacks alternative heads"
    base_heads = [tuple(h) for h in meta[0]["heads"]]

    def build(i, heads):
        m = meta[i]
        pl = m["prompt_length"]
        cap = {}
        for (l, h) in heads:
            key = (f"rec{i}_head_{l}_{h}" if (l, h) in base_heads
                   else f"rec{i}_alt_{h}")
            rows = data[key]  # (T, pl)
            cap[(l, h)] = tuple(
                AttentionRows(query_start=pl + t, weights=rows[t:t + 1])
                for t in range(rows.shape[0])
            )
        return cap

    def decisions(i, heads):
        m = meta[i]
        return apply_commit_policy(
            build(i, heads), [tuple(h) for h in heads], m["n_tokens"],
            m["prompt_length"], m["src_start"], m["src_end"], m["cend"],
            mode="paper", mass_threshold=m["mass_threshold"],
        )

    def argmax_trajectory(i, heads):
        m = meta[i]
        cap = build(i, heads)
        per_head = []
        for (l, h) in heads:
            rows = {}
            for e in cap[(l, h)]:
                for idx, row in enumerate(np.asarray(e.weights)):
                    rows[e.query_start + idx] = row[m["src_start"]:m["src_end"]]
            per_head.append(rows)
        available = 0
        while (available < m["n_tokens"]
               and all(m["prompt_length"] + available in r for r in per_head)):
            available += 1
        vals = np.array([[r[m["prompt_length"] + i] for r in per_head]
                         for i in range(available)])
        from whisperlivekit.simul_mt_capture import _paper_stabilized_argmax
        return _paper_stabilized_argmax(vals)

    # tier 1 (small perturbation): top-1 head swapped must move the
    # stabilized argmax trajectory on at least one record
    subs1 = dict(zip(base_heads[:1], [(9, a) for a in alt_indices[:1]]))
    heads1 = [subs1.get(h, h) for h in base_heads]
    traj_diff = sum(
        1 for i in range(len(meta))
        if not np.array_equal(argmax_trajectory(i, base_heads),
                              argmax_trajectory(i, heads1))
    )
    assert traj_diff > 0, "top-1 head substitution left every argmax trajectory unchanged"

    # tier 2 (larger corruption): top-3 heads swapped must flip at least one
    # recorded commit decision
    subs3 = dict(zip(base_heads[:3], [(9, a) for a in alt_indices[:3]]))
    heads3 = [subs3.get(h, h) for h in base_heads]
    dec_diff = sum(
        1 for i in range(len(meta)) if decisions(i, heads3) != meta[i]["result"]
    )
    assert dec_diff > 0, "top-3 head substitution left every commit decision unchanged"
