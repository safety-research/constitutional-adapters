"""The item-cluster paired percentile bootstrap over ratios for the IFEval canonical battery.

Pinned: on a k = 1 BINARY fixture the generalised estimator equals ``tinycap_math500_thinking.paired_bootstrap_conditional``
BIT FOR BIT (same index draws, same exact-rational arithmetic, same replicate signs, same bounds) — the precedent of
``test_tinycap_math500_thinking.py``'s pin against pairing; the refusals (differing id sets, a bad triple, a zero-scored
leg, an unknown leg); at k = 2 the point is Σnum/Σden and the instruction-level triples weight by ``n_inst``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sisteer.capeval.stats import item_bootstrap as IB
from sisteer.capeval.tiers import pairing as pr

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("tinycap_math500_thinking", "scripts/tinycap_math500_thinking.py")


def _four_legs(n=200):
    ids = [f"g_{i:04x}" for i in range(n)]

    def leg(correct, unscored=()):
        return {i: ((0, None) if j in unscored else (1, 1.0 if j in correct else 0.0)) for j, i in enumerate(ids)}

    return {"none_off": leg(set(range(120))), "psr_off": leg(set(range(107)) | set(range(150, 163))),
            "none_on": leg(set(range(126)), unscored={196, 197}), "psr_on": leg(set(range(143)), unscored={198, 199})}


CONTRASTS = {"psr_minus_none_off": {"psr_off": 1, "none_off": -1}, "psr_minus_none_on": {"psr_on": 1, "none_on": -1},
             "interaction": {"psr_on": 1, "none_on": -1, "psr_off": -1, "none_off": 1}}


def test_bit_for_bit_equal_to_the_math_bootstrap_on_a_k1_binary_fixture():
    legs = _four_legs()
    want = M.paired_bootstrap_conditional(legs, CONTRASTS, n_resamples=2000, seed=0)
    got = IB.paired_item_bootstrap_ratios(IB.binary_legs(legs), CONTRASTS, n_resamples=2000, seed=0)
    for top in ("B", "seed", "level", "n_items", "n_scored_by_leg", "n_censored_by_leg", "n_correct_by_leg",
                "n_degenerate_replicates", "n_replicates_used", "acc_by_leg", "complete_case"):
        assert got[top] == want[top], top
    assert set(got["stats"]) == set(want["stats"])
    for c in want["stats"]:
        assert got["stats"][c] == want["stats"][c], c
    # a different seed / B moves the interval — the draws are a function of (seed, n, B)
    other = IB.paired_item_bootstrap_ratios(IB.binary_legs(legs), CONTRASTS, n_resamples=2000, seed=1)
    assert other["stats"]["psr_minus_none_off"]["replicate_signs"] != got["stats"]["psr_minus_none_off"]["replicate_signs"]


def test_refusals_name_the_defect():
    legs = IB.binary_legs(_four_legs())
    with pytest.raises(IB.ItemBootstrapError, match="ATTEMPTED id sets differ"):
        IB.paired_item_bootstrap_ratios({**legs, "none_off": {k: v for k, v in list(legs["none_off"].items())[:-1]}}, CONTRASTS, n_resamples=10)
    with pytest.raises(IB.ItemBootstrapError, match="not among the legs"):
        IB.paired_item_bootstrap_ratios(legs, {"x": {"nope": 1}}, n_resamples=10)
    with pytest.raises(IB.ItemBootstrapError, match="nonzero integer"):
        IB.paired_item_bootstrap_ratios(legs, {"x": {"none_off": 0}}, n_resamples=10)
    bad = {k: dict(v) for k, v in legs.items()}
    bad["none_off"]["g_0000"] = (2, 1, 1)
    with pytest.raises(IB.ItemBootstrapError, match="0 <= num <= den <= den_full"):
        IB.paired_item_bootstrap_ratios(bad, CONTRASTS, n_resamples=10)
    zero = {k: dict(v) for k, v in legs.items()}
    zero["none_off"] = {i: (0, 0, 1) for i in zero["none_off"]}
    with pytest.raises(IB.ItemBootstrapError, match="scored no draw"):
        IB.paired_item_bootstrap_ratios(zero, CONTRASTS, n_resamples=10)
    with pytest.raises(IB.ItemBootstrapError, match="interval level"):
        IB.paired_item_bootstrap_ratios(legs, CONTRASTS, n_resamples=10, ci=1.0)


def test_k2_ratio_estimands_and_the_ifeval_triples():
    draws = [
        {"item_id": "a", "sample_idx": 0, "scored": True, "prompt_level_strict": True, "inst_level_strict": 3, "num_instructions": 3},
        {"item_id": "a", "sample_idx": 1, "scored": True, "prompt_level_strict": False, "inst_level_strict": 2, "num_instructions": 3},
        {"item_id": "b", "sample_idx": 0, "scored": True, "prompt_level_strict": False, "inst_level_strict": 0, "num_instructions": 1},
        {"item_id": "b", "sample_idx": 1, "scored": False, "prompt_level_strict": None, "inst_level_strict": None, "num_instructions": 1},
    ]
    p = IB.ifeval_leg_triples(draws, metric="prompt_level_strict")
    assert p == {"a": (1, 2, 2), "b": (0, 1, 2)}
    i = IB.ifeval_leg_triples(draws, metric="inst_level_strict")
    assert i == {"a": (5, 6, 6), "b": (0, 1, 2)}
    legs = {"cell": p, "none": {"a": (2, 2, 2), "b": (1, 2, 2)}}
    est = IB.paired_item_bootstrap_ratios(legs, {"cell_minus_none": {"cell": 1, "none": -1}}, n_resamples=200, seed=0)
    assert est["acc_by_leg"]["cell"]["point"] == 1 / 3 and est["acc_by_leg"]["none"]["point"] == 0.75
    assert est["acc_by_leg"]["cell"]["n_censored"] == 1 and est["acc_by_leg"]["cell"]["acc_plus"] == 0.5
    assert est["stats"]["cell_minus_none"]["point"] == 1 / 3 - 0.75
    assert est["stats"]["cell_minus_none"]["ci_low"] <= est["stats"]["cell_minus_none"]["point"] <= est["stats"]["cell_minus_none"]["ci_high"]
    assert est["complete_case"]["cell_minus_none"] == {"n_items": 2, "delta": float(1 / 3 - 0.75)}
    assert (pr.DEFAULT_BOOTSTRAP, pr.DEFAULT_BOOTSTRAP_SEED, pr.DEFAULT_CI) == (10_000, 0, 0.95)
