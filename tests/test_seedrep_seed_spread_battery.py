"""The battery extensions of seed_spread.py: the seed-spread ratio and its inputs
— binomial / seeded-bootstrap per-seed SE, the RMS pooling, the triplet-gated paired contrast vs the same-seed anchor and
the per-arm summary. Every assertion is a NUMBER (float64) or an exact status string; each statistic is also tested on a
KNOWN-BAD input (an incomplete triplet, a zero SE, a NaN, a missing SE) and must refuse or mark unavailable, never yield 0."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SEEDREP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load("seed_spread")
tg = sys.modules["triplet_gate"]
Record = tg.Record


# ---------------------------------------------------------------- per-seed SE
def test_binomial_se_in_pp_and_its_zero_and_bad_inputs():
    se, why = ss.binomial_se(150, 500, scale=100.0)
    assert se == pytest.approx(100.0 * math.sqrt(0.30 * 0.70 / 500), abs=1e-12) and why is None
    assert ss.binomial_se(150, 500)[0] == pytest.approx(0.020493901531919196, abs=1e-15)      # fraction units
    se0, why0 = ss.binomial_se(0, 200, scale=100.0)
    assert se0 == 0.0 and why0.startswith("k ∈ {0, n}")                                          # 0 with a reason, never silent
    se1, why1 = ss.binomial_se(200, 200)
    assert se1 == 0.0 and "k = 200, n = 200" in why1
    assert ss.binomial_se(None, 200) == (None, "no k/n on the reading")
    assert ss.binomial_se(5, 0) == (None, "n = 0")
    assert ss.binomial_se(7, 5)[0] is None and "outside" in ss.binomial_se(7, 5)[1]


def test_bootstrap_se_is_seeded_from_identity_and_float64():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a, wa = ss.bootstrap_se(vals, identity=("petri", "none", 1))
    b, _ = ss.bootstrap_se(vals, identity=("petri", "none", 1))
    c, _ = ss.bootstrap_se(vals, identity=("petri", "none", 2))
    assert wa is None and a == b                                   # same identity → byte-identical draws
    assert a != c                                                  # a different cell → different draws
    assert a == pytest.approx(0.8963198130699921, abs=1e-12)       # pinned value (numpy default_rng, 4000 draws)
    # ≈ sd/sqrt(n) = 3.0277/3.1623 = 0.9574; the bootstrap of the mean is biased low by sqrt((n-1)/n) → 0.9083 expected
    assert abs(a - 0.9083) < 0.03
    assert ss.bootstrap_se([4.0], identity=("x",)) == (None, "1 item(s): bootstrap SE needs ≥ 2")
    assert ss.bootstrap_se([1.0, float("nan")], identity=("x",))[1] == "NaN among the per-item values"
    assert ss.bootstrap_se([1, 2, 3], identity=("x",), scale=100.0)[0] == pytest.approx(100.0 * ss.bootstrap_se([1, 2, 3], identity=("x",))[0])


def test_item_seed_is_stable_and_identity_sensitive():
    assert ss.item_seed("pasttense", "none", 1, "bootstrap_se") == ss.item_seed("pasttense", "none", 1, "bootstrap_se")
    assert ss.item_seed("pasttense", "none", 1) != ss.item_seed("pasttense", "none", 2)
    assert 0 <= ss.item_seed("a") < 2 ** 32


def test_pooled_se_is_the_rms_and_refuses_a_missing_seed():
    se, why = ss.pooled_se({1: 3.0, 2: 4.0, 3: 0.0})
    assert why is None and se == pytest.approx(math.sqrt((9 + 16 + 0) / 3))
    assert ss.pooled_se({1: 3.0, 2: 4.0}) == (None, "no per-seed SE for s3")
    assert ss.pooled_se({1: None, 2: 4.0, 3: 1.0}) == (None, "no per-seed SE for s1")


# ---------------------------------------------------------------- the ratio
def test_seed_spread_ratio_known_numbers():
    # pasttense `none` of the battery fixture: 30 / 34 / 32 pp at n = 500 → SD 2.0; SEs 2.0494 / 2.1185 / 2.0861
    ses = {s: ss.binomial_se(k, 500, scale=100.0)[0] for s, k in zip((1, 2, 3), (150, 170, 160))}
    r = ss.seed_spread_ratio({1: 30.0, 2: 34.0, 3: 32.0}, ses)
    assert r["status"] == "ok" and r["sd_between"] == pytest.approx(2.0)
    assert r["se_within"] == pytest.approx(2.0848661, abs=1e-6)
    assert r["ratio"] == pytest.approx(0.9592942, abs=1e-6)
    assert r["sd_excess"] == 0.0                                   # SD_between² < SE² → clipped at 0, never negative
    assert r["definition"] == ss.RATIO_DEFINITION and r["per_seed_se"]["s2"] == pytest.approx(ses[2])
    r2 = ss.seed_spread_ratio({1: 10.0, 2: 20.0, 3: 30.0}, {1: 2.0, 2: 2.0, 3: 2.0})
    assert r2["ratio"] == pytest.approx(5.0) and r2["sd_excess"] == pytest.approx(math.sqrt(100.0 - 4.0))


def test_seed_spread_ratio_known_bad_inputs_are_marked_never_zero():
    # incomplete triplet → unavailable (the caller passes None after the gate refused)
    r = ss.seed_spread_ratio(None, {1: 1.0, 2: 1.0, 3: 1.0})
    assert r["ratio"] is None and r["sd_between"] is None and r["status"].startswith("unavailable: incomplete replicate triplet")
    # a missing per-seed SE → unavailable, naming the seed
    r = ss.seed_spread_ratio({1: 1.0, 2: 2.0, 3: 3.0}, {1: 1.0, 2: None, 3: 1.0})
    assert r["ratio"] is None and r["sd_between"] == pytest.approx(1.0) and r["status"] == "unavailable: no per-seed SE for s2"
    # every SE zero (k ∈ {0, n} on every seed) → undefined, not infinite, not zero
    r = ss.seed_spread_ratio({1: 0.0, 2: 0.0, 3: 0.0}, {1: 0.0, 2: 0.0, 3: 0.0})
    assert r["ratio"] is None and r["status"].startswith("undefined: per-seed SE is 0 on every seed")
    # three identical readings with a non-zero SE → ratio 0 is the TRUE value (seeds identical), status ok
    r = ss.seed_spread_ratio({1: 12.0, 2: 12.0, 3: 12.0}, {1: 1.0, 2: 1.0, 3: 1.0})
    assert r["ratio"] == 0.0 and r["status"] == "ok"
    # two values only → the gate refuses (spread over survivors is the number this module exists to prevent)
    with pytest.raises(tg.TripletRefused, match="COMPLETE triplet"):
        ss.seed_spread_ratio({1: 1.0, 2: 2.0}, {1: 1.0, 2: 1.0, 3: 1.0})
    with pytest.raises(tg.TripletRefused, match="COMPLETE triplet"):
        ss.seed_spread_ratio({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}, {1: 1.0, 2: 1.0, 3: 1.0})


# ---------------------------------------------------------------- paired contrast + per-arm summary
def test_paired_contrast_uses_the_same_seed_anchor_and_gates_the_triplet():
    arm = {1: Record(1, 10.0), 2: Record(2, 12.0), 3: Record(3, 11.0)}
    none = {1: Record(1, 30.0), 2: Record(2, 34.0), 3: Record(3, 32.0)}
    c = ss.paired_contrast(arm, none)
    assert c["per_seed"] == {"s1": -20.0, "s2": -22.0, "s3": -21.0} and c["status"] == "ok"
    assert c["mean"] == pytest.approx(-21.0) and c["range"] == pytest.approx(2.0) and c["sd"] == pytest.approx(1.0)
    assert c["min"] == -22.0 and c["max"] == -20.0 and c["missing"] == []
    # a missing arm seed → unavailable; the two survivors are shown per seed but NO mean over them
    arm2 = {1: Record(1, 10.0), 2: Record(2, 12.0), 3: Record(3, None, False, "no store (x)")}
    c2 = ss.paired_contrast(arm2, none)
    assert c2["mean"] is None and c2["range"] is None and c2["per_seed"] == {"s1": -20.0, "s2": -22.0, "s3": None}
    assert c2["status"] == "unavailable: incomplete replicate triplet (missing: s3 — arm: no store (x))"
    assert c2["missing"] == [{"seed": 3, "reason": "arm: no store (x)"}]
    # a missing ANCHOR seed voids that seed's delta too (never paired against another seed's none)
    none2 = {1: Record(1, 30.0), 2: Record(2, None, False, "no store"), 3: Record(3, 32.0)}
    c3 = ss.paired_contrast(arm, none2)
    assert c3["mean"] is None and "s2 — anchor: no store" in c3["status"]
    # an anchor record absent altogether
    c4 = ss.paired_contrast(arm, {1: none[1], 3: none[3]})
    assert "s2 — anchor: no record" in c4["status"]


def test_arm_seed_summary_complete_and_incomplete():
    readings = {1: Record(1, 30.0), 2: Record(2, 34.0), 3: Record(3, 32.0)}
    ses = {1: 2.0, 2: 2.0, 3: 2.0}
    a = ss.arm_seed_summary(readings, ses)
    assert a["coverage"] == "3/3" and a["n_seeds"] == 3 and a["status"] == "ok"
    assert a["mean"] == 32.0 and a["sd"] == pytest.approx(2.0) and a["range"] == 4.0 and a["min"] == 30.0 and a["max"] == 34.0
    assert a["seed_spread"]["ratio"] == pytest.approx(1.0) and a["values"] == {"s1": 30.0, "s2": 34.0, "s3": 32.0}
    b = ss.arm_seed_summary({1: Record(1, 30.0), 2: Record(2, 34.0)}, {1: 2.0, 2: 2.0, 3: None})
    assert b["coverage"] == "2/3" and b["n_seeds"] == 2
    assert b["mean"] is None and b["sd"] is None and b["range"] is None                 # NEVER the mean of the two survivors
    assert b["status"] == "unavailable: incomplete replicate triplet (missing: s3 — no store)"
    assert b["missing"] == [{"seed": 3, "reason": "no store"}] and b["values"]["s3"] is None
    assert b["seed_spread"]["ratio"] is None and b["seed_spread"]["status"].startswith("unavailable")
    c = ss.arm_seed_summary({1: Record(1, 30.0), 2: Record(2, 34.0), 3: Record(3, None, False, "held: rerun")}, ses)
    assert c["coverage"] == "2/3" and "s3 — held: rerun" in c["status"]
    # a duplicate replicate hidden under a FOREIGN key (the dict-interface trap the gate documents) refuses: the key and
    # the record's own seed disagree
    with pytest.raises(tg.TripletRefused, match="mapping key s4 holds a record for s3"):
        ss.arm_seed_summary({1: Record(1, 30.0), 2: Record(2, 34.0), 3: Record(3, 32.0), 4: Record(3, 33.0)}, ses)
    with pytest.raises(tg.UnexpectedSeed):
        ss.arm_seed_summary({1: Record(1, 30.0), 2: Record(2, 34.0), 3: Record(3, 32.0), 0: Record(0, 33.0)}, ses)
    with pytest.raises(tg.TripletRefused):
        ss.paired_contrast({1: Record(1, 1.0), 2: Record(1, 2.0), 3: Record(3, 3.0)}, {s: Record(s, 0.0) for s in (1, 2, 3)})
