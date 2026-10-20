"""Per-scenario dispersion of Petri judge scores — the third term beside the
mean shift and the coverage of a paired contrast (sisteer.capeval.petri.
scenario_dispersion).

Every assertion is a DERIVATION on constructed scores (textbook SD via the
standard library, the existing read-out's bootstrap draw scheme re-stated
here independently), never a literal copied from a filed output. CPU-only,
numpy only."""

import math
import statistics

import numpy as np
import pytest

from sisteer.capeval.petri import scenario_dispersion as sd

SIDS = [f"scenario_{i:02d}" for i in range(10)]
A = {s: float(v) for s, v in zip(SIDS, [3, 5, 4, 6, 2, 7, 5, 8, 4, 6])}
SHIFT = [0, 1, -1, 2, 0, 3, 1, -2, 4, 2]
B = {s: A[s] + e for s, e in zip(SIDS, SHIFT)}
N_BOOT = 2_000


def _existing_paired_bootstrap(a, b, n_boot, seed):
    """The campaign read-out's paired mean difference, re-stated: shared ids
    sorted, default_rng(seed).integers(0, n, (n_boot, n)), sorted replicate
    means, m[int(0.025 n_boot)] and m[int(0.975 n_boot) - 1]."""
    ids = sorted(set(a) & set(b))
    d = np.asarray([b[i] - a[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = np.sort(d[idx].mean(axis=1))
    return float(d.mean()), float(m[int(0.025 * n_boot)]), float(m[int(0.975 * n_boot) - 1])


# --- the per cell x behaviour SD ---------------------------------------------------


def test_sample_sd_is_the_textbook_sample_sd():
    vals = [2, 4, 4, 4, 5, 5, 7, 9]
    mean = sum(vals) / len(vals)
    textbook = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
    assert sd.sample_sd(vals) == pytest.approx(textbook, abs=1e-12)
    assert sd.sample_sd(vals) == pytest.approx(statistics.stdev(vals), abs=1e-12)
    assert sd.sample_sd([7.0]) is None
    assert sd.sample_sd([]) is None


def test_dispersion_block_carries_n_interval_and_the_small_n_caveat():
    vals = list(A.values())
    blk = sd.dispersion_block(vals, n_boot=N_BOOT, seed=0)
    assert blk["n"] == 10 and blk["ddof"] == 1
    assert blk["sd"] == pytest.approx(statistics.stdev(vals), abs=1e-12)
    # the percentile bootstrap of the SD brackets the point estimate on a
    # spread-out sample, and is a genuine interval
    assert blk["sd_lo"] < blk["sd"] < blk["sd_hi"]
    # n = 10 is small: the flag is set and the caveat names the n in the OUTPUT
    assert blk["small_n"] is True
    assert "n=10" in blk["caveat"] and "1..10" in blk["caveat"]
    # a wide sample is not flagged
    wide = sd.dispersion_block([float(i % 7) for i in range(40)], n_boot=200, seed=0)
    assert wide["n"] == 40 and wide["small_n"] is False and wide["caveat"] is None
    # one value: no SD, no interval, still counted
    one = sd.dispersion_block([5.0], n_boot=50, seed=0)
    assert one == {**one, "n": 1, "sd": None, "sd_lo": None, "sd_hi": None,
                   "small_n": True}


# --- the paired contrast: sd_diff, d_z, d_pooled ------------------------------------


def test_paired_dispersion_derives_sd_diff_dz_and_dpooled_from_the_shared_scenarios():
    out = sd.paired_dispersion(A, B, n_boot=N_BOOT, seed=0)
    diffs = [B[s] - A[s] for s in sorted(A)]
    assert out["n_paired"] == 10
    assert out["delta"] == pytest.approx(statistics.fmean(diffs), abs=1e-12)
    assert out["sd_diff"] == pytest.approx(statistics.stdev(diffs), abs=1e-12)
    assert out["d_z"] == pytest.approx(statistics.fmean(diffs) / statistics.stdev(diffs), abs=1e-12)
    pooled = math.sqrt((statistics.variance(list(A.values()))
                        + statistics.variance(list(B.values()))) / 2)
    assert out["pooled_sd"] == pytest.approx(pooled, abs=1e-12)
    assert out["d_pooled"] == pytest.approx(statistics.fmean(diffs) / pooled, abs=1e-12)
    # sign convention: b minus a, so a cell scoring higher than bare is positive
    assert out["delta"] > 0 and out["d_z"] > 0 and out["d_pooled"] > 0
    # intervals bracket their point estimates and are ordered
    for name in ("sd_diff", "d_z", "d_pooled"):
        assert out[f"{name}_lo"] <= out[name] <= out[f"{name}_hi"], name
    # every replicate is accounted for: finite + degenerate = n_boot
    assert out["d_z_n_boot_finite"] + out["d_z_n_boot_degenerate"] == N_BOOT
    assert out["d_pooled_n_boot_finite"] + out["d_pooled_n_boot_degenerate"] == N_BOOT
    # the definitions ride with the numbers
    assert set(out["definitions"]) == {"sd_diff", "d_z", "d_pooled", "interval"}


def test_delta_and_its_interval_are_identical_to_the_existing_paired_mean_difference():
    """Same rows, same draws: the dispersion block's delta / delta_lo /
    delta_hi are bit-identical to the read-out's existing paired mean
    difference, so the two terms cannot disagree."""
    for seed in (0, 1, 7):
        out = sd.paired_dispersion(A, B, n_boot=N_BOOT, seed=seed)
        delta, lo, hi = _existing_paired_bootstrap(A, B, N_BOOT, seed)
        assert (out["delta"], out["delta_lo"], out["delta_hi"]) == (delta, lo, hi)


def test_constant_shift_leaves_dz_undefined_with_the_reason_stated():
    b_const = {s: v + 2.0 for s, v in A.items()}
    out = sd.paired_dispersion(A, b_const, n_boot=N_BOOT, seed=0)
    assert out["delta"] == pytest.approx(2.0) and out["sd_diff"] == 0.0
    assert out["d_z"] is None and out["d_z_lo"] is None and out["d_z_hi"] is None
    assert "sd_diff is 0" in out["d_z_reason"]
    # the pooled variant is still defined: both cells vary across scenarios
    assert out["pooled_sd"] == pytest.approx(statistics.stdev(list(A.values())), abs=1e-12)
    assert out["d_pooled"] == pytest.approx(2.0 / out["pooled_sd"], abs=1e-12)


def test_degenerate_replicates_are_counted_not_dropped_silently():
    a = {"s1": 5.0, "s2": 5.0, "s3": 5.0}
    b = {"s1": 5.0, "s2": 5.0, "s3": 6.0}      # diffs 0, 0, 1
    out = sd.paired_dispersion(a, b, n_boot=N_BOOT, seed=0)
    # a replicate drawing the same scenario three times has sd 0: at n = 3
    # that happens in a non-trivial share of draws and is REPORTED
    assert out["d_z_n_boot_degenerate"] > 0
    assert out["d_z_n_boot_finite"] + out["d_z_n_boot_degenerate"] == N_BOOT
    assert math.isfinite(out["d_z_lo"]) and math.isfinite(out["d_z_hi"])
    assert out["d_z"] == pytest.approx((1 / 3) / statistics.stdev([0, 0, 1]), abs=1e-12)


def test_shared_ids_only_pending_when_disjoint_and_one_shared_is_not_estimable():
    b_extra = {**B, "scenario_99": 9.0, "scenario_98": 1.0}
    out = sd.paired_dispersion(A, b_extra, n_boot=200, seed=0)
    assert out["n_paired"] == 10       # the extras never enter
    assert out["delta"] == pytest.approx(sd.paired_dispersion(A, B, n_boot=200, seed=0)["delta"])
    assert sd.paired_dispersion(A, {"other": 1.0}) == {"status": "PENDING", "reason": "no shared scenarios"}
    one = sd.paired_dispersion({"s": 3.0, "t": 4.0}, {"s": 5.0, "u": 1.0}, n_boot=200, seed=0)
    assert one["n_paired"] == 1 and one["delta"] == 2.0
    assert one["sd_diff"] is None and one["d_z"] is None and one["d_pooled"] is None
    assert "n_paired < 2" in one["d_z_reason"]


def test_small_n_caveat_is_in_the_paired_output_and_absent_at_wide_n():
    out = sd.paired_dispersion(A, B, n_boot=200, seed=0)
    assert out["small_n"] is True and "n=10" in out["caveat"]
    wide_a = {f"s{i}": float(i % 5) for i in range(40)}
    wide_b = {f"s{i}": float(i % 5) + (i % 3) for i in range(40)}
    wide = sd.paired_dispersion(wide_a, wide_b, n_boot=200, seed=0)
    assert wide["n_paired"] == 40 and wide["small_n"] is False and wide["caveat"] is None


def test_meta_states_the_definitions_threshold_and_scale():
    m = sd.meta()
    assert m["small_n_threshold"] == sd.SMALL_N and m["ddof"] == 1
    assert m["score_scale"] == [1.0, 10.0]
    assert set(m["definitions"]) >= {"sd", "sd_diff", "d_z", "d_pooled", "interval", "small_n"}
    assert "<n>" in m["small_n_caveat"]


# --- the per-cell pooled paired d_z -------------------------------------------------

BEH2 = {"b1": {s: v for s, v in zip(SIDS[:4], [2.0, 5.0, 4.0, 7.0])},
        "b2": {s: v for s, v in zip(SIDS[:5], [1.0, 1.0, 3.0, 6.0, 8.0])},
        "b3_only_in_bare": {SIDS[0]: 9.0}}
SHIFT2 = {"b1": [1, -1, 2, 0], "b2": [0, 3, -2, 1, 4]}
CELL2 = {"b1": {s: BEH2["b1"][s] + e for s, e in zip(SIDS[:4], SHIFT2["b1"])},
         "b2": {s: BEH2["b2"][s] + e for s, e in zip(SIDS[:5], SHIFT2["b2"])},
         "b4_only_in_cell": {SIDS[1]: 2.0}}
CELL2["b2"][SIDS[9]] = 5.0          # a scenario bare never scored: never enters


def test_cell_paired_dispersion_pools_the_shared_items_and_derives_dz():
    out = sd.cell_paired_dispersion(BEH2, CELL2, n_boot=N_BOOT, seed=0)
    diffs = [CELL2["b1"][s] - BEH2["b1"][s] for s in SIDS[:4]] + [CELL2["b2"][s] - BEH2["b2"][s] for s in SIDS[:5]]
    assert out["n"] == 9 and out["n_behaviors"] == 2
    assert out["n_per_behavior"] == {"b1": 4, "b2": 5}      # b3 / b4 / the unscored scenario never enter
    assert out["delta"] == pytest.approx(statistics.fmean(diffs), abs=1e-12)
    assert out["sd_diff"] == pytest.approx(statistics.stdev(diffs), abs=1e-12)
    assert out["d_z"] == pytest.approx(statistics.fmean(diffs) / statistics.stdev(diffs), abs=1e-12)
    assert out["d_z_lo"] <= out["d_z"] <= out["d_z_hi"]
    assert out["d_z_n_boot_finite"] + out["d_z_n_boot_degenerate"] == N_BOOT
    # the cell delta is the item-weighted mean of the per-behaviour deltas — derived, not assumed
    per = {beh: sd.paired_dispersion(BEH2[beh], CELL2[beh], n_boot=50, seed=0)["delta"] for beh in ("b1", "b2")}
    weighted = (per["b1"] * 4 + per["b2"] * 5) / 9
    assert out["delta"] == pytest.approx(weighted, abs=1e-12)
    assert "n_per_behavior" in out["pooling"] and "cell_d_z" in out["definitions"]


def test_cell_paired_dispersion_bootstrap_is_the_existing_scheme_over_the_pooled_vector():
    """The per-cell interval is the read-out's own paired bootstrap applied
    to the pooled difference vector in sorted (behaviour, scenario) order."""
    pooled_a = {f"{beh}|{s}": BEH2[beh][s] for beh in ("b1", "b2") for s in BEH2[beh]}
    pooled_b = {f"{beh}|{s}": CELL2[beh][s] for beh in ("b1", "b2") for s in CELL2[beh]}
    for seed in (0, 3):
        out = sd.cell_paired_dispersion(BEH2, CELL2, n_boot=N_BOOT, seed=seed)
        delta, lo, hi = _existing_paired_bootstrap(pooled_a, pooled_b, N_BOOT, seed)
        assert (out["delta"], out["delta_lo"], out["delta_hi"]) == (delta, lo, hi)


def test_cell_paired_dispersion_states_n_small_n_and_undefined_cases():
    # a complete 6 x 10 cell: n = 60, not small, but the pooling note still rides
    behs = [f"beh{k}" for k in range(6)]
    a = {beh: {s: float((i + k) % 7) for i, s in enumerate(SIDS)} for k, beh in enumerate(behs)}
    b = {beh: {s: a[beh][s] + ((i * (k + 1)) % 3) for i, s in enumerate(SIDS)} for k, beh in enumerate(behs)}
    out = sd.cell_paired_dispersion(a, b, n_boot=200, seed=0)
    assert out["n"] == 60 and out["n_behaviors"] == 6 and out["small_n"] is False and out["caveat"] is None
    assert out["n_per_behavior"] == {beh: 10 for beh in behs} and "pooled" in out["pooling"]
    # a 6 x 10 cell with a single scenario missing from one behaviour: n = 59 and the weights say which
    b2 = {beh: dict(v) for beh, v in b.items()}; b2["beh3"].pop(SIDS[4])
    assert sd.cell_paired_dispersion(a, b2, n_boot=50, seed=0)["n_per_behavior"]["beh3"] == 9
    # constant shift: d_z undefined, reason stated, delta intact
    const = {beh: {s: v + 1.5 for s, v in a[beh].items()} for beh in behs}
    c = sd.cell_paired_dispersion(a, const, n_boot=50, seed=0)
    assert c["delta"] == pytest.approx(1.5) and c["sd_diff"] == 0.0 and c["d_z"] is None and "sd_diff is 0" in c["d_z_reason"]
    # nothing shared: PENDING like the existing term; one item: not estimable
    assert sd.cell_paired_dispersion({"x": {"s": 1.0}}, {"y": {"s": 2.0}})["status"] == "PENDING"
    one = sd.cell_paired_dispersion({"x": {"s": 1.0}}, {"x": {"s": 3.0}}, n_boot=50, seed=0)
    assert one["n"] == 1 and one["delta"] == 2.0 and one["d_z"] is None and "n < 2" in one["d_z_reason"]
    assert one["small_n"] is True and "n=1" in one["caveat"]
