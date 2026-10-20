"""``pairing.paired_bootstrap_conditional_2x2``: the ONE estimand
of the thinking-transfer 2x2 and its SHARP contrast bounds.
Synthetic legs only. The existing pairing coverage lives in
``test_compare_served_hf_task.py`` and is untouched."""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import numpy as np
import pytest

from sisteer.capeval.tiers import pairing as pr

IDS = [f"item_{i:03d}" for i in range(200)]
N = len(IDS)


def leg(correct, *, unscored=(), ids=IDS):
    """A leg mapping: ``correct`` = the indices scored correct; ``unscored`` =
    the indices masked out (PRIMARY branch)."""
    correct, unscored = set(correct), set(unscored)
    return {i: ((0, None) if j in unscored else (1, 1.0 if j in correct else 0.0))
            for j, i in enumerate(ids)}


def legs_of(none_off, psr_off, none_on, psr_on):
    return {("none", "off"): none_off, ("psr", "off"): psr_off,
            ("none", "on"): none_on, ("psr", "on"): psr_on}


# -- determinism, zero effects, agreement with the complete-case tool -------------


def test_replicate_signs_come_from_the_replicate_arrays_and_one_sided_discordance_binds_them_only_without_censoring():
    """Every contrast files ``replicate_signs`` — the negative / zero /
    positive counts of its kept replicates, exact zeros (one integer division
    per replicate). A one-sided complete-case discordance pins every replicate
    to one side ONLY when no mask is resampled (then the conditional bootstrap
    is the empirical bootstrap of the pairs); with items censored in one leg
    and correct in the other, per-replicate denominators let replicates cross
    zero even though every complete-case discordance is a loss."""
    off = (leg(range(120)), leg(set(range(110)) | set(range(120, 130))))
    # no censoring, 10 losses / 0 gains: not one positive replicate
    r = pr.paired_bootstrap_conditional_2x2(legs_of(*off, leg(range(150)), leg(range(140))), n_resamples=2000, seed=0)
    for c in pr.CONTRASTS:
        rs = r["stats"][c]["replicate_signs"]
        assert set(rs) == {"negative", "zero", "positive"} and all(isinstance(v, int) for v in rs.values())
        assert sum(rs.values()) == r["n_replicates_used"] == 2000
    s = r["stats"]["effect_on"]
    assert s["replicate_signs"]["positive"] == 0 and s["ci_high"] < 0
    # 20 items censored in none/on and correct in psr/on, the same 10 complete-case losses: the
    # registered bootstrap draws positives (a replicate heavy in those items lifts psr's numerator
    # and shrinks none's denominator) — the property does not apply literally
    r2 = pr.paired_bootstrap_conditional_2x2(
        legs_of(*off, leg(range(150), unscored=range(180, 200)), leg(set(range(140)) | set(range(180, 200)))),
        n_resamples=2000, seed=0)
    s2 = r2["stats"]["effect_on"]
    assert s2["replicate_signs"]["positive"] > 0
    assert sum(s2["replicate_signs"].values()) == r2["n_replicates_used"]
    # the counts agree with the percentile CI: an upper edge below zero means ≥ 97.5 % of replicates are negative
    for st, used in ((s, r["n_replicates_used"]), (s2, r2["n_replicates_used"])):
        if st["ci_high"] < 0:
            assert st["replicate_signs"]["negative"] / used >= 0.97
    # exact zeros: with two discordant items and no censoring, a replicate is at zero iff it holds neither
    r3 = pr.paired_bootstrap_conditional_2x2(legs_of(*off, leg(range(150)), leg(range(148))), n_resamples=2000, seed=0)
    z = r3["stats"]["effect_on"]["replicate_signs"]["zero"]
    assert z > 0 and z == pytest.approx(2000 * (1 - 2 / 200) ** 200, rel=0.35)


def test_deterministic_under_a_seed_and_zero_when_the_four_legs_are_identical():
    same = leg(range(120))
    legs = legs_of(same, dict(same), dict(same), dict(same))
    r1 = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=2000, seed=0)
    r2 = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=2000, seed=0)
    assert r1 == r2
    for c in pr.CONTRASTS:
        s = r1["stats"][c]
        assert s["point"] == 0.0 and s["ci_low"] == 0.0 and s["ci_high"] == 0.0
        assert s["bound_low"] == s["bound_high"] == 0.0
    assert r1["n_censored_by_leg"] == {k: 0 for k in ("none/off", "none/on", "psr/off", "psr/on")}
    r3 = pr.paired_bootstrap_conditional_2x2(legs_of(leg(range(120)), leg(range(150)), same, same),
                                             n_resamples=2000, seed=1)
    assert r3["stats"]["effect_off"]["ci_low"] != r1["stats"]["effect_off"]["ci_low"]


def test_with_every_mask_one_effect_off_equals_the_complete_case_paired_bootstrap():
    """The complete-case and conditional estimands coincide when nothing is
    unscored: same seed, same draw discipline (one index matrix per
    replicate), so the point and CI agree — to floating rounding (a difference
    of two quotients vs one quotient of the difference)."""
    none_off = leg(set(range(120)) | set(range(150, 163)))       # 133 correct
    psr_off = leg(set(range(107)) | set(range(120, 163)))     # 150 correct; discordant both ways
    legs = legs_of(none_off, psr_off, leg(range(126)), leg(range(143)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=10_000, seed=0)
    cc = pr.paired_bootstrap({i: v for i, (_, v) in none_off.items()},
                             {i: v for i, (_, v) in psr_off.items()}, n_resamples=10_000, seed=0)
    s = r["stats"]["effect_off"]
    assert abs(s["point"] - cc["delta"]) < 1e-12
    assert abs(s["ci_low"] - cc["ci_low"]) < 1e-12 and abs(s["ci_high"] - cc["ci_high"]) < 1e-12
    assert s["bound_low"] == s["bound_high"] == pytest.approx(s["point"])
    assert r["complete_case"]["effect_off"] == {"n_items": N, "delta": pytest.approx(cc["delta"])}
    assert r["complete_case"]["label"] == pr.COMPLETE_CASE_LABEL


# -- the synthetic where the two estimands disagree in sign ----------------------


def test_d12_synthetic_conditional_reads_minus_1_02_pp_where_the_intersection_reads_plus_1_04():
    """none 140/196 scored-correct, psr 138/196; the four items psr leaves
    unscored are all CORRECT in none, the four none leaves unscored all WRONG
    in psr -> the 192-item intersection reads +1.04 pp while the conditional
    estimand reads -1.02 pp. The intersection number comes back under
    ``complete_case``, labelled."""
    none_unscored, psr_unscored = {0, 1, 2, 3}, {4, 5, 6, 7}
    none_correct = set(range(4, 144))            # 140 scored-correct, incl. psr's unscored 4..7
    psr_correct = set(range(8, 146))             # 138 scored-correct; 0..3 (none's unscored) wrong
    assert len(none_correct) == 140 and len(psr_correct) == 138
    on = leg(range(100))
    legs = legs_of(leg(none_correct, unscored=none_unscored), leg(psr_correct, unscored=psr_unscored),
                   dict(on), dict(on))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=3000, seed=0)
    s = r["stats"]["effect_off"]
    assert s["point"] == pytest.approx(138 / 196 - 140 / 196)              # -1.02 pp
    cc = r["complete_case"]["effect_off"]
    assert cc["n_items"] == 192 and cc["delta"] == pytest.approx(2 / 192)   # +1.04 pp
    assert r["n_censored_by_leg"] == {"none/off": 4, "psr/off": 4, "none/on": 0, "psr/on": 0}
    assert r["n_scored_by_leg"]["none/off"] == 196
    # the sharp bound of effect_off: lower = acc⁻_psr − acc⁺_none, upper = acc⁺_psr − acc⁻_none
    assert s["bound_low"] == pytest.approx(138 / 200 - 144 / 200)
    assert s["bound_high"] == pytest.approx(142 / 200 - 140 / 200)
    assert s["bound_low"] < s["point"] < s["bound_high"]
    # the complete-case number is a DIFFERENT estimand: it need not lie inside the
    # bound (only the two branch points must, §6.1 (i)) — here it sits just outside
    assert not (s["bound_low"] <= cc["delta"] <= s["bound_high"])


# -- refusals ----------------------------------------------------------------------


def test_refuses_an_attempted_id_set_mismatch_listing_the_ids():
    legs = legs_of(leg(range(100)), leg(range(100)), leg(range(100)), leg(range(100)))
    short = dict(legs[("psr", "on")])
    del short["item_007"]
    del short["item_008"]
    legs[("psr", "on")] = short
    with pytest.raises(pr.PairingError, match=r"ATTEMPTED id sets differ.*psr/on lacks 2.*item_007"):
        pr.paired_bootstrap_conditional_2x2(legs, n_resamples=10)


def test_refuses_wrong_leg_keys_bad_masks_and_a_value_on_a_masked_out_item():
    legs = legs_of(leg(range(100)), leg(range(100)), leg(range(100)), leg(range(100)))
    with pytest.raises(pr.PairingError, match="exactly the four legs"):
        pr.paired_bootstrap_conditional_2x2({k: v for k, v in legs.items() if k != ("psr", "on")},
                                            n_resamples=10)
    bad = dict(legs)
    bad[("none", "on")] = {**legs[("none", "on")], "item_000": (2, 1.0)}
    with pytest.raises(pr.PairingError, match="mask 2 is not 0 or 1"):
        pr.paired_bootstrap_conditional_2x2(bad, n_resamples=10)
    bad[("none", "on")] = {**legs[("none", "on")], "item_000": (0, 1.0)}
    with pytest.raises(pr.PairingError, match="masked-out item carries no value"):
        pr.paired_bootstrap_conditional_2x2(bad, n_resamples=10)
    bad[("none", "on")] = {**legs[("none", "on")], "item_000": (1, 0.5)}
    with pytest.raises(pr.PairingError, match="value must be 0 or 1"):
        pr.paired_bootstrap_conditional_2x2(bad, n_resamples=10)
    with pytest.raises(pr.PairingError, match="interval level"):
        pr.paired_bootstrap_conditional_2x2(legs, n_resamples=10, ci=1.5)
    with pytest.raises(pr.PairingError, match=">= 1 resamples"):
        pr.paired_bootstrap_conditional_2x2(legs, n_resamples=0)


def test_degenerate_replicates_are_counted_and_refused_above_the_threshold():
    """A leg with ONE scored item: a replicate that never draws it has a zero
    denominator — discarded and counted; the default 0.1 % ceiling refuses,
    a raised ceiling reports the count."""
    ids = IDS[:20]
    nearly_all_censored = leg([0], unscored=range(1, 20), ids=ids)
    legs = legs_of(leg(range(10), ids=ids), leg(range(12), ids=ids), leg(range(10), ids=ids), nearly_all_censored)
    with pytest.raises(pr.PairingError, match=r"replicates had a leg with no scored item"):
        pr.paired_bootstrap_conditional_2x2(legs, n_resamples=500, seed=0)
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=500, seed=0, max_degenerate_frac=0.9)
    assert r["n_degenerate_replicates"] > 0
    assert r["n_replicates_used"] + r["n_degenerate_replicates"] == 500
    with pytest.raises(pr.PairingError, match="scored no item"):
        pr.paired_bootstrap_conditional_2x2(
            legs_of(leg(range(10), ids=ids), leg(range(12), ids=ids), leg(range(10), ids=ids),
                    leg([], unscored=range(20), ids=ids)), n_resamples=10)


# -- the score-as-is branch reproduces the summary's acc under that policy ---------


def test_score_as_is_masks_all_one_with_unterminated_wrong_reproduces_that_policy_acc():
    """PRIMARY: 4 unscored of 200, 143 correct -> acc 143/196. SCORE-AS-IS:
    the same items masked 1 with value 0 -> acc 143/200 (the all-wrong
    corner), which is also the primary bound's ``acc_minus``."""
    primary = leg(range(143), unscored=range(196, 200))
    as_is = {i: (1, 0.0) if m == 0 else (1, v) for i, (m, v) in primary.items()}
    off = leg(range(120))
    rp = pr.paired_bootstrap_conditional_2x2(legs_of(dict(off), dict(off), dict(off), primary), n_resamples=200)
    ra = pr.paired_bootstrap_conditional_2x2(legs_of(dict(off), dict(off), dict(off), as_is), n_resamples=200)
    assert rp["acc_by_leg"]["psr/on"]["point"] == pytest.approx(143 / 196)
    assert ra["acc_by_leg"]["psr/on"]["point"] == pytest.approx(143 / 200)
    assert ra["acc_by_leg"]["psr/on"]["point"] == pytest.approx(rp["acc_by_leg"]["psr/on"]["acc_minus"])
    assert ra["n_censored_by_leg"]["psr/on"] == 0 and rp["n_censored_by_leg"]["psr/on"] == 4
    # both branch points lie inside the primary bound (§6.1 (i))
    s = rp["stats"]["effect_on"]
    assert s["bound_low"] <= ra["stats"]["effect_on"]["point"] <= s["bound_high"]
    assert s["bound_low"] <= s["point"] <= s["bound_high"]


# -- the bounds -------------------------------------------------------------


def _hand_bounds(counts):
    """counts = {leg: (n_correct, n_scored)} over N attempted -> the sharp
    interval per contrast from leg-wise ADVERSE imputations, by hand."""
    minus = {k: c / N for k, (c, s) in counts.items()}
    plus = {k: (c + (N - s)) / N for k, (c, s) in counts.items()}
    out = {}
    for name, signs in pr.CONTRAST_SIGNS.items():
        lo = sum((minus[k] if sg > 0 else -plus[k]) for k, sg in signs.items())
        hi = sum((plus[k] if sg > 0 else -minus[k]) for k, sg in signs.items())
        out[name] = (lo, hi)
    return out


def test_bounds_are_points_without_censoring_and_the_hand_computed_adverse_interval_with_it():
    counts = {("none", "off"): (120, 200), ("psr", "off"): (150, 200),
              ("none", "on"): (126, 196), ("psr", "on"): (143, 196)}
    legs = legs_of(leg(range(120)), leg(range(150)),
                   leg(range(126), unscored=range(196, 200)), leg(range(143), unscored=range(196, 200)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=2000, seed=0)
    hand = _hand_bounds(counts)
    for c in pr.CONTRASTS:
        s = r["stats"][c]
        assert s["bound_low"] == pytest.approx(hand[c][0]) and s["bound_high"] == pytest.approx(hand[c][1])
        # the conditional point lies inside its bound (§6.1 (i))
        assert s["bound_low"] - 1e-12 <= s["point"] <= s["bound_high"] + 1e-12
        # the bound widened by sampling contains the point bound
        assert s["bound_ci_low"] <= s["bound_low"] + 1e-12 and s["bound_ci_high"] >= s["bound_high"] - 1e-12
    # OFF is a point; ON carries the whole width (4 + 4 censored -> 4 pp on effect_on)
    assert r["stats"]["effect_off"]["bound_low"] == r["stats"]["effect_off"]["bound_high"]
    assert r["stats"]["effect_on"]["bound_high"] - r["stats"]["effect_on"]["bound_low"] == pytest.approx(0.04)
    assert r["stats"]["effect_on"]["bound_low"] == pytest.approx(0.065)
    assert r["stats"]["effect_on"]["bound_high"] == pytest.approx(0.105)
    # the DiD's bound uses the four-leg assignment: psr/on & none/off at acc⁻, none/on & psr/off at acc⁺ (lower)
    did_lo = 143 / 200 - 130 / 200 - 150 / 200 + 120 / 200
    did_hi = 147 / 200 - 126 / 200 - 150 / 200 + 120 / 200
    assert r["stats"]["did"]["bound_low"] == pytest.approx(did_lo)
    assert r["stats"]["did"]["bound_high"] == pytest.approx(did_hi)


def test_the_contrast_bound_is_strictly_wider_than_the_all_legs_same_direction_interval():
    """Imputing every censored item the SAME way in every leg gives
    [acc⁻_psr − acc⁻_none, acc⁺_psr − acc⁺_none] — which does NOT bound the
    contrast when both legs are censored; the leg-wise adverse interval is
    strictly wider."""
    legs = legs_of(leg(range(120)), leg(range(150)),
                   leg(range(126), unscored=range(196, 200)), leg(range(143), unscored=range(190, 194)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=200, seed=0)
    a_none, a_psr = r["acc_by_leg"]["none/on"], r["acc_by_leg"]["psr/on"]
    same_lo = a_psr["acc_minus"] - a_none["acc_minus"]
    same_hi = a_psr["acc_plus"] - a_none["acc_plus"]
    s = r["stats"]["effect_on"]
    assert s["bound_low"] < min(same_lo, same_hi) and s["bound_high"] > max(same_lo, same_hi)
    assert s["bound_high"] - s["bound_low"] == pytest.approx(8 / 200)


def test_mutation_swapping_the_adverse_direction_would_flip_the_interval():
    """The bound logic both ways: reversing the leg-wise imputation direction
    (the +1 leg at acc⁺ and the −1 leg at acc⁻ for the LOWER end) produces an
    interval whose ends are SWAPPED — low > high — i.e. not an interval; and
    with zero censoring the bound collapses to the point. Pinned so a sign
    slip in ``CONTRAST_SIGNS``'s application cannot pass as a bound."""
    legs = legs_of(leg(range(120)), leg(range(150)),
                   leg(range(126), unscored=range(196, 200)), leg(range(143), unscored=range(196, 200)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=100, seed=0)
    a = r["acc_by_leg"]
    swapped_low = a["psr/on"]["acc_plus"] - a["none/on"]["acc_minus"]
    swapped_high = a["psr/on"]["acc_minus"] - a["none/on"]["acc_plus"]
    assert swapped_low > swapped_high                              # the mutant is not an interval
    assert r["stats"]["effect_on"]["bound_low"] == pytest.approx(swapped_high)
    assert r["stats"]["effect_on"]["bound_high"] == pytest.approx(swapped_low)
    none_censored = pr.paired_bootstrap_conditional_2x2(
        legs_of(leg(range(120)), leg(range(150)), leg(range(126)), leg(range(143))), n_resamples=100)
    for c in pr.CONTRASTS:
        s = none_censored["stats"][c]
        assert s["bound_low"] == s["bound_high"] == pytest.approx(s["point"])


def test_sign_identified_reads_the_bound_not_the_ci():
    """A contrast whose CI excludes zero but whose sharp bound straddles it has
    ``excludes_zero`` True and ``sign_identified`` False — the censoring, not
    the sampling, is what withholds the sign."""
    legs = legs_of(leg(range(120)), leg(range(150)),
                   leg(range(128), unscored=range(196, 200)), leg(range(130), unscored=range(196, 200)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=500, seed=0)
    s = r["stats"]["effect_on"]
    assert s["bound_low"] == pytest.approx(-0.01) and s["bound_high"] == pytest.approx(0.03)
    assert s["sign_identified"] is False
    assert r["stats"]["effect_off"]["sign_identified"] is True and r["stats"]["effect_off"]["excludes_zero"]


def test_replicate_ci_of_a_leg_accuracy_uses_its_own_denominator():
    """Own-denominator inside the replicate: with the 4 censored items all
    landing on WRONG-if-scored items the conditional acc never moves toward
    143/200 — every replicate's acc is a ratio of drawn correct to drawn
    scored, so its mean sits near 143/196, not 143/200."""
    legs = legs_of(leg(range(120)), leg(range(150)), leg(range(126), unscored=range(196, 200)),
                   leg(range(143), unscored=range(196, 200)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=4000, seed=0)
    a = r["acc_by_leg"]["psr/on"]
    assert a["ci_low"] < 143 / 196 < a["ci_high"]
    assert a["n_scored"] == 196 and a["n_censored"] == 4 and a["n_correct"] == 143
    assert math.isclose(a["point"], 143 / 196)
    assert isinstance(r["B"], int) and r["seed"] == 0 and r["level"] == 0.95 and r["n_items"] == N
    assert np.isfinite([a["ci_low"], a["ci_high"]]).all()


# -- exact arithmetic over the integer counts (review follow-up) -------------------


def test_equal_effects_give_an_exactly_zero_did_and_bound_never_a_floating_residue():
    """Two exactly equal effects (30/200 in both arms, different baselines):
    the DiD point is exactly ``0.0``, its bound exactly ``[0.0, 0.0]``, its sign
    NOT identified and ``point_sign == 0`` — the rational arithmetic, where a
    floating sum of four ratios once left ``[-1.1e-16, -1.1e-16]`` and read
    "negative, sign identified". With one ON leg censored so that the bound's
    LOWER end is exactly zero as a rational, that endpoint is exactly ``0.0``
    and the sign stays unidentified."""
    legs = legs_of(leg(range(120)), leg(range(150)), leg(range(110)), leg(range(140)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=200, seed=0)
    d = r["stats"]["did"]
    assert d["point"] == 0.0 and d["bound_low"] == 0.0 and d["bound_high"] == 0.0
    assert d["point_sign"] == 0 and d["sign_identified"] is False
    assert r["stats"]["effect_off"]["point"] == r["stats"]["effect_on"]["point"] == 0.15
    assert r["stats"]["effect_on"]["point_sign"] == 1 and r["stats"]["effect_off"]["point_sign"] == 1
    # the floating sum of the same four ratios in the bound's summation order (+psr/on, -none/on,
    # -psr/off, +none/off) is NOT zero — the -1.1e-16 the rule used to read as "negative, identified"
    assert (0 + 140 / 200 - 110 / 200 - 150 / 200 + 120 / 200) != 0.0
    # a censored leg whose adverse imputation lands exactly on zero: did_low = 140/200 - 110/200
    # - 150/200 + 120/200 = 0 exactly; did_high = 4/200
    censored = legs_of(leg(range(120)), leg(range(150)), leg(range(106), unscored=range(196, 200)),
                       leg(range(140)))
    r2 = pr.paired_bootstrap_conditional_2x2(censored, n_resamples=200, seed=0)
    d2 = r2["stats"]["did"]
    assert d2["bound_low"] == 0.0 and d2["bound_high"] == pytest.approx(0.02)
    assert d2["sign_identified"] is False and d2["point_sign"] == 1
    # shift_none's bound lower end is the rational 106/200 - 120/200 = -14/200, rounded ONCE
    assert r2["stats"]["shift_none"]["bound_low"] == -14 / 200


def test_point_sign_is_the_exact_sign_of_the_rational_point_zero_its_own_value():
    """``point_sign`` is -1 / 0 / +1 of the EXACT conditional-contrast rational
    (cross-multiplied, never a float difference): two legs whose conditional
    accuracies are equal as rationals but have different denominators
    (120/200 vs 117/195) read 0; a one-item edge reads its sign."""
    equal = legs_of(leg(range(120)), leg(range(117), unscored=range(195, 200)),
                    leg(range(100)), leg(range(100)))
    r = pr.paired_bootstrap_conditional_2x2(equal, n_resamples=100, seed=0)
    e = r["stats"]["effect_off"]
    assert e["point"] == 0.0 and e["point_sign"] == 0
    assert r["stats"]["effect_on"]["point_sign"] == 0 and r["stats"]["did"]["point_sign"] == 0
    edge = legs_of(leg(range(120)), leg(range(118), unscored=range(195, 200)),
                   leg(range(100)), leg(range(100)))
    r = pr.paired_bootstrap_conditional_2x2(edge, n_resamples=100, seed=0)
    assert r["stats"]["effect_off"]["point_sign"] == 1                  # 118/195 > 120/200
    assert r["stats"]["effect_off"]["point"] == pytest.approx(118 / 195 - 120 / 200)
    down = legs_of(leg(range(120)), leg(range(116), unscored=range(195, 200)), leg(range(100)), leg(range(100)))
    assert pr.paired_bootstrap_conditional_2x2(down, n_resamples=100, seed=0)["stats"]["effect_off"]["point_sign"] == -1
    # the complete-case delta is exact too
    # the complete-case delta is exact too: over the 195 items both legs scored, none keeps
    # all 120 of its corrects and psr its 117 -> exactly -3/195
    cc = pr.paired_bootstrap_conditional_2x2(equal, n_resamples=100, seed=0)["complete_case"]["effect_off"]
    assert cc["n_items"] == 195 and cc["delta"] == -3 / 195


# -- exact REPLICATE arithmetic (closing review, P2: the verdict's inclusive edge) --

#: the analysis fixture's OFF legs (tests/oos_synthetic_store.py): none 120 / psr 150
#: correct of 200 -> effect_off +15 pp, seed-0 95 % CI [+8, +22] pp at B = 10,000
FIXTURE_OFF_NONE = set(range(107)) | set(range(150, 163))
FIXTURE_OFF_PSR = set(range(150))


def _record_draws(monkeypatch):
    """Wrap numpy's generator so every ``integers`` call the bootstrap makes is
    recorded — ``(args, kwargs, the returned matrix)`` — the draw-sequence pin."""
    calls = []
    real = np.random.default_rng

    class Recording:
        def __init__(self, g):
            self._g = g

        def integers(self, *a, **k):
            out = self._g.integers(*a, **k)
            calls.append((a, k, out.copy()))
            return out

        def __getattr__(self, name):
            return getattr(self._g, name)

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: Recording(real(seed)))
    return calls


def test_replicate_contrasts_are_exact_grid_values_and_the_fixture_off_ci_is_0_08_0_22():
    """Closing review (P2): the replicate contrasts subtracted two separately
    rounded leg accuracies while the full-set points were exact rationals, and
    the verdict compared the two representations at an inclusive edge. On the
    analysis fixture (OFF none 120 / psr 150; ON none 100 / psr 144, no
    censoring; B = 10,000, seed 0) the OFF CI came back
    ``[0.07999999999999996, 0.21999999999999997]`` against an exact ON effect of
    ``0.22`` — "amplifies". With the IDENTICAL draws the integer counts over
    the common denominator give ``[0.08, 0.22]``, every replicate statistic a
    correctly rounded multiple of ``1/200``, and the ON effect sits ON the
    inclusive upper edge."""
    legs = legs_of(leg(FIXTURE_OFF_NONE), leg(FIXTURE_OFF_PSR), leg(range(100)), leg(range(144)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=10_000, seed=0)
    e = r["stats"]["effect_off"]
    assert (e["ci_low"], e["ci_high"]) == (16 / 200, 44 / 200) == (0.08, 0.22)
    assert r["stats"]["effect_on"]["point"] == 44 / 200 == 0.22 == float(Fraction(44, 200))
    assert e["ci_high"] == r["stats"]["effect_on"]["point"]                 # ONE representation
    # without censoring EVERY statistic is an exactly representable grid value: a multiple of
    # 1/200 as ONE correctly rounded float (the CI endpoints are order statistics of such values)
    for c in pr.CONTRASTS:
        for key in ("point", "ci_low", "ci_high", "bound_low", "bound_high", "bound_ci_low", "bound_ci_high"):
            x = r["stats"][c][key]
            assert x == round(x * N) / N == float(Fraction(round(x * N), N)), (c, key, x)
    for k, a in r["acc_by_leg"].items():
        for key in ("point", "ci_low", "ci_high"):
            assert a[key] == round(a[key] * N) / N, (k, key, a[key])
    # the representation the replicates once used — a difference of two separately rounded
    # accuracies — is NOT the grid value for most pairs 44 (or 16) items apart
    assert {a / N - (a - 44) / N for a in range(44, N + 1)} - {0.22}    # e.g. 0.75 - 0.53
    assert {a / N - (a - 16) / N for a in range(16, N + 1)} - {0.08}
    # the lower-edge counterpart: an ON effect of exactly 16/200 sits ON the lower edge
    r2 = pr.paired_bootstrap_conditional_2x2(
        legs_of(leg(FIXTURE_OFF_NONE), leg(FIXTURE_OFF_PSR), leg(range(100)), leg(range(116))),
        n_resamples=10_000, seed=0)
    assert (r2["stats"]["effect_off"]["ci_low"], r2["stats"]["effect_off"]["ci_high"]) == (0.08, 0.22)
    assert r2["stats"]["effect_on"]["point"] == 16 / 200 == 0.08 == r2["stats"]["effect_off"]["ci_low"]


def test_each_replicate_contrast_is_the_correctly_rounded_exact_rational_with_censoring_too():
    """``_exact_replicate_contrasts`` against ``fractions.Fraction`` on random
    integer counts (differing denominators per leg, zeros included): every
    point equals ``float(Σ s·Fraction(correct, scored))`` exactly and every
    bound end ``float(Σ ±Fraction(·, n))`` — one division of exact integers,
    the same representation the full-set rationals take. A zero denominator
    is not finite (the caller discards the replicate)."""
    rng = np.random.default_rng(1)
    B = 1500
    scored = {k: rng.integers(0, N + 1, size=B) for k in pr.LEG_KEYS}
    correct = {k: rng.integers(0, s + 1) for k, s in scored.items()}
    out = pr._exact_replicate_contrasts(correct, scored, N)
    checked = 0
    for c, signs in pr.CONTRAST_SIGNS.items():
        pt, lo, hi = out[c]
        assert pt.shape == lo.shape == hi.shape == (B,)
        for i in range(B):
            if any(scored[k][i] == 0 for k in signs):
                assert not np.isfinite(pt[i])
                continue
            cnt = {k: (int(correct[k][i]), int(scored[k][i])) for k in signs}
            want = float(sum(s * Fraction(cnt[k][0], cnt[k][1]) for k, s in signs.items()))
            want_lo = float(sum((Fraction(cnt[k][0], N) if s > 0 else -Fraction(cnt[k][0] + N - cnt[k][1], N))
                                for k, s in signs.items()))
            want_hi = float(sum((Fraction(cnt[k][0] + N - cnt[k][1], N) if s > 0 else -Fraction(cnt[k][0], N))
                                for k, s in signs.items()))
            assert (pt[i], lo[i], hi[i]) == (want, want_lo, want_hi), (c, i, cnt)
            checked += 1
    assert checked > 4 * B
    # the two separately-rounded-accuracies representation differs from it somewhere
    k1, k2 = ("psr", "off"), ("none", "off")
    with np.errstate(divide="ignore", invalid="ignore"):
        old = correct[k1] / scored[k1] - correct[k2] / scored[k2]
    finite = np.isfinite(old) & np.isfinite(out["effect_off"][0])
    assert (old[finite] != out["effect_off"][0][finite]).any()
    # a leg count too large for exact float64 integers refuses instead of rounding twice
    big = {i: (1, 1.0) for i in range(9742)}
    with pytest.raises(pr.PairingError, match="exactly-representable range"):
        pr.paired_bootstrap_conditional_2x2(legs_of(dict(big), dict(big), dict(big), dict(big)), n_resamples=1)


def test_the_draw_sequence_is_pinned_seed_0_index_matrix_unchanged(monkeypatch):
    """The exact-arithmetic change touched nothing before or at the draw: one
    ``rng.integers(0, n, size=(m, n))`` per chunk, in the same order — the
    seed-0 resampled index matrix is byte-identical to numpy's own
    ``Generator(PCG64(0)).integers(0, 200, size=(B, 200))`` and to its
    recorded sha256."""
    calls = _record_draws(monkeypatch)
    legs = legs_of(leg(FIXTURE_OFF_NONE), leg(FIXTURE_OFF_PSR), leg(range(100)), leg(range(144)))
    r = pr.paired_bootstrap_conditional_2x2(legs, n_resamples=50, seed=0)
    [(args, kwargs, matrix)] = calls                        # 50 replicates of 200 items: ONE chunk
    assert args == (0, N) and kwargs == {"size": (50, N)}
    assert matrix.dtype == np.int64 and matrix.shape == (50, N) and int(matrix.sum()) == 995131
    assert (matrix == np.random.Generator(np.random.PCG64(0)).integers(0, N, size=(50, N))).all()
    assert hashlib.sha256(matrix.tobytes()).hexdigest() == (
        "6ee1d82999122dbdba21cd5fd155f04a4c431e747ef996c043f3eb0d1c4fe81c")
    assert r["B"] == 50 and r["n_replicates_used"] == 50
    # the replicate statistics are a pure function of that matrix and the counts
    hit = {k: np.asarray([int(legs[k][i][1]) for i in IDS]) for k in pr.LEG_KEYS}
    d = np.sort((hit[("psr", "off")][matrix].sum(1) - hit[("none", "off")][matrix].sum(1)) / N)
    lo, hi = np.quantile(d, [0.025, 0.975])
    assert (r["stats"]["effect_off"]["ci_low"], r["stats"]["effect_off"]["ci_high"]) == (float(lo), float(hi))


# -- the thinking-OFF comparator at cap 8192: replicate coherence ----


def test_two_calls_with_the_same_seed_share_their_index_draws_so_relabelled_2x2s_are_replicate_coherent(monkeypatch):
    """The comparator's six contrasts are two further calls of THIS function
    with the legs relabelled into the 2x2's vocabulary: the
    index matrix is a pure function of ``(seed, n, B)`` — drawn once, before
    any leg content is read — so every call over the same 200 ids reuses the
    IDENTICAL replicate draws. Recorded here over three calls with different
    leg contents (the original 2x2; the cap-relief relabelling with the
    OFF/8192 legs in the ``on`` slots; the matched relabelling with them in
    the ``off`` slots): one matrix, three times; the shared contrasts agree
    bit for bit across calls; and per cell the decomposition identity
    ``shift = cap_relief + shift_matched`` holds to an ulp on the replicate
    arrays rebuilt from the recorded matrix (each term one correctly rounded
    division, so the float sum may sit one ulp off the third — never more).
    ``pairing.py`` itself is untouched by the comparator."""
    calls = _record_draws(monkeypatch)
    off_none, off_psr = leg(FIXTURE_OFF_NONE), leg(FIXTURE_OFF_PSR)
    off8k_none, off8k_psr = leg(set(range(107)) | set(range(150, 169))), leg(range(153))
    on_none, on_psr = leg(range(140)), leg(range(160))
    B = 400
    orig = pr.paired_bootstrap_conditional_2x2(legs_of(off_none, off_psr, on_none, on_psr), n_resamples=B, seed=0)
    cap = pr.paired_bootstrap_conditional_2x2(legs_of(off_none, off_psr, off8k_none, off8k_psr), n_resamples=B, seed=0)
    matched = pr.paired_bootstrap_conditional_2x2(legs_of(off8k_none, off8k_psr, on_none, on_psr), n_resamples=B, seed=0)
    assert len(calls) == 3
    for args, kwargs, _m in calls:
        assert args == (0, N) and kwargs == {"size": (B, N)}
    matrix = calls[0][2]
    assert (calls[1][2] == matrix).all() and (calls[2][2] == matrix).all()
    # the contrasts two calls share are ONE number: effect_off (the OFF/512 legs) in the
    # original and the cap-relief call; effect_on (the ON legs) in the original and the
    # matched call; effect_off_8192 — the cap call's "effect_on" and the matched call's "effect_off"
    def numbers(stat):          # every field but ``legs`` — the slot labels name the internal 2x2 slots
        return {k: v for k, v in stat.items() if k != "legs"}
    assert numbers(cap["stats"]["effect_off"]) == numbers(orig["stats"]["effect_off"])
    assert numbers(matched["stats"]["effect_on"]) == numbers(orig["stats"]["effect_on"])
    assert numbers(matched["stats"]["effect_off"]) == numbers(cap["stats"]["effect_on"])
    assert matched["stats"]["effect_off"]["legs"] == {"psr/off": 1, "none/off": -1}
    assert cap["stats"]["effect_on"]["legs"] == {"psr/on": 1, "none/on": -1}
    assert cap["acc_by_leg"]["none/off"] == orig["acc_by_leg"]["none/off"]
    assert matched["acc_by_leg"]["none/off"] == cap["acc_by_leg"]["none/on"]      # the OFF/8192 leg, both slots
    # the decomposition identity per cell, on the replicate arrays rebuilt from the recorded
    # matrix exactly as the bootstrap forms them (no censoring: ONE division of integer counts)
    hits = {name: np.asarray([int(L[i][1]) for i in IDS]) for name, L in
            (("off_none", off_none), ("off_psr", off_psr), ("off8k_none", off8k_none),
             ("off8k_psr", off8k_psr), ("on_none", on_none), ("on_psr", on_psr))}
    count = {name: h[matrix].sum(axis=1) for name, h in hits.items()}
    for kind in pr.LEG_CELLS:
        total = (count[f"on_{kind}"] - count[f"off_{kind}"]) / N
        budget = (count[f"off8k_{kind}"] - count[f"off_{kind}"]) / N
        render = (count[f"on_{kind}"] - count[f"off8k_{kind}"]) / N
        assert np.abs(total - (budget + render)).max() <= 1e-12
        # each call's percentile CI of shift_<kind> IS the quantile pair of the rebuilt array
        for rep, arr in ((orig, total), (cap, budget), (matched, render)):
            lo, hi = np.quantile(arr, [0.025, 0.975])
            assert (rep["stats"][f"shift_{kind}"]["ci_low"], rep["stats"][f"shift_{kind}"]["ci_high"]) == (float(lo), float(hi))
        # the full-set points add exactly too (exact rationals, converted once)
        assert orig["stats"][f"shift_{kind}"]["point"] == pytest.approx(
            cap["stats"][f"shift_{kind}"]["point"] + matched["stats"][f"shift_{kind}"]["point"], abs=1e-15)
    # the matched DiD is effect_on − effect_off_8192, exactly
    assert matched["stats"]["did"]["point"] == pytest.approx(
        orig["stats"]["effect_on"]["point"] - cap["stats"]["effect_on"]["point"], abs=1e-15)
