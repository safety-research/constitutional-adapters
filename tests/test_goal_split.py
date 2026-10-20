"""Goal-level train/held-out split: determinism, disjointness, and the
thin-stratum visibility the family-level design did not need.

The hazard being pinned: a split that shares a goal between sides makes the
held-out evaluation partly in-sample, and nothing downstream reports it —
the numbers look fine. So disjointness is asserted directly, and the naive
alternative (splitting on prompts rather than goals) is shown to violate it.
"""

import hashlib

from sisteer.eval.goal_split import (GOAL_SPLIT_SALT, check_disjoint,
                                     goal_is_heldout, split_goals,
                                     split_report)

GOALS = [f"g{i:04d}" for i in range(400)]


def test_split_is_deterministic_and_order_independent():
    a = split_goals(GOALS)
    b = split_goals(list(reversed(GOALS)))
    assert a == b, "input order must not move the split"
    assert a == split_goals(GOALS), "same call must reproduce"


def test_sides_are_disjoint_and_exhaustive():
    tr, he = split_goals(GOALS)
    assert set(tr) & set(he) == set()
    assert set(tr) | set(he) == set(GOALS)
    check_disjoint(tr, he)          # must not raise


def test_check_disjoint_catches_a_shared_goal():
    tr, he = split_goals(GOALS)
    try:
        check_disjoint(tr, he + [tr[0]])
    except ValueError as e:
        assert "BOTH sides" in str(e)
    else:
        raise AssertionError("a shared goal must raise — this is the one "
                             "invariant the design rests on")


def test_frac_is_approximately_honoured():
    tr, he = split_goals(GOALS, frac=0.3)
    share = len(he) / len(GOALS)
    assert 0.24 < share < 0.36, f"held-out share {share:.3f} far from 0.3"


def test_salt_is_distinct_from_the_pair_holdout():
    # Sharing the pair split's salt would silently correlate two independent
    # questions: a goal could be held out BECAUSE its prompts were.
    assert GOAL_SPLIT_SALT != "two_si_holdout"
    pair_side = [hashlib.sha256(("two_si_holdout" + g).encode()).digest()[0] / 255.0 < 0.3
                 for g in GOALS]
    goal_side = [goal_is_heldout(g, frac=0.3) for g in GOALS]
    agree = sum(p == q for p, q in zip(pair_side, goal_side)) / len(GOALS)
    assert agree < 0.85, ("the two splits agree suspiciously often; they "
                          "should be independent")


def test_report_exposes_thin_families():
    # pasttense-shaped: very few goals, so one side can be too thin to
    # evaluate even though the union looks healthy.
    rep = split_report({"bon": GOALS[:184], "pasttense": GOALS[:8],
                        "wildjailbreak_harmful": GOALS})
    assert rep["families"]["pasttense"]["total"] == 8
    assert rep["union"]["total"] == len(GOALS)
    thin = rep["families"]["pasttense"]
    assert thin["train_goals"] + thin["heldout_goals"] == 8
    # the report must make the thinness legible rather than burying it in a total
    assert thin["heldout_goals"] < 6


def test_goal_split_record_must_carry_salt_and_frac():
    from sisteer.eval.goal_split import GOAL_SPLIT, split_state
    try:
        split_state(GOAL_SPLIT)
    except ValueError as e:
        assert "salt" in str(e)
    else:
        raise AssertionError("a goal_split without its salt/frac is "
                             "indistinguishable from family_held_out")


def test_parked_record_must_carry_date_and_reason():
    from sisteer.eval.goal_split import PARKED, split_state
    try:
        split_state(PARKED, reason="because")
    except ValueError as e:
        assert "since" in str(e) or "reason" in str(e)
    else:
        raise AssertionError("a silently parked family makes a later narrow "
                             "run look better than an earlier broad one")


def test_standing_regimes_cover_the_four_kept_and_park_the_rest():
    from sisteer.eval.goal_split import FAMILY_REGIMES, GOAL_SPLIT, PARKED
    for f in ("bon", "pasttense", "codeattack", "renellm"):
        assert FAMILY_REGIMES[f]["regime"] == GOAL_SPLIT
        assert FAMILY_REGIMES[f]["salt"] and FAMILY_REGIMES[f]["frac"]
    for f in ("artprompt", "cipherchat", "drattack", "multilingual", "aim"):
        assert FAMILY_REGIMES[f]["regime"] == PARKED
        assert FAMILY_REGIMES[f]["since"] == "2026-08-24"
