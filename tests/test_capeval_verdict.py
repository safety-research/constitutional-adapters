"""Verdict-procedure tests — the
full-lattice exhaustiveness property, per-label goldens, rule-1
confirmation gating, the v0 confirmatory guard, and SCREEN goldens."""

import itertools

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.stats.fingerprint import (
    SCREEN_FAIL,
    SCREEN_PASS,
    TargetStatus,
    Verdict,
    confirmatory_verdict,
    screen_status,
    verdict,
)
from sisteer.capeval.stats.noninferiority import DomainStatus, StatsError

TARGETS = ("established", "excluded", "indeterminate")
STATUS_VALUES = ("preserved", "degraded", "indeterminate")


# --------------------------------------------------------------------------
# full-lattice enumeration: exactly one rule fires everywhere
# --------------------------------------------------------------------------


def _expected_rule(target, statuses, t0, h3m, n_res, min_res):
    """The six DESIGN §2.3 rules stated as INDEPENDENT predicates (not a
    first-match chain) — the test asserts exactly one is true for every
    lattice point and that verdict() returns that one."""
    deg = sum(1 for s in statuses.values() if s == "degraded")
    r1 = t0 or deg >= 3 or h3m
    r2 = (not r1) and (target == "indeterminate" or n_res < min_res)
    rest = not r1 and not r2
    r3 = rest and target == "established" and 1 <= deg <= 2
    r4 = rest and target == "established" and deg == 0
    r5 = rest and target == "excluded" and 1 <= deg <= 2
    r6 = rest and target == "excluded" and deg == 0
    fired = [v for v, hit in (
        (Verdict.BROAD_DEGRADATION, r1),
        (Verdict.INSUFFICIENT_COVERAGE, r2),
        (Verdict.SPECIFIC_WITH_COLLATERAL, r3),
        (Verdict.SPECIFIC, r4),
        (Verdict.COLLATERAL_WITHOUT_TARGET, r5),
        (Verdict.INERT, r6),
    ) if hit]
    return fired


def test_verdict_full_lattice_exhaustive_and_mutually_exclusive():
    n_res_cases = [(0, 0), (0, 1), (4, 5), (5, 5), (9, 7)]
    checked = 0
    for k in range(0, 7):  # 0..6 gating domains
        for combo in itertools.product(STATUS_VALUES, repeat=k):
            statuses = {f"d{i}": s for i, s in enumerate(combo)}
            for target, t0, h3m, (n_res, min_res) in itertools.product(
                    TARGETS, (False, True), (False, True), n_res_cases):
                fired = _expected_rule(target, statuses, t0, h3m,
                                       n_res, min_res)
                assert len(fired) == 1, (
                    f"lattice hole/overlap at {target} {combo} {t0} {h3m} "
                    f"{n_res}/{min_res}: {fired}")
                got = verdict(target, statuses, t0, h3m, n_res, min_res)
                assert got is fired[0]
                # rules 3-6 unreachable below the min_resolved floor
                if got not in (Verdict.BROAD_DEGRADATION,
                               Verdict.INSUFFICIENT_COVERAGE):
                    assert n_res >= min_res and target != "indeterminate"
                checked += 1
    assert checked == sum(3 ** k for k in range(7)) * 3 * 2 * 2 * 5


# --------------------------------------------------------------------------
# per-label goldens
# --------------------------------------------------------------------------


def test_verdict_goldens():
    pres3 = {"a": "preserved", "b": "preserved", "c": "preserved"}
    # 1. BROAD DEGRADATION — three ways in
    assert verdict("established", dict(pres3, a="degraded", b="degraded",
                                       c="degraded"),
                   False, False, 9, 7) is Verdict.BROAD_DEGRADATION
    assert verdict("established", pres3, True, False, 9, 7) \
        is Verdict.BROAD_DEGRADATION
    assert verdict("established", pres3, False, True, 9, 7) \
        is Verdict.BROAD_DEGRADATION
    # 2. INSUFFICIENT COVERAGE
    assert verdict("indeterminate", pres3, False, False, 9, 7) \
        is Verdict.INSUFFICIENT_COVERAGE
    assert verdict("established", pres3, False, False, 6, 7) \
        is Verdict.INSUFFICIENT_COVERAGE
    # 3. SPECIFIC-WITH-COLLATERAL
    assert verdict("established", dict(pres3, a="degraded"),
                   False, False, 9, 7) is Verdict.SPECIFIC_WITH_COLLATERAL
    # 4. SPECIFIC
    assert verdict("established", pres3, False, False, 9, 7) is Verdict.SPECIFIC
    # 5. COLLATERAL-WITHOUT-TARGET
    assert verdict("excluded", dict(pres3, a="degraded", b="degraded"),
                   False, False, 9, 7) is Verdict.COLLATERAL_WITHOUT_TARGET
    # 6. INERT
    assert verdict("excluded", pres3, False, False, 9, 7) is Verdict.INERT


def test_rule1_requires_confirmation():
    """An UNconfirmed Tier-0 flag / 3×m_d harm never reaches rule 1 —
    the booleans are `*_confirmed` by contract (F16); callers pass False
    until the fresh-subprocess rerun reproduces the flag."""
    statuses = {"a": "preserved", "b": "preserved"}
    assert verdict("established", statuses, False, False, 9, 7) \
        is Verdict.SPECIFIC
    assert verdict("established", statuses, True, False, 9, 7) \
        is Verdict.BROAD_DEGRADATION


def test_verdict_input_refusals():
    with pytest.raises(StatsError, match="target"):
        verdict("mystery", {}, False, False, 0, 0)
    with pytest.raises(StatsError, match="statuses"):
        verdict("established", {"a": "fine"}, False, False, 0, 0)
    with pytest.raises(StatsError, match="n_resolved"):
        verdict("established", {}, False, False, -1, 0)
    # enum inputs accepted
    assert verdict(TargetStatus.ESTABLISHED, {}, False, False, 0, 0) \
        is Verdict.SPECIFIC


# --------------------------------------------------------------------------
# v0 guard: verdict() is never invoked when confirmatory is false
# --------------------------------------------------------------------------


def _manifest(confirmatory: bool) -> cm.Manifest:
    return cm.Manifest(
        sweep="s", mode="calibration", base="m", base_revision="0" * 40,
        engine="hf", tiers=[0], confirmatory=confirmatory,
        ablations=[cm.AblationSpec(name="none", kind="none")])


def test_confirmatory_guard_refuses_v0_manifests():
    with pytest.raises(StatsError, match="D4/F1"):
        confirmatory_verdict(_manifest(False), "established", {},
                             False, False, 9, 7)


def test_confirmatory_guard_is_the_only_route_and_loader_bars_it():
    # a hand-built confirmatory manifest reaches verdict() through the
    # guard (v1 path)...
    got = confirmatory_verdict(_manifest(True), "established", {},
                               False, False, 9, 7)
    assert got is Verdict.SPECIFIC
    # ...but the v0 LOADER refuses to produce such a manifest at all, so
    # the route is structurally closed for v0 runs.
    with pytest.raises(cm.ManifestError, match="confirmatory"):
        cm.validate_manifest(_manifest(True))


# --------------------------------------------------------------------------
# SCREEN-PASS / SCREEN-FAIL goldens
# --------------------------------------------------------------------------


def _st(domain, harm, margin=0.02, *, gating=True, role=None):
    return DomainStatus(
        domain=domain, kind="accuracy", harm=harm, margin=margin,
        status="indeterminate", upper_bound=harm + 0.01,
        lower_bound=harm - 0.01, alpha=0.05, k_family=9, n_items=100,
        in_gating_set=gating, role=role, mcnemar_p=None)


def test_screen_goldens():
    ok = {"math": _st("math", 0.005), "knowledge": _st("knowledge", -0.01)}
    assert screen_status(ok, tier0_flag_confirmed=False) == SCREEN_PASS
    # a confirmed tier-0 flag fails the screen outright
    assert screen_status(ok, tier0_flag_confirmed=True) == SCREEN_FAIL
    # point-estimate harm beyond margin on a GATING domain fails
    bad = dict(ok, math=_st("math", 0.05))
    assert screen_status(bad, tier0_flag_confirmed=False) == SCREEN_FAIL
    # the same harm on an acceptance-role (non-gating) domain does NOT
    # touch the screen (r2-F6)
    acc = dict(ok, math_s2parity=_st("math_s2parity", 0.30, gating=False,
                                     role="acceptance"))
    assert screen_status(acc, tier0_flag_confirmed=False) == SCREEN_PASS
    # ... nor does an excluded (non-gating) domain
    exc = dict(ok, instruction_following=_st("instruction_following", 0.40,
                                             gating=False))
    assert screen_status(exc, tier0_flag_confirmed=False) == SCREEN_PASS
