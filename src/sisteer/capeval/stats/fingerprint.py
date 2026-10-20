"""Fingerprint: domain statuses, screen status, target status, and the
(gated) verdict procedure.

v0 invocation policy: ``verdict()`` ships fully implemented and
property-tested but is NEVER invoked for v0 outputs — every v0 manifest
is ``confirmatory: false``, the only value the v0 loader accepts. What
v0 emits per arm instead: the fingerprint domain statuses (§7.2) plus a
screen status — SCREEN-FAIL iff a confirmed Tier-0 flag exists OR any
gating domain's point-estimate harm exceeds its margin; SCREEN-PASS
otherwise. Since 2026-08-18, "confirmed Tier-0 flag" means the CANARY —
coarsened-KL magnitudes are reported as an annotation on the fingerprint
(``kl_coarse_lb``) and can never fail a screen or feed a verdict.
``confirmatory_verdict()`` is the ONLY entry that reaches ``verdict()``
from run outputs, and it refuses unless the manifest says
``confirmatory: true`` — which the v0 loader refuses to load. The pure
function stays directly callable for its property tests.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sisteer.capeval.domains import Registry
from sisteer.capeval.manifest import Manifest, PrimarySpec
from sisteer.capeval.stats.calibrate import (
    Calibration,
    check_calibration,
    harness_version,
)
from sisteer.capeval.stats.noninferiority import (
    STATUS_DEGRADED,
    STATUS_INDETERMINATE,
    STATUS_PRESERVED,
    STATUSES,
    DomainStatus,
    StatsError,
    check_engine_matched,
    check_model_matched,
    effective_task_version,
    rows_frame,
    statuses_for_arm,
)

SCREEN_PASS = "SCREEN-PASS"
SCREEN_FAIL = "SCREEN-FAIL"
SCREENS = (SCREEN_PASS, SCREEN_FAIL)


class TargetStatus(str, enum.Enum):
    ESTABLISHED = "established"
    EXCLUDED = "excluded"
    INDETERMINATE = "indeterminate"


class Verdict(str, enum.Enum):
    """DESIGN §2.3 labels, first-match rules 1-6."""

    BROAD_DEGRADATION = "BROAD DEGRADATION"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT COVERAGE"
    SPECIFIC_WITH_COLLATERAL = "SPECIFIC-WITH-COLLATERAL"
    SPECIFIC = "SPECIFIC"
    COLLATERAL_WITHOUT_TARGET = "COLLATERAL-WITHOUT-TARGET"
    INERT = "INERT"


# --------------------------------------------------------------------------
# verdict — pure, total (§7.3)
# --------------------------------------------------------------------------


def verdict(
    target: str,
    statuses: Mapping[str, str],
    tier0_flag_confirmed: bool,
    harm_gt_3margin_confirmed: bool,
    n_resolved: int,
    min_resolved: int,
) -> Verdict:
    """DESIGN §2.3 rules 1-6 as a first-match chain, verbatim. Total by
    construction: after rule 2 the target is established or excluded and
    the degraded count is 0, 1-2, or >=3 — >=3 was caught by rule 1, so
    rules 3-6 cover everything left. ``statuses`` ranges over the GATING
    SET only; both ``*_confirmed`` booleans mean confirmed by the F16
    rerun (an unconfirmed flag NEVER reaches rule 1).
    ``tier0_flag_confirmed`` = a confirmed CANARY failure (the
    KL-over-κ clause is gone from §2.3 rule 1).
    """
    target_vals = tuple(t.value for t in TargetStatus)
    if isinstance(target, TargetStatus):
        target = target.value
    if target not in target_vals:
        raise StatsError(f"verdict: target {target!r} not in {target_vals}")
    for name, st in statuses.items():
        if st not in STATUSES:
            raise StatsError(f"verdict: statuses[{name!r}] = {st!r} not in "
                             f"{STATUSES}")
    for label, val in (("n_resolved", n_resolved),
                       ("min_resolved", min_resolved)):
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise StatsError(f"verdict: {label} must be a non-negative int, "
                             f"got {val!r}")

    degraded = sum(1 for st in statuses.values() if st == STATUS_DEGRADED)

    # 1. BROAD DEGRADATION
    if tier0_flag_confirmed or degraded >= 3 or harm_gt_3margin_confirmed:
        return Verdict.BROAD_DEGRADATION
    # 2. INSUFFICIENT COVERAGE
    if target == TargetStatus.INDETERMINATE.value or n_resolved < min_resolved:
        return Verdict.INSUFFICIENT_COVERAGE
    # 3-6: target ∈ {established, excluded}; degraded ∈ {0, 1, 2}.
    if target == TargetStatus.ESTABLISHED.value:
        if 1 <= degraded <= 2:
            return Verdict.SPECIFIC_WITH_COLLATERAL  # 3
        return Verdict.SPECIFIC  # 4 (degraded == 0)
    if 1 <= degraded <= 2:
        return Verdict.COLLATERAL_WITHOUT_TARGET  # 5
    return Verdict.INERT  # 6


def confirmatory_verdict(manifest: Manifest, *args: Any, **kwargs: Any) -> Verdict:
    """The ONLY run-output path to verdict(), and it refuses on every v0
    manifest: verdicts belong to the confirmatory tier set, and the v0
    loader accepts ``confirmatory: false`` only — so this refusal is
    structurally unreachable from a loaded v0 manifest, which is the
    point."""
    if manifest.confirmatory is not True:
        raise StatsError(
            "verdict() is never invoked for non-confirmatory runs (D4/F1): "
            "v0 emits fingerprint statuses + SCREEN-PASS/SCREEN-FAIL only. "
            "A confirmatory run is a v1 concern (Tiers 0-2).")
    return verdict(*args, **kwargs)


# --------------------------------------------------------------------------
# screen status (§7.3)
# --------------------------------------------------------------------------


def screen_status(statuses: Mapping[str, DomainStatus],
                  tier0_flag_confirmed: bool) -> str:
    """SCREEN-FAIL iff a confirmed Tier-0 flag exists OR any GATING
    domain's point-estimate harm exceeds its margin. Acceptance-role /
    excluded domains carry statuses but never touch the screen (r2-F6).
    ``tier0_flag_confirmed`` means a confirmed CANARY failure — the
    caller must never derive it from KL flags."""
    if tier0_flag_confirmed:
        return SCREEN_FAIL
    for st in statuses.values():
        if st.in_gating_set and st.harm > st.margin:
            return SCREEN_FAIL
    return SCREEN_PASS


# --------------------------------------------------------------------------
# target status (DESIGN §2.1)
# --------------------------------------------------------------------------


@dataclass
class PrimaryResult:
    metric: str
    direction: str
    min_effect: float
    delta: float  # oriented: positive = moved in the declared direction
    lower_95: float  # plain one-sided 95% percentile bound
    upper_95: float
    holm_lower: float  # bound at this primary's Holm step level
    p_value: float  # one-sided bootstrap p vs THIS primary's min_effect
    n_items: int


def _leg_map(leg: Any, metric: str, what: str) -> Dict[str, float]:
    """One target leg as an {item_id: value} map: legs arrive as
    TargetLegScores or an item-keyed mapping — NEVER a bare positional
    sequence, which would silently mispair rows consolidated (or
    resumed) in different orders."""
    from sisteer.capeval.targets import TargetLegScores

    if isinstance(leg, TargetLegScores):
        ids, values = list(leg.item_ids), list(leg.values)
        if len(ids) != len(set(ids)):
            dups = sorted({i for i in ids if ids.count(i) > 1})[:5]
            raise StatsError(f"target_status: primary {metric!r} {what} leg "
                             f"has duplicate item_ids (e.g. {dups})")
        return {i: float(v) for i, v in zip(ids, values)}
    if isinstance(leg, Mapping):
        if not leg:
            raise StatsError(f"target_status: primary {metric!r} {what} leg "
                             "is empty")
        return {str(i): float(v) for i, v in leg.items()}
    raise StatsError(
        f"target_status: primary {metric!r} {what} leg must be a "
        "TargetLegScores or an {item_id: value} mapping — positional "
        f"sequences pair by accident of row order (r1 #7); got "
        f"{type(leg).__name__}")


def target_status(
    primary_legs: Sequence[Tuple[PrimarySpec, Any, Any]],
    *,
    alpha: float = 0.05,
    b: int = 10_000,
    seed: int = 0,
) -> Tuple[TargetStatus, List[PrimaryResult]]:
    """DESIGN §2.1 semantics. ``primary_legs`` = per primary: (spec,
    base leg, arm leg), each leg a ``TargetLegScores`` or an
    ``{item_id: value}`` mapping — pairing is BY ITEM ID: the two legs
    must cover exactly the same item set, and a mismatch is a refusal
    naming the differences, exactly like capability-domain pairing.

    - **established** — every primary clears its min_effect under Holm
      step-down over one-sided bootstrap p-values against each primary's
      OWN min_effect (r1 #8): p-values ordered ascending, the t-th tested
      primary at level α/(p−t), stop on first failure. With one primary
      this is the plain 95% test.
    - **excluded** — at least one primary's plain one-sided 95% upper
      bound < its min_effect (the intended effect did not materialize).
    - **indeterminate** — otherwise.
    """
    if not primary_legs:
        raise StatsError("target_status: no primaries")
    rng = np.random.default_rng(seed)
    results: List[PrimaryResult] = []
    boots: List[np.ndarray] = []
    for spec, base_leg, arm_leg in primary_legs:
        base_map = _leg_map(base_leg, spec.metric, "base")
        arm_map = _leg_map(arm_leg, spec.metric, "arm")
        if set(base_map) != set(arm_map):
            only_b = sorted(set(base_map) - set(arm_map))[:5]
            only_a = sorted(set(arm_map) - set(base_map))[:5]
            raise StatsError(
                f"target_status: primary {spec.metric!r} legs cover "
                f"different item sets (base-only e.g. {only_b}, arm-only "
                f"e.g. {only_a}; {len(base_map)} vs {len(arm_map)} items) — "
                "same prompts by construction, so a mismatch is an "
                "incomplete leg, not a pairing choice")
        ids = sorted(base_map)
        base = np.array([base_map[i] for i in ids], dtype=np.float64)
        arm = np.array([arm_map[i] for i in ids], dtype=np.float64)
        deltas = arm - base if spec.direction == "up" else base - arm
        idx = rng.integers(0, deltas.size, size=(b, deltas.size))
        boot = deltas[idx].mean(axis=1)
        boots.append(boot)
        # one-sided bootstrap p against THIS primary's min_effect (r1 #8):
        # the (+1)/(B+1) form never returns an exact 0.
        p_val = (int(np.sum(boot <= spec.min_effect)) + 1) / (b + 1)
        results.append(PrimaryResult(
            metric=spec.metric, direction=spec.direction,
            min_effect=float(spec.min_effect), delta=float(deltas.mean()),
            lower_95=float(np.quantile(boot, alpha)),
            upper_95=float(np.quantile(boot, 1.0 - alpha)),
            holm_lower=float("nan"),  # filled below
            p_value=float(p_val),
            n_items=int(deltas.size),
        ))

    # Holm step-down (r1 #8): order by EVIDENCE AGAINST EACH PRIMARY'S
    # OWN min_effect (ascending p-value) — never by raw bound magnitude,
    # which scales with the metric and hands the strictest level to
    # whichever primary happens to have the largest units. The t-th
    # tested primary runs at α/(p−t); stop on the first failure (later
    # primaries are untested, establishment already failed).
    p = len(results)
    order = sorted(range(p), key=lambda i: results[i].p_value)
    established = True
    for step, i in enumerate(order):
        level = alpha / (p - step)
        results[i].holm_lower = float(np.quantile(boots[i], level))
        if established and results[i].p_value > level:
            established = False  # stop-on-first-failure: nothing later
            # can rescue establishment; bounds still filled for display
    if established:
        return TargetStatus.ESTABLISHED, results
    if any(r.upper_95 < r.min_effect for r in results):
        return TargetStatus.EXCLUDED, results
    return TargetStatus.INDETERMINATE, results


# --------------------------------------------------------------------------
# the per-arm fingerprint (v0 output shape)
# --------------------------------------------------------------------------


def build_fingerprint(
    arm_rows: Any,
    ref_rows: Any,
    registry: Registry,
    calibration: Calibration,
    *,
    model: str,
    arm_name: str,
    margins: Optional[Mapping[str, float]] = None,
    exclusions: Sequence[str] = (),
    tier0_flag_confirmed: bool = False,
    target: Optional[Tuple[TargetStatus, List[PrimaryResult]]] = None,
    alpha: float = 0.05,
    b: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """The per-arm fingerprint JSON: domain statuses (gating, excluded,
    and acceptance-role alike), z-display vs the calibration σ_d, screen
    status, target status when supplied, a gate log naming every domain
    that could NOT be computed (nothing filtered silently), and the
    ``kl_coarse_lb`` ANNOTATION (per-corpus magnitudes + mean, from the
    arm's ``kl_coarse_lb_corpus`` rows; None when the cell filed none) —
    reported for magnitude/triage, never a screen or verdict input.

    Refuses without a MATCHING calibration (model_revision, engine,
    harness_version) and on any cross-engine pairing.
    """
    arm_df, ref_df = rows_frame(arm_rows), rows_frame(ref_rows)
    engine = check_engine_matched(arm_df, ref_df)
    model_id, model_revision = check_model_matched(arm_df, ref_df)
    # config-identity gate (r1 stats-core #2): the calibration must cover
    # every task these legs measured, at the same task_version — the
    # domains version alone is not config identity.
    known = {t for d in registry.domains for t in d.tasks}
    run_task_versions: Dict[str, str] = {}
    for df in (arm_df, ref_df):
        for task in sorted(set(df["task"]) & known):
            # the EFFECTIVE version (PW14 / E22 H1): an EvalPlus task's column
            # read with its rows' metric_def metadata, as the calibration's was
            run_task_versions[task] = effective_task_version(df, task, "build_fingerprint")
    check_calibration(calibration, model_revision=model_revision,
                      engine=engine, harness_version=harness_version(registry),
                      model_id=model_id, task_versions=run_task_versions)

    statuses = statuses_for_arm(
        arm_df, ref_df, registry, model=model, margins=margins,
        exclusions=exclusions, alpha=alpha, b=b, seed=seed)

    gate_log: List[str] = []
    for dom in registry.domains:
        if dom.kind in ("accuracy", "nll") and dom.name not in statuses:
            gate_log.append(f"domain {dom.name}: no rows in this cell — "
                            "status not computed")
    for name in exclusions:
        gate_log.append(f"domain {name}: excluded from gating set "
                        "(manifest off_target_exclusions); status still "
                        "computed and reported when rows exist")

    domains_out: Dict[str, Any] = {}
    for name, st in statuses.items():
        d = st.as_dict()
        dc = calibration.domains.get(name)
        d["z_display"] = (st.harm / dc.sigma_d
                          if dc is not None and dc.sigma_d > 0 else None)
        d["delta"] = -st.harm  # score-direction delta for display
        domains_out[name] = d

    # Annotation: per-corpus coarsened-KL magnitudes, straight from
    # the arm's kl_coarse_lb_corpus rows. Magnitude/triage information
    # only — nothing below reads it into the screen.
    kl_rows = arm_df[arm_df["metric"] == "kl_coarse_lb_corpus"]
    kl_annotation = None
    if len(kl_rows):
        per_corpus = {task: float(g["value"].mean())
                      for task, g in kl_rows.groupby("task")}
        kl_annotation = {
            "per_corpus": per_corpus,
            "mean": float(np.mean(list(per_corpus.values()))),
            "unit": "nats_per_position",
            "note": "annotation/triage only — never a screen or verdict "
                    "input (D22)",
        }

    out: Dict[str, Any] = {
        # bump on ANY reader-visible shape change: a reader seeing an
        # unknown schema_version must refuse, never blank (simplify-bug
        # #6 contract; the bound field names stay upper_bound/lower_bound).
        # v2: screen no longer consumes KL flags; kl_coarse_lb
        # annotation added.
        "schema_version": 2,
        "arm": arm_name,
        "engine": engine,
        "model_id": model_id,
        "model_revision": model_revision,
        "harness_version": harness_version(registry),
        "screen": screen_status(statuses, tier0_flag_confirmed),
        "tier0_flag_confirmed": bool(tier0_flag_confirmed),
        "kl_coarse_lb": kl_annotation,
        "domains": domains_out,
        "gate_log": gate_log,
        "verdict": None,  # v0 NEVER fills this
    }
    if target is not None:
        t_status, primaries = target
        out["target"] = {
            "status": t_status.value,
            "primaries": [vars(p) for p in primaries],
            "ci_note": "primary CIs are unadjusted one-sided 95% "
                       "percentile-bootstrap bounds; establishment uses "
                       "Holm across primaries (DESIGN §2.1)",
        }
    return out
