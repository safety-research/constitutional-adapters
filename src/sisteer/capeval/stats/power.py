"""Simulation-based power per (model, domain). No closed-form
Φ anywhere: power is estimated by simulating the FULL preserved/degraded
classification pipeline, including the Bonferroni correction at the
configured family size, under two truths — harm = 0 and harm = 2·m_d.

The simulation is HIERARCHICAL, mirroring the conditional null's two
levels: each simulated cell first draws a whole calibration CONTRAST
(a run-level effect, coherent across every task in the domain), then
resamples items with replacement WITHIN each task from that contrast's
stored per-item null sample, and finally runs the actual registered
aggregate (task_weight-weighted, stratified paired bootstrap) and
classifier. Multi-task domains are therefore simulated task-by-task
exactly as status computation aggregates them — never as one pooled
item population — and run-level spread uncertainty propagates through
the contrast draw itself, with no spread/SE scale-factor approximation
anywhere.

Accuracy domains: the resampled contrasts carry the calibration-measured
null discordance with its run-level coherence. When the observed
discordance falls below the rule-of-three floor (zero or near-zero
observed events), symmetric per-item flips at the FLOORED rate (r3-1
transitions: expected up-flips == down-flips) are injected on top — an
empty observed-discordance sample can never yield power 1 (r2-F3).
harm = 2·m_d adds NET directional flips of arm-correct items.

NLL domains: per-window null deltas are resampled from the drawn
contrast; harm = 2·m_d adds a per-window shift of the harm size. A
degenerate all-zero delta sample requires ``discretization_scale`` and
simulates ±scale deviations at the rule-of-three rate — never σ = 0.

Margins: the SAME resolved per-run margin used by statuses_for_arm must
reach the simulation (r1 #13) — ``power_table``/``domain_power`` take
manifest overrides and never silently fall back to domains.yaml when an
override exists.

Threshold ≥ 0.8. Underpowered domains stay in the fingerprint (point
estimates + CIs still reported and screen-relevant); the annotation is
report-side, and gating-set removal feeds verdict() only on confirmatory
runs (v1 — none in v0, F1). Expect tiny (~100-item) domains to be
unpowered for preserved-certification at m = 2 pt (§16.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sisteer.capeval.domains import Domain, Registry
from sisteer.capeval.stats.calibrate import (
    Calibration,
    DomainCalibration,
    TaskCalibration,
)
from sisteer.capeval.stats.noninferiority import (
    StatsError,
    _task_weights,
    classify,
)

POWER_THRESHOLD = 0.8


class PowerError(StatsError):
    """A power-simulation refusal (degenerate inputs without the info to
    handle them honestly)."""


@dataclass
class PowerResult:
    domain: str
    kind: str
    power_preserved_h0: float
    power_degraded_h2m: float
    unpowered: bool  # power_preserved_h0 < POWER_THRESHOLD
    n_items: int
    margin: float  # the RESOLVED margin the classifier also used (r1 #13)
    family_k: int
    n_contrasts: int = 0
    discordance_rate: Optional[float] = None  # accuracy (post-floor)
    discordance_floored: Optional[bool] = None
    noise_spread: Optional[float] = None  # nll (nats/token, pooled SD)
    n_sims: int = 0
    b_boot: int = 0


def rule_of_three_floor(n_pairs: int) -> float:
    """Upper 95%-ish bound on a rate with zero observed events in
    n_pairs trials: 3/n."""
    if not isinstance(n_pairs, int) or n_pairs <= 0:
        raise PowerError(f"rule_of_three_floor: n_pairs must be a positive "
                         f"int, got {n_pairs!r}")
    return 3.0 / n_pairs


def floored_discordance_rate(observed_discordant: int, n_pairs: int
                             ) -> Tuple[float, bool]:
    """The r2-F3 rule: the observed per-item null discordance rate,
    floored by rule-of-three so an empty sample never yields rate 0."""
    if observed_discordant < 0 or n_pairs <= 0:
        raise PowerError("floored_discordance_rate: bad inputs")
    floor = rule_of_three_floor(n_pairs)
    rate = observed_discordant / n_pairs
    return (max(rate, floor), rate < floor)


def _classify_replicate(task_deltas: Sequence[np.ndarray],
                        weights: np.ndarray, margin: float, alpha: float,
                        family_k: int, b_boot: int,
                        rng: np.random.Generator) -> str:
    """One simulated cell through the ACTUAL §7.2 machinery: per-task
    oriented harm deltas -> task_weight-weighted registered aggregate,
    stratified (within-task) paired bootstrap, Bonferroni-adjusted
    quantile bounds, classify() (r1 #4/#6)."""
    point = float(sum(w * d.mean() for w, d in zip(weights, task_deltas)))
    boots = np.zeros(b_boot, dtype=np.float64)
    for w, d in zip(weights, task_deltas):
        idx = rng.integers(0, d.size, size=(b_boot, d.size))
        boots += w * d[idx].mean(axis=1)
    upper = float(np.quantile(boots, 1.0 - alpha / family_k))
    lower = float(np.quantile(boots, alpha / family_k))
    return classify(point, upper, lower, margin)


def _check_tasks(domain: Domain, tasks: Sequence[TaskCalibration],
                 need: str) -> int:
    """Shared validation; returns the contrast count."""
    if not tasks:
        raise PowerError(f"domain {domain.name!r}: no per-task calibration "
                         "samples — rebuild the calibration (r1 #4/#6)")
    n_cs = set()
    for t in tasks:
        sample = getattr(t, need)
        if sample is None or not len(sample):
            raise PowerError(f"domain {domain.name!r} task {t.task!r}: "
                             f"calibration lacks {need} — rebuild the "
                             "calibration (r1 #4/#6)")
        n_cs.add(len(sample))
        if t.n_items <= 1:
            raise PowerError(f"domain {domain.name!r} task {t.task!r}: "
                             "n_items must be > 1")
    if len(n_cs) != 1:
        raise PowerError(f"domain {domain.name!r}: tasks disagree on the "
                         f"contrast count {sorted(n_cs)}")
    return n_cs.pop()


def simulate_power_accuracy(
    domain: Domain,
    tasks: Sequence[TaskCalibration],
    *,
    margin: float,
    family_k: int,
    alpha: float = 0.05,
    n_sims: int = 400,
    b_boot: int = 800,
    seed: int = 0,
) -> Dict[str, Any]:
    """See module docstring. Hierarchical: draw a contrast (run effect,
    shared across tasks), resample items within task, classify with the
    registered aggregate. The rule-of-three floor injects symmetric r3-1
    flips (P(flip|correct)=rate/2p̂_t, P(flip|incorrect)=rate/2(1−p̂_t))
    when the observed null discordance sits below the floor — slightly
    conservative when a nonzero-but-tiny observed rate is also present,
    which only costs power, never validity."""
    n_c = _check_tasks(domain, tasks, "arm_values_by_contrast")
    for t in tasks:
        if t.ref_values is None or len(t.ref_values) != t.n_items:
            raise PowerError(f"domain {domain.name!r} task {t.task!r}: "
                             "calibration lacks the trial-0 ref_values leg "
                             "— rebuild the calibration (r1 #4/#6)")
        if not (0.0 < t.base_mean < 1.0):
            raise PowerError(
                f"domain {domain.name!r} task {t.task!r}: base_acc must be "
                f"in (0, 1) to define a transition model, got {t.base_mean} "
                "(a saturated probe cannot be power-modeled honestly — flag "
                "it instead)")
    refs = [np.asarray(t.ref_values, dtype=np.float64) >= 0.5 for t in tasks]
    arms = [[np.asarray(a, dtype=np.float64) >= 0.5
             for a in t.arm_values_by_contrast] for t in tasks]
    observed = sum(int(np.sum(a != r)) for r, per_c in zip(refs, arms)
                   for a in per_c)
    n_pairs = sum(t.n_items for t in tasks) * n_c
    rate, floored = floored_discordance_rate(observed, n_pairs)
    weights = _task_weights(domain, tasks)
    sign = 1.0 if domain.direction == "up" else -1.0

    rng = np.random.default_rng(seed)
    n_pres = n_degr = 0
    for _ in range(n_sims):
        c = int(rng.integers(0, n_c))  # ONE run effect, coherent across tasks
        d0_tasks: List[np.ndarray] = []
        d2_tasks: List[np.ndarray] = []
        for t, ref, per_c in zip(tasks, refs, arms):
            idx = rng.integers(0, t.n_items, size=t.n_items)
            ref_s, arm_s = ref[idx], per_c[c][idx]
            if floored:
                p1 = min(1.0, rate / (2.0 * t.base_mean))
                p0 = min(1.0, rate / (2.0 * (1.0 - t.base_mean)))
                flip = np.where(arm_s, rng.random(t.n_items) < p1,
                                rng.random(t.n_items) < p0)
                arm_s = arm_s ^ flip
            d0 = sign * (ref_s.astype(np.float64) - arm_s.astype(np.float64))
            d0_tasks.append(d0)
            # harm = 2·m_d: NET directional flips of arm-correct items
            # (direction "down": of arm-incorrect items, upward)
            if domain.direction == "up":
                q = min(1.0, 2.0 * margin / t.base_mean)
                extra = arm_s & (rng.random(t.n_items) < q)
                arm_h2 = arm_s & ~extra
            else:
                q = min(1.0, 2.0 * margin / (1.0 - t.base_mean))
                extra = ~arm_s & (rng.random(t.n_items) < q)
                arm_h2 = arm_s | extra
            d2_tasks.append(sign * (ref_s.astype(np.float64)
                                    - arm_h2.astype(np.float64)))
        if _classify_replicate(d0_tasks, weights, margin, alpha, family_k,
                               b_boot, rng) == "preserved":
            n_pres += 1
        if _classify_replicate(d2_tasks, weights, margin, alpha, family_k,
                               b_boot, rng) == "degraded":
            n_degr += 1

    return {
        "power_preserved_h0": n_pres / n_sims,
        "power_degraded_h2m": n_degr / n_sims,
        "discordance_rate": rate,
        "discordance_floored": floored,
        "n_contrasts": n_c,
    }


def simulate_power_nll(
    domain: Domain,
    tasks: Sequence[TaskCalibration],
    *,
    margin: float,  # relative (e.g. 0.03)
    family_k: int,
    alpha: float = 0.05,
    n_sims: int = 400,
    b_boot: int = 800,
    seed: int = 0,
    discretization_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """See module docstring. Noise is resampled per task from the DRAWN
    contrast's per-window null deltas — run-level spread propagates
    through the contrast draw, with no spread/SE scale factor (r1 #5). A
    degenerate all-zero sample requires ``discretization_scale`` and
    simulates ±scale deviations at the rule-of-three rate instead of
    σ = 0.

    Simplification (documented): windows are treated as equal-token, so
    each task's harm statistic is mean(delta)/base_nll — the token-
    weighted version differs only by window-length variation, second-
    order for the 1024-token fixed windows Tier 0 uses."""
    n_c = _check_tasks(domain, tasks, "deltas_by_contrast")
    for t in tasks:
        if t.base_mean <= 0:
            raise PowerError(f"domain {domain.name!r} task {t.task!r}: "
                             "base_nll must be positive")
    deltas = [[np.asarray(d, dtype=np.float64) for d in t.deltas_by_contrast]
              for t in tasks]
    pooled = np.concatenate([d for per_t in deltas for d in per_t])
    spread = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0
    degenerate = bool(np.all(pooled == 0.0)) or spread == 0.0
    if degenerate:
        if discretization_scale is None:
            raise PowerError(
                "simulate_power_nll: the observed null spread is exactly "
                "zero — pass discretization_scale so the rule-of-three "
                "treatment can apply (never simulate σ=0)")
        if discretization_scale <= 0:
            raise PowerError("simulate_power_nll: discretization_scale must "
                             "be positive")
        r = rule_of_three_floor(sum(t.n_items for t in tasks) * n_c)
    weights = _task_weights(domain, tasks)
    # harm orientation: relative_nll harm = (arm − ref)/ref, then flipped
    # when direction == "up" (domain_harm's rule)
    sign = 1.0 if domain.direction == "down" else -1.0

    rng = np.random.default_rng(seed)
    n_pres = n_degr = 0
    for _ in range(n_sims):
        c = int(rng.integers(0, n_c))
        d0_tasks: List[np.ndarray] = []
        d2_tasks: List[np.ndarray] = []
        for t, per_c in zip(tasks, deltas):
            if degenerate:
                u = rng.random(t.n_items)
                noise = np.where(u < r / 2, discretization_scale,
                                 np.where(u < r, -discretization_scale, 0.0))
            else:
                pool = per_c[c]
                noise = pool[rng.integers(0, pool.size, size=t.n_items)]
            d0 = sign * noise / t.base_mean  # relative-harm units
            d0_tasks.append(d0)
            d2_tasks.append(d0 + 2.0 * margin)  # per-window harm-size shift
        if _classify_replicate(d0_tasks, weights, margin, alpha, family_k,
                               b_boot, rng) == "preserved":
            n_pres += 1
        if _classify_replicate(d2_tasks, weights, margin, alpha, family_k,
                               b_boot, rng) == "degraded":
            n_degr += 1

    return {
        "power_preserved_h0": n_pres / n_sims,
        "power_degraded_h2m": n_degr / n_sims,
        "noise_spread": spread if not degenerate else discretization_scale,
        "degenerate_null": degenerate,
        "n_contrasts": n_c,
    }


def domain_power(
    domain: Domain,
    calib_dom: DomainCalibration,
    *,
    family_k: int,
    margin: Optional[float] = None,
    alpha: float = 0.05,
    n_sims: int = 400,
    b_boot: int = 800,
    seed: int = 0,
    discretization_scale: Optional[float] = None,
) -> PowerResult:
    """Power for one domain from its calibration record. ``margin`` is
    the RESOLVED per-run margin (manifest override applied) — it must be
    the same value statuses_for_arm classified with (r1 #13); None falls
    back to the registry default ONLY because no override exists."""
    if domain.kind not in ("accuracy", "nll"):
        raise PowerError(f"domain {domain.name!r} is kind {domain.kind!r} — "
                         "power applies to accuracy/nll domains only")
    m = float(domain.margin if margin is None else margin)
    if m <= 0:
        raise PowerError(f"domain {domain.name!r}: margin must be positive, "
                         f"got {m}")
    tasks = list(calib_dom.tasks.values())
    if domain.kind == "accuracy":
        sim = simulate_power_accuracy(
            domain, tasks, margin=m, family_k=family_k, alpha=alpha,
            n_sims=n_sims, b_boot=b_boot, seed=seed)
        extra = {"discordance_rate": sim["discordance_rate"],
                 "discordance_floored": sim["discordance_floored"]}
    elif domain.kind == "nll":
        sim = simulate_power_nll(
            domain, tasks, margin=m, family_k=family_k, alpha=alpha,
            n_sims=n_sims, b_boot=b_boot, seed=seed,
            discretization_scale=discretization_scale)
        extra = {"noise_spread": sim["noise_spread"]}
    return PowerResult(
        domain=domain.name, kind=domain.kind,
        power_preserved_h0=sim["power_preserved_h0"],
        power_degraded_h2m=sim["power_degraded_h2m"],
        unpowered=sim["power_preserved_h0"] < POWER_THRESHOLD,
        n_items=sum(t.n_items for t in tasks), margin=m,
        family_k=family_k, n_contrasts=sim["n_contrasts"],
        n_sims=n_sims, b_boot=b_boot, **extra)


def power_table(
    calibration: Calibration,
    registry: Registry,
    *,
    model: str,
    margins: Optional[Mapping[str, float]] = None,
    exclusions: Sequence[str] = (),
    alpha: float = 0.05,
    n_sims: int = 400,
    b_boot: int = 800,
    seed: int = 0,
    nll_discretization_scale: Optional[float] = None,
) -> Dict[Tuple[str, str], PowerResult]:
    """The (model, domain) → PowerResult table gating_set() and the
    report consume. ``margins`` are the run's manifest overrides — the
    SAME mapping handed to statuses_for_arm, so classification and power
    always price the same margin (r1 #13). Domains absent from the
    calibration are skipped WITH a stderr-visible note via the returned
    table's absence (the report lists them from calibration.notes) —
    never silently defaulted.

    ``nll_discretization_scale`` (nats/window, absolute) reaches every
    NLL domain's ``simulate_power_nll`` — required only when a domain's
    observed null spread is exactly zero (a fully deterministic replay,
    e.g. the k=12 Llama-3.1-8B HF calibration), where the rule-of-three
    treatment simulates ±scale deviations instead of σ = 0. Non-degenerate
    NLL domains ignore it; without it a degenerate domain still refuses
    loudly (never silently skipped)."""
    margins = dict(margins or {})
    unknown = sorted(set(margins) - {d.name for d in registry.domains})
    if unknown:
        raise PowerError(f"power_table: margin override(s) for unregistered "
                         f"domain(s) {unknown}")
    gating = registry.gating_set(model, {}, exclusions)
    family_k = len(gating)
    out: Dict[Tuple[str, str], PowerResult] = {}
    for i, dom in enumerate(registry.domains):
        if dom.kind not in ("accuracy", "nll"):
            continue
        calib_dom = calibration.domains.get(dom.name)
        if calib_dom is None:
            continue
        out[(model, dom.name)] = domain_power(
            dom, calib_dom, family_k=family_k,
            margin=margins.get(dom.name), alpha=alpha,
            n_sims=n_sims, b_boot=b_boot, seed=seed + i,
            discretization_scale=(nll_discretization_scale
                                  if dom.kind == "nll" else None))
    return out
