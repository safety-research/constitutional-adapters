"""Calibration: the trial-0-reference conditional null.

k base self-runs of the SAME model/engine/config: trial 0 is the
designated reference; the null is defined CONDITIONAL on it (r2-F2). The
estimand is the spread of (domain harm | trial 0) — exactly the quantity
every real comparison needs, because every arm pairs against this same
trial 0. Conditional on trial 0, the contrasts (trials 1..k−1, each
paired per-item against trial 0, mirroring the arm-vs-trial-0 comparison
one-for-one) are independent draws of the fresh-run leg, so the
two-level bootstrap below — outer draw = contrasts with replacement,
inner draw = items with replacement stratified by task, statistic = the
registered domain harm — is a standard bootstrap of conditionally-iid
replicates; no shared-leg covariance correction is needed, and none is
applied.

Outputs per domain: σ_d (spread of a single fresh-contrast harm), its
own bootstrap SE, empirical tail quantiles, the accuracy null
discordance rate (power input, with the rule-of-three floor applied
downstream by stats/power.py), and an NLL per-window null-delta
sample. Plus: canary fail-lines (§4.4) and κ — DEPRECATED as a flag
line since 2026-08-18: recorded as an ANNOTATION only, the raw p99
of the null contrasts' kl_coarse_lb with no floor (identically 0.0 on
deterministic replay stacks, which is exactly why the old 1e-4 floor
was degenerate — every active arm exceeded it). Coarsened KL feeds no
screen, no verdict, and no acceptance detection; its remaining roles
are the no-op exact-zero certificate (stats/controls.py), a reported
magnitude annotation (stats/fingerprint.py), and triage ordering.

Growth rule: if SE(σ_d) > 0.25·σ_d on any GATING domain, add
trials in batches of 3 (max k=12) and re-derive.

Calibration σ_d is used ONLY for z-score display and power (§7.4) —
never as any arm's CI (F6): every arm CI is that arm's own paired
bootstrap (noninferiority.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sisteer.capeval.domains import Registry
from sisteer.capeval.stats.noninferiority import (
    CANARY_AGG_ITEM,
    CANARY_DIRECTIONS,
    CANARY_TASK,
    METRIC_KL,
    StatsError,
    bootstrap_harms,
    domain_harm,
    effective_task_version,
    pair_domain,
    rows_frame,
)

ESTIMAND = "conditional_on_reference"
GROWTH_SE_FACTOR = 0.25  # SE(σ_d) > 0.25·σ_d ⇒ add trials
GROWTH_BATCH = 3
MAX_K = 12
NLL_SAMPLE_CAP = 1000

TAIL_QS = {"p01": 0.01, "p05": 0.05, "p50": 0.50, "p95": 0.95, "p99": 0.99}


def harness_version(registry: Registry) -> str:
    """The calibration/harness version key: bumping domains.yaml's
    ``version`` bumps this (§2.2)."""
    return f"domains-v{registry.version}"


@dataclass
class TaskCalibration:
    """Per-(domain, task) null persistence: the §7.4 power simulation
    needs WHOLE-CONTRAST run effects and per-task structure, so each
    contrast's per-item paired sample is stored — never flattened into
    one pooled item population.

    Item order is index-aligned across contrasts (pair_task sorts by
    item_id and refuses mismatched item sets). Accuracy tasks store the
    shared trial-0 leg once (``ref_values``) plus each contrast's arm leg
    (``arm_values_by_contrast``); NLL tasks store each contrast's
    per-window null deltas in nats/token (``deltas_by_contrast``)."""

    task: str
    n_items: int
    base_mean: float  # trial-0 acc / trial-0 nats-per-token for THIS task
    ref_values: Optional[List[float]] = None
    arm_values_by_contrast: Optional[List[List[float]]] = None
    deltas_by_contrast: Optional[List[List[float]]] = None


@dataclass
class DomainCalibration:
    domain: str
    kind: str
    sigma_d: float
    sigma_d_se: float
    tails: Dict[str, float]
    n_contrasts: int
    n_items: int
    needs_more_trials: bool
    # per-task, per-contrast null samples — the §7.4 hierarchical power
    # inputs:
    tasks: Dict[str, TaskCalibration] = field(default_factory=dict)
    # the observed per-contrast domain harms (display/diagnostics):
    contrast_harms: Optional[List[float]] = None
    # accuracy domains (rule-of-three floor inputs, r2-F3):
    null_discordant: Optional[int] = None
    null_pairs: Optional[int] = None
    null_discordance_rate: Optional[float] = None
    # nll domains: a seeded ≤1000-value pooled sample of per-window null
    # deltas (nats/token) — display/diagnostics; power uses ``tasks``:
    null_window_deltas_sample: Optional[List[float]] = None
    base_mean: Optional[float] = None  # base acc / base nats-per-token


@dataclass
class Calibration:
    model_revision: str
    engine: str
    harness_version: str
    estimand: str
    reference_trial: int
    n_contrasts: int
    b_boot: int
    seed: int
    created_at: float
    domains: Dict[str, DomainCalibration]
    canary_thresholds: Optional[Dict[str, Dict[str, float]]]
    # DEPRECATED annotation: raw p99 of the null kl_coarse_lb, no
    # floor. Never a flag/screen/verdict input; kept a field so older
    # stored calibration.json files (kappa = the old floored value)
    # keep loading.
    kappa: Optional[float]
    # config-identity fields: the domains version alone is NOT
    # sufficient config identity.
    model_id: str = ""
    task_versions: Dict[str, str] = field(default_factory=dict)
    eval_config_digest: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def check_calibration(calib: Calibration, *, model_revision: str,
                      engine: str, harness_version: str,
                      model_id: Optional[str] = None,
                      task_versions: Optional[Mapping[str, str]] = None,
                      eval_config_digest: Optional[str] = None) -> None:
    """The fingerprint's refusal gate (authoritative-source rule): no
    silent default σ — a run without a MATCHING calibration refuses.

    Beyond (model_revision, engine, harness_version), a caller that knows
    its run's ``model_id``, per-task ``task_versions``, and/or
    ``eval_config_digest`` passes them and they MUST match: every task
    the run measured must appear in the calibration with the same
    task_version — reusing a null derived under a different task config
    (or a task the null never covered) silently borrows a foreign
    instrument's σ, canary thresholds, and κ."""
    mismatches = [
        f"{name}: calibration has {got!r}, run has {want!r}"
        for name, got, want in (
            ("model_revision", calib.model_revision, model_revision),
            ("engine", calib.engine, engine),
            ("harness_version", calib.harness_version, harness_version),
        ) if got != want
    ]
    if model_id is not None and calib.model_id != model_id:
        mismatches.append(f"model_id: calibration has {calib.model_id!r}, "
                          f"run has {model_id!r}")
    if eval_config_digest is not None \
            and calib.eval_config_digest != eval_config_digest:
        mismatches.append(
            f"eval_config_digest: calibration has "
            f"{calib.eval_config_digest!r}, run has {eval_config_digest!r}")
    for task, ver in sorted((task_versions or {}).items()):
        got = calib.task_versions.get(task)
        if got is None:
            mismatches.append(f"task {task!r}: absent from the calibration "
                              "(the null never covered this instrument)")
        elif got != ver:
            mismatches.append(f"task {task!r}: calibration has task_version "
                              f"{got!r}, run has {ver!r}")
    if mismatches:
        raise StatsError("calibration mismatch — refusing to fingerprint "
                         "with a foreign null: " + "; ".join(mismatches))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build_calibration(
    rows: Any,
    registry: Registry,
    *,
    b_boot: int = 10_000,
    seed: int = 0,
    reference_trial: int = 0,
    template_hash: Optional[str] = None,
    gen_params: Optional[Mapping[str, Any]] = None,
) -> Calibration:
    """rows: every calibration cell's rows (arm 'none', trials 0..k−1,
    attempt 0). See module docstring for the estimand and mechanics.

    ``template_hash``/``gen_params`` feed the persisted
    ``eval_config_digest`` alongside the per-task task_version map read
    from the rows; a None component is recorded as a stated null inside
    the digest, never dropped."""
    df = rows_frame(rows)
    df = df[(df["arm_name"] == "none") & (df["attempt"] == 0)]
    if len(df) == 0:
        raise StatsError("build_calibration: no arm='none' attempt=0 rows")
    model_revision = _unique(df, "model_revision")
    model_id = _unique(df, "model_id")
    engine = _unique(df, "engine")
    task_versions = _registry_task_versions(df, registry)
    trials = sorted(set(df["trial_id"]))
    if reference_trial not in trials:
        raise StatsError(f"build_calibration: reference trial "
                         f"{reference_trial} absent (trials: {trials})")
    contrast_trials = [t for t in trials if t != reference_trial]
    if len(contrast_trials) < 2:
        raise StatsError("build_calibration: need >= 2 non-reference trials "
                         "(k >= 3) — a spread cannot come from one contrast")
    ref_df = df[df["trial_id"] == reference_trial]
    contrast_dfs = {t: df[df["trial_id"] == t] for t in contrast_trials}

    rng = np.random.default_rng(seed)
    domains: Dict[str, DomainCalibration] = {}
    notes: List[str] = []
    gating_names = {d.name for d in registry.domains
                    if d.gating and d.role != "acceptance"}

    for dom in registry.domains:
        if dom.kind not in ("accuracy", "nll"):
            continue
        # per-contrast paired task arrays vs trial 0
        per_contrast = {}
        for t, cdf in contrast_dfs.items():
            pairs = pair_domain(cdf, ref_df, dom)
            if pairs:
                per_contrast[t] = pairs
        if not per_contrast:
            notes.append(f"domain {dom.name}: no rows — not calibrated")
            continue
        if len(per_contrast) != len(contrast_trials):
            missing = sorted(set(contrast_trials) - set(per_contrast))
            raise StatsError(f"domain {dom.name!r}: rows present for some "
                             f"contrasts but missing for trials {missing} — "
                             "an incomplete calibration array is a refusal, "
                             "not a smaller k")

        contrasts = list(per_contrast.values())
        n_c = len(contrasts)
        # σ_d point estimate: the SD of the OBSERVED contrast harms —
        # the contrasts are conditionally-iid draws of the fresh-run leg
        # (r2-F2), so their sample SD is the plug-in estimand.
        observed = np.array([domain_harm(dom, pairs) for pairs in contrasts],
                            dtype=np.float64)
        sigma_d = float(observed.std(ddof=1))
        # Two-level bootstrap for SE(σ_d) + tails: outer = contrasts with
        # replacement, inner = items with replacement (stratified by task,
        # inside bootstrap_harms). Per replicate: n_c recomputed harms →
        # a σ_d* (their SD); the pooled recomputed harms give the
        # empirical null tail quantiles (item noise included — the p99
        # no-op band is deliberately the full single-comparison spread).
        n_pool = max(64, b_boot // n_c)
        pools = np.stack([bootstrap_harms(dom, pairs, b=n_pool, rng=rng)
                          for pairs in contrasts])  # (n_c, n_pool)
        outer = rng.integers(0, n_c, size=(b_boot, n_c))
        pick = rng.integers(0, n_pool, size=(b_boot, n_c))
        drawn = pools[outer, pick]  # (b_boot, n_c)
        sigma_d_se = float(drawn.std(axis=1, ddof=1).std(ddof=1))
        pooled = drawn.ravel()
        tails = {name: float(np.quantile(pooled, q))
                 for name, q in TAIL_QS.items()}
        # The growth rule is the stated inequality SE(σ_d) > 0.25·σ_d,
        # applied DIRECTLY — no extra σ_d > 0 guard. Five identical
        # observed contrast harms (σ_d = 0) with a positive
        # inner-bootstrap SE mean the zero spread is UNSTABLE, which is
        # exactly when more trials are needed; the truly degenerate 0/0
        # case already fails "0 > 0".
        needs_more = (dom.name in gating_names
                      and sigma_d_se > GROWTH_SE_FACTOR * sigma_d)

        dc = DomainCalibration(
            domain=dom.name, kind=dom.kind, sigma_d=sigma_d,
            sigma_d_se=sigma_d_se, tails=tails, n_contrasts=n_c,
            n_items=sum(p.n_items for p in contrasts[0]),
            needs_more_trials=needs_more,
            contrast_harms=[float(h) for h in observed],
        )
        # per-task, per-contrast persistence (r1 stats-core #4/#6): the
        # hierarchical power simulation draws whole contrasts, then
        # resamples items WITHIN task — so the samples are stored with
        # both levels of structure intact. pair_domain keeps task order
        # and pair_task keeps item order stable across contrasts.
        for t, p0 in enumerate(contrasts[0]):
            if dom.kind == "accuracy":
                dc.tasks[p0.task] = TaskCalibration(
                    task=p0.task, n_items=p0.n_items,
                    base_mean=float(p0.ref_values.mean()),
                    ref_values=[float(v) for v in p0.ref_values],
                    arm_values_by_contrast=[
                        [float(v) for v in pairs[t].arm_values]
                        for pairs in contrasts],
                )
            else:
                base_rate = float(p0.ref_values.sum() / p0.ref_tokens.sum())
                dc.tasks[p0.task] = TaskCalibration(
                    task=p0.task, n_items=p0.n_items, base_mean=base_rate,
                    deltas_by_contrast=[
                        [float(d) for d in (pairs[t].arm_values
                                            / pairs[t].arm_tokens
                                            - pairs[t].ref_values
                                            / pairs[t].ref_tokens)]
                        for pairs in contrasts],
                )
        if dom.kind == "accuracy":
            disc = pairs_total = 0
            vals = []
            for pairs in contrasts:
                for p in pairs:
                    disc += int(np.sum((p.arm_values >= 0.5)
                                       != (p.ref_values >= 0.5)))
                    pairs_total += p.n_items
                    vals.append(p.ref_values)
            dc.null_discordant = disc
            dc.null_pairs = pairs_total
            dc.null_discordance_rate = disc / pairs_total if pairs_total else None
            dc.base_mean = float(np.concatenate(vals).mean())
        else:
            deltas = []
            toks = []
            for pairs in contrasts:
                for p in pairs:
                    deltas.append(p.arm_values / p.arm_tokens
                                  - p.ref_values / p.ref_tokens)
                    toks.append(p.ref_values.sum() / p.ref_tokens.sum())
            all_d = np.concatenate(deltas)
            if len(all_d) > NLL_SAMPLE_CAP:
                idx = np.random.default_rng(seed + 1).choice(
                    len(all_d), size=NLL_SAMPLE_CAP, replace=False)
                all_d = all_d[idx]
            dc.null_window_deltas_sample = [float(x) for x in all_d]
            dc.base_mean = float(np.mean(toks))
        domains[dom.name] = dc

    canary = _canary_thresholds_from_rows(df, reference_trial, notes)
    kappa = _kappa_from_rows(df, reference_trial, b_boot, seed, notes)

    from sisteer.capeval.store import eval_config_digest as _ecd
    digest = (_ecd(task_versions, template_hash=template_hash,
                   gen_params=gen_params) if task_versions else None)
    if not task_versions:
        notes.append("eval_config_digest: no registry-task rows — digest "
                     "not derived")

    return Calibration(
        model_revision=model_revision, engine=engine,
        harness_version=harness_version(registry), estimand=ESTIMAND,
        reference_trial=reference_trial, n_contrasts=len(contrast_trials),
        b_boot=b_boot, seed=seed, created_at=time.time(),
        domains=domains, canary_thresholds=canary, kappa=kappa,
        model_id=model_id, task_versions=task_versions,
        eval_config_digest=digest, notes=notes,
    )


def _unique(df: pd.DataFrame, col: str) -> Any:
    vals = sorted(set(df[col]))
    if len(vals) != 1:
        raise StatsError(f"build_calibration: mixed {col!r} values {vals} — "
                         "a calibration is per (model_id, model_revision, "
                         "engine)")
    return vals[0]


def _registry_task_versions(df: pd.DataFrame,
                            registry: Registry) -> Dict[str, str]:
    """{task: EFFECTIVE task_version} for every REGISTRY task present in the
    rows (reserved _meta/target rows never enter config identity); a task
    with mixed versions inside one calibration is a refusal (r1 stats-core
    #2). PW14 (E22 H1): the version is ``noninferiority.effective_task_version``
    — for an EvalPlus task the column read together with its rows'
    ``metric_def`` metadata, so the map this feeds to ``eval_config_digest``
    (and ``check_calibration``) cannot give an old plus-only calibration and a
    PW12-era base-and-plus run one identity, and a mixed or contradictory
    group refuses here rather than pooling."""
    known = {t for d in registry.domains for t in d.tasks}
    out: Dict[str, str] = {}
    for task in sorted(set(df["task"]) & known):
        out[task] = effective_task_version(df, task, "build_calibration")
    return out


def growth_needed(calib: Calibration) -> List[str]:
    """Gating domains whose SE(σ_d) trips the growth rule — add
    GROWTH_BATCH trials (cap MAX_K) and re-derive (§7.1)."""
    return sorted(name for name, dc in calib.domains.items()
                  if dc.needs_more_trials)


# --------------------------------------------------------------------------
# canary thresholds (§4.4) — owned by calibration, never hardcoded
# --------------------------------------------------------------------------


def canary_thresholds_from_values(
    per_trial: Mapping[str, Sequence[float]],
    reference_index: int = 0,
    directions: Mapping[str, str] = CANARY_DIRECTIONS,
) -> Dict[str, Dict[str, float]]:
    """Fail-lines from per-trial base-run indicator values: per
    indicator, fail-line = worst base-run value plus the p99 null spread
    in the HARMFUL direction (harmful-oriented deltas vs the reference
    trial; with k−1=5 contrasts the empirical p99 ≈ the max, floored at
    0 so the line is never inside the observed base range)."""
    out: Dict[str, Dict[str, float]] = {}
    for name, values in per_trial.items():
        if name not in directions:
            raise StatsError(f"canary indicator {name!r} has no registered "
                             f"harmful direction; known: {sorted(directions)}")
        vals = np.asarray(list(values), dtype=np.float64)
        if len(vals) < 2:
            raise StatsError(f"canary indicator {name!r}: need >=2 trials, "
                             f"got {len(vals)}")
        direction = directions[name]
        ref = vals[reference_index]
        deltas = np.delete(vals, reference_index) - ref
        harmful = deltas if direction == "up" else -deltas
        spread = max(0.0, float(np.quantile(harmful, 0.99)))
        worst = float(vals.max() if direction == "up" else vals.min())
        fail_line = worst + spread if direction == "up" else worst - spread
        out[name] = {"fail_line": fail_line, "worst_base": worst,
                     "null_spread_p99": spread}
    return out


def _canary_thresholds_from_rows(df: pd.DataFrame, reference_trial: int,
                                 notes: List[str]) -> Optional[Dict]:
    sub = df[(df["task"] == CANARY_TASK) & (df["item_id"] == CANARY_AGG_ITEM)]
    if len(sub) == 0:
        notes.append("canary: no aggregate rows — thresholds not derived "
                     "(the canary gate refuses to run without them)")
        return None
    per_trial: Dict[str, List[float]] = {}
    trials = sorted(set(sub["trial_id"]))
    if reference_trial not in trials:
        raise StatsError("canary rows present but missing the reference trial")
    for name in sorted(set(sub["metric"])):
        vals = []
        for t in trials:
            v = sub[(sub["metric"] == name) & (sub["trial_id"] == t)]["value"]
            if len(v) != 1:
                raise StatsError(f"canary indicator {name!r}: expected one "
                                 f"aggregate row per trial, trial {t} has "
                                 f"{len(v)}")
            vals.append(float(v.iloc[0]))
        per_trial[name] = vals
    ref_index = trials.index(reference_trial)
    return canary_thresholds_from_values(per_trial, reference_index=ref_index)


# --------------------------------------------------------------------------
# κ (§7.1, deprecated): raw p99 of the null contrasts' kl_coarse_lb —
# an ANNOTATION recorded with the calibration, never a flag line. The old
# 1e-4 floor is gone: on deterministic replay the null is identically zero
# and the floored value classified every active arm, so the honest number
# is the degenerate one.
# --------------------------------------------------------------------------


def _kappa_from_rows(df: pd.DataFrame, reference_trial: int, b_boot: int,
                     seed: int, notes: List[str]) -> Optional[float]:
    sub = df[(df["metric"] == METRIC_KL) & (df["trial_id"] != reference_trial)]
    if len(sub) == 0:
        notes.append("kappa: no null kl_coarse_lb rows — the null-KL p99 "
                     "annotation not derived (D22: annotation only, no gate "
                     "depends on it)")
        return None
    if not np.isfinite(sub["value"].astype(float)).all():
        raise StatsError("κ derivation: non-finite kl_coarse_lb in the NULL "
                         "contrasts — base-vs-base KL cannot be +inf; the "
                         "harness indicts itself (§7.3)")
    rng = np.random.default_rng(seed + 2)
    means = []
    groups = list(sub.groupby(["task", "trial_id"]))
    n_boot = max(1, b_boot // max(1, len(groups)))
    for (_task, _trial), g in groups:
        vals = g["value"].to_numpy(dtype=np.float64)
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means.append(vals[idx].mean(axis=1))
    pooled = np.concatenate(means)
    return float(np.quantile(pooled, 0.99))


def derive_kappa(kl_rows: Any, *, reference_trial: int = 0,
                 b_boot: int = 10_000, seed: int = 0) -> float:
    """Standalone derivation of the null-KL p99 annotation from
    calibration KL rows (refuses when no rows exist — the in-build path
    records a note instead because a tier-less synthetic calibration is
    legitimate in tests). The value is unfloored and annotation-only —
    nothing flags, screens, or gates on it."""
    df = rows_frame(kl_rows)
    notes: List[str] = []
    kappa = _kappa_from_rows(df, reference_trial, b_boot, seed, notes)
    if kappa is None:
        raise StatsError("derive_kappa: no null kl_coarse_lb rows")
    return kappa


# --------------------------------------------------------------------------
# persistence (calibration.json)
# --------------------------------------------------------------------------


def save_calibration(calib: Calibration, path: Union[str, Path]) -> Path:
    """Atomic write (tmp+fsync+replace) of calibration.json."""
    from sisteer.capeval.store import _write_json_atomic

    path = Path(path)
    _write_json_atomic(path, asdict(calib))
    return path


def load_calibration(path: Union[str, Path]) -> Calibration:
    path = Path(path)
    if not path.is_file():
        raise StatsError(f"no calibration at {path} — run the calibration "
                         "array first; fingerprint refuses to run without a "
                         "matching calibration (no silent default σ)")
    raw = json.loads(path.read_text())
    try:
        raw["domains"] = {
            k: DomainCalibration(**{
                **v,
                "tasks": {t: TaskCalibration(**tv)
                          for t, tv in (v.get("tasks") or {}).items()},
            })
            for k, v in raw["domains"].items()}
        return Calibration(**raw)
    except TypeError as e:
        raise StatsError(f"{path}: not a calibration.json this version "
                         f"reads: {e}")
