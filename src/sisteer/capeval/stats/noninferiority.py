"""Non-inferiority machinery + the shared pairing/aggregation
primitives every stats module uses.

Statuses per domain (harm orientation from domains.yaml ``direction``):

- **preserved** — one-sided upper CI bound on harm < m_d;
- **degraded** — one-sided lower CI bound on harm > 0 AND point estimate
  > m_d;
- **indeterminate** — otherwise.

Multiplicity: Bonferroni within each predeclared one-sided family —
the preservation family (tests vs margin) and the degradation family
(tests vs 0), each of size k = the gating-domain count. The
simultaneous bounds are obtained by bootstrap-quantile inversion: the
1−α/k (resp. α/k) empirical quantile of the domain's own paired-
bootstrap distribution. They are labeled *percentile-bootstrap
approximate simultaneous bounds* everywhere — approximate, not
guaranteed conservative. Holm step-down is the v1 refinement.

Acceptance-role domains (r2-F6/r3-3) get plain one-sided 95% percentile-
bootstrap CIs — their own family, UNADJUSTED — they gate nothing.

McNemar exact p on the discordant pairs is computed but DISPLAY-ONLY:
it tests a zero null, not the margin, so it never decides a status.

Metric-name contract (rows → aggregation; shared with the row writers
via the integrator): accuracy tasks emit one ``METRIC_ACC`` row per
item (value ∈ {0,1} or a [0,1] score); NLL tasks emit
``METRIC_NLL_SUM`` and ``METRIC_TOKEN_COUNT`` rows per window
(aggregates recompute Σnum/Σden from rows, never from cached scalars);
coarsened-KL tasks emit ``METRIC_KL`` per window; the canary emits
per-indicator aggregate rows under item_id ``CANARY_AGG_ITEM``.
Summary rows under item_id ``AGGREGATE_ITEM_ID`` are never paired
items — ``_metric_map`` drops them (2026-08-17 incident); only the
canary pairs on aggregates, and it does so through its own selector,
never ``_metric_map``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sisteer.capeval.domains import Domain, Registry
from sisteer.capeval.store import ROW_COLUMNS

# --- Metric-name contract (see module docstring) ----------------------------

METRIC_ACC = "acc"
METRIC_NLL_SUM = "nll_sum"
METRIC_TOKEN_COUNT = "token_count"
METRIC_KL = "kl_coarse_lb"
CANARY_TASK = "tier0/canary"
# The summary pseudo-item id shared by every tier's aggregate rows. It is
# NEVER a paired item — _metric_map drops it (2026-08-17 incident, below);
# only the canary DELIBERATELY pairs on it (aggregate-only indicators, via
# stats.calibrate._canary_thresholds_from_rows, which never touches
# _metric_map).
AGGREGATE_ITEM_ID = "__aggregate__"
CANARY_AGG_ITEM = AGGREGATE_ITEM_ID
# Canary indicators (§4.4) and each one's HARMFUL direction.
CANARY_DIRECTIONS: Dict[str, str] = {
    "max_rep_run_p95": "up",
    "distinct3": "down",
    "mean_tok_logprob": "down",
    "nonascii_frac": "up",
    "abrupt_eos_rate": "up",
}

STATUS_PRESERVED = "preserved"
STATUS_DEGRADED = "degraded"
STATUS_INDETERMINATE = "indeterminate"
STATUSES = (STATUS_PRESERVED, STATUS_DEGRADED, STATUS_INDETERMINATE)

BOUND_METHOD = ("percentile-bootstrap approximate simultaneous bounds "
                "(Bonferroni per one-sided family)")


class StatsError(ValueError):
    """A stats-layer refusal (pairing mismatch, cross-engine rows, …)."""


# --------------------------------------------------------------------------
# Row frames + pairing
# --------------------------------------------------------------------------


def rows_frame(rows: Any) -> pd.DataFrame:
    """list[ResultRow] | list[dict] | DataFrame -> DataFrame with (at
    least) the frozen ROW_COLUMNS. Extra columns are a refusal."""
    if isinstance(rows, pd.DataFrame):
        df = rows
    else:
        import dataclasses

        recs = [dataclasses.asdict(r) if dataclasses.is_dataclass(r) else dict(r)
                for r in rows]
        df = pd.DataFrame.from_records(recs)
    missing = sorted(set(ROW_COLUMNS) - set(df.columns))
    extra = sorted(set(df.columns) - set(ROW_COLUMNS))
    if missing or extra:
        raise StatsError(f"rows frame schema drift — missing {missing}, "
                         f"extra {extra} (frozen ROW_COLUMNS)")
    return df


def _single(df: pd.DataFrame, col: str, what: str) -> Any:
    vals = sorted(set(df[col]))
    if len(vals) != 1:
        raise StatsError(f"{what}: expected a single {col!r}, got {vals}")
    return vals[0]


def check_engine_matched(arm_df: pd.DataFrame, ref_df: pd.DataFrame) -> str:
    """Engine-matched rule (DESIGN §3): refuse to pair rows whose engine
    fields differ, naming both sides."""
    e_arm = _single(arm_df, "engine", "arm rows")
    e_ref = _single(ref_df, "engine", "reference rows")
    if e_arm != e_ref:
        arm = sorted(set(arm_df["arm_name"]))
        raise StatsError(
            f"engine-matched refusal: arm rows ({arm}) ran on engine "
            f"{e_arm!r} but the reference rows ran on {e_ref!r} — deltas "
            "are only ever computed within one engine")
    return e_arm


def check_model_matched(arm_df: pd.DataFrame,
                        ref_df: pd.DataFrame) -> Tuple[str, str]:
    """Model-identity rule: two legs may pair only when they measured
    the SAME model — single and equal model_id and model_revision on
    both sides. Item IDs and the domains version alone are NOT config
    identity. Returns (model_id, model_revision)."""
    out = []
    for col in ("model_id", "model_revision"):
        v_arm = _single(arm_df, col, "arm rows")
        v_ref = _single(ref_df, col, "reference rows")
        if v_arm != v_ref:
            raise StatsError(
                f"model-matched refusal: arm rows carry {col}={v_arm!r} but "
                f"the reference rows carry {v_ref!r} — a paired delta "
                "against a different model is not a comparison")
        out.append(v_arm)
    return out[0], out[1]


def _row_metric_def(extra_json: Any) -> Any:
    """The ``metric_def`` an EvalPlus row's ``extra_json`` names (a JSON
    string on disk, a mapping in memory), ``None`` when absent."""
    if extra_json is None or (isinstance(extra_json, float) and math.isnan(extra_json)):
        return None
    if isinstance(extra_json, str):
        try:
            extra_json = json.loads(extra_json)
        except ValueError as e:
            raise StatsError(f"row extra_json is not JSON ({e})") from e
    if not isinstance(extra_json, Mapping):
        return None
    return extra_json.get("metric_def")


def effective_task_version(df: pd.DataFrame, task: str, what: str) -> str:
    """The ONE version ``task``'s rows in ``df`` are under — the instrument
    identity every pooling reader keys on (``pair_task``,
    ``calibrate._registry_task_versions`` and hence ``eval_config_digest``,
    ``fingerprint.build_fingerprint``). PW14 (E22 H1): for an EvalPlus task
    the ``task_version`` column alone is not that identity — the code between
    e7c0da4 and 765251a computed ``pass@1_plus`` as base AND plus and still
    emitted the bare ``evalplus-0.3.1``, the literal the retained plus-only
    rows carry — so each raw version is read TOGETHER with the ``metric_def``
    metadata of the task's ``pass@1_plus`` / ``acc`` rows
    (``contracts.evalplus_effective_task_version``): bare + explicit
    ``base_and_plus`` is the suffixed version, bare + absent is plus-only, and
    a MIXED or CONTRADICTORY group refuses, named. Two raw versions that
    resolve to one effective version are one instrument (PW12-era bare rows
    beside suffixed ones); two effective versions within one leg refuse as
    before (``mixed task_version``). Non-EvalPlus versions pass through."""
    from sisteer.capeval.tinycap import contracts as tc  # stdlib-only; lazy (no import cycle)

    rows = df[df["task"] == task]
    raw = sorted(set(rows["task_version"]))
    effective: Dict[str, str] = {}
    for version in raw:
        if not tc.is_evalplus_task_version(version):
            effective[version] = version
            continue
        group = rows[(rows["task_version"] == version)
                     & rows["metric"].isin(tc.EVALPLUS_DEFINED_METRICS)]
        defs = {_row_metric_def(x) for x in group["extra_json"]}
        try:
            effective[version] = tc.evalplus_effective_task_version(version, defs)
        except ValueError as e:
            raise StatsError(f"{what}: task {task!r}: {e}") from e
    distinct = sorted(set(effective.values()))
    if len(distinct) != 1:
        raise StatsError(f"{what}: task {task!r} carries mixed task_version "
                         f"values {raw} within one leg (effective versions {distinct})")
    return distinct[0]


@dataclass
class TaskPairs:
    """Per-task paired per-item arrays (arm leg vs reference leg)."""

    task: str
    item_ids: Tuple[str, ...]
    # accuracy: score arrays; NLL: (nll_sum, token_count) arrays per leg.
    arm_values: np.ndarray
    ref_values: np.ndarray
    arm_tokens: Optional[np.ndarray] = None
    ref_tokens: Optional[np.ndarray] = None

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def _metric_map(df: pd.DataFrame, task: str, metric: str,
                what: str) -> Dict[str, float]:
    sub = df[(df["task"] == task) & (df["metric"] == metric)]
    # 2026-08-17 incident: the s2parity probes filed a per-(task, trial)
    # summary row under item_id AGGREGATE_ITEM_ID with the SAME metric
    # name as the per-item rows ("acc"), and it rode into pairing as a
    # 201st "item" (PARITY_N = 200) for both parity domains. The POINT
    # delta is provably unchanged (the pseudo-item equals its own leg's
    # mean, and adding a leg's mean to itself leaves the mean fixed), but
    # the bootstrap resampled it like an item, so CIs were computed at
    # n+1 with a zero-deviation item at the mean. Excluding here — the
    # pairing side — also repairs ALREADY-STORED campaign rows, which an
    # emission-side rename alone cannot. The canary path is untouched: it
    # pairs on aggregate rows deliberately and never goes through
    # _metric_map.
    sub = sub[sub["item_id"] != AGGREGATE_ITEM_ID]
    if len(sub) == 0:
        raise StatsError(f"{what}: no per-item {metric!r} rows for task "
                         f"{task!r}")
    ids = list(sub["item_id"])
    if len(ids) != len(set(ids)):
        dups = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise StatsError(f"{what}: duplicate item_id rows for task {task!r} "
                         f"metric {metric!r} (e.g. {dups}) — one row per "
                         "(cell, item, metric)")
    vals = sub["value"].astype(float).to_numpy()
    _check_metric_values(vals, ids, task, metric, what)
    return dict(zip(ids, vals))


def _check_metric_values(vals: np.ndarray, ids: Sequence[str], task: str,
                         metric: str, what: str) -> None:
    """Refuse corrupt measurements BEFORE they enter pairing: a NaN
    accuracy or a zero token_count would flow into bootstrap quantiles
    as NaN, fall through classify() to indeterminate, and read
    NaN > margin as False — a failed domain presenting as
    SCREEN-PASS."""
    def _name(mask) -> str:
        bad = [ids[i] for i in np.flatnonzero(mask)[:5]]
        return f"(e.g. items {bad})"

    finite = np.isfinite(vals)
    if not finite.all():
        raise StatsError(
            f"{what}: non-finite {metric!r} value(s) for task {task!r} "
            f"{_name(~finite)} — a NaN/inf measurement is a scorer failure "
            "to refuse, never data to aggregate")
    if metric == METRIC_ACC:
        bad = (vals < 0.0) | (vals > 1.0)
        if bad.any():
            raise StatsError(f"{what}: {metric!r} value(s) outside [0, 1] "
                             f"for task {task!r} {_name(bad)}")
    if metric == METRIC_TOKEN_COUNT:
        bad = (vals <= 0) | (vals != np.floor(vals))
        if bad.any():
            raise StatsError(
                f"{what}: non-positive or non-integral {metric!r} value(s) "
                f"for task {task!r} {_name(bad)} — an empty/fractional "
                "window is a windowing failure, not a denominator")


def pair_task(arm_df: pd.DataFrame, ref_df: pd.DataFrame, task: str,
              kind: str) -> TaskPairs:
    """Pair one task's per-item rows by item_id. Item sets must match
    EXACTLY — a partial overlap means one leg is incomplete (resume gap)
    and silently intersecting would bias the delta. The task_version must
    also match between legs: same item IDs under a changed task
    config/version is a different instrument, and a delta across
    instruments is not a comparison."""
    tv_arm = effective_task_version(arm_df, task, "arm")
    tv_ref = effective_task_version(ref_df, task, "reference")
    if tv_arm != tv_ref:
        raise StatsError(
            f"pairing refusal for task {task!r}: arm rows carry "
            f"task_version {tv_arm!r} but the reference rows carry "
            f"{tv_ref!r} — the task config changed between legs; rerun the "
            "reference on the same instrument")
    if kind == "accuracy":
        a = _metric_map(arm_df, task, METRIC_ACC, "arm")
        r = _metric_map(ref_df, task, METRIC_ACC, "reference")
    elif kind == "nll":
        a = _metric_map(arm_df, task, METRIC_NLL_SUM, "arm")
        r = _metric_map(ref_df, task, METRIC_NLL_SUM, "reference")
    else:
        raise StatsError(f"pair_task: kind {kind!r} is not a paired kind")
    if set(a) != set(r):
        only_a = sorted(set(a) - set(r))[:5]
        only_r = sorted(set(r) - set(a))[:5]
        raise StatsError(
            f"pairing refusal for task {task!r}: item sets differ "
            f"(arm-only e.g. {only_a}, reference-only e.g. {only_r}; "
            f"{len(a)} vs {len(r)} items) — same items by construction, so "
            "a mismatch is an incomplete cell, not a pairing choice")
    ids = tuple(sorted(a))
    if kind == "accuracy":
        return TaskPairs(
            task=task, item_ids=ids,
            arm_values=np.array([a[i] for i in ids], dtype=np.float64),
            ref_values=np.array([r[i] for i in ids], dtype=np.float64),
        )
    at = _metric_map(arm_df, task, METRIC_TOKEN_COUNT, "arm")
    rt = _metric_map(ref_df, task, METRIC_TOKEN_COUNT, "reference")
    if set(at) != set(a) or set(rt) != set(r):
        raise StatsError(f"task {task!r}: token_count rows do not cover the "
                         "same items as nll_sum rows (F14 rows incomplete)")
    return TaskPairs(
        task=task, item_ids=ids,
        arm_values=np.array([a[i] for i in ids], dtype=np.float64),
        ref_values=np.array([r[i] for i in ids], dtype=np.float64),
        arm_tokens=np.array([at[i] for i in ids], dtype=np.float64),
        ref_tokens=np.array([rt[i] for i in ids], dtype=np.float64),
    )


def pair_domain(arm_df: pd.DataFrame, ref_df: pd.DataFrame,
                domain: Domain, *,
                allow_arm_task_subset: bool = False) -> List[TaskPairs]:
    """All of a domain's tasks paired — but only tasks PRESENT in both
    legs' rows; a task present in one leg only is a refusal. Domains with
    no tasks present at all return [] (the caller records the gap; a
    tier0-only cell legitimately has no tier1 rows).

    ``allow_arm_task_subset`` (controls-acceptance seam, integrator
    2026-08-15): the reference leg is calibration trial 0 and carries
    EVERY task, while a tier-0-only control rung legitimately carries
    none of a Tier-1 domain's tasks. With the flag, a domain whose tasks
    are ENTIRELY absent from the arm leg returns [] (not measured in
    this cell) instead of refusing. It is a domain-level allowance only:
    an arm leg carrying a PARTIAL subset of the domain's tasks still
    refuses — a silently missing task would bias the domain aggregate —
    and arm-only tasks (reference incomplete) always refuse. Default
    stays the strict refusal for real arms."""
    arm_present = {task: bool((arm_df["task"] == task).any())
                   for task in domain.tasks}
    domain_absent_in_arm = not any(arm_present.values())
    pairs = []
    for task in domain.tasks:
        in_arm = arm_present[task]
        in_ref = bool(((ref_df["task"] == task)).any())
        if in_arm != in_ref:
            if (allow_arm_task_subset and in_ref and domain_absent_in_arm):
                continue  # whole domain unmeasured in this cell -> []
            raise StatsError(f"domain {domain.name!r}: task {task!r} present "
                             f"in {'arm' if in_arm else 'reference'} rows "
                             "only — one leg is incomplete")
        if in_arm:
            pairs.append(pair_task(arm_df, ref_df, task, domain.kind))
    return pairs


# --------------------------------------------------------------------------
# The registered aggregate (F14) + paired bootstrap
# --------------------------------------------------------------------------


def _task_weights(domain: Domain, pairs: Sequence[TaskPairs]) -> np.ndarray:
    tw = domain.task_weight
    if isinstance(tw, Mapping):
        try:
            w = np.array([float(tw[p.task]) for p in pairs], dtype=np.float64)
        except KeyError as e:
            raise StatsError(f"domain {domain.name!r}: task_weight mapping "
                             f"missing task {e}")
    else:
        w = np.full(len(pairs), float(tw), dtype=np.float64)
    if not np.all(w > 0):
        raise StatsError(f"domain {domain.name!r}: non-positive task weight")
    return w / w.sum()


def domain_harm(domain: Domain, pairs: Sequence[TaskPairs],
                idx_per_task: Optional[Sequence[np.ndarray]] = None) -> float:
    """The registered harm aggregate, recomputed from (optionally
    resampled) per-item rows — never from cached scalars (F14).

    accuracy/identity: harm_t = mean(ref − arm) oriented by direction;
    nll/relative_nll: harm_t = (Σnll_arm/Σtok_arm − Σnll_ref/Σtok_ref)
    / (Σnll_ref/Σtok_ref). Domain harm = task_weight-weighted mean.
    """
    if not pairs:
        raise StatsError(f"domain {domain.name!r}: no paired tasks")
    weights = _task_weights(domain, pairs)
    harms = np.empty(len(pairs), dtype=np.float64)
    for t, p in enumerate(pairs):
        idx = (idx_per_task[t] if idx_per_task is not None
               else np.arange(p.n_items))
        if domain.kind == "accuracy":
            diff = p.ref_values[idx] - p.arm_values[idx]
            harm = float(diff.mean())
            if domain.direction == "down":
                harm = -harm
        else:  # nll / relative_nll
            arm_nll = p.arm_values[idx].sum() / p.arm_tokens[idx].sum()
            ref_nll = p.ref_values[idx].sum() / p.ref_tokens[idx].sum()
            if ref_nll <= 0:
                raise StatsError(f"domain {domain.name!r} task {p.task!r}: "
                                 f"non-positive reference NLL {ref_nll}")
            harm = (arm_nll - ref_nll) / ref_nll
            if domain.direction == "up":
                harm = -harm
        harms[t] = harm
    return float(np.dot(weights, harms))


def bootstrap_harms(domain: Domain, pairs: Sequence[TaskPairs], *,
                    b: int, rng: np.random.Generator) -> np.ndarray:
    """B paired-bootstrap harm replicates: items resampled with
    replacement WITHIN each task (stratified), the registered aggregate
    recomputed per replicate (F14)."""
    out = np.empty(b, dtype=np.float64)
    idx_draws = [rng.integers(0, p.n_items, size=(b, p.n_items))
                 for p in pairs]
    for rep in range(b):
        out[rep] = domain_harm(domain, pairs,
                               [d[rep] for d in idx_draws])
    return out


# --------------------------------------------------------------------------
# McNemar (display-only)
# --------------------------------------------------------------------------


def mcnemar_exact_p(pairs: Sequence[TaskPairs]) -> float:
    """Exact two-sided McNemar p on the pooled discordant pairs
    (binomial(n=b+c, 1/2)). DISPLAY-ONLY: tests a zero null, not the
    margin — never decides a status."""
    b_ct = c_ct = 0
    for p in pairs:
        ref = p.ref_values >= 0.5
        arm = p.arm_values >= 0.5
        b_ct += int(np.sum(ref & ~arm))
        c_ct += int(np.sum(~ref & arm))
    n = b_ct + c_ct
    if n == 0:
        return 1.0
    k = min(b_ct, c_ct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


# --------------------------------------------------------------------------
# Domain statuses
# --------------------------------------------------------------------------


@dataclass
class DomainStatus:
    """One domain's non-inferiority result for one arm."""

    domain: str
    kind: str
    harm: float  # point estimate, harm orientation
    margin: float
    status: str  # preserved | degraded | indeterminate
    upper_bound: float  # 1-α/k percentile of the harm bootstrap
    lower_bound: float  # α/k percentile
    alpha: float
    k_family: int  # 1 == unadjusted (acceptance role / excluded rows)
    n_items: int
    in_gating_set: bool
    role: Optional[str]
    mcnemar_p: Optional[float]  # display-only; None for NLL domains
    bound_method: str = BOUND_METHOD

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def classify(harm: float, upper: float, lower: float, margin: float) -> str:
    if upper < margin:
        return STATUS_PRESERVED
    if lower > 0 and harm > margin:
        return STATUS_DEGRADED
    return STATUS_INDETERMINATE


def domain_status(arm_df: pd.DataFrame, ref_df: pd.DataFrame, domain: Domain,
                  *, margin: float, k_family: int, in_gating_set: bool,
                  alpha: float = 0.05, b: int = 10_000,
                  seed: int = 0,
                  allow_arm_task_subset: bool = False) -> Optional[DomainStatus]:
    """One domain's status (None when no rows for its tasks exist in
    either leg — the caller records the gap in the gate log)."""
    if domain.kind not in ("accuracy", "nll"):
        raise StatsError(f"domain {domain.name!r} is kind {domain.kind!r} — "
                         "gate/display domains never get non-inferiority "
                         "statuses (§2.2)")
    if k_family < 1:
        raise StatsError(f"k_family must be >= 1, got {k_family}")
    pairs = pair_domain(arm_df, ref_df, domain,
                        allow_arm_task_subset=allow_arm_task_subset)
    if not pairs:
        return None
    rng = np.random.default_rng(seed)
    harm = domain_harm(domain, pairs)
    boots = bootstrap_harms(domain, pairs, b=b, rng=rng)
    upper = float(np.quantile(boots, 1.0 - alpha / k_family))
    lower = float(np.quantile(boots, alpha / k_family))
    return DomainStatus(
        domain=domain.name, kind=domain.kind, harm=harm, margin=float(margin),
        status=classify(harm, upper, lower, margin),
        upper_bound=upper, lower_bound=lower, alpha=alpha, k_family=k_family,
        n_items=sum(p.n_items for p in pairs),
        in_gating_set=in_gating_set, role=domain.role,
        mcnemar_p=(mcnemar_exact_p(pairs) if domain.kind == "accuracy" else None),
    )


def statuses_for_arm(
    arm_rows: Any,
    ref_rows: Any,
    registry: Registry,
    *,
    model: str,
    margins: Optional[Mapping[str, float]] = None,
    exclusions: Sequence[str] = (),
    alpha: float = 0.05,
    b: int = 10_000,
    seed: int = 0,
    allow_arm_task_subset: bool = False,
) -> Dict[str, DomainStatus]:
    """Statuses for every accuracy/nll domain with rows present (gating,
    excluded, and acceptance-role alike — excluded/acceptance rows still
    run and report, they just leave the families/screen).

    Family size k = the gating-domain count for this run; rows outside
    the gating set get k=1 (unadjusted).
    """
    arm_df, ref_df = rows_frame(arm_rows), rows_frame(ref_rows)
    check_engine_matched(arm_df, ref_df)
    check_model_matched(arm_df, ref_df)
    margins = dict(margins or {})
    gating = {d.name for d in registry.gating_set(model, {}, exclusions)}
    k = len(gating)
    if k == 0:
        raise StatsError("statuses_for_arm: empty gating set — exclusions "
                         "removed every gating domain")
    unknown_margin = sorted(set(margins) - {d.name for d in registry.domains})
    if unknown_margin:
        raise StatsError(f"margin override(s) for unregistered domain(s) "
                         f"{unknown_margin}")
    out: Dict[str, DomainStatus] = {}
    for i, dom in enumerate(registry.domains):
        if dom.kind not in ("accuracy", "nll"):
            continue
        in_g = dom.name in gating
        st = domain_status(
            arm_df, ref_df, dom,
            margin=margins.get(dom.name, dom.margin),
            k_family=(k if in_g else 1), in_gating_set=in_g,
            alpha=alpha, b=b, seed=seed + i,
            allow_arm_task_subset=allow_arm_task_subset,
        )
        if st is not None:
            out[dom.name] = st
    return out
