"""The IFEval canonical battery — the ITEM-CLUSTER PAIRED PERCENTILE BOOTSTRAP over RATIO estimands.

Per leg, per ITEM, a ``(num, den, den_full)`` triple of non-negative integers: the numerator and denominator the item
contributes to the leg's ratio — prompt-level ``(Σ_trials pass, n_scored_trials, n_trials)``, instruction-level
``(Σ_trials followed, n_inst x n_scored_trials, n_inst x n_trials)`` — and the FULL denominator it would contribute had
every draw scored. The point of a leg is ``Σnum / Σden`` (IFEval's own definition of the instruction-level number:
followed instructions over all instructions); a contrast is the signed sum of its legs' ratios; the 95 % interval is a
percentile bootstrap over ITEMS (the item is the resampling unit — the two trials of one item are correlated), B = 10,000,
seed 0 (``pairing.DEFAULT_BOOTSTRAP`` / ``DEFAULT_BOOTSTRAP_SEED`` / ``DEFAULT_CI``), ONE index draw per replicate applied
to EVERY leg (paired: the shared seeds make the (item, trial) pairing meaningful). Unscored draws leave both numerator and
denominator (counted in ``den_full - den``, never imputed); the censoring-sensitivity bounds impute them adversely.

EXACT ARITHMETIC AS ``tinycap_math500_thinking.paired_bootstrap_conditional`` (its generalisation from binary k = 1 items
to integer ratios): the full-set point and bounds are ``Fraction``s over the integer totals converted once; each
replicate statistic is ONE float division of exact integers — a contrast over the PRODUCT of its legs' replicate
denominators, the bounds over the replicate's full denominator; the index draws are ``rng.integers(0, n, size=(m, n))``
in the same chunk order, a pure function of ``(seed, n, B)``; degenerate replicates (a leg with a zero denominator) are
discarded and counted; intervals through ``pairing._quantiles``. On a k = 1 BINARY fixture the output equals
``paired_bootstrap_conditional``'s BIT FOR BIT (pinned by ``tests/test_item_bootstrap.py``).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from sisteer.capeval.tiers import pairing as pr

Triple = tuple[int, int, int]


class ItemBootstrapError(ValueError):
    """The legs or contrasts are not what the estimator is defined over."""


def check_legs(legs: Mapping[str, Mapping[str, Triple]], contrasts: Mapping[str, Mapping[str, int]]) -> list[str]:
    """ONE common attempted id set across every leg (a difference refuses, listing ids — the estimand is defined over one
    common set of attempted items); every contrast naming only known legs with nonzero integer signs; per item a
    ``(num, den, den_full)`` triple of non-negative ints with ``num <= den <= den_full`` and ``den_full >= 1``."""
    if not isinstance(legs, Mapping) or not legs:
        raise ItemBootstrapError("the paired bootstrap needs a non-empty mapping of legs")
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise ItemBootstrapError("the paired bootstrap needs at least one contrast")
    for name, signs in contrasts.items():
        if not isinstance(signs, Mapping) or not signs:
            raise ItemBootstrapError(f"contrast {name!r} names no leg")
        for k, s in signs.items():
            if k not in legs:
                raise ItemBootstrapError(f"contrast {name!r} names a leg {k!r} that is not among the legs {sorted(legs)}")
            if isinstance(s, bool) or not isinstance(s, int) or s == 0:
                raise ItemBootstrapError(f"contrast {name!r}: the sign of leg {k!r} must be a nonzero integer, got {s!r}")
    ids_by_leg = {k: set(legs[k]) for k in legs}
    common = set.intersection(*ids_by_leg.values())
    union = set.union(*ids_by_leg.values())
    if common != union:
        detail = []
        for k in legs:
            missing = sorted(union - ids_by_leg[k])
            if missing:
                detail.append(f"{k} lacks {len(missing)} (first {missing[:3]})")
        raise ItemBootstrapError("the legs' ATTEMPTED id sets differ — the estimand is defined over ONE common set of "
                                 f"attempted items; refusing: {'; '.join(detail)}")
    if not common:
        raise ItemBootstrapError("the paired bootstrap has no items")
    for k in legs:
        for i, triple in legs[k].items():
            if not (isinstance(triple, (tuple, list)) and len(triple) == 3):
                raise ItemBootstrapError(f"{k} item {i!r}: expected (num, den, den_full), got {triple!r}")
            num, den, full = triple
            for name, v in (("num", num), ("den", den), ("den_full", full)):
                if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
                    raise ItemBootstrapError(f"{k} item {i!r}: {name} {v!r} is not a non-negative int")
            if not 0 <= num <= den <= full or full < 1:
                raise ItemBootstrapError(f"{k} item {i!r}: need 0 <= num <= den <= den_full and den_full >= 1, got {triple!r}")
    return sorted(common)


def binary_legs(legs: Mapping[str, Mapping[str, tuple[int, float | None]]]) -> dict[str, dict[str, Triple]]:
    """The k = 1 BINARY shape ``{label: {item: (mask, value)}}`` of the MATH bootstrap as ratio triples: a scored item
    ``(value, 1, 1)``, a masked-out item ``(0, 0, 1)``."""
    out: dict[str, dict[str, Triple]] = {}
    for k, items in legs.items():
        out[k] = {}
        for i, (mask, value) in items.items():
            out[k][i] = (int(value) if mask == 1 else 0, 1 if mask == 1 else 0, 1)
    return out


def paired_item_bootstrap_ratios(legs: Mapping[str, Mapping[str, Triple]], contrasts: Mapping[str, Mapping[str, int]], *,
                                 n_resamples: int = pr.DEFAULT_BOOTSTRAP, seed: int = pr.DEFAULT_BOOTSTRAP_SEED,
                                 ci: float = pr.DEFAULT_CI,
                                 max_degenerate_frac: float = pr.DEFAULT_MAX_DEGENERATE_FRAC) -> dict[str, Any]:
    """The estimator (module docstring). Returns pairing's document shape: ``acc_by_leg`` (``point``, ``ci_low``,
    ``ci_high``, the adverse-imputation ``acc_minus`` / ``acc_plus``, ``n_correct`` = Σnum, ``n_scored`` = Σden,
    ``n_censored`` = Σ(den_full - den)), ``stats`` per contrast (``point``, ``ci_low``, ``ci_high``, ``replicate_signs``,
    ``bound_low`` / ``bound_high`` + their CIs, ``excludes_zero``, ``point_sign``, ``sign_identified``, ``legs``),
    ``complete_case`` (the items scored in every leg of the contrast)."""
    if not 0.0 < ci < 1.0:
        raise ItemBootstrapError(f"the interval level must be in (0, 1), got {ci!r}")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ItemBootstrapError(f"the bootstrap needs >= 1 resamples, got {n_resamples!r}")
    if not 0.0 <= max_degenerate_frac < 1.0:
        raise ItemBootstrapError(f"max_degenerate_frac must be in [0, 1), got {max_degenerate_frac!r}")
    ids = check_legs(legs, contrasts)
    n = len(ids)
    keys = list(legs)
    k_max = max(len(signs) for signs in contrasts.values())
    max_den = max(int(legs[k][i][2]) for k in keys for i in ids)
    if k_max * (n * max_den) ** k_max >= 2 ** 53:
        raise ItemBootstrapError(f"{n} items with denominators up to {max_den} over a {k_max}-leg contrast: the exact integer "
                                 "replicate arithmetic would leave float64's exactly-representable range — refusing rather than "
                                 "rounding twice")
    num = {k: np.asarray([int(legs[k][i][0]) for i in ids], dtype=np.int64) for k in keys}
    den = {k: np.asarray([int(legs[k][i][1]) for i in ids], dtype=np.int64) for k in keys}
    full = {k: np.asarray([int(legs[k][i][2]) for i in ids], dtype=np.int64) for k in keys}
    tot_num = {k: int(num[k].sum()) for k in keys}
    tot_den = {k: int(den[k].sum()) for k in keys}
    tot_full = {k: int(full[k].sum()) for k in keys}
    for k in keys:
        if tot_den[k] == 0:
            raise ItemBootstrapError(f"{k} scored no draw — its ratio is undefined")
    n_censored = {k: tot_full[k] - tot_den[k] for k in keys}

    def full_set(acc: Mapping[str, Any], minus: Mapping[str, Any], plus: Mapping[str, Any]) -> dict[str, tuple[Any, Any, Any]]:
        out: dict[str, tuple[Any, Any, Any]] = {}
        for name, signs in contrasts.items():
            point = sum(s * acc[k] for k, s in signs.items())
            low = sum((minus[k] if s > 0 else -plus[k]) for k, s in signs.items())
            high = sum((plus[k] if s > 0 else -minus[k]) for k, s in signs.items())
            out[name] = (point, low, high)
        return out

    acc_fr = {k: Fraction(tot_num[k], tot_den[k]) for k in keys}
    minus_fr = {k: Fraction(tot_num[k], tot_full[k]) for k in keys}
    plus_fr = {k: Fraction(tot_num[k] + n_censored[k], tot_full[k]) for k in keys}
    point = full_set(acc_fr, minus_fr, plus_fr)

    rng = np.random.default_rng(seed)
    rep_acc = {k: np.empty(n_resamples) for k in keys}
    rep_stat = {c: np.empty(n_resamples) for c in contrasts}
    rep_low = {c: np.empty(n_resamples) for c in contrasts}
    rep_high = {c: np.empty(n_resamples) for c in contrasts}
    keep = np.ones(n_resamples, dtype=bool)
    chunk = max(1, 4_000_000 // n)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        sl = slice(done, done + m)
        den_r = {k: den[k][idx].sum(axis=1) for k in keys}
        num_r = {k: num[k][idx].sum(axis=1) for k in keys}
        full_r = {k: full[k][idx].sum(axis=1) for k in keys}
        for k in keys:
            degenerate = den_r[k] == 0
            keep[sl] &= ~degenerate
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = num_r[k] / den_r[k]
            rep_acc[k][sl] = np.where(degenerate, np.nan, cond)
        for c, signs in contrasts.items():
            d: Any = 1
            for k in signs:
                d = d * den_r[k]
            nm: Any = 0
            for k, s in signs.items():
                others: Any = 1
                for j in signs:
                    if j != k:
                        others = others * den_r[j]
                nm = nm + s * num_r[k] * others
            low_num: Any = 0
            high_num: Any = 0
            for k, s in signs.items():
                censored = full_r[k] - den_r[k]
                low_num = low_num + (num_r[k] if s > 0 else -(num_r[k] + censored))
                high_num = high_num + ((num_r[k] + censored) if s > 0 else -num_r[k])
            # the bounds' denominator: the replicate's FULL denominator — every leg of a contrast shares the item set,
            # and the full denominators are per item, so the first leg's full sum is every leg's (== n at k = 1 binary)
            first = next(iter(signs))
            with np.errstate(divide="ignore", invalid="ignore"):
                rep_stat[c][sl], rep_low[c][sl], rep_high[c][sl] = nm / d, low_num / full_r[first], high_num / full_r[first]
        done += m
    n_degenerate = int((~keep).sum())
    if n_degenerate > max_degenerate_frac * n_resamples:
        raise ItemBootstrapError(f"{n_degenerate} of {n_resamples} replicates had a leg with no scored draw (> "
                                 f"{max_degenerate_frac:.1%}) — a leg is nearly all censored; the ratio is not estimable here")
    alpha = 1.0 - ci
    acc_by_leg: dict[str, dict[str, Any]] = {}
    for k in keys:
        lo, hi = pr._quantiles(rep_acc[k][keep], ci)
        acc_by_leg[k] = {"point": float(acc_fr[k]), "ci_low": lo, "ci_high": hi,
                         "acc_minus": float(minus_fr[k]), "acc_plus": float(plus_fr[k]),
                         "n_correct": tot_num[k], "n_scored": tot_den[k], "n_censored": n_censored[k]}
    stats: dict[str, dict[str, Any]] = {}
    for c, signs in contrasts.items():
        pt, blo, bhi = point[c]
        reps = rep_stat[c][keep]
        lo, hi = pr._quantiles(reps, ci)
        stats[c] = {
            "point": float(pt), "ci_low": lo, "ci_high": hi,
            "replicate_signs": {"negative": int((reps < 0).sum()), "zero": int((reps == 0).sum()), "positive": int((reps > 0).sum())},
            "bound_low": float(blo), "bound_high": float(bhi),
            "bound_ci_low": float(np.quantile(rep_low[c][keep], alpha / 2.0)),
            "bound_ci_high": float(np.quantile(rep_high[c][keep], 1.0 - alpha / 2.0)),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "point_sign": (pt > 0) - (pt < 0),
            "sign_identified": bool(blo > 0 or bhi < 0),
            "legs": dict(signs),
        }
    complete: dict[str, Any] = {"label": pr.COMPLETE_CASE_LABEL}
    for c, signs in contrasts.items():
        both = np.ones(n, dtype=bool)
        for k in signs:
            both &= den[k] > 0
        n_both = int(both.sum())
        if n_both == 0:
            complete[c] = {"n_items": 0, "delta": None, "unavailable": "no item scored in every leg of the contrast"}
            continue
        delta = sum(s * Fraction(int(num[k][both].sum()), int(den[k][both].sum())) for k, s in signs.items())
        complete[c] = {"n_items": n_both, "delta": float(delta)}
    return {"B": int(n_resamples), "seed": int(seed), "level": float(ci), "n_items": n,
            "n_scored_by_leg": dict(tot_den), "n_censored_by_leg": dict(n_censored),
            "n_correct_by_leg": dict(tot_num), "n_degenerate_replicates": n_degenerate,
            "n_replicates_used": int(keep.sum()), "acc_by_leg": acc_by_leg, "stats": stats,
            "complete_case": complete}


def ifeval_leg_triples(draws: Sequence[Mapping[str, Any]], *, metric: str) -> dict[str, Triple]:
    """One leg's per-item triples for ``metric`` from its draw records (each with ``item_id``, ``scored`` and the four
    IFEval values + ``num_instructions``): prompt-level ``(Σ pass, n_scored_trials, n_trials)``; instruction-level
    ``(Σ followed, n_inst x n_scored_trials, n_inst x n_trials)``. An unscored draw contributes to ``den_full`` only."""
    prompt_level = metric.startswith("prompt")
    out: dict[str, list[int]] = {}
    for d in draws:
        iid = str(d["item_id"])
        acc = out.setdefault(iid, [0, 0, 0])
        n_inst = int(d.get("num_instructions") or 0)
        weight = 1 if prompt_level else n_inst
        acc[2] += weight
        if not d.get("scored", True):
            continue
        v = d.get(metric)
        if v is None:
            continue
        acc[1] += weight
        acc[0] += int(bool(v)) if prompt_level else int(v)
    return {i: (a[0], a[1], a[2]) for i, a in out.items()}


__all__ = ["ItemBootstrapError", "Triple", "binary_legs", "check_legs", "ifeval_leg_triples", "paired_item_bootstrap_ratios"]
