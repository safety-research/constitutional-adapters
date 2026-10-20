"""Per-scenario DISPERSION of Petri judge scores — the third term beside the
mean shift and the coverage of a paired contrast.

Petri files one judged score (1..10) per (cell, behaviour, scenario). The
campaign's read-out reports, per cell x behaviour, the mean over scenarios
with a percentile bootstrap interval, and — paired on the SHARED scenario
ids against the lane's own unsteered cell — the mean difference with its
paired bootstrap interval, the exact sign test, and the coverage counts
(``n_paired``, ``n_pos`` / ``n_neg`` / ``n_tie``). What neither block
carries is the spread of the scores those means are taken over: no
per-behaviour SD, no SD of the paired differences, and therefore no
per-behaviour standardized effect. The only "Cohen's d" in the campaign
standardizes a cell's six behaviour-mean differences by the SD ACROSS the
six behaviour means — a between-behaviour heterogeneity term computed from
``results.json`` means alone, not the per-scenario noise the mean shift is
actually measured against.

This module computes the missing term from the SAME per-scenario values the
existing terms use, so it cannot disagree with them:

* :func:`dispersion_block` — per cell x behaviour: the sample SD
  (``ddof=1``) of the scenario scores, with a percentile bootstrap interval
  over scenarios.
* :func:`paired_dispersion` — per paired contrast, on the SHARED scenario
  ids in sorted order (the existing pairing): the SD of the paired
  differences; the paired standardized mean difference
  ``d_z = mean(diff) / sd(diff)`` (Cohen's d for paired designs); and,
  labelled as such, the pooled-SD variant
  ``d_pooled = mean(diff) / sqrt((var_a + var_b) / 2)`` over the same shared
  scenarios — the per-behaviour, per-scenario form of the artifact's
  statistic. Every interval is a percentile bootstrap over scenarios with
  the SAME draw scheme, ``n_boot`` and ``seed`` as the existing paired mean
  difference (``rng.integers(0, n, size=(n_boot, n))`` on
  ``numpy.random.default_rng(seed)``), so the replicate means are identical
  to the existing term's; the block returns ``delta`` / ``delta_lo`` /
  ``delta_hi`` so the caller can ASSERT that equality rather than trust it.

* :func:`cell_paired_dispersion` — per CELL: every (behaviour, scenario)
  item the cell shares with bare, paired on that key, pooled across the
  behaviours into ONE vector of paired differences; ``delta``, ``sd_diff`` and
  the paired ``d_z = delta / sd_diff`` over it, same bootstrap scheme, ``n``
  = the paired items (60 for a complete 6 x 10 cell). This is the honest
  single number for a per-cell axis: the artifact's ``d_audit`` standardizes
  a cell's shift in the six behaviour MEANS by the SD ACROSS those six means —
  between-behaviour heterogeneity, unpaired over scenarios — so a cell whose
  six means happen to be uniform gets a large d from a small shift. The
  per-cell paired d_z is standardized by the scenario-level noise of the
  paired differences instead. Behaviours enter with their item counts (equal
  weights when every behaviour is complete); the block says so.

Small n is stated in the output, not only in prose. Petri audits 10
scenarios per behaviour per cell. Every block carries ``n`` (or
``n_paired``), ``small_n`` (``n < SMALL_N``) and, when small, a ``caveat``
string. A standardized effect at n = 10 is a ratio of two noisy estimates
and its percentile bootstrap interval is anti-conservative; replicates
whose resampled SD is zero leave the effect undefined and are COUNTED
(``n_boot_degenerate``) rather than dropped silently — the interval is over
the finite replicates and says how many there were.

Pure numpy; no I/O.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

#: the existing read-out's bootstrap: 10,000 percentile draws, seed 0
DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 0
#: below this many scenarios every block flags ``small_n`` and carries the
#: caveat text — an SD from fewer draws is itself a noisy estimate
SMALL_N = 30
#: sample SD (n - 1 in the denominator) throughout
SD_DDOF = 1
#: Petri's judged scale, stated so the floor/ceiling caveat is checkable
SCORE_SCALE = (1.0, 10.0)

SMALL_N_CAVEAT = (
    "n={n} scenarios (< {thr}): the SD is estimated from {n} draws and its "
    "own interval is wide; a standardized effect is a ratio of two noisy "
    "estimates and its percentile bootstrap interval at this n is "
    "anti-conservative; judged scores are bounded {lo:g}..{hi:g}, so the SD "
    "is floor/ceiling-limited. Read the raw mean shift beside it; do not rank "
    "cells on the standardized effect alone.")

DEFINITIONS = {
    "sd": f"sample SD (ddof={SD_DDOF}) of the per-scenario judge scores of one cell x behaviour",
    "sd_diff": f"sample SD (ddof={SD_DDOF}) of the paired differences (cell minus bare) over the SHARED scenario ids",
    "d_z": "paired standardized mean difference: delta / sd_diff (Cohen's d for paired designs); positive = the cell scores HIGHER than bare",
    "d_pooled": "pooled-SD standardized mean difference over the shared scenarios: delta / sqrt((var_bare + var_cell) / 2); the per-scenario form of the artifact's cross-behaviour d",
    "cell_d_z": ("per-cell paired standardized mean difference: every (behaviour, scenario) item shared with bare, pooled into one vector "
                 "of paired differences; delta / sd_diff over it; behaviours weighted by their item counts (equal when complete). "
                 "NOT the artifact's d_audit (shift in the six behaviour means / SD across those means, unpaired)"),
    "interval": ("percentile bootstrap over scenarios with the existing paired mean difference's draw scheme "
                 "(numpy default_rng(seed).integers(0, n, (n_boot, n))); for a standardized effect the interval is over "
                 "the replicates whose resampled SD is non-zero, n_boot_degenerate counts the rest"),
    "small_n": f"true when fewer than {SMALL_N} scenarios enter the block; the caveat field then states why the numbers are fragile",
}


def caveat_for(n: int) -> Optional[str]:
    """The small-n caveat text for ``n`` scenarios, or None at or above
    :data:`SMALL_N`."""
    if n >= SMALL_N:
        return None
    return SMALL_N_CAVEAT.format(n=n, thr=SMALL_N, lo=SCORE_SCALE[0],
                                 hi=SCORE_SCALE[1])


def _draws(n: int, n_boot: int, seed: int) -> np.ndarray:
    """The existing read-out's bootstrap index draws, bit for bit."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _pct_interval(sorted_vals: np.ndarray
                  ) -> Tuple[Optional[float], Optional[float]]:
    """The existing read-out's percentile convention on a SORTED replicate
    array: ``m[int(0.025 k)]`` and ``m[int(0.975 k) - 1]``."""
    k = int(sorted_vals.size)
    if k == 0:
        return None, None
    lo_i = int(0.025 * k)
    hi_i = max(int(0.975 * k) - 1, lo_i)
    return float(sorted_vals[lo_i]), float(sorted_vals[hi_i])


def sample_sd(values: Sequence[float]) -> Optional[float]:
    """Sample SD (``ddof=1``) of ``values``; None below two values."""
    v = np.asarray(list(values), dtype=float)
    if v.size < 2:
        return None
    return float(np.std(v, ddof=SD_DDOF))


def dispersion_block(values: Sequence[float], n_boot: int = DEFAULT_N_BOOT,
                     seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Per cell x behaviour: the SD of the scenario scores with a percentile
    bootstrap interval over scenarios, the n, and the small-n flag/caveat."""
    v = np.asarray(list(values), dtype=float)
    n = int(v.size)
    out: Dict[str, Any] = {
        "n": n, "sd": sample_sd(v), "sd_lo": None, "sd_hi": None,
        "ddof": SD_DDOF, "n_boot": n_boot, "seed": seed,
        "small_n": n < SMALL_N, "caveat": caveat_for(n),
    }
    if n >= 2:
        reps = np.sort(v[_draws(n, n_boot, seed)].std(axis=1, ddof=SD_DDOF))
        out["sd_lo"], out["sd_hi"] = _pct_interval(reps)
    return out


def _standardized(delta: float, rep_mean: np.ndarray, scale: Optional[float],
                  rep_scale: np.ndarray, name: str,
                  undefined_reason: str) -> Dict[str, Any]:
    """``delta / scale`` with a percentile bootstrap interval over the
    replicates whose resampled scale is non-zero; the zero-scale replicates
    are counted, never dropped silently."""
    if scale is None or scale == 0:
        return {name: None, f"{name}_lo": None, f"{name}_hi": None,
                f"{name}_reason": undefined_reason}
    finite = rep_scale > 0
    reps = np.sort(rep_mean[finite] / rep_scale[finite])
    lo, hi = _pct_interval(reps)
    return {name: float(delta / scale), f"{name}_lo": lo, f"{name}_hi": hi,
            f"{name}_n_boot_finite": int(finite.sum()),
            f"{name}_n_boot_degenerate": int((~finite).sum())}


def paired_dispersion(a: Mapping[str, float], b: Mapping[str, float],
                      n_boot: int = DEFAULT_N_BOOT,
                      seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """``b`` minus ``a`` over the SHARED scenario ids (sorted): the SD of the
    paired differences and the standardized mean differences (paired
    ``d_z``, pooled ``d_pooled``), each with a paired percentile bootstrap
    interval over scenarios, plus ``delta`` / ``delta_lo`` / ``delta_hi``
    from the SAME replicates for the caller to check against the existing
    paired mean difference. Mirrors the existing term's PENDING shape when
    the two cells share no scenario."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"status": "PENDING", "reason": "no shared scenarios"}
    av = np.asarray([a[i] for i in ids], dtype=float)
    bv = np.asarray([b[i] for i in ids], dtype=float)
    d = bv - av
    n = len(ids)
    delta = float(d.mean())
    idx = _draws(n, n_boot, seed)
    rep_d = d[idx]
    rep_mean = rep_d.mean(axis=1)
    delta_lo, delta_hi = _pct_interval(np.sort(rep_mean))
    out: Dict[str, Any] = {
        "n_paired": n, "delta": delta, "delta_lo": delta_lo, "delta_hi": delta_hi,
        "sd_diff": sample_sd(d), "sd_diff_lo": None, "sd_diff_hi": None,
        "ddof": SD_DDOF, "n_boot": n_boot, "seed": seed,
        "small_n": n < SMALL_N, "caveat": caveat_for(n),
        "definitions": {k: DEFINITIONS[k] for k in ("sd_diff", "d_z", "d_pooled", "interval")},
    }
    if n < 2:
        reason = "n_paired < 2: no dispersion is estimable from one shared scenario"
        out.update({"d_z": None, "d_z_lo": None, "d_z_hi": None, "d_z_reason": reason,
                    "pooled_sd": None, "d_pooled": None, "d_pooled_lo": None,
                    "d_pooled_hi": None, "d_pooled_reason": reason})
        return out
    rep_sd = rep_d.std(axis=1, ddof=SD_DDOF)
    out["sd_diff_lo"], out["sd_diff_hi"] = _pct_interval(np.sort(rep_sd))
    out.update(_standardized(
        delta, rep_mean, out["sd_diff"], rep_sd, "d_z",
        "sd_diff is 0: every shared scenario shifted by the same amount, so "
        "the paired standardized effect is undefined — read delta"))
    var_a = float(av.var(ddof=SD_DDOF))
    var_b = float(bv.var(ddof=SD_DDOF))
    pooled = math.sqrt((var_a + var_b) / 2.0)
    out["pooled_sd"] = pooled
    rep_pooled = np.sqrt((av[idx].var(axis=1, ddof=SD_DDOF)
                          + bv[idx].var(axis=1, ddof=SD_DDOF)) / 2.0)
    out.update(_standardized(
        delta, rep_mean, pooled, rep_pooled, "d_pooled",
        "pooled_sd is 0: both cells score every shared scenario identically "
        "within themselves, so the pooled standardized effect is undefined — "
        "read delta"))
    return out


def meta() -> Dict[str, Any]:
    """The description a read-out files beside the numbers."""
    return {"definitions": dict(DEFINITIONS), "small_n_threshold": SMALL_N,
            "small_n_caveat": SMALL_N_CAVEAT.format(
                n="<n>", thr=SMALL_N, lo=SCORE_SCALE[0], hi=SCORE_SCALE[1]),
            "score_scale": list(SCORE_SCALE), "ddof": SD_DDOF}


def cell_paired_dispersion(a: Mapping[str, Mapping[str, float]],
                           b: Mapping[str, Mapping[str, float]],
                           n_boot: int = DEFAULT_N_BOOT,
                           seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Per CELL, ``b`` minus ``a`` over every (behaviour, scenario) item the two
    cells share (``{behaviour: {scenario_id: score}}`` on both sides), pooled
    into ONE vector of paired differences in sorted (behaviour, scenario)
    order: ``delta``, ``sd_diff`` and the paired ``d_z = delta / sd_diff``,
    each with a percentile bootstrap over the items (the existing draw
    scheme), ``n`` = the paired items, ``n_per_behavior`` the count each
    behaviour contributes (its weight). Mirrors the PENDING shape when no
    item is shared. Zero-SD replicates are counted, never dropped; small n
    is flagged and explained in the output."""
    items = sorted((beh, sid) for beh in set(a) & set(b)
                   for sid in set(a[beh]) & set(b[beh]))
    if not items:
        return {"status": "PENDING", "reason": "no shared (behaviour, scenario) items"}
    d = np.asarray([b[beh][sid] - a[beh][sid] for beh, sid in items], dtype=float)
    n = len(items)
    n_per: Dict[str, int] = {}
    for beh, _ in items:
        n_per[beh] = n_per.get(beh, 0) + 1
    delta = float(d.mean())
    idx = _draws(n, n_boot, seed)
    rep_d = d[idx]
    rep_mean = rep_d.mean(axis=1)
    delta_lo, delta_hi = _pct_interval(np.sort(rep_mean))
    out: Dict[str, Any] = {
        "n": n, "n_behaviors": len(n_per), "n_per_behavior": dict(sorted(n_per.items())),
        "delta": delta, "delta_lo": delta_lo, "delta_hi": delta_hi,
        "sd_diff": sample_sd(d), "sd_diff_lo": None, "sd_diff_hi": None,
        "ddof": SD_DDOF, "n_boot": n_boot, "seed": seed,
        "small_n": n < SMALL_N, "caveat": caveat_for(n),
        "pooling": ("paired differences of every shared (behaviour, scenario) item pooled into one vector; each behaviour "
                    "weighs by its item count (n_per_behavior) — equal weights only when every behaviour is complete; "
                    "the six behaviours are pooled as exchangeable paired differences, so this is a cell-level summary, "
                    "not a test of any one behaviour"),
        "definitions": {"cell_d_z": DEFINITIONS["cell_d_z"], "interval": DEFINITIONS["interval"]},
    }
    if n < 2:
        out.update({"d_z": None, "d_z_lo": None, "d_z_hi": None,
                    "d_z_reason": "n < 2: no dispersion is estimable from one shared item"})
        return out
    rep_sd = rep_d.std(axis=1, ddof=SD_DDOF)
    out["sd_diff_lo"], out["sd_diff_hi"] = _pct_interval(np.sort(rep_sd))
    out.update(_standardized(
        delta, rep_mean, out["sd_diff"], rep_sd, "d_z",
        "sd_diff is 0: every shared item shifted by the same amount, so the "
        "paired standardized effect is undefined — read delta"))
    return out
