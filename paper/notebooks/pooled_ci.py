"""pooled_ci: the ITEM-CLUSTERED interval of a cell pooled over several decode draws (pure numpy / pandas; no registry access).

A pooled cell is the mean of R readings of ONE object on ONE eval: the unsteered model's runs that anchor every steered delta, a
decode-only condition's draws (the Refusal steer, the two prompted SIs), a fitted object's re-decodes. The draws re-decode the SAME
items, often near-deterministically (MMLU bit-identical per item, BFCL at T 0.001, BigCodeBench at T 0), so they are not independent
samples and an interval that treats them as such (`se = hypot(sqrt(sum se_r^2) / R, sd / sqrt(R))`, the notebook's rule until
2026-11-19) is about sqrt(R) too narrow when they agree. The unit of independence is the ITEM (the cluster: a goal, an item, a
Petri scenario), so the cell's interval is built over items:

  each run r is a ratio over its units i, y_r = sum_s lambda_s * (sum_{i in s} a_ri) / (sum_{i in s} b_ri) (one stratum, lambda 1,
  everywhere but the pooled Petri score: six behaviours, lambda 1/6). The cell is a weighted mean Y = sum_r w_r y_r (draws: w 1/R).
  Linearising each run's ratio,
      psi_ri = lambda_s(i) * (a_ri - yhat_rs * b_ri) / B_rs          (0 where run r has no unit i)
      psi_i  = sum_r w_r psi_ri
      V      = sum_s n_s / (n_s - 1) * sum_{i in s} psi_i^2          (n_s: the units of stratum s over the union of the runs)
  With every run on the same units and b = 1 this is "each item's mean over the draws, then the standard error over items". It
  reduces to one run's clustered variance when the draws are identical and shrinks as 1/R when they are independent decode noise.

How the variance becomes the interval depends on the form of the runs' own intervals, so a pooled cell reads like its single-run
neighbours on the same panel:
  * "wilson" (binary-cell rates: ASR, accuracy, refused rates, whose own intervals are Wilson on (k, n) over their n cells): the
    runs' interval at the cluster-effective size, Wilson(p, n_eff) with n_eff = sum_r w_r n_r V_r^cells / V, V_r^cells being the
    variance the run's own Wilson interval assumes, its n_r cells independent (`cells_var`), so n_r V_r^cells is the per-cell variance
    that interval reads; with every run on n cells this is n / deff, deff = V / sum_r w_r V_r^cells (1 when both are 0). Weighting the
    per-cell variances rather than taking n = sum_r w_r n_r over deff keeps a pool of runs on different numbers of items (a 200-item
    head beside a 1,531-item pool) at its clustered variance: n / deff would read the runs' n and their own variances in separate
    weighted means, sum w n * sum w / n >= 1, and narrow the interval by that factor. Where
    every unit is one cell that IS the run's clustered variance, so identical draws reproduce one run's own Wilson interval exactly;
    where a unit holds several cells (several trials of one goal), deff also carries the within-unit design effect, so identical
    draws give one run's ITEM-CLUSTERED width (wider than its own Wilson interval, which counts the cells as independent). Near 0 %
    or 100 % the bar does not collapse; all-zero draws and one hit in one draw give neighbouring intervals (no jump);
  * "direct" (score means away from the bounds: the Petri score, the over-refusal label mean): se = sqrt(V).

The interval is then centred on the cell's mean, se = (hi - lo) / 3.92 as every single-run row's is. Nothing here reads a registry:
the callers (`registry_nb.item_units`) build each run's units from its item rows by the same rule its reading uses and check that
they reproduce it.

A POOL OF FIT SEEDS (2026-11-21) is clustered the same way: the seeds are different fitted models read on the SAME items against the
SAME anchor, so their steered runs are pooled at once over the items (`cluster_var` on every seed's runs, each weighted by its share of
the pool; the cluster is the item under every seed), and the variance between the seeds' readings is added (`fit_var`: the seeds are a
random sample of fits), with the shared anchor counted once (`seed_var`). Identical seeds give one seed's interval; seeds independent
by item give the old within term, sum_s c_s^2 V_s."""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

Z95 = 1.959963984540054                      # scipy.stats.norm.ppf(0.975)
FORMS = ("wilson", "direct")


class UnitsUnavailable(Exception):
    """A run's item rows cannot give this reading's units: the source holds none for it, or the reading's rule needs something the
    item rows do not carry. The cell cannot be item-clustered; the caller marks it (the identical-draws bound) and says so."""


class UnitsMismatch(Exception):
    """Units built from a run's item rows do not reproduce its reading: the unit rule has drifted from the reading's rule, or the rows
    are another judgment's. A defect, never a missing-data case: raised, never turned into a bound."""


class ItemIdsDisagree(Exception):
    """Runs that declare the same item set do not share its item ids: read as two disjoint samples they would narrow the interval,
    the very defect the clustering fixes."""


class DegenerateCell(Exception):
    """Every unit of every run is constant within its run, so the clustered variance is 0, and the runs are not one reading."""


class Units(NamedTuple):
    """One run's reading as a ratio over its cluster units.

    table    index: unit id (str); columns a, b (float) and stratum (str); a unit with b = 0 is absent from the reading
    present  every item id the run's item rows carry, scored or not (the alignment guard's set: an unscored item is the same item)
    value    the reading the units give (filed units: a rate in 0-1, a Petri score in 1-10)
    n        sum of b over the units (the reading's denominator, n of its own interval)"""
    table: pd.DataFrame
    present: frozenset
    value: float
    n: float


def make_units(a: pd.Series, b=1.0, stratum=None, present=None) -> Units:
    """Units from per-unit numerators `a` (indexed by unit id), denominators `b` (a Series on the same index, or a scalar) and an
    optional `stratum` per unit (Series or scalar). Units with b = 0 are dropped (absent from the reading); `present` defaults to the
    units' ids."""
    a = pd.Series(a, dtype=float)
    a.index = a.index.astype(str)
    b = pd.Series(b, index=a.index, dtype=float) if np.isscalar(b) else pd.Series(b, dtype=float).reindex(a.index)
    if b.isna().any() or a.isna().any():
        raise ValueError("a unit with a missing numerator or denominator")
    st = (pd.Series("all" if stratum is None else stratum, index=a.index, dtype=object) if stratum is None or np.isscalar(stratum)
          else pd.Series(stratum, dtype=object).reindex(a.index).astype(str))
    t = pd.DataFrame({"a": a, "b": b, "stratum": st})
    t = t[t.b > 0]
    if not len(t):
        raise UnitsUnavailable("no unit with a positive denominator")
    ids = frozenset(map(str, present)) if present is not None else frozenset(t.index)
    return Units(t, ids, reading_of(t), float(t.b.sum()))


def reading_of(table: pd.DataFrame) -> float:
    """The reading a unit table gives: the ratio sum a / sum b, averaged over the strata with equal weight."""
    per = [g.a.sum() / g.b.sum() for _, g in table.groupby("stratum", sort=True)]
    return float(np.mean(per))


def influence(units: Sequence[Units]) -> tuple:
    """(psi, stratum): psi_ri per unit (rows: the union of the runs' units) and run (columns 0..R-1), 0 where a run has no unit; the
    stratum of each unit. A unit id in two strata refuses."""
    cols, labels = {}, []
    for r, u in enumerate(units):
        t = u.table
        S = t.stratum.nunique()
        parts = []
        for s, g in t.groupby("stratum", sort=True):
            B = float(g.b.sum())
            y = float(g.a.sum()) / B
            parts.append((g.a - y * g.b) / B / S)
        cols[r] = pd.concat(parts)
        labels.append(t.stratum)
    lab = pd.concat(labels)
    both = lab.groupby(level=0).nunique()
    if (both > 1).any():
        i = both[both > 1].index[0]
        raise ValueError(f"unit {i!r} is in two strata ({', '.join(sorted(set(lab.loc[[i]])))})")
    lab = lab[~lab.index.duplicated()]
    psi = pd.DataFrame(cols).fillna(0.0)
    return psi, lab.reindex(psi.index)


def cluster_var(units: Sequence[Units], weights=None) -> float:
    """V of sum_r w_r y_r (default w_r = 1 / R), clustered by unit, with the n_s / (n_s - 1) correction per stratum. One run: its own
    clustered variance. A stratum with a single unit refuses (its variance cannot be estimated)."""
    if not len(units):
        raise ValueError("no run")
    w = np.full(len(units), 1.0 / len(units)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(units):
        raise ValueError(f"{len(w)} weights for {len(units)} runs")
    psi, strata = influence(units)
    p = psi.to_numpy() @ w
    v = 0.0
    for s in pd.unique(strata):
        m = (strata == s).to_numpy()
        ns = int(m.sum())
        if ns < 2:
            raise ValueError(f"stratum {s!r} holds {ns} unit: its variance cannot be estimated")
        v += ns / (ns - 1) * float((p[m] ** 2).sum())
    return v


def cells_var(u: Units) -> float:
    """The variance of one run's reading that its own Wilson interval on (k, n) assumes: its n = sum b CELLS independent. A unit of b
    binary cells, a of them successes, adds a (1 - p)^2 + (b - a) p^2 to its stratum's sum of squares, with the N_s / (N_s - 1)
    correction over the stratum's N_s cells. A run whose every unit is ONE cell returns its clustered variance (`cluster_var([u])`),
    the same quantity, so the Wilson form's design effect then measures the draws' agreement alone. A unit of several cells must be a
    count of binary cells (0 <= a <= b, a whole number), else the reading is not a rate a Wilson interval describes and this refuses."""
    t = u.table
    if (t.b == 1).all():
        return cluster_var([u])
    a, b = t.a.to_numpy(dtype=float), t.b.to_numpy(dtype=float)
    if (a < -1e-9).any() or (a > b + 1e-9).any() or not np.allclose(a, np.round(a), rtol=0.0, atol=1e-9):
        raise ValueError("a Wilson-form rate whose multi-cell units are not counts of binary cells")
    S = t.stratum.nunique()
    v = 0.0
    for s, g in t.groupby("stratum", sort=True):
        N = float(g.b.sum())
        if N < 2:
            raise ValueError(f"stratum {s!r} holds {N:g} cell: its variance cannot be estimated")
        p = float(g.a.sum()) / N
        ss = float((g.a * (1.0 - p) ** 2 + (g.b - g.a) * p ** 2).sum())
        v += N / (N - 1.0) * ss / (N * N) / (S * S)
    return v


def wilson(k: float, n: float, z: float = Z95) -> tuple:
    """Wilson score interval of k successes in n (k may be fractional)."""
    if not n > 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    h = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / den
    return (c - h, c + h)


class Pooled(NamedTuple):
    """A pooled reading's interval: `se` in the caller's plotted units, the bounds of the form's own interval (`lo`, `hi`, plotted
    units; Wilson's are asymmetric), `method` in words, and the design effect / effective n where the form uses them."""
    se: float
    lo: float
    hi: float
    method: str
    deff: float = np.nan
    n_eff: float = np.nan


def pooled(units: Sequence[Units], weights=None, form: str = "direct", scale: float = 1.0, rate_unit: float = 1.0,
           own_se: Sequence[float] | None = None) -> Pooled:
    """The interval of sum_r w_r y_r (default the mean of the runs) from their units (module docstring). `scale` takes a reading to
    plotted units (SCALE); `rate_unit` is a "wilson" reading's value at a rate of 1 (IS_RATE: 1 for a 0-1 rate). Weights that do not
    sum to 1 scale the interval of the normalised mean (a weighted sum is linear). `own_se`: the runs' own se in plotted units, read
    only in the degenerate "direct" case (V = 0), where runs with identical unit tables are one reading and take the first run's own
    interval; non-identical constant runs raise DegenerateCell."""
    if form not in FORMS:
        raise ValueError(f"form {form!r}: one of {FORMS}")
    w = np.full(len(units), 1.0 / len(units)) if weights is None else np.asarray(weights, dtype=float)
    W = float(w.sum())
    if not W > 0:
        raise ValueError("weights sum to 0")
    wn = w / W
    v = cluster_var(units, wn)
    y = float(sum(x * u.value for x, u in zip(wn, units)))
    if form == "wilson":
        vr = np.array([cells_var(u) for u in units])                  # the variance each run's own Wilson interval assumes
        vref = float(wn @ vr)
        deff = v / vref if vref > 0 else 1.0
        if vref <= 0 and v > 1e-15:
            raise ValueError("a positive pooled variance over runs that are each constant")
        n_ref = float(sum(x * u.n for x, u in zip(wn, units)))
        # THE CLUSTER-EFFECTIVE n: the per-cell variance the runs' own Wilson intervals read (n_r V_r^cells, weighted as the pool weights
        # the runs) over the pooled variance, so Wilson's nominal variance at n_eff is V whatever the runs' n (n_ref / deff where they share n)
        nv = float(sum(x * u.n * r for x, u, r in zip(wn, units, vr)))
        n_eff = n_ref if vref <= 0 else (nv / v if v > 0 else np.inf)
        p = min(max(y / rate_unit, 0.0), 1.0)
        if not np.isfinite(n_eff):
            lo, hi = p, p
        else:
            lo, hi = wilson(p * n_eff, n_eff)
        s = rate_unit * scale * W
        return Pooled((hi - lo) / 3.92 * s, lo * rate_unit * scale, hi * rate_unit * scale,
                      "item-clustered (Wilson at the effective n)", deff, n_eff)
    if v > 0:
        se = math.sqrt(v) * scale * W
        return Pooled(se, (y * scale - 1.96 * se / W), (y * scale + 1.96 * se / W), "item-clustered")
    same = all(u.table[["a", "b"]].sort_index().equals(units[0].table[["a", "b"]].sort_index()) for u in units[1:])
    if not same:
        raise DegenerateCell("every unit of every run is constant within its run and the runs differ: the clustered variance is 0")
    se0 = float(own_se[0]) * W if own_se is not None and len(own_se) else np.nan
    return Pooled(se0, y * scale - 1.96 * se0 / W, y * scale + 1.96 * se0 / W, "identical constant draws: one run's own interval")


def fit_var(y: Sequence[float], coef=None) -> float:
    """The BETWEEN-FIT variance of a pool of fits (fit seeds) sum_s c_s y_s: s_y^2 * sum_s c_s^2, s_y^2 the sample variance (ddof 1)
    of the fits' readings y_s (default c_s = 1 / S: s_y^2 / S, the rule set on 2026-09-20). The fits are a random sample of fits of one
    recipe, so their spread is part of the pool's uncertainty whatever the items say. 0 for one fit; NaN when a reading is NaN."""
    y = np.asarray(y, dtype=float)
    S = len(y)
    if S < 2:
        return 0.0
    c = np.full(S, 1.0 / S) if coef is None else np.asarray(coef, dtype=float)
    if len(c) != S:
        raise ValueError(f"{len(c)} coefficients for {S} fits")
    return float(np.var(y, ddof=1) * float(c @ c))


def seed_var(v_items: float, y: Sequence[float], v_anchor: float = 0.0, coef=None) -> float:
    """THE SEED POOL's variance (the fit-seed rule, 2026-11-21): V = V_items + V_fits + V_anchor.
      * V_items: the item-clustered variance of the seeds' steered runs pooled AT ONCE (`cluster_var` over every seed's runs, each
        weighted by its share of the pool), so an item's outcomes under all the seeds enter as one cluster: seeds that agree item by
        item give one seed's variance, not a third of it, and seeds independent by item give sum_s c_s^2 V_s, the old within term;
      * V_fits: the between-fit variance (`fit_var`), kept as it was: the seeds are a random sample of fits;
      * V_anchor: the shared anchor's variance, counted once (its runs weighted as the pool weights them).
    The three are added: the steered part and the anchor are independent terms (unpaired, as every delta)."""
    return float(v_items) + fit_var(y, coef) + float(v_anchor)


def newcombe_bounds(p_a: float, lo_a: float, hi_a: float, p_b: float, lo_b: float, hi_b: float) -> tuple:
    """Newcombe's hybrid interval (his method 10) for the DIFFERENCE p_a - p_b of two independent rates, from each side's own interval
    (lo, hi) in place of the two Wilson intervals on (k, n) his method takes: (d - sqrt((p_a - lo_a)^2 + (hi_b - p_b)^2),
    d + sqrt((hi_a - p_a)^2 + (p_b - lo_b)^2)). With both sides' Wilson intervals on their (k, n) it is `newcombe_diff`; a pooled
    side passes its Wilson interval at the cluster-effective n."""
    d = p_a - p_b
    return (d - math.sqrt((p_a - lo_a) ** 2 + (hi_b - p_b) ** 2), d + math.sqrt((hi_a - p_a) ** 2 + (p_b - lo_b) ** 2))


def bound_se(own_se: Sequence[float], weights=None) -> float:
    """The identical-draws bound: sum_r w_r se_r over the runs' own se. For correlated estimators SD(sum w_r y_r) <= sum w_r SD(y_r)
    whatever the correlation, so it is never narrower than the truth; NaN when a run has no own interval (unmeasured)."""
    s = np.asarray(own_se, dtype=float)
    w = np.full(len(s), 1.0 / len(s)) if weights is None else np.asarray(weights, dtype=float)
    if not len(s) or not np.isfinite(s).all():
        return np.nan
    return float(w @ s)


#: the smallest share of item ids two runs declaring one item set must share (`check_alignment`); measured on the committed frame
#: (paper/tests/test_registry_nb_pooled_ci_slow.py pins every pool's overlap)
ALIGNMENT_MIN = 0.99


def overlap(units: Sequence[Units]) -> float:
    """The smallest pairwise |Ia & Ib| / min(|Ia|, |Ib|) of the runs' present item ids (1 = every run holds the same items)."""
    sets = [u.present for u in units]
    vals = [len(a & b) / max(1, min(len(a), len(b))) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(min(vals)) if vals else 1.0


def check_alignment(units: Sequence[Units], item_sets: Sequence, names: Sequence[str] = (), minimum: float = ALIGNMENT_MIN) -> float:
    """Raise ItemIdsDisagree when two runs declaring the same item set share fewer than `minimum` of their present item ids (a spelling
    mismatch between producers would otherwise read as independence and narrow the bar). Returns the smallest overlap among runs
    that declare one item set (1.0 when no two do)."""
    worst = 1.0
    by: dict = {}
    for j, s in enumerate(item_sets):
        if s is not None and s == s and str(s):
            by.setdefault(str(s), []).append(j)
    for s, idx in by.items():
        if len(idx) < 2:
            continue
        ov = overlap([units[j] for j in idx])
        worst = min(worst, ov)
        if ov < minimum:
            who = ", ".join(str(names[j]) for j in idx) if names else f"{len(idx)} runs"
            raise ItemIdsDisagree(f"runs declaring item set {s} share only {ov:.3f} of their item ids (minimum {minimum}): {who}")
    return worst
