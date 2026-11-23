"""appendix_common: the paired comparison figure the appendix figures share, and their save helper.

    import appendix_common as AC
    fig = AC.paired_comparison_figure(rows, "figA_<name>", categories={"multi-turn": R.CATEGORIES["multi-turn"]},
                                      conditions=["published", "control_pool"],
                                      condition_labels={"published": "published stopping rule", "control_pool": "control-pool stopping rule"})
    AC.paired_means(rows, categories)            # the mean marks the figure draws, with why any is not drawn
    AC.not_drawn(rows, categories)               # the line to print under the figure: each mean left out or ringed, and why

THE FORM is `registry_nb.rule_comparison_figure` (the control-pool stopping comparison) generalised to any objects, evaluations and two or
three conditions: one x group per evaluation with thin grey separators, a dark separator, then a bold `mean` group with larger marks; in
each group the objects in the paper's order and colours; inside an object its seeds fanned; inside a seed the conditions side by side as
MARKER SHAPES (circle, diamond, square), each a small point with its own vertical 95% interval, the seed's marks joined by a thin line in
the object's colour. y is the change from the unsteered model with positive = better, with a zero line. Several category panels sit side
by side or stacked (`layout`), each titled top-left in bold; the legend is on top (colours = objects, shapes = conditions, line = 95% CI).

THE INPUT is a tidy table, one row per drawn mark: `object, seed, category, eval, condition, delta, ci_lo, ci_hi`, optionally `n` (items
behind the reading, summed into the means table, never drawn or weighted) and `open` (True draws the mark open, the paper's fill
convention: the unsteered anchor came from outside the paper frame). `delta` and the interval are in the eval's own unit, oriented by the
category's sign exactly as the notebook orients a registry delta: `categories` takes the notebook's `R.CATEGORIES` form, `{category:
(evals, sign, y label)}` (or `(evals, y label)`, sign +1), so a port that holds attack-success deltas passes sign -1 and one that holds
defense-success deltas passes +1. Left out, the categories and evals are read from the rows in order of appearance with sign +1. A row with
a missing `delta` is a reading expected but not yet in the data: it draws the gap glyph at zero and keeps its condition's mean off. `seed`
is empty in every row of an unseeded object (one reading per cell, as a prompted SI or the Refusal steer), which draws one mark per
condition at its slot's centre; an object mixing empty and named seeds is refused.

THE MEAN MARK is the notebook's category mean, computed BY the notebook's functions rather than restated here: `registry_nb.category_means`
(within seed, the unweighted mean over the category's evals, se = sqrt(sum se_e^2) / k, se_e = (ci_hi - ci_lo) / 3.92) and then
`registry_nb.pool_seeds` (over seeds with equal weight: the seeds' steered runs item-clustered across the seeds, plus the between-fit
variance sd(seed means)^2 / n, plus the shared anchor once; `registry_nb.seed_pool_se`), the two calls `rule_comparison_figure`'s mean
column makes. A row carries its delta's parts (`registry_nb.delta_parts`: `se_steer`, its steered se, and `anchor_terms`, its anchor runs
with their weights, which a one-seed mean reads; `steer_terms` and `split_anchor_terms`, its steered and anchor runs, which a mean over
several seeds reads); a mean over several seeds whose rows lack them cannot be split, and takes the identical-seeds bound over the seeds'
whole intervals plus the between-fit variance, marked in its pooled row's `se_method`. The rows reach them under opaque
object, category and model keys, so the notebook's paper-specific lookups (decode-only conditions, Petri's pooled eval, cells out by
design) never apply to a port's rows. `mean_rule="complete"` (the comparison figure's rule) draws a condition's mean only when it averages
every seed the object has in the category, each over every eval the category names, so the joined marks always average the same readings;
`mean_rule="ring"` draws any mean and rings one over fewer seeds or evals (Figure 4's ring).

THE STYLE is imported from `registry_nb`, never copied: `R.style()` (fonts, sizes, spines), `R.COLOR` / `R.LABEL` (the object palette and
names; `colors=` / `labels=` add objects the paper does not name, and an object with no colour refuses by name), the ink colours, the
comparison figure's mark areas, bar widths, gap glyph and legend handles (the ring's glyph relabelled: `RING_HANDLE`), `R.tick` for eval
labels and `R.legend_stack` for the legend.
`save_figure` writes png, pdf and svg with `bbox_inches="tight"` at the style's 200 dpi into `R.FIGS` (the notebook's `figures/`, which
the zip cell reads) and into every directory named in APPENDIX_FIGS_MIRROR (os.pathsep-separated) or passed as `extra_dirs`.
"""
from __future__ import annotations

import contextlib
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import registry_nb as R

TIDY_COLUMNS = ("object", "seed", "category", "eval", "condition", "delta", "ci_lo", "ci_hi")
KEY_COLUMNS = ("object", "seed", "category", "eval", "condition")
OPTIONAL_COLUMNS = ("n", "open", "se_steer", "anchor_terms", "steer_terms", "split_anchor_terms")
# The conditions' shapes: the comparison figure's two (circle for the first, diamond for the second), then a square for a third.
CONDITION_MARKERS = tuple(R.CTLSTOP_RULE_MARKER[r] for r in R.CTLSTOP_RULES) + ("s",)
# Equal ink per shape: matplotlib's diamond and square fill their unit square, the circle pi/4 of it, so both take the diamond's scale.
MARKER_AREA = {"o": R.CTLSTOP_RULE_AREA["published"], "D": R.CTLSTOP_RULE_AREA["control_pool"], "s": R.CTLSTOP_RULE_AREA["control_pool"]}
SEED_MARK, MEAN_MARK, SEED_BAR, MEAN_BAR = R.CTLSTOP_SEED_MARK, R.CTLSTOP_MEAN_MARK, R.CTLSTOP_SEED_BAR, R.CTLSTOP_MEAN_BAR
# Within-seed spacing of the conditions (x the seed spacing) and of the mean marks (x the object slot), by number of conditions.
PAIR_HALF = {1: 0.0, 2: 0.21, 3: 0.30}
MEAN_PAIR_HALF = {1: 0.0, 2: 0.14, 3: 0.22}
# The comparison figure's canvas width, 3 groups x 5.0 in + 1.4 in of y axis: every figure is drawn in the same font sizes, so a wider
# canvas prints its text smaller than the figures beside it; the default caps a paired figure at this width.
MODEL_WIDTH = 16.4
DEFAULT_YLABEL = "Δ from unsteered (+ is better)"
MEAN_RULES = ("complete", "ring")
# THE RING'S KEY: Figure 4's ring glyph (`R.RING_HANDLE`'s marker, size and rim, read from it), relabelled because `mean_rule="ring"` rings a
# mean over fewer SEEDS as well as one over fewer evals, and Figure 4's label names the evals only.
RING_TEXT = "ring: a mean over fewer seeds or evals than its object has (named under the figure)"
RING_HANDLE = Line2D([], [], ls="", marker=R.RING_HANDLE.get_marker(), ms=R.RING_HANDLE.get_markersize(),
                     markerfacecolor=R.RING_HANDLE.get_markerfacecolor(), markeredgecolor=R.RING_HANDLE.get_markeredgecolor(),
                     markeredgewidth=R.RING_HANDLE.get_markeredgewidth(), label=RING_TEXT)
MIRROR_ENV = "APPENDIX_FIGS_MIRROR"
FORMATS = ("png", "pdf", "svg")
_MODEL = "__appendix_common__"         # the model key the notebook's mean functions see; it matches no paper model


# ---- style and saving ----------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def style_context():
    """The notebook's publication style (`R.style()`) for the length of the block, the caller's rcParams restored after."""
    with plt.rc_context():
        R.style()
        yield


def mirror_dirs() -> list:
    """The extra directories APPENDIX_FIGS_MIRROR names (os.pathsep-separated), in order; empty when it is unset."""
    return [os.path.expanduser(p) for p in os.environ.get(MIRROR_ENV, "").split(os.pathsep) if p.strip()]


def save_figure(fig, name: str, dirs=None, extra_dirs=()) -> list:
    """Write `fig` as `<name>.png`, `.pdf` and `.svg` (tight bounding box, the style's 200 dpi) into each of `dirs` (default: the
    notebook's `R.FIGS`, read at call time), then each APPENDIX_FIGS_MIRROR and `extra_dirs` directory. Returns the paths written."""
    if not name or os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"figure name {name!r}: a bare file stem, no directory")
    targets = []
    for d in (list(dirs) if dirs is not None else [R.FIGS]) + mirror_dirs() + list(extra_dirs):
        if d not in targets:
            targets.append(d)
    out = []
    with style_context():
        for d in targets:
            os.makedirs(d, exist_ok=True)
            for ext in FORMATS:
                p = os.path.join(d, f"{name}.{ext}")
                fig.savefig(p, bbox_inches="tight")
                out.append(p)
    return out


# ---- the tidy table ----------------------------------------------------------------------------------------------------------------
def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def _appearance(values) -> list:
    return list(dict.fromkeys(values))


def resolve_categories(rows: pd.DataFrame, categories=None) -> dict:
    """{category: (evals, sign, y label)} in drawing order. `categories` takes the notebook's `{category: (evals, sign, y label)}`, or
    `(evals, y label)` / a bare eval list (sign +1, the default y label); left out, the rows' categories and evals in order of appearance."""
    if categories is None:
        return {c: (_appearance(rows.loc[rows.category == c, "eval"]), 1, DEFAULT_YLABEL) for c in _appearance(rows.category)}
    out = {}
    for c, spec in categories.items():
        nested = isinstance(spec, (list, tuple)) and len(spec) in (2, 3) and isinstance(spec[0], (list, tuple))
        if nested and len(spec) == 3:
            evals, sign, ylabel = spec
        elif nested:
            (evals, ylabel), sign = spec, 1
        else:
            evals, sign, ylabel = spec, 1, DEFAULT_YLABEL
        evals = [str(e) for e in evals]
        if not evals or len(set(evals)) != len(evals):
            raise ValueError(f"category {c!r}: evals must be a non-empty list without repeats, got {evals}")
        if sign not in (1, -1):
            raise ValueError(f"category {c!r}: sign must be +1 or -1, got {sign!r}")
        out[str(c)] = (evals, int(sign), str(ylabel))
    return out


def tidy(rows, categories=None) -> pd.DataFrame:
    """The validated table with the plotted columns added: `y`, `lo`, `hi` (delta and interval times the category's sign, the bounds
    swapped where the sign flips them), `pending` (no delta yet), `open` (False where the column is absent), `n` (NaN where absent).
    Refuses by name: a missing column, an empty key, a repeated mark, an interval that does not hold its point or has one bound only, and
    a (category, eval) the categories do not declare (it would otherwise be silently left off the figure)."""
    df = pd.DataFrame(rows).copy()
    miss = [c for c in TIDY_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"tidy rows lack column(s) {miss}; needed: {list(TIDY_COLUMNS)} (+ optional {list(OPTIONAL_COLUMNS)})")
    if not len(df):
        raise ValueError("no rows to draw")
    for c in KEY_COLUMNS:
        if c == "seed":
            continue
        if df[c].isna().any() or df[c].astype(str).str.strip().eq("").any():
            raise ValueError(f"column {c!r} has an empty value in {int((df[c].isna() | df[c].astype(str).str.strip().eq('')).sum())} row(s)")
        df[c] = df[c].astype(str)
    # AN UNSEEDED OBJECT (one reading per cell: a prompted SI, the Refusal steer) leaves `seed` empty in every row and draws one mark per
    # condition at its slot's centre; an object that mixes empty and named seeds is a data error, refused rather than drawn as a fourth seed.
    df["seed"] = df["seed"].map(lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip())
    mixed = sorted(o for o, s in df.groupby("object").seed if s.eq("").any() and s.ne("").any())
    if mixed:
        raise ValueError(f"object(s) {mixed} mix rows with and without a seed: an unseeded object leaves `seed` empty in every row")
    for c in ("delta", "ci_lo", "ci_hi"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    df["n"] = pd.to_numeric(df["n"], errors="raise").astype(float) if "n" in df.columns else np.nan
    if "open" in df.columns:
        ok = df["open"].map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)) or isinstance(v, (bool, np.bool_))
                            or (isinstance(v, (int, np.integer)) and v in (0, 1)))
        if not ok.all():
            raise ValueError(f"column 'open' must hold booleans, got {sorted(set(map(repr, df.loc[~ok, 'open'])))[:4]}")
        df["open"] = df["open"].map(lambda v: bool(v) if isinstance(v, (bool, np.bool_, int, np.integer)) else False).astype(bool)
    else:
        df["open"] = False
    dup = df[df.duplicated(list(KEY_COLUMNS), keep=False)]
    if len(dup):
        raise ValueError(f"{len(dup)} rows repeat a mark (object, seed, category, eval, condition), e.g. "
                         f"{dup[list(KEY_COLUMNS)].head(3).to_dict('records')}; pool a cell's readings into one row first")
    df["pending"] = df.delta.isna()
    one = df.ci_lo.isna() ^ df.ci_hi.isna()
    if (one & ~df.pending).any():
        raise ValueError(f"{int((one & ~df.pending).sum())} row(s) carry one interval bound only")
    have = ~df.pending & df.ci_lo.notna()
    tol = 1e-9 * np.maximum(1.0, df[["delta", "ci_lo", "ci_hi"]].abs().max(axis=1))
    bad = have & ((df.ci_lo > df.ci_hi + tol) | (df.delta < df.ci_lo - tol) | (df.delta > df.ci_hi + tol))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) whose interval does not hold its delta (ci_lo <= delta <= ci_hi), e.g. "
                         f"{df.loc[bad, list(TIDY_COLUMNS)].head(3).to_dict('records')}")
    cats = resolve_categories(df, categories)
    declared = {(c, e) for c, (evals, _s, _y) in cats.items() for e in evals}
    stray = sorted(set(zip(df.category, df["eval"])) - declared)
    if stray:
        raise ValueError(f"rows on (category, eval) pairs the categories do not declare: {stray[:6]}")
    sign = df.category.map({c: s for c, (_e, s, _y) in cats.items()}).astype(float)
    a, b = sign * df.ci_lo, sign * df.ci_hi
    df["y"] = sign * df.delta
    df["lo"], df["hi"] = np.fmin(a, b), np.fmax(a, b)
    df.loc[df.pending, ["y", "lo", "hi"]] = np.nan
    return df


# ---- the means ---------------------------------------------------------------------------------------------------------------------
def seed_means(rows, categories=None) -> pd.DataFrame:
    """One row per (object, condition, category, seed): `registry_nb.category_means` over the rows (pending rows left out), mapped back
    to the tidy names. `y` / `lo` / `hi` are oriented (+ is better); `k` how many evals the seed's mean covers, `n_evals` how many the
    category names, `n` the summed item counts (NaN without an `n` column)."""
    t = tidy(rows, categories)
    cats = resolve_categories(t, categories)
    pairs = _appearance(zip(t.object, t.condition))
    okey = {p: f"__ac{i}" for i, p in enumerate(pairs)}
    back = {v: k for k, v in okey.items()}
    out = []
    for ci, (cat, (evals, _sign, ylabel)) in enumerate(cats.items()):
        sub = t[(t.category == cat) & ~t.pending]
        if not len(sub):
            continue
        ckey = f"__cat{ci}"
        d = pd.DataFrame({"model_id": _MODEL, "object_label": [okey[p] for p in zip(sub.object, sub.condition)], "seed": sub.seed.values,
                          "eval_id": sub["eval"].values, "d_y": sub.y.values, "d_lo": sub.lo.values, "d_hi": sub.hi.values,
                          "anchor_in_frame": ~sub.open.values.astype(bool), "ceiling": False, "anchor_value": np.nan})
        if {"se_steer", "anchor_terms"} <= set(sub.columns):          # the delta's parts: the shared anchor counted once in the seed pool
            d["se_steer"] = sub.se_steer.values
            d["anchor_terms"] = list(sub.anchor_terms.values)
            if "steer_terms" in sub.columns:                          # its steered and anchor runs: the seeds item-clustered in the seed pool
                d["steer_terms"] = [v if isinstance(v, dict) else {} for v in sub.steer_terms.values]
                d["split_anchor_terms"] = ([v if isinstance(v, dict) else {} for v in sub.split_anchor_terms.values]
                                           if "split_anchor_terms" in sub.columns else [{}] * len(sub))
        cm = R.category_means(d, categories={ckey: (list(evals), 1, ylabel)})       # already oriented: sign +1
        n = sub.groupby(["object", "condition", "seed"]).n.sum(min_count=1)
        for _, r in cm.iterrows():
            o, c = back[r.object_label]
            out.append(dict(object=o, condition=c, category=cat, seed=str(r.seed), y=float(r.y), lo=float(r.lo), hi=float(r.hi),
                            k=int(r.k), n_evals=len(evals), partial=bool(r.partial), open=bool(r.open), evals=r.evals,
                            n=float(n.get((o, c, str(r.seed)), np.nan)), se_steer=float(r.se_steer), anchor_terms=r.anchor_terms,
                            steer_terms=r.steer_terms, split_anchor_terms=r.split_anchor_terms))
    return pd.DataFrame(out, columns=["object", "condition", "category", "seed", "y", "lo", "hi", "k", "n_evals", "partial", "open", "evals", "n",
                                      "se_steer", "anchor_terms", "steer_terms", "split_anchor_terms"])


def paired_means(rows, categories=None, mean_rule: str = "complete") -> pd.DataFrame:
    """One row per (object, condition, category) that has any row: the seed-pooled mean `registry_nb.pool_seeds` gives over
    `seed_means` (the comparison figure's mean column), `drawn` whether the figure draws it under `mean_rule`, `ring` whether it is drawn
    ringed, and `why` the reason a mean is not drawn or is ringed (empty otherwise). `expected_seeds` are the seeds the object has in the
    category under any condition, pending rows included; `seeds` the ones this condition's mean averages."""
    if mean_rule not in MEAN_RULES:
        raise ValueError(f"mean_rule={mean_rule!r}: one of {MEAN_RULES}")
    t = tidy(rows, categories)
    cats = resolve_categories(t, categories)
    sm = seed_means(t[[c for c in list(TIDY_COLUMNS) + list(OPTIONAL_COLUMNS) if c in t.columns]], categories=cats)
    out = []
    for cat, (evals, _s, _y) in cats.items():
        tc = t[t.category == cat]
        for o in _appearance(tc.object):
            want = sorted(set(tc.loc[tc.object == o, "seed"]), key=_natural)
            for c in _appearance(tc.loc[tc.object == o, "condition"]):
                g = sm[(sm.category == cat) & (sm.object == o) & (sm.condition == c)]
                pend = tc[(tc.object == o) & (tc.condition == c) & tc.pending]
                pend_txt = (f"{len(pend)} reading(s) not yet in the data ("
                            + "; ".join(f"{r.seed} {R.EVAL_LABEL.get(r['eval'], r['eval'])}".strip() for _, r in pend.iterrows()) + ")") if len(pend) else ""
                rec = dict(object=o, condition=c, category=cat, y=np.nan, lo=np.nan, hi=np.nan, n_seeds=0, seeds="", expected_seeds=",".join(want),
                           k=0, n_evals=len(evals), partial=True, open=False, spread=np.nan, n=np.nan, drawn=False, ring=False, why="")
                if len(g):
                    pooled = R.pool_seeds(g.rename(columns={"object": "object_label"}).assign(
                        model_id=_MODEL, ceiling=False, at_cap=np.nan, degen=np.nan, cap_mismatch=False, caps="", item_sets="",
                        escalation="", anchors="", anchor_abs=np.nan)).iloc[0]
                    have = sorted(g.seed, key=_natural)
                    short = [s for s, k in zip(g.seed, g.k) if int(k) < len(evals)]
                    rec.update(y=float(pooled.y), lo=float(pooled.lo), hi=float(pooled.hi), n_seeds=int(pooled.n_seeds), seeds=",".join(have),
                               k=int(g.k.min()), partial=bool(short) or have != want, open=bool(pooled.open), spread=float(pooled.spread),
                               n=float(g.n.sum(min_count=1)))
                    why = (pend_txt if pend_txt else
                           f"its readings cover seeds {', '.join(have)} of {', '.join(want)}" if have != want else
                           f"seed {', '.join(sorted(short, key=_natural))} covers fewer evals than the category's {len(evals)}" if short else "")
                    rec["why"] = why
                    rec["drawn"] = not why or mean_rule == "ring"
                    rec["ring"] = bool(why) and mean_rule == "ring"
                else:
                    rec["why"] = pend_txt or "no reading in the data yet"
                out.append(rec)
    return pd.DataFrame(out, columns=["object", "condition", "category", "y", "lo", "hi", "n_seeds", "seeds", "expected_seeds", "k", "n_evals",
                                      "partial", "open", "spread", "n", "drawn", "ring", "why"])


def not_drawn(rows, categories=None, mean_rule: str = "complete", labels: dict | None = None, condition_labels: dict | None = None) -> list:
    """What the figure leaves out or rings, one phrase per (object, condition[, category]) for the line printed under it, as
    `registry_nb.ctlstop_not_drawn` does for the comparison figure: every mean `paired_means` does not draw or draws ringed, with its
    reason (pending readings are named in it). Empty when the figure draws every mark plainly."""
    m = paired_means(rows, categories, mean_rule)
    lab = {**R.LABEL, **(labels or {})}
    many = m.category.nunique() > 1
    out = []
    for _, r in m[m.why.ne("")].iterrows():
        head = f"{lab.get(r.object, r.object)}, {(condition_labels or {}).get(r.condition, r.condition)}" + (f" ({r.category})" if many else "")
        out.append(f"{head}: {'ringed mean' if r.ring else 'no mean mark'}, {r.why}")
    return out


# ---- the figure --------------------------------------------------------------------------------------------------------------------
def _order(given, values, what: str) -> list:
    present = _appearance(values)
    if given is None:
        return present
    given = [str(v) for v in given]
    extra = [v for v in present if v not in given]
    if extra:
        raise ValueError(f"rows carry {what} {extra} that the {what} order {given} leaves out")
    return given


def _object_order(objects, values) -> list:
    if objects is not None:
        return _order(objects, values, "objects")
    present = _appearance(values)
    rank = {o: i for i, o in enumerate(R.OBJECTS)}
    return sorted(present, key=lambda o: (rank.get(o, len(rank)), present.index(o)))


def _layout(layout, cats: dict) -> list:
    if layout is None or layout == "row":
        rows_ = [list(cats)]
    elif layout == "column":
        rows_ = [[c] for c in cats]
    else:
        rows_ = [[str(c) for c in row] for row in layout]
    named = [c for row in rows_ for c in row]
    if sorted(named) != sorted(cats) or len(named) != len(set(named)):
        raise ValueError(f"layout {rows_} must name each category once: {list(cats)}")
    return rows_


def _cells(grid: list, span: dict, merge: float = 0.5) -> tuple:
    """ONE x SCALE FOR EVERY PANEL: ({category: (first, last + 1) grid column}, the columns' width ratios). Each panel's x range is its
    groups plus the comparison figure's 0.2 margin (xlim -0.6 to n - 0.4), laid left to right along its row; the grid's columns are cut at
    every panel edge of every row, so a panel's width is proportional to its x range and a group is equally wide in every panel and row.
    Edges closer than `merge` x units are merged into one cut (a sliver column cannot hold an axis margin under constrained layout)."""
    edges, ends = set(), {}
    for row in grid:
        x = 0.0
        for c in row:
            ends[c] = (round(x, 3), round(x + span[c] + 0.2, 3))
            edges |= set(ends[c])
            x += span[c] + 0.2
    cuts = []
    for e in sorted(edges):
        if not cuts or e - cuts[-1] >= merge:
            cuts.append(e)
    near = lambda v: min(range(len(cuts)), key=lambda k: abs(cuts[k] - v))
    cells = {c: (near(a), near(b)) for c, (a, b) in ends.items()}
    return cells, [cuts[k + 1] - cuts[k] for k in range(len(cuts) - 1)]


def _condition_handle(marker: str, label: str) -> Line2D:
    """The comparison figure's shape key: the condition's marker in grey with a dark rim."""
    return Line2D([], [], ls="", marker=marker, ms=10, markerfacecolor=R.MUTED, markeredgecolor=R.INK, markeredgewidth=1.4, label=label)


def paired_comparison_figure(rows, name: str | None = None, *, categories=None, conditions=None, objects=None, layout=None,
                             colors: dict | None = None, labels: dict | None = None, condition_labels: dict | None = None,
                             markers: dict | None = None, eval_labels: dict | None = None, titles: dict | None = None,
                             mean_rule: str = "complete", panel_objects: str = "present", sharey: bool | None = None,
                             unit: float | None = None, height: float = 4.2, seed_half: float = 0.33, pair_half: float | None = None,
                             mean_pair_half: float | None = None, max_width: float | None = MODEL_WIDTH, pad_ylim: tuple | None = (-5.0, 10.0),
                             pending_label: str | None = None, ylabel_wrap: int | None = None,
                             legend_extra=(), objects_per_column: int = 2, out_dirs=None, extra_dirs=()):
    """The paired comparison figure for the tidy `rows` (see the module docstring), saved as `name` when one is given (`save_figure`
    into `out_dirs`, default `R.FIGS`, plus the mirrors), and returned.

    `conditions` orders the conditions (default: order of appearance; at most as many as there are shapes); `markers` maps a condition to
    a matplotlib marker (default circle, diamond, square in condition order); `condition_labels` names them in the legend. `objects` orders
    the objects (default: the paper's `R.OBJECTS` order, then order of appearance); `colors` / `labels` add to `R.COLOR` / `R.LABEL`.
    `layout` is "row" (default: the categories side by side), "column" (stacked), or rows of category names; all panels share one x
    scale, and `sharey` (default: within a row, when the panels share a y label) shares a row's y axis. `panel_objects="present"` gives a
    panel slots only for the objects it has rows for, "all" gives every panel every object. `eval_labels` overrides `R.tick`; `titles`
    the category titles. `unit` is inches per eval group; by default the comparison figure's 5.0 for its 24 marks per group, scaled to
    the marks the fullest group holds (at least 1.6), then narrowed so the canvas is at most `max_width` inches wide (default the comparison figure's own 16.4, so the text prints at the
    size it has in the figures beside it; None: no cap). Where
    the group is narrower than that natural width the marks shrink with it (area x unit / natural unit), so a dense figure thins its marks
    rather than overlapping them; the text keeps its size. `seed_half` / `pair_half` / `mean_pair_half` the fan of seeds, the spacing of
    a seed's conditions and of the mean marks (defaults by condition count, the comparison figure's at two). `pad_ylim` keeps at least that y range in view (the
    comparison figure's -5 to 10 pp; None for other units). `ylabel_wrap` folds a y label to lines of that many characters (default
    from `height`, 25 at 4.2 in: a longer line overruns the panel and the tight bounding box crops it). `legend_extra` appends handles to
    the legend's last column.
    Every mark carries a gid `paired|<condition>|<object>|<seed or mean>|<category>|<eval or mean>` (a gap glyph `paired-pending|...`)."""
    t = tidy(rows, categories)
    cats = resolve_categories(t, categories)
    conds = _order(conditions, t.condition, "conditions")
    mk = dict(zip(conds, CONDITION_MARKERS))
    mk.update({str(k): v for k, v in (markers or {}).items()})
    if any(c not in mk for c in conds):
        raise ValueError(f"{len(conds)} conditions {conds} and only {len(CONDITION_MARKERS)} default shapes: pass markers= for the rest")
    if len(set(mk[c] for c in conds)) != len(conds):
        raise ValueError(f"two conditions share a shape: {[(c, mk[c]) for c in conds]}")
    objs = _object_order(objects, t.object)
    col = {**R.COLOR, **(colors or {})}
    nocol = [o for o in objs if o not in col]
    if nocol:
        raise ValueError(f"no colour for object(s) {nocol}: R.COLOR names the paper's objects; pass colors={{object: '#rrggbb'}}")
    lab = {**R.LABEL, **(labels or {})}
    clab = {c: (condition_labels or {}).get(c, c) for c in conds}
    if panel_objects not in ("present", "all"):
        raise ValueError(f"panel_objects={panel_objects!r}: 'present' or 'all'")
    grid = _layout(layout, cats)
    means = paired_means(t[[c for c in list(TIDY_COLUMNS) + list(OPTIONAL_COLUMNS) if c in t.columns]], categories=cats, mean_rule=mean_rule)
    seeds = {o: sorted(set(t.loc[t.object == o, "seed"]), key=_natural) for o in objs}
    slots = {c: ([o for o in objs if (t.object.eq(o) & t.category.eq(c)).any()] if panel_objects == "present" else list(objs)) for c in cats}
    nc = len(conds)
    ph = PAIR_HALF.get(nc, 0.30) if pair_half is None else pair_half
    mph = MEAN_PAIR_HALF.get(nc, 0.22) if mean_pair_half is None else mean_pair_half
    span = {c: len(cats[c][0]) + 1 for c in cats}
    width_cols = max(sum(span[c] for c in row) for row in grid)
    side = 1.4 * max(len(r) for r in grid)                   # the y axis room per panel, as the comparison figure's 1.4 in
    marks = max(len(slots[c]) * max(len(seeds[o]) for o in slots[c]) * nc for c in cats)
    natural = 5.0 * marks / 24.0                             # the comparison figure: 5.0 in for 4 objects x 3 seeds x 2 conditions
    if unit is None:
        unit = max(natural, 1.6)
        if max_width is not None:
            if max_width <= side:
                raise ValueError(f"max_width={max_width} leaves no room for {width_cols} eval groups beside the y axes ({side:.1f} in)")
            unit = min(unit, (max_width - side) / width_cols)
    unit = float(unit)
    scale = min(1.0, unit / natural)
    wrap = int(ylabel_wrap) if ylabel_wrap is not None else max(12, int((height - 0.6) * 7.2))   # ~10 pt a character at the 20 pt label
    shared = {}
    for row in grid:
        same = len({cats[c][2] for c in row}) == 1
        shared[tuple(row)] = (same if sharey is None else bool(sharey)) and len(row) > 1

    with style_context():
        fig = plt.figure(figsize=(width_cols * unit + side, height * len(grid)), layout="constrained")
        cells, ratios = _cells(grid, span)
        gs = fig.add_gridspec(len(grid), len(ratios), width_ratios=ratios)
        axes, flags = {}, dict(open=False, ring=False, pending=False)
        for i, row in enumerate(grid):
            first = None
            for cat in row:
                a, b = cells[cat]
                ax = fig.add_subplot(gs[i, a:b], sharey=first if (first is not None and shared[tuple(row)]) else None)
                first = first or ax
                axes[cat] = ax
                _draw_panel(ax, t[t.category == cat], means[means.category == cat], cat, cats[cat], slots[cat], seeds, conds, mk, col,
                            seed_half, ph, mph, eval_labels or {}, flags, scale)
                ax.set_title((titles or {}).get(cat, cat))
                if ax is first or not shared[tuple(row)]:
                    ax.set_ylabel(R._wrap_label(cats[cat][2], wrap))
                else:
                    ax.tick_params(labelleft=False)
        if pad_ylim is not None:                               # after every panel is drawn, so a shared axis is set once on all its data
            for row in grid:
                for c in (row[:1] if shared[tuple(row)] else row):
                    lo_, hi_ = axes[c].get_ylim()
                    axes[c].set_ylim(min(lo_, pad_ylim[0]), max(hi_, pad_ylim[1]))
        drawn = [o for o in objs if any(o in slots[c] for c in cats)]
        groups = ([[Patch(facecolor=col[o], edgecolor="none", label=lab.get(o, o)) for o in drawn[k:k + objects_per_column]]
                   for k in range(0, len(drawn), objects_per_column)]
                  + [[_condition_handle(mk[c], clab[c]) for c in conds],
                     [R.CI_HANDLE]
                     + ([Line2D([], [], ls="", marker="x", ms=9, color=R.MUTED, markeredgewidth=1.6,
                                label=pending_label or R.CTLSTOP_PENDING_TEXT)] if flags["pending"] else [])
                     + ([R.OPEN_HANDLE] if flags["open"] else []) + ([RING_HANDLE] if flags["ring"] else []) + list(legend_extra)])
        nrow = max(len(g) for g in groups)
        handles = [h for g in groups for h in g + [Line2D([], [], ls="", label=" ") for _ in range(nrow - len(g))]]
        R.legend_stack(fig, [(None, handles, len(groups))])
    if name:
        save_figure(fig, name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig


def _draw_panel(ax, t, means, cat, spec, slots, seeds, conds, mk, col, seed_half, ph, mph, eval_labels, flags, scale=1.0):
    """One category panel: the per-seed marks of every slot per eval, the mean group, the separators, the zero line and the ticks."""
    evals = spec[0]
    n = len(evals)
    step = 0.82 / max(len(slots), 1)
    offs = np.linspace(-1.0, 1.0, len(conds)) if len(conds) > 1 else np.zeros(1)
    by = {(r.object, r.seed, r["eval"], r.condition): r for _, r in t.iterrows()}

    def mark(x, y, lo, hi, o, c, size, bar, gid, open_=False, ring=False):
        if pd.notna(lo) and pd.notna(hi):
            ax.plot([x, x], [lo, hi], color=col[o], lw=bar, alpha=0.85, solid_capstyle="butt", zorder=2)
        shape = dict(s=size * MARKER_AREA.get(mk[c], 1.0), marker=mk[c])
        style = (dict(facecolors="white", edgecolors=col[o], linewidths=1.6) if open_ else dict(color=col[o], edgecolors="white", linewidths=0.6))
        ax.scatter([x], [y], **shape, zorder=4, gid=gid, **style)
        if ring:
            ax.scatter([x], [y], **shape, facecolors="none", edgecolors="black", linewidths=1.4, zorder=5)

    def gap_glyph(x, gid, s):
        ax.scatter([x], [0.0], zorder=3, gid=gid, **{**R.CTLSTOP_PENDING_MARK, "s": s})
        flags["pending"] = True

    def join(pts, o, lw):
        """The thin line in the object's colour through one seed's (or the mean's) drawn marks, in condition order."""
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col[o], lw=lw, alpha=0.9, zorder=3)

    for i, e in enumerate(evals):
        for j, o in enumerate(slots):
            ns = len(seeds[o])
            gap = seed_half * step * 2 / max(ns - 1, 1)
            for xs, sd in zip(R._spread(i - 0.41 + step * (j + 0.5), ns, seed_half * step), seeds[o]):
                pts = []
                for c, dx in zip(conds, offs * ph * gap):
                    r = by.get((o, sd, e, c))
                    if r is None:
                        continue
                    gid = f"{cat}|{e}"
                    if r.pending:
                        gap_glyph(xs + dx, f"paired-pending|{c}|{o}|{sd}|{gid}", SEED_MARK * scale)
                        continue
                    flags["open"] |= bool(r.open)
                    mark(xs + dx, r.y, r.lo, r.hi, o, c, SEED_MARK * scale, SEED_BAR, f"paired|{c}|{o}|{sd}|{gid}", open_=bool(r.open))
                    pts.append((xs + dx, r.y))
                join(pts, o, 1.0)
    for j, o in enumerate(slots):
        xc, pts = n - 0.41 + step * (j + 0.5), []
        for c, dx in zip(conds, offs * mph * step):
            m = means[(means.object == o) & (means.condition == c)]
            if not len(m):
                continue
            m = m.iloc[0]
            gid = f"{c}|{o}|mean|{cat}|{R.MEAN_COLUMN}"
            if m.drawn:
                flags["open"] |= bool(m.open)
                flags["ring"] |= bool(m.ring)
                mark(xc + dx, m.y, m.lo, m.hi, o, c, MEAN_MARK * scale, MEAN_BAR, f"paired|{gid}", open_=bool(m.open), ring=bool(m.ring))
                pts.append((xc + dx, m.y))
            else:
                gap_glyph(xc + dx, f"paired-pending|{gid}", 90 * scale)
        join(pts, o, 1.6)
    for i in range(1, n):
        ax.axvline(i - 0.5, color=R.RULE, lw=1.0)
    ax.axvline(n - 0.5, color=R.INK2, lw=1.3)                 # the separator before the mean group
    ax.axhline(0, color=R.INK2, lw=1.1, zorder=1)
    ticks = [eval_labels.get(e) or R.tick(e) for e in evals] + [R.MEAN_COLUMN]
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(-0.6, len(ticks) - 0.4)
