"""The comparison half of `judge_compare.py`: read two renders (the filed judges and a judge policy), compare every LLM-judged paper
figure of the notebook's figure map (`judge_compare.figure_map`), draw the side-by-side and difference figures, and write the numbers a
report quotes (`compare.json`, `compare.md`).

Units, all in each figure's own orientation (a delta against the unsteered model, up = better: defense success, benign compliance,
reduction in misalignment):
  cell    (model, object, seed, eval) after the figure's own decode-draw pooling (`registry_nb.pool_draws`)
  eval    (model, object, eval): the seeds pooled as the appendix's per-eval marks are (`registry_nb.pool_cell`: item-clustered across
          the seeds, the between-fit variance and the anchor once);
          the per-eval columns only, never a category's pooled column (`registry_nb.POOLED_EVAL`, Petri's pooled score), which is the
          category's mean column and is read once, there
  mean    (model, object, category): `registry_nb.pool_seeds` over `category_means`, the large mark after the separator
A difference is QUALITATIVE when the ordering of the objects within one (model, column) changes (Spearman, the best and the worst object,
a pair that swaps with the two intervals clear of each other under either judge), when an interval's relation to zero changes (a
reduction or an increase significant under one judge and not under the other), or when a sign flips with both intervals clear of
zero; a common offset of every value is a LEVEL shift, reported as the mean change and the slope of the policy's values on the filed
ones. The mean-column verdict changes are also counted once per distinct mark (`verdict_change_totals`): the main figures' Qwen marks
are the appendix figures' Qwen marks, so a change listed under Fig. 4 and Fig. 12 is one change. Each carries how far the significant
judge's interval clears zero, in the category's unit.

Which frames a figure is compared from is read off its frames' columns: a delta frame (eval_id, d_y) with its category means (category,
seed, y), the transfer gaps (category marks without seeds, per-eval `diff`), best-of-N curve points (n, asr), frontier points (dx, dy) and
the padding ladder (pad, delta). A figure whose frames a render did not dump is marked "not compared: frame not dumped"; a figure drawn
from judged rows whose frames have none of these shapes refuses (judge_compare.FigureMapError).
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from itertools import zip_longest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# NO SOURCE IS FORCED HERE: the comparison reads the two renders' frames only, and takes from the notebook module its labels, categories
# and pooling rules, which no registry source changes; the module resolves its own source (REGISTRY_NB_SOURCE, default auto). A render
# that needs the live registry gets it from `judge_compare.render_env`, which sets REGISTRY_NB_SOURCE for the render's kernel.
import registry_nb as R                                         # noqa: E402
import judge_compare as JC                                       # noqa: E402

A_COL, B_COL = "#2a78d6", "#eb6834"                             # categorical slots 1 and 2 (dataviz reference palette, light)
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4"
MISSING = object()
MARK_COLUMNS = ["model_id", "object_label", "column", "y", "lo", "hi"]


def label(o: str) -> str:
    return R.LABEL.get(o, o)


def load(renders: str, name: str):
    import nbformat
    p = os.path.join(renders, name)
    if not os.path.exists(os.path.join(p, "frames.pkl")):
        return None
    with open(os.path.join(p, "frames.pkl"), "rb") as f:
        fr = pickle.load(f)
    heads = {}
    for k, v in fr.items():
        if isinstance(v, pd.DataFrame):
            if v.attrs.get("head") is not None:
                heads[k] = v.attrs.get("head")
            v.attrs = {}                     # the notebook hangs frames on attrs; pandas compares attrs on every merge
    fr["_attr_heads"] = heads
    R.restore_pooled_ci_state(fr.get("POOLED_CI_STATE"))          # the render's units, so its dots pool here as they did there
    nb = nbformat.read(os.path.join(p, "registry_qwen_figures.executed.ipynb"), as_version=4)
    fr["tables"] = JC.all_tables(nb)
    fr["nb"] = nb
    fr["dir"] = p
    fr["name"] = name
    rj = os.path.join(p, "render.json")
    fr["render_json"] = json.load(open(rj)) if os.path.exists(rj) else None
    return fr


# ---- units ------------------------------------------------------------------------------------------------------------------------
def _cat(cat: str):
    return R.APPENDIX_CATEGORIES.get(cat) or R.CATEGORIES.get(cat)


def _sign(cat: str) -> float:
    return float(_cat(cat)[1]) if _cat(cat) else 1.0


def _cat_evals(cat: str, pooled: bool = True) -> list:
    """A category's eval ids, with its pooled column (Petri's pooled score) when `pooled`."""
    if not _cat(cat):
        return []
    return list(_cat(cat)[0]) + ([R.POOLED_EVAL[cat]] if pooled and cat in R.POOLED_EVAL else [])


def unit(cat: str) -> str:
    lab = str(_cat(cat)[2]) if _cat(cat) else ""
    return "pp" if "(pp)" in lab else (lab or "?")


def eval_marks(d: pd.DataFrame, cat: str, models=None) -> pd.DataFrame:
    """(model, object, eval) marks of a category's per-eval columns, oriented: the appendix's mark for the cell (`registry_nb.pool_cell`,
    the render's rule: decode draws item-clustered, then the fit seeds item-clustered across each other with the between-fit variance and
    the anchor once). The category's pooled column (`registry_nb.POOLED_EVAL`) is not an eval mark: it is the category's mean column,
    read by `mean_marks` with the figure's own pooling, so it is counted once and with one interval."""
    evals = _cat_evals(cat, pooled=False)
    x = d[d.eval_id.isin(evals) & d.d_y.notna() & d.object_label.ne("none")]
    if models:
        x = x[x.model_id.isin(models)]
    if not len(x):
        return pd.DataFrame(columns=["model_id", "object_label", "column", "y", "lo", "hi", "n_seeds", "y_abs", "anchor"])
    p = R.pool_cell(x, y="d_y", lo="d_lo", hi="d_hi")
    fits = R.pool_draws(x, item_sets_apart=False).groupby(["model_id", "object_label", "eval_id"]).agg(y_abs=("y", "mean"), anchor=("anchor_value", "mean"))
    s = _sign(cat)
    rows = []
    for r in p.itertuples():
        f = fits.loc[(r.model_id, r.object_label, r.eval_id)]
        y, lo, hi = s * float(r.d_y), min(s * float(r.d_lo), s * float(r.d_hi)), max(s * float(r.d_lo), s * float(r.d_hi))
        rows.append(dict(model_id=r.model_id, object_label=r.object_label, column=r.eval_id, y=y, lo=lo, hi=hi, n_seeds=len(r.seed_parts),
                         y_abs=float(f.y_abs), anchor=float(f.anchor)))
    return pd.DataFrame(rows)


def mean_marks(cm: pd.DataFrame, cat: str, models=None) -> pd.DataFrame:
    x = cm[cm.category.eq(cat) & cm.object_label.ne("none")]
    if models:
        x = x[x.model_id.isin(models)]
    if not len(x):
        return pd.DataFrame(columns=["model_id", "object_label", "column", "y", "lo", "hi", "n_seeds", "anchor"])
    p = R.pool_seeds(x)
    anc = x.groupby(["model_id", "object_label"]).anchor_abs.mean()
    return pd.DataFrame(dict(model_id=p.model_id, object_label=p.object_label, column="mean", y=p.y, lo=p.lo, hi=p.hi, n_seeds=p.n_seeds,
                             anchor=[anc.get((m, o), np.nan) for m, o in zip(p.model_id, p.object_label)]))


def verdict(lo, hi) -> str:
    if lo != lo or hi != hi:
        return "?"
    return "+" if lo > 0 else ("-" if hi < 0 else "0")


def clearance(lo, hi) -> float:
    """How far an interval clears zero (its nearest bound's distance from zero); NaN when it straddles zero."""
    return float(lo) if lo > 0 else (float(-hi) if hi < 0 else np.nan)


def compare_marks(a: pd.DataFrame, b: pd.DataFrame, keys=("model_id", "column")) -> dict:
    """Per (model, column) group: Spearman of the objects' values, the best / worst object under each judge, the pairs whose order flips
    (and whether the flip is robust: the two intervals clear of each other under either judge), the verdict changes (interval vs
    zero, with how far the significant judge's interval clears zero) and sign flips, the level shift (mean b - a) and the slope of b
    on a."""
    from scipy.stats import spearmanr
    j = a.merge(b, on=list(keys) + ["object_label"], suffixes=("_a", "_b"), how="outer")
    out = dict(groups=[], verdict_changes=[], sign_flips=[], robust_swaps=[], n_marks=int(len(j)),
               n_both=int((j.y_a.notna() & j.y_b.notna()).sum()))
    both = j[j.y_a.notna() & j.y_b.notna()]
    if len(both) > 1:
        out["level_shift_mean"] = float((both.y_b - both.y_a).mean())
        out["level_shift_sd"] = float((both.y_b - both.y_a).std(ddof=1))
        out["slope_b_on_a"] = float(np.polyfit(both.y_a, both.y_b, 1)[0]) if both.y_a.nunique() > 1 else np.nan
        out["pearson"] = float(np.corrcoef(both.y_a, both.y_b)[0, 1]) if both.y_a.nunique() > 1 else np.nan
    for r in both.itertuples():
        va, vb = verdict(r.lo_a, r.hi_a), verdict(r.lo_b, r.hi_b)
        if va != vb:
            near = min(abs(x) for x in (r.lo_a, r.hi_a, r.lo_b, r.hi_b) if x == x)
            sig = [s for s, v in (("a", va), ("b", vb)) if v in "+-"]
            clear = [c for c in ((clearance(r.lo_a, r.hi_a) if "a" in sig else np.nan), (clearance(r.lo_b, r.hi_b) if "b" in sig else np.nan)) if c == c]
            out["verdict_changes"].append(dict(model=r.model_id, column=r.column, object=r.object_label, a=f"{r.y_a:+.2f} [{r.lo_a:+.2f}, {r.hi_a:+.2f}]",
                                               b=f"{r.y_b:+.2f} [{r.lo_b:+.2f}, {r.hi_b:+.2f}]", verdict_a=va, verdict_b=vb,
                                               nearest_bound_to_zero=float(near), significant="+".join(sig) or "?",
                                               clearance=float(min(clear)) if clear else np.nan,
                                               y_a=float(r.y_a), lo_a=float(r.lo_a), hi_a=float(r.hi_a),
                                               y_b=float(r.y_b), lo_b=float(r.lo_b), hi_b=float(r.hi_b)))
        if np.sign(r.y_a) != np.sign(r.y_b) and va != "0" and vb != "0":
            out["sign_flips"].append(dict(model=r.model_id, column=r.column, object=r.object_label, a=r.y_a, b=r.y_b))
    for gk, g in both.groupby(list(keys)):
        g = g.sort_values("y_a", ascending=False)
        rec = dict(zip(keys, gk if isinstance(gk, tuple) else (gk,)))
        rec.update(n=len(g), best_a=g.loc[g.y_a.idxmax(), "object_label"], best_b=g.loc[g.y_b.idxmax(), "object_label"],
                   worst_a=g.loc[g.y_a.idxmin(), "object_label"], worst_b=g.loc[g.y_b.idxmin(), "object_label"],
                   mean_shift=float((g.y_b - g.y_a).mean()))
        rec["spearman"] = float(spearmanr(g.y_a, g.y_b).statistic) if len(g) > 2 and g.y_a.nunique() > 1 and g.y_b.nunique() > 1 else np.nan
        swaps = []
        rows = list(g.itertuples())
        for i in range(len(rows)):
            for k in range(i + 1, len(rows)):
                p, q = rows[i], rows[k]
                if (p.y_a - q.y_a) * (p.y_b - q.y_b) < 0:
                    clear_a = p.lo_a > q.hi_a or q.lo_a > p.hi_a
                    clear_b = p.lo_b > q.hi_b or q.lo_b > p.hi_b
                    swaps.append(dict(pair=f"{p.object_label} / {q.object_label}", a=f"{p.y_a:+.2f} vs {q.y_a:+.2f}", b=f"{p.y_b:+.2f} vs {q.y_b:+.2f}",
                                      robust=bool(clear_a or clear_b)))
        rec["swaps"] = len(swaps)
        rec["robust_swaps"] = sum(s["robust"] for s in swaps)
        out["robust_swaps"] += [dict(rec_key=str(gk), **s) for s in swaps if s["robust"]]
        out["groups"].append(rec)
    out["only_a"] = [dict(model=r.model_id, column=r.column, object=r.object_label) for r in j[j.y_a.notna() & j.y_b.isna()].itertuples()]
    out["only_b"] = [dict(model=r.model_id, column=r.column, object=r.object_label) for r in j[j.y_b.notna() & j.y_a.isna()].itertuples()]
    return out


def anchors(d: pd.DataFrame, cat: str) -> pd.DataFrame:
    """The unsteered readings (absolute, plotted units) behind a category's columns, one per (model, eval)."""
    x = d[d.eval_id.isin(_cat_evals(cat)) & d.anchor_value.notna()]
    return x.groupby(["model_id", "eval_id"]).anchor_value.mean().rename("anchor").reset_index()


# ---- which frames a figure is drawn from --------------------------------------------------------------------------------------------
def shape(v) -> str:
    """What a frame holds, read off its columns."""
    if not isinstance(v, pd.DataFrame):
        return "aux"
    c = set(v.columns)
    if {"eval_id", "d_y", "d_lo", "d_hi", "object_label"} <= c:
        return "rows"
    if {"category", "object_label", "y", "lo", "hi", "seed"} <= c:
        return "means"
    if {"category", "object_label", "y", "lo", "hi"} <= c:
        return "category_marks"
    if {"eval_id", "object_label", "diff", "lo", "hi"} <= c:
        return "eval_diffs"
    if {"n", "asr", "object_label"} <= c:
        return "bon"
    if {"dx", "dy", "object_label"} <= c:
        return "frontier"
    if {"pad", "delta", "lo", "hi", "object_label"} <= c:
        return "padding"
    return "aux"


def _rubrics(fr: dict, n: str):
    """The judged rubrics a frame of one render carries: a sorted list, None for a value with no rubric column, MISSING when the render
    did not dump it."""
    fr_rub = fr.get("FRAME_RUBRICS")
    if isinstance(fr_rub, dict) and n in fr_rub:
        return fr_rub[n]
    if n in fr:
        v = fr[n]
        if isinstance(v, pd.DataFrame) and "judge_rubric_name" in v.columns:
            return sorted(str(x) for x in v["judge_rubric_name"].dropna().unique())
        return None
    return MISSING


def judged_evals(A: dict, B: dict) -> set:
    """The evals any frame of either render reads from a judged rubric (`RUBRIC_FAMILY`): the evals an LLM judge scores."""
    out = set()
    for fr in (A, B):
        for v in fr.values():
            if isinstance(v, pd.DataFrame) and {"judge_rubric_name", "eval_id"} <= set(v.columns):
                out |= set(v.loc[v.judge_rubric_name.isin(list(R.RUBRIC_FAMILY)), "eval_id"].dropna())
    return out


def _frame_evals(name: str, A: dict, B: dict) -> set:
    return set().union(*[set(fr[name].eval_id.dropna()) for fr in (A, B)
                         if isinstance(fr.get(name), pd.DataFrame) and "eval_id" in fr[name].columns] or [set()])


def judged_status(frames: list, A: dict, B: dict, parents: dict, jev: set | None = None) -> dict:
    """Whether a figure is LLM-judged: from each frame it draws, follow the generator's assignments upstream until a frame carries a
    rubric column (the rows it was drawn from); the figure is judged when one of those carries a rubric of `RUBRIC_FAMILY`, or holds rows
    of an eval the judges score (`jev`; a sidecar's rows name their judge but no rubric). Both renders are read (a held row reads NaN
    under one judge and still names its rubric under the other). `judged` is True, False, or None when a frame on the way was not
    dumped and no judged one was found."""
    def both(n):
        ra, rb = _rubrics(A, n), _rubrics(B, n)
        vals = [r for r in (ra, rb) if r is not MISSING]
        if not vals:
            return MISSING
        if all(v is None for v in vals):
            return None
        return sorted(set().union(*[v for v in vals if v is not None]))
    seen, todo, found, missing = set(), list(frames), {}, []
    while todo:
        nxt = []
        for n in todo:
            if n in seen:
                continue
            seen.add(n)
            r = both(n)
            if r is MISSING:
                missing.append(n)
            elif r is None:
                nxt += parents.get(n, [])
            else:
                found[n] = r
        todo = nxt
    judged = any(x in R.RUBRIC_FAMILY for rs in found.values() for x in rs) or any(_frame_evals(n, A, B) & (jev or set()) for n in found)
    return dict(judged=True if judged else (None if missing else False), rows_frames=found, not_dumped=missing)


# ---- figures ----------------------------------------------------------------------------------------------------------------------
def stack_pngs(paths: list, labels: list, out: str) -> bool:
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
    if len(ims) != len(paths):
        return False
    w = max(i.width for i in ims)
    band = 60
    canvas = Image.new("RGB", (w, sum(i.height + band for i in ims)), "white")
    y = 0
    for im, lab in zip(ims, labels):
        dr = ImageDraw.Draw(canvas)
        dr.text((20, y + 15), lab, fill=(11, 11, 11), font_size=36) if hasattr(dr, "text") else None
        canvas.paste(im, (0, y + band))
        y += im.height + band
    canvas.save(out)
    return True


def difference_figure(ma: pd.DataFrame, mb: pd.DataFrame, title: str, out: str, la: str, lb: str, models=None) -> None:
    """Each object's mark under both judges with its interval, one panel per (model, column)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = ma.merge(mb, on=["model_id", "column", "object_label"], suffixes=("_a", "_b"), how="outer")
    if not len(j):
        return
    groups = [(m, c) for (m, c), _ in j.groupby(["model_id", "column"], sort=False)]
    order_m = [m for m in R.MODEL_ORDER if m in {g[0] for g in groups}]
    cols = list(dict.fromkeys(c for _, c in groups))
    cols = [c for c in cols if c != "mean"] + [c for c in cols if c == "mean"]          # the mean column last, as the figures draw it
    fig, axes = plt.subplots(len(order_m), len(cols), figsize=(2.4 * len(cols) + 1.5, 2.9 * len(order_m) + 0.6), squeeze=False, sharey="row")
    objs = [o for o in R.OBJECTS if o in set(j.object_label)] + sorted(set(j.object_label) - set(R.OBJECTS))
    for ri, m in enumerate(order_m):
        for ci, c in enumerate(cols):
            ax = axes[ri][ci]
            g = j[(j.model_id == m) & (j.column == c)].set_index("object_label").reindex(objs)
            yy = np.arange(len(objs))
            for side, col, off in (("a", A_COL, -0.17), ("b", B_COL, 0.17)):
                v, lo, hi = g[f"y_{side}"], g[f"lo_{side}"], g[f"hi_{side}"]
                ax.errorbar(v, yy + off, xerr=[(v - lo).clip(lower=0), (hi - v).clip(lower=0)], fmt="o", ms=4, color=col, lw=1.1, capsize=0)
            ax.axvline(0, color=RULE, lw=1, zorder=0)
            ax.set_yticks(yy)
            ax.set_yticklabels([label(o) for o in objs], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{R.MODEL_LABEL.get(m, m)} · {R.tick(c).replace(chr(10), ' ')}", fontsize=9, color=INK)
            ax.tick_params(axis="x", labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    top = 1.0 - 0.75 / fig.get_figheight()
    fig.suptitle(title, fontsize=11, color=INK, y=0.998, va="top")
    fig.legend([Line2D([], [], marker="o", color=A_COL, ls=""), Line2D([], [], marker="o", color=B_COL, ls="")], [la, lb],
               loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.32 / fig.get_figheight()), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out + ".png", dpi=130)
    fig.savefig(out + ".pdf")
    plt.close(fig)


# ---- one figure ---------------------------------------------------------------------------------------------------------------------
def _delta_pair(rows: str | None, means: str | None, layout, A: dict, B: dict, title: str, out_stem: str, la: str, lb: str,
                jev: set) -> dict:
    """A delta frame and its category means, compared per category: the figure's layout's categories (all, without a layout) on whose
    evals the frame holds rows the judges score (`jev`)."""
    judged = (_frame_evals(rows, A, B) & jev) if rows else set()
    pool = list(layout) if layout else list(R.APPENDIX_CATEGORIES)
    cats, skipped = [], {}
    for c in pool:
        if not _cat(c):
            skipped[c] = "not an eval category (a panel hook)"
        elif rows and not set(_cat_evals(c)) & judged:
            skipped[c] = "no LLM-judged rows in the frame"
        else:
            cats.append(c)
    if not rows and means:                                   # means alone: the categories the frame holds
        held = set(A[means].category) | set(B[means].category)
        cats = [c for c in cats if c in held]
    res = dict(rows_frame=rows, means_frame=means, categories={}, categories_skipped={k: v for k, v in skipped.items() if layout})
    for cat in cats:
        empty = pd.DataFrame(columns=MARK_COLUMNS)
        ea, eb = (eval_marks(A[rows], cat), eval_marks(B[rows], cat)) if rows else (empty, empty)
        ma, mb = (mean_marks(A[means], cat), mean_marks(B[means], cat)) if means else (empty, empty)
        rec = dict(unit=unit(cat), evals=compare_marks(ea, eb), mean=compare_marks(ma, mb),
                   marks=pd.concat([ma.assign(judge="a"), mb.assign(judge="b")]).round(4).to_dict("records"))
        if rows:
            rec["anchors"] = anchors(A[rows], cat).merge(anchors(B[rows], cat), on=["model_id", "eval_id"], how="outer",
                                                         suffixes=("_a", "_b")).round(3).to_dict("records")
        res["categories"][cat] = rec
        difference_figure(pd.concat([ea, ma]), pd.concat([eb, mb]), f"{title}, {cat}: {la} vs {lb}",
                          f"{out_stem}__{cat.replace(' ', '_')}__difference", la, lb)
    return res


def _category_marks(p: pd.DataFrame) -> pd.DataFrame:
    """Category marks without seeds (the transfer gaps), one column per category."""
    return p.rename(columns={"category": "column"})[MARK_COLUMNS]


def _eval_diffs(p: pd.DataFrame) -> pd.DataFrame:
    """Per-eval differences with their intervals (the transfer gaps per eval), one column per eval."""
    return p.rename(columns={"eval_id": "column", "diff": "y"})[MARK_COLUMNS]


def _bon(p: pd.DataFrame) -> pd.DataFrame:
    g = p.dropna(subset=["n"]).groupby(["model_id", "object_label", "n"]).asr.mean().reset_index()
    g = g.assign(column=g.n.astype(int).map(lambda n: f"N={n}"), y=100 * (1 - g.asr), lo=np.nan, hi=np.nan)
    return g[MARK_COLUMNS]


def _padding(p: pd.DataFrame) -> pd.DataFrame:
    seed = p["seed"].fillna("").astype(str) if "seed" in p.columns else pd.Series("", index=p.index)
    return pd.DataFrame(dict(model_id=p["model_id"] if "model_id" in p.columns else R.MODEL_ID,
                             object_label=p.object_label.astype(str) + np.where(seed.ne(""), " · " + seed, ""),
                             column=p.pad.map(lambda x: f"pad {int(x):,}" if x == x else "pad ?"), y=p.delta, lo=p.lo, hi=p.hi))


def _frontier(pa: pd.DataFrame, pb: pd.DataFrame) -> dict:
    key = [c for c in ("object_label", "dose", "seed", "lane") if c in pa.columns]
    ycol = next((c for c in ("dy", "y", "d_y") if c in pa.columns), None)
    xcol = next((c for c in ("dx", "x", "d_x") if c in pa.columns), None)
    j = pa.merge(pb, on=key, how="outer", suffixes=("_a", "_b"))
    return dict(rungs_a=int(len(pa)), rungs_b=int(len(pb)), key=key, ycol=ycol, xcol=xcol,
                rows=j[[c for c in j.columns if c in key or c.split("_")[0] in (ycol, xcol)]].round(3).to_dict("records") if ycol else [])


def compare_figure(name: str, spec: dict, A: dict, B: dict, parents: dict, figs: str, la: str, lb: str, jev: set | None = None) -> dict:
    """One figure of the map: whether it is LLM-judged, and its comparison from the frames it draws."""
    frames = spec["frames"] + spec["hook_frames"]
    jev = judged_evals(A, B) if jev is None else jev
    st = judged_status(frames, A, B, parents, jev)
    rec = dict(llm_judged=st["judged"], rows_frames=st["rows_frames"])
    if st["judged"] is None:
        rec["lineage_not_dumped"] = st["not_dumped"]
    missing = [f for f in frames if f not in A or f not in B]
    if st["judged"] is False:
        rec.update(compared=False, not_compared="not LLM-judged: the rows it is drawn from carry no judged rubric "
                   f"({', '.join(f'{k}: {v or []}' for k, v in st['rows_frames'].items()) or 'no registry rows'})")
        return rec
    if missing:                    # every frame or none: a delta frame is paired with its means by position, which a gap would shift
        rec.update(compared=False, frames_not_dumped=missing,
                   not_compared=f"not compared: frame not dumped ({', '.join(missing)}); re-render with this DUMP cell")
        return rec
    kinds = {f: shape(A[f]) for f in frames}
    title = f"{spec['paper']} ({name})"
    stem = os.path.join(figs, name)
    rows = [f for f in spec["frames"] if kinds.get(f) == "rows"]
    means = [f for f in spec["frames"] if kinds.get(f) == "means"]
    pairs = list(zip_longest(rows, means))
    if len(pairs) == 1:
        rec.update(_delta_pair(pairs[0][0], pairs[0][1], spec["layout_categories"], A, B, title, stem, la, lb, jev))
    elif pairs:
        rec["panels"] = {(r or m): _delta_pair(r, m, spec["layout_categories"], A, B, f"{title} {r or m}", f"{stem}__{r or m}", la, lb, jev)
                         for r, m in pairs}
    for kind, key, fn in (("category_marks", "transfer", _category_marks), ("eval_diffs", "transfer_evals", _eval_diffs),
                          ("bon", "bon", _bon), ("padding", "padding", _padding)):
        fs = [f for f in frames if kinds.get(f) == kind]
        res = {f: compare_marks(fn(A[f]), fn(B[f])) for f in fs}
        if res:
            rec[key] = next(iter(res.values())) if len(res) == 1 else res
    fr = {f: _frontier(A[f], B[f]) for f in frames if kinds.get(f) == "frontier"}
    if fr:
        rec["frontier"] = fr
    compared = [f for f in frames if kinds[f] != "aux"]
    if not compared:
        if st["judged"]:
            raise JC.FigureMapError(f"{title}: it is drawn from judged rows ({', '.join(st['rows_frames'])}) but none of its frames "
                                    f"({', '.join(frames)}) has a shape the comparison reads")
        rec.update(compared=False, not_compared=f"not compared: none of its frames ({', '.join(frames)}) has a shape the comparison reads, "
                   f"and whether it is LLM-judged is unknown (not dumped: {', '.join(st['not_dumped'])})")
        return rec
    rec.update(compared=True, compared_frames=compared)
    return rec


# ---- totals and provenance ----------------------------------------------------------------------------------------------------------
def figure_label(rec: dict, name: str) -> str:
    """How a distinct mark names a figure that draws it: the README's paper figure ("Fig. 4"), or, for a figure the paper does not print
    (`Notebook only`, several such rows) or one with no paper number, its file name, so two notebook-only figures stay apart."""
    paper = rec.get("paper")
    return name if not paper or paper == JC.NOTEBOOK_ONLY else paper


def verdict_totals(figures: dict) -> dict:
    """The mean-column verdict changes over every figure, listed per figure (`entries`) and once per distinct mark (`distinct`): two
    entries are one mark when the model, category and object agree and both judges' readings are identical (a main figure's Qwen
    mark redrawn in an appendix figure), with the figures that draw it and how far the significant judge's interval clears zero."""
    entries, seen = [], {}
    for name, rec in figures.items():
        blocks = [(None, rec.get("categories") or {})] + [(p, v.get("categories") or {}) for p, v in (rec.get("panels") or {}).items()]
        for panel, cats in blocks:
            for cat, c in cats.items():
                for v in c["mean"]["verdict_changes"]:
                    e = dict(v, category=cat, unit=c.get("unit") or unit(cat), figure=name, paper=rec.get("paper"), panel=panel)
                    entries.append(e)
                    k = (v["model"], cat, v["object"]) + tuple(None if v[x] != v[x] else round(v[x], 6)
                                                               for x in ("y_a", "lo_a", "hi_a", "y_b", "lo_b", "hi_b"))
                    if k in seen:
                        seen[k]["figures"].append(figure_label(rec, name))
                    else:
                        seen[k] = dict({x: e[x] for x in ("model", "category", "object", "unit", "a", "b", "verdict_a", "verdict_b",
                                                           "significant", "clearance", "nearest_bound_to_zero")}, figures=[figure_label(rec, name)])
    distinct = list(seen.values())
    bands = {}
    for d in distinct:
        b = bands.setdefault(d["unit"], {"<= 1": 0, "1 to 3": 0, "> 3": 0, "?": 0})
        c = d["clearance"]
        b["?" if c != c else ("<= 1" if c <= 1 else ("1 to 3" if c <= 3 else "> 3"))] += 1
    by_cat = {}
    for d in distinct:
        by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
    return dict(unit_of_count="mean column (model, category, object)", entries=len(entries), distinct=len(distinct),
                distinct_by_category=by_cat, clearance_bands=bands,
                clearance_note="the distance from zero of the nearest bound of the significant judge's interval, in the category's unit",
                changes=sorted(distinct, key=lambda d: (d["category"], d["model"], d["object"])))


def render_record(fr: dict) -> dict:
    """What a render was drawn under: its judge policy, threshold and Petri judge, the frames' heads and the item-row head per judge,
    read defensively (a frames.pkl that predates a field says so rather than guessing)."""
    st = fr.get("SETTINGS") or {}
    meta = fr.get("render_json") or {}
    jframes = (fr.get("JUDGE_REPORT") or {}).get("frames") or []
    pframes = (fr.get("PETRI_JUDGE_REPORT") or {}).get("frames") or []
    thr, src = st.get("SR_THRESHOLD"), "the kernel's settings (registry_nb.SR_THRESHOLD)"
    if thr is None:
        ths = sorted({f.get("threshold") for f in jframes if f.get("threshold") is not None})
        thr, src = (ths[0] if len(ths) == 1 else (ths or None)), "the judge report"
    if thr is None:
        env = dict(meta.get("env") or {})
        thr = env.get("REGISTRY_NB_SR_THRESHOLD") or (meta.get("provenance") or {}).get("REGISTRY_NB_SR_THRESHOLD")
        src = "render.json"
    if thr is None:
        thr, src = "not recorded by this render", None
    names, heads, recorded = {}, {}, False
    for f in jframes:
        names.update(f.get("judges") or {})
        if "item_heads" in f:
            recorded = True
        for jid, h in (f.get("item_heads") or {}).items():
            heads.setdefault(jid, set()).add(h)
    older = [dict(judge=jid, judge_name=names.get(jid), item_head=h, frame_head=f.get("frame_head"))
             for f in jframes for jid, h in (f.get("item_heads") or {}).items()
             if h is not None and f.get("frame_head") is not None and h < f["frame_head"]]
    older += [dict(judge=f.get("judge"), judge_name=f.get("judge_model"), item_head=f.get("item_head"), frame_head=f.get("frame_head"))
              for f in pframes if f.get("item_head") is not None and f.get("frame_head") is not None and f["item_head"] < f["frame_head"]]
    if not jframes:
        item_heads = "none read: no judged reading was re-derived (the filed judges' rows)"
    elif not recorded:
        item_heads = "not recorded by this render (its frames.pkl predates JUDGE_REPORT item_heads)"
    else:
        item_heads = {jid: dict(judge=names.get(jid), heads=sorted(hs, key=lambda x: (x is None, x))) for jid, hs in heads.items()}
    return dict(name=fr.get("name"), judges=fr.get("JUDGES"), source=fr.get("SOURCE"), threshold=thr, threshold_from=src,
                petri_judge=st.get("PETRI_JUDGE", "not recorded by this render") or "the policy's", head=fr.get("HEAD"),
                frame_heads=dict(dumped_frames=fr.get("_attr_heads") or {},
                                 judge_report=[f.get("frame_head", "not recorded") for f in jframes],
                                 petri_judge_report=[f.get("frame_head", "not recorded") for f in pframes]),
                item_heads=item_heads,
                petri_item_heads=[dict(judge=f.get("judge"), judge_model=f.get("judge_model"), item_head=f.get("item_head", "not recorded"),
                                       frame_head=f.get("frame_head", "not recorded")) for f in pframes],
                items_older_than_frames=older, render=meta or "no render.json")


def judge_label(j) -> str:
    if j in (None, "primary"):
        return "filed judges (Claude)"
    return "re-judges (MiniMax-M3; GLM-5.3 on Petri)" if j == "rejudge" else f"judges: {j}"


def write_md(res: dict, path: str) -> None:
    L = [f"# {res['base']} against {res['other']}", ""]
    for side in ("base", "other"):
        r = res["renders"][side]
        L.append(f"- {side} `{r['name']}`: judges {r['judges']}, threshold {r['threshold']}, source {r['source']}, frame head {r['head']}, "
                 f"item heads {json.dumps(r['item_heads'], default=str) if isinstance(r['item_heads'], dict) else r['item_heads']}")
    t = res["verdict_change_totals"]
    L += ["", f"Mean-column verdict changes: {t['entries']} figure entries, {t['distinct']} distinct marks "
              f"({', '.join(f'{k} {v}' for k, v in t['distinct_by_category'].items()) or 'none'}).",
          "Clearance of the significant judge's interval from zero: "
          + "; ".join(f"{u}: " + ", ".join(f"{k} {v}" for k, v in b.items() if v) for u, b in t["clearance_bands"].items()), "",
          "| Figure | File | Compared | Categories / parts | Mean verdict changes |", "|---|---|---|---|---|"]
    for name, f in res["figures"].items():
        cats = dict(f.get("categories") or {})
        for p, v in (f.get("panels") or {}).items():
            cats.update({f"{c} ({p})": x for c, x in (v.get("categories") or {}).items()})
        parts = list(cats) + [k for k in ("transfer", "transfer_evals", "bon", "frontier", "padding") if k in f]
        nvc = sum(len(c["mean"]["verdict_changes"]) for c in cats.values())
        L.append(f"| {f['paper']} | `{name}` | {'yes' if f.get('compared') else f.get('not_compared', 'no')} | {', '.join(parts) or '-'} | {nvc} |")
    if res.get("figures_not_in_map"):
        L += ["", "Not compared (figure files the renders wrote that the README's figure map does not name): "
              + ", ".join(f"`{n}`" for n in res["figures_not_in_map"]) + "."]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---- the comparison ---------------------------------------------------------------------------------------------------------------
def compare(args) -> int:
    A, B = load(args.renders, args.base), load(args.renders, args.other)
    if A is None or B is None:
        raise SystemExit(f"need both renders under {args.renders}: {args.base}, {args.other}")
    fmap = JC.figure_map(getattr(args, "readme", None), getattr(args, "generator", None))
    parents = JC.lineage(getattr(args, "generator", None))
    os.makedirs(args.figs, exist_ok=True)
    la, lb = judge_label(A.get("JUDGES")), judge_label(B.get("JUDGES"))
    res = dict(base=args.base, other=args.other, base_head=A.get("HEAD"), other_head=B.get("HEAD"), judges=B.get("JUDGES"),
               renders=dict(base=render_record(A), other=render_record(B)), figures={})
    jev = judged_evals(A, B)
    for name, spec in fmap.items():
        fa, fb = os.path.join(A["dir"], "figures", f"{name}.png"), os.path.join(B["dir"], "figures", f"{name}.png")
        side = os.path.join(args.figs, f"{name}__filed_vs_rejudge.png")
        rec = dict(paper=spec["paper"], what=spec["what"], function=spec["function"], frames=spec["frames"], hook_frames=spec["hook_frames"],
                   layout_categories=spec["layout_categories"], drawn_a=os.path.exists(fa), drawn_b=os.path.exists(fb),
                   side_by_side=stack_pngs([fa, fb], [f"{spec['paper']} {name}: {la}", f"{spec['paper']} {name}: {lb}"], side))
        for f, tag in ((fa, "filed"), (fb, "rejudge")):
            if os.path.exists(f):
                shutil.copy(f, os.path.join(args.figs, f"{name}__{tag}.png"))
                if os.path.exists(f[:-4] + ".pdf"):
                    shutil.copy(f[:-4] + ".pdf", os.path.join(args.figs, f"{name}__{tag}.pdf"))
        rec.update(compare_figure(name, spec, A, B, parents, args.figs, la, lb, jev))
        res["figures"][name] = rec
    res["verdict_change_totals"] = verdict_totals(res["figures"])
    res["figures_not_in_map"] = JC.figures_not_in_map([os.path.join(A["dir"], "figures"), os.path.join(B["dir"], "figures")], fmap)
    ta, tb = A["tables"], B["tables"]
    res["tables"] = {lab: dict(identical=ta.get(lab) == tb.get(lab), in_a=lab in ta, in_b=lab in tb) for lab in sorted(set(ta) | set(tb))}
    res["judge_report"] = B.get("JUDGE_REPORT")
    with open(os.path.join(args.figs, "compare.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    write_md(res, os.path.join(args.figs, "compare.md"))
    for name, f in res["figures"].items():
        print(f"{f['paper']:<8} {name:<36} " + ("compared" if f.get("compared") else f.get("not_compared", "not compared")))
    t = res["verdict_change_totals"]
    print(f"mean-column verdict changes: {t['entries']} entries, {t['distinct']} distinct; clearance bands {json.dumps(t['clearance_bands'])}")
    if res["figures_not_in_map"]:
        print(f"!! not compared: {len(res['figures_not_in_map'])} figure files the renders wrote are not in the README's figure map: "
              + ", ".join(res["figures_not_in_map"]))
    return 0
