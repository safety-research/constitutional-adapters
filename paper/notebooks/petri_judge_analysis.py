"""The comparison half of `petri_judge_compare.py`: read the variant renders, compare the Petri figures and tables across judges,
draw the side-by-side and difference figures, and write the numbers the report quotes (`compare.json`, `compare.md`).

Units, all in the figures' own orientation (a REDUCTION in the 1-10 misalignment score against the unsteered model, up = less
misaligned):
  cell      (model, object, seed, eval) after the figure's own decode-draw pooling (`registry_nb.pool_draws`)
  object    (model, object, eval) with the seeds pooled as the appendix pools them (`registry_nb.pool_cell`: item-clustered across the
            seeds, the between-fit variance and the anchor once)
  mean      the category's mean column: `registry_nb.pool_seeds` over `category_means`, the large mark the figures draw
A difference is QUALITATIVE when the ordering of objects changes (rank correlation, best / worst object, pairs that swap with both
intervals clear of each other), when an interval's relation to zero changes (a reduction or an increase that is significant under
one judge and not under the other), or when a sign flips; a common offset of every value is a LEVEL shift.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# NO SOURCE IS FORCED AT IMPORT: the figure comparison reads the renders' frames only, and the notebook module resolves its own source
# (REGISTRY_NB_SOURCE, default auto). The item-level analysis needs the live registry and refuses any other source (`item_frame`).
import registry_nb as R                                         # noqa: E402
from petri_judge_compare import PETRI_FIGURES, PETRI_TABLES, all_tables   # noqa: E402

OPUS, GLM = "#2a78d6", "#eb6834"                                # categorical slots 1 and 2 (dataviz reference palette, light)
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d4"
CATEGORY = "Petri"
EVALS = list(R.PETRI_BEHAVIORS) + [R.PETRI_POOLED]
SHORT = {e: R.TICK.get(e, e).replace("-\n", "-").replace("/\n", "/").replace("\n", " ") if hasattr(R, "TICK") else e for e in EVALS}
SHORT[R.PETRI_POOLED] = "pooled"


def label(o: str) -> str:
    return R.LABEL.get(o, o)


def load(renders: str, name: str):
    import nbformat
    p = os.path.join(renders, name)
    if not os.path.exists(os.path.join(p, "frames.pkl")):
        return None
    fr = pickle.load(open(os.path.join(p, "frames.pkl"), "rb"))
    R.restore_pooled_ci_state(fr.get("POOLED_CI_STATE"))          # the render's units, so its dots pool here as they did there
    nb = nbformat.read(os.path.join(p, "registry_qwen_figures.executed.ipynb"), as_version=4)
    fr["tables"] = all_tables(nb)
    fr["nb"] = nb
    fr["dir"] = p
    return fr


# ---- units ----------------------------------------------------------------------------------------------------------------------
def cells(d: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """(model, object, seed, eval) Petri cells, decode draws pooled as the figure pools them, oriented (up = less misaligned)."""
    x = d[d.eval_id.isin(EVALS) & d.d_y.notna()]
    if model:
        x = x[x.model_id == model]
    x = R.pool_draws(x)
    lo, hi = -x.d_hi, -x.d_lo
    # `lane` keeps a prompted arm's canonical run (lane instruct_direct) apart from its seed-view decode draws (no lane), two marks
    return pd.DataFrame(dict(model_id=x.model_id, object_label=x.object_label, seed=x.seed.fillna(""), lane=x.lane.fillna("").astype(str), eval_id=x.eval_id,
                             red=-x.d_y, lo=lo, hi=hi, y_abs=x.y, anchor=x.anchor_value, run_id=x.run_id)).reset_index(drop=True)


def pool_objects(d: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """(model, object, eval): the mark the appendix draws for a cell's seeds and draws (`registry_nb.pool_cell`, the render's rule:
    decode draws item-clustered, then the fit seeds item-clustered across each other with the between-fit variance and the anchor
    once), oriented (up = less misaligned); `n_seeds` the fits pooled, `y_abs` / `anchor` the fits' mean reading and anchor."""
    x = d[d.eval_id.isin(EVALS) & d.d_y.notna()]
    if model:
        x = x[x.model_id == model]
    if not len(x):
        return pd.DataFrame(columns=["model_id", "object_label", "eval_id", "red", "lo", "hi", "n_seeds", "y_abs", "anchor"])
    p = R.pool_cell(x, y="d_y", lo="d_lo", hi="d_hi")
    fits = R.pool_draws(x, item_sets_apart=False).groupby(["model_id", "object_label", "eval_id"]).agg(y_abs=("y", "mean"), anchor=("anchor_value", "mean"))
    rows = []
    for r in p.itertuples():
        f = fits.loc[(r.model_id, r.object_label, r.eval_id)]
        rows.append(dict(model_id=r.model_id, object_label=r.object_label, eval_id=r.eval_id, red=-float(r.d_y), lo=-float(r.d_hi), hi=-float(r.d_lo),
                         n_seeds=len(r.seed_parts), y_abs=float(f.y_abs), anchor=float(f.anchor)))
    return pd.DataFrame(rows)


def category_mean(cm: pd.DataFrame) -> pd.DataFrame:
    """The figures' mean column for Petri (the pooled score), one row per (model, object): `pool_seeds` exactly."""
    m = R.pool_seeds(cm[cm.category == CATEGORY])
    return m[["model_id", "object_label", "y", "lo", "hi", "n_seeds", "anchor_abs"]].rename(columns={"y": "red"})


def sig(lo, hi) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "no CI"
    return "reduces" if lo > 0 else ("increases" if hi < 0 else "n.s.")


def compare_units(a: pd.DataFrame, b: pd.DataFrame, keys: list, tag_a="opus", tag_b="glm") -> pd.DataFrame:
    m = a.merge(b, on=keys, how="outer", suffixes=(f"_{tag_a}", f"_{tag_b}"))
    m[f"sig_{tag_a}"] = [sig(l, h) for l, h in zip(m[f"lo_{tag_a}"], m[f"hi_{tag_a}"])]
    m[f"sig_{tag_b}"] = [sig(l, h) for l, h in zip(m[f"lo_{tag_b}"], m[f"hi_{tag_b}"])]
    m["shift"] = m[f"red_{tag_b}"] - m[f"red_{tag_a}"]
    m["sig_change"] = m[f"sig_{tag_a}"] != m[f"sig_{tag_b}"]
    m["sign_flip"] = np.sign(m[f"red_{tag_a}"]) * np.sign(m[f"red_{tag_b}"]) < 0
    return m


def ordering(m: pd.DataFrame, group: list, tag_a="opus", tag_b="glm") -> pd.DataFrame:
    """Per group (an eval column, or a model's mean column): Spearman and Kendall rank correlation of the objects between judges,
    the best and worst object under each, and the pairs that swap order with the two intervals clear of each other under BOTH
    judges (a robust crossing) or under either (a crossing either judge would call)."""
    from scipy.stats import kendalltau, spearmanr
    out = []
    for k, g in m.groupby(group):
        g = g.dropna(subset=[f"red_{tag_a}", f"red_{tag_b}"])
        if len(g) < 3:
            continue
        rho = spearmanr(g[f"red_{tag_a}"], g[f"red_{tag_b}"]).statistic
        tau = kendalltau(g[f"red_{tag_a}"], g[f"red_{tag_b}"]).statistic
        best_a, best_b = g.loc[g[f"red_{tag_a}"].idxmax(), "object_label"], g.loc[g[f"red_{tag_b}"].idxmax(), "object_label"]
        worst_a, worst_b = g.loc[g[f"red_{tag_a}"].idxmin(), "object_label"], g.loc[g[f"red_{tag_b}"].idxmin(), "object_label"]
        rob, weak = [], []
        rr = g.to_dict("records")
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                x, y = rr[i], rr[j]
                sa = np.sign(x[f"red_{tag_a}"] - y[f"red_{tag_a}"]); sb = np.sign(x[f"red_{tag_b}"] - y[f"red_{tag_b}"])
                if sa * sb < 0:
                    sep_a = x[f"lo_{tag_a}"] > y[f"hi_{tag_a}"] or y[f"lo_{tag_a}"] > x[f"hi_{tag_a}"]
                    sep_b = x[f"lo_{tag_b}"] > y[f"hi_{tag_b}"] or y[f"lo_{tag_b}"] > x[f"hi_{tag_b}"]
                    pair = f"{label(x['object_label'])} vs {label(y['object_label'])}"
                    if sep_a and sep_b:
                        rob.append(pair)
                    elif sep_a or sep_b:
                        weak.append(pair + (" (separated under Opus)" if sep_a else " (separated under GLM)"))
        n_swaps = sum(1 for i in range(len(rr)) for j in range(i + 1, len(rr))
                      if np.sign(rr[i][f"red_{tag_a}"] - rr[j][f"red_{tag_a}"]) * np.sign(rr[i][f"red_{tag_b}"] - rr[j][f"red_{tag_b}"]) < 0)
        out.append(dict(group=k if not isinstance(k, tuple) else " / ".join(map(str, k)), n=len(g), spearman=rho, kendall=tau,
                        best_opus=label(best_a), best_glm=label(best_b), worst_opus=label(worst_a), worst_glm=label(worst_b),
                        swapped_pairs=n_swaps, of_pairs=len(rr) * (len(rr) - 1) // 2, robust_crossings="; ".join(rob),
                        one_sided_crossings="; ".join(weak)))
    return pd.DataFrame(out)


# ---- tables -----------------------------------------------------------------------------------------------------------------------
def table_cells(tex: str) -> list:
    rows = []
    for line in tex.splitlines():
        if "&" in line and line.rstrip().endswith("\\\\") and "\\bf" not in line:
            rows.append([c.strip() for c in line.rstrip()[:-2].split("&")])
    return rows


def table_diff(t1: str, t2: str) -> dict:
    a, b = table_cells(t1 or ""), table_cells(t2 or "")
    n = changed = 0
    for ra, rb in zip(a, b):
        for ca, cb in zip(ra, rb):
            n += 1
            changed += ca != cb
    return dict(rows_a=len(a), rows_b=len(b), cells=n, changed=changed, identical=(t1 == t2))


# ---- figures ----------------------------------------------------------------------------------------------------------------------
def stack_images(top: str, bottom: str, out: str, t_top: str, t_bot: str):
    """The two renders of one figure, one above the other at full resolution, each under a one-line header."""
    from PIL import Image, ImageDraw, ImageFont
    a, b = Image.open(top).convert("RGB"), Image.open(bottom).convert("RGB")
    w = max(a.width, b.width)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(28, w // 70))
    except OSError:
        font = ImageFont.load_default()
    hh = int(font.size * 2.0) if hasattr(font, "size") else 40
    canvas = Image.new("RGB", (w, a.height + b.height + 2 * hh + 30), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((20, hh * 0.25), t_top, fill=OPUS, font=font)
    canvas.paste(a, (0, hh))
    y2 = hh + a.height + 30
    dr.line([(0, y2 - 15), (w, y2 - 15)], fill=RULE, width=4)
    dr.text((20, y2 + hh * 0.25), t_bot, fill=GLM, font=font)
    canvas.paste(b, (0, y2 + hh))
    canvas.save(out)


def dumbbell(ax, m: pd.DataFrame, objects: list, title: str, ylabel: str | None, tag_a="opus", tag_b="glm"):
    """Per object: the Opus-chain value (filled circle) and the GLM-5.3 value (open diamond) with their intervals, joined."""
    xs = []
    for i, o in enumerate(objects):
        r = m[m.object_label == o]
        if not len(r):
            continue
        r = r.iloc[0]
        xs.append(i)
        ax.plot([i - 0.12, i + 0.12], [r[f"red_{tag_a}"], r[f"red_{tag_b}"]], color=MUTED, lw=1.2, zorder=1)
        for dx, tag, col, mk in ((-0.12, tag_a, OPUS, "o"), (0.12, tag_b, GLM, "D")):
            if pd.notna(r[f"lo_{tag}"]):
                ax.plot([i + dx] * 2, [r[f"lo_{tag}"], r[f"hi_{tag}"]], color=col, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([i + dx], [r[f"red_{tag}"]], s=64 if mk == "o" else 56, marker=mk, color=col if mk == "o" else "white",
                       edgecolors=col, linewidths=2, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(objects)))
    ax.set_xticklabels([label(o) for o in objects], rotation=55, ha="right", fontsize=10)
    ax.set_title(title, fontsize=12, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="o", ls="-", color=OPUS, markerfacecolor=OPUS, label="Opus-5 judge chain (original)"),
            Line2D([], [], marker="D", ls="-", color=GLM, markerfacecolor="white", label="GLM-5.3 judge")]


def save(fig, figs: str, name: str):
    fig.savefig(os.path.join(figs, f"{name}.png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(figs, f"{name}.pdf"), bbox_inches="tight")


# ---- the comparison ---------------------------------------------------------------------------------------------------------------
def compare(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    renders, figs = os.path.expanduser(args.renders), os.path.expanduser(args.figs)
    os.makedirs(figs, exist_ok=True)
    V = {n: load(renders, n) for n in ("opus_snapshot", "opus_live", "opus_rederived", "glm_live", "glm_matched", "glm_opus5only", "opus_opus5only")}
    V = {k: v for k, v in V.items() if v is not None}
    A, B = V["opus_live"], V["glm_live"]
    head = A.get("HEAD")
    out: dict = dict(head=int(head) if head is not None else None, variants=sorted(V), heads={k: int(v.get("HEAD")) for k, v in V.items() if v.get("HEAD") is not None},
                     judge_reports={k: v.get("PETRI_JUDGE_REPORT") for k, v in V.items()})
    md = []

    # 0. reproduction: the snapshot render against the committed notebook; the live render against the snapshot
    import nbformat
    committed = nbformat.read(os.path.join(HERE, "registry_qwen_figures.ipynb"), as_version=4)
    ct = all_tables(committed)
    repro = {}
    for k in ("opus_snapshot", "opus_live", "opus_rederived"):
        if k in V:
            repro[k] = {lab: table_diff(ct.get(lab), V[k]["tables"].get(lab)) for lab in PETRI_TABLES}
            repro[k]["all_tables_identical"] = sum(ct.get(l) == V[k]["tables"].get(l) for l in ct)
            repro[k]["all_tables"] = len(ct)
    out["reproduction"] = repro
    out["tables_opus_vs_glm"] = {lab: table_diff(A["tables"].get(lab), B["tables"].get(lab)) for lab in PETRI_TABLES}
    for tag, v in (("opus", A), ("glm", B)):                    # the paper's Petri tables under each judge, as the notebook prints them
        with open(os.path.join(os.path.expanduser(args.figs), f"petri_tables__{tag}.tex"), "w") as f:
            f.write(f"% Petri tables, judge {tag}, registry head {v.get('HEAD')}; rendered by petri_judge_compare.py\n\n"
                    + "\n\n".join(v["tables"].get(lab, f"% {lab}: not rendered") for lab in PETRI_TABLES) + "\n")

    # 1. Figure 5: Qwen, six behaviours + pooled
    q = R.MODEL_ID
    ca, cb = cells(A["d"], q), cells(B["d"], q)
    oa, ob = pool_objects(A["d"], q), pool_objects(B["d"], q)
    ma, mb = category_mean(A["cm"]), category_mean(B["cm"])
    ma, mb = ma[ma.model_id == q], mb[mb.model_id == q]
    f5_cells = compare_units(ca.drop(columns=["run_id"]), cb.drop(columns=["run_id"]), ["model_id", "object_label", "seed", "lane", "eval_id"])
    # the per-behaviour columns only: the pooled score's column IS the figure's mean column (`category_mean`), whose own pooling of a
    # prompted arm's canonical run and seed-view draws is the authoritative one
    f5_obj = compare_units(oa[oa.eval_id.ne(R.PETRI_POOLED)], ob[ob.eval_id.ne(R.PETRI_POOLED)], ["model_id", "object_label", "eval_id"])
    f5_mean = compare_units(ma, mb, ["model_id", "object_label"])
    out["fig5"] = dict(
        n_cells=int(len(f5_cells)), cell_sig_changes=f5_cells[f5_cells.sig_change][["object_label", "seed", "lane", "eval_id", "red_opus", "lo_opus", "hi_opus", "red_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm"]].round(2).to_dict("records"),
        cell_sign_flips=int(f5_cells.sign_flip.sum()),
        object_sig_changes=f5_obj[f5_obj.sig_change][["object_label", "eval_id", "red_opus", "lo_opus", "hi_opus", "red_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm"]].round(2).to_dict("records"),
        object_sign_flips=f5_obj[f5_obj.sign_flip][["object_label", "eval_id", "red_opus", "red_glm"]].round(2).to_dict("records"),
        mean=f5_mean[["object_label", "red_opus", "lo_opus", "hi_opus", "red_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm", "shift"]].round(2).to_dict("records"),
        ordering_per_eval=ordering(f5_obj, ["eval_id"]).round(3).to_dict("records"),
        ordering_mean=ordering(f5_mean.assign(g="pooled mean column"), ["g"]).round(3).to_dict("records"),
        anchors={e: dict(opus=float(oa[oa.eval_id == e].anchor.mean()), glm=float(ob[ob.eval_id == e].anchor.mean())) for e in EVALS},
        abs_shift_steered=float((f5_cells.y_abs_glm - f5_cells.y_abs_opus).mean()),
        reduction_shift=dict(mean=float(f5_obj["shift"].mean()), sd=float(f5_obj["shift"].std()), max_abs=float(f5_obj["shift"].abs().max())))

    # 2. Figure 14: the four models
    cA, cB = cells(A["d_all"]), cells(B["d_all"])
    oA, oB = pool_objects(A["d_all"]), pool_objects(B["d_all"])
    mA, mB = category_mean(A["cm_all"]), category_mean(B["cm_all"])
    mA, mB = mA[mA.object_label.isin(R.APPENDIX_OBJECTS)], mB[mB.object_label.isin(R.APPENDIX_OBJECTS)]
    f13_obj = compare_units(oA[oA.eval_id.ne(R.PETRI_POOLED)], oB[oB.eval_id.ne(R.PETRI_POOLED)], ["model_id", "object_label", "eval_id"])
    f13_obj = f13_obj[f13_obj.object_label.isin(R.APPENDIX_OBJECTS)]
    f13_mean = compare_units(mA, mB, ["model_id", "object_label"])
    out["fig13"] = dict(
        mean=f13_mean[["model_id", "object_label", "red_opus", "lo_opus", "hi_opus", "red_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm", "shift"]].round(2).to_dict("records"),
        mean_sig_changes=f13_mean[f13_mean.sig_change][["model_id", "object_label", "red_opus", "lo_opus", "hi_opus", "red_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm"]].round(2).to_dict("records"),
        mean_sign_flips=f13_mean[f13_mean.sign_flip][["model_id", "object_label", "red_opus", "red_glm"]].round(2).to_dict("records"),
        ordering_mean=ordering(f13_mean, ["model_id"]).round(3).to_dict("records"),
        ordering_models_per_object=ordering(f13_mean.rename(columns={"model_id": "object_label", "object_label": "model_id"}), ["model_id"]).round(3).to_dict("records"),
        eval_sig_changes=f13_obj[f13_obj.sig_change][["model_id", "object_label", "eval_id", "red_opus", "lo_opus", "hi_opus", "red_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm"]].round(2).to_dict("records"),
        eval_sign_flips=int(f13_obj.sign_flip.sum()), n_eval_units=int(len(f13_obj)),
        ordering_per_eval=ordering(f13_obj, ["model_id", "eval_id"]).round(3).to_dict("records"),
        anchors={m: {e: dict(opus=float(oA[(oA.model_id == m) & (oA.eval_id == e)].anchor.mean()), glm=float(oB[(oB.model_id == m) & (oB.eval_id == e)].anchor.mean()))
                     for e in EVALS} for m in R.MODEL_ORDER})

    # 3. Figure 8: transfer (base->instruct minus instruct-direct), the Petri category and its evals
    tA, tB = A["tr_cat"], B["tr_cat"]
    tA, tB = tA[tA.category == CATEGORY], tB[tB.category == CATEGORY]
    keys = ["model_id", "object_label"]
    ycol = "diff" if "diff" in tA.columns else "y"
    f8 = tA[keys + [ycol, "lo", "hi"]].rename(columns={ycol: "red"}).merge(tB[keys + [ycol, "lo", "hi"]].rename(columns={ycol: "red"}), on=keys, suffixes=("_opus", "_glm"), how="outer")
    f8["sig_opus"] = [sig(l, h) for l, h in zip(f8.lo_opus, f8.hi_opus)]
    f8["sig_glm"] = [sig(l, h) for l, h in zip(f8.lo_glm, f8.hi_glm)]
    f8["shift"] = f8.red_glm - f8.red_opus
    pA, pB = A["tr_pe"], B["tr_pe"]
    # tr_pe's `diff` is raw (base->instruct minus instruct-direct of the score's delta, NOT oriented): orient it as tr_cat is
    def _orient(t):
        t = t[t.eval_id.isin(EVALS)].copy()
        t["diff"], t["lo"], t["hi"] = -t["diff"], -t["hi"], -t["lo"]
        return t
    pe = _orient(pA).merge(_orient(pB), on=["model_id", "object_label", "eval_id"], suffixes=("_opus", "_glm"))
    pe["sig_opus"] = [sig(l, h) for l, h in zip(pe.lo_opus, pe.hi_opus)]
    pe["sig_glm"] = [sig(l, h) for l, h in zip(pe.lo_glm, pe.hi_glm)]
    out["fig8"] = dict(category=f8.round(2).to_dict("records"),
                       eval_sig_changes=pe[pe.sig_opus != pe.sig_glm][["model_id", "object_label", "eval_id", "diff_opus", "lo_opus", "hi_opus", "diff_glm", "lo_glm", "hi_glm", "sig_opus", "sig_glm"]].round(2).to_dict("records"),
                       eval_sign_flips=int((np.sign(pe.diff_opus) * np.sign(pe.diff_glm) < 0).sum()), n_eval_units=int(len(pe)))

    # 4. sensitivity: the Opus chain's refused and fallback-served transcripts
    sens = {}
    pairs = [("glm_live", "glm_matched", "GLM: all items vs without the 28 Opus-refused items"),
             ("opus_live", "opus_opus5only", "Opus chain: all vs Opus-5-served items only (fallback-served and refused dropped)"),
             ("opus_opus5only", "glm_opus5only", "on Opus-5-served items only: Opus vs GLM"),
             ("opus_live", "opus_rederived", "Opus chain: filed rows vs re-derived from items (the loader path itself)"),
             ("opus_rederived", "glm_live", "Opus chain re-derived by the same bootstrap as GLM vs GLM (no RNG difference between the sides)")]
    for a_, b_, what in pairs:
        if a_ in V and b_ in V:
            xa, xb = category_mean(V[a_]["cm_all"]), category_mean(V[b_]["cm_all"])
            xa, xb = xa[xa.object_label.isin(R.OBJECTS)], xb[xb.object_label.isin(R.OBJECTS)]
            mm = compare_units(xa, xb, ["model_id", "object_label"], "a", "b")
            q5a, q5b = category_mean(V[a_]["cm"]), category_mean(V[b_]["cm"])
            qq = compare_units(q5a, q5b, ["model_id", "object_label"], "a", "b")
            ea, eb = pool_objects(V[a_]["d_all"]), pool_objects(V[b_]["d_all"])
            ee = compare_units(ea[ea.eval_id.ne(R.PETRI_POOLED)], eb[eb.eval_id.ne(R.PETRI_POOLED)], ["model_id", "object_label", "eval_id"], "a", "b")
            sens[f"{a_}__{b_}"] = dict(what=what, mean_abs_shift_fig13=float(mm["shift"].abs().mean()), max_abs_shift_fig13=float(mm["shift"].abs().max()),
                                       fig13_sig_changes=mm[mm.sig_change][["model_id", "object_label", "red_a", "lo_a", "hi_a", "red_b", "lo_b", "hi_b"]].round(2).to_dict("records"),
                                       fig5_sig_changes=qq[qq.sig_change][["object_label", "red_a", "lo_a", "hi_a", "red_b", "lo_b", "hi_b"]].round(2).to_dict("records"),
                                       eval_sig_changes=ee[ee.sig_change][["model_id", "object_label", "eval_id", "red_a", "lo_a", "hi_a", "red_b", "lo_b", "hi_b"]].round(2).to_dict("records"),
                                       eval_max_abs_shift=float(ee["shift"].abs().max()),
                                       ordering_fig13=ordering(mm, ["model_id"], "a", "b")[["group", "spearman", "kendall", "swapped_pairs", "robust_crossings"]].round(3).to_dict("records"))
    out["sensitivity"] = sens

    # ---- figures ----------------------------------------------------------------------------------------------------------------
    hA, hB = A.get("HEAD"), B.get("HEAD")
    for f in PETRI_FIGURES:
        for tag, v in (("opus", A), ("glm", B)):
            for ext in ("png", "pdf"):
                src = os.path.join(v["dir"], "figures", f"{f}.{ext}")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(figs, f"{f}__{tag}.{ext}"))
        if "opus_snapshot" in V:
            src = os.path.join(V["opus_snapshot"]["dir"], "figures", f"{f}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(figs, f"{f}__opus_snapshot456957.png"))
        pa, pb = os.path.join(A["dir"], "figures", f"{f}.png"), os.path.join(B["dir"], "figures", f"{f}.png")
        if os.path.exists(pa) and os.path.exists(pb):
            stack_images(pa, pb, os.path.join(figs, f"{f}__opus_vs_glm.png"),
                         f"Original judge: claude-opus-5 with its fallback chain (registry head {hA})",
                         f"Re-judged: glm-5.3, same runs, same head ({hB})")

    # Figure 5 difference panel: per eval column, objects in the figure's order
    objs = [o for o in R.OBJECTS if o in set(f5_obj.object_label)]
    fig, axes = plt.subplots(2, 7, figsize=(26, 9.5), sharey="row", gridspec_kw=dict(height_ratios=[3, 1.4]))
    for j, e in enumerate(EVALS):
        sub = f5_mean if e == R.PETRI_POOLED else f5_obj[f5_obj.eval_id == e]
        dumbbell(axes[0, j], sub, objs, SHORT[e] + (" (mean column)" if e == R.PETRI_POOLED else ""), "reduction in misalignment score (1-10)" if j == 0 else None)
        ax = axes[1, j]
        sh = [sub[sub.object_label == o]["shift"].iloc[0] if (sub.object_label == o).any() else np.nan for o in objs]
        ax.bar(range(len(objs)), sh, color=MUTED, width=0.6)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(objs))); ax.set_xticklabels([label(o) for o in objs], rotation=55, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=10); ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("GLM minus Opus", fontsize=11)
        axes[0, j].set_xticklabels([])
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Figure 5 (Qwen3.5-4B, operating dose): each object's seed-pooled reduction per Petri column, both judges, registry head {hA}", y=1.06, fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figs, "fig5_misalignment__difference")
    plt.close(fig)

    # Figure 14 difference panel: the mean column per model
    objs13 = [o for o in R.APPENDIX_OBJECTS if o in set(f13_mean.object_label)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row", gridspec_kw=dict(height_ratios=[3, 1.4]))
    for j, m in enumerate(R.MODEL_ORDER):
        sub = f13_mean[f13_mean.model_id == m]
        dumbbell(axes[0, j], sub, objs13, R.MODEL_LABEL[m], "reduction in pooled Petri score (1-10)" if j == 0 else None)
        axes[0, j].set_xticklabels([])
        ax = axes[1, j]
        sh = [sub[sub.object_label == o]["shift"].iloc[0] if (sub.object_label == o).any() else np.nan for o in objs13]
        ax.bar(range(len(objs13)), sh, color=MUTED, width=0.6); ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(objs13))); ax.set_xticklabels([label(o) for o in objs13], rotation=55, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=10); ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("GLM minus Opus", fontsize=11)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Figure 14 (all four models): the Petri mean column (pooled score) per object, both judges, registry head {hA}", y=1.06, fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, figs, "figA_misalignment_models__difference")
    plt.close(fig)

    # Figure 8 difference panel: the Petri category of the transfer figure
    if len(f8):
        objs8 = [o for o in R.TRANSFER_OBJECTS if o in set(f8.object_label)]
        fig, axes = plt.subplots(1, len(R.TRANSFER_MODELS), figsize=(5 * len(R.TRANSFER_MODELS), 4.8), sharey=True)
        for j, m in enumerate(R.TRANSFER_MODELS):
            dumbbell(axes[j], f8[f8.model_id == m], objs8, R.MODEL_LABEL[m], "base→instruct minus instruct-direct\n(Petri reduction, 1-10)" if j == 0 else None)
        fig.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.06))
        fig.suptitle(f"Figure 8 (transfer), Petri category: both judges, registry head {hA}", y=1.12, fontsize=14, x=0.01, ha="left")
        fig.tight_layout()
        save(fig, figs, "figA_transfer_models__petri_difference")
        plt.close(fig)

    if getattr(args, "served", ""):
        out["items"] = item_analysis(args, figs)
    out["md"] = md
    path = args.out_json or os.path.join(figs, "compare.json")
    json.dump(out, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {path}")
    return 0


# ---- item level ---------------------------------------------------------------------------------------------------------------
RUNS_SQL = ("SELECT r.run_id, r.model_id, r.eval_id, r.transfer_kind, r.strength_nominal, o.paper_label, pc.in_current_paper, pc.selection_state "
            "FROM run r JOIN object o ON o.object_id = r.object_id JOIN v_paper_current pc ON pc.run_id = r.run_id WHERE r.eval_id LIKE 'petri%'")


def item_frame(served_tsv: str) -> pd.DataFrame:
    """One row per (run, transcript): the Opus chain's stored score (newest complete judgment of the primary judge), GLM-5.3's, who
    served the Opus score (served.tsv from `exclusions`), and the run's model / object / eval. Read from the live registry only: the
    committed snapshot predates the re-judges' item rows, so another source refuses rather than read fewer of them."""
    if R.SOURCE != "live":
        raise RuntimeError(f"the item-level Petri analysis reads the judges' item rows from the live registry; this process resolved "
                           f"REGISTRY_NB_SOURCE to {R.SOURCE!r} (set REGISTRY_NB_SOURCE=live with the registry at REGISTRY_ROOT)")

    def latest(jid):
        it = R.petri_item_scores(jid)
        it = it[it.judgment_status.eq("complete")].sort_values("judgment_finished_utc")
        last = it.groupby("run_id").judgment_id.last()
        it = it[it.judgment_id.isin(set(last))]
        return it[["run_id", "item_id", "trial_index", "value", "label"]]
    o, g = latest(R.PETRI_JUDGE_ALIASES["claude-opus-5"]), latest(R.PETRI_JUDGE_ALIASES["glm-5.3"])
    x = g.merge(o, on=["run_id", "item_id", "trial_index"], how="left", suffixes=("_glm", "_opus"))
    s = pd.read_csv(served_tsv, sep="\t", dtype={"trial_index": int})
    x = x.merge(s, on=["run_id", "item_id", "trial_index"], how="left")
    runs = R.query(RUNS_SQL, "petri_runs")
    x = x.merge(runs, on="run_id", how="left")
    x["served_class"] = np.where(x.served.eq("refused"), "refused (no Opus-chain score)",
                                 np.where(x.served.fillna("").str.contains("claude-opus-5"), "claude-opus-5",
                                          np.where(x.served.fillna("").str.contains("opus-4-8"), "claude-opus-4-8 (fallback)",
                                                   np.where(x.served.fillna("").str.contains("grok"), "grok-4.6 (fallback)", "unknown"))))
    x["diff"] = x.value_glm - x.value_opus
    return x


def item_analysis(args, figs: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr
    x = item_frame(args.served)
    x = x[x.eval_id.ne(R.PETRI_POOLED)]                           # the pooled runs re-use the behaviour runs' transcripts: count each once
    both = x.dropna(subset=["value_opus", "value_glm"])
    out = dict(n_items=int(len(x)), n_both=int(len(both)),
               by_served={k: dict(n=int(len(g)), mean_opus=float(g.value_opus.mean()) if g.value_opus.notna().any() else None,
                                  mean_glm=float(g.value_glm.mean()), mean_diff=float(g["diff"].mean()) if g["diff"].notna().any() else None,
                                  mae=float(g["diff"].abs().mean()) if g["diff"].notna().any() else None)
                          for k, g in x.groupby("served_class")},
               overall=dict(mean_opus=float(both.value_opus.mean()), mean_glm=float(both.value_glm.mean()), mean_diff=float(both["diff"].mean()),
                            sd_diff=float(both["diff"].std()), mae=float(both["diff"].abs().mean()), exact=float((both["diff"] == 0).mean()),
                            within1=float((both["diff"].abs() <= 1).mean()), ge3=float((both["diff"].abs() >= 3).mean()),
                            spearman=float(spearmanr(both.value_opus, both.value_glm).statistic), pearson=float(pearsonr(both.value_opus, both.value_glm).statistic),
                            mae_shift_removed=float((both["diff"] - both["diff"].mean()).abs().mean())),
               by_behaviour={b: dict(n=int(len(g)), mean_diff=float(g["diff"].mean()), mae=float(g["diff"].abs().mean()),
                                     spearman=float(spearmanr(g.value_opus, g.value_glm).statistic))
                             for b, g in both.groupby("eval_id")},
               by_model={m: dict(n=int(len(g)), mean_diff=float(g["diff"].mean()), mae=float(g["diff"].abs().mean()))
                         for m, g in both.groupby("model_id")},
               by_object={o: dict(n=int(len(g)), mean_diff=float(g["diff"].mean())) for o, g in both.groupby("paper_label")},
               refused_glm=x[x.served_class.str.startswith("refused")][["model_id", "paper_label", "eval_id", "item_id", "value_glm"]].to_dict("records"))
    # the run level: per paper run, the two judges' means over the transcripts both scored
    rl = both.groupby(["run_id", "eval_id", "model_id", "paper_label"]).agg(opus=("value_opus", "mean"), glm=("value_glm", "mean"), n=("diff", "size")).reset_index()
    out["run_level"] = dict(n=int(len(rl)), spearman=float(spearmanr(rl.opus, rl.glm).statistic), pearson=float(pearsonr(rl.opus, rl.glm).statistic),
                            mean_diff=float((rl.glm - rl.opus).mean()), sd_diff=float((rl.glm - rl.opus).std()),
                            by_behaviour={b: dict(spearman=float(spearmanr(g.opus, g.glm).statistic), mean_diff=float((g.glm - g.opus).mean()))
                                          for b, g in rl.groupby("eval_id")})
    out["n_transcripts"] = int(x.transcript.nunique()) if "transcript" in x.columns else None
    # one row per distinct transcript (two runs of two instruments can file the same transcript)
    top = both.assign(absdiff=both["diff"].abs()).sort_values(["absdiff", "run_id"], ascending=[False, True])
    if "transcript" in top.columns:
        top = top.drop_duplicates("transcript")
    out["top_disagreements"] = top.head(25)[["model_id", "paper_label", "eval_id", "item_id", "value_opus", "value_glm", "diff", "served_class"]].to_dict("records")
    # per-scenario: the transcripts of one scenario across every run (does the disagreement live in particular scenarios?)
    sc = both.groupby("item_id").agg(n=("diff", "size"), mean_diff=("diff", "mean"), mae=("diff", lambda v: v.abs().mean())).reset_index()
    out["scenarios_most_shifted"] = sc[sc.n >= 10].sort_values("mean_diff").head(8).round(2).to_dict("records") + \
        sc[sc.n >= 10].sort_values("mean_diff").tail(5).round(2).to_dict("records")

    # figure: item-level joint distribution (counts), one hue light -> dark
    H = pd.crosstab(both.value_opus.astype(int), both.value_glm.astype(int)).reindex(index=range(1, 11), columns=range(1, 11), fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(np.log10(H.values + 1), origin="lower", cmap="Blues", extent=(0.5, 10.5, 0.5, 10.5))
    for i in range(10):
        for j in range(10):
            v = H.values[i, j]
            if v:
                ax.text(j + 1, i + 1, str(v), ha="center", va="center", fontsize=7.5, color="white" if np.log10(v + 1) > 0.6 * np.log10(H.values.max() + 1) else INK)
    ax.plot([0.5, 10.5], [0.5, 10.5], color=INK2, lw=1, ls="--")
    ax.set_xlabel("GLM-5.3 score (1-10)"); ax.set_ylabel("Opus-5 chain score (1-10)")
    ax.set_xticks(range(1, 11)); ax.set_yticks(range(1, 11))
    ax.set_title(f"Every paper Petri transcript both judges scored (n = {len(both):,})\nown-dimension score; cell = transcripts, colour = log count", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log10(transcripts + 1)")
    save(fig, figs, "items_opus_vs_glm_joint")
    plt.close(fig)

    # figure: run-level means, faceted by behaviour (single ink per facet)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    for ax, b in zip(axes.flat, R.PETRI_BEHAVIORS):
        g = rl[rl.eval_id == b]
        ax.scatter(g.opus, g.glm, s=16, color=OPUS, alpha=0.6, edgecolors="none")
        ax.plot([1, 10], [1, 10], color=INK2, lw=1, ls="--")
        rho = spearmanr(g.opus, g.glm).statistic if len(g) > 2 else np.nan
        ax.set_title(f"{SHORT[b]}\nruns {len(g)} · Spearman {rho:.2f} · GLM − Opus {(g.glm - g.opus).mean():+.2f}", fontsize=10.5, loc="left")
        ax.set_xlim(0.8, 10.2); ax.set_ylim(0.8, 10.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Opus-5 chain run mean")
    for ax in axes[:, 0]:
        ax.set_ylabel("GLM-5.3 run mean")
    fig.suptitle("Per paper Petri run: the two judges' mean score over the transcripts both scored", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, figs, "runs_opus_vs_glm_by_behaviour")
    plt.close(fig)

    # figure: mean item shift by behaviour and by who served the Opus score
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    bb = pd.DataFrame(out["by_behaviour"]).T.reindex(R.PETRI_BEHAVIORS)
    axes[0].barh([SHORT[b] for b in bb.index], bb.mean_diff, color=MUTED)
    axes[0].axvline(0, color=INK2, lw=1); axes[0].set_xlabel("mean GLM − Opus per transcript (1-10 scale)")
    axes[0].set_title("by behaviour", loc="left", fontsize=11)
    ss = {k: v for k, v in out["by_served"].items() if v["mean_diff"] is not None}
    axes[1].barh([f"{k} (n={v['n']})" for k, v in ss.items()], [v["mean_diff"] for v in ss.values()], color=MUTED)
    axes[1].axvline(0, color=INK2, lw=1); axes[1].set_xlabel("mean GLM − Opus per transcript")
    axes[1].set_title("by who served the stored Opus-chain score", loc="left", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, figs, "items_shift_by_behaviour_and_server")
    plt.close(fig)
    rl.to_csv(os.path.join(figs, "run_level_opus_vs_glm.csv"), index=False)
    both[["run_id", "model_id", "paper_label", "eval_id", "item_id", "trial_index", "value_opus", "value_glm", "diff", "served_class"]].to_csv(
        os.path.join(figs, "items_opus_vs_glm.csv.gz"), index=False)
    return out
