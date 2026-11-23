"""appendix_judge_comparison: the judge-robustness figure of appendix D (`fig:app_judges`, figA_judge_comparison_a and _b) and the numbers
the paragraph beside it quotes, drawn from frozen readings.

    import appendix_judge_comparison as JC
    jc = JC.load()                                   # data/appendix_judge_comparison.json, held to its contract (JC.problems)
    for part in JC.PARTS:
        JC.draw(jc, part); plt.show()                # figA_judge_comparison_a / _b, saved into R.FIGS
    print(JC.summary_text(jc))                       # the paragraph's numbers, each recomputed from the file
    print(JC.provenance_line(jc))

WHAT IS COMPARED. The same Qwen3.5-4B transcripts graded twice: by the Claude judges the registry filed first (claude-haiku-4-5 for
StrongREJECT, claude-sonnet-5 for over-refusal, claude-opus-5 for Petri; circles) and by the non-Claude judges the paper reads
(minimax-m3, glm-5.3 for Petri; diamonds). Each mark is one reading's change from the unsteered model, oriented as the main figures
orient it (up is better), with its 95% interval; inside a seed the two judges' marks sit side by side, joined by a thin line in the
object's colour, and the bold column after the separator is the category mean (Figure 4's ring rule).

THE READINGS ARE FROZEN. They are the main figures' own deltas at registry head 537565, the head the paper's final figures were drawn
at, read under the two judging policies (`REGISTRY_NB_JUDGES=primary` and `rejudge`) by the notebook's load chain: the per-eval marks
`pool_draws` over the frame's rows of a cell, the mean marks `pool_seeds` over its category means. `data/build_appendix_judge_comparison.py`
writes the file from a frozen copy of the registry and records how; this module opens no registry, so the figure and the numbers are
the same in every render, whatever its source, judging policy or StrongREJECT threshold.

THE NUMBERS (`text_numbers`), each read from the file:
  * the rank correlation (Spearman) of the interventions between the two judges' mean marks in each model x evaluation-category column
    of the all-model figures (four models, five judged categories: 20 columns), its minimum and how many columns reach 0.90;
  * the undefended Qwen3.5-4B model's static attack success under each judge: per attack, the mean of the unsteered readings the static
    panel's rows are differenced against;
  * the Difference LoRA's multi-turn mean mark under each judge;
  * the largest change of an over-refusal or Petri mean mark between the judges.

THE DRAWING is the delivered figure's own panel loop (one column per eval and a bold mean column after a separator; in a column the
objects in the paper's order; in a slot the three fit seeds fanned; in a seed the two judges' marks side by side) on `registry_nb`'s
style and helpers: `R.style()`, the object palette and names, the control-pool comparison's two marker shapes and their area balance, its
mark areas and bar widths on part a, `R.tick` for eval labels, `R._panel_grid` and `R.legend_stack`; files are written by `R.save`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import registry_nb as R

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "appendix_judge_comparison.json"
BUILDER = "paper/notebooks/data/build_appendix_judge_comparison.py"
SCHEMA = "appendix_judge_comparison/1"
FIGURE = "figA_judge_comparison"
FIGURES = (f"{FIGURE}_a", f"{FIGURE}_b")
#: the registry head the readings are pinned to: the head the paper's final figures were drawn at
AS_OF = 537565
JUDGES = ("original", "new")
POLICY = {"original": "primary", "new": "rejudge"}
JUDGED = ("static", "multi-query", "multi-turn", "over-refusal", "Petri")
#: the judge model of each category under each policy; the file must bear these out
JUDGE_MODEL = {"original": {"static": "claude-haiku-4-5", "multi-query": "claude-haiku-4-5", "multi-turn": "claude-haiku-4-5",
                            "over-refusal": "claude-sonnet-5", "Petri": "claude-opus-5"},
               "new": {"static": "minimax-m3", "multi-query": "minimax-m3", "multi-turn": "minimax-m3",
                       "over-refusal": "minimax-m3", "Petri": "glm-5.3"}}
JUDGE_LABEL = {"minimax-m3": "MiniMax-M3", "glm-5.3": "GLM-5.3"}
MODELS = ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it")

# THE TWO PARTS, each about the canvas width of Figure 4 so it scales to the text width at the main figures' type size. Part a is the
# jailbreak side, one panel per row (Figure 3's static, Figure 4's multi-query and multi-turn), part b over-refusal and then Petri, each
# across the full width. Row heights: part a's three rows at 4.1 in keep its canvas under 14 in with the two-row legend; part b's two
# rows at 4.6 in give Petri, its densest panel, room for its seven columns.
# ONE MARK SIZE PER FIGURE, set by its most crowded panel: part a takes the control-pool comparison's areas and bar widths, part b smaller
# ones, since Petri's seven columns leave it 2.1 in per column and at larger marks its densest behaviours merge.
PARTS = {"a": dict(name=FIGURES[0], layout=[["static"], ["multi-query"], ["multi-turn"]], height=4.1,
                   marks=dict(seed=float(R.CTLSTOP_SEED_MARK), mean=float(R.CTLSTOP_MEAN_MARK), seed_bar=R.CTLSTOP_SEED_BAR,
                              mean_bar=R.CTLSTOP_MEAN_BAR)),
         "b": dict(name=FIGURES[1], layout=[["over-refusal"], ["Petri"]], height=4.6,
                   marks=dict(seed=28.0, mean=105.0, seed_bar=1.0, mean_bar=2.2))}
WIDTH = 16.4                                                        # figure inches before the tight crop: the control-pool comparison's
LEGEND_GAP = 0.0                                                    # between the object key and the judge key
# the two judges take the control-pool comparison's two shapes (circle, diamond) and its area balance
JUDGE_RULE = {"original": "published", "new": "control_pool"}
MARKER = {j: R.CTLSTOP_RULE_MARKER[JUDGE_RULE[j]] for j in JUDGES}
AREA = {j: R.CTLSTOP_RULE_AREA[JUDGE_RULE[j]] for j in JUDGES}
SEED_HALF, PAIR_HALF, MEAN_PAIR_HALF = 0.34, 0.24, 0.22             # seeds +- SEED_HALF x slot; a pair +- PAIR_HALF x seed gap
MISSING_MARK = dict(marker="x", color=R.MUTED, s=40, lw=1.4)
OBJECTS = list(R.OBJECTS)
SEEDS = tuple(R.CTLSTOP_SEEDS)

MARK_COLUMNS = ["judge", "category", "eval", "object", "seed", "y", "lo", "hi", "open", "partial", "n", "k", "judge_model"]
ANCHOR_COLUMNS = ["judge", "eval", "asr"]
MEAN_COLUMNS = ["judge", "model", "category", "object", "y"]
BLOCKS = {"marks": MARK_COLUMNS, "static_anchors": ANCHOR_COLUMNS, "mean_columns": MEAN_COLUMNS}
#: the counts the figure draws: 390 marks in part a and 372 in part b, per judge half of each; 20 mean columns
N_MARKS = {"a": 390, "b": 372}
N_COLUMNS = len(MODELS) * len(JUDGED)
#: a rank correlation at a quoted bound counts as reaching it (Spearman over 9 objects is 1 - 6 sum d^2 / 720, exactly 0.9 at sum d^2 = 12)
RHO_TOL = 1e-9


class JudgeComparisonDataError(ValueError):
    """The committed file breaks its contract: the figure is not drawn from it."""


# ---- the data ----------------------------------------------------------------------------------------------------------------------
def canonical(obj) -> str:
    """The text a payload hash is taken over: sorted keys, no whitespace, floats as Python writes them (exact round trip)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload(data: dict) -> dict:
    """The readings the payload hash covers: the three blocks."""
    return {k: data[k] for k in BLOCKS}


def payload_sha256(data: dict) -> str:
    return hashlib.sha256(canonical(payload(data)).encode()).hexdigest()


def frame(data: dict, block: str) -> pd.DataFrame:
    b = data[block]
    return pd.DataFrame(b["rows"], columns=b["columns"])


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def problems(data: dict) -> list:
    """Every way the file breaks its contract, by name (empty when it holds)."""
    out = []
    if data.get("schema") != SCHEMA:
        return [f"schema {data.get('schema')!r}, expected {SCHEMA!r}"]
    for block, cols in BLOCKS.items():
        b = data.get(block)
        if not isinstance(b, dict) or b.get("columns") != cols or not isinstance(b.get("rows"), list):
            out.append(f"{block}: expected columns {cols} and a list of rows")
        elif any(not isinstance(r, list) or len(r) != len(cols) for r in b["rows"]):
            out.append(f"{block}: a row of the wrong length")
    if out:
        return out
    pv = data.get("provenance") or {}
    if pv.get("script") != BUILDER or len(str(pv.get("script_sha256", ""))) != 64:
        out.append(f"provenance names {pv.get('script')!r}, not the builder {BUILDER} with its sha256")
    if pv.get("payload_sha256") != payload_sha256(data):
        out.append("provenance.payload_sha256 is not the readings' hash: the readings changed after the builder wrote them")
    reg = pv.get("registry") or {}
    if reg.get("as_of") != AS_OF:
        out.append(f"provenance.registry.as_of {reg.get('as_of')!r}, expected {AS_OF}")
    if reg.get("head_read_by_frames") is not None and any(h < AS_OF for h in (reg.get("head_read_by_frames") or {}).values()):
        out.append("the frames were read below the pinned head")
    if [s for s in _strings(data) if s.startswith(("/", "~")) or "/home/" in s or "\\Users\\" in s]:
        out.append("the file names an absolute or home path")
    m = frame(data, "marks")
    if set(m.judge) != set(JUDGES) or set(m.category) != set(JUDGED):
        out.append(f"marks: judges {sorted(set(m.judge))}, categories {sorted(set(m.category))}")
        return out
    for part, spec in PARTS.items():
        cats = [c for row in spec["layout"] for c in row]
        n = int(m.category.isin(cats).sum())
        if n != N_MARKS[part]:
            out.append(f"marks: {n} in part {part}, expected {N_MARKS[part]}")
    for j in JUDGES:
        for cat in JUDGED:
            got = sorted(set(m[(m.judge == j) & (m.category == cat) & (m["eval"] != R.MEAN_COLUMN)].judge_model))
            if got != [JUDGE_MODEL[j][cat]]:
                out.append(f"marks: {j} {cat} judged by {got}, expected {JUDGE_MODEL[j][cat]}")
    if m.duplicated(["judge", "category", "eval", "object", "seed"]).any():
        out.append("marks: two readings for one (judge, category, eval, object, seed)")
    bad = m[~m["object"].isin(OBJECTS)]
    if len(bad):
        out.append(f"marks: objects outside the paper's {sorted(set(bad['object']))}")
    for r in m.itertuples():
        if not (isinstance(r.y, float) and math.isfinite(r.y)):
            out.append(f"marks: {r.judge} {r.category} {r.object} {r.seed} {r.eval} has no value")
            break
        lo_na, hi_na = pd.isna(r.lo), pd.isna(r.hi)
        if lo_na != hi_na or (not lo_na and not r.lo <= r.y <= r.hi):
            out.append(f"marks: {r.judge} {r.category} {r.object} {r.seed} {r.eval}: interval [{r.lo}, {r.hi}] does not hold {r.y}")
            break
    paired = m.groupby(["category", "eval", "object", "seed"]).judge.nunique()
    if (paired != 2).any():
        out.append(f"marks: {int((paired != 2).sum())} readings with only one judge's mark")
    a = frame(data, "static_anchors")
    if sorted(map(tuple, a[["judge", "eval"]].values.tolist())) != sorted((j, e) for j in JUDGES for e in R.CATEGORIES["static"][0]):
        out.append("static_anchors: one unsteered reading per judge and static attack")
    elif not a.asr.between(0, 100).all():
        out.append("static_anchors: an attack success outside 0-100 %")
    c = frame(data, "mean_columns")
    cols = c.groupby(["model", "category"]).judge.nunique()
    if set(c.model) != set(MODELS) or set(c.category) != set(JUDGED) or len(cols) != N_COLUMNS or (cols != 2).any():
        out.append(f"mean_columns: {len(cols)} model x category columns with both judges, expected {N_COLUMNS}")
    return out


def load(path=None) -> dict:
    """The committed file (default data/appendix_judge_comparison.json), refused by name when it breaks its contract."""
    p = pathlib.Path(path) if path else DATA
    data = json.loads(p.read_text(encoding="utf-8"))
    bad = problems(data)
    if bad:
        raise JudgeComparisonDataError(f"{p.name}: " + "; ".join(bad[:8]) + (f"; ... {len(bad) - 8} more" if len(bad) > 8 else ""))
    return data


# ---- the numbers -------------------------------------------------------------------------------------------------------------------
def rank_correlations(data: dict) -> pd.DataFrame:
    """Per (model, category) column of the all-model figures: the Spearman rank correlation of the objects' mean marks between the two
    judges, over the objects both judges draw, with how many there are (the Pearson correlation of the average ranks, as
    `scipy.stats.spearmanr` computes it)."""
    c = frame(data, "mean_columns")
    a, b = (c[c.judge == j].drop(columns="judge") for j in JUDGES)
    both = a.merge(b, on=["model", "category", "object"], suffixes=("_a", "_b"))
    rows = []
    for (mo, cat), g in both.groupby(["model", "category"]):
        rho = float(np.corrcoef(g.y_a.rank().to_numpy(float), g.y_b.rank().to_numpy(float))[0, 1]) if len(g) > 2 else float("nan")
        rows.append(dict(model=mo, category=cat, n=len(g), rho=rho))
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out.model, MODELS)
    out["category"] = pd.Categorical(out.category, JUDGED)
    return out.sort_values(["model", "category"]).reset_index(drop=True)


def _pct_range(v) -> tuple:
    return int(round(min(v))), int(round(max(v)))


def text_numbers(data: dict) -> dict:
    """The numbers the paragraph beside the figure quotes, each recomputed from the file."""
    rc = rank_correlations(data)
    a = frame(data, "static_anchors")
    m = frame(data, "marks")
    means = m[m["eval"] == R.MEAN_COLUMN].set_index(["judge", "category", "object"]).y
    shift = {cat: float(max(abs(means[("new", cat, o)] - means[("original", cat, o)]) for o in OBJECTS)) for cat in ("over-refusal", "Petri")}
    return {
        "rank_columns": int(len(rc)),
        "rank_min": float(rc.rho.min()),
        "rank_columns_ge_0_70": int((rc.rho >= 0.70 - RHO_TOL).sum()),
        "rank_columns_ge_0_90": int((rc.rho >= 0.90 - RHO_TOL).sum()),
        "static_unsteered_asr_pct": {j: _pct_range(a[a.judge == j].asr) for j in JUDGES},
        "multi_turn_difference_lora_pp": {j: int(round(means[(j, "multi-turn", "difference_lora")])) for j in JUDGES},
        "largest_mean_shift": {cat: round(v, 2) for cat, v in shift.items()},
    }


def summary_text(data: dict) -> str:
    """The lines printed under the figure: the paragraph's numbers, then every column's rank correlation."""
    t = text_numbers(data)
    rc = rank_correlations(data)
    st, mt = t["static_unsteered_asr_pct"], t["multi_turn_difference_lora_pp"]
    lines = [
        f"rank correlation of the interventions between the two judges (Spearman over the objects' mean marks): at least 0.70 in "
        f"{t['rank_columns_ge_0_70']} of {t['rank_columns']} model x evaluation-category columns and at least 0.90 in "
        f"{t['rank_columns_ge_0_90']} (lowest {t['rank_min']:.2f})",
        f"undefended Qwen3.5-4B static attack success: {st['original'][0]}-{st['original'][1]}% under the Claude judge, "
        f"{st['new'][0]}-{st['new'][1]}% under MiniMax-M3",
        f"multi-turn mean, Difference LoRA: {mt['original']:+d} pp under the Claude judge, {mt['new']:+d} pp under MiniMax-M3",
        f"largest change of a mean mark between the judges: over-refusal {t['largest_mean_shift']['over-refusal']:.2f} pp, "
        f"Petri {t['largest_mean_shift']['Petri']:.2f} points",
        "per column (model, category: objects, rho): "
        + "; ".join(f"{R.MODEL_LABEL.get(r.model, r.model)} {r.category}: {r.n}, {r.rho:.2f}" for r in rc.itertuples()),
    ]
    a = frame(data, "static_anchors").pivot(index="eval", columns="judge", values="asr")
    lines.append("unsteered static attack success per attack (Claude / MiniMax-M3, %): "
                 + "; ".join(f"{R.EVAL_LABEL.get(e, e)} {a.loc[e, 'original']:.1f} / {a.loc[e, 'new']:.1f}" for e in R.CATEGORIES["static"][0]))
    return "\n".join(lines)


def provenance_line(data: dict) -> str:
    pv = data["provenance"]
    reg, fr = pv["registry"], pv["frames"]
    return (f"readings: {pathlib.Path(DATA).relative_to(HERE.parents[1])} (payload sha256 {pv['payload_sha256'][:12]}), written by {pv['script']} "
            f"from a frozen registry copy (head {reg['head_in_copy']}) pinned to head {reg['as_of']}, the notebook's load chain at "
            f"{fr['code_commit'][:9]} under REGISTRY_NB_JUDGES={POLICY['original']} and {POLICY['new']}; every render draws the file alone")


# ---- the figure ----------------------------------------------------------------------------------------------------------------------
def _num(x):
    return None if x is None or (isinstance(x, float) and x != x) else float(x)


def judge_labels(data: dict, cats: list) -> dict:
    """The two legend labels of a part, naming only the judges its panels show."""
    m = frame(data, "marks")
    models = {c: sorted(set(m[(m.judge == "new") & (m.category == c) & (m["eval"] != R.MEAN_COLUMN)].judge_model)) for c in cats}
    new = [JUDGE_LABEL[models[c][0]] for c in cats if c != "Petri"]
    petri = [f"Petri: {JUDGE_LABEL[models['Petri'][0]]}"] if "Petri" in cats else []
    return {"original": "original judges (Claude)", "new": f"new judges ({'; '.join(sorted(set(new)) + petri)})"}


def _readings(data: dict) -> dict:
    """{judge: {category: (per-eval rows indexed by (object, seed, eval), mean rows indexed by object)}}"""
    m = frame(data, "marks")
    out = {}
    for j in JUDGES:
        out[j] = {}
        for cat in JUDGED:
            g = m[(m.judge == j) & (m.category == cat)]
            ev = g[g["eval"] != R.MEAN_COLUMN].set_index(["object", "seed", "eval"])
            mn = g[g["eval"] == R.MEAN_COLUMN].set_index("object")
            out[j][cat] = (ev, mn)
    return out


def figure(data: dict, part: str):
    """Draw one part (not saved). Returns (fig, [(ax, category)], info): info carries the drawn values, the readings with no mark and the
    legend's extra keys."""
    spec = PARTS[part]
    layout, height, sz = spec["layout"], spec["height"], spec["marks"]
    cats = [c for row in layout for c in row]
    labels = judge_labels(data, cats)
    reads = _readings(data)
    CAT = {c: R.categories_for(R.MODEL_ID)[c] for c in cats}
    widths = {c: len(CAT[c][0]) + (1 if len(CAT[c][0]) > 1 else 0) for c in cats}
    fig = plt.figure(figsize=(WIDTH, height * len(layout)), layout="constrained")
    axes = R._panel_grid(fig, layout, widths)
    values, missing = [], []
    flags = dict(open=False, ring=False)

    def mark(ax, x, y, lo, hi, o, j, size, bar, gid, open_=False, ring=False):
        if lo is not None and hi is not None:
            ax.plot([x, x], [lo, hi], color=R.COLOR[o], lw=bar, alpha=0.85, solid_capstyle="butt", zorder=2)
        shape = dict(s=size * AREA[j], marker=MARKER[j])
        style = (dict(facecolors="white", edgecolors=R.COLOR[o], linewidths=1.4) if open_ else
                 dict(color=R.COLOR[o], edgecolors="white", linewidths=0.5))
        ax.scatter([x], [y], **shape, zorder=4, gid=gid, **style)
        if ring:
            ax.scatter([x], [y], **shape, facecolors="none", edgecolors="black", linewidths=1.2, zorder=5)

    def join(ax, pts, o, lw):
        if len(pts) == 2:
            ax.plot(*zip(pts["original"], pts["new"]), color=R.COLOR[o], lw=lw, alpha=0.9, zorder=3)

    for ax, cat in axes:
        evals, _sign, ylabel = CAT[cat]
        n = len(evals)
        has_mean = n > 1
        step = 0.82 / len(OBJECTS)
        gap = SEED_HALF * step * 2 / max(len(SEEDS) - 1, 1)              # the distance between two seeds of one slot
        idx = {j: reads[j][cat][0] for j in JUDGES}
        for i, e in enumerate(evals):
            for jo, o in enumerate(OBJECTS):
                xc = i - 0.41 + step * (jo + 0.5)
                seeds = [""] if o in R.DECODE_ONLY else list(SEEDS)
                for xs, sd in zip(R._spread(xc, len(seeds), SEED_HALF * step), seeds):
                    pts = {}
                    for j, dx in (("original", -PAIR_HALF * gap), ("new", PAIR_HALF * gap)):
                        key = (o, sd, e)
                        if key in idx[j].index:
                            r = idx[j].loc[key]
                            lo, hi = _num(r.lo), _num(r.hi)
                            flags["open"] |= bool(r.open)
                            mark(ax, xs + dx, r.y, lo, hi, o, j, sz["seed"], sz["seed_bar"], f"judges|{j}|{o}|{sd or '-'}|{e}", open_=bool(r.open))
                            pts[j] = (xs + dx, r.y)
                            values.append(dict(judge=j, category=cat, eval=e, object=o, seed=sd, y=r.y, lo=lo, hi=hi, x=xs + dx))
                        elif any(key in idx[jj].index for jj in JUDGES):
                            ax.scatter([xs + dx], [0.0], zorder=3, gid=f"judges-missing|{j}|{o}|{sd or '-'}|{e}", **MISSING_MARK)
                            missing.append(dict(judge=j, category=cat, eval=e, object=o, seed=sd))
                    join(ax, pts, o, 0.9)
        if has_mean:
            for jo, o in enumerate(OBJECTS):
                xc, pts = n - 0.41 + step * (jo + 0.5), {}
                for j, dx in (("original", -MEAN_PAIR_HALF * step), ("new", MEAN_PAIR_HALF * step)):
                    mn = reads[j][cat][1]
                    if o in mn.index:
                        r = mn.loc[o]
                        lo, hi = _num(r.lo), _num(r.hi)
                        flags["open"] |= bool(r.open)
                        flags["ring"] |= bool(r.partial)
                        mark(ax, xc + dx, r.y, lo, hi, o, j, sz["mean"], sz["mean_bar"], f"judges|{j}|{o}|mean|{R.MEAN_COLUMN}",
                             open_=bool(r.open), ring=bool(r.partial))
                        pts[j] = (xc + dx, r.y)
                        values.append(dict(judge=j, category=cat, eval=R.MEAN_COLUMN, object=o, seed=r.seed, y=r.y, lo=lo, hi=hi, x=xc + dx))
                    else:
                        ax.scatter([xc + dx], [0.0], zorder=3, gid=f"judges-missing|{j}|{o}|mean|{R.MEAN_COLUMN}", **{**MISSING_MARK, "s": 70})
                        missing.append(dict(judge=j, category=cat, eval=R.MEAN_COLUMN, object=o, seed=""))
                join(ax, pts, o, 1.5)
        for i in range(1, n):
            ax.axvline(i - 0.5, color=R.RULE, lw=1.0)
        if has_mean:
            ax.axvline(n - 0.5, color=R.INK2, lw=1.3)                    # the separator before the mean column
        ax.axhline(0, color=R.INK2, lw=1.1, zorder=1)
        ticks = [R.tick(e) for e in evals] + ([R.MEAN_COLUMN] if has_mean else [])
        ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks)
        if has_mean:
            ax.get_xticklabels()[-1].set_fontweight("bold")
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(lo_, -5), max(hi_, 10))                          # the main figures' minimum range
        ax.set_xlim(-0.6, len(ticks) - 0.4)
        ax.set_title(cat); ax.set_ylabel(R._wrap_label(ylabel))

    # THE LEGEND, TWO ROWS ON TOP: the main figures' object key (five columns: the Refusal steer over the 95% CI key, the two prompted SIs,
    # each corpus's ReLU steer above its LoRA), then the two judges and any extra keys on a row of their own. Objects are colour patches,
    # since the shape carries the judge.
    extra = ([Line2D([], [], ls="", marker="x", ms=9, color=R.MUTED, markeredgewidth=1.6, label="no reading under this judge")] if missing else []) \
        + ([R.OPEN_HANDLE] if flags["open"] else []) + ([R.RING_HANDLE] if flags["ring"] else [])
    patch = lambda o: Patch(facecolor=R.COLOR[o], edgecolor="none", label=R.LABEL[o])
    judge_handle = lambda j, lab: Line2D([], [], ls="", marker=MARKER[j], ms=10, markerfacecolor=R.MUTED, markeredgecolor=R.INK,
                                         markeredgewidth=1.4, label=lab)
    groups = [[patch(OBJECTS[0]), R.CI_HANDLE]] + [[patch(o) for o in OBJECTS[k:k + 2]] for k in range(1, len(OBJECTS), 2)]
    rows = max(len(g) for g in groups)
    obj_handles = [h for g in groups for h in g + [Line2D([], [], ls="", label=" ") for _ in range(rows - len(g))]]
    key = [judge_handle("original", labels["original"]), judge_handle("new", labels["new"])] + extra
    R.legend_stack(fig, [(None, obj_handles, len(groups)), (None, key, len(key))], gap=LEGEND_GAP)
    return fig, axes, dict(values=values, missing=missing, open=flags["open"], ring=flags["ring"], labels=labels)


def _unit(fig, ax) -> float:
    """A laid-out axis's inches per column (its width over its x range); the canvas is drawn first, so constrained layout has run."""
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    return ax.get_window_extent().width / fig.dpi / (x1 - x0)


def draw(data: dict, part: str, show: bool = False):
    """Draw one part at the main figures' type size in a fresh style context and save it (png, pdf, svg) into R.FIGS as
    figA_judge_comparison_<part>. Refuses a canvas where two neighbouring tick labels touch. Returns (fig, info) with the measured
    inches per column of every panel in info["units"]."""
    rc = {k: v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"}
    with plt.rc_context(rc):                                         # the canvas does not depend on what an earlier cell set
        R.style()
        fig, axes, info = figure(data, part)
        info["units"] = {cat: _unit(fig, ax) for ax, cat in axes}
        for ax, cat in axes:
            boxes = [t.get_window_extent() for t in ax.get_xticklabels()]
            for b0, b1 in zip(boxes, boxes[1:]):
                if b0.x1 >= b1.x0:
                    plt.close(fig)
                    raise RuntimeError(f"{PARTS[part]['name']}: {cat}'s tick labels overlap at the notebook's type size")
        R.save(fig, PARTS[part]["name"])
    if not show:
        plt.close(fig)
    return fig, info


def not_drawn(info: dict) -> list:
    """Every reading one judge has and the other lacks (drawn as a grey x at zero); empty for the committed file."""
    return [f"{x['category']} {x['object']} {x['seed'] or '-'} {x['eval']} ({x['judge']} judge)" for x in info["missing"]]
