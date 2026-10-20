"""Figures for the object-axis deck: 4 methods x 9 instructions on
Llama-3.1-8B, all trained on the same pairs.

Methods (deck naming -> data source):
  vector          = stage-1 distill_soft (KL -> additive residual vector)
  rank-1 LoRA     = object_ablation lora (KL -> B(A.x) on layer-15 o_proj)
  SysVec obj.     = stage-1 bidpo (DPO -> additive vector; SysVec's loss)
  Steering Token  = object_ablation token (KL -> soft input token)

Outputs (originals) -> the figure mirror directory object-axis/, per the
mirror rule.
  steering_at_budget.png   compliance of in-budget scales vs prompted
  mmlu_at_max_steering.png MMLU delta at each cell's best-compliance scale
  judge_hypersteer.png     (only if judge reports exist for all four)
  judge_stolfo.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path.home() / "figs" / "si-steering-object-axis"
# the main checkout (the parent of git's common dir), whose out/ holds the runs this deck reads
MAIN = Path(subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--path-format=absolute",
                           "--git-common-dir"], capture_output=True, text=True, check=True).stdout.strip()).parent
S1 = MAIN / "out" / "stage1_llama8b"
OBJ = MAIN / "out" / "object_ablation"
# flags are parsed before anything is created or read, so --help has no side effect and a missing input refuses by name
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--s1-dir", type=Path, default=S1, help=f"stage-1 outputs, written by scripts/stage1_frontier.py (default {S1})")
_ap.add_argument("--obj-dir", type=Path, default=OBJ, help=f"object-ablation outputs, written by scripts/object_ablation.py (default {OBJ})")
_ap.add_argument("--out", type=Path, default=OUT, help=f"figure directory (default {OUT})")
_args = _ap.parse_args()
S1, OBJ, OUT = _args.s1_dir, _args.obj_dir, _args.out
_missing = [f"{flag} {d / 'frontier.json'} (written by scripts/{producer})" for flag, d, producer in
            (("--s1-dir", S1, "stage1_frontier.py"), ("--obj-dir", OBJ, "object_ablation.py"))
            if not (d / "frontier.json").is_file()]
if _missing:
    sys.exit("REFUSED: missing input(s): " + "; ".join(_missing))
OUT.mkdir(parents=True, exist_ok=True)

INSTR = ["french", "lowercase", "concise", "json", "bullets",
         "include_notably", "no_sorry", "pirate", "multi_french_concise"]
METHODS = [  # (label, color, source, key)
    ("vector (KL)", "#2a78d6", "s1", "distill_soft"),
    ("rank-1 LoRA (KL)", "#eb6834", "obj", "lora"),
    ("SysVec objective (DPO)", "#1baf7a", "s1", "bidpo"),
    ("Steering Token (KL)", "#eda100", "obj", "token"),
]
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
SURFACE = "#fcfcfb"

s1_frontier = json.loads((S1 / "frontier.json").read_text())
obj_frontier = json.loads((OBJ / "frontier.json").read_text())


def cell(source: str, key: str, instr: str) -> dict | None:
    f = s1_frontier if source == "s1" else obj_frontier
    return f.get(instr, {}).get(key)


def base_mmlu(source: str, key: str, instr: str) -> float | None:
    if source == "obj":
        d = json.loads((OBJ / instr / key / "results.json").read_text())
        return d["capability"]["unsteered"]["mmlu"]
    for sf in S1.glob("summaries*.json"):
        for s in json.loads(sf.read_text()):
            if s["instruction"] == instr and s["method"] == key:
                return (s.get("capability") or {}).get("unsteered", {}).get("mmlu")
    return None


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TXT2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def dot_panel(values, xlabel, title, fname, xlim, vline=None, prompted=None):
    """values[m][i] -> float | None, lanes dodged per method."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    n = len(INSTR)
    lane = 0.19
    for yi, instr in enumerate(INSTR):
        y0 = n - 1 - yi
        if prompted is not None and prompted[yi] is not None:
            ax.plot([prompted[yi]], [y0], marker="|", ms=16, mew=2.2,
                    color=TXT2, zorder=3)
        for mi, (label, color, *_ ) in enumerate(METHODS):
            v = values[mi][yi]
            y = y0 + lane * (1.5 - mi)
            if v is None:
                ax.plot([xlim[0]], [y], marker="x", ms=5, mew=1.4,
                        color=color, alpha=0.55, zorder=3, clip_on=False)
            else:
                ax.plot([v], [y], "o", ms=7, color=color, zorder=4,
                        mec=SURFACE, mew=1.0)
    if vline is not None:
        ax.axvline(vline, color=TXT2, lw=1.0, ls=":")
    ax.set_yticks(range(n))
    ax.set_yticklabels(reversed(INSTR), fontsize=9, color=TXT)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=TXT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=12, loc="left", pad=34)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, label=l)
               for l, c, *_ in METHODS]
    handles.append(plt.Line2D([], [], marker="|", ls="", ms=10, mew=2.2,
                              color=TXT2, label="prompted"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=TXT,
              columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / fname)


# -- Fig A: steering @ reasonable MMLU (in-budget frontier compliance) -------
valsA, prompted = [], []
for instr in INSTR:
    c = cell("s1", "distill_soft", instr)
    prompted.append(c["prompted"] if c else None)
for label, color, source, key in METHODS:
    row = []
    for instr in INSTR:
        c = cell(source, key, instr)
        fr = c.get("frontier") if c else None
        row.append(fr["compliance"] if fr else None)
    valsA.append(row)
dot_panel(valsA, "strict compliance at the best IN-BUDGET scale (x = none in budget)",
          "Steering at reasonable MMLU (budget: MMLU drop ≤ 2pt, distinct-3gram ≥ 0.35)",
          "steering_at_budget.png", (-0.02, 1.02), prompted=prompted)

# -- Fig B: MMLU drop @ matched vs maximal steering ---------------------------
def mmlu_delta(source, key, instr, mode):
    """mode='matched': smallest scale reaching ~prompted compliance;
    mode='max': the best-compliance scale."""
    c = cell(source, key, instr)
    base = base_mmlu(source, key, instr)
    if not c or base is None:
        return None
    cands = [x for x in c.get("candidates", []) if x.get("mmlu") is not None]
    if not cands:
        return None
    if mode == "max":
        pick = max(cands, key=lambda x: x["compliance"])
    else:
        target = 0.95 * c["prompted"]
        ok = [x for x in cands if x["compliance"] >= target]
        pick = min(ok, key=lambda x: x["scale"]) if ok else None
    return (pick["mmlu"] - base) * 100 if pick else None


def two_panel_b():
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    n = len(INSTR)
    lane = 0.19
    titles = ["at MATCHED steering (smallest scale reaching ≥95% of prompted)",
              "at MAXIMAL steering (the best-compliance scale)"]
    for pi, mode in enumerate(("matched", "max")):
        ax = axes[pi]
        style(ax)
        for yi, instr in enumerate(INSTR):
            y0 = n - 1 - yi
            for mi, (label, color, source, key) in enumerate(METHODS):
                v = mmlu_delta(source, key, instr, mode)
                y = y0 + lane * (1.5 - mi)
                if v is None:
                    ax.plot([-34], [y], marker="x", ms=5, mew=1.4, color=color,
                            alpha=0.55, zorder=3, clip_on=False)
                else:
                    ax.plot([v], [y], "o", ms=7, color=color, zorder=4,
                            mec=SURFACE, mew=1.0)
        ax.axvline(-2.0, color=TXT2, lw=1.0, ls=":")
        ax.set_xlim(-35, 6)
        ax.set_title(titles[pi], color=TXT, fontsize=10.5, loc="left", pad=8)
        ax.set_xlabel("MMLU change (points)", color=TXT, fontsize=10)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(reversed(INSTR), fontsize=9, color=TXT)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, label=l)
               for l, c, *_ in METHODS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 1.0),
               ncol=4, frameon=False, fontsize=8.5, labelcolor=TXT)
    fig.suptitle("Collateral: MMLU cost of steering (dotted = 2pt budget; x = unmeasured)",
                 color=TXT, fontsize=12, x=0.06, y=1.06, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "mmlu_at_max_steering.png", facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "mmlu_at_max_steering.png")


two_panel_b()

# -- Judge figures (only when all four methods are judged) -------------------
j_s1 = S1 / "judge_sonnet.json"
j_obj = OBJ / "judge_sonnet.json"
if j_s1.exists() and j_obj.exists():
    js1 = json.loads(j_s1.read_text())
    jobj = json.loads(j_obj.read_text())

    def jcell(source, key, instr):
        j = js1 if source == "s1" else jobj
        return j.get(instr, {}).get(key if source == "s1" else key)

    # HyperSteer harmonic per arm, mean over the 9 instructions
    arms = ("baseline", "prompted", "steered", "both")
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.xaxis.grid(False)
    width = 0.19
    import numpy as np

    xs = np.arange(len(arms))
    for mi, (label, color, source, key) in enumerate(METHODS):
        means = []
        for arm in arms:
            vals = []
            for instr in INSTR:
                c = jcell(source, key, instr)
                h = ((c or {}).get(arm) or {}).get("hypersteer") or {}
                if h.get("harmonic") is not None:
                    vals.append(h["harmonic"])
            means.append(sum(vals) / len(vals) if vals else 0)
        ax.bar(xs + width * (mi - 1.5), means, width * 0.92, color=color,
               label=label, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, color=TXT)
    ax.set_ylabel("HyperSteer harmonic mean (0–2)", color=TXT, fontsize=10)
    ax.set_title("Sonnet-5 judge, HyperSteer protocol: harmonic({base, steer, fluency})",
                 color=TXT, fontsize=12, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "judge_hypersteer.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "judge_hypersteer.png")

    # Stolfo quality: steered minus prompted, per method (mean +/- SE over instr)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for mi, (label, color, source, key) in enumerate(METHODS):
        deltas = []
        for instr in INSTR:
            c = jcell(source, key, instr)
            qp = ((c or {}).get("prompted") or {}).get("stolfo_quality") or {}
            qs = ((c or {}).get("steered") or {}).get("stolfo_quality") or {}
            if qp.get("mean") is not None and qs.get("mean") is not None:
                deltas.append(qs["mean"] - qp["mean"])
        if deltas:
            m = sum(deltas) / len(deltas)
            se = (sum((d - m) ** 2 for d in deltas) / max(len(deltas) - 1, 1)) ** 0.5 / len(deltas) ** 0.5
            ax.errorbar([m], [3 - mi], xerr=[se], fmt="o", ms=8, color=color,
                        ecolor=color, elinewidth=2, capsize=4, zorder=4)
    ax.axvline(0, color=TXT2, lw=1.0, ls=":")
    ax.set_yticks(range(4))
    ax.set_yticklabels([m[0] for m in reversed(METHODS)], fontsize=9, color=TXT)
    ax.set_xlabel("instruction-blind quality, steered − prompted (mean ± SE over instructions)",
                  color=TXT, fontsize=10)
    ax.set_title("Sonnet-5 judge, Stolfo protocol: collateral quality cost",
                 color=TXT, fontsize=12, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "judge_stolfo.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / "judge_stolfo.png")
else:
    print("judge reports incomplete; skipped judge figures")
