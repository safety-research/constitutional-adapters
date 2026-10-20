#!/usr/bin/env python
"""Figures for the fit-support-transfer experiments — experiment 2.

Reads ``fst_exp2_analysis.json`` (``scripts/fst_exp2_analyze.py``) and renders
one figure per finding, PNG + PDF, into a run-local directory:

1. ``exp2_effect_by_setting``  — effect (steered − none) per setting on the three
   readouts with paired-bootstrap 95 % CIs and setting (1)'s CI drawn as the
   tolerance band per family; collapsed cells marked.
2. ``exp2_asr_degeneracy``     — per cell: ASR@0.5 / @0.75 beside the on-attack
   degeneracy rate (the collapse-reads-as-defense guard, spec fact 4), with the
   collapse rule printed on the figure.
3. ``exp2_parity_s15``         — per-goal §15 (HF target) vs re-run (served)
   ``best_score`` on the diagonal, for the cells that exist in both.
4. ``exp2_goal_heatmap``       — goal × cell ``best_score``; dropped goals shown
   with their ``refused:<code>``.
5. ``exp2_render_check``       — dropped goals by code and refused requests by
   conversation turn (setting (2)).

Style: the dataviz reference palette (light surface, recessive hairline grid,
thin marks, a legend whenever two or more series share an axis, text in ink
tokens never in a series colour). Colour follows the CELL FAMILY (the SI of the
base cell: ``safety_refusal`` blue, ``si_constitution`` orange, …; ``none``
muted) and never a setting's rank, so the same family wears the same hue on
every figure. Every figure carries a stamp: the input sha256 prefixes, the run
identity sha, the GRADER model the analysis verified (code review S-A1), the
render date, and the OUT-OF-SUPPORT / EXPERIMENTAL label. Every paired
quantity is per FAMILY (``none`` + that family's settings — code review S-A3):
the legend carries each family's n, the collapse line is drawn per family.

W&B: the entity and project are the site's own (``--wandb-entity`` /
``--wandb-project``, else ``WANDB_ENTITY`` / ``WANDB_PROJECT``, else the site
config's ``SISTEER_WANDB_ENTITY`` / ``SISTEER_WANDB_PROJECT``); with either
missing, W&B logging refuses by name before any figure is drawn. Run
``fst-exp2-<run>``; one ``wandb.Image`` per figure AND an artifact of the
figure directory. ``--no-wandb`` skips it entirely (tests never contact W&B;
``tests/conftest.py`` also sets ``WANDB_MODE=disabled``). The API key is never
read here — the run-time environment carries it.

``--mirror`` rsyncs the figure dir to ``~/figs/fst-exp2/<run>/`` (a local
figure mirror); ``--mirror-print-only`` prints the exact command instead of
running it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

#: the analysis file this renderer reads (fst_exp2_analyze.SCHEMA)
ANALYSIS_SCHEMA = "fst_exp2_analysis/v3"
#: where the W&B destination is read, in order, after the command-line flags (site values; no default)
WANDB_ENTITY_ENV = ("WANDB_ENTITY", "SISTEER_WANDB_ENTITY")
WANDB_PROJECT_ENV = ("WANDB_PROJECT", "SISTEER_WANDB_PROJECT")
MIRROR_ROOT = Path("~/figs/fst-exp2")

# ---- dataviz reference palette (light mode; see the dataviz skill) ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS_CRITICAL = "#d03b3b"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: fixed family -> slot assignment (never cycled; an unknown family takes the
#: next free slot in sorted order, deterministic across runs)
FAMILY_SLOT = {"safety_refusal": 0, "si_constitution": 1, "si_irrelevant": 2}
READOUTS = ("asr@0.5", "asr@0.75", "best_score")
READOUT_TITLE = {"asr@0.5": "ASR@0.5 (fraction of goals)",
                 "asr@0.75": "ASR@0.75 (fraction of goals)",
                 "best_score": "mean best StrongREJECT score"}
SETTING_SHORT = {0: "unsteered baseline", 1: "(1) last turn", 2: "(2) all turns", 3: "(3) all positions"}
OOS = "OUT-OF-SUPPORT / EXPERIMENTAL: settings (2) and (3)"


def family_colours(analysis: dict) -> dict[str, str]:
    fams = sorted({c["family"] for c in analysis["cells"].values()} - {"none"})
    taken = {FAMILY_SLOT[f] for f in fams if f in FAMILY_SLOT}
    free = [i for i in range(len(SERIES)) if i not in taken]
    out = {"none": MUTED}
    for f in fams:
        slot = FAMILY_SLOT.get(f)
        if slot is None:
            if not free:
                raise SystemExit("[fst_figs] more than eight cell families — fold "
                                 "or facet, never generate a ninth hue")
            slot = free.pop(0)
        out[f] = SERIES[slot]
    return out


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE_AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _stamp_text(analysis: dict, run: str, date: str) -> str:
    ins = analysis["inputs"]
    parts = [f"{k} {v['sha256'][:8]}" for k, v in sorted(ins.items())]
    ri = analysis.get("run_identity") or {}
    ident = (ri.get("sha256") or "none")[:12]
    grader = (analysis.get("grader") or {}).get("model") or "unknown"
    return (f"run {run} · identity {ident} · grader {grader} · inputs: " + " · ".join(parts)
            + f" · rendered {date} · {OOS}")


def _fam_list(fams: list[str]) -> str:
    return ", ".join(fams)


def _save(fig, figs_dir: Path, stem: str, stamp: str, figures: list) -> None:
    fig.text(0.01, 0.004, stamp, fontsize=6, color=MUTED, ha="left", va="bottom")
    figs_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"{stem}.{ext}", dpi=160, facecolor=SURFACE,
                    bbox_inches="tight")
    figures.append((stem, figs_dir / f"{stem}.png"))
    plt.close(fig)


def _ordered_cells(analysis: dict) -> list[str]:
    cells = analysis["cells"]
    return sorted(cells, key=lambda k: (cells[k]["family"] != "none", cells[k]["family"],
                                        cells[k]["setting"]))


# ---------------------------------------------------------------- figures


def fig_effect_by_setting(analysis: dict, colours: dict, figs_dir: Path, stamp: str,
                          figures: list) -> None:
    cells, effects, bands = analysis["cells"], analysis["effects"], analysis["tolerance_bands"]
    sat_bf = analysis["saturation"]["by_family"]
    pbf = analysis["paired"]["by_family"]
    fams = sorted({cells[c]["family"] for c in effects})
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(4.4 * len(READOUTS), 4.6), sharey=False)
    width = 0.8 / max(1, len(fams))
    for ax, ro in zip(axes, READOUTS):
        # saturation and the primary readout are PER FAMILY (S-A3): none's
        # baseline is read on each family's own paired goals
        sat_fams = [f for f in fams if sat_bf[f]["by_readout"][ro]["saturated"]]
        prim_fams = [f for f in fams if sat_bf[f]["primary_readout"] == ro]
        for fi, fam in enumerate(fams):
            col = colours[fam]
            fade = 0.35 if fam in sat_fams else 1.0
            offs = (fi - (len(fams) - 1) / 2) * width
            band = bands.get(fam, {}).get(ro)
            if band is not None:
                ax.fill_between([0.5 + offs - width / 2, 3.5 + offs + width / 2],
                                band["lo"], band["hi"], color=col, alpha=0.12, linewidth=0)
            for cid in sorted(effects, key=lambda k: cells[k]["setting"]):
                c = cells[cid]
                if c["family"] != fam:
                    continue
                e = effects[cid][ro]
                x = c["setting"] + offs
                collapsed = c["degeneracy"]["collapsed"]      # True / False / None (E3)
                unknown = collapsed is None
                ax.errorbar([x], [e["effect"]],
                            yerr=[[e["effect"] - e["ci_low"]], [e["ci_high"] - e["effect"]]],
                            fmt=("s" if unknown else "o"), color=col, ecolor=col, elinewidth=2,
                            capsize=3, markersize=7,
                            markerfacecolor=(SURFACE if (collapsed or unknown) else col),
                            markeredgewidth=2, alpha=fade)
                if collapsed:
                    ax.annotate("collapsed", (x, e["ci_high"]), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=7, color=STATUS_CRITICAL)
                elif unknown:
                    ax.annotate("degeneracy unknown", (x, e["ci_high"]),
                                textcoords="offset points", xytext=(0, 5), ha="center",
                                fontsize=7, color=STATUS_CRITICAL)
        ax.axhline(0, color=BASELINE_AXIS, linewidth=1)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([SETTING_SHORT[1], SETTING_SHORT[2], SETTING_SHORT[3]], fontsize=8,
                           color=INK_2)
        ax.set_xlim(0.4, 3.6)
        ax.set_ylabel("steered − none", color=INK_2, fontsize=9)
        if sat_fams:
            every = len(sat_fams) == len(fams)
            _style(ax, READOUT_TITLE[ro] + (" — SATURATED, no verdict" if every else
                                            f" — SATURATED for {_fam_list(sat_fams)}"))
            lines = [f"{f}: none = "
                     f"{100 * sat_bf[f]['by_readout'][ro]['none_mean_on_paired_goals']:.0f}% "
                     f"on its {sat_bf[f]['n_paired_goals']} paired goals" for f in sat_fams]
            ax.text(0.5, 0.5, "\n".join(lines) + "\n(saturated: no transfer verdict)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
                    color=STATUS_CRITICAL)
        elif prim_fams:
            every = len(prim_fams) == len(fams)
            _style(ax, READOUT_TITLE[ro] + (" — PRIMARY readout" if every else
                                            f" — PRIMARY for {_fam_list(prim_fams)}"))
        else:
            _style(ax, READOUT_TITLE[ro])
    handles = [Patch(facecolor=colours[f], label=f"{f} (n = {pbf[f]['n']} paired goals)")
               for f in fams]
    handles.append(Patch(facecolor=MUTED, alpha=0.2, label="setting (1) 95% CI = tolerance band"))
    fig.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), ncol=len(handles))
    fig.suptitle("Experiment 2 — steering effect by setting (paired bootstrap 95% CI over each "
                 "family's paired goals, n in the legend; hollow circle = collapsed, hollow "
                 "square = degeneracy unknown)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save(fig, figs_dir, "exp2_effect_by_setting", stamp, figures)


def _rate_label(rate, k: int, n: int, n_responses: int, tag: str) -> str:
    if rate is None:
        return f"{tag} unavailable (0 scorable of {n_responses} responses)"
    return f"{tag} {100 * rate:.1f}% ({k}/{n})"


def degeneracy_bar_label(d: dict) -> str:
    """The text beside a cell's degeneracy bar: the paired-goal rate (the
    collapse rule's, over the cell's FAMILY paired set) and the all-graded
    rate, each with its k/n denominator; ``none`` — no family of its own —
    names each family's paired set with its n (S-A3); a null rate reads
    'unavailable', never 0.0% (code review E3); the collapse state is spelled
    out (COLLAPSED / DEGENERACY UNKNOWN)."""
    dp = d["paired"]
    if dp is None:
        parts = [_rate_label(v["rate_distinct3"], v["n_degenerate_distinct3"],
                             v["n_scored_distinct3"], v["n_responses"],
                             f"paired[{fam} n={v['n_goals']}]")
                 for fam, v in sorted(d["paired_by_family"].items())]
    else:
        parts = [_rate_label(dp["rate_distinct3"], dp["n_degenerate_distinct3"],
                             dp["n_scored_distinct3"], dp["n_responses"], "paired")]
    parts.append(_rate_label(d["rate_distinct3"], d["n_degenerate_distinct3"],
                             d["n_scored_distinct3"], d["n_responses"], "all graded"))
    label = ("\n" if dp is None else " · ").join(parts)
    if d["collapsed"] is None:
        return label + "  DEGENERACY UNKNOWN"
    return label + ("  COLLAPSED" if d["collapsed"] else "")


def fig_asr_degeneracy(analysis: dict, colours: dict, figs_dir: Path, stamp: str,
                       figures: list) -> None:
    cells = analysis["cells"]
    order = _ordered_cells(analysis)
    y = np.arange(len(order))
    rule = analysis["degeneracy_rule"]
    by_fam = rule["by_family"]
    fams = sorted(by_fam)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 0.55 * len(order) + 2.6), sharey=True)
    h = 0.36
    for i, cid in enumerate(order):
        c = cells[cid]
        col = colours[c["family"]]
        for k, (t, alpha) in enumerate((("0.5", 0.45), ("0.75", 1.0))):
            if c["asr"][t] is not None:
                ax1.barh(y[i] + (k - 0.5) * h, c["asr"][t], height=h - 0.04, color=col, alpha=alpha)
        d = c["degeneracy"]
        dp = d["paired"]
        # the BAR is the rate the collapse rule reads: over the cell's FAMILY
        # paired goals, the denominator none shares (S-A3); the all-graded
        # descriptive rate is the open marker. A null rate draws NO bar (E3:
        # unavailable is not 0%). none has no family: one thin bar per family.
        if dp is None:
            pbf = sorted(d["paired_by_family"].items())
            hh = (2 * h - 0.04) / max(1, len(pbf))
            rates = [v["rate_distinct3"] for _, v in pbf if v["rate_distinct3"] is not None]
            rate = max(rates) if rates else None
            for k, (fam, v) in enumerate(pbf):
                if v["rate_distinct3"] is not None:
                    ax2.barh(y[i] - (2 * h - 0.04) / 2 + hh * (k + 0.5), v["rate_distinct3"],
                             height=hh - 0.01, color=colours[fam], alpha=0.6, edgecolor=SURFACE,
                             linewidth=0.5)
        else:
            rate = dp["rate_distinct3"]
            if rate is not None:
                ax2.barh(y[i], rate, height=2 * h - 0.04, color=col,
                         hatch=("////" if d["collapsed"] else None), edgecolor=SURFACE,
                         linewidth=0.5)
        if d["rate_distinct3"] is not None:
            ax2.plot([d["rate_distinct3"]], [y[i]], marker="o", markerfacecolor=SURFACE,
                     markeredgecolor=INK_2, markersize=6, markeredgewidth=1.2, linestyle="none")
        if d["rate_classify"] is not None:
            ax2.plot([d["rate_classify"]], [y[i]], marker="D", color=INK_2, markersize=5,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
        flagged = d["collapsed"] is None or bool(d["collapsed"])
        ax2.text(min(max(rate or 0.0, d["rate_distinct3"] or 0.0) + 0.02, 0.62), y[i],
                 degeneracy_bar_label(d), va="center", fontsize=7,
                 color=(STATUS_CRITICAL if flagged else INK_2),
                 bbox={"facecolor": SURFACE, "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
    # the collapse line is PER FAMILY (S-A3): none's rate on that family's
    # paired goals + the rise, drawn across that family's rows only
    thr = rule["collapse_rise_threshold"]
    line_labels: list[str] = []
    for fam in fams:
        rows = [y[i] for i, cid in enumerate(order) if cells[cid]["family"] == fam]
        if not rows:
            continue
        none_rate = by_fam[fam]["none_rate_distinct3_on_paired_goals"]
        n_fam = by_fam[fam]["n_paired_goals"]
        if none_rate is not None:
            x = none_rate + thr
            ax2.plot([x, x], [min(rows) - 0.45, max(rows) + 0.45], color=colours[fam],
                     linewidth=1.2, linestyle=":")
            line_labels.append(f"collapse line {fam}: none on its {n_fam} paired goals "
                               f"({100 * none_rate:.1f}%) + {thr:.2f}")
        else:
            ax2.text(0.5, (min(rows) + max(rows)) / 2,
                     f"collapse line unavailable ({fam}): none has no scorable response on "
                     f"its {n_fam} paired goals", ha="center", va="center", fontsize=7.5,
                     color=STATUS_CRITICAL)
            line_labels.append(f"collapse line {fam}: unavailable (none unscorable on its "
                               f"{n_fam} paired goals)")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{cid}\n{SETTING_SHORT[cells[cid]['setting']]}" for cid in order],
                        fontsize=7.5, color=INK_2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.0)
    ax2.set_xlim(0, 1.0)
    ax1.set_xlabel("attack success rate (fraction of graded goals)", color=INK_2, fontsize=9)
    ax2.set_xlabel("degeneracy rate (distinct-3 < 0.25); bar = family paired goals, ○ = all graded",
                   color=INK_2, fontsize=8.5)
    _style(ax1, "ASR@0.5 (light) and ASR@0.75 (solid) per cell")
    _style(ax2, "degeneracy beside ASR — a collapsed cell is not a defense")
    for ax in (ax1, ax2):
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.grid(False, axis="y")
    handles = [Patch(facecolor=colours[f], label=f"{f} (n = {by_fam[f]['n_paired_goals']} "
                                                  "paired goals)") for f in fams]
    handles += [Patch(facecolor=MUTED, alpha=0.45, label="ASR@0.5"),
                Patch(facecolor=MUTED, label="ASR@0.75"),
                Patch(facecolor=MUTED, label="degeneracy on the family's paired goals (bar)"),
                plt.Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                           linestyle="none", label="degeneracy, all graded goals (descriptive)"),
                plt.Line2D([], [], marker="D", color=INK_2, linestyle="none",
                           label="degeneracy.classify flag rate (all graded)")]
    handles += [plt.Line2D([], [], color=colours[f], linestyle=":", label=lbl)
                for f, lbl in zip(fams, line_labels)]
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(handles), 4))
    fig.suptitle("Experiment 2 — ASR beside on-attack degeneracy (collapsed iff "
                 f"{rule['collapsed_iff']})", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    _save(fig, figs_dir, "exp2_asr_degeneracy", stamp, figures)


def fig_parity_s15(analysis: dict, colours: dict, figs_dir: Path, stamp: str,
                   figures: list) -> None:
    cells, parity = analysis["cells"], analysis["parity_s15"]
    compared = [c for c in _ordered_cells(analysis) if parity.get(c, {}).get("status") == "compared"]
    fig, axes = plt.subplots(1, max(1, len(compared)), figsize=(4.4 * max(1, len(compared)), 4.4),
                             squeeze=False)
    axes = list(axes[0])
    if not compared:
        ax = axes[0]
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.5, "no cell shared with the §15 wave", ha="center", va="center",
                color=INK_2, fontsize=10)
    for ax, cid in zip(axes, compared):
        p = parity[cid]
        col = colours[cells[cid]["family"]]
        gids = sorted(p["per_goal"])
        xs = np.array([p["per_goal"][g]["s15_best_score"] for g in gids])
        ys = np.array([p["per_goal"][g]["rerun_best_score"] for g in gids])
        jit = (np.arange(len(gids)) % 5 - 2) * 0.004  # deterministic de-overlap
        ax.plot([0, 1], [0, 1], color=BASELINE_AXIS, linewidth=1)
        ax.scatter(xs + jit, ys - jit, s=42, color=col, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
        st = p["sign_test"]
        pv = "n/a" if st["p_value"] is None else f"{st['p_value']:.2f}"
        ax.text(0.02, 0.98,
                f"n = {p['n_goals']} goals\nmean best §15 {p['mean_best_s15']:.3f} → "
                f"re-run {p['mean_best_rerun']:.3f}\nsign test p = {pv} "
                f"(+{st['n_pos']} / −{st['n_neg']} / ={st['n_zero']})\n"
                f"ASR@0.75 §15 {100 * p['asr@0.75']['s15']:.0f}% → re-run "
                f"{100 * p['asr@0.75']['rerun']:.0f}%",
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK_2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("§15 best_score (HF target, 87-goal wave)", color=INK_2, fontsize=9)
        ax.set_ylabel("re-run best_score (served target)", color=INK_2, fontsize=9)
        _style(ax, f"{cid} — {SETTING_SHORT[cells[cid]['setting']]}")
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    fig.suptitle("Experiment 2 — serving parity: setting (1) and none vs their §15 values, "
                 "per goal", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _save(fig, figs_dir, "exp2_parity_s15", stamp, figures)


def fig_goal_heatmap(analysis: dict, colours: dict, figs_dir: Path, stamp: str,
                     figures: list) -> None:
    cells = analysis["cells"]
    order = _ordered_cells(analysis)
    goals = analysis["goal_ids"]
    M = np.full((len(goals), len(order)), np.nan)
    for j, cid in enumerate(order):
        for i, g in enumerate(goals):
            pg = cells[cid]["per_goal"].get(g)
            if pg is not None:
                M[i, j] = pg["best_score"]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blues", SEQ_BLUES)
    cmap.set_bad(GRID)
    fig, ax = plt.subplots(figsize=(1.3 * len(order) + 3, 0.42 * len(goals) + 2.2))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    dropped = analysis["paired"]["dropped"]
    for j, cid in enumerate(order):
        for i, g in enumerate(goals):
            if np.isnan(M[i, j]):
                reason = dropped.get(cid, {}).get(g, "dropped")
                ax.text(j, i, reason.replace("refused:all_turns_", "refused:\n"),
                        ha="center", va="center", fontsize=5.5, color=INK_2)
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=(SURFACE if M[i, j] > 0.6 else INK))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{cid}\n{SETTING_SHORT[cells[cid]['setting']]}" for cid in order],
                       fontsize=7, color=INK_2, rotation=20, ha="right")
    ax.set_yticks(range(len(goals)))
    ax.set_yticklabels([g[:8] + "…" if len(g) > 9 else g for g in goals], fontsize=7,
                       color=INK_2)
    ax.set_facecolor(SURFACE)
    ax.set_title("Experiment 2 — best StrongREJECT score per goal × cell (grey = dropped)",
                 color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("best_score", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, figs_dir, "exp2_goal_heatmap", stamp, figures)


def fig_render_check(analysis: dict, colours: dict, figs_dir: Path, stamp: str,
                     figures: list) -> None:
    cells = analysis["cells"]
    order = [c for c in _ordered_cells(analysis)
             if cells[c]["render_check"]["n_goals_dropped"]
             or cells[c]["render_check"]["refused_requests_by_code"]]
    codes = sorted({code for c in order
                    for code in list(cells[c]["render_check"]["dropped_goals_by_code"])
                    + list(cells[c]["render_check"]["refused_requests_by_code"])})
    code_col = {code: SERIES[i % len(SERIES)] for i, code in enumerate(codes)}  # ≤ 4 codes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if not order:
        for ax in (ax1, ax2):
            ax.set_facecolor(SURFACE)
            ax.axis("off")
        ax1.text(0.5, 0.5, "no render-check refusals — every setting-(2) request rendered "
                 "consistently", ha="center", va="center", color=INK_2, fontsize=10)
    else:
        x = np.arange(len(order))
        w = min(0.8 / max(1, len(codes)), 0.35)
        for k, code in enumerate(codes):
            offs = (k - (len(codes) - 1) / 2) * w
            ax1.bar(x + offs, [cells[c]["render_check"]["dropped_goals_by_code"].get(code, 0)
                               for c in order], width=w - 0.03, color=code_col[code])
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{c}\n{SETTING_SHORT[cells[c]['setting']]}" for c in order],
                            fontsize=7.5, color=INK_2)
        ax1.set_xlim(-0.7, len(order) - 0.3)
        ax1.set_ylabel("goals dropped (latched refused:<code>)", color=INK_2, fontsize=9)
        ax1.yaxis.get_major_locator().set_params(integer=True)
        _style(ax1, "dropped goals by code")
        turns = sorted({int(t) for c in order
                        for v in cells[c]["render_check"]["refused_requests_by_code_and_turn"].values()
                        for t in v if t != "unknown"})
        bottoms = np.zeros(len(turns))
        for code in codes:
            vals = np.array([sum(cells[c]["render_check"]["refused_requests_by_code_and_turn"]
                                 .get(code, {}).get(str(t), 0) for c in order) for t in turns],
                            dtype=float)
            ax2.bar([str(t) for t in turns], vals, bottom=bottoms, color=code_col[code],
                    edgecolor=SURFACE, linewidth=1, width=0.35)
            bottoms += vals
        ax2.set_xlim(-0.7, len(turns) - 0.3)
        ax2.set_xlabel("conversation turn ((n_messages + 1) // 2)", color=INK_2, fontsize=9)
        ax2.set_ylabel("refused requests (timing rows)", color=INK_2, fontsize=9)
        ax2.yaxis.get_major_locator().set_params(integer=True)
        _style(ax2, "refused requests by conversation turn, stacked by code")
        handles = [Patch(facecolor=code_col[c], label=c) for c in codes]
        fig.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 0.025), ncol=len(handles))
    fig.suptitle("Experiment 2 — the setting-(2) render check: message-derived vs located "
                 "spans (mismatch drops the goal; counted, never guessed)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    _save(fig, figs_dir, "exp2_render_check", stamp, figures)


def render_all(analysis: dict, figs_dir: Path, *, run: str, date: str) -> list[tuple[str, Path]]:
    colours = family_colours(analysis)
    stamp = _stamp_text(analysis, run, date)
    figures: list[tuple[str, Path]] = []
    for fn in (fig_effect_by_setting, fig_asr_degeneracy, fig_parity_s15,
               fig_goal_heatmap, fig_render_check):
        fn(analysis, colours, figs_dir, stamp, figures)
    return figures


# ------------------------------------------------------------- W&B / mirror


def log_wandb(figures: list[tuple[str, Path]], figs_dir: Path, *, run: str, mode: str,
              project: str, entity: str, config: dict) -> None:
    import wandb

    wb = wandb.init(project=project, entity=entity, name=f"fst-exp2-{run}",
                    job_type="analysis", mode=mode, config=config)
    for stem, path in figures:
        wandb.log({f"exp2/{stem}": wandb.Image(str(path))})
    art = wandb.Artifact(f"figures-fst-exp2-{run}", type="figures")
    art.add_dir(str(figs_dir))
    wandb.log_artifact(art)
    wb.finish()


def wandb_destination(entity: str | None, project: str | None,
                      env: dict | None = None) -> tuple[str, str]:
    """The (entity, project) W&B logs to: each flag, else the first non-empty
    variable of its ``*_ENV`` list. Either missing refuses by name — a site value
    is never guessed, and wandb's implicit default entity is never used."""
    env = os.environ if env is None else env

    def pick(flag: str | None, names: tuple[str, ...]) -> str:
        for v in (flag, *(env.get(n) for n in names)):
            if v and v.strip():
                return v.strip()
        return ""

    ent, proj = pick(entity, WANDB_ENTITY_ENV), pick(project, WANDB_PROJECT_ENV)
    missing = [n for n, v in (("an entity", ent), ("a project", proj)) if not v]
    if missing:
        raise SystemExit(
            f"[fst_figs] W&B logging needs {' and '.join(missing)}: pass --wandb-entity / "
            f"--wandb-project, or set {' / '.join(WANDB_ENTITY_ENV)} and {' / '.join(WANDB_PROJECT_ENV)} "
            "(or pass --no-wandb)")
    return ent, proj


def mirror_command(figs_dir: Path, run: str, dest_root: Path = MIRROR_ROOT) -> list[str]:
    dest = Path(os.path.expanduser(str(dest_root))) / run
    return ["rsync", "-avz", "--prune-empty-dirs", "--include=*/", "--include=*.png",
            "--include=*.pdf", "--exclude=*", f"{figs_dir}/", f"{dest}/"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, help="fst_exp2_analysis.json")
    ap.add_argument("--figs-dir", required=True, help="run-local figure directory")
    ap.add_argument("--run", required=True, help="the run name (fst-exp2-<run> on W&B)")
    ap.add_argument("--stamp-date", default=None,
                    help="date on the stamp (default: today, UTC)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    ap.add_argument("--wandb-project", default=None,
                    help="default: WANDB_PROJECT, else SISTEER_WANDB_PROJECT")
    ap.add_argument("--wandb-entity", default=None,
                    help="default: WANDB_ENTITY, else SISTEER_WANDB_ENTITY")
    ap.add_argument("--mirror", action="store_true",
                    help=f"rsync the figure dir to {MIRROR_ROOT}/<run>/")
    ap.add_argument("--mirror-print-only", action="store_true",
                    help="print the mirror command instead of running it")
    ap.add_argument("--mirror-dest", default=str(MIRROR_ROOT))
    return ap


def main(argv: list[str] | None = None) -> list[tuple[str, Path]]:
    args = build_parser().parse_args(argv)
    if not args.no_wandb:
        args.wandb_entity, args.wandb_project = wandb_destination(args.wandb_entity, args.wandb_project)
    analysis = json.loads(Path(args.analysis).read_text())
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise SystemExit(f"[fst_figs] {args.analysis}: not a {ANALYSIS_SCHEMA} file "
                         f"(schema {analysis.get('schema')!r})")
    date = args.stamp_date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    figs_dir = Path(args.figs_dir)
    figures = render_all(analysis, figs_dir, run=args.run, date=date)
    for stem, path in figures:
        print(f"[fst_figs] {path} (+ .pdf)", flush=True)
    if not args.no_wandb:
        config = {"run": args.run, "inputs": analysis["inputs"],
                  "run_identity_sha256": (analysis.get("run_identity") or {}).get("sha256"),
                  "n_goals": analysis["n_goals"], "cells": sorted(analysis["cells"])}
        log_wandb(figures, figs_dir, run=args.run, mode=args.wandb_mode,
                  project=args.wandb_project, entity=args.wandb_entity, config=config)
        print(f"[fst_figs] logged {len(figures)} figures + artifact to W&B "
              f"{args.wandb_entity}/{args.wandb_project} run fst-exp2-{args.run} "
              f"(mode={args.wandb_mode})", flush=True)
    if args.mirror or args.mirror_print_only:
        cmd = mirror_command(figs_dir, args.run, Path(args.mirror_dest))
        print("[fst_figs] mirror: " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if args.mirror and not args.mirror_print_only:
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)
    return figures


if __name__ == "__main__":
    main()
