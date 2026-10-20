#!/usr/bin/env python
"""defense_figs.py — the AlphaSteer defence wave's dose curves, drawn from alphasteer_defense_curves.json.

    python scripts/alphasteer_f22/defense_figs.py \
        --curves $GRADE/alphasteer_f22_defense/alphasteer_defense_curves.json \
        --out-dir <out-dir>

READS THE CURVE RECORD AND NOTHING ELSE. Every number on every axis comes from a field the curve
builder wrote; nothing here re-derives a rate, re-reads a store, or fills a gap. A rung the record
calls `missing` or `not served` is DRAWN AS ABSENT (an open marker on the dose axis with the reason
in the caption), never interpolated and never silently dropped — a reader must be able to tell "we
measured zero effect" from "we have not measured this yet".

THREE PANELS PER MODEL, sharing one dose axis so the same λ lines up across instruments:
  * DEFENCE — mean StrongREJECT defense@0.5 over the four static families, steered minus the
    same-standup unsteered anchor, in percentage points.
  * BEST-OF-N — the normalized-AUC delta vs that anchor. NEGATIVE IS GOOD here (less attack
    success under a 75-draw best-of-N), so the axis is inverted to keep "up = better defended"
    true on every panel. The two instruments are NEVER pooled: they are separate estimands and
    the figure says so in its own legend rather than in a caption nobody reads.
  * DEGENERATION — the steered-minus-anchor delta of the worst of the five detectors, which is
    the honesty panel. A defence number is only worth reading where this is near zero: a model
    that loops, hits the cap, or emits nothing scores as "defended" by construction. ABSOLUTE
    shares are deliberately NOT plotted: nemotron's grok420 baseline carries a ~0.2 loop share at
    EVERY dose including the sign control, so the absolute reading brands its whole ladder
    contaminated while the dose-induced delta is ~0.02 at the bottom three rungs.

THE SIGN CONTROL IS A POINT ON THE AXIS, not a separate chart: it sits at -1x the smallest positive
λ and is what makes the curve a dose-response rather than a pair of conditions. It is drawn with a
distinct open marker so it cannot be mistaken for a rung, and it is never averaged into anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util as _ilu
# THE ROW A PANEL DRAWS IS CHOSEN BY THE SAME FUNCTION THE OPERATING-POINT RECORD USES. Two copies of
# "which decode do we read" would drift the moment one of them was re-ruled (per-lane copies propagate
# defects); one selector, loaded from the sibling file, cannot.
_OP_SPEC = _ilu.spec_from_file_location("_oppoints_for_figs", Path(__file__).with_name("operating_points.py"))
_op = _ilu.module_from_spec(_OP_SPEC)
_OP_SPEC.loader.exec_module(_op)
select_curve_row = _op.select_curve_row

DETECTORS = ("classifier_share", "at_cap_share", "zlib_collapse_share",
             "contentless_share", "rep_ratio_loop_share")
INK, MUTED, RULE = "#1f1f1d", "#9a9994", "#e3e2de"
C_DEF, C_BON, C_DEG = "#4a5f8a", "#3f6b4a", "#a8462a"
#: above this steered-minus-anchor delta the defence number on that row is not readable as defence
CONTAM = 0.10


def apply_style() -> None:
    matplotlib.rcParams.update({
        "axes.linewidth": 1.9, "xtick.major.width": 1.9, "ytick.major.width": 1.9,
        "xtick.major.size": 7, "ytick.major.size": 7, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.titlesize": 16,
        "legend.fontsize": 12.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "savefig.dpi": 200,
        "pdf.fonttype": 42, "lines.solid_capstyle": "round", "font.size": 14,
    })


def worst_degeneration(static: dict) -> tuple[float | None, str]:
    """The largest STEERED-MINUS-ANCHOR share over (family x detector), and which one it was.

    The subtraction is the whole point: an absolute share mixes the model's own baseline
    pathology on a family into the dose's effect, and on nemotron's grok420 the baseline alone
    is ~0.2 at every dose. Returns (None, "") when the record carries no degeneration block."""
    arm = (static.get("degeneration") or {}).get("alphasteer") or {}
    non = (static.get("degeneration") or {}).get("none") or {}
    if not arm:
        return None, ""
    worst, which = 0.0, ""
    for family, dets in arm.items():
        for det in DETECTORS:
            a = ((dets.get(det) or {}).get("share") or 0.0)
            n = (((non.get(family) or {}).get(det) or {}).get("share") or 0.0)
            if a - n > worst:
                worst, which = a - n, f"{family}/{det.replace('_share', '')}"
    return worst, which


def rows_of(curves: dict, target: str) -> list[dict]:
    """The target's dose rows in DOSE ORDER, one per cell, chosen under the decode ruling.

    A target with two decode arms (nemotron: top_p 0.95 and 1.0) carries each rung twice, and the
    two are SEPARATE instruments — `never_pool` says so on every row. Until 2026-09-14 this drew the
    dose-consistent (0.95) arm throughout. The rule since then is top_p 1.0 wherever it exists, 0.95
    where it doesn't, applied PER INSTRUMENT: static defence and degeneration exist at 1.0 and are
    drawn there; best-of-N exists at 0.95 only and is drawn there. `select_curve_row` composes that
    row and flags it `composite`; the panel then says so on its face rather than in a caption."""
    block = curves["targets"][target]
    out = []
    sign = select_curve_row(curves, target, None, sign_control=True)
    if sign is not None:
        out.append(sign)
    for i in sorted({r["rung_i"] for r in block.get("rows", [])
                     if r.get("rung_i") is not None and not r.get("sign_control")}):
        r = select_curve_row(curves, target, i)
        if r is not None:
            out.append(r)
    return sorted(out, key=lambda r: r["lambda"])


def decode_split_note(rows: list[dict]) -> str | None:
    """One line for a panel whose halves are at two decodes, or None when they are not."""
    comp = [r["decode_selection"] for r in rows if (r.get("decode_selection") or {}).get("composite")]
    if not comp:
        return None
    sel = comp[0]
    return (f"defence & degeneration @ top_p {sel['static_top_p']} · best-of-N @ top_p {sel['bon_top_p']}")


def draw(curves: dict, out_dir: Path) -> list[Path]:
    apply_style()
    targets = list(curves["targets"])
    # sharey="row" ON PURPOSE: the four columns are the figure's comparison. Per-model autoscaling
    # drew gemma's +0.91 degeneration and nemotron's +0.17 at the same apparent height and put the
    # 0.10 contamination line in four different places, which is the one reading this panel exists
    # to prevent. Shared, gemma's collapse dominates the row — which is the true statement.
    fig, axes = plt.subplots(3, len(targets), figsize=(4.5 * len(targets), 11.4),
                             sharex="col", sharey="row", squeeze=False)
    written: list[Path] = []
    n_absent = 0

    split_targets: list[str] = []
    for col, target in enumerate(targets):
        rows = rows_of(curves, target)
        ax_def, ax_bon, ax_deg = axes[0][col], axes[1][col], axes[2][col]
        note = decode_split_note(rows)
        if note:
            split_targets.append(target)
            # ON THE PANEL, not in the caption: a reader comparing nemotron's defence point with its
            # best-of-N point must see that they are at different samplers where they look.
            ax_def.text(0.03, 0.96, note, transform=ax_def.transAxes, ha="left", va="top",
                        fontsize=9.5, color=MUTED, zorder=5)

        def_x, def_y, sign_pt = [], [], None
        bon_x, bon_y, bon_sign = [], [], None
        deg_x, deg_y, deg_sign = [], [], None
        absent_x = []
        absent_def, absent_bon, absent_deg = [], [], []

        for r in rows:
            lam = r["lambda"]
            if r["status"].startswith("not served"):
                continue                      # off the served grid by ruling; the caption says so
            static = r.get("static") or {}
            dd = ((static.get("mean_defense") or {}).get("delta_vs_none"))
            bd = (r.get("bon") or {}).get("delta_nauc_vs_none")
            wd, _ = worst_degeneration(static)
            # ABSENCE IS PER PANEL, not per row. A "partial" rung — static defence graded, best-of-N
            # not — used to draw its defence point and simply OMIT the best-of-N one, which is the
            # very confusion this figure exists to prevent: a reader saw a dose with no best-of-N
            # marker and could not tell "no effect" from "not graded yet". Each panel now marks its
            # own missing half.
            if dd is None and bd is None:
                absent_x.append(lam); n_absent += 1
                continue
            if dd is None:
                absent_def.append(lam)
            if bd is None:
                absent_bon.append(lam)
            if wd is None:
                absent_deg.append(lam)
            if dd is not None:
                (sign_pt := (lam, dd * 100)) if r["sign_control"] else (
                    def_x.append(lam), def_y.append(dd * 100))
            if bd is not None:
                if r["sign_control"]:
                    bon_sign = (lam, bd)
                else:
                    bon_x.append(lam); bon_y.append(bd)
            if wd is not None:
                if r["sign_control"]:
                    deg_sign = (lam, wd)
                else:
                    deg_x.append(lam); deg_y.append(wd)

        for ax, extra in ((ax_def, absent_def), (ax_bon, absent_bon), (ax_deg, absent_deg)):
            ax.axhline(0, color=RULE, lw=1.6, zorder=0)
            ax.axvline(0, color=RULE, lw=1.6, zorder=0)
            for x in list(absent_x) + list(extra):
                ax.plot([x], [0], marker="o", ms=9, mfc="none", mec=MUTED, mew=1.8,
                        ls="none", zorder=2)

        # The dashed connector crosses zero from the sign control to rung 1. It is a READING AID
        # for the axis, not an interpolation: no rung exists between them, and the curve builder's
        # contract forbids sorting the control as rung 1 or averaging it in — neither of which a
        # dashed segment does. Without it the diamond reads as an orphaned second condition rather
        # than the negative end of one dose axis, which is exactly what it is.
        if sign_pt and def_x:
            ax_def.plot([sign_pt[0], def_x[0]], [sign_pt[1], def_y[0]], ls=(0, (5, 4)),
                        color=C_DEF, lw=1.7, zorder=2)
        ax_def.plot(def_x, def_y, "-o", color=C_DEF, lw=2.4, ms=8, zorder=3)
        if sign_pt:
            ax_def.plot([sign_pt[0]], [sign_pt[1]], marker="D", ms=10, mfc="white",
                        mec=C_DEF, mew=2.4, ls="none", zorder=4)
        if bon_sign and bon_x:
            ax_bon.plot([bon_sign[0], bon_x[0]], [bon_sign[1], bon_y[0]], ls=(0, (5, 4)),
                        color=C_BON, lw=1.7, zorder=2)
        ax_bon.plot(bon_x, bon_y, "-o", color=C_BON, lw=2.4, ms=8, zorder=3)
        if bon_sign:
            ax_bon.plot([bon_sign[0]], [bon_sign[1]], marker="D", ms=10, mfc="white",
                        mec=C_BON, mew=2.4, ls="none", zorder=4)
        if deg_sign and deg_x:
            ax_deg.plot([deg_sign[0], deg_x[0]], [deg_sign[1], deg_y[0]], ls=(0, (5, 4)),
                        color=C_DEG, lw=1.7, zorder=2)
        ax_deg.plot(deg_x, deg_y, "-o", color=C_DEG, lw=2.4, ms=8, zorder=3)
        if deg_sign:
            ax_deg.plot([deg_sign[0]], [deg_sign[1]], marker="D", ms=10, mfc="white",
                        mec=C_DEG, mew=2.4, ls="none", zorder=4)
        ax_deg.axhline(CONTAM, color=C_DEG, lw=1.5, ls=(0, (4, 3)), zorder=1)

        ax_def.set_title(target, color=INK, pad=10)
        ax_deg.set_xlabel("λ")
        if col == 0:
            ax_def.set_ylabel("defence Δ pp")
            ax_bon.set_ylabel("BoN ΔnAUC")
            ax_deg.set_ylabel("degeneration Δ")

    # ONE inversion for the whole shared row: calling invert_yaxis() per column on a sharey axis
    # toggles the same axis N times and an even N silently un-inverts it.
    axes[1][0].invert_yaxis()                 # negative = fewer successes = better defended = up

    fig.suptitle("Dose response", fontsize=19, y=0.985)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    split_txt = ("" if not split_targets else
                 f" {', '.join(split_targets)} ran two samplers: drawn at top_p 1.0 where a 1.0 measurement exists "
                 f"and 0.95 where it does not (ruled 2026-09-14), as the panel states; the two are not pooled.")
    fig.text(0.5, 0.012,
             "Diamonds = sign control (−λ₁). An open grey circle on a panel's zero line = that panel's half not yet "
             "graded for that dose, marked PER PANEL so a partial rung cannot read as a measured zero. "
             "Best-of-N axis inverted so up is better defended on every panel. Degeneration is steered minus the "
             f"same-standup anchor; dashed line = {CONTAM:.2f}, above which the defence number is not readable as defence."
             + split_txt,
             ha="center", va="bottom", fontsize=11.5, color=MUTED, wrap=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"final_defense_dose_response.{ext}"
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    print(f"[defense-figs] {n_absent} rung(s) drawn as ABSENT (measured, not yet graded)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    curves = json.loads(a.curves.expanduser().read_text(encoding="utf-8"))
    for p in draw(curves, a.out_dir.expanduser()):
        print(f"[defense-figs] -> {p}")


if __name__ == "__main__":
    main()
