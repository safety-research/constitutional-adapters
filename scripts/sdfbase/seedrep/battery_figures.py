"""Figures for qwen-seedrep-native-battery — from ``battery_summary.json`` ONLY.

`battery_tables.py` computes every number (headline per (arm, seed), the ADMITTED triplet's mean / SD / range, the
seed-spread ratio, the admitted paired deltas, the at-cap shares, the standup utilisation and its bar VERDICTS); this
module DRAWS them and never recomputes an aggregate or re-judges a bar — a display filter (``--legs``) changes which
panels exist, not any value in them. Each drawing function returns a render record (what was drawn: points, missing and
not-admitted cells, the aggregate ticks, the axis range, the banner) that the tests assert against the summary; the same
record feeds ``figures_manifest.json``, and the W&B artifact and the served mirror are built from THAT manifest, never
from the directory (a reused directory would republish a stale figure).

Provisional status is drawn, not implied: a PARTIAL summary (an absent upstream input) puts a **PARTIAL — PROVISIONAL**
banner naming the absent evidence on EVERY figure; a leg the engine-identity verdict rejected carries a **LEG REJECTED**
banner; a readable value that was NOT admitted (coverage missing, audit not AGREES, identity FAIL, ladder not final) is
drawn as a hollow seed marker with its own legend entry, never as a measurement; and the title asserts "vendor preset,
native serving" ONLY when the request-evidence audit filed AGREES for every arm-run of the leg — otherwise it says how
many are unverified.

Figures (filenames are stable; every PNG has a PDF twin):

* ``<leg>_arms.png`` — one panel, the 27 arm-runs of one leg: x = arm in the fixed order (anchor · prompted ×2 · control
  PSR/LoRA · constitution PSR/LoRA · LoRA difference · Arditi c = 0.3), three seed markers per arm (seeds are the entity:
  s1 blue ○, s2 orange □, s3 aqua △ — the first three categorical slots, all-pairs safe), the seed mean as an ink tick with
  the range as a thin ink bar (ADMITTED triplets only); a seed with no reading is a hollow muted marker at the axis
  floor; a readable-but-not-admitted seed a hollow marker in its seed colour; the arm's admitted coverage beside a
  missing aggregate; the at-cap share rides as a muted strip under the arm labels (measured · NOT FINAL · MISSING · n/a);
  the y-axis is IDENTICAL across the legs of a family.
* ``summary_<family>.png`` — arm × leg heatmap of the admitted seed mean (one-hue sequential ramp; the range in each cell),
  a hatched cell with the admitted coverage where the triplet is incomplete, and a seed-spread-ratio strip (second one-hue
  ramp; `n/a` with the reason where the ratio has no denominator).
* ``diag_utilisation.png`` (Running/S, SM %, AGGREGATE tok/s ÷ S5, H200-h ÷ sized per standup — the four bars, coloured by their own verdicts),
  ``diag_at_cap.png`` (final-rung at-cap share per (leg, arm, seed) vs the 1 % rule; NOT FINAL and MISSING counted),
  ``diag_determinism.png`` (identical/n per leg; unmeasured legs distinct from 0), ``diag_h200h.png`` (H200-h per
  CONTAINER leg, actual vs expected — the tables' totals, withheld when a standup lacks its hours).

Look per the ``dataviz`` skill through the lane's ``figures.py`` tokens (surface, ink, hairline solid grid, ≥ 8 px markers
with a surface ring, text in ink tokens, a legend whenever ≥ 2 series share an axis; status colours are the skill's fixed
status palette and never a series colour). Titles ≤ 2 lines + the banner; the axes take ≥ 80 % of the figure width.

W&B / mirror are OPT-IN (``--wandb`` / ``--mirror``); the default and the tests are a dry run that writes files only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import figures as F  # noqa: E402  (palette tokens + style_axes + save)

SEEDS = (1, 2, 3)
SEED_COLOR = {1: F.SERIES[0], 2: F.SERIES[1], 3: F.SERIES[2]}      # slots 1–3: all-pairs safe in both modes
SEED_MARKER = {1: "o", 2: "s", 3: "^"}
SEED_OFFSET = {1: -0.2, 2: 0.0, 3: 0.2}
AT_CAP_Y = -0.19                                                     # axes-fraction y of the at-cap strip (under the two-line tick labels)
ARM_TICK = {"none": "none\n(unsteered)", "prompted__si_constitution": "prompted:\nconstitution", "prompted__safety_refusal": "prompted:\nsafety refusal",
            "constv2ctl2-full__psr_ce_all": "control\nPSR", "constv2ctl2-full__lora_r4_ce": "control\nLoRA r4",
            "constv2-mix-full__psr_ce_all": "constitution\nPSR", "constv2-mix-full__lora_r4_ce": "constitution\nLoRA r4",
            "constv2-mix-full__lora_r4_ce_minus_ctl": "LoRA difference\n(const − ctl)", "ar_c0p3__arditi_allpos": "Arditi all-pos\n(c = 0.3)"}
RATIO_CLIP = 3.0
#: the OFFICIAL publication namespace. A provisional (PARTIAL) run never publishes here: its run name, artifact
#: collection and mirror directory all carry `-PARTIAL-<evidence hash>` (P2-3), so a provisional artifact can neither become
#: the official collection's `latest` nor overwrite an official mirrored figure.
DEFAULT_MIRROR = Path.home() / "figs" / "qwen-seedrep-native-battery"
#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's SISTEER_WANDB_PROJECT_SEEDREP first)
WANDB_PURPOSE = "seedrep"
WANDB_RUN_NAME = "seedrep-native-battery"
WANDB_ARTIFACT = "figures-seedrep-native-battery"
PARTIAL_TAG = "PARTIAL"
EVIDENCE_HASH_LEN = 12


def evidence_hash(status: Mapping[str, Any]) -> str:
    """sha256 (first 12 hex) of the canonical JSON of what the output's evidence IS — its status, absent inputs, the
    missing-evidence summary and the rejected legs — so two provisional publications with different evidence never share a
    namespace and the same evidence always maps to the same one."""
    body = {"partial": bool(status.get("partial")), "table_status": status.get("table_status"), "absent_inputs": status.get("absent_inputs") or [],
            "provisional_reasons": status.get("provisional_reasons") or [], "rejected_legs": status.get("rejected_legs") or [],
            "missing_evidence": (status.get("evidence") or {}).get("missing_evidence") or []}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:EVIDENCE_HASH_LEN]


def publication_names(status: Mapping[str, Any], *, run_name: str = WANDB_RUN_NAME, artifact: str = WANDB_ARTIFACT,
                      mirror_root: Path | str | None = None) -> dict[str, Any]:
    """The EFFECTIVE W&B run name, artifact collection and mirror directory for a summary / figures manifest: the official
    names for an official output; `<name>-PARTIAL-<evidence hash>` on all three for anything provisional. `mirror_root`
    defaults to the module's DEFAULT_MIRROR read at CALL time."""
    partial = bool(status.get("partial"))
    h = evidence_hash(status) if partial else None
    sfx = f"-{PARTIAL_TAG}-{h}" if partial else ""
    root = Path(mirror_root if mirror_root is not None else DEFAULT_MIRROR)
    return {"partial": partial, "evidence_hash": h, "run_name": run_name + sfx, "artifact": artifact + sfx,
            "mirror_dest": root.with_name(root.name + sfx), "official_mirror": root, "official_artifact": artifact, "official_run_name": run_name}


def refuse_official_namespace(status: Mapping[str, Any], *, mirror_dest: Path | str | None = None, artifact: str | None = None,
                              run_name: str | None = None) -> None:
    """A provisional output may not be published under any official name, however it was asked for."""
    if not status.get("partial"):
        return
    names = publication_names(status)
    hits = []
    if mirror_dest is not None and Path(mirror_dest).expanduser().resolve() == names["official_mirror"].expanduser().resolve():
        hits.append(f"mirror {names['official_mirror']}")
    if artifact is not None and artifact == names["official_artifact"]:
        hits.append(f"artifact {artifact!r}")
    if run_name is not None and run_name == names["official_run_name"]:
        hits.append(f"run {run_name!r}")
    if hits:
        raise RuntimeError("a PARTIAL (provisional) output may not be published under the official namespace: " + ", ".join(hits)
                           + f" — use the derived names {names['run_name']!r} / {names['artifact']!r} / {names['mirror_dest']}")
#: the dataviz skill's FIXED status palette (never a series colour): the PARTIAL banner fill and the REJECTED banner fill
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

# a second one-hue sequential ramp (orange, the next categorical slot) for the ratio strip — light → dark
ORANGE_RAMP = ["#fbe3d6", "#f7c6ad", "#f3a984", "#ee8b5b", "#eb6834", "#c9532a", "#a34220", "#7d3218"]
BLUE_RAMP = [F.SEQ_BLUE[s] for s in (100, 200, 300, 400, 450, 550, 650, 700)]


def _hex_lum(h: str) -> float:
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ramp_color(v: float, lo: float, hi: float, ramp: Sequence[str]) -> str:
    if hi <= lo:
        return ramp[0]
    t = min(1.0, max(0.0, (float(v) - lo) / (hi - lo)))
    return ramp[min(len(ramp) - 1, int(round(t * (len(ramp) - 1))))]


def text_on(fill: str) -> str:
    return "#ffffff" if _hex_lum(fill) < 0.45 else F.INK


def load_summary(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- provisional status: ONE banner every figure carries when it applies
@dataclass(frozen=True)
class Banner:
    """What a figure must say about its evidence before its title: the PARTIAL banner (absent inputs + the missing-evidence
    summary) and / or the REJECTED banner (the engine-identity verdict). Empty for an official figure of accepted legs."""

    text: str | None
    kind: str | None            # partial | rejected | None

    @property
    def on(self) -> bool:
        return bool(self.text)

    @property
    def extra_h(self) -> float:          # inches added to the figure height
        return 0.45 if self.on else 0.0

    @property
    def dy(self) -> float:               # axes-fraction shift of the title block
        return 0.09 if self.on else 0.0


def rejected_legs(summary: Mapping[str, Any], leg_ids: Sequence[str]) -> list[str]:
    return [l for l in leg_ids if str(summary["legs"][l].get("leg_status", "")).startswith("REJECTED")]


def banner_for(summary: Mapping[str, Any], leg_ids: Sequence[str] = ()) -> Banner:
    """The banner for a figure over ``leg_ids`` (one leg → its own REJECTED text; several → the list; none → PARTIAL only)."""
    parts, kind = [], None
    if summary.get("partial"):
        ev = summary.get("evidence") or {}
        why = summary.get("provisional_reasons") or (["absent evidence " + ", ".join(summary.get("absent_inputs") or [])] if summary.get("absent_inputs") else [])
        parts.append("PARTIAL — PROVISIONAL: " + "; ".join(why) + (f"; {ev.get('n_admissible')}/{ev.get('n_arm_runs')} arm-runs admitted" if ev else ""))
        kind = "partial"
    rej = rejected_legs(summary, leg_ids)
    if rej:
        if len(leg_ids) == 1:
            parts.append("LEG REJECTED — " + str(summary["legs"][rej[0]]["leg_status"])[len("REJECTED: "):])
        else:
            parts.append("LEG(S) REJECTED by the engine-identity verdict: " + ", ".join(rej))
        kind = "rejected"
    return Banner("\n".join(parts) if parts else None, kind)


def _finish(fig, banner: Banner, title: str, *, adjust: Mapping[str, float], x: float = 0.02,
            handles: Sequence[Line2D] | None = None, ncol: int = 1, legend_fontsize: float = 8) -> None:
    """Draw the banner (status colour, never a series colour), the two-line title under it, the legend and the axes margins
    — the one place the banner's geometry lives."""
    top = 0.995 - banner.dy
    if banner.on:
        fill = STATUS_CRITICAL if banner.kind == "rejected" else STATUS_WARNING
        fig.text(0.5, 0.995, banner.text, ha="center", va="top", fontsize=8.5, color=text_on(fill), fontweight="bold",
                 bbox={"boxstyle": "square,pad=0.35", "facecolor": fill, "edgecolor": "none"}, wrap=True)
    fig.suptitle(title, color=F.INK, x=x, ha="left", y=top, fontsize=10)
    if handles:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, top - 0.065), ncol=ncol, fontsize=legend_fontsize, frameon=False)
    fig.subplots_adjust(**{**adjust, "top": adjust["top"] - (0.075 if banner.on else 0.0)})


def _record(banner: Banner, **fields: Any) -> dict[str, Any]:
    return {**fields, "banner": banner.text}


#: The ledger-less reconciliation contract (`battery_tables.CONTRACT_LEDGERLESS`; pinned equal by test): an AGREES issued under it
#: is a QUALIFIED verdict, and the evidence line says so wherever it asserts the decode contract
LEDGERLESS_CONTRACT = "seedrep_request_evidence_reconciliation/2-D106"


def evidence_line(leg: Mapping[str, Any]) -> str:
    """What the figure may ASSERT about a leg: the decode contract only where the audit filed AGREES for every arm-run -- and then with
    the reconciliation split stated: how many AGREES were ledger-less (qualified, body digest not available) or ledger-backed, and how
    many verdicts were refused as unqualified; a qualified AGREES is never rendered as a bare one."""
    n, n_adm, n_agr = leg["n_arm_runs"], leg["n_admissible"], leg["n_audit_agrees"]
    n_ll, n_lb, n_unq = leg.get("n_audit_ledgerless", 0), leg.get("n_audit_ledger_backed", 0), leg.get("n_audit_unqualified", 0)
    qual = ((f"; D106: {n_ll} ledger-less under {LEDGERLESS_CONTRACT}" + (", body digest not available" if n_agr == n else "")) if n_ll
            else (f"; {n_lb} ledger-backed" if n_lb else ""))
    unq = f"; {n_unq} unqualified" if n_unq else ""
    decode = (f"vendor preset, native serving (audit AGREES on every arm-run{qual})" if n_agr == n
              else f"decode contract UNVERIFIED on {n - n_agr}/{n} arm-runs (audit AGREES {n_agr}/{n}{unq}{qual})")
    return f"{n_adm}/{n} arm-runs admitted ({leg['n_readable']} readable) · {decode} · engine identity {leg['identity_verdict']}"


# ---------------------------------------------------------------- axis ranges — identical across the legs of a family (a display range, computed once)
def family_axis_range(summary: Mapping[str, Any], family: str) -> tuple[float, float]:
    legs = summary["families"][family]["legs"]
    units = {summary["legs"][l]["spec"]["units"] for l in legs if l in summary["legs"]}
    if units == {"score"}:
        return 1.0, 10.0
    vals = [r["readable_value"] + (r["se"] or 0.0) for l in legs if l in summary["legs"] for r in summary["legs"][l]["rows"]
            if r["readable_value"] is not None]
    if not vals:
        return 0.0, 100.0
    hi = min(100.0, 10.0 * math.ceil(max(vals) * 1.08 / 10.0))
    return 0.0, max(hi, 10.0)


def _arm_positions(summary: Mapping[str, Any]) -> tuple[list[str], dict[str, float], list[tuple[str, float, float]]]:
    order = summary["arm_order"]
    groups = [a["group"] for a in summary["arms"]]
    xs: dict[str, float] = {}
    x = 0.0
    spans: list[tuple[str, float, float]] = []
    prev = None
    for arm, g in zip(order, groups):
        if prev is not None and g != prev:
            x += 0.55                                        # a gap between groups
        xs[arm] = x
        if not spans or spans[-1][0] != g:
            spans.append((g, x, x))
        else:
            spans[-1] = (g, spans[-1][1], x)
        prev = g
        x += 1.0
    return order, xs, spans


def _seed_legend_handles(with_missing: bool, with_not_admitted: bool) -> list[Line2D]:
    h = [Line2D([0], [0], ls="none", marker=SEED_MARKER[s], ms=7.5, color=SEED_COLOR[s], mec=F.SURFACE, mew=1.2, label=f"seed {s}")
         for s in SEEDS]
    h.append(Line2D([0], [0], color=F.INK, lw=2.2, label="seed mean (tick) and range (bar), admitted triplets"))
    if with_not_admitted:
        h.append(Line2D([0], [0], ls="none", marker="o", ms=7.5, mfc="none", mec=F.SERIES[0], mew=1.4, label="readable, NOT admitted (provisional / rejected)"))
    if with_missing:
        h.append(Line2D([0], [0], ls="none", marker="o", ms=7.5, mfc="none", mec=F.MUTED, mew=1.2, label="no reading"))
    return h


def _at_cap_strip(rows: Mapping[tuple[str, int], Mapping[str, Any]], arm: str) -> tuple[str, str]:
    statuses = [str(rows[(arm, s)]["at_cap_status"]) for s in SEEDS]
    shares = [rows[(arm, s)]["at_cap_share"] for s in SEEDS if rows[(arm, s)]["at_cap_status"] == "measured"]
    if all(x.startswith("not applicable") for x in statuses):
        return "", "not_applicable"
    if any(x.startswith("not final") for x in statuses):
        return "NOT FINAL", "not_final"
    if len(shares) == 3:
        return f"≤{100 * max(shares):.1f} %", "measured"
    if shares:
        return f"≤{100 * max(shares):.1f} % ({len(shares)}/3)", "partial"
    return "MISSING", "missing"


def arms_figure(summary: Mapping[str, Any], leg_id: str, out_dir: Path | str, *, ylim: tuple[float, float] | None = None) -> dict[str, Any]:
    leg = summary["legs"][leg_id]
    spec = leg["spec"]
    lo, hi = ylim if ylim is not None else family_axis_range(summary, spec["family"])
    order, xs, spans = _arm_positions(summary)
    label_of = {a["arm"]: a["label"] for a in summary["arms"]}
    rows = {(r["arm"], r["seed"]): r for r in leg["rows"]}
    banner = banner_for(summary, [leg_id])
    fig, ax = plt.subplots(figsize=(11.0, 5.4 + banner.extra_h))
    F.style_axes(ax)
    rec = _record(banner, leg=leg_id, points=[], not_admitted=[], missing=[], aggregates={}, no_aggregate={}, at_cap_strip={},
                  ylim=[lo, hi], provisional=bool(summary.get("partial")), leg_status=leg.get("leg_status"), evidence=evidence_line(leg))
    floor = lo + 0.03 * (hi - lo)
    any_missing = any_not_admitted = False
    for arm in order:
        x = xs[arm]
        a = leg["arms"][arm]
        for s in SEEDS:
            r = rows[(arm, s)]
            xi = x + SEED_OFFSET[s]
            if r["admissible"] and r["value"] is not None:
                ax.plot([xi], [r["value"]], ls="none", marker=SEED_MARKER[s], ms=7.5, color=SEED_COLOR[s], mec=F.SURFACE, mew=1.2, zorder=4)
                rec["points"].append([arm, s, r["value"]])
            elif r["readable_value"] is not None:
                any_not_admitted = True
                ax.plot([xi], [r["readable_value"]], ls="none", marker=SEED_MARKER[s], ms=7.5, mfc="none", mec=SEED_COLOR[s], mew=1.4, zorder=3)
                rec["not_admitted"].append([arm, s, r["readable_value"], r["admissibility"]])
            else:
                any_missing = True
                ax.plot([xi], [floor], ls="none", marker=SEED_MARKER[s], ms=7.5, mfc="none", mec=F.MUTED, mew=1.2, zorder=3)
                rec["missing"].append([arm, s])
        if a["status"] == "ok":
            ax.plot([x, x], [a["min"], a["max"]], color=F.INK, lw=1.2, zorder=2)
            ax.plot([x - 0.32, x + 0.32], [a["mean"], a["mean"]], color=F.INK, lw=2.2, zorder=5)
            rec["aggregates"][arm] = {"mean": a["mean"], "min": a["min"], "max": a["max"]}
        else:
            ax.annotate(f"n/a\n{a['coverage']} admitted", (x, floor + 0.06 * (hi - lo)), color=F.MUTED, fontsize=7.5, ha="center", va="bottom")
            rec["no_aggregate"][arm] = a["coverage"]
        txt, state = _at_cap_strip(rows, arm)
        rec["at_cap_strip"][arm] = {"state": state, "text": txt}
        if txt:
            ax.annotate(txt, (x, AT_CAP_Y), xycoords=("data", "axes fraction"), color=F.MUTED, fontsize=7, ha="center", va="top",
                        annotation_clip=False)
    for i, (g, x0, x1) in enumerate(spans):
        ax.annotate(summary["group_label"][g], ((x0 + x1) / 2, 1.005), xycoords=("data", "axes fraction"), color=F.INK2, fontsize=8,
                    ha="center", va="bottom")
        if i:
            ax.axvline(x0 - 0.775, color=F.GRID, lw=0.8, zorder=1)
    ax.set_xticks([xs[a] for a in order])
    ax.set_xticklabels([ARM_TICK.get(a, label_of[a]) for a in order], fontsize=8)
    ax.set_xlim(-0.7, max(xs.values()) + 0.7)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(spec["metric"], color=F.INK2)
    if any(v["state"] != "not_applicable" for v in rec["at_cap_strip"].values()):
        ax.annotate("at cap\n(final rung,\nmax over seeds)", (-0.005, AT_CAP_Y), xycoords="axes fraction", color=F.MUTED, fontsize=7,
                    ha="right", va="top", annotation_clip=False)
    ladder_note = "" if spec["ladder"] else "; fixed cap (no ladder)"
    _finish(fig, banner, f"{spec['label']} — {spec['metric']}, per arm and seed\n{rec['evidence']} · Arditi at c = 0.3 raw (user override; rule: c = 0.2){ladder_note}",
            x=0.03, handles=_seed_legend_handles(any_missing, any_not_admitted), ncol=3, legend_fontsize=7.5,
            adjust={"left": 0.065, "right": 0.995, "top": 0.845, "bottom": 0.165})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{leg_id}_arms.pdf"
    fig.savefig(pdf, bbox_inches="tight", facecolor=F.SURFACE)      # the PDF twin from the SAME figure object
    png = F.save(fig, out_dir / f"{leg_id}_arms.png")               # save() closes the figure
    rec["path"] = str(png)
    rec["pdf"] = str(pdf)
    return rec


def summary_figure(summary: Mapping[str, Any], family: str, out_dir: Path | str) -> dict[str, Any]:
    fam = summary["families"][family]
    legs = [l for l in fam["legs"] if l in summary["legs"]]
    rows_by_leg = {l: summary["legs"][l]["arms"] for l in legs}
    petri = family == "petri" and summary.get("petri_behaviours")
    if petri:      # Petri: the six behaviours are the columns (pooled first)
        cols = [("petri", "pooled")] + [(b, b) for b in summary["petri_behaviours"]]
    else:
        cols = [(l, summary["legs"][l]["spec"]["label"]) for l in legs]
    order = summary["arm_order"]
    label_of = {a["arm"]: a["label"] for a in summary["arms"]}
    lo, hi = family_axis_range(summary, family)
    banner = banner_for(summary, legs)
    n_r, n_c = len(order), len(cols)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(max(9.5, 1.55 * n_c + 5.2), 0.62 * n_r + 2.2 + banner.extra_h),
                                  gridspec_kw={"width_ratios": [n_c, n_c]}, sharey=True)
    rec = _record(banner, family=family, cells={}, ratio_cells={}, range=[lo, hi], columns=[c[0] for c in cols],
                  provisional=bool(summary.get("partial")), rejected_legs=rejected_legs(summary, legs))
    for axx in (ax, axr):
        F.style_axes(axx, ygrid=False)
        axx.set_xlim(0, n_c)
        axx.set_ylim(0, n_r)
        axx.set_yticks([n_r - i - 0.5 for i in range(n_r)])
        axx.set_xticks([j + 0.5 for j in range(n_c)])
        axx.tick_params(length=0)
        for side in ("left", "bottom"):
            axx.spines[side].set_visible(False)
    ax.set_yticklabels([label_of[a] for a in order], fontsize=8)
    for axx in (ax, axr):
        axx.set_xticklabels([c[1] for c in cols], rotation=25, ha="right", fontsize=8)
    for j, (col, _) in enumerate(cols):
        for i, arm in enumerate(order):
            y = n_r - i - 1
            a = (summary["petri_behaviours"][col][arm] if (petri and col != "petri") else rows_by_leg[col][arm])
            sp = a["seed_spread"]
            for axx, val, kind in ((ax, a["mean"], "mean"), (axr, sp["ratio"], "ratio")):
                if val is None:
                    axx.add_patch(Rectangle((j + 0.03, y + 0.03), 0.94, 0.94, facecolor=F.SURFACE, edgecolor=F.GRID, hatch="////", lw=0.8))
                    why = (f"{a['coverage']} admitted" if a["status"] != "ok" else
                           ("no SE" if "no per-seed SE" in (sp["status"] or "") else "undefined"))
                    axx.text(j + 0.5, y + 0.5, f"MISSING\n{why}" if kind == "mean" else f"n/a\n{why}", ha="center", va="center", fontsize=7, color=F.MUTED)
                    (rec["cells"] if kind == "mean" else rec["ratio_cells"])[f"{col}|{arm}"] = {"value": None, "why": why}
                    continue
                if kind == "mean":
                    fill = ramp_color(val, lo, hi, BLUE_RAMP)
                    txt = f"{val:.1f}\nrange {a['range']:.1f}"
                else:
                    fill = ramp_color(min(val, RATIO_CLIP), 0.0, RATIO_CLIP, ORANGE_RAMP)
                    txt = f"{val:.2f}" + ("+" if val > RATIO_CLIP else "")
                axx.add_patch(Rectangle((j + 0.03, y + 0.03), 0.94, 0.94, facecolor=fill, edgecolor=F.SURFACE, lw=2))
                axx.text(j + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=7.5, color=text_on(fill))
                (rec["cells"] if kind == "mean" else rec["ratio_cells"])[f"{col}|{arm}"] = {"value": val}
    units = "score 1–10" if lo == 1.0 and hi == 10.0 else "%"
    ax.set_title(f"admitted seed mean, {units} (range in cell)\nramp {lo:g}–{hi:g}", color=F.INK2, fontsize=9)
    axr.set_title(f"seed-spread ratio = SD over seeds ÷ per-seed SE\nramp 0–{RATIO_CLIP:g}; ≈ 1 = sampling noise only", color=F.INK2, fontsize=9)
    _finish(fig, banner, f"{fam['label']} — arm × leg, 27 arm-runs per leg\nhatched = no admitted triplet (admitted coverage shown); Arditi at c = 0.3 raw (read-only reference)",
            adjust={"left": 0.19, "right": 0.99, "top": 0.855, "bottom": 0.2, "wspace": 0.05})
    out_dir = Path(out_dir)
    rec["path"] = str(F.save(fig, out_dir / f"summary_{family}.png"))
    return rec


# ---------------------------------------------------------------- diagnostics (standup tables copied through the summary)
def _standups(summary: Mapping[str, Any], legs: Sequence[str] | None) -> list[dict[str, Any]]:
    st = summary.get("standups") or []
    return [s for s in st if legs is None or any(l in legs for l in s.get("legs", []))]


def _standup_label(s: Mapping[str, Any]) -> str:
    return f"{s['container_leg']}·{s['view']}·{s['stage']}"


#: the utilisation panels: (utilisation field, the verdict column, title, the numeric bar drawn as a reference line). The
#: tok/s panel is the AGGREGATE decode throughput's ratio to S5 (`tok_s_aggregate` / `s5_tok_s`, sampling-coverage
#: gated by the writer), and the FOURTH bar is the projected-hours ratio (`h200_h_ratio` ≤ 1.5)
UTILISATION_PANELS = (("running_ratio", "running_bar", "median Running ÷ S (bar 0.8)", 0.8),
                      ("median_sm_decode", "sm_bar", "median SM % in decode (bar 85)", 85.0),
                      ("tok_s_ratio_s5", "tok_bar", "AGGREGATE decode tok/s ÷ S5 ceiling (bar 0.8)", 0.8),
                      ("h200_h_ratio", "hours_bar", "H200-h actual ÷ sized (bar ≤ 1.5)", 1.5))


def diag_utilisation_figure(summary: Mapping[str, Any], out_dir: Path | str, legs: Sequence[str] | None = None) -> dict[str, Any]:
    """Per standup: the FOUR utilisation readings, coloured by its OWN bar verdict (`running_bar` / `sm_bar` / `tok_bar` /
    `hours_bar`: true · false · unmeasured(<why>)) — never re-judged here (the tok/s reading is the writer's aggregate ratio)."""
    st = _standups(summary, legs)
    panels = UTILISATION_PANELS
    banner = banner_for(summary)
    fig, axes = plt.subplots(1, len(panels), figsize=(16.0, 3.9 + banner.extra_h))
    rec = _record(banner, n_standups=len(st), n_with_reading=0, below_bar=[], unmeasured=[], panels=[p[0] for p in panels], provisional=bool(summary.get("partial")))
    xs = list(range(len(st)))
    for ax, (key, bar_key, title, bar) in zip(axes, panels):
        F.style_axes(ax)
        for x, s in zip(xs, st):
            u = s.get("utilisation")
            v = None if u is None else u.get(key)
            verdict = None if u is None else str(u.get(bar_key) or "").lower()
            if v is None or not verdict or verdict.startswith("unmeasured"):
                ax.plot([x], [0], ls="none", marker="o", ms=6, mfc="none", mec=F.MUTED, mew=1.1)
                if key == "running_ratio":
                    rec["unmeasured"].append(s["attempt_id"])
                continue
            below = verdict != "true"
            ax.plot([x], [v], ls="none", marker="o", ms=6.5, color=F.SERIES[1] if below else F.SERIES[0], mec=F.SURFACE, mew=1.0, zorder=3)
            if below:
                rec["below_bar"].append([s["attempt_id"], key, v])
        ax.axhline(bar, color=F.BASELINE, lw=1.0)
        ax.set_title(title, color=F.INK2, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([_standup_label(s) for s in st], rotation=90, fontsize=5.5)
        ax.set_xlim(-0.8, len(st) - 0.2)
    rec["n_with_reading"] = sum(1 for s in st if s.get("utilisation") is not None)
    handles = [Line2D([0], [0], ls="none", marker="o", ms=6.5, color=F.SERIES[0], mec=F.SURFACE, label="bar met (WS3-G verdict)"),
               Line2D([0], [0], ls="none", marker="o", ms=6.5, color=F.SERIES[1], mec=F.SURFACE, label="bar NOT met (WS3-G verdict)"),
               Line2D([0], [0], ls="none", marker="o", ms=6, mfc="none", mec=F.MUTED, label="no reading / unmeasured")]
    _finish(fig, banner, f"Utilisation per standup (criterion 5) — {rec['n_with_reading']}/{len(st)} standups carry a reading\n"
            "each dot one accepted attempt (container leg · engine · stage); the four bars = the fail-closed fan-out gate (`bars_met`)",
            handles=handles, ncol=3, adjust={"left": 0.045, "right": 0.995, "top": 0.78, "bottom": 0.3, "wspace": 0.2})
    rec["path"] = str(F.save(fig, Path(out_dir) / "diag_utilisation.png"))
    return rec


def diag_at_cap_figure(summary: Mapping[str, Any], out_dir: Path | str, legs: Sequence[str] | None = None) -> dict[str, Any]:
    leg_ids = [l for l in summary["legs"] if (legs is None or l in legs)
               and not all(str(r["at_cap_status"]).startswith("not applicable") for r in summary["legs"][l]["rows"])]
    banner = banner_for(summary)
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * max(len(leg_ids), 1) + 1.9 + banner.extra_h))
    F.style_axes(ax, ygrid=False, xgrid=True)
    rec = _record(banner, legs={}, provisional=bool(summary.get("partial")))
    for i, l in enumerate(leg_ids):
        y = len(leg_ids) - i - 1
        rows = summary["legs"][l]["rows"]
        measured = [r for r in rows if r["at_cap_status"] == "measured"]
        n_missing = sum(1 for r in rows if str(r["at_cap_status"]).startswith("MISSING"))
        n_not_final = sum(1 for r in rows if str(r["at_cap_status"]).startswith("not final"))
        for r in measured:
            ax.plot([100 * r["at_cap_share"]], [y + SEED_OFFSET[r["seed"]] * 0.9], ls="none", marker=SEED_MARKER[r["seed"]], ms=6.5,
                    color=SEED_COLOR[r["seed"]], mec=F.SURFACE, mew=1.0, alpha=0.9, zorder=3)
        note = f"{len(measured)}/{len(rows)} measured" + (f", {n_not_final} NOT FINAL" if n_not_final else "") + (f", {n_missing} MISSING" if n_missing else "")
        ax.annotate(note, (1.0, y), xycoords=("axes fraction", "data"), color=F.MUTED, fontsize=7.5, ha="right", va="center")
        rec["legs"][l] = {"measured": len(measured), "missing": n_missing, "not_final": n_not_final, "n": len(rows),
                          "max_share": max((r["at_cap_share"] for r in measured), default=None)}
    ax.axvline(1.0, color=F.BASELINE, lw=1.0)
    ax.annotate("1 % rule", (1.0, len(leg_ids) - 0.45), color=F.INK2, fontsize=7.5, ha="left", va="bottom")
    ax.set_yticks(range(len(leg_ids)))
    ax.set_yticklabels([summary["legs"][l]["spec"]["label"] for l in reversed(leg_ids)], fontsize=8)
    ax.set_ylim(-0.6, len(leg_ids) - 0.4)
    ax.set_xlim(left=0)
    ax.set_xlabel("share of items at the final rung's cap (%)", color=F.INK2)
    _finish(fig, banner, "At-cap share at the final rung, per (leg, arm, seed) — criterion 6\none marker per arm-run; fixed-cap legs at their cap; MMLU (no cap) not shown",
            handles=_seed_legend_handles(False, False)[:3], ncol=3, adjust={"left": 0.26, "right": 0.98, "top": 0.8, "bottom": 0.2})
    rec["path"] = str(F.save(fig, Path(out_dir) / "diag_at_cap.png"))
    return rec


def diag_determinism_figure(summary: Mapping[str, Any], out_dir: Path | str, legs: Sequence[str] | None = None) -> dict[str, Any]:
    leg_ids = [l for l in summary["legs"] if legs is None or l in legs]
    banner = banner_for(summary)
    fig, ax = plt.subplots(figsize=(10.0, 3.6 + banner.extra_h))
    F.style_axes(ax)
    rec = _record(banner, legs={}, provisional=bool(summary.get("partial")))
    for i, l in enumerate(leg_ids):
        d = summary["legs"][l].get("determinism")
        if not d or d.get("n") in (None, 0) or d.get("identical") is None:
            ax.plot([i], [0], ls="none", marker="o", ms=7, mfc="none", mec=F.MUTED, mew=1.2)
            ax.annotate("n/a", (i, 0.03), color=F.MUTED, fontsize=7.5, ha="center", va="bottom")
            rec["legs"][l] = None
            continue
        frac = d["identical"] / d["n"]
        ax.bar([i], [frac], width=0.55, color=F.SERIES[0], linewidth=0)
        ax.annotate(f"{d['identical']}/{d['n']}", (i, frac + 0.02), color=F.INK2, fontsize=7.5, ha="center", va="bottom")
        rec["legs"][l] = {"n": d["n"], "identical": d["identical"]}
    ax.set_xticks(range(len(leg_ids)))
    ax.set_xticklabels([summary["legs"][l]["spec"]["label"] for l in leg_ids], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share reproduced byte-for-byte", color=F.INK2)
    _finish(fig, banner, "Determinism spot-check per leg (criterion 3): identical re-issues / n requests on the s1 engine\nhollow = no record (unmeasured, not zero)",
            adjust={"left": 0.07, "right": 0.995, "top": 0.82, "bottom": 0.3})
    rec["path"] = str(F.save(fig, Path(out_dir) / "diag_determinism.png"))
    return rec


def diag_h200h_figure(summary: Mapping[str, Any], out_dir: Path | str, legs: Sequence[str] | None = None) -> dict[str, Any]:
    """H200-hours per CONTAINER leg, drawn from `summary["h200_hours"]` ONLY (battery_tables.h200_hours: the totals over EVERY
    accepted attempt, withheld with the missing attempts named when one lacks its hours). This renderer reduces
    no rows: a withheld total is a hollow marker with its MISSING status, never a sum of the standups that happen to have one."""
    leg_ids = [l for l in summary["legs"] if legs is None or l in legs]
    cleg_of = summary.get("container_leg_of") or {l: l for l in leg_ids}
    clegs = list(dict.fromkeys(cleg_of[l] for l in leg_ids))
    hours = summary.get("h200_hours") or {}
    banner = banner_for(summary)
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * max(len(clegs), 1) + 1.8 + banner.extra_h))
    F.style_axes(ax, ygrid=False, xgrid=True)
    rec = _record(banner, legs={}, provisional=bool(summary.get("partial")))
    for i, c in enumerate(clegs):
        y = len(clegs) - i - 1
        h = hours.get(c)
        if h is None or not h.get("complete"):
            status = (h or {}).get("status") or "MISSING (no h200_hours record)"
            ax.plot([0], [y], ls="none", marker="o", ms=7, mfc="none", mec=F.MUTED, mew=1.2)
            ax.annotate(status if len(status) <= 60 else status[:57] + "…", (0.4, y), color=F.MUTED, fontsize=7, va="center")
            rec["legs"][c] = {"actual": None, "expected": None, "complete": False, "status": status,
                              "n_standups": (h or {}).get("n_standups"), "n_with_actual": (h or {}).get("n_with_actual")}
            continue
        a, z = h["actual"], h["expected"]
        ax.plot([z, a], [y, y], color=F.BASELINE, lw=2.0, zorder=2)
        ax.plot([z], [y], ls="none", marker="o", ms=7.5, color=F.GRAYS[0], mec=F.SURFACE, mew=1.2, zorder=3)
        ax.plot([a], [y], ls="none", marker="o", ms=7.5, color=F.SERIES[0], mec=F.SURFACE, mew=1.2, zorder=4)
        rec["legs"][c] = {"actual": a, "expected": z, "complete": True, "status": h["status"], "n_standups": h["n_standups"], "ratio": h.get("ratio")}
    labels = []
    for c in clegs:
        served = [l for l in leg_ids if cleg_of[l] == c]
        labels.append(c if served == [c] else f"{c} ({', '.join(served)})")
    ax.set_yticks(range(len(clegs)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.set_ylim(-0.6, len(clegs) - 0.4)
    ax.set_xlim(left=0)
    ax.set_xlabel("H200-hours (total over EVERY accepted attempt of the container leg; withheld when one lacks its hours)", color=F.INK2)
    handles = [Line2D([0], [0], ls="none", marker="o", ms=7.5, color=F.SERIES[0], mec=F.SURFACE, label="actual"),
               Line2D([0], [0], ls="none", marker="o", ms=7.5, color=F.GRAYS[0], mec=F.SURFACE, label="expected (sized, plan §5)"),
               Line2D([0], [0], ls="none", marker="o", ms=7, mfc="none", mec=F.MUTED, label="total withheld (a standup without hours)")]
    _finish(fig, banner, "H200-hours per container leg, actual vs expected (criterion 5: within 1.5×)\nhollow = total withheld, MISSING attempts named in the summary",
            handles=handles, ncol=3, adjust={"left": 0.3, "right": 0.98, "top": 0.8, "bottom": 0.22})
    rec["path"] = str(F.save(fig, Path(out_dir) / "diag_h200h.png"))
    return rec


# ---------------------------------------------------------------- build all, manifest, mirror, W&B
DIAG_FIGURES = {"utilisation": diag_utilisation_figure, "at_cap": diag_at_cap_figure, "determinism": diag_determinism_figure, "h200h": diag_h200h_figure}


def _sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_all(summary: Mapping[str, Any], out_dir: Path | str, *, legs: Sequence[str] | None = None,
              summary_path: Path | str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leg_ids = [l for l in summary["legs"] if legs is None or l in legs]
    figures: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}
    for fam in summary["families"]:
        lo, hi = family_axis_range(summary, fam)
        for l in summary["families"][fam]["legs"]:
            if l in leg_ids:
                rec = arms_figure(summary, l, out_dir, ylim=(lo, hi))
                figures[f"{l}/arms"] = {"path": rec["path"], "sha256": _sha256(rec["path"])}
                if Path(rec["pdf"]).is_file():
                    figures[f"{l}/arms_pdf"] = {"path": rec["pdf"], "sha256": _sha256(rec["pdf"])}
                records[f"{l}/arms"] = rec
        if any(l in leg_ids for l in summary["families"][fam]["legs"]):
            rec = summary_figure(summary, fam, out_dir)
            figures[f"summary/{fam}"] = {"path": rec["path"], "sha256": _sha256(rec["path"])}
            records[f"summary/{fam}"] = rec
    for name, fn in DIAG_FIGURES.items():
        rec = fn(summary, out_dir, leg_ids)
        figures[f"diag/{name}"] = {"path": rec["path"], "sha256": _sha256(rec["path"])}
        records[f"diag/{name}"] = rec
    manifest = {"figures": figures, "legs": leg_ids, "legs_filter": (list(legs) if legs is not None else None),
                "summary": str(summary_path) if summary_path else None,
                "summary_sha256": (_sha256(summary_path) if summary_path and Path(summary_path).is_file() else None),
                "partial": bool(summary.get("partial")), "table_status": summary.get("table_status"), "absent_inputs": summary.get("absent_inputs", []),
                "provisional_reasons": summary.get("provisional_reasons", []),
                "evidence": summary.get("evidence"), "rejected_legs": rejected_legs(summary, leg_ids),
                "note": "aggregates are the summary's (admitted triplets only); a --legs filter changes which panels exist, never a value"}
    names = publication_names(manifest)
    manifest["publication"] = {"run_name": names["run_name"], "artifact": names["artifact"], "mirror_dest": str(names["mirror_dest"]),
                               "evidence_hash": names["evidence_hash"], "official": not names["partial"]}
    (out_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return {"manifest": manifest, "records": records, "out_dir": str(out_dir)}


def mirror(manifest: Mapping[str, Any], out_dir: Path | str, dest: Path | str | None = None) -> subprocess.CompletedProcess:
    """rsync exactly the manifest's files (+ the manifest) to the served mirror — never the directory. The
    destination defaults to the manifest's EFFECTIVE namespace (`publication_names`: the official directory for an official
    run, `<dir>-PARTIAL-<evidence hash>` for a provisional one); an explicit destination equal to the official directory is
    refused for a provisional manifest."""
    out_dir = Path(out_dir)
    dest = Path(dest) if dest is not None else publication_names(manifest)["mirror_dest"]
    refuse_official_namespace(manifest, mirror_dest=dest)
    dest.mkdir(parents=True, exist_ok=True)
    rel = sorted({str(Path(v["path"]).resolve().relative_to(out_dir.resolve())) for v in manifest["figures"].values()} | {"figures_manifest.json"})
    return subprocess.run(["rsync", "-a", "--files-from=-", f"{out_dir}/", f"{dest}/"], input="\n".join(rel) + "\n", check=True,
                          capture_output=True, text=True)


def log_wandb(manifest: Mapping[str, Any], tables_dir: Path | str | None, *, project: str | None = None, name: str | None = None,
              artifact: str | None = None, config: Mapping[str, Any] | None = None) -> str:
    """Every PNG as ``wandb.Image`` under its manifest key (`<leg>/arms`, `summary/<family>`, `diag/<what>`), a
    ``wandb.Table`` per leg from ``tables/<leg>.csv`` and ``<leg>_arms.csv``, and the artifact built from the MANIFEST's
    files; the run config and the artifact metadata carry the provisional status, the evidence summary and its hash. The run
    name and the artifact collection default to the manifest's EFFECTIVE names (official, or `-PARTIAL-<evidence hash>`);
    an explicit official name for a provisional manifest is refused."""
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "") != "offline":
        raise RuntimeError("WANDB_API_KEY is not in the environment (the site's shell exports it); refusing to log")
    names = publication_names(manifest)
    refuse_official_namespace(manifest, artifact=artifact, run_name=name)
    name = name or names["run_name"]
    artifact = artifact or names["artifact"]
    import pandas as pd
    import wandb

    status = {"partial": manifest["partial"], "table_status": manifest.get("table_status"), "absent_inputs": manifest.get("absent_inputs"),
              "provisional_reasons": manifest.get("provisional_reasons"), "evidence": manifest.get("evidence"), "rejected_legs": manifest.get("rejected_legs"),
              "evidence_hash": names["evidence_hash"], "publication": {"run_name": name, "artifact": artifact}}
    if project is None:
        from sisteer.wandb_entity import wandb_project
        project = wandb_project(WANDB_PURPOSE)
    run = wandb.init(project=project, name=name, job_type="analysis", config={**dict(config or {}), **status}, reinit=True)
    try:
        imgs = {k: wandb.Image(v["path"]) for k, v in manifest["figures"].items() if v["path"].endswith(".png")}
        run.log(imgs)
        if tables_dir:
            for leg in manifest["legs"]:
                for suffix in ("", "_arms"):
                    p = Path(tables_dir) / f"{leg}{suffix}.csv"
                    if p.is_file():
                        run.log({f"{leg}/table{suffix}": wandb.Table(dataframe=pd.read_csv(p))})
        art = wandb.Artifact(artifact, type="figures", metadata={**status, "summary_sha256": manifest["summary_sha256"]})
        for k, v in manifest["figures"].items():
            art.add_file(v["path"], name=k.replace("/", "__") + Path(v["path"]).suffix)
        art.add_file(str(Path(manifest["figures"][next(iter(manifest["figures"]))]["path"]).parent / "figures_manifest.json"), name="figures_manifest.json")
        run.log_artifact(art)
        return run.url
    finally:
        run.finish()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True, help="tables/battery_summary.json written by battery_tables.py")
    ap.add_argument("--out", type=Path, required=True, help="figure directory")
    ap.add_argument("--legs", default=None, help="comma-separated leg ids to draw (a display filter; values are never recomputed)")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--mirror-dest", type=Path, default=None,
                    help=f"mirror directory (default: {DEFAULT_MIRROR} for an official run, <that>-{PARTIAL_TAG}-<evidence hash> for a provisional one; "
                         "the official directory is refused for a provisional run)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write figures only; never mirror or log (the default when neither flag is given)")
    a = ap.parse_args(argv)
    summary = load_summary(a.summary)
    legs = [x.strip() for x in a.legs.split(",")] if a.legs else None
    res = build_all(summary, a.out, legs=legs, summary_path=a.summary)
    m = res["manifest"]
    pub = m["publication"]
    print(f"[battery_figures] {len(m['figures'])} files under {a.out}" + (f" — PARTIAL / PROVISIONAL ({'; '.join(m['provisional_reasons'])})" if m["partial"] else ""))
    print(f"[battery_figures] publication namespace: run {pub['run_name']!r} · artifact {pub['artifact']!r} · mirror {pub['mirror_dest']}"
          + ("" if pub["official"] else f" (PARTIAL, evidence hash {pub['evidence_hash']})"))
    if m["rejected_legs"]:
        print(f"[battery_figures] REJECTED legs (engine identity): {', '.join(m['rejected_legs'])}")
    if a.dry_run or not (a.mirror or a.wandb):
        print("[battery_figures] dry run: no mirror, no W&B")
        return 0
    try:
        if a.mirror:
            dest = a.mirror_dest if a.mirror_dest is not None else Path(pub["mirror_dest"])
            mirror(m, a.out, dest)
            print(f"[battery_figures] mirrored {len(m['figures'])} files to {dest}")
        if a.wandb:
            print("[battery_figures] wandb:", log_wandb(m, Path(a.summary).parent, config={"summary": str(a.summary)}))
    except RuntimeError as exc:
        print(f"[battery_figures] REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
