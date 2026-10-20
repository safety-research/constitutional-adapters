"""Petri Bloom report: tables + figures from a ``petri_stats.analyze``
result.

Outputs, per run:
- **tables** (pandas frames, written as CSV + folded into ``report.md``):
  base screen, confirmatory per-method macro, cellwise supplementary,
  exploratory per-behavior deltas, by-cell, by-SI, by-class, Class B
  baselines, modality views, the completeness map.
- **figures** rendered with matplotlib and (a) logged to **W&B project
  ``petri``** — its own project, kept separate from this repository's
  behavioral runs — and (b) mirrored to a per-run subdirectory of the
  published figures directory so the full-resolution originals are one
  URL away.
- **report.md** naming every figure by its W&B panel key + run name + the
  mirror URL, and stating the run's limitations.

Import discipline: ``matplotlib`` and ``wandb`` are imported ONLY inside
the functions that need them, behind loud guards — the CPU test suite
collects and runs this module without either installed. Everything else
(tables, report text) is plain pandas/stdlib.

The misalignment axis is reported ALONE: this module has no code path
that reads the two-SI misuse attack-success table, and the report
states that limitation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from sisteer.capeval.petri import petri_stats as ps


class PetriReportError(RuntimeError):
    """A report-build refusal (missing dep, malformed analysis, …)."""


#: Stated STATIC limitations. The report carries
#: all of them verbatim; counts come from the analysis itself. The
#: single-target-model line is deliberately NOT here: ``limitations()`` below
#: derives it from the analysis provenance at render time. It used to be
#: hardcoded, and the hardcoded id shipped a Llama-3.1 line on a Gemma-4-E4B
#: audit (caught on the published petri_gemma4e4b report, 2026-09-01) — a
#: model name in this report may only ever come from the analysis itself.
LIMITATIONS = (
    ps.MISALIGNMENT_AXIS_STATEMENT,
    "Bounded scenario counts (~10 frozen scenarios per behavior; the "
    "exact per-behavior counts are recorded in the analysis) — power is "
    "limited and never overstated.",
    "Unadjusted per-behavior intervals are DESCRIPTIVE/EXPLORATORY only; "
    "the confirmatory result is the per-method macro family under its "
    "max-T simultaneous band.",
    "Auditor fallback creates an instrument-selection tradeoff. The "
    "Sonnet-only view holds the instrument constant but conditions on "
    "content-correlated shedding; the all-audited view retains the scored "
    "sample but varies the instrument. Neither view is unbiased.",
)


def limitations(analysis: Mapping[str, Any]) -> tuple:
    """The stated limitations for THIS analysis — the single-target line is
    bound to ``analysis['provenance']`` at render time, never hardcoded.

    When the analysis was built in unit-frame mode (no provenance mapping
    bound, ``validated`` False) the line says the identity is not bound
    rather than guessing one: emitting no model name is honest; emitting a
    wrong one is the exact bug this function replaces.
    """
    prov = analysis.get("provenance") or {}
    model_id = prov.get("model_id")
    if model_id:
        rev = str(prov.get("model_revision") or "")[:8]
        single = (f"Single target model: {model_id}"
                  + (f" @ {rev}" if rev else "")
                  + " — no claim generalizes across models from this audit "
                  "alone.")
    else:
        single = ("Single target model (identity not bound: analysis built "
                  "without a provenance mapping) — no claim generalizes "
                  "across models from this audit alone.")
    return (single,) + LIMITATIONS

#: Figure base URL (where the site serves its figure mirror): $SISTEER_FIGS_URL_BASE from the environment, else the
#: variable's own name, so an unconfigured site renders visibly unresolved links rather than a maintainer's host (the
#: identifier pass, RELEASE_CHECKLIST.md item 1). Override per build via build(..., figs_url_base=...).
DEFAULT_FIGS_URL_BASE = os.environ.get("SISTEER_FIGS_URL_BASE") or "$SISTEER_FIGS_URL_BASE"
#: W&B project — deliberately its own project, never this repository's
#: behavioral-runs project.
WANDB_PROJECT = "petri"

# Validated dataviz reference palette (light mode) — categorical order,
# diverging pair, neutral midpoint, ink tokens (dataviz skill).
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_RED = "#e34948"
_GRAY_MID = "#f0efec"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_SURFACE = "#fcfcfb"
_CATEGORICAL = (_BLUE, _ORANGE, _AQUA)


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless; never require a display
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:  # pragma: no cover - venv carries matplotlib
        raise PetriReportError(
            "matplotlib is required to render the Petri figures — install "
            "it in the reporting environment (the CPU test venv carries "
            "it; figures are never rendered on cluster nodes)") from e
    return matplotlib.pyplot


# --- tables ---------------------------------------------------------------------


def _instrument_columns(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten one contrast's two auditor-instrument views for CSV/Markdown.

    The full successful-leg distributions stay first-class columns rather
    than being reduced to a warning flag. Entries without the new analysis
    block remain readable and emit empty additive columns.
    """
    views = entry.get("instrument_views") or {}
    cells = list((views.get("cells") or {}).items())
    (cell_a, dist_a) = cells[0] if cells else (None, {})
    (cell_b, dist_b) = cells[1] if len(cells) > 1 else (None, {})
    sonnet = views.get("sonnet_only") or {}
    audited = views.get("all_audited") or {}
    criterion = views.get("pooling_criterion") or {}
    headline = views.get("pooled_headline") or {}
    return {
        "auditor_cell_a": cell_a,
        "auditor_cell_a_leg_distribution": (
            json.dumps(dist_a, sort_keys=True) if dist_a else None),
        "auditor_cell_b": cell_b,
        "auditor_cell_b_leg_distribution": (
            json.dumps(dist_b, sort_keys=True) if dist_b else None),
        "auditor_leg_tv_distance": criterion.get("distance"),
        "auditor_leg_tv_threshold": criterion.get("threshold"),
        "pooled_headline_status": headline.get("status"),
        "pooled_headline_reason": headline.get("reason"),
        "sonnet_only_status": sonnet.get("status"),
        "sonnet_only_delta": sonnet.get("delta"),
        "sonnet_only_ci95_lo":
            (sonnet.get("ci95_pointwise") or [None, None])[0],
        "sonnet_only_ci95_hi":
            (sonnet.get("ci95_pointwise") or [None, None])[1],
        "sonnet_only_n_common_by_behavior": (
            json.dumps(sonnet.get("n_common_by_behavior"), sort_keys=True)
            if sonnet.get("n_common_by_behavior") is not None else None),
        "all_audited_status": audited.get("status"),
        "all_audited_delta": audited.get("delta"),
        "all_audited_ci95_lo":
            (audited.get("ci95_pointwise") or [None, None])[0],
        "all_audited_ci95_hi":
            (audited.get("ci95_pointwise") or [None, None])[1],
        "all_audited_n_by_behavior": (
            json.dumps(audited.get("n_by_behavior"), sort_keys=True)
            if audited.get("n_by_behavior") is not None else None),
        "auditor_instrument_bias_note": views.get("bias_note"),
    }


def build_tables(analysis: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Every report table as a pandas frame (JSON-serializable values)."""
    if analysis.get("schema_version") != 1:
        raise PetriReportError(
            f"unknown analysis schema_version "
            f"{analysis.get('schema_version')!r} — rebuild the analysis "
            "with this branch's petri_stats")
    tables: Dict[str, pd.DataFrame] = {}

    bs = analysis["base_screen"]
    screen_ran = bs.get("status", "ok") == "ok"
    # an UNAVAILABLE screen invalidates the frozen family as a whole: no
    # behavior's eligibility is known — not even one whose own base slice
    # screened fine — so `eligible` is null on EVERY row, never False, and
    # the screen's status sits beside the per-behavior status (review
    # 2026-09-05 P2-a: an unaffected behavior with mean 4.0 rendered
    # eligible=False)
    tables["base_screen"] = pd.DataFrame([
        {"behavior": b, "modality": v["modality"],
         "status": v.get("status", "ok"), "reason": v.get("reason"),
         "screen_status": bs.get("status", "ok"),
         "base_mean": v["base_mean"], "n_items": v["n_items"],
         "floor": bs["floor"],
         "no_headroom": v["no_headroom"],
         "eligible": (b in bs["eligible_behaviors"]) if screen_ran else None}
        for b, v in bs["behaviors"].items()])

    pm = analysis["per_method_macro"]
    per_method_rows = []
    for m, e in pm["methods"].items():
        per_method_rows.append({
            "method": m, "status": e["status"],
            "delta_constitution_minus_safety": e.get("delta"),
            "ci95_maxt_lo":
                (e.get("ci95_simultaneous_maxt") or [None, None])[0],
            "ci95_maxt_hi":
                (e.get("ci95_simultaneous_maxt") or [None, None])[1],
            "ci95_pointwise_lo":
                (e.get("ci95_pointwise") or [None, None])[0],
            "ci95_pointwise_hi":
                (e.get("ci95_pointwise") or [None, None])[1],
            "p_boot": e.get("p_boot"), "p_holm": e.get("p_holm"),
            "reason": e.get("reason"),
            **_instrument_columns(e),
        })
    tables["per_method_macro"] = pd.DataFrame(per_method_rows)

    tables["cellwise_supplementary"] = pd.DataFrame([
        {"method": v["method"], "behavior": v["behavior"],
         "status": v.get("status"), "delta": v.get("delta"),
         "ci95_maxt_lo":
             (v.get("ci95_simultaneous_maxt") or [None, None])[0],
         "ci95_maxt_hi":
             (v.get("ci95_simultaneous_maxt") or [None, None])[1],
         "reason": v.get("reason"), **_instrument_columns(v)}
        for v in analysis["cellwise_supplementary"]["cells"].values()])

    # NO-HEADROOM behaviors carry a status + diagnostics, never a paired
    # delta/CI — .get() keeps those rows without estimates.
    # Every leg diagnostic rides the row — leg means, item counts and the
    # refused leg by name — so a NO-HEADROOM row whose leg the auditor fully
    # refused never renders identically to the unrefused row (review
    # 2026-09-05 P2-b: the JSON knew constitution_mean=None but the CSV kept
    # only base_mean)
    def _exploratory_row(v: Mapping[str, Any]) -> Dict[str, Any]:
        diag = v.get("diagnostics") or {}
        return {
            "method": v["method"], "behavior": v["behavior"],
            "status": v.get("status"),
            "delta": v.get("delta"),
            "ci95_lo": (v.get("ci95_unadjusted") or [None, None])[0],
            "ci95_hi": (v.get("ci95_unadjusted") or [None, None])[1],
            "base_mean": diag.get("base_mean"),
            "constitution_mean": diag.get("constitution_mean"),
            "safety_mean": diag.get("safety_mean"),
            "constitution_n_items": diag.get("constitution_n_items"),
            "safety_n_items": diag.get("safety_n_items"),
            "refused_legs": ", ".join(diag.get("refused_legs") or []) or None,
            "refusal": diag.get("refusal"),
            "eligible": v["eligible"], "no_headroom": v["no_headroom"],
            "reason": v.get("reason"),
            "unavailable_behaviors": v.get("unavailable_behaviors"),
            **_instrument_columns(v)}

    tables["by_behavior_exploratory"] = pd.DataFrame([
        _exploratory_row(v)
        for v in analysis["by_behavior_exploratory"]["cells"].values()])

    by_cell_rows = []
    for cell, e in analysis["by_cell"].items():
        row = {"cell": cell, "cell_class": e["cell_class"],
               "si_id": e["si_id"], "method": e["method"],
               "control_of": e["control_of"], "engine": e["engine"]}
        for b, v in e["behaviors"].items():
            row[f"{b}"] = v["mean_behavior_score"]
            # an empty mean is explained by its item count in the same row
            row[f"{b}_n_items"] = v.get("n_items")
        vb = e.get("vs_base") or {}
        row["vs_base_status"] = vb.get("status")
        row["vs_base_leg"] = vb.get("base_leg")
        row["vs_base_macro_delta"] = vb.get("macro_delta")
        # a PAIRING_REFUSED/UNAVAILABLE comparison keeps its WHY: refusals
        # are surfaced, never stripped to a bare status
        row["vs_base_reason"] = vb.get("reason")
        row.update({f"vs_base_{k}": v
                    for k, v in _instrument_columns(vb).items()})
        # the cell's audited footing beside its scores (petri_runner module
        # docstring: an auditor shed is a first-class outcome, never a score)
        af = e.get("audited_footing") or {}
        aud = af.get("audited") or {}
        ref = af.get("auditor_refused") or {}
        row["audited_items"] = aud.get("items")
        row["expected_items"] = aud.get("expected")
        row["auditor_refused"] = ref.get("total")
        row["auditor_refused_content_correlated"] = ref.get("content_correlated")
        row["auditor_refused_outage"] = ref.get("outage")
        row["auditor_refused_undetermined"] = ref.get("undetermined")
        chain = af.get("auditor_chain") or {}
        row["auditor_sonnet_only"] = chain.get("sonnet_only")
        row["auditor_continued_by_grok"] = chain.get("continued_by_grok")
        row["auditor_continued_by_qwen"] = chain.get("continued_by_qwen")
        row["auditor_refused_all_legs"] = chain.get(
            "auditor_refused_all_legs")
        row["auditor_unavailable"] = chain.get("auditor_unavailable")
        by_cell_rows.append(row)
    tables["by_cell"] = pd.DataFrame(by_cell_rows)

    bc = analysis["by_class"]
    aggregate_details = bc["class_a"].get("per_si_macro") or {}
    # the per-SI aggregate is over the FIXED method family: the row names
    # the expected and available method counts and every unavailable cell,
    # so an empty mean is never read as "a mean over fewer methods"
    by_class_rows = []
    for si, mean in bc["class_a"]["per_si_macro_mean"].items():
        agg = aggregate_details.get(si) or {}
        by_class_rows.append(
            {"cell_class": "class_a", "si": si,
             "status": agg.get("status", "ok"),
             "reason": agg.get("reason"),
             "per_si_macro_mean": mean,
             "n_methods_expected": agg.get("n_methods_expected"),
             "n_methods_available": agg.get("n_methods_available"),
             "unavailable_cells": agg.get("unavailable_cells", []),
             "n_confirmatory_methods_ok":
                 bc["class_a"]["n_methods_published"],
             "n_cells_with_rows": len(bc["class_a"]["cells"]),
             "note": None})
    by_class_rows.append(
        {"cell_class": "class_b", "si": None, "status": None, "reason": None,
         "per_si_macro_mean": None, "n_methods_expected": None,
         "n_methods_available": None, "unavailable_cells": None,
         "n_confirmatory_methods_ok": None,
         "n_cells_with_rows": len(bc["class_b"]["cells"]),
         "note": bc["class_b"]["note"]})
    tables["by_class"] = pd.DataFrame(by_class_rows)

    tables["by_si"] = pd.DataFrame([
        {"method": m, "si": si, "status": leg["status"],
         "macro_mean": leg.get("macro_mean"),
         # the leg's macro verdict sits beside its mean: an empty mean on an
         # ok leg is explained here, never left as a silent None
         "macro_status": leg.get("macro_status",
                                 leg["status"] if leg["status"] != "ok"
                                 else ("ok" if leg.get("macro_mean")
                                       is not None else "UNAVAILABLE")),
         "macro_reason": leg.get("macro_reason", leg.get("reason")),
         "unavailable_behaviors": leg.get("unavailable_behaviors"),
         **{f"{b}": v for b, v in
            (leg.get("per_behavior_mean") or {}).items()},
         **{f"{b}_n_items": v for b, v in
            (leg.get("per_behavior_n_items") or {}).items()}}
        for m, legs in analysis["by_si"].items()
        for si, leg in legs.items()])

    cb_rows = []
    for cell, e in analysis["class_b_baselines"].items():
        if e["status"] != "ok":
            cb_rows.append({"cell": cell, "status": e["status"],
                            "reason": e.get("reason")})
            continue
        for b, v in e["behaviors"].items():
            # a fully refused slice is UNAVAILABLE on its own row — the
            # served cell's status is not the slice's status
            cb_rows.append({"cell": cell,
                            "status": v.get("status", "ok" if v["mean"]
                                            is not None else "UNAVAILABLE"),
                            "behavior": b,
                            "mean": v["mean"],
                            "ci95_lo":
                                (v.get("ci95_unadjusted") or [None, None])[0],
                            "ci95_hi":
                                (v.get("ci95_unadjusted") or [None, None])[1],
                            "n_items": v.get("n_items"),
                            "no_headroom": v["no_headroom"],
                            "reason": v.get("reason")})
    tables["class_b_baselines"] = pd.DataFrame(cb_rows)

    mv_rows = []
    for modality in ("conversation", "agent"):
        view = analysis["modality_views"].get(modality) or {}
        for m, e in (view.get("methods") or {}).items():
            mv_rows.append({
                "modality": modality, "method": m, "status": e["status"],
                "delta": e.get("delta"),
                "ci95_lo": (e.get("ci95_pointwise") or [None, None])[0],
                "ci95_hi": (e.get("ci95_pointwise") or [None, None])[1],
                "reason": e.get("reason"), **_instrument_columns(e)})
    tables["modality_views"] = pd.DataFrame(mv_rows)

    floor_rows = []
    for floor_key, floor in (analysis.get("floor_sensitivity", {})
                             .get("floors", {})).items():
        for method, entry in floor.get("per_method_macro", {}).items():
            floor_rows.append({
                "floor": floor_key, "method": method,
                "status": entry.get("status"), "delta": entry.get("delta"),
                "ci95_lo":
                    (entry.get("ci95_pointwise") or [None, None])[0],
                "ci95_hi":
                    (entry.get("ci95_pointwise") or [None, None])[1],
                "reason": entry.get("reason"),
                **_instrument_columns(entry),
            })
    tables["floor_sensitivity"] = pd.DataFrame(floor_rows)

    comp = analysis["completeness"]
    tables["completeness"] = pd.DataFrame(
        [{"cell": c, "kind": "canonical", **v}
         for c, v in comp["cells"].items()]
        + [{"cell": c, "kind": "control", **v}
           for c, v in comp["controls"].items()])

    ref_rows = []
    for key, e in analysis["prompted_references"]["contrasts"].items():
        ref_rows.append({"contrast": key, "status": e["status"],
                         "cross_path": e.get("cross_path"),
                         "macro_delta": e.get("macro_delta"),
                         "reason": e.get("reason"),
                         **_instrument_columns(e)})
    tables["prompted_references"] = pd.DataFrame(ref_rows)
    return tables


# --- figures --------------------------------------------------------------------


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelcolor=_INK)
    ax.grid(True, axis="x", color=_GRAY_MID, linewidth=0.8)
    ax.set_axisbelow(True)


def render_figures(analysis: Mapping[str, Any], figs_dir: Path,
                   run_name: str) -> Dict[str, Path]:
    """Render every figure; returns ``{wandb_panel_key: path}``. Filenames
    are stable and descriptive (``final_<what>.png``) so runs diff."""
    plt = _require_matplotlib()
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    def _save(fig, panel_key: str, filename: str) -> None:
        path = figs_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor=_SURFACE)
        plt.close(fig)  # always — long runs leak figure memory otherwise
        out[panel_key] = path

    # 1. Base-elicitability screen: magnitude → single-hue bars + the
    #    pre-registered floor as a labelled reference line. NO-HEADROOM
    #    bars carry an explicit text flag (never color alone).
    #    An UNAVAILABLE row (every base item a terminal auditor refusal)
    #    gets NO bar: a bar at zero would place the refusal on the score
    #    axis, outside the 1-10 range, as if it were a measurement. Its
    #    label is anchored in axes coordinates, so the row encodes no
    #    x-value at all.
    bs = analysis["base_screen"]
    behaviors = list(bs["behaviors"])
    means = [bs["behaviors"][b]["base_mean"] for b in behaviors]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(behaviors) + 1.6))
    y = np.arange(len(behaviors))
    available = [i for i, mean in enumerate(means) if mean is not None]
    if available:
        ax.barh(y[available], [means[i] for i in available], height=0.62,
                color=_BLUE, zorder=3)
    ax.axvline(bs["floor"], color=_RED, linestyle="--", linewidth=1.4,
               zorder=4)
    ax.text(bs["floor"], len(behaviors) - 0.35,
            f"  pre-registered floor {bs['floor']:g}", color=_RED,
            fontsize=8, va="bottom")
    for i, b in enumerate(behaviors):
        if means[i] is None:
            ax.text(0.01, i, "UNAVAILABLE — terminal auditor refusals "
                    "(no base mean)", va="center", fontsize=8, color=_INK2,
                    style="italic", transform=ax.get_yaxis_transform())
            continue
        label = f"{means[i]:.2f}"
        if bs["behaviors"][b]["no_headroom"]:
            label += "  NO-HEADROOM"
        ax.text(means[i] + 0.08, i, label, va="center", fontsize=8,
                color=_INK)
    ax.set_yticks(y, behaviors, fontsize=8)
    ax.set_xlim(0, 10.4)
    ax.set_xlabel("undefended-base mean behavior score (1-10)",
                  fontsize=9, color=_INK)
    ax.set_title("Base-elicitability screen (plan §8)", fontsize=10,
                 color=_INK)
    _style_axes(ax)
    _save(fig, "petri/base_elicitability", "final_base_elicitability_screen.png")

    # 2. Confirmatory per-method macro contrasts: polarity around 0 →
    #    zero reference line; simultaneous (thin) vs pointwise (thick)
    #    intervals, both legended.
    pm = analysis["per_method_macro"]
    ok = [(m, e) for m, e in pm["methods"].items() if e["status"] == "ok"]
    if ok:
        fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(ok) + 1.8))
        y = np.arange(len(ok))
        maxt_labelled = False
        for i, (m, e) in enumerate(ok):
            # the max-T band is UNAVAILABLE when the fixed 8-method family
            # is not fully populated (H7) — plot pointwise only then
            band = e.get("ci95_simultaneous_maxt")
            if band is not None:
                ax.plot(band, [i, i], color=_INK2, linewidth=1.2, zorder=3,
                        label=(None if maxt_labelled
                               else "max-T simultaneous 95%"))
                maxt_labelled = True
            lo_p, hi_p = e["ci95_pointwise"]
            ax.plot([lo_p, hi_p], [i, i], color=_BLUE, linewidth=3.4,
                    solid_capstyle="butt", zorder=4,
                    label="pointwise 95% (descriptive)" if i == 0 else None)
            ax.plot([e["delta"]], [i], "o", color=_INK, markersize=5,
                    zorder=5)
        ax.axvline(0, color=_INK2, linewidth=1.0)
        ax.set_yticks(y, [m for m, _ in ok], fontsize=8)
        ax.set_xlabel("macro constitution − safety (behavior score)",
                      fontsize=9, color=_INK)
        ax.set_title("CONFIRMATORY: per-method macro contrasts "
                     "(equal-weight eligible behaviors)", fontsize=10,
                     color=_INK)
        ax.legend(fontsize=8, frameon=False, loc="best")
        _style_axes(ax)
        ax.grid(True, axis="y", color=_GRAY_MID, linewidth=0.0)
        _save(fig, "petri/macro_contrasts",
              "final_macro_contrasts_confirmatory.png")

    # 3. Cellwise supplementary deltas: polarity → diverging blue↔red with
    #    a neutral midpoint. An unavailable slot keeps its row/column and
    #    is drawn as the neutral "bad" colour WITH an explicit label — it
    #    encodes no value and is never a dropped row.
    cw = analysis["cellwise_supplementary"]["cells"]
    cw_ok = [v for v in cw.values() if v.get("status", "ok") == "ok"]
    methods = [m for m in analysis["per_method_macro"]["family"]
               if any(v["method"] == m for v in cw.values())]
    cw_behaviors = sorted({v["behavior"] for v in cw.values()})
    if methods and cw_behaviors:
        grid = np.full((len(methods), len(cw_behaviors)), np.nan)
        for v in cw_ok:
            grid[methods.index(v["method"]),
                 cw_behaviors.index(v["behavior"])] = v["delta"]
        cmap = LinearSegmentedColormap.from_list(
            "petri_div", [_BLUE, _GRAY_MID, _RED])
        cmap.set_bad(color="#e5e4e1")
        finite = np.abs(grid[np.isfinite(grid)])
        vmax = max(1e-9, float(finite.max())) if finite.size else 1.0
        fig, ax = plt.subplots(
            figsize=(1.1 * len(cw_behaviors) + 2.4,
                     0.55 * len(methods) + 1.6))
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-vmax,
                       vmax=vmax, aspect="auto")
        for i in range(len(methods)):
            for j in range(len(cw_behaviors)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=8, color=_INK2, style="italic")
                else:
                    ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center",
                            va="center", fontsize=8, color=_INK)
        ax.set_xticks(range(len(cw_behaviors)), cw_behaviors, fontsize=8,
                      rotation=30, ha="right")
        ax.set_yticks(range(len(methods)), methods, fontsize=8)
        ax.set_title("Supplementary: cellwise constitution − safety "
                     "(max-T family)", fontsize=10, color=_INK)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Δ behavior score", fontsize=8, color=_INK2)
        _save(fig, "petri/cellwise_deltas",
              "final_cellwise_deltas_supplementary.png")

    # 4. Class B baselines: ≤3 series → the first three categorical slots
    #    (validated all-pairs), legend present.
    #    A fully refused (cell, behavior) slice has mean None: it gets NO
    #    bar (never a numeric stand-in) and an "UNAVAILABLE" label anchored
    #    to the axis floor in axes coordinates, so it encodes no score.
    cb = analysis["class_b_baselines"]
    served = [(c, e) for c, e in cb.items() if e["status"] == "ok"]
    if served:
        cb_behaviors = list(served[0][1]["behaviors"])
        x = np.arange(len(cb_behaviors))
        width = 0.8 / max(1, len(served))
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        for k, (cell, e) in enumerate(served):
            offsets = x + (k - (len(served) - 1) / 2) * width
            vals = [e["behaviors"][b]["mean"] for b in cb_behaviors]
            avail = [j for j, v in enumerate(vals) if v is not None]
            if avail:
                ax.bar(offsets[avail], [vals[j] for j in avail],
                       width=width * 0.92, color=_CATEGORICAL[k % 3],
                       label=cell, zorder=3)
            for j, v in enumerate(vals):
                if v is None:
                    ax.text(offsets[j], 0.02, "UNAVAILABLE", rotation=90,
                            ha="center", va="bottom", fontsize=7,
                            color=_INK2, style="italic",
                            transform=ax.get_xaxis_transform())
        ax.set_xticks(x, cb_behaviors, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0, 10.4)  # the judge scale, so bars read as scores
        ax.set_ylabel("mean behavior score (1-10)", fontsize=9, color=_INK)
        ax.set_title("Class B SI-agnostic baselines (never in the SI "
                     "contrast)", fontsize=10, color=_INK)
        ax.legend(fontsize=8, frameon=False)
        _style_axes(ax)
        ax.grid(True, axis="y", color=_GRAY_MID, linewidth=0.8)
        ax.grid(False, axis="x")
        _save(fig, "petri/class_b_baselines", "final_class_b_baselines.png")

    # 5. Modality-stratified macro (secondary): 2 series → blue/orange.
    mv = analysis["modality_views"]
    mv_methods = sorted({m for mod in ("conversation", "agent")
                         for m, e in (mv.get(mod, {}).get("methods")
                                      or {}).items()
                         if e["status"] == "ok"})
    if mv_methods:
        fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(mv_methods) + 1.8))
        y = np.arange(len(mv_methods))
        offsets = {"conversation": -0.16, "agent": 0.16}
        colors = {"conversation": _BLUE, "agent": _ORANGE}
        for mod in ("conversation", "agent"):
            pts, ys, err = [], [], []
            for i, m in enumerate(mv_methods):
                e = (mv.get(mod, {}).get("methods") or {}).get(m) or {}
                if e.get("status") != "ok":
                    continue
                pts.append(e["delta"])
                ys.append(i + offsets[mod])
                lo, hi = e["ci95_pointwise"]
                err.append((e["delta"] - lo, hi - e["delta"]))
            if pts:
                err_arr = np.array(err).T
                ax.errorbar(pts, ys, xerr=err_arr, fmt="o", markersize=5,
                            color=colors[mod], linewidth=1.6, label=mod,
                            zorder=4)
        ax.axvline(0, color=_INK2, linewidth=1.0)
        ax.set_yticks(y, mv_methods, fontsize=8)
        ax.set_xlabel("macro constitution − safety (behavior score)",
                      fontsize=9, color=_INK)
        ax.set_title("SECONDARY: modality-stratified macro views",
                     fontsize=10, color=_INK)
        ax.legend(fontsize=8, frameon=False)
        _style_axes(ax)
        _save(fig, "petri/modality_views", "final_modality_macro_secondary.png")

    return out


# --- report assembly --------------------------------------------------------------


def _wandb_log(analysis: Mapping[str, Any],
               tables: Mapping[str, pd.DataFrame],
               figures: Mapping[str, Path], run_name: str,
               wandb_mode: str) -> Optional[str]:
    try:
        import wandb
    except ImportError as e:
        raise PetriReportError(
            "wandb is required for a non-disabled report run — install it "
            "or pass wandb_mode='disabled'") from e
    run = wandb.init(project=WANDB_PROJECT, name=run_name, mode=wandb_mode,
                     config={"confirmatory_family":
                             analysis["confirmatory_family"],
                             "seed": analysis["seed"],
                             "n_boot": analysis["n_boot"],
                             "base_floor": analysis["base_screen"]["floor"]})
    for panel_key, path in figures.items():
        run.log({panel_key: wandb.Image(str(path))})
    for name, table in tables.items():
        if not table.empty:
            run.log({f"petri/tables/{name}":
                     wandb.Table(dataframe=table.astype(object)
                                 .where(table.notna(), None))})
    if figures:
        art = wandb.Artifact(f"figures-{run_name}", type="figures")
        art.add_dir(str(next(iter(figures.values())).parent))
        run.log_artifact(art)
    url = getattr(run, "url", None)
    run.finish()
    return url


def _mirror_figs(figures: Mapping[str, Path], mirror_dir: Path) -> Path:
    """Copy the rendered figures into the served mirror directory —
    additive, one per-run subdirectory."""
    mirror_dir = Path(mirror_dir)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for path in figures.values():
        shutil.copy2(path, mirror_dir / path.name)
    return mirror_dir


def eligible_behaviors_line(analysis: Mapping[str, Any]) -> str:
    """The one-line eligible-behaviors statement for the report header AND
    the driver's console summary. When the screen RAN it states the
    computed set (an empty list is an honest NO_ELIGIBLE_BEHAVIORS result);
    when the screen could not run it says UNAVAILABLE and why, never a
    computed-looking ``[]`` (review 2026-09-05 P2-a)."""
    bs = analysis["base_screen"]
    head = f"eligible behaviors (base screen, floor {bs['floor']:g}): "
    if bs.get("status", "ok") == "ok":
        return head + f"{bs['eligible_behaviors']}"
    return (head + f"**{bs['status']}** — the screen could not run for "
            f"behavior(s) {bs.get('unavailable_behaviors')}; no eligible "
            "set is claimed")


def _audited_items_line(analysis: Mapping[str, Any]) -> str:
    """Per-behavior audited scenario set from the AUTHORITATIVE footing —
    a number where the footing declares it, the literal ``UNAVAILABLE``
    where it does not (never a count of surviving rows)."""
    per_b = (analysis.get("audited_items") or {}).get("behaviors") or {}
    shown = {b: (e["n_items"] if e.get("n_items") is not None
                 else e.get("status", "UNAVAILABLE"))
             for b, e in per_b.items()}
    return ("audited scenario set per behavior (authoritative, from the "
            f"runner's footing): {json.dumps(shown)}")


def _md_table(df: pd.DataFrame) -> str:
    """A plain markdown table (no `tabulate` dependency — the capeval test
    venv doesn't carry it). Floats rendered to 4 significant digits."""
    def _fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in df.columns)
                     + " |")
    return "\n".join(lines)


def render_markdown(analysis: Mapping[str, Any],
                    tables: Mapping[str, pd.DataFrame],
                    figures: Mapping[str, Path], run_name: str,
                    figs_url_base: str = DEFAULT_FIGS_URL_BASE,
                    wandb_url: Optional[str] = None) -> str:
    """The report text: confirmatory result first, figures named by W&B
    panel key + run name + the grading host URL, limitations stated."""
    lines = [f"# Petri Bloom two-SI misalignment audit — `{run_name}`", ""]
    lines += [f"- confirmatory family: `{analysis['confirmatory_family']}`"
              " (per-method macro constitution − safety, max-T"
              " simultaneous 95% band; Holm fallback reported)",
              f"- bootstrap: n_boot={analysis['n_boot']},"
              f" seed={analysis['seed']} (deterministic)",
              f"- {_audited_items_line(analysis)}",
              # the schema-v1 `counts` field: the base cell's numeric items
              f"- base cell numeric items per behavior: "
              f"{json.dumps(analysis['counts'])}",
              f"- {eligible_behaviors_line(analysis)}"]
    bs = analysis["base_screen"]
    if bs.get("status", "ok") != "ok":
        # an unavailable screen is named as such, with its reason — never
        # presented as a computed empty eligible set
        lines.append(f"- base-elicitability screen: **{bs['status']}** — "
                     f"{bs.get('reason')} (unavailable behaviors: "
                     f"{bs.get('unavailable_behaviors')})")
    if wandb_url:
        lines.append(f"- W&B run: {wandb_url}")
    lines.append("")

    policy = analysis.get("auditor_instrument_policy") or {}
    lines.append("## Auditor-instrument sensitivity")
    lines.append(
        "Every between-cell contrast is shown in two adjacent columns: "
        "the common Sonnet-only subset, which holds the auditor constant "
        "but selects on content-correlated shedding, and all audited "
        "samples, which retains the scored sample but allows the auditor "
        "to vary. Neither view is unbiased; their movement is the result.")
    if policy:
        lines.append(
            f"A single pooled headline is refused when the successful-leg "
            f"distributions' total-variation distance is greater than the "
            f"recorded threshold {policy.get('pooling_threshold')}. The "
            "full leg distribution remains in every contrast row.")
    refusals = []
    for method, entry in analysis.get("per_method_macro", {}).get(
            "methods", {}).items():
        if entry.get("status") == "INSTRUMENT_MIX_REFUSED":
            refusals.append(f"- `{method}`: {entry.get('reason')}")
    if refusals:
        lines.append("Pooled confirmatory headlines refused:")
        lines.extend(refusals)
    lines.append("")

    lines.append("## Figures")
    for panel_key, path in figures.items():
        lines.append(f"- `{panel_key}` in run `{run_name}` — full-res at "
                     f"{figs_url_base.rstrip('/')}/{run_name}/{path.name}")
    lines.append("")

    for name in ("base_screen", "per_method_macro", "cellwise_supplementary",
                 "by_behavior_exploratory", "modality_views",
                 "floor_sensitivity", "by_si",
                 "by_class", "class_b_baselines", "by_cell",
                 "prompted_references", "completeness"):
        table = tables.get(name)
        if table is None or table.empty:
            continue
        lines.append(f"## {name}")
        if name in ("by_behavior_exploratory", "cellwise_supplementary",
                    "modality_views", "floor_sensitivity"):
            lines.append("_" + analysis[name]["label"] + "_")
        lines.append("")
        lines.append(_md_table(table))
        lines.append("")

    lines.append("## Limitations")
    for lim in limitations(analysis):
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def build(analysis: Mapping[str, Any], out_dir: Path, run_name: str, *,
          figs_dir: Optional[Path] = None,
          wandb_mode: str = "disabled",
          mirror_dir: Optional[Path] = None,
          mirror: bool = True,
          figs_url_base: str = DEFAULT_FIGS_URL_BASE) -> Dict[str, Any]:
    """Build the full report: tables (CSV), figures (PNG), report.md,
    W&B logging (project ``petri``) and the figs mirror.

    ``wandb_mode='disabled'`` skips wandb entirely (never imported);
    ``mirror=False`` skips the figs mirror. Returns paths + the W&B URL.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = Path(figs_dir) if figs_dir else out_dir / "figs"

    tables = build_tables(analysis)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    figures = render_figures(analysis, figs_dir, run_name)

    wandb_url = None
    if wandb_mode != "disabled":
        wandb_url = _wandb_log(analysis, tables, figures, run_name,
                               wandb_mode)

    mirrored: Optional[Path] = None
    if mirror and figures:
        mirrored = _mirror_figs(
            figures,
            Path(mirror_dir) if mirror_dir
            else Path.home() / "figs" / run_name)

    md = render_markdown(analysis, tables, figures, run_name,
                         figs_url_base=figs_url_base, wandb_url=wandb_url)
    report_path = out_dir / "report.md"
    report_path.write_text(md, encoding="utf-8")

    return {"report_md": report_path, "tables_dir": tables_dir,
            "figs_dir": figs_dir, "figures": dict(figures),
            "mirror_dir": mirrored, "wandb_url": wandb_url}
