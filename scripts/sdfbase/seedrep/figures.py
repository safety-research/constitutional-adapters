"""Figures, figure mirror and W&B logging for the seed-replicate analysis.

Look governed by the ``dataviz`` skill (reference palette, ``references/palette.md``):
thin marks (2px lines, >= 8px markers with a surface ring), hairline SOLID gridlines one
step off the surface, categorical hues in fixed slot order and assigned to ENTITIES
(the arm keeps its colour on every figure), the "emphasis" form where one series is
the point and the rest are context (replicate pairs in the accent, vs-original pairs
in de-emphasis grays), a legend whenever two or more series share an axis, text in
ink tokens never in the series colour, small multiples instead of a second axis.

Outputs are PNGs. ``mirror_figs`` rsyncs a figure directory to the served mirror
(``DEFAULT_MIRROR``); ``log_wandb`` logs every PNG as a ``wandb.Image`` and attaches the
directory as an artifact. Both are opt-in (``--mirror`` / ``--wandb`` on the CLIs) and
never run under tests.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------- palette (dataviz reference instance, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7",
            400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
            700: "#0d366b"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# de-emphasis grays for context series (three steps so three context lines stay tellable)
GRAYS = ["#6f6e69", "#9a9891", "#c3c2b7"]
# arms are entities: LoRA is slot 1 (blue), ReLU steering (PSR) is slot 2 (orange) on every figure
ARM_COLOR = {"lora": SERIES[0], "psr": SERIES[1]}
ARM_LABEL = {"lora": "LoRA (rank 4)", "psr": "ReLU steering (PSR)"}
# the three replicate pairs of ONE class: three steps of the accent ramp + three marker shapes
REPLICATE_STEPS = [SEQ_BLUE[350], SEQ_BLUE[450], SEQ_BLUE[600]]
MARKERS = ["o", "s", "^"]
CONTRAST_COLOR = {"const_minus_ctl": SERIES[0], "diff_minus_none": SERIES[2]}   # slots 1 and 3 (all-pairs safe)
CONTRAST_LABEL = {"const_minus_ctl": "Δ constitution − control", "diff_minus_none": "Δ difference − none"}
ROLE_LABEL = {"constitution": "constitution", "control": "control", "difference": "difference"}

DEFAULT_MIRROR = Path.home() / "figs" / "qwen-base-seed-replicates"
#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's SISTEER_WANDB_PROJECT_SEEDREP first)
WANDB_PURPOSE = "seedrep"
WANDB_RUN_NAME = "seedrep-analysis"
WANDB_JOB_TYPE = "analysis"
WANDB_ARTIFACT = "figures-seedrep"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": BASELINE,
    "legend.frameon": False,
})


def style_axes(ax, *, ygrid: bool = True, xgrid: bool = False) -> None:
    """Recessive chrome: hairline solid grid one step off the surface, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, path: Path | str, dpi: int = 160) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _pair_style(kind: str, idx: int) -> dict:
    if kind == "replicate":
        return dict(color=REPLICATE_STEPS[idx % 3], marker=MARKERS[idx % 3], lw=2.0, ms=6.5,
                    mec=SURFACE, mew=1.0, zorder=3)
    return dict(color=GRAYS[idx % 3], marker=MARKERS[idx % 3], lw=1.6, ms=5.5, mec=SURFACE, mew=1.0,
                zorder=2, alpha=0.95)


def _pair_legend(ax, pairs: Sequence[Mapping]) -> None:
    handles = []
    for i, p in enumerate([p for p in pairs if p["kind"] == "replicate"]):
        st = _pair_style("replicate", i)
        handles.append(Line2D([0], [0], color=st["color"], marker=st["marker"], lw=2, ms=6, mec=SURFACE,
                              label=f"{p['label']} (replicate pair)"))
    for i, p in enumerate([p for p in pairs if p["kind"] == "vs_original"]):
        st = _pair_style("vs_original", i)
        handles.append(Line2D([0], [0], color=st["color"], marker=st["marker"], lw=1.6, ms=5, mec=SURFACE,
                              label=f"{p['label']} (vs batch-8 seed-0 original)"))
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)


def per_layer_pairs_figure(*, class_id: str, arm: str, role: str, layers: Sequence[int],
                           components: Sequence[str], pairs: Sequence[Mapping], stat: str,
                           out: Path | str, subtitle: str = "") -> Path:
    """One small-multiple panel per component (LoRA: ΔW; PSR: direction, gate w, |Δ gate b|),
    layer on x, one line per pair; replicate pairs in the accent ramp, vs-original pairs in
    grays. ``stat`` is ``cosine`` or ``relnorm``; ``pairs`` rows carry
    ``{label, kind, series: {component: {layer: value}}}``."""
    n = len(components)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.6, 3.6), squeeze=False, sharex=True)
    ylabel = {"cosine": "cosine similarity", "relnorm": "relative-norm difference ‖a−b‖ / mean(‖a‖,‖b‖)",
              "absdiff": "|Δ|"}
    for ax, comp in zip(axes[0], components):
        style_axes(ax)
        comp_stat = "absdiff" if comp == "gate_b" else stat
        for kind in ("vs_original", "replicate"):     # context first, emphasis on top
            ks = [p for p in pairs if p["kind"] == kind]
            for i, p in enumerate(ks):
                ser = p["series"].get(comp) or {}
                xs = [l for l in layers if l in ser]
                ys = [ser[l] for l in xs]
                if xs:
                    ax.plot(xs, ys, **_pair_style(kind, i))
        if comp_stat == "cosine":
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color=BASELINE, lw=0.8)
        elif comp_stat == "relnorm":
            ax.set_ylim(bottom=0)
        ticks = list(layers) if len(layers) <= 16 else [l for i, l in enumerate(layers) if i % 4 == 0 or i == len(layers) - 1]
        ax.set_xticks(ticks)
        ax.set_xlabel("layer")
        ax.set_title({"dW": "ΔW = B Aᵀ (per targeted layer)", "direction": "direction d_l", "gate_w": "gate weight w_l",
                      "gate_b": "gate bias |b_a − b_b|"}.get(comp, comp), color=INK2)
        ax.set_ylabel(ylabel[comp_stat], color=INK2)
    fig.suptitle(f"{class_id}  ·  {ARM_LABEL.get(arm, arm)}, {ROLE_LABEL.get(role, role)}"
                 + (f"\n{subtitle}" if subtitle else ""), color=INK, x=0.02, ha="left", y=1.02)
    fig.subplots_adjust(top=0.82, wspace=0.32)
    _pair_legend(axes[0][0], pairs)
    return save(fig, out)


def stability_summary_figure(rows: Sequence[Mapping], out: Path | str, primary: Mapping[str, str]) -> Path:
    """Stacked cosine per class: parents beside their difference object, one panel per arm
    (small multiples, shared y). Replicate pairs = accent dots with the mean as a tick;
    vs-original pairs = gray dots. ``rows`` carry ``{class_id, arm, role, status,
    replicate: [3 floats] | None, vs_original: [floats] | None}``."""
    arms = [a for a in ("lora", "psr") if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(1, max(len(arms), 1), figsize=(4.6 * max(len(arms), 1) + 0.4, 3.9),
                             squeeze=False, sharey=True)
    order = {"constitution": 0, "control": 1, "difference": 2}
    for ax, arm in zip(axes[0], arms):
        style_axes(ax)
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: order.get(r["role"], 9))
        xs = list(range(len(sub)))
        for x, r in zip(xs, sub):
            rep, orig = r.get("replicate"), r.get("vs_original")
            if rep:
                jit = [-0.08, 0.0, 0.08][: len(rep)]
                ax.plot([x + j for j in jit], rep, ls="none", marker="o", ms=7.5, color=ARM_COLOR[arm],
                        mec=SURFACE, mew=1.2, zorder=3)
                m = sum(rep) / len(rep)
                ax.plot([x - 0.22, x + 0.22], [m, m], color=INK, lw=2.0, zorder=4)
                ax.annotate(f"{m:.3f}", (x - 0.27, m), color=INK2, fontsize=8, va="center", ha="right")
            if orig:
                jit = [-0.06, 0.0, 0.06][: len(orig)]
                ax.plot([x + 0.34 + j for j in jit], orig, ls="none", marker="D", ms=5.5, color=GRAYS[0],
                        mec=SURFACE, mew=1.0, zorder=2)
            if not rep:
                ax.annotate("unavailable", (x, 0.0), color=MUTED, fontsize=8, ha="center", va="center",
                            rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([ROLE_LABEL.get(r["role"], r["role"]) for r in sub])
        ax.set_xlim(-0.75, len(sub) - 0.3)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{ARM_LABEL[arm]} — stacked cosine of {primary.get(arm, '')}", color=INK2)
    axes[0][0].set_ylabel("stacked cosine similarity", color=INK2)
    handles = [Line2D([0], [0], ls="none", marker="o", ms=7, color=SERIES[0], mec=SURFACE, label="replicate pair (s1–s3)"),
               Line2D([0], [0], color=INK, lw=2, label="mean over replicate pairs")]
    if any(r.get("vs_original") for r in rows):   # the batch-8 reference column is opt-in (O23): no entry for an absent series
        handles.append(Line2D([0], [0], ls="none", marker="D", ms=5.5, color=GRAYS[0], mec=SURFACE,
                              label="replicate vs batch-8 seed-0 original"))
    axes[0][0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5 * len(arms), -0.14), ncol=len(handles))
    fig.suptitle("Object-space seed stability by class (higher = more seed-stable)", color=INK, x=0.02, ha="left")
    return save(fig, out)


def seed_spread_figure(panels: Sequence[Mapping], out: Path | str) -> Path:
    """One panel per category (rows = legs), one column per arm type; per leg the two
    contrasts stacked: seed points (dots) with a thin range bar, the paper's delta as an
    ink diamond, the cross-view replicate floor as a light gray band around zero.
    ``panels``: ``[{category, units, legs: [{leg, label, floor, arms: {lora: {contrast:
    {seeds: [..] | None, paper: float | None, status}}}}]}]``."""
    panels = [p for p in panels if p["legs"]]
    if not panels:
        fig, ax = plt.subplots(figsize=(6, 2))
        style_axes(ax, ygrid=False)
        ax.text(0.5, 0.5, "no legs available", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return save(fig, out)
    arms = ("lora", "psr")
    heights = [max(1.0, 0.62 * len(p["legs"]) + 0.6) for p in panels]
    fig, axes = plt.subplots(len(panels), len(arms), figsize=(5.0 * len(arms) + 0.8, sum(heights) + 1.0),
                             squeeze=False, gridspec_kw={"height_ratios": heights})
    offs = {"const_minus_ctl": +0.17, "diff_minus_none": -0.17}
    for pi, panel in enumerate(panels):
        for ai, arm in enumerate(arms):
            ax = axes[pi][ai]
            style_axes(ax, ygrid=False, xgrid=True)
            legs = panel["legs"]
            for yi, leg in enumerate(legs):
                y = len(legs) - 1 - yi
                fl = leg.get("floor")
                if fl is not None:
                    ax.axhspan(y - 0.42, y + 0.42, xmin=0, xmax=1, color=SURFACE, lw=0)   # keep row bands clean
                    ax.fill_betweenx([y - 0.42, y + 0.42], -fl, fl, color=GRID, alpha=0.9, lw=0, zorder=1)
                for contrast, off in offs.items():
                    c = ((leg.get("arms") or {}).get(arm) or {}).get(contrast) or {}
                    seeds = c.get("seeds")
                    col = CONTRAST_COLOR[contrast]
                    if seeds:
                        ax.plot([min(seeds), max(seeds)], [y + off] * 2, color=col, lw=1.6, zorder=2)
                        ax.plot(seeds, [y + off] * len(seeds), ls="none", marker="o", ms=7, color=col,
                                mec=SURFACE, mew=1.1, zorder=3)
                    elif c.get("status"):
                        ax.annotate("unavailable", (0.99, y + off), xycoords=("axes fraction", "data"), color=MUTED,
                                    fontsize=7, va="center", ha="right", zorder=5)
                    if c.get("paper") is not None:
                        ax.plot([c["paper"]], [y + off], ls="none", marker="D", ms=6.5, color=INK, mec=SURFACE,
                                mew=1.0, zorder=4)
            ax.set_yticks(range(len(legs)))
            ax.set_yticklabels([l["label"] for l in reversed(legs)])
            ax.set_ylim(-0.6, len(legs) - 0.4)
            ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
            if pi == 0:
                ax.set_title(ARM_LABEL[arm], color=INK2)
            if ai == 0:
                ax.set_ylabel(panel["category"], color=INK2)
            if pi == len(panels) - 1:
                ax.set_xlabel(f"delta ({panel['units']})", color=INK2)
    handles = [Line2D([0], [0], ls="none", marker="o", ms=7, color=CONTRAST_COLOR["const_minus_ctl"], mec=SURFACE,
                      label=CONTRAST_LABEL["const_minus_ctl"] + " per seed (s1–s3, range bar)"),
               Line2D([0], [0], ls="none", marker="o", ms=7, color=CONTRAST_COLOR["diff_minus_none"], mec=SURFACE,
                      label=CONTRAST_LABEL["diff_minus_none"] + " per seed (s1–s3, range bar)"),
               Line2D([0], [0], ls="none", marker="D", ms=6.5, color=INK, mec=SURFACE, label="paper's Qwen delta (batch-8 seed-0)"),
               plt.Rectangle((0, 0), 1, 1, color=GRID, label="± cross-view replicate floor (none vs none across Qwen views)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Seed spread of the Qwen deltas beside the paper's readings", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return save(fig, out)


# ---------------------------------------------------------------- mirror + W&B
def mirror_figs(src: Path | str, dest: Path | str = DEFAULT_MIRROR) -> subprocess.CompletedProcess:
    """rsync the figure directory to the served mirror (adds/updates only)."""
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["rsync", "-a", "--include=*/", "--include=*.png", "--include=*.json", "--include=*.md",
                           "--exclude=*", f"{src}/", f"{dest}/"], check=True, capture_output=True, text=True)


def log_wandb(figures: Mapping[str, Path | str], figs_dir: Path | str, *, project: str | None = None,
              name: str = WANDB_RUN_NAME, job_type: str = WANDB_JOB_TYPE, artifact: str = WANDB_ARTIFACT,
              config: Mapping | None = None, prefix: str = "seedrep") -> str:
    """Log each PNG as ``wandb.Image`` under ``<prefix>/<name>`` and attach ``figs_dir`` as
    the ``figures-seedrep`` artifact. ``WANDB_API_KEY`` comes from the environment
    (exported by the secret-manager CLI). Returns the run URL."""
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "") != "offline":
        raise RuntimeError("WANDB_API_KEY is not in the environment (the site's shell exports it); refusing to log")
    import wandb

    if project is None:
        from sisteer.wandb_entity import wandb_project
        project = wandb_project(WANDB_PURPOSE)
    run = wandb.init(project=project, name=name, job_type=job_type, config=dict(config or {}), reinit=True)
    try:
        run.log({f"{prefix}/{k}": wandb.Image(str(v)) for k, v in figures.items()})
        art = wandb.Artifact(artifact, type="figures")
        art.add_dir(str(figs_dir))
        run.log_artifact(art)
        url = run.url
    finally:
        run.finish()
    return url


__all__ = ["SERIES", "SEQ_BLUE", "SURFACE", "INK", "INK2", "MUTED", "GRID", "BASELINE", "GRAYS", "ARM_COLOR",
           "ARM_LABEL", "CONTRAST_COLOR", "CONTRAST_LABEL", "DEFAULT_MIRROR", "style_axes", "save",
           "per_layer_pairs_figure", "stability_summary_figure", "seed_spread_figure", "mirror_figs", "log_wandb"]
