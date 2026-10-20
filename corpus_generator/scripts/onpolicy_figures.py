#!/usr/bin/env python3
"""Figures for the on-policy step-5 variant (2026-09-01).

Three-way comparison — the shipped Opus corpus (final-production-20260831-v2)
vs the two on-policy samplers (gemma-4-E4B-it, Qwen3.5-4B):

1. onpolicy_score_distributions.png — per-axis (alignment, realism) score
   distributions, share of scored records. Opus bars come from the v2 filter
   report's pre-filter distributions; on-policy bars are recomputed from the
   materialized records (works before AND after floors are frozen).
2. onpolicy_kept_dropped_by_rule.png — per-model drop counts by filter rule
   plus the kept bar, from whichever filter report currently exists
   (calibration reports are labeled as such).
3. onpolicy_response_length_threeway.png — assistant-text token counts under
   the SAME reference tokenizer (staged llama3) for all three sources.

Colors: entity-fixed (dataviz rule) — Opus #2a78d6, E4B #eb6834, Qwen #1baf7a
(validated: CVD ΔE 9.2 worst pair, normal 27.6; aqua's 2.74:1 surface contrast
is relieved by direct labels + the reports as table view).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_generator.io import iter_jsonl  # noqa: E402
from corpus_generator.tokens import TokenCounter  # noqa: E402

OUT_DIR = Path.home() / "figs" / "corpus_generator-production"

SOURCES = {  # entity -> (label, fixed hue)
    "opus": ("Opus 5 (shipped v2 corpus)", "#2a78d6"),
    "e4b": ("gemma-4-E4B-it (on-policy)", "#eb6834"),
    "qwen": ("Qwen3.5-4B (on-policy)", "#1baf7a"),
}
INK = "#1a1a19"
MUTED = "#6b6a63"
SURFACE = "#fcfcfb"

V2_FINAL = ROOT / "data/advice/final-production-20260831-v2.jsonl"
V2_REPORT = ROOT / "data/advice/final-production-20260831-v2.filter-report.json"
RUNS = {
    "e4b": ROOT / "data/runs/onpolicy-e4b-20260901",
    "qwen": ROOT / "data/runs/onpolicy-qwen35-20260901",
}
FINALS = {
    "e4b": ROOT / "data/advice/final-onpolicy-e4b-20260901.jsonl",
    "qwen": ROOT / "data/advice/final-onpolicy-qwen35-20260901.jsonl",
}
SCORED = {
    "e4b": ROOT / "data/advice/scored-onpolicy-e4b-20260901.jsonl",
    "qwen": ROOT / "data/advice/scored-onpolicy-qwen35-20260901.jsonl",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)


def _report(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def onpolicy_records(key: str) -> list[dict]:
    p = RUNS[key] / "advice" / "records.jsonl"
    return list(iter_jsonl(p)) if p.exists() else []


def fig_score_distributions() -> None:
    v2 = _report(V2_REPORT)
    dists: dict[str, dict[str, dict[int, int]]] = {
        "opus": {ax: {int(k): v for k, v in d.items()}
                 for ax, d in v2["score_distributions"].items()}
    }
    for key in ("e4b", "qwen"):
        d: dict[str, dict[int, int]] = {"alignment": {}, "realism": {}}
        for r in onpolicy_records(key):
            s = r.get("scores") or {}
            for ax in d:
                if ax in s:
                    d[ax][s[ax]] = d[ax].get(s[ax], 0) + 1
        dists[key] = d

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax_i, axis_name in enumerate(("alignment", "realism")):
        ax = axes[ax_i]
        _style(ax)
        xs = np.arange(11)
        width = 0.27
        for i, key in enumerate(SOURCES):
            counts = dists[key].get(axis_name, {})
            total = sum(counts.values()) or 1
            ys = [counts.get(x, 0) / total for x in xs]
            label, color = SOURCES[key]
            ax.bar(xs + (i - 1) * width, ys, width=width * 0.92, color=color,
                   label=label if ax_i == 0 else None, zorder=3)
            # selective direct labels: only the modal bin per series
            mode_x = int(np.argmax(ys))
            if ys[mode_x] > 0:
                ax.text(mode_x + (i - 1) * width, ys[mode_x] + 0.012,
                        f"{ys[mode_x]:.0%}", ha="center", fontsize=8, color=INK)
        ax.set_title(f"{axis_name} (Opus scorer, 0–10)", fontsize=11,
                     color=INK)
        ax.set_xticks(xs)
        ax.set_xlabel("score", fontsize=9, color=MUTED)
        if ax_i == 0:
            ax.set_ylabel("share of scored records", fontsize=9, color=MUTED)
    n_scored = {k: sum(dists[k]["alignment"].values()) for k in dists}
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3,
               fontsize=9, frameon=False)
    fig.suptitle(
        "Constitution-adherence score distributions: Opus corpus vs on-policy samplers "
        f"(n scored: opus {n_scored['opus']:,} / e4b {n_scored['e4b']:,} / "
        f"qwen {n_scored['qwen']:,})",
        fontsize=12, color=INK, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "onpolicy_score_distributions.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_kept_dropped() -> None:
    # full funnel: hard-gate/sanity/dedup drops (scored report) + the
    # below-floor score cut + the two kept tiers (scored-all, floored final)
    reports = {}
    for key, final in FINALS.items():
        sr = _report(SCORED[key].with_name(SCORED[key].stem + ".filter-report.json"))
        fr = _report(final.with_name(final.stem + ".filter-report.json"))
        if sr is None:
            sr = _report(
                ROOT / "data/advice" /
                f"calibration-onpolicy-{'e4b' if key == 'e4b' else 'qwen35'}-20260901.filter-report.json"
            )
        if sr:
            reports[key] = (sr, fr)
    if not reports:
        print("kept/dropped: no filter reports yet — skipped")
        return
    hard_rules = sorted({r for sr, _ in reports.values() for r in sr["rule_counts"]})
    rows = ["kept: scored-all", "kept: final (a>=7 & r>=7)", "dropped: below floor"] + [
        f"dropped: {r}" for r in hard_rules
    ]

    def row_vals(sr, fr):
        scored_n = sr["output_count"]
        final_n = fr["output_count"] if fr else 0
        below = scored_n - final_n if fr else 0
        return [scored_n, final_n, below] + [sr["rule_counts"].get(r, 0) for r in hard_rules]

    reports_flat = {k: row_vals(sr, fr) for k, (sr, fr) in reports.items()}
    fig, ax = plt.subplots(
        figsize=(8, 1.0 + 0.42 * len(rows)), facecolor=SURFACE
    )
    _style(ax)
    ax.xaxis.grid(True, color="#e6e5df", linewidth=0.6)
    ax.yaxis.grid(False)
    ys = np.arange(len(rows))[::-1]
    h = 0.36
    allvals = [v for vals in reports_flat.values() for v in vals] or [1]
    for i, (key, vals) in enumerate(reports_flat.items()):
        label, color = SOURCES[key]
        off = (i - (len(reports_flat) - 1) / 2) * h
        ax.barh(ys + off, vals, height=h * 0.9, color=color, label=label, zorder=3)
        for y, v in zip(ys + off, vals):
            ax.text(v + max(allvals) * 0.01, y, f"{v:,}", va="center", fontsize=8,
                    color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(rows, fontsize=9, color=INK)
    ax.set_title(
        "On-policy filter funnel: hard gates -> scored-all -> floored final",
        fontsize=12, color=INK,
    )
    ax.set_xlabel("records", fontsize=9, color=MUTED)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "onpolicy_kept_dropped_by_rule.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_response_lengths() -> None:
    counter = TokenCounter.load(ROOT / "data/tokenizers")
    tok = counter.tokenizers["llama3"]

    def lengths(texts):
        return np.array([len(tok.encode(t, add_special_tokens=False).ids)
                         for t in texts])

    series = {
        "opus": lengths([r["assistant"] for r in iter_jsonl(V2_FINAL)]),
        "e4b": lengths([r["assistant"] for r in onpolicy_records("e4b")]),
        "qwen": lengths([r["assistant"] for r in onpolicy_records("qwen")]),
    }
    series = {k: v for k, v in series.items() if len(v)}
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    hi = max(np.percentile(v, 99.5) for v in series.values())
    bins = np.linspace(0, hi, 60)
    for key, v in series.items():
        label, color = SOURCES[key]
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2,
                color=color, label=f"{label}  (median {int(np.median(v)):,})",
                zorder=3)
    ax.set_title(
        "Assistant response length — same reference tokenizer (llama3) for all three",
        fontsize=12, color=INK,
    )
    ax.set_xlabel("assistant tokens", fontsize=9, color=MUTED)
    ax.set_ylabel("density", fontsize=9, color=MUTED)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "onpolicy_response_length_threeway.png", dpi=200,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_score_distributions()
    fig_kept_dropped()
    fig_response_lengths()
    print(f"figures -> {OUT_DIR}")
