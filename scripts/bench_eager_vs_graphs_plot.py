#!/usr/bin/env python
"""Fold one or more ``bench_eager_vs_graphs.py`` results.json files into a
markdown table and one figure (tok/s by arm, small multiples per model x B,
each bar labelled with its ratio to the production floor ``pt_eager_plugin``).

  bench_eager_vs_graphs_plot.py RESULTS.json [RESULTS.json ...] --out-dir DIR

Runs on the grading host (matplotlib); the cluster job writes only results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ARM_ORDER_ALL = ["pt_eager_plugin", "pt_eager_noplugin", "pt_graphs",
                 "lora_eager_noplugin", "lora_graphs", "lora_graphs_r8"]
ARM_LABEL = {"pt_eager_plugin": "eager\n+ plugin\n(floor)",
             "pt_eager_noplugin": "eager\nno plugin",
             "pt_graphs": "graphs",
             "lora_eager_noplugin": "LoRA\neager",
             "lora_graphs": "LoRA\ngraphs\n(cap 64)",
             "lora_graphs_r8": "LoRA\ngraphs\n(cap 8)"}
FLOOR = "pt_eager_plugin"
# categorical slots 1 (blue) and 2 (orange) of the validated default palette:
# identity = passthrough vs LoRA-adapter arms; graphs-on arms carry a hatch so the
# eager/graphs distinction never rides on color alone
HUE = {"pt": "#2a78d6", "lora": "#eb6834"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def load(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        r = json.loads(p.read_text())
        key = r["facts"]["model_key"]
        r["_path"] = str(p)
        by_model[key] = r
    return by_model


def point_map(r: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    """(arm, B) -> one aggregated point: mean over the repeats of every numeric
    field, plus ``tok_s_min`` / ``tok_s_max`` / ``n_rep`` for the spread."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for p in r["points"]:
        groups.setdefault((p["arm"], p["B"]), []).append(p)
    out: Dict[tuple, Dict[str, Any]] = {}
    for k, ps in groups.items():
        agg = dict(ps[0])
        for f in ("server_tok_s", "ms_per_step", "client_tok_s", "running_mean", "sm_util_mean"):
            vals = [p[f] for p in ps if p.get(f) is not None]
            agg[f] = sum(vals) / len(vals) if vals else None
        agg["errors"] = sum(p["errors"] for p in ps)
        toks = [p["server_tok_s"] for p in ps if p.get("server_tok_s") is not None]
        agg["tok_s_min"], agg["tok_s_max"] = (min(toks), max(toks)) if toks else (None, None)
        agg["n_rep"] = len(ps)
        out[k] = agg
    return out


def arms_present(by_model: Dict[str, Dict[str, Any]]) -> List[str]:
    seen = {p["arm"] for r in by_model.values() for p in r["points"]}
    return [a for a in ARM_ORDER_ALL if a in seen]


def table(by_model: Dict[str, Dict[str, Any]]) -> str:
    order = arms_present(by_model)
    lines = ["| model | B | arm | tok/s (mean) | min–max | ms/step | x floor | running mean | SM % | errors |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in by_model.items():
        pm = point_map(r)
        bs = sorted({b for _, b in pm})
        for B in bs:
            floor = pm.get((FLOOR, B), {}).get("server_tok_s")
            for arm in order:
                p = pm.get((arm, B))
                if p is None:
                    continue
                t = p["server_tok_s"]
                ratio = (t / floor) if (t and floor) else None
                spread = (f"{round(p['tok_s_min']):,}–{round(p['tok_s_max']):,}"
                          if p.get("tok_s_min") is not None and p["n_rep"] > 1 else "")
                lines.append(
                    f"| {key} | {B} | {arm} | {t and round(t):,} | {spread} | "
                    f"{p['ms_per_step'] and round(p['ms_per_step'], 1)} | "
                    f"{ratio and f'{ratio:.2f}x'} | "
                    f"{p['running_mean'] and round(p['running_mean'], 1)} | "
                    f"{p['sm_util_mean'] and round(p['sm_util_mean'])} | {p['errors']} |")
    for key, r in by_model.items():
        lines.append("")
        lines.append(f"{key}: lora_bites={r.get('lora_bites')} failures={r.get('failures')} "
                     f"vllm={r['facts'].get('vllm_version')} gpu={r['facts'].get('gpu_name')} "
                     f"job={r['facts'].get('slurm_job_id')} node={r['facts'].get('host')}")
        for arm, rec in r.get("arms", {}).items():
            lf = rec.get("log_facts", {})
            lines.append(f"  {arm}: ready {rec.get('ready_s') and round(rec['ready_s'])}s, "
                         f"plugin_loaded={lf.get('plugin_loaded')} cuda_graphs={lf.get('cuda_graphs')} "
                         f"lora_loaded={lf.get('lora_loaded')}"
                         + (f" ERROR {rec['error']}" if rec.get("error") else ""))
    return "\n".join(lines)


def figure(by_model: Dict[str, Dict[str, Any]], out: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; table only")
        return None
    models = list(by_model)
    order = arms_present(by_model)
    bs = sorted({b for r in by_model.values() for _, b in point_map(r)})
    fig, axes = plt.subplots(len(models), len(bs), figsize=(4.8 * len(bs), 3.8 * len(models)),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, key in enumerate(models):
        pm = point_map(by_model[key])
        for j, B in enumerate(bs):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            floor = pm.get((FLOOR, B), {}).get("server_tok_s")
            xs, hs, cols, hatches, labels, err_lo, err_hi = [], [], [], [], [], [], []
            for k, arm in enumerate(order):
                p = pm.get((arm, B))
                t = p["server_tok_s"] if p else None
                xs.append(k)
                hs.append(t or 0.0)
                cols.append(HUE["lora" if arm.startswith("lora") else "pt"])
                hatches.append("//" if "graphs" in arm else "")
                if p and p.get("tok_s_min") is not None and p["n_rep"] > 1:
                    err_lo.append(t - p["tok_s_min"]); err_hi.append(p["tok_s_max"] - t)
                else:
                    err_lo.append(0.0); err_hi.append(0.0)
                if t and floor:
                    labels.append(f"{t / floor:.2f}x")
                elif p is None:
                    labels.append("not run")
                else:
                    labels.append("failed" if not t else "")
            bars = ax.bar(xs, hs, width=0.55, color=cols, edgecolor="#fcfcfb", linewidth=1.5)
            for b, h in zip(bars, hatches):
                b.set_hatch(h)
            if any(err_lo) or any(err_hi):
                # min–max of the repeated windows: the run-to-run spread, not a CI
                ax.errorbar(xs, hs, yerr=[err_lo, err_hi], fmt="none", ecolor=TEXT2,
                            elinewidth=1.2, capsize=3)
            ymax = max(h + e for h, e in zip(hs, err_hi)) if any(hs) else 1.0
            for x, h, e, lab in zip(xs, hs, err_hi, labels):
                ax.text(x, h + e + ymax * 0.02, lab, ha="center", va="bottom", fontsize=9, color=TEXT)
            ax.set_ylim(0, ymax * 1.18)
            ax.set_xticks(xs)
            ax.set_xticklabels([ARM_LABEL[a] for a in order], fontsize=8.5, color=TEXT2)
            ax.set_title(f"{key}  ·  B = {B}", fontsize=10.5, color=TEXT, loc="left")
            if j == 0:
                ax.set_ylabel("generation tok/s (engine counter)", fontsize=9, color=TEXT2)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="y", colors=TEXT2, labelsize=8.5, length=0)
            ax.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=HUE["pt"], label="passthrough (no adapter)"),
               Patch(facecolor=HUE["lora"], label="native rank-4 o_proj LoRA"),
               Patch(facecolor="#ffffff", edgecolor=TEXT2, hatch="//", label="CUDA graphs on")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Decode throughput by serving arm; label = ratio to today's eager+plugin floor",
                 fontsize=11.5, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "final_tok_s_by_arm.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    by_model = load(a.results)
    md = table(by_model)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "summary_table.md").write_text(md + "\n")
    print(md)
    p = figure(by_model, a.out_dir)
    if p:
        print(f"\nfigure: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
