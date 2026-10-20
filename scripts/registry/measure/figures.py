#!/usr/bin/env python3
"""figures.py -- the S1 / S7 figures from the measurement scripts' JSON.

    python scripts/registry/measure/figures.py --out DIR [--copy DIR] [--windows-before windows.json] [--windows-after windows.json]
                                                [--latency latency.json] [--benchmarks benchmarks.json] [--state DIR] [--root DIR]
                                                [--tick-min 10] [--json]

Every input is the JSON another script printed (``windows.py --json``, ``fold_latency.py --json``, ``benchmarks.py --json``) or a state
directory read directly (the reads namespace for the tick timeline, the shard summaries for the admission figure); a figure whose input
is absent is SKIPPED by name, never drawn from a guess. Written (PNG, 160 dpi) into ``--out`` and copied into ``--copy``:

* ``window_durations_before_after.png`` -- per-window totals, the baseline (its end definition marked, ``pre-rev3``) beside the
  post-cutover windows (``rev3``), medians and S1's lines (4 / 5 / 12 min)
* ``window_stage_breakdown.png`` -- per window the stage classes stacked (seconds), one fixed color per class
* ``changed_set_vs_duration.png`` -- changed-set size against the window total, with and without a cluster read told apart
* ``admission_vs_read_per_shard.png`` -- per shard the admission seconds against the read seconds, by adapter
* ``fold_latency_ecdf.png`` -- the completed samples' latency ECDF by class with S7's bounds (N + 10, 2N + 10 min); the censored count named
* ``tick_timeline.png`` -- every operation (tick / window) of the reads namespace as a bar from its start to its end, colored by state

Colors and marks follow the dataviz method (categorical hues in fixed slot order, never cycled; thin marks; hairline solid grid; a legend
for two or more series; text in ink, never a series color). Needs matplotlib (the repository's dev group); the Agg backend, no display.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_OUT = os.path.expanduser("~/figs/registry-slurm-only")
DEFAULT_COPY = ""   # no second copy unless --copy names one
DEFAULT_STATE = os.path.expanduser(os.environ.get("REGISTRY_LANDING_STATE") or "~/.local/state/registry-landing")
DEFAULT_ROOT = os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")
#: the reference categorical palette, slot order fixed (dataviz `references/palette.md`, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
DPI = 160
#: the stage classes FOLDED to eight, in a fixed order with a fixed slot each (identity follows the class, never its rank in one window;
#: never a ninth hue): windows.py's finer classes map onto them (`ingest_end` -> ingest, `migrate_plan` / `reconcile` -> migrate, the rest
#: of the small ones -> other)
STAGE_ORDER = ["quiet", "dims", "cluster", "adapter", "ingest", "migrate", "store_mirror", "other"]
STAGE_FOLD = {"quiet": "quiet", "dims": "dims", "dims_skip": "other", "cluster": "cluster", "cluster_skip": "other", "adapter": "adapter",
              "adapter_refused": "adapter", "ingest": "ingest", "ingest_end": "ingest", "migrate": "migrate", "migrate_plan": "migrate",
              "reconcile": "migrate", "store_mirror": "store_mirror"}
STAGE_SLOT = {c: SERIES[i] for i, c in enumerate(STAGE_ORDER)}
OP_STATES = ["receipted", "unchanged", "yielded", "dry_run", "cancelled", "failed", "watch_timeout", "other"]
OP_SLOT = {s: SERIES[i] for i, s in enumerate(OP_STATES)}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 8})
    return plt


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _save(plt, fig, out_dir: str, name: str, written: List[str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return path


def _parse(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(str(s), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------------------------
# the figures
# --------------------------------------------------------------------------------------------


def fig_durations(plt, before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]], out: str, written: List[str]) -> Optional[str]:
    groups: List[Tuple[str, List[float], str]] = []
    for label, doc, color in (("before", before, SERIES[0]), ("after", after, SERIES[1])):
        if not doc:
            continue
        rows = [w for w in doc.get("windows") or [] if w.get("completed") and w.get("total_min") is not None]
        if not rows:
            continue
        defs = sorted({str(w.get("end_definition")) for w in rows})
        groups.append((f"{label} (n={len(rows)}; end {', '.join(defs)})", [float(w["total_min"]) for w in rows], color))
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    import random
    rnd = random.Random(7)
    for i, (label, vals, color) in enumerate(groups):
        xs = [i + rnd.uniform(-0.18, 0.18) for _ in vals]
        ax.scatter(xs, vals, s=28, color=color, edgecolors=SURFACE, linewidths=1.0, label=label, zorder=3)
        med = sorted(vals)[len(vals) // 2] if len(vals) % 2 else sum(sorted(vals)[len(vals) // 2 - 1: len(vals) // 2 + 1]) / 2
        ax.hlines(med, i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.33, med, f"median {med:.1f}", va="center", ha="left", color=INK2, fontsize=8)
    for y in (4, 5, 12):
        ax.axhline(y, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(-0.5, len(groups) - 0.5 + 0.55)
    ax.set_xticks(range(len(groups)))
    # one group: its full label IS the tick label (no legend box for a single series); two: short ticks, the legend carries the counts
    ax.set_xticklabels([g[0] if len(groups) == 1 else g[0].split(" (")[0] for g in groups])
    ax.set_ylabel("window total (min)")
    ax.set_title("Landing window totals: baseline and post-cutover\nS1 bounds (the dashed lines): median <= 4, mean <= 5, max <= 12 min", fontsize=9)
    ax.xaxis.grid(False)
    if len(groups) >= 2:
        ax.legend(loc="upper right")
    return _save(plt, fig, out, "window_durations_before_after.png", written)


def fig_stage_breakdown(plt, after: Optional[Dict[str, Any]], before: Optional[Dict[str, Any]], out: str, written: List[str]) -> Optional[str]:
    doc = after or before
    if not doc:
        return None
    rows = doc.get("windows") or []
    stages = doc.get("stages") or []
    pairs = [(w, st) for w, st in zip(rows, stages) if w.get("completed")]
    if not pairs:
        return None
    pairs = pairs[-20:]
    labels = [str(w["start"])[5:16].replace("T", " ") for w, _ in pairs]
    fig, ax = plt.subplots(figsize=(7.6, max(2.6, 0.28 * len(pairs) + 1.2)))
    left = [0.0] * len(pairs)
    used = []
    for cls in STAGE_ORDER:
        vals = []
        for w, st in pairs:
            secs = sum(float(s["seconds"]) for s in st if STAGE_FOLD.get(str(s.get("class")), "other") == cls and s.get("seconds") is not None)
            if cls == "quiet" and w.get("drain_s") is not None:
                secs = float(w["drain_s"])
            vals.append(secs / 60.0)
        if not any(vals):
            continue
        ax.barh(range(len(pairs)), vals, left=left, height=0.62, color=STAGE_SLOT[cls], edgecolor=SURFACE, linewidth=1.0, label=cls)
        left = [a + b for a, b in zip(left, vals)]
        used.append(cls)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("minutes")
    ax.set_title("Window stage breakdown (the last %d completed windows)" % len(pairs))
    ax.yaxis.grid(False)
    if len(used) >= 2:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.9 / max(len(pairs), 4)), ncol=min(4, len(used)))
    return _save(plt, fig, out, "window_stage_breakdown.png", written)


def fig_changed_set(plt, after: Optional[Dict[str, Any]], before: Optional[Dict[str, Any]], out: str, written: List[str]) -> Optional[str]:
    rows = [w for doc in (before, after) if doc for w in doc.get("windows") or [] if w.get("completed") and w.get("total_min") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    import random
    rnd = random.Random(11)
    for flag, label, color in ((True, "with a cluster read", SERIES[0]), (False, "no cluster read", SERIES[1])):
        pts = [(int(w.get("changed_set_n") or 0), float(w["total_min"])) for w in rows if bool(w.get("cluster_read")) == flag]
        if pts:
            ax.scatter([p[0] + rnd.uniform(-0.15, 0.15) for p in pts], [p[1] for p in pts], s=28, color=color, edgecolors=SURFACE, linewidths=1.0,
                       label=f"{label} (n={len(pts)})", zorder=3)
    ax.set_xlabel("changed set (adapters named on the start line)")
    ax.set_ylabel("window total (min)")
    ax.set_title("Changed-set size against window total")
    ax.legend(loc="upper left")
    return _save(plt, fig, out, "changed_set_vs_duration.png", written)


def shard_points(root: Optional[str]) -> List[Dict[str, Any]]:
    d = os.path.join(root or "", "state", "cluster_reads")
    out: List[Dict[str, Any]] = []
    if not root or not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".summaries"):
            continue
        base = os.path.join(d, name)
        for adapter in sorted(os.listdir(base)):
            ad = os.path.join(base, adapter)
            if not os.path.isdir(ad):
                continue
            for fn in sorted(os.listdir(ad)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(ad, fn), "r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                except (OSError, ValueError):
                    continue
                t = (doc.get("timings_s") or {}) if isinstance(doc, dict) else {}
                if doc.get("ok") and t.get("pass") is not None:
                    out.append({"window": name[: -len(".summaries")], "adapter": adapter, "pass_s": float(t["pass"]),
                                "admission_s": float(t.get("admission_pre") or 0) + float(t.get("admission_post") or 0)})
    return out


def fig_admission(plt, points: List[Dict[str, Any]], out: str, written: List[str]) -> Optional[str]:
    if not points:
        return None
    adapters = sorted({p["adapter"] for p in points})[:8]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, a in enumerate(adapters):
        pts = [p for p in points if p["adapter"] == a]
        ax.scatter([p["pass_s"] for p in pts], [p["admission_s"] for p in pts], s=28, color=SERIES[i], edgecolors=SURFACE, linewidths=1.0,
                   label=f"{a} (n={len(pts)})", zorder=3)
    lim = max([p["pass_s"] for p in points] + [p["admission_s"] for p in points] + [1.0]) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.text(lim * 0.98, lim * 0.98, "admission = read", ha="right", va="top", color=MUTED, fontsize=7)
    ax.set_xlabel("read (the pass) per shard, s")
    ax.set_ylabel("admission (pre + post) per shard, s")
    ax.set_title("Admission against read, per shard")
    if len(adapters) >= 2:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        ax.text(0.02, 0.98, adapters[0], transform=ax.transAxes, va="top", color=INK2, fontsize=8)
    return _save(plt, fig, out, "admission_vs_read_per_shard.png", written)


def fig_latency_ecdf(plt, latency: Optional[Dict[str, Any]], out: str, written: List[str], tick_min: float = 10.0) -> Optional[str]:
    if not latency:
        return None
    samples = latency.get("samples") or []
    done = [s for s in samples if s.get("outcome") == "completed" and s.get("latency_s") is not None]
    cens = [s for s in samples if s.get("outcome") == "censored"]
    if not done and not cens:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    classes = sorted({str(s.get("class")) for s in done})[:8]
    for i, c in enumerate(classes):
        vals = sorted(float(s["latency_s"]) / 60.0 for s in done if str(s.get("class")) == c)
        ys = [(k + 1) / len(vals) for k in range(len(vals))]
        ax.step(vals, ys, where="post", color=SERIES[i], linewidth=2, label=f"{c} (n={len(vals)})")
        ax.scatter([vals[-1]], [1.0], s=24, color=SERIES[i], edgecolors=SURFACE, linewidths=1.0, zorder=4)
    tm = float((latency.get("summary") or {}).get("tick_min") or tick_min)
    for x, txt in ((tm + 10, f"p50 bound N+10 = {tm + 10:.0f}"), (2 * tm + 10, f"p90 bound 2N+10 = {2 * tm + 10:.0f}")):
        ax.axvline(x, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
        ax.text(x, 0.02, txt, rotation=90, va="bottom", ha="right", color=MUTED, fontsize=7)
    ax.set_xlabel("latency from the producer's committed change to the fold (min)")
    ax.set_ylabel("share of completed samples")
    ax.set_ylim(0, 1.05)
    by_reason = (latency.get("summary") or {}).get("censored_by_reason") or {}
    ax.set_title(f"Fold latency ECDF by class -- {len(done)} completed, {len(cens)} censored" + (f" {dict(sorted(by_reason.items()))}" if by_reason else ""), fontsize=9)
    if len(classes) >= 2:
        ax.legend(loc="lower right")
    elif classes:
        ax.text(0.98, 0.05, f"{classes[0]} (n={len(done)})", transform=ax.transAxes, ha="right", color=INK2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no completed sample", transform=ax.transAxes, ha="center", color=INK2)
    return _save(plt, fig, out, "fold_latency_ecdf.png", written)


def ops_rows(state: Optional[str]) -> List[Dict[str, Any]]:
    d = os.path.join(state or "", "reads", "ops")
    out: List[Dict[str, Any]] = []
    if not state or not os.path.isdir(d):
        return out
    rd = os.path.join(state, "reads", "receipts")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), "r", encoding="utf-8") as fh:
                op = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(op, dict) or op.get("schema") != "reads_op/1":
            continue
        hist = [h for h in (op.get("history") or []) if isinstance(h, dict) and _parse(h.get("utc"))]
        start = _parse(op.get("started_utc") or op.get("reserved_utc")) or (_parse(hist[0]["utc"]) if hist else None)
        end = _parse(op.get("ended_utc")) or (_parse(hist[-1]["utc"]) if hist else None) or _parse(op.get("updated_utc"))
        if start is None or end is None:
            continue
        has_receipt = os.path.isfile(os.path.join(rd, fn))
        out.append({"op_id": op.get("op_id") or fn[:-5], "owner": op.get("owner"), "state": str(op.get("state")), "start": start, "end": max(end, start),
                    "has_receipt": has_receipt, "history": [(str(h.get("state")), _parse(h.get("utc"))) for h in hist]})
    out.sort(key=lambda r: r["start"])
    return out


def fig_tick_timeline(plt, ops: List[Dict[str, Any]], out: str, written: List[str]) -> Optional[str]:
    if not ops:
        return None
    ops = ops[-60:]
    t0 = min(o["start"] for o in ops)
    fig, ax = plt.subplots(figsize=(7.6, max(2.4, 0.2 * len(ops) + 1.2)))
    seen = set()
    for i, o in enumerate(ops):
        st = o["state"] if o["state"] in OP_SLOT else "other"
        x0 = (o["start"] - t0).total_seconds() / 60.0
        w = max((o["end"] - o["start"]).total_seconds() / 60.0, 0.15)
        ax.barh(i, w, left=x0, height=0.6, color=OP_SLOT[st], edgecolor=SURFACE, linewidth=1.0, label=st if st not in seen else None)
        seen.add(st)
        if o["has_receipt"]:
            ax.scatter([x0 + w], [i], s=26, marker="D", color=INK, edgecolors=SURFACE, linewidths=1.0, zorder=4)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels([f'{o["owner"] or "?"} {str(o["op_id"])[-30:]}' for o in ops], fontsize=6)
    ax.set_ylim(-0.7, max(len(ops) - 0.3, 3.3))
    ax.invert_yaxis()
    ax.set_xlabel(f"minutes since {t0.strftime('%Y-%m-%d %H:%M')}Z")
    ax.set_title("Read operations (ticks and windows): start to end, by state; a diamond marks a receipt")
    ax.yaxis.grid(False)
    if len(seen) >= 1:
        ax.legend(loc="lower right", ncol=min(4, len(seen)))
    return _save(plt, fig, out, "tick_timeline.png", written)


def render_all(out: str, copy: Optional[str] = None, windows_before: Optional[str] = None, windows_after: Optional[str] = None,
               latency: Optional[str] = None, benchmarks: Optional[str] = None, state: Optional[str] = None, root: Optional[str] = None,
               tick_min: float = 10.0) -> Dict[str, Any]:
    plt = _mpl()
    written: List[str] = []
    skipped: Dict[str, str] = {}
    before, after, lat = _load(windows_before), _load(windows_after), _load(latency)
    if fig_durations(plt, before, after, out, written) is None:
        skipped["window_durations_before_after.png"] = "no completed window in either input"
    if fig_stage_breakdown(plt, after, before, out, written) is None:
        skipped["window_stage_breakdown.png"] = "no completed window with stages"
    if fig_changed_set(plt, after, before, out, written) is None:
        skipped["changed_set_vs_duration.png"] = "no completed window"
    if fig_admission(plt, shard_points(root), out, written) is None:
        skipped["admission_vs_read_per_shard.png"] = f"no ok shard summary under {os.path.join(root or '', 'state', 'cluster_reads')}"
    if fig_latency_ecdf(plt, lat, out, written, tick_min) is None:
        skipped["fold_latency_ecdf.png"] = "no latency samples (fold_latency.py --json)"
    if fig_tick_timeline(plt, ops_rows(state), out, written) is None:
        skipped["tick_timeline.png"] = f"no op records under {os.path.join(state or '', 'reads', 'ops')}"
    copied: List[str] = []
    if copy:
        os.makedirs(copy, exist_ok=True)
        for p in written:
            dst = os.path.join(copy, os.path.basename(p))
            shutil.copy2(p, dst)
            copied.append(dst)
    return {"written": written, "copied": copied, "skipped": skipped, "benchmarks": benchmarks if benchmarks and os.path.isfile(benchmarks) else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--copy", default=DEFAULT_COPY, help="a second directory the PNGs are copied into ('' for none)")
    ap.add_argument("--windows-before", help="windows.py --json of the baseline span")
    ap.add_argument("--windows-after", help="windows.py --json of the post-cutover span (--first 10)")
    ap.add_argument("--latency", help="fold_latency.py --json")
    ap.add_argument("--benchmarks", help="benchmarks.py --json (recorded beside the figures' manifest)")
    ap.add_argument("--state", default=DEFAULT_STATE, help="the reads namespace for the tick timeline")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the registry root whose state/cluster_reads summaries feed the admission figure")
    ap.add_argument("--tick-min", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("figures.py needs matplotlib (the repository's dev group)", file=sys.stderr)
        return 2
    res = render_all(a.out, a.copy or None, a.windows_before, a.windows_after, a.latency, a.benchmarks, a.state, a.root, a.tick_min)
    manifest = {"rendered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **res}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "figures.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if a.json:
        print(json.dumps(manifest, indent=1))
    else:
        for p in res["written"]:
            print(f"wrote {p}")
        for name, why in res["skipped"].items():
            print(f"skipped {name}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
