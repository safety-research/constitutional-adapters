"""capeval report builder — owns the W&B run lifecycle and figure
mirroring end-to-end.

Inputs, all under the sweep dir (base deps only — pandas/pyarrow/
matplotlib/yaml; wandb imported only when a W&B run is requested):

  rows.parquet                REQUIRED (store.consolidate output; the
                              driver's `report` runs after `consolidate`)
  fingerprint.json            per-arm domain statuses + screen (stats/
                              fingerprint.py — frozen §7.3 shape:
                              {arm: {"domains": {d: {"delta","status",...}},
                                     "screen": "SCREEN-PASS"|"SCREEN-FAIL"}})
  calibration/calibration.json   (§7.1)
  controls_acceptance.json    controls pass/fail + dose ladders (§8)

Missing OPTIONAL inputs are never silently skipped: each lands in the gate
log as a MISSING-INPUT entry and the affected figure carries the note
(authoritative-source rule — this module never recomputes statistics that
stats/ owns; it renders filed outputs).

Outputs: <sweep>/report/index.html + final_<what>_<sweep>.png figures,
mirrored to figs_dir (on the grading host it defaults to a per-sweep
figures dir, served over a private network; on the cluster pass
$FIGS_DIR), plus the W&B summary run (project `capeval`; §11.2: the
report step runs ONLINE from the grading host — the offline+sync
convention is for cluster nodes, which never run this).

Wall-clock discipline: HF columns report
P50/P95 over completed (tier × cell) end-to-end in-job times (model-ready
→ tier end); vLLM columns report the mean labeled with its n (2) and no
P50/P95; the two are separate tables and never compared. Slurm queue
waits (sacct, cluster state not harness cost) are their own table.

Figure style: the dataviz-skill reference palette (status colors for the
status-heatmap — preserved=good #0ca30c, degraded=critical #d03b3b,
indeterminate=neutral #f0efec; light surface #fcfcfb; ink #0b0b0b) with
status letters IN every cell so color never carries meaning alone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# --- palette (dataviz reference instance; see module docstring) --------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
STATUS_GOOD = "#0ca30c"       # preserved / SCREEN-PASS
STATUS_CRITICAL = "#d03b3b"   # degraded / SCREEN-FAIL
NEUTRAL = "#f0efec"           # indeterminate (diverging-midpoint gray)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4

STATUS_STYLE = {  # face color, letter, letter color
    "preserved": (STATUS_GOOD, "P", "#ffffff"),
    "degraded": (STATUS_CRITICAL, "D", "#ffffff"),
    "indeterminate": (NEUTRAL, "?", INK),
}

MIN_FREE_BYTES = 300 * 1024 * 1024  # R10: refuse below 300MB free


@dataclass
class ReportPaths:
    html: Path
    figures: Dict[str, Path] = field(default_factory=dict)
    figs_dir: Optional[Path] = None
    wandb_url: Optional[str] = None


@dataclass
class GateLog:
    """Every gate/flag/refusal/skip with its reason — nothing silent (§12)."""

    entries: List[Dict[str, str]] = field(default_factory=list)

    def add(self, kind: str, source: str, detail: str) -> None:
        self.entries.append({"kind": kind, "source": source, "detail": detail})
        print(f"[report:{kind}] {source}: {detail}")


# ------------------------------------------------------------------ loading


def _load_optional_json(path: Path, gate: GateLog, what: str) -> Optional[dict]:
    if path.is_file():
        return json.loads(path.read_text())
    gate.add("MISSING-INPUT", str(path),
             f"{what} not found — its section renders as absent, not silently dropped")
    return None


def _domains_registry() -> Dict[str, dict]:
    import yaml

    root = Path(__file__).resolve().parents[4]
    reg = yaml.safe_load((root / "configs" / "capeval" / "domains.yaml").read_text())
    return reg["domains"]


# ------------------------------------------------------------------ tables


def wall_clock_tables(df, gate: Optional[GateLog] = None) -> Tuple[Any, Any]:
    """(hf_table, vllm_table) from the F7 ts rows. Cell identity =
    (arm_name, coeff, trial_id, attempt, engine); per-tier end-to-end =
    ts_tier<k>_end − ts_model_ready (§11.2). Returns pandas DataFrames;
    empty frames when no ts rows exist.

    Timestamp COVERAGE is reconciled against the tier×cell identities
    that actually filed measurement rows: a cell whose tier rows exist
    but whose ts_model_ready / ts_tier<k>_end is missing lands in the
    gate log and in the table's ``n_missing`` column — a slow/preempted
    subset can never silently vanish from the funnel numbers. vLLM
    Tier-1 with n<2 is gate-logged (2 repeats are required)."""
    import pandas as pd

    keys = ["arm_name", "coeff", "trial_id", "attempt", "engine"]

    def _cell_key(cell_id: tuple) -> tuple:
        # Canonicalize NaN -> None: calibration cells carry coeff=NaN, and
        # two separate groupbys yield NaN objects that never compare equal,
        # so an expected[cell_id] key could never match covered's — every
        # calibration cell was double-counted as BOTH measured and missing
        # (12 false MISSING-TS alarms on the k=12 calibration sweep).
        return tuple(None if (isinstance(v, float) and v != v) else v
                     for v in cell_id)

    ts = df[df["metric"].str.startswith("ts_")]
    # every (cell, tier) with actual measurement rows expects timestamps
    meas = df[df["tier"].isin([0, 1]) & (df["domain"] != "_meta")]
    expected: Dict[tuple, set] = {}
    for cell_id, grp in meas.groupby(keys, dropna=False):
        expected[_cell_key(cell_id)] = set(str(t) for t in grp["tier"].unique())

    records = []
    missing: List[str] = []
    covered: Dict[tuple, set] = {}
    for cell_id, grp in ts.groupby(keys, dropna=False):
        cell_key = _cell_key(cell_id)
        stamps = dict(zip(grp["metric"], grp["value"]))
        ready = stamps.get("ts_model_ready")
        if ready is None:
            if cell_key in expected:
                missing.append(f"{cell_key}: ts_model_ready missing")
            continue
        for metric, value in stamps.items():
            if metric.startswith("ts_tier") and metric.endswith("_end"):
                tier = metric[len("ts_tier"):-len("_end")]
                covered.setdefault(cell_key, set()).add(tier)
                records.append(dict(zip(keys, cell_id), tier=tier,
                                    end_to_end_s=value - ready))
    n_missing_by = {}
    for cell_id, tiers in expected.items():
        for tier in sorted(tiers - covered.get(cell_id, set())):
            missing.append(f"{cell_id}: ts_tier{tier}_end missing")
            n_missing_by[(cell_id[4], tier)] = n_missing_by.get(
                (cell_id[4], tier), 0) + 1
    if gate is not None:
        for entry in missing:
            gate.add("MISSING-TS", "wall-clock coverage", entry)

    long = pd.DataFrame(records)
    if long.empty:
        return long, long

    def _summ(sub, engine):
        rows = []
        for tier, g in sub.groupby("tier"):
            v = g["end_to_end_s"]
            n_missing = n_missing_by.get((engine, str(tier)), 0)
            if engine == "hf":
                rows.append({"tier": tier, "n": len(v), "n_missing": n_missing,
                             "p50_s": float(v.quantile(0.5)),
                             "p95_s": float(v.quantile(0.95))})
            else:  # vLLM: mean labeled n, NO P50/P95 (r2-F7)
                rows.append({"tier": tier, "n": len(v), "n_missing": n_missing,
                             "mean_s": float(v.mean())})
                if tier == "1" and len(v) < 2 and gate is not None:
                    gate.add("INCOMPLETE-TIMING", "vllm wall-clock",
                             f"tier 1 has n={len(v)} repeats — D10 requires 2; "
                             "the mean is not the acceptance number")
        if gate is not None:
            for r in rows:
                if r["n_missing"]:
                    gate.add("INCOMPLETE-TIMING", f"{engine} wall-clock",
                             f"tier {r['tier']}: {r['n_missing']} cell(s) "
                             "missing timestamps — the P50/P95/mean above is "
                             "NOT acceptance-grade until they are recovered")
        return pd.DataFrame(rows)

    hf = _summ(long[long["engine"] == "hf"], "hf")
    vllm = _summ(long[long["engine"] == "vllm"], "vllm")
    return hf, vllm


def queue_wait_table(df, gate: GateLog, cluster_host: Optional[str] = None):
    """Slurm queue waits (Submit→Start) via sacct AT REPORT TIME (F7) —
    their own table, cluster state not harness cost. Loud gate-log entry
    when sacct is unreachable (e.g. building off-cluster with no ssh).
    Off-cluster, sacct runs over ssh on `cluster_host`, else the site's
    $SISTEER_CLUSTER_SSH; with neither the table is SKIPPED loudly (no
    default host: the identifier pass, RELEASE_CHECKLIST.md item 1)."""
    import pandas as pd

    job_ids = sorted({j for j in df["slurm_job_id"].dropna().unique() if str(j).strip()})
    if not job_ids:
        gate.add("SKIP", "queue-wait table", "no slurm_job_id values in rows")
        return pd.DataFrame()
    cmd = ["sacct", "-j", ",".join(map(str, job_ids)), "-X", "--noheader",
           "--parsable2", "--format=JobID,Submit,Start"]
    if not shutil.which("sacct"):
        cluster_host = cluster_host or os.environ.get("SISTEER_CLUSTER_SSH")
        if not cluster_host:
            gate.add("SKIP", "queue-wait table", "sacct is not on PATH and no cluster host is named (cluster_host / $SISTEER_CLUSTER_SSH)")
            return pd.DataFrame()
        cmd = ["ssh", "-o", "BatchMode=yes", cluster_host] + cmd
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                             check=True).stdout
    except Exception as e:  # noqa: BLE001 - loud, never silent
        gate.add("MISSING-INPUT", "sacct",
                 f"queue waits unavailable ({type(e).__name__}: {e}) — "
                 "run the report where sacct/ssh reaches the cluster")
        return pd.DataFrame()
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 3 and parts[1] not in ("Unknown", "") and parts[2] not in ("Unknown", ""):
            submit, start = pd.Timestamp(parts[1]), pd.Timestamp(parts[2])
            rows.append({"job_id": parts[0],
                         "queue_wait_s": (start - submit).total_seconds()})
    return pd.DataFrame(rows)


# must match scripts/capeval_run.py's writer. v2: screen consumes
# no KL flags; arm entries carry the kl_coarse_lb annotation.
FINGERPRINT_SCHEMA_VERSION = 2


def load_fingerprint_arms(fingerprint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap the fingerprint.json wire format, refusing a schema the
    reader was not written for (simplify-pass #6: the reader previously
    invented ci_low/ci_high field names the writer never produced)."""
    version = fingerprint.get("schema_version")
    if version != FINGERPRINT_SCHEMA_VERSION:
        raise SystemExit(
            f"fingerprint.json schema_version {version!r} != expected "
            f"{FINGERPRINT_SCHEMA_VERSION} — the writer "
            "(capeval_run.py fingerprint) and this reader must move together")
    return fingerprint["arms"]


def fingerprint_table(arms: Mapping[str, Any], registry: Mapping[str, dict]):
    """Per-arm × domain long table: Δ, harm bounds, margin, status,
    screen (§12). Field names are the WRITER's (stats.noninferiority
    DomainStatus.as_dict: harm-oriented upper_bound/lower_bound + the
    resolved margin) — never re-derived from the registry when the
    status dict carries them (simplify-pass #6)."""
    import pandas as pd

    rows = []
    for arm, entry in sorted(arms.items()):
        for dom, d in sorted(entry.get("domains", {}).items()):
            reg = registry.get(dom, {})
            rows.append({
                "arm": arm, "domain": dom,
                "delta": d.get("delta"),
                "harm": d.get("harm"),
                "harm_lower": d.get("lower_bound"),
                "harm_upper": d.get("upper_bound"),
                "margin": d.get("margin", reg.get("margin")),
                "status": d.get("status"),
                "role": (d.get("role") or reg.get("role")
                         or ("gating" if d.get("in_gating_set", reg.get("gating"))
                             else "reported")),
                "screen": entry.get("screen"),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures


def _new_fig(width, height):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return plt, fig, ax


def fingerprint_heatmap(fingerprint: Mapping[str, Any],
                        registry: Mapping[str, dict], out_png: Path) -> Path:
    """Arms × domains status heatmap (§12), status-colored per the dataviz
    STATUS palette with a letter in every cell (never color alone); the
    screen verdict is a final annotated column; acceptance-role domains
    are labeled as such (they never touch the screen, r2-F6)."""
    domains = [d for d, r in registry.items()
               if r.get("kind") in ("accuracy", "nll")]  # status-bearing only
    arms = sorted(fingerprint)
    plt, fig, ax = _new_fig(max(7.0, 1.1 * len(domains) + 3), 0.62 * len(arms) + 2.4)
    for yi, arm in enumerate(arms):
        doms = fingerprint[arm].get("domains", {})
        for xi, dom in enumerate(domains):
            d = doms.get(dom)
            if d is None:
                face, letter, lcol, text = SURFACE, "·", MUTED, "not run"
            else:
                face, letter, lcol = STATUS_STYLE.get(
                    d.get("status", "indeterminate"), STATUS_STYLE["indeterminate"])
                delta = d.get("delta")
                text = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—"
            ax.add_patch(plt.Rectangle((xi + 0.04, yi + 0.04), 0.92, 0.92,
                                       facecolor=face, edgecolor=SURFACE, lw=2))
            ax.text(xi + 0.5, yi + 0.36, letter, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=lcol)
            ax.text(xi + 0.5, yi + 0.72, text, ha="center", va="center",
                    fontsize=7, color=lcol if d is not None else MUTED)
        screen = fingerprint[arm].get("screen", "?")
        ok = screen == "SCREEN-PASS"
        ax.text(len(domains) + 0.15, yi + 0.5, ("✓ " if ok else "✗ ") + screen,
                va="center", fontsize=9, fontweight="bold",
                color=STATUS_GOOD if ok else STATUS_CRITICAL)
    labels = [d + ("\n(acceptance)" if registry[d].get("role") == "acceptance" else "")
              for d in domains]
    ax.set_xticks([i + 0.5 for i in range(len(domains))])
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8, color=INK)
    ax.set_yticks([i + 0.5 for i in range(len(arms))])
    ax.set_yticklabels(arms, fontsize=9, color=INK)
    ax.set_xlim(0, len(domains) + 2.6)
    ax.set_ylim(len(arms), 0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("capability fingerprint — P preserved · D degraded · ? indeterminate"
                 " (acceptance-role domains never gate the screen)",
                 fontsize=10, color=INK_SECONDARY, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def dose_response_figure(controls: Mapping[str, Any], out_png: Path) -> Path:
    """Control dose–response (§8/§12): Tier-0 damage vs ladder rung per
    control arm, from controls_acceptance.json ({"arms": {name:
    {"coeffs": [...], "tier0_damage": {metric: [...]}}}})."""
    arms = controls.get("arms", {})
    ladders = {a: v for a, v in arms.items()
               if len(v.get("coeffs", [])) > 1 and v.get("tier0_damage")}
    plt, fig, ax = _new_fig(7, 4.4)
    for i, (arm, v) in enumerate(sorted(ladders.items())):
        color = SERIES[i % len(SERIES)]
        for metric, values in sorted(v["tier0_damage"].items()):
            ax.plot(v["coeffs"], values, marker="o", ms=6, lw=2, color=color,
                    label=f"{arm} · {metric}")
            ax.annotate(arm, (v["coeffs"][-1], values[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("ladder rung (coeff / ε)", color=INK_SECONDARY)
    ax.set_ylabel("Tier-0 damage", color=INK_SECONDARY)
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    if ladders:
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY)
    ax.margins(x=0.18)  # room for the direct end-labels
    ax.set_title("control dose–response\n(Tier-0 damage detectable by the "
                 "4×/ε=3e-3 rung — §7.3 acceptance)",
                 fontsize=9, color=INK_SECONDARY, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


def instrument_comparison_figure(fingerprint: Mapping[str, Any], out_png: Path) -> Path:
    """S2-parity vs lm-eval instrument deltas (§9): paired dots per arm for
    (math vs math_s2parity) and (knowledge vs knowledge_s2parity) —
    disagreement is a style-sensitivity FINDING, rendered, not hidden."""
    pairs = (("math", "math_s2parity"), ("knowledge", "knowledge_s2parity"))
    arms = sorted(fingerprint)
    plt, fig, ax = _new_fig(7, 0.5 * len(arms) + 2.2)
    y = 0
    yticks, ylabels = [], []
    for arm in arms:
        doms = fingerprint[arm].get("domains", {})
        for li, (lm, parity) in enumerate(pairs):
            a, b = doms.get(lm, {}).get("delta"), doms.get(parity, {}).get("delta")
            if a is None and b is None:
                continue
            color = SERIES[li]
            if a is not None:
                ax.plot([a], [y], "o", ms=8, color=color)
            if b is not None:
                ax.plot([b], [y], "s", ms=8, mfc="none", mec=color, mew=2)
            if a is not None and b is not None:
                ax.plot([a, b], [y, y], "-", lw=1.5, color=color, alpha=0.6)
            yticks.append(y)
            ylabels.append(f"{arm} · {lm}")
            y += 1
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8, color=INK)
    ax.set_xlabel("Δ vs base (● lm-eval instrument · □ S2-parity probe, unadjusted CI family)",
                  fontsize=9, color=INK_SECONDARY)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title("instrument comparison —\nparity probes carry acceptance #2 (D6/D11)",
                 fontsize=9, color=INK_SECONDARY, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------ html


def _df_html(df, title: str) -> str:
    if df is None or getattr(df, "empty", True):
        return f"<h3>{title}</h3><p class='muted'>no data (see gate log)</p>"
    return f"<h3>{title}</h3>" + df.to_html(index=False, border=0, na_rep="—",
                                            float_format=lambda v: f"{v:.4g}")


def render_html(sweep: str, sections: Sequence[str], out: Path) -> Path:
    style = f"""
    body {{ font: 14px system-ui, -apple-system, 'Segoe UI', sans-serif;
            color: {INK}; background: {SURFACE}; margin: 2em auto; max-width: 72em; }}
    h1, h2, h3 {{ color: {INK}; }} .muted {{ color: {MUTED}; }}
    table {{ border-collapse: collapse; margin: 0.6em 0; }}
    th, td {{ padding: 3px 10px; text-align: left; font-variant-numeric: tabular-nums; }}
    th {{ color: {INK_SECONDARY}; border-bottom: 1px solid {GRID}; }}
    tr:nth-child(even) {{ background: #f6f6f3; }}
    img {{ max-width: 100%; border: 1px solid {GRID}; margin: 0.5em 0; }}
    """
    body = "\n".join(sections)
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                   f"<title>capeval — {sweep}</title><style>{style}</style></head>"
                   f"<body><h1>capeval report — {sweep}</h1>{body}</body></html>")
    return out


# ------------------------------------------------------------------ build


def _check_disk(where: Path) -> None:
    probe = where
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"report refused: only {free / 1e6:.0f}MB free at "
                         f"{where} (< 300MB floor, R10)")


def build(sweep_dir: Path, sweep_name: str, figs_dir: Optional[Path] = None,
          wandb_mode: str = "disabled", cluster_host: Optional[str] = None) -> ReportPaths:
    import pandas as pd

    sweep_dir = Path(sweep_dir)
    _check_disk(sweep_dir)
    report_dir = sweep_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    gate = GateLog()

    parquet = sweep_dir / "rows.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(
            f"{parquet} missing — run `capeval_run.py --manifest … consolidate` "
            "first (rows.parquet is the report's authoritative input, §2.3)")
    df = pd.read_parquet(parquet)

    registry = _domains_registry()
    fingerprint_doc = _load_optional_json(sweep_dir / "fingerprint.json", gate,
                                          "per-arm fingerprint/screen statuses (stats/fingerprint.py)")
    fingerprint = (load_fingerprint_arms(fingerprint_doc)
                   if fingerprint_doc is not None else None)
    calibration = _load_optional_json(sweep_dir / "calibration" / "calibration.json",
                                      gate, "calibration null (σ_d, κ, canary thresholds)")
    controls = _load_optional_json(sweep_dir / "controls_acceptance.json", gate,
                                   "controls acceptance report (stats/controls.py)")

    # Reconcile the fingerprint against what the sweep actually ran
    # (review infra #8): every arm with filed measurement rows must have
    # a fingerprint entry, and every entry must cover every registered
    # gating domain — an absent arm/domain is a gate-log failure, never a
    # silently smaller table presented as a complete screen.
    if fingerprint is not None:
        expected_arms = sorted(
            set(df[(df["kind"] != "none") & (df["domain"] != "_meta")
                   & (df["attempt"] == 0)]["arm_name"].unique()))
        for arm in expected_arms:
            if arm not in fingerprint:
                gate.add("MISSING-FINGERPRINT", "fingerprint.json",
                         f"arm {arm!r} has filed rows but no fingerprint "
                         "entry — the screen below is INCOMPLETE")
        gating_domains = [n for n, d in registry.items()
                         if d.get("gating") and d.get("kind") in ("accuracy", "nll")]
        for arm, entry in sorted(fingerprint.items()):
            missing_doms = [d for d in gating_domains
                            if d not in entry.get("domains", {})]
            for dom in missing_doms:
                gate.add("MISSING-FINGERPRINT", f"fingerprint.json[{arm}]",
                         f"gating domain {dom!r} has no status — screen "
                         "coverage incomplete for this arm")
            # per-arm gate log from build_fingerprint, surfaced verbatim
            for note in entry.get("gate_log", []):
                gate.add("FINGERPRINT-GATE", arm, str(note))

    # A sweep of the removed in-process vLLM lane whose preflight failed
    # recorded this marker; presence is gate-log-loud, absence is the normal
    # case and logs nothing.
    phase_l_marker = sweep_dir / "phase_l_unavailable.json"
    if phase_l_marker.is_file():
        info = json.loads(phase_l_marker.read_text())
        gate.add("PHASE-L-UNAVAILABLE", "vllm lane",
                 f"{info.get('reason', 'unrecorded reason')} — "
                 f"{info.get('n_cells', '?')} cell(s) recorded as "
                 "unavailable, exit 0 (D13 graceful skip)")

    figures: Dict[str, Path] = {}
    if fingerprint:
        figures["fingerprint_heatmap"] = fingerprint_heatmap(
            fingerprint, registry,
            report_dir / f"final_fingerprint_heatmap_{sweep_name}.png")
        figures["instrument_comparison"] = instrument_comparison_figure(
            fingerprint, report_dir / f"final_instrument_comparison_{sweep_name}.png")
    if controls:
        figures["dose_response"] = dose_response_figure(
            controls, report_dir / f"final_dose_response_{sweep_name}.png")

    fp_table = fingerprint_table(fingerprint, registry) if fingerprint else pd.DataFrame()
    hf_wall, vllm_wall = wall_clock_tables(df, gate)
    queue = queue_wait_table(df, gate, cluster_host)

    # flag rows surface verbatim in the gate log — including their
    # extra_json (where the refusal/gate REASON lives) and the cell
    # identity, so the log is reason-complete (review infra #16)
    flags = df[df["metric"].str.startswith(("flag_", "gate_"))]
    for _, r in flags.iterrows():
        ident = (f"coeff={r['coeff']} trial={r['trial_id']} "
                 f"attempt={r['attempt']} engine={r['engine']}")
        extra_val = r["extra_json"]
        extra = (f" extra={extra_val}"
                 if isinstance(extra_val, str) and extra_val else "")
        gate.add("FLAG-ROW", f"{r['arm_name']}/{r['task']}",
                 f"{r['metric']}={r['value']} [{ident}]{extra}")
    gate_df = pd.DataFrame(gate.entries)

    sections = []
    if fingerprint:
        def _kl_note(e):
            # the KL annotation, rendered beside the screen for triage —
            # explicitly not part of the screen verdict
            ann = e.get("kl_coarse_lb")
            if not isinstance(ann, Mapping) or ann.get("mean") is None:
                return ""
            return (f" <span class='muted'>(kl_coarse_lb mean "
                    f"{ann['mean']:.3g} nats/pos — annotation, not a "
                    f"screen input)</span>")
        sections.append("<h2>Screen</h2><ul>" + "".join(
            f"<li><b>{a}</b>: {e.get('screen')}{_kl_note(e)}</li>"
            for a, e in sorted(fingerprint.items())) + "</ul>")
    for key, label in (("fingerprint_heatmap", "Fingerprint heatmap"),
                       ("dose_response", "Control dose–response"),
                       ("instrument_comparison", "Instrument comparison")):
        if key in figures:
            sections.append(f"<h2>{label}</h2><img src='{figures[key].name}'>")
    sections.append(_df_html(fp_table, "Fingerprint table (Δ, CI, margin, status)"))
    sections.append(_df_html(hf_wall, "Wall-clock — HF engine (P50/P95, end-to-end "
                                      "in-job: model-ready → tier end)"))
    sections.append(_df_html(vllm_wall, "Wall-clock — vLLM engine (mean, n stated; "
                                        "no P50/P95 from 2 repeats; never compared to HF)"))
    sections.append(_df_html(queue, "Slurm queue waits (sacct; cluster state, "
                                    "not harness cost)"))
    if calibration:
        sections.append("<h3>Calibration</h3><pre>" +
                        json.dumps({k: calibration[k] for k in sorted(calibration)
                                    if k != "per_item"}, indent=1, default=str)[:4000] +
                        "</pre>")
    sections.append(_df_html(gate_df, "Gate log (every gate/flag/refusal/skip — "
                                      "nothing filtered silently)"))
    html = render_html(sweep_name, sections, report_dir / "index.html")

    # -- figure mirroring (standing convention) ------------------------------
    if figs_dir is None:
        env_figs = os.environ.get("FIGS_DIR")
        figs_dir = Path(env_figs) if env_figs else Path.home() / "figs" / f"capeval-{sweep_name}"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for p in list(figures.values()) + [html]:
        shutil.copy2(p, figs_dir / p.name)

    # -- W&B summary run (§11.2/§12) -----------------------------------------
    wandb_url = None
    if wandb_mode != "disabled":
        import wandb

        run = wandb.init(project="capeval", name=f"report-{sweep_name}",
                         mode=wandb_mode, config={"sweep": sweep_name})
        for key, p in figures.items():
            run.log({f"report/{key}": wandb.Image(str(p))})
        for name, table in (("fingerprint_table", fp_table),
                            ("wall_clock_hf", hf_wall), ("wall_clock_vllm", vllm_wall),
                            ("queue_waits", queue), ("gate_log", gate_df)):
            if table is not None and not table.empty:
                run.log({f"report/{name}": wandb.Table(dataframe=table)})
        art = wandb.Artifact(f"figures-{run.name}", type="figures")
        art.add_dir(str(report_dir))
        run.log_artifact(art)
        wandb_url = getattr(run, "url", None)
        run.finish()

    return ReportPaths(html=html, figures=figures, figs_dir=figs_dir,
                       wandb_url=wandb_url)
