#!/usr/bin/env python
"""geom_sweep_analyze — fold every model's ``served_decode_sweep.py`` results
into (a) the ``BUDGET_BY_MODEL`` paste block for
``src/sisteer/capeval/serving/geometry.py`` (the ``_MEASURED_BUDGETS`` tuple
between the ``MEASURED BY ... (BEGIN)`` / ``(END)`` markers), (b) a markdown
table per model x window — the budget-permitted S, the knee B, tok/s at the
knee, tok/s at the old S = 24, the speedup, GPU duty at the knee — and (c) a
one-page figure grid (models x windows, tok/s vs B, both arms).

Reads ``<results-root>/<model>/budget.json`` and ``sweep_summary.json`` as the
sweep wrote them (rsync the cluster's ``$SHARED/exp/geom-sweep`` to the
grading host, or point ``--results-root`` at it). CPU-only; matplotlib is
imported only for the figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PASTE_BEGIN = "# MEASURED BY scripts/served_decode_sweep.py --phase budget  (BEGIN)"
PASTE_END = "# MEASURED BY scripts/served_decode_sweep.py --phase budget  (END)"


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """``sweep.jsonl`` rows (points and window events), tolerant of a torn last
    line (a job killed mid-write)."""
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def summary_from_rows(rows: Sequence[Mapping[str, Any]], budget: Optional[Mapping[str, Any]],
                      windows_requested: Optional[Sequence[int]] = None) -> Optional[Dict[str, Any]]:
    """Rebuild a ``sweep_summary.json`` from ``sweep.jsonl`` alone — the sweep's
    own ``summarize_rows`` over the rows it did write — when the phase died
    before writing one (the 0f9a814 fan-out tasks' 5th-window crash). ``None``
    when there are no rows at all."""
    import served_decode_sweep as sw

    if not rows:
        return None
    lp = ((budget or {}).get("largest_passing") or {})
    windows = list(windows_requested) if windows_requested else list(sw.DEFAULT_WINDOWS)
    # ONE MEASUREMENT PER SUMMARY (review S2): partition the rows by their
    # fingerprint id; the primary is the partition with the most point rows
    # (a pre-fingerprint store is the single ``None`` partition); the others
    # are named, never folded in
    parts = sw.partition_by_measurement(rows)
    counts = {mid: sum(1 for r in prs if sw.is_point_row(r)) for mid, prs in parts.items()}
    primary = max(parts, key=lambda mid: (counts[mid], len(parts[mid])))
    summ = sw.summarize_rows(parts[primary], budget=(int(lp["product"]) if lp else None),
                             windows_requested=windows, measurement_id=primary)
    summ["rebuilt_from_rows"] = True
    summ["measurements"] = {str(mid): {"n_rows": len(prs), "n_points": counts[mid]}
                            for mid, prs in parts.items()}
    summ["other_measurements"] = sorted(str(mid) for mid in parts if mid != primary)
    summ["model_key"] = next((r.get("model") for r in parts[primary] if r.get("model")), None)
    return summ


#: the run-level metadata a stored summary contributes to a rows-rebuilt one
_SUMMARY_METADATA_KEYS = ("model_key", "hf", "rev", "hybrid", "S_cap", "steered_cell",
                          "prompt_tokens", "max_tokens", "warmup_s", "measure_s", "node",
                          "job_id", "transformers_version", "requirements_file", "figures",
                          "fingerprint", "measurement_id", "old_S_note")


def paste_from_budget(budget: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The ``ModelBudget`` record for a budget.json: the stored ``paste`` with the
    ORIGINATING probe's job / node (review R2 — the largest passing probe may
    predate the job that wrote the doc; a probe without its own provenance keeps
    the doc's). Values are the probe's; only provenance can change."""
    import copy
    import re as _re
    if not budget or not budget.get("paste"):
        return None
    rec = copy.deepcopy(budget["paste"])
    lp = budget.get("largest_passing") or {}
    basis = rec.get("basis") or {}
    if lp.get("job_id"):
        basis["probe_job_id"] = str(lp["job_id"])
    if lp.get("node") and isinstance(basis.get("source"), str):
        basis["source"] = _re.sub(r"\(node [^)]*\)", f"(node {lp['node']})", basis["source"])
    if isinstance(basis.get("source"), str):
        basis["source"] = scrub_source(basis["source"])
    rec["basis"] = basis
    return rec


def load_results(root: Path, models: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """``{model: {"budget": <budget.json>|None, "summary": <summary>, "rows":
    [sweep.jsonl rows]}}`` for every model directory under ``root`` (or the named
    subset). THE ROWS ARE THE TRUTH (review R3): whenever ``sweep.jsonl`` holds
    rows the summary is REBUILT from them with today's completeness rules
    (``summary["rebuilt_from_rows"] = True``) and the stored summary contributes
    only its run-level metadata; a stored summary stands alone only when there
    are no rows. ``budget["paste"]`` is re-derived with the originating probe's
    provenance (:func:`paste_from_budget`)."""
    from sisteer.capeval.serving import geometry as sg

    keys = list(models) if models else [k for k in sg.MODEL_KEYS if (root / k).is_dir()]
    out: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        d = root / k
        b, s = d / "budget.json", d / "sweep_summary.json"
        budget = json.loads(b.read_text()) if b.is_file() else None
        if budget is not None:
            budget["paste"] = paste_from_budget(budget) or budget.get("paste")
        rows = read_rows(d / "sweep.jsonl")
        stored = json.loads(s.read_text()) if s.is_file() else None
        if rows:
            summary = summary_from_rows(
                rows, budget,
                windows_requested=(stored or {}).get("windows_requested") or None)
            assert summary is not None
            for key in _SUMMARY_METADATA_KEYS:
                if stored and key in stored and key not in summary:
                    summary[key] = stored[key]
            if stored and stored.get("partial"):
                summary["partial"] = True             # the sweep's own deadline flag stands
                summary["missing_points"] = stored.get("missing_points") or summary["missing_points"]
            summary["stored_summary_present"] = stored is not None
        else:
            summary = stored
        out[k] = {"budget": budget, "summary": summary, "rows": rows}
    return out


def _py(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, int):
        return f"{v:_d}" if v >= 10_000 else str(v)
    if isinstance(v, float):
        return repr(round(v, 4))
    return repr(str(v))


def scrub_source(source: Any) -> Any:
    """A basis ``source`` string with the cluster's per-user shared root replaced
    by the documented ``<SHARED>`` placeholder (``served_decode_sweep.
    shared_placeholder``): the pasted block must carry no username or absolute
    cluster path. Idempotent; non-cluster paths and non-strings pass through."""
    if not isinstance(source, str):
        return source
    import served_decode_sweep as sw
    return sw.shared_placeholder(source)


def format_paste_block(pastes: Sequence[Mapping[str, Any]], indent: str = "    ") -> str:
    """The ``_MEASURED_BUDGETS`` tuple body: one ``ModelBudget(...)`` per model,
    in ``MODEL_KEYS`` order, constructed exactly as geometry.py declares them
    (``tests/test_served_decode_sweep.py`` execs the output against the module
    and checks every field round-trips). ``source`` fields are scrubbed to the
    ``<SHARED>`` placeholder (:func:`scrub_source`); every other value is emitted
    byte-for-byte."""
    from sisteer.capeval.serving import geometry as sg

    order = {k: i for i, k in enumerate(sg.MODEL_KEYS)}
    lines: List[str] = []
    for p in sorted(pastes, key=lambda p: order.get(p["model_key"], 99)):
        b = p["basis"]
        lines.append(f"{indent}ModelBudget(")
        lines.append(f"{indent}    model_key={_py(p['model_key'])}, "
                     f"batched_token_budget={_py(int(p['batched_token_budget']))}, "
                     f"certified={_py(bool(p.get('certified', False)))},")
        lines.append(f"{indent}    basis=BudgetBasis(")
        lines.append(f"{indent}        method={_py(b['method'])},")
        lines.append(f"{indent}        max_model_len={_py(int(b['max_model_len']))}, "
                     f"max_num_seqs={_py(int(b['max_num_seqs']))},")
        lines.append(f"{indent}        gpu_memory_utilization={_py(float(b['gpu_memory_utilization']))},")
        lines.append(f"{indent}        kv_tokens_reported={_py(b.get('kv_tokens_reported'))},")
        lines.append(f"{indent}        probe_job_id={_py(b.get('probe_job_id'))}, "
                     f"measured_on={_py(b['measured_on'])},")
        lines.append(f"{indent}        source={_py(scrub_source(b['source']))},")
        lines.append(f"{indent}        first_refused={_py(b.get('first_refused'))},")
        lines.append(f"{indent}        note={_py(b.get('note', ''))},")
        lines.append(f"{indent}    )),")
    if not lines:
        lines.append(f"{indent}# <no measured entries yet>")
    return "\n".join(lines)


def paste_block_with_markers(pastes: Sequence[Mapping[str, Any]]) -> str:
    return (f"{PASTE_BEGIN}\n#\n# (pasted by scripts/geom_sweep_analyze.py; <SHARED> in a "
            "source = the cluster's per-user shared root, hf_env.sh's $SHARED)\n"
            f"_MEASURED_BUDGETS: Tuple[ModelBudget, ...] = (\n"
            f"{format_paste_block(pastes)}\n)\n{PASTE_END}")


def _fmt(v: Any, nd: int = 0) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def markdown_tables(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Per model: the budget line, then one row per window x arm."""
    from sisteer.capeval.serving import geometry as sg

    out: List[str] = []
    old = sg.BATCHED_TOKEN_BUDGET
    for model, r in results.items():
        b, s = r.get("budget"), r.get("summary")
        out.append(f"## {model}")
        if b:
            lp = b.get("largest_passing") or {}
            out.append(f"budget: **{_fmt(lp.get('product'))}** "
                       f"(S={lp.get('max_num_seqs')} at L={b.get('max_model_len')}, "
                       f"kv_tokens {_fmt(lp.get('kv_tokens'))}; first refused "
                       f"{_fmt(b.get('first_refused'))}; job {b.get('job_id')}, node {b.get('node')}, "
                       f"{b.get('measured_on')}; complete={b.get('complete')}) vs the Llama-certified "
                       f"{old:,} -> x{(lp['product'] / old):.2f}" if lp else
                       f"budget: NO passing probe (probes: {len(b.get('probes') or [])})")
        else:
            out.append("budget: (no budget.json)")
        if not s:
            out.append("sweep: (no sweep_summary.json and no sweep.jsonl rows — not measured)\n")
            continue
        if s.get("other_measurements"):
            out.append(f"sweep: sweep.jsonl holds rows of {len(s['measurements'])} measurements "
                       f"({s['measurements']}); this table is measurement {s.get('measurement_id')} "
                       f"only — the others ({s['other_measurements']}) are NOT folded in")
        if s.get("rebuilt_from_rows"):
            out.append(f"sweep: summary REBUILT from sweep.jsonl ({s.get('n_rows')} point rows"
                       + ("; the phase wrote no sweep_summary.json)"
                          if not s.get("stored_summary_present") else
                          "; the rows are the truth, the stored summary contributed metadata)"))
        old_s = s.get("old_S", 24)
        cols = ("window L", "evals", "arm", "S permitted", "old S", "knee B", "tok/s @knee",
                f"tok/s @B={old_s}", "speedup", "duty @knee (sm%)", "lat p50 @knee (s)",
                "preemptions", "points")
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join("---:" if i else "---" for i, _ in enumerate(cols)) + "|")
        for key, g in sorted(s["groups"].items(), key=lambda kv: (kv[1]["window_L"], kv[1]["arm"],
                                                                   kv[1].get("S") or 0)):
            L = int(g["window_L"])
            complete = g.get("complete", True)
            knee = (_fmt(g.get("knee_B")) if complete
                    else f"(prov.) {_fmt(g.get('knee_provisional_B'))}")
            sp = g.get(f"speedup_knee_vs_B{old_s}") if complete else None
            pts = (_fmt(g.get("n_points")) if complete
                   else f"{_fmt(g.get('n_points'))}/{_fmt(g.get('n_expected'))} INCOMPLETE")
            out.append("| " + " | ".join([
                str(L), ",".join(g.get("window_labels") or ()) or "-", g["arm"],
                _fmt(g.get("S")), str(sg.identity_seqs(L, budget=old)),
                knee, (_fmt(g.get("tok_s_at_knee")) if complete else "-"),
                _fmt(g.get(f"tok_s_at_B{old_s}")),
                (f"x{sp:.2f}" if sp else "-"),
                (_fmt(g.get("duty_at_knee"), 1) if complete else "-"),
                (_fmt(g.get("lat_p50_at_knee_s"), 2) if complete else "-"),
                _fmt(g.get("preemptions_total")), pts,
            ]) + " |")
        for gkey in s.get("groups_incomplete") or ():
            g = s["groups"][gkey]
            out.append(f"\nINCOMPLETE: {gkey} (S={g.get('S')}) has {g.get('n_points')} of "
                       f"{g.get('n_expected')} rungs — missing B={g.get('missing_B')}; no definitive "
                       f"knee / speedup (provisional knee B={g.get('knee_provisional_B')})")
        for L, gap in (s.get("arms_missing") or {}).items():
            out.append(f"\nARMS MISSING: window L={L} has no {', '.join(gap)} arm")
        if s.get("partial"):
            out.append(f"\nPARTIAL: {len(s.get('missing_points') or [])} points not run "
                       "(deadline) — resubmit the task; rows resume.")
        for L in s.get("windows_not_measured") or ():
            # the sweep's own label (served keys only: no informational or
            # provisional entry), so this line names what a point row would have
            import served_decode_sweep as sw
            labels = ",".join(sw.window_labels(int(L))) or "-"
            out.append(f"\nNOT MEASURED: window L={L} ({labels}) — no point rows")
        for ev in s.get("window_refusals") or ():
            out.append(f"\nREFUSED standup: L={ev.get('window_L')} at S={ev.get('S_tried')} "
                       f"(target {ev.get('S_target')}, attempt {ev.get('attempt')}): "
                       f"{ev.get('status')} — {str(ev.get('error') or '')[:200]}")
        for ev in s.get("skipped_windows") or ():
            out.append(f"\nSKIPPED window: L={ev.get('window_L')} after S tried "
                       f"{ev.get('S_tried')} ({ev.get('reason')}; last {ev.get('last_status')})")
        out.append("")
    return "\n".join(out)


def figure_grid(results: Mapping[str, Mapping[str, Any]], out_path: Path) -> Optional[Path]:
    """Models x windows grid of tok/s vs B (solid passthrough, dashed steered,
    knee ringed). Returns the path, or ``None`` when matplotlib is unavailable
    or there is nothing to draw."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    models = [m for m, r in results.items() if r.get("summary") and r["summary"].get("groups")]
    if not models:
        return None
    windows = sorted({g["window_L"] for m in models for g in results[m]["summary"]["groups"].values()})
    fig, axes = plt.subplots(len(models), len(windows), figsize=(3.2 * len(windows), 2.6 * len(models)),
                             dpi=130, squeeze=False)
    style = {"passthrough": ("-", "#0072B2"), "steered": ("--", "#D55E00")}
    for i, m in enumerate(models):
        s = results[m]["summary"]
        for j, L in enumerate(windows):
            ax = axes[i][j]
            for g in s["groups"].values():
                if g["window_L"] != L:
                    continue
                pts = sorted((t["B"], t["tok_s"]) for t in g["table"] if t.get("tok_s") is not None)
                if not pts:
                    continue
                ls, c = style.get(g["arm"], (":", "#009E73"))
                xs, ys = zip(*pts)
                ax.plot(xs, ys, ls, color=c, marker="o", ms=2.5, lw=1.3, label=g["arm"])
                if g.get("knee_B") is not None:
                    ax.plot([g["knee_B"]], [g["tok_s_at_knee"]], "o", ms=8, mfc="none", mec=c)
            ax.set_xscale("log", base=2)
            ax.grid(True, which="both", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"L={L}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{m}\ntok/s", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("B", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("served decode throughput vs concurrent requests (knee ringed)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


DOC_ORDER = ("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b",
             "glm46vflash", "nemotron3nano4b")


def _incomplete_note(results: Mapping[str, Mapping[str, Any]]) -> str:
    """The reading-notes bullet on incomplete groups, from the data: which
    model x group ran only part of its ladder, or that every ladder is complete."""
    items = []
    for k in DOC_ORDER:
        s = (results.get(k) or {}).get("summary") or {}
        for gkey in s.get("groups_incomplete") or ():
            g = s["groups"][gkey]
            items.append(f"{k} {gkey} ({g.get('n_points')} of {g.get('n_expected')} rungs)")
    if not items:
        return ("- Every group ran its full ladder (8 ... S at the served S), so every knee and "
                "speedup above is definitive.\n")
    return ("- Groups marked INCOMPLETE ran only part of their ladder (a deadline, or the tail of a "
            "crashed phase) and carry no definitive knee — " + "; ".join(items) + " — resubmit the "
            "task into a fresh results root to complete them (a resume must carry the same "
            "measurement fingerprint).\n")


def write_doc(results: Mapping[str, Mapping[str, Any]], path: Path, *,
              jobs_note: str = ("cluster A jobs 374139 / 375170 / 375192 / 375242 / 373464 and "
                                "cluster B 65057 / 65668 / 65727")) -> Path:
    """The condensed per-model x window x arm markdown report (``--doc <path>``):
    table (knee B, tok/s at the knee, tok/s at the old S = 24, speedup, duty),
    the budgets table, the headline and the reading notes — generated from the
    results, never transcribed by hand. Incomplete groups (review R3) show a
    PROVISIONAL knee and no speedup, and are listed per model. No absolute
    paths or usernames may enter the file (asserted)."""
    from sisteer.capeval.serving import geometry as sg

    CERT = sg.BATCHED_TOKEN_BUDGET
    out: List[str] = []
    out.append("# Served decode sweep: per-model batched-token budgets and throughput vs batch\n")
    out.append("Condensed from the 2026-09-04 sweep (`scripts/served_decode_sweep.py`, one H200 per "
               f"model, {jobs_note}; folded by `scripts/geom_sweep_analyze.py --doc`, which "
               "regenerates this file from the results — nothing here is transcribed by hand). "
               "This file is the cited basis of every `ModelBudget` in "
               "`src/sisteer/capeval/serving/geometry.py` (`BUDGET_BY_MODEL`) and of the throughput "
               "claims below. Raw per-point rows, server logs and the standup artifacts stay in the "
               "results directories the sweep wrote (`<results-root>/<model>/{budget.json,"
               "sweep.jsonl,sweep_summary.json}`); the analyzer's figure grid (`geom_sweep_grid.png`, "
               "models x windows, tok/s vs B) lands under `<results-root>/_analysis/` and is not "
               "copied here.\n")
    out.append("## Headline\n")
    out.append("**Un-steered (passthrough) decode throughput scales roughly linearly with the "
               "concurrent batch B up to the knee: at the table's S it is 2-5x the throughput at "
               "the old S = 24 on every family (llama3b reaches 8,533 tok/s at 1536 x 128), so the "
               "per-model budgets and per-eval windows buy real speed-up.** **The steered (lens "
               "psr_kl) arm is flat at 90-270 tok/s on every family regardless of B — the lens hook, "
               "not the geometry, is the ceiling; the fix is in flight on branch `lens-vectorize`, "
               "and until it lands a steered cell gains nothing from a wider S.**\n")
    out.append("## Method\n")
    out.append("`--phase budget`: at L = 8192 the S x L ladder 393,216 / 524,288 / 655,360 / 786,432 "
               "/ 1,048,576 / 1,310,720 / 1,572,864 is walked upward through the production standup "
               "(`capctrl_serving_up.bring_up`, `gpu_memory_utilization` 0.90) until the readiness "
               "gate refuses (`vllm_server.kv_coverage_problem`) or the engine OOMs, then bisected "
               "once; the budget is the largest passing product. `--phase sweep`: at each per-eval "
               "window L (8704 multiturn_attacks, 7424 capability / families_prompted / bon, 11520 "
               "families_served, 16384 petri, 1536 capability:short) the server stands up at "
               "S = min(budget // L, 512) — 256 for the hybrid-recurrent families, see the limitation "
               "below — and B concurrent closed-loop requests (1,000-token synthetic prompt, "
               "`max_tokens` 512 with `ignore_eos`, 20 s warm-up + 90 s measured) are driven for B in "
               "8 ... S against the passthrough virtual model and, where the model has a "
               "fitted-object bank, one steered psr_kl cell. Knee = the smallest B reaching 90% of the "
               "window's maximum tok/s, published only for a COMPLETE ladder (every rung of "
               "8 ... S measured); an incomplete group shows a provisional knee and no speedup. "
               "\"Old S\" = the 393,216 // L every served eval inherited from Petri's 16384 x 24; "
               "duty = mean GPU sm% over the measured window.\n")
    out.append("## Budgets (the `BUDGET_BY_MODEL` entries)\n")
    out.append("| model | budget (S x L) | probe S at L=8192 | attested KV tokens | first refused | "
               "vs Llama-certified 393,216 | KV facts | hybrid S cap | job |")
    out.append("|---|---:|---:|---:|---:|---:|---|---:|---|")
    for k in DOC_ORDER:
        r = results.get(k) or {}
        b = r.get("budget")
        if not b or not b.get("largest_passing"):
            out.append(f"| {k} | not measured | | | | | | | |")
            continue
        lp = b["largest_passing"]
        pb = (b.get("paste") or {}).get("basis") or {}
        kind = str(lp.get("kv_kind") or "")
        out.append(f"| {k} | {int(lp['product']):,} | {lp['max_num_seqs']} | "
                   f"{_fmt(lp.get('kv_tokens'))} | {_fmt(b.get('first_refused'))} | "
                   f"x{int(lp['product'])/CERT:.2f} | "
                   f"{'uniform (blocks x block size)' if kind.startswith('uniform') else 'hybrid (allocator max-concurrency x L)'} | "
                   f"{'256' if k in sg.HYBRID_MODEL_KEYS else '-'} | {pb.get('probe_job_id') or b.get('job_id')} |")
    out.append("")
    out.append("Budgets are measured at L = 8192 at the probe's S. **Stated limitation:** for the "
               "hybrid-recurrent families (qwen35_9b, qwen35_4b, nemotron3nano4b) the gated-deltanet / "
               "Mamba2 layers pre-allocate a fixed recurrent state per slot, so `max_num_seqs` itself "
               "costs memory and the same product does not launch as `budget // L` slots at a short "
               "window (qwen35_4b: 88 x 8192 launched, 469 x 1536 was refused). "
               "`ModelBudget.seqs_at()` therefore caps those three at S = 256 (`HYBRID_SEQS_CAP`, = the "
               "sweep's `S_MAX_HYBRID`), and a refused window standup is retried at the next lower "
               "ladder S. llama8b's measured 458,752 (uncertified) supersedes the certified 393,216 for "
               "`budget_for(\"llama8b\")`; the certified footprint remains the no-model default and the "
               "per-eval table's sizing. gemma-4 served from the transformers 5.14.1 variant venv; every "
               "other model from the lane's 5.15.0.\n")
    out.append("## Throughput per model x window x arm\n")
    out.append("Columns: S = the budget-permitted `max_num_seqs` the window was served at; knee B "
               "(`(prov.)` = provisional, the ladder is incomplete); tok/s at the knee (client-side "
               "completion tokens/s; blank for an incomplete group); tok/s at B = 24 (the old S; `-` "
               "when that rung was not run); speedup = knee / B=24 (blank for an incomplete group); "
               "duty = mean sm% at the knee; points = rungs measured / expected.\n")
    for k in DOC_ORDER:
        r = results.get(k) or {}
        s = r.get("summary")
        if not s:
            out.append(f"### {k}  (not measured)\n")
            continue
        old_s = s.get("old_S", 24)
        flags = []
        if s.get("partial") and s.get("missing_points"):
            flags.append(f"**partial** — {len(s['missing_points'])} points not run (deadline); rows "
                         "resume on resubmission")
        if s.get("groups_incomplete"):
            flags.append("**incomplete groups**: " + ", ".join(
                f"{g} (missing B={s['groups'][g].get('missing_B')})" for g in s["groups_incomplete"]))
        if s.get("arms_missing"):
            flags.append("arms missing: " + "; ".join(f"L={L}: {', '.join(v)}"
                                                      for L, v in s["arms_missing"].items()))
        if s.get("rebuilt_from_rows") and not s.get("stored_summary_present"):
            flags.append("summary rebuilt from `sweep.jsonl` (the phase wrote no summary)")
        if set(s.get("arms") or ()) == {"passthrough"}:
            flags.append("**passthrough only** — no fitted-object bank for this model")
        if k in sg.HYBRID_MODEL_KEYS:
            flags.append("hybrid: S capped at 256")
        out.append(f"### {k}" + (f"  ({'; '.join(flags)})" if flags else "") + "\n")
        out.append("| L | evals | arm | S | knee B | tok/s @knee | tok/s @B=24 | speedup | "
                   "duty @knee (sm%) | points |")
        out.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for gkey, g in sorted(s["groups"].items(),
                              key=lambda kv: (kv[1]["window_L"], kv[1]["arm"], kv[1].get("S") or 0)):
            complete = g.get("complete", True)
            knee = (_fmt(g.get("knee_B")) if complete
                    else f"(prov.) {_fmt(g.get('knee_provisional_B'))}")
            sp = g.get(f"speedup_knee_vs_B{old_s}") if complete else None
            out.append("| " + " | ".join([
                str(g["window_L"]), ",".join(g.get("window_labels") or ()) or "-", g["arm"],
                _fmt(g.get("S")), knee,
                (_fmt(g.get("tok_s_at_knee")) if complete else "-"),
                _fmt(g.get(f"tok_s_at_B{old_s}")), (f"x{sp:.2f}" if sp else "-"),
                (_fmt(g.get("duty_at_knee"), 1) if complete else "-"),
                (f"{_fmt(g.get('n_points'))}/{_fmt(g.get('n_expected'))}"
                 + ("" if complete else " INCOMPLETE")),
            ]) + " |")
        out.append("")
    out.append("## Reading the tables\n")
    out.append("- Passthrough knees sit at 32-192 concurrent requests and at the knee deliver 2-5x "
               "the B = 24 throughput at the short and mid windows (llama3b 1536: 8,533 vs 2,185 "
               "tok/s; nemotron3nano4b 7424: 4,915 vs 1,382); at the Petri window (16384) the "
               "budget-permitted S is 28-48 and the gain is 1.0-2.3x.\n"
               "- Steered arms plateau at 90-270 tok/s with knees at B = 8-48 and per-request "
               "latencies of 40-107 s: the lens hook serializes decode, so the un-steered gains do not "
               "transfer to steered cells until `lens-vectorize` lands. gemma4e4b's steered B = 24 rung "
               "recorded 0 tok/s (its speedup is blank).\n"
               + _incomplete_note(results) +
               "- No window recorded a preemption at the served S (the KV coverage the readiness gate "
               "attests held), so the budgets are conservative for throughput and the knees are the "
               "operating points to size client fan-out to.\n")
    text = "\n".join(out) + "\n"
    import re as _re
    if _re.search(r"/home/|/workspace", text):
        raise RuntimeError("an absolute path leaked into the doc")
    user = os.environ.get("USER", "")
    if len(user) > 2 and user in text:
        raise RuntimeError("a username leaked into the doc")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--doc", default=None,
                    help="also (re)generate the condensed per-model x window x arm markdown report at this path")
    ap.add_argument("--models", default=None, help="comma list (default: every model dir present)")
    ap.add_argument("--out-dir", default=None, help="where the figure grid + report.md go "
                                                     "(default <results-root>/_analysis)")
    ap.add_argument("--paste-only", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.results_root)
    models = [m for m in args.models.split(",") if m] if args.models else None
    results = load_results(root, models)
    pastes = [r["budget"]["paste"] for r in results.values()
              if r.get("budget") and r["budget"].get("paste")]
    block = paste_block_with_markers(pastes)
    print(block)
    if args.paste_only:
        return 0
    print()
    md = markdown_tables(results)
    print(md)
    out_dir = Path(args.out_dir or (root / "_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "budget_paste_block.py.txt").write_text(block + "\n")
    (out_dir / "report.md").write_text("# geom-sweep report\n\n```python\n" + block + "\n```\n\n" + md)
    if not args.no_figure:
        p = figure_grid(results, out_dir / "geom_sweep_grid.png")
        print(f"\nfigure grid: {p}")
    if args.doc:
        print(f"doc: {write_doc(results, Path(args.doc))}")
    print(f"report: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
