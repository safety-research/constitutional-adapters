#!/usr/bin/env python
"""Petri (alignment audit) read-out for the full22 wave — five audit tags (gemma4e4b, gemma4e4b_it, qwen35_4b, qwen35_4b_it,
nemotron3nano4b), every canonical cell, from the producers' own records under ~/.local/state/petri/petri_full22_<tag>/:

  cells.json                         the canonical cell roster (cell_id -> petri_run_key, intervention_hash, serving identity)
  cells/<key>/results.json           the runner's per-behaviour summary (present ONLY when the cell COMPLETED: 60/60 audited)
  cells/<key>/rows.jsonl             every filed judge score: (task, item=scenario, metric=dimension) -> value, 1..10
  cells/<key>/petri_incomplete.json  the H9 failure ledger: unresolved (behaviour, scenario, dimension) entries with the reason
  inspect_logs/<key>/*.eval          the inspect-ai logs; the JUDGE and AUDITOR model ids are read from their model_roles

Petri's judge is its own (claude-opus-5 per the logs' model_roles — asserted per leg, never assumed), NOT the StrongREJECT
haiku judge; the auditor is claude-sonnet-5 with a grok / abliterated-qwen fallback chain whose engagement is recorded per
cell (auditor_chain). Nothing is re-judged here: the scores were filed by the audit; this script reads them out with
intervals and paired contrasts.

Per cell x behaviour: n_items (scenarios scored), mean of the behaviour's primary dimension (1..10; higher = MORE of the
audited failure behaviour, except where the behaviour is a desirable trait — the name says which), a 10,000-draw percentile
bootstrap 95% interval over scenarios (seed 0), plus eval_awareness and scenario_realism means; the cell's completion state
(COMPLETE 60/60, or INCOMPLETE on the overflow ledger with the unresolved scenarios named); paired contrasts vs the tag's own
bare `none` cell on the SHARED scenarios (mean difference, paired bootstrap 95% over scenarios, and the sign-test p). Rows are
also pooled per cell across the six behaviours as a mean-of-behaviour-means (each behaviour weighted equally).

usage: petri_full22_stats.py --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path.home() / ".local/state/petri"
# nemotron_complete (2026-09-08): ONE tag, TWO enumerations. The six wave-1 legs (none, base4, none__lens0) were run under
# the 254ccbb4 manifest (harness hash 7ac0…/b76a…); adding the prompted cell moved every harness hash (six
# <behavior>/PROMPTED_SAFETY definitions joined the inventory — the only differing input), so the new legs run under the
# S1/S2 enumeration (cells.json) while the finished ones stand under cells.json.wave1-254ccbb4. Each cell is read from the
# enumeration that holds its results (a cell with results in BOTH would be a problem, never a silent pick); the
# enumeration and harness hash are recorded per cell. Paired contrasts vs `none` pair on scenario ids, which are the
# same frozen bundle in both enumerations.
TAGS = ["qwen35_4b_it"]   # the two batch-64 cells audited under the _newcells out-suffix root (proxy 498858)
ENUMERATIONS = ["cells.json.wave1-254ccbb4", "cells.json"]
PER_LEG_EXPECTED = 60      # 6 behaviours x 10 scenarios x 1 epoch
N_BOOT, SEED = 10_000, 0
# the transcript cell ids the Petri roster's cell ids correspond to (Petri uses `si_<si>__<arm>` with '-' -> '_')
def transcripts_cell_id(petri_cell_id: str) -> str:
    if petri_cell_id == "none":
        return "none"
    if petri_cell_id.startswith("prompted__"):   # the prompted baseline: harness `prompted__safety` <-> TrainTest `prompted__safety_refusal`
        return "prompted__" + {"safety": "safety_refusal", "constitution": "si_constitution", "irrelevant": "si_irrelevant"}.get(petri_cell_id[len("prompted__"):], petri_cell_id[len("prompted__"):])
    s = petri_cell_id
    if s.startswith("si_"):
        s = s[3:]
    s = s.replace("si_safety__", "safety_refusal__")
    si, _, arm = s.partition("__")
    si = {"constv2_mix_full": "constv2-mix-full", "constv2ctl2_full": "constv2ctl2-full", "safety": "safety_refusal"}.get(si, si)
    return f"{si}__{arm}"


def boot_ci_mean(vals, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = np.sort(v[idx].mean(axis=1))
    return (float(m[int(0.025 * n_boot)]), float(m[int(0.975 * n_boot) - 1]))


def paired_mean_diff(a: dict, b: dict, n_boot=N_BOOT, seed=SEED) -> dict:
    """b minus a over the shared item ids; percentile bootstrap over items; exact two-sided sign test on non-zero diffs."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"status": "PENDING", "reason": "no shared scenarios"}
    d = np.asarray([b[i] - a[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = np.sort(d[idx].mean(axis=1))
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    nz = pos + neg
    p = 1.0 if nz == 0 else min(1.0, 2 * sum(comb(nz, i) for i in range(min(pos, neg) + 1)) / (2 ** nz))
    lo, hi = float(m[int(0.025 * n_boot)]), float(m[int(0.975 * n_boot) - 1])
    return {"n_paired": len(ids), "delta": float(d.mean()), "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "n_pos": pos, "n_neg": neg, "n_tie": len(ids) - nz, "sign_test_p": p, "n_boot": n_boot, "seed": seed}


def read_roles(tag: str, key: str) -> dict:
    """judge / auditor / target model ids from the inspect logs of this leg (header only; asserted identical across logs)."""
    from inspect_ai.log import read_eval_log
    roles: dict = {}
    seen = set()
    for p in sorted(glob.glob(str(ROOT / f"petri_full22_{tag}_newcells" / "inspect_logs" / key / "*.eval"))):
        try:
            lg = read_eval_log(p, header_only=True)
        except Exception as e:  # noqa: BLE001
            roles.setdefault("unreadable_logs", []).append(f"{os.path.basename(p)}: {type(e).__name__}")
            continue
        mr = getattr(lg.eval, "model_roles", None) or {}
        tup = tuple(sorted((r, getattr(m, "model", str(m))) for r, m in mr.items()))
        seen.add(tup)
        roles.setdefault("n_logs", 0); roles["n_logs"] += 1
    roles["model_roles_distinct"] = [dict(t) for t in sorted(seen)]
    return roles


def load_rows(cell_dir: Path):
    rows = []
    p = cell_dir / "rows.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            e = json.loads(r.get("extra_json") or "{}")
            # a PROMPTED cell audits under the SI-variant behavior dir (<behavior>/PROMPTED_<SI>); the behaviour is the base
            # name (= the primary metric), the variant dir is kept beside it
            bkey = e.get("behavior_dir_key") or r["task"].split("/", 1)[-1]
            rows.append({"behavior": bkey.split("/", 1)[0], "behavior_dir_key": bkey, "dimension": e.get("dimension"),
                         "metric": r["metric"], "scenario_id": r["item_id"], "value": float(r["value"]), "epoch": e.get("epoch"),
                         "auditor_used": [u.get("label") for u in (e.get("auditor_chain") or {}).get("used", [])]})
    return rows


def cell_block(tag: str, c: dict) -> dict:
    key = c["petri_run_key"]
    cd = ROOT / f"petri_full22_{tag}_newcells" / "cells" / key
    results = json.loads((cd / "results.json").read_text()) if (cd / "results.json").exists() else None
    ledger = json.loads((cd / "petri_incomplete.json").read_text()) if (cd / "petri_incomplete.json").exists() else {}
    unresolved = ledger.get("unresolved") or []
    rows = load_rows(cd)
    behaviors: dict = {}
    by_beh: dict = {}
    for r in rows:
        by_beh.setdefault(r["behavior"], []).append(r)
    for beh, rs in sorted(by_beh.items()):
        prim = {r["scenario_id"]: r["value"] for r in rs if r["metric"] == beh}
        ea = {r["scenario_id"]: r["value"] for r in rs if r["metric"] == "eval_awareness"}
        sr = {r["scenario_id"]: r["value"] for r in rs if r["metric"] == "scenario_realism"}
        lo, hi = boot_ci_mean(list(prim.values()))
        unres_here = [u for u in unresolved if u.get("behavior") == beh]
        behaviors[beh] = {
            "behavior_dir_keys": sorted({r["behavior_dir_key"] for r in rs}),
            "n_items": len(prim), "n_expected": 10, "mean": (statistics.fmean(prim.values()) if prim else None), "lo": lo, "hi": hi,
            "per_scenario": dict(sorted(prim.items())),
            "eval_awareness_mean": (statistics.fmean(ea.values()) if ea else None),
            "scenario_realism_mean": (statistics.fmean(sr.values()) if sr else None),
            "incomplete": bool(unres_here), "unresolved_scenarios": sorted({u["scenario_id"] for u in unres_here}),
            "results_json_mean": ((results or {}).get("behaviors", {}).get(beh, {}).get("mean", {}).get(beh) if results else None),
        }
        if results and behaviors[beh]["results_json_mean"] is not None and behaviors[beh]["mean"] is not None:
            if abs(behaviors[beh]["results_json_mean"] - behaviors[beh]["mean"]) > 1e-6:
                raise SystemExit(f"{tag}/{c['cell_id']}/{beh}: rows.jsonl mean {behaviors[beh]['mean']} != results.json mean "
                                 f"{behaviors[beh]['results_json_mean']} — the two producer records disagree; refusing")
    n_items_total = sum(b["n_items"] for b in behaviors.values())
    chain = (results or {}).get("auditor_chain")
    if chain is None:
        # derive from rows when the runner wrote no results.json (ledgered cell)
        used = {}
        for r in rows:
            k = "sonnet_only" if r["auditor_used"] == ["sonnet-5"] else "+".join(r["auditor_used"]) or "unknown"
            used[k] = used.get(k, 0) + 1
        chain = {"from_rows": used}
    means = [b["mean"] for b in behaviors.values() if b["mean"] is not None]
    # Ledger snapshot: a separate lane is re-running the context-overflow entries of the ledgered cells
    # (petri_run.py run --overflow-rerun). Their unresolved entries are IN FLIGHT, not final
    # abstentions, and this read-out of such a cell is a PRE-RE-RUN SNAPSHOT that must be refreshed once the ledger clears.
    rows_p, ledger_p = cd / "rows.jsonl", cd / "petri_incomplete.json"
    snapshot = {"rows_jsonl_mtime": (rows_p.stat().st_mtime if rows_p.exists() else None), "rows_filed_at_snapshot": len(rows),
                "ledger_mtime": (ledger_p.stat().st_mtime if ledger_p.exists() else None), "ledger_written_at": ledger.get("written_at"),
                "snapshot_at": datetime.now(timezone.utc).isoformat()}
    rerun_state = ("in_flight_by_another_lane (petri_run.py run --overflow-rerun); the unresolved entries are NOT abstentions; "
                   "this cell's read-out is a PRE-RE-RUN SNAPSHOT and must be refreshed") if unresolved else "not_needed"
    return {
        "rerun_state": rerun_state, "ledger_snapshot": snapshot,
        "petri_cell_id": c["cell_id"], "cell": transcripts_cell_id(c["cell_id"]), "petri_run_key": key,
        "intervention_hash": c.get("intervention_hash"), "serving_identity": c.get("serving_identity"),
        "virtual_model": c.get("virtual_model_name"), "kind": c.get("kind"),
        "status": "COMPLETE" if results is not None else ("INCOMPLETE_OVERFLOW_LEDGER_RERUN_IN_FLIGHT" if unresolved else "NO_RESULTS_NO_LEDGER"),
        "audited_items": (results or {}).get("audited", {}).get("items", n_items_total), "audited_expected": PER_LEG_EXPECTED,
        "rows_filed": len(rows), "n_unresolved_ledger_entries": len(unresolved),
        "unresolved": [{"behavior": u.get("behavior"), "scenario_id": u.get("scenario_id"), "dimension": u.get("dimension"),
                        "reason_class": ("target_context_overflow" if "context overflow" in str(u.get("reason", "")) else "other"),
                        "retries": u.get("retries")} for u in unresolved],
        "auditor_chain": chain, "auditor_refused": (results or {}).get("auditor_refused"),
        "behaviors": behaviors,
        "mean_of_behavior_means": (statistics.fmean(means) if means else None),
        "completed_at": (results or {}).get("completed_at"),
        "model_roles": read_roles(tag, key),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = {"meta": {"source_root": str(ROOT), "expected_per_leg": PER_LEG_EXPECTED, "interval": "percentile bootstrap over scenarios, 10,000 draws, seed 0",
                    "pairing": "paired mean difference on SHARED scenario ids vs the tag's own bare none; bootstrap over scenarios; exact two-sided sign test",
                    "written_at": datetime.now(timezone.utc).isoformat(), "judge": {}}, "tags": {}}
    judge_models, auditor_models = set(), set()
    for tag in TAGS:
        enums = {e: json.loads((ROOT / f"petri_full22_{tag}_newcells" / e).read_text()) for e in ENUMERATIONS if (ROOT / f"petri_full22_{tag}_newcells" / e).exists()}
        if not enums:
            out["tags"][tag] = {"status": "PENDING", "reason": f"no enumeration under {ROOT / f'petri_full22_{tag}'}"}
            continue
        roster = {"audit": next(iter(enums.values())).get("audit"), "absent": sum((e.get("absent") or [] for e in enums.values()), [])}
        # per cell: the enumeration whose run key has results (or rows); both having results is a problem
        chosen: dict = {}
        for ename, e in enums.items():
            for c in e["canonical"]:
                cd = ROOT / f"petri_full22_{tag}_newcells" / "cells" / c["petri_run_key"]
                has = (cd / "results.json").exists() or (cd / "rows.jsonl").exists()
                prev = chosen.get(c["cell_id"])
                if has and prev is not None and prev[2]:
                    raise SystemExit(f"{tag}/{c['cell_id']}: results under BOTH enumerations ({prev[0]} and {ename}) — refusing to pick")
                if prev is None or (has and not prev[2]):
                    chosen[c["cell_id"]] = (ename, c, has)
        cells = {}
        for cid, (ename, c, has) in chosen.items():
            b = cell_block(tag, c)
            b["enumeration"] = ename
            b["harness_config_hash"] = c.get("harness_config_hash")
            cells[b["cell"]] = b
            for mr in b["model_roles"].get("model_roles_distinct", []):
                if "judge" in mr: judge_models.add(mr["judge"])
                if "auditor" in mr: auditor_models.add(mr["auditor"])
        bare = cells.get("none")
        vs_bare = {}
        for cid, b in cells.items():
            if cid == "none" or bare is None:
                continue
            vs_bare[cid] = {}
            for beh, bb in b["behaviors"].items():
                a = bare["behaviors"].get(beh, {}).get("per_scenario", {})
                vs_bare[cid][beh] = paired_mean_diff(a, bb["per_scenario"])
        n_complete = sum(1 for b in cells.values() if b["status"] == "COMPLETE")
        out["tags"][tag] = {"audit": roster.get("audit"), "n_cells": len(cells), "n_complete": n_complete,
                            "n_incomplete_ledgered": sum(1 for b in cells.values() if b["status"].startswith("INCOMPLETE")),
                            "absent": roster.get("absent"), "cells": cells, "vs_bare": vs_bare}
        print(f"== {tag}: {n_complete}/{len(cells)} cells COMPLETE")
        for cid, b in cells.items():
            beh = " ".join(f"{k[:12]}={v['mean']:.2f}(n{v['n_items']})" for k, v in b["behaviors"].items() if v['mean'] is not None)
            print(f"   {cid:50s} {b['status']:28s} items={b['audited_items']}/{b['audited_expected']} {beh}")
    inflight = [f"{t}/{c}" for t, tb in out["tags"].items() for c, b in (tb.get("cells") or {}).items() if b["rerun_state"] != "not_needed"]
    out["meta"]["overflow_rerun"] = {"cells_with_in_flight_rerun": inflight,
                                     "note": ("a separate lane is re-running the target-context-overflow entries of these cells (27 samples across five "
                                              "cells incl. qwen35_4b none__lens0, which is outside the canonical roster read here); their unresolved "
                                              "entries are in flight, not abstentions; every such cell's block is a pre-re-run snapshot "
                                              "(ledger_snapshot) to be refreshed by re-running this script when the ledgers clear")}
    out["meta"]["judge"] = {"judge_models_seen": sorted(judge_models), "auditor_models_seen": sorted(auditor_models),
                            "statement": "read from every leg's inspect logs' model_roles (header only); Petri's own judge, not the StrongREJECT judge"}
    if judge_models != {"anthropic/claude-opus-5"}:
        raise SystemExit(f"judge models seen across legs: {sorted(judge_models)} — expected exactly anthropic/claude-opus-5; refusing")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}; judge={sorted(judge_models)} auditor={sorted(auditor_models)}")


if __name__ == "__main__":
    main()
