#!/usr/bin/env python
"""Petri (alignment audit) per-scenario read-out for cells audited under a SIDECAR root of a full22 view — ONE generator,
parameterised, in place of the per-lane copies (petri_stats_newcells.py, petri_full22_prompted_stats.py, ... — each a copy of
w7_rest/petri_full22_stats.py with its TAGS and root retargeted; a defect fixed in one never reached the others).

A sidecar root is ~/.local/state/petri/petri_full22_<tag><suffix>/ written by a driver that audits cells ADDED after the
view's wave froze its enumeration (the `_prompted` and `_newcells` out-suffix legs) or an ARM cell (`_arditi`, `_alphasteer`):
petri_run_key is content-addressed by the full harness config, so the sidecar's enumeration re-keys every cell, and the
driver runs only the new cells — its `none` is a placeholder for pairing, never audited there. Per root:

  cells.json                         the sidecar's cell roster (cell_id -> petri_run_key, intervention_hash, serving identity)
  cells/<key>/results.json           the runner's per-behaviour summary (present ONLY when the cell COMPLETED: 60/60 audited)
  cells/<key>/rows.jsonl             every filed judge score: (task, item=scenario, metric=dimension) -> value, 1..10
  cells/<key>/petri_incomplete.json  the H9 failure ledger: unresolved (behaviour, scenario, dimension) entries with the reason
  cells/<key>/manifest.json          the resolved leg: scenario_bundle_hash, behavior_files_hash, model id + revision
  inspect_logs/<key>/*.eval          the inspect-ai logs; the JUDGE and AUDITOR model ids are read from their model_roles

Petri's judge is its own (claude-opus-5 per the logs' model_roles — asserted per leg, never assumed), NOT the StrongREJECT
haiku judge; the auditor is claude-sonnet-5 with a grok / abliterated-qwen fallback chain whose engagement is recorded per
cell (auditor_chain). Nothing is re-judged here: the scores were filed by the audit; this script reads them out with
intervals and paired contrasts.

Per cell x behaviour: n_items (scenarios scored), mean of the behaviour's primary dimension (1..10; higher = MORE of the
audited failure behaviour, except where the behaviour is a desirable trait — the name says which), a 10,000-draw percentile
bootstrap 95% interval over scenarios (seed 0), plus eval_awareness and scenario_realism means; the cell's completion state;
and the paired contrast on the SHARED scenarios (mean difference, paired bootstrap 95% over scenarios, and the sign-test p)
against the ANCHOR:
  --anchor own    the sidecar root's own `none` cell (the newcells precedent: it filed no scores, so every contrast is PENDING
                  and the export pairs against the wave's unsteered audit itself)
  --anchor wave   the VIEW's frozen `none` under petri_full22_<tag>/ (the pairing anchor the sidecar drivers name), read from
                  the enumeration that holds its results (cells.json first, then the cells.json.* snapshots — Nemotron's wave
                  cells were re-keyed by a later enumeration). The anchor's run key, enumeration, scenario bundle and behaviour
                  files hashes are recorded beside the contrast; a scenario-bundle mismatch refuses to pair (PENDING, reason
                  named); a behaviour-files or run-key mismatch is recorded and warned, not hidden. When --wave-stats names
                  stats files that carry the tag's `none`, the anchor's per-scenario scores are asserted EQUAL to the filed
                  ones (two producer records disagreeing is a refusal, never a silent pick).
The sidecar's own `none` stays in `cells` as the roster lists it (unaudited: status NO_RESULTS_NO_LEDGER, no per-scenario
scores), so a reader that pairs (view, none) against (view, cell) across stats files keeps using the wave's filed anchor.

usage: petri_stats_sidecar.py --suffix _arditi --tags gemma4e4b_it,qwen35_4b_it,nemotron3nano4b,llama8b_it --anchor wave --out <json>
       petri_stats_sidecar.py --suffix _newcells --tags qwen35_4b_it --anchor own --out <json>   (reproduces petri_stats_newcells.py)
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

PER_LEG_EXPECTED = 60      # 6 behaviours x 10 scenarios x 1 epoch
N_BOOT, SEED = 10_000, 0


# the transcript cell ids the Petri roster's cell ids correspond to (Petri uses `si_<si>__<arm>` with '-' -> '_'); a cell id
# with no `__` and no `si_` prefix (none, arditi, alphasteer) is its own transcript id
def transcripts_cell_id(petri_cell_id: str) -> str:
    if petri_cell_id == "none":
        return "none"
    if petri_cell_id.startswith("prompted__"):   # the prompted baseline: harness `prompted__safety` <-> TrainTest `prompted__safety_refusal`
        return "prompted__" + {"safety": "safety_refusal", "constitution": "si_constitution", "irrelevant": "si_irrelevant"}.get(petri_cell_id[len("prompted__"):], petri_cell_id[len("prompted__"):])
    if "__" not in petri_cell_id and not petri_cell_id.startswith("si_"):
        return petri_cell_id
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


def read_roles(root: Path, key: str) -> dict:
    """judge / auditor / target model ids from the inspect logs of this leg (header only; asserted identical across logs)."""
    from inspect_ai.log import read_eval_log
    roles: dict = {}
    seen = set()
    for p in sorted(glob.glob(str(root / "inspect_logs" / key / "*.eval"))):
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


def leg_manifest(cell_dir: Path) -> dict:
    p = cell_dir / "manifest.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    return {k: m.get(k) for k in ("scenario_bundle_hash", "behavior_files_hash", "model_id", "model_revision", "harness_config_hash")}


def primary_per_scenario(rows) -> dict:
    """behaviour -> {scenario_id: primary-dimension score}"""
    per: dict = {}
    for r in rows:
        if r["metric"] == r["behavior"]:
            per.setdefault(r["behavior"], {})[r["scenario_id"]] = r["value"]
    return {b: dict(sorted(v.items())) for b, v in sorted(per.items())}


def cell_block(root: Path, label: str, c: dict) -> dict:
    key = c["petri_run_key"]
    cd = root / "cells" / key
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
                raise SystemExit(f"{label}/{c['cell_id']}/{beh}: rows.jsonl mean {behaviors[beh]['mean']} != results.json mean "
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
        "model_roles": read_roles(root, key),
        "leg_manifest": leg_manifest(cd),
    }


def enumerations(root: Path) -> list:
    """the enumeration records under a root: the current cells.json, then every frozen cells.json.* snapshot"""
    return [n for n in ["cells.json"] if (root / n).exists()] + sorted(os.path.basename(p) for p in glob.glob(str(root / "cells.json.*")))


def resolve_run_key(root: Path, c: dict) -> tuple:
    """(run_key, source): the enumeration whose results.json actually exists for this cell — the CURRENT cells.json first,
    then the frozen snapshots cells.json.* (a later enumeration re-keys a cell whose audit already ran under an earlier
    snapshot: Nemotron's five wave cells after its derived + prompted cells were added; the same rule agg_petri.py applies).
    Falls back to the current key, with the source saying so, when no enumeration has results."""
    cur = c["petri_run_key"]
    if (root / "cells" / cur / "results.json").exists():
        return cur, "cells.json"
    for sn in sorted(glob.glob(str(root / "cells.json.*"))):
        for x in (json.loads(Path(sn).read_text()).get("canonical") or []):
            if x["cell_id"] == c["cell_id"] and x["petri_run_key"] != cur and (root / "cells" / x["petri_run_key"] / "results.json").exists():
                return x["petri_run_key"], os.path.basename(sn)
    return cur, "cells.json (no enumeration has results for this cell)"


def wave_anchor(state: Path, tag: str, sidecar_none: dict | None, wave_stats: list) -> dict:
    """the view's frozen `none` under petri_full22_<tag>/: per-scenario primary scores + identity, cross-checked against the
    filed wave stats when a file carries the tag."""
    root = state / f"petri_full22_{tag}"
    cj = root / "cells.json"
    if not cj.exists():
        return {"status": "PENDING", "reason": f"no cells.json under {root}"}
    c = next((x for x in (json.loads(cj.read_text()).get("canonical") or []) if x["cell_id"] == "none"), None)
    if c is None:
        return {"status": "PENDING", "reason": f"no `none` cell in {cj}"}
    key, src = resolve_run_key(root, c)
    cd = root / "cells" / key
    rows = load_rows(cd)
    per = primary_per_scenario(rows)
    man = leg_manifest(cd)
    out = {"status": "OK" if per else "PENDING", "root": root.name, "petri_run_key": key, "petri_run_key_source": src, "rows_filed": len(rows),
           "n_scores": sum(len(v) for v in per.values()), "harness_config_hash": c.get("harness_config_hash"), "intervention_hash": c.get("intervention_hash"),
           "leg_manifest": man, "per_scenario": per,
           "sidecar_none_run_key": (sidecar_none or {}).get("petri_run_key"),
           "run_key_matches_sidecar_none": (key == (sidecar_none or {}).get("petri_run_key")),
           "cross_checked_against": []}
    if not per:
        out["reason"] = f"no primary-dimension scores under {cd}"
    for f in wave_stats:
        try:
            d = json.loads(Path(f).read_text())
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"--wave-stats {f}: unreadable ({type(e).__name__}: {e})")
        filed = ((d.get("tags") or {}).get(tag) or {}).get("cells", {}).get("none")
        if not filed:
            continue
        filed_per = {b: dict(bb.get("per_scenario") or {}) for b, bb in (filed.get("behaviors") or {}).items() if bb.get("per_scenario")}
        if filed_per != per:
            raise SystemExit(f"{tag}: the wave `none` read from {cd} disagrees with the per-scenario scores filed in {f} "
                             f"(tags.{tag}.cells.none) — two producer records disagree; refusing")
        if filed.get("petri_run_key") != key:
            raise SystemExit(f"{tag}: wave `none` run key {key} (from {src}) != {filed.get('petri_run_key')} filed in {f}; refusing")
        out["cross_checked_against"].append(str(f))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", required=True, help="the sidecar root suffix, e.g. _arditi, _alphasteer, _newcells, _prompted")
    ap.add_argument("--tags", required=True, help="comma-separated view tags; the root read is petri_full22_<tag><suffix>")
    ap.add_argument("--anchor", choices=["own", "wave"], default="wave")
    ap.add_argument("--wave-stats", default="", help="comma-separated stats files whose tags.<tag>.cells.none the wave anchor is asserted equal to (when present)")
    ap.add_argument("--state", default=str(Path.home() / ".local/state/petri"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    state = Path(os.path.expanduser(args.state))
    tags = [t for t in args.tags.split(",") if t]
    wave_stats = [os.path.expanduser(f) for f in args.wave_stats.split(",") if f]
    pairing = ("paired mean difference on SHARED scenario ids vs the tag's own bare none; bootstrap over scenarios; exact two-sided sign test"
               if args.anchor == "own" else
               "paired mean difference on SHARED scenario ids vs the VIEW's frozen bare none under petri_full22_<tag>/ (the sidecar's own none is "
               "an unaudited placeholder); bootstrap over scenarios; exact two-sided sign test; anchor identity recorded per tag in `anchor`")
    out = {"meta": {"source_root": str(state), "suffix": args.suffix, "anchor_mode": args.anchor, "tags_requested": tags,
                    "expected_per_leg": PER_LEG_EXPECTED, "interval": "percentile bootstrap over scenarios, 10,000 draws, seed 0",
                    "pairing": pairing, "written_at": datetime.now(timezone.utc).isoformat(), "judge": {}}, "tags": {}}
    judge_models, auditor_models = set(), set()
    warnings = []
    for tag in tags:
        root = state / f"petri_full22_{tag}{args.suffix}"
        enums = {e: json.loads((root / e).read_text()) for e in enumerations(root)}
        if not enums:
            out["tags"][tag] = {"status": "PENDING", "reason": f"no enumeration under {root}", "root": root.name}
            print(f"== {tag}: PENDING — no enumeration under {root}")
            continue
        roster = {"audit": next(iter(enums.values())).get("audit"), "absent": sum((e.get("absent") or [] for e in enums.values()), [])}
        # per cell: the enumeration whose run key has results (or rows); both having results is a problem
        chosen: dict = {}
        for ename, e in enums.items():
            for c in e["canonical"]:
                cd = root / "cells" / c["petri_run_key"]
                has = (cd / "results.json").exists() or (cd / "rows.jsonl").exists()
                prev = chosen.get(c["cell_id"])
                if has and prev is not None and prev[2]:
                    raise SystemExit(f"{tag}/{c['cell_id']}: results under BOTH enumerations ({prev[0]} and {ename}) — refusing to pick")
                if prev is None or (has and not prev[2]):
                    chosen[c["cell_id"]] = (ename, c, has)
        cells = {}
        for cid, (ename, c, has) in chosen.items():
            b = cell_block(root, f"{tag}{args.suffix}", c)
            b["enumeration"] = ename
            b["harness_config_hash"] = c.get("harness_config_hash")
            cells[b["cell"]] = b
            for mr in b["model_roles"].get("model_roles_distinct", []):
                if "judge" in mr: judge_models.add(mr["judge"])
                if "auditor" in mr: auditor_models.add(mr["auditor"])
        # the anchor the contrasts pair against
        if args.anchor == "own":
            bare = cells.get("none")
            anchor_per = {beh: bb.get("per_scenario", {}) for beh, bb in (bare or {}).get("behaviors", {}).items()}
            anchor = {"mode": "own", "petri_run_key": (bare or {}).get("petri_run_key"), "n_scores": sum(len(v) for v in anchor_per.values())}
        else:
            sidecar_none = next((c for _, c, _ in chosen.values() if c["cell_id"] == "none"), None)
            anchor = {"mode": "wave", **wave_anchor(state, tag, sidecar_none, wave_stats)}
            anchor_per = anchor.get("per_scenario") or {}
            if anchor.get("status") != "OK":
                warnings.append(f"{tag}: wave anchor unavailable — {anchor.get('reason')}; every contrast PENDING")
            elif not anchor["run_key_matches_sidecar_none"]:
                warnings.append(f"{tag}: the sidecar's `none` run key {anchor['sidecar_none_run_key']} != the wave's {anchor['petri_run_key']} "
                                f"(harness inventories differ; same scenario bundle is checked per cell below) — pairing on scenario ids")
        vs_bare = {}
        for cid, b in cells.items():
            if cid == "none":
                continue
            vs_bare[cid] = {}   # an anchor without scores yields a PENDING block per behaviour ("no shared scenarios"), as the precedent filed
            if args.anchor == "wave":
                ab, cb = (anchor.get("leg_manifest") or {}).get("scenario_bundle_hash"), (b.get("leg_manifest") or {}).get("scenario_bundle_hash")
                if b["status"] != "NO_RESULTS_NO_LEDGER" and ab and cb and ab != cb:
                    vs_bare[cid] = {"status": "PENDING", "reason": f"scenario bundle differs: cell {cb} vs anchor {ab} — the scenario ids are not the same items; not paired"}
                    warnings.append(f"{tag}/{cid}: {vs_bare[cid]['reason']}")
                    continue
                af, cf = (anchor.get("leg_manifest") or {}).get("behavior_files_hash"), (b.get("leg_manifest") or {}).get("behavior_files_hash")
                if b["status"] != "NO_RESULTS_NO_LEDGER" and af and cf and af != cf:
                    warnings.append(f"{tag}/{cid}: behavior_files_hash differs from the anchor's ({cf} vs {af}); paired on the shared scenario ids, recorded per cell")
                    b["anchor_behavior_files_match"] = False
                elif af and cf:
                    b["anchor_behavior_files_match"] = True
            for beh, bb in b["behaviors"].items():
                vs_bare[cid][beh] = paired_mean_diff(anchor_per.get(beh, {}), bb["per_scenario"])
        n_complete = sum(1 for b in cells.values() if b["status"] == "COMPLETE")
        out["tags"][tag] = {"audit": roster.get("audit"), "root": root.name, "n_cells": len(cells), "n_complete": n_complete,
                            "n_incomplete_ledgered": sum(1 for b in cells.values() if b["status"].startswith("INCOMPLETE")),
                            "absent": roster.get("absent"), "cells": cells, "vs_bare": vs_bare, "anchor": anchor}
        print(f"== {tag}{args.suffix}: {n_complete}/{len(cells)} cells COMPLETE; anchor={anchor['mode']}"
              + (f" {anchor.get('root')}/{anchor.get('petri_run_key')} ({anchor.get('petri_run_key_source')}), {anchor.get('n_scores')} scores, "
                 f"matches sidecar none key={anchor.get('run_key_matches_sidecar_none')}, cross-checked={len(anchor.get('cross_checked_against') or [])} file(s)"
                 if anchor["mode"] == "wave" else f" ({anchor.get('n_scores')} scores)"))
        for cid, b in cells.items():
            beh = " ".join(f"{k[:12]}={v['mean']:.2f}(n{v['n_items']})" for k, v in b["behaviors"].items() if v['mean'] is not None)
            print(f"   {cid:50s} {b['status']:28s} items={b['audited_items']}/{b['audited_expected']} {beh}")
            for beh_, d in (vs_bare.get(cid) or {}).items():
                if "delta" in d:
                    print(f"      vs anchor {beh_:28s} delta={d['delta']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}] n={d['n_paired']} sign_p={d['sign_test_p']:.3f}")
                elif cid != "none":
                    print(f"      vs anchor {beh_:28s} {d}")
    inflight = [f"{t}/{c}" for t, tb in out["tags"].items() for c, b in (tb.get("cells") or {}).items() if b["rerun_state"] != "not_needed"]
    out["meta"]["overflow_rerun"] = {"cells_with_in_flight_rerun": inflight,
                                     "note": ("a separate lane is re-running the target-context-overflow entries of these cells (27 samples across five "
                                              "cells incl. qwen35_4b none__lens0, which is outside the canonical roster read here); their unresolved "
                                              "entries are in flight, not abstentions; every such cell's block is a pre-re-run snapshot "
                                              "(ledger_snapshot) to be refreshed by re-running this script when the ledgers clear")}
    out["meta"]["judge"] = {"judge_models_seen": sorted(judge_models), "auditor_models_seen": sorted(auditor_models),
                            "statement": "read from every leg's inspect logs' model_roles (header only); Petri's own judge, not the StrongREJECT judge"}
    out["meta"]["warnings"] = warnings
    for w in warnings:
        print(f"WARNING {w}")
    if judge_models != {"anthropic/claude-opus-5"}:
        raise SystemExit(f"judge models seen across legs: {sorted(judge_models)} — expected exactly anthropic/claude-opus-5; refusing"
                         + (" (no finished leg under this suffix yet: nothing to file)" if not judge_models else ""))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}; judge={sorted(judge_models)} auditor={sorted(auditor_models)}")


if __name__ == "__main__":
    main()
