#!/usr/bin/env python
"""Live smoke of the per-item store adapters on THIS machine's stores (one live check per adapter that
reports counts without asserting exact numbers).

Builds a scratch registry root, loads the dimension seeds through the seeds branch's ``registry dimensions load``
(the seeds checkout's ``src`` on PYTHONPATH — that branch is not merged into this one), then runs each adapter's
``discover`` → incremental pass → ingest and prints, per adapter: inputs, groups refreshed / refused (with the
refusal reasons grouped), records written, ingest ok / refused / blocked, rows per table, degeneration states,
judgment counts (items judged, refusals, errors, api pools), the ``unverified`` field tally over the runs, and the
two-judge side-by-side count for (run, item) pairs present in both an original store and the regrade twin.

Bounded by construction: every adapter opens named files and lists one directory level; the sampled judge caches
(~50 MB × 162) and the twin's Haiku originals are NOT sha-hashed here (``--hash-caches`` turns that on; the
adapters' default is to hash them — the smoke says which it did).

    PYTHONPATH=$PWD/src nice -n 10 .venv/bin/python scripts/registry/smoke_items_adapters.py \\
        --root /tmp/registry-smoke-items --json /tmp/registry-smoke-items/report.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry.adapters import base as AB  # noqa: E402
from sisteer.registry.adapters import runner as RUN  # noqa: E402

HOME = os.path.expanduser("~")
#: the main checkout (the parent of git's common dir), under which the peer checkouts below sit; this tree when git cannot say
_COMMON = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True,
                         text=True).stdout.strip()
MAIN = os.path.dirname(_COMMON) if _COMMON else REPO
DEFAULT_SEEDS_WT = os.path.join(MAIN, "worktrees", "registry-seeds")
PY = sys.executable


def _peer_cli(worktree: str, root: str, *args: str) -> subprocess.CompletedProcess:
    """Run ``registry`` from a PEER worktree's source tree (its own schema / vocabulary) against the scratch root."""
    env = dict(os.environ, PYTHONPATH=os.path.join(worktree, "src"))
    return subprocess.run([PY, "-m", "sisteer.registry.cli", "--root", root, *args], env=env, capture_output=True, text=True, cwd=worktree)


def load_dimensions(root: str, seeds_wt: str) -> Dict[str, object]:
    """The seeds branch's seeds through its own loader AND harvester (its vocabulary additions are not on this branch's core yet:
    a seed row spelled with one of them would be refused by this branch's validator, so the seeds branch ingests them)."""
    t0 = time.time()
    p = _peer_cli(seeds_wt, root, "dimensions", "load", "--seeds", os.path.join(seeds_wt, "configs", "registry"), "--workers", "0")
    if p.returncode != 0:
        raise SystemExit(f"dimension seeds refused (rc={p.returncode}):\n{p.stderr[-4000:]}")
    rep = json.loads(p.stdout)
    return {"seed_revision": rep.get("revision"), "records": rep.get("records"), "warnings": len(rep.get("warnings") or []),
            "ingest": rep.get("ingest"), "refusals_head": [l[:220] for l in p.stderr.splitlines() if l.startswith("refused:")][:8],
            "seconds": round(time.time() - t0, 1), "ingested_by": "registry-seeds worktree harvester"}


def load_objects(root: str, objects_wt: str, seeds_wt: str) -> Dict[str, object]:
    """PR-3d's object rows through its ``registry objects scan --emit`` (the banks reachable from this machine), then one
    ingest cycle under the seeds worktree's harvester (same schema owner as the dimension rows)."""
    t0 = time.time()
    p = _peer_cli(objects_wt, root, "objects", "scan", "--emit", "--allow-refusals", "--no-hpo", "--seeds", os.path.join(seeds_wt, "configs", "registry"), "--json")
    out: Dict[str, object] = {"rc": p.returncode, "seconds_scan": round(time.time() - t0, 1)}
    try:
        rep = json.loads(p.stdout)
        ob = rep.get("objects") or {}
        out.update(objects=ob.get("n_objects") or ob.get("objects"), refusals=ob.get("n_refusals") or len(ob.get("refusals") or []), emit=rep.get("emit"))
    except Exception:
        out["stdout_tail"] = p.stdout[-800:]
        out["stderr_tail"] = p.stderr[-800:]
    t1 = time.time()
    q = _peer_cli(seeds_wt, root, "ingest", "--workers", "0")
    out["ingest_rc"] = q.returncode
    out["ingest_stdout_tail"] = q.stdout[-600:]
    out["seconds_ingest"] = round(time.time() - t1, 1)
    return out


def seed_none_and_prompted_objects(root: str, now: str) -> Dict[str, object]:
    """The `none` object and one `si:<sha>` object per ingested si_text row — dimension facts no adapter owns yet;
    written into the scratch root so runs on unsteered / prompted cells can fold."""
    from sisteer.registry import identity
    from sisteer.registry import records as R
    conn = DB.connect(DB.db_path(root), readonly=True)
    si_rows = conn.execute("SELECT si_id, si_text_sha256, si_kind, si_token_count_by_model_json FROM si_text").fetchall()
    have = {r[0] for r in conn.execute("SELECT object_id FROM object")}
    conn.close()
    prod = {"script": "scripts/registry/smoke_items_adapters.py", "code_sha": "unverified", "session": None, "host": "smoke", "substrate": "grading_host"}
    label = {"safety_refusal": ("safety_si", "safety"), "si_constitution": ("constitution_si", "constitution"),
             "si_constitution_full": ("constitution_si_full", "constitution"), "si_irrelevant": ("other", "irrelevant")}
    recs = []
    if "none" not in have:
        p = {"object_id": "none", "object_kind": "none", "paper_label": "none", "campaign": "full22", "object_role": "baseline_none", "fit_target_kind": "none",
             "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable", "fit_positions": "not_applicable", "fit_seed_confidence": "not_applicable",
             "derivation": "none", "artifact_sha256": "none", "status": "approved", "notes": "the unsteered object (smoke-seeded; a seed owner is needed)"}
        recs.append(R.make_record("object", p, prod, identity.adapter_record_id(p, "object"), now))
    for si_id, sha, kind, counts in si_rows:
        oid = f"si:{sha}"
        if oid in have or not isinstance(sha, str) or len(sha) != 64:
            continue
        try:
            by_model = json.loads(counts or "{}")
        except Exception:
            by_model = {}
        n_tok = next((int(v) for v in by_model.values() if isinstance(v, int)), None) if isinstance(by_model, dict) else None
        pl, role = label.get(str(si_id), ("other", "prompt"))
        p = {"object_id": oid, "object_kind": "system_instruction", "paper_label": pl, "campaign": "full22", "object_role": "prompt", "fit_target_kind": "declared_prompt",
             "si_text_sha256": sha, "si_token_count": n_tok, "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable", "fit_positions": "not_applicable",
             "fit_seed_confidence": "not_applicable", "derivation": "none", "artifact_sha256": sha, "status": "approved",
             "notes": f"prompted object for si_text {si_id} (smoke-seeded from the ingested si_text row; a seed owner is needed)"}
        recs.append(R.make_record("object", p, prod, identity.adapter_record_id(p, "object"), now))
    if recs:
        R.write_records(os.path.join(root, "records", "smoke_objects", "000001-object.jsonl"), recs)
    ing = H.ingest_once(root, workers=0, min_free=0)
    return {"written": len(recs), "ingest": {"ok": ing.ok, "refused": ing.refused, "blocked": ing.blocked}, "refusals_head": [str(r)[:220] for r in ing.refusals[:6]]}


def table_counts(conn, run_ids: Optional[List[str]] = None) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for t in ("run", "judgment", "run_metric", "run_contrast", "item_result", "item_flags", "run_degeneration", "artifact", "computation"):
        out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    out["run_by_state"] = dict(conn.execute("SELECT lifecycle_state, COUNT(*) FROM run GROUP BY 1").fetchall())
    out["degeneration_by_state"] = dict(conn.execute("SELECT measurement_state, COUNT(*) FROM run_degeneration GROUP BY 1").fetchall())
    out["items_by_metric"] = dict(conn.execute("SELECT metric_id, COUNT(*) FROM item_result GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall())
    out["judgments"] = {"n": out["judgment"], "items_judged": conn.execute("SELECT COALESCE(SUM(n_items_judged),0) FROM judgment").fetchone()[0],
                        "refusals": conn.execute("SELECT COALESCE(SUM(n_judge_refusals),0) FROM judgment").fetchone()[0],
                        "errors": conn.execute("SELECT COALESCE(SUM(n_errors),0) FROM judgment").fetchone()[0],
                        "by_pool": dict(conn.execute("SELECT COALESCE(api_pool,'(mixed/unknown)'), COUNT(*) FROM judgment GROUP BY 1").fetchall()),
                        "by_judge_model": dict(conn.execute("SELECT jd.judge_model, COUNT(*) FROM judgment jm JOIN judge jd USING(judge_id) GROUP BY 1").fetchall())}
    out["items_by_pool"] = dict(conn.execute("SELECT COALESCE(api_pool,'(none)'), COUNT(*) FROM item_result GROUP BY 1").fetchall())
    out["runs_by_eval"] = dict(conn.execute("SELECT eval_id, COUNT(*) FROM run GROUP BY 1").fetchall())
    out["runs_by_model"] = dict(conn.execute("SELECT model_id, COUNT(*) FROM run GROUP BY 1").fetchall())
    out["blocked_records"] = dict(conn.execute("SELECT missing_parent_table, COUNT(*) FROM blocked_records GROUP BY 1").fetchall())
    unv: collections.Counter = collections.Counter()
    for (fc,) in conn.execute("SELECT field_confidence_json FROM run WHERE field_confidence_json IS NOT NULL"):
        try:
            d = json.loads(fc)
        except Exception:
            continue
        for col, ent in (d or {}).items():
            if isinstance(ent, dict) and ent.get("confidence") == "unverified":
                unv[col] += 1
    out["unverified_run_fields"] = dict(unv.most_common())
    return out


def side_by_side(conn) -> Dict[str, object]:
    rows = conn.execute("SELECT COUNT(*) FROM (SELECT run_id, item_id, trial_index, sample_index, turn_index, COUNT(DISTINCT judge_id) nj "
                        "FROM v_judge_side_by_side GROUP BY 1,2,3,4,5 HAVING nj >= 2)").fetchone()[0]
    by_models = conn.execute("SELECT group_concat(judge_model, '+'), COUNT(*) FROM (SELECT run_id, item_id, trial_index, sample_index, turn_index, judge_model "
                             "FROM v_judge_side_by_side GROUP BY 1,2,3,4,5,6 ORDER BY 6) GROUP BY run_id, item_id, trial_index, sample_index, turn_index").fetchall()
    combos: collections.Counter = collections.Counter()
    for models, _ in by_models:
        if "+" in (models or ""):
            combos[models] += 1
    runs_with_two = conn.execute("SELECT COUNT(DISTINCT run_id) FROM (SELECT run_id, item_id, trial_index, sample_index, turn_index, COUNT(DISTINCT judge_id) nj "
                                 "FROM v_judge_side_by_side GROUP BY 1,2,3,4,5 HAVING nj >= 2)").fetchone()[0]
    return {"run_item_pairs_with_two_judges": rows, "runs_with_two_judges": runs_with_two, "judge_model_combos": dict(combos)}


def run_adapter(root: str, name: str, adapter, now: str, limit_groups: Optional[int]) -> Dict[str, object]:
    rep: Dict[str, object] = {"adapter": name}
    t0 = time.time()
    # the S2 dispatch-site guard: this smoke discovers DIRECTLY on the adapter it constructed, so the factory
    # guard never saw it -- a host with no store access scope refuses here by name, before any store is listed
    AB.require_store_access(f"smoke_items_adapters: direct discovery of {name}")
    try:
        inputs = adapter.discover()
    except AB.AdapterRefusal as exc:
        rep.update(discover_refused=str(exc)[:400], seconds=round(time.time() - t0, 1))
        return rep
    rep["inputs"] = len(inputs)
    rep["input_bytes"] = sum(i.size for i in inputs)
    rep["discover_seconds"] = round(time.time() - t0, 1)
    groups = collections.Counter(adapter.comparison_group(i) for i in inputs)
    rep["groups"] = len(groups)
    if limit_groups and len(groups) > limit_groups:
        keep = set(sorted(groups)[:limit_groups])
        inputs = [i for i in inputs if adapter.comparison_group(i) in keep]
        adapter.discover = lambda _inputs=inputs: list(_inputs)  # type: ignore[method-assign]
        rep["groups_limited_to"] = limit_groups
    conn = DB.connect(DB.db_path(root), readonly=True)
    t1 = time.time()
    pr = RUN.run_pass(root, [adapter], conn=conn, now=now)
    conn.close()
    rep["pass_seconds"] = round(time.time() - t1, 1)
    rep["groups_refreshed"] = pr.groups_refreshed
    rep["groups_refused"] = pr.groups_refused
    rep["records_written"] = pr.records_written
    reasons: collections.Counter = collections.Counter()
    examples: Dict[str, str] = {}
    for r in pr.results:
        if r.error:
            key = r.error.split(":", 1)[-1].strip()[:110]
            reasons[key] += 1
            examples.setdefault(key, r.group[:160])
    rep["refusal_reasons"] = [{"reason": k, "n": v, "example_group": examples[k]} for k, v in reasons.most_common(25)]
    t2 = time.time()
    ing = H.ingest_once(root, workers=0, min_free=0)
    rep["ingest_seconds"] = round(time.time() - t2, 1)
    rep["ingest"] = {"files": ing.files, "records": ing.records, "ok": ing.ok, "refused": ing.refused, "blocked": ing.blocked}
    rep["ingest_refusals_head"] = [str(r)[:240] for r in ing.refusals[:10]]
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="scratch registry root (created)")
    ap.add_argument("--seeds-worktree", default=DEFAULT_SEEDS_WT)
    ap.add_argument("--adapters", default="cellstore,overrefusal_items,sampled_static,sampled_controls,regrade_twin")
    ap.add_argument("--hash-caches", action="store_true", help="sha-hash the sampled judge caches / twin originals (the adapters' default; slow)")
    ap.add_argument("--limit-groups", type=int, default=None, help="read at most N comparison groups per adapter (a bounded rehearsal)")
    ap.add_argument("--skip-dimensions", action="store_true", help="the root already holds the dimension rows")
    ap.add_argument("--objects-worktree", default=os.path.join(MAIN, "worktrees", "registry-adapters-objects"),
                    help="PR-3d worktree whose `registry objects scan --emit` seeds the object rows ('' to skip)")
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args()
    AB.install_store_audit_from_env()           # REGISTRY_STORE_AUDIT=<log>: record every store access (the refusal tests read it)
    root = os.path.abspath(args.root)
    H.ensure_layout(root)
    report: Dict[str, object] = {"root": root, "hash_caches": bool(args.hash_caches), "adapters": []}
    from sisteer.registry import records as R
    now = R.utc_now()
    if not args.skip_dimensions:
        report["dimensions"] = load_dimensions(root, args.seeds_worktree)
        print("dimensions:", json.dumps(report["dimensions"], indent=None)[:1500], flush=True)
        if args.objects_worktree and os.path.isdir(args.objects_worktree):
            report["objects"] = load_objects(root, args.objects_worktree, args.seeds_worktree)
            print("objects:", json.dumps(report["objects"], indent=None, default=str)[:1500], flush=True)
        report["seeded_objects"] = seed_none_and_prompted_objects(root, now)
        print("none/prompted objects:", json.dumps(report["seeded_objects"], indent=None)[:800], flush=True)
    from sisteer.registry.adapters import cellstore as CS
    from sisteer.registry.adapters import overrefusal_items as OR
    from sisteer.registry.adapters import regrade_twin as RT
    from sisteer.registry.adapters import sampled_controls as SC
    from sisteer.registry.adapters import sampled_static as SS
    factories = {
        "cellstore": lambda: CS.CellStoreAdapter(),
        "overrefusal_items": lambda: OR.OverrefusalItemsAdapter(),
        "sampled_static": lambda: SS.SampledStaticAdapter(hash_caches=args.hash_caches),
        "sampled_controls": lambda: SC.SampledControlsAdapter(),
        "regrade_twin": lambda: RT.RegradeTwinAdapter(hash_originals=args.hash_caches),
    }
    conn = DB.connect(DB.db_path(root), readonly=True)
    before = table_counts(conn)
    conn.close()
    for name in [a.strip() for a in args.adapters.split(",") if a.strip()]:
        print(f"== {name}", flush=True)
        try:
            rep = run_adapter(root, name, factories[name](), now, args.limit_groups)
        except AB.GradingHostStoreReadRefused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 5
        conn = DB.connect(DB.db_path(root), readonly=True)
        after = table_counts(conn)
        conn.close()
        rep["table_delta"] = {t: after[t] - before[t] for t in ("run", "judgment", "run_metric", "run_contrast", "item_result", "item_flags", "run_degeneration", "artifact")}
        before = after
        report["adapters"].append(rep)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("ingest_refusals_head",)}, indent=None, default=str)[:3000], flush=True)
        if rep.get("ingest_refusals_head"):
            print("  ingest refusals:", *rep["ingest_refusals_head"][:5], sep="\n    ")
    conn = DB.connect(DB.db_path(root), readonly=True)
    report["totals"] = table_counts(conn)
    report["side_by_side"] = side_by_side(conn)
    conn.close()
    print("== totals", json.dumps(report["totals"], indent=1, default=str)[:4000])
    print("== side by side", json.dumps(report["side_by_side"], indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        print("report:", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
