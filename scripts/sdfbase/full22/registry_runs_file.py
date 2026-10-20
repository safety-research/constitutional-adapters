#!/usr/bin/env python
"""full22: ``<standup dir>/registry_runs.json`` (sisteer.registry.hooks.RUNS_FILE) from the job's pending file, so the native standup's
``write_endpoint`` hook (hooks.emit_serving_patches) records the ATTESTED serving facts — context_window_tokens, served_max_num_seqs, the
engine version, the adapter fingerprint — as one ``run_patch`` per run this standup serves (the Modal lane's
seedrep_served.write_registry_runs, ported to the cluster shell: nothing on Slurm wrote the file before, so every sbatch lane's rows carry
context_window_tokens NULL).

  python scripts/sdfbase/full22/registry_runs_file.py --pending <shared>/registry/pending/<job>.json --out <standup dir> \\
      --evals strongreject_bon[,pasttense,...]

The pending file is emit_cli register's (jobplan.PlanBundle.as_pending): ``runs[]`` with ``route.eval_id`` / ``route.object`` (the CONFIG cell
entry: ``none``, ``prompted__si_constitution``, ``constv2-mix-full__lora_r4_ce``) and ``plan.campaign`` / ``plan.serving_path``. THE KEY IS
THE ENDPOINT'S OWN SPELLING (the Modal lane's rule): endpoint.json's ``adapters`` / ``served_positions`` speak HARNESS ids
(``si_constv2_mix_full__lora_r4_ce``, ``prompted__constitution``), so every route object is translated through the ONE table
(sisteer.capeval.petri.registry_adapter.harness_cell_id); a cell it cannot express REFUSES by name, never a silent rename. Only the runs of
the evals THIS standup serves are named (a static standup must not stamp its window onto the controls runs of the same job); a served key
measured by several runs (the -all job: four static families + the store on one cell) carries the LIST form hooks.load_runs_file reads.
Exit 0 with the file written (one summary line); 2 = a usage / shape problem, 3 = no run of the named evals in the pending file (nothing to
record is a defect for a registering kind, not a quiet standup).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

RUNS_FILE = "registry_runs.json"   # == sisteer.registry.hooks.RUNS_FILE (asserted at import below)
TAG = "[registry]"


class RunsFileRefused(Exception):
    pass


def runs_map(pending: Mapping[str, object], evals: List[str], *, harness_id=None) -> Dict[str, object]:
    """``{served key: run | [run, ...]}`` for the pending file's runs of ``evals`` — ``run = {run_id, campaign, expected: {serving_path}}``;
    ``harness_id`` translates a route object to the endpoint's spelling (default registry_adapter.harness_cell_id)."""
    if harness_id is None:
        from sisteer.capeval.petri import registry_adapter as ra
        harness_id = ra.harness_cell_id
    want = [e for e in evals if e]
    if not want:
        raise RunsFileRefused("--evals names no eval (the evals THIS standup serves)")
    runs = pending.get("runs") if isinstance(pending, Mapping) else None
    if not isinstance(runs, list) or not runs:
        raise RunsFileRefused("the pending file carries no `runs` list (emit_cli register writes one entry per registered (eval, object))")
    by_key: Dict[str, List[Dict[str, object]]] = {}
    seen: set = set()
    for i, r in enumerate(runs):
        route = r.get("route") if isinstance(r, Mapping) else None
        plan = r.get("plan") if isinstance(r, Mapping) else None
        if not isinstance(route, Mapping) or not isinstance(plan, Mapping) or not r.get("run_id"):
            raise RunsFileRefused(f"runs[{i}] has no route / plan / run_id (the pending file is not emit_cli register's shape)")
        eval_id, obj = route.get("eval_id"), route.get("object")
        if not eval_id or not obj:
            raise RunsFileRefused(f"runs[{i}] ({r.get('run_id')}) names no route.eval_id / route.object")
        if eval_id not in want:
            continue
        rid = str(r["run_id"])
        if rid in seen:
            continue
        seen.add(rid)
        try:
            served_key = str(harness_id(str(obj)))
        except Exception as e:  # noqa: BLE001 — a cell the harness cannot express is a refusal by name, never a silent rename
            raise RunsFileRefused(f"runs[{i}] ({rid}): route.object {obj!r} has no harness spelling ({type(e).__name__}: {e})") from None
        campaign = plan.get("campaign")
        if not campaign:
            raise RunsFileRefused(f"runs[{i}] ({rid}): plan.campaign is empty")
        entry: Dict[str, object] = {"run_id": rid, "campaign": str(campaign),
                                    "expected": {"serving_path": str(plan.get("serving_path") or "lora_native")}}
        by_key.setdefault(served_key, []).append(entry)
    if not by_key:
        raise RunsFileRefused(f"no run of evals {want} in the pending file (its runs: "
                              f"{sorted({str((r.get('route') or {}).get('eval_id')) for r in runs if isinstance(r, Mapping)})})")
    return {k: (v[0] if len(v) == 1 else v) for k, v in by_key.items()}


def write_runs_file(out_dir: Path, runs: Mapping[str, object]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / RUNS_FILE
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(runs, indent=1, sort_keys=True))
    os.replace(tmp, p)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--pending", required=True, help="the job's pending file ($SISTEER_REGISTRY_RUN_IDENTITY: <shared>/registry/pending/<job>.json)")
    ap.add_argument("--out", required=True, help="the standup's out dir (where endpoint.json lands; the hook reads registry_runs.json beside it)")
    ap.add_argument("--evals", required=True, help="comma list: the evals THIS standup serves (the static families and/or strongreject_bon)")
    args = ap.parse_args(argv)
    from sisteer.registry import hooks as H
    assert H.RUNS_FILE == RUNS_FILE, (H.RUNS_FILE, RUNS_FILE)
    try:
        try:
            pending = json.loads(Path(args.pending).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RunsFileRefused(f"cannot read the pending file {args.pending} ({type(e).__name__}: {e})") from None
        runs = runs_map(pending, [e.strip() for e in args.evals.split(",")])
    except RunsFileRefused as e:
        nothing = str(e).startswith("no run of evals")
        print(f"{TAG} served runs file {'not written' if nothing else 'REFUSED'}: {e}", file=sys.stderr, flush=True)
        return 3 if nothing else 2
    p = write_runs_file(Path(args.out), runs)
    n = sum(len(v) if isinstance(v, list) else 1 for v in runs.values())
    print(f"{TAG} served runs file: {n} run(s) under {len(runs)} served key(s) ({', '.join(sorted(runs))}) of evals {args.evals} -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
