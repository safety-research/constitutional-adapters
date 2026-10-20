#!/usr/bin/env python3
"""The Qwen seed-replicate battery — serving facts fanned out per logical run.

``hooks.emit_serving_patches`` takes a one-to-one ``{served_key: run}`` map at ``write_endpoint`` (``native_standup``),
which cannot carry several evals per served key and fires before the per-child identities exist. So this script runs
on the grading host, after the run ids exist and the accepted attempt's serve directory is pulled, and calls the REAL hook
ONCE PER LOGICAL RUN with a single-entry map keyed by the run's SERVED KEY — the route recorded in the plan
(``extra.served_key`` = the ``endpoint["adapters"]`` / ``served_positions`` key of that arm's route), never the run id:
the hook reads each key as ``served_key`` to select ``endpoint["adapters"][served_key]`` and the positions map, so a
run id as key would silently drop every fitted run's adapter fingerprint and positions facts. A map keyed by a run id
is REFUSED here before the hook is called; so is a fitted / Arditi route whose served key has no adapter block on the
endpoint (the facts would be silently absent), and a ``none`` / prompted route that DOES have one (a mis-keyed plan).

  emit_serving_facts_fanout.py --plans-dir RS/registry/plans --leg <leg> [--leg …] --view <s1|s2|s3|arditi> \\
                               --endpoint <accepted attempt>/serve/endpoint.json [--attestation …/serving_attestation.json] \\
                               --spool ~/registry/records [--dry-run]

One call covers every logical run the engine served: every plan file of the given legs on the given view (the controls
engine serves xstest + or_bench + mmlu, so pass all three legs; Petri's per-leg file already holds its six behaviour runs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path and not any(p.endswith("/src") and "sisteer" in os.listdir(p) for p in sys.path if os.path.isdir(p)):
    sys.path.insert(0, str(REPO / "src"))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sisteer.registry import hooks, identity  # noqa: E402
from sisteer.registry import records as R  # noqa: E402

import registry_plans as RP  # noqa: E402

SCRIPT = "scripts/sdfbase/seedrep/emit_serving_facts_fanout.py"
ADAPTER_ROUTE_KINDS = ("fitted", "arditi")
PASSTHROUGH_ROUTE_KINDS = ("none", "prompted")


class FanoutRefused(ValueError):
    """A served-key map that would make the hook record the wrong facts: nothing is emitted."""


def _is_run_id(key: str) -> bool:
    try:
        identity.parse_run_id(key)
        return True
    except ValueError:
        return False


def check_keyed_by_served_key(runs: Mapping[str, Mapping[str, object]]) -> None:
    """Refuse any key that is a registry run id (``<campaign>:<eval>:<inst12>:<sha16>``): the hook reads keys as served keys."""
    for key in runs:
        if _is_run_id(str(key)):
            raise FanoutRefused(f"runs map is keyed by run id {key!r}; emit_serving_patches reads keys as SERVED keys (endpoint adapters / "
                                "served_positions) — key the map by the route recorded in the plan (extra.served_key)")


def load_endpoint(endpoint_path, attestation_path=None, engine_identity: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The record the hook reads: ``endpoint.json`` + (when given) the attestation-derived engine identity — the standup
    passes ``engine_identity`` from memory (``native_standup._registry_serving_facts``), which the file does not carry."""
    with open(endpoint_path, "r", encoding="utf-8") as fh:
        ep = json.load(fh)
    if not isinstance(ep, dict):
        raise FanoutRefused(f"{endpoint_path}: not a JSON object")
    if engine_identity is not None:
        ep = dict(ep, engine_identity=dict(engine_identity))
    elif attestation_path is not None:
        with open(attestation_path, "r", encoding="utf-8") as fh:
            att = json.load(fh)
        try:
            from sisteer.capeval.serving import serving_attestation as sa
            ep = dict(ep, engine_identity=sa.engine_identity(att))
        except Exception as exc:  # noqa: BLE001 — the attestation is an optional enrichment; say what was not read
            print(f"[fanout] attestation not folded into engine_identity ({type(exc).__name__}: {exc}); serving_engine_version may be absent",
                  file=sys.stderr)
    return ep


def runs_for_engine(plan_files: Sequence[RP.PlanFile], endpoint: Mapping[str, Any]) -> List[Tuple[RP.PlanFile, str, str, Dict[str, object]]]:
    """``[(plan file, route, served_key, run)]`` for every logical run of the given plan files, validated against the endpoint."""
    adapters = endpoint.get("adapters") if isinstance(endpoint.get("adapters"), dict) else {}
    out: List[Tuple[RP.PlanFile, str, str, Dict[str, object]]] = []
    for pf in plan_files:
        for route, ident in pf.identities.items():
            ex = ident.get("extra") or {}
            served_key = str(ex.get("served_key") or "")
            kind = str(ex.get("arm_kind") or "")
            if not served_key:
                raise FanoutRefused(f"{pf.path}: route {route!r} records no extra.served_key")
            if _is_run_id(served_key):
                raise FanoutRefused(f"{pf.path}: extra.served_key {served_key!r} is a run id, not a served key")
            if kind in ADAPTER_ROUTE_KINDS and served_key not in adapters:
                raise FanoutRefused(f"{pf.path}: {kind} route {served_key!r} has no adapter block on the endpoint (adapters: {sorted(adapters)}); "
                                    "its fingerprint and positions would be silently absent")
            if kind in PASSTHROUGH_ROUTE_KINDS and served_key in adapters:
                raise FanoutRefused(f"{pf.path}: {kind} route {served_key!r} HAS an adapter block on the endpoint; a passthrough route is mis-keyed")
            run = {"run_id": str(ident["run_id"]), "campaign": str(ident["campaign"]),
                   "expected": {"serving_path": str(ident.get("serving_path") or "lora_native")}}
            out.append((pf, route, served_key, run))
    return out


def fanout(endpoint: Mapping[str, Any], runs: Sequence[Tuple[Any, str, str, Mapping[str, object]]], *, spool: Path,
           dry_run: bool = False) -> List[Dict[str, object]]:
    """One ``emit_serving_patches`` call per logical run, each with a single-entry ``{served_key: run}`` map."""
    out: List[Dict[str, object]] = []
    for pf, route, served_key, run in runs:
        single = {served_key: dict(run)}
        check_keyed_by_served_key(single)
        payload = hooks.serving_patch_payload(endpoint, served_key, run)
        entry: Dict[str, object] = {"run_id": run["run_id"], "route": route, "served_key": served_key, "plan": str(getattr(pf, "path", pf)),
                                    "fields": sorted(payload["fields"]), "verify": sorted(payload["verify"]), "paths": []}
        if not dry_run:
            paths = hooks.emit_serving_patches(endpoint, single, spool=str(spool), registry_emit=True, script=SCRIPT)
            if not paths:
                raise FanoutRefused(f"emit_serving_patches wrote nothing for run {run['run_id']} (served key {served_key!r}); see the [registry] notice")
            entry["paths"] = list(paths)
        out.append(entry)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plans-dir", required=True)
    ap.add_argument("--leg", action="append", required=True, help="repeatable; every leg the engine served (controls: xstest, or_bench, mmlu)")
    ap.add_argument("--view", required=True, help="s1 | s2 | s3 | arditi")
    ap.add_argument("--endpoint", required=True, help="the accepted attempt's serve/endpoint.json")
    ap.add_argument("--attestation", default=None, help="the accepted attempt's serve/serving_attestation.json (engine identity)")
    ap.add_argument("--engine-identity", default=None, help="JSON engine identity to fold in (overrides --attestation)")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        plan_files: List[RP.PlanFile] = []
        for leg in a.leg:
            sel = RP.plan_files_for(a.plans_dir, leg=leg, view=a.view)
            if not sel:
                raise FanoutRefused(f"no plan files for leg {leg!r} on view {a.view!r} under {a.plans_dir}")
            plan_files.extend(sel)
        ep = load_endpoint(a.endpoint, a.attestation, json.loads(a.engine_identity) if a.engine_identity else None)
        runs = runs_for_engine(plan_files, ep)
        out = fanout(ep, runs, spool=Path(a.spool), dry_run=a.dry_run)
    except (FanoutRefused, RP.RegistryPlanRefused) as exc:
        print(f"emit_serving_facts_fanout: REFUSED: {exc}", file=sys.stderr)
        return 2
    for e in out:
        print(json.dumps(e, sort_keys=True))
    print(f"emit_serving_facts_fanout: {len(out)} run patch(es) {'planned (dry run)' if a.dry_run else 'written'} for view {a.view}, legs {a.leg}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
