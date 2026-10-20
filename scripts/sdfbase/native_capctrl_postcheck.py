#!/usr/bin/env python
"""The DRIVER post-check of ``native_capctrl_smoke.sbatch``.

The sbatch runs ``served_capability_run.py generate --controls mmlu`` against the native capctrl standup and then
judges the store it wrote. The check lives here, as a module with a CLI, so it is validated ON CPU against a store the
REAL driver writes (``tests/test_native_capctrl_postcheck.py`` runs the driver over the native test world and hands
this module its store) before the orchestrator spends a GPU allocation on it. The inline check it replaces required
``manifest.route.serving_engine`` / ``manifest.route.reduction_mode`` — but the legacy MMLU control produces no
tinycap atoms, so ``_tinycap_manifest()`` (which creates ``route``) never runs, and a CORRECT run would have failed.

What an MMLU-only run on a native standup actually writes, and where (the authoritative locations):

- ``<store>/cells/<key>/manifest.json`` (``served_capability_run`` ``manifest``): the served ENGINE at the TOP LEVEL
  (``serving_engine``, unhashed, present iff native) and in the HASHED identity
  (``resolved_config.serving.serving_engine``), ``serving_identity`` a ``+lora+`` route with ``serving_path == "lora"``,
  ``steer_positions == "all"`` (a native adapter applies at every token), ``resolved_config.mmlu.response_prefill is
  False`` (the lens re-location declaration is never sent on a lora route), ``row_constants`` without an engine column
  (``store.ResultRow`` has none; rows are keyed by engine through ``harness_config_hash``). A ``route`` block appears
  only on a tinycap manifest (atoms) — checked when present, never required.
- ``<store>/evidence.json``: the steered route's bites proof against the engine's zero cell (``steering_bites``:
  applicable, verified, twin ``none__lora0``), the MMLU decision block (``mmlu_scoring``: lora route, declaration not
  sent, "adapter, all positions") with the logprob bites beside it, and the engine stamp's serving path.
- the STANDUP's ``endpoint.json`` (``--endpoint``): ``engine`` / ``serving_engine`` ``lora_native``, ``reduction_mode``
  ``not_applicable`` (an explicit value; a native record states no ``lens_reduction_mode``), and the path of its
  positive-controls record (``native_standup.run_positive_controls``: ``passed`` true).

Exit 0 with ``{"problems": [], "facts": {...}}`` on stdout when the store is what a correct native MMLU run writes;
exit 1 with every problem named otherwise. Read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path):
    """The parsed document, or the exception (a caller names the file in its problem line)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:  # noqa: PERF203 — one file, one verdict
        return e


#: the MMLU decision block the driver records on a lora route (``served_capability_run``: ``evidence["mmlu_scoring"]``)
MMLU_SCORING_LORA = {"route_path": "lora", "declaration_sent": False,
                     "steering_at_scored_position": "adapter, all positions"}


def driver_store_problems(store: Path, *, endpoint: Optional[Dict[str, Any]], cell: Optional[str]
                          ) -> Dict[str, Any]:
    """``{"problems": [...], "facts": {...}}`` for the driver's ``store`` (its ``--out-root``), the standup's
    ``endpoint`` record (parsed; None = not checked) and the served TrainTest ``cell`` id (None = not checked)."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.tinycap import contracts as tc

    native = tc.SERVING_ENGINE_LORA_NATIVE
    problems: List[str] = []
    facts: Dict[str, Any] = {}
    store = Path(store)

    # --- the cell manifests ---------------------------------------------------------------------------------------
    mans = sorted(store.rglob("manifest.json"))
    facts["n_manifests"] = len(mans)
    if not mans:
        problems.append(f"{store}: the driver wrote no manifest.json")
    for m in mans:
        d = _load_json(m)
        if isinstance(d, Exception):
            problems.append(f"{m}: unreadable ({d})")
            continue
        rc = d.get("resolved_config") or {}
        serving = rc.get("serving") or {}
        ident = d.get("serving_identity")
        path = None
        if ident:
            try:
                path = cm.parse_serving_identity(ident).path
            except Exception as e:  # noqa: BLE001 — named below
                problems.append(f"{m}: serving_identity {ident!r} does not parse ({e})")
        route = d.get(tc.MANIFEST_KEY_ROUTE)
        facts[m.parent.name] = {
            "serving_engine": d.get(tc.SERVED_ENGINE_FIELD),
            "hashed_serving_engine": serving.get(tc.SERVED_ENGINE_FIELD),
            "serving_identity": ident, "serving_path": d.get("serving_path"),
            "steer_positions": d.get("steer_positions"), "cell_id": d.get("cell_id"),
            "mmlu": rc.get("mmlu"), "route": route,
        }
        if d.get(tc.SERVED_ENGINE_FIELD) != native:
            problems.append(f"{m}: top-level {tc.SERVED_ENGINE_FIELD}={d.get(tc.SERVED_ENGINE_FIELD)!r} (a native "
                            "manifest names the engine top-level, unhashed; absent = lens by era)")
        if serving.get(tc.SERVED_ENGINE_FIELD) != native:
            problems.append(f"{m}: resolved_config.serving.{tc.SERVED_ENGINE_FIELD}="
                            f"{serving.get(tc.SERVED_ENGINE_FIELD)!r} (the HASHED copy; absent = a lens-keyed store)")
        if serving.get("serving_identity") != ident:
            problems.append(f"{m}: resolved_config.serving.serving_identity {serving.get('serving_identity')!r} != "
                            f"the manifest's {ident!r}")
        if cell is not None and d.get("cell_id") != cell:
            problems.append(f"{m}: cell_id={d.get('cell_id')!r} != the served cell {cell!r}")
        if path != "lora":
            problems.append(f"{m}: serving_identity {ident!r} is not a +lora+ route (the LoRA cell must be served "
                            "natively, never as a lens or passthrough leg)")
        if d.get("serving_path") != "lora":
            problems.append(f"{m}: serving_path={d.get('serving_path')!r} != 'lora'")
        if d.get("steer_positions") != "all":
            problems.append(f"{m}: steer_positions={d.get('steer_positions')!r} (a native adapter applies at every "
                            "token: 'all')")
        mmlu = rc.get("mmlu")
        if isinstance(mmlu, dict) and mmlu.get("response_prefill") is not False:
            problems.append(f"{m}: resolved_config.mmlu.response_prefill={mmlu.get('response_prefill')!r} (the lens "
                            "re-location declaration is never sent on a lora route)")
        rcs = d.get("row_constants") or {}
        if rcs.get("harness_config_hash") != d.get("harness_config_hash"):
            problems.append(f"{m}: row_constants.harness_config_hash {rcs.get('harness_config_hash')!r} != the "
                            f"manifest's {d.get('harness_config_hash')!r}")
        if tc.SERVED_ENGINE_FIELD in rcs:
            problems.append(f"{m}: row_constants carries {tc.SERVED_ENGINE_FIELD} (store.ResultRow has no such column; "
                            "rows are keyed by engine through harness_config_hash)")
        if route is not None:
            # a tinycap manifest (atoms): the route block names the engine and the not_applicable reduction mode
            if route.get(tc.SERVED_ENGINE_FIELD) != native:
                problems.append(f"{m}: route.{tc.SERVED_ENGINE_FIELD}={route.get(tc.SERVED_ENGINE_FIELD)!r}")
            if route.get(tc.REDUCTION_MODE_FIELD) != tc.REDUCTION_MODE_NOT_APPLICABLE:
                problems.append(f"{m}: route.{tc.REDUCTION_MODE_FIELD}={route.get(tc.REDUCTION_MODE_FIELD)!r}")

    # --- the driver's evidence -----------------------------------------------------------------------------------
    ev_path = store / "evidence.json"
    if not ev_path.is_file():
        problems.append(f"{ev_path}: missing — the driver writes its steering-bites / MMLU evidence here")
    else:
        ev = _load_json(ev_path)
        if isinstance(ev, Exception):
            problems.append(f"{ev_path}: unreadable ({ev})")
        else:
            bites = ev.get("steering_bites") or {}
            lp = ev.get("logprob_steering_bites") or {}
            stamp = ev.get("engine_stamp") or {}
            facts["evidence"] = {
                "serving_identity": ev.get("serving_identity"),
                "steering_bites": {k: bites.get(k) for k in ("applicable", "verified", "twin", "n_differing")},
                "logprob_steering_bites": {k: lp.get(k) for k in ("applicable", "verified", "twin")},
                "mmlu_scoring": ev.get("mmlu_scoring"), "engine_stamp_serving_path": stamp.get("serving_path"),
            }
            if not (bites.get("applicable") is True and bites.get("verified") is True):
                problems.append(f"{ev_path}: steering_bites applicable={bites.get('applicable')!r} "
                                f"verified={bites.get('verified')!r} (the steered route owes a verified bites proof)")
            if bites.get("twin") != tc.TWIN_CELL_ID_LORA:
                problems.append(f"{ev_path}: steering_bites.twin={bites.get('twin')!r} != {tc.TWIN_CELL_ID_LORA!r} "
                                "(the engine's own zero cell, never the lens twin)")
            if stamp.get("serving_path") != "lora":
                problems.append(f"{ev_path}: engine_stamp.serving_path={stamp.get('serving_path')!r} != 'lora'")
            if ev.get("mmlu_scoring") is not None:   # the MMLU control ran
                if ev.get("mmlu_scoring") != MMLU_SCORING_LORA:
                    problems.append(f"{ev_path}: mmlu_scoring={ev.get('mmlu_scoring')!r} != {MMLU_SCORING_LORA!r}")
                if not (lp.get("applicable") and lp.get("verified") is True and lp.get("twin") == tc.TWIN_CELL_ID_LORA):
                    problems.append(f"{ev_path}: logprob_steering_bites applicable={lp.get('applicable')!r} "
                                    f"verified={lp.get('verified')!r} twin={lp.get('twin')!r} (the loglikelihood "
                                    f"positive control against {tc.TWIN_CELL_ID_LORA})")

    # --- the standup's record --------------------------------------------------------------------------------------
    if endpoint is not None:
        pc_path = endpoint.get("positive_controls")
        facts["endpoint"] = {"engine": endpoint.get(tc.ENDPOINT_KEY_ENGINE),
                             "serving_engine": endpoint.get(tc.SERVED_ENGINE_FIELD),
                             "reduction_mode": endpoint.get(tc.REDUCTION_MODE_FIELD), "positive_controls": pc_path}
        if endpoint.get(tc.ENDPOINT_KEY_ENGINE) != native or endpoint.get(tc.SERVED_ENGINE_FIELD) != native:
            problems.append(f"endpoint.json: engine={endpoint.get(tc.ENDPOINT_KEY_ENGINE)!r} "
                            f"{tc.SERVED_ENGINE_FIELD}={endpoint.get(tc.SERVED_ENGINE_FIELD)!r}")
        if endpoint.get(tc.REDUCTION_MODE_FIELD) != tc.REDUCTION_MODE_NOT_APPLICABLE:
            problems.append(f"endpoint.json: {tc.REDUCTION_MODE_FIELD}={endpoint.get(tc.REDUCTION_MODE_FIELD)!r} != "
                            f"{tc.REDUCTION_MODE_NOT_APPLICABLE!r} (the native standup's explicit value)")
        if "lens_reduction_mode" in endpoint:
            problems.append("endpoint.json: a native record states lens_reduction_mode (two engines on one record)")
        if not pc_path:
            problems.append("endpoint.json names no positive_controls record")
        elif not Path(pc_path).is_file():
            problems.append(f"{pc_path}: the standup's positive-controls record is missing")
        else:
            rec = _load_json(Path(pc_path))
            if isinstance(rec, Exception):
                problems.append(f"{pc_path}: unreadable ({rec})")
            else:
                facts["positive_controls"] = {"passed": rec.get("passed"),
                                              "controls": {n: (c or {}).get("passed")
                                                           for n, c in (rec.get("controls") or {}).items()}}
                if rec.get("passed") is not True:
                    problems.append(f"{pc_path}: passed={rec.get('passed')!r} "
                                    f"(failed_control={rec.get('failed_control')!r})")
    return {"problems": problems, "facts": facts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True, help="the driver's --out-root")
    ap.add_argument("--endpoint", default=None, help="the standup's endpoint.json (optional)")
    ap.add_argument("--cell", default=None, help="the served TrainTest cell id (optional)")
    a = ap.parse_args(argv)
    endpoint = None
    if a.endpoint:
        endpoint = _load_json(Path(a.endpoint))
        if isinstance(endpoint, Exception):
            print(json.dumps({"problems": [f"{a.endpoint}: unreadable ({endpoint})"], "facts": {}}))
            return 1
    out = driver_store_problems(Path(a.store), endpoint=endpoint, cell=a.cell)
    print(json.dumps(out, default=str))
    return 0 if not out["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
