#!/usr/bin/env python
"""misalign_cells.py — the misalign runner's ``cells.json`` (+ ``cells_full.json``) from the standup's OWN enumeration.

    misalign_cells.py --cells-served <cells_served.json> --endpoint <endpoint.json> --out <cells.json> --full <cells_full.json>
                      --target <T> --wave-sha <SHA> --model-key <K> --wave-view <V> [--organisms a,b] [--config <misalign config>]
                      [--measure-none 0|1] [--served-cluster C] [--shard N] [--slurm-job-id ID]
                      [--baseline-pins <pins.json> --baseline-family <fam>]

The body misalign_proxy.sbatch used to carry inline (em-misalign WP-C), lifted here so the SERVED != MEASURED rule
is one testable function:

* SERVED legs = ``cells_served.json``'s ``canonical`` + ``controls`` (the derived ``none__lens0`` / ``none__lora0`` twin),
  every one held to ``endpoint.json``'s ``served_models`` — a leg the proxy does not serve refuses (never a 404 read as a
  refusal downstream).
* MEASURED legs = the served legs under the policy. ``measure_none: true`` (the default; every panel to date, byte for byte):
  every served leg is measured, the ``none`` entry included — a fresh unsteered MEASUREMENT. ``measure_none: false`` (a
  ``LANE_POLICY`` config's ``misalign_policy`` block): the ``none`` leg (class ``base``) and every controls leg (class
  ``control``) are SERVED — the standup's health checks and steering-bites run on the bare route — and never measured:
  no ``none`` row in ``cells.json``, so the runner produces no ``none`` output dir; the pairing anchor is the campaign's
  existing main-view ``none`` store. When the policy also names ``measured_cells``, the measured legs
  must translate to EXACTLY that set (registry_adapter.traintest_cell_id) — a silent subset or superset refuses.
* ``cells_full.json`` records EVERY served leg (``legs``) with ``measured: true|false`` per leg and the policy under
  ``measured_policy``, and ``cells.json``'s ``serving`` block carries the same policy, so a reader of either record sees
  what was served, what was measured, and why.

Organism overlays (em-misalign) and the baselines' pins ride exactly as before.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_SRC = Path(__file__).resolve().parents[4] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: the cell classes a `measure_none: false` policy leaves SERVED but UNMEASURED
UNMEASURED_CLASSES_WHEN_NONE_OUT: Tuple[str, ...] = ("base", "control")
CONTROL_SUFFIXES: Tuple[str, ...] = ("__lens0", "__lora0")
BASELINE_PIN_KEYS: Tuple[str, ...] = ("rung", "rung_i", "lambda", "alpha", "object_sha256", "selection_outcome", "contamination", "source")


class MisalignCellsError(SystemExit):
    pass


def _leg_class(c: Mapping[str, Any], *, is_control: bool) -> str:
    cc = c.get("cell_class")
    if isinstance(cc, str) and cc:
        return cc
    cid = str(c.get("cell_id", ""))
    if is_control or cid.endswith(CONTROL_SUFFIXES):
        return "control"
    if cid == "none":
        return "base"
    if cid.startswith("prompted__"):
        return "prompted"
    return "object"


def read_policy(config_path: Optional[Path]) -> Dict[str, Any]:
    """The config's ``misalign_policy`` block (a LANE_POLICY config), or the default policy (measure_none true, no
    measured_cells constraint). A block with a non-boolean ``measure_none`` or a non-list ``measured_cells`` refuses."""
    default = {"measure_none": True, "measured_cells": None, "lane": None, "source": "default (no misalign_policy block)"}
    if config_path is None:
        return default
    import yaml
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise MisalignCellsError(f"cannot read the misalign config {config_path}: {e}")
    blk = cfg.get("misalign_policy")
    if blk is None:
        return {**default, "source": f"default ({Path(config_path).name} carries no misalign_policy block)"}
    if not isinstance(blk, Mapping) or not isinstance(blk.get("measure_none"), bool):
        raise MisalignCellsError(f"{config_path}: misalign_policy must be a mapping with a boolean measure_none, got {blk!r}")
    mc = blk.get("measured_cells")
    if mc is not None and (not isinstance(mc, list) or not all(isinstance(c, str) and c for c in mc)):
        raise MisalignCellsError(f"{config_path}: misalign_policy.measured_cells must be a list of cell ids, got {mc!r}")
    return {"measure_none": bool(blk["measure_none"]), "measured_cells": (list(mc) if mc is not None else None),
            "lane": blk.get("lane"), "pairing_anchor": blk.get("pairing_anchor"),
            "source": f"{Path(config_path).name} misalign_policy"}


def build_cells(cells_served: Mapping[str, Any], endpoint: Mapping[str, Any], *, target: str, wave_sha: str,
                model_key: str, wave_view: str, organisms: Sequence[str] = (), policy: Optional[Mapping[str, Any]] = None,
                served_cluster: Optional[str] = None, shard: Optional[str] = None, slurm_job_id: Optional[str] = None,
                baseline_pins: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(cells.json document, cells_full.json document)`` — see the module docstring."""
    pol = dict(policy or {"measure_none": True, "measured_cells": None, "source": "default"})
    measure_none = bool(pol.get("measure_none", True))
    measured_cells = pol.get("measured_cells")
    served = set(endpoint.get("served_models") or [])
    if not served:
        raise MisalignCellsError("endpoint.json names no served_models — nothing to hold the legs to")
    canonical = list(cells_served.get("canonical") or [])
    controls = list(cells_served.get("controls") or [])
    legs = [(c, False) for c in canonical] + [(c, True) for c in controls]
    if not legs:
        raise MisalignCellsError("cells_served.json enumerates no legs")
    rows: List[Dict[str, Any]] = []
    full: List[Dict[str, Any]] = []
    unmeasured: List[str] = []
    for c, is_control in legs:
        vm = c["virtual_model_name"]
        if vm not in served:
            raise MisalignCellsError(f"cell {c['cell_id']} vmodel {vm} is not served ({sorted(served)})")
        cls = _leg_class(c, is_control=is_control)
        measured = True
        if not measure_none and cls in UNMEASURED_CLASSES_WHEN_NONE_OUT:
            measured = False
        # si_id ONLY on prompted cells (the runner sends the SI bytes as the first system message on exactly these routes)
        si = c.get("si_id") if c.get("cell_class") == "prompted" else None
        row = {"cell_id": c["cell_id"], "virtual_model": vm, "serving_identity": c["serving_identity"],
               "intervention_hash": c["intervention_hash"], "si_id": si, "kind": c["kind"]}
        full.append({**c, "si_id_for_runner": si, "measured": measured})
        if measured:
            rows.append(row)
        else:
            unmeasured.append(c["cell_id"])
        for org in organisms:
            ov = f"{vm}@{org}"
            if ov not in served:
                raise MisalignCellsError(f"overlay {ov} is not served by the proxy (organisms declared: {list(organisms)})")
            orow = {"cell_id": f"{org}::{c['cell_id']}", "virtual_model": ov, "serving_identity": c["serving_identity"],
                    "intervention_hash": c["intervention_hash"], "si_id": si, "kind": "organism"}
            full.append({**c, "cell_id": f"{org}::{c['cell_id']}", "virtual_model_name": ov, "organism": org,
                         "si_id_for_runner": si, "measured": measured})
            if measured:
                rows.append(orow)
            else:
                unmeasured.append(orow["cell_id"])
    if not rows:
        raise MisalignCellsError(f"the policy leaves no measured leg (served: {[c['cell_id'] for c, _ in legs]}; "
                                 f"measure_none={measure_none})")
    if measured_cells is not None:
        from sisteer.capeval.petri import registry_adapter as ra
        got = sorted(ra.traintest_cell_id(r["cell_id"]) for r in rows if "::" not in r["cell_id"])
        want = sorted(measured_cells)
        if got != want:
            raise MisalignCellsError(f"the measured legs {got} != the policy's measured_cells {want} — "
                                     "a silent subset or superset is refused")
    bl_pins = None
    if baseline_pins is not None:
        bl_pins = {c: {k: baseline_pins[c][k] for k in BASELINE_PIN_KEYS if k in baseline_pins[c]}
                   for c in ("arditi", "alphasteer") if c in baseline_pins}
        for r in full:
            if r["cell_id"] in bl_pins:
                r["baseline_pin"] = bl_pins[r["cell_id"]]
    measured_policy = {"measure_none": measure_none, "measured_cells_declared": measured_cells,
                       "lane": pol.get("lane"), "pairing_anchor": pol.get("pairing_anchor"), "source": pol.get("source"),
                       "served_legs": [c["cell_id"] for c, _ in legs], "measured_legs": [r["cell_id"] for r in rows],
                       "unmeasured_served": unmeasured}
    cells_doc = {"cells": rows,
                 "serving": {"substrate": "slurm", "slurm_job_id": slurm_job_id,
                             "baseline_pins": bl_pins,   # None outside the `_bl` panels
                             # the CLUSTER that served the rows (cluster_a|cluster_b) — node names repeat across clusters
                             "served_cluster": served_cluster,
                             "shard": shard,
                             "attested_max_num_seqs": endpoint.get("attested_max_num_seqs"), "attested_max_model_len": endpoint.get("attested_max_model_len"),
                             "max_num_seqs": endpoint.get("max_num_seqs"), "max_model_len": endpoint.get("max_model_len"), "node": endpoint.get("node"),
                             # SERVED != MEASURED: what was served, what was measured, and the policy that decided it
                             "measured_policy": measured_policy,
                             # the server-default sampling policy the standup composed under (None on a pre-key standup)
                             "server_default_policy": endpoint.get("server_default_policy"),
                             "override_generation_config": endpoint.get("override_generation_config")}}
    full_doc = {"target": target, "model_key": model_key, "wave_view": wave_view, "wave_sha": wave_sha,
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": {k: endpoint.get(k) for k in ("model", "model_revision", "model_key", "proxy_url", "max_model_len", "max_num_seqs",
                                                          "attested_max_model_len", "attested_max_num_seqs", "lens_reduction_mode",
                                                          "thinking_mode", "node", "slurm_job_id", "served_models", "cells", "harness_cells",
                                                          "bank_objects", "mrope_attestation", "geometry", "server_default_policy",
                                                          "override_generation_config")},
                "cells_served": {k: cells_served.get(k) for k in ("manifest", "config", "defenses_dir", "reduction_mode", "chat_template",
                                                                  "server_default_policy")},
                "measured_policy": measured_policy,
                "legs": full}
    return cells_doc, full_doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cells-served", type=Path, required=True)
    ap.add_argument("--endpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="cells.json (the runner's contract)")
    ap.add_argument("--full", type=Path, required=True, help="cells_full.json (every served leg + the policy)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--wave-sha", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--wave-view", required=True)
    ap.add_argument("--organisms", default="", help="comma list of organism names (em-misalign D26)")
    ap.add_argument("--config", type=Path, default=None, help="the misalign config; its misalign_policy block (if any) is the policy")
    ap.add_argument("--measure-none", choices=("0", "1"), default=None,
                    help="override the policy's measure_none (1 = measure the served `none`; 0 = serve it, never measure it)")
    ap.add_argument("--served-cluster", default=os.environ.get("F22_SERVED_CLUSTER"))
    ap.add_argument("--shard", default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    ap.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID"))
    ap.add_argument("--baseline-pins", type=Path, default=(Path(os.environ["F22_BASELINE_PINS"]) if os.environ.get("F22_BASELINE_PINS") else None))
    ap.add_argument("--baseline-family", default=os.environ.get("F22_BASELINE_FAMILY"))
    a = ap.parse_args(argv)
    try:
        cs = json.loads(a.cells_served.read_text(encoding="utf-8"))
        ep = json.loads(a.endpoint.read_text(encoding="utf-8"))
        policy = read_policy(a.config)
        if a.measure_none is not None:
            policy = {**policy, "measure_none": a.measure_none == "1", "source": f"{policy['source']} + --measure-none {a.measure_none}"}
        bl = None
        if a.baseline_pins is not None:
            if not a.baseline_family:
                raise MisalignCellsError("--baseline-pins needs --baseline-family (F22_BASELINE_FAMILY)")
            bl = json.loads(a.baseline_pins.read_text(encoding="utf-8"))[a.baseline_family]
        organisms = [o for o in a.organisms.split(",") if o]
        cells_doc, full_doc = build_cells(cs, ep, target=a.target, wave_sha=a.wave_sha, model_key=a.model_key,
                                          wave_view=a.wave_view, organisms=organisms, policy=policy,
                                          served_cluster=a.served_cluster, shard=a.shard, slurm_job_id=a.slurm_job_id,
                                          baseline_pins=bl)
    except MisalignCellsError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    a.out.write_text(json.dumps(cells_doc, indent=2), encoding="utf-8")
    a.full.write_text(json.dumps(full_doc, indent=2, default=str), encoding="utf-8")
    mp = cells_doc["serving"]["measured_policy"]
    print(f"cells.json: {len(cells_doc['cells'])} measured legs -> " + ", ".join(f"{r['cell_id']}={r['virtual_model']}" for r in cells_doc["cells"])
          + (f"; SERVED, NOT MEASURED (measure_none false, {mp['source']}): {mp['unmeasured_served']}" if mp["unmeasured_served"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
