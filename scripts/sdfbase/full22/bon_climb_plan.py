#!/usr/bin/env python
"""Plan a lane's canonical best-of-N: which cells CLIMB their 512 store to 4,096 and which draw
NATIVELY at 4,096 (sisteer.eval.attacks.bon_climb), and write the climb manifest for the first kind.

The canonical best-of-N is the 512 store PLUS the 4,096 climb of its cap-coherent at-cap draws
 — and a NATIVE 4,096 pass for a cell that has no 512 store (the Llama
lane). This tool makes that branch explicit BEFORE any GPU time: it walks the cells of a config,
reads each cell's store dir, and emits

  <out>/plan.json               {cell_id: {mode, cap, store_dir, n_rows, n_at_cap, n_loop, n_climb}}
  <out>/climb_manifest.jsonl    one row per AT-CAP 512 draw of every CLIMB cell, classified
                                (`climb` true = cap-coherent = regenerated; loops are never climbed)

Both are inputs to bon_climb_run.py. A cell with no 512 store appears in plan.json with mode
`native_4096` and NO manifest rows — never as a climb cell with zero rows, which would report
"no escalations" as if the cell were clean.

  python scripts/sdfbase/full22/bon_climb_plan.py --config <cfg> --private-root <root> --out <dir> \
      [--lane <name>] [--cells a,b]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--private-root", required=True, help="the wave's private transcripts root")
    ap.add_argument("--out", required=True, help="where plan.json + climb_manifest.jsonl are written")
    ap.add_argument("--lane", default=None, help="the lane name recorded on every row (default: the config's tag)")
    ap.add_argument("--job-id", default=None, help="the array job this plan classifies the store for (the shards wait for their own job's plan)")
    ap.add_argument("--cells", default=None, help="comma list (default: the config's explicit_cells)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import jailbreak_run as jr
    import two_si_run as tsr   # the store job's own goal resolver: the plan's universe is goals x n_max
    from sisteer.eval.attacks import bon_climb as bc

    cfg = jr._load_cfg(args.config)
    cells = ([c for c in args.cells.split(",") if c] if args.cells
             else list(jr._valid_defense_ids(cfg)))
    lane = args.lane or Path(args.config).stem.replace("sdfbase_full22_", "")
    root = jr._private_root(cfg, args.private_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    plan, manifest = {}, []
    for cell in sorted(cells):
        store = Path(jr._bon_defense_dir(cfg, root, cell))
        try:
            mode = bc.climb_mode(store)                    # complete -> climb; absent -> native_4096; IN PROGRESS -> refused by name
        except bc.BonClimbError as exc:
            raise SystemExit(f"[bon-climb-plan] {lane}/{cell}: {exc}")
        # THE SAMPLER the climb re-draws at is the STORE'S (bc.CLIMB_DECODE `as_drawn`; user ruling 2026-09-17):
        # recorded per cell here from the store's own rows (None = a legacy store, drawn at the grandfathered T=1
        # row before the record existed; the driver resolves that row by name), and per row on the manifest
        row = {"cell_id": cell, "store_dir": str(store), "mode": mode, "cap": bc.CLIMB_CAP,
               "decode": bc.CLIMB_DECODE,
               "store_sampler": bc.store_sampler(store) if mode == bc.MODE_CLIMB else None}
        if mode == bc.MODE_CLIMB:
            # the EXACT completeness over the plan's own universe (the config's goals x n_max, trial 0 — what the store job planned): every
            # slot covered or a PROVEN no-fit, else refused by name; a foreign key (another writer's row) never counts toward the plan
            # the store's own goal universe: the writer records `confirmatory` on bon_served_run.json (the wave shell runs it with
            # --confirmatory); a record without the key predates that field and is read as the wave's convention (confirmatory)
            _run = json.load(open(store / bc.BON_STORE_MANIFEST)) if (store / bc.BON_STORE_MANIFEST).is_file() else {}
            _conf = _run.get("confirmatory"); _conf = True if _conf is None else bool(_conf)
            goals = tsr._eval_goals(cfg, confirmatory=_conf, attack="bon"); n_max = int(cfg["bon"]["n_max"])
            gaps = bc.store_planned_gaps(store, [(g.goal_id, i, 0) for g in goals for i in range(n_max)])
            if gaps["gaps"]:
                kinds = {}
                for g in gaps["gaps"]:
                    kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
                raise SystemExit(f"[bon-climb-plan] {lane}/{cell}: {len(gaps['gaps'])} of {gaps['n_planned']} planned slot(s) are neither covered nor a proven "
                                 f"no-fit ({kinds}) — the 512 store is not complete for this plan; re-run the store job (never climb over undecided slots)")
            row.update(planned_check={k: v for k, v in gaps.items() if k != "gaps"})
            # the store's DECIDED no-fit slots ride the plan by name (the coverage table reads them as decided, not as missing draws) —
            # the PLANNED ones; a foreign key (another writer's row) is listed in planned_check.foreign_keys, never as a no-fit of this plan
            _planned = {(g.goal_id, i, 0) for g in goals for i in range(n_max)}
            nofit = [x for x in bc.store_nofit_slots(store) if (x["goal_id"], x["sample_idx"], x["trial"]) in _planned]
            row.update(n_nofit=len(nofit), nofit_slots=nofit, store_units=bc.store_units(store))
            rows = bc.climb_manifest_rows(store, lane=lane, cell_id=cell)
            climb = [r for r in rows if r["climb"]]
            n_all = sum(1 for l in (store / bc.BON_RESPONSES).read_text().splitlines() if l.strip())
            row.update(n_rows=n_all, n_at_cap=len(rows), n_loop=len(rows) - len(climb), n_climb=len(climb))
            manifest.extend(rows)
        else:
            row.update(n_rows=0, n_at_cap=None, n_loop=None, n_climb=None,
                       note=f"no {bc.BASE_CAP} store: best-of-N is drawn NATIVELY at {bc.CLIMB_CAP}")
        plan[cell] = row
        print(f"[bon-climb-plan] {lane}/{cell}: {mode}"
              + (f" rows={row['n_rows']} at_cap={row['n_at_cap']} loops(not climbed)={row['n_loop']} "
                 f"climb={row['n_climb']}" if mode == bc.MODE_CLIMB else " (nothing to climb — native pass)"),
              flush=True)
    doc = {"schema": "bon_climb_plan/1", "lane": lane, "config": args.config,
           "private_root": str(root), "cap": bc.CLIMB_CAP, "base_cap": bc.BASE_CAP,
           "decode": bc.CLIMB_DECODE,
           "decode_rule": ("as_drawn: each climb cell re-draws its at-cap slots at the sampler its 512 store "
                           "records (cells[*].store_sampler; a legacy store: the grandfathered T=1 row by name); "
                           "a native cell draws at the served model's vendor preset (user ruling 2026-09-17)"),
           "classifier_fingerprint": bc.classifier_fingerprint(),
           "loop_rule": ("LOOP = 4-gram rep_ratio over the text OR its last 200 words > 0.5 "
                         "(cap_ladder); a loop is NEVER climbed — a repeating draw repeats further — "
                         "and is an accepted residual-floor component. The secondary degeneracy "
                         "classifier is a FLAG only: such a draw IS climbed."),
           "by_mode": bc.lanes_by_mode(list(plan.values())), "cells": plan,
           # the attempt this plan classified the store for (the sbatch's shards wait for THEIR job's plan; a plan of another attempt is stale)
           "job_id": args.job_id or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
           "planned_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    (out / "plan.json").write_text(json.dumps(doc, indent=1))
    with (out / "climb_manifest.jsonl").open("w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    n_native = len(doc["by_mode"][bc.MODE_NATIVE])
    print(f"[bon-climb-plan] {len(plan)} cells: {len(doc['by_mode'][bc.MODE_CLIMB])} climb, {n_native} native@{bc.CLIMB_CAP}; "
          f"{sum(1 for r in manifest if r['climb'])} draws to climb -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
