#!/usr/bin/env python
"""Harvest a rung's Phase E deliverables into its record: the LIVENESS quadruple, a copy of every
standup's positive controls, and the at-cap ITEM IDS.

WHY THIS EXISTS. `ar_cell.sbatch` writes `cell_facts.json` with the cell's identity (dose, unit,
phase, served layer, geometry, sampler, config/manifest shas) but has never written the three
things Phase E's selection actually requires (§4.2, §8.2):

  * `logprob_liveness.{verified, n_probes, n_conclusive, divergence.steered_vs_scale0_median}` —
    a rung without it is UNSELECTABLE, because a dose admissible merely because it is INERT must
    never be selected;
  * a durable copy of EVERY standup's `positive_controls.json` — the successful one, the 8192
    re-draw's, and any FAILED attempt's, which is exactly when liveness matters most (the file is
    written before the failure path tears the server down);
  * the at-cap ITEM IDS, never a share: attribution is by item id intersected with the safe
    subset, and apportioning by a share against the wrong denominator produced a spurious result
    once already.

So the harvest ran nowhere, and the first rung finished without them. This is that step, written
ONCE and used two ways: `ar_cell.sbatch`'s record leg calls it in-job for every future rung, and it
runs post-hoc over a finished store for a rung that predates the wiring. Same code path, so every
rung's record has identical structure.

BINDS TO THE STORED SCHEMA, not to a paraphrase. The quadruple lives at
`controls.steering_bites.cells[<cell>].logprob_liveness` with keys `applicable, verified, kind,
twin, n_probes, n_conclusive, min_conclusive_probes, divergence.{steered_vs_scale0_median,
clears_bar}`; the at-cap ids live UNDER the control's own key in `gen_stats.json`
(`gen_stats["xstest"]["at_cap_item_ids"]`), not at the top level — reading the top level is how
they get mistaken for absent.

NEVER OVERWRITES. Existing `cell_facts.json` keys are left exactly as they are unless `--force`;
a liveness copy whose name is taken gets the next attempt index. A rung with no readable liveness
records the REASON in `cell_facts.liveness.error` rather than omitting the key, because a silently
missing deliverable reads as "not measured" when it may mean "not harvested".

    python scripts/arditi/harvest_rung_facts.py --rec-dir <cells/<TAG>_t<T>> \\
        --serve-root <serving/> --controls-root <controls/<TAG>_t<T>> --cell arditi_allpos
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

#: the stored path to the block (verified against a produced file)
LIVENESS_PATH = ("controls", "steering_bites", "cells")
#: the four fields Phase E reads, plus the bar the verdict is taken against
QUADRUPLE = ("verified", "n_probes", "n_conclusive")
DIVERGENCE_KEY = "steered_vs_scale0_median"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def liveness_block(pc: dict, cell: str) -> dict | None:
    """`controls.steering_bites.cells[<cell>].logprob_liveness`, or None."""
    node = pc
    for k in LIVENESS_PATH:
        node = (node or {}).get(k) or {}
    return ((node.get(cell) or {}).get("logprob_liveness")) or None


def quadruple_of(block: dict) -> dict:
    div = block.get("divergence") or {}
    out = {k: block.get(k) for k in QUADRUPLE}
    out["divergence.steered_vs_scale0_median"] = div.get(DIVERGENCE_KEY)
    # the bar the verdict was taken against, so a reader need not guess what "conclusive" meant
    for extra in ("applicable", "kind", "twin", "min_conclusive_probes", "statistic"):
        if extra in block:
            out[extra] = block[extra]
    if isinstance(div, dict) and "clears_bar" in div:
        out["divergence.clears_bar"] = div["clears_bar"]
    return out


def find_standups(serve_root: Path, rec_dir: Path) -> list[tuple[str, Path]]:
    """Every standup's positive_controls.json for THIS rung, as (phase, path).

    The phase is the standup directory's own name (`cell` = standup 1, `cell_rung8192` = the
    8192 re-draw), so two standups of one job never collide."""
    out = []
    for pc in sorted(serve_root.rglob("positive_controls.json")):
        out.append((pc.parent.name, pc))
    return out


def harvest(rec_dir: Path, serve_root: Path, controls_root: Path, cell: str, jobid: str,
            *, force: bool = False) -> dict:
    facts_p = rec_dir / "cell_facts.json"
    facts = _load(facts_p) if facts_p.is_file() else {}
    report: dict = {"cell_facts": str(facts_p), "copied": [], "liveness": None, "at_cap": {}}

    # ---- 1. copy EVERY standup's positive controls, never overwriting -----------------------
    per_standup = {}
    for phase, pc in find_standups(serve_root, rec_dir):
        attempt = 0
        while True:
            dst = rec_dir / f"liveness_{phase}_{jobid}_{attempt}.json"
            if not dst.exists():
                break
            attempt += 1
        shutil.copy2(pc, dst)
        report["copied"].append(str(dst.name))
        try:
            blk = liveness_block(_load(pc), cell)
            per_standup[f"{phase}_{jobid}_{attempt}"] = (
                quadruple_of(blk) if blk else {"error": f"no logprob_liveness for {cell!r} in {pc.name}"})
        except (OSError, ValueError) as e:
            per_standup[f"{phase}_{jobid}_{attempt}"] = {"error": f"{type(e).__name__}: {e}"}

    # the rung's headline liveness is the FIRST standup's (the one the trade was measured on);
    # every standup's is kept beside it, because a failed attempt's reading is evidence too
    headline = None
    for key in sorted(per_standup):
        if "error" not in per_standup[key]:
            headline = per_standup[key]
            break
    liveness = {"source": "positive_controls.json controls.steering_bites.cells."
                          f"{cell}.logprob_liveness (the STORED schema)",
                "per_standup": per_standup,
                **(headline or {"error": "no standup produced a readable logprob_liveness block"})}
    report["liveness"] = liveness

    # ---- 2. at-cap ITEM IDS, per control and cell, from each store's own gen_stats ----------
    at_cap: dict = {}
    for gs_p in sorted(controls_root.rglob("gen_stats.json")):
        try:
            gs = _load(gs_p)
        except (OSError, ValueError) as e:
            at_cap[str(gs_p.relative_to(controls_root))] = {"error": f"{type(e).__name__}: {e}"}
            continue
        for control, blk in gs.items():
            if not isinstance(blk, dict) or "at_cap_item_ids" not in blk:
                continue
            ids = blk.get("at_cap_item_ids") or []
            # <rung root>/cells/<cell key> -> keep the rung and the arm apart
            rung = gs_p.parent.parent.parent.name
            arm = gs_p.parent.name.split("__", 1)[0]
            at_cap.setdefault(control, {}).setdefault(rung, {})[arm] = {
                "n_at_cap": blk.get("n_at_cap"), "cap": blk.get("cap"),
                "at_cap_item_ids": ids, "n_item_ids": len(ids),
                "store": str(gs_p.parent)}
    report["at_cap"] = {c: {r: {a: v["n_at_cap"] for a, v in arms.items()}
                            for r, arms in rungs.items()} for c, rungs in at_cap.items()}

    # ---- 3. merge, never clobbering --------------------------------------------------------
    wrote = []
    for key, val in (("liveness", liveness), ("at_cap", at_cap)):
        if key in facts and not force:
            report.setdefault("kept_existing", []).append(key)
            continue
        facts[key] = val
        wrote.append(key)
    facts_p.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    report["wrote_keys"] = wrote
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rec-dir", required=True, type=Path)
    ap.add_argument("--serve-root", required=True, type=Path,
                    help="the job's serving dir (searched recursively for every standup)")
    ap.add_argument("--controls-root", required=True, type=Path)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--jobid", default="manual")
    ap.add_argument("--force", action="store_true", help="overwrite existing liveness/at_cap keys")
    a = ap.parse_args(argv)
    for p in (a.rec_dir, a.serve_root, a.controls_root):
        if not p.is_dir():
            print(f"REFUSING: no directory at {p}", file=sys.stderr)
            return 2
    rep = harvest(a.rec_dir, a.serve_root, a.controls_root, a.cell, a.jobid, force=a.force)
    lv = rep["liveness"]
    print(f"[harvest] copied {len(rep['copied'])} liveness record(s): {', '.join(rep['copied']) or 'none'}")
    print(f"[harvest] liveness: verified={lv.get('verified')} n_probes={lv.get('n_probes')} "
          f"n_conclusive={lv.get('n_conclusive')} "
          f"divergence.steered_vs_scale0_median={lv.get('divergence.steered_vs_scale0_median')}")
    for control, rungs in sorted(rep["at_cap"].items()):
        for rung, arms in sorted(rungs.items()):
            print(f"[harvest] at_cap {control}/{rung}: " +
                  ", ".join(f"{a}={n}" for a, n in sorted(arms.items())))
    print(f"[harvest] cell_facts keys written: {rep['wrote_keys'] or 'none'}"
          + (f" (kept existing: {rep['kept_existing']})" if rep.get("kept_existing") else ""))
    return 0 if lv.get("verified") is not None or "error" not in lv else 1


if __name__ == "__main__":
    sys.exit(main())
