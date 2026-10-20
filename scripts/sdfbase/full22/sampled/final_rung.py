"""The ONE per-(cell, control) final-rung rule — stdlib only, so the producer side (grade_sampled.sh's planner, run under whatever --python)
and the readers (compare_greedy_sampled.py, analysis_sampled.py) share one implementation instead of forking it.

A (cell, control)'s FINAL rung is the highest rung root whose store for the cell carries a COMPLETE gen_stats block for THAT control. It
is never chosen per cell: the two controls of one cell leave the ladder at different rungs (real gemma4e4b t1: `none`'s xstest exited at
2048 while its or_bench_hard_1k reached 8192, so the cell's 8192 store holds no xstest at all).
"""
from __future__ import annotations

import json
from pathlib import Path

CONTROL_LADDER_RUNGS = (512, 2048, 8192, 16384)   # the controls ladder as generated (= ladder_decide.LADDERS["controls"]; analysis_sampled asserts they agree)


def control_store_dirs(cells_root: Path, refuse=ValueError) -> dict:
    """cell_id -> store dir, matched on manifest.json's cell_id (the dir name mangles `__` to `--` and hashes the gen block, so it differs
    per rung); falls back to the dir-name prefix when a manifest is absent. Two stores for one cell raise `refuse(msg)`."""
    out: dict = {}
    cells_root = Path(cells_root)
    if not cells_root.is_dir():
        return out
    for d in sorted(cells_root.iterdir()):
        if not d.is_dir():
            continue
        mp = d / "manifest.json"
        if mp.exists():
            cid = json.loads(mp.read_text()).get("cell_id")
        else:
            cid = d.name.split("__controls__")[0].replace("--", "__")
        if cid in out:
            raise refuse(f"two stores for cell {cid!r} under {cells_root}: {out[cid].name} and {d.name}")
        out[cid] = d
    return out


def rung_store(controls_root: Path, cap: int, tag: str, trial: int, cell: str, refuse=ValueError) -> Path | None:
    root = Path(controls_root) / f"controls_cap{cap}" / f"{tag}_t{trial}" / "cells"
    return control_store_dirs(root, refuse).get(cell) if root.is_dir() else None


def final_rung_store(controls_root: Path, tag: str, trial: int, cell: str, control: str, refuse=ValueError) -> tuple[int | None, Path | None, list[int]]:
    """(final cap, its store, the rungs present for the control): the highest rung root whose store for `cell` carries a complete gen_stats
    block for `control`; (None, None, []) when the cell has no store at any rung."""
    present: list[int] = []
    best: tuple[int | None, Path | None] = (None, None)
    for cap in CONTROL_LADDER_RUNGS:
        st = rung_store(controls_root, cap, tag, trial, cell, refuse)
        if st is None:
            continue
        gsp = st / "gen_stats.json"
        if gsp.exists() and control in json.loads(gsp.read_text()):
            present.append(cap); best = (cap, st)
    return best[0], best[1], present
