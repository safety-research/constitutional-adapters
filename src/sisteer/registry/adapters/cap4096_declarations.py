"""The cap-4096 multi-turn LANE FILE's declarations -> ``source_state`` rows.

``<run-state>/transfer_stats_full22_cap4096.json`` is the multi-turn wave's live lane file
(a refresher rewrites it on every landing). Its MEASUREMENTS are the r2 export adapter's to ingest (PR-3a); what THIS adapter states is
what the file names NOW under the JSON paths the round's rulings reference, so a self-retiring ruling (``ruling.expires_when_json``)
evaluates against live evidence instead of reading ``missing``:

* for every ``models.<tag>.cells.<cell>`` node, the GRADE ROOTS its attack blocks name —
  ``sorted({block.root for block in node.values() if dict and block.root})``, the reader's ``load.declaration_root_state`` rule the ruling
  transcribes as its ``members_rule`` — under exactly that dotted path (no ``$.`` prefix: the rulings spell ``models.llama8b_it.cells.arditi``);
* for every grade root's ``<run-state>/<root>/<tag>/grading_provenance.json``, its sha under ``$`` (the rulings' ``input_sha_pinned`` guards
  pin the GRADE ROOT's provenance, which changes only when a cell is re-graded — not the lane file, which every landing rewrites).

Paths are spelled the way the rulings spell them: ``~``-relative under HOME. That is this adapter's ``Input.uri`` (the artifact, the
watermark and the state rows all carry it), while ``Input.path`` is the resolved file that was hashed. A ruled path the file no longer
resolves is stated ``path_resolves = 0`` rather than left to a stale row: the paths the rulings reference are declared here
(:data:`RULED_PATHS`, :data:`RULED_GUARDS`, transcribed from ``configs/registry/rulings_r2_v1.jsonl``), and every path this source
stated before (``ctx.conn``) is re-stated. Nothing is emitted from ``read`` — this source has no measurements of its own
(``covers_sources`` is empty); its whole product is the marker's ``source_state`` rows.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterator, List, Sequence, Tuple

from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_glob
from . import declared_common as DC
from .declared_common import HOME, read_json

NAME = "cap4096_lane"
GROUP = "declarations"
TRANSFER_STATS_FILE_NAME = "transfer_stats_full22_cap4096.json"
PROVENANCE_GLOB = os.path.join("grades_*", "*", "grading_provenance.json")
#: (json_path, ruling_id) — the lane-file paths the round's rulings evaluate (`grades_root_absent` conditions), rulings_r2_v1.jsonl
RULED_PATHS: Tuple[Tuple[str, str], ...] = (
    ("models.llama8b_it.cells.arditi", "r2v1-supersede-llama-arditi-r2-multiturn-r0p1"),
)
#: (path under the run-state, ruling_id) — the grade-root provenance files the rulings pin (`input_sha_pinned` guards)
RULED_GUARDS: Tuple[Tuple[str, str], ...] = (
    (os.path.join("grades_mt4karr2_ch_llama8b_it", "llama8b_it", "grading_provenance.json"), "r2v1-supersede-llama-arditi-r2-multiturn-r0p1"),
)


def ruling_uri(path: str, home: str = HOME) -> str:
    """A path spelled the way the rulings spell it: ``~``-relative when it lies under ``home``, else unchanged."""
    home = home.rstrip("/")
    if path == home or path.startswith(home + "/"):
        return "~" + path[len(home):]
    return path


def root_members(node: object) -> List[str]:
    """``load.declaration_root_state``'s membership rule: the distinct ``root`` of every dict block under a cell node, sorted."""
    if not isinstance(node, dict):
        return []
    return sorted({str(b["root"]) for b in node.values() if isinstance(b, dict) and b.get("root")})


class Cap4096LaneDeclarationsAdapter:
    name = NAME
    covers_sources: Tuple[str, ...] = ()          # declarations only: the lane file's measurements are the r2 export adapter's
    rng_scope = "none"
    allow_symlinks = False

    def __init__(self, run_state: str = DC.RUN_STATE, home: str = HOME):
        self.run_state, self.home = run_state.rstrip("/"), home
        self.roots = (self.run_state,)
        self.lane_file = os.path.join(self.run_state, TRANSFER_STATS_FILE_NAME)
        self.last_facts: Dict[str, object] = {}

    # -- inputs -----------------------------------------------------------------------------------
    def _input(self, path: str, role: str) -> Input:
        inp = check_input(path, role=role, allow_symlink=self.allow_symlinks)
        return inp.with_uri(ruling_uri(path, self.home))

    def discover(self) -> List[Input]:
        if not os.path.lexists(self.lane_file):
            raise AdapterRefusal(self.lane_file, "input missing")
        out = [self._input(self.lane_file, "report")]
        for p in sorted(store_glob(os.path.join(self.run_state, PROVENANCE_GLOB))):
            out.append(self._input(p, "report"))
        return out

    def comparison_group(self, inp: Input) -> str:
        return GROUP

    @property
    def source_id(self) -> str:
        return f"{self.name}:{GROUP}"

    def _lane_legacy(self, inputs: Sequence[Input]) -> Input:
        lane_legacy = next((i for i in inputs if os.path.basename(i.path) == TRANSFER_STATS_FILE_NAME), None)
        if lane_legacy is None:
            raise AdapterRefusal(self.lane_file, "the lane file is not among the group's inputs")
        return lane_legacy

    # -- protocol -----------------------------------------------------------------------------------
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        lane_legacy = self._lane_legacy(inputs)
        d = read_json(lane_legacy.path)
        models = d.get("models") if isinstance(d, dict) else None
        if not isinstance(models, dict):
            raise AdapterRefusal(lane_legacy.uri, "no `models` block: not the cap-4096 lane file this adapter states")
        cells = {f"{tag}.{cid}": root_members(node) for tag, blk in models.items() if isinstance(blk, dict)
                 for cid, node in ((blk.get("cells") or {}).items() if isinstance(blk.get("cells"), dict) else ())}
        self.last_facts = {"models": sorted(models), "cells": len(cells), "roots": sorted({r for m in cells.values() for r in m}),
                           "provenance_files": len(inputs) - 1, "ruled_paths_resolving": [jp for jp, _ in RULED_PATHS if jp.split(".", 1)[1].replace(".cells.", ".") in cells]}
        return iter(())                                     # no measurement records: the state rows are the product

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        lane_legacy = self._lane_legacy(inputs)
        d = read_json(lane_legacy.path)
        models = d.get("models") if isinstance(d, dict) else None
        if not isinstance(models, dict):
            raise AdapterRefusal(lane_legacy.uri, "no `models` block: not the cap-4096 lane file this adapter states")
        sid = self.source_id
        rows: List[SourceStateRow] = []
        stated = set()
        for tag, blk in sorted(models.items()):
            cells = blk.get("cells") if isinstance(blk, dict) else None
            if not isinstance(cells, dict):
                continue
            for cid, node in sorted(cells.items()):
                jp = f"models.{tag}.cells.{cid}"
                resolves = isinstance(node, dict)
                rows.append(SourceStateRow(sid, lane_legacy.uri, jp, 1, 1 if resolves else 0, root_members(node), lane_legacy.sha256))
                stated.add(jp)
        for jp, _rid in RULED_PATHS:
            if jp not in stated:                       # a ruled path the file no longer carries: stated absent, never left stale
                rows.append(SourceStateRow(sid, lane_legacy.uri, jp, 1, 0, [], lane_legacy.sha256))
                stated.add(jp)
        if ctx.conn is not None:
            prior = ctx.conn.execute("SELECT DISTINCT json_path FROM source_state WHERE source_id = ? AND input_uri = ?", (sid, lane_legacy.uri)).fetchall()
            for (jp,) in prior:
                if jp not in stated:
                    rows.append(SourceStateRow(sid, lane_legacy.uri, str(jp), 1, 0, [], lane_legacy.sha256))
                    stated.add(jp)
        seen_guards = set()
        for inp in inputs:
            if inp is lane_legacy:
                continue
            rel = os.path.relpath(inp.path, self.run_state)
            rows.append(SourceStateRow(sid, inp.uri, "$", 1, 1, [rel], inp.sha256))
            seen_guards.add(rel)
        for rel, _rid in RULED_GUARDS:
            if rel not in seen_guards:                 # a pinned provenance file that is gone: found = 0 (the guard reads `missing`, visibly)
                rows.append(SourceStateRow(sid, ruling_uri(os.path.join(self.run_state, rel), self.home), "$", 0, 0, [], None))
        return rows


def register() -> None:
    from .base import register as _register
    _register(Cap4096LaneDeclarationsAdapter.name, lambda: Cap4096LaneDeclarationsAdapter())
