#!/usr/bin/env python
"""tinycap-shared-standup §1.9.4 (inode rule 4): the wave-close PLANNER — what a finished wave may prune
from the volume, decided from the lane's RECORDS on the grading host, never from a listing of the volume.

    python scripts/modal/tinycap_close_wave.py --sha <full sha> \\
        --launch-set <launch_sets_shared.json> --records-dir <bcbw_records_shared> \\
        --relaunches <relaunches.json> \\
        --provenance <run-state>/bigcodebench_full_shared/PROVENANCE.json \\
        --provenance <run-state>/bfcl_full_shared/PROVENANCE.json \\
        [--out inventory.json] [--confirm --app-name <app> [--modal-python ~/venvs/modal/bin/python]]

The plan REFUSES (exit 2, nothing written to the volume) unless EVERY launch-set row of the wave (rows
whose ``sha`` is this sha) has a PASS gate record at this sha under ``--records-dir``
(``gate_*_<cell>_<suffix>.json``: ``verdict PASS``, ``sha``, ``cell``, ``suffix`` equal) AND every store
is listed in ``stores_mirrored`` of the PROVENANCE of every eval the row files AND every declared row
dir is mirrored (a PROVENANCE ``records`` entry under ``records/rows/<row>/<suffix>/`` or a
``row_dirs_mirrored`` list naming ``rows/<row>/<suffix>``). What it then names for deletion, every
path VOLUME-RELATIVE (``tinycap/…``), kinds in ``KINDS``:

- the tree archive ``tinycap/repos/sisteer-<sha>.tar.gz`` is NEVER in a wave's inventory: under
  tarball-only staging it is the ONE volume copy of the tree for every row at the sha, whichever launch set
  scheduled it — including a wave queued at the sha that has not started and so has no row dir yet — so no
  per-wave close can prove it unconsumed. It is pruned only by an explicit cross-wave operation, which is
  not this landing's. A ``tarball`` entry is refused by kind on both halves (``inventory_refusals`` here,
  ``close_wave_inventory_problems`` in the container).
- ``parked_store``: the stores ``relaunches.json`` parks for this wave (an entry whose ``sha`` is this
  sha or whose ``parked_suffix`` carries its sha7; the store's row dir comes from the launch-set row
  that names the entry's wave, or the entry's own ``model_row``) — an INCOMPLETE store is never
  resumed, so once its relaunch passed the gate the parked one is inodes for nothing.
- ``negctl_store``: the negative-control smoke's ``…-<sha7>-…-negctl`` stores (never gated,
  never mirrored) — a per-row glob the container expands.
- ``preflight_artefact``: ``tinycap/gen/_preflight/<row>/<sha>`` (the prompt-budget artefacts the
  preflight writes per (row, tree)) and the ``_preflight/probe`` file.
- ``row_dir``: ``tinycap/gen/rows/<row>/<suffix>`` for every ``row_dir`` the launch-set rows declare,
  once mirrored (the row dir is a RECORD of the wave; the mirror carries it byte for byte).

Row facts the planner reads off a ``launch_sets/v1`` row (any eval block's ``launch`` / ``carried`` /
``carried_late`` lists): ``sha`` (else ``launched_as.sha``, else the launch set's ONE tree), ``cell``,
``model_row`` (else ``launched_as.row_dir``, else ``<model_key>-f22<view>``), ``suffix`` (else
``launched_as.suffix``, else ``bcbw-<bcbw_wave>-<sha7>-<launch_tag>`` + ``-rep2`` when ``repeat_of`` is
set), ``tag``, and ``row_dir`` (``rows/<row>/<suffix>`` or the volume path) — a row WITHOUT ``row_dir``
refuses, because the prune cannot guess which row dir a cell rode (the Arditi cell's suffix differs from
its row dir's). The container half is ``tinycap_served.close_wave`` (``modal run
scripts/modal/tinycap_served.py::close_wave --inventory-json '<json>' [--confirm]``), which re-checks
every entry, prints the inventory with inode counts, and deletes only under ``--confirm``.
Imports only the contracts (no Modal): the plan is pure over the records handed to it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402

INVENTORY_SCHEMA = "tinycap_close_wave_inventory/1"
PLAN_SCHEMA = "tinycap_close_wave_plan/1"
KIND_PARKED_STORE, KIND_NEGCTL_STORE, KIND_PREFLIGHT, KIND_ROW_DIR = (
    "parked_store", "negctl_store", "preflight_artefact", "row_dir")
KINDS: Tuple[str, ...] = (KIND_PARKED_STORE, KIND_NEGCTL_STORE, KIND_PREFLIGHT, KIND_ROW_DIR)
#: The retired entry kind — the tree archive is shared at its sha and is never a wave's to prune; an inventory
#: carrying one is refused by name on both halves
KIND_TARBALL_RETIRED = "tarball"
TREE_ARCHIVE_REASON = "shared at sha; pruned only by an explicit cross-wave operation (not this landing)"
VOL_PATH = "/vol"
GEN_ROOT_REL = C.VOL_GEN_ROOT[len(VOL_PATH) + 1:]           # tinycap/gen
REPOS_ROOT_REL = C.VOL_REPOS_ROOT[len(VOL_PATH) + 1:]       # tinycap/repos
ROWS_SUBDIR, PREFLIGHT_SUBDIR, NEGCTL_TAIL = "rows", "_preflight", "negctl"
THINKING_DIR = f"thinking_{C.THINKING_OFF}"
GATE_PASS = "PASS"
EXIT_OK, EXIT_REFUSED = 0, 2
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
LAUNCHER_REL = "scripts/modal/tinycap_served.py"
MODAL_PYTHON_DEFAULT = "~/venvs/modal/bin/python"


class PlanError(ValueError):
    """An input the planner cannot read (a malformed record); a REFUSAL is not an error — it is the plan's verdict."""


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_matches(a: Optional[str], b: Optional[str]) -> bool:
    """Two shas name one commit when one is a prefix of the other (an abbreviation against the full sha)."""
    if not a or not b:
        return False
    a, b = str(a).strip().lower(), str(b).strip().lower()
    return a.startswith(b) or b.startswith(a)


# --------------------------------------------------------------------------- #
# the launch set's rows                                                         #
# --------------------------------------------------------------------------- #


def _lane_rows(block: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """`launch` + `carried` + `carried_late` of one eval block, one row per (model_key, tuple_id) — the b64 mirror's
    `lane_rows` rule (a row in several lists keeps its `launch` copy)."""
    seen: Set[Tuple[Any, Any]] = set()
    rows: List[Mapping[str, Any]] = []
    for lst in ("launch", "carried", "carried_late"):
        for r in block.get(lst) or []:
            if not isinstance(r, Mapping):
                continue
            tid = r.get("tuple_id") or f"{r.get('view')}/{r.get('cell')}"
            key = (r.get("model_key"), tid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def _single_tree_sha(launch_set: Mapping[str, Any]) -> Optional[str]:
    trees = launch_set.get("trees")
    if isinstance(trees, list) and len(trees) == 1 and isinstance(trees[0], Mapping) and trees[0].get("sha"):
        return str(trees[0]["sha"])
    return None


def _row_dir_rel(value: Any) -> Optional[str]:
    """`rows/<row>/<suffix>` from a declared row dir: that spelling, a `/vol/tinycap/gen/rows/…` path, or a
    `tinycap/gen/rows/…` relative path; anything else is None (refused by the caller)."""
    if not isinstance(value, str) or not value:
        return None
    v = value.strip().rstrip("/")
    for prefix in (f"{C.VOL_GEN_ROOT}/", f"{GEN_ROOT_REL}/"):
        if v.startswith(prefix):
            v = v[len(prefix):]
            break
    parts = v.split("/")
    return v if len(parts) == 3 and parts[0] == ROWS_SUBDIR and all(parts) else None


def wave_rows(launch_set: Mapping[str, Any], sha: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The wave's rows (this sha) with their store paths, plus the refusals for rows the planner cannot place.
    Returns ``([{rel, model_row, cell, suffix, tag, evals, sha, row_dir, source_eval}], [refusal, …])``; a store
    filed by two evals is ONE row with both evals."""
    evals = launch_set.get("evals")
    if not isinstance(evals, Mapping) or not evals:
        raise PlanError("launch set has no `evals` blocks (launch_sets/v1)")
    default_sha = _single_tree_sha(launch_set)
    by_rel: Dict[str, Dict[str, Any]] = {}
    refusals: List[str] = []
    for eval_id, block in evals.items():
        if not isinstance(block, Mapping):
            continue
        for r in _lane_rows(block):
            la = r.get("launched_as") if isinstance(r.get("launched_as"), Mapping) else {}
            row_sha = r.get("sha") or la.get("sha") or default_sha
            if not sha_matches(row_sha, sha):
                continue                                   # another wave's row (or a carried row at another tree)
            label = f"{eval_id}:{r.get('model_key')}:{r.get('tuple_id') or r.get('cell')}"
            cell = r.get("cell")
            if not cell:
                refusals.append(f"{label}: row names no cell")
                continue
            model_row = r.get("model_row") or la.get("row_dir")
            if not model_row and r.get("model_key") and r.get("view"):
                model_row = f"{r['model_key']}-f22{r['view']}"
            if not model_row:
                refusals.append(f"{label}: cannot place the row (no model_row / launched_as.row_dir / model_key+view)")
                continue
            suffix = r.get("suffix") or la.get("suffix")
            if not suffix and r.get("launch_tag") and (r.get("bcbw_wave") or r.get("tag")):
                suffix = f"bcbw-{r.get('bcbw_wave') or r.get('tag')}-{str(sha)[:7]}-{str(r['launch_tag']).strip('-')}"
                if r.get("repeat_of"):
                    suffix += "-rep2"
            if not suffix:
                refusals.append(f"{label}: row names no suffix (nor launched_as.suffix / launch_tag) — the prune cannot name its store")
                continue
            row_dir = _row_dir_rel(r.get("row_dir"))
            if row_dir is None:
                refusals.append(f"{label}: row declares no row_dir (rows/<row>/<suffix>) — the prune cannot guess which row dir "
                                "this cell rode; the launch set must name it")
            rel = f"{C.OUT_SUBDIR_NATIVE}/{model_row}/{cell}/{THINKING_DIR}/{suffix}"
            row = by_rel.setdefault(rel, {"rel": rel, "model_row": str(model_row), "cell": str(cell), "suffix": str(suffix),
                                          "tag": r.get("tag") or r.get("bcbw_wave"), "evals": [], "sha": str(row_sha),
                                          "row_dir": row_dir, "wave": r.get("bcbw_wave") or r.get("tag")})
            if eval_id not in row["evals"]:
                row["evals"].append(str(eval_id))
    return sorted(by_rel.values(), key=lambda x: x["rel"]), refusals


# --------------------------------------------------------------------------- #
# the gate records and the mirror's provenance                                 #
# --------------------------------------------------------------------------- #


def gate_record_for(records_dir: str | Path, row: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The gate record of ``row`` under ``records_dir`` (``gate_*_<cell>_<suffix>.json`` whose ``cell`` and ``suffix``
    equal the row's): ``(record, path)`` or ``(None, None)``."""
    cell, suffix = row["cell"], row["suffix"]
    for p in sorted(Path(records_dir).glob(f"gate_*_{cell}_{suffix}.json")):
        try:
            rec = read_json(p)
        except (OSError, ValueError):
            continue
        if isinstance(rec, Mapping) and rec.get("cell") == cell and rec.get("suffix") == suffix:
            return dict(rec), str(p)
    return None, None


def gate_problem(rec: Optional[Mapping[str, Any]], path: Optional[str], sha: str) -> Optional[str]:
    if rec is None:
        return "no gate record"
    if not sha_matches(rec.get("sha"), sha):
        return f"gate record {path} is at sha {rec.get('sha')!r}, not this wave's {sha}"
    if rec.get("verdict") != GATE_PASS:
        return f"gate record {path} reads {rec.get('verdict')!r}, not {GATE_PASS} ({'; '.join(map(str, rec.get('problems') or [])) or 'no problems listed'})"
    return None


def provenance_stores(prov: Mapping[str, Any]) -> Set[str]:
    return {str(s).strip("/") for s in (prov.get("stores_mirrored") or []) if isinstance(s, str)}


def provenance_row_dirs(prov: Mapping[str, Any]) -> Set[str]:
    """`rows/<row>/<suffix>` for every row dir the mirror carries: a `records/rows/<row>/<suffix>/…` entry in `records`
    (the b64 mirror files every beside file under `records/`), or an explicit `row_dirs_mirrored` list."""
    out: Set[str] = set()
    for rec in prov.get("records") or []:
        if not isinstance(rec, str):
            continue
        parts = rec.strip("/").split("/")
        if len(parts) >= 4 and parts[0] == "records" and parts[1] == ROWS_SUBDIR:
            out.add("/".join(parts[1:4]))
    for rd in prov.get("row_dirs_mirrored") or []:
        rel = _row_dir_rel(rd)
        if rel:
            out.add(rel)
    return out


# --------------------------------------------------------------------------- #
# the relaunch index                                                            #
# --------------------------------------------------------------------------- #


def parked_stores(relaunches: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], sha: str
                  ) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """``(entries, refusals, notes)``: one ``{rel, why}`` per relaunch entry of THIS wave that names a ``parked_suffix``
    (an entry whose ``sha`` is this sha, or whose parked suffix carries its sha7); the row dir comes from the entry's
    ``model_row`` or from a launch-set row naming the entry's wave. An entry naming no parked suffix is a note (nothing
    to prune by name); one whose wave no row places is a refusal."""
    sha7 = sha[:7]
    model_row_of_wave: Dict[str, str] = {}
    for r in rows:
        for key in (r.get("wave"), r.get("tag")):
            if key:
                model_row_of_wave.setdefault(str(key), str(r["model_row"]))
    entries: List[Dict[str, str]] = []
    refusals: List[str] = []
    notes: List[str] = []
    for key, e in sorted((relaunches or {}).items()):
        if not isinstance(e, Mapping):
            continue
        parked = e.get("parked_suffix")
        mine = sha_matches(e.get("sha"), sha) or (isinstance(parked, str) and f"-{sha7}" in parked)
        if not mine:
            continue
        if not parked:
            notes.append(f"relaunch {key}: names no parked_suffix — nothing to prune by name")
            continue
        wave, _, cell = str(key).partition("/")
        cell = e.get("cell") or cell
        model_row = e.get("model_row") or model_row_of_wave.get(wave)
        if not model_row or not cell:
            refusals.append(f"relaunch {key}: cannot place the parked store (no launch-set row names wave {wave!r} and the "
                            "entry carries no model_row)")
            continue
        entries.append({"rel": f"{C.OUT_SUBDIR_NATIVE}/{model_row}/{cell}/{THINKING_DIR}/{parked}",
                        "why": f"parked by relaunches.json {key} ({str(e.get('tag') or '')}): {str(e.get('why') or '')[:160]}"})
    return entries, refusals, notes


# --------------------------------------------------------------------------- #
# the plan                                                                      #
# --------------------------------------------------------------------------- #


def _entry(kind: str, rel: str, why: str, *, glob: bool = False) -> Dict[str, Any]:
    e: Dict[str, Any] = {"kind": kind, "rel": rel, "why": why}
    if glob:
        e["glob"] = True
    return e


def inventory_refusals(inventory: Mapping[str, Any]) -> List[str]:
    """Why the planner will not hand an inventory to the container — every ``tarball`` entry by name (the tree
    archive is shared at its sha; no per-wave close may name it, whoever assembled the inventory)."""
    out: List[str] = []
    for i, e in enumerate(inventory.get("entries") or []):
        if isinstance(e, Mapping) and e.get("kind") == KIND_TARBALL_RETIRED:
            out.append(f"entry {i}: tarball {e.get('rel')!r} — the tree archive is {TREE_ARCHIVE_REASON}; not this wave's to prune")
    return out


def negctl_globs(rows: Sequence[Mapping[str, Any]], sha7: str) -> List[Tuple[str, str]]:
    """The negative-control smoke's stores, scoped to THIS wave's suffixes — ``(model_row,
    <primary launch suffix>-negctl)`` per distinct primary suffix of the wave's rows (a relaunch ``-r<N>`` and a
    repeat ``-rep2`` fold to their primary; the smoke files under ``<primary suffix>-negctl``) — never
    ``*-<sha7>-*negctl``, which would take another launch set's smoke at the same commit."""
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for r in rows:
        suffix = re.sub(r"(-rep\d+)?$", "", str(r["suffix"]))
        suffix = re.sub(r"-r\d+$", "", suffix)
        if f"-{sha7}" not in suffix:
            continue
        key = (str(r["model_row"]), f"{suffix}-{NEGCTL_TAIL}")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def approve_inventory(inventory: Dict[str, Any], *, ok: bool) -> Dict[str, Any]:
    """Code review R3-1: the planner's mark — ``plan_verdict`` PASS with the digest the container recomputes
    (``contracts.close_wave_inventory_digest``), or REFUSED with no digest; the container acts on PASS alone."""
    inventory.pop(C.CLOSE_WAVE_PLAN_DIGEST_KEY, None)
    inventory[C.CLOSE_WAVE_PLAN_VERDICT_KEY] = C.CLOSE_WAVE_PLAN_PASS if ok else C.CLOSE_WAVE_PLAN_REFUSED
    if ok:
        inventory[C.CLOSE_WAVE_PLAN_DIGEST_KEY] = C.close_wave_inventory_digest(inventory)
    return inventory


def plan(*, sha: str, launch_set: Mapping[str, Any], records_dir: str | Path, provenances: Sequence[Mapping[str, Any]],
         relaunches: Optional[Mapping[str, Any]] = None, now: Optional[float] = None) -> Dict[str, Any]:
    """The pure plan: ``{schema, ok, sha, refusals, notes, rows, inventory}``. ``ok`` False = REFUSED (the inventory is
    still listed, marked ``would_prune_once`` and ``plan_verdict REFUSED``, so the operator sees what is held back and
    why — and the container refuses to act on it). A PASS inventory carries the planner's mark (``plan_verdict`` PASS +
    ``plan_digest``). The tree archive ``sisteer-<sha>.tar.gz`` is NEVER in the inventory: shared by every wave at
    the sha — a wave queued there has no row dir yet, so nothing a per-wave close can read proves it unconsumed."""
    sha = str(sha).strip().lower()
    if not _SHA40.match(sha):
        raise PlanError(f"--sha must be the full 40-hex commit, got {sha!r}")
    sha7 = sha[:7]
    refusals: List[str] = []
    notes: List[str] = []
    rows, row_refusals = wave_rows(launch_set, sha)
    refusals.extend(row_refusals)
    if not rows:
        refusals.append(f"the launch set has no row at sha {sha} — nothing to close")
    provs = [p for p in provenances if isinstance(p, Mapping)]
    stores_by_eval: Dict[str, Set[str]] = {}
    for p in provs:
        stores_by_eval.setdefault(str(p.get("eval") or p.get("source") or "?"), set()).update(provenance_stores(p))
    mirrored_row_dirs: Set[str] = set()
    for p in provs:
        mirrored_row_dirs |= provenance_row_dirs(p)
    row_reports: List[Dict[str, Any]] = []
    declared_row_dirs: Dict[str, str] = {}
    for row in rows:
        rec, path = gate_record_for(records_dir, row)
        gp = gate_problem(rec, path, sha)
        mirror_problems = []
        for ev in row["evals"]:
            if ev not in stores_by_eval:
                mirror_problems.append(f"no PROVENANCE given for eval {ev!r}")
            elif row["rel"] not in stores_by_eval[ev]:
                mirror_problems.append(f"not in stores_mirrored of the {ev} mirror")
        if gp:
            refusals.append(f"{row['rel']}: {gp}")
        for mp in mirror_problems:
            refusals.append(f"{row['rel']}: {mp}")
        if row.get("row_dir"):
            declared_row_dirs.setdefault(row["row_dir"], row["model_row"])
        row_reports.append({**row, "gate": path, "gate_problem": gp, "mirror_problems": mirror_problems})
    for rd in sorted(declared_row_dirs):
        if rd not in mirrored_row_dirs:
            refusals.append(f"row dir {rd}: not mirrored (no PROVENANCE records/{rd}/… entry or row_dirs_mirrored)")
    parked, parked_refusals, parked_notes = parked_stores(relaunches or {}, rows, sha)
    refusals.extend(parked_refusals)
    notes.extend(parked_notes)
    # --- the inventory ---
    entries: List[Dict[str, Any]] = []
    # the tree archive is shared by every wave at this sha — a wave QUEUED at the sha has no row dir yet, so no
    # per-wave record can prove it unconsumed; it is never in this inventory
    tree_archive: Dict[str, Any] = {"pruned": False, "reason": TREE_ARCHIVE_REASON}
    notes.append(f"tree archive {REPOS_ROOT_REL}/sisteer-{sha}.tar.gz kept: {TREE_ARCHIVE_REASON} (D23)")
    for pe in parked:
        entries.append(_entry(KIND_PARKED_STORE, f"{GEN_ROOT_REL}/{pe['rel']}", pe["why"]))
    # code review R3-2: the smoke's stores by THIS wave's suffixes (never every `*-<sha7>-*negctl` of the model row)
    for mr, negctl_suffix in negctl_globs(rows, sha7):
        entries.append(_entry(KIND_NEGCTL_STORE, f"{GEN_ROOT_REL}/{C.OUT_SUBDIR_NATIVE}/{mr}/*/{THINKING_DIR}/{negctl_suffix}",
                              "the negative-control smoke's stores of this wave (plan §1.3: partial markers, never gated or mirrored)", glob=True))
    model_rows = sorted({r["model_row"] for r in rows})
    for mr in model_rows:
        entries.append(_entry(KIND_PREFLIGHT, f"{GEN_ROOT_REL}/{PREFLIGHT_SUBDIR}/{mr}/{sha}",
                              "the preflight's per-(row, tree) artefacts (prompt-budget records)"))
    entries.append(_entry(KIND_PREFLIGHT, f"{GEN_ROOT_REL}/{PREFLIGHT_SUBDIR}/probe", "the preflight's writability probe file"))
    for rd in sorted(declared_row_dirs):
        entries.append(_entry(KIND_ROW_DIR, f"{GEN_ROOT_REL}/{rd}", "the row dir, mirrored as a record of the wave (plan §1.1.2 step 4)"))
    ok = not refusals
    inventory = {"schema": INVENTORY_SCHEMA, "sha": sha, "created_at": float(now if now is not None else time.time()),
                 "entries": entries, "n_rows": len(rows), "would_prune_once": None if ok else refusals,
                 "tree_archive": tree_archive}
    assert not inventory_refusals(inventory)                        # by construction: no kind the planner emits names the archive
    approve_inventory(inventory, ok=ok)
    return {"schema": PLAN_SCHEMA, "ok": ok, "sha": sha, "refusals": refusals, "notes": notes, "rows": row_reports,
            "inventory": inventory}


def meter_decision_text(sha: str, inodes_freed: int) -> str:
    """The ledger line the lane wrapper records after a confirmed prune (a 0 h `--decision`)."""
    return f"close-wave {sha[:7]}: pruned {int(inodes_freed)} inodes"


def modal_run_argv(inventory: Mapping[str, Any], *, confirm: bool, modal_python: str = MODAL_PYTHON_DEFAULT,
                   launcher: Optional[str | Path] = None) -> List[str]:
    """The container call: `modal run scripts/modal/tinycap_served.py::close_wave --inventory-json <json> [--confirm]`."""
    argv = [os.path.expanduser(modal_python), "-m", "modal", "run", f"{launcher or (REPO / LAUNCHER_REL)}::close_wave",
            "--inventory-json", json.dumps(inventory, sort_keys=True)]
    if confirm:
        argv.append("--confirm")
    return argv


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sha", required=True, help="the wave's tree sha (full 40-hex)")
    ap.add_argument("--launch-set", required=True, help="launch_sets_shared.json (launch_sets/v1)")
    ap.add_argument("--records-dir", required=True, help="the gate records dir (gate_*.json)")
    ap.add_argument("--relaunches", default=None, help="relaunches.json (the D298 shape); absent = no parked store")
    ap.add_argument("--provenance", action="append", default=[], help="a mirror root's PROVENANCE.json (repeat per eval)")
    ap.add_argument("--out", default=None, help="write the plan (with the inventory) here")
    ap.add_argument("--confirm", action="store_true", help="run the container prune (deletes!) — needs --app-name")
    ap.add_argument("--app-name", default=None, help=f"the Modal app name the launcher runs as (${C.MODAL_APP_NAME_ENV})")
    ap.add_argument("--modal-python", default=MODAL_PYTHON_DEFAULT)
    ap.add_argument("--launcher", default=None, help=f"path to {LAUNCHER_REL} (default: this checkout's)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    launch_set = read_json(a.launch_set)
    relaunches = read_json(a.relaunches) if a.relaunches else {}
    provs = [read_json(p) for p in a.provenance]
    if not provs:
        print("[close-wave] REFUSED: no --provenance given — the prune needs the mirror's PROVENANCE per eval", flush=True)
        return EXIT_REFUSED
    rep = plan(sha=a.sha, launch_set=launch_set, records_dir=a.records_dir, provenances=provs, relaunches=relaunches)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"[close-wave] sha {rep['sha']}: {len(rep['rows'])} launch-set row(s); {len(rep['inventory']['entries'])} inventory "
          f"entr{'y' if len(rep['inventory']['entries']) == 1 else 'ies'}; {'OK' if rep['ok'] else 'REFUSED'}", flush=True)
    for n in rep["notes"]:
        print(f"[close-wave]   note: {n}", flush=True)
    for r in rep["refusals"]:
        print(f"[close-wave]   REFUSED: {r}", flush=True)
    for e in rep["inventory"]["entries"]:
        print(f"[close-wave]   {e['kind']:<18} {e['rel']}{'  (glob)' if e.get('glob') else ''}", flush=True)
    if not rep["ok"]:
        return EXIT_REFUSED
    if not a.confirm:
        print("[close-wave] dry run: nothing touched on the volume; re-run with --confirm --app-name <app> to prune "
              "(the container prints the inventory with inode counts before deleting)", flush=True)
        return EXIT_OK
    if not a.app_name:
        print("[close-wave] REFUSED: --confirm needs --app-name (the launcher's App is built under it)", flush=True)
        return EXIT_REFUSED
    for r in inventory_refusals(rep["inventory"]):                  # the planner never hands the container a tarball entry
        print(f"[close-wave] REFUSED: {r}", flush=True)
    if inventory_refusals(rep["inventory"]):
        return EXIT_REFUSED
    cmd = modal_run_argv(rep["inventory"], confirm=True, modal_python=a.modal_python, launcher=a.launcher)
    env = dict(os.environ, **{C.MODAL_APP_NAME_ENV: a.app_name})
    print(f"[close-wave] running: {C.MODAL_APP_NAME_ENV}={a.app_name} " + " ".join(cmd[:5]) + " …", flush=True)
    r = subprocess.run(cmd, env=env, text=True, capture_output=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print(f"[close-wave] container prune FAILED (rc {r.returncode})", flush=True)
        return EXIT_REFUSED
    freed = None
    for line in r.stdout.splitlines():
        if line.startswith("[close-wave] {"):
            try:
                freed = json.loads(line[len("[close-wave] "):]).get("inodes_freed")
            except ValueError:
                pass
    m = re.search(r'"inodes_freed":\s*(\d+)', r.stdout)
    if freed is None and m:
        freed = int(m.group(1))
    print(f"[close-wave] record in the lane meter (0 h): --decision {meter_decision_text(rep['sha'], freed or 0)!r}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
