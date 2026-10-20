#!/usr/bin/env python
"""stage_full22_bank_modal.py — stage ONE (model_key, view) of the full22 object bank on the Modal volume ``twosi-fits``, in the
layout the tinycap harness reads, and — with ``--parents`` — the parents every derived (LoRA-subtraction / paired-write) cell of
that view is held to. Generalised BY MODEL KEY from the two 4B-era lane scripts (``stage_seed_banks_modal.sh`` +
``stage_derived_parents_modal.sh``): one stager, the roster and the pins as its inputs, no per-model copy-edit.

The layouts (the ONE spelling every consumer resolves — ``contracts.full22_objects_dir`` / ``train_budget.parent_cell_dir``)::

    /objects/sdfbase/full22/<model_key>/<view>/<cell_id>/{object.pt, <identity sidecar>}     bare cell ids; EXACTLY two files
    /objects/sdfbase/full22/<model_key>/<view>/staged_manifest.json                           tinycap-full22-staged-objects/1, put LAST
    /out/<fit model_key>/cells_all/<parent cell_id>/{object.pt, train_budget_meta.json, metrics.jsonl, staged_from.json}

The PLAN is derived from the tree, never hand-listed: ``configs/sdfbase_full22_roster.yaml`` ``targets.<mk>.views.<view>.cells``
(through ``scripts/sdfbase/full22/f22_roster.py`` ``staged_cells`` — the roster's ONE reader) maps ``<cell_id>: <staged_id>``; the
expected sha of every object is ``configs/sdfbase_full22_pins.json[<mk>][<staged_id>].object_sha256``; the identity sidecar follows
the staged id's lane prefix (``base…`` -> train_budget_meta.json, ``derived…`` / ``paired…`` -> derived_meta.json, ``arditi`` ->
fit_provenance.json — the 4B script's map); a DOSE-SCALED COPY (``d0p75__…``, ``half__…``: scripts/sdfbase/scale_psr_dose.py) carries
its PARENT's identity sidecar with a ``dose_scaled_from`` block, so its lane is resolved from the ONE sidecar present in its source
dir, and its ``dose_scaled_from.parent_object_sha256`` must be pinned under the model key. The source is the staging mirror
``$STAGE/sdfbase/full22_objects/<mk>/cells/<staged_id>/`` (which also holds epoch checkpoints, COMPLETE, RESULT, …: exactly
TWO files are staged per object, never more).

VERIFIED BEFORE ANY PUT (every problem is a refusal by name; nothing is put while one stands):
  * the source object's sha256 == the pin; the sidecar's recorded ``object_sha256`` == the pin; the sidecar names THIS cell
    (``cell_id`` for a fit, ``derived_id`` for a derived object); exactly ONE identity sidecar sits in the source dir;
  * a derived object's ``parent_*`` blocks each name a parent whose sha is PINNED under this model key for that parent cell
    (``base__<parent cell>`` for the full22 lanes), and the parents agree on their fit ``model_key``;
  * under ``--parents`` each parent's mirror dir: object sha == the derived sidecar's block == the parent's train_budget_meta.json
    ``object_sha256`` == the pin; the sidecar hashes to the block's ``budget_meta_sha256``; it names the parent cell;
  * the registry's ``object`` rows (``~/registry/registry.sqlite``, read-only; ``--registry-root``): a row that DISAGREES with the
    pin (artifact sha / bytes / parents) refuses; an absent row or an unreachable database is recorded ``unverified``, not a block.

THE REMOTE PROTOCOL under ``--go`` (default is DRY: everything local, every remote command printed, no Modal call at all):
  0. the inode floor probe — OPT-IN via ``--inode-probe <modal script>`` (the ``w2_volume_probe.py`` contract: ``modal run
     <script> --floor-pct N`` prints one JSON line ``{inodes_used, inodes_total, pct_used, …}`` and exits 3 at/over the floor). This
     repo carries no standalone probe app and this module adds no Modal function, so without the flag the probe is RECORDED as not
     run and a warning is printed.
  1. per file: destination absent -> ``modal volume put`` (then re-checked present); present -> ``modal volume get`` + sha256 —
     identical is kept and recorded, DIFFERENT REFUSES by name (never overwritten, never deleted).
  2. after every file of the view: each cell dir listed (``ls --json``: exactly the planned names, nothing else) and every file
     pulled back (``get``) and sha256-compared to the local bytes.
  3. ``staged_manifest.json`` built, compared to any already-published one minus ``written_at`` (identical = already published;
     different = refused), and put LAST, then pulled back.
  4. the parents, same per-file protocol (``staged_from.json``: an existing one must name the same object sha). NO ``COMPLETE``
     marker is staged for a parent: the mirror's COMPLETE manifests 14–17 files (epoch checkpoints, fit_selection.json, …) that are
     deliberately NOT staged, and ``sisteer.eval.train_budget._complete_marker_problems`` accepts an ABSENT marker ("staged handovers
     may ship object + sidecar only") while refusing a PRESENT one whose manifested files are missing — so a staged COMPLETE would
     read as a torn publication and refuse the very parent it describes.
  5. receipts ``<receipts-dir>/bank.json`` (+ ``parents.json`` under ``--parents``) — ONLY after everything verified. A dry run writes
     its receipts under ``<receipts-dir>/dry/`` instead, flagged ``dry: true`` with ``verified_utc: null``: a dry preview run AFTER a
     live stage must never overwrite the live receipt the launch-set prerequisite check reads. Any refusal exits non-zero and writes
     NO receipt.

Pure functions first (``plan_bank`` / ``plan_parents`` / ``build_manifest`` / ``registry_check``); the Modal CLI is reached only
through ``VolumeCLI`` over an injectable ``runner`` (``subprocess.run``-shaped), so tests double it. Modal credentials come from the
environment; nothing here reads, prints or logs one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_LOCAL = HERE.parents[1]

VOLUME = "twosi-fits"
VOL_MOUNT = "/vol"
BANK_ROOT = "objects/sdfbase/full22"          # /vol/objects/sdfbase/full22/<model_key>/<view>/<cell_id>/
PARENTS_ROOT = "out"                           # /vol/out/<fit model_key>/cells_all/<parent cell_id>/
MANIFEST_NAME = "staged_manifest.json"
MANIFEST_SCHEMA = "tinycap-full22-staged-objects/1"
RECEIPT_SCHEMA = "stage-full22-bank-receipt/1"
STAGED_FROM_NAME = "staged_from.json"
STAGED_BY = "scripts/modal/stage_full22_bank_modal.py"
ROSTER_REL = "configs/sdfbase_full22_roster.yaml"
PINS_REL = "configs/sdfbase_full22_pins.json"
ROSTER_READER_REL = "scripts/sdfbase/full22/f22_roster.py"
OBJECT_NAME = "object.pt"
BUDGET_META = "train_budget_meta.json"
DERIVED_META = "derived_meta.json"
FIT_PROVENANCE = "fit_provenance.json"
METRICS_NAME = "metrics.jsonl"
#: every identity sidecar a staged dir may declare; exactly ONE is legal (train_budget.IDENTITY_SIDECARS)
IDENTITY_SIDECARS = (BUDGET_META, DERIVED_META, FIT_PROVENANCE)
#: the parent fields a derived sidecar's ``parent_*`` block must carry for the stage to bind it
PARENT_FIELDS = ("cell_id", "model_key", "object_sha256")
DOSE_BLOCK = "dose_scaled_from"       # scale_psr_dose.py's record inside a dose copy's (parent-inherited) sidecar
DEFAULT_MODAL = Path.home() / "venvs" / "modal" / "bin" / "modal"
DEFAULT_SOURCE_ROOT = "$STAGE/sdfbase/full22_objects"   # the staging directory's bank mirror; expanded through sisteer.site at use
DEFAULT_STAGE_ROOT = Path(tempfile.gettempdir()) / "stage-full22-bank"
EXIT_REFUSED = 2

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
Log = Callable[[str], None]


class Refusal(Exception):
    """A fail-closed refusal: named, exits non-zero, writes no receipt."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal(f"{path}: unreadable JSON ({e})") from None


def default_runner(argv: Sequence[str], **kw: Any) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(argv), capture_output=True, text=True, **kw)


# --------------------------------------------------------------------------------------------------------------------------- #
# the roster + pins (the tree's data, through the roster's one reader)                                                          #
# --------------------------------------------------------------------------------------------------------------------------- #

def load_roster_reader(tree: Path):
    """``f22_roster`` — the tree's copy when it carries one (its semantics belong to the tree whose roster is read), else this
    checkout's. Import-light (yaml only)."""
    for candidate in (Path(tree) / ROSTER_READER_REL, REPO_LOCAL / ROSTER_READER_REL):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("_stage_full22_f22_roster", candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise Refusal(f"no roster reader at {ROSTER_READER_REL} under {tree} or {REPO_LOCAL}")


def load_pins(tree: Path, model_key: str) -> Dict[str, dict]:
    pins_all = load_json(Path(tree) / PINS_REL)
    pins = pins_all.get(model_key) if isinstance(pins_all, dict) else None
    if not isinstance(pins, dict) or not pins:
        raise Refusal(f"{PINS_REL} carries no pins for model key {model_key!r} (keys: {sorted(pins_all) if isinstance(pins_all, dict) else '?'})")
    return pins


def pin_sha(pins: Mapping[str, dict], staged_id: str, *, model_key: str) -> str:
    pin = pins.get(staged_id)
    sha = pin.get("object_sha256") if isinstance(pin, dict) else None
    if not isinstance(sha, str) or len(sha) != 64:
        raise Refusal(f"{PINS_REL}[{model_key}] names no object_sha256 for staged id {staged_id!r}")
    return sha


def view_cells(tree: Path, model_key: str, view: str) -> Tuple[Dict[str, str], List[str]]:
    """``({cell_id: staged_id} the view stages, [prompted cells served without an object])`` through ``f22_roster``."""
    reader = load_roster_reader(tree)
    roster = reader.load_roster(Path(tree) / ROSTER_REL)
    try:
        served = reader.served_cells(roster, model_key, view)
    except KeyError as e:
        targets = sorted((roster.get("targets") or {}).keys())
        views = sorted(((roster.get("targets") or {}).get(model_key) or {}).get("views") or {})
        raise Refusal(f"{ROSTER_REL}: no (target, view) = ({model_key!r}, {view!r}) — missing key {e}; targets {targets}; "
                      f"views of {model_key!r}: {views}") from None
    staged = reader.staged_cells(roster, model_key, view)
    if not staged:
        raise Refusal(f"{model_key}/{view}: the roster stages no object for this view (only prompted cells?) — nothing to stage")
    return dict(staged), [c for c, s in served.items() if s is None]


# --------------------------------------------------------------------------------------------------------------------------- #
# lanes -> identity sidecars                                                                                                    #
# --------------------------------------------------------------------------------------------------------------------------- #

def lane_of(staged_id: str) -> str:
    if "__" not in staged_id:
        raise Refusal(f"staged id {staged_id!r} is not <lane>__<cell_id>")
    return staged_id.split("__", 1)[0]


def sidecar_for(staged_id: str, source_dir: Optional[Path] = None) -> Tuple[str, str]:
    """``(lane_kind, identity sidecar)`` by the staged id's lane prefix — the 4B script's map, lane-suffix tolerant (``base64s1``,
    ``derived_base64s1``, ``paired_eb64s1`` are the seed lanes' spellings). Any OTHER lane is admitted only as a DOSE-SCALED COPY:
    with ``source_dir`` given, exactly one identity sidecar sits there and carries a ``dose_scaled_from`` block (the copy inherits
    its parent's record — a derived parent's derived_meta.json, a fit parent's train_budget_meta.json — so the prefix alone cannot
    name it). An unknown lane without that evidence REFUSES by name rather than guessing an identity."""
    lane = lane_of(staged_id)
    if "arditi" in staged_id:
        return "arditi", FIT_PROVENANCE
    if lane.startswith("lecun"):
        # dose-1.0 random-init ReLU pair: the LeCun-init lane of the s2rl view —
        # `lecun__<cell>` is a FIT (its identity sidecar is train_budget_meta.json, like `base__`), `lecun_derived__<cell>` the composition of two
        # such fits (derived_meta.json, like `derived__`); the same bank full22_lecun the d0p75l dose copies were minted from, staged as-is
        return ("derived", DERIVED_META) if "derived" in lane else ("base", BUDGET_META)
    if lane.startswith("derived"):
        return "derived", DERIVED_META
    if lane.startswith("paired"):
        return "paired", DERIVED_META
    if lane.startswith("base"):
        return "base", BUDGET_META
    if source_dir is not None and Path(source_dir).is_dir():
        present = _identity_sidecars_present(Path(source_dir))
        if len(present) == 1:
            meta = load_json(Path(source_dir) / present[0])
            if isinstance(meta, dict) and isinstance(meta.get(DOSE_BLOCK), dict):
                return "dose", present[0]
        raise Refusal(f"staged id {staged_id!r}: lane {lane!r} is not base/derived/paired/arditi and {source_dir} carries identity sidecars "
                      f"{present} with no {DOSE_BLOCK!r} block — not a dose-scaled copy; refusing to guess an identity")
    raise Refusal(f"staged id {staged_id!r}: lane {lane!r} maps to no identity sidecar (known: base…, derived…, paired…, arditi; a dose copy "
                  f"needs its source dir's {DOSE_BLOCK!r} block)")


def is_derived_kind(kind: str) -> bool:
    return kind in ("derived", "paired")


def _derived_lane(lane: str) -> bool:
    """A lane whose staged dirs are derived objects (never a parent)."""
    return lane.startswith("derived") or lane.startswith("paired") or lane.startswith("lecun_derived")     # The LeCun composition is derived too


# --------------------------------------------------------------------------------------------------------------------------- #
# the plan                                                                                                                      #
# --------------------------------------------------------------------------------------------------------------------------- #

@dataclass
class PlannedFile:
    name: str
    local: Path
    remote: str            # absolute volume path (leading slash) — put / get spelling
    sha256: str
    bytes: int

    @property
    def remote_rel(self) -> str:   # the ls spelling (no leading slash)
        return self.remote.lstrip("/")


@dataclass
class ParentRef:
    role: str
    cell_id: str
    model_key: str
    object_sha256: str
    budget_meta_sha256: Optional[str]
    staged_id: str          # the pinned staged id carrying this parent (e.g. base__<cell>)


@dataclass
class PlannedCell:
    cell_id: str
    staged_id: str
    lane_kind: str
    sidecar: str
    source_dir: Path
    object_sha256: str
    object_bytes: int
    sidecar_sha256: str
    sidecar_bytes: int
    remote_dir: str
    files: List[PlannedFile]
    parents: List[ParentRef] = field(default_factory=list)
    dose_source: Optional[dict] = None      # a dose copy: {parent_object_sha256, parent_staged_id, scale} bound to the pins


@dataclass
class BankPlan:
    model_key: str
    view: str
    tree: Path
    tree_sha: str
    source: Path
    remote_view_root: str
    cells: List[PlannedCell]
    prompted_cells_not_staged: List[str]
    roster_sha256: str
    pins_sha256: str
    pins_compared: Dict[str, str]

    @property
    def files(self) -> List[PlannedFile]:
        return [f for c in self.cells for f in c.files]


@dataclass
class PlannedParent:
    cell_id: str
    fit_model_key: str
    staged_id: str
    source_dir: Path
    object_sha256: str
    budget_meta_sha256: str
    consumers: List[str]        # the derived cell ids holding this parent
    remote_dir: str
    files: List[PlannedFile]
    staged_from: dict


@dataclass
class ParentsPlan:
    model_key: str
    view: str
    parents: List[PlannedParent]

    @property
    def files(self) -> List[PlannedFile]:
        return [f for p in self.parents for f in p.files]


def remote_view_root(model_key: str, view: str) -> str:
    return f"/{BANK_ROOT}/{model_key}/{view}"


def _identity_sidecars_present(d: Path) -> List[str]:
    return [s for s in IDENTITY_SIDECARS if (d / s).is_file()]


def _parent_blocks(meta: Mapping[str, Any]) -> Dict[str, dict]:
    return {k: v for k, v in meta.items() if k.startswith("parent_") and isinstance(v, dict)}


def resolve_parent_staged_id(pins: Mapping[str, dict], block: Mapping[str, Any], *, model_key: str, where: str) -> str:
    """The pinned staged id of ONE parent block: a pin whose staged id ends ``__<parent cell_id>`` AND whose object_sha256 is the
    block's. The full22 lanes spell it ``base__<cell>``; the seed lanes ``base64s<s>__<cell>``. No such pin = the parent is not
    pinned under this model key = refuse (the task's rule: a derived object's parents must equal the pins of ``base__<parent cell>``).
    When the canonical ``base__<cell>`` pin exists it MUST be the match (a different sha there is a refit under the same id)."""
    cell = str(block.get("cell_id"))
    want = str(block.get("object_sha256"))
    canonical = f"base__{cell}"
    hits = [sid for sid, pin in pins.items()
            if "__" in sid and sid.endswith(f"__{cell}") and isinstance(pin, dict) and pin.get("object_sha256") == want
            and not _derived_lane(sid.split("__", 1)[0])]
    if canonical in pins:
        got = pin_sha(pins, canonical, model_key=model_key)
        if got == want:
            return canonical
        # dose 0.75: a SIBLING-LANE parent — the LeCun-init refit pinned `lecun__<cell>`
        # (the s2rl view, bank full22_lecun) shares its cell id with the zero-init fit pinned `base__<cell>`; a derived / dose-copied object
        # of the LeCun lane names the lecun parent's sha, which the canonical pin does not carry. Admitted when exactly ONE non-derived pin
        # ending `__<cell>` carries that sha (the fallback rule below, no longer short-circuited by the canonical id); any other difference
        # still refuses by name.
        if len(hits) != 1:
            raise Refusal(f"{where}: parent {cell} names sha {want[:12]} but {PINS_REL}[{model_key}][{canonical}] pins {got[:12]} — "
                          "the derived object's parent is not the pinned object under that id (and no sibling-lane pin carries it exactly once)")
    if len(hits) != 1:
        raise Refusal(f"{where}: parent {cell} (sha {want[:12]}) is pinned under {len(hits)} staged id(s) {hits} for {model_key!r} — "
                      "the parents of a staged derived object must be pinned exactly once")
    return hits[0]


def plan_cell(cell_id: str, staged_id: str, *, model_key: str, view: str, source: Path, pins: Mapping[str, dict]) -> PlannedCell:
    where = f"{model_key}/{view}/{cell_id} ({staged_id})"
    src = Path(source) / staged_id
    if not src.is_dir():
        raise Refusal(f"{where}: source dir {src} is missing")
    kind, sidecar = sidecar_for(staged_id, src)
    obj = src / OBJECT_NAME
    if not obj.is_file():
        raise Refusal(f"{where}: {obj} is missing")
    present = _identity_sidecars_present(src)
    if present != [sidecar]:
        raise Refusal(f"{where}: identity sidecars present {present} but lane {kind!r} stages exactly [{sidecar!r}] — "
                      "an ambiguous or absent identity is not staged")
    want = pin_sha(pins, staged_id, model_key=model_key)
    got = sha256_file(obj)
    if got != want:
        raise Refusal(f"{where}: source object sha {got[:12]} != pin {want[:12]} ({obj})")
    meta = load_json(src / sidecar)
    if not isinstance(meta, dict):
        raise Refusal(f"{where}: {sidecar} is not a JSON object")
    recorded = meta.get("object_sha256")
    if recorded != want:
        raise Refusal(f"{where}: {sidecar} records object_sha256 {str(recorded)[:12]} != pin {want[:12]} — a stale object/sidecar pair")
    id_key = "derived_id" if sidecar == DERIVED_META else "cell_id"
    if meta.get(id_key) != cell_id:
        raise Refusal(f"{where}: {sidecar} names {id_key}={meta.get(id_key)!r}, not the cell {cell_id!r}")
    dose_source = None
    if kind == "dose":
        blk = meta.get(DOSE_BLOCK) or {}
        psha, psid = blk.get("parent_object_sha256"), blk.get("parent_staged_id")
        if not isinstance(psha, str) or len(psha) != 64:
            raise Refusal(f"{where}: {sidecar} {DOSE_BLOCK} names no parent_object_sha256")
        pinned = [sid for sid, pin in pins.items() if isinstance(pin, dict) and pin.get("object_sha256") == psha]
        if psid and psid in pins and pin_sha(pins, psid, model_key=model_key) != psha:
            raise Refusal(f"{where}: {DOSE_BLOCK}.parent_staged_id {psid!r} is pinned at {pin_sha(pins, psid, model_key=model_key)[:12]}, "
                          f"not the recorded parent {psha[:12]}")
        if not pinned:
            raise Refusal(f"{where}: the dose copy's parent {psha[:12]} ({psid!r}) is pinned under no staged id of {model_key!r}")
        dose_source = {"parent_object_sha256": psha, "parent_staged_id": psid or pinned[0], "pinned_as": pinned, "scale": blk.get("scale")}
    parents: List[ParentRef] = []
    if sidecar == DERIVED_META:
        blocks = _parent_blocks(meta)
        if len(blocks) < 2:
            raise Refusal(f"{where}: {sidecar} records {len(blocks)} parent block(s); a derived object has >= 2")
        for role, block in blocks.items():
            missing = [f for f in PARENT_FIELDS if block.get(f) in (None, "")]
            if missing:
                raise Refusal(f"{where}: {sidecar} {role} records no {missing}")
            psid = resolve_parent_staged_id(pins, block, model_key=model_key, where=where)
            parents.append(ParentRef(role=role, cell_id=str(block["cell_id"]), model_key=str(block["model_key"]),
                                     object_sha256=str(block["object_sha256"]), budget_meta_sha256=block.get("budget_meta_sha256"),
                                     staged_id=psid))
        keys = sorted({p.model_key for p in parents})
        if len(keys) != 1:
            raise Refusal(f"{where}: the parents name different fit model keys {keys}")
    rdir = f"{remote_view_root(model_key, view)}/{cell_id}"
    files = [PlannedFile(OBJECT_NAME, obj, f"{rdir}/{OBJECT_NAME}", got, obj.stat().st_size),
             PlannedFile(sidecar, src / sidecar, f"{rdir}/{sidecar}", sha256_file(src / sidecar), (src / sidecar).stat().st_size)]
    assert len(files) == 2, "exactly two files per staged object"
    return PlannedCell(cell_id=cell_id, staged_id=staged_id, lane_kind=kind, sidecar=sidecar, source_dir=src, object_sha256=got,
                       object_bytes=files[0].bytes, sidecar_sha256=files[1].sha256, sidecar_bytes=files[1].bytes, remote_dir=rdir,
                       files=files, parents=parents, dose_source=dose_source)


def plan_bank(tree: Path, model_key: str, view: str, source: Path, *, tree_sha: str) -> BankPlan:
    """The whole view's plan, every local check done; raises :class:`Refusal` on the first problem."""
    tree = Path(tree)
    cells_map, prompted = view_cells(tree, model_key, view)
    pins = load_pins(tree, model_key)
    cells = [plan_cell(cid, sid, model_key=model_key, view=view, source=Path(source), pins=pins) for cid, sid in cells_map.items()]
    return BankPlan(model_key=model_key, view=view, tree=tree, tree_sha=tree_sha, source=Path(source),
                    remote_view_root=remote_view_root(model_key, view), cells=cells, prompted_cells_not_staged=prompted,
                    roster_sha256=sha256_file(tree / ROSTER_REL), pins_sha256=sha256_file(tree / PINS_REL),
                    pins_compared={c.staged_id: c.object_sha256 for c in cells})


def build_staged_from(parent: "PlannedParent", *, written_at: str, tree_sha: str) -> dict:
    return {"schema": "stage-full22-parent-staged-from/1", "staged_by": STAGED_BY, "tree_sha": tree_sha, "written_at": written_at,
            "source": f"grading_host:{parent.source_dir}", "cell_id": parent.cell_id, "fit_model_key": parent.fit_model_key,
            "staged_id": parent.staged_id, "object_sha256": parent.object_sha256, "budget_meta_sha256": parent.budget_meta_sha256,
            "files": {f.name: {"sha256": f.sha256, "bytes": f.bytes} for f in parent.files if f.name != STAGED_FROM_NAME},
            "consumers": sorted(parent.consumers),
            "complete_marker": "deliberately absent: the fit's COMPLETE manifests files not staged here (epoch checkpoints, …); "
                               "train_budget._complete_marker_problems accepts an absent marker and refuses an incoherent present one"}


def plan_parents(bank: BankPlan, *, pins: Mapping[str, dict], stage_dir: Path, written_at: str) -> ParentsPlan:
    """Every derived cell's parents, de-duplicated by (fit model key, cell id), each verified in the mirror: object sha == the derived
    sidecar's block == the parent's train_budget_meta.json == the pin; the sidecar hashes to the block's budget_meta_sha256; it names
    the parent cell; no second identity sidecar. ``staged_from.json`` is written under ``stage_dir`` (the one file this stager
    authors). No COMPLETE (module docstring, step 4)."""
    by_key: Dict[Tuple[str, str], Tuple[ParentRef, List[str]]] = {}
    for c in bank.cells:
        for p in c.parents:
            key = (p.model_key, p.cell_id)
            if key in by_key:
                prev, consumers = by_key[key]
                if prev.object_sha256 != p.object_sha256:
                    raise Refusal(f"parents: {c.cell_id} and {consumers} name {p.cell_id} under {p.model_key} with different shas "
                                  f"({p.object_sha256[:12]} vs {prev.object_sha256[:12]})")
                consumers.append(c.cell_id)
            else:
                by_key[key] = (p, [c.cell_id])
    parents: List[PlannedParent] = []
    for (fit_mk, cell), (ref, consumers) in by_key.items():
        where = f"parent {fit_mk}/{cell} ({ref.staged_id}; consumed by {consumers})"
        src = Path(bank.source) / ref.staged_id
        obj, tbm = src / OBJECT_NAME, src / BUDGET_META
        if not obj.is_file() or not tbm.is_file():
            raise Refusal(f"{where}: {src} lacks {OBJECT_NAME} and/or {BUDGET_META}")
        present = _identity_sidecars_present(src)
        if present != [BUDGET_META]:
            raise Refusal(f"{where}: identity sidecars present {present}; a fitted parent carries exactly [{BUDGET_META!r}]")
        osha = sha256_file(obj)
        if osha != ref.object_sha256:
            raise Refusal(f"{where}: mirror object sha {osha[:12]} != the derived sidecar's parent sha {ref.object_sha256[:12]}")
        want = pin_sha(pins, ref.staged_id, model_key=bank.model_key)
        if osha != want:
            raise Refusal(f"{where}: mirror object sha {osha[:12]} != pin {want[:12]}")
        meta = load_json(tbm)
        if not isinstance(meta, dict) or meta.get("object_sha256") != osha:
            raise Refusal(f"{where}: {BUDGET_META} records object_sha256 {str((meta or {}).get('object_sha256'))[:12]} != object {osha[:12]}")
        if meta.get("cell_id") != cell:
            raise Refusal(f"{where}: {BUDGET_META} names cell_id={meta.get('cell_id')!r}, not {cell!r}")
        tsha = sha256_file(tbm)
        if ref.budget_meta_sha256 is not None and tsha != ref.budget_meta_sha256:
            raise Refusal(f"{where}: {BUDGET_META} hashes to {tsha[:12]}, not the derived sidecar's budget_meta_sha256 "
                          f"{ref.budget_meta_sha256[:12]} — the parent's identity record changed under the derivation")
        rdir = f"/{PARENTS_ROOT}/{fit_mk}/cells_all/{cell}"
        files = [PlannedFile(OBJECT_NAME, obj, f"{rdir}/{OBJECT_NAME}", osha, obj.stat().st_size),
                 PlannedFile(BUDGET_META, tbm, f"{rdir}/{BUDGET_META}", tsha, tbm.stat().st_size)]
        metrics = src / METRICS_NAME
        if metrics.is_file():
            files.append(PlannedFile(METRICS_NAME, metrics, f"{rdir}/{METRICS_NAME}", sha256_file(metrics), metrics.stat().st_size))
        parent = PlannedParent(cell_id=cell, fit_model_key=fit_mk, staged_id=ref.staged_id, source_dir=src, object_sha256=osha,
                               budget_meta_sha256=tsha, consumers=sorted(consumers), remote_dir=rdir, files=files, staged_from={})
        parent.staged_from = build_staged_from(parent, written_at=written_at, tree_sha=bank.tree_sha)
        sf_dir = Path(stage_dir) / PARENTS_ROOT / fit_mk / "cells_all" / cell
        sf_dir.mkdir(parents=True, exist_ok=True)
        sf = sf_dir / STAGED_FROM_NAME
        sf.write_text(json.dumps(parent.staged_from, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        parent.files.append(PlannedFile(STAGED_FROM_NAME, sf, f"{rdir}/{STAGED_FROM_NAME}", sha256_file(sf), sf.stat().st_size))
        parents.append(parent)
    return ParentsPlan(model_key=bank.model_key, view=bank.view, parents=parents)


# --------------------------------------------------------------------------------------------------------------------------- #
# the manifest (tinycap-full22-staged-objects/1 — the 4B fields, plus the per-cell identity fields this stager records)         #
# --------------------------------------------------------------------------------------------------------------------------- #

def build_manifest(bank: BankPlan, *, verified: Mapping[str, Mapping[str, Any]], dry: bool, written_at: Optional[str]) -> dict:
    """``verified``: ``{remote path: {"verified_by": str}}`` for every staged file (empty on a dry run)."""
    cells = {}
    for c in bank.cells:
        files = [{"path": f"{c.cell_id}/{f.name}", "sha256": f.sha256, "bytes": f.bytes,
                  "verified_by": (verified.get(f.remote) or {}).get("verified_by", "not verified (dry run)")} for f in c.files]
        cells[c.cell_id] = {"cell": c.cell_id, "cell_id": c.cell_id, "staged_id": c.staged_id, "lane_kind": c.lane_kind,
                            "object_sha256": c.object_sha256, "bytes": c.object_bytes,
                            "identity_sidecar": c.sidecar, "sidecar": c.sidecar, "sidecar_sha256": c.sidecar_sha256,
                            "source": str(c.source_dir), "pin_match": True,
                            "volume_path": f"{VOLUME}:{c.remote_dir}", "objects_dir_entry": f"{VOL_MOUNT}{c.remote_dir}",
                            "n_files": len(files), "files": files,
                            **({"parents": {p.role: {"cell_id": p.cell_id, "model_key": p.model_key, "object_sha256": p.object_sha256,
                                                     "staged_id": p.staged_id} for p in c.parents}} if c.parents else {}),
                            **({DOSE_BLOCK: {k: c.dose_source[k] for k in ("parent_object_sha256", "parent_staged_id", "scale")}} if c.dose_source else {})}
    man = {"schema": MANIFEST_SCHEMA, "wave": "full22", "model_key": bank.model_key, "view": bank.view,
           "bank_source": f"grading_host:{bank.source} (sha-verified against the pins)", "dereferenced": True,
           "roster": ROSTER_REL, "roster_sha256": bank.roster_sha256, "pins": PINS_REL, "pins_sha256": bank.pins_sha256,
           "tree_sha": bank.tree_sha, "volume": VOLUME, "volume_path": bank.remote_view_root,
           "objects_dir": f"{VOL_MOUNT}{bank.remote_view_root}", "manifest_name": MANIFEST_NAME, "staged_by": STAGED_BY,
           "remote_verified": {"method": "modal volume ls --json names per cell dir; modal volume get + sha256 per file",
                               "n_files": 0 if dry else len(bank.files), **({"dry": True} if dry else {})},
           "prompted_cells_not_staged": list(bank.prompted_cells_not_staged), "cells": cells}
    if written_at is not None:
        man["written_at"] = written_at
    return man


def manifest_content(m: Mapping[str, Any]) -> dict:
    return {k: v for k, v in m.items() if k != "written_at"}


# --------------------------------------------------------------------------------------------------------------------------- #
# the registry cross-check (read-only; a disagreeing row refuses, an absent one is recorded)                                    #
# --------------------------------------------------------------------------------------------------------------------------- #

def _site_path(value: str, key: str) -> Path:
    """``value`` with its site placeholder expanded through sisteer.site (this checkout's module); an unset one refuses by name."""
    if str(REPO_LOCAL / "src") not in sys.path:
        sys.path.insert(0, str(REPO_LOCAL / "src"))
    from sisteer import site as _site
    try:
        return _site.expand_path(value, key=key)
    except (_site.PlaceholderError, _site.SiteConfigError) as exc:
        raise Refusal(str(exc)) from exc


def registry_check(objects: Mapping[str, Mapping[str, Any]], *, registry_root: Optional[str] = None) -> dict:
    """``objects``: ``{sha: {"staged_id", "bytes", "parents": [sha, …] or None}}``. Returns
    ``{"status": match|mismatch|unverified, "detail": str, "db": path, "rows": {sha: {...}}}``; the caller refuses on ``mismatch``."""
    sys.path.insert(0, str(REPO_LOCAL / "src"))
    try:
        from sisteer.registry import db as DB
        root = registry_root or os.environ.get("REGISTRY_ROOT") or DB.DEFAULT_ROOT
        path = DB.db_path(root)
        if not os.path.isfile(path):
            return {"status": "unverified", "detail": f"no registry database at {path}", "db": path, "rows": {}}
        conn = DB.connect(path, readonly=True)
    except Exception as e:  # noqa: BLE001 — every failure here is 'unverified', by name
        return {"status": "unverified", "detail": f"registry unreachable: {type(e).__name__}: {e}", "db": None, "rows": {}}
    rows: Dict[str, dict] = {}
    problems: List[str] = []
    try:
        for sha, want in objects.items():
            r = conn.execute("SELECT object_id, object_kind, campaign, campaign_cell_id_legacy, artifact_sha256, artifact_bytes, status, "
                             "derivation, parent_object_ids_json, fit_model_id FROM object WHERE object_id = ? OR artifact_sha256 = ?",
                             (sha, sha)).fetchall()
            if not r:
                rows[sha] = {"status": "unregistered", "staged_id": want.get("staged_id")}
                continue
            if len(r) > 1:
                problems.append(f"{sha[:12]}: {len(r)} object rows")
            row = dict(r[0])
            rec = {"status": "match", "object_id": row["object_id"], "object_kind": row["object_kind"], "campaign": row["campaign"],
                   "campaign_cell_id_legacy": row["campaign_cell_id_legacy"], "artifact_bytes": row["artifact_bytes"],
                   "registry_status": row["status"], "derivation": row["derivation"], "fit_model_id": row["fit_model_id"],
                   "staged_id": want.get("staged_id")}
            if row["artifact_sha256"] not in (None, sha):
                problems.append(f"{sha[:12]}: registry artifact_sha256 {str(row['artifact_sha256'])[:12]} != object")
                rec["status"] = "mismatch"
            if str(row["object_id"]) != sha:
                # a row that merely SHARES the artifact sha under another object_id is another object's row —
                # provenance is claimed only when the registry's identity (object_id) AND its artifact sha both equal the staged object
                problems.append(f"{sha[:12]}: the registry row found by this artifact sha is object_id {str(row['object_id'])[:12]}…, not this object's — "
                                "provenance is not claimed for another object's row")
                rec["status"] = "mismatch"
            if row["artifact_bytes"] is not None and want.get("bytes") is not None and int(row["artifact_bytes"]) != int(want["bytes"]):
                problems.append(f"{sha[:12]}: registry artifact_bytes {row['artifact_bytes']} != local {want['bytes']}")
                rec["status"] = "mismatch"
            if want.get("parents") is not None and row["parent_object_ids_json"]:
                try:
                    reg_parents = sorted(json.loads(row["parent_object_ids_json"]))
                except ValueError:
                    reg_parents = None
                if reg_parents is not None and reg_parents != sorted(want["parents"]):
                    problems.append(f"{sha[:12]}: registry parents {[p[:12] for p in reg_parents]} != sidecar {[p[:12] for p in sorted(want['parents'])]}")
                    rec["status"] = "mismatch"
            legacy = row["campaign_cell_id_legacy"]
            if legacy and want.get("staged_id") and legacy != want["staged_id"]:
                rec["note"] = f"campaign_cell_id_legacy {legacy!r} != staged id {want['staged_id']!r} (recorded, not a refusal)"
            rows[sha] = rec
    finally:
        conn.close()
    if problems:
        return {"status": "mismatch", "detail": "; ".join(problems), "db": path, "rows": rows}
    unreg = [s[:12] for s, r in rows.items() if r["status"] == "unregistered"]
    if unreg:
        return {"status": "unverified", "detail": f"no object row for {unreg} (the other {len(rows) - len(unreg)} match)", "db": path, "rows": rows}
    return {"status": "match", "detail": f"{len(rows)} object row(s) agree on artifact sha, bytes and parents", "db": path, "rows": rows}


def registry_objects_of(bank: BankPlan, parents: Optional[ParentsPlan]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in bank.cells:
        # the registry's parent_object_ids: a derivation's parents; for a DOSE copy its ONE dose source (derivation dose_scale), while
        # the copy's inherited sidecar carries the fit lineage's parent_* blocks — those are bound to the pins, not to this row
        reg_parents = [c.dose_source["parent_object_sha256"]] if c.dose_source else ([p.object_sha256 for p in c.parents] if c.parents else None)
        out[c.object_sha256] = {"staged_id": c.staged_id, "bytes": c.object_bytes, "parents": reg_parents}
    for p in (parents.parents if parents else []):
        out.setdefault(p.object_sha256, {"staged_id": p.staged_id, "bytes": p.files[0].bytes, "parents": None})
    return out


# --------------------------------------------------------------------------------------------------------------------------- #
# the Modal volume CLI, behind the runner seam                                                                                   #
# --------------------------------------------------------------------------------------------------------------------------- #

class VolumeCLI:
    def __init__(self, modal_bin: Path, volume: str = VOLUME, *, runner: Runner = default_runner, log: Log = print, environment: Optional[str] = None):
        self.modal, self.volume, self.runner, self.log = str(modal_bin), volume, runner, log
        #: the Modal ENVIRONMENT every command is scoped to (`--env`, right after the subcommand verb; `modal run --env` must precede the
        #: script path or the script receives it). None = the CLI's own resolution (MODAL_ENVIRONMENT, the active profile, the workspace
        #: default): a volume name is per environment, so a caller that launched `modal run -e X` reads and writes the same X here.
        self.environment = environment or None

    def argv(self, *args: str) -> List[str]:
        out = [self.modal, *args]
        if self.environment:
            at = 3 if args[:1] == ("volume",) else 2        # after `modal volume <verb>` / after `modal run`
            out[at:at] = ["--env", self.environment]
        return out

    def _run(self, *args: str) -> "subprocess.CompletedProcess[str]":
        return self.runner(self.argv(*args))

    def exists(self, remote_rel: str) -> bool:
        return self._run("volume", "ls", self.volume, remote_rel).returncode == 0

    def ls_json(self, remote_rel: str) -> List[dict]:
        r = self._run("volume", "ls", "--json", self.volume, remote_rel)
        if r.returncode != 0:
            raise Refusal(f"modal volume ls {remote_rel} failed: {(r.stderr or '').strip()[-300:]}")
        try:
            return json.loads(r.stdout or "[]")
        except ValueError as e:
            raise Refusal(f"modal volume ls --json {remote_rel}: unparseable output ({e})") from None

    def get(self, remote_abs: str, into: Path) -> Path:
        r = self._run("volume", "get", "--force", self.volume, remote_abs, str(into))
        if r.returncode != 0:
            raise Refusal(f"modal volume get {remote_abs} failed: {(r.stderr or '').strip()[-300:]}")
        got = Path(into) / Path(remote_abs).name
        if not got.is_file():
            raise Refusal(f"modal volume get {remote_abs} returned 0 but left no file at {got}")
        return got

    def get_tree(self, remote_abs: str, into: Path) -> Path:
        """A DIRECTORY pull: ``modal volume get --force <volume> <remote_abs> <into>`` with ``into`` an EXISTING local directory. The CLI
        downloads a folder recursively (modal 1.5.4, cli/_download.py: every entry lands at ``<into>/<entry path relative to the folder's
        PARENT>``), so the folder arrives as ``<into>/<basename(remote_abs)>/…``; a destination that does not exist is taken as a FILE
        path and the download fails on the first directory entry (verified 2026-10-12 on the stage-07 smoke stores). Returns that
        directory; refuses by name when the CLI fails or leaves no directory there. Read-only on the volume."""
        into = Path(into)
        if not into.is_dir():
            raise Refusal(f"modal volume get {remote_abs}: the local destination {into} must be an existing directory (the CLI lays the folder "
                          "under it as <destination>/<folder name>/)")
        r = self._run("volume", "get", "--force", self.volume, remote_abs, str(into))
        if r.returncode != 0:
            raise Refusal(f"modal volume get {remote_abs} failed: {(r.stderr or '').strip()[-300:]}")
        got = into / Path(remote_abs.rstrip("/")).name
        if not got.is_dir():
            raise Refusal(f"modal volume get {remote_abs} returned 0 but left no directory at {got}")
        return got

    def put(self, local: Path, remote_abs: str, *, force: bool = False) -> None:
        r = self._run("volume", "put", *(["--force"] if force else []), self.volume, str(local), remote_abs)
        if r.returncode != 0:
            raise Refusal(f"modal volume put {remote_abs} failed: {(r.stderr or '').strip()[-300:]}")
        if not self.exists(remote_abs.lstrip("/")):
            raise Refusal(f"{remote_abs} is not on the volume after a put that returned 0")

    def remote_sha(self, remote_abs: str) -> str:
        with tempfile.TemporaryDirectory(prefix="stage-full22-get-") as t:
            return sha256_file(self.get(remote_abs, Path(t)))

    def remote_json(self, remote_abs: str) -> Any:
        with tempfile.TemporaryDirectory(prefix="stage-full22-get-") as t:
            return load_json(self.get(remote_abs, Path(t)))

    def inode_probe(self, script: Path, floor_pct: float) -> dict:
        r = self._run("run", str(script), "--floor-pct", str(floor_pct))
        line = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
        try:
            rec = json.loads(line) if line else {}
        except ValueError:
            rec = {"raw": line}
        if r.returncode == 3:
            raise Refusal(f"the volume is at/above the {floor_pct} % inode floor — nothing staged: {line}")
        if r.returncode != 0:
            raise Refusal(f"inode probe {script} failed (rc {r.returncode}): {(r.stderr or '').strip()[-300:]}")
        return {"status": "ok", "floor_pct": floor_pct, **(rec if isinstance(rec, dict) else {"raw": rec})}


def stage_files(files: Sequence[PlannedFile], vol: VolumeCLI, *, dry: bool, log: Log, label: str) -> Dict[str, dict]:
    """Step 1 per file. Returns ``{remote: {"outcome": put|kept_identical|dry, "sha256", "bytes"}}``; a differing destination refuses."""
    out: Dict[str, dict] = {}
    for f in files:
        if dry:
            log(f"DRY  {vol.modal} volume ls {vol.volume} {f.remote_rel}   (absent -> put; present -> get + sha256, refuse on a difference)")
            log(f"DRY  {vol.modal} volume put {vol.volume} {f.local} {f.remote}   ({f.bytes} B, sha {f.sha256[:12]})")
            out[f.remote] = {"outcome": "dry", "sha256": f.sha256, "bytes": f.bytes}
            continue
        if vol.exists(f.remote_rel):
            remote_sha = vol.remote_sha(f.remote)
            if remote_sha != f.sha256:
                raise Refusal(f"{label}: {f.remote} differs on the volume (local {f.sha256[:12]} vs volume {remote_sha[:12]}) — "
                              "refusing to overwrite; a human decides")
            log(f"  identical on the volume, kept: {f.remote}")
            out[f.remote] = {"outcome": "kept_identical", "sha256": f.sha256, "bytes": f.bytes}
            continue
        log(f"  put {f.remote} ({f.bytes} B, sha {f.sha256[:12]})")
        vol.put(f.local, f.remote)
        out[f.remote] = {"outcome": "put", "sha256": f.sha256, "bytes": f.bytes}
    return out


def verify_remote(dirs: Mapping[str, Sequence[PlannedFile]], vol: VolumeCLI, *, log: Log, label: str,
                  exact_listing: bool = True) -> Dict[str, dict]:
    """Step 2: each remote dir lists exactly the planned names (``exact_listing``), every file pulled back and sha256-compared."""
    verified: Dict[str, dict] = {}
    problems: List[str] = []
    for rdir, files in dirs.items():
        names = {Path(e["filename"]).name for e in vol.ls_json(rdir.lstrip("/")) if e.get("type") == "file"}
        want = {f.name for f in files}
        if exact_listing and names != want:
            problems.append(f"{rdir}: remote files {sorted(names)} != planned {sorted(want)}")
        elif not want <= names:
            problems.append(f"{rdir}: remote files {sorted(names)} lack planned {sorted(want - names)}")
        for f in files:
            got = vol.remote_sha(f.remote)
            if got != f.sha256:
                problems.append(f"{f.remote}: volume sha {got[:12]} != local {f.sha256[:12]}")
            else:
                verified[f.remote] = {"verified_by": "get+sha256 and ls names", "sha256": f.sha256, "bytes": f.bytes}
    if problems:
        raise Refusal(f"{label}: remote verification FAILED — " + "; ".join(problems))
    log(f"  {label}: remote VERIFIED ({len(verified)} files by get + sha256; listings exact)")
    return verified


#: the ONE difference `--replace-manifest` may replace: the tree sha a later landing re-staged the SAME bytes under (the s1
#: manifest was published at 22bf6312c before the u0/s1 row split; the post-split restage of identical objects names the landed sha)
REPLACEABLE_MANIFEST_KEYS = ("tree_sha",)


def manifest_replace_problem(published: Mapping[str, Any], fresh: Mapping[str, Any]) -> Optional[str]:
    """Why the published manifest may NOT be replaced by ``fresh`` under `--replace-manifest` (None = replaceable): it was written by
    another stager, or the two differ in any key beyond :data:`REPLACEABLE_MANIFEST_KEYS` (``written_at`` aside)."""
    if published.get("staged_by") != STAGED_BY:
        return f"published by {published.get('staged_by')!r}, not this stager — another lane's manifest is never replaced"
    a, b = manifest_content(published), manifest_content(fresh)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    extra = [k for k in diff if k not in REPLACEABLE_MANIFEST_KEYS]
    if extra:
        return f"differing keys {diff}: only {list(REPLACEABLE_MANIFEST_KEYS)} may be replaced"
    if not diff:
        return "identical: nothing to replace"
    return None


def publish_manifest(man: dict, vol: VolumeCLI, *, remote_dir: str, stage_dir: Path, dry: bool, log: Log,
                     replace_manifest: bool = False) -> dict:
    """Step 3: compare to a published manifest minus ``written_at`` (identical = skip; different = refuse — unless ``replace_manifest``
    and the ONLY difference is the tree sha on a manifest this stager wrote: then replaced, logged, recorded), else put LAST + get back."""
    remote = f"{remote_dir}/{MANIFEST_NAME}"
    local_dir = Path(stage_dir) / remote_dir.lstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / MANIFEST_NAME
    local.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if dry:
        log(f"DRY  write {local}; then {vol.modal} volume put {vol.volume} {local} {remote}   (LAST, after the view verifies)")
        return {"outcome": "dry", "remote": remote, "local": str(local)}
    if vol.exists(remote.lstrip("/")):
        published = vol.remote_json(remote)
        if manifest_content(published) == manifest_content(man):
            log(f"  {remote}: an identical manifest is already published (written_at {published.get('written_at')}); nothing put")
            return {"outcome": "already_published", "remote": remote, "remote_written_at": published.get("written_at"), "local": str(local)}
        a, b = manifest_content(published), manifest_content(man)
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if not replace_manifest:
            raise Refusal(f"{remote}: a DIFFERENT staged_manifest.json is already published (staged_by {published.get('staged_by')!r}; "
                          f"differing keys {diff}) — refusing to replace it; a human decides (--replace-manifest admits a tree_sha-only "
                          "difference on this stager's own manifest)")
        why = manifest_replace_problem(published, man)
        if why:
            raise Refusal(f"{remote}: --replace-manifest refused — {why}")
        log(f"  {remote}: REPLACING this stager's manifest (tree_sha {str(published.get('tree_sha'))[:12]} -> {str(man.get('tree_sha'))[:12]}; "
            f"the same bytes re-staged under the landed sha) — --replace-manifest")
        vol.put(local, remote, force=True)
        got = vol.remote_sha(remote)
        if got != sha256_file(local):
            raise Refusal(f"{remote}: bytes differ after the replacing put ({got[:12]} vs {sha256_file(local)[:12]})")
        return {"outcome": "replaced_tree_sha", "remote": remote, "local": str(local), "replaced": {"tree_sha": published.get("tree_sha"),
                "written_at": published.get("written_at")}}
    log(f"  put {remote} (LAST)")
    vol.put(local, remote)
    got = vol.remote_sha(remote)
    if got != sha256_file(local):
        raise Refusal(f"{remote}: bytes differ after the put ({got[:12]} vs {sha256_file(local)[:12]})")
    return {"outcome": "put", "remote": remote, "sha256": got, "local": str(local)}


def stage_parent_files(parents: ParentsPlan, vol: VolumeCLI, *, dry: bool, log: Log) -> Dict[str, dict]:
    """Step 4: the parents' files through :func:`stage_files`; an existing ``staged_from.json`` is kept when it names the same object
    sha (its timestamp differs by construction) and refuses otherwise."""
    out: Dict[str, dict] = {}
    for p in parents.parents:
        log(f"-- parent {p.fit_model_key}/{p.cell_id} (sha {p.object_sha256[:12]}; consumed by {p.consumers}) -> {p.remote_dir}")
        plain = [f for f in p.files if f.name != STAGED_FROM_NAME]
        sf = [f for f in p.files if f.name == STAGED_FROM_NAME]
        out.update(stage_files(plain, vol, dry=dry, log=log, label=f"parent {p.cell_id}"))
        for f in sf:
            if dry:
                log(f"DRY  {vol.modal} volume put {vol.volume} {f.local} {f.remote}   (kept if present and naming sha {p.object_sha256[:12]})")
                out[f.remote] = {"outcome": "dry", "sha256": f.sha256, "bytes": f.bytes}
            elif vol.exists(f.remote_rel):
                # pulled INTO the local stage path: the kept record (its own written_at) becomes what the verification compares and
                # the receipt carries, so both describe the file that is actually on the volume, never the one this run drafted
                got = vol.get(f.remote, f.local.parent)
                assert got == f.local, (got, f.local)
                existing = load_json(f.local)
                if not isinstance(existing, dict) or existing.get("object_sha256") != p.object_sha256:
                    raise Refusal(f"parent {p.cell_id}: {f.remote} exists naming object_sha256 "
                                  f"{str((existing or {}).get('object_sha256'))[:12]}, not {p.object_sha256[:12]} — refusing")
                f.sha256, f.bytes, p.staged_from = sha256_file(f.local), f.local.stat().st_size, existing
                log(f"  exists naming the same object, kept: {f.remote} (written_at {existing.get('written_at')})")
                out[f.remote] = {"outcome": "kept_same_object", "sha256": f.sha256, "bytes": f.bytes}
            else:
                log(f"  put {f.remote}")
                vol.put(f.local, f.remote)
                out[f.remote] = {"outcome": "put", "sha256": f.sha256, "bytes": f.bytes}
    return out


# --------------------------------------------------------------------------------------------------------------------------- #
# receipts                                                                                                                      #
# --------------------------------------------------------------------------------------------------------------------------- #

def bank_receipt(bank: BankPlan, *, outcomes: Mapping[str, dict], verified: Mapping[str, dict], manifest: dict, manifest_result: dict,
                 registry: dict, inode_probe: dict, dry: bool, tree_info: Mapping[str, Any], started_at: str) -> dict:
    now = utc_now()
    return {"schema": RECEIPT_SCHEMA, "step": "bank", "dry": dry, "model_key": bank.model_key, "view": bank.view, "volume": VOLUME,
            "view_root": bank.remote_view_root, "objects_dir": f"{VOL_MOUNT}{bank.remote_view_root}", "tree": str(bank.tree),
            "tree_sha": bank.tree_sha, **dict(tree_info), "source": str(bank.source), "staged_by": STAGED_BY,
            "started_utc": started_at, "written_utc": now, "verified_utc": None if dry else now,
            "verified_by": f"{STAGED_BY} (per file: absent -> put, present -> get + sha256; view: ls names + get + sha256; manifest last)",
            "roster_sha256": bank.roster_sha256, "pins_sha256": bank.pins_sha256, "pins_compared": dict(bank.pins_compared),
            "cells": {c.cell_id: {"staged_id": c.staged_id, "lane_kind": c.lane_kind, "identity_sidecar": c.sidecar,
                                  "object_sha256": c.object_sha256, "object_bytes": c.object_bytes, "sidecar_sha256": c.sidecar_sha256,
                                  "source": str(c.source_dir), "volume_path": f"{VOLUME}:{c.remote_dir}",
                                  "parents": [{"role": p.role, "cell_id": p.cell_id, "model_key": p.model_key,
                                               "object_sha256": p.object_sha256, "staged_id": p.staged_id} for p in c.parents],
                                  DOSE_BLOCK: c.dose_source,
                                  "files": [{"name": f.name, "remote_path": f.remote, "sha256": f.sha256, "bytes": f.bytes,
                                             "outcome": outcomes.get(f.remote, {}).get("outcome"),
                                             "verification": (verified.get(f.remote) or {}).get("verified_by", "not verified (dry run)")}
                                            for f in c.files]} for c in bank.cells},
            "prompted_cells_not_staged": list(bank.prompted_cells_not_staged), "manifest_remote_path": manifest_result.get("remote"),
            "manifest_outcome": manifest_result.get("outcome"), "manifest": manifest, "registry_check": registry, "inode_probe": inode_probe}


def parents_receipt(parents: ParentsPlan, bank: BankPlan, *, outcomes: Mapping[str, dict], verified: Mapping[str, dict], registry: dict,
                    dry: bool, tree_info: Mapping[str, Any], started_at: str) -> dict:
    now = utc_now()
    roots = sorted({f"/{PARENTS_ROOT}/{p.fit_model_key}/cells_all" for p in parents.parents})
    return {"schema": RECEIPT_SCHEMA, "step": "parents", "dry": dry, "model_key": bank.model_key, "view": bank.view, "volume": VOLUME,
            "root": roots[0] if len(roots) == 1 else roots, "parents_root": f"{VOL_MOUNT}/{PARENTS_ROOT}", "tree": str(bank.tree),
            "tree_sha": bank.tree_sha, **dict(tree_info), "source": str(bank.source), "staged_by": STAGED_BY,
            "started_utc": started_at, "written_utc": now, "verified_utc": None if dry else now,
            "complete_marker": "not staged (train_budget._complete_marker_problems accepts an absent marker; the mirror's manifests files not staged here)",
            "parents": {p.cell_id: {"fit_model_key": p.fit_model_key, "staged_id": p.staged_id, "source": str(p.source_dir),
                                    "object_sha256": p.object_sha256, "budget_meta_sha256": p.budget_meta_sha256,
                                    "consumers": p.consumers, "volume_path": f"{VOLUME}:{p.remote_dir}", "staged_from": p.staged_from,
                                    "files": [{"name": f.name, "remote_path": f.remote, "sha256": f.sha256, "bytes": f.bytes,
                                               "outcome": outcomes.get(f.remote, {}).get("outcome"),
                                               "verification": (verified.get(f.remote) or {}).get("verified_by", "not verified (dry run)")}
                                              for f in p.files]} for p in parents.parents},
            "registry_check": registry}


def write_receipt(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------------------------------- #
# the run                                                                                                                       #
# --------------------------------------------------------------------------------------------------------------------------- #

def describe_plan(bank: BankPlan, parents: Optional[ParentsPlan], log: Log) -> None:
    log(f"plan {bank.model_key}/{bank.view}: {len(bank.cells)} cell(s) x 2 files -> {bank.remote_view_root}  (tree {bank.tree_sha[:12]}, "
        f"roster {bank.roster_sha256[:12]}, pins {bank.pins_sha256[:12]})")
    for c in bank.cells:
        log(f"  {c.cell_id}  <- {c.staged_id}  [{c.lane_kind}; {c.sidecar}]  sha {c.object_sha256[:12]}  {c.object_bytes} B"
            + (f"  parents: " + ", ".join(f"{p.role}={p.cell_id}@{p.object_sha256[:12]}->{p.staged_id}" for p in c.parents) if c.parents else "")
            + (f"  dose x{c.dose_source['scale']} of {c.dose_source['parent_staged_id']}@{c.dose_source['parent_object_sha256'][:12]}" if c.dose_source else ""))
    if bank.prompted_cells_not_staged:
        log(f"  prompted (served, no object, not staged): {bank.prompted_cells_not_staged}")
    if parents is not None:
        log(f"parents: {len(parents.parents)} dir(s) -> /{PARENTS_ROOT}/<fit model_key>/cells_all/<cell>/ (no COMPLETE)")
        for p in parents.parents:
            log(f"  {p.fit_model_key}/{p.cell_id}  <- {p.staged_id}  sha {p.object_sha256[:12]}  files {[f.name for f in p.files]}")


def run(*, tree: Path, model_key: str, view: str, source: Path, receipts_dir: Path, go: bool, parents: bool, tree_sha: str,
        tree_info: Optional[Mapping[str, Any]] = None, modal_bin: Path = DEFAULT_MODAL, stage_dir: Optional[Path] = None,
        registry_root: Optional[str] = None, inode_probe: Optional[Path] = None, floor_pct: float = 90.0,
        runner: Runner = default_runner, log: Log = print, replace_manifest: bool = False) -> Dict[str, Any]:
    """The whole stage. Raises :class:`Refusal` (no receipt) on any problem; returns the receipts written."""
    started = utc_now()
    dry = not go
    tree_info = dict(tree_info or {})
    stage_dir = Path(stage_dir) if stage_dir else DEFAULT_STAGE_ROOT / model_key / view / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stage_dir.mkdir(parents=True, exist_ok=True)
    log(f"mode: {'LIVE' if go else 'DRY'} — {model_key}/{view}, tree {tree} @ {tree_sha[:12]}, source {source}, stage {stage_dir}")
    if go and tree_info.get("config_dirty"):
        raise Refusal(f"the tree's roster/pins are modified against HEAD {tree_info['config_dirty']} — a manifest pins tree_sha, so stage "
                      "a committed roster + pins (commit, or pass --allow-dirty-config to record the dirty state knowingly)")
    if go and tree_info.get("tree_on_remote") is False:
        raise Refusal(f"tree sha {tree_sha} is on no remote branch — push first (or --allow-unpushed-tree); a staged manifest must name a reachable tree")

    # ---- the plan: every local check, before any Modal call -------------------------------------------------------------------
    bank = plan_bank(tree, model_key, view, source, tree_sha=tree_sha)
    pins = load_pins(tree, model_key)
    pplan = plan_parents(bank, pins=pins, stage_dir=stage_dir, written_at=started) if parents else None
    describe_plan(bank, pplan, log)
    registry = registry_check(registry_objects_of(bank, pplan), registry_root=registry_root)
    log(f"registry check: {registry['status']} — {registry['detail']}")
    if registry["status"] == "mismatch":
        raise Refusal(f"the registry's object rows disagree with the pins: {registry['detail']}")

    vol = VolumeCLI(modal_bin, VOLUME, runner=runner, log=log)
    # ---- 0. the inode floor -------------------------------------------------------------------------------------------------
    if inode_probe is None:
        probe_rec = {"status": "not_run", "detail": "no --inode-probe script given (this module adds no Modal function; the b64 lane's "
                                                    "w2_volume_probe.py satisfies the contract)", "floor_pct": floor_pct}
        log(("WARNING: " if go else "DRY  ") + f"inode floor probe not run ({probe_rec['detail']})")
    elif dry:
        probe_rec = {"status": "dry", "script": str(inode_probe), "floor_pct": floor_pct}
        log(f"DRY  {vol.modal} run {inode_probe} --floor-pct {floor_pct}   (exit 3 = at/above the floor = refuse)")
    else:
        probe_rec = vol.inode_probe(Path(inode_probe), floor_pct)
        probe_rec["script"] = str(inode_probe)
        log(f"inode probe: {json.dumps(probe_rec)}")
    n_inodes = len(bank.cells) + len(bank.files) + 1 + ((len(pplan.parents) + len(pplan.files)) if pplan else 0)
    log(f"inode delta if everything is new: {len(bank.cells)} cell dirs + {len(bank.files)} files + 1 manifest"
        + (f" + {len(pplan.parents)} parent dirs + {len(pplan.files)} files" if pplan else "") + f" = {n_inodes} inodes")

    # ---- 1-3. the view -------------------------------------------------------------------------------------------------------
    log(f"== view {bank.view} -> {bank.remote_view_root}")
    outcomes = stage_files(bank.files, vol, dry=dry, log=log, label=f"view {bank.view}")
    verified: Dict[str, dict] = {}
    if not dry:
        verified = verify_remote({c.remote_dir: c.files for c in bank.cells}, vol, log=log, label=f"view {bank.view}")
    else:
        log(f"DRY  verify {bank.remote_view_root}: ls --json per cell dir (exactly the two names) + get + sha256 per file; THEN the manifest")
    manifest = build_manifest(bank, verified=verified, dry=dry, written_at=None if dry else utc_now())
    manifest_result = publish_manifest(manifest, vol, remote_dir=bank.remote_view_root, stage_dir=stage_dir, dry=dry, log=log,
                                       replace_manifest=replace_manifest)

    # ---- 4. the parents ------------------------------------------------------------------------------------------------------
    p_outcomes: Dict[str, dict] = {}
    p_verified: Dict[str, dict] = {}
    if pplan is not None:
        log(f"== parents -> /{PARENTS_ROOT}/<fit model_key>/cells_all/")
        p_outcomes = stage_parent_files(pplan, vol, dry=dry, log=log)
        if not dry:
            p_verified = verify_remote({p.remote_dir: p.files for p in pplan.parents}, vol, log=log, label="parents", exact_listing=False)
        else:
            log("DRY  verify each parent dir: ls --json (planned names present) + get + sha256 per file")

    # ---- 5. receipts (only now) -----------------------------------------------------------------------------------------------
    receipts_dir = Path(receipts_dir)
    if dry:
        # a dry run's receipts live beside the live ones, never in their place — `<receipts-dir>/dry/` (the launch-set
        # prerequisite check reads `<receipts-dir>/bank.json`, which only a LIVE, fully verified stage may write)
        receipts_dir = receipts_dir / "dry"
        log(f"DRY  receipts -> {receipts_dir} (the live receipts in {receipts_dir.parent} are not touched)")
    out: Dict[str, Any] = {"bank": str(receipts_dir / "bank.json"), "dry": dry, "stage_dir": str(stage_dir), "receipts_dir": str(receipts_dir)}
    write_receipt(receipts_dir / "bank.json", bank_receipt(bank, outcomes=outcomes, verified=verified, manifest=manifest, manifest_result=manifest_result,
                                                           registry=registry, inode_probe=probe_rec, dry=dry, tree_info=tree_info, started_at=started))
    if pplan is not None:
        out["parents"] = str(receipts_dir / "parents.json")
        write_receipt(receipts_dir / "parents.json", parents_receipt(pplan, bank, outcomes=p_outcomes, verified=p_verified, registry=registry,
                                                                     dry=dry, tree_info=tree_info, started_at=started))
    log(f"DONE ({'LIVE' if go else 'DRY — nothing put; re-run with --go'}): receipts {sorted(v for k, v in out.items() if k in ('bank', 'parents'))}")
    return out


# --------------------------------------------------------------------------------------------------------------------------- #
# the CLI                                                                                                                       #
# --------------------------------------------------------------------------------------------------------------------------- #

def git_tree_info(tree: Path) -> Dict[str, Any]:
    """``tree_sha`` (HEAD), whether the roster/pins are modified against it, and whether HEAD is on a remote branch."""
    def git(*a: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(["git", "-C", str(tree), *a], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise Refusal(f"{tree}: not a git tree ({head.stderr.strip()[-200:]}); pass --tree-sha to stage from a non-git tree")
    dirty = git("status", "--porcelain", "--", ROSTER_REL, PINS_REL)
    remote = git("branch", "-r", "--contains", head.stdout.strip())
    return {"tree_sha": head.stdout.strip(), "tree_sha_source": "git rev-parse HEAD",
            "config_dirty": [l.strip() for l in dirty.stdout.splitlines() if l.strip()],
            "tree_on_remote": bool(remote.stdout.strip()) if remote.returncode == 0 else None,
            "tree_remote_branches": [b.strip() for b in remote.stdout.splitlines() if b.strip()][:8]}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, help="the roster target / pins key (e.g. qwen35_4b)")
    ap.add_argument("--view", required=True, help="the roster view under targets.<model-key>.views (e.g. s1)")
    ap.add_argument("--tree", type=Path, default=REPO_LOCAL, help=f"the repo tree whose roster + pins define the plan (default {REPO_LOCAL})")
    ap.add_argument("--source", type=Path, default=None, help="the grading host's mirror root holding <staged_id>/ dirs (default $STAGE/sdfbase/full22_objects/<model-key>/cells, $STAGE being the site's staging directory)")
    ap.add_argument("--receipts-dir", type=Path, required=True, help="where bank.json (+ parents.json) are written on a fully verified stage")
    ap.add_argument("--parents", action="store_true", help="also stage every derived cell's parents under /out/<fit model_key>/cells_all/")
    ap.add_argument("--go", action="store_true", help="LIVE: put to the volume (default DRY: nothing remote, every command printed)")
    ap.add_argument("--modal", type=Path, default=DEFAULT_MODAL, help="the Modal CLI (default ~/venvs/modal/bin/modal)")
    ap.add_argument("--stage-dir", type=Path, default=None, help="where the manifest / staged_from.json are written locally before a put")
    ap.add_argument("--registry-root", default=None, help="the registry root holding registry.sqlite (default $REGISTRY_ROOT or ~/registry)")
    ap.add_argument("--inode-probe", type=Path, default=None, help="OPT-IN: a `modal run` script printing the volume's inode JSON line (exit 3 = at/above the floor)")
    ap.add_argument("--floor-pct", type=float, default=90.0, help="the inode floor the probe refuses at (default 90)")
    ap.add_argument("--tree-sha", default=None, help="override the tree sha (a non-git tree, tests); recorded as such in the receipt")
    ap.add_argument("--allow-dirty-config", action="store_true", help="LIVE only: stage although roster/pins are modified against HEAD (recorded)")
    ap.add_argument("--allow-unpushed-tree", action="store_true", help="LIVE only: stage although HEAD is on no remote branch (recorded)")
    ap.add_argument("--replace-manifest", action="store_true",
                    help="LIVE only: replace a published staged_manifest.json THIS stager wrote when the ONLY difference is tree_sha (the same "
                         "bytes re-staged under a later landed sha, D1861); any other difference, or another lane's manifest, still refuses")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    log = lambda s: print(f"[stage-full22 {time.strftime('%H:%M:%SZ', time.gmtime())}] {s}", flush=True)  # noqa: E731
    try:
        tree = Path(a.tree).resolve()
        if a.tree_sha:
            info: Dict[str, Any] = {"tree_sha": a.tree_sha, "tree_sha_source": "--tree-sha argument", "config_dirty": [], "tree_on_remote": None}
        else:
            info = git_tree_info(tree)
        if a.allow_dirty_config and info.get("config_dirty"):
            info["config_dirty_allowed"] = list(info["config_dirty"]); info["config_dirty"] = []
        if a.allow_unpushed_tree and info.get("tree_on_remote") is False:
            info["tree_on_remote_allowed"] = True; info["tree_on_remote"] = None
        source = Path(a.source) if a.source else _site_path(DEFAULT_SOURCE_ROOT, "--source") / a.model_key / "cells"
        run(tree=tree, model_key=a.model_key, view=a.view, source=source, receipts_dir=Path(a.receipts_dir), go=a.go, parents=a.parents,
            tree_sha=info.pop("tree_sha"), tree_info=info, modal_bin=Path(a.modal), stage_dir=a.stage_dir, registry_root=a.registry_root,
            inode_probe=a.inode_probe, floor_pct=a.floor_pct, log=log, replace_manifest=bool(a.replace_manifest and a.go))
        return 0
    except Refusal as e:
        log(f"REFUSED: {e}")
        log("no receipt written")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
