#!/usr/bin/env python3
"""reproduce/tools/stage_demo_bank_modal.py — the worked example's object bank on the Modal volume, from the paper's COMMITTED objects.

    python reproduce/tools/stage_demo_bank_modal.py --model K [--view V] --receipts-dir DIR [--volume twosi-fits] [--environment ENV] [--modal BIN]
                                                    [--go] [--bank-root /objects/sdfbase/<key>] [--table T] [--repo R] [--registry-root R]

Stage 07 (reproduce/07_bcb_bfcl.sh --phase stage) serves a front-door key's phase views on Modal through the tinycap launcher, which serves
ONE bank and ONE explicit-cells config per registry ROW (contracts.MODELS[<slug>]; lib/models.py launcher_slug: the KEY's row for the one
steered wave, `<key>-<view>` for a dose view — the paper's dose rows' shape). The wave's row reads its bank at `objects_dir` =
/vol/objects/sdfbase/<key>; a dose view's row at /vol/objects/sdfbase/<key>/<view> — both OUTSIDE the full22 root, so the full22 stager's
layout (scripts/modal/stage_full22_bank_modal.py: /vol/objects/sdfbase/full22/<model_key>/<view>/, planned from the full22 roster + pins)
is not this bank's. This tool PLANS from objects/DEMO.json + reproduce/models.yaml instead (reproduce/tools/import_objects.py's Plan over the
view: one committed object per object cell — the paper's object on the wave, the paper's committed DOSE-SCALED COPY of it on a dose view
(sisteer.objects.resolve(model, label, dose=); the same bytes stage 09 imports into the cluster bank as `d<tok>__<cell_id>`), the launcher
token = lib/models.py roster_cell_id, the SAME cell id on the wave and on the dose view: the bank, not the token, carries the dose) and REUSES
the full22 stager's volume protocol (VolumeCLI, stage_files, verify_remote, the receipts): put-if-absent / get+sha / refuse-on-difference /
`ls --json` names / pull-back verify / the manifest LAST. Every volume command is scoped to the site's Modal ENVIRONMENT (`--environment`,
default $SISTEER_MODAL_ENVIRONMENT — the one 07 hands the launcher as `modal run -e`; a volume name is per environment, so a stage into the
CLI's default environment would be a bank the launcher never reads). Credentials come from the environment; nothing here reads, prints or
logs one.

The layout it writes::

    <bank root>/<token>/{object.pt, <the ONE identity sidecar>}          EXACTLY two files; never COMPLETE (the committed COMPLETE manifests
                                                                            metrics.jsonl, which is not staged; a present-but-incoherent marker refuses)
    <bank root>/staged_manifest.json                                       demo-import-staged-objects/1, put LAST
    /out/<parent model_key>/cells_all/<parent cell_id>/{object.pt, train_budget_meta.json}
                                                                            the four parents the two derived cells are held to (tinycap's
                                                                            derived_provenance reads DERIVED_PARENTS_ROOT = /vol/out); no COMPLETE
                                                                            (a dose copy inherits its parent's sidecar, so it names the same four)

`--bank-root` defaults to contracts.MODELS[<launcher slug>].objects_dir with the volume mount stripped (never a literal); a root under the
full22 root refuses (is_full22_row would turn on the full22 pins by cell). VERIFIED BEFORE ANY PUT, each a refusal by name: the committed
object's bytes hash to its object id (committed_object_bytes_path == objects/<dir>/object.pt); exactly ONE identity sidecar sits in its
directory and names the token (`cell_id` for a fit, `derived_id` for a derived object, the refusal direction's `cell_id` is the harness id);
the sidecar records the object's sha; configs/sdfbase_full22_pins.json pins every one of the objects AND the four parents under the PAPER key
(committed_objects_of) by object id — the seed-row staged ids — and a dose view's copy under the KEY's own block by its staged id
`d<tok>__<cell_id>` (the block 05 / 09 --step pins writes from the importer; the paper key pins the copy's PARENT, named by the copy's
`dose_scaled_from` stamp, which must also name the view's dose); each parent's mirror dir (objects/<model_id>/<parent sha>) holds object.pt
hashing to the derived sidecar's block and train_budget_meta.json hashing to the block's budget_meta_sha256 and naming the parent cell; the
registry's object rows (read-only; --registry-root) agree with the ids (an absent row or database is recorded `unverified`, never a block).
Default DRY: every remote command printed, no Modal call; `--go` puts. Receipts `<receipts-dir>/bank.json` + `parents.json` only after
everything verified; a dry run's receipts go under `<receipts-dir>/dry/` (07 passes `<state>/07_bank/<view>`; the preflight phase of a dose
view reads the live bank.json as the identity its derived hashes are held to).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FULL22_STAGER = REPO / "scripts" / "modal" / "stage_full22_bank_modal.py"
IMPORTER = HERE / "import_objects.py"
MANIFEST_SCHEMA = "demo-import-staged-objects/1"
RECEIPT_SCHEMA = "stage-demo-bank-receipt/1"
STAGED_BY = "reproduce/tools/stage_demo_bank_modal.py"
PARENT_FILES = ("object.pt", "train_budget_meta.json")
EXIT_REFUSED = 2


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod            # the full22 stager's dataclasses resolve their postponed annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


M = _load_by_path("_demo_bank_full22_stager", FULL22_STAGER)      # VolumeCLI, stage_files, verify_remote, PlannedFile, Refusal, sha256_file, …
IO = _load_by_path("_demo_bank_import_objects", IMPORTER)         # Plan (objects/DEMO.json + models.yaml -> the committed rows), ImportRefused
Refusal = M.Refusal


def _contracts(repo: Path):
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.capeval.tinycap import contracts as C  # noqa: WPS433 — the launcher's row + volume constants
    return C


def volume_rel(container_or_volume_path: str, *, mount: str = M.VOL_MOUNT) -> str:
    """A path as the ``modal volume`` CLI takes it (volume-root, leading slash): a container path under the mount (``/vol/...``) has the
    mount stripped by a CHECKED conversion; a volume-root path is returned as is; anything else refuses by name."""
    p = str(container_or_volume_path)
    if p == mount or p.startswith(mount + "/"):
        rest = p[len(mount):]
        return rest or "/"
    if p.startswith("/"):
        return p
    raise Refusal(f"{p!r} is neither a container path under the volume mount {mount!r} nor a volume-root path")


@dataclass
class DemoCell:
    view_cell: str
    token: str
    staged_id: str
    object_id: str
    source_dir: Path
    sidecar: str
    paper_label: str
    kind: str
    pinned_as: List[str]
    remote_dir: str
    files: List[Any]
    parents: List[dict] = field(default_factory=list)
    #: a DOSE view's copy: {scale, parent_object_sha256, parent_staged_id, parent_pinned_as} — the sidecar's dose_scaled_from stamp, bound to the pins
    dose: Optional[dict] = None


@dataclass
class DemoParent:
    cell_id: str
    model_key: str
    object_sha256: str
    budget_meta_sha256: str
    source_dir: Path
    pinned_as: List[str]
    consumers: List[str]
    remote_dir: str
    files: List[Any]


@dataclass
class DemoPlan:
    key: str
    view: str
    slug: str                   # the launcher's registry row serving the view (lib/models.py launcher_slug): the key, or <key>-<view> for a dose view
    bank_root: str              # volume-root (leading slash, no mount): /objects/sdfbase/<key>[/<view>]
    objects_dir: str            # the row's container path: /vol/objects/sdfbase/<key>[/<view>]
    paper_key: str
    cells: List[DemoCell]
    parents: List[DemoParent]
    prompted_cells_not_staged: List[str]
    table_sha256: str
    pins_sha256: str
    demo_sha256: str
    manifest_sha256: str

    @property
    def files(self) -> List[Any]:
        return [f for c in self.cells for f in c.files]

    @property
    def parent_files(self) -> List[Any]:
        return [f for p in self.parents for f in p.files]


def default_bank_root(key: str, repo: Path = REPO) -> str:
    C = _contracts(repo)
    try:
        row = C.model_row(key)
    except C.UnknownModel as e:
        raise Refusal(f"contracts.MODELS has no row for {key!r} ({e}) — the launcher cannot serve the key, so there is no bank to stage")
    return volume_rel(row.objects_dir)


def _pinned_as(pins_block: Mapping[str, Any], object_id: str) -> List[str]:
    return sorted(sid for sid, e in pins_block.items() if isinstance(e, dict) and e.get("object_sha256") == object_id)


def plan_demo_bank(key: str, view: Optional[str], *, repo: Path = REPO, table: Optional[str] = None, bank_root: Optional[str] = None) -> DemoPlan:
    """Every local check, before any Modal call; raises :class:`Refusal` on the first problem."""
    repo = Path(repo)
    C = _contracts(repo)
    MT = IO._load_models_module(repo)
    table_path = table or os.environ.get("SISTEER_MODELS_TABLE") or str(repo / "reproduce" / "models.yaml")
    try:
        m = MT.model(key, table_path)
        wave = view or MT.baseline_view(key, table_path)
        if wave not in MT.views(key, table_path):
            raise Refusal(f"{key}: {wave!r} is not a view of the key (views: {MT.views(key, table_path)})")
        plan = IO.Plan(key, views=[wave], table=table_path, repo=repo)
        launcher = dict(MT.launcher_cells(key, wave, path=table_path))
        slug = MT.launcher_slug(key, wave, table_path)
    except SystemExit as e:                 # ImportRefused / baseline_view's refusal: the table's own words
        raise Refusal(str(e)) from None
    except (KeyError, ValueError) as e:
        raise Refusal(f"{key}/{view}: {e}") from None
    paper_key = MT.committed_objects_of(m) or ""
    try:
        row = C.model_row(slug)
    except C.UnknownModel as e:
        raise Refusal(f"{key}/{wave}: contracts.MODELS has no row {slug!r} ({e}) — the launcher serves one bank per row, so a view is served from "
                      "its own row (lib/models.py launcher_slug); add the row (src/sisteer/capeval/tinycap/contracts.py) before staging its bank") from None
    root = bank_root or volume_rel(row.objects_dir)
    if not root.startswith("/") or root.startswith(M.VOL_MOUNT + "/"):
        raise Refusal(f"--bank-root {root!r} must be a volume-root path (leading slash, no {M.VOL_MOUNT} mount)")
    full22 = volume_rel(C.VOL_OBJECTS_FULL22)
    if root == full22 or root.startswith(full22 + "/"):
        raise Refusal(f"--bank-root {root!r} sits under the full22 root {full22} — the front door's bank is its own (is_full22_row would turn on "
                      "the full22 pins by cell and refuse the slug)")
    pins_all = M.load_json(repo / M.PINS_REL)
    pins_paper = pins_all.get(paper_key) if isinstance(pins_all, dict) else None
    if not isinstance(pins_paper, dict) or not pins_paper:
        raise Refusal(f"{M.PINS_REL} carries no block for the paper key {paper_key!r} (committed_objects_of) — the committed objects' pins cross-check "
                      "needs it (keys: " + ", ".join(sorted(pins_all)) + ")")
    manifest_by_id = plan.manifest_by_id
    cells: List[DemoCell] = []
    parents_by_key: Dict[tuple, DemoParent] = {}
    for r in plan.rows:
        tok = launcher.get(r["view_cell"])
        if tok is None:
            raise Refusal(f"{key}/{wave}/{r['view_cell']}: the models table maps it to no launcher token")
        src = repo / "objects" / r["dir"]
        obj = src / M.OBJECT_NAME
        if not obj.is_file():
            raise Refusal(f"{key}/{wave}/{tok}: {obj} is missing (a sparse checkout: `git sparse-checkout add objects/{r['dir']}`)")
        got = M.sha256_file(obj)
        oid = str(r["object_sha256_expected"])
        if got != oid:
            raise Refusal(f"{key}/{wave}/{tok}: objects/{r['dir']}/object.pt hashes to {got[:12]}, not the committed object id {oid[:12]}")
        present = M._identity_sidecars_present(src)
        if present != [r["identity_sidecar"]]:
            raise Refusal(f"{key}/{wave}/{tok}: identity sidecars present {present}; exactly one, {r['identity_sidecar']!r}, is staged")
        side = r["identity_sidecar"]
        meta = M.load_json(src / side)
        if not isinstance(meta, dict):
            raise Refusal(f"{key}/{wave}/{tok}: {side} is not a JSON object")
        if meta.get("object_sha256") != oid:
            raise Refusal(f"{key}/{wave}/{tok}: {side} records object_sha256 {str(meta.get('object_sha256'))[:12]} != the object id {oid[:12]}")
        named = meta.get("derived_id") if side == M.DERIVED_META else meta.get("cell_id")
        if str(named) != tok:
            raise Refusal(f"{key}/{wave}/{tok}: {side} names {'derived_id' if side == M.DERIVED_META else 'cell_id'}={named!r}, not the launcher token {tok!r}")
        dose: Optional[dict] = None
        if r.get("dose") is not None:
            # a DOSE view's committed copy (import_objects.Plan: the manifest's dose_scale row whose stamp names the DEMO parent at the view's
            # dose): pinned under the KEY's own block by its staged id (05 / 09 --step pins), its PARENT under the paper key; the sidecar's
            # dose_scaled_from stamp re-asserted here, so a copy of another parent or at another dose never reaches the view's bank
            stamp = meta.get(M.DOSE_BLOCK)
            if not isinstance(stamp, dict):
                raise Refusal(f"{key}/{wave}/{tok}: {side} carries no {M.DOSE_BLOCK!r} block — not a dose-scaled copy")
            parent = str(r["dose_parent"])
            if str(stamp.get("parent_object_sha256")) != parent or abs(float(stamp.get("scale") or 0.0) - float(r["dose"])) > 1e-9:
                raise Refusal(f"{key}/{wave}/{tok}: {side} dose_scaled_from names parent {str(stamp.get('parent_object_sha256'))[:12]} at scale "
                              f"{stamp.get('scale')!r}; the view's copy is of {parent[:12]} at {float(r['dose']):g}")
            pins_key = pins_all.get(key) if isinstance(pins_all, dict) else None
            own = (pins_key or {}).get(r["staged_id"]) if isinstance(pins_key, dict) else None
            if not isinstance(own, dict) or own.get("object_sha256") != oid:
                raise Refusal(f"{key}/{wave}/{tok}: {M.PINS_REL}[{key}] pins no {r['staged_id']!r} with object_sha256 {oid[:12]} (the copy's staged id; "
                              "09_dose_ladder.sh --step stage / import_objects.py pins writes it) — the committed copy is not the view's pinned one")
            parent_pinned = _pinned_as(pins_paper, parent)
            if not parent_pinned:
                raise Refusal(f"{key}/{wave}/{tok}: the copy's parent {parent[:12]} is pinned under no staged id of {M.PINS_REL}[{paper_key}] — "
                              "the copy is not scaled from the paper's object")
            pinned = [r["staged_id"]]
            dose = {"scale": float(r["dose"]), "parent_object_sha256": parent, "parent_staged_id": stamp.get("parent_staged_id"),
                    "parent_pinned_as": {paper_key: parent_pinned}, "producer": stamp.get("producer")}
        else:
            pinned = _pinned_as(pins_paper, oid)
            if not pinned:
                raise Refusal(f"{key}/{wave}/{tok}: {M.PINS_REL}[{paper_key}] pins no entry with object_sha256 {oid[:12]} (the seed-row staged ids) — "
                              "the committed object is not the paper's pinned one")
        rdir = f"{root}/{tok}"
        files = [M.PlannedFile(M.OBJECT_NAME, obj, f"{rdir}/{M.OBJECT_NAME}", got, obj.stat().st_size),
                 M.PlannedFile(side, src / side, f"{rdir}/{side}", M.sha256_file(src / side), (src / side).stat().st_size)]
        cell = DemoCell(view_cell=r["view_cell"], token=tok, staged_id=r["staged_id"], object_id=oid, source_dir=src, sidecar=side,
                        paper_label=r["paper_label"], kind=r["kind"], pinned_as=pinned, remote_dir=rdir, files=files, dose=dose)
        if side == M.DERIVED_META:
            blocks = M._parent_blocks(meta)
            if len(blocks) < 2:
                raise Refusal(f"{key}/{wave}/{tok}: {side} records {len(blocks)} parent block(s); a derived object has >= 2")
            for role, blk in blocks.items():
                missing = [f for f in ("cell_id", "model_key", "object_sha256", "budget_meta_sha256") if blk.get(f) in (None, "")]
                if missing:
                    raise Refusal(f"{key}/{wave}/{tok}: {side} {role} records no {missing}")
                psha, pcell, pmk = str(blk["object_sha256"]), str(blk["cell_id"]), str(blk["model_key"])
                prow = manifest_by_id.get(psha)
                if prow is None or not prow.get("dir"):
                    raise Refusal(f"{key}/{wave}/{tok}: parent {role} {pcell} ({psha[:12]}) is not in objects/MANIFEST.json")
                psrc = repo / "objects" / str(prow["dir"])
                pobj, ptbm = psrc / M.OBJECT_NAME, psrc / M.BUDGET_META
                if not pobj.is_file() or not ptbm.is_file():
                    raise Refusal(f"{key}/{wave}/{tok}: parent {pcell}: {psrc} lacks {M.OBJECT_NAME} and/or {M.BUDGET_META}")
                if M.sha256_file(pobj) != psha:
                    raise Refusal(f"{key}/{wave}/{tok}: parent {pcell}: objects/{prow['dir']}/object.pt does not hash to {psha[:12]}")
                tsha = M.sha256_file(ptbm)
                if tsha != str(blk["budget_meta_sha256"]):
                    raise Refusal(f"{key}/{wave}/{tok}: parent {pcell}: {M.BUDGET_META} hashes to {tsha[:12]}, not the derived sidecar's "
                                  f"budget_meta_sha256 {str(blk['budget_meta_sha256'])[:12]} — the parent's identity record changed under the derivation")
                tbm = M.load_json(ptbm)
                if not isinstance(tbm, dict) or tbm.get("cell_id") != pcell or tbm.get("object_sha256") != psha:
                    raise Refusal(f"{key}/{wave}/{tok}: parent {pcell}: {M.BUDGET_META} names cell_id={tbm.get('cell_id')!r} / object {str(tbm.get('object_sha256'))[:12]}")
                ppinned = _pinned_as(pins_paper, psha)
                if not ppinned:
                    raise Refusal(f"{key}/{wave}/{tok}: parent {pcell} ({psha[:12]}) is pinned under no staged id of {M.PINS_REL}[{paper_key}]")
                pk = (pmk, pcell)
                if pk in parents_by_key:
                    if parents_by_key[pk].object_sha256 != psha:
                        raise Refusal(f"parents: {tok} and {parents_by_key[pk].consumers} name {pcell} under {pmk} with different shas")
                    parents_by_key[pk].consumers.append(tok)
                else:
                    prdir = f"/{M.PARENTS_ROOT}/{pmk}/cells_all/{pcell}"
                    pfiles = [M.PlannedFile(M.OBJECT_NAME, pobj, f"{prdir}/{M.OBJECT_NAME}", psha, pobj.stat().st_size),
                              M.PlannedFile(M.BUDGET_META, ptbm, f"{prdir}/{M.BUDGET_META}", tsha, ptbm.stat().st_size)]
                    parents_by_key[pk] = DemoParent(cell_id=pcell, model_key=pmk, object_sha256=psha, budget_meta_sha256=tsha, source_dir=psrc,
                                                    pinned_as=ppinned, consumers=[tok], remote_dir=prdir, files=pfiles)
                cell.parents.append({"role": role, "cell_id": pcell, "model_key": pmk, "object_sha256": psha, "budget_meta_sha256": tsha})
        cells.append(cell)
    prompted = [c for c in launcher.values() if c != "none" and c.startswith(MT.NO_OBJECT_PREFIX)]
    return DemoPlan(key=key, view=wave, slug=slug, bank_root=root, objects_dir=f"{M.VOL_MOUNT}{root}", paper_key=paper_key, cells=cells,
                    parents=list(parents_by_key.values()), prompted_cells_not_staged=prompted,
                    table_sha256=M.sha256_file(Path(table_path)), pins_sha256=M.sha256_file(repo / M.PINS_REL),
                    demo_sha256=M.sha256_file(repo / IO.DEMO_REL), manifest_sha256=M.sha256_file(repo / IO.MANIFEST_REL))


def _pinned_as_block(plan: DemoPlan, c: DemoCell) -> Dict[str, List[str]]:
    """``{pins block: [staged ids]}`` the cell's object is pinned under: the paper key's (the wave's objects) or the KEY's own (a dose copy)."""
    return {plan.key: c.pinned_as} if c.dose else {plan.paper_key: c.pinned_as}


def build_manifest(plan: DemoPlan, *, verified: Mapping[str, Mapping[str, Any]], dry: bool, written_at: Optional[str]) -> dict:
    cells = {}
    for c in plan.cells:
        files = [{"path": f"{c.token}/{f.name}", "sha256": f.sha256, "bytes": f.bytes,
                  "verified_by": (verified.get(f.remote) or {}).get("verified_by", "not verified (dry run)")} for f in c.files]
        cells[c.token] = {"cell": c.token, "cell_id": c.token, "view_cell": c.view_cell, "paper_label": c.paper_label, "staged_id": c.staged_id,
                          "kind": c.kind, "object_id": c.object_id, "object_sha256": c.object_id, "bytes": c.files[0].bytes,
                          "identity_sidecar": c.sidecar, "sidecar_sha256": c.files[1].sha256, "source": f"committed:objects/{c.source_dir.relative_to(c.source_dir.parents[1])}",
                          "pinned_as": _pinned_as_block(plan, c), "volume_path": f"{plan.bank_root}/{c.token}", "objects_dir_entry": f"{plan.objects_dir}/{c.token}",
                          "n_files": len(files), "files": files, **({"parents": c.parents} if c.parents else {}), **({"dose": c.dose} if c.dose else {})}
    # The manifest's KEY SET is frozen: the battery's staged_manifest.json is already published on the volume
    # from earlier smokes and publish_manifest refuses a different manifest at the same path, so a key added here (`launcher_slug`, say)
    # would make the battery's stage phase refuse its own bank. The slug lives in the bank RECEIPT (bank.json) only; a dose view's manifest sits
    # at its own path (`<the key's bank>/<view>/`) and carries what it needs per CELL (`dose`, the key's pins block) — never a new top-level key.
    # tests/test_stage_demo_bank_modal.py pins the battery manifest's top-level and per-cell key sets.
    man = {"schema": MANIFEST_SCHEMA, "model_key": plan.key, "view": plan.view, "paper_key": plan.paper_key,
           "bank_source": "the paper's committed objects (objects/DEMO.json + objects/MANIFEST.json, sha-verified against the object ids and the paper key's pins)",
           "table": "reproduce/models.yaml", "table_sha256": plan.table_sha256, "demo": IO.DEMO_REL, "demo_sha256": plan.demo_sha256,
           "manifest": IO.MANIFEST_REL, "manifest_sha256": plan.manifest_sha256, "pins": M.PINS_REL, "pins_sha256": plan.pins_sha256,
           "volume_path": plan.bank_root, "objects_dir": plan.objects_dir, "manifest_name": M.MANIFEST_NAME, "staged_by": STAGED_BY,
           "remote_verified": {"method": "modal volume ls --json names per cell dir; modal volume get + sha256 per file",
                               "n_files": 0 if dry else len(plan.files), **({"dry": True} if dry else {})},
           "parents_root": f"{M.VOL_MOUNT}/{M.PARENTS_ROOT}",
           "parents": {f"{p.model_key}/{p.cell_id}": {"object_sha256": p.object_sha256, "budget_meta_sha256": p.budget_meta_sha256, "consumers": sorted(p.consumers),
                                                       "volume_path": p.remote_dir, "pinned_as": {plan.paper_key: p.pinned_as}} for p in plan.parents},
           "prompted_cells_not_staged": list(plan.prompted_cells_not_staged), "cells": cells}
    if written_at is not None:
        man["written_at"] = written_at
    return man


def publish_manifest(man: dict, vol, *, remote_dir: str, stage_dir: Path, dry: bool, log) -> dict:
    """The manifest LAST: an identical one already published (minus written_at) is kept; a different one refuses; else put + pulled back."""
    remote = f"{remote_dir}/{M.MANIFEST_NAME}"
    local_dir = Path(stage_dir) / remote_dir.lstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    local = local_dir / M.MANIFEST_NAME
    local.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if dry:
        log(f"DRY  write {local}; then {vol.modal} volume put {vol.volume} {local} {remote}   (LAST, after the bank verifies)")
        return {"outcome": "dry", "remote": remote, "local": str(local)}
    if vol.exists(remote.lstrip("/")):
        published = vol.remote_json(remote)
        if M.manifest_content(published) == M.manifest_content(man):
            log(f"  {remote}: an identical manifest is already published (written_at {published.get('written_at')}); nothing put")
            return {"outcome": "already_published", "remote": remote, "remote_written_at": published.get("written_at"), "local": str(local)}
        a, b = M.manifest_content(published), M.manifest_content(man)
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise Refusal(f"{remote}: a DIFFERENT {M.MANIFEST_NAME} is already published (staged_by {published.get('staged_by')!r}; differing keys {diff}) — "
                      "refusing to replace it; a human decides")
    log(f"  put {remote} (LAST)")
    vol.put(local, remote)
    got = vol.remote_sha(remote)
    if got != M.sha256_file(local):
        raise Refusal(f"{remote}: bytes differ after the put ({got[:12]} vs {M.sha256_file(local)[:12]})")
    return {"outcome": "put", "remote": remote, "sha256": got, "local": str(local)}


def describe(plan: DemoPlan, log) -> None:
    log(f"plan {plan.key}/{plan.view}: {len(plan.cells)} committed object(s) x 2 files -> {plan.bank_root} (objects_dir {plan.objects_dir}; launcher row "
        f"{plan.slug}; paper key {plan.paper_key}; table {plan.table_sha256[:12]}, pins {plan.pins_sha256[:12]}, DEMO {plan.demo_sha256[:12]})")
    for c in plan.cells:
        blk = plan.key if c.dose else plan.paper_key
        log(f"  {c.token}  <- {c.view_cell} ({c.paper_label}; {c.staged_id}) objects/{c.source_dir.relative_to(c.source_dir.parents[1])}  [{c.kind}; {c.sidecar}]  "
            f"sha {c.object_id[:12]}  {c.files[0].bytes} B  pinned as {blk}:{c.pinned_as}"
            + ((f"  dose {c.dose['scale']:g} copy of {c.dose['parent_object_sha256'][:12]} (pinned as {plan.paper_key}:{c.dose['parent_pinned_as'][plan.paper_key]})") if c.dose else "")
            + (("  parents: " + ", ".join(f"{p['role']}={p['cell_id']}@{p['object_sha256'][:12]}" for p in c.parents)) if c.parents else ""))
    if plan.prompted_cells_not_staged:
        log(f"  prompted (served, no object, not staged): {plan.prompted_cells_not_staged}")
    log(f"parents: {len(plan.parents)} dir(s) -> /{M.PARENTS_ROOT}/<parent model_key>/cells_all/<cell>/ ({'/'.join(PARENT_FILES)}; no COMPLETE)")
    for p in plan.parents:
        log(f"  {p.model_key}/{p.cell_id}  sha {p.object_sha256[:12]}  budget_meta {p.budget_meta_sha256[:12]}  consumed by {p.consumers}  pinned as {plan.paper_key}:{p.pinned_as}")


def receipt(plan: DemoPlan, *, step: str, outcomes: Mapping[str, dict], verified: Mapping[str, dict], manifest: Optional[dict], manifest_result: Optional[dict],
            registry: dict, dry: bool, volume: str, started_at: str, environment: Optional[str] = None) -> dict:
    now = M.utc_now()
    base = {"schema": RECEIPT_SCHEMA, "step": step, "dry": dry, "model_key": plan.key, "view": plan.view, "launcher_slug": plan.slug, "paper_key": plan.paper_key, "volume": volume,
            "environment": environment, "bank_root": plan.bank_root, "objects_dir": plan.objects_dir, "staged_by": STAGED_BY, "started_utc": started_at, "written_utc": now,
            "verified_utc": None if dry else now, "table_sha256": plan.table_sha256, "pins_sha256": plan.pins_sha256, "demo_sha256": plan.demo_sha256,
            "manifest_sha256": plan.manifest_sha256, "registry_check": registry}

    def files(fs):
        return [{"name": f.name, "remote_path": f.remote, "sha256": f.sha256, "bytes": f.bytes, "outcome": outcomes.get(f.remote, {}).get("outcome"),
                 "verification": (verified.get(f.remote) or {}).get("verified_by", "not verified (dry run)")} for f in fs]
    if step == "bank":
        return {**base, "cells": {c.token: {"view_cell": c.view_cell, "paper_label": c.paper_label, "staged_id": c.staged_id, "object_id": c.object_id,
                                            "identity_sidecar": c.sidecar, "source": str(c.source_dir), "pinned_as": _pinned_as_block(plan, c),
                                            "volume_path": f"{volume}:{c.remote_dir}", "parents": c.parents, "dose": c.dose, "files": files(c.files)} for c in plan.cells},
                "prompted_cells_not_staged": list(plan.prompted_cells_not_staged), "manifest_remote_path": (manifest_result or {}).get("remote"),
                "manifest_outcome": (manifest_result or {}).get("outcome"), "manifest": manifest}
    return {**base, "parents_root": f"{M.VOL_MOUNT}/{M.PARENTS_ROOT}",
            "complete_marker": "not staged (train_budget._complete_marker_problems accepts an absent marker; the committed COMPLETE manifests metrics.jsonl, which is not staged)",
            "parents": {f"{p.model_key}/{p.cell_id}": {"object_sha256": p.object_sha256, "budget_meta_sha256": p.budget_meta_sha256, "source": str(p.source_dir),
                                                         "consumers": sorted(p.consumers), "pinned_as": {plan.paper_key: p.pinned_as}, "volume_path": f"{volume}:{p.remote_dir}",
                                                         "files": files(p.files)} for p in plan.parents}}


def run(*, key: str, view: Optional[str], receipts_dir: Path, go: bool, volume: str, modal_bin: Path, repo: Path = REPO, table: Optional[str] = None,
        bank_root: Optional[str] = None, stage_dir: Optional[Path] = None, registry_root: Optional[str] = None, environment: Optional[str] = None,
        runner=M.default_runner, log=print) -> Dict[str, Any]:
    started = M.utc_now()
    dry = not go
    environment = environment or None
    plan = plan_demo_bank(key, view, repo=repo, table=table, bank_root=bank_root)
    stage_dir = Path(stage_dir) if stage_dir else Path(tempfile.gettempdir()) / "stage-demo-bank" / key / plan.view / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stage_dir.mkdir(parents=True, exist_ok=True)
    log(f"mode: {'LIVE' if go else 'DRY'} — {key}/{plan.view}, repo {repo}, volume {volume} (Modal environment {environment or 'the CLI default'}), stage {stage_dir}")
    describe(plan, log)
    # the registry's parent_object_ids: a derivation's parents; for a DOSE copy its ONE dose source (derivation dose_scale — import_objects.py
    # registry states it so), while the copy's inherited sidecar carries the parent's fit lineage, bound to the pins above, not to its row
    objects = {c.object_id: {"staged_id": c.staged_id, "bytes": c.files[0].bytes,
                             "parents": ([c.dose["parent_object_sha256"]] if c.dose else ([p["object_sha256"] for p in c.parents] or None))} for c in plan.cells}
    for p in plan.parents:
        objects.setdefault(p.object_sha256, {"staged_id": p.pinned_as[0] if p.pinned_as else p.cell_id, "bytes": p.files[0].bytes, "parents": None})
    registry = M.registry_check(objects, registry_root=registry_root)
    log(f"registry check: {registry['status']} — {registry['detail']}")
    if registry["status"] == "mismatch":
        raise Refusal(f"the registry's object rows disagree with the committed objects: {registry['detail']}")
    vol = M.VolumeCLI(modal_bin, volume, runner=runner, log=log, environment=environment)      # every ls/get/put scoped to the site's environment
    for f in plan.files + plan.parent_files:
        assert f.remote.startswith("/") and not f.remote.startswith(M.VOL_MOUNT + "/"), f.remote      # volume-root paths only reach the CLI
    log(f"== bank {plan.bank_root}")
    outcomes = M.stage_files(plan.files, vol, dry=dry, log=log, label=f"bank {plan.key}/{plan.view}")
    verified: Dict[str, dict] = {}
    if not dry:
        verified = M.verify_remote({c.remote_dir: c.files for c in plan.cells}, vol, log=log, label=f"bank {plan.key}/{plan.view}")
    else:
        log(f"DRY  verify {plan.bank_root}: ls --json per cell dir (exactly the two names) + get + sha256 per file; THEN the manifest")
    log(f"== parents -> /{M.PARENTS_ROOT}/<parent model_key>/cells_all/")
    p_outcomes = M.stage_files(plan.parent_files, vol, dry=dry, log=log, label="parents")
    p_verified: Dict[str, dict] = {}
    if not dry:
        p_verified = M.verify_remote({p.remote_dir: p.files for p in plan.parents}, vol, log=log, label="parents", exact_listing=False)
    else:
        log("DRY  verify each parent dir: ls --json (the two planned names present) + get + sha256 per file")
    manifest = build_manifest(plan, verified=verified, dry=dry, written_at=None if dry else M.utc_now())
    manifest_result = publish_manifest(manifest, vol, remote_dir=plan.bank_root, stage_dir=stage_dir, dry=dry, log=log)
    receipts_dir = Path(receipts_dir)
    if dry:
        receipts_dir = receipts_dir / "dry"
        log(f"DRY  receipts -> {receipts_dir} (the live receipts in {receipts_dir.parent} are not touched)")
    out = {"bank": str(receipts_dir / "bank.json"), "parents": str(receipts_dir / "parents.json"), "dry": dry, "stage_dir": str(stage_dir), "receipts_dir": str(receipts_dir),
           "bank_root": plan.bank_root, "objects_dir": plan.objects_dir, "environment": environment}
    M.write_receipt(receipts_dir / "bank.json", receipt(plan, step="bank", outcomes=outcomes, verified=verified, manifest=manifest, manifest_result=manifest_result,
                                                        registry=registry, dry=dry, volume=volume, started_at=started, environment=environment))
    M.write_receipt(receipts_dir / "parents.json", receipt(plan, step="parents", outcomes=p_outcomes, verified=p_verified, manifest=None, manifest_result=None,
                                                           registry=registry, dry=dry, volume=volume, started_at=started, environment=environment))
    log(f"DONE ({'LIVE' if go else 'DRY — nothing put; re-run with --go'}): receipts {out['bank']}, {out['parents']}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the models-table key whose row declares committed_objects_of (reproduce/models.yaml)")
    ap.add_argument("--view", default=None, help="the phase view: the key's one steered wave (default; lib/models.py baseline_view) or a dose view, whose "
                                                 "committed dose copies go to the view's own row's bank (launcher_slug <key>-<view>)")
    ap.add_argument("--receipts-dir", type=Path, required=True, help="where bank.json + parents.json are written on a fully verified stage (<state>/07_bank)")
    ap.add_argument("--volume", default=None, help="the Modal volume (default $SISTEER_MODAL_VOLUME, else contracts.MODAL_VOLUME)")
    ap.add_argument("--environment", default=None, help="the Modal environment every volume command is scoped to (`--env`; default $SISTEER_MODAL_ENVIRONMENT, "
                                                       "else the CLI's own resolution) — the one the launcher runs under")
    ap.add_argument("--modal", default=str(M.DEFAULT_MODAL), help=f"the Modal CLI (default {M.DEFAULT_MODAL}; `<python> -m modal` is accepted)")
    ap.add_argument("--bank-root", default=None, help="the volume-root bank path (default contracts.MODELS[<launcher slug>].objects_dir minus the mount)")
    ap.add_argument("--table", default=None, help="the models table (default reproduce/models.yaml / $SISTEER_MODELS_TABLE)")
    ap.add_argument("--repo", default=str(REPO), help="the checkout holding objects/ (default: this file's)")
    ap.add_argument("--stage-dir", type=Path, default=None, help="where the manifest is written locally before a put")
    ap.add_argument("--registry-root", default=None, help="the registry root the read-only object cross-check reads (default $REGISTRY_ROOT or the grading host default)")
    ap.add_argument("--go", action="store_true", help="LIVE: put to the volume (default DRY: nothing remote, every command printed)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    log = lambda s: print(f"[stage-demo-bank {time.strftime('%H:%M:%SZ', time.gmtime())}] {s}", flush=True)  # noqa: E731
    repo = Path(a.repo).resolve()
    try:
        volume = a.volume or os.environ.get("SISTEER_MODAL_VOLUME") or _contracts(repo).MODAL_VOLUME
        environment = a.environment or os.environ.get("SISTEER_MODAL_ENVIRONMENT") or None
        parts = str(a.modal).split()
        runner = M.default_runner
        if len(parts) > 1:
            prefix = parts
            runner = lambda argv, **kw: M.default_runner([*prefix, *list(argv)[1:]], **kw)  # noqa: E731 — `<python> -m modal` as one CLI
        run(key=a.model, view=a.view, receipts_dir=Path(a.receipts_dir), go=a.go, volume=volume, modal_bin=Path(parts[0]), repo=repo, table=a.table,
            bank_root=a.bank_root, stage_dir=a.stage_dir, registry_root=a.registry_root, environment=environment, runner=runner, log=log)
        return 0
    except Refusal as e:
        log(f"REFUSED: {e}")
        log("no receipt written")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
