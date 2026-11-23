#!/usr/bin/env python
"""stage_lane_bank_modal.py — stage ONE lane view of the Qwen3.5-4B bank (a view with its OWN roster and pins: the
control-pool-stopped ``ctlstop64s<k>`` or the SI-distilled refit ``sikl4s<k>``, target ``qwen35_4b``) on the Modal volume
``twosi-fits``, in the full22 bank layout the tinycap harness reads, with the parents its derived cells are held to.

    python scripts/modal/stage_lane_bank_modal.py --view sikl4s1 --source <dir of sikl4s1__<cell>/ dirs> \\
        --receipts-dir <dir> [--environment <modal env>] [--go]
    python scripts/modal/stage_lane_bank_modal.py --view ctlstop64s1 --source <dir of ctlstop64s1__<cell>/ dirs> \\
        --parents-source <dir of base64s1__<cell>/ dirs> --receipts-dir <dir> [--environment <modal env>] [--go]

THE LANES (:data:`LANES`, one row per view family; ``contracts.bcbw_view_sources`` names the same roster + pins pair, which
tests/test_sikl4_bcbw_lane.py holds equal):

- ``ctlstop64s<k>`` (added 2026-11-14): the reseed_ctlstop campaign's control-pool-stopped objects. Two of the four cells
  are DIFFERENCES whose two parents carry DIFFERENT fit model keys by construction (the constitution fit under
  ``qwen35_4b_base_eb64_s<k>``, pinned in the production pins as ``base64s<k>__<cell>``; the control-pool-stopped control fit under
  ``qwen35_4b_base_eb64_s<k>_ctlstop``, pinned in the lane pins as ``ctlstop64s<k>__<cell>``), which the production stager refuses, so
  the parents are staged too and ``--parents-source`` is required.
- ``sikl4s<k>`` (added 2026-11-15): the six SI-distilled refit objects of fit seed k, every one a FITTED object
  (identity sidecar ``train_budget_meta.json``): no derived cell, no parent, no ``--parents-source``. A derived pin in this lane refuses.

WHY NOT ``stage_full22_bank_modal.py``: that stager plans from the PRODUCTION roster and pins only, and its lanes and parent rule are the
production ones; these views live in their own roster (cells in the cross-bank pin form ``{pins, pin}``) and pins. This module re-plans
for that shape and reuses the production stager's remote protocol unchanged (``VolumeCLI``, ``stage_files``, ``verify_remote``,
``publish_manifest``, ``stage_parent_files``, the read-only registry check, the receipts).

LAYOUT (the harness's ``contracts.full22_objects_dir`` / ``train_budget.parent_cell_dir`` spellings)::

    /objects/sdfbase/full22/qwen35_4b/<view>/<cell_id>/{object.pt, <identity sidecar>}      exactly two files
    /objects/sdfbase/full22/qwen35_4b/<view>/staged_manifest.json                            put LAST
    /out/<fit model_key>/cells_all/<parent cell_id>/{object.pt, train_budget_meta.json[, metrics.jsonl], staged_from.json}   (ctlstop only)

VERIFIED BEFORE ANY PUT: each object's sha256 == its pin; exactly one identity sidecar, the one the pin names; the sidecar records the
pinned sha and names the cell; each derived cell's parent blocks name a pinned object and, under the parents step, each parent's mirror
object, ``train_budget_meta.json`` (its own sha == the derived sidecar's ``budget_meta_sha256``) and cell id agree; the lane roster and
pins are unmodified against the staged tree's HEAD. Default is DRY; ``--go`` puts. Credentials come from the environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import yaml

HERE = Path(__file__).resolve().parent


def _load_stager():
    spec = importlib.util.spec_from_file_location("_stage_full22_bank_modal", HERE / "stage_full22_bank_modal.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_stager()
Refusal = S.Refusal

MODEL_KEY = "qwen35_4b"
PROD_PINS_REL = S.PINS_REL
STAGED_BY = "scripts/modal/stage_lane_bank_modal.py"
CTLSTOP_FIT_SUFFIX = "_ctlstop"


class Lane(NamedTuple):
    """One view family with its own roster + pins. ``parents``: whether the lane pins derived cells whose parents are staged under
    ``/out`` (``--parents-source`` then required); ``wave``: the staged manifest's ``wave`` field."""
    name: str
    roster_rel: str
    pins_rel: str
    wave: str
    parents: bool


#: view family prefix -> lane (a view is ``<prefix><k>``, k in 1..3)
LANES: Mapping[str, Lane] = {
    "ctlstop64s": Lane("ctlstop", "configs/sdfbase_full22_roster_ctlstop_eb64.yaml", "configs/ctlstop_eb64_pins.json",
                       "full22-ctlstop", True),
    "sikl4s": Lane("sikl4", "configs/sdfbase_full22_roster_sikl4.yaml", "configs/sikl4_pins.json", "full22-sikl4", False),
}
_VIEW_RE = re.compile(r"^(?P<prefix>" + "|".join(map(re.escape, LANES)) + r")(?P<k>[1-3])$")


def lane_of(view: str) -> Lane:
    m = _VIEW_RE.match(str(view))
    if not m:
        raise Refusal(f"view {view!r} is not a lane view ({' | '.join(p + '<1-3>' for p in LANES)})")
    return LANES[m.group("prefix")]


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise Refusal(f"{path}: unreadable YAML ({e})") from None


def view_cells(tree: Path, view: str) -> Dict[str, Tuple[str, str]]:
    """``{cell_id: (pins file, pin key)}`` of the lane roster's view — every cell in the pin-mapping form."""
    lane = lane_of(view)
    roster = load_yaml(Path(tree) / lane.roster_rel)
    try:
        cells = roster["targets"][MODEL_KEY]["views"][view]["cells"]
    except (KeyError, TypeError):
        raise Refusal(f"{lane.roster_rel}: no targets.{MODEL_KEY}.views.{view}.cells") from None
    out: Dict[str, Tuple[str, str]] = {}
    for cid, ent in cells.items():
        if not isinstance(ent, dict) or set(ent) != {"pins", "pin"}:
            raise Refusal(f"{lane.roster_rel} {view}/{cid}: {ent!r} is not the pin-mapping form {{pins, pin}}")
        out[str(cid)] = (str(ent["pins"]), str(ent["pin"]))
    if not out:
        raise Refusal(f"{lane.roster_rel} {view}: no cells")
    return out


def pin_record(tree: Path, view: str, pins_rel: str, pin_key: str) -> Dict[str, Any]:
    lane = lane_of(view)
    if pins_rel != lane.pins_rel:
        raise Refusal(f"pin {pin_key!r} reads {pins_rel!r}; the {lane.name} lane admits only {lane.pins_rel!r}")
    mk, _, staged_id = pin_key.partition(".")
    pins = S.load_json(Path(tree) / pins_rel)
    rec = (pins.get(mk) or {}).get(staged_id) if isinstance(pins, dict) else None
    if not isinstance(rec, dict) or len(str(rec.get("object_sha256") or "")) != 64:
        raise Refusal(f"{pins_rel}[{mk}][{staged_id}] names no object_sha256")
    if rec.get("staged_id") != staged_id:
        raise Refusal(f"{pins_rel}[{mk}][{staged_id}] records staged_id {rec.get('staged_id')!r}")
    return rec


def parent_pin(tree: Path, block: Mapping[str, Any], view: str, *, where: str) -> Tuple[str, str, str]:
    """``(pins file, staged id, sha)`` of one parent block of a ctlstop difference: a ``…_ctlstop`` fit key reads the ctlstop pins as
    ``ctlstop64s<k>__<cell>``, any other the production pins as ``base64s<k>__<cell>``; the sha must be the block's."""
    lane = lane_of(view)
    if not lane.parents:
        raise Refusal(f"{where}: the {lane.name} lane stages no parents")
    cell, want, fit_mk = str(block.get("cell_id")), str(block.get("object_sha256")), str(block.get("model_key"))
    k = view[-1]
    if fit_mk.endswith(CTLSTOP_FIT_SUFFIX):
        rel, sid = lane.pins_rel, f"ctlstop64s{k}__{cell}"
    else:
        rel, sid = PROD_PINS_REL, f"base64s{k}__{cell}"
    pins = S.load_json(Path(tree) / rel)
    got = ((pins.get(MODEL_KEY) or {}).get(sid) or {}).get("object_sha256") if isinstance(pins, dict) else None
    if got != want:
        raise Refusal(f"{where}: parent {fit_mk}/{cell} sha {want[:12]} is not {rel}[{MODEL_KEY}][{sid}] ({str(got)[:12]})")
    return rel, sid, want


def plan_cell(tree: Path, view: str, cell_id: str, pins_rel: str, pin_key: str, source: Path) -> "S.PlannedCell":
    lane = lane_of(view)
    rec = pin_record(tree, view, pins_rel, pin_key)
    staged_id = rec["staged_id"]
    where = f"{MODEL_KEY}/{view}/{cell_id} ({staged_id})"
    src = Path(source) / staged_id
    obj = src / S.OBJECT_NAME
    if not obj.is_file():
        raise Refusal(f"{where}: {obj} is missing")
    sidecar = str(rec.get("identity_sidecar") or "")
    present = S._identity_sidecars_present(src)
    if present != [sidecar]:
        raise Refusal(f"{where}: identity sidecars present {present}; the pin names exactly [{sidecar!r}]")
    want = rec["object_sha256"]
    got = S.sha256_file(obj)
    if got != want:
        raise Refusal(f"{where}: object sha {got[:12]} != pin {want[:12]}")
    meta = S.load_json(src / sidecar)
    if not isinstance(meta, dict) or meta.get("object_sha256") not in (None, want):
        raise Refusal(f"{where}: {sidecar} records object_sha256 {str((meta or {}).get('object_sha256'))[:12]} != pin {want[:12]}")
    id_key = "derived_id" if sidecar == S.DERIVED_META else "cell_id"
    if meta.get(id_key) != cell_id:
        raise Refusal(f"{where}: {sidecar} names {id_key}={meta.get(id_key)!r}, not {cell_id!r}")
    if sidecar == S.DERIVED_META and not lane.parents:
        raise Refusal(f"{where}: a derived object ({sidecar}) in the {lane.name} lane, which pins fitted objects only and stages no parents")
    kind = "base" if sidecar == S.BUDGET_META else ("paired" if str(rec.get("kind")) == "psr_paired_write" else "derived")
    parents: List[Any] = []
    if sidecar == S.DERIVED_META:
        blocks = S._parent_blocks(meta)
        if sorted(blocks) != ["parent_neg", "parent_pos"]:
            raise Refusal(f"{where}: parent blocks {sorted(blocks)} != [parent_neg, parent_pos]")
        for role, block in sorted(blocks.items()):
            missing = [f for f in S.PARENT_FIELDS if block.get(f) in (None, "")]
            if missing:
                raise Refusal(f"{where}: {role} records no {missing}")
            _rel, psid, _sha = parent_pin(tree, block, view, where=where)
            parents.append(S.ParentRef(role=role, cell_id=str(block["cell_id"]), model_key=str(block["model_key"]),
                                       object_sha256=str(block["object_sha256"]), budget_meta_sha256=block.get("budget_meta_sha256"),
                                       staged_id=psid))
        if not any(p.model_key.endswith(CTLSTOP_FIT_SUFFIX) for p in parents):
            raise Refusal(f"{where}: no parent is a control-pool-stopped fit ({[p.model_key for p in parents]})")
    rdir = f"{S.remote_view_root(MODEL_KEY, view)}/{cell_id}"
    files = [S.PlannedFile(S.OBJECT_NAME, obj, f"{rdir}/{S.OBJECT_NAME}", got, obj.stat().st_size),
             S.PlannedFile(sidecar, src / sidecar, f"{rdir}/{sidecar}", S.sha256_file(src / sidecar), (src / sidecar).stat().st_size)]
    return S.PlannedCell(cell_id=cell_id, staged_id=staged_id, lane_kind=kind, sidecar=sidecar, source_dir=src, object_sha256=got,
                         object_bytes=files[0].bytes, sidecar_sha256=files[1].sha256, sidecar_bytes=files[1].bytes, remote_dir=rdir,
                         files=files, parents=parents)


def plan_bank(tree: Path, view: str, source: Path, *, tree_sha: str) -> "S.BankPlan":
    lane = lane_of(view)
    cells = [plan_cell(tree, view, cid, rel, key, source) for cid, (rel, key) in view_cells(tree, view).items()]
    return S.BankPlan(model_key=MODEL_KEY, view=view, tree=Path(tree), tree_sha=tree_sha, source=Path(source),
                      remote_view_root=S.remote_view_root(MODEL_KEY, view), cells=cells, prompted_cells_not_staged=[],
                      roster_sha256=S.sha256_file(Path(tree) / lane.roster_rel), pins_sha256=S.sha256_file(Path(tree) / lane.pins_rel),
                      pins_compared={c.staged_id: c.object_sha256 for c in cells})


def plan_parents(bank: "S.BankPlan", *, source: Path, parents_source: Path, stage_dir: Path, written_at: str) -> "S.ParentsPlan":
    """Every derived cell's parents, de-duplicated by (fit model key, cell id), each verified in its mirror (the ctlstop lane)."""
    by_key: Dict[Tuple[str, str], Tuple[Any, List[str]]] = {}
    for c in bank.cells:
        for p in c.parents:
            key = (p.model_key, p.cell_id)
            if key in by_key and by_key[key][0].object_sha256 != p.object_sha256:
                raise Refusal(f"parents: {key} named with two shas")
            by_key.setdefault(key, (p, []))[1].append(c.cell_id)
    out: List[Any] = []
    for (fit_mk, cell), (ref, consumers) in by_key.items():
        where = f"parent {fit_mk}/{cell} ({ref.staged_id}; consumed by {consumers})"
        src = (Path(source) if fit_mk.endswith(CTLSTOP_FIT_SUFFIX) else Path(parents_source)) / ref.staged_id
        obj, tbm = src / S.OBJECT_NAME, src / S.BUDGET_META
        if not obj.is_file() or not tbm.is_file():
            raise Refusal(f"{where}: {src} lacks {S.OBJECT_NAME} and/or {S.BUDGET_META}")
        if S._identity_sidecars_present(src) != [S.BUDGET_META]:
            raise Refusal(f"{where}: identity sidecars {S._identity_sidecars_present(src)}; a fitted parent carries exactly [{S.BUDGET_META!r}]")
        osha = S.sha256_file(obj)
        if osha != ref.object_sha256:
            raise Refusal(f"{where}: mirror object sha {osha[:12]} != the derived sidecar's {ref.object_sha256[:12]}")
        meta = S.load_json(tbm)
        if not isinstance(meta, dict) or meta.get("object_sha256") != osha or meta.get("cell_id") != cell:
            raise Refusal(f"{where}: {S.BUDGET_META} does not record (object {osha[:12]}, cell {cell})")
        tsha = S.sha256_file(tbm)
        if ref.budget_meta_sha256 is not None and tsha != ref.budget_meta_sha256:
            raise Refusal(f"{where}: {S.BUDGET_META} hashes to {tsha[:12]}, not the derived sidecar's {str(ref.budget_meta_sha256)[:12]}")
        rdir = f"/{S.PARENTS_ROOT}/{fit_mk}/cells_all/{cell}"
        files = [S.PlannedFile(S.OBJECT_NAME, obj, f"{rdir}/{S.OBJECT_NAME}", osha, obj.stat().st_size),
                 S.PlannedFile(S.BUDGET_META, tbm, f"{rdir}/{S.BUDGET_META}", tsha, tbm.stat().st_size)]
        metrics = src / S.METRICS_NAME
        if metrics.is_file():
            files.append(S.PlannedFile(S.METRICS_NAME, metrics, f"{rdir}/{S.METRICS_NAME}", S.sha256_file(metrics), metrics.stat().st_size))
        parent = S.PlannedParent(cell_id=cell, fit_model_key=fit_mk, staged_id=ref.staged_id, source_dir=src, object_sha256=osha,
                                 budget_meta_sha256=tsha, consumers=sorted(consumers), remote_dir=rdir, files=files, staged_from={})
        parent.staged_from = S.build_staged_from(parent, written_at=written_at, tree_sha=bank.tree_sha)
        parent.staged_from["staged_by"] = STAGED_BY
        sf_dir = Path(stage_dir) / S.PARENTS_ROOT / fit_mk / "cells_all" / cell
        sf_dir.mkdir(parents=True, exist_ok=True)
        sf = sf_dir / S.STAGED_FROM_NAME
        sf.write_text(S.json.dumps(parent.staged_from, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        parent.files.append(S.PlannedFile(S.STAGED_FROM_NAME, sf, f"{rdir}/{S.STAGED_FROM_NAME}", S.sha256_file(sf), sf.stat().st_size))
        out.append(parent)
    return S.ParentsPlan(model_key=MODEL_KEY, view=bank.view, parents=out)


def run(*, tree: Path, view: str, source: Path, parents_source: Optional[Path], receipts_dir: Path, go: bool, tree_sha: str,
        tree_info: Mapping[str, Any], modal_bin: Path, environment: Optional[str], stage_dir: Optional[Path] = None,
        registry_root: Optional[str] = None, log=print) -> Dict[str, Any]:
    lane = lane_of(view)
    started = S.utc_now()
    dry = not go
    if lane.parents and parents_source is None:
        raise Refusal(f"the {lane.name} lane stages its derived cells' parents: pass --parents-source")
    if not lane.parents and parents_source is not None:
        raise Refusal(f"the {lane.name} lane has no derived cell and stages no parents: drop --parents-source")
    stage_dir = Path(stage_dir) if stage_dir else S.DEFAULT_STAGE_ROOT / f"{lane.name}-{view}" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stage_dir.mkdir(parents=True, exist_ok=True)
    log(f"mode: {'LIVE' if go else 'DRY'} — {MODEL_KEY}/{view} ({lane.name} lane), tree {tree} @ {tree_sha[:12]}, env {environment or '(CLI default)'}")
    if go and tree_info.get("tree_on_remote") is False:
        raise Refusal(f"tree sha {tree_sha} is on no remote branch — push first")
    if go and tree_info.get("lane_config_dirty"):
        raise Refusal(f"the lane roster/pins are modified against HEAD {tree_info['lane_config_dirty']} — a manifest pins tree_sha; commit first")
    bank = plan_bank(tree, view, source, tree_sha=tree_sha)
    pplan = plan_parents(bank, source=source, parents_source=parents_source, stage_dir=stage_dir, written_at=started) if lane.parents else None
    S.describe_plan(bank, pplan, log)
    registry = S.registry_check(S.registry_objects_of(bank, pplan), registry_root=registry_root)
    log(f"registry check: {registry['status']} — {registry['detail']}")
    if registry["status"] == "mismatch":
        raise Refusal(f"the registry's object rows disagree with the pins: {registry['detail']}")
    vol = S.VolumeCLI(modal_bin, S.VOLUME, log=log, environment=environment)
    probe_rec = {"status": "not_run", "detail": "no inode probe in this stager (a small lane bank)"}
    log(f"== view {view} -> {bank.remote_view_root}")
    outcomes = S.stage_files(bank.files, vol, dry=dry, log=log, label=f"view {view}")
    verified: Dict[str, dict] = {}
    if not dry:
        verified = S.verify_remote({c.remote_dir: c.files for c in bank.cells}, vol, log=log, label=f"view {view}")
    manifest = S.build_manifest(bank, verified=verified, dry=dry, written_at=None if dry else S.utc_now())
    manifest.update({"roster": lane.roster_rel, "pins": lane.pins_rel, "staged_by": STAGED_BY, "wave": lane.wave})
    if lane.parents:
        manifest["parents_pins"] = {"ctlstop": lane.pins_rel, "production": PROD_PINS_REL}
    manifest_result = S.publish_manifest(manifest, vol, remote_dir=bank.remote_view_root, stage_dir=stage_dir, dry=dry, log=log)
    receipts_dir = Path(receipts_dir) / ("dry" if dry else "")
    out: Dict[str, Any] = {"bank": str(receipts_dir / "bank.json"), "parents": None, "dry": dry}
    if pplan is not None:
        log("== parents -> /out/<fit model_key>/cells_all/")
        p_outcomes = S.stage_parent_files(pplan, vol, dry=dry, log=log)
        p_verified: Dict[str, dict] = {}
        if not dry:
            p_verified = S.verify_remote({p.remote_dir: p.files for p in pplan.parents}, vol, log=log, label="parents", exact_listing=False)
    brec = S.bank_receipt(bank, outcomes=outcomes, verified=verified, manifest=manifest, manifest_result=manifest_result,
                          registry=registry, inode_probe=probe_rec, dry=dry, tree_info=tree_info, started_at=started)
    brec.update({"staged_by": STAGED_BY, "environment": environment, "roster": lane.roster_rel, "pins": lane.pins_rel, "lane": lane.name})
    S.write_receipt(receipts_dir / "bank.json", brec)
    if pplan is not None:
        prec = S.parents_receipt(pplan, bank, outcomes=p_outcomes, verified=p_verified, registry=registry, dry=dry,
                                 tree_info=tree_info, started_at=started)
        prec.update({"staged_by": STAGED_BY, "environment": environment, "lane": lane.name})
        out["parents"] = str(receipts_dir / "parents.json")
        S.write_receipt(receipts_dir / "parents.json", prec)
    log(f"DONE ({'LIVE' if go else 'DRY — nothing put; re-run with --go'}): receipts {out['bank']}" + (f", {out['parents']}" if out["parents"] else ""))
    return out


def tree_info_for(tree: Path, view: str) -> Dict[str, Any]:
    """The production stager's tree facts, plus whether THIS lane's roster / pins are modified against HEAD."""
    info = S.git_tree_info(tree)
    lane = lane_of(view)
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--", lane.roster_rel, lane.pins_rel],
                           capture_output=True, text=True)
    info["lane_config_dirty"] = [l.strip() for l in dirty.stdout.splitlines() if l.strip()]
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, help=" | ".join(f"{p}1..3" for p in LANES))
    ap.add_argument("--tree", type=Path, default=S.REPO_LOCAL)
    ap.add_argument("--source", type=Path, required=True, help="dir holding the lane bank's <staged_id>/ dirs (<view>__<cell>)")
    ap.add_argument("--parents-source", type=Path, default=None,
                    help="ctlstop lane only: dir holding the production seed fits' <staged_id>/ dirs (base64s<k>__<cell>)")
    ap.add_argument("--receipts-dir", type=Path, required=True)
    ap.add_argument("--environment", default=None, help="the Modal environment (default $SISTEER_MODAL_ENVIRONMENT, else the CLI's own)")
    ap.add_argument("--modal", type=Path, default=S.DEFAULT_MODAL)
    ap.add_argument("--stage-dir", type=Path, default=None)
    ap.add_argument("--registry-root", default=None)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    log = lambda s: print(f"[stage-lane {time.strftime('%H:%M:%SZ', time.gmtime())}] {s}", flush=True)  # noqa: E731
    try:
        info = tree_info_for(Path(a.tree).resolve(), a.view)
        env = a.environment or S.os.environ.get("SISTEER_MODAL_ENVIRONMENT") or None
        run(tree=Path(a.tree).resolve(), view=a.view, source=a.source, parents_source=a.parents_source, receipts_dir=a.receipts_dir,
            go=a.go, tree_sha=info.pop("tree_sha"), tree_info=info, modal_bin=a.modal, environment=env, stage_dir=a.stage_dir,
            registry_root=a.registry_root, log=log)
        return 0
    except Refusal as e:
        log(f"REFUSED: {e}")
        return S.EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
