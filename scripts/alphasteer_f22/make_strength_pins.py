#!/usr/bin/env python
"""alphasteer-full22 STRENGTH FRONTIER — GENERATE configs/alphasteer_f22_strength_pins.json from the strength
bake records; never type a sha by hand.

The pin of every served cell is the object_sha256 the bake recorded: `strength_bake.json` (written by
scripts/alphasteer_f22/bake_strength.py on the cluster at <objects root>/<target>/strength/, pulled to the
grading host under --records-root/<target>/) names every cell's staged id, strength, view, mechanism and sha, and
make_bank_view.py verifies the staged copy against THIS file.

What is verified before a sha is written:
  * the record's schema and its embedded `sha256` (bake_rungs' convention: sha256 over the sort_keys dump
    without the field) — an edited or torn record is refused, not read;
  * the record's `target` equals the roster's; its `strengths` equal the roster's `strengths` and
    as22_strength.STRENGTHS; every cell's `view` / `staged_id` equal the roster's view -> staged id mapping;
  * the anchor (strength 1.0) cell's sha EQUALS the parent's and is flagged `is_parent`; every scaled cell's
    sha DIFFERS from the parent's; every cell's verification block is PASS;
  * the record's parent sha agrees with the CAMPAIGN pins' `parent_object_sha256` for the targets whose
    campaign entry carries one (gemma / qwen / nemotron; glm's campaign entry has no object and is recorded
    as unpinned there) — the same parent the campaign's ladder was baked off.

A target whose record does not exist yet is REFUSED by default, naming the target; `--allow-pending` is the
ONE explicit way to write a partial file: the missing target's views get `{"status": "pending_bake",
"object_sha256": null}` — never a sha — and the strict `--check` refuses such a file. make_bank_view.py refuses
an unpinned (null-sha) object outright.

LANES: `--lane alphasteer_f22` (default; the record's parent MUST equal the campaign pins' parent_object_sha256 where
one exists) or `--lane alphasteer_f22_rebake` (the re-baked parents: the campaign's parent is RECORDED beside ours as
a different lineage — `campaign_parent_pin_agrees: false` — and equality would be refused, since it would mean the
re-bake changed nothing). Records written under schema /2 must carry the `layers` check PASS, and every cell's layer
list must equal the record's parent layers (the roster's `parent_layers`, when declared, must equal them too).

usage:
  python scripts/alphasteer_f22/make_strength_pins.py [--lane L] [--records-root DIR]      # write
  python scripts/alphasteer_f22/make_strength_pins.py --check                    # strict: exit 1 on drift OR pending
  python scripts/alphasteer_f22/make_strength_pins.py --allow-pending [--check]  # the partial form, explicit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from as22_strength import (DEFAULT_LANE, LANES, STRENGTHS, Lane, is_anchor, lane as lane_of,  # noqa: E402
                           layer_gaps, load_roster, roster_parent_layers, staged_id, strength_of_view, view_name)
from as22_targets import CELL_ID  # noqa: E402
from bake_strength import ACCEPTED_SCHEMAS, BAKE_RECORD, MECHANISM_ANCHOR, MECHANISM_SCALED, SCHEMA, record_sha  # noqa: E402

CAMPAIGN_PINS = REPO / "configs" / "alphasteer_f22_pins.json"
GRADE_ROOT = os.environ.get("AS22_STRENGTH_GRADE_ROOT", "$GRADE")   # the grade root; $GRADE expands through sisteer.site at use


def default_records_root(ln: Lane) -> Path:
    """Where a lane's bake records are pulled to on the grading host: <grade root>/<lane records_subdir>/<target>/strength_bake.json."""
    from sisteer import site as _site
    return _site.expand_path(GRADE_ROOT, key="AS22_STRENGTH_GRADE_ROOT") / ln.records_subdir
GENERATOR = "scripts/alphasteer_f22/make_strength_pins.py"
STATUS_SERVED = "served"
STATUS_PENDING = "pending_bake"


class PinsRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"make_strength_pins: REFUSED — {msg}")


def roster_views(roster: dict, target: str) -> Dict[str, Tuple[float, str]]:
    """``{view: (strength, staged id)}`` for the target, verified against the code table."""
    t = roster["targets"][target]
    out: Dict[str, Tuple[float, str]] = {}
    for view, v in (t.get("views") or {}).items():
        s = strength_of_view(view)
        if v.get("strength") != s:
            raise PinsRefusal(f"{target}: roster view {view!r} records strength {v.get('strength')!r} but its name spells {s!r}")
        cells = dict(v.get("cells") or {})
        if cells != {CELL_ID: staged_id(s)}:
            raise PinsRefusal(f"{target}: roster view {view!r} cells {cells} != {{{CELL_ID}: {staged_id(s)}}}")
        out[view] = (s, staged_id(s))
    roster_parent_layers(t)                       # validated when declared (ints, strictly ascending)
    want = [view_name(s) for s in roster.get("strengths") or []]
    if list(out) != want or [float(x) for x in roster.get("strengths") or []] != [float(x) for x in STRENGTHS]:
        raise PinsRefusal(f"{target}: roster views {list(out)} / strengths {roster.get('strengths')} != the ladder "
                          f"{[view_name(s) for s in STRENGTHS]} (as22_strength.STRENGTHS)")
    return out


def read_record(records_root: Path, target: str) -> Optional[dict]:
    p = Path(records_root) / target / BAKE_RECORD
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("schema") not in ACCEPTED_SCHEMAS:
        raise PinsRefusal(f"{p}: schema {rec.get('schema')!r} not in {ACCEPTED_SCHEMAS!r}")
    if rec.get("sha256") != record_sha(rec):
        raise PinsRefusal(f"{p}: sha256 does not verify under the writer's convention — an edited or torn record; refusing")
    if rec.get("target") != target:
        raise PinsRefusal(f"{p}: target {rec.get('target')!r} != {target!r}")
    rec["_path"] = str(Path(target) / BAKE_RECORD)
    return rec


def campaign_entry(target: str, campaign_pins_path: Path = CAMPAIGN_PINS) -> dict:
    pins = json.loads(Path(campaign_pins_path).read_text(encoding="utf-8"))
    return dict((pins.get(target) or {}).get("selected__alphasteer") or {})


def entries_from_record(target: str, views: Dict[str, Tuple[float, str]], rec: dict, camp: dict, *,
                        ln: Lane, tinfo: dict) -> Dict[str, Any]:
    parent = rec.get("parent") or {}
    parent_sha = parent.get("object_sha256")
    if not isinstance(parent_sha, str) or len(parent_sha) != 64:
        raise PinsRefusal(f"{target}: record parent.object_sha256 {parent_sha!r} is not a sha256")
    if [float(x) for x in rec.get("strengths") or []] != [s for s, _ in views.values()]:
        raise PinsRefusal(f"{target}: record strengths {rec.get('strengths')} != roster {[s for s, _ in views.values()]}")
    if rec.get("lane", DEFAULT_LANE) != ln.name or rec.get("parent_lineage", "campaign_lambda1") != ln.parent_lineage:
        raise PinsRefusal(f"{target}: record lane / lineage {rec.get('lane', DEFAULT_LANE)!r} / "
                          f"{rec.get('parent_lineage', 'campaign_lambda1')!r} != the lane's {ln.name!r} / {ln.parent_lineage!r}")
    parent_layers = [int(l) for l in (parent.get("layers") or [])]
    declared = roster_parent_layers(tinfo)
    if declared is not None and declared != parent_layers:
        raise PinsRefusal(f"{target}: roster parent_layers {declared} != the record's parent layers {parent_layers}")
    camp_parent = camp.get("parent_object_sha256")
    if ln.parent_lineage == "campaign_lambda1":
        # the campaign's own parents: the record's parent MUST be the one the campaign's ladder was baked off
        if camp_parent is not None and camp_parent != parent_sha:
            raise PinsRefusal(f"{target}: record parent {parent_sha[:12]} != the campaign pins' parent_object_sha256 "
                              f"{str(camp_parent)[:12]} — not the parent the campaign's ladder was baked off")
        agrees = (camp_parent == parent_sha) if camp_parent is not None else None
    else:
        # a DIFFERENT lineage (the re-baked parents): the campaign's parent is recorded beside ours for the reader,
        # never compared for equality — equality here would mean the re-bake changed nothing
        if camp_parent is not None and camp_parent == parent_sha:
            raise PinsRefusal(f"{target}: lineage {ln.parent_lineage!r} but the record's parent {parent_sha[:12]} IS the "
                              "campaign's parent — this is not a re-baked object")
        agrees = False if camp_parent is not None else None
    needed = ("tensors", "scalar", "served_ops", "served_edit", "sign") + (("layers",) if rec.get("schema") == SCHEMA else ())
    cells = {c["staged_id"]: c for c in rec.get("cells") or []}
    out: Dict[str, Any] = {}
    for view, (s, sid) in views.items():
        c = cells.get(sid)
        if c is None:
            raise PinsRefusal(f"{target}: record has no cell for {sid}")
        sha = c.get("object_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise PinsRefusal(f"{target} {sid}: object_sha256 {sha!r} is not a sha256")
        if c.get("strength") != s or c.get("view") != view:
            raise PinsRefusal(f"{target} {sid}: record strength/view {c.get('strength')!r}/{c.get('view')!r} != {s!r}/{view!r}")
        ver = c.get("verification") or {}
        checks = {k["check"]: k["status"] for k in ver.get("checks") or []}
        if ver.get("status") != "PASS" or not checks or any(v != "PASS" for v in checks.values()):
            raise PinsRefusal(f"{target} {sid}: verification {ver.get('status')!r} / {checks} is not all PASS")
        if is_anchor(s):
            if c.get("mechanism") != MECHANISM_ANCHOR or sha != parent_sha or not c.get("is_parent"):
                raise PinsRefusal(f"{target} {sid}: the anchor must be the parent (mechanism {MECHANISM_ANCHOR}, sha == "
                                  f"parent {parent_sha[:12]}), got {c.get('mechanism')!r} / {sha[:12]}")
            if "sign" in checks:
                raise PinsRefusal(f"{target} {sid}: the anchor carries a sign check — it is not a scaled cell")
        else:
            if c.get("mechanism") != MECHANISM_SCALED or sha == parent_sha or c.get("is_parent"):
                raise PinsRefusal(f"{target} {sid}: a scaled cell must carry mechanism {MECHANISM_SCALED} and differ "
                                  f"from the parent, got {c.get('mechanism')!r} / {sha[:12]}")
            for need in needed:
                if checks.get(need) != "PASS":
                    raise PinsRefusal(f"{target} {sid}: verification lacks a PASS for {need!r}")
        if rec.get("schema") == SCHEMA and [int(l) for l in c.get("layers") or []] != parent_layers:
            raise PinsRefusal(f"{target} {sid}: cell layers {c.get('layers')} != the parent's {parent_layers}")
        out[sid] = {
            "lane": ln.name, "parent_lineage": ln.parent_lineage,
            "object_sha256": sha,
            "status": STATUS_SERVED,
            "strength": s, "view": view, "cell_id": CELL_ID,
            "mechanism": c.get("mechanism"), "is_parent": bool(c.get("is_parent")),
            "object_is_symlink": bool(c.get("object_is_symlink")),
            "parent_object_sha256": parent_sha,
            "intervention_hash": c.get("intervention_hash"),
            "n_layers": c.get("n_layers"), "layers": c.get("layers"),
            "layer_gaps": layer_gaps(c.get("layers") or []), "layers_contiguous": not layer_gaps(c.get("layers") or []),
            "identity_sidecar": "fit_provenance.json",
            # the cell dir RELATIVE to <objects root>/<target>/ on the cluster — what stage_strength.sh copies
            "source": f"strength/cells/{sid}",
            "verification_checks": checks,
            "bake_record": rec["_path"], "bake_record_sha256": rec["sha256"], "bake_code_sha": rec.get("code_sha"),
            "campaign_selection_status": camp.get("status"),
            "campaign_selected_lambda": camp.get("lambda"),
            "campaign_parent_object_sha256": camp_parent,
            "campaign_parent_pin_agrees": agrees,
        }
    return out


def pending_entries(views: Dict[str, Tuple[float, str]]) -> Dict[str, Any]:
    return {sid: {"object_sha256": None, "status": STATUS_PENDING, "strength": s, "view": view,
                  "note": ("the strength bake for this target has not landed (no strength_bake.json under the records "
                           "root); written by --allow-pending ONLY; a strict --check refuses this file and "
                           "make_bank_view.py refuses a null-sha pin")}
            for view, (s, sid) in views.items()}


def render(roster: dict, records_root: Path, *, allow_pending: bool, ln: Lane = None,
           campaign_pins_path: Path = CAMPAIGN_PINS) -> Tuple[Dict[str, Any], List[str]]:
    ln = ln or lane_of(DEFAULT_LANE)
    if roster.get("bank_root") != ln.bank_root_rel or roster.get("parent_lineage") != ln.parent_lineage:
        raise PinsRefusal(f"roster bank_root / parent_lineage {roster.get('bank_root')!r} / {roster.get('parent_lineage')!r} "
                          f"!= lane {ln.name}'s {ln.bank_root_rel!r} / {ln.parent_lineage!r}")
    if not roster.get("targets"):
        raise PinsRefusal(f"lane {ln.name}: roster names no targets")
    pins: Dict[str, Any] = {}
    pending: List[str] = []
    for target in roster["targets"]:               # THE lane's target set: the roster's keys
        views = roster_views(roster, target)
        rec = read_record(records_root, target)
        if rec is None:
            pending.append(target)
            pins[target] = pending_entries(views)
            continue
        pins[target] = entries_from_record(target, views, rec, campaign_entry(target, campaign_pins_path),
                                           ln=ln, tinfo=roster["targets"][target])
    if pending and not allow_pending:
        raise PinsRefusal("no strength bake record for: " + ", ".join(pending) +
                          f" (looked under {records_root}/<target>/{BAKE_RECORD}). Pass --allow-pending to write an "
                          "explicitly PARTIAL file (null shas, refused by the strict --check)")
    return pins, pending


def render_text(pins: Dict[str, Any]) -> str:
    return json.dumps(pins, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--lane", default=DEFAULT_LANE, choices=sorted(LANES),
                    help="which lane's roster / records / pins (default: the campaign-parent lane)")
    ap.add_argument("--records-root", type=Path, default=None,
                    help="default: $GRADE/<lane records_subdir> (AS22_STRENGTH_GRADE_ROOT overrides the root)")
    ap.add_argument("--roster", type=Path, default=None, help="default: the lane's roster")
    ap.add_argument("--campaign-pins", type=Path, default=CAMPAIGN_PINS)
    ap.add_argument("--out", type=Path, default=None, help="default: the lane's pins file")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file drifts from the records")
    ap.add_argument("--allow-pending", action="store_true", help="write / accept a PARTIAL file for unbaked targets")
    args = ap.parse_args(argv)
    ln = lane_of(args.lane)
    args.records_root = args.records_root or default_records_root(ln)
    args.out = args.out or ln.pins_path
    roster = yaml.safe_load(Path(args.roster).read_text(encoding="utf-8")) if args.roster else load_roster(ln)
    pins, pending = render(roster, args.records_root, allow_pending=args.allow_pending, ln=ln,
                           campaign_pins_path=args.campaign_pins)
    text = render_text(pins)
    if args.check:
        if not args.out.is_file():
            print(f"DRIFT: {args.out} missing"); return 1
        if args.out.read_text(encoding="utf-8") != text:
            print(f"DRIFT: {args.out} differs from the records under {args.records_root}"); return 1
        if pending:
            print(f"PENDING: {args.out} carries {STATUS_PENDING} for {pending} (accepted under --allow-pending)")
        n = sum(1 for t in pins for e in pins[t].values() if e["status"] == STATUS_SERVED)
        print(f"ok: {args.out} ({n} served cells over {len(pins)} targets, {len(pending)} pending)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".tmp-{args.out.name}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.out)
    for t, cells in pins.items():
        for sid, e in cells.items():
            print(f"{t:<16} {sid:<24} {e['status']:<14} s={e.get('strength')!s:<5} sha={str(e.get('object_sha256'))[:12]}")
    print(f"wrote {args.out}" + (f" (PENDING: {pending})" if pending else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
