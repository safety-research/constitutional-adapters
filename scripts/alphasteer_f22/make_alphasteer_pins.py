#!/usr/bin/env python
"""alphasteer-full22 Phase F — GENERATE configs/alphasteer_f22_pins.json from the ladder selection
records; never type a sha by hand.

The pin of every served cell is the object_sha256 of the SELECTED baked rung: `lambda_ladder.json`
(written by scripts/alphasteer_f22/ladder_select.py on the grading host, one per target under
--records-root/<target>/) names it as `chosen.i` / `chosen.object_sha256`, and the rung it names lives
on the cluster at <objects root>/<target>/ladder/lam<i>/cells_view/<view>/alphasteer/object.pt — the
dir stage_alphasteer.sh copies into <target>/cells/selected__alphasteer/ and make_bank_view.py then
verifies against THIS file.

What is verified before a sha is written:
  * the record has its `.COMPLETE` marker and its `sha256` verifies under the writer's payload
    convention (ladder_select.write_json: sha256 over the indent=1 sort_keys dump without the field) —
    an edited or torn record is refused, not read;
  * the record's `model_key` / `view` equal the roster's target / view (as22_targets.VIEW_OF);
  * `chosen.object_sha256` equals `rungs[chosen.i].object_sha256` (two spellings of one fact);
  * the roster's expectation and the record's outcome AGREE: a target whose view serves `alphasteer`
    must have a CHOSEN rung (`chosen` or `a4_fallback`); a target the roster lists under `not_served`
    must have `outcome == no_admissible_operating_point`. A disagreement in either direction refuses —
    that is how "every generator refuses to stage or serve a glm object" is enforced.

A target whose record does not exist yet is REFUSED by default, naming the target: the pins cannot
be complete until every served target's selection has landed. `--allow-pending` is the ONE explicit
way to write a partial file: the missing targets get `{"status": "pending_selection", "object_sha256":
null}` — never a sha — and a strict `--check` (this script's, the preflight) refuses such a file until the
records land. make_bank_view.py refuses an unpinned (null-sha) object outright, so nothing can be
served off a pending entry either.

usage:
  python scripts/alphasteer_f22/make_alphasteer_pins.py [--records-root DIR]            # write
  python scripts/alphasteer_f22/make_alphasteer_pins.py --check                          # strict: exit 1 on drift OR pending
  python scripts/alphasteer_f22/make_alphasteer_pins.py --allow-pending [--check]        # the partial form, explicit
"""
from __future__ import annotations

import argparse
import hashlib
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

from as22_targets import CELL_ID, TARGETS, view_of  # noqa: E402

ROSTER = REPO / "configs" / "alphasteer_f22_roster.yaml"
PINS = REPO / "configs" / "alphasteer_f22_pins.json"
#: where the selection queue wrote lambda_ladder.json on the grading host: <records root>/<target>/lambda_ladder.json
#: (default in the grade root's placeholder form, $GRADE, expanded through sisteer.site after the arguments are parsed)
DEFAULT_RECORDS_ROOT = Path(os.environ.get("AS22_RECORDS_ROOT", "$GRADE/alphasteer_f22/ladder"))
#: the staged id of the selected rung (the roster maps `alphasteer` -> this)
STAGED_ID = "selected__alphasteer"
RECORD_SCHEMA = "alphasteer_lambda_ladder/1"
GENERATOR = "scripts/alphasteer_f22/make_alphasteer_pins.py"

STATUS_SERVED = "served"
STATUS_NO_OP = "no_admissible_operating_point"
STATUS_PENDING = "pending_selection"


class PinsRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"make_alphasteer_pins: REFUSED — {msg}")


def _payload_sha(payload: Dict[str, Any]) -> str:
    """ladder_select._payload_sha, restated stdlib-only (this generator must not import the
    selection driver); tests/test_alphasteer_f22_run.py pins the two equal."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest()


def load_roster(path: Path = ROSTER) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def roster_expectation(roster: dict, target: str) -> Tuple[str, Optional[dict]]:
    """``("served", None)`` when the target's view maps `alphasteer` -> STAGED_ID; ``("not_served",
    marker)`` when the view lists no cell and `not_served` carries the STAGED_ID marker. Anything
    else is a malformed roster and refuses."""
    t = roster["targets"][target]
    view = view_of(target)
    if view not in t.get("views", {}):
        raise PinsRefusal(f"{target}: roster has no view {view!r} (as22_targets.VIEW_OF says {view})")
    cells = dict(t["views"][view].get("cells") or {})
    markers = [m for m in (t.get("not_served") or []) if m.get("staged_id") == STAGED_ID]
    if cells == {CELL_ID: STAGED_ID} and not markers:
        return "served", None
    if not cells and len(markers) == 1:
        m = markers[0]
        for k in ("outcome", "decision", "reason"):
            if not m.get(k):
                raise PinsRefusal(f"{target}: not_served marker lacks {k!r}")
        if m["outcome"] != STATUS_NO_OP:
            raise PinsRefusal(f"{target}: not_served outcome {m['outcome']!r} is not {STATUS_NO_OP!r}")
        return "not_served", m
    raise PinsRefusal(f"{target}: roster view {view!r} cells {cells} with not_served markers {len(markers)} — "
                      f"a served target maps exactly {{{CELL_ID}: {STAGED_ID}}} and carries no marker; a "
                      f"not-served target lists no cell and exactly one {STAGED_ID} marker")


def read_record(records_root: Path, target: str) -> Optional[dict]:
    """The verified lambda_ladder.json for ``target``, or None when it does not exist yet."""
    p = Path(records_root) / target / "lambda_ladder.json"
    if not p.is_file():
        return None
    if not (p.with_name(p.name + ".COMPLETE")).is_file():
        raise PinsRefusal(f"{p} exists without its .COMPLETE marker — a torn or in-progress record; refusing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("schema") != RECORD_SCHEMA:
        raise PinsRefusal(f"{p}: schema {rec.get('schema')!r} != {RECORD_SCHEMA!r}")
    if rec.get("sha256") != _payload_sha(rec):
        raise PinsRefusal(f"{p}: sha256 does not verify under the writer's payload convention — an edited or "
                          "torn record; refusing to pin from it")
    marker = json.loads(p.with_name(p.name + ".COMPLETE").read_text(encoding="utf-8"))
    if marker.get("sha256") != rec["sha256"]:
        raise PinsRefusal(f"{p}.COMPLETE names sha256 {str(marker.get('sha256'))[:12]} but the record is "
                          f"{rec['sha256'][:12]}")
    if rec.get("model_key") != target:
        raise PinsRefusal(f"{p}: model_key {rec.get('model_key')!r} != {target!r}")
    if rec.get("view") != view_of(target):
        raise PinsRefusal(f"{p}: view {rec.get('view')!r} != as22_targets.VIEW_OF[{target}] = {view_of(target)!r}")
    rec["_path"] = str(Path(target) / "lambda_ladder.json")     # relative to the records root: no machine path
    return rec


def entry_from_record(target: str, expect: str, marker: Optional[dict], rec: dict) -> Dict[str, Any]:
    chosen = rec.get("chosen") or {}
    outcome = rec.get("outcome")
    no_op = bool(chosen.get("no_admissible_operating_point"))
    if outcome == STATUS_NO_OP or no_op:
        if not (outcome == STATUS_NO_OP and no_op):
            raise PinsRefusal(f"{target}: record outcome {outcome!r} and chosen.no_admissible_operating_point={no_op} "
                              "disagree")
        if expect != "not_served":
            raise PinsRefusal(f"{target}: the roster SERVES this target but its ladder record says "
                              f"{STATUS_NO_OP} — move it under not_served (D72) or re-select; refusing to pin")
        return {
            "object_sha256": None,
            "status": STATUS_NO_OP,
            "outcome": outcome,
            "decision": marker["decision"],
            "ladder_select_exit": marker.get("ladder_select_exit"),
            "chosen_i": None, "lambda": None, "intervention_hash": None,
            "n_rungs": chosen.get("n_rungs"), "n_live": chosen.get("n_live"),
            "rule": chosen.get("rule"),
            "why": list(chosen.get("why") or []),
            "ladder_record": rec["_path"],
            "ladder_record_sha256": rec["sha256"],
            "ladder_code_sha": rec.get("code_sha"),
            "note": "EXPLICIT no-admissible-operating-point marker: nothing is staged or served for "
                    "this target; a null object_sha256 is refused by make_bank_view.py by construction",
        }
    if outcome not in ("chosen", "a4_fallback"):
        raise PinsRefusal(f"{target}: unknown record outcome {outcome!r}")
    if expect != "served":
        raise PinsRefusal(f"{target}: the roster lists this target under not_served but its ladder record "
                          f"CHOSE rung {chosen.get('i')} — the roster and the record disagree; refusing")
    i, sha = chosen.get("i"), chosen.get("object_sha256")
    if not isinstance(i, int) or i < 1:
        raise PinsRefusal(f"{target}: chosen.i {i!r} is not a rung index")
    if not isinstance(sha, str) or len(sha) != 64:
        raise PinsRefusal(f"{target}: chosen.object_sha256 {sha!r} is not a sha256")
    rungs = {int(r["i"]): r for r in rec.get("rungs") or []}
    if i not in rungs:
        raise PinsRefusal(f"{target}: chosen.i {i} has no rung block in the record")
    if rungs[i].get("object_sha256") != sha:
        raise PinsRefusal(f"{target}: chosen.object_sha256 {sha[:12]} != rungs[{i}].object_sha256 "
                          f"{str(rungs[i].get('object_sha256'))[:12]}")
    if rungs[i].get("admissible") is not True or rungs[i].get("certified") is not True:
        raise PinsRefusal(f"{target}: chosen rung {i} is admissible={rungs[i].get('admissible')} "
                          f"certified={rungs[i].get('certified')} — a chosen rung must be both")
    view = view_of(target)
    return {
        "object_sha256": sha,
        "status": STATUS_SERVED,
        "outcome": outcome,
        "fallback_a4": bool(chosen.get("fallback_a4")),
        "chosen_i": i,
        "lambda": chosen.get("lambda"),
        "intervention_hash": chosen.get("intervention_hash"),
        "parent_object_sha256": rungs[i].get("parent_object_sha256"),
        "rule": chosen.get("rule"),
        # the rung dir RELATIVE to <objects root>/<target>/ on the cluster — what stage_alphasteer.sh copies
        "source": f"ladder/lam{i}/cells_view/{view}/{CELL_ID}",
        "staged_to": f"cells/{STAGED_ID}",
        "identity_sidecar": "fit_provenance.json",
        "ladder_record": rec["_path"],
        "ladder_record_sha256": rec["sha256"],
        "ladder_code_sha": rec.get("code_sha"),
    }


def pending_entry(target: str) -> Dict[str, Any]:
    return {"object_sha256": None, "status": STATUS_PENDING,
            "note": ("the ladder selection for this target has not landed (no lambda_ladder.json under the "
                     "records root); written by --allow-pending ONLY; a strict --check refuses this file and "
                     "make_bank_view.py refuses a null-sha pin")}


def render(roster: dict, records_root: Path, *, allow_pending: bool) -> Tuple[Dict[str, Any], List[str]]:
    """``(pins, pending_targets)``. Refuses (SystemExit) on any inconsistency; on a missing record
    unless ``allow_pending``."""
    if set(roster["targets"]) != set(TARGETS):
        raise PinsRefusal(f"roster targets {sorted(roster['targets'])} != as22_targets.TARGETS {sorted(TARGETS)}")
    pins: Dict[str, Any] = {}
    pending: List[str] = []
    for target in roster["targets"]:          # roster order: the served targets first, the marker last
        expect, marker = roster_expectation(roster, target)
        rec = read_record(records_root, target)
        if rec is None:
            if expect == "not_served":
                raise PinsRefusal(f"{target}: the roster lists it under not_served but no ladder record exists "
                                  "to attest that outcome — a marker must be backed by the exit-4 record")
            pending.append(target)
            pins[target] = {STAGED_ID: pending_entry(target)}
            continue
        pins[target] = {STAGED_ID: entry_from_record(target, expect, marker, rec)}
    if pending and not allow_pending:
        raise PinsRefusal("no ladder selection record for: " + ", ".join(pending) +
                          f" (looked under {records_root}/<target>/lambda_ladder.json). The pins cannot be "
                          "complete until every served target's selection has landed; pass --allow-pending to "
                          "write an explicitly PARTIAL file (null shas, refused by the strict --check)")
    return pins, pending


def render_text(pins: Dict[str, Any]) -> str:
    return json.dumps(pins, indent=2, sort_keys=True) + "\n"


def pending_targets_in(pins: Dict[str, Any]) -> List[str]:
    return [t for t, cells in pins.items() if any(e.get("status") == STATUS_PENDING for e in cells.values())]


def served_targets_in(pins: Dict[str, Any]) -> List[str]:
    return [t for t, cells in pins.items()
            if any(e.get("status") == STATUS_SERVED and e.get("object_sha256") for e in cells.values())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    ap.add_argument("--roster", type=Path, default=ROSTER)
    ap.add_argument("--out", type=Path, default=PINS)
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file drifts from the records")
    ap.add_argument("--allow-pending", action="store_true",
                    help="write / accept a PARTIAL file for targets whose selection has not landed")
    args = ap.parse_args(argv)
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from sisteer import site as _site
    args.records_root = _site.expand_path(args.records_root, key="--records-root")
    roster = load_roster(args.roster)
    pins, pending = render(roster, args.records_root, allow_pending=args.allow_pending)
    text = render_text(pins)
    if args.check:
        if not args.out.is_file():
            print(f"DRIFT: {args.out} missing"); return 1
        if args.out.read_text(encoding="utf-8") != text:
            print(f"DRIFT: {args.out} differs from the records under {args.records_root}"); return 1
        if pending:
            print(f"PENDING: {args.out} carries pending_selection for {pending} (accepted under --allow-pending)")
        print(f"ok: {args.out} ({len(served_targets_in(pins))} served, "
              f"{len([t for t in pins if pins[t][STAGED_ID].get('status') == STATUS_NO_OP])} not served, "
              f"{len(pending)} pending)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".tmp-{args.out.name}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(args.out)
    for t, cells in pins.items():
        e = cells[STAGED_ID]
        print(f"{t:<16} {e['status']:<32} sha={str(e.get('object_sha256'))[:12]} rung={e.get('chosen_i')}")
    print(f"wrote {args.out}" + (f" (PENDING: {pending})" if pending else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
