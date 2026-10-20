#!/usr/bin/env python3
"""The coverage table for the Qwen seed-replicate native battery.

    coverage_table.py build --stores RS/stores --expected RS/expected_arm_runs.json --launch-plan RS/launch_plan.json
        --audit-summary RS/audits/request_log_summary.csv --identity-legs RS/tables/engine_identity_legs.json
        --out RS/tables/coverage.csv [--petri-behaviours b1,b2]
    coverage_table.py expected --view-config configs/sdfbase_full22_qwen35_4b_s1.yaml [--view-config …_s2.yaml …_s3.yaml]
        --out RS/expected_arm_runs.json

15 legs × 27 arm-runs = 405 rows, each `complete` | `missing(<reason>)`. The EXPECTED set is a file the orchestrator
derives once (`expected`: per s-view config's `explicit_cells` (the config puts `none`, the 5 fitted cells and the two
prompted routes there) + the Arditi arm at s ∈ {1,2,3}) — never a count typed into this table. A row is `complete`
iff ALL of:

* the arm-run's FINAL ACCEPTED MATERIALISATION is resolved through the ladder / completion rules (`at_cap_table.row_for`:
  the 1 % item-filtered trigger from `ladder_decide` / `inspect_rung`, the ceiling, the fixed-cap legs) — a ladder that
  is not final (`escalate_pending(<next cap>)`: the base or a rung is above the trigger and the next rung root is absent
  or unmaterialised; `ladder_gap(...)`: a rung root above a predecessor that was never materialised) is a NAMED miss,
  never "base AGREES suffices";
* at EVERY stage of that materialisation's ancestry (the base and each rung root the final rung rests on), for EVERY
  standup the launch plan (`--launch-plan`, `RS/launch_plan.json`, schema seedrep_launch_plan/1) declares at
  (container leg, view, stage) as serving the arm-run (`arm_runs[].cell == arm`, `.s == s`) — the plan is the DECLARED
  shard / producer inventory, so an absent planned shard is enumerated as a miss: an accepted attempt root is pulled and
  bound to that planned standup (its recorded `standup_id` / `control_key`, W and wave_uuid — an undeclared attempt, another
  tree's or wave's, or two accepted attempts for one standup REFUSE the table); it holds `COMPLETE.json`; the store's own
  completeness holds (the writer's declared samples, `audit_common`); misalign's `RESULT` says `rc=0`; and the
  request-evidence verdict for (leg, arm, s, STAGE, shard) is `AGREES` AND was filed for THAT attempt id — an old attempt's
  verdict is stale evidence, a named miss; a pending, failed or unaudited rung is a named miss (a non-AGREES store is a
  coverage miss WITH its reason, never re-labelled);
* the container leg's engine-identity verdict is PASS (a FAIL → every arm-run of the leg is `missing(identity: …)`, §9.3).

The row records `final_stage`, `ladder_state`, `rungs_present` and `n_stages_verified`. Every reason is named; the table is
refused (exit 2) rather than written partial when the expected file, the audit summary or the identity verdicts are
missing. Exit 1 when any row is missing.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import at_cap_table as act  # noqa: E402  (the ladder / completion rules: which rung is FINAL per arm-run)
import audit_common as ac  # noqa: E402
from audit_common import AttemptRoot, StoreRead, refuse  # noqa: E402

COLUMNS: Tuple[str, ...] = ("leg", "arm", "s", "view", "status", "reason?", "final_stage?", "ladder_state?", "rungs_present?",
                            "n_stages_verified", "attempt_id?", "n_rows?", "audit_verdict", "store_dir?")
COMPLETE = "complete"
#: the ladder states under which an arm-run's materialisation is FINAL (`at_cap_table._state` / FIXED_CAP_LEGS / MMLU)
FINAL_LADDER_STATES: Tuple[str, ...] = ("exit_le_1pct", "exit_ceiling", "fixed_cap", "not_applicable")
_RC_RE = re.compile(r"\brc=(\S+)")


# --------------------------------------------------------------------------- #
# expected arm-runs                                                             #
# --------------------------------------------------------------------------- #

def _explicit_cells(view_config: Path) -> List[str]:
    """`explicit_cells:` of an s-view config (a flat YAML list) — read without PyYAML so the table stays stdlib."""
    lines = view_config.read_text(encoding="utf-8").splitlines()
    cells: List[str] = []
    in_block = False
    for l in lines:
        if l.startswith("explicit_cells:"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"^\s*-\s*(\S+)\s*$", l)
            if m:
                cells.append(m.group(1))
                continue
            if l.strip() and not l.startswith("#"):
                break
    if not cells:
        refuse(f"{view_config}: no explicit_cells block")
    return cells


def build_expected(view_configs: Sequence[Path]) -> Dict[str, Any]:
    arm_runs: List[Dict[str, Any]] = []
    for vc in view_configs:
        view = re.search(r"_(s[1-9][0-9]*)\.yaml$", vc.name)
        if not view:
            refuse(f"{vc}: the file name does not end in _s<s>.yaml")
        v = view.group(1)
        cells = _explicit_cells(vc)
        for arm in PROMPTED_MISSING(cells):
            refuse(f"{vc}: explicit_cells lacks the prompted route {arm!r} (WS0-1 adds both prompted routes to every s-view)")
        for c in cells:
            arm_runs.append({"arm": c, "view": v, "s": ac.seed_of_view(v)})
    for s in (1, 2, 3):
        arm_runs.append({"arm": ac.ARDITI_CELL, "view": ac.ARDITI_VIEW, "s": s})
    return {"legs": list(ac.LEGS), "arm_runs": arm_runs, "source": [str(p) for p in view_configs]}


def PROMPTED_MISSING(cells: Sequence[str]) -> List[str]:
    return [a for a in ac.PROMPTED_ARMS if a not in cells]


# --------------------------------------------------------------------------- #
# the audit summary and identity verdicts                                       #
# --------------------------------------------------------------------------- #

def read_audit_summary(path: Path) -> Dict[Tuple[str, str, int, str, str], Dict[str, str]]:
    """(leg, arm, s, stage, shard) → the request-evidence verdict row. EVERY stage is kept: a rung's verdict certifies the
    rung store an arm-run's final materialisation rests on. Two rows for one key refuse (two stores claim one arm-run)."""
    if not Path(path).is_file():
        refuse(f"{path}: missing — run request_evidence_audit.py first (a non-AGREES store is a coverage miss with its reason)")
    out: Dict[Tuple[str, str, int, str, str], Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["leg"], row["arm"], int(row["s"]), row.get("stage") or ac.STAGE_BASE, row.get("shard") or "")
            except (KeyError, ValueError):
                refuse(f"{path}: a row lacks leg / arm / s")
            if key in out:
                refuse(f"{path}: two verdicts for {key} — two stores claim one arm-run at one stage and shard")
            out[key] = row
    if not out:
        refuse(f"{path}: no verdict rows")
    return out


def read_identity_legs(path: Path) -> Dict[str, Dict[str, Any]]:
    if not Path(path).is_file():
        refuse(f"{path}: missing — run engine_identity_table.py first")
    legs = ac.read_json(path)
    if not isinstance(legs, dict) or not legs:
        refuse(f"{path}: not a non-empty object")
    return legs


# --------------------------------------------------------------------------- #
# per-leg store completeness                                                    #
# --------------------------------------------------------------------------- #

def result_rc(root: AttemptRoot) -> Optional[str]:
    p = root.path / ac.RESULT_FILE
    if not p.is_file():
        return None
    last = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not last:
        return None
    m = _RC_RE.search(last[-1])
    return m.group(1) if m else None


def store_status(root: AttemptRoot, store: StoreRead, *, s: int) -> Tuple[str, Optional[str]]:
    """`(status, reason)` from the attempt root and the store's own artefacts (the audit and identity verdicts
    are layered on by :func:`build`)."""
    if ac.complete_record(root) is None:
        return "missing", f"attempt incomplete: no {ac.COMPLETE_FILE} on the accepted root"
    if not store.complete:
        return "missing", f"store incomplete: {store.completeness_note}"
    if root.leg == "misalign":
        rc = result_rc(root)
        if rc is None:
            return "missing", f"misalign: no {ac.RESULT_FILE} line with rc= on the root"
        if rc != "0":
            return "missing", f"misalign: RESULT rc={rc}"
    return COMPLETE, None


def _stage_label(stage: str) -> str:
    return "" if stage == ac.STAGE_BASE else f"rung {stage}: "


def _fold(verdicts: Sequence[str]) -> Optional[str]:
    """ONE `audit_verdict` column from the shard verdicts read at a stage: the value they share, `mixed`, or None when none was read."""
    if not verdicts:
        return None
    return verdicts[0] if len(set(verdicts)) == 1 else "mixed"


def verify_stage(*, leg: str, arm: str, s: int, stage: str, planned: Sequence[Mapping[str, Any]], bound: Mapping[str, AttemptRoot],
                 store_of: Mapping[Tuple[str, str, str, int], StoreRead], audits: Mapping[Tuple[str, str, int, str, str], Dict[str, str]]
                 ) -> Tuple[Optional[str], List[Tuple[AttemptRoot, StoreRead]], Optional[str]]:
    """ONE stage of an arm-run's ancestry against the launch plan's inventory: EVERY planned standup serving the arm-run
    at this stage has an accepted attempt bound (`bound[standup_id]`), that attempt holds the arm-run's store and is
    COMPLETE, the store is complete by its writer's own rule (misalign: RESULT rc=0), and the request-evidence verdict for
    (leg, arm, s, stage, shard) is AGREES and was filed for THAT attempt. Returns `(first shortfall | None, the verified
    (root, store) pairs, the folded audit_verdict — None when no verdict was read)` — every shortfall names the stage,
    the standup / shard and the attempt."""
    label = _stage_label(stage)
    seen: List[str] = []
    verified: List[Tuple[AttemptRoot, StoreRead]] = []
    if not planned:
        return f"{label}the launch plan declares no standup serving this arm-run at this stage", verified, _fold(seen)
    for prow in planned:
        sid = str(prow["standup_id"])
        root = bound.get(sid)
        if root is None:
            return f"{label}planned standup {sid} has no accepted attempt", verified, _fold(seen)
        st = store_of.get((sid, leg, arm, s))
        if st is None:
            return f"{label}[{root.shard}] accepted attempt {root.attempt_id} of standup {sid} holds no store for this arm-run", verified, _fold(seen)
        status, why = store_status(root, st, s=s)
        if status != COMPLETE:
            return f"{label}[{root.shard}] {why}", verified, _fold(seen)
        a = audits.get((leg, arm, s, stage, root.shard))
        if a is None:
            return ("unaudited: no request-evidence verdict for this arm-run" if stage == ac.STAGE_BASE
                    else f"{label}unaudited: no request-evidence verdict for this arm-run at stage {stage}"), verified, _fold(seen)
        if a.get("attempt_id") != root.attempt_id:
            return (f"{label}[{root.shard}] the audit verdict is for attempt {a.get('attempt_id')!r}, not the accepted "
                    f"{root.attempt_id!r} — stale evidence"), verified, _fold(seen)
        seen.append(a.get("verdict", "unaudited"))
        if a.get("verdict") != "AGREES":
            return f"{label}audit {a.get('verdict')}: [{root.shard}] {a.get('problem') or 'no reason recorded'}", verified, _fold(seen)
        verified.append((root, st))
    return None, verified, _fold(seen)


def build(stores_root: Path, *, expected: Path, launch_plan: Path, audit_summary: Path, identity_legs: Path,
          petri_behaviours: Sequence[str], carry_over: Optional[Path] = None) -> List[Dict[str, Any]]:
    """``carry_over`` (fj5): `RS/carry_over.json` -- the validated carry mapping that binds a CARRIED attempt's historical root (its
    own W / wave / standup id, an S5 FULL source included) to the planned standup it stands for; without it a carried root refuses as
    an attempt of another tree (loud, never silently admitted)."""
    legs, arm_runs = ac.read_expected(expected)
    plan = ac.read_launch_plan(launch_plan)
    audits = read_audit_summary(audit_summary)
    identity = read_identity_legs(identity_legs)
    roots = ac.discover_attempt_roots(stores_root)  # EVERY stage: the rung roots are the final materialisation's ancestry
    carried = ac.read_carry_mapping(carry_over, plan)
    bound = ac.bind_roots_to_plan(roots, plan, carried=carried)     # every pulled attempt is a planned standup's accepted attempt (or a carried one)
    if "petri" in legs and not petri_behaviours:
        refuse("--petri-behaviours is required when the expected legs include petri")
    trig_static, trig_gsm8k = act.trigger_share_static(), act.trigger_share_gsm8k()
    reads = ac.RootReads()  # every store / ledger / evidence file of the wave read ONCE for this build
    # (standup_id, leg, arm, s) → the arm-run's store on that accepted attempt (the seed is the store's own)
    store_of: Dict[Tuple[str, str, str, int], StoreRead] = {}
    for leg in legs:
        cleg = ac.CONTAINER_LEG_OF[leg]
        for sid, root in bound.items():
            if root.leg != cleg:
                continue
            for st in reads.stores(root, leg, petri_behaviours):
                s = ac.seed_of_store(root, st)
                key = (sid, leg, st.arm, s)
                if key in store_of:
                    refuse(f"{root.label}: two {leg} stores for arm {st.arm!r} s{s} on attempt {root.attempt_id}")
                store_of[key] = st
    rows: List[Dict[str, Any]] = []
    for leg in legs:
        cleg = ac.CONTAINER_LEG_OF[leg]
        leg_identity = identity.get(cleg)
        if leg_identity is None:
            refuse(f"{identity_legs}: no identity verdict for container leg {cleg!r}")
        for run in arm_runs:
            arm, view, s = str(run["arm"]), str(run["view"]), int(run["s"])
            planned_base = plan.serving(cleg, view, ac.STAGE_BASE, arm, s)
            if not planned_base:
                refuse(f"{plan.path}: no planned base standup of ({cleg}, {view}) serves arm-run ({leg}, {arm}, s{s}) — the expected "
                       "arm-runs and the launch plan disagree")
            status, reason = COMPLETE, None
            final_stage: Optional[str] = None
            ladder_state: Optional[str] = None
            rungs_present: Optional[List[int]] = None
            verified: List[Tuple[AttemptRoot, StoreRead]] = []
            audit_verdict = "unaudited"
            # the BASE stage first — every planned base standup accepted, holding the store, complete, its verdict AGREES —
            # then the FINAL accepted materialisation through the ladder / completion rules (`at_cap_table.row_for` is the
            # authority) and every rung stage the final rung rests on
            why, ok, folded = verify_stage(leg=leg, arm=arm, s=s, stage=ac.STAGE_BASE, planned=planned_base, bound=bound,
                                           store_of=store_of, audits=audits)
            if folded is not None:
                audit_verdict = folded
            if why is not None:
                status, reason = "missing", why
            else:
                verified.extend(ok)
                final_stage = ac.STAGE_BASE
                ladder = act.row_for(leg, arm, s, view, roots, petri=petri_behaviours, trig_static=trig_static, trig_gsm8k=trig_gsm8k,
                                     reads=reads)
                ladder_state = str(ladder["ladder_state"])
                rungs_present = list(ladder["rungs_present"])
                rung_stages = [str(c) for c in rungs_present[1:]] if ladder["applies"] else []
                if ladder_state.startswith("ladder_gap"):
                    status, reason = "missing", f"ladder: {ladder_state} — a rung the ladder requires below the final one was never materialised"
                elif ladder_state not in FINAL_LADDER_STATES:
                    status, reason = "missing", (f"ladder: {ladder_state} — the final rung is not materialised (rungs present: "
                                                 f"{rungs_present}, final at-cap share {ladder['share_at_cap_final']})")
                for stage in rung_stages:
                    why, ok, folded = verify_stage(leg=leg, arm=arm, s=s, stage=stage, planned=plan.serving(cleg, view, stage, arm, s),
                                                   bound=bound, store_of=store_of, audits=audits)
                    if folded is not None:
                        audit_verdict = folded
                    if why is not None:
                        if status == COMPLETE:
                            status, reason = "missing", why
                        break
                    verified.extend(ok)
                    final_stage = stage
            if status == COMPLETE and leg_identity.get("verdict") != "PASS":
                status, reason = "missing", "identity: " + "; ".join(leg_identity.get("problems") or ["engine identity FAIL"])[:400]
            final_hits = [(r, st) for r, st in verified if r.stage == final_stage] if final_stage else []
            rows.append({"leg": leg, "arm": arm, "s": s, "view": view, "status": status if status == COMPLETE else f"missing({reason})",
                         "reason": reason, "final_stage": final_stage if status == COMPLETE else None, "ladder_state": ladder_state,
                         "rungs_present": rungs_present, "n_stages_verified": len({r.stage for r, _ in verified}),
                         "attempt_id": ",".join(dict.fromkeys(r.attempt_id for r, _ in verified)) if verified else None,
                         "n_rows": sum(st.n_rows for _, st in final_hits) if final_hits else None, "audit_verdict": audit_verdict,
                         "store_dir": ";".join(str(st.store_dir) for _, st in final_hits) if final_hits else None})
    return rows


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--stores", type=Path, required=True)
    b.add_argument("--expected", type=Path, required=True)
    b.add_argument("--launch-plan", type=Path, required=True, help="RS/launch_plan.json — the declared standup / shard inventory")
    b.add_argument("--audit-summary", type=Path, required=True)
    b.add_argument("--identity-legs", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--petri-behaviours", default="")
    b.add_argument("--carry-over", type=Path, default=None, help="RS/carry_over.json — binds CARRIED attempts (D112) to their planned standups")
    e = sub.add_parser("expected")
    e.add_argument("--view-config", type=Path, action="append", required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "expected":
        exp = build_expected(args.view_config)
        ac.write_json(args.out, exp)
        print(f"[coverage] expected: {len(exp['legs'])} legs × {len(exp['arm_runs'])} arm-runs → {args.out}")
        return ac.EXIT_OK
    petri = [x.strip() for x in args.petri_behaviours.split(",") if x.strip()]
    rows = build(args.stores, expected=args.expected, launch_plan=args.launch_plan, audit_summary=args.audit_summary,
                 identity_legs=args.identity_legs, petri_behaviours=petri, carry_over=args.carry_over)
    ac.write_csv(args.out, rows, COLUMNS)
    n_missing = sum(1 for r in rows if r["status"] != COMPLETE)
    print(f"[coverage] {len(rows)} arm-run(s): {len(rows) - n_missing} complete, {n_missing} missing → {args.out}")
    for r in rows:
        if r["status"] != COMPLETE:
            print(f"  {r['leg']}/{r['arm']}__s{r['s']}: {r['status']}")
    return ac.EXIT_OK if n_missing == 0 else ac.EXIT_FAILED


if __name__ == "__main__":
    sys.exit(ac.run_main(main))
