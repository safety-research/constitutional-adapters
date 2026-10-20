#!/usr/bin/env python3
"""The Qwen seed-replicate battery — the at-cap table.

    at_cap_table.py --stores RS/stores --expected RS/expected_arm_runs.json --out RS/tables/at_cap.csv [--petri-behaviours b1,b2]

Per (leg, arm, seed): the FINAL-rung at-cap share and WHICH rung the headline table reads, derived from the pulled
stores themselves (the base attempt root + every rung root of the same (container leg, view), stage = the rung cap):

* static ×4 — the base rows' `at_cap` (`finish_reason == "length"`) and, per rung root, its `escalation.json`
  (`n_at_cap_after` / `n_rows`, the rung store being a MIXED-CAP store of re-drawn + copied-forward rows); the rung the
  table reads is the highest rung root present; the 1 % rule is `ladder_decide.TRIGGER_SHARE` (imported);
* over-refusal — each rung store's `gen_stats.json[<control>]` (`n_at_cap` / `n_total`; the final rung = the highest
  rung whose store carries a complete block for THAT control — `final_rung.py`'s rule, per (cell, control));
* misalign — the base `agentic/<cell>/responses.jsonl` rows at cap, the rung root's rows replacing the selected
  pairs (`truncated_selection`'s zlib rule imported from the runner): `share_at_cap_final` counts EVERY row still at
  cap after the rung (a loop stays at cap by construction — `misalign_ladder_merge`'s `at_cap_after_all`), and
  `share_truncated_eligible` the at-cap rows the rule would still send up;
* GSM8K — each rung store's `outcomes/<task>.json` header (`n_at_cap` / `n_attempted_samples`); the 1 % rule is
  `inspect_rung.GSM8K_TRIGGER_THRESHOLD` (imported);
* BoN / multi-turn / Petri — a FIXED cap (no ladder): the share of the arm's TARGET completions recorded with
  `finish_reason == "length"` (BoN rows carry it themselves) over the ATTEMPT INVENTORY on the row's request path (fz5):
  the usage ledger's served rows where the bound plan declares a ledger or one exists undeclared (`finish_reason`, cap from
  `max_tokens_requested`), else — the ordinary Modal shape of every non-GSM8K leg, `RoutingProxy` writes no ledger — the
  response hook's COMPLETIONS (`response_evidence.jsonl`: 2xx + a real completion id; a 503-then-retry's failed attempt is an
  attempt, never a completion) bound to their REQUEST RECORDS, whose `max_tokens` is the cap the driver asked for;
  `rung_read = fixed:<cap>`. The identity of the partition (record ↔ attempt, either inventory) is a PREREQUISITE of the share
  (`audit_common.partition_identity_problems`), the seed attribution on the Arditi engine comes from the record's `extra.s`,
  and a served completion with no `finish_reason` or a record with no `max_tokens` REFUSES — never a default;
* MMLU — `max_tokens 1`; the at-cap notion does not apply and the row says so (`applies false`).

Every row carries the qualification the request-evidence verdicts carry (`ledger` ∈ `declared` | `present_undeclared` |
`not_on_path`, `attempt_source` ∈ `ledger` | `response_evidence`, `reconciliation_contract`), resolved over the accepted attempt
roots the row rests on (`--plan` binds the topology to the launch plan's children, as the audit's does; without it the plan
builders' leg rule decides); a DECLARED ledger that is missing (GSM8K) refuses by name, it never makes the row ledger-less.

`ladder_state` is one of `exit_le_1pct` | `exit_ceiling` | `escalate_pending(<next cap>)` (the base or a rung is
above the trigger and no higher rung root is present — the table is NOT final for that arm-run) | `ladder_gap(...)` (the
rungs present are not the ladder's prefix: a rung root above a predecessor that was never materialised — not final) |
`fixed_cap` | `not_applicable`. The expected set (`coverage_table.py expected`) fixes the 405 rows; an arm-run with no store is
`no_store` (never blank). Exit 1 when any arm-run is `escalate_pending` or `no_store`; exit 2 on missing inputs.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_common as ac  # noqa: E402
from audit_common import C, AttemptRoot, StoreRead, refuse  # noqa: E402

COLUMNS: Tuple[str, ...] = ("leg", "arm", "s", "view", "applies", "base_cap", "final_cap", "rung_read", "rungs_present", "n",
                            "n_at_cap_final", "share_at_cap_final", "share_truncated_eligible?", "over_1pct", "ladder_state",
                            "source", "attempt_ids",
                            # the attempt inventory the row's request path carries — the same three keys every
                            # request-evidence verdict records (`request_evidence_audit.SUMMARY_COLUMNS`), never blank
                            "ledger", "attempt_source", "reconciliation_contract")
QUALIFICATION_KEYS: Tuple[str, ...] = ("ledger", "attempt_source", "reconciliation_contract")
STATIC_LADDER: Tuple[int, ...] = (4096, 8192, 16384)
CONTROLS_LADDER: Tuple[int, ...] = (512, 2048, 8192, 16384)
GSM8K_LADDER: Tuple[int, ...] = (8192, 16384)
MISALIGN_LADDER: Tuple[int, ...] = (3000, 16384)
FIXED_CAP_LEGS: Tuple[str, ...] = ("bon", *ac.MULTITURN_ATTACKS, "petri")


def trigger_share_static() -> float:
    return float(ac.import_by_path("ladder_decide", "scripts/sdfbase/full22/sampled/ladder_decide.py",
                                   what="the ladder_decide rule").TRIGGER_SHARE)


def trigger_share_gsm8k() -> float:
    from sisteer.capeval.tiers import inspect_rung
    return float(inspect_rung.GSM8K_TRIGGER_THRESHOLD)


def _stage_cap(root: AttemptRoot) -> Optional[int]:
    try:
        return int(root.stage)
    except ValueError:
        return None


def _roots_of(roots: Sequence[AttemptRoot], leg: str, view: str) -> Tuple[List[AttemptRoot], Dict[int, List[AttemptRoot]]]:
    base = [r for r in roots if r.leg == leg and r.view == view and r.stage == ac.STAGE_BASE]
    rungs: Dict[int, List[AttemptRoot]] = {}
    for r in roots:
        if r.leg == leg and r.view == view and r.stage != ac.STAGE_BASE:
            cap = _stage_cap(r)
            if cap is not None:
                rungs.setdefault(cap, []).append(r)
    return base, rungs


def _state(share: float, cap: int, ladder: Sequence[int], trigger: float, present: Sequence[int]) -> Tuple[str, bool]:
    """The ladder state of an arm-run whose rungs `present` (the base cap first, then every rung root read) end at `cap`.
    The rungs present must be the ladder's PREFIX: the base at the ladder's first cap and every predecessor
    of the final rung materialised — a base 4096 + a 16384 rung with no 8192 root is `ladder_gap(...)`, never final: the
    16384 store's copied rows would claim an ancestor that was never accepted."""
    over = share > trigger
    prefix = list(ladder[:len(present)])
    if list(present) != prefix:
        missing = [c for c in prefix if c not in present]
        return f"ladder_gap(present {list(present)}, ladder prefix {prefix}, missing {missing})", over
    if not over:
        return "exit_le_1pct", over
    if cap >= ladder[-1]:
        return "exit_ceiling", over
    return f"escalate_pending({ladder[ladder.index(cap) + 1]})", over


def _find_store(roots: Sequence[AttemptRoot], leg: str, arm: str, s: int, petri: Sequence[str], reads: ac.RootReads
                ) -> List[Tuple[AttemptRoot, StoreRead]]:
    """The (root, store) pairs of one arm-run on `roots` (a root holding no store of the leg contributes none)."""
    return [(r, st) for r in roots for st in reads.stores(r, leg, petri) if st.arm == arm and ac.seed_of_store(r, st) == s]


def _share_rows(draws: Sequence[ac.Draw]) -> Tuple[int, int]:
    n = len(draws)
    unread = [d for d in draws if d.at_cap is None and d.finish_reason is None]
    if unread:
        refuse(f"{len(unread)} of {n} draws carry neither at_cap nor finish_reason — the at-cap share cannot be read")
    n_cap = sum(1 for d in draws if (d.at_cap if d.at_cap is not None else d.finish_reason == "length"))
    return n, n_cap


VALID_SEED_BASES = ac.VALID_SEED_BASES


@dataclass(frozen=True)
class ServedAttempt:
    """ONE served TARGET completion of a partition as the at-cap read needs it: the cap the driver asked for and how the
    completion ended. On the ledger inventory both come off the ledger row (`max_tokens_requested`, `finish_reason`); on the
    response inventory the cap comes off the REQUEST RECORD the completion binds to (`max_tokens`, the wire body's
    completion cap — the record is the client's own account of what it asked) and the finish reason off the response hook's
    row. Either may be None when the source did not record it — the readers refuse that by name, never default it."""

    request_id: str
    cap: Optional[int]
    finish_reason: Optional[str]


def _inventory_of(root: AttemptRoot, reads: ac.RootReads, plan: Optional[ac.LaunchPlan]) -> ac.AttemptInventory:
    return reads.attempt_inventory(root, plan)


def served_task_attempts(root: AttemptRoot, *, task: str, vmodel: str, s: int, reads: Optional[ac.RootReads] = None,
                         plan: Optional[ac.LaunchPlan] = None) -> List[ServedAttempt]:
    """The SERVED completions of (task, vmodel) on `root`'s attempt inventory (:func:`audit_common.attempt_inventory`) that
    belong to seed base `s`. The record ↔ attempt identity of the partition is a PREREQUISITE of any share read from it:
    a record with no attempt, an attempt with no record, or a matched pair whose labels disagree is refused here exactly
    as the canonical reconciliation refuses it — over the ledger's rows or, on a `not_on_path` row, the response hook's
    (the same `attempts_by_id` / `attempt_source` topology the audit is given, so a ledger-less leg reconciles records ↔
    response evidence here too). Served = a ledger row not `rejected`, or a response row that is a COMPLETION (2xx + a
    real completion id — a retry's failed attempt is in the inventory and counts for identity, never as a completion).
    On a view root that is every such completion (one seed per engine). On the ARDITI root three seed bases draw one
    task on one vmodel, so a completion belongs to the seed its request-evidence record's `extra.s` names (REQUIRED on
    every task record there) and is admitted to `s` only when it says so; a served completion whose request has no task
    record, or whose record carries no valid `extra.s` (one of :data:`VALID_SEED_BASES`), REFUSES: the per-seed share
    cannot be partitioned, and pooling the three seeds would read 33.3 % for every seed instead of the truth."""
    reads = reads or ac.RootReads()
    inv, records = _inventory_of(root, reads, plan), reads.records(root)
    problems = ac.partition_identity_problems(records, inv.rows, task=task, vmodel=vmodel, attempt_source=inv.attempt_source)
    if problems:
        refuse(f"{root.label}: the ({task!r}, {vmodel!r}) partition is not one inventory ({inv.attempt_source}) — " + "; ".join(problems) +
               " — no at-cap share is read from a partition the request-evidence audit would refuse")
    if inv.attempt_source == "ledger":
        served = [ServedAttempt(str(r[C.LEDGER_REQUEST_ID_FIELD]), r.get("max_tokens_requested"), r.get("finish_reason"))
                  for r in inv.rows if r.get("task") == task and r.get("vmodel") == vmodel and not r.get("rejected") and r.get(C.LEDGER_REQUEST_ID_FIELD)]
        noun = "ledger row(s)"
    else:
        rec_by_id = {str(r["request_id"]): r for r in records if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK and r.get("request_id")}
        served = []
        for r in inv.rows:
            if not (r.get("task") == task and r.get("vmodel") == vmodel and ac.response_row_is_completion(r)):
                continue
            rid = str(r["request_id"])
            rec = rec_by_id.get(rid)  # present: the identity clause above admitted every completion through its record
            if rec is None:
                refuse(f"{root.label}: served completion {rid!r} of ({task!r}, {vmodel!r}) has no task record — the identity clause let an orphan through")
            served.append(ServedAttempt(rid, rec.get("max_tokens"), r.get("finish_reason")))
        noun = "response-evidence completion(s)"
    if root.view != ac.ARDITI_VIEW:
        return served
    s_by_rid = ac.extra_s_by_request_id(records)
    bad: List[Tuple[Any, Any]] = []
    out: List[ServedAttempt] = []
    for a in served:
        v = s_by_rid.get(a.request_id)
        if v is None or v not in VALID_SEED_BASES:
            bad.append((a.request_id, v))
        elif v == s:
            out.append(a)
    if bad:
        refuse(f"{root.label}: {len(bad)} served {task!r} {noun} on the Arditi engine without a valid seed attribution "
               f"(request-evidence extra.s in {list(VALID_SEED_BASES)}; first (request_id, extra.s): {bad[:3]}) — the per-seed "
               "at-cap share cannot be partitioned")
    return out


def _cap_field(inv: ac.AttemptInventory) -> str:
    return "max_tokens_requested" if inv.attempt_source == "ledger" else "the request records' max_tokens"


def _served_cap(atts: Sequence[ServedAttempt], *, where: str, task: str, inv: ac.AttemptInventory) -> int:
    """The ONE cap of a served partition (a fixed-cap leg's cap): several values, a missing value or an empty partition refuse."""
    caps = {a.cap for a in atts}
    if len(caps) != 1 or None in caps:
        refuse(f"{where}: {_cap_field(inv)} is not one value on the {task!r} rows ({sorted(map(str, caps))[:3]})")
    return int(next(iter(caps)))


def _served_share(root: AttemptRoot, *, task: str, vmodel: str, s: int, reads: ac.RootReads, plan: Optional[ac.LaunchPlan]) -> Tuple[int, int, int]:
    """(n, n_at_cap, cap) over the served TASK completions of (task, vmodel) belonging to seed `s` on the row's inventory. A served
    completion whose source recorded no `finish_reason` refuses: the truncation share is never read off a default."""
    inv = _inventory_of(root, reads, plan)
    atts = served_task_attempts(root, task=task, vmodel=vmodel, s=s, reads=reads, plan=plan)
    if not atts:
        refuse(f"{root.label}: the {'usage ledger' if inv.attempt_source == 'ledger' else ac.RESPONSE_EVIDENCE_FILE} holds no served "
               f"completion for task {task!r} vmodel {vmodel!r} seed {s}")
    unread = [a.request_id for a in atts if a.finish_reason is None]
    if unread:
        refuse(f"{root.label}: {len(unread)} of {len(atts)} served {task!r} completion(s) carry no finish_reason (first: {unread[:3]}) — the "
               "at-cap share cannot be read; a missing finish reason is never read as 'stop'")
    return len(atts), sum(1 for a in atts if a.finish_reason == "length"), _served_cap(atts, where=root.label, task=task, inv=inv)


def _qualification(used: Sequence[AttemptRoot], *, reads: ac.RootReads, plan: Optional[ac.LaunchPlan], where: str) -> Dict[str, str]:
    """The qualification of ONE arm-run over every accepted attempt root its reading rests on (the base shard(s) and every
    rung root read): the roots must agree — one arm-run, one contract — else refuse by name (`Evidence.merge`'s rule)."""
    quals = {r.attempt_id: _inventory_of(r, reads, plan).qualification for r in used}
    distinct = {tuple(sorted(q.items())) for q in quals.values()}
    if len(distinct) != 1:
        refuse(f"{where}: the contributing attempt roots reconcile on different attempt inventories ({quals}) — one arm-run, one contract")
    return dict(next(iter(quals.values())))


def row_for(leg: str, arm: str, s: int, view: str, roots: Sequence[AttemptRoot], *, petri: Sequence[str],
            trig_static: float, trig_gsm8k: float, reads: Optional[ac.RootReads] = None,
            plan: Optional[ac.LaunchPlan] = None) -> Dict[str, Any]:
    """The at-cap row of ONE arm-run. `reads` (per build) memoises every store / ledger / evidence read across the rows; `plan`
    (the BOUND launch plan, when the caller has one) is the topology the row's usage ledger is declared by."""
    reads = reads or ac.RootReads()
    cleg = ac.CONTAINER_LEG_OF[leg]
    base_roots, rung_roots = _roots_of(roots, cleg, view)
    common = {"leg": leg, "arm": arm, "s": s, "view": view}
    hits = _find_store(base_roots, leg, arm, s, petri, reads)
    if not hits:
        return {**common, "applies": leg != "mmlu", "base_cap": "?", "final_cap": "?", "rung_read": "none", "rungs_present": [],
                "n": 0, "n_at_cap_final": 0, "share_at_cap_final": "no_store", "share_truncated_eligible": None, "over_1pct": "no_store",
                "ladder_state": "no_store", "source": "none", "attempt_ids": [], **{k: "no_store" for k in QUALIFICATION_KEYS}}
    _, store = hits[0]
    aids = [r.attempt_id for r, _ in hits]
    used: List[AttemptRoot] = [r for r, _ in hits]  # every accepted attempt root the reading rests on, for the qualification
    where = f"{leg}/{arm}/s{s}"

    def qual() -> Dict[str, str]:
        return _qualification(used, reads=reads, plan=plan, where=where)
    if leg == "mmlu":
        return {**common, "applies": False, "base_cap": 1, "final_cap": 1, "rung_read": "fixed:1", "rungs_present": [1],
                "n": len(store.draws), "n_at_cap_final": 0, "share_at_cap_final": 0.0, "share_truncated_eligible": None,
                "over_1pct": False, "ladder_state": "not_applicable", "source": "mmlu: max_tokens 1 prefill, at-cap not applicable",
                "attempt_ids": aids, **qual()}
    if leg in FIXED_CAP_LEGS:
        n = n_cap = 0
        cap: Optional[int] = None
        for hroot, hstore in hits:  # every shard of the arm-run (multi-turn is sharded by goal sub-shard)
            vmodel = ac.read_standup(hroot).vmodel_of(arm)
            inv = _inventory_of(hroot, reads, plan)
            ledgerless = inv.attempt_source != "ledger"
            if leg == "bon":
                hn, hc = _share_rows(hstore.draws)
                hcap = _served_cap(served_task_attempts(hroot, task=ac.BON_TASK, vmodel=vmodel, s=s, reads=reads, plan=plan),
                                   where=hroot.label, task=ac.BON_TASK, inv=inv)
                n, n_cap = n + hn, n_cap + hc
                src = "bon rows finish_reason + request-record cap (D106: ledger not on path)" if ledgerless else "bon rows finish_reason + ledger cap"
            else:
                hcap = None
                for t in hstore.tasks:
                    tn, tc, tcap = _served_share(hroot, task=t, vmodel=vmodel, s=s, reads=reads, plan=plan)
                    n, n_cap = n + tn, n_cap + tc
                    if hcap is not None and tcap != hcap:
                        refuse(f"{hroot.label}: two caps across the {leg} tasks ({hcap}, {tcap})")
                    hcap = tcap
                src = ("response evidence finish_reason (target task completions; D106: ledger not on path)" if ledgerless
                       else "usage ledger finish_reason (target task rows)")
            if cap is not None and hcap != cap:
                refuse(f"{where}: shards disagree on the cap ({cap}, {hcap})")
            cap = hcap
        share = n_cap / n if n else 0.0
        return {**common, "applies": True, "base_cap": cap, "final_cap": cap, "rung_read": f"fixed:{cap}", "rungs_present": [cap],
                "n": n, "n_at_cap_final": n_cap, "share_at_cap_final": round(share, 4), "share_truncated_eligible": None,
                "over_1pct": share > trig_static, "ladder_state": "fixed_cap", "source": src, "attempt_ids": aids, **qual()}
    if leg in ac.STATIC_FAMILIES:
        n = n_cap = 0
        for _, hstore in hits:  # a static leg may be sharded (S3 may add a 2nd shard per view)
            hn, hc = _share_rows(hstore.draws)
            n, n_cap = n + hn, n_cap + hc
        if store.cap is None:
            refuse(f"{store.store_dir}: the static store records no cap")
        base_cap = int(store.cap)
        present = [base_cap]
        final_cap, final_n, final_cap_n, src = base_cap, n, n_cap, "base rows at_cap"
        for cap in sorted(rung_roots):
            rh = _find_store(rung_roots[cap], leg, arm, s, petri, reads)
            if not rh:
                continue
            rroot, rstore = rh[0]
            esc_hits = ac.walk_files(rstore.store_dir, "escalation.json", max_depth=2)
            if not esc_hits:
                refuse(f"{rstore.store_dir}: rung root without escalation.json — not a complete rung store")
            esc = ac.read_json(esc_hits[0])
            if rstore.cap != cap:
                refuse(f"{rstore.store_dir}: the rung store's cap {rstore.cap} (escalation.json cap_to) != the root's stage {cap}")
            present.append(cap)
            final_cap, final_n, final_cap_n = cap, int(esc["n_rows"]), int(esc["n_at_cap_after"])
            src = "rung escalation.json n_at_cap_after/n_rows"
            aids.append(rroot.attempt_id)
            used.append(rroot)
        share = final_cap_n / final_n if final_n else 0.0
        state, over = _state(share, final_cap, STATIC_LADDER, trig_static, present=present)
        return {**common, "applies": True, "base_cap": base_cap, "final_cap": final_cap, "rung_read": f"cap{final_cap}",
                "rungs_present": present, "n": final_n, "n_at_cap_final": final_cap_n, "share_at_cap_final": round(share, 4),
                "share_truncated_eligible": None, "over_1pct": over, "ladder_state": state, "source": src, "attempt_ids": aids, **qual()}
    if leg in ac.CONTROLS:
        base_cap = _controls_cap(hits[0][0], store, leg, s=s, reads=reads, plan=plan)
        gs = _gen_stats(store, leg)
        present = [base_cap]
        final_cap, n, n_cap, src = base_cap, int(gs["n_total"]), int(gs["n_at_cap"]), "gen_stats.json n_at_cap/n_total"
        for cap in sorted(rung_roots):
            rh = _find_store(rung_roots[cap], leg, arm, s, petri, reads)
            if not rh:
                continue
            rroot, rstore = rh[0]
            _controls_cap(rroot, rstore, leg, s=s, reads=reads, plan=plan)  # == cap: the stage check inside holds the store to the root
            rgs = _gen_stats(rstore, leg)
            present.append(cap)
            final_cap, n, n_cap = cap, int(rgs["n_total"]), int(rgs["n_at_cap"])
            aids.append(rroot.attempt_id)
            used.append(rroot)
        share = n_cap / n if n else 0.0
        state, over = _state(share, final_cap, CONTROLS_LADDER, trig_static, present=present)
        return {**common, "applies": True, "base_cap": base_cap, "final_cap": final_cap, "rung_read": f"cap{final_cap}",
                "rungs_present": present, "n": n, "n_at_cap_final": n_cap, "share_at_cap_final": round(share, 4),
                "share_truncated_eligible": None, "over_1pct": over, "ladder_state": state, "source": src, "attempt_ids": aids, **qual()}
    if leg == "gsm8k":
        hdr = _outcomes_header(store)
        base_cap = int(hdr["max_tokens"])
        present = [base_cap]
        final_cap, n, n_cap = base_cap, int(hdr["n_attempted_samples"]), int(hdr["n_at_cap"])
        for cap in sorted(rung_roots):
            rh = _find_store(rung_roots[cap], leg, arm, s, petri, reads)
            if not rh:
                continue
            rroot, rstore = rh[0]
            rh_hdr = _outcomes_header(rstore)
            present.append(cap)
            final_cap, n, n_cap = cap, int(rh_hdr["n_attempted_samples"]), int(rh_hdr["n_at_cap"])
            aids.append(rroot.attempt_id)
            used.append(rroot)
        share = n_cap / n if n else 0.0
        state, over = _state(share, final_cap, GSM8K_LADDER, trig_gsm8k, present=present)
        return {**common, "applies": True, "base_cap": base_cap, "final_cap": final_cap, "rung_read": f"cap{final_cap}",
                "rungs_present": present, "n": n, "n_at_cap_final": n_cap, "share_at_cap_final": round(share, 4),
                "share_truncated_eligible": None, "over_1pct": over, "ladder_state": state,
                "source": "outcomes header n_at_cap/n_attempted_samples", "attempt_ids": aids, **qual()}
    if leg.startswith("misalign_"):
        from sisteer.capeval.misalign.runner import RERUN_MIN_ZLIB_RATIO, zlib_ratio
        base_cap = _misalign_cap(store)
        base_draws = {(d.item_id, d.sample_idx): d for d in store.draws}
        at_cap = {k for k, d in base_draws.items() if d.finish_reason == "length"}
        degenerate = {k for k in at_cap if zlib_ratio(base_draws[k].text or "") < RERUN_MIN_ZLIB_RATIO}
        present = [base_cap]
        final_cap = base_cap
        still_at_cap = set(at_cap)
        src = "base rows finish_reason (zlib rule for eligibility)"
        for cap in sorted(rung_roots):
            rh = _find_store(rung_roots[cap], leg, arm, s, petri, reads)
            if not rh:
                continue
            rroot, rstore = rh[0]
            present.append(cap)
            final_cap = cap
            rerun = {(d.item_id, d.sample_idx): d for d in rstore.draws}
            still_at_cap = {k for k in at_cap if k not in rerun or rerun[k].finish_reason == "length"}
            still_at_cap |= {k for k, d in rerun.items() if d.finish_reason == "length" and k in at_cap}
            src = "base rows ∪ rung rows (selected pairs replaced; at-cap after the rung)"
            aids.append(rroot.attempt_id)
            used.append(rroot)
        n = len(base_draws)
        eligible = {k for k in still_at_cap if k not in degenerate}
        share = len(still_at_cap) / n if n else 0.0
        share_elig = len(eligible) / n if n else 0.0
        state, over = _state(share_elig, final_cap, MISALIGN_LADDER, trig_static, present=present)
        return {**common, "applies": True, "base_cap": base_cap, "final_cap": final_cap, "rung_read": f"cap{final_cap}",
                "rungs_present": present, "n": n, "n_at_cap_final": len(still_at_cap), "share_at_cap_final": round(share, 4),
                "share_truncated_eligible": round(share_elig, 4), "over_1pct": over, "ladder_state": state, "source": src,
                "attempt_ids": aids, **qual()}
    refuse(f"unknown leg {leg!r}")


def _gen_stats(store: StoreRead, control: str) -> Dict[str, Any]:
    p = store.store_dir / "gen_stats.json"
    if not p.is_file():
        refuse(f"{store.store_dir}: no gen_stats.json")
    gs = ac.read_json(p)
    blk = gs.get(control) if isinstance(gs, dict) else None
    if not isinstance(blk, dict) or blk.get("n_at_cap") is None or blk.get("n_total") is None or blk.get("cap") is None:
        refuse(f"{p}: no complete {control!r} block with cap / n_at_cap / n_total — the writer's _meta_stats stamps all three; no default is read")
    return blk


def _controls_cap(root: AttemptRoot, store: StoreRead, control: str, *, s: int, reads: ac.RootReads, plan: Optional[ac.LaunchPlan] = None) -> int:
    """The controls store's cap held to EVERY declaration of it — the spool header's `gen_cfg.max_new_tokens`
    (the reader's `store.cap`, itself held equal to `gen_stats[control].cap`), the cap the control's requests drawn on THIS root
    ASKED for (the usage ledger's `max_tokens_requested`, or on a ledger-less row the request records' `max_tokens`), and
    a rung root's stage — never a default (the old `or 512` certified a cap nothing declared)."""
    gs = _gen_stats(store, control)
    if store.cap is None:
        refuse(f"{store.store_dir}: the controls store records no cap")
    cap = int(store.cap)
    if int(gs["cap"]) != cap:
        refuse(f"{store.store_dir}/gen_stats.json: {control} cap {gs['cap']} != the spool header's gen_cfg.max_new_tokens {cap}")
    vmodel = ac.read_standup(root).vmodel_of(store.arm)
    atts = served_task_attempts(root, task=ac.CONTROLS_TASK_OF[control], vmodel=vmodel, s=s, reads=reads, plan=plan)
    caps = {a.cap for a in atts}
    if atts and caps != {cap}:
        inv = _inventory_of(root, reads, plan)
        refuse(f"{root.label}: {'the ledger' if inv.attempt_source == 'ledger' else 'the request records'}'"
               f"{'s max_tokens_requested' if inv.attempt_source == 'ledger' else ' max_tokens'} {sorted(map(str, caps))} on the {control} "
               f"requests drawn here != the store's cap {cap}")
    if root.stage != ac.STAGE_BASE and _stage_cap(root) != cap:
        refuse(f"{root.label}: rung stage {root.stage} != the store's cap {cap} — the root and its store disagree on the rung")
    return cap


def _outcomes_header(store: StoreRead) -> Dict[str, Any]:
    from sisteer.capeval.tinycap import completion as _completion
    p = store.store_dir / C.TASK_OUTCOMES_DIR / _completion.marker_filename(ac.GSM8K_TASK)
    if not p.is_file():
        refuse(f"{store.store_dir}: no {p.name} outcomes record")
    hdr = ac.read_json(p)
    for k in ("max_tokens", "n_attempted_samples", "n_at_cap"):
        if hdr.get(k) is None:
            refuse(f"{p}: header lacks {k}")
    return hdr


def _misalign_cap(store: StoreRead) -> int:
    meta = ac.read_json(store.store_dir / "meta.json")
    cap = ((meta.get("params") or {}).get("max_tokens") if isinstance(meta, dict) else None) or (meta.get("max_tokens") if isinstance(meta, dict) else None)
    if cap is None:
        refuse(f"{store.store_dir}/meta.json: no params.max_tokens")
    return int(cap)


def build(stores_root: Path, *, expected: Path, petri: Sequence[str], plan: Optional[ac.LaunchPlan] = None) -> List[Dict[str, Any]]:
    legs, arm_runs = ac.read_expected(expected)
    if "petri" in legs and not petri:
        refuse("--petri-behaviours is required when the expected legs include petri")
    roots = ac.discover_attempt_roots(stores_root)
    trig_static, trig_gsm8k = trigger_share_static(), trigger_share_gsm8k()
    reads = ac.RootReads()  # every store / ledger / evidence file read ONCE for this build
    rows = []
    for leg in legs:
        for run in arm_runs:
            rows.append(row_for(leg, str(run["arm"]), int(run["s"]), str(run["view"]), roots, petri=petri,
                                trig_static=trig_static, trig_gsm8k=trig_gsm8k, reads=reads, plan=plan))
    return rows


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stores", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--petri-behaviours", default="")
    ap.add_argument("--plan", type=Path, default=None,
                    help="RS/launch_plan.json — the BOUND plan whose children declare each row's usage ledger (D106); without it the plan "
                         "builders' leg contract decides, and the row says so")
    args = ap.parse_args(argv)
    petri = [x.strip() for x in args.petri_behaviours.split(",") if x.strip()]
    plan = ac.read_launch_plan(args.plan) if args.plan else None
    rows = build(args.stores, expected=args.expected, petri=petri, plan=plan)
    ac.write_csv(args.out, rows, COLUMNS)
    bad = [r for r in rows if r["ladder_state"] == "no_store" or str(r["ladder_state"]).startswith(("escalate_pending", "ladder_gap"))]
    print(f"[at-cap] {len(rows)} arm-run(s): {len(rows) - len(bad)} final, {len(bad)} not final → {args.out}")
    for r in bad:
        print(f"  {r['leg']}/{r['arm']}__s{r['s']}: {r['ladder_state']} (share {r['share_at_cap_final']})")
    return ac.EXIT_OK if not bad else ac.EXIT_FAILED


if __name__ == "__main__":
    sys.exit(ac.run_main(main))
