#!/usr/bin/env python3
"""reproduce/lib/admission.py — the GLOBAL unsteered-first gate. stdlib only.

No steered EVAL wave (any view != u0, any family) is submitted until EVERY logical unit of the model's frozen baseline
inventory (`<state>/gate/expected_units.json`, frozen by 05_stage_roster.sh from the SAME unit plan 06 submits — lib/stage.py
`unit_plan`) is
  * `registered` in the binding ledger (06 bound it: unit -> job id -> run id(s) read from the job's own pending file),
  * complete in the registry under exactly the FROZEN bindings, PER EVAL INSIDE THE UNIT (a static pair unit covers both of
    its attacks; a padded level covers the four statics): every bound run row is complete (lifecycle_state judged|ingested),
    names one of the unit's evals, and carries that eval's frozen model_id / code_sha / instrument_id — a baseline judged under
    another code sha or instrument is not this baseline; when the eval freezes a `decode_policy_ref`, the row's
    `decode_policy_id` must be NON-NULL, resolve to that ref, and the row's `decode_policy_match` must be `exact` (a null id or
    an `ambiguous` / `no_match` / `unrecorded` derivation is refused); every expected eval has at least one bound run,
  * COVERED by WORK IDENTITY, from the family's AUTHORITATIVE artifact (`<state>/gate/coverage/<unit>.json`, collected by 06 --gate
    through lib/coverage_probe.py on every gate check; the record is TIED to the binding's run ids, so coverage
    collected for other runs never counts): per shard the set of UNIQUE COMPLETED goal ids (error outcomes excluded) must contain
    every goal id the freeze expects of that shard (the producer's own goal recipe), the BoN climb's expected set is its run-time
    manifest's, the tinycap stores must hold the item set's declared count of unique item ids (and match its ids sha256 when the
    seeds record one), the agentic runner every scenario with its draws — never a sum of heterogeneous JSONL line counts; a Petri
    cell store short of a frozen `<behavior>/<scenario>` refuses by name,
  * verdicted: `<state>/gate/<family>.json` carries `verdict: pass` (06 --gate <family> after the daemon graded it) — and the verdict is
    BOUND TO THE RUNS IT RULED: its `run_ids` (what 06 --gate recorded: the family's registered bindings
    at ruling time, stage.py family_bound_run_ids) must equal AS A SET the run ids the family's units are bound to NOW; a family whose
    bindings moved since the ruling (a resubmission re-bound a unit; the bind-only re-gate wrote no verdict, or the verdict step failed
    after it) refuses by name listing the run ids added and removed and the remedy (re-gate with --headline/--range), and a verdict with
    no or empty `run_ids` refuses by name (re-rule this family). The tie is to RUNS, never to a coverage generation: a coverage refresh,
    a new probe sha or a re-read over the SAME runs is new reading, not new information, and still admits (the reader may improve
    under a frozen wave). NOT closed by this: the same runs re-ruled with an out-of-range headline whose verdict
    write FAILED leaves the old pass standing — 06 --gate's loud non-zero exit on that step is the signal. A pass ruled on the registry's
    grades (`06 --gate --verdict`: it records `headline_legs`) is bound to those GRADES too: the family's unsteered legs are re-read now
    (lib/gate_headline.py read_legs, the reader that ruled), and a leg whose judgment was since superseded or abandoned (a re-grade begun),
    another judgment or value, a leg added or dropped, or a leg that no longer reads at all refuses the family by name until it is ruled
    again. A verdict recorded by hand (`--headline/--range`) carries no legs and stays bound to its runs alone.
A never-submitted unit (`unbound`), a submitted-but-unregistered one, a missing shard, a binding mismatch or a failing
verdict REFUSES the launch BY NAME (exit 3). `--force-gate <unit>=<reason>` is the recorded override (gate/forced.jsonl).
NOT gated: HPO, fits, difference and refusal-direction construction. BCB/BFCL (Modal) gate only their own steered
runs (07_bcb_bfcl.sh keeps its own two-unit inventory under gate/bcb_bfcl/).
A bound run the PRODUCER retired as a duplicate registration of ONE store — the BoN climb registration the store512's supersedes,
filed by the closer as a `run_state superseded` record (reason duplicate_registration, extra.superseded_by_run_id; no ruling, by
design) — is admitted through that successor as a ruled one is, once `supersession` has CHECKED the claim against the registry's
rows (store_uri, eval, object and model equal on both), and its coverage is read at the unit's own frozen artifact path (the work
never moved). And a climb unit's coverage must name EVERY expected shard PER RUN and PER CELL: since the companion key the array's
four tasks register one run per cell, which 06 --gate probes ONCE PER BOUND TASK from the coverage_probe.py at the inventory's frozen
sha (it reads one --shard-index per call) and `stage.py coverage` folds into one document naming the tasks asked (`shard_indices`;
before that fold, one call with the flag repeated read the last task's shard alone from the frozen probe); each bound run's document
must read the shard of every array task the binding names for it, and each cell's runs together every expected shard — a shard read
for another run or cell stands in for nothing (one union over the eval let a whole cell cover its neighbour's unread shards); a
document of one shard for a run bound to several tasks refuses by name.

    python admission.py check --state-root DIR --registry PATH|none [--force-gate LWID=REASON]... [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage import (binding_state, bound_array_tasks, contract_dict_sha256, coverage_contract, coverage_path, effective_contract,  # noqa: E402
                   family_bound_run_ids, hold_out_receipt_matches, hold_out_receipts, ledger_sha_provenance, read_amendments,
                   reconcile_provenance, unit_recorded_sha)
import coverage_probe as CP  # noqa: E402  — DONE_PREDICATES: the coverage contract's predicate half

COMPLETE_STATES = ("judged", "ingested")
INVENTORY_SCHEMA = 2


class AdmissionRefused(SystemExit):
    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__(3)


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


RUN_COLUMNS = ("run_id", "lifecycle_state", "model_id", "code_sha", "instrument_id", "decode_policy_id", "decode_policy_match", "eval_id", "store_uri", "object_id")


def frozen_object_id(unit: dict, cell: Optional[str]) -> Optional[str]:
    """The unit's FROZEN object pin for a bound run's cell (the inventory's `objects: {cell: object_id}` map) when the freeze carries one; None
    otherwise (the u0 inventories carry none: their cells are `none` and the prompted baselines, whose object ids the registry mints). When a
    pin exists, the run's — and a successor's — object_id must equal it."""
    objs = unit.get("objects") or {}
    if not cell or not isinstance(objs, dict):
        return None
    v = objs.get(cell)
    return str(v) if v else None


def registry_row(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    cur = conn.execute(f"SELECT {', '.join(RUN_COLUMNS)} FROM run WHERE run_id = ?", (run_id,))
    r = cur.fetchone()
    if r is None:
        return None
    return dict(zip(RUN_COLUMNS, r))


def policy_ref(conn: sqlite3.Connection, policy_id: str) -> Optional[str]:
    """The human `policy_ref` of a run row's decode_policy_id (the registry's decode_policy table); None when the table has no
    such row — the caller then reports it rather than guessing."""
    try:
        r = conn.execute("SELECT policy_ref FROM decode_policy WHERE policy_id = ?", (policy_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return r[0] if r else None


def supersession(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    """The successor a `supersede` RULING names for a superseded run — {successor, ruling_id, source}: the run row's
    own `superseded_by_run_id` when the registry set it (the schema column), else the unexpired `supersede` ruling whose scope names the run
    (scope_json.run_id, or scope_json.match.run_ids) and whose target_json names `superseded_by_run_id` (the ruling is authoritative; the
    run_state note only repeats it). None when nothing names a successor; {"ambiguous": [...]} when rulings name more than
    one. A registry without the ruling table or the column (a fixture, an older schema) names none through that path.
    When neither the column nor a ruling names one, the PRODUCER's own duplicate-registration
    record may (`producer_supersession`: run_state_event, reason duplicate_registration — checked against the rows, never trusted; `ruling_id`
    None). A ruling that names a successor always wins over the record: the ruling is authoritative."""
    try:
        r = conn.execute("SELECT superseded_by_run_id, status_ruling_id FROM run WHERE run_id = ?", (run_id,)).fetchone()
    except sqlite3.OperationalError:
        r = None
    col_succ = str(r[0]) if r and r[0] else None
    col_ruling = str(r[1]) if r and r[1] else None

    def _ruling(rid_: str, sj, tj, action, expired):
        try:
            sc = json.loads(sj or "{}") or {}; tg = json.loads(tj or "{}") or {}
        except (json.JSONDecodeError, AttributeError, TypeError):
            sc, tg = {}, {}
        names = ({sc.get("run_id")} | set((sc.get("match") or {}).get("run_ids") or [])) if isinstance(sc, dict) else set()
        return {"ruling_id": str(rid_), "action": action, "expired_utc": expired, "names_run": run_id in names,
                "successor": (str(tg.get("superseded_by_run_id")) if isinstance(tg, dict) and tg.get("superseded_by_run_id") else None)}
    try:
        rows = conn.execute("SELECT ruling_id, scope_json, target_json, action, expired_utc FROM ruling ORDER BY ruled_utc, ruling_id").fetchall()
    except sqlite3.OperationalError:
        rows = None
    if col_succ:
        # admission-fix-2: the column is a POINTER; its authority is the ruling it links to — resolved and checked, never trusted
        if rows is None:
            return {"refused": f"superseded_by_run_id names {col_succ} but the registry carries no ruling table to grant it (absent authority)"}
        all_r = [_ruling(*x) for x in rows]
        if col_ruling:
            link = [x for x in all_r if x["ruling_id"] == col_ruling]
            if not link:
                return {"refused": f"superseded_by_run_id names {col_succ} and status_ruling_id {col_ruling}, but no such ruling exists (absent authority)"}
        else:
            link = [x for x in all_r if x["action"] == "supersede" and not x["expired_utc"] and x["names_run"] and x["successor"] == col_succ]
            if not link:
                return {"refused": f"superseded_by_run_id names {col_succ} but no ACTIVE `supersede` ruling names this run with that successor (absent authority)"}
        x = link[0]
        if x["action"] != "supersede":
            return {"refused": f"superseded_by_run_id names {col_succ} but the linked ruling {x['ruling_id']} is a {x['action']!r} ruling, not a supersede"}
        if x["expired_utc"]:
            return {"refused": f"superseded_by_run_id names {col_succ} but the linked ruling {x['ruling_id']} is EXPIRED ({x['expired_utc']}) — a retired authority grants nothing"}
        if not x["names_run"]:
            return {"refused": f"superseded_by_run_id names {col_succ} but the linked ruling {x['ruling_id']}'s scope does not name this run (conflicting authority)"}
        if x["successor"] != col_succ:
            return {"refused": f"superseded_by_run_id names {col_succ} but the linked ruling {x['ruling_id']} names {x['successor']!r} as the successor (conflicting authority)"}
        return {"successor": col_succ, "ruling_id": x["ruling_id"], "source": "run.superseded_by_run_id + ruling " + x["ruling_id"]}
    found: Dict[str, str] = {}
    for x in (_ruling(*row_) for row_ in (rows or [])):
        if x["action"] == "supersede" and not x["expired_utc"] and x["names_run"] and x["successor"]:
            found.setdefault(x["successor"], x["ruling_id"])
    if len(found) > 1:
        return {"ambiguous": sorted(found), "rulings": found}
    if found:
        succ, rid = next(iter(found.items()))
        return {"successor": succ, "ruling_id": rid, "source": "ruling.target_json.superseded_by_run_id"}
    # no ruling names one — the producer's own duplicate-registration record may (its claim checked against the rows below)
    return producer_supersession(conn, run_id)


PRODUCER_DUPLICATE_REASON = "duplicate_registration"      # file_superseded's `extra.reason` (scripts/sdfbase/full22/registry_close_stores.py)
#: The bytes-and-count condition: two registrations of one store_uri are DUPLICATES only when the bytes and the item count the closes
#: recorded agree — the run row's `store_completion_sha256` and `n_items_generated` (the closer files both on the retired climb before its
#: `superseded`). Read beside RUN_COLUMNS, never through it: a registry whose `run` table lacks a column (a fixture, an older schema)
#: refuses the claim by name instead of failing every row read.
STORE_FACT_COLUMNS = ("store_completion_sha256", "n_items_generated")


def store_facts_row(conn: sqlite3.Connection, run_id: str) -> Tuple[Optional[dict], Optional[str]]:
    """({column: value} of STORE_FACT_COLUMNS for the run row, None) — a recorded 0 is a value, only NULL is unrecorded — or (None, why) when
    the registry's `run` table has no such column: the facts cannot be read, and a claim that depends on them is refused, never assumed."""
    try:
        r = conn.execute(f"SELECT {', '.join(STORE_FACT_COLUMNS)} FROM run WHERE run_id = ?", (run_id,)).fetchone()
    except sqlite3.OperationalError as exc:
        missing = next((c for c in STORE_FACT_COLUMNS if c in str(exc)), None)
        return None, (f"the registry's run table has no `{missing}` column" if missing else f"the registry's run table cannot be read ({exc})")
    if r is None:
        return None, None
    return dict(zip(STORE_FACT_COLUMNS, r)), None


def _fact_equal(col: str, a: object, b: object) -> bool:
    """Exact equality of a recorded store fact — the item count as an integer when both sides parse as one (the schema column is INTEGER;
    a filing may spell it as text), the digest as the exact string the closer filed (registry_close_stores.bon_duplicate_mismatch compares the
    same way)."""
    if col == "n_items_generated":
        try:
            return int(a) == int(b)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return a == b


def producer_supersession(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    """The successor the PRODUCER's own `run_state superseded` record names for a run it
    retired as a DUPLICATE REGISTRATION of one store — run_state_event.extra_json {reason: duplicate_registration, superseded_by_run_id, decision}
    (the full22 closer's record for a BoN climb registration retired by the store512's, and for a multi-turn twin of one store;
    file_superseded in registry_close_stores.py). No ruling exists for these by design (the census effect is the producer's record; the
    `superseded_by_run_id` column is identity, never patched), so the record is a CLAIM and the authority is the standing rule plus the
    registry's own facts, checked here: the named successor is another registration of the SAME store — store_uri recorded on both rows and
    equal, the same eval, the same object, the same model — AND of the same store's BYTES AND COUNT: two registrations are duplicates only
    when `store_completion_sha256` and `n_items_generated` are identical, so both
    are required recorded on both rows (STORE_FACT_COLUMNS; a no-fit store's 0 is recorded) and equal — a later patch that leaves one URI
    with two digests or two counts is not one measurement, and a registry that cannot show the facts refuses by name. A claim the rows do not
    bear out refuses by name; records naming different successors are ambiguous; a record with another reason (or none) names nothing.
    `ruling_id` is None on this path; `source` says what granted it."""
    try:
        evs = conn.execute("SELECT extra_json FROM run_state_event WHERE run_id = ? AND state = 'superseded' ORDER BY event_utc", (run_id,)).fetchall()
    except sqlite3.OperationalError:
        return None
    claims: Dict[str, str] = {}
    for (ej,) in evs:
        try:
            x = json.loads(ej or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            x = {}
        if isinstance(x, dict) and x.get("reason") == PRODUCER_DUPLICATE_REASON and x.get("superseded_by_run_id"):
            claims.setdefault(str(x["superseded_by_run_id"]), str(x.get("decision") or "no decision named"))
    if not claims:
        return None
    if len(claims) > 1:
        return {"ambiguous": sorted(claims), "rulings": {s: f"run_state_event duplicate_registration ({d})" for s, d in claims.items()}}
    succ, decision = next(iter(claims.items()))
    label = f"the producer's duplicate_registration record ({decision})"
    if succ == run_id:
        return {"refused": f"{label} names the run itself as its successor"}
    mine, theirs = registry_row(conn, run_id), registry_row(conn, succ)
    if mine is None or theirs is None:
        return {"refused": f"{label} names {succ} as the other registration of this run's store, but {'this run' if mine is None else succ} is not in the registry — "
                           f"the claim cannot be checked"}
    for col in ("store_uri", "eval_id", "object_id", "model_id"):
        a, b = mine.get(col), theirs.get(col)
        if col == "store_uri":
            a, b = (str(a).rstrip("/") if a else None), (str(b).rstrip("/") if b else None)
        if not a or not b:
            return {"refused": f"{label} names {succ} as the other registration of this run's store, but {col} is unrecorded on {'this run' if not a else succ} — "
                               f"a duplicate registration is verified by the rows, never by the record"}
        if a != b:
            return {"refused": f"{label} names {succ} as the other registration of this run's store, but the rows differ in {col} ({a!r} vs {b!r}) — not one store; "
                               f"the record's claim is not borne out"}
    # the second clause, by the rows: the same URI must carry the same bytes and the same item count on both registrations
    facts: Dict[str, dict] = {}
    for who, rid_ in (("this run", run_id), (succ, succ)):
        f, why = store_facts_row(conn, rid_)
        if why:
            return {"refused": f"{label} names {succ} as the other registration of this run's store, but {why} — the store's bytes and item count "
                               f"(D456: identical store_completion_sha256 and n_items_generated) cannot be verified; the claim is not granted"}
        facts[who] = f or {}
    for col in STORE_FACT_COLUMNS:
        a, b = facts["this run"].get(col), facts[succ].get(col)
        if a is None or b is None:
            return {"refused": f"{label} names {succ} as the other registration of this run's store, but {col} is unrecorded on {'this run' if a is None else succ} — "
                               f"a duplicate registration is verified by the rows, never by the record (D456: identical store_completion_sha256 and n_items_generated)"}
        if not _fact_equal(col, a, b):
            return {"refused": f"{label} names {succ} as the other registration of this run's store, but the rows differ in {col} ({a!r} vs {b!r}) — one store_uri with "
                               f"different bytes or item count is not a duplicate registration (D456); the record's claim is not borne out"}
    return {"successor": succ, "ruling_id": None, "duplicate_registration": True, "decision": decision,
            "source": f"run_state_event superseded, reason duplicate_registration, decision {decision} (one store registered twice; store_uri, eval, object and model "
                      f"equal on both rows, and the recorded store_completion_sha256 and n_items_generated identical — D456)"}


def bound_run_owner(state_root: str, run_id: str) -> Optional[Tuple[dict, dict]]:
    """(unit, binding) of the frozen unit whose REGISTERED binding names `run_id` — the unit whose own admission vouches for the run's contract
    and tree; None when no unit of the inventory binds it (a successor admitted nowhere stands in for nothing)."""
    inv_p = os.path.join(state_root, "gate", "expected_units.json")
    if not os.path.isfile(inv_p):
        return None
    ledger = os.path.join(state_root, "submissions.jsonl")
    for u in (_load(inv_p).get("units") or []):
        b = binding_state(ledger, u["logical_work_id"])
        if b.get("state") == "registered" and run_id in (b.get("run_ids") or []):
            return u, b
    return None


def check_run_coverage(frozen: dict, c: dict, _acc, ev: str) -> List[str]:
    """One bound run's coverage against its eval's FROZEN spec, from the probe document 06 --gate recorded (lib/coverage_probe.py).
    Returns the problems WITHOUT the unit/run prefix. The jailbreak layouts' per-goal comparison is done by the caller per
    (eval, shard); this checks the probe's presence and the layouts whose expectation is per run."""
    cov = frozen["coverage"]; lay = cov["layout"]
    if lay == "unverified":
        return [f"coverage layout for eval {ev!r} is UNVERIFIED: {cov.get('reason')}"]
    if c.get("probe_error") or not isinstance(c.get("probe"), dict):
        return [f"the coverage probe did not run or failed: {c.get('probe_error') or 'no probe document'} — re-run 06 --gate"]
    pr = c["probe"]
    if pr.get("layout") != lay:
        return [f"the coverage probe read layout {pr.get('layout')!r}, the unit freezes {lay!r} — re-freeze / re-gate"]
    probs: List[str] = []
    if lay == "jailbreak_cellstore":
        if not pr.get("exists"):
            return [f"store {pr.get('path')} does not exist"]
        for s in frozen.get("expected_shards", []):
            blk = (pr.get("shards") or {}).get(s)
            if blk is None or not blk.get("present"):
                probs.append(f"shard {s}: no progress.jsonl under {pr.get('path')} ({'; '.join((blk or {}).get('notes') or [])})")
        return probs
    if lay == "bon_store512":
        blk = (pr.get("shards") or {}).get("shard00") or {}
        want = set((cov.get("expected") or {}).get("shard00") or [])
        if not want:
            return ["the freeze carries no expected BoN goal ids (unresolved at freeze time) — re-freeze"]
        if not blk.get("present"):
            return [f"no bon_responses.jsonl under {pr.get('path')}"]
        missing = sorted(want - set(blk.get("completed") or []))
        if missing:
            probs.append(f"{len(missing)} of {len(want)} expected goal(s) lack {cov.get('n_max')} draws (first: {missing[:5]}; errored: {len(blk.get('errored') or [])})")
        return probs
    if lay == "bon_climb_flat":
        for s, blk in (pr.get("shards") or {}).items():
            if blk.get("expected") is None:
                probs.append(f"shard {s}: the climb manifest is missing ({'; '.join(blk.get('notes') or [])}) — the expected set is unknown"); continue
            missing = sorted(set(blk["expected"]) - set(blk.get("completed") or []))
            if missing:
                probs.append(f"shard {s}: {len(missing)} of {len(blk['expected'])} manifest goal(s) not fully climbed (first: {missing[:5]}; errored draws: {len(blk.get('errored') or [])})")
        if not pr.get("shards"):
            probs.append("the probe reported no shard")
        return probs
    if lay == "capability_cellstore":
        it = pr.get("items") or {}
        if it.get("refusal"):
            # a sampled-controls store the producer's own complete-store reader refuses (gen_stats unstamped, a torn spool, an unbound meta
            # row, a tally that disagrees — at_cap_controls.read_control, mirrored by the probe) is refused here by its words
            return [f"the item artifact is not a complete store: {it['refusal']}"]
        if not it.get("present"):
            return [f"no item artifact (rows.jsonl, or the sampled controls' private/gens_<task>.jsonl stamped complete by gen_stats.json) under {pr.get('path')}"]
        n = len(it.get("completed") or []); want = cov.get("n_items")
        if not want:
            return ["the freeze carries no item-set size (unresolved at freeze time) — re-freeze"]
        if n < int(want):
            probs.append(f"{n} unique item(s) of task {cov.get('task')!r} < the item set's {want} ({cov.get('item_set_id')}); tasks seen {it.get('tasks_seen')}")
        if cov.get("items_sha256"):
            if not it.get("items_sha256"):
                probs.append(f"the store attests no item-id list (manifest mmlu_item_set.item_ids) while the item set {cov.get('item_set_id')} records ids sha256 {cov['items_sha256'][:12]}… — the width cannot be attested")
            elif it["items_sha256"] != cov["items_sha256"]:
                probs.append(f"the store's item ids sha256 {it['items_sha256'][:12]}… != the item set {cov.get('item_set_id')}'s {cov['items_sha256'][:12]}… (another item set ran)")
        return probs
    if lay == "misalign_responses":
        blk = (pr.get("shards") or {}).get("shard00") or {}
        want = set((cov.get("expected") or {}).get("shard00") or [])
        if not want:
            return ["the freeze carries no expected scenario ids (unresolved at freeze time) — re-freeze"]
        if not blk.get("present"):
            return [f"no responses.jsonl under {pr.get('path')} (the run's recorded store: the harness cell dir; D3003)"]
        missing = sorted(want - set(blk.get("completed") or []))
        if missing:
            probs.append(f"{len(missing)} of {len(want)} scenario(s) of behavior {cov.get('behavior')!r} lack {cov.get('n_samples')} draws (first: {missing[:5]})")
        return probs
    if lay == "petri_cellstore":
        # the Petri per-cell store on the grading host (stage-E prep pass 2): completed `<behavior>/<scenario>` ids ⊇ the frozen
        # bundle's — a scenario the runner's failure ledger holds unresolved, or one short of a required (epoch, dimension), is named
        blk = (pr.get("shards") or {}).get("shard00") or {}
        want = set((cov.get("expected") or {}).get("shard00") or [])
        if not want:
            return ["the freeze carries no expected Petri scenario ids (the bundle was not enumerated at freeze time) — re-freeze"]
        if not pr.get("exists"):
            return [f"Petri cell store {pr.get('path')} does not exist (the run's recorded store_uri)"]
        if not blk.get("present"):
            return [f"no rows.jsonl under {pr.get('path')}"]
        missing = sorted(want - set(blk.get("completed") or []))
        if missing:
            unresolved = [m for m in missing if m in set(blk.get("errored") or [])]
            probs.append(f"{len(missing)} of {len(want)} expected Petri scenario(s){' of behavior ' + repr(cov.get('behavior')) if cov.get('behavior') else ''} "
                         f"not complete (first: {missing[:5]}; {len(unresolved)} unresolved in the runner's failure ledger; results.json "
                         f"{'present' if pr.get('results_json_present') else 'absent'})")
        return probs
    return [f"unknown coverage layout {lay!r}"]


_CONTRACT_CACHE: Dict[str, Tuple[float, int, Optional[str]]] = {}


def inventory_contract_sha(state_root: str) -> Optional[str]:
    """The sha of the coverage contract the frozen inventory carries NOW under this tree's probe (stage.coverage_contract over its units with
    coverage_probe.DONE_PREDICATES) — what every coverage record must have been collected under; None without an inventory. Cached
    per (path, mtime, size): check() reads it once per unit."""
    inv_p = os.path.join(state_root, "gate", "expected_units.json")
    try:
        st = os.stat(inv_p)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    hit = _CONTRACT_CACHE.get(inv_p)
    if hit and hit[:2] == key:
        return hit[2]
    try:
        sha = coverage_contract(_load(inv_p).get("units") or [], CP.DONE_PREDICATES)["sha256"]
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        sha = None
    _CONTRACT_CACHE[inv_p] = (key[0], key[1], sha)
    return sha


def inventory_family_units(state_root: str, family: str) -> List[dict]:
    """The frozen inventory's units of one family (the family's CURRENT bindings are read over every unit the inventory names in it —
    the same units 06 --gate's verdict step unions). [] without a readable inventory (the verdict check then compares against no bindings
    and refuses a verdict that names runs: nothing is bound where nothing is frozen)."""
    try:
        inv = _load(os.path.join(state_root, "gate", "expected_units.json"))
    except (OSError, json.JSONDecodeError):
        return []
    return [u for u in (inv.get("units") or []) if isinstance(u, dict) and u.get("family") == family]


def verdict_run_binding_problems(unit: dict, state_root: str, v: dict, family_units: Optional[List[dict]] = None) -> List[str]:
    """A pass verdict is bound to the RUNS it ruled, not to a coverage generation. The standing verdict's
    `run_ids` (cmd_verdict records the family's registered bindings at ruling time — stage.py family_bound_run_ids) must equal AS A SET the
    union of the run ids the family's units are bound to NOW (the same helper over the inventory's units of the family); otherwise the family
    is refused by name with the run ids ADDED (bound now, never ruled) and REMOVED (ruled, no longer bound) and the remedy. A verdict with no
    or empty `run_ids` names no run it ruled over and is refused by name ("re-rule this family"). NO attempt-generation marker, NO tie to the
    coverage record's probe sha or collected time: a coverage refresh, a new probe sha or a new coverage generation over the SAME runs is a
    re-read and still admits (the runs are what was measured; the reader may improve under a frozen wave). One case is NOT closed here: the
    same runs re-ruled with an out-of-range headline whose verdict write failed, the old pass standing — 06 --gate exits non-zero on that
    step by name, which is the signal."""
    lw = unit["logical_work_id"]; fam = unit["family"]
    remedy = (f"re-gate the family: 06_eval_waves.sh --gate {fam} --verdict (or --headline <value> --range <lo,hi> by hand) "
              f"(records the verdict over the runs bound now)")
    ruled_raw = v.get("run_ids")
    ruled = [str(x) for x in ruled_raw if x] if isinstance(ruled_raw, list) else []
    if not ruled:
        return [f"{lw}: family {fam} verdict {v.get('verdict')!r} (ruled {v.get('ruled_utc')}) records no run_ids ({ruled_raw!r}) — it names no run it ruled over and "
                f"vouches for none (D3008: a verdict is bound to the runs it ruled); re-rule this family — {remedy}"]
    units = family_units if family_units is not None else inventory_family_units(state_root, fam)
    bound = family_bound_run_ids(os.path.join(state_root, "submissions.jsonl"), units, fam)
    added = sorted(set(bound) - set(ruled)); removed = sorted(set(ruled) - set(bound))
    if not added and not removed:
        return []
    return [f"{lw}: family {fam} verdict {v.get('verdict')!r} (ruled {v.get('ruled_utc')}) ruled over runs that differ from the family's current registered bindings: "
            f"{len(added)} bound now that it never ruled {added}; {len(removed)} it ruled that are no longer bound {removed} — a resubmission re-bound a unit of the "
            f"family since the ruling and no verdict was recorded over the new ids (D3008: a pass vouches for the runs it ruled, never for runs nobody ruled); {remedy}"]


def _leg_words(leg: Optional[tuple]) -> str:
    """A leg identity (gate_headline.leg_identity) in the words a refusal gives: the judgment it was read under and the value."""
    if leg is None:
        return "no leg"
    return f"judgment {leg[4] or '(rule-scored rows)'} value {leg[5]:g}" if isinstance(leg[5], (int, float)) else f"judgment {leg[4]} value {leg[5]!r}"


def verdict_grade_problems(unit: dict, state_root: str, conn: sqlite3.Connection, v: dict, family_units: Optional[List[dict]] = None,
                           reads: Optional[Dict[str, object]] = None) -> List[str]:
    """A pass ruled on the registry's grades (`06 --gate --verdict`; cmd_verdict records every leg it read under `headline_legs`) vouches for
    those grades, not only for its runs: the family's unsteered legs are re-read NOW with the reader that ruled (lib/gate_headline.py
    read_legs) and must read exactly as recorded (gate_headline.leg_identity: the run, eval, metric, threshold, judgment and value). A leg
    that no longer reads (its judgment superseded or abandoned: a re-grade begun, the replacement not complete) refuses the family by name,
    and so does a changed grade (another judgment current, a value refiled, a leg added or dropped by a hold-out applied since), until the
    family is ruled again. Before, admission checked the verdict and its run ids alone, so a pass outlived the grade it was ruled on, and the
    re-gate's skip (it writes nothing) left it standing. A verdict recorded by hand carries no legs: the operator's ruling, bound to its runs
    alone. `reads` caches one re-read per family across the family's units (check() passes one dict per call)."""
    if v.get("verdict") != "pass" or "headline_legs" not in v:
        return []
    lw = unit["logical_work_id"]; fam = unit["family"]
    head = f"{lw}: family {fam} verdict 'pass' (ruled {v.get('ruled_utc')})"
    remedy = f"re-run 06_eval_waves.sh --gate {fam} --verdict once the legs are graded (it rules on the grades the registry holds then)"
    # imported here, not at the top: gate_headline imports this module (supersession)
    import gate_headline as GH

    got = reads.get(fam) if reads is not None else None
    if got is None:
        try:
            why = GH.schema_problem(conn)
            got = {"error": why} if why else GH.read_legs(conn, os.path.join(state_root, "submissions.jsonl"),
                                                           family_units if family_units is not None else inventory_family_units(state_root, fam), fam)
        except sqlite3.Error as exc:
            got = {"error": f"the registry could not be read ({exc})"}
        if reads is not None:
            reads[fam] = got
    if got.get("error"):
        return [f"{head} was ruled on the registry's grades, which cannot be read back here: {got['error']} — {remedy}"]
    if got["problems"]:
        return [f"{head} was ruled on grades the registry no longer reads whole: {'; '.join(got['problems'])} — {remedy}"]
    ruled_raw = v.get("headline_legs")
    ruled = {GH.leg_identity(x)[0]: GH.leg_identity(x) for x in ruled_raw if isinstance(x, dict)} if isinstance(ruled_raw, list) else {}
    now = {GH.leg_identity(x)[0]: GH.leg_identity(x) for x in got["legs"]}
    changed = [f"run {rid}: ruled {_leg_words(ruled.get(rid))}, reads {_leg_words(now.get(rid))} now"
               for rid in sorted(set(ruled) | set(now), key=str) if ruled.get(rid) != now.get(rid)]
    if not ruled or changed:
        return [f"{head}: the registry's grades changed since the ruling: {'; '.join(changed) or 'the verdict records no leg'} — {remedy}"]
    return []


def check_unit(unit: dict, state_root: str, conn: Optional[sqlite3.Connection], family_units: Optional[List[dict]] = None,
               verdict_reads: Optional[Dict[str, object]] = None) -> List[str]:
    lw = unit["logical_work_id"]
    probs: List[str] = []
    b = binding_state(os.path.join(state_root, "submissions.jsonl"), lw)
    if b["state"] == "unbound":
        return [f"{lw}: never submitted (unbound in the binding ledger)"]
    if b["state"] == "submitted":
        return [f"{lw}: submitted (job {b.get('job_id')} on {b.get('cluster')}) but not registered — no run id bound yet"]
    run_ids = list(dict.fromkeys(b.get("run_ids") or []))      # a climb run bound once per array task (one run per cell) reads as ONE run here
    if not run_ids:
        return [f"{lw}: registered binding names no run ids"]
    # the unit's recorded sha (its provenance's latest entry, else the freeze sha), and every recorded advance must carry the
    # FROZEN contract's hash — a re-run whose contract differed was refused at submit; a recorded hash that differs is a refusal here too
    ledger = os.path.join(state_root, "submissions.jsonl")
    # (Medium) the APPEND-ONLY attempts ledger is the truth for a unit's sha; the sidecar must agree with it entry for
    # entry (a repaired or copied sidecar with a foreign last sha admits nothing)
    disagreements = [pr for pr in reconcile_provenance(ledger) if pr.startswith(lw + ":")]
    if disagreements:
        return probs + disagreements
    prov = ledger_sha_provenance(ledger)
    want_sha = unit_recorded_sha(prov, unit)
    # the contract admission identifies the unit by is the FROZEN one or its latest RECEIPTED amendment (a seed row's stated
    # defaults made explicit changed the instrument ids; the amendment tool checked the row diff and that no run completed before it)
    amend = read_amendments(ledger)
    eff = effective_contract(amend, unit)
    frozen_hash = contract_dict_sha256(eff)
    known_hashes = {frozen_hash} | {a.get("old_contract_sha256") for a in amend.get(lw, [])} | {a.get("new_contract_sha256") for a in amend.get(lw, [])}
    for e in prov.get(lw, []):
        if e.get("contract_sha256") and e["contract_sha256"] not in known_hashes:
            probs.append(f"{lw}: provenance entry at sha {str(e.get('sha'))[:12]} ({e.get('kind')}, {e.get('utc')}) records contract {e['contract_sha256'][:12]}… != the "
                         f"effective contract {frozen_hash[:12]}… (nor any receipted amendment of it) — the unit's eval contract changed across trees; that is a new wave, not this baseline")
    if conn is None:
        return probs + [f"{lw}: no registry to check completeness against (fail closed)"]
    evals: Dict[str, dict] = unit["evals"]
    # an eval the model's ruling holds out stays in the contract but is not REQUIRED — no bound run, no completeness, no
    # coverage is asked of it; its runs, when the job produced some, are neither judged nor counted here (check() says `excluded_by_ruling`)
    held: Dict[str, dict] = unit.get("held_out") or {}
    # 30-fix: a stamp lifts a requirement only with its append-only receipt (the `hold_out` event the freeze or the
    # live apply wrote, naming this overrides document and this unit under this eval) — a stamp without one admits nothing
    if held:
        receipts = hold_out_receipts(os.path.join(state_root, "attempts.jsonl"))
        for e, h in sorted(held.items()):
            if not any(hold_out_receipt_matches(r, h, e, lw) for r in receipts):
                probs.append(f"{lw}: eval {e} is stamped held_out ({h.get('source')}) but the attempts ledger carries no matching hold_out receipt "
                             f"(override {h.get('override_id')} sha {str(h.get('overrides_sha256'))[:12]}…) — a stamp without its append-only receipt admits nothing; "
                             f"`stage.py hold-out --out <inventory> --overrides <file>` repairs it")
        if probs:
            return probs
    seen_evals: Dict[str, List[str]] = {e: [] for e in evals}
    rows: Dict[str, dict] = {}
    successors: Dict[str, dict] = {}          # rid -> {run_id, ruling_id, store_uri, owner}: the ruled successor whose work stands in for the run's
    cells: Dict[str, Optional[str]] = {str(r.get("run_id")): r.get("object") for r in (b.get("runs") or [])}   # the binding's cell per run (route.object)
    tasks_by_rid = bound_array_tasks(state_root, lw)       # the array tasks the binding names per run (the shared climb shape binds one run to four)
    for rid in run_ids:
        row = registry_row(conn, rid)
        if row is None:
            probs.append(f"{lw}: bound run {rid} is not in the registry"); continue
        rows[rid] = row
        if row.get("eval_id") in held:
            continue
        ev = row.get("eval_id")
        # admission-fix-2: the OBJECT is part of the identity on every path — a run of no stated object is not this unit's,
        # and where the freeze pins the cell's object the row must carry that object
        pins = unit.get("objects") or {}
        if pins:
            # admission-fix-3: a unit WITH object pins admits a run only through a RESOLVABLE pinned cell — a binding without
            # the run's cell (route.object; stage.py bind --run-ids without --runs-json) or a cell the pins do not name cannot be matched to its
            # pin and used to bypass the digest comparison silently; both refuse by name (06 --gate re-binds from the pending file with the cells)
            cell = cells.get(rid)
            if not cell:
                probs.append(f"{lw}: run {rid} ({ev}) has no cell recorded in its binding (route.object missing) while the unit pins objects for {sorted(pins)} — "
                             f"a run of no stated cell cannot be matched to its pin; re-bind from the pending file (06 --gate)"); continue
            if cell not in pins:
                probs.append(f"{lw}: run {rid} ({ev}) is bound to cell {cell!r}, which the unit's object pins do not name ({sorted(pins)}) — a foreign cell"); continue
        pin = frozen_object_id(unit, cells.get(rid))
        if not row.get("object_id"):
            probs.append(f"{lw}: run {rid} ({ev}) records NO object_id — the object identity is unrecorded; a run of no stated object is not this unit's"); continue
        if pin is not None and row["object_id"] != pin:
            probs.append(f"{lw}: run {rid} ({ev}) is of object {row['object_id']!r} but the unit's frozen pin for cell {cells.get(rid)!r} is {pin!r} — not this unit's object"); continue
        # a run a `supersede` RULING retired stands in for nothing itself — its SUCCESSOR does, when the ruling names
        # one, the successor is complete, of the same eval, and bound to a unit of this inventory (whose own admission vouches for its tree);
        # the identity checks below read the successor's row, and the coverage half requires the probe at the successor's store. A superseded
        # run with no named successor, or an incomplete / absent / foreign / unbound one, refuses BY NAME — never a force, never a re-freeze
        evidence = row; sha_want = want_sha
        if row["lifecycle_state"] == "superseded":
            sup = supersession(conn, rid)
            if sup is None:
                probs.append(f"{lw}: run {rid} ({ev}) is superseded and no `supersede` ruling names its successor (neither run.superseded_by_run_id nor a "
                             f"ruling's target_json.superseded_by_run_id, nor a producer's duplicate_registration run_state record) — nothing stands in for its work"); continue
            if sup.get("refused"):
                probs.append(f"{lw}: run {rid} ({ev}) is superseded but {sup['refused']} — nothing stands in for its work"); continue
            if sup.get("ambiguous"):
                probs.append(f"{lw}: run {rid} ({ev}) is superseded by rulings naming different successors {sup['ambiguous']} — ambiguous, nothing admitted"); continue
            auth = sup.get("ruling_id") or sup.get("source")            # the producer path has no ruling id; its source says what granted it
            succ = registry_row(conn, sup["successor"])
            if succ is None:
                probs.append(f"{lw}: run {rid} ({ev}) is superseded by {sup['successor']} ({auth}), which is not in the registry"); continue
            if succ["lifecycle_state"] not in COMPLETE_STATES:
                probs.append(f"{lw}: run {rid} ({ev}) is superseded by {sup['successor']} ({auth}), which is {succ['lifecycle_state']}, not complete (judged|ingested)")
            if succ.get("eval_id") != ev:
                probs.append(f"{lw}: run {rid} ({ev}) is superseded by {sup['successor']}, which is eval {succ.get('eval_id')!r} — a successor of another eval stands in for nothing"); continue
            # admission-fix-2: the successor must be of the SAME OBJECT — both ids recorded and equal (null on either side is
            # a refusal, never a match), and equal to the unit's frozen pin for the cell when the freeze carries one; a complete, inventory-bound
            # run of ANOTHER steered object under the same eval would otherwise certify this object's unit with that object's goals
            if not succ.get("object_id"):
                probs.append(f"{lw}: run {rid} ({ev}) is superseded by {sup['successor']}, whose object identity is unrecorded (object_id null) — a successor stands in for nothing without a matching object"); continue
            if succ["object_id"] != row["object_id"]:
                probs.append(f"{lw}: run {rid} ({ev}) is of object {row['object_id']!r} but its successor {sup['successor']} is of object {succ['object_id']!r} — a successor of ANOTHER object stands in for nothing"); continue
            if pin is not None and succ["object_id"] != pin:
                probs.append(f"{lw}: run {rid} ({ev}) successor {sup['successor']} is of object {succ['object_id']!r} != the unit's frozen pin {pin!r} for cell {cells.get(rid)!r}"); continue
            owner = bound_run_owner(state_root, sup["successor"])
            if owner is None:
                probs.append(f"{lw}: run {rid} ({ev}) is superseded by {sup['successor']} ({auth}), which no unit of this inventory binds — a successor admitted "
                             f"nowhere stands in for nothing"); continue
            successors[rid] = {"run_id": sup["successor"], "ruling_id": sup.get("ruling_id"), "source": sup.get("source"), "store_uri": succ.get("store_uri"),
                               "owner": owner[0]["logical_work_id"]}
            evidence = succ; sha_want = unit_recorded_sha(prov, owner[0])      # the successor's tree is its OWN unit's recorded sha, vouched there
        elif row["lifecycle_state"] not in COMPLETE_STATES:
            probs.append(f"{lw}: run {rid} is {row['lifecycle_state']}, not complete (judged|ingested)")
        who = rid if evidence is row else f"{rid} via successor {evidence['run_id']}"
        if ev not in evals:
            probs.append(f"{lw}: run {rid} is eval {ev!r}, not one of the unit's evals {sorted(evals)} (a run this unit never planned)"); continue
        seen_evals[ev].append(rid)
        frozen = dict(evals[ev]); frozen.update({k: v for k, v in (eff["evals"].get(ev) or {}).items() if k in ("instrument_id", "decode_policy_ref")})
        # a unit's runs must carry THE UNIT's recorded sha — the sha of its first accepted attempt, advanced only through the
        # funnel's re-run path (per-unit provenance) — never one wave-wide sha; the tree is provenance, the contract is the identity
        # (the instrument id and decode-policy ref compared are the EFFECTIVE contract's)
        for col, want in (("model_id", unit.get("model_id")), ("code_sha", sha_want), ("instrument_id", frozen.get("instrument_id"))):
            if want and evidence.get(col) and evidence[col] != want:
                probs.append(f"{lw}: run {who} ({ev}) {col}={evidence[col]!r} != {'the unit' + chr(39) + 's recorded sha' if col == 'code_sha' else 'frozen'} {want!r} "
                             f"(a baseline under other bindings is not this baseline)")
        if frozen.get("decode_policy_ref"):
            # with a frozen decode identity, a NULL policy id or a non-`exact` derivation
            # (decode_match.stamp_run_rows sets decode_policy_id only for an exact match; the schema permits null) is refused
            if not evidence.get("decode_policy_id"):
                probs.append(f"{lw}: run {who} ({ev}) records NO decode_policy_id (decode_policy_match={evidence.get('decode_policy_match')!r}) while the unit freezes decode policy {frozen['decode_policy_ref']!r} — an unrecorded decode is not this baseline")
            else:
                if evidence.get("decode_policy_match") != "exact":
                    probs.append(f"{lw}: run {who} ({ev}) decode_policy_match={evidence.get('decode_policy_match')!r}, not 'exact' (frozen {frozen['decode_policy_ref']!r})")
                ref = policy_ref(conn, evidence["decode_policy_id"])
                if ref is None:
                    probs.append(f"{lw}: run {who} ({ev}) decode_policy_id={evidence['decode_policy_id']!r} resolves to no decode_policy row (cannot compare with frozen {frozen['decode_policy_ref']!r})")
                elif ref != frozen["decode_policy_ref"]:
                    probs.append(f"{lw}: run {who} ({ev}) decode policy {ref!r} != frozen {frozen['decode_policy_ref']!r}")
    for ev, rids in seen_evals.items():
        if not rids and ev not in held:
            probs.append(f"{lw}: expected eval {ev!r} has no bound run (the unit registered {sorted({r.get('eval_id') for r in rows.values()})})")
    cov_p = coverage_path(state_root, lw)
    if not os.path.isfile(cov_p):
        probs.append(f"{lw}: no coverage record ({cov_p}) — 06_eval_waves.sh --gate collects it")
    else:
        cov = _load(cov_p)
        # the record is TIED to the coverage contract the inventory carries NOW (the unit set, every identity, this tree's done
        # predicates) exactly as it is tied to the binding's run ids — a record collected under another contract (a rerun stamped since, an
        # edited inventory, a changed predicate) or before the contract existed proves nothing until 06 --gate reads the wave again
        want_cc = inventory_contract_sha(state_root)
        got_cc = (cov.get("coverage_contract") or {}).get("sha256") if isinstance(cov.get("coverage_contract"), dict) else None
        if want_cc and got_cc != want_cc:
            probs.append(f"{lw}: the coverage record was collected under coverage contract {got_cc[:12] + '…' if got_cc else 'none (before the contract existed)'}, the "
                         f"inventory's is now {want_cc[:12]}… (a rerun was stamped since, the inventory changed, or the probe's done predicates did) — re-run 06 --gate")
        elif set(cov.get("run_ids") or []) != set(run_ids):
            probs.append(f"{lw}: the coverage record is for runs {sorted(cov.get('run_ids') or [])}, the binding names {sorted(run_ids)} — re-run 06 --gate")
        else:
            by_eval_shard: Dict[Tuple[str, str], set] = {}       # (eval, shard) -> completed ids over EVERY bound run of the eval (a climb's cells, a shard-per-run layout)
            for rid in run_ids:
                c = (cov.get("runs") or {}).get(rid) or {}
                row = rows.get(rid)
                ev = (row or {}).get("eval_id") or c.get("eval_id")
                frozen = evals.get(ev)
                if frozen is None or ev in held:
                    continue                                   # already reported above / held out by ruling
                if rid in successors:
                    # the record must have been probed at the SUCCESSOR's recorded store (06 --gate resolves the same
                    # ruling and writes `successor` on the run) — a record probed anywhere else for a superseded run proves nothing
                    s = successors[rid]; got = c.get("successor") or {}
                    if got.get("run_id") != s["run_id"] or c.get("store_uri") != s["store_uri"] or not s["store_uri"]:
                        # for a producer-retired duplicate registration the successor's store IS this run's (verified above), and the probe read
                        # the unit's own frozen artifact path; the record must still name the successor the registry names, with that store
                        where = "was not probed at the successor's recorded store" if s.get("ruling_id") else "does not name the successor the registry names"
                        probs.append(f"{lw}: run {rid} ({ev}) is superseded by {s['run_id']} but the coverage record {where} "
                                     f"(record: successor {got.get('run_id')!r}, store {c.get('store_uri')!r}; registry: store {s['store_uri']!r}) — re-run 06 --gate")
                        continue
                probs.extend(f"{lw}: run {rid} ({ev}) {p}" for p in check_run_coverage(frozen, c, by_eval_shard.setdefault, ev))
            # the jailbreak layouts are judged per (eval, shard) across the unit's runs: a goal is covered when SOME complete run of the
            # eval completed it (one run per cell, every cell's store holds the whole shard) — every frozen goal id of every shard
            for ev, frozen in evals.items():
                if ev in held:
                    continue
                if frozen["coverage"]["layout"] == "jailbreak_cellstore" and seen_evals.get(ev):
                    for s, want in (frozen["coverage"].get("expected") or {}).items():
                        for rid in seen_evals[ev]:
                            have = (((cov.get("runs") or {}).get(rid) or {}).get("probe") or {}).get("shards", {}).get(s)
                            if have is None:
                                continue                       # reported per run above
                            missing = sorted(set(want) - set(have.get("completed") or []))
                            if missing:
                                probs.append(f"{lw}: run {rid} ({ev}) shard {s}: {len(missing)} of {len(want)} expected goal(s) not completed (first: {missing[:5]}; errored: {len(have.get('errored') or [])})")
                if frozen["coverage"]["layout"] == "bon_climb_flat" and seen_evals.get(ev):
                    # EVERY expected climb shard must have been READ — the per-run check above judges the shards a document reports, never the
                    # ones it does not. The requirement is PER RUN and PER CELL, never one union over the eval — with two bound cells a whole
                    # four-shard document for one cell covered the other cell's unread shards, and a cell whose document read its last task's
                    # shard alone certified on its neighbour. Per run: the document reads the shard of every array task the BINDING names for
                    # it (bound_array_tasks — the shared shape binds one run id to the cell's four tasks; the per-task shape, one task each).
                    # Per cell (the binding's route.object): the union over the cell's runs reads every expected shard (a per-task cell whose
                    # task never bound is named here). A shard read for another run or another cell stands in for nothing.
                    want_sh = set(frozen.get("expected_shards") or [])
                    if not want_sh:
                        probs.append(f"{lw}: eval {ev} freezes no expected climb shards — re-freeze"); continue
                    by_cell: Dict[Optional[str], set] = {}
                    for rid in seen_evals[ev]:
                        c = (cov.get("runs") or {}).get(rid) or {}; pr = c.get("probe") if isinstance(c.get("probe"), dict) else {}
                        sh = set(pr.get("shards") or {})
                        by_cell.setdefault(cells.get(rid), set()).update(sh)
                        # the run's tasks: the binding's per-task rows (06 binds one per task's pending file), else what the coverage record kept of them
                        tasks = tasks_by_rid.get(rid) or sorted({int(t) for t in (c.get("array_tasks") or []) if t is not None}) or \
                            ([int(c["array_task"])] if c.get("array_task") is not None else [])
                        if not tasks:
                            probs.append(f"{lw}: run {rid} ({ev}) carries no array task identity (neither the binding's per-task rows nor the coverage record name one) — "
                                         f"the climb shards it must cover cannot be known; re-bind from the pending files (06 --gate)"); continue
                        want_run = {f"shard{t:02d}" for t in tasks}
                        foreign = sorted(want_run - want_sh)
                        if foreign:
                            probs.append(f"{lw}: run {rid} ({ev}) is bound to array task(s) {tasks} naming shard(s) {foreign} the eval does not freeze (expected: {sorted(want_sh)}) — "
                                         f"a foreign task"); continue
                        missing_run = sorted(want_run - sh)
                        if missing_run:
                            cause = ""
                            if len(tasks) > 1 and "shard_indices" not in pr:
                                cause = (f" — the run is bound to array tasks {tasks} (one registration for the cell's climb tasks since D2599's companion key) yet its probe "
                                         f"document names no `shard_indices`: the coverage_probe.py at the inventory's frozen sha reads ONE --shard-index per call (it predates "
                                         f"the shared-run probe of 2026-10-11), so this record holds one call's document for the run — a 06 --gate before the per-task fold "
                                         f"(2026-10-12), or a record written by hand; re-run 06 --gate, which probes the run once per bound task from the mirrored checkout "
                                         f"at that sha and folds the documents (stage.py coverage --probes-json)")
                            probs.append(f"{lw}: run {rid} ({ev}) is bound to array task(s) {tasks} but its probe document reads shard(s) {sorted(sh)} — its own shard(s) "
                                         f"{missing_run} were never read (a shard read for another run of the eval stands in for nothing){cause}")
                    for cell, read_sh in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
                        missing_sh = sorted(want_sh - read_sh)
                        if missing_sh:
                            probs.append(f"{lw}: eval {ev}, cell {cell!r}: expected climb shard(s) {missing_sh} were read for no bound run of the cell (read: {sorted(read_sh)}; "
                                         f"expected: {sorted(want_sh)}) — every shard of the cell must be read")
    vp = os.path.join(state_root, "gate", f"{unit['family']}.json")
    if not os.path.isfile(vp):
        probs.append(f"{lw}: no sanity verdict for family {unit['family']} ({vp})")
    else:
        v = _load(vp)
        if v.get("verdict") != "pass":
            probs.append(f"{lw}: family {unit['family']} verdict is {v.get('verdict')!r}, not pass")
        # the verdict is bound to the RUNS it ruled — its run_ids must be the family's registered bindings NOW (the `pass` check above stays;
        # this one is independent of it: a `pass` over other runs, or over none, admits nothing)
        run_probs = verdict_run_binding_problems(unit, state_root, v, family_units)
        probs.extend(run_probs)
        # and a pass ruled on the registry's grades to those GRADES: its legs must read the same now (asked only over the same runs, so a
        # resubmission is named once, by the run check above)
        if not run_probs:
            probs.extend(verdict_grade_problems(unit, state_root, conn, v, family_units, verdict_reads))
    return probs


def registry_hold_out_rulings(conn: Optional[sqlite3.Connection], model_key: str) -> Dict[str, str]:
    """{eval_id: ruling_id} — the registry's standing hold_out / exclude / drop_eval rulings whose match names the eval and either this
    model key or no model at all. {} without a registry, or on a registry that predates the ruling table."""
    out: Dict[str, str] = {}
    if conn is None:
        return out
    try:
        rows = conn.execute("SELECT ruling_id, scope_json FROM ruling WHERE action IN ('hold_out', 'exclude', 'drop_eval') AND expired_utc IS NULL ORDER BY ruled_utc").fetchall()
    except sqlite3.OperationalError:
        return out
    for rid, sj in rows:
        try:
            m = (json.loads(sj) or {}).get("match") or {}
        except (json.JSONDecodeError, AttributeError):
            continue
        models = m.get("model_ids") or []
        if models and model_key not in models:
            continue
        for e in m.get("eval_ids") or []:
            out.setdefault(str(e), str(rid))
    return out


def hold_out_notes(unit: dict, rulings: Dict[str, str]) -> List[str]:
    """One line per held-out eval of a unit: `excluded_by_ruling (<source>)` plus whether the registry carries a matching ruling (a NOTE,
    never a gate: the registry's ruling record is the registry owner's ingest, and the inventory's stamp is the ruling's local provenance)."""
    lw = unit["logical_work_id"]; out = []
    for e, h in sorted((unit.get("held_out") or {}).items()):
        reg = f"registry ruling {rulings[e]}" if e in rulings else "registry: no matching ruling yet (the ruling record is ingested by the registry's owner)"
        out.append(f"{lw}: eval {e} excluded_by_ruling ({h.get('source')}; {reg})")
    return out


def check(state_root: str, registry: Optional[str], forced: Dict[str, str], families: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    inv_p = os.path.join(state_root, "gate", "expected_units.json")
    if not os.path.isfile(inv_p):
        return [f"no frozen baseline inventory at {inv_p} — run 05_stage_roster.sh first"], []
    inv = _load(inv_p)
    if inv.get("schema") != INVENTORY_SCHEMA:
        return [f"frozen inventory {inv_p} has schema {inv.get('schema')!r}, this gate reads schema {INVENTORY_SCHEMA} — re-freeze (05_stage_roster.sh --step freeze)"], []
    # demo-import M7 (Q10, tightening 2): a forced LWID that names no unit of the FROZEN inventory is refused BY NAME — before this
    # `if lw in forced` below ignored a typo silently and the operator read an unrelated 'never submitted' for the unit they meant
    known = {u["logical_work_id"] for u in inv["units"]}
    unknown = [lw for lw in forced if lw not in known]
    if unknown:
        return [f"--force-gate names no unit of the frozen inventory: {lw!r} (the inventory's units: {', '.join(sorted(known))})" for lw in unknown], []
    conn = None
    if registry and registry != "none":
        if not os.path.isfile(registry):
            return [f"registry database {registry} does not exist (fail closed)"], []
        conn = sqlite3.connect(f"file:{registry}?mode=ro", uri=True)
    problems: List[str] = []; ok: List[str] = []
    verdict_reads: Dict[str, object] = {}       # one re-read of a family's graded legs per call, shared by its units
    try:
        rulings = registry_hold_out_rulings(conn, str(inv.get("model") or "")) if any(u.get("held_out") for u in inv["units"]) else {}
        for unit in inv["units"]:
            if families and unit["family"] not in families:
                continue
            lw = unit["logical_work_id"]
            if lw in forced:
                ok.append(f"{lw}: FORCED ({forced[lw]})"); continue
            p = check_unit(unit, state_root, conn, [u for u in inv["units"] if u.get("family") == unit["family"]],   # The family's units, for the verdict's run binding
                           verdict_reads)
            if p:
                problems.extend(p)
            else:
                ok.append(lw)
                ok.extend(hold_out_notes(unit, rulings))
    finally:
        if conn is not None:
            conn.close()
    return problems, ok


def record_forced(state_root: str, forced: Dict[str, str]) -> None:
    if not forced:
        return
    p = os.path.join(state_root, "gate", "forced.jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        for lw, why in forced.items():
            fh.write(json.dumps({"logical_work_id": lw, "reason": why, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--state-root", required=True); p.add_argument("--registry", required=True, help="registry.sqlite path, or `none`")
    p.add_argument("--force-gate", action="append", default=[], metavar="LWID=REASON"); p.add_argument("--families", default="")
    p.add_argument("--json", action="store_true"); p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    forced = {}
    for f in a.force_gate:
        if "=" not in f:
            print(f"REFUSED: --force-gate wants LWID=REASON, got {f!r}", file=sys.stderr); return 2
        k, v = f.split("=", 1)
        # demo-import M7 (Q10, tightening 1): an EMPTY reason is refused — a forced admission is recorded with why (the hold-out
        # precedent refuses a provenance-less override the same way); `LWID=` used to be accepted and recorded with reason ''
        if not k.strip() or not v.strip():
            print(f"REFUSED: --force-gate wants LWID=REASON with a non-empty reason, got {f!r} — a forced admission is recorded with why", file=sys.stderr); return 2
        forced[k] = v
    problems, ok = check(a.state_root, a.registry, forced, a.families.split(",") if a.families else None)
    if a.json:
        print(json.dumps({"admitted": not problems, "ok": ok, "problems": problems}, indent=2))
    else:
        for o in ok:
            print(f"ok   {o}")
        for pr in problems:
            print(f"REFUSED: {pr}", file=sys.stderr)
    if problems:
        print(f"ADMISSION REFUSED: {len(problems)} problem(s); {len(ok)} unit(s) ok — the unsteered battery is the gate (D15)", file=sys.stderr)
        return 3
    if not a.dry_run:
        record_forced(a.state_root, forced)
    print(f"ADMITTED: every unit of the frozen inventory is registered, complete under its frozen bindings, covered and verdicted ({len(ok)} units)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
