"""unscoreable_items.json — WHY a served control store has no aggregate row and no gen_stats.json.

The served capability driver (scripts/served_capability_run.py) refuses a partial aggregate: when
any requested item cannot be scored it exits before the ``__aggregate__`` row is filed and before
gen_stats.json is written. Without a record in the store, a reader of that store meets three files
(manifest.json, rows.jsonl, evidence.json) and an ABSENCE it can only explain by inference — an
MMLU store with 199 of 200 items scored because one answer letter fell below the server's top-K
logprobs looks exactly like a store whose job was killed.

This module is the record and its reader:

* :func:`write_record` — ``unscoreable_items.json`` written INTO the cell's store dir (the same
  directory a reader of that store lists), naming the items, the missing letter token ids, the
  top_logprobs cap in force, the scored-vs-requested counts, the exact bounds the unscored items
  leave on the requested-set accuracy, and a one-line statement that the aggregate and
  gen_stats.json are absent BECAUSE of this. ``written_by`` says who wrote it: ``"driver"`` on the
  refusal path itself, or ``"repair_pass"`` for a record written after the fact from evidence
  outside the store; the two are told apart by that field, never by guesswork. No driver on this
  tree writes the record (``scripts/served_capability_run.py`` has no write of it), and the cell
  record ar_cell.sbatch files declares that, so :data:`STATUS_REFUSED` is unreachable for the
  stores this tree produces and an unscoreable item leaves :data:`STATUS_PARTIAL_NO_RECORD`.
* :func:`mmlu_store_status` — one status per store that tells "complete" from "refused with N of
  M scored" from "undeterminable items" (the driver's own ``topk_undeterminable`` rows: an
  ATTEMPTED run whose aggregate the driver refused because every answer letter of some item fell
  outside the top-K) from "partial, no record" (a run cut short; the store does not record why)
  from "never ran" (no MMLU evidence of any kind), so no reader infers from a missing file.
* :func:`mmlu_stores_facts` — the per-store fragment cell_facts.json carries (ar_cell.sbatch), so
  the battery's own identity record states the completeness of every MMLU store beside the numbers,
  per MMLU leg: the 200-item leg under ``<controls_root>/mmlu/cells/`` and the whole-pool leg under
  ``<controls_root>/mmlu_full/cells/`` are two store roots, reported separately.

Stdlib only, on purpose: the reader must be importable anywhere a store is read, including a bare
``python3`` with no venv.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

RECORD_NAME = "unscoreable_items.json"
SCHEMA = "served_capability_run/unscoreable_items/v1"

WRITTEN_BY_DRIVER = "driver"
WRITTEN_BY_REPAIR_PASS = "repair_pass"

#: sisteer.capeval.tiers.parity.PARITY_TASK_MMLU / the aggregate pseudo-item, restated here so
#: this module stays import-free of the torch-bearing tiers; tests/test_unscoreable_record.py
#: pins both against their owners.
MMLU_TASK = "s2parity/mmlu_letter"
AGGREGATE_ITEM_ID = "__aggregate__"

#: per-item failure kinds. Only the first is "unscoreable" in the strict sense — the letter's
#: logprob exists at the scored position but the server's top-K did not carry it; the other two
#: are a request that returned nothing usable and rerun on resume.
KIND_LETTER_OUTSIDE_TOP_K = "letter_outside_top_k"
KIND_NO_LOGPROBS = "no_logprobs"
KIND_NO_RESPONSE = "no_response"

#: the per-item row the driver files for an MMLU item whose EVERY answer letter fell outside the
#: returned top-K (scripts/served_capability_run.py ``MMLU_UNDETERMINABLE_METRIC``): not an ``acc``
#: row, so the item stays unscored and the driver files no aggregate while one exists. It is MMLU
#: evidence of an attempted run — a store carrying only such rows was run, never "never ran" — and
#: it establishes why the aggregate is absent; tests/test_unscoreable_record.py pins the spelling
#: against the driver's.
UNDETERMINABLE_METRIC = "topk_undeterminable"

STATUS_COMPLETE = "complete"
STATUS_REFUSED = "refused_unscoreable"
STATUS_UNDETERMINABLE_ITEMS = "undeterminable_items"
STATUS_PARTIAL_NO_RECORD = "partial_no_record"
STATUS_NEVER_RAN = "never_ran"
STATUS_INCONSISTENT = "inconsistent"
STATUSES = (STATUS_COMPLETE, STATUS_REFUSED, STATUS_UNDETERMINABLE_ITEMS, STATUS_PARTIAL_NO_RECORD,
            STATUS_NEVER_RAN, STATUS_INCONSISTENT)

#: the MMLU legs' store roots under a controls root (ar_cell.sbatch): ``mmlu`` is the 200-item leg
#: (``--out-root $CTL_ROOT/mmlu``), ``mmlu_full`` the whole-pool leg (``legs=mmlu_full``,
#: ``--out-root $CTL_ROOT/mmlu_full``). Two roots, two requested counts, reported separately.
MMLU_LEGS = ("mmlu", "mmlu_full")

#: the manifest / row-constant fields that identify a store; whichever the caller has are copied.
IDENTITY_KEYS = ("cell_id", "harness_cell_id", "arm_name", "model_id", "model_revision",
                 "intervention_hash", "harness_config_hash", "engine", "slurm_job_id", "run_id")


class UnscoreableError(ValueError):
    """A record that would misdescribe the store — refused at the write."""


# --------------------------------------------------------------------------- #
# building a record                                                            #
# --------------------------------------------------------------------------- #


def letter_item(item_id: str, *, position_order: Optional[int],
                missing_letter_token_ids: Sequence[int], missing_letters: Sequence[str],
                gold: Optional[str], present_letter_logprobs: Optional[Mapping[str, float]],
                message: str) -> dict:
    """One item whose answer letter(s) fell below the server's top-K logprobs. ``gold`` and
    ``present_letter_logprobs`` are what the driver OBSERVED (None when the writer did not see
    them — a repair pass reading a log). They are diagnostics, not a score: the driver's rule is
    that an item with any letter absent is refused, never picked among the letters present."""
    if not missing_letter_token_ids:
        raise UnscoreableError(f"{item_id}: a {KIND_LETTER_OUTSIDE_TOP_K} item names at least one "
                               "missing letter token id")
    if len(missing_letters) != len(missing_letter_token_ids):
        raise UnscoreableError(f"{item_id}: {len(missing_letters)} letters for "
                               f"{len(missing_letter_token_ids)} missing token ids")
    return {
        "item_id": str(item_id),
        "position_order": (None if position_order is None else int(position_order)),
        "kind": KIND_LETTER_OUTSIDE_TOP_K,
        "missing_letter_token_ids": [int(t) for t in missing_letter_token_ids],
        "missing_letters": [str(s) for s in missing_letters],
        "gold": (None if gold is None else str(gold)),
        "present_letter_logprobs": (None if present_letter_logprobs is None
                                    else {str(k): float(v) for k, v in present_letter_logprobs.items()}),
        "message": str(message),
    }


def other_item(item_id: str, *, kind: str, position_order: Optional[int], message: str) -> dict:
    """An item that failed for a reason other than a letter outside the top-K (no response, no
    logprobs in the response). It reruns on resume; it is listed so the counts add up."""
    if kind not in (KIND_NO_LOGPROBS, KIND_NO_RESPONSE):
        raise UnscoreableError(f"{item_id}: kind {kind!r} is not one of "
                               f"{[KIND_NO_LOGPROBS, KIND_NO_RESPONSE]}")
    return {"item_id": str(item_id),
            "position_order": (None if position_order is None else int(position_order)),
            "kind": kind, "missing_letter_token_ids": [], "missing_letters": [],
            "gold": None, "present_letter_logprobs": None, "message": str(message)}


def mmlu_failure_block(*, top_logprobs: int, n_requested: int, n_scored: int,
                       n_correct_scored: int, items: Sequence[Mapping[str, Any]],
                       task: str = MMLU_TASK) -> dict:
    """The ``controls["mmlu"]`` block: counts, the cap in force, every failed item, and the EXACT
    bounds the unscored items leave on the requested-set accuracy.

    The bounds are exact, not an approximation: an unscored item's accuracy is 0 or 1, so the
    requested-set accuracy lies in ``[n_correct / n_requested, (n_correct + n_unscored) /
    n_requested]``. For a ``letter_outside_top_k`` item the value is moreover well defined under
    the SAME instrument — a letter absent from the top-K ranks below every letter present, so the
    argmax over the present letters is the argmax over all four and a larger K could not change
    the pick; only which present letter won is unrecorded."""
    n_requested, n_scored, n_correct = int(n_requested), int(n_scored), int(n_correct_scored)
    n_unscored = n_requested - n_scored
    if n_requested <= 0:
        raise UnscoreableError(f"n_requested must be positive, got {n_requested}")
    if n_unscored <= 0:
        raise UnscoreableError(f"nothing unscored: {n_scored} scored of {n_requested} requested — "
                               "a complete leg files its aggregate, not this record")
    if not 0 <= n_correct <= n_scored:
        raise UnscoreableError(f"n_correct_scored {n_correct} outside [0, {n_scored}]")
    items = [dict(it) for it in items]
    if len(items) != n_unscored:
        raise UnscoreableError(f"the block names {len(items)} failed item(s) but "
                               f"{n_unscored} = {n_requested} requested - {n_scored} scored are "
                               "unscored — every unscored item must be named")
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise UnscoreableError(f"duplicate item ids in the failure list: {ids}")
    kinds = Counter(it["kind"] for it in items)
    lo, hi = n_correct / n_requested, (n_correct + n_unscored) / n_requested
    return {
        "task": task,
        "top_logprobs": int(top_logprobs),
        "n_requested": n_requested,
        "n_scored": n_scored,
        "n_unscored": n_unscored,
        "n_correct_scored": n_correct,
        "acc_over_scored": (n_correct / n_scored if n_scored else None),
        "acc_over_scored_definition": f"n_correct_scored / n_scored — the {n_scored}-item value; "
                                      "NOT the aggregate the driver would have filed",
        "acc_bounds_over_requested": [lo, hi],
        "acc_bounds_rationale": (
            "EXACT: an unscored item scores 0 or 1, so the requested-set accuracy lies in "
            "[n_correct_scored / n_requested, (n_correct_scored + n_unscored) / n_requested]. For "
            "a letter_outside_top_k item the value is also well defined under this instrument: a "
            "letter absent from the top-K ranks below every letter present, so the argmax over the "
            "present letters is the argmax over all four and a larger K could not change the pick; "
            "only which present letter won is unrecorded"),
        "kinds": dict(sorted(kinds.items())),
        "unscoreable_item_ids": [it["item_id"] for it in items
                                 if it["kind"] == KIND_LETTER_OUTSIDE_TOP_K],
        "items": items,
        "resume": (f"the {n_scored} scored items are in rows.jsonl and resume; a re-run at the same "
                   f"--top-logprobs {int(top_logprobs)} re-requests only the unscored items. A LARGER "
                   "--top-logprobs enters harness_config_hash and therefore the store key: that is "
                   "a different instrument and a different store, not a resume of this one"),
    }


def why_no_aggregate(control: str, block: Mapping[str, Any]) -> str:
    """The one-line statement every record carries, per control."""
    kinds = ", ".join(f"{n} {k}" for k, n in sorted(dict(block["kinds"]).items()))
    return (f"{control}: the __aggregate__ row and gen_stats.json are ABSENT because "
            f"{block['n_unscored']} of {block['n_requested']} requested items could not be scored "
            f"({kinds}; top_logprobs={block['top_logprobs']}); scripts/served_capability_run.py "
            f"refuses a partial aggregate and exits before gen_stats.json is written — "
            f"{block['n_scored']} scored items are in rows.jsonl")


def identity_from(*sources: Mapping[str, Any]) -> dict:
    """The identity fields present in any of ``sources`` (a row-constants template, a manifest,
    a manifest's row_constants), first source wins per key."""
    out: dict = {}
    for src in sources:
        for k in IDENTITY_KEYS:
            if k in src and k not in out and src[k] is not None:
                out[k] = src[k]
    return out


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """temp + fsync + os.replace (the store's own discipline): existence means fully published."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_record(*, written_by: str, identity: Mapping[str, Any],
                 controls: Mapping[str, Mapping[str, Any]],
                 extra: Optional[Mapping[str, Any]] = None) -> dict:
    """The record as a dict, validated — what :func:`write_record` publishes. Separate so a
    repair pass can compare a would-be record against one already on disk before writing."""
    if written_by not in (WRITTEN_BY_DRIVER, WRITTEN_BY_REPAIR_PASS):
        raise UnscoreableError(f"written_by must be {WRITTEN_BY_DRIVER!r} or "
                               f"{WRITTEN_BY_REPAIR_PASS!r}, got {written_by!r}")
    if not controls:
        raise UnscoreableError("a record names at least one control's failure block")
    for name, blk in controls.items():
        for k in ("n_requested", "n_scored", "n_unscored", "top_logprobs", "items", "kinds"):
            if k not in blk:
                raise UnscoreableError(f"controls[{name!r}] lacks {k!r}")
    record: dict = {
        "schema": SCHEMA,
        "record": RECORD_NAME,
        "written_by": written_by,
        "written_at": _utc_now(),
        "identity": dict(identity),
        "why_no_aggregate": {name: why_no_aggregate(name, blk) for name, blk in controls.items()},
        "controls": {name: dict(blk) for name, blk in controls.items()},
    }
    if extra:
        clash = sorted(set(extra) & set(record))
        if clash:
            raise UnscoreableError(f"extra keys collide with the record's own: {clash}")
        record.update(dict(extra))
    return record


def write_record(cell_dir: Union[str, Path], *, written_by: str, identity: Mapping[str, Any],
                 controls: Mapping[str, Mapping[str, Any]],
                 extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Publish ``unscoreable_items.json`` in ``cell_dir``. Nothing else in the directory is
    touched. A record already there (an earlier failed attempt of the same store) is not lost:
    it is nested under ``previous_attempts`` of the new one, oldest first."""
    record = build_record(written_by=written_by, identity=identity, controls=controls,
                          extra=extra)
    cell_dir = Path(cell_dir)
    path = cell_dir / RECORD_NAME
    previous = load_record(cell_dir)
    if previous is not None:
        history = list(previous.pop("previous_attempts", []) or [])
        history.append(previous)
        record["previous_attempts"] = history
    _write_json_atomic(path, record)
    return path


# --------------------------------------------------------------------------- #
# reading a store                                                              #
# --------------------------------------------------------------------------- #


def load_record(cell_dir: Union[str, Path]) -> Optional[dict]:
    path = Path(cell_dir) / RECORD_NAME
    if not path.is_file():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        raise UnscoreableError(f"{path}: not a {SCHEMA} record (schema={rec.get('schema')!r})"
                               if isinstance(rec, dict) else f"{path}: not a JSON object")
    return rec


def mmlu_rows(cell_dir: Union[str, Path], *, task: str = MMLU_TASK) -> Dict[str, Any]:
    """Every MMLU row of ``task`` in rows.jsonl, sorted by kind: ``acc`` (the scored per-item rows),
    ``undeterminable`` (the driver's :data:`UNDETERMINABLE_METRIC` rows — attempted items no letter
    of which the top-K carried), ``has_aggregate`` (an ``__aggregate__`` row of the task exists).
    A torn FINAL line (a write cut mid-way) is ignored, as the store's own reader does; a corrupt
    non-final line is a damaged file and a refusal."""
    path = Path(cell_dir) / "rows.jsonl"
    out: Dict[str, Any] = {"acc": [], "undeterminable": [], "has_aggregate": False}
    if not path.is_file():
        return out
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise UnscoreableError(f"{path}: corrupt non-final line {i + 1}")
        if r.get("task") != task:
            continue
        if r.get("item_id") == AGGREGATE_ITEM_ID:
            out["has_aggregate"] = True
        elif r.get("metric") == "acc":
            out["acc"].append(r)
        elif r.get("metric") == UNDETERMINABLE_METRIC:
            out["undeterminable"].append(r)
    return out


def mmlu_item_rows(cell_dir: Union[str, Path], *, task: str = MMLU_TASK) -> Tuple[List[dict], bool]:
    """``(per-item acc rows of task, whether an __aggregate__ row of task exists)`` — the scored
    half of :func:`mmlu_rows`, for readers pairing stores on metric ``acc``."""
    rows = mmlu_rows(cell_dir, task=task)
    return rows["acc"], rows["has_aggregate"]


def mmlu_store_status(cell_dir: Union[str, Path], *, n_requested: Optional[int] = None) -> dict:
    """What one MMLU store IS, read from the store itself and nothing else.

    ``status`` is one of
      * ``complete``             — gen_stats.json carries the mmlu aggregate;
      * ``refused_unscoreable``  — no gen_stats.json, and ``unscoreable_items.json`` says why:
                                    N of M scored, the items and the top-K cap named;
      * ``undeterminable_items`` — no gen_stats.json, no record, but the driver's own
                                    ``topk_undeterminable`` rows for items still unscored: an
                                    ATTEMPTED run whose aggregate the driver refused (it files none
                                    while any item is undeterminable), the items named from the rows;
      * ``partial_no_record``    — acc rows but neither gen_stats.json nor a record nor an
                                    undeterminable row: a run cut short; the store does NOT record
                                    why (a job killed, a request that returned nothing usable, or a
                                    driver predating the record), and this reader does not guess;
      * ``never_ran``            — no MMLU row of any kind, no aggregate, no gen_stats.json, no record;
      * ``inconsistent``         — the files disagree with each other (named in ``why``).
    ``n_requested`` may be supplied by a caller who knows the requested count (the parity N) for
    the statuses where the store cannot state it; otherwise it is None there. ``n_undeterminable``
    and ``undeterminable_item_ids`` count the items with an undeterminable row and no later acc row
    (a resumed run re-attempts them; one that then scored is a scored item, its old row history)."""
    cell_dir = Path(cell_dir)
    gs_path = cell_dir / "gen_stats.json"
    rec = load_record(cell_dir)
    rows = mmlu_rows(cell_dir)
    items, has_aggregate = rows["acc"], rows["has_aggregate"]
    n_scored = len(items)
    n_correct = int(round(sum(float(r.get("value") or 0.0) for r in items)))
    scored_ids = {str(r.get("item_id")) for r in items}
    undet_ids = sorted({str(r.get("item_id")) for r in rows["undeterminable"]} - scored_ids)
    out: dict = {
        "store": str(cell_dir),
        "status": None,
        "n_scored": n_scored,
        "n_requested": n_requested,
        "n_correct_scored": n_correct,
        "n_undeterminable": len(undet_ids),
        "undeterminable_item_ids": undet_ids,
        "acc": None,
        "acc_bounds_over_requested": None,
        "top_logprobs": None,
        "unscoreable_item_ids": [],
        "record": (str(cell_dir / RECORD_NAME) if rec is not None else None),
        "record_written_by": (rec.get("written_by") if rec is not None else None),
        "gen_stats": (str(gs_path) if gs_path.is_file() else None),
        "why": None,
    }
    gs = json.loads(gs_path.read_text(encoding="utf-8")) if gs_path.is_file() else None
    if isinstance(gs, dict) and isinstance(gs.get("mmlu"), dict):
        blk = gs["mmlu"]
        if not has_aggregate:
            out["status"] = STATUS_INCONSISTENT
            out["why"] = ("gen_stats.json carries an mmlu block but rows.jsonl has no "
                          "__aggregate__ row — the producer files the aggregate before gen_stats")
            return out
        out.update(status=STATUS_COMPLETE, n_requested=blk.get("n"), acc=blk.get("acc"),
                   top_logprobs=blk.get("top_logprobs"),
                   why="gen_stats.json carries the mmlu aggregate"
                       + (f"; {RECORD_NAME} from an earlier attempt is present as history"
                          if rec is not None else ""))
        return out
    if rec is not None:
        blk = (rec.get("controls") or {}).get("mmlu")
        if not isinstance(blk, dict):
            out["status"] = STATUS_INCONSISTENT
            out["why"] = f"{RECORD_NAME} names no mmlu block (controls={sorted(rec.get('controls') or {})})"
            return out
        if int(blk["n_scored"]) != n_scored:
            out["status"] = STATUS_INCONSISTENT
            out["why"] = (f"{RECORD_NAME} says {blk['n_scored']} scored but rows.jsonl carries "
                          f"{n_scored} mmlu item rows")
            return out
        if has_aggregate:
            out["status"] = STATUS_INCONSISTENT
            out["why"] = f"{RECORD_NAME} present but rows.jsonl carries an __aggregate__ row"
            return out
        out.update(status=STATUS_REFUSED, n_requested=int(blk["n_requested"]),
                   acc=blk.get("acc_over_scored"),
                   acc_bounds_over_requested=blk.get("acc_bounds_over_requested"),
                   top_logprobs=blk.get("top_logprobs"),
                   unscoreable_item_ids=list(blk.get("unscoreable_item_ids") or []),
                   why=(rec.get("why_no_aggregate") or {}).get("mmlu"))
        return out
    if n_scored == 0 and not rows["undeterminable"] and not has_aggregate:
        # NO MMLU evidence of any kind: never_ran is reserved for this shape. A store with only
        # undeterminable rows was run (the driver filed them), and reads so below.
        out["status"] = STATUS_NEVER_RAN
        out["why"] = ("no mmlu item rows (neither acc nor topk_undeterminable), no __aggregate__ "
                      f"row, no gen_stats.json and no {RECORD_NAME}: the leg never scored or "
                      "attempted an item in this store (a standup or positive-control abort before "
                      "scoring, or a job killed first)")
        return out
    if has_aggregate:
        out["status"] = STATUS_INCONSISTENT
        out["why"] = "rows.jsonl carries an mmlu __aggregate__ row but gen_stats.json is absent"
        return out
    if undet_ids:
        # the rows ESTABLISH a cause: the driver files no aggregate while any item is undeterminable
        # (every answer letter outside the top-K), and these items are still unscored. Whether the
        # run also ended for another reason is not recorded, and is not claimed.
        out["status"] = STATUS_UNDETERMINABLE_ITEMS
        out["why"] = (f"{n_scored} mmlu acc rows and {len(undet_ids)} {UNDETERMINABLE_METRIC} rows for "
                      f"items still unscored ({', '.join(undet_ids[:5])}{', ...' if len(undet_ids) > 5 else ''}), "
                      f"no __aggregate__ row, no gen_stats.json and no {RECORD_NAME}: the driver files "
                      "no aggregate while any item is undeterminable (every answer letter outside the "
                      "returned top-K), so the aggregate's absence is established by the rows; whether "
                      "the run also ended for another reason is not recorded; the requested count is "
                      "not stated by the store")
        return out
    out["status"] = STATUS_PARTIAL_NO_RECORD
    out["why"] = (f"{n_scored} mmlu item rows, no __aggregate__ row, no gen_stats.json, no "
                  f"{UNDETERMINABLE_METRIC} row for an unscored item and no {RECORD_NAME}: a run cut "
                  "before completion; the store does not record why (a job killed, a request that "
                  "returned nothing usable, or a driver predating the record), and this reader does "
                  "not guess; the requested count is not stated by the store")
    return out


def mmlu_stores_facts(controls_root: Union[str, Path], *, legs: Sequence[str] = MMLU_LEGS,
                      n_requested_by_leg: Optional[Mapping[str, int]] = None) -> dict:
    """The cell_facts.json fragment: for each MMLU leg in ``legs`` (:data:`MMLU_LEGS`: the 200-item
    ``mmlu`` leg and the whole-pool ``mmlu_full`` leg, two store roots), every store under
    ``<controls_root>/<leg>/cells/`` with its status, keyed by the store's cell key
    (``arditi__controls__...``), plus the arm each key belongs to — under ``legs[<leg>]``, never
    merged: the two legs request different item counts and are different measurements. A leg
    whose root is absent reads ``present: False`` (that leg did not run under this controls root).
    ``n_requested_by_leg`` supplies the requested count per leg where the caller knows it."""
    out: dict = {
        "record": RECORD_NAME,
        "reader": "sisteer.capeval.unscoreable.mmlu_store_status",
        "statuses": list(STATUSES),
        "legs": {},
    }
    for leg in legs:
        root = Path(controls_root) / leg / "cells"
        rec: dict = {"root": f"{leg}/cells", "present": root.is_dir(), "stores": {}}
        if not root.is_dir():
            rec["why"] = f"no {root}: the {leg} leg did not run under this controls root"
        else:
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                st = mmlu_store_status(d, n_requested=(n_requested_by_leg or {}).get(leg))
                st["arm"] = d.name.split("__", 1)[0]
                rec["stores"][d.name] = st
        out["legs"][leg] = rec
    return out
