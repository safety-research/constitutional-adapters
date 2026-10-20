"""``registry cap-check``: the read-only report on the token-cap ladder facts of the live ``run`` rows --
the half of the rule that needs a store read, kept OUT of the fold.

The ruling: a run whose instrument's ladder has more than one level while its own ``token_cap_ladder_json`` is empty, or whose
``token_cap_tokens`` is not the level its ``token_cap_rung_index`` names, is wrong. The record-level half (the index, the ladder and the
cap of ONE record agreeing with each other) is a pure validator rule, :func:`validate.run_cap_ladder_refusal`, applied to every ``run``
record that states an index. The instrument half -- "its instrument's ladder has more than one level" -- is a cross-table read that
would be arrival-dependent at the fold (the instrument row may land after the run; replay applies records in written order), so it is a
REPORT: this module. Nothing here writes; the CLI resolves its root without the ``state/last_run`` stamp.

What the report holds, per campaign or whole:

- ``rule``: the live rows that state an index, re-checked with the validator's own function -- the live table disagreeing with the
  validator (a row folded before the rule that a replay would now refuse) is the one finding that exits 1; 0 live today.
- ``shapes``: the legacy shapes, admitted by the validator and counted here -- a ladder stated with no index (6,340 live rows on
  2026-10-03; 4 of them with a cap outside their own ladder), an empty ladder with no index (2,399, split by the instrument's ladder:
  one level, several -- 1,136, by campaign -- or none), and the rows on a multi-level instrument that state no index at all.
- ``instrument``: the cross-check -- every run on an instrument WITH a ladder, one level or several, whose cap is outside it (8 live:
  a 16,384 draw on a [512, 2048, 8192] instrument, an 8,192 draw on a [512, 4096] one; the 826
  single-level instruments carry 6,620 runs, none outside -- the check reaches them since 2026-10-03: a run on a
  ``[512]`` instrument drawn at 8,192 with no ladder and no index of its own is a legacy shape the validator admits, and a check scoped
  to multi-level instruments passed it in silence), the runs on an instrument with NO ladder, counted as unchecked rather than passed
  (0 live), every run whose own ladder differs from its instrument's (0 live), and the count of runs whose cap differs from the
  instrument's DECLARED cap (expected for a ladder run below its final level; ``v_instrument_mismatch`` carries the same fact per run).

Lists are bounded (``limit``, the CLI's first 20) unless asked whole (``--all``); counts are always whole.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional

from . import validate as VAL

DECISION = "D648"
#: Rows listed per bucket unless the caller asks for every one.
DEFAULT_LIMIT = 20
#: The run columns every listed row carries.
ROW_COLUMNS = ("run_id", "campaign", "model_id", "eval_id", "instrument_id", "lifecycle_state", "token_cap_tokens", "token_cap_rung_index",
               "token_cap_ladder_json")

_ROW_SELECT = ", ".join(f"r.{c}" for c in ROW_COLUMNS)


def _ladder(text) -> list:
    """A ladder column as the list it serializes (``[]`` for NULL or anything that is not a JSON array)."""
    if isinstance(text, list):
        return text
    if not isinstance(text, str):
        return []
    try:
        val = json.loads(text)
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


def _where(campaign: Optional[str], *more: str) -> str:
    parts = list(more)
    if campaign is not None:
        parts.append("r.campaign = :campaign")
    return (" WHERE " + " AND ".join(parts)) if parts else ""


def _rows(conn: sqlite3.Connection, sql: str, params: Dict[str, object], limit: Optional[int]) -> List[Dict[str, object]]:
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _count(conn: sqlite3.Connection, sql: str, params: Dict[str, object]) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def report(conn: sqlite3.Connection, campaign: Optional[str] = None, limit: Optional[int] = DEFAULT_LIMIT) -> Dict[str, object]:
    """The report as one JSON-able dict (the module docstring names its sections). Reads ``run`` and ``instrument`` only."""
    conn.row_factory = sqlite3.Row
    params: Dict[str, object] = {"campaign": campaign} if campaign is not None else {}
    cap_in_own = "EXISTS (SELECT 1 FROM json_each(r.token_cap_ladder_json) le WHERE le.value = r.token_cap_tokens)"
    cap_in_instr = "EXISTS (SELECT 1 FROM json_each(i.token_cap_ladder_json) le WHERE le.value = r.token_cap_tokens)"
    multi = "json_array_length(i.token_cap_ladder_json) > 1"
    single = "json_array_length(i.token_cap_ladder_json) = 1"
    # the column is NOT NULL with default '[]' on both tables: every instrument is exactly one of laddered / unladdered
    laddered = "json_array_length(i.token_cap_ladder_json) > 0"
    unladdered = "json_array_length(i.token_cap_ladder_json) = 0"
    join = "FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id"

    n_runs = _count(conn, "SELECT COUNT(*) FROM run r" + _where(campaign), params)

    # -- rule (i) on the live rows: the validator's own function, row by row -------------------------------------------------------
    stated = _rows(conn, f"SELECT {_ROW_SELECT} FROM run r" + _where(campaign, "r.token_cap_rung_index IS NOT NULL") + " ORDER BY r.run_id",
                   params, None)
    violations: List[Dict[str, object]] = []
    for row in stated:
        bad = VAL.run_cap_ladder_refusal(row["token_cap_tokens"], row["token_cap_rung_index"], row["token_cap_ladder_json"])
        if bad:
            violations.append({**row, "field": bad[0], "reason": bad[1]})

    # -- the legacy shapes (no index) -----------------------------------------------------------------------------------------------
    no_index = "r.token_cap_rung_index IS NULL"
    ladder_set = "json_array_length(r.token_cap_ladder_json) > 0"
    ladder_empty = "json_array_length(r.token_cap_ladder_json) = 0"
    shapes: Dict[str, object] = {
        "ladder_stated_no_index": {
            "count": _count(conn, "SELECT COUNT(*) FROM run r" + _where(campaign, no_index, ladder_set), params),
            "cap_in_own_ladder": _count(conn, "SELECT COUNT(*) FROM run r" + _where(campaign, no_index, ladder_set, cap_in_own), params),
            "cap_outside_own_ladder": _rows(conn, f"SELECT {_ROW_SELECT} FROM run r" + _where(campaign, no_index, ladder_set, f"NOT {cap_in_own}")
                                            + " ORDER BY r.run_id", params, limit),
        },
        "ladder_empty_no_index": {
            "count": _count(conn, "SELECT COUNT(*) FROM run r" + _where(campaign, no_index, ladder_empty), params),
            "on_single_level_instrument": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, no_index, ladder_empty, single), params),
            "on_instrument_without_a_ladder": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, no_index, ladder_empty, unladdered), params),
            "on_multi_level_instrument": {
                "count": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, no_index, ladder_empty, multi), params),
                "by_campaign": {r["campaign"]: r["n"] for r in conn.execute(
                    f"SELECT r.campaign, COUNT(*) AS n {join}" + _where(campaign, no_index, ladder_empty, multi)
                    + " GROUP BY r.campaign ORDER BY n DESC, r.campaign", params).fetchall()},
                "runs": _rows(conn, f"SELECT {_ROW_SELECT}, i.token_cap_ladder_json AS instrument_ladder_json {join}"
                              + _where(campaign, no_index, ladder_empty, multi) + " ORDER BY r.run_id", params, limit),
            },
        },
        "no_index_on_multi_level_instrument": {
            "count": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, no_index, multi), params),
            "by_campaign": {r["campaign"]: r["n"] for r in conn.execute(
                f"SELECT r.campaign, COUNT(*) AS n {join}" + _where(campaign, no_index, multi) + " GROUP BY r.campaign ORDER BY n DESC, r.campaign",
                params).fetchall()},
        },
    }

    # -- the instrument cross-check ----------------------------------------------------------------------------------------------------
    # Every run on an instrument WITH a ladder -- one level or several -- is checked; a run on an instrument with no ladder has nothing to
    # be outside of and is COUNTED as unchecked, never passed in silence. Until 2026-10-03 the two cap-outside
    # queries were scoped to multi-level instruments (`multi`), so a run drawn at 8,192 on a [512] instrument with no ladder and no index
    # of its own -- a legacy shape the validator admits -- fell out of the "exhaustive" cross-check and the report said 0.
    instr_cols = ("i.token_cap_tokens AS instrument_token_cap_tokens, i.token_cap_ladder_json AS instrument_ladder_json, i.read_at AS instrument_read_at")
    instrument: Dict[str, object] = {
        "instruments_with_a_ladder": _count(conn, f"SELECT COUNT(*) FROM instrument i WHERE {laddered}", {}),
        "multi_level_instruments": _count(conn, f"SELECT COUNT(*) FROM instrument i WHERE {multi}", {}),
        "runs_on_instruments_with_a_ladder": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, laddered), params),
        "runs_on_multi_level_instruments": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, multi), params),
        "runs_on_instruments_without_a_ladder": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, unladdered), params),
        "cap_outside_instrument_ladder": _rows(conn, f"SELECT {_ROW_SELECT}, {instr_cols} {join}" + _where(campaign, laddered, f"NOT {cap_in_instr}")
                                               + " ORDER BY r.run_id", params, limit),
        "cap_outside_instrument_ladder_count": _count(conn, f"SELECT COUNT(*) {join}" + _where(campaign, laddered, f"NOT {cap_in_instr}"), params),
        "run_ladder_differs_from_instrument": _rows(conn, f"SELECT {_ROW_SELECT}, {instr_cols} {join}"
                                                    + _where(campaign, ladder_set, "json_array_length(i.token_cap_ladder_json) > 0",
                                                             "json(r.token_cap_ladder_json) <> json(i.token_cap_ladder_json)") + " ORDER BY r.run_id", params, limit),
        "run_ladder_differs_from_instrument_count": _count(conn, f"SELECT COUNT(*) {join}"
                                                           + _where(campaign, ladder_set, "json_array_length(i.token_cap_ladder_json) > 0",
                                                                    "json(r.token_cap_ladder_json) <> json(i.token_cap_ladder_json)"), params),
        "cap_differs_from_instrument_declared_cap": _count(conn, f"SELECT COUNT(*) {join}"
                                                           + _where(campaign, "r.token_cap_tokens IS NOT i.token_cap_tokens"), params),
    }
    listed = (instrument["cap_outside_instrument_ladder"] + instrument["run_ladder_differs_from_instrument"]      # type: ignore[operator]
              + shapes["ladder_empty_no_index"]["on_multi_level_instrument"]["runs"])                               # type: ignore[index]
    for row in listed:
        row["instrument_ladder"] = _ladder(row.get("instrument_ladder_json"))   # the list beside the column's text (its spacing is the producer's)
    return {
        "decision": DECISION,
        "campaign": campaign,
        "limit": limit,
        "runs": n_runs,
        "rule": {"index_stated": len(stated), "violations": violations},
        "shapes": shapes,
        "instrument": instrument,
        "exit_code": 1 if violations else 0,
    }


def _row_line(row: Dict[str, object], *extra: str) -> str:
    base = (f"  {row['run_id']}  {row['lifecycle_state']}  cap {row['token_cap_tokens']}  index {row.get('token_cap_rung_index')}  "
            f"ladder {row.get('token_cap_ladder_json')}")
    return base + "".join(f"  {e}" for e in extra)


def lines(rep: Dict[str, object]) -> List[str]:
    """The report as text, one finding per line; the JSON form is the whole thing."""
    rule = rep["rule"]            # type: ignore[index]
    shapes = rep["shapes"]        # type: ignore[index]
    inst = rep["instrument"]      # type: ignore[index]
    scope = f"campaign {rep['campaign']}" if rep.get("campaign") else "every campaign"
    out = [f"cap-check ({DECISION}; {scope}; read-only): {rep['runs']} runs",
           f"record rule on the live rows: {rule['index_stated']} state a ladder index, {len(rule['violations'])} disagree with it"
           + (" -- the live table disagrees with the validator (a replay would refuse these rows)" if rule["violations"] else "")]
    for row in rule["violations"]:
        out.append(_row_line(row, f"{row['field']}: {row['reason']}"))
    lsn = shapes["ladder_stated_no_index"]
    out.append(f"legacy shape, a ladder stated with no index: {lsn['count']} runs ({lsn['cap_in_own_ladder']} with the cap in the ladder, "
               f"{lsn['count'] - lsn['cap_in_own_ladder']} outside it)")
    for row in lsn["cap_outside_own_ladder"]:
        out.append(_row_line(row, "cap outside the run's own ladder"))
    len_ = shapes["ladder_empty_no_index"]
    oml = len_["on_multi_level_instrument"]
    out.append(f"legacy shape, an empty ladder with no index: {len_['count']} runs ({len_['on_single_level_instrument']} on a single-level "
               f"instrument, {oml['count']} on a multi-level instrument"
               + (": " + ", ".join(f"{c} {n}" for c, n in oml["by_campaign"].items()) if oml["by_campaign"] else "")
               + f", {len_['on_instrument_without_a_ladder']} on an instrument with no ladder)")
    for row in oml["runs"]:
        out.append(_row_line(row, f"instrument ladder {row.get('instrument_ladder_json')}"))
    nim = shapes["no_index_on_multi_level_instrument"]
    out.append(f"runs on a multi-level instrument stating no index: {nim['count']}"
               + (" (" + ", ".join(f"{c} {n}" for c, n in nim["by_campaign"].items()) + ")" if nim["by_campaign"] else ""))
    out.append(f"instrument cross-check: {inst['instruments_with_a_ladder']} instruments with a ladder ({inst['multi_level_instruments']} of them "
               f"multi-level), {inst['runs_on_instruments_with_a_ladder']} runs on them checked ({inst['runs_on_multi_level_instruments']} on a "
               f"multi-level instrument), {inst['runs_on_instruments_without_a_ladder']} on an instrument with no ladder, unchecked; cap outside "
               f"the instrument's ladder: {inst['cap_outside_instrument_ladder_count']}; run ladder unlike the instrument's: "
               f"{inst['run_ladder_differs_from_instrument_count']}; cap differing from the instrument's declared cap: "
               f"{inst['cap_differs_from_instrument_declared_cap']} (expected for a ladder run below its final level)")
    for row in inst["cap_outside_instrument_ladder"]:
        out.append(_row_line(row, f"cap outside the instrument's ladder {row['instrument_ladder_json']} (declared cap "
                                  f"{row['instrument_token_cap_tokens']}, read at {row['instrument_read_at']})"))
    for row in inst["run_ladder_differs_from_instrument"]:
        out.append(_row_line(row, f"run ladder unlike the instrument's {row['instrument_ladder_json']}"))
    if rep.get("limit") is not None:
        out.append(f"(lists bounded to the first {rep['limit']} rows of each bucket; --all lifts the bound; counts are whole)")
    return out
