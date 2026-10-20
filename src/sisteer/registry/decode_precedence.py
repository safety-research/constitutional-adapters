"""Best-of-N decode precedence: the vendor preset over T = 1 within one key (ruling
``r2v4-reference-bon-climb-decode-precedence-vendor-preset-over-t1``).

The rule. Within one best-of-N key -- (model, object, standup) on ``strongreject_bon`` at the cap-4,096 climb and the N = 75
headline -- the VENDOR-PRESET decode row is the row a figure reads when one exists; the T = 1 climb row (``decode_source =
grandfathered_T1``) is the fallback only where no vendor-preset row exists. An unsteered ``none`` anchor follows the decode of the
steered row it pairs with (a T = 1 steered row pairs with a T = 1 none, a vendor row with a vendor none), so the rule never touches
an anchor, and nothing here lists one.

A precedence, not a selection. ``v_paper_current`` knows nothing of the rule, and at filing no steered key carried both decodes
(census 2026-09-28 17:37Z: Qwen's seed standups s1-s3 are vendor-preset and selected; Qwen's IT-fit rows and every steered gemma /
llama / nemotron key carry the T = 1 climb only and are selected), so nothing changed hands. This module is the guard the ruling
names in its ``target_json.enforcement``: ONE query (:data:`PRECEDENCE_SQL`) listing every steered key where a SELECTED T = 1 row
sits beside a vendor-preset row, run by the status page (a count up top and a section) and by the live-database test
(``tests/test_registry_bon_decode_precedence.py``, which asserts the list is empty). When a key appears the remedy is a KEY-SCOPED
supersede of that key's T = 1 rows (:func:`supersede_ruling`, dry run first), never a blanket rule.

What "vendor preset" means here, from the vocabulary the registry already has (:func:`decode_kind`):

* the DERIVED match first -- a run whose ``decode_policy_id`` (``decode_match.stamp_run_rows``, from the parameters the run SENT) names
  a ``<model>|bon|<thinking>|vendor_preset@vN`` policy is vendor-preset; one naming any other policy (``...|grandfathered_T1@v1``) is not;
* the producer's ``decode_source`` where nothing matched (``decode_policy_id`` NULL: every T = 1 climb row reads ``unrecorded`` today, and a
  vendor store whose reader files no sent parameters would too) -- ``generation_config``, ``model_card`` and ``vendor_recommendation``
  are the three ``DECODE_SOURCES`` that name a vendor's published numbers (every ``|bon|`` vendor_preset policy in decode_policy.json
  carries one of the first two); ``grandfathered_T1``, ``runner_forced_T1``, ``vllm_default``, ``greedy``, ``legacy_contracts_row`` and
  ``unverified`` are not.

The policy id is preferred because it is derived from what was sent; the source is the fallback because the id is NULL exactly where
the derivation had nothing to read. A row the two disagree on (a ``grandfathered_T1`` source under a vendor_preset policy) reads
vendor: the derived match wins over the producer's word.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Dict, List, Mapping, Optional

#: The reference ruling this guard enforces (``~/registry/rulings/added/2026-09-28-ruling-<id>.jsonl``).
RULING_ID = "r2v4-reference-bon-climb-decode-precedence-vendor-preset-over-t1"
EVAL_ID = "strongreject_bon"
#: The best-of-N climb the rule speaks of: the cap-4,096 climb read at the N = 75 headline (``instrument.bon_n``).
CLIMB_CAP_TOKENS = 4096
HEADLINE_BON_N = 75
T1_DECODE_SOURCE = "grandfathered_T1"
#: The ``DECODE_SOURCES`` that name a vendor's published numbers (vocab.EVIDENCE["DECODE_SOURCES"]).
VENDOR_DECODE_SOURCES = ("generation_config", "model_card", "vendor_recommendation")
#: The policy-id spelling of a vendor preset (decode_policy.json: ``<model>|<eval>|<thinking>|vendor_preset@vN``).
VENDOR_POLICY_MARK = "|vendor_preset@"
#: The two states a listed key can be in. ``both_selected`` is the violation: a selected T = 1 row beside a selected vendor-preset
#: row. ``vendor_not_in_roster`` is informational: a vendor-preset row exists for the key but the roster has not named it, so the
#: T = 1 row is still the row read -- nothing to supersede yet, but the row the rule prefers is there.
VIOLATION_STATE = "both_selected"
PENDING_STATE = "vendor_not_in_roster"
STATES = (VIOLATION_STATE, PENDING_STATE)


def vendor_preset_sql(alias: str = "r") -> str:
    """The SQL of the vendor-preset rule on a ``run`` row aliased ``alias`` (1 / 0)."""
    pid = f"{alias}.decode_policy_id"
    src = f"{alias}.decode_source"
    sources = ", ".join(f"'{s}'" for s in VENDOR_DECODE_SOURCES)
    return (f"CASE WHEN {pid} IS NOT NULL AND {pid} <> '' THEN {pid} GLOB '*{VENDOR_POLICY_MARK}*' "
            f"ELSE {src} IN ({sources}) END")


def decode_kind_sql(alias: str = "r") -> str:
    """``vendor`` | ``t1`` | ``other`` for a ``run`` row aliased ``alias`` -- the same order as :func:`decode_kind`."""
    return (f"CASE WHEN {vendor_preset_sql(alias)} THEN 'vendor' "
            f"WHEN {alias}.decode_source = '{T1_DECODE_SOURCE}' THEN 't1' ELSE 'other' END")


def is_vendor_preset(decode_source: Optional[str], decode_policy_id: Optional[str]) -> bool:
    """The vendor-preset rule in Python, term for term the SQL of :func:`vendor_preset_sql`."""
    if decode_policy_id:
        return VENDOR_POLICY_MARK in decode_policy_id
    return decode_source in VENDOR_DECODE_SOURCES


def decode_kind(decode_source: Optional[str], decode_policy_id: Optional[str]) -> str:
    if is_vendor_preset(decode_source, decode_policy_id):
        return "vendor"
    return "t1" if decode_source == T1_DECODE_SOURCE else "other"


#: Every best-of-N row the rule speaks of, with its decode kind and selection: ``strongreject_bon`` at the climb cap and the headline
#: N, steered and unsteered alike (the census reads this; the guard restricts to steered keys below).
_ROWS_CTE = f"""
WITH climb AS (
    SELECT pc.run_id, r.model_id, r.object_id, o.paper_label, r.view_legacy, r.instrument_id, r.decode_source, r.decode_policy_id,
           r.decode_policy_match, pc.selection_state, {decode_kind_sql('r')} AS decode_kind
    FROM v_paper_current pc
    JOIN run r ON r.run_id = pc.run_id
    JOIN object o ON o.object_id = r.object_id
    JOIN instrument i ON i.instrument_id = r.instrument_id
    WHERE r.eval_id = '{EVAL_ID}' AND r.token_cap_tokens = {CLIMB_CAP_TOKENS} AND i.bon_n = {HEADLINE_BON_N}
)"""

#: THE guard query: one row per steered key (model, object, paper label, standup) where a SELECTED T = 1 row sits beside a
#: vendor-preset row -- ``state`` says whether that row is selected too (the violation) or not in the roster (pending). The
#: run ids and instrument ids of both sides ride the row so the remedy can be scoped from it.
PRECEDENCE_SQL = _ROWS_CTE + f""",
keyed AS (
    SELECT model_id, object_id, paper_label, view_legacy,
           COUNT(*) FILTER (WHERE decode_kind = 't1' AND selection_state = 'selected') AS n_t1_selected,
           COUNT(*) FILTER (WHERE decode_kind = 'vendor' AND selection_state = 'selected') AS n_vendor_selected,
           COUNT(*) FILTER (WHERE decode_kind = 'vendor' AND selection_state = 'not_in_roster') AS n_vendor_not_in_roster,
           json_group_array(run_id) FILTER (WHERE decode_kind = 't1' AND selection_state = 'selected') AS t1_run_ids_json,
           json_group_array(DISTINCT instrument_id) FILTER (WHERE decode_kind = 't1' AND selection_state = 'selected') AS t1_instrument_ids_json,
           json_group_array(run_id) FILTER (WHERE decode_kind = 'vendor' AND selection_state IN ('selected', 'not_in_roster')) AS vendor_run_ids_json,
           json_group_array(DISTINCT decode_source) FILTER (WHERE decode_kind = 'vendor') AS vendor_decode_sources_json,
           json_group_array(DISTINCT COALESCE(decode_policy_id, '')) FILTER (WHERE decode_kind = 'vendor') AS vendor_decode_policy_ids_json
    FROM climb WHERE paper_label <> 'none'
    GROUP BY model_id, object_id, paper_label, view_legacy
)
SELECT CASE WHEN n_vendor_selected > 0 THEN '{VIOLATION_STATE}' ELSE '{PENDING_STATE}' END AS state,
       model_id, object_id, paper_label, view_legacy, n_t1_selected, n_vendor_selected, n_vendor_not_in_roster,
       t1_run_ids_json, t1_instrument_ids_json, vendor_run_ids_json, vendor_decode_sources_json, vendor_decode_policy_ids_json
FROM keyed
WHERE n_t1_selected > 0 AND (n_vendor_selected > 0 OR n_vendor_not_in_roster > 0)
ORDER BY CASE WHEN n_vendor_selected > 0 THEN 0 ELSE 1 END, model_id, paper_label, view_legacy, object_id"""

#: The census the guard is computed from: counts per (model, steered, decode source, decode policy, decode kind, selection).
CENSUS_SQL = _ROWS_CTE + """
SELECT model_id, CASE WHEN paper_label <> 'none' THEN 1 ELSE 0 END AS steered, decode_source, COALESCE(decode_policy_id, '') AS decode_policy_id,
       decode_policy_match, decode_kind, selection_state, COUNT(*) AS n_runs, COUNT(DISTINCT object_id || '|' || COALESCE(view_legacy, '')) AS n_keys
FROM climb
GROUP BY model_id, steered, decode_source, decode_policy_id, decode_policy_match, decode_kind, selection_state
ORDER BY model_id, steered DESC, decode_kind, decode_source, selection_state"""


def _dicts(conn: sqlite3.Connection, sql: str) -> List[Dict[str, object]]:
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def precedence_rows(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    """Every steered key the guard lists, violations first (:data:`PRECEDENCE_SQL`)."""
    return _dicts(conn, PRECEDENCE_SQL)


def violations(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    """The keys that break the rule: a selected T = 1 row beside a selected vendor-preset row. EMPTY is the invariant."""
    return [r for r in precedence_rows(conn) if r["state"] == VIOLATION_STATE]


def census(conn: sqlite3.Connection) -> List[Dict[str, object]]:
    return _dicts(conn, CENSUS_SQL)


def _slug(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "no-standup").lower()).strip("-") or "no-standup"


def supersede_scope(row: Mapping[str, object]) -> Dict[str, object]:
    """The KEY-SCOPED supersede of one listed key's T = 1 rows: every field of the key (model, object, eval, standup), the T = 1 decode
    source, the climb cap and the T = 1 rows' instruments -- exactly the rows the rule says are the fallback where a vendor row exists.
    The ``match`` lists are the ones ``v_paper_current`` matches on (schema_def: ``model_ids`` ... ``instrument_ids``)."""
    match: Dict[str, object] = {
        "model_ids": [row["model_id"]], "object_ids": [row["object_id"]], "eval_ids": [EVAL_ID],
        "decode_sources": [T1_DECODE_SOURCE], "token_caps": [CLIMB_CAP_TOKENS],
        "instrument_ids": sorted(json.loads(str(row["t1_instrument_ids_json"]))),
    }
    if row.get("view_legacy") is not None:
        match["views_legacy"] = [row["view_legacy"]]
    return {"match": match,
            "note": (f"the T = 1 ({T1_DECODE_SOURCE}) best-of-N climb rows of one key -- {row['model_id']} x {row['paper_label']} "
                     f"({str(row['object_id'])[:12]}) at standup {row.get('view_legacy') or '(none recorded)'}, cap {CLIMB_CAP_TOKENS}, "
                     f"N {HEADLINE_BON_N} -- beside which a vendor-preset row is selected (ruling {RULING_ID})")}


def supersede_ruling(row: Mapping[str, object], ruled_utc: str, ruled_by: str, round_id: str = "r2", ruling_id_prefix: str = "r2v4",
                     source_ref: Optional[str] = None) -> Dict[str, object]:
    """The supersede ruling the remedy files for one violating key, as ``registry ruling add`` takes it (a dict; nothing is written).
    ``round_id`` must be the OPEN round the key's runs are flagged under (a ruling naming a round no ``paper_round`` row carries is
    dependency-blocked at ingest); ``ruling_id_prefix`` is the round-and-roster-version spelling the repo's ruling ids open with.
    Dry-run it first: the covered run ids are the row's ``t1_run_ids_json`` and the vendor rows that stand are its ``vendor_run_ids_json``."""
    t1_runs = json.loads(str(row["t1_run_ids_json"]))
    vendor_runs = json.loads(str(row["vendor_run_ids_json"]))
    rid = f"{ruling_id_prefix}-supersede-bon-t1-{_slug(row['model_id'])}-{_slug(row['paper_label'])}-{_slug(row.get('view_legacy'))}-by-vendor-preset"
    return {
        "ruling_id": rid, "round_id": round_id, "action": "supersede", "scope_kind": "key", "scope_json": supersede_scope(row),
        "target_json": {"superseded_by_run_ids": sorted(vendor_runs), "covered_run_ids_at_filing": sorted(t1_runs),
                        "rule": RULING_ID, "vendor_decode_sources": json.loads(str(row["vendor_decode_sources_json"])),
                        "why": ("within one best-of-N key the vendor-preset decode row is the row a figure reads when it exists; the T = 1 "
                                "climb row is the fallback only where no vendor-preset row exists (the user, 2026-09-28)")},
        "reason": (f"Ruling {RULING_ID} (the user, first-hand, 2026-09-28): for the best-of-N climb use the vendor presets where they exist "
                   f"and fall back to T = 1 only where they do not. This key ({row['model_id']} x {row['paper_label']} at standup "
                   f"{row.get('view_legacy') or '(none recorded)'}) now carries a selected vendor-preset row beside its selected T = 1 row, "
                   f"so the T = 1 row yields to it."),
        "source_ref": source_ref or f"ruling {RULING_ID}; D581; the guard's listing (decode_precedence.precedence_rows)",
        "ruled_utc": ruled_utc, "ruled_by": ruled_by,
    }


def runs_in_scope(conn: sqlite3.Connection, scope: Mapping[str, object]) -> List[str]:
    """Every ``run`` row the fields of a :func:`supersede_scope` name, selected or not -- a direct read of the key, so the dry run can
    show whether the scope reaches a run the guard did not list (an excluded or superseded T = 1 row of the same key)."""
    m = dict(scope["match"])  # type: ignore[arg-type]
    sql = ("SELECT run_id FROM run WHERE model_id = ? AND object_id = ? AND eval_id = ? AND decode_source = ? AND token_cap_tokens = ? "
           "AND instrument_id IN (" + ", ".join("?" for _ in m["instrument_ids"]) + ")")
    args: List[object] = [m["model_ids"][0], m["object_ids"][0], m["eval_ids"][0], m["decode_sources"][0], m["token_caps"][0], *m["instrument_ids"]]
    if "views_legacy" in m:
        sql += " AND view_legacy = ?"
        args.append(m["views_legacy"][0])
    return sorted(str(r[0]) for r in conn.execute(sql + " ORDER BY run_id", args).fetchall())
