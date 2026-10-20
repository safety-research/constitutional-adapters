"""Decode-policy match: the DERIVED pass that copies ``v_run_decode_policy_match`` onto the run row.

If an exact match between the decode parameters a run recorded and a decode_policy row can be verified, the policy
goes into the row. The verification is the SQL object ``v_run_decode_policy_match`` (``schema_def``: scope,
parameter-by-parameter equality, the supersession windows); this module is the ONE writer of
``run.decode_policy_id`` / ``run.decode_policy_match`` after ingest, called from the two places every other derived
run fact is recomputed -- the end of ``harvest.ingest_once`` and the end of ``harvest.rebuild`` -- so a rebuild
reproduces the live stamps byte for byte (the pass is a pure function of the ``run``, ``eval`` and ``decode_policy``
tables; no clock enters it).

The SQL object weighs an eval-scoped row and a category-scoped row equally, so a run whose parameters both admit reads
``ambiguous`` there (374 of 7,862 live runs on 2026-09-19, every one a ``<model>|<eval>|<thinking>|<mode>`` row beside its
``<model>|<category>|<thinking>|<mode>`` row with the same numbers). This pass breaks that tie in Python (a change to the
SQL object forces a rebuild, a change here re-stamps on the next cycle): among the candidates the SQL object left
standing, the rows in force at ``generated_utc`` (the temporal filter the object already applies; a row that took
effect after the run never beats one that governed it), then the ONE eval-scoped row over any category-scoped rows --
``resolve()``'s own order. What survives both is a true duplicate policy and stays ``ambiguous``. A resolved tie is
auditable on the row:
``extra_json.decode_policy_tiebreak = {"over": [<the losing policy ids>], "rule": "eval_over_category"}``; the object's
``candidates_json`` still lists every candidate.

A producer's run record carries the placeholder ``decode_policy_match = no_policy`` (and no policy id); the fold writes
that placeholder back whenever the run row re-folds, and the pass re-stamps it at the end of the same cycle. The two
columns are stamped in ``field_versions_json`` as ``[derived, v_run_decode_policy_match]`` -- the derivation's name in
the slot a record's ``(written_utc, sha)`` order key occupies for a recorded column -- so a reader of the stamps sees
that no record set them.

``python -m sisteer.registry.decode_match --db PATH`` is the READ-ONLY dry run: it computes the match over a database
that may predate the SQL object (the body SELECT is run directly, never created) and prints the distribution plus the
top (campaign, reason) pairs behind ``no_match``, writing nothing.
"""
from __future__ import annotations

import collections
import json
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from . import schema_def as S
from .identity import canonical_json

MATCH_SQL_NAME = "v_run_decode_policy_match"
#: the six sampling parameters the policy table models (decode_policy.PARAM_KEYS, restated here so this module stays free of
#: the harness-side reader)
PARAMS: Tuple[str, ...] = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
#: the ``field_versions_json`` slot a derived column carries in place of a record's (written_utc, sha) order key
STAMP: List[str] = ["derived", MATCH_SQL_NAME]
STAMPED_COLUMNS: Tuple[str, ...] = ("decode_policy_id", "decode_policy_match")
#: ``run.extra_json`` key a tie the Python pass resolved is written under (absent on every other row)
TIEBREAK_KEY = "decode_policy_tiebreak"
#: the one tie-break rule this pass applies (the value of ``extra_json.decode_policy_tiebreak.rule``)
TIEBREAK_RULE = "eval_over_category"


def match_select_sql() -> str:
    """The SQL object's body as a bare SELECT (for a read-only database that does not carry the object yet)."""
    sql = S.VIEW_SQL[MATCH_SQL_NAME].strip()
    m = re.match(rf"CREATE VIEW {MATCH_SQL_NAME} AS\s*(.*);\s*$", sql, flags=re.DOTALL)
    assert m, sql[:80]
    return m.group(1)


_RUN_READS = (re.compile(r"\bFROM run r\b"), re.compile(r"\bFROM run WHERE\b"), re.compile(r"\bJOIN run r ON\b"))
_SCOPED_CTE = "scoped_run"


def scoped_match_select_sql(run_ids: Sequence[str]) -> str:
    """The SQL object's body restricted to ``run_ids`` INSIDE its CTE chain: a leading ``scoped_run`` CTE reads those rows of ``run``
    once, and every read of ``run`` in the chain (``run_sent``, ``run_sent_unmodelled``, ``in_scope``, ``by_source``, the outer
    SELECT) reads it instead, so nothing materializes over the whole table (the live registry: ~12,500 runs x ~160 policy rows per
    claim before this). The ids are inlined as SQL literals (``schema_def.sql_literal``) because the body is spliced into other
    statements as a subquery. Returns a parenthesized SELECT, usable wherever :data:`MATCH_SQL_NAME` is."""
    body = match_select_sql()
    assert body.startswith("WITH "), body[:40]
    n = 0
    for rx in _RUN_READS:
        body, k = rx.subn(lambda m: m.group(0).replace("run", _SCOPED_CTE, 1), body)
        n += k
    # every base read of `run` in the chain is now the scoped one (the eight the SQL object has: run_sent x4, run_sent_unmodelled,
    # in_scope, by_source, the outer SELECT); a ninth read added to the object must be scoped here too, so the count is exact
    remaining = len(re.findall(r"\b(?:FROM|JOIN) run\b", body))
    assert n == 8 and remaining == 0, (n, remaining)
    ids = ", ".join("'" + str(r).replace("'", "''") + "'" for r in run_ids)      # SQL string literals, quotes doubled
    scoped = f"WITH {_SCOPED_CTE} AS (SELECT * FROM run WHERE run_id IN ({ids})),\n" + body[len("WITH "):]
    return f"({scoped})"


def tiebreak(conn: sqlite3.Connection, source: str = MATCH_SQL_NAME) -> Dict[str, Dict[str, object]]:
    """The Python tie-break over every run the SQL object reads ``ambiguous``: READ-ONLY, a pure function of the same tables.

    ``source`` is the SQL object's name, or its body SELECT in parentheses for a database that does not carry the object. Returns
    one entry per ambiguous run: ``policy_id`` (the row that wins, or None when the tie stands), ``over`` (the losing candidates),
    ``candidates`` / ``in_force`` / ``eval_scoped`` (the sets the decision read). The order: the candidates in force at
    ``generated_utc`` when any is (a candidate is already known not superseded before the run, so in force = took effect by
    then), else every candidate; within that pool exactly one eval-scoped row wins over the category-scoped rows; otherwise
    (two eval rows, or only category rows) the tie stands."""
    rows = conn.execute(f"""
SELECT m.run_id, j.value AS policy_id,
       CASE WHEN p.eval_id IS NOT NULL THEN 1 ELSE 0 END AS eval_scoped,
       CASE WHEN p.effective_from_utc = 'unverified'
                 OR p.effective_from_utc <= COALESCE(r.generated_utc, r.started_utc, r.queued_utc) THEN 1 ELSE 0 END AS in_force
FROM {source} m JOIN run r ON r.run_id = m.run_id, json_each(m.candidates_json) j
JOIN decode_policy p ON p.policy_id = j.value
WHERE m.match = 'ambiguous'
ORDER BY m.run_id, j.value""").fetchall()
    per_run: Dict[str, List[sqlite3.Row]] = collections.OrderedDict()
    for row in rows:
        per_run.setdefault(row[0], []).append(row)
    out: Dict[str, Dict[str, object]] = {}
    for run_id, cands in per_run.items():
        in_force = [c[1] for c in cands if c[3]]
        pool = [c for c in cands if c[3]] if in_force else list(cands)
        eval_scoped = [c[1] for c in pool if c[2]]
        winner = eval_scoped[0] if len(eval_scoped) == 1 and len(pool) > 1 else None
        out[run_id] = {"policy_id": winner, "over": [c[1] for c in pool if c[1] != winner] if winner else [],
                       "candidates": [c[1] for c in cands], "in_force": in_force, "eval_scoped": [c[1] for c in pool if c[2]]}
    return out


def tiebreak_audit(entry: Dict[str, object]) -> str:
    """The ``extra_json.decode_policy_tiebreak`` value for a resolved tie (compact, key-sorted; what the row carries)."""
    return canonical_json({"over": entry["over"], "rule": TIEBREAK_RULE})


def stamp_run_rows(conn: sqlite3.Connection, run_ids: Optional[Sequence[str]] = None) -> int:
    """Copy the derived verdict onto every run row whose stamped columns differ from it; returns the rows written.

    Runs inside the caller's transaction. The verdict is the SQL object's, except where :func:`tiebreak` resolves an
    ``ambiguous`` one: that run is stamped ``exact`` with the winning row and carries the losers under
    ``extra_json.decode_policy_tiebreak``; a row the tie-break does not decide has no such key (a stale one is removed).
    ``decode_policy_id`` is set only for an ``exact`` verdict; ``field_versions_json`` gains the derived stamp for both
    columns. Idempotent: a second call in the same state writes nothing. ``run_ids`` restricts the pass to those runs (the
    scoped fold re-stamps the rows its own records rewrote, ``harvest.fold_only_files``); the cycle passes none."""
    if run_ids is not None and not run_ids:
        return 0
    stamp = json.dumps(STAMP, separators=(",", ":"))
    source = MATCH_SQL_NAME if run_ids is None else scoped_match_select_sql(run_ids)
    resolved = {rid: e for rid, e in tiebreak(conn, source).items() if e["policy_id"]}
    scope = "" if run_ids is None else f"  AND run.run_id IN ({', '.join(':r%d' % i for i in range(len(run_ids)))})\n"
    scoped: Dict[str, object] = {} if run_ids is None else {f"r{i}": rid for i, rid in enumerate(run_ids)}
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS decode_tiebreak (run_id TEXT PRIMARY KEY, policy_id TEXT NOT NULL, audit TEXT NOT NULL)")
    conn.execute("DELETE FROM temp.decode_tiebreak")
    conn.executemany("INSERT INTO temp.decode_tiebreak (run_id, policy_id, audit) VALUES (?, ?, ?)",
                     [(rid, e["policy_id"], tiebreak_audit(e)) for rid, e in resolved.items()])
    try:
        sql = f"""
UPDATE run SET
  decode_policy_id = v.policy_id,
  decode_policy_match = v.match,
  field_versions_json = json_set(field_versions_json, '$.decode_policy_id', json(:stamp), '$.decode_policy_match', json(:stamp)),
  extra_json = CASE WHEN v.audit IS NULL THEN json_remove(extra_json, '$.{TIEBREAK_KEY}')
                    ELSE json_set(extra_json, '$.{TIEBREAK_KEY}', json(v.audit)) END
FROM (SELECT m.run_id, COALESCE(t.policy_id, m.policy_id) AS policy_id,
             CASE WHEN t.run_id IS NULL THEN m.match ELSE 'exact' END AS match, t.audit AS audit
      FROM {source} m LEFT JOIN temp.decode_tiebreak t ON t.run_id = m.run_id) AS v
WHERE v.run_id = run.run_id
{scope}  AND (run.decode_policy_id IS NOT v.policy_id
       OR run.decode_policy_match IS NOT v.match
       OR json_extract(run.field_versions_json, '$.decode_policy_match[1]') IS NOT :derivation
       OR json_extract(run.field_versions_json, '$.decode_policy_id[1]') IS NOT :derivation
       OR json_extract(run.extra_json, '$.{TIEBREAK_KEY}') IS NOT v.audit)
"""
        cur = conn.execute(sql, {"stamp": stamp, "derivation": MATCH_SQL_NAME, **scoped})
        return int(cur.rowcount or 0)
    finally:
        conn.execute("DROP TABLE IF EXISTS temp.decode_tiebreak")


# --------------------------------------------------------------------------------------------
# read-only dry run
# --------------------------------------------------------------------------------------------


def _num(v: object) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def no_match_reasons(conn: sqlite3.Connection, run_id: str) -> List[str]:
    """Why each in-scope policy failed for one run (Python restatement of the SQL predicate, for the report only):
    ``no_policy_in_scope`` when none admits the run; else per policy ``<policy_id>: <first failing check>``."""
    r = conn.execute("SELECT r.model_id, r.eval_id, e.eval_category, r.thinking_mode, r.decode_seed_rule, r.decode_sent_json, "
                     "r.decode_temperature_sent, r.decode_top_p_sent, r.decode_top_k_sent, "
                     "COALESCE(r.generated_utc, r.started_utc, r.queued_utc) AS generated "
                     "FROM run r JOIN eval e ON e.eval_id = r.eval_id WHERE r.run_id = ?", (run_id,)).fetchone()
    if r is None:
        return ["run missing"]
    sent: Dict[str, float] = {}
    unmodelled = []
    for k, v in (json.loads(r["decode_sent_json"] or "{}") or {}).items():
        n = _num(v)
        if k in PARAMS and n is not None:
            sent[k] = n
        else:
            unmodelled.append(k)
    for k, col in (("temperature", "decode_temperature_sent"), ("top_p", "decode_top_p_sent"), ("top_k", "decode_top_k_sent")):
        if r[col] is not None:
            sent.setdefault(k, float(r[col]))
    pols = conn.execute("SELECT p.*, (SELECT q.effective_from_utc FROM decode_policy q WHERE q.policy_id = p.superseded_by_policy_id) AS successor_from "
                        "FROM decode_policy p WHERE p.model_id = ? AND p.thinking_mode = ? AND (p.eval_id = ? OR (p.eval_id IS NULL AND p.eval_category = ?)) "
                        "ORDER BY p.policy_id", (r["model_id"], r["thinking_mode"], r["eval_id"], r["eval_category"])).fetchall()
    if not pols:
        same_model = conn.execute("SELECT COUNT(*) FROM decode_policy p JOIN eval e ON (p.eval_id = e.eval_id OR (p.eval_id IS NULL AND p.eval_category = e.eval_category)) "
                                  "WHERE p.model_id = ? AND e.eval_id = ?", (r["model_id"], r["eval_id"])).fetchone()[0]
        return [f"no_policy_in_scope (thinking_mode {r['thinking_mode']} differs from every row of the key)" if same_model
                else f"no_policy_in_scope (no row for {r['model_id']} x {r['eval_id']} / {r['eval_category']})"]
    out = []
    for p in pols:
        pid = p["policy_id"]
        if unmodelled:
            out.append(f"{pid}: run sent unmodelled key(s) {sorted(unmodelled)}")
            continue
        if p["seed_rule"] != "unverified" and p["seed_rule"] != r["decode_seed_rule"]:
            out.append(f"{pid}: seed_rule {r['decode_seed_rule']} != policy {p['seed_rule']}")
            continue
        if p["superseded_by_policy_id"] and p["successor_from"] not in (None, "unverified") and not (p["successor_from"] > r["generated"]):
            out.append(f"{pid}: superseded before the run (successor effective {p['successor_from'][:10]}, run generated {r['generated'][:10]})")
            continue
        not_sent = set((json.loads(p["params_not_sent_json"] or "{}") or {}).keys())
        pinned = {k: float(p[k]) for k in PARAMS if p[k] is not None}
        why = None
        for k, pv in pinned.items():
            if k in sent:
                if sent[k] != pv:
                    why = f"{k} sent {sent[k]:g} != policy {pv:g}"
                    break
            elif k not in not_sent:
                why = f"policy pins {k} {pv:g}, run did not send it"
                break
        if why is None:
            for k, sv in sent.items():
                if k not in pinned:
                    why = f"run sent {k} {sv:g}, policy leaves it unpinned"
                    break
        out.append(f"{pid}: {why or 'matches'}")
    return out


def distribution(conn: sqlite3.Connection, top: int = 5) -> Dict[str, object]:
    """The read-only report: counts per verdict and the top (campaign, reason) pairs behind ``no_match``."""
    source = f"({match_select_sql()})"
    rows = [dict(r) for r in conn.execute(f"SELECT r.run_id, r.campaign, v.policy_id, v.match, v.candidates_json, v.in_scope_json "
                                          f"FROM {source} v JOIN run r ON r.run_id = v.run_id").fetchall()]
    sql_counts = collections.Counter(row["match"] for row in rows)
    # the verdict the stamp pass writes: the SQL object's, with the Python tie-break applied to its `ambiguous` rows
    broken = tiebreak(conn, source)
    resolved: collections.Counter = collections.Counter()
    for row in rows:
        e = broken.get(row["run_id"])
        if e and e["policy_id"]:
            row["match"], row["policy_id"] = "exact", e["policy_id"]
            resolved[(row["campaign"], e["policy_id"], json.dumps(e["over"]))] += 1
    counts = collections.Counter(row["match"] for row in rows)
    by_policy = collections.Counter(row["policy_id"] for row in rows if row["match"] == "exact")
    reasons: collections.Counter = collections.Counter()
    ambiguous: collections.Counter = collections.Counter()
    for row in rows:
        if row["match"] == "no_match":
            for why in no_match_reasons(conn, row["run_id"]):
                # strip the policy id so identical reasons across policies of one key aggregate
                reasons[(row["campaign"], re.sub(r"^[^:]+: ", "", why))] += 1
        elif row["match"] == "ambiguous":
            ambiguous[(row["campaign"], row["candidates_json"])] += 1
    return {"counts": dict(counts), "sql_counts": dict(sql_counts), "n_runs": len(rows), "exact_by_policy": by_policy.most_common(),
            "no_match_top": reasons.most_common(top), "no_match_all": reasons.most_common(),
            "ambiguous": ambiguous.most_common(), "tiebreak_resolved": resolved.most_common()}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="READ-ONLY dry run of the decode-policy match over a registry database")
    ap.add_argument("--db", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rep = distribution(conn, a.top)
    finally:
        conn.close()
    if a.json:
        print(canonical_json(rep))
        return 0
    print(f"runs: {rep['n_runs']}  (stamped verdict; the SQL object's own count in brackets where the tie-break changed it)")
    for k in ("exact", "ambiguous", "no_match", "unrecorded"):
        n, m = rep["counts"].get(k, 0), rep["sql_counts"].get(k, 0)
        print(f"  {k:<11} {n}" + (f"  [{m}]" if n != m else ""))
    if rep["tiebreak_resolved"]:
        print("ties the Python pass resolves (campaign, winner, over):")
        for (camp, pid, over), n in rep["tiebreak_resolved"]:
            print(f"  {n:>5}  {camp}: {pid} over {over}")
    print("exact by policy:")
    for pid, n in rep["exact_by_policy"]:
        print(f"  {n:>5}  {pid}")
    print(f"top {a.top} (campaign, reason) behind no_match:")
    for (camp, why), n in rep["no_match_top"]:
        print(f"  {n:>5}  {camp}: {why}")
    if rep["ambiguous"]:
        print("still ambiguous after the tie-break (campaign, candidates):")
        for (camp, cands), n in rep["ambiguous"]:
            print(f"  {n:>5}  {camp}: {cands}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
