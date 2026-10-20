"""``registry refusals list | explain | summary``: the refusal FACTS read back, read-only.

A refusal is a judgment the harvester made about one record under one judging stamp. ``record_refusal`` keeps the CURRENT judgment per
record (a re-attempt under a changed stamp replaces it); every judgment ever made is a ``record_refusal_fact`` row, folded from the
``record_refusal`` records the harvester writes into the log (``harvest.derive_refusal_fact_records``), so the facts survive a rebuild and
travel with every mirror. This module reads them:

* :func:`list_facts` -- the facts, newest cycle first, each beside the record's current judgment (``still_refused`` = this fact IS the
  current judgment; ``--still-refused`` keeps those alone);
* :func:`explain` -- one record (by sha, or by the uri it was refused under): its facts per stamp, the refused bytes re-read from the
  file by sha, the CURRENT validator's verdict on them (admitted, or refused naming the field -- a landed fix lifts a refusal without
  touching the log), and the RESOLUTION at the record's scale: run-scale through a ``row_history`` primary-key seek and the live row
  (a LATER row folded at the key, an earlier row stands, or no row), item-scale through the item table by natural key (the live row is
  the greatest order key), ``unknown`` when the bytes are not a record whose key parses;
* :func:`summary` -- the counts: facts, records, per stamp, per record kind, per field, backfilled, re-judged, current refusals.

Writes nothing: the CLI resolves the root without the ``state/last_run`` stamp.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from . import harvest as H
from . import records as R
from . import reducer as RED
from . import schema_def as S
from . import validate as VAL
from .identity import canonical_json

DEFAULT_LIMIT = 200
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: What the resolution of a refused record can read (``explain``): the key's row folded from a LATER record (the correction landed); an
#: EARLIER record's row stands (nothing newer at the key); no row at the key at all; or the bytes are not a record whose key parses.
RESOLUTIONS = ("later_row_folded", "earlier_row_stands", "no_row", "unknown")

_FACT_SQL = ("SELECT f.record_sha256, f.record_uri, f.record_type, f.field, f.reason, f.judged_with, f.refused_utc, f.cycle_utc, "
             "f.producer_json, f.extra_json, r.judged_with AS current_judged_with, r.refused_utc AS current_refused_utc, r.reason AS current_reason, "
             "CASE WHEN r.record_sha256 IS NOT NULL AND COALESCE(r.judged_with, 'unverified') = f.judged_with THEN 1 ELSE 0 END AS still_refused "
             "FROM record_refusal_fact f LEFT JOIN record_refusal r ON r.record_sha256 = f.record_sha256")


def list_facts(conn: sqlite3.Connection, still_refused: bool = False, record_type: Optional[str] = None,
               limit: Optional[int] = DEFAULT_LIMIT) -> List[Dict[str, object]]:
    """The facts, newest derivation first, each beside the record's current judgment."""
    where: List[str] = []
    params: List[object] = []
    if still_refused:
        where.append("r.record_sha256 IS NOT NULL AND COALESCE(r.judged_with, 'unverified') = f.judged_with")
    if record_type:
        where.append("f.record_type = ?")
        params.append(record_type)
    sql = _FACT_SQL + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY f.cycle_utc DESC, f.refused_utc DESC, f.record_sha256"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def summary(conn: sqlite3.Connection) -> Dict[str, object]:
    """The counts over the facts and the current judgments."""
    one = lambda sql, *p: conn.execute(sql, p).fetchone()[0]  # noqa: E731
    by = lambda sql: {str(r[0]): int(r[1]) for r in conn.execute(sql).fetchall()}  # noqa: E731
    return {
        "facts": int(one("SELECT COUNT(*) FROM record_refusal_fact")),
        "records": int(one("SELECT COUNT(DISTINCT record_sha256) FROM record_refusal_fact")),
        "current_refusals": int(one("SELECT COUNT(*) FROM record_refusal")),
        "current_without_fact": int(one("SELECT COUNT(*) FROM record_refusal r WHERE NOT EXISTS (SELECT 1 FROM record_refusal_fact f "
                                        "WHERE f.record_sha256 = r.record_sha256 AND f.judged_with = COALESCE(r.judged_with, 'unverified'))")),
        "backfilled": int(one("SELECT COUNT(*) FROM record_refusal_fact WHERE json_extract(extra_json, '$.backfilled') = 1")),
        "re_judged_records": int(one("SELECT COUNT(*) FROM (SELECT record_sha256 FROM record_refusal_fact GROUP BY record_sha256 HAVING COUNT(*) > 1)")),
        "by_judged_with": by("SELECT judged_with, COUNT(*) FROM record_refusal_fact GROUP BY judged_with ORDER BY judged_with"),
        "by_record_type": by("SELECT COALESCE(record_type, '(no readable type)'), COUNT(*) FROM record_refusal_fact GROUP BY 1 ORDER BY 1"),
        "by_field": by("SELECT COALESCE(field, '(envelope)'), COUNT(*) FROM record_refusal_fact GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20"),
        "latest_cycle_utc": one("SELECT MAX(cycle_utc) FROM record_refusal_fact"),
        "earliest_refused_utc": one("SELECT MIN(refused_utc) FROM record_refusal_fact"),
    }


# --------------------------------------------------------------------------------------------
# explain: the facts, the bytes, the current verdict, the resolution
# --------------------------------------------------------------------------------------------


def read_refused_record(root: str, record_uri: str, sha: str) -> Optional[Dict[str, object]]:
    """The refused record itself, re-read from its file by sha (the line a ``<file>#<line>`` uri names, else a bounded scan), or None."""
    base, _, frag = str(record_uri).partition("#")
    path = H.resolve_uri(root, base)
    if path is None:
        return None
    try:
        if frag.isdigit():
            want = int(frag)
            with R.open_record_file(path) as fh:
                for i, line in enumerate(fh, start=1):
                    if i == want:
                        line = line.strip()
                        if not line:
                            return None
                        rec = json.loads(line)
                        return rec if isinstance(rec, dict) and R.record_sha(rec) == sha else None
            return None
        seen = 0
        for _, rec, rsha in R.read_record_file(path):
            seen += 1
            if rsha == sha:
                return rec
            if seen >= 200_000:
                return None
    except (OSError, ValueError, EOFError, UnicodeDecodeError):
        return None
    return None


def _key_for(record: Dict[str, object]) -> Tuple[Optional[str], Optional[Dict[str, object]]]:
    """``(table, natural key)`` of the row the record would fold, or ``(None, None)`` when no key parses."""
    rt = record.get("record_type")
    payload = record.get("payload")
    if not isinstance(payload, dict) or rt not in R.RECORD_TYPES:
        return None, None
    try:
        if rt in R.RECORD_TYPE_TABLE:
            table = R.RECORD_TYPE_TABLE[rt]
            return table, RED._key_of(table, payload)
        if rt in ("run_patch", "run_ingest") and payload.get("run_id"):
            return "run", {"run_id": str(payload["run_id"])}
    except KeyError:
        return None, None
    return None, None


def resolve(conn: sqlite3.Connection, record: Optional[Dict[str, object]], sha: str) -> Dict[str, object]:
    """Where the refused record's KEY stands now (:data:`RESOLUTIONS`), at the record's scale."""
    if record is None:
        return {"state": "unknown", "why": "the refused bytes could not be re-read from the file by sha"}
    table, key = _key_for(record)
    written = record.get("written_utc")
    if table is None or key is None or not isinstance(written, str):
        return {"state": "unknown", "why": "the bytes are not a record whose natural key parses (no record_type / payload / written_utc)"}
    out: Dict[str, object] = {"table": table, "natural_key": key, "refused_order_key": [written, sha]}
    if table in S.RUN_SCALE_TABLES:
        key_json = canonical_json(key)
        # the fold's own order (reducer.ORDER_SQL): at one stamp a run_state to a lateral / terminal state ranks
        # AFTER a progression event whatever the shas, so the refused record's place among the key's events -- "later" or not -- is read
        # by the same three-part key, its rank from its own bytes the way TIE_RANK_SQL reads the folded events' from row_history
        payload = record.get("payload")
        rank = RED.tie_rank(str(record.get("record_type")), payload if isinstance(payload, dict) else {})
        out["refused_order_key"] = [written, rank, sha]
        events = conn.execute(f"SELECT written_utc, record_sha256, record_type, record_uri, {RED.TIE_RANK_SQL} FROM row_history "
                              f"WHERE table_name = ? AND natural_key_json = ? ORDER BY {RED.ORDER_SQL}", (table, key_json)).fetchall()
        later = [e for e in events if (str(e[0]), int(e[4]), str(e[1])) > (written, rank, sha)]
        where = " AND ".join(f"{c} = ?" for c in key)
        live = conn.execute(f"SELECT 1 FROM {table} WHERE {where}", tuple(key.values())).fetchone() is not None
        out.update(scale="run", live_row=live, events=len(events), later_events=len(later),
                   latest_event=({"written_utc": events[-1][0], "record_sha256": events[-1][1], "record_type": events[-1][2], "record_uri": events[-1][3]}
                                 if events else None),
                   state="later_row_folded" if later else ("earlier_row_stands" if events else "no_row"))
        return out
    if table in S.ITEM_SCALE_TABLES:
        full = {c: key.get(c, S.SENTINELS.get(c)) for c in R.NATURAL_KEYS[table]}
        where = " AND ".join(f"{c} = ?" for c in full)
        row = conn.execute(f"SELECT written_utc, record_sha256 FROM {table} WHERE {where}", tuple(full.values())).fetchone()
        out.update(scale="item", live_row=row is not None)
        if row is None:
            out["state"] = "no_row"
        else:
            live_key = (R.epoch_ms_to_utc(int(row[0])), bytes(row[1]).hex())
            out["live_order_key"] = list(live_key)
            out["state"] = "later_row_folded" if live_key > (written, sha) else "earlier_row_stands"
        return out
    out.update(scale="ledger", state="unknown", why="a ledger-scale fact has no folded key to resolve against")
    return out


def current_verdict(record: Optional[Dict[str, object]], sha: str) -> Dict[str, object]:
    """The CURRENT validator's judgment on the refused bytes (pure; the fold-time checks -- dependencies, verify, FK -- are not re-run)."""
    if record is None:
        return {"stamp": H.validator_stamp(), "verdict": "unknown", "why": "the refused bytes could not be re-read"}
    res = VAL.validate_record(record, "<refusals explain>", sha)
    if isinstance(res, VAL.Malformed):
        return {"stamp": H.validator_stamp(), "verdict": "refused", "field": res.field, "reason": res.reason}
    return {"stamp": H.validator_stamp(), "verdict": "admitted", "note": "the pure validator admits these bytes under the current code; a fold-time "
                                                                          "refusal (a verify mismatch, a foreign key, a cross-run successor) is not re-run here"}


def explain(root: str, conn: sqlite3.Connection, ident: str) -> Dict[str, object]:
    """One record by sha (64 hex) or by the uri it was refused under: its facts per stamp, the current judgment, the bytes, the current
    verdict and the resolution. A uri naming several refused records (a file with no line) explains each."""
    if _SHA256_RE.match(ident):
        shas = [ident]
    else:
        shas = [str(r[0]) for r in conn.execute("SELECT DISTINCT record_sha256 FROM (SELECT record_sha256, record_uri FROM record_refusal_fact UNION "
                                                "SELECT record_sha256, record_uri FROM record_refusal) WHERE record_uri = ? OR record_uri LIKE ? "
                                                "ORDER BY record_sha256", (ident, ident + "#%")).fetchall()]
    out: Dict[str, object] = {"ident": ident, "records": []}
    for sha in shas:
        facts = [dict(r) for r in conn.execute(_FACT_SQL + " WHERE f.record_sha256 = ? ORDER BY f.refused_utc, f.judged_with", (sha,)).fetchall()]
        cur = conn.execute("SELECT record_uri, record_type, field, reason, refused_utc, judged_with FROM record_refusal WHERE record_sha256 = ?", (sha,)).fetchone()
        uri = str(cur[0]) if cur else (str(facts[0]["record_uri"]) if facts else None)
        record = read_refused_record(root, uri, sha) if uri else None
        out["records"].append({"record_sha256": sha, "record_uri": uri, "current_judgment": dict(cur) if cur else None, "facts": facts,
                               "record": record, "current_verdict": current_verdict(record, sha), "resolution": resolve(conn, record, sha)})
    if not shas:
        out["note"] = "no refusal fact or current refusal names this sha / uri"
    return out


# --------------------------------------------------------------------------------------------
# text rendering
# --------------------------------------------------------------------------------------------


def list_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    out = [f"{len(rows)} refusal fact(s), newest derivation first"]
    for r in rows:
        tag = "still refused" if r.get("still_refused") else f"re-judged under {r.get('current_judged_with') or 'unverified'}" if r.get("current_judged_with") is not None or r.get("current_refused_utc") else "no current refusal"
        out.append(f"{r['refused_utc']}  {r['judged_with']}  {r.get('record_type') or '-'}  {r.get('field') or '-'}  {r['record_uri']}  [{tag}]  {str(r['reason'])[:160]}")
    return out


def summary_lines(rep: Dict[str, object]) -> List[str]:
    out = [f"refusal facts {rep['facts']} over {rep['records']} record(s); current refusals {rep['current_refusals']} "
           f"({rep['current_without_fact']} without a fact yet); backfilled {rep['backfilled']}; re-judged records {rep['re_judged_records']}",
           f"earliest refusal {rep['earliest_refused_utc']}; latest fact derivation {rep['latest_cycle_utc']}"]
    out.append("by judging stamp: " + (", ".join(f"{k}={v}" for k, v in rep["by_judged_with"].items()) or "none"))
    out.append("by record kind: " + (", ".join(f"{k}={v}" for k, v in rep["by_record_type"].items()) or "none"))
    out.append("by field: " + (", ".join(f"{k}={v}" for k, v in rep["by_field"].items()) or "none"))
    return out


def explain_lines(rep: Dict[str, object]) -> List[str]:
    out = [f"explain {rep['ident']}: {len(rep['records'])} record(s)"]
    if rep.get("note"):
        out.append(rep["note"])
    for e in rep["records"]:
        out.append(f"record {e['record_sha256']} refused under {e['record_uri']}")
        cur = e["current_judgment"]
        out.append("  current judgment: " + (f"{cur['refused_utc']} stamp {cur['judged_with'] or 'unverified'} field {cur['field'] or '-'}: {cur['reason']}" if cur else "none (no current refusal row)"))
        for f in e["facts"]:
            out.append(f"  fact: refused {f['refused_utc']} under stamp {f['judged_with']} (derived {f['cycle_utc']}{', backfilled' if json.loads(f['extra_json'] or '{}').get('backfilled') else ''}): {f['reason']}")
        v = e["current_verdict"]
        out.append(f"  current validator ({v['stamp']}): {v['verdict']}" + (f" -- {v.get('field')}: {v.get('reason')}" if v["verdict"] == "refused" else f" -- {v.get('note') or v.get('why')}"))
        res = e["resolution"]
        out.append(f"  resolution: {res['state']}" + (f" ({res.get('scale')} scale, {res.get('table')} {json.dumps(res.get('natural_key'))}; live row {res.get('live_row')}"
                                                     + (f"; latest event {res['latest_event']['written_utc']} {res['latest_event']['record_type']} {res['latest_event']['record_uri']}" if res.get("latest_event") else "")
                                                     + ")" if res.get("table") else f" ({res.get('why')})"))
    return out
