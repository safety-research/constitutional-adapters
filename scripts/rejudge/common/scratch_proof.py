#!/usr/bin/env python3
"""Prove, on a SCRATCH copy of the registry, that filing a re-judge changes no existing row and no paper view.

    python scratch_proof.py digest --db SCRATCH/registry.sqlite --out before.json [--views v_paper_current ...]
    python scratch_proof.py digest --db SCRATCH/registry.sqlite --out after.json --new-judge JUDGE_ID [--new-judge ...]
    python scratch_proof.py compare before.json after.json

``digest`` reads every TABLE of the database (read-only) and, for each, the row count and an order-independent digest: the sum mod
2**64 of a 64-bit blake2b of each row's canonical text (``repr`` of the full column tuple, ``rowid`` excluded). With ``--new-judge`` it
also counts and digests the rows that BELONG to the filing -- a row naming one of the new judges or one of their judgments
(``judge_id`` / ``judgment_id`` / ``claimed_required_judge_id`` columns), or one of the filing's record files (a ``record_uri`` /
``file_uri`` / ``uri`` under ``records/rejudge/``, or a ``record_file_id`` of such a file) -- so ``compare`` can check, per table,

    count_after - count_new == count_before   and   digest_after - digest_new == digest_before   (mod 2**64)

i.e. the multiset of rows that are NOT the filing's is exactly the multiset before: nothing existing was updated, deleted, superseded
or re-pointed, whatever table it lives in -- except the ``run`` rows a judgment refolds, compared on content (``BOOKKEEPING``).
A view keyed by judge (``v_pending_judgment`` pairs every judge with every run whose instrument names one, so a new judge row adds
its own rows) is judged on the rows of the judges that existed before. Each paper view (and any view named with ``--views``) is digested whole and must be
IDENTICAL before and after in content (a paper view that gained the filing's rows would fail); a view named with ``--views`` (e.g.
``v_judge_side_by_side``, which SHOULD gain the new judge's rows) is reported, not judged. A table whose non-filing rows moved is listed with the
counts so it can be inspected (a full per-row diff: ``rowdiff``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

MASK = (1 << 64) - 1
#: bookkeeping columns: which record file last touched a row and that file's rank. A judgment record also refolds its RUN's key
#: (the reducer's ``judgment_state`` effect), and the refold re-points the run row's ``record_uri`` / ``ingest_seq`` at the newest event
#: -- the judgment's file -- without changing any content column; every view that selects ``ingest_seq`` shows the new rank
BOOKKEEPING = ("record_uri", "ingest_seq")
#: tables whose EXISTING rows a filing may re-point (bookkeeping only): compared on content over all rows, re-points counted
REPOINTABLE = ("run",)
PAPER_VIEWS = ("v_paper_current", "v_paper_long", "v_paper_objects", "v_roster_match", "v_run_current", "v_source_status")
EXTRA_VIEWS = ("v_pending_judgment",)


def h64(row: Sequence[Any]) -> int:
    return int.from_bytes(hashlib.blake2b(repr(tuple(row)).encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def ro(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)
    con.text_factory = lambda b: b.decode("utf-8", "surrogateescape")
    return con


def columns(con: sqlite3.Connection, name: str) -> List[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_xinfo("{name}")') if r[6] in (0, 2, 3)] or [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]


def digest_rows(rows: Iterable[Sequence[Any]], content_idx: Optional[Sequence[int]] = None) -> Dict[str, int]:
    n, s, c = 0, 0, 0
    for r in rows:
        n += 1
        s = (s + h64(r)) & MASK
        if content_idx is not None:
            c = (c + h64([r[i] for i in content_idx])) & MASK
    out = {"count": n, "digest": s}
    if content_idx is not None:
        out["content_digest"] = c
    return out


def content_index(cols: Sequence[str]) -> List[int]:
    return [i for i, c in enumerate(cols) if c not in BOOKKEEPING and not c.endswith("ingest_seq")]


def new_row_predicate(cols: Sequence[str], judges: Sequence[str], new_file_ids: Sequence[int], fact_uris: Sequence[str] = ()):
    """A per-row test: does this row belong to the filing? ``fact_uris``: the harvester's own fact files ABOUT the filing's files
    (``file_arrival`` facts), listed after reading them."""
    facts = set(fact_uris)
    js = set(judges)
    idx = {c: i for i, c in enumerate(cols)}
    fid = set(new_file_ids)
    jid_cols = [idx[c] for c in ("judge_id", "claimed_required_judge_id") if c in idx]
    uri_cols = [idx[c] for c in ("record_uri", "file_uri", "uri") if c in idx]
    fid_col = idx.get("record_file_id")
    jm_col = idx.get("judgment_id")
    new_judgments: set = set()

    def pred(row) -> bool:
        for i in jid_cols:
            if row[i] in js:
                return True
        for i in uri_cols:
            v = row[i]
            if isinstance(v, str) and (v.startswith("records/rejudge/") or v in facts):
                return True
        if fid_col is not None and row[fid_col] in fid:
            return True
        if jm_col is not None and row[jm_col] in new_judgments:
            return True
        return False
    return pred, new_judgments


def cmd_digest(args) -> int:
    con = ro(args.db)
    t0 = time.time()
    out: Dict[str, Any] = {"db": args.db, "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tables": {}, "views": {},
                           "new_judges": args.new_judge or []}
    new_jm: List[str] = []
    new_fids: List[int] = []
    if args.new_judge:
        marks = ",".join("?" * len(args.new_judge))
        new_jm = [r[0] for r in con.execute(f"SELECT judgment_id FROM judgment WHERE judge_id IN ({marks})", args.new_judge)]
        facts = list(args.fact_uri or [])
        new_fids = [r[0] for r in con.execute("SELECT record_file_id FROM record_ingest WHERE file_uri LIKE 'records/rejudge/%'")]
        if facts:
            marks2 = ",".join("?" * len(facts))
            new_fids += [r[0] for r in con.execute(f"SELECT record_file_id FROM record_ingest WHERE file_uri IN ({marks2})", facts)]
        out["new_judgments"] = len(new_jm)
        out["new_record_files"] = len(new_fids)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in ([] if args.views_only else tables):
        cols = columns(con, t)
        sel = ", ".join(f'"{c}"' for c in cols)
        t1 = time.time()
        ci = content_index(cols)
        if args.new_judge:
            pred, nj = new_row_predicate(cols, args.new_judge, new_fids, args.fact_uri or [])
            nj.update(new_jm)
            n = s = c = nn = sn = 0
            repointed = 0
            for row in con.execute(f'SELECT {sel} FROM "{t}"'):
                h = h64(row)
                n += 1
                s = (s + h) & MASK
                c = (c + h64([row[i] for i in ci])) & MASK
                if pred(row):
                    if t in REPOINTABLE:
                        repointed += 1
                    else:
                        nn += 1
                        sn = (sn + h) & MASK
            out["tables"][t] = {"count": n, "digest": s, "content_digest": c, "new_count": nn, "new_digest": sn, "repointed": repointed}
        else:
            out["tables"][t] = digest_rows(con.execute(f'SELECT {sel} FROM "{t}"'), ci)
        print(f"[digest] {t}: {out['tables'][t]['count']} rows" + (f", {out['tables'][t].get('new_count')} the filing's" if args.new_judge else "")
              + f" ({time.time() - t1:.0f}s)", flush=True)
    for v in list(PAPER_VIEWS) + list(EXTRA_VIEWS) + list(args.views or []):
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = ?", (v,)).fetchone():
            out["views"][v] = {"missing": True}
            continue
        t1 = time.time()
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{v}")')]
        ci = content_index(cols)
        jcols = [i for i, c in enumerate(cols) if c in ("required_judge_id", "judge_id")]
        nj = set(args.new_judge or [])
        n = s = c = ne = ce = 0
        for row in con.execute(f'SELECT * FROM "{v}"'):
            n += 1
            s = (s + h64(row)) & MASK
            hc = h64([row[i] for i in ci])
            c = (c + hc) & MASK
            if not any(row[i] in nj for i in jcols):
                ne += 1
                ce = (ce + hc) & MASK
        # rows of a view keyed by judge (a worklist pairs EVERY judge with every run whose instrument names one) are compared on the
        # existing judges' rows: a new judge adds its own rows by the view's design, and none of the rows already there may move
        out["views"][v] = {"count": n, "digest": s, "content_digest": c, "existing_judges_count": ne, "existing_judges_content_digest": ce,
                           "judge_keyed": bool(jcols)}
        out["views"][v]["columns"] = len(cols)
        print(f"[digest] view {v}: {out['views'][v]['count']} rows ({time.time() - t1:.0f}s)", flush=True)
    out["secs"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


def cmd_compare(args) -> int:
    b = json.load(open(args.before))
    a = json.load(open(args.after))
    if args.after_views:
        a["views"] = json.load(open(args.after_views))["views"]
    problems: List[str] = []
    rows = []
    for t, tb in sorted(b["tables"].items()):
        ta = a["tables"].get(t)
        if ta is None:
            problems.append(f"table {t} vanished")
            continue
        nn, sn = ta.get("new_count", 0), ta.get("new_digest", 0)
        # a FOLLOW-UP proof's before digest already holds an earlier filing's rows (digested with the same --new-judge): those are
        # the filing's too, and only the rows that are nobody's filing must be the same multiset (0 for a first filing's before)
        nb, sb = tb.get("new_count", 0), tb.get("new_digest", 0)
        if t in REPOINTABLE:
            # existing rows may be re-pointed (bookkeeping columns only): same rows, same CONTENT
            same = ta["count"] == tb["count"] and ta.get("content_digest") == tb.get("content_digest")
            bookkeeping_only = same and ta["digest"] != tb["digest"]
        else:
            same = (ta["count"] - nn == tb["count"] - nb) and ((ta["digest"] - sn) & MASK) == ((tb["digest"] - sb) & MASK)
            bookkeeping_only = False
        rows.append({"table": t, "before": tb["count"], "after": ta["count"], "filing_rows": nn, "filing_rows_before": nb,
                     "existing_rows_unchanged": same, "repointed_rows": ta.get("repointed", 0), "bookkeeping_changed": bookkeeping_only})
        if not same:
            problems.append(f"table {t}: rows other than the filing's moved (before {tb['count']}, after {ta['count']}, filing {nn})")
    for t in sorted(set(a["tables"]) - set(b["tables"])):
        problems.append(f"table {t} appeared")
    views = []
    for v, vb in sorted(b["views"].items()):
        va = a["views"].get(v, {})
        same = vb.get("count") == va.get("count") and vb.get("digest") == va.get("digest")
        content_same = vb.get("count") == va.get("count") and vb.get("content_digest") == va.get("content_digest")
        existing_same = (va.get("existing_judges_count", va.get("count")) == vb.get("existing_judges_count", vb.get("count"))
                         and va.get("existing_judges_content_digest", va.get("content_digest"))
                         == vb.get("existing_judges_content_digest", vb.get("content_digest")))
        informational = v not in PAPER_VIEWS and v not in EXTRA_VIEWS
        views.append({"view": v, "before": vb.get("count"), "after": va.get("count"), "identical": same,
                      "identical_except_ingest_seq_record_uri": content_same, "existing_judges_rows_unchanged": existing_same,
                      "new_judge_rows": (va.get("count") or 0) - (va.get("existing_judges_count", va.get("count")) or 0),
                      "new_judge_rows_before": (vb.get("count") or 0) - (vb.get("existing_judges_count", vb.get("count")) or 0),
                      "informational": informational})
        if not existing_same and not informational:
            problems.append(f"view {v} differs in content (before {vb.get('count')} rows, after {va.get('count')})")
    rep = {"before": args.before, "after": args.after, "tables": rows, "views": views, "problems": problems,
           "verdict": "PASS" if not problems else "FAIL"}
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
    for r in rows:
        if r["filing_rows"] or not r["existing_rows_unchanged"]:
            print(f"  {r['table']:32s} before {r['before']:>10} after {r['after']:>10} filing {r['filing_rows']:>8} existing unchanged: {r['existing_rows_unchanged']}")
        elif r["repointed_rows"] or r["bookkeeping_changed"]:
            print(f"  {r['table']:32s} before {r['before']:>10} after {r['after']:>10} re-pointed (record_uri/ingest_seq only) {r['repointed_rows']}, content unchanged: {r['existing_rows_unchanged']}")
    print(f"  {sum(1 for r in rows if r['existing_rows_unchanged'])}/{len(rows)} tables: every pre-existing row unchanged")
    for v in views:
        print(f"  view {v['view']:24s} {v['before']} -> {v['after']} identical: {v['identical']}, identical except ingest_seq/record_uri: "
              f"{v['identical_except_ingest_seq_record_uri']}, existing judges' rows unchanged: {v['existing_judges_rows_unchanged']} "
              f"(+{v['new_judge_rows']} rows of the new judges)" + (" [informational]" if v["informational"] else ""))
    print(rep["verdict"], *problems, sep="\n  ")
    return 0 if not problems else 1


def cmd_rowdiff(args) -> int:
    """Full per-row diff of ONE table between two databases (rows only in one side)."""
    cb, ca = ro(args.before_db), ro(args.after_db)
    cols = columns(cb, args.table)
    sel = ", ".join(f'"{c}"' for c in cols)
    hb = {h64(r): r for r in cb.execute(f'SELECT {sel} FROM "{args.table}"')}
    ha = {h64(r): r for r in ca.execute(f'SELECT {sel} FROM "{args.table}"')}
    only_b = [hb[k] for k in hb.keys() - ha.keys()]
    only_a = [ha[k] for k in ha.keys() - hb.keys()]
    print(json.dumps({"columns": cols, "only_before": [list(map(str, r)) for r in only_b[:args.limit]], "n_only_before": len(only_b),
                      "only_after": [list(map(str, r)) for r in only_a[:args.limit]], "n_only_after": len(only_a)}, indent=1)[:20000])
    return 0


def cmd_viewdiff(args) -> int:
    """Per-key column diff of one view between two databases, the new judges' rows left out: which columns of the rows that existed
    before moved, how often, an example of each, and -- for a JSON list column of judgments -- whether the pre-existing entries are
    unchanged (only the new judges' entries added)."""
    nj = set(args.new_judge or [])
    keys = args.key.split(",")

    def load(db):
        cur = ro(db).execute(f'SELECT * FROM "{args.view}"')
        cols = [d[0] for d in cur.description]
        out = {}
        for r in cur:
            d = dict(zip(cols, r))
            if any(d.get(c) in nj for c in ("required_judge_id", "judge_id")):
                continue
            out[tuple(d[k] for k in keys)] = d
        return cols, out
    cols, b = load(args.before_db)
    _, a = load(args.after_db)
    changed: Dict[str, int] = {}
    examples: Dict[str, Any] = {}
    only_new_entries = True
    for k, rb in b.items():
        ra = a.get(k)
        if ra is None:
            continue
        for c in cols:
            if c in (args.ignore or []) or rb[c] == ra[c]:
                continue
            changed[c] = changed.get(c, 0) + 1
            examples.setdefault(c, {"key": list(map(str, k)), "before": str(rb[c])[:300], "after": str(ra[c])[:300]})
            if c.endswith("_json"):
                try:
                    lb, la = json.loads(rb[c]), json.loads(ra[c])
                    if isinstance(lb, list) and isinstance(la, list):
                        if [x for x in la if not (isinstance(x, dict) and x.get("judge_id") in nj)] != lb:
                            only_new_entries = False
                except ValueError:
                    pass
    rep = {"view": args.view, "existing_rows_before": len(b), "existing_rows_after": len(a), "same_keys": set(a) == set(b),
           "changed_columns": changed, "examples": examples, "json_lists_only_gained_new_judges_entries": only_new_entries}
    print(json.dumps(rep, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digest")
    d.add_argument("--db", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--new-judge", action="append")
    d.add_argument("--fact-uri", action="append", help="a harvester fact file about the filing's files (exact logical uri)")
    d.add_argument("--views", nargs="*")
    d.add_argument("--views-only", action="store_true", help="digest the views only (no table scan)")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--out")
    c.add_argument("--after-views", help="a views-only digest of the after database that replaces after's views")
    r = sub.add_parser("rowdiff")
    r.add_argument("--before-db", required=True)
    r.add_argument("--after-db", required=True)
    r.add_argument("--table", required=True)
    r.add_argument("--limit", type=int, default=20)
    v = sub.add_parser("viewdiff")
    v.add_argument("--before-db", required=True)
    v.add_argument("--after-db", required=True)
    v.add_argument("--view", required=True)
    v.add_argument("--key", required=True, help="comma-separated key columns, e.g. run_id,required_judge_id")
    v.add_argument("--new-judge", action="append")
    v.add_argument("--ignore", nargs="*", default=["claim_age_s"], help="columns that move with the clock")
    args = ap.parse_args(argv)
    return {"digest": cmd_digest, "compare": cmd_compare, "rowdiff": cmd_rowdiff, "viewdiff": cmd_viewdiff}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
