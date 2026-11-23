#!/usr/bin/env python3
"""Dry-read one bcbw adapter over a mirror root -- discover, group, read -- and write what it would emit, WITHOUT any ingest or database write.

The tool behind every "dry run over the live mirror" the bcbw readers' ledgers cite: per group a ``<out>/<group>.records.jsonl`` (one line per record,
``record_id`` dropped with ``--no-record-id`` so two trees' emissions compare as payloads) and ``<out>/<group>.status.json`` (the reader's ``status_extra``:
the rows, contrasts, refusals, parked stores, comparisons), plus ``<out>/summary.json``. Run it on a base tree and on a branch with the same root and diff
the directories: a solo reader untouched by a change must produce byte-identical files.

A contrast-filing reader asks the pass's anchor currency, which loads from a registry connection: with ``--db PATH`` the read gets a READ-ONLY connection
(``db.connect(PATH, readonly=True)``, URI ``mode=ro``: a write through it raises, and it never takes the harvester's lock), each group's read is preceded by
``ctx.begin_group(<group>)`` as the runner does, and that group's anchor decisions land in ``<out>/<group>.anchor_decisions.json``. Without ``--db`` the context
carries no connection, as before, and a group that files a contrast refuses at the currency by name. Against the LIVE database the two trees' runs must be
back to back (the harvester folds between runs, so a diff can be the database's, not the tree's): re-run both on a diff before reading it as the tree's.

    PYTHONPATH=src python scripts/registry/bcbw_dry_read.py --adapter bigcodebench_full_shared --root <mirror root> [--solo-root <solo mirror root>] [--db <registry.sqlite>] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="the adapter's registered name (bigcodebench_full, bfcl_full_shared, ...)")
    ap.add_argument("--root", required=True, help="the mirror root to read")
    ap.add_argument("--solo-root", default=None, help="a shared reader's solo mirror root (default: the reader's)")
    ap.add_argument("--out", required=True, help="the directory the records and status files are written under")
    ap.add_argument("--no-record-id", action="store_true", help="drop record_id from every record (it folds the record's paths, which differ between trees)")
    ap.add_argument("--db", default=None, help="a registry database opened READ-ONLY for the pass's anchor currency (default: no connection)")
    args = ap.parse_args(argv)
    from sisteer.registry import adapters as ADP
    from sisteer.registry import db as DB
    from sisteer.registry import validate as VAL
    from sisteer.registry.adapters import lineage_common as LC
    from sisteer.registry.adapters.base import AdapterContext
    reg = ADP.load_all()
    if args.adapter not in reg:
        print(f"no adapter {args.adapter!r}; registered: {', '.join(sorted(reg))}", file=sys.stderr)
        return 2
    cls = type(reg[args.adapter]())
    kw = {"root": args.root, "seeds": LC.Seeds.locate()}
    if args.solo_root:
        kw["solo_root"] = args.solo_root
    ad = cls(**kw)
    os.makedirs(args.out, exist_ok=True)
    conn = DB.connect(args.db, readonly=True) if args.db else None
    ctx = AdapterContext(root="", now="2026-01-01T00:00:00.000Z", producer={"script": "scripts/registry/bcbw_dry_read.py", "session": "dry-read", "host": "TESTHOST", "code_sha": "unverified",
                                                                            "substrate": "grading_host"}, conn=conn)
    inputs = ad.discover()
    groups: Dict[str, list] = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    summary: Dict[str, object] = {"adapter": args.adapter, "root": args.root, "solo_root": getattr(ad, "solo_root", None), "db": args.db, "n_inputs": len(inputs), "groups": {}}
    for g in sorted(groups):
        recs = []
        ctx.begin_group(g)          # the group's anchor decisions start empty, as the runner's read does
        for batch in ad.read(ctx, g, groups[g]):
            recs.extend(batch.records)
        if conn is not None:
            with open(os.path.join(args.out, f"{g}.anchor_decisions.json"), "w", encoding="utf-8") as fh:
                json.dump(list(ctx.decision_log), fh, indent=1, sort_keys=True, default=str)
        bad = [str(v) for v in (VAL.validate_record(r, path="<dry-read>") for r in recs) if isinstance(v, VAL.Malformed)]
        by_type: Dict[str, int] = {}
        for r in recs:
            by_type[r["record_type"]] = by_type.get(r["record_type"], 0) + 1
        with open(os.path.join(args.out, f"{g}.records.jsonl"), "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps({k: v for k, v in r.items() if not (args.no_record_id and k == "record_id")}, sort_keys=True, default=str) + "\n")
        facts = ad.status_extra(g)
        with open(os.path.join(args.out, f"{g}.status.json"), "w", encoding="utf-8") as fh:
            json.dump(facts, fh, indent=1, sort_keys=True, default=str)
        summary["groups"][g] = {"n_inputs": len(groups[g]), "n_records": len(recs), "by_type": by_type, "runs": facts.get("runs"), "n_refusals": facts.get("n_refusals"),
                                "n_not_run": facts.get("n_not_run"), "n_parked": facts.get("n_parked"), "malformed": bad[:5],
                                **({"n_anchor_decisions": len(ctx.decision_log)} if conn is not None else {})}
    states = [s.__dict__ for s in ad.source_states(ctx, inputs)]
    with open(os.path.join(args.out, "source_states.json"), "w", encoding="utf-8") as fh:
        json.dump(states, fh, indent=1, sort_keys=True, default=str)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True, default=str)
    if conn is not None:
        conn.close()
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
