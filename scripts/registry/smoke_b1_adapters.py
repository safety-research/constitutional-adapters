#!/usr/bin/env python
"""Live smoke for the lineage adapters, one per adapter.

Runs the named adapters over the REAL files into a temporary registry root through the core runner and
harvester, seeding STAND-IN dimension rows from the seed YAMLs and stand-in
object rows for every object sha the adapters resolved, then prints per
adapter: records written, rows per table, `unverified` field counts, refusals (row-level with reasons,
harvester refusals by reason) and blocked records. Counts are PRINTED, never asserted.

    PYTHONPATH=$PWD/src nice -n 10 .venv/bin/python scripts/registry/smoke_b1_adapters.py --adapter r2_export --keep /tmp/reg-b1

The seeds are read from this checkout's ``configs/registry/`` (``REGISTRY_SEEDS_DIR=<dir>`` names another directory).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
import time

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN

ADAPTERS = ("r2_export", "legacy_glm", "multiturn_grades", "petri_stats", "bon_stores", "degeneracy_union")


def _import_all():
    import importlib
    for n in ADAPTERS:
        try:
            importlib.import_module(f"sisteer.registry.adapters.{n}")
        except ModuleNotFoundError as exc:
            if n not in str(exc):
                raise


def _unverified_counts(conn):
    out = {}
    for t in ("run", "run_metric", "object"):
        rows = conn.execute(f"SELECT field_confidence_json FROM {t}").fetchall() if t != "run_metric" else []
        n_fields = 0
        by_field = collections.Counter()
        for (fc,) in rows:
            try:
                d = json.loads(fc or "{}")
            except json.JSONDecodeError:
                continue
            for k, v in d.items():
                if isinstance(v, dict) and v.get("confidence") == "unverified":
                    n_fields += 1
                    by_field[k] += 1
        out[t] = {"n_unverified_fields": n_fields, "by_field": dict(by_field.most_common(12))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", nargs="*", default=list(ADAPTERS))
    ap.add_argument("--keep", help="keep the temporary registry root at this path")
    ap.add_argument("--no-standins", action="store_true", help="do not seed stand-in dimension / object rows (everything blocks)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="keep an existing --keep root (re-ingest / re-read into it) instead of removing it")
    ap.add_argument("--full", action="store_true", help="re-read every comparison group even when its inputs are unchanged")
    args = ap.parse_args(argv)
    AB.install_store_audit_from_env()           # REGISTRY_STORE_AUDIT=<log>: record every store access (the refusal tests read it)
    _import_all()
    reg = AB.registered()
    root = args.keep or tempfile.mkdtemp(prefix="reg-b1-")
    if args.keep and os.path.isdir(args.keep) and not args.reuse:
        shutil.rmtree(args.keep)
    H.ensure_layout(root)
    seeds = LC.Seeds.locate()
    producer = dict(H.harvester_producer(root), script="scripts/registry/smoke_b1_adapters.py")
    now = R.utc_now()
    report = {"root": root, "adapters": {}}
    try:
        from sisteer.registry.adapters import dimensions as DIM   # the dimensions loader; absent on a checkout without it
    except ImportError:
        DIM = None
    if not args.no_standins:
        if DIM is not None:
            bundle, paths = DIM.write_seed_records(root, seeds.source_dir, producer=producer, now=now, verify_si=False)
            # the round row and the `none` object are not dimension seeds (rounds and objects are owned elsewhere): stand-ins for the smoke
            extra_rows = [r for r in LC.standin_dimension_records(seeds, producer, now) if r["record_type"] in ("paper_round", "object")]
            R.write_records(os.path.join(root, "records", "standins", "000001-round-and-none.jsonl"), extra_rows)
            rep0 = H.ingest_once(root, workers=0, min_free=0)
            report["dimensions"] = {"loader": "WS-B0 adapters/dimensions.py", "revision": bundle.revision_id, "records": sum(bundle.counts().values()),
                                    "files": len(paths), "ok": rep0.ok, "refused": rep0.refused, "blocked": rep0.blocked,
                                    "refusal_reasons": collections.Counter(r.split(": ", 1)[-1][:110] for r in rep0.refusals).most_common(15)}
        else:
            dims = LC.standin_dimension_records(seeds, producer, now)
            R.write_records(os.path.join(root, "records", "standins", "000001-dimensions.jsonl"), dims)
            rep0 = H.ingest_once(root, workers=0, min_free=0)
            report["standin_dimensions"] = {"records": len(dims), "ok": rep0.ok, "refused": rep0.refused, "blocked": rep0.blocked,
                                            "refusal_reasons": collections.Counter(r.split(": ", 1)[-1][:110] for r in rep0.refusals).most_common(15)}
    adapters = []
    for name in args.adapter:
        if name not in reg:
            print(f"adapter {name} not registered", file=sys.stderr)
            return 2
        adapters.append(reg[name]())
    t0 = time.monotonic()
    # the S2 dispatch-site guard: the pass itself requires a store access scope too; refusing here first names
    # this smoke, before the scratch root's database is opened
    try:
        AB.require_store_access("smoke_b1_adapters: the pass over " + ", ".join(args.adapter))
    except AB.GradingHostStoreReadRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 5
    conn = DB.connect(DB.db_path(root), readonly=True)
    pas = RUN.run_pass(root, adapters, conn=conn, now=R.utc_now(), producer=producer, full=args.full)
    conn.close()
    report["pass_s"] = round(time.monotonic() - t0, 1)
    # stand-in objects for every object the adapters resolved
    if not args.no_standins:
        objs = []
        seen = set()
        for ad in adapters:
            res = getattr(ad, "_objects", None) or getattr(ad, "objects", None)
            for rec_path in R.iter_record_files(os.path.join(root, "records", "adapters", ad.name)):
                kind = os.path.basename(rec_path).split("-", 1)[1]
                if not (kind.startswith("run.") or kind.startswith("disposition.")):
                    continue
                for _, rec, _ in R.read_record_file(rec_path):
                    p = rec["payload"]
                    oid = p.get("object_id")
                    if oid in seen or oid in (None, "none"):
                        continue
                    seen.add(oid)
                    view, cid = p.get("view_legacy"), p.get("campaign_cell_id_legacy")
                    if not (view and cid):
                        continue
                    ro = LC.ObjectResolver(seeds).resolve(view, cid, served_sha256=oid if not oid.startswith("si:") else None)
                    mv = seeds.model_for_view_legacy(view)
                    r = LC.standin_object_record(ro, seeds, producer, now, mv.get("fit_model_id") or mv.get("model_id"))
                    if r:
                        objs.append(r)
        if objs:
            R.write_records(os.path.join(root, "records", "standins", "000002-objects.jsonl"), objs)
        report["standin_objects"] = len(objs)
    t1 = time.monotonic()
    ing = H.ingest_once(root, workers=0, min_free=0)
    ing2 = H.ingest_once(root, workers=0, min_free=0)   # a second cycle lets late parents unblock children
    report["ingest_s"] = round(time.monotonic() - t1, 1)
    report["ingest"] = {"ok": ing.ok + ing2.ok, "refused": ing.refused + ing2.refused, "blocked": ing2.blocked, "unblocked": ing.unblocked + ing2.unblocked,
                        "duplicates": ing.duplicates + ing2.duplicates}
    reasons = collections.Counter()
    for r in ing.refusals + ing2.refusals:
        reasons[r.split(": ", 1)[-1][:140]] += 1
    report["ingest"]["refusal_reasons"] = reasons.most_common(25)
    conn = DB.connect(DB.db_path(root), readonly=True)
    for res in pas.results:
        ad = next(a for a in adapters if a.name == res.adapter)
        report["adapters"][res.adapter] = {
            "group": res.group, "accepted": res.accepted, "records_written": res.records_written, "error": res.error,
            "stats": getattr(ad, "stats", {}), "row_refusals": len(getattr(ad, "refusals", [])),
            "row_refusal_reasons": getattr(ad, "refusals", LC.RowRefusals(res.adapter)).summary(),
            "row_refusal_examples": [f"{it['key']} -- {it['why']}"[:220] for it in getattr(ad, "refusals", LC.RowRefusals(res.adapter)).items[:8]],
        }
    tables = {}
    for t in ("run", "judgment", "run_metric", "run_contrast", "run_degeneration", "item_result", "item_flags", "artifact", "disposition", "ruling",
              "computation", "source_revision", "observation", "withdrawal", "source_state", "blocked_records", "record_refusal"):
        tables[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    report["tables"] = tables
    report["runs_by_state"] = dict(conn.execute("SELECT lifecycle_state, COUNT(*) FROM run GROUP BY 1").fetchall())
    report["blocked_by_parent"] = dict(conn.execute("SELECT missing_parent_table || ':' || record_type, COUNT(*) FROM blocked_records GROUP BY 1").fetchall())
    report["blocked_parent_examples"] = [r[0] for r in conn.execute("SELECT DISTINCT missing_parent_table || '=' || missing_parent_id FROM blocked_records LIMIT 12").fetchall()]
    report["unverified"] = _unverified_counts(conn)
    report["degeneration_states"] = dict(conn.execute("SELECT measure || '/' || measurement_state, COUNT(*) FROM run_degeneration GROUP BY 1").fetchall())
    conn.close()
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(json.dumps(report, indent=1, default=str))
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
