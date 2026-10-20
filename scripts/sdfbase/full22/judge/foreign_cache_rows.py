#!/usr/bin/env python
"""Record every PRODUCER-STAMPED foreign row in a judge cache — a verdict another lineage filed under a claude-sonnet-5
protocol key through a fallback backend (grok-4.6 via OpenRouter, 450 rows in the shared cache on 2026-09-09) — and which
stores reference each one, so a later reader can detect them without re-deriving the diagnosis.

Why the key cannot tell them apart: the cache key is (prompt, generation, rubric, protocol, judge_model) and the fallback
writer used the sonnet-5 key; `judge_items.judge_model` names the PROTOCOL's judge, never the producer. The producer is
visible only in the row itself (`org_used='openrouter'`, `verdict_json.fallback=true` + judge_backend + judge_model), and in
a filed row only as `cache_org` / `verdict_extra_keys`. The served judge now REFUSES such a row at the read (evicts it from
the private cache and re-judges live) and `seed_private_cache.py` scrubs them globally; this script is the detector.

    foreign_cache_rows.py [--cache SHARED.sqlite] [--roots DIR,...] --out RECORD.json [--md RECORD.md]

Referencing stores are found two ways: (1) every judge_items.jsonl under the roots (request_key match — exact for judged
stores); (2) for stores without judge_items, the request keys recomputed with the judge's own Cell.request_keys (PYTHONPATH=src:scripts).
"""
import argparse, collections, glob, json, sqlite3, sys, time
from pathlib import Path

ANTHROPIC_ORGS = {"lowprio", "highprio", "batch"}


def foreign_rows(cache: Path):
    c = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
    out = {}
    for key, vj, org, created in c.execute("SELECT key, verdict_json, org_used, created_at FROM verdicts"):
        try: v = json.loads(vj)
        except Exception: v = {"_unparseable": vj[:80]}
        if v.get("fallback") or (org is not None and org not in ANTHROPIC_ORGS) or "_unparseable" in v:
            out[key] = {"verdict": v, "org_used": org, "created_at": created,
                        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)) if created else None}
    c.close()
    return out


def refs_from_judge_items(roots, keys):
    refs = collections.defaultdict(list); scanned = 0
    for root in roots:
        for ji in glob.glob(str(Path(root).expanduser() / "*/cells/*/judge_items.jsonl")):
            scanned += 1
            lane = ji.split("/")[-4]; cell = ji.split("/")[-2]
            for l in open(ji):
                if not l.strip(): continue
                r = json.loads(l)
                if r.get("request_key") in keys:
                    refs[r["request_key"]].append({"root": str(root), "lane": lane, "cell": cell, "control": r.get("control"), "item_id": r.get("item_id"),
                                                   "filed_status": r.get("status"), "filed_label": r.get("label"), "filed_cache_org": r.get("cache_org"),
                                                   "filed_from_foreign_row": r.get("cache_org") == "openrouter" or bool(r.get("verdict_extra_keys"))})
    return refs, scanned


def refs_from_stores(roots, keys, already):
    """Stores with no judge_items: recompute the request keys with the judge's own Cell (best effort)."""
    refs = collections.defaultdict(list); n = 0; errs = []
    try:
        sys.path[:0] = ["src", "scripts"]
        import served_capability_judge as sj
        from sisteer.capeval import corpora
        from sisteer.capeval.tiers import judged
        pins = corpora.load_pins(); rows = {"xstest": judged.xstest_rows(pins), "or_bench_hard_1k": judged.or_bench_rows(pins)}
    except Exception as e:  # noqa: BLE001
        return refs, n, [f"harness import failed: {type(e).__name__}: {e}"[:200]]
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists(): continue
        targets = sorted(p.name for p in root.iterdir() if (p / "cells").is_dir())
        try:
            cells = sj.discover(root, targets, "", rows, list(sj.JUDGED_CONTROLS))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{root}: discover failed: {type(e).__name__}: {e}"[:200]); continue
        for c in cells:
            if (c.cell_dir / "judge_items.jsonl").exists(): continue
            n += 1
            for control in c.controls:
                try: rk = c.request_keys(control)
                except Exception as e:  # noqa: BLE001
                    errs.append(f"{c.target}/{c.cell_id}/{control}: {type(e).__name__}"[:200]); continue
                for iid, k in rk.items():
                    if k in keys:
                        refs[k].append({"root": str(root), "lane": c.target, "cell": c.name, "control": control, "item_id": iid,
                                        "filed_status": "unjudged store (key recomputed)", "filed_from_foreign_row": False})
    return refs, n, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / ".local/state/capeval/judge_cache.sqlite"))
    ap.add_argument("--roots", default="", help="comma list of controls roots (<root>/<lane>/cells/<store>)")
    ap.add_argument("--out", required=True); ap.add_argument("--md", default=None)
    a = ap.parse_args()
    cache = Path(a.cache).expanduser(); roots = [r for r in a.roots.split(",") if r]
    fr = foreign_rows(cache)
    refs, n_ji = refs_from_judge_items(roots, set(fr))
    refs2, n_st, errs = refs_from_stores(roots, set(fr), refs)
    for k, v in refs2.items(): refs[k].extend(v)
    by_model = collections.Counter(str(v["verdict"].get("judge_model")) for v in fr.values())
    by_org = collections.Counter(str(v["org_used"]) for v in fr.values())
    by_reason = collections.Counter(str(v["verdict"].get("fallback_reason") or ("fallback" if v["verdict"].get("fallback") else "non-anthropic org")) for v in fr.values())
    filed_foreign = [r for rs in refs.values() for r in rs if r.get("filed_from_foreign_row")]
    referenced = {k: v for k, v in refs.items()}
    rec = {"cache": str(cache), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_foreign_rows": len(fr), "by_judge_model": dict(by_model), "by_org_used": dict(by_org), "by_fallback_reason": dict(by_reason),
           "created_at_range": [min((v["created_at_iso"] for v in fr.values() if v["created_at_iso"]), default=None), max((v["created_at_iso"] for v in fr.values() if v["created_at_iso"]), default=None)],
           "n_keys_referenced_by_scanned_stores": len(referenced), "n_judge_items_files_scanned": n_ji, "n_unjudged_stores_keyed": n_st,
           "n_filed_rows_that_used_a_foreign_row": len(filed_foreign), "filed_rows_that_used_a_foreign_row": filed_foreign, "errors": errs,
           "how_to_tell": ("a foreign row is org_used='openrouter' with verdict_json {fallback:true, judge_backend, judge_model, label}; a sonnet-5 row is "
                           "{label} with org_used lowprio/highprio/batch. The KEY is identical in form. In a filed judge_items row the producer shows only as "
                           "cache_org / verdict_extra_keys; judge_model there is the protocol's judge."),
           "what_a_reader_must_do": ("never consume these keys as sonnet-5 verdicts: seed a private cache with seed_private_cache.py (global scrub of "
                                     "producer-stamped rows) and judge with served_capability_judge.py, which refuses+evicts any such row at the read and re-judges live; "
                                     "check judge_summary.controls.*.n_verdicts_with_foreign_keys == 0 and no judge_items.cache_org == 'openrouter'."),
           "rows": {k: {**v, "referenced_by": refs.get(k, [])} for k, v in fr.items()}}
    Path(a.out).expanduser().write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if a.md:
        lines = [f"# Foreign rows in the judge cache `{cache}` — {rec['written_at']}", "",
                 f"{len(fr)} producer-stamped foreign rows (by producing model {dict(by_model)}; org {dict(by_org)}; reason {dict(by_reason)}; written {rec['created_at_range'][0]} … {rec['created_at_range'][1]}).",
                 f"{len(referenced)} of them are referenced by stores under the scanned roots ({n_ji} judge_items files, {n_st} unjudged stores keyed); "
                 f"{len(filed_foreign)} filed rows used a foreign row (listed in the JSON).", "", "| lane | cell | control | item | filed status | used foreign row |", "|---|---|---|---|---|---|"]
        for k, rs in sorted(referenced.items(), key=lambda kv: (kv[1][0]["lane"], kv[1][0]["item_id"])):
            for r in rs: lines.append(f"| {r['lane']} | {r['cell'][:48]} | {r['control']} | {r['item_id']} | {r.get('filed_status')} | {'YES' if r.get('filed_from_foreign_row') else 'no'} |")
        lines += ["", rec["how_to_tell"], "", rec["what_a_reader_must_do"]]
        Path(a.md).expanduser().write_text("\n".join(lines) + "\n")
    print(json.dumps({k: rec[k] for k in ("n_foreign_rows", "by_judge_model", "by_org_used", "by_fallback_reason", "created_at_range", "n_keys_referenced_by_scanned_stores", "n_filed_rows_that_used_a_foreign_row", "n_unjudged_stores_keyed")}), "errors:", errs[:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
