"""Seed a PRIVATE copy of the shared judge cache for one judge run, dropping
the entries this run must not consume (run with PYTHONPATH=src:scripts).

Why: the shared cache is keyed by (prompt, generation, rubric, protocol,
judge_model) and is written by more than one judging lineage; a grok-4.6
fallback label filed under the claude-sonnet-5 key, or a reason-less legacy
abstention sentinel, would be consumed by the harness as a sonnet-5 verdict.
Those entries are NOT deleted from the shared cache (peers' re-reads depend
on them): a `VACUUM INTO` copy is made, only those keys are deleted there,
the run points the harness at the copy via CAPEVAL_JUDGE_CACHE, and
merge_cache_back.py later INSERT OR IGNOREs the copy back.

The scrub is GLOBAL as well as store-scoped (2026-09-09): a store-scoped scrub
deletes only the foreign rows the stores present at seed time reference, so a
store that lands later (the GLM lane, whose controls store arrived after the
five-lane seed) can still resolve a grok-fallback row — 5 filed GLM rows did,
flagged by the judge's `verdict_extra_keys` but filed. Every row whose verdict
has another lineage's shape (a non-abstention verdict carrying keys beyond
`label`, e.g. fallback/judge_backend/judge_model) or is a reason-less legacy
abstention sentinel is therefore deleted from the private copy whatever the
stores reference; the store-scoped pass still runs first so the seed record
names the items each removal would have touched.

usage: python seed_private_cache.py --controls-root DIR --out CACHE.sqlite
       python seed_private_cache.py --scrub-existing CACHE.sqlite   # apply the global scrub to a private cache already in use
"""
import argparse
import collections
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # scripts/
import served_capability_judge as sj  # noqa: E402
from sisteer.capeval import corpora, judging  # noqa: E402
from sisteer.capeval.tiers import judged  # noqa: E402


def _foreign_shape(v: dict):
    """Why a cached verdict must not be consumed as a sonnet-5 verdict, or None if it may."""
    if v.get("abstained") and not v.get("reason"):
        return "legacy abstention sentinel without a reason; re-attempt live"
    if not v.get("abstained") and (set(v) - sj._LABEL_KEYS):
        return (f"foreign-lineage verdict (judge_model={v.get('judge_model')}, "
                f"backend={v.get('judge_backend')}) under a sonnet-5 protocol key")
    return None


_ANTHROPIC_ORGS = {"lowprio", "highprio", "batch"}


def _producer_foreign(v: dict, org: str | None) -> str | None:
    """The GLOBAL rule: only what the PRODUCER stamped as foreign. The shared cache holds several protocols' verdicts
    (pairwise `{"verdict": "a"}` rows, abstention sentinels of other lineages, ...) whose keys cannot be told apart
    from ours by a hash, so shape-based rules are for the store-scoped pass over OUR keys (and the judge's read-time
    refusal) only. First cut of this function used the shape rule globally and removed 66,772 pairwise verdicts from a
    private copy (2026-09-09 01:52Z; restored from the shared cache — see the *.scrub.*.json + *.restore.*.json records)."""
    if v.get("fallback"):
        return f"fallback-stamped verdict (judge_model={v.get('judge_model')}, backend={v.get('judge_backend')})"
    if org is not None and org not in _ANTHROPIC_ORGS:
        return f"org_used={org!r} is not an Anthropic pool"
    return None


def global_scrub(dc: sqlite3.Connection) -> dict:
    """Delete every PRODUCER-STAMPED foreign row (fallback-stamped, or written under a non-Anthropic org) from the
    private copy, whatever the stores reference. Returns a record: counts by reason, org_used, producing judge_model,
    and the deleted keys."""
    victims = []
    for key, vj, org, created in dc.execute("SELECT key, verdict_json, org_used, created_at FROM verdicts").fetchall():
        try:
            v = json.loads(vj)
        except Exception:  # noqa: BLE001 — an unparseable verdict is not a sonnet-5 label either
            victims.append((key, "unparseable verdict_json", org, None, created)); continue
        why = _producer_foreign(v, org)
        if why:
            victims.append((key, why, org, v.get("judge_model"), created))
    dc.executemany("DELETE FROM verdicts WHERE key=?", [(k,) for k, *_ in victims])
    dc.commit()
    return {"n_removed": len(victims),
            "by_reason": dict(collections.Counter(w[:60] for _, w, *_ in victims)),
            "by_org_used": dict(collections.Counter(o for _, _, o, *_ in victims)),
            "by_judge_model": dict(collections.Counter(str(m) for _, _, _, m, _ in victims)),
            "keys": [k for k, *_ in victims]}


def scrub_existing(path: Path) -> int:
    """Apply the global scrub to a private cache that is already in use; write the record beside it."""
    dc = sqlite3.connect(path, timeout=60)
    n_before = dc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    rec = global_scrub(dc)
    n_after = dc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    dc.close()
    note = {"scrubbed": str(path), "scrubbed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_entries_before": n_before, "n_entries_after": n_after, **rec}
    out = path.with_name(path.name + f".scrub.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    out.write_text(json.dumps(note, indent=1, sort_keys=True) + "\n")
    print("scrubbed", path, "before", n_before, "after", n_after, "removed", rec["n_removed"], rec["by_reason"], rec["by_judge_model"], "->", out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls-root", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--targets", default="", help="comma list of target dirs (default all)")
    ap.add_argument("--scrub-existing", default=None, help="apply the global scrub to this private cache and exit")
    args = ap.parse_args()
    if args.scrub_existing:
        return scrub_existing(Path(args.scrub_existing).expanduser())
    if not (args.controls_root and args.out):
        ap.error("--controls-root and --out are required unless --scrub-existing is given")
    src = judging.default_cache_path()
    dst = Path(args.out).expanduser()
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite a run cache")
    sc = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    sc.execute("VACUUM INTO ?", (str(dst),))
    sc.close()
    os.chmod(dst, 0o600)
    pins = corpora.load_pins()
    rows = {"xstest": judged.xstest_rows(pins), "or_bench_hard_1k": judged.or_bench_rows(pins)}
    root = Path(args.controls_root).expanduser()
    targets = ([x for x in args.targets.split(",") if x]
               or sorted(p.name for p in root.iterdir() if (p / "cells").is_dir()))
    cells = sj.discover(root, targets, "", rows, list(sj.JUDGED_CONTROLS))
    dc = sqlite3.connect(dst)
    n_before = dc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    removed = {}
    for c in cells:
        for control in c.controls:
            for iid, k in c.request_keys(control).items():
                if k in removed:
                    continue
                r = dc.execute("SELECT verdict_json, org_used, created_at FROM verdicts WHERE key=?",
                               (k,)).fetchone()
                if not r:
                    continue
                v = json.loads(r[0])
                if v.get("abstained") and not v.get("reason"):
                    why = "legacy abstention sentinel without a reason; re-attempt live"
                elif not v.get("abstained") and (set(v) - sj._LABEL_KEYS):
                    why = (f"foreign-lineage verdict (judge_model={v.get('judge_model')}, "
                           f"backend={v.get('judge_backend')}) under the {c.judge_model} key")
                else:
                    continue
                removed[k] = {"reason": why, "verdict": v, "org_used": r[1], "created_at": r[2],
                              "example_item": f"{c.target}/{c.cell_id}/{control}/{iid}"}
    dc.executemany("DELETE FROM verdicts WHERE key=?", [(k,) for k in removed])
    dc.commit()
    global_removed = global_scrub(dc)
    n_after = dc.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]
    dc.close()
    note = {"seeded_from": str(src), "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_entries_source": n_before, "n_entries_private": n_after, "n_removed": len(removed),
            "removed": removed, "global_scrub": global_removed}
    dst.with_name(dst.name + ".seed.json").write_text(json.dumps(note, indent=1, sort_keys=True) + "\n")
    print("source", n_before, "private", n_after, "removed", len(removed),
          dict(collections.Counter(v["reason"][:40] for v in removed.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
