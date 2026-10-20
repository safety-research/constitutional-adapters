#!/usr/bin/env python
"""Repair a static store that TWO writers appended to concurrently (the sampled canary 84812 and the full job
84829 on the same (family, cell, trial) shards, 2026-09-09): keep the FIRST row per (goal_id, sample_idx, trial) in
responses.jsonl and the FIRST record per goal_id in progress.jsonl; PARK every removed row beside the file as
``<name>.duplicates.<UTC stamp>.jsonl`` (never deleted); write ``dedupe_record.json`` in the shard dir with the
counts and how many duplicate pairs DIFFER in response text (both writers used the same per-item seed on two
servers, so identical text is possible but not guaranteed). Refuses a shard whose writers are still active
(any file modified within --quiet-s seconds), a row without the sampled fields, or duplicates whose `decode`
blocks differ (that is not a race, that is two decoders — never merged). Idempotent: a second run finds nothing.

usage: dedupe_raced_store.py --cell-root <…/<family>/<cell>> [--quiet-s 600] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KEY_FIELDS = ("goal_id", "sample_idx", "trial")
REQUIRED = ("decode", "sampling_seed", "finish_reason", "response")


def _rows(path: Path) -> list:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i + 1}: not JSON ({e}) — a torn line; refusing to dedupe a damaged file")
    return out


def _write(path: Path, rows: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def dedupe_shard(shard: Path, *, quiet_s: float, dry_run: bool, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    resp, prog = shard / "responses.jsonl", shard / "progress.jsonl"
    for p in (resp, prog):
        if not p.is_file():
            raise SystemExit(f"{shard}: missing {p.name}")
        if now - p.stat().st_mtime < quiet_s:
            raise SystemExit(f"{p} was modified {now - p.stat().st_mtime:.0f}s ago (< {quiet_s}s) — a writer may "
                             "still be active; refusing (wait for the job's drivers to finish)")
    rows = _rows(resp)
    keep, dups, decodes = [], [], set()
    seen: dict = {}
    for r in rows:
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise SystemExit(f"{resp}: a row lacks {missing} — not a sampled store; refusing")
        decodes.add(json.dumps(r["decode"], sort_keys=True))
        k = tuple(r.get(f) for f in KEY_FIELDS)
        if k in seen:
            dups.append((seen[k], r))
        else:
            seen[k] = r
            keep.append(r)
    if len(decodes) > 1:
        raise SystemExit(f"{resp}: {len(decodes)} distinct decode blocks — two decoders, not a race; refusing")
    precs = _rows(prog)
    pkeep, pdups, pseen = [], [], set()
    for r in precs:
        g = r.get("goal_id")
        if g in pseen:
            pdups.append(r)
        else:
            pseen.add(g); pkeep.append(r)
    rec = {"shard": str(shard), "n_rows_before": len(rows), "n_rows_after": len(keep), "n_duplicate_rows": len(dups),
           "n_duplicate_pairs_text_differs": sum(1 for a, b in dups if a["response"] != b["response"]),
           "n_duplicate_pairs_text_identical": sum(1 for a, b in dups if a["response"] == b["response"]),
           "n_progress_before": len(precs), "n_progress_after": len(pkeep), "n_duplicate_progress": len(pdups),
           "goals_after": len(pseen), "key": list(KEY_FIELDS), "rule": "first row per key kept (the canary's, written first)",
           "at": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run}
    if dry_run or (not dups and not pdups):
        rec["changed"] = False
        return rec
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write(shard / f"responses.jsonl.duplicates.{stamp}.jsonl", [b for _, b in dups])
    _write(shard / f"progress.jsonl.duplicates.{stamp}.jsonl", pdups)
    _write(resp, keep)
    _write(prog, pkeep)
    rec["changed"] = True
    rec["parked"] = [f"responses.jsonl.duplicates.{stamp}.jsonl", f"progress.jsonl.duplicates.{stamp}.jsonl"]
    (shard / "dedupe_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-root", required=True, help="the <private_run>/<family>/<cell> dir holding shard*/")
    ap.add_argument("--quiet-s", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kept-writer", default="", help="provenance of the rows KEPT (e.g. the canary job id + name)")
    ap.add_argument("--dropped-writer", default="", help="provenance of the rows PARKED (e.g. the full job id + name)")
    a = ap.parse_args(argv)
    root = Path(a.cell_root)
    shards = sorted(p for p in root.glob("shard*") if p.is_dir())
    if not shards:
        raise SystemExit(f"{root}: no shard dirs")
    total = {"n_rows_before": 0, "n_rows_after": 0, "n_duplicate_rows": 0, "n_duplicate_pairs_text_differs": 0,
             "n_progress_after": 0, "goals_after": 0}
    for s in shards:
        rec = dedupe_shard(s, quiet_s=a.quiet_s, dry_run=a.dry_run)
        if rec.get("changed"):
            rp = s / "dedupe_record.json"
            r2 = json.loads(rp.read_text())
            r2["writers"] = {"kept": a.kept_writer, "parked": a.dropped_writer,
                              "rule": "the first-written row per key is kept; both writers used the same per-item "
                                      "sampling_seed and decode block, so the parked rows are a same-seed replicate "
                                      "from a second Slurm job"}
            rp.write_text(json.dumps(r2, indent=1) + "\n")
        for k in total:
            total[k] += rec[k]
        print(json.dumps({k: rec[k] for k in ("shard", "n_rows_before", "n_rows_after", "n_duplicate_rows",
                                              "n_duplicate_pairs_text_differs", "goals_after", "changed")}))
    print(json.dumps({"TOTAL": total, "dry_run": a.dry_run}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
