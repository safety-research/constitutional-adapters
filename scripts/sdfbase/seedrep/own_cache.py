"""The OWNER RECORD of a controls judge cache (grade_battery.sh) — one implementation for both callers: the runbook's
planner (plan-time state: fresh / resume / refused) and the unit's first step per (store, rung), which records the owner BEFORE
the judge creates the sqlite, so a cache with no owner record can only be foreign.

    own_cache.py <cache.sqlite> <store dir> <tag> <cap> <controls root> <stamp>
        records <cache>.owner.json {store, tag, cap, controls_root, created_by, created_at} if absent (refusing a cache that already
        exists without one), else verifies the record names this store / tag / cap / root.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def owner_path(cache) -> Path:
    cache = Path(cache)
    return cache.with_name(cache.name + ".owner.json")


def want_record(store, tag, cap, controls_root) -> dict:
    return {"store": str(store), "tag": str(tag), "cap": int(cap), "controls_root": str(controls_root)}


def owner_state(cache, want: dict) -> tuple[str, str]:
    """("fresh" | "resume" | "refused", why): an owner record naming this store -> resume; one naming another store / tag / cap ->
    refused; a cache with no record -> refused (not this run's; a stale cache would serve verdicts); nothing -> fresh."""
    o = owner_path(cache)
    if o.is_file():
        got = json.loads(o.read_text())
        bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if bad:
            return "refused", f"the cache's owner record {o} names another store / tag / cap: {bad}"
        return "resume", f"cache owned by this store ({o}); the judge's store-level resume skips every item already filed"
    if Path(cache).exists():
        return "refused", f"cache {cache} exists with no owner record — not this run's cache (a stale cache would serve verdicts)"
    return "fresh", "absent; the owner is recorded, then the judge creates an empty sqlite"


def record(cache, want: dict, stamp: str) -> str:
    state, why = owner_state(cache, want)
    if state == "refused":
        raise SystemExit(f"REFUSED: {why}")
    o = owner_path(cache)
    if state == "fresh":
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps({**want, "created_by": stamp, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
        return f"[own-cache] recorded: {o}"
    return f"[own-cache] resume: {o}"


def main(argv=None) -> None:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 6:
        raise SystemExit("usage: own_cache.py <cache.sqlite> <store dir> <tag> <cap> <controls root> <stamp>")
    print(record(a[0], want_record(a[1], a[2], a[3], a[4]), a[5]))


if __name__ == "__main__":
    main()
