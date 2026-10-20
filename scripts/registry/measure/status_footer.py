#!/usr/bin/env python3
"""status_footer.py — the status page's render series from ``state/status_render.jsonl``.

    python scripts/registry/measure/status_footer.py [--path FILE] [--hours 24] [--budget-ms 120000] [--cadence-s 600] [--json]

Each line is one render (``sisteer.registry.status.render_history_line``): ``built_utc``, ``total_ms``, per-section
``ms`` / ``as_of`` / ``from_cache`` / ``state``, the cached-with-refresh set. Reports: the number of renders in the window, the
LONGEST run of consecutive renders with ``total_ms`` ≤ the budget and the current streak (the ≥ 5 claim), the build-stamp
cadence (every gap between consecutive renders; the largest; how many exceed the cadence), and section freshness at the latest
render (each section's ``as_of`` age; the sections from cache; the sections with no cache).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join(os.environ.get("REGISTRY_ROOT", os.path.expanduser("~/registry")), "state", "status_render.jsonl")


def parse_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(str(s), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def read_lines(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    continue
    return out


def analyze(lines: List[Dict[str, object]], hours: float, budget_ms: float, cadence_s: float, now: Optional[datetime] = None) -> Dict[str, object]:
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    rows = [(parse_utc(str(l.get("built_utc"))), l) for l in lines]
    rows = [(t, l) for t, l in rows if t is not None and t >= cutoff]
    rows.sort(key=lambda r: r[0])
    totals = [float(l.get("total_ms") or 0) for _, l in rows]
    longest = cur = 0
    for v in totals:
        cur = cur + 1 if v <= budget_ms else 0
        longest = max(longest, cur)
    gaps = [(b - a).total_seconds() for (a, _), (b, _) in zip(rows, rows[1:])]
    freshness: Dict[str, object] = {}
    if rows:
        t_last, last = rows[-1]
        for name, st in (last.get("sections") or {}).items():
            as_of = parse_utc(st.get("as_of")) if isinstance(st, dict) else None
            freshness[name] = {"from_cache": bool(st.get("from_cache")) if isinstance(st, dict) else None, "state": st.get("state") if isinstance(st, dict) else None,
                               "as_of_age_s": round((t_last - as_of).total_seconds(), 1) if as_of else None}
    return {"path_lines": len(lines), "renders_in_window": len(rows), "hours": hours, "budget_ms": budget_ms, "cadence_s": cadence_s,
            "totals_ms": {"median": round(statistics.median(totals), 1) if totals else None, "max": max(totals) if totals else None,
                          "min": min(totals) if totals else None, "n_over_budget": sum(1 for v in totals if v > budget_ms)},
            "consecutive_within_budget": {"longest": longest, "current": cur, "at_least_5": longest >= 5},
            "cadence": {"n_gaps": len(gaps), "max_gap_s": round(max(gaps), 1) if gaps else None, "median_gap_s": round(statistics.median(gaps), 1) if gaps else None,
                        "n_gaps_over": sum(1 for g in gaps if g > cadence_s), "every_gap_within": bool(gaps) and all(g <= cadence_s for g in gaps),
                        "first_built": rows[0][1].get("built_utc") if rows else None, "last_built": rows[-1][1].get("built_utc") if rows else None,
                        "age_of_last_s": round((now - rows[-1][0]).total_seconds(), 1) if rows else None},
            "latest": {"built_utc": rows[-1][1].get("built_utc"), "total_ms": rows[-1][1].get("total_ms"), "heavy_set": rows[-1][1].get("heavy_set"),
                       "from_cache": sorted(n for n, f in freshness.items() if f["from_cache"]),
                       "no_cache": sorted(n for n, f in freshness.items() if f["state"] == "no_cache"),
                       "max_as_of_age_s": max((f["as_of_age_s"] for f in freshness.values() if f["as_of_age_s"] is not None), default=None)} if rows else None,
            "freshness": freshness}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--budget-ms", type=float, default=120000.0)
    ap.add_argument("--cadence-s", type=float, default=600.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = analyze(read_lines(a.path), a.hours, a.budget_ms, a.cadence_s)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
        return 0
    print(f"{a.path}: {rep['path_lines']} lines; {rep['renders_in_window']} renders in the last {a.hours:g} h")
    t = rep["totals_ms"]; c = rep["consecutive_within_budget"]; g = rep["cadence"]
    print(f"total_ms median {t['median']}, max {t['max']}, over {a.budget_ms:g} ms: {t['n_over_budget']}; consecutive within budget: longest {c['longest']}, "
          f"current {c['current']} (>= 5: {c['at_least_5']})")
    print(f"build stamp cadence: {g['n_gaps']} gaps, median {g['median_gap_s']} s, max {g['max_gap_s']} s, over {a.cadence_s:g} s: {g['n_gaps_over']} "
          f"(every gap within: {g['every_gap_within']}); last built {g['last_built']} ({g['age_of_last_s']} s ago)")
    if rep["latest"]:
        L = rep["latest"]
        print(f"latest render: heavy set {L['heavy_set']}; from cache {L['from_cache']}; no cache {L['no_cache']}; oldest section as_of {L['max_as_of_age_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
