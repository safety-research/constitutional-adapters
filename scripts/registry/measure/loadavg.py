#!/usr/bin/env python3
"""loadavg.py — the grading host's 1-min load average over the last 24 h from sysstat.

    python scripts/registry/measure/loadavg.py [--hours 24] [--files /var/log/sysstat/saNN ...] [--from-text FILE] [--ps] [--json]

Runs ``sar -q -f <file>`` (``LC_ALL=C S_TIME_FORMAT=ISO``: ISO dates and 24-h times) over the daily files that cover the window
(today's and yesterday's by default) and reports the ``ldavg-1`` series: samples, interval, median / mean / p90 / max, and the
share of samples over a threshold (4, the S5 target). NOTE sysstat samples every 10 min here (``sysstat-collect.timer``,
``OnCalendar=*:00/10``) — ``ldavg-1`` IS the 1-min load average, sampled at that cadence (144 points per 24 h), not once a minute.
``--ps`` also lists the registry processes resident now (``ps``), the "no registry process but the fold" check inside a window.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

SA_DIR = "/var/log/sysstat"
HEADER_RE = re.compile(r"^\S.*?\s+(\d{4}-\d\d-\d\d|\d\d/\d\d/\d\d)\s")
ROW_RE = re.compile(r"^(\d\d:\d\d:\d\d)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)")


def sar_text(path: str) -> str:
    env = dict(os.environ, LC_ALL="C", S_TIME_FORMAT="ISO")
    r = subprocess.run(["sar", "-q", "-f", path], capture_output=True, text=True, env=env, check=False)
    if r.returncode != 0:
        raise SystemExit(f"sar -q -f {path}: rc={r.returncode}: {r.stderr.strip()[:200]}")
    return r.stdout


def parse_sar(text: str) -> List[Tuple[datetime, float, float, float, int]]:
    """``(local timestamp, ldavg-1, ldavg-5, ldavg-15, runq)`` per sample; the date comes from the header line."""
    day: Optional[datetime] = None
    out: List[Tuple[datetime, float, float, float, int]] = []
    prev: Optional[datetime] = None
    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m and ("Linux" in line or "_x86_64_" in line or "CPU" in line):
            d = m.group(1)
            for fmt in ("%Y-%m-%d", "%m/%d/%y"):
                try:
                    day = datetime.strptime(d, fmt)
                    break
                except ValueError:
                    continue
            prev = None
            continue
        m = ROW_RE.match(line)
        if not m or day is None:
            continue
        hh, mm, ss = (int(x) for x in m.group(1).split(":"))
        t = day.replace(hour=hh, minute=mm, second=ss)
        if prev is not None and t < prev:
            t += timedelta(days=1)
        prev = t
        out.append((t, float(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(2))))
    return out


def default_files(now: datetime, hours: float) -> List[str]:
    days = int(hours // 24) + 2
    files = []
    for i in range(days - 1, -1, -1):
        d = now - timedelta(days=i)
        p = os.path.join(SA_DIR, f"sa{d.day:02d}")
        if os.path.exists(p) and datetime.fromtimestamp(os.path.getmtime(p)) >= now - timedelta(days=days):
            files.append(p)
    return files


def analyze(samples: List[Tuple[datetime, float, float, float, int]], hours: float, now: datetime, threshold: float = 4.0) -> Dict[str, object]:
    cutoff = now - timedelta(hours=hours)
    win = sorted((s for s in samples if cutoff <= s[0] <= now), key=lambda s: s[0])
    l1 = [s[1] for s in win]
    gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(win, win[1:])]
    q = sorted(l1)
    p90 = q[min(len(q) - 1, int(round(0.9 * (len(q) - 1))))] if q else None
    return {"hours": hours, "samples": len(win), "interval_s": round(statistics.median(gaps), 0) if gaps else None,
            "first": win[0][0].isoformat() if win else None, "last": win[-1][0].isoformat() if win else None,
            "ldavg_1": {"median": round(statistics.median(l1), 2) if l1 else None, "mean": round(statistics.mean(l1), 2) if l1 else None,
                        "p90": p90, "max": max(l1) if l1 else None, "min": min(l1) if l1 else None,
                        "share_over_threshold": round(sum(1 for v in l1 if v > threshold) / len(l1), 3) if l1 else None, "threshold": threshold},
            "ldavg_5_median": round(statistics.median([s[2] for s in win]), 2) if win else None,
            "series": [{"t": s[0].isoformat(), "ldavg_1": s[1]} for s in win]}


def registry_processes() -> List[Dict[str, object]]:
    r = subprocess.run(["ps", "-eo", "pid,etimes,pcpu,rss,args"], capture_output=True, text=True, check=False)
    out = []
    for line in r.stdout.splitlines()[1:]:
        if "sisteer.registry" in line and "loadavg.py" not in line:
            parts = line.split(None, 4)
            if len(parts) == 5:
                out.append({"pid": int(parts[0]), "etimes_s": int(parts[1]), "pcpu": float(parts[2]), "rss_kb": int(parts[3]), "args": parts[4][:160]})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--files", nargs="*", help=f"sysstat daily files (default: the ones under {SA_DIR} covering the window)")
    ap.add_argument("--from-text", help="a saved `sar -q` output instead of running sar")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--now", help="the window's end (local time, YYYY-MM-DDTHH:MM:SS; default now) — for a saved text")
    ap.add_argument("--ps", action="store_true", help="also list the registry processes resident now")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    now = datetime.strptime(a.now, "%Y-%m-%dT%H:%M:%S") if a.now else datetime.now()
    if a.from_text:
        with open(a.from_text, "r", encoding="utf-8") as fh:
            samples = parse_sar(fh.read())
        files = [a.from_text]
    else:
        files = a.files or default_files(now, a.hours)
        samples = []
        for f in files:
            samples += parse_sar(sar_text(f))
    rep = analyze(samples, a.hours, now, a.threshold)
    rep["files"] = files
    if a.ps:
        rep["registry_processes"] = registry_processes()
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
        return 0
    L = rep["ldavg_1"]
    print(f"ldavg-1 over the last {a.hours:g} h ({rep['samples']} samples every ~{rep['interval_s']} s, {rep['first']} → {rep['last']}): "
          f"median {L['median']}, mean {L['mean']}, p90 {L['p90']}, max {L['max']}; share over {a.threshold:g}: {L['share_over_threshold']}")
    if a.ps:
        procs = rep["registry_processes"]
        print(f"registry processes now: {len(procs)}")
        for p in procs:
            print(f"  pid {p['pid']} up {p['etimes_s']} s cpu {p['pcpu']}% {p['args']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
