#!/usr/bin/env python3
"""landings.py [--state DIR] [--legacy GLOB]... [--json] — the landing chain's per-landing minutes (gate, live-DB, PR+merge,
total), re-run after Phase 2.

Two log shapes are read, so the before and after numbers come from one parser:

* ``$STATE/land/<branch>.out`` (land.sh since Phase 2): ``[land] @ <step>:<start|end> <UTC ISO>`` stamps per step
  (merge, verdict, baseline-wait, live-db, push, status, pr, merge-pr, cleanup) and ``[land] verdict=<reuse|waiver|slurm-gate>``.
  A landing whose main moved under its gated tip repeats merge → pr (``[land] re-gate round=<n>``): a step's minutes run
  from its FIRST start to its LAST end, and ``rounds`` counts the repeats. The live-DB step now runs
  on Slurm (``live_db_slurm.sh``): its sub-steps stamp ``live-db-head``, ``live-db-mirror-sync``, ``live-db-tree``,
  ``live-db-ship``, ``live-db-job``, ``live-db-collect`` inside ``live-db``, and ``[land] live-db job <id>`` names the Slurm job
  (``live_db_job``; the sub-step minutes in ``live_db_steps``; a landing with no such line ran its live-DB step on the
  grading host, ``live_db_where = grading_host``).
* the legacy ``/tmp/land-*.out`` (the pre-Phase-2 land.sh, redirected by the queue scripts): a ``HH:MM:SS`` UTC stamp on
  the gate line (``full registry suite on Modal for merged tip``, ``gate verdict LAND already recorded``, ``FULL SUITE
  WAIVED``) and on the live-DB line, the end from the file's mtime, the live-DB step's end from the branch's
  ``land-<branch>.pytest.log`` mtime in the same directory (the branch comes from ``Deleted branch <b>`` or the
  ``gate-land-<b>.out`` the verdict was reused from), the date anchored to the mtime (a stamp later than the end
  wraps to the previous day). Only a file that reached ``LAND_RC=0`` / ``[land] LANDED`` is a landing; the others
  are listed with their exit.

Minutes: gate = verdict step (legacy: gate line -> live-DB line, only when a Modal run happened inside the file);
live-DB = the pytest step; PR+merge = push -> cleanup end (legacy: pytest.log mtime -> .out mtime); total = first
stamp -> end. Medians per column close the table. Stdlib only; read-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_STATE = os.path.expanduser(os.environ.get("REGISTRY_LANDING_STATE", "~/.local/state/registry-landing"))
STAMP = re.compile(r"^\[land\] @ (?P<step>[a-z][a-z-]*):(?P<edge>start|end) (?P<utc>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s*$", re.M)
VERDICT_KIND = re.compile(r"^\[land\] verdict=(?P<kind>reuse|waiver|slurm-gate)\s*$", re.M)
REGATE = re.compile(r"^\[land\] re-gate round=(?P<n>\d+):", re.M)
LEGACY_GATE = re.compile(r"^\[land\] full registry suite on Modal for merged tip (?P<sha>[0-9a-f]{8}) (?P<hms>\d\d:\d\d:\d\d)\s*$", re.M)
LEGACY_REUSE = re.compile(r"^\[land\] gate verdict LAND already recorded for merged tip (?P<sha>[0-9a-f]{8}) in (?P<path>\S+) — reusing", re.M)
LEGACY_WAIVE = re.compile(r"^\[land\] FULL SUITE WAIVED BY THE USER for merged tip (?P<sha>[0-9a-f]{8}):.* (?P<hms>\d\d:\d\d:\d\d)\s*$", re.M)
LEGACY_LIVE = re.compile(r"^\[land\] live-DB test files on merged tip (?P<sha>[0-9a-f]{8}) \((?P<n>\d+) files.*\) (?P<hms>\d\d:\d\d:\d\d)\s*$", re.M)
PR = re.compile(r"/pull/(?P<n>\d+)\s*$", re.M)
DELETED = re.compile(r"^Deleted branch (?P<b>\S+) \(was", re.M)
GATE_FILE = re.compile(r"gate-land-(?P<b>[^/\s]+?)(?:\.[a-z0-9-]+)?\.out")
RC = re.compile(r"^LAND_RC=(?P<rc>\d+)\s*$", re.M)
LANDED = re.compile(r"^\[land\] LANDED (?P<b>\S+) as PR (?P<n>\d+)", re.M)
LIVE_DB_JOB = re.compile(r"^\[land\] live-db job (?P<id>\d+) ", re.M)
LIVE_DB_SUBSTEPS = ("head", "mirror-sync", "tree", "ship", "job", "collect")
REFUSED = re.compile(r"^(GATE NOT LAND|GATE: no recorded LAND verdict|LIVE-DB TESTS RED|CONFLICT:|BASELINE for main|NOT MERGEABLE)", re.M)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _mtime(p: str) -> datetime:
    return datetime.fromtimestamp(os.stat(p).st_mtime, tz=timezone.utc)


def _anchor(hms: str, end: datetime) -> datetime:
    """A `HH:MM:SS` UTC stamp placed on the end's day, or the day before when it is later than the end (midnight wrap)."""
    h, m, s = (int(x) for x in hms.split(":"))
    t = end.replace(hour=h, minute=m, second=s, microsecond=0)
    if t > end + timedelta(seconds=5):
        t -= timedelta(days=1)
    return t


def _min(a: datetime | None, b: datetime | None) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a).total_seconds() / 60.0, 1)


def parse_new(path: str, text: str) -> dict:
    stamps: dict = {}
    for m in STAMP.finditer(text):
        t = datetime.strptime(m.group("utc"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        d = stamps.setdefault(m.group("step"), {})
        if m.group("edge") == "start":
            d.setdefault("start", t)      # a repeated round keeps the first start …
        else:
            d["end"] = t                  # … and takes the last end
    m = VERDICT_KIND.search(text)
    kind = m.group("kind") if m else None
    landed = LANDED.search(text)
    rc = RC.search(text)
    branch = landed.group("b") if landed else Path(path).stem
    pr = (landed.group("n") if landed else None) or (PR.findall(text) or [None])[-1]
    start = stamps.get("merge", {}).get("start")
    end = stamps.get("cleanup", {}).get("end") or _mtime(path)
    gate = _min(stamps.get("verdict", {}).get("start"), stamps.get("verdict", {}).get("end"))
    bwait = _min(stamps.get("baseline-wait", {}).get("start"), stamps.get("baseline-wait", {}).get("end"))
    live = _min(stamps.get("live-db", {}).get("start"), stamps.get("live-db", {}).get("end"))
    prm = _min(stamps.get("push", {}).get("start"), stamps.get("cleanup", {}).get("end"))
    ok = bool(landed) or (rc is not None and rc.group("rc") == "0")
    jobs = LIVE_DB_JOB.findall(text)
    steps = {k: _min(stamps.get(f"live-db-{k}", {}).get("start"), stamps.get(f"live-db-{k}", {}).get("end")) for k in LIVE_DB_SUBSTEPS
             if f"live-db-{k}" in stamps}
    return {"file": os.path.basename(path), "shape": "phase2", "branch": branch, "end_utc": _iso(end), "gate_kind": {"reuse": "REUSED", "waiver": "WAIVED",
            "slurm-gate": "SLURM"}.get(kind or "", "?"), "gate_min": gate, "baseline_wait_min": bwait, "live_db_min": live, "pr_merge_min": prm,
            "total_min": _min(start, end), "pr": pr, "landed": ok, "exit": None if ok else (rc.group("rc") if rc else "incomplete"),
            "rounds": 1 + len(REGATE.findall(text)), "live_db_where": "slurm" if jobs else "grading_host", "live_db_job": jobs[-1] if jobs else None,
            "live_db_jobs": jobs, "live_db_steps": steps}


def parse_legacy(path: str, text: str) -> dict:
    end = _mtime(path)
    d = os.path.dirname(path)
    rc = RC.search(text)
    deleted, reuse, gate, waive, lm = (p.search(text) for p in (DELETED, LEGACY_REUSE, LEGACY_GATE, LEGACY_WAIVE, LEGACY_LIVE))
    branch = None
    if deleted:
        branch = deleted.group("b")
    elif reuse:
        g = GATE_FILE.search(reuse.group("path"))
        branch = g.group("b") if g else None
    kind, t_gate = "?", None
    if gate:
        kind, t_gate = "MODAL", _anchor(gate.group("hms"), end)
    elif reuse:
        kind = "REUSED"
    elif waive:
        kind, t_gate = "WAIVED", _anchor(waive.group("hms"), end)
    elif not lm and deleted:
        kind = "LOCAL-SUITE"    # the pre-2026-09-24 chain: the full suite ran locally and left no stamp in this file
    t_live = _anchor(lm.group("hms"), end) if lm else None
    t_live_end = None
    if branch:
        pl = os.path.join(d, f"land-{branch}.pytest.log")
        if os.path.exists(pl):
            t_live_end = _mtime(pl)
    ok = rc is not None and rc.group("rc") == "0" or (rc is None and bool(deleted))
    if lm and kind == "MODAL":
        # the merged-tip sha on the gate line and the live-DB line must agree, else the gate ran on another tip (a retry file)
        gate_min = _min(t_gate, t_live) if gate.group("sha") == lm.group("sha") else None
    else:
        gate_min = None
    live = _min(t_live, t_live_end) if (t_live_end and t_live and t_live_end >= t_live) else None
    prm = _min(t_live_end, end) if (t_live_end and t_live_end <= end) else None
    first = t_gate or t_live
    exit_ = None
    if not ok:
        exit_ = rc.group("rc") if rc else ("refused" if REFUSED.search(text) else "incomplete")
    return {"file": os.path.basename(path), "shape": "legacy", "branch": branch, "end_utc": _iso(end), "gate_kind": kind, "gate_min": gate_min,
            "baseline_wait_min": None, "live_db_min": live, "pr_merge_min": prm, "total_min": _min(first, end), "pr": (PR.findall(text) or [None])[-1],
            "landed": ok, "exit": exit_, "rounds": 1, "live_db_where": "grading_host", "live_db_job": None, "live_db_jobs": [], "live_db_steps": {}}


def parse_file(path: str) -> dict:
    text = Path(path).read_text(errors="replace")
    return parse_new(path, text) if STAMP.search(text) else parse_legacy(path, text)


def collect(state: str, legacy: list) -> list:
    files = sorted(glob.glob(os.path.join(state, "land", "*.out")))
    for g in legacy:
        files += sorted(glob.glob(os.path.expanduser(g)))
    rows = [parse_file(f) for f in files]
    rows.sort(key=lambda r: r["end_utc"])
    return rows


def _cell(v) -> str:
    return "–" if v is None else f"{v:.1f}"


def _med(vals):
    vals = [v for v in vals if v is not None]
    return (round(statistics.median(vals), 1), len(vals)) if vals else (None, 0)


def render(rows: list) -> str:
    out = ["| file | branch | end UTC | gate | gate min | baseline wait | live-DB min | live-DB where | PR+merge | total | PR |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    landings = [r for r in rows if r["landed"]]
    for r in landings:
        where = r.get("live_db_where", "grading_host") + (f" job {r['live_db_job']}" if r.get("live_db_job") else "")
        out.append(f"| {r['file']} | {r['branch'] or '?'} | {r['end_utc']} | {r['gate_kind']} | {_cell(r['gate_min'])} | {_cell(r['baseline_wait_min'])} | "
                   f"{_cell(r['live_db_min'])} | {where} | {_cell(r['pr_merge_min'])} | {_cell(r['total_min'])} | {r['pr'] or '–'} |")
    meds = {k: _med([r[k] for r in landings]) for k in ("gate_min", "live_db_min", "pr_merge_min", "total_min")}
    gate_inside = _med([r["gate_min"] for r in landings if r["gate_kind"] in ("MODAL", "SLURM")])
    tot_reused = _med([r["total_min"] for r in landings if r["gate_kind"] in ("REUSED", "WAIVED")])
    out.append("")
    out.append(f"landings: {len(landings)} of {len(rows)} files; medians — gate inside land.sh {gate_inside[0]} min (n={gate_inside[1]}), "
               f"live-DB {meds['live_db_min'][0]} min (n={meds['live_db_min'][1]}), PR+merge {meds['pr_merge_min'][0]} min (n={meds['pr_merge_min'][1]}), "
               f"total {meds['total_min'][0]} min (n={meds['total_min'][1]}); total with a reused/waived verdict {tot_reused[0]} min (n={tot_reused[1]})")
    others = [r for r in rows if not r["landed"]]
    if others:
        out.append("not landings: " + ", ".join(f"{r['file']} ({r['exit']})" for r in others))
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--legacy", action="append", default=[], help="a glob of legacy land-*.out files to read too (repeatable), e.g. '/tmp/land-*.out'")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = collect(a.state, a.legacy)
    if a.json:
        print(json.dumps({"rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}, indent=1, default=str))
    else:
        sys.stdout.write(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
