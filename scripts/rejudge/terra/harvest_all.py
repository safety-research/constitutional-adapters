#!/usr/bin/env python3
"""Harvest SEVERAL run directories of the C/P/I track in one process -- one combined progress file -- each with the track's harvester
(:class:`rejudge.terra.harvest.TerraHarvester`: the shared harvester plus the account pause, class-scoped re-asks and the
moderation-share hold), each under its own writer lock, ledger, PAUSED file and health rules.

    PYTHONPATH=src:scripts python scripts/rejudge/terra/harvest_all.py --run-dir FULL/cpi --run-dir FULL/agentic \
        --progress FULL/progress.json [--poll-s 120] [--max-hours 72] [--dry-run] [--once]

A family is one run directory (one parser each), so a judge's full run of the track is two or more run directories; this drives them from
one loop.
Every cycle runs each harvester's cycle in turn (cycle one submits EVERY pending batch of EVERY run directory before anything is
polled: all batches in flight at once), rewrites each run directory's own ``progress.json``, then the combined ``--progress`` file:
the run directories' states, summed requests / outcomes / spend, every pause reason and the combined ``done``.

``--dry-run``: no network and no key needed -- each run directory's submit dry run (what would be POSTed, per batch) plus its
``dry_run.json`` (requests, batches, bytes, estimated cost and its bands), written to the combined progress file as the plan.

Exit codes (the shared harvester's, combined): 0 DONE (every run directory done), 4 PAUSED (a run directory stopped paused with
nothing in flight; the others ran to their end), 5 STOPPED UNFINISHED (a batch set aside or a request without a terminal outcome), 3 at
``--max-hours`` with work left (re-run to resume: the ledgers are the state). A paused run directory never stops the others.

The OpenRouter key is read from ``$OPENROUTER_API_KEY`` and never printed, logged or written anywhere; a long run belongs in a process
that outlives the shell that started it, with the key in that process's environment only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import harvester as H  # noqa: E402
from rejudge.common.state import utc_now  # noqa: E402
from rejudge.terra import harvest as TH  # noqa: E402

PRIORITY = {H.EXIT_PAUSED: 3, H.EXIT_UNFINISHED: 2, H.EXIT_MAX_HOURS: 1, H.EXIT_DONE: 0}


def combined_code(codes: List[int]) -> int:
    """PAUSED over UNFINISHED over DONE: the run as a whole is done only when every run directory is."""
    return max(codes, key=lambda c: PRIORITY.get(c, 2)) if codes else H.EXIT_DONE


def _read(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_combined(path: str, run_dirs: List[str], cycles: int, state: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    per = {rd: _read(os.path.join(rd, "progress.json")) or {} for rd in run_dirs}
    outcomes: Counter = Counter()
    req_total = req_term = 0
    spend = Counter()
    for p in per.values():
        r = p.get("requests") or {}
        req_total += int(r.get("total") or 0)
        req_term += int(r.get("terminal") or 0)
        outcomes.update({k: int(v) for k, v in (r.get("outcomes") or {}).items()})
        for k, v in (p.get("spend_usd") or {}).items():
            if isinstance(v, (int, float)) and k != "max_spend_usd":
                spend[k] += float(v)
    out = {"updated_utc": utc_now(), "state": state, "cycles": cycles,
           "done": bool(per) and all(bool(p.get("done")) for p in per.values()),
           "requests": {"total": req_total, "terminal": req_term, "outcomes": dict(outcomes)},
           "spend_usd": {k: round(v, 6) for k, v in spend.items()},
           "paused": {rd: p.get("paused") for rd, p in per.items() if p.get("paused")},
           "key": next((p.get("key") for p in per.values() if p.get("key")), None),
           "run_dirs": {rd: {k: p.get(k) for k in ("track", "model", "done", "busy", "batches", "requests", "spend_usd", "paused", "readiness")}
                        for rd, p in per.items()}}
    if extra:
        out.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--progress", required=True, help="the combined progress file")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--max-hours", type=float, default=72.0)
    ap.add_argument("--dry-run", action="store_true", help="report what would be submitted; no network, no key")
    ap.add_argument("--once", action="store_true", help="one cycle of each, then exit")
    args = ap.parse_args(argv)
    run_dirs = [os.path.abspath(os.path.expanduser(r)) for r in args.run_dir]
    if len(set(run_dirs)) != len(run_dirs):
        raise SystemExit("a run directory is named twice")
    progress = os.path.abspath(os.path.expanduser(args.progress))
    logs = {rd: open(os.path.join(rd, "harvester.log"), "a") for rd in run_dirs}

    def logger(rd: str):
        tag = os.path.basename(rd.rstrip("/"))

        def log(msg: str) -> None:
            line = f"{utc_now()} {msg}"
            print(f"[{tag}] {line}", flush=True)
            logs[rd].write(line + "\n")
            logs[rd].flush()
        return log

    if args.dry_run:
        plan: Dict[str, Any] = {}
        for rd in run_dirs:
            h = TH.TerraHarvester(rd, None, log=logger(rd), dry_run=True)
            try:
                plan[rd] = {"submit": h.cycle(submit=True).get("submit"), "dry_run": _read(os.path.join(rd, "dry_run.json")),
                            "track": {k: h.track.get(k) for k in ("track", "judge", "model", "batch_model_variant", "provider", "max_requests",
                                                                  "max_spend_usd", "truncated_retry_max_tokens", "request")}}
                h.progress(done=False)
            finally:
                h.ledger.close()
        out = write_combined(progress, run_dirs, 1, "dry_run", {"plan": plan})
        print(json.dumps({rd: {"requests": (p.get("dry_run") or {}).get("groups", {}).get("all", {}).get("requests"),
                               "batches": (p.get("dry_run") or {}).get("batches"),
                               "cost_bands": {b: v.get("usd_with_fee") for b, v in ((p.get("dry_run") or {}).get("cost_bands") or {}).items()
                                              if isinstance(v, dict)}} for rd, p in plan.items()}, indent=1))
        print(f"dry run: nothing submitted; plan in {progress} ({out['requests']['total']} requests registered)")
        return 0

    client = H.BatchClient()
    hs = {rd: TH.TerraHarvester(rd, client, log=logger(rd)) for rd in run_dirs}
    for rd, h in hs.items():
        h.log(f"[harvest_all] start run_dir={rd} track={h.track.get('track')} model={h.track.get('model')} pid={os.getpid()} "
              f"with {len(run_dirs) - 1} other run dir(s) in this process")
    t_end = time.time() + args.max_hours * 3600
    cycles = 0
    while True:
        cycles += 1
        busy: Dict[str, bool] = {}
        for rd, h in hs.items():
            try:
                h.cycle(submit=True)
            except Exception:  # noqa: BLE001 -- one bad cycle of one run dir is logged; the loop goes on
                h.log("[harvest_all] cycle error:\n" + traceback.format_exc())
            busy[rd] = h.busy()
            h.progress(done=not busy[rd])
        write_combined(progress, run_dirs, cycles, "running" if any(busy.values()) else "stopping")
        if args.once:
            return 0
        if not any(busy.values()):
            finals = {rd: h.final_state() for rd, h in hs.items()}
            for rd, (code, line) in finals.items():
                hs[rd].log(f"[harvest_all] {line} (exit {code})")
            code = combined_code([c for c, _ in finals.values()])
            write_combined(progress, run_dirs, cycles, {0: "done", 4: "paused", 5: "unfinished"}.get(code, str(code)),
                           {"exit_code": code, "final": {rd: line for rd, (_, line) in finals.items()}})
            return code
        if time.time() > t_end:
            for h in hs.values():
                h.log(f"[harvest_all] --max-hours reached with work left; exiting {H.EXIT_MAX_HOURS} (re-run to resume)")
            write_combined(progress, run_dirs, cycles, "max_hours", {"exit_code": H.EXIT_MAX_HOURS})
            return H.EXIT_MAX_HOURS
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
