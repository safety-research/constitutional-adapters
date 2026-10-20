#!/usr/bin/env python3
"""reproduce/lib/grant_ledger.py — the grading host's durable, append-only ledgers (stdlib only).

Two files per model under `$SISTEER_STATE_DIR/<model>/`:
  submissions.jsonl   every sbatch / twin / cancel / watcher / reconcile event (`repro_submitted`, `reconcile.sh`)
  attempts.jsonl      the twin election's transaction log (§3.9): `attempt` (an attempt exists: logical unit, cluster,
                      job id), `observed` (the attempt is RUNNING = waiting on its grant), `winner` (the COMMITTED
                      election — written, fsync'd, BEFORE any remote GRANT/DENY), `granted` / `denied` (the remote
                      sentinel landed), `grant_retired` (a dead, provably unregistered winner's GRANT tombstoned before
                      its replacement's), `registered` (the winner's run id(s) are known), `halted` (an inconsistency —
                      fail closed), `closed` (the unit is done).
Every append is one JSON line + fsync, so a restarted coordinator replays exactly what was committed.

    python grant_ledger.py append --file F --json '{...}'
    python grant_ledger.py state  --file F            # the derived election state per logical unit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ledger:
    def __init__(self, path: str):
        self.path = path

    def append(self, rec: dict) -> dict:
        rec = dict(rec)
        rec.setdefault("utc", utc_now())
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        line = json.dumps(rec, sort_keys=True) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return rec

    def records(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    # a torn last line (crash mid-write) is ignored; anything committed before it stands
                    continue
        return out

    # ---- the election view -----------------------------------------------------------------------------------------
    def state(self) -> Dict[str, dict]:
        """{logical_work_id: {winner, attempts: {attempt_id: {...}}, granted: [...], denied: [...], registered: {...},
        halted: str|None, closed: bool}} — a pure function of the records."""
        st: Dict[str, dict] = {}

        def unit(lw: str) -> dict:
            return st.setdefault(lw, {"winner": None, "winner_n": 0, "attempts": {}, "granted": [], "denied": [], "retired": [],
                                      "registered": {}, "halted": None, "closed": False, "pool_jobs": [], "torn_down": False, "torn_down_jobs": [],
                                      "retiring_jobs": []})

        for r in self.records():
            ev = r.get("event")
            lw = r.get("logical_work_id")
            if not lw:
                continue
            u = unit(lw)
            if ev == "attempt":
                # `spec` (+ its candidate `clusters`) is what a detached `elect` or `reconcile` re-issues verbatim; `spool` / `pending_dir`
                # are the producer paths AS SUBMITTED, which the fenced proof reads (findings 2 and 3)
                u["attempts"][r["attempt_id"]] = {"cluster": r.get("cluster"), "job_id": r.get("job_id"), "job_name": r.get("job_name"),
                                                  "utc": r.get("utc"), "kind": r.get("kind", "target"), "spec": r.get("spec"),
                                                  "registers": r.get("registers"),      # the registration contract (item (iv)); absent on older records
                                                  "twin": bool(r.get("twin")), "spool": r.get("spool"), "pending_dir": r.get("pending_dir"),
                                                  "nonce": r.get("nonce"), "identified_by": r.get("identified_by")}   # The reservation nonce + how the id was obtained (launch | nonce)
            elif ev == "observed":
                u["attempts"].setdefault(r["attempt_id"], {})["observed_utc"] = r.get("utc")
            elif ev == "winner":
                u["winner"] = r["attempt_id"]; u["winner_n"] += 1
            elif ev == "granted":
                u["granted"].append(r["attempt_id"])
            elif ev == "denied":
                u["denied"].append(r["attempt_id"])
            elif ev == "grant_retired":        # a dead, provably unregistered winner's GRANT was tombstoned before a re-grant
                u["retired"].append(r["attempt_id"])
            elif ev == "registered":
                u["registered"][r["attempt_id"]] = r.get("run_ids", [])
            elif ev == "halted":
                u["halted"] = r.get("reason", "halted")
            elif ev == "unhalted":
                u["halted"] = None
            elif ev == "closed":
                u["closed"] = True
            elif ev == "pool_member":
                u["pool_jobs"].append({"cluster": r.get("cluster"), "job_id": r.get("job_id")})
            elif ev == "retiring":      # 31d: the member was marked RETIRING (its <jobid>.RETIRE written); it self-exits after draining
                u["retiring_jobs"].append(str(r.get("job_id")))
            elif ev == "torn_down":     # the pool member's <jobid>.STOP was written: that job's COMPLETED is its teardown, not a result
                u["torn_down"] = True
                u["torn_down_jobs"].append({"cluster": r.get("cluster"), "job_id": str(r.get("job_id"))})
        return st

    def commit_winner(self, logical_work_id: str, attempt_id: str, **meta) -> dict:
        return self.append({"event": "winner", "logical_work_id": logical_work_id, "attempt_id": attempt_id, **meta})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("append"); p.add_argument("--file", required=True); p.add_argument("--json", required=True)
    p = sub.add_parser("state"); p.add_argument("--file", required=True)
    a = ap.parse_args(argv)
    L = Ledger(a.file)
    if a.cmd == "append":
        L.append(json.loads(a.json))
    else:
        print(json.dumps(L.state(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
