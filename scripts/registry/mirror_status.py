#!/usr/bin/env python3
"""The per-cluster INGEST WATERMARK the twin-submit election reads (check 4 of the four-way positive proof: "the cluster's inbox
ingested past the job's end") — DERIVED read-only from the registry, written where the caller says.

The registry keeps no per-cluster inbound watermark of its own: `scan_inbox` moves every `inbox/<host>/…` file into `records/…` (the host
leaves the URI), `state/mirror_state.json` is the OUTBOUND fan-out (the grading host → clusters), and `state/last_run/ingest.json` records the last
`registry ingest [--pull]` as one global fact. So this tool states, per substrate:

  ingested_through_utc  = the later of (a) the last SUCCESSFUL `ingest --pull` utc (`state/last_run/ingest.json`: rc 0, `--pull` in argv —
                          a pull covers every configured cluster's inbox, so the fact holds for each of them) and (b) the newest
                          `record_ingest.arrival_utc` of a record file that yielded a row of a run whose `run.substrate` is that substrate
                          (`run` / `run_state_event` → `record_uri` → `record_ingest`) — the last time a file from that cluster's producers
                          was folded;
  basis                 = which of the two it is, with both values beside it.

A substrate with neither reads `null` — the election then HALTS by name (never a guess). `--label <substrate>=<cluster label>` adds the
site's own label as a second key for the same entry (a `slurm` cluster's `cluster_label`). Output: `{"<substrate|label>":
{"ingested_through_utc", "basis", "last_pull_utc", "last_file_arrival_utc", "computed_utc"}}` to `--out FILE` (tmp + rename) and stdout.
This tool never writes under the registry root; the caller points `--out` at its own state directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, Mapping, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry.records import read_json_or_none, utc_now, write_json_atomic  # noqa: E402
from sisteer.registry.vocab import SUBSTRATES  # noqa: E402


def last_pull_utc(root: str) -> Optional[str]:
    d = read_json_or_none(os.path.join(root, "state", "last_run", "ingest.json"))
    if not isinstance(d, dict):
        return None
    argv = d.get("argv") or []
    if int(d.get("rc", 1) or 0) != 0 or "--pull" not in [str(a) for a in argv]:
        return None
    return str(d.get("utc")) if d.get("utc") else None


def last_file_arrival_by_substrate(conn: sqlite3.Connection) -> Dict[str, Optional[str]]:
    """MAX(record_ingest.arrival_utc) over the record files that yielded a run row or a run_state_event row of runs of each substrate."""
    sql = ("SELECT substrate, MAX(arrival_utc) FROM ("
           "SELECT r.substrate AS substrate, ri.arrival_utc AS arrival_utc FROM run r JOIN record_ingest ri ON ri.file_uri = r.record_uri "
           "UNION ALL "
           "SELECT r.substrate, ri.arrival_utc FROM run_state_event e JOIN run r ON r.run_id = e.run_id "
           "JOIN record_ingest ri ON ri.file_uri = e.record_uri) WHERE arrival_utc IS NOT NULL GROUP BY substrate")
    return {str(sub): str(utc) for sub, utc in conn.execute(sql).fetchall()}


def mirror_status(root: str, labels: Optional[Mapping[str, str]] = None) -> Dict[str, Dict[str, Optional[str]]]:
    pull = last_pull_utc(root)
    with DB.reading(DB.db_path(root)) as conn:
        arrivals = last_file_arrival_by_substrate(conn)
    now = utc_now()
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for sub in SUBSTRATES:
        arr = arrivals.get(sub)
        known = [(v, b) for v, b in ((pull, "last_pull"), (arr, "last_file_arrival")) if v]
        through, basis = max(known, key=lambda vb: vb[0]) if known else (None, "unknown")   # the later of the two; a tie is the pull's
        entry = {"ingested_through_utc": through, "basis": basis, "last_pull_utc": pull, "last_file_arrival_utc": arr, "computed_utc": now}
        out[sub] = entry
        if labels and sub in labels and labels[sub] not in out:
            out[labels[sub]] = dict(entry, substrate=sub)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry-root", default=os.environ.get("REGISTRY_ROOT") or DB.DEFAULT_ROOT)
    ap.add_argument("--out", default=None, help="write the JSON here (tmp + rename); stdout always")
    ap.add_argument("--label", action="append", default=None, metavar="SUBSTRATE=LABEL", help="also key the entry under the site's cluster label")
    args = ap.parse_args(argv)
    labels: Dict[str, str] = {}
    for spec in args.label or ():
        if "=" not in spec:
            print(f"REFUSED: --label {spec!r}: expected SUBSTRATE=LABEL", file=sys.stderr)
            return 2
        k, _, v = spec.partition("=")
        labels[k.strip()] = v.strip()
    if not os.path.exists(DB.db_path(args.registry_root)):
        print(f"REFUSED: no registry database under {args.registry_root}", file=sys.stderr)
        return 2
    status = mirror_status(args.registry_root, labels)
    if args.out:
        write_json_atomic(args.out, status, indent=1, sort_keys=True)
    print(json.dumps(status, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
