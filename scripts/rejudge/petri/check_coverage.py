#!/usr/bin/env python3
"""Check that the verdict run directories PARTITION the enumerated judge inputs: every key in exactly one run directory.

    PYTHONPATH=src:scripts python scripts/rejudge/petri/check_coverage.py --enum ENUM --run-dir SMOKE --run-dir PILOT --run-dir FULL

The smoke, the pilot and the full run are built disjoint (``build_manifests.py --exclude-keys``) from key lists, and a run directory
is immutable once written: rebuild the pilot with another cut and forget the full run, and some inputs are either sent twice (the filing
then refuses them as terminal in two ledgers, after they were paid for twice) or never sent (their runs stay partial). Run this before
the full run submits, and after any rebuild. Reads the manifests only (no network, no ledger); a probe run directory (``id_suffix``) is
refused. Exit 0 when the run directories partition the enumeration, 1 otherwise; the report names the overlaps and the gaps.

A follow-up run directory, which re-sends exactly the keys another run directory holds as infrastructure failures (a failed batch's,
after its re-asks; ``file_records.read_outcomes`` lets the follow-up's outcome settle them), overlaps that run directory by design: this
check reads manifests, not outcomes, so run it on the first-attempt run directories, not on a follow-up.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
from typing import Dict, Iterable, List, Set

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402


def enum_keys(enum_dir: str) -> Set[str]:
    out = set()
    with gzip.open(os.path.join(enum_dir, "inputs.jsonl.gz"), "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if "error" not in r:
                out.add(r["key"])
    return out


def run_keys(run_dir: str) -> Set[str]:
    """Every custom_id the run directory's manifests hold (first attempts and re-asks alike)."""
    tr = BF.read_track(run_dir)
    if tr.get("id_suffix"):
        raise ValueError(f"{run_dir} is a probe (id_suffix {tr['id_suffix']!r}); probes are not verdict ledgers")
    return {c for k in BF.list_manifests(run_dir) for c in BF.read_manifest(run_dir, k)["custom_ids"]}


def coverage(keys: Set[str], run_dirs: Iterable[str]) -> Dict[str, object]:
    where: Dict[str, List[str]] = collections.defaultdict(list)
    per = {}
    for rd in run_dirs:
        ks = run_keys(rd)
        per[rd] = len(ks)
        for k in ks:
            where[k].append(rd)
    dup = {k: v for k, v in where.items() if len(v) > 1}
    missing = sorted(keys - set(where))
    extra = sorted(set(where) - keys)
    return {"enumerated": len(keys), "per_run_dir": per, "in_two_or_more": len(dup), "in_none": len(missing), "not_enumerated": len(extra),
            "partition": not dup and not missing and not extra,
            "examples": {"in_two_or_more": [(k[:16], v) for k, v in sorted(dup.items())[:5]], "in_none": [k[:16] for k in missing[:5]],
                         "not_enumerated": [k[:16] for k in extra[:5]]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enum", required=True)
    ap.add_argument("--run-dir", action="append", required=True)
    args = ap.parse_args(argv)
    rep = coverage(enum_keys(args.enum), [os.path.abspath(os.path.expanduser(r)) for r in args.run_dir])
    print(json.dumps(rep, indent=1))
    print("PARTITION: every enumerated input is in exactly one run directory" if rep["partition"] else
          "NOT A PARTITION: rebuild the run directories that overlap or leave inputs out before anything submits")
    return 0 if rep["partition"] else 1


if __name__ == "__main__":
    sys.exit(main())
