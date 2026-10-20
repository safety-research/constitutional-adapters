#!/usr/bin/env python3
"""READ-ONLY check of the run-set drop guard's versioning against a registry
database: which ACCEPTED revisions the guard would HOLD on a replay under the unversioned rule, and which it holds under the versioned
one (``harvest.RUN_SET_DROP_UNDECLARED_SINCE``: the rule applies to a marker observed at or after the stamp its commit landed on main).

The guard compares a marker's run set with the source's accepted head's at activation. On a replay every revision is folded again, so
a revision the live fold ACCEPTED before the rule existed is held when it shrank the set undeclared -- and the head it leaves in place
makes the NEXT revision compare against a stale parent (the R3 bootstrap: 45 such revisions and 6 chained behind them, 2026-10-05).
This tool reproduces that replay from the database alone: per source, the accepted revisions in ``(observed_utc, revision_id)`` order,
each compared against the head the simulated rule left current (the run ids from the ``observation`` rows of both, the declaration from
the marker's parked extra, ``harvest.run_set_drop_comparison``), once with the rule unconditional and once gated by the stamp.

Prints the two hold lists (adapter, source, observed stamp, before/after the gate), the live holds (``accepted = 0`` under the rule's
refusal) with how many predate the stamp -- the versioned rule touches a live row only if that number is not 0 -- and, with
``--expect-ids FILE`` (a JSON list), the symmetric difference against an expected hold list. ``--json OUT`` writes everything. Exit 0
when the versioned replay holds nothing and no live hold predates the stamp; 2 otherwise; 3 when ``--expect-ids`` disagree.

Bounded reads: ``source_revision`` whole (thousands of rows), ``observation`` whole (``source_id, revision_id, observation_key``; ~400k
rows live), nothing else -- never ``row_history``. The database is opened ``?mode=ro`` with ``query_only`` set; nothing is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import records as R  # noqa: E402

DEFAULT_DB = os.path.join(os.environ.get("REGISTRY_ROOT") or os.path.expanduser("~/registry"), "registry.sqlite")


def load(db_path: str):
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    try:
        revs = [dict(r) for r in conn.execute("SELECT revision_id, source_id, adapter, observed_utc, accepted, refusal, extra_json, "
                                              "superseded_by_revision_id FROM source_revision")]
        ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for sid, rev, key in conn.execute("SELECT source_id, revision_id, observation_key FROM observation"):
            ids[(str(sid), str(rev))].update(H.run_ids_of_observation_keys([key]))
    finally:
        conn.close()
    return revs, ids


def simulate(revs: Sequence[dict], ids: Dict[Tuple[str, str], Set[str]], since: Optional[str]) -> List[dict]:
    """The replay's holds under the rule gated by ``since`` (None: unconditional): per source, the accepted chain in order, each revision
    compared against the head the simulation left current -- the comparison :meth:`harvest.Ingestor._undeclared_run_drop` makes."""
    held: List[dict] = []
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for r in revs:
        if int(r["accepted"] or 0):
            by_source[str(r["source_id"])].append(r)
    for sid in sorted(by_source):
        head: Optional[str] = None
        for r in sorted(by_source[sid], key=lambda x: (str(x["observed_utc"]), str(x["revision_id"]))):
            rev = str(r["revision_id"])
            applies = since is None or H.run_set_drop_rule_applies(str(r["observed_utc"]))
            if head is not None and applies:
                prev_ids, new_ids = ids.get((sid, head), set()), ids.get((sid, rev), set())
                if prev_ids and (prev_ids - new_ids) and len(new_ids) < len(prev_ids):
                    try:
                        extra = json.loads(r["extra_json"] or "{}")
                    except ValueError:
                        extra = {}
                    declaration, _ = H.declared_run_drops_of(extra if isinstance(extra, dict) else None)
                    cmp = H.run_set_drop_comparison(prev_ids, new_ids, declaration)
                    if cmp["tripped"]:
                        held.append({"revision_id": rev, "source_id": sid, "adapter": r["adapter"], "observed_utc": r["observed_utc"],
                                     "compared_with": head, "runs_previous": cmp["runs_previous"], "runs_now": cmp["runs_now"],
                                     "runs_dropped_undeclared": cmp["runs_dropped_undeclared"],
                                     "before_stamp": not H.run_set_drop_rule_applies(str(r["observed_utc"]))})
                        continue                     # the head stays, as the harvester's hold leaves it
            head = rev
    return held


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB, help=f"the registry database (opened read-only; default {DEFAULT_DB})")
    ap.add_argument("--since", default=H.RUN_SET_DROP_UNDECLARED_SINCE, help="the versioned rule's stamp (default: the code's constant)")
    ap.add_argument("--expect-ids", help="a JSON list of revision ids the UNVERSIONED replay is expected to hold (the R3 evidence's 51)")
    ap.add_argument("--json", dest="json_out", help="write the full result here")
    args = ap.parse_args(argv)
    since = args.since
    revs, ids = load(args.db)
    unversioned = simulate(revs, ids, None)
    versioned = simulate(revs, ids, since) if since == H.RUN_SET_DROP_UNDECLARED_SINCE else _simulate_with(revs, ids, since)
    live_holds = [r for r in revs if not int(r["accepted"] or 0) and H.RUN_SET_DROP_KEY in str(r["refusal"] or "")]
    live_before = [r for r in live_holds if R.utc_to_epoch_ms(str(r["observed_utc"])) < R.utc_to_epoch_ms(since)]
    out = {"db": os.path.abspath(args.db), "since": since, "n_revisions": len(revs), "n_accepted": sum(int(r["accepted"] or 0) for r in revs),
           "unversioned_holds": unversioned, "n_unversioned_holds": len(unversioned),
           "n_unversioned_before_stamp": sum(1 for h in unversioned if h["before_stamp"]),
           "versioned_holds": versioned, "n_versioned_holds": len(versioned),
           "live_holds": [{k: r[k] for k in ("revision_id", "source_id", "adapter", "observed_utc")} for r in live_holds],
           "n_live_holds": len(live_holds), "n_live_holds_before_stamp": len(live_before)}
    print(f"run-set drop guard versioning check: {out['n_accepted']} accepted revisions of {out['n_revisions']}; rule stamp {since}")
    print(f"  unversioned replay holds {len(unversioned)} accepted revision(s) ({out['n_unversioned_before_stamp']} observed before the stamp, "
          f"{len(unversioned) - out['n_unversioned_before_stamp']} after it, chained behind a stale head):")
    for h in unversioned:
        print(f"    {h['observed_utc']} {h['revision_id']} {h['adapter']:<22} {h['source_id'][:60]:<60} {h['runs_previous']}->{h['runs_now']} "
              f"({h['runs_dropped_undeclared']} undeclared) vs {h['compared_with']} {'BEFORE' if h['before_stamp'] else 'after'}")
    print(f"  versioned replay holds {len(versioned)} accepted revision(s)" + ("" if not versioned else ":"))
    for h in versioned:
        print(f"    {h['observed_utc']} {h['revision_id']} {h['adapter']} {h['source_id'][:60]} {h['runs_previous']}->{h['runs_now']} vs {h['compared_with']}")
    print(f"  live holds under the rule: {len(live_holds)}, of them observed before the stamp: {len(live_before)} (must be 0: the versioned rule "
          f"admits a pre-stamp marker, so a pre-stamp live hold would be a live fold change)")
    rc = 0 if not versioned and not live_before else 2
    if args.expect_ids:
        with open(args.expect_ids, encoding="utf-8") as fh:
            expect = set(json.load(fh))
        got = {h["revision_id"] for h in unversioned}
        out["expect"] = {"n": len(expect), "missing": sorted(expect - got), "unexpected": sorted(got - expect)}
        print(f"  expected {len(expect)} unversioned hold(s): {len(expect & got)} found, {len(expect - got)} missing, {len(got - expect)} unexpected")
        for rid in out["expect"]["missing"]:
            print(f"    missing   {rid}")
        for rid in out["expect"]["unexpected"]:
            print(f"    unexpected {rid}")
        if expect != got:
            rc = 3
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"  written: {args.json_out}")
    return rc


def _simulate_with(revs, ids, since: str) -> List[dict]:
    """``simulate`` under another stamp (``--since``): the rule's predicate re-pointed for the run."""
    saved = H.RUN_SET_DROP_UNDECLARED_SINCE
    H.RUN_SET_DROP_UNDECLARED_SINCE = since
    try:
        return simulate(revs, ids, since)
    finally:
        H.RUN_SET_DROP_UNDECLARED_SINCE = saved


if __name__ == "__main__":
    sys.exit(main())
