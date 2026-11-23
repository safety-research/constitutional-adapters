#!/usr/bin/env python3
"""Carry a finished run directory's verdicts (the ARM: e.g. the judge comparison's run of the same model) into a full run as a REUSE run
directory, after proving each one answers the request the full run would send (prepare only: no network, the arm is only read).

    PYTHONPATH=src:scripts python scripts/rejudge/terra/reuse.py --source ARM_RUN_DIR --scope SCOPE/cpi \
        --judge minimax-m3 --family cpi --out REUSE/cpi

A key is carried only when ALL of these hold, else it is left for the full run (listed in ``reuse.json`` with the reason):
  * the arm sent the judge's model, endpoint and parser (``track.json``), its harvest is over (no ``PAUSED``, no batch pending or in
    flight, every request terminal) and its ledger has no write-ahead log left (it is opened ``immutable``: nothing of it is touched);
  * the key is in the scope, and the arm's ORIGINAL body for it (attempt 0, read through ``batchfile.read_body``, which checks the
    manifest's sha256) is BYTE-IDENTICAL to the body :func:`rejudge.terra.track.entry_for` builds for the full run under the judge's
    config entry -- same text, rubric, schema, request variant, reasoning setting and cap;
  * its attempt sequence is the one the full run's harvester would have produced from the same answers: each outcome but the last
    re-asked and the last terminal under the track's re-ask budget (``track.RETRY``, counted per class as the track's harvester counts)
    and the entry's cut-call policy (a cut call re-asked at ``truncated_retry_max_tokens``, or terminal at once without one), with no
    account / held outcome anywhere in it;
  * every re-ask the arm sent for it is that same body, its ``max_tokens`` the entry's retry cap after a cut call and unchanged after
    anything else.

The REUSE directory holds ``state.sqlite`` (the carried keys' ``requests`` and ``attempts`` rows copied verbatim: outcome, verdict,
error, usage, cost, batch id, provider, harvest time), ``track.json`` (marked ``reuse_only``; no manifests, so a harvester pointed at it
has nothing to submit), ``keys.txt`` (for ``build_manifests.py --exclude-keys``), ``reuse.json`` and ``provenance/`` (the arm's
track.json, manifests and progress, for the record). The filing reads it like any run directory (``file_records.py --run-dir``).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import shutil
import sqlite3
import sys
from typing import Any, Dict, List, Mapping, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common.state import Ledger, utc_now  # noqa: E402
from rejudge.terra import harvest as TH  # noqa: E402
from rejudge.terra import judges as J  # noqa: E402
from rejudge.terra import track as T  # noqa: E402

LIVE = ("pending", "balance_queued", "submitted", "submit_unknown")


def open_source(run_dir: str) -> sqlite3.Connection:
    """The arm's ledger, opened without touching it: ``immutable=1`` (no locks, no -shm) when its WAL is empty, else refused."""
    path = os.path.join(run_dir, "state.sqlite")
    wal = path + "-wal"
    if os.path.exists(wal) and os.path.getsize(wal) > 0:
        raise SystemExit(f"refused: {wal} is not empty (a harvester may still be writing {run_dir}); reuse a finished arm only")
    con = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def source_bodies(run_dir: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """custom_id -> {manifest attempt: the body that manifest sent} over every manifest of the arm (sha256-checked)."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = collections.defaultdict(dict)
    for key in BF.list_manifests(run_dir):
        man = BF.read_manifest(run_dir, key)
        for r in BF.body_requests(BF.read_body(run_dir, man)):
            out[r["custom_id"]][int(man.get("attempt") or 0)] = r["body"]
    return out


def scope_rows(scope: str, keys: set) -> Dict[str, Dict[str, Any]]:
    out = {}
    with gzip.open(os.path.join(scope, "texts.jsonl.gz"), "rt") as fh:
        for line in fh:
            if line[7:71] in keys:
                x = json.loads(line)
                out.setdefault(x["k"], x)
    return out


def policy_consistent(outcomes: List[str], retry: Mapping[str, int], retry_cap: Optional[int]) -> bool:
    """Would the full run's harvester have produced exactly this attempt sequence from the same answers? Every outcome but the last is
    re-asked and the last is terminal: a verdict always ends it; a counted outcome ends it when its class's attempts exceed its re-asks
    (``TerraHarvester._record``); a cut call with re-asks left but no higher cap configured ends it at once (``build_retries``
    finalizes it); an account / held outcome (never counted) is refused outright."""
    if not outcomes:
        return False
    counts: collections.Counter = collections.Counter()
    for i, o in enumerate(outcomes):
        cls = TH.attempt_class(o)
        if cls == "never":
            return False
        if o == "verdict":
            terminal = True
        else:
            counts[cls] += 1
            terminal = counts[cls] > int(retry.get(o, 0))
            if not terminal and o == "truncated" and not retry_cap:
                terminal = True
        if terminal != (i == len(outcomes) - 1):
            return False
    return True


def build(source: str, scope: str, judge: str, family: str, out: str) -> Dict[str, Any]:
    source = os.path.abspath(os.path.expanduser(source))
    out = os.path.abspath(os.path.expanduser(out))
    if os.path.exists(os.path.join(out, "state.sqlite")):
        raise SystemExit(f"{out} already holds a ledger: a reuse directory is built once")
    strack = BF.read_track(source)
    cfg = J.get(judge)
    fc = J.family_cfg(judge, family)
    want = {"model": cfg["openrouter_model"], "endpoint": T.module(family).ENDPOINT, "parser": J.parser_spec(judge, family), "family": family}
    got = {k: strack.get(k) for k in want}
    if got != want:
        raise SystemExit(f"refused: the arm sent {got}, the judge entry {judge!r} sends {want}")
    if os.path.exists(os.path.join(source, "PAUSED")):
        raise SystemExit(f"refused: {source} is PAUSED")
    con = open_source(source)
    live = con.execute(f"SELECT COUNT(*) FROM batches WHERE state IN ({','.join('?' * len(LIVE))})", LIVE).fetchone()[0]
    nonterm = con.execute("SELECT COUNT(*) FROM requests WHERE terminal = 0").fetchone()[0]
    if live or nonterm:
        raise SystemExit(f"refused: {source} is still moving ({live} batches live, {nonterm} requests not terminal)")
    rows = {r["custom_id"]: dict(r) for r in con.execute("SELECT * FROM requests")}
    attempts: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in con.execute("SELECT * FROM attempts ORDER BY custom_id, attempt"):
        attempts[r["custom_id"]].append(dict(r))
    bodies = source_bodies(source)
    texts = scope_rows(scope, set(rows))
    rq = J.request_settings(judge, family)
    retry_cap = fc.get("truncated_retry_max_tokens")
    carried, refused = [], collections.Counter()
    refused_keys: Dict[str, str] = {}
    for cid, r in sorted(rows.items()):
        why = None
        sent = bodies.get(cid) or {}
        if cid not in texts:
            why = "not_in_scope"
        elif 0 not in sent:
            why = "no_original_body"
        else:
            mine = T.entry_for(family, texts[cid], rq)
            outs = [a["outcome"] for a in attempts[cid]]
            if BF.serialize_request(cid, sent[0]) != BF.serialize_request(mine["custom_id"], mine["body"]):
                why = "original_body_differs"
            elif not policy_consistent(outs, T.RETRY, retry_cap):
                why = "attempts_not_the_full_runs_policy"
            elif sorted(sent) != list(range(len(outs))) or [a["attempt"] for a in attempts[cid]] != list(range(1, len(outs) + 1)):
                why = "attempt_bodies_incomplete"
            else:
                cap = int(mine["body"]["max_tokens"])
                for k in range(1, len(outs)):          # sent[k] is ledger attempt k + 1, re-asked after outcome outs[k - 1]
                    cap = int(retry_cap) if outs[k - 1] == "truncated" else cap
                    if sent[k] != dict(mine["body"], max_tokens=cap):
                        why = "re_ask_body_differs"
                        break
        if why:
            refused[why] += 1
            refused_keys[cid] = why
            continue
        carried.append(cid)
    con.close()
    os.makedirs(out, exist_ok=True)
    os.chmod(out, 0o700)
    led = Ledger(out).writer()
    cols_r = list(rows[carried[0]].keys()) if carried else []
    for cid in carried:
        r = rows[cid]
        led.conn.execute(f"INSERT INTO requests({','.join(cols_r)}) VALUES ({','.join('?' * len(cols_r))})", [r[c] for c in cols_r])
        for a in attempts[cid]:
            cols_a = list(a.keys())
            led.conn.execute(f"INSERT INTO attempts({','.join(cols_a)}) VALUES ({','.join('?' * len(cols_a))})", [a[c] for c in cols_a])
    led.conn.commit()
    led.event("REUSED", f"{len(carried)} terminal outcomes carried from {source} (bodies byte-identical to the full run's); "
                        f"{sum(refused.values())} left for the full run {dict(refused)}")
    outcomes = collections.Counter(rows[c]["outcome"] for c in carried)
    led.close()
    spec = {"track": J.track_name(judge, family) + "_reuse", "family": family, "judge": judge, "model": cfg["openrouter_model"],
            "endpoint": want["endpoint"], "parser": want["parser"], "request": rq, "reuse_only": True,
            "reused_from": {"run_dir": source, "track": strack.get("track"), "built_utc": strack.get("built_utc"),
                            "keys_file": strack.get("keys_file")}}
    if retry_cap:
        spec["truncated_retry_max_tokens"] = int(retry_cap)
    BF.write_track(out, spec)
    with open(os.path.join(out, "keys.txt"), "w") as fh:
        fh.write("".join(c + "\n" for c in carried))
    prov = os.path.join(out, "provenance")
    os.makedirs(os.path.join(prov, "manifests"), exist_ok=True)
    for f in ("track.json", "progress.json", "dry_run.json"):
        if os.path.exists(os.path.join(source, f)):
            shutil.copy2(os.path.join(source, f), os.path.join(prov, f))
    for key in BF.list_manifests(source):
        shutil.copy2(os.path.join(source, "manifests", key + ".json"), os.path.join(prov, "manifests", key + ".json"))
    rep = {"built_utc": utc_now(), "source": source, "scope": os.path.abspath(scope), "judge": judge, "family": family, "out": out,
           "source_requests": len(rows), "carried": len(carried), "carried_outcomes": dict(outcomes), "left_for_full_run": dict(refused),
           "left_keys": refused_keys, "request_settings": rq, "truncated_retry_max_tokens": retry_cap,
           "check": "attempt-0 body byte-identical to track.entry_for under the judge entry; the attempt sequence the full run's policy "
                    "would produce (track.RETRY per class, the entry's cut-call cap); each re-ask the same body at the cap that policy sends"}
    with open(os.path.join(out, "reuse.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="a finished run directory of the same judge protocol (read only)")
    ap.add_argument("--scope", required=True)
    ap.add_argument("--judge", required=True, choices=sorted(J.JUDGES))
    ap.add_argument("--family", required=True, choices=J.FAMILIES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rep = build(args.source, args.scope, args.judge, args.family, args.out)
    print(json.dumps({k: v for k, v in rep.items() if k != "left_keys"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
