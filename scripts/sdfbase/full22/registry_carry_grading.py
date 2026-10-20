#!/usr/bin/env python3
"""Carry a retired run's GRADING records to its successor.

A pre-fix static run (registered 8192 / no pass before static-reg-fix 158242e5d) is re-registered from its existing store at the drawn rung
by the closer (registry_close_stores.py --successor-token-cap) and its row filed `superseded` naming the successor. The store is durable and
was graded once; nothing is recomputed. This tool re-files, under the SUCCESSOR id, the retired run's `judgment`, `item_result`, `run_metric`
and `run_degeneration` records and the grader's own `run_state judged` + post-judgment `run_patch` — payloads EQUAL except the ids and the
run_id they bind: `run_id` -> the successor; `judgment_id` -> the identity form for a NON-claimed judgment of the successor
(identity.judgment_id(successor, judge_id): no session, no claim). The re-filed judgment is a NEW judgment record carrying its provenance in
extra: carried_from_run_id, carried_from_judgment_id, reason, decision. Runs where the retired run's records are readable (the registry
host's records root: ~/registry/records/<campaign>/<run dir>/<seq>-<type>.jsonl[.gz]) and writes through the same RecordWriter the graders
use (run-scoped, push --final). Refuses by name: no complete judgment, more than one complete judgment, a retired run that is not
`superseded` by this successor in the spool (the lineage must be filed first), a successor whose dir already carries the carried judgment
(idempotent: nothing written twice). Dry by default: --write emits.

usage: registry_carry_grading.py --pairs <tsv from the closer's --pairs-out> [--records-root ~/registry/records] [--spool ~/registry/records]
                                 [--campaign sdfbase_full22] [--code-sha SHA] [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

REASON = "token_cap_identity_relabel"
DECISION = "D1900"
CARRIED_TYPES = ("judgment", "item_result", "run_metric", "run_degeneration", "run_state", "run_patch")
SCRIPT = "scripts/sdfbase/full22/registry_carry_grading.py"


def _say(msg: str) -> None:
    print(f"[carry-grading] {msg}", file=sys.stderr, flush=True)


def read_pairs(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#") or ln.startswith("cancelled\t"):
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                raise SystemExit(f"{path}: a pairs line needs retired<TAB>successor: {ln[:80]!r}")
            out.append((parts[0], parts[1]))
    return out


def records_of(records_root: str, campaign: str, run_id: str) -> List[Tuple[int, str, dict]]:
    """(seq, type, payload) for every record of the run in its records directory, file order; [] when the directory is absent."""
    from sisteer.registry import identity, records as R
    d = os.path.join(records_root, campaign, identity.run_dir_name(run_id)); out: List[Tuple[int, str, dict]] = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m:
            continue
        for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
            p = rec.get("payload") if isinstance(rec, dict) else None
            if isinstance(p, dict) and p.get("run_id") == run_id:
                out.append((int(m.group("seq")), str(m.group("type")), p))
    return out


def carried_payloads(old_recs: List[Tuple[int, str, dict]], *, old: str, new: str, judge_id: str, old_jid: str, new_jid: str) -> List[Tuple[str, dict]]:
    """The successor's records from the retired run's: the grading records (every type in CARRIED_TYPES written AT or AFTER the judgment's
    sequence, plus every judgment / item_result / run_metric / run_degeneration wherever it sits), ids rebound, payloads otherwise equal."""
    jseq = min(s for s, t, p in old_recs if t == "judgment" and p.get("judgment_id") == old_jid)
    out: List[Tuple[str, dict]] = []
    for seq, t, p in old_recs:
        if t not in CARRIED_TYPES:
            continue
        # only the COMPLETE judgment's rows travel — an abandoned attempt's judgment and the item / metric /
        # degeneration rows bound to it stay behind (a row bound to another judgment id, or a judgment record that is not the complete one)
        if t == "judgment" and p.get("judgment_id") != old_jid:
            continue
        if t in ("item_result", "run_metric", "run_degeneration") and p.get("judgment_id") not in (None, "", old_jid):
            continue
        if t in ("run_state", "run_patch") and seq < jseq:
            continue                                     # the job's own lifecycle records (running / generated / the job patch) are the successor's own
        if t == "run_state" and p.get("state") != "judged":
            continue
        q = json.loads(json.dumps(p))                    # a deep copy; the payload stays otherwise equal
        q["run_id"] = new
        if q.get("judgment_id") == old_jid:
            q["judgment_id"] = new_jid
        if t == "judgment":
            extra = q.get("extra")
            extra = dict(extra) if isinstance(extra, dict) else ({} if extra in (None, "") else {"_prior_extra": extra})
            extra.update({"carried_from_run_id": old, "carried_from_judgment_id": old_jid, "reason": REASON, "decision": DECISION})
            q["extra"] = extra
            q.pop("claimed_by_session", None); q.pop("claimed_utc", None)     # a non-claimed judgment: its id form carries no session / claim
        out.append((t, q))
    return out


def carry_one(old: str, new: str, *, records_root: str, spool: str, campaign: str, producer: dict, write: bool) -> str:
    from sisteer.registry import emit, identity
    old_recs = records_of(records_root, campaign, old)
    if not old_recs:
        return f"REFUSED {old[:40]}…: no records under {records_root}/{campaign}/{identity.run_dir_name(old)} — run this where the registry host's records are"
    judgments = [p for _s, t, p in old_recs if t == "judgment"]
    complete = [p for p in judgments if str(p.get("status") or "") == "complete" and not p.get("superseded_by_judgment_id")]
    if len(complete) != 1:
        return f"REFUSED {old[:40]}…: {len(complete)} complete judgment(s) ({len(judgments)} total) — exactly one is carried; a grading question first"
    jp = complete[0]; old_jid = str(jp["judgment_id"]); judge_id = str(jp["judge_id"])
    sup = [p for _s, t, p in old_recs if t == "run_state" and p.get("state") == "superseded" and (p.get("extra") or {}).get("superseded_by_run_id") == new]
    if not sup:
        return f"REFUSED {old[:40]}…: the retired run is not filed `superseded` by {new[:40]}… in {records_root} — the closer's lineage comes first"
    new_jid = identity.judgment_id(new, judge_id)
    new_recs = records_of(spool, campaign, new)
    if not any(t == "run" for _s, t, p in new_recs) and not any(t == "run" for _s, t, p in records_of(records_root, campaign, new)):
        return f"REFUSED {new[:40]}…: the successor has no `run` record in {spool} / {records_root} — register and close it first"
    payloads = carried_payloads(old_recs, old=old, new=new, judge_id=judge_id, old_jid=old_jid, new_jid=new_jid)
    counts: Dict[str, int] = {}
    for t, _q in payloads:
        counts[t] = counts.get(t, 0) + 1
    # what the successor ALREADY carries, per type, bound to the carried judgment — a write interrupted after
    # the judgment file left a successor with a judgment and no rows, which the old "already carries judgment" check skipped as idempotent
    present: Dict[str, int] = {}
    for _s, t, p in new_recs:
        if t == "judgment" and p.get("judgment_id") == new_jid:
            present[t] = present.get(t, 0) + 1
        elif t in ("item_result", "run_metric", "run_degeneration") and p.get("judgment_id") == new_jid:
            present[t] = present.get(t, 0) + 1
        elif t == "run_state" and p.get("state") == "judged" and (p.get("extra") or {}).get("carried_from_run_id") == old:
            present[t] = present.get(t, 0) + 1
        elif t == "run_patch" and (p.get("extra") or {}).get("carried_from_run_id") == old:
            present[t] = present.get(t, 0) + 1
    missing = {t: n for t, n in counts.items() if present.get(t, 0) < n}
    if not missing:
        return f"skip {new[:40]}…: already carries {counts} for judgment {new_jid} (idempotent; every expected record type verified)"
    if not write:
        return (f"DRY {old[:40]}… -> {new[:40]}…: would carry {counts} (judgment {old_jid} -> {new_jid}, judge {judge_id[:12]}…)"
                + (f"; REPLAY of the missing {sorted(missing)} on a partially carried successor" if present else ""))
    w = emit.RecordWriter(spool, campaign, new, producer, push_script=os.environ.get(emit.PUSH_SCRIPT_ENV))
    by_type: Dict[str, List[dict]] = {}
    order: List[str] = []
    for t, q in payloads:                               # one file per type, the retired run's order of types kept (item_result before judgment, as the grader wrote them)
        if t not in missing:
            continue                                     # (e): replay ONLY what the successor lacks
        if t in ("run_state", "run_patch"):
            q.setdefault("extra", {})
            if isinstance(q["extra"], dict):
                q["extra"].setdefault("carried_from_run_id", old)
        if t not in by_type:
            by_type[t] = []; order.append(t)
        by_type[t].append(q)
    for t in order:
        w.write_many(t, by_type[t])
    rc = w.push(final=True)
    if rc != 0:
        return f"REFUSED {old[:40]}… -> {new[:40]}…: wrote {sorted(order)} but the push returned rc={rc} — NOT carried until a push lands (re-run replays nothing, pushes the spooled records)"
    verb = "replayed" if present else "carried"
    return f"{verb} {old[:40]}… -> {new[:40]}…: {dict((t, counts[t]) for t in order)} (judgment {old_jid} -> {new_jid}; push rc={rc})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True); ap.add_argument("--records-root", default=os.path.expanduser("~/registry/records"))
    ap.add_argument("--spool", default=None, help="where the successor's records are written (default: the records root — the registry host's own spool)")
    ap.add_argument("--campaign", default="sdfbase_full22"); ap.add_argument("--code-sha", default=None); ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    from sisteer.registry import emit
    producer = emit.producer_block(SCRIPT, code_sha=a.code_sha)
    spool = a.spool or a.records_root
    rc = 0
    for old, new in read_pairs(a.pairs):
        msg = carry_one(old, new, records_root=a.records_root, spool=spool, campaign=a.campaign, producer=producer, write=a.write)
        _say(msg)
        if msg.startswith("REFUSED"):
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
