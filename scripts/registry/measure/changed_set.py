#!/usr/bin/env python3
"""changed_set.py -- the changed adapter set per read operation, from the receipts and the op records: the changed set is
COMPUTED at every window and every tick, never a surrogate -- `--full` is not one.

    python scripts/registry/measure/changed_set.py [--state DIR] [--root DIR] [--since ISO] [--until ISO] [--owner tick|window] [--json]

Reads ``$STATE/reads/ops/<op_id>.json`` (every operation's record: owner, the ``changed`` adapters WITH THE REASONS the tick computed --
``never read whole (no receipt)``, ``pin moved: <slug> a -> b``, ``module changed: <path> ...`` -- the ``skipped`` ones with theirs, the
``unchanged`` ones, the state history) and ``$STATE/reads/receipts/<op_id>.json`` (what the operation READ: per adapter the pins, each
pin's PUBLISHED record, shards ok / total, the revisions pulled, the re-activations, the refusal). A window's op record carries no
``changed`` map (the operator computed the set with ``cluster-reads changed`` and passed it as ``ADAPTERS=``), so for a window the set
is the receipt's adapters. Per operation: owner, op id, started / ended, state, the changed set and its size, the skipped count, the
reasons by kind (never read / pin moved / module changed), read whole / refused counts, revisions pulled, re-activations, the newest
``change_utc`` among the pins, and the pins' input bytes from the grading host manifests (``<root>/state/store_manifests/<slug>/MANIFEST-<sha8>.json``;
None when a pin has no manifest on this host). Summary: operations, by owner, by state, changed-set size distribution, the reason kinds'
counts, and the adapters most often changed. Stdlib only; read-only; every file unreadable or of another schema is counted, never guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_STATE = os.path.expanduser(os.environ.get("REGISTRY_LANDING_STATE") or "~/.local/state/registry-landing")
DEFAULT_ROOT = os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")
OP_SCHEMA = "reads_op/1"
RECEIPT_SCHEMA = "reads_receipt/1"
#: the reason kinds `reads_protocol.changed_set` spells (the prefix of each reason string)
REASON_KINDS = (("never read whole", "never_read"), ("pin moved", "pin_moved"), ("module changed", "module_changed"),
                ("no longer declared", "component_dropped"))


def _load_dir(path: str, schema: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """``{op_id: doc}`` of every JSON file of ``schema`` under ``path`` and the count of files that were unreadable or of another schema."""
    out: Dict[str, Dict[str, Any]] = {}
    bad = 0
    if not os.path.isdir(path):
        return out, 0
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(path, fn), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            bad += 1
            continue
        if not isinstance(doc, dict) or doc.get("schema") != schema:
            bad += 1
            continue
        op = doc.get("op_id") or (doc.get("op") or {}).get("op_id") or doc.get("window") or fn[:-5]
        out[str(op)] = doc
    return out, bad


def manifest_bytes(root: Optional[str], slug: str, sha8: str) -> Tuple[Optional[int], Optional[int]]:
    """``(bytes, n_files)`` of a generation from the grading host manifest copy, else ``(None, None)``."""
    if not root:
        return None, None
    p = os.path.join(root, "state", "store_manifests", slug, f"MANIFEST-{sha8}.json")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None, None
    if not isinstance(doc, dict):
        return None, None
    b, n = doc.get("bytes"), doc.get("n_files")
    return (int(b) if isinstance(b, (int, float)) else None), (int(n) if isinstance(n, (int, float)) else None)


def reason_kind(reason: str) -> str:
    low = str(reason).lower()
    return next((kind for prefix, kind in REASON_KINDS if low.startswith(prefix)), "other")


def operation_row(op_id: str, op: Optional[Dict[str, Any]], rec: Optional[Dict[str, Any]], root: Optional[str] = None) -> Dict[str, Any]:
    """One operation's row from its op record and / or its receipt (either may be absent: a window has a receipt and a record without a
    ``changed`` map; a tick that found nothing changed has a record and no receipt)."""
    owner = (op or {}).get("owner") or ((rec or {}).get("op") or {}).get("owner")
    changed_map: Dict[str, List[str]] = dict((op or {}).get("changed") or {})
    skipped: Dict[str, str] = dict((op or {}).get("skipped") or {})
    unchanged: List[str] = list((op or {}).get("unchanged") or [])
    adapters: Dict[str, Dict[str, Any]] = dict((rec or {}).get("adapters") or {})
    if not changed_map and adapters:
        changed_map = {a: [] for a in sorted(adapters)}          # a window: the set the operator named is what the receipt read
    kinds: Dict[str, int] = {}
    for reasons in changed_map.values():
        for r in reasons:
            k = reason_kind(r)
            kinds[k] = kinds.get(k, 0) + 1
    pins: Dict[str, str] = {}
    change, published = [], []
    pulled = react = whole = refused = 0
    for e in adapters.values():
        pins.update({str(k): str(v) for k, v in (e.get("pins") or {}).items()})
        for pub in (e.get("published") or {}).values():
            if isinstance(pub, dict):
                if pub.get("change_utc"):
                    change.append(str(pub["change_utc"]))
                if pub.get("published_utc"):
                    published.append(str(pub["published_utc"]))
        pulled += len(e.get("revisions_pulled") or [])
        react += len(e.get("reactivated") or [])
        whole += 1 if e.get("read_whole") else 0
        refused += 1 if e.get("refused") else 0
    total_b = total_n = 0
    missing: List[str] = []
    for slug, sha8 in sorted(pins.items()):
        b, n = manifest_bytes(root, slug, sha8)
        if b is None:
            missing.append(f"{slug}/{sha8}")
        else:
            total_b += b
            total_n += n or 0
    history = list((op or {}).get("history") or [])
    return {"op_id": op_id, "owner": owner, "state": (op or {}).get("state"), "started_utc": (op or {}).get("started_utc") or (op or {}).get("reserved_utc"),
            "ended_utc": (op or {}).get("ended_utc"), "receipt_utc": (rec or {}).get("utc"), "window": (rec or {}).get("window") or (op or {}).get("window"),
            "tree_sha": (rec or {}).get("tree_sha") or (op or {}).get("tree_sha"),
            "changed_set": sorted(changed_map), "changed_set_n": len(changed_map), "reasons": {a: list(v) for a, v in sorted(changed_map.items())},
            "reason_kinds": kinds, "skipped_n": len(skipped), "unchanged_n": len(unchanged), "skipped": skipped,
            "has_record": op is not None, "has_receipt": rec is not None, "receipt_adapters_n": len(adapters), "read_whole_n": whole if rec else None,
            "refused_n": refused if rec else None, "revisions_pulled": pulled if rec else None, "reactivated": react if rec else None,
            "change_utc": max(change) if change else None, "published_utc": max(published) if published else None, "pins": pins,
            "input_bytes": total_b if pins and not missing else None, "input_files": total_n if pins and not missing else None,
            "pins_without_manifest": missing, "history": [{"state": h.get("state"), "utc": h.get("utc")} for h in history if isinstance(h, dict)]}


def collect(state: str, root: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None,
            owner: Optional[str] = None) -> Dict[str, Any]:
    rd = os.path.join(state, "reads")
    ops, bad_ops = _load_dir(os.path.join(rd, "ops"), OP_SCHEMA)
    recs, bad_recs = _load_dir(os.path.join(rd, "receipts"), RECEIPT_SCHEMA)
    rows = []
    for op_id in sorted(set(ops) | set(recs)):
        row = operation_row(op_id, ops.get(op_id), recs.get(op_id), root)
        stamp = str(row.get("started_utc") or row.get("receipt_utc") or "")
        if since and stamp and stamp < since:
            continue
        if until and stamp and stamp >= until:
            continue
        if owner and row.get("owner") != owner:
            continue
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("started_utc") or r.get("receipt_utc") or ""))
    return {"state": state, "reads_dir": rd, "rows": rows, "summary": summarize(rows), "unreadable": {"ops": bad_ops, "receipts": bad_recs}}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_owner: Dict[str, int] = {}
    by_state: Dict[str, int] = {}
    kinds: Dict[str, int] = {}
    per_adapter: Dict[str, int] = {}
    sizes = []
    for r in rows:
        by_owner[str(r.get("owner"))] = by_owner.get(str(r.get("owner")), 0) + 1
        by_state[str(r.get("state"))] = by_state.get(str(r.get("state")), 0) + 1
        for k, v in (r.get("reason_kinds") or {}).items():
            kinds[k] = kinds.get(k, 0) + int(v)
        for a in r.get("changed_set") or []:
            per_adapter[a] = per_adapter.get(a, 0) + 1
        sizes.append(int(r.get("changed_set_n") or 0))
    top = sorted(per_adapter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    return {"operations": len(rows), "by_owner": by_owner, "by_state": by_state, "reason_kinds": kinds,
            "changed_set_n": {"median": statistics.median(sizes) if sizes else None, "max": max(sizes) if sizes else None,
                              "zero": sum(1 for s in sizes if s == 0), "distribution": {str(s): sizes.count(s) for s in sorted(set(sizes))}},
            "with_receipt": sum(1 for r in rows if r.get("has_receipt")), "with_refusal": sum(1 for r in rows if r.get("refused_n")),
            "revisions_pulled": sum(int(r.get("revisions_pulled") or 0) for r in rows), "most_changed": top}


def render(t: Dict[str, Any]) -> str:
    lines = ["| op | owner | started (UTC) | state | changed set | n | skipped | reasons | whole/refused | rev pulled | input MB | change_utc |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted((r.get("reason_kinds") or {}).items())) or "—"
        rcpt = f'{r["read_whole_n"]}/{r["refused_n"]}' if r.get("has_receipt") else "no receipt"
        mb = f'{r["input_bytes"] / 1e6:.0f}' if r.get("input_bytes") is not None else "—"
        lines.append(f'| {r["op_id"]} | {r.get("owner") or "?"} | {r.get("started_utc") or r.get("receipt_utc") or "—"} | {r.get("state") or "—"} | '
                     f'{" ".join(r["changed_set"]) or "—"} | {r["changed_set_n"]} | {r["skipped_n"]} | {kinds} | {rcpt} | '
                     f'{r["revisions_pulled"] if r["revisions_pulled"] is not None else "—"} | {mb} | {r.get("change_utc") or "—"} |')
    s = t["summary"]
    lines.append("")
    lines.append(f'operations {s["operations"]} (by owner {s["by_owner"]}; by state {s["by_state"]}); changed-set size median {s["changed_set_n"]["median"]} '
                 f'max {s["changed_set_n"]["max"]} zero {s["changed_set_n"]["zero"]}; reasons {s["reason_kinds"]}; with receipt {s["with_receipt"]}, '
                 f'with a refusal {s["with_refusal"]}; revisions pulled {s["revisions_pulled"]}; most changed {s["most_changed"][:5]}')
    if t["unreadable"]["ops"] or t["unreadable"]["receipts"]:
        lines.append(f'unreadable or foreign files: ops {t["unreadable"]["ops"]}, receipts {t["unreadable"]["receipts"]}')
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=DEFAULT_STATE, help=f"REGISTRY_LANDING_STATE (default {DEFAULT_STATE})")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the registry root whose state/store_manifests resolve the pins' input bytes")
    ap.add_argument("--since", help="operations started at or after this UTC stamp")
    ap.add_argument("--until", help="operations started before this UTC stamp")
    ap.add_argument("--owner", choices=("tick", "window"), help="only this owner's operations")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    t = collect(a.state, a.root, a.since, a.until, a.owner)
    if a.json:
        print(json.dumps(t, indent=1, default=str))
    else:
        sys.stdout.write(render(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
