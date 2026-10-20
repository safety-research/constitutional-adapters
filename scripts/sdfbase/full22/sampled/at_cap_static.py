#!/usr/bin/env python
"""at_cap_static.py — how many rows of ONE sampled static store hit the completion cap.

    at_cap_static.py --private-root R --private-run NAME --attack FAM --defense CELL [--json]

Reads every ``<R>/<NAME>/<FAM>/<CELL>/shard*/responses.jsonl`` of a SAMPLED static-family run
(scripts/sdfbase/full22/sampled/DESIGN.md) and prints the number of rows whose ``at_cap`` is true
(``finish_reason == "length"``) — the count the cap-escalation caller reads to decide whether to
launch ``escalate_static.py`` for this (attack, defense). stdout is the bare integer (``--json``: one
JSON line with ``n_at_cap``, ``n_rows``, ``n_shards`` and the store identity); the row total goes
to stderr either way. Exit 0 also when the count is 0.

REFUSES (exit 2) rather than printing 0 whenever 0 would be a lie:
  * a row without a ``finish_reason`` key or without a ``decode`` block — a GREEDY store (the wave's
    greedy rows record neither), which has no at-cap ledger at all; reading it as "nothing at cap"
    would skip the ladder for a store the ladder never saw;
  * a row whose recorded ``at_cap`` disagrees with its ``finish_reason``;
  * a torn line (a shard still being written, or killed mid-append) — the store is not settled;
  * no shard dirs, or a store with zero rows — a missing store is not a store with nothing at cap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class AtCapRefusal(RuntimeError):
    """The store cannot be read as an at-cap ledger (see the module docstring)."""


def shard_dirs(cell_root: Path) -> list[Path]:
    """The ``shard*`` dirs of an (attack, defense) store, in shard order."""
    return sorted(p for p in Path(cell_root).glob("shard*") if p.is_dir())


def read_rows(path: Path) -> list[dict]:
    """Every JSON row of ``path`` in file order; a torn line REFUSES (an unsettled
    store must not be counted), a missing file is an empty list."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AtCapRefusal(f"{path}:{n} is not a JSON row ({exc}) — a torn line: "
                               "the store is not settled; refusing to count it") from exc
    return rows


def row_at_cap(row: dict, *, where: str) -> bool:
    """``at_cap`` of ONE sampled row, derived from ``finish_reason`` and cross-checked
    against the recorded flag. A row that is not a sampled row (no ``finish_reason``
    key, no ``decode`` block) REFUSES — it is a greedy row, and a greedy store has no
    at-cap ledger."""
    if "finish_reason" not in row:
        raise AtCapRefusal(
            f"{where}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r}) "
            "carries no finish_reason — a GREEDY row; a greedy store has no at-cap ledger "
            "and must never read as 'nothing at cap'; refusing")
    if row.get("decode") is None:
        raise AtCapRefusal(
            f"{where}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r}) "
            "carries no decode block — not a sampled row; refusing")
    at_cap = row["finish_reason"] == "length"
    if "at_cap" in row and bool(row["at_cap"]) != at_cap:
        raise AtCapRefusal(
            f"{where}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r}) "
            f"records at_cap={row['at_cap']!r} but finish_reason={row['finish_reason']!r}; "
            "refusing an inconsistent row")
    return at_cap


def scan_static_store(cell_root: Path) -> dict:
    """Walk ``cell_root/shard*/responses.jsonl`` and return ``{n_shards, n_rows,
    n_at_cap, at_cap_keys}`` (``at_cap_keys`` = the ``(goal_id, sample_idx)`` pairs at
    cap, in store order). Raises :class:`AtCapRefusal` per the module docstring."""
    cell_root = Path(cell_root)
    shards = shard_dirs(cell_root)
    if not shards:
        raise AtCapRefusal(f"no shard* dirs under {cell_root} — there is no store to count "
                           "(a missing store is not a store with nothing at cap); refusing")
    n_rows = 0
    n_at_cap = 0
    keys: list = []
    for shard in shards:
        path = shard / "responses.jsonl"
        for row in read_rows(path):
            n_rows += 1
            if row_at_cap(row, where=str(path)):
                n_at_cap += 1
                keys.append((row.get("goal_id"), row.get("sample_idx")))
    if n_rows == 0:
        raise AtCapRefusal(f"{len(shards)} shard dir(s) under {cell_root} hold ZERO rows — an "
                           "empty store cannot be told from a missing one; refusing")
    return {"n_shards": len(shards), "n_rows": n_rows, "n_at_cap": n_at_cap,
            "at_cap_keys": keys}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--private-root", required=True)
    ap.add_argument("--private-run", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--json", action="store_true", help="one JSON line instead of the bare integer")
    ap.add_argument("--share", action="store_true",
                    help="one JSON line {n_at_cap, n_rows, share} (share = n_at_cap / n_rows) — "
                         "the ladder's share readout; the bare integer stays the default")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cell_root = Path(args.private_root) / args.private_run / args.attack / args.defense
    try:
        res = scan_static_store(cell_root)
    except AtCapRefusal as exc:
        print(f"[at-cap-static] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"[at-cap-static] {args.private_run}/{args.attack}/{args.defense}: shards={res['n_shards']} "
          f"rows={res['n_rows']} at_cap={res['n_at_cap']}", file=sys.stderr, flush=True)
    if args.json:
        print(json.dumps({"private_run": args.private_run, "attack": args.attack,
                          "defense": args.defense, "n_shards": res["n_shards"],
                          "n_rows": res["n_rows"], "n_at_cap": res["n_at_cap"]}), flush=True)
    elif args.share:
        print(json.dumps({"n_at_cap": res["n_at_cap"], "n_rows": res["n_rows"],
                          "share": res["n_at_cap"] / res["n_rows"]}), flush=True)
    else:
        print(res["n_at_cap"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
