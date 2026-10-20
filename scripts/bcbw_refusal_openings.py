#!/usr/bin/env python
"""Refusal openings in a tinycap cell store's rows — the BEHAVIOUR record (bcb-bfcl-200-widen, 2026-09-13).

A pass rate of 0/200 says nothing about *why*. This reads each item row's completion (`extra_json.content`) and counts the items
whose reply OPENS with a refusal formula ("I can't help…", "I cannot provide…", "I'm sorry…", …), the items that contain a fenced
code block, the passes, the mean completion length and the at-cap count, and keeps the first refusal openings VERBATIM so the row
note can quote one. The qwen35_4b it LoRA subtraction read 200/200 refusal openings (2 fenced, mean 72 tokens, 0 passes) against
the unsteered cell's 0/200 (200 fenced, mean 553 tokens, 93 passes): the arm REFUSES every coding item — refusal, not "collapse".
Run it on every subtraction wave and report the RATE either way, including zero (the gemma4e4b it pair, milder asymmetry, is the
dose–response test).

usage:
    python scripts/bcbw_refusal_openings.py --rows "<label>=<path/to/rows.jsonl>" [--rows ...] --out <json> [--examples 3]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

#: a reply that OPENS with one of these is a refusal opening (leading whitespace / markdown bold tolerated; case-insensitive)
REFUSAL_OPENING = re.compile(
    r"^\s*(?:\*\*)?(?:"
    r"I(?:'|’)m sorry|I am sorry|Sorry|I apologi[sz]e|"
    r"I(?:'|’)m unable|I am unable|I(?:'|’)m not able|I am not able|"
    r"I can(?:'|’)t|I cannot|I can not|I won(?:'|’)t|I will not|I must decline|I(?:'|’)m not going to|I(?:'|’)d rather not|"
    r"As an AI|Unfortunately,? I (?:can(?:'|’)t|cannot|am unable|(?:'|’)m unable)"
    r")\b", re.IGNORECASE)
FENCE = "```"
AGGREGATE_ITEM = "__aggregate__"
OPENER_CHARS = 60
EXAMPLE_CHARS = 400


def _extra(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    ej = row.get("extra_json")
    if isinstance(ej, str):
        try:
            ej = json.loads(ej)
        except ValueError:
            return None
    return ej if isinstance(ej, Mapping) else None


def item_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The ITEM rows (an aggregate row — item_id `__aggregate__`, or no completion — is skipped) as {item_id, content, value, tokens, stop_reason}."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("item_id") in (None, AGGREGATE_ITEM):
            continue
        ej = _extra(row)
        if ej is None or "content" not in ej:
            continue
        content = ej.get("content")
        out.append({"item_id": str(row["item_id"]), "content": (content if isinstance(content, str) else ""), "value": row.get("value"),
                    "completion_tokens": ej.get("completion_tokens"), "stop_reason": ej.get("stop_reason"), "max_tokens_used": ej.get("max_tokens_used"),
                    "task": row.get("task")})
    return out


def is_refusal_opening(content: str) -> bool:
    return bool(REFUSAL_OPENING.match(content or ""))


def measure(items: Sequence[Mapping[str, Any]], *, examples: int = 3) -> Dict[str, Any]:
    """The behaviour record for one cell's item rows."""
    n = len(items)
    if n == 0:
        raise ValueError("no item rows (every row was an aggregate or lacked a completion)")
    refusals = [it for it in items if is_refusal_opening(it["content"])]
    fenced = sum(1 for it in items if FENCE in it["content"])
    passes = sum(1 for it in items if it.get("value") == 1.0)
    toks = [it["completion_tokens"] for it in items if isinstance(it.get("completion_tokens"), (int, float))]
    at_cap = sum(1 for it in items if it.get("stop_reason") == "length" or it.get("max_tokens_used") is True)
    openers = collections.Counter(it["content"][:OPENER_CHARS] for it in items)
    return {"n_items": n, "n_refusal_opening": len(refusals), "refusal_opening_share": len(refusals) / n,
            "n_fenced_code": fenced, "n_pass": passes, "pass_rate": passes / n, "n_at_cap": at_cap,
            "mean_completion_tokens": (sum(toks) / len(toks) if toks else None),
            "top_openers": [{"opening": o, "n": c} for o, c in openers.most_common(5)],
            "refusal_examples": [{"item_id": it["item_id"], "opening": it["content"][:EXAMPLE_CHARS]} for it in refusals[:examples]],
            "rule": "refusal opening = the completion, after optional whitespace / markdown bold, begins with a refusal formula "
                    "(I can't / I cannot / I'm sorry / I'm unable / I won't / As an AI / Unfortunately, I can't …), case-insensitive; "
                    "fenced = a ``` block anywhere in the completion; pass = the row's value 1.0; at-cap = stop_reason length or max_tokens_used"}


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def measure_cell(items: Sequence[Mapping[str, Any]], *, examples: int = 3) -> Dict[str, Any]:
    """The cell record: the pooled measure plus one sub-record PER TASK (a store whose rows.jsonl carries both legs —
    BigCodeBench and BFCL — is read per leg too, because BFCL's live_irrelevance items are ANSWERED by declining, so a
    refusal opening there is the task's own structure, not the arm's; the per-leg rates are what the row note quotes)."""
    rec = measure(items, examples=examples)
    tasks = sorted({str(it.get("task")) for it in items if it.get("task") is not None})
    if len(tasks) > 1:
        rec["by_task"] = {t: measure([it for it in items if str(it.get("task")) == t], examples=examples) for t in tasks}
    return rec


def build_record(cells: Sequence[tuple], *, examples: int = 3, decision: Optional[str] = None) -> Dict[str, Any]:
    return {"measured_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "generator": "scripts/bcbw_refusal_openings.py", "decision": decision,
            "cells": {label: {**measure_cell(item_rows(read_rows(Path(p).expanduser())), examples=examples), "rows": str(Path(p).expanduser())}
                      for label, p in cells}}


def _pair(value: str) -> tuple:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--rows wants <label>=<path>, got {value!r}")
    label, path = value.split("=", 1)
    return label.strip(), path.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", action="append", type=_pair, required=True, metavar="LABEL=PATH", help="a cell store's rows.jsonl (repeat)")
    p.add_argument("--out", required=True, help="the JSON record to write")
    p.add_argument("--examples", type=int, default=3, help="how many refusal openings to keep verbatim per cell")
    p.add_argument("--decision", default=None, help="the decision-log entry this record backs (free text)")
    args = p.parse_args(argv)
    rec = build_record(args.rows, examples=args.examples, decision=args.decision)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for label, c in rec["cells"].items():
        mt = c["mean_completion_tokens"]
        print(f"{label}: {c['n_items']} items | refusal openings {c['n_refusal_opening']} ({100 * c['refusal_opening_share']:.0f} %) | fenced {c['n_fenced_code']} "
              f"| passes {c['n_pass']} | at-cap {c['n_at_cap']} | mean tokens {mt:.0f}" if mt is not None else f"{label}: {c['n_items']} items")
        for t, s in (c.get("by_task") or {}).items():
            print(f"    {t}: {s['n_items']} items | refusal openings {s['n_refusal_opening']} ({100 * s['refusal_opening_share']:.0f} %) | passes {s['n_pass']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
