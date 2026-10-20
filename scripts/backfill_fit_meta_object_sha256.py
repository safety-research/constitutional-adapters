#!/usr/bin/env python
"""Backfill ``object_sha256`` into Phase B ``fit_meta.json`` records that predate the field.

The four Phase B AlphaSteer fits (gemma4e4b, qwen35_4b, glm46vflash, nemotron3nano4b) were written before
fit_class_b recorded the object's hash. They are verified and their inputs are hash-pinned, so re-fitting them
to obtain a field would be silly: this hashes each object IN PLACE and records it beside a marker that says
it was backfilled. Everything it writes, and every refusal, is ``sisteer.extraction.fit_meta_binding.
backfill_object_sha256`` (read that docstring); this file is the command line.

STDLIB ONLY, and it loads the binding module BY FILE PATH rather than ``import sisteer...`` — so it runs on
a cluster login shell's bare python3 next to the objects: no venv, no PYTHONPATH, and immune to anything the
package ``__init__`` may one day import (today it imports nothing; the tool must not depend on that staying so).

usage (the alphasteer_f22 objects root layout, <root>/<target>/cells/alphasteer/):
  python scripts/backfill_fit_meta_object_sha256.py --root $SHARED/exp/objects/alphasteer_f22 \\
      --target gemma4e4b,qwen35_4b,glm46vflash,nemotron3nano4b [--dry-run] [--date YYYY-MM-DD]
  python scripts/backfill_fit_meta_object_sha256.py --cell-dir <dir> [--cell-dir <dir> ...]     # explicit dirs
One JSON line per cell on stdout; refusals on stderr; exit 1 if any cell refused, else 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = "scripts/backfill_fit_meta_object_sha256.py"


def load_binding_module():
    p = REPO / "src" / "sisteer" / "extraction" / "fit_meta_binding.py"
    spec = importlib.util.spec_from_file_location("fit_meta_binding", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def repo_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip() or None
    except Exception:  # noqa: BLE001 — a checkout without git is still allowed to backfill; the record says None
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="objects root; cells at <root>/<target>/cells/alphasteer")
    ap.add_argument("--target", action="append", default=[], help="target key (repeat or comma-separate)")
    ap.add_argument("--cell-dir", action="append", default=[], type=Path, help="an explicit cell dir")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d", time.gmtime()),
                    help="the date written into object_sha256_source (default: today, UTC)")
    ap.add_argument("--tool-commit", default=None, help="recorded in object_sha256_backfill (default: this checkout's HEAD)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args(argv)
    targets = [t.strip() for v in args.target for t in v.split(",") if t.strip()]
    if targets and args.root is None:
        ap.error("--target needs --root")
    cells = [(t, args.root.expanduser().resolve() / t / "cells" / "alphasteer") for t in targets]
    cells += [(str(d), d.expanduser().resolve()) for d in args.cell_dir]
    if not cells:
        ap.error("name at least one cell: --root + --target, or --cell-dir")
    fmb = load_binding_module()
    tool_commit = args.tool_commit or repo_commit()
    rc = 0
    for what, cell in cells:
        try:
            out = fmb.backfill_object_sha256(cell, date=args.date, tool=TOOL, tool_commit=tool_commit,
                                             dry_run=args.dry_run, what=what)
            print(json.dumps({"cell": what, **out}, sort_keys=True), flush=True)
        except ValueError as e:
            print(f"[backfill] REFUSED {e}", file=sys.stderr, flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
