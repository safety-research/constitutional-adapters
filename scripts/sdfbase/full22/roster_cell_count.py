#!/usr/bin/env python
"""full22: print the number of cells the ROSTER names for one (target, view) — the reference the
job-side short-view guard (`common.sh f22_bank_view`) checks the generated config and the built bank
view against. A standalone script, not an inline heredoc, so the guard stays readable in the sbatch
and testable on its own.

  python scripts/sdfbase/full22/roster_cell_count.py <target> <view>            -> the SERVED count (what
                                                     the generated config's explicit_cells holds minus `none`)
  python scripts/sdfbase/full22/roster_cell_count.py <target> <view> --staged   -> the STAGED count (what the
                                                     bank view holds: served minus the prompted__<si> baselines,
                                                     which serve the SI text and have no object)
  python scripts/sdfbase/full22/roster_cell_count.py <target> <view> [--staged] --roster <yaml>
                                                     -> the same counts read from THAT roster (default: the
                                                     production configs/sdfbase_full22_roster.yaml). The jobs pass
                                                     common.sh's F22_ROSTER, so a certification-only roster
                                                     (llama8b/psrcert) is counted by the same reader; a
                                                     (target, view) the named roster does not carry is exit 2,
                                                     never a count
  exit 0 with the count on stdout; a message on stderr, exit 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f22_roster import load_roster, pop_roster_arg, served_cells, staged_cells  # noqa: E402  (view cells + enabled lanes — the generator's set; its staged subset)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        argv, roster_path = pop_roster_arg(argv)
    except ValueError as e:
        print(str(e), file=sys.stderr); return 2
    staged = "--staged" in argv
    argv = [a for a in argv if a != "--staged"]
    if len(argv) != 2:
        print("usage: roster_cell_count.py <target> <view> [--staged] [--roster <yaml>]", file=sys.stderr)
        return 2
    target, view = argv[:2]
    if not Path(roster_path).is_file():
        print(f"no roster at {roster_path}", file=sys.stderr); return 2
    roster = load_roster(roster_path)
    try:
        cells = (staged_cells if staged else served_cells)(roster, target, view)
    except KeyError:
        print(f"roster names no view {target}/{view}", file=sys.stderr)
        return 2
    print(len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
