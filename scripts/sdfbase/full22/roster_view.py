#!/usr/bin/env python
"""full22: print ONE fact of a (target, view) for the sbatch scripts — the view's legs, its controls
driver list, its serving window override, or its prompted-only flag — through f22_roster's readers
(the same functions the generator and the printers use), so a job and its plan cannot disagree.

  python scripts/sdfbase/full22/roster_view.py <target> <view> legs           -> comma list (static,bon,...)
  python scripts/sdfbase/full22/roster_view.py <target> <view> controls       -> comma list for --controls
                                                                                 (gsm8k,mmlu,xstest,or_bench_hard_1k)
  python scripts/sdfbase/full22/roster_view.py <target> <view> max_model_len  -> the ONE window for every standup, or "" (a
                                                                                 table view, or a per-leg view: read `window`)
  python scripts/sdfbase/full22/roster_view.py <target> <view> window <leg>   -> the leg's standup window (static | bon | controls |
                                                                                 capability), or "" (the table's); one-window and
                                                                                 per-leg views both answer (f22_roster.view_windows)
  python scripts/sdfbase/full22/roster_view.py <target> <view> window gsm8k   -> the roster's controls_window.gsm8k for a gsm8k-bearing
                                                                                 TABLE view (the wave's controls standup: the 10-shot
                                                                                 gsm8k prompt + the 512 cap overflowed 2,048), or ""
                                                                                 (no gsm8k in the view's list, or the view's own override)
  python scripts/sdfbase/full22/roster_view.py <target> <view> rung_window <rung> -> the ladder rung's window (2048 | 8192 | 16384),
                                                                                 or "" (the model's table rung window)
  python scripts/sdfbase/full22/roster_view.py <target> <view> prompted_only  -> 1 | 0
  python scripts/sdfbase/full22/roster_view.py <target> <view> has_leg <leg>  -> exit 0 if the view runs it, 3 if not
  ... [--roster <yaml>]   the roster to read (anywhere in the argv; default: configs/sdfbase_full22_roster.yaml,
                         the production roster). The jobs pass common.sh's F22_ROSTER, so a certification-only
                         roster (llama8b/psrcert) is read the same way, by the same readers.
  exit 0 with the value on stdout; a message on stderr, exit 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f22_roster import (
    LEGS,
    SERVING_WINDOW_RUNGS,
    load_roster,
    pop_roster_arg,
    view_controls,
    view_legs,
    view_prompted_only,
    view_serving,
    view_windows,
)

#: the standup a leg rides (the `window <leg>` field): BoN rides the static standup (target_static.sbatch: ONE
#: families_served:short standup serves the four static families and the BoN sub-phase); the capability leg (MMLU
#: letter-logprob) rides the controls standup (the same capability:short standup as the over-refusal controls); `gsm8k`
#: is the roster's controls window for a gsm8k-bearing TABLE view (f22_roster.gsm8k_controls_window, 2026-09-14)
WINDOW_LEG_OF = {"static": "static", "bon": "static", "controls": "controls", "capability": "controls", "gsm8k": "controls_gsm8k"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        argv, roster_path = pop_roster_arg(argv)
    except ValueError as e:
        print(str(e), file=sys.stderr); return 2
    if len(argv) < 3:
        print(__doc__, file=sys.stderr); return 2
    target, view, field = argv[:3]
    if not Path(roster_path).is_file():
        print(f"no roster at {roster_path}", file=sys.stderr); return 2
    roster = load_roster(roster_path)
    try:
        if field == "legs":
            print(",".join(view_legs(roster, target, view)))
        elif field == "controls":
            print(",".join(view_controls(roster, target, view)))
        elif field == "max_model_len":
            print(view_serving(roster, target, view).get("max_model_len", ""))
        elif field == "window":
            if len(argv) != 4 or argv[3] not in WINDOW_LEG_OF:
                print(f"window needs one of {list(WINDOW_LEG_OF)}", file=sys.stderr); return 2
            L = view_windows(roster, target, view)[WINDOW_LEG_OF[argv[3]]]
            print("" if L is None else L)
        elif field == "rung_window":
            if len(argv) != 4 or not argv[3].isdigit() or int(argv[3]) not in SERVING_WINDOW_RUNGS:
                print(f"rung_window needs one of {list(SERVING_WINDOW_RUNGS)}", file=sys.stderr); return 2
            L = view_windows(roster, target, view)["rungs"].get(int(argv[3]))
            print("" if L is None else L)
        elif field == "prompted_only":
            print(1 if view_prompted_only(roster, target, view) else 0)
        elif field == "has_leg":
            if len(argv) != 4 or argv[3] not in LEGS:
                print(f"has_leg needs one of {list(LEGS)}", file=sys.stderr); return 2
            return 0 if argv[3] in view_legs(roster, target, view) else 3
        else:
            print(f"unknown field {field!r}", file=sys.stderr); return 2
    except KeyError:
        print(f"roster names no view {target}/{view}", file=sys.stderr); return 2
    except ValueError as e:
        print(str(e), file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
