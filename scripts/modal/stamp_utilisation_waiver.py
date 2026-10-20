#!/usr/bin/env python3
"""Stamp `RS/utilisation_waiver.json` -- the readings-bearing waiver through which the launch certificate ADMITS an UNMET
utilisation verdict (`smokes.json` S5.bars, `bars_met: false`).

    stamp_utilisation_waiver.py --rs RS --plan RS/launch_plan.json --ruling D198

The waiver is DERIVED from the recorded verdict (`seedrep_launch_plan.build_utilisation_waiver`): its unmet bars and their readings are
copied from the verdict, its canonical sha256 is bound, the plan's W and the ruling label are stamped -- nothing is typed by hand. Refuses
(exit 2) when the S5 block carries no verdict, the verdict does not validate against the plan's W and the S5 execution wave, `bars_met`
is true (nothing to waive), the ruling label is not D198, or a DIFFERENT waiver is already stamped (write-once; an identical one is a no-op).
Exit 0: the waiver's path and sha256 and the unmet bars are printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedrep_launch_plan as lp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rs", type=Path, required=True, help="the results root (RS/smokes.json holds the S5 verdict; the waiver is written beside it)")
    ap.add_argument("--plan", type=Path, required=True, help="RS/launch_plan.json (its W binds the waiver)")
    ap.add_argument("--ruling", default=lp.UTILISATION_WAIVER_RULING, help=f"the ruling label; only {lp.UTILISATION_WAIVER_RULING} is admissible")
    ap.add_argument("--out", type=Path, default=None, help=f"where to write (default RS/{lp.UTILISATION_WAIVER_FILE})")
    a = ap.parse_args(argv)
    out = a.out or (a.rs / lp.UTILISATION_WAIVER_FILE)
    try:
        plan = json.loads(Path(a.plan).read_text())
        smokes_p = a.rs / "smokes.json"
        if not smokes_p.is_file():
            raise ValueError(f"{smokes_p}: missing -- no S5 verdict to waive")
        smokes = json.loads(smokes_p.read_text())
        s5 = smokes.get("S5") if isinstance(smokes, dict) else None
        if not isinstance(s5, dict) or s5.get("bars") is None:
            raise ValueError(f"{smokes_p}: the S5 block carries no utilisation verdict (S5.bars) -- nothing to waive")
        waiver = lp.build_utilisation_waiver(s5["bars"], W=str(plan.get("W")), execution_wave_uuid=s5.get("execution_wave_uuid"), ruling=a.ruling)
        if out.is_file():
            old = json.loads(out.read_text())
            if {k: v for k, v in old.items() if k != "stamped_utc"} != {k: v for k, v in waiver.items() if k != "stamped_utc"}:
                raise ValueError(f"{out}: a DIFFERENT waiver is already stamped (write-once); refusing to overwrite")
            waiver = old
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
            tmp.write_text(json.dumps(waiver, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, out)
    except (OSError, ValueError, KeyError, TypeError) as e:
        sys.stderr.write(f"stamp_utilisation_waiver: REFUSED: {type(e).__name__}: {e}\n")
        return 2
    print(f"stamped {out} sha256={hashlib.sha256(out.read_bytes()).hexdigest()} by={waiver['by']} W={waiver['W'][:12]} bars_sha256={waiver['bars_sha256'][:16]} "
          f"unmet={[u['bar'] for u in waiver['unmet']]}")
    for u in waiver["unmet"]:
        print(f"  {u['bar']}: value {u['value']} vs threshold {u['threshold']} (met {u['met']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
