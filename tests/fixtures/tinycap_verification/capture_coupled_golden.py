#!/usr/bin/env python
"""Captures ``coupled_filing_golden.json`` — the COUPLED path's filing of the synthetic fixture on
the PRE-LIFT tree (tinycap-shared-standup landing 2: the filing tail was lifted out of
``run_one_task`` into ``_file_task_results``; this golden is what pins the lift byte-identical,
``tests/test_inspect_served_verification_phase.py::test_the_coupled_path_through_the_lifted_tail_
is_byte_identical_to_the_pre_lift_golden``).

    git archive 975d575e8 src/sisteer configs/capeval | tar -x -C /tmp/l2-base
    INSPECT_DISPLAY=none PYTHONPATH=src ~/venvs/tinycap-laneB/bin/python \\
        tests/fixtures/tinycap_verification/capture_coupled_golden.py --base-src /tmp/l2-base/src

Two phases. (1) In THIS tree: the fixture's twelve completions are SCORED IN PLACE under the
target provider's own key (M1b: ``score_async`` inside a local ``sandboxenv_context``, the test
module's ``verify`` scorer) — the log a coupled run leaves behind. (2) In a SUBPROCESS whose ``PYTHONPATH``
is the base tree's ``src``: the base ``run_one_task`` files that log (a fake ``eval_async`` returns
it) into a bare store, and the summary (minus the wall clock), the outcome record and the emitted
rows (minus wall clocks) are written in the golden's shape. The owned provider module is stubbed
in the subprocess (its import needs Modal) and the D73 transport install is a no-op — neither
touches a filed byte. Re-capturing is a deliberate filing-shape change, never maintenance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

BASE_SCRIPT = r'''
import asyncio, json, shutil, sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])            # the BASE tree's src, ahead of everything
coupled_log, out_path, store_dir = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
stub = types.ModuleType("sisteer.capeval.tiers.sandbox_owned")
stub.OwnedModalSandboxEnvironment = type("OwnedModalSandboxEnvironment", (), {})
stub.PROVIDER_NAME = "modal_owned"
sys.modules["sisteer.capeval.tiers.sandbox_owned"] = stub
from sisteer.capeval.tinycap import verifier_transport as VT
VT.install_exec_spill = lambda *a, **k: None
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tinycap import contracts as C
import inspect_ai
from inspect_ai.log import read_eval_log
assert not hasattr(isv.TaskPlan, "generation_only") or "generation_only" not in isv.TaskPlan.__dataclass_fields__, "not the base tree"

class _Store:
    def __init__(self, d):
        self.cell_dir = Path(d); self.cell_dir.mkdir(parents=True, exist_ok=True)

class _Emitter:
    def __init__(self): self.rows = []
    def emit(self, **kw): self.rows.append(kw)

async def fake_eval(task, **kwargs):
    d = Path(kwargs["log_dir"]); d.mkdir(parents=True, exist_ok=True)
    dst = d / coupled_log.name; shutil.copyfile(coupled_log, dst)
    return [read_eval_log(str(dst))]
inspect_ai.eval_async = fake_eval
ROW_TASK = C.inspect_task(C.CONTROL_BIGCODEBENCH_FULL)
IDS = [str(i) for i in range(12)]
plan = isv.TaskPlan(row_task=ROW_TASK, name=C.CONTROL_BIGCODEBENCH_FULL, task=object(), sample_ids=list(IDS),
                    samples_per_item=1, limit=None, max_tokens=4096, primary_scorer="verify", expected_n=12, n_run=12,
                    temperature=0.0, top_p=1.0, top_k=0, item_set="full", thinking=C.THINKING_OFF, cap_source="off_table",
                    sandbox=C.bcb_sandbox_facts(), note="synthetic fixture (tests/fixtures/tinycap_verification)")
store, em = _Store(store_dir), _Emitter()
summary = asyncio.run(isv.run_one_task(plan, model=None, store=store, template={}, thinking=C.THINKING_OFF, concurrency=32,
                                       emitter=em, route_virtual_model="petri__base", limits={}))
fn = C.marker_filename(ROW_TASK)
import re
_TIMING = re.compile(r"Ran (\d+) tests? in \d+\.\d+s")
def _stable(o):
    if isinstance(o, dict): return {k: _stable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_stable(v) for v in o]
    if isinstance(o, str): return _TIMING.sub(r"Ran \1 tests in <T>s", o)
    return o
got = {"summary": {k: v for k, v in summary.items() if k != "wall_clock_s"},
       "outcomes": json.loads((store.cell_dir / C.TASK_OUTCOMES_DIR / fn).read_text()),
       "rows": [_stable({k: v for k, v in r.items() if k != "wall_clock_s"}) for r in em.rows]}
got = json.loads(json.dumps(got, default=str))
out_path.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
print("captured", out_path, "rows", len(got["rows"]), "items", len(got["outcomes"]["items"]))
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-src", required=True, help="the PRE-LIFT tree's src/ (git archive of the landing base)")
    ap.add_argument("--out", default=str(HERE / "coupled_filing_golden.json"))
    args = ap.parse_args()
    os.environ.setdefault("INSPECT_DISPLAY", "none")
    import test_inspect_served_verification_phase as T   # the synthetic task + the verify scorer, one spelling

    original = T.fixture_log()
    with tempfile.TemporaryDirectory(prefix="l2-golden-") as tmp:
        tmp = Path(tmp)
        # the fixture SCORED IN PLACE under the target provider's key (M1b over the local sandbox): the log
        # a coupled run leaves behind (a replay vehicle would key model_usage under `replay/…`)
        _log, coupled_path = asyncio.run(T.CV.run_mechanism(
            T.M1B, original=original, coupled_task=T.synthetic_task(original), max_sandboxes=4,
            log_dir=tmp / "coupled"))
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        r = subprocess.run([sys.executable, "-c", BASE_SCRIPT, str(args.base_src), str(coupled_path), str(args.out),
                            str(tmp / "store" / "cells" / "k")], env=env, text=True, capture_output=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr[-4000:])
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
