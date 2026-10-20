"""The ATTACKS leg hands the low-dose liveness record to the smoke and to that cell's drivers.

THE DEFECT (2026-09-18). Liveness is NUMERIC at low dose because the wave's steering_bites
TEXT probe compares greedy output on five BENIGN prompts (`served_target.PROBE_PROMPTS`), and at a
small enough dose a CORRECT object changes no greedy token there -- so the job refuses a steer that
is live and merely SMALL. Precedent: cluster A 497871, `constv2-mix-full__lora_r4_ce_minus_ctl` at
nominal 0.1 read 0/5 differing with salt isolation AND the intervention hashes both PASSING.

The numeric check was wired into `common_sampled.sh` only, so `target_sampled.sbatch` (the controls
ladder) was protected and `target_attacks.sbatch` (the multi-turn leg) was not -- it called the bare
`f22_positive_control`. The dose ladder that exercises this lane's difference-LoRA arm is the SAME
cell family as 497871, so the attacks leg is exactly where the gap bites. Sibling phase, same
defect: the helpers moved to `common.sh` and the attacks leg now uses them.

THE TWO INVARIANTS THAT MATTER, and why they are asserted by EXECUTION rather than by grep:
  * a steered cell's record reaches THAT cell's processes -- the smoke (a subshell export) and each
    of that cell's drivers (an `env` prefix on the interpreter);
  * it reaches NOTHING else: not `none`, not another cell, and never the job shell itself. A record
    leaking into the job shell would admit every cell on one cell's numeric evidence.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ATTACKS = REPO / "scripts/sdfbase/full22/target_attacks.sbatch"
COMMON = REPO / "scripts/sdfbase/full22/common.sh"
SAMPLED = REPO / "scripts/sdfbase/full22/sampled/common_sampled.sh"


def _run(script: str) -> str:
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, f"rc={p.returncode}\n{p.stdout}\n{p.stderr}"
    return p.stdout


def test_helpers_are_defined_exactly_once_and_in_common():
    """The move must not leave a second copy behind: two definitions silently diverge."""
    for fn in ("f22s_low_dose_liveness", "f22s_launch_cell"):
        n = len(re.findall(rf"(?m)^{fn}\(\)", COMMON.read_text() + "\n" + SAMPLED.read_text()))
        assert n == 1, f"{fn} defined {n} times across common.sh + common_sampled.sh"
        assert re.search(rf"(?m)^{fn}\(\)", COMMON.read_text()), f"{fn} must live in common.sh"


def test_both_lanes_resolve_the_helpers():
    """The sampled lane must still see them, and the attacks leg must see them from common.sh alone."""
    for src in (SAMPLED, COMMON):
        out = _run(
            f'SHARED=/tmp F22_ROSTER=x source {src} 2>/dev/null\n'
            'for fn in f22s_low_dose_liveness f22s_launch_cell; do '
            'declare -F "$fn" >/dev/null && echo "OK $fn" || echo "MISSING $fn"; done'
        )
        assert "MISSING" not in out, f"{src.name}: {out}"


def _positive_control_block() -> str:
    t = ATTACKS.read_text()
    m = re.search(r"(?ms)^f22s_low_dose_liveness\n(?:.*?)^  break; done.*?$", t)
    assert m, "the attacks leg's positive-control block is not in the expected shape"
    return m.group(0)


def test_the_smoke_gets_the_record_for_a_steered_cell_and_the_job_shell_never_does():
    block = _positive_control_block()
    out = _run(
        'set -uo pipefail\n'
        'declare -A F22S_LIVENESS_RECORD=([steered_cell]=/tmp/rec.json)\n'
        'CELLS=(none steered_cell); RC=0\n'
        'f22s_low_dose_liveness(){ :; }\n'   # records are pre-seeded; the writer is exercised elsewhere
        'f22_positive_control(){ echo "SMOKE cell=$1 rec=${SISTEER_LOW_DOSE_LIVENESS_RECORD:-NONE}"; }\n'
        + block + '\n'
        'echo "JOBSHELL rec=${SISTEER_LOW_DOSE_LIVENESS_RECORD:-NONE}"\n'
    )
    # `none` is skipped; the first OBJECT cell is probed, and it carries its record
    assert "SMOKE cell=steered_cell rec=/tmp/rec.json" in out, out
    assert "SMOKE cell=none" not in out, out
    # the record must NOT leak into the job shell (it was a subshell export)
    assert "JOBSHELL rec=NONE" in out, out


def test_a_cell_without_a_record_falls_back_to_the_bare_gate():
    """No writer at this sha / an unevaluable cell => byte-identical to before, gate still stands."""
    block = _positive_control_block()
    out = _run(
        'set -uo pipefail\n'
        'declare -A F22S_LIVENESS_RECORD=()\n'
        'CELLS=(steered_cell); RC=0\n'
        'f22s_low_dose_liveness(){ :; }\n'
        'f22_positive_control(){ echo "SMOKE cell=$1 rec=${SISTEER_LOW_DOSE_LIVENESS_RECORD:-NONE}"; }\n'
        + block
    )
    assert "SMOKE cell=steered_cell rec=NONE" in out, out


def test_only_the_owning_cells_drivers_carry_the_record():
    """The driver env prefix is per cell: `none`'s drivers must be byte-identical to before."""
    t = ATTACKS.read_text()
    m = re.search(r'(?m)^  rec="\$\{F22S_LIVENESS_RECORD\[\$cell\]:-\}"; LIVE_ENV=\(\)\n'
                  r'  \[ -n "\$rec" \] && LIVE_ENV=\(env "SISTEER_LOW_DOSE_LIVENESS_RECORD=\$rec"\)\n', t)
    assert m, "the driver-launch record prefix is not in the expected shape"
    out = _run(
        'set -uo pipefail\n'
        'declare -A F22S_LIVENESS_RECORD=([steered_cell]=/tmp/rec.json)\n'
        'MPY=/usr/bin/python3\n'
        'for cell in none steered_cell; do\n'
        + m.group(0) +
        '  echo "DRIVER cell=$cell argv=[${LIVE_ENV[@]+${LIVE_ENV[@]}} $MPY]"\n'
        'done\n'
    )
    assert "DRIVER cell=none argv=[ /usr/bin/python3]" in out, out
    assert ("DRIVER cell=steered_cell argv=[env "
            "SISTEER_LOW_DOSE_LIVENESS_RECORD=/tmp/rec.json /usr/bin/python3]") in out, out


def test_liveness_runs_before_the_positive_control_not_after():
    """A record written after the smoke has already refused is worthless."""
    t = ATTACKS.read_text()
    i_live = t.index("\nf22s_low_dose_liveness\n")
    i_pc = t.index("f22_positive_control", i_live)
    i_drv = t.index("scripts/jailbreak_run.py generate")
    assert i_live < i_pc < i_drv, "liveness must be written before the smoke, and both before the drivers"
