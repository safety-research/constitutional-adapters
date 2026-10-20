"""The sampled lane's job registers EXACTLY its phase's evals. f22_registry_register (common.sh) reads
F22_REGISTRY_EVALS, else the job-NAME table whose `f22s-*` row is the lane's UNION (pasttense codeattack renellm grok420 xstest or_bench_hard)
— a kind, not a phase — so the static 8192 rung 838299 registered 12 runs, 4 of them controls it never produces. f22s_parse_phase now exports
the phase's own evals (the static families for static_base / static_esc / static_esc16, "xstest or_bench_hard" for controls); a caller's value
(the front door's stage.py) wins; the extend / capability / gsm8k phases keep the table. Run for real: the lane's shell, sourced under bash."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

SH = "scripts/sdfbase/full22/sampled/common_sampled.sh"


def _parse(phase: str, preset: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "F22_REGISTRY_EVALS"}
    if preset is not None:
        env["F22_REGISTRY_EVALS"] = preset
    r = subprocess.run(["bash", "-c", f'source {SH} && f22s_parse_phase "$1" && printf "%s|%s|%s" "$PHASE_KIND" "${{FAMS[*]-}}" "${{F22_REGISTRY_EVALS-unset}}"', "_", phase],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (phase, r.stdout[-800:], r.stderr[-800:])
    return r.stdout.strip().rsplit("|", 2)[-1], r.stdout.strip()


@pytest.mark.parametrize("phase, evals", [
    ("static_esc_t0", "pasttense codeattack renellm grok420"),
    ("static_esc_t1", "pasttense codeattack renellm grok420"),
    ("static_esc_t0:pasttense,renellm", "pasttense renellm"),
    ("static_base_t1", "pasttense codeattack renellm grok420"),
    ("static_esc16_t0:grok420", "grok420"),
    ("controls_t0", "xstest or_bench_hard"),
    ("controls_esc_t1", "xstest or_bench_hard"),
    ("controls_esc_t0:2048", "xstest or_bench_hard"),
])
def test_a_static_or_controls_phase_exports_exactly_its_own_evals(phase, evals):
    got, full = _parse(phase)
    assert got == evals, (phase, full)


@pytest.mark.parametrize("phase", ["ladder_extend_t0", "capability_t1", "gsm8k_t0"])
def test_the_other_phases_keep_the_job_kind_table_for_now(phase):
    got, full = _parse(phase)
    assert got == "unset", (phase, full)


def test_a_callers_value_wins_and_the_export_precedes_the_registration():
    got, full = _parse("static_esc_t0", preset="pasttense")
    assert got == "pasttense", full                                                        # the front door's stage.py sets the unit's evals
    # the registration reads the variable (common.sh) and the sbatch parses the phase BEFORE it resolves the config (where it registers)
    common = (REPO / "scripts" / "sdfbase" / "full22" / "common.sh").read_text()
    assert 'evals_csv="${F22_REGISTRY_EVALS:-$(f22_registry_evals_for_job "${SLURM_JOB_NAME:-}")}"' in common
    assert 'f22s-*)              echo "pasttense codeattack renellm grok420 xstest or_bench_hard";;' in common      # the UNION row stands as the fallback
    sb = (REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "target_sampled.sbatch").read_text()
    assert 'f22_init "$WAVE_SHA"; f22s_parse_phase "$PHASE"; f22s_resolve_config "$KEY" "$TRIAL" "$VIEW"' in sb
    cs = (REPO / SH).read_text()
    assert cs.index('export F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-${FAMS[*]}}"') > cs.index('for f in "${FAMS[@]}"; do case "$f" in')     # after FAMS is known
