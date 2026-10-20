"""Lint for scripts/sbatch/capeval_judged.sbatch (capeval-controls §3.4).

Static text checks pinning the cluster hard rules and the §6 submission
contract — the failure modes here (a --export= sneaking in, --mem=0, a
dev QoS in sbatch) each have a real multi-user outage behind them.
"""

import re
from pathlib import Path

SBATCH = Path(__file__).parents[1] / "scripts" / "sbatch" / "capeval_judged.sbatch"


def _text():
    return SBATCH.read_text()


def test_hard_rules():
    text = _text()
    directives = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    # forbidden strings checked on directives + executable lines (the
    # file's own warning comments legitimately NAME the forbidden forms)
    active = directives + [l for l in text.splitlines()
                           if l.strip() and not l.lstrip().startswith("#")]
    assert not any("--export=" in l for l in active)  # hard rule 1: crashes slurmd
    assert not any("--mem=0" in l for l in active)    # hard rule 2: blocks the node
    qos = [l for l in directives if "--qos" in l]
    assert qos and "high" in qos[0]
    assert not any(re.search(r"--qos=\S*dev", l) for l in directives)  # srun-only
    assert any("--mem=128G" in l for l in directives)
    assert any("--gres=gpu:1" in l for l in directives)


def test_submission_contract():
    text = _text()
    directives = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert any("--open-mode=append" in l for l in directives)  # retries append
    assert any("--no-requeue" in l for l in directives)  # escalation owns requeue
    assert any("--job-name=evals_" in l for l in directives)  # session-unique
    assert any("--time=" in l for l in directives)  # never the 24h default
    # our OWN checkout + the shared preemption/mirror scaffolding
    assert "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-evals-repo" in text
    assert "capeval_common.sh" in text  # -> stage2_common: hf_env.sh (shared
    # copy), traps, STOP sentinel, QoS escalation, $FIGS_DIR mirror
    assert "s2_init" in text and "s2_run" in text
    assert "NCCL_SOCKET_IFNAME=" not in text   # the socket interface is the site's (its nccl.conf), never exported
    assert "NCCL_NVLS_ENABLE=0" in text
    # array-only: unset SLURM_ARRAY_TASK_ID must die loudly
    assert "SLURM_ARRAY_TASK_ID is unset" in text


def test_task_decode_matches_driver():
    """The array decode names exactly the driver's task CLI names."""
    text = _text()
    m = re.search(r"TASKS=\(([^)]*)\)", text)
    assert m, "no TASKS array in the sbatch"
    assert m.group(1).split() == ["or_bench", "xstest"]        # alpaca retired 2026-09-11
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "capeval_judged_run_lint",
        SBATCH.parents[1] / "capeval_judged_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capeval_judged_run_lint"] = mod
    spec.loader.exec_module(mod)
    assert sorted(mod.TASKS) == ["or_bench", "xstest"]
