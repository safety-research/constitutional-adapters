"""scripts/sbatch/hpo_trial.sbatch under HPO_JOB_MODE=recon as an ARRAY, executed under fakes (the
fit_class_b harness's pattern: exported bash functions for git / scontrol / srun, a python that answers the cu128 assert and
delegates every other snippet to the real interpreter): an array task passes `--rung $SLURM_ARRAY_TASK_ID` to `tune_arm.py recon`
with the spec's ladder; a task outside the ladder refuses (exit 3) BEFORE any srun; a non-array job runs the serial ladder (no
--rung) exactly as before."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "sbatch" / "hpo_trial.sbatch"
STUB_SHA = "0123456789abcdef0123456789abcdef01234567"

_STUBS = r"""
git() { case "$*" in *rev-parse*) echo "$STUB_SHA";; *status*) :;; *) :;; esac; }
scontrol() { printf '   Comment=sha:%s \n   Command=%s\n' "$STUB_SHA" "$HPO_SBATCH"; }
srun() { printf '%s\n' "$*" >> "$SRUN_RECORD"; }
export -f git scontrol srun
exec bash "$HPO_SBATCH" "$@"
"""


def _world(tmp_path: Path, spec_extra: dict | None = None):
    shared = tmp_path / "shared"
    (shared / "scripts").mkdir(parents=True)
    (shared / "scripts" / "hf_env.sh").write_text("")
    (shared / "scripts" / "mirror-figs.sh").write_text("#!/bin/bash\nexit 0\n"); (shared / "scripts" / "mirror-figs.sh").chmod(0o755)
    venv = shared / "envs" / "fitenv" / "bin"; venv.mkdir(parents=True)
    # the cu128 assert is answered; every other -c snippet (spec parsing, the config's batch size) runs on the real interpreter
    (venv / "python").write_text("#!/bin/bash\n" 'case "$*" in *"import torch"*) echo "cu128 runtime assert OK: fake"; exit 0;; esac\n'
                                 f'exec {sys.executable} "$@"\n')
    (venv / "python").chmod(0o755)
    root = tmp_path / "hpo_root"; root.mkdir()
    spec = tmp_path / "recon_spec.json"
    spec.write_text(json.dumps({"kind": "recon_spec", "lr_ladder": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "steps": 150, "clip_percentile": 95,
                                "warmup_frac": 0.15, "init_scheme": "random_lecun_1e-3", "si": "constv2ctl2-full", **(spec_extra or {})}))
    cfg = tmp_path / "fits.yaml"; cfg.write_text("batch_size_by_arm:\n  psr_ce_all: 1\n")
    pairs = tmp_path / "pairs.jsonl"; pairs.write_text("{}\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "STUB_", "HPO_", "SISTEER_", "SDFBASE_"))}
    env.update(USER="nobody-testhost", SLURM_JOB_ID="4242", SLURM_JOB_NAME="hpo-m-relu-recon", STUB_SHA=STUB_SHA,
               HPO_SBATCH=str(SBATCH), SRUN_RECORD=str(tmp_path / "srun.log"), SISTEER_SHARED_ROOT=str(shared),
               SDFBASE_FIT_ENV_NAME="fitenv", HPO_JOB_MODE="recon", PATH=os.environ["PATH"], HOME=str(tmp_path))
    return env, [str(spec), str(root), "psr_ce_all", str(cfg), str(pairs)], root


def _run(env, args):
    return subprocess.run(["bash", "-c", _STUBS, "harness", *args], capture_output=True, text=True, env=env, timeout=120)


def test_array_task_trains_exactly_its_rung(tmp_path):
    env, args, root = _world(tmp_path)
    env.update(SLURM_ARRAY_JOB_ID="4242", SLURM_ARRAY_TASK_ID="2")
    r = _run(env, args)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    sruns = (tmp_path / "srun.log").read_text().splitlines()
    assert len(sruns) == 1, sruns
    s = sruns[0]
    assert " recon " in s and s.rstrip().endswith("--rung 2"), s
    assert "--lr-ladder 0.0001,0.0003,0.001,0.003,0.01" in s and "--steps 150" in s and "--clip-percentile 95" in s
    assert "--warmup-frac 0.15" in s and "--init-scheme random_lecun_1e-3" in s and f"--out {root}/recon.json" in s
    assert f"--spec {args[0]}" in s, s                             # the rung is stamped with the spec it realises
    assert "recon array task 2: rung 2 of 5" in out
    assert "--passes" not in s                                    # no cycled_passes in the spec -> the paper's one pass
    assert (Path(env["SISTEER_SHARED_ROOT"]) / "exp/figs/hpo-m-relu-recon/RESULT").read_text().startswith("DONE rc=0")


def test_task_outside_the_ladder_refuses_before_any_srun(tmp_path):
    env, args, _root = _world(tmp_path)
    env.update(SLURM_ARRAY_JOB_ID="4242", SLURM_ARRAY_TASK_ID="5")
    r = _run(env, args)
    out = r.stdout + r.stderr
    assert r.returncode == 3, out[-3000:]
    assert "outside the recon ladder (0..4, 5 rungs)" in out
    assert not (tmp_path / "srun.log").exists()


def test_non_array_job_runs_the_serial_ladder(tmp_path):
    env, args, _root = _world(tmp_path)
    r = _run(env, args)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    sruns = (tmp_path / "srun.log").read_text().splitlines()
    assert len(sruns) == 1 and "--rung" not in sruns[0] and " recon " in sruns[0], sruns


def test_recorded_cycled_passes_reach_the_rung_as_passes(tmp_path):
    """Only a spec carrying cycled_passes > 1 (the front door's recorded opt-in) makes the rung cycle."""
    env, args, _root = _world(tmp_path, spec_extra={"cycled_passes": 3, "rows": 4102, "steps_per_pass": 65})
    env.update(SLURM_ARRAY_JOB_ID="4242", SLURM_ARRAY_TASK_ID="0")
    r = _run(env, args)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    s = (tmp_path / "srun.log").read_text().splitlines()[0]
    assert "--passes 3" in s and s.rstrip().endswith("--rung 0"), s
    assert "cycled_passes=3 (the operator's opt-in recorded in the spec)" in out
