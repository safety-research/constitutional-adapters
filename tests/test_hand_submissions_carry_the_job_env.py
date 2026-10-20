"""A job submitted by hand gets past every setting it refuses without.

Cluster job scripts take their shared root, project directory and the rest from the job environment the front door composes from
the cluster's site block (`submit_twin.py job-env`), and refuse by name without them. The two submit wrappers run on a login node
and hand their own environment to sbatch, so each refuses before submitting anything unless the settings its job needs are there,
and submits with the cluster's QoS and a log directory under the shared root when they are. Each wrapper runs here against a stub
sbatch that records its arguments and environment; the job's own required-setting lines then run under that recorded environment
and must all pass. The job environment the front door prints for a synthetic site passes the same lines for every job these
routes submit, build_views.sbatch included (its header documents a direct submission under that environment).

Synthetic site and a stub sbatch; nothing contacts a cluster."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, write_site  # noqa: E402

SB = REPO / "scripts" / "sbatch"
WRAPPERS = {"submit_evals_cell_controls.sh": "evals_cell_controls.sbatch", "submit_mamba_build.sh": "mamba_build.sbatch"}
HAND_SUBMITTED_JOBS = ("scripts/sbatch/evals_cell_controls.sbatch", "scripts/sbatch/mamba_build.sbatch",
                       "scripts/sbatch/twosi_fits_llama.sbatch", "scripts/sdfbase/full22/strength/build_views.sbatch")
_REQUIRED = re.compile(r"\$\{(SISTEER_[A-Z0-9_]+):\?")


def _code_lines(path: Path) -> list:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _required(path: Path) -> set:
    """The settings a script refuses by name without (`${SISTEER_X:?...}`)."""
    return {name for ln in _code_lines(path) for name in _REQUIRED.findall(ln)}


def _job_passes_its_requirements(job: Path, env: dict) -> subprocess.CompletedProcess:
    """The job's own lines that refuse without a setting, run under `env` (each is one assignment)."""
    lines = [ln for ln in _code_lines(job) if _REQUIRED.search(ln)]
    assert lines, f"{job.name} refuses without no setting"
    script = "\n".join(lines) + '\necho "JOB_SHARED=${SHARED:-}"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env}, timeout=30)


def _clean_env(**extra) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SISTEER_", "CLUSTER_", "REGISTRY_", "WANDB_")) and k != "QOS"}
    e.update(extra)
    return e


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_a_submit_wrapper_hands_its_job_every_setting_the_job_refuses_without(wrapper, tmp_path):
    job = SB / WRAPPERS[wrapper]
    missing_checks = _required(job) - _required(SB / wrapper)
    assert not missing_checks, f"{wrapper} submits {job.name} without checking {sorted(missing_checks)}"
    sb_dir, bindir, rec = tmp_path / "sbatch", tmp_path / "bin", tmp_path / "rec"
    sb_dir.mkdir(); bindir.mkdir()
    shutil.copy2(SB / wrapper, sb_dir / wrapper)
    shutil.copy2(job, sb_dir / job.name)                        # the real job beside its wrapper, as in the repository
    (bindir / "sbatch").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {rec}.argv\nenv > {rec}.env\n')
    (bindir / "sbatch").chmod(0o755)
    root = tmp_path / "shared"
    site = {"SISTEER_SHARED_ROOT": str(root), "SISTEER_PROJECT_ROOT": str(tmp_path / "project"), "SISTEER_CLUSTER_QOS": "qos_a"}

    def run(env):
        Path(f"{rec}.argv").unlink(missing_ok=True)
        return subprocess.run(["bash", str(sb_dir / wrapper), "--time=1:00:00"], capture_output=True, text=True,
                              env=_clean_env(PATH=f"{bindir}:{os.environ['PATH']}", **env), timeout=60)

    r = run(site)
    assert r.returncode == 0, r.stderr
    argv = Path(f"{rec}.argv").read_text().splitlines()
    assert argv[0] == "--qos=qos_a" and argv[1].startswith(f"--output={root}/exp/"), argv
    assert str(sb_dir / job.name) in argv and "--time=1:00:00" in argv, argv
    assert Path(argv[1].split("=", 1)[1]).parent.is_dir()      # the log parent exists before sbatch (Slurm never creates it)
    recorded = dict(ln.split("=", 1) for ln in Path(f"{rec}.env").read_text().splitlines() if "=" in ln)
    j = _job_passes_its_requirements(job, {k: v for k, v in recorded.items() if k.startswith("SISTEER_")})
    assert j.returncode == 0 and f"JOB_SHARED={root}" in j.stdout, j.stderr
    # QOS in the body env overrides the cluster's
    r = run({**site, "QOS": "qos_body"})
    assert r.returncode == 0 and Path(f"{rec}.argv").read_text().splitlines()[0] == "--qos=qos_body", r.stderr
    # without any one setting the job needs (or the QoS), the wrapper refuses by name and submits nothing
    for missing in sorted(_required(job) | {"SISTEER_CLUSTER_QOS"}):
        r = run({k: v for k, v in site.items() if k != missing})
        assert r.returncode != 0 and missing in r.stderr, (missing, r.stderr)
        assert not Path(f"{rec}.argv").exists(), missing


@pytest.mark.parametrize("job", HAND_SUBMITTED_JOBS)
def test_the_front_door_job_env_passes_every_hand_submitted_jobs_requirements(job, tmp_path):
    site = write_site(tmp_path)
    r = subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(site / "config.env"), "job-env", "--cluster",
                        "cluster_a", "--export"], capture_output=True, text=True, env=_clean_env(), timeout=120)
    assert r.returncode == 0, r.stderr
    env = {}
    for ln in r.stdout.splitlines():
        assert ln.startswith("export "), ln
        k, v = shlex.split(ln[len("export "):])[0].split("=", 1)
        env[k] = v
    path = REPO / job
    assert _required(path) <= set(env), sorted(_required(path) - set(env))
    j = _job_passes_its_requirements(path, env)
    assert j.returncode == 0 and f"JOB_SHARED={env['SISTEER_SHARED_ROOT']}" in j.stdout, j.stderr
