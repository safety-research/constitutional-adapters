"""03_hpo.sh's live `submit()` mirrors the pinned sha to the target cluster BEFORE its first submission of an invocation (06/09's
`slurm_mirror_clusters` step; paper-companion-repo stage E, 2026-09-30): the stage pins `git rev-parse HEAD` at run time, and on a
shared tree where peers commit between a mirror and a submission the job's `cd <eval root>/<sha>` failed at sbatch time with nothing
submitted. Executes 03's own `submit()` fragment under the sourced libraries against a `local:` site, with `mirror_repo.sh` replaced
by a recorder (the push of a whole repo is not the contract here) and `slurm_submit` by a stub: the recorder sees `--cluster <label>
--sha <HEAD>` exactly once for two submissions and before either; a dry run mirrors nothing."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, assert_no_tokens, write_site  # noqa: E402


def _fragment() -> str:
    src = (REPO / "reproduce/03_hpo.sh").read_text()
    return src[src.index("job_env(){"):src.index("# registry state:")]     # job_env + submit


def _run(tmp_path: Path, *, dry: bool) -> tuple[subprocess.CompletedProcess, Path, str]:
    fake = tmp_path / "fake_a"; fake.mkdir()
    shared = tmp_path / "shared_a"
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{fake}"}, shared={"cluster_a": str(shared)})
    lib = tmp_path / "lib"; lib.mkdir()
    rec = tmp_path / "mirror_calls.txt"
    (lib / "mirror_repo.sh").write_text("#!/bin/bash\n" f'echo "$@" >> {rec}\n' 'echo "mirror_repo: recorder $*"\n')
    (lib / "mirror_repo.sh").chmod(0o755)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    script = (f"set -u; source {REPO}/reproduce/lib/common.sh; source {REPO}/reproduce/lib/slurm.sh; REPRO_CONFIG={site}/config.env; "
              f"REPRO_DRY_RUN={1 if dry else 0}; repro_load_config; REPRO_MODEL=qwen35_4b; export SISTEER_CONFIG={site}/config.env\n"
              f"REPRO_LIB_DIR={lib}; HAVE_SLURM_LIB=1; CLUSTER=cluster_a; SHA={head}; MODEL=qwen35_4b; STAGE=03_hpo; CLASS=relu; "
              f"CHECKOUT={shared}/sisteer-eval-repo/{head}; CAMPAIGN=two_si; CL_HOST=cluster-a-login\n"
              'lwid(){ echo "$MODEL/$STAGE/$CLASS/$1"; }\n'
              'disp(){ echo "$1"; }\n'
              'slurm_submit(){ echo "STUB slurm_submit $*"; }\n'
              f"{_fragment()}\n"
              'submit hpo.relu.recon hpo-m-relu-recon scripts/sbatch/hpo_trial.sbatch "spec root psr_ce_all cfg pairs" 1-00:00:00 --env "HPO_JOB_MODE=recon"\n'
              'submit hpo.relu.r1 hpo-m-relu-r1 scripts/sbatch/hpo_trial.sbatch "design root psr_ce_all cfg pairs 3" 12:00:00 --array 0-15\n')
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "nobody-testhost"}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    return r, rec, head


def test_live_submit_mirrors_the_pinned_sha_once_and_before_the_first_submission(tmp_path):
    r, rec, head = _run(tmp_path, dry=False)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = rec.read_text().splitlines() if rec.exists() else []
    assert len(calls) == 1, calls                                          # once per invocation, not per unit
    assert f"--cluster cluster_a" in calls[0] and f"--sha {head}" in calls[0], calls[0]
    assert out.count("STUB slurm_submit") == 2, out
    assert out.index("mirror_repo: recorder") < out.index("STUB slurm_submit"), out
    assert "--wave-sha " + head in out
    scan = out.replace(str(tmp_path), "<tmp>").replace(os.path.expanduser("~"), "<home>")
    assert_no_tokens(scan, "the submission's output")


def test_dry_run_mirrors_nothing(tmp_path):
    r, rec, _head = _run(tmp_path, dry=True)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert not rec.exists(), rec.read_text() if rec.exists() else ""
    assert "STUB slurm_submit" not in out and out.count("DRY-RUN: ssh") == 2 and "submitted cluster_a DRY hpo.relu.recon" in out, out
