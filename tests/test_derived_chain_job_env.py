"""submit_derived_chain.sh hands every derived-build job the cluster's shared root and QoS from that cluster's site block
(SISTEER_SHARED_ROOT, SISTEER_CLUSTER_QOS): the job script reads its root from the job env and refuses without it, so the
chain must carry it. A site block without CLUSTER_SHARED_ROOT is refused by name before anything is submitted.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "seedrep" / "submit_derived_chain.sh"
JOB = REPO / "scripts" / "sbatch" / "sdfbase_derived_build.sbatch"
SHA = "a" * 40


def _world(tmp_path: Path, *, shared: bool = True) -> dict:
    binp = tmp_path / "bin"
    binp.mkdir()
    (binp / "sbatch").write_text(
        "#!/bin/bash\n"
        f"c={tmp_path}/counter; n=$(( $(cat \"$c\" 2>/dev/null || echo 2000) + 1 )); echo $n > \"$c\"\n"
        f"printf '%s\\t%s\\t%s\\n' \"$n\" \"${{SISTEER_SHARED_ROOT:-}}\" \"${{SISTEER_CLUSTER_QOS:-}}\" >> {tmp_path}/sbatch.log\n"
        "echo \"$n;cluster\"\n")
    (binp / "sbatch").chmod(0o755)
    site = tmp_path / "site"
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="site_one site_two"\nSISTEER_PRIMARY_CLUSTER=site_one\n')
    for k, reg, lab in (("a", "cluster_a", "site_one"), ("b", "cluster_b", "site_two")):
        root = f"CLUSTER_SHARED_ROOT=/shared-{k}\n" if shared else ""
        (site / "clusters" / f"{lab}.env").write_text(
            f"CLUSTER_SSH=login-{k}\n{root}CLUSTER_PARTITION=p-{k}\nCLUSTER_QOS=qos-{k}\nCLUSTER_SUBSTRATE={reg}\n"
            f"CLUSTER_LABEL={lab}\nCLUSTER_EVAL_REPO_ROOT=/eval-{k}\nCLUSTER_LOG_DIR=/logs-{k}\n")
    env = {k: v for k, v in os.environ.items() if k not in ("RS", "SISTEER_SHARED_ROOT", "SISTEER_CLUSTER_QOS")}
    env.update(PATH=f"{binp}:{env['PATH']}", SISTEER_CONFIG=str(site / "config.env"))
    return env


def _run(tmp_path: Path, env: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), "synthetic_key", "synthetic_bank", "--sha", SHA, "--local", "--repo", "/ckpt/F",
                           "--rs", str(tmp_path / "rs"), *extra], capture_output=True, text=True, env=env, timeout=60)


def test_every_job_carries_the_cluster_root_and_qos(tmp_path):
    r = _run(tmp_path, _world(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [l.split("\t") for l in (tmp_path / "sbatch.log").read_text().splitlines()]
    assert rows and all(row[1:] == ["/shared-a", "qos-a"] for row in rows), rows


def test_a_block_without_a_shared_root_is_refused_before_any_submission(tmp_path):
    r = _run(tmp_path, _world(tmp_path, shared=False))
    assert r.returncode == 2 and "CLUSTER_SHARED_ROOT" in r.stderr, r.stderr
    assert not (tmp_path / "sbatch.log").exists()


def test_the_job_reads_its_root_from_the_job_env_only():
    text = JOB.read_text()
    assert 'SHARED="$SISTEER_SHARED_ROOT"' in text and 'CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"' in text
    assert '[ -d "/workspace' not in text and "SHARED=/workspace" not in text
