"""reproduce/lib/mirror_repo.sh and lib/hf_fetch.sh: mirror_repo composes the mirror push + the
detached checkout for a configured label (dry run) and refuses a non-commit; hf_fetch takes ONE repo per call (a second repo
id is refused — a multi-repo `hf download` fetches nothing with exit 0), supports --repo-type dataset / --revision /
--include, and under a fake ssh the token travels on STDIN only — never on argv, never printed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, write_fake_ssh, write_site  # noqa: E402

SHA = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def test_mirror_repo_dry_run_and_refusals(tmp_path):
    site = write_site(tmp_path)
    env = dict(os.environ, SISTEER_STATE_DIR=str(tmp_path / "state"))
    r = subprocess.run(["bash", str(LIB / "mirror_repo.sh"), "--cluster", "cluster_b", "--sha", SHA, "--dry-run", "--config", str(site / "config.env")], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"ssh cluster-b-login" in r.stdout and f"/shared_cluster_b/repos/constitutional-adapters" in r.stdout
    assert f"git -C {REPO} push -q ssh://cluster-b-login{tmp_path}/shared_cluster_b/repos/constitutional-adapters" in r.stdout
    assert f"worktree add --detach '{tmp_path}/shared_cluster_b/sisteer-eval-repo/{SHA}' '{SHA}'" in r.stdout
    bad = subprocess.run(["bash", str(LIB / "mirror_repo.sh"), "--cluster", "cluster_b", "--sha", "f" * 40, "--dry-run", "--config", str(site / "config.env")], capture_output=True, text=True, env=env)
    assert bad.returncode == 2 and "is not a commit" in bad.stderr
    short = subprocess.run(["bash", str(LIB / "mirror_repo.sh"), "--cluster", "cluster_b", "--sha", "abc", "--config", str(site / "config.env")], capture_output=True, text=True, env=env)
    assert short.returncode == 2 and "40hex" in short.stderr


def test_hf_fetch_one_repo_per_call_token_on_stdin(tmp_path):
    site = write_site(tmp_path)
    rec = tmp_path / "ssh.log"; stdin_rec = tmp_path / "ssh.stdin"
    write_fake_ssh(tmp_path / "bin", rec, stdout="[cluster_a] cached\n", rc=0, stdin_record=stdin_rec)
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}", SISTEER_STATE_DIR=str(tmp_path / "state"), SISTEER_SECRETS_MODE="env", HF_TOKEN="synthetic-hf-token")
    r = subprocess.run(["bash", str(LIB / "hf_fetch.sh"), "--cluster", "cluster_a", "--repo", "org/model", "--revision", "e" * 40, "--include", "tokenizer*", "--include", "special_tokens_map.json",
                        "--config", str(site / "config.env")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    argv = json.loads(rec.read_text().splitlines()[-1])
    remote = argv[-1]
    assert argv[-2] == "cluster-a-login"
    assert "synthetic-hf-token" not in json.dumps(argv) and "synthetic-hf-token" not in r.stdout + r.stderr     # never argv, never printed
    assert stdin_rec.read_text().strip() == "synthetic-hf-token"                                                   # stdin only
    assert "hf download --repo-type model org/model --revision " + "e" * 40 in remote and "--include tokenizer\\* --include special_tokens_map.json" in remote
    assert "srun -p gpu_cluster_a --qos=qos_cluster_a --account=acct_cluster_a" in remote and f"HF_HOME={tmp_path}/shared_cluster_a/hf_cache" in remote
    assert "read -r T" in remote and 'export HF_TOKEN="$T"' in remote and "unset T" in remote
    # datasets, and the refusals
    r2 = subprocess.run(["bash", str(LIB / "hf_fetch.sh"), "--cluster", "cluster_a", "--repo", "org/dataset", "--repo-type", "dataset", "--dry-run", "--config", str(site / "config.env")], capture_output=True, text=True, env=env)
    assert r2.returncode == 0 and "--repo-type dataset org/dataset" in r2.stdout and "<token from source 'env' on stdin>" in r2.stdout
    two = subprocess.run(["bash", str(LIB / "hf_fetch.sh"), "--cluster", "cluster_a", "--repo", "org/a org/b", "--dry-run", "--config", str(site / "config.env")], capture_output=True, text=True, env=env)
    assert two.returncode == 2 and "one repo per call" in two.stderr
    extra = subprocess.run(["bash", str(LIB / "hf_fetch.sh"), "--cluster", "cluster_a", "--repo", "org/a", "org/b", "--dry-run", "--config", str(site / "config.env")], capture_output=True, text=True, env=env)
    assert extra.returncode == 2 and "one --repo per call" in extra.stderr
    # token source none: an empty line on stdin, no HF_TOKEN export required
    none_site = write_site(tmp_path / "n", extra_cluster="")
    for f in (none_site / "clusters").glob("*.env"):
        f.write_text(f.read_text().replace("CLUSTER_HF_TOKEN_SOURCE=env", "CLUSTER_HF_TOKEN_SOURCE=none"))
    env2 = dict(env); env2.pop("HF_TOKEN")
    r3 = subprocess.run(["bash", str(LIB / "hf_fetch.sh"), "--cluster", "cluster_a", "--repo", "org/public", "--config", str(none_site / "config.env")], capture_output=True, text=True, env=env2, timeout=60)
    assert r3.returncode == 0 and stdin_rec.read_text().strip() == ""
