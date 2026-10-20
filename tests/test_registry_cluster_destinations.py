"""The names a cluster destination is reached by, derived the same way in the landing scripts (site_env.sh) and in Python
(sisteer.registry.reads_protocol): the checkout's directory name names the clone on the cluster (<shared>/repos/<name>) and the gate
venv (<shared>/envs/<name>-gate-<hash12>); the repo-mirror tool gets the cluster's mirror selector, never its registry label. A checkout
named ``foo`` gives repos/foo and envs/foo-gate-* on both sides, and a linked worktree of it gives the same."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path

from sisteer.registry import reads_protocol as RP
from sisteer.registry import stores as ST

from test_landing_site_env import SITE_ENV, SUB_A, SUB_B, list_only, write_site

REPO = Path(__file__).resolve().parents[1]
LANDING = REPO / "scripts" / "registry" / "landing"
GENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")


def _checkout(tmp_path: Path, name: str = "foo") -> tuple[Path, Path, str]:
    """A git checkout named ``name`` with the two dependency files committed, and a linked worktree of it; returns (main, worktree, sha)."""
    main = tmp_path / name
    main.mkdir()
    (main / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (main / "uv.lock").write_text("version = 1\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "seed"], ["worktree", "add", "-q", str(main / "worktrees" / "lane"), "-b", "lane"]):
        subprocess.run(["git", *args], cwd=str(main), check=True, env=GENV, capture_output=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(main), capture_output=True, text=True, check=True).stdout.strip()
    return main, main / "worktrees" / "lane", sha


def _shell(script: str, env: dict) -> str:
    r = subprocess.run(["bash", "-c", f'set -u -o pipefail; source "{SITE_ENV}"; {script}'], capture_output=True, text=True, env=env, cwd="/")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def test_a_checkout_named_foo_gives_repos_foo_and_envs_foo_gate_in_shell_and_python_alike(tmp_path):
    main, wt, sha = _checkout(tmp_path)
    assert RP.checkout_name(str(main)) == "foo" and RP.checkout_name(str(wt)) == "foo", "a linked worktree names the checkout it came from"
    assert RP.checkout_name(str(tmp_path / "not-a-repo")) == "not-a-repo"
    # the gate venv: Python reads the dependency files from the git objects at the tree; the shell hashes them the window's way
    py_venv = RP.gate_venv_path(str(wt), sha, "/shared/x")
    ph = hashlib.sha256((main / "pyproject.toml").read_bytes()).hexdigest()
    lh = hashlib.sha256((main / "uv.lock").read_bytes()).hexdigest()
    h12 = hashlib.sha256(f"{ph}\n{lh}\n".encode()).hexdigest()[:12]
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "nohome"), "CLUSTER_SHARED": "/shared/x", "SISTEER_REPO": str(main)}
    sh_venv = _shell(f'site_env_load CLUSTER_SHARED MIRROR_REPOS_NAME; site_env_gate_venv "$CLUSTER_SHARED" "$MIRROR_REPOS_NAME" {h12}', env)
    assert py_venv == sh_venv == f"/shared/x/envs/foo-gate-{h12}", (py_venv, sh_venv)
    # the clone: the shell's derived path is the Python side's <shared>/repos/<checkout name>
    clone = _shell("site_env_load CLUSTER_SHARED CLUSTER_REPO_MIRROR; echo $CLUSTER_REPO_MIRROR", env)
    assert clone == f"/shared/x/repos/{RP.checkout_name(str(wt))}" == "/shared/x/repos/foo"


def test_the_mirror_selector_comes_from_the_environment_pair_then_the_site_block_then_the_label(tmp_path):
    site = tmp_path / "site"
    write_site(site, clusters={"site_one": {"CLUSTER_MIRROR_SELECTOR": "one_in_the_tool"}, "site_two": {}})
    cfg = {"SISTEER_CONFIG": str(site / "config.env")}
    assert RP.mirror_selector(SUB_A, cfg) == "one_in_the_tool"
    assert RP.mirror_selector(SUB_B, cfg) == "site_two", "no selector in the block: its site label"
    # the landing scripts' exported pair wins for its own registry label, and only for it
    pair = dict(cfg, CLUSTER_DEST=SUB_A, MIRROR_REPOS_CLUSTER="from_env")
    assert RP.mirror_selector(SUB_A, pair) == "from_env" and RP.mirror_selector(SUB_B, pair) == "site_two"
    # no site config, or a label no block carries: the label itself (the tool says whether it knows it)
    assert RP.mirror_selector("label_z", {"SISTEER_CONFIG": str(tmp_path / "none" / "config.env")}) == "label_z"
    assert RP.mirror_selector("label_z", cfg) == "label_z"
    # the shell side reads the same block the same way
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "nohome"), "SISTEER_CONFIG": str(site / "config.env")}
    assert _shell("site_env_load CLUSTER_DEST MIRROR_REPOS_CLUSTER; echo $CLUSTER_DEST $MIRROR_REPOS_CLUSTER", env).split() == [SUB_A, "one_in_the_tool"]


def test_the_reads_tree_hands_the_mirror_tool_the_selector_and_the_checkout_name(tmp_path):
    main, wt, sha = _checkout(tmp_path)
    site = tmp_path / "site"
    write_site(site, clusters={"site_one": {"CLUSTER_MIRROR_SELECTOR": "one_in_the_tool"}})
    calls = []

    def runner(argv, timeout_s, input=None):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="TREE existing /shared/gates/trees/" + sha + "\n", stderr="")
    dest = ST.Destination(name=SUB_A, host="login-shim", shared="/shared", source_repo=str(wt), repo_rel="repos/foo")
    export = tmp_path / "landing" / "tree_export.sh"
    export.parent.mkdir()
    export.write_text("#!/bin/bash\necho TREE existing\n")
    old = os.environ.get("SISTEER_CONFIG")
    os.environ["SISTEER_CONFIG"] = str(site / "config.env")
    try:
        RP.ensure_tree(dest, sha, "/x/mirror-repos.sh", str(export), runner=runner, log=lambda s: None)
    finally:
        if old is None:
            os.environ.pop("SISTEER_CONFIG", None)
        else:
            os.environ["SISTEER_CONFIG"] = old
    assert calls[0] == ["/x/mirror-repos.sh", "--cluster", "one_in_the_tool", "foo"], calls[0]


def test_a_read_destination_outside_the_default_fan_out_hands_the_tool_its_own_blocks_selector(tmp_path, monkeypatch):
    """A cluster whose block file sits in the site's clusters/ directory but not in SISTEER_CLUSTERS (the front door's default fan-out) is
    still addressed by its registry label: the reads tree hands the repo-mirror tool that block's selector, and the shell side with
    CLUSTER_DEST set reads every value from the same block. Two blocks carrying one label: the label itself, for the tool to judge."""
    main, wt, sha = _checkout(tmp_path)
    site = tmp_path / "site"
    write_site(site, clusters={"site_one": {"CLUSTER_MIRROR_SELECTOR": "one_in_the_tool"}, "site_two": {"CLUSTER_MIRROR_SELECTOR": "two_in_the_tool"}})
    list_only(site, "site_one")
    cfg = {"SISTEER_CONFIG": str(site / "config.env")}
    assert RP.mirror_selector(SUB_B, cfg) == "two_in_the_tool" and RP.mirror_selector(SUB_A, cfg) == "one_in_the_tool"
    calls = []

    def runner(argv, timeout_s, input=None):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="TREE existing /shared/site_two/gates/trees/" + sha + "\n", stderr="")
    dest = ST.Destination(name=SUB_B, host="login-site_two", shared="/shared/site_two", source_repo=str(wt), repo_rel="repos/foo")
    export = tmp_path / "landing" / "tree_export.sh"
    export.parent.mkdir()
    export.write_text("#!/bin/bash\necho TREE existing\n")
    for k in ("CLUSTER_DEST", "MIRROR_REPOS_CLUSTER", "SISTEER_SITE_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    RP.ensure_tree(dest, sha, "/x/mirror-repos.sh", str(export), runner=runner, log=lambda s: None)
    assert calls[0] == ["/x/mirror-repos.sh", "--cluster", "two_in_the_tool", "foo"], calls[0]
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "nohome"), "SISTEER_CONFIG": str(site / "config.env"), "CLUSTER_DEST": SUB_B}
    got = _shell("site_env_load CLUSTER_HOST CLUSTER_SHARED MIRROR_REPOS_CLUSTER; echo $CLUSTER_HOST $CLUSTER_SHARED $MIRROR_REPOS_CLUSTER", env)
    assert got.split() == ["login-site_two", "/shared/site_two", "two_in_the_tool"], got
    two = (site / "clusters" / "site_two.env").read_text()
    (site / "clusters" / "site_three.env").write_text(two.replace("CLUSTER_LABEL=site_two", "CLUSTER_LABEL=site_three"))
    assert RP.mirror_selector(SUB_B, cfg) == SUB_B, "two blocks carry the label: neither block's selector"
    assert RP.mirror_selector(SUB_A, cfg) == "one_in_the_tool"


def test_site_cutover_reads_the_same_terminal_op_states_as_the_reads_protocol():
    spec = importlib.util.spec_from_file_location("_site_cutover", LANDING / "site_cutover.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    assert tuple(sc.OP_TERMINAL_STATES) == tuple(RP.OP_TERMINAL_STATES)


def test_no_landing_script_builds_in_a_host_a_shared_root_or_the_old_checkout_name():
    """The login host, the shared root and the clone come from the site config; the gate venv and the clone derive from the checkout's
    name. The cluster-side job scripts read the shared root from their environment (or the venv's path), never from a literal, and name
    no log path: a Slurm directive cannot read the site's shared root, so the submitter passes one under it (the tests of each
    submitter check the argv)."""
    names = ["window.sh", "gate_slurm.sh", "battery_slurm.sh", "baseline_refresh.sh", "live_db_slurm.sh", "recover.sh", "store_verify.sbatch",
             "gate_venv.sbatch", "site_env.sh", "site_cutover.py"]
    old = "si-" + "steering"
    for name in names:
        for i, line in enumerate((LANDING / name).read_text().splitlines(), 1):
            assert "/workspace" not in line and old not in line and not __import__("re").search(r"\bnode-\d+\b", line), (name, i, line)
    for name in ("store_verify.sbatch", "gate_venv.sbatch"):
        directives = [l for l in (LANDING / name).read_text().splitlines() if l.startswith("#SBATCH")]
        assert not any(d.startswith(("#SBATCH --output", "#SBATCH --error", "#SBATCH -o", "#SBATCH -e")) for d in directives), (name, directives)
        assert "#SBATCH --open-mode=append" in directives, (name, directives)
    rp = (REPO / "src" / "sisteer" / "registry" / "reads_protocol.py").read_text()
    assert f"{old}-gate-" not in rp and f'"--cluster", dest.name' not in rp


def test_every_landing_submission_names_its_log_under_the_sites_shared_root():
    """gate_slurm.sh, battery_slurm.sh and live_db_slurm.sh pass --output under <shared root>/exp/logs on every sbatch they issue and create
    that directory first: the job scripts name no log path, and a fixed one would not exist on a cluster whose shared root differs. The
    argv itself is checked against a shim by the gate's and the battery's tests."""
    import re
    for name in ("gate_slurm.sh", "battery_slurm.sh", "live_db_slurm.sh"):
        text = (LANDING / name).read_text()
        assert "CL_LOGS=$SHARED/exp/logs" in text, name
        subs = [l for l in text.splitlines() if re.search(r"\bsbatch --parsable\b", l) and not l.lstrip().startswith("#")]
        assert subs and all(re.search(r'--output="\$(LOGS|CL_LOGS)/%x_(%j|%A_%a)\.out"', l) for l in subs), (name, subs)
        assert re.search(r"mkdir -p [^\n]*\$CL_LOGS", text), name
