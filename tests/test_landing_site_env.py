"""scripts/registry/landing/site_env.sh: the cluster names and paths the landing scripts read from the site config.

Each cluster carries three names with one job each: the site label (the clusters/<label>.env file), the registry label
(CLUSTER_SUBSTRATE: every --dest) and the mirror selector (CLUSTER_MIRROR_SELECTOR, default the site label: only the repo-mirror
tool's --cluster argument). These tests run the helper in a scratch site: the values come from the primary cluster's block, or all
from the block of the cluster CLUSTER_DEST names whether SISTEER_CLUSTERS lists it or not, the environment wins, a missing value or
a CLUSTER_DEST no single block carries is refused by name, the clone and the checkout name derive from the checkout's directory, and
every landing script that sources it refuses by name when no site config supplies a name, while its --help needs none.

``write_site`` is the fixture the other landing tests import to lay out a neutral site."""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LANDING = REPO / "scripts" / "registry" / "landing"
SITE_ENV = LANDING / "site_env.sh"
_spec = importlib.util.spec_from_file_location("_landing_clusters", REPO / "reproduce" / "lib" / "clusters.py")
CLUSTERS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CLUSTERS)
#: the registry labels the cluster-block reader accepts for the two named clusters, read from it (never spelled here)
SUB_A, SUB_B = CLUSTERS.KNOWN_SUBSTRATES[0], CLUSTERS.KNOWN_SUBSTRATES[1]
KEYS = ("CLUSTER_HOST", "CLUSTER_SHARED", "CLUSTER_DEST", "MIRROR_REPOS_CLUSTER", "SITE_REPO", "MIRROR_REPOS_NAME", "CLUSTER_REPO_MIRROR",
        "SITE_CONFIG_FILE")


def write_site(site_dir: Path, *, repo: str | None = None, primary: str | None = "site_one", clusters: dict | None = None,
               registry_root: str | None = None) -> Path:
    """A neutral site config under ``site_dir``: config.env naming the clusters (``clusters``: label -> extra block keys) and the
    primary, one clusters/<label>.env per cluster with every required key. Returns the config.env path."""
    clusters = clusters if clusters is not None else {"site_one": {}}
    (site_dir / "clusters").mkdir(parents=True, exist_ok=True)
    lines = [f'SISTEER_CLUSTERS="{" ".join(clusters)}"']
    if primary:
        lines.append(f"SISTEER_PRIMARY_CLUSTER={primary}")
    if repo:
        lines.append(f"SISTEER_REPO={repo}")
    if registry_root:
        lines.append(f"SISTEER_REGISTRY_ROOT={registry_root}")
    (site_dir / "config.env").write_text("\n".join(lines) + "\n")
    for i, (label, extra) in enumerate(clusters.items()):
        block = {"CLUSTER_SSH": f"login-{label}", "CLUSTER_SHARED_ROOT": f"/shared/{label}", "CLUSTER_PARTITION": "general",
                 "CLUSTER_QOS": "high", "CLUSTER_SUBSTRATE": (SUB_A, SUB_B)[i % 2], "CLUSTER_LABEL": label}
        block.update(extra)
        (site_dir / "clusters" / f"{label}.env").write_text("".join(f"{k}={v}\n" for k, v in block.items() if v is not None))
    return site_dir / "config.env"


def _load(env: dict, keys: str = "", cwd: str | None = None) -> tuple[int, dict, str]:
    """site_env_load KEYS in a clean bash; returns (rc, the exported values, stderr)."""
    script = (f'set -u -o pipefail; source "{SITE_ENV}"; site_env_load {keys}; rc=$?; '
              + "".join(f'printf "%s=%s\\n" {k} "${{{k}:-}}"; ' for k in KEYS) + 'exit $rc')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, cwd=cwd or "/")
    vals = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    return r.returncode, vals, r.stderr


def _env(home: Path, **extra) -> dict:
    env = {"PATH": os.environ["PATH"], "HOME": str(home)}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def test_the_values_come_from_the_primary_block_and_the_clone_from_the_checkouts_name(tmp_path):
    home = tmp_path / "home"
    write_site(home / ".config" / "sisteer", repo=str(tmp_path / "foo"),
               clusters={"site_one": {}, "site_two": {"CLUSTER_MIRROR_SELECTOR": "two_in_the_tool"}})
    rc, v, err = _load(_env(home))
    assert rc == 0, err
    assert v["CLUSTER_HOST"] == "login-site_one" and v["CLUSTER_SHARED"] == "/shared/site_one" and v["CLUSTER_DEST"] == SUB_A
    # the mirror selector defaults to the site label; the clone and the name derive from the checkout's directory
    assert v["MIRROR_REPOS_CLUSTER"] == "site_one"
    assert v["SITE_REPO"] == str(tmp_path / "foo") and v["MIRROR_REPOS_NAME"] == "foo" and v["CLUSTER_REPO_MIRROR"] == "/shared/site_one/repos/foo"
    assert v["SITE_CONFIG_FILE"] == str(home / ".config" / "sisteer" / "config.env")
    # the second cluster as the primary: its explicit selector, its own registry label
    write_site(home / ".config" / "sisteer", repo=str(tmp_path / "foo"), primary="site_two",
               clusters={"site_one": {}, "site_two": {"CLUSTER_MIRROR_SELECTOR": "two_in_the_tool"}})
    rc, v, err = _load(_env(home))
    assert rc == 0, err
    assert v["CLUSTER_DEST"] == SUB_B and v["MIRROR_REPOS_CLUSTER"] == "two_in_the_tool" and v["CLUSTER_REPO_MIRROR"] == "/shared/site_two/repos/foo"


def test_a_block_that_names_its_clone_keeps_it_and_shared_expands_in_it(tmp_path):
    home = tmp_path / "home"
    write_site(home / ".config" / "sisteer", repo=str(tmp_path / "foo"), clusters={"site_one": {"CLUSTER_REPO_MIRROR": "'$SHARED/repos/elsewhere'"}})
    rc, v, err = _load(_env(home))
    assert rc == 0 and v["CLUSTER_REPO_MIRROR"] == "/shared/site_one/repos/elsewhere", (rc, v, err)


def test_sisteer_config_names_another_site_file_and_the_environment_wins_over_it(tmp_path):
    other = tmp_path / "prepared"
    write_site(other, repo="/checkouts/bar", clusters={"site_one": {"CLUSTER_SSH": "login-prepared"}})
    rc, v, err = _load(_env(tmp_path / "nohome", SISTEER_CONFIG=other / "config.env"))
    assert rc == 0 and v["CLUSTER_HOST"] == "login-prepared" and v["MIRROR_REPOS_NAME"] == "bar", (v, err)
    rc, v, err = _load(_env(tmp_path / "nohome", SISTEER_CONFIG=other / "config.env", CLUSTER_HOST="login-env", MIRROR_REPOS_CLUSTER="sel-env",
                            CLUSTER_REPO_MIRROR="/elsewhere/clone"))
    assert rc == 0 and v["CLUSTER_HOST"] == "login-env" and v["MIRROR_REPOS_CLUSTER"] == "sel-env" and v["CLUSTER_REPO_MIRROR"] == "/elsewhere/clone"
    assert v["CLUSTER_SHARED"] == "/shared/site_one", "a value the environment does not set still comes from the block"


def test_with_every_value_in_the_environment_no_site_config_is_read(tmp_path):
    env = _env(tmp_path / "nohome", CLUSTER_HOST="h", CLUSTER_SHARED="/s/", CLUSTER_DEST="label_x", MIRROR_REPOS_CLUSTER="m",
               SISTEER_REPO=tmp_path / "baz")
    rc, v, err = _load(env, "CLUSTER_HOST CLUSTER_SHARED CLUSTER_DEST MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR")
    assert rc == 0, err
    assert v["CLUSTER_SHARED"] == "/s" and v["CLUSTER_REPO_MIRROR"] == "/s/repos/baz" and v["SITE_CONFIG_FILE"] == ""


def test_a_missing_value_is_refused_by_name_with_the_setting_that_supplies_it(tmp_path):
    nohome = tmp_path / "nohome"
    for key, words in (("CLUSTER_HOST", "no login host: set CLUSTER_SSH"), ("CLUSTER_SHARED", "no shared root: set CLUSTER_SHARED_ROOT"),
                       ("CLUSTER_DEST", "no registry label: set CLUSTER_SUBSTRATE"),
                       ("MIRROR_REPOS_CLUSTER", "no mirror selector: set CLUSTER_MIRROR_SELECTOR")):
        rc, v, err = _load(_env(nohome), key)
        assert rc == 2 and f"REFUSED: site_env: {words}" in err and str(nohome / ".config" / "sisteer" / "config.env") in err, (key, err)
    # a site config without its primary cluster is refused by name too, not guessed
    home = tmp_path / "home"
    write_site(home / ".config" / "sisteer", primary=None, clusters={"site_one": {}, "site_two": {}})
    rc, v, err = _load(_env(home), "CLUSTER_HOST")
    assert rc == 2 and "sets no SISTEER_PRIMARY_CLUSTER" in err, err
    # an unknown key is a caller's mistake, named
    rc, v, err = _load(_env(home), "CLUSTER_NOPE")
    assert rc == 2 and "unknown key CLUSTER_NOPE" in err


def list_only(site_dir: Path, *labels: str) -> None:
    """Rewrite the site's SISTEER_CLUSTERS (the front door's default fan-out) to ``labels``; every block file stays in clusters/."""
    cfg = site_dir / "config.env"
    lines = [f'SISTEER_CLUSTERS="{" ".join(labels)}"' if ln.startswith("SISTEER_CLUSTERS=") else ln for ln in cfg.read_text().splitlines()]
    cfg.write_text("\n".join(lines) + "\n")


def test_a_named_registry_label_reads_every_value_from_that_clusters_block_listed_or_not(tmp_path):
    """CLUSTER_DEST names a cluster by registry label: the login host, the shared root, the mirror selector and the clone all come from the
    block that carries it, among every clusters/<label>.env in the site directory, whether SISTEER_CLUSTERS lists it or not. Unset, the
    primary's block supplies every value, CLUSTER_DEST included."""
    home = tmp_path / "home"
    site = home / ".config" / "sisteer"
    write_site(site, repo=str(tmp_path / "foo"), clusters={"site_one": {}, "site_two": {"CLUSTER_MIRROR_SELECTOR": "two_in_the_tool"}})
    list_only(site, "site_one")
    rc, v, err = _load(_env(home, CLUSTER_DEST=SUB_B))
    assert rc == 0, err
    assert (v["CLUSTER_HOST"], v["CLUSTER_SHARED"], v["CLUSTER_DEST"], v["MIRROR_REPOS_CLUSTER"], v["CLUSTER_REPO_MIRROR"]) == (
        "login-site_two", "/shared/site_two", SUB_B, "two_in_the_tool", "/shared/site_two/repos/foo"), v
    rc, v, err = _load(_env(home))
    assert rc == 0 and (v["CLUSTER_HOST"], v["CLUSTER_DEST"], v["MIRROR_REPOS_CLUSTER"]) == ("login-site_one", SUB_A, "site_one"), (v, err)
    rc, v, err = _load(_env(home, CLUSTER_DEST=SUB_A))
    assert rc == 0 and (v["CLUSTER_HOST"], v["CLUSTER_SHARED"]) == ("login-site_one", "/shared/site_one"), "the primary's own label: its block"
    # an environment value still wins over the named block, key by key
    rc, v, err = _load(_env(home, CLUSTER_DEST=SUB_B, CLUSTER_HOST="login-env"))
    assert rc == 0 and v["CLUSTER_HOST"] == "login-env" and v["CLUSTER_SHARED"] == "/shared/site_two", (v, err)


def test_a_named_registry_label_no_single_block_carries_is_refused_by_name(tmp_path):
    """A CLUSTER_DEST no block carries, one that two blocks carry, or a site label written where the registry label goes is refused by
    name: no value is taken from the primary's block in its place."""
    home = tmp_path / "home"
    site = home / ".config" / "sisteer"
    write_site(site, clusters={"site_one": {}, "site_two": {}})
    list_only(site, "site_one")
    rc, v, err = _load(_env(home, CLUSTER_DEST="label_z"), "CLUSTER_HOST")
    assert rc == 2 and "REFUSED: site_env:" in err and "CLUSTER_DEST=label_z: cluster 'label_z' is neither a site label" in err, err
    assert v["CLUSTER_HOST"] == "", "nothing from the primary's block"
    rc, v, err = _load(_env(home, CLUSTER_DEST="site_two"), "CLUSTER_HOST")
    assert rc == 2 and f"CLUSTER_DEST=site_two is the site label of clusters/site_two.env, whose registry label is {SUB_B}" in err, err
    two = (site / "clusters" / "site_two.env").read_text()
    (site / "clusters" / "site_three.env").write_text(two.replace("CLUSTER_LABEL=site_two", "CLUSTER_LABEL=site_three"))
    rc, v, err = _load(_env(home, CLUSTER_DEST=SUB_B), "CLUSTER_HOST")
    assert rc == 2 and f"registry label {SUB_B!r} is carried by more than one cluster block (" in err, err
    assert "site_two" in err and "site_three" in err, err
    rc, v, err = _load(_env(home, CLUSTER_DEST=SUB_A), "CLUSTER_HOST")
    assert rc == 0 and v["CLUSTER_HOST"] == "login-site_one", "a label one block carries still resolves"


def test_the_checkout_falls_back_to_the_main_working_tree_of_the_repository_holding_the_helper(tmp_path):
    """No SISTEER_REPO anywhere: the checkout is the main working tree of the repository site_env.sh sits in, so a copy of the landing
    scripts inside a linked worktree still names the checkout the mirror tool keys the clone by."""
    genv = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")
    main = tmp_path / "foo"
    (main / "scripts").mkdir(parents=True)
    (main / "scripts" / "site_env.sh").write_text(SITE_ENV.read_text())
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "seed"], ["worktree", "add", "-q", str(main / "worktrees" / "lane"), "-b", "lane"]):
        subprocess.run(["git", *args], cwd=str(main), check=True, env=genv, capture_output=True)
    copy = main / "worktrees" / "lane" / "scripts" / "site_env.sh"
    script = f'set -u; source "{copy}"; site_env_load SITE_REPO MIRROR_REPOS_NAME; echo "$SITE_REPO $MIRROR_REPOS_NAME"'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=_env(tmp_path / "nohome"))
    assert r.returncode == 0 and r.stdout.split() == [str(main), "foo"], r.stdout + r.stderr


LANDING_RUNS = {
    # script -> (argv after the script, the exit code of its refusal)
    "gate_slurm.sh": (["a" * 40], 2),
    "battery_slurm.sh": (["a" * 40, "--files", "tests/test_x.py"], 2),
    "baseline_refresh.sh": ([], 1),
    "live_db_slurm.sh": (["a" * 40, "--branch", "b", "--main", "f" * 40], 5),
    "recover.sh": (["b", "title", "body"], 5),
}


def test_every_landing_script_refuses_by_name_without_a_site_config_and_prints_its_help_without_one(tmp_path):
    for name, (argv, code) in LANDING_RUNS.items():
        env = _env(tmp_path / "nohome", REGISTRY_LANDING_STATE=tmp_path / "state" / name, SISTEER_REPO=tmp_path / "foo")
        r = subprocess.run(["bash", str(LANDING / name), *argv], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == code, (name, r.returncode, r.stdout[-600:], r.stderr[-600:])
        assert "REFUSED: site_env: no login host: set CLUSTER_SSH" in r.stderr, (name, r.stderr[-600:])
        h = subprocess.run(["bash", str(LANDING / name), "--help"], capture_output=True, text=True, env=env, timeout=60)
        assert h.returncode == 0 and h.stdout.startswith(f"# {name}"), (name, h.stdout[:200], h.stderr[-300:])


def test_the_helper_parses_and_spells_no_host_or_path_default():
    assert subprocess.run(["bash", "-n", str(SITE_ENV)], capture_output=True).returncode == 0
    text = SITE_ENV.read_text()
    for token in ("/workspace", "/home/", "repos/si-", "node-"):
        assert token not in text, token
