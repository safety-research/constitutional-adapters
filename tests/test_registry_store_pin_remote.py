"""stores: the tick's pin on a REMOTE destination (a login host, not a local directory) when the running revision is not on the cluster's
repo mirror yet and the site's repo-mirror tool is present. The tool is handed the name it knows the cluster by (the site's selector,
:func:`stores.mirror_selector`) and the checkout's directory name (:func:`stores.checkout_dir_name`; a linked worktree names the checkout
it was added from), which is also the name of the destination's default mirror (``repos/<name>``), so the clone the tool fills is the one
the pin checks. A selector the tool does not know is refused by the tool, and the revision then reaches the mirror by the direct push.
The destination's ssh steps run here, through a runner that removes the ssh hop."""
from __future__ import annotations

import os
import shlex
import subprocess

import pytest

from sisteer.registry import stores as ST

HOST = "login-test"
LABEL = "cluster_a"
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.invalid"}
SITE_KEYS = ("CLUSTER_DEST", "MIRROR_REPOS_CLUSTER", "SISTEER_CONFIG", "SISTEER_SITE_DIR")


def _git(repo, *args):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=GIT_ENV).stdout.strip()


def _write(path, text):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _on_mirror(mirror, sha):
    return subprocess.run(["git", "-C", str(mirror), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, env=GIT_ENV).returncode == 0


def _site(tmp_path, shared, selector=None):
    """A site config naming two clusters: ``site_one`` (registry label cluster_a, the tool's name for it ``selector`` when given) and
    ``site_two`` (cluster_b, no selector)."""
    site = tmp_path / "site"
    _write(site / "config.env", 'SISTEER_CLUSTERS="site_one site_two"\nSISTEER_PRIMARY_CLUSTER=site_one\n')
    for label, sub, extra in (("site_one", "cluster_a", f"CLUSTER_MIRROR_SELECTOR={selector}\n" if selector else ""),
                              ("site_two", "cluster_b", "")):
        _write(site / "clusters" / f"{label}.env", f"CLUSTER_SSH={HOST}\nCLUSTER_SHARED_ROOT={shared}\nCLUSTER_PARTITION=part\n"
                                                    f"CLUSTER_QOS=qos\nCLUSTER_SUBSTRATE={sub}\nCLUSTER_LABEL={label}\n{extra}")
    return str(site / "config.env")


class SshHere:
    """The grading host's runner with the ssh hop removed: a destination step (``ssh <opts> HOST <command>``) runs its command here and a
    push to ``HOST:<path>`` goes to ``<path>``; the checkout's git steps and the repo-mirror tool run as given."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, timeout_s, input=None):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "ssh":
            assert argv[-2] == HOST, argv
            argv = ["sh", "-c", argv[-1]]
        else:
            argv = [a[len(HOST) + 1:] if a.startswith(HOST + ":") else a for a in argv]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, input=input, env=GIT_ENV)

    def direct_pushes(self):
        return [c for c in self.calls if any("refs/heads/gate/" in a for a in c)]


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A checkout ``checkouts/foo`` with a linked worktree whose branch carries a fresh commit, the cluster's clone of it at
    ``<shared>/repos/foo`` holding only the base commit, and a repo-mirror tool that knows one cluster, by the name ``tool_one``,
    and fills ``<shared>/repos/<name>`` from ``checkouts/<name>`` (every branch)."""
    for key in SITE_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site"))       # no site config unless a test names one
    main = tmp_path / "checkouts" / "foo"
    for rel in ("src/sisteer/__init__.py", "src/sisteer/registry/__init__.py", "src/sisteer/registry/stores.py",
                "scripts/registry/landing/store_verify.sbatch"):
        _write(main / rel, "# the pinned verifier\n")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "base")
    wt = tmp_path / "worktrees" / "wt"
    _git(main, "worktree", "add", "-q", "-b", "feature", str(wt))
    shared = tmp_path / "shared"
    mirror = shared / "repos" / "foo"
    subprocess.run(["git", "init", "-q", "--bare", str(mirror)], check=True, env=GIT_ENV)
    _git(main, "push", "-q", str(mirror), "main")
    _write(wt / "src/sisteer/registry/stores.py", "# the pinned verifier, revised\n")
    _git(wt, "commit", "-q", "-am", "fresh")
    sha = _git(wt, "rev-parse", "HEAD")
    assert not _on_mirror(mirror, sha), "the fresh commit is not on the cluster's clone yet"
    log = tmp_path / "tool.log"
    tool = tmp_path / "bin" / "mirror-repos.sh"
    _write(tool, "#!/bin/sh\n"
                 f"echo \"$*\" >> {shlex.quote(str(log))}\n"
                 "[ \"$1\" = --cluster ] && [ \"$2\" = tool_one ] || { echo \"unknown cluster: $2\" >&2; exit 2; }\n"
                 f"exec git -c core.hooksPath=/dev/null -C {shlex.quote(str(main.parent))}/\"$3\" push -q "
                 f"{shlex.quote(str(shared))}/repos/\"$3\" '+refs/heads/*:refs/heads/*'\n")
    os.chmod(tool, 0o755)
    monkeypatch.setattr(ST, "MIRROR_REPOS", str(tool))
    dest = ST.Destination(name=LABEL, host=HOST, shared=str(shared), source_repo=str(wt))
    return {"dest": dest, "wt": wt, "shared": shared, "mirror": mirror, "sha": sha, "log": log}


def _tool_calls(w):
    return w["log"].read_text().splitlines() if w["log"].exists() else []


def _pinned(w, pin):
    assert pin.revision == w["sha"] and pin.tree == w["dest"].tree_dir(w["sha"])
    assert open(os.path.join(pin.tree, "REVISION")).read().strip() == w["sha"] and os.path.isfile(pin.sbatch_path)
    assert open(os.path.join(pin.tree, "src/sisteer/registry/stores.py")).read() == "# the pinned verifier, revised\n"


def test_the_tool_gets_the_site_selector_and_the_checkout_name_and_fills_the_clone_the_pin_checks(remote, tmp_path, monkeypatch):
    w = remote
    monkeypatch.setenv("SISTEER_CONFIG", _site(tmp_path, w["shared"], selector="tool_one"))
    dest = w["dest"]
    assert dest.repo_rel == "repos/foo" and dest.mirror_dir == str(w["mirror"]), "the default mirror is named after the checkout"
    run = SshHere()
    pin = ST.pin_cluster_tree(dest, ST.Remote(dest, run))
    assert _tool_calls(w) == ["--cluster tool_one foo"], "the tool's name for the cluster and the checkout's name, not the label"
    assert pin.pushed_via == "mirror-repos.sh" and not run.direct_pushes()
    assert _on_mirror(w["mirror"], w["sha"]) and any(c[0] == "ssh" for c in run.calls), "the mirror checks ran on the destination"
    _pinned(w, pin)


def test_the_landing_scripts_selector_pair_names_the_cluster_when_it_names_this_label(remote, monkeypatch):
    w = remote
    monkeypatch.setenv("CLUSTER_DEST", LABEL)
    monkeypatch.setenv("MIRROR_REPOS_CLUSTER", "tool_one")
    run = SshHere()
    pin = ST.pin_cluster_tree(w["dest"], ST.Remote(w["dest"], run))
    assert _tool_calls(w) == ["--cluster tool_one foo"] and pin.pushed_via == "mirror-repos.sh" and not run.direct_pushes()
    _pinned(w, pin)


def test_a_selector_the_tool_refuses_leaves_the_revision_to_the_direct_push(remote, tmp_path, monkeypatch):
    """No site config and no exported pair: the selector is the registry label, which this tool does not know (rc 2); the pin goes on
    and pushes the revision to the mirror itself. A site block without a selector names the cluster by its site label, refused alike."""
    w = remote
    run = SshHere()
    pin = ST.pin_cluster_tree(w["dest"], ST.Remote(w["dest"], run))
    assert _tool_calls(w) == [f"--cluster {LABEL} foo"]
    assert pin.pushed_via == f"git push gate/{w['sha'][:7]}" and len(run.direct_pushes()) == 1
    assert _git(w["mirror"], "rev-parse", f"gate/{w['sha'][:7]}") == w["sha"]
    _pinned(w, pin)
    w["log"].unlink()
    again = ST.pin_cluster_tree(w["dest"], ST.Remote(w["dest"], SshHere()))
    assert _tool_calls(w) == [] and again.pushed_via == "already on the mirror" and again.tree == pin.tree
    _write(w["wt"] / "src/sisteer/registry/stores.py", "# the pinned verifier, revised twice\n")
    _git(w["wt"], "commit", "-q", "-am", "fresh again")
    sha2 = _git(w["wt"], "rev-parse", "HEAD")
    monkeypatch.setenv("SISTEER_CONFIG", _site(tmp_path, w["shared"]))
    run2 = SshHere()
    pin2 = ST.pin_cluster_tree(w["dest"], ST.Remote(w["dest"], run2))
    assert _tool_calls(w) == ["--cluster site_one foo"], "a block without a selector: its site label"
    assert pin2.revision == sha2 and pin2.pushed_via == f"git push gate/{sha2[:7]}" and len(run2.direct_pushes()) == 1
    assert open(os.path.join(pin2.tree, "REVISION")).read().strip() == sha2


def test_the_selector_is_the_exported_pair_then_the_site_block_then_the_label(tmp_path):
    shared = tmp_path / "shared"
    cfg = {"SISTEER_CONFIG": _site(tmp_path, shared, selector="tool_one")}
    assert ST.mirror_selector("cluster_a", cfg) == "tool_one"
    assert ST.mirror_selector("cluster_b", cfg) == "site_two", "a block without a selector: its site label"
    pair = {**cfg, "CLUSTER_DEST": "cluster_a", "MIRROR_REPOS_CLUSTER": "from_env"}
    assert ST.mirror_selector("cluster_a", pair) == "from_env"
    assert ST.mirror_selector("cluster_b", pair) == "site_two", "the pair names the selector of its own label only"
    assert ST.mirror_selector("label_z", cfg) == "label_z", "a label no block names: the label itself"
    assert ST.mirror_selector("cluster_a", {"SISTEER_CONFIG": str(tmp_path / "none" / "config.env")}) == "cluster_a"
    assert ST.mirror_selector("cluster_a", {"SISTEER_SITE_DIR": str(tmp_path / "site")}) == "tool_one", "the site directory's config.env"


def test_a_destination_outside_the_default_fan_out_takes_the_selector_of_its_own_block(tmp_path):
    """SISTEER_CLUSTERS is the front door's default fan-out, not the set of clusters a store destination can name: a cluster whose block
    file sits in the site's clusters/ directory unlisted still hands the tool its own selector. Two block files carrying one registry
    label are ambiguous, so the label itself goes to the tool (which refuses it, and the revision goes by the direct push)."""
    shared = tmp_path / "shared"
    cfg = {"SISTEER_CONFIG": _site(tmp_path, shared, selector="tool_one")}
    site = tmp_path / "site"
    _write(site / "config.env", 'SISTEER_CLUSTERS="site_one"\nSISTEER_PRIMARY_CLUSTER=site_one\n')
    two = (site / "clusters" / "site_two.env").read_text()
    _write(site / "clusters" / "site_two.env", two + "CLUSTER_MIRROR_SELECTOR=tool_two\n")
    assert ST.mirror_selector("cluster_b", cfg) == "tool_two", "an unlisted cluster's block: its own selector"
    assert ST.mirror_selector("cluster_a", cfg) == "tool_one"
    _write(site / "clusters" / "site_three.env", two.replace("CLUSTER_LABEL=site_two", "CLUSTER_LABEL=site_three")
           + "CLUSTER_MIRROR_SELECTOR=tool_three\n")
    assert ST.mirror_selector("cluster_b", cfg) == "cluster_b", "two blocks carry the label: neither block's selector"
    assert ST.mirror_selector("cluster_a", cfg) == "tool_one", "a label one block carries still resolves"


class SbatchHere(SshHere):
    """SshHere, with the destination's `sbatch` recorded and answered with a job id instead of run."""

    def __call__(self, argv, timeout_s, input=None):
        if argv[0] == "ssh" and argv[-1].startswith("sbatch "):
            self.calls.append(list(argv))
            return subprocess.CompletedProcess(list(argv), 0, stdout="4242\n", stderr="")
        return super().__call__(argv, timeout_s, input)


def test_the_verifier_job_logs_under_the_destinations_shared_root(tmp_path):
    """The verifier's log path is the destination's: ``<shared root>/exp/logs/<job name>_<job id>.out``, its directory created on the
    destination before the submission, on a shared root no fixed log path could name."""
    shared = tmp_path / "cluster_b_root" / "u"
    dest = ST.Destination(name=LABEL, host=HOST, shared=str(shared))
    assert dest.log_dir == f"{shared}/exp/logs"
    run = SbatchHere()
    script = "/cluster/trees/abc/scripts/registry/landing/store_verify.sbatch"
    job, _tlim = ST.Remote(dest, run).submit("store-verify-x", ["verify", "g", "m", "o", "2"], 10, 1, 100.0, script=script)
    assert job == "4242"
    i_mk = next(i for i, c in enumerate(run.calls) if c[0] == "ssh" and c[-1].startswith("sh -c ") and "mkdir -p" in c[-1])
    i_sb = next(i for i, c in enumerate(run.calls) if c[0] == "ssh" and c[-1].startswith("sbatch "))
    assert i_mk < i_sb and (shared / "exp" / "logs").is_dir(), "the log directory exists before the job can open its log"
    sb = shlex.split(run.calls[i_sb][-1])
    assert f"--output={shared}/exp/logs/%x_%j.out" in sb and sb.index(f"--output={shared}/exp/logs/%x_%j.out") < sb.index(script), sb
    assert sb[sb.index(script) + 1:] == ["verify", "g", "m", "o", "2"]
