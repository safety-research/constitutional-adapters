"""reproduce/06_eval_waves.sh landed as 100644 — the index-only landing staged the blob with the default mode — so a fresh checkout (every
minted cluster checkout) would get a launcher that fails Permission denied when run directly; the grading host's worktree hid it (its
filesystem mode was 775). The rule, pinned IN THE TREE — `git ls-tree -r HEAD`, the committed tree the landing pushed, never the filesystem
(which a checkout does not carry) and never the INDEX (a blob-staging landing can commit 100644 while the checkout's index still records
100755, so `git ls-files -s` passed on exactly the tree it was meant to refuse — the shared checkout's index said 100755 for 06 while
48ea3abe8 and 2f057556f carried 100644): every reproduce/NN_*.sh launcher and every reproduce/lib/*.sh script that is RUN directly is
100755. The sourced libs (attempt_wait, common, hf_env, secrets, slurm) are not launchers and are not pinned here."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

if not (REPO / ".git").exists():                                                                  # a `git archive` export carries no HEAD
    pytest.skip("not a git checkout: tree modes are read from HEAD's tree (a landing's scratch export has none; CI and every clone do)", allow_module_level=True)

RUN_DIRECTLY = ("hf_fetch.sh", "mirror_figs.sh", "mirror_repo.sh", "reconcile.sh", "stop_run.sh", "watch_job.sh")


def _tree_modes(prefix: str, repo: Path = REPO, rev: str = "HEAD") -> dict:
    """``{path: mode}`` under ``prefix`` in ``rev``'s COMMITTED TREE (`git ls-tree -r <rev> -- <prefix>`: `<mode> <type> <blob>\\t<path>`) —
    the tree a landing pushed, which is what every fresh checkout receives. Never `git ls-files -s`: that is the checkout's INDEX, which a
    blob-staging landing leaves at the OLD mode (mode-fix-2), so it passes on exactly the tree this test exists to refuse."""
    r = subprocess.run(["git", "ls-tree", "-r", rev, "--", prefix], cwd=repo, capture_output=True, text=True, check=True)
    out = {}
    for line in r.stdout.splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, _blob = meta.split()
        if kind == "blob":
            out[path] = mode
    assert out, f"git ls-tree -r {rev} saw no blob under {prefix} in {repo} (not a git checkout?)"
    return out


def test_every_stage_launcher_is_executable_in_the_tree():
    modes = _tree_modes("reproduce")
    launchers = sorted(p for p in modes if p.startswith("reproduce/") and p.count("/") == 1 and p[10:12].isdigit() and p.endswith(".sh"))
    assert len(launchers) == 11 and "reproduce/06_eval_waves.sh" in launchers, launchers
    bad = {p: m for p, m in modes.items() if p in launchers and m != "100755"}
    assert not bad, f"stage launchers not 100755 in the tree: {bad}"
    for p in launchers:
        assert (REPO / p).read_text().startswith("#!/"), p                                   # a launcher: run directly, so the mode matters


def test_the_lib_scripts_that_are_run_directly_are_executable_in_the_tree():
    modes = _tree_modes("reproduce/lib")
    bad = {n: modes.get(f"reproduce/lib/{n}") for n in RUN_DIRECTLY if modes.get(f"reproduce/lib/{n}") != "100755"}
    assert not bad, f"lib scripts run directly but not 100755 in the tree: {bad}"


def test_the_committed_tree_is_the_witness_not_the_index(tmp_path):
    """mode-fix-2: the shape that fooled the old read — HEAD carries a launcher at 100644 while the checkout's INDEX records
    100755 (a blob-staging landing committed the default mode; the shared checkout's index kept the executable bit). `git ls-files -s`
    says 100755 and would pass; `_tree_modes` reads HEAD's tree and refuses. Synthetic repo, no personal paths."""
    repo = tmp_path / "repo"; (repo / "reproduce").mkdir(parents=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@TESTHOST", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@TESTHOST", "HOME": str(tmp_path)}
    g = lambda *a: subprocess.run(["git", *a], cwd=repo, env=env, capture_output=True, text=True, check=True)  # noqa: E731
    g("init", "-q"); (repo / "reproduce" / "06_eval_waves.sh").write_text("#!/bin/bash\necho hi\n")
    g("add", "reproduce/06_eval_waves.sh"); g("update-index", "--chmod=-x", "reproduce/06_eval_waves.sh"); g("commit", "-q", "-m", "launcher committed at 100644")
    g("update-index", "--chmod=+x", "reproduce/06_eval_waves.sh")                                 # the index now says 100755; HEAD still says 100644
    index = subprocess.run(["git", "ls-files", "-s", "--", "reproduce"], cwd=repo, env=env, capture_output=True, text=True, check=True).stdout
    assert index.split()[0] == "100755"                                                             # the OLD read would have passed this tree
    assert _tree_modes("reproduce", repo=repo) == {"reproduce/06_eval_waves.sh": "100644"}          # the committed tree is what a checkout gets
    g("update-index", "--chmod=+x", "reproduce/06_eval_waves.sh"); g("commit", "-q", "-m", "chmod +x in the tree")
    assert _tree_modes("reproduce", repo=repo) == {"reproduce/06_eval_waves.sh": "100755"}
