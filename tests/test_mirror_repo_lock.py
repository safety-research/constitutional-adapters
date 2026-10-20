"""`reproduce/lib/mirror_repo.sh` takes ONE run per mirror clone at a time.

Two lanes' mirror runs against the same cluster clone raced: a `git worktree add` in progress read as "dirty checkout" to the
other run (`REFUSED: eval checkout … dirty checkout`, transient, self-healing). The whole remote section now holds a flock on this
host keyed by the target (ssh host + mirror path). Executed: two concurrent runs of the REAL script (the `local:` seam, a tiny
synthetic git repo as `--repo`) — the second acquires the lock only after the first releases it (timestamps on stderr), both end
with a clean checkout; a run on a DIFFERENT mirror does not wait; a lock held past the wait bound refuses by name."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "reproduce" / "lib" / "mirror_repo.sh"
ACQ = re.compile(r"mirror_repo: lock acquired ([0-9.]+) ")
REL = re.compile(r"mirror_repo: lock released ([0-9.]+) ")


def _tiny_repo(root: Path) -> tuple[Path, list[str]]:
    r = root / "tiny"; r.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True, env=env)
    shas = []
    for i in range(2):
        (r / f"f{i}.txt").write_text(f"{i}\n")
        subprocess.run(["git", "-C", str(r), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(r), "commit", "-q", "-m", f"c{i}"], check=True, env=env)
        shas.append(subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, env=env).stdout.strip())
    return r, shas


def _run(root: Path, repo: Path, sha: str, mirror: Path, hold: str | None = None, wait: str | None = None) -> subprocess.Popen:
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "SISTEER_MIRROR_LOCK_DIR": str(root / "locks"),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    if hold is not None:
        env["SISTEER_MIRROR_HOLD_S"] = hold
    if wait is not None:
        env["SISTEER_MIRROR_LOCK_WAIT_S"] = wait
    return subprocess.Popen(["bash", str(SCRIPT), "--ssh", f"local:{root / 'fake'}", "--mirror", str(mirror), "--eval-root", str(root / "eval"),
                             "--sha", sha, "--repo", str(repo)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def _stamps(err: str) -> tuple[float, float]:
    a, r = ACQ.search(err), REL.search(err)
    assert a and r, err
    return float(a.group(1)), float(r.group(1))


def test_two_concurrent_runs_on_one_mirror_serialize_and_both_end_clean(tmp_path):
    repo, shas = _tiny_repo(tmp_path)
    mirror = tmp_path / "shared" / "repos" / "constitutional-adapters"
    a = _run(tmp_path, repo, shas[0], mirror, hold="2")
    time.sleep(0.3)                                                          # A holds the lock; B arrives while it does
    b = _run(tmp_path, repo, shas[1], mirror)
    oa, ea = a.communicate(timeout=120); ob, eb = b.communicate(timeout=120)
    assert a.returncode == 0 and b.returncode == 0, (oa, ea, ob, eb)
    a_acq, a_rel = _stamps(ea); b_acq, b_rel = _stamps(eb)
    assert a_acq < b_acq and b_acq >= a_rel, (a_acq, a_rel, b_acq, b_rel)    # B entered only after A released
    assert a_rel - a_acq >= 2.0                                              # A held it through the seam + its work
    # the report line names the checkout and ends with the remote's verdict (`git worktree add` prints "Preparing worktree …" first)
    assert f"/eval/{shas[0]}" in oa and oa.rstrip().endswith("created") and f"/eval/{shas[1]}" in ob and ob.rstrip().endswith("created"), (oa, ob)
    for sha in shas:                                                         # both checkouts are at their sha and clean
        co = tmp_path / "eval" / sha
        assert subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == sha
        assert subprocess.run(["git", "-C", str(co), "status", "--porcelain"], capture_output=True, text=True).stdout == ""
    # a third run on the SAME sha verifies the existing checkout under the lock, and touches nothing
    c = _run(tmp_path, repo, shas[0], mirror); oc, ec = c.communicate(timeout=120)
    assert c.returncode == 0 and f"/eval/{shas[0]}" in oc and oc.rstrip().endswith("verified"), oc + ec
    assert len(list((tmp_path / "locks").glob("mirror.*.lock"))) == 1      # one lock file per target


def test_a_run_on_another_mirror_does_not_wait_and_a_held_lock_refuses_past_the_bound(tmp_path):
    repo, shas = _tiny_repo(tmp_path)
    m1 = tmp_path / "shared" / "repos" / "one"; m2 = tmp_path / "shared" / "repos" / "two"
    a = _run(tmp_path, repo, shas[0], m1, hold="3")
    time.sleep(0.3)
    b = _run(tmp_path, repo, shas[0], m2)                                    # a different target: its own lock, no wait
    ob, eb = b.communicate(timeout=120)
    assert b.returncode == 0, ob + eb
    b_acq, _ = _stamps(eb)
    # while A still holds m1, a run on m1 with a short wait bound is refused by name (nothing pushed, nothing created)
    c = _run(tmp_path, repo, shas[1], m1, wait="1"); oc, ec = c.communicate(timeout=120)
    assert c.returncode == 1 and "REFUSED: another mirror_repo.sh run holds the mirror" in ec and "waited 1s" in ec, oc + ec
    assert not (tmp_path / "eval" / shas[1]).exists()
    oa, ea = a.communicate(timeout=120)
    assert a.returncode == 0, oa + ea
    a_acq, a_rel = _stamps(ea)
    assert b_acq < a_rel                                                     # B did not wait for A's lock
    assert "lock acquired" not in ec                                         # the refused run never entered the section
