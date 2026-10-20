"""Run the repository's full CPU pytest suite on Modal at an exact git SHA.

This replaces the retired Slurm CPU gate. The grading host only stages a commit; the
full suite runs in a CPU-only Modal function. The tree is a shallow standalone
git checkout pushed from the local private repo to ``/vol/repos/sisteer-<sha>``
(no git credentials enter Modal), copied to container-local disk, and verified
for exact HEAD + clean tracked files before pytest starts.

The image is resolved from this launch checkout's frozen ``pyproject.toml`` +
``uv.lock`` with the ``cpu`` extra and ``dev`` group. Their SHA-256 hashes are
baked into the invocation and compared with the staged SHA's files before the
suite runs; a dependency-file mismatch refuses instead of testing one commit
under another commit's environment. Pytest runs exactly ``tests/`` with
``PYTHONPATH=src`` and writes a JUnit report, full log, and parsed failure list
to ``/vol/cpu-suite/<sha>/``.

Modal 1.5 names an ephemeral app with ``modal run --name``/``-n``. The local
entrypoint requires ``cpu-suite-<shortsha>`` so an unnamed run cannot inherit a
stale identity.

Run from the repository worktree::

  ~/venvs/modal/bin/python -m modal run -n cpu-suite-<shortsha> \
      scripts/modal/cpu_suite.py --stage --sha <sha> --go [--timeout-s 6600]

``--timeout-s`` bounds pytest (default ``SUITE_TIMEOUT_S_DEFAULT`` = 140 min, under the function's 2.5 h): a
TIMEOUT summary carries ``status: TIMEOUT``, ``pytest_rc: -9`` and no counts — it is not a gate result, re-run
with a larger bound rather than reading it as a pass.

``--shards N`` (default 1 = one container, byte-identical to before) runs the suite as N containers, each one
shard of ``tests/test_*.py`` (``shard_files``: a size-balanced partition, so the union is the full suite), and
writes ONE merged ``summary.json`` in the single-run shape (``merge_shard_records``) plus per-shard
``shards/<i>-of-<N>/{pytest.log,junit.xml,summary.json}`` — of which only the FAILED shards' survive the finish
(PASSED SHARDS' ARTIFACTS DO NOT PERSIST, below). Use it when Modal preempts the long container: on
2026-09-16 one ~110-min container was preempted four times in a row, each restart from scratch; ten ~11-min
shards are each exposed for a tenth of the time and a preempted shard restarts alone. ``--timeout-s`` then
bounds EACH shard.

THE SHARDED FINISH IS SERVER-SIDE (2026-09-16). The merged ``summary.json`` and the own-tree prune of a sharded
gate run in ``finalize_shards``, a Modal function the entrypoint spawns with the shard call ids right after the
shards; the client only WAITS for it. Before, the client gathered the shards, wrote the merged summary and then
called the remote prune — so a client killed mid-gate (a romp restart, an OOM kill; ~35 gates a day, most
``--shards 10``) left an UNRECORDED tree and, under ``--detach``, a lingering zero-task app that the abandoned
clause read as live: the twosi-fits volume refilled to 89 % of its inodes 2.5 days after a full sweep. A shard
that raised at the function level (a container crash, a function timeout) is merged as ``SHARD_ERROR`` — the
gate is recorded as not-a-pass, and its tree is pruned like a FAIL's (re-stageable).

PASSED SHARDS' ARTIFACTS DO NOT PERSIST (2026-09-18). Once the merged ``summary.json`` is written and committed,
``finalize_shards`` removes ``shards/<i>-of-<N>/`` for every shard whose status in the merged record is exactly
``PASS``, and only those (``_prune_passed_shard_artifacts``). A FAIL / TIMEOUT / SHARD_ERROR shard's directory
stays untouched — its ``pytest.log`` and ``junit.xml`` are how a red gate is diagnosed, and losing them would make
the gate uninvestigable. The merged ``<results_root>/<sha>/summary.json`` ALWAYS stays: it is what every lane
reads for its verdict and what pull request bodies cite, and its content is the merge and nothing else (the
prune is reported in the RETURNED record's ``shard_artifacts_pruned`` = ``{removed, kept, missing,
shards_dir_removed}``, the way ``own_tree_pruned`` is). When every shard passed ``shards/`` itself goes too, by
``rmdir`` — only once it is empty, so another layout's leftovers at the same sha stay. Nothing outside
``<results_root>/<sha>/shards/`` and no other sha is ever touched, and a passing shard's directory that is
already gone is recorded as missing, not raised. Why: nothing pruned the results tree; ``/vol/cpu-suite`` held
635 per-commit gate directories on 2026-09-18 and gains eight to fifteen a day across the active lanes, each
ten-shard gate 31 files in 12 directories (the merged summary plus ten shards' log, JUnit and record), and the
twosi-fits volume reached 88.9 % of its 500,000-inode quota the day before — when inodes run out, creating a
file fails and EVERY lane's gate breaks at once. An all-pass ten-shard gate now leaves 1 file in 1 directory; a
gate with one failing shard leaves that shard's three files beside the summary. Consequence for a reader: a
passing shard's ``shard_records[i].results_dir`` names a directory that no longer exists — read the failure set
from the merged record and a failing shard's log from its kept directory. Directories recorded before this rule
are NOT swept retroactively by the finalizer (it touches only the sha it is finalizing).

A nonzero pytest result is reported in full and then returned as a nonzero
local command. The report characterises the Modal failures it actually sees;
it does not encode or forgive any historical environment-failure list.

Results and the failure list land at ``/vol/cpu-suite/<sha>/{summary.json,
junit.xml, pytest.log}``; read them back with ``modal volume get twosi-fits
/cpu-suite/<sha>/summary.json <local>``. Two properties of the container that
the grading host and the cluster did not have, both visible in the failure list
rather than hidden: the suite runs as root (a test whose fake user's ``/home``
must be un-creatable sees it created), and no normative spec files, API keys
or fitted object banks exist in the image (the tests that read them fail on
the missing file, exactly as they did on the cluster's CPU gate).

THE MODAL BASELINE (measured 2026-09-04 at 1876795, 5 failed / 5928 passed /
43 skipped / 5 xfailed / 9 xpassed in ~10 min) is NOT the Slurm gate's list
of five, and the next reader must not compare against that list:
  * four of the Slurm gate's five environmental failures recur for the same
    missing-environment reason — ``test_constitution_si`` and
    ``test_irrelevant_si`` (the normative spec files live outside the repo),
    ``test_grading_live`` (an API key file), ``test_stacked_defense`` (a
    fitted object at a ``/tmp`` seam);
  * the fifth Slurm failure, a hub-resolution loader test of an arm since
    removed from the repo, PASSES here;
  * one failure is NEW and root-specific: ``test_capeval_stage2_common::
    test_all_unset_defaults_are_todays_hardcoded_values`` stubs a fake user
    whose ``/home/<user>`` must not be creatable and asserts the resulting
    lock-file error text; root creates the directory, so the error never
    appears. It passes on the grading host (non-root) and passed on the cluster.
Any failure outside that set is a regression to look at, not environment.

STAGED TREES ARE PRUNED, NARROWLY (2026-09-11; unrecorded clause 2026-09-13).
``/vol/repos/sisteer-<40 hex>`` is SHARED with two other Modal apps that stage
full-sha trees there and read them at run time (lens_vectorize_gate,
petri_smoke) — so a staged tree is not this runner's to delete just because a
suite recorded a summary for its sha. (fst_multiturn, grokmt_served,
capctrl_served, misalign_vendor_shards, petri_agentloop_control and
tinycap_served stage under their own ``/vol/<lane>/repos``; the sweep never
looks there, and its ``SHA40`` match skips every abbreviated-sha tree.) The
volume reached 83 % of its inode quota (199 staged trees, 169 recorded suites)
because nothing pruned, and 88 % two days later with a third of the bulk in
trees no rule could touch. Four rules, all loud and re-stageable:

* ``--stage --go`` deletes the tree (and its tarball) that THIS invocation
  created, after the results are committed — never a tree that already
  existed when it staged (``unpack_tree`` reports ``pre_existed``; a
  ``--restage`` over an existing tree counts as pre-existing); ``--keep-tree``
  opts out. A peer app that reused that tree in the same window sees its own
  "run --stage" refusal, not a silent gap.
* ``--prune`` (app name ``cpu-suite-prune``, no sha) removes staged trees whose
  sha HAS a recorded ``summary.json`` AND whose tree is older than
  ``--prune-older-than-hours`` (default 48): a GPU leg runs within hours of its
  staging, so age is the guard against deleting a live peer's tree — 12 h is
  safe whenever ``modal app list`` shows no live lens-gate / petri-smoke app.
* the same sweep removes trees with NO ``summary.json`` (a peer's staging, or
  a suite that died before recording) once they are older than
  ``--prune-unrecorded-older-than-hours`` (default 168 = 7 days; ``0`` or a
  negative value disables the clause): nothing runs from a tree a week after
  staging it, and after the 2026-09-13 sweeps 34 of the 35 trees left were
  these. Recent trees, ``repos/cfgs`` and every non-``sisteer-<40 hex>`` entry
  are never touched. ``--prune-dry-run`` lists without deleting; the record
  names every unrecorded tree's age so the accumulation stays visible.
* the same sweep removes an ABANDONED tree — no ``summary.json`` AND no live
  ``cpu-suite-<sha7>`` app for its sha — once it is older than
  ``--prune-abandoned-older-than-hours`` (default 6; ``0`` or negative
  disables). Measured 2026-09-14: 66 of 67 young unrecorded trees had no
  results dir at all — stagings whose gate never ran (``--stage`` without
  ``--go``, a spawn refused at the 100-app workspace cap) or died with its
  client before the summary; they refilled the volume in 36 h. Liveness is
  read on the CLIENT (``live_app_descriptions``: ``modal app list --json``; the
  container holds no workspace credentials) and passed in. If the listing
  fails, or a lens-gate / petri-smoke app is live (their trees carry no sha in
  the app name), the clause is SUSPENDED and only the 7-day rule applies.
  ``--stage`` leaves ``sisteer-<sha>.staged.json`` beside the tree (who staged
  it, when) so an abandoned tree names its stager in the record; the stamp
  goes with the tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import modal

DEFAULT_APP_NAME = "sisteer-cpu-suite"
#: pytest's wall-clock bound (``--timeout-s``). The suite ran ~85 min at 13.6k tests on 2026-09-14 and the 90-min default
#: TIMED OUT the joint suite on e54553a9 (pytest killed, no JUnit, `counts: None`); at 15.9k tests (2026-09-16) the 110-min
#: default was reached again -- 6,582 s recorded on one tree, and the next tree was killed at 92 % after 6,700 s (the same
#: TIMEOUT shape). 140 min leaves ~10 min of the function's own 2.5 h limit for the tree copy and the persist step.
#: Re-size it from the last recorded ``elapsed_s`` when the suite grows again.
SUITE_TIMEOUT_S_DEFAULT = 140 * 60
app = modal.App(DEFAULT_APP_NAME)
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"
REPOS = f"{VOL_PATH}/repos"
RESULTS_ROOT = f"{VOL_PATH}/cpu-suite"
try:
    LOCAL_REPO = Path(__file__).resolve().parents[2]
except IndexError:  # Modal re-imports this module at the flat /root/cpu_suite.py
    LOCAL_REPO = Path("/repo")
UV_VERSION = "0.12.1"
DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
#: The Unix identity the suite runs under. Modal containers set no USER, and
#: the cluster-script tests read it even with every Slurm seam stubbed (six
#: test_stage2_run cases die on ``USER: unbound variable`` without one). It
#: must NOT be ``root``: two tests treat any USER of 3+ characters as the
#: username that must not leak into generated docs, and ``root`` is ordinary
#: prose in the very text they scan ("the cluster's per-user shared root"),
#: which produced two false failures on the 2026-09-04 run. A string that
#: occurs nowhere in the repository cannot false-positive that way.
SUITE_USER = "modalsuiteuser"


def volume_client_path(path: str) -> str:
    """Translate a mounted ``/vol/...`` path to the Volume client's ``/...``."""
    if path == VOL_PATH:
        return "/"
    prefix = f"{VOL_PATH}/"
    if not path.startswith(prefix):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# These hashes describe the exact dependency resolution used to build the
# image. Literals survive Modal's flat in-container module re-import; locally,
# fail before launch if the image inputs changed without refreshing the guard.
DEPENDENCY_HASHES = {
    "pyproject.toml": "8c7124d8e6205d74bdf5be80f1a6b3b52533543db94c37d014e3c0d4e88ae321",
    "uv.lock": "3db46ce4a23447b17c3e2fcf6d825306f35abcb3d214faa112d9ec4e9a9d99d3",
}
if modal.is_local():
    actual_hashes = {name: sha256_file(LOCAL_REPO / name) for name in DEPENDENCY_FILES}
    if actual_hashes != DEPENDENCY_HASHES:
        raise RuntimeError(f"CPU image dependency hash guard is stale: {actual_hashes}")

# The results registry is steered per run by ``suite_env`` (a throwaway spool), never by an image-wide
# switch: the off switch the image carried made every emitter a no-op for the tests under it too.


def uv_project_without_workspace(repo: Path) -> Path:
    """A build-time copy of the project's metadata for ``Image.uv_sync``. The root pyproject declares a uv workspace (the
    corpus generator under corpus_generator/ is its member, 2026-10-08), and Modal's ``uv_sync`` refuses any pyproject with a
    ``[tool.uv.workspace]`` table outright ("uv workspaces are not supported", ``modal._image._check_pyproject_toml``) before
    uv runs — staging the member's pyproject would not get past it, and ``--no-install-workspace`` is already what Modal
    passes. The image needs the HARNESS's dependencies only, so the copy drops that one table and keeps every other byte;
    ``uv.lock`` rides unchanged beside it (``uv sync --frozen --no-install-workspace`` reads the lock as it is). The
    dependency-hash guard above keys on the REAL files, not on this copy."""
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    stripped, n = re.subn(r"(?ms)^\[tool\.uv\.workspace\]\n.*?(?=^\[)", "", text)
    if n != 1 or "[tool.uv.workspace]" in stripped:
        raise RuntimeError("pyproject.toml: expected exactly one [tool.uv.workspace] table to drop for the image build")
    d = Path(tempfile.mkdtemp(prefix="cpu-suite-uv-project-"))
    (d / "pyproject.toml").write_text(stripped, encoding="utf-8")
    shutil.copyfile(repo / "uv.lock", d / "uv.lock")
    return d


UV_PROJECT_DIR = uv_project_without_workspace(LOCAL_REPO) if modal.is_local() else LOCAL_REPO

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .uv_sync(str(UV_PROJECT_DIR), groups=["dev"], extras=["cpu"], frozen=True,
             uv_version=UV_VERSION)
    .run_commands("git config --global --add safe.directory '*'")
    .env({"USER": SUITE_USER, "LOGNAME": SUITE_USER, "WANDB_MODE": "disabled",
          "TOKENIZERS_PARALLELISM": "false"})
)


def _git(repo: str | Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, timeout=120).stdout.strip()


def local_repo_root() -> Path:
    """The shared git object's repository root behind this worktree."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=str(LOCAL_REPO), capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent


def verify_checkout(repo: str | Path, sha: str) -> dict:
    """Exact HEAD and clean tracked files, the same tree facts as the GPU gate."""
    repo = str(repo)
    head = _git(repo, "rev-parse", "HEAD")
    clean = bool(head) and subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=repo,
        capture_output=True,
    ).returncode == 0
    ok = bool(head) and head == sha and clean
    why = None if ok else ("not a git checkout" if not head else
                           f"HEAD {head} != {sha}" if head != sha else
                           "modified tracked files")
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def shallow_clone_tarball(sha: str, dest_dir: Path) -> Path:
    """Create a shallow standalone checkout of ``sha`` and tar its ``.git``."""
    src = local_repo_root()
    if _git(src, "cat-file", "-t", sha) != "commit":
        raise SystemExit(f"[cpu-suite] {sha} is not a commit in {src}")
    work = Path(tempfile.mkdtemp(prefix="cpu-suite-stage-"))
    tree = work / f"sisteer-{sha}"
    try:
        tree.mkdir()
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        subprocess.run([
            "git", "-c", "uploadpack.allowAnySHA1InWant=true", "-C", str(tree),
            "fetch", "-q", "--depth", "1", str(src), sha,
        ], check=True)
        subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--detach", "FETCH_HEAD"],
                       check=True)
        checked = verify_checkout(tree, sha)
        if not checked["ok"]:
            raise SystemExit(f"[cpu-suite] shallow checkout failed verification: {checked}")
        tar_path = dest_dir / f"sisteer-{sha}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(tree, arcname=f"sisteer-{sha}")
        return tar_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4,
              memory=16384)
def unpack_tree(sha: str, force: bool = False, app_name: str = "") -> dict:
    """Install a client-uploaded checkout tarball under ``/vol/repos``, and
    leave a ``sisteer-<sha>.staged.json`` stamp beside it naming ``app_name``
    (the stager) — the sweep reports it for an abandoned tree."""
    VOL.reload()
    dst = Path(REPOS) / f"sisteer-{sha}"
    tar_path = Path(REPOS) / f"sisteer-{sha}.tar.gz"
    # whether a tree was already here before this call: the pruning rule's input
    # (a tree this invocation did not create is never this run's to delete)
    out = {"sha": sha, "tree": str(dst), "pre_existed": bool(dst.exists())}
    if dst.exists() and not force:
        checked = verify_checkout(dst, sha)
        out.update(existing=True, verify=checked)
        if tar_path.exists():
            tar_path.unlink()
        out["stamp"] = _write_staged_stamp(Path(REPOS), sha, app_name, created=False)
        VOL.commit()
        if not checked["ok"]:
            out["error"] = f"existing checkout does not verify: {checked['why']} (use --restage)"
        return out
    if not tar_path.is_file():
        return {**out, "error": f"no uploaded tarball at {tar_path}"}
    tmp = Path(REPOS) / f".cpu-unpack-{sha}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="fully_trusted")
        extracted = tmp / f"sisteer-{sha}"
        if not (extracted / ".git").exists():
            return {**out, "error": "uploaded checkout has no .git directory"}
        if dst.exists():
            shutil.rmtree(dst)
        extracted.rename(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    checked = verify_checkout(dst, sha)
    out.update(existing=False, verify=checked)
    if not checked["ok"]:
        out["error"] = f"unpacked checkout does not verify: {checked['why']}"
    out["stamp"] = _write_staged_stamp(Path(REPOS), sha, app_name, created=True)
    VOL.commit()
    return out


def dependency_hashes(tree: Path) -> dict:
    return {name: sha256_file(tree / name) for name in DEPENDENCY_FILES}


def pytest_argv(python: str, junit_path: str, files: list[str] | None = None) -> list[str]:
    """The suite command. Without ``files`` it is the FULL suite (``tests``: no test name or path is omitted);
    with ``files`` it is one shard — the files ``shard_files`` dealt this container, whose union over the shards
    is the same full suite."""
    target = list(files) if files else ["tests"]
    # --continue-on-collection-errors: a module that fails to import is ONE error node in the JUnit, and every other
    # module still runs. Without it pytest stops at collection ("Interrupted: 2 errors during collection") and a
    # shard silently reports 4 tests instead of ~1500 (shard 1 of cpu-suite-8968828, 2026-09-16) — a gate that
    # compares failing nodes would never see the tests that did not run.
    return [python, "-m", "pytest", *target, "--tb=short", "--continue-on-collection-errors", f"--junitxml={junit_path}"]


#: the partition rule, stamped into the merged record so a compare tool can refuse a mixed-layout comparison (a
#: sharded gate vs an unsharded baseline runs the tests in a different order, and an order-dependent test can flip)
SHARD_RULE = "tests/test_*.py sorted by name, shard i takes files[i::N] (round-robin by name; stable across edits)"


def shard_files(tree: Path, index: int, count: int) -> list[str]:
    """Shard ``index`` of ``count`` over every ``tests/test_*.py`` of ``tree``, as tree-relative POSIX paths.

    Deterministic and a partition — files sorted by NAME, shard i takes ``files[i::count]`` — so the union over
    ``range(count)`` is every file exactly once, and the layout changes only when a test file is added or removed,
    never when one grows: the first cut dealt by size (LPT) re-shuffled the whole map whenever any file changed
    size, and a same-layout comparison (the gate's rule for a sharded verdict) was impossible across two commits.
    Balance is statistical (~65 files per shard at N=10 over 650). Sharding exists because one ~110-minute
    container was preempted four times in a row on 2026-09-16 (each restart from scratch, so the gate never
    finished); ten short containers are each exposed for a tenth of the time and a preempted shard restarts alone."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} of {count}: need 0 <= index < count, count >= 1")
    files = sorted(q.relative_to(tree).as_posix() for q in (tree / "tests").glob("test_*.py"))
    return files[index::count]


def merge_shard_records(recs: list[dict], sha: str, run_name: str) -> dict:
    """ONE gate record from the shards' records, in the single-run record's shape so every reader of
    ``summary.json`` (``status``, ``pytest_rc``, ``counts``, ``failures[].node``) is unchanged.

    A shard that never ran pytest (NO_TREE / BAD_TREE / DEPENDENCY_MISMATCH) makes the whole record that status
    with ``pytest_rc`` None — a gate that could not run is not a pass. Otherwise counts are summed, the failure
    nodes are the union (sorted by node; shards are disjoint so nothing is double-counted), ``status`` is TIMEOUT
    if any shard timed out, else PASS iff every shard's ``pytest_rc`` is 0, and ``pytest_rc`` is 0 or the first
    nonzero shard rc."""
    recs = list(recs)
    refused = [r for r in recs if r.get("status") not in ("PASS", "FAIL", "TIMEOUT")]
    shard_records = [{k: r.get(k) for k in ("shard", "status", "pytest_rc", "counts", "elapsed_s", "results_dir", "error")}
                     for r in recs]
    layout = shard_layout(recs)
    if refused:
        return {"sha": sha, "app": run_name, "status": refused[0].get("status"), "pytest_rc": None,
                "error": refused[0].get("error") or f"shard refused: {refused[0].get('status')}",
                "shards": len(recs), "shard_layout": layout, "shard_records": shard_records}
    counts = None
    if recs and all(isinstance(r.get("counts"), dict) for r in recs):
        counts = {k: sum(int(r["counts"].get(k) or 0) for r in recs) for k in ("tests", "failures", "errors", "skipped")}
    failures = sorted((f for r in recs for f in (r.get("failures") or [])), key=lambda f: str(f.get("node")))
    rcs = [r.get("pytest_rc") for r in recs]
    rc = 0 if recs and all(x == 0 for x in rcs) else next((x for x in rcs if x != 0), None)
    timed_out = any(r.get("status") == "TIMEOUT" for r in recs)
    first = recs[0] if recs else {}
    return {"sha": sha, "app": run_name,
            "status": "TIMEOUT" if timed_out else ("PASS" if rc == 0 else "FAIL"), "pytest_rc": rc,
            "elapsed_s": max((float(r.get("elapsed_s") or 0) for r in recs), default=0.0),
            "command": [r.get("command") for r in recs], "tree": first.get("tree"),
            "dependency_hashes": first.get("dependency_hashes"), "versions": first.get("versions"),
            "counts": counts, "failures": failures,
            "shards": len(recs), "shard_layout": layout, "shard_records": shard_records}


def shard_layout(recs: list[dict]) -> dict:
    """``{count, rule, map_sha256}``: the layout a sharded record ran under. ``map_sha256`` hashes the sorted
    (file, shard index) pairs, so two records compare like-for-like only when the hashes match — a compare tool
    reads ``shards`` / ``shard_layout`` and refuses or flags a mixed-layout comparison instead of reading an
    order-dependent flip as a phantom node."""
    pairs = sorted((f, int((r.get("shard") or {}).get("index", -1))) for r in recs for f in ((r.get("shard") or {}).get("files") or []))
    return {"count": len(recs), "rule": SHARD_RULE,
            "map_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest()}


#: The results registry's spool variable (``sisteer.registry.emit.SPOOL_ENV``; this module cannot import the
#: package, it runs before the tree is staged) and the env knobs that would re-route or re-disable emission.
REGISTRY_SPOOL_ENV = "SISTEER_REGISTRY_SPOOL"
REGISTRY_SPOOL_PREFIX = "cpu-suite-registry-spool-"
#: Popped, never inherited: ``SISTEER_REGISTRY_DISABLED`` makes every emitter a no-op; ``REGISTRY_EMIT=0`` does
#: the same through the hooks; ``SISTEER_REGISTRY_PUSH`` names a script the writer would run on each run
#: directory, which could carry records OFF the throwaway spool.
REGISTRY_ENV_POPPED = ("SISTEER_REGISTRY_DISABLED", "REGISTRY_EMIT", "SISTEER_REGISTRY_PUSH")
#: Set to ``1`` in the suite subprocess env: the pytest run declares itself the sanctioned whole-suite runner.
#: ``tests/conftest.py`` refuses a whole-tree invocation on any other host (``tests/_full_suite_guard.py``; added
#: 2026-09-28, after two whole-suite runs on the grading host of 2.5 h and 3 h 23 min pushed that shared box to
#: load 22–34). Both gates build their pytest env here — the Modal container and the Slurm shards
#: (``gate_task.py``, which scrubs ``SLURM_*`` from this env on purpose) — so the allowance rides the runner, not
#: a substrate probe or the shard count (an unsharded run names ``tests`` whole; a low ``--shards`` would
#: otherwise cross the guard's file limit).
FULL_SUITE_ALLOW_ENV = "SISTEER_ALLOW_FULL_SUITE"


def suite_env(base: dict, tree: Path, spool_root: Path | str | None = None) -> dict:
    """The pytest subprocess env. Creates a FRESH throwaway registry spool under ``spool_root`` (default: the
    container's tmp) and names it in ``SISTEER_REGISTRY_SPOOL``; the caller removes it after the run.

    The gate's smoke fits are PRODUCERS to the results registry — 879 ``class_b`` object records
    from tiny smoke models reached the live spool on the volume — so the gate first set ``SISTEER_REGISTRY_DISABLED=1``.
    That switch is a global no-op for every emitter method, so 12 registry tests that emit into their own tmp
    spool failed inside the gate. A throwaway spool keeps both: emission stays ON (the tests see a working emitter),
    and ``hooks.resolve_spool`` / ``emit.spool_root`` return ``$SISTEER_REGISTRY_SPOOL`` BEFORE probing the Modal
    volume's ``/vol/registry`` (emit.py ``spool_root``: the env value returns first), so nothing reaches the volume."""
    env = dict(base)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(tree / "src")
    # Modal containers do not guarantee $USER, while cluster-script tests use
    # the standard Unix identity variable even with every Slurm seam stubbed;
    # set explicitly (never inherited) so a container-provided ``root`` cannot
    # reach the username-leak tests — see SUITE_USER.
    env["USER"] = SUITE_USER
    env["LOGNAME"] = SUITE_USER
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env[FULL_SUITE_ALLOW_ENV] = "1"          # this IS the suite; tests/conftest.py refuses a whole-tree run elsewhere
    for name in REGISTRY_ENV_POPPED:
        env.pop(name, None)
    env[REGISTRY_SPOOL_ENV] = tempfile.mkdtemp(prefix=REGISTRY_SPOOL_PREFIX,
                                               dir=str(spool_root) if spool_root is not None else None)
    env["HF_HOME"] = f"{VOL_PATH}/hf"
    env.pop("HF_HUB_OFFLINE", None)
    # Results-registry emission is OFF for the suite. The container mounts the twosi-fits volume at /vol with
    # registry/ provisioned, so the harness's one switch (on wherever a spool resolves) made every smoke fit
    # the tests run a producer: 2,727 tiny-model `object` records (855 run directories) had reached the live
    # log by 2026-09-18, stamped scripts/fit_class_b.py / host modal. tests/conftest.py sets the same
    # variable for every pytest session; this is the gate's own copy, so a conftest regression cannot leak.
    return env


def run_pytest(argv: list[str], tree: Path, env: dict, log: Path, timeout_s: int) -> tuple[int, bool]:
    """Run the suite subprocess (its own session, so a timeout kills the whole process group) with stdout+stderr
    into ``log``; returns ``(rc, timed_out)``. The throwaway registry spool ``env`` names (``suite_env``) is removed
    when the subprocess is gone, whatever its rc or a timeout: its records are the smoke fits' and belong nowhere."""
    timed_out = False
    try:
        with log.open("wb") as f:
            proc = subprocess.Popen(argv, cwd=tree, env=env, stdout=f,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
    finally:
        spool = env.get(REGISTRY_SPOOL_ENV)
        if spool:
            shutil.rmtree(spool, ignore_errors=True)
    return rc, timed_out


def parse_junit(path: Path) -> dict:
    """Parse pytest's JUnit XML into counts and a stable failure-node list."""
    if not path.is_file():
        return {"counts": None, "failures": [], "error": "pytest wrote no JUnit XML"}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {key: sum(int(float(s.attrib.get(key, 0) or 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    failures = []
    for tc in root.iter("testcase"):
        bad = list(tc.findall("failure")) + list(tc.findall("error"))
        if not bad:
            continue
        classname = tc.attrib.get("classname") or tc.attrib.get("file") or "unknown"
        node = f"{classname}::{tc.attrib.get('name', 'unknown')}"
        failures.append({
            "node": node,
            "kind": bad[0].tag,
            "message": (bad[0].attrib.get("message") or "").strip()[:500],
        })
    return {"counts": counts, "failures": failures}


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


SHA40 = re.compile(r"^sisteer-([0-9a-f]{40})$")
PRUNE_OLDER_THAN_HOURS_DEFAULT = 48.0
#: Trees with no recorded summary (a peer app's staging, or a suite that died
#: before recording) go only past this age: no GPU leg runs a week after its
#: staging. ``None`` (or a CLI value <= 0) restores "never".
PRUNE_UNRECORDED_OLDER_THAN_HOURS_DEFAULT = 168.0
#: An ABANDONED tree — unrecorded AND no live ``cpu-suite-<sha7>`` app for its
#: sha — goes past this age. 2026-09-14: 66 of 67 young unrecorded trees had no
#: results dir at all (gate never ran, or died with its client). 6 h since
#: 2026-09-16 (was 12): a live gate is covered by LIVENESS whatever its age, so
#: the horizon only guards the window in which a dead gate's operator re-runs
#: ``--go`` without ``--stage`` (a loud NO_TREE refusal, one re-stage); a shard's
#: function timeout is 2.5 h and a sharded gate finishes in ~30 min, so 6 h is
#: still more than twice the longest honest run. At 12 h the clause could not
#: fire inside a day of gates and the volume refilled between sweeps.
PRUNE_ABANDONED_OLDER_THAN_HOURS_DEFAULT = 6.0
#: A DETACHED app with NO running task that is itself older than the abandoned
#: horizon is not live (``partition_live_apps``): under ``--detach`` an app whose
#: client died lingers at 0 tasks indefinitely (six such cpu-suite apps on
#: 2026-09-13), and each one shielded its sha's tree from the clause. A live gate
#: always has >= 1 task (its shard, or the finalizer waiting on the shards), and
#: a young app is never excluded, so a restart gap cannot un-live a real gate.
#: Gate app names are mandatory ``cpu-suite-<sha7>`` (``main`` refuses others),
#: which is what lets liveness be keyed by sha.
GATE_APP_RE = re.compile(r"^cpu-suite-([0-9a-f]{7})$")
#: The other stagers of ``/vol/repos/sisteer-<40 hex>``. Their app names carry
#: no sha, so while one is live NO unrecorded tree can be called abandoned.
TENANT_APP_PREFIXES = ("lens-gate", "lens-vectorize", "petri-smoke", "petri_smoke")
#: ``/vol/repos/sisteer-<sha>.staged.json`` — who staged the tree and when;
#: OUTSIDE the checkout so ``verify_checkout``'s cleanness is untouched.
STAGED_STAMP_SUFFIX = ".staged.json"


def live_apps_raw(argv: list | None = None) -> list | None:
    """The workspace's app listing (``modal app list --json``: description, state, tasks, created_at),
    read on the CLIENT — the sweep container holds no workspace credentials. ``None`` when the listing
    fails for any reason: the sweep then treats liveness as undeterminable and the abandoned clause stays
    off, never "no apps are live"."""
    argv = list(argv or [sys.executable, "-m", "modal", "app", "list", "--json"])
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            return None
        apps = json.loads(proc.stdout)
        return list(apps) if isinstance(apps, list) else None
    except Exception:  # noqa: BLE001 — any failure is "undeterminable"
        return None


def _created_age_s(app_rec: dict, now: float) -> float | None:
    """Seconds since the listing's ``created_at`` (an ISO timestamp with offset), or None when unreadable."""
    raw = app_rec.get("created_at")
    if not raw:
        return None
    try:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(str(raw))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return float(now) - t.timestamp()
    except (TypeError, ValueError):
        return None


def partition_live_apps(apps: list, *, idle_older_than_s: float | None = None,
                        now: float | None = None) -> tuple:
    """``(live_descriptions, idle_excluded)`` from a raw listing: an app is live when its state starts with
    ``ephemeral``; a DETACHED one with ZERO running tasks whose ``created_at`` is older than
    ``idle_older_than_s`` is EXCLUDED as idle (returned in the second list as ``{description, tasks, age_h}``
    so the sweep's log names what it stopped counting). ``None`` = no exclusion (the 2026-09-14 rule). An
    unreadable ``tasks`` or ``created_at`` keeps the app live (fail closed). Pure."""
    now = time.time() if now is None else now
    live, idle = [], []
    for a in apps:
        state = str(a.get("state") or "")
        if not state.startswith("ephemeral"):
            continue
        desc = str(a.get("description") or "")
        if idle_older_than_s is not None and "detached" in state:
            try:
                n_tasks = int(str(a.get("tasks")))
            except (TypeError, ValueError):
                n_tasks = None
            age = _created_age_s(a, now)
            if n_tasks == 0 and age is not None and age > float(idle_older_than_s):
                idle.append({"description": desc, "tasks": 0, "age_h": round(age / 3600.0, 1)})
                continue
        live.append(desc)
    return sorted(live), sorted(idle, key=lambda d: d["description"])


def live_app_descriptions(argv: list | None = None, *, idle_older_than_s: float | None = None,
                          now: float | None = None) -> list | None:
    """Descriptions of the workspace's LIVE apps (:func:`live_apps_raw` -> :func:`partition_live_apps`), or
    ``None`` when the listing is unavailable. ``idle_older_than_s`` excludes aged zero-task detached apps."""
    apps = live_apps_raw(argv)
    if apps is None:
        return None
    return partition_live_apps(apps, idle_older_than_s=idle_older_than_s, now=now)[0]


def liveness_view(live_apps: list | None) -> dict:
    """What the abandoned clause may key on, from a live-app list: the sha7s
    of live gate apps, the live co-tenant apps, and whether the clause is
    determinable at all (it is not when the list is missing or a co-tenant is
    live). Pure."""
    if live_apps is None:
        return {"determinable": False, "why": "live-app list unavailable", "n_live_apps": None,
                "live_gate_shas": [], "tenant_live": []}
    tenants = sorted(d for d in live_apps if str(d).startswith(TENANT_APP_PREFIXES))
    gates = sorted({m.group(1) for d in live_apps for m in [GATE_APP_RE.match(str(d))] if m})
    if tenants:
        return {"determinable": False,
                "why": f"co-tenant app(s) live: {', '.join(tenants)} — their trees carry no sha in the app name",
                "n_live_apps": len(live_apps), "live_gate_shas": gates, "tenant_live": tenants}
    return {"determinable": True, "why": None, "n_live_apps": len(live_apps),
            "live_gate_shas": gates, "tenant_live": []}


def _stamp_path(repos_dir: Path, sha: str) -> Path:
    return Path(repos_dir) / f"sisteer-{sha}{STAGED_STAMP_SUFFIX}"


def _write_staged_stamp(repos_dir: Path, sha: str, app_name: str, *, created: bool) -> dict:
    """Record who staged ``sisteer-<sha>`` and when. A stamp already beside a
    PRE-EXISTING tree is the original stager's and is left alone."""
    p = _stamp_path(repos_dir, sha)
    if p.is_file() and not created:
        return _read_staged_stamp(repos_dir, sha) or {}
    stamp = {"sha": sha, "staged_by": app_name or None, "created": bool(created),
             "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p.write_text(json.dumps(stamp, indent=1))
    return stamp


def _read_staged_stamp(repos_dir: Path, sha: str) -> dict | None:
    p = _stamp_path(repos_dir, sha)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"error": "unreadable stamp"}


def prune_candidates(staged_mtimes: dict, recorded: set, *, now: float,
                     older_than_s: float, keep: set,
                     unrecorded_older_than_s: float | None = None,
                     abandoned_older_than_s: float | None = None,
                     live_gate_shas: set | None = None) -> list:
    """The shas whose staged tree may go, never one in ``keep``: recorded (a
    ``summary.json`` exists for the sha) AND older than ``older_than_s``; or
    UNRECORDED and older than ``unrecorded_older_than_s``; or ABANDONED —
    unrecorded, ``sha[:7]`` not among ``live_gate_shas``, and older than
    ``abandoned_older_than_s``. A ``None`` cutoff switches its clause off, and
    ``live_gate_shas=None`` (liveness undeterminable) switches the abandoned
    clause off whatever its cutoff. Pure, so the rule is testable without a
    volume."""
    out = []
    for sha, mtime in staged_mtimes.items():
        if sha in keep:
            continue
        age = now - float(mtime)
        if sha in recorded:
            if age >= float(older_than_s):
                out.append(sha)
            continue
        if unrecorded_older_than_s is not None and age >= float(unrecorded_older_than_s):
            out.append(sha)
            continue
        if (abandoned_older_than_s is not None and live_gate_shas is not None
                and sha[:7] not in live_gate_shas and age >= float(abandoned_older_than_s)):
            out.append(sha)
    return sorted(out)


def _prune_trees_in(repos_dir: Path, results_root: Path, *, older_than_hours: float,
                    dry_run: bool, keep: set, now: float | None = None,
                    unrecorded_older_than_hours: float | None = None,
                    abandoned_older_than_hours: float | None = None,
                    live_apps: list | None = None) -> dict:
    """Sweep ``repos_dir`` for ``sisteer-<40 hex>`` trees (nothing else is
    looked at) and remove the :func:`prune_candidates`; a pruned sha's stray
    ``sisteer-<sha>.tar.gz`` and ``.staged.json`` stamp go with it.
    ``unrecorded_older_than_hours`` (``None`` = never) is the cutoff for trees
    with no ``summary.json``; ``abandoned_older_than_hours`` the cutoff for
    unrecorded trees with no live gate app, keyed on ``live_apps`` through
    :func:`liveness_view` (``None`` or a live co-tenant suspends that clause).
    Returns the record, which names every unrecorded tree's age in hours, its
    stager where a stamp exists, and the liveness view the clause used."""
    now = time.time() if now is None else now
    staged: dict = {}
    for child in Path(repos_dir).iterdir():
        m = SHA40.match(child.name)
        if not m or not child.is_dir():
            continue
        staged[m.group(1)] = child.stat().st_mtime
    recorded = {sha for sha in staged if (Path(results_root) / sha / "summary.json").is_file()}
    unrecorded = set(staged) - recorded
    unrec_s = None if unrecorded_older_than_hours is None else float(unrecorded_older_than_hours) * 3600.0
    aband_s = None if abandoned_older_than_hours is None else float(abandoned_older_than_hours) * 3600.0
    view = liveness_view(live_apps)
    live_gate = set(view["live_gate_shas"]) if (aband_s is not None and view["determinable"]) else None
    victims = prune_candidates(staged, recorded, now=now, older_than_s=float(older_than_hours) * 3600.0,
                               keep=set(keep), unrecorded_older_than_s=unrec_s,
                               abandoned_older_than_s=aband_s, live_gate_shas=live_gate)
    removed = []
    for sha in victims:
        tree = Path(repos_dir) / f"sisteer-{sha}"
        tar = Path(repos_dir) / f"sisteer-{sha}.tar.gz"
        stamp = _stamp_path(repos_dir, sha)
        if not dry_run:
            shutil.rmtree(tree, ignore_errors=False)
            if tar.is_file():
                tar.unlink()
            if stamp.is_file():
                stamp.unlink()
        removed.append(sha)
    key = "would_remove" if dry_run else "removed"
    unrec_victims = set(victims) & unrecorded
    abandoned = sorted(s for s in unrec_victims if unrec_s is None or (now - float(staged[s])) < unrec_s)
    stagers = {sha: st.get("staged_by") for sha in sorted(unrecorded)
               if (st := _read_staged_stamp(repos_dir, sha)) is not None}
    return {"n_staged": len(staged), "n_recorded": len(recorded), "n_unrecorded": len(unrecorded),
            "older_than_hours": float(older_than_hours),
            "unrecorded_older_than_hours": None if unrec_s is None else float(unrecorded_older_than_hours),
            "abandoned_older_than_hours": None if aband_s is None else float(abandoned_older_than_hours),
            "liveness": view,
            "dry_run": bool(dry_run),
            "unrecorded_ages_h": {sha: round((now - float(staged[sha])) / 3600.0, 1) for sha in sorted(unrecorded)},
            "unrecorded_staged_by": stagers,
            "kept_unrecorded": sorted(unrecorded - set(victims)),
            "kept_recent_recorded": sorted((recorded - set(victims)) - set(keep)),
            "kept_explicitly": sorted(set(keep) & set(staged)),
            key: removed,
            f"{key}_unrecorded": sorted(unrec_victims),
            f"{key}_abandoned": abandoned}


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4,
              memory=8192, retries=0)
def prune_staged_trees(older_than_hours: float = PRUNE_OLDER_THAN_HOURS_DEFAULT,
                       dry_run: bool = True, keep: list | None = None,
                       unrecorded_older_than_hours: float | None = PRUNE_UNRECORDED_OLDER_THAN_HOURS_DEFAULT,
                       abandoned_older_than_hours: float | None = PRUNE_ABANDONED_OLDER_THAN_HOURS_DEFAULT,
                       live_apps: list | None = None) -> dict:
    """The ``--prune`` sweep on the volume (module docstring: recorded AND aged,
    unrecorded AND a week old, or abandoned AND half a day old — nothing else).
    ``live_apps`` is the client's live-app list; ``None`` keeps the abandoned
    clause off."""
    VOL.reload()
    rec = _prune_trees_in(Path(REPOS), Path(RESULTS_ROOT), older_than_hours=older_than_hours,
                          dry_run=dry_run, keep=set(keep or ()),
                          unrecorded_older_than_hours=unrecorded_older_than_hours,
                          abandoned_older_than_hours=abandoned_older_than_hours, live_apps=live_apps)
    if not dry_run:
        VOL.commit()
    return rec


def _prune_own_tree(sha: str) -> dict:
    """Delete the tree (and tarball, and stamp) THIS invocation staged, once its results are on the volume."""
    tree = Path(REPOS) / f"sisteer-{sha}"
    tar = Path(REPOS) / f"sisteer-{sha}.tar.gz"
    stamp = _stamp_path(Path(REPOS), sha)
    out = {"tree": str(tree), "removed_tree": False, "removed_tarball": False, "removed_stamp": False}
    if tree.is_dir():
        shutil.rmtree(tree, ignore_errors=False)
        out["removed_tree"] = True
    if tar.is_file():
        tar.unlink()
        out["removed_tarball"] = True
    if stamp.is_file():
        stamp.unlink()
        out["removed_stamp"] = True
    return out


#: The finalizer outlives the slowest shard: the shards' function timeout is 2.5 h.
FINALIZE_TIMEOUT_S = int(3 * 3600)
#: The merged status of a shard whose CALL failed (raised, or timed out at the function level): not a pass.
SHARD_ERROR_STATUS = "SHARD_ERROR"


def shard_error_record(sha: str, run_name: str, index: int, count: int, call_id: str, exc: BaseException) -> dict:
    """The record a shard whose ``FunctionCall.get()`` raised is merged under (module docstring)."""
    return {"sha": sha, "app": run_name, "status": SHARD_ERROR_STATUS, "pytest_rc": None,
            "shard": {"index": int(index), "count": int(count)}, "call_id": str(call_id),
            "error": f"{type(exc).__name__}: {exc}"[:500]}


#: The ONE per-shard status whose artifacts the finalizer removes (module docstring): exactly this, nothing else.
SHARD_ARTIFACTS_PRUNED_STATUS = "PASS"


def _prune_passed_shard_artifacts(results_root: Path, sha: str, rec: dict) -> dict:
    """Remove ``<results_root>/<sha>/shards/<i>-of-<N>/`` for every shard whose status in the merged record ``rec``
    is exactly ``PASS``, and only those (module docstring: PASSED SHARDS' ARTIFACTS DO NOT PERSIST). Every other
    shard's directory — FAIL, TIMEOUT, SHARD_ERROR, a refusal — stays untouched: its ``pytest.log`` and ``junit.xml``
    are how a red gate is diagnosed. ``shards/`` itself goes only when every shard passed AND it is then empty
    (``rmdir``, so a directory the record does not name — another layout's leftovers at the same sha — stays).
    Reaches nothing outside ``<results_root>/<sha>/shards/`` and no other sha; a passing shard's directory that
    is already gone is ``missing``, not an error. Returns ``{removed, kept, missing, shards_dir_removed}`` (names
    are ``<i>-of-<N>``; ``kept`` maps a name to the status that kept it), plus ``errors`` when an rmtree failed."""
    base = Path(results_root) / sha / "shards"
    out: dict = {"removed": [], "kept": {}, "missing": [], "shards_dir_removed": False}
    shard_recs = [r for r in (rec.get("shard_records") or []) if isinstance(r, dict)]
    for sr in shard_recs:
        shard = sr.get("shard") or {}
        if not isinstance(shard, dict) or shard.get("index") is None or shard.get("count") is None:
            continue
        idx, cnt = int(shard["index"]), int(shard["count"])
        if cnt < 2:                       # an unsharded record writes beside summary.json, never under shards/
            continue
        name = f"{idx}-of-{cnt}"
        if sr.get("status") != SHARD_ARTIFACTS_PRUNED_STATUS:
            out["kept"][name] = sr.get("status")
            continue
        d = base / name
        if not d.is_dir():
            out["missing"].append(name)
            continue
        try:
            shutil.rmtree(d, ignore_errors=False)
            out["removed"].append(name)
        except OSError as e:
            out.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"[:300]
    all_passed = bool(shard_recs) and all(sr.get("status") == SHARD_ARTIFACTS_PRUNED_STATUS for sr in shard_recs)
    if all_passed and base.is_dir():
        try:
            base.rmdir()
            out["shards_dir_removed"] = True
        except OSError:                   # not empty: a directory the record does not name stays, and so does shards/
            pass
    return out


def _finalize_shards_in(sha: str, run_name: str, call_ids: list, prune_own: bool, *, get_call, repos_dir: Path,
                        results_root: Path, vol=None) -> dict:
    """The finalizer's body over injectable seams (``get_call(call_id) -> record``, the volume dirs, the volume
    handle), so it is testable without Modal. Gathers every shard's record (a raising call becomes a
    ``SHARD_ERROR`` record), merges them (``merge_shard_records``), writes ``<results_root>/<sha>/summary.json``
    atomically, commits, and — the record being on the volume — removes the PASSED shards' artifact directories
    (``_prune_passed_shard_artifacts``; every other shard's stays, the merged summary always stays) and prunes the
    tree THIS gate created."""
    recs = []
    n = len(call_ids)
    for i, cid in enumerate(call_ids):
        try:
            r = get_call(cid)
        except Exception as e:  # noqa: BLE001 — a shard that raised or timed out at the function level
            r = shard_error_record(sha, run_name, i, n, cid, e)
        recs.append(r)
        print(f"[cpu-suite] shard {i}/{n}: status={r.get('status')} rc={r.get('pytest_rc')} "
              f"counts={r.get('counts')} failures={len(r.get('failures') or [])} elapsed_s={r.get('elapsed_s')}", flush=True)
    rec = merge_shard_records(recs, sha, run_name)
    if vol is not None:
        vol.reload()
    dest = Path(results_root) / sha
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "summary.json.tmp"
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, dest / "summary.json")
    if vol is not None:
        vol.commit()
    rec["results_dir"] = str(dest)
    rec["finalized_by"] = "finalize_shards"
    # the merged record is on the volume: the PASSED shards' log / JUnit / record directories go, and only those
    # (module docstring) — a failing shard's directory is how the red gate gets diagnosed, and summary.json stays
    pruned = _prune_passed_shard_artifacts(Path(results_root), sha, rec)
    rec["shard_artifacts_pruned"] = pruned
    if pruned.get("errors"):
        print(f"[cpu-suite] shard-artifact prune: {len(pruned['errors'])} directory(ies) could NOT be removed "
              f"(the summary is recorded; they stay): {json.dumps(pruned['errors'])}", flush=True)
    if vol is not None and (pruned["removed"] or pruned["shards_dir_removed"]):
        vol.commit()
    if prune_own:
        # the record is on the volume; the tree THIS gate created goes (module docstring) — a FAIL / SHARD_ERROR
        # gate's tree too: re-stageable, and nothing reads it
        tree = Path(repos_dir) / f"sisteer-{sha}"
        tar = Path(repos_dir) / f"sisteer-{sha}.tar.gz"
        stamp = _stamp_path(Path(repos_dir), sha)
        out = {"tree": str(tree), "removed_tree": False, "removed_tarball": False, "removed_stamp": False}
        if tree.is_dir():
            shutil.rmtree(tree, ignore_errors=False)
            out["removed_tree"] = True
        if tar.is_file():
            tar.unlink()
            out["removed_tarball"] = True
        if stamp.is_file():
            stamp.unlink()
            out["removed_stamp"] = True
        rec["own_tree_pruned"] = out
        if vol is not None:
            vol.commit()
    print("[cpu-suite] SUMMARY", json.dumps(rec, indent=1, default=str), flush=True)
    return rec


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=FINALIZE_TIMEOUT_S, cpu=1, memory=2048, retries=0)
def finalize_shards(sha: str, run_name: str, call_ids: list, prune_own: bool) -> dict:
    """SERVER-SIDE finish of a sharded gate (module docstring): waits on the shard calls by id, merges, records
    the summary, removes the passed shards' artifact directories, prunes the own tree. Spawned by the entrypoint
    right after the shards, so a client that dies mid-gate cannot leave the tree unrecorded."""
    return _finalize_shards_in(sha, run_name, list(call_ids), bool(prune_own),
                               get_call=lambda cid: modal.FunctionCall.from_id(cid).get(),
                               repos_dir=Path(REPOS), results_root=Path(RESULTS_ROOT), vol=VOL)


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=int(2.5 * 3600), cpu=16,
              memory=65536, retries=0, secrets=[modal.Secret.from_name("hf-token")])
def run_cpu_suite(sha: str, run_name: str, timeout_s: int = SUITE_TIMEOUT_S_DEFAULT,
                  prune_own_tree: bool = False, shard_index: int = 0, shard_count: int = 1) -> dict:
    """Run pytest at ``sha`` — the full suite, or shard ``shard_index`` of ``shard_count`` (``shard_files``) — and
    persist its log, JUnit, and failure list (a shard under ``shards/<i>-of-<n>/``; the merged record is the
    entrypoint's)."""
    started = time.time()
    VOL.reload()
    src = Path(REPOS) / f"sisteer-{sha}"
    if not (src / ".git").exists():
        return {"sha": sha, "status": "NO_TREE",
                "error": f"{src} is not a staged checkout — run --stage (a suite that staged its own tree "
                         "prunes it once its summary is recorded; trees are re-stageable from the sha)"}
    tree = Path("/tmp/cpu-suite") / f"sisteer-{sha}"
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(src, tree, symlinks=True)
    checked = verify_checkout(tree, sha)
    if not checked["ok"]:
        return {"sha": sha, "status": "BAD_TREE", "tree": checked}
    got_hashes = dependency_hashes(tree)
    if got_hashes != DEPENDENCY_HASHES:
        return {"sha": sha, "status": "DEPENDENCY_MISMATCH",
                "expected": DEPENDENCY_HASHES, "staged": got_hashes,
                "error": "staged pyproject.toml/uv.lock differ from the image build inputs"}

    sharded = int(shard_count) > 1
    shard = {"index": int(shard_index), "count": int(shard_count)} if sharded else None
    sub = f"shards/{int(shard_index)}-of-{int(shard_count)}" if sharded else ""
    work = Path("/tmp/cpu-suite-results") / sha / sub
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    log = work / "pytest.log"
    junit = work / "junit.xml"
    files = shard_files(tree, int(shard_index), int(shard_count)) if sharded else None
    if sharded:
        shard["n_files"] = len(files)
        shard["files"] = list(files)          # the merged record hashes the file -> shard map from these
    if sharded and not files:
        # more shards than test files: this one has nothing to run, and running pytest with no paths would be
        # the FULL suite — record an empty pass instead
        rec = {"sha": sha, "app": run_name, "status": "PASS", "pytest_rc": 0, "elapsed_s": round(time.time() - started, 1),
               "command": [], "tree": checked, "dependency_hashes": got_hashes, "versions": {},
               "counts": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}, "failures": [], "shard": shard,
               "note": "empty shard (more shards than test files)"}
        (work / "summary.json").write_text(json.dumps(rec, indent=2, default=str))
        dest = Path(RESULTS_ROOT) / sha / sub
        _atomic_copy(work / "summary.json", dest / "summary.json")
        VOL.commit()
        rec["results_dir"] = str(dest)
        return rec
    argv = pytest_argv(sys.executable, str(junit), files)
    env = suite_env(os.environ, tree)
    shown = ' '.join(argv) if not sharded else ' '.join(argv[:3]) + f" <{len(files)} files of shard {shard_index}/{shard_count}> " + ' '.join(argv[3 + len(files):])
    print(f"[cpu-suite] app={run_name} sha={sha} command={shown}", flush=True)
    rc, timed_out = run_pytest(argv, tree, env, log, int(timeout_s))

    parsed = parse_junit(junit)
    try:
        import importlib.metadata as md
        versions = {name: md.version(name) for name in
                    ("pytest", "torch", "transformers", "datasets", "numpy")}
    except Exception as e:  # noqa: BLE001
        versions = {"error": f"{type(e).__name__}: {e}"}
    rec = {
        "sha": sha, "app": run_name, "status": "TIMEOUT" if timed_out else
        ("PASS" if rc == 0 else "FAIL"), "pytest_rc": rc,
        "elapsed_s": round(time.time() - started, 1), "command": argv,
        "tree": checked, "dependency_hashes": got_hashes, "versions": versions,
        **parsed, **({"shard": shard} if sharded else {}),
    }
    (work / "summary.json").write_text(json.dumps(rec, indent=2, default=str))
    dest = Path(RESULTS_ROOT) / sha / sub
    for name in ("pytest.log", "junit.xml", "summary.json"):
        p = work / name
        if p.is_file():
            _atomic_copy(p, dest / name)
    VOL.commit()
    rec["results_dir"] = str(dest)
    if prune_own_tree:
        # the record is on the volume; the tree THIS invocation created goes (module docstring)
        rec["own_tree_pruned"] = _prune_own_tree(sha)
        VOL.commit()
    print("[cpu-suite] SUMMARY", json.dumps(rec, indent=1, default=str), flush=True)
    if log.is_file():
        print("[cpu-suite] LOG TAIL\n" + log.read_text(errors="replace")[-12000:], flush=True)
    return rec


@app.local_entrypoint()
def main(sha: str = "", stage: bool = False, go: bool = False,
         restage: bool = False, timeout_s: int = SUITE_TIMEOUT_S_DEFAULT, keep_tree: bool = False,
         prune: bool = False, prune_dry_run: bool = False,
         prune_older_than_hours: float = PRUNE_OLDER_THAN_HOURS_DEFAULT,
         prune_unrecorded_older_than_hours: float = PRUNE_UNRECORDED_OLDER_THAN_HOURS_DEFAULT,
         prune_abandoned_older_than_hours: float = PRUNE_ABANDONED_OLDER_THAN_HOURS_DEFAULT,
         shards: int = 1):
    if prune or prune_dry_run:
        run_name = app.description or DEFAULT_APP_NAME
        if run_name != "cpu-suite-prune":
            raise SystemExit("[cpu-suite] run the sweep with `modal run -n cpu-suite-prune ... --prune` "
                             f"(or --prune-dry-run); got app name {run_name!r}")
        unrec = float(prune_unrecorded_older_than_hours)
        unrec = unrec if unrec > 0 else None                  # <= 0 disables the unrecorded clause
        aband = float(prune_abandoned_older_than_hours)
        aband = aband if aband > 0 else None                  # <= 0 disables the abandoned clause
        # liveness is the client's to read: the sweep container has no workspace credentials. An aged zero-task
        # DETACHED app (a dead client's leftover) is not live (partition_live_apps) — named in the log
        live, idle = None, []
        if aband is not None:
            raw = live_apps_raw()
            if raw is not None:
                live, idle = partition_live_apps(raw, idle_older_than_s=aband * 3600.0)
        print(f"[cpu-suite] liveness for the abandoned clause: {json.dumps(liveness_view(live))}", flush=True)
        if idle:
            print(f"[cpu-suite] idle detached apps NOT counted live (0 tasks, older than {aband:g} h): {json.dumps(idle)}", flush=True)
        rec = prune_staged_trees.remote(float(prune_older_than_hours), bool(prune_dry_run or not prune), [],
                                        unrecorded_older_than_hours=unrec,
                                        abandoned_older_than_hours=aband, live_apps=live)
        print(json.dumps(rec, indent=1))
        return
    if not sha:
        sha = _git(LOCAL_REPO, "rev-parse", "HEAD")
        print(f"[cpu-suite] --sha not given: using local HEAD {sha}; local edits do not travel")
    if len(sha) != 40:
        raise SystemExit("[cpu-suite] pass a full 40-hex --sha")
    expected_name = f"cpu-suite-{sha[:7]}"
    run_name = app.description or DEFAULT_APP_NAME
    if run_name != expected_name:
        raise SystemExit(f"[cpu-suite] run with `modal run -n {expected_name} ...`; got app name "
                         f"{run_name!r}. The explicit descriptive app name is mandatory")
    print(f"[cpu-suite] app={run_name} app_id={app.app_id} sha={sha}", flush=True)
    if stage:
        with tempfile.TemporaryDirectory(prefix="cpu-suite-tar-") as td:
            tar = shallow_clone_tarball(sha, Path(td))
            print(f"[cpu-suite] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {REPOS}/")
            with VOL.batch_upload(force=True) as batch:
                batch.put_file(str(tar), f"{volume_client_path(REPOS)}/{tar.name}")
            staged = unpack_tree.remote(sha, force=restage, app_name=run_name)
            print(json.dumps(staged, indent=1))
            if staged.get("error"):
                raise SystemExit(f"[cpu-suite] staging failed: {staged['error']}")
            created_here = staged.get("pre_existed") is False
        if not go:
            return
    else:
        created_here = False
    if not go:
        raise SystemExit("[cpu-suite] pass --stage and/or --go")
    prune_own = bool(stage and created_here and not keep_tree)
    if stage and not created_here and not keep_tree:
        print(f"[cpu-suite] {REPOS}/sisteer-{sha} already existed before this --stage: it is a peer's "
              "(or a prior run's) tree and is NOT pruned by this run", flush=True)
    shards = int(shards)
    if shards < 1:
        raise SystemExit(f"[cpu-suite] --shards must be >= 1, got {shards}")
    if shards == 1:
        call = run_cpu_suite.spawn(sha, run_name, int(timeout_s), prune_own)
        print(f"[cpu-suite] spawned call {call.object_id}", flush=True)
        rec = call.get()
    else:
        # N containers, each one shard of tests/test_*.py (shard_files: a partition); the shards never prune the
        # tree they share. The merged record and the own-tree prune run SERVER-SIDE (finalize_shards), spawned
        # NOW with the shard call ids — a client killed mid-gate cannot leave an unrecorded tree; it only waits
        calls = [run_cpu_suite.spawn(sha, run_name, int(timeout_s), False, i, shards) for i in range(shards)]
        print(f"[cpu-suite] spawned {shards} shard calls: {' '.join(c.object_id for c in calls)}", flush=True)
        fin = finalize_shards.spawn(sha, run_name, [c.object_id for c in calls], prune_own)
        print(f"[cpu-suite] spawned finalizer {fin.object_id} (server-side merge + own-tree prune)", flush=True)
        rec = fin.get()
    print(json.dumps(rec, indent=1, default=str))
    if rec.get("pytest_rc") != 0:
        failures = [f.get("node") for f in rec.get("failures") or []]
        raise SystemExit(f"[cpu-suite] {rec.get('status')}: pytest rc={rec.get('pytest_rc')}; "
                         f"failures={failures}")


if __name__ == "__main__":
    raise SystemExit("run with `python -m modal run -n cpu-suite-<shortsha> scripts/modal/cpu_suite.py ...`")
