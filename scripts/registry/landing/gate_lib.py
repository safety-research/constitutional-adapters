#!/usr/bin/env python3
"""Shared helpers for the landing gate on Slurm (stdlib only; lives in scripts/registry/landing/, shipped to the
cluster by gate_slurm.sh as $SHARED/gates/bin/gate_lib.py and run from the exported tree's own copy by battery.sbatch).

The partition, the pytest command, the suite environment, the JUnit parser and the merged-record shape are THIS module's:
``shard_files``, ``pytest_argv``, ``suite_env``, ``run_pytest``, ``parse_junit``,
``verify_checkout``, ``dependency_hashes``, ``merge_shard_records``, ``shard_layout`` and ``SHARD_RULE``. Until C9 they were loaded
from the tree's Modal-era suite module under a ``unittest.mock.MagicMock`` standing in for the ``modal`` package; the functions
were copied here as they stood (``suite_env`` no longer points ``HF_HOME`` at a Modal volume path: the caller, gate_task.py,
sets the cluster's), the loader and the stub are gone, and no chain file binds the name ``modal`` at all. A Slurm summary.json keeps
the merged record's shape (``sha, status, pytest_rc, elapsed_s, counts, failures[{node,kind,message}], shards,
shard_layout{count,rule,map_sha256}, shard_records``) plus a ``substrate: slurm`` block, so every reader of a summary is unchanged.

Sub-commands:
  shard-files TREE INDEX COUNT                 one shard's tree-relative test files, one per line
  finalize --sha S --tree T --results-root R --count N --run-name NAME [--array-job-id ID]
                                               merge R/S/shards/<i>-of-<N>/summary.json -> R/S/summary.json
                                               (a shard with no record is merged as TIMEOUT: not a gate result)
  amend --sha S --results-root R --base-summary P --rerun-dir D --files F [F...] --job-id J --reason TEXT
                                               R/S/summary.json = P with the failure list REPLACED for the nodes of the
                                               named test files by the targeted re-run's (D/summary.json, gate_task
                                               --files); every changed node is recorded under ``amendments``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# ---- the suite's facts: the dependency files the venv is keyed on, the identity the suite runs under ----------------------------
DEPENDENCY_FILES = ("pyproject.toml", "uv.lock")
#: The Unix identity the suite runs under. The cluster-script tests read USER even with every Slurm seam stubbed (six
#: test_stage2_run cases die on ``USER: unbound variable`` without one). It must NOT be ``root``: two tests treat any USER of
#: 3+ characters as the username that must not leak into generated docs, and ``root`` is ordinary prose in the very text they
#: scan ("the cluster's per-user shared root"), which produced two false failures on the 2026-09-04 run. A string that occurs
#: nowhere in the repository cannot false-positive that way. (The spelling predates the Slurm gate and is kept so a summary of
#: the Slurm era compares like-for-like with the recorded ones.)
SUITE_USER = "modalsuiteuser"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo: str | Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=120).stdout.strip()


def verify_checkout(repo: str | Path, sha: str) -> dict:
    """Exact HEAD and clean tracked files, the same tree facts as the GPU gate."""
    repo = str(repo)
    head = _git(repo, "rev-parse", "HEAD")
    clean = bool(head) and subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=repo, capture_output=True).returncode == 0
    ok = bool(head) and head == sha and clean
    why = None if ok else ("not a git checkout" if not head else f"HEAD {head} != {sha}" if head != sha else "modified tracked files")
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def dependency_hashes(tree: Path) -> dict:
    return {name: sha256_file(Path(tree) / name) for name in DEPENDENCY_FILES}


def pytest_argv(python: str, junit_path: str, files: list[str] | None = None) -> list[str]:
    """The suite command. Without ``files`` it is the FULL suite (``tests``: no test name or path is omitted); with ``files`` it
    is one shard -- the files ``shard_files`` dealt this task, whose union over the shards is the same full suite."""
    target = list(files) if files else ["tests"]
    # --continue-on-collection-errors: a module that fails to import is ONE error node in the JUnit, and every other module still
    # runs. Without it pytest stops at collection ("Interrupted: 2 errors during collection") and a shard silently reports 4 tests
    # instead of ~1500 (shard 1 of the 2026-09-16 suite) -- a gate that compares failing nodes would never see the tests that did
    # not run.
    return [python, "-m", "pytest", *target, "--tb=short", "--continue-on-collection-errors", f"--junitxml={junit_path}"]


#: the partition rule, stamped into the merged record so a compare tool can refuse a mixed-layout comparison (a sharded gate vs an
#: unsharded baseline runs the tests in a different order, and an order-dependent test can flip)
SHARD_RULE = "tests/test_*.py sorted by name, shard i takes files[i::N] (round-robin by name; stable across edits)"


def shard_files(tree: Path, index: int, count: int) -> list[str]:
    """Shard ``index`` of ``count`` over every ``tests/test_*.py`` of ``tree``, as tree-relative POSIX paths.

    Deterministic and a partition -- files sorted by NAME, shard i takes ``files[i::count]`` -- so the union over ``range(count)``
    is every file exactly once, and the layout changes only when a test file is added or removed, never when one grows: the first
    cut dealt by size (LPT) re-shuffled the whole map whenever any file changed size, and a same-layout comparison (the gate's rule
    for a sharded verdict) was impossible across two commits. Balance is statistical (~65 files per shard at N=10 over 650)."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} of {count}: need 0 <= index < count, count >= 1")
    tree = Path(tree)
    files = sorted(q.relative_to(tree).as_posix() for q in (tree / "tests").glob("test_*.py"))
    return files[index::count]


def merge_shard_records(recs: list[dict], sha: str, run_name: str) -> dict:
    """ONE gate record from the shards' records, in the single-run record's shape so every reader of ``summary.json``
    (``status``, ``pytest_rc``, ``counts``, ``failures[].node``) is unchanged.

    A shard that never ran pytest (NO_TREE / BAD_TREE / DEPENDENCY_MISMATCH) makes the whole record that status with ``pytest_rc``
    None -- a gate that could not run is not a pass. Otherwise counts are summed, the failure nodes are the union (sorted by node;
    shards are disjoint so nothing is double-counted), ``status`` is TIMEOUT if any shard timed out, else PASS iff every shard's
    ``pytest_rc`` is 0, and ``pytest_rc`` is 0 or the first nonzero shard rc."""
    recs = list(recs)
    refused = [r for r in recs if r.get("status") not in ("PASS", "FAIL", "TIMEOUT")]
    shard_records = [{k: r.get(k) for k in ("shard", "status", "pytest_rc", "counts", "elapsed_s", "results_dir", "error")} for r in recs]
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
    """``{count, rule, map_sha256}``: the layout a sharded record ran under. ``map_sha256`` hashes the sorted (file, shard index)
    pairs, so two records compare like-for-like only when the hashes match -- a compare tool reads ``shards`` / ``shard_layout``
    and refuses or flags a mixed-layout comparison instead of reading an order-dependent flip as a phantom node."""
    pairs = sorted((f, int((r.get("shard") or {}).get("index", -1))) for r in recs for f in ((r.get("shard") or {}).get("files") or []))
    return {"count": len(recs), "rule": SHARD_RULE, "map_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest()}


#: The results registry's spool variable (``sisteer.registry.emit.SPOOL_ENV``; this module cannot import the package, it runs
#: before the tree is staged) and the env knobs that would re-route or re-disable emission.
REGISTRY_SPOOL_ENV = "SISTEER_REGISTRY_SPOOL"
REGISTRY_SPOOL_PREFIX = "cpu-suite-registry-spool-"
#: Popped, never inherited: ``SISTEER_REGISTRY_DISABLED`` makes every emitter a no-op (the failure the throwaway spool undoes); ``REGISTRY_EMIT=0``
#: does the same through the hooks; ``SISTEER_REGISTRY_PUSH`` names a script the writer would run on each run directory, which could
#: carry records OFF the throwaway spool.
REGISTRY_ENV_POPPED = ("SISTEER_REGISTRY_DISABLED", "REGISTRY_EMIT", "SISTEER_REGISTRY_PUSH")
#: Set to ``1`` in the suite subprocess env: the pytest run declares itself the sanctioned whole-suite runner. ``tests/conftest.py``
#: refuses a whole-tree invocation on any other host (``tests/_full_suite_guard.py``, after two whole-suite runs of 2.5 h and
#: 3 h 23 min pushed the shared grading host to load 22-34). The Slurm shards build their pytest env here
#: (``gate_task.py``, which scrubs ``SLURM_*`` from this env on purpose), so the allowance rides the runner, not a substrate probe or
#: the shard count (an unsharded run names ``tests`` whole; a low shard count would otherwise cross the guard's file limit).
FULL_SUITE_ALLOW_ENV = "SISTEER_ALLOW_FULL_SUITE"


def suite_env(base: dict, tree: Path, spool_root: Path | str | None = None) -> dict:
    """The pytest subprocess env. Creates a FRESH throwaway registry spool under ``spool_root`` (default: the host's tmp) and names
    it in ``SISTEER_REGISTRY_SPOOL``; the caller removes it after the run (``run_pytest`` does).

    The gate's smoke fits are PRODUCERS to the results registry -- 879 ``class_b`` object records from tiny smoke models
    reached a live spool -- so the gate first set ``SISTEER_REGISTRY_DISABLED=1``. That switch is a global no-op for every
    emitter method, so 12 registry tests that emit into their own tmp spool failed inside the gate. A throwaway spool
    keeps both: emission stays ON (the tests see a working emitter), and ``hooks.resolve_spool`` / ``emit.spool_root`` return
    ``$SISTEER_REGISTRY_SPOOL`` first, so nothing reaches a live spool. ``HF_HOME`` is the CALLER's to set (gate_task.py sets the
    cluster's; the stub-era copy pointed it at a volume path that exists on no cluster node); ``HF_HUB_OFFLINE`` is dropped."""
    env = dict(base)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(Path(tree) / "src")
    # the standard Unix identity variable, set explicitly (never inherited) so a host-provided ``root`` cannot reach the
    # username-leak tests -- see SUITE_USER
    env["USER"] = SUITE_USER
    env["LOGNAME"] = SUITE_USER
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env[FULL_SUITE_ALLOW_ENV] = "1"          # this IS the suite; tests/conftest.py refuses a whole-tree run elsewhere
    for name in REGISTRY_ENV_POPPED:
        env.pop(name, None)
    env[REGISTRY_SPOOL_ENV] = tempfile.mkdtemp(prefix=REGISTRY_SPOOL_PREFIX, dir=str(spool_root) if spool_root is not None else None)
    env.pop("HF_HUB_OFFLINE", None)
    return env


def run_pytest(argv: list[str], tree: Path, env: dict, log: Path, timeout_s: int) -> tuple[int, bool]:
    """Run the suite subprocess (its own session, so a timeout kills the whole process group) with stdout+stderr into ``log``;
    returns ``(rc, timed_out)``. The throwaway registry spool ``env`` names (``suite_env``) is removed when the subprocess is gone,
    whatever its rc or a timeout: its records are the smoke fits' and belong nowhere."""
    timed_out = False
    try:
        with Path(log).open("wb") as f:
            proc = subprocess.Popen(argv, cwd=str(tree), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
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
    path = Path(path)
    if not path.is_file():
        return {"counts": None, "failures": [], "error": "pytest wrote no JUnit XML"}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {key: sum(int(float(s.attrib.get(key, 0) or 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    failures = []
    for tc in root.iter("testcase"):
        bad = list(tc.findall("failure")) + list(tc.findall("error"))
        if not bad:
            continue
        classname = tc.attrib.get("classname") or tc.attrib.get("file") or "unknown"
        node = f"{classname}::{tc.attrib.get('name', 'unknown')}"
        failures.append({"node": node, "kind": bad[0].tag, "message": (bad[0].attrib.get("message") or "").strip()[:500]})
    return {"counts": counts, "failures": failures}


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slurm_task_states(array_job_id: str) -> list | dict:
    """Every task's terminal state from sacct (which shards timed out / node-failed), or an error dict."""
    try:
        out = subprocess.run(["sacct", "-n", "-P", "-X", "-o", "JobID,JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End",
                              "-j", str(array_job_id)], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001 — diagnostics only
        return {"error": f"{type(e).__name__}: {e}"}
    keys = ("job_id", "job_id_raw", "state", "exit_code", "elapsed", "node", "start", "end")
    return [dict(zip(keys, line.split("|"))) for line in out.splitlines() if line.strip()]


def cmd_shard_files(a) -> None:
    for f in shard_files(Path(a.tree), a.index, a.count):
        print(f)


def cmd_finalize(a) -> None:
    dest = Path(a.results_root) / a.sha
    recs, missing = [], []
    for i in range(a.count):
        p = dest / "shards" / f"{i}-of-{a.count}" / "summary.json"
        r = None
        if p.is_file():
            try:
                r = json.loads(p.read_text())
            except ValueError:
                r = None
        if r is None:
            missing.append(i)
            r = {"sha": a.sha, "app": a.run_name, "status": "TIMEOUT", "pytest_rc": -9, "counts": None, "failures": [],
                 "shard": {"index": i, "count": a.count}, "results_dir": str(p.parent),
                 "error": "no shard record: the array task wrote no summary.json (killed at the walltime, node failure, "
                          "or it never ran) — not a gate result"}
        recs.append(r)
        print(f"[gate] shard {i}/{a.count}: status={r.get('status')} rc={r.get('pytest_rc')} counts={r.get('counts')} "
              f"failures={len(r.get('failures') or [])} elapsed_s={r.get('elapsed_s')} node={r.get('node')}", flush=True)
    rec = merge_shard_records(recs, a.sha, a.run_name)
    rec["substrate"] = "slurm"
    rec["gate"] = {"array_job_id": a.array_job_id, "finalize_job_id": os.environ.get("SLURM_JOB_ID"),
                   "cluster": os.environ.get("SLURM_CLUSTER_NAME"), "finalized_utc": utc_now(), "tree": a.tree,
                   "venv": next((r.get("venv") for r in recs if r.get("venv")), None),
                   "missing_shards": missing, "shard_count": a.count,
                   "env_overrides": next((r.get("env_overrides") for r in recs if r.get("env_overrides")), None)}
    if a.array_job_id:
        rec["slurm_tasks"] = slurm_task_states(a.array_job_id)
    write_json_atomic(dest / "summary.json", rec)
    nodes = [f.get("node") for f in rec.get("failures") or []]
    print(f"[gate] SUMMARY sha={a.sha} status={rec.get('status')} pytest_rc={rec.get('pytest_rc')} counts={rec.get('counts')} "
          f"elapsed_s={rec.get('elapsed_s')} shards={rec.get('shards')} layout={rec.get('shard_layout', {}).get('map_sha256', '')[:12]} "
          f"missing_shards={missing}", flush=True)
    print("[gate] FAILURE NODES (%d):" % len(nodes), flush=True)
    for n in nodes:
        print("  " + str(n), flush=True)
    print(f"[gate] wrote {dest / 'summary.json'}", flush=True)
    if rec.get("status") not in ("PASS", "FAIL"):
        print(f"[gate] NON-TERMINAL status {rec.get('status')}: not a gate result", flush=True)
        sys.exit(3)


def _file_module(path: str) -> str:
    """``tests/test_x.py`` -> the JUnit classname prefix ``tests.test_x`` (parse_junit's node = classname::name)."""
    return path[:-3].replace("/", ".") if path.endswith(".py") else path.replace("/", ".")


def _node_in_files(node: str, mods: list) -> bool:
    for m in mods:
        if node.startswith(m + "::") or node.startswith(m + ".") or node == f"unknown::{m}" or node.startswith(f"unknown::{m}"):
            return True
    return False


def cmd_amend(a) -> None:
    base = json.loads(Path(a.base_summary).read_text())
    rerun_p = Path(a.rerun_dir) / "summary.json"
    if not rerun_p.is_file():
        raise SystemExit(f"[gate] no re-run record at {rerun_p}")
    rerun = json.loads(rerun_p.read_text())
    if rerun.get("status") not in ("PASS", "FAIL") or not rerun.get("counts") or not rerun["counts"].get("tests"):
        raise SystemExit(f"[gate] re-run is not a terminal result with tests: status={rerun.get('status')} counts={rerun.get('counts')}")
    want = sorted(a.files)
    got = sorted((rerun.get("shard") or {}).get("files") or [])
    if got != want:
        raise SystemExit(f"[gate] re-run ran {got}, amend asked for {want}")
    mods = [_file_module(f) for f in want]
    old = [f for f in base.get("failures") or [] if _node_in_files(f["node"], mods)]
    keep = [f for f in base.get("failures") or [] if not _node_in_files(f["node"], mods)]
    new = [f for f in rerun.get("failures") or [] if _node_in_files(f["node"], mods)]
    stray = [f["node"] for f in rerun.get("failures") or [] if not _node_in_files(f["node"], mods)]
    if stray:
        raise SystemExit(f"[gate] re-run reports nodes outside the named files: {stray}")
    old_n, new_n = {f["node"]: f for f in old}, {f["node"]: f for f in new}
    entries = []
    for node in sorted(set(old_n) | set(new_n)):
        entries.append({"node": node, "old": "failed" if node in old_n else "passed", "new": "failed" if node in new_n else "passed",
                        "old_kind": old_n.get(node, {}).get("kind"), "new_kind": new_n.get(node, {}).get("kind"),
                        "job_id": a.job_id, "reason": a.reason, "rerun_dir": str(a.rerun_dir),
                        "rerun_elapsed_s": rerun.get("elapsed_s"), "rerun_env_overrides": rerun.get("env_overrides")})
    counts = dict(base["counts"])
    for kind in ("failure", "error"):
        key = "failures" if kind == "failure" else "errors"
        counts[key] = int(counts.get(key) or 0) - sum(1 for f in old if f.get("kind") == kind) + sum(1 for f in new if f.get("kind") == kind)
    out = dict(base)
    out["failures"] = sorted(keep + new, key=lambda f: str(f.get("node")))
    out["counts"] = counts
    out["status"] = "FAIL" if out["failures"] else "PASS"
    out["pytest_rc"] = (base.get("pytest_rc") or 1) if out["failures"] else 0
    out.setdefault("amendments", []).extend(entries)
    out["amendment_log"] = (out.get("amendment_log") or []) + [{
        "files": want, "job_id": a.job_id, "reason": a.reason, "amended_utc": utc_now(), "base_summary": str(a.base_summary),
        "rerun_dir": str(a.rerun_dir), "rerun_counts": rerun.get("counts"), "nodes_removed": sorted(old_n), "nodes_added": sorted(new_n),
        "unchanged_failed_nodes_in_files": sorted(set(old_n) & set(new_n))}]
    dest = Path(a.results_root) / a.sha / "summary.json"
    write_json_atomic(dest, out)
    print(f"[gate] AMENDED {dest}: files={want} removed={sorted(old_n)} added={sorted(new_n)} counts={counts} status={out['status']} "
          f"(re-run: {rerun['counts']} in {rerun.get('elapsed_s')}s, job {a.job_id})", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard-files"); s.add_argument("tree"); s.add_argument("index", type=int); s.add_argument("count", type=int)
    s.set_defaults(fn=cmd_shard_files)
    f = sub.add_parser("finalize")
    f.add_argument("--sha", required=True); f.add_argument("--tree", required=True); f.add_argument("--results-root", required=True)
    f.add_argument("--count", type=int, required=True); f.add_argument("--run-name", required=True); f.add_argument("--array-job-id", default=None)
    f.set_defaults(fn=cmd_finalize)
    m = sub.add_parser("amend")
    m.add_argument("--sha", required=True); m.add_argument("--results-root", required=True); m.add_argument("--base-summary", required=True)
    m.add_argument("--rerun-dir", required=True); m.add_argument("--files", nargs="+", required=True)
    m.add_argument("--job-id", required=True); m.add_argument("--reason", required=True)
    m.set_defaults(fn=cmd_amend)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
