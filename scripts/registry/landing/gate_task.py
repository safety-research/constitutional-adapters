#!/usr/bin/env python3
"""One shard of the Slurm landing gate, run under the shared gate venv's python inside one array task.

The steps and the functions are gate_lib's (no suite module is loaded from the tree any more): copy the
exported tree to node-local scratch, ``verify_checkout`` (exact HEAD, clean), ``dependency_hashes`` against the venv's ready
marker (one commit is never tested under another commit's environment), ``shard_files`` (the by-name partition), ``pytest_argv``
(``--tb=short --continue-on-collection-errors --junitxml``), ``suite_env`` + ``run_pytest`` (throwaway registry spool,
USER=modalsuiteuser, WANDB off, popped registry knobs, process-group timeout), ``parse_junit``, and the shard record written to
``<results-root>/<sha>/shards/<i>-of-<N>/{pytest.log,junit.xml,summary.json}``.

Cluster-specific choices, all recorded in the record's ``env_overrides``:
  PYTHONPATH = <tree>/src (the suite needs nothing else on the path)
  HF_HOME    = the cluster's (hf_env.sh; tiny test models are in the cohort cache)
  HOME/TMPDIR= fresh per-task node-local dirs (a pristine home like a container's; tmp files never land on NFS)
  PATH       = <venv>/bin first (console scripts a test spawns resolve)
  SLURM_*/SLURMD_*/SRUN_*/SBATCH_* removed (the suite sees a plain CPU box; substrate probes stay off)
Exit code: 0 for PASS or FAIL (a record was written — the verdict is summary.json's, not sacct's), 3 TIMEOUT, 2 refusal.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_lib import dependency_hashes, parse_junit, pytest_argv, run_pytest, shard_files, suite_env, utc_now, verify_checkout, write_json_atomic  # noqa: E402

STRIP_PREFIXES = ("SLURM_", "SLURMD_", "SRUN_", "SBATCH_")
#: Slurm's client and daemon commands. With --hide-slurm the suite's PATH is a symlink farm of the host's PATH dirs
#: MINUS these, so a tool that picks `srun` when it is on the host (screen_verdict.reclassify_runner) takes its
#: "this host directly (srun unavailable)" path. Env scrubbing alone is not enough: with SLURM_* gone the
#: nested srun asked for a NEW allocation, which landed on another node where the task's tmp dir did not exist
#: (attempt 1: stray jobs D_modalsuiteuser_screen-reclassify-* 697054/697077, one INTRODUCED node on both tips).
SLURM_CMDS = frozenset("srun sbatch salloc squeue sacct sacctmgr scontrol sinfo scancel sstat sprio sshare sattach sbcast sdiag "
                       "sreport strigger sview scrontab seff sh5util sgather sjobexitmod sjstat slurmd slurmstepd slurmctld "
                       "slurmdbd slurmrestd".split())


def slurm_free_path(work: Path, host_path: str) -> tuple[str, list]:
    """A symlink farm ``work/bin`` mirroring every entry of ``host_path``'s directories (first dir wins, as PATH does)
    except SLURM_CMDS; returns (its path, the hidden command names found)."""
    farm = work / "bin"
    farm.mkdir(parents=True, exist_ok=True)
    hidden, seen = [], set()
    for d in host_path.split(os.pathsep):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            if name in SLURM_CMDS:
                hidden.append(f"{d}/{name}")
                continue
            try:
                os.symlink(os.path.join(d, name), farm / name)
            except OSError:
                pass
    return str(farm), sorted(hidden)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--tree", required=True, help="the exported tree on shared storage (git checkout at --sha)")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--work", required=True, help="node-local scratch for this task (removed by the sbatch trap)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--timeout-s", type=int, default=2400)
    ap.add_argument("--hide-slurm", action="store_true", help="PATH without Slurm's commands (SLURM_CMDS): the suite sees no srun")
    ap.add_argument("--files", nargs="+", default=None,
                    help="run exactly these tree-relative test files instead of the shard partition (a targeted re-run); "
                         "the record goes to <results-root>/<sha>/reruns/<--rerun-label>/ and never into shards/")
    ap.add_argument("--rerun-label", default=None)
    ap.add_argument("--local-venv", action="store_true",
                    help="run pytest from a node-local byte copy of --venv: fresh interpreters a test spawns "
                         "import from local disk instead of VAST, which is what made shard 11 ~3x slower here (D-state imports)")
    a = ap.parse_args()
    started = time.time()
    if a.files:
        import hashlib
        label = a.rerun_label or ("files-" + hashlib.sha256("\n".join(a.files).encode()).hexdigest()[:8])
        dest = Path(a.results_root) / a.sha / "reruns" / label
    else:
        dest = Path(a.results_root) / a.sha / "shards" / f"{a.index}-of-{a.count}"
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    base = {"sha": a.sha, "app": a.run_name, "shard": {"index": a.index, "count": a.count}, "substrate": "slurm",
            "node": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "venv": a.venv, "started_utc": utc_now()}

    def refuse(status: str, **extra) -> int:
        rec = {**base, "status": status, "pytest_rc": None, "elapsed_s": round(time.time() - started, 1), **extra}
        write_json_atomic(dest / "summary.json", rec)
        print("[gate] REFUSED " + json.dumps(rec, indent=1, default=str), flush=True)
        return 2

    tree = work / f"sisteer-{a.sha}"
    shutil.rmtree(tree, ignore_errors=True)
    t0 = time.time()
    shutil.copytree(a.tree, tree, symlinks=True)
    print(f"[gate] copied tree -> {tree} in {time.time() - t0:.1f}s", flush=True)
    checked = verify_checkout(tree, a.sha)
    if not checked["ok"]:
        return refuse("BAD_TREE", tree=checked, error=f"tree does not verify: {checked['why']}")
    got = dependency_hashes(tree)
    ready_p = Path(a.venv) / ".gate-ready.json"
    if not ready_p.is_file():
        return refuse("NO_VENV", error=f"no ready marker at {ready_p}")
    ready = json.loads(ready_p.read_text())
    if got != ready.get("dependency_hashes"):
        return refuse("DEPENDENCY_MISMATCH", expected=ready.get("dependency_hashes"), staged=got,
                      error="tree's pyproject.toml/uv.lock differ from the venv's build inputs")

    files = list(a.files) if a.files else shard_files(tree, a.index, a.count)
    shard = {"index": a.index, "count": a.count, "n_files": len(files), "files": list(files)}
    if a.files:
        shard["rerun"] = True
        shard["label"] = dest.name
        missing = [f for f in files if not (tree / f).is_file()]
        if missing:
            return refuse("BAD_FILES", error=f"--files not in the tree: {missing}")
    if not files:
        rec = {**base, "shard": shard, "status": "PASS", "pytest_rc": 0, "elapsed_s": round(time.time() - started, 1),
               "command": [], "tree": checked, "dependency_hashes": got, "versions": {},
               "counts": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}, "failures": [],
               "note": "empty shard (more shards than test files)", "results_dir": str(dest)}
        write_json_atomic(dest / "summary.json", rec)
        print("[gate] SUMMARY " + json.dumps(rec, indent=1, default=str), flush=True)
        return 0

    local_out = work / "out"
    local_out.mkdir(parents=True, exist_ok=True)
    log, junit = local_out / "pytest.log", local_out / "junit.xml"
    venv_used = Path(a.venv)
    if a.local_venv:
        t0 = time.time()
        venv_used = work / "venv"
        shutil.rmtree(venv_used, ignore_errors=True)
        import subprocess
        # no -a: VAST carries ACL/xattr metadata the node's overlay /tmp cannot take ("preserving permissions: Operation
        # not supported" -> cp rc 1 after a complete copy). Mode + timestamps + symlinks are all a venv needs; the copy
        # is then VERIFIED by importing from it rather than trusted from cp's rc.
        cp = subprocess.run(["cp", "-dR", "--preserve=mode,timestamps", a.venv, str(venv_used)], capture_output=True, text=True)
        probe = subprocess.run([str(venv_used / "bin" / "python"), "-c", "import pytest, torch, transformers; print(torch.__version__)"],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            return refuse("NO_VENV", error="node-local venv copy does not import pytest/torch/transformers",
                          cp_rc=cp.returncode, cp_stderr_tail=cp.stderr[-1500:], probe_stderr_tail=probe.stderr[-1500:])
        print(f"[gate] copied venv -> {venv_used} in {time.time() - t0:.1f}s (cp rc={cp.returncode}, "
              f"{len(cp.stderr.splitlines())} cp warnings; import probe OK: torch {probe.stdout.strip()})", flush=True)
    python = str(venv_used / "bin" / "python")
    argv = pytest_argv(python, str(junit), files)

    cluster_hf_home = os.environ.get("HF_HOME")
    env = suite_env(os.environ, tree, spool_root=work)   # the suite's env: spool, $USER, WANDB, popped knobs, PYTHONPATH=src
    home, tmp = work / "home", work / "tmp"
    home.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    if cluster_hf_home:
        env["HF_HOME"] = cluster_hf_home
    else:
        env.pop("HF_HOME", None)
    env["HOME"] = str(home)
    env["TMPDIR"] = str(tmp)
    hidden_slurm: list = []
    if a.hide_slurm:
        farm, hidden_slurm = slurm_free_path(work, env.get("PATH", ""))
        env["PATH"] = f"{venv_used / 'bin'}:{farm}"
    else:
        env["PATH"] = f"{venv_used / 'bin'}:{env.get('PATH', '')}"
    stripped = sorted(k for k in env if k.startswith(STRIP_PREFIXES))
    for k in stripped:
        env.pop(k, None)
    overrides = {"PYTHONPATH": env["PYTHONPATH"], "HF_HOME": env.get("HF_HOME"), "HOME": env["HOME"], "TMPDIR": env["TMPDIR"],
                 "USER": env.get("USER"), "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"), "stripped_env": stripped,
                 "PATH_prefix": str(venv_used / "bin"), "venv_local_copy": bool(a.local_venv), "venv_source": a.venv,
                 "timeout_s": int(a.timeout_s), "hide_slurm": bool(a.hide_slurm), "hidden_slurm_cmds": hidden_slurm,
                 "PATH": env["PATH"] if a.hide_slurm else None}
    print(f"[gate] app={a.run_name} sha={a.sha} shard={a.index}/{a.count} n_files={len(files)} timeout_s={a.timeout_s}", flush=True)
    print("[gate] command: " + " ".join(argv[:3]) + f" <{len(files)} files> " + " ".join(argv[3 + len(files):]), flush=True)
    print("[gate] env overrides: " + json.dumps(overrides, default=str), flush=True)
    rc, timed_out = run_pytest(argv, tree, env, log, int(a.timeout_s))
    parsed = parse_junit(junit)
    try:
        import importlib.metadata as md
        versions = {name: md.version(name) for name in ("pytest", "torch", "transformers", "datasets", "numpy")}
    except Exception as e:  # noqa: BLE001
        versions = {"error": f"{type(e).__name__}: {e}"}
    rec = {**base, "status": "TIMEOUT" if timed_out else ("PASS" if rc == 0 else "FAIL"), "pytest_rc": rc,
           "elapsed_s": round(time.time() - started, 1), "command": argv, "tree": checked, "dependency_hashes": got,
           "versions": versions, **parsed, "shard": shard, "env_overrides": overrides, "finished_utc": utc_now(),
           "results_dir": str(dest)}
    for name, p in (("pytest.log", log), ("junit.xml", junit)):
        if p.is_file():
            tmpd = dest / (name + ".tmp")
            shutil.copyfile(p, tmpd)
            os.replace(tmpd, dest / name)
    write_json_atomic(dest / "summary.json", rec)
    shown = {k: v for k, v in rec.items() if k not in ("command", "shard", "env_overrides")}
    shown["shard"] = {k: v for k, v in shard.items() if k != "files"}
    print("[gate] SUMMARY " + json.dumps(shown, indent=1, default=str), flush=True)
    if log.is_file():
        tail = log.read_text(errors="replace").splitlines()[-40:]
        print("[gate] LOG TAIL\n" + "\n".join(tail), flush=True)
    shutil.rmtree(tree, ignore_errors=True)
    if a.local_venv:
        shutil.rmtree(venv_used, ignore_errors=True)
    return 3 if timed_out else 0


if __name__ == "__main__":
    sys.exit(main())
