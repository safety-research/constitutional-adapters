"""The grading daemon OWNS the Petri drive's start.

The Petri proxy job serves the target on its cluster and writes `<wave root>/petri/<tag>/serving/endpoint.json` when it is up; the drive
(`petri_drive_full22.sh <model> <view> <cluster> <wave sha>`) runs on the grading host against it. The daemon's Petri pass runs on its own
thread (12), re-discovers the waves from the model's state every pass (9), prepares a persistent detached checkout at the WAVE sha with the
grading host bank view built there and runs the drive from it (10), holds the proxy job the endpoint names to RUNNING and keys every outcome on
the endpoint identity — one completed drive per identity, failures retried under a cap of three (11) — and claims under a per-tag flock
before it spawns, reconciling a claim without a live process on the next pass (13). Executed with a fake `ssh` that runs the remote command
locally (a fake `squeue` beside it answers job states from files), a fake drive and a fake make_bank_view.py committed in a REAL git
repository whose HEAD moves past the wave sha, and a real venv for the interpreter. Synthetic site, neutral labels, job ids 11111 / 22222."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import stat
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, fake_python, run_stage, write_site  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "registry"))
import grade_daemon as GD  # noqa: E402

from sisteer.registry import db as DB  # noqa: E402

MODEL, VIEW, TAG = "qwen35_4b_demo", "battery", "qwen35_4b_demo_battery"      # the demo's one wave: the row's DECLARED `battery` (demo-import M3)

# a fake ssh: `-n` = stdin from /dev/null (as the real one); the remote command is the LAST argument and runs here, with this bin dir
# first on PATH so `squeue` is the fake beside it (the endpoint and the job states live under the test's tmp)
FAKE_SSH = '''#!/bin/bash
case " $* " in *" -n "*) exec </dev/null;; esac
export PATH="$(cd "$(dirname "$0")" && pwd):$PATH"
cmd="${@: -1}"
exec bash -c "$cmd"
'''
# a fake squeue: `squeue -j <id> -h -o %T` answers from $FAKE_SQUEUE_DIR/<id> (its state), Slurm's "Invalid job id" for a job not there,
# and a broken controller when $FAKE_SQUEUE_DIR/BROKEN exists
FAKE_SQUEUE = '''#!/bin/bash
d="${FAKE_SQUEUE_DIR:?}"; id=""; prev=""
for a in "$@"; do [ "$prev" = "-j" ] && id="$a"; prev="$a"; done
[ -e "$d/BROKEN" ] && { echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1; }
[ -f "$d/$id" ] && { cat "$d/$id"; exit 0; }
echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1
'''
# a fake secret-manager CLI (named by SISTEER_SECRET_CMD, which the drive's allowlisted environment carries): one deterministic fake item per
# org name, so the hash the daemon expects (`<command> <org> | sha256sum`, a pipeline) and the hash the fake drive attests can be computed by the
# test; fails while $FAKE_OP_FAIL exists
FAKE_SECRET_CMD = '''#!/bin/bash
[ -e "${FAKE_OP_FAIL:-/nonexistent}" ] && { echo "fake-secret-cmd: vault unreachable (fake)" >&2; exit 1; }
echo "fake-vault-item-$1"
'''
# the fake drive (COMMITTED in the fake repository, so the checkout at the wave sha carries it): records how it was called, holds while
# $FAKE_DRIVE_HOLD exists (bounded), writes the result summary the real drive's verdict writes to $PETRI_DRIVE_RESULT_FILE (the content of
# $FAKE_DRIVE_RESULT when that control file exists — `none` = write nothing — else a complete one), exits with the code in $FAKE_DRIVE_RC
FAKE_DRIVE = '''#!/bin/bash
echo "secrets=$SISTEER_SECRETS_MODE wt=$F22_WT py=$F22_PY pythonpath=$PYTHONPATH cwd=$PWD bank=${F22_GRADING_HOST_BANK_ROOT:-unset} cells=${F22_PETRI_CELLS:-unset} nargs=$# args=$*" >> "${FAKE_DRIVE_LOG:?}"
echo "key_present=${ANTHROPIC_API_KEY+1}" >> "${FAKE_DRIVE_LOG:?}.env"     # whether a static Anthropic key reached this drive (never its value)
echo "drive running for $*"
# the real drive's DRIVE_START attests sha256[:12] of the key it resolved for $F22_AUDITOR_ORG (never the key); $FAKE_DRIVE_KEY_SHA12 forges the
# attestation, $FAKE_DRIVE_NO_ATTEST (a file) makes this a drive from before the field existed. Under SISTEER_SECRETS_MODE=wif the real drive
# attests `auditor_auth=wif:<org> ids_sha12=<sha256(ids file)[:12]>` instead and never calls the secret-manager CLI; $FAKE_DRIVE_IDS_SHA12 forges that digest,
# $FAKE_DRIVE_ATTEST_KEY_UNDER_WIF (a file) makes this a drive at a wave sha still on the key path
org="${F22_AUDITOR_ORG:-lowprio}"
if [ "${SISTEER_SECRETS_MODE:-op}" = wif ]; then
  case "$org" in lowprio) stem=loprio;; highprio) stem=hiprio;; *) stem=$org;; esac
  ids=$(sha256sum < "${SISTEER_WIF_ENV_DIR:-$HOME/.anthropic}/wif-$stem.env" | cut -c1-12); [ -n "${FAKE_DRIVE_IDS_SHA12:-}" ] && ids="$FAKE_DRIVE_IDS_SHA12"
  attest="auditor_auth=wif:$org ids_sha12=$ids"
  [ -e "${FAKE_DRIVE_ATTEST_KEY_UNDER_WIF:-/nonexistent}" ] && attest="auditor_key_sha12=$(printf %s stale-key-path | sha256sum | cut -c1-12)"
else
  sha=$("${SISTEER_SECRET_CMD:?}" "$org" | tr -d '\n' | sha256sum | cut -c1-12); [ -n "${FAKE_DRIVE_KEY_SHA12:-}" ] && sha="$FAKE_DRIVE_KEY_SHA12"
  attest="auditor_key_sha12=$sha"
fi
if [ -e "${FAKE_DRIVE_NO_ATTEST:-/nonexistent}" ]; then echo "[t] DRIVE_START fake drive_id=$$ auditor_org=$org"
else echo "[t] DRIVE_START fake drive_id=$$ auditor_org=$org $attest secrets_mode=${SISTEER_SECRETS_MODE:-op} cap=${PETRI_AUDITOR_RETRY_SLEEP_CAP_S:-unset} width=${PETRI_EVAL_MAX_CONNECTIONS:-unset}"; fi
# a LEG that outlives this wrapper (the real drive's legs are subshells of its session): lingers while $FAKE_DRIVE_LINGER exists (bounded), ignoring
# TERM when $FAKE_DRIVE_LINGER_IGNORE_TERM exists; its pid is appended to $FAKE_DRIVE_LINGER.pids
if [ -e "${FAKE_DRIVE_LINGER:-/nonexistent}" ]; then
  ( [ -e "${FAKE_DRIVE_LINGER_IGNORE_TERM:-/nonexistent}" ] && trap '' TERM; i=0; while [ -e "$FAKE_DRIVE_LINGER" ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i+1)); done ) >/dev/null 2>&1 </dev/null &
  echo "$!" >> "${FAKE_DRIVE_LINGER}.pids"
fi
echo "ovf=${F22_OVERFLOW_RERUN:-unset} nargs=$# args=$*" >> "${FAKE_DRIVE_LOG:?}.ovf"     # the overflow pass's knob and positional 5 (petri-completion), never its own line
i=0; while [ -n "${FAKE_DRIVE_HOLD:-}" ] && [ -e "$FAKE_DRIVE_HOLD" ] && [ $i -lt 300 ]; do sleep 0.1; i=$((i+1)); done
if [ -n "${PETRI_DRIVE_RESULT_FILE:-}" ] && [ -n "${FAKE_DRIVE_RESULT:-}" ] && [ -f "$FAKE_DRIVE_RESULT" ] && [ "$(cat "$FAKE_DRIVE_RESULT")" != none ]; then
  V="$PETRI_DRIVE_RESULT_FILE.verify.json"; [ -f "${FAKE_DRIVE_VERIFY:-/nonexistent}" ] && cp "$FAKE_DRIVE_VERIFY" "$V"
  sed "s#__VERIFY__#$V#" "$FAKE_DRIVE_RESULT" > "$PETRI_DRIVE_RESULT_FILE"
fi
# the real drive touches the proxy's STOP when every leg in scope ended and none failed (the proxy then exits and unlinks endpoint.json)
[ -f "${FAKE_DRIVE_STOP:-/nonexistent}" ] && [ -n "${FAKE_ENDPOINT:-}" ] && touch "$(dirname "$FAKE_ENDPOINT")/STOP"
rc=0; [ -n "${FAKE_DRIVE_RC:-}" ] && [ -f "$FAKE_DRIVE_RC" ] && rc=$(cat "$FAKE_DRIVE_RC")
exit "$rc"
'''
# a runner for the two-PROCESS tests: one pass over one spec (or, with `prune`, prune in a loop until the stop file appears); the summary
# as the last stdout line (the daemon's event lines precede it)
RUNNER = '''import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import grade_daemon as GD
GD.PETRI_IMPORT_CHECK = "import json"
cfg = json.load(open(sys.argv[2]))
while not os.path.exists(sys.argv[3]):
    time.sleep(0.002)
if len(sys.argv) > 4 and sys.argv[4] == "prune":
    n = 0; stop = sys.argv[5]
    while not os.path.exists(stop):
        n += len(GD.prune_petri_checkouts(state_dir=cfg["kw"]["state_dir"], repo=cfg["kw"]["repo"], specs=[cfg["spec"]])); time.sleep(0.01)
    print(json.dumps({"pruned": n}), flush=True)
else:
    print(json.dumps(GD.petri_drive_pass([cfg["spec"]], **cfg["kw"])), flush=True)
'''
OBJECTS = ("none", "prompted__a", "prompted__b")                        # the proxy job registers one run per (eval, object)
EVALS = tuple(f"petri_b{i}" for i in range(6)) + ("petri_pooled",)
# the fake make_bank_view.py (committed too): --check passes only when the view's VIEW_MANIFEST.json exists; a build writes it
FAKE_MBV = '''import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--bank"); ap.add_argument("--target"); ap.add_argument("--view"); ap.add_argument("--roster"); ap.add_argument("--check", action="store_true")
a = ap.parse_args()
view = os.path.join(a.bank, "cells_view", a.view); man = os.path.join(view, "VIEW_MANIFEST.json")
with open(os.environ["FAKE_MBV_LOG"], "a") as fh: fh.write(("check " if a.check else "build ") + view + " roster=" + a.roster + "\\n")
if a.check:
    print("ok" if os.path.isfile(man) else "PROBLEM no view"); sys.exit(0 if os.path.isfile(man) else 2)
os.makedirs(view, exist_ok=True); json.dump({"target": a.target, "view": a.view, "entries": []}, open(man, "w")); print("ok: built")
'''


def _exe(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IXUSR); return p


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=str(cwd), capture_output=True, text=True, check=True).stdout.strip()


class World:
    """The fakes, the repository, the state dir and the spec of one (model, view) wave."""

    def __init__(self, tmp: Path, monkeypatch):
        self.tmp = tmp
        self.bin = tmp / "bin"; _exe(self.bin / "ssh", FAKE_SSH); _exe(self.bin / "squeue", FAKE_SQUEUE); _exe(self.bin / "fake-secret-cmd", FAKE_SECRET_CMD)
        self.op_fail = tmp / "op_fail"
        monkeypatch.setenv("PATH", f"{self.bin}:{os.environ.get('PATH', '')}"); monkeypatch.setenv("FAKE_OP_FAIL", str(self.op_fail))
        monkeypatch.setenv("SISTEER_SECRET_CMD", str(self.bin / "fake-secret-cmd"))      # the fake vault, named by the site setting
        self.squeue_dir = tmp / "squeue"; self.squeue_dir.mkdir()
        self.repo = tmp / "repo"; self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _exe(self.repo / GD.PETRI_DRIVE_SCRIPT_DEFAULT, FAKE_DRIVE)
        (self.repo / "scripts/sdfbase/full22/make_bank_view.py").write_text(FAKE_MBV)
        (self.repo / "configs").mkdir(); (self.repo / "configs/sdfbase_full22_roster.yaml").write_text("targets: {}\n")
        (self.repo / "src").mkdir(); (self.repo / "src/README").write_text("the checkout's src\n")
        (self.repo / ".gitignore").write_text(".venv/\n")
        _git(self.repo, "add", "-A"); _git(self.repo, "commit", "-q", "-m", "the launched commit")
        self.sha = _git(self.repo, "rev-parse", "HEAD")
        (self.repo / "scripts/sdfbase/full22/later.txt").write_text("a commit after 06\n")
        _git(self.repo, "add", "-A"); _git(self.repo, "commit", "-q", "-m", "a later commit"); self.tip = _git(self.repo, "rev-parse", "HEAD")
        assert self.tip != self.sha
        self.venv = tmp / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(self.venv)], check=True, capture_output=True)
        self.python = str(self.venv / "bin" / "python")
        self.state_root = tmp / "state" / MODEL; self.state_root.mkdir(parents=True)
        self.state = self.state_root / "grade"
        self.remote = tmp / "remote" / "exp" / "sdfbase_full22_native"
        self.endpoint = self.remote / "petri" / TAG / "serving" / "endpoint.json"
        self.bundle = tmp / "bundle"; self.bundle.mkdir(); self.bank_root = tmp / "bank"
        self.drive_log = tmp / "drive_calls.log"; self.mbv_log = tmp / "mbv_calls.log"; self.mbv_log.write_text("")
        self.hold = tmp / "hold"; self.rc = tmp / "drive_rc"; self.result_ctl = tmp / "drive_result_ctl"; self.verify_ctl = tmp / "drive_verify_ctl"
        self.stop_ctl = tmp / "drive_stop_ctl"
        self.shared = tmp / "remote"                                          # the cluster's $SHARED: registry/pending/<job>.json lives under it
        monkeypatch.setenv("FAKE_DRIVE_LOG", str(self.drive_log)); monkeypatch.setenv("FAKE_SQUEUE_DIR", str(self.squeue_dir))
        monkeypatch.setenv("FAKE_MBV_LOG", str(self.mbv_log)); monkeypatch.setenv("FAKE_DRIVE_HOLD", str(self.hold)); monkeypatch.setenv("FAKE_DRIVE_RC", str(self.rc))
        monkeypatch.setenv("FAKE_DRIVE_RESULT", str(self.result_ctl)); monkeypatch.setenv("FAKE_DRIVE_VERIFY", str(self.verify_ctl))
        monkeypatch.setenv("FAKE_DRIVE_STOP", str(self.stop_ctl)); monkeypatch.setenv("FAKE_ENDPOINT", str(self.endpoint))
        monkeypatch.setenv("SISTEER_SECRETS_MODE", "op"); monkeypatch.delenv("F22_ROSTER", raising=False)
        monkeypatch.setattr(GD, "PETRI_IMPORT_CHECK", "import json")          # the real drive imports inspect_ai + petri_bloom; the venv here has neither
        for name in ("_PETRI_SAID", "_PETRI_PREPARED", "_PETRI_DISCOVERY_SAID"):
            monkeypatch.setattr(GD, name, set())
        monkeypatch.setattr(GD, "_PETRI_PY_FIT", {}); monkeypatch.setattr(GD, "_PETRI_PROCS", {})
        self.spec = {"tag": TAG, "model": MODEL, "view": VIEW, "cluster": "cluster_a", "ssh": "cluster-a-login", "endpoint": str(self.endpoint), "sha": self.sha, "job_id": "11111",
                     "shared": str(self.shared)}
        self.kw = dict(state_dir=str(self.state), repo=str(self.repo), script=GD.PETRI_DRIVE_SCRIPT_DEFAULT, ssh_bin=str(self.bin / "ssh"), dry_run=False,
                       python=self.python, bank_root=str(self.bank_root), bundle_dir=str(self.bundle), env_allow=["FAKE_*"])   # the fakes' knobs ride the allowlist

    def run_ids(self, job_id: str, objects=OBJECTS) -> dict:
        """{object: [run ids]} of the proxy job's registration (one run per (eval, object))."""
        return {o: [f"run_{job_id}_{o}_{e}" for e in EVALS] for o in objects}

    def register(self, job_id: str) -> None:
        """The proxy job's job-start registration on the cluster: `<shared>/registry/pending/<job>.json` (f22_registry_register's shape)."""
        pend = self.shared / "registry" / "pending" / f"{job_id}.json"; pend.parent.mkdir(parents=True, exist_ok=True)
        runs = [{"run_id": rid, "route": {"eval_id": e, "object": o}, "plan": {"campaign": "sdfbase_full22", "eval_id": e}}
                for o, rids in self.run_ids(job_id).items() for e, rid in zip(EVALS, rids)]
        pend.write_text(json.dumps({"job_id": job_id, "runs": runs}))

    def drive_result(self, job_id: str, filed_objects=OBJECTS, out_objects=(), skipped_objects=(), verdict="complete", rc=0, planned=None, filed=None, stop=None) -> None:
        """What the fake drive will leave behind: the verify summary rows (scope in / out / skipped, filed for `filed_objects`), the result
        summary whose handoff block counts the drive's OWN scope (runs_planned = in scope, runs_filed = filed) unless overridden, and whether it
        touches the proxy's STOP (the real drive does when every leg in scope ended and none failed: default = a complete verdict)."""
        rows = []
        for o, rids in self.run_ids(job_id).items():
            for e, rid in zip(EVALS, rids):
                scope = "skipped" if o in skipped_objects else ("out" if o in out_objects else "in")
                f = scope == "in" and o in filed_objects
                rows.append({"run_id": rid, "eval_id": e, "object": o, "object_wave": o, "scope": scope, "filed": f, "state": "generated" if f else None,
                             "store_uri": f"/cells/{o}" if f else None})
        n_in = len([r for r in rows if r["scope"] == "in"]); n_f = len([r for r in rows if r["filed"]])
        self.verify_ctl.write_text(json.dumps({"n_registered": len(rows), "n_planned": n_in, "n_filed": n_f, "runs": rows}))
        self.result_ctl.write_text(json.dumps({"verdict": verdict, "rc": rc, "legs": {"complete": n_f // 7, "incomplete_overflow": 0, "failed": 0},
                                               "handoff": {"mode": "on", "finish_rc": 0, "verify_rc": 0 if n_f == n_in else 2, "summary": "__VERIFY__",
                                                           "runs_planned": planned if planned is not None else n_in, "runs_filed": filed if filed is not None else n_f}}))
        if (verdict == "complete" and rc == 0) if stop is None else stop:
            self.stop_ctl.write_text("")
        else:
            self.stop_ctl.unlink(missing_ok=True)

    def proxy_stops(self, job_id: str) -> None:
        """What the proxy does after STOP: the standup exits and unlinks endpoint.json; the job leaves the queue."""
        assert (self.endpoint.parent / "STOP").exists(), "the drive did not touch STOP"
        self.endpoint.unlink(missing_ok=True); self.job_gone(job_id)

    def endpoint_up(self, job_id: str, mtime: int, skipped=()) -> None:
        """The proxy is up: endpoint.json (naming its job and its cells_served.json), the standup's record with the cells it skipped by name,
        the job's registration on the cluster, and — unless a test overrides it — a drive result that files every registered run."""
        self.endpoint.parent.mkdir(parents=True, exist_ok=True)
        cs = self.endpoint.parent / "cells_served.json"
        cs.write_text(json.dumps({"engine": "lora_native", "cells_skipped": [{"cell_id": o, "reason": "skipped by name"} for o in skipped]}))
        self.endpoint.write_text(json.dumps({"slurm_job_id": job_id, "mode": "petri", "node": "TESTHOST", "cells_served": str(cs)}))
        os.utime(self.endpoint, (mtime, mtime))
        self.register(job_id)
        self.drive_result(job_id, filed_objects=tuple(o for o in OBJECTS if o not in skipped), skipped_objects=tuple(skipped))

    def job(self, job_id: str, state: str) -> None:
        (self.squeue_dir / job_id).write_text(state + "\n")

    def job_gone(self, job_id: str) -> None:
        (self.squeue_dir / job_id).unlink(missing_ok=True)

    def run_pass(self, spec=None, **over):
        return GD.petri_drive_pass([spec or self.spec], **dict(self.kw, **over))

    def events(self, kind=None):
        p = self.state / "grade_daemon.jsonl"
        ev = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        return [e for e in ev if kind is None or e["event"] == kind]

    def record(self) -> dict:
        return json.loads((self.state / f"petri_drive.{TAG}.json").read_text())

    def drive_calls(self):
        return self.drive_log.read_text().splitlines() if self.drive_log.exists() else []

    def wait_calls(self, n: int, timeout=15.0) -> list:
        """The detached fake drive has written its n-th call line (a real child: give it the moment it needs)."""
        end = time.time() + timeout
        while time.time() < end and len(self.drive_calls()) < n:
            time.sleep(0.05)
        calls = self.drive_calls(); assert len(calls) >= n, calls
        return calls

    def wait_exit(self, att: dict, timeout=15.0) -> None:
        """The detached drive of this attempt has left its rc (the spawn wrapper writes it when the drive exits) and its process is gone
        or a zombie (this test process is its parent and reaps it only through the daemon's poll)."""
        end = time.time() + timeout
        while time.time() < end:
            if os.path.isfile(att["rc_file"]) and not GD._proc_alive(att["pid"], att["proc_start"]):
                return
            time.sleep(0.05)
        raise AssertionError(f"attempt {att['n']} (pid {att['pid']}) did not exit: {att}")

    def write_wave(self, job_id="11111", cluster="cluster_a", sha=None, ledger_sha=None, bare=False, attempt_id=None) -> None:
        """06's two files: WAVE.<view>.json at `sha` and one more `submitted` ledger row for the petri unit (the LAST row wins) carrying
        submit_twin's `extra.sha` — the sha the proxy was SUBMITTED at (`ledger_sha`, default the wave's: a proxy minted with its wave);
        `bare` writes a row WITHOUT it (an older ledger's row); `attempt_id` names the row's attempt (the attempts ledger's key, `attempt`)."""
        (self.state_root / f"WAVE.{VIEW}.json").write_text(json.dumps({"model": MODEL, "view": VIEW, "wave_sha": sha or self.sha, "clusters": "cluster_a,cluster_b", "n_units": 3}))
        row = {"event": "submitted", "cluster": cluster, "job_id": job_id, "job_name": f"f22-petri-{TAG}", "unit": "petri.proxy",
               "logical_work_id": f"{MODEL}/{VIEW}/petri/proxy", "model": MODEL}
        if attempt_id:
            row["attempt_id"] = attempt_id
        if not bare:
            row["extra"] = {"kind": "petri", "group": None, "sha": ledger_sha or sha or self.sha}
        with open(self.state_root / "submissions.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def attempt(self, attempt_id: str, job_id: str, sha: str, lwid=f"{MODEL}/{VIEW}/petri/proxy") -> None:
        """The funnel's attempts ledger (`<state root>/attempts.jsonl`, submit_twin.record_attempt): the `attempt` event with the FULL spec it launched."""
        with open(self.state_root / "attempts.jsonl", "a") as fh:
            fh.write(json.dumps({"event": "attempt", "logical_work_id": lwid, "attempt_id": attempt_id, "cluster": "cluster_a", "job_id": job_id, "unit": "petri.proxy",
                                 "sha": sha, "spec": {"unit": "petri.proxy", "lwid": lwid, "sha": sha}}) + "\n")

    def main_argv(self, *extra: str):
        # main() preflights the registry's SCHEMA and holds every Petri start while it is not the code's, so the registry handed to
        # main() is a REAL one at the code's version (db.open_fresh: every migration applied), not make_registry's run-table stand-in
        reg = self.tmp / "reg"; reg.mkdir(exist_ok=True)
        if not (reg / DB.DB_NAME).exists():
            DB.open_fresh(str(reg / DB.DB_NAME)).close()
        cfg = self.tmp / "configs"; cfg.mkdir(exist_ok=True)
        return ["--registry-root", str(reg), "--model-id", "m", "--state-dir", str(self.state), "--repo", str(self.repo), "--configs-root", str(cfg),
                "--python", sys.executable, "--petri-python", self.python, "--no-contrasts", "--ssh-bin", str(self.bin / "ssh"),
                "--petri-state", str(self.state_root), "--petri-model", MODEL, "--petri-cluster", f"cluster_a=cluster-a-login={self.remote}={self.shared}",
                "--petri-bundle-dir", str(self.bundle), "--petri-bank-root", str(self.bank_root), "--petri-interval", "0.2", "--petri-drive-env-allow", "FAKE_*", *extra]


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    yield w
    w.hold.unlink(missing_ok=True)      # release any held fake drive so nothing outlives the test


# ------------------------------------------------------------------------------------------------------------ discovery
def test_discovery_every_pass_finds_a_wave_written_after_the_daemon_started(world):
    clusters = GD.parse_petri_clusters([f"cluster_a=cluster-a-login={world.remote}={world.shared}"])
    assert clusters == {"cluster_a": {"ssh": "cluster-a-login", "wave_root": str(world.remote), "shared": str(world.shared)}}
    assert GD.parse_petri_clusters([f"cluster_a=cluster-a-login={world.remote}"])["cluster_a"]["shared"] == str(world.shared)   # 3 parts: $SHARED = before /exp/
    # before 06 ran: nothing to watch, no event
    assert GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state)) == [] and world.events() == []
    # a wave without a petri unit yet: one event, no spec; a partial marker is never a wave
    (world.state_root / f"WAVE.{VIEW}.json").write_text(json.dumps({"model": MODEL, "view": VIEW, "wave_sha": world.sha}))
    (world.state_root / "WAVE.s2.partial.json").write_text(json.dumps({"model": MODEL, "view": "s2", "wave_sha": world.sha}))
    assert GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state)) == []
    assert [e["tag"] for e in world.events("petri_drive_no_unit")] == [TAG]
    # 06 places the petri unit: the spec appears on the next read, with the ledger's job id; a re-run's later row wins
    world.write_wave(job_id="11111")
    (specs,) = GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))
    assert specs == {"tag": TAG, "model": MODEL, "view": VIEW, "cluster": "cluster_a", "ssh": "cluster-a-login", "sha": world.sha, "job_id": "11111",
                     "job_name": f"f22-petri-{TAG}", "endpoint": str(world.endpoint), "shared": str(world.shared)}
    world.write_wave(job_id="22222")
    assert GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))[0]["job_id"] == "22222"
    # a proxy on a cluster the daemon was not given: said once, no spec
    world.write_wave(job_id="33333", cluster="cluster_zz")
    assert GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state)) == []
    GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))
    assert len(world.events("petri_drive_no_cluster")) == 1
    with pytest.raises(ValueError, match="expected <label>=<ssh alias>=<wave root>"):
        GD.parse_petri_clusters(["cluster_a=host"])


def test_a_daemon_started_before_the_wave_starts_the_drive_once_06_writes_it(world, monkeypatch):
    """The documented order: 08 --daemon first, then 06 places the wave. The daemon's Petri thread finds WAVE.u0.json on a later pass,
    finds the endpoint up and the proxy RUNNING, and starts the drive — with no restart and no startup-time spec list."""
    monkeypatch.setattr(GD, "TICK_INTERVAL_FLOOR_S", 0.2); monkeypatch.setattr(GD, "PETRI_INTERVAL_FLOOR_S", 0.05)
    calls = []

    def fake_tick(args, *a, **k):
        calls.append(time.time())
        if len(calls) == 1:                      # 06 runs while the daemon is already up
            world.write_wave(job_id="11111"); world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
        else:                                    # give the Petri thread its passes before the loop ends
            end = time.time() + 10
            while time.time() < end and not world.drive_calls():
                time.sleep(0.05)
        return {}
    monkeypatch.setattr(GD, "tick", fake_tick)
    assert not (world.state_root / "WAVE.u0.json").exists()
    assert GD.main(world.main_argv("--max-ticks", "2", "--interval", "0.2")) == 0
    started = world.events("petri_drive_started")
    assert len(started) == 1 and started[0]["tag"] == TAG and started[0]["wave_sha"] == world.sha
    assert len(world.drive_calls()) == 1


# ------------------------------------------------------------------------------------------- the checkout at the wave sha
def test_the_drive_runs_from_a_persistent_checkout_at_the_wave_sha_with_the_bank_view_built_there(world):
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    s = world.run_pass()
    assert s["started"] == 1, s
    checkout = world.state / "checkouts" / world.sha
    rec = world.record(); att = rec["attempts"][0]
    assert att["state"] == "started" and att["checkout"] == str(checkout) and isinstance(att["pid"], int) and isinstance(att["proc_start"], int)
    assert att["argv"] == ["bash", str(checkout / GD.PETRI_DRIVE_SCRIPT_DEFAULT), MODEL, VIEW, "cluster_a", world.sha]
    assert att["identity"] == {"job_id": "11111", "endpoint_mtime": "1700000000", "wave_sha": world.sha}
    # the checkout IS the launched commit while --repo's HEAD has moved on; a detached worktree of --repo's repository
    assert _git(checkout, "rev-parse", "HEAD") == world.sha and _git(world.repo, "rev-parse", "HEAD") == world.tip
    assert not (checkout / "scripts/sdfbase/full22/later.txt").exists()
    assert sum(1 for ln in _git(world.repo, "worktree", "list").splitlines() if str(checkout) in ln) == 1
    # its .venv is a directory of links into the venv that owns the fit interpreter, and runs (the drive resolves $WT/.venv/bin/python on older commits)
    assert (checkout / ".venv").is_dir() and not (checkout / ".venv").is_symlink()
    assert (checkout / ".venv/bin").is_symlink() and Path(os.readlink(checkout / ".venv/bin")) == world.venv / "bin" and (checkout / ".venv/pyvenv.cfg").is_symlink()
    assert subprocess.run([str(checkout / ".venv/bin/python"), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, check=True).stdout.strip() == str(checkout / ".venv")
    assert _git(checkout, "status", "--porcelain") == ""                        # the link is gitignored: the drive's dirty-tree check holds
    # the drive ran FROM the checkout, told so, with the checkout's src first
    world.wait_exit(att)
    (call,) = world.drive_calls()
    # ... and READS the bank view under the root this daemon built it under (F22_GRADING_HOST_BANK_ROOT = --petri-bank-root; demo-import 2026-10-11:
    # the drive's own literal was the paper's legacy the grading host bank, a third place the front door's staged objects never reached)
    assert call == (f"secrets=op wt={checkout} py={world.python} pythonpath={checkout}/src cwd={checkout} bank={world.bank_root} cells=unset "
                    f"nargs=4 args={MODEL} {VIEW} cluster_a {world.sha}")
    assert "F22_GRADING_HOST_BANK_ROOT" in att["env_keys"]
    assert "drive running for" in (world.state / f"petri_drive.{TAG}.1.log").read_text() and (world.state / f"petri_drive.{TAG}.1.log.rc").read_text().strip() == "0"
    # the grading host bank view was BUILT at the checkout (absent before), then verified; a later pass verifies without rebuilding
    view = world.bank_root / MODEL / "cells_view" / VIEW
    assert (view / "VIEW_MANIFEST.json").exists()
    mbv = world.mbv_log.read_text().splitlines()
    assert mbv == [f"build {view} roster={checkout / 'configs/sdfbase_full22_roster.yaml'}", f"check {view} roster={checkout / 'configs/sdfbase_full22_roster.yaml'}"]
    (prepared,) = world.events("petri_drive_prepared")
    assert prepared["checkout"] == str(checkout) and prepared["note"] == "bank view built" and prepared["python"] == world.python
    # the checkout persists and is REUSED (the same path, one worktree), across passes and across a restart of the preparation cache
    GD._PETRI_PREPARED.clear()
    assert GD.petri_checkout(str(world.repo), world.sha, str(world.state), python=world.python) == (str(checkout), None)
    assert GD.petri_prepare(world.spec, checkout=str(checkout), python=world.python, bank_root=str(world.bank_root), bundle_dir=str(world.bundle), roster=GD.PETRI_ROSTER_DEFAULT) == (None, "")
    assert sum(1 for ln in _git(world.repo, "worktree", "list").splitlines() if str(checkout) in ln) == 1
    assert world.mbv_log.read_text().splitlines()[-1].startswith("check ")
    # a checkout dir at the WRONG sha is refused by name, never used
    wrong = world.state / "checkouts" / ("f" * 40); wrong.mkdir()
    subprocess.run(["git", "-C", str(world.repo), "worktree", "add", "-q", "--detach", str(wrong), world.tip], check=True, capture_output=True)
    path, why = GD.petri_checkout(str(world.repo), "f" * 40, str(world.state))
    assert path is None and "not the wave sha ffffffffffff" in why


def test_an_unfit_interpreter_or_a_missing_bundle_is_a_reason_not_an_attempt(world):
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    bad_py = _exe(world.tmp / "badpy", "#!/bin/bash\necho 'ModuleNotFoundError: No module named inspect_ai' >&2; exit 1\n")
    s = world.run_pass(python=str(bad_py))
    assert s["unprepared"] == 1 and s["started"] == 0 and not world.drive_calls()
    (ev,) = world.events("petri_drive_unprepared")
    assert "lacks what the drive imports" in ev["reason"] and not (world.state / f"petri_drive.{TAG}.json").exists()
    s = world.run_pass(bundle_dir=str(world.tmp / "no-bundle"))
    assert s["unprepared"] == 1 and "no Petri scenario bundle" in world.events("petri_drive_unprepared")[-1]["reason"]
    # eager preparation while the endpoint is ABSENT: the checkout and the bank view are ready before the proxy is
    world.endpoint.unlink()
    s = world.run_pass()
    assert s["waiting"] == 1 and (world.state / "checkouts" / world.sha / ".git").exists() and (world.bank_root / MODEL / "cells_view" / VIEW / "VIEW_MANIFEST.json").exists()
    assert [e["event"] for e in world.events()][-2:] == ["petri_drive_waiting", "petri_drive_prepared"]


def test_an_unprepared_or_waiting_wave_is_said_again_at_a_new_wave_sha(world):
    """demo-import (2026-10-11): the once-keys of `petri_drive_unprepared`, `petri_drive_waiting` and `petri_drive_prepared` carry the WAVE
    SHA. A wave re-launched at a new sha (06 --resubmit rewrites WAVE.<view>.json) that fails to prepare for the SAME reason is a new fact
    and logs again; before this the key was (tag, reason) alone, and the daemon's log stayed silent through the re-launch — the demo's smoke
    read as a stuck discovery when it was the same unprepared bank at a new sha."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    no_bundle = str(world.tmp / "no-bundle")
    s = world.run_pass(bundle_dir=no_bundle)
    assert s["unprepared"] == 1 and [e["wave_sha"] for e in world.events("petri_drive_unprepared")] == [world.sha]
    s = world.run_pass(bundle_dir=no_bundle)                                        # the same wave again: said once
    assert s["unprepared"] == 1 and len(world.events("petri_drive_unprepared")) == 1
    s = world.run_pass(spec=dict(world.spec, sha=world.tip), bundle_dir=no_bundle)   # the wave re-launched at the tip, the SAME reason: said again
    assert s["unprepared"] == 1
    ev = world.events("petri_drive_unprepared")
    assert [e["wave_sha"] for e in ev] == [world.sha, world.tip] and all("no Petri scenario bundle" in e["reason"] for e in ev), ev
    # the waiting event the same way: an absent endpoint is said once PER SHA, and the eager preparation that succeeds is recorded per sha
    world.endpoint.unlink()
    for sha in (world.sha, world.sha, world.tip, world.tip):
        assert world.run_pass(spec=dict(world.spec, sha=sha))["waiting"] == 1
    assert [e["wave_sha"] for e in world.events("petri_drive_waiting")] == [world.sha, world.tip]
    assert [e["wave_sha"] for e in world.events("petri_drive_prepared")] == [world.sha, world.tip]
    assert {e["checkout"] for e in world.events("petri_drive_prepared")} == {str(world.state / "checkouts" / world.sha), str(world.state / "checkouts" / world.tip)}


def test_discovery_follows_the_units_submitted_sha_never_the_wave_files_and_refuses_a_unit_whose_sha_is_unknown(world):
    """petri-completion: the drive checks out the sha the UNIT's bound submission was made at — the petri
    unit's last `submitted` row's `extra.sha` (submit_twin's record of the sha the proxy was SUBMITTED at), cross-checked against the funnel's
    attempts ledger — never WAVE.<view>.json's wave_sha, which a re-minted proxy (`--resubmit` / `--units petri.proxy` leave WAVE unchanged) or
    the overflow pass (06 --pass overflow never rewrites WAVE) leaves behind: on the smoke the battery WAVE pinned a sha whose drive had neither
    `legs=ledgered` nor `petri_run.py legs-status`. A wave file at another sha is said once per pair (`petri_drive_sha_mismatch`, informational;
    the spec follows the ROW); a row whose sha cannot be established — no extra.sha and no `attempt` event, or the two disagreeing — is REFUSED by
    name (`petri_drive_sha_unresolved`, once per (unit, job, reason)) and yields no spec: never a silent fall-back to the wave file."""
    clusters = GD.parse_petri_clusters([f"cluster_a=cluster-a-login={world.remote}={world.shared}"])
    disc = lambda: GD.discover_petri_specs(str(world.state_root), MODEL, clusters, log_state_dir=str(world.state))   # noqa: E731
    world.write_wave(ledger_sha=world.sha)                                            # submitted at the wave's sha: nothing to say
    assert [s["sha"] for s in disc()] == [world.sha] and not world.events("petri_drive_sha_mismatch")
    world.write_wave(sha=world.tip, ledger_sha=world.sha)                             # the wave re-written at the tip; the proxy's row still names the launch sha
    for _ in range(2):
        assert [s["sha"] for s in disc()] == [world.sha]                              # the spec follows the unit's row, not the wave file
    (ev,) = world.events("petri_drive_sha_mismatch")
    assert ev["tag"] == TAG and ev["wave_sha"] == world.tip and ev["ledger_sha"] == world.sha and ev["job_id"] == "11111" and "the drive follows the unit's sha" in ev["note"]
    world.write_wave(job_id="22222", sha=world.sha, ledger_sha=world.tip)            # the proxy RE-MINTED at the tip (--units petri.proxy): WAVE stays at the launch sha
    assert [(s["sha"], s["job_id"]) for s in disc()] == [(world.tip, "22222")] and len(world.events("petri_drive_sha_mismatch")) == 2
    # an older ledger's row (no extra.sha) and no attempt event: refused by name, no spec, said once; never the wave file's sha
    world.write_wave(job_id="33333", bare=True)
    for _ in range(2):
        assert disc() == []
    (ref,) = world.events("petri_drive_sha_unresolved")
    assert ref["tag"] == TAG and ref["job_id"] == "33333" and ref["wave_sha"] == world.sha and "neither the submissions row (extra.sha) nor the attempts ledger" in ref["reason"]
    # the attempts ledger names the attempt's sha: resolved from there
    world.write_wave(job_id="44444", bare=True, attempt_id="a4"); world.attempt("a4", "44444", world.tip)
    (s4,) = disc(); assert (s4["sha"], s4["job_id"]) == (world.tip, "44444")
    # the row and the ledger disagree: refused by name (which tree the proxy runs is not established)
    world.write_wave(job_id="55555", ledger_sha=world.sha, attempt_id="a5"); world.attempt("a5", "55555", world.tip)
    assert disc() == [] and "disagree" in world.events("petri_drive_sha_unresolved")[-1]["reason"] and len(world.events("petri_drive_sha_unresolved")) == 2
    # the pure rule
    assert GD.petri_unit_sha({"extra": {"sha": "abc"}, "attempt_id": "x", "job_id": "1"}, []) == ("abc", "the submissions row's extra.sha")
    assert GD.petri_unit_sha({"extra": {"sha": "abc"}, "attempt_id": "x"}, [{"event": "attempt", "attempt_id": "x", "sha": "abc"}])[1].endswith("(= the attempts ledger's)")
    assert GD.petri_unit_sha({"attempt_id": "x", "job_id": "1"}, [{"event": "attempt", "attempt_id": "x", "spec": {"sha": "def"}}]) == ("def", "the attempts ledger's attempt x")
    assert GD.petri_unit_sha({"job_id": "1", "logical_work_id": "m/v/petri/proxy"}, [{"event": "attempt", "job_id": "1", "logical_work_id": "m/v/petri/proxy", "sha": "ghi"}])[0] == "ghi"
    assert GD.petri_unit_sha({"job_id": "1", "logical_work_id": "m/v/petri/proxy"}, [{"event": "attempt", "job_id": "1", "logical_work_id": "m/v/petri/overflow", "sha": "ghi"}])[0] is None
    assert GD.petri_unit_sha({}, [])[0] is None and GD.petri_unit_sha({"extra": {"sha": "a"}, "attempt_id": "x"}, [{"event": "attempt", "attempt_id": "x", "sha": "b"}])[0] is None


# ------------------------------------------------------- liveness, outcomes, the retry cap, the endpoint identity
def test_a_stale_endpoint_never_starts_and_outcomes_are_recorded_with_the_retry_cap_per_endpoint(world):
    # 1. the file exists but the job that wrote it is gone (the file a hard death leaves behind): stale, said once, nothing started
    world.endpoint_up("11111", 1_700_000_000); world.job_gone("11111")
    assert world.run_pass()["stale"] == 1 and world.run_pass()["stale"] == 1
    (stale,) = world.events("petri_drive_stale_endpoint")
    assert stale["job_id"] == "11111" and stale["job_state"] == "gone" and not world.drive_calls() and not (world.state / f"petri_drive.{TAG}.json").exists()
    # a job in the queue but not serving (COMPLETING) is stale too; a squeue that cannot answer is neither a start nor a stale verdict
    world.job("11111", "COMPLETING"); assert world.run_pass()["stale"] == 1
    (world.squeue_dir / "BROKEN").write_text(""); assert world.run_pass()["unreachable"] == 1 and world.events("petri_drive_liveness_unknown")
    (world.squeue_dir / "BROKEN").unlink()
    # 2. RUNNING: started; the drive fails (exit 1) -> recorded `failed` with its log, retried on the next pass while the proxy lives
    world.job("11111", "RUNNING"); world.rc.write_text("1")
    assert world.run_pass()["started"] == 1
    world.wait_exit(world.record()["attempts"][0])
    s = world.run_pass()                                    # settles attempt 1, starts attempt 2 in the same pass
    assert s["started"] == 1, s
    ex = world.events("petri_drive_exited")
    assert len(ex) == 1 and ex[0]["state"] == "failed" and ex[0]["exit_code"] == 1 and ex[0]["attempt"] == 1 and ex[0]["log"] == str(world.state / f"petri_drive.{TAG}.1.log")
    world.wait_exit(world.record()["attempts"][1]); assert world.run_pass()["started"] == 1
    world.wait_exit(world.record()["attempts"][2])
    s = world.run_pass()                                    # the third failure caps this endpoint identity
    assert s == dict(s, started=0, capped=1) and len(world.events("petri_drive_gave_up")) == 1
    assert world.run_pass()["capped"] == 1 and len(world.events("petri_drive_gave_up")) == 1     # said once
    rec = world.record()
    assert [a["state"] for a in rec["attempts"]] == ["failed"] * 3 and len(world.drive_calls()) == 3
    assert world.events("petri_drive_gave_up")[0]["logs"] == [str(world.state / f"petri_drive.{TAG}.{n}.log") for n in (1, 2, 3)]
    # 3. a REPLACED proxy (new job id, new mtime) is a new endpoint identity: a fresh count; this drive completes
    world.rc.write_text("0"); world.endpoint_up("22222", 1_700_000_500); world.job("22222", "RUNNING")
    assert world.run_pass()["started"] == 1
    att = world.record()["attempts"][3]; assert att["identity"] == {"job_id": "22222", "endpoint_mtime": "1700000500", "wave_sha": world.sha}
    world.wait_exit(att)
    s = world.run_pass()
    assert s["completed"] == 1 and s["started"] == 0 and world.events("petri_drive_exited")[-1]["state"] == "completed"
    assert world.run_pass()["completed"] == 1 and len(world.drive_calls()) == 4              # never again for this endpoint
    assert world.record()["attempts"][3]["exit_code"] == 0
    # 4. a daemon RESTART settles a drive it did not spawn: the wrapper's rc file, the pid + start time (no Popen to poll)
    world.endpoint_up("22222", 1_700_000_900); world.hold.write_text("")
    assert world.run_pass()["started"] == 1
    GD._PETRI_PROCS.clear()                                  # as after a restart: the Popen is gone
    assert world.run_pass()["running"] == 1                  # alive in /proc with the recorded start time: still running, no second start
    world.hold.unlink(); world.wait_exit(world.record()["attempts"][4])
    s = world.run_pass()
    assert s["completed"] == 1 and world.record()["attempts"][4]["state"] == "completed" and world.record()["attempts"][4]["exit_code"] == 0
    assert len(world.drive_calls()) == 5
    # the ledger's job id is a cross-check, not the arbiter: the endpoint's own slurm_job_id is what is held to RUNNING
    started = world.events("petri_drive_started")
    assert started[-1]["identity"]["job_id"] == "22222" and started[-1]["ledger_job_id"] == "11111" and "endpoint.json names 22222" in started[-1]["note"]


def test_a_drive_exiting_0_is_completed_only_when_its_summary_verifies_every_planned_run(world):
    """Re-check 1: rc 0 earns `completed` only through the drive's result summary — verdict `complete` and every planned run verified filed;
    a summary short of that, or no summary at all, is `failed` (retried under the cap); the drive's exit 3 (legs pending on overflow
    ledgers) is `incomplete`: not a success, not retried, and a later pass starts nothing for that endpoint."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    world.drive_result("11111", filed=20)                                        # the drive's own count says 20 of its 21
    assert world.run_pass()["started"] == 1; world.wait_exit(world.record()["attempts"][0])
    world.result_ctl.write_text("none")
    assert world.run_pass()["started"] == 1                                     # attempt 1 settled failed, attempt 2 started
    ex = world.events("petri_drive_exited")
    assert ex[0]["exit_code"] == 0 and ex[0]["state"] == "failed" and "only 20 of 21 planned run(s) verified filed" in ex[0]["reason"]
    assert world.record()["attempts"][0]["result"]["handoff"]["runs_filed"] == 20
    world.wait_exit(world.record()["attempts"][1])
    world.drive_result("11111"); world.rc.write_text("3")
    assert world.run_pass()["started"] == 1                                     # attempt 2 (no summary) failed, attempt 3 started
    assert world.events("petri_drive_exited")[1]["state"] == "failed" and "no result summary" in world.events("petri_drive_exited")[1]["reason"]
    world.wait_exit(world.record()["attempts"][2])
    s = world.run_pass()
    assert s["incomplete"] == 1 and s["started"] == 0
    ex3 = world.events("petri_drive_exited")[2]
    assert ex3["exit_code"] == 3 and ex3["state"] == "incomplete" and "overflow" in ex3["reason"]
    assert world.run_pass()["incomplete"] == 1 and len(world.drive_calls()) == 3          # never retried, never completed
    world.rc.write_text("0"); world.endpoint_up("22222", 1_700_000_600); world.job("22222", "RUNNING")
    assert world.run_pass()["started"] == 1; world.wait_exit(world.record()["attempts"][3])
    assert world.run_pass()["completed"] == 1 and world.record()["attempts"][3]["result"]["handoff"]["runs_filed"] == 21


def test_the_drives_verdict_is_its_exit_code_and_verify_reads_the_filed_runs(tmp_path, monkeypatch):
    """Re-check 1, the drive's side, executed: `f22_drive_verdict` (extracted from the real drive) maps (failed legs, overflow-pending legs,
    finish rc, verify rc) to 1 / 3 / 0; `petri_registry_handoff.py verify` over a REAL identity (the proxy job's registration over the real
    seeds) and a real spool refuses until every planned run in scope holds `generated` + a store_uri, honours the standup's skipped objects
    and a selected-objects scope, and writes the summary the drive's verdict reads."""
    drive = (REPO / "scripts/sdfbase/full22/petri_drive_full22.sh").read_text()
    fn = drive[drive.index("f22_drive_verdict(){"):]; fn = fn[:fn.index("\n") + 1]
    def verdict(nfail, ninc, hrc, vrc):
        return subprocess.run(["bash", "-c", fn + f'f22_drive_verdict "$@"', "_", str(nfail), str(ninc), str(hrc), vrc], capture_output=True, text=True, check=True).stdout.strip()
    assert [verdict(0, 0, 0, "0"), verdict(0, 0, 0, ""), verdict(1, 0, 0, "0"), verdict(0, 2, 0, ""), verdict(0, 2, 1, ""), verdict(0, 0, 1, "0"), verdict(0, 0, 0, "2")] == ["0", "0", "1", "3", "1", "1", "1"]
    assert "exit \"$DRIVE_RC\"" in drive and 'finish --identity "$REG_IDENTITY") 2>&1 | tee -a "$LOG"; HANDOFF_RC=${PIPESTATUS[0]}' in drive
    # the verify step over a real registration + spool
    import test_petri_registry_handoff as H
    from sisteer.registry import emit, hooks
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV, "SLURM_JOB_ID", "SLURM_JOB_NAME"):
        monkeypatch.delenv(k, raising=False)                                   # what that module's _clean_env fixture does
    hooks._DEFAULT_NOTICED.clear()
    reg = H._register(tmp_path); spool = tmp_path / "grading_spool"
    ident = H._import(tmp_path, reg, spool)
    runs = json.loads(ident.read_text())["runs"]; assert len(runs) == 21
    def verify(*args):
        return H._handoff("--spool", str(spool), "verify", "--identity", str(ident), "--summary", str(tmp_path / "verify.json"), *args)
    r = verify(); summ = json.loads((tmp_path / "verify.json").read_text())
    assert r.returncode == 2 and "21 of 21 planned run(s) are NOT filed" in r.stderr and summ["n_planned"] == 21 and summ["n_filed"] == 0
    none_runs = [x for x in runs if x["route"]["object"] == "none"]; assert len(none_runs) == 7
    producer = emit.producer_block("test", code_sha=None)
    for x in none_runs:
        w = emit.RecordWriter(str(spool), x["plan"]["campaign"], x["run_id"], producer)
        w.emit_patch(fields={"store_uri": str(tmp_path / "cells" / "none")}); w.emit_state("generated", "completion_artifact", str(tmp_path / "cells/none/results.json"))
    r = verify(); summ = json.loads((tmp_path / "verify.json").read_text())
    assert r.returncode == 2 and summ["n_filed"] == 7 and summ["n_unfiled"] == 14
    r = verify("--objects", "none"); summ = json.loads((tmp_path / "verify.json").read_text())
    assert r.returncode == 0 and summ["n_planned"] == 7 and summ["n_filed"] == 7 and summ["n_out_of_scope"] == 14 and "7 of 7 planned run(s) filed" in r.stdout
    r = verify("--skipped", "prompted__safety_refusal,prompted__si_constitution"); summ = json.loads((tmp_path / "verify.json").read_text())
    assert r.returncode == 0 and summ["n_skipped"] == 14 and summ["n_planned"] == 7
    assert all(row["filed"] for row in summ["runs"] if row["scope"] == "in") and all(row["store_uri"] for row in summ["runs"] if row["scope"] == "in")
    r = verify("--objects", "no_such_object"); assert r.returncode == 2 and "no planned run is in scope" in r.stderr


def test_a_daemon_dying_between_spawn_and_record_never_yields_a_second_drive(world):
    """Re-check 2, the exact window with a REAL child: the daemon claims, spawns, and dies before saving `started` (the record says `claimed`,
    pid None). The child inherited the tag lock's fd and released it only after publishing its pid marker, so the next pass — a fresh
    daemon, no Popen to poll — finds the marker under the lock, ADOPTS the attempt as `started`, sees it alive, and starts nothing; when
    it exits, the rc file settles it. One drive, ever."""
    class Died(BaseException):
        pass
    real_popen = subprocess.Popen
    def popen_then_die(*a, **k):
        real_popen(*a, **k)
        raise Died()
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING"); world.hold.write_text("")
    with pytest.raises(Died):
        world.run_pass(popen=popen_then_die)
    rec = world.record(); att = rec["attempts"][0]
    assert att["state"] == "claimed" and att["pid"] is None
    # the lock was held by the child until its marker existed: once we can take it, the marker is there
    import fcntl
    fd = os.open(str(world.state / f"petri_drive.{TAG}.lock"), os.O_RDWR | os.O_CREAT)
    end = time.time() + 10
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
        except OSError:
            assert time.time() < end, "the tag lock never became free"
            time.sleep(0.01)
    assert os.path.isfile(att["pid_file"]), "the lock was released before the wrapper published its pid"
    fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)
    marker = json.loads(Path(att["pid_file"]).read_text()); assert isinstance(marker["pid"], int) and GD._proc_alive(marker["pid"], marker["proc_start"])
    GD._PETRI_PROCS.clear()                                                     # the spawning daemon is gone
    s = world.run_pass()
    assert s["running"] == 1 and s["started"] == 0
    att = world.record()["attempts"][0]
    assert att["state"] == "started" and att["pid"] == marker["pid"] and att["proc_start"] == marker["proc_start"] and att["adopted"] is True
    (adopted,) = world.events("petri_drive_adopted"); assert adopted["pid"] == marker["pid"]
    assert len(world.wait_calls(1)) == 1
    world.hold.unlink(); world.wait_exit(att)
    s = world.run_pass()
    assert s["completed"] == 1 and world.record()["attempts"][0]["exit_code"] == 0 and len(world.drive_calls()) == 1
    assert [e["event"] for e in world.events() if e["event"].startswith("petri_drive_") and e["event"] not in ("petri_drive_prepared", "petri_drive_expected", "petri_drive_key_attested")] == \
        ["petri_drive_adopted", "petri_drive_exited"]                          # no `started`: the dying daemon never recorded it; no second start


def test_two_processes_overlapping_a_settlement_and_a_start_produce_one_drive(world):
    """Re-check 3: two daemon PROCESSES pass over the same tag at the same moment when an exited attempt needs settling AND a new endpoint
    identity is due a start. Every read / reconcile / save is under the tag lock, so one process does both (one `exited`, one
    `started`) and the other skips the pass (`locked`); the record is consistent and exactly one drive runs."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING"); world.hold.write_text("")
    assert world.run_pass()["started"] == 1
    world.hold.unlink(); world.wait_exit(world.record()["attempts"][0])           # exited, NOT yet settled
    world.endpoint_up("22222", 1_700_000_500); world.job("22222", "RUNNING"); world.hold.write_text("")
    runner = world.tmp / "runner.py"; runner.write_text(RUNNER)
    cfg = world.tmp / "cfg.json"; cfg.write_text(json.dumps({"spec": world.spec, "kw": world.kw}))
    go = world.tmp / "go"; n0 = len(world.events())                           # the setup's own events end here
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    procs = [subprocess.Popen([sys.executable, str(runner), str(REPO / "scripts" / "registry"), str(cfg), str(go)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for _ in range(2)]
    time.sleep(1.0); go.write_text("")                                          # both are spinning on the go file: release them together
    outs = [p.communicate(timeout=120) for p in procs]
    summaries = [json.loads(o.splitlines()[-1]) for o, _e in outs]
    assert sum(x["started"] for x in summaries) == 1 and sum(x["locked"] + x["running"] for x in summaries) == 1, (summaries, [e for _o, e in outs])
    rec = world.record()
    assert [a["state"] for a in rec["attempts"]] == ["completed", "started"] and rec["attempts"][1]["identity"]["job_id"] == "22222"
    ev = [e["event"] for e in world.events()[n0:] if e["event"] in ("petri_drive_exited", "petri_drive_started")]
    assert sorted(ev) == ["petri_drive_exited", "petri_drive_started"], ev          # one settlement, one start, across both processes
    assert len(world.wait_calls(2)) == 2
    world.hold.unlink(); world.wait_exit(rec["attempts"][1])
    assert world.run_pass()["completed"] == 1 and len(world.drive_calls()) == 2


def test_a_partial_drive_is_an_error_counted_toward_the_cap_and_gave_up_names_the_uncovered_runs(world):
    """The endpoint's EXPECTED run set comes from the proxy job's own registration (minus the cells the standup skipped),
    never from a drive's summary; the daemon's drive is the FULL required scope, so a drive that filed less is an ERROR — `partial:<covered>/
    <required>` with its coverage recorded truthfully, counted toward the cap like a failure (the identical drive is not relaunched past it),
    and `petri_drive_gave_up` names the uncovered runs. A proxy that registered nothing is never claimed."""
    world.endpoint_up("11111", 1_700_000_000, skipped=("prompted__b",)); world.job("11111", "RUNNING")
    world.drive_result("11111", filed_objects=("none",), out_objects=("prompted__a",), skipped_objects=("prompted__b",), stop=False)   # the drive fell short: 7 of 14
    assert world.run_pass()["started"] == 1
    rec = world.record(); key = GD._identity_key(rec["attempts"][0]["identity"]); ep = rec["endpoints"][key]
    assert len(ep["expected_ids"]) == 21 and ep["skipped_objects"] == ["prompted__b"] and len(ep["skipped_runs"]) == 7 and ep["complete"] is False
    (exp_ev,) = world.events("petri_drive_expected"); assert exp_ev["n_expected"] == 21 and exp_ev["registration"].endswith("/registry/pending/11111.json")
    world.wait_exit(rec["attempts"][0])
    s = world.run_pass()                                                         # attempt 1 settles PARTIAL (an error); under the cap: attempt 2
    assert s["started"] == 1 and s["completed"] == 0, s
    rec = world.record(); a1 = rec["attempts"][0]; ep = rec["endpoints"][key]
    assert a1["state"] == "partial" and a1["reason"].startswith("partial:7/14") and "ERROR" in a1["reason"]
    assert a1["coverage"] == {"filed_by_this_drive": 7, "covered": 7, "required": 14, "expected": 21, "skipped": 7, "remaining": 7}
    assert sorted(ep["covered"]) == sorted(world.run_ids("11111")["none"]) and ep["complete"] is False and sorted(ep["remaining"]) == sorted(world.run_ids("11111")["prompted__a"])
    ex1 = world.events("petri_drive_exited")[0]; assert ex1["state"] == "partial" and "attempt 1 of 3" in ex1["note"] and "fell short" in ex1["note"]
    world.wait_exit(rec["attempts"][1]); assert world.run_pass()["started"] == 1   # attempt 2 partial -> attempt 3
    world.wait_exit(world.record()["attempts"][2])
    s = world.run_pass()                                                         # the third shortfall caps this endpoint
    assert s["capped"] == 1 and s["started"] == 0 and s["completed"] == 0, s
    (gave_up,) = world.events("petri_drive_gave_up")
    assert gave_up["n_uncovered"] == 7 and sorted(gave_up["uncovered"]) == sorted(world.run_ids("11111")["prompted__a"]) and gave_up["covered"] == 7 and gave_up["required"] == 14
    assert [a["state"] for a in world.record()["attempts"]] == ["partial"] * 3 and len(world.drive_calls()) == 3
    assert world.run_pass()["capped"] == 1 and len(world.events("petri_drive_gave_up")) == 1 and len(world.drive_calls()) == 3    # said once; nothing relaunches
    # a proxy that registered nothing: no claim, the reason said once
    world.endpoint_up("33333", 1_700_001_000); world.job("33333", "RUNNING"); (world.shared / "registry" / "pending" / "33333.json").unlink()
    s = world.run_pass(); assert s["unprepared"] == 1 and s["started"] == 0
    (ev,) = world.events("petri_drive_expected_unknown"); assert "no job-start registration" in ev["reason"]


def test_the_daemons_drive_is_the_full_required_scope_and_a_completed_drive_stops_the_proxy(world, monkeypatch):
    """The REAL selection behaviour: the drive's environment is an allowlist of the daemon's — a selection knob in the daemon's
    environment is dropped (named in the start event), the drive receives the four-argument argv (legs = ALL), and the real drive's own
    selection block evaluated under that environment resolves to the full scope (`RELEASE_SCOPE=all`, no cells) where the raw environment
    would select. The STOP interplay: a completed full-scope drive touches the proxy's STOP, the proxy exits (endpoint.json gone, job gone),
    the endpoint is `completed`, and nothing relaunches."""
    # 1. the allowlist as a function: knobs dropped, the site block / secrets the drive needs kept, unrelated tokens not passed
    base = {"PATH": "/bin", "HOME": "/h", "F22_PETRI_CELLS": "none", "F22_PETRI_OBJECTS_ONLY": "1", "F22_PETRI_RELEASE_SCOPE": "selected", "F22_OVERFLOW_RERUN": "1",
            "F22_SECOND_PASS": "1", "F22_ROSTER": "configs/r.yaml", "SISTEER_OP_ITEM_OPENROUTER": "x", "CLUSTER__cluster_a__CLUSTER_SSH": "h", "OPENROUTER_API_KEY": "k",
            "SSH_AUTH_SOCK": "/s", "OVERRIDE_ID": "o", "GH_TOKEN": "g", "MODAL_TOKEN_ID": "m", "WANDB_API_KEY": "w", "RANDOM_THING": "r", "FAKE_X": "f"}
    env, dropped, overlay_names = GD.petri_drive_env(base, ["FAKE_*"])
    assert overlay_names == []                                                      # no state dir: no overlay
    assert dropped == ["F22_OVERFLOW_RERUN", "F22_PETRI_CELLS", "F22_PETRI_OBJECTS_ONLY", "F22_PETRI_RELEASE_SCOPE", "F22_SECOND_PASS"]
    assert set(env) == {"PATH", "HOME", "F22_ROSTER", "SISTEER_OP_ITEM_OPENROUTER", "CLUSTER__cluster_a__CLUSTER_SSH", "OPENROUTER_API_KEY", "SSH_AUTH_SOCK", "OVERRIDE_ID",
                        "FAKE_X", "SISTEER_SECRETS_MODE"}
    assert GD.petri_drive_env({"F22_PETRI_CELLS": "x"}, ["F22_PETRI_CELLS"])[1] == ["F22_PETRI_CELLS"]      # the deny list wins over any allow
    # 2. the REAL drive's selection block, executed under the raw and the sanitized environment
    drive = (REPO / "scripts/sdfbase/full22/petri_drive_full22.sh").read_text()
    block = drive[drive.index('PETRI_CELLS="${F22_PETRI_CELLS:-}"'):drive.index("# THE TARGET DECODE CONTRACT")]
    script = "set -uo pipefail\nLEGS=ALL\n" + block + '\necho "PETRI_CELLS=[$PETRI_CELLS] RELEASE_SCOPE=$RELEASE_SCOPE"\n'
    raw = {"PATH": os.environ["PATH"], "F22_PETRI_CELLS": "none", "F22_PETRI_RELEASE_SCOPE": "selected"}
    r_raw = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=raw)
    r_san = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=GD.petri_drive_env(raw)[0])
    assert r_raw.returncode == 0 and r_raw.stdout.strip().endswith("PETRI_CELLS=[none] RELEASE_SCOPE=selected"), r_raw.stdout + r_raw.stderr
    assert r_san.returncode == 0 and r_san.stdout.strip().endswith("PETRI_CELLS=[] RELEASE_SCOPE=all"), r_san.stdout + r_san.stderr
    # 3. end to end: the knob sits in the daemon's environment; the drive never sees it and gets the full four-argument argv
    monkeypatch.setenv("F22_PETRI_CELLS", "none"); monkeypatch.setenv("F22_PETRI_RELEASE_SCOPE", "selected")
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    assert world.run_pass()["started"] == 1
    (started,) = world.events("petri_drive_started"); assert started["env_dropped"] == ["F22_PETRI_CELLS", "F22_PETRI_RELEASE_SCOPE"]
    att = world.record()["attempts"][0]; assert att["env_dropped"] == ["F22_PETRI_CELLS", "F22_PETRI_RELEASE_SCOPE"] and "GH_TOKEN" not in att["env_keys"]
    world.wait_exit(att)
    (call,) = world.drive_calls(); assert " cells=unset nargs=4 args=" in call, call
    # 4. the STOP interplay: the completed full-scope drive touched STOP; the proxy exits; the endpoint is completed; nothing relaunches
    assert (world.endpoint.parent / "STOP").exists()
    world.proxy_stops("11111")
    s = world.run_pass()
    assert s["waiting"] == 1 and s["started"] == 0 and s["completed"] == 0, s              # settled completed; endpoint.json is gone
    rec = world.record(); key = GD._identity_key(rec["attempts"][0]["identity"])
    assert rec["attempts"][0]["state"] == "completed" and rec["endpoints"][key]["complete"] is True and rec["endpoints"][key]["remaining"] == []
    assert not (world.state / "checkouts" / world.sha).exists() or True                     # (pruned or not: irrelevant here)
    world.endpoint_up("11111", 1_700_000_000); world.job_gone("11111")                      # a stale file from the same job: stale, not a start
    s = world.run_pass(); assert s["stale"] == 1 and s["started"] == 0 and len(world.drive_calls()) == 1
    world.job("11111", "RUNNING")                                                            # even alive again with the same identity: completed, nothing relaunches
    s = world.run_pass(); assert s["completed"] == 1 and s["started"] == 0 and len(world.drive_calls()) == 1


def test_a_prune_racing_a_start_at_the_same_sha_never_removes_the_running_drives_checkout(world):
    """Re-check 28: the sha looks prunable (a completed attempt, nothing live) while another PROCESS is about to claim a replacement endpoint
    at that sha. The pruner and every claim hold the global claims lock, so the pruner's reading of the live attempts is current at its
    removal: either it prunes before the claim (the claim re-makes the checkout) or it sees the claim and keeps it — the running drive's
    checkout exists either way."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    assert world.run_pass()["started"] == 1; world.wait_exit(world.record()["attempts"][0])
    checkout = world.state / "checkouts" / world.sha
    # settle WITHOUT pruning (prune runs at the end of a pass; here the pass prunes since the endpoint completed): re-make the prunable state
    s = world.run_pass(); assert s["completed"] == 1 and s["pruned"] == 1 and not checkout.exists()
    assert GD.petri_checkout(str(world.repo), world.sha, str(world.state), python=world.python)[0] == str(checkout)   # back, as a sha that LOOKS prunable
    world.endpoint_up("22222", 1_700_000_500); world.job("22222", "RUNNING"); world.hold.write_text("")
    runner = world.tmp / "runner.py"; runner.write_text(RUNNER)
    cfg = world.tmp / "cfg.json"; cfg.write_text(json.dumps({"spec": world.spec, "kw": world.kw}))
    go = world.tmp / "go"; stop = world.tmp / "stop"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    starter = subprocess.Popen([sys.executable, str(runner), str(REPO / "scripts" / "registry"), str(cfg), str(go)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    pruner = subprocess.Popen([sys.executable, str(runner), str(REPO / "scripts" / "registry"), str(cfg), str(go), "prune", str(stop)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    time.sleep(1.0); go.write_text("")
    so, se = starter.communicate(timeout=120); stop.write_text(""); po, pe = pruner.communicate(timeout=60)
    started = json.loads(so.splitlines()[-1]); pruned = json.loads(po.splitlines()[-1])
    assert started["started"] == 1, (so, se)
    att = world.record()["attempts"][-1]
    assert att["state"] == "started" and att["checkout"] == str(checkout) and GD._proc_alive(att["pid"], att["proc_start"])
    assert checkout.is_dir() and _git(checkout, "rev-parse", "HEAD") == world.sha, pruned          # the running drive's tree is intact
    assert os.path.isdir(f"/proc/{att['pid']}/cwd") and os.readlink(f"/proc/{att['pid']}/cwd") == str(checkout)
    assert not [e for e in world.events("petri_checkout_pruned") if e["utc"] > att["claimed_utc"]], "a prune removed the checkout after the claim"
    world.hold.unlink(); world.wait_exit(att); world.run_pass()


# ---------------------------------------------------------------------------- the pass on its own cadence
def test_the_petri_pass_runs_while_a_long_dispatch_blocks_the_grading_loop(world, monkeypatch):
    monkeypatch.setattr(GD, "TICK_INTERVAL_FLOOR_S", 0.2); monkeypatch.setattr(GD, "PETRI_INTERVAL_FLOOR_S", 0.05)
    world.write_wave(); world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    seen_during_tick = []

    def slow_tick(args, *a, **k):                     # a judge subprocess holding the grading loop
        end = time.time() + 6
        while time.time() < end and not world.drive_calls():
            time.sleep(0.05)
        seen_during_tick.append(bool(world.drive_calls()))
        return {}
    monkeypatch.setattr(GD, "tick", slow_tick)
    t0 = time.time(); t0_utc = GD.utc_now()
    assert GD.main(world.main_argv("--max-ticks", "1")) == 0
    assert seen_during_tick == [True], "the drive did not start while the tick was blocked"
    (started,) = world.events("petri_drive_started")
    assert t0_utc <= started["utc"] <= GD.utc_now() and time.time() - t0 < 6     # stamped inside this run (the clock is millisecond-grained: `<` could tie)
    # --once runs the single pass inline after its tick (no thread); a pass that raises is one event, never the end of the loop
    monkeypatch.setattr(GD, "tick", lambda *a, **k: {})
    monkeypatch.setattr(GD, "petri_drive_pass", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    stop = threading.Event(); stop.set()
    GD.petri_loop(GD.build_parser().parse_args(world.main_argv()), [], {"cluster_a": {"ssh": "x", "wave_root": "/r"}}, stop)
    assert world.events("petri_pass_failed")[-1]["error"] == "RuntimeError: boom"


# ------------------------------------------------------------------ the claim, the lock, simultaneous callers
def test_simultaneous_callers_start_one_drive_and_a_dead_claim_is_reconciled(world):
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING"); world.hold.write_text("")
    # 1. a claim written by a daemon that died before its spawn (or between Popen and the record): reconciled to `failed`, then a start
    rec = GD._petri_new_record(world.spec)
    rec["attempts"].append({"n": 1, "state": "claimed", "identity": {"job_id": "11111", "endpoint_mtime": "1700000000", "wave_sha": world.sha}, "pid": None,
                            "proc_start": None, "log": str(world.state / f"petri_drive.{TAG}.1.log"), "rc_file": None, "claimed_utc": "2026-01-01T00:00:00Z"})
    world.state.mkdir(parents=True); GD._petri_save(str(world.state / f"petri_drive.{TAG}.json"), rec)
    s = world.run_pass()
    assert s["started"] == 1
    ex = world.events("petri_drive_exited")
    assert len(ex) == 1 and ex[0]["attempt"] == 1 and ex[0]["state"] == "failed" and "claimed but never started" in ex[0]["reason"]
    assert [a["state"] for a in world.record()["attempts"]] == ["failed", "started"]
    # 2. while attempt 2 runs: every caller sees `running`, none starts (the record, not the process, is the arbiter across daemons)
    assert world.run_pass()["running"] == 1 and len(world.wait_calls(1)) == 1
    world.hold.unlink(); world.wait_exit(world.record()["attempts"][1])
    assert world.run_pass()["completed"] == 1
    # 3. two callers at once on a fresh endpoint identity: exactly one drive
    world.endpoint_up("22222", 1_700_000_600); world.job("22222", "RUNNING"); world.hold.write_text("")
    barrier = threading.Barrier(2); results = []

    def caller():
        barrier.wait(timeout=10); results.append(world.run_pass())
    ts = [threading.Thread(target=caller) for _ in range(2)]
    [t.start() for t in ts]; [t.join(timeout=60) for t in ts]
    assert sum(r["started"] for r in results) == 1 and sum(r["running"] + r["locked"] for r in results) == 1, results
    assert len(world.wait_calls(2)) == 2 and [a["state"] for a in world.record()["attempts"]] == ["failed", "completed", "started"]
    world.hold.unlink(); world.wait_exit(world.record()["attempts"][2]); world.run_pass()
    # 4. a lock another process holds: this pass skips with an event, starts nothing, and never waits
    world.endpoint_up("33333", 1_700_000_700); world.job("33333", "RUNNING")
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl,sys,time; fd=open(sys.argv[1],'w'); fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush=True); time.sleep(20)",
                               str(world.state / f"petri_drive.{TAG}.lock")], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.time(); s = world.run_pass()
        assert s["locked"] == 1 and s["started"] == 0 and time.time() - t0 < 5 and world.events("petri_drive_lock_held")
        assert len(world.drive_calls()) == 2                       # attempts 2 and 3 ran; the dead claim (1) never spawned; nothing new
    finally:
        holder.kill(); holder.wait()
    assert world.run_pass()["started"] == 1                    # the lock released: the start proceeds
    world.wait_exit(world.record()["attempts"][3]); assert len(world.wait_calls(3)) == 3


def test_a_checkout_whose_wave_is_closed_is_pruned_and_re_made_on_demand(world):
    """The state dir does not accumulate checkouts: one whose wave is closed to the drive — its drive completed, or no current wave names
    its sha any more — is removed once nothing runs from it; a replaced proxy re-makes it on demand; no spec known = nothing pruned."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    assert world.run_pass()["started"] == 1
    checkout = world.state / "checkouts" / world.sha
    world.wait_exit(world.record()["attempts"][0])
    s = world.run_pass()                                                     # (b) the drive completed for the wave at this sha
    assert s["completed"] == 1 and s["pruned"] == 1 and not checkout.exists() and str(checkout) not in _git(world.repo, "worktree", "list")
    (ev,) = world.events("petri_checkout_pruned"); assert ev["wave_sha"] == world.sha and ev["reason"].startswith("the drive completed for every wave")
    world.endpoint.unlink()
    s = world.run_pass()                                                     # absent endpoint after a completed drive: no eager re-make
    assert s["waiting"] == 1 and s["pruned"] == 0 and not checkout.exists()
    world.endpoint_up("22222", 1_700_000_500); world.job("22222", "RUNNING"); world.hold.write_text("")
    assert world.run_pass()["started"] == 1 and checkout.exists()           # a replaced proxy: re-made on demand, the drive starts from it
    # a checkout a drive RUNS from is never pruned, even once the wave has moved to a newer sha
    (world.repo / "configs/refrozen.txt").write_text("x\n"); _git(world.repo, "add", "-A"); _git(world.repo, "commit", "-q", "-m", "re-frozen"); sha2 = _git(world.repo, "rev-parse", "HEAD")
    spec2 = dict(world.spec, sha=sha2)
    s = GD.petri_drive_pass([spec2], **world.kw)
    assert s["running"] == 1 and s["pruned"] == 0 and checkout.exists()
    world.hold.unlink(); world.wait_exit(world.record()["attempts"][1]); world.endpoint.unlink()
    s = GD.petri_drive_pass([spec2], **world.kw)                            # (a) settled; no current wave names the old sha -> pruned; sha2 prepared eagerly
    assert s["completed"] == 0 and s["waiting"] == 1 and s["pruned"] == 1
    assert not checkout.exists() and (world.state / "checkouts" / sha2 / ".git").exists()
    assert world.events("petri_checkout_pruned")[-1] == dict(world.events("petri_checkout_pruned")[-1], checkout=str(checkout), reason="no current wave names this sha")
    assert GD.prune_petri_checkouts(state_dir=str(world.state), repo=str(world.repo), specs=[]) == [] and (world.state / "checkouts" / sha2).exists()


# --------------------------------------------------------------------------------------------- the front door and the parser
def test_dry_run_prints_the_start_and_records_nothing(world, capsys):
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    s = world.run_pass(dry_run=True)
    assert s["started"] == 1
    out = capsys.readouterr().out
    assert f"DRY-RUN: checkout {world.state}/checkouts/{world.sha}; F22_WT=<checkout> F22_PY={world.python}" in out and f"{MODEL} {VIEW} cluster_a {world.sha}" in out
    assert not (world.state / f"petri_drive.{TAG}.json").exists() and not (world.state / "checkouts").exists() and not world.drive_calls()


def test_spec_parser_refuses_a_missing_key_by_name_and_needs_a_state_dir():
    good = "tag=k_s1;model=k;view=u0;cluster=cluster_a;ssh=host;endpoint=/x/endpoint.json;sha=" + "c" * 40
    assert GD.parse_petri_specs([good])[0]["endpoint"] == "/x/endpoint.json"
    with pytest.raises(ValueError, match="missing \\['sha'\\]"):
        GD.parse_petri_specs([good.rsplit(";", 1)[0]])
    with pytest.raises(ValueError, match="not key=value"):
        GD.parse_petri_specs(["tag=k;nonsense"])
    with pytest.raises(ValueError, match="--state-dir"):
        GD.petri_drive_pass([GD.parse_petri_specs([good])[0]], state_dir=None, repo="/r", script="s", ssh_bin="ssh", dry_run=True)
    assert GD.main(["--model-id", "m", "--petri-state", "/s", "--petri-model", "m"]) == 2      # refused: no --state-dir


def test_08_hands_the_daemon_the_state_dir_the_model_the_interpreter_and_every_clusters_alias_and_wave_root(tmp_path):
    """The front door composes no spec list any more: the daemon gets the model's state dir, the model key, the fit
    interpreter and one `--petri-cluster label=ssh=<that cluster's wave root>` per configured cluster, and discovers the waves itself.
    The interpreter is one that passes the stage's Petri preflight (it imports inspect_ai + petri_bloom): a stand-in that answers the
    probe, so the hand-over is asserted whatever the runner's venv carries (tests/test_grade_petri_preflight.py pins the other branch)."""
    py = fake_python(tmp_path, petri=True)
    site = write_site(tmp_path, extra_config=f"SISTEER_PYTHON={py}")
    state = tmp_path / "state" / MODEL; state.mkdir(parents=True)
    r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", MODEL, site=site, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:") and "grade_daemon.py" in ln]
    assert len(dry) == 1, r.stdout[-1500:]
    argv = shlex.split(dry[0][len("DRY-RUN:"):])
    assert "--petri-drive" not in argv
    assert argv[argv.index("--petri-state") + 1] == str(state) and argv[argv.index("--petri-model") + 1] == MODEL
    assert argv[argv.index("--petri-python") + 1] == py
    assert "Petri drive DISABLED" not in r.stderr and "petri=drive" in r.stdout
    clusters = [argv[i + 1] for i, a in enumerate(argv) if a == "--petri-cluster"]
    assert clusters == [f"cluster_a=cluster-a-login={tmp_path}/shared_cluster_a/exp/sdfbase_full22_native={tmp_path}/shared_cluster_a",
                        f"cluster_b=cluster-b-login={tmp_path}/shared_cluster_b/exp/sdfbase_full22_native={tmp_path}/shared_cluster_b"], clusters
    assert GD.parse_petri_clusters(clusters)["cluster_b"] == {"ssh": "cluster-b-login", "wave_root": f"{tmp_path}/shared_cluster_b/exp/sdfbase_full22_native",
                                                             "shared": f"{tmp_path}/shared_cluster_b"}
    # ... and the bank root the daemon builds the drive's view under AND hands the drive (demo-import 2026-10-11): the site's SISTEER_OBJECTS_BANK,
    # whose default is $SISTEER_STATE_DIR/objects — where 05_stage_roster.sh staged the model's objects — never the daemon's legacy the grading host default
    assert argv[argv.index("--petri-bank-root") + 1] == f"{tmp_path}/state/objects", argv
    assert GD.PETRI_GRADING_HOST_BANK_ROOT_DEFAULT not in argv


def test_08_hands_the_daemon_the_sites_objects_bank_when_the_site_names_one(tmp_path):
    """SISTEER_OBJECTS_BANK set in the site config (reproduce/site.example documents it): 05's --bank default and the daemon's
    --petri-bank-root are that one value, so the view the daemon builds is under the bank the objects were staged into."""
    py = fake_python(tmp_path, petri=True)
    bank = tmp_path / "site_bank"
    site = write_site(tmp_path, extra_config=f"SISTEER_PYTHON={py}\nSISTEER_OBJECTS_BANK={bank}")
    (tmp_path / "state" / MODEL).mkdir(parents=True)
    r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", MODEL, site=site, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    (dry,) = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:") and "grade_daemon.py" in ln]
    argv = shlex.split(dry[len("DRY-RUN:"):])
    assert argv[argv.index("--petri-bank-root") + 1] == str(bank), argv
    # the same value is 05's bank default (its say line names it)
    r5 = run_stage("05_stage_roster.sh", "--dry-run", "--step", "freeze", "--model", MODEL, site=site, timeout=300)
    assert f"bank={bank} " in r5.stdout, r5.stdout[-1500:] + r5.stderr[-1500:]


def _alive(pid: int) -> bool:
    st, _ = GD._proc_stat(pid)
    return st is not None and st != "Z"


def _wait_gone(pids, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and any(_alive(p) for p in pids):
        time.sleep(0.05)
    assert not any(_alive(p) for p in pids), [(p, GD._proc_stat(p)) for p in pids]


def test_the_operator_overlay_is_exactly_three_validated_knobs_and_an_invalid_file_refuses_whole(tmp_path):
    """`<state-dir>/petri_drive_env.json` carries EXACTLY the drive's three knobs (its ONE Anthropic key's
    org, the retry-sleep ceiling, the width), each validated; a present but invalid file (not JSON, not an object, a name outside the three
    — the whole daemon-env allowlist is NOT the overlay's — a value out of range) raises OverlayRefused for the whole file, never a partial
    application; the reasons name shapes, never a value; an absent file is no overlay."""
    state = tmp_path / "state"; state.mkdir()
    f = state / "petri_drive_env.json"
    f.write_text(json.dumps({"F22_AUDITOR_ORG": "highprio", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S": 300, "PETRI_EVAL_MAX_CONNECTIONS": "2"}))
    env, dropped, names = GD.petri_drive_env({"PATH": "/bin", "HOME": str(tmp_path), "F22_AUDITOR_ORG": "lowprio", "F22_PETRI_CELLS": "x"}, (), str(state))
    assert env["F22_AUDITOR_ORG"] == "highprio" and env["PETRI_AUDITOR_RETRY_SLEEP_CAP_S"] == "300" and env["PETRI_EVAL_MAX_CONNECTIONS"] == "2"
    assert dropped == ["F22_PETRI_CELLS"] and names == ["F22_AUDITOR_ORG", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S", "PETRI_EVAL_MAX_CONNECTIONS"]
    ev = [json.loads(l) for l in (state / "grade_daemon.jsonl").read_text().splitlines()]
    (o,) = [e for e in ev if e["event"] == "petri_drive_env_overlay"]
    assert o["applied"] == names and "highprio" not in json.dumps(o)
    probe = "".join(random.choices(string.ascii_letters + string.digits, k=24))     # a value that must never reach a reason or an event
    bads = [('{"F22_AUDITOR_ORG": "highpr', "not JSON"), ('["F22_AUDITOR_ORG"]', "not a JSON object"),
            (json.dumps({"F22_AUDITOR_ORG": "highprio", "SISTEER_SECRETS_MODE": "env"}), "SISTEER_SECRETS_MODE: not an overlay knob"),
            (json.dumps({"F22_AUDITOR_ORG": "highprio", "PATH": "/x"}), "PATH: not an overlay knob"),
            (json.dumps({"F22_AUDITOR_ORG": "highprio", "HOME": "/x", "SISTEER_SECRET_CMD": "/x"}), "SISTEER_SECRET_CMD: not an overlay knob"),
            (json.dumps({"ANTHROPIC_API_KEY": probe}), "ANTHROPIC_API_KEY: not an overlay knob"),
            (json.dumps({"F22_AUDITOR_ORG": "batch"}), "F22_AUDITOR_ORG: one of lowprio, highprio"),
            (json.dumps({"F22_AUDITOR_ORG": probe}), "F22_AUDITOR_ORG: one of lowprio, highprio"),
            (json.dumps({"F22_AUDITOR_ORG": "highprio", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S": 0}), "PETRI_AUDITOR_RETRY_SLEEP_CAP_S: a positive integer <= 3600 (got 0)"),
            (json.dumps({"PETRI_AUDITOR_RETRY_SLEEP_CAP_S": 3601}), "(got 3601)"), (json.dumps({"PETRI_AUDITOR_RETRY_SLEEP_CAP_S": True}), "(got a bool)"),
            (json.dumps({"PETRI_EVAL_MAX_CONNECTIONS": 101}), "PETRI_EVAL_MAX_CONNECTIONS: a positive integer <= 100 (got 101)"),
            (json.dumps({"PETRI_EVAL_MAX_CONNECTIONS": "many"}), "(got a str)")]
    for text, reason in bads:
        f.write_text(text)
        with pytest.raises(GD.OverlayRefused) as ei:
            GD.petri_drive_env({"PATH": "/bin", "F22_AUDITOR_ORG": "lowprio"}, (), str(state))
        assert any(reason in r for r in ei.value.reasons), (text, ei.value.reasons)
        assert ei.value.file == str(f) and probe not in str(ei.value)
    ev = [json.loads(l) for l in (state / "grade_daemon.jsonl").read_text().splitlines()]
    assert len([e for e in ev if e["event"] == "petri_drive_env_overlay"]) == 1          # a refused file applies nothing and logs no application
    f.unlink()
    env2, _, names2 = GD.petri_drive_env({"PATH": "/bin"}, (), str(state))
    assert "F22_AUDITOR_ORG" not in env2 and names2 == [] and env2["SISTEER_SECRETS_MODE"] == "op"
    if os.geteuid() != 0:
        f.write_text("{}"); f.chmod(0)
        with pytest.raises(GD.OverlayRefused) as ei:
            GD.petri_drive_env({"PATH": "/bin"}, (), str(state))
        assert ei.value.reasons == ["unreadable: PermissionError"]
        f.chmod(0o644)


def test_the_expected_key_hash_is_the_vault_items_hash_computed_in_a_pipeline(tmp_path):
    """The daemon knows what the drive's key SHOULD hash to — the named org's vault item under op mode, hashed by
    `<command> <org> | sha256sum` (the secret-manager CLI SISTEER_SECRET_CMD names) so the value never enters the process; the inherited ANTHROPIC_API_KEY under env mode; None with a reason
    when the fetch fails (then no attempt starts)."""
    b = tmp_path / "bin"; _exe(b / "fake-secret-cmd", FAKE_SECRET_CMD)
    env = {"PATH": f"{b}:{os.environ['PATH']}", "HOME": str(tmp_path), "F22_AUDITOR_ORG": "highprio", "FAKE_OP_FAIL": str(tmp_path / "fail"),
           "SISTEER_SECRET_CMD": str(b / "fake-secret-cmd")}
    hp = hashlib.sha256(b"fake-vault-item-highprio").hexdigest()[:12]
    r = GD.petri_expected_key_sha12(env)
    assert (r["mode"], r["org"], r["sha12"], r["verified"], r["error"], r["kind"]) == ("op", "highprio", hp, True, None, None)
    assert GD.petri_expected_key_sha12(dict(env, F22_AUDITOR_ORG="lowprio"))["sha12"] == hashlib.sha256(b"fake-vault-item-lowprio").hexdigest()[:12]
    (tmp_path / "fail").write_text("")
    r = GD.petri_expected_key_sha12(env)
    assert r["sha12"] is None and r["kind"] == "unresolved" and "failed (rc 1)" in r["error"] and "vault unreachable" in r["error"]
    (tmp_path / "fail").unlink()
    # env mode: the inherited key is held against the NAMED vault item fetched independently, never against itself
    probe = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    e_env = dict(env, SISTEER_SECRETS_MODE="env", ANTHROPIC_API_KEY=probe)
    r = GD.petri_expected_key_sha12(e_env)                                        # a lowprio key labelled highprio: refused, never attested
    assert r["sha12"] is None and r["kind"] == "mismatch" and "not the vault item 'highprio'" in r["error"]
    assert r["inproc_sha12"] == hashlib.sha256(probe.encode()).hexdigest()[:12] and r["vault_sha12"] == hp and probe not in json.dumps(r)
    r = GD.petri_expected_key_sha12(dict(e_env, ANTHROPIC_API_KEY="fake-vault-item-highprio"))   # the inherited key IS the named item: proven
    assert (r["sha12"], r["verified"], r["kind"], r["error"]) == (hp, True, None, None)
    (tmp_path / "fail").write_text("")                                           # the vault unreachable under env mode:
    r = GD.petri_expected_key_sha12(e_env, switched=True)                        #   an org the overlay SWITCHED to cannot be proven -> refused
    assert r["sha12"] is None and r["kind"] == "unresolved" and "cannot be fetched" in r["error"]
    r = GD.petri_expected_key_sha12(e_env, switched=False)                       #   the default org is attested UNVERIFIED, and says so
    assert r["sha12"] == hashlib.sha256(probe.encode()).hexdigest()[:12] and r["verified"] is False and "unverified" in r["error"]
    (tmp_path / "fail").unlink()
    r = GD.petri_expected_key_sha12({"SISTEER_SECRETS_MODE": "env", "PATH": env["PATH"]})
    assert r["sha12"] is None and "no ANTHROPIC_API_KEY" in r["error"]


def test_a_malformed_or_foreign_overlay_refuses_the_attempt_before_the_claim_and_a_valid_one_is_attested(world):
    """A truncated file, a misspelled knob, a forbidden name beside a valid knob, an out-of-range value — each REFUSES
    the attempt by name BEFORE the claim: nothing recorded, nothing spawned, no partial application (the previous behaviour applied the valid
    keys and launched on the default pool, repeating the stall against an idle Petri GPU). The refusal stands until the file is fixed; then the
    attempt starts with the knobs applied, `petri_drive_started` names the org and the hash the key must have, the drive's DRIVE_START attests
    the hash of the key it resolved, and the next pass records the match."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True)
    overlay = world.state / "petri_drive_env.json"
    probe = "".join(random.choices(string.ascii_letters + string.digits, k=24))
    bads = ['{"F22_AUDITOR_ORG": "highpr', json.dumps({"F22_AUDITOR_ORG": "highprio", "SISTEER_SECRETS_MODE": "env"}),
            json.dumps({"F22_AUDITOR_ORG": "highprio", "PATH": "/x"}), json.dumps({"F22_AUDITOR_ORG": "highprio", "ANTHROPIC_API_KEY": probe}),
            json.dumps({"F22_AUDITOR_ORG": "batch"}), json.dumps({"F22_AUDITOR_ORG": "highprio", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S": 0}),
            json.dumps({"PETRI_EVAL_MAX_CONNECTIONS": 1000}), json.dumps(["F22_AUDITOR_ORG"])]
    for text in bads:
        overlay.write_text(text)
        s = world.run_pass()
        assert s["refused"] == 1 and s["started"] == 0, (text, s)
    assert not world.drive_calls() and (world.record().get("attempts") or []) == []       # nothing spawned, nothing claimed
    refusals = world.events("petri_drive_overlay_refused")
    assert len(refusals) == len(bads) and all(r["file"] == str(overlay) for r in refusals)
    reasons = [r for e in refusals for r in e["reasons"]]
    assert any("not JSON" in r for r in reasons) and any("SISTEER_SECRETS_MODE: not an overlay knob" in r for r in reasons)
    assert any("F22_AUDITOR_ORG: one of lowprio, highprio" in r for r in reasons) and any("(got 0)" in r for r in reasons)
    assert probe not in json.dumps(world.events()) and not world.events("petri_drive_env_overlay")
    # fixed: the next pass starts, every knob applied, the pool named and its expected hash recorded (never a value)
    overlay.write_text(json.dumps({"F22_AUDITOR_ORG": "highprio", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S": 300, "PETRI_EVAL_MAX_CONNECTIONS": "2"}))
    s = world.run_pass()
    assert s["started"] == 1 and s["refused"] == 0, s
    expected = hashlib.sha256(b"fake-vault-item-highprio").hexdigest()[:12]
    (st,) = world.events("petri_drive_started")
    assert st["auditor_org"] == "highprio" and st["secrets_mode"] == "op" and st["auditor_key_sha12_expected"] == expected
    assert st["overlay"] == ["F22_AUDITOR_ORG", "PETRI_AUDITOR_RETRY_SLEEP_CAP_S", "PETRI_EVAL_MAX_CONNECTIONS"]
    (ov,) = world.events("petri_drive_env_overlay"); assert ov["applied"] == st["overlay"] and "300" not in json.dumps(ov)
    att = world.record()["attempts"][-1]
    assert {k: att["auditor"][k] for k in ("mode", "org", "sha12", "verified", "error")} == {"mode": "op", "org": "highprio", "sha12": expected, "verified": True, "error": None}
    assert att["overlay"] == st["overlay"]
    world.wait_exit(att)
    log = Path(att["log"]).read_text()
    assert f"auditor_org=highprio auditor_key_sha12={expected} secrets_mode=op cap=300 width=2" in log, log
    world.run_pass()                                                                    # settles; the attestation is compared once
    (at,) = world.events("petri_drive_key_attested")
    assert at["match"] is True and at["sha12_logged"] == expected and at["sha12_expected"] == expected and at["org_logged"] == "highprio"
    assert world.record()["attempts"][-1]["key_attested"] is True and not world.events("problem")
    # the vault unreachable: the hash cannot be resolved, so no attempt starts (the drive would die on the same fetch) — a reason, not a failure
    world.proxy_stops("11111"); world.endpoint_up("22222", 2000); world.job("22222", "RUNNING")
    world.op_fail.write_text("")
    s = world.run_pass()
    assert s["refused"] == 1 and s["started"] == 0, s
    (ku,) = world.events("petri_drive_key_unresolved"); assert ku["org"] == "highprio" and "vault unreachable" in ku["error"]
    assert len(world.record()["attempts"]) == 1
    world.op_fail.unlink()
    assert world.run_pass()["started"] == 1


def test_a_forged_attestation_is_a_problem_and_a_drive_without_one_is_recorded_unattested(world, monkeypatch):
    """The pool is PROVEN by comparing the hash the drive attests with the one the daemon expected — a drive whose key is not the
    named org's item (an inherited key, a stale item, an override) is a `problem`, once; a drive from before the field is `unattested`, once."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    monkeypatch.setenv("FAKE_DRIVE_KEY_SHA12", "0" * 12)
    assert world.run_pass()["started"] == 1
    att = world.record()["attempts"][-1]; world.wait_exit(att)
    world.run_pass(); world.run_pass()
    (at,) = world.events("petri_drive_key_attested")
    assert at["match"] is False and at["sha12_logged"] == "0" * 12 and at["sha12_expected"] == hashlib.sha256(b"fake-vault-item-lowprio").hexdigest()[:12]
    (pr,) = [e for e in world.events("problem") if e["kind"] == "petri_auditor_key_mismatch"]
    assert pr["attempt"] == 1 and pr["org_expected"] == "lowprio" and world.record()["attempts"][0]["key_attested"] is False
    monkeypatch.delenv("FAKE_DRIVE_KEY_SHA12")
    noattest = world.tmp / "noattest"; noattest.write_text(""); monkeypatch.setenv("FAKE_DRIVE_NO_ATTEST", str(noattest))
    world.proxy_stops("11111"); world.endpoint_up("22222", 2000); world.job("22222", "RUNNING")
    assert world.run_pass()["started"] == 1
    att = world.record()["attempts"][-1]; world.wait_exit(att)
    world.run_pass(); world.run_pass()
    (un,) = world.events("petri_drive_key_unattested")
    assert un["attempt"] == 2 and "re-freeze" in un["note"] and world.record()["attempts"][1]["key_attested"] == "unattested"
    assert len(world.events("petri_drive_key_attested")) == 1


def test_a_settled_attempts_surviving_legs_are_stopped_and_block_the_next_start_until_gone(world, monkeypatch):
    """2026-10-01: a SIGTERM to the attempt-3 WRAPPER left its four petri_run.py legs alive under the old configuration, holding the per-leg
    locks; attempts 4-6 each read `already locked, skip` on every leg as 4 failed and burnt the cap in four minutes. The daemon now names an
    attempt's survivors by SESSION id (the wrapper's pid, start_new_session) when it settles the attempt, sends the group TERM, refuses to start
    beside them by name (`petri_drive_previous_legs_alive`, nothing claimed), sends KILL once SURVIVOR_KILL_AFTER_S has passed, and starts once
    they are gone."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    linger = world.tmp / "linger"; linger.write_text("")
    monkeypatch.setenv("FAKE_DRIVE_LINGER", str(linger)); monkeypatch.setenv("FAKE_DRIVE_LINGER_IGNORE_TERM", str(world.tmp / "ignore_term"))
    (world.tmp / "ignore_term").write_text("")
    world.rc.write_text("1")                                                         # the wrapper FAILS; its leg lingers, ignoring TERM
    try:
        assert world.run_pass()["started"] == 1
        att = world.record()["attempts"][0]; world.wait_exit(att)
        pids = [int(x) for x in (world.tmp / "linger.pids").read_text().split()]
        assert pids and all(_alive(p) for p in pids), pids
        s = world.run_pass()                                                         # settles attempt 1 (failed), TERMs its session: still alive -> blocked
        assert s["blocked"] == 1 and s["started"] == 0, s
        assert world.record()["attempts"][0]["state"] == "failed" and len(world.record()["attempts"]) == 1
        (stopped,) = world.events("petri_drive_legs_stopped")
        assert stopped["signal"] == "SIGTERM" and set(stopped["pids"]) >= set(pids) and stopped["attempt"] == 1 and stopped["pid"] == att["pid"]
        (blk,) = world.events("petri_drive_previous_legs_alive"); assert set(blk["pids"]) >= set(pids)
        surv = world.record()["attempts"][0]["survivors"]; assert surv["term_utc"] and set(surv["pids"]) >= set(pids)
        assert all(_alive(p) for p in pids)                                          # TERM was ignored
        s = world.run_pass()                                                         # inside the window: no KILL yet, still blocked, still one attempt
        assert s["blocked"] == 1 and s["started"] == 0 and len(world.events("petri_drive_legs_stopped")) == 1
        monkeypatch.setattr(GD, "SURVIVOR_KILL_AFTER_S", 0.0)
        s = world.run_pass()                                                         # the window passed: KILL; this pass still saw them alive
        assert s["started"] == 0, s
        ev = world.events("petri_drive_legs_stopped"); assert len(ev) == 2 and ev[1]["signal"] == "SIGKILL"
        _wait_gone(pids)
        linger.unlink()                                                              # attempt 2's drive spawns no leg
        s = world.run_pass()                                                         # gone: the next attempt starts
        assert s["started"] == 1 and s["blocked"] == 0, s
        assert [a["n"] for a in world.record()["attempts"]] == [1, 2] and world.record()["attempts"][0]["survivors"]["kill_utc"]
    finally:
        linger.unlink(missing_ok=True)
    # a leg that honours TERM is gone by the next pass: stopped once, never killed (the KILL window back at its default), the next attempt starts
    monkeypatch.setattr(GD, "SURVIVOR_KILL_AFTER_S", 60.0)
    world.wait_exit(world.record()["attempts"][1])
    (world.tmp / "ignore_term").unlink(); linger.write_text("")                     # attempt 3's leg lingers and honours TERM
    world.endpoint_up("11111", 1001); world.job("11111", "RUNNING"); world.rc.write_text("1")   # a fresh endpoint identity: two failures already count toward the first's cap
    try:
        s = world.run_pass()                                                         # attempt 2 settles (failed, no survivors); attempt 3 starts
        assert s["started"] == 1 and s["blocked"] == 0, s
        att3 = world.record()["attempts"][2]; world.wait_exit(att3)
        pids3 = [int(x) for x in (world.tmp / "linger.pids").read_text().split()][len(pids):]
        assert pids3 and all(_alive(p) for p in pids3), pids3
        s = world.run_pass()                                                         # TERM lands; the leg dies at once (or within the pass)
        assert s["blocked"] + s["started"] == 1, s
        _wait_gone(pids3)
        if s["started"] == 0:
            assert world.run_pass()["started"] == 1
        ev = world.events("petri_drive_legs_stopped")
        assert [e["signal"] for e in ev if e["attempt"] == 3] == ["SIGTERM"] and not [e for e in ev if e["attempt"] == 2]
        assert [a["n"] for a in world.record()["attempts"]] == [1, 2, 3, 4]
    finally:
        linger.unlink(missing_ok=True)


def test_under_env_mode_an_inherited_key_labelled_with_another_org_refuses_the_start(world, monkeypatch):
    """Under SISTEER_SECRETS_MODE=env the drive uses the inherited ANTHROPIC_API_KEY whatever F22_AUDITOR_ORG says, and
    the daemon's expected hash was computed from that same key — a lowprio key labelled highprio attested true. Now the inherited key's hash is
    held against the NAMED vault item fetched independently: a mismatch refuses the start by name (`petri_drive_key_mismatch`, nothing claimed);
    the matching key starts, attested and verified."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True)
    (world.state / "petri_drive_env.json").write_text(json.dumps({"F22_AUDITOR_ORG": "highprio"}))
    probe = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env"); monkeypatch.setenv("ANTHROPIC_API_KEY", probe)       # an inherited key that is NOT highprio
    s = world.run_pass()
    assert s["refused"] == 1 and s["started"] == 0, s
    (mm,) = world.events("petri_drive_key_mismatch")
    assert mm["org"] == "highprio" and mm["mode"] == "env" and "not the vault item 'highprio'" in mm["error"] and probe not in json.dumps(world.events())
    assert (world.record().get("attempts") or []) == [] and not world.drive_calls()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-vault-item-highprio")                                    # the inherited key IS the named item
    s = world.run_pass()
    assert s["started"] == 1 and s["refused"] == 0, s
    (st,) = world.events("petri_drive_started")
    hp = hashlib.sha256(b"fake-vault-item-highprio").hexdigest()[:12]
    assert st["auditor_org"] == "highprio" and st["secrets_mode"] == "env" and st["auditor_key_sha12_expected"] == hp
    att = world.record()["attempts"][-1]; assert att["auditor"]["verified"] is True and att["auditor"]["inproc_sha12"] == hp
    world.wait_exit(att); world.run_pass()
    (at,) = world.events("petri_drive_key_attested"); assert at["match"] is True and at["sha12_logged"] == hp


def test_a_capped_endpoints_lingering_term_ignoring_leg_is_still_killed(world, monkeypatch):
    """After the third failed attempt `_petri_verdict` returned `capped` before the survivors were looked at, so a
    TERM-ignoring leg of the last attempt kept auditing under its locks for good. The survivors are now stopped before EVERY verdict: TERM at
    settlement, `blocked` (not `capped`) while they live, KILL past the window, and only then the cap."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    world.rc.write_text("1"); world.result_ctl.write_text("none")
    for i in range(2):                                                           # two plain failures
        assert world.run_pass()["started"] == 1, i
        world.wait_exit(world.record()["attempts"][-1])
    linger = world.tmp / "linger"; linger.write_text(""); (world.tmp / "ignore_term").write_text("")
    monkeypatch.setenv("FAKE_DRIVE_LINGER", str(linger)); monkeypatch.setenv("FAKE_DRIVE_LINGER_IGNORE_TERM", str(world.tmp / "ignore_term"))
    try:
        s = world.run_pass()                                                     # attempt 2 settles; attempt 3 starts and leaves a TERM-ignoring leg
        assert s["started"] == 1, s
        att3 = world.record()["attempts"][2]; world.wait_exit(att3)
        pids = [int(x) for x in (world.tmp / "linger.pids").read_text().split()]
        assert pids and all(_alive(p) for p in pids)
        s = world.run_pass()                                                     # the third failure settles: TERM, and BLOCKED — not capped while the leg lives
        assert s["blocked"] == 1 and s["capped"] == 0 and s["started"] == 0, s
        assert not world.events("petri_drive_gave_up") and world.events("petri_drive_legs_stopped")[-1]["signal"] == "SIGTERM"
        assert all(_alive(p) for p in pids)
        monkeypatch.setattr(GD, "SURVIVOR_KILL_AFTER_S", 0.0)
        s = world.run_pass()                                                     # past the window: KILL (still blocked this pass)
        assert s["blocked"] == 1 and s["capped"] == 0, s
        assert world.events("petri_drive_legs_stopped")[-1]["signal"] == "SIGKILL"
        _wait_gone(pids)
        s = world.run_pass()                                                     # gone: now the endpoint is capped, once
        assert s["capped"] == 1 and s["blocked"] == 0 and s["started"] == 0, s
        (gave_up,) = world.events("petri_drive_gave_up"); assert gave_up["attempts"] == 3
        assert [a["state"] for a in world.record()["attempts"]] == ["failed"] * 3 and world.record()["attempts"][2]["survivors"]["kill_utc"]
    finally:
        linger.unlink(missing_ok=True)


def test_a_drive_refusing_over_held_locks_exits_4_blocked_and_is_never_counted_toward_the_cap(world):
    """The drive's side of the same defect: every leg's per-cell flock held means a previous attempt is still running, and the drive now refuses
    by name with rc 4 before launching anything (petri_drive_full22.sh). The daemon records that exit as `blocked` — not `failed`: four such
    exits do not reach the cap of three, `petri_drive_gave_up` never fires, and the endpoint is driven once the locks are free."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    world.rc.write_text(str(GD.PETRI_DRIVE_RC_LOCKS_HELD)); world.result_ctl.write_text("none")
    for i in range(4):
        s = world.run_pass()
        assert s["started"] == 1, (i, s)
        world.wait_exit(world.record()["attempts"][-1])
    s = world.run_pass()
    assert s["started"] == 1 and s["capped"] == 0, s
    rec = world.record()
    assert [a["state"] for a in rec["attempts"][:4]] == ["blocked"] * 4 and rec["attempts"][4]["state"] == "started"
    exits = world.events("petri_drive_exited")
    assert len(exits) == 4 and all(e["state"] == "blocked" and e["exit_code"] == 4 and "per-leg locks" in e["reason"] for e in exits)
    assert not world.events("petri_drive_gave_up") and GD._petri_verdict(rec, GD._identity_key(rec["attempts"][0]["identity"])) is None
    world.wait_exit(rec["attempts"][4])
    world.rc.write_text("0"); world.drive_result("11111")                              # the locks are free: the fifth attempt is blocked too; the sixth completes
    s = world.run_pass(); assert s["started"] == 1
    world.wait_exit(world.record()["attempts"][5])
    world.run_pass()
    assert world.record()["attempts"][5]["state"] == "completed" and world.run_pass()["completed"] == 1


def test_a_drive_inherited_across_the_daemons_re_exec_is_reaped_and_settled_by_its_rc(world):
    """The daemon re-execs itself on code drift and loses the Popen of a drive it spawned; the drive stays its child. Before: `_proc_alive`
    read the exited wrapper as a zombie (settled by the rc file, correctly) but nothing reaped it, so a zombie stayed for the daemon's life.
    Now the settlement reaps a zombie whose identity (pid + start time) is the attempt's."""
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    assert world.run_pass()["started"] == 1
    att = world.record()["attempts"][0]
    # the re-exec'd daemon has no Popen for the child it inherited. The Popen object itself is KEPT alive here: an unreferenced, unwaited
    # Popen is parked in subprocess._active by its __del__ and reaped by the next Popen() anyone creates — which would make the daemon's
    # own reap (the thing under test) unobservable, and did on the cluster gate (2026-10-08: _proc_stat gave None, not 'Z')
    inherited = list(GD._PETRI_PROCS.values()); GD._PETRI_PROCS.clear()
    world.wait_exit(att)
    assert GD._proc_stat(att["pid"])[0] == "Z"                                       # a zombie: this process is its parent
    s = world.run_pass()
    assert world.record()["attempts"][0]["state"] == "completed" and s["completed"] == 1, s
    assert GD._proc_stat(att["pid"]) == (None, None), "the zombie was not reaped"
    del inherited


# ------------------------------------------------------------------------------------------ the OVERFLOW PASS (petri-completion, 2026-10-13)
OVER_TAG = f"{TAG}.overflow"


def _overflow_row(world, job_id="33333", cluster="cluster_a", sha=None, bare=False):
    """06 --pass overflow's ledger row: a `submitted` row of unit petri.overflow (the LAST one wins) carrying `extra.sha` = the sha the re-run proxy
    was submitted at (default the wave's; `bare` = none, an older ledger's row)."""
    row = {"event": "submitted", "cluster": cluster, "job_id": job_id, "job_name": f"f22-petri-{TAG}-rerun", "unit": "petri.overflow",
           "logical_work_id": f"{MODEL}/{VIEW}/petri/overflow", "model": MODEL}
    if not bare:
        row["extra"] = {"kind": "target", "group": None, "sha": sha or world.sha}
    with open(world.state_root / "submissions.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _overflow_spec(world, job_id="33333", first_job_id="11111") -> dict:
    return {"tag": OVER_TAG, "base_tag": TAG, "pass": "overflow", "model": MODEL, "view": VIEW, "cluster": "cluster_a", "ssh": "cluster-a-login", "sha": world.sha,
            "job_id": job_id, "job_name": f"f22-petri-{TAG}-rerun", "endpoint": str(world.remote / "petri" / TAG / "serving_rerun" / "endpoint.json"),
            "shared": str(world.shared), "first_job_id": first_job_id}


def _rerun_endpoint_up(world, job_id="33333", mtime=1_700_002_000):
    """The rerun proxy is up: serving_rerun/endpoint.json naming its job, its cells_served.json, its job-start registration (the declared cells)."""
    ep = Path(_overflow_spec(world, job_id)["endpoint"]); ep.parent.mkdir(parents=True, exist_ok=True)
    cs = ep.parent / "cells_served.json"; cs.write_text(json.dumps({"engine": "lora_native", "cells_skipped": []}))
    ep.write_text(json.dumps({"slurm_job_id": job_id, "mode": "petri", "node": "TESTHOST", "cells_served": str(cs)})); os.utime(ep, (mtime, mtime))
    world.register(job_id); world.job(job_id, "RUNNING")
    return ep


def _first_record(world, job_id="11111", state="incomplete"):
    rec = {"tag": TAG, "model": MODEL, "view": VIEW, "cluster": "cluster_a", "wave_sha": world.sha, "cap": 3,
           "attempts": [{"n": 1, "state": state, "identity": {"job_id": job_id, "endpoint_mtime": "1", "wave_sha": world.sha}, "pid": None}]}
    world.state.mkdir(parents=True, exist_ok=True)
    (world.state / f"petri_drive.{TAG}.json").write_text(json.dumps(rec))


def test_discovery_yields_the_overflow_spec_beside_the_first_pass_with_its_own_tag_endpoint_and_start_gate(world):
    clusters = GD.parse_petri_clusters([f"cluster_a=cluster-a-login={world.remote}={world.shared}"])
    world.write_wave(job_id="11111")
    (first,) = GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))
    assert "pass" not in first and first["tag"] == TAG                                   # the first-pass spec is byte-identical to before the field
    _overflow_row(world, "33333")
    first2, over = GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))
    assert first2 == first
    assert over == _overflow_spec(world, "33333", "11111")
    assert over["endpoint"].endswith(f"/petri/{TAG}/serving_rerun/endpoint.json") and over["tag"] == f"{TAG}.overflow" and over["first_job_id"] == "11111"
    _overflow_row(world, "44444")                                                            # a re-minted overflow proxy: the LAST row wins
    assert GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))[1]["job_id"] == "44444"
    world.write_wave(job_id="22222")                                                         # the first pass re-minted: the gate follows the first pass's CURRENT job
    assert GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))[1]["first_job_id"] == "22222"
    _overflow_row(world, "55555", cluster="cluster_zz")                                      # an overflow proxy on an unknown cluster: said once, no spec
    assert len(GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))) == 1
    assert [e["tag"] for e in world.events("petri_drive_no_cluster")] == [OVER_TAG]
    # the explicit spec grammar: `pass` defaults to first; an overflow spec needs its first_job_id
    (spec,) = GD.parse_petri_specs([f"tag={TAG};model={MODEL};view={VIEW};cluster=cluster_a;ssh=h;endpoint=/e;sha={world.sha}"])
    assert GD._spec_pass(spec) == "first"
    with pytest.raises(ValueError, match="first_job_id"):
        GD.parse_petri_specs([f"tag={OVER_TAG};model={MODEL};view={VIEW};cluster=cluster_a;ssh=h;endpoint=/e;sha={world.sha};pass=overflow"])
    with pytest.raises(ValueError, match="pass must be"):
        GD.parse_petri_specs([f"tag={OVER_TAG};model={MODEL};view={VIEW};cluster=cluster_a;ssh=h;endpoint=/e;sha={world.sha};pass=second"])


def test_the_overflow_drive_starts_only_after_the_first_pass_settled_incomplete_and_carries_the_pass_env_and_argv(world, monkeypatch):
    """The start gate: no first-pass record -> waiting (said once per reason); a live first-pass drive -> waiting; an `incomplete` attempt for the
    first pass's bound job -> the overflow drive starts from the rerun endpoint with F22_OVERFLOW_RERUN=1 (the daemon's OWN setting, after the
    deny-list stripped the inherited copy) and `ledgered` as positional 5, recorded `pass: overflow` on its own record; the first-pass spec's
    child has neither even when the daemon's environment carries the knob."""
    monkeypatch.setenv("F22_OVERFLOW_RERUN", "1")                                            # the operator's environment: denied to every drive
    spec = _overflow_spec(world)
    _rerun_endpoint_up(world, "33333")
    s = world.run_pass(spec); assert s["waiting"] == 1 and s["started"] == 0, s
    (ev,) = world.events("petri_overflow_waiting_first_pass"); assert ev["tag"] == OVER_TAG and ev["first_job_id"] == "11111" and "no first-pass record" in ev["reason"]
    _first_record(world, "11111", state="started")
    s = world.run_pass(spec); assert s["waiting"] == 1 and s["started"] == 0
    assert [e["reason"] for e in world.events("petri_overflow_waiting_first_pass")][-1].startswith("a first-pass drive of job 11111 is live")
    world.run_pass(spec); assert len(world.events("petri_overflow_waiting_first_pass")) == 2    # said once per (tag, job, reason)
    _first_record(world, "11111", state="failed")
    s = world.run_pass(spec); assert s["waiting"] == 1 and "settled incomplete" in world.events("petri_overflow_waiting_first_pass")[-1]["reason"]
    _first_record(world, "11111", state="incomplete")
    s = world.run_pass(spec); assert s["started"] == 1, s
    rec = json.loads((world.state / f"petri_drive.{OVER_TAG}.json").read_text())
    att = rec["attempts"][0]; assert att["pass"] == "overflow" and att["argv"][-1] == "ledgered" and len(att["argv"]) == 7 and "F22_OVERFLOW_RERUN" in att["env_keys"]
    assert att["env_dropped"] == ["F22_OVERFLOW_RERUN"] and att["identity"]["job_id"] == "33333"
    (started,) = world.events("petri_drive_started"); assert started["pass"] == "overflow" and started["legs"] == "ledgered" and started["tag"] == OVER_TAG
    world.wait_exit(att)
    (call,) = world.drive_calls(); assert call.endswith(f"nargs=5 args={MODEL} {VIEW} cluster_a {world.sha} ledgered"), call
    assert Path(str(world.drive_log) + ".ovf").read_text().splitlines() == [f"ovf=1 nargs=5 args={MODEL} {VIEW} cluster_a {world.sha} ledgered"]
    assert not (world.state / f"petri_drive.{TAG}.json").read_text().count('"pass": "overflow"')   # the first pass's record is untouched
    # the first-pass spec's child: four arguments, the knob dropped, legs ALL
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    (world.state / f"petri_drive.{TAG}.json").unlink()
    s = world.run_pass(); assert s["started"] == 1
    att1 = world.record()["attempts"][0]; world.wait_exit(att1)
    assert att1["pass"] == "first" and att1["env_dropped"] == ["F22_OVERFLOW_RERUN"] and "F22_OVERFLOW_RERUN" not in att1["env_keys"]
    call1 = world.drive_calls()[-1]; assert call1.endswith(f"nargs=4 args={MODEL} {VIEW} cluster_a {world.sha}"), call1
    assert Path(str(world.drive_log) + ".ovf").read_text().splitlines()[-1] == f"ovf=unset nargs=4 args={MODEL} {VIEW} cluster_a {world.sha}"
    assert world.events("petri_drive_started")[-1]["pass"] == "first" and world.events("petri_drive_started")[-1]["legs"] == "ALL"
    # the dry run prints the overflow start it would make
    assert GD.PETRI_DRIVE_ENV_DENY == ("F22_PETRI_CELLS", "F22_PETRI_OBJECTS_ONLY", "F22_PETRI_RELEASE_SCOPE", "F22_OVERFLOW_RERUN", "F22_SECOND_PASS")


def test_the_dry_run_names_the_overflow_drives_env_and_positional(world, capsys):
    _first_record(world, "11111", state="incomplete"); _rerun_endpoint_up(world, "33333")
    s = world.run_pass(_overflow_spec(world), dry_run=True)
    out = capsys.readouterr().out
    assert s["started"] == 1 and "F22_OVERFLOW_RERUN=1 " in out and out.rstrip().split("  (endpoint identity")[0].endswith(f"{world.sha} ledgered > {world.state}/petri_drive.{OVER_TAG}.<n>.log"), out
    assert not (world.state / f"petri_drive.{OVER_TAG}.json").exists()


def test_each_passs_drive_checks_out_the_units_submitted_sha_when_the_wave_file_pins_another(world):
    """End to end for BOTH passes: WAVE.<view>.json pins the launch sha; the FIRST pass's proxy was re-minted at the tip
    (--units petri.proxy / --resubmit leave WAVE unchanged) and the OVERFLOW proxy was submitted at the tip too (06 --pass overflow never rewrites
    WAVE). Both discovered specs carry the tip; both drives run from `<state>/checkouts/<tip>` with the tip as their sha positional — the wave
    file's sha is checked out nowhere. An overflow row without a sha is refused by name while the first pass's spec stands; the first pass's
    refusal leaves the overflow spec standing on its own (its start gate still reads the first pass's record)."""
    clusters = GD.parse_petri_clusters([f"cluster_a=cluster-a-login={world.remote}={world.shared}"])
    disc = lambda: GD.discover_petri_specs(str(world.state_root), MODEL, clusters, str(world.state))   # noqa: E731
    world.write_wave(job_id="22222", sha=world.sha, ledger_sha=world.tip)              # WAVE at the launch sha; the re-minted proxy's row at the tip
    _overflow_row(world, "33333", sha=world.tip)                                          # the re-run proxy submitted at the tip
    first, over = disc()
    assert first["sha"] == world.tip and over["sha"] == world.tip and over["first_job_id"] == "22222" and over["job_id"] == "33333"
    assert sorted((e["tag"], e["wave_sha"], e["ledger_sha"]) for e in world.events("petri_drive_sha_mismatch")) == sorted([(TAG, world.sha, world.tip), (OVER_TAG, world.sha, world.tip)])
    # the FIRST pass: the drive starts from the checkout at the TIP, the tip as positional 4, the identity keyed on it
    world.endpoint_up("22222", 1_700_000_000); world.job("22222", "RUNNING")
    s = world.run_pass(first); assert s["started"] == 1, s
    att = world.record()["attempts"][0]; world.wait_exit(att)
    assert att["checkout"] == str(world.state / "checkouts" / world.tip) and att["argv"][-1] == world.tip and att["identity"]["wave_sha"] == world.tip and att["pass"] == "first"
    assert world.drive_calls()[-1].endswith(f"nargs=4 args={MODEL} {VIEW} cluster_a {world.tip}"), world.drive_calls()
    assert world.record()["wave_sha"] == world.tip and world.events("petri_drive_started")[-1]["wave_sha"] == world.tip
    assert (world.state / "checkouts" / world.tip).is_dir() and not (world.state / "checkouts" / world.sha).exists()
    # the OVERFLOW pass: the first pass settled incomplete for job 22222 -> the overflow drive starts from the checkout at ITS sha (the tip), `ledgered`
    _first_record(world, "22222", state="incomplete"); _rerun_endpoint_up(world, "33333")
    s = world.run_pass(over); assert s["started"] == 1, s
    orec = json.loads((world.state / f"petri_drive.{OVER_TAG}.json").read_text()); oatt = orec["attempts"][0]; world.wait_exit(oatt)
    assert oatt["checkout"] == str(world.state / "checkouts" / world.tip) and oatt["argv"][-2:] == [world.tip, "ledgered"] and oatt["pass"] == "overflow"
    assert world.drive_calls()[-1].endswith(f"nargs=5 args={MODEL} {VIEW} cluster_a {world.tip} ledgered"), world.drive_calls()
    assert orec["wave_sha"] == world.tip and not (world.state / "checkouts" / world.sha).exists()
    # an overflow row without a sha (and no attempt event): the overflow spec is refused by name; the first pass's spec stands
    _overflow_row(world, "44444", bare=True)
    (only,) = disc(); assert only["tag"] == TAG
    (ref,) = world.events("petri_drive_sha_unresolved"); assert ref["tag"] == OVER_TAG and ref["job_id"] == "44444"
    # the first pass's row without a sha: its spec is refused; the overflow spec (its own row at the tip) stands, gated on the first pass's current job
    _overflow_row(world, "55555", sha=world.tip); world.write_wave(job_id="66666", bare=True)
    (only,) = disc(); assert only["tag"] == OVER_TAG and only["sha"] == world.tip and only["first_job_id"] == "66666"
    assert [e["job_id"] for e in world.events("petri_drive_sha_unresolved")] == ["44444", "66666"]


def _verify_doc(tmp: Path, *, scope, planned, filed, rows, detail=(), verify_rc=0, verdict="complete", rc=0, mode="on"):
    tmp.mkdir(parents=True, exist_ok=True)
    v = tmp / "verify.json"; v.write_text(json.dumps({"n_planned": planned, "n_filed": filed, "runs": rows}))
    r = tmp / "result.json"
    r.write_text(json.dumps({"verdict": verdict, "rc": rc, "legs": {"detail": list(detail)},
                             "handoff": {"mode": mode, "finish_rc": 0, "verify_rc": verify_rc if scope != "control_only" else None, "summary": str(v) if scope != "control_only" else None,
                                         "runs_planned": planned, "runs_filed": filed, "verify_scope": scope}}))
    return str(r)


def test_the_overflow_verify_rule_credits_out_of_scope_runs_and_a_control_only_pass_while_the_first_pass_is_byte_identical(world, tmp_path):
    """An overflow attempt's exit 0 — required = expected − skipped − OUT-OF-SCOPE (the registered runs of cells the proxy declared but the
    drive did not select: `none`'s, declared for the twin; on a resubmit the cells an earlier overflow drive completed); a control-only pass
    (runs_planned 0, no verify) is `completed` iff every in-scope leg is a control leg marked complete; a first-pass attempt with planned 0 is
    `failed` with today's text."""
    ids = world.run_ids("33333")                                                    # 3 objects x 7 evals = 21 expected
    ident = GD._petri_identity("33333", "1700002000", world.sha); key = GD._identity_key(ident)
    def rec_with(expected):
        return {"tag": OVER_TAG, "attempts": [], "endpoints": {key: {"identity": ident, "expected_ids": sorted(expected), "covered": [], "skipped_runs": [], "complete": False}}}
    def rows(filed_objects, out_objects):
        return [{"run_id": rid, "eval_id": e, "object": o, "object_wave": o, "scope": "out" if o in out_objects else "in", "filed": o in filed_objects, "state": "generated" if o in filed_objects else None}
                for o, rids in ids.items() for e, rid in zip(EVALS, rids)]
    # (h) none's 7 out, 14 filed of 14 planned -> completed
    rec = rec_with(sum(ids.values(), []))
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "h", scope="selected", planned=14, filed=14, rows=rows(("prompted__a", "prompted__b"), ("none",)))}
    state, why, _doc = GD._petri_verify_result(rec, att)
    assert (state, why) == ("completed", None) and att["coverage"] == {"filed_by_this_drive": 14, "covered": 14, "required": 14, "expected": 21, "skipped": 0, "out_of_scope": 7, "remaining": 0, "verify_scope": "selected"}
    assert rec["endpoints"][key]["complete"] is True and sorted(rec["endpoints"][key]["out_of_scope"]) == sorted(ids["none"])
    # (h2) a resubmit: 14 out (none + the earlier-completed prompted__a), 7 filed of 7 planned -> completed
    rec = rec_with(sum(ids.values(), [])); 
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "h2", scope="selected", planned=7, filed=7, rows=rows(("prompted__b",), ("none", "prompted__a")))}
    assert GD._petri_verify_result(rec, att)[0] == "completed" and att["coverage"]["out_of_scope"] == 14 and att["coverage"]["required"] == 7
    # a shortfall inside the scope is still partial, counted toward the cap
    rec = rec_with(sum(ids.values(), [])); 
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "hp", scope="selected", planned=14, filed=14, rows=rows(("prompted__a",), ("none",)))}
    state, why, _ = GD._petri_verify_result(rec, att); assert state == "partial" and why.startswith("partial:7/14") and "out of scope 7" in why
    # (h3) control-only: planned 0, no summary, one in-scope control leg complete -> completed, required 0
    rec = rec_with(sum(ids.values(), [])); 
    detail = [{"idx": 10, "cell_id": "none__lora0", "control_of": "none", "status": "complete", "in_scope": True}, {"idx": 0, "cell_id": "none", "control_of": None, "status": "out_of_scope", "in_scope": False}]
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "h3", scope="control_only", planned=0, filed=0, rows=[], detail=detail)}
    state, why, _ = GD._petri_verify_result(rec, att); assert (state, why) == ("completed", None) and att["coverage"]["required"] == 0 and att["coverage"]["control_legs"] == ["none__lora0"]
    # (h4) control-only with a non-control leg in scope -> failed by name
    rec = rec_with(sum(ids.values(), [])); 
    bad = [{"idx": 10, "cell_id": "none__lora0", "control_of": "none", "status": "complete", "in_scope": True}, {"idx": 4, "cell_id": "x", "control_of": None, "status": "complete", "in_scope": True}]
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "h4", scope="control_only", planned=0, filed=0, rows=[], detail=bad)}
    state, why, _ = GD._petri_verify_result(rec, att); assert state == "failed" and "no registered run in scope but a non-control / incomplete leg in scope" in why
    # (h5) a FIRST-pass attempt with planned 0: today's text, byte for byte
    rec = rec_with(sum(ids.values(), [])); 
    att = {"identity": ident, "result_file": _verify_doc(tmp_path / "h5", scope="control_only", planned=0, filed=0, rows=[], detail=detail)}
    state, why, _ = GD._petri_verify_result(rec, att); assert state == "failed" and why == "exit 0 but only 0 of 0 planned run(s) verified filed on this host's spool (verify_rc=None)"
    att = {"pass": "first", "identity": ident, "result_file": _verify_doc(tmp_path / "h5", scope="selected", planned=14, filed=14, rows=rows(("prompted__a", "prompted__b"), ("none",)))}
    state, why, _ = GD._petri_verify_result(rec, att); assert state == "partial" and why.startswith("partial:14/21")   # the first pass never credits out-of-scope runs
    # (h6) an overflow attempt with planned 0 and no control_only -> failed
    rec = rec_with(sum(ids.values(), [])); 
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "h6", scope="selected", planned=0, filed=0, rows=[])}
    state, why, _ = GD._petri_verify_result(rec, att); assert state == "failed" and "only 0 of 0 planned run(s)" in why
    # an overflow result naming no verify scope at all (a drive older than this contract) -> failed by name
    rec = rec_with(sum(ids.values(), [])); 
    att = {"pass": "overflow", "identity": ident, "result_file": _verify_doc(tmp_path / "h7", scope=None, planned=14, filed=14, rows=rows(("prompted__a", "prompted__b"), ("none",)))}
    state, why, _ = GD._petri_verify_result(rec, att); assert state == "failed" and "names no verify scope" in why
    # the gate as a function
    _first_record(world, "11111", state="incomplete")
    assert GD.petri_overflow_gate(str(world.state), _overflow_spec(world)) == (True, "first-pass job 11111 settled incomplete")
    assert GD.petri_overflow_gate(str(world.state), _overflow_spec(world, first_job_id="99999"))[0] is False
    assert GD.petri_overflow_gate(None, _overflow_spec(world))[0] is False


def test_the_daemon_still_denies_the_env_cell_selector_the_proxys_declaration_is_the_only_leg_source():
    """The legs come from the proxy's declared cells in cells_served.json (petri_run.requested_cells_of); an env override of the
    selection stays the wrong door -- F22_PETRI_CELLS and its siblings remain on PETRI_DRIVE_ENV_DENY and never reach the drive."""
    assert "F22_PETRI_CELLS" in GD.PETRI_DRIVE_ENV_DENY and "F22_PETRI_OBJECTS_ONLY" in GD.PETRI_DRIVE_ENV_DENY
    env, dropped, overlay = GD.petri_drive_env({"F22_PETRI_CELLS": "si_x", "PATH": "/bin", "HOME": "/h"}, extra_allow=("PATH", "HOME"))
    assert "F22_PETRI_CELLS" not in env and "F22_PETRI_CELLS" in dropped and overlay == []


# ------------------------------------------------------------------------------------------ the attestation under wif

WIF_IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_11111111", "ANTHROPIC_ORGANIZATION_ID": "org_22222222",
           "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "ws_44444444"}


def _wif_world(world, monkeypatch) -> Path:
    """The world under SISTEER_SECRETS_MODE=wif: an ids dir with the three pools' files (and the stray empty wif-lowprio.env), named by
    SISTEER_WIF_ENV_DIR (a SISTEER_ variable: it rides the drive's allowlist); the fake vault FAILS (the secret-manager CLI is never consulted under wif)
    and the daemon's own environment carries a stray ANTHROPIC_API_KEY that must never reach the drive."""
    d = world.tmp / "wif"; d.mkdir(exist_ok=True)
    for stem in ("loprio", "hiprio", "batch"):
        (d / f"wif-{stem}.env").write_text("# synthetic ids (TESTHOST)\n" + "".join(f"{k}={v}-{stem}\n" for k, v in WIF_IDS.items()))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif"); monkeypatch.setenv("SISTEER_WIF_ENV_DIR", str(d))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stray-" + "x" * 20)
    world.op_fail.write_text("")
    return d


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _ids_file(d: Path, org: str) -> Path:
    """The pool's federation ids file under the ids dir ``d``, as the vault names it (the drive's environment names the dir)."""
    from sisteer import vault
    return Path(vault.wif_env_path(org, env={vault.WIF_ENV_DIR_ENV: str(d)}))


def test_under_wif_the_daemon_expects_the_ids_files_digest_never_a_key_and_the_drive_attests_it(world, monkeypatch):
    """Under SISTEER_SECRETS_MODE=wif the daemon hashes NO vault item (the secret-manager CLI fails here and is never asked); what the
    drive must attest is sha256 of the named org's federation ids FILE, read through the vault under the DRIVE's environment;
    `petri_drive_started` names it as auditor_ids_sha12_expected (auditor_auth wif, no key hash); the drive's environment carries no
    ANTHROPIC_API_KEY though the daemon's does (dropped by name); the drive's DRIVE_START attests `auditor_auth=wif:<org> ids_sha12=<digest>`
    and the settlement records the match; the overlay's F22_AUDITOR_ORG=highprio moves the expectation to the other org's file."""
    d = _wif_world(world, monkeypatch)
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    s = world.run_pass()
    assert s["started"] == 1 and s["refused"] == 0, s
    lo = _digest(_ids_file(d, "lowprio"))
    (st,) = world.events("petri_drive_started")
    assert st["auditor_org"] == "lowprio" and st["secrets_mode"] == "wif" and st["auditor_auth"] == "wif"
    assert st["auditor_ids_sha12_expected"] == lo and st["auditor_key_sha12_expected"] is None
    assert "ANTHROPIC_API_KEY" in st["env_dropped"]
    att = world.record()["attempts"][-1]
    assert {k: att["auditor"].get(k) for k in ("mode", "org", "sha12", "verified", "error", "kind", "auth", "ids_file")} == {
        "mode": "wif", "org": "lowprio", "sha12": lo, "verified": True, "error": None, "kind": None, "auth": "wif", "ids_file": _ids_file(d, "lowprio").name}
    assert "ANTHROPIC_API_KEY" in att["env_dropped"] and "ANTHROPIC_API_KEY" not in att["env_keys"] and "SISTEER_WIF_ENV_DIR" in att["env_keys"]
    world.wait_exit(att)
    calls = world.drive_calls(); assert len(calls) == 1 and "secrets=wif" in calls[0], calls
    assert Path(str(world.drive_log) + ".env").read_text().splitlines() == ["key_present="]      # the daemon's stray key never reached the drive
    log = Path(att["log"]).read_text()
    assert f" auditor_org=lowprio auditor_auth=wif:lowprio ids_sha12={lo} secrets_mode=wif " in log and "auditor_key_sha12" not in log, log
    world.run_pass()
    (at,) = world.events("petri_drive_key_attested")
    assert at["match"] is True and at["auth_expected"] == "wif" and at["auth_logged"] == "wif" and at["org_attested"] == "lowprio" and at["org_logged"] == "lowprio"
    assert at["sha12_logged"] == lo and at["sha12_expected"] == lo and world.record()["attempts"][-1]["key_attested"] is True
    assert not world.events("problem") and not world.events("petri_drive_key_unresolved")
    assert "stray-" not in json.dumps(world.events()) and "stray-" not in json.dumps(world.record())
    # the overlay switches the org: the expectation is the other org's file; the secret-manager CLI still never asked
    world.proxy_stops("11111"); world.endpoint_up("22222", 2000); world.job("22222", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True)
    (world.state / "petri_drive_env.json").write_text(json.dumps({"F22_AUDITOR_ORG": "highprio"}))
    s = world.run_pass(); assert s["started"] == 1 and s["refused"] == 0, s
    hi = _digest(_ids_file(d, "highprio"))
    st2 = world.events("petri_drive_started")[-1]
    assert st2["auditor_org"] == "highprio" and st2["auditor_ids_sha12_expected"] == hi != lo and st2["overlay"] == ["F22_AUDITOR_ORG"]
    att2 = world.record()["attempts"][-1]; world.wait_exit(att2); world.run_pass()
    at2 = world.events("petri_drive_key_attested")[-1]
    assert at2["match"] is True and at2["org_attested"] == "highprio" and at2["sha12_logged"] == hi and at2["attempt"] == 2
    assert not world.events("problem")


def test_under_wif_a_forged_digest_or_a_key_attestation_is_a_problem_and_an_empty_ids_file_refuses_the_start(world, monkeypatch):
    """Under federation: a drive whose attested ids digest is not the named org's file as the daemon read it is a `problem`
    (petri_auditor_key_mismatch, auth wif), once; a drive at a wave sha still on the KEY path attesting a key hash under a wif daemon is a
    problem naming the other auth path; and an org whose ids file is empty (the stray wif-lowprio.env shape) refuses the start by name
    (`petri_drive_key_unresolved`, auth wif, the file named) -- nothing claimed, nothing spawned -- until the file is fixed. The refusal is a
    GATE: an attempt whose wif attestation is forged or a key's settles `unattested` -- never `completed` though its drive exited 0 with a
    complete result, the endpoint's coverage never credited, not counted toward the cap and never retried for that endpoint (one
    `petri_drive_unattested`, then no start) -- where before the problem was logged beside a `completed` attempt."""
    d = _wif_world(world, monkeypatch)
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    monkeypatch.setenv("FAKE_DRIVE_IDS_SHA12", "0" * 12)
    assert world.run_pass()["started"] == 1
    att = world.record()["attempts"][-1]; world.wait_exit(att); s1 = world.run_pass(); s2 = world.run_pass()
    (at,) = world.events("petri_drive_key_attested")
    assert at["match"] is False and at["sha12_logged"] == "0" * 12 and at["sha12_expected"] == _digest(_ids_file(d, "lowprio")) and at["auth_logged"] == "wif"
    (pb,) = world.events("problem")
    assert pb["kind"] == "petri_auditor_key_mismatch" and pb["auth_expected"] == "wif" and "federation ids the drive attested are not the named org's ids file" in pb["note"]
    assert "REFUSES the attempt" in pb["note"]
    rec = world.record(); a1 = rec["attempts"][-1]
    assert a1["key_attested"] is False and a1["state"] == "unattested" and a1["exit_code"] == 0, a1          # exit 0, a complete result -- and NOT completed
    assert "attestation refused under SISTEER_SECRETS_MODE=wif" in a1["reason"] and "never retried" in a1["reason"] and "re-freeze" in a1["reason"]
    assert a1["result"]["verdict"] == "complete" and a1.get("coverage") is None and a1.get("attestation_stop") is None     # the drive had exited: nothing to stop
    (ex1,) = world.events("petri_drive_exited"); assert ex1["state"] == "unattested" and "not counted toward the cap and not retried" in ex1["note"]
    key1 = GD._identity_key(a1["identity"])
    assert not (rec.get("endpoints") or {}).get(key1, {}).get("covered") and not (rec.get("endpoints") or {}).get(key1, {}).get("complete")
    assert GD._petri_verdict(rec, key1) == "unattested"
    assert s1["unattested"] == 1 and s1["started"] == 0 and s1["completed"] == 0 and s2["unattested"] == 1 and s2["started"] == 0, (s1, s2)
    (un,) = world.events("petri_drive_unattested"); assert un["attempt"] == 1 and un["log"] == a1["log"] and "re-freezes the wave" in un["note"]
    assert not world.events("petri_drive_gave_up") and len(world.drive_calls()) == 1                              # no retry, no cap
    monkeypatch.delenv("FAKE_DRIVE_IDS_SHA12")
    # a drive still on the key path under a wif daemon: the other auth, named -- and the same gate
    world.proxy_stops("11111"); world.endpoint_up("22222", 2000); world.job("22222", "RUNNING")
    keypath = world.tmp / "attest_key"; keypath.write_text(""); monkeypatch.setenv("FAKE_DRIVE_ATTEST_KEY_UNDER_WIF", str(keypath))
    assert world.run_pass()["started"] == 1
    att2 = world.record()["attempts"][-1]; world.wait_exit(att2); world.run_pass(); s4 = world.run_pass()
    at2 = world.events("petri_drive_key_attested")[-1]
    assert at2["match"] is False and at2["auth_expected"] == "wif" and at2["auth_logged"] == "key" and at2["org_attested"] is None and at2["attempt"] == 2
    pb2 = world.events("problem")[-1]
    assert pb2["kind"] == "petri_auditor_key_mismatch" and pb2["attempt"] == 2 and "the drive attested a key while the daemon runs under SISTEER_SECRETS_MODE=wif" in pb2["note"]
    a2 = world.record()["attempts"][-1]
    assert a2["n"] == 2 and a2["state"] == "unattested" and "the drive attested a key" in a2["reason"] and s4["unattested"] == 1 and s4["started"] == 0
    assert len(world.events("petri_drive_unattested")) == 2 and len(world.drive_calls()) == 2
    keypath.unlink(); monkeypatch.delenv("FAKE_DRIVE_ATTEST_KEY_UNDER_WIF")
    # an empty ids file: refused by name before the claim, nothing spawned, until the file is fixed
    world.proxy_stops("22222"); world.endpoint_up("33333", 3000); world.job("33333", "RUNNING")
    _ids_file(d, "lowprio").write_text("")
    n_att = len(world.record()["attempts"])
    s = world.run_pass()
    assert s["refused"] == 1 and s["started"] == 0, s
    (ku,) = world.events("petri_drive_key_unresolved")
    assert ku["org"] == "lowprio" and ku["mode"] == "wif" and ku["auth"] == "wif" and f"{_ids_file(d, 'lowprio')} is empty" in ku["error"] and "federation ids file" in ku["note"]
    assert len(world.record()["attempts"]) == n_att and len(world.drive_calls()) == 2
    _ids_file(d, "lowprio").write_text("".join(f"{k}={v}\n" for k, v in WIF_IDS.items()))
    assert world.run_pass()["started"] == 1
    assert world.events("petri_drive_started")[-1]["auditor_ids_sha12_expected"] == _digest(_ids_file(d, "lowprio"))


def test_under_wif_a_running_drive_whose_attestation_is_refused_is_stopped_before_it_files_and_never_completes(world, monkeypatch):
    """Round 1 on 7778694d0: the attestation was read while the drive ran and a mismatch logged as a problem, yet the drive went on auditing and
    FILING under a pool the daemon could not prove, and its exit 0 was then verified `completed`. Now the first pass that reads a refused wif
    attestation (here a key hash: a wave sha still on the key path, launched under a wif daemon) sends the running drive's session TERM at once
    (`petri_drive_stopped_unattested`, recorded on the attempt), so no leg files anything: the result summary is never written, the proxy's
    STOP never touched; the settlement states the attempt `unattested` (never `completed`, never `failed`-and-retried), the endpoint is left
    without coverage and no new attempt starts for it."""
    _wif_world(world, monkeypatch)
    world.endpoint_up("11111", 1000); world.job("11111", "RUNNING")
    keypath = world.tmp / "attest_key"; keypath.write_text(""); monkeypatch.setenv("FAKE_DRIVE_ATTEST_KEY_UNDER_WIF", str(keypath))
    world.hold.write_text("")                                             # the drive holds after DRIVE_START: its legs would be auditing now
    assert world.run_pass()["started"] == 1
    att = world.record()["attempts"][-1]
    end = time.time() + 15.0
    while time.time() < end and "DRIVE_START " not in Path(att["log"]).read_text():
        time.sleep(0.05)
    assert "auditor_key_sha12=" in Path(att["log"]).read_text()           # the drive attests a KEY under a wif daemon
    s = world.run_pass()                                                  # the pass that reads the attestation: the drive is stopped, the attempt still live
    assert s["running"] == 1 and s["started"] == 0, s
    (stop,) = world.events("petri_drive_stopped_unattested")
    assert stop["attempt"] == 1 and stop["pid"] == att["pid"] and stop["delivered"] is True and stop["key_attested"] is False and stop["org_expected"] == "lowprio"
    assert "the drive attested a key while the daemon runs under SISTEER_SECRETS_MODE=wif" in stop["reason"]
    live = world.record()["attempts"][-1]
    assert live["state"] == "started" and live["attestation_stop"]["signal"] == "SIGTERM" and live["attestation_stop"]["pid"] == att["pid"]
    (pb,) = world.events("problem"); assert pb["kind"] == "petri_auditor_key_mismatch" and pb["attempt"] == 1
    end = time.time() + 15.0
    while time.time() < end and GD._proc_alive(att["pid"], att["proc_start"]):
        time.sleep(0.05)
    assert not GD._proc_alive(att["pid"], att["proc_start"]), "the drive's session did not stop on TERM"
    s2 = world.run_pass()                                                 # settles: unattested, never completed; nothing was filed
    a = world.record()["attempts"][-1]
    assert a["state"] == "unattested" and a["exit_code"] != 0 and "attestation refused under SISTEER_SECRETS_MODE=wif" in a["reason"], a
    assert a.get("coverage") is None and a["result"] is None and not os.path.exists(att["result_file"])    # the drive never reached its verdict
    assert not (world.endpoint.parent / "STOP").exists()                 # nothing told the proxy the legs were done
    assert len(world.events("petri_drive_stopped_unattested")) == 1 and len(world.events("problem")) == 1
    ex = world.events("petri_drive_exited")[-1]; assert ex["state"] == "unattested" and ex["attempt"] == 1
    rec = world.record(); key = GD._identity_key(a["identity"])
    assert GD._petri_verdict(rec, key) == "unattested" and not (rec.get("endpoints") or {}).get(key, {}).get("covered")
    assert s2["unattested"] == 1 and s2["started"] == 0 and s2["completed"] == 0 and s2["running"] == 0, s2
    assert world.run_pass()["started"] == 0 and len(world.events("petri_drive_unattested")) == 1 and len(world.drive_calls()) == 1   # never retried
    world.hold.unlink(missing_ok=True)
