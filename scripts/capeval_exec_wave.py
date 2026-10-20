"""capeval EvalPlus execution wave — grading-host-side orchestrator.

Runs AFTER a generation-producing array completes. Flow (F19: the grading
host orchestrates over ssh and NEVER stages payloads; transfers are
login host <-> sandbox DIRECT — the sandbox carries the operator's ~/.ssh
incl. config, so the login host resolves there):

  1. list awaiting cells on the cluster (generation_complete.json present,
     results.json absent — the F8 marker pair).
  2. if any: ensure-up the sandbox VM (EXIT trap installed FIRST: a
     crashed wave can never leave the VM billing — never-fail stop with
     TERMINATED verification; --keep-up opts out loudly).
  3. provision re-check (docker, digest-pinned image, evalplus data seed);
     the printed digest must equal the pinned one.
  4. [impl-verify] sandbox->login-host ssh probe; on failure fall back to
     per-cell streaming THROUGH the grading host with immediate cleanup
     (loud — never accumulation, R10).
  5. per cell: sandbox pulls samples.jsonl from the login host ->
     digest-pinned `docker run … evalplus.evaluate` -> sandbox pushes
     *_eval_results.json back -> a login-host-side stdlib converter
     appends per-item rows (the pass@1_base/plus per-task
     conversion; pass@1_plus = base AND plus, evalplus's own headline,
     named metric_def=base_and_plus in extra_json — PW12 — and in the row
     task_version, evalplus-0.3.1+base_and_plus/<dataset hash>, so the
     definition is identity on the row; existing rows of the task under
     another task_version REFUSE the conversion — PW13) + the wave's
     start/provision/exec/transfer timestamp rows (F7) to rows.jsonl,
     merges the driver's results_pending.json, and writes results.json
     LAST via tmp+fsync+replace (the only cell-done marker).
  6. stop the VM (the trap; also explicit, idempotent).

INTEGRATION NOTE: an EvalPlus generation cell's manifest.json
must carry ``row_constants`` — the ROW_COLUMNS-constant fields (sweep,
run_id, model_id, model_revision, arm_name, kind, coeff,
intervention_hash, engine, trial_id, seed, attempt, harness_config_hash,
env_digest) — and the driver leaves ``results_pending.json`` beside it
(scripts/capeval_run.py writes it when a tier reports
``awaiting_execution``). The converter refuses loudly when either is
missing; nothing is guessed.

Usage (grading host):
  python3 scripts/capeval_exec_wave.py --sweep-dir \
      $CLUSTER_REPO_MIRROR/out/capeval/<sweep> [--keep-up]
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
SANDBOX_SH = str(HERE / "capeval_sandbox.sh")

# The evaluator image digest pinned at provisioning (part of
# harness_config_hash, §5.5). The wave REFUSES to execute unless the
# provision step's printed digest is byte-for-byte this string: an env
# override or stale sandbox script must never stamp results under an
# unregistered image while the config hash still represents the pinned
# one.
PINNED_IMAGE_DIGEST = ("ganler/evalplus@sha256:"
                       "26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740")

# HumanEval+ problem count (evalplus 0.3.1 / v0.1.10; tier1 #4). The
# converter refuses any other task-ID set. Overridable ONLY via the
# hidden --expect-n-tasks test seam (canned 2-task fixtures).
EXPECTED_N_TASKS = 164

# Canonical cell-key shape (store.cell_key):
#   {arm}__{tier_or_task}__t{trial}__a{attempt}__{ihash8}__{chash8}
# Anything else is REJECTED before any remote interpolation (review
# infra #3): cell basenames come from cluster directory listings and are
# interpolated into remote shell commands — a crafted name must never
# execute on the sandbox (which holds a cluster-authorized ssh identity).
CELL_KEY_RE = re.compile(
    r"^[A-Za-z0-9._-]+__[A-Za-z0-9._-]+__t\d+__a\d+"
    r"__[0-9a-f]{8}__[0-9a-f]{8}$")

# The frozen §5.5 conversion + results.json discipline, executed ON the
# cluster host (stdlib only; python3 - <celldir> <wave_ts_json> <n_tasks>
# reads this program from stdin). Kept in one string so the wave has no
# cluster-side repo dependency. Concurrency discipline (review infra #4):
# an exclusive flock over the whole conversion, an already-done check
# under the lock, and per-(task,item,metric) dedupe against the existing
# rows.jsonl — so overlapping waves and crash-retries can never
# double-append rows or double-finalize.
CONVERTER_PY = r"""
import fcntl, json, os, sys, time, glob
celldir, wave_ts, n_expected = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
lock = open(os.path.join(celldir, ".convert.lock"), "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(f"REFUSING: another conversion holds {celldir}/.convert.lock "
             "(overlapping wave) - this cell stays retryable")
if os.path.isfile(os.path.join(celldir, "results.json")):
    print("ALREADY CONVERTED: results.json exists; nothing to do")
    sys.exit(0)
manifest_path = os.path.join(celldir, "manifest.json")
if not os.path.isfile(manifest_path):
    sys.exit(f"REFUSING: {manifest_path} missing - cannot construct rows")
manifest = json.load(open(manifest_path))
rc = manifest.get("row_constants")
if not rc:
    sys.exit("REFUSING: manifest.json has no row_constants (WS-A writes it "
             "at generation time; see capeval_exec_wave.py docstring)")
pending_path = os.path.join(celldir, "results_pending.json")
if not os.path.isfile(pending_path):
    sys.exit("REFUSING: results_pending.json missing - the driver commits it "
             "before generation_complete.json; a cell without it has no "
             "generation-phase summary to merge (integration contract)")
summary = json.load(open(pending_path))
results_files = sorted(glob.glob(os.path.join(celldir, "*_eval_results.json")))
if len(results_files) != 1:
    sys.exit(f"REFUSING: expected exactly one *_eval_results.json in {celldir}, "
             f"found {results_files}")
ev = json.load(open(results_files[0]))
if "eval" not in ev:
    sys.exit(f"REFUSING: {results_files[0]} has no 'eval' key - evalplus "
             "output shape drifted; route through the integrator")
expected_ids = {f"HumanEval/{i}" for i in range(n_expected)}
got_ids = set(ev["eval"])
if got_ids != expected_ids:
    sys.exit(f"REFUSING: eval covers {len(got_ids)} task ids, expected the "
             f"pinned {n_expected} HumanEval+ set (missing e.g. "
             f"{sorted(expected_ids - got_ids)[:5]}, unexpected e.g. "
             f"{sorted(got_ids - expected_ids)[:5]}) - a partial eval never "
             "finalizes a cell")
ROW_COLUMNS = ("sweep","run_id","model_id","model_revision","arm_name","kind",
               "coeff","intervention_hash","engine","tier","task","task_version",
               "domain","item_id","trial_id","seed","attempt","position_order",
               "metric","value","extra_json","harness_config_hash","env_digest",
               "slurm_job_id","created_at","wall_clock_s")
def row(**kw):
    base = dict.fromkeys(ROW_COLUMNS)
    base.update({k: rc.get(k) for k in rc})
    base.update(tier=1, task="humaneval_plus", task_version=kw.pop("task_version"),
                domain="code", position_order=None, slurm_job_id=rc.get("slurm_job_id"),
                created_at=time.time(), wall_clock_s=None, extra_json=None)
    base.update(kw)
    missing = [k for k in ROW_COLUMNS if k not in ("coeff","position_order",
               "extra_json","slurm_job_id","wall_clock_s") and base[k] is None]
    if missing:
        sys.exit(f"REFUSING: row_constants leave required columns unset: {missing}")
    return {k: base[k] for k in ROW_COLUMNS}
# the pass@1_plus DEFINITION is identity on the row (PW13 / tinycap E21 M1): the
# framework pin + the definition, then the dataset hash; the rows this converter
# wrote before PW13 carry the bare pin ("evalplus-0.3.1/<hash>") and are plus_only
# (until PW12) or base_and_plus under the old label (PW12) — either refuses below
tv = "evalplus-0.3.1+base_and_plus/" + str(ev.get("hash", "nohash"))
n_pass_base = n_pass_plus = n = 0
rows = []
for task_id, items in sorted(ev["eval"].items()):
    if len(items) != 1:
        sys.exit(f"REFUSING: {task_id} has {len(items)} samples - v0 is one "
                 "greedy sample per task (plan 5.5)")
    item = items[0]
    for key in ("base_status", "plus_status"):
        if key not in item:
            sys.exit(f"REFUSING: {task_id} eval item lacks {key} - evalplus "
                     "output shape drifted")
    b = 1.0 if item["base_status"] == "pass" else 0.0
    # evalplus's headline (0.3.1 evaluate.py new_correct): plus passes only if
    # base does too; the DEFINITION is named on the row so the plus-only rows
    # this converter wrote before PW12 never share its identity
    p = 1.0 if (b and item["plus_status"] == "pass") else 0.0
    plus_def = {"metric_def": "base_and_plus"}
    rows.append(row(item_id=task_id, metric="pass@1_base", value=b, task_version=tv))
    rows.append(row(item_id=task_id, metric="pass@1_plus", value=p, task_version=tv,
                    extra_json=json.dumps(plus_def, sort_keys=True)))
    # the normalized gating row the code domain pairs on (value =
    # pass@1_plus, the registered gating metric - simplify-pass #7)
    rows.append(row(item_id=task_id, metric="acc", value=p, task_version=tv,
                    extra_json=json.dumps({"source_metric": "pass@1_plus", **plus_def},
                                          sort_keys=True)))
    n += 1; n_pass_base += b; n_pass_plus += p
for event, ts in sorted(wave_ts.items()):
    rows.append(row(item_id="-", metric="ts_" + event, value=float(ts), task_version=tv))
rows_path = os.path.join(celldir, "rows.jsonl")
seen = set()
foreign = {}
if os.path.isfile(rows_path):
    for line in open(rows_path):
        if line.strip():
            r0 = json.loads(line)
            seen.add((r0.get("task"), r0.get("item_id"), r0.get("metric")))
            if r0.get("task") == "humaneval_plus" and r0.get("task_version") != tv:
                foreign[r0.get("task_version")] = foreign.get(r0.get("task_version"), 0) + 1
if foreign:
    # PW13 (tinycap E21 M2, the HF path's twin): the dedupe below is on (task, item_id,
    # metric) and cannot see the definition; skipping rows written under another one
    # and then writing results.json under this one would label two definitions current
    sys.exit(f"REFUSING: {rows_path} already holds humaneval_plus rows under another "
             f"task_version ({foreign} rows by version; this converter writes {tv!r}, "
             "pass@1_plus = base AND plus) - a re-conversion would leave rows.jsonl under "
             "the old definition and results.json under the new one. The cell stands as "
             "written; a current-definition reading belongs in a NEW cell identity, never "
             "appended here (PW13)")
rows = [r for r in rows if (r["task"], r["item_id"], r["metric"]) not in seen]
with open(rows_path, "a") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
    f.flush(); os.fsync(f.fileno())
summary.setdefault("phases", {})["evalplus_execution"] = {
    "n_tasks": n, "pass@1_base": (n_pass_base / n) if n else None,
    "pass@1_plus": (n_pass_plus / n) if n else None, "metric_def": "base_and_plus",
    "eval_results_file": os.path.basename(results_files[0]),
    "wave_timestamps": wave_ts}
tmp = os.path.join(celldir, ".results.json.tmp")
with open(tmp, "w") as f:
    json.dump(summary, f, indent=1); f.flush(); os.fsync(f.fileno())
os.replace(tmp, os.path.join(celldir, "results.json"))  # done marker, LAST
print(f"CONVERTED {n} tasks ({len(rows)} new rows); results.json written")
"""


def _run(argv: List[str], *, input_text: Optional[str] = None,
         check: bool = True) -> subprocess.CompletedProcess:
    print(f"[wave] $ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, input=input_text, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(argv)}")
    return proc


class Wave:
    STOP_RETRIES = 3  # bounded retries per stop_vm call (review infra #2)

    def __init__(self, args: argparse.Namespace) -> None:
        self.sweep_dir = args.sweep_dir.rstrip("/")
        self.cluster = args.cluster_host
        self.keep_up = args.keep_up
        self.sandbox_sh = args.sandbox_script
        self.expect_n_tasks = args.expect_n_tasks
        self._stopped = False       # ONLY after a VERIFIED TERMINATED stop
        self._stop_failed = False   # an unverified stop forces a nonzero exit
        self._started = False
        self._sandbox_ip: Optional[str] = None
        self._ts_wave_start: Optional[float] = None
        self._ts_provision: Optional[float] = None

    # -- transports ---------------------------------------------------------
    def cluster_ssh(self, cmd: str, *, input_text: Optional[str] = None,
                    argv_tail: Optional[List[str]] = None,
                    check: bool = True) -> subprocess.CompletedProcess:
        argv = ["ssh", "-o", "BatchMode=yes", self.cluster, cmd]
        if argv_tail:
            # ssh joins remote args with spaces and re-parses them through
            # the remote shell — quote each one or JSON/paths word-split.
            argv += [shlex.quote(a) for a in argv_tail]
        return _run(argv, input_text=input_text, check=check)

    def sandbox(self, *argv: str, check: bool = True) -> subprocess.CompletedProcess:
        return _run([self.sandbox_sh, *argv], check=check)

    # -- lifecycle ----------------------------------------------------------
    def stop_vm(self) -> bool:
        """Stop the VM with VERIFIED termination (review infra #2).

        ``_stopped`` is set only AFTER `stop --verify` confirms
        TERMINATED — never before the attempt — so the atexit backstop
        retries a failed stop instead of believing it already happened.
        Retries are bounded per call; an unverified stop is LOUD and
        flips ``_stop_failed``, which forces the wave's exit code
        nonzero (a green wave must imply a stopped VM). The stop itself
        never raises out of this method (a stop failure must not mask
        the wave's own result — it changes the exit code instead)."""
        if self._stopped or not self._started:
            return True
        if self.keep_up:
            print("[wave] --keep-up: SANDBOX VM LEFT RUNNING AND BILLING — "
                  "stop it with: scripts/capeval_sandbox.sh stop --verify",
                  file=sys.stderr, flush=True)
            return True
        last_err: Optional[Exception] = None
        for attempt in range(1, self.STOP_RETRIES + 1):
            try:
                self.sandbox("stop", "--verify", check=True)
                self._stopped = True
                self._stop_failed = False
                print("[wave] sandbox stopped (verified TERMINATED)")
                return True
            except Exception as e:  # noqa: BLE001 - retry, then loud failure
                last_err = e
                print(f"[wave] stop attempt {attempt}/{self.STOP_RETRIES} "
                      f"failed: {e}", file=sys.stderr, flush=True)
        self._stop_failed = True
        print(f"[wave] CRITICAL: sandbox stop UNVERIFIED after "
              f"{self.STOP_RETRIES} attempts ({last_err}) — the VM may still "
              "be BILLING. Stop it manually: "
              "scripts/capeval_sandbox.sh stop --verify. The wave will exit "
              "nonzero.", file=sys.stderr, flush=True)
        return False

    def install_trap(self) -> None:
        atexit.register(self.stop_vm)

        def _sig(handler_signum, _frame):
            self.stop_vm()
            sys.exit(128 + handler_signum)

        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, _sig)

    # -- steps ---------------------------------------------------------------
    def awaiting_cells(self) -> List[str]:
        """Cells awaiting execution: BOTH generation_complete.json AND
        results_pending.json present (the driver commits pending first,
        marker last — review tier1 #3), results.json absent. Every
        basename must match the canonical cell-key regex before it is
        ever interpolated into a remote command (review infra #3)."""
        cmd = (f"for d in {shlex.quote(self.sweep_dir)}/cells/*/; do "
               f'if [ -f "$d/generation_complete.json" ] && '
               f'[ -f "$d/results_pending.json" ] && '
               f'[ ! -f "$d/results.json" ]; '
               f'then echo "$d"; fi; done')
        proc = self.cluster_ssh(cmd)
        cells = [line.strip().rstrip("/") for line in proc.stdout.splitlines()
                 if line.strip()]
        for celldir in cells:
            key = Path(celldir).name
            if not CELL_KEY_RE.fullmatch(key):
                raise RuntimeError(
                    f"cell dir basename {key!r} does not match the canonical "
                    "cell-key shape (store.cell_key) — refusing to "
                    "interpolate it into any remote command (review infra #3)")
        return cells

    def sandbox_reaches_cluster(self) -> bool:
        """[impl-verify] sandbox->login-host direct ssh."""
        proc = self.sandbox("run", "--",
                            f"ssh -o BatchMode=yes -o ConnectTimeout=10 "
                            f"{self.cluster} true", check=False)
        ok = proc.returncode == 0
        verdict = "OK" if ok else (
            "FAILED — falling back to per-cell streaming through the grading host "
            "with immediate cleanup (F19 fallback)")
        print(f"[wave] sandbox->{self.cluster} direct ssh: {verdict}")
        return ok

    def run_cell(self, celldir: str, direct: bool, digest: str) -> None:
        key = Path(celldir).name
        if not CELL_KEY_RE.fullmatch(key):  # belt-and-suspenders (infra #3)
            raise RuntimeError(f"cell key {key!r} fails the canonical regex")
        work = f"capeval_wave/{key}"
        try:
            self._run_cell_inner(celldir, direct, digest, key, work)
        finally:
            # per-cell sandbox cleanup on EVERY path (failed cells must not
            # accumulate on the VM disk; the cell stays retryable — its
            # results.json only exists after a full conversion)
            self.sandbox("run", "--", f"rm -rf ~/{shlex.quote(work)}", check=False)

    def _run_cell_inner(self, celldir: str, direct: bool, digest: str,
                        key: str, work: str) -> None:
        # every interpolated remote fragment is shell-quoted (infra #3);
        # ~ stays outside the quotes so the remote shell still expands it
        q_cell = shlex.quote(celldir)
        q_work = shlex.quote(work)
        q_digest = shlex.quote(digest)
        # narrow the overlapping-wave window: skip if another wave already
        # finalized this cell (the converter re-checks under its flock)
        done = self.cluster_ssh(f"test -f {shlex.quote(celldir + '/results.json')}",
                                check=False)
        if done.returncode == 0:
            print(f"[wave] cell already finalized by another wave: {key}")
            return
        ts = {"wave_start": self._ts_wave_start or time.time(),
              "provision": self._ts_provision or time.time()}
        samples_remote = f"{celldir}/samples.jsonl"
        self.sandbox("run", "--", f"mkdir -p ~/{q_work}")
        if direct:
            self.sandbox("run", "--",
                         f"scp -q {shlex.quote(self.cluster)}:{shlex.quote(samples_remote)} ~/{q_work}/")
        else:
            self._relay_via_grading_host(f"{self.cluster}:{samples_remote}",
                                   f"{self._sandbox_scp_target()}:{work}/",
                                   local_name="samples.jsonl")
        ts["transfer_in"] = time.time()
        ts["exec_start"] = time.time()
        self.sandbox("run", "--",
                     f"cd ~/{q_work} && docker run --rm -v ~/{q_work}:/app {q_digest} "
                     f"evalplus.evaluate --dataset humaneval --samples /app/samples.jsonl")
        ts["exec_end"] = time.time()
        if direct:
            self.sandbox("run", "--",
                         f"scp -q ~/{q_work}/*_eval_results.json "
                         f"{shlex.quote(self.cluster)}:{shlex.quote(celldir + '/')}")
        else:
            self._relay_via_grading_host(
                f"{self._sandbox_scp_target()}:{work}/*_eval_results.json",
                f"{self.cluster}:{celldir}/samples_eval_results.json",
                local_name="eval_results.json")
        ts["transfer_out"] = time.time()
        # Convert + finalize ON the cluster host (results.json written LAST;
        # the converter holds the per-cell flock + dedupes — infra #4).
        self.cluster_ssh("python3", input_text=CONVERTER_PY,
                         argv_tail=["-", celldir,
                                    json.dumps(ts, separators=(",", ":")),
                                    str(self.expect_n_tasks)])
        print(f"[wave] cell done: {key}")

    def _relay_via_grading_host(self, src: str, dst: str, *, local_name: str) -> None:
        """F19 fallback: stream one payload through the grading host, per cell,
        with immediate cleanup (the tempdir dies with the transfer)."""
        with tempfile.TemporaryDirectory(prefix="capeval-wave-") as td:
            local = Path(td) / local_name
            _run(["scp", "-q", src, str(local)])
            _run(["scp", "-q", str(local), dst])

    def _sandbox_scp_target(self) -> str:
        # run() captured the IP from its own ensure-up; re-resolve only if
        # this is somehow called before it (the IP is never cached across
        # a stop/start — within one wave the VM stays up).
        ip = self._sandbox_ip
        if ip is None:
            proc = self.sandbox("ensure-up")
            ip = proc.stdout.strip().splitlines()[-1]
        return f"{os.environ.get('CAPEVAL_SANDBOX_USER', os.environ.get('USER'))}@{ip}"

    def run(self) -> int:
        self._ts_wave_start = time.time()  # actual wave start (infra #15)
        cells = self.awaiting_cells()
        if not cells:
            print("[wave] no awaiting cells (generation_complete + "
                  "results_pending without results) — VM not started.")
            return 0
        print(f"[wave] {len(cells)} awaiting cell(s):")
        for c in cells:
            print(f"[wave]   {c}")
        self.install_trap()
        self._started = True          # from here the trap owns the stop
        up = self.sandbox("ensure-up")
        self._sandbox_ip = up.stdout.strip().splitlines()[-1]
        proc = self.sandbox("provision")
        digest = proc.stdout.strip().splitlines()[-1]
        if digest != PINNED_IMAGE_DIGEST:
            # refuse BEFORE any cell executes (review infra #6): results
            # must never be stamped under an unregistered evaluator image.
            raise RuntimeError(
                f"provision reported image digest {digest!r} != the pinned "
                f"{PINNED_IMAGE_DIGEST!r} — refusing to execute; fix the "
                "sandbox provisioning (or the pin) rather than running an "
                "unregistered evaluator image")
        self._ts_provision = time.time()
        direct = self.sandbox_reaches_cluster()
        failures = 0
        for celldir in cells:
            try:
                self.run_cell(celldir, direct, digest)
            except Exception as e:  # noqa: BLE001 - one cell never kills the wave
                failures += 1
                print(f"[wave] CELL FAILED ({celldir}): {e}", file=sys.stderr)
        self.stop_vm()
        print(f"[wave] complete: {len(cells) - failures}/{len(cells)} cells "
              f"(provision done at {self._ts_provision:.0f})")
        # an UNVERIFIED stop makes the wave fail even with all cells green
        # (review infra #2): exit 0 must imply the VM is TERMINATED.
        return 1 if (failures or self._stop_failed) else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sweep-dir", required=True,
                    help="sweep dir ON THE CLUSTER (…/out/capeval/<sweep>)")
    ap.add_argument("--cluster-host", default=None,
                    help="the cluster's login host (default: CLUSTER_SSH of the "
                         "site config's cluster block)")
    ap.add_argument("--sandbox-script", default=SANDBOX_SH,
                    help=argparse.SUPPRESS)  # test seam
    ap.add_argument("--expect-n-tasks", type=int, default=EXPECTED_N_TASKS,
                    help=argparse.SUPPRESS)  # test seam (canned 2-task fixture)
    ap.add_argument("--keep-up", action="store_true",
                    help="do NOT stop the VM on exit (interactive debugging; "
                         "prints a loud billing reminder)")
    args = ap.parse_args(argv)
    if not args.cluster_host:
        args.cluster_host = site_login_host()
    return Wave(args).run()


def site_login_host() -> str:
    """The login host of the job's cluster (SISTEER_CLUSTER_LABEL), else of
    the site's primary cluster: its block's CLUSTER_SSH in the site config
    (sisteer.site); refused by name when the site names none."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site as _site
    s = _site.load_site()
    label = os.environ.get("SISTEER_CLUSTER_LABEL") or s.primary
    host = ((s.block(label) if label else None) or {}).get("CLUSTER_SSH")
    if not host:
        raise SystemExit("capeval_exec_wave: no --cluster-host, and the site "
                         f"config ({s.path or 'none on this host'}) names no "
                         "login host (CLUSTER_SSH) for its cluster")
    return host


if __name__ == "__main__":
    sys.exit(main())
