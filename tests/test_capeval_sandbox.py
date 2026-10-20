"""capeval sandbox tests.

All against STUBBED gcloud/ssh (GCLOUD=stub etc. — no cloud access):
  * wrapper: start-idempotence (RUNNING -> no start), IP re-resolution on
    EVERY ensure-up (never cached), bounded wait-ssh with retries and a
    loud deadline failure, provision idempotence (no apt/pull when probes
    pass; digest is the last stdout line), pull-on-missing-image,
    stop --verify refusing a non-TERMINATED end state.
  * exec wave: EXIT-trap stop fires on a mid-wave crash (never-fail
    semantics), --keep-up opts out loudly, no-cells -> VM never started.
  * the full happy-path wave with recorded command lines against a local
    simulation of the login host + sandbox: direct pull/push, the
    digest-pinned docker command, converter output — per-item pass@1
    rows, wave ts rows, results_pending.json merge, results.json written
    LAST (F8).
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalplus_synthetic as EP  # noqa: E402  (tests/ helper: the synthetic EvalPlus sample pair)

REPO = Path(__file__).parents[1]
SANDBOX_SH = REPO / "scripts" / "capeval_sandbox.sh"
EXEC_WAVE = REPO / "scripts" / "capeval_exec_wave.py"
DIGEST = ("ganler/evalplus@sha256:"
          "26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740")

GCLOUD_STUB = r"""#!/bin/bash
echo "GCLOUD $*" >> "$STUB_LOG"
case "$*" in
  *"--format=value(status)"*) cat "$STATE_DIR/status" ;;
  *"--format=value(networkInterfaces[0].networkIP)"*) cat "$STATE_DIR/ip" ;;
  *" start "*) echo RUNNING > "$STATE_DIR/status" ;;
  *" stop "*) [ -n "${STUB_STOP_NOOP:-}" ] || echo TERMINATED > "$STATE_DIR/status" ;;
esac
"""

SSH_STUB = r"""#!/bin/bash
echo "SSH $*" >> "$STUB_LOG"
if [ -f "$STATE_DIR/ssh_fail_count" ]; then
  n=$(cat "$STATE_DIR/ssh_fail_count")
  if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$STATE_DIR/ssh_fail_count"; exit 255; fi
fi
case "$*" in
  *"command -v docker"*) exit "${STUB_NO_DOCKER:-0}" ;;
  *"docker image inspect"*) exit "${STUB_NO_IMAGE:-0}" ;;
  *"apt-get"*|*"docker pull"*) exit 0 ;;
  *"test -s"*) exit 0 ;;
esac
exit 0
"""


def _write_exec(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture()
def stub_env(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "status").write_text("TERMINATED\n")
    (state / "ip").write_text("10.128.0.3\n")
    log = tmp_path / "stub.log"
    log.touch()
    env = dict(
        os.environ,
        GCLOUD=str(_write_exec(tmp_path / "gcloud", GCLOUD_STUB)),
        CAPEVAL_SSH=str(_write_exec(tmp_path / "ssh_stub", SSH_STUB)),
        STUB_LOG=str(log), STATE_DIR=str(state),
        CAPEVAL_SSH_POLL="0", CAPEVAL_SSH_DEADLINE="5",
        CAPEVAL_SANDBOX_USER="testuser",
    )
    return env, state, log


def _sandbox(env, *args, check=False):
    return subprocess.run(["bash", str(SANDBOX_SH), *args],
                          capture_output=True, text=True, env=env, timeout=60)


# ------------------------------------------------------------- wrapper


def test_ensure_up_starts_terminated_and_reports_ip(stub_env):
    env, state, log = stub_env
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "10.128.0.3"
    assert "instances start" in log.read_text()


def test_ensure_up_is_idempotent_when_running(stub_env):
    env, state, log = stub_env
    (state / "status").write_text("RUNNING\n")
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode == 0, proc.stderr
    assert "instances start" not in log.read_text()


def test_ip_re_resolved_every_ensure_up_never_cached(stub_env):
    env, state, log = stub_env
    p1 = _sandbox(env, "ensure-up")
    (state / "ip").write_text("10.128.0.77\n")  # IP changed across stop/start
    p2 = _sandbox(env, "ensure-up")
    assert p1.stdout.strip().splitlines()[-1] == "10.128.0.3"
    assert p2.stdout.strip().splitlines()[-1] == "10.128.0.77"
    assert log.read_text().count("networkInterfaces[0].networkIP") == 2


def test_wait_ssh_retries_until_up(stub_env):
    env, state, log = stub_env
    (state / "ssh_fail_count").write_text("2")
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode == 0, proc.stderr
    assert log.read_text().count("SSH ") == 3  # 2 failures + 1 success


def test_wait_ssh_deadline_fails_loud(stub_env):
    env, state, log = stub_env
    (state / "ssh_fail_count").write_text("999999")
    env = dict(env, CAPEVAL_SSH_DEADLINE="1")
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode != 0
    assert "not up within 1s" in proc.stderr


def test_provision_is_idempotent_and_prints_digest(stub_env):
    env, state, log = stub_env
    proc = _sandbox(env, "provision")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == DIGEST
    text = log.read_text()
    assert "apt-get install" not in text and "docker pull" not in text


def test_provision_pulls_image_when_missing_at_digest(stub_env):
    env, state, log = stub_env
    env = dict(env, STUB_NO_IMAGE="1")
    proc = _sandbox(env, "provision")
    assert proc.returncode == 0, proc.stderr
    assert f"docker pull '{DIGEST}'" in log.read_text()


def test_stop_verify_refuses_non_terminated(stub_env):
    env, state, log = stub_env
    (state / "status").write_text("RUNNING\n")
    env = dict(env, STUB_STOP_NOOP="1")
    proc = _sandbox(env, "stop", "--verify")
    assert proc.returncode != 0
    assert "not TERMINATED" in proc.stderr
    ok = _sandbox(dict(env, STUB_STOP_NOOP=""), "stop", "--verify")
    assert ok.returncode == 0 and "verified TERMINATED" in ok.stderr


def test_ensure_up_failure_stops_the_vm_it_started(stub_env):
    """Review infra #1: ensure-up started the VM but ssh never came up —
    the failure-only trap must stop it again (verified), so a direct
    `ensure-up` invocation can never leave a fresh VM billing."""
    env, state, log = stub_env
    (state / "ssh_fail_count").write_text("999999")
    env = dict(env, CAPEVAL_SSH_DEADLINE="1")
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode != 0
    text = log.read_text()
    assert "instances start" in text
    assert "instances stop" in text          # the failure trap fired
    assert (state / "status").read_text().strip() == "TERMINATED"
    assert "failure-stop complete (verified TERMINATED)" in proc.stderr


def test_ensure_up_failure_never_stops_a_vm_it_did_not_start(stub_env):
    """The failure trap is FAILURE-ONLY and START-ONLY: an already-RUNNING
    VM (someone else's session) is never stopped by a failing ensure-up."""
    env, state, log = stub_env
    (state / "status").write_text("RUNNING\n")
    (state / "ssh_fail_count").write_text("999999")
    env = dict(env, CAPEVAL_SSH_DEADLINE="1")
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode != 0
    text = log.read_text()
    assert "instances start" not in text
    assert "instances stop" not in text


def test_ensure_up_success_clears_the_failure_trap(stub_env):
    env, state, log = stub_env
    proc = _sandbox(env, "ensure-up")
    assert proc.returncode == 0
    assert "instances stop" not in log.read_text()
    assert (state / "status").read_text().strip() == "RUNNING"


# ------------------------------------------------------------- exec wave

CLUSTER_SSH_STUB = r"""#!/bin/bash
echo "CLUSTER_SSH $*" >> "$STUB_LOG"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
unset 'args[0]'                       # drop the host — execute locally
exec bash -c "${args[*]}"             # ssh semantics: space-join + shell
"""

SANDBOX_SCRIPT_STUB = r"""#!/bin/bash
sub="$1"; shift
echo "SANDBOX $sub $*" >> "$STUB_LOG"
case "$sub" in
  ensure-up) echo "10.128.0.99" ;;
  provision)
    [ -n "${STUB_PROVISION_FAIL:-}" ] && { echo "provision boom" >&2; exit 1; }
    echo "$STUB_DIGEST" ;;
  stop)
    if [ -n "${STUB_STOP_FAIL:-}" ]; then echo "stop boom" >&2; exit 1; fi
    echo "SANDBOX_STOPPED $*" >> "$STUB_LOG" ;;
  status) echo TERMINATED ;;
  run)
    [ "$1" = "--" ] && shift
    cmd="$*"
    cmd="${cmd//\~/$SANDBOX_HOME}"
    if [[ "$cmd" == *"ssh -o BatchMode=yes"* ]]; then exit "${STUB_NO_DIRECT:-0}"; fi
    if [[ "$cmd" == *"docker run"* ]]; then
      dir=$(echo "$cmd" | sed -n 's|.*-v \([^:]*\):/app.*|\1|p')
      cp "$STUB_EVAL_RESULTS" "$dir/samples_eval_results.json"
      exit 0
    fi
    cmd="${cmd//scp -q login-0:/cp }"
    cmd="${cmd// login-0:/ }"
    cmd="${cmd//scp -q /cp }"
    exec bash -c "$cmd"
    ;;
esac
"""

ROW_CONSTANTS = {
    "sweep": "proto_synthetic", "run_id": "run0", "model_id": "test-org/tiny-model",
    "model_revision": "0123456789abcdef0123456789abcdef01234567",
    "arm_name": "arm_alpha_c2", "kind": "steering", "coeff": 2.0,
    "intervention_hash": "a1b2c3d4e5f60718", "engine": "hf",
    "trial_id": 0, "seed": 100, "attempt": 0,
    "harness_config_hash": "0f1e2d3c4b5a6978", "env_digest": "99aabbccddeeff00",
    "slurm_job_id": None,
}


@pytest.fixture()
def wave_world(tmp_path):
    """A local simulation: cluster sweep dir + sandbox home + stubs."""
    log = tmp_path / "stub.log"
    log.touch()
    sweep = tmp_path / "cluster" / "sweep"
    cell = sweep / "cells" / "arm_alpha_c2__tier1__t0__a0__a1b2c3d4__0f1e2d3c"
    cell.mkdir(parents=True)
    (cell / "samples.jsonl").write_text(EP.samples_jsonl())
    (cell / "generation_complete.json").write_text("{}")
    (cell / "manifest.json").write_text(json.dumps({"row_constants": ROW_CONSTANTS}))
    (cell / "results_pending.json").write_text(json.dumps(
        {"phases": {"tier1": {"awaiting_execution": True}}}))
    sandbox_home = tmp_path / "sandbox_home"
    sandbox_home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _write_exec(bindir / "ssh", CLUSTER_SSH_STUB)
    sandbox_script = _write_exec(tmp_path / "sandbox_stub.sh", SANDBOX_SCRIPT_STUB)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        STUB_LOG=str(log), SANDBOX_HOME=str(sandbox_home),
        STUB_DIGEST=DIGEST,
        STUB_EVAL_RESULTS=str(EP.write(tmp_path / "evalplus_pair")[1]),
    )
    return env, sweep, cell, sandbox_script, log


def _wave(env, sweep, sandbox_script, *extra):
    return subprocess.run(
        [sys.executable, str(EXEC_WAVE), "--sweep-dir", str(sweep),
         "--sandbox-script", str(sandbox_script),
         "--expect-n-tasks", "2",  # canned 2-task fixture (test seam)
         "--cluster-host", "login-0",  # a synthetic login host (the stub runs every transfer locally)
         *extra],
        capture_output=True, text=True, env=env, timeout=120)


def test_wave_happy_path_direct_transfers_and_conversion(wave_world):
    env, sweep, cell, sandbox_script, log = wave_world
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    # F8 discipline: results.json written, rows appended
    results = json.loads((cell / "results.json").read_text())
    exec_phase = results["phases"]["evalplus_execution"]
    assert exec_phase["n_tasks"] == 2
    assert exec_phase["pass@1_base"] == 0.5 and exec_phase["pass@1_plus"] == 0.5
    assert results["phases"]["tier1"]["awaiting_execution"] is True  # pending merged
    rows = [json.loads(l) for l in (cell / "rows.jsonl").read_text().splitlines()]
    metrics = sorted(r["metric"] for r in rows)
    assert metrics.count("pass@1_base") == 2 and metrics.count("pass@1_plus") == 2
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert by[("HumanEval/0", "pass@1_base")]["value"] == 1.0
    assert by[("HumanEval/1", "pass@1_plus")]["value"] == 0.0
    # simplify-pass #7: the normalized GATING row the code domain pairs on
    assert metrics.count("acc") == 2
    assert by[("HumanEval/0", "acc")]["value"] == 1.0
    assert by[("HumanEval/1", "acc")]["value"] == 0.0
    assert json.loads(by[("HumanEval/0", "acc")]["extra_json"]) == {
        "source_metric": "pass@1_plus", "metric_def": "base_and_plus"}
    assert json.loads(by[("HumanEval/0", "pass@1_plus")]["extra_json"]) == {"metric_def": "base_and_plus"}
    assert exec_phase["metric_def"] == "base_and_plus"                         # PW12: the definition is named
    # wave timestamp rows (F7 + review infra #15): wave start + provision
    # are first-class rows alongside the transfer/exec stamps
    ts_metrics = {r["metric"] for r in rows if r["metric"].startswith("ts_")}
    assert {"ts_wave_start", "ts_provision", "ts_exec_start", "ts_exec_end",
            "ts_transfer_in", "ts_transfer_out"} <= ts_metrics
    for r in rows:  # every row fully schema'd from row_constants
        assert r["domain"] == "code" and r["task"] == "humaneval_plus"
        assert r["engine"] == "hf" and r["intervention_hash"] == "a1b2c3d4e5f60718"

    text = log.read_text()
    assert f"docker run --rm -v ~/capeval_wave/{cell.name}:/app {DIGEST}" in text
    assert "evalplus.evaluate --dataset humaneval --samples /app/samples.jsonl" in text
    assert "SANDBOX_STOPPED --verify" in text          # trap/end stop
    assert "scp -q login-0:" in text                    # direct pull recorded
    # the wave itself never staged the payload (F19): the sandbox pulled it
    assert f"scp -q login-0:{cell}/samples.jsonl" in text


def test_wave_reconversion_never_duplicates_rows(wave_world):
    """Review infra #4: a retry after a crash between row-append and
    results.json (simulated by deleting results.json) dedupes against the
    existing rows.jsonl — pass rates can never double-count."""
    env, sweep, cell, sandbox_script, log = wave_world
    assert _wave(env, sweep, sandbox_script).returncode == 0
    rows_before = (cell / "rows.jsonl").read_text().splitlines()
    (cell / "results.json").unlink()  # simulate: appended but not finalized
    assert _wave(env, sweep, sandbox_script).returncode == 0
    rows_after = (cell / "rows.jsonl").read_text().splitlines()
    assert rows_after == rows_before  # idempotent — zero new rows
    assert (cell / "results.json").is_file()


def test_wave_refuses_unpinned_provision_digest(wave_world):
    """Review infra #6: the provision-reported digest must be byte-equal
    to the pinned one — otherwise the wave refuses BEFORE any cell runs
    (and still stops the VM via the trap)."""
    env, sweep, cell, sandbox_script, log = wave_world
    env = dict(env, STUB_DIGEST="ganler/evalplus@sha256:" + "ab" * 32)
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode != 0
    assert "pinned" in proc.stderr
    text = log.read_text()
    assert "docker run" not in text                # no cell executed
    assert "SANDBOX_STOPPED --verify" in text      # trap still stopped it
    assert not (cell / "results.json").exists()


def test_wave_rejects_non_canonical_cell_names(wave_world):
    """Review infra #3: a crafted cell dir name must never reach a remote
    shell — the wave refuses before starting the VM."""
    env, sweep, cell, sandbox_script, log = wave_world
    evil = sweep / "cells" / "x; touch pwned #__tier1__t0__a0__aaaaaaaa__bbbbbbbb"
    evil.mkdir(parents=True)
    for f in ("generation_complete.json", "results_pending.json"):
        (evil / f).write_text("{}")
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode != 0
    assert "canonical cell-key" in proc.stderr
    assert "SANDBOX ensure-up" not in log.read_text()  # VM never started


def test_wave_requires_results_pending_to_execute(wave_world):
    """Review tier1 #3 / infra #5: generation_complete alone is not
    enough — a cell without results_pending.json is not awaiting (the
    driver commits pending BEFORE the marker), so nothing executes."""
    env, sweep, cell, sandbox_script, log = wave_world
    (cell / "results_pending.json").unlink()
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode == 0
    assert "no awaiting cells" in proc.stdout
    assert "SANDBOX ensure-up" not in log.read_text()


def test_converter_refuses_missing_results_pending(wave_world):
    """Review infra #5: the converter itself refuses a cell without the
    generation-phase summary — it never substitutes {} and finalizes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wave_mod", EXEC_WAVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env, sweep, cell, sandbox_script, log = wave_world
    (cell / "samples_eval_results.json").write_text(
        json.dumps(EP.eval_results()))
    (cell / "results_pending.json").unlink()
    proc = subprocess.run(
        [sys.executable, "-", str(cell), "{}", "2"],
        input=mod.CONVERTER_PY, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "results_pending.json missing" in proc.stderr
    assert not (cell / "results.json").exists()
    # and a wrong task count refuses too (tier1 #4: the pinned 164 set)
    (cell / "results_pending.json").write_text("{}")
    proc2 = subprocess.run(
        [sys.executable, "-", str(cell), "{}", "164"],
        input=mod.CONVERTER_PY, capture_output=True, text=True, timeout=60)
    assert proc2.returncode != 0 and "pinned 164" in proc2.stderr


def test_converter_scores_plus_as_base_and_plus(wave_world):
    """PW12 (tinycap E19, NEW High): ``pass@1_plus`` is evalplus 0.3.1's own
    headline — a sample passes the plus set only when it passes the BASE
    tests AND the plus tests (``evaluate.py`` ``new_correct``). This
    converter scored ``plus_status`` alone until PW12 while the served
    converter had moved to base AND plus (PW10), so the same
    ``pass@1_plus`` / ``acc`` / ``evalplus-0.3.1`` identity carried two
    definitions. Every converter is now base AND plus and NAMES the
    definition (``metric_def``) on the rows and in the results phase."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wave_mod", EXEC_WAVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env, sweep, cell, sandbox_script, log = wave_world
    results = EP.eval_results()
    results["eval"]["HumanEval/1"][0].update(base_status="fail", plus_status="pass")   # the mixed item
    (cell / "samples_eval_results.json").write_text(json.dumps(results))
    proc = subprocess.run([sys.executable, "-", str(cell), "{}", "2"], input=mod.CONVERTER_PY,
                          capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    by = {(r["item_id"], r["metric"]): r for r in rows}
    assert by[("HumanEval/1", "pass@1_base")]["value"] == 0.0
    assert by[("HumanEval/1", "pass@1_plus")]["value"] == 0.0 == by[("HumanEval/1", "acc")]["value"]
    assert by[("HumanEval/0", "pass@1_plus")]["value"] == 1.0 == by[("HumanEval/0", "acc")]["value"]
    assert json.loads(by[("HumanEval/1", "pass@1_plus")]["extra_json"]) == {"metric_def": "base_and_plus"}
    assert json.loads(by[("HumanEval/1", "acc")]["extra_json"]) == {
        "source_metric": "pass@1_plus", "metric_def": "base_and_plus"}
    phase = json.loads((cell / "results.json").read_text())["phases"]["evalplus_execution"]
    assert phase["pass@1_base"] == 0.5 and phase["pass@1_plus"] == 0.5 and phase["metric_def"] == "base_and_plus"
    # the definition is identity on the row: the pin + the definition, then the dataset hash (PW13)
    assert {r["task_version"] for r in rows} == {"evalplus-0.3.1+base_and_plus/" + str(results.get("hash", "nohash"))}
    # the literals the stdlib converter types ARE the contract's names (it imports no sisteer)
    from sisteer.capeval.tinycap import contracts as C
    assert f'{{"metric_def": "{C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS}"}}' in mod.CONVERTER_PY
    assert 'p = 1.0 if (b and item["plus_status"] == "pass") else 0.0' in mod.CONVERTER_PY
    assert 'p = 1.0 if item["plus_status"] == "pass" else 0.0' not in mod.CONVERTER_PY


def test_converter_refuses_rows_under_another_definition(wave_world):
    """PW13 (tinycap E21 M2, the HF path's twin): the converter dedupes on
    ``(task, item_id, metric)``, which cannot see the definition — the rows
    it wrote before PW13 (``evalplus-0.3.1/<hash>``: plus_only until PW12,
    base_and_plus under the old label after) left in a cell that never got
    its ``results.json`` would be skipped and the cell finalized under the
    new definition. Existing ``humaneval_plus`` rows under another
    ``task_version`` now REFUSE the conversion, nothing written; the
    converter's own rows still resume and dedupe."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("wave_mod", EXEC_WAVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env, sweep, cell, sandbox_script, log = wave_world
    results = EP.eval_results()
    (cell / "samples_eval_results.json").write_text(json.dumps(results))
    old_tv = "evalplus-0.3.1/" + str(results.get("hash", "nohash"))
    new_tv = "evalplus-0.3.1+base_and_plus/" + str(results.get("hash", "nohash"))
    old_rows = [{**ROW_CONSTANTS, "tier": 1, "task": "humaneval_plus", "task_version": old_tv, "domain": "code",
                 "item_id": "HumanEval/0", "position_order": None, "metric": m, "value": 1.0, "extra_json": None,
                 "created_at": 1.0, "wall_clock_s": None} for m in ("pass@1_base", "pass@1_plus", "acc")]
    (cell / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in old_rows))
    before = (cell / "rows.jsonl").read_bytes()

    def convert():
        return subprocess.run([sys.executable, "-", str(cell), "{}", "2"], input=mod.CONVERTER_PY,
                              capture_output=True, text=True, timeout=60, check=False)

    proc = convert()
    assert proc.returncode != 0
    assert "REFUSING" in proc.stderr and "another task_version" in proc.stderr
    assert old_tv in proc.stderr and new_tv in proc.stderr and "NEW cell identity" in proc.stderr
    assert (cell / "rows.jsonl").read_bytes() == before and not (cell / "results.json").exists()
    # the literal the converter writes names the definition (identity on the row, PW13)
    assert 'tv = "evalplus-0.3.1+base_and_plus/" + str(ev.get("hash", "nohash"))' in mod.CONVERTER_PY
    # its own rows resume: a crash between the append and results.json (the reconversion case) still dedupes
    (cell / "rows.jsonl").unlink()
    assert convert().returncode == 0
    rows = [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert {r["task_version"] for r in rows} == {new_tv} and len(rows) == 6
    (cell / "results.json").unlink()
    assert convert().returncode == 0
    assert [json.loads(line) for line in (cell / "rows.jsonl").read_text().splitlines()] == rows
    assert (cell / "results.json").is_file()


def test_wave_unverified_stop_forces_nonzero_exit(wave_world):
    """Review infra #2: all cells green but the stop never verifies ->
    the wave retries boundedly, is LOUD, and exits nonzero (a green wave
    must imply a TERMINATED VM)."""
    env, sweep, cell, sandbox_script, log = wave_world
    env = dict(env, STUB_STOP_FAIL="1")
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode != 0
    assert "UNVERIFIED" in proc.stderr and "BILLING" in proc.stderr
    assert proc.stderr.count("stop attempt") >= 3      # bounded retries
    # the cells themselves succeeded — only the stop forces the failure
    assert (cell / "results.json").is_file()


def test_wave_no_awaiting_cells_never_starts_vm(wave_world):
    env, sweep, cell, sandbox_script, log = wave_world
    (cell / "results.json").write_text("{}")  # cell already done
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode == 0
    assert "VM not started" in proc.stdout
    assert "SANDBOX ensure-up" not in log.read_text()


def test_wave_exit_trap_stops_vm_on_mid_wave_crash(wave_world):
    env, sweep, cell, sandbox_script, log = wave_world
    env = dict(env, STUB_PROVISION_FAIL="1")
    proc = _wave(env, sweep, sandbox_script)
    assert proc.returncode != 0
    text = log.read_text()
    assert "SANDBOX ensure-up" in text
    assert "SANDBOX_STOPPED --verify" in text  # the EXIT trap fired


def test_wave_keep_up_skips_stop_and_reminds_loudly(wave_world):
    env, sweep, cell, sandbox_script, log = wave_world
    env = dict(env, STUB_PROVISION_FAIL="1")
    proc = _wave(env, sweep, sandbox_script, "--keep-up")
    assert proc.returncode != 0
    assert "SANDBOX_STOPPED" not in log.read_text()
    assert "LEFT RUNNING AND BILLING" in proc.stderr
