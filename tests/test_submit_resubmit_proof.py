"""`submit --resubmit`'s FENCED PROOF over every task of the bound job (paper-companion-repo stage E, 2026-09-30).


* (5) a REGISTERED binding (06 --gate bound it from the job's pending file) is decided by the same proof and RETIRED before the new
  attempt is bound — gate → die → rerun → the gate reads the NEW attempt (before: refused, and a new `submitted` binding could never
  supersede the stale registered one);
* (6) the proof fails CLOSED: no registry database → refused; the cluster's ingest watermark must have passed the dead job's END
  (records in transit would be upserted under the wrong generation); the pending file's run ids are looked up in the registry and a
  run past `running` refuses by name;
* (7) an ARRAY's children register under their RAW ids: every task is enumerated from sacct and fenced on its own; a live task skips
  the unit; a COMPLETED task refuses by name (with the `--array=<dead tasks>` a partial re-run would need — a human's decision, the
  gate reads one binding per unit); every task dead → one re-run of the whole unit under the same identities;
* (iv) the REGISTRATION CONTRACT (`--registers registry|none`, kind `other` → none): a unit that registers nothing is provable at
  sacct-terminal alone — no spool, no watermark — said by name in the proof output; a record written before the field resolves by kind.
Executed against the funnel's real CLI with the front-door fake Slurm (arrays included), a synthetic registry and site."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, array_tasks, job_ids, make_registry, run_f22_registration, set_job_state, synthetic_checkout  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

MMLU = ["submit", "--unit", "mmlu", "--lwid", "m/s1/mmlu/mmlu", "--script", "scripts/sdfbase/full22/mmlu_full.sbatch", "--name", "mmluF-m-s1",
        "--args", f"m s1 {SHA}", "--clusters", "cluster_a", "--wave-sha", SHA]
CLIMB = ["submit", "--unit", "bon.climb", "--lwid", "m/s1/bon/climb", "--script", "scripts/sdfbase/full22/bon_climb.sbatch", "--name", "f22-bonclimb-m-s1",
         "--args", f"m s1 {SHA} 4", "--clusters", "cluster_a", "--wave-sha", SHA, "--array", "0-3"]
PROBE = ["submit", "--unit", "hpo.lora.probe", "--lwid", "m/03_hpo/lora/hpo.lora.probe", "--script", "scripts/sbatch/serving_probe.sbatch",
         "--name", "hpo-m-lora-probe", "--args", "m org/toy " + "0" * 40, "--clusters", "cluster_a", "--wave-sha", SHA, "--kind", "other"]


def _attempts(w):
    return [e for e in w.ledger() if e["event"] == "attempt"]


def _pend(w, lab, raw, run_ids):
    d = w.shared[lab] / "registry/pending"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{raw}.json").write_text(json.dumps({"job_id": raw, "runs": [{"run_id": r} for r in run_ids]}))


def _bind_registered(w, lwid, run_ids):
    """What 06 --gate does: bind the unit `registered` from its job's pending file."""
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "bind", "--ledger", str(w.state / "submissions.jsonl"), "--logical-work-id", lwid,
                        "--state", "registered", "--run-ids", ",".join(run_ids)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def _binding(w, stem):
    p = w.state / "bindings"
    return {q.name: json.loads(q.read_text()) for q in p.glob(f"{stem}.*")}


def _certify_fit_venv(w):
    venv = w.shared["cluster_a"] / "envs" / "sdfbase-fit"; venv.mkdir(parents=True, exist_ok=True)
    (venv / "BUILD-COMPLETE").write_text("2026-09-16T05:09:17+00:00 40b49ecf00359e44\n")
    (w.shared["cluster_a"] / "envs" / "sdfbase-fit.build").mkdir(exist_ok=True)
    (w.shared["cluster_a"] / "envs" / "sdfbase-fit.build" / "requirements.txt").write_text("torch==2.11.0\n")


# ----------------------------------------------------------------------------------------------------- (5) gate -> die -> rerun -> gate
def test_a_registered_binding_of_a_dead_job_is_retired_and_the_rerun_is_what_the_gate_reads_next(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r1 = w.run(*MMLU)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    _pend(w, "cluster_a", "1001", ["run-a", "run-b"])
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "running", "cluster_job_id": "1001"},
                                            {"run_id": "run-b", "lifecycle_state": "running", "cluster_job_id": "1001"}])
    _bind_registered(w, "m/s1/mmlu/mmlu", ["run-a", "run-b"])                        # the gate ran while the job was RUNNING
    b = _binding(w, "m__s1__mmlu__mmlu"); assert set(b) == {"m__s1__mmlu__mmlu.submitted.json", "m__s1__mmlu__mmlu.registered.json"}
    # live: skipped, the registered binding untouched
    r2 = w.run(*MMLU, "--resubmit")
    assert r2.returncode == 0 and "skipped cluster_a 1001 mmlu (still RUNNING" in r2.stdout and job_ids(w.fake["cluster_a"]) == [1001], r2.stdout
    # the job dies; the registry has ingested past its end; the spool is traversable
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T00:20:00")
    r3 = w.run(*MMLU, "--resubmit")
    out = r3.stdout + r3.stderr
    assert r3.returncode == 0 and "submitted cluster_a 1002 mmlu" in r3.stdout, out
    assert "RERUN m/s1/mmlu/mmlu" in out and "retired the registered binding" in out and "REGISTERED before dying" in out
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.submitted.json"]["job_id"] == "1002" and "m__s1__mmlu__mmlu.registered.json" not in b
    retired = [k for k in b if ".registered.retired." in k]
    assert len(retired) == 1 and b[retired[0]]["job_id"] == "1001" and b[retired[0]]["run_ids"] == ["run-a", "run-b"]
    sup = [e for e in w.ledger() if e["event"] == "superseded_attempt"]
    assert len(sup) == 1 and sup[0]["job_id"] == "1001" and sup[0]["binding_state"] == "registered" and sup[0]["run_ids"] == ["run-a", "run-b"]
    # the gate binds the NEW attempt: binding_state reads job 1002
    _pend(w, "cluster_a", "1002", ["run-a", "run-b"])
    _bind_registered(w, "m/s1/mmlu/mmlu", ["run-a", "run-b"])
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.registered.json"]["job_id"] == "1002" and b["m__s1__mmlu__mmlu.registered.json"]["attempt_id"] == "m/s1/mmlu/mmlu@cluster_a@2"


# ----------------------------------------------------------------------------------------------------- (6) fail closed
def test_the_proof_fails_closed_on_a_missing_registry_and_a_run_past_running(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    assert w.run(*MMLU).returncode == 0
    _pend(w, "cluster_a", "1001", ["run-a"])
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_a")
    # no registry database at all: refused, nothing minted
    (w.tmp / "registry" / "registry.sqlite").unlink()
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "no registry database" in r.stderr and "fail closed" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    # the pending file's run is past `running` in the registry: refused BY NAME (a result, never overwritten)
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "generated", "cluster_job_id": "1001"}])
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "past `running`" in r.stderr and "run-a" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    # back to `running`: the rerun goes out
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "running", "cluster_job_id": "1001"}])
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 0 and "submitted cluster_a 1002 mmlu" in r.stdout and "RERUN" in r.stdout + r.stderr, r.stdout + r.stderr


# ----------------------------------------------------------------------------------------------------- (7) arrays: every child fenced
def test_an_arrays_children_are_fenced_one_by_one_and_only_an_all_dead_array_is_rerun(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r1 = w.run(*CLIMB)
    assert r1.returncode == 0 and "submitted cluster_a 1001 bon.climb" in r1.stdout, r1.stdout + r1.stderr
    tasks = dict((k, raw) for raw, k in array_tasks(w.fake["cluster_a"], "1001"))          # task index -> RAW child id
    assert sorted(tasks) == [0, 1, 2, 3] and tasks[0] == "1001"
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T09:00:00")
    # tasks 0-2 die after registering under their RAW ids (the live case: bon.climb 0-2 dead, 3 running) -> the live task skips the unit
    for k in (0, 1, 2):
        _pend(w, "cluster_a", tasks[k], [f"run-{k}"]); set_job_state(w.fake["cluster_a"], tasks[k], "FAILED", end=f"2026-01-01T0{k+1}:00:00")
    make_registry(w.tmp / "registry", runs=[{"run_id": f"run-{k}", "lifecycle_state": "running", "cluster_job_id": tasks[k]} for k in (0, 1, 2)])
    r2 = w.run(*CLIMB, "--resubmit")
    assert r2.returncode == 0 and f"skipped cluster_a 1001 bon.climb (still task 3 ({tasks[3]}) RUNNING" in r2.stdout and job_ids(w.fake["cluster_a"]) == sorted(int(x) for x in tasks.values()), r2.stdout + r2.stderr
    # task 3 COMPLETES with a generated run: the unit is refused by name, with the --array the dead tasks would need
    _pend(w, "cluster_a", tasks[3], ["run-3"]); set_job_state(w.fake["cluster_a"], tasks[3], "COMPLETED", end="2026-01-01T04:00:00")
    make_registry(w.tmp / "registry", runs=[{"run_id": f"run-{k}", "lifecycle_state": "running", "cluster_job_id": tasks[k]} for k in (0, 1, 2)]
                  + [{"run_id": "run-3", "lifecycle_state": "generated", "cluster_job_id": tasks[3]}])
    r3 = w.run(*CLIMB, "--resubmit")
    assert r3.returncode == 3 and "COMPLETED task(s) 3" in r3.stderr and "--array=0,1,2" in r3.stderr and "human's decision" in r3.stderr, r3.stderr
    assert len(_attempts(w)) == 1
    # task 3 dead too (a run past `running` still blocks: it is a result)
    set_job_state(w.fake["cluster_a"], tasks[3], "FAILED", end="2026-01-01T04:30:00")
    r4 = w.run(*CLIMB, "--resubmit")
    assert r4.returncode == 3 and "past `running`" in r4.stderr and "run-3" in r4.stderr, r4.stderr
    # every task dead, every run still `running`: ONE re-run of the whole unit under the same identities, the array spec intact
    make_registry(w.tmp / "registry", runs=[{"run_id": f"run-{k}", "lifecycle_state": "running", "cluster_job_id": tasks[k]} for k in range(4)])
    r5 = w.run(*CLIMB, "--resubmit")
    out = r5.stdout + r5.stderr
    assert r5.returncode == 0 and "submitted cluster_a" in r5.stdout and "bon.climb" in r5.stdout and "RERUN m/s1/bon/climb" in out, out
    assert "task(s) 0,1,2,3" in out and "4 run id(s)" in out, out
    a2 = _attempts(w)[-1]; assert a2["spec"]["array"] == "0-3" and a2["attempt_id"] == "m/s1/bon/climb@cluster_a@2"
    sup = [e for e in w.ledger() if e["event"] == "superseded_attempt"][0]
    assert sorted(t["task"] for t in sup["tasks"]) == [0, 1, 2, 3] and sup["run_ids"] == ["run-0", "run-1", "run-2", "run-3"]
    new_id = str(job_ids(w.fake["cluster_a"])[-1] if not array_tasks(w.fake["cluster_a"], a2["job_id"]) else a2["job_id"])
    assert len(array_tasks(w.fake["cluster_a"], a2["job_id"])) == 4, new_id


def test_an_array_whose_children_never_registered_is_resubmitted_on_the_classic_proof(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    assert w.run(*CLIMB).returncode == 0
    tasks = dict((k, raw) for raw, k in array_tasks(w.fake["cluster_a"], "1001"))
    for k in range(4):
        set_job_state(w.fake["cluster_a"], tasks[k], "NODE_FAIL", end="2026-01-01T00:0{}:00".format(k + 1))
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T00:20:00")
    r = w.run(*CLIMB, "--resubmit")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT m/s1/bon/climb" in out and "never registered" in out and "all checks negative for every task" in out, out
    assert _attempts(w)[-1]["spec"]["array"] == "0-3" and len(_attempts(w)) == 2


# ----------------------------------------------------------------------------------------------------- (iv) the registration contract
def test_a_kind_other_unit_is_re_minted_on_sacct_alone_and_a_registering_unit_keeps_the_fenced_proof(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r1 = w.run(*PROBE)
    assert r1.returncode == 0 and "submitted cluster_a 1001 hpo.lora.probe" in r1.stdout, r1.stdout + r1.stderr
    assert _attempts(w)[0]["registers"] == "none" and _attempts(w)[0]["spec"]["registers"] == "none"      # kind other -> none by default
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:02:00")
    assert not (w.tmp / "registry" / "state" / "mirror_status.json").exists() and not (w.shared["cluster_a"] / "registry/records").exists()
    r2 = w.run(*PROBE, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "submitted cluster_a 1002 hpo.lora.probe" in r2.stdout, out
    assert "registers=none (kind other)" in out and "check 1 is the whole proof" in out
    # a registering unit keeps the fenced proof (the same job shape, contract registry): with NO producer spool to traverse the proof
    # refuses (not proof of absence); with a traversable, empty spool and no pending file it is never-registered and re-mints
    r3 = w.run(*MMLU)
    assert r3.returncode == 0 and _attempts(w)[-1]["registers"] == "registry"
    set_job_state(w.fake["cluster_a"], "1003", "NODE_FAIL", end="2026-01-01T00:10:00")
    r4 = w.run(*MMLU, "--resubmit")
    assert r4.returncode == 3 and "could not be traversed" in r4.stderr and job_ids(w.fake["cluster_a"]) == [1001, 1002, 1003], r4.stderr
    w.spool("cluster_a")
    r5 = w.run(*MMLU, "--resubmit")
    assert r5.returncode == 0 and "RESUBMIT m/s1/mmlu/mmlu" in r5.stdout + r5.stderr and "submitted cluster_a 1004 mmlu" in r5.stdout, r5.stdout + r5.stderr


def test_a_fit_kind_unit_declared_registers_none_is_re_minted_on_sacct_alone_and_a_pre_field_record_resolves_by_kind(tmp_path):
    w = World(tmp_path); _certify_fit_venv(w)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    recon = ["submit", "--unit", "hpo.relu.recon", "--lwid", "m/03_hpo/relu/hpo.relu.recon", "--script", "scripts/sbatch/hpo_trial.sbatch",
             "--name", "hpo-m-relu-recon", "--args", "spec root psr_ce_all cfg pairs", "--clusters", "cluster_a", "--wave-sha", SHA, "--kind", "fit",
             "--env", "HPO_JOB_MODE=recon", "--array", "0-4"]
    r1 = w.run(*recon, "--registers", "none")
    assert r1.returncode == 0 and "submitted cluster_a 1001 hpo.relu.recon" in r1.stdout, r1.stdout + r1.stderr
    assert _attempts(w)[0]["registers"] == "none" and _attempts(w)[0]["kind"] == "fit"
    for raw, _k in array_tasks(w.fake["cluster_a"], "1001"):
        set_job_state(w.fake["cluster_a"], raw, "TIMEOUT", end="2026-01-01T03:00:00")
    r2 = w.run(*recon, "--registers", "none", "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "hpo.relu.recon" in r2.stdout and "registers=none (kind fit)" in out and "check 1 is the whole proof" in out, out
    assert len(_attempts(w)) == 2 and _attempts(w)[1]["spec"]["array"] == "0-4"
    # a bad value refuses at the CLI
    r5 = w.run(*recon, "--registers", "maybe")
    assert r5.returncode == 2 and "invalid choice" in r5.stderr
    # a ledger record written before the field (no `registers`): kind other -> none; the projection carries the field when present
    r6 = w.run(*PROBE)
    assert r6.returncode == 0, r6.stdout + r6.stderr
    ledger = w.state / "attempts.jsonl"
    recs = [json.loads(l) for l in ledger.read_text().splitlines()]
    for e in recs:
        if e.get("logical_work_id") == "m/03_hpo/lora/hpo.lora.probe":
            e.pop("registers", None); (e.get("spec") or {}).pop("registers", None)
    ledger.write_text("".join(json.dumps(e) + "\n" for e in recs))
    jid = str(job_ids(w.fake["cluster_a"])[-1]); set_job_state(w.fake["cluster_a"], jid, "FAILED", end="2026-01-01T00:05:00")
    r7 = w.run(*PROBE, "--resubmit")
    assert r7.returncode == 0 and "registers=none (kind other)" in r7.stdout + r7.stderr, r7.stdout + r7.stderr
    sys.path.insert(0, str(LIB))
    import submit_twin as ST  # noqa: E402
    assert ST.registration_contract("other", None) == "none" and ST.registration_contract("fit", None) == "registry"
    assert ST.registration_contract("target", "none") == "none" and ST.registration_contract(None, None) == "registry"


# ----------------------------------------------------------------------------------------------------- (2) `none` is bound to the allowlist
RECON = ["submit", "--unit", "hpo.relu.recon", "--lwid", "m/03_hpo/relu/hpo.relu.recon", "--script", "scripts/sbatch/hpo_trial.sbatch",
         "--name", "hpo-m-relu-recon", "--args", "spec root psr_ce_all cfg pairs", "--clusters", "cluster_a", "--wave-sha", SHA, "--kind", "fit", "--array", "0-2"]


def test_none_is_bound_to_the_known_non_registering_scripts_and_the_recon_mode_derives_it(tmp_path):
    """`--registers none` used to be accepted for ANY kind and script. Now a registering script declared
    `none` is refused at submit time (nothing minted); the HPO array derives `none` in HPO_JOB_MODE=recon without a token and refuses
    `none` in its trial mode (it emits fit_trial there); a kind-other unit with a registering script is `registry`."""
    w = World(tmp_path); _certify_fit_venv(w)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = w.run(*CLIMB, "--registers", "none")                                             # bon_climb registers every task's runs
    assert r.returncode == 2 and "--registers none contradicts 'scripts/sdfbase/full22/bon_climb.sbatch'" in r.stderr and "needs the fenced proof" in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and not _attempts(w)
    r = w.run(*RECON, "--env", "HPO_JOB_MODE=trial", "--registers", "none")             # the trial mode emits fit_trial
    assert r.returncode == 2 and "contradicts 'scripts/sbatch/hpo_trial.sbatch' under HPO_JOB_MODE=trial" in r.stderr and job_ids(w.fake["cluster_a"]) == [], r.stderr
    r = w.run(*RECON, "--env", "HPO_JOB_MODE=recon")                                    # the recon mode: none, derived from the script + mode
    assert r.returncode == 0 and _attempts(w)[-1]["registers"] == "none" and _attempts(w)[-1]["spec"]["registers"] == "none", r.stdout + r.stderr
    r = w.run(*[*RECON[:2], "hpo.relu.trial", "--lwid", "m/03_hpo/relu/hpo.relu.trial", *RECON[5:]], "--env", "HPO_JOB_MODE=trial")
    assert r.returncode == 0 and _attempts(w)[-1]["registers"] == "registry", r.stdout + r.stderr
    sys.path.insert(0, str(LIB))
    import submit_twin as ST  # noqa: E402
    rc = ST.registration_contract
    assert rc("fit", None, script="scripts/sbatch/memprobe_fit.sbatch") == "none" and rc("other", None, script="scripts/sbatch/serving_probe.sbatch") == "none"
    assert rc("fit", None, script="scripts/sbatch/hpo_trial.sbatch", env={"HPO_JOB_MODE": "recon"}) == "none"
    assert rc("fit", None, script="scripts/sbatch/hpo_trial.sbatch", env={"HPO_JOB_MODE": "trial"}) == "registry"
    assert rc("other", None, script="scripts/sdfbase/full22/petri_proxy.sbatch") == "registry"          # a registering script wins over the kind
    assert rc("target", "registry", script="scripts/sbatch/memprobe_fit.sbatch") == "registry"           # an explicit registry always stands
    assert rc("other", None) == "none" and rc("target", None) == "registry"                             # no script known: a record older than the field
    import io, contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):                       # Refused prints its reason and exits 2 (a SystemExit); the message is on stderr
        try:
            rc("target", "none", script="scripts/sdfbase/full22/target_static.sbatch"); raise AssertionError("a registering script declared none must refuse")
        except ST.Refused as e:
            assert e.code == 2
    assert "contradicts 'scripts/sdfbase/full22/target_static.sbatch'" in err.getvalue()


def test_a_none_contract_whose_dead_task_left_registration_evidence_is_refused_as_misdeclared(tmp_path):
    """Under `none` the evidence that CAN be read is still read: a dead task that left a pending file (or a registry row under its raw
    id) refuses the re-mint by name — a misdeclared contract never authorises a re-run over results (the misdeclared registering array)."""
    w = World(tmp_path); _certify_fit_venv(w)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r1 = w.run(*RECON, "--env", "HPO_JOB_MODE=recon")
    assert r1.returncode == 0 and _attempts(w)[0]["registers"] == "none", r1.stdout + r1.stderr
    tasks = dict((k, raw) for raw, k in array_tasks(w.fake["cluster_a"], "1001"))
    for k in range(3):
        set_job_state(w.fake["cluster_a"], tasks[k], "TIMEOUT", end=f"2026-01-01T0{k+1}:00:00")
    _pend(w, "cluster_a", tasks[1], ["run-recon-1"])                                     # task 1 registered after all: the declaration was wrong
    r2 = w.run(*RECON, "--env", "HPO_JOB_MODE=recon", "--resubmit")
    assert r2.returncode == 3 and "declares registers=none but its dead job left registration evidence" in r2.stderr and "task 1: pending file" in r2.stderr, r2.stderr
    assert len(_attempts(w)) == 1 and "misdeclared" in r2.stderr
    (w.shared["cluster_a"] / "registry/pending" / f"{tasks[1]}.json").unlink()
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-recon-2", "lifecycle_state": "running", "cluster_job_id": tasks[2]}])   # a registry row under task 2's raw id
    r3 = w.run(*RECON, "--env", "HPO_JOB_MODE=recon", "--resubmit")
    assert r3.returncode == 3 and "task 2: registry run(s) ['run-recon-2']" in r3.stderr and len(_attempts(w)) == 1, r3.stderr
    make_registry(w.tmp / "registry")                                                    # no evidence anywhere: sacct-terminal is the whole proof
    r4 = w.run(*RECON, "--env", "HPO_JOB_MODE=recon", "--resubmit")
    out = r4.stdout + r4.stderr
    assert r4.returncode == 0 and "registers=none (kind fit)" in out and "no task left a pending file, spool record or registry row" in out and len(_attempts(w)) == 2, out


# ----------------------------------------------------------------------------------------------------- (3) the transition token
def _setup_registered_dead_mmlu(w):
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    assert w.run(*MMLU).returncode == 0
    _pend(w, "cluster_a", "1001", ["run-a"])
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "running", "cluster_job_id": "1001"}])
    _bind_registered(w, "m/s1/mmlu/mmlu", ["run-a"])
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T00:20:00")


def test_a_crash_after_retirement_leaves_a_token_the_gate_honours_and_the_next_call_restores_and_re_runs(tmp_path):
    """The crash BEFORE sbatch: the retired registered binding used to be the only record — a concurrent
    `06 --gate` could rebind the old pending file and a retry proved the OLD job dead again. Now the transition token is written first:
    while it stands, binding_state says `transition`, the gate's `bind --state registered` refuses and a plain `submit` skips; the next
    `--resubmit` finds no job of the intended name since the token, RESTORES the registered binding, clears the token and decides again."""
    w = World(tmp_path); _setup_registered_dead_mmlu(w)
    r = w.run(*MMLU, "--resubmit", SISTEER_TWIN_CRASH_AT="resubmit_after_retire")
    assert r.returncode == 97 and "CRASH INJECTED at resubmit_after_retire" in r.stderr, r.stdout + r.stderr
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert "m__s1__mmlu__mmlu.transition.json" in b and "m__s1__mmlu__mmlu.registered.json" not in b and any(".registered.retired." in k for k in b)
    tok = b["m__s1__mmlu__mmlu.transition.json"]
    assert tok["intended_attempt"] == "m/s1/mmlu/mmlu@cluster_a@2" and tok["old_binding"]["job_id"] == "1001" and tok["old_binding"]["state"] == "registered" and tok["retired"]
    assert job_ids(w.fake["cluster_a"]) == [1001]                                           # nothing was launched
    # the gate cannot bind the superseded generation meanwhile; a plain submit skips the unit
    g = subprocess.run([sys.executable, str(LIB / "stage.py"), "bind", "--ledger", str(w.state / "submissions.jsonl"), "--logical-work-id", "m/s1/mmlu/mmlu",
                        "--state", "registered", "--run-ids", "run-a"], capture_output=True, text=True, timeout=60)
    assert g.returncode == 2 and "mid re-run" in g.stderr and "superseded generation" in g.stderr, g.stderr
    # ANY next invocation reconciles the token first: a plain submit finds no job of that name since the token -> restores the registered
    # binding, clears the token, and then skips the unit as it always did (nothing launched)
    # (31e-fix-3: non-launch is declared only on an AGED token — sacct rc 0 and empty, squeue empty by the reserved name, the token past the
    # accounting-visibility floor — so this seconds-old token is aged by hand first; a young one is refused as "not yet")
    young = w.run(*MMLU); assert young.returncode == 3 and "accounting-visibility floor" in young.stderr and "m__s1__mmlu__mmlu.transition.json" in _binding(w, "m__s1__mmlu__mmlu"), young.stderr
    tp = w.state / "bindings" / "m__s1__mmlu__mmlu.transition.json"; tp.write_text(json.dumps({**json.loads(tp.read_text()), "utc": "2026-01-01T00:00:00Z"}))
    plain = w.run(*MMLU)
    out = plain.stdout + plain.stderr
    assert plain.returncode == 0 and "RECONCILE m/s1/mmlu/mmlu: no job of name mmluF-m-s1" in out and "registered binding is restored" in out, out
    assert "skipped cluster_a 1001 mmlu (already registered" in plain.stdout and job_ids(w.fake["cluster_a"]) == [1001]
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert "m__s1__mmlu__mmlu.transition.json" not in b and b["m__s1__mmlu__mmlu.registered.json"]["job_id"] == "1001" and not any(".registered.retired." in k for k in b)
    # the next --resubmit decides again from the restored binding and launches ONCE (a fresh token, retired, cleared)
    r2 = w.run(*MMLU, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "RECONCILE" not in out and "RERUN m/s1/mmlu/mmlu" in out and "retired the registered binding" in out, out
    assert "submitted cluster_a 1002 mmlu" in r2.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002]
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert "m__s1__mmlu__mmlu.transition.json" not in b and b["m__s1__mmlu__mmlu.submitted.json"]["job_id"] == "1002"
    assert any(k.startswith("m__s1__mmlu__mmlu.transition.json.done.") for k in b) and "m__s1__mmlu__mmlu.registered.json" not in b
    assert [e["attempt_id"] for e in _attempts(w)] == ["m/s1/mmlu/mmlu@cluster_a@1", "m/s1/mmlu/mmlu@cluster_a@2"]


def test_a_crash_after_sbatch_accepted_is_adopted_from_the_token_and_never_launched_twice(tmp_path):
    """Hole 3, the crash AFTER sbatch accepted but before the ledger / binding: the job exists and nothing records it; a retry used to
    prove the OLD job dead and launch a duplicate. Now the next call finds the job by the token's NAME + submit time and ADOPTS it as
    the intended attempt — one job, one attempt, the gate then binds it."""
    w = World(tmp_path); _setup_registered_dead_mmlu(w)
    r = w.run(*MMLU, "--resubmit", SISTEER_TWIN_CRASH_AT="submit_after_sbatch")
    assert r.returncode == 97 and "submitted cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr    # sbatch accepted, then the crash
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002] and len(_attempts(w)) == 1                     # unrecorded
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert "m__s1__mmlu__mmlu.transition.json" in b and "m__s1__mmlu__mmlu.registered.json" not in b
    r2 = w.run(*MMLU, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "RECONCILE m/s1/mmlu/mmlu: the transition token's job exists — mmluF-m-s1 1002" in out and "ADOPTED as m/s1/mmlu/mmlu@cluster_a@2" in out, out
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002], "a duplicate was launched"
    assert "skipped cluster_a 1002 mmlu (still RUNNING" in r2.stdout                                   # the adopted job is live: nothing more this call
    a2 = _attempts(w)[-1]; assert a2["attempt_id"] == "m/s1/mmlu/mmlu@cluster_a@2" and a2["job_id"] == "1002" and a2["spec"]["name"] == "mmluF-m-s1"
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.submitted.json"]["job_id"] == "1002" and "m__s1__mmlu__mmlu.transition.json" not in b
    subs = [json.loads(l) for l in (w.state / "submissions.jsonl").read_text().splitlines() if '"submitted"' in l]
    assert [s["job_id"] for s in subs] == ["1001", "1002"]
    # the gate binds the adopted attempt
    _pend(w, "cluster_a", "1002", ["run-a"]); _bind_registered(w, "m/s1/mmlu/mmlu", ["run-a"])
    assert _binding(w, "m__s1__mmlu__mmlu")["m__s1__mmlu__mmlu.registered.json"]["job_id"] == "1002"


# ------------------------------------------------------------------------- (3b) the reservation names the DESTINATION cluster
def _mmlu_on(label):
    """The MMLU argv with --clusters <label>: a cross-cluster --resubmit lands the replacement where --clusters says (24d)."""
    out = list(MMLU); out[out.index("--clusters") + 1] = label; return out


def _dead_mmlu_with_a_destination(tmp_path):
    """Job 1001 on cluster_a: registered, dead, provable (the (3) setup); cluster_b ready to take the replacement — starts on submit, an id space
    that never collides with a's (the ledger's `known` set is by id), its producer spool traversable."""
    w = World(tmp_path); _setup_registered_dead_mmlu(w)
    (w.fake["cluster_b"] / "policy").write_text("start_on_submit\n"); (w.fake["cluster_b"] / "next_id").write_text("5000\n")
    w.spool("cluster_b")
    return w


def test_a_cross_cluster_replacement_reserves_the_destination_and_a_crash_after_sbatch_adopts_the_one_job(tmp_path):
    """The transition token reserved the OLD binding's cluster — the intended attempt, the same-name
    snapshot and the spool all from `b['cluster']` — while submit_one launched on --clusters' first label. A crash after sbatch accepted on the
    destination was reconciled by searching the old cluster, found no such job there, cleared the token as a non-launch, and the retry launched
    a SECOND job against the same stores and run identities. Now the reservation names the destination: the launched job is found there, adopted
    as the destination's attempt and bound there, and the 24d placement receipt the launcher could not write is written at the adoption — once."""
    w = _dead_mmlu_with_a_destination(tmp_path)
    r = w.run(*_mmlu_on("cluster_b"), "--resubmit", SISTEER_TWIN_CRASH_AT="submit_after_sbatch")
    assert r.returncode == 97 and "RERUN m/s1/mmlu/mmlu: bound job 1001 on cluster_a" in r.stdout and "submitted cluster_b 5001 mmlu" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_a"]) == [1001] and job_ids(w.fake["cluster_b"]) == [5001] and len(_attempts(w)) == 1       # launched on b, unrecorded
    b = _binding(w, "m__s1__mmlu__mmlu"); tok = b["m__s1__mmlu__mmlu.transition.json"]
    assert tok["cluster"] == "cluster_b" and tok["intended_attempt"] == "m/s1/mmlu/mmlu@cluster_b@1" and tok["job_name"] == "mmluF-m-s1", tok
    assert tok["old_binding"] == {"state": "registered", "job_id": "1001", "attempt_id": "m/s1/mmlu/mmlu@cluster_a@1", "run_ids": ["run-a"], "array": None, "cluster": "cluster_a"}, tok["old_binding"]
    assert tok["moved_from"] == "cluster_a" and tok["replaced_state"] == "FAILED" and tok["spool"] == f"{w.shared['cluster_b']}/registry/records" and tok["retired"], tok
    assert tok["snapshot_job_ids"] == [], tok                                                     # cluster_b held no same-name job at reservation time
    # the next call reconciles on the DESTINATION: the job is there with this reservation's nonce -> adopted as the destination's attempt
    r2 = w.run(*_mmlu_on("cluster_b"), "--resubmit"); out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "RECONCILE m/s1/mmlu/mmlu: the transition token's job exists — mmluF-m-s1 5001 on cluster_b" in out and "ADOPTED as m/s1/mmlu/mmlu@cluster_b@1" in out, out
    assert ("PLACEMENT RECEIPT m/s1/mmlu/mmlu: cluster_a (replaced job 1001 on cluster_a, FAILED) -> cluster_b (attempt m/s1/mmlu/mmlu@cluster_b@1, job 5001) "
            "— written at the adoption") in out, out
    assert "skipped cluster_b 5001 mmlu (still RUNNING" in r2.stdout and "sbatch never happened" not in out, out
    assert job_ids(w.fake["cluster_a"]) == [1001] and job_ids(w.fake["cluster_b"]) == [5001], "a duplicate was launched"
    a2 = _attempts(w)[-1]
    assert a2["attempt_id"] == "m/s1/mmlu/mmlu@cluster_b@1" and a2["cluster"] == "cluster_b" and a2["job_id"] == "5001" and a2["identified_by"] == "nonce" and a2["spool"] == tok["spool"], a2
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.submitted.json"]["job_id"] == "5001" and b["m__s1__mmlu__mmlu.submitted.json"]["cluster"] == "cluster_b" and "m__s1__mmlu__mmlu.transition.json" not in b
    assert "m__s1__mmlu__mmlu.registered.json" not in b and any(".registered.retired." in k for k in b)            # the superseded generation stays retired
    rc = [e for e in w.ledger() if e["event"] == "placement_receipt"]
    assert len(rc) == 1 and {k: rc[0][k] for k in ("from_cluster", "to_cluster", "replaced_job_id", "replaced_attempt_id", "replaced_state", "attempt_id", "job_id")} == \
        {"from_cluster": "cluster_a", "to_cluster": "cluster_b", "replaced_job_id": "1001", "replaced_attempt_id": "m/s1/mmlu/mmlu@cluster_a@1", "replaced_state": "FAILED",
         "attempt_id": "m/s1/mmlu/mmlu@cluster_b@1", "job_id": "5001"}, rc
    assert rc[0]["reason"].startswith("fenced dead-job replacement on another cluster (submit_twin --resubmit): "), rc[0]["reason"]
    # a third call finds the adopted job live and adds nothing: no second receipt, no job
    r3 = w.run(*_mmlu_on("cluster_b"), "--resubmit")
    assert r3.returncode == 0 and "skipped cluster_b 5001 mmlu (still RUNNING" in r3.stdout and "RECONCILE" not in r3.stdout + r3.stderr, r3.stdout + r3.stderr
    assert len([e for e in w.ledger() if e["event"] == "placement_receipt"]) == 1 and job_ids(w.fake["cluster_b"]) == [5001]


def test_a_cross_cluster_replacement_that_died_after_its_ledger_line_is_finished_on_the_destination(tmp_path):
    """The same reservation, the crash one step later (the attempt is in the ledger, nothing binds or watches it): the token's intended
    attempt is now the destination's, so it IS the ledger's half-recorded attempt — verified by the nonce read back from the live job and
    FINISHED on the destination (the submissions row, the `submitted` binding, the watcher, the 24d receipt); never cleared as a non-launch,
    never launched twice."""
    w = _dead_mmlu_with_a_destination(tmp_path)
    r = w.run(*_mmlu_on("cluster_b"), "--resubmit", SISTEER_TWIN_CRASH_AT="submit_after_ledger")
    assert r.returncode == 97 and "submitted cluster_b 5001 mmlu" in r.stdout, r.stdout + r.stderr
    a2 = _attempts(w)[-1]; assert a2["attempt_id"] == "m/s1/mmlu/mmlu@cluster_b@1" and a2["cluster"] == "cluster_b" and a2["job_id"] == "5001", a2
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.transition.json"]["intended_attempt"] == a2["attempt_id"] and b["m__s1__mmlu__mmlu.submitted.json"]["job_id"] == "1001"   # the half-recorded launch: the old submitted binding still stands
    assert not [e for e in w.ledger() if e["event"] == "placement_receipt"]
    r2 = w.run(*_mmlu_on("cluster_b"), "--resubmit"); out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "the token's attempt m/s1/mmlu/mmlu@cluster_b@1 was already in the ledger as job 5001 on cluster_b" in out, out
    assert "verified by the nonce read back from the live job; finished: submissions row, submitted binding, watcher" in out, out
    assert "PLACEMENT RECEIPT m/s1/mmlu/mmlu: cluster_a (replaced job 1001 on cluster_a, FAILED) -> cluster_b (attempt m/s1/mmlu/mmlu@cluster_b@1, job 5001)" in out, out
    assert "skipped cluster_b 5001 mmlu (still RUNNING" in r2.stdout and job_ids(w.fake["cluster_b"]) == [5001] and job_ids(w.fake["cluster_a"]) == [1001] and len(_attempts(w)) == 2
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.submitted.json"]["job_id"] == "5001" and b["m__s1__mmlu__mmlu.submitted.json"]["cluster"] == "cluster_b" and "m__s1__mmlu__mmlu.transition.json" not in b
    rc = [e for e in w.ledger() if e["event"] == "placement_receipt"]
    assert len(rc) == 1 and rc[0]["from_cluster"] == "cluster_a" and rc[0]["to_cluster"] == "cluster_b" and rc[0]["job_id"] == "5001" and rc[0]["attempt_id"] == a2["attempt_id"], rc


# ----------------------------------------------------------------------------------------------------- (1) the fold proof
def test_a_dead_jobs_record_files_must_be_folded_by_the_harvester_before_its_identities_are_re_run(tmp_path):
    """The proof accepted any per-cluster ingest watermark past the task's END, but that watermark is the
    last arrival of ANY run on the substrate, not a completed fold of THIS job's records — a re-run could start while the dead job's
    spool was mid-fold and upsert into the new attempt's identities. Now every record file the dead job left in its producer spool must
    be in the registry's `record_ingest` with the SAME sha256 and record count and no `record_cursor` (folded to a durable cursor).
    Executed with the REAL emitter (the wave shell's job-start registration writes the spool + the pending file) and the REAL harvester
    (`ingest_once` over a real registry root): unfolded → refused naming the file; folded → the re-run goes out; a file the producer
    re-wrote after the fold → refused by name."""
    import shutil
    from sisteer.registry import harvest as H
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    assert w.run(*MMLU).returncode == 0                                                           # job 1001, name mmluF-m-s1
    fd = "qwen35_4b_demo"                                           # the front-door model; its battery wave config (the declared wave; the s1 spelling was renamed), in the shape 05 --step configs renders
    cfg_rel = f"configs/sdfbase_full22_{fd}_battery.yaml"
    checkout = synthetic_checkout(tmp_path / "co", configs={cfg_rel: f"private_run: sdfbase_full22_{fd}_battery\nexplicit_cells: [none, prompted__safety_refusal, prompted__si_constitution]\n"})
    r = run_f22_registration(checkout, w.shared["cluster_a"], key=fd, view="battery", cfg_rel=cfg_rel, job_id="1001", job_name="mmluF-m-s1",
                             sha=SHA, evals="pasttense,codeattack,renellm,grok420", cluster_label="cluster_a")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    spool = w.shared["cluster_a"] / "registry" / "records"
    files = sorted(p.relative_to(spool) for p in spool.rglob("*.jsonl"))
    assert files and (w.shared["cluster_a"] / "registry" / "pending" / "1001.json").is_file(), files
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    # a REAL registry root, nothing of the job folded yet: refused, naming a record file
    reg = w.tmp / "registry"; shutil.rmtree(reg); H.ensure_layout(str(reg)); H.ingest_once(str(reg), workers=0, min_free=0)
    r1 = w.run(*MMLU, "--resubmit")
    assert r1.returncode == 3 and "record file(s) of the dead job are not folded" in r1.stderr and "not in record_ingest" in r1.stderr, r1.stderr
    assert str(files[0]) in r1.stderr and job_ids(w.fake["cluster_a"]) == [1001], r1.stderr
    # the pull lands the spool under the registry's records/ and the harvester folds it (the real fold, not a manufactured timestamp)
    for rel in files:
        dst = reg / "records" / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(spool / rel, dst)
    rep = H.ingest_once(str(reg), workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    import sqlite3
    conn = sqlite3.connect(str(reg / "registry.sqlite"))
    assert conn.execute("SELECT COUNT(*) FROM record_ingest WHERE file_uri LIKE 'records/sdfbase_full22/%'").fetchone()[0] == len(files)
    assert conn.execute("SELECT COUNT(*) FROM record_cursor").fetchone()[0] == 0
    conn.close()
    r2 = w.run(*MMLU, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "submitted cluster_a 1002 mmlu" in r2.stdout and "RERUN m/s1/mmlu/mmlu" in out, out
    import re as _re
    m = _re.search(r"(\d+) record file\(s\) folded to a durable cursor", out)
    assert m and int(m.group(1)) == len(files), (m.group(0) if m else None, len(files), [str(f) for f in files][:6])
    # the producer re-writes a folded file (a late record): its sha256 no longer matches the fold -> refused by name
    set_job_state(w.fake["cluster_a"], "1002", "FAILED", end="2026-01-01T00:30:00")
    _pend(w, "cluster_a", "1002", ["run-late"])
    late = spool / files[0]
    late.write_text(late.read_text() + json.dumps({"record_type": "run_state", "payload": {"cluster_job_id": "1002", "note": "late"}}) + "\n")
    r3 = w.run(*MMLU, "--resubmit")
    assert r3.returncode == 3 and "not folded" in r3.stderr and ("sha256 differs" in r3.stderr or "n_records differ" in r3.stderr) and str(files[0]) in r3.stderr, r3.stderr
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002]


# ----------------------------------------------------------------------------------------------------- a torn-down pool re-mints
POOL = ["submit", "--unit", "multiturn.attacker1", "--lwid", "m/s1/multiturn/mt-s1-cluster_a/attackers/1", "--script", "scripts/sdfbase/full22/attacker_pool.sbatch",
        "--name", "f22-atk-m-1", "--args", SHA, "--clusters", "cluster_a", "--kind", "attacker", "--group", "mt-s1-cluster_a", "--wave-sha", SHA]


def test_a_torn_down_pool_members_completed_is_its_teardown_and_re_mints_under_the_none_contract(tmp_path):
    """The multi-turn attacker pool registers nothing and exits 0 on its `<jobid>.STOP`, so after `teardown-pool`
    its sacct state is COMPLETED. The proof refused it as "results" and derived `registry` for its script (not allowlisted), so a torn-down
    pool could never be re-minted and its group's targets never re-run. Now the pool script is `none` by allowlist, and a COMPLETED
    member whose job the ledger records as torn_down (THAT job, that cluster) is fenced like a dead one; a COMPLETED member not torn
    down, or torn down under another job id, still refuses (a completion is a result until the ledger says otherwise)."""
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r1 = w.run(*POOL)
    assert r1.returncode == 0 and "submitted cluster_a 1001 multiturn.attacker1" in r1.stdout, r1.stdout + r1.stderr
    assert _attempts(w)[0]["kind"] == "attacker" and _attempts(w)[0]["registers"] == "none"    # unspecified at submit: the funnel derives and records it
    sys.path.insert(0, str(LIB))
    import submit_twin as ST  # noqa: E402
    assert ST.registration_contract("attacker", None, script="scripts/sdfbase/full22/attacker_pool.sbatch") == "none"
    assert ST.registration_contract("attacker", None) == "registry"                             # no script on record: the kind alone does not say none
    # COMPLETED with no teardown on record: a result -> refused, nothing minted
    set_job_state(w.fake["cluster_a"], "1001", "COMPLETED", end="2026-01-01T00:20:00")
    r2 = w.run(*POOL, "--resubmit")
    assert r2.returncode == 3 and "COMPLETED task(s) 1001" in r2.stderr and job_ids(w.fake["cluster_a"]) == [1001], r2.stderr
    # a torn_down record naming ANOTHER job id does not release it
    with open(w.state / "attempts.jsonl", "a") as f:
        f.write(json.dumps({"event": "torn_down", "logical_work_id": POOL[4], "job_id": "999", "cluster": "cluster_a", "utc": "2026-01-01T00:21:00Z"}) + "\n")
    r3 = w.run(*POOL, "--resubmit")
    assert r3.returncode == 3 and "COMPLETED task(s) 1001" in r3.stderr and job_ids(w.fake["cluster_a"]) == [1001], r3.stderr
    # the real teardown: the per-member STOP + the ledger's torn_down for THIS job -> COMPLETED is read as the teardown and the pool re-mints
    r4 = w.run("teardown-pool", "--group", "mt-s1-cluster_a")
    assert r4.returncode == 0, r4.stdout + r4.stderr
    assert (w.shared["cluster_a"] / "exp/reproduce/m/waves/mt-s1-cluster_a/attackers/1001.STOP").exists()
    r5 = w.run(*POOL, "--resubmit")
    out = r5.stdout + r5.stderr
    assert r5.returncode == 0 and "submitted cluster_a 1002 multiturn.attacker1" in r5.stdout, out
    assert "TORN DOWN" in out and "not a result" in out and "registers=none (kind attacker)" in out and "check 1 is the whole proof" in out, out
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002]
    assert [e["job_id"] for e in w.ledger() if e["event"] == "pool_member"] == ["1001", "1002"]     # the re-run is a member the next teardown finds


# ----------------------------------------------------------------------------------------------------- (8) the Petri resume proof (petri-completion, 2026-10-13)
#: a Petri proxy of the demo's battery view — the audit's manifest (configs/capeval/manifests/petri_full22_qwen35_4b_demo_battery.yaml) is on
#: this tree, which the proof requires (spec.args[0:2] -> the tag); the World's model `m` is the funnel's state namespace, not the key
PETRI = ["submit", "--unit", "petri.proxy", "--lwid", "m/battery/petri/proxy", "--script", "scripts/sdfbase/full22/petri_proxy.sbatch", "--name", "f22-petri-m_battery",
         "--args", f"qwen35_4b_demo battery {SHA}", "--clusters", "cluster_a", "--wave-sha", SHA]
OVERFLOW = ["submit", "--unit", "petri.overflow", "--lwid", "m/battery/petri/overflow", "--script", "scripts/sdfbase/full22/petri_proxy.sbatch", "--name", "f22-petri-m_battery-rerun",
            "--args", f"qwen35_4b_demo battery {SHA} overflow-rerun none,prompted__si_constitution,constv2ctl2-full__lora_r4_ce", "--clusters", "cluster_a", "--wave-sha", SHA]
#: the frozen enumeration's legs (the demo record's shape, synthetic run keys): leg 9 is the one the dead drive completed
PETRI_LEGS = ["none", "prompted__safety", "prompted__constitution", "arditi_allpos", "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce",
              "si_constv2_mix_full__lora_r4_ce_minus_ctl", "si_constv2ctl2_full__psr_ce_all", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__psr_ce_all_minus_ctl", "none__lora0"]
PETRI_KEYS = [f"{i:016x}" for i in range(len(PETRI_LEGS))]
PETRI_EVALS = ("petri_pooled", "petri_b1", "petri_b2", "petri_b3", "petri_b4", "petri_b5", "petri_b6")


def _petri_out(w) -> Path:
    """The synthetic out root the stores live under: `<OUT>/cells/<run key>` with the frozen `cells.json` beside them."""
    out = w.tmp / "petri_out"; (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "cells.json").write_text(json.dumps({"audit": "petri_full22_qwen35_4b_demo_battery", "canonical": [{"cell_id": c, "petri_run_key": k} for c, k in zip(PETRI_LEGS[:-1], PETRI_KEYS[:-1])],
                                                "controls": [{"cell_id": PETRI_LEGS[-1], "petri_run_key": PETRI_KEYS[-1], "control_of": "none"}]}))
    return out


def _legs_status_stub(w, *, done=(9,), refused=(), rc=0) -> Path:
    """A stand-in `petri_run.py` for `legs-status` (SISTEER_PETRI_RUN, the test seam): the canned reading, its argv recorded."""
    out = _petri_out(w)
    legs = [{"idx": i, "cell_id": c, "control_of": "none" if c == "none__lora0" else None, "kind": "none" if c.startswith("none") else "x", "petri_run_key": k,
             "cell_dir": str(out / "cells" / k), "done": i in done, "refused": i in refused, "reason": "synthetic refusal" if i in refused else None,
             "results_present": i in done, "unresolved": 0, "overflow_unresolved": 0} for i, (c, k) in enumerate(zip(PETRI_LEGS, PETRI_KEYS))]
    doc = {"out_root": str(out), "cells_json": str(out / "cells.json"), "audit": "petri_full22_qwen35_4b_demo_battery", "serving_path": "lora_native", "record_refused": None, "legs": legs}
    (w.tmp / "legs_status.json").write_text(json.dumps(doc))
    stub = w.tmp / "petri_run_stub.py"
    stub.write_text("import json, sys\nopen(sys.argv[0] + '.argv', 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                    f"sys.exit({rc}) if {rc} else print(open({str(w.tmp / 'legs_status.json')!r}).read())\n")
    return stub


def _petri_runs(w, job, *, generated_legs=(9,), store_for_generated=True, objects=None):
    """The proxy job's 70 runs (10 objects x 7 evals) in the registry: `generated` with a store_uri under the synthetic out root for the legs in
    `generated_legs` (the dead drive's close-leg filed them), `running` for the rest; the pending file naming every run."""
    out = _petri_out(w); rows = []; ids = []
    for i, cell in enumerate(PETRI_LEGS[:-1]):                       # the control leg registers nothing
        for ev in PETRI_EVALS:
            rid = f"sdfbase_full22:{ev}:0123456789ab:{i:02d}{ev[-2:]}" + "0" * 10
            ids.append(rid)
            gen = i in generated_legs
            rows.append({"run_id": rid, "lifecycle_state": "generated" if gen else "running", "cluster_job_id": str(job), "object_id": cell,
                         "store_uri": (str(out / "cells" / PETRI_KEYS[i]) if (gen and store_for_generated) else None)})
    make_registry(w.tmp / "registry", runs=rows)
    _pend(w, "cluster_a", str(job), ids)
    return ids


def _dead_petri(w, argv=PETRI, job="1001"):
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = w.run(*argv)
    assert r.returncode == 0 and f"submitted cluster_a {job} " in r.stdout, r.stdout + r.stderr
    set_job_state(w.fake["cluster_a"], job, "TIMEOUT", end="2026-01-01T04:00:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T04:20:00")


def test_a_petri_proxy_whose_dead_drive_filed_a_leg_is_re_minted_under_the_drives_own_done_reading(tmp_path):
    """The demo's 1010177 (2026-10-13): TIMEOUT with leg 9 COMPLETE (its 7 runs `generated`), 63 `running`. The past-running refusal stands for
    every other script; a Petri proxy's past-running runs are checked against `petri_run.py legs-status` over the stores the registry names —
    leg 9 reads done -> RERUN under the same identities, the note names the leg, the ledger carries the proof record."""
    w = World(tmp_path); _dead_petri(w)
    _petri_runs(w, "1001", generated_legs=(9,))
    stub = _legs_status_stub(w, done=(9,))
    r = w.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(stub))
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "submitted cluster_a 1002 petri.proxy" in r.stdout, out
    assert "RERUN m/battery/petri/proxy" in out and "7 past `running` under the resume proof" in out, out
    assert "7 run(s) past running sit on leg(s) [9:si_constv2_mix_full__psr_ce_all_minus_ctl] the resumed drive reads as done" in out and "never regenerated" in out, out
    sup = [e for e in w.ledger() if e["event"] == "superseded_attempt"]
    assert len(sup) == 1 and sup[0]["resume_proof"]["out_root"] == str(w.tmp / "petri_out") and sup[0]["resume_proof"]["manifest"].endswith("petri_full22_qwen35_4b_demo_battery.yaml")
    assert [(d["idx"], d["cell_id"], len(d["run_ids"])) for d in sup[0]["resume_proof"]["done_legs"]] == [(9, "si_constv2_mix_full__psr_ce_all_minus_ctl", 7)]
    assert len(sup[0]["resume_proof"]["legs_status_sha256"]) == 64 and len(sup[0]["run_ids"]) == 70       # every run id of the dead job, the 7 generated included
    argv = json.loads((w.tmp / "petri_run_stub.py.argv").read_text().splitlines()[-1])
    assert argv[argv.index("--out-root") + 1] == str(w.tmp / "petri_out") and argv[argv.index("--manifest") + 1].endswith("petri_full22_qwen35_4b_demo_battery.yaml") and "legs-status" in argv
    a2 = _attempts(w)[-1]; assert a2["attempt_id"] == "m/battery/petri/proxy@cluster_a@2" and a2["spec"]["args"] == ["qwen35_4b_demo", "battery", SHA]


def test_a_past_running_run_on_a_leg_the_resumed_drive_would_regenerate_keeps_the_refusal(tmp_path):
    w = World(tmp_path); _dead_petri(w)
    _petri_runs(w, "1001", generated_legs=(9,))
    # legs-status reads leg 9 as NOT done (its results.json gone, say): the existing refusal, plus the leg named
    r = w.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w, done=())))
    assert r.returncode == 3 and "past `running`" in r.stderr and "would REGENERATE leg 9 si_constv2_mix_full__psr_ce_all_minus_ctl" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    # ... or refuses it (an identity mismatch is a refusal, never "not done")
    r = w.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w, done=(), refused=(9,))))
    assert r.returncode == 3 and "would REGENERATE leg 9" in r.stderr and "synthetic refusal" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    # a generated run with no store_uri cannot be located: refused by name
    _petri_runs(w, "1001", generated_legs=(9,), store_for_generated=False)
    r = w.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w, done=(9,))))
    assert r.returncode == 3 and "names no store_uri" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    # legs-status failing: refused by name, nothing minted
    _petri_runs(w, "1001", generated_legs=(9,))
    r = w.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w, done=(9,), rc=2)))
    assert r.returncode == 3 and "legs-status failed" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    # no frozen enumeration beside the stores: nothing is provably done (the stub is built first: it lays the record down)
    stub = _legs_status_stub(w, done=(9,)); (w.tmp / "petri_out" / "cells.json").unlink()
    r = w.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(stub))
    assert r.returncode == 3 and "no frozen enumeration" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr
    assert len(_attempts(w)) == 1


def test_the_proof_is_the_petri_proxys_alone_and_reads_the_stores_locally(tmp_path):
    """Every other script keeps the past-running refusal byte-identical (the MMLU pin); a site whose grading host is not this machine refuses the
    proof by name (the stores live there — never an ssh guess); the dry run says what the proof would read."""
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    assert w.run(*MMLU).returncode == 0
    _pend(w, "cluster_a", "1001", ["run-a"]); set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_a"); w.ingested("cluster_a")
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "generated", "cluster_job_id": "1001", "store_uri": "/x/cells/y"}])
    r = w.run(*MMLU, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w)))
    assert r.returncode == 3 and "left run(s) past `running` in the registry ({'1001': ['run-a']}) — those are results; attach/resume them, never re-run" in r.stderr, r.stderr
    assert "REGENERATE" not in r.stderr and "legs-status" not in r.stderr and job_ids(w.fake["cluster_a"]) == [1001]
    sys.path.insert(0, str(LIB))
    import submit_twin as ST  # noqa: E402
    assert ST.resume_proof_for("scripts/sdfbase/full22/petri_proxy.sbatch") is ST.petri_resume_proof and ST.resume_proof_for("./scripts/sdfbase/full22/petri_proxy.sbatch") is ST.petri_resume_proof
    assert ST.resume_proof_for("scripts/sdfbase/full22/mmlu_full.sbatch") is None and ST.resume_proof_for(None) is None
    # a remote grading host: the proof refuses by name before reading anything
    (tmp_path / "remote").mkdir(); w2 = World(tmp_path / "remote", config_extra="SISTEER_GRADING_HOST=cluster-a-login\n"); _dead_petri(w2)
    _petri_runs(w2, "1001", generated_legs=(9,))
    r = w2.run(*PETRI, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w2, done=(9,))))
    assert r.returncode == 3 and "the Petri stores live on the grading host 'cluster-a-login'" in r.stderr and job_ids(w2.fake["cluster_a"]) == [1001], r.stderr
    assert not (w2.tmp / "petri_run_stub.py.argv").exists()
    # the dry run names the proof a petri unit would run (nothing decided; the unit is BOUND in w2, where the dead proxy is)
    r = w2.run("--dry-run", *PETRI, "--resubmit")
    assert r.returncode == 0 and "a petri unit's past-running runs would be checked against petri_run.py legs-status" in r.stdout, r.stdout + r.stderr
    assert "bound submitted to job 1001" in r.stdout and job_ids(w2.fake["cluster_a"]) == [1001]
    r = w.run("--dry-run", *MMLU, "--resubmit")
    assert r.returncode == 0 and "legs-status" not in r.stdout and "nothing decided in a dry run" in r.stdout


def test_the_overflow_units_past_running_runs_are_proven_the_same_way(tmp_path):
    """The rerun proxy drove leg 2 to completion (its `pass-overflow-rerun` runs `generated`, stores under the SAME out root) and died;
    `--resubmit` on the overflow unit (a 5-element spec.args) re-mints it under the proof — leg 2 read as done — with the declaration intact."""
    w = World(tmp_path); _dead_petri(w, OVERFLOW)
    _petri_runs(w, "1001", generated_legs=(2,))
    r = w.run(*OVERFLOW, "--resubmit", SISTEER_PETRI_RUN=str(_legs_status_stub(w, done=(2,))))
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "submitted cluster_a 1002 petri.overflow" in r.stdout and "RERUN m/battery/petri/overflow" in out, out
    assert "sit on leg(s) [2:prompted__constitution] the resumed drive reads as done" in out
    a2 = _attempts(w)[-1]
    assert a2["spec"]["args"] == ["qwen35_4b_demo", "battery", SHA, "overflow-rerun", "none,prompted__si_constitution,constv2ctl2-full__lora_r4_ce"]   # the declaration, byte for byte
    sbatch = (w.fake["cluster_a"] / "jobs" / "1002.argv").read_text()
    assert "overflow-rerun none,prompted__si_constitution,constv2ctl2-full__lora_r4_ce" in sbatch and "-J f22-petri-m_battery-rerun" in sbatch
