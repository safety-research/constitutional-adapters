"""Hardware at emission time: `run.gpu_count` / `run.gpu_type` come from the job's OWN environment —
`SLURM_GPUS_ON_NODE` / `SLURM_JOB_GPUS` / `SLURM_STEP_GPUS` / `CUDA_VISIBLE_DEVICES` for the count, `nvidia-smi
--query-gpu=name` for the type — with `sacct -X -n -P -o AllocTRES -j <job>` as the cluster-side fallback reader, captured
by `emit.hardware_from_env` inside `emit_cli state --state running` and the store hook's `run` record, every probe guarded
so nothing reaches the workload. Before this the job shells computed a count from `CUDA_VISIBLE_DEVICES` alone and the
sacct branch of check 14 existed only as the `walltime_source` vocabulary word.

Everything here is a fake environment: env dicts, stub `sacct` / `nvidia-smi` scripts on a private PATH (one records its
argv, one exits non-zero, one hangs past the timeout), a synthetic pending file and spool. No real Slurm, no real GPU.
"""
from __future__ import annotations

import json
import os
import stat
import textwrap

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, emit_cli, hooks
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

SYNTH_GPU = "NVIDIA H200 (synthetic)"
ALLOC_TRES_TYPED = "billing=64,cpu=64,gres/gpu:h200=4,mem=512G,node=1"


def _stub(bin_dir, name, body):
    p = bin_dir / name
    p.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _private_path(tmp_path, monkeypatch, *, nvidia_smi: bool = False, sacct: str = "", sacct_rc: int = 0):
    """A PATH holding only the stubs asked for (plus bash's own dir), and NO scheduler / GPU variables."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for stale in bin_dir.iterdir():          # a fresh PATH every call: no stub from an earlier step answers
        stale.unlink()
    bash_dir = os.path.dirname(os.path.realpath("/bin/bash"))
    if nvidia_smi:
        _stub(bin_dir, "nvidia-smi", f'printf "%s\\n%s\\n" "{SYNTH_GPU}" "{SYNTH_GPU}"\n')
    if sacct or sacct_rc:
        _stub(bin_dir, "sacct", f'echo "$@" > "{bin_dir}/sacct.argv"\nprintf "%s\\n" "{sacct}"\nexit {sacct_rc}\n')
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{bash_dir}")
    for var in emit.GPU_COUNT_ENV_LADDER + ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CLUSTER_NAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); these tests write to its own tmp spool, so it opts in
    return bin_dir


# --------------------------------------------------------------------------------------------
# the pure readers
# --------------------------------------------------------------------------------------------

def test_gpu_count_env_ladder_reads_counts_and_device_lists_in_order():
    assert emit.gpu_count_from_env({"SLURM_GPUS_ON_NODE": "4"}) == (4, "SLURM_GPUS_ON_NODE")
    assert emit.gpu_count_from_env({"SLURM_GPUS_ON_NODE": "h200:8"}) == (8, "SLURM_GPUS_ON_NODE")           # typed count
    assert emit.gpu_count_from_env({"SLURM_GPUS_ON_NODE": "h200:4,a100:2"}) == (6, "SLURM_GPUS_ON_NODE")    # several types
    assert emit.gpu_count_from_env({"SLURM_JOB_GPUS": "0,1,2"}) == (3, "SLURM_JOB_GPUS")
    assert emit.gpu_count_from_env({"SLURM_STEP_GPUS": "5"}) == (1, "SLURM_STEP_GPUS")
    assert emit.gpu_count_from_env({"CUDA_VISIBLE_DEVICES": "0,1"}) == (2, "CUDA_VISIBLE_DEVICES")
    assert emit.gpu_count_from_env({"CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb,GPU-cccc"}) == (3, "CUDA_VISIBLE_DEVICES")
    # precedence: the scheduler's count beats the harness's device list; a set-but-unparseable rung is skipped, not guessed
    assert emit.gpu_count_from_env({"SLURM_GPUS_ON_NODE": "2", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}) == (2, "SLURM_GPUS_ON_NODE")
    assert emit.gpu_count_from_env({"SLURM_GPUS_ON_NODE": "eight", "CUDA_VISIBLE_DEVICES": "0,1"}) == (2, "CUDA_VISIBLE_DEVICES")
    assert emit.gpu_count_from_env({"SLURM_GPUS_ON_NODE": "", "CUDA_VISIBLE_DEVICES": ""}) == (None, None)
    assert emit.gpu_count_from_env({}) == (None, None)


def test_parse_alloc_tres_untyped_typed_and_absent():
    assert emit.parse_alloc_tres("billing=8,cpu=64,gres/gpu=4,mem=512G,node=1") == {"gpu_count": 4}
    assert emit.parse_alloc_tres(ALLOC_TRES_TYPED) == {"gpu_count": 4, "gpu_type": "h200"}
    assert emit.parse_alloc_tres("cpu=8,gres/gpu=6,gres/gpu:h200=4,gres/gpu:a100=2,mem=64G") == {"gpu_count": 6}   # untyped wins; two types → no type
    assert emit.parse_alloc_tres("cpu=8,gres/gpu:h200=4,gres/gpu:a100=2") == {"gpu_count": 6}
    assert emit.parse_alloc_tres("cpu=8,mem=64G,node=1") == {}
    assert emit.parse_alloc_tres("") == {} and emit.parse_alloc_tres("garbage") == {}


def test_hardware_from_sacct_runs_the_documented_argv_against_a_stub(tmp_path, monkeypatch):
    bin_dir = _private_path(tmp_path, monkeypatch, sacct=ALLOC_TRES_TYPED)
    assert emit.hardware_from_sacct("4242") == {"gpu_count": 4, "gpu_type": "h200", "source": "sacct_alloc_tres"}
    assert (bin_dir / "sacct.argv").read_text().split() == ["-X", "-n", "-P", "-o", "AllocTRES", "-j", "4242"]
    assert emit.hardware_from_sacct(None) == {} and emit.hardware_from_sacct("  ") == {}


def test_hardware_from_sacct_is_empty_without_sacct_or_when_it_fails(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch)                       # no sacct on PATH at all
    assert emit.hardware_from_sacct("4242") == {}
    _private_path(tmp_path, monkeypatch, sacct=ALLOC_TRES_TYPED, sacct_rc=1)   # present, exits 1: its output is not trusted
    assert emit.hardware_from_sacct("4242") == {}
    _private_path(tmp_path, monkeypatch, sacct="cpu=8,mem=64G,node=1")          # no gpu entry (a CPU job)
    assert emit.hardware_from_sacct("4242") == {}


def test_gpu_type_from_nvidia_smi_first_line_absent_and_timeout(tmp_path, monkeypatch):
    bin_dir = _private_path(tmp_path, monkeypatch, nvidia_smi=True)
    assert emit.gpu_type_from_nvidia_smi() == SYNTH_GPU
    _stub(bin_dir, "nvidia-smi", "sleep 5\necho late\n")
    assert emit.gpu_type_from_nvidia_smi(timeout_s=0.3) is None      # a hung tool is a None, never a stall past the timeout
    _stub(bin_dir, "nvidia-smi", "echo 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    assert emit.gpu_type_from_nvidia_smi() is None
    _private_path(tmp_path, monkeypatch)                            # no nvidia-smi at all
    assert emit.gpu_type_from_nvidia_smi() is None


def test_hardware_from_env_precedence_fallback_and_sources(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch, nvidia_smi=True, sacct="cpu=8,gres/gpu=8,mem=1000G,node=1")
    # the environment states the count: sacct is not consulted for it, nvidia-smi names the type
    hw = emit.hardware_from_env({"SLURM_GPUS_ON_NODE": "2", "SLURM_JOB_ID": "4242"})
    assert hw == {"gpu_count": 2, "gpu_count_source": "SLURM_GPUS_ON_NODE", "gpu_type": SYNTH_GPU, "gpu_type_source": "nvidia_smi"}
    # nothing in the environment: sacct's AllocTRES fills the count (the job id from the argument or the env)
    assert emit.hardware_from_env({}, job_id="4242")["gpu_count"] == 8
    assert emit.hardware_from_env({}, job_id="4242")["gpu_count_source"] == "sacct_alloc_tres"
    assert emit.hardware_from_env({"SLURM_JOB_ID": "4242"})["gpu_count"] == 8
    # no job id anywhere and no variables: nothing is filed for the count; the type still comes from nvidia-smi
    hw = emit.hardware_from_env({})
    assert hw["gpu_count"] is None and hw["gpu_count_source"] is None and hw["gpu_type"] == SYNTH_GPU
    # sacct's typed gres name is the type of last resort
    _private_path(tmp_path, monkeypatch, sacct=ALLOC_TRES_TYPED)
    hw = emit.hardware_from_env({}, job_id="4242")
    assert (hw["gpu_count"], hw["gpu_type"], hw["gpu_type_source"]) == (4, "h200", "sacct_alloc_tres")
    assert emit.hardware_note(hw) == "gpu_count 4 from sacct_alloc_tres; gpu_type from sacct_alloc_tres"
    assert emit.hardware_note({"gpu_count": None, "gpu_type": None}) is None


def test_nothing_raises_when_every_probe_is_broken(tmp_path, monkeypatch):
    bin_dir = _private_path(tmp_path, monkeypatch)
    _stub(bin_dir, "nvidia-smi", "kill -SEGV $$\n")
    _stub(bin_dir, "sacct", "printf 'gres/gpu=not-a-number\\n'\nexit 0\n")
    hw = emit.hardware_from_env({"SLURM_GPUS_ON_NODE": "many", "CUDA_VISIBLE_DEVICES": ""}, job_id="4242")
    assert hw == {"gpu_count": None, "gpu_count_source": None, "gpu_type": None, "gpu_type_source": None}


# --------------------------------------------------------------------------------------------
# the job shell path: emit_cli state --state running, through the harvester
# --------------------------------------------------------------------------------------------

def _registry_with_a_queued_run(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    spool = os.path.join(root, "records")
    R.write_records(os.path.join(spool, "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    plan = synth.run_payload("hw_store", i=1)
    rid = plan["run_id"]
    R.write_records(os.path.join(spool, "wrapper", "000001-run.jsonl"),
                    [synth.rec("run", plan, synth.t(1)),
                     synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": synth.t(1), "evidence_kind": "submit_wrapper"}, synth.t(1))])
    pend = tmp_path / "pending"
    pend.mkdir()
    (pend / "4242.json").write_text(json.dumps({"run_id": rid, "job_id": "4242", "plan": plan}))
    return root, spool, str(pend), rid


def _records(spool):
    return [(p, rec) for p in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(p)]


def test_state_running_captures_hardware_from_the_job_environment_and_names_the_sources(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch, nvidia_smi=True)
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")      # the harness's wider view loses to the scheduler's count
    root, spool, pend, rid = _registry_with_a_queued_run(tmp_path)
    assert emit_cli.state("running", pending_dir=pend, job="4242", node="TESTHOST", scheduler_state="RUNNING", spool=spool) == 0
    recs = [r for _, r in _records(spool) if r["record_type"] in ("run_state", "run_patch") and r["payload"].get("state") != "queued"]
    st = next(r["payload"] for r in recs if r["record_type"] == "run_state")
    patch = next(r["payload"] for r in recs if r["record_type"] == "run_patch")
    assert patch["fields"]["gpu_count"] == 2 and patch["fields"]["gpu_type"] == SYNTH_GPU
    assert patch["fields"]["node_hostname"] == "TESTHOST" and patch["fields"]["cluster_job_id"] == "4242"
    assert st["state"] == "running" and st["note"] == f"gpu_count 2 from SLURM_GPUS_ON_NODE; gpu_type from nvidia_smi"
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT gpu_count, gpu_type, node_hostname, lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert tuple(row) == (2, SYNTH_GPU, "TESTHOST", "running")
    note = conn.execute("SELECT note FROM run_state_event WHERE run_id = ? AND state = 'running'", (rid,)).fetchone()[0]
    assert "SLURM_GPUS_ON_NODE" in note and "nvidia_smi" in note
    conn.close()


def test_state_running_explicit_flags_win_and_a_caller_note_is_kept(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch, nvidia_smi=True)
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    root, spool, pend, rid = _registry_with_a_queued_run(tmp_path)
    assert emit_cli.state("running", pending_dir=pend, job="4242", node="TESTHOST", gpu_count=1, gpu_type="declared-type", note="by hand",
                          spool=spool) == 0
    recs = [r for _, r in _records(spool) if r["payload"].get("state") != "queued"]
    patch = next(r["payload"] for r in recs if r["record_type"] == "run_patch")
    st = next(r["payload"] for r in recs if r["record_type"] == "run_state")
    assert patch["fields"]["gpu_count"] == 1 and patch["fields"]["gpu_type"] == "declared-type"
    assert st["note"] == "by hand"                                # nothing captured, so nothing appended


def test_state_running_falls_back_to_sacct_when_the_environment_says_nothing(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch, sacct=ALLOC_TRES_TYPED)     # no nvidia-smi, no SLURM_* / CUDA variables
    root, spool, pend, rid = _registry_with_a_queued_run(tmp_path)
    assert emit_cli.state("running", pending_dir=pend, job="4242", node="TESTHOST", spool=spool) == 0
    recs = [r for _, r in _records(spool) if r["payload"].get("state") != "queued"]
    patch = next(r["payload"] for r in recs if r["record_type"] == "run_patch")
    st = next(r["payload"] for r in recs if r["record_type"] == "run_state")
    assert patch["fields"]["gpu_count"] == 4 and patch["fields"]["gpu_type"] == "h200"
    assert st["note"] == "gpu_count 4 from sacct_alloc_tres; gpu_type from sacct_alloc_tres"


def test_state_running_files_nothing_for_hardware_it_cannot_see(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch)                            # no tools, no variables: a bare container
    root, spool, pend, rid = _registry_with_a_queued_run(tmp_path)
    assert emit_cli.state("running", pending_dir=pend, job="4242", node="TESTHOST", spool=spool) == 0
    recs = [r for _, r in _records(spool) if r["payload"].get("state") != "queued"]
    patch = next(r["payload"] for r in recs if r["record_type"] == "run_patch")
    st = next(r["payload"] for r in recs if r["record_type"] == "run_state")
    assert "gpu_count" not in patch["fields"] and "gpu_type" not in patch["fields"]
    assert st.get("note") is None


# --------------------------------------------------------------------------------------------
# the store hook's run record
# --------------------------------------------------------------------------------------------

IHASH = "a1b2c3d4e5f60718"
RESOLVED = {"sweep": "synthsweep", "run_id": "synthsweep.0.t1.a0", "model_id": synth.MODEL_ID, "model_revision": "0123456789abcdef0123456789abcdef01234567",
            "arm_name": "constitution_lora", "kind": "lora", "coeff": 1.0, "intervention_hash": IHASH, "engine": "lora_native",
            "trial_id": 1, "seed": 7, "attempt": 0, "harness_config_hash": "0f1e2d3c4b5a6978", "env_digest": "99aabbccddeeff00",
            "slurm_job_id": "4242", "tiers": [1], "cell_index": 0}


def _identity():
    return {"campaign": synth.CAMPAIGN, "eval_id": synth.EVAL_ID, "item_set_id": synth.ITEM_SET_ID, "instrument_id": synth.INSTRUMENT_ID,
            "model_id": synth.MODEL_ID, "object_id": synth.OBJECT_SHA, "transfer_kind": "base_to_instruct", "instrument_match": "match",
            "harness": "bespoke_sisteer", "harness_version": "0.1", "scoring_rule": "judge_threshold", "prompt_template_sha256": synth.SHA256_A,
            "decode_seed_rule": "none", "decode_source": "greedy", "token_cap_tokens": 512, "thinking_mode": "off",
            "serving_path": "lora_native", "applied_positions_mode": "all", "multi_turn_steering_scope": "single_turn",
            "n_items_planned": 20, "strength_nominal": 1.0, "strength_nominal_source": "filed", "is_operating_point": 1}


def test_store_recorder_run_record_states_hardware_with_recorded_confidence(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch, nvidia_smi=True)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1,2,3")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    rec = hooks.StoreRecorder("synthkey", RESOLVED, {"*": _identity()}, spool=str(tmp_path / "spool"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store")
    p = rec.run_payload("*")
    assert p["gpu_count"] == 4 and p["gpu_type"] == SYNTH_GPU and p["cluster_job_id"] == "4242"
    fc = p["field_confidence_json"]
    assert fc["gpu_count"] == {"confidence": "recorded", "reason": "SLURM_JOB_GPUS read at the store manifest"}
    assert fc["gpu_type"] == {"confidence": "recorded", "reason": "nvidia_smi read at the store manifest"}
    record = R.make_record("run", p, rec.producer, "11111111-2222-3333-4444-555555555555", synth.t(5))
    out = VAL.validate_record(record, "<memory>", R.record_sha(record))
    assert not isinstance(out, VAL.Malformed), out
    # an identity that states its own hardware is not overwritten by the probe
    ident = dict(_identity(), gpu_count=1, gpu_type="declared-type")
    rec2 = hooks.StoreRecorder("synthkey2", RESOLVED, {"*": ident}, spool=str(tmp_path / "spool2"), code_sha=synth.code_sha("a"))
    rec2.store_dir = str(tmp_path / "store2")
    p2 = rec2.run_payload("*")
    assert p2["gpu_count"] == 1 and p2["gpu_type"] == "declared-type" and "gpu_count" not in p2.get("field_confidence_json", {})


def test_store_recorder_run_record_leaves_hardware_null_on_a_bare_host(tmp_path, monkeypatch):
    _private_path(tmp_path, monkeypatch)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    rec = hooks.StoreRecorder("synthkey", RESOLVED, {"*": _identity()}, spool=str(tmp_path / "spool"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store")
    p = rec.run_payload("*")
    assert "gpu_count" not in p and "gpu_type" not in p
    fc = p.get("field_confidence_json", {})          # present only when some field carries a confidence entry
    assert "gpu_count" not in fc and "gpu_type" not in fc


# --------------------------------------------------------------------------------------------
# the job shells: one capture path, in the emitter
# --------------------------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("rel,helper", [("scripts/sbatch/stage2_common.sh", "s2_registry_state() {"),
                                        ("scripts/sdfbase/full22/common.sh", "f22_registry_state(){")])
def test_job_shells_leave_hardware_capture_to_the_emitter(rel, helper):
    s = open(os.path.join(REPO, rel), encoding="utf-8").read()
    body = s[s.index(helper):]
    body = body[:body.index("\n}\n") + 3]
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))     # the commands, not the comments
    assert "emit_cli state" in code and "--gpu-count" not in code and "--gpu-type" not in code and "nvidia-smi" not in code, code
    assert "SLURM_GPUS_ON_NODE" in body      # the comment names where the facts now come from
