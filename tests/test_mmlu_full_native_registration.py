"""mmlu_full.sbatch registers NATIVELY for the front door's models:
a FRONT-DOOR key (reproduce/models.yaml) sets F22_REGISTRY_EVALS=mmlu, `f22_registry_register` (the REAL `sisteer.registry.emit_cli
register --from-config`, the registry's own planner over the checkout's seeds) mints the run under the canonical 1,531-item
instrument `mmlu_letter_prefill_full_1531` / item set `mmlu_1531` and writes the job's pending file, `mmlu_full_registry_check`
REFUSES any other registered identity, and `mmlu_full_registry_end` files the run's terminal state from the job. The paper's own
tags (no F22_REGISTRY_EVALS, a key the table does not carry) register nothing — the sbatch's guard, byte for byte. Everything runs
the sbatch's OWN block (extracted from its text) with the real common.sh in a synthetic checkout: placeholder job ids, synthetic
configs, hostname-free. Nothing here names a real host, user, path or session."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import COMMON_SH, MMLU_FULL_SBATCH, REPO, mmlu_full_registry_block, read_spool, run_mmlu_full_registration, synthetic_checkout  # noqa: E402
from sisteer.registry import synth

SHA = synth.code_sha("d")
FRONT_DOOR = "qwen35_4b_demo"
PAPER = "qwen35_4b"
CONFIGS = {
    f"configs/sdfbase_full22_{FRONT_DOOR}_battery.yaml": f"private_run: sdfbase_full22_{FRONT_DOOR}_battery\nexplicit_cells: [none, prompted__safety_refusal, prompted__si_constitution]\nbon:\n  n_max: 75\n  goal_subset: jbb\n",
    f"configs/sdfbase_full22_{PAPER}.yaml": f"private_run: sdfbase_full22_{PAPER}\nexplicit_cells: [none]\n",
}


def _seed_item_set_of_the_full_instrument() -> str:
    inst = yaml.safe_load((REPO / "configs/registry/instruments.yaml").read_text())
    rows = inst["instruments"] if isinstance(inst, dict) and "instruments" in inst else inst
    row = next(r for r in rows if r.get("instrument_id") == "mmlu_letter_prefill_full_1531")
    (iset,) = row["item_set_ids"]
    return iset


def test_the_sbatch_pins_the_seeds_item_set_and_files_its_terminal_state_before_the_result_line():
    src = MMLU_FULL_SBATCH.read_text()
    m = re.search(r"^MMLU_FULL_ITEM_SET=(\w+)", src, re.M)
    assert m and m.group(1) == _seed_item_set_of_the_full_instrument() == "mmlu_1531"
    assert 'python3 reproduce/lib/models.py front-door "$KEY"' in src and "export F22_REGISTRY_EVALS=mmlu" in src     # the front-door decision, by the table's PHASE views
    # the predicate: the test model is a front-door key, the paper's table rows (main / it only) are not — and models.py's rule IS
    # make_full22_configs.front_door_models' (PHASE_VIEW_RE mirrored; the two never drift apart)
    import subprocess
    for key, want in (("qwen35_4b_demo", 0), ("qwen35_4b", 1), ("gemma4e4b", 1), ("nemotron3nano4b", 1), ("no_such_key", 1)):
        assert subprocess.run([sys.executable, str(REPO / "reproduce/lib/models.py"), "front-door", key], capture_output=True).returncode == want, key
    sys.path.insert(0, str(REPO / "reproduce/lib")); sys.path.insert(0, str(REPO / "scripts/sdfbase"))
    import models as M
    import make_full22_configs as gen
    assert M.PHASE_VIEW_RE.pattern == gen.PHASE_VIEW_RE.pattern and sorted(M.front_door_keys()) == sorted(gen.front_door_models()) == ["qwen35_4b_demo"]
    # the block sits between f22_init and f22_resolve_config (registration happens INSIDE f22_resolve_config), the check right after
    i_init, i_block, i_resolve, i_check = (src.index(s) for s in ('f22_init "$WAVE_SHA"\n', "# --- registry (D65) begin", 'f22_resolve_config "$KEY" "$VIEW"\n', "\nmmlu_full_registry_check\n"))
    assert i_init < i_block < i_resolve < i_check
    # the terminal state from the job, before f22_result's final push (the served driver never writes results.json for the hook)
    cl = src[src.index("cleanup(){"):src.index("trap cleanup")]
    assert cl.index('mmlu_full_registry_end "$RC" "$ec"') < cl.index('f22_result "mmlu_full')
    assert "f22_registry_state generated COMPLETED final" in src and "f22_registry_state cancelled CANCELLED final" in src and "f22_registry_state failed" in src
    # the paper's tags keep the guard: the job name still matches no kind in the registry table
    table = re.search(r"f22_registry_evals_for_job\(\)\{.*?\n\}", COMMON_SH.read_text(), re.S)
    assert table and "mmluF" not in table.group(0) and "#SBATCH --job-name=mmluF" in src
    assert mmlu_full_registry_block()                                                                          # the block extracts


def test_a_front_door_key_registers_natively_under_the_1531_item_instrument(tmp_path):
    co = synthetic_checkout(tmp_path, configs=CONFIGS)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True)
    r = run_mmlu_full_registration(co, shared, key=FRONT_DOOR, view="battery", cfg_rel=f"configs/sdfbase_full22_{FRONT_DOOR}_battery.yaml", job_id="7001", sha=SHA, evals=None)
    assert r.returncode == 0 and "CHECK_RC=0" in r.stdout, r.stdout + r.stderr
    assert "EVALS=mmlu" in r.stdout                                                     # the table key set F22_REGISTRY_EVALS itself
    pend = shared / "registry/pending/7001.json"
    assert pend.is_file() and f"IDENT={pend}" in r.stdout
    doc = json.loads(pend.read_text())
    runs = doc["runs"]
    assert {r_["route"]["object"] for r_ in runs} == {"none", "prompted__safety_refusal", "prompted__si_constitution"}   # one run per cell of the view
    assert all(r_["plan"]["item_set_id"] == "mmlu_1531" and r_["plan"]["eval_id"] == "mmlu" and r_["plan"]["model_id"] == FRONT_DOOR for r_ in runs)
    assert len({r_["plan"]["instrument_id"] for r_ in runs}) == 1 and re.fullmatch(r"[0-9a-f]{64}", runs[0]["plan"]["instrument_id"])
    assert all(r_["plan"]["code_sha"] == SHA for r_ in runs)
    assert "3 run(s) registered natively on item set mmlu_1531" in r.stdout
    # the planner's id IS the frozen one: the same instrument stage.py freezes for this model
    sys.path.insert(0, str(REPO / "reproduce/lib"))
    import stage as ST
    frozen = ST.canonical_instruments(FRONT_DOOR, ["mmlu"], str(co / "configs/registry"))["mmlu"]
    assert frozen["instrument_id"] == runs[0]["plan"]["instrument_id"] and frozen["seed_instrument_id"] == "mmlu_letter_prefill_full_1531"
    # the spool holds the run records the registration emitted (one `run` + `running` per cell) — never a hand-written row
    recs = read_spool(shared / "registry/records")
    assert sorted(rt for rt, _ in recs if rt == "run") == ["run"] * 3 and [p["state"] for rt, p in recs if rt == "run_state"] == ["running"] * 3


def test_a_paper_key_without_the_variable_registers_nothing_the_guard(tmp_path):
    co = synthetic_checkout(tmp_path, configs=CONFIGS)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True)
    r = run_mmlu_full_registration(co, shared, key=PAPER, view="main", cfg_rel=f"configs/sdfbase_full22_{PAPER}.yaml", job_id="7002", sha=SHA, evals=None)
    assert r.returncode == 0 and "CHECK_RC=0" in r.stdout and "IDENT= " in r.stdout and "EVALS=\n" in r.stdout + "\n", r.stdout + r.stderr
    assert not (shared / "registry/pending/7002.json").exists() and not list((shared / "registry/records").rglob("*.jsonl"))
    assert "declares no evals" in r.stderr                                             # common.sh's own line: the job name matches no kind


def test_a_registration_the_planner_refuses_or_files_narrow_stops_the_job(tmp_path):
    co = synthetic_checkout(tmp_path, configs=CONFIGS)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True)
    # a paper key FORCED onto the front door's path: the seeds hold two canonical `mmlu` rows for it, the planner refuses, no pending
    # file — the job stops by name instead of running 1,531 items nobody can bind
    r = run_mmlu_full_registration(co, shared, key=PAPER, view="main", cfg_rel=f"configs/sdfbase_full22_{PAPER}.yaml", job_id="7003", sha=SHA, evals="mmlu")
    assert r.returncode == 1 and "wrote no pending file" in r.stdout + r.stderr and "CHECK_RC" not in r.stdout, r.stdout + r.stderr
    assert "2 canonical seeded instruments" in r.stderr
    # a pending file on the 200-item set (a wrapper's, say): refused — a full-width run is never filed under a narrower identity
    pend = shared / "registry/pending/7004.json"; pend.parent.mkdir(parents=True, exist_ok=True)
    pend.write_text(json.dumps({"run_id": "r", "job_id": "7004", "runs": [{"run_id": "r", "route": {"object": "none"}, "plan": {"eval_id": "mmlu", "item_set_id": "mmlu_200"}}]}))
    r2 = run_mmlu_full_registration(co, shared, key=FRONT_DOOR, view="battery", cfg_rel=f"configs/sdfbase_full22_{FRONT_DOOR}_battery.yaml", job_id="7004", sha=SHA, evals="mmlu")
    assert r2.returncode == 1 and "another item set" in r2.stdout + r2.stderr and "never filed under a narrower item set" in r2.stdout + r2.stderr, r2.stdout + r2.stderr
    # no spool on the cluster: refused by name (the gate binds this unit from its native registration)
    r3 = run_mmlu_full_registration(co, shared, key=FRONT_DOOR, view="battery", cfg_rel=f"configs/sdfbase_full22_{FRONT_DOOR}_battery.yaml", job_id="7005", sha=SHA, evals="mmlu",
                                    extra="unset SISTEER_REGISTRY_SPOOL")
    assert r3.returncode == 1 and "SISTEER_REGISTRY_SPOOL is unset — no spool to register into" in r3.stdout + r3.stderr   # static-reg-fix-3: common.sh refuses the unresolved spool itself, ahead of the launcher's own belt


@pytest.mark.parametrize("rc,ec,stop,want", [("0", "0", False, "generated"), ("2", "0", False, "failed"), ("0", "1", False, "failed"), ("0", "0", True, "cancelled")])
def test_the_terminal_state_comes_from_the_job(tmp_path, rc, ec, stop, want):
    co = synthetic_checkout(tmp_path, configs=CONFIGS)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True)
    if stop:
        (shared / "STOP").write_text("")
    r = run_mmlu_full_registration(co, shared, key=FRONT_DOOR, view="battery", cfg_rel=f"configs/sdfbase_full22_{FRONT_DOOR}_battery.yaml", job_id="7006", sha=SHA, evals="mmlu",
                                   after=f'TOTAL_FAIL=0; mmlu_full_registry_end {rc} {ec}; echo END_RC=$?')
    assert r.returncode == 0 and "END_RC=0" in r.stdout, r.stdout + r.stderr
    states = [p["state"] for rt, p in read_spool(shared / "registry/records") if rt == "run_state"]
    assert states.count("running") == 3 and states.count(want) == 3 and set(states) == {"running", want}, states
    # without an identity (the paper's tags) the function is a no-op
    r0 = run_mmlu_full_registration(co, shared, key=PAPER, view="main", cfg_rel=f"configs/sdfbase_full22_{PAPER}.yaml", job_id="7007", sha=SHA, evals=None,
                                    after='TOTAL_FAIL=0; mmlu_full_registry_end 0 0; echo END_RC=$?')
    assert r0.returncode == 0 and "END_RC=0" in r0.stdout and not (shared / "registry/pending/7007.json").exists()
