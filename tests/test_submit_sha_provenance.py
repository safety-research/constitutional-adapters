"""At the funnel: a unit of the frozen inventory minted at a tree other than its recorded sha must carry an IDENTICAL eval contract at
this tree (the registry seeds' canonical instrument + decode-policy ref, the shards, the item set) — verified before anything is minted;
the launched attempt then records the unit's sha provenance (first | advance answering a death + the fix commit | rerun) in
gate/expected_units.provenance.json and as a ledger event. A contract that differs refuses by name and mints nothing; a unit outside the
inventory (pools, fits) records no provenance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

MODEL_ID = "qwen35_4b_demo"
NEW = "d" * 40
def _mmlu(sha: str = SHA) -> list:
    """The mmlu unit's submit as 06 composes it: the positional wave sha in the args IS the submitted sha (the job checks out that tree)."""
    return ["submit", "--unit", "mmlu", "--lwid", "m/s1/mmlu/mmlu", "--script", "scripts/sdfbase/full22/mmlu_full.sbatch", "--name", "mmluF-m-s1",
            "--args", f"m s1 {sha}", "--clusters", "cluster_a"]


MMLU = _mmlu(SHA)


def _inventory(w: World, *, mutate=None) -> dict:
    """A frozen inventory holding the mmlu unit with THIS checkout's canonical contract (so the funnel's contract check compares like with like)."""
    canon = ST.canonical_instruments(MODEL_ID, ["mmlu"], None)
    c = canon["mmlu"]; assert "error" not in c, c
    unit = {"logical_work_id": "m/s1/mmlu/mmlu", "family": "mmlu", "stage_family": "mmlu", "unit": "mmlu", "model_id": MODEL_ID, "code_sha": SHA,
            "script": "scripts/sdfbase/full22/mmlu_full.sbatch", "sbatch_name": "mmluF-m-s1", "array": None, "padding_tokens": None, "registration": "native",
            "private_run": None,
            "evals": {"mmlu": {"eval_id": "mmlu", "instrument_id": c["instrument_id"], "instrument_id_source": "test", "decode_policy_ref": c["decode_policy_ref"],
                               "thinking_mode": "off", "expected_shards": ["shard00"],
                               "coverage": {"layout": "capability_cellstore", "item_set_id": c.get("item_set_id"), "n_items": c.get("n_items"), "items_sha256": c.get("items_sha256")}}}}
    # the freeze persists the unit's composed WORKLOAD identity (commit 26); a synthetic inventory carries it too, as the real one does, so the
    # composition is read from the inventory — a synthetic code_sha is not a commit of the repo, and since commit 26b the frozen-tree
    # derivation refuses that by name instead of falling back to "no baseline" (a `mutate` may override it)
    unit["workload"] = {k: v for k, v in ST.spec_identity({"script": unit["script"], "args": ["m", "s1", SHA], "env": {}, "sha": SHA, "array": None,
                                                           "group": None, "kind": "target"}).items() if k in ST.WORKLOAD_FIELDS}
    if mutate:
        mutate(unit)
    doc = {"schema": 2, "model": "m", "view": "s1", "code_sha": SHA, "units": [unit]}
    (w.state / "gate").mkdir(parents=True, exist_ok=True)
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), doc)
    return unit


def _mirror(w: World, sha: str) -> None:
    """The fake cluster's mirrored checkout at `sha` (the World lays down only the freeze sha's): the funnel cd's into it before sbatch."""
    for sh in w.shared.values():
        co = sh / "sisteer-eval-repo" / sha / "scripts/sdfbase/full22"; co.mkdir(parents=True, exist_ok=True)
        (co / "target_attacks.sbatch").write_text("#!/bin/bash\nf22_attempt_wait   # sisteer-attempt-wait-hook\nf22_init\n")


def _ledger(w):
    return [json.loads(l) for l in (w.state / "attempts.jsonl").read_text().splitlines() if l.strip()]


def test_a_first_attempt_records_its_sha_and_a_rerun_at_a_new_tree_advances_only_with_an_identical_contract(tmp_path):
    w = World(tmp_path); _mirror(w, NEW)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _inventory(w)
    r1 = w.run(*MMLU, "--wave-sha", SHA)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    prov = ST.read_provenance(str(w.state / "submissions.jsonl"))["m/s1/mmlu/mmlu"]
    assert [e["kind"] for e in prov] == ["first"] and prov[0]["sha"] == SHA and prov[0]["job_id"] == "1001" and prov[0]["answers"] is None
    assert [e for e in _ledger(w) if e["event"] == "sha_provenance"][0]["sha"] == SHA
    # the job dies; the re-run at a NEW tree with an identical contract advances the unit's sha and records the death it answers
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:10:00")
    w.spool("cluster_a")
    r2 = w.run(*_mmlu(NEW), "--wave-sha", NEW, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "submitted cluster_a 1002 mmlu" in r2.stdout, out
    assert f"PROVENANCE m/s1/mmlu/mmlu: sha {SHA[:12]} -> {NEW[:12]} (advance)" in out and "the eval contract and the workload spec are identical at the new tree" in out, out
    prov = ST.read_provenance(str(w.state / "submissions.jsonl"))["m/s1/mmlu/mmlu"]
    assert [e["kind"] for e in prov] == ["first", "advance"] and prov[1]["sha"] == NEW and prov[1]["answers"]["job_id"] == "1001" and "NODE_FAIL" in prov[1]["answers"]["state"]
    assert prov[1]["contract_sha256"] == prov[0]["contract_sha256"] == ST.contract_sha256(ST.inventory_unit(ST.load_inventory(str(w.state / "submissions.jsonl")), "m/s1/mmlu/mmlu"))
    assert ST.sha_table(str(w.state / "submissions.jsonl"))["m/s1/mmlu/mmlu"] == {"sha": NEW, "frozen_sha": SHA, "n_recorded": 2, "advanced": True, "amended": 0}
    # the attempt record carries the provenance facts the reconciler re-issues verbatim
    att = [e for e in _ledger(w) if e["event"] == "attempt"][-1]
    assert att["spec"]["provenance"]["kind"] == "advance" and att["spec"]["provenance"]["from_sha"] == SHA


def test_a_contract_that_differs_at_the_new_tree_refuses_by_name_and_mints_nothing(tmp_path):
    w = World(tmp_path); _mirror(w, NEW)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    # the frozen contract names another decode-policy ref than this tree's seeds resolve: a first attempt at the freeze sha is fine
    # (nothing to compare — the freeze IS this contract), a first attempt at another tree refuses by name
    def mutate(u):
        u["evals"]["mmlu"]["decode_policy_ref"] = "other|mmlu|off|vendor_preset"
    _inventory(w, mutate=mutate)
    r = w.run(*_mmlu(NEW), "--wave-sha", NEW)          # the spec AT the new tree (its positional sha = the submitted sha): the workload matches the composition, the contract does not
    assert r.returncode == 3 and "eval CONTRACT at tree" in r.stderr and "evals.mmlu.decode_policy_ref: 'other|mmlu|off|vendor_preset' ->" in r.stderr \
        and "a contract change is a new wave" in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and not (w.state / "gate" / ST.PROVENANCE_FILE).exists()
    # a unit outside the inventory records no provenance
    r = w.run("submit", "--unit", "hpo.lora.probe", "--lwid", "m/03_hpo/lora/hpo.lora.probe", "--script", "scripts/sbatch/serving_probe.sbatch",
              "--name", "hpo-m-lora-probe", "--args", "m org/toy " + "0" * 40, "--clusters", "cluster_a", "--wave-sha", NEW, "--kind", "other")
    assert r.returncode == 0 and not (w.state / "gate" / ST.PROVENANCE_FILE).exists(), r.stdout + r.stderr


def test_the_workload_spec_is_compared_on_every_submit_and_a_changed_knob_refuses_before_sbatch(tmp_path):
    """The contract check compared only the registry contract (instruments, refs) and only when the sha changed —
    an env knob that reaches the job (the multi-turn seed base, a positional window) or a changed positional arg would re-run a different
    workload under the same frozen baseline unit, on the same sha unnoticed. Now the WORKLOAD spec 06 composed for this submission (script,
    args with the wave-sha token normalised, every env knob, array, group) is compared against the unit's FIRST attempt's on every
    submit, same sha included; the one tolerated change is a decode-policy ref env knob re-spelled as the same row's versioned id."""
    w = World(tmp_path); _mirror(w, NEW)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _inventory(w)
    base = MMLU + ["--env", "F22_REGISTRY_EVALS=mmlu F22_MT_TARGET_SEED_BASE=20260909 F22_MT_TARGET_POLICY_REF=qwen35_4b_demo|multi_turn|off|vendor_preset"]
    r1 = w.run(*base, "--wave-sha", SHA)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:10:00"); w.spool("cluster_a")
    # SAME sha, the seed base changed: refused before sbatch, by name
    changed = MMLU + ["--env", "F22_REGISTRY_EVALS=mmlu F22_MT_TARGET_SEED_BASE=1 F22_MT_TARGET_POLICY_REF=qwen35_4b_demo|multi_turn|off|vendor_preset"]
    r2 = w.run(*changed, "--wave-sha", SHA, "--resubmit")
    assert r2.returncode == 3 and "WORKLOAD spec differs from the baseline's" in r2.stderr and "env.F22_MT_TARGET_SEED_BASE: '20260909' -> '1'" in r2.stderr, r2.stderr
    assert job_ids(w.fake["cluster_a"]) == [1001]
    # a changed positional arg (a window), new sha: refused too
    args_changed = ["submit", "--unit", "mmlu", "--lwid", "m/s1/mmlu/mmlu", "--script", "scripts/sdfbase/full22/mmlu_full.sbatch", "--name", "mmluF-m-s1",
                    "--args", f"m s1 {NEW} 16384", "--clusters", "cluster_a", "--env", "F22_REGISTRY_EVALS=mmlu F22_MT_TARGET_SEED_BASE=20260909 F22_MT_TARGET_POLICY_REF=qwen35_4b_demo|multi_turn|off|vendor_preset"]
    r3 = w.run(*args_changed, "--wave-sha", NEW, "--resubmit")
    assert r3.returncode == 3 and "args: ['m', 's1', '{sha}'] -> ['m', 's1', '{sha}', '16384']" in r3.stderr, r3.stderr
    # the ref re-spelled as the same row's versioned id (the plan-side correction), new sha: tolerated, the re-run goes out
    respelled = _mmlu(NEW) + ["--env", "F22_REGISTRY_EVALS=mmlu F22_MT_TARGET_SEED_BASE=20260909 F22_MT_TARGET_POLICY_REF=qwen35_4b_demo|multi_turn|off|vendor_preset@v1"]
    r4 = w.run(*respelled, "--wave-sha", NEW, "--resubmit")
    assert r4.returncode == 0 and "submitted cluster_a 1002 mmlu" in r4.stdout and "the eval contract and the workload spec are identical at the new tree" in r4.stdout + r4.stderr, r4.stdout + r4.stderr
    # a positional wave sha that is not the submitted --wave-sha: the job would check out another tree — refused as a spec difference
    r_stale = w.run(*_mmlu(SHA), "--wave-sha", NEW, "--resubmit")
    assert r_stale.returncode == 3 and "args: ['m', 's1', '{sha}'] -> ['m', 's1', '" + SHA + "']" in r_stale.stderr, r_stale.stderr
    # a different row behind the same env knob is NOT tolerated
    set_job_state(w.fake["cluster_a"], "1002", "NODE_FAIL", end="2026-01-01T00:20:00")
    other = _mmlu(NEW) + ["--env", "F22_REGISTRY_EVALS=mmlu F22_MT_TARGET_SEED_BASE=20260909 F22_MT_TARGET_POLICY_REF=other|multi_turn|off|vendor_preset@v1"]
    r5 = w.run(*other, "--wave-sha", NEW, "--resubmit")
    assert r5.returncode == 3 and "env.F22_MT_TARGET_POLICY_REF" in r5.stderr, r5.stderr



REAL = "qwen35_4b_demo"
LW = f"{REAL}/s1/mmlu/mmlu"


def _real_inventory(w: World) -> dict:
    """The frozen inventory of the REAL model's mmlu unit (its plan row's script/name; the canonical contract), so the funnel can re-derive
    the freeze-time composition from it (`stage.frozen_composition` re-plans the unit at the inventory's code_sha)."""
    def mut(u):
        u["logical_work_id"] = LW
        # the freeze persists the unit's composed WORKLOAD identity (commit 26) — the synthetic inventory carries it too, as the real one does,
        # so the composition is read from the inventory (a synthetic code_sha is not a commit of the repo: the frozen-tree derivation would refuse)
        (row,) = [r for r in ST.unit_plan(REAL, "s1", families=["mmlu"], sha=SHA, clusters=["cluster_a", "cluster_b"]) if r["logical_work_id"] == LW]
        u["workload"] = ST.workload_identity_of_row(row, SHA)
    unit = _inventory(w, mutate=mut)
    doc = json.load(open(w.state / "gate" / "expected_units.json")); doc["model"] = REAL; doc["nshards"] = 9
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), doc)
    return unit


def _row(sha: str) -> dict:
    (row,) = [r for r in ST.unit_plan(REAL, "s1", families=["mmlu"], sha=sha, clusters=["cluster_a", "cluster_b"]) if r["logical_work_id"] == LW]
    return row


def _submit_row(row: dict, *, extra_args=(), env_extra="F22_ROSTER=roster.yaml") -> list:
    env = " ".join(f"{k}={v}" for k, v in row["env"].items())
    return ["submit", "--unit", "mmlu", "--lwid", LW, "--script", row["script"], "--name", row["name"], "--args", " ".join([*map(str, row["args"]), *extra_args]),
            "--clusters", "cluster_a", "--env", f"{env_extra} {env}".strip()]


def test_the_workload_baseline_is_the_freeze_time_composition_and_pre_freeze_attempts_are_never_it(tmp_path):
    """The baseline is the unit plan re-derived from the frozen inventory at its code_sha; the unit's first attempt AT the frozen tree
    must equal it; attempts that predate the freeze (a proving pass at an earlier tree) are provisional history. The actorattack targets'
    11:13Z pass lacked the window positional a fix added before the freeze, every frozen-tree attempt carried it, and the proof refused
    the re-run against the pre-freeze shape (2026-09-30 19:54Z)."""
    PRE = "b" * 40
    w = World(tmp_path); _mirror(w, PRE); _mirror(w, NEW)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _real_inventory(w)                                                       # frozen at SHA
    comp = ST.frozen_composition(str(w.state / "submissions.jsonl"), LW, table=None, labels=["cluster_a", "cluster_b"])
    assert comp and comp["script"] == _row(SHA)["script"] and comp["baseline"] == {"kind": "composition", "sha": SHA, "source": "inventory"}
    # a proving pass BEFORE the freeze, at a tree whose composition lacked a positional: admitted against the composition? no — it differs
    # (the window positional is part of the frozen composition), and that is the point: pre-freeze shapes are not this wave's
    r0 = w.run(*_submit_row(_row(PRE), extra_args=["16384"]), "--wave-sha", PRE)
    assert r0.returncode == 3 and "WORKLOAD spec differs from the baseline's" in r0.stderr and "no attempt at the frozen tree yet" in r0.stderr, r0.stderr
    # record that pre-freeze pass as HISTORY (the ledger's shape when pass 1 ran before the proof existed): an attempt event at PRE, other args
    hist = {"event": "attempt", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@1", "cluster": "cluster_a", "job_id": "900", "sha": PRE, "kind": "target",
            "spec": {"script": _row(PRE)["script"], "args": [*map(str, _row(PRE)["args"]), "16384"], "env": {**_row(PRE)["env"], "F22_ROSTER": "roster.yaml"}, "sha": PRE,
                     "array": None, "group": None, "kind": "target"}, "utc": "2026-01-01T00:00:00Z"}
    (w.state / "attempts.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(w.state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(hist) + "\n")
    # the frozen-tree attempt: the composition exactly -> admitted (the pre-freeze history is never the baseline)
    r1 = w.run(*_submit_row(_row(SHA)), "--wave-sha", SHA)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:10:00"); w.spool("cluster_a")
    base, probs = ST.workload_baseline(str(w.state / "submissions.jsonl"), LW, table=None, labels=["cluster_a", "cluster_b"])
    assert probs == [] and base["baseline"]["kind"] == "composition+attempt" and base["baseline"]["job_id"] == "1001" and base["env"]["F22_ROSTER"] == "roster.yaml"
    # a re-run at a NEW tree with the frozen composition: admitted; with the pre-freeze shape: refused, naming the composition AND the attempt
    r2 = w.run(*_submit_row(_row(NEW)), "--wave-sha", NEW, "--resubmit")
    assert r2.returncode == 0 and "submitted cluster_a 1002 mmlu" in r2.stdout, r2.stdout + r2.stderr
    set_job_state(w.fake["cluster_a"], "1002", "NODE_FAIL", end="2026-01-01T00:20:00"); w.spool("cluster_a")
    r3 = w.run(*_submit_row(_row(NEW), extra_args=["16384"]), "--wave-sha", NEW, "--resubmit")
    assert r3.returncode == 3 and "WORKLOAD spec differs from the baseline's" in r3.stderr and "-> [" in r3.stderr and "'16384']" in r3.stderr, r3.stderr
    assert f"baseline = the freeze-time composition at {SHA[:12]} (its first frozen-tree attempt = job 1001 on cluster_a" in r3.stderr, r3.stderr
    # a site-plumbing knob changed since the frozen-tree attempt is still a workload difference (the attempt supplies those keys)
    r4 = w.run(*_submit_row(_row(NEW), env_extra="F22_ROSTER=other.yaml"), "--wave-sha", NEW, "--resubmit")
    assert r4.returncode == 3 and "env.F22_ROSTER: 'roster.yaml' -> 'other.yaml'" in r4.stderr, r4.stderr


def test_a_frozen_tree_attempt_that_is_not_the_composition_refuses_naming_both(tmp_path):
    w = World(tmp_path); _mirror(w, NEW)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _real_inventory(w)
    foreign = {"event": "attempt", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@1", "cluster": "cluster_a", "job_id": "901", "sha": SHA, "kind": "target",
               "spec": {"script": _row(SHA)["script"], "args": [*map(str, _row(SHA)["args"]), "16384"], "env": dict(_row(SHA)["env"]), "sha": SHA, "array": None,
                        "group": None, "kind": "target"}, "utc": "2026-01-01T00:00:00Z"}
    (w.state / "attempts.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(w.state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(foreign) + "\n")
    r = w.run(*_submit_row(_row(SHA)), "--wave-sha", SHA, "--resubmit")
    assert r.returncode == 3 and "first frozen-tree attempt (job 901 on cluster_a" in r.stderr and "is not the freeze-time composition at" in r.stderr and "(D306)" in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == []


def test_a_legacy_inventory_whose_frozen_tree_the_repo_lacks_refuses_by_name_never_a_traceback(tmp_path):
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    def mut(u):
        u["logical_work_id"] = LW                       # no `workload`: a legacy inventory; its code_sha (SHA) is not a commit of this repo
        u.pop("workload", None)
    _inventory(w, mutate=mut)
    doc = json.load(open(w.state / "gate" / "expected_units.json")); doc["model"] = REAL; doc["nshards"] = 9
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), doc)
    r = w.run(*_submit_row(_row(SHA)), "--wave-sha", SHA)
    assert r.returncode == 3 and "the freeze-time workload baseline cannot be established" in r.stderr and "is not a commit of" in r.stderr and "Traceback" not in r.stderr, r.stderr
    assert job_ids(w.fake["cluster_a"]) == []
