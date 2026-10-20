"""Native emission is the DEFAULT on main (2026-09-17): the two mechanisms that make it so, proven without a GPU.

1. SELF-REGISTRATION AT JOB START — ``emit_cli register --from-config`` builds the job's runs from its resolved
   config + the real seeds, writes every ``run`` + the pending file (a ``runs`` list with routing facts), is
   idempotent, and ``state`` / ``push`` then reach EVERY run the file names; the job shells (``full22/common.sh``
   ``f22_registry_register`` from ``f22_resolve_config``, ``stage2_common.sh`` ``s2_registry_register`` from
   ``s2_init``) drive it in a stub cluster — pending file, ``run_state running`` per run, the terminal push.
2. THE STORE RECORDS BY DEFAULT — a ``CellStore`` built the way each wave driver builds it (``petri_run``,
   ``cell_controls_run``, ``served_capability_run``, ``misalign.rows``: no recorder argument) reaches the hooks
   under the job's pending identity, routing each row onto the registered run of ITS object x eval; with
   ``SISTEER_REGISTRY_DISABLED=1`` the store is byte-identical and no record exists; ``resolve_cell`` keeps its
   explicit recorder; the misalignment runner's direct ``responses.jsonl`` append files a generation record.
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import store as cs
from sisteer.capeval.misalign import runner as MR
from sisteer.registry import emit, emit_cli, hooks, identity
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "configs" / "registry"
F22 = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
S2 = REPO / "scripts" / "sbatch" / "stage2_common.sh"
STAGED_SHA = hashlib.sha256(b"synthetic staged object for wiring").hexdigest()
REV = "0123456789abcdef0123456789abcdef01234567"
IHASH = "a1b2c3d4e5f60718"
CHASH = "0f1e2d3c4b5a6978"


# --------------------------------------------------------------------------------------------
# fixtures: a synthetic checkout over the real seeds, a job registered from it
# --------------------------------------------------------------------------------------------


def _checkout(tmp_path: Path, objects=("none", "synth__lora_r4", "prompted__si_constitution")) -> dict:
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    os.symlink(SEEDS, root / "configs" / "registry")
    cfg = root / "configs" / "synth_full22_qwen35_4b.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": list(objects)}))
    (root / "configs" / "sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(
        {"targets": {"qwen35_4b": {"views": {"main": {"cells": {"synth__lora_r4": "base__synth__lora_r4", "prompted__si_constitution": None}}}}}}))
    (root / "configs" / "sdfbase_full22_pins.json").write_text(json.dumps({"qwen35_4b": {"base__synth__lora_r4": {"object_sha256": STAGED_SHA}}}))
    return {"root": root, "cfg": cfg}


def _records(spool):
    return [(p, rec) for p in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(p)]


def _seen(spool) -> set:
    """Record ids present now — the listing is path-ordered, so 'the records written since' is a set difference."""
    return {rec["record_id"] for _, rec in _records(spool)}


def _since(spool, seen: set):
    return [(p, rec) for p, rec in _records(spool) if rec["record_id"] not in seen]


def _register(tmp_path: Path, evals: str, job="4242", job_name="f22-petri-qwen35_4b", **kw) -> dict:
    co = _checkout(tmp_path)
    spool = tmp_path / "spool"
    pending = tmp_path / "pending"
    argv = ["register", "--from-config", str(co["cfg"]), "--model-key", "qwen35_4b", "--evals", evals, "--serving-path", "lora_native",
            "--campaign", "sdfbase_full22", "--job-id", job, "--job-name", job_name, "--pending-dir", str(pending), "--spool", str(spool),
            "--code-sha", synth.code_sha("a"), "--substrate", "cluster_a", "--no-push"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", v]
    rc = emit_cli.main(argv)
    return {"rc": rc, "spool": spool, "pending": pending / f"{job}.json", "pending_dir": pending, "checkout": co}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV, "SLURM_JOB_STATE"):
        monkeypatch.delenv(k, raising=False)
    hooks._DEFAULT_NOTICED.clear()


# --------------------------------------------------------------------------------------------
# 1. self-registration through emit_cli
# --------------------------------------------------------------------------------------------


def test_register_from_config_writes_runs_and_a_routed_pending_file(tmp_path, capsys):
    reg = _register(tmp_path, "petri_self_preservation,agentic_murder")
    assert reg["rc"] == 0
    err = capsys.readouterr().err
    assert "6 run(s) registered at job start" in err
    pend = json.loads(reg["pending"].read_text())
    assert pend["registered_at"] == "job_start" and pend["job_id"] == "4242" and len(pend["runs"]) == 6
    assert pend["run_id"] == pend["runs"][0]["run_id"] and pend["plan"]["run_id"] == pend["run_id"]
    routes = {(r["route"]["eval_id"], r["route"]["object"]) for r in pend["runs"]}
    assert routes == {(e, o) for e in ("petri_self_preservation", "agentic_murder") for o in ("none", "synth__lora_r4", "prompted__si_constitution")}
    recs = _records(reg["spool"])
    assert [r["record_type"] for _, r in recs] == ["run"] * 6
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec
        assert rec["payload"]["lifecycle_state"] == "running" and rec["payload"]["extra"]["registered_at"] == "job_start"
        assert rec["payload"]["queued_utc"] == rec["payload"]["started_utc"] and rec["payload"]["cluster_job_id"] == "4242"
    # idempotent: a second registration keeps the file and writes nothing
    n = len(recs)
    reg2 = _register(tmp_path / "again", "petri_self_preservation", job="4242")
    assert reg2["rc"] == 0
    argv = ["register", "--from-config", str(reg["checkout"]["cfg"]), "--model-key", "qwen35_4b", "--evals", "xstest", "--serving-path", "lora_native",
            "--campaign", "sdfbase_full22", "--job-id", "4242", "--pending-dir", str(reg["pending_dir"]), "--spool", str(reg["spool"]), "--no-push"]
    assert emit_cli.main(argv) == 0
    assert "already exists" in capsys.readouterr().err and len(_records(reg["spool"])) == n
    assert json.loads(reg["pending"].read_text()) == pend


def test_register_from_config_refuses_with_exit_2_and_writes_nothing(tmp_path, capsys):
    """static-reg-fix-2: a refused job-start registration exits 2 (it used to exit 0 with "job continues"; six steered jobs and
    838299's 12 unproduced runs ran on that) — common.sh f22_registry_register stops the job on the missing pending file."""
    co = _checkout(tmp_path)
    base = ["register", "--from-config", str(co["cfg"]), "--serving-path", "lora_native", "--campaign", "c", "--job-id", "1",
            "--pending-dir", str(tmp_path / "p"), "--spool", str(tmp_path / "s"), "--no-push"]
    # missing facts → notice, exit 2
    assert emit_cli.main(base) == 2
    assert "missing --model-key" in capsys.readouterr().err
    # an unknown model key → refused, exit 2, nothing written
    assert emit_cli.main(base + ["--model-key", "nonsuch", "--evals", "xstest"]) == 2
    assert "self-registration refused" in capsys.readouterr().err and not (tmp_path / "s").exists() and not (tmp_path / "p").exists()
    # every pair skipped → nothing written, exit 2
    assert emit_cli.main(base + ["--model-key", "qwen35_4b", "--evals", "nonsuch"]) == 2
    err = capsys.readouterr().err
    assert "skipped eval nonsuch" in err and "no run could be registered" in err and not (tmp_path / "s").exists()


def test_state_and_push_reach_every_registered_run(tmp_path, monkeypatch, capsys):
    reg = _register(tmp_path, "petri_self_preservation")
    pend = json.loads(reg["pending"].read_text())
    rids = [r["run_id"] for r in pend["runs"]]
    assert len(rids) == 3
    monkeypatch.setenv("SLURM_JOB_NAME", "f22-petri-qwen35_4b")
    assert emit_cli.main(["state", "--state", "running", "--pending-dir", str(reg["pending_dir"]), "--job", "4242", "--node", "TESTHOST",
                          "--gpu-count", "1", "--scheduler-state", "RUNNING", "--spool", str(reg["spool"])]) == 0
    assert "applied to 3 runs" in capsys.readouterr().err
    states = [(r["payload"]["run_id"], r["payload"]["state"]) for _, r in _records(reg["spool"]) if r["record_type"] == "run_state"]
    assert sorted(states) == sorted((rid, "running") for rid in rids)
    patches = [r["payload"] for _, r in _records(reg["spool"]) if r["record_type"] == "run_patch"]
    assert len(patches) == 3 and all(p["fields"]["node_hostname"] == "TESTHOST" and p["fields"]["job_name"] == "f22-petri-qwen35_4b" for p in patches)
    # a terminal state, likewise for every run; the records validate
    assert emit_cli.main(["state", "--state", "failed", "--pending-dir", str(reg["pending_dir"]), "--job", "4242", "--scheduler-state", "FAILED",
                          "--final", "--spool", str(reg["spool"])]) == 0
    failed = [r["payload"]["run_id"] for _, r in _records(reg["spool"]) if r["record_type"] == "run_state" and r["payload"]["state"] == "failed"]
    assert sorted(failed) == sorted(rids)
    for p, rec in _records(reg["spool"]):
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec
    # push: one call per run directory
    calls = tmp_path / "calls.log"
    script = tmp_path / "push.sh"
    script.write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\nexit 0\n")
    script.chmod(0o755)
    monkeypatch.setenv(emit.PUSH_SCRIPT_ENV, str(script))
    assert emit_cli.main(["push", "--pending-dir", str(reg["pending_dir"]), "--job", "4242", "--spool", str(reg["spool"]), "--final"]) == 0
    pushed = sorted(line.split()[0] for line in calls.read_text().strip().splitlines())
    assert pushed == sorted(os.path.join(str(reg["spool"]), "sdfbase_full22", identity.run_dir_name(rid)) for rid in rids)


def test_hardware_from_the_jobs_env_reaches_every_registered_run(tmp_path, monkeypatch, capsys):
    """The hardware patch x the multi-run pending file (the 2026-09-17 integration of the two): with no ``--gpu-count`` on the line, the
    emitter reads the hardware from the job's OWN environment ONCE (the environment is the job's, not a run's) and every run the
    pending file names gets the same ``gpu_count`` patch, with the source named in each run's ``running`` state note."""
    reg = _register(tmp_path, "petri_self_preservation")
    rids = [r["run_id"] for r in json.loads(reg["pending"].read_text())["runs"]]
    assert len(rids) == 3
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "h200:8")
    monkeypatch.setenv("SLURM_JOB_NAME", "f22-petri-qwen35_4b")
    assert emit_cli.main(["state", "--state", "running", "--pending-dir", str(reg["pending_dir"]), "--job", "4242", "--node", "TESTHOST",
                          "--scheduler-state", "RUNNING", "--spool", str(reg["spool"])]) == 0
    err = capsys.readouterr().err
    assert "applied to 3 runs" in err and err.count("gpu_count 8 from SLURM_GPUS_ON_NODE") == 1, err       # probed once, not per run
    recs = _records(reg["spool"])
    patches = {r["payload"]["run_id"]: r["payload"]["fields"] for _, r in recs if r["record_type"] == "run_patch"}
    assert sorted(patches) == sorted(rids) and all(f["gpu_count"] == 8 and f["node_hostname"] == "TESTHOST" for f in patches.values())
    states = {r["payload"]["run_id"]: r["payload"] for _, r in recs if r["record_type"] == "run_state"}
    assert sorted(states) == sorted(rids) and all("gpu_count 8 from SLURM_GPUS_ON_NODE" in (s.get("note") or "") for s in states.values())
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec


# --------------------------------------------------------------------------------------------
# 2. the routed identity and the store default
# --------------------------------------------------------------------------------------------


def test_identity_from_env_routes_by_object_and_task(tmp_path, monkeypatch):
    reg = _register(tmp_path, "petri_self_preservation,agentic_murder")
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(reg["pending"]))
    ident = hooks.identity_from_env()
    assert ident is not None and "eval_id" not in ident and "run_id" not in ident       # a MAP, not one identity
    routed = hooks.normalise_identities(ident)
    assert "synth__lora_r4|self_preservation" in routed and "none|agentic_murder" in routed and "none|murder" in routed
    # a bare spelling routes only when ONE registered run carries it: three objects carry each eval here → none bare
    assert "self_preservation" not in routed and "murder" not in routed
    single = _register(tmp_path / "one", "xstest", job="77")
    (single["checkout"]["cfg"]).write_text(yaml.safe_dump({"explicit_cells": ["none"]}))
    single = _register(tmp_path / "one2", "xstest", job="78")
    pend = json.loads(single["pending"].read_text())
    assert len(pend["runs"]) == 3          # three objects declared by the default checkout
    routed = hooks.routed_identities([pend["runs"][0]])
    assert "xstest" in routed and "none|judged/xstest_v2" in routed


def _petri_constants(**over):
    c = {"sweep": "petri_full22_qwen35_4b", "run_id": "petri_full22_qwen35_4b.synth__lora_r4.deadbeef", "model_id": "Qwen/Qwen3.5-4B",
         "model_revision": REV, "arm_name": "synth__lora_r4", "kind": "lora_r4", "coeff": None, "intervention_hash": IHASH,
         "engine": "petri_proxy:qwen35_4b", "trial_id": 0, "seed": 20260909, "attempt": 0, "harness_config_hash": CHASH,
         "env_digest": "99aabbccddeeff00", "slurm_job_id": "4242"}
    c.update(over)
    return c


def _controls_constants(token: str, **over):
    c = {"sweep": "two_si_controls_served", "run_id": f"served-controls-{token}", "model_id": "Qwen/Qwen3.5-4B", "model_revision": REV,
         "arm_name": token, "kind": "cell", "coeff": 1.0, "intervention_hash": IHASH, "engine": "vllm", "trial_id": 0, "seed": 0, "attempt": 0,
         "harness_config_hash": CHASH, "env_digest": "99aabbccddeeff00", "slurm_job_id": "4242"}
    c.update(over)
    return c


def _row(constants, task, item, metric="score", value=1.0, **kw):
    base = dict(constants, tier=1, task=task, task_version="v1", domain="safety", item_id=item, position_order=None, metric=metric,
                value=value, extra_json=None, created_at=1755000000.0, wall_clock_s=None)
    base.update(kw)
    return cs.ResultRow(**base)


def _drive(store: cs.CellStore, resolved: dict, rows) -> None:
    store.validate_resume(resolved)
    store.write_cell_manifest(resolved)
    store.append_ts("job_start", 1755000000.0)
    for r in rows:
        store.append_row(r)
    store.append_ts("cell_end", 1755000010.0)
    store.write_results({"n": len(rows)})


def _tree_shas(root: Path):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def _run_ids_by(pending_path: Path):
    pend = json.loads(pending_path.read_text())
    return {(r["route"]["eval_id"], r["route"]["object"]): r["run_id"] for r in pend["runs"]}


def test_petri_and_misalign_shaped_stores_record_by_default_under_the_pending_identity(tmp_path, monkeypatch):
    """The Petri and misalignment stores name their object in ``arm_name`` (the cell id) and their eval in the row's
    ``task`` (``petri/<behaviour>``, ``agentic/<behaviour>``) — built exactly as the drivers build them (no recorder
    argument), every row lands on the registered run of ITS (object, eval)."""
    reg = _register(tmp_path, "petri_self_preservation,petri_defer_to_users,agentic_murder")
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(reg["pending"]))
    monkeypatch.setenv(emit.SPOOL_ENV, str(reg["spool"]))
    monkeypatch.setattr(cs.time, "time", lambda: 1755000000.0)
    rids = _run_ids_by(reg["pending"])
    seen = _seen(reg["spool"])
    # petri_run.py: CellStore(out_root, cell.petri_run_key, row_constants=row_constants)
    const = _petri_constants()
    store = cs.CellStore(tmp_path / "petri_out", "synth_petri_run_key", row_constants=const)
    assert isinstance(store.recorder, hooks.StoreRecorder)
    resolved = {**const, "row_constants": dict(const), "cell_id": "synth__lora_r4", "petri_run_key": "synth_petri_run_key", "tiers": [1], "cell_index": 0}
    rows = [_row(const, "petri/self_preservation", f"sp{i}") for i in range(3)] + [_row(const, "petri/defer_to_users", f"du{i}") for i in range(2)]
    _drive(store, resolved, rows)
    recs = _since(reg["spool"], seen)
    runs = [r["payload"] for _, r in recs if r["record_type"] == "run"]
    assert {r["run_id"] for r in runs} == {rids[("petri_self_preservation", "synth__lora_r4")], rids[("petri_defer_to_users", "synth__lora_r4")]}
    assert all(r["object_id"] == STAGED_SHA and r["intervention_hash"] == IHASH and r["extra"]["registered_at"] == "job_start" for r in runs)
    items = [r["payload"] for _, r in recs if r["record_type"] == "item_result"]
    by_run = {}
    for it in items:
        by_run.setdefault(it["run_id"], set()).add(it["item_id"])
    assert by_run == {rids[("petri_self_preservation", "synth__lora_r4")]: {"sp0", "sp1", "sp2"}, rids[("petri_defer_to_users", "synth__lora_r4")]: {"du0", "du1"}}
    gen = [(r["payload"]["run_id"], r["payload"]["state"]) for _, r in recs if r["record_type"] == "run_state"]
    assert sorted(gen) == sorted([(rids[("petri_self_preservation", "synth__lora_r4")], "running"), (rids[("petri_defer_to_users", "synth__lora_r4")], "running"),
                                  (rids[("petri_self_preservation", "synth__lora_r4")], "generated"), (rids[("petri_defer_to_users", "synth__lora_r4")], "generated")])
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec
    # misalign/rows.py: CellStore(sweep_dir, key, row_constants=row_constants) with arm_name = cell_id, task = agentic/<behaviour>
    seen = _seen(reg["spool"])
    mconst = _petri_constants(sweep="misalign_qwen35_4b", arm_name="none", kind="none", intervention_hash="0" * 16, engine="misalign:qwen35_4b")
    mstore = cs.CellStore(tmp_path / "mis_out", "0123456789abcdef", row_constants=mconst)
    _drive(mstore, {**mconst, "row_constants": dict(mconst), "cell_id": "none", "family": "agentic"},
           [_row(mconst, "agentic/murder", f"m{i}", metric="misaligned", value=float(i % 2)) for i in range(4)])
    recs = _since(reg["spool"], seen)
    assert {r["payload"]["run_id"] for _, r in recs if r["record_type"] in ("run", "item_result")} == {rids[("agentic_murder", "none")]}
    assert len([r for _, r in recs if r["record_type"] == "item_result"]) == 4


def test_controls_shaped_stores_route_by_the_served_object_sha(tmp_path, monkeypatch, capsys):
    """``cell_controls_run`` / ``served_capability_run`` open one store per CONTROL: ``arm_name`` is the control token
    (no object), the row's ``task`` the control's eval spelling, and the manifest's ``fit_provenance.object_sha`` (16
    hex) names the served object — that sha routes the store onto its object's run; a bare ``none`` store (kind none)
    routes onto the ``none`` object's run; an unregistered task is counted, named once and NOT filed."""
    reg = _register(tmp_path, "xstest,or_bench_hard")
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(reg["pending"]))
    monkeypatch.setenv(emit.SPOOL_ENV, str(reg["spool"]))
    monkeypatch.setattr(cs.time, "time", lambda: 1755000000.0)
    rids = _run_ids_by(reg["pending"])
    seen = _seen(reg["spool"])
    # served_capability_run: arm_name = store_key_token(cell.id) (`__` → `--`), the TRUE id under `cell_id`, the served
    # object's sha under fit_provenance — the token alone is enough (its inverse is the job's object entry)
    const = _controls_constants("synth--lora_r4")
    store = cs.CellStore(tmp_path / "ctl_out" / "qwen35_4b", "synth--lora_r4__controls__t0__a0__a1b2c3d4__0f1e2d3c", row_constants=const)
    resolved = {**const, "row_constants": dict(const), "fit_provenance": {"object_sha": STAGED_SHA[:16], "positions_at_fit": "all"},
                "controls": ["xstest"], "gen": {"max_new_tokens": 512}}
    _drive(store, resolved, [_row(const, "xstest", f"x{i}", metric="xst_label", value=1.0) for i in range(3)] + [_row(const, "mmlu", "q1", metric="acc")])
    recs = _since(reg["spool"], seen)
    rid = rids[("xstest", "synth__lora_r4")]
    assert {r["payload"]["run_id"] for _, r in recs if r["record_type"] in ("run", "item_result", "run_state")} == {rid}
    assert len([r for _, r in recs if r["record_type"] == "item_result"]) == 3           # the mmlu row matched no registered run
    assert "rows of task 'mmlu'" in capsys.readouterr().err
    # the HF controls path (cell_controls_run: engine hf, kind cell, token = the object id, no sha, no cell_id) on the bare
    # object: the token IS the entry → the `none` run; the or_bench rows route by the eval's producer spelling
    seen = _seen(reg["spool"])
    hconst = _controls_constants("none", sweep="two_si_controls", engine="hf", intervention_hash="0" * 16)
    hstore = cs.CellStore(tmp_path / "hf_out", "none__controls__t0__a0__00000000__0f1e2d3c", row_constants=hconst)
    _drive(hstore, {**hconst, "row_constants": dict(hconst), "controls": ["or_bench_hard_1k"]},
           [_row(hconst, "or_bench_hard_1k", f"o{i}", metric="orb_label") for i in range(2)])
    recs = _since(reg["spool"], seen)
    assert {r["payload"]["run_id"] for _, r in recs if r["record_type"] in ("run", "item_result")} == {rids[("or_bench_hard", "none")]}
    # a served-controls store on a prompted object: the token inverse `prompted--si_constitution` → `prompted__si_constitution`
    seen = _seen(reg["spool"])
    pconst = _controls_constants("prompted--si_constitution", kind="prompted")
    pstore = cs.CellStore(tmp_path / "ctl_out2" / "qwen35_4b", "prompted--si_constitution__controls__t0__a0__a1b2c3d4__0f1e2d3c", row_constants=pconst)
    _drive(pstore, {**pconst, "row_constants": dict(pconst), "cell_id": "prompted__si_constitution", "controls": ["xstest"]},
           [_row(pconst, "judged/xstest_v2", "x9", metric="xst_label")])
    recs = _since(reg["spool"], seen)
    assert {r["payload"]["run_id"] for _, r in recs if r["record_type"] in ("run", "item_result")} == {rids[("xstest", "prompted__si_constitution")]}


def test_default_recorder_is_byte_identical_and_silent_when_disabled_or_unregistered(tmp_path, monkeypatch, capsys):
    reg = _register(tmp_path, "petri_self_preservation")
    monkeypatch.setattr(cs.time, "time", lambda: 1755000000.0)
    const = _petri_constants()
    resolved = {**const, "row_constants": dict(const), "cell_id": "synth__lora_r4"}
    rows = lambda c: [_row(c, "petri/self_preservation", f"sp{i}") for i in range(3)]
    # ON: the default recorder records
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(reg["pending"]))
    monkeypatch.setenv(emit.SPOOL_ENV, str(reg["spool"]))
    seen = _seen(reg["spool"])
    _drive(cs.CellStore(tmp_path / "on", "k", row_constants=const), resolved, rows(const))
    assert _since(reg["spool"], seen)
    # DISABLED: no recorder, byte-identical store
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    off = cs.CellStore(tmp_path / "off", "k", row_constants=const)
    assert off.recorder is None
    _drive(off, resolved, rows(const))
    monkeypatch.delenv(emit.DISABLED_ENV)
    assert _tree_shas(tmp_path / "on") == _tree_shas(tmp_path / "off")
    assert not filecmp.dircmp(tmp_path / "on", tmp_path / "off").diff_files
    # UNREGISTERED (spool resolves, no pending identity): no recorder, ONE notice per store key, nothing written
    monkeypatch.delenv(hooks.RUN_IDENTITY_ENV)
    seen = _seen(reg["spool"])
    capsys.readouterr()
    for _ in range(3):
        assert cs.CellStore(tmp_path / "unreg", "k2", row_constants=const).recorder is None
    err = capsys.readouterr().err
    assert err.count("SISTEER_REGISTRY_RUN_IDENTITY is unset") == 1 and not _since(reg["spool"], seen)
    # the explicit sentinel skips the default entirely
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(reg["pending"]))
    assert cs.CellStore(tmp_path / "sentinel", "k3", row_constants=const, recorder=cs.NO_RECORDER).recorder is None


def test_resolve_cell_passes_its_own_verdict_to_the_store():
    src = (REPO / "src" / "sisteer" / "capeval" / "runtime.py").read_text()
    assert "recorder=recorder if recorder is not None else cs.NO_RECORDER" in src
    store_src = (REPO / "src" / "sisteer" / "capeval" / "store.py").read_text()
    assert "NO_RECORDER = object()" in store_src and "_default_recorder(key, self.row_constants)" in store_src


def test_misalign_runner_files_a_generation_record_for_its_cell(tmp_path, monkeypatch, capsys):
    reg = _register(tmp_path, "agentic_murder,agentic_leaking")
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(reg["pending"]))
    monkeypatch.setenv(emit.SPOOL_ENV, str(reg["spool"]))
    rids = _run_ids_by(reg["pending"])
    resp = tmp_path / "agentic" / "synth__lora_r4" / "responses.jsonl"
    resp.parent.mkdir(parents=True)
    resp.write_text('{"item_id": "i1"}\n')
    seen = _seen(reg["spool"])
    MR._registry_generation_complete("agentic", "synth__lora_r4", [{"behavior": "murder"}, {"behavior": "murder"}], resp, 7)
    recs = _since(reg["spool"], seen)
    assert [r["record_type"] for _, r in recs] == ["run_patch"]
    p = recs[0][1]["payload"]
    assert p["run_id"] == rids[("agentic_murder", "synth__lora_r4")] and p["fields"]["n_items_generated"] == 7
    assert p["fields"]["extra_json"]["generation_complete_uri"] == str(resp) and p["fields"]["extra_json"]["family"] == "agentic"
    assert isinstance(VAL.validate_record(recs[0][1], "<t>"), VAL.Validated)
    # an object no run claims: nothing written, one notice, no exception
    seen = _seen(reg["spool"])
    MR._registry_generation_complete("agentic", "nonsuch__object", [{"behavior": "murder"}], resp, 1)
    assert not _since(reg["spool"], seen) and "no registered run claims object 'nonsuch__object'" in capsys.readouterr().err
    # disabled: silent no-op
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    MR._registry_generation_complete("agentic", "synth__lora_r4", [{"behavior": "murder"}], resp, 1)
    assert not _since(reg["spool"], seen)


# --------------------------------------------------------------------------------------------
# 3. the job shells, end to end in a stub cluster
# --------------------------------------------------------------------------------------------


def _stub_cluster(tmp_path: Path) -> dict:
    """A fake $SHARED with the registry provisioned (spool, pending, registry_env.sh, a counting push script)."""
    shared = tmp_path / "shared"
    (shared / "scripts" / "registry").mkdir(parents=True)
    (shared / "registry" / "pending").mkdir(parents=True)
    (shared / "registry" / "records").mkdir(parents=True)
    calls = shared / "push_calls.log"
    push = shared / "scripts" / "registry" / "push_records.sh"
    push.write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\nexit 0\n")
    push.chmod(0o755)
    (shared / "scripts" / "registry" / "registry_env.sh").write_text(
        "#!/usr/bin/env bash\n" f"export SISTEER_REGISTRY_SPOOL={shared}/registry/records\nexport SISTEER_REGISTRY_PUSH={push}\n")
    co = _checkout(tmp_path)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home"), "USER": "nobody-synthetic", "SHARED": str(shared),
           "S2_REPO_DIR": str(REPO), "REPO_DIR": str(REPO), "SLURM_JOB_ID": "4242", "SLURMD_NODENAME": "TESTHOST",
           "SLURM_JOB_NAME": "f22-petri-qwen35_4b", "CUDA_VISIBLE_DEVICES": "0"}
    return {"shared": shared, "env": env, "calls": calls, "checkout": co}


def test_f22_registers_itself_files_running_and_pushes_at_the_end(tmp_path):
    st = _stub_cluster(tmp_path)
    co = st["checkout"]
    script = f"""
set -u
REPO_DIR="{REPO}"; WAVE_SHA="{synth.code_sha('a')}"; CLUSTER=cluster_a
KEY=qwen35_4b; VIEW=main; CFG="{co['cfg']}"; F22_SERVING_PATH=lora_native
F22_ROSTER="{co['root']}/configs/sdfbase_full22_roster.yaml"
source {F22}
f22_registry_evals_for_job f22-petri-x; f22_registry_evals_for_job f22-stat-x-2; f22_registry_evals_for_job f22-misalign-em-x; echo "unknown=[$(f22_registry_evals_for_job weird-job)]"
f22_registry_state running RUNNING; echo "rc_pre=$?"          # f22_init's call: nothing to attach to yet
f22_registry_register; echo "rc_reg=$?"
echo "ident=${{SISTEER_REGISTRY_RUN_IDENTITY:-unset}}"
f22_registry_register; echo "rc_reg2=$?"                          # idempotent
f22_registry_result "COMPLETED exit=0"; echo "rc_res=$?"
f22_registry_result "FAILED exit=1"; echo "rc_fail=$?"
"""
    env = dict(st["env"])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0].startswith("petri_pooled petri_cyber") and lines[1] == "renellm grok420" and lines[2].startswith("em_risky_financial") and lines[3] == "unknown=[]"
    assert "rc_pre=0" in out.stdout and "rc_reg=0" in out.stdout and "rc_reg2=0" in out.stdout and "rc_res=0" in out.stdout and "rc_fail=0" in out.stdout
    pend_path = st["shared"] / "registry" / "pending" / "4242.json"
    assert f"ident={pend_path}" in out.stdout
    assert "no run to attach to" in out.stdout + out.stderr             # f22_init's early call, before the registration
    pend = json.loads(pend_path.read_text())
    # the petri job kind: 7 evals x 3 objects
    assert pend["registered_at"] == "job_start" and len(pend["runs"]) == 21
    rids = {r["run_id"] for r in pend["runs"]}
    recs = _records(st["shared"] / "registry" / "records")
    runs = [r["payload"] for _, r in recs if r["record_type"] == "run"]
    assert {r["run_id"] for r in runs} == rids and all(r["code_sha"] == synth.code_sha("a") and r["substrate"] == "cluster_a" for r in runs)
    running = [r["payload"]["run_id"] for _, r in recs if r["record_type"] == "run_state" and r["payload"]["state"] == "running"]
    assert sorted(running) == sorted(rids)                                 # `running` filed for every run, once
    failed = [r["payload"]["run_id"] for _, r in recs if r["record_type"] == "run_state" and r["payload"]["state"] == "failed"]
    assert sorted(failed) == sorted(rids)
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec
    # the COMPLETED result pushed every run directory --final
    pushed = [ln for ln in st["calls"].read_text().splitlines() if ln.endswith("--final")]
    assert len({ln.split()[0] for ln in pushed}) == 21


def test_f22_registration_declines_by_name_when_it_cannot_state_the_job(tmp_path):
    st = _stub_cluster(tmp_path)
    co = st["checkout"]
    env = dict(st["env"], SLURM_JOB_NAME="weird-job")
    script = f"""
set -u
REPO_DIR="{REPO}"; WAVE_SHA="{synth.code_sha('a')}"; CLUSTER=cluster_a
KEY=qwen35_4b; VIEW=main; CFG="{co['cfg']}"; F22_SERVING_PATH=lora_native
source {F22}
f22_registry_register; echo "rc=$?"; echo "ident=${{SISTEER_REGISTRY_RUN_IDENTITY:-unset}}"
F22_REGISTRY_EVALS="xstest or_bench_hard" f22_registry_register; echo "rc2=$?"; echo "ident2=${{SISTEER_REGISTRY_RUN_IDENTITY:-unset}}"
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "rc=0" in out.stdout and "ident=unset" in out.stdout and "declares no evals" in out.stderr
    assert "rc2=0" in out.stdout and f"ident2={st['shared']}/registry/pending/4242.json" in out.stdout
    pend = json.loads((st["shared"] / "registry" / "pending" / "4242.json").read_text())
    assert {r["route"]["eval_id"] for r in pend["runs"]} == {"xstest", "or_bench_hard"}
    # no spool provisioned (SHARED points nowhere: no helper, no SISTEER_REGISTRY_SPOOL) — static-reg-fix-3: the kind that declares no
    # evals still declines by name (rc 0, the notice), but a REGISTERING kind is now FATAL before anything else — it used to return success here,
    # which is how a sampled rung could reach its standup unregistered. The refusal names the helper that is absent and the rule.
    (st["shared"] / "scripts" / "registry" / "registry_env.sh").write_text("#!/usr/bin/env bash\n")
    env2 = {k: v for k, v in env.items() if k != "SISTEER_REGISTRY_SPOOL"}; env2["SHARED"] = str(tmp_path / "nowhere")   # the same weird-job name: the first call still declines by kind
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env2, timeout=120)
    both = out.stdout + out.stderr
    assert out.returncode == 1 and "rc=0" in out.stdout and "ident=unset" in out.stdout and "declares no evals" in out.stderr, both
    assert "rc2=" not in out.stdout and f"FATAL: registry: self-registration REFUSED for job 4242 (weird-job) — the registry helper {tmp_path / 'nowhere'}/scripts/registry/registry_env.sh is absent and SISTEER_REGISTRY_SPOOL is unset — no spool to register into" in both, both
    assert "a job whose runs do not exist in the registry does not run (D984 / D1265)" in both


def test_s2_registers_from_the_declared_block(tmp_path):
    st = _stub_cluster(tmp_path)
    co = st["checkout"]
    base = f"""
set -u
S2_REPO_DIR="{REPO}"; SUBMIT_SHA="{synth.code_sha('b')}"
source {S2}
"""
    # nothing declared: one notice, nothing registered, exit 0
    out = subprocess.run(["bash", "-c", base + 's2_registry_register; echo "rc=$?"; echo "ident=${SISTEER_REGISTRY_RUN_IDENTITY:-unset}"'],
                         capture_output=True, text=True, env=dict(st["env"], SLURM_JOB_NAME="s2-synth"), timeout=120)
    assert out.returncode == 0 and "rc=0" in out.stdout and "ident=unset" in out.stdout and "declares no S2_REGISTRY_CONFIG" in out.stderr
    # declared: runs + pending + identity exported; then `running` for every run
    decl = f'S2_REGISTRY_CONFIG="{co["cfg"]}"; S2_REGISTRY_MODEL_KEY=qwen35_4b; S2_REGISTRY_EVALS="xstest,or_bench_hard"; S2_REGISTRY_SERVING_PATH=lora_native\n'
    out = subprocess.run(["bash", "-c", base + decl + 's2_registry_register; echo "rc=$?"; s2_registry_state running RUNNING; echo "rc2=$?"; echo "ident=${SISTEER_REGISTRY_RUN_IDENTITY:-unset}"'],
                         capture_output=True, text=True, env=dict(st["env"], SLURM_JOB_NAME="s2-synth"), timeout=300)
    assert out.returncode == 0, out.stderr
    assert "rc=0" in out.stdout and "rc2=0" in out.stdout and f"ident={st['shared']}/registry/pending/4242.json" in out.stdout
    pend = json.loads((st["shared"] / "registry" / "pending" / "4242.json").read_text())
    assert len(pend["runs"]) == 6 and all(r["plan"]["campaign"] == "two_si" and r["plan"]["code_sha"] == synth.code_sha("b") for r in pend["runs"])
    recs = _records(st["shared"] / "registry" / "records")
    running = [r["payload"]["run_id"] for _, r in recs if r["record_type"] == "run_state" and r["payload"]["state"] == "running"]
    assert sorted(running) == sorted(r["run_id"] for r in pend["runs"])


# --------------------------------------------------------------------------------------------
# 4. text pins: where the calls sit
# --------------------------------------------------------------------------------------------


def test_shell_text_pins():
    f22 = F22.read_text()
    assert "f22_registry_register(){" in f22 and "f22_registry_evals_for_job(){" in f22
    resolve = f22[f22.index("f22_resolve_config(){"):f22.index("# --- model facts a standup must attest")]
    assert "f22_registry_register || true" in resolve
    reg = f22[f22.index("f22_registry_register(){"):f22.index("# --- THE LAUNCH-TIME ARGV HOOK")]
    assert "python3 -m sisteer.registry.emit_cli register --from-config" in reg and "python3 -S" not in reg
    assert 'if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; return 0; fi' in reg      # a wrapper's file is kept
    assert "--export=" not in f22.replace("never --export=", "").replace("NEVER --export=", "").replace("never `--export=`", "")
    mis = (REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_proxy.sbatch").read_text()
    i_bind = mis.index('CFG="configs/misalign_full22_${TARGET}.yaml"')
    i_reg = mis.index("f22_registry_register || true")
    assert i_bind < i_reg and "F22_REGISTRY_EVALS=" in mis[i_bind:i_reg]
    s2 = S2.read_text()
    assert "s2_registry_register() {" in s2
    init = s2[s2.index("s2_init() {"):]
    assert init.index("s2_registry_register || true") < init.index('s2_registry_state running "${SLURM_JOB_STATE:-RUNNING}" || true') < init.index("uv sync --frozen")
    for f in (F22, S2, REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_proxy.sbatch"):
        subprocess.run(["bash", "-n", str(f)], check=True)


def test_modal_shard_registers_in_container_and_files_its_terminal_state():
    s = (REPO / "scripts" / "modal" / "misalign_vendor_shards.py").read_text()
    assert "def _registry_register(" in s and "def _registry_state(" in s
    shard = s[s.index("def shard("):s.index("def main(")]
    i_reg = shard.index("_registry_register(tree, CFG, MODEL_KEY")
    i_run = shard.index('"sisteer.capeval.misalign.runner"')
    i_state = shard.index("_registry_state(tree, reg_job")
    assert i_reg < i_run < i_state
    assert 'renv["SISTEER_REGISTRY_RUN_IDENTITY"] = reg' in shard
    helper = s[s.index("def _registry_register("):s.index("def _registry_state(")]
    assert '"register", "--from-config"' in helper and '"--substrate", "modal"' in helper and "except Exception" in helper
    assert 'if not Path(REGISTRY_VOL_ROOT).is_dir():' in helper           # an unprovisioned volume → nothing
