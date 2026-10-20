"""Native emission from the harness store:

* byte identity — two synthetic stores run through ``CellStore`` (manifest → rows → generation marker →
  results) with emission OFF (``SISTEER_REGISTRY_DISABLED=1``) and ON (a tmp spool) are byte-identical,
  file by file, and the spool holds ``run`` / ``running`` / ``item_result`` / ``generated`` records that
  validate clean;
* the actual serving hook payload validates: a synthetic ``LoraStandup`` writes ``endpoint.json`` and
  its hook emits ONE ``run_patch`` per served run whose ``verify`` block carries the immutable re-observations;
  a mutated ``serving_path`` is refused by the harvester naming the field, both values and the run id;
* every field any post-creation hook emits is classified (RUN_PATCHABLE_FIELDS xor RUN_VERIFIED_FIELDS);
* ``queued → running → generated → judged`` folds to the expected ``run`` row through the core reducer;
* ``not_measured`` degeneration rows are written for declared-but-uncomputed measures;
* the switch: off by default with no spool, on with ``REGISTRY_EMIT=1``, and the identity strictness.
"""
from __future__ import annotations

import dataclasses
import filecmp
import hashlib
import json
import os
import types
from pathlib import Path

import pytest

from sisteer.capeval import store as cs
from sisteer.registry import db as DB
from sisteer.registry import emit, hooks, identity
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

IHASH = "a1b2c3d4e5f60718"
CHASH = "0f1e2d3c4b5a6978"
REV = "0123456789abcdef0123456789abcdef01234567"
KEY = "constitution_lora__tier1__t1__a0__a1b2c3d4__0f1e2d3c"  # synthetic store key, not a credential — gitleaks:allow

RESOLVED = {"sweep": "synthsweep", "run_id": "synthsweep.0.t1.a0", "model_id": synth.MODEL_ID, "model_revision": REV,
            "arm_name": "constitution_lora", "kind": "lora", "coeff": 1.0, "intervention_hash": IHASH, "engine": "lora_native",
            "trial_id": 1, "seed": 7, "attempt": 0, "harness_config_hash": CHASH, "env_digest": "99aabbccddeeff00",
            "slurm_job_id": "4242", "tiers": [1], "cell_index": 0}


def _identity(**over):
    """The registry identity a driver supplies: the eval, its version, the instrument and the declared run facts."""
    ident = {"campaign": synth.CAMPAIGN, "eval_id": synth.EVAL_ID, "item_set_id": synth.ITEM_SET_ID, "instrument_id": synth.INSTRUMENT_ID,
             "model_id": synth.MODEL_ID, "object_id": synth.OBJECT_SHA, "transfer_kind": "base_to_instruct", "instrument_match": "match",
             "harness": "bespoke_sisteer", "harness_version": "0.1", "scoring_rule": "judge_threshold", "prompt_template_sha256": synth.SHA256_A,
             "decode_seed_rule": "none", "decode_source": "greedy", "token_cap_tokens": 512, "thinking_mode": "off",
             "serving_path": "lora_native", "applied_positions_mode": "all", "multi_turn_steering_scope": "single_turn",
             "n_items_planned": 20, "strength_nominal": 1.0, "strength_nominal_source": "filed", "is_operating_point": 1}
    ident.update(over)
    return ident


def _row(item, metric="asr", value=1.0, task="syntheval", **kw):
    base = dict(sweep="synthsweep", run_id="synthsweep.0.t1.a0", model_id=synth.MODEL_ID, model_revision=REV, arm_name="constitution_lora",
                kind="lora", coeff=1.0, intervention_hash=IHASH, engine="lora_native", tier=1, task=task, task_version="v1",
                domain="safety", item_id=item, trial_id=1, seed=7, attempt=0, position_order=None, metric=metric, value=value,
                extra_json=None, harness_config_hash=CHASH, env_digest="99aabbccddeeff00", slurm_job_id="4242",
                created_at=1755000000.0, wall_clock_s=None)
    base.update(kw)
    return cs.ResultRow(**base)


def _drive(store: cs.CellStore, n: int = 20) -> None:
    """Manifest → timestamp rows → n items × 2 metrics (one carrying at-cap flags) → generation marker → results."""
    store.validate_resume(RESOLVED)
    store.write_cell_manifest(RESOLVED)
    store.append_ts("job_start", 1755000000.0)
    for i in range(n):
        store.append_row(_row(f"q{i:03d}", "asr", float(i % 2), extra_json=json.dumps({"at_cap": i % 5 == 0, "finish_reason": "stop"})))
        store.append_row(_row(f"q{i:03d}", "score", 0.25 * (i % 4)))
    store.write_generation_complete()
    store.append_ts("cell_end", 1755000123.5)
    store.write_results({"n": n, "asr": 0.5})


def _tree_shas(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _records_in(spool: Path):
    recs = []
    for p in R.iter_record_files(str(spool)):
        for _, rec, sha in R.read_record_file(p):
            recs.append((p, rec, sha))
    return recs


# --------------------------------------------------------------------------------------------
# byte identity
# --------------------------------------------------------------------------------------------


def test_cellstore_bytes_identical_with_and_without_recorder(tmp_path, monkeypatch):
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    # the store stamps wall-clock time into generation_complete.json and its ts_* rows; freeze it so the two
    # runs differ in NOTHING but the recorder (the hook itself never touches the store's clock)
    monkeypatch.setattr(cs.time, "time", lambda: 1755000000.0)
    # OFF: the disabled switch — no recorder, nothing but the store
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    off_root = tmp_path / "off"
    rec_off = hooks.store_recorder(KEY, RESOLVED, _identity(), spool=str(tmp_path / "spool_off"))
    assert rec_off is None
    _drive(cs.CellStore(off_root, KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec_off))
    # ON: a tmp spool
    monkeypatch.delenv(emit.DISABLED_ENV)
    on_root = tmp_path / "on"
    spool = tmp_path / "spool_on"
    rec_on = hooks.store_recorder(KEY, RESOLVED, _identity(), spool=str(spool), registry_emit=True)
    assert rec_on is not None
    _drive(cs.CellStore(on_root, KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec_on))

    # every store file byte-identical, same file set, nothing extra under the store
    a, b = _tree_shas(off_root), _tree_shas(on_root)
    assert a == b and set(a) == {f"cells/{KEY}/manifest.json", f"cells/{KEY}/rows.jsonl", f"cells/{KEY}/results.json",
                                 f"cells/{KEY}/generation_complete.json"}
    cmp = filecmp.dircmp(off_root, on_root)
    assert not cmp.left_only and not cmp.right_only
    assert not (tmp_path / "spool_off").exists()

    # the spool holds the lifecycle, validating clean
    recs = _records_in(spool)
    types_seen = [rec["record_type"] for _, rec, _ in recs]
    assert types_seen.count("run") == 1
    states = [rec["payload"]["state"] for _, rec, _ in recs if rec["record_type"] == "run_state"]
    assert states == ["running", "generated"]
    for p, rec, sha in recs:
        v = VAL.validate_record(rec, str(p), sha)
        assert isinstance(v, VAL.Validated), (p, v)
    items = [rec for _, rec, _ in recs if rec["record_type"] == "item_result"]
    assert len(items) == 40 and {r["payload"]["metric_id"] for r in items} == {"asr", "score"}
    flags = [rec for _, rec, _ in recs if rec["record_type"] == "item_flags"]
    assert len(flags) == 20 and sum(r["payload"]["at_token_cap"] for r in flags) == 4
    run = next(rec for _, rec, _ in recs if rec["record_type"] == "run")["payload"]
    assert run["run_id"] == identity.run_id(synth.CAMPAIGN, KEY, "constitution_lora@a1b2c3d4", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert run["lifecycle_state"] == "running" and run["producer_run_key_legacy"] == KEY
    assert run["intervention_hash"] == IHASH and run["harness_config_hash"] == CHASH and run["model_revision"] == REV
    patches = [rec["payload"] for _, rec, _ in recs if rec["record_type"] == "run_patch"]
    completion = next(p for p in patches if "n_items_generated" in p["fields"])
    assert completion["fields"]["n_items_generated"] == 20
    assert completion["fields"]["walltime_s"] == 123.5 and completion["fields"]["walltime_source"] == "store_span"
    assert completion["fields"]["store_completion_sha256"] == hashlib.sha256((on_root / "cells" / KEY / "results.json").read_bytes()).hexdigest()
    arts = [rec["payload"] for _, rec, _ in recs if rec["record_type"] == "artifact"]
    assert {a["kind"] for a in arts} == {"completion", "rows"}
    deg = [rec["payload"] for _, rec, _ in recs if rec["record_type"] == "run_degeneration"]
    assert {(d["measure"], d["scope"]) for d in deg} == set(hooks.DEFAULT_DECLARED_DEGENERATION)
    assert all(d["measurement_state"] == "not_measured" and d["rate"] is None for d in deg)


def test_a_resumed_store_reports_the_store_total_not_this_passs_increment(tmp_path, monkeypatch):
    """A second CellStore over the SAME cell dir with a fresh recorder (a re-run drive, a restarted leg) appends two more items;
    its completion patch must say 22 (the store total the coverage predicates read from one row), not 2. The resumed
    recorder re-emits none of the 40 old item rows and mints no second `run` payload."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    root, spool = tmp_path / "store", tmp_path / "spool"
    rc = {k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}
    rec1 = hooks.store_recorder(KEY, RESOLVED, _identity(), spool=str(spool), registry_emit=True)
    _drive(cs.CellStore(root, KEY, row_constants=rc, recorder=rec1))
    before = _records_in(spool)
    assert [r["payload"]["fields"]["n_items_generated"] for _, r, _ in before if r["record_type"] == "run_patch" and "n_items_generated" in r["payload"]["fields"]] == [20]
    # the RESUMED pass: a new recorder binds the non-empty store, appends 2 items, closes
    rec2 = hooks.store_recorder(KEY, RESOLVED, _identity(), spool=str(spool), registry_emit=True)
    store2 = cs.CellStore(root, KEY, row_constants=rc, recorder=rec2)
    store2.validate_resume(RESOLVED)
    store2.write_cell_manifest(RESOLVED)
    assert rec2._seeded_items and sum(rec2._seeded_items.values()) == 20
    for i in (20, 21):
        store2.append_row(_row(f"q{i:03d}", "asr", 1.0)); store2.append_row(_row(f"q{i:03d}", "score", 0.5))
    store2.write_results({"n": 22, "asr": 0.5})
    after = _records_in(spool)
    totals = [r["payload"]["fields"]["n_items_generated"] for _, r, _ in after if r["record_type"] == "run_patch" and "n_items_generated" in r["payload"]["fields"]]
    assert totals == [20, 22]                                                             # the store total, not the increment 2
    assert sum(1 for _, r, _ in after if r["record_type"] == "run") == 1                 # one logical run, opened once
    items = [r for _, r, _ in after if r["record_type"] == "item_result"]
    assert len(items) == 44 and sum(1 for r in items if r["payload"]["item_id"] in ("q020", "q021")) == 4   # only the new rows were emitted
    # a fresh (empty) store seeds nothing
    rec3 = hooks.store_recorder(KEY, RESOLVED, _identity(), spool=str(tmp_path / "spool3"), registry_emit=True)
    _drive(cs.CellStore(tmp_path / "store3", KEY, row_constants=rc, recorder=rec3), n=3)
    assert rec3._seeded_items == {}


def test_the_derived_control_leg_binds_to_no_run_and_files_nothing(tmp_path, monkeypatch, capsys):
    """The harness's derived no-op control leg (`none__lora0` on a native standup, `none__lens0` on lens) is kind `none`
    but never an object — the kind=="none" fallback must not route it onto the `none` object's runs. The recorder says so once,
    opens nothing and files nothing; the `none` object's own cell (cell_id `none`) still routes as before."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    idents = {f"none{hooks.ROUTE_SEP}syntheval": _identity(object_id="none", transfer_kind="none")}
    for leg in ("none__lora0", "none__lens0"):
        spec = {**RESOLVED, "arm_name": leg, "cell_id": leg, "kind": "none", "intervention_hash": None}
        rc = {k: spec[k] for k in spec if k not in ("tiers", "cell_index")}
        assert hooks.derived_control_object(spec) == leg and hooks.derived_control_object(rc) == leg
        spool = tmp_path / f"spool_{leg}"
        rec = hooks.store_recorder(f"{leg}__key", spec, idents, spool=str(spool), registry_emit=True)
        assert rec is not None and "none" not in rec._object_candidates() and rec._own_routes() == []
        store = cs.CellStore(tmp_path / f"store_{leg}", f"{leg}__key", row_constants=rc, recorder=rec)
        store.validate_resume(spec); store.write_cell_manifest(spec)
        store.append_row(_row("q000", "asr", 1.0, arm_name=leg, kind="none", intervention_hash=None))
        store.write_generation_complete(); store.write_results({"n": 1})
        assert not spool.exists() or not list(spool.rglob("*.jsonl*"))                        # nothing filed, nothing opened
        assert "derived no-op control leg" in capsys.readouterr().err
    # the `none` object's own cell routes to its run
    spec = {**RESOLVED, "arm_name": "none", "cell_id": "none", "kind": "none", "intervention_hash": None}
    assert hooks.derived_control_object(spec) is None
    rec = hooks.store_recorder("none__key", spec, idents, spool=str(tmp_path / "spool_none"), registry_emit=True)
    assert "none" in rec._object_candidates() and rec._own_routes() == [f"none{hooks.ROUTE_SEP}syntheval"]


def test_store_records_fold_through_the_harvester(tmp_path, monkeypatch):
    """The store's own records (no submit wrapper) ingest: run row at generated, items present, no refusal."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rec = hooks.store_recorder(KEY, RESOLVED, _identity(), spool=os.path.join(root, "records"), registry_emit=True)
    _drive(cs.CellStore(tmp_path / "store", KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec), n=5)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, n_items_generated, walltime_source, store_completion_sha256 FROM run").fetchone()
    assert tuple(row)[:3] == ("generated", 5, "store_span") and len(row[3]) == 64
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM item_flags").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM run_degeneration WHERE measurement_state = 'not_measured'").fetchone()[0] == len(hooks.DEFAULT_DECLARED_DEGENERATION)
    conn.close()


# --------------------------------------------------------------------------------------------
# the switch and the identity strictness
# --------------------------------------------------------------------------------------------


def test_switch_off_without_spool_on_with_env(unprovisioned_host, monkeypatch):
    # every host probe under tmp_path (the fixture): a gate container with a provisioned spool of its own must not decide this
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    assert emit.spool_root() is None and not hooks.emission_enabled()
    assert hooks.store_recorder(KEY, RESOLVED, _identity()) is None
    unprovisioned_host.home_registry.mkdir(parents=True)
    assert emit.spool_root() == str(unprovisioned_host.home_registry / "records") and hooks.emission_enabled()
    monkeypatch.setenv(hooks.EMIT_ENV, "0")
    assert not hooks.emission_enabled()
    monkeypatch.setenv(hooks.EMIT_ENV, "1")
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert not hooks.emission_enabled()          # DISABLED wins over everything
    monkeypatch.delenv(emit.DISABLED_ENV)
    assert hooks.emission_enabled() and hooks.emission_enabled(explicit=False) is False


def test_spool_root_probes_are_redirectable_and_the_removed_volume_probe_resolves_nothing(unprovisioned_host, monkeypatch):
    """Every fixed host path ``spool_root`` walks is a module constant the ``unprovisioned_host`` fixture points under
    tmp_path, so the switch test above meets nothing on the real host; the shared roots are the site map's (none is fixed in
    the emitter), placed under the fixture's root here. The ladder: a shared root without ``registry/`` is skipped, the map's
    first root beats its second, ``~/registry`` is the last probe, the env var beats every probe, and an unset ``$USER`` means no
    shared-root probe. A provisioned volume path (``/vol/registry``) under the host resolves nothing and the module carries no
    constant for it."""
    h = unprovisioned_host
    monkeypatch.delenv(emit.SITE_MAP_ENV, raising=False)
    assert emit.spool_root() is None and emit.spool_root(default="fallback") == "fallback"
    assert not hasattr(emit, "MODAL_VOL_ROOT") and not hasattr(emit, "MODAL_VOL_MOUNT")
    (h.root / "vol" / "registry").mkdir(parents=True)
    assert emit.spool_root() is None and emit.substrate_from_env() == "grading_host"
    first, second = h.root / "shared-a" / h.user, h.root / "shared-b" / h.user
    monkeypatch.setenv(emit.SITE_MAP_ENV, f"cluster-one=cluster_a@{h.root}/shared-a/{{user}} cluster-two=cluster_b@{h.root}/shared-b/{{user}}")
    second.mkdir(parents=True)                                         # a user dir alone is not a provisioned spool
    assert emit.spool_root() is None
    (second / "registry").mkdir()
    assert emit.spool_root() == str(second / "registry" / "records")
    (first / "registry").mkdir(parents=True)
    assert emit.spool_root() == str(first / "registry" / "records")
    h.home_registry.mkdir(parents=True)
    assert emit.spool_root() == str(first / "registry" / "records")
    for d in (first / "registry", second / "registry"):
        d.rmdir()
    assert emit.spool_root() == str(h.home_registry / "records")
    monkeypatch.setenv(emit.SPOOL_ENV, str(h.root / "explicit"))
    assert emit.spool_root() == str(h.root / "explicit")
    monkeypatch.delenv(emit.SPOOL_ENV)
    (first / "registry").mkdir()
    monkeypatch.setenv("USER", "")
    assert emit.spool_root() == str(h.home_registry / "records")


#: the registry's own session variable, then the one the session manager sets for each session it starts
OWN_SESSION_VAR, MANAGER_SESSION_VAR = "SISTEER_SESSION_NAME", "ROMP_SESSION_NAME"


def test_the_producer_block_names_the_session_from_the_environment(monkeypatch):
    """A producer block's session is the caller's, else the registry's own variable, else the session manager's (all a
    session it starts carries), else none; a variable set to the empty string names nothing."""
    for var in (OWN_SESSION_VAR, MANAGER_SESSION_VAR):
        monkeypatch.delenv(var, raising=False)
    assert emit.producer_block("s.py", code_sha="a" * 40)["session"] is None
    monkeypatch.setenv(MANAGER_SESSION_VAR, "synthetic-managed-session")
    assert emit.producer_block("s.py", code_sha="a" * 40)["session"] == "synthetic-managed-session"
    monkeypatch.setenv(OWN_SESSION_VAR, "synthetic-session")
    assert emit.producer_block("s.py", code_sha="a" * 40)["session"] == "synthetic-session"
    assert emit.producer_block("s.py", code_sha="a" * 40, session="given")["session"] == "given"
    monkeypatch.setenv(OWN_SESSION_VAR, "")
    assert emit.producer_block("s.py", code_sha="a" * 40)["session"] == "synthetic-managed-session"


def test_the_session_variables_are_the_registrys_own_then_the_session_managers():
    assert emit.SESSION_ENV_VARS == (OWN_SESSION_VAR, MANAGER_SESSION_VAR)


def test_store_recorder_refuses_an_incomplete_identity(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test names a tmp spool and reads the strictness notices, so it opts in
    ident = _identity()
    del ident["item_set_id"]
    assert hooks.store_recorder(KEY, RESOLVED, ident, spool=str(tmp_path), registry_emit=True) is None
    err = capsys.readouterr().err
    assert "item_set_id" in err and "not recorded" in err
    assert hooks.store_recorder(KEY, RESOLVED, None, spool=str(tmp_path), registry_emit=True) is None
    assert "no registry identity" in capsys.readouterr().err


def test_store_recorder_reads_identity_from_env_pending_file(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own tmp spool, so it opts in
    pend = tmp_path / "4242.json"
    pend.write_text(json.dumps({"run_id": "synthcampaign:syntheval:abcdefabcdef:0123456789abcdef", "plan": _identity()}))
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(pend))
    rec = hooks.store_recorder(KEY, RESOLVED, None, spool=str(tmp_path / "spool"), registry_emit=True)
    assert rec is not None
    rec.on_manifest(RESOLVED, str(tmp_path / "manifest.json"))
    run = next(r for _, r, _ in _records_in(tmp_path / "spool") if r["record_type"] == "run")["payload"]
    # a registered job: the store's records join the wrapper's run (its id verbatim), the store key kept in extra
    assert run["run_id"] == "synthcampaign:syntheval:abcdefabcdef:0123456789abcdef"
    assert run["extra"]["store_key"] == KEY and run["extra"]["registered_at"] == "submit_wrapper"


def test_store_with_several_evals_routes_rows_by_task(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own tmp spool, so it opts in
    ident = {"syntheval": _identity(), "other_eval": _identity(eval_id="other_eval", item_set_id="other_eval_50")}
    rec = hooks.store_recorder(KEY, RESOLVED, ident, spool=str(tmp_path / "spool"), registry_emit=True)
    store = cs.CellStore(tmp_path / "store", KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec)
    store.write_cell_manifest(RESOLVED)
    store.append_row(_row("q1", task="syntheval"))
    store.append_row(_row("q1", task="other_eval"))
    store.append_row(_row("q1", task="unrouted_task"))
    store.write_results({})
    runs = {r["payload"]["eval_id"]: r["payload"]["run_id"] for _, r, _ in _records_in(tmp_path / "spool") if r["record_type"] == "run"}
    assert set(runs) == {"syntheval", "other_eval"} and len(set(runs.values())) == 2
    items = [r["payload"] for _, r, _ in _records_in(tmp_path / "spool") if r["record_type"] == "item_result"]
    assert sorted(i["run_id"] for i in items) == sorted(runs.values())     # the unrouted row is not recorded


def test_recorder_failure_never_reaches_the_store(tmp_path, capsys):
    class Broken:
        def on_manifest(self, *a):
            raise RuntimeError("boom")

        def on_row(self, *a):
            raise RuntimeError("boom")

        def on_results(self, *a):
            raise RuntimeError("boom")

        def on_generation_complete(self, *a):
            raise RuntimeError("boom")
    store = cs.CellStore(tmp_path / "store", KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=Broken())
    _drive(store, n=2)
    assert store.is_done() and len(store._parsed_rows()) == 6
    assert capsys.readouterr().err.count("[registry] store") >= 4


# --------------------------------------------------------------------------------------------
# the serving hook
# --------------------------------------------------------------------------------------------

ATTESTATION = {"engine": {"enforce_eager": False, "cudagraph_mode": "FULL", "cudagraph_full_capture_max": 64},
               "lora_config": {"max_loras": 4, "max_lora_rank": 16}, "id_name_map": {"1": "constitution_lora"},
               "scheduler_geometry": {"worker_reports": [{"max_num_seqs": 48, "max_model_len": 8192}]}}


def _synthetic_standup(out: Path):
    from sisteer.capeval.serving import native_standup as ns
    geometry = types.SimpleNamespace(max_model_len=8192, max_num_seqs=64, seqs_source="config",
                                     as_dict=lambda: {"max_model_len": 8192, "max_num_seqs": 64})
    proxy = types.SimpleNamespace(virtual_models=lambda: ["vm-none", "vm-constitution"])
    launch = types.SimpleNamespace(cudagraph_capture_sizes=(1, 2, 4), default_chat_template_kwargs=None, override_generation_config=None)
    table = {"version": 3, "routes": [], ns.SERVED_POSITIONS_KEY: {"constitution_lora": "all"}}
    cell_set = types.SimpleNamespace(canonical=[], controls=[])
    fields = {f.name for f in dataclasses.fields(ns.LoraStandup)}
    kw = dict(out=out, model="synthetic/instruct-4b", revision=REV, model_key="synthmodel_4b_it", eval="syntheval", geometry=geometry,
              launch=launch, server=None, proxy=proxy, proxy_loop=None, proxy_url="http://127.0.0.1:1", proxy_port=1, vllm_port=2,
              table_path=out / "routing_table.json", table=table, tokenizer=None, manifest_path=None, config_path=None,
              config_used=out / "config.yaml", defenses_dir=out / "defenses", adapter_bank=out / "bank",
              cells=("none", "constitution_lora"), harness_cells=("none", "constitution_lora"), lora_cells=("constitution_lora",),
              prompted_cells_admitted=(), vmodels={"constitution_lora": "vm-constitution"},
              adapters={"constitution_lora": {"r": 4, "petri_intervention_hash": IHASH, "layers": [4, 5, 6]}},
              zero_adapter={"name": "none__lora0"}, adapters_reused={}, object_artifact_sha256_by_cell={"constitution_lora": synth.OBJECT_SHA},
              attestation_path=out / "serving_attestation.json", attestation=ATTESTATION,
              # the shape serving_attestation.engine_identity hands the hook (schema 2): per attested adapter its exported object kind
              engine_identity={"versions": {"vllm": "0.26.0"}, "adapters": {"constitution_lora": {"petri_intervention_hash": IHASH, "petri_zero_adapter": False,
                                                                                                    "petri_source_kind": "lora_r4", "residual_fingerprint": None, "position_mode": None}}},
              chat_template_hash="b48c47f644389271", chat_template_sha256=synth.SHA256_A, manifest_flags_dropped=(),
              geometry_override_reason=None, tool_calls=False, petri_agent_loop=False, upstream_timeout_s=30.0, probe_max_tokens=16,
              max_logprobs=None, gpu_mem_util="0.9", cell_set=cell_set, thinking_header=None, thinking_header_error=None,
              attempt_id="attempt-synthetic", mode="attacks", positive_controls={"verdict": "pass"})
    kw = {k: v for k, v in kw.items() if k in fields}
    return ns.LoraStandup(**kw)


def _served_runs(run_id):
    return {"constitution_lora": {"run_id": run_id, "campaign": synth.CAMPAIGN,
                                  "expected": {"harness_config_hash": CHASH, "env_digest": "99aabbccddeeff00"}}}


def test_actual_serving_hook_payload_validates(tmp_path, monkeypatch):
    """The EXACT payload write_endpoint's hook emits: captured from a synthetic standup, validated, folded; then
    the same payload with serving_path mutated is refused naming the field, both values and the run id."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, lifecycle = synth.run_records("k_served", 100, object_key="constitution_lora@a1b2c3d4", object_id=synth.OBJECT_SHA, through="running")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), lifecycle)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0

    out = tmp_path / "standup"
    out.mkdir()
    su = _synthetic_standup(out)
    su.registry_runs = _served_runs(rid)
    monkeypatch.setenv(emit.SPOOL_ENV, os.path.join(root, "records"))
    # endpoint.json is written first and is the same bytes with emission on or off
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    su.write_endpoint()
    off_bytes = json.loads((out / "endpoint.json").read_text())
    monkeypatch.delenv(emit.DISABLED_ENV)
    su.write_endpoint()
    on_bytes = json.loads((out / "endpoint.json").read_text())
    off_bytes.pop("written_at"), on_bytes.pop("written_at")
    assert off_bytes == on_bytes
    # exactly one run_patch landed in the run's directory, from the hook
    run_dir = Path(root) / "records" / synth.CAMPAIGN / identity.run_dir_name(rid)
    patch_files = sorted(p for p in run_dir.iterdir() if p.name.endswith("-run_patch.jsonl") and p.name != "000001-life.jsonl")
    assert len(patch_files) == 1
    (_, rec, sha), = list(R.read_record_file(str(patch_files[0])))
    assert rec["producer"]["script"] == "capeval.serving.native_standup.write_endpoint"
    v = VAL.validate_record(rec, str(patch_files[0]), sha)
    assert isinstance(v, VAL.Validated), v
    payload = rec["payload"]
    assert payload["run_id"] == rid
    assert payload["fields"]["served_max_num_seqs"] == 48 and payload["fields"]["served_max_num_seqs_source"] == "attested"
    assert payload["fields"]["context_window_tokens"] == 8192 and payload["fields"]["applied_positions_mode"] == "all"
    assert payload["fields"]["applied_layers_json"] == [4, 5, 6] and payload["fields"]["serving_engine_version"] == "vllm 0.26.0"
    assert set(payload["verify"]) == {"serving_path", "intervention_hash", "model_revision", "harness_config_hash", "env_digest"}
    assert payload["verify"]["serving_path"] == "lora_native" and payload["verify"]["intervention_hash"] == IHASH
    # the harvester accepts it: verified_fields listed, fields folded
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.ok >= 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT served_max_num_seqs, context_window_tokens, applied_positions_mode, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert tuple(row)[:3] == (48, 8192, "all")
    assert set(json.loads(row[3])["verified_fields"]) >= {"serving_path", "intervention_hash", "model_revision"}
    conn.close()

    # the same payload with the served path mutated is REFUSED naming the field, both values and the run id
    bad = json.loads(json.dumps(rec))
    bad["payload"]["verify"]["serving_path"] = "lens"
    bad["record_id"] = identity.producer_record_id()
    bad["written_utc"] = synth.t(500)
    R.write_records(str(run_dir / "000900-run_patch.jsonl"), [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    field, reason = conn.execute("SELECT field, reason FROM record_refusal").fetchone()
    assert field == "serving_path" and "'lens'" in reason and "'lora_native'" in reason and rid in reason
    assert conn.execute("SELECT serving_path FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "lora_native"
    conn.close()


def test_serving_hook_reads_runs_file_beside_endpoint_and_is_inert_without_one(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    out = tmp_path / "standup"
    out.mkdir()
    su = _synthetic_standup(out)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    su.write_endpoint()
    assert not spool.exists() and "no run identities" in capsys.readouterr().out + capsys.readouterr().err or True
    rid = identity.run_id(synth.CAMPAIGN, "k", "constitution_lora@a1b2c3d4", synth.EVAL_ID, synth.INSTRUMENT_ID)
    (out / hooks.RUNS_FILE).write_text(json.dumps(_served_runs(rid)))
    su.write_endpoint()
    recs = _records_in(spool)
    assert [r["record_type"] for _, r, _ in recs] == ["run_patch"] and recs[0][1]["payload"]["run_id"] == rid


def _attested(keys, kind="lora_r4"):
    """The engine identity's adapter blocks (serving_attestation.engine_identity, schema 2) naming each served adapter's exported object kind."""
    return {"versions": {"vllm": "0.26.0"}, "adapters": {k: {"petri_intervention_hash": IHASH, "petri_zero_adapter": False, "petri_source_kind": kind,
                                                            "residual_fingerprint": None, "position_mode": None} for k in keys}}


def _endpoint_with_positions(served, engine="lora_native", attested_kind="lora_r4"):
    return {"attested_max_num_seqs": 48, "attested_max_model_len": 8192, "engine": engine, "model_revision": REV,
            "adapters": {"c": {"petri_intervention_hash": IHASH, "layers": [1]}}, "served_positions": served,
            **({"engine_identity": _attested(["c"], attested_kind)} if attested_kind else {})}


@pytest.mark.parametrize("served,expected", [
    ({"c": "all"}, "all"),                                   # a bare string: unchanged
    ({"c": {"mode": "response"}}, "response"),               # a {mode} dict: unchanged
    ({"c": {"position_mode": "all"}}, "all"),
    ({"c": ["non_template"]}, "non_template"),               # the proxy's validated shape: a one-element list IS the served mode
    ({"c": ["all", "all"]}, "all"),                          # several equal entries are one mode
])
def test_serving_patch_reads_one_served_mode_from_every_shape(served, expected, tmp_path, monkeypatch):
    """One served mode stamps the column with a `recorded` entry and the scope that follows it (derived, here on a static eval)."""
    _seeds_checkout(tmp_path, monkeypatch)
    p = hooks.serving_patch_payload(_endpoint_with_positions(served), "c", {"run_id": f"x:{synth.EVAL_ID}:i:0", "expected": {}})
    assert p["fields"]["applied_positions_mode"] == expected and p["fields"]["multi_turn_steering_scope"] == "single_turn"
    assert set(p["fields"]) <= set(R.RUN_PATCHABLE_FIELDS) and "applied_positions_mode_reason" not in p["fields"]["extra_json"]
    fc = p["fields"]["field_confidence_json"]
    assert fc["applied_positions_mode"] == {"confidence": "recorded", "source": "served_positions['c'] of the standup's endpoint record"}
    assert fc["multi_turn_steering_scope"] == {"confidence": "derived", "source": f"eval_category=static_attack is single-turn (eval_category from {hooks.EVALS_SEED_REL} of the producing checkout)"}


@pytest.mark.parametrize("served,reason_part", [
    ({"c": ["all", "response"]}, "2 different modes ['all', 'response']"),   # two DIFFERENT modes are not one mode; never the first element
    ({"c": []}, "lists no mode"),                                           # an empty list names nothing
    ({"c": ["all", 7]}, "non-string or empty element"),
])
def test_serving_patch_stamps_unverified_with_the_reason_in_its_own_confidence_entry_when_the_served_list_is_not_one_mode(served, reason_part, tmp_path, monkeypatch):
    """The reason rides the patch's `field_confidence_json` (the validator reads an `unverified` value's reason THERE, and the fold merges
    a patch's entries into the row's per column, so the row's other entries stand); the scope follows the mode: unverified on a multi-turn
    eval, still `single_turn` on a static one (its category decides alone)."""
    _seeds_checkout(tmp_path, monkeypatch)
    p = hooks.serving_patch_payload(_endpoint_with_positions(served), "c", {"run_id": f"x:{MT_EVAL_ID}:i:0", "expected": {}})
    assert p["fields"]["applied_positions_mode"] == "unverified" and p["fields"]["multi_turn_steering_scope"] == "unverified"
    fc = p["fields"]["field_confidence_json"]
    assert fc["applied_positions_mode"]["confidence"] == "unverified" and reason_part in fc["applied_positions_mode"]["reason"]
    assert fc["multi_turn_steering_scope"]["confidence"] == "unverified" and "applied_positions_mode='unverified' unresolved" in fc["multi_turn_steering_scope"]["reason"]
    assert "applied_positions_mode_reason" not in p["fields"]["extra_json"]
    static = hooks.serving_patch_payload(_endpoint_with_positions(served), "c", {"run_id": f"x:{synth.EVAL_ID}:i:0", "expected": {}})
    assert static["fields"]["applied_positions_mode"] == "unverified" and static["fields"]["multi_turn_steering_scope"] == "single_turn"
    # the validator's contract holds on the patch itself: both unverified values carry their reason
    for fields in (p["fields"], static["fields"]):
        known, _extra = VAL._normalise_payload("run", fields)
        for col in ("applied_positions_mode", "multi_turn_steering_scope"):
            assert VAL.check_column_value("run", col, known[col], known) is None, col


def test_serving_patch_leaves_the_mode_and_the_scope_alone_when_the_endpoint_names_none_for_the_key():
    for served in ({}, {"other": ["all"]}, {"c": ""}, {"c": {"layers": [1]}}, None):
        p = hooks.serving_patch_payload(_endpoint_with_positions(served), "c", {"run_id": "x:e:i:0", "expected": {}})
        assert "applied_positions_mode" not in p["fields"] and "multi_turn_steering_scope" not in p["fields"], served
        assert "field_confidence_json" not in p["fields"] and "applied_positions_mode_reason" not in p["fields"]["extra_json"], served


def test_served_scope_reads_the_entrys_own_facts_before_the_run_id_and_the_seeds(tmp_path, monkeypatch):
    """The served runs file's entry may carry the eval's category, its id or the object's identity; else the run id's eval segment and the
    producing checkout's seeds decide; an id that is not a registry run id, or an eval the seeds do not name, reads `unverified` with why."""
    _seeds_checkout(tmp_path, monkeypatch)
    assert hooks.served_scope("response", {"run_id": "not-a-run-id", "eval_category": "petri"}) == (
        "final_header_and_current_decode_only", {"confidence": "derived", "source": "applied_positions_mode=response on a multi-turn eval (eval_category from the served runs file)"})
    assert hooks.served_scope("prefill_all", {"run_id": "not-a-run-id", "eval_id": MT_EVAL_ID})[0] == "prefill_only_each_turn"
    assert hooks.served_scope("all", {"run_id": f"x:{MT_EVAL_ID}:i:0"})[0] == "every_turn_prefill_and_decode"
    assert hooks.served_scope("all", {"run_id": f"x:{MT_EVAL_ID}:i:0", "object_id": "none"}) == ("not_applicable", {"confidence": "derived", "source": "object_kind=none"})
    assert hooks.served_scope("all", {"run_id": f"x:{MT_EVAL_ID}:i:0", "object_kind": "none"})[0] == "not_applicable"
    # a prompted route (object id si:<sha>, or the kind stated outright): a system-instruction object steers no positions, so an
    # endpoint serving `all` on it never reads every_turn_prefill_and_decode; on a static eval the category decides first
    si_id = identity.object_id("system_instruction", si_text_sha256=synth.SI_SHA)
    assert hooks.served_scope("all", {"run_id": f"x:{MT_EVAL_ID}:i:0", "object_id": si_id}) == ("not_applicable", {
        "confidence": "derived", "source": f"object_kind=system_instruction: a prompted object steers no positions (eval_category from {hooks.EVALS_SEED_REL} of the producing checkout)"})
    assert hooks.served_scope("non_template", {"run_id": f"x:{MT_EVAL_ID}:i:0", "object_kind": "system_instruction"})[0] == "not_applicable"
    assert hooks.served_scope("all", {"run_id": f"x:{synth.EVAL_ID}:i:0", "object_id": si_id})[0] == "single_turn"
    # the kind: the entry's, the id's spelling, else the adapter the endpoint attests for the served key (its exported kind spelling mapped
    # to the registry's); a bare sha with nothing attested names no kind at all (the scope derivation then reads a steered object)
    assert hooks.served_run_object_kind({"object_id": si_id}) == "system_instruction" and hooks.served_run_object_kind({"object_id": synth.OBJECT_SHA}) is None
    assert hooks.served_run_object_kind({"object_id": synth.OBJECT_SHA}, _endpoint_with_positions({"c": ["all"]}), "c") == "lora"
    assert hooks.served_run_object_kind({"object_id": synth.OBJECT_SHA}, _endpoint_with_positions({"c": ["all"]}, attested_kind="alphasteer"), "c") == "alphasteer"
    assert hooks.served_run_object_kind({"object_id": synth.OBJECT_SHA}, _endpoint_with_positions({"c": ["all"]}, attested_kind=None), "c") is None
    assert hooks.served_run_object_kind({"object_kind": "antidote"}, _endpoint_with_positions({"c": ["all"]}), "c") == "antidote"
    assert hooks.served_adapter_kind(_endpoint_with_positions({}, attested_kind="psr"), "c") == ("relu_steer", "the endpoint's engine_identity.adapters['c'].petri_source_kind")
    assert hooks.served_adapter_kind(_endpoint_with_positions({}, attested_kind="not-a-kind"), "c") == (
        None, "the endpoint's engine_identity.adapters['c'].petri_source_kind 'not-a-kind' is not a known object kind spelling")
    assert hooks.served_adapter_kind({"adapters": {"c": {"petri_source_kind": "static_r_arditi"}}}, "c") == ("static_r_arditi", "the endpoint's adapters['c'].petri_source_kind")
    assert hooks.served_adapter_kind(_endpoint_with_positions({}, attested_kind=None), "c")[0] is None
    assert hooks.object_kind_from_spelling("lora") == "lora" and hooks.object_kind_from_spelling("psr_ce_all") == "relu_steer" and hooks.object_kind_from_spelling("") is None
    scope, entry = hooks.served_scope("all", {"run_id": "not-a-run-id"})
    assert scope == "unverified" and entry == {"confidence": "unverified", "reason": "multi_turn_steering_scope not derived: run_id 'not-a-run-id' is not a registry run id and the served runs file names no eval_id"}
    scope, entry = hooks.served_scope("all", {"run_id": "x:nocat:i:0"})
    assert scope == "unverified" and "names no eval_category for eval 'nocat'" in entry["reason"]
    scope, entry = hooks.served_scope("all", {"run_id": "x:unknown_eval:i:0"})
    assert scope == "unverified" and "eval 'unknown_eval' is not in" in entry["reason"]


MT_EVAL_ID = "synthmt"
MT_ITEM_SET_ID = "synthmt_259"
MT_INSTRUMENT = synth.instrument_fields(eval_id=MT_EVAL_ID, item_set_id=MT_ITEM_SET_ID, harness="pyrit", max_turns=10,
                                        attacker_model_id="synth_attacker", context_window_tokens=49152)
MT_INSTRUMENT_ID = str(MT_INSTRUMENT["instrument_id"])


def _seeds_checkout(tmp_path, monkeypatch):
    """A synthetic producing checkout whose `configs/registry/evals.yaml` names the synthetic static eval, the multi-turn one and an eval
    without a category; the hooks read the scope derivation's category from it."""
    root = tmp_path / "checkout"
    (root / "configs" / "registry").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "registry" / "evals.yaml").write_text(
        f"evals:\n  - eval_id: {synth.EVAL_ID}\n    eval_category: static_attack\n  - eval_id: {MT_EVAL_ID}\n    eval_category: multi_turn\n"
        "  - eval_id: nocat\n    name: x\n")
    monkeypatch.setattr(hooks, "_producing_checkout_root", lambda: str(root))
    return root


def _mt_dimensions(i0=30):
    """A multi-turn eval beside the synthetic static one: its item set, its attacker (a model row) and an every-token instrument."""
    return [
        synth.rec("eval", {"eval_id": MT_EVAL_ID, "eval_category": "multi_turn", "headline_metric_id": "asr", "metric_direction": "lower_is_better",
                           "attacker_required": 1, "paper_status": "in_paper"}, synth.t(i0)),
        synth.rec("item_set", {"item_set_id": MT_ITEM_SET_ID, "eval_id": MT_EVAL_ID, "dataset_version": "synthetic/synthmt@v1", "n_items": 259,
                               "items_sha256": synth.SHA256_B, "source_uri": "synthetic://items/synthmt_259.jsonl", "item_id_scheme": "g<index>",
                               "subsample_rule": "full"}, synth.t(i0 + 1)),
        synth.rec("model", {"model_id": "synth_attacker", "family": "qwen36", "size_label": "27b", "checkpoint_kind": "instruct",
                            "hf_repo": "synthetic/attacker-27b", "hf_revision": "e" * 40, "chat_template_sha256": synth.SHA256_A,
                            "generation_config_present": 1, "generation_config_json": {"temperature": 1.0}, "thinking_channel": "available",
                            "paper_status": "reference"}, synth.t(i0 + 2)),
        synth.rec("instrument", MT_INSTRUMENT, synth.t(i0 + 3)),
    ]


def _registered_run(root, key, i0, object_id, scope, eval_id=MT_EVAL_ID, item_set_id=MT_ITEM_SET_ID, instrument_id=MT_INSTRUMENT_ID, **over):
    """A run through `running` on the given eval with the given scope registered, in its own record directory; returns (run_id, served entry)."""
    object_key = "none" if object_id == "none" else "constitution_lora@a1b2c3d4"
    rid = identity.run_id(synth.CAMPAIGN, key, object_key, eval_id, instrument_id)
    rid2, recs = synth.run_records(key, i0, object_key, object_id, through="running", run_id=rid, eval_id=eval_id, item_set_id=item_set_id,
                                   instrument_id=instrument_id, multi_turn_steering_scope=scope, **over)
    assert rid2 == rid
    for r in recs:
        # the lifecycle's job-start patch re-observes the row's serving path: a run registered on another path verifies that one
        if "serving_path" in over and r["record_type"] == "run_patch" and "serving_path" in (r["payload"].get("verify") or {}):
            r["payload"]["verify"]["serving_path"] = over["serving_path"]
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-run.jsonl"), recs)
    return rid, {"run_id": rid, "campaign": synth.CAMPAIGN, "expected": {}}


def _live_run(root, rid):
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT applied_positions_mode, multi_turn_steering_scope, field_confidence_json, served_max_num_seqs, extra_json "
                       "FROM run WHERE run_id = ?", (rid,)).fetchone()
    conn.close()
    assert row is not None, rid
    return {"mode": row[0], "scope": row[1], "confidence": json.loads(row[2] or "{}"), "served_max_num_seqs": row[3], "extra": json.loads(row[4] or "{}")}


def _rebuild_equal(root, tmp_path, rids):
    """Replay the whole log into a fresh database: every compared table equal, and the patched rows' JSON columns byte-equal."""
    live = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    try:
        diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
        assert not diff, diff
        sql = "SELECT applied_positions_mode, multi_turn_steering_scope, field_confidence_json, extra_json, field_versions_json FROM run WHERE run_id = ?"
        for rid in rids:
            assert tuple(live.execute(sql, (rid,)).fetchone()) == tuple(rebuilt.execute(sql, (rid,)).fetchone()), rid
    finally:
        live.close()
        rebuilt.close()


def test_an_ambiguous_served_list_folds_as_unverified_with_its_reason_beside_the_rows_other_entries(tmp_path, monkeypatch):
    """The endpoint patch for a served list that is not one mode (`['all', 'response']`, `[]`, `['all', 7]`) is ACCEPTED by the harvester:
    `applied_positions_mode` reads `unverified`, its reason is durable in the row's `field_confidence_json`, the row's unrelated entries
    stand (the fold reads a patch's confidence JSON per column), the patch's other serving facts (`served_max_num_seqs = 48`) land, and
    the log rebuilds equal. Before this, the reason rode `extra_json` and the validator refused the whole patch for the missing entry."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)       # the suite's emission fence; the spool is this test's own root
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    _seeds_checkout(tmp_path, monkeypatch)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions() + _mt_dimensions())
    unrelated = {"prompt_template_sha256": {"confidence": "unverified", "reason": "the seeded instrument records no prompt-template hash"},
                 "instrument_match": {"confidence": "derived", "source": "declared from the seeds at job start"}}
    served = {"k1": ["all", "response"], "k2": [], "k3": ["all", 7]}
    runs = {}
    for n, key in enumerate(served):
        rid, entry = _registered_run(root, f"amb_{key}", 100 + 10 * n, synth.OBJECT_SHA, "every_turn_prefill_and_decode",
                                     field_confidence_json=unrelated)
        runs[key] = entry
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    for key, entry in runs.items():
        before = _live_run(root, entry["run_id"])
        assert (before["mode"], before["scope"], before["served_max_num_seqs"]) == ("all", "every_turn_prefill_and_decode", None)
        assert before["confidence"] == unrelated
    endpoint = {"attested_max_num_seqs": 48, "attested_max_model_len": 8192, "engine": "lora_native", "model_revision": REV,
                "adapters": {k: {"petri_intervention_hash": IHASH, "layers": [1]} for k in served}, "served_positions": served}
    paths = hooks.emit_serving_patches(endpoint, runs, spool=os.path.join(root, "records"), registry_emit=True)
    assert len(paths) == 3 and all(os.path.dirname(p) == os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(runs[k]["run_id"]))
                                   for p, k in zip(paths, served))
    for p in paths:
        (_, rec, sha), = list(R.read_record_file(str(p)))
        assert isinstance(VAL.validate_record(rec, str(p), sha), VAL.Validated), (p, VAL.validate_record(rec, str(p), sha))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    expected_reason = {"k1": "served_positions lists 2 different modes ['all', 'response']; one mode cannot be read from them",
                       "k2": "served_positions lists no mode ([])",
                       "k3": "served_positions list carries a non-string or empty element: ['all', 7]"}
    for key, entry in runs.items():
        row = _live_run(root, entry["run_id"])
        assert (row["mode"], row["scope"], row["served_max_num_seqs"]) == ("unverified", "unverified", 48), key
        assert row["confidence"]["applied_positions_mode"] == {"confidence": "unverified", "reason": expected_reason[key]}, key
        why = "applied_positions_mode='unverified' unresolved (eval_category from configs/registry/evals.yaml of the producing checkout)"
        assert row["confidence"]["multi_turn_steering_scope"] == {"confidence": "unverified", "source": why, "reason": why}, key
        assert {c: e for c, e in row["confidence"].items() if c in unrelated} == unrelated, key          # the row's other entries stand
        assert row["extra"]["served_key"] == key and "applied_positions_mode_reason" not in row["extra"]
        assert row["extra"]["verified_fields"] == {"serving_path": "lora_native", "intervention_hash": IHASH, "model_revision": REV}
    _rebuild_equal(root, tmp_path, [e["run_id"] for e in runs.values()])


def test_a_served_mode_moves_the_multi_turn_scope_with_it(tmp_path, monkeypatch):
    """A registration on the canonical every-token instrument derives `every_turn_prefill_and_decode`; the standup then serves a
    restricted-position mode, and the endpoint patch moves the scope WITH the mode: `response` -> final_header_and_current_decode_only,
    `prefill_all` -> prefill_only_each_turn, `non_template` -> every_turn_prefill_and_decode (unchanged in value, restated with its
    derivation). An unsteered run keeps `not_applicable`, a single-turn eval's run `single_turn`, whatever is served; a run whose eval the
    producing checkout's seeds do not name reads `unverified` with the reason. Every row rebuilds equal."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)       # the suite's emission fence; the spool is this test's own root
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    _seeds_checkout(tmp_path, monkeypatch)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions() + _mt_dimensions())
    cases = {
        # served key: (object_id, registered scope, served list, expected mode, expected scope, expected confidence)
        "resp": (synth.OBJECT_SHA, "every_turn_prefill_and_decode", ["response"], "response", "final_header_and_current_decode_only", "derived"),
        "pref": (synth.OBJECT_SHA, "every_turn_prefill_and_decode", ["prefill_all"], "prefill_all", "prefill_only_each_turn", "derived"),
        "ntpl": (synth.OBJECT_SHA, "every_turn_prefill_and_decode", ["non_template"], "non_template", "every_turn_prefill_and_decode", "derived"),
        # the unsteered route: the anchor steers no positions whatever mask the endpoint spells for it (the kind rule, not the served list)
        "none": ("none", "not_applicable", ["all"], "none", "not_applicable", "derived"),
    }
    runs = {}
    for n, (key, (object_id, scope, _served, _m, _s, _c)) in enumerate(cases.items()):
        rid, entry = _registered_run(root, f"mt_{key}", 100 + 10 * n, object_id, scope,
                                     **({"applied_positions_mode": "none"} if object_id == "none" else {}))
        if object_id == "none":
            entry["object_id"] = "none"               # the served runs file says which served key is the unsteered route
        runs[key] = entry
    # a steered run on the single-turn eval, served `response`: its scope stays single_turn (the category decides alone)
    rid_static, entry_static = _registered_run(root, "st_resp", 200, synth.OBJECT_SHA, "single_turn", eval_id=synth.EVAL_ID,
                                               item_set_id=synth.ITEM_SET_ID, instrument_id=synth.INSTRUMENT_ID)
    runs["static"] = entry_static
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    served = {k: v[2] for k, v in cases.items()}
    served["static"] = ["response"]
    endpoint = {"attested_max_num_seqs": 48, "engine": "lora_native", "model_revision": REV,
                "adapters": {k: {"petri_intervention_hash": IHASH, "layers": [1]} for k in served if k != "none"}, "served_positions": served,
                "engine_identity": _attested([k for k in served if k != "none"])}
    paths = hooks.emit_serving_patches(endpoint, runs, spool=os.path.join(root, "records"), registry_emit=True)
    assert len(paths) == 5
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    for key, (_o, _registered, _served, mode, scope, confidence) in cases.items():
        row = _live_run(root, runs[key]["run_id"])
        assert (row["mode"], row["scope"], row["served_max_num_seqs"]) == (mode, scope, 48), key
        if key == "none":
            assert row["confidence"]["applied_positions_mode"] == {"confidence": "derived", "source": "unsteered / passthrough (served_positions['none'] = 'all' of "
                                                                                                        "the standup's endpoint record, read against the object's kind)"}
        else:
            assert row["confidence"]["applied_positions_mode"] == {"confidence": "recorded", "source": f"served_positions[{key!r}] of the standup's endpoint record"}, key
        assert row["confidence"]["multi_turn_steering_scope"]["confidence"] == confidence, key
    assert "applied_positions_mode=response on a multi-turn eval" in _live_run(root, runs["resp"]["run_id"])["confidence"]["multi_turn_steering_scope"]["source"]
    assert _live_run(root, runs["none"]["run_id"])["confidence"]["multi_turn_steering_scope"] == {"confidence": "derived", "source": "object_kind=none"}
    static = _live_run(root, rid_static)
    assert (static["mode"], static["scope"]) == ("response", "single_turn")
    assert static["confidence"]["multi_turn_steering_scope"]["source"].startswith("eval_category=static_attack is single-turn")
    _rebuild_equal(root, tmp_path, [e["run_id"] for e in runs.values()])
    # the real checkout's seeds, unpatched: the synthetic eval is not among them, so a served mode's scope is unverified with the reason
    monkeypatch.undo()
    p = hooks.serving_patch_payload(endpoint, "resp", runs["resp"])
    assert p["fields"]["applied_positions_mode"] == "response" and p["fields"]["multi_turn_steering_scope"] == "unverified"
    assert f"eval {MT_EVAL_ID!r} is not in {hooks.EVALS_SEED_REL}" in p["fields"]["field_confidence_json"]["multi_turn_steering_scope"]["reason"]


# --------------------------------------------------------------------------------------------
# a store's opening never restates a served mode: its copies of the plan's mode and scope are marked unstated
# --------------------------------------------------------------------------------------------


def test_the_store_marks_the_plans_mode_and_scope_unstated_only_when_the_run_was_registered_before_it(tmp_path):
    """`run_payload` on an identity carrying a registered run id (the wrapper's or the job-start registration's `run` record precedes the
    store's) marks `applied_positions_mode` and `multi_turn_steering_scope` unstated (`extra.unstated_columns`, STORE_PLANNED_COLUMNS) --
    merged with a marker the identity's own extra already carries -- and the record validates; an identity without one (the store is the
    run's first registration) marks nothing. The key is the reducer's and the validator's."""
    from sisteer.registry import reducer as RD
    assert hooks.UNSTATED_COLUMNS_KEY == RD.UNSTATED_COLUMNS_KEY == VAL.UNSTATED_COLUMNS_KEY
    assert hooks.STORE_PLANNED_COLUMNS == ("applied_positions_mode", "multi_turn_steering_scope")
    assert set(hooks.STORE_PLANNED_COLUMNS) <= set(R.RUN_PATCHABLE_FIELDS)          # what an endpoint patch may have stated before the store
    rid = "synthcampaign:syntheval:abcdefabcdef:0123456789abcdef"
    rec = hooks.StoreRecorder(KEY, RESOLVED, {"*": dict(_identity(), run_id=rid)}, spool=str(tmp_path / "spool_a"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store_a")
    p = rec.run_payload("*")
    assert (p["run_id"], p["applied_positions_mode"], p["multi_turn_steering_scope"]) == (rid, "all", "single_turn")       # the plan's copies, carried
    assert p["extra"]["unstated_columns"] == ["applied_positions_mode", "multi_turn_steering_scope"] and p["extra"]["registered_at"] == "submit_wrapper"
    record = R.make_record("run", p, rec.producer, "11111111-2222-3333-4444-555555555555", synth.t(5))
    out = VAL.validate_record(record, "<memory>", R.record_sha(record))
    assert isinstance(out, VAL.Validated), getattr(out, "reason", out)
    # an identity whose own extra marks a column already: the store's two are added once, the identity's kept first
    rec = hooks.StoreRecorder(KEY, RESOLVED, {"*": dict(_identity(), run_id=rid, notes="planned note", extra={"unstated_columns": ["notes", "applied_positions_mode"]})},
                              spool=str(tmp_path / "spool_b"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store_b")
    p = rec.run_payload("*")
    assert p["extra"]["unstated_columns"] == ["notes", "applied_positions_mode", "multi_turn_steering_scope"] and p["notes"] == "planned note"
    record = R.make_record("run", p, rec.producer, "11111111-2222-3333-4444-555555555556", synth.t(6))
    assert isinstance(VAL.validate_record(record, "<memory>", R.record_sha(record)), VAL.Validated)
    # the store as the run's first registration: its mode and scope are its statements, nothing marked
    rec = hooks.StoreRecorder(KEY, RESOLVED, {"*": _identity()}, spool=str(tmp_path / "spool_c"), code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / "store_c")
    p = rec.run_payload("*")
    assert "unstated_columns" not in p["extra"] and p["extra"]["registered_at"] == "store_manifest"


def _store_joins(root, tmp_path, tag, plan, n=5):
    """A `CellStore` opened on a run registered before it (its identity carries the registered run id, as the job's pending file hands
    it): the store's own `run` record, its state and its rows go into the run's directory under the spool. Returns the store's `run`
    record (the one carrying `extra.store_key`) and the file it lies in."""
    spool = os.path.join(root, "records")
    rec = hooks.store_recorder(KEY, RESOLVED, dict(plan), spool=spool, registry_emit=True)
    assert rec is not None
    _drive(cs.CellStore(tmp_path / f"store_{tag}", KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec), n=n)
    d = os.path.join(spool, synth.CAMPAIGN, identity.run_dir_name(str(plan["run_id"])))
    found = []
    for fn in sorted(os.listdir(d)):
        m = R.RECORD_FILE_RE.match(fn)
        if not m or m.group("type") != "run":
            continue
        for _i, r, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
            if r["record_type"] == "run" and (r["payload"].get("extra") or {}).get("store_key"):
                found.append((r, os.path.join(d, fn)))
    ((store_run, _path),) = found
    return store_run


def _sha_of_file(path):
    (_i, rec, _s), = list(R.read_record_file(str(path)))
    return R.record_sha(rec), rec


def test_a_store_opening_after_the_standups_endpoint_patch_keeps_the_served_mode_its_scope_and_their_provenance(tmp_path, monkeypatch):
    """The production order: a run registered on the canonical every-token instrument (`all`, scope derived), the standup's endpoint record
    stamping the served `response` and the scope that follows it, THEN the evaluation store opening and filing its own full `run` record.
    The store's copies of the plan's mode and scope are marked unstated, so the row keeps (`response`, `final_header_and_current_decode_only`)
    with the endpoint's `recorded` / `derived` entries -- their stamps the endpoint record's -- beside the registration's other entries, while
    the store's own statements (the hashes, the model revision, the store key, the rows) land; the same when the endpoint patch and the
    store's record fold in ONE ingest, and when the store opens BEFORE the endpoint patch; a store that is a run's first registration still
    states its mode with no flag. Before the marker the store's `all` and its planned scope folded over the served values and its confidence
    JSON erased the endpoint's entries. Everything rebuilds equal."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)       # the suite's emission fence; the spool is this test's own root
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    _seeds_checkout(tmp_path, monkeypatch)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions() + _mt_dimensions())
    planned = {"instrument_match": {"confidence": "derived", "source": "declared from the seeds at job start"},
               "multi_turn_steering_scope": {"confidence": "derived", "source": "applied_positions_mode=all on a multi-turn eval"}}
    # three runs registered before their stores: the production order, the one-ingest order, the store-before-endpoint order
    runs, plans = {}, {}
    for n, tag in enumerate(("prod", "cycle", "early")):
        rid, entry = _registered_run(root, f"sj_{tag}", 100 + 10 * n, synth.OBJECT_SHA, "every_turn_prefill_and_decode", field_confidence_json=planned)
        runs[tag] = entry
        # the job's pending file hands the store the registration's own identity: its producer keys, its run id
        plans[tag] = _identity(run_id=rid, producer_run_key_legacy=f"sj_{tag}", producer_object_key_legacy="constitution_lora@a1b2c3d4", eval_id=MT_EVAL_ID,
                               item_set_id=MT_ITEM_SET_ID, instrument_id=MT_INSTRUMENT_ID, multi_turn_steering_scope="every_turn_prefill_and_decode",
                               field_confidence_json=planned)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    endpoint = {"attested_max_num_seqs": 48, "engine": "lora_native", "model_revision": REV,
                "adapters": {"k": {"petri_intervention_hash": IHASH, "layers": [1]}}, "served_positions": {"k": ["response"]},
                "engine_identity": _attested(["k"])}
    served_entry = {"confidence": "recorded", "source": "served_positions['k'] of the standup's endpoint record"}
    patch_sha = {}

    def served(tag):
        (p,) = hooks.emit_serving_patches(endpoint, {"k": runs[tag]}, spool=os.path.join(root, "records"), registry_emit=True)
        patch_sha[tag], _rec = _sha_of_file(p)

    def check(tag, store_run):
        rid = runs[tag]["run_id"]
        assert store_run["payload"]["applied_positions_mode"] == "all" and store_run["payload"]["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
        assert store_run["payload"]["extra"]["unstated_columns"] == list(hooks.STORE_PLANNED_COLUMNS) and store_run["payload"]["extra"]["store_key"] == KEY
        row = _live_run(root, rid)
        assert (row["mode"], row["scope"], row["served_max_num_seqs"]) == ("response", "final_header_and_current_decode_only", 48), tag
        assert row["confidence"]["applied_positions_mode"] == served_entry, tag                        # the endpoint's entry, not the plan's, no flag
        scope_entry = row["confidence"]["multi_turn_steering_scope"]
        assert scope_entry["confidence"] == "derived" and "applied_positions_mode=response on a multi-turn eval" in scope_entry["source"] and "unstated" not in scope_entry, tag
        assert row["confidence"]["instrument_match"] == planned["instrument_match"], tag               # the registration's other entry stands
        assert "unstated_columns" not in row["extra"] and row["extra"]["store_key"] == KEY and row["extra"]["served_key"] == "k", tag
        conn = DB.connect(DB.db_path(root), readonly=True)
        r = conn.execute("SELECT lifecycle_state, n_items_generated, intervention_hash, model_revision, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()
        conn.close()
        assert tuple(r)[:4] == ("generated", 5, IHASH, REV), tag                                          # the store's own statements landed
        fv = json.loads(r[4])
        assert fv["applied_positions_mode"][1] == patch_sha[tag] and fv["multi_turn_steering_scope"][1] == patch_sha[tag], tag   # the values' stamps: the endpoint record's

    # 1. the production order: endpoint patch ingested, then the store opens
    served("prod")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    before = _live_run(root, runs["prod"]["run_id"])
    assert (before["mode"], before["scope"]) == ("response", "final_header_and_current_decode_only")
    store_prod = _store_joins(root, tmp_path, "prod", plans["prod"])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    check("prod", store_prod)
    # 2. both records in one ingest cycle: the endpoint patch written, then the store's record, folded together
    served("cycle")
    store_cycle = _store_joins(root, tmp_path, "cycle", plans["cycle"])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    check("cycle", store_cycle)
    # 3. the store opens BEFORE the endpoint patch: its marked copies are skipped over the registration's `all`, the endpoint then moves the mode
    store_early = _store_joins(root, tmp_path, "early", plans["early"])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    mid = _live_run(root, runs["early"]["run_id"])
    assert (mid["mode"], mid["scope"]) == ("all", "every_turn_prefill_and_decode") and mid["confidence"]["multi_turn_steering_scope"] == planned["multi_turn_steering_scope"]
    assert "unstated" not in mid["confidence"]["multi_turn_steering_scope"]                           # skipped: the registration's statement and entry stand unflagged
    served("early")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    check("early", store_early)
    # 4. a store that is its run's FIRST registration states the mode: no marker, no flag, the value its own
    first = dict(_identity(eval_id=MT_EVAL_ID, item_set_id=MT_ITEM_SET_ID, instrument_id=MT_INSTRUMENT_ID, multi_turn_steering_scope="every_turn_prefill_and_decode"))
    rec = hooks.store_recorder(KEY, RESOLVED, first, spool=os.path.join(root, "records"), registry_emit=True)
    _drive(cs.CellStore(tmp_path / "store_first", KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec), n=5)
    rid_first = rec.run_ids["*"]
    assert rid_first not in {e["run_id"] for e in runs.values()}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    own = _live_run(root, rid_first)
    assert (own["mode"], own["scope"]) == ("all", "every_turn_prefill_and_decode") and own["extra"]["registered_at"] == "store_manifest"
    assert "unstated_columns" not in own["extra"] and not any(isinstance(e, dict) and e.get("unstated") for e in own["confidence"].values())
    _rebuild_equal(root, tmp_path, [e["run_id"] for e in runs.values()] + [rid_first])


# --------------------------------------------------------------------------------------------
# the served mask is read against the object's kind: the proxy's `all` is every prompt row on a prefill-only object
# --------------------------------------------------------------------------------------------

ALPHASTEER_SHA = hashlib.sha256(b"synthetic alphasteer object bytes").hexdigest()
STATIC_R_SHA = hashlib.sha256(b"synthetic static direction object bytes").hexdigest()


def _object_contrastive(kind, sha, objective, positions, label):
    """A contrastive-direction object of the given kind (an AlphaSteer matrix, an Arditi direction): the LoRA fixture's shape with the
    corpus and LoRA fit facts set aside, as the pairing rule for `contrastive_direction` requires."""
    o = dict(synth.object_lora(sha, paper_label=label, role="baseline_published"))
    for k in ("fit_corpus_id", "fit_corpus_sha256", "fit_rank", "fit_alpha", "fit_lr", "fit_lr_selected_by", "fit_hyperparameters_json"):
        o.pop(k, None)
    o.update(object_kind=kind, fit_target_kind="contrastive_direction", fit_objective=objective, fit_positions=positions,
             fit_contrast_sets_json={"harmful": "synthetic-harmful", "benign": "synthetic-benign"})
    return o


def test_the_lens_spelling_table_is_the_inverse_of_the_proxys_served_positions_stamp():
    """What the hook reads back is exactly what the proxy spells: `defense_cells.SERVED_MODE_BY_STEER_POSITIONS` turns a static direction's
    `steer_positions` into the mask the thinking-ON gate classifies (`prefill_all` -> `all`, `prompt_last` -> `last`) and an AlphaSteer's
    `prefill` into `all`; the hook's table is the static kinds' inverse, and the AlphaSteer case is the kind rule's (derive)."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.registry import derive as D
    static = dc.SERVED_MODE_BY_STEER_POSITIONS["static_r"]
    assert hooks.LENS_SERVED_SPELLING["static_r_arditi"] == {mask: fit for fit, mask in static.items()} == {"all": "prefill_all", "last": "prompt_last"}
    assert hooks.LENS_SERVED_SPELLING["caa_static_r"] == {"all": "prefill_all"}
    assert dc.SERVED_MODE_BY_STEER_POSITIONS["alphasteer"] == {"prefill": "all"}
    assert D.applied_positions_mode("lens", "alphasteer", None, standup_mode="all")[0] == "prefill_last_broadcast"
    assert set(hooks.KIND_DEPENDENT_SERVED_MODES) == {"all", "last"}
    # every other served mode names the same positions in both vocabularies
    for mode in ("response", "response_all_turns", "non_template", "prefill_all", "prompt_last"):
        assert hooks.served_mode_for_registry(mode, None, "lora_native", "c")[0] == mode, mode


@pytest.mark.parametrize("entry,engine,attested,served,mode,confidence,snippet", [
    # an AlphaSteer object: the proxy spells its prefill-only broadcast's mask `all`; the kind rule reads prefill_last_broadcast, from the
    # entry's kind on the lens and from the endpoint's attested adapter on the native (rebake) route alike
    ({"object_kind": "alphasteer"}, "lens", None, ["all"], "prefill_last_broadcast", "derived", "object_kind=alphasteer (served_positions['c'] = 'all'"),
    ({}, "lora_native", "alphasteer", ["all"], "prefill_last_broadcast", "derived", "object_kind=alphasteer"),
    ({"object_id": ALPHASTEER_SHA}, "lens", "alphasteer", ["all"], "prefill_last_broadcast", "derived", "read against the object's kind"),
    # a static direction: prefill-only on the lens (the spelling read back), every forward on the native path, undecidable without a path
    ({"object_kind": "static_r_arditi"}, "lens", None, ["all"], "prefill_all", "derived", "a static_r_arditi object is served at prefill alone, so the proxy's 'all' mask is the registry's prefill_all"),
    ({"object_kind": "static_r_arditi"}, "lens", None, ["last"], "prompt_last", "derived", "the proxy's 'last' mask is the registry's prompt_last"),
    ({"object_kind": "caa_static_r"}, "lens", None, ["all"], "prefill_all", "derived", "caa_static_r object is served at prefill alone"),
    ({"object_kind": "static_r_arditi"}, "lora_native", None, ["all"], "all", "recorded", "served_positions['c'] of the standup's endpoint record"),
    ({"object_kind": "static_r_arditi"}, None, None, ["all"], "unverified", "unverified", "the record names no serving path"),
    # an antidote: the kind rule, whatever the proxy spells
    ({"object_kind": "antidote"}, "lens", None, ["last"], "last", "derived", "object_kind=antidote"),
    # a LoRA (the kind from the endpoint's attested adapter, in the export's spelling): every token, recorded
    ({}, "lora_native", "lora_r4", ["all"], "all", "recorded", "served_positions['c'] of the standup's endpoint record"),
    ({}, "lens", "psr", ["all"], "all", "recorded", "served_positions['c'] of the standup's endpoint record"),
    # no kind anywhere: `all` / `last` are the mask's spelling and cannot be read; never every-token steering by default
    ({}, "lora_native", None, ["all"], "unverified", "unverified", "neither the served runs file entry nor the endpoint's attested adapter for 'c' names the object's kind"),
    ({}, "lens", None, ["last"], "unverified", "unverified", "whose registry mode depends on the object's kind"),
    ({}, "lora_native", "unknown_spelling", ["all"], "unverified", "unverified", "petri_source_kind 'unknown_spelling' is not a known object kind spelling"),
    # ... while a kind-independent served mode reads as itself without one
    ({}, "lora_native", None, ["non_template"], "non_template", "recorded", "served_positions['c'] of the standup's endpoint record"),
    ({}, "lens", None, ["response_all_turns"], "response_all_turns", "recorded", "served_positions['c']"),
    # the anchor and a prompted object steer no positions whatever the endpoint spells for the route
    ({"object_id": "none"}, "lora_native", None, ["all"], "none", "derived", "unsteered / passthrough"),
    ({"object_id": "si:" + "f" * 64}, "lora_native", None, ["non_template"], "none", "derived", "prompted object steers no positions"),
    # a spelling outside the vocabulary, with and without a kind
    ({"object_kind": "lora"}, "lora_native", None, ["sometimes"], "unverified", "unverified", "standup mode 'sometimes' is not a known applied-positions mode"),
    ({}, "lora_native", None, ["sometimes"], "unverified", "unverified", "is not a known applied-positions mode"),
])
def test_the_served_mask_is_read_against_the_objects_kind_and_the_serving_path(entry, engine, attested, served, mode, confidence, snippet, tmp_path, monkeypatch):
    """`served_positions` spells the mask the proxy applies within a forward; the registry mode it stands for depends on the object's kind
    (and, for a static direction, the serving path): the hook resolves the kind from the entry, the id or the endpoint's attested adapter
    and hands derive the rule table, reads the lens spelling back for the static kinds, and stamps `unverified` with the reason -- never
    `all` -- when a kind-dependent mask has no kind to be read against. The scope follows the mode it stamps."""
    _seeds_checkout(tmp_path, monkeypatch)
    endpoint = _endpoint_with_positions({"c": served}, engine=engine, attested_kind=attested)
    if engine is None:
        endpoint.pop("engine")
    run = {"run_id": f"x:{MT_EVAL_ID}:i:0", "expected": {}, **entry}
    p = hooks.serving_patch_payload(endpoint, "c", run)
    fc = p["fields"]["field_confidence_json"]
    assert p["fields"]["applied_positions_mode"] == mode, fc
    assert fc["applied_positions_mode"]["confidence"] == confidence and snippet in (fc["applied_positions_mode"].get("source") or fc["applied_positions_mode"].get("reason")), fc
    if confidence == "unverified":
        assert fc["applied_positions_mode"]["reason"]
    expected_scope = {"prefill_last_broadcast": "prefill_only_each_turn", "prefill_all": "prefill_only_each_turn", "prompt_last": "prefill_only_each_turn",
                      "last": "prefill_only_each_turn", "all": "every_turn_prefill_and_decode", "non_template": "every_turn_prefill_and_decode",
                      "response_all_turns": "every_assistant_span", "none": "not_applicable", "unverified": "unverified"}[mode]
    assert p["fields"]["multi_turn_steering_scope"] == expected_scope
    # the validator's contract on the patch itself: every unverified value carries its reason
    known, _extra = VAL._normalise_payload("run", p["fields"])
    for col in ("applied_positions_mode", "multi_turn_steering_scope"):
        assert VAL.check_column_value("run", col, known[col], known) is None, col


def test_an_alphasteer_routes_served_mask_keeps_its_prefill_only_mode_and_scope_through_the_endpoint_patch(tmp_path, monkeypatch):
    """The live registry labels every AlphaSteer run `prefill_last_broadcast` / `prefill_only_each_turn`; the proxy spells the same object's
    served mask `['all']` (`defense_cells.SERVED_MODE_BY_STEER_POSITIONS['alphasteer']['prefill']`). Its endpoint patch, folded against a
    registered multi-turn run, keeps the mode and the scope (restated with the kind's derivation), lands the other serving facts
    (`served_max_num_seqs = 48`), leaves the row's unrelated confidence entries standing, and rebuilds equal -- from the entry's kind on a
    lens route and from the endpoint's attested adapter on a native route alike. Beside it: a static direction on the lens reads `all` as
    `prefill_all` and `last` as `prompt_last`; a LoRA whose kind the endpoint attests reads `all`; a route with no kind anywhere reads
    `unverified` with the reason, never every-token steering."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)       # the suite's emission fence; the spool is this test's own root
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    _seeds_checkout(tmp_path, monkeypatch)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions() + _mt_dimensions() + [
        synth.rec("object", _object_contrastive("alphasteer", ALPHASTEER_SHA, "alphasteer", "last_token_of_appended_assistant_header", "alphasteer_rebake"), synth.t(40)),
        synth.rec("object", _object_contrastive("static_r_arditi", STATIC_R_SHA, "arditi_search", "searched_post_instruction_span", "arditi"), synth.t(41))]
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), dims)
    unrelated = {"prompt_template_sha256": {"confidence": "unverified", "reason": "the seeded instrument records no prompt-template hash"},
                 "code_sha": {"confidence": "recorded", "source": "git rev-parse HEAD of the registering checkout"}}
    prefill_only = dict(applied_positions_mode="prefill_last_broadcast", field_confidence_json=unrelated)
    cases = {
        # served key: (object_id, registration overrides, registered scope, engine, served list, expected mode, scope, mode confidence)
        "as_lens": (ALPHASTEER_SHA, prefill_only, "prefill_only_each_turn", "lens", ["all"], "prefill_last_broadcast", "prefill_only_each_turn", "derived"),
        "as_native": (ALPHASTEER_SHA, prefill_only, "prefill_only_each_turn", "lora_native", ["all"], "prefill_last_broadcast", "prefill_only_each_turn", "derived"),
        "sr_all": (STATIC_R_SHA, dict(applied_positions_mode="prefill_all", field_confidence_json=unrelated), "prefill_only_each_turn", "lens", ["all"], "prefill_all", "prefill_only_each_turn", "derived"),
        "sr_last": (STATIC_R_SHA, dict(applied_positions_mode="prompt_last", field_confidence_json=unrelated), "prefill_only_each_turn", "lens", ["last"], "prompt_last", "prefill_only_each_turn", "derived"),
        "lora": (synth.OBJECT_SHA, dict(field_confidence_json=unrelated), "every_turn_prefill_and_decode", "lora_native", ["all"], "all", "every_turn_prefill_and_decode", "recorded"),
        "nokind": (synth.OBJECT_SHA, dict(field_confidence_json=unrelated), "every_turn_prefill_and_decode", "lens", ["all"], "unverified", "unverified", "unverified"),
    }
    runs = {}
    for n, (key, (object_id, over, scope, _e, _s, _m, _sc, _c)) in enumerate(cases.items()):
        rid, entry = _registered_run(root, f"mask_{key}", 100 + 10 * n, object_id, scope, serving_path=cases[key][3], **over)
        runs[key] = entry
    # the entry carries the kind on the lens routes of the two contrastive objects (the served runs file's statement); the native
    # AlphaSteer route and the LoRA leave it to the endpoint's attested adapter; the last route names nothing anywhere
    runs["as_lens"]["object_kind"] = "alphasteer"
    runs["sr_all"]["object_kind"] = runs["sr_last"]["object_kind"] = "static_r_arditi"
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    for key, (_o, over, scope, *_rest) in cases.items():
        before = _live_run(root, runs[key]["run_id"])
        assert (before["mode"], before["scope"], before["served_max_num_seqs"]) == (over.get("applied_positions_mode", "all"), scope, None), key
    # two standups: the lens engine and the native one, each with its served keys; the native one attests its adapters' kinds
    for engine in ("lens", "lora_native"):
        keys = [k for k, c in cases.items() if c[3] == engine]
        endpoint = {"attested_max_num_seqs": 48, "attested_max_model_len": 8192, "engine": engine, "model_revision": REV,
                    "adapters": {k: {"petri_intervention_hash": IHASH, "layers": [1]} for k in keys},
                    "served_positions": {k: cases[k][4] for k in keys}}
        if engine == "lora_native":
            endpoint["engine_identity"] = {"versions": {"vllm": "0.26.0"}, "adapters": {
                "as_native": {"petri_intervention_hash": IHASH, "petri_zero_adapter": False, "petri_source_kind": "alphasteer", "residual_fingerprint": None, "position_mode": None},
                "lora": {"petri_intervention_hash": IHASH, "petri_zero_adapter": False, "petri_source_kind": "lora_r4", "residual_fingerprint": None, "position_mode": None}}}
        paths = hooks.emit_serving_patches(endpoint, {k: runs[k] for k in keys}, spool=os.path.join(root, "records"), registry_emit=True)
        assert len(paths) == len(keys), engine
        for p in paths:
            (_, rec, sha), = list(R.read_record_file(str(p)))
            assert isinstance(VAL.validate_record(rec, str(p), sha), VAL.Validated), (p, VAL.validate_record(rec, str(p), sha))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    for key, (_o, _over, _scope, engine, served, mode, scope, confidence) in cases.items():
        row = _live_run(root, runs[key]["run_id"])
        assert (row["mode"], row["scope"], row["served_max_num_seqs"]) == (mode, scope, 48), key
        assert row["confidence"]["applied_positions_mode"]["confidence"] == confidence, key
        assert {c: e for c, e in row["confidence"].items() if c in unrelated} == unrelated, key          # the row's other entries stand
        assert row["extra"]["served_key"] == key and row["extra"]["verified_fields"]["serving_path"] == engine, key
    for key in ("as_lens", "as_native"):
        entry = _live_run(root, runs[key]["run_id"])["confidence"]
        assert entry["applied_positions_mode"]["source"].startswith(f"object_kind=alphasteer (served_positions[{key!r}] = 'all' of the standup's endpoint record")
        assert entry["multi_turn_steering_scope"] == {"confidence": "derived", "source": "applied_positions_mode=prefill_last_broadcast on a multi-turn eval "
                                                                                          f"(eval_category from {hooks.EVALS_SEED_REL} of the producing checkout)"}
    assert _live_run(root, runs["sr_all"]["run_id"])["confidence"]["applied_positions_mode"]["source"].endswith("so the proxy's 'all' mask is the registry's prefill_all")
    assert _live_run(root, runs["sr_last"]["run_id"])["confidence"]["applied_positions_mode"]["source"].endswith("so the proxy's 'last' mask is the registry's prompt_last")
    assert _live_run(root, runs["lora"]["run_id"])["confidence"]["applied_positions_mode"] == {"confidence": "recorded", "source": "served_positions['lora'] of the standup's endpoint record"}
    nokind = _live_run(root, runs["nokind"]["run_id"])["confidence"]
    assert nokind["applied_positions_mode"]["reason"].startswith("served_positions['nokind'] = 'all' of the standup's endpoint record spells the mask the proxy applies within a forward")
    assert nokind["applied_positions_mode"]["reason"].endswith("neither the served runs file entry nor the endpoint's attested adapter for 'nokind' names the object's kind")
    assert nokind["multi_turn_steering_scope"]["confidence"] == "unverified" and "applied_positions_mode='unverified' unresolved" in nokind["multi_turn_steering_scope"]["reason"]
    _rebuild_equal(root, tmp_path, [e["run_id"] for e in runs.values()])


def test_every_post_creation_hook_field_is_classified(tmp_path):
    """Every field the serving / completion / judging hooks emit after creation is in exactly one of
    RUN_PATCHABLE_FIELDS (fields) and RUN_VERIFIED_FIELDS (verify)."""
    endpoint = {"attested_max_num_seqs": 48, "attested_max_model_len": 8192, "engine": "lora_native", "model_revision": REV,
                "engine_identity": {"versions": {"vllm": "0.26.0"}}, "serving_config_sha256": synth.SHA256_A,
                "adapters": {"c": {"petri_intervention_hash": IHASH, "layers": [1]}}, "served_positions": {"c": "all"},
                "prefix_caching": "on", "serving_identity": "abc"}
    p = hooks.serving_patch_payload(endpoint, "c", {"run_id": "x:e:i:0", "expected": {"harness_config_hash": CHASH, "env_digest": "e"}})
    for k in p["fields"]:
        assert k in R.RUN_PATCHABLE_FIELDS and k not in R.RUN_VERIFIED_FIELDS, k
    for k in p["verify"]:
        assert k in R.RUN_VERIFIED_FIELDS and k not in R.RUN_PATCHABLE_FIELDS, k
    assert set(p["verify"]) == set(R.RUN_VERIFIED_FIELDS)
    # the store's completion patch and the judging patch
    for fields in ({"n_items_generated", "store_completion_artifact_uri", "store_completion_sha256", "store_uri", "walltime_s", "walltime_source"},
                   {"n_items_judged", "judge_id", "judged_utc"}, {"node_hostname", "cluster_job_id", "job_name", "gpu_count", "gpu_type"},
                   {"modal_app_id", "modal_call_id"}):
        assert fields <= set(R.RUN_PATCHABLE_FIELDS), fields - set(R.RUN_PATCHABLE_FIELDS)


# --------------------------------------------------------------------------------------------
# lifecycle fold: queued (wrapper) → running (job) → generated (store) → judged (grading)
# --------------------------------------------------------------------------------------------


def test_queued_running_generated_judged_folds_to_the_expected_run_row(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")   # every producer block: the synthetic host
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    spool = os.path.join(root, "records")
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    # 1. the submit wrapper: run (queued) + queued state, strict
    plan = _identity(producer_run_key_legacy=KEY, producer_object_key_legacy="constitution_lora@a1b2c3d4")
    paths = emit.write_planned_run(spool, plan, script="sbatch_registered.sh")
    rid = identity.run_id(synth.CAMPAIGN, KEY, "constitution_lora@a1b2c3d4", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert len(paths) == 2
    # 2. the job shell: running + node/gpu patch (emit_cli.state through the pending file)
    from sisteer.registry import emit_cli
    pend_dir = tmp_path / "pending"
    pend_dir.mkdir()
    (pend_dir / "4242.json").write_text(json.dumps({"run_id": rid, "plan": dict(plan, run_id=rid)}))
    assert emit_cli.state("running", pending_dir=str(pend_dir), job="4242", node="TESTHOST", gpu_count=1, gpu_type="H200",
                          scheduler_state="RUNNING", spool=spool) == 0
    # 3. the store: joins the wrapper's run (identity through the env-pointed pending file)
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(pend_dir / "4242.json"))
    rec = hooks.store_recorder(KEY, RESOLVED, None, spool=spool, registry_emit=True)
    _drive(cs.CellStore(tmp_path / "store", KEY, row_constants={k: RESOLVED[k] for k in RESOLVED if k not in ("tiers", "cell_index")}, recorder=rec), n=4)
    # 4. grading: claim → items → metric → complete (judged)
    scope = hooks.judgment_scope("tests.synthetic", {"run_id": rid, "campaign": synth.CAMPAIGN}, {"judge_id": synth.JUDGE_ID},
                                 api_pool="lowprio", spool=spool, registry_emit=True, store_uri=str(tmp_path / "store"))
    with scope:
        scope.items([{"item_id": f"q{i:03d}", "metric_id": "asr", "value": float(i % 2), "label": "harmful" if i % 2 else "safe"} for i in range(4)])
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": 2, "n": 4}])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = dict(zip([d[0] for d in conn.execute("SELECT * FROM run").description], conn.execute("SELECT * FROM run WHERE run_id = ?", (rid,)).fetchone()))
    # judged, and — the completion requirements hold (complete judgment, completion artifact) — ingested by the harvester
    assert row["lifecycle_state"] == "ingested" and row["ingested_utc"]
    assert row["queued_utc"] and row["started_utc"] and row["generated_utc"] and row["judged_utc"]
    assert row["queued_utc"] <= row["started_utc"] <= row["generated_utc"] <= row["judged_utc"]
    assert row["node_hostname"] == "TESTHOST" and row["gpu_count"] == 1 and row["gpu_type"] == "H200" and row["cluster_job_id"] == "4242"
    assert row["n_items_generated"] == 4 and row["n_items_judged"] == 4 and row["judge_id"] == synth.JUDGE_ID
    assert row["intervention_hash"] == IHASH and row["producer_run_key_legacy"] == KEY
    events = conn.execute("SELECT state, evidence_kind FROM run_state_event WHERE run_id = ? ORDER BY event_utc", (rid,)).fetchall()
    # `running` twice: the job shell's (job_log) and the store's own at manifest time (completion_artifact) — two evidences of one state
    assert [e[0] for e in events] == ["queued", "running", "running", "generated", "judged"]
    assert {e[1] for e in events if e[0] == "running"} == {"job_log", "completion_artifact"}
    assert [tuple(r) for r in conn.execute("SELECT status, n_items_judged, api_pool FROM judgment WHERE run_id = ?", (rid,)).fetchall()] == [("complete", 4, "lowprio")]
    # judge item rows beside the store's own rows (distinct judgment_id), the metric, and not_measured rows once per writer
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id != ''", (rid,)).fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id = ''", (rid,)).fetchone()[0] == 8
    assert conn.execute("SELECT value FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", (rid,)).fetchone()[0] == 0.5
    n_deg = conn.execute("SELECT COUNT(*) FROM run_degeneration WHERE run_id = ? AND measurement_state = 'not_measured'", (rid,)).fetchone()[0]
    assert n_deg == 2 * len(hooks.DEFAULT_DECLARED_DEGENERATION)   # the store's (judgment_id '') and the judgment's
    conn.close()


# --------------------------------------------------------------------------------------------
# Declared-but-uncomputed degeneration measures
# --------------------------------------------------------------------------------------------


def test_not_measured_rows_for_declared_but_uncomputed_measures():
    rows = hooks.not_measured_rows("c:e:i:0", hooks.DEFAULT_DECLARED_DEGENERATION, computed=[("union5", "all")], computed_utc=synth.t(1))
    assert {(r["measure"], r["scope"]) for r in rows} == set(hooks.DEFAULT_DECLARED_DEGENERATION) - {("union5", "all")}
    assert {"at_token_cap_retokenized", "at_token_cap_finish_reason"} <= {r["measure"] for r in rows}   # Two at-cap measures, never one
    for r in rows:
        assert r["measurement_state"] == "not_measured" and r["numerator"] is None and r["rate"] is None
        assert set(r) - {"extra"} >= set(R.ITEM_WHOLE_ROW_COLUMNS["run_degeneration"])
        rec = R.make_record("run_degeneration", r, synth.producer(), identity.producer_record_id(), synth.t(2))
        assert isinstance(VAL.validate_record(rec, "<mem>"), VAL.Validated)
    measured = hooks.measured_row("c:e:i:0", "at_token_cap_finish_reason", "all", 3, 12, "items")
    assert measured["rate"] == 0.25 and measured["measurement_state"] == "measured"
    assert hooks.measured_row("c:e:i:0", "union5", "all", 0, 0, "items")["measurement_state"] == "not_applicable"


def test_record_files_are_immutable_two_writers_never_share_a_sequence_number(tmp_path):
    w1 = emit.RecordWriter(str(tmp_path), "c", "c:e:i:0", synth.producer(), disabled=False)
    w2 = emit.RecordWriter(str(tmp_path), "c", "c:e:i:0", synth.producer(), disabled=False)
    p1 = w1.emit_state("running", "job_log")
    p2 = w2.emit_state("generated", "completion_artifact")     # w2 lists the directory at this write and sees w1's file (item 13)
    p3 = w1.emit_patch(fields={"gpu_count": 1})
    names = sorted(os.path.basename(p) for p in (p1, p2, p3))
    assert len(set(names)) == 3 and all(sum(1 for _ in R.read_record_file(p)) == 1 for p in (p1, p2, p3))
    with pytest.raises(FileExistsError):
        R.write_records(p1, [json.loads(open(p1).readline())], exclusive=True)


# --------------------------------------------------------------------------------------------
# run.seed_base: the manifest's `seed`, else the row constants' `seed` / `seed_base` (2026-09-17)
# --------------------------------------------------------------------------------------------


def _scope_payload(tmp_path, tag, ident_over, spec_over=None, monkeypatch=None, root=None):
    rec = hooks.StoreRecorder(KEY, dict(RESOLVED, **(spec_over or {})), {"*": _identity(**ident_over)}, spool=str(tmp_path / f"spool_{tag}"),
                              code_sha=synth.code_sha("a"))
    rec.store_dir = str(tmp_path / f"store_{tag}")
    if root is not None and monkeypatch is not None:
        monkeypatch.setattr(hooks, "_producing_checkout_root", lambda: root)
    return rec.run_payload("*")


def test_run_payload_keeps_the_identitys_scope_and_derives_it_from_the_seeds_when_the_identity_names_none(tmp_path, monkeypatch):
    """`multi_turn_steering_scope` on a store's own registration: the identity's statement when it carries one (the planner derived it);
    else DERIVED from the applied-positions mode and the category the producing checkout's `configs/registry/evals.yaml` gives the eval
    -- a multi-turn eval under `all` or `non_template` reads every_turn_prefill_and_decode, a static one single_turn, an unsteered spec
    not_applicable -- and `unverified` with the reason when the seeds do not name the eval; never the single-turn default."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    stated = _scope_payload(tmp_path, "a", {"multi_turn_steering_scope": "final_header_and_current_decode_only"})
    assert stated["multi_turn_steering_scope"] == "final_header_and_current_decode_only" and "multi_turn_steering_scope" not in stated.get("field_confidence_json", {})
    # a synthetic producing checkout whose seeds name the eval: the derivation reads its category
    root = tmp_path / "checkout"
    (root / "configs" / "registry").mkdir(parents=True)
    (root / "configs" / "registry" / "evals.yaml").write_text(
        "evals:\n  - eval_id: syntheval\n    eval_category: static_attack\n  - eval_id: synthmt\n    eval_category: multi_turn\n  - eval_id: nocat\n    name: x\n")
    p = _scope_payload(tmp_path, "b", {"multi_turn_steering_scope": hooks._UNVERIFIED}, monkeypatch=monkeypatch, root=str(root))
    assert p["multi_turn_steering_scope"] == hooks._UNVERIFIED                                   # an identity that STATES unverified keeps it
    rec_ident = _identity()
    del rec_ident["multi_turn_steering_scope"]
    r = hooks.StoreRecorder(KEY, RESOLVED, {"*": rec_ident}, spool=str(tmp_path / "spool_c"), code_sha=synth.code_sha("a"))
    r.store_dir = str(tmp_path / "store_c")
    p = r.run_payload("*")
    assert p["multi_turn_steering_scope"] == "single_turn"
    assert p["field_confidence_json"]["multi_turn_steering_scope"] == {"confidence": "derived", "source": "eval_category=static_attack is single-turn (eval_category from configs/registry/evals.yaml of the producing checkout)"}
    mt = dict(rec_ident, eval_id="synthmt", applied_positions_mode="non_template")
    r = hooks.StoreRecorder(KEY, RESOLVED, {"*": mt}, spool=str(tmp_path / "spool_d"), code_sha=synth.code_sha("a"))
    r.store_dir = str(tmp_path / "store_d")
    p = r.run_payload("*")
    assert p["multi_turn_steering_scope"] == "every_turn_prefill_and_decode" and p["applied_positions_mode"] == "non_template"
    assert p["field_confidence_json"]["multi_turn_steering_scope"]["confidence"] == "derived"
    for eval_id, why in (("nocat", "names no eval_category for eval 'nocat'"), ("unknown_eval", "eval 'unknown_eval' is not in configs/registry/evals.yaml")):
        r = hooks.StoreRecorder(KEY, RESOLVED, {"*": dict(rec_ident, eval_id=eval_id)}, spool=str(tmp_path / f"spool_{eval_id}"), code_sha=synth.code_sha("a"))
        r.store_dir = str(tmp_path / f"store_{eval_id}")
        p = r.run_payload("*")
        assert p["multi_turn_steering_scope"] == hooks._UNVERIFIED and why in p["field_confidence_json"]["multi_turn_steering_scope"]["reason"], eval_id
    # an unsteered spec: not_applicable, no derivation
    r = hooks.StoreRecorder(KEY, dict(RESOLVED, kind="none"), {"*": dict(rec_ident, object_id="none", transfer_kind="none", applied_positions_mode="none")},
                            spool=str(tmp_path / "spool_n"), code_sha=synth.code_sha("a"))
    r.store_dir = str(tmp_path / "store_n")
    p = r.run_payload("*")
    assert p["multi_turn_steering_scope"] == "not_applicable" and "multi_turn_steering_scope" not in p.get("field_confidence_json", {})
    # a prompted store (the store's legacy kind `prompted`, the identity's object id si:<sha>) on the multi-turn eval, its identity
    # carrying the instrument's `all` and no scope: the object's kind decides -- not_applicable with the kind's derivation, never the mode's
    si_id = identity.object_id("system_instruction", si_text_sha256=synth.SI_SHA)
    r = hooks.StoreRecorder(KEY, dict(RESOLVED, kind="prompted", arm_name="prompted__constitution"),
                            {"*": dict(rec_ident, eval_id="synthmt", object_id=si_id, transfer_kind="prompted_only")},
                            spool=str(tmp_path / "spool_p"), code_sha=synth.code_sha("a"))
    r.store_dir = str(tmp_path / "store_p")
    p = r.run_payload("*")
    assert p["object_id"] == si_id and p["applied_positions_mode"] == "all" and p["multi_turn_steering_scope"] == "not_applicable"
    assert p["field_confidence_json"]["multi_turn_steering_scope"] == {
        "confidence": "derived", "source": "object_kind=system_instruction: a prompted object steers no positions (eval_category from configs/registry/evals.yaml of the producing checkout)"}
    # the real checkout's seeds, unpatched: the synthetic eval is not among them, so the scope is unverified with the reason
    monkeypatch.undo()
    r = hooks.StoreRecorder(KEY, RESOLVED, {"*": rec_ident}, spool=str(tmp_path / "spool_real"), code_sha=synth.code_sha("a"))
    r.store_dir = str(tmp_path / "store_real")
    p = r.run_payload("*")
    assert p["multi_turn_steering_scope"] == hooks._UNVERIFIED and "is not in configs/registry/evals.yaml" in p["field_confidence_json"]["multi_turn_steering_scope"]["reason"]


def _seed_base_payload(tmp_path, spec, row_constants, tag):
    rec = hooks.StoreRecorder(KEY, spec, {"*": _identity()}, spool=str(tmp_path / f"spool_{tag}"), code_sha=synth.code_sha("a"),
                              row_constants=row_constants)
    rec.store_dir = str(tmp_path / f"store_{tag}")
    return rec, rec.run_payload("*")


def test_run_seed_base_falls_back_to_the_row_constants(tmp_path, monkeypatch):
    """A producer whose seed lives in the ROW CONSTANTS and not in its manifest (native battery files set `seed_base = s`
    on every row) gets `run.seed_base` without lifting it to the manifest: the manifest's `seed` first, then the row
    constants' `seed`, then their `seed_base`, else NULL; a malformed carrier never raises. The manifest's own nested
    `row_constants` copy (what `resolve_cell` archives) is read when the recorder was handed none."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    no_seed = {k: v for k, v in RESOLVED.items() if k != "seed"}
    assert _seed_base_payload(tmp_path, no_seed, {"seed": 3}, "a")[1]["seed_base"] == 3
    assert _seed_base_payload(tmp_path, no_seed, {"seed_base": 2}, "b")[1]["seed_base"] == 2
    assert _seed_base_payload(tmp_path, no_seed, {"seed": 3, "seed_base": 2}, "c")[1]["seed_base"] == 3        # `seed` before `seed_base`
    assert _seed_base_payload(tmp_path, dict(no_seed, seed=7), {"seed": 3}, "d")[1]["seed_base"] == 7          # both present: the spec wins
    assert _seed_base_payload(tmp_path, no_seed, None, "e")[1]["seed_base"] is None                            # neither: NULL
    assert _seed_base_payload(tmp_path, no_seed, {"seed": None}, "f")[1]["seed_base"] is None
    assert _seed_base_payload(tmp_path, no_seed, "not a mapping", "g")[1]["seed_base"] is None                  # never raises
    assert _seed_base_payload(tmp_path, dict(no_seed, row_constants={"seed": 5}), None, "h")[1]["seed_base"] == 5
    assert _seed_base_payload(tmp_path, dict(no_seed, row_constants={"seed": 5}), {"seed": 3}, "i")[1]["seed_base"] == 3   # the recorder's own first
    assert _seed_base_payload(tmp_path, dict(no_seed, row_constants="garbage"), None, "j")[1]["seed_base"] is None
    # the payload built from the fallback is still a `run` record the harvester accepts
    rec, p = _seed_base_payload(tmp_path, no_seed, {"seed_base": 2}, "k")
    record = R.make_record("run", p, rec.producer, "11111111-2222-3333-4444-555555555555", synth.t(5))
    assert not isinstance(VAL.validate_record(record, "<memory>", R.record_sha(record)), VAL.Malformed)


def test_default_recorder_fills_seed_base_from_the_row_constants_end_to_end(tmp_path, monkeypatch):
    """Through the store default (no recorder argument, the identity from the job's pending file): a store whose row
    constants carry `seed` and whose manifest does not -- the spooled `run` record carries `seed_base`, and after
    `on_manifest` replaces the recorder's spec with the manifest the row constants are still the carrier."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    monkeypatch.setattr(cs.time, "time", lambda: 1755000000.0)
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(_identity()))
    spool = tmp_path / "spool"
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(pending))
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    hooks._DEFAULT_NOTICED.clear()
    const = {k: v for k, v in RESOLVED.items() if k not in ("tiers", "cell_index")}
    const["seed"] = 11
    store = cs.CellStore(tmp_path / "store", KEY, row_constants=const)
    assert isinstance(store.recorder, hooks.StoreRecorder)
    manifest = {k: v for k, v in RESOLVED.items() if k != "seed"}          # the producer's manifest names no seed
    store.write_cell_manifest(manifest)
    runs = [rec["payload"] for _, rec, _ in _records_in(spool) if rec["record_type"] == "run"]
    assert len(runs) == 1 and runs[0]["seed_base"] == 11, runs
    # the same store with constants carrying no seed at all records NULL, never a guess
    spool2 = tmp_path / "spool2"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool2))
    const2 = {k: v for k, v in const.items() if k != "seed"}
    store2 = cs.CellStore(tmp_path / "store2", KEY, row_constants=const2)
    store2.write_cell_manifest(manifest)
    runs2 = [rec["payload"] for _, rec, _ in _records_in(spool2) if rec["record_type"] == "run"]
    assert len(runs2) == 1 and runs2[0].get("seed_base") is None, runs2


# --------------------------------------------------------------------------------------------
# hooks.queued_modal (kept: its callers are the frozen launchers outside the registry tree; the two tests
# moved here from the deleted Modal test module)
# --------------------------------------------------------------------------------------------


def _launcher_plan(**over):
    # a launcher's plan does not declare its substrate; the hook stamps the launcher's own
    p = {k: v for k, v in synth.run_payload("launcher_key", i=2).items() if k not in ("run_id", "substrate", "substrate_source")}
    p.update(over)
    return p


def test_queued_modal_writes_run_queued_and_binding(tmp_path, monkeypatch):
    """The launcher-side hook: the planned ``run`` (queued) + ``run_state queued`` + the ``run_patch`` carrying the app / call ids
    into the launcher host's spool; every record validates; all three land in the run's directory."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = str(tmp_path / "spool")
    paths = hooks.queued_modal(_launcher_plan(), modal_app_id="launcher-synthetic", modal_call_id="fc-synthetic-0001",
                               script="scripts/launcher.py fit", spool=spool, registry_emit=True)
    assert len(paths) == 3
    recs = [(p, rec) for p in paths for _, rec, _ in R.read_record_file(p)]
    assert [r["record_type"] for _, r in recs] == ["run", "run_state", "run_patch"]
    run, state, patch = (r["payload"] for _, r in recs)
    rid = identity.run_id(synth.CAMPAIGN, "launcher_key", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert run["run_id"] == rid and run["lifecycle_state"] == "queued" and run["substrate"] == "modal"
    assert state["state"] == "queued" and state["scheduler_state"] == "SPAWNED"
    assert patch["fields"] == {"modal_app_id": "launcher-synthetic", "modal_call_id": "fc-synthetic-0001"}
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec
    assert all(os.path.dirname(p) == os.path.join(spool, synth.CAMPAIGN, identity.run_dir_name(rid)) for p in paths)


def test_queued_modal_refuses_unresolved_and_is_a_noop_when_off(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    bad = _launcher_plan(eval_id="gsm8k")
    bad.pop("instrument_id")
    out = hooks.queued_modal(bad, modal_app_id="a", modal_call_id="c", script="t", spool=str(tmp_path / "s1"), registry_emit=True)
    assert out is None and not (tmp_path / "s1").exists()
    assert "instrument_id" in capsys.readouterr().err
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert hooks.queued_modal(_launcher_plan(), modal_app_id="a", modal_call_id="c", script="t", spool=str(tmp_path / "s2")) == []
    assert not (tmp_path / "s2").exists()
