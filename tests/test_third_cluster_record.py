"""Migration 0018: the generic `slurm` substrate for a third Slurm cluster.

* a `run`, an `object` (`fit_substrate`) and a `fit_search` from a synthetic third cluster (`cluster_c`) emit with `substrate = slurm`
  + `extra_json.cluster_label`, fold through the harvester and validate against the code's schema (the three CHECKs 0018 rewrites);
* `slurm` WITHOUT a cluster label is refused BY NAME at the writer (nothing written);
* an UNKNOWN substrate is refused by name at the writer -- never aliased to `grading_host`;
* the emitter's stamp lands `cluster_label` on every record a `slurm` writer emits (the run, its state, its items, a judgment).
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, emit_hpo, hooks, identity, synth
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V

LABEL = "cluster_c"


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return root


def _fold(root):
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return rep


def test_slurm_is_a_substrate_and_binds_the_three_checks():
    assert "slurm" in V.SUBSTRATES and "slurm" in V.FIT_SEARCH_SUBSTRATES and "unverified" not in V.SUBSTRATES
    from sisteer.registry import schema_def as S
    for table, col in (("run", "substrate"), ("object", "fit_substrate"), ("fit_search", "substrate")):
        c = S.TABLE_BY_NAME[table].column(col)
        assert "slurm" in V.VOCABULARIES[c.vocab], (table, col, c.vocab)
    assert "slurm" in V.EVIDENCE["SUBSTRATES"] and "cluster_label" in V.EVIDENCE["SUBSTRATES"]["slurm"]


def test_third_cluster_run_object_and_search_emit_fold_and_validate(tmp_path, monkeypatch):
    root = _root(tmp_path)
    spool = os.path.join(root, "records", "c")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv(emit.CLUSTER_LABEL_ENV, LABEL)
    producer = emit.producer_block("tests.third_cluster", code_sha=synth.code_sha("a"), substrate="slurm")
    # -- a run registered from the third cluster (the run payload names slurm; the writer stamps the label into extra)
    plan = synth.run_payload("c3_key", substrate="slurm", substrate_source="recorded")
    rid = str(plan.pop("run_id"))
    plan["lifecycle_state"] = "running"
    plan["started_utc"] = plan["queued_utc"]
    ids = emit.check_planned_run(plan, producer)
    assert ids == rid
    w = emit.RecordWriter(spool, synth.CAMPAIGN, rid, producer)
    assert w.payload_extra[emit.CLUSTER_LABEL_KEY] == LABEL
    w.emit_run(dict(plan, run_id=rid))
    w.emit_state("running", "job_log", event_utc=plan["started_utc"])
    w.emit_items([{**{c: None for c in R.ITEM_WHOLE_ROW_COLUMNS["item_result"]}, "run_id": rid, "item_id": "q1", "metric_id": "generated",
                   "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": "", "created_utc": synth.t(1), "source_revision_id": ""}])
    # -- an object fit on the third cluster
    obj = synth.object_lora()
    obj["fit_substrate"] = "slurm"
    wo = emit.RecordWriter(spool, synth.CAMPAIGN, f"object__{obj['object_id'][:16]}", producer)
    wo.emit_object(obj)
    # -- a search run on the third cluster, through the native HPO emitter
    sid = emit_hpo.open_search(campaign=synth.CAMPAIGN, model_id=synth.FIT_MODEL_ID, object_kind="lora", method="sobol",
                               search_space={"lr": {"kind": "log_uniform", "low": 1e-5, "high": 1e-3}}, n_trials=2, substrate="slurm",
                               code_sha=synth.code_sha("a"), spool=spool, started_utc=synth.t(0))
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        run = conn.execute("SELECT substrate, substrate_source, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
        assert run[0] == "slurm" and run[1] == "recorded" and json.loads(run[2])["cluster_label"] == LABEL
        ev = conn.execute("SELECT extra_json FROM run_state_event WHERE run_id = ? AND state = 'running'", (rid,)).fetchone()
        assert json.loads(ev[0])["cluster_label"] == LABEL
        it = conn.execute("SELECT extra_json FROM item_result WHERE run_id = ?", (rid,)).fetchone()
        assert json.loads(it[0])["cluster_label"] == LABEL
        o = conn.execute("SELECT fit_substrate, extra_json FROM object WHERE object_id = ?", (obj["object_id"],)).fetchone()
        assert o[0] == "slurm" and json.loads(o[1])["cluster_label"] == LABEL
        fs = conn.execute("SELECT substrate, extra_json FROM fit_search WHERE search_id = ?", (sid,)).fetchone()
        assert fs[0] == "slurm" and json.loads(fs[1])["cluster_label"] == LABEL
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
        assert DB.schema_drift(conn) == []
    finally:
        conn.close()
    # every record's payload validates on its own (the CHECK lists are the vocabulary's)
    for path in R.iter_record_files(spool):
        for _i, rec, _sha in R.read_record_file(path, compute_sha=False):
            v = VAL.validate_record(rec, path)
            assert isinstance(v, VAL.Validated), (path, v)


def test_slurm_without_a_cluster_label_refuses_by_name_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.CLUSTER_LABEL_ENV, raising=False)
    spool = str(tmp_path / "spool")
    producer = emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="slurm")
    with pytest.raises(emit.SubstrateRefused) as ei:
        emit.RecordWriter(spool, synth.CAMPAIGN, "r", producer)
    assert "slurm" in str(ei.value) and emit.CLUSTER_LABEL_ENV in str(ei.value) and "cluster_label" in str(ei.value)
    assert not os.path.exists(spool)
    # an explicit payload_extra label satisfies it without the variable
    w = emit.RecordWriter(spool, synth.CAMPAIGN, "r", producer, payload_extra={"cluster_label": LABEL})
    assert w.payload_extra["cluster_label"] == LABEL
    # the hook path (a guarded writer) says one line and continues, never raising into the workload
    monkeypatch.setenv(emit.SPOOL_ENV, spool)
    monkeypatch.setenv(emit.SUBSTRATE_ENV, "slurm")
    out = hooks.emit_object_from_sidecar(str(tmp_path / "nowhere"), campaign="c", object_kind="lora", paper_label="other", object_role="control",
                                         fit_target_kind="corpus_document_fit", fit_objective="ce", script="t")
    assert out is None or out == []


def test_unknown_substrate_refuses_by_name_never_grading_host(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = str(tmp_path / "spool")
    with pytest.raises(emit.SubstrateRefused) as ei:
        emit.RecordWriter(spool, synth.CAMPAIGN, "r", emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="cluster_c"))
    msg = str(ei.value)
    assert "'cluster_c'" in msg and "|".join(V.SUBSTRATES) in msg and "never aliased to grading_host" in msg
    assert not os.path.exists(spool)
    # the job-start registration path names the substrate explicitly (`--substrate $CLUSTER`): the refusal reaches its guard
    from sisteer.registry import emit_cli
    pend = tmp_path / "pending"
    rc = emit_cli.register_from_config(str(tmp_path / "no.yaml"), model_key="qwen35_4b", evals=["pasttense"], serving_path="lora_native",
                                       campaign="c", job_id="1", pending_dir=str(pend), substrate="cluster_c", spool=spool, push=False)
    assert rc == 2 and not (pend / "1.json").exists()      # nothing registered; exit 2 — the shell stops the job on it
    # ... and past the plan (a config that builds): the substrate is refused ONCE, by name, before any writer is built -- the one
    # `[registry]` line the contract promises, never a SubstrateRefused traceback out of the first RecordWriter
    from sisteer.registry import jobplan as JP
    monkeypatch.setattr(JP, "build_plans", lambda **kw: JP.PlanBundle(runs=[object()]))
    capsys.readouterr()
    rc = emit_cli.register_from_config(str(tmp_path / "any.yaml"), model_key="qwen35_4b", evals=["pasttense"], serving_path="lora_native",
                                       campaign="c", job_id="2", pending_dir=str(pend), substrate="cluster_c", spool=spool, push=False)
    err = capsys.readouterr().err
    assert rc == 2 and not (pend / "2.json").exists() and not os.path.exists(spool)   # refused ONCE by name, exit 2
    assert "self-registration refused (SubstrateRefused" in err and "'cluster_c'" in err and "never aliased to grading_host" in err
    # a producer with NO substrate at all (a synthetic writer) is not refused: there is nothing to check
    w = emit.RecordWriter(spool, synth.CAMPAIGN, "r", {"script": "t", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST"})
    assert w.payload_extra == {}


def test_parse_store_root_map_admits_slurm_and_refuses_an_unknown_substrate():
    from sisteer.registry import pending as P
    m = P.parse_store_root_map(["slurm=/scratch/user/exp=/mirror/cluster_c"])
    assert m == {"slurm": [("/scratch/user/exp", "/mirror/cluster_c")]}
    with pytest.raises(ValueError) as ei:
        P.parse_store_root_map(["cluster_c=/a=/b"])
    assert "cluster_c" in str(ei.value)
