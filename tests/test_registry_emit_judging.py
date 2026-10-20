"""Native emission from the judging drivers: the shared ``judgment_scope`` helper
(claim → judge item rows → metrics → complete + ``run_state judged`` + the completion patch +
``not_measured`` rows), a failure inside the scope filing ``partial``, and the served-capability judge
driver's own hook (``_registry_record_store``) over synthetic ``judge_items.jsonl`` /
``judge_summary.json`` — every driver output written first and byte-identical with the hook on or off.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, hooks, identity
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

REPO = Path(__file__).resolve().parents[1]


def _records(spool):
    out = []
    for p in R.iter_record_files(str(spool)):
        for _, rec, sha in R.read_record_file(p):
            out.append((p, rec, sha))
    return out


def _judged_root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("k_judge", 100, through="generated")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    return root, rid


def test_judgment_scope_claims_files_and_completes(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    root, rid = _judged_root(tmp_path)
    spool = os.path.join(root, "records")
    judge = {"judge_model": "judge-model-x", "rubric_sha256": synth.SHA256_A, "protocol_sha256": synth.SHA256_B, "threshold_rule": "score > 0.5"}
    scope = hooks.judgment_scope("scripts/synthetic_judge.py", {"run_id": rid, "campaign": synth.CAMPAIGN}, judge, api_pool="lowprio",
                                 spool=spool, registry_emit=True, session="synthetic-session", store_uri="synthetic:///store/k_judge")
    assert scope is not None and scope.judge_id == synth.JUDGE_ID
    with scope:
        scope.items([{"item_id": f"q{i:05d}", "metric_id": "asr", "value": float(i % 3 == 0), "label": "harmful" if i % 3 == 0 else "safe"}
                     for i in range(30)])
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 10 / 30, "k": 10, "n": 30, "ci_lo": 0.19, "ci_hi": 0.51, "ci_method": "wilson95"}])
        scope.degeneration([hooks.measured_row(rid, "at_token_cap_finish_reason", "all", 3, 30, "items")])
    recs = _records(Path(spool) / synth.CAMPAIGN / identity.run_dir_name(rid))
    types_ = [r["record_type"] for _, r, _ in recs if r["producer"]["script"] == "scripts/synthetic_judge.py"]
    assert types_[0] == "judgment" and types_.count("judgment") == 2 and "run_state" in types_ and "run_patch" in types_
    for p, rec, sha in recs:
        assert isinstance(VAL.validate_record(rec, str(p), sha), VAL.Validated), (p, rec)
    j = [r["payload"] for _, r, _ in recs if r["record_type"] == "judgment"]
    assert j[0]["status"] == "in_progress" and j[0]["claimed_by_session"] == "synthetic-session" and j[0]["api_pool"] == "lowprio"
    assert j[1]["status"] == "complete" and j[1]["n_items_judged"] == 30 and j[1]["judgment_id"] == j[0]["judgment_id"]
    deg = [r["payload"] for _, r, _ in recs if r["record_type"] == "run_degeneration"]
    # the one measure computed is `measured`; every other declared measure has a not_measured row
    assert {(d["measure"], d["measurement_state"]) for d in deg} == {("at_token_cap_finish_reason", "measured"), ("union5", "not_measured"),
                                                                     ("at_token_cap_retokenized", "not_measured"), ("loop_first512", "not_measured")}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, n_items_judged, judge_id FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert row[0] in ("judged", "ingested") and row[1] == 30 and row[2] == synth.JUDGE_ID
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id = ?", (rid, j[0]["judgment_id"])).fetchone()[0] == 30
    assert conn.execute("SELECT rate FROM run_degeneration WHERE run_id = ? AND measure = 'at_token_cap_finish_reason'", (rid,)).fetchone()[0] == pytest.approx(0.1)
    conn.close()


def test_judgment_scope_files_partial_on_failure_and_is_a_noop_when_off(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = str(tmp_path / "spool")
    rid = synth.run_payload("k")["run_id"]
    scope = hooks.judgment_scope("d", {"run_id": rid}, {"judge_id": synth.JUDGE_ID}, spool=spool, registry_emit=True)
    with pytest.raises(RuntimeError):
        with scope:
            scope.items([{"item_id": "q1", "metric_id": "asr", "value": 1.0, "label": "harmful"}])
            raise RuntimeError("judge died")
    j = [r["payload"] for _, r, _ in _records(spool) if r["record_type"] == "judgment"]
    assert [x["status"] for x in j] == ["in_progress", "partial"] and j[1]["n_errors"] == 1
    # the campaign came from the run id (no explicit campaign)
    assert scope.campaign == synth.CAMPAIGN
    # off: None, nothing written
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert hooks.judgment_scope("d", {"run_id": rid}, {"judge_id": synth.JUDGE_ID}, spool=str(tmp_path / "off")) is None
    assert not (tmp_path / "off").exists()


# --------------------------------------------------------------------------------------------
# the served-capability judge driver's hook over its own output files
# --------------------------------------------------------------------------------------------


def _load_driver():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    return __import__("served_capability_judge")


def _items(control, n=6):
    rows = []
    for i in range(n):
        status = "judged" if i % 3 else "abstained"
        rec = {"control": control, "task": f"{control}_task", "item_id": f"{control}-{i}", "request_key": f"k{i}", "judge_model": "judge-model-x",
               "rubric_sha": synth.SHA256_A, "protocol_sha": synth.SHA256_B, "cache_org": "lowprio", "verdict_created_at": 1.0,
               "cached_before_run": False, "status": status}
        if status == "judged":
            rec.update(label="full_refusal" if i % 2 else "compliance", label_source="store")
        else:
            rec.update(abstention_reason="refusal")
        rows.append(rec)
    return rows


def _summary(cell_dir, controls):
    return {"target": "synthtarget", "cell": cell_dir.name, "cell_id": "constitution_lora", "intervention_hash": "a1b2c3d4e5f60718",
            "model_id": synth.MODEL_ID, "controls": {
                c: {"status": "ok", "task": f"{c}_task", "n": 6, "at_cap_share": 0.5,
                    "over_refusal": {"point_estimate_over_judged": 0.5, "wilson95_over_judged": [0.15, 0.85], "n_judged": 4, "n": 6,
                                     "n_abstained": 2, "abstention_rate": 2 / 6, "abstention_band_full_n": [0.33, 0.67], "full_refusal_n": 2}}
                for c in controls}}


def test_driver_hook_writes_judgment_items_and_metric_and_files_are_byte_identical(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    drv = _load_driver()
    root, rid = _judged_root(tmp_path)
    spool = os.path.join(root, "records")
    monkeypatch.setenv(emit.SPOOL_ENV, spool)
    cell_dir = tmp_path / "controls" / "synthtarget" / "cells" / "constitution_lora__xstest__t0__a0__a1b2c3d4__0f1e2d3c"
    cell_dir.mkdir(parents=True)
    cell = types.SimpleNamespace(target="synthtarget", name=cell_dir.name, cell_dir=cell_dir, judge_model="judge-model-x")
    controls = ["xstest", "or_bench_hard_1k"]
    summary, items = _summary(cell_dir, controls), _items("xstest") + _items("or_bench_hard_1k")
    args = types.SimpleNamespace(org="lowprio")

    def write_all():
        drv._write_json(cell_dir / drv.SUMMARY_NAME, summary)
        drv._write_jsonl(cell_dir / drv.ITEMS_NAME, items)
        return {n: hashlib.sha256((cell_dir / n).read_bytes()).hexdigest() for n in (drv.SUMMARY_NAME, drv.ITEMS_NAME)}

    # hook off (no --registry-runs): nothing emitted
    monkeypatch.setattr(drv, "REGISTRY_RUNS", None)
    before = write_all()
    drv._registry_record_store(cell, summary, items, args)
    assert not os.path.isdir(os.path.join(spool, synth.CAMPAIGN, identity.run_dir_name(rid))) or not [
        p for p in os.listdir(os.path.join(spool, synth.CAMPAIGN, identity.run_dir_name(rid))) if "judgment" in p]
    # hook on: one control mapped to the run, the other unmapped (not recorded)
    monkeypatch.setattr(drv, "REGISTRY_RUNS", {f"synthtarget/{cell_dir.name}": {"xstest": {"run_id": rid, "campaign": synth.CAMPAIGN}}})
    after = write_all()
    drv._registry_record_store(cell, summary, items, args)
    assert before == after   # the driver's files: the same bytes with the hook on
    # the judge the driver minted is a dimension row (the dimension seed's job for the real judges); without it the
    # judgment and the run's judge patch stay dependency-BLOCKED (never refused) until it arrives
    driver_judge = {"judge_model": "judge-model-x", "rubric_sha256": synth.SHA256_A, "protocol_sha256": synth.SHA256_B,
                    "threshold_rule": "label == full_refusal"}
    rep0 = H.ingest_once(root, workers=0, min_free=0)
    assert rep0.refused == 0 and rep0.blocked > 0, rep0.as_dict()
    R.write_records(os.path.join(root, "records", "dims", "000002-judge.jsonl"), [synth.rec("judge", {
        "judge_id": identity.judge_id(driver_judge), "judge_role": "judge", "rubric_name": "over_refusal", "rubric_version": "1",
        "score_scale": "label", "judge_backend": "live", **driver_judge}, synth.t(50))])
    recs = _records(Path(spool) / synth.CAMPAIGN / identity.run_dir_name(rid))
    mine = [(p, r, s) for p, r, s in recs if r["producer"]["script"] == "scripts/served_capability_judge.py"]
    assert [r["record_type"] for _, r, _ in mine][:1] == ["judgment"]
    for p, rec, sha in mine:
        assert isinstance(VAL.validate_record(rec, str(p), sha), VAL.Validated), (p, rec)
    items_out = [r["payload"] for _, r, _ in mine if r["record_type"] == "item_result"]
    assert len(items_out) == 6 and {i["metric_id"] for i in items_out} == {"over_refusal"}
    assert sorted(i["label"] for i in items_out) == sorted(["abstained:refusal", "abstained:refusal", "full_refusal", "full_refusal", "compliance", "compliance"])
    assert [i["value"] for i in items_out if i["label"] == "full_refusal"] == [1.0, 1.0] and all(i["api_pool"] == "lowprio" for i in items_out)
    metric = next(r["payload"] for _, r, _ in mine if r["record_type"] == "run_metric")
    assert metric["metric_id"] == "over_refusal" and metric["value"] == 0.5 and metric["k"] == 2 and metric["n"] == 4 and metric["ci_method"] == "wilson95"
    assert metric["n_abstentions"] == 2 and metric["frac_at_token_cap"] == 0.5
    judgments = [r["payload"] for _, r, _ in mine if r["record_type"] == "judgment"]
    assert [j["status"] for j in judgments] == ["in_progress", "complete"] and judgments[1]["n_items_judged"] == 4 and judgments[1]["n_judge_refusals"] == 2
    assert judgments[1]["judge_id"] == identity.judge_id({"judge_model": "judge-model-x", "rubric_sha256": synth.SHA256_A,
                                                          "protocol_sha256": synth.SHA256_B, "threshold_rule": "label == full_refusal"})
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.unblocked > 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT judge_id, n_items_judged FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == identity.judge_id(driver_judge)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()


def test_driver_hook_never_raises(tmp_path, monkeypatch, capsys):
    drv = _load_driver()
    monkeypatch.setattr(drv, "REGISTRY_RUNS", {"t/c": {"xstest": {"run_id": "not-a-run-id"}}})
    cell = types.SimpleNamespace(target="t", name="c", cell_dir=tmp_path, judge_model="j")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own tmp spool, so it opts in
    monkeypatch.setenv(hooks.EMIT_ENV, "1")
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "spool"))
    drv._registry_record_store(cell, {"controls": {"xstest": {"status": "ok", "over_refusal": {}}}}, [], types.SimpleNamespace(org="lowprio"))
    err = capsys.readouterr()
    assert "not recorded" in err.out + err.err


def test_driver_exposes_registry_runs_flag():
    s = (REPO / "scripts" / "served_capability_judge.py").read_text()
    assert '"--registry-runs"' in s and "_registry_record_store(cell, summary, items, args)" in s
    # the hook runs AFTER the final writes of both output files
    assert s.index("_write_jsonl(cell.cell_dir / ITEMS_NAME, items)\n    _registry_record_store") > 0
