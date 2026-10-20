"""degap-fix-5: the gap-fill walker's identity is read from EVERY response-bearing shard, an undated
or legacy aggregate pends, and the export re-checks currency and identity at filing time.

  * static_grading / mt_degeneracy_union read every shard that carries responses.jsonl; a shard with replies and no served_target.json used to
    pass under the other shards' identity (wrong denominator, wrong at-cap) — now REFUSED by name; the attested shards must agree on cell,
    harness cell, virtual_model, object sha, intervention_hash and decode.cap (distinct serving jobs stay allowed: a two-group store);
  * a joined aggregate without written_at, without completion.gated for the cell, or without both counts is StaleAggregate -> PENDING with
    the reason (it used to pass as "attested: False" and the run read measured);
  * cmd_export re-reads v_run_current.is_current at filing time (a retired registration is excluded and named), compares each cell's
    identity to its manifest row, and refuses rows measured before this fix (no per-shard attestation record) — nothing from an older
    diagnostic pass can be exported without a re-measure.
Synthetic stores and ids only; a whitespace tokenizer; the real union classifier, store loaders and registry harvester."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))
SRC = REPO / "scripts/sdfbase/full22/degeneracy_gap_fill.py"
UNION_SRC = REPO / "src"
from test_degeneracy_gap_fill_identity import (  # noqa: E402  (the three real-shaped synthetic layouts + the fixture world)
    OBJ, TEXT, _Tok, _aggregate, _mt_run, _mt_store, _static_run, _static_store, _touch_past)


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("degeneracy_gap_fill_quarantine_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def world(m, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(UNION_SRC))
    du, sg, stamp = m.load_union(UNION_SRC)
    m._W.clear()
    m._W.update({"du": du, "sg": sg, "mtu": m.load_mt_runner(REPO), "models": {}, "host": "grading_host", "stamp": stamp, "pre": {},
                 "xstest_safe": None, "xstest_safe_prov": {}})
    monkeypatch.setattr(m, "hf_model_for", lambda model_id, models: ("stub/tokenizer", None))
    monkeypatch.setattr(m, "tokenizer_for", lambda hf, rev: _Tok())
    monkeypatch.setattr(m, "tokenizer_snapshot", lambda hf, rev: {"stub": True})
    repro = tmp_path / "reproduce-state"
    repro.mkdir()
    monkeypatch.setattr(m, "REPRODUCE_STATE", repro)
    return m


# ------------------------------------------------------------------------------------------------------------------ (1) every shard attested

def test_a_response_bearing_shard_without_served_target_refuses_the_whole_store_by_name(world, tmp_path):
    m = world
    store = _static_store(tmp_path / "exp")                                     # two attested shards, measured
    c = m.one_run(_static_run(store))
    assert c["status"] == "measured" and c["store"]["identity"]["transcripts_results:store"]["response_shards"] == ["shard00", "shard01"]
    assert c["store"]["identity"]["transcripts_results:store"]["shards_with_served_target"] == ["shard00", "shard01"]
    (store / "shard01" / "served_target.json").unlink()                        # shard01 keeps its replies, loses its identity
    _touch_past(store, "2026-10-03T10:00:00Z")
    c = m.one_run(_static_run(store))
    assert c["status"] == "refused", c.get("note")
    assert "response-bearing shard(s) ['shard01'] carry no readable served_target.json" in c["note"] and "unattested" in c["note"] and "attested: ['shard00']" in c["note"]
    # the same for a multi-turn store (the mt loader reads every shard too)
    mt = _mt_store(tmp_path / "exp2")
    _aggregate(m.REPRODUCE_STATE)
    (mt / "shard01" / "served_target.json").unlink(); _touch_past(mt, "2026-10-03T10:00:00Z")
    c = m.one_run(_mt_run(mt))
    assert c["status"] == "refused" and "['shard01'] carry no readable served_target.json" in c["note"]


def test_attested_shards_must_agree_on_identity_but_may_differ_in_serving_job(world, tmp_path):
    m = world
    store = _static_store(tmp_path / "a")
    st = json.loads((store / "shard01" / "served_target.json").read_text())
    st["object_artifact_sha256_by_cell"] = {k: "e54842ec26c17bfb0d27158d47cedbacdde369dc65f60e93c5ed6774f0a9d472" for k in st["object_artifact_sha256_by_cell"]}
    (store / "shard01" / "served_target.json").write_text(json.dumps(st)); _touch_past(store, "2026-10-03T10:00:00Z")
    c = m.one_run(_static_run(store))
    assert c["status"] == "refused" and "disagree on the object sha" in c["note"], c.get("note")
    store = _static_store(tmp_path / "b")
    st = json.loads((store / "shard01" / "served_target.json").read_text()); st["intervention_hash"] = "deadbeefdeadbeef"
    (store / "shard01" / "served_target.json").write_text(json.dumps(st)); _touch_past(store, "2026-10-03T10:00:00Z")
    c = m.one_run(_static_run(store))
    assert c["status"] == "refused" and "disagree on intervention_hash" in c["note"], c.get("note")
    store = _static_store(tmp_path / "c")
    st = json.loads((store / "shard01" / "served_target.json").read_text()); st["virtual_model"] = "petri__si_other"
    (store / "shard01" / "served_target.json").write_text(json.dumps(st)); _touch_past(store, "2026-10-03T10:00:00Z")
    c = m.one_run(_static_run(store))
    assert c["status"] == "refused" and "disagree on the served virtual_model" in c["note"], c.get("note")
    # two serving JOBS on one store (the g01234 / g5678 groups) remain a positive identity
    mt = _mt_store(tmp_path / "d", shards=1)
    sd = mt / "shard01"; sd.mkdir()
    for f in ("served_target.json", "attacker_engine.json", "progress.jsonl", "responses.jsonl", "target_timing.jsonl"):
        (sd / f).write_text((mt / "shard00" / f).read_text().replace("872043", "872050").replace("g0a", "g1a").replace("g0b", "g1b"))
    _touch_past(mt, "2026-10-03T10:00:00Z"); _aggregate(m.REPRODUCE_STATE)
    c = m.one_run(_mt_run(mt, declared_jobs=["872043", "872050"]))                                   # degap-fix-6: both groups declared on the run
    assert c["status"] == "measured", c.get("note")
    assert c["store"]["identity"]["store_uri:transcripts_results"]["checks"]["job"]["store"] == ["872043", "872050"]


# ------------------------------------------------------------------------------------------------------------------ (2) undated aggregates pend

def test_a_legacy_or_undated_aggregate_leaves_the_run_pending_never_measured(world, tmp_path):
    m = world
    store = _mt_store(tmp_path / "exp")
    agg = _aggregate(m.REPRODUCE_STATE)
    d = json.loads(agg.read_text())
    # (a) a legacy aggregate: the cell is named, no completion.gated at all
    d2 = json.loads(json.dumps(d)); d2["cells"]["constv2-mix-full__lora_r4_ce"].pop("completion"); agg.write_text(json.dumps(d2))
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "carries no completion.gated facts for cell 'constv2-mix-full__lora_r4_ce'" in c["note"] and "legacy aggregate" in c["note"], c.get("note")
    # (b) completion present but a count missing
    d3 = json.loads(json.dumps(d)); d3["cells"]["constv2-mix-full__lora_r4_ce"]["completion"]["gated"].pop("n_shards"); agg.write_text(json.dumps(d3))
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "incomplete completion facts" in c["note"] and "n_progress_rows / n_shards needed" in c["note"], c.get("note")
    # (c) no written_at: the aggregate cannot be dated at all
    d4 = json.loads(json.dumps(d)); d4.pop("written_at"); agg.write_text(json.dumps(d4))
    c = m.one_run(_mt_run(store))
    assert c["status"] == "pending" and "carries no written_at" in c["note"], c.get("note")
    # the intact aggregate measures
    agg.write_text(json.dumps(d))
    c = m.one_run(_mt_run(store))
    assert c["status"] == "measured" and c["store"]["identity"]["store_uri:transcripts_results"]["aggregate_join"]["attested"] is True


# ------------------------------------------------------------------------------------------------------------------ (3) the export re-checks

def _measured_cell(m, rid, store_uri, eval_id="pasttense", *, attested=True, with_shards=True, job="809290", aggregate_join=True):
    from test_degeneracy_gap_fill import _measured_cell as _mc
    c = _mc()
    c.update({"run_id": rid, "key": rid, "status": "measured", "host": "cluster_a", "eval_id": eval_id, "model_id": "q", "lane": None, "cell": None,
              "store_uri": store_uri, "judgment_ids": ["j"], "instrument_id": "i", "measured_at": "2026-10-03T20:00:00Z", "denominator_unit": "draws",
              "tokenizer": {"model_id": "stub", "revision": None}, "reader": "static_grading.load_store_rows", "pooling": {"cap": 16384},
              "store": {"paths": [store_uri], "shards": {"shard00": "a"}, "identity": {"transcripts_results:store": {
                  "attested": attested, "attested_by": ["job"], "checks": {"job": {"registry": job, "store": [job]}},
                  **({"response_shards": ["shard00"], "shards_with_served_target": ["shard00"]} if with_shards else {}),
                  **({"aggregate_join": {"attested": aggregate_join}} if eval_id in m.MT else {})}}}})
    return c


def test_export_files_only_current_identity_ready_cells_and_names_every_exclusion(m, tmp_path):
    from sisteer.registry import db as DB, harvest as H, identity, records as R, synth
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid_ok, recs = synth.run_records("k_ok", 100, through="judged")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid_ok), "000001-life.jsonl"), recs)
    rid_gone, recs2 = synth.run_records("k_gone", 120, through="judged")           # retired AFTER the manifest was built: a superseded twin
    recs2.append(synth.rec("run_state", {"run_id": rid_gone, "state": "superseded", "event_utc": synth.t(130), "evidence_kind": "completion_artifact",
                                         "evidence_uri": "x/progress.jsonl", "note": "twin"}, synth.t(130)))
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid_gone), "000001-life.jsonl"), recs2)
    rid_old, recs3 = synth.run_records("k_old", 140, through="judged")             # current, but measured by a pre-fix-5 tool (no per-shard record)
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid_old), "000001-life.jsonl"), recs3)
    rid_mis, recs4 = synth.run_records("k_mis", 160, through="judged")             # current, but the cell's store_uri is not the manifest's
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid_mis), "000001-life.jsonl"), recs4)
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep.as_dict()
    db = Path(DB.db_path(root))
    cur = m.current_runs(db, [rid_ok, rid_gone, rid_old, rid_mis])
    assert cur[rid_ok][0] == 1 and cur[rid_gone][0] == 0 and "superseded" in str(cur[rid_gone][1]) and cur[rid_old][0] == 1 and cur[rid_mis][0] == 1
    out = tmp_path / "out"; out.mkdir()
    cells = {rid_ok: _measured_cell(m, rid_ok, "synthetic:///store/k_ok", job="4242"), rid_gone: _measured_cell(m, rid_gone, "synthetic:///store/k_gone", job="4242"),
             rid_old: _measured_cell(m, rid_old, "synthetic:///store/k_old", with_shards=False, job="4242"), rid_mis: _measured_cell(m, rid_mis, "/s/mis-measured", job="4242")}
    for c in cells.values():
        c["instrument_id"] = synth.INSTRUMENT_ID; c["trial_index"] = 0                                   # degap-fix-6: the measurement records the registry's instrument + trial
    (out / "degeneracy_gap_fill.json").write_text(json.dumps({"definition_version": f"{m.SCHEMA}@deadbeef", "tool_sha256s": ["t"], "meta": {"runs": []}, "cells": cells}))
    manifest = tmp_path / "manifest.jsonl"
    # degap-fix-6: the manifest row IS the registry row's identity (manifest_from_registry copies it), so these rows state the synthetic registry's —
    # store_uri synthetic:///store/<key>, the job 4242 its run_patch filed, synth's instrument, cap 512, decode greedy; the export's ONE snapshot compares them
    rows = [{"run_id": r, "model_id": "q", "eval_id": "pasttense", "instrument_id": synth.INSTRUMENT_ID, "store_uri": f"synthetic:///store/k_{tag}", "cell": None,
             "cluster_job_id": "4242", "cap": 512, "reply_cap": None, "trial_index": 0, "pass_label": None, "decode_source": "greedy",
             "producer_object_key": "none", "judgment_ids": ["j"], "metric_rows": [{}]}
            for r, tag in ((rid_ok, "ok"), (rid_gone, "gone"), (rid_old, "old"), (rid_mis, "mis"))]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    class A: pass
    a = A(); a.out = str(out); a.manifest = str(manifest); a.batch = "b1"; a.dir = str(tmp_path / "by_run_id"); a.hosts = ""; a.filed = ""; a.registry = str(db)
    assert m.cmd_export(a) == 0
    body = json.loads(next((tmp_path / "by_run_id").glob("b1.v*.json")).read_text())
    assert [r["run_id"] for r in body["rows"]] == [rid_ok]                                                # the one current, identity-ready cell
    row = body["rows"][0]                                                                                  # the front door's NULL cell key stays NULL (the reader
    assert row["campaign_cell_id_legacy"] is None and row["provenance"]["producer_object_key"] == "none"  # compares NULL == NULL); the cell key rides in provenance
    assert row["provenance"]["cluster_job_id"] == "4242" and row["provenance"]["export_tool"] == "degap-fix-6" and row["provenance"]["identity_attestation"] and body["registry_snapshot_utc"]
    assert body["self_check"]["status"] == "passed"
    why = {x["run_id"]: x["why"] for x in body["excluded_not_export_ready"]}
    assert set(why) == {rid_gone, rid_old, rid_mis}
    assert "not current in the registry at filing time" in why[rid_gone] and "superseded" in why[rid_gone]
    assert "no per-shard attestation record (measured before degap-fix-5)" in why[rid_old]
    assert "store_uri '/s/mis-measured' != manifest 'synthetic:///store/k_mis'" in why[rid_mis] and "registry store_uri='synthetic:///store/k_mis' != measurement '/s/mis-measured'" in why[rid_mis]   # degap-fix-6: the snapshot names it too
    assert body["excluded_already_filed"] == [] and body["registry"] == str(db) and body["currency_checked_utc"]
    # a cell whose attested job is not the manifest's registry job, or an undated multi-turn join, is not export-ready either
    assert any("attested registry job 809291 != manifest cluster_job_id 4242" in p for p in m.export_identity_problems(_measured_cell(m, rid_ok, "synthetic:///store/k_ok", job="809291"), rows[0]))
    mt_row = {**rows[0], "eval_id": "crescendo"}
    assert any("aggregate join not dated" in p for p in m.export_identity_problems(_measured_cell(m, rid_ok, "synthetic:///store/k_ok", eval_id="crescendo", aggregate_join=False, job="4242"), mt_row))
    assert m.export_identity_problems(_measured_cell(m, rid_ok, "synthetic:///store/k_ok", eval_id="crescendo", job="4242"), mt_row) == []
    # the inherited self-check refused EVERY front-door row ("identity field campaign_cell_id_legacy empty"); a NULL cell without the key refuses still
    chk = m.self_check_rows([{**body["rows"][0], "provenance": {}}], "deadbeef")
    assert chk["status"] == "failed" and "no producer_object_key in provenance" in chk["problems"][0]
