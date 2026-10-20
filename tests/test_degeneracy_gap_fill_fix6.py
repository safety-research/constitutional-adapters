"""The job rule:

* every response-bearing shard's served_target.json is a COMPLETE identity — a `{}` manifest beside a complete one, a legacy manifest without
  the job / object / cap, or any missing field refuses the whole store by name (the shard and the fields), before a reply is read;
* every serving job the shards name is a DECLARED job of the run: {cluster_job_id} ∪ {the producer key's @job} ∪ extra.serving_jobs
 ∪ the funnel ledger's attempt job ids for the unit, each verified terminal through sacct on its cluster;
  an undeclared job refuses by name with the remedy; a serving_jobs entry the ledger does not know is a cross-check problem;
* the export reads currency AND identity from ONE registry snapshot and compares both with the manifest and the measurement — a run_patch
  between measurement and export excludes the run by name.

Synthetic stores, ledgers and registries; sacct stubbed; the live registry is never read."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_degeneracy_gap_fill_identity import (  # noqa: E402,F401  (the fixtures `m` and `world` ride along)
    OBJ, _aggregate, _mt_run, _mt_store, _served_target, _static_run, _static_store, _touch_past, m, world)
from test_degeneracy_gap_fill import _measured_cell, _export_ready, _registry_current  # noqa: E402


# ------------------------------------------------------------------------------------------------- (1) complete per-shard identity

def test_an_empty_manifest_beside_a_complete_one_refuses_the_store_naming_the_shard(world, tmp_path):
    m = world
    store = _static_store(tmp_path / "exp")
    (store / "shard01" / "served_target.json").write_text("{}"); _touch_past(store, "2026-10-03T10:00:00Z")
    c = m.one_run(_static_run(store, declared_jobs=["809290"]))
    assert c["status"] == "refused" and "INCOMPLETE served_target.json" in c["note"] and "shard01: missing ['<empty manifest>']" in c["note"], c.get("note")


def test_a_legacy_manifest_missing_job_object_or_cap_refuses_naming_the_fields(world, tmp_path):
    m = world
    for drop, want in (("routing_table", "routing_table"), ("object_artifact_sha256_by_cell", "object_artifact_sha256_by_cell"), ("decode", "decode.cap")):
        store = _static_store(tmp_path / f"exp_{drop}")
        st = json.loads((store / "shard01" / "served_target.json").read_text()); st.pop(drop)
        (store / "shard01" / "served_target.json").write_text(json.dumps(st)); _touch_past(store, "2026-10-03T10:00:00Z")
        c = m.one_run(_static_run(store, declared_jobs=["809290"]))
        assert c["status"] == "refused" and "INCOMPLETE served_target.json" in c["note"] and f"shard01: missing [" in c["note"] and want in c["note"], (drop, c.get("note"))
    assert m.served_target_missing({}) == ["<empty manifest>"]
    full = _served_target(cell="constv2-mix-full__lora_r4_ce", job="809290", obj=OBJ, cap=16384, trial=0, ih="155f", written="2026-10-03T10:00:00Z", serving_dir="q")
    assert m.served_target_missing(full) == []
    none_cell = _served_target(cell="none", job="809290", obj=None, cap=16384, trial=0, ih=None, written="2026-10-03T10:00:00Z", serving_dir="q")
    assert m.served_target_missing(none_cell) == []                                             # a none / prompted cell serves no object: not required
    assert m.served_target_missing({**full, "routing_table": "/x/serving/nojob/static/routing_table.json"}) == ["routing_table:<serving job>"]


# ------------------------------------------------------------------------------------------------- (2) declared serving jobs

def _two_group_store(tmp_path, m):
    store = _mt_store(tmp_path / "exp", shards=1)
    sd = store / "shard01"; sd.mkdir()
    for f in ("served_target.json", "attacker_engine.json", "progress.jsonl", "responses.jsonl", "target_timing.jsonl"):
        (sd / f).write_text((store / "shard00" / f).read_text().replace("872043", "872050").replace("g0a", "g1a").replace("g0b", "g1b"))
    _touch_past(store, "2026-10-03T10:00:00Z"); _aggregate(m.REPRODUCE_STATE)
    return store


def test_an_undeclared_serving_job_refuses_by_name_with_the_remedy_and_a_declared_one_measures(world, tmp_path):
    m = world
    store = _two_group_store(tmp_path, m)
    c = m.one_run(_mt_run(store))                                                                # the registry knows job 872043 only
    assert c["status"] == "refused" and "serving job(s) ['872050'] (routing_table) are not DECLARED jobs of run" in c["note"], c.get("note")
    assert "declare it on the run (extra.serving_jobs, D1918) or in the funnel ledger (attempts.jsonl, D1915)" in c["note"]
    c = m.one_run(_mt_run(store, declared_jobs=["872043", "872050"], declared_jobs_unverified=[]))
    assert c["status"] == "measured" and c["store"]["identity"]["store_uri:transcripts_results"]["checks"]["declared_jobs"] == {"declared": ["872043", "872050"], "unverified": [], "store": ["872043", "872050"]}
    c = m.one_run(_mt_run(store, declared_jobs=["872043"], declared_jobs_unverified=["872050@cluster_a:unknown"]))
    assert c["status"] == "refused" and "sacct-unverified ledger job(s) ['872050@cluster_a:unknown']" in c["note"], c.get("note")
    c = m.one_run(_mt_run(store, declared_jobs=["872043", "872050"], declared_jobs_problem="extra.serving_jobs names 999, which neither the funnel ledger nor the registry's job for this run knows"))
    assert c["status"] == "refused" and "declared jobs are not trustworthy" in c["note"] and "999" in c["note"], c.get("note")


def _ledger(tmp_path, model="q", rows=()):
    root = tmp_path / "state"; (root / model).mkdir(parents=True)
    (root / model / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _attempt(lwid, job, cluster="cluster_a", kind="target", n=1):
    return {"event": "attempt", "logical_work_id": lwid, "attempt_id": f"{lwid}@{cluster}@{n}", "cluster": cluster, "job_id": job, "kind": kind}


def test_declared_jobs_come_from_the_registry_the_serving_jobs_extra_and_the_sacct_verified_ledger(m, tmp_path):
    run = _mt_run(Path("/s"), cluster_job_id="872043", producer_run_key="f22-att-q_s1-g01234-crescendo@872043", view="q_s1", model_id="q", eval_id="crescendo")
    assert m.unit_from_producer_identity(run) == "q/s1/multiturn/crescendo/g01234"
    # backfill-fix-1: the trial comes from the producer key, never registry trial_index (0 for both static trials); the padstat
    # default names its padding unit; a view that does not split names nothing
    assert m.unit_from_producer_identity(_static_run(Path("/s"), view="q_u0", model_id="q", trial_index=0, producer_run_key="sdfbase_full22_q_u0_sampled_t1")) == "q/u0/static/t1"
    assert m.unit_from_producer_identity(_static_run(Path("/s"), view="q_u0", model_id="q")) == "q/u0/padding/p16384"
    assert m.unit_from_producer_identity(_mt_run(Path("/s"), view="other_s1", model_id="q")) is None
    led = _ledger(tmp_path, rows=[_attempt("q/s1/multiturn/crescendo/g01234", "872043"), _attempt("q/s1/multiturn/crescendo/g5678", "169669", cluster="cluster_b"),
                                  _attempt("q/s1/multiturn/crescendo/g5678", "872050", n=2), _attempt("q/s1/multiturn/pair/g5678", "872060"),
                                  _attempt("q/s1/multiturn/mt-s1-cluster_a/attackers/1", "872001", kind="attacker"), _attempt("q/s2r/multiturn/crescendo/g5678", "833001")])
    calls = []
    def sacct(cluster, ids, cache=None):
        calls.append((cluster, tuple(ids)))
        return {j: ({"872043": "COMPLETED", "872050": "COMPLETED", "169669": None}.get(j)) for j in ids}
    d = m.declared_jobs_for(run, extra=None, ledger_state=led, sacct=sacct)
    assert d["declared_jobs"] == ["872043", "872050"] and d["declared_jobs_unverified"] == ["169669@cluster_b:unknown"] and d["declared_jobs_problem"] is None
    assert d["declared_jobs_unit"] == "q/s1/multiturn/crescendo/g01234" and d["declared_jobs_unit_how"] == "ledger"
    assert set(d["declared_jobs_sources"]["872043"]) == {"cluster_job_id", "producer_run_key", "ledger:q/s1/multiturn/crescendo/g01234@cluster_a@1@cluster_a:COMPLETED"}
    assert d["declared_jobs_sources"]["872050"] == ["ledger:q/s1/multiturn/crescendo/g5678@cluster_a@2@cluster_a:COMPLETED"]
    assert sorted(calls) == [("cluster_a", ("872043", "872050")), ("cluster_b", ("169669",))]          # one sacct per cluster, never the pair / attacker / other-view jobs
    # extra.serving_jobs is first-class; an entry the ledger does not know is the cross-check problem
    d = m.declared_jobs_for(run, extra={"serving_jobs": ["872050", "872043"]}, ledger_state=led, sacct=sacct)
    assert d["declared_jobs"] == ["872043", "872050"] and d["declared_jobs_problem"] is None and "extra.serving_jobs" in d["declared_jobs_sources"]["872050"]
    d = m.declared_jobs_for(run, extra={"serving_jobs": ["999999"]}, ledger_state=led, sacct=sacct)
    assert "999999" in d["declared_jobs"] and "extra.serving_jobs names 999999" in d["declared_jobs_problem"]
    d = m.declared_jobs_for(run, extra=None, ledger_state=None, sacct=sacct)                               # no ledger: the registry's job only
    assert d["declared_jobs"] == ["872043"] and d["declared_jobs_unverified"] == []


def test_the_manifest_rows_carry_declared_jobs_read_from_the_run_rows_extra_and_the_ledger(m, tmp_path):
    from sisteer.registry import db as DB, harvest as H, identity, records as R, synth
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    # the synthetic registry's own identity (its model and eval are seeded dimensions; a view_legacy of `<model>_<view>` names the unit's prefix)
    model = synth.run_payload("k", "none", "none", 0)["model_id"]
    rid, recs = synth.run_records("k_mt", 100, through="judged", view_legacy=f"{model}_s1")
    recs.append(synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"serving_jobs": ["4242", "4343"], "serving_jobs_source": "funnel ledger D1915"}}}, synth.t(130)))
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep.as_dict()
    db = Path(DB.db_path(root))
    led = _ledger(tmp_path, model=model, rows=[_attempt(f"{model}/s1/petri/proxy", "4242"), _attempt(f"{model}/s1/petri/proxy", "4343", n=2)])   # the eval is no attack: the petri prefix
    rows = m.manifest_from_registry(db, model=model, all_current=True, ledger_state=led, sacct=lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids})
    row = next(r for r in rows if r["run_id"] == rid)
    assert row["declared_jobs"] == ["4242", "4343"] and row["declared_jobs_problem"] is None and row["declared_jobs_unverified"] == []
    assert "extra.serving_jobs" in row["declared_jobs_sources"]["4343"] and "cluster_job_id" in row["declared_jobs_sources"]["4242"]


# ------------------------------------------------------------------------------------------------- (3) the export's one snapshot

def test_the_export_compares_one_registry_snapshot_with_the_manifest_and_the_measurement(m, tmp_path):
    out = tmp_path / "out"; out.mkdir()
    cells = {}
    for tag, cap in (("ok", 4096), ("patched", 4096)):
        c = _measured_cell(); c.update({"run_id": f"sdfbase_full22:pasttense:aaaa:{tag}", "host": "grading_host", "eval_id": "pasttense", "model_id": "q", "lane": "l", "cell": "none",
                                        "judgment_ids": [""], "reader": "r", "denominator_unit": "draws", "cap": cap, "instrument_id": "inst", "trial_index": 0, "pass_label": None})
        cells[c["run_id"]] = _export_ready(c, f"/s/{c['run_id']}")
    (out / "degeneracy_gap_fill.json").write_text(json.dumps({"definition_version": "degeneracy_gap_fill/1@deadbeef", "tool_sha256s": ["t"], "meta": {"runs": []}, "cells": cells}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps({"run_id": r, "store_uri": f"/s/{r}", "cell": "none", "eval_id": "pasttense", "instrument_id": "inst", "cap": 4096, "trial_index": 0, "declared_jobs": ["1"]}) for r in cells) + "\n")
    rows = {r: {"token_cap_tokens": 4096, "trial_index": 0} for r in cells}
    db = _registry_current(tmp_path, list(cells), rows=rows)
    # a run_patch between the measurement and the export: the registry's store_uri moved
    conn = sqlite3.connect(db); conn.execute("UPDATE run SET store_uri = '/s/moved' WHERE run_id = ?", ("sdfbase_full22:pasttense:aaaa:patched",)); conn.commit(); conn.close()
    class A: pass
    a = A(); a.out = str(out); a.manifest = str(manifest); a.batch = "b6"; a.dir = str(tmp_path / "by_run_id"); a.hosts = ""; a.filed = ""; a.registry = str(db)
    assert m.cmd_export(a) == 0
    body = json.loads(next((tmp_path / "by_run_id").glob("b6.v*.json")).read_text())
    assert [r["run_id"] for r in body["rows"]] == ["sdfbase_full22:pasttense:aaaa:ok"] and body["registry_snapshot_utc"] and body["currency_checked_utc"] == body["registry_snapshot_utc"]
    assert body["rows"][0]["provenance"]["export_tool"] == "degap-fix-6" and body["rows"][0]["provenance"]["registry_snapshot_utc"] and body["rows"][0]["provenance"]["declared_jobs"] == ["1"]
    why = {x["run_id"]: x["why"] for x in body["excluded_not_export_ready"]}
    assert set(why) == {"sdfbase_full22:pasttense:aaaa:patched"} and "registry store_uri='/s/moved' != manifest '/s/sdfbase_full22:pasttense:aaaa:patched'" in why["sdfbase_full22:pasttense:aaaa:patched"]
    assert "!= measurement" in why["sdfbase_full22:pasttense:aaaa:patched"]
    # the snapshot is one read: currency and identity for the same ids, and a run absent from the run table is named
    snap = m.registry_snapshot(db, list(cells) + ["absent"])
    assert snap["sdfbase_full22:pasttense:aaaa:ok"]["is_current"] == 1 and snap["sdfbase_full22:pasttense:aaaa:ok"]["token_cap_tokens"] == 4096 and "absent" not in snap
    assert m.snapshot_identity_problems(None, {}, {}) == ["no run row in the registry snapshot"]
    assert m.snapshot_identity_problems({"store_uri": "/s/x", "token_cap_tokens": 4096}, {"store_uri": "/s/x", "cap": 8192}, {"store_uri": "/s/x", "cap": 4096}) == ["registry token_cap_tokens=4096 != manifest 8192"]
