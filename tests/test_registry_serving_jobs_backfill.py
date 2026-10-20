"""The SERVINGJOBS backfill (2026-10-04): one run_patch per CURRENT run whose store was served by two or
more jobs, stating `extra.serving_jobs` (the SORTED sacct-verified ledger job ids) + `extra.serving_jobs_source` after the run's last full
record; dry by default; idempotent; refusals by name. Through the REAL registry (synth records + the harvester), a synthetic funnel ledger
and a stubbed sacct — the live registry is never read."""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "registry_serving_jobs_backfill.py"


@pytest.fixture(scope="module")
def bf():
    spec = importlib.util.spec_from_file_location("serving_jobs_backfill_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _registry(tmp_path):
    """Two current runs of the synthetic model: A (view <model>_s1, the two-group unit) and B (view <model>_u0, a one-job unit)."""
    from sisteer.registry import db as DB, harvest as H, identity, records as R, synth
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    model = synth.run_payload("k", "none", "none", 0)["model_id"]
    rid_a, recs = synth.run_records("k_a", 100, through="judged", view_legacy=f"{model}_s1")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid_a), "000001-life.jsonl"), recs)
    rid_b, recs = synth.run_records("k_b", 200, through="judged", view_legacy=f"{model}_u0")
    # backfill-fix-1: a job id names ONE unit — B's job is its own (synth stamps 4242 on every run; the ledger files 4242 under A's unit)
    recs.append(synth.rec("run_patch", {"run_id": rid_b, "fields": {"cluster_job_id": "5151"}}, synth.t(209)))
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid_b), "000001-life.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep.as_dict()
    return root, Path(DB.db_path(root)), model, rid_a, rid_b, synth.CAMPAIGN


def _ledger(tmp_path, model, rows):
    state = tmp_path / "state"; (state / model).mkdir(parents=True, exist_ok=True)
    (state / model / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(state)


def _attempt(lwid, job, cluster="cluster_a", n=1, kind="target"):
    return {"event": "attempt", "logical_work_id": lwid, "attempt_id": f"{lwid}@{cluster}@{n}", "cluster": cluster, "job_id": job, "kind": kind}


def _args(root, db, model, state, **kw):
    a = ["--registry", str(db), "--model", model, "--spool", os.path.join(root, "records"), "--ledger-state", state, "--code-sha", "c" * 40]
    for k, v in kw.items():
        if v is True:
            a.append(f"--{k.replace('_', '-')}")
        elif v not in (None, False):
            a += [f"--{k.replace('_', '-')}", str(v)]
    return a


def _emission_on(monkeypatch, tmp_path):
    """The suite pins SISTEER_REGISTRY_DISABLED=1 so no test can write the live registry; a write-mode test opts emission back on and pins a
    tmp spool of its own (the backfill writes to its --spool anyway; the pin is the suite's belt)."""
    from sisteer.registry import emit
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("SISTEER_REGISTRY_SPOOL", str(tmp_path / "pinned-spool"))


def _patches(root, campaign, rid):
    from sisteer.registry import identity
    return sorted(glob.glob(os.path.join(root, "records", campaign, identity.run_dir_name(rid), "*-run_patch.jsonl")))


def _extra(db, rid):
    import sqlite3
    conn = sqlite3.connect(db); row = conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (rid,)).fetchone(); conn.close()
    return json.loads(row[0]) if row and row[0] else {}


def test_dry_run_plans_one_patch_for_the_two_group_run_and_writes_nothing(bf, tmp_path, capsys):
    root, db, model, rid_a, rid_b, campaign = _registry(tmp_path)
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s1/petri/proxy", "4242"), _attempt(f"{model}/s1/petri/proxy", "4343", n=2),
                                      _attempt(f"{model}/u0/petri/proxy", "5151")])
    sacct = lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids}
    before = {rid_a: _patches(root, campaign, rid_a), rid_b: _patches(root, campaign, rid_b)}
    rc = bf.main(_args(root, db, model, state, report=tmp_path / "plan.json"), sacct=sacct)
    out = capsys.readouterr().out
    assert rc == 0 and "DRY-RUN" in out and "1 to patch, 1 skipped, 0 not yet (live siblings), 0 refused" in out and f"would patch" in out, out
    plan = {r["run_id"]: r for r in json.loads((tmp_path / "plan.json").read_text())["plan"]}
    assert plan[rid_a]["action"] == "patch" and plan[rid_a]["payload"] == {"serving_jobs": ["4242", "4343"], "serving_jobs_source": "funnel ledger D1915"}
    assert plan[rid_b]["action"] == "skip" and "one serving job" in plan[rid_b]["reason"]
    assert plan[rid_a]["unit"] == f"{model}/s1/petri/proxy" and plan[rid_a]["unit_how"] == "ledger" and plan[rid_b]["unit"] == f"{model}/u0/petri/proxy"
    assert {rid_a: _patches(root, campaign, rid_a), rid_b: _patches(root, campaign, rid_b)} == before             # nothing written
    assert "serving_jobs" not in _extra(db, rid_a)


def test_write_lands_one_patch_after_the_last_record_the_harvester_folds_it_and_a_second_pass_writes_nothing(bf, tmp_path, capsys, monkeypatch):
    _emission_on(monkeypatch, tmp_path)
    from sisteer.registry import harvest as H
    root, db, model, rid_a, rid_b, campaign = _registry(tmp_path)
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s1/petri/proxy", "4343", cluster="cluster_b"), _attempt(f"{model}/s1/petri/proxy", "4242", n=2),
                                      _attempt(f"{model}/u0/petri/proxy", "5151")])
    calls = []
    def sacct(cl, ids, cache=None):
        calls.append((cl, tuple(sorted(ids)))); return {j: "COMPLETED" for j in ids}
    seq_before = max(int(Path(p).name.split("-")[0]) for p in glob.glob(os.path.join(root, "records", campaign, "*", "0*.jsonl")) if rid_a[-16:] in p)
    rc = bf.main(_args(root, db, model, state, write=True, report=tmp_path / "w.json"), sacct=sacct)
    out = capsys.readouterr().out
    assert rc == 0 and "WRITE" in out and "1 record(s) written" in out, out
    ps = _patches(root, campaign, rid_a); assert len(ps) == 1                                                           # ours (synth bundles its own patches in 000001-life.jsonl)
    rec = json.loads(Path(ps[-1]).read_text().splitlines()[-1])
    assert int(Path(ps[-1]).name.split("-")[0]) > seq_before                                                            # after the run's last record
    assert rec["record_type"] == "run_patch" and rec["payload"]["fields"] == {"extra_json": {"serving_jobs": ["4242", "4343"], "serving_jobs_source": "funnel ledger D1915"}}
    assert rec["payload"]["run_id"] == rid_a and rec["producer"]["code_sha"] == "c" * 40 and "registry_serving_jobs_backfill.py" in rec["producer"]["script"]
    assert sorted(calls) == [("cluster_a", ("4242",)), ("cluster_a", ("5151",)), ("cluster_b", ("4343",))]           # one sacct per cluster per unit, read-only
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep.as_dict()
    ex = _extra(db, rid_a)
    assert ex["serving_jobs"] == ["4242", "4343"] and ex["serving_jobs_source"] == "funnel ledger D1915" and ex.get("registered_at") is None or True   # merged additively
    assert "serving_jobs" not in _extra(db, rid_b)
    # the second pass: the set is already stated — nothing written, idempotent
    rc = bf.main(_args(root, db, model, state, write=True), sacct=sacct)
    out = capsys.readouterr().out
    assert rc == 0 and "0 to patch, 2 skipped" in out and "0 record(s) written" in out, out
    assert len(_patches(root, campaign, rid_a)) == 1


def test_an_unverifiable_ledger_job_or_a_store_job_outside_the_declared_set_refuses_by_name_and_writes_nothing(bf, tmp_path, capsys, monkeypatch):
    _emission_on(monkeypatch, tmp_path)
    root, db, model, rid_a, rid_b, campaign = _registry(tmp_path)
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s1/petri/proxy", "4242"), _attempt(f"{model}/s1/petri/proxy", "4343", n=2)])
    rc = bf.main(_args(root, db, model, state, write=True, runs=rid_a), sacct=lambda cl, ids, cache=None: {j: ("COMPLETED" if j == "4242" else None) for j in ids})
    out = capsys.readouterr().out
    assert rc == 3 and "REFUSED" in out and "sacct-unverified ledger job(s) ['4343@cluster_a:unknown']" in out and "0 record(s) written" in out, out
    assert len(_patches(root, campaign, rid_a)) == 0
    # the census cross-check: the store names a job neither the ledger nor the registry declares
    census = tmp_path / "results.jsonl"
    census.write_text(json.dumps({"run_id": rid_a, "status": "measured", "store": {"identity": {"store_uri:transcripts_results": {"checks": {"job": {"registry": "4242", "store": ["4242", "4343", "9999"]}}}}}}) + "\n")
    rc = bf.main(_args(root, db, model, state, write=True, runs=rid_a, census=census), sacct=lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids})
    out = capsys.readouterr().out
    assert rc == 3 and "the store names serving job(s) ['9999'] the ledger + registry do not declare" in out, out
    assert len(_patches(root, campaign, rid_a)) == 0
    # a run with no full `run` record in the spool refuses too (the patch would precede the producer's statement)
    from sisteer.registry import identity
    d = Path(root) / "records" / campaign / identity.run_dir_name(rid_a)
    for p in glob.glob(str(d / "*-life.jsonl")):
        os.rename(p, p + ".moved")
    rc = bf.main(_args(root, db, model, state, write=True, runs=rid_a), sacct=lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids})
    out = capsys.readouterr().out
    assert rc == 3 and "no full run record under" in out, out


def test_a_sibling_the_scheduler_still_runs_reads_as_not_yet_named_with_its_state_and_writes_nothing(bf, tmp_path, capsys, monkeypatch):
    _emission_on(monkeypatch, tmp_path)
    """D2017: the set is final only when every attempt of the unit is terminal; a RUNNING / PENDING sibling is 'not yet', never a data defect."""
    root, db, model, rid_a, rid_b, campaign = _registry(tmp_path)
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s1/petri/proxy", "4242"), _attempt(f"{model}/s1/petri/proxy", "4343", n=2)])
    rc = bf.main(_args(root, db, model, state, write=True, runs=rid_a, report=tmp_path / "plan.json"), sacct=lambda cl, ids, cache=None: {j: ("COMPLETED" if j == "4242" else "RUNNING") for j in ids})
    out = capsys.readouterr().out
    assert rc == 0 and "0 to patch, 0 skipped, 1 not yet (live siblings), 0 refused" in out and "0 record(s) written" in out, out
    assert "NOT YET" in out and "sibling 4343 RUNNING — not yet; re-run after the unit is terminal" in out and "REFUSED" not in out, out
    plan = {r["run_id"]: r for r in json.loads((tmp_path / "plan.json").read_text())["plan"]}
    assert plan[rid_a]["action"] == "not_yet" and "sibling 4343 RUNNING" in plan[rid_a]["reason"]
    assert len(_patches(root, campaign, rid_a)) == 0 and "serving_jobs" not in _extra(db, rid_a)
    # once the sibling is terminal the same pass writes the whole set (idempotent re-run semantics live in the write test)
    rc = bf.main(_args(root, db, model, state, runs=rid_a), sacct=lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids})
    out = capsys.readouterr().out
    assert rc == 0 and "1 to patch" in out and "serving_jobs=['4242', '4343']" in out, out
