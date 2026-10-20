"""degen-fix-5 (2026-10-03): a controls run whose registry row carries NO n_items_generated is refused BEFORE the
claim. The degen-fix-3 refusal lived in population_filter, which the judge reached only after claim_all and after the grader produced item
records — so a NULL-count xstest run filed a partial judgment + `judged` + not_measured degeneration rows and the driver still wrote
JUDGE_DONE (the exception is caught in _registry_record_store). 27 such pending xstest runs were found registry-wide (other models).

Pinned here, through the REAL worklist over a real registry world and a real-shaped sampled store:
  * a NULL-count PENDING xstest run drives registry_worklist_cells to a nonzero exit naming the run, with ZERO claims (no judgment row, the run
    still pending) and ZERO filings (nothing spooled);
  * the same store with its count registered claims and carries the count to _registry_record_store (the positive control);
  * the rule is xstest's: an or_bench row without a count is not refused (its population is the whole store);
  * the non-worklist entry (--registry-runs JSON) refuses before any judging, by name;
  * the pure check lists every offending run.
Synthetic worlds only (TESTHOST, placeholder ids); a whitespace tokenizer stands in for the target's."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tests", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.registry import emit  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import identity as I  # noqa: E402
from sisteer.registry import pending as P  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from test_native_judging_cardinality import (  # noqa: E402  (the registry world + the real-shaped sampled store writer)
    EVALS, _WhitespaceTok, _conn, _dims, _instrument, _load, _registry, _write_sampled_store)
import degeneration_measures as DM  # noqa: E402


def _judgment_records(spool: str) -> list:
    out = []
    for dp, _dn, fns in os.walk(spool):
        out += [os.path.join(dp, f) for f in fns if "judgment" in f]
    return out


def _generated_run_without_a_count(key: str, i0: int, eval_id: str, store: str) -> tuple:
    """The registration class found in the wild: queued -> running -> generated with NO n_items_generated patch (the column stays NULL)."""
    inst = _instrument(eval_id)
    p = synth.run_payload(key, "none", "none", i0, eval_id=eval_id, item_set_id=f"{eval_id}_set", instrument_id=inst["instrument_id"],
                          decode_source=EVALS[eval_id][1], store_uri=store)
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, "none", eval_id, str(inst["instrument_id"]))
    rid = str(p["run_id"])
    t = synth.t
    recs = [synth.rec("run", p, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log"}, t(i0 + 1)),
            synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                    "evidence_uri": store}, t(i0 + 3))]
    return rid, recs


def _registry_with_null_counts(tmp_path, monkeypatch, runs) -> tuple:
    """``runs`` = [(key, eval_id, store, n_items | None)]; None = no count ever registered. Emission ON, spool = <root>/records."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims())
    rids = []
    for k, (key, eval_id, store, n_items) in enumerate(runs):
        rid, recs = _generated_run_without_a_count(key, 200 + 10 * k, eval_id, store)
        if n_items is not None:
            recs.append(synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": n_items}}, synth.t(200 + 10 * k + 4)))
        R.write_records(os.path.join(root, "records", "c", f"{k + 1:06d}-run.jsonl"), recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(emit.SPOOL_ENV, raising=False)
    monkeypatch.delenv(emit.PUSH_SCRIPT_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def _sampled_store(tmp_path, ids_xs, ids_orb=None) -> Path:
    store = tmp_path / "controls" / "synthtarget" / "cells" / "none__controls__t0__a0__2ff9cdff__0f1e2d3c"
    store.mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({"resolved_config": {"gen": {"do_sample": True, "max_new_tokens": 64}}}))
    (store / "gen_stats.json").write_text("{}")
    _write_sampled_store(store, "gens_xstest_v2", ids_xs, cap=64, at_cap_ids=set())
    if ids_orb is not None:
        _write_sampled_store(store, "gens_or_bench_hard_1k", ids_orb, cap=64, at_cap_ids=set())
    return store


def _judge(monkeypatch, tag: str):
    drv = _load(f"served_capability_judge_null_count_{tag}", "scripts/served_capability_judge.py")
    monkeypatch.setattr(drv, "Cell", lambda target, d, rows, controls: types.SimpleNamespace(
        target=target, cell_dir=d, name=d.name, controls=list(controls), judge_model="judge-model-x", gen_stats={}))
    DMm = drv._degeneration_measures()
    monkeypatch.setattr(DMm, "target_tokenizer", lambda mid, roster=None: _WhitespaceTok())
    return drv


def _ns(session="s-A"):
    return argparse.Namespace(registry_root=None, judge=None, session=session, campaign=None, org="lowprio", match="", worklist="registry",
                              run_id=None, model_id=None, store_root_map=None)


# ------------------------------------------------------------------------------------------------------------------ the regression

def test_a_null_count_pending_xstest_run_is_refused_before_any_claim_with_zero_claims_and_zero_filings(tmp_path, monkeypatch):
    store = _sampled_store(tmp_path, [f"xstest-{i}" for i in range(6)])
    root, (r_xs,) = _registry_with_null_counts(tmp_path, monkeypatch, [("k_xs_null", "xstest", str(store), None)])
    conn = _conn(root)
    assert conn.execute("SELECT n_items_generated FROM run WHERE run_id = ?", (r_xs,)).fetchone()[0] is None       # the NULL count, as registered
    assert [r["run_id"] for r in P.pending(conn)] == [r_xs]                                                      # and the run IS pending
    conn.close()
    drv = _judge(monkeypatch, "worklist")
    with pytest.raises(SystemExit) as ei:
        drv.registry_worklist_cells(_ns(), {}, ["xstest"])
    msg = str(ei.value)
    assert ei.value.code not in (0, None)                                                                        # a nonzero exit
    assert "refusing before any claim" in msg and "n_items_generated" in msg and r_xs in msg and "nothing claimed" in msg, msg
    assert "zero claims, zero filings" in msg
    assert _judgment_records(H.records_dir(root)) == []                                                          # zero filings
    assert drv.REGISTRY_RUNS is None                                                                              # the claim bookkeeping never ran
    conn = _conn(root)
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                                        # zero claims
    assert [r["run_id"] for r in P.pending(conn)] == [r_xs]                                                      # still pending, untouched
    conn.close()


def test_the_same_store_with_its_count_registered_claims_and_carries_the_count(tmp_path, monkeypatch):
    store = _sampled_store(tmp_path, [f"xstest-{i}" for i in range(6)])
    root, (r_xs,) = _registry(tmp_path, monkeypatch, [("k_xs", "xstest", str(store), 6, "none", "none", None)])
    drv = _judge(monkeypatch, "positive")
    wl, cells = drv.registry_worklist_cells(_ns(), {}, ["xstest"])
    assert len(cells) == 1 and cells[0].controls == ["xstest"]
    key = f"{cells[0].target}/{cells[0].name}"
    assert drv.REGISTRY_RUNS[key]["xstest"]["run_id"] == r_xs and drv.REGISTRY_RUNS[key]["xstest"]["n_items_generated"] == 6
    conn = _conn(root)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ? AND status = 'in_progress'", (r_xs,)).fetchone()[0] == 1   # claimed
    conn.close()


def test_the_rule_is_xstests_an_or_bench_row_without_a_count_is_not_refused(tmp_path, monkeypatch):
    store = _sampled_store(tmp_path, [f"xstest-{i}" for i in range(6)], [f"orb-{i}" for i in range(4)])
    root, (r_xs, r_orb) = _registry_with_null_counts(tmp_path, monkeypatch, [("k_xs", "xstest", str(store), 6), ("k_orb", "or_bench_hard", str(store), None)])
    drv = _judge(monkeypatch, "scope")
    wl, cells = drv.registry_worklist_cells(_ns(), {}, ["xstest", "or_bench_hard_1k"])
    assert cells[0].controls == ["or_bench_hard_1k", "xstest"]
    key = f"{cells[0].target}/{cells[0].name}"
    assert drv.REGISTRY_RUNS[key]["or_bench_hard_1k"]["n_items_generated"] is None and drv.REGISTRY_RUNS[key]["xstest"]["n_items_generated"] == 6
    conn = _conn(root)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'in_progress'").fetchone()[0] == 2
    conn.close()


def test_the_registry_runs_file_path_refuses_before_judging_by_name(monkeypatch):
    drv = _load("served_capability_judge_null_count_file", "scripts/served_capability_judge.py")
    runs = {"synthtarget/none__controls__t0__a0__2ff9cdff__0f1e2d3c": {
        "xstest": {"run_id": "c:xstest:aaaaaaaaaaaa:0000000000000001", "n_items_generated": None, "judge_id": "j"},
        "or_bench_hard_1k": {"run_id": "c:or_bench_hard:bbbbbbbbbbbb:0000000000000002", "n_items_generated": None, "judge_id": "j"}}}
    with pytest.raises(SystemExit) as ei:
        drv._check_registry_runs_xstest_counts(runs)
    assert ei.value.code not in (0, None)
    assert "refusing before judging" in str(ei.value) and "c:xstest:aaaaaaaaaaaa:0000000000000001" in str(ei.value) and "bbbbbbbbbbbb" not in str(ei.value)
    runs["synthtarget/none__controls__t0__a0__2ff9cdff__0f1e2d3c"]["xstest"]["n_items_generated"] = 250
    assert drv._check_registry_runs_xstest_counts(runs) is None
    assert drv._check_registry_runs_xstest_counts(None) is None and drv._check_registry_runs_xstest_counts({}) is None


def test_check_xstest_counts_lists_every_offending_run_and_ignores_other_controls():
    with pytest.raises(RuntimeError) as ei:
        DM.check_xstest_counts({("/s/a", "xstest"): (None, "run-a"), ("/s/b", "xstest"): ("", "run-b"), ("/s/c", "xstest"): (6, "run-c"),
                                ("/s/d", "or_bench_hard_1k"): (None, "run-d")})
    msg = str(ei.value)
    assert "run run-a (store /s/a)" in msg and "run run-b (store /s/b)" in msg and "run-c" not in msg and "run-d" not in msg
    assert "BEFORE the claim" in msg and "zero claims, zero filings" in msg
    DM.check_xstest_counts({("/s/c", "xstest"): (6, "run-c"), ("/s/d", "or_bench_hard_1k"): (None, "run-d")})
    DM.check_xstest_counts(None)
    DM.check_xstest_counts({})
    # preflight runs the count check FIRST — before any store is opened (a store that does not exist never gets as far as its shape)
    with pytest.raises(RuntimeError) as ei2:
        DM.preflight([(Path("/nonexistent/store"), ["xstest"])], ["m"], counts={("/nonexistent/store", "xstest"): (None, "run-x")})
    assert "run run-x" in str(ei2.value)
