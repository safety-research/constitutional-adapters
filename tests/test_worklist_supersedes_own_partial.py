"""A COMPLETE judgment filed through the registry worklist supersedes the same judge's
earlier PARTIAL judgments of the run.

A re-judge appends under a NEW judgment_id (pending.claim mints one per claim; the registry's run_metric key carries the judgment id),
and no code path wrote `superseded`, so the partial stayed current beside the complete and v_paper_long carried duplicate metric rows
(three misaligned_rate rows on one run, 2026-10-02). Now `RegistryWorklist.scope` reads the (run, judge)'s partial judgments when the
scope is built (the claim is held, so the set cannot change under it) and the scope, on `complete("complete")` -- AFTER its completion
record is on the spool -- emits one judgment record per partial: the SAME judgment_id, status `superseded`, `superseded_by_judgment_id`
= the complete judgment, and `extra.superseded_reason` naming it; nothing else is restated (the partial's own provenance stands under the
per-column fold). The registry's currency predicate already honours `superseded` (schema_def.NON_CURRENT_JUDGMENT_STATUSES). A
partial by ANOTHER judge, another session's in_progress claim, a partial re-filed as partial, and a first complete with no partial behind
it emit nothing. Synthetic ids throughout."""
from __future__ import annotations

import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import hooks
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import worklist as W

SEED_ID = "synthetic_judge_two"   # the readable id judges.yaml would carry for synth.JUDGE_ID_2; the loader parks it in extra_json
DRIVER = "tests/stub_driver.py"


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _dims_with_seed_id():
    dims = synth.seed_dimensions()
    for r in dims:
        if r["record_type"] == "judge" and r["payload"]["judge_id"] == synth.JUDGE_ID_2:
            r["payload"]["seed_judge_id"] = SEED_ID
    return dims


def _root_with(tmp_path, runs, monkeypatch):
    """A registry with the dimension seed and ``runs`` = [(key, store_uri)] ingested through ``generated``."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", _dims_with_seed_id())
    rids = []
    for k, (key, store) in enumerate(runs):
        rid, recs = synth.run_records(key, 100 + 20 * k, through="generated")
        recs[0]["payload"]["store_uri"] = store
        _write(root, "c", f"00000{k + 1}-run.jsonl", recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def _fold(root):
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled records (finish() folds nothing)
    assert rep.refused == 0, rep.as_dict()
    return rep


def _items(n):
    return [{"item_id": f"q{i:05d}", "metric_id": "asr", "value": float(i % 2), "label": "harmful" if i % 2 else "safe"} for i in range(n)]


def _metric(n):
    return [{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": n // 2, "n": n}]


def _grade_partial(n):
    def grade(row, scope):
        scope.items(_items(n))
        scope.metrics(_metric(n))
        scope.complete("partial", n_items_judged=n)
    return grade


def _grade_complete(n):
    def grade(row, scope):                 # no explicit complete: the scope's __exit__ files `complete`, the drivers' common shape
        scope.items(_items(n))
        scope.metrics(_metric(n))
    return grade


def _spool_judgments(root, rid):
    d = os.path.join(root, "records", synth.CAMPAIGN, I.run_dir_name(rid))
    out = []
    for p in R.iter_record_files(d):
        for _, rec, _ in R.read_record_file(p):
            if rec["record_type"] == "judgment":
                out.append(rec["payload"])
    return out


def test_a_complete_judgment_supersedes_the_same_judges_earlier_partial_and_nothing_else(tmp_path, monkeypatch, capsys):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    # an earlier dispatch's PARTIAL by judge J2 (session s-A)
    wl_a = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    rows = wl_a.pending(eval_category="static_attack")
    assert [r["run_id"] for r in rows] == [rid] and rows[0]["pending_reason"] == "no_judgment"
    wl_a.grade(rows, _grade_partial(150))
    _fold(root)
    partial_id = wl_a.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"]
    # ANOTHER judge's partial (J1, the instrument judge; session s-C), then a live in_progress claim on (R, J1) by session s-B
    wl_c = W.RegistryWorklist(root, [synth.JUDGE_ID], "s-C", driver=DRIVER, api_pool="lowprio")
    rows_c = wl_c.pending(eval_category="static_attack")
    assert [r["run_id"] for r in rows_c] == [rid]
    wl_c.grade(rows_c, _grade_partial(120))
    _fold(root)
    other_partial = wl_c.claims[(rid, synth.JUDGE_ID)]["judgment_id"]
    other_claim = P.claim(root, rid, synth.JUDGE_ID, "s-B")
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone()[0] == "partial"
    pend = [r for r in P.pending_for(conn, SEED_ID) if r["run_id"] == rid]
    assert len(pend) == 1 and pend[0]["pending_reason"] == "partial"           # the run pends for J2 through its partial
    fin, cu = tuple(conn.execute("SELECT finished_utc, claimed_utc FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone())
    assert W.own_partial_judgments(conn, rid, synth.JUDGE_ID_2) == [{"judgment_id": partial_id, "n_items_judged": 150, "finished_utc": fin,
                                                                      "claimed_by_session": "s-A", "claimed_utc": cu, "producer_session": "s-A"}]
    assert cu and I.judgment_id(rid, synth.JUDGE_ID_2, "s-A", cu) == partial_id        # a claim-minted id reproduces from the row's claim pair
    assert [p["judgment_id"] for p in W.own_partial_judgments(conn, rid, synth.JUDGE_ID)] == [other_partial]   # J1's own partial; never the in_progress claim
    conn.close()
    # the re-judge by J2 (session s-A2) completes
    wl2 = W.RegistryWorklist(root, [SEED_ID], "s-A2", driver=DRIVER, api_pool="lowprio")
    rows2 = wl2.pending(eval_category="static_attack")
    assert [r["run_id"] for r in rows2] == [rid]
    scope = wl2.scope(rows2[0])
    assert isinstance(scope, W.SupersedingJudgingScope) and [p["judgment_id"] for p in scope.supersedes] == [partial_id]
    with scope:
        _grade_complete(200)(rows2[0], scope)
    complete_id = scope.judgment_id
    assert scope.completion_emitted and scope.superseded == [partial_id]
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    p = conn.execute("SELECT status, superseded_by_judgment_id, n_items_judged, claimed_by_session, api_pool, "
                     "json_extract(extra_json, '$.superseded_reason') FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone()
    assert p[0] == "superseded" and p[1] == complete_id and complete_id in p[5] and "D1675" in p[5]
    assert (p[2], p[3], p[4]) == (150, "s-A", "lowprio")                        # the partial's own fields stand: only the status changed
    assert tuple(conn.execute("SELECT status, n_items_judged, n_errors FROM judgment WHERE judgment_id = ?", (complete_id,)).fetchone()) == ("complete", 200, 0)
    # the other judge's partial and the other session's claim are untouched
    assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (other_partial,)).fetchone()) == ("partial", None)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (other_claim,)).fetchone()[0] == "in_progress"
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 4
    # currency: J2's current item and metric rows are the complete's only; the other judge's partial rows still stand
    cur_items = {r[0] for r in conn.execute("SELECT DISTINCT judgment_id FROM v_item_result_current WHERE run_id = ?", (rid,))}
    assert cur_items == {complete_id, other_partial}
    cur_metrics = {r[0] for r in conn.execute("SELECT rm.judgment_id FROM run_metric rm LEFT JOIN judgment jm ON jm.judgment_id = rm.judgment_id "
                                              "WHERE rm.run_id = ? AND " + S.non_current_judgment_sql("rm.judgment_id", "jm.status"), (rid,))}
    assert cur_metrics == {complete_id, other_partial}
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (rid,)).fetchone()[0] == 3     # the tables keep every row
    # the run is covered for J2 (no longer pending) and still pending for J1 (claimed); its lifecycle is judged
    assert [r for r in P.pending_for(conn, SEED_ID) if r["run_id"] == rid] == []
    j1 = [r for r in P.pending_for(conn, synth.JUDGE_ID) if r["run_id"] == rid]
    assert len(j1) == 1 and j1[0]["pending_reason"] == "claimed"
    assert tuple(conn.execute("SELECT lifecycle_state, n_items_judged FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("judged", 200)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()
    # the superseded record states ONLY the status change -- no restated provenance of the partial
    sup = [j for j in _spool_judgments(root, rid) if j.get("status") == "superseded"]
    assert len(sup) == 1 and sup[0]["judgment_id"] == partial_id and sup[0]["superseded_by_judgment_id"] == complete_id
    assert set(sup[0]) - {"extra"} == {"judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id", "claimed_by_session", "claimed_utc"}
    assert (sup[0]["claimed_by_session"], sup[0]["claimed_utc"]) == ("s-A", cu)          # the CLAIMED form's pair -- the partial's, restated
    assert I.judgment_id(rid, synth.JUDGE_ID_2, sup[0]["claimed_by_session"], sup[0]["claimed_utc"]) == sup[0]["judgment_id"]
    assert sup[0]["judge_id"] == synth.JUDGE_ID_2 and complete_id in sup[0]["extra"]["superseded_reason"]
    assert f"partial judgment {partial_id}" in capsys.readouterr().err


def test_a_partial_refile_supersedes_nothing_and_a_first_complete_emits_no_superseded_record(tmp_path, monkeypatch):
    root, (r_first, r_twice) = _root_with(tmp_path, [("k_first", "file:///stores/k_first"), ("k_twice", "file:///stores/k_twice")], monkeypatch)
    wl = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    rows = {r["run_id"]: r for r in wl.pending(eval_category="static_attack")}
    # a first complete with no partial behind it: nothing to supersede, exactly one completion record
    scope = wl.scope(rows[r_first])
    assert scope.supersedes == []
    with scope:
        _grade_complete(200)(rows[r_first], scope)
    assert scope.superseded == []
    # a partial re-filed as partial: the earlier partial stands (nothing displaced it)
    wl.grade([rows[r_twice]], _grade_partial(100))
    _fold(root)
    wl_b = W.RegistryWorklist(root, [SEED_ID], "s-B", driver=DRIVER, api_pool="lowprio")
    rows_b = [r for r in wl_b.pending(eval_category="static_attack") if r["run_id"] == r_twice]
    assert len(rows_b) == 1 and rows_b[0]["pending_reason"] == "partial"
    scope_b = wl_b.scope(rows_b[0])
    assert [p["n_items_judged"] for p in scope_b.supersedes] == [100]
    with scope_b:
        _grade_partial(110)(rows_b[0], scope_b)
    assert scope_b.superseded == []
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [tuple(r) for r in conn.execute("SELECT status FROM judgment WHERE run_id = ?", (r_first,))] == [("complete",)]
    assert sorted(r[0] for r in conn.execute("SELECT status FROM judgment WHERE run_id = ?", (r_twice,))) == ["partial", "partial"]
    assert [p["n_items_judged"] for p in W.own_partial_judgments(conn, r_twice, synth.JUDGE_ID_2)] == [100, 110]   # finished order
    conn.close()
    assert [j for j in _spool_judgments(root, r_first) + _spool_judgments(root, r_twice) if j.get("status") == "superseded"] == []


def test_a_disabled_emission_still_refuses_the_scope_by_name(tmp_path, monkeypatch):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    wl = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER)
    rows = wl.pending(eval_category="static_attack")
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    with pytest.raises(W.RegistryUnavailable, match="emission is disabled"):
        wl.scope(rows[0])


# --------------------------------------------------------------------------------------------
# the completion and its supersessions are ONE atomic record file
# --------------------------------------------------------------------------------------------


def _run_dir_files(root, rid):
    d = os.path.join(root, "records", synth.CAMPAIGN, I.run_dir_name(rid))
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _records_in(root, rid, name):
    return [rec for _, rec, _ in R.read_record_file(os.path.join(root, "records", synth.CAMPAIGN, I.run_dir_name(rid), name))]


def test_the_completion_and_its_supersessions_are_one_atomic_record_file_in_fold_order(tmp_path, monkeypatch, capsys):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    wl_a = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    wl_a.grade(wl_a.pending(eval_category="static_attack"), _grade_partial(150))
    _fold(root)
    partial_id = wl_a.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"]
    wl2 = W.RegistryWorklist(root, [SEED_ID], "s-A2", driver=DRIVER, api_pool="lowprio")
    (row,) = wl2.pending(eval_category="static_attack")
    scope = wl2.scope(row)
    assert isinstance(scope.writer, W.AtomicBatchWriter) and scope.writer.producer["script"] == DRIVER
    before = set(_run_dir_files(root, rid))
    with scope:
        _grade_complete(200)(row, scope)
    complete_id = scope.judgment_id
    new = [f for f in _run_dir_files(root, rid) if f not in before]
    assert not [f for f in new if f.endswith(".tmp")]                                                  # no temp left behind
    judgment_files = [f for f in new if f.endswith("-judgment.jsonl")]
    assert len(judgment_files) == 1, new                                                               # ONE judgment file for the completion AND its supersessions
    recs = _records_in(root, rid, judgment_files[0])
    assert [r["payload"]["status"] for r in recs] == ["complete", "superseded"]                       # the complete first, the superseded after it
    assert recs[0]["payload"]["judgment_id"] == complete_id and recs[1]["payload"]["judgment_id"] == partial_id
    assert recs[1]["payload"]["superseded_by_judgment_id"] == complete_id and recs[0]["written_utc"] < recs[1]["written_utc"]   # fold order = file order
    assert set(recs[1]["payload"]) - {"extra"} == {"judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id", "claimed_by_session", "claimed_utc"}
    assert [r["producer"]["script"] for r in recs] == [DRIVER, DRIVER]
    # the other completion records (the degeneration not_measured rows, run_state judged, run_patch) follow in their own files, as before
    assert {f.split("-", 1)[1] for f in new} >= {"judgment.jsonl", "run_state.jsonl", "run_patch.jsonl"}
    assert scope.superseded == [partial_id] and "in the completion's own record file" in capsys.readouterr().err
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT status, superseded_by_judgment_id, n_items_judged FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone()) == ("superseded", complete_id, 150)
    assert tuple(conn.execute("SELECT status, n_items_judged FROM judgment WHERE judgment_id = ?", (complete_id,)).fetchone()) == ("complete", 200)
    assert {r[0] for r in conn.execute("SELECT DISTINCT judgment_id FROM v_item_result_current WHERE run_id = ?", (rid,))} == {complete_id}
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "judged"
    conn.close()
    # the payload builder is the ONE shape (the daemon's reconciliation and the correction tool build the same records), fed the partial's row
    # as own_partial_judgments reads it -- its identity fields included
    with DB.reading(DB.db_path(root)) as conn:
        p_sess, p_cu = tuple(conn.execute("SELECT claimed_by_session, claimed_utc FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone())
    built = W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, complete_id, [{"judgment_id": partial_id, "n_items_judged": 150, "claimed_by_session": p_sess, "claimed_utc": p_cu}],
                                           now=recs[1]["payload"]["extra"]["superseded_utc"])
    assert built == [recs[1]["payload"]]


def test_an_interrupted_batch_write_leaves_nothing_on_the_spool_and_the_partial_standing(tmp_path, monkeypatch):
    """An interruption injected where the old code had written the complete and not yet its supersessions: the atomic write fails -- no
    record file of any kind lands, completion_emitted stays False, the claim stays in_progress for the failure path, the partial stands."""
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    wl_a = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    wl_a.grade(wl_a.pending(eval_category="static_attack"), _grade_partial(150))
    _fold(root)
    partial_id = wl_a.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"]
    wl2 = W.RegistryWorklist(root, [SEED_ID], "s-A2", driver=DRIVER, api_pool="lowprio")
    (row,) = wl2.pending(eval_category="static_attack")
    scope = wl2.scope(row)
    real_link = os.link

    def dying_link(src, dst, *a, **k):
        if "-judgment.jsonl" in dst:
            raise OSError(28, "No space left on device (synthetic)")
        return real_link(src, dst, *a, **k)

    monkeypatch.setattr(W.os, "link", dying_link)
    before = set(_run_dir_files(root, rid))
    with pytest.raises(OSError, match="synthetic"):
        with scope:
            _grade_complete(200)(row, scope)
    new = [f for f in _run_dir_files(root, rid) if f not in before]
    assert not [f for f in new if "judgment" in f or f.endswith(".tmp")], new                       # no completion, no supersession, no temp
    assert scope.closed and not scope.completion_emitted and scope.superseded == []
    _fold(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone()) == ("partial", None)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (scope.judgment_id,)).fetchone()[0] == "in_progress"   # the claim, for the failure path
    assert [r["pending_reason"] for r in P.pending_for(conn, SEED_ID) if r["run_id"] == rid] == ["claimed"]
    conn.close()


def test_write_supersessions_and_reconcile_use_the_same_writer_and_shape_and_are_idempotent(tmp_path, monkeypatch):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    wl_a = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    wl_a.grade(wl_a.pending(eval_category="static_attack"), _grade_partial(150))
    _fold(root)
    partial_id = wl_a.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"]
    # the legacy half-written state: a complete filed through the plain hooks scope (no supersession)
    jid = P.claim(root, rid, synth.JUDGE_ID_2, "s-legacy")
    with DB.reading(DB.db_path(root)) as conn:
        info = P.active_claim(conn, rid, synth.JUDGE_ID_2)
    legacy = hooks.judgment_scope(DRIVER, {"run_id": rid, "campaign": synth.CAMPAIGN}, {"judge_id": synth.JUDGE_ID_2}, spool=H.records_dir(root),
                                  registry_emit=True, session="s-legacy", claim={"judgment_id": jid, "claimed_utc": info["claimed_utc"]})
    with legacy:
        _grade_complete(200)(None, legacy)
    _fold(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone()[0] == "partial"      # still current beside the complete
        cands = W.supersession_candidates(conn, model_ids=[synth.MODEL_ID])
    assert [(c["run_id"], c["judge_id"], c["complete_id"], [p["judgment_id"] for p in c["partials"]]) for c in cands] == [(rid, synth.JUDGE_ID_2, jid, [partial_id])]
    # a dry run builds the payloads and writes nothing
    path, payloads = W.write_supersessions(root, campaign=synth.CAMPAIGN, run_id=rid, judge_id=synth.JUDGE_ID_2, complete_id=jid, partials=cands[0]["partials"],
                                           script="tests/tool.py", session="s-tool", disabled=True)
    assert path is None and [p["judgment_id"] for p in payloads] == [partial_id] and payloads[0]["superseded_by_judgment_id"] == jid
    before = set(_run_dir_files(root, rid))
    done = W.reconcile_supersessions(root, model_ids=[synth.MODEL_ID], session="s-recon", script="tests/recon.py")
    assert [(d["run_id"], d["partial_ids"], d["complete_id"]) for d in done] == [(rid, [partial_id], jid)] and os.path.isfile(done[0]["path"])
    new = [f for f in _run_dir_files(root, rid) if f not in before]
    assert len(new) == 1 and new[0].endswith("-judgment.jsonl")
    (rec,) = _records_in(root, rid, new[0])
    assert rec["payload"]["status"] == "superseded" and rec["producer"]["script"] == "tests/recon.py" and "reconciled" in rec["payload"]["extra"]["superseded_reason"]
    assert W.reconcile_supersessions(root, model_ids=[synth.MODEL_ID], session="s-recon", script="tests/recon.py") == []    # the spool record stops a second write
    _fold(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (partial_id,)).fetchone()) == ("superseded", jid)
        assert W.supersession_candidates(conn, model_ids=[synth.MODEL_ID]) == []


# ------------------------------------------------------------------------------------------- the identity inputs, by FORM
# The registry's rule: a judgment record's id must be reproducible from its own fields, under whichever of the
# registry's three identity forms minted it -- PLAIN identity.judgment_id(run, judge) (the 2026-09 static-attack imports: zero fields to
# state), CLAIMED uuid5 over run|judge|claimed_by_session|claimed_utc (pending.claim's mint), PRODUCER over run|judge|producer_session|
# claimed_utc. A same-id `superseded` record states exactly that form's fields; a partial NO form reproduces refuses the whole batch by name,
# nothing written. Synthetic ids throughout; the fixture registry only.

BASE_KEYS = {"judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id"}
STAMP = "2026-09-17T00:00:00.000Z"


def _legacy_partial(root, rid, judge, *, judgment_id, n_items=150, **identity_fields):
    """A PARTIAL judgment record in a legacy shape: the given judgment_id and exactly the identity fields passed (the plain form states none;
    the producer form states producer_session + claimed_utc) -- what a pre-claim row looks like."""
    w = emit.RecordWriter(H.records_dir(root), synth.CAMPAIGN, rid, emit.producer_block("tests/legacy_import.py", session="s-legacy"), disabled=False)
    w.emit_judgment({"judgment_id": judgment_id, "run_id": rid, "judge_id": judge, "status": "partial", "started_utc": STAMP, "finished_utc": STAMP,
                     "n_items_judged": n_items, "api_pool": "lowprio", "store_uri": "synthetic:///legacy", **identity_fields})
    _fold(root)
    return judgment_id


def _strip_claim_inputs(root, jid):
    """A row NO form reproduces: a claim-minted id whose claim pair the fixture registry's row no longer states (the producer_session the
    scope stated stands, its claimed_utc does not) -- put on the row directly; own_partial_judgments and the reconciliation read the TABLE."""
    conn = DB.connect(DB.db_path(root))
    try:
        conn.execute("UPDATE judgment SET claimed_by_session = NULL, claimed_utc = NULL WHERE judgment_id = ?", (jid,))
        conn.commit()
    finally:
        conn.close()


def _retire_through_scope(root, rid, session):
    """The re-judge: a worklist scope completes the run and retires the standing partial(s) in the completion's own record file; returns
    (scope, the superseded payloads in that file)."""
    wl2 = W.RegistryWorklist(root, [SEED_ID], session, driver=DRIVER, api_pool="lowprio")
    (row,) = wl2.pending(eval_category="static_attack")
    before = set(_run_dir_files(root, rid))
    scope = wl2.scope(row)
    with scope:
        _grade_complete(200)(row, scope)
    (jf,) = [f for f in _run_dir_files(root, rid) if f not in before and f.endswith("-judgment.jsonl")]
    return scope, [r["payload"] for r in _records_in(root, rid, jf) if r["payload"]["status"] == "superseded"]


def test_a_claim_minted_partial_states_its_claim_pair_and_the_owners_helper_states_it_when_present(tmp_path, monkeypatch):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    wl_a = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    wl_a.grade(wl_a.pending(eval_category="static_attack"), _grade_partial(150))
    _fold(root)
    partial_id = wl_a.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"]
    with DB.reading(DB.db_path(root)) as conn:
        (p,) = W.own_partial_judgments(conn, rid, synth.JUDGE_ID_2)
    assert p["claimed_by_session"] == "s-A" and p["claimed_utc"] and p["producer_session"] == "s-A"
    assert W.judgment_identity_form(p, run_id=rid, judge_id=synth.JUDGE_ID_2) == ("claimed", {"claimed_by_session": "s-A", "claimed_utc": p["claimed_utc"]})
    # (a) the builder: the claim pair on the payload, the id reproducible from it -- the brief's assertion, verbatim
    (payload,) = W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, "11111111-2222-4333-8444-555555555555", [p])
    assert (payload["claimed_by_session"], payload["claimed_utc"]) == ("s-A", p["claimed_utc"])
    assert I.judgment_id(rid, synth.JUDGE_ID_2, payload["claimed_by_session"], payload["claimed_utc"]) == payload["judgment_id"] == partial_id
    assert set(payload) - {"extra"} == BASE_KEYS | {"claimed_by_session", "claimed_utc"}
    # (b) the scope's attach path writes the same pair into the completion's own record file; the fold keeps it on the row
    scope, sup = _retire_through_scope(root, rid, "s-A2")
    assert [(x["judgment_id"], x["claimed_by_session"], x["claimed_utc"]) for x in sup] == [(partial_id, "s-A", p["claimed_utc"])]
    _fold(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id, claimed_by_session, claimed_utc FROM judgment WHERE judgment_id = ?",
                                  (partial_id,)).fetchone()) == ("superseded", scope.judgment_id, "s-A", p["claimed_utc"])
    # (c) the registry owner's helper, when the identity module has it, is what states the inputs -- called with the partial's row plus run/judge
    seen = []

    def helper(r):
        seen.append(dict(r))
        return {"claimed_by_session": r["claimed_by_session"], "claimed_utc": r["claimed_utc"], "identity_stated_by": "helper"}

    monkeypatch.setattr(I, "judgment_identity_inputs", helper, raising=False)
    (via_helper,) = W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, scope.judgment_id, [p])
    assert via_helper["identity_stated_by"] == "helper" and (via_helper["claimed_by_session"], via_helper["claimed_utc"]) == ("s-A", p["claimed_utc"])
    assert seen[0]["judgment_id"] == partial_id and (seen[0]["run_id"], seen[0]["judge_id"]) == (rid, synth.JUDGE_ID_2)


def test_a_plain_form_partial_retires_with_a_record_stating_no_identity_inputs(tmp_path, monkeypatch):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    plain_id = _legacy_partial(root, rid, synth.JUDGE_ID_2, judgment_id=I.judgment_id(rid, synth.JUDGE_ID_2))     # no session, no time: the pre-claim imports
    with DB.reading(DB.db_path(root)) as conn:
        (p,) = W.own_partial_judgments(conn, rid, synth.JUDGE_ID_2)
    assert p["judgment_id"] == plain_id and (p["claimed_by_session"], p["claimed_utc"], p["producer_session"]) == (None, None, None)
    assert W.judgment_identity_form(p, run_id=rid, judge_id=synth.JUDGE_ID_2) == ("plain", {})
    (payload,) = W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, "11111111-2222-4333-8444-555555555555", [p])
    assert set(payload) - {"extra"} == BASE_KEYS and I.judgment_id(payload["run_id"], payload["judge_id"]) == payload["judgment_id"] == plain_id
    # the scope retires it in the completion's own file, stating only run / judge (+ status / successor); the fold reads it superseded
    scope, sup = _retire_through_scope(root, rid, "s-A2")
    assert [set(x) - {"extra"} for x in sup] == [BASE_KEYS] and sup[0]["judgment_id"] == plain_id and sup[0]["superseded_by_judgment_id"] == scope.judgment_id
    _fold(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id, claimed_by_session, claimed_utc, n_items_judged FROM judgment WHERE judgment_id = ?",
                                  (plain_id,)).fetchone()) == ("superseded", scope.judgment_id, None, None, 150)
        assert W.supersession_candidates(conn, model_ids=[synth.MODEL_ID]) == []
    # the owner's helper answers {} for the plain form -- the same record
    monkeypatch.setattr(I, "judgment_identity_inputs", lambda r: {}, raising=False)
    assert set(W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, scope.judgment_id, [p])[0]) - {"extra"} == BASE_KEYS


def test_a_producer_session_form_partial_states_its_pair(tmp_path, monkeypatch):
    root, (rid,) = _root_with(tmp_path, [("k", "file:///stores/k")], monkeypatch)
    cu = "2026-09-17T01:02:03.000Z"
    prod_id = _legacy_partial(root, rid, synth.JUDGE_ID_2, judgment_id=I.judgment_id(rid, synth.JUDGE_ID_2, "producer-x", cu), producer_session="producer-x", claimed_utc=cu)
    with DB.reading(DB.db_path(root)) as conn:
        (p,) = W.own_partial_judgments(conn, rid, synth.JUDGE_ID_2)
    assert (p["claimed_by_session"], p["claimed_utc"], p["producer_session"]) == (None, cu, "producer-x")
    assert W.judgment_identity_form(p, run_id=rid, judge_id=synth.JUDGE_ID_2) == ("producer", {"producer_session": "producer-x", "claimed_utc": cu})
    (payload,) = W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, "11111111-2222-4333-8444-555555555555", [p])
    assert set(payload) - {"extra"} == BASE_KEYS | {"producer_session", "claimed_utc"} and (payload["producer_session"], payload["claimed_utc"]) == ("producer-x", cu)
    assert I.judgment_id(rid, synth.JUDGE_ID_2, payload["producer_session"], payload["claimed_utc"]) == payload["judgment_id"] == prod_id
    scope, sup = _retire_through_scope(root, rid, "s-A2")
    assert [(x["judgment_id"], x.get("producer_session"), x.get("claimed_utc"), "claimed_by_session" in x) for x in sup] == [(prod_id, "producer-x", cu, False)]
    _fold(root)
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id, producer_session, claimed_utc FROM judgment WHERE judgment_id = ?",
                                  (prod_id,)).fetchone()) == ("superseded", scope.judgment_id, "producer-x", cu)


def test_a_partial_no_identity_form_reproduces_refuses_the_whole_batch_by_name_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    root, (rid, r2) = _root_with(tmp_path, [("k", "file:///stores/k"), ("k2", "file:///stores/k2")], monkeypatch)
    wl_a = W.RegistryWorklist(root, [SEED_ID], "s-A", driver=DRIVER, api_pool="lowprio")
    wl_a.grade(wl_a.pending(eval_category="static_attack"), _grade_partial(150))      # a claim-minted partial on EACH run
    _fold(root)
    partial_1, partial_2 = wl_a.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"], wl_a.claims[(r2, synth.JUDGE_ID_2)]["judgment_id"]
    _strip_claim_inputs(root, partial_1)                                                 # run 1's partial: no form reproduces it any more
    with DB.reading(DB.db_path(root)) as conn:
        (p1,) = W.own_partial_judgments(conn, rid, synth.JUDGE_ID_2)
        (p2,) = W.own_partial_judgments(conn, r2, synth.JUDGE_ID_2)
    assert (p1["claimed_by_session"], p1["claimed_utc"], p1["producer_session"]) == (None, None, "s-A") and p2["claimed_utc"]
    # (1) the form finder names what every form lacked; the builder refuses by name for the WHOLE batch (a good partial beside it builds nothing)
    with pytest.raises(W.JudgmentIdentityUnreproducible) as ei:
        W.judgment_identity_form(p1, run_id=rid, judge_id=synth.JUDGE_ID_2)
    assert ei.value.judgment_id == partial_1 and ei.value.forms_tried == ["plain: does not reproduce the id", "claimed: claimed_by_session, claimed_utc not stated",
                                                                          "producer: claimed_utc not stated"]
    assert partial_1 in str(ei.value) and "D639" in str(ei.value)
    with pytest.raises(W.JudgmentIdentityUnreproducible):
        W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, "11111111-2222-4333-8444-555555555555", [p2, p1])
    # a row stating every field, none of which minted its id, is the same refusal
    odd = dict(p2, judgment_id=I.judgment_id(rid, synth.JUDGE_ID_2, "someone", "2026-01-01T00:00:00.000Z"), producer_session="someone-else")
    with pytest.raises(W.JudgmentIdentityUnreproducible) as ei2:
        W.superseded_judgment_payloads(rid, synth.JUDGE_ID_2, "11111111-2222-4333-8444-555555555555", [odd])
    assert ei2.value.forms_tried == ["plain: does not reproduce the id", "claimed: does not reproduce the id", "producer: does not reproduce the id"]
    # (2) write_supersessions: refused before the writer exists -- no record file, no temp file
    before = set(_run_dir_files(root, rid))
    with pytest.raises(W.JudgmentIdentityUnreproducible):
        W.write_supersessions(root, campaign=synth.CAMPAIGN, run_id=rid, judge_id=synth.JUDGE_ID_2, complete_id="11111111-2222-4333-8444-555555555555",
                              partials=[p1], script="tests/tool.py", session="s-tool")
    assert set(_run_dir_files(root, rid)) == before
    # (3) the scope: refused at CONSTRUCTION, by name, before any grading or write -- the claim it took stands for the daemon's own-session reclaim
    wl2 = W.RegistryWorklist(root, [SEED_ID], "s-A2", driver=DRIVER, api_pool="lowprio")
    rows = {r["run_id"]: r for r in wl2.pending(eval_category="static_attack")}
    with pytest.raises(W.JudgmentIdentityUnreproducible, match=partial_1):
        wl2.scope(rows[rid])
    assert set(_run_dir_files(root, rid)) == before
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (partial_1,)).fetchone()[0] == "partial"
        claim = P.active_claim(conn, rid, synth.JUDGE_ID_2)
    assert claim and claim["claimed_by_session"] == "s-A2"
    # the other run's partial (claim-minted, reproducible) is superseded as before: the refusal is per (run, judge), never the worklist's
    scope2 = wl2.scope(rows[r2])
    with scope2:
        _grade_complete(200)(rows[r2], scope2)
    assert scope2.superseded == [partial_2]
    # (4) the reconciliation: a legacy complete (a plain hooks scope completing the claim the refused scope left standing -- the earlier shape,
    # no supersession) beside the irreproducible partial is REFUSED by name, nothing written, said on stderr; the entry names it under
    # `refused`; a second pass refuses again (nothing on the spool to make it idempotent-by-record); the good pair is untouched
    jid = claim["judgment_id"]
    legacy = hooks.judgment_scope(DRIVER, {"run_id": rid, "campaign": synth.CAMPAIGN}, {"judge_id": synth.JUDGE_ID_2}, spool=H.records_dir(root),
                                  registry_emit=True, session="s-A2", claim={"judgment_id": jid, "claimed_utc": claim["claimed_utc"]})
    with legacy:
        _grade_complete(200)(None, legacy)
    _fold(root)
    with DB.reading(DB.db_path(root)) as conn:
        cands = W.supersession_candidates(conn, model_ids=[synth.MODEL_ID])
    assert [(c["run_id"], c["complete_id"], [p["judgment_id"] for p in c["partials"]]) for c in cands] == [(rid, jid, [partial_1])]
    before = set(_run_dir_files(root, rid))
    capsys.readouterr()
    for _ in range(2):
        (d,) = W.reconcile_supersessions(root, model_ids=[synth.MODEL_ID], session="s-recon", script="tests/recon.py")
        assert d["refused"].startswith("JudgmentIdentityUnreproducible: ") and partial_1 in d["refused"]
        assert (d["run_id"], d["judge_id"], d["complete_id"], d["partial_ids"], d["path"], d["refused_partial_ids"]) == (rid, synth.JUDGE_ID_2, jid, [], None, [partial_1])
        assert set(_run_dir_files(root, rid)) == before
    assert "reconcile_supersessions: JudgmentIdentityUnreproducible" in capsys.readouterr().err
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (partial_1,)).fetchone()) == ("partial", None)
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (partial_2,)).fetchone()) == ("superseded", scope2.judgment_id)
