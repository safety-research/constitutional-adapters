"""The grading queue and claims: a second judge after ingestion is pending; a complete
judgment with full coverage is not; a partial one keeps the run pending with the missing count; failed /
cancelled runs are never pending; an active claim is listed and blocks a second claimant; release returns
the run to pending; a stale claim can be taken over with --force.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    H.ingest_once(root, workers=0, min_free=0)
    return root, ids


def test_second_judge_after_ingestion_is_pending(tmp_path):
    root, ids = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert {r["run_id"] for r in P.pending(conn, synth.JUDGE_ID_2)} == set(ids["run_ids"])
    assert P.pending(conn, synth.JUDGE_ID) == []
    assert P.pending(conn) == []   # default = instrument judge
    conn.close()


def test_complete_with_full_coverage_is_not_pending_partial_is(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated")
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [r["run_id"] for r in P.pending(conn)] == [rid]
    conn.close()
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-partial.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                       "n_items_judged": 150, "status": "partial"}, synth.t(200))])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = P.pending(conn)
    assert len(rows) == 1 and rows[0]["items_missing"] == 50
    conn.close()
    _write(root, "c", "000003-complete.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                        "n_items_judged": 200, "status": "complete"}, synth.t(300))])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert P.pending(conn) == []
    conn.close()


def test_failed_and_cancelled_never_pending(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    for k, st in enumerate(("failed", "cancelled")):
        rid, recs = synth.run_records(f"k{k}", 100 + 50 * k, through="running")
        recs.append(synth.rec("run_state", {"run_id": rid, "state": st, "event_utc": synth.t(100 + 50 * k + 9), "evidence_kind": "scheduler_state",
                                            "scheduler_state": st.upper()}, synth.t(100 + 50 * k + 9)))
        _write(root, "c", f"00000{k}-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment").fetchone()[0] == 0
    assert {r[0] for r in conn.execute("SELECT lifecycle_state FROM run")} == {"failed", "cancelled"}
    conn.close()


def test_active_claim_listed_and_blocks_second_claim_then_release(tmp_path):
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][0]
    jid = P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1000))
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid][0]
    assert row["claim_session"] == "session-A" and row["claim_utc"] == synth.t(1000)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "in_progress"
    # the ingested run stays ingested (a second judge's claim does not move a finished run to judging)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "ingested"
    conn.close()
    with pytest.raises(P.ClaimHeld) as ei:
        P.claim(root, rid, synth.JUDGE_ID_2, "session-B", now=synth.t(1001))
    assert ei.value.session == "session-A"
    # the same session re-claiming gets its own judgment id back
    assert P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1002)) == jid
    assert P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1003))
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "abandoned"
    row = [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid][0]
    assert row["claim_session"] is None
    conn.close()
    assert not P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1004))


def test_stale_claim_takeover_with_force(tmp_path):
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][1]
    old = P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now="2026-01-01T00:00:00.000Z")
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID_2, "session-B", now="2026-01-02T00:00:00.000Z")   # not forced
    new = P.claim(root, rid, synth.JUDGE_ID_2, "session-B", force=True, stale_hours=12, now="2026-01-02T00:00:00.000Z")
    assert new != old
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = {r[0]: r[1] for r in conn.execute("SELECT judgment_id, status FROM judgment WHERE run_id = ?", (rid,))}
    assert st[old] == "abandoned" and st[new] == "in_progress"
    conn.close()
    # a fresh claim cannot be forced over
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID_2, "session-C", force=True, now="2026-01-02T00:10:00.000Z")


def test_claim_on_generated_run_moves_it_to_judging(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated")
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    P.claim(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(500))
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "judging"
    conn.close()
    P.release(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(600))
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "generated"
    conn.close()


# --------------------------------------------------------------------------------------------
# PR-5a: eligibility per requested judge is EXPLICIT (pending_reason), coverage is the item set's, ingested
# runs are included, an abandoned claim leaves the run pending again, an active claim is shown with its age
# --------------------------------------------------------------------------------------------


def _generated_root(tmp_path, n_items=200):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated", n_items=n_items)
    _write(root, "c", "000001-run.jsonl", recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    return root, rid


def _row(root, judge, rid):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows = [r for r in P.pending_for(conn, judge) if r["run_id"] == rid]
        return rows[0] if rows else None
    finally:
        conn.close()


def test_no_judgment_is_pending_for_every_judge_with_reason_and_coverage(tmp_path):
    root, rid = _generated_root(tmp_path)
    a, b = _row(root, synth.JUDGE_ID, rid), _row(root, synth.JUDGE_ID_2, rid)
    assert a["pending_reason"] == b["pending_reason"] == "no_judgment"
    assert a["is_instrument_judge"] == 1 and b["is_instrument_judge"] == 0
    assert a["required_coverage"] == 200 and a["items_missing"] is None and a["n_abandoned"] == 0
    assert a["eval_category"] == "static_attack" and a["harness"] == "bespoke_sisteer" and a["decode_source"] == "greedy"
    assert a["lifecycle_state"] == "generated" and a["store_uri"] == "synthetic:///store/k"


def test_judged_by_another_judge_is_pending_for_the_requested_one_only(tmp_path):
    root, rid = _generated_root(tmp_path)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-j.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                 "n_items_judged": 200, "status": "complete"}, synth.t(200)),
                                         synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(201),
                                                                 "evidence_kind": "completion_artifact"}, synth.t(201))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _row(root, synth.JUDGE_ID, rid) is None
    b = _row(root, synth.JUDGE_ID_2, rid)
    assert b["pending_reason"] == "awaiting_second_judge" and b["n_complete_other_judges"] == 1
    assert b["lifecycle_state"] in ("judged", "ingested")   # the harvester may already have derived ingested
    assert json.loads(b["existing_judgments_json"])[0]["judge_id"] == synth.JUDGE_ID


def test_abandoned_claim_leaves_the_run_pending_with_the_count(tmp_path):
    root, rid = _generated_root(tmp_path)
    P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(300))
    row = _row(root, synth.JUDGE_ID, rid)
    assert row["pending_reason"] == "claimed" and row["claim_session"] == "s-A" and row["claim_utc"] == synth.t(300)
    assert row["claim_age_s"] is not None and row["claim_age_s"] > 0 and row["claim_judgment_id"]
    assert P.release(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(301))
    row = _row(root, synth.JUDGE_ID, rid)
    # last_abandoned_utc is the abandoned claim's own claimed_utc (its claim time): the release record re-states the claim's
    # claimed_utc rather than the release time, so its id reproduces under the claim form of the identity rule; the release time is the
    # judgment row's status stamp
    assert row["pending_reason"] == "abandoned" and row["n_abandoned"] == 1 and row["last_abandoned_utc"] == synth.t(300)
    assert row["claim_session"] is None and row["lifecycle_state"] == "generated"
    # a second claim by another session now succeeds (the run is pending again)
    P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(302))
    assert _row(root, synth.JUDGE_ID, rid)["claim_session"] == "s-B"


def test_partial_and_short_complete_judgments_keep_the_run_pending(tmp_path):
    root, rid = _generated_root(tmp_path)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-partial.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                       "n_items_judged": 120, "status": "partial"}, synth.t(200))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    row = _row(root, synth.JUDGE_ID, rid)
    assert row["pending_reason"] == "partial" and row["items_missing"] == 80
    _write(root, "c", "000003-short.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                     "n_items_judged": 150, "status": "complete"}, synth.t(300))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    row = _row(root, synth.JUDGE_ID, rid)
    assert row["pending_reason"] == "coverage_short" and row["items_missing"] == 50


def test_required_coverage_is_capped_by_the_declared_item_set(tmp_path):
    # the run reports 450 generated items; the instrument's item set declares 200 -> coverage 200
    root, rid = _generated_root(tmp_path, n_items=450)
    row = _row(root, synth.JUDGE_ID, rid)
    assert row["n_items_generated"] == 450 and row["required_coverage"] == 200
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-c.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                 "n_items_judged": 200, "status": "complete"}, synth.t(200))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _row(root, synth.JUDGE_ID, rid) is None


def test_ingested_runs_are_pending_for_a_second_judge_and_the_status_page_count_agrees(tmp_path):
    root, ids = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    states = {r[0] for r in conn.execute("SELECT lifecycle_state FROM run")}
    assert states == {"ingested"}
    rows = P.pending_for(conn, synth.JUDGE_ID_2)
    assert {r["run_id"] for r in rows} == set(ids["run_ids"]) and {r["pending_reason"] for r in rows} == {"awaiting_second_judge"}
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE is_instrument_judge = 1").fetchone()[0] == 0
    conn.close()


def test_judge_refusals_count_toward_coverage(tmp_path):
    """A judge refusal is an outcome the judgment records, not a gap: 180 judged + 20 declined cover 200."""
    root, rid = _generated_root(tmp_path)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-c.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                 "n_items_judged": 180, "n_judge_refusals": 15, "status": "complete"}, synth.t(200))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    row = _row(root, synth.JUDGE_ID, rid)
    assert row["pending_reason"] == "coverage_short" and row["items_missing"] == 5
    assert json.loads(row["existing_judgments_json"])[0]["n_judge_refusals"] == 15
    _write(root, "c", "000003-c.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                 "n_items_judged": 180, "n_judge_refusals": 20, "status": "complete"}, synth.t(300))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _row(root, synth.JUDGE_ID, rid) is None


def test_judge_errors_count_toward_coverage(tmp_path):
    """An item whose judging errored is an outcome the judgment records (no verdict), not a gap a re-grade would fill: 180 judged +
    15 declined + 3 errored leave 2 missing; with 5 errored they cover 200 and the run leaves the queue -- `items_missing` and the
    harvester's completion read the one fragment (`schema_def.judgment_covered_items_sql`)."""
    root, rid = _generated_root(tmp_path)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-c.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": 180,
                                                                 "n_judge_refusals": 15, "n_errors": 3, "status": "complete"}, synth.t(200))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    row = _row(root, synth.JUDGE_ID, rid)
    assert row["pending_reason"] == "coverage_short" and row["items_missing"] == 2
    assert json.loads(row["existing_judgments_json"])[0]["n_errors"] == 3
    _write(root, "c", "000003-c.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": 180,
                                                                 "n_judge_refusals": 15, "n_errors": 5, "status": "complete"}, synth.t(300))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _row(root, synth.JUDGE_ID, rid) is None


# --------------------------------------------------------------------------------------------
# the judge join is scoped by the instrument: a rule-scored run (no judge) is pending for nobody
# --------------------------------------------------------------------------------------------


def test_rule_scored_run_is_pending_for_no_judge_and_an_llm_scored_run_for_its_instrument_judge(tmp_path):
    """An instrument without a judge (rule-scored) has nobody to wait for: its runs take no row, however many judges the
    table holds (the bare cross join listed each once per judge). An instrument that names a judge keeps the designed
    run x judge pairing: the default worklist offers the run to its instrument's judge, once, and a second judge sees it
    before filing anything (the --judge worklist). A judge that CLAIMS the rule-scored run is then listed for it alone."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid_llm, recs = synth.run_records("k", 100, through="generated")
    rid_rule, recs_rule = synth.rule_scored_run_records("cap", 200, n_scored=150)     # short of coverage: stays generated
    _write(root, "c", "000001-run.jsonl", recs + recs_rule)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM judge").fetchone()[0] == 2
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_rule,)).fetchone()[0] == "generated"
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id = ?", (rid_rule,)).fetchone()[0] == 0
    rows = conn.execute("SELECT required_judge_id, pending_reason FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1",
                        (rid_llm,)).fetchall()
    assert [tuple(r) for r in rows] == [(synth.JUDGE_ID, "no_judgment")]
    assert [r["run_id"] for r in P.pending(conn)] == [rid_llm]
    assert {r["run_id"] for r in P.pending_for(conn, synth.JUDGE_ID_2)} == {rid_llm}
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE is_instrument_judge = 1").fetchone()[0] == 1
    conn.close()
    P.claim(root, rid_rule, synth.JUDGE_ID_2, "s-A", now=synth.t(300))
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = conn.execute("SELECT required_judge_id, pending_reason, is_instrument_judge FROM v_pending_judgment WHERE run_id = ?",
                        (rid_rule,)).fetchall()
    assert [tuple(r) for r in rows] == [(synth.JUDGE_ID_2, "claimed", 0)]
    conn.close()


def test_the_worklist_skips_a_run_whose_store_is_already_judged_on_a_superseded_sibling_but_not_before_the_supersession(tmp_path, capsys):
    """The correction re-registers a unit under a rung-keyed run id over the SAME store; once the old run is superseded
    (run.superseded_by_run_id names the new one) and carries a complete judgment by the instrument judge, the new run is skipped by the
    shared worklist with a line naming the sibling and its judgment -- the re-filed judgment fills it, never a re-judge. Before the
    supersession the new run is pending as any other (that window is the daemon's dispatch hold); the view itself still lists it (the skip
    is the worklist's); a run over another store is untouched."""
    from sisteer.registry import worklist as W
    store = "synthetic:///store/shared-static-unit"
    for superseded in (True, False):
        root = str(tmp_path / ("reg_sup" if superseded else "reg_pre"))
        H.ensure_layout(root)
        _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
        rid_new, recs_new = synth.run_records("k_cap4096", 300, through="generated", store_uri=store)
        rid_old, recs_old = synth.run_records("k_old", 100, through="judged", store_uri=store, **({"superseded_by_run_id": rid_new} if superseded else {}))
        rid_other, recs_other = synth.run_records("k_other", 500, through="generated")
        rid_c, recs_c = synth.run_records("k_not_designated", 700, through="generated", store_uri=store)      # same store + eval, NOT the designated successor
        _write(root, "c", "000001-old.jsonl", recs_old); _write(root, "c", "000002-new.jsonl", recs_new); _write(root, "c", "000003-other.jsonl", recs_other)
        _write(root, "c", "000004-c.jsonl", recs_c)
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
        conn = DB.connect(DB.db_path(root), readonly=True)
        assert conn.execute("SELECT superseded_by_run_id FROM run WHERE run_id = ?", (rid_old,)).fetchone()[0] == (rid_new if superseded else None)
        assert sorted(r["run_id"] for r in P.pending(conn)) == sorted([rid_new, rid_other, rid_c])    # the view lists them all either way
        conn.close()
        capsys.readouterr()
        rows = W.RegistryWorklist(root, None, "s-test", None, driver="synthetic").pending()
        err = capsys.readouterr().err
        if superseded:
            # only the DESIGNATED successor is skipped: run C shares the store and eval but is not what the old run was superseded by
            assert sorted(r["run_id"] for r in rows) == sorted([rid_other, rid_c]), rows
            assert f"run {rid_new} skipped" in err and rid_old in err and "superseded sibling" in err and I.judgment_id(rid_old, synth.JUDGE_ID) in err
            assert f"run {rid_c} skipped" not in err
        else:
            assert sorted(r["run_id"] for r in rows) == sorted([rid_new, rid_other, rid_c]) and "skipped" not in err


def test_an_identity_stale_complete_judgment_on_the_superseded_sibling_cannot_suppress_its_successor_while_a_current_one_still_does(tmp_path, capsys):
    """The superseded-sibling skip read ANY complete judgment by the judge on the old run, so a judgment CLAIMED under instrument X
    (claimed_instrument_id / claimed_required_judge_id = X / J, as pending.claim binds it) on a sibling since corrected to instrument X2
    (the same-id `run` record moving the instrument) -- stale-bound, covering nothing anywhere -- hid the designated successor with no
    refusal. Now the judgment must be identity-current under the OLD run: (b) before the correction the current-bound judgment suppresses
    the successor exactly as before (the line names the sibling and its judgment); (a) after the correction the successor is listed and no
    skip line names it; (c) a NULL-bound (legacy) judgment reads current and still suppresses through the same correction. The REAL view
    and worklist, no stubs.

    Re-pinned to the 0025 view (the registry owner's dd109e676, migration 0025_pending_excludes_superseded: a superseded run is never
    pending, its successor is, and `pending.claim` of a superseded run refuses ClaimNotPending by name). The first build claimed the old
    run AFTER it was superseded and read the view's `identity_stale` listing of the retired run after the correction; under 0025 the claim
    refuses and the listing is gone. So the claim binds {X, J} while the old run is STILL pending, the supersession lands afterwards as the
    owner's same-id `run` re-emission (test_registry_pending_excludes_superseded's shape; identity restated, so the judgment stays current),
    and the retired run is asserted ABSENT from the view throughout -- never `identity_stale`. The skip itself reads the run and judgment
    TABLES, not the view, so the sibling rule it pins is the same under 0025."""
    from sisteer.registry import schema_def as S
    from sisteer.registry import worklist as W
    store = "synthetic:///store/shared-static-unit"
    inst_x2 = synth.instrument_fields(token_cap_tokens=1024, token_cap_ladder_json="[1024]", notes="the same-judge corrected instrument")   # judge J, another cap
    x2 = str(inst_x2["instrument_id"]); assert x2 != synth.INSTRUMENT_ID and inst_x2["judge_id"] == synth.JUDGE_ID
    dims = synth.seed_dimensions() + [synth.rec("instrument", inst_x2, synth.t(31))]                 # X2 registered up front (the owner's extra_dims)

    def _reemit_old_run(root, rid_old, rid_new, seq, **over):
        """The owner's same-id `run` re-emission of the OLD run: the store, the eval and the supersession restated (the filter keys on them)
        plus ``over`` -- nothing more (the supersession alone, identity-neutral), or ``instrument_id = X2`` (the identity correction)."""
        corrected = synth.run_payload("k_old", i=100, store_uri=store, superseded_by_run_id=rid_new, **over)
        corrected["run_id"] = rid_old
        _write(root, "c", f"{seq:06d}-correction.jsonl", [synth.rec("run", corrected, synth.t(seq))])
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()

    def _run_row(root, rid):
        conn = DB.connect(DB.db_path(root), readonly=True)
        try:
            return tuple(conn.execute("SELECT instrument_id, superseded_by_run_id, store_uri FROM run WHERE run_id = ?", (rid,)).fetchone())
        finally:
            conn.close()

    def _view(root):
        conn = DB.connect(DB.db_path(root), readonly=True)
        try:
            return {r["run_id"]: r["pending_reason"] for r in P.pending(conn)}
        finally:
            conn.close()

    def _judgment_current(root, jid):
        """schema_def.judgment_identity_current_sql over the judgment's run and ITS instrument -- the skip's own conjunct, read directly."""
        conn = DB.connect(DB.db_path(root), readonly=True)
        try:
            return conn.execute(f"SELECT {S.judgment_identity_current_sql('j', 'r', 'i')} FROM judgment j JOIN run r ON r.run_id = j.run_id "
                                "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id WHERE j.judgment_id = ?", (jid,)).fetchone()[0]
        finally:
            conn.close()

    def _worklist(root):
        capsys.readouterr()
        rows = W.RegistryWorklist(root, None, "s-test", None, driver="synthetic").pending()
        return {str(r["run_id"]): r for r in rows}, capsys.readouterr().err

    # ---- (b) then (a): the old run's complete judgment CLAIMED under X while the old run is STILL pending, THEN superseded by rid_new
    root = str(tmp_path / "reg_bound")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", dims)
    rid_new, recs_new = synth.run_records("k_cap4096", 300, through="generated", store_uri=store)
    rid_old, recs_old = synth.run_records("k_old", 100, through="generated", store_uri=store)            # no successor yet: pending as any other
    rid_other, recs_other = synth.run_records("k_other", 500, through="generated")
    _write(root, "c", "000001-old.jsonl", recs_old); _write(root, "c", "000002-new.jsonl", recs_new); _write(root, "c", "000003-other.jsonl", recs_other)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert sorted(_view(root)) == sorted([rid_old, rid_new, rid_other])
    jid = P.claim(root, rid_old, synth.JUDGE_ID, "s-A", now=synth.t(1000))                       # the claim binds {X, J} -- the live identity
    _write(root, "c", "000004-old-complete.jsonl", [
        synth.rec("judgment", {"judgment_id": jid, "run_id": rid_old, "judge_id": synth.JUDGE_ID, "status": "complete", "n_items_judged": 200,
                               "started_utc": synth.t(1000), "finished_utc": synth.t(1001), "claimed_by_session": "s-A", "claimed_utc": synth.t(1000)}, synth.t(1001)),
        synth.rec("run_state", {"run_id": rid_old, "state": "judged", "event_utc": synth.t(1002), "evidence_kind": "completion_artifact"}, synth.t(1002)),
        synth.rec("run_patch", {"run_id": rid_old, "fields": {"n_items_judged": 200, "judge_id": synth.JUDGE_ID}}, synth.t(1003))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT status, claimed_instrument_id, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()) == ("complete", synth.INSTRUMENT_ID, synth.JUDGE_ID)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_old,)).fetchone()[0] in ("judged", "ingested")   # covered: the harvester may promote it to ingested
    conn.close()
    assert _run_row(root, rid_old) == (synth.INSTRUMENT_ID, None, store) and _judgment_current(root, jid) == 1
    assert sorted(_view(root)) == sorted([rid_new, rid_other])                                       # the old run is covered, not listed; no supersession yet
    rows, err = _worklist(root)
    assert sorted(rows) == sorted([rid_new, rid_other]) and "skipped" not in err                   # the pre-supersession window: nothing skipped
    # the supersession alone (identity-neutral, X restated): the retired run leaves the view outright (0025) and a claim of it refuses by name
    _reemit_old_run(root, rid_old, rid_new, 2000)
    assert _run_row(root, rid_old) == (synth.INSTRUMENT_ID, rid_new, store) and _judgment_current(root, jid) == 1
    reasons = _view(root)
    assert rid_old not in reasons and reasons[rid_new] == "no_judgment" and reasons[rid_other] == "no_judgment", reasons
    with pytest.raises(P.ClaimNotPending, match="superseded by run"):
        P.claim(root, rid_old, synth.JUDGE_ID, "s-B", now=synth.t(2100))
    # (b) identity-CURRENT (claimed under X, the old run still on X): the successor is suppressed exactly as before
    rows, err = _worklist(root)
    assert sorted(rows) == [rid_other], rows
    assert f"run {rid_new} skipped" in err and rid_old in err and "superseded sibling" in err and jid in err
    # (a) the correction lands: the old run moves to X2, the judgment it carries is stale-bound (claimed X, run X2) -- the successor is LISTED and
    # no skip line names it (before this fix the stale judgment hid it with no refusal); the retired run stays out of the view (0025: never
    # identity_stale, the listing the first build read)
    _reemit_old_run(root, rid_old, rid_new, 3000, instrument_id=x2)
    assert _run_row(root, rid_old) == (x2, rid_new, store) and _judgment_current(root, jid) == 0
    reasons = _view(root)
    assert rid_old not in reasons and reasons[rid_new] == "no_judgment" and reasons[rid_other] == "no_judgment", reasons
    rows, err = _worklist(root)
    assert rid_new in rows and rid_other in rows, sorted(rows)
    assert f"run {rid_new} skipped" not in err and "superseded sibling" not in err, err
    assert set(rows) == set(reasons)                                                                # the worklist's rows are the 0025 view's, nothing skipped
    # ---- (c) a NULL-bound (legacy) judgment -- run_records' `judged` shape, no claim binding -- reads current and still suppresses after the same correction
    root2 = str(tmp_path / "reg_legacy")
    H.ensure_layout(root2)
    _write(root2, "dims", "000001-dims.jsonl", dims)
    rid_new2, recs_new2 = synth.run_records("k_cap4096", 300, through="generated", store_uri=store)
    rid_old2, recs_old2 = synth.run_records("k_old", 100, through="judged", store_uri=store, superseded_by_run_id=rid_new2)
    _write(root2, "c", "000001-old.jsonl", recs_old2); _write(root2, "c", "000002-new.jsonl", recs_new2)
    assert H.ingest_once(root2, workers=0, min_free=0).refused == 0
    _reemit_old_run(root2, rid_old2, rid_new2, 3000, instrument_id=x2)
    jid2 = I.judgment_id(rid_old2, synth.JUDGE_ID)
    conn = DB.connect(DB.db_path(root2), readonly=True)
    assert conn.execute("SELECT claimed_instrument_id FROM judgment WHERE judgment_id = ?", (jid2,)).fetchone()[0] is None
    conn.close()
    assert _run_row(root2, rid_old2) == (x2, rid_new2, store) and _judgment_current(root2, jid2) == 1
    assert sorted(_view(root2)) == [rid_new2]                                                        # the retired run is out of the view here too (0025)
    rows, err = _worklist(root2)
    assert rid_new2 not in rows and f"run {rid_new2} skipped" in err and "superseded sibling" in err and jid2 in err
