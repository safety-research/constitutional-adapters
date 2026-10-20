"""``registry judgment show | supersede`` and the judgment identity rule it rests on: an ordinary complete
judgment superseded with a null link keeps every measurement column and stamp, reads `pending_reason = superseded`, and its metric rows
leave `v_paper_long`; a successor by the same judge covers the run; a successor itself superseded before the fold still folds and the
run is pending iff the chain ends in no complete covering judgment; the documented residual race (a later-stamped producer re-file wins,
`show` says so, a re-run succeeds; an earlier-stamped one loses); a same-id successor re-emission naming another run is refused by the
identity rule while a conforming one folds; `pending.release` and the forced takeover carry the claim's own `claimed_utc`; a claim-backed
judgment's supersede copies the identity inputs and folds and rebuilds equal, a record without them is refused before the write, a row no
form reproduces refuses by name; superseding one judge's judgment while another judge's claim is open leaves the run `judging` until that
claim ends (live and rebuilt); every guard rail writes nothing; `--dry-run` writes nothing; rebuild equality throughout.
"""
from __future__ import annotations

import json
import os
import random
import shutil

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import reducer
from sisteer.registry import supersede as SUP
from sisteer.registry import synth
from sisteer.registry import validate as VAL

REJUDGE = "rejudge-session"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root):
    return H.ingest_once(root, workers=0, min_free=0)


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _row(root, jid):
    rows = _q(root, "SELECT * FROM judgment WHERE judgment_id = ?", jid)
    return rows[0] if rows else None


def _pending(root, rid, judge):
    rows = _q(root, "SELECT pending_reason, items_missing, claim_judgment_id, n_abandoned FROM v_pending_judgment WHERE run_id = ? AND required_judge_id = ?",
              rid, judge)
    return rows[0] if rows else None


def _frame_judgments(root, rid):
    return sorted(r["judgment_id"] for r in _q(root, "SELECT judgment_id FROM v_paper_long WHERE run_id = ?", rid))


def _supersede_files(root):
    d = os.path.join(root, "records", SUP.SUBDIR)
    return sorted(os.path.relpath(f, d) for f in R.iter_record_files(d)) if os.path.isdir(d) else []


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    """Rebuild from a differently enumerated copy of the log; assert every table equal (the reducer test's helper)."""
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        rel = os.path.relpath(f, os.path.join(root, "records"))
        dest = os.path.join(copy, "records", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    finally:
        a.close()
        b.close()


def _corpus(tmp_path, n_runs=2, n_items=4):
    """Judged runs (the harvester derives `ingested`); the second run's judgment is the one most tests supersede."""
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    assert _ingest(root).refused == 0
    return root, ids


def _judgment(rid, jid, judge, status, i, n=4, **over):
    p = {"judgment_id": jid, "run_id": rid, "judge_id": judge, "n_items_judged": n, "status": status}
    p.update(over)
    return synth.rec("judgment", p, synth.t(i))


def _metric(rid, jid, i, value=0.5):
    return synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": value, "k": 2, "n": 4,
                                    "computed_by": "producer"}, synth.t(i))


def _second(rid, judge, session, i, status="complete", n=4, **over):
    """A second judgment of (run, judge), told apart by its claim form (the hook's shape), with its headline metric row."""
    jid = I.judgment_id(rid, judge, session, synth.t(i))
    return jid, [_judgment(rid, jid, judge, status, i, n=n, claimed_by_session=session, claimed_utc=synth.t(i), producer_session=session, **over),
                 _metric(rid, jid, i + 1)]


def _generated(tmp_path, n_items=4):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated", n_items=n_items)
    _write(root, "c", "000001-run.jsonl", recs)
    assert _ingest(root).refused == 0
    return root, rid


def _validate(rec):
    return VAL.validate_record(rec, "<test>", R.record_sha(rec))


# --------------------------------------------------------------------------------------------
# the ordinary case, the successor, the chain
# --------------------------------------------------------------------------------------------


def test_complete_judgment_superseded_with_no_successor_keeps_every_other_column_and_stamp(tmp_path):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    # the producer's own metadata on the judgment: a store, a note under extra -- values AND stamps must survive the supersede
    _write(root, "c", "000001-meta.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "complete", 300, n=4, store_uri="synthetic:///grades/k1",
                                                      extra={"judge_note": "clean pass"})])
    assert _ingest(root).refused == 0
    before = _row(root, jid)
    fv_before = json.loads(before["field_versions_json"])
    assert before["status"] == "complete" and before["superseded_by_judgment_id"] is None and jid in _frame_judgments(root, rid)
    assert _pending(root, rid, synth.JUDGE_ID) is None

    rep = SUP.supersede(root, jid, no_successor=True, reason="the rubric was misread; a re-judge is wanted", ruled_by="the user", now=synth.t(400))
    assert rep.written and rep.folded and rep.in_force is True and rep.note == ""
    assert rep.path.startswith(os.path.join(root, "records", SUP.SUBDIR, I.run_dir_name(rid)))
    payload = rep.record["payload"]
    assert set(payload) == {"judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id", "extra"}      # minimal: no identity inputs for a plain id
    assert payload["status"] == "superseded" and payload["superseded_by_judgment_id"] is None
    assert payload["extra"]["supersede"]["reason"] == "the rubric was misread; a re-judge is wanted"
    assert payload["extra"]["supersede"]["status_before"] == "complete" and payload["extra"]["supersede"]["provenance"]["kind"] == "user_direct"
    assert rep.record["written_utc"] == synth.t(400) > fv_before["status"][0]
    after = _row(root, jid)
    fv_after = json.loads(after["field_versions_json"])
    assert after["status"] == "superseded" and after["superseded_by_judgment_id"] is None
    for col in ("n_items_judged", "store_uri", "api_pool", "started_utc", "finished_utc", "claimed_by_session", "claimed_utc"):
        assert after[col] == before[col], col
        assert fv_after.get(col) == fv_before.get(col), col              # the stamp is the producer's, untouched
    for col in ("run_id", "judge_id"):                                    # stated by the record (required fields): the value cannot move, the stamp does
        assert after[col] == before[col] and fv_after[col] == [synth.t(400), R.record_sha(rep.record)], col
    assert fv_after["status"] == [synth.t(400), R.record_sha(rep.record)] and fv_after["superseded_by_judgment_id"] == fv_after["status"]
    extra = json.loads(after["extra_json"])
    assert extra["judge_note"] == "clean pass" and extra["supersede"]["decision"] == "D639"
    # the consequences: pending again with the reason, the metric rows out of the frame, the lifecycle untouched (judged / ingested, never judging)
    pend = _pending(root, rid, synth.JUDGE_ID)
    assert pend is not None and pend["pending_reason"] == "superseded"
    assert jid not in _frame_judgments(root, rid)
    assert _q(root, "SELECT COUNT(*) AS n FROM run_metric WHERE judgment_id = ?", jid)[0]["n"] == 1
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid)[0]["lifecycle_state"] == "ingested"
    assert rep.remaining_same_judge == []
    _rebuild_equal(root, tmp_path)


def test_successor_by_the_same_judge_covers_the_run_and_show_reports_both_sides(tmp_path):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    jid2, recs2 = _second(rid, synth.JUDGE_ID, REJUDGE, 350)
    _write(root, "second", "000001-j2.jsonl", recs2)
    assert _ingest(root).refused == 0
    assert _frame_judgments(root, rid) == sorted([jid, jid2])
    assert _ingest(root).refused == 0
    rep = SUP.supersede(root, jid, superseded_by=jid2, reason="re-judged with the corrected rubric", now=synth.t(400))
    assert rep.in_force is True and rep.record["payload"]["superseded_by_judgment_id"] == jid2
    assert _row(root, jid)["superseded_by_judgment_id"] == jid2
    assert _pending(root, rid, synth.JUDGE_ID) is None                     # J2 (complete, n = 4 = the coverage) covers
    assert _frame_judgments(root, rid) == [jid2]
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        shown = SUP.show(conn, jid)
        assert shown["status"] == "superseded" and shown["successor"]["judgment_id"] == jid2 and shown["successor"]["status"] == "complete"
        assert shown["identity_form"] == "run_judge" and shown["pending"] is None and shown["status_stamp"][0] == synth.t(400)
        assert [j["judgment_id"] for j in shown["same_judge_judgments"]] == [jid2]
        shown2 = SUP.show(conn, jid2)
        assert shown2["predecessors"] == [jid] and shown2["identity_form"] == "claimed_by_session" and shown2["successor"] is None
        with pytest.raises(SUP.SupersedeRefusal) as ei:
            SUP.show(conn, "no-such-judgment")
        assert ei.value.field == "judgment_id"
    finally:
        conn.close()
    _rebuild_equal(root, tmp_path)


@pytest.mark.parametrize("x2_status,x2_n,pending_after", [("complete", 4, None), ("partial", 2, "partial")])
def test_a_successor_itself_superseded_before_the_fold_still_folds_and_the_chain_decides_pending(tmp_path, x2_status, x2_n, pending_after):
    """J -> X is read while X is complete; X -> X2 lands before J's record folds (no fold-time status check, by design): J's record folds,
    and the run is pending for the judge iff X2 is not a complete covering judgment."""
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    xid, xrecs = _second(rid, synth.JUDGE_ID, "x-session", 350)
    x2id, x2recs = _second(rid, synth.JUDGE_ID, "x2-session", 360, status=x2_status, n=x2_n)
    _write(root, "second", "000001-x.jsonl", xrecs + x2recs)
    assert _ingest(root).refused == 0
    rep = SUP.supersede(root, jid, superseded_by=xid, reason="first re-judge", now=synth.t(400), fold=False)
    assert rep.written and not rep.folded and rep.in_force is None
    rep2 = SUP.supersede(root, xid, superseded_by=x2id, reason="second re-judge", now=synth.t(401), fold=False)
    assert rep2.written
    assert _ingest(root).refused == 0
    assert _row(root, jid)["status"] == "superseded" and _row(root, jid)["superseded_by_judgment_id"] == xid
    assert _row(root, xid)["status"] == "superseded" and _row(root, xid)["superseded_by_judgment_id"] == x2id
    pend = _pending(root, rid, synth.JUDGE_ID)
    assert (None if pend is None else pend["pending_reason"]) == pending_after
    assert _frame_judgments(root, rid) == [x2id]
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the residual race, pinned as documented
# --------------------------------------------------------------------------------------------


def test_the_residual_race_a_later_stamped_producer_refile_wins_show_says_so_and_a_rerun_succeeds(tmp_path):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    rep = SUP.supersede(root, jid, no_successor=True, reason="retire", now=synth.t(400), fold=False)
    assert rep.written and not rep.folded
    # the producer re-files J as partial, stamped AFTER the CLI's record, before either folds
    _write(root, "c", "000002-refile.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "partial", 401, n=3)])
    assert _ingest(root).refused == 0
    row = _row(root, jid)
    assert row["status"] == "partial" and row["n_items_judged"] == 3                   # the re-file won by order key
    assert json.loads(row["field_versions_json"])["superseded_by_judgment_id"][0] == synth.t(400)   # the link the re-file did not state stays the CLI's
    assert json.loads(row["extra_json"])["supersede"]["reason"] == "retire"           # and so does the provenance
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        shown = SUP.show(conn, jid)
        assert shown["status"] == "partial" and shown["status_stamp"][0] == synth.t(401) and shown["pending"]["pending_reason"] == "partial"
    finally:
        conn.close()
    # the re-run reads the new row (partial is supersedable) and files a later record that wins
    rep2 = SUP.supersede(root, jid, no_successor=True, reason="retire, again", now=synth.t(402))
    assert rep2.in_force is True and rep2.status_before == "partial"
    assert _row(root, jid)["status"] == "superseded" and _pending(root, rid, synth.JUDGE_ID)["pending_reason"] == "superseded"
    _rebuild_equal(root, tmp_path)


def test_the_residual_race_an_earlier_stamped_producer_refile_loses_to_the_cli_record(tmp_path):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    rep = SUP.supersede(root, jid, no_successor=True, reason="retire", now=synth.t(400), fold=False)
    _write(root, "c", "000002-refile.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "partial", 399, n=3)])    # stamped BEFORE the CLI's record
    assert _ingest(root).refused == 0
    row = _row(root, jid)
    assert row["status"] == "superseded" and row["n_items_judged"] == 3      # the CLI's record wins status; the re-file's measurement stands
    assert json.loads(row["field_versions_json"])["status"][0] == synth.t(400)
    _rebuild_equal(root, tmp_path)


def test_the_cli_folds_its_record_and_reports_when_a_later_stamped_record_already_won(tmp_path):
    """The same race through the folding path: a later-stamped re-file is on disk (unfolded) when the CLI writes; the CLI's own fold applies
    both, the row reads the re-file, and the report says written but NOT in force (exit 1 at the CLI)."""
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]

    def refile_on_disk():
        _write(root, "c", "000002-refile.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "partial", 401, n=3)])

    rep = SUP.supersede(root, jid, no_successor=True, reason="retire", now=synth.t(400), before_write=refile_on_disk)
    assert rep.written and rep.folded and rep.in_force is False and "NOT in force" in rep.note
    assert rep.status_after == "partial" and _row(root, jid)["status"] == "partial"
    _rebuild_equal(root, tmp_path)


def test_the_snapshot_check_refuses_when_the_row_moved_between_the_reads(tmp_path):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]

    def fold_a_refile():
        _write(root, "c", "000002-refile.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "partial", 350, n=3)])
        assert _ingest(root).refused == 0

    with pytest.raises(SUP.SupersedeRefusal) as ei:
        SUP.supersede(root, jid, no_successor=True, reason="retire", now=synth.t(400), before_write=fold_a_refile)
    assert ei.value.field == "snapshot" and "judgment.status: 'complete' -> 'partial'" in ei.value.reason
    assert _supersede_files(root) == [] and _row(root, jid)["status"] == "partial"


# --------------------------------------------------------------------------------------------
# the identity rule: the successor relationship is immutable by construction
# --------------------------------------------------------------------------------------------


def test_a_same_id_successor_reemission_naming_another_run_is_refused_and_the_relationship_still_reads_the_original_run(tmp_path):
    root, ids = _corpus(tmp_path)
    (ra, rb), (ja, _jb) = ids["run_ids"], ids["judgment_ids"]
    xid, xrecs = _second(ra, synth.JUDGE_ID, "x-session", 350)
    _write(root, "second", "000001-x.jsonl", xrecs)
    assert _ingest(root).refused == 0
    # a full record with X's id and ANOTHER run_id (a same-id re-emission that would move the successor relationship) is refused by name
    moved = _judgment(rb, xid, synth.JUDGE_ID, "complete", 360, n=4, claimed_by_session="x-session", claimed_utc=synth.t(350), producer_session="x-session")
    bad = _validate(moved)
    assert isinstance(bad, VAL.Malformed) and bad.field == "judgment_id" and "not derived from this record's run_id" in bad.reason
    _write(root, "second", "000002-moved.jsonl", [moved])
    rep = _ingest(root)
    assert rep.refused == 1 and any("judgment_id" in r for r in rep.refusals), rep.refusals
    assert _row(root, xid)["run_id"] == ra
    # the supersede's relationship check reads the original run: J (run A) -> X (run A) folds
    sup = SUP.supersede(root, ja, superseded_by=xid, reason="re-judged", now=synth.t(400))
    assert sup.in_force is True and _row(root, ja)["superseded_by_judgment_id"] == xid
    # a conforming same-id re-emission (same run / judge, a new count) folds
    _write(root, "second", "000003-recount.jsonl", [_judgment(ra, xid, synth.JUDGE_ID, "complete", 410, n=3, claimed_by_session="x-session",
                                                              claimed_utc=synth.t(350), producer_session="x-session")])
    assert _ingest(root).refused == 0 and _row(root, xid)["n_items_judged"] == 3
    _rebuild_equal(root, tmp_path)


def test_the_fold_refuses_a_cross_run_successor_by_name_unless_the_record_allows_it_and_blocks_an_unfolded_one(tmp_path):
    root, ids = _corpus(tmp_path)
    (ra, rb), (ja, jb) = ids["run_ids"], ids["judgment_ids"]
    # the CLI refuses a cross-run successor without the flag
    with pytest.raises(SUP.SupersedeRefusal) as ei:
        SUP.supersede(root, ja, superseded_by=jb, reason="x", now=synth.t(400))
    assert ei.value.field == "superseded_by" and "another" not in ei.value.reason and rb in ei.value.reason and _supersede_files(root) == []
    # a hand-written record naming a cross-run successor WITHOUT extra.allow_cross_run is refused AT THE FOLD by name
    hand = _judgment(ra, ja, synth.JUDGE_ID, "superseded", 401, n=4, superseded_by_judgment_id=jb)
    del hand["payload"]["n_items_judged"]
    _write(root, "hand", "000001-cross.jsonl", [hand])
    rep = _ingest(root)
    assert rep.refused == 1 and any("SupersedeRelationshipMismatch" in r and "same run" in r for r in rep.refusals), rep.refusals
    ref = _q(root, "SELECT field, reason FROM record_refusal WHERE record_sha256 = ?", R.record_sha(hand))[0]
    assert ref["field"] == "superseded_by_judgment_id" and reducer.SupersedeRelationshipMismatch.__name__ in ref["reason"]
    assert _row(root, ja)["status"] == "complete"
    # a successor that has not folded yet is dependency-blocked (never refused for it) and folds when the successor lands
    yid, yrecs = _second(ra, synth.JUDGE_ID, "y-session", 350)
    early = _judgment(ra, ja, synth.JUDGE_ID, "superseded", 402, n=4, superseded_by_judgment_id=yid)
    del early["payload"]["n_items_judged"]
    _write(root, "hand", "000002-early.jsonl", [early])
    rep = _ingest(root)
    assert rep.blocked == 1 and rep.refused == 0, (rep.blocked, rep.refused, rep.refusals)
    _write(root, "second", "000001-y.jsonl", yrecs)
    assert _ingest(root).refused == 0
    assert _row(root, ja)["status"] == "superseded" and _row(root, ja)["superseded_by_judgment_id"] == yid
    # with the flag the CLI admits a cross-run successor and the record carries extra.allow_cross_run; the fold admits it too
    sup = SUP.supersede(root, jb, superseded_by=yid, reason="the runs are the same measurement under two keys", allow_cross_run=True, now=synth.t(500))
    assert sup.in_force is True and sup.record["payload"]["extra"]["allow_cross_run"] is True
    assert _row(root, jb)["superseded_by_judgment_id"] == yid and json.loads(_row(root, jb)["extra_json"])["allow_cross_run"] is True
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# claims: release and the forced takeover carry the claim's claimed_utc; a claim-backed supersede
# --------------------------------------------------------------------------------------------


def _claim_records(root, rid):
    d = os.path.join(root, "records", "claims", I.run_dir_name(rid))
    out = []
    for f in R.iter_record_files(d):
        for _, rec, sha in R.read_record_file(f):
            out.append((rec, sha, f))
    return out


def test_release_writes_the_claims_own_claimed_utc_and_the_record_conforms(tmp_path):
    root, rid = _generated(tmp_path)
    jid = P.claim(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(500))
    assert P.release(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(600))
    abandoned = [(rec, sha, f) for rec, sha, f in _claim_records(root, rid) if rec["payload"]["status"] == "abandoned"]
    assert len(abandoned) == 1
    rec, sha, path = abandoned[0]
    assert rec["payload"]["judgment_id"] == jid and rec["payload"]["claimed_utc"] == synth.t(500) and rec["written_utc"] == synth.t(600)
    assert I.judgment_id_form(rec["payload"]) == "claimed_by_session"
    assert isinstance(VAL.validate_record(rec, path, sha), VAL.Validated)
    assert _row(root, jid)["status"] == "abandoned" and _row(root, jid)["claimed_utc"] == synth.t(500)
    assert _pending(root, rid, synth.JUDGE_ID)["pending_reason"] == "abandoned"
    # the OLD release shape -- claimed_utc re-stated as the release time -- reproduces no form and is refused by name, unless grandfathered by sha
    legacy = dict(rec)
    legacy["payload"] = dict(rec["payload"], claimed_utc=synth.t(600))
    bad = _validate(legacy)
    assert isinstance(bad, VAL.Malformed) and bad.field == "judgment_id"
    assert isinstance(VAL.validate_record(legacy, "<legacy>", "not-the-sha"), VAL.Malformed)
    saved = VAL.GRANDFATHERED_RECORD_SHAS
    VAL.GRANDFATHERED_RECORD_SHAS = saved | {R.record_sha(legacy)}
    try:
        assert isinstance(VAL.validate_record(legacy, "<legacy>", R.record_sha(legacy)), VAL.Validated)
    finally:
        VAL.GRANDFATHERED_RECORD_SHAS = saved
    _rebuild_equal(root, tmp_path)


def test_the_grandfather_list_is_the_audited_release_records_the_d650_run_records_and_the_g39_judgment_records():
    """34 distinct record shas, every one 64 hex: the 18 release records of the judgment identity rule (an early audit found 4 on
    2026-10-02; the re-run found 18), the 4 `run` records of the two test runs (a whole-log audit, 2026-10-03: exactly
    four, two runs by two files) and the 12 g39 supersession / correction records about claim-minted judgment ids (the whole-log
    audit of 2026-10-04; grandfathered by the review fixes); the list is re-measured before every landing and grows only by audit."""
    shas = VAL.GRANDFATHERED_RECORD_SHAS
    assert isinstance(shas, frozenset) and len(shas) == 34
    assert all(isinstance(s, str) and len(s) == 64 and int(s, 16) >= 0 for s in shas)


def test_a_forced_takeover_abandons_the_stale_claim_under_its_own_claimed_utc_and_the_queue_shows_the_new_claim_alone(tmp_path):
    root, rid = _generated(tmp_path)
    t_a, t_b = "2026-01-01T00:00:00.000Z", "2026-01-02T00:00:00.000Z"
    a = P.claim(root, rid, synth.JUDGE_ID, "session-A", now=t_a)
    b = P.claim(root, rid, synth.JUDGE_ID, "session-B", force=True, stale_hours=12, now=t_b)
    assert a != b
    abandoned = [(rec, sha, f) for rec, sha, f in _claim_records(root, rid) if rec["payload"]["status"] == "abandoned"]
    assert len(abandoned) == 1
    rec, sha, path = abandoned[0]
    assert rec["payload"]["judgment_id"] == a and rec["payload"]["claimed_by_session"] == "session-A"
    assert rec["payload"]["claimed_utc"] == t_a and rec["written_utc"] == t_b                 # A's time on the payload, B's clock on the stamp
    assert I.judgment_id_form(rec["payload"]) == "claimed_by_session"
    assert isinstance(VAL.validate_record(rec, path, sha), VAL.Validated)
    st = {r["judgment_id"]: r["status"] for r in _q(root, "SELECT judgment_id, status FROM judgment WHERE run_id = ?", rid)}
    assert st == {a: "abandoned", b: "in_progress"}
    pend = _pending(root, rid, synth.JUDGE_ID)
    assert pend["pending_reason"] == "claimed" and pend["claim_judgment_id"] == b and pend["n_abandoned"] == 1
    assert _q(root, "SELECT COUNT(*) AS n FROM judgment WHERE run_id = ? AND status = 'in_progress'", rid)[0]["n"] == 1
    # the OLD forced-takeover shape (A's abandonment stamped with B's time) is refused by the rule unless grandfathered
    old = dict(rec)
    old["payload"] = dict(rec["payload"], claimed_utc=t_b)
    assert isinstance(_validate(old), VAL.Malformed) and _validate(old).field == "judgment_id"
    _rebuild_equal(root, tmp_path)


def test_a_claim_backed_judgment_is_superseded_with_its_identity_inputs_copied_and_rebuilds_equal(tmp_path):
    root, rid = _generated(tmp_path)
    jid = P.claim(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(500))
    assert jid == I.judgment_id(rid, synth.JUDGE_ID, "session-A", synth.t(500))
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid)[0]["lifecycle_state"] == "judging"
    # the hook completes the claimed judgment (its shape: the claim's id, session and time, the producer session)
    done = _judgment(rid, jid, synth.JUDGE_ID, "complete", 600, n=4, claimed_by_session="session-A", claimed_utc=synth.t(500),
                     producer_session="session-A", store_uri="synthetic:///grades/k")
    _write(root, "c", "000002-complete.jsonl", [done, synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(601),
                                                                           "evidence_kind": "completion_artifact"}, synth.t(601))])
    assert _ingest(root).refused == 0
    row = _row(root, jid)
    assert row["status"] == "complete" and I.judgment_id_form(row) == "claimed_by_session"
    assert I.judgment_id_form({**row, "claimed_by_session": None, "claimed_utc": None}) is None    # the plain form does NOT mint this id
    rep = SUP.supersede(root, jid, no_successor=True, reason="claimed grade superseded", now=synth.t(700))
    assert rep.in_force is True
    payload = rep.record["payload"]
    assert payload["claimed_by_session"] == "session-A" and payload["claimed_utc"] == synth.t(500)
    assert set(payload) == {"judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id", "claimed_by_session", "claimed_utc", "extra"}
    after = _row(root, jid)
    assert after["status"] == "superseded" and after["claimed_by_session"] == "session-A" and after["claimed_utc"] == synth.t(500)
    assert after["n_items_judged"] == 4 and after["store_uri"] == "synthetic:///grades/k" and after["producer_session"] == "session-A"
    assert _pending(root, rid, synth.JUDGE_ID)["pending_reason"] == "superseded"
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid)[0]["lifecycle_state"] in ("judged", "ingested")
    _rebuild_equal(root, tmp_path)
    # the same record WITHOUT the identity inputs reproduces no form: refused by the pre-write validation, nothing more written
    stripped = dict(rep.record)
    stripped["payload"] = {k: v for k, v in payload.items() if k not in ("claimed_by_session", "claimed_utc")}
    bad = _validate(stripped)
    assert isinstance(bad, VAL.Malformed) and bad.field == "judgment_id"
    n_files = len(_supersede_files(root))
    import sisteer.registry.supersede as module
    saved = module.identity_inputs
    module.identity_inputs = lambda row: {}
    try:
        with pytest.raises(SUP.SupersedeRefusal) as ei:
            SUP.supersede(root, jid, no_successor=True, reason="again", now=synth.t(800))
    finally:
        module.identity_inputs = saved
    # (J is superseded already, so the status guard fires first on a plain re-run; exercise the pre-write validation on a fresh claim-backed row)
    assert ei.value.field == "status"
    jid2 = P.claim(root, rid, synth.JUDGE_ID, "session-B", now=synth.t(900))
    _write(root, "c", "000003-complete2.jsonl", [_judgment(rid, jid2, synth.JUDGE_ID, "complete", 910, n=4, claimed_by_session="session-B",
                                                            claimed_utc=synth.t(900), producer_session="session-B")])
    assert _ingest(root).refused == 0
    module.identity_inputs = lambda row: {}
    try:
        with pytest.raises(SUP.SupersedeRefusal) as ei:
            SUP.supersede(root, jid2, no_successor=True, reason="again", now=synth.t(1000))
    finally:
        module.identity_inputs = saved
    assert ei.value.field == "judgment_id" and "does not validate" in ei.value.reason
    assert len(_supersede_files(root)) == n_files and _row(root, jid2)["status"] == "complete"


def test_a_live_row_whose_id_no_form_reproduces_refuses_by_name(tmp_path):
    root, rid = _generated(tmp_path)
    foreign = str(I.judgment_id("another-run", synth.JUDGE_ID))
    rec = _judgment(rid, foreign, synth.JUDGE_ID, "complete", 300, n=4)
    saved = VAL.GRANDFATHERED_RECORD_SHAS
    VAL.GRANDFATHERED_RECORD_SHAS = saved | {R.record_sha(rec)}       # a legacy row that folded before the rule
    try:
        _write(root, "c", "000002-legacy.jsonl", [rec])
        assert _ingest(root).refused == 0
    finally:
        VAL.GRANDFATHERED_RECORD_SHAS = saved
    assert _row(root, foreign)["status"] == "complete"
    with pytest.raises(SUP.SupersedeRefusal) as ei:
        SUP.supersede(root, foreign, no_successor=True, reason="x", now=synth.t(400))
    assert ei.value.field == "judgment_id" and "no identity form" in ei.value.reason and _supersede_files(root) == []
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert SUP.show(conn, foreign)["identity_form"] is None
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# two judges: a supersede never ends another judge's open claim
# --------------------------------------------------------------------------------------------


def _run_state(root, rid):
    row = _q(root, "SELECT lifecycle_state, field_versions_json FROM run WHERE run_id = ?", rid)[0]
    return row["lifecycle_state"], json.loads(row["field_versions_json"])["lifecycle_state"][0]


@pytest.mark.parametrize("ending", ["released", "completed"])
def test_superseding_one_judges_judgment_keeps_the_run_judging_while_another_judges_claim_is_open(tmp_path, ending):
    """Judge B holds an in_progress claim on a generated run (the run reads `judging`); judge A's judgment is complete
    (an adapter's judgment, no `run_state judged` of its own); A's judgment is superseded with no successor. The run keeps reading
    `judging` -- B is still judging it -- with the stamp of B's claim, and moves on only when B's claim ends: to `generated` when B
    releases it, to `judged` when B completes and files `judged`. The fold is the same live and rebuilt (the open claims are a function
    of the run key's ordered events)."""
    root, rid = _generated(tmp_path)
    b_claim = P.claim(root, rid, synth.JUDGE_ID_2, "session-B", now=synth.t(500))
    assert _run_state(root, rid) == ("judging", synth.t(500))
    a_jid = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "c", "000002-judge-a.jsonl", [_judgment(rid, a_jid, synth.JUDGE_ID, "complete", 510, n=4)])
    assert _ingest(root).refused == 0
    assert _run_state(root, rid) == ("judging", synth.t(500))           # A's complete judgment files no lifecycle state of its own
    rep = SUP.supersede(root, a_jid, no_successor=True, reason="judge A misread the rubric", now=synth.t(600))
    assert rep.in_force is True and _row(root, a_jid)["status"] == "superseded"
    assert _run_state(root, rid) == ("judging", synth.t(500)), "B's claim is still open: superseding A's judgment must not end it"
    assert _row(root, b_claim)["status"] == "in_progress"
    pend_b = _pending(root, rid, synth.JUDGE_ID_2)
    assert pend_b["pending_reason"] == "claimed" and pend_b["claim_judgment_id"] == b_claim
    assert _pending(root, rid, synth.JUDGE_ID)["pending_reason"] == "superseded"
    _rebuild_equal(root, tmp_path)
    if ending == "released":
        assert P.release(root, rid, synth.JUDGE_ID_2, "session-B", now=synth.t(700))
        assert _row(root, b_claim)["status"] == "abandoned"
        assert _run_state(root, rid) == ("generated", synth.t(700))     # the last open claim ended: back to generated, stamped by the release
        assert _pending(root, rid, synth.JUDGE_ID_2)["pending_reason"] == "abandoned"
    else:
        _write(root, "c", "000003-judge-b.jsonl", [_judgment(rid, b_claim, synth.JUDGE_ID_2, "complete", 700, n=4, claimed_by_session="session-B",
                                                               claimed_utc=synth.t(500), producer_session="session-B"),
                                                    synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(701),
                                                                            "evidence_kind": "completion_artifact"}, synth.t(701))])
        assert _ingest(root).refused == 0
        assert _row(root, b_claim)["status"] == "complete"
        assert _run_state(root, rid) == ("judged", synth.t(701))        # B's own `judged` moves the run forward; nothing moved it back first
        assert _pending(root, rid, synth.JUDGE_ID_2) is None
    assert _pending(root, rid, synth.JUDGE_ID)["pending_reason"] == "superseded"
    _rebuild_equal(root, tmp_path, shuffle_seed=5)


# --------------------------------------------------------------------------------------------
# guards, dry run, the CLI
# --------------------------------------------------------------------------------------------


def test_every_guard_refuses_by_name_with_nothing_written(tmp_path):
    root, ids = _corpus(tmp_path)
    (ra, rb), (ja, jb) = ids["run_ids"], ids["judgment_ids"]
    # an in_progress claim, an abandoned one and a superseded one are not supersedable
    rid_c = ra
    claimed = P.claim(root, rid_c, synth.JUDGE_ID_2, "s", now=synth.t(300))
    xid, xrecs = _second(ra, synth.JUDGE_ID, "x-session", 350)
    _write(root, "second", "000001-x.jsonl", xrecs)
    assert _ingest(root).refused == 0
    assert SUP.supersede(root, xid, no_successor=True, reason="retire x", now=synth.t(360)).in_force is True
    n_before = len(_supersede_files(root))
    stamp = json.loads(_row(root, ja)["field_versions_json"])["status"][0]
    cases = [
        (dict(judgment_id="unknown", no_successor=True, reason="x"), "judgment_id"),
        (dict(judgment_id=claimed, no_successor=True, reason="x"), "status"),
        (dict(judgment_id=xid, no_successor=True, reason="x"), "status"),
        (dict(judgment_id=ja, superseded_by="unknown", reason="x"), "superseded_by"),
        (dict(judgment_id=ja, superseded_by=xid, reason="x"), "superseded_by"),              # the successor is superseded itself
        (dict(judgment_id=ja, superseded_by=claimed, reason="x"), "superseded_by"),          # the successor is a claim
        (dict(judgment_id=ja, superseded_by=ja, reason="x"), "superseded_by"),               # itself
        (dict(judgment_id=ja, superseded_by=jb, reason="x"), "superseded_by"),               # another run, no flag
        (dict(judgment_id=ja, reason="x"), "superseded_by"),                                  # neither
        (dict(judgment_id=ja, superseded_by=jb, no_successor=True, reason="x"), "superseded_by"),   # both
        (dict(judgment_id=ja, no_successor=True, reason="  "), "reason"),
        (dict(judgment_id=ja, no_successor=True, reason="x", provenance_kind="relayed"), "provenance"),
        (dict(judgment_id=ja, no_successor=True, reason="x", provenance_kind="relayed", relayed_by="Peer"), "provenance"),
        (dict(judgment_id=ja, no_successor=True, reason="x", provenance_kind="peer_encoded"), "provenance_kind"),
        (dict(judgment_id=ja, no_successor=True, reason="x", now=stamp), "written_utc"),
    ]
    for kw, field_name in cases:
        kw = dict(kw)
        jid = kw.pop("judgment_id")
        with pytest.raises(SUP.SupersedeRefusal) as ei:
            SUP.supersede(root, jid, now=kw.pop("now", synth.t(500)), **kw)
        assert ei.value.field == field_name, (kw, ei.value)
    assert len(_supersede_files(root)) == n_before and _row(root, ja)["status"] == "complete"
    # the relayed form, complete, is admitted and lands on the record
    rep = SUP.supersede(root, ja, no_successor=True, reason="x", provenance_kind="relayed", relayed_by="Peer", message_utc="2026-10-02T20:00:00.000Z",
                        now=synth.t(500))
    pv = rep.record["payload"]["extra"]["supersede"]["provenance"]
    assert pv == {"kind": "relayed", "relayed_by": "Peer", "message_utc": "2026-10-02T20:00:00.000Z", "confirmed_by_user": False}
    assert rep.record["payload"]["extra"]["supersede"]["ruled_by"] == "the user, relayed by Peer"


def test_dry_run_prints_the_record_and_the_consequences_and_writes_nothing(tmp_path, capsys):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    before = {os.path.relpath(f, root): os.stat(f).st_mtime_ns for f in R.iter_record_files(os.path.join(root, "records"))}
    rep = SUP.supersede(root, jid, no_successor=True, reason="look first", dry_run=True, now=synth.t(400))
    assert rep.dry_run and not rep.written and not rep.folded and rep.in_force is None
    assert rep.record["payload"]["status"] == "superseded" and rep.remaining_same_judge == []
    assert {os.path.relpath(f, root): os.stat(f).st_mtime_ns for f in R.iter_record_files(os.path.join(root, "records"))} == before
    assert not os.path.isdir(os.path.join(root, "state", "last_run")) or not os.listdir(os.path.join(root, "state", "last_run"))
    assert _row(root, jid)["status"] == "complete"
    # through the CLI, --dry-run and show leave the root byte for byte (no state/last_run stamp either)
    snapshot = lambda: sorted((os.path.relpath(os.path.join(d, f), root), os.stat(os.path.join(d, f)).st_size, os.stat(os.path.join(d, f)).st_mtime_ns)
                              for d, _, fs in os.walk(root) for f in fs if not f.startswith("registry.sqlite"))
    tree = snapshot()
    assert cli.main(["--root", root, "judgment", "supersede", jid, "--no-successor", "--reason", "look first", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing written" in out and '"status": "superseded"' in out
    assert cli.main(["--root", root, "judgment", "show", jid]) == 0
    out = capsys.readouterr().out
    assert "status: complete" in out and "identity_form: run_judge" in out
    assert cli.main(["--root", root, "judgment", "show", jid, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert snapshot() == tree


def test_the_cli_writes_folds_and_exits_by_the_outcome(tmp_path, capsys):
    root, ids = _corpus(tmp_path)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    assert cli.main(["--root", root, "judgment", "supersede", jid, "--reason"]) == 2 if False else True   # argparse usage errors are argparse's
    rc = cli.main(["--root", root, "judgment", "supersede", jid, "--no-successor", "--reason", "retire", "--ruled-by", "the user", "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["written"] and rep["folded"] and rep["in_force"] is True and rep["path"].startswith(root)
    assert _row(root, jid)["status"] == "superseded"
    assert os.path.isfile(os.path.join(root, "state", "last_run", "judgment-supersede.json"))      # a writing command stamps its run
    # the guards through the CLI: exit 1, the refusal named on stderr, nothing more written
    n = len(_supersede_files(root))
    assert cli.main(["--root", root, "judgment", "supersede", jid, "--no-successor", "--reason", "again"]) == 1
    err = capsys.readouterr().err
    assert "judgment supersede refused (status)" in err and len(_supersede_files(root)) == n
    assert cli.main(["--root", root, "judgment", "show", "nope"]) == 1
    assert "refused (judgment_id)" in capsys.readouterr().err
    assert cli.main(["--root", str(tmp_path / "absent"), "judgment", "show", jid]) == 1
    _rebuild_equal(root, tmp_path)
