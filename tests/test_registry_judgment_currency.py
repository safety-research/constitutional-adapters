"""The two paper-side judgment-currency rules of the record-refusal-fact migration, rendered beside the status rule at every
paper-side site -- and the identity rule in `v_pending_judgment` too:

* complete outranks partial per (run, judge): a `partial` judgment beside a `complete` one of the SAME (run_id, judge_id) is not
  current (`v_paper_long`, `v_multi_source_rows`, the headline / coverage readers, the item objects); a partial reading alone is admitted;
  a complete judgment of ANOTHER judge demotes nothing; `v_pending_judgment`'s `partial` path keeps working.
* the identity BOUND AT CLAIM must be the run's CURRENT identity: `pending.claim` states `claimed_instrument_id` /
  `claimed_required_judge_id` from the run row; a same-id run identity correction after the claim reads the completed judgment non-current
  in every paper frame and ownership scope while the run is listed for the new judge (`n_complete_other_judges = 1`); the new judge's
  complete judgment is then the one headline row; a SECOND judge's judgment on an uncorrected run stays current (its binding is the run's
  identity, whatever its judge); NULL bindings are current; a correction that changes neither binding demotes nothing; a run whose
  instrument names no judge falls back to `run.judge_id`; rebuild equality throughout.
* the worklist counts identity-current judgments only: a same-judge identity correction (X to X2, both requiring J) leaves J's
  complete judgment non-current on the paper side AND pending for J (`pending_reason = identity_stale`), never unmeasured with no pending
  job; the displacing complete witness is itself identity-current, so a stale complete beside a current partial of the same judge does
  not hide the partial (the partial is the paper row, the worklist reads `partial` with `items_missing` from it alone).
* a stale-bound CLAIM is not held: an in-progress claim bound to a since-corrected identity can only
  complete into a judgment the rule excludes, so the worklist does not present it as `claimed` (the run reads `identity_stale` with no claim
  columns; `registry pending-judgment` prints it), and `pending.claim` closes it `abandoned` (restating its own `claimed_utc`)
  and binds a NEW claim to the current identity -- the same session's retry gets a fresh id, another session takes the run over without
  `force`, never `ClaimHeld` (a current-bound claim is held exactly as before); completing the OLD id afterwards yields a judgment that is
  non-current everywhere while the run stays pending (claimed by the new claim) and `judged`, never `ingested`; and the same identity test
  rides the other consumers of the coverage predicate -- `v_roster_match` / the measurement-judge rule (a stale-bound complete judgment
  satisfies no roster entry: the entry stays a gap) and the harvester's completion requirement.
* `pending.claim` returns only a claim bound to the identity the fold LEFT: a run correction already spooled
  and not yet folded is folded FIRST under the lock (else the claim bound the old identity and the same call folded the correction over
  it), and after the claim's own fold the minted claim's binding is re-checked against the folded run -- a correction landing between the
  two folds abandons the claim and re-mints it one millisecond later under the same lock; `active_claim` reads identity-current claims
  only, so a worklist never grades under a stale one.

Live at the rules' landing (read-only, 2026-10-03 16:00Z): 0 (run, judge) pairs carry both a partial and a complete judgment,
and the two binding columns are new, so every live judgment is NULL-bound: each rule moves zero live rows.
"""
from __future__ import annotations

import json
import os
import random
import shutil

import pytest

from sisteer.registry import cli as CLI
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry.adapters import lineage_common as LC

#: a second instrument of the same eval whose declared judge is the second judge: the identity a correction moves a run to
INSTRUMENT_Y = synth.instrument_fields(judge_id=synth.JUDGE_ID_2, notes="the corrected instrument")
INSTRUMENT_Y_ID = str(INSTRUMENT_Y["instrument_id"])
assert INSTRUMENT_Y_ID != synth.INSTRUMENT_ID
#: a third instrument of the same eval naming the SAME required judge as the seeded one (another token cap): the identity a same-judge
#: correction moves a run to
INSTRUMENT_X2 = synth.instrument_fields(token_cap_tokens=1024, token_cap_ladder_json="[1024]", notes="the same-judge corrected instrument")
INSTRUMENT_X2_ID = str(INSTRUMENT_X2["instrument_id"])
assert INSTRUMENT_X2_ID not in (synth.INSTRUMENT_ID, INSTRUMENT_Y_ID) and INSTRUMENT_X2["judge_id"] == synth.JUDGE_ID
#: two more same-judge instruments (other token caps): the identities a chain of spooled corrections moves a run through
INSTRUMENT_X3 = synth.instrument_fields(token_cap_tokens=2048, token_cap_ladder_json="[2048]", notes="the third same-judge instrument")
INSTRUMENT_X3_ID = str(INSTRUMENT_X3["instrument_id"])
INSTRUMENT_X4 = synth.instrument_fields(token_cap_tokens=4096, token_cap_ladder_json="[4096]", notes="the fourth same-judge instrument")
INSTRUMENT_X4_ID = str(INSTRUMENT_X4["instrument_id"])
assert len({synth.INSTRUMENT_ID, INSTRUMENT_Y_ID, INSTRUMENT_X2_ID, INSTRUMENT_X3_ID, INSTRUMENT_X4_ID}) == 5


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root, now=None):
    return H.ingest_once(root, workers=0, min_free=0, now=now)


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _frame(root, rid):
    return sorted((r["judgment_id"], r["metric_id"]) for r in _q(root, "SELECT judgment_id, metric_id FROM v_paper_long WHERE run_id = ?", rid))


def _items(root, name, rid):
    return {r["judgment_id"]: r["n"] for r in _q(root, f"SELECT judgment_id, COUNT(*) AS n FROM {name} WHERE run_id = ? GROUP BY judgment_id", rid)}


def _coverage(root, rid):
    return _q(root, "SELECT coverage_state FROM v_paper_current WHERE run_id = ?", rid)[0]["coverage_state"]


def _pending(root, rid):
    return {r["required_judge_id"]: r for r in _q(root, "SELECT required_judge_id, pending_reason, n_complete_other_judges, is_instrument_judge FROM v_pending_judgment WHERE run_id = ?", rid)}


def _judgment(rid, jid, judge, status, i, n=4, **over):
    p = {"judgment_id": jid, "run_id": rid, "judge_id": judge, "n_items_judged": n, "status": status}
    p.update(over)
    return synth.rec("judgment", p, synth.t(i))


def _metric(rid, jid, i, value=0.5):
    return synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": value, "k": 2, "n": 4,
                                    "computed_by": "producer"}, synth.t(i))


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
        assert not diff, diff
    finally:
        a.close()
        b.close()


def _generated(tmp_path, n_items=4, extra_dims=()):
    """A generated run (no judgment yet) on the seeded instrument, the second instrument seeded beside it."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + [synth.rec("instrument", INSTRUMENT_Y, synth.t(30))] + list(extra_dims))
    rid, recs = synth.run_records("k", 100, through="generated", n_items=n_items)
    _write(root, "c", "000001-run.jsonl", recs)
    assert _ingest(root, synth.t(900)).refused == 0
    return root, rid


# --------------------------------------------------------------------------------------------
# complete outranks partial per (run, judge)
# --------------------------------------------------------------------------------------------


def test_a_partial_judgment_alone_is_admitted_and_leaves_the_frame_once_a_complete_one_of_the_same_judge_folds(tmp_path):
    root, rid = _generated(tmp_path)
    partial = I.judgment_id(rid, synth.JUDGE_ID, "s-partial", synth.t(200))
    _write(root, "j", "000001-partial.jsonl", [_judgment(rid, partial, synth.JUDGE_ID, "partial", 200, n=3, claimed_by_session="s-partial", claimed_utc=synth.t(200)),
                                               _metric(rid, partial, 201)] + synth.item_records(rid, partial, 202, n_items=3, seed=1))
    assert _ingest(root, synth.t(1000)).refused == 0
    assert _frame(root, rid) == [(partial, "asr")]                                   # a partial reading alone is admitted, as before
    assert _items(root, "v_item_result_current", rid) == {partial: 3} and _items(root, "v_item_headline", rid) == {partial: 3}
    assert _pending(root, rid)[synth.JUDGE_ID]["pending_reason"] == "partial"           # the pending path is untouched
    complete = I.judgment_id(rid, synth.JUDGE_ID)
    _write(root, "j", "000002-complete.jsonl", [_judgment(rid, complete, synth.JUDGE_ID, "complete", 300, n=4), _metric(rid, complete, 301, value=0.75),
                                                synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(302), "evidence_kind": "completion_artifact"}, synth.t(302))]
           + synth.item_records(rid, complete, 303, n_items=4, seed=2))
    assert _ingest(root, synth.t(2000)).refused == 0
    assert _frame(root, rid) == [(complete, "asr")]                                  # one headline row: the complete judgment's
    assert _items(root, "v_item_result_current", rid) == {complete: 4} and _items(root, "v_item_result_provenances", rid) == {complete: 4}
    assert _items(root, "v_item_headline", rid) == {complete: 4}
    assert _coverage(root, rid) == "measured"
    assert _q(root, "SELECT status FROM judgment WHERE judgment_id = ?", partial) == [{"status": "partial"}]   # the row stands; it is just not current
    assert synth.JUDGE_ID not in _pending(root, rid)                                   # the complete judgment covers the run
    assert _q(root, "SELECT COUNT(*) AS n FROM item_result WHERE run_id = ?", rid) == [{"n": 7}]
    _rebuild_equal(root, tmp_path)


def test_a_complete_judgment_of_another_judge_does_not_demote_a_partial_one(tmp_path):
    root, rid = _generated(tmp_path)
    partial = I.judgment_id(rid, synth.JUDGE_ID, "s-partial", synth.t(200))
    other = I.judgment_id(rid, synth.JUDGE_ID_2)
    _write(root, "j", "000001-both.jsonl", [_judgment(rid, partial, synth.JUDGE_ID, "partial", 200, n=3, claimed_by_session="s-partial", claimed_utc=synth.t(200)),
                                            _metric(rid, partial, 201), _judgment(rid, other, synth.JUDGE_ID_2, "complete", 300, n=4), _metric(rid, other, 301, value=0.75)])
    assert _ingest(root, synth.t(2000)).refused == 0
    assert _frame(root, rid) == sorted([(partial, "asr"), (other, "asr")])
    pend = _pending(root, rid)
    assert pend[synth.JUDGE_ID]["pending_reason"] == "partial" and pend[synth.JUDGE_ID]["n_complete_other_judges"] == 1
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the identity bound at claim is the run's current identity
# --------------------------------------------------------------------------------------------


def test_pending_claim_binds_the_runs_identity_and_the_binding_folds_and_rebuilds_equal(tmp_path):
    root, rid = _generated(tmp_path)
    jid = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    row = _q(root, "SELECT claimed_instrument_id, claimed_required_judge_id, status FROM judgment WHERE judgment_id = ?", jid)[0]
    assert row == {"claimed_instrument_id": synth.INSTRUMENT_ID, "claimed_required_judge_id": synth.JUDGE_ID, "status": "in_progress"}
    # a second judge's claim binds the SAME identity (the run's), not its own judge
    jid2 = P.claim(root, rid, synth.JUDGE_ID_2, "s-B", now=synth.t(1001))
    row2 = _q(root, "SELECT claimed_instrument_id, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", jid2)[0]
    assert row2 == {"claimed_instrument_id": synth.INSTRUMENT_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    # the release restates no binding (the per-column fold keeps the claim's); the record about the claim carries none
    assert P.release(root, rid, synth.JUDGE_ID_2, "s-B", now=synth.t(1002))
    rel = [rec for p in R.iter_record_files(os.path.join(root, "records", "claims")) for _, rec, _ in R.read_record_file(p)
           if rec["payload"]["judgment_id"] == jid2 and rec["payload"]["status"] == "abandoned"]
    assert len(rel) == 1 and "claimed_instrument_id" not in rel[0]["payload"] and "claimed_required_judge_id" not in rel[0]["payload"]
    assert _q(root, "SELECT claimed_instrument_id, status FROM judgment WHERE judgment_id = ?", jid2) == [{"claimed_instrument_id": synth.INSTRUMENT_ID, "status": "abandoned"}]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert P.run_identity_binding(conn, rid) == (synth.INSTRUMENT_ID, synth.JUDGE_ID) and P.run_identity_binding(conn, "no-such-run") == (None, None)
    conn.close()
    _rebuild_equal(root, tmp_path)


def _claim_and_complete(root, rid, judge, session, i):
    """A claim by ``judge`` completed by the hook's shape (the claim's id, its identity inputs restated, no binding restated)."""
    jid = P.claim(root, rid, judge, session, now=synth.t(i))
    _write(root, "j", f"{i:06d}-complete.jsonl", [_judgment(rid, jid, judge, "complete", i + 1, n=4, claimed_by_session=session, claimed_utc=synth.t(i)),
                                                  _metric(rid, jid, i + 2),
                                                  synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(i + 3), "evidence_kind": "completion_artifact"}, synth.t(i + 3))]
           + synth.item_records(rid, jid, i + 4, n_items=4, seed=i))
    return jid


def test_an_identity_correction_after_the_claim_demotes_the_completed_judgment_and_lists_the_run_for_the_new_judge(tmp_path):
    root, rid = _generated(tmp_path)
    jid = _claim_and_complete(root, rid, synth.JUDGE_ID, "s-A", 1000)
    assert _ingest(root, synth.t(2000)).refused == 0
    assert _frame(root, rid) == [(jid, "asr")] and _items(root, "v_item_result_current", rid) == {jid: 4} and _coverage(root, rid) == "measured"
    assert _q(root, "SELECT claimed_instrument_id, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", jid)[0] == \
        {"claimed_instrument_id": synth.INSTRUMENT_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    # the same-id run identity correction: a full `run` record moving the run onto instrument Y (whose judge is the second judge)
    corrected = synth.run_payload("k", i=100, instrument_id=INSTRUMENT_Y_ID)
    corrected["run_id"] = rid
    _write(root, "c", "000002-correction.jsonl", [synth.rec("run", corrected, synth.t(3000))])
    assert _ingest(root, synth.t(3100)).refused == 0
    assert _q(root, "SELECT instrument_id FROM run WHERE run_id = ?", rid) == [{"instrument_id": INSTRUMENT_Y_ID}]
    # the judgment is a fact that stands; it is non-current on the paper side: frame, items, headline, coverage
    assert _q(root, "SELECT status FROM judgment WHERE judgment_id = ?", jid) == [{"status": "complete"}]
    assert _frame(root, rid) == [] and _items(root, "v_item_result_current", rid) == {} and _items(root, "v_item_result_provenances", rid) == {}
    assert _items(root, "v_item_headline", rid) == {} and _coverage(root, rid) == "absent"
    assert _q(root, "SELECT COUNT(*) AS n FROM item_result WHERE run_id = ?", rid) == [{"n": 4}] and _q(root, "SELECT COUNT(*) AS n FROM run_metric WHERE run_id = ?", rid) == [{"n": 1}]
    # the run is listed for the NEW judge (the corrected instrument's), the old judge's complete judgment counted as another judge's
    pend = _pending(root, rid)
    assert pend[synth.JUDGE_ID_2]["pending_reason"] == "awaiting_second_judge" and pend[synth.JUDGE_ID_2]["n_complete_other_judges"] == 1
    assert pend[synth.JUDGE_ID_2]["is_instrument_judge"] == 1
    # the old judge's stale-bound judgment covers nothing for ITS judge either: J is eligible again, as a second judge now
    assert pend[synth.JUDGE_ID]["pending_reason"] == "identity_stale" and pend[synth.JUDGE_ID]["is_instrument_judge"] == 0
    assert pend[synth.JUDGE_ID]["n_complete_other_judges"] == 0
    # the new judge's complete judgment: one headline row, the new judge's
    jid2 = _claim_and_complete(root, rid, synth.JUDGE_ID_2, "s-B", 4000)
    assert _ingest(root, synth.t(5000)).refused == 0
    assert _q(root, "SELECT claimed_instrument_id, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", jid2)[0] == \
        {"claimed_instrument_id": INSTRUMENT_Y_ID, "claimed_required_judge_id": synth.JUDGE_ID_2}
    assert _frame(root, rid) == [(jid2, "asr")] and _items(root, "v_item_result_current", rid) == {jid2: 4} and _coverage(root, rid) == "measured"
    assert synth.JUDGE_ID_2 not in _pending(root, rid)
    _rebuild_equal(root, tmp_path)


def test_a_second_judges_judgment_on_an_uncorrected_run_stays_current_whatever_its_binding_and_null_bindings_are_current(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    rid, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    assert _ingest(root, synth.t(900)).refused == 0
    # a second judge's complete judgment, NULL-bound (an adapter that does not know the instrument) and one bound to the run's identity
    j_null = I.judgment_id(rid, synth.JUDGE_ID_2)
    j_bound = I.judgment_id(rid, synth.JUDGE_ID_2, "s-two", synth.t(300))
    _write(root, "second", "000001-second.jsonl", [_judgment(rid, j_null, synth.JUDGE_ID_2, "complete", 200), _metric(rid, j_null, 201),
                                                   _judgment(rid, j_bound, synth.JUDGE_ID_2, "complete", 300, claimed_by_session="s-two", claimed_utc=synth.t(300),
                                                             claimed_instrument_id=synth.INSTRUMENT_ID, claimed_required_judge_id=synth.JUDGE_ID),
                                                   _metric(rid, j_bound, 301)])
    assert _ingest(root, synth.t(1000)).refused == 0
    assert _frame(root, rid) == sorted([(jid, "asr"), (j_null, "asr"), (j_bound, "asr")])     # the judge-based wording would have demoted both second-judge rows
    # a correction that changes neither binding (the run re-emitted on the same instrument with another store uri) demotes nothing
    same = synth.run_payload("synthstore_001", "constitution_lora@deadbeef", synth.OBJECT_SHA, i=120, store_uri="synthetic:///store/moved")
    same["run_id"] = rid
    _write(root, "c", "000009-same-identity.jsonl", [synth.rec("run", same, synth.t(2000))])
    assert _ingest(root, synth.t(2100)).refused == 0
    assert _q(root, "SELECT store_uri, instrument_id FROM run WHERE run_id = ?", rid) == [{"store_uri": "synthetic:///store/moved", "instrument_id": synth.INSTRUMENT_ID}]
    assert _frame(root, rid) == sorted([(jid, "asr"), (j_null, "asr"), (j_bound, "asr")])
    # the lineage helper states the binding only when a caller passes it
    assert "claimed_instrument_id" not in LC.judgment_payload(rid, synth.JUDGE_ID, 4, synth.t(1), None)
    assert LC.judgment_payload(rid, synth.JUDGE_ID, 4, synth.t(1), None, claimed_instrument_id=synth.INSTRUMENT_ID, claimed_required_judge_id=synth.JUDGE_ID)["claimed_required_judge_id"] == synth.JUDGE_ID
    _rebuild_equal(root, tmp_path)


def test_a_run_whose_instrument_names_no_judge_binds_and_compares_run_judge_id(tmp_path):
    """A rule-scored instrument names no judge: the binding falls back to `run.judge_id` (a run_patch of `judge_id`, the hook's shape), and
    a later patch moving `judge_id` reads the completed judgment non-current."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid, recs = synth.rule_scored_run_records("rk", 100, through="generated", n_items=4)
    recs.append(synth.rec("run_patch", {"run_id": rid, "fields": {"judge_id": synth.JUDGE_ID}}, synth.t(150)))
    _write(root, "c", "000001-run.jsonl", recs)
    assert _ingest(root, synth.t(900)).refused == 0
    assert _q(root, "SELECT i.judge_id AS ij, r.judge_id AS rj FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id = ?", rid) == [{"ij": None, "rj": synth.JUDGE_ID}]
    jid = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    assert _q(root, "SELECT claimed_instrument_id, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", jid)[0] == \
        {"claimed_instrument_id": synth.RULE_INSTRUMENT_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    _write(root, "j", "000002-complete.jsonl", [_judgment(rid, jid, synth.JUDGE_ID, "complete", 1001, n=4, claimed_by_session="s-A", claimed_utc=synth.t(1000)),
                                                synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "acc", "threshold": -1.0, "value": 0.5, "computed_by": "producer"}, synth.t(1002))])
    assert _ingest(root, synth.t(2000)).refused == 0
    assert _frame(root, rid) and (jid, "acc") in _frame(root, rid)
    _write(root, "c", "000003-judge-moved.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"judge_id": synth.JUDGE_ID_2}}, synth.t(3000))])
    assert _ingest(root, synth.t(3100)).refused == 0
    assert (jid, "acc") not in _frame(root, rid)                                            # the required judge moved: the binding differs
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the worklist counts identity-current judgments only; the displacing complete witness is identity-current
# --------------------------------------------------------------------------------------------


def _correct_to_x2(root, rid, i):
    """The same-id run identity correction onto instrument X2 -- another token cap, the SAME required judge as the seeded instrument."""
    corrected = synth.run_payload("k", i=100, instrument_id=INSTRUMENT_X2_ID)
    corrected["run_id"] = rid
    _write(root, "c", f"{i:06d}-correction.jsonl", [synth.rec("run", corrected, synth.t(i))])
    assert _ingest(root, synth.t(i + 100)).refused == 0
    assert _q(root, "SELECT r.instrument_id, i.judge_id FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id = ?", rid) == \
        [{"instrument_id": INSTRUMENT_X2_ID, "judge_id": synth.JUDGE_ID}]


def _worklist_row(root, rid, judge):
    rows = _q(root, "SELECT pending_reason, items_missing, required_coverage, n_complete_other_judges, is_instrument_judge, claim_judgment_id "
                    "FROM v_pending_judgment WHERE run_id = ? AND required_judge_id = ?", rid, judge)
    assert len(rows) <= 1
    return rows[0] if rows else None


def test_a_same_judge_identity_correction_lists_the_run_for_re_judging_instead_of_leaving_it_unmeasured(tmp_path):
    """The run moves from instrument X to X2, both requiring judge J. J's complete judgment, bound at claim to X, leaves every paper
    frame -- and the worklist must not count it as covering J's work, else the run has neither a paper row nor a pending job. It
    reads `identity_stale`; a re-claim binds X2 and its completion is the paper row."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    jid = _claim_and_complete(root, rid, synth.JUDGE_ID, "s-A", 1000)
    assert _ingest(root, synth.t(2000)).refused == 0
    assert _frame(root, rid) == [(jid, "asr")] and _coverage(root, rid) == "measured" and _worklist_row(root, rid, synth.JUDGE_ID) is None
    _correct_to_x2(root, rid, 3000)
    # non-current on the paper side: frame, items, headline, coverage -- the judgment row itself stands, a fact
    assert _q(root, "SELECT status, claimed_instrument_id FROM judgment WHERE judgment_id = ?", jid) == [{"status": "complete", "claimed_instrument_id": synth.INSTRUMENT_ID}]
    assert _frame(root, rid) == [] and _items(root, "v_item_result_current", rid) == {} and _items(root, "v_item_headline", rid) == {}
    assert _coverage(root, rid) == "absent"
    # ... AND pending for J again: the stale-bound complete judgment covers nothing in the worklist; the reason names why; nothing current
    # leaves an items_missing figure (as for a run with no judgment); no other judge has judged it
    assert _worklist_row(root, rid, synth.JUDGE_ID) == {"pending_reason": "identity_stale", "items_missing": None, "required_coverage": 4,
                                                        "n_complete_other_judges": 0, "is_instrument_judge": 1, "claim_judgment_id": None}
    assert _pending(root, rid)[synth.JUDGE_ID_2]["pending_reason"] == "awaiting_second_judge"      # the second-judge design is untouched
    # J claims again: the claim binds the CORRECTED identity, the worklist reads the live claim, the completion is the one headline row
    jid2 = P.claim(root, rid, synth.JUDGE_ID, "s-A2", now=synth.t(4000))
    assert _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed" and _worklist_row(root, rid, synth.JUDGE_ID)["claim_judgment_id"] == jid2
    _write(root, "j", "004000-complete.jsonl", [_judgment(rid, jid2, synth.JUDGE_ID, "complete", 4001, n=4, claimed_by_session="s-A2", claimed_utc=synth.t(4000)),
                                                _metric(rid, jid2, 4002, value=0.75)] + synth.item_records(rid, jid2, 4004, n_items=4, seed=4000))
    assert _ingest(root, synth.t(5000)).refused == 0
    assert _q(root, "SELECT claimed_instrument_id, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", jid2)[0] == \
        {"claimed_instrument_id": INSTRUMENT_X2_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    assert _frame(root, rid) == [(jid2, "asr")] and _items(root, "v_item_result_current", rid) == {jid2: 4} and _coverage(root, rid) == "measured"
    assert _worklist_row(root, rid, synth.JUDGE_ID) is None                                         # the current judgment covers
    assert _q(root, "SELECT COUNT(*) AS n FROM item_result WHERE run_id = ?", rid) == [{"n": 8}]       # both judgments' items are kept
    _rebuild_equal(root, tmp_path)


def test_a_stale_bound_complete_judgment_does_not_displace_a_current_partial_one_of_the_same_judge(tmp_path):
    """After the same-judge correction, J's new PARTIAL reading under the corrected identity is the paper row -- the displacing complete
    witness must itself be identity-current, else the stale complete would have hidden the partial and both readings would have left the
    paper. The worklist reads `partial` with `items_missing` from the current partial alone; J's complete reading under X2 displaces it."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    stale = _claim_and_complete(root, rid, synth.JUDGE_ID, "s-A", 1000)
    _correct_to_x2(root, rid, 3000)
    assert _frame(root, rid) == [] and _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "identity_stale"
    partial = P.claim(root, rid, synth.JUDGE_ID, "s-A2", now=synth.t(4000))
    _write(root, "j", "004000-partial.jsonl", [_judgment(rid, partial, synth.JUDGE_ID, "partial", 4001, n=3, claimed_by_session="s-A2", claimed_utc=synth.t(4000)),
                                               _metric(rid, partial, 4002, value=0.25)] + synth.item_records(rid, partial, 4003, n_items=3, seed=7))
    assert _ingest(root, synth.t(5000)).refused == 0
    assert _q(root, "SELECT status, claimed_instrument_id FROM judgment WHERE judgment_id = ?", partial) == [{"status": "partial", "claimed_instrument_id": INSTRUMENT_X2_ID}]
    # the partial is the paper row: the stale complete is no displacing witness -- frame, item objects, headline
    assert _frame(root, rid) == [(partial, "asr")]
    assert _items(root, "v_item_result_current", rid) == {partial: 3} and _items(root, "v_item_result_provenances", rid) == {partial: 3}
    assert _items(root, "v_item_headline", rid) == {partial: 3}
    assert _worklist_row(root, rid, synth.JUDGE_ID) == {"pending_reason": "partial", "items_missing": 1, "required_coverage": 4,
                                                        "n_complete_other_judges": 0, "is_instrument_judge": 1, "claim_judgment_id": None}
    # a complete reading under the corrected identity displaces the partial and covers the run
    complete = _claim_and_complete(root, rid, synth.JUDGE_ID, "s-A3", 6000)
    assert _ingest(root, synth.t(7000)).refused == 0
    assert _frame(root, rid) == [(complete, "asr")] and _items(root, "v_item_result_current", rid) == {complete: 4} and _coverage(root, rid) == "measured"
    assert _worklist_row(root, rid, synth.JUDGE_ID) is None
    assert sorted(r["status"] for r in _q(root, "SELECT status FROM judgment WHERE run_id = ? AND judge_id = ?", rid, synth.JUDGE_ID)) == ["complete", "complete", "partial"]
    assert _q(root, "SELECT COUNT(*) AS n FROM item_result WHERE run_id = ?", rid) == [{"n": 11}]
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# a stale-bound claim is not held
# --------------------------------------------------------------------------------------------


def _complete(root, rid, jid, judge, session, claimed_i, i, n=4, status="complete", with_state=True):
    """The completion of an EXISTING claim ``jid`` by the hook's shape: the claim form restated (``claimed_by_session`` + the claim's own
    ``claimed_utc``, the identity inputs), no binding restated, a metric row, the item rows and -- by default -- the run's `judged` event."""
    recs = [_judgment(rid, jid, judge, status, i, n=n, claimed_by_session=session, claimed_utc=synth.t(claimed_i)), _metric(rid, jid, i + 1)]
    if with_state:
        recs.append(synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(i + 2), "evidence_kind": "completion_artifact"}, synth.t(i + 2)))
    _write(root, "j", f"{i:06d}-complete-{jid[:8]}.jsonl", recs + synth.item_records(rid, jid, i + 3, n_items=n, seed=i))


def _judgment_rows(root, rid):
    return {r["judgment_id"]: r for r in _q(root, "SELECT judgment_id, status, claimed_utc, claimed_by_session, claimed_instrument_id, "
                                                   "claimed_required_judge_id FROM judgment WHERE run_id = ?", rid)}


def _claim_records(root, rid):
    """Every claim-file record of the run as ``(path, payload)``."""
    return [(path, rec["payload"]) for path in R.iter_record_files(os.path.join(root, "records", "claims"))
            for _, rec, _ in R.read_record_file(path) if rec["payload"]["run_id"] == rid]


def test_a_stale_bound_claim_is_not_claimed_in_the_worklist_and_the_same_session_gets_a_fresh_id_bound_to_the_corrected_identity(tmp_path, capsys):
    """J claims under X; the run is corrected to X2 (same required judge). The claim is bound to X, so it can only complete into a
    judgment the identity rule excludes: the worklist reads `identity_stale` with no claim columns (never `claimed`), `registry
    pending-judgment` prints it, and J's own retry gets a FRESH id bound to X2 while the old claim reads `abandoned` under its own
    `claimed_utc` -- one file, both records; the new claim is then held as any current one."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    old = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    assert _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed" and _worklist_row(root, rid, synth.JUDGE_ID)["claim_judgment_id"] == old
    _correct_to_x2(root, rid, 3000)
    # the worklist: the stale-bound claim is not presented as held -- identity_stale, every claim column NULL, the claim row still listed
    row = _q(root, "SELECT pending_reason, claim_judgment_id, claim_session, claim_utc, claim_age_s, items_missing, existing_judgments_json "
                   "FROM v_pending_judgment WHERE run_id = ? AND required_judge_id = ?", rid, synth.JUDGE_ID)[0]
    assert row["pending_reason"] == "identity_stale" and row["items_missing"] is None
    assert row["claim_judgment_id"] is None and row["claim_session"] is None and row["claim_utc"] is None and row["claim_age_s"] is None
    assert [(j["judgment_id"], j["status"]) for j in json.loads(row["existing_judgments_json"])] == [(old, "in_progress")]
    assert _judgment_rows(root, rid)[old]["status"] == "in_progress" and _judgment_rows(root, rid)[old]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    assert _pending(root, rid)[synth.JUDGE_ID_2]["pending_reason"] == "awaiting_second_judge"      # the second-judge design is untouched
    # the CLI prints the reason
    assert CLI.main(["--root", root, "pending-judgment", "--judge", synth.JUDGE_ID]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(rid)]
    assert len(lines) == 1 and "\tidentity_stale\t" in lines[0] and "claimed" not in lines[0]
    # the same session retries: no ClaimHeld, no force -- a fresh id minted over the retry time, bound to X2; the old claim abandoned
    new = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    assert new != old and new == I.judgment_id(rid, synth.JUDGE_ID, "s-A", synth.t(4000))
    rows = _judgment_rows(root, rid)
    assert rows[old] == {"judgment_id": old, "status": "abandoned", "claimed_utc": synth.t(1000), "claimed_by_session": "s-A",
                         "claimed_instrument_id": synth.INSTRUMENT_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    assert rows[new] == {"judgment_id": new, "status": "in_progress", "claimed_utc": synth.t(4000), "claimed_by_session": "s-A",
                         "claimed_instrument_id": INSTRUMENT_X2_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    # the abandoned record restates the claim's own claimed_utc and no binding (the forced-takeover shape; its id reproduces under the
    # claim form), and it rides the same file as the new claim
    recs = _claim_records(root, rid)
    ab = [(path, pl) for path, pl in recs if pl["judgment_id"] == old and pl["status"] == "abandoned"]
    assert len(ab) == 1 and ab[0][1]["claimed_utc"] == synth.t(1000) and ab[0][1]["claimed_by_session"] == "s-A"
    assert "claimed_instrument_id" not in ab[0][1] and "claimed_required_judge_id" not in ab[0][1] and I.judgment_id_form(ab[0][1]) is not None
    assert {path for path, pl in recs if pl["judgment_id"] == new} == {ab[0][0]}
    # the worklist reads the live claim; the same session asking again gets the same id (a current-bound claim is held as before)
    assert _worklist_row(root, rid, synth.JUDGE_ID) == {"pending_reason": "claimed", "items_missing": None, "required_coverage": 4,
                                                        "n_complete_other_judges": 0, "is_instrument_judge": 1, "claim_judgment_id": new}
    assert P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4001)) == new
    with pytest.raises(P.ClaimHeld) as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(4002))
    assert ei.value.session == "s-A"
    # the new claim completes: the paper row
    _complete(root, rid, new, synth.JUDGE_ID, "s-A", 4000, 5000)
    assert _ingest(root, synth.t(6000)).refused == 0
    assert _frame(root, rid) == [(new, "asr")] and _coverage(root, rid) == "measured" and _worklist_row(root, rid, synth.JUDGE_ID) is None
    _rebuild_equal(root, tmp_path)


def test_another_session_takes_over_a_stale_bound_claim_without_force_while_a_current_bound_claim_is_held(tmp_path):
    """Before the correction s-B is refused (ClaimHeld, the rule as before); after it s-A's claim is bound to X while the run is on X2, so
    s-B takes the run over with no `force` and no stale age: s-A's claim reads `abandoned`, s-B's claim binds X2, and s-A is now the one
    refused. The worklist driver's own check (`active_claim` by the newest claimed_utc) sees s-B's claim."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    old = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(1001))
    _correct_to_x2(root, rid, 3000)
    new = P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(4000))                      # 3,000 s old: well inside stale_hours; no force
    assert new != old and new == I.judgment_id(rid, synth.JUDGE_ID, "s-B", synth.t(4000))
    rows = _judgment_rows(root, rid)
    assert (rows[old]["status"], rows[old]["claimed_by_session"], rows[old]["claimed_utc"]) == ("abandoned", "s-A", synth.t(1000))
    assert (rows[new]["status"], rows[new]["claimed_by_session"], rows[new]["claimed_instrument_id"], rows[new]["claimed_required_judge_id"]) == \
        ("in_progress", "s-B", INSTRUMENT_X2_ID, synth.JUDGE_ID)
    with pytest.raises(P.ClaimHeld) as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4001))
    assert ei.value.session == "s-B"
    assert not P.release(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4002))                # s-A holds nothing in progress any more
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert P.active_claim(conn, rid, synth.JUDGE_ID) == {"judgment_id": new, "claimed_by_session": "s-B", "claimed_utc": synth.t(4000)}
    conn.close()
    assert _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed" and _worklist_row(root, rid, synth.JUDGE_ID)["claim_judgment_id"] == new
    assert _q(root, "SELECT claim_session FROM v_pending_judgment WHERE run_id = ? AND required_judge_id = ?", rid, synth.JUDGE_ID) == [{"claim_session": "s-B"}]
    # a SECOND judge's claim binds the run's identity too: made after the correction it is current, and held for its session as before
    j2 = P.claim(root, rid, synth.JUDGE_ID_2, "s-C", now=synth.t(4100))
    assert _judgment_rows(root, rid)[j2]["claimed_instrument_id"] == INSTRUMENT_X2_ID
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID_2, "s-D", now=synth.t(4101))
    _rebuild_equal(root, tmp_path)


def test_completing_the_old_id_after_the_correction_is_non_current_everywhere_while_the_run_stays_pending_and_judged(tmp_path):
    """The claim bound to X is closed by the retry; its consumer then completes the OLD id anyway (the hook writes the claim it was handed).
    The judgment folds (a fact: status complete, the latest statement) and is non-current everywhere -- frame, item objects, headline,
    coverage -- while the run stays pending for J (`claimed`, by the new claim) and stays `judged`: the harvester's completion requirement
    reads identity-current complete judgments only, so the run is never `ingested` with no current reading. The new claim's completion
    under X2 is the paper row and ingests the run."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    old = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    _correct_to_x2(root, rid, 3000)
    new = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    assert _judgment_rows(root, rid)[old]["status"] == "abandoned"
    _complete(root, rid, old, synth.JUDGE_ID, "s-A", 1000, 5000)                            # the old consumer completes the old id
    assert _ingest(root, synth.t(6000)).refused == 0
    rows = _judgment_rows(root, rid)
    assert (rows[old]["status"], rows[old]["claimed_instrument_id"]) == ("complete", synth.INSTRUMENT_ID) and rows[new]["status"] == "in_progress"
    # non-current everywhere; the rows are kept
    assert _frame(root, rid) == [] and _items(root, "v_item_result_current", rid) == {} and _items(root, "v_item_result_provenances", rid) == {}
    assert _items(root, "v_item_headline", rid) == {} and _coverage(root, rid) == "absent"
    assert _q(root, "SELECT COUNT(*) AS n FROM item_result WHERE run_id = ?", rid) == [{"n": 4}]
    # the run stays pending for J: the live claim is the new one
    assert _worklist_row(root, rid, synth.JUDGE_ID) == {"pending_reason": "claimed", "items_missing": None, "required_coverage": 4,
                                                        "n_complete_other_judges": 0, "is_instrument_judge": 1, "claim_judgment_id": new}
    # ... and judged, not ingested: the stale complete judgment covers the arithmetic but not the identity
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [{"lifecycle_state": "judged"}]
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM judgment jx JOIN run r ON r.run_id = jx.run_id WHERE jx.judgment_id = ? AND {S.judgment_covers_sql('jx', 'r')}",
                            (old,)).fetchone()[0] == 1                                       # status and arithmetic alone would have held
        ok, why = H.completion_requirements_hold(conn, rid)
        assert not ok and "no complete judgment" in why and "identity_stale" in why and "1 bound at claim" in why
    assert _ingest(root, synth.t(6500)).harvester_records == 0                               # a further cycle derives nothing: the rule is stable
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [{"lifecycle_state": "judged"}]
    # the new claim completes under X2: the paper row, the run ingested, nothing pending for J
    _complete(root, rid, new, synth.JUDGE_ID, "s-A", 4000, 7000)
    assert _ingest(root, synth.t(8000)).refused == 0
    assert _frame(root, rid) == [(new, "asr")] and _items(root, "v_item_result_current", rid) == {new: 4} and _coverage(root, rid) == "measured"
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [{"lifecycle_state": "ingested"}]
    assert _worklist_row(root, rid, synth.JUDGE_ID) is None
    assert _q(root, "SELECT COUNT(*) AS n FROM item_result WHERE run_id = ?", rid) == [{"n": 8}]
    _rebuild_equal(root, tmp_path)


def test_a_stale_bound_complete_judgment_satisfies_no_roster_entry_and_the_entry_stays_a_gap(tmp_path):
    """A roster entry of the open round names instrument X2 and judge J. After the correction the run IS on X2 and carries a complete,
    covering judgment by J -- bound to X. Status and arithmetic alone would have matched the entry (and the run would have read in the
    paper with no frame row); the identity test leaves `v_roster_match` empty for the pair, the entry listed in `v_roster_gaps`, and the
    harvester holding the run at `judged`. The re-judge under X2 satisfies the entry, closes the gap and ingests the run."""
    entry = synth.rec("roster_entry", {"roster_entry_id": "re_x2", "round_id": "r_synth", "comparison_id": "fig_synth", "model_id": synth.MODEL_ID,
                                       "eval_id": synth.EVAL_ID, "instrument_id": INSTRUMENT_X2_ID, "judge_id": synth.JUDGE_ID, "object_id": "none",
                                       "strength_rule": "operating_point", "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}, synth.t(50))
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31)), entry])
    assert _q(root, "SELECT roster_entry_id FROM v_roster_gaps") == [{"roster_entry_id": "re_x2"}]
    old = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    _correct_to_x2(root, rid, 3000)
    _complete(root, rid, old, synth.JUDGE_ID, "s-A", 1000, 5000)                            # the stale-bound complete judgment
    assert _ingest(root, synth.t(6000)).refused == 0
    assert _judgment_rows(root, rid)[old]["status"] == "complete"
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM judgment jx JOIN run r ON r.run_id = jx.run_id WHERE jx.judgment_id = ? AND {S.judgment_covers_sql('jx', 'r')}",
                            (old,)).fetchone()[0] == 1                                       # the arithmetic covers: only the identity test excludes it
        ok, why = H.completion_requirements_hold(conn, rid)
        assert not ok and "identity_stale" in why
    assert _q(root, "SELECT run_id, roster_entry_id FROM v_roster_match") == []
    assert _q(root, "SELECT roster_entry_id FROM v_roster_gaps") == [{"roster_entry_id": "re_x2"}]
    assert _frame(root, rid) == [] and _coverage(root, rid) == "absent"
    assert _q(root, "SELECT roster_entry_ids_json FROM v_paper_current WHERE run_id = ?", rid) == [{"roster_entry_ids_json": "[]"}]
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [{"lifecycle_state": "judged"}]
    assert _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "identity_stale"
    # the re-judge under X2: the entry is satisfied, the gap closes, the frame row advertises the entry, the run ingests
    new = P.claim(root, rid, synth.JUDGE_ID, "s-A2", now=synth.t(7000))
    _complete(root, rid, new, synth.JUDGE_ID, "s-A2", 7000, 7100)
    assert _ingest(root, synth.t(8000)).refused == 0
    assert _q(root, "SELECT run_id, roster_entry_id, comparison_id FROM v_roster_match") == [{"run_id": rid, "roster_entry_id": "re_x2", "comparison_id": "fig_synth"}]
    assert _q(root, "SELECT roster_entry_id FROM v_roster_gaps") == []
    assert _frame(root, rid) == [(new, "asr")] and _coverage(root, rid) == "measured"
    assert _q(root, "SELECT roster_entry_ids_json FROM v_paper_long WHERE run_id = ?", rid) == [{"roster_entry_ids_json": '["re_x2"]'}]
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [{"lifecycle_state": "ingested"}]
    assert _worklist_row(root, rid, synth.JUDGE_ID) is None
    _rebuild_equal(root, tmp_path)


def _spool_correction(root, rid, instrument_id, i):
    """A same-id run identity correction onto ``instrument_id`` written to the spool and NOT folded -- where a producer writes a run's records
    (``records/<campaign>/<run_dir>/<seq>-run.jsonl``, ``emit.RecordWriter``): the directory ``pending.run_record_files`` scopes the claim's
    fold-first to."""
    corrected = synth.run_payload("k", i=100, instrument_id=instrument_id)
    corrected["run_id"] = rid
    return _write(root, os.path.join("c", I.run_dir_name(rid)), f"{i:06d}-run.jsonl", [synth.rec("run", corrected, synth.t(i))])


def _run_instrument(root, rid):
    return _q(root, "SELECT instrument_id FROM run WHERE run_id = ?", rid)[0]["instrument_id"]


def test_claim_folds_a_spooled_run_correction_first_and_returns_a_claim_bound_to_the_folded_identity(tmp_path):
    """A correction already on the spool when claim() runs is folded BEFORE the identity is read, so the
    claim returned is bound to the identity the fold left -- for a first claim, for another session meeting an existing claim (the
    fold makes that claim stale, so the run is taken over without force) and for the same session re-claiming (a fresh id)."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31)), synth.rec("instrument", INSTRUMENT_X3, synth.t(32)),
                                                 synth.rec("instrument", INSTRUMENT_X4, synth.t(33))])
    # a first claim with a correction spooled and unfolded: bound to X2, the run row folded to X2, no abandoned record
    _spool_correction(root, rid, INSTRUMENT_X2_ID, 3000)
    assert _run_instrument(root, rid) == synth.INSTRUMENT_ID                                   # still unfolded
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    assert _run_instrument(root, rid) == INSTRUMENT_X2_ID
    assert _judgment_rows(root, rid) == {c1: {"judgment_id": c1, "status": "in_progress", "claimed_utc": synth.t(4000), "claimed_by_session": "s-A",
                                              "claimed_instrument_id": INSTRUMENT_X2_ID, "claimed_required_judge_id": synth.JUDGE_ID}}
    assert _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed" and _worklist_row(root, rid, synth.JUDGE_ID)["claim_judgment_id"] == c1
    # another session, a further correction spooled: the fold first makes c1 stale, so s-B takes the run over without force, bound to X3
    _spool_correction(root, rid, INSTRUMENT_X3_ID, 5000)
    c2 = P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(6000))
    rows = _judgment_rows(root, rid)
    assert _run_instrument(root, rid) == INSTRUMENT_X3_ID and c2 != c1
    assert (rows[c1]["status"], rows[c1]["claimed_utc"]) == ("abandoned", synth.t(4000))
    assert (rows[c2]["status"], rows[c2]["claimed_by_session"], rows[c2]["claimed_instrument_id"]) == ("in_progress", "s-B", INSTRUMENT_X3_ID)
    # the same session, a further correction spooled: its own claim is stale after the fold, a fresh id bound to X4
    _spool_correction(root, rid, INSTRUMENT_X4_ID, 7000)
    c3 = P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(8000))
    rows = _judgment_rows(root, rid)
    assert _run_instrument(root, rid) == INSTRUMENT_X4_ID and c3 not in (c1, c2)
    assert rows[c2]["status"] == "abandoned" and (rows[c3]["status"], rows[c3]["claimed_by_session"], rows[c3]["claimed_instrument_id"]) == ("in_progress", "s-B", INSTRUMENT_X4_ID)
    # nothing spooled: the current-bound claim is held as before
    assert P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(8001)) == c3
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(8002))
    assert _worklist_row(root, rid, synth.JUDGE_ID)["claim_judgment_id"] == c3
    with DB.reading(DB.db_path(root)) as conn:
        assert P.active_claim(conn, rid, synth.JUDGE_ID)["judgment_id"] == c3
    _rebuild_equal(root, tmp_path)


def _is_claim_fold(root, kw):
    """Whether an ``ingest_once`` call is a claim's fold of its own record file (``only_files`` naming a file under ``records/claims/``)."""
    return any("claims" in os.path.relpath(f, root).split(os.sep) for f in (kw.get("only_files") or ()))


def test_claim_re_checks_its_binding_after_its_own_fold_and_re_mints_when_a_correction_landed_in_between(tmp_path, monkeypatch):
    """A correction that folds TOGETHER with the claim's own file moves the identity after the binding was read: the re-check under the
    lock abandons the just-minted claim and re-mints one millisecond later, bound to the folded identity; the id returned is the
    re-minted one and the worklist reads it as the claim. Every fold claim() runs is scoped (its own file, the run's spooled run
    records), and the lock keeps every other writer out, so the only way the identity moves between the two reads is a correction riding
    the claim's own fold -- which is what this test hands the scoped fold; the re-check is the belt that holds whatever the fold admitted."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    real = H.ingest_once
    calls = []

    def ingest_with_a_late_correction(root_, conn=None, **kw):
        calls.append(kw)
        if _is_claim_fold(root, kw) and sum(1 for c in calls if _is_claim_fold(root, c)) == 1:   # the fold of the claim's own file: a correction rides it
            kw = dict(kw, only_files=list(kw["only_files"]) + [_spool_correction(root, rid, INSTRUMENT_X2_ID, 3000)])
        return real(root_, conn, **kw)

    monkeypatch.setattr(H, "ingest_once", ingest_with_a_late_correction)
    got = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    monkeypatch.undo()
    first = I.judgment_id(rid, synth.JUDGE_ID, "s-A", synth.t(4000))
    later = R.epoch_ms_to_utc(R.utc_to_epoch_ms(synth.t(4000)) + 1)
    claim_folds = [c for c in calls if _is_claim_fold(root, c)]
    assert len(claim_folds) == 2 and all(len(c["only_files"]) >= 1 for c in calls)             # the claim's fold (with the correction riding it) and the re-mint's; every fold scoped
    assert got != first and got == I.judgment_id(rid, synth.JUDGE_ID, "s-A", later)
    rows = _judgment_rows(root, rid)
    assert rows[first] == {"judgment_id": first, "status": "abandoned", "claimed_utc": synth.t(4000), "claimed_by_session": "s-A",
                           "claimed_instrument_id": synth.INSTRUMENT_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    assert rows[got] == {"judgment_id": got, "status": "in_progress", "claimed_utc": later, "claimed_by_session": "s-A",
                         "claimed_instrument_id": INSTRUMENT_X2_ID, "claimed_required_judge_id": synth.JUDGE_ID}
    assert _run_instrument(root, rid) == INSTRUMENT_X2_ID
    assert _worklist_row(root, rid, synth.JUDGE_ID) == {"pending_reason": "claimed", "items_missing": None, "required_coverage": 4,
                                                        "n_complete_other_judges": 0, "is_instrument_judge": 1, "claim_judgment_id": got}
    with DB.reading(DB.db_path(root)) as conn:
        assert P.active_claim(conn, rid, synth.JUDGE_ID) == {"judgment_id": got, "claimed_by_session": "s-A", "claimed_utc": later}
    # the abandoned record of the first claim restates its own claimed_utc (its id reproduces), stamped a millisecond later with the re-mint
    ab = [pl for _, pl in _claim_records(root, rid) if pl["judgment_id"] == first and pl["status"] == "abandoned"]
    assert len(ab) == 1 and ab[0]["claimed_utc"] == synth.t(4000) and I.judgment_id_form(ab[0]) is not None
    # the same session asks again with nothing moving: the re-minted claim is returned
    assert P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(5000)) == got
    _rebuild_equal(root, tmp_path)


def test_claim_fails_loudly_when_the_identity_keeps_moving_under_the_lock(tmp_path, monkeypatch):
    """The re-mint is capped: an identity that moves after every fold (REBIND_CAP times) is a fault to report, never a stale claim to
    return; the last claim minted stands in_progress and the run is listed as claimed by it once the moving stops."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31)), synth.rec("instrument", INSTRUMENT_X3, synth.t(32)),
                                                 synth.rec("instrument", INSTRUMENT_X4, synth.t(33)), synth.rec("instrument", INSTRUMENT_Y, synth.t(34))])
    real = H.ingest_once
    moves = iter([INSTRUMENT_X2_ID, INSTRUMENT_X3_ID, INSTRUMENT_X4_ID, synth.INSTRUMENT_ID])
    calls = []

    def ingest_with_a_correction_every_time(root_, conn=None, **kw):
        calls.append(kw)
        if _is_claim_fold(root, kw):                                                           # every claim fold (the first and each re-mint) carries a fresh correction
            kw = dict(kw, only_files=list(kw["only_files"]) + [_spool_correction(root, rid, next(moves), 3000 + len(calls))])
        return real(root_, conn, **kw)

    monkeypatch.setattr(H, "ingest_once", ingest_with_a_correction_every_time)
    with pytest.raises(RuntimeError, match="identity moved under the claim lock"):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    monkeypatch.undo()
    assert sum(1 for c in calls if _is_claim_fold(root, c)) == 1 + P.REBIND_CAP               # the claim's fold, then one per re-mint; every fold scoped
    assert all(c.get("only_files") for c in calls)
    statuses = sorted(r["status"] for r in _judgment_rows(root, rid).values())
    assert statuses == ["abandoned"] * P.REBIND_CAP + ["in_progress"]
    _rebuild_equal(root, tmp_path)


def test_the_rules_are_rendered_at_every_paper_side_site_and_the_identity_rule_in_the_pending_queue():
    d744, d757 = S.judgment_displaced_by_complete_sql("jm", "r", "i"), S.judgment_identity_current_sql("jm", "r", "i")
    assert S.judgment_identity_current_sql("jcp", "r", "i") in d744 and "jcp.status = 'complete' AND" in d744     # the witness is identity-current
    for name in ("v_paper_long", "v_paper_current", "v_multi_source_rows"):
        assert d744 in S.VIEW_SQL[name] and d757 in S.VIEW_SQL[name] and S.non_current_judgment_sql("rm.judgment_id", "jm.status") in S.VIEW_SQL[name], name
    for name in ("v_item_result_current", "v_item_result_provenances"):
        sql = S.VIEW_SQL[name]
        assert S.item_judgment_live("item_result", "ir") in sql and "jcp.status = 'complete'" in sql and "jx.claimed_instrument_id" in sql, name
        assert "JOIN run rcp ON rcp.run_id = jcp.run_id" in sql and S.judgment_identity_current_sql("jcp", "rcp", "icp") in sql, name
        assert {"judgment", "run", "instrument"} <= set(S.VIEW_REFERENCES[name])
    assert "instrument" in S.VIEW_REFERENCES["v_multi_source_rows"] and "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id" in S.VIEW_SQL["v_multi_source_rows"]
    # the worklist: the identity test on the covering judgment, on items_missing and on the partial / coverage_short reasons, the
    # identity_stale reason between coverage_short and superseded; the displacing complete witness is NOT a worklist concern
    pending = S.VIEW_SQL["v_pending_judgment"]
    assert S.judgment_identity_current_sql("jc", "r", "i") in pending and S.judgment_identity_current_sql("j4", "r", "i") in pending
    assert pending.count(S.judgment_identity_current_sql("j7", "r", "i")) == 3 and "jcp." not in pending and "THEN 'partial'" in pending
    assert pending.index("THEN 'coverage_short'") < pending.index("THEN 'identity_stale'") < pending.index("THEN 'superseded'")
    assert "instrument" in S.VIEW_REFERENCES["v_pending_judgment"]
    # the active-claim join reads identity-current claims only, and a stale-bound claim reads identity_stale
    assert S.judgment_identity_current_sql("j3", "r", "i") in pending and "j7.status IN ('in_progress', 'partial', 'complete')" in pending
    # the other consumers of the coverage predicate conjoin the identity test: the roster match (three probes), the measurement-judge rule
    # (three covering branches, rendered over the caller's run / instrument aliases) and the harvester (its own test)
    roster = S.VIEW_SQL["v_roster_match"]
    assert roster.count(S.judgment_identity_current_sql("jx", "r", "i")) == 3 == roster.count(S.judgment_covers_sql("jx"))
    assert S.MEASUREMENT_JUDGE_SQL == S.measurement_judge_sql() and S.MEASUREMENT_JUDGE_SQL.count(S.judgment_identity_current_sql("jm", "r", "i")) == 3 == S.MEASUREMENT_JUDGE_SQL.count(S.judgment_covers_sql("jm"))
    assert S.measurement_judge_sql("rx", "ix").count(S.judgment_identity_current_sql("jm", "rx", "ix")) == 3 and S.MEASUREMENT_JUDGE_SQL in S.frame_admission_sql()
    assert S.judgment_currency_sql("rm.judgment_id", "jm", "r", "i") in S.frame_admission_sql()
    assert S.TABLE_BY_NAME["judgment"].column_names[-2:] == ["claimed_instrument_id", "claimed_required_judge_id"]
    assert S.TABLE_BY_NAME["judgment"].column("claimed_instrument_id").fk == ("instrument", "instrument_id")
    assert S.TABLE_BY_NAME["judgment"].column("claimed_required_judge_id").fk == ("judge", "judge_id")
    # the item-count contract is on the columns themselves
    assert "VERDICT" in S.TABLE_BY_NAME["judgment"].column("n_items_judged").why_null and "counted apart from n_items_judged" in S.TABLE_BY_NAME["judgment"].column("n_judge_refusals").why_null
