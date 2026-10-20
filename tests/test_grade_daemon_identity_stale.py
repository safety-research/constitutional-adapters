"""The daemon's spooled-hold fold check CLEARS, never quarantines, a run whose bound judgment folded while the run still pends because the
run identity it was bound to was corrected -- the registry's window re-pends the SAME (run, required judge) row with pending_reason
IDENTITY_STALE_REASON ("identity_stale"; the stale judgment's row is untouched, still in existing_judgments_json, but counts for nothing).
Before this, the check read any bound judgment folded `<> in_progress` while the run still pends as a coverage defect
(`_quarantine_folded_short`), so the identity-stale re-pend would have been quarantined as coverage_short instead of re-dispatched. Every
OTHER folded-while-pending case keeps the coverage_short path (pinned here beside the new one, and in tests/test_grade_daemon.py's binding
test).

The view does not produce `identity_stale` today (read-only census 2026-10-04: 0 rows), so the owner's future view is stood in for by a stub
on pending.pending that relabels ONE run's pending_reason; the row shape is otherwise the real one -- the run pends after its bound judgment
folded COMPLETE (the judgment's status unchanged, named in existing_judgments_json), which the real view gives for a complete short of the
run's coverage. Synthetic registry, synthetic ids."""
from __future__ import annotations

import json
import os
import sys

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth

import test_grade_daemon as TGD
from test_grade_daemon import _events, _load_daemon_module, _mirror_args, _root
from test_grade_daemon_schema_hold import _db, _grader_env, _tick_args

REPO = TGD.REPO
sys.path.insert(0, str(REPO / "scripts" / "registry"))


def _fold(root):
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0, rep.as_dict()


CORRECTED_INSTRUMENT = "1111111122222222333333334444444455555555666666667777777788888888"   # a synthetic 64-hex instrument id: the corrected identity


def _relabel(monkeypatch, run_id, reason, binding=None, others=None):
    """The registry's future view, stood in for: pending.pending (the daemon's `run_pending` / fold-check / quarantine read and the worklist's
    pending read) reports `reason` for ``run_id`` when the real view lists the run -- and, when ``binding`` is given, the CORRECTED identity
    the registry's predicate states (the row's instrument_id / required_judge_id after the correction); ``others`` = {run_id: {reason,
    binding}} for more runs. Every other row, filter and column is the real view's."""
    real = P.pending
    plan = {str(run_id): (reason, binding)}
    for rid, spec in (others or {}).items():
        plan[str(rid)] = (spec.get("reason"), spec.get("binding"))

    def stub(conn, judge_id=None, campaign=None, **filters):
        rows = real(conn, judge_id, campaign, **filters)
        for r in rows:
            spec = plan.get(str(r["run_id"]))
            if spec is None:
                continue
            rs, bd = spec
            if rs:
                r["pending_reason"] = rs
            if bd:
                r.update(bd)
        return rows

    monkeypatch.setattr(P, "pending", stub)
    return real


def _ledger(tmp_path, name):
    p = tmp_path / "state" / name
    return json.loads(p.read_text())["runs"] if p.exists() else {}


def test_a_bound_judgment_folded_under_a_corrected_identity_clears_the_hold_for_re_dispatch_while_a_short_one_still_quarantines(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    assert gd.IDENTITY_STALE_REASON == "identity_stale"
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1"); monkeypatch.setenv("FAKE_DRIVER_SHORT_ITEMS", "150")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_stale, r_short = rids["job7@none"], rids["job7@steer"]
    # tick 1: one dispatch grades both runs; the graders file COMPLETE judgments (150 of 200 items) on the spool, no fold: a hold bound to each
    s1 = gd.tick(args, *rest)
    ev1 = _events(tmp_path)
    (sp,) = [e for e in ev1 if e["event"] == "dispatch_spooled"]
    assert s1["dispatched"] == 2 and sorted(sp["run_ids"]) == sorted([r_stale, r_short])          # `dispatched` counts runs: both, in one grader
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    (j_stale,) = ledger[r_stale]["judgment_ids"]; (j_short,) = ledger[r_short]["judgment_ids"]
    # the fold: both bound judgments reach the registry COMPLETE (status unchanged from here on) and both runs still pend -- the real view says
    # coverage_short for both and names the judgment in existing_judgments_json; the owner's window would say identity_stale for the run whose
    # identity was corrected under its judgment: stood in for on r_stale alone
    _fold(root)
    with DB.reading(_db(root)) as conn:
        assert {tuple(r) for r in conn.execute("SELECT judgment_id, status FROM judgment WHERE judgment_id IN (?, ?)", (j_stale, j_short))} == {(j_stale, "complete"), (j_short, "complete")}
        pend = {x["run_id"]: x for x in P.pending(conn, None, synth.CAMPAIGN, run_id=(r_stale, r_short))}
    assert {rid: x["pending_reason"] for rid, x in pend.items()} == {r_stale: "coverage_short", r_short: "coverage_short"}
    assert [j["judgment_id"] for j in json.loads(pend[r_stale]["existing_judgments_json"]) if j["status"] == "complete"] == [j_stale]
    _relabel(monkeypatch, r_stale, gd.IDENTITY_STALE_REASON, binding={"instrument_id": CORRECTED_INSTRUMENT})   # the identity corrected under the judgment
    monkeypatch.delenv("FAKE_DRIVER_SHORT_ITEMS")                                # the re-dispatch judges the run in full
    # tick 2: the identity-stale run is CLEARED by name and re-dispatched in the same tick; the other folded-while-pending run is quarantined
    # as before (the unchanged coverage_short path); the quarantine ledger never names the cleared run
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev1):]
    (st,) = [e for e in ev2 if e["event"] == "spooled_hold_identity_stale"]
    assert (st["run_id"], st["judgment_ids"], st["pending_reason"]) == (r_stale, [j_stale], gd.IDENTITY_STALE_REASON)
    assert [(f["judgment_id"], f["judge_id"], f["status"], f["n_items_judged"], f["stated_binding"]) for f in st["folded"]] == \
        [(j_stale, synth.JUDGE_ID, "complete", 150, {"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID})]   # the claim states the binding since schema 23
    assert st["verified"] is True and st["required_judge_id"] == synth.JUDGE_ID and st["current_binding"]["instrument_id"] == CORRECTED_INSTRUMENT
    assert st["bound_binding"]["instrument_id"] != CORRECTED_INSTRUMENT and st["bound_binding"]["required_judge_id"] == synth.JUDGE_ID       # the binding differs
    cleared = {e["run_id"]: e for e in ev2 if e["event"] == "spooled_hold_cleared"}
    assert list(cleared) == [r_stale] and cleared[r_stale]["reason"] == "identity_stale_rebind" and gd.IDENTITY_STALE_REASON in cleared[r_stale]["note"]
    (rel,) = [e for e in ev2 if e["event"] == "spooled_hold_released"]
    assert rel["run_id"] == r_short and rel["reason"] == "folded_still_pending" and rel["quarantined"] and "coverage_short" in rel["defect"] and "n_items_judged=150" in rel["defect"]
    assert [sorted(e["run_ids"]) for e in ev2 if e["event"] == "dispatch"] == [[r_stale]]
    assert (s2["dispatched"], s2["quarantined"], s2["spooled_hold"]) == (1, 1, 0)
    q = _ledger(tmp_path, "dispatch_quarantine.json")
    assert set(q) == {r_short} and q[r_short]["rc"] == "coverage_short" and q[r_short]["judgment_ids"] == [j_short]
    sp2 = [e for e in ev2 if e["event"] == "dispatch_spooled"]
    assert sp2 and sp2[-1]["run_ids"] == [r_stale]
    ledger2 = _ledger(tmp_path, "dispatch_spooled.json")
    assert set(ledger2) == {r_stale} and ledger2[r_stale]["judgment_ids"] and ledger2[r_stale]["judgment_ids"] != [j_stale]    # a NEW hold, bound to the fresh judgment
    # the fresh judgment covers the run (200 of 200): after the fold the run leaves the pending view and tick 3 clears the new hold as folded;
    # the quarantined run stays quarantined, nothing is dispatched
    _fold(root)
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev1) + len(ev2):]
    assert [(e["run_id"], e["reason"]) for e in ev3 if e["event"] == "spooled_hold_cleared"] == [(r_stale, "folded")]
    assert not [e for e in ev3 if e["event"] in ("dispatch", "spooled_hold_identity_stale", "spooled_hold_released")]
    assert (s3["dispatched"], s3["quarantined"]) == (0, 1) and set(_ledger(tmp_path, "dispatch_quarantine.json")) == {r_short} and _ledger(tmp_path, "dispatch_spooled.json") == {}
    with DB.reading(_db(root)) as conn:
        assert [r["run_id"] for r in P.pending(conn, None, synth.CAMPAIGN, run_id=(r_stale, r_short))] == [r_short]


def test_a_dry_run_reports_the_identity_stale_clear_without_writing_the_ledger(tmp_path, monkeypatch):
    """The unit: `spooled_filter` on a hand-written hold whose judgment folded and whose run reads identity_stale -- cleared (eligible), the event
    named; under --dry-run the same decision is reported and the ledger is left as it was."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1"); monkeypatch.setenv("FAKE_DRIVER_SHORT_ITEMS", "150")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_stale = rids["job7@none"]
    gd.tick(args, *rest)
    _fold(root)
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    (j_stale,) = ledger[r_stale]["judgment_ids"]
    _relabel(monkeypatch, r_stale, gd.IDENTITY_STALE_REASON, binding={"instrument_id": CORRECTED_INSTRUMENT})
    st = str(tmp_path / "state")
    rows = gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    n_ev = len(_events(tmp_path))
    eligible, held = gd.spooled_filter(st, root, rows, dry_run=True)
    assert r_stale in [r["run_id"] for r in eligible] and r_stale not in [r["run_id"] for r in held]
    assert _ledger(tmp_path, "dispatch_spooled.json") == ledger                                      # a dry run writes nothing
    ev = _events(tmp_path)[n_ev:]
    assert [e["run_id"] for e in ev if e["event"] == "spooled_hold_identity_stale"] == [r_stale]
    assert [(e["run_id"], e["reason"]) for e in ev if e["event"] == "spooled_hold_cleared"] == [(r_stale, "identity_stale_rebind")]
    assert not [e for e in ev if e["event"] == "spooled_hold_released" and e["run_id"] == r_stale]       # the other held run's coverage_short release is its own business
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()                                 # a dry run quarantines nothing either
    eligible, held = gd.spooled_filter(st, root, rows, dry_run=False)
    assert r_stale in [r["run_id"] for r in eligible] and _ledger(tmp_path, "dispatch_spooled.json") == {}   # the real pass pops the hold


# ------------------------------------------------------------------------------------------------
# The identity BINDING is the pair the pending view states for a run -- instrument_id and required_judge_id. The spooled hold and the
# quarantine record it at filing time; the identity_stale clear is VERIFIED against it (the folded judgment must be the required judge's and
# made under a binding that differs from the run's current one); the quarantine releases on a verified change of it for the same run.


def _write_wrong_judge_complete(root, rid, judge, n_items=200):
    """A COMPLETE judgment by ANOTHER judge for the run (a plain-form id), folded: the wrong-judge completion the hold may have bound."""
    from sisteer.registry import emit, identity as I
    jid = I.judgment_id(rid, judge)
    w = emit.RecordWriter(H.records_dir(root), synth.CAMPAIGN, rid, emit.producer_block("tests/wrong_judge_grader.py", session="s-wrong"), disabled=False)
    w.emit_judgment({"judgment_id": jid, "run_id": rid, "judge_id": judge, "status": "complete", "started_utc": "2026-09-17T00:00:00.000Z",
                     "finished_utc": "2026-09-17T00:00:00.000Z", "n_items_judged": n_items, "api_pool": "lowprio", "store_uri": "synthetic:///wrong"})
    _fold(root)
    return jid


def test_a_quarantined_run_whose_identity_binding_is_corrected_is_released_once_and_named_while_an_unchanged_one_stays(tmp_path, monkeypatch):
    """A run already in dispatch_quarantine.json (folded coverage-short) whose instrument identity alone is later corrected is re-pended as
    identity_stale, but only the spooled hold knew the reason; the quarantine released on daemon sha / store generation / the release file / not
    pending -- none of which an identity correction moves -- so the run stayed pending-but-ungraded for the daemon sha's lifetime. Now the
    quarantine record carries the run's identity binding at filing time (and each folded judgment's judge and binding) and releases on a
    VERIFIED change of that binding for the same run, naming the old and the new; a run whose binding did not change stays quarantined."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1"); monkeypatch.setenv("FAKE_DRIVER_SHORT_ITEMS", "150")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_corr, r_same = rids["job7@none"], rids["job7@steer"]
    gd.tick(args, *rest)                                                      # both graded complete SHORT, spooled, holds bound
    _fold(root)
    s2 = gd.tick(args, *rest)                                                 # both folded coverage_short -> quarantined (the unchanged path)
    ev2 = _events(tmp_path)
    q = _ledger(tmp_path, "dispatch_quarantine.json")
    assert set(q) == {r_corr, r_same} and s2["quarantined"] == 2 and all(q[r]["rc"] == "coverage_short" for r in q)
    with DB.reading(_db(root)) as conn:
        inst = {r["run_id"]: r["instrument_id"] for r in P.pending(conn, None, synth.CAMPAIGN, run_id=(r_corr, r_same))}
    for r in (r_corr, r_same):                                                # the record carries the binding at filing time and the judgment's judge / binding
        assert q[r]["identity_binding"] == {"instrument_id": inst[r], "required_judge_id": synth.JUDGE_ID}
        (jb,) = q[r]["judgment_bindings"]
        assert jb["judgment_id"] == q[r]["judgment_ids"][0] and jb["judge_id"] == synth.JUDGE_ID and jb["binding"] == q[r]["identity_binding"]
    # the owner's window: r_corr's instrument identity is corrected (the same required judge), the row re-pends as identity_stale; r_same is
    # re-pended under the same reason WITHOUT a binding change (the reason alone is never a release)
    _relabel(monkeypatch, r_corr, gd.IDENTITY_STALE_REASON, binding={"instrument_id": CORRECTED_INSTRUMENT}, others={r_same: {"reason": gd.IDENTITY_STALE_REASON}})
    monkeypatch.delenv("FAKE_DRIVER_SHORT_ITEMS")
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev2):]
    rel = [e for e in ev3 if e["event"] == "quarantine_released"]
    assert [(e["run_id"], e["reason"]) for e in rel] == [(r_corr, "identity_rebound")]
    assert rel[0]["was_binding"] == {"instrument_id": inst[r_corr], "required_judge_id": synth.JUDGE_ID}
    assert rel[0]["binding"] == {"instrument_id": CORRECTED_INSTRUMENT, "required_judge_id": synth.JUDGE_ID} and rel[0]["judgment_ids"] == q[r_corr]["judgment_ids"]
    assert [sorted(e["run_ids"]) for e in ev3 if e["event"] == "dispatch"] == [[r_corr]]                  # re-dispatched once, against the current identity
    assert (s3["dispatched"], s3["quarantined"]) == (1, 1) and set(_ledger(tmp_path, "dispatch_quarantine.json")) == {r_same}
    # tick 4: nothing changed for r_same -- still quarantined, no release, nothing dispatched (the identity_stale reason alone moves nothing)
    _fold(root)
    s4 = gd.tick(args, *rest)
    ev4 = _events(tmp_path)[len(ev2) + len(ev3):]
    assert not [e for e in ev4 if e["event"] in ("quarantine_released", "dispatch")] and (s4["dispatched"], s4["quarantined"]) == (0, 1)
    assert set(_ledger(tmp_path, "dispatch_quarantine.json")) == {r_same}


def test_a_wrong_judge_completion_bound_in_the_hold_is_quarantined_not_cleared_even_when_the_run_reads_identity_stale(tmp_path, monkeypatch):
    """The registry's identity_stale predicate is existential over ANY old judgment of the required judge, and the hold binds EVERY complete
    judgment the grader spooled regardless of judge -- so an earlier stale required-judge judgment plus a newly folded WRONG-JUDGE completion
    read identity_stale and the previous branch would have cleared the hold instead of quarantining the failed dispatch. Now the clear is
    verified: the folded judgment must be the required judge's and made under a binding that differs from the current identity; here one bound
    judgment is another judge's, so the coverage_short path decides and the event names both judges. The right-judge case stays a clear."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1"); monkeypatch.setenv("FAKE_DRIVER_SHORT_ITEMS", "150")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_wrong, r_right = rids["job7@none"], rids["job7@steer"]
    gd.tick(args, *rest)
    _fold(root)
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    (j_wrong_stale,) = ledger[r_wrong]["judgment_ids"]; (j_right,) = ledger[r_right]["judgment_ids"]
    # a WRONG-JUDGE completion for r_wrong, folded, and bound in its hold beside the required judge's (what spooled_hold_add does when both are in
    # the spool at hold time: the ledger is put in that state directly)
    j_other = _write_wrong_judge_complete(root, r_wrong, synth.JUDGE_ID_2)
    ledger[r_wrong]["judgment_ids"] = sorted([j_wrong_stale, j_other])
    gd.spooled_save(str(tmp_path / "state"), ledger)
    # both runs read identity_stale with a corrected instrument (the owner's predicate on their stale required-judge judgment)
    _relabel(monkeypatch, r_wrong, gd.IDENTITY_STALE_REASON, binding={"instrument_id": CORRECTED_INSTRUMENT},
             others={r_right: {"reason": gd.IDENTITY_STALE_REASON, "binding": {"instrument_id": CORRECTED_INSTRUMENT}}})
    monkeypatch.delenv("FAKE_DRIVER_SHORT_ITEMS")
    st = str(tmp_path / "state")
    rows = gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    n_ev = len(_events(tmp_path))
    eligible, held = gd.spooled_filter(st, root, rows, dry_run=False)
    ev = _events(tmp_path)[n_ev:]
    assert sorted(r["run_id"] for r in eligible) == sorted([r_right, r_wrong]) and held == []    # both holds are gone: one cleared, one released INTO the quarantine
    eligible2, held2 = gd.quarantine_filter(st, root, eligible, dry_run=False)                    # ...which holds the wrong-judge run (its recorded binding IS the current one: no rebound)
    assert [r["run_id"] for r in eligible2] == [r_right] and [r["run_id"] for r in held2] == [r_wrong]
    stale = {e["run_id"]: e for e in ev if e["event"] == "spooled_hold_identity_stale"}
    assert stale[r_right]["verified"] is True and [f["judge_id"] for f in stale[r_right]["folded"]] == [synth.JUDGE_ID]
    assert stale[r_wrong]["verified"] is False and sorted(f["judge_id"] for f in stale[r_wrong]["folded"]) == sorted([synth.JUDGE_ID, synth.JUDGE_ID_2])
    assert stale[r_wrong]["required_judge_id"] == synth.JUDGE_ID and synth.JUDGE_ID_2 in stale[r_wrong]["verdict"] and j_other in stale[r_wrong]["verdict"]
    cleared = {e["run_id"]: e["reason"] for e in ev if e["event"] == "spooled_hold_cleared"}
    assert cleared == {r_right: "identity_stale_rebind"}
    (rel,) = [e for e in ev if e["event"] == "spooled_hold_released"]
    assert rel["run_id"] == r_wrong and rel["reason"] == "folded_still_pending" and rel["quarantined"]
    assert "identity_stale not accepted" in rel["defect"] and synth.JUDGE_ID_2 in rel["defect"] and f"required_judge_id={synth.JUDGE_ID}" in rel["defect"]
    q = _ledger(tmp_path, "dispatch_quarantine.json")
    assert set(q) == {r_wrong} and q[r_wrong]["rc"] == "coverage_short" and sorted(q[r_wrong]["judgment_ids"]) == sorted([j_wrong_stale, j_other])
    assert sorted(b["judge_id"] for b in q[r_wrong]["judgment_bindings"]) == sorted([synth.JUDGE_ID, synth.JUDGE_ID_2])
    assert q[r_wrong]["identity_binding"] == {"instrument_id": CORRECTED_INSTRUMENT, "required_judge_id": synth.JUDGE_ID}
    # a hold whose binding did NOT change under an identity_stale reason is not cleared either: the unit on the verdict
    cur = {"instrument_id": "inst-a", "required_judge_id": synth.JUDGE_ID}
    ok, why = gd.identity_stale_clear_verdict([{"judgment_id": "j1", "run_id": "run-r", "judge_id": synth.JUDGE_ID, "stated_binding": None}], run_id="run-r",
                                              required_judge=synth.JUDGE_ID, current=cur, bound=dict(cur))
    assert ok is False and "did not change" in why
    ok, why = gd.identity_stale_clear_verdict([{"judgment_id": "j1", "run_id": "run-r", "judge_id": synth.JUDGE_ID, "stated_binding": None}], run_id="run-r",
                                              required_judge=synth.JUDGE_ID, current=cur, bound=None)
    assert ok is False and "no binding evidence" in why
    ok, why = gd.identity_stale_clear_verdict([{"judgment_id": "j1", "run_id": "run-r", "judge_id": synth.JUDGE_ID, "stated_binding": {"instrument_id": "inst-old", "required_judge_id": synth.JUDGE_ID}}],
                                              run_id="run-r", required_judge=synth.JUDGE_ID, current=cur, bound=dict(cur))
    assert ok is True and "stated binding" in why and "run run-r's" in why    # the judgment's own stated binding (the owner's columns) wins over the daemon's


# ------------------------------------------------------------------------------------------------
# The verdict checks the RUN too: a hold binds every complete judgment id in the run's spool directory (pending.spool_complete_judgments reads
# the directory without inspecting the payload's run_id), and a grouped grader can write another run's completion through this run's
# RecordWriter (emit.RecordWriter._with_run keeps an explicit foreign run_id), so a VALID foreign-run judgment -- same judge, differing binding
# -- would have cleared this run's hold for repeated grading and the event would have claimed this run's judgment folded.


def _write_completion_through_other_runs_writer(root, writer_run, judged_run, judge, n_items=200):
    """Run `judged_run`'s otherwise-valid COMPLETE judgment written through run `writer_run`'s RecordWriter -- the record lands in writer_run's spool
    directory with payload run_id = judged_run (emit.RecordWriter._with_run keeps it), the harvester folds it under judged_run."""
    from sisteer.registry import emit, identity as I
    jid = I.judgment_id(judged_run, judge, "s-grouped", "2026-09-17T00:00:00.000Z")
    w = emit.RecordWriter(H.records_dir(root), synth.CAMPAIGN, writer_run, emit.producer_block("tests/grouped_grader.py", session="s-grouped"), disabled=False)
    w.emit_judgment({"judgment_id": jid, "run_id": judged_run, "judge_id": judge, "status": "complete", "started_utc": "2026-09-17T00:00:00.000Z",
                     "finished_utc": "2026-09-17T00:00:00.000Z", "n_items_judged": n_items, "api_pool": "lowprio", "store_uri": "synthetic:///grouped",
                     "claimed_by_session": "s-grouped", "claimed_utc": "2026-09-17T00:00:00.000Z"})
    _fold(root)
    return jid


def test_a_foreign_run_judgment_bound_in_the_hold_is_quarantined_not_cleared_even_when_judge_and_binding_would_pass(tmp_path, monkeypatch):
    """Run S's valid completion written through run R's writer sits in R's spool directory; R's hold (which binds every
    complete judgment id it finds there) binds S's id; the harvester folds it under S. With S's judge == R's required judge and a binding that
    differs from R's corrected identity, the previous verdict cleared R for another grading and the event said R's judgment had folded. Now the
    verdict requires every folded judgment's run_id to be the held run: R is quarantined (coverage_short, the defect and the event naming both
    runs); the right-run sibling -- the same shape with run_id == R -- still clears."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1"); monkeypatch.setenv("FAKE_DRIVER_SHORT_ITEMS", "150")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_held, s_other = rids["job7@none"], rids["job7@steer"]
    gd.tick(args, *rest)                                                      # both graded complete SHORT, spooled, holds bound to each own judgment
    st = str(tmp_path / "state")
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    (j_r,) = ledger[r_held]["judgment_ids"]
    # the permissive writer path: S's valid completion through R's RecordWriter -> R's directory; the hold re-bound from that directory the way
    # spooled_hold_add collects it (every complete judgment record there, whatever run_id the payload names) now binds S's id beside R's own
    j_s = _write_completion_through_other_runs_writer(root, r_held, s_other, synth.JUDGE_ID)
    rows_now = {r["run_id"]: r for r in gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})}
    gd.spooled_hold_add(st, group=[rows_now[r_held]], family="static", model_key="synthmodel_4b_u0", campaign=synth.CAMPAIGN, log_path=ledger[r_held]["dispatch_log"],
                        registry_root=root, dispatch_utc=None)
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    assert sorted(ledger[r_held]["judgment_ids"]) == sorted([j_r, j_s])
    with DB.reading(_db(root)) as conn:
        assert tuple(conn.execute("SELECT run_id, judge_id, status FROM judgment WHERE judgment_id = ?", (j_s,)).fetchone()) == (s_other, synth.JUDGE_ID, "complete")
        rows = {r["judgment_id"]: r for r in gd.judgment_rows_with_binding(conn, [j_r, j_s])}
    assert rows[j_r]["run_id"] == r_held and rows[j_s]["run_id"] == s_other                # the rows carry the run each judgment folded under
    # R re-pends identity_stale under a corrected instrument (the owner's predicate on its stale judgment); S is covered by its valid completion
    _relabel(monkeypatch, r_held, gd.IDENTITY_STALE_REASON, binding={"instrument_id": CORRECTED_INSTRUMENT})
    monkeypatch.delenv("FAKE_DRIVER_SHORT_ITEMS")
    rows = gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    # the right-run sibling first: with only R's OWN judgment bound (the same shape, run_id == R) the clear still verifies -- a dry pass, nothing saved
    ledger[r_held]["judgment_ids"] = [j_r]; gd.spooled_save(st, ledger)
    n_ev = len(_events(tmp_path))
    gd.spooled_filter(st, root, rows, dry_run=True)
    (ok_ev,) = [e for e in _events(tmp_path)[n_ev:] if e["event"] == "spooled_hold_identity_stale" and e["run_id"] == r_held]
    assert ok_ev["verified"] is True and ok_ev["foreign_run_ids"] == [] and [f["run_id"] for f in ok_ev["folded"]] == [r_held]
    assert set(_ledger(tmp_path, "dispatch_spooled.json")) >= {r_held}                   # a dry run clears nothing on disk
    # now with S's foreign-run judgment bound beside it: NOT cleared -- released into the quarantine, both runs named
    ledger[r_held]["judgment_ids"] = sorted([j_r, j_s]); gd.spooled_save(st, ledger)
    n_ev = len(_events(tmp_path))
    eligible, held = gd.spooled_filter(st, root, rows, dry_run=False)
    ev = _events(tmp_path)[n_ev:]
    (stale,) = [e for e in ev if e["event"] == "spooled_hold_identity_stale" and e["run_id"] == r_held]
    assert stale["verified"] is False and stale["foreign_run_ids"] == [s_other] and sorted(f["run_id"] for f in stale["folded"]) == sorted([r_held, s_other])
    assert "foreign-run judgment" in stale["verdict"] and j_s in stale["verdict"] and s_other in stale["verdict"] and r_held in stale["verdict"]
    assert [(e["run_id"], e["reason"]) for e in ev if e["event"] == "spooled_hold_released"] == [(r_held, "folded_still_pending")]
    (rel,) = [e for e in ev if e["event"] == "spooled_hold_released"]
    assert "identity_stale not accepted" in rel["defect"] and "foreign-run judgment" in rel["defect"] and s_other in rel["defect"] and f"run {s_other}, judge" in rel["defect"]
    assert r_held not in [e["run_id"] for e in ev if e["event"] == "spooled_hold_cleared"]
    q = _ledger(tmp_path, "dispatch_quarantine.json")
    assert set(q) == {r_held} and q[r_held]["rc"] == "coverage_short" and sorted(q[r_held]["judgment_ids"]) == sorted([j_r, j_s])
    assert sorted((b["judgment_id"], b["run_id"]) for b in q[r_held]["judgment_bindings"]) == sorted([(j_r, r_held), (j_s, s_other)])
    eligible2, held2 = gd.quarantine_filter(st, root, eligible, dry_run=False)
    assert r_held in [r["run_id"] for r in held2]                                         # quarantined, not re-dispatched
    # the pure verdict: a foreign run_id refuses by name even when the judge and the binding would pass
    cur = {"instrument_id": CORRECTED_INSTRUMENT, "required_judge_id": synth.JUDGE_ID}
    old = {"instrument_id": "inst-old", "required_judge_id": synth.JUDGE_ID}
    ok, why = gd.identity_stale_clear_verdict([{"judgment_id": "j-s", "run_id": "run-s", "judge_id": synth.JUDGE_ID, "stated_binding": dict(old)}],
                                              run_id="run-r", required_judge=synth.JUDGE_ID, current=cur, bound=dict(old))
    assert ok is False and "foreign-run judgment j-s folded under run run-s" in why and "run run-r's hold" in why
    ok, why = gd.identity_stale_clear_verdict([{"judgment_id": "j-r", "run_id": "run-r", "judge_id": synth.JUDGE_ID, "stated_binding": dict(old)}],
                                              run_id="run-r", required_judge=synth.JUDGE_ID, current=cur, bound=dict(old))
    assert ok is True and "run run-r's" in why


# ------------------------------------------------------------------------------------------------
# A run's REQUIRED JUDGE moves J1 -> J2 (the registry's same-id `run` record onto an instrument naming J2) while J1 is grading: the hold is
# bound to {X, J1}; J1's completion folds; the view lists the run for J2 as `awaiting_second_judge` (J1's judgment is another judge's there),
# never `identity_stale`, so the identity_stale branch never saw it and the fold was quarantined coverage_short with the CORRECTED binding
# recorded -- after which the quarantine's identity_rebound release could never fire and J2 was never dispatched without an operator release.
# Now the binding change itself is the trigger (`spooled_hold_binding_changed`): the same three checks (this run's judgments, the judge required
# AT DISPATCH, a binding that differs from the current one) verify the clear whatever the reason says; an unchanged binding keeps the
# coverage_short path. The correction is REAL here (no stub on the view): the registry's own `awaiting_second_judge`.

INSTRUMENT_J2 = synth.instrument_fields(judge_id=synth.JUDGE_ID_2, notes="the corrected instrument: its declared judge is the second judge")
INSTRUMENT_J2_ID = str(INSTRUMENT_J2["instrument_id"])
assert INSTRUMENT_J2_ID != synth.INSTRUMENT_ID


def _move_required_judge(root, rid, run_key, object_key, object_id, i0):
    """The owner's identity correction: INSTRUMENT_J2 registered, then a same-id `run` record moving the run onto it -- the run's required
    judge becomes JUDGE_ID_2. The store fields are restated unchanged (the hold's store generation must not move, or the hold would clear as
    `store_regenerated` before the fold check). Folded; returns the run's pending row for J2."""
    R.write_records(os.path.join(root, "records", "dims", "000002-dims-j2.jsonl"), [synth.rec("instrument", INSTRUMENT_J2, synth.t(2990))])
    _fold(root)
    corrected = synth.run_payload(run_key, object_key, object_id, i0, model_id=synth.MODEL_ID, view_legacy="synthmodel_4b_u0", instrument_id=INSTRUMENT_J2_ID,
                                  store_uri=f"/cluster_a/user/exp/stores/{run_key.replace('@', '_')}")
    corrected["run_id"] = rid
    R.write_records(os.path.join(root, "records", "c", "000011-correction.jsonl"), [synth.rec("run", corrected, synth.t(3000))])
    _fold(root)
    with DB.reading(_db(root)) as conn:
        (row,) = P.pending(conn, None, synth.CAMPAIGN, run_id=rid)
    assert (row["instrument_id"], row["required_judge_id"], row["is_instrument_judge"]) == (INSTRUMENT_J2_ID, synth.JUDGE_ID_2, 1), row
    return row


def test_a_required_judge_correction_landing_while_the_judge_grades_clears_the_hold_on_the_binding_change_and_the_new_judge_is_dispatched(tmp_path, monkeypatch):
    """The finding, end to end on the real view: J1 grades the run IN FULL (200 of 200), the correction J1 -> J2 lands while the completion is on
    the spool, the completion folds and the run pends for J2 as `awaiting_second_judge`. Before: quarantined coverage_short (the corrected binding
    recorded, J2 never dispatched). Now: `spooled_hold_binding_changed` verified, the hold cleared `binding_changed_rebind`, nothing quarantined,
    the run dispatched for J2 in the same tick, J2's completion covers it and the fresh hold clears `folded`."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1")   # full judgments: a fold under the OLD identity would cover
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_moved, r_other = rids["job7@none"], rids["job7@steer"]
    s1 = gd.tick(args, *rest)                                                 # tick 1: J1 grades both runs in full; their completions sit on the spool, a hold bound to each
    ev1 = _events(tmp_path)
    assert s1["dispatched"] == 2
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    (j1,) = ledger[r_moved]["judgment_ids"]
    assert ledger[r_moved]["identity_binding"] == {"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID}      # bound at dispatch: {X, J1}
    # the correction lands while J1's completion is in flight: the run's required judge is J2 now
    _move_required_judge(root, r_moved, "job7@none", "none", "none", 100)
    _fold(root)                                                               # the completions fold: r_other is covered; r_moved pends for J2 with J1's completion as another judge's
    with DB.reading(_db(root)) as conn:
        pend = {x["run_id"]: x for x in P.pending(conn, None, synth.CAMPAIGN, run_id=(r_moved, r_other))}
        jrow = conn.execute("SELECT status, judge_id, n_items_judged FROM judgment WHERE judgment_id = ?", (j1,)).fetchone()
    assert list(pend) == [r_moved] and pend[r_moved]["pending_reason"] == "awaiting_second_judge" and pend[r_moved]["required_judge_id"] == synth.JUDGE_ID_2
    assert tuple(jrow) == ("complete", synth.JUDGE_ID, 200) and pend[r_moved]["n_complete_other_judges"] == 1         # J1's full completion, counted as another judge's
    st = str(tmp_path / "state")
    rows = gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    assert [r["run_id"] for r in rows] == [r_moved] and rows[0]["required_judge_id"] == synth.JUDGE_ID_2
    # a dry pass first: the decision reported, the event distinct from identity_stale, nothing written
    n_ev = len(_events(tmp_path))
    eligible, held = gd.spooled_filter(st, root, rows, dry_run=True)
    ev = _events(tmp_path)[n_ev:]
    (bc,) = [e for e in ev if e["event"] == "spooled_hold_binding_changed"]
    assert bc["run_id"] == r_moved and bc["verified"] is True and bc["pending_reason"] == "awaiting_second_judge" and bc["judgment_ids"] == [j1]
    assert bc["bound_binding"] == {"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID}
    assert bc["current_binding"] == {"instrument_id": INSTRUMENT_J2_ID, "required_judge_id": synth.JUDGE_ID_2}
    assert (bc["bound_required_judge_id"], bc["required_judge_id"]) == (synth.JUDGE_ID, synth.JUDGE_ID_2)                 # the old and the new required judge, both named
    assert [(f["judgment_id"], f["run_id"], f["judge_id"], f["status"], f["n_items_judged"]) for f in bc["folded"]] == [(j1, r_moved, synth.JUDGE_ID, "complete", 200)]
    assert bc["foreign_run_ids"] == [] and "the judge required at dispatch" in bc["verdict"] and synth.JUDGE_ID in bc["verdict"]
    assert not [e for e in ev if e["event"] == "spooled_hold_identity_stale"]
    assert [e["reason"] for e in ev if e["event"] == "spooled_hold_cleared" and e["run_id"] == r_moved] == ["binding_changed_rebind"]
    assert not [e for e in ev if e["event"] == "spooled_hold_released"] and r_moved in [r["run_id"] for r in eligible] and held == []
    assert _ledger(tmp_path, "dispatch_spooled.json") == ledger and not (tmp_path / "state" / "dispatch_quarantine.json").exists()   # a dry run writes nothing
    # tick 2: cleared by name, nothing quarantined, and the run dispatched for J2 in the same tick (the fresh claim binds {Y, J2})
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev1):]
    assert [e["reason"] for e in ev2 if e["event"] == "spooled_hold_cleared" and e["run_id"] == r_moved] == ["binding_changed_rebind"] * 2   # the dry pass + the real one
    assert [e["reason"] for e in ev2 if e["event"] == "spooled_hold_cleared" and e["run_id"] == r_other] == ["folded"] * 2
    cl = [e for e in ev2 if e["event"] == "spooled_hold_cleared" and e["run_id"] == r_moved][-1]                      # the real pass (the dry pass logged the same)
    assert "DAEMONQUARFIX" in cl["note"] and "counts for nothing" in cl["note"]
    assert not [e for e in ev2 if e["event"] in ("spooled_hold_released", "spooled_hold_identity_stale")]
    assert [sorted(e["run_ids"]) for e in ev2 if e["event"] == "dispatch"] == [[r_moved]]
    assert (s2["dispatched"], s2["quarantined"], s2["spooled_hold"]) == (1, 0, 0) and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    ledger2 = _ledger(tmp_path, "dispatch_spooled.json")
    (j2,) = ledger2[r_moved]["judgment_ids"]
    assert set(ledger2) == {r_moved} and j2 != j1 and ledger2[r_moved]["identity_binding"] == {"instrument_id": INSTRUMENT_J2_ID, "required_judge_id": synth.JUDGE_ID_2}
    # J2's completion folds and covers the run: tick 3 clears the fresh hold as folded, nothing dispatched, nothing ever quarantined
    _fold(root)
    with DB.reading(_db(root)) as conn:
        assert tuple(conn.execute("SELECT judge_id, status, claimed_required_judge_id FROM judgment WHERE judgment_id = ?", (j2,)).fetchone()) == (synth.JUDGE_ID_2, "complete", synth.JUDGE_ID_2)
        assert P.pending(conn, None, synth.CAMPAIGN, run_id=(r_moved, r_other)) == []
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev1) + len(ev2):]
    assert [(e["run_id"], e["reason"]) for e in ev3 if e["event"] == "spooled_hold_cleared"] == [(r_moved, "folded")]
    assert not [e for e in ev3 if e["event"] in ("dispatch", "spooled_hold_binding_changed", "spooled_hold_identity_stale", "spooled_hold_released")]
    assert (s3["dispatched"], s3["quarantined"]) == (0, 0) and _ledger(tmp_path, "dispatch_spooled.json") == {} and not (tmp_path / "state" / "dispatch_quarantine.json").exists()


def test_an_unchanged_binding_keeps_the_coverage_short_path_when_the_judges_differ_and_the_verdict_refuses_each_defect_by_name(tmp_path, monkeypatch):
    """The negative: the hold's binding IS the run's current (post-correction) identity {Y, J2} while the folded completion is J1's -- the judge in
    the judgment differs from the required judge, the binding does not -- so nothing moved under this dispatch: coverage_short as today (no
    `spooled_hold_binding_changed`, no identity_stale, the quarantine recording the current binding, no identity_rebound release on a later
    pass). Then the verdict's own refusals: a completion by the CURRENT judge bound in the hold is a wrong-judge completion against the
    dispatch judge, a foreign-run judgment, an unchanged binding, a hold with no bound judge, a judgment whose stated binding is already current."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_NO_FOLD", "1")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_same, r_other = rids["job7@steer"], rids["job7@none"]
    gd.tick(args, *rest)                                                      # J1 grades both in full; holds bound to {X, J1}
    st = str(tmp_path / "state")
    ledger = _ledger(tmp_path, "dispatch_spooled.json")
    (j1,) = ledger[r_same]["judgment_ids"]
    _move_required_judge(root, r_same, "job7@steer", "lora@abcd", synth.OBJECT_SHA, 120)
    # the hold's binding set to the CURRENT identity: a hold whose dispatch saw the corrected run (its binding did not move under it)
    ledger[r_same]["identity_binding"] = {"instrument_id": INSTRUMENT_J2_ID, "required_judge_id": synth.JUDGE_ID_2}
    gd.spooled_save(st, ledger)
    _fold(root)                                                               # J1's completion folds: r_same pends for J2 as awaiting_second_judge
    with DB.reading(_db(root)) as conn:
        (pend,) = P.pending(conn, None, synth.CAMPAIGN, run_id=(r_same, r_other))
    assert pend["run_id"] == r_same and pend["pending_reason"] == "awaiting_second_judge"
    rows = gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    n_ev = len(_events(tmp_path))
    eligible, held = gd.spooled_filter(st, root, rows, dry_run=False)
    ev = _events(tmp_path)[n_ev:]
    assert not [e for e in ev if e["event"] in ("spooled_hold_binding_changed", "spooled_hold_identity_stale")]                # no binding change: neither branch
    (rel,) = [e for e in ev if e["event"] == "spooled_hold_released"]
    assert rel["run_id"] == r_same and rel["reason"] == "folded_still_pending" and rel["quarantined"] and rel["judgment_ids"] == [j1]
    assert "coverage_short" in rel["defect"] and "pending_reason=awaiting_second_judge" in rel["defect"] and f"required_judge_id={synth.JUDGE_ID_2}" in rel["defect"]
    assert "not accepted" not in rel["defect"]                                 # no verdict was tried: the binding did not change
    assert [e["run_id"] for e in ev if e["event"] == "spooled_hold_cleared"] == [r_other]
    q = _ledger(tmp_path, "dispatch_quarantine.json")
    assert set(q) == {r_same} and q[r_same]["rc"] == "coverage_short" and q[r_same]["judgment_ids"] == [j1]
    assert q[r_same]["identity_binding"] == {"instrument_id": INSTRUMENT_J2_ID, "required_judge_id": synth.JUDGE_ID_2}
    (jb,) = q[r_same]["judgment_bindings"]
    assert jb["judge_id"] == synth.JUDGE_ID and jb["binding"] in ({"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID}, q[r_same]["identity_binding"])
    eligible2, held2 = gd.quarantine_filter(st, root, eligible, dry_run=False)
    assert [r["run_id"] for r in held2] == [r_same] and not [e for e in _events(tmp_path)[n_ev:] if e["event"] == "quarantine_released"]   # the binding has not moved since filing
    # the verdict's refusals, by name
    old = {"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID}
    cur = {"instrument_id": INSTRUMENT_J2_ID, "required_judge_id": synth.JUDGE_ID_2}
    j_own = {"judgment_id": "j-1", "run_id": "run-r", "judge_id": synth.JUDGE_ID, "stated_binding": None}
    ok, why = gd.binding_changed_clear_verdict([j_own], run_id="run-r", current=cur, bound=dict(old))
    assert ok is True and "run run-r's" in why and f"the judge required at dispatch {synth.JUDGE_ID}'s" in why and "the daemon's binding at dispatch" in why
    ok, why = gd.binding_changed_clear_verdict([dict(j_own, stated_binding=dict(old))], run_id="run-r", current=cur, bound=dict(old))
    assert ok is True and "the judgment's stated binding" in why
    ok, why = gd.binding_changed_clear_verdict([j_own, {"judgment_id": "j-2", "run_id": "run-r", "judge_id": synth.JUDGE_ID_2, "stated_binding": None}], run_id="run-r", current=cur, bound=dict(old))
    assert ok is False and f"another judge than the judge required at dispatch {synth.JUDGE_ID}" in why and "j-2" in why and synth.JUDGE_ID_2 in why   # the CURRENT judge's completion is the wrong judge here
    ok, why = gd.binding_changed_clear_verdict([dict(j_own, judgment_id="j-s", run_id="run-s")], run_id="run-r", current=cur, bound=dict(old))
    assert ok is False and "foreign-run judgment j-s folded under run run-s" in why and "the binding change does not explain this fold" in why
    ok, why = gd.binding_changed_clear_verdict([j_own], run_id="run-r", current=cur, bound=dict(cur))
    assert ok is False and "no binding change to explain the fold" in why
    ok, why = gd.binding_changed_clear_verdict([j_own], run_id="run-r", current=cur, bound=None)
    assert ok is False and "no bound required judge" in why
    ok, why = gd.binding_changed_clear_verdict([j_own], run_id="run-r", current=cur, bound={"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": None})
    assert ok is False and "no bound required judge" in why
    ok, why = gd.binding_changed_clear_verdict([dict(j_own, stated_binding=dict(cur))], run_id="run-r", current=cur, bound=dict(old))
    assert ok is False and "did not change under it" in why and "the binding change is not this judgment's" in why
    ok, why = gd.binding_changed_clear_verdict([j_own], run_id="run-r", current=None, bound=dict(old))
    assert ok is False and "cannot be verified" in why
    # the identity_stale face reads unchanged (its texts are pinned above); the two faces share the three checks
    ok, why = gd.identity_stale_clear_verdict([j_own], run_id="run-r", required_judge=synth.JUDGE_ID, current={"instrument_id": CORRECTED_INSTRUMENT, "required_judge_id": synth.JUDGE_ID}, bound=dict(old))
    assert ok is True and "the identity_stale" not in why and "the run's required judge" in why
