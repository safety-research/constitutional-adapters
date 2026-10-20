"""`RegistryWorklist.claim` re-reads the run's CURRENT designated judge before `pending.claim`.

The claim accepted any (run_id, judge_id) pair the worklist row carried: a same-id corrective FULL run record switching the run to another
registered instrument (whose designated judge differs) between the worklist read and the claim let the grader claim and file under the OLD
judge while the new required judge stayed ungraded. Now `require_current_judge` reads the run's instrument's `judge_id` the way
v_pending_judgment derives it (`JOIN instrument i ON i.instrument_id = r.instrument_id`, `is_instrument_judge = (i.judge_id = jd.judge_id)`)
and refuses by name -- `StaleRequiredJudge`, a `pending.ClaimHeld` so every driver's existing claim refusal names it -- when the row was
selected as the instrument's judge and the current designation differs; `claim_all` releases what it took and re-raises. A row selected for
a judge that was never the instrument's (a second judge) is untouched. Synthetic ids throughout."""
from __future__ import annotations

import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import worklist as W

DRIVER = "tests/stub_driver.py"
#: a second registered instrument of the same eval whose designated judge is JUDGE_ID_2 (the instrument id is minted from its fields)
OTHER_INSTRUMENT = synth.instrument_fields(judge_id=synth.JUDGE_ID_2, token_cap_tokens=1024)
OTHER_INSTRUMENT_ID = str(OTHER_INSTRUMENT["instrument_id"])
assert OTHER_INSTRUMENT_ID != synth.INSTRUMENT_ID


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _root_with_two_instruments(tmp_path, monkeypatch, keys):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions() + [synth.rec("instrument", OTHER_INSTRUMENT, synth.t(40))]
    _write(root, "dims", "000001-dims.jsonl", dims)
    rids = []
    for k, key in enumerate(keys):
        rid, recs = synth.run_records(key, 100 + 20 * k, through="generated")
        recs[0]["payload"]["store_uri"] = f"file:///stores/{key}"
        _write(root, "c", f"00000{k + 1}-run.jsonl", recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def _switch_instrument(root, rid, key, seq):
    """The corrective SAME-ID full run record: only instrument_id changes (to the instrument whose judge is JUDGE_ID_2); folded."""
    p = synth.run_payload(key, i=300 + seq, instrument_id=OTHER_INSTRUMENT_ID)
    p["run_id"] = rid                                             # the same identity: a correction of the registered instrument, not a new run
    _write(root, "c", f"00009{seq}-run-correction.jsonl", [synth.rec("run", p, synth.t(300 + seq))])
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0, rep.as_dict()


def test_a_run_switched_to_another_instruments_judge_between_the_read_and_the_claim_refuses_by_name_and_claims_nothing(tmp_path, monkeypatch):
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_switch"])
    wl = W.RegistryWorklist(root, None, "s-A", driver=DRIVER)                          # the default worklist: each run's instrument judge
    rows = [r for r in wl.pending(eval_category="static_attack") if r["run_id"] == rid and r["is_instrument_judge"]]
    assert len(rows) == 1 and rows[0]["required_judge_id"] == synth.JUDGE_ID and rows[0]["instrument_id"] == synth.INSTRUMENT_ID
    _switch_instrument(root, rid, "k_switch", 1)                                        # between the read and the claim
    with DB.reading(DB.db_path(root)) as conn:
        assert W.current_required_judge(conn, rid) == (synth.JUDGE_ID_2, OTHER_INSTRUMENT_ID)
    with pytest.raises(P.ClaimHeld) as ei:                                              # the drivers' existing refusal class
        wl.claim_all(rows)
    assert isinstance(ei.value, W.StaleRequiredJudge)
    msg = str(ei.value)
    assert f"run {rid} now designates judge {synth.JUDGE_ID_2}" in msg and f"instrument {OTHER_INSTRUMENT_ID}" in msg
    assert f"the worklist row carried {synth.JUDGE_ID}" in msg and "re-select" in msg
    assert wl.claims == {}
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0        # ZERO claims written
    # a re-select sees the new designation: the run pends for JUDGE_ID_2 as the instrument judge now
    fresh = [r for r in P.pending_for(conn, synth.JUDGE_ID_2) if r["run_id"] == rid]
    assert len(fresh) == 1 and fresh[0]["is_instrument_judge"] == 1
    conn.close()


def test_claim_all_releases_the_claims_it_took_before_the_stale_row_and_a_second_judge_row_is_untouched(tmp_path, monkeypatch):
    root, (r_ok, r_switch) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_ok", "k_switch2"])
    wl = W.RegistryWorklist(root, None, "s-A", driver=DRIVER)
    by_id = {r["run_id"]: r for r in wl.pending(eval_category="static_attack") if r["is_instrument_judge"]}
    rows = [by_id[r_ok], by_id[r_switch]]
    _switch_instrument(root, r_switch, "k_switch2", 2)
    with pytest.raises(W.StaleRequiredJudge):
        wl.claim_all(rows)
    assert wl.claims == {}
    conn = DB.connect(DB.db_path(root), readonly=True)
    # the first row's claim was taken and then RELEASED (abandoned); the stale row never claimed
    assert [r[0] for r in conn.execute("SELECT status FROM judgment WHERE run_id = ?", (r_ok,))] == ["abandoned"]
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (r_switch,)).fetchone()[0] == 0
    # a SECOND-judge row (JUDGE_ID_2 explicitly, never the instrument's judge for r_ok) claims regardless of the instrument's designation
    wl2 = W.RegistryWorklist(root, [synth.JUDGE_ID_2], "s-B", driver=DRIVER)
    second = [r for r in wl2.pending(eval_category="static_attack") if r["run_id"] == r_ok]
    assert len(second) == 1 and second[0]["is_instrument_judge"] == 0
    conn.close()
    assert wl2.claim_all(second)[(r_ok, synth.JUDGE_ID_2)]["judge_id"] == synth.JUDGE_ID_2
    # the pure check
    with DB.reading(DB.db_path(root)) as conn:
        W.require_current_judge(conn, {"run_id": r_ok, "required_judge_id": synth.JUDGE_ID, "is_instrument_judge": 1})       # still the designation
        W.require_current_judge(conn, {"run_id": r_switch, "required_judge_id": synth.JUDGE_ID, "is_instrument_judge": 0})   # a second-judge row: no check
        with pytest.raises(W.StaleRequiredJudge):
            W.require_current_judge(conn, {"run_id": r_switch, "required_judge_id": synth.JUDGE_ID, "is_instrument_judge": 1})
        with pytest.raises(W.StaleRequiredJudge):
            W.require_current_judge(conn, {"run_id": r_switch, "required_judge_id": synth.JUDGE_ID})                        # absent flag: the instrument's judge was the basis


# --------------------------------------------------------------------------------------------
# the comparison runs INSIDE pending.claim's locked pre-write path; an absent designation is not permission
# --------------------------------------------------------------------------------------------


def test_a_correction_folded_right_before_the_locked_claim_is_refused_inside_the_lock_with_nothing_written(tmp_path, monkeypatch):
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_race"])
    wl = W.RegistryWorklist(root, None, "s-A", driver=DRIVER)
    rows = [r for r in wl.pending(eval_category="static_attack") if r["run_id"] == rid and r["is_instrument_judge"]]
    with DB.reading(DB.db_path(root)) as conn:
        W.require_current_judge(conn, rows[0])                                   # the worklist's own earlier read would have passed
    real_claim = P.claim
    seen = {}

    def racing_claim(root_, run_id, judge_id, session, **kw):                    # the correction lands AFTER that read, right before the lock
        _switch_instrument(root_, run_id, "k_race", 3)
        seen["precheck"] = kw.get("precheck")
        return real_claim(root_, run_id, judge_id, session, **kw)

    monkeypatch.setattr(W.P, "claim", racing_claim)
    with pytest.raises(W.StaleRequiredJudge) as ei:
        wl.claim_all(rows)
    assert callable(seen["precheck"]) and f"now designates judge {synth.JUDGE_ID_2}" in str(ei.value)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0 and wl.claims == {}   # refused inside the lock: no claim
    conn.close()


def test_the_precheck_runs_under_the_write_lock_after_the_parents_and_before_any_record(tmp_path, monkeypatch):
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_lock"])
    calls = []

    def precheck(conn):
        calls.append(conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0])
        with pytest.raises(DB.LockHeld):                                          # the write lock is HELD while the precheck runs (flock: a second descriptor blocks)
            with DB.write_lock(root, timeout_s=0.2):
                pass
        raise P.ClaimHeld(rid, "precheck", 0)

    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", precheck=precheck)
    assert calls == [0]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0        # nothing written
    conn.close()
    with pytest.raises(P.ClaimNotFolded):                                            # the parents first: an unknown run never reaches the precheck
        P.claim(root, "synthcampaign:syntheval:aaaaaaaaaaaa:0000000000000000", synth.JUDGE_ID, "s-A", precheck=lambda conn: calls.append("reached"))
    assert calls == [0]
    jid = P.claim(root, rid, synth.JUDGE_ID, "s-A", precheck=lambda conn: None)    # a passing precheck claims as before
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "in_progress"
    conn.close()


def test_a_run_switched_to_a_rule_scored_instrument_refuses_the_default_judge_row(tmp_path, monkeypatch):
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_rule"])
    wl = W.RegistryWorklist(root, None, "s-A", driver=DRIVER)
    rows = [r for r in wl.pending(eval_category="static_attack") if r["run_id"] == rid and r["is_instrument_judge"]]
    _write(root, "dims", "000002-rule-dims.jsonl", synth.seed_rule_scored_dimensions())                      # the rule-scored eval, item set and instrument (no designated judge)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    p = synth.run_payload("k_rule", i=320, instrument_id=synth.RULE_INSTRUMENT_ID)   # the run switched to it: an instrument without a judge
    p["run_id"] = rid
    _write(root, "c", "000094-run-rule.jsonl", [synth.rec("run", p, synth.t(320))])
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with DB.reading(DB.db_path(root)) as conn:
        assert W.current_required_judge(conn, rid) == (None, synth.RULE_INSTRUMENT_ID)
    with pytest.raises(W.StaleRequiredJudge) as ei:
        wl.claim_all(rows)
    assert W.NO_DESIGNATED_JUDGE in str(ei.value) and "now designates judge" in str(ei.value) and wl.claims == {}
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0
    conn.close()


# --------------------------------------------------------------------------------------------
# a DISTINCT exported refusal class, the CLI naming it, a precheck that writes is refused
# --------------------------------------------------------------------------------------------


def test_the_stale_judge_refusal_is_a_claim_precheck_refused_distinct_from_a_held_claim(tmp_path, monkeypatch):
    assert issubclass(W.StaleRequiredJudge, P.ClaimPrecheckRefused) and issubclass(P.ClaimPrecheckRefused, P.ClaimHeld)
    assert [c.__name__ for c in W.StaleRequiredJudge.__mro__[:4]] == ["StaleRequiredJudge", "ClaimPrecheckRefused", "ClaimHeld", "RuntimeError"]
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_mro"])
    wl = W.RegistryWorklist(root, None, "s-A", driver=DRIVER)
    rows = [r for r in wl.pending(eval_category="static_attack") if r["run_id"] == rid and r["is_instrument_judge"]]
    _switch_instrument(root, rid, "k_mro", 4)
    with pytest.raises(P.ClaimPrecheckRefused) as ei:                                     # the owner's distinct class catches it
        wl.claim_all(rows)
    e = ei.value
    assert isinstance(e, W.StaleRequiredJudge) and e.run_id == rid and e.current_judge == synth.JUDGE_ID_2 and e.row_judge == synth.JUDGE_ID
    assert str(e).startswith(f"claim refused: run {rid} now designates judge {synth.JUDGE_ID_2}") and e.reason.startswith("now designates judge")
    assert not isinstance(P.ClaimHeld(rid, "someone-else", 12), P.ClaimPrecheckRefused)       # a held claim is the OTHER refusal: retry later
    g = P.ClaimPrecheckRefused(rid, "synthetic reason")
    assert str(g) == f"claim refused (precheck): run {rid}: synthetic reason" and (g.reason, g.session, g.age_s) == ("synthetic reason", "", 0)


def test_the_cli_exits_1_naming_the_refusal_class(tmp_path, monkeypatch, capsys):
    from sisteer.registry import cli
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_cli"])
    argv = ["pending-judgment", "--judge", synth.JUDGE_ID, "--claim", "--run-id", rid, "--session", "cli-A"]

    def refuse(exc):
        def _claim(*a, **k):
            raise exc
        return _claim

    monkeypatch.setattr(P, "claim", refuse(P.ClaimPrecheckRefused(rid, "synthetic: the designated judge moved")))
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert f"claim refused: ClaimPrecheckRefused: claim refused (precheck): run {rid}: synthetic: the designated judge moved" in err
    monkeypatch.setattr(P, "claim", refuse(P.ClaimHeld(rid, "other-session", 7)))
    assert cli.main(argv) == 1
    assert f"claim refused: ClaimHeld: run {rid} is claimed by session 'other-session' (7 s ago)" in capsys.readouterr().err
    monkeypatch.setattr(P, "claim", refuse(W.StaleRequiredJudge(rid, synth.JUDGE_ID_2, OTHER_INSTRUMENT_ID, synth.JUDGE_ID)))
    assert cli.main(argv) == 1
    assert "claim refused: StaleRequiredJudge: claim refused: run" in capsys.readouterr().err


def test_a_precheck_that_writes_is_refused_and_the_connection_is_left_writable(tmp_path, monkeypatch):
    root, (rid,) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_ro"])
    ro = DB.connect(DB.db_path(root), readonly=True)
    before = tuple(ro.execute("SELECT lifecycle_state, n_items_generated FROM run WHERE run_id = ?", (rid,)).fetchone())

    def writing(conn):
        conn.execute("UPDATE run SET n_items_generated = 7 WHERE run_id = ?", (rid,))      # a write under the claim lock

    with pytest.raises(P.ClaimPrecheckRefused) as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-A", precheck=writing)
    assert "attempted a write" in str(ei.value) and "read-only" in str(ei.value) and str(ei.value).startswith(f"claim refused (precheck): run {rid}")
    assert tuple(ro.execute("SELECT lifecycle_state, n_items_generated FROM run WHERE run_id = ?", (rid,)).fetchone()) == before    # nothing changed
    assert ro.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0                                   # no claim written
    # a read error that is NOT a write is the precheck's own and propagates as itself
    with pytest.raises(sqlite3.OperationalError):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", precheck=lambda conn: conn.execute("SELECT 1 FROM no_such_table_synthetic").fetchone())
    # the connection and the lock are left as they were: a read-only precheck claims right after
    jid = P.claim(root, rid, synth.JUDGE_ID, "s-A", precheck=lambda conn: conn.execute("SELECT 1").fetchone())
    assert ro.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "in_progress"
    ro.close()


# --------------------------------------------------------------------------------------------
# ClaimPendingCorrection, a ClaimNotFolded SUBCLASS, routes to retry-later by TYPE
# --------------------------------------------------------------------------------------------


def test_claim_all_releases_and_re_raises_a_claim_not_folded_subclass_by_type_the_owners_claim_pending_correction(tmp_path, monkeypatch):
    """pending.claim will refuse with `ClaimPendingCorrection(ClaimNotFolded)` while an identity correction of the run is pending; it is treated
    exactly like ClaimNotFolded. The worklist matches claim refusals by ISINSTANCE (an `except` on ClaimHeld / ClaimNotFolded), never an exact
    class, so the subclass -- the registry's REAL class since schema 23 (a synthetic stand-in before it) -- is released-and-re-raised by claim_all
    with NO change to the refusal set: a refusal on the second row never leaks the first row's claim; the pass goes on when the refusal lifts."""
    assert W.RETRY_LATER_CLAIM_REFUSAL == "ClaimPendingCorrection"
    cls = getattr(P, W.RETRY_LATER_CLAIM_REFUSAL)                                                  # the owner's REAL class (schema 23)
    assert cls is P.ClaimPendingCorrection and issubclass(cls, P.ClaimNotFolded) and cls is not P.ClaimNotFolded
    assert W._CLAIM_REFUSALS == (P.ClaimHeld, P.ClaimNotFolded)                                   # the refusal set did NOT widen: routed by TYPE
    exc = cls("run x", "run_patch")                                                                # (run_id, what): the registry's signature
    assert exc.run_id == "run x" and str(exc).endswith("retry after the next harvester cycle (nothing written)")
    assert W.is_retry_later_refusal(cls("run x", "run_patch")) and str(cls("run x", "run_patch")).startswith("run run x: an identity correction")
    assert W.is_retry_later_refusal(P.ClaimNotFolded("run x", "no run row yet")) and not W.is_retry_later_refusal(P.ClaimHeld("r", "s-B", 0)) and not W.is_retry_later_refusal(RuntimeError("x"))
    root, (r_a, r_b) = _root_with_two_instruments(tmp_path, monkeypatch, ["k_pc_a", "k_pc_b"])
    wl = W.RegistryWorklist(root, None, "s-A", driver=DRIVER)
    by_id = {r["run_id"]: r for r in wl.pending(eval_category="static_attack") if r["is_instrument_judge"]}
    rows = [by_id[r_a], by_id[r_b]]
    real_claim = P.claim

    def claim(root_, rid, jid, session, **kw):                                                     # the owner's refusal on the SECOND row
        if rid == r_b:
            raise cls(rid, "adapter revision")
        return real_claim(root_, rid, jid, session, **kw)

    monkeypatch.setattr(P, "claim", claim)
    with pytest.raises(P.ClaimNotFolded, match="identity correction") as ei:                       # caught and re-raised as the subclass it is
        wl.claim_all(rows)
    assert type(ei.value) is cls and wl.claims == {}
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [r[0] for r in conn.execute("SELECT status FROM judgment WHERE run_id = ?", (r_a,))] == ["abandoned"]      # taken, then RELEASED
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (r_b,)).fetchone()[0] == 0                 # never claimed
    conn.close()
    # the refusal lifted: the same worklist claims both as before
    monkeypatch.setattr(P, "claim", real_claim)
    claims = wl.claim_all(rows)
    assert sorted(k[0] for k in claims) == sorted([r_a, r_b])
