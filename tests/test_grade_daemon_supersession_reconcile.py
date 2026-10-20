"""The daemon's tick RECONCILES a complete judgment standing beside the same judge's partials.

The scope once wrote the complete judgment first and its supersessions afterwards, separately: a grader dying in between left the
registry with a covering complete beside still-current partials -- the complete removed the run from v_pending_judgment, so no
dispatch repaired it, and the partials' item / metric rows stayed current. The scope now writes both as ONE atomic record file
(tests/test_worklist_supersedes_own_partial.py); for the legacy half-written state -- constructed here directly, a plain
hooks.JudgingScope completion over a run whose same-judge partial stands -- every tick runs `worklist.reconcile_supersessions`:
the same superseded records through the same atomic writer, one `supersession_reconciled` event per (run, judge) naming the ids;
a second tick finds the record on the spool (or the status folded) and writes nothing. A run with partials and NO complete is
never touched (the scope's own path at its next dispatch). Synthetic registry."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import hooks
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import worklist as W

import test_grade_daemon as TGD
from test_grade_daemon import _events, _load_daemon_module, _mirror_args, _root
from test_grade_daemon_schema_hold import _db, _grader_env, _tick_args

REPO = TGD.REPO
sys.path.insert(0, str(REPO / "scripts" / "registry"))


def _fold(root):
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0, rep.as_dict()


def _legacy_judgment(root, rid, judge, session, status, n_items):
    """The LEGACY shape: a plain hooks.JudgingScope on a registered claim -- a complete filed with NO supersession of the standing partial."""
    jid = P.claim(root, rid, judge, session)
    with DB.reading(DB.db_path(root)) as conn:
        info = P.active_claim(conn, rid, judge)
    scope = hooks.judgment_scope("tests/legacy_grader.py", {"run_id": rid, "campaign": synth.CAMPAIGN}, {"judge_id": judge}, api_pool="lowprio",
                                 spool=H.records_dir(root), registry_emit=True, session=session, store_uri="synthetic:///store",
                                 claim={"judgment_id": jid, "claimed_utc": info["claimed_utc"]})
    with scope:
        scope.items([{"item_id": f"q{i:05d}", "metric_id": "asr", "value": float(i % 2), "label": "harmful" if i % 2 else "safe"} for i in range(n_items)])
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": n_items // 2, "n": n_items}])
        scope.complete(status, n_items_judged=n_items)
    _fold(root)
    return jid


def test_the_tick_reconciles_a_complete_standing_beside_the_same_judges_partials_once_and_leaves_a_run_without_a_complete_alone(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_pair, r_only = rids["job7@none"], rids["job7@steer"]
    judge = synth.JUDGE_ID
    # the legacy half-written state: a partial, then a complete that superseded nothing (both by the instrument judge)
    partial_a = _legacy_judgment(root, r_pair, judge, "s-dead", "partial", 150)
    complete_a = _legacy_judgment(root, r_pair, judge, "s-dead2", "complete", 200)
    # a run with a partial and NO complete: the scope's own path at its next dispatch -- never reconciled here
    partial_only = _legacy_judgment(root, r_only, judge, "s-dead3", "partial", 120)
    conn = DB.connect(_db(root), readonly=True)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (partial_a,)).fetchone()[0] == "partial"
    assert [r["run_id"] for r in P.pending_for(conn, judge) if r["run_id"] == r_pair] == []              # the complete covers: no dispatch repairs it
    cands = W.supersession_candidates(conn, model_ids=[synth.MODEL_ID])
    assert [(c["run_id"], c["complete_id"], [p["judgment_id"] for p in c["partials"]]) for c in cands] == [(r_pair, complete_a, [partial_a])]
    conn.close()
    # tick 1: the pair is reconciled at the top of the tick -- the same superseded record through the same atomic writer, one event naming
    # the ids -- while the run with a partial and NO complete is simply PENDING: the tick dispatches it and the grader's own path (the
    # worklist scope) supersedes its partial in the completion's own record file; the reconciliation never names it
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (rec,) = [e for e in ev if e["event"] == "supersession_reconciled"]
    assert rec["run_id"] == r_pair and rec["judge_id"] == judge and rec["complete_id"] == complete_a and rec["partial_ids"] == [partial_a]
    assert os.path.isfile(rec["path"]) and rec["path"].startswith(os.path.join(H.records_dir(root), synth.CAMPAIGN, I.run_dir_name(r_pair)))
    assert s1["supersessions_reconciled"] == 1 and s1["dispatched"] == 1
    assert [e["run_ids"] for e in ev if e["event"] == "dispatch"] == [[r_only]]                       # only the uncovered run was dispatched
    _fold(root)
    conn = DB.connect(_db(root), readonly=True)
    p = conn.execute("SELECT status, superseded_by_judgment_id, n_items_judged, claimed_by_session, json_extract(extra_json, '$.superseded_reason') "
                     "FROM judgment WHERE judgment_id = ?", (partial_a,)).fetchone()
    assert tuple(p[:4]) == ("superseded", complete_a, 150, "s-dead") and complete_a in p[4] and "reconciled by the grade daemon" in p[4]
    q = conn.execute("SELECT status, superseded_by_judgment_id, json_extract(extra_json, '$.superseded_reason') FROM judgment WHERE judgment_id = ?", (partial_only,)).fetchone()
    assert q[0] == "superseded" and q[1] not in (None, complete_a) and "reconciled" not in q[2]            # the GRADER's own supersession, not the pass
    assert tuple(conn.execute("SELECT claimed_by_session, status FROM judgment WHERE judgment_id = ?", (q[1],)).fetchone()) == ("fake-driver", "complete")
    assert {r[0] for r in conn.execute("SELECT DISTINCT judgment_id FROM v_item_result_current WHERE run_id = ?", (r_pair,))} == {complete_a}
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()
    # tick 2: nothing left to reconcile -- the statuses folded; and a dry pass reports nothing to do
    s2 = gd.tick(args, *rest)
    assert s2["supersessions_reconciled"] == 0 and not [e for e in _events(tmp_path)[len(ev):] if e["event"] in ("supersession_reconciled", "dispatch")]
    assert W.reconcile_supersessions(root, model_ids=[synth.MODEL_ID], session="s-recon", script="tests/reconcile.py", dry_run=True) == []
    with DB.reading(_db(root)) as conn:
        assert W.supersession_candidates(conn, model_ids=[synth.MODEL_ID]) == []


# ------------------------------------------------------------------------------------------- the identity inputs, by FORM
BASE_KEYS = {"judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id"}


def _superseded_payloads_in(path):
    return [rec["payload"] for _, rec, _ in R.read_record_file(path) if rec["record_type"] == "judgment" and rec["payload"].get("status") == "superseded"]


def _plain_partial(root, rid, judge, n_items):
    """A PLAIN-form partial: judgment_id = identity.judgment_id(run, judge), no session, no time, no claim fields on the record -- the shape of
    the 2026-09 static-attack imports (387 partials on the live registry)."""
    jid = I.judgment_id(rid, judge)
    w = emit.RecordWriter(H.records_dir(root), synth.CAMPAIGN, rid, emit.producer_block("tests/legacy_import.py", session="s-legacy"), disabled=False)
    w.emit_judgment({"judgment_id": jid, "run_id": rid, "judge_id": judge, "status": "partial", "started_utc": "2026-09-17T00:00:00.000Z",
                     "finished_utc": "2026-09-17T00:00:00.000Z", "n_items_judged": n_items, "api_pool": "lowprio", "store_uri": "synthetic:///legacy"})
    _fold(root)
    return jid


def _strip_claim_inputs(root, jid):
    """A row NO identity form reproduces (a claim-minted id whose claim pair the row no longer states), put on the fixture registry's row
    directly: the reconciliation reads the table."""
    conn = DB.connect(_db(root))
    try:
        conn.execute("UPDATE judgment SET claimed_by_session = NULL, claimed_utc = NULL WHERE judgment_id = ?", (jid,))
        conn.commit()
    finally:
        conn.close()


def test_the_daemons_reconciliation_through_the_real_worklist_states_each_partials_identity_form(tmp_path, monkeypatch):
    """(c) of the brief: the tick's pass -> worklist.reconcile_supersessions -> write_supersessions over a REAL registry writes, per pair, a
    superseded record stating the fields of the form that mints the partial's id: the claim pair for a claim-minted partial (its id reproduces
    from them), nothing for a plain-form one (its id reproduces from run / judge alone). Both fold superseded."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_claim, r_plain, judge = rids["job7@none"], rids["job7@steer"], synth.JUDGE_ID
    partial_a = _legacy_judgment(root, r_claim, judge, "s-dead", "partial", 150)       # the CLAIMED form (pending.claim's mint)
    complete_a = _legacy_judgment(root, r_claim, judge, "s-dead2", "complete", 200)
    plain_b = _plain_partial(root, r_plain, judge, 150)                                 # the PLAIN form
    complete_b = _legacy_judgment(root, r_plain, judge, "s-dead3", "complete", 200)
    with DB.reading(_db(root)) as conn:
        (pa,) = W.own_partial_judgments(conn, r_claim, judge); (pb,) = W.own_partial_judgments(conn, r_plain, judge)
    assert W.judgment_identity_form(pa, run_id=r_claim, judge_id=judge) == ("claimed", {"claimed_by_session": "s-dead", "claimed_utc": pa["claimed_utc"]})
    assert W.judgment_identity_form(pb, run_id=r_plain, judge_id=judge) == ("plain", {})
    s1 = gd.tick(args, *rest)
    recs = {e["run_id"]: e for e in _events(tmp_path) if e["event"] == "supersession_reconciled"}
    assert set(recs) == {r_claim, r_plain} and (s1["supersessions_reconciled"], s1["supersessions_refused"], s1["dispatched"]) == (2, 0, 0)
    (sup_a,) = _superseded_payloads_in(recs[r_claim]["path"])
    assert (sup_a["judgment_id"], sup_a["superseded_by_judgment_id"], sup_a["claimed_by_session"], sup_a["claimed_utc"]) == (partial_a, complete_a, "s-dead", pa["claimed_utc"])
    assert I.judgment_id(r_claim, judge, sup_a["claimed_by_session"], sup_a["claimed_utc"]) == sup_a["judgment_id"]
    (sup_b,) = _superseded_payloads_in(recs[r_plain]["path"])
    assert set(sup_b) - {"extra"} == BASE_KEYS and (sup_b["judgment_id"], sup_b["superseded_by_judgment_id"]) == (plain_b, complete_b)
    assert I.judgment_id(sup_b["run_id"], sup_b["judge_id"]) == sup_b["judgment_id"]
    _fold(root)
    with DB.reading(_db(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id, claimed_by_session, claimed_utc FROM judgment WHERE judgment_id = ?",
                                  (partial_a,)).fetchone()) == ("superseded", complete_a, "s-dead", pa["claimed_utc"])
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id, claimed_by_session, claimed_utc FROM judgment WHERE judgment_id = ?",
                                  (plain_b,)).fetchone()) == ("superseded", complete_b, None, None)
        assert W.supersession_candidates(conn, model_ids=[synth.MODEL_ID]) == []
        assert [r["run_id"] for r in P.pending_for(conn, judge) if r["run_id"] in (r_claim, r_plain)] == []


def test_the_tick_refuses_by_name_a_pair_whose_partial_no_identity_form_reproduces_and_writes_nothing(tmp_path, monkeypatch):
    """A complete beside a partial NO form reproduces: the pass is one `supersession_refused` event naming JudgmentIdentityUnreproducible and
    the partial, `supersessions_refused` 1, `supersessions_reconciled` 0, no record file; the same again next tick (nothing on the spool makes
    it idempotent-by-record); the partial stands beside the complete, which still covers the run."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    r_pair, judge = rids["job7@none"], synth.JUDGE_ID
    partial_a = _legacy_judgment(root, r_pair, judge, "s-dead", "partial", 150)
    complete_a = _legacy_judgment(root, r_pair, judge, "s-dead2", "complete", 200)
    _strip_claim_inputs(root, partial_a)
    run_dir = os.path.join(H.records_dir(root), synth.CAMPAIGN, I.run_dir_name(r_pair))
    before = sorted(os.listdir(run_dir))
    for tick in (1, 2):
        s = gd.tick(args, *rest)
        _fold(root)                                                                                  # the dispatched run's judgment folds; the stripped row stays as it is
        ev = _events(tmp_path)
        refused = [e for e in ev if e["event"] == "supersession_refused"]
        assert len(refused) == tick and not [e for e in ev if e["event"] == "supersession_reconciled"]
        assert (refused[-1]["run_id"], refused[-1]["judge_id"], refused[-1]["complete_id"], refused[-1]["partial_ids"]) == (r_pair, judge, complete_a, [partial_a])
        assert refused[-1]["refused"].startswith("JudgmentIdentityUnreproducible: ") and partial_a in refused[-1]["refused"] and "D639" in refused[-1]["refused"]
        assert (s["supersessions_refused"], s["supersessions_reconciled"]) == (1, 0) and not s.get("problems")
        assert sorted(os.listdir(run_dir)) == before                                                 # nothing written for the pair
        assert [e["run_ids"] for e in ev if e["event"] == "dispatch"] == [[rids["job7@steer"]]]      # the covered run is never dispatched
    with DB.reading(_db(root)) as conn:
        assert tuple(conn.execute("SELECT status, superseded_by_judgment_id FROM judgment WHERE judgment_id = ?", (partial_a,)).fetchone()) == ("partial", None)
        assert [r["run_id"] for r in P.pending_for(conn, judge) if r["run_id"] == r_pair] == []
