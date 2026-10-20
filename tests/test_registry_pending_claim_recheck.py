"""A claim re-checks that (run, judge) is STILL PENDING after its scoped fold.

`pending.claim` folded the run's own files first and then re-checked parents and the designated judge, but never that the pair was still
pending: a run another session completed between the judging daemon's selection and this claim (its completion folded by the harvester's
cycle, or spooled in the run's own files and folded by the claim's own scoped fold) got a SECOND claim and a second complete judgment of
the same (run, judge), which no currency rule excludes (superseded, abandoned and displaced partial judgments are excluded; a second
complete is not). Now, under the claim lock and before anything is written, `v_pending_judgment` -- the selection the daemon chose the run
from -- is re-read for the pair (`pending.not_pending_why`, one bounded read); a pair no longer listed because a complete identity-current
judgment by this judge covers the run, or because the run's lifecycle state left the judgeable set, refuses `ClaimNotPending` -- a
`ClaimNotFolded`, so the daemon routes it as retry-later / re-select (`claim refused: ClaimNotPending: ` on the CLI, exit 1 -- the class name
first, so a consumer classes the refusal by type without parsing the message), never a quarantine.
Unchanged: the normal path, the held-claim refusal, the forced takeover of a stale claim (the run is still listed as `claimed`), an
identity-stale complete judgment (it covers nothing and leaves the pair listed as `identity_stale`), and the pairing hole (a
judge-less instrument whose fallback judge has filed nothing pairs with no judge row in `v_pending_judgment`, and the producer's direct
claim stands).
"""
from __future__ import annotations

import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from test_registry_judgment_currency import (INSTRUMENT_X2, INSTRUMENT_X2_ID, _complete, _correct_to_x2, _generated, _ingest, _judgment,
                                             _judgment_rows, _metric, _q, _rebuild_equal, _root, _worklist_row, _write)

#: the start of every ClaimNotPending message after the run id: `run <run_id>: not pending for judge <judge_id> any more -- <why>; re-select (nothing written)`
PREFIX = "not pending for judge"


def _claim_files(root, rid):
    d = os.path.join(root, "records", "claims", I.run_dir_name(rid))
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _record_files(root):
    return sorted(R.iter_record_files(os.path.join(root, "records")))


def _select(root, rid, judge):
    """What the judging daemon does first: read the pending row of (run, judge) from v_pending_judgment -- its selection."""
    return _worklist_row(root, rid, judge)


def _why(root, rid, judge):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return P.not_pending_why(conn, rid, judge)
    finally:
        conn.close()


def test_a_run_another_session_completed_between_the_selection_and_the_claim_is_refused_not_pending_with_nothing_written(tmp_path):
    root, rid = _generated(tmp_path)
    # two sessions select the run (the daemon's read of v_pending_judgment): no judgment yet
    assert _select(root, rid, synth.JUDGE_ID)["pending_reason"] == "no_judgment"
    assert _why(root, rid, synth.JUDGE_ID) is None
    # A claims and completes; the harvester's cycle folds the completion -- the run leaves the selection
    a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    _complete(root, rid, a, synth.JUDGE_ID, "s-A", 1000, 2000)
    assert _ingest(root, synth.t(2500)).refused == 0
    assert _select(root, rid, synth.JUDGE_ID) is None
    files_before, rows_before, claim_files_before = _record_files(root), _judgment_rows(root, rid), _claim_files(root, rid)
    # B, which selected before the fold, claims now: refused by name, a ClaimNotFolded (the daemon's retry-later / re-select route)
    with pytest.raises(P.ClaimNotPending) as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(3000))
    exc = ei.value
    assert isinstance(exc, P.ClaimNotFolded) and exc.run_id == rid and exc.judge_id == synth.JUDGE_ID
    assert str(exc).startswith(f"run {rid}: {PREFIX} {synth.JUDGE_ID} any more -- ") and str(exc).endswith("; re-select (nothing written)")
    assert f"complete judgment {a} by this judge (session 's-A'" in exc.why and "already covers it" in exc.why and exc.why in str(exc)
    # nothing written: no claim file, no judgment row, no record anywhere
    assert _record_files(root) == files_before and _judgment_rows(root, rid) == rows_before and _claim_files(root, rid) == claim_files_before
    # exactly ONE complete judgment of the pair, and the pair is still not pending
    rows = _judgment_rows(root, rid)
    assert [j["judgment_id"] for j in rows.values() if j["status"] == "complete"] == [a] and _select(root, rid, synth.JUDGE_ID) is None
    # A's own retry after its completion is refused the same way (never a second claim by the completing session either); so is a
    # FORCED claim: the pair is genuinely complete, and force takes over a stale in-progress claim only
    with pytest.raises(P.ClaimNotPending):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3001))
    with pytest.raises(P.ClaimNotPending):
        P.claim(root, rid, synth.JUDGE_ID, "s-B", force=True, stale_hours=0.0, now=synth.t(3002))
    assert _record_files(root) == files_before
    # the second-judge design is untouched: J2 is still pending for the run and claims it
    assert _select(root, rid, synth.JUDGE_ID_2)["pending_reason"] == "awaiting_second_judge"
    P.claim(root, rid, synth.JUDGE_ID_2, "s-C", now=synth.t(3100))
    _rebuild_equal(root, tmp_path)


def test_a_completion_spooled_in_the_runs_own_files_is_folded_first_and_refuses_the_second_claim(tmp_path):
    """The judging drivers spool their completion records under records/<campaign>/<run_dir>/ (hooks.judgment_scope over the registry's
    records/), the directory `pending.run_record_files` lists: B's scoped fold-first folds A's completion before the re-check, so B is
    refused without waiting for the harvester's cycle -- and the selection B read a moment earlier still said `claimed`."""
    root, rid = _generated(tmp_path)
    a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    own = f"{synth.CAMPAIGN}/{I.run_dir_name(rid)}"
    _write(root, own, "000001-judgment.jsonl", [_judgment(rid, a, synth.JUDGE_ID, "complete", 2000, n=4, claimed_by_session="s-A", claimed_utc=synth.t(1000)),
                                                 _metric(rid, a, 2001)])
    assert _select(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed"          # unfolded: the selection still reads A's claim
    with pytest.raises(P.ClaimNotPending, match=PREFIX) as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(3000))
    assert f"complete judgment {a}" in ei.value.why
    rows = _judgment_rows(root, rid)
    assert list(rows) == [a] and rows[a]["status"] == "complete" and _select(root, rid, synth.JUDGE_ID) is None
    assert len(_claim_files(root, rid)) == 1                                           # A's claim file alone; B wrote nothing
    _rebuild_equal(root, tmp_path)


def test_the_normal_path_is_unchanged_held_claim_same_session_forced_takeover_release_and_a_second_judge(tmp_path):
    root, rid = _generated(tmp_path)
    a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    assert _select(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed"
    with pytest.raises(P.ClaimHeld):                                                   # a live current-bound claim is held, as before
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(1001))
    assert P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1002)) == a           # the same session gets its own id back
    # the forced takeover of a stale claim: the run is still listed (`claimed`), so force proceeds exactly as before
    b = P.claim(root, rid, synth.JUDGE_ID, "s-B", force=True, stale_hours=12, now="2026-01-02T00:00:00.000Z")
    rows = _judgment_rows(root, rid)
    assert b != a and rows[a]["status"] == "abandoned" and rows[b]["status"] == "in_progress"
    assert P.release(root, rid, synth.JUDGE_ID, "s-B", now="2026-01-02T00:00:01.000Z")
    assert _select(root, rid, synth.JUDGE_ID)["pending_reason"] == "abandoned"
    c = P.claim(root, rid, synth.JUDGE_ID, "s-C", now="2026-01-02T00:00:02.000Z")
    d = P.claim(root, rid, synth.JUDGE_ID_2, "s-D", now="2026-01-02T00:00:03.000Z")   # a second judge claims the same run beside it
    assert len({a, b, c, d}) == 4
    _rebuild_equal(root, tmp_path)


def test_an_identity_stale_complete_judgment_is_not_complete_for_the_re_check_and_the_run_is_claimed(tmp_path):
    """After the same-judge correction J's complete judgment is stale-bound, covers nothing and leaves the pair listed
    (`identity_stale`); the re-check reads the selection's own coverage rule, so the re-claim stands -- never ClaimNotPending."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    _complete(root, rid, a, synth.JUDGE_ID, "s-A", 1000, 2000)
    assert _ingest(root, synth.t(2500)).refused == 0 and _select(root, rid, synth.JUDGE_ID) is None
    assert _why(root, rid, synth.JUDGE_ID) is not None                                 # complete and current: not pending
    _correct_to_x2(root, rid, 3000)
    assert _select(root, rid, synth.JUDGE_ID)["pending_reason"] == "identity_stale" and _why(root, rid, synth.JUDGE_ID) is None
    b = P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(4000))
    rows = _judgment_rows(root, rid)
    assert rows[b]["status"] == "in_progress" and rows[b]["claimed_instrument_id"] == INSTRUMENT_X2_ID and rows[a]["status"] == "complete"
    _rebuild_equal(root, tmp_path)


def test_a_judge_less_instruments_fallback_judge_still_claims_the_pair_v_pending_judgment_never_lists(tmp_path):
    """The pairing limitation: a judge-less instrument whose fallback `run.judge_id` has filed nothing pairs with no judge row in
    v_pending_judgment, so the pair is never listed -- and the producer's direct claim on it stands (the rule-scored re-judge
    `test_a_run_whose_instrument_names_no_judge_binds_and_compares_run_judge_id` relies on). Once the judge has filed, the pair is listed and
    the ordinary rule applies: complete and current means not pending."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid, recs = synth.rule_scored_run_records("rk", 100, through="generated", n_items=4)
    recs.append(synth.rec("run_patch", {"run_id": rid, "fields": {"judge_id": synth.JUDGE_ID}}, synth.t(150)))
    _write(root, "c", "000001-run.jsonl", recs)
    assert _ingest(root, synth.t(900)).refused == 0
    assert _select(root, rid, synth.JUDGE_ID) is None and _why(root, rid, synth.JUDGE_ID) is None      # unpaired, yet claimable
    a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    assert _select(root, rid, synth.JUDGE_ID)["pending_reason"] == "claimed"           # paired now: the judge has filed
    assert P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1001)) == a
    _write(root, "j", "000002-complete.jsonl", [_judgment(rid, a, synth.JUDGE_ID, "complete", 1001, n=4, claimed_by_session="s-A", claimed_utc=synth.t(1000)),
                                                synth.rec("run_metric", {"run_id": rid, "judgment_id": a, "metric_id": "acc", "threshold": -1.0, "value": 0.5,
                                                                         "computed_by": "producer"}, synth.t(1002))])
    assert _ingest(root, synth.t(2000)).refused == 0
    assert _select(root, rid, synth.JUDGE_ID) is None
    with pytest.raises(P.ClaimNotPending, match=PREFIX):
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(3000))
    assert list(_judgment_rows(root, rid)) == [a]


def test_a_run_whose_lifecycle_state_admits_no_judging_is_refused_by_name_with_nothing_written(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running", n_items=4)
    recs.append(synth.rec("run_state", {"run_id": rid, "state": "cancelled", "event_utc": synth.t(109), "evidence_kind": "scheduler_state",
                                        "scheduler_state": "CANCELLED"}, synth.t(109)))
    _write(root, "c", "000001-run.jsonl", recs)
    assert _ingest(root, synth.t(900)).refused == 0
    assert _q(root, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [{"lifecycle_state": "cancelled"}]
    assert _select(root, rid, synth.JUDGE_ID) is None and _why(root, rid, synth.JUDGE_ID) == "lifecycle_state=cancelled admits no judging"
    before = _record_files(root)
    with pytest.raises(P.ClaimNotPending, match="lifecycle_state=cancelled admits no judging") as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(1000))
    assert isinstance(ei.value, P.ClaimNotFolded)
    assert _record_files(root) == before and _judgment_rows(root, rid) == {} and _claim_files(root, rid) == []


def test_the_cli_prints_claim_refused_with_the_message_and_exits_1(tmp_path, capsys):
    root, rid = _generated(tmp_path)
    a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    _complete(root, rid, a, synth.JUDGE_ID, "s-A", 1000, 2000)
    assert _ingest(root, synth.t(2500)).refused == 0
    before = _record_files(root)
    rc = cli.main(["--root", root, "pending-judgment", "--claim", "--run-id", rid, "--judge", synth.JUDGE_ID, "--session", "s-B"])
    out = capsys.readouterr()
    assert rc == 1 and out.out == ""
    assert out.err.startswith(f"claim refused: ClaimNotPending: run {rid}: {PREFIX} {synth.JUDGE_ID} any more -- complete judgment {a} by this judge")
    assert "; re-select (nothing written)" in out.err
    assert _record_files(root) == before
    # the second judge's CLI claim stands (exit 0, the id on stdout)
    rc = cli.main(["--root", root, "pending-judgment", "--claim", "--run-id", rid, "--judge", synth.JUDGE_ID_2, "--session", "s-C"])
    out = capsys.readouterr()
    assert rc == 0 and out.out.strip() in _judgment_rows(root, rid)


def test_the_judgeable_lifecycle_states_are_the_selections_own():
    """The states under which v_pending_judgment lists a run at all, stated once (`pending.JUDGEABLE_LIFECYCLE_STATES`) and read by the
    re-check and the parity diagnostic; the rendered SQL of v_pending_judgment names exactly them."""
    from sisteer.registry import schema_def as S
    sql = S.VIEW_SQL["v_pending_judgment"]
    rendered = "lifecycle_state IN (" + ", ".join(f"'{s}'" for s in P.JUDGEABLE_LIFECYCLE_STATES) + ")"
    assert rendered in sql
    assert set(P.JUDGEABLE_LIFECYCLE_STATES) == {"generated", "judging", "judged", "ingested"}


def test_claim_all_releases_the_first_claim_when_the_second_run_completed_after_the_selection(tmp_path):
    """`RegistryWorklist.claim_all` released the claims it took only on `ClaimHeld`, so a driver that claimed
    its first run and met `ClaimNotPending` on a later one (another session's completion folded between the selection and the claim) left
    the first claim orphaned `in_progress` with no judging behind it, and every later session met that claim as held. Two runs: B selects
    both; A claims and completes the SECOND and the cycle folds it; B's `claim_all` claims the first, is refused on the second
    (`ClaimNotPending`, re-raised), and has released the first -- abandoned, the cache empty, the second run carrying A's complete
    judgment alone and no claim of B's; the only record written beyond B's first claim is its release."""
    from sisteer.registry import worklist as W
    root, r1 = _generated(tmp_path)
    r2, recs = synth.run_records("k2", 300, through="generated", n_items=4)
    _write(root, "c", "000002-run.jsonl", recs)
    assert _ingest(root, synth.t(950)).refused == 0
    wl = W.RegistryWorklist(root, judges=[synth.JUDGE_ID], session="s-B")
    rows = sorted(wl.pending(), key=lambda r: [r1, r2].index(str(r["run_id"])))          # B's selection: r1 then r2, both no_judgment
    assert [str(r["run_id"]) for r in rows] == [r1, r2] and all(r["pending_reason"] == "no_judgment" for r in rows)
    # A claims and completes r2; the harvester's cycle folds it -- r2 leaves the selection B already holds
    a = P.claim(root, r2, synth.JUDGE_ID, "s-A", now=synth.t(1000))
    _complete(root, r2, a, synth.JUDGE_ID, "s-A", 1000, 2000)
    assert _ingest(root, synth.t(2500)).refused == 0 and _select(root, r2, synth.JUDGE_ID) is None
    files_before = _record_files(root)
    with pytest.raises(P.ClaimNotPending) as ei:
        wl.claim_all(rows)
    assert ei.value.run_id == r2 and isinstance(ei.value, P.ClaimNotFolded)
    # the first claim was released: abandoned, not orphaned in_progress; B holds nothing; a later session claims r1 at once
    rows1 = _judgment_rows(root, r1)
    assert len(rows1) == 1 and list(rows1.values())[0]["status"] == "abandoned" and list(rows1.values())[0]["claimed_by_session"] == "s-B"
    assert wl.claims == {} and _select(root, r1, synth.JUDGE_ID)["pending_reason"] == "abandoned"
    # r2 carries A's complete judgment alone: B never claimed it
    assert list(_judgment_rows(root, r2)) == [a] and _judgment_rows(root, r2)[a]["status"] == "complete"
    # the only records written by B are its first claim (before the race) and that claim's release, both under r1's claims directory;
    # beside them only the scoped fold's own arrival facts (records/harvester/) -- no claim file of B's on r2
    new_files = sorted(set(_record_files(root)) - set(files_before))
    new_claims = [f for f in new_files if f"{os.sep}records{os.sep}claims{os.sep}" in f]
    assert len(new_claims) == 2 and all(I.run_dir_name(r1) in f for f in new_claims), new_files
    assert all(f in new_claims or f"{os.sep}records{os.sep}harvester{os.sep}" in f for f in new_files), new_files
    assert len(_claim_files(root, r2)) == 1                                              # A's claim file alone
    c = P.claim(root, r1, synth.JUDGE_ID, "s-C", now=synth.t(3000))
    assert _judgment_rows(root, r1)[c]["status"] == "in_progress"
    _rebuild_equal(root, tmp_path)


#: the multi-run judging drivers that claim a whole worklist before their first judge call (worklist.claim_all) and refuse whole on a
#: claim refusal; the sweep below pins that every one of them handles EVERY refusal, not ClaimHeld alone
MULTI_RUN_DRIVERS = ("scripts/grokmt_grade.py", "scripts/served_capability_judge.py", "scripts/petri_run.py", "scripts/grade_bon_live.py",
                     "scripts/sdfbase/seedrep/grade_full22_native.py", "scripts/sdfbase/full22/static_padding/grade_padded_static.py",
                     "scripts/sdfbase/full22/misalign/misalign_judge.py", "scripts/sdfbase/full22/sampled/grade_static_sampled.py")


def test_every_multi_run_driver_refuses_whole_on_every_claim_refusal_not_claim_held_alone():
    """A `ClaimNotFolded` (a run not folded, a correction pending, a pair no longer pending) escaping a driver's `except ClaimHeld` was a
    traceback, never a refusal by name -- and `claim_all` has already released what it took, so the driver's own message is the right
    one. Every multi-run driver catches the pair; no ClaimHeld-only handler remains under scripts/ or the registry package."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in MULTI_RUN_DRIVERS:
        src = open(os.path.join(repo, rel), encoding="utf-8").read()
        assert "except (RW.P.ClaimHeld, RW.P.ClaimNotFolded) as e:" in src, rel
        assert "except RW.P.ClaimHeld as e:" not in src, rel
    wl = open(os.path.join(repo, "src", "sisteer", "registry", "worklist.py"), encoding="utf-8").read()
    assert "except (P.ClaimHeld, P.ClaimNotFolded):" in wl and "except P.ClaimHeld:" not in wl
    import subprocess
    stray = subprocess.run(["grep", "-rln", "--include=*.py", r"except RW\.P\.ClaimHeld as e:\|except P\.ClaimHeld:", os.path.join(repo, "scripts"),
                            os.path.join(repo, "src")], capture_output=True, text=True).stdout.split()
    assert stray == [], stray
