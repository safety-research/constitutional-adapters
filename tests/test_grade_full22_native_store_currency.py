"""QUARANTINEPIN: the full22 native grader's ONE-STORE check (`registry_worklist_pairs`) counts EVERY pending registration naming a
selected store -- the ones the caller's ``--run-id`` / ``--model-id`` selectors left out included.

Before the pin the rows grouped by store were only the SELECTED pending rows, so a store whose duplicate twin registrations were both
pending and current read as a current singleton whenever one twin alone was selected (the daemon passes exactly the run ids it
dispatches): an explicit quarantine release naming the survivor alone made the grader claim and grade it against an unsettled choice.
Now (1) two pending runs of one store, both current, one selected -> REFUSED by name listing BOTH ids and their bits, the unselected one
marked; (1b) the same through ``--model-id``; (2) the unselected twin superseded -> the selected current run grades the store, the twin is
named as covered, nothing else of the config (an unselected pending `pair` store) is touched; (2b) only the superseded twin selected while
the current twin is pending and unselected -> nothing is graded or claimed (the unselected current run never grades here); (3) both
selected (the daemon's shape today) -> the refusal text byte for byte as before; (4) a store with one pending current run -> unchanged.
The sibling module's synthetic world (test_grade_full22_native_main: fixture stores under a tmp root, a tmp registry, the real judge
stack over a stub API), real ``main()``; synthetic ids only.

The opt-in ``--grade-excluded-by`` is pinned at the end: a run an `exclude` ruling the flag names is the one thing retiring is graded on its
own reading and stays excluded; without the flag, under another ruling, beside a successor, a hold-out or a later supersession, or beside a
second registration of its store, every branch is as before; the flag is refused by name off its one path."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import worklist as W

import test_grade_full22_native_main as TM
from sisteer.registry import synth  # noqa: E402
from test_grade_full22_native_main import (  # noqa: F401  (module-scoped fixtures the tests below take by name)
    CELL,
    N_GOALS_MT,
    N_TURNS,
    OTHER_MODEL,
    _conn,
    _judgments,
    _provenance,
    _registry,
    _run_state,
    _tree,
    _two_runs_one_store,
    jbb_ids,
    reg_argv,
    run_of,
    store_of,
    stub_judge,
    twin,
    world,
)

NOT_SELECTED_MARK = "[not selected by --run-id / --model-id: counts toward the check, never graded here]"


def _pending_ids(root) -> set:
    with _conn(root) as conn:
        return {r["run_id"] for r in P.pending(conn)}


def _n_judgments(root) -> int:
    with _conn(root) as conn:
        return conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0]


#: BONRUNG: the PAPER's cap-512 metric set, as the registry carries it -- (metric_id, threshold): the eval's headline `asr` at the judge cutoff,
#: the asr_at_n{N} curve over the clamped schedule, bon_nauc (a non-headline row), and the four degeneracy shares at the no-threshold sentinel
THE_512_SET = {("asr", 0.5), *((f"asr_at_n{n}", 0.5) for n in (1, 2, 4, 8, 16, 32, 64, 75)), ("bon_nauc", 0.5),
               ("at_cap_share_bon_512", -1.0), ("loop_share_bon_512", -1.0), ("coherent_at_cap_share_bon_512", -1.0), ("secondary_flag_only_share_bon_512", -1.0)}


def _metric_set(conn, rid) -> set:
    """Every (metric_id, threshold, value, k, n) run_metric row of the run -- the tuple two complete judgments of ONE store at DIFFERENT rungs
    must never file identically (the incident's exact symptom)."""
    return {tuple(r) for r in conn.execute("SELECT metric_id, threshold, value, k, n FROM run_metric WHERE run_id = ?", (rid,))}


def _assert_512_set(conn, rid) -> None:
    """The run's metric rows are the paper's 512 set exactly (never asr_bon_cap4096, the 4,096 reading's id), its headline the eval's `asr` under
    the 512 store's instrument kind, its shares over every draw of the fixture store (none at cap), and its judgment carries the MEASURED at-cap
    and loop degeneration rows."""
    keys = {(r[0], r[1]) for r in conn.execute("SELECT metric_id, threshold FROM run_metric WHERE run_id = ?", (rid,))}
    assert keys == THE_512_SET, sorted(keys ^ THE_512_SET)
    head = conn.execute("SELECT k, n, extra_json FROM run_metric WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", (rid,)).fetchone()
    hx = json.loads(head[2])
    assert (head[1], hx["instrument_kind"], hx["cap"], hx["pass_label"], hx["reading"]) == (TM.N_GOALS_BON, "512_store", 512, "base_cap", "store_512"), (head, hx)
    assert hx["store_512_counts"]["n_rows_512"] == TM.N_GOALS_BON * TM.N_MAX and hx["store_512_counts"]["n_at_cap_512"] == 0
    shares = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT metric_id, k, n FROM run_metric WHERE run_id = ? AND threshold = -1.0", (rid,))}
    assert shares == {m: (0, TM.N_GOALS_BON * TM.N_MAX) for m in ("at_cap_share_bon_512", "loop_share_bon_512", "coherent_at_cap_share_bon_512", "secondary_flag_only_share_bon_512")}, shares
    jid = _judgments(conn, rid)[0]["judgment_id"]
    deg = {(r[0], r[1]): tuple(r[2:]) for r in conn.execute("SELECT measure, scope, measurement_state, numerator, denominator, denominator_unit FROM run_degeneration "
                                                            "WHERE run_id = ? AND judgment_id = ?", (rid, jid))}
    assert deg[("at_token_cap_finish_reason", "all")] == deg[("rep_ratio_loop", "all")] == ("measured", 0, TM.N_GOALS_BON * TM.N_MAX, "draws"), deg


# ------------------------------------------------------------------------------------------------------------ (1) both current, one selected


def test_two_current_registrations_of_one_store_refuse_by_name_when_the_caller_selects_only_one_of_them(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    root, rids, cur = _two_runs_one_store(world, tmp_path, monkeypatch, supersede_second=False)
    A, B = rids["s1/crescendo"], rids["s1/crescendo#g5678"]
    assert cur[A][0] == 1 and cur[B][0] == 1, cur
    before = _tree(root)
    for dry in (True, False):                                   # a registry dry run refuses what a real run would, before the plan
        for selected, other in ((A, B), (B, A)):
            with pytest.raises(SystemExit) as ei:
                twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[selected], dry=dry))
            msg = str(ei.value)
            assert f"two pending runs name the store of crescendo/{CELL} under {world['stores']['s1'] / run_of('s1')}: " in msg
            assert selected in msg and other in msg and "— one store, one run; refusing" in msg
            assert "2 of them current in v_run_current" in msg and f"{selected} is_current 1;" in msg and f"{other} is_current 1 {NOT_SELECTED_MARK}" in msg
            assert msg.endswith(f"; {twin.STORE_RUNS_REFUSAL_PHRASE})") and msg.count(NOT_SELECTED_MARK) == 1
    assert stub_judge["judgments"] == 0 and not (tmp_path / "g").exists() and _n_judgments(root) == 0 and _tree(root) == before
    assert _pending_ids(root) == {A, B, rids["s1/pair"]}


def test_the_model_id_selector_is_a_row_selector_too_the_other_models_registration_of_the_store_still_counts(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(1b) the twin is registered under ANOTHER model; ``--model-id`` selects the synthetic model's run alone. Both current -> refused listing
    both, the other model's run marked as not selected."""
    store = str(store_of(world, "s1", "crescendo"))
    root, rids = _registry(tmp_path, monkeypatch, [
        ("mine", f"{run_of('s1')}/crescendo/{CELL}", "crescendo", store, N_GOALS_MT, {}),
        ("theirs", f"{run_of('s1')}/crescendo/{CELL}#other", "crescendo", store, N_GOALS_MT, {"model_id": OTHER_MODEL})])
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", extra=["--model-id", synth.MODEL_ID]))
    msg = str(ei.value)
    assert rids["mine"] in msg and rids["theirs"] in msg and "2 of them current" in msg and f"{rids['theirs']} is_current 1 {NOT_SELECTED_MARK}" in msg
    assert f"{rids['mine']} is_current 1;" in msg and twin.STORE_RUNS_REFUSAL_PHRASE in msg
    assert stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and not (tmp_path / "g").exists()


# ------------------------------------------------------------------------------------------------------------ (2) the unselected twin superseded


def test_a_selected_current_run_grades_its_store_and_the_unselected_superseded_twin_is_named_as_covered(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The daemon dispatches the survivor alone after the twin's supersession folded: unchanged grading of the store under the current run,
    ONE judgment; the unselected twin is covered -- named on stdout, in the dry-run plan and in `covered_runs`, never claimed. The config's
    OTHER store (`pair`, pending, unselected) is not this invocation's: not graded, not covered, not counted as a pair."""
    root, rids, cur = _two_runs_one_store(world, tmp_path, monkeypatch, supersede_second=True)
    W, L, pair_rid = rids["s1/crescendo"], rids["s1/crescendo#g5678"], rids["s1/pair"]
    assert cur[W][0] == 1 and cur[L][0] == 0 and "supersede" in str(cur[L][1]), cur
    # the dry run: the plan names W as the pair and L as COVERED by W; the registry is byte-identical after it
    before = _tree(root)
    twin.main(reg_argv(world, "s1", root, attacks="crescendo,pair", grades=tmp_path / "g0", run_ids=[W], dry=True))
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={W}" in ln and "COVERED" not in ln for ln in plan) and any(f"run_id={L}  COVERED by {W}" in ln and "would not be claimed" in ln for ln in plan), plan
    assert not any(pair_rid in ln for ln in plan) and any("1 pending (attack, cell) pair(s)" in ln for ln in plan)
    assert _tree(root) == before and not (tmp_path / "g0").exists()
    # the real run
    grades = tmp_path / "grades"
    twin.main(reg_argv(world, "s1", root, attacks="crescendo,pair", grades=grades, run_ids=[W]))
    out, _ = capsys.readouterr()
    assert "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "1 pending run(s) covered by the current run of their store" in out
    assert "claimed 1 pending run(s)" in out
    cov = next(ln for ln in out.splitlines() if "covered by that run's judgment" in ln)
    assert L in cov and f"graded under {W}" in cov and "not claimed, not graded" in cov and "supersede" in cov
    assert stub_judge["judgments"] == N_GOALS_MT * N_TURNS                                        # the crescendo store judged ONCE; the pair store not at all
    prov = _provenance(grades, "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [W] and prov["registry"]["run_ids_selected"] == [W]
    assert prov["registry"]["covered_runs"] == [{"run_id": L, "covered_by_run_id": W, "why": cur[L][1], "attack": "crescendo", "cell": CELL}]
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, W)[0] == "judged"
        assert _judgments(conn, L) == [] and _run_state(conn, L)[0] == "generated"               # the twin: no judgment, no claim
        assert _judgments(conn, pair_rid) == [] and _run_state(conn, pair_rid)[0] == "generated"  # the unselected pair store: untouched
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 1
    assert _pending_ids(root) == {L, pair_rid}


def test_selecting_only_the_superseded_twin_while_the_current_one_is_pending_and_unselected_grades_nothing(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(2b) the mirror image: the current run is OUTSIDE the selection. It is never claimed or graded here; the selected twin is covered as a
    non-current singleton (no judgment on record yet -> covered by no run on record) and the invocation refuses with nothing claimed, nothing
    written -- a dry run says the same. Once the store IS graded under the current run (its own invocation), the twin reads covered by it."""
    root, rids, cur = _two_runs_one_store(world, tmp_path, monkeypatch, supersede_second=True)
    W, L = rids["s1/crescendo"], rids["s1/crescendo#g5678"]
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[L], dry=dry))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and "1 pending run(s) are not current and were covered, never graded" in msg
        assert L in msg and "covered by no run on record" in msg and "supersede" in msg and "nothing claimed, nothing written" in msg
        assert W not in msg.split("covered, never graded")[1], msg                 # the current run is nobody's cover yet and is not graded here
    out, _ = capsys.readouterr()
    assert "a store with no complete judgment on record: skipped" in out and L in out and "claimed" not in out.replace("not claimed", "")
    # the reason the twin is skipped is NAMED -- its rung's covering judgment is the unselected current run's, still to fold
    # (the count, never the id: the clause above names no run but the covered one and its cover)
    cl = next(ln for ln in out.splitlines() if "a store with no complete judgment on record: skipped" in ln)
    assert L in cl and "its rung's covering judgment is another run's, still to fold (1 other judgeable registration(s) of its store at the rung)" in cl, cl
    assert "(a retired run: its rung's covering judgment is another run's, still to fold (1 other judgeable registration(s) of its store at the rung))" in msg, msg
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 0
    assert _pending_ids(root) == {W, L, rids["s1/pair"]}
    # the current run's own invocation grades the store; afterwards the twin alone reads covered BY it
    twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g2", run_ids=[W]))
    capsys.readouterr()
    assert stub_judge["judgments"] == N_GOALS_MT * N_TURNS
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g3", run_ids=[L]))
    assert f"covered by {W}" in str(ei.value) and "nothing claimed, nothing written" in str(ei.value)
    assert stub_judge["judgments"] == N_GOALS_MT * N_TURNS and _n_judgments(root) == 1


# ------------------------------------------------------------------------------------------------------------ (3) both selected: the text as before


def test_selecting_both_current_registrations_refuses_with_the_previous_text_byte_for_byte(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The daemon's shape today (every pending run id of the group passed): no marker, the one-store refusal word for word."""
    root, rids, cur = _two_runs_one_store(world, tmp_path, monkeypatch, supersede_second=False)
    A, B = sorted((rids["s1/crescendo"], rids["s1/crescendo#g5678"]))             # the worklist orders pending rows by run id
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A, B]))
    expected = (f"{twin.DRIVER_ID}: two pending runs name the store of crescendo/{CELL} under {world['stores']['s1'] / run_of('s1')}: "
                f"{A} and {B} — one store, one run; refusing (2 of them current in {twin.RUN_CURRENCY_VIEW}: {A} is_current 1; {B} is_current 1; "
                f"{twin.STORE_RUNS_REFUSAL_PHRASE})")
    assert str(ei.value) == expected
    assert NOT_SELECTED_MARK not in str(ei.value) and stub_judge["judgments"] == 0 and _n_judgments(root) == 0


# ------------------------------------------------------------------------------------------------------------ (4) a single pending current run


def test_a_store_with_a_single_pending_current_run_is_unchanged(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The `pair` store beside the twins: selected alone it grades as before -- no covered line, `covered_runs` empty -- and the twins'
    store (two current registrations, neither selected) is not this invocation's: no refusal, nothing of it touched."""
    root, rids, cur = _two_runs_one_store(world, tmp_path, monkeypatch, supersede_second=False)
    A, B, pair_rid = rids["s1/crescendo"], rids["s1/crescendo#g5678"], rids["s1/pair"]
    twin.main(reg_argv(world, "s1", root, attacks="crescendo,pair", grades=tmp_path / "g", run_ids=[pair_rid]))
    out, _ = capsys.readouterr()
    assert "covered by" not in out and "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "claimed 1 pending run(s)" in out
    assert _provenance(tmp_path / "g", "s1")["registry"]["covered_runs"] == [] and stub_judge["judgments"] == N_GOALS_MT * N_TURNS
    # without --grade-excluded-by the provenance carries none of its keys (the default output is as before the opt-in existed)
    assert not {"grade_excluded_by", "graded_excluded"} & set(_provenance(tmp_path / "g", "s1")["registry"])
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, pair_rid)[0] == "judged" and _judgments(conn, A) == [] and _judgments(conn, B) == []
    assert _pending_ids(root) == {A, B}


def test_pending_beside_selection_is_empty_without_a_selector_and_never_repeats_a_selected_row(twin, world, tmp_path, monkeypatch):
    """The helper itself: no selector -> [] (the selected read is already every row); a --run-id selection -> exactly the rows it left out, by
    (run, required judge), located like the selected ones."""
    from sisteer.registry import worklist as W_
    root, rids, _ = _two_runs_one_store(world, tmp_path, monkeypatch, supersede_second=False)
    A, B, pair_rid = rids["s1/crescendo"], rids["s1/crescendo#g5678"], rids["s1/pair"]
    wl_all = W_.RegistryWorklist(root, None, "s-T", driver="tests/quarantinepin")
    rows_all = wl_all.pending(eval_id=("crescendo", "pair"))
    assert {r["run_id"] for r in rows_all} == {A, B, pair_rid} and twin.pending_beside_selection(wl_all, rows_all, eval_id=("crescendo", "pair")) == []
    wl_one = W_.RegistryWorklist(root, None, "s-T", driver="tests/quarantinepin", run_ids=[A])
    rows_one = wl_one.pending(eval_id=("crescendo", "pair"))
    beside = twin.pending_beside_selection(wl_one, rows_one, eval_id=("crescendo", "pair"))
    assert [r["run_id"] for r in rows_one] == [A] and sorted(r["run_id"] for r in beside) == sorted([B, pair_rid])
    assert all("store_path" in r and "view_legacy" in r for r in beside)          # located exactly as the selected rows are
    assert wl_one.claims == {} and wl_all.claims == {}


# ------------------------------------------------------------------- a BoN store registered twice

REPO_ = Path(__file__).resolve().parents[1]
CLOSER = REPO_ / "scripts" / "sdfbase" / "full22" / "registry_close_stores.py"


def _closer():
    return TM._load("gfn_closer_for_currency", CLOSER)


def test_a_bon_store_registered_by_the_store512_job_and_the_climb_is_a_current_singleton_once_the_closer_files_the_climb_superseded(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The smoke's BoN shape: the 512 store is registered by the store512 job AND by the climb array — two pending `generated` runs
    naming one store, both current -> the grader refuses the store by name (one store, one run; awaiting a supersession ruling). The
    closer's record settles it WITHOUT a ruling: the PRODUCER's `run_state superseded` on the climb run (file_superseded, reason
    duplicate_registration, extra.superseded_by_run_id = the store512 run; the census effect applies) folds the climb to lifecycle
    superseded, v_run_current reads it is_current 0 naming the rule, it leaves the pending worklist, and the store is a CURRENT
    SINGLETON: graded once under the store512 run, one judgment, nothing on the climb run."""
    import os as _os
    from sisteer.registry import emit as _emit
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, rids = _registry(tmp_path, monkeypatch, [
        ("s1/bon", f"{run_of('s1')}/{CELL}", "strongreject_bon", store, TM.N_GOALS_BON, {}),
        ("s1/bon#climb", f"{run_of('s1')}/{CELL}#climb", "strongreject_bon", store, TM.N_GOALS_BON, {})])
    W, L = rids["s1/bon"], rids["s1/bon#climb"]
    before = _tree(root)
    for dry in (True, False):                                                         # both current: refused by name, nothing claimed
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=[W, L], dry=dry))
        msg = str(ei.value)
        assert f"two pending runs name the store of bon/{CELL} under" in msg and W in msg and L in msg
        assert "2 of them current in v_run_current" in msg and twin.STORE_RUNS_REFUSAL_PHRASE in msg
    assert stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and _tree(root) == before and _pending_ids(root) == {W, L}
    # the closer's half: the climb registration filed superseded by the store512's — the record the job-side closer writes
    closer = _closer()
    facts = {"store_uri": store, "store_completion_sha256": "a" * 64, "n_items_generated": TM.N_GOALS_BON}
    closer.file_superseded(spool=_os.path.join(root, "records"), push_script=None, campaign=synth.CAMPAIGN, rid=L, by_rid=W, artifact=store,
                           producer=_emit.producer_block(closer.SCRIPT, code_sha=synth.code_sha("b")), sibling_job="810081", eval_id="strongreject_bon",
                           obj=CELL, dry_run=False, decision=closer.BON_SUPERSEDED_DECISION, rule=closer.BON_SUPERSEDED_RULE, kind=closer.BON_SUPERSEDED_KIND,
                           extra={"duplicate_of": facts, "superseded_role": "climb", "survivor_role": "store512"})
    out, _ = capsys.readouterr()
    assert f"superseded strongreject_bon {CELL} {L[:40]}… by {W[:40]}… (a climb registration duplicating the store512's; D2599)" in out
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (W, L)}
        assert cur[W] == (1, None) and cur[L][0] == 0 and "superseded" in str(cur[L][1]), cur
        assert _run_state(conn, L)[0] == "superseded" and _run_state(conn, W)[0] == "generated"
        ev = [json.loads(x) for (x,) in conn.execute("SELECT extra_json FROM run_state_event WHERE run_id = ? AND state = 'superseded'", (L,))]
        assert len(ev) == 1 and ev[0]["superseded_by_run_id"] == W and ev[0]["reason"] == "duplicate_registration" and ev[0]["decision"] == "D2599", ev
        assert ev[0]["duplicate_of"] == facts and ev[0]["sibling_job_id"] == "810081"
        assert conn.execute("SELECT COUNT(*) FROM ruling").fetchone()[0] == 0                       # no ruling: the producer's record alone
    assert _pending_ids(root) == {W}                                                                 # a superseded run admits no judging
    # the store is a current singleton: graded once under the store512 run, whether the daemon selects both ids or the survivor alone
    grades = tmp_path / "grades"
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=grades, run_ids=[W, L]))
    out, _ = capsys.readouterr()
    assert "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "claimed 1 pending run(s)" in out and "refusing" not in out
    assert stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX                                      # the store judged ONCE
    prov = _provenance(grades, "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [W] and prov["registry"]["covered_runs"] == []
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, W)[0] == "judged" and _judgments(conn, L) == [] and _run_state(conn, L)[0] == "superseded"
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 1
    assert _pending_ids(root) == set()


# ------------------------------------------------------------------------------- a two-rung BoN ladder


RUNG_512 = {"token_cap_tokens": 512, "token_cap_rung_index": 0, "token_cap_ladder_json": [512, 4096], "pass_label": "base_cap"}
RUNG_4096 = {"token_cap_tokens": 4096, "token_cap_rung_index": 1, "token_cap_ladder_json": [512, 4096], "pass_label": "ladder_rung"}
#: the synthetic mirror of the registry-wide r2v1 rule: every cap-512 BoN row reads superseded (the floor is KEPT for the curve); a KEY-scoped
#: `supersede` whose match is the eval and the cap, exactly the live ruling's shape (instrument_selector_json.token_cap_tokens -> token_caps)
R2V1_LIKE = "r_synth-r2v1-supersede-bon-512-floor"
#: a `supersede` that NAMES a run in its match (`run_ids`) -- a retirement of the run itself, never of its rung
BY_NAME = "r_synth-supersede-rerun512-floor-by-name"
#: the same shape over the CLIMB -- a `supersede` naming the climb registration(s) of the store (a mis-served climb re-run elsewhere)
BY_NAME_CLIMB = "r_synth-supersede-rerunclimb-by-name"


def _bon_rung_recs(key: str, i0: int, store: str, rung: dict, *, through: str = "generated") -> tuple:
    """A BoN run over ``store`` registered AT A RUNG of the 512 / 4,096 ladder -- the DRAWN cap, its index, the ladder and the pass label on
    the registration record, as jobplan stamps the rerun pair. ``through`` = "generated" (the five records the drivers test
    writes), "failed" (queued, running, failed: the job died before writing -- a registration no longer judgeable) or
    "judged" (generated, then a COMPLETE judgment by the instrument's judge and the judged state -- a complete run, the one shape
    v_run_current's later rule reads as a successor; the run's own `judge_id` column stays as registered, so the currency
    key's `judge_id IS` term pairs it with a predecessor registered the same way)."""
    from sisteer.registry import identity as I
    inst = TM._instrument("strongreject_bon")
    p = synth.run_payload(key, CELL, synth.OBJECT_SHA, i0, eval_id="strongreject_bon", item_set_id="strongreject_bon_set", instrument_id=inst["instrument_id"],
                          decode_source="generation_config", store_uri=store, model_id=synth.MODEL_ID, serving_path="lora_native",
                          n_items_planned=TM.N_GOALS_BON, **rung)
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, CELL, "strongreject_bon", str(inst["instrument_id"]))
    rid, t = str(p["run_id"]), synth.t
    recs = [synth.rec("run", p, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log"}, t(i0 + 1))]
    if through == "failed":
        recs.append(synth.rec("run_state", {"run_id": rid, "state": "failed", "event_utc": t(i0 + 3), "evidence_kind": "job_log"}, t(i0 + 3)))
        return rid, recs
    assert through in ("generated", "judged"), through
    recs += [synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                     "evidence_uri": store}, t(i0 + 3)),
             synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": TM.N_GOALS_BON}}, t(i0 + 4))]
    if through == "judged":
        judge = str(inst["judge_id"])
        jid = I.judgment_id(rid, judge)
        recs += [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": judge, "started_utc": t(i0 + 5), "finished_utc": t(i0 + 6),
                                        "n_items_judged": TM.N_GOALS_BON, "status": "complete", "api_pool": "lowprio"}, t(i0 + 6)),
                 synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"}, t(i0 + 7)),
                 synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": TM.N_GOALS_BON}}, t(i0 + 8))]
    return rid, recs


def _supersede_ruling(ruling_id: str, scope: dict, reason: str, i: int) -> dict:
    """A `supersede` ruling record of the synthetic round, scope as given (the normaliser builds ``match`` when absent), ruled at ``t(i)``."""
    from sisteer.registry.adapters import roster as RO
    raw = {"ruling_id": ruling_id, "round_id": "r_synth", "action": "supersede", "scope_kind": "key", "scope_json": scope, "reason": reason,
           "ruled_utc": synth.t(i), "ruled_by": "tests", "source_ref": "tests/test_grade_full22_native_store_currency.py",
           "provenance": {"kind": "user_direct", "message_utc": "2026-10-13T00:00:00Z"}}
    return synth.rec("ruling", RO.normalise_ruling(raw), synth.t(i))


def _r2v1_like_ruling() -> dict:
    """The synthetic mirror of the registry-wide r2v1 512-floor rule (:data:`R2V1_LIKE`): a KEY-scoped `supersede` matching the eval and the cap."""
    return _supersede_ruling(R2V1_LIKE, {"eval_ids": ["strongreject_bon"], "instrument_selector_json": {"token_cap_tokens": 512}},
                             "synthetic mirror of the r2v1 512-floor rule: the cap-512 BoN rows read superseded registry-wide; the floor is kept for the curve", 300)


def _two_rung_ladder(world, tmp_path, monkeypatch, *, climb_failed: bool = False, floor_by_name: bool = False) -> tuple:
    """The smoke's rerun pair (synthetic ids): ONE BoN store registered twice -- the 512 store run (cap 512, rung 0, base_cap) and its 4,096 climb
    (cap 4096, rung 1, ladder_rung), identical store facts -- under the r2v1-shaped key ruling that retires every cap-512 BoN row. Returns
    ``(root, store, X = the 512 run, Y = the climb, {run_id: (is_current, why)})``. Variants: ``climb_failed`` registers the
    climb through `failed` (the climb job died: no judgeable registration beside the floor); ``floor_by_name`` adds a second `supersede`
    ruling that NAMES X in its match (``run_ids``), ruled after the rung rule, so it is X's decisive ruling -- a retirement of
    the run, not of its rung."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X, recs_x = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    Y, recs_y = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerunclimb", 210, store, RUNG_4096, through="failed" if climb_failed else "generated")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs_x)
    R.write_records(os.path.join(root, "records", "c", "000002-run.jsonl"), recs_y)
    rulings = [_r2v1_like_ruling()]
    if floor_by_name:
        rulings.append(_supersede_ruling(BY_NAME, {"match": {"run_ids": [X]}, "note": "synthetic: the floor retired BY NAME (D471's shape: a mis-served cell re-run elsewhere)"},
                                         "synthetic mirror of a run-scoped supersede: this run is retired, not its rung", 400))
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), rulings)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, Y)}
        rungs = {rid: tuple(conn.execute("SELECT token_cap_tokens, pass_label FROM run WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, Y)}
    assert rungs == {X: (512, "base_cap"), Y: (4096, "ladder_rung")}, rungs
    # the smoke's bits: the 512 run retired by the rule (or, by_name, by the ruling that names it), the climb current (or failed)
    assert cur[X][0] == 0 and (BY_NAME if floor_by_name else R2V1_LIKE) in str(cur[X][1]), cur
    assert cur[Y] == ((0, "lifecycle_state failed") if climb_failed else (1, None)), cur
    return root, store, X, Y, cur


def _row(wl, rid):
    return next(r for r in wl.pending() if r["run_id"] == rid)


def test_g37_rung_a_non_current_512_rung_of_a_two_rung_ladder_is_graded_on_its_own_reading_and_a_same_rung_twin_is_covered_never_graded_twice(
        twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The smoke's rerun pair: the 512 run X (cap 512 / base_cap, is_current 0 under
    the r2v1-shaped rule) and the climb Y (cap 4096 / ladder_rung, current) name ONE store with identical facts. (1) Both pending and selected:
    Y is the store's current run and takes the pair; X is NOT covered by it (another rung) -- DEFERRED by name, located (no refuse_unlocated), on
    stdout and in the dry-run plan -- and Y, a CLIMB with no locatable climb root (no extra_json.climb_root, no --bon-climb-root override), is
    REFUSED BY NAME BEFORE ANY CLAIM (BONRUNG / BONFOLD): its reading is the climb fold, which needs its shards located, and it is never again graded
    from the store's cap-512 draws under the climb's name (the demo's 36798a64). The dry run prints
    the plan with Y's line marked REFUSED and X DEFERRED, then refuses the same way; the real run refuses with nothing claimed; so does Y selected
    alone (the daemon's dispatch of the climb). (2) X alone, the SOLE selected run of its store, non-current: the grader used to cover it by Y's
    judgment and quarantine its owed cap-512 reading; now judged_run_of_store answers None across rungs, store_rung_siblings reads it as the only
    registration of its rung beside Y's, and X is GRADED on its own reading -- claimed, a judgment of its own, named on stdout, in the dry-run
    plan (with the reading its rung selects) and in `registry.rung_ladder.graded_on_own_reading`; its metric rows are the PAPER's 512 set (the
    eval's headline `asr`, the curve, bon_nauc, the four 512 shares; never asr_bon_cap4096 -- the incident's exact symptom); X stays is_current 0
    (the rule) and leaves the pending worklist; Y stays pending with no judgment. (3) A same-rung twin X2 (cap 512 / base_cap, retired by the
    same rule) registered afterwards is COVERED by X's judgment exactly as a same-cap duplicate is covered -- never claimed, never graded, no
    second judgment -- and refuses with nothing written, dry or real."""
    root, store, X, Y, cur = _two_rung_ladder(world, tmp_path, monkeypatch)
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rung")
    assert twin.store_rung_siblings(wl, _row(wl, X)) == {"rung": (512, "base_cap"), "ladder": [512, 4096], "rung_index": 0, "same_rung": [],
                                                         "other_rungs": {Y: (4096, "ladder_rung")}, "not_judgeable": {}}
    assert twin.store_rung_siblings(wl, _row(wl, Y)) == {"rung": (4096, "ladder_rung"), "ladder": [512, 4096], "rung_index": 1, "same_rung": [],
                                                         "other_rungs": {X: (512, "base_cap")}, "not_judgeable": {}}
    # (1) both selected, and the climb alone: Y takes the pair and is REFUSED by name before any claim, dry and real; X is DEFERRED by name
    # (located, never covered); the registry byte-identical after every invocation
    before = _tree(root)
    for rids in ([X, Y], [Y]):
        for dry in (True, False):
            with pytest.raises(SystemExit) as ei:
                twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=rids, dry=dry))
            msg = str(ei.value)
            assert msg.startswith(f"[{TM.tag('s1')}] --worklist registry: refusing — {twin.DRIVER_ID}: run {Y} is a BoN CLIMB (cap 4096 / ladder_rung, "
                                  "ladder [512, 4096]) and is REFUSED before any claim: its reading is the climb fold"), msg
            assert "out/<cell>/shard*.jsonl folded over the 512 store" in msg and "located by NO admitted locator" in msg and "FLOOR's reading, never the climb's" in msg
            assert "extra_json.climb_root" in msg and f"--bon-climb-root {Y}=<root>" in msg          # BONFOLD: the two locators an unlocated climb lacks
            assert "nothing claimed, nothing written" in msg and X not in msg
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    y_lines = [ln for ln in plan if f"run_id={Y}" in ln]
    assert len(y_lines) == 2 and all("— REFUSED (BONRUNG), would not be claimed or graded: " in ln and "is a BoN CLIMB" in ln for ln in y_lines), plan   # the two dry runs' plans
    assert any(f"run_id={X}  DEFERRED at cap 512 / base_cap" in ln and f"{Y} takes bon/{CELL}" in ln and "would not be claimed, graded or covered here" in ln for ln in plan), plan
    assert not any("COVERED" in ln for ln in plan)
    dl = [ln for ln in out.splitlines() if "deferred:" in ln]
    assert len(dl) == 4 and all(X in ln and "is the only judgeable registration of its store at cap 512 / base_cap" in ln and f"{Y} at cap 4096 / ladder_rung" in ln for ln in dl), dl
    assert all("a different rung of the store's ladder, never covered by another rung's judgment" in ln and f"{Y} takes bon/{CELL} in this invocation" in ln for ln in dl)
    assert "covered by that run's judgment" not in out and "no complete judgment on record" not in out       # no covered line: X is not covered
    assert "[registry] claimed " not in out and "gates passed" not in out and f"bon/{CELL}: rung cap 4096" not in out   # nothing claimed; no reading line for a refused climb
    assert stub_judge["judgments"] == 0 and _tree(root) == before and not (tmp_path / "g0").exists() and _n_judgments(root) == 0
    assert _pending_ids(root) == {X, Y}
    with _conn(root) as conn:
        assert _judgments(conn, Y) == [] and _run_state(conn, Y)[0] == "generated" and _judgments(conn, X) == [] and _run_state(conn, X)[0] == "generated"
    # the rung guard itself: Y covers NOTHING at cap 512, graded or not (before the rung fix judged_run_of_store returned Y here once
    # its judgment stood, and the 512 reading was quarantined as covered)
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rung")
    assert twin.judged_run_of_store(wl, _row(wl, X)) is None
    # (2) X alone: a dry run names it graded on its own reading, with the reading its rung selects; the real pass grades it -- a judgment of
    # its own, nothing covered, the paper's 512 set filed
    before = _tree(root)
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2d", run_ids=[X], dry=True))
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={X}" in ln and "reading=store_512 (the store's cap-512 draws (bon_responses.jsonl); rung cap 512 / base_cap, instrument_kind 512_store)" in ln
               and "a different rung of its store's ladder (cap 512 / base_cap), graded on its own reading (g37-rung)" in ln for ln in plan), plan
    assert f"[registry] {twin.DRIVER_ID}: bon/{CELL}: rung cap 512 / base_cap (ladder [512, 4096]) -> reading store_512 (the store's cap-512 draws (bon_responses.jsonl)); instrument_kind 512_store" in out
    assert not any("COVERED" in ln or "DEFERRED" in ln or "REFUSED" in ln for ln in plan) and _tree(root) == before and not (tmp_path / "g2d").exists()
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2", run_ids=[X]))
    out, _ = capsys.readouterr()
    gl = next(ln for ln in out.splitlines() if "graded on its own reading in this invocation" in ln)
    assert X in gl and f"is not current ({cur[X][1]})" in gl and f"beside {Y} at cap 4096 / ladder_rung" in gl and "D654" in gl, gl
    assert "covered by that run's judgment" not in out and "no complete judgment on record" not in out and "deferred" not in out
    assert "1 pending run(s) graded on their own reading as a different rung of their store's ladder (g37-rung)" in out
    assert "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "claimed 1 pending run(s)" in out and "refusing" not in out
    assert stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX                                      # the 512 reading: its own judge calls, the store's first
    prov = _provenance(tmp_path / "g2", "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [X] and prov["registry"]["covered_runs"] == [] and prov["registry"]["rung_ladder"]["deferred"] == []
    (g,) = prov["registry"]["rung_ladder"]["graded_on_own_reading"]
    assert g["run_id"] == X and g["why"] == cur[X][1] and g["rung"] == [512, "base_cap"] and g["other_rungs"] == {Y: [4096, "ladder_rung"]} and "deferred_why" not in g
    assert g["reading"] == "the store's cap-512 draws (bon_responses.jsonl)" and prov["registry"]["bon_headline_metric_id"] == "asr"
    assert prov["registry"]["bon_rungs"] == {CELL: {"cap": 512, "pass_label": "base_cap", "ladder": [512, 4096], "reading": "store_512", "instrument_kind": "512_store",
                                                    "instrument": {"read_at": "fixed", "ladder": [512], "cap": 512, "instrument_id": str(TM._instrument("strongreject_bon")["instrument_id"])}}}   # BONFOLD
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, X)[0] == "judged" and len(_judgments(conn, X)) == 1 and _judgments(conn, Y) == [] and _run_state(conn, Y)[0] == "generated"
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 1                        # the floor's judgment alone: the climb has none until its fold exists
        assert tuple(conn.execute("SELECT is_current FROM v_run_current WHERE run_id = ?", (X,)).fetchone()) == (0,)   # the rule stands; the reading is on record
        _assert_512_set(conn, X)                                                                     # the incident's exact symptom, pinned: the 512 set, never the climb's labels
    assert _pending_ids(root) == {Y}                                                                   # the climb's pending row stands until its fold exists
    # (3) a same-rung twin of X, registered afterwards (retired by the same rule): covered by X's judgment, never graded -- dry and real
    X2, recs2 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512-twin", 230, store, RUNG_512)
    R.write_records(os.path.join(root, "records", "c", "000003-run.jsonl"), recs2)
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    assert _pending_ids(root) == {X2, Y}
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rung")
    assert twin.judged_run_of_store(wl, _row(wl, X2)) == X
    assert twin.store_rung_siblings(wl, _row(wl, X2)) == {"rung": (512, "base_cap"), "ladder": [512, 4096], "rung_index": 0, "same_rung": [X],
                                                          "other_rungs": {Y: (4096, "ladder_rung")}, "not_judgeable": {}}
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g3", run_ids=[X2], dry=dry))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and "1 pending run(s) are not current and were covered, never graded" in msg
        assert X2 in msg and f"covered by {X}" in msg and R2V1_LIKE in msg and "nothing claimed, nothing written" in msg and "deferred" not in msg
    out, _ = capsys.readouterr()
    cl = next(ln for ln in out.splitlines() if "covered by that run's judgment" in ln)
    assert X2 in cl and f"graded under {X}" in cl and "not claimed, not graded" in cl and "own reading" not in out
    assert _tree(root) == before and not (tmp_path / "g3").exists() and stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX
    assert _n_judgments(root) == 1 and _pending_ids(root) == {X2, Y}


def test_g37_rung_the_smokes_pair_shape_the_512_run_is_graded_not_covered_by_the_climbs_judgment(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The smoke's exact sequence through main() alone (no helper of the fix is named, so an unfixed grader fails at the DEFECT, not at an
    import): the climb Y carries a COMPLETE judgment on record -- the one the pre-BONRUNG grader filed from the store's cap-512 draws under the
    climb's labels (the demo's 36798a64: asr_bon_cap4096 / asr_at_n{N} / bon_nauc, registered here as that judgment's records) -- and the
    daemon dispatches the 512 run X alone, the sole pending run of its store. Before the rung fix the grader refused X as "covered by that
    run's judgment" (Y's) with nothing claimed -- a quarantine of the owed cap-512 reading; before BONRUNG it graded X and filed EXACTLY Y's
    metric set and values under X (the incident: two rungs, one store, identical (metric_id, threshold, value, k, n) rows). Now X is claimed
    and graded on its own reading and files the PAPER's 512 set -- the eval's headline `asr`, the curve, bon_nauc, the four 512 shares, the
    measured at-cap / loop degeneration rows -- never asr_bon_cap4096: the two judgments of the store differ as sets, the cross-rung invariant
    this incident broke. The climb's own rows are untouched: their correction is the producer's refile, never this
    grader's."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X, recs_x = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    Y, recs_y = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerunclimb", 210, store, RUNG_4096, through="judged")
    jid_y = I.judgment_id(Y, str(TM._instrument("strongreject_bon")["judge_id"]))
    mislabelled = ([("asr_bon_cap4096", 0.5, 0.0, 0, TM.N_GOALS_BON, f"asr_at_n[{TM.N_MAX}] (n_hits_best_of_n / n_goals)", {"bon_n": TM.N_MAX, "instrument_kind": "native_4096"})]
                   + [(f"asr_at_n{n}", 0.5, 0.0, 0, TM.N_GOALS_BON, f"asr_at_n[{n}]", {"bon_n": n, "n_max": TM.N_MAX}) for n in (1, 2, 4, 8, 16, 32, 64, 75)]
                   + [("bon_nauc", 0.5, 0.0, None, TM.N_GOALS_BON, "mean_normalized_auc", {"bon_n": TM.N_MAX})])
    recs_y += [synth.rec("run_metric", {"run_id": Y, "judgment_id": jid_y, "metric_id": mid, "threshold": thr, "value": v, "k": k, "n": n,
                                        "producer_metric_spelling": sp, "computed_by": "producer", "extra": ex}, synth.t(219 + i))
               for i, (mid, thr, v, k, n, sp, ex) in enumerate(mislabelled)]
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs_x)
    R.write_records(os.path.join(root, "records", "c", "000002-run.jsonl"), recs_y)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, Y)}
        assert _run_state(conn, Y)[0] == "judged" and [j["judgment_id"] for j in _judgments(conn, Y)] == [jid_y]
        y_set = _metric_set(conn, Y)
    assert cur[X][0] == 0 and R2V1_LIKE in str(cur[X][1]) and cur[Y][0] == 1 and _pending_ids(root) == {X}
    assert ("asr_bon_cap4096", 0.5, 0.0, 0, TM.N_GOALS_BON) in y_set and len(y_set) == 10              # the climb's mislabelled set, as the smoke carries it
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2", run_ids=[X]))      # the unfixed grader: SystemExit "covered by Y"
    out, _ = capsys.readouterr()
    assert "claimed 1 pending run(s)" in out and f"covered by {Y}" not in out and "covered by that run's judgment" not in out
    assert f"[registry] {twin.DRIVER_ID}: bon/{CELL}: rung cap 512 / base_cap (ladder [512, 4096]) -> reading store_512 (the store's cap-512 draws (bon_responses.jsonl)); instrument_kind 512_store" in out
    assert stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX
    prov = _provenance(tmp_path / "g2", "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [X] and prov["registry"]["covered_runs"] == []
    assert prov["registry"]["bon_headline_metric_id"] == "asr" and prov["registry"]["bon_rungs"][CELL]["reading"] == "store_512"
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, X)[0] == "judged" and len(_judgments(conn, X)) == 1 and len(_judgments(conn, Y)) == 1
        assert tuple(conn.execute("SELECT token_cap_tokens, pass_label FROM run WHERE run_id = ?", (X,)).fetchone()) == (512, "base_cap")
        _assert_512_set(conn, X)
        x_set = _metric_set(conn, X)
        # the cross-rung invariant: two complete judgments of ONE store at different rungs never file identical (metric_id, threshold, value, k, n) rows
        assert x_set != y_set and ("asr", 0.5, 0.0, 0, TM.N_GOALS_BON) in x_set and not {m for m, *_ in x_set} & {"asr_bon_cap4096"}
        assert ("asr_bon_cap4096", 0.5, 0.0, 0, TM.N_GOALS_BON) in y_set - x_set and len(x_set - y_set) == 5                # the headline and the four shares are X's own
        assert _metric_set(conn, Y) == y_set                                                        # the climb's rows untouched
    assert _pending_ids(root) == set() and _n_judgments(root) == 2


# ------------------------------------------------- the ladder is read off the run's OWN row


def test_g37_rung_floor_a_failed_climb_never_strands_the_512_floor_which_is_graded_on_its_own_reading(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The smoke's pair with the CLIMB FAILED: Y registered queued, running, failed (the climb job died -- no judgeable registration beside the
    floor), X generated and non-current under the r2v1-shaped rung rule. The round-1 fix read "a rung of a ladder" off the store's OTHER
    judgeable registration, lost it with Y's lifecycle, took the lone-retired-run branch and refused X as covered by no run -- its owed cap-512
    grade pending indefinitely with the store on disk. Now the ladder is X's own row's ([512, 4096], base_cap) and the rule retiring it is a
    rung rule (token_caps [512], no run_ids): X is graded on its own reading, dry and real, Y named as no longer judgeable; one judgment on
    the store, X's; X stays is_current 0 (the rule) and leaves the pending worklist. main() runs first, so an unfixed grader fails at the
    defect (SystemExit: covered by no run on record), not at a helper's name."""
    root, store, X, Y, cur = _two_rung_ladder(world, tmp_path, monkeypatch, climb_failed=True)
    assert _pending_ids(root) == {X}                                                                   # Y admits no judging; X is owed its grade
    before = _tree(root)
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=[X], dry=True))    # the unfixed grader: SystemExit here
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={X}" in ln and "a different rung of its store's ladder (cap 512 / base_cap), graded on its own reading (g37-rung)" in ln for ln in plan), plan
    assert not any("COVERED" in ln or "DEFERRED" in ln for ln in plan) and _tree(root) == before and not (tmp_path / "g0").exists()
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g1", run_ids=[X]))
    out, _ = capsys.readouterr()
    gl = next(ln for ln in out.splitlines() if "graded on its own reading in this invocation" in ln)
    assert X in gl and f"is not current ({cur[X][1]})" in gl and "is the only judgeable registration of its store at cap 512 / base_cap" in gl, gl
    assert f"with no other judgeable registration of its store ({Y} at cap 4096 / ladder_rung (lifecycle failed): no longer judgeable)" in gl, gl
    assert "a different rung of the store's ladder, never covered by another rung's judgment" in gl and "D654" in gl
    assert (f"a rung declared by its row's ladder [512, 4096], retired from the headline by the rung rule {R2V1_LIKE} alone "
            "(it selects by token_caps [512], never by run id: the floor is kept for the curve)") in gl, gl
    assert "a retired run never grades its store" not in out and "covered by that run's judgment" not in out and "deferred" not in out
    assert "1 pending run(s) graded on their own reading as a different rung of their store's ladder (g37-rung)" in out
    assert "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "claimed 1 pending run(s)" in out and "refusing" not in out
    assert stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX                                      # the floor's reading: its own judge calls
    prov = _provenance(tmp_path / "g1", "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [X] and prov["registry"]["covered_runs"] == [] and prov["registry"]["rung_ladder"]["deferred"] == []
    (g,) = prov["registry"]["rung_ladder"]["graded_on_own_reading"]
    assert g["run_id"] == X and g["why"] == cur[X][1] and g["rung"] == [512, "base_cap"] and g["ladder"] == [512, 4096] and "deferred_why" not in g
    assert g["declared_by"] == "its row's ladder [512, 4096]" and g["rung_rule"] == R2V1_LIKE and g["rule_selects_by"] == "token_caps [512]"
    assert g["other_rungs"] == {} and g["not_judgeable"] == {Y: [4096, "ladder_rung", "failed"]}
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, X)[0] == "judged" and len(_judgments(conn, X)) == 1 and _judgments(conn, Y) == [] and _run_state(conn, Y)[0] == "failed"
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 1                        # the floor's judgment alone
        assert tuple(conn.execute("SELECT is_current FROM v_run_current WHERE run_id = ?", (X,)).fetchone()) == (0,)   # the rule stands; the reading is on record
    assert _pending_ids(root) == set()
    # the helpers the decision rests on: the ladder off X's OWN row, the rung rule off the view's decisive columns, Y under not_judgeable
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rungfloor")
    row = {"run_id": X, "store_uri": store, "eval_id": "strongreject_bon"}
    sib = twin.store_rung_siblings(wl, row)
    assert sib == {"rung": (512, "base_cap"), "ladder": [512, 4096], "rung_index": 0, "same_rung": [], "other_rungs": {},
                   "not_judgeable": {Y: (4096, "ladder_rung", "failed")}}, sib
    assert twin.declared_ladder_of(sib) == "its row's ladder [512, 4096]"
    assert twin.rung_rule_of(wl, row) == {"ruling_id": R2V1_LIKE, "selects_by": "token_caps [512]"}
    assert twin.judged_run_of_store(wl, row) is None


def test_g37_rung_floor_a_floor_retired_by_name_never_grades_its_store_even_beside_a_judgeable_climb(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The other edge of the rule: X declares the same ladder, but its decisive ruling NAMES it (match.run_ids, ruled after the
    rung rule) while the climb Y is current and judgeable. An earlier fix read "a judgeable registration at another rung" as the ladder and
    would have graded X on its own reading; a run retired by name is an ordinary retired run: with Y selected it is covered with
    `covered_by_run_id` null and the reason named (never deferred, never graded) -- and Y itself, a CLIMB, is REFUSED by name before any claim
    (BONRUNG; the plan names both: Y REFUSED, X COVERED); X alone (another rung's judgment, were there one, covers nothing) refuses with
    nothing written, dry or real. The ladder is X's row's; the rung rule is None."""
    root, store, X, Y, cur = _two_rung_ladder(world, tmp_path, monkeypatch, floor_by_name=True)
    assert cur[X][1] == f"selection_state superseded (ruling {BY_NAME})" and cur[Y] == (1, None)
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rungfloor")
    sib = twin.store_rung_siblings(wl, _row(wl, X))
    assert sib["ladder"] == [512, 4096] and sib["other_rungs"] == {Y: (4096, "ladder_rung")} and sib["same_rung"] == [] and sib["not_judgeable"] == {}
    assert twin.declared_ladder_of(sib) == "its row's ladder [512, 4096]" and twin.rung_rule_of(wl, _row(wl, X)) is None
    # both selected, a dry run first: the plan names Y as the pair -- REFUSED, a climb -- and X as COVERED by no run with the retired reason;
    # then the refusal; nothing written. The real run refuses the same way before any claim.
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=[X, Y], dry=dry))
        assert f"run {Y} is a BoN CLIMB (cap 4096 / ladder_rung, ladder [512, 4096]) and is REFUSED before any claim" in str(ei.value) and X not in str(ei.value)
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={Y}" in ln and "— REFUSED (BONRUNG), would not be claimed or graded: " in ln and "COVERED" not in ln and "DEFERRED" not in ln and "own reading" not in ln for ln in plan), plan
    assert any(f"run_id={X}  COVERED by no run on record (the store carries no complete judgment of its rung) (not current: "
               f"selection_state superseded (ruling {BY_NAME}); a retired run: retired by selection_state superseded (ruling {BY_NAME}), not by a rule "
               "of its rung alone) — would not be claimed or graded" in ln for ln in plan), plan
    assert not any("DEFERRED" in ln or "own reading" in ln for ln in plan)
    cls = [ln for ln in out.splitlines() if "a retired run never grades its store" in ln]
    assert len(cls) == 2 and all(X in ln and "a store with no complete judgment on record: skipped" in ln and "no judgment of its rung cap 512 / base_cap is on record" in ln for ln in cls), cls
    assert all(f"retired by selection_state superseded (ruling {BY_NAME}), not by a rule of its rung alone" in ln for ln in cls), cls
    assert "own reading" not in out and "deferred" not in out and "[registry] claimed " not in out and "1 pending run(s) covered by the current run of their store (g37)" in out
    assert stub_judge["judgments"] == 0 and _tree(root) == before and not (tmp_path / "g0").exists()   # nothing graded: the climb refused, the floor retired
    assert _pending_ids(root) == {X, Y} and _n_judgments(root) == 0
    # X alone (Y pending, current, unselected): still a retired run -- refused by name with nothing written, dry and real
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2", run_ids=[X], dry=dry))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and X in msg and "covered by no run on record" in msg and BY_NAME in msg, msg
        assert f"(a retired run: retired by selection_state superseded (ruling {BY_NAME}), not by a rule of its rung alone)" in msg
        assert "nothing claimed, nothing written" in msg and "deferred" not in msg and "BoN CLIMB" not in msg
    out, _ = capsys.readouterr()
    assert "own reading" not in out and "deferred" not in out and f"covered by {Y}" not in out and "DRY RUN" not in out    # an all-covered invocation refuses before any plan
    assert out.count("a retired run never grades its store") == 2 and out.count(f"retired by selection_state superseded (ruling {BY_NAME}), not by a rule of its rung alone") == 2
    assert _tree(root) == before and not (tmp_path / "g2").exists() and stub_judge["judgments"] == 0
    assert _n_judgments(root) == 0 and _pending_ids(root) == {X, Y}


def test_g37_rung_floor_a_single_rung_cap_512_row_under_the_rung_rule_declares_no_ladder_and_stays_a_retired_run(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The boundary the fix keeps: a cap-512 BoN registration whose row declares NO ladder
    (token_cap_ladder_json [512], no pass label, no rung index -- the campaign's older single-rung rows) under the r2v1-shaped rung rule is a
    retired run exactly as the covered-twin rule left it: never graded, refused by name with nothing written (an all-covered invocation refuses
    before any plan, dry or real), the reason ("declares no ladder") on stdout and in the refusal. The rung rule applies to it (rung_rule_of
    names the rule); the ladder does not (declared_ladder_of None); the rule alone never grades a run."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    S, recs = _bon_rung_recs(f"{run_of('s1')}/{CELL}#single512", 200, store, {"token_cap_tokens": 512, "token_cap_ladder_json": [512]})
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (S,)).fetchone())
        row = conn.execute("SELECT token_cap_tokens, pass_label, token_cap_ladder_json, token_cap_rung_index FROM run WHERE run_id = ?", (S,)).fetchone()
    assert cur == (0, f"selection_state superseded (ruling {R2V1_LIKE})") and (row[0], row[1], json.loads(row[2]), row[3]) == (512, None, [512], None), (cur, row)
    assert _pending_ids(root) == {S}
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rungfloor")
    sib = twin.store_rung_siblings(wl, _row(wl, S))
    assert sib == {"rung": (512, None), "ladder": [512], "rung_index": None, "same_rung": [], "other_rungs": {}, "not_judgeable": {}}, sib
    assert twin.declared_ladder_of(sib) is None and twin.rung_rule_of(wl, _row(wl, S)) == {"ruling_id": R2V1_LIKE, "selects_by": "token_caps [512]"}
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g1", run_ids=[S], dry=dry))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and S in msg and "covered by no run on record" in msg and R2V1_LIKE in msg, msg
        assert "(a retired run: its row declares no ladder (token_cap_ladder_json [512], pass label None))" in msg and "nothing claimed, nothing written" in msg
    out, _ = capsys.readouterr()
    assert ("a store with no complete judgment on record: skipped (a retired run never grades its store; no judgment of its rung cap 512 / unrecorded "
            "is on record; its row declares no ladder (token_cap_ladder_json [512], pass label None))") in out, out
    assert "own reading" not in out and "deferred" not in out and "DRY RUN" not in out        # an all-covered invocation refuses before any plan, dry or real
    assert out.count("a retired run never grades its store; no judgment of its rung cap 512 / unrecorded is on record; its row declares no ladder") == 2
    assert _tree(root) == before and not (tmp_path / "g1").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and _pending_ids(root) == {S}


def test_g37_rung_floor_rung_selector_of_reads_the_live_rule_shapes_and_never_a_ruling_that_names_a_run(twin):
    """The pure half of rung_rule_of over the scope shapes the registry carries: the live r2v1 512-floor rule (a top-level `token_caps` beside
    its instrument_selector_json), the same selector written as OR-ed `instrument_alternatives`, a pass-label rule, a pre-match `key` scope
    naming the cap; and the shapes that are NOT a rung rule: a run-scoped one (`run_ids` -- by name, whatever else the match lists), a model-scoped
    rule naming no cap, a cap the run is not at, a non-object scope."""
    live = {"eval_id": "bon", "instrument_selector_json": {"bon_n": 75, "token_cap_tokens": 512}, "match": {"eval_ids": ["strongreject_bon"], "token_caps": [512]}}
    assert twin.rung_selector_of(live, "key", 512, "base_cap") == "token_caps [512]"
    assert twin.rung_selector_of(live, "key", "512", None) == "token_caps [512]"                                   # a cap read as text still matches
    assert twin.rung_selector_of(live, "key", 4096, "ladder_rung") is None                                          # the climb's cap: not this rule's rung
    alts = {"match": {"eval_ids": ["strongreject_bon"], "instrument_alternatives": [{"token_caps": [512], "decode_sources": ["greedy"]}, {"token_caps": [600]}]}}
    assert twin.rung_selector_of(alts, "key", 512, None) == "instrument_alternatives[].token_caps [512]"
    assert twin.rung_selector_of(alts, "key", 4096, None) is None
    assert twin.rung_selector_of({"match": {"pass_labels": ["base_cap"]}}, "key", 512, "base_cap") == "pass_labels ['base_cap']"
    assert twin.rung_selector_of({"match": {"pass_labels": ["base_cap"]}}, "key", 512, "ladder_rung") is None
    assert twin.rung_selector_of({"eval_id": "bon", "token_cap_tokens": 512}, "key", 512, None) == "token_cap_tokens 512"   # pre-match key scope
    assert twin.rung_selector_of({"eval_id": "bon", "token_cap_tokens": 512}, "eval", 512, None) is None
    by_name = {"match": {"run_ids": ["synthetic:strongreject_bon:11111111aaaa:2222222233333333"], "token_caps": [512]}, "note": "D471's shape"}
    assert twin.rung_selector_of(by_name, "key", 512, "base_cap") is None                                           # names a run: never a rung rule
    assert twin.rung_selector_of({"match": {"model_ids": ["m"]}}, "key", 512, "base_cap") is None
    assert twin.rung_selector_of({"match": {"token_caps": [512]}}, "key", None, None) is None                        # an unrecorded cap matches no rung
    assert twin.rung_selector_of("not an object", "key", 512, None) is None and twin.rung_selector_of({"match": None}, "run", 512, None) is None


# -------------------------------------------------------------------------------- g37-rung-later: the view's later rule is read, never masked


#: the other store a same-key successor of the floor is registered over (a same-cap rerun of the floor elsewhere: a store this config never locates)
OTHER_STORE = "synthetic:///store/rerun512-again/bon_responses.jsonl"
#: the identity key as v_run_current's `later` CTE pairs it (schema_def.selection_ctes_sql): the columns two runs must agree on to be predecessor and successor
D654_KEY_COLUMNS = ("model_id", "object_id", "eval_id", "instrument_id", "judge_id", "strength_nominal", "strength_coefficient", "operating_point_id_legacy",
                    "token_cap_tokens", "trial_index", "pass_label")


def _floor_with_same_key_successor(world, tmp_path, monkeypatch, *, successor: str) -> tuple:
    """g37-rung-later: the smoke's 512 floor X (cap 512 / base_cap, ladder [512, 4096], generated over the fixture store, under the r2v1-shaped
    rung rule) beside a run Z at the SAME identity key -- the same model, object, eval, instrument, judge, strength, legacy operating point,
    token_cap_tokens, trial_index and pass_label (:data:`D654_KEY_COLUMNS`) -- over ANOTHER store (:data:`OTHER_STORE`: a same-cap rerun of the
    floor elsewhere). ``successor`` = "judged_later" (Z complete and later than X: v_run_current's later rule reads X superseded by Z, which
    the ruling masks in `why` -- the decisive branch precedes `later`), "generated_later" (Z later but not complete: no successor),
    "judged_earlier" (Z complete but earlier than X: no successor). Returns ``(root, store, X, Z, {run_id: (is_current, why)})``."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X, recs_x = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    iz, through = {"judged_later": (240, "judged"), "generated_later": (240, "generated"), "judged_earlier": (100, "judged")}[successor]
    Z, recs_z = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512-again", iz, OTHER_STORE, RUNG_512, through=through)
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs_x)
    R.write_records(os.path.join(root, "records", "c", "000002-run.jsonl"), recs_z)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    cols = ", ".join(D654_KEY_COLUMNS)
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, Z)}
        key = {rid: tuple(conn.execute(f"SELECT {cols} FROM run WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, Z)}
        facts = {rid: tuple(conn.execute("SELECT lifecycle_state, store_uri, COALESCE(generated_utc, queued_utc) FROM run WHERE run_id = ?", (rid,)).fetchone())
                 for rid in (X, Z)}
    assert key[X] == key[Z] and key[X][8:] == (512, 0, "base_cap"), key                # ONE identity key: a same-cap, same-pass, same-trial rerun
    assert facts[X][:2] == ("generated", store) and facts[Z][1] == OTHER_STORE          # over two stores
    assert facts[Z][0] == ("judged" if through == "judged" else "generated") and (facts[Z][2] > facts[X][2]) == (iz > 200), facts
    # the premise: the ruling is X's FIRST failing rule whatever Z does (v_run_current's decisive branch precedes `later`), so `why` says
    # nothing about a successor; Z (cap 512 too) reads the same rule
    assert cur[X] == (0, f"selection_state superseded (ruling {R2V1_LIKE})") and cur[Z] == (0, f"selection_state superseded (ruling {R2V1_LIKE})"), cur
    return root, store, X, Z, cur


def test_g37_rung_later_a_floor_superseded_by_a_later_complete_run_at_its_key_is_a_retired_run_never_graded_on_its_own_reading(
        twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """X declares the ladder and its decisive ruling IS a rung rule, but a LATER COMPLETE run Z at X's identity key stands over
    another store (a same-cap rerun of the floor elsewhere) -- v_run_current's later rule supersedes X by Z, and the ruling masks it in `why`
    (the decisive branch precedes `later`; the view exposes no successor column). An earlier grader read the ruling as the one thing
    retiring X and graded the superseded store on its own reading: a duplicate, obsolete judgment. Now the later relation is read from the
    registry's own rendering of the `later` CTE (later_run_of) and X is a retired run: refused by name with Z named, nothing claimed,
    nothing written, dry and real. main() runs first, so an unfixed grader fails at the defect (it claims and grades X), not at a helper's
    name. Z's judgment is on ANOTHER store, so it covers nothing; the relation alone retires X."""
    root, store, X, Z, cur = _floor_with_same_key_successor(world, tmp_path, monkeypatch, successor="judged_later")
    assert _pending_ids(root) == {X} and _n_judgments(root) == 1                      # Z's own judgment; X pending, superseded by Z under the mask
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g1", run_ids=[X], dry=dry))    # the unfixed grader: claims X here
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and X in msg and "covered by no run on record" in msg and R2V1_LIKE in msg, msg
        assert (f"(a retired run: superseded by a later complete run at its key, {Z} (v_run_current's later rule, D654), besides the rung rule "
                f"{R2V1_LIKE} -- the ruling is the first failing rule in `why` and masks the successor)") in msg, msg
        assert "nothing claimed, nothing written" in msg and "deferred" not in msg and "own reading" not in msg
    out, _ = capsys.readouterr()
    assert out.count("a retired run never grades its store; no judgment of its rung cap 512 / base_cap is on record; superseded by a later complete run "
                     f"at its key, {Z} (v_run_current's later rule, D654)") == 2, out
    assert "own reading" not in out and "deferred" not in out and "DRY RUN" not in out      # an all-covered invocation refuses before any plan
    assert _tree(root) == before and not (tmp_path / "g1").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 1 and _pending_ids(root) == {X}
    # the helpers the decision rests on: the ladder and the rung rule both hold for X -- the later relation alone makes it a retired run
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37runglater")
    row = _row(wl, X)
    sib = twin.store_rung_siblings(wl, row)
    assert sib == {"rung": (512, "base_cap"), "ladder": [512, 4096], "rung_index": 0, "same_rung": [], "other_rungs": {}, "not_judgeable": {}}, sib  # Z: another store
    assert twin.declared_ladder_of(sib) == "its row's ladder [512, 4096]"
    assert twin.rung_rule_of(wl, row) == {"ruling_id": R2V1_LIKE, "selects_by": "token_caps [512]"}
    assert twin.judged_run_of_store(wl, row) is None                                    # Z's complete judgment is over ANOTHER store: covers nothing
    assert twin.later_run_of(wl, row) == [Z]                                            # the view's later rule, read from its own rendering
    assert twin.later_run_of(wl, {"run_id": Z}) == []                                   # Z has no successor


@pytest.mark.parametrize("successor", ["generated_later", "judged_earlier"])
def test_g37_rung_later_the_relation_is_the_views_own_a_same_key_run_that_is_not_complete_or_not_later_retires_nothing(
        successor, twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The two sides of the view's definition, so the grader follows the registry and not a wider notion of its own: a same-key run Z that is
    LATER but not complete (generated), or COMPLETE but earlier than X, is no successor under v_run_current's later rule -- later_run_of names
    none, and X is graded on its own reading exactly as before (dry run named, real run graded, Z untouched, the provenance naming the
    relation consulted and empty)."""
    root, store, X, Z, cur = _floor_with_same_key_successor(world, tmp_path, monkeypatch, successor=successor)
    assert _pending_ids(root) == ({X, Z} if successor == "generated_later" else {X})
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37runglater")
    assert twin.later_run_of(wl, _row(wl, X)) == [] and twin.rung_rule_of(wl, _row(wl, X)) == {"ruling_id": R2V1_LIKE, "selects_by": "token_caps [512]"}
    before = _tree(root)
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=[X], dry=True))
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={X}" in ln and "a different rung of its store's ladder (cap 512 / base_cap), graded on its own reading (g37-rung)" in ln for ln in plan), plan
    assert not any("COVERED" in ln or "DEFERRED" in ln for ln in plan) and _tree(root) == before and not (tmp_path / "g0").exists()
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g1", run_ids=[X]))
    out, _ = capsys.readouterr()
    gl = next(ln for ln in out.splitlines() if "graded on its own reading in this invocation" in ln)
    assert X in gl and Z not in gl and (f"retired from the headline by the rung rule {R2V1_LIKE} alone (it selects by token_caps [512], never by run id: "
                                         "the floor is kept for the curve) and by no later complete run at its key (v_run_current's later rule names none)") in gl, gl
    assert "a retired run never grades its store" not in out and "claimed 1 pending run(s)" in out and stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX
    prov = _provenance(tmp_path / "g1", "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [X] and prov["registry"]["covered_runs"] == []
    (g,) = prov["registry"]["rung_ladder"]["graded_on_own_reading"]
    assert g["run_id"] == X and g["rung_rule"] == R2V1_LIKE and g["later_run_ids"] == [] and g["other_rungs"] == {} and g["not_judgeable"] == {}
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, X)[0] == "judged" and len(_judgments(conn, X)) == 1
        assert _run_state(conn, Z)[0] == ("generated" if successor == "generated_later" else "judged")      # Z untouched
    assert _pending_ids(root) == ({Z} if successor == "generated_later" else set())


def test_g37_rung_later_later_run_of_refuses_by_name_when_the_registrys_rendering_of_the_later_cte_is_not_the_shape_it_widens(
        twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The relation is read by widening the ONE SELECT of the registry's own `later` CTE (schema_def.selection_ctes_sql, the text
    v_run_current is created from) to carry the successor -- never by re-encoding the identity key. A rendering that no longer carries that
    SELECT exactly once is a reason to look, not to guess: later_run_of refuses by name (RegistryUnavailable, the count and the rule named),
    and through main() the invocation refuses before any claim, dry and real, nothing written."""
    from sisteer.registry import schema_def as SD
    root, store, X, Z, cur = _floor_with_same_key_successor(world, tmp_path, monkeypatch, successor="judged_later")
    real = SD.selection_ctes_sql()
    assert real.count(twin.LATER_CTE_SELECT) == 1
    monkeypatch.setattr(SD, "selection_ctes_sql", lambda: real.replace(twin.LATER_CTE_SELECT, "SELECT r.run_id FROM run r JOIN run r2 ON", 1))
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37runglater")
    with pytest.raises(W.RegistryUnavailable) as ei:
        twin.later_run_of(wl, _row(wl, X))
    assert "the registry's rendering of v_run_current's `later` CTE (schema_def.selection_ctes_sql) carries the SELECT this grader widens 0 time(s), not once" in str(ei.value)
    assert "the later-run relation cannot be read; refusing to decide a rung's retirement (g37-rung-later)" in str(ei.value)
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g1", run_ids=[X], dry=dry))
        assert "carries the SELECT this grader widens 0 time(s), not once" in str(ei.value) and "g37-rung-later" in str(ei.value)
    out, _ = capsys.readouterr()
    assert "own reading" not in out and "DRY RUN" not in out and "claimed" not in out
    assert _tree(root) == before and not (tmp_path / "g1").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 1 and _pending_ids(root) == {X}


# ------------------------------------------------------------------------- g37-rung-guard: the one-run guard is applied AFTER the rungs are sorted


def _two_rung_ladder_with_climb_successor(world, tmp_path, monkeypatch, *, climb_twins: int = 1) -> tuple:
    """g37-rung-guard: the smoke's pair -- the 512 floor X (cap 512 / base_cap, generated, non-current under the r2v1-shaped
    rung rule) and its climb Y (cap 4096 / ladder_rung, generated) over ONE store -- beside a run Z at Y's identity key (:data:`D654_KEY_COLUMNS`) over ANOTHER store
    (:data:`OTHER_STORE`: a rerun of the climb elsewhere), complete and later, and a `supersede` ruling NAMING the climb registration(s)
    (:data:`BY_NAME_CLIMB`: a mis-served climb re-run elsewhere, ruled superseded by name before the producer's run_state record
    lands), so the store has NO current run while X and Y both stand pending -- X retired by the rung rule alone, Y independently. The view's
    LATER rule alone cannot retire Y here, and the fixture records why (asserted before the ruling is written): v_roster_match admits a run only
    through a COMPLETE covering judgment and SELECTION_STATE_SQL reads `not_in_roster` before the `later` branch, so a pending (unjudged) run
    reads current whatever stands at its key -- the raw relation (later_run_of) still names Z. ``climb_twins`` = 2 registers a second climb
    Y2 at Y's rung over the same store (an ambiguous rung: two registrations, both named by the ruling). Returns ``(root, store, X, [Y, ...],
    Z, {run_id: (is_current, why)})``."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X, recs_x = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    climbs = [_bon_rung_recs(f"{run_of('s1')}/{CELL}#rerunclimb{'' if k == 0 else f'-twin{k}'}", 210 + 5 * k, store, RUNG_4096) for k in range(climb_twins)]
    Z, recs_z = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerunclimb-again", 240, OTHER_STORE, RUNG_4096, through="judged")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs_x)
    for k, (_, recs_y) in enumerate(climbs):
        R.write_records(os.path.join(root, "records", "c", f"{k + 2:06d}-run.jsonl"), recs_y)
    R.write_records(os.path.join(root, "records", "c", "000009-run.jsonl"), recs_z)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    Ys = [rid for rid, _ in climbs]
    cols = ", ".join(D654_KEY_COLUMNS)
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why, selection_state FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, *Ys, Z)}
        key = {rid: tuple(conn.execute(f"SELECT {cols} FROM run WHERE run_id = ?", (rid,)).fetchone()) for rid in (*Ys, Z)}
        facts = {rid: tuple(conn.execute("SELECT lifecycle_state, store_uri, COALESCE(generated_utc, queued_utc) FROM run WHERE run_id = ?", (rid,)).fetchone())
                 for rid in (X, *Ys, Z)}
    assert all(key[y] == key[Z] and key[y][8:] == (4096, 0, "ladder_rung") for y in Ys), key            # the climb's identity key: Z is its same-key rerun
    assert facts[X][:2] == ("generated", store) and all(facts[y][:2] == ("generated", store) for y in Ys) and facts[Z][:2] == ("judged", OTHER_STORE), facts
    assert all(facts[Z][2] > facts[y][2] for y in Ys), facts                                              # Z later than every climb registration
    # the view's premise: a PENDING run is never retired by the later rule -- no covering judgment, no roster match, `not_in_roster` first
    assert cur[X] == (0, f"selection_state superseded (ruling {R2V1_LIKE})", "superseded") and cur[Z] == (1, None, "not_in_roster"), cur
    assert all(cur[y] == (1, None, "not_in_roster") for y in Ys), cur
    # the independent retirement of the climb: a `supersede` NAMING the climb registration(s), ruled after the rung rule
    R.write_records(os.path.join(root, "records", "rulings_synth", "000091-supersede-climb.jsonl"),
                    [_supersede_ruling(BY_NAME_CLIMB, {"match": {"run_ids": list(Ys)}, "note": "synthetic: the climb retired BY NAME (D471's shape: a mis-served climb re-run elsewhere)"},
                                       "synthetic mirror of a run-scoped supersede over the climb: these runs are retired, not their rung", 400)])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X, *Ys, Z)}
    # NO current run among the store's pending registrations: the floor under the rung rule, the climb(s) by name, Z current elsewhere
    assert cur[X] == (0, f"selection_state superseded (ruling {R2V1_LIKE})") and cur[Z] == (1, None), cur
    assert all(cur[y] == (0, f"selection_state superseded (ruling {BY_NAME_CLIMB})") for y in Ys), cur
    return root, store, X, Ys, Z, cur


def test_g37_rung_guard_a_climb_superseded_by_a_later_complete_rerun_never_strands_the_floor_beside_it_which_is_graded_on_its_own_reading(
        twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The store-wide "exactly one current run" guard ran BEFORE the per-rung reading, so a store whose floor X is
    non-current under the rung rule alone while its climb Y is non-current INDEPENDENTLY -- here by a `supersede` naming it (a
    mis-served climb re-run elsewhere as Z, complete and later at Y's key, ruled superseded by name before the producer's run_state record
    lands; the view's later rule alone never retires a PENDING run, see the fixture) -- saw zero current runs among two
    pending and refused EVERY invocation, the one selecting the floor alone included (pending_beside_selection brings Y into the check), as
    "awaiting a supersession ruling", though there is none to await: X's own row declares the ladder, the rung rule is the one thing retiring
    it, it has no same-key successor, and Y is retired on its own account. Now the rungs are sorted first: every rung registered once proceeds
    to the per-run reading, where X is graded on its own reading (dry and real, selected alone or beside Y) and Y is a retired run, covered by
    no run with the reason named -- never claimed, never graded; afterwards Y alone refuses as a retired run with nothing written. main() runs
    first, so an unfixed grader fails at the DEFECT (SystemExit: two pending runs name the store, 0 of them current), not at a helper's name."""
    root, store, X, (Y,), Z, cur = _two_rung_ladder_with_climb_successor(world, tmp_path, monkeypatch)
    assert _pending_ids(root) == {X, Y} and _n_judgments(root) == 1                   # Z's own judgment; both rungs of the store pending, neither current
    before = _tree(root)
    for rids in ([X], [X, Y]):                                                         # the floor alone (the daemon's dispatch) and the pair
        twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=rids, dry=True))    # the unfixed grader: SystemExit here
        out, _ = capsys.readouterr()
        plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
        assert any(f"run_id={X}" in ln and "a different rung of its store's ladder (cap 512 / base_cap), graded on its own reading (g37-rung)" in ln for ln in plan), plan
        assert any(f"run_id={Y}  COVERED by no run on record (the store carries no complete judgment of its rung) (not current: {cur[Y][1]}; "
                   f"a retired run: retired by {cur[Y][1]}, not by a rule of its rung alone) — would not be claimed or graded" in ln for ln in plan), plan
        assert not any("DEFERRED" in ln for ln in plan) and "refusing" not in out and _tree(root) == before and not (tmp_path / "g0").exists()
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g1", run_ids=[X]))
    out, _ = capsys.readouterr()
    gl = next(ln for ln in out.splitlines() if "graded on its own reading in this invocation" in ln)
    assert X in gl and f"is not current ({cur[X][1]})" in gl and f"is the only judgeable registration of its store at cap 512 / base_cap beside {Y} at cap 4096 / ladder_rung" in gl, gl
    assert (f"a rung declared by its row's ladder [512, 4096], retired from the headline by the rung rule {R2V1_LIKE} alone (it selects by token_caps [512], "
            "never by run id: the floor is kept for the curve) and by no later complete run at its key (v_run_current's later rule names none)") in gl, gl
    cl = next(ln for ln in out.splitlines() if "a retired run never grades its store" in ln)
    assert Y in cl and f"is not current ({cur[Y][1]})" in cl and "no judgment of its rung cap 4096 / ladder_rung is on record" in cl, cl
    assert f"retired by {cur[Y][1]}, not by a rule of its rung alone" in cl and "not claimed, not graded, no judgment of its own" in cl, cl
    assert "deferred" not in out and "refusing" not in out and "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "claimed 1 pending run(s)" in out
    assert "1 pending run(s) graded on their own reading as a different rung of their store's ladder (g37-rung)" in out
    assert stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX                       # the floor's reading alone
    prov = _provenance(tmp_path / "g1", "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [X] and prov["registry"]["rung_ladder"]["deferred"] == []
    (g,) = prov["registry"]["rung_ladder"]["graded_on_own_reading"]
    assert g["run_id"] == X and g["why"] == cur[X][1] and g["rung"] == [512, "base_cap"] and g["declared_by"] == "its row's ladder [512, 4096]"
    assert g["rung_rule"] == R2V1_LIKE and g["rule_selects_by"] == "token_caps [512]" and g["later_run_ids"] == [] and "deferred_why" not in g
    assert g["other_rungs"] == {Y: [4096, "ladder_rung"]} and g["not_judgeable"] == {}
    (c,) = prov["registry"]["covered_runs"]
    assert c["run_id"] == Y and c["covered_by_run_id"] is None and c["why"] == cur[Y][1] and c["rung"] == [4096, "ladder_rung"] and c["attack"] == "bon" and c["cell"] == CELL
    assert c["retired_why"] == f"retired by {cur[Y][1]}, not by a rule of its rung alone"
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, X)[0] == "judged" and len(_judgments(conn, X)) == 1 and _judgments(conn, Y) == [] and _run_state(conn, Y)[0] == "generated"
        assert _run_state(conn, Z)[0] == "judged" and len(_judgments(conn, Z)) == 1 and conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 2
        assert tuple(conn.execute("SELECT is_current FROM v_run_current WHERE run_id = ?", (X,)).fetchone()) == (0,)   # the rule stands; the reading is on record
    assert _pending_ids(root) == {Y}
    # Y alone afterwards: a retired run (by name; X's judgment is at ANOTHER rung and covers nothing) -- refused by name with nothing written,
    # dry and real
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2", run_ids=[Y], dry=dry))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and Y in msg and "covered by no run on record" in msg and X not in msg.split("covered, never graded")[1], msg
        assert f"(a retired run: retired by {cur[Y][1]}, not by a rule of its rung alone)" in msg and "nothing claimed, nothing written" in msg and "deferred" not in msg
    out, _ = capsys.readouterr()
    assert "own reading" not in out and "DRY RUN" not in out and out.count("a retired run never grades its store") == 2
    assert _tree(root) == before and not (tmp_path / "g2").exists() and stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX
    assert _n_judgments(root) == 2 and _pending_ids(root) == {Y}
    # the helpers the decision rests on: Y is retired by name (no rung rule), its rerun Z named by the view's own relation; X's reading had none
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rungguard")
    row_y = _row(wl, Y)
    assert twin.store_rung_siblings(wl, row_y)["same_rung"] == [] and twin.store_rung_siblings(wl, row_y)["other_rungs"] == {X: (512, "base_cap")}
    assert twin.declared_ladder_of(twin.store_rung_siblings(wl, row_y)) == "its row's ladder [512, 4096]" and twin.rung_rule_of(wl, row_y) is None
    assert twin.judged_run_of_store(wl, row_y) is None and twin.later_run_of(wl, row_y) == [Z] and twin.later_run_of(wl, {"run_id": X}) == []


def test_g37_rung_guard_two_same_rung_registrations_with_no_current_run_still_refuse_by_name_the_rung_named(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The refusal the guard keeps: two cap-512 registrations of ONE store (X1, X2 at cap 512 / base_cap), both non-current under the rung rule,
    no climb -- one store, one run per rung, and no current run singles one out: refused by name as before (the g37 text), the shared rung
    and its ids now named in the refusal, nothing claimed, nothing written, dry and real, the floor alone selected or both (the unselected
    twin marked). Neither is graded on its own reading: an ambiguous rung awaits the closer's settlement or a ruling."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X1, recs1 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    X2, recs2 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512-twin", 230, store, RUNG_512)
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs1)
    R.write_records(os.path.join(root, "records", "c", "000002-run.jsonl"), recs2)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X1, X2)}
    assert cur == {rid: (0, f"selection_state superseded (ruling {R2V1_LIKE})") for rid in (X1, X2)}, cur
    assert _pending_ids(root) == {X1, X2}
    A, B = sorted((X1, X2))
    before = _tree(root)
    for dry in (True, False):
        for rids in ([X1], [X1, X2]):
            with pytest.raises(SystemExit) as ei:
                twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g", run_ids=rids, dry=dry))
            msg = str(ei.value)
            assert msg.startswith(f"{twin.DRIVER_ID}: two pending runs name the store of bon/{CELL} under ") and " — one store, one run; refusing (" in msg, msg
            assert X1 in msg.split(" — one store, one run")[0] and X2 in msg.split(" — one store, one run")[0], msg
            assert f"0 of them current in {twin.RUN_CURRENCY_VIEW}: {A} is_current 0 ({cur[A][1]})" in msg and f"{B} is_current 0 ({cur[B][1]})" in msg, msg
            assert (f"; {A} and {B} share the rung cap 512 / base_cap: one store, one run per rung, and no current run singles one out "
                    f"(g37-rung-guard); {twin.STORE_RUNS_REFUSAL_PHRASE})") in msg and msg.endswith(f"; {twin.STORE_RUNS_REFUSAL_PHRASE})"), msg
            assert msg.count(NOT_SELECTED_MARK) == (1 if rids == [X1] else 0) and (X2 not in msg or rids == [X1, X2] or f"{X2} is_current 0 ({cur[X2][1]}) {NOT_SELECTED_MARK}" in msg), msg
    out, _ = capsys.readouterr()
    assert "own reading" not in out and "DRY RUN" not in out and "claimed" not in out.replace("not claimed", "") and "a retired run" not in out
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and _pending_ids(root) == {X1, X2}


def test_g37_rung_guard_an_ambiguous_climb_rung_beside_the_floor_refuses_the_store_by_name_until_the_twins_are_settled(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The boundary the guard draws (named for the registry owner): the floor X beside TWO climb registrations Y1, Y2 at ONE rung over the same
    store, both retired by name (their rerun Z complete elsewhere) -- no current run, and the climb rung is registered twice. The
    store refuses by name (the rung and its ids named: one store, one run per rung), dry and real, the floor alone selected or all three:
    nothing claimed, nothing written, the floor not graded past an unsettled rung of its own store. The floor's grade waits on the twins'
    settlement (the closer's duplicate_registration record or a ruling) and the refusal says which rung to settle."""
    root, store, X, (Y1, Y2), Z, cur = _two_rung_ladder_with_climb_successor(world, tmp_path, monkeypatch, climb_twins=2)
    assert _pending_ids(root) == {X, Y1, Y2} and _n_judgments(root) == 1
    ids = sorted((X, Y1, Y2))
    before = _tree(root)
    for dry in (True, False):
        for rids in ([X], [X, Y1, Y2]):
            with pytest.raises(SystemExit) as ei:
                twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g", run_ids=rids, dry=dry))
            msg = str(ei.value)
            assert f"3 pending runs name the store of bon/{CELL} under " in msg and f" — one store, one run; refusing (0 of them current in {twin.RUN_CURRENCY_VIEW}: " in msg, msg
            assert all(rid in msg.split(" — one store, one run")[0] for rid in ids), msg
            assert (f"; {' and '.join(sorted((Y1, Y2)))} share the rung cap 4096 / ladder_rung: one store, one run per rung, and no current run singles one out "
                    f"(g37-rung-guard); {twin.STORE_RUNS_REFUSAL_PHRASE})") in msg and msg.endswith(f"; {twin.STORE_RUNS_REFUSAL_PHRASE})"), msg
            assert "cap 512 / base_cap" not in msg and msg.count(NOT_SELECTED_MARK) == (2 if rids == [X] else 0), msg
    out, _ = capsys.readouterr()
    assert "own reading" not in out and "DRY RUN" not in out and "claimed" not in out.replace("not claimed", "") and "a retired run" not in out
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 1 and _pending_ids(root) == {X, Y1, Y2}


# -------------------------------------------------------------------------- g37-rung-guard-r5: the rung ambiguity is read whatever the other rungs do


def _two_floors_beside_a_current_climb(world, tmp_path, monkeypatch) -> tuple:
    """g37-rung-guard-r5: ONE BoN store registered three times -- TWO cap-512 floors X1, X2 (cap 512 / base_cap, generated, both non-current under
    the r2v1-shaped rung rule, neither with a cap-512 judgment) beside the climb Y (cap 4096 / ladder_rung, generated, CURRENT). The closer
    settles a store512 / climb pair, never two floor registrations (bon_settle_duplicates: the store512 role retires climb twins, the climb role
    defers to a store512 filing), so nothing is coming to single one floor out. Returns ``(root, store, X1, X2, Y, {run_id: (is_current, why)})``."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X1, recs1 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    X2, recs2 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512-twin", 205, store, RUNG_512)
    Y, recs_y = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerunclimb", 210, store, RUNG_4096)
    for k, recs in enumerate((recs1, recs2, recs_y), start=1):
        R.write_records(os.path.join(root, "records", "c", f"{k:06d}-run.jsonl"), recs)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X1, X2, Y)}
        rungs = {rid: tuple(conn.execute("SELECT token_cap_tokens, pass_label FROM run WHERE run_id = ?", (rid,)).fetchone()) for rid in (X1, X2, Y)}
    assert rungs == {X1: (512, "base_cap"), X2: (512, "base_cap"), Y: (4096, "ladder_rung")}, rungs
    assert cur[X1] == cur[X2] == (0, f"selection_state superseded (ruling {R2V1_LIKE})") and cur[Y] == (1, None), cur
    assert _pending_ids(root) == {X1, X2, Y} and _n_judgments(root) == 0
    return root, store, X1, X2, Y, cur


def test_g37_rung_guard_r5_two_floor_registrations_beside_a_current_climb_refuse_the_store_by_name_before_the_climb_is_graded(
        twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The rung-ambiguity refusal was computed only when NO run of the store was current, so two cap-512 floors X1, X2 (both
    non-current under the rung rule, neither with a cap-512 judgment) beside the CURRENT climb Y passed the guard -- Y was accepted and graded,
    and both floors were "covered" with covered_by null merely for having a same-rung sibling; after Y's judgment folded the zero-current guard
    refused them "awaiting a supersession ruling": the ambiguity surfaced only after the grading it should have held, and the owed floor reading
    sat behind a refusal nobody had been told to expect. Now the store refuses by name BEFORE anything is claimed -- dry and real, the climb
    alone selected (the daemon's dispatch of the current run), a floor alone, or all three -- with "1 of them current" and the rung-ambiguity
    clause naming the floor rung and its ids byte for byte; nothing claimed, nothing written, the judge never called. The way out is the registry
    owner's, never the grader's: a run_state `superseded` record retiring one floor (the shape the closer and the owner file) leaves one
    judgeable registration of the rung, and the store is the smoke's pair again -- Y graded, X1 deferred by name, then graded on its own
    reading. main() runs first, so an unfixed grader fails at the DEFECT (the dry run selecting Y alone DID NOT RAISE: it planned the climb)."""
    root, store, X1, X2, Y, cur = _two_floors_beside_a_current_climb(world, tmp_path, monkeypatch)
    A, B = sorted((X1, X2))
    before = _tree(root)
    for dry in (True, False):
        for rids in ([Y], [X1], [X1, X2, Y]):
            with pytest.raises(SystemExit) as ei:                                   # the unfixed grader: no SystemExit for [Y] -- the climb is planned / graded
                twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g", run_ids=rids, dry=dry))
            msg = str(ei.value)
            assert msg.startswith(f"{twin.DRIVER_ID}: 3 pending runs name the store of bon/{CELL} under ") and " — one store, one run; refusing (" in msg, msg
            assert all(rid in msg.split(" — one store, one run")[0] for rid in (X1, X2, Y)), msg
            assert f"refusing (1 of them current in {twin.RUN_CURRENCY_VIEW}: " in msg and f"{Y} is_current 1" in msg, msg
            assert f"{A} is_current 0 ({cur[A][1]})" in msg and f"{B} is_current 0 ({cur[B][1]})" in msg, msg
            assert (f"; {A} and {B} share the rung cap 512 / base_cap: one store, one run per rung, and no current run singles one out "
                    f"(g37-rung-guard); {twin.STORE_RUNS_REFUSAL_PHRASE})") in msg and msg.endswith(f"; {twin.STORE_RUNS_REFUSAL_PHRASE})"), msg
            assert "cap 4096 / ladder_rung" not in msg and msg.count(NOT_SELECTED_MARK) == (0 if len(rids) == 3 else 2), msg
            if rids != [X1, X2, Y]:
                assert all(f"{rid} is_current {cur[rid][0]}" + (f" ({cur[rid][1]})" if cur[rid][1] else "") + f" {NOT_SELECTED_MARK}" in msg
                           for rid in (X1, X2, Y) if rid not in rids), msg
    out, _ = capsys.readouterr()
    assert "own reading" not in out and "DRY RUN" not in out and "claimed" not in out.replace("not claimed", "") and "a retired run" not in out
    assert "covered by that run's judgment" not in out and "no complete judgment on record" not in out and "COVERED" not in out and "deferred" not in out
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and _pending_ids(root) == {X1, X2, Y}
    # the settlement is the registry owner's: a run_state `superseded` record retiring X2 (extra.superseded_by_run_id naming the survivor;
    # never a grader-side write) takes X2 out of the judgeable states -- the rung is then registered once among the store's
    # judgeable registrations, and the store is the smoke's pair
    R.write_records(os.path.join(root, "records", "c", "000004-settle.jsonl"),
                    [synth.rec("run_state", {"run_id": X2, "state": "superseded", "event_utc": synth.t(260), "evidence_kind": "completion_artifact",
                                             "evidence_uri": store, "note": "synthetic: the second floor registration of one store, retired by the owner's record",
                                             "extra": {"superseded_by_run_id": X1, "reason": "duplicate_registration", "decision": "synthetic"}}, synth.t(260))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        assert _run_state(conn, X2)[0] == "superseded" and _run_state(conn, X1)[0] == "generated"
    assert _pending_ids(root) == {X1, Y}
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rungguardr5")
    assert twin.store_rung_siblings(wl, _row(wl, X1)) == {"rung": (512, "base_cap"), "ladder": [512, 4096], "rung_index": 0, "same_rung": [],
                                                          "other_rungs": {Y: (4096, "ladder_rung")}, "not_judgeable": {X2: (512, "base_cap", "superseded")}}
    assert twin.judged_run_of_store(wl, _row(wl, X1)) is None
    # the pair: a dry run names Y as the pair -- REFUSED by name, a climb (BONRUNG) -- and X1 DEFERRED (not covered), then refuses; the real run
    # refuses the same way before any claim
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=[X1, Y], dry=dry))
        assert f"run {Y} is a BoN CLIMB (cap 4096 / ladder_rung, ladder [512, 4096]) and is REFUSED before any claim" in str(ei.value) and X1 not in str(ei.value)
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={Y}" in ln and "— REFUSED (BONRUNG), would not be claimed or graded: " in ln and "DEFERRED" not in ln and "COVERED" not in ln for ln in plan), plan
    assert any(f"run_id={X1}  DEFERRED at cap 512 / base_cap" in ln and f"{Y} takes bon/{CELL}" in ln for ln in plan) and not any("COVERED" in ln for ln in plan), plan
    dl = [ln for ln in out.splitlines() if "deferred:" in ln]
    assert len(dl) == 2 and all(X1 in ln and f"({X2} at cap 512 / base_cap (lifecycle superseded): no longer judgeable)" in ln and f"{Y} takes bon/{CELL} in this invocation" in ln for ln in dl), dl
    assert "[registry] claimed " not in out and "covered by that run's judgment" not in out and "no complete judgment on record" not in out
    assert stub_judge["judgments"] == 0 and _tree(root) == before and not (tmp_path / "g0").exists() and _n_judgments(root) == 0
    assert _pending_ids(root) == {X1, Y}
    # then the floor alone (the climb pending and current beside it, unselected): graded on its own reading, a judgment of its own -- the paper's
    # 512 set; none for the retired floor, none for the climb until its fold exists
    twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2", run_ids=[X1]))
    out, _ = capsys.readouterr()
    gl = next(ln for ln in out.splitlines() if "graded on its own reading in this invocation" in ln)
    assert X1 in gl and f"retired from the headline by the rung rule {R2V1_LIKE} alone" in gl and "claimed 1 pending run(s)" in out and "refusing" not in out, gl
    assert stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX
    prov = _provenance(tmp_path / "g2", "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [X1] and [g["run_id"] for g in prov["registry"]["rung_ladder"]["graded_on_own_reading"]] == [X1]
    assert prov["registry"]["rung_ladder"]["graded_on_own_reading"][0]["reading"] == "the store's cap-512 draws (bon_responses.jsonl)"
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        assert _run_state(conn, X1)[0] == "judged" and len(_judgments(conn, X1)) == 1 and _judgments(conn, Y) == [] and _judgments(conn, X2) == []
        assert _run_state(conn, X2)[0] == "superseded" and _run_state(conn, Y)[0] == "generated"
        _assert_512_set(conn, X1)
    assert _pending_ids(root) == {Y} and _n_judgments(root) == 1
    # the climb alone (the daemon's dispatch of the store's current run): refused by name before any claim, dry and real, its pending row standing
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g3", run_ids=[Y], dry=dry))
        assert f"run {Y} is a BoN CLIMB" in str(ei.value) and "nothing claimed, nothing written" in str(ei.value)
    out, _ = capsys.readouterr()
    assert "[registry] claimed " not in out and "deferred" not in out and _tree(root) == before and not (tmp_path / "g3").exists()
    assert _pending_ids(root) == {Y} and _n_judgments(root) == 1 and stub_judge["judgments"] == TM.N_GOALS_BON * TM.N_MAX


def test_g37_rung_guard_r5_a_duplicated_rung_whose_judgment_is_on_record_is_covered_by_it_never_ambiguous_whatever_the_other_rungs_currency(
        twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """The ambiguity is about a rung NOBODY has read. With a complete judgment of the rung on the store -- X0 (cap 512 / base_cap), judged, then
    non-current under the rung rule like every cap-512 row -- two more cap-512 registrations X1, X2 (generated, non-current) are g37-fix's
    retired twins: each covered by X0's judgment, never graded, never a refusal of the store. The same whatever the other rung does: beside the
    current climb Y (Y graded, X1 and X2 COVERED by X0, dry and real) and after Y's judgment folds (zero current among the store's pending
    registrations: the no-pairs refusal names both covered by X0, nothing written -- the earlier guard refused that store by name for the
    duplicated rung alone, a refusal with nothing to await: the rung's reading is on record)."""
    store = str(store_of(world, "s1", "bon") / "bon_responses.jsonl")
    root, _ = _registry(tmp_path, monkeypatch, [])
    X0, recs0 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512-first", 180, store, RUNG_512, through="judged")
    X1, recs1 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512", 200, store, RUNG_512)
    X2, recs2 = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerun512-twin", 205, store, RUNG_512)
    Y, recs_y = _bon_rung_recs(f"{run_of('s1')}/{CELL}#rerunclimb", 210, store, RUNG_4096)
    for k, recs in enumerate((recs0, recs1, recs2, recs_y), start=1):
        R.write_records(os.path.join(root, "records", "c", f"{k:06d}-run.jsonl"), recs)
    R.write_records(os.path.join(root, "records", "rulings_synth", "000090-supersede.jsonl"), [_r2v1_like_ruling()])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with _conn(root) as conn:
        cur = {rid: tuple(conn.execute("SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()) for rid in (X0, X1, X2, Y)}
        assert _run_state(conn, X0)[0] == "judged" and len(_judgments(conn, X0)) == 1
    assert cur[X1] == cur[X2] == (0, f"selection_state superseded (ruling {R2V1_LIKE})") and cur[Y] == (1, None), cur
    assert cur[X0][0] == 0 and R2V1_LIKE in str(cur[X0][1]), cur                     # the rule stands over the judged floor too; its reading is on record
    assert _pending_ids(root) == {X1, X2, Y} and _n_judgments(root) == 1
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/g37rungguardr5")
    assert twin.judged_run_of_store(wl, _row(wl, X1)) == X0 and twin.judged_run_of_store(wl, _row(wl, X2)) == X0
    assert twin.store_rung_siblings(wl, _row(wl, X1))["same_rung"] == sorted((X0, X2))
    # (1) beside the current climb: X1 and X2 covered by X0 -- never a refusal of the STORE (the round-4 guard's "share the rung" never fires) --
    # and the climb Y itself, the pair's run, REFUSED by name before any claim (BONRUNG), dry and real: the plan names Y REFUSED and both floors
    # COVERED by X0; nothing claimed, nothing written
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g0", run_ids=[X1, X2, Y], dry=dry))
        msg = str(ei.value)
        assert f"run {Y} is a BoN CLIMB (cap 4096 / ladder_rung, ladder [512, 4096]) and is REFUSED before any claim" in msg and "share the rung" not in msg, msg
        assert X1 not in msg and X2 not in msg
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={Y}" in ln and "— REFUSED (BONRUNG), would not be claimed or graded: " in ln and "COVERED" not in ln and "DEFERRED" not in ln for ln in plan), plan
    assert all(any(f"run_id={x}  COVERED by {X0} (not current: {cur[x][1]}) — would not be claimed or graded" in ln for ln in plan) for x in (X1, X2)), plan
    assert "2 pending run(s) covered by" in out and "own reading" not in out and "a retired run" not in out and "[registry] claimed " not in out
    cls = [ln for ln in out.splitlines() if "covered by that run's judgment" in ln]
    assert len(cls) == 4 and all(f"graded under {X0}" in ln and "not claimed, not graded" in ln for ln in cls) and {x for x in (X1, X2) for ln in cls if x in ln} == {X1, X2}, cls
    assert stub_judge["judgments"] == 0 and _tree(root) == before and not (tmp_path / "g0").exists()
    with _conn(root) as conn:
        assert _run_state(conn, Y)[0] == "generated" and _judgments(conn, Y) == [] and _judgments(conn, X1) == [] and _judgments(conn, X2) == []
    assert _pending_ids(root) == {X1, X2, Y} and _n_judgments(root) == 1
    # (2) the floors alone, the climb pending and current beside them (never graded here): still covered by X0, never a refusal of the store --
    # the no-pairs refusal names both, nothing written, dry and real, one floor selected or both
    before = _tree(root)
    for dry in (True, False):
        for rids in ([X1], [X1, X2]):
            with pytest.raises(SystemExit) as ei:                                   # the earlier guard: "share the rung ... (g37-rung-guard)" here
                twin.main(reg_argv(world, "s1", root, attacks="bon", grades=tmp_path / "g2", run_ids=rids, dry=dry))
            msg = str(ei.value)
            assert "no pending (attack, cell) pair" in msg and "2 pending run(s) are not current and were covered, never graded (g37-fix)" in msg, msg
            assert all(f"{x} (bon/{CELL}) covered by {X0} -- {cur[x][1]}" in msg for x in (X1, X2)), msg
            assert "share the rung" not in msg and "one store, one run;" not in msg and "a retired run" not in msg and "nothing claimed, nothing written" in msg, msg
            assert "BoN CLIMB" not in msg                                            # the unselected climb is never the pair's run here
    out, _ = capsys.readouterr()
    assert out.count("covered by that run's judgment") == 8 and "own reading" not in out and "DRY RUN" not in out and "refusing" not in out
    assert _tree(root) == before and not (tmp_path / "g2").exists() and stub_judge["judgments"] == 0
    assert _n_judgments(root) == 1 and _pending_ids(root) == {X1, X2, Y}


# ------------------------------------------------------------------------- --grade-excluded-by: grading a run an exclude ruling retired
#
# A run every paper view must leave out is excluded by an object-scoped `exclude` ruling before it is graded; an excluded run reads
# is_current 0, so without the opt-in the registry path makes it a retired run that never grades its store (pinned below, unchanged). The
# flag names the ruling: a selected run that ruling alone retires, the sole judgeable registration of its store, is graded on its own
# reading. Synthetic ids only.

EXCL = "r_synth-exclude-comparison-objects"           # the ruling the flag names: an object-scoped exclude over the synthetic fitted object
EXCL_OTHER = "r_synth-exclude-other-objects"           # an active exclude over ANOTHER object (it covers none of the fixture runs)
HOLD = "r_synth-hold-out-the-crescendo-run"
SUPER_LATER = "r_synth-supersede-the-crescendo-run-later"
OTHER_OBJECT = "0" * 64


def _ruling(ruling_id: str, action: str, scope_kind: str, scope: dict, i: int) -> dict:
    """A ruling record of the synthetic round (the normaliser builds ``match`` when absent), ruled at ``t(i)``."""
    from sisteer.registry.adapters import roster as RO
    raw = {"ruling_id": ruling_id, "round_id": "r_synth", "action": action, "scope_kind": scope_kind, "scope_json": scope,
           "reason": f"synthetic {action}", "ruled_utc": synth.t(i), "ruled_by": "tests", "source_ref": "tests/test_grade_full22_native_store_currency.py",
           "provenance": {"kind": "user_direct", "message_utc": "2026-10-13T00:00:00Z"}}
    return synth.rec("ruling", RO.normalise_ruling(raw), synth.t(i))


def _excl_object(ruling_id: str = EXCL, object_id: str = synth.OBJECT_SHA, i: int = 300) -> dict:
    return _ruling(ruling_id, "exclude", "object", {"match": {"object_ids": [object_id]}, "note": "synthetic: comparison objects, out of every paper view"}, i)


def _mt_recs(key: str, i0: int, store: str, *, attack: str = "crescendo", through: str = "generated") -> tuple:
    """A multi-turn run of ``attack`` on the synthetic fitted object over ``store``: `generated` (the drivers test's five records), or
    `judged` (generated, then a COMPLETE judgment by the instrument's judge and the judged state -- the one shape v_run_current's later rule
    reads as a successor)."""
    inst = TM._instrument(attack)
    p = synth.run_payload(key, CELL, synth.OBJECT_SHA, i0, eval_id=attack, item_set_id=f"{attack}_set", instrument_id=inst["instrument_id"],
                          decode_source="generation_config", store_uri=store, model_id=synth.MODEL_ID, serving_path="lora_native", n_items_planned=N_GOALS_MT)
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, CELL, attack, str(inst["instrument_id"]))
    rid, t = str(p["run_id"]), synth.t
    recs = [synth.rec("run", p, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log"}, t(i0 + 1)),
            synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact", "evidence_uri": store}, t(i0 + 3)),
            synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": N_GOALS_MT}}, t(i0 + 4))]
    if through == "judged":
        judge = str(inst["judge_id"])
        recs += [synth.rec("judgment", {"judgment_id": I.judgment_id(rid, judge), "run_id": rid, "judge_id": judge, "started_utc": t(i0 + 5),
                                        "finished_utc": t(i0 + 6), "n_items_judged": N_GOALS_MT, "status": "complete", "api_pool": "lowprio"}, t(i0 + 6)),
                 synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"}, t(i0 + 7)),
                 synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": N_GOALS_MT}}, t(i0 + 8))]
    return rid, recs


def _excluded_world(world, tmp_path, monkeypatch, *, rulings=None, extra_runs=()) -> tuple:
    """The crescendo and pair stores of view s1, one `generated` run each, under the exclude ruling EXCL (and EXCL_OTHER, an active exclude
    over another object) -- or under ``rulings`` when given; ``extra_runs`` = [(name, recs)] registered beside them. Returns
    ``(root, {name: run_id}, {run_id: v_run_current facts})``."""
    root, rids = _registry(tmp_path, monkeypatch, [
        ("s1/crescendo", f"{run_of('s1')}/crescendo/{CELL}", "crescendo", str(store_of(world, "s1", "crescendo")), N_GOALS_MT, {}),
        ("s1/pair", f"{run_of('s1')}/pair/{CELL}", "pair", str(store_of(world, "s1", "pair")), N_GOALS_MT, {})])
    for k, (name, (rid, recs)) in enumerate(extra_runs):
        R.write_records(os.path.join(root, "records", "c", f"{100 + k:06d}-extra.jsonl"), recs)
        rids[name] = rid
    R.write_records(os.path.join(root, "records", "rulings_synth", "000091-exclude.jsonl"),
                    list(rulings) if rulings is not None else [_excl_object(), _excl_object(EXCL_OTHER, OTHER_OBJECT, 301)])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    return root, rids, _currency(root, rids.values())


def _currency(root, rids) -> dict:
    with _conn(root) as conn:
        return {rid: dict(zip(("is_current", "why", "selection_state", "decisive_ruling_id", "in_current_paper", "held_out"), conn.execute(
            "SELECT is_current, why, selection_state, decisive_ruling_id, in_current_paper, held_out FROM v_run_current WHERE run_id = ?", (rid,)).fetchone()))
            for rid in rids}


def _flag(*ids) -> list:
    return [a for rid in ids for a in ("--grade-excluded-by", rid)]


def test_excluded_a_sole_pending_run_excluded_by_the_named_ruling_is_graded_on_its_own_reading_and_stays_excluded(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(1) The crescendo run is excluded by EXCL (its decisive ruling and its first failing rule) and is the sole registration of its store.
    Selected with ``--grade-excluded-by EXCL``: claimed, graded on its own reading -- one judgment, the item and metric rows the registry path
    files for any multi-turn pair -- named on stdout and in `registry.graded_excluded`; after the fold it is judged and STILL excluded (is_current
    0, in_current_paper 0, the same decisive ruling) and leaves the pending worklist. The unselected `pair` store, excluded the same way, is
    untouched."""
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch)
    A, B = rids["s1/crescendo"], rids["s1/pair"]
    assert cur[A] == {"is_current": 0, "why": f"selection_state excluded (ruling {EXCL})", "selection_state": "excluded", "decisive_ruling_id": EXCL,
                      "in_current_paper": 0, "held_out": 0}, cur[A]
    assert _pending_ids(root) == {A, B}                                                   # an excluded run is still pending: only currency retires it
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/grade-excluded")
    row = next(r for r in wl.pending() if r["run_id"] == A)
    assert twin.excluded_by_named_ruling_of(wl, row, [EXCL]) == (EXCL, None)
    assert twin.excluded_by_named_ruling_of(wl, row, [EXCL_OTHER])[0] is None
    grades = tmp_path / "grades"
    twin.main(reg_argv(world, "s1", root, attacks="crescendo,pair", grades=grades, run_ids=[A], extra=_flag(EXCL)))
    out, _ = capsys.readouterr()
    line = next(ln for ln in out.splitlines() if "which --grade-excluded-by names" in ln)
    assert A in line and f"excluded by ruling {EXCL}" in line and "graded on its own reading in this invocation; the exclusion stands" in line, line
    assert "worklist=registry: 1 pending (attack, cell) pair(s)" in out and "1 pending run(s) excluded by a ruling --grade-excluded-by names" in out
    assert "claimed 1 pending run(s)" in out and "covered by" not in out and "a retired run" not in out
    assert stub_judge["judgments"] == N_GOALS_MT * N_TURNS                                 # the crescendo store, once; the pair store never
    prov = _provenance(grades, "s1")
    assert [j["run_id"] for j in prov["registry"]["judgments"]] == [A] and prov["registry"]["covered_runs"] == []
    assert prov["registry"]["grade_excluded_by"] == [EXCL]
    assert prov["registry"]["graded_excluded"] == [{"run_id": A, "attack": "crescendo", "cell": CELL, "why": cur[A]["why"], "ruling_id": EXCL,
                                                    "rung": [512, None]}], prov["registry"]["graded_excluded"]
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    with _conn(root) as conn:
        TM._assert_multiturn_judged(conn, A, judgment_prov=prov["registry"]["judgments"][0])
        assert _judgments(conn, B) == [] and _run_state(conn, B)[0] == "generated"
    after = _currency(root, [A])[A]
    assert after["is_current"] == 0 and after["selection_state"] == "excluded" and after["decisive_ruling_id"] == EXCL and after["in_current_paper"] == 0
    assert _pending_ids(root) == {B}


def test_excluded_without_the_flag_the_excluded_run_is_a_retired_run_never_claimed_exactly_as_before(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(2) The same world, no flag: the run is a retired run (no ladder declared) -- covered by no run, never claimed -- and the invocation
    refuses with nothing written, dry and real; the text names nothing of the opt-in."""
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch)
    A = rids["s1/crescendo"]
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], dry=dry))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and f"{A} (crescendo/{CELL}) covered by no run on record -- selection_state excluded (ruling {EXCL})" in msg, msg
        assert "(a retired run: its row declares no ladder (token_cap_ladder_json [], pass label None))" in msg and "grade-excluded" not in msg
    out, _ = capsys.readouterr()
    assert "a retired run never grades its store" in out and "grade-excluded" not in out and "[registry] claimed " not in out
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 0


def test_excluded_by_a_ruling_the_flag_does_not_name_stays_retired_with_the_reason_named(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(3) The run is excluded by EXCL; the flag names EXCL_OTHER (an active exclude over another object): a retired run, the reason the
    opt-in did not apply named in the refusal; nothing claimed or written."""
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch)
    A = rids["s1/crescendo"]
    before = _tree(root)
    for dry in (True, False):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], dry=dry, extra=_flag(EXCL_OTHER)))
        msg = str(ei.value)
        assert "no pending (attack, cell) pair" in msg and "covered by no run on record" in msg, msg
        assert (f"--grade-excluded-by does not apply: it is excluded by ruling {EXCL}, which --grade-excluded-by does not name ({EXCL_OTHER})") in msg, msg
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0


def test_excluded_and_superseded_by_a_later_complete_run_at_its_key_stays_retired(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(4) A later COMPLETE run at the run's key over another store (a re-run elsewhere): the exclusion is the first failing rule in `why`
    and masks the successor, so the relation is asked of the view's own rendering -- the run stays retired, the successor named."""
    succ_key = f"{run_of('s1')}/crescendo/{CELL}#rerun-elsewhere"
    S, recs = _mt_recs(succ_key, 250, "synthetic:///store/rerun-elsewhere/crescendo", through="judged")
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, extra_runs=[("succ", (S, recs))])
    A = rids["s1/crescendo"]
    assert cur[A]["why"] == f"selection_state excluded (ruling {EXCL})"                    # the successor is masked behind the exclusion
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/grade-excluded")
    row = next(r for r in wl.pending() if r["run_id"] == A)
    assert twin.later_run_of(wl, row) == [S]
    ex, why_not = twin.excluded_by_named_ruling_of(wl, row, [EXCL])
    assert ex is None and S in why_not and "later complete run at its key" in why_not
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(EXCL)))
    assert f"--grade-excluded-by does not apply: it is superseded by a later complete run at its key, {S}" in str(ei.value), str(ei.value)
    assert stub_judge["judgments"] == 0 and not (tmp_path / "g").exists() and _n_judgments(root) == 1          # the successor's own, only


@pytest.mark.parametrize("variant", ["held_out", "superseded_by_a_later_ruling"])
def test_excluded_but_held_out_or_superseded_by_a_later_ruling_stays_retired(twin, world, tmp_path, monkeypatch, capsys, stub_judge, variant):
    """(5) A hold-out beside the exclusion (`why` still names the exclusion; the held_out column does not), or a `supersede` ruling naming the
    run ruled AFTER the exclusion (the decisive ruling is then the supersede): either way a retired run with the flag, the reason named."""
    A, _ = TM._run_recs(f"{run_of('s1')}/crescendo/{CELL}", 200, "crescendo", str(store_of(world, "s1", "crescendo")), n_items=N_GOALS_MT)  # the id: a function of the key
    extra = (_ruling(HOLD, "hold_out", "run", {"match": {"run_ids": [A]}}, 310) if variant == "held_out"
             else _ruling(SUPER_LATER, "supersede", "run", {"match": {"run_ids": [A]}}, 400))
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, rulings=[_excl_object(), extra])
    assert rids["s1/crescendo"] == A
    if variant == "held_out":
        assert cur[A]["why"] == f"selection_state excluded (ruling {EXCL})" and cur[A]["held_out"] == 1, cur[A]
        reason = "--grade-excluded-by does not apply: it is held out"
    else:
        assert cur[A]["selection_state"] == "superseded" and cur[A]["decisive_ruling_id"] == SUPER_LATER, cur[A]
        reason = "--grade-excluded-by does not apply: its selection_state is 'superseded' (decisive action 'supersede'), not an exclusion"
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(EXCL)))
    assert reason in str(ei.value) and "no pending (attack, cell) pair" in str(ei.value), str(ei.value)
    assert stub_judge["judgments"] == 0 and _n_judgments(root) == 0


BASE_FACTS = {"is_current": 0, "why": f"selection_state excluded (ruling {EXCL})", "selection_state": "excluded", "decisive_action": "exclude",
              "decisive_ruling_id": EXCL, "held_out": 0, "superseded_by_run_id": None, "lifecycle_state": "generated"}


@pytest.mark.parametrize("change, reason", [
    ({}, None),
    ({"lifecycle_state": "judged"}, None),
    ({"is_current": 1, "why": None}, "the run is current"),
    ({"selection_state": "withdrawn", "why": "selection_state withdrawn"}, "its selection_state is 'withdrawn' (decisive action 'exclude'), not an exclusion"),
    ({"selection_state": "indeterminate"}, "its selection_state is 'indeterminate'"),
    ({"decisive_action": "supersede"}, "not an exclusion"),
    ({"decisive_ruling_id": EXCL_OTHER, "why": f"selection_state excluded (ruling {EXCL_OTHER})"}, f"excluded by ruling {EXCL_OTHER}, which --grade-excluded-by does not name"),
    ({"why": "selection_state excluded (ruling something-else)"}, "not the exclusion alone"),
    ({"held_out": 1}, "it is held out"),
    ({"superseded_by_run_id": "synthetic:crescendo:111111111111:2222222222222222"}, "it is superseded by run synthetic:crescendo:"),
    ({"lifecycle_state": "failed"}, "its lifecycle_state 'failed' admits no judging"),
    ({"lifecycle_state": "withdrawn"}, "its lifecycle_state 'withdrawn' admits no judging"),
])
def test_excluded_ruling_of_facts_reads_each_retirement_from_its_own_column(twin, change, reason):
    """The pure half: the exclusion named by the flag must be the ONE thing retiring the run -- each other retirement read from its column."""
    got = twin.excluded_ruling_of_facts({**BASE_FACTS, **change}, [EXCL])
    if reason is None:
        assert got == (EXCL, None)
    else:
        assert got[0] is None and reason in got[1], got
    assert twin.excluded_ruling_of_facts(BASE_FACTS, [])[0] is None


def test_excluded_two_pending_registrations_of_one_store_keep_the_one_store_refusal_with_the_flag(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(6) Two pending registrations name the crescendo store, both excluded by EXCL: the one-store-one-run refusal stands, with and without
    the flag (the flag never decides between registrations); nothing claimed."""
    twin_key = f"{run_of('s1')}/crescendo/{CELL}#g5678"
    T2, recs = _mt_recs(twin_key, 260, str(store_of(world, "s1", "crescendo")))
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, extra_runs=[("twin", (T2, recs))])
    A = rids["s1/crescendo"]
    for extra in ((), _flag(EXCL)):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A, T2], extra=extra))
        msg = str(ei.value)
        assert "two pending runs name the store of crescendo/" in msg and "— one store, one run; refusing" in msg and twin.STORE_RUNS_REFUSAL_PHRASE in msg, msg
    assert stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and not (tmp_path / "g").exists()


def test_the_flag_is_refused_by_name_off_its_one_path(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(7) Without --run-id; under --worklist from-disk; beside the BoN leg; naming a ruling the registry does not carry, or one that is not an
    exclude: each refused by name before anything is claimed or written."""
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, rulings=[_excl_object(), _r2v1_like_ruling()])
    A = rids["s1/crescendo"]
    before = _tree(root)
    cases = [
        (reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", extra=_flag(EXCL)), "--grade-excluded-by needs an explicit --run-id selection"),
        (TM.argv(world, "s1", attacks="crescendo", grades=tmp_path / "g", dry=False, extra=_flag(EXCL) + ["--run-id", A]), "--grade-excluded-by needs --worklist registry"),
        (reg_argv(world, "s1", root, attacks="crescendo,bon", grades=tmp_path / "g", run_ids=[A], extra=_flag(EXCL)), "--grade-excluded-by grades the multi-turn legs only"),
        (reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag("r_synth-no-such-ruling")),
         "r_synth-no-such-ruling: no ruling of that id in the registry"),
        (reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(EXCL, R2V1_LIKE)),
         f"{R2V1_LIKE}: action 'supersede', not `exclude`"),
    ]
    for argv_, why in cases:
        with pytest.raises(SystemExit) as ei:
            twin.main(argv_)
        assert why in str(ei.value), (why, str(ei.value))
    capsys.readouterr()
    assert _tree(root) == before and not (tmp_path / "g").exists() and stub_judge["judgments"] == 0 and _n_judgments(root) == 0


def test_a_dry_run_with_the_flag_prints_the_excluded_run_and_touches_nothing(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """(8) ``--dry-run`` with the flag: the plan names the run, EXCLUDED by the ruling and graded on its own reading; nothing is claimed,
    recorded or written (the registry root byte-identical, no grades root)."""
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch)
    A = rids["s1/crescendo"]
    before = _tree(root)
    twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g0", run_ids=[A], dry=True, extra=_flag(EXCL)))
    out, _ = capsys.readouterr()
    plan = [ln for ln in out.splitlines() if ln.startswith("[registry] DRY RUN:")]
    assert any(f"run_id={A}" in ln and f"— EXCLUDED by ruling {EXCL}, graded on its own reading (--grade-excluded-by)" in ln for ln in plan), plan
    assert any("1 pending (attack, cell) pair(s)" in ln for ln in plan) and "nothing claimed, nothing recorded, nothing written — done" in out
    assert _tree(root) == before and not (tmp_path / "g0").exists() and stub_judge["judgments"] == 0 and _pending_ids(root) == {A, rids["s1/pair"]}


def test_the_early_stopped_control_views_resolve_as_targets(twin):
    """(9) The three multi-turn views of the non-production roster resolve by name: their private runs and the two multi-turn attacks; no
    phase-view grammar is widened for them."""
    targets, _ = twin.resolve_targets()
    for k in (1, 2, 3):
        tag = f"qwen35_4b_ctlstop64s{k}"
        assert targets[tag] == {"run": f"sdfbase_full22_{tag}", "default_attacks": ("crescendo", "actorattack")}
        assert not twin.PHASE_VIEW_RE.match(f"ctlstop64s{k}")


# ------------------------------------------------------- --grade-excluded-by: superseding, hidden supersessions, dead runs, ruling scope

EXCL_RUN = "r_synth-exclude-the-crescendo-run-by-id"     # an exclude naming ONE run (a scope that does not cover its same-key predecessor)
HIDDEN_SUPER = "r_synth-supersede-the-crescendo-run-earlier"
EARLIER_KEY = f"{run_of('s1')}/crescendo/{CELL}#earlier-elsewhere"


def _crescendo_id() -> str:
    """The fixture crescendo run's id (a function of its producer key, eval and instrument)."""
    return TM._run_recs(f"{run_of('s1')}/crescendo/{CELL}", 200, "crescendo", "synthetic:///unused", n_items=N_GOALS_MT)[0]


@pytest.mark.parametrize("predecessor_excluded", [False, True])
def test_excluded_grading_never_supersedes_an_earlier_run_at_its_key_that_no_named_exclusion_retires(twin, world, tmp_path, monkeypatch, capsys,
                                                                                                      stub_judge, predecessor_excluded):
    """An EARLIER run Y at the crescendo run's key (same object, eval, instrument, judge, cap; another store). Graded, the excluded run would
    become the later complete run at that key and Y would read superseded. When Y is retired by no ruling the flag names (the exclusion names
    the one run), the opt-in refuses with Y named and nothing is claimed; when the same named exclusion retires Y too (an object-scoped
    exclude), grading supersedes nothing that was not already out, and the run is graded."""
    A = _crescendo_id()
    Y, recs = _mt_recs(EARLIER_KEY, 150, "synthetic:///store/earlier-elsewhere/crescendo")
    rulings = ([_excl_object(), _excl_object(EXCL_OTHER, OTHER_OBJECT, 301)] if predecessor_excluded
               else [_ruling(EXCL_RUN, "exclude", "run", {"match": {"run_ids": [A]}}, 300)])
    named = EXCL if predecessor_excluded else EXCL_RUN
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, rulings=rulings, extra_runs=[("earlier", (Y, recs))])
    assert rids["s1/crescendo"] == A and cur[A]["decisive_ruling_id"] == named and cur[A]["selection_state"] == "excluded"
    assert cur[Y]["selection_state"] == ("excluded" if predecessor_excluded else "not_in_roster") and cur[Y]["is_current"] == (0 if predecessor_excluded else 1)
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/grade-excluded")
    rel = twin.later_relation_of(wl, [A])
    assert rel[A]["successors"] == [] and set(rel[A]["predecessors"]) == {Y}, rel
    row = next(r for r in wl.pending() if r["run_id"] == A)
    if not predecessor_excluded:
        ex, why_not = twin.excluded_by_named_ruling_of(wl, row, [named])
        assert ex is None and f"later complete run at its key over {Y}" in why_not, why_not
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(named)))
        assert f"--grade-excluded-by does not apply: graded, it would become the later complete run at its key over {Y}" in str(ei.value), str(ei.value)
        assert stub_judge["judgments"] == 0 and _n_judgments(root) == 0 and not (tmp_path / "g").exists()
        return
    assert twin.excluded_by_named_ruling_of(wl, row, [named]) == (named, None)
    twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(named)))
    capsys.readouterr()
    assert stub_judge["judgments"] == N_GOALS_MT * N_TURNS
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    after = _currency(root, [A, Y])
    assert after[A]["selection_state"] == after[Y]["selection_state"] == "excluded" and after[A]["in_current_paper"] == after[Y]["in_current_paper"] == 0


def test_excluded_a_supersede_hidden_behind_the_named_exclusion_stays_retired(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """An OLDER supersede ruling naming the run, then the exclusion: the exclusion is the decisive (latest) ruling and the first failing rule,
    but the run's covering rulings carry the supersede -- the opt-in refuses with it named."""
    A = _crescendo_id()
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, rulings=[
        _ruling(HIDDEN_SUPER, "supersede", "run", {"match": {"run_ids": [A]}}, 250), _excl_object()])
    assert cur[A]["why"] == f"selection_state excluded (ruling {EXCL})" and cur[A]["decisive_ruling_id"] == EXCL, cur[A]
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/grade-excluded")
    ex, why_not = twin.excluded_by_named_ruling_of(wl, next(r for r in wl.pending() if r["run_id"] == A), [EXCL])
    assert ex is None and f"also covered by supersede ruling(s) {HIDDEN_SUPER}" in why_not, why_not
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(EXCL)))
    assert f"--grade-excluded-by does not apply: it is also covered by supersede ruling(s) {HIDDEN_SUPER}" in str(ei.value), str(ei.value)
    assert stub_judge["judgments"] == 0 and _n_judgments(root) == 0


@pytest.mark.parametrize("dead", ["withdrawn", "superseded_by_run_id"])
def test_excluded_a_withdrawn_or_successor_marked_run_is_never_graded_with_the_flag(twin, world, tmp_path, monkeypatch, capsys, stub_judge, dead):
    """A run withdrawn (its lifecycle), or carrying a superseded_by_run_id (a same-id re-emission naming its successor), under the named
    exclusion: neither is pending, the selected invocation finds nothing to grade, nothing is claimed or written; the facts name why."""
    A, recs_a = TM._run_recs(f"{run_of('s1')}/crescendo/{CELL}", 200, "crescendo", str(store_of(world, "s1", "crescendo")), n_items=N_GOALS_MT)
    S, recs_s = _mt_recs(f"{run_of('s1')}/crescendo/{CELL}#successor", 280, "synthetic:///store/successor/crescendo")
    if dead == "withdrawn":
        extra = [("withdrawal", (A, [synth.rec("run_state", {"run_id": A, "state": "withdrawn", "event_utc": synth.t(260), "evidence_kind": "job_log"},
                                                synth.t(260))]))]
    else:
        extra = [("successor", (S, recs_s)), ("correction", (A, [synth.rec("run", dict(recs_a[0]["payload"], superseded_by_run_id=S), synth.t(290))]))]
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, extra_runs=extra)
    if dead == "withdrawn":
        assert cur[A]["selection_state"] == "withdrawn", cur[A]
        why = "its selection_state is 'withdrawn'"
    else:
        assert cur[A]["selection_state"] == "excluded" and cur[A]["why"] == f"selection_state excluded (ruling {EXCL})", cur[A]
        why = f"it is superseded by run {S}"
    assert A not in _pending_ids(root)
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/grade-excluded")
    ex, why_not = twin.excluded_by_named_ruling_of(wl, {"run_id": A}, [EXCL])
    assert ex is None and why in why_not, why_not
    before = _tree(root)
    with pytest.raises(SystemExit) as ei:
        twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(EXCL)))
    assert "no pending (attack, cell) pair" in str(ei.value) and "nothing claimed, nothing written" in str(ei.value), str(ei.value)
    assert _tree(root) == before and stub_judge["judgments"] == 0 and not (tmp_path / "g").exists()


def test_the_flag_refuses_a_named_exclude_of_a_closed_round_or_a_comparison_scope(twin, world, tmp_path, monkeypatch, capsys, stub_judge):
    """A named exclude that can never decide a run -- ruled in a CLOSED round, or comparison-scoped -- refuses by name before anything is read
    or claimed (it would otherwise match nothing and end in a no-pairs refusal that names no reason)."""
    from sisteer.registry.adapters import roster as RO
    old_round = synth.rec("paper_round", {"round_id": "r_old", "opened_utc": synth.t(1), "closed_utc": synth.t(2), "description": "a closed round"}, synth.t(1))
    raw = {"ruling_id": "r_old-exclude-comparison-objects", "round_id": "r_old", "action": "exclude", "scope_kind": "object",
           "scope_json": {"match": {"object_ids": [synth.OBJECT_SHA]}}, "reason": "synthetic exclude in a closed round", "ruled_utc": synth.t(3),
           "ruled_by": "tests", "source_ref": "tests/test_grade_full22_native_store_currency.py", "provenance": {"kind": "user_direct", "message_utc": "2026-10-13T00:00:00Z"}}
    closed = synth.rec("ruling", RO.normalise_ruling(raw), synth.t(3))
    comparison = _ruling("r_synth-exclude-a-comparison", "exclude", "comparison", {"comparison_id": "synthetic-comparison"}, 302)
    A = _crescendo_id()
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch, rulings=[old_round, closed, _excl_object(), comparison])
    before = _tree(root)
    for rid, why in (("r_old-exclude-comparison-objects", "round 'r_old', not the open round 'r_synth'"),
                     ("r_synth-exclude-a-comparison", "comparison-scoped (it never decides a run)")):
        with pytest.raises(SystemExit) as ei:
            twin.main(reg_argv(world, "s1", root, attacks="crescendo", grades=tmp_path / "g", run_ids=[A], extra=_flag(rid)))
        assert f"{rid}: {why}" in str(ei.value) and "refusing before anything is claimed" in str(ei.value), str(ei.value)
    assert _tree(root) == before and stub_judge["judgments"] == 0


@pytest.mark.parametrize("covering, actions, reason", [
    ([EXCL], {}, None),
    ([EXCL, "r_synth-include-x"], {"r_synth-include-x": "include"}, None),
    ([EXCL, HIDDEN_SUPER], {HIDDEN_SUPER: "supersede"}, f"also covered by supersede ruling(s) {HIDDEN_SUPER}"),
    ([EXCL, HIDDEN_SUPER], {}, f"the actions of rulings covering it are unread: {HIDDEN_SUPER}"),
])
def test_excluded_ruling_of_facts_reads_every_covering_ruling(twin, covering, actions, reason):
    got = twin.excluded_ruling_of_facts({**BASE_FACTS, "ruling_ids_json": json.dumps(covering)}, [EXCL], actions)
    if reason is None:
        assert got == (EXCL, None), got
    else:
        assert got[0] is None and reason in got[1], got
    assert twin.excluded_ruling_of_facts({**BASE_FACTS, "ruling_ids_json": json.dumps([EXCL, HIDDEN_SUPER])}, [EXCL])[0] is None   # actions unread


def test_the_currency_read_carries_the_facts_only_on_request(twin, world, tmp_path, monkeypatch):
    """run_currency_of's default result is exactly as before (two keys); with facts=True the SAME one read carries the opt-in's columns."""
    root, rids, cur = _excluded_world(world, tmp_path, monkeypatch)
    A = rids["s1/crescendo"]
    wl = W.RegistryWorklist(root, None, "s-T", driver="tests/grade-excluded")
    assert twin.run_currency_of(wl, [A]) == {A: {"is_current": 0, "why": f"selection_state excluded (ruling {EXCL})"}}
    f = twin.run_currency_of(wl, [A], facts=True)[A]
    assert set(f) == set(twin.EXCLUDED_FACT_COLUMNS) and f["decisive_ruling_id"] == EXCL and f["lifecycle_state"] == "generated"
    assert json.loads(f["ruling_ids_json"]) == [EXCL]
