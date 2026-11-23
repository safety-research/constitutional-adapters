"""The multi-turn grader's claims never outlive the dispatch that took them (``scripts/sdfbase/seedrep/grade_full22_native.py``).

The defect: under ``--worklist registry`` every located pair is claimed before the judge is built, and a multi-turn cell's JB1
refusal ended the dispatch in that cell without releasing anything, so every LATER claimed cell's claim stayed ``in_progress`` with
nothing judging it and every later session met it as held. Pinned here, on the sibling module's synthetic world (placeholder replies,
``TESTHOST`` stores, a tmp registry, a stand-in judge API):

* grading is unchanged: a cell's JB1 refusal still ends the dispatch in that cell, raising the grader's own line;
* any exception that leaves the grader once the claims start releases every claim this invocation holds whose scope never opened
  (main's ClaimGuard): a cell the dispatch never reached -- after a JB1 refusal or any other failure, or a later attack -- is pending
  again at once;
* the guard is armed BEFORE ``worklist.claim_all``: an exception that interrupts the claim loop itself (the write lock held past its
  timeout on a later claim, which claim_all does not release for) releases the claims already taken;
* a pair whose scope opened is never released (its scope closed it), a claim this invocation never took (another session's, or one
  claim_all already released) is never touched, and a failed release never masks the grader's own exception.
"""
from __future__ import annotations

import json

import pytest

from sisteer.registry import db as DB
from sisteer.registry import pending as P

import test_grade_full22_native_main as TM
from test_grade_judge_blocked_allowance import (  # noqa: F401  (fixtures the tests below take by name)
    CELL, _fold, _item_rows, _judgment_row, judge, mt_argv, mt_registry, mt_world, twin)

LORA = TM.bf.LORA_CELL


def _status(conn, rid) -> list:
    return [j["status"] for j in _judgment_row(conn, rid)]


def _in_progress(conn, rids) -> list:
    return [tuple(r) for r in conn.execute(
        f"SELECT run_id, judgment_id FROM judgment WHERE status = 'in_progress' AND run_id IN ({','.join('?' * len(rids))})", list(rids))]


def test_a_jb1_refusal_ends_the_dispatch_and_releases_the_later_cells_claim(twin, mt_world, judge, tmp_path, monkeypatch, capsys):
    root, rids = mt_registry(tmp_path, monkeypatch, mt_world, [("actorattack", CELL), ("actorattack", LORA)])
    judge["plan"] = {("actorattack", "base", 0, 1): "refusal", ("actorattack", "lora", 0, 0): 1.0}
    with pytest.raises(RuntimeError) as ei:
        twin.main(mt_argv(mt_world, root, tmp_path / "grades", cells=(CELL, LORA), run_ids=list(rids.values())))
    # the grader's own JB1 line, exactly as before: the dispatch ends in the refusing cell
    assert type(ei.value) is RuntimeError
    assert str(ei.value).startswith("1 judge item(s) still UNSCORED after live grading (bounded retries) — refusing to aggregate a partial grade (JB1).")
    out, err = capsys.readouterr()
    assert f"[actorattack] {LORA}" not in out                                    # the later cell was never graded
    assert f"released 1 claim(s) whose pair the dispatch never reached (no scope opened), so they are pending again: {rids[('actorattack', LORA)]}" in err
    with TM._conn(root) as conn:                                                 # BEFORE any harvester fold: the release folded itself
        assert _status(conn, rids[("actorattack", LORA)]) == ["abandoned"]
        assert rids[("actorattack", LORA)] in {r["run_id"] for r in P.pending(conn)}
    _fold(root)
    with TM._conn(root) as conn:
        assert _status(conn, rids[("actorattack", CELL)]) == ["partial"]
        assert _item_rows(conn, rids[("actorattack", CELL)], "strongreject_score") == []
        assert _item_rows(conn, rids[("actorattack", LORA)], "strongreject_score") == []
        assert _in_progress(conn, rids.values()) == []                          # no claim outlives the dispatch
        assert set(rids.values()) <= {r["run_id"] for r in P.pending(conn)}


def test_a_non_jb1_failure_releases_the_claims_the_dispatch_never_reached(twin, mt_world, judge, tmp_path, monkeypatch, capsys):
    root, rids = mt_registry(tmp_path, monkeypatch, mt_world, [("actorattack", CELL), ("actorattack", LORA)])
    judge["plan"] = {("actorattack", "base", 1, 0): "boom"}
    with pytest.raises(ValueError, match="synthetic non-JB1 failure"):          # the grader's own exception propagates, never masked
        twin.main(mt_argv(mt_world, root, tmp_path / "grades", cells=(CELL, LORA), run_ids=list(rids.values())))
    err = capsys.readouterr().err
    assert f"released 1 claim(s) whose pair the dispatch never reached (no scope opened), so they are pending again: {rids[('actorattack', LORA)]}" in err
    with TM._conn(root) as conn:                                              # BEFORE any harvester fold: the release folded itself
        assert _status(conn, rids[("actorattack", LORA)]) == ["abandoned"]
        assert rids[("actorattack", LORA)] in {r["run_id"] for r in P.pending(conn)}
        assert _status(conn, rids[("actorattack", CELL)]) == ["in_progress"]   # its scope opened: closed by its own (spooled) partial
    _fold(root)
    with TM._conn(root) as conn:
        assert _status(conn, rids[("actorattack", CELL)]) == ["partial"] and _in_progress(conn, rids.values()) == []


def test_a_jb1_refusal_releases_a_later_attacks_claim(twin, mt_world, judge, tmp_path, monkeypatch, capsys):
    root, rids = mt_registry(tmp_path, monkeypatch, mt_world, [("crescendo", CELL), ("actorattack", CELL)])
    judge["plan"] = {("crescendo", "base", 0, 1): "refusal"}
    with pytest.raises(RuntimeError, match=r"still UNSCORED after live grading \(bounded retries\)"):
        twin.main(mt_argv(mt_world, root, tmp_path / "grades", attacks="crescendo,actorattack", run_ids=list(rids.values())))
    assert f"pending again: {rids[('actorattack', CELL)]}" in capsys.readouterr().err
    _fold(root)
    with TM._conn(root) as conn:
        assert _status(conn, rids[("crescendo", CELL)]) == ["partial"]
        assert _status(conn, rids[("actorattack", CELL)]) == ["abandoned"]
        assert _in_progress(conn, rids.values()) == []
        assert {rids[("crescendo", CELL)], rids[("actorattack", CELL)]} <= {r["run_id"] for r in P.pending(conn)}


def test_a_lock_timeout_inside_claim_all_releases_the_claims_it_took(twin, mt_world, judge, tmp_path, monkeypatch, capsys):
    """claim_all releases what it took only on a claim REFUSAL; the write lock held past its timeout on the second claim is a plain
    RuntimeError (db.LockHeld) that it lets through with the first claim still in_progress. The guard is armed before claim_all, so
    the first claim is released, and the grader's own exception propagates."""
    root, rids = mt_registry(tmp_path, monkeypatch, mt_world, [("actorattack", CELL), ("actorattack", LORA)])
    real_claim, calls = P.claim, []

    def _claim(*a, **kw):
        calls.append(a[1])
        if len(calls) == 2:
            raise DB.LockHeld("synthetic write lock held past its timeout")
        return real_claim(*a, **kw)

    monkeypatch.setattr(P, "claim", _claim)
    with pytest.raises(DB.LockHeld, match="synthetic write lock held"):
        twin.main(mt_argv(mt_world, root, tmp_path / "grades", cells=(CELL, LORA), run_ids=list(rids.values())))
    assert len(calls) == 2 and judge["calls"] == 0                            # it died in the claim loop, before any judge call
    first, second = calls
    err = capsys.readouterr().err
    assert f"released 1 claim(s) whose pair the dispatch never reached (no scope opened), so they are pending again: {first}" in err
    with TM._conn(root) as conn:
        assert _status(conn, first) == ["abandoned"] and _status(conn, second) == []
        assert _in_progress(conn, rids.values()) == []
        assert set(rids.values()) <= {r["run_id"] for r in P.pending(conn)}


def test_a_claim_refusal_leaves_another_sessions_claim_alone(twin, mt_world, judge, tmp_path, monkeypatch, capsys):
    """A claim another session holds refuses the dispatch whole (claim_all releases what this call took); the guard, armed before
    claim_all, releases nothing more and never touches the other session's claim."""
    root, rids = mt_registry(tmp_path, monkeypatch, mt_world, [("actorattack", CELL), ("actorattack", LORA)])
    with TM._conn(root) as conn:
        rows = {r["run_id"]: r for r in P.pending(conn)}
    held_rid = max(rids.values())                                              # whichever claim_all reaches second or first, it refuses
    held = P.claim(root, held_rid, str(rows[held_rid]["required_judge_id"]), "s-B")
    with pytest.raises(SystemExit, match="--worklist registry: refusing"):
        twin.main(mt_argv(mt_world, root, tmp_path / "grades", cells=(CELL, LORA), run_ids=list(rids.values())))
    assert "released" not in capsys.readouterr().err and judge["calls"] == 0
    with TM._conn(root) as conn:
        assert _in_progress(conn, rids.values()) == [(held_rid, held)]          # the other session's claim, untouched
        (other,) = [r for r in rids.values() if r != held_rid]
        assert _status(conn, other) in ([], ["abandoned"])                     # never claimed, or claimed and released by claim_all


def test_a_clean_dispatch_releases_nothing(twin, mt_world, judge, tmp_path, monkeypatch, capsys):
    root, rids = mt_registry(tmp_path, monkeypatch, mt_world, [("actorattack", CELL), ("actorattack", LORA)])
    judge["plan"] = {("actorattack", "lora", 1, 1): 0.875}
    twin.main(mt_argv(mt_world, root, tmp_path / "grades", cells=(CELL, LORA), run_ids=list(rids.values())))
    assert "released" not in capsys.readouterr().err
    _fold(root)
    with TM._conn(root) as conn:
        assert _status(conn, rids[("actorattack", CELL)]) == ["complete"] and _status(conn, rids[("actorattack", LORA)]) == ["complete"]
    agg = json.loads((tmp_path / "grades" / TM.tag(mt_world["view"]) / "two_si_actorattack_aggregate.json").read_text())
    assert set(agg["cells"]) == {CELL, LORA}


def test_claim_guard_releases_only_unopened_pairs_this_worklist_holds_and_never_raises(twin, capsys):
    calls = []

    class _WL:
        # the worklist's claims map: (run_id, required_judge_id) of every claim it holds; r-never was never claimed by it
        claims = {(r, "j"): {} for r in ("r-open", "r-bad", "r-ok", "r-gone")}

        def release(self, row):
            calls.append(row["run_id"])
            if row["run_id"] == "r-bad":
                raise RuntimeError("synthetic lock held")
            return row["run_id"] != "r-gone"

    g = twin.ClaimGuard()
    assert g.release_unopened() == []                                          # nothing tracked: nothing to do
    g.track(_WL(), {("crescendo", "a"): {"run_id": "r-open", "required_judge_id": "j"},
                    ("crescendo", "b"): {"run_id": "r-bad", "required_judge_id": "j"},
                    ("pair", "a"): {"run_id": "r-ok", "required_judge_id": "j"},
                    ("pair", "b"): {"run_id": "r-gone", "required_judge_id": "j"},
                    ("pair", "c"): {"run_id": "r-never", "required_judge_id": "j"}})
    g.open(("crescendo", "a"))
    assert g.release_unopened() == ["r-ok"]
    assert sorted(calls) == ["r-bad", "r-gone", "r-ok"]                         # never the opened pair, never a claim it does not hold
    err = capsys.readouterr().err
    assert "could not release the claim on run r-bad (crescendo/b): RuntimeError: synthetic lock held" in err and "pending again: r-ok" in err
