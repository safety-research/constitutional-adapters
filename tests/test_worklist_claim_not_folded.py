"""`pending.claim` refuses by name -- `pending.ClaimNotFolded` -- a claim whose run or judge row has not folded yet (nothing written; the
caller retries after the harvester folds the run). `RegistryWorklist.claim_all` released the claims it had just taken and re-raised only on
`ClaimHeld`, so a ClaimNotFolded on the SECOND row of a worklist would have leaked the first row's claim. Now the refusal tuple
`worklist._CLAIM_REFUSALS` (ClaimHeld + ClaimNotFolded when the registry's pending module has it; resolved at import, so the module loads
on a tree without the fix, where ClaimHeld stands in) gets the same release-and-re-raise. Synthetic registry (test_grade_daemon._root)
only."""
from __future__ import annotations

import pytest

from sisteer.registry import db as DB
from sisteer.registry import pending as P
from sisteer.registry import synth
from sisteer.registry import worklist as RW

from test_grade_daemon import _root


def _refusal(cls, run_id: str):
    """An instance of whichever class the module resolved for ClaimNotFolded: the real one (run_id, what) after the hotfix; ClaimHeld's
    signature (run_id, session, age) on a tree without it."""
    return cls(run_id, "someone-else@TESTHOST", 1) if cls is P.ClaimHeld else cls(run_id, "test: no run row yet; nothing written")


def test_a_not_folded_refusal_on_the_second_row_releases_the_first_rows_claim_and_re_raises(tmp_path, monkeypatch):
    root, rids = _root(tmp_path)
    assert RW._CLAIM_REFUSALS[0] is P.ClaimHeld and RW._CLAIM_REFUSALS[-1] is getattr(P, "ClaimNotFolded", P.ClaimHeld)
    cls = RW._CLAIM_REFUSALS[-1]
    wl = RW.RegistryWorklist(root, None, "fake-driver", synth.CAMPAIGN, driver="test_worklist_claim_not_folded", model_ids=[synth.MODEL_ID])
    rows = sorted(wl.pending(), key=lambda r: str(r["run_id"]))
    assert [str(r["run_id"]) for r in rows] == sorted(rids[k] for k in ("job7@none", "job7@steer"))
    first, second = str(rows[0]["run_id"]), str(rows[1]["run_id"]); judge = str(rows[0]["required_judge_id"])
    real_claim = P.claim; calls = []

    def refusing_claim(root_, run_id, judge_id, session, **kw):
        calls.append(run_id)
        if run_id == second:
            raise _refusal(cls, run_id)
        return real_claim(root_, run_id, judge_id, session, **kw)
    monkeypatch.setattr(RW.P, "claim", refusing_claim)
    with pytest.raises(cls) as ei:
        wl.claim_all(rows)
    assert calls == [first, second] and second in str(ei.value)
    assert wl.claims == {}                                                                      # the first row's claim is forgotten by the worklist ...
    with DB.reading(DB.db_path(root)) as conn:
        assert P.active_claim(conn, first, judge) is None                                      # ... and released in the registry (abandoned), never leaked
        assert conn.execute("SELECT status FROM judgment WHERE run_id = ? ORDER BY claimed_utc DESC LIMIT 1", (first,)).fetchone()[0] == "abandoned"
        assert P.active_claim(conn, second, judge) is None and conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (second,)).fetchone()[0] == 0
    # the existing contract stands: a ClaimHeld on the second row releases the first the same way
    def held_claim(root_, run_id, judge_id, session, **kw):
        if run_id == second:
            raise P.ClaimHeld(run_id, "x@TESTHOST", 1)
        return real_claim(root_, run_id, judge_id, session, **kw)
    monkeypatch.setattr(RW.P, "claim", held_claim)
    with pytest.raises(P.ClaimHeld):
        wl.claim_all(rows)
    assert wl.claims == {}
    with DB.reading(DB.db_path(root)) as conn:
        assert P.active_claim(conn, first, judge) is None
    # with the refusals out of the way the worklist claims both (the real claim, unpatched)
    monkeypatch.setattr(RW.P, "claim", real_claim)
    claims = wl.claim_all(rows)
    assert sorted(rid for rid, _j in claims) == [first, second]
    with DB.reading(DB.db_path(root)) as conn:
        assert P.active_claim(conn, first, judge)["claimed_by_session"] == "fake-driver" and P.active_claim(conn, second, judge) is not None
