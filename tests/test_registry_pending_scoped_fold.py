"""A claim or a release folds ONLY its own just-written record under the write lock -- never a harvester cycle.

Before, `pending.claim` and `pending.release` ran `harvest.ingest_once` (a FULL single-threaded cycle: every new or changed spool
file, the blocked queue, the ranks, the cycle stamp) inside `db.write_lock` at every call; a judging driver claiming 7 runs paid 7
cycles, a spool backlog the refused harvester could not drain made each one longer, and the hold starved the harvester (8
refusals/h on 2026-10-02). Now `ingest_once(only_files=[path])` folds the claim file alone (`harvest.fold_only_files`): the
unrelated spooled files stay unfolded, `state/harvest_cycle.json` is untouched, the hold is milliseconds -- and the claim is still
visible in `v_pending_judgment` / the `judgment` row the moment `claim` returns, which is the semantics a second claimant relies
on. `pending.ingest_now` stays the one explicit full cycle.
"""
from __future__ import annotations

import json
import logging
import os
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth

N_BACKLOG = 200


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    return root, ids


def _spool_backlog(root, n=N_BACKLOG):
    """``n`` unrelated run record files nobody has folded yet -- the spool a refused harvester leaves behind."""
    paths = []
    for i in range(n):
        _rid, recs = synth.run_records(f"backlog{i}", 5000 + i, through="generated", n_items=1)
        paths.append(_write(root, "backlog", f"{i + 1:06d}-run.jsonl", recs))
    return paths


def _stamp(root):
    p = os.path.join(H.state_dir(root), H.CYCLE_STATE_NAME)
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read(), os.stat(p).st_mtime_ns


def _delivered_backlog(conn):
    return conn.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE 'records/backlog/%'").fetchone()[0]


def _hold_seconds(caplog, verb):
    msgs = [r.getMessage() for r in caplog.records if r.name == "sisteer.registry.pending" and r.getMessage().startswith(f"pending: {verb} ")]
    assert msgs, [r.getMessage() for r in caplog.records]
    m = msgs[-1]
    held = float(m.split("held the write lock ")[1].split(" s,")[0])
    folded = int(m.split("folded ")[1].split(" record")[0])
    return held, folded


def test_claim_with_a_spool_backlog_folds_only_its_own_record(tmp_path, caplog):
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][0]
    _spool_backlog(root)
    stamp_before = _stamp(root)
    caplog.set_level(logging.INFO, logger="sisteer.registry.pending")
    t0 = time.monotonic()
    jid = P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1000))
    wall = time.monotonic() - t0
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        # the claim is visible the moment claim() returns: the judgment row and the pending row both say so
        assert tuple(conn.execute("SELECT status, claimed_by_session FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()) == ("in_progress", "session-A")
        row = [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid][0]
        assert row["claim_session"] == "session-A" and row["claim_utc"] == synth.t(1000)
        # and NOTHING else was folded: the backlog is still spooled, the run table did not grow
        assert _delivered_backlog(conn) == 0
        assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE 'records/claims/%'").fetchone()[0] == 1
        # the claim file's arrival fact is written and folded by the claim itself, so a rebuild before any cycle agrees
        assert conn.execute("SELECT COUNT(*) FROM file_arrival WHERE file_uri LIKE 'records/claims/%'").fetchone()[0] == 1
    finally:
        conn.close()
    assert _stamp(root) == stamp_before                      # a claim is not a harvester cycle: the stamp the status page ages is untouched
    held, folded = _hold_seconds(caplog, "claim")
    assert folded == 1 and held < 1.0 and wall < 5.0, (held, wall)
    # the harvester's own cycle then folds the backlog, as before
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert _delivered_backlog(conn) == N_BACKLOG and conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2 + N_BACKLOG
        assert conn.execute("SELECT COUNT(*) FROM file_arrival WHERE file_uri LIKE 'records/claims/%'").fetchone()[0] == 1   # still exactly one
    finally:
        conn.close()
    assert _stamp(root) != stamp_before


def test_release_with_a_spool_backlog_folds_only_its_own_record(tmp_path, caplog):
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][0]
    jid = P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1000))
    _spool_backlog(root)
    stamp_before = _stamp(root)
    caplog.set_level(logging.INFO, logger="sisteer.registry.pending")
    assert P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1001))
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "abandoned"
        assert [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid][0]["claim_session"] is None
        assert _delivered_backlog(conn) == 0 and conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2
    finally:
        conn.close()
    assert _stamp(root) == stamp_before
    held, folded = _hold_seconds(caplog, "release")
    assert folded == 1 and held < 1.0
    # a release with nothing to release folds nothing and still reports its hold
    assert not P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1002))
    assert _hold_seconds(caplog, "release")[1] == 0


def test_two_sequential_claimants_the_second_sees_in_progress(tmp_path):
    """The semantics the scoped fold must keep: claimant B, arriving after A returned, is refused on A's folded claim; after A's
    release B claims; B's claim is visible to a third reader at once."""
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][1]
    _spool_backlog(root, n=20)
    jid_a = P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(2000))
    with pytest.raises(P.ClaimHeld) as ei:
        P.claim(root, rid, synth.JUDGE_ID_2, "session-B", now=synth.t(2001))
    assert ei.value.session == "session-A" and ei.value.run_id == rid
    assert P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(2002)) == jid_a     # the same session gets its own id back
    assert P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(2003))
    jid_b = P.claim(root, rid, synth.JUDGE_ID_2, "session-B", now=synth.t(2004))
    assert jid_b != jid_a
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid_a,)).fetchone()[0] == "abandoned"
        assert tuple(conn.execute("SELECT status, claimed_by_session FROM judgment WHERE judgment_id = ?", (jid_b,)).fetchone()) == ("in_progress", "session-B")
        assert [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid][0]["claim_session"] == "session-B"
        assert _delivered_backlog(conn) == 0                 # four claims and a release later the backlog is still the harvester's
    finally:
        conn.close()
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID_2, "session-C", now=synth.t(2005))


def test_ingest_now_is_still_the_full_cycle(tmp_path):
    root, _ids = _setup(tmp_path)
    _spool_backlog(root, n=30)
    stamp_before = _stamp(root)
    rep = P.ingest_now(root)
    # the 30 backlog files plus the cycle's own arrival-facts file, which a full cycle writes and folds
    assert rep.refused == 0 and rep.files == 31 and rep.arrival_facts == 30 and rep.completed_utc is not None
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert _delivered_backlog(conn) == 30
    finally:
        conn.close()
    assert _stamp(root) != stamp_before                      # the explicit full fold stamps the cycle, as it always did


def test_only_files_folds_the_named_plain_files_and_refuses_the_rest(tmp_path):
    """`harvest.ingest_once(only_files=...)` directly: the named file folds (ranked, delivered, its rows live) and the backlog
    beside it does not; a file already delivered under the same bytes is not new work; a path outside the root, a non-file and a
    revision directory's file are refused by name; the cycle stamp is never written."""
    root, _ids = _setup(tmp_path)
    backlog = _spool_backlog(root, n=5)
    stamp_before = _stamp(root)
    conn = DB.open_or_create(DB.db_path(root))
    try:
        rep = H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=[backlog[2]])
        # the named file plus its own arrival-facts file, nothing else; no cycle completion
        assert rep.files == 2 and rep.arrival_facts == 1 and rep.ok > 0 and rep.refused == 0 and rep.completed_utc is None
        assert _delivered_backlog(conn) == 1
        assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM file_arrival WHERE file_uri LIKE 'records/backlog/%'").fetchone()[0] == 1
        # relative to the root works too, and a file already folded under these bytes is skipped
        rel = os.path.relpath(backlog[2], root)
        rep2 = H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=[rel, backlog[0]])
        assert rep2.files == 2 and rep2.arrival_facts == 1 and _delivered_backlog(conn) == 2
        assert H.fold_only_files(root, conn, [backlog[0]], H.IngestReport(), synth.t(1)) == []
        with pytest.raises(ValueError, match="not under the registry root"):
            H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=[str(tmp_path / "elsewhere.jsonl")])
        with pytest.raises(ValueError, match="not a record file"):
            H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=[os.path.join(root, "records", "backlog", "missing.jsonl")])
        rev = os.path.join(root, "records", "adapters", "synthetic", "a" * 16, "000001-run.jsonl")
        os.makedirs(os.path.dirname(rev), exist_ok=True)
        with open(rev, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"record_type": "run"}) + "\n")
        with pytest.raises(ValueError, match="revision directory"):
            H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=[rev])
        assert _delivered_backlog(conn) == 2
    finally:
        conn.close()
    assert _stamp(root) == stamp_before


# ---- the claim must have FOLDED: refused by name otherwise, never completed by a full cycle --------------------


def _copy_log(root, dest):
    import shutil
    for f in R.iter_record_files(os.path.join(root, "records")):
        rel = os.path.relpath(f, os.path.join(root, "records"))
        dest_f = os.path.join(dest, "records", rel)
        os.makedirs(os.path.dirname(dest_f), exist_ok=True)
        shutil.copy2(f, dest_f)


def _claim_files(root, rid):
    d = os.path.join(root, "records", "claims")
    return sorted(os.path.relpath(os.path.join(dp, f), d) for dp, _ds, fs in os.walk(d) for f in fs if rid.split(":")[-1] in dp)


def test_a_claim_on_a_run_still_in_the_spool_is_refused_by_name_with_nothing_written(tmp_path):
    """The registry owner's rule (2026-10-03): a claim that cannot fold is never written. The run's own record is in the spool,
    unfolded -> the pre-check refuses by name BEFORE any record exists: no claim file, no judgment row, no blocked record, the cycle
    stamp untouched; a second claimant is refused the same way; once the harvester folds the run a claim succeeds and is visible at
    once, and the next claimant sees it."""
    root, _ids = _setup(tmp_path)
    rid, recs = synth.run_records("spooled", 7000, through="generated", n_items=2)
    _write(root, "spool", "000001-run.jsonl", recs)                      # the run's own record: written, not folded
    stamp_before = _stamp(root)
    before = sorted(R.iter_record_files(os.path.join(root, "records")))
    for session in ("session-A", "session-B"):
        with pytest.raises(P.ClaimNotFolded) as ei:
            P.claim(root, rid, synth.JUDGE_ID_2, session, now=synth.t(3000))
        assert ei.value.run_id == rid and "not folded yet; retry after the harvester folds it" in str(ei.value) and "nothing written" in str(ei.value)
    assert sorted(R.iter_record_files(os.path.join(root, "records"))) == before          # zero records written
    assert _claim_files(root, rid) == []
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    finally:
        conn.close()
    assert _stamp(root) == stamp_before
    # the harvester folds the run; a claim now folds and is visible at once; a second claimant sees it (the semantics kept)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    jid = P.claim(root, rid, synth.JUDGE_ID_2, "session-C", now=synth.t(3002))
    with pytest.raises(P.ClaimHeld):
        P.claim(root, rid, synth.JUDGE_ID_2, "session-D", now=synth.t(3003))
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert tuple(conn.execute("SELECT status, claimed_by_session FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()) == ("in_progress", "session-C")
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 1       # no refused attempt left a row
        row = [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid][0]
        assert row["claim_session"] == "session-C"
    finally:
        conn.close()
    assert len(_claim_files(root, rid)) == 1
    # the other parent too: a judge with no row refuses before writing
    with pytest.raises(P.ClaimNotFolded, match="has no row"):
        P.claim(root, rid, "0" * 64, "session-E", now=synth.t(3004))
    assert len(_claim_files(root, rid)) == 1


def test_release_checks_its_own_fold_too_and_the_cli_exits_non_zero(tmp_path, monkeypatch, capsys):
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][0]
    P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1000))
    # the check itself: a report that blocked the record, or a row that does not read what was written, refuses by name
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        jid = conn.execute("SELECT judgment_id FROM judgment WHERE run_id = ? AND status = 'in_progress'", (rid,)).fetchone()[0]
        blocked = H.IngestReport(); blocked.blocked = 1
        with pytest.raises(P.ClaimNotFolded, match="release"):
            P._require_folded(root, conn, rid, jid, "abandoned", blocked, "release")
        with pytest.raises(P.ClaimNotFolded, match="wanted 'abandoned'"):
            P._require_folded(root, conn, rid, jid, "abandoned", H.IngestReport(), "release")
        P._require_folded(root, conn, rid, jid, "in_progress", H.IngestReport(), "claim")      # what claim verified
    finally:
        conn.close()
    assert P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1001))              # the real release folds and verifies
    # the CLI: a refused claim or release is exit 1 with the reason on stderr
    from sisteer.registry import cli
    monkeypatch.setattr(P, "claim", lambda *a, **k: (_ for _ in ()).throw(P.ClaimNotFolded(rid, "claim: test")))
    monkeypatch.setattr(P, "release", lambda *a, **k: (_ for _ in ()).throw(P.ClaimNotFolded(rid, "release: test")))
    rc = cli.main(["--root", root, "pending-judgment", "--claim", "--run-id", rid, "--judge", synth.JUDGE_ID_2, "--session", "s"])
    err = capsys.readouterr().err
    assert rc == 1 and err.startswith("claim refused: ClaimNotFolded: ") and "not folded yet" in err
    rc = cli.main(["--root", root, "pending-judgment", "--release", rid, "--judge", synth.JUDGE_ID_2, "--session", "s"])
    err = capsys.readouterr().err
    assert rc == 1 and err.startswith("release refused: ClaimNotFolded: ") and "not folded yet" in err


def test_rebuild_agrees_immediately_after_a_claim_before_any_full_cycle(tmp_path):
    """The scoped fold ranks the claim file at a live arrival and writes its arrival fact itself, so a rebuild of the log before the
    next harvester cycle gives the same ledger order and the same ingest_seq on every seq-bearing row."""
    root, ids = _setup(tmp_path)
    rid = ids["run_ids"][0]
    P.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1000))
    assert P.release(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1001))
    P.claim(root, rid, synth.JUDGE_ID_2, "session-B", now=synth.t(1002))
    copy = str(tmp_path / "copy")
    _copy_log(root, copy)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        ledger = lambda c: sorted(tuple(r) for r in c.execute(f"SELECT {H.LEDGER_ORDER_SQL}, n_records, ingest_seq FROM record_ingest"))
        assert ledger(a) == ledger(b)
        for t in H.SEQ_TABLES:
            assert sorted(tuple(r) for r in a.execute(f"SELECT * FROM {t}")) == sorted(tuple(r) for r in b.execute(f"SELECT * FROM {t}")), t
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, diff
        assert a.execute("SELECT COUNT(*) FROM file_arrival WHERE file_uri LIKE 'records/claims/%'").fetchone()[0] == 3
    finally:
        a.close()
        b.close()
