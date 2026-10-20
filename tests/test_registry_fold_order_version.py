"""The fold order has a VERSION, recorded in the database and checked by every writer (registry-fold-order-version, 2026-10-09).

Two same-stamp fold orders wrote into the live database between 2026-10-04 and 2026-10-08 (the harvester's base checkout two days behind
the tie-break commit, a judgment daemon on an older worktree), and nothing in the rows said which. Now `reducer.FOLD_ORDER_VERSION` is
stamped as the `schema_migration` row `version = 0` (outside the migration sequence, invisible to `MAX(version)`) by the first writer and
checked by every writer before it folds (`db.require_fold_order`): equal -> proceed; absent -> stamp and proceed (the bootstrap, never a
refusal of the live database); another -> `db.FoldOrderDrift`, a `SchemaDrift` by type, naming both orders, nothing written. Readers never
check. The two apply sorts in `harvest` spell the fold's order through `Validated.order_key`, and `replica.residual_same_stamp_class` lists
the rows the pre-tie-break order left in `judged` where the current order gives `superseded`. Synthetic registries under tmp only."""
from __future__ import annotations

import inspect
import json
import os
import random
import sys
from pathlib import Path

import pytest

from sisteer.registry import cli as CLI
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import migrations as M
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import reducer as RD
from sisteer.registry import replica as RP
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V

REPO = Path(__file__).resolve().parents[1]
OLD_ORDER = RD.FOLD_ORDERS[1]
PRE_TIE_BREAK_SQL = ("written_utc, record_sha256", "written_utc DESC, record_sha256 DESC")


# ---- helpers ------------------------------------------------------------------------------------------------------------------------
def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root, **kw):
    kw.setdefault("workers", 0)
    kw.setdefault("min_free", 0)
    kw.setdefault("derive_facts", False)
    return H.ingest_once(root, **kw)


def _q(path, sql, *params):
    with DB.reading(path) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _seed_and_run(root, key="k", i0=100, through="generated"):
    """A seeded run whose own lifecycle events sit at t(i0)…t(i0+2); keep i0 well below the stamps a test folds after it (t(300))."""
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records(key, i0, through=through)
    _write(root, "c", f"000001-run-{key}.jsonl", recs)
    return rid


def _pair(rid: str, stamp: str):
    """A `judged` and a `superseded` run_state at ONE written_utc whose shas put `judged` LAST under the pre-tie-break order, so the two orders
    disagree on the pair instead of agreeing by the sha draw."""
    judged = synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": stamp, "evidence_kind": "completion_artifact"}, stamp)
    for n in range(512):
        sup = synth.rec("run_state", {"run_id": rid, "state": "superseded", "event_utc": stamp, "evidence_kind": "ruling",
                                      "note": f"superseded by a later run of the same key ({n})"}, stamp)
        if R.record_sha(sup) < R.record_sha(judged):
            return judged, sup
    raise AssertionError("no adversarial note in 512 tries")


def _judgment(rid: str, stamp: str):
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    return jid, synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": stamp, "finished_utc": stamp,
                                       "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, stamp)


def _stamp_row(path):
    rows = _q(path, "SELECT version, applied_utc, description FROM schema_migration WHERE version = ?", DB.FOLD_ORDER_ROW_VERSION)
    return rows[0] if rows else None


def _set_recorded_order(path, order):
    conn = DB.connect(path)
    try:
        conn.execute("UPDATE schema_migration SET description = ? WHERE version = ?", (order, DB.FOLD_ORDER_ROW_VERSION))
    finally:
        conn.close()


def _unstamp(path):
    """A database from before the stamp: the row removed (what every live database looks like the moment the code lands)."""
    conn = DB.connect(path)
    try:
        conn.execute("DELETE FROM schema_migration WHERE version = ?", (DB.FOLD_ORDER_ROW_VERSION,))
    finally:
        conn.close()
    assert _stamp_row(path) is None


def _fold_pair_under(root, rid, stamp, order_sql, monkeypatch, name="000002-pair.jsonl"):
    """Fold the pair (with its judgment) into ``root`` under the reducer order ``order_sql`` -- the pre-tie-break pair forces the legacy
    `judged` outcome into the rows, the way the live harvester did before its checkout took the tie-break."""
    judged, sup = _pair(rid, stamp)
    _, judgment = _judgment(rid, stamp)
    with monkeypatch.context() as mp:
        mp.setattr(RD, "ORDER_SQL", order_sql[0])
        mp.setattr(RD, "ORDER_DESC_SQL", order_sql[1])
        _write(root, "c", name, [judgment, sup, judged])         # the lateral event FIRST in the file: arrival order must not matter
        rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _q(DB.db_path(root), "SELECT COUNT(*) FROM blocked_records") == [(0,)], "a judged blocked on its judgment inside the cycle is unblocked by its end"
    return judged, sup


# ---- (a) the tie folds `superseded` on both fold paths; the residual class reads the legacy rows and only them -------------------------
def test_a_same_stamp_tie_folds_superseded_incrementally_and_in_a_rebuild_and_the_residual_class_is_empty(tmp_path, monkeypatch):
    root = _root(tmp_path)
    rid = _seed_and_run(root)
    _ingest(root)
    live = DB.db_path(root)
    stamp = synth.t(300)
    judged, sup = _fold_pair_under(root, rid, stamp, (RD.ORDER_SQL, RD.ORDER_DESC_SQL), monkeypatch)
    # the incremental path: superseded, the stamp of the lateral record, the judged evidence kept
    assert _q(live, "SELECT lifecycle_state, judged_utc FROM run WHERE run_id = ?", rid) == [("superseded", stamp)]
    fv = json.loads(_q(live, "SELECT field_versions_json FROM run WHERE run_id = ?", rid)[0][0])
    assert fv["lifecycle_state"] == [stamp, R.record_sha(sup)]
    with DB.reading(live) as conn:
        assert RP.residual_same_stamp_class(conn) == []
    # the rebuild: the same row, the same empty class, every table equal, and both files stamped with the code's order
    rebuilt = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, rebuilt, workers=0)
    assert _q(rebuilt, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("superseded",)]
    with DB.reading(rebuilt) as conn:
        assert RP.residual_same_stamp_class(conn) == []
    with DB.reading(live) as a, DB.reading(rebuilt) as b:
        diff = H.verify_equal(a, b)
    assert not {k: v for k, v in diff.items() if v}, diff
    for path in (live, rebuilt):
        row = _stamp_row(path)
        assert row is not None and row[2] == RD.FOLD_ORDER_VERSION and row[1].endswith("Z"), (path, row)


def test_a_row_the_old_order_left_in_judged_is_the_residual_class_and_a_refold_clears_it(tmp_path, monkeypatch, capsys):
    """The legacy state forced into the rows (the pre-tie-break ORDER_SQL, as the live harvester folded until 2026-10-06): the row reads
    `judged`, the helper names exactly it -- not a run whose `superseded` came at a LATER stamp (folds last under either order), not a run
    that is plainly judged, not a run the current order already folded -- and the registry's own refold writer moves it out of the class."""
    root = _root(tmp_path)
    rid_legacy = _seed_and_run(root, key="legacy", i0=100)
    rid_later = _seed_and_run(root, key="later", i0=110)
    rid_plain = _seed_and_run(root, key="plain", i0=120)
    rid_current = _seed_and_run(root, key="current", i0=130)
    _ingest(root)
    live = DB.db_path(root)
    stamp = synth.t(300)
    # the legacy pair, folded under the old order -> judged (the sha draw)
    _fold_pair_under(root, rid_legacy, stamp, PRE_TIE_BREAK_SQL, monkeypatch, name="000002-legacy.jsonl")
    assert _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid_legacy) == [("judged",)], "the old order's draw"
    # a judged run whose superseded event came one stamp LATER (folds last under both orders)
    judged_l, _ = _pair(rid_later, stamp)
    _, judgment_l = _judgment(rid_later, stamp)
    later_sup = synth.rec("run_state", {"run_id": rid_later, "state": "superseded", "event_utc": synth.t(301), "evidence_kind": "ruling"}, synth.t(301))
    _write(root, "c", "000003-later.jsonl", [judgment_l, judged_l, later_sup])
    # a plainly judged run
    _, judgment_p = _judgment(rid_plain, stamp)
    judged_p = synth.rec("run_state", {"run_id": rid_plain, "state": "judged", "event_utc": stamp, "evidence_kind": "completion_artifact"}, stamp)
    _write(root, "c", "000004-plain.jsonl", [judgment_p, judged_p])
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _q(DB.db_path(root), "SELECT COUNT(*) FROM blocked_records") == [(0,)], "a judged blocked on its judgment inside the cycle is unblocked by its end"
    # a pair the CURRENT order folded
    _fold_pair_under(root, rid_current, stamp, (RD.ORDER_SQL, RD.ORDER_DESC_SQL), monkeypatch, name="000005-current.jsonl")
    assert _q(live, "SELECT run_id, lifecycle_state FROM run ORDER BY run_id") == sorted(
        [(rid_legacy, "judged"), (rid_later, "superseded"), (rid_plain, "judged"), (rid_current, "superseded")])
    with DB.reading(live) as conn:
        assert RP.residual_same_stamp_class(conn) == [rid_legacy]
    # the refold writer (what `replica refold-apply` / the window's restamp slot run) moves the row under the current order
    key = I.canonical_json({"run_id": rid_legacy})
    conn = DB.connect(live)
    try:
        assert RP.refold_apply(conn, [key]) == 1
        assert RP.residual_same_stamp_class(conn) == []
    finally:
        conn.close()
    assert _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid_legacy) == [("superseded",)]
    # the CLI listing, read-only: the ids, the count, both orders -- and a reader lists a database of ANY recorded order without stamping
    _set_recorded_order(live, OLD_ORDER)
    assert CLI.main(["--root", root, "replica", "residual-class", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d == {"db": os.path.abspath(live), "n": 0, "run_ids": [], "fold_order_recorded": OLD_ORDER, "fold_order_code": RD.FOLD_ORDER_VERSION}
    assert _stamp_row(live)[2] == OLD_ORDER, "a reader never stamps"


def test_residual_class_cli_lists_the_legacy_rows_one_per_line(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    rid = _seed_and_run(root)
    _ingest(root)
    _fold_pair_under(root, rid, synth.t(300), PRE_TIE_BREAK_SQL, monkeypatch)
    live = DB.db_path(root)
    assert CLI.main(["replica", "residual-class", "--db", live]) == 0
    cap = capsys.readouterr()
    assert cap.out.splitlines() == [rid] and "1 run(s) in the residual same-stamp class" in cap.err and RD.FOLD_ORDER_VERSION in cap.err
    assert CLI.main(["replica", "residual-class", "--db", str(tmp_path / "absent.sqlite")]) == RP.RefoldInputRefused.exit_code
    assert "no database at" in capsys.readouterr().err


# ---- (b) the stamp: fresh and bootstrap; the refusal by name in every writer path; readers untouched ---------------------------------
def test_a_fresh_database_is_stamped_at_creation_and_an_unstamped_one_at_its_first_write(tmp_path):
    fresh = str(tmp_path / "fresh.sqlite")
    conn = DB.open_fresh(fresh)
    conn.close()
    row = _stamp_row(fresh)
    assert row is not None and row[0] == 0 and row[2] == RD.FOLD_ORDER_VERSION and row[1].endswith("Z")
    assert _q(fresh, "SELECT MAX(version) FROM schema_migration") == [(M.code_version(),)], "the stamp is invisible to the schema version"
    with DB.reading(fresh) as ro:
        assert M.db_version(ro) == M.code_version() and DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION
        DB.require_shape(ro, fresh)                                              # a reader's check passes a stamped database
    # the bootstrap: a database from before the stamp records none; a reader's check leaves it so; the first WRITER stamps and proceeds
    root = _root(tmp_path)
    rid = _seed_and_run(root)
    _ingest(root)
    live = DB.db_path(root)
    _unstamp(live)
    with DB.reading(live) as ro:
        assert DB.fold_order_version(ro) is None
        DB.require_shape(ro, live)                                               # no stamp is no finding for a reader
        assert DB.require_fold_order(ro, live, stamp=False) is None
    assert _stamp_row(live) is None, "a reader never stamps"
    before = R.utc_now()
    rep = _ingest(root)                                                          # the writers' open stamps, then folds as before
    assert not rep.refusals
    row = _stamp_row(live)
    assert row is not None and row[2] == RD.FOLD_ORDER_VERSION and row[1] >= before
    assert _q(live, "SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("generated",)]
    # idempotent: a second writer open leaves the stamp (its applied_utc included) as it is
    conn = DB.open_or_create(live)
    conn.close()
    assert _stamp_row(live) == row
    # the other writers' bootstrap: fold_only_files and a passed-in connection of ingest_once stamp an unstamped database too
    _unstamp(live)
    conn = DB.connect(live)
    try:
        path = _write(root, "c", "000002-later.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_planned": 7}}, synth.t(500))])
        assert H.fold_only_files(root, conn, [path], H.IngestReport(), synth.t(600)) == [H.logical_uri(root, path)]
    finally:
        conn.close()
    assert _stamp_row(live) is not None and _stamp_row(live)[2] == RD.FOLD_ORDER_VERSION
    _unstamp(live)
    conn = DB.connect(live)
    try:
        assert not H.ingest_once(root, conn, workers=0, min_free=0, derive_facts=False).refusals
    finally:
        conn.close()
    assert _stamp_row(live)[2] == RD.FOLD_ORDER_VERSION
    # the refold writer's bootstrap, inside its own transaction
    _unstamp(live)
    conn = DB.connect(live)
    try:
        RP.refold_apply(conn, [I.canonical_json({"run_id": rid})])
    finally:
        conn.close()
    assert _stamp_row(live)[2] == RD.FOLD_ORDER_VERSION


def _expect_refused(fn, live, before_rows):
    with pytest.raises(DB.FoldOrderDrift) as ei:
        fn()
    exc = ei.value
    assert isinstance(exc, DB.SchemaDrift), "a SchemaDrift by type: every writer's refusal path treats it as the drift it is"
    msg = str(exc)
    assert OLD_ORDER in msg and RD.FOLD_ORDER_VERSION in msg and "nothing is written" in msg, msg
    assert live in msg or os.path.realpath(live) in msg, msg                    # the database named (a connection's own path resolves symlinks)
    assert exc.have_fold_order == OLD_ORDER and exc.want_fold_order == RD.FOLD_ORDER_VERSION
    assert exc.have == exc.want == M.code_version() and len(exc.lines) == 1 and OLD_ORDER in exc.lines[0]
    assert "registry rebuild" in msg and "fast-forward" in msg, "both remedies named: the older code, or the older database"
    assert _q(live, "SELECT run_id, lifecycle_state, field_versions_json FROM run ORDER BY run_id") == before_rows, "nothing written"
    assert _stamp_row(live)[2] == OLD_ORDER, "the stamp is never rewritten by a refusal"
    return exc


def test_a_database_recording_another_order_is_refused_by_name_in_every_writer_path_and_opens_for_every_reader(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    rid = _seed_and_run(root)
    _ingest(root)
    live = DB.db_path(root)
    _set_recorded_order(live, OLD_ORDER)
    before = _q(live, "SELECT run_id, lifecycle_state, field_versions_json FROM run ORDER BY run_id")
    deliveries = _q(live, "SELECT COUNT(*) FROM record_delivery")
    new_file = _write(root, "c", "000002-patch.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_planned": 7}}, synth.t(500))])
    key = I.canonical_json({"run_id": rid})
    # (1) the writers' open
    _expect_refused(lambda: DB.open_or_create(live), live, before)
    # (2) a harvester cycle on its own connection, and the CLI's exit 3 with the message
    _expect_refused(lambda: _ingest(root), live, before)
    assert CLI.main(["--root", root, "ingest", "--workers", "0"]) == 3
    err = capsys.readouterr().err
    assert err.startswith("refused: ") and OLD_ORDER in err and RD.FOLD_ORDER_VERSION in err
    # (3) fold_only_files and (4) ingest_once on a connection the caller opened
    conn = DB.connect(live)
    try:
        _expect_refused(lambda: H.fold_only_files(root, conn, [new_file], H.IngestReport(), synth.t(600)), live, before)
        _expect_refused(lambda: H.ingest_once(root, conn, workers=0, min_free=0, derive_facts=False), live, before)
        _expect_refused(lambda: H.ingest_once(root, conn, workers=0, min_free=0, derive_facts=False, only_files=[new_file]), live, before)
        # (5) the replica's refold writer: the apply, the restamp plan (its dry run) and the apply bound to a plan
        _expect_refused(lambda: RP.refold_apply(conn, [key]), live, before)
        _expect_refused(lambda: RP.restamp_keys_plan(conn, [rid]), live, before)
        _expect_refused(lambda: RP.restamp_keys_apply(conn, {"ok": True, "keys": [key], "changes": [], "changes_sha256": "0" * 64}), live, before)
    finally:
        conn.close()
    assert _q(live, "SELECT COUNT(*) FROM record_delivery") == deliveries, "no file was delivered"
    # (6) a claim's fold (pending.claim opens the writers' way under the write lock)
    _expect_refused(lambda: P.claim(root, rid, synth.JUDGE_ID, "session-A", now=synth.t(700)), live, before)
    assert _q(live, "SELECT COUNT(*) FROM judgment WHERE run_id = ?", rid) == [(0,)]
    # (7) the replica CLI: refold-apply refused by its own name, exit 12
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([key]), encoding="utf-8")
    assert CLI.main(["replica", "refold-apply", "--db", live, "--keys", str(keys)]) == RP.SchemaBehind.exit_code
    err = capsys.readouterr().err
    assert err.startswith("replica refold-apply REFUSED: registry database at") and OLD_ORDER in err and "replica migrate" not in err
    # readers: untouched, and never a stamp
    with DB.reading(live) as ro:
        assert DB.fold_order_version(ro) == OLD_ORDER
        assert H.history(root, ro, "run", {"run_id": rid})[-1]["state"]["lifecycle_state"] == "generated"
        assert [d["key"] for d in RP.refold_dump(ro)] == [key]
        assert M.shape_plan(ro)["drift"] == [] and M.render_shape_migration(ro) == ""
        with pytest.raises(DB.FoldOrderDrift):
            DB.require_shape(ro, live)                      # the reader's check names the drift (the daemon's preflight) without writing
    assert CLI.main(["--root", root, "query", "SELECT COUNT(*) FROM run"]) == 0
    # the status page (a reader): rendered, and its banner carries the fold-order line as a drift line; the stamp untouched
    from sisteer.registry import status as ST
    page, _timings = ST.render(root, out_path=str(tmp_path / "status.html"), register=False)
    html = Path(page).read_text(encoding="utf-8")
    assert "fold order: the database records" in html and OLD_ORDER in html and RD.FOLD_ORDER_VERSION in html
    assert CLI.main(["migrate", "--plan", "--db", live]) == 0
    out = capsys.readouterr().out
    assert "shape: matches the code's schema" in out and "pending: none" in out
    assert _stamp_row(live)[2] == OLD_ORDER
    # the grade daemon: a judging process's terminal line is held as schema drift, the preflight reads the gap by kind, an unstamped database is no gap
    sys.path.insert(0, str(REPO / "scripts" / "registry"))
    import grade_daemon as GD  # noqa: E402
    term = f"sisteer.registry.db.FoldOrderDrift: {DB.fold_order_message(live, OLD_ORDER, RD.FOLD_ORDER_VERSION)}"
    assert GD.dispatch_failure_kind(1, "Traceback (most recent call last):\n  ...\n" + term + "\n") == "schema_drift"
    gap = GD.schema_gap(root)
    assert gap is not None and gap["kind"] == "fold_order" and gap["n_drift"] == 1 and OLD_ORDER in gap["message"] and RD.FOLD_ORDER_VERSION in gap["message"]
    assert "refold" in GD.schema_hold_reason(gap) and OLD_ORDER in GD.schema_hold_reason(gap)
    _unstamp(live)
    assert GD.schema_gap(root) is None and _stamp_row(live) is None, "a database recording no order is no gap, and the preflight stamps nothing"


def test_replica_ingest_head_refuses_a_fold_order_drift_by_its_own_name_not_as_a_schema_behind(tmp_path):
    """`ingest_head` wraps a SchemaDrift as SchemaBehind ("replica migrate first"); a fold-order drift is not a migration's to fix and keeps its
    own name and remedy."""
    root = _root(tmp_path)
    _seed_and_run(root)
    _ingest(root)
    head = RP.take_head(DB.db_path(root), wait_min=0)
    lay = RP.init_layout(str(tmp_path / "rep"), root)
    work = str(tmp_path / "work.sqlite")
    RP._copy_db(DB.db_path(root), work)
    _set_recorded_order(work, OLD_ORDER)
    with pytest.raises(DB.FoldOrderDrift) as ei:
        RP.ingest_head(work, lay, head, workers=0)
    assert not isinstance(ei.value, RP.SchemaBehind) and OLD_ORDER in str(ei.value)
    _set_recorded_order(work, RD.FOLD_ORDER_VERSION)
    assert RP.ingest_head(work, lay, head, workers=0)["delta_files"] == 0        # equal: nothing to fold, and no refusal


# ---- (c) the two apply sorts are the fold's order ------------------------------------------------------------------------------------
def _validated(record_type, written_utc, sha, payload):
    return VAL.Validated(record_type, "id", written_utc, {}, dict(payload), {}, None, sha, "u")


def test_the_apply_sorts_order_a_shuffled_transaction_exactly_as_order_key_does():
    rng = random.Random(7)
    vals = []
    for i in range(300):
        stamp = synth.t(rng.randrange(12))
        sha = R.record_sha({"i": i})
        rt = rng.choice(("run_state", "run_state", "run_ingest", "run_patch", "run", "judgment", "source_revision"))
        payload = {"state": rng.choice(V.LIFECYCLE_STATES)} if rt == "run_state" else {"state": "ingested"} if rt == "run_ingest" else {"i": i}
        vals.append(_validated(rt, stamp, sha, payload))
    # the fold's own order over the same records as history events
    def as_event(v):
        return RD.HistoryEvent(v.written_utc, v.sha, v.record_type, v.uri, dict(v.payload))
    by_fold = [v.sha for v in sorted(vals, key=lambda v: RD.order_key(as_event(v)))]
    by_apply = [v.sha for v in sorted(vals, key=H.apply_order) if v.record_type != "source_revision"]
    markers = [v.sha for v in sorted(vals, key=H.apply_order) if v.record_type == "source_revision"]
    assert by_apply == [s for s in by_fold if s not in set(markers)], "the apply order IS order_key, a revision's marker ahead of the rest"
    assert [v.sha for v in sorted(vals, key=H.apply_order)][:len(markers)] == markers
    assert by_apply != [v.sha for v in sorted((v for v in vals if v.record_type != "source_revision"), key=lambda v: (v.written_utc, v.sha))], \
        "the rank decides somewhere: the pre-tie-break spelling is a different order"
    for v in vals:
        assert H.apply_order(v)[1] == v.order_key == (v.written_utc, RD.tie_rank(v.record_type, v.payload), v.sha)
    # the two sites spell the rule THROUGH apply_order and nowhere as (written_utc, sha)
    for fn in (H.Ingestor._ingest_chunk, H.Ingestor._activate_revision):      # the chunk path and a revision's pass A
        src = inspect.getsource(fn)
        assert "apply_order(" in src, fn.__name__
        assert "written_utc, pr.sha" not in src and "written_utc, v.sha" not in src, fn.__name__
    src = inspect.getsource(H.apply_order)
    assert "v.order_key" in src


# ---- (d) the plan step: a metadata row is not a shape change ---------------------------------------------------------------------------
def test_the_window_plan_step_sees_no_shape_change_in_a_stamped_database(tmp_path, capsys):
    path = str(tmp_path / "stamped.sqlite")
    DB.open_fresh(path).close()
    assert _stamp_row(path)[2] == RD.FOLD_ORDER_VERSION
    with DB.reading(path) as conn:
        plan = M.shape_plan(conn)
        assert plan["drift"] == [] and plan["rebuild"] == [] and plan["add_columns"] == [] and plan["create"] == [] and plan["undeclared_tables"] == []
        assert all(not o["recreate"] and not o["drop"] for o in plan["objects"].values())
        assert M.render_shape_migration(conn, plan) == "" and M.pending_migrations(conn) == [] and DB.schema_drift(conn) == []
    assert CLI.main(["migrate", "--plan", "--db", path]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"schema version {M.code_version()}; code {M.code_version()}; pending: none" and out[1] == "shape: matches the code's schema"
    # the stamp's VALUE is not the plan's to read either: another recorded order still plans clean (the writers refuse it, the plan does not)
    _set_recorded_order(path, OLD_ORDER)
    assert CLI.main(["migrate", "--plan", "--db", path]) == 0
    assert "shape: matches the code's schema" in capsys.readouterr().out
    # `registry migrate` (the apply) is a schema writer, not a fold writer: it neither refuses nor stamps
    _unstamp(path)
    assert CLI.main(["migrate", "--db", path]) == 0
    assert "applied nothing" in capsys.readouterr().out and _stamp_row(path) is None


def test_the_version_strings_and_the_reserved_row_are_pinned():
    assert RD.FOLD_ORDER_VERSION == "fold_order/2: written_utc, tie_rank, record_sha256"
    assert RD.FOLD_ORDERS == {1: "fold_order/1: written_utc, record_sha256", 2: RD.FOLD_ORDER_VERSION}
    assert DB.FOLD_ORDER_ROW_VERSION == 0 and all(v >= 1 for v, _, _ in M.migration_files()), "0 is outside the migration sequence"
    assert "written_utc, tie_rank, record_sha256" in RD.ORDER_SQL.replace(RD.TIE_RANK_SQL, "tie_rank"), "the string names the live rule"
    md = (REPO / "docs" / "registry" / "records.md").read_text(encoding="utf-8")
    assert RD.FOLD_ORDER_VERSION in md and "`version = 0`" in md
