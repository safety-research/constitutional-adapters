"""A superseded run is never pending; its successor is (the registry ruling of 2026-10-04; migration 0025_pending_excludes_superseded).

Under the identity-current sibling rule, after an identity correction (instrument X to X2, the same judge) on a SUPERSEDED run
(`run.superseded_by_run_id` set) whose complete judgment had been claimed under X, `v_pending_judgment` listed the OLD run itself as
`identity_stale` -- a retired run claimable again beside its successor. Now the selection excludes a run with a successor outright
(`schema_def.run_superseded_sql`, one predicate), `pending.not_pending_why` names it through the same predicate
(`ClaimNotPending`: `superseded by run <id>; admits no judging`), and the identity_stale listing of a NON-superseded run is unchanged.
The migration recreates the one object: the plan on the shape one version below it is one recreate of that object and nothing else,
the committed body is the render, no Python post-step.
"""
from __future__ import annotations

import os
import shutil

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import identity as I
from sisteer.registry import migrations as M
from sisteer.registry import pending as P
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from test_registry_judgment_currency import INSTRUMENT_X2, INSTRUMENT_X2_ID, _complete, _generated, _ingest, _judgment_rows, _q, _rebuild_equal, _write
from test_registry_migrations import AF_TABLES, FU4_TABLES, NTPL_TABLES, TF_VIEWS, _sans_tf, _sans_tf_block  # noqa: E402  -- migration 0026's lines and blocks (0029's five creates, 0031's two rebuilds), set aside here
from test_registry_pending_claim_recheck import PREFIX, _claim_files, _record_files, _select, _why

SLUG = "pending_excludes_superseded"


def _correct(root, rid, key, i, **over):
    """A same-id `run` re-emission of ``rid`` (producer key ``key``) stating ``over`` -- an identity correction, a successor, or both."""
    corrected = synth.run_payload(key, i=100, **over)
    corrected["run_id"] = rid
    _write(root, "c", f"{i:06d}-correction-{key}.jsonl", [synth.rec("run", corrected, synth.t(i))])
    assert _ingest(root, synth.t(i + 50)).refused == 0


def _pending_runs_for(root, judge):
    return sorted(r["run_id"] for r in _q(root, "SELECT run_id FROM v_pending_judgment WHERE required_judge_id = ?", judge))


def test_a_superseded_run_corrected_after_its_judgment_is_never_pending_and_its_successor_is(tmp_path):
    root, r1 = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    r2, recs = synth.run_records("k2", 300, through="generated", n_items=4)                 # the successor (same store family, same eval)
    _write(root, "c", "000002-run.jsonl", recs)
    r3, recs3 = synth.run_records("k3", 400, through="generated", n_items=4)                # a NON-superseded control for the identity_stale listing
    _write(root, "c", "000003-run.jsonl", recs3)
    assert _ingest(root, synth.t(950)).refused == 0
    # J completes r1 and r3 under X
    for rid, i in ((r1, 1000), (r3, 1100)):
        a = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(i))
        _complete(root, rid, a, synth.JUDGE_ID, "s-A", i, i + 500)
    assert _ingest(root, synth.t(2500)).refused == 0
    assert _pending_runs_for(root, synth.JUDGE_ID) == [r2]
    # r1 is superseded by r2 (the re-registration / supersede shape: the old run names its successor) ...
    _correct(root, r1, "k", 3000, superseded_by_run_id=r2)
    assert _q(root, "SELECT superseded_by_run_id FROM run WHERE run_id = ?", r1) == [{"superseded_by_run_id": r2}]
    assert _pending_runs_for(root, synth.JUDGE_ID) == [r2]
    # ... and then corrected to X2 (same judge), as is r3. Before 0025 the stale-bound complete made BOTH read identity_stale; now only r3
    _correct(root, r1, "k", 4000, superseded_by_run_id=r2, instrument_id=INSTRUMENT_X2_ID)
    _correct(root, r3, "k3", 4100, instrument_id=INSTRUMENT_X2_ID)
    assert _q(root, "SELECT instrument_id, superseded_by_run_id FROM run WHERE run_id = ?", r1) == [{"instrument_id": INSTRUMENT_X2_ID, "superseded_by_run_id": r2}]
    assert _pending_runs_for(root, synth.JUDGE_ID) == sorted([r2, r3])
    assert _select(root, r1, synth.JUDGE_ID) is None                                        # the retired run is not listed for any judge
    assert _q(root, "SELECT COUNT(*) AS n FROM v_pending_judgment WHERE run_id = ?", r1) == [{"n": 0}]
    assert _select(root, r3, synth.JUDGE_ID)["pending_reason"] == "identity_stale"          # the non-superseded run reads as before
    assert _select(root, r2, synth.JUDGE_ID)["pending_reason"] == "no_judgment"
    # a claim of the retired run refuses by name, nothing written; its successor and the control still claim
    why = _why(root, r1, synth.JUDGE_ID)
    assert why == f"superseded by run {r2}; admits no judging"
    files_before, rows_before = _record_files(root), _judgment_rows(root, r1)
    with pytest.raises(P.ClaimNotPending, match=PREFIX) as ei:
        P.claim(root, r1, synth.JUDGE_ID, "s-B", now=synth.t(5000))
    assert isinstance(ei.value, P.ClaimNotFolded) and ei.value.why == why and str(ei.value).endswith("; re-select (nothing written)")
    assert _record_files(root) == files_before and _judgment_rows(root, r1) == rows_before
    with pytest.raises(P.ClaimNotPending, match="superseded by run"):                      # force changes nothing: the run is retired
        P.claim(root, r1, synth.JUDGE_ID, "s-B", force=True, stale_hours=0.0, now=synth.t(5001))
    b = P.claim(root, r2, synth.JUDGE_ID, "s-B", now=synth.t(5100))
    c = P.claim(root, r3, synth.JUDGE_ID, "s-C", now=synth.t(5200))
    rows = _judgment_rows(root, r2), _judgment_rows(root, r3)
    assert rows[0][b]["status"] == "in_progress" and rows[1][c]["status"] == "in_progress" and rows[1][c]["claimed_instrument_id"] == INSTRUMENT_X2_ID
    assert len(_claim_files(root, r1)) == 1                                                 # A's one claim file on r1; B wrote none
    # the parity diagnostic names the successor too (worklist_parity's reason for a unit whose run is not pending)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert P._not_pending_reason(conn, r1, synth.JUDGE_ID) == f"superseded by run {r2}; never pending (its successor is)"
        assert conn.execute(f"SELECT {S.run_superseded_sql('r')} FROM run r WHERE r.run_id = ?", (r1,)).fetchone()[0] == 1
        assert conn.execute(f"SELECT {S.run_superseded_sql('r')} FROM run r WHERE r.run_id = ?", (r2,)).fetchone()[0] == 0
    finally:
        conn.close()
    _rebuild_equal(root, tmp_path)


def test_the_predicate_is_the_selections_own_and_appears_once_in_the_rendered_object():
    sql = S.VIEW_SQL["v_pending_judgment"]
    assert sql.count("AND NOT (" + S.run_superseded_sql("r") + ")") == 1 and S.run_superseded_sql("r") == "r.superseded_by_run_id IS NOT NULL"


# --------------------------------------------------------------------------------------------
# the migration: one recreate (v_pending_judgment) on the shape one version below it; the committed body is the render; no post-step
# --------------------------------------------------------------------------------------------


def _pes_migration():
    return next(f for f in M.migration_files() if f[1] == SLUG)


def _shape_below_pes(path: str) -> None:
    """A database at the shape one version below the migration: every earlier file applied by the runner itself."""
    version = _pes_migration()[0]
    conn = DB.connect(path)
    for v, slug, mpath in M.migration_files():
        if v < version:
            M._apply_one(conn, v, slug, open(mpath, encoding="utf-8").read())
    conn.commit()
    conn.close()


PES_DRIFT = ["v_pending_judgment differs from the code's SQL"]


def test_the_migration_plan_on_the_shape_below_it_is_one_recreate_of_v_pending_judgment_and_the_committed_body_is_the_render(tmp_path, capsys):
    version, slug, mpath = _pes_migration()
    assert version == M.code_version() - 7, "the newest file but seven on this tree: 0026 (the twin fold), 0027 (registry follow-ups-4), 0028 (reference_model_never_selected), 0029 (analysis filing), 0030 (fold_order_barrier) and 0031 (non_template_positions: a rebuild of instrument and run) follow it (renumbered at landing if another branch's lands first)"
    assert os.path.basename(mpath) == f"{version:04d}_{SLUG}.sql"
    path = str(tmp_path / "below.sqlite")
    _shape_below_pes(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    assert _sans_tf(DB.schema_drift(conn)) == PES_DRIFT                                    # 0026's three lines aside (its own test pins them)
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == [] and plan["add_columns"] == [] and plan["undeclared_tables"] == [] and [c for c in plan["create"] if c not in AF_TABLES] == []   # 0027's run_contrast rebuild, 0031's two and 0029's five creates aside
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    assert [v for v in plan["objects"]["view"]["recreate"] if v not in TF_VIEWS] == ["v_pending_judgment"] and plan["objects"]["view"]["drop"] == []
    rendered = M.render_shape_migration(conn, plan)
    committed = open(mpath, encoding="utf-8").read()
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n")
    body = committed[committed.index("-- drift: "):]
    assert body.strip() == _sans_tf_block(rendered).strip()                                 # today's render carries 0026's three blocks too
    assert body.count("DROP VIEW IF EXISTS v_pending_judgment;") == 1 and body.count("\nCREATE VIEW v_pending_judgment AS") == 1
    assert "DROP VIEW IF EXISTS v_pending_judgment;\n" + S.VIEW_SQL["v_pending_judgment"].strip() in body
    for absent in ("__new", "INSERT INTO", "ALTER TABLE", "CREATE TABLE", "DROP TABLE", "CREATE INDEX", "CREATE TRIGGER"):
        assert absent not in body, absent
    assert SLUG not in M.POST_APPLY                                                        # no Python post-step: nothing is keyed by the slug
    # the object before the file is 0023's text (0024 did not touch it): no superseded conjunct; the lifecycle WHERE as it always read
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'v_pending_judgment'").fetchone()[0]
    assert S.run_superseded_sql("r") not in before and "superseded_by_run_id" not in before and "lifecycle_state IN ('generated', 'judging', 'judged', 'ingested')" in before
    conn.close()
    # the CLI's plan on a COPY: the pending file, one recreate line naming the one object, no rebuild / add-column / create / drop, exit 1
    root = str(tmp_path / "copyroot")
    os.makedirs(root)
    shutil.copy(path, DB.db_path(root))
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert f"schema version {version - 1}; code {M.code_version()}; pending: {version:04d}_{SLUG}, " in out     # 0026 follows it in the pending list
    recreate_line = next(ln for ln in out.splitlines() if ln.startswith("recreate view(s): "))
    assert "v_pending_judgment" in recreate_line and f"  {PES_DRIFT[0]}" in out            # one line for the views; 0026's three are named beside
    for absent in ("add column", "drop undeclared", "recreate index"):
        assert absent not in out, out
    assert [ln for ln in out.splitlines() if ln.startswith("recreate trigger")] == [], out   # 0030's two fence triggers ride 0031's run rebuild
    assert [ln for ln in out.splitlines() if ln.startswith("create table") and ln.split()[-1] not in AF_TABLES] == [], out   # 0029's five creates aside
    assert [ln for ln in out.splitlines() if ln.startswith("rebuild table ") and "run_contrast" not in ln and ln.split()[2].rstrip(":") not in NTPL_TABLES] == [], out   # 0027's and 0031's rebuild lines aside
    assert sum(1 for ln in out.splitlines() if ln.startswith("recreate ")) == 1                 # the views' line alone
    # the runner: the version row and the object, no note (no post-step), zero drift, the writers' open accepts it
    conn = DB.connect(path)
    notes = []
    later = list(range(version, M.code_version() + 1))
    assert [v for v, _, _ in M.pending_migrations(conn)] == later                          # 0025, 0026, 0027, 0028 and 0029 pending on the shape below
    assert M.apply_migrations(conn, notes) == later and notes == []                         # 0025, then 0026 (the twin fold), 0027, 0028 and 0029 (no post-step, any of them)
    assert DB.schema_drift(conn) == [] and M.db_version(conn) == M.code_version()        # the later files applied after it
    assert conn.execute("SELECT description FROM schema_migration WHERE version = ?", (version,)).fetchone()[0] == SLUG
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name = 'v_pending_judgment'").fetchone()[0] == S.VIEW_SQL["v_pending_judgment"].strip().rstrip(";")
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment").fetchone()[0] == 0        # selectable on the empty database
    conn.close()
    DB.open_or_create(path).close()
