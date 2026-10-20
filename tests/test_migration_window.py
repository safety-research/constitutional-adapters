"""A migration lands inside a BOUNDED WINDOW, never as a bare merge.

* the committed `0018_substrate_slurm.sql` IS the body `registry migrate --plan --sql` renders against the version-17 shape (derived,
  never hand-edited), it takes a v17 database with rows in the three tables past version 18 with the rows intact and zero drift;
* `registry migrate --plan` on a v17 database lists `0018_substrate_slurm` first among the pending files and the three rebuilds;
* `registry migrate` REFUSES by name (exit 4) while another process holds the registry's write lock -- never applies concurrently;
* the landing wrapper `scripts/registry/migrate_window.sh` refuses `plan`/`apply` while a writer unit is active (fake `systemctl`),
  lands the migration when they are stopped, writes the verdict, and its `merge-gate` refuses until the verdict says the code's
  version + shape match + integrity + unchanged row counts. The window lands ONE migration (`--expect` names it, and the plan refuses
  any other pending file), so the window tests run against the shape just below the NEWEST migration -- `WINDOW_EXPECT`, today the
  fold-observation migration (one table created with its one index; no column, no recreate, no rebuild; its Python post-step derives
  the table's rows from row_history) -- the way the landing window itself will; 0018 landed on 2026-09-30, 0019 on 2026-10-01, 0020
  on 2026-10-01, 0021 on 2026-10-02, 0022 (the member-pass rule: `v_run_current` alone) on 2026-10-02, 0023 (the
  record-refusal-fact migration: one table created, five columns added on three tables, seven
  `v_*` objects recreated, no rebuild) on 2026-10-04.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import migrations as M
from sisteer.registry import schema_def as S
from test_registry_schema import frozen_column_docs

REPO = Path(__file__).resolve().parents[1]
MIG = REPO / "src" / "sisteer" / "registry" / "migrations"
WINDOW = REPO / "scripts" / "registry" / "migrate_window.sh"
EXPECT = "0018_substrate_slurm"
#: the migration the landing window lands next: the newest file (its tests build the shape one version below it)
WINDOW_EXPECT = f"{M.code_version():04d}_{M.migration_files()[-1][1]}"
WINDOW_BELOW = M.code_version() - 1
#: what the newest migration does to the shape below it, as the window's plan stage reports it: the tables it REBUILDS (their row counts
#: are recorded and verified unchanged; none for the fold-observation migration, as for the record-refusal-fact one before it, and the
#: script then says `rebuilds: none`), the v_* objects it recreates (none for the fold-observation migration; the record-refusal-fact
#: migration recreated the five paper-side currency objects, the worklist and the roster match; 0022 `v_run_current` alone, 0021 the
#: eight currency objects and the paper-input head that follows them)
#: (0025 `pending_excludes_superseded`: the one recreate of `v_pending_judgment`; 0026 `twin_fold_view_legacy_value_with_draws`: no
#: rebuild, the three recreates of `v_multi_source_check`, `v_multi_source_rows` and `v_run_current`; 0027
#: `decisive_first_source_local_anchor_solo_stand_in`, the newest file: the one recreate of `v_run_current` and the REBUILD of `run_contrast`
#: -- its anchor_kind CHECK grows `solo_stand_in` -- nothing created or added; the window script records the rebuilt table and its
#: row count, and the landing window needs WINDOW_ALLOW_REBUILD=1 for it, scripts/registry/landing/window.sh; 0028
#: `reference_model_never_selected`: the one recreate of `v_run_current` -- a run of the worked example's key reads not_in_roster /
#: in_current_paper 0 before any include ruling -- and nothing else; 0029 `analysis_filing`: five tables CREATED with the analysis
#: tables' four indexes -- `analysis`, `analysis_input`, `analysis_artifact`, `ordered_fact_record`, the harvester's index of their
#: records by key, and `ordered_fact_carrier`, its audit of every file that carried one -- nothing rebuilt, altered or recreated (0028's
#: recreate is already in the shape below), so the script says `rebuilds: none` and prints no recreate line; 0030 `fold_order_barrier`, the newest
#: file: the deployment barrier of the fold-order version -- the two fold-order fence triggers on `run` recreated (`trg_run_fold_order_fence_ins`
#: / `_upd`, `schema_def.fold_order_fence_triggers`), nothing rebuilt, created or added; the plan reads `pending: 0030_fold_order_barrier`,
#: `shape differs from the code's schema:` with the two `trigger … is absent` lines and `recreate trigger(s): …`, the script says
#: `rebuilds: none`; the version row holds a pre-guard writer at 29 at its own version check, the triggers refuse a `run` write by any
#: connection without the code's `registry_fold_order()`; 0031 `non_template_positions`: `non_template` joins the
#: applied-positions CHECK of `instrument` and `run`, a REBUILD of both -- the plan reads `pending: 0031_non_template_positions`, the two
#: `lacks 'non_template'` drift lines, `rebuild table instrument` and `rebuild table run`, and NO recreate line: the run rebuild recreates
#: the table's six triggers itself, the two fold-order fence triggers among them (`WINDOW_REBUILT_TRIGGERS`, checked in place after the
#: apply); the script records both tables' row counts (the fixture's one instrument and three runs) and verifies them unchanged, and the
#: landing window applies it under WINDOW_ALLOW_REBUILD=1; 0032 `site_label_rename`, the newest file: the five tables whose CHECK names a
#: site label rebuilt, every label column copied through the runner's alias table -- on the shape below it, built from these files,
#: nothing differs, so the plan reads `pending: 0032_site_label_rename` with no drift line and the script says `rebuilds: none`; the run
#: rebuild recreates the table's triggers in place (`WINDOW_REBUILT_TRIGGERS`); a database built before the labels were neutral plans the
#: five rebuilds and lands under WINDOW_ALLOW_REBUILD=1)
WINDOW_REBUILDS = ()
#: the fixture's row counts in the rebuilt tables (`_v17_registry` seeds one instrument and three runs), as the plan record and the verdict carry them
WINDOW_ROW_COUNTS = {}
WINDOW_VIEWS = ()
#: what the newest migration CREATES and ADDS beside its recreates (the fold-observation migration: one table and nothing else; the
#: record-refusal-fact migration: one table, five columns on three tables; 0025: nothing) -- the planner's `create table` / `add column(s)`
#: lines, which the window script prints and the plan record carries; empty for a view-only file
WINDOW_CREATES = ()                                                   # 0030 creates nothing: 0029's five tables are already in the shape below
WINDOW_ADD_COLUMNS = ()


from test_registry_migrations import FENCE_TRIGGERS, _insert_row as _insert, _valid_row, _apply_body_without_version, _sans_fence_render, _sans_ntpl_drift   # the migration suite's own row builder + inserter
#: the triggers the newest migration recreates ON ITS OWN (the planner's `recreate trigger(s)` line): none for 0032 -- its run rebuild
#: recreates the table's triggers inside the rebuild block, the two fold-order fence triggers among them, so those are checked in place
#: after the apply (`WINDOW_REBUILT_TRIGGERS`) rather than read from a recreate line
WINDOW_TRIGGERS = ()
WINDOW_REBUILT_TRIGGERS = FENCE_TRIGGERS


def _v17_registry(root: Path, through: int = 17) -> str:
    """A registry root whose database carries the version-``through`` shape (the frozen 0001..through applied; 17 by default) with
    rows in the three tables 0018 rebuilds."""
    from sisteer.registry import harvest as H
    H.ensure_layout(str(root))
    path = DB.db_path(str(root))
    conn = DB.connect(path)
    for version, slug, mpath in M.migration_files():
        if version <= through:
            M._apply_one(conn, version, slug, open(mpath).read())
    assert M.db_version(conn) == through
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert(conn, "model", _valid_row("model", model_id="m"))
    _insert(conn, "eval", _valid_row("eval", eval_id="e", headline_metric_id="asr_x", producer_spellings_json="[]"))
    _insert(conn, "item_set", _valid_row("item_set", item_set_id="i", eval_id="e"))
    _insert(conn, "instrument", _valid_row("instrument", instrument_id="ins", eval_id="e", item_set_id="i", judge_id=None))
    _insert(conn, "object", _valid_row("object", object_id="none", object_kind="none", paper_label="none", object_role="baseline_none",
                                       fit_target_kind="none", fit_thinking_render="not_applicable", fit_objective="not_applicable",
                                       fit_positions="not_applicable", fit_seed_confidence="not_applicable", derivation="none",
                                       artifact_sha256="none", status="approved"))
    _insert(conn, "object", _valid_row("object", object_id="o1", object_kind="none", paper_label="none", object_role="baseline_none",
                                       fit_target_kind="none", fit_thinking_render="not_applicable", fit_objective="not_applicable",
                                       fit_positions="not_applicable", fit_seed_confidence="not_applicable", derivation="none",
                                       artifact_sha256="none", status="approved", fit_substrate="cluster_b"))
    for k, sub in (("r1", "cluster_a"), ("r2", "cluster_b"), ("r3", "modal")):
        _insert(conn, "run", _valid_row("run", run_id=k * 8, producer_run_key_legacy=k, model_id="m", object_id="none", eval_id="e",
                                        item_set_id="i", instrument_id="ins", substrate=sub, lifecycle_state="queued", queued_utc="2026-01-01T00:00:00.000Z", record_uri="synthetic://" + k))
    _insert(conn, "fit_search", _valid_row("fit_search", search_id="s1", campaign="c", substrate="unverified", started_utc="2026-01-01T00:00:00.000Z"))
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    conn.close()
    return str(root)


def _counts(path: str) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("run", "object", "fit_search")}
    finally:
        con.close()


# --------------------------------------------------------------------------------------------
# the migration file
# --------------------------------------------------------------------------------------------


def test_0018_is_the_rendered_body_against_the_v17_shape_and_lands_rows_intact(tmp_path):
    committed = (MIG / f"{EXPECT}.sql").read_text()
    conn = DB.connect_memory()
    for version, slug, mpath in M.migration_files():
        if version <= 17:
            M._apply_one(conn, version, slug, open(mpath).read())
    # the v17 shape with the LATER changes already in place (0019 recreates v_paper_long; 0020 alters item_result, creates
    # item_withdrawal, rebuilds the two record-kind ledgers and recreates three views; 0021 recreates nine views, v_run_current
    # among them created; 0022 recreates v_run_current again; a later migration drifts on every earlier shape), so the render
    # against it is 0018's own body
    _apply_body_without_version(conn, 20)
    _apply_body_without_version(conn, 21)
    for later in range(22, M.code_version() + 1):      # every later file -- 0022 recreates v_run_current again, then the record-refusal-fact
        if later in (31, 32):                           # migration: one table, five columns, seven recreates (SQL only; its lift has no rows here).
            continue                                    # 0031 and 0032 rebuild `run`, 0018's own table, to the code's text: NOT pre-applied
        _apply_body_without_version(conn, later)
    plan = M.shape_plan(conn)
    # 0031's `instrument` rebuild rides along in the plan (the code's order: fit_search, instrument, object, run) and is stripped from the render
    assert [r["table"] for r in plan["rebuild"]] == ["fit_search", "instrument", "object", "run"] and plan["create"] == [] and plan["add_columns"] == []
    assert _sans_ntpl_drift(plan["drift"]) == ["table fit_search: column substrate CHECK (substrate IN (...)) lacks 'slurm'",
                                               "table object: column fit_substrate CHECK (fit_substrate IN (...)) lacks 'slurm'",
                                               "table run: column substrate CHECK (substrate IN (...)) lacks 'slurm'"]
    rendered = M.render_shape_migration(conn, plan)
    # the committed file = the render plus its title line (the only hand-written line). The object rebuild renders the CODE's CREATE text,
    # which carries the two lineage columns the record-refusal-fact migration declared after the tail (pre-applied above, so the copy lists
    # them too): the frozen body predates them, so the comparison reads today's render without those two columns
    for col in ("parent_object_id", "named_role"):
        head, comment = S.render_column(S.TABLE_BY_NAME["object"].column(col)).split("  -- ", 1)
        rendered = rendered.replace(f"  {head},  -- {comment}\n", "")
    rendered = rendered.replace(", parent_object_id, named_role", "")
    # ... and the run rebuild recreates the table's triggers, which since 0030 include the two fold-order fence triggers the frozen body predates
    rendered = frozen_column_docs(_sans_fence_render(rendered))           # the descriptions reworded since 0018 landed
    body = "\n".join(l for l in committed.splitlines() if not l.startswith(f"-- {EXPECT}:")).lstrip("\n")
    assert body.strip() == rendered.strip()
    conn.close()
    # applied to a v17 database with rows in the three tables: the code's version, rows intact, zero drift, the new value admitted
    root = _v17_registry(tmp_path / "reg")
    path = DB.db_path(root)
    before = _counts(path)
    conn = DB.connect(path)
    assert M.apply_migrations(conn) == list(range(18, M.code_version() + 1)) and M.db_version(conn) == M.code_version()
    assert DB.schema_drift(conn) == []
    assert _counts(path) == before
    assert set(r[0] for r in conn.execute("SELECT substrate FROM run").fetchall()) == {"cluster_a", "cluster_b", "modal"}
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert(conn, "run", _valid_row("run", run_id="s" * 8, producer_run_key_legacy="s", model_id="m", object_id="none", eval_id="e",
                                    item_set_id="i", instrument_id="ins", substrate="slurm", lifecycle_state="queued", queued_utc="2026-01-01T00:00:00.000Z", record_uri="synthetic://s"))
    conn.execute("UPDATE object SET fit_substrate = 'slurm' WHERE object_id = 'o1'")
    conn.execute("UPDATE fit_search SET substrate = 'slurm' WHERE search_id = 's1'")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "run", _valid_row("run", run_id="t" * 8, producer_run_key_legacy="t", model_id="m", object_id="none", eval_id="e",
                                        item_set_id="i", instrument_id="ins", substrate="cluster_c", lifecycle_state="queued", queued_utc="2026-01-01T00:00:00.000Z", record_uri="synthetic://t"))
    conn.close()
    DB.open_or_create(path).close()          # the writers' open accepts the migrated file


def test_plan_on_a_v17_database_names_0018_first_and_the_three_rebuilds(tmp_path, capsys):
    root = _v17_registry(tmp_path / "reg")
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1        # pending: not yet current
    out = capsys.readouterr().out
    pending = ", ".join(f"{v:04d}_{slug}" for v, slug, _ in M.migration_files() if v >= 18)
    assert pending.startswith(EXPECT) and f"schema version 17; code {M.code_version()}; pending: {pending}" in out
    for t, n in (("fit_search", 1), ("object", 2), ("run", 3)):
        assert f"rebuild table {t}: {n} rows to copy" in out, out
    # the plan is against the CODE's shape, so the later files' work is listed too (0020: the two ledgers, the ALTER, the new table)
    for t in WINDOW_REBUILDS:
        assert f"rebuild table {t}: {WINDOW_ROW_COUNTS[t]} rows to copy" in out, out
    assert "add column(s) to table item_result: withdrawn_by_record_sha256, withdrawn_reason (0 rows, none copied; item-scale)" in out
    assert "create table item_withdrawal" in out


# --------------------------------------------------------------------------------------------
# the write lock
# --------------------------------------------------------------------------------------------


def _hold_lock(root: str, held: mp.Event, release: mp.Event) -> None:
    with DB.write_lock(root, timeout_s=5.0):
        held.set()
        release.wait(30)


def test_migrate_refuses_by_name_while_the_write_lock_is_held_and_applies_once_released(tmp_path, capsys):
    root = _v17_registry(tmp_path / "reg", through=WINDOW_BELOW)
    ctx = mp.get_context("fork")
    held, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(root, held, release))
    proc.start()
    try:
        assert held.wait(10)
        rc = cli.main(["--root", root, "migrate"])
        err = capsys.readouterr().err
        assert rc == 4 and "REFUSED" in err and "write lock is held" in err and "nothing was applied" in err
        with DB.reading(DB.db_path(root)) as conn:
            assert M.db_version(conn) == WINDOW_BELOW           # never applied concurrently
    finally:
        release.set()
        proc.join(10)
    assert cli.main(["--root", root, "migrate"]) == 0
    assert f"applied [{M.code_version()}]; schema version {M.code_version()}" in capsys.readouterr().out
    with DB.reading(DB.db_path(root)) as conn:
        assert M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []


# --------------------------------------------------------------------------------------------
# the landing wrapper under a fake systemctl
# --------------------------------------------------------------------------------------------


def _fake_systemctl(bin_dir: Path) -> Path:
    p = bin_dir / "systemctl"
    p.write_text("#!/usr/bin/env bash\n"
                 "# fake: `systemctl --user is-active <unit>` answers active for units named in $FAKE_ACTIVE_UNITS, inactive (exit 3) otherwise\n"
                 "unit=\"${@: -1}\"\n"
                 "for a in ${FAKE_ACTIVE_UNITS:-}; do [ \"$a\" = \"$unit\" ] && { echo active; exit 0; }; done\n"
                 "echo inactive; exit 3\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _run_window(stage: str, root: str, state: Path, bin_dir: Path, env_extra: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({"PATH": f"{bin_dir}:{env.get('PATH', '')}", "PYTHONPATH": str(REPO / "src"), "REGISTRY_PYTHON": sys.executable})
    env.update(env_extra)
    return subprocess.run(["bash", str(WINDOW), stage, "--expect", WINDOW_EXPECT, "--root", root, "--state", str(state), "--src", str(REPO / "src")],
                          capture_output=True, text=True, env=env, timeout=600)


def test_window_refuses_while_a_writer_is_active_then_lands_and_gates_the_merge(tmp_path):
    root = _v17_registry(tmp_path / "reg", through=WINDOW_BELOW)
    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_systemctl(bin_dir)
    # merge-gate before anything ran: refused by name
    r = _run_window("merge-gate", root, state, bin_dir, {})
    assert r.returncode == 10 and "no verdict" in r.stderr
    # a writer timer active: plan refuses, the database is untouched
    r = _run_window("plan", root, state, bin_dir, {"FAKE_ACTIVE_UNITS": "registry-harvest.timer"})
    assert r.returncode == 5 and "registry-harvest.timer" in r.stderr and "nothing touched" in r.stderr
    with DB.reading(DB.db_path(root)) as conn:
        assert M.db_version(conn) == WINDOW_BELOW
    # apply without a plan: refused
    r = _run_window("apply", root, state, bin_dir, {})
    assert r.returncode == 8 and "plan stage first" in r.stderr
    # writers stopped: plan names exactly the newest migration, the tables it rebuilds (the record-refusal-fact migration: none, the script
    # says so) and records their row counts (none), and the object it recreates
    r = _run_window("plan", root, state, bin_dir, {})
    assert r.returncode == 0, r.stderr
    assert f"plan OK: pending exactly {WINDOW_EXPECT}; rebuilds: {','.join(WINDOW_REBUILDS) or 'none'}" in r.stdout
    if WINDOW_VIEWS:
        assert f"recreate view(s): {', '.join(WINDOW_VIEWS)}" in r.stdout
    if WINDOW_TRIGGERS:
        assert f"recreate trigger(s): {', '.join(WINDOW_TRIGGERS)}" in r.stdout
    if not WINDOW_VIEWS and not WINDOW_TRIGGERS:
        assert "recreate " not in r.stdout                                                 # a file that recreates nothing prints no recreate line
    assert [ln for ln in r.stdout.splitlines() if ln.startswith("recreate ")] == ([f"recreate trigger(s): {', '.join(WINDOW_TRIGGERS)}"] if WINDOW_TRIGGERS else []) + ([f"recreate view(s): {', '.join(WINDOW_VIEWS)}"] if WINDOW_VIEWS else [])
    for t in WINDOW_REBUILDS:
        assert f"rebuild table {t}: {WINDOW_ROW_COUNTS[t]} rows to copy" in r.stdout, r.stdout
    assert [ln for ln in r.stdout.splitlines() if ln.startswith("rebuild table ") and not any(ln.startswith(f"rebuild table {t}:") for t in WINDOW_REBUILDS)] == []
    for t in WINDOW_CREATES:
        assert f"create table {t}" in r.stdout
    for t, cols in WINDOW_ADD_COLUMNS:
        assert re.search(rf"add column\(s\) to table {t}: {re.escape(cols)} \(\d+ rows, none copied", r.stdout), (t, r.stdout)
    if not WINDOW_ADD_COLUMNS:
        assert "add column" not in r.stdout
    plan_rec = json.loads((state / f"window-{WINDOW_EXPECT}.plan.json").read_text())
    assert plan_rec["version_before"] == WINDOW_BELOW and plan_rec["rebuilt_tables"] == list(WINDOW_REBUILDS)
    assert plan_rec["row_counts_before"] == WINDOW_ROW_COUNTS
    # a writer waking up between plan and apply: apply refuses
    r = _run_window("apply", root, state, bin_dir, {"FAKE_ACTIVE_UNITS": "registry-cluster-reads.service"})
    assert r.returncode == 5 and "registry-cluster-reads.service" in r.stderr
    # apply + verify
    r = _run_window("apply", root, state, bin_dir, {})
    assert r.returncode == 0, r.stderr
    r = _run_window("verify", root, state, bin_dir, {})
    assert r.returncode == 0, r.stderr + r.stdout
    verdict = json.loads((state / f"migration-{WINDOW_EXPECT}.json").read_text())
    assert verdict["ok"] and verdict["db_version"] == M.code_version() == verdict["code_version"] and verdict["shape_matches"] and verdict["integrity"] == "ok"
    assert verdict["row_counts_before"] == verdict["row_counts_after"] == WINDOW_ROW_COUNTS
    assert _counts(DB.db_path(root)) == {"run": 3, "object": 2, "fit_search": 1}        # the rows the fixture seeded are still there
    with DB.reading(DB.db_path(root)) as conn:
        assert [c[1] for c in conn.execute("PRAGMA table_info(item_result)")][-3:] == ["ingest_seq", "withdrawn_by_record_sha256", "withdrawn_reason"]
        assert conn.execute("SELECT COUNT(*) FROM item_withdrawal").fetchone()[0] == 0
        for v in WINDOW_VIEWS:                                                              # every recreated object is selectable
            assert conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0] >= 0, v
        for t in WINDOW_TRIGGERS + WINDOW_REBUILT_TRIGGERS:                                 # every recreated trigger is in place, the rebuild's too
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?", (t,)).fetchone() is not None, t
        assert [c[1] for c in conn.execute("PRAGMA table_info(v_paper_current)")][-4:] == ["is_current", "why", "coverage_state", "coverage_detail_json"]
    r = _run_window("merge-gate", root, state, bin_dir, {})
    assert r.returncode == 0 and "merge-gate OK" in r.stdout
    # a tampered verdict (shape mismatch) refuses the merge
    verdict["shape_matches"] = False
    (state / f"migration-{WINDOW_EXPECT}.json").write_text(json.dumps(verdict))
    r = _run_window("merge-gate", root, state, bin_dir, {})
    assert r.returncode == 11 and "shape did not match" in r.stderr


def test_window_plan_refuses_when_the_database_is_already_current(tmp_path):
    root = _v17_registry(tmp_path / "reg", through=WINDOW_BELOW)
    assert cli.main(["--root", root, "migrate"]) == 0
    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_systemctl(bin_dir)
    r = _run_window("plan", root, state, bin_dir, {})
    assert r.returncode == 7 and f"expected exactly '{WINDOW_EXPECT}'" in r.stderr


def test_window_plan_refuses_a_database_two_migrations_behind(tmp_path):
    """The window lands ONE migration: a v17 database has 0018 AND the newest file pending, and the plan stage refuses by name rather
    than landing a rebuild nobody planned for (the pending list is printed for the operator)."""
    root = _v17_registry(tmp_path / "reg")
    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_systemctl(bin_dir)
    r = _run_window("plan", root, state, bin_dir, {})
    assert r.returncode == 7 and f"expected exactly '{WINDOW_EXPECT}'" in r.stderr and EXPECT in r.stderr
    with DB.reading(DB.db_path(root)) as conn:
        assert M.db_version(conn) == 17
