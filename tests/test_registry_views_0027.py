"""Migration 0027 `decisive_first_source_local_anchor_solo_stand_in` (registry follow-ups-4), one rule per test on a synthetic corpus:

* the DECISIVE supersede / exclude ruling precedes the roster branch of `v_run_current`'s selection_state CASE: a run a
  ruling retired reads `superseded` / `excluded` whether or not a roster entry matches it (0021 read `not_in_roster` beside
  `decisive_action` for a roster-less run, with an is_current belt for the pair); `is_current` does not move (both spellings were
  non-current), `why` reads the in-roster text, and `in_current_paper` was already 0. The migration moved the RULING alone: the
  producer's own retirement and the later-run rule keep their roster-first place, so a roster-less run a later run superseded still
  reads not_in_roster and stays current, exactly as before the file.

The migration mechanics (the file is the render against the v26 shape; the objects before and after it) are the first test's; the
run-key pins of tests/test_registry_views_0021.py read the new spelling.
"""
from __future__ import annotations

import os
import shutil

import test_registry_views_0021 as T21
from test_registry_migrations import AF_TABLES, FU4_SLUG, FU4_TABLE_DRIFT, NTPL_TABLES, _pre_0027_run_current, _pre_0028_run_current, _sans_af_drift, _sans_af_render, _sans_fu4_drift

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry.adapters import anchor_current as AC
from test_registry_schema import frozen_column_docs

SLUG = FU4_SLUG
#: the objects the file recreates (`v_run_current`'s line is 0026's on the shape below; the table line is the anchor-kind commit's)
FU4_VIEWS = ("v_run_current",)
FU4_VIEW_DRIFT = [f"{v} differs from the code's SQL" for v in FU4_VIEWS]


# --------------------------------------------------------------------------------------------
# the migration on the shape below it
# --------------------------------------------------------------------------------------------


def _fu4_migration():
    return next(f for f in M.migration_files() if f[1] == SLUG)


def _shape_below(path: str) -> None:
    version = _fu4_migration()[0]
    conn = DB.connect(path)
    for v, slug, mpath in M.migration_files():
        if v < version:
            M._apply_one(conn, v, slug, open(mpath, encoding="utf-8").read())
    conn.commit()
    conn.close()


def test_the_migration_plan_on_the_shape_below_it_and_the_committed_body_is_the_render(tmp_path, capsys):
    version, slug, mpath = _fu4_migration()
    assert version == M.code_version() - 5, "the newest file but five on this tree: 0028 (reference_model_never_selected), 0029 (analysis filing, D357), 0030 (fold_order_barrier: the two fold-order fence triggers on run) and 0031 (non_template_positions: the applied-positions CHECK of instrument and run, a rebuild of both) follow it; their lines are set aside below"
    assert os.path.basename(mpath) == f"{version:04d}_{SLUG}.sql"
    path = str(tmp_path / "below.sqlite")
    _shape_below(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    assert sorted(_sans_fu4_drift(DB.schema_drift(conn))) == sorted(FU4_VIEW_DRIFT)
    plan = M.shape_plan(conn)
    # 0031's two rebuilds (instrument, run) ride along in the plan against the code's shape and are set aside; 0027's own is run_contrast
    assert [(r["table"], r["rows"]) for r in plan["rebuild"] if r["table"] not in NTPL_TABLES] == [("run_contrast", 0)] and plan["add_columns"] == [] and plan["undeclared_tables"] == [] and [c for c in plan["create"] if c not in AF_TABLES] == []
    assert [r["table"] for r in plan["rebuild"] if r["table"] in NTPL_TABLES] == list(NTPL_TABLES)
    assert sorted(_sans_af_drift(DB.schema_drift(conn))) == sorted(FU4_TABLE_DRIFT + FU4_VIEW_DRIFT)   # 0029's seven lines aside
    # a rebuild recreates its table's triggers itself, so none is listed as a recreate of its own -- 0030's two fence triggers on `run`
    # included, since 0031 rebuilds `run` on this shape
    assert plan["objects"] == {"index": {"recreate": [], "drop": []}, "trigger": {"recreate": [], "drop": []},
                               "view": {"recreate": sorted(FU4_VIEWS), "drop": []}}
    # the render against the v26 shape carries the CODE's v_run_current (0028's text); the committed file carries 0027's -- taken back
    # through the frozen 0027 text (tests/test_registry_migrations.py::_pre_0028_run_current), as the earlier files' tests do for 0027's
    code_rc = S.VIEW_SQL["v_run_current"].strip()
    rendered = frozen_column_docs(M.render_shape_migration(conn, plan))    # the descriptions reworded since 0027 landed read at the frozen text
    assert rendered.count("DROP VIEW IF EXISTS v_run_current;\n" + code_rc) == 1
    rendered = rendered.replace("DROP VIEW IF EXISTS v_run_current;\n" + code_rc, "DROP VIEW IF EXISTS v_run_current;\n" + _pre_0028_run_current(code_rc))
    committed = open(mpath, encoding="utf-8").read()
    header = committed[:committed.index("-- drift: ")]
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n") and "Three registry follow-ups" in header and "source_local_anchor" in header
    body = committed[committed.index("-- drift: "):]
    assert body.strip() == _sans_af_render(rendered).strip()                                # today's render carries 0029's three creates too (and 0028's text of the object)
    assert body.count("DROP VIEW IF EXISTS ") == len(FU4_VIEWS) and body.count("\nCREATE VIEW ") == len(FU4_VIEWS)
    # the documented rebuild dance over run_contrast and nothing else (0002 the worked example): the new table from the code's DDL with the
    # grown CHECK, every column copied in primary-key order, the six sentinel-column FK triggers recreated, ANALYZE
    assert body.count("-- rebuild run_contrast:") == 1 and body.count("CREATE TABLE run_contrast__new (") == 1
    assert "INSERT INTO run_contrast__new (run_id, anchor_run_id, judgment_id, metric_id, threshold, computation_id, anchor_judgment_id, anchor_kind," in body
    assert "DROP TABLE run_contrast;\nALTER TABLE run_contrast__new RENAME TO run_contrast;" in body and body.count("CREATE TRIGGER trg_run_contrast_") == 6
    assert "ANALYZE run_contrast;" in body and "DROP TABLE" not in body.replace("DROP TABLE run_contrast;", "")
    assert "'solo_stand_in'" in body[body.index("CREATE TABLE run_contrast__new"):body.index("INSERT INTO run_contrast__new")]
    assert "WINDOW_ALLOW_REBUILD=1" in header and "11,869" in header
    for v in FU4_VIEWS:
        assert "DROP VIEW IF EXISTS " + v + ";\n" + _pre_0028_run_current(S.VIEW_SQL[v]).strip() in body, v      # 0027's text, 0028's follows it
    assert SLUG not in M.POST_APPLY                                                        # no Python post-step: nothing is keyed by the slug
    # the object before the file: 0026's text -- the roster branch before the decisive ruling, the is_current belt for the pair
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'v_run_current'").fetchone()[0]
    assert before == _pre_0027_run_current(S.VIEW_SQL["v_run_current"]).strip().rstrip(";")
    assert before.index("WHEN pm.run_id IS NULL THEN 'not_in_roster'") < before.index("d.action = 'supersede'")
    assert "OR (selection_state = 'not_in_roster' AND decisive_action IN ('exclude', 'supersede')) THEN 0" in before and "decisive ruling" in before
    ddl_before = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'run_contrast'").fetchone()[0]
    assert "'borrowed', 'full_dose'))" in ddl_before and "solo_stand_in" not in ddl_before
    # the code's text: the decisive ruling first, no belt, no `decisive ruling` why
    code = S.VIEW_SQL["v_run_current"]
    sel = S.SELECTION_STATE_SQL
    assert sel.index("WHEN d.action = 'include' THEN 'selected'") < sel.index("WHEN d.action = 'supersede' THEN 'superseded'") \
        < sel.index("WHEN d.action = 'exclude' THEN 'excluded'") < sel.index("WHEN pm.run_id IS NULL THEN 'not_in_roster'") \
        < sel.index("WHEN r.lifecycle_state = 'superseded' OR r.superseded_by_run_id IS NOT NULL OR l.run_id IS NOT NULL THEN 'superseded'")
    assert "decisive_action IN ('exclude', 'supersede')" not in code and "decisive ruling" not in code
    assert sel.index("WHEN d.action = 'exclude' THEN 'excluded'") < sel.index(f"WHEN {S.SOURCE_LOCAL_ANCHOR_SQL} THEN 'source_local_anchor'") \
        < sel.index("WHEN pm.run_id IS NULL THEN 'not_in_roster'")
    conn.close()
    # the CLI's plan on a COPY: the pending file, the recreate line, exit 1
    root = str(tmp_path / "copyroot")
    os.makedirs(root)
    shutil.copy(path, DB.db_path(root))
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert f"schema version {version - 1}; code {M.code_version()}; pending: {version:04d}_{SLUG}, " in out     # 0028 and 0029 follow it in the pending list
    assert "recreate view(s): " + ", ".join(sorted(FU4_VIEWS)) in out and "rebuild table run_contrast: 0 rows to copy" in out
    for d in FU4_TABLE_DRIFT + FU4_VIEW_DRIFT:
        assert f"  {d}" in out
    for absent in ("add column", "drop undeclared", "recreate index"):
        assert absent not in out, out
    assert [ln for ln in out.splitlines() if ln.startswith("recreate trigger")] == [], out   # 0030's two fence triggers ride 0031's run rebuild
    assert [ln for ln in out.splitlines() if ln.startswith("rebuild table ") and ln.split()[2].rstrip(":") in NTPL_TABLES] == [f"rebuild table {t}: 0 rows to copy (run-scale)" for t in NTPL_TABLES], out   # 0031's two
    assert [ln for ln in out.splitlines() if ln.startswith("create table") and ln.split()[-1] not in AF_TABLES] == [], out   # 0029's five creates aside
    # applied (0028 and 0029 follow): the code's version, zero drift, the object at the code's text, every paper view selectable
    conn = DB.connect(path)
    assert M.apply_migrations(conn) == list(range(version, M.code_version() + 1)) and M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []
    for v in FU4_VIEWS:
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (v,)).fetchone()[0] == S.VIEW_SQL[v].strip().rstrip(";"), v
    for v in FU4_VIEWS + ("v_paper_long", "v_paper_current"):
        assert conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0] == 0, v
    ddl_after = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'run_contrast'").fetchone()[0]
    assert "'borrowed', 'full_dose', 'solo_stand_in'))" in ddl_after and len(conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'run_contrast'").fetchall()) == 6
    conn.close()


# --------------------------------------------------------------------------------------------
# the decisive ruling reads by name whatever the roster says; the later-run rule keeps its roster-first place
# --------------------------------------------------------------------------------------------


def test_a_roster_less_run_under_a_decisive_ruling_reads_the_rulings_spelling(tmp_path):
    """The roster matches the steered run only. A key-scoped exclude on the corpus's roster-less `none`: selection_state `excluded`,
    in_current_paper 0, is_current 0, why `selection_state excluded (ruling …)` -- the text a roster-matched run carries; a later key-scoped
    supersede: `superseded` with its ruling named. The same ruling with a roster entry that matches the run reads IDENTICALLY (the label no
    longer depends on the roster), and both spellings classify the same way for the anchor currency."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, lora_run = ids["run_ids"]
    T21._write(root, "roster", "000001-roster.jsonl", [T21._roster_entry(1)])
    T21._write(root, "roster", "000002-exclude.jsonl", [T21._ruling("R_ex", "exclude", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=60)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc, pc = T21._by_run(conn, "v_run_current"), T21._by_run(conn, "v_paper_current")
    without_entry = (rc[none_run]["selection_state"], rc[none_run]["in_current_paper"], rc[none_run]["is_current"], rc[none_run]["why"], rc[none_run]["decisive_action"])
    assert without_entry == ("excluded", 0, 0, "selection_state excluded (ruling R_ex)", "exclude")
    assert rc[none_run]["roster_entry_ids_json"] == "[]", "no entry matches the run: the label is the ruling's all the same"
    assert (pc[none_run]["selection_state"], pc[none_run]["is_current"], pc[none_run]["why"]) == ("excluded", 0, rc[none_run]["why"])
    assert (rc[lora_run]["selection_state"], rc[lora_run]["in_current_paper"], rc[lora_run]["is_current"]) == ("selected", 1, 1)
    assert AC._classify_why(rc[none_run]["why"]) == AC._classify_why("decisive ruling exclude (ruling R_ex)") == "excluded"
    conn.close()
    # a later supersede ruling is the decisive one: superseded, named
    T21._write(root, "roster", "000003-supersede.jsonl", [T21._ruling("R_sup", "supersede", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=70,
                                                                       target_json={"superseded_by": "later-run"})])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("superseded", 0, "selection_state superseded (ruling R_sup)")
    assert AC._classify_why(rc[none_run]["why"]) == AC._classify_why("decisive ruling supersede (ruling R_sup)") == "superseded"
    conn.close()
    # a roster entry that matches the run: the same ruling, the same reading
    T21._write(root, "roster", "000004-roster.jsonl", [T21._roster_entry(2, object_id="none", comparison="baseline")])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert rc[none_run]["roster_entry_ids_json"] != "[]"
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("superseded", 0, "selection_state superseded (ruling R_sup)")
    conn.close()


def test_a_roster_less_run_a_later_run_superseded_keeps_the_roster_first_reading(tmp_path):
    """The migration moved the RULING alone. Two roster-less `none` runs at one key, the second complete later: the earlier reads
    not_in_roster and is CURRENT (is_current 1, why NULL), as it did before migration 0027 -- the later-run rule sits after the
    roster branch, so the label and the currency of a roster-less run no ruling touched do not move. With a roster entry matching
    both, the earlier reads superseded by the later run (the 0021 reading) and the later one selected."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, _lora_run = ids["run_ids"]
    later, recs = synth.run_records("synthstore_none_later", 900, "none", "none", 4)
    T21._write(root, "later", "000900-later.jsonl", recs)
    T21._write(root, "roster", "000001-roster.jsonl", [T21._roster_entry(1)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    for rid in (none_run, later):
        assert (rc[rid]["selection_state"], rc[rid]["in_current_paper"], rc[rid]["is_current"], rc[rid]["why"], rc[rid]["decisive_action"]) == ("not_in_roster", 0, 1, None, None), rid
    conn.close()
    T21._write(root, "roster", "000002-roster.jsonl", [T21._roster_entry(2, object_id="none", comparison="baseline")])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert (rc[none_run]["selection_state"], rc[none_run]["is_current"], rc[none_run]["why"]) == ("superseded", 0, "selection_state superseded (a later run at the same key)")
    assert (rc[later]["selection_state"], rc[later]["in_current_paper"], rc[later]["is_current"]) == ("selected", 1, 1)
    conn.close()
