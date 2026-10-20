"""Migration 0028 `reference_model_never_selected`: a run of the front-door WORKED EXAMPLE's key (`schema_def.WORKED_EXAMPLE_MODEL_IDS`:
qwen35_4b_demo / qwen35_4b_demo_base) is never `selected` into the paper and never a paper row, whatever the roster or the rulings say (the
never-a-result requirement: both selection CASEs of `v_run_current` tested the decisive `include` before the missing-roster branch, and the
live registry carries a model-unrestricted include -- the full-suite BigCodeBench / BFCL appendix one -- so a completed demo run matching it
would have read `selected`, in_current_paper 1). The predicate is the KEY SET, not `paper_status`: `reference` means "not one of the paper's
models", and a reference model's run IS in the frame when a roster entry or a ruling puts it there (the 35B's 26 appendix runs stay selected
by that include); a `worked_example` paper_status would be a new value of the CHECK on `model`, a table rebuild (the allowed fallback). One
rule per test on a synthetic corpus:

* the migration mechanics: the file is the render against the v27 shape (one recreate of `v_run_current`, nothing else), the object
  before and after it;
* a completed worked-example run under a model-unrestricted include reads `not_in_roster` / in_current_paper 0 / is_current 1 while the
  SAME ruling selects the paper model's run AND a `reference` model's run (`selected` / 1); `v_paper_current` agrees;
* a decisive exclude still retires a worked-example run by name (`excluded`, is_current 0): retirement is the stronger fact;
* a roster entry naming the key changes nothing: still never a paper row.
"""
from __future__ import annotations

import os
import shutil

import test_registry_views_0021 as T21
from test_registry_migrations import AF_TABLES, NTPL_TABLES, RM_SLUG, _pre_0028_run_current, _sans_af_drift, _sans_af_render, _sans_fence_drift, _sans_fence_render

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import migrations as M
from sisteer.registry import schema_def as S
from sisteer.registry import synth

SLUG = RM_SLUG
RM_VIEWS = ("v_run_current",)
RM_VIEW_DRIFT = [f"{v} differs from the code's SQL" for v in RM_VIEWS]
#: the worked example's key, written into the synthetic corpus as a reference model (the real row: configs/registry/models.yaml); `v_run_current`
#: names it by KEY (WORKED_EXAMPLE_MODEL_IDS), so a second reference model in the corpus, REF, keeps today's semantics
DEMO = "qwen35_4b_demo"
REF = "ref_model_it"
N_ITEMS = 4


# --------------------------------------------------------------------------------------------
# the migration on the shape below it
# --------------------------------------------------------------------------------------------


def _rm_migration():
    return next(f for f in M.migration_files() if f[1] == SLUG)


def _shape_below(path: str) -> None:
    version = _rm_migration()[0]
    conn = DB.connect(path)
    for v, slug, mpath in M.migration_files():
        if v < version:
            M._apply_one(conn, v, slug, open(mpath, encoding="utf-8").read())
    conn.commit()
    conn.close()


def test_the_migration_plan_on_the_shape_below_it_is_one_recreate_and_the_committed_body_is_the_render(tmp_path, capsys):
    version, slug, mpath = _rm_migration()
    assert version == M.code_version() - 4, "the newest file but four on this tree: 0029 (analysis_filing, D357), 0030 (fold_order_barrier: the two fold-order fence triggers on run) and 0031 (non_template_positions: the applied-positions CHECK of instrument and run, a rebuild of both) follow it; their lines are set aside below"
    assert os.path.basename(mpath) == f"{version:04d}_{SLUG}.sql"
    path = str(tmp_path / "below.sqlite")
    _shape_below(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    assert sorted(_sans_fence_drift(_sans_af_drift(DB.schema_drift(conn)))) == sorted(RM_VIEW_DRIFT)   # 0029's nine lines and 0030's two aside
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"]] == list(NTPL_TABLES) and plan["add_columns"] == [] and plan["undeclared_tables"] == [] and [c for c in plan["create"] if c not in AF_TABLES] == []   # 0031's two rebuilds and 0029's five creates aside
    assert plan["objects"] == {"index": {"recreate": [], "drop": []}, "trigger": {"recreate": [], "drop": []},   # 0030's two triggers ride 0031's run rebuild
                               "view": {"recreate": sorted(RM_VIEWS), "drop": []}}
    rendered = M.render_shape_migration(conn, plan)
    committed = open(mpath, encoding="utf-8").read()
    header = committed[:committed.index("-- drift: ")]
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n") and "never be a result" in header and "never selected" in header and "WORKED_EXAMPLE_MODEL_SQL" in header
    assert "no table is rebuilt" in header and "no data is rewritten" in header and "NONE live" in header and "26 appendix runs" in header
    assert "paper_status" in header and "0 runs carry model_id qwen35_4b_demo or qwen35_4b_demo_base" in header
    body = committed[committed.index("-- drift: "):]
    assert body.strip() == _sans_fence_render(_sans_af_render(rendered)).strip()            # today's render carries 0029's five creates and 0030's two triggers too
    assert body.count("DROP VIEW IF EXISTS ") == 1 and body.count("\nCREATE VIEW ") == 1 and "CREATE TABLE" not in body and "ALTER TABLE" not in body
    for v in RM_VIEWS:
        assert "DROP VIEW IF EXISTS " + v + ";\n" + S.VIEW_SQL[v].strip() in body, v
    assert SLUG not in M.POST_APPLY                                                        # no Python post-step: nothing is keyed by the slug
    # the object before the file: 0027's text -- include before the roster branch, no worked-example branch
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'v_run_current'").fetchone()[0]
    assert before == _pre_0028_run_current(S.VIEW_SQL["v_run_current"]).strip().rstrip(";")
    assert "qwen35_4b_demo" not in before
    # the code's text: the worked-example branch after `withdrawn` and before `include` in BOTH CASEs, by KEY, no model join, no paper_status
    sel, pap, code = S.SELECTION_STATE_SQL, S.IN_CURRENT_PAPER_SQL, S.VIEW_SQL["v_run_current"]
    assert S.WORKED_EXAMPLE_MODEL_IDS == ("qwen35_4b_demo", "qwen35_4b_demo_base")
    assert S.WORKED_EXAMPLE_MODEL_SQL == "r.model_id IN ('qwen35_4b_demo', 'qwen35_4b_demo_base')" and "paper_status" not in code and "JOIN model" not in code
    assert sel.index("WHEN w.run_id IS NOT NULL THEN 'withdrawn'") < sel.index(S.WORKED_EXAMPLE_MODEL_SQL) < sel.index("WHEN d.action = 'include' THEN 'selected'")
    assert f"WHEN {S.WORKED_EXAMPLE_MODEL_SQL} AND COALESCE(d.action, '') NOT IN ('exclude', 'supersede') THEN 'not_in_roster'" in sel
    assert pap.index("WHEN w.run_id IS NOT NULL THEN 0") < pap.index(f"WHEN {S.WORKED_EXAMPLE_MODEL_SQL} THEN 0") < pap.index("WHEN d.action = 'include' THEN 1")
    assert "model" not in S.VIEW_REFERENCES["v_run_current"]                                 # no join: the references are 0027's
    conn.close()
    # the CLI's plan on a COPY: the pending file, the recreate line, no rebuild, exit 1
    root = str(tmp_path / "copyroot")
    os.makedirs(root)
    shutil.copy(path, DB.db_path(root))
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert f"schema version {version - 1}; code {M.code_version()}; pending: {version:04d}_{SLUG}, " in out     # 0029 and 0030 follow it in the pending list
    assert "recreate view(s): " + ", ".join(sorted(RM_VIEWS)) in out and f"  {RM_VIEW_DRIFT[0]}" in out
    for absent in ("add column", "drop undeclared", "recreate index", "recreate trigger"):
        assert absent not in out, out
    assert [ln for ln in out.splitlines() if ln.startswith("rebuild table ")] == [f"rebuild table {t}: 0 rows to copy (run-scale)" for t in NTPL_TABLES], out   # 0031's two rebuilds, nothing else
    assert [ln for ln in out.splitlines() if ln.startswith("create table") and ln.split()[-1] not in AF_TABLES] == [], out   # 0029's five creates aside
    # applied (0029, 0030 and 0031 follow): the code's version, zero drift, the object at the code's text, every paper view selectable
    conn = DB.connect(path)
    assert M.apply_migrations(conn) == [version, version + 1, version + 2, version + 3, version + 4] and M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []
    for v in RM_VIEWS:
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (v,)).fetchone()[0] == S.VIEW_SQL[v].strip().rstrip(";"), v
    for v in RM_VIEWS + ("v_paper_long", "v_paper_current"):
        assert conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0] == 0, v
    conn.close()


# --------------------------------------------------------------------------------------------
# the rule on a synthetic corpus
# --------------------------------------------------------------------------------------------


def _reference_model(mid, i):
    return synth.rec("model", {"model_id": mid, "family": "qwen35", "size_label": "4b", "checkpoint_kind": "instruct", "hf_repo": "org/x",
                               "hf_revision": "d" * 40, "chat_template_sha256": synth.SHA256_A, "thinking_channel": "available",
                               "paper_status": "reference"}, synth.t(i))


def _roster_entry_for(mid, i):
    """A roster entry naming ``mid`` on the corpus's eval / instrument / judge, by object id (T21._roster_entry names the paper model)."""
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": "fig_synth", "model_id": mid, "eval_id": synth.EVAL_ID,
         "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point", "object_id": "none",
         "source_citation": "synthetic", "inferred": 0}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _completed_run(root, key, i0, mid):
    """A COMPLETED run (judged, with its item rows) of model ``mid`` on the corpus's eval, written into its own run directory."""
    rid, recs = synth.run_records(key, i0, "none", "none", n_items=N_ITEMS, model_id=mid)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    files = [(str(r["record_type"]), [r]) for r in recs]
    files.append(("item_result", synth.item_records(rid, jid, i0 + 5, N_ITEMS, seed=i0)))
    synth.write_run_dir(root, synth.CAMPAIGN, rid, files)
    return rid


def _corpus(tmp_path):
    """The T21 corpus (the paper model's `none` and LoRA runs, in_paper) plus a COMPLETED run of the worked example's key and a COMPLETED
    run of another `reference` model (REF: the 35B's situation -- not the paper's, in the frame only by roster or ruling); both reference."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=N_ITEMS)
    paper_none, _paper_lora = ids["run_ids"]
    T21._write(root, "dimensions", "000002-reference-models.jsonl", [_reference_model(DEMO, 30), _reference_model(REF, 31)])
    demo_run = _completed_run(root, "demo_none", 300, DEMO)
    ref_run = _completed_run(root, "ref_none", 400, REF)
    return root, paper_none, demo_run, ref_run


def _state(conn, rid, table="v_run_current"):
    r = T21._by_run(conn, table)[rid]
    return r["selection_state"], r["in_current_paper"], r["is_current"]


def test_a_model_unrestricted_include_selects_the_paper_models_run_and_never_the_reference_models(tmp_path):
    """The live shape of the finding: an eval-scoped include with NO model term (the full-suite BigCodeBench / BFCL include is key-scoped
    without a model term; `eval` is the synthetic corpus's spelling of the same breadth) covers the completed demo run, the completed run
    of another `reference` model and the paper model's `none` alike. The paper model's run reads selected / 1; so does the OTHER reference
    model's (the 35B's situation: in the frame by the deliberate include, exactly as before this file); the demo's reads not_in_roster / 0
    and stays current (is_current 1): not a result, not retired. v_paper_current agrees, and the decisive action is recorded on the demo
    run all the same."""
    root, paper_none, demo_run, ref_run = _corpus(tmp_path)
    T21._write(root, "roster", "000001-include.jsonl", [T21._ruling("R_inc", "include", "eval", {"eval_id": synth.EVAL_ID}, i=60)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT paper_status FROM model WHERE model_id = ?", (DEMO,)).fetchone()[0] == "reference"
    assert conn.execute("SELECT paper_status FROM model WHERE model_id = ?", (REF,)).fetchone()[0] == "reference"
    assert _state(conn, ref_run) == ("selected", 1, 1), "a reference model's run under the include is a paper row, as before (by key, not by status)"
    assert _state(conn, ref_run, "v_paper_current")[:2] == ("selected", 1)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (demo_run,)).fetchone()[0] in ("judged", "ingested")
    rc = T21._by_run(conn, "v_run_current")
    assert rc[demo_run]["decisive_action"] == "include" and rc[demo_run]["decisive_ruling_id"] == "R_inc", "the ruling covers the demo run"
    assert _state(conn, demo_run) == ("not_in_roster", 0, 1)
    assert rc[demo_run]["why"] is None                                                      # current: nothing retired it
    assert _state(conn, paper_none) == ("selected", 1, 1) and rc[paper_none]["decisive_ruling_id"] == "R_inc"
    assert _state(conn, demo_run, "v_paper_current")[:2] == ("not_in_roster", 0) and _state(conn, paper_none, "v_paper_current")[:2] == ("selected", 1)
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", (demo_run,)).fetchone()[0] == 0
    conn.close()


def test_a_decisive_exclude_still_retires_a_worked_example_run_by_name(tmp_path):
    """Retirement is the stronger fact: a later key-scoped exclude on the demo run reads `excluded`, in_current_paper 0, is_current 0,
    with the ruling named -- exactly as a paper model's run would (migration 0027's reading); the include it outranks no longer selects."""
    root, paper_none, demo_run, _ref_run = _corpus(tmp_path)
    T21._write(root, "roster", "000001-include.jsonl", [T21._ruling("R_inc", "include", "eval", {"eval_id": synth.EVAL_ID}, i=60)])
    T21._write(root, "roster", "000002-exclude.jsonl", [T21._ruling("R_ex", "exclude", "key", {"model_id": DEMO, "object_id": "none"}, i=70)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert _state(conn, demo_run) == ("excluded", 0, 0) and rc[demo_run]["why"] == "selection_state excluded (ruling R_ex)"
    assert _state(conn, paper_none) == ("selected", 1, 1)                                   # the key-scoped exclude names the demo model only
    conn.close()


def test_a_roster_entry_naming_the_worked_example_changes_nothing(tmp_path):
    """No live roster names the key (0 entries); were one written, the run would match it and still never be a paper row:
    not_in_roster / 0 / current, the include notwithstanding."""
    root, paper_none, demo_run, _ref_run = _corpus(tmp_path)
    T21._write(root, "roster", "000001-roster.jsonl", [_roster_entry_for(DEMO, 1)])
    T21._write(root, "roster", "000002-include.jsonl", [T21._ruling("R_inc", "include", "eval", {"eval_id": synth.EVAL_ID}, i=60)])
    T21._ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rc = T21._by_run(conn, "v_run_current")
    assert rc[demo_run]["roster_entry_ids_json"] != "[]", "the entry matches the demo run"
    assert _state(conn, demo_run) == ("not_in_roster", 0, 1)
    assert _state(conn, paper_none) == ("selected", 1, 1)
    conn.close()
