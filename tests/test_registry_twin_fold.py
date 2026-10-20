"""Migration 0026 `twin_fold_view_legacy_value_with_draws`:

* the `r2_export` <-> `sampled_lane` TWINS fold onto the pooled reading -- `Ownership.pairing_rule view_legacy` (the tag both readers file),
  `check_rule value_with_draws` (the values agree to 1e-9, the export's `n` equals the pooled `extra.n_draws`, the export carries `k`; a pooled
  owner files `k` NULL and `n` as goals), the two declarations of `ownership.TWIN_OWNERSHIP` (the three models' `none`, unfitted; Qwen's ReLU
  constitution steer on `instruct_direct`, `nominal_one`), `v_multi_source_rows` projecting `check_rule` and `n_draws`, `v_multi_source_check`
  branching on the rule. Per export run the `asr@0.5` / `asr@0.75` identities FOLD onto the same-tag pooled row (the one paper row), `asr@0.25`
  and the degeneration shares STAND as the export's stopgaps (never drop a uniquely filed measurement), a same-tag disagreement reads
  `refused`, an owner row without draws never folds, another tag's pooled row pairs with nothing.
* the PAIRWISE coexistence: two `sampled_lane` runs under `pooled_t0_t1` at one run-identity key whose `view_legacy` differ are two points'
  readings, never predecessor and successor -- the `later` CTE of `v_run_current` skips the pair (`vocab.COEXIST_ACROSS_VIEWS_BY_SOURCE`,
  rendered by `schema_def.coexist_across_views_sql`); a later SAME-tag pooled run still supersedes, another source's two-tag pair keeps the
  later-run rule, two unlabeled pooled runs pair as before.
* the migration itself: on the shape one version below it the planner reports exactly the three view recreates (`v_multi_source_check`,
  `v_multi_source_rows`, `v_run_current`) and nothing else, the committed body IS the render, no post-step; `v_paper_long` and
  `v_paper_current` read the check view and are unchanged in text.
* the live twin test's post-fold assertions (tests/test_registry_twins_live.py::assert_post_fold) RUN here against a synthetic registry at the
  new shape -- they skip live until the migration is applied to the live registry.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

import test_registry_views_0021 as T21
import test_registry_views_0022 as T22
from test_registry_migrations import AF_TABLES, FU4_TABLES, NTPL_TABLES, TF_DRIFT, TF_VIEWS, _pre_0027_run_current, _sans_fu4_drift, _sans_fu4_render
from test_registry_twins_live import assert_post_fold

from sisteer.registry import adapters as ADP
from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import migrations as M
from sisteer.registry import ownership as OW
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V

SLUG = "twin_fold_view_legacy_value_with_draws"
DECL_NONE = OW.BY_ID["sampled_static_8192/none/static_attack/none"]
DECL_RELU = OW.BY_ID["sampled_static_8192/constitution_relu/static_attack/instruct_direct"]
MODEL_ID = "qwen35_4b_it"                        # one of the declared scope's spellings
EVAL_ID, ITEM_SET_ID = "synth_static", "synth_static_259"
TAG_A, TAG_B = "qwen35_4b_it", "qwen35_4b"       # two tags (view_legacy) at one key
OWNER_CAMPAIGN, SECOND_CAMPAIGN = "synth_sampled_campaign", "synth_export_campaign"
EXPORT_COMP = I.computation_id("synthetic.export_build", "1", "none")
READER_COMP = I.computation_id("synthetic.pooled_stats", "1", "none")
INST = synth.instrument_fields(eval_id=EVAL_ID, item_set_id=ITEM_SET_ID, token_cap_tokens=8192, token_cap_ladder_json="[4096, 8192]", harness_config_hash="t" * 8)
INST_ID = str(INST["instrument_id"])
INST_4K = synth.instrument_fields(eval_id=EVAL_ID, item_set_id=ITEM_SET_ID, token_cap_tokens=4096, token_cap_ladder_json="[4096]", harness_config_hash="f" * 8)   # the export's 4,096-token instrument
INST_4K_ID = str(INST_4K["instrument_id"])
TAG_C = "qwen35_4b_nopool"                       # a tag with an 8,192 export reading and no pooled twin
N_DRAWS, N_GOALS = 518, 259
EXPORT_K = {0.25: 90, 0.5: 70, 0.75: 40}        # the export's counts over the draws; the pooled row's value equals k / N_DRAWS


# --------------------------------------------------------------------------------------------
# the migration on the shape below it
# --------------------------------------------------------------------------------------------


def _tf_migration():
    return next(f for f in M.migration_files() if f[1] == SLUG)


def _shape_below(path: str) -> None:
    version = _tf_migration()[0]
    conn = DB.connect(path)
    for v, slug, mpath in M.migration_files():
        if v < version:
            M._apply_one(conn, v, slug, open(mpath, encoding="utf-8").read())
    conn.commit()
    conn.close()


def test_the_migration_plan_on_the_shape_below_it_is_three_view_recreates_and_the_committed_body_is_the_render(tmp_path, capsys):
    version, slug, mpath = _tf_migration()
    assert version == M.code_version() - 6, "the newest file but six on this tree: 0027 (registry follow-ups-4), 0028 (reference_model_never_selected), 0029 (analysis filing), 0030 (fold_order_barrier) and 0031 (non_template_positions: a rebuild of instrument and run) follow it"
    assert os.path.basename(mpath) == f"{version:04d}_{SLUG}.sql"
    path = str(tmp_path / "below.sqlite")
    _shape_below(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    assert sorted(_sans_fu4_drift(DB.schema_drift(conn))) == sorted(TF_DRIFT)                # 0027's run_contrast line aside (its own test pins it)
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == [] and plan["add_columns"] == [] and plan["undeclared_tables"] == [] and [c for c in plan["create"] if c not in AF_TABLES] == []   # 0027's run_contrast rebuild, 0031's two and 0029's five creates aside
    assert plan["objects"] == {"index": {"recreate": [], "drop": []}, "trigger": {"recreate": [], "drop": []},   # 0030's two fence triggers ride 0031's run rebuild
                               "view": {"recreate": sorted(TF_VIEWS), "drop": []}}
    rendered = M.render_shape_migration(conn, plan)
    committed = open(mpath, encoding="utf-8").read()
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n") and "a ruling of 2026-10-03" in committed[:committed.index("-- drift: ")]
    body = committed[committed.index("-- drift: "):]
    # today's render carries migration 0027's work too (tests/test_registry_views_0027.py pins it): `v_run_current` at 0027's text and, from
    # its anchor-kind commit on, the run_contrast rebuild -- `_sans_fu4_render` takes the render back to before that file
    assert body.strip() == _sans_fu4_render(rendered).strip()
    assert body.count("DROP VIEW IF EXISTS ") == 3 and body.count("\nCREATE VIEW ") == 3
    for v in TF_VIEWS:
        code = S.VIEW_SQL[v].strip() if v != "v_run_current" else _pre_0027_run_current(S.VIEW_SQL[v].strip())
        assert "DROP VIEW IF EXISTS " + v + ";\n" + code in body, v
    for absent in ("__new", "INSERT INTO", "ALTER TABLE", "CREATE TABLE", "DROP TABLE", "CREATE INDEX", "CREATE TRIGGER"):
        assert absent not in body, absent
    assert SLUG not in M.POST_APPLY                                                        # no Python post-step: nothing is keyed by the slug
    # the objects before the file: no coexist exception, no check rule, no draws, no twin scope
    before = {v: conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (v,)).fetchone()[0] for v in TF_VIEWS}
    assert S.COEXIST_ACROSS_VIEWS_SQL not in before["v_run_current"] and "migration 0026" not in before["v_run_current"]
    assert "check_rule" not in before["v_multi_source_rows"] and "n_draws" not in before["v_multi_source_rows"] and "sampled_static_8192/" not in before["v_multi_source_rows"]
    assert "value_with_draws" not in before["v_multi_source_check"] and "n_draws_owner" not in before["v_multi_source_check"]
    # v_paper_long / v_paper_current read the check view and do not move
    for v in ("v_paper_long", "v_paper_current"):
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (v,)).fetchone()[0] == S.VIEW_SQL[v].strip().rstrip(";"), v
    conn.close()
    # the CLI's plan on a COPY: the pending file, one recreate line naming the three, nothing else, exit 1
    root = str(tmp_path / "copyroot")
    os.makedirs(root)
    shutil.copy(path, DB.db_path(root))
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert f"schema version {version - 1}; code {M.code_version()}; pending: {version:04d}_{SLUG}" in out
    assert "recreate view(s): " + ", ".join(sorted(TF_VIEWS)) in out
    for d in TF_DRIFT:
        assert f"  {d}" in out
    for absent in ("add column", "drop undeclared", "recreate index"):
        assert absent not in out, out
    assert [ln for ln in out.splitlines() if ln.startswith("recreate trigger")] == [], out   # 0030's two fence triggers ride 0031's run rebuild
    assert [ln for ln in out.splitlines() if ln.startswith("create table") and ln.split()[-1] not in AF_TABLES] == [], out   # 0029's five creates aside
    assert [ln for ln in out.splitlines() if ln.startswith("rebuild table ") and "run_contrast" not in ln and ln.split()[2].rstrip(":") not in NTPL_TABLES] == [], out   # 0027's and 0031's rebuild lines aside
    assert sum(1 for ln in out.splitlines() if ln.startswith("recreate ")) == 1                 # the views' line alone
    # applied (0027, 0028 and 0029 follow): the code's version, zero drift, the three objects at the code's text, every paper view selectable
    conn = DB.connect(path)
    assert M.apply_migrations(conn) == list(range(version, M.code_version() + 1)) and M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []
    for v in TF_VIEWS:
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (v,)).fetchone()[0] == S.VIEW_SQL[v].strip().rstrip(";"), v
    for v in TF_VIEWS + ("v_paper_long", "v_paper_current"):
        assert conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0] == 0, v
    conn.close()


def test_the_rules_render_from_their_tables_and_the_declarations_are_the_plans():
    assert V.COEXIST_ACROSS_VIEWS_BY_SOURCE == {"sampled_lane": ("pooled_t0_t1",)}
    assert S.COEXIST_ACROSS_VIEWS_SQL == ("NOT ((r.lane_source_legacy = 'sampled_lane' AND r2.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('pooled_t0_t1')"
                                          " AND r2.view_legacy IS NOT r.view_legacy))")
    assert S.VIEW_SQL["v_run_current"].count(S.COEXIST_ACROSS_VIEWS_SQL) == 1
    assert S.coexist_across_views_sql("a", "b").startswith("NOT ((a.lane_source_legacy = 'sampled_lane' AND b.lane_source_legacy")
    assert OW.TWIN_OWNERSHIP == (DECL_NONE, DECL_RELU) and all(d in OW.declarations() for d in OW.TWIN_OWNERSHIP)
    for d in OW.TWIN_OWNERSHIP:
        assert (d.owner_source, d.secondary_source, d.pairing_rule, d.check_rule, d.eval_category) == ("sampled_lane", "r2_export", "view_legacy", "value_with_draws", "static_attack")
        assert d.pairing_key_sql() == "(COALESCE(r.view_legacy, '') || '|' || r.instrument_id)" and d.served_operating_point_id is None and d.exclude_eval_ids == ()
        assert d.token_cap_tokens == 8192 and d.owner_pass_labels == ("pooled_t0_t1",)
        assert "\n    AND r.token_cap_tokens = 8192" in d.scope_sql() and "AND (r.lane_source_legacy = 'r2_export' OR r.pass_label IN ('pooled_t0_t1'))" in d.scope_sql()
    assert DECL_NONE.model_ids == ("gemma4e4b_it", "qwen35_4b_it", "llama8b_it") and DECL_NONE.transfer_kind == "none" and DECL_NONE.strength_rule == "unfitted"
    assert DECL_RELU.model_ids == ("qwen35_4b_it",) and DECL_RELU.transfer_kind == "instruct_direct" and DECL_RELU.strength_rule == "nominal_one"
    # nominal_one admits a run filing no operating-point id at nominal 1.0 and no other (the 0.5-nominal dose runs stay out; a run filing an id stays out)
    conn = DB.connect_memory()
    probe = lambda op, nominal: conn.execute(f"SELECT {DECL_RELU.strength_sql('r')} FROM (SELECT ? AS operating_point_id_legacy, ? AS strength_nominal) r", (op, nominal)).fetchone()[0]  # noqa: E731
    assert probe(None, 1.0) == 1 and probe(None, 1.0 + 1e-12) == 1 and probe(None, 0.5) == 0 and probe(None, None) in (0, None) and probe("ladder_canon/lam1", 1.0) == 0
    conn.close()
    # the rendered rows view carries the check rule and the draws beside every block; the check view branches on the rule
    rows = S.VIEW_SQL["v_multi_source_rows"]
    assert rows.count("AS check_rule,") == len(OW.declarations()) and rows.count("json_extract(rm.extra_json, '$.n_draws') AS n_draws,") == len(OW.declarations())
    assert rows.count("'value_with_draws' AS check_rule,") == 2 and rows.count("'counts' AS check_rule,") == len(OW.declarations()) - 2
    check = S.VIEW_SQL["v_multi_source_check"]
    assert "s.check_rule = 'counts' AND" in check and "s.check_rule = 'value_with_draws' AND" in check and "ow.n_draws AS n_draws_owner" in check
    # the owner's adapter outranks the secondary's on both twin scopes
    rows_p = {r["declaration_id"]: r for r in OW.check_priorities(ADP.load_all())}
    for d in OW.TWIN_OWNERSHIP:
        r = rows_p[d.declaration_id]
        assert r["owner_priorities"] == {"sampled_static": 25} and r["secondary_priorities"] == {"r2_export": 20}


# --------------------------------------------------------------------------------------------
# the pairwise coexistence (Q11)
# --------------------------------------------------------------------------------------------


def _pooled(root, source, i0, view, name):
    """One complete pooled_t0_t1 run of ``source`` at 8,192 on the `none` object at the corpus's one key, tagged ``view`` (NULL when None)."""
    _, rid = T22._family(root, source, i0, members=(), pooled="pooled_t0_t1", view=view, name=name)
    return rid


def test_two_different_tag_pooled_runs_of_the_sampled_source_coexist_and_a_later_same_tag_one_still_supersedes(tmp_path):
    root = T22._open(tmp_path)
    a = _pooled(root, "sampled_lane", 5000, TAG_A, "ta")
    b = _pooled(root, "sampled_lane", 6000, TAG_B, "tb")                   # later, another tag: a second POINT's reading, not a successor
    conn = T22._conn(root)
    st = T21._states(conn, [a, b])
    assert st[a] == T22.SELECTED and st[b] == T22.SELECTED, st
    conn.close()
    # a later SAME-tag pooled run retires its predecessor (the exception does not apply when the views are equal)
    a2 = _pooled(root, "sampled_lane", 7000, TAG_A, "ta2")
    conn = T22._conn(root)
    st = T21._states(conn, [a, a2, b])
    assert st[a] == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)") and st[a2] == T22.SELECTED and st[b] == T22.SELECTED, st
    conn.close()
    # two UNLABELED pooled runs of the sampled source pair as before (NULL pairs with NULL: the later supersedes); the labeled ones stand
    # (an unlabeled run's view differs from a labeled one's, so neither side retires the other)
    n1 = _pooled(root, "sampled_lane", 10000, None, "n1")
    n2 = _pooled(root, "sampled_lane", 11000, None, "n2")
    conn = T22._conn(root)
    st = T21._states(conn, [n1, n2, a2, b])
    assert st[n1][0] == "superseded" and st[n2] == T22.SELECTED and st[a2] == T22.SELECTED and st[b] == T22.SELECTED, st
    conn.close()
    # the same two-tag shape under another source keeps the later-run rule -- in a corpus of its own: the run-identity key has no source term,
    # so a later run of ANOTHER source at the sampled runs' key would retire them too (the cross-source dedupe the rule keeps)
    # (`relu_native` here: a campaign source. `strength_mtdiff_s1`, the spelling this test used to borrow, is a source-local-anchor source since
    # migration 0027 -- its `none` reads source_local_anchor, never selected -- and the point here is the later-run rule, not that frame rule)
    other = T22._open(tmp_path / "other")
    sa = _pooled(other, "relu_native", 8000, TAG_A, "sa")
    sb = _pooled(other, "relu_native", 9000, TAG_B, "sb")
    conn = T22._conn(other)
    st = T21._states(conn, [sa, sb])
    assert st[sa] == ("superseded", 0, 0, "selection_state superseded (a later run at the same key)") and st[sb] == T22.SELECTED, st
    conn.close()


# --------------------------------------------------------------------------------------------
# the twin fold through the views
# --------------------------------------------------------------------------------------------


def _scope_dims():
    t = synth.t
    model = dict(synth.seed_dimensions()[2]["payload"])
    model["model_id"] = MODEL_ID
    return [
        synth.rec("model", model, t(30)),
        synth.rec("eval", {"eval_id": EVAL_ID, "eval_category": "static_attack", "headline_metric_id": "asr", "metric_direction": "lower_is_better",
                           "attacker_required": 0, "paper_status": "in_paper"}, t(31)),
        synth.rec("item_set", {"item_set_id": ITEM_SET_ID, "eval_id": EVAL_ID, "dataset_version": "synthetic/static@v1", "n_items": N_GOALS,
                               "items_sha256": synth.SHA256_B, "source_uri": "synthetic://items/static_259.jsonl", "item_id_scheme": "g<index>",
                               "subsample_rule": "full"}, t(32)),
        synth.rec("instrument", INST, t(33)),
        synth.rec("instrument", INST_4K, t(38)),
        synth.rec("computation", {"computation_id": EXPORT_COMP, "algorithm": "synthetic.export_build", "version": "1", "seed_provenance": "none",
                                  "code_sha": synth.CODE_SHA}, t(34)),
        synth.rec("computation", {"computation_id": READER_COMP, "algorithm": "synthetic.pooled_stats", "version": "1", "seed_provenance": "none",
                                  "code_sha": synth.CODE_SHA}, t(35)),
        # the roster admits the `none` runs of this model / eval / instrument on the kind: both sources' runs are paper rows
        synth.rec("roster_entry", {"roster_entry_id": "re_twin", "round_id": "r_synth", "comparison_id": "baseline_static", "model_id": MODEL_ID,
                                   "object_id": "none", "eval_id": EVAL_ID, "instrument_id": INST_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
                                   "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}, t(36)),
        synth.rec("roster_entry", {"roster_entry_id": "re_twin_4k", "round_id": "r_synth", "comparison_id": "baseline_static_4k", "model_id": MODEL_ID,
                                   "object_id": "none", "eval_id": EVAL_ID, "instrument_id": INST_4K_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
                                   "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}, t(37)),
    ]


def _run(campaign, key, source, i0, *, view, pooled, inst=None, cap=8192, pass_label=None, **over):
    """One judged `none` run of the scope under ``source`` at the 8,192 static instrument (or ``inst`` at ``cap``), tagged ``view``; ``pooled``
    files it under pass_label pooled_t0_t1 (the reader's reading; ``pass_label`` names another of the reader's passes) -- the export's run
    carries no pass label, so the two are distinct run keys and never successors."""
    inst = inst or INST
    inst_id = str(inst["instrument_id"])
    rid = I.run_id(campaign, key, "none", EVAL_ID, inst_id)
    p = synth.run_payload(key, "none", "none", i0)
    p.update(run_id=rid, campaign=campaign, model_id=MODEL_ID, eval_id=EVAL_ID, item_set_id=ITEM_SET_ID, instrument_id=inst_id, token_cap_tokens=cap,
             lane_source_legacy=source, view_legacy=view, n_items_planned=N_GOALS)
    if pooled:
        p["pass_label"] = "pooled_t0_t1"
    elif pass_label:
        p["pass_label"] = pass_label
    p.update(over)
    t, host = synth.t, synth.TESTHOST
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs = [
        synth.rec("run", p, t(i0)),
        synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
        synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log", "scheduler_state": "RUNNING"}, t(i0 + 1)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": host, "cluster_job_id": "4242", "gpu_count": 1}, "verify": {"serving_path": "lora_native"}}, t(i0 + 2)),
        synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                "evidence_uri": f"synthetic:///store/{key}/results.json"}, t(i0 + 3)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": N_GOALS, "store_completion_artifact_uri": f"synthetic:///store/{key}/results.json",
                                                          "store_completion_sha256": synth.SHA256_B}}, t(i0 + 4)),
        synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": t(i0 + 5), "finished_utc": t(i0 + 6),
                               "n_items_judged": N_GOALS, "status": "complete", "api_pool": "lowprio"}, t(i0 + 6)),
        synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"}, t(i0 + 7)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": N_GOALS, "judge_id": synth.JUDGE_ID}}, t(i0 + 8)),
    ]
    return rid, jid, recs


def _metric(rid, jid, metric, thr, value, k, n, i, **cols):
    p = {"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr, "value": value, "k": k, "n": n, "computed_by": "producer"}
    p.update(cols)
    return synth.rec("run_metric", p, synth.t(i))


def _corpus(tmp_path, *, owner_value_at_0p75=None, owner_draws=N_DRAWS, other_tag_owner=True):
    """The twin pair: the export's run (tag A; asr@0.25 / 0.5 / 0.75 with k and n = the draws, plus a degeneration share) and the pooled reader's
    run of the SAME tag (asr@0.5 / 0.75, k NULL, n the goals, `extra.n_draws` the draws, the values equal to the export's unless
    ``owner_value_at_0p75`` says otherwise; ``owner_draws`` None files no draws) -- plus, by default, a pooled run of ANOTHER tag at the same key
    (it pairs with nothing on the export side). Returns the root and the ids."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=4)                     # synth's dimensions (the round, the `none` object) + one out-of-scope run
    T21._write(root, "dimensions", "000002-scope.jsonl", _scope_dims())
    ex, exj, exr = _run(SECOND_CAMPAIGN, "export_store", "r2_export", 1000, view=TAG_A, pooled=False)
    ex_metrics = [_metric(ex, exj, "asr", thr, k / N_DRAWS, k, N_DRAWS, 1010 + j, computation_id=EXPORT_COMP) for j, (thr, k) in enumerate(sorted(EXPORT_K.items()))]
    ex_metrics.append(_metric(ex, exj, "defended_collapse_share_default", -1.0, 12 / N_DRAWS, 12, N_DRAWS, 1020, computation_id=EXPORT_COMP))
    synth.write_run_dir(root, SECOND_CAMPAIGN, ex, [(str(r["record_type"]), [r]) for r in exr] + [("run_metric", ex_metrics)])
    po, poj, por = _run(OWNER_CAMPAIGN, "pooled_store_a", "sampled_lane", 2000, view=TAG_A, pooled=True)
    extra = {"member_run_ids": ["m1", "m2"], "n_draws": owner_draws} if owner_draws is not None else {"member_run_ids": ["m1", "m2"]}
    v75 = (EXPORT_K[0.75] / N_DRAWS) if owner_value_at_0p75 is None else owner_value_at_0p75
    po_metrics = [_metric(po, poj, "asr", 0.5, EXPORT_K[0.5] / N_DRAWS, None, N_GOALS, 2010, computation_id=READER_COMP, extra=extra),
                  _metric(po, poj, "asr", 0.75, v75, None, N_GOALS, 2011, computation_id=READER_COMP, extra=extra)]
    synth.write_run_dir(root, OWNER_CAMPAIGN, po, [(str(r["record_type"]), [r]) for r in por] + [("run_metric", po_metrics)])
    ids = {"export_run": ex, "pooled_run": po, "export_judgment": exj, "pooled_judgment": poj}
    # the export's 4,096-token reading of the SAME tag + eval (its own instrument; n the goals) -- live, every tag has one beside its 8,192 run
    e4, e4j, e4r = _run(SECOND_CAMPAIGN, "export_store_4k", "r2_export", 4000, view=TAG_A, pooled=False, inst=INST_4K, cap=4096)
    e4_metrics = [_metric(e4, e4j, "asr", thr, k / N_GOALS, k, N_GOALS, 4010 + j, computation_id=EXPORT_COMP) for j, (thr, k) in enumerate(sorted({0.25: 40, 0.5: 30, 0.75: 20}.items()))]
    synth.write_run_dir(root, SECOND_CAMPAIGN, e4, [(str(r["record_type"]), [r]) for r in e4r] + [("run_metric", e4_metrics)])
    ids["export_4k_run"] = e4
    # an 8,192-token export of a tag with NO pooled twin: in the scope, no owner row -- a stopgap that keeps its value (its own trial index: the
    # run-identity key has no tag term, so a later export of another tag at the very same key would supersede the first -- the rule, not the fold)
    ec, ecj, ecr = _run(SECOND_CAMPAIGN, "export_store_c", "r2_export", 5000, view=TAG_C, pooled=False, trial_index=1)
    ec_metrics = [_metric(ec, ecj, "asr", 0.5, 100 / N_DRAWS, 100, N_DRAWS, 5010, computation_id=EXPORT_COMP)]
    synth.write_run_dir(root, SECOND_CAMPAIGN, ec, [(str(r["record_type"]), [r]) for r in ecr] + [("run_metric", ec_metrics)])
    ids["export_nopool_run"] = ec
    # the pooled reader's per-trial MEMBER at the pooled row's cap and instrument (ladder_rung): never an owner row of the scope
    mem, memj, memr = _run(OWNER_CAMPAIGN, "pooled_store_a_member", "sampled_lane", 6000, view=TAG_A, pooled=False, pass_label="ladder_rung")
    mem_metrics = [_metric(mem, memj, "asr", 0.5, 0.33, 85, N_GOALS, 6010, computation_id=READER_COMP)]
    synth.write_run_dir(root, OWNER_CAMPAIGN, mem, [(str(r["record_type"]), [r]) for r in memr] + [("run_metric", mem_metrics)])
    ids["member_run"] = mem
    if other_tag_owner:
        pb, pbj, pbr = _run(OWNER_CAMPAIGN, "pooled_store_b", "sampled_lane", 3000, view=TAG_B, pooled=True)
        pb_metrics = [_metric(pb, pbj, "asr", 0.5, 0.9, None, N_GOALS, 3010, computation_id=READER_COMP, extra={"n_draws": N_DRAWS}),
                      _metric(pb, pbj, "asr", 0.75, 0.8, None, N_GOALS, 3011, computation_id=READER_COMP, extra={"n_draws": N_DRAWS})]
        synth.write_run_dir(root, OWNER_CAMPAIGN, pb, [(str(r["record_type"]), [r]) for r in pbr] + [("run_metric", pb_metrics)])
        ids["other_tag_run"] = pb
    T21._ingest(root)
    return root, ids


def _check(conn, tag=TAG_A):
    return {(r["metric_id"], float(r["threshold"])): dict(r) for r in conn.execute("SELECT * FROM v_multi_source_check WHERE declaration_id = ? AND pairing_key = ?",
                                                                                   (DECL_NONE.declaration_id, f"{tag}|{INST_ID}"))}


def _long(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM v_paper_long WHERE model_id = ? AND eval_id = ? ORDER BY metric_id, threshold, lane_source_legacy", (MODEL_ID, EVAL_ID))]


def test_the_duplicate_identities_fold_onto_the_same_tag_pooled_row_and_the_unique_export_identities_stand(tmp_path):
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    assert set(chk) == {("asr", 0.25), ("asr", 0.5), ("asr", 0.75), ("defended_collapse_share_default", -1.0)}
    assert all(r["check_rule"] == "value_with_draws" and r["pairing_key"] == f"{TAG_A}|{INST_ID}" for r in chk.values())   # the tag AND the instrument pair
    # the 4,096-token export (its own instrument, cap 4096) is OUT of the 8,192 scope: no check row of its own, a plain current row with its value;
    # the pooled reader's member pass is no owner row; the export tag with no pooled twin is a stopgap that keeps its value
    scoped = {r[0] for r in conn.execute("SELECT run_id FROM v_multi_source_rows WHERE declaration_id = ?", (DECL_NONE.declaration_id,))}
    assert ids["export_4k_run"] not in scoped and ids["member_run"] not in scoped and ids["export_nopool_run"] in scoped
    plain = [dict(r) for r in _long(conn) if r["run_id"] == ids["export_4k_run"]]
    assert len(plain) == 3 and all(r["multi_source_role"] is None and r["multi_source_state"] is None and r["value"] is not None and r["in_current_paper"] == 1 for r in plain)
    nopool = _check(conn, TAG_C)
    assert set(nopool) == {("asr", 0.5)} and nopool[("asr", 0.5)]["state"] == "stopgap"
    nopool_row = [dict(r) for r in _long(conn) if r["run_id"] == ids["export_nopool_run"]]
    assert len(nopool_row) == 1 and nopool_row[0]["multi_source_state"] == "stopgap" and nopool_row[0]["value"] == pytest.approx(100 / N_DRAWS)
    assert conn.execute("SELECT COUNT(*) FROM v_multi_source_check WHERE declaration_id = ? AND (owner_rows > 1 OR secondary_rows > 1)", (DECL_NONE.declaration_id,)).fetchone()[0] == 0
    stop, degen = chk[("asr", 0.25)], chk[("defended_collapse_share_default", -1.0)]
    assert stop["state"] == degen["state"] == "stopgap" and stop["owner_run_id"] is None and stop["k_secondary"] == EXPORT_K[0.25] and stop["n_secondary"] == N_DRAWS
    for thr in (0.5, 0.75):
        f = chk[("asr", thr)]
        assert f["state"] == "folded", f["reason"]
        assert (f["owner_run_id"], f["secondary_run_id"]) == (ids["pooled_run"], ids["export_run"])
        assert f["k_owner"] is None and f["n_owner"] == N_GOALS and f["n_draws_owner"] == N_DRAWS and f["k_secondary"] == EXPORT_K[thr] and f["n_secondary"] == N_DRAWS
        assert abs(f["value_owner"] - f["value_secondary"]) <= 1e-9
        assert "value agrees to 1e-9" in f["reason"] and "pooled draws" in f["reason"] and f"n_draws={N_DRAWS}" in f["reason"]
        assert json.loads(f["carried_json"]) == []                                       # the export carries no degeneracy sidecar for these rows
    # the frame: at 0.5 / 0.75 the pooled row is the ONE paper row (folded, naming the export run); at 0.25 and the degeneration share the export's stands
    # the paper rows (in_current_paper 1) by identity: the reader's ladder_rung member at the same tag / instrument is in the frame as a
    # member_pass row (not a paper row) and outside the scope (no owner row), so it is set aside here and asserted below
    by_key = {}
    for r in _long(conn):
        if r["in_current_paper"] == 1:
            by_key.setdefault((r["metric_id"], float(r["threshold"]), r["view_legacy"], r["instrument_id"]), []).append(r)
    member = [dict(r) for r in _long(conn) if r["run_id"] == ids["member_run"]]
    assert len(member) == 1 and member[0]["selection_state"] == "member_pass" and member[0]["in_current_paper"] == 0 and member[0]["multi_source_role"] is None
    for thr in (0.5, 0.75):
        rows = by_key[("asr", thr, TAG_A, INST_ID)]
        assert [r["lane_source_legacy"] for r in rows] == ["sampled_lane"] and rows[0]["multi_source_state"] == "folded" and rows[0]["multi_source_secondary_run_id"] == ids["export_run"]
        assert rows[0]["in_current_paper"] == 1 and rows[0]["value"] == pytest.approx(EXPORT_K[thr] / N_DRAWS) and rows[0]["k"] is None and rows[0]["n"] == N_GOALS
    for key in (("asr", 0.25, TAG_A, INST_ID), ("defended_collapse_share_default", -1.0, TAG_A, INST_ID)):
        rows = by_key[key]
        assert [r["lane_source_legacy"] for r in rows] == ["r2_export"] and rows[0]["multi_source_state"] == "stopgap" and rows[0]["in_current_paper"] == 1
    # the other tag's pooled rows pair with nothing: plain owner rows of their own point, both tags' pooled runs current (the Q11 exception)
    other = [r for r in _long(conn) if r["run_id"] == ids["other_tag_run"]]
    assert len(other) == 2 and all(r["multi_source_role"] == "owner" and r["multi_source_state"] is None and r["in_current_paper"] == 1 for r in other)
    states = {r[0]: r[1] for r in conn.execute("SELECT run_id, selection_state FROM v_run_current WHERE run_id IN (?, ?, ?, ?)", (ids["export_run"], ids["pooled_run"], ids["other_tag_run"], ids["export_4k_run"]))}
    assert states == {ids["export_run"]: "selected", ids["pooled_run"]: "selected", ids["other_tag_run"]: "selected", ids["export_4k_run"]: "selected"}
    # the live module's post-fold assertions, on this fixture (they skip live until the migration is applied to the live registry) -- and they
    # read v_paper_long in exactly ONE statement (the temp-table fold) however many pairs they are handed, and leave no temp table behind
    seen = []
    conn.set_trace_callback(seen.append)
    assert_post_fold(conn, [{"export_run": ids["export_run"], "pooled_run": ids["pooled_run"]}] * 3)
    conn.set_trace_callback(None)
    assert sum("v_paper_long" in stmt for stmt in seen) == 1, [stmt[:120] for stmt in seen if "v_paper_long" in stmt]
    assert conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE type = 'table'").fetchone()[0] == 0
    conn.close()


def test_a_same_tag_disagreement_or_an_owner_without_draws_refuses_and_never_folds(tmp_path):
    root, ids = _corpus(tmp_path, owner_value_at_0p75=0.123, other_tag_owner=False)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    assert chk[("asr", 0.5)]["state"] == "folded" and chk[("asr", 0.75)]["state"] == "refused"
    ref = chk[("asr", 0.75)]
    assert "disagree under value_with_draws" in ref["reason"] and "value=0.123" in ref["reason"] and "neither value is admitted" in ref["reason"] and ref["carried_json"] == "[]"
    rows = [r for r in _long(conn) if r["metric_id"] == "asr" and float(r["threshold"]) == 0.75 and r["run_id"] in (ids["export_run"], ids["pooled_run"])]
    assert [r["lane_source_legacy"] for r in rows] == ["sampled_lane"] and rows[0]["multi_source_state"] == "refused" and rows[0]["value"] is None   # the owner row keeps the identity, valueless
    conn.close()
    root, ids = _corpus(tmp_path / "nodraws", owner_draws=None, other_tag_owner=False)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    assert {k: v["state"] for k, v in chk.items() if k[0] == "asr"} == {("asr", 0.25): "stopgap", ("asr", 0.5): "refused", ("asr", 0.75): "refused"}
    assert all(v["n_draws_owner"] is None and "n_draws=null" in v["reason"] for k, v in chk.items() if k in (("asr", 0.5), ("asr", 0.75)))
    conn.close()
