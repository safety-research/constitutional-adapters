"""Multi-source ownership of one measurement (`sisteer.registry.ownership`, the paper's `config.MULTI_SOURCE_CELL_OWNERSHIP`
at paper-figures 982c26c8): two sources read the same measurement — the r2 export (owner) and the all-positions refusal steer's own
reader (secondary) — and the registry must resolve the pair to ONE row of `v_paper_long`, never silently.

Synthetic owner + secondary rows (invented ids, hostname TESTHOST; the scope spellings are the declaration's config strings, not
recorded data) exercise every state:

* STOPGAP — the owner carries no row at the identity: the secondary's row stands, its note says so.
* FOLDED — k and n agree exactly: the owner row stands, the secondary row leaves the frame, `degenerate_share` / `loop_share`
  travel onto the owner where it had none, `at_cap_share` NEVER does (the owner's own figure stands, not the secondary's re-tokenized
  reading — the preference is scoped to the row's own run).
* REFUSED — k differs (or the owner already carries a different carry reading): visible in `v_multi_source_check`, on the status
  page and in `registry status --json`; the owner row keeps its identity with value / k / n / ci NULL and `status = refused: …`;
  nothing resolves it by priority.

`v_paper_long` never carries the identity twice; both views are run-scale so the frame stays in the run-level snapshot; the
adapters' declared priorities put the owner above the secondary; the SQL's carried readings equal the frame's own derivation of
the secondary's row.

The run-pairing contract (migration 0005): a declaration is per TRANSFER KIND, admits runs by a STRENGTH RULE (the served operating
point, or unfitted), may EXCLUDE eval ids of its category by name, and pairs an owner run with a secondary run by a PAIRING RULE
rendered as `pairing_key` (the producer run key both sources spell alike, or '' when the identity alone pairs); BOTH sides are their
source's own computation rows only, so a stats reader's blank-computation twin filed onto a scoped run never counts. Exercised on a
second synthetic scope shaped like the Llama-3.1-8B Petri measurements, where the model's own reader OWNS and the export is secondary
(paperlib's rule of 2026-09-14): one paper label, TWO fitted objects under two transfer kinds, both
sources sharing the producer key per object, the twin on every export run, one identity whose `n` differs (the reader's 9 against the
export's 10) and a pooled eval left out of scope — six declarations of that shape are in force (`llama8b_it/<label>/petri/<transfer_kind>`).
"""
from __future__ import annotations

import json
import os
import re

import pytest

from sisteer.registry import adapters as ADP
from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import ownership as OW
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import snapshot as SN
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry import reads_protocol as RP

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "registry", "paper_load_columns_f33dca5d.json")

DECL = OW.BY_ID["qwen35_4b_it/arditi_allpos/multi_turn"]
MODEL_ID = DECL.model_ids[0]                      # the declaration's scope spelling
LABEL = DECL.paper_label
OP_ID = DECL.served_operating_point_id
OWNER_SRC, SECOND_SRC = DECL.owner_source, DECL.secondary_source
OWNER_CAMPAIGN, SECOND_CAMPAIGN = "synth_owner_campaign", "synth_secondary_campaign"
EVAL_ID = "synth_mt"                              # an invented multi-turn eval
ITEM_SET_ID = "synth_mt_259"
OBJ_SHA = "d" * 64
OBJ_KEY = f"{LABEL}@synth"
OWNER_AT_CAP, SECOND_AT_CAP = 0.07, 0.2            # two statistics under one paper column: the export's figure vs the sidecar's re-tokenized
SECOND_DEGEN, SECOND_LOOP = 0.3, 0.1
#: every declared source files its metric rows under a computation (the export's build; a reader's stats pass — no blank-computation
#: key on any of them on the live registry); BOTH sides of the fold are those rows only, so every synthetic scoped row names one too
EXPORT_COMP = I.computation_id("synthetic.export_build", "1", "none")
READER_COMP = I.computation_id("synthetic.reader_stats", "1", "none")


def _export_computation(i: int):
    return synth.rec("computation", {"computation_id": EXPORT_COMP, "algorithm": "synthetic.export_build", "version": "1", "seed_provenance": "none",
                                     "code_sha": synth.CODE_SHA}, synth.t(i))


def _reader_computation(i: int):
    return synth.rec("computation", {"computation_id": READER_COMP, "algorithm": "synthetic.reader_stats", "version": "1", "seed_provenance": "none",
                                     "code_sha": synth.CODE_SHA}, synth.t(i))


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _instrument(tag: str):
    return synth.instrument_fields(eval_id=EVAL_ID, item_set_id=ITEM_SET_ID, token_cap_tokens=4096, token_cap_ladder_json="[4096]",
                                   max_turns=10, harness_config_hash=tag * 8)


INST_OWNER = _instrument("o")
INST_SECOND = _instrument("s")


def _scope_dims():
    """The scope's dimensions beside synth's: the model, a multi-turn eval + item set, two instruments (the two readers mint
    different instruments in production, so recency never supersedes one with the other), the object under the declared label."""
    t = synth.t
    model = dict(synth.seed_dimensions()[2]["payload"])       # the instruct model record, re-keyed to the scope's model id
    model["model_id"] = MODEL_ID
    return [
        synth.rec("model", model, t(30)),
        synth.rec("eval", {"eval_id": EVAL_ID, "eval_category": "multi_turn", "headline_metric_id": "asr", "metric_direction": "lower_is_better",
                           "attacker_required": 1, "paper_status": "in_paper"}, t(31)),
        synth.rec("item_set", {"item_set_id": ITEM_SET_ID, "eval_id": EVAL_ID, "dataset_version": "synthetic/mt@v1", "n_items": 259,
                               "items_sha256": synth.SHA256_B, "source_uri": "synthetic://items/mt_259.jsonl", "item_id_scheme": "g<index>",
                               "subsample_rule": "full"}, t(32)),
        synth.rec("instrument", INST_OWNER, t(33)),
        synth.rec("instrument", INST_SECOND, t(34)),
        synth.rec("object", synth.object_lora(OBJ_SHA, paper_label=LABEL, role="baseline_published"), t(35)),
        _export_computation(36),
        _reader_computation(37),
    ]


def _run(campaign: str, key: str, inst, source: str, i0: int):
    """One judged run of the scope under `source`, at the served operating point (synth.run_records' lifecycle, re-keyed)."""
    rid = I.run_id(campaign, key, OBJ_KEY, EVAL_ID, str(inst["instrument_id"]))
    p = synth.run_payload(key, OBJ_KEY, OBJ_SHA, i0)
    p.update(run_id=rid, campaign=campaign, model_id=MODEL_ID, eval_id=EVAL_ID, item_set_id=ITEM_SET_ID, instrument_id=str(inst["instrument_id"]),
             transfer_kind="instruct_direct", multi_turn_steering_scope="every_turn_prefill_and_decode", token_cap_tokens=4096,
             lane_source_legacy=source, operating_point_id_legacy=OP_ID, strength_coefficient=0.3, strength_nominal=1.0,
             strength_nominal_source="derived", is_operating_point=1, n_items_planned=259)
    t, host = synth.t, synth.TESTHOST
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs = [
        synth.rec("run", p, t(i0)),
        synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
        synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log", "scheduler_state": "RUNNING"}, t(i0 + 1)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": host, "cluster_job_id": "4242", "gpu_count": 1},
                                "verify": {"serving_path": "lora_native"}}, t(i0 + 2)),
        synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                "evidence_uri": f"synthetic:///store/{key}/results.json"}, t(i0 + 3)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 259, "store_completion_artifact_uri": f"synthetic:///store/{key}/results.json",
                                                          "store_completion_sha256": synth.SHA256_B}}, t(i0 + 4)),
        synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": t(i0 + 5), "finished_utc": t(i0 + 6),
                               "n_items_judged": 259, "status": "complete", "api_pool": "lowprio"}, t(i0 + 6)),
        synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"}, t(i0 + 7)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 259, "judge_id": synth.JUDGE_ID}}, t(i0 + 8)),
    ]
    return rid, jid, recs


def _metric(rid, jid, metric, thr, k, n, i, **cols):
    p = {"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr, "value": k / n, "k": k, "n": n, "computed_by": "producer"}
    p.update(cols)
    return synth.rec("run_metric", p, synth.t(i))


def _degeneration(rid, jid, measure, rate, n, i, unit="replies"):
    k = int(round(rate * n))
    return synth.rec("run_degeneration", {
        "run_id": rid, "judgment_id": jid, "measure": measure, "scope": "all", "measurement_state": "measured", "numerator": k, "denominator": n,
        "denominator_unit": unit, "pooling_json": {"unit": unit}, "aggregation": "union" if measure == "union5" else "single_detector",
        "rate": k / n, "definition_version": "synthetic_sidecar/1", "threshold_params_json": {}, "computed_by": "producer",
        "computed_utc": synth.t(i), "source_uri": "synthetic://sidecar", "notes": None}, synth.t(i))


def _corpus(tmp_path, *, owner=True, owner_degenerate=None, owner_k_at_0p75=4, secondary_rows=None):
    """The scope's two readers. Owner rows (r2_export): asr@0.5, asr@0.75 (k differs from the secondary's → refused), asr_held_in;
    NO asr@0.25 (→ the secondary's is a stopgap). Secondary rows (arditi_allpos): all four, every one carrying the sidecar's
    degenerate_share / loop_share (its loop_share names the classifier detector) and its re-tokenized at-cap figure."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=4)                      # synth's dimensions + one out-of-scope run (a control)
    _write(root, "dimensions", "000002-scope.jsonl", _scope_dims())
    owner_extra = {"loop_share_measure_legacy": "rep_ratio_loop", "status_legacy": "graded", "note": "owner note"}
    second_extra = {"loop_share_measure_legacy": "classifier"}
    second_cols = dict(frac_at_token_cap=SECOND_AT_CAP, frac_degenerate_union5=SECOND_DEGEN, frac_degenerate_classifier=SECOND_LOOP, extra=second_extra,
                       computation_id=READER_COMP)
    ids = {}
    if owner:
        orid, ojid, orecs = _run(OWNER_CAMPAIGN, "owner_store", INST_OWNER, OWNER_SRC, 1000)
        ocols = dict(frac_at_token_cap=OWNER_AT_CAP, extra=owner_extra, computation_id=EXPORT_COMP)
        if owner_degenerate is not None:
            ocols["frac_degenerate_union5"] = owner_degenerate
        metrics = [_metric(orid, ojid, "asr", 0.5, 10, 259, 1010, **ocols),
                   _metric(orid, ojid, "asr", 0.75, owner_k_at_0p75, 259, 1011, **ocols),
                   _metric(orid, ojid, "asr_held_in", -1.0, 6, 163, 1012, **ocols)]
        synth.write_run_dir(root, OWNER_CAMPAIGN, orid, [(str(r["record_type"]), [r]) for r in orecs] + [("run_metric", metrics)])
        ids.update(owner_run=orid, owner_judgment=ojid)
    srid, sjid, srecs = _run(SECOND_CAMPAIGN, "secondary_store", INST_SECOND, SECOND_SRC, 2000)
    rows = secondary_rows if secondary_rows is not None else [("asr", 0.25, 14, 259), ("asr", 0.5, 10, 259), ("asr", 0.75, 5, 259), ("asr_held_in", -1.0, 6, 163)]
    metrics = [_metric(srid, sjid, m, thr, k, n, 2010 + j, **second_cols) for j, (m, thr, k, n) in enumerate(rows)]
    # the sidecar's counts over the responses (a denominator on which every declared share is an integral count, as a writer's mirror
    # column is exactly k / n of its own reading)
    degs = [_degeneration(srid, sjid, "at_token_cap_retokenized", SECOND_AT_CAP, 4000, 2020),
            _degeneration(srid, sjid, "union5", SECOND_DEGEN, 4000, 2021), _degeneration(srid, sjid, "classifier", SECOND_LOOP, 4000, 2022)]
    synth.write_run_dir(root, SECOND_CAMPAIGN, srid, [(str(r["record_type"]), [r]) for r in srecs] + [("run_metric", metrics), ("run_degeneration", degs)])
    ids.update(secondary_run=srid, secondary_judgment=sjid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked)
    return root, ids


def _check(conn):
    return {(r["eval_id"], r["metric_id"], r["threshold"]): dict(r) for r in conn.execute("SELECT * FROM v_multi_source_check")}


def _long(conn):
    sql = "SELECT * FROM v_paper_long WHERE model_id = ? AND paper_label = ? ORDER BY lane_source_legacy, metric_id, threshold"
    return [dict(r) for r in conn.execute(sql, (MODEL_ID, LABEL))]


def _pinned_columns():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["columns"]


def _frame(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return paper.paper_long_frame(conn, columns=_pinned_columns(), current_only=False)
    finally:
        conn.close()


def _null(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


# --------------------------------------------------------------------------------------------
# the declaration and the priorities
# --------------------------------------------------------------------------------------------

def test_declaration_mirrors_the_papers_rule():
    d = DECL
    assert paper.MODEL_LEGACY[d.model_ids[0]] == "qwen35_4b" and d.paper_label == "arditi_allpos"
    assert paper.COMPONENT_LEGACY[d.eval_category] == "multi-turn"
    assert (d.owner_source, d.secondary_source) == ("r2_export", "arditi_allpos")
    assert d.match == ("pairing_key", "eval_id", "metric_id", "threshold") and d.check == ("k", "n")
    assert d.carry == ("degenerate_share", "loop_share") and d.never_carry == ("at_cap_share",)
    assert d.served_operating_point_id == "ar_c0p3__arditi_allpos" and d.strength_rule == "served_operating_point"
    # the scope's transfer kind is the one its runs file (instruct_direct, verified on the live registry); the two readers spell the
    # producer run key differently (the export the store's directory name, the reader a <source>/<ladder>/<store>/<model> path), so
    # the identity alone pairs: the fold the SQL rendered before the pairing key existed
    assert d.transfer_kind == "instruct_direct" and d.pairing_rule == "identity_only" and d.pairing_key_sql() == "''"
    assert d.exclude_eval_ids == (), "the whole multi-turn category is in scope"
    sql = d.scope_sql()
    assert "r.transfer_kind = 'instruct_direct'" in sql and "r.operating_point_id_legacy = 'ar_c0p3__arditi_allpos'" in sql and "r.strength_nominal = 1.0" in sql
    assert "NOT IN" not in sql
    # the SQL is generic over exactly these sets and rules: a declaration asking for others is refused at import, never silently ignored
    kw = dict(model_ids=("m",), paper_label="l", eval_category="c", transfer_kind="instruct_direct", owner_source="a", secondary_source="b",
              served_operating_point_id="op", since="2026-09-18", ruling="r", pairing_rule="identity_only")
    OW.Ownership("ok", **kw)
    for bad in (dict(carry=("at_cap_share",)), dict(check=("value",)), dict(match=("eval_id", "metric_id", "threshold")), dict(secondary_source="a"),
                dict(transfer_kind="sideways"), dict(pairing_rule="by_name"), dict(strength_rule="any"),
                dict(served_operating_point_id=None), dict(served_operating_point_id="  "), dict(strength_rule="unfitted"),
                dict(exclude_eval_ids=("",)), dict(exclude_eval_ids=("x", "x")), dict(exclude_eval_ids=(None,))):
        with pytest.raises(ValueError):
            OW.Ownership("bad", **dict(kw, **bad))
    # an excluded eval id renders as a NOT IN on the run's eval id, beside the category
    x = OW.Ownership("excl", **dict(kw, exclude_eval_ids=("pooled_x", "pooled_y")))
    assert "e.eval_category = 'c'\n    AND r.eval_id NOT IN ('pooled_x', 'pooled_y')" in x.scope_sql()
    # unfitted: no operating point to name, and the SQL admits a run filing neither an id nor a nominal
    u = OW.Ownership("unfitted", **dict(kw, served_operating_point_id=None, strength_rule="unfitted", transfer_kind="none"))
    assert u.strength_sql() == "(r.operating_point_id_legacy IS NULL AND r.strength_nominal IS NULL)"
    assert "r.transfer_kind = 'none'" in u.scope_sql() and u.strength_sql() in u.scope_sql()
    assert OW.STRENGTH_RULES == ("served_operating_point", "unfitted", "nominal_one") and OW.PAIRING_RULES == ("identity_only", "producer_run_key", "view_legacy")
    assert OW.CHECK_RULES == ("counts", "value_with_draws")
    # migration 0026's rules: a check rule outside the tuple is refused; nominal_one names no operating point; view_legacy pairs on the tag
    for bad in (dict(check_rule="exact"), dict(strength_rule="nominal_one")):
        with pytest.raises(ValueError):
            OW.Ownership("bad", **dict(kw, **bad))
    n1 = OW.Ownership("nominal_one", **dict(kw, served_operating_point_id=None, strength_rule="nominal_one", pairing_rule="view_legacy", check_rule="value_with_draws"))
    assert n1.strength_sql() == "(r.operating_point_id_legacy IS NULL AND ABS(r.strength_nominal - 1.0) <= 1e-9)" and n1.pairing_key_sql() == "(COALESCE(r.view_legacy, '') || '|' || r.instrument_id)"
    assert "token_cap_tokens" not in n1.scope_sql() and "pass_label" not in n1.scope_sql()        # undeclared: every cap, every owner pass
    for bad in (dict(token_cap_tokens=0), dict(token_cap_tokens=True), dict(token_cap_tokens="8192"), dict(owner_pass_labels=("sideways",)), dict(owner_pass_labels=("base_cap", "base_cap"))):
        with pytest.raises(ValueError):
            OW.Ownership("bad", **dict(kw, **bad))
    capped = OW.Ownership("capped", **dict(kw, token_cap_tokens=8192, owner_pass_labels=("pooled_t0_t1",)))
    assert "\n    AND r.token_cap_tokens = 8192\n    AND (r.lane_source_legacy = 'b' OR r.pass_label IN ('pooled_t0_t1'))" in capped.scope_sql()   # the secondary's rows, or an owner row under the pooled pass
    # the two twin scopes are the registry's own: the paper side's MULTI_SOURCE_CELL_OWNERSHIP mirrors the Qwen all-positions rule alone
    twins = [d for d in OW.declarations() if d.owner_source == "sampled_lane"]
    assert [d.declaration_id for d in twins] == ["sampled_static_8192/none/static_attack/none", "sampled_static_8192/constitution_relu/static_attack/instruct_direct"]
    assert all(d.pairing_rule == "view_legacy" and d.check_rule == "value_with_draws" and d.secondary_source == "r2_export" for d in twins)
    assert all(d.strength_rule in ("unfitted", "nominal_one") and d.served_operating_point_id is None for d in twins)


def test_adapter_priorities_owner_outranks_secondary():
    reg = ADP.load_all()
    rows = OW.check_priorities(reg)
    row = next(r for r in rows if r["declaration_id"] == DECL.declaration_id)
    assert row["owner_priorities"] == {"r2_export": IC.PRIORITY_PAPER_SOURCE} and row["secondary_priorities"] == {"arditi_allpos": IC.PRIORITY_SECONDARY_SOURCE}
    assert IC.PRIORITY_OWNING_READER > IC.PRIORITY_PAPER_SOURCE > IC.PRIORITY_SECONDARY_SOURCE > IC.PRIORITY_GENERIC_READER > IC.PRIORITY_MIRROR
    # a registry where the two sit at one priority is refused by name (the source-id tiebreak would decide)
    class Flat:
        name = "arditi_allpos"
        covers_sources = ("arditi_allpos",)
        priority = IC.PRIORITY_PAPER_SOURCE
    flat = dict(reg, arditi_allpos=lambda: Flat())
    with pytest.raises(ValueError, match="outrank"):
        OW.check_priorities(flat)


def test_views_read_run_degeneration_and_stay_in_the_run_level_snapshot():
    """Since migration 0010 the three views read `run_degeneration` (the derived degeneration rates beside the writers' mirrors) and no
    OTHER item-scale table; the run-level snapshot carries that one table for them (snapshot.RUNLEVEL_ITEM_TABLES), so they stay in it."""
    other_item = set(S.ITEM_SCALE_TABLES) - set(SN.RUNLEVEL_ITEM_TABLES)
    for v in ("v_multi_source_rows", "v_multi_source_check", "v_paper_long"):
        assert "run_degeneration" in S.VIEW_TABLES[v] and not S.VIEW_TABLES[v] & other_item, v
        assert v in SN.RUNLEVEL_VIEWS, v
    assert S.VIEW_TABLES["v_multi_source_check"] == frozenset({"run", "object", "eval", "run_metric", "judgment", "instrument", "observation", "run_degeneration",
                                                               "source_revision"})   # judgment since 0021; instrument since the record-refusal-fact migration (the identity bound at claim is compared to the run's current instrument and required judge)


# --------------------------------------------------------------------------------------------
# the three states, in the view and in v_paper_long
# --------------------------------------------------------------------------------------------

def test_check_view_states_stopgap_folded_refused(tmp_path):
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    assert set(chk) == {(EVAL_ID, "asr", 0.25), (EVAL_ID, "asr", 0.5), (EVAL_ID, "asr", 0.75), (EVAL_ID, "asr_held_in", -1.0)}
    stop, fold, ref, held = chk[(EVAL_ID, "asr", 0.25)], chk[(EVAL_ID, "asr", 0.5)], chk[(EVAL_ID, "asr", 0.75)], chk[(EVAL_ID, "asr_held_in", -1.0)]
    assert stop["state"] == "stopgap" and stop["owner_run_id"] is None and stop["owner_rows"] == 0 and stop["k_secondary"] == 14
    assert "stopgap" in stop["reason"] and OWNER_SRC in stop["reason"]
    assert fold["state"] == "folded" and (fold["k_owner"], fold["n_owner"], fold["k_secondary"], fold["n_secondary"]) == (10, 259, 10, 259)
    assert json.loads(fold["carried_json"]) == ["degenerate_share", "loop_share"]
    assert fold["degenerate_share_secondary"] == pytest.approx(SECOND_DEGEN) and fold["loop_share_secondary"] == pytest.approx(SECOND_LOOP)
    assert fold["degenerate_share_owner"] is None and fold["loop_share_owner"] is None
    assert fold["loop_share_measure_secondary"] == "classifier" and fold["loop_share_measure_owner"] == "rep_ratio_loop"
    # at_cap_share: both sides visible, two different statistics, never carried
    assert fold["at_cap_share_owner"] == pytest.approx(OWNER_AT_CAP) and fold["at_cap_share_secondary"] == pytest.approx(SECOND_AT_CAP)
    assert "at_cap_share" not in fold["carried_json"]
    assert held["state"] == "folded" and (held["k_owner"], held["n_owner"]) == (6, 163) and held["threshold"] == -1.0
    assert ref["state"] == "refused" and (ref["k_owner"], ref["k_secondary"]) == (4, 5) and ref["n_owner"] == ref["n_secondary"] == 259
    assert "disagree on k / n" in ref["reason"] and "k=4" in ref["reason"] and "k=5" in ref["reason"] and "never averaged" in ref["reason"]
    assert ref["carried_json"] == "[]"
    assert (fold["owner_run_id"], fold["secondary_run_id"]) == (ids["owner_run"], ids["secondary_run"])
    for st in OW.STATES:
        assert st in {r["state"] for r in chk.values()}
    conn.close()


def test_paper_long_carries_each_identity_once_and_marks_the_states(tmp_path):
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = _long(conn)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["eval_id"], r["metric_id"], r["threshold"]), []).append(r)
    # never twice: one row per identity, whatever the state
    assert {k: len(v) for k, v in by_key.items()} == {(EVAL_ID, "asr", 0.25): 1, (EVAL_ID, "asr", 0.5): 1, (EVAL_ID, "asr", 0.75): 1, (EVAL_ID, "asr_held_in", -1.0): 1}
    stop = by_key[(EVAL_ID, "asr", 0.25)][0]
    assert stop["lane_source_legacy"] == SECOND_SRC and stop["multi_source_role"] == "secondary" and stop["multi_source_state"] == "stopgap"
    assert stop["value"] == pytest.approx(14 / 259) and stop["k"] == 14 and "stopgap" in stop["multi_source_reason"]
    fold = by_key[(EVAL_ID, "asr", 0.5)][0]
    assert fold["lane_source_legacy"] == OWNER_SRC and fold["multi_source_role"] == "owner" and fold["multi_source_state"] == "folded"
    assert fold["value"] == pytest.approx(10 / 259) and fold["k"] == 10 and fold["n"] == 259
    assert fold["multi_source_secondary_run_id"] == ids["secondary_run"] and json.loads(fold["multi_source_carried_json"]) == ["degenerate_share", "loop_share"]
    assert fold["multi_source_degenerate_share"] == pytest.approx(SECOND_DEGEN) and fold["multi_source_loop_share"] == pytest.approx(SECOND_LOOP)
    assert fold["multi_source_loop_share_measure"] == "classifier"
    ref = by_key[(EVAL_ID, "asr", 0.75)][0]
    assert ref["lane_source_legacy"] == OWNER_SRC and ref["multi_source_state"] == "refused"
    assert ref["value"] is None and ref["k"] is None and ref["n"] is None and ref["ci_lo"] is None and ref["ci_hi"] is None
    assert "disagree on k / n" in ref["multi_source_reason"]
    # the secondary's folded and refused rows are gone from the frame; the owner's are the rows that stand
    assert all(r["lane_source_legacy"] == OWNER_SRC for k, rs in by_key.items() for r in rs if k != (EVAL_ID, "asr", 0.25))
    # the out-of-scope control run is untouched
    control = [dict(r) for r in conn.execute("SELECT multi_source_role, multi_source_state, value FROM v_paper_long WHERE model_id = ?", (synth.MODEL_ID,))]
    assert control and all(r["multi_source_role"] is None and r["multi_source_state"] is None and r["value"] is not None for r in control)
    # the two runs are not superseded by recency (different instruments, as the two readers mint in production): the fold is the ONLY dedupe
    states = {r[0]: r[1] for r in conn.execute("SELECT run_id, selection_state FROM v_paper_current WHERE run_id IN (?, ?)", (ids["owner_run"], ids["secondary_run"]))}
    assert set(states.values()) == {"not_in_roster"}
    conn.close()


def test_frame_carries_degeneracy_but_never_at_cap_share(tmp_path):
    root, ids = _corpus(tmp_path)
    df = _frame(root)
    scope = df[(df.model == "qwen35_4b") & (df.arm == LABEL)]
    assert len(scope) == 4 and set(scope.component) == {"multi-turn"}
    fold = scope[(scope.metric == "asr") & (scope.threshold == 0.5)].iloc[0]
    assert fold["source"] == OWNER_SRC and fold["multi_source_state"] == "folded"
    assert fold["degenerate_share"] == pytest.approx(SECOND_DEGEN) and fold["loop_share"] == pytest.approx(SECOND_LOOP)      # carried
    assert fold["at_cap_share"] == pytest.approx(OWNER_AT_CAP)                                                              # the owner's own figure
    assert fold["at_cap_share"] != pytest.approx(SECOND_AT_CAP)                                                             # never the sidecar's re-tokenized
    assert fold["status"] == "graded" and fold["value"] == pytest.approx(10 / 259) and fold["k"] == 10
    assert str(fold["note"]).startswith("owner note") and "FOLDED (multi-source ownership)" in str(fold["note"])
    assert "carried from" in str(fold["degeneracy_basis"]) and "classifier share" in str(fold["degeneracy_basis"])
    held = scope[scope.metric == "asr_held_in"].iloc[0]
    assert held["multi_source_state"] == "folded" and held["degenerate_share"] == pytest.approx(SECOND_DEGEN) and held["at_cap_share"] == pytest.approx(OWNER_AT_CAP)
    ref = scope[(scope.metric == "asr") & (scope.threshold == 0.75)].iloc[0]
    assert ref["source"] == OWNER_SRC and ref["multi_source_state"] == "refused"
    assert _null(ref["value"]) and _null(ref["k"]) and _null(ref["n"])
    assert str(ref["status"]).startswith("refused: ") and "disagree on k / n" in str(ref["status"])
    assert "REFUSED (multi-source ownership)" in str(ref["note"])
    assert _null(ref["degenerate_share"]) and _null(ref["loop_share"])                                                    # nothing carried onto a refusal
    stop = scope[(scope.metric == "asr") & (scope.threshold == 0.25)].iloc[0]
    assert stop["source"] == SECOND_SRC and stop["multi_source_state"] == "stopgap" and stop["value"] == pytest.approx(14 / 259)
    assert "STOPGAP ROW (multi-source ownership)" in str(stop["note"])
    # the secondary's re-tokenized at-cap figure reaches no owner row's at_cap_share
    owners = scope[scope.source == OWNER_SRC]
    assert not any(v == pytest.approx(SECOND_AT_CAP) for v in owners["at_cap_share"].tolist() if not _null(v))
    # every refused-NULL column names its reason, in the style of LONG_COLUMN_NULL_REASONS
    for c, why in paper.LONG_COLUMN_REFUSED_NULL_REASONS.items():
        assert c in paper.LONG_COLUMN_SOURCES and "refused" in why


def test_sql_carried_readings_equal_the_frames_derivation_of_the_secondary_row(tmp_path):
    """The check view derives the carried readings from run-scale columns (the writer's mirrors + extras); the frame derives the same
    paper columns for the secondary's OWN row through its full chain (run_degeneration rates on the headline row first). The two
    must agree, or the fold would carry a number the frame itself would not print."""
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    conn.close()
    df = _frame(root)
    stop = df[(df.source == SECOND_SRC) & (df.metric == "asr") & (df.threshold == 0.25)].iloc[0]     # the secondary's own row, standing as a stopgap
    view = chk[(EVAL_ID, "asr", 0.25)]
    assert stop["degenerate_share"] == pytest.approx(view["degenerate_share_secondary"])
    assert stop["loop_share"] == pytest.approx(view["loop_share_secondary"])
    assert stop["at_cap_share"] == pytest.approx(view["at_cap_share_secondary"]) == pytest.approx(SECOND_AT_CAP)


def test_owner_carrying_a_different_carry_reading_refuses(tmp_path):
    root, ids = _corpus(tmp_path, owner_degenerate=0.25)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    fold = chk[(EVAL_ID, "asr", 0.5)]
    assert fold["state"] == "refused" and "degenerate_share=0.25" in fold["reason"] and "never overwritten" in fold["reason"]
    assert fold["carried_json"] == "[]"
    rows = {(r["metric_id"], r["threshold"]): r for r in _long(conn)}
    assert rows[("asr", 0.5)]["multi_source_state"] == "refused" and rows[("asr", 0.5)]["value"] is None
    conn.close()


def test_all_stopgap_while_the_owner_has_no_rows(tmp_path):
    """Today's live state: the owner carries no row for any identity; every secondary row stands and says so."""
    root, ids = _corpus(tmp_path, owner=False)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    assert len(chk) == 4 and {r["state"] for r in chk.values()} == {"stopgap"}
    rows = _long(conn)
    assert len(rows) == 4 and all(r["multi_source_state"] == "stopgap" and r["value"] is not None for r in rows)
    conn.close()
    df = _frame(root)
    scope = df[(df.model == "qwen35_4b") & (df.arm == LABEL)]
    assert all("STOPGAP ROW" in str(n) for n in scope["note"])
    assert all(v == pytest.approx(SECOND_AT_CAP) for v in scope["at_cap_share"])       # a stopgap row keeps its own figure


def test_each_sources_degeneration_measures_stay_current_on_their_own_run(tmp_path):
    """The fold is the ownership views'; the `_current` views still keep every source's own `run_degeneration` measures (the two
    runs never share a key), so the secondary's readings remain readable beside the owner's rows."""
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    cur = {(r[0], r[1]): r[2] for r in conn.execute("SELECT run_id, measure, rate FROM v_run_degeneration_current WHERE measurement_state = 'measured'")}
    assert cur[(ids["secondary_run"], "union5")] == pytest.approx(SECOND_DEGEN)
    assert cur[(ids["secondary_run"], "at_token_cap_retokenized")] == pytest.approx(SECOND_AT_CAP)
    assert not any(rid == ids["owner_run"] for rid, _ in cur)
    conn.close()


# --------------------------------------------------------------------------------------------
# surfaced: the status page and `registry status --json`
# --------------------------------------------------------------------------------------------

def test_status_page_and_json_surface_the_states(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(RP.STATE_ENV, str(tmp_path / "landing-state"))      # the page reads the landing state: an empty per-test one, never the host's
    root, ids = _corpus(tmp_path)
    ctx = status.status_json(root)
    assert ctx["multi_source_counts"] == {"refused": 1, "stopgap": 1, "folded": 2}
    states = [r["state"] for r in ctx["multi_source"]]
    assert states[0] == "refused" and sorted(states) == ["folded", "folded", "refused", "stopgap"]
    assert ctx["multi_source_refused"][0]["metric_id"] == "asr" and ctx["multi_source_refused"][0]["threshold"] == 0.75
    assert ctx["multi_source_declarations"][0]["owner"] == OWNER_SRC and ctx["multi_source_declarations"][0]["never_carry"] == "at_cap_share"
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "Multi-source readings" in html and "1 refused multi-source identity" in html and "disagree on k / n" in html
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\b\w*_legacy\b", " ", text)
    for word in ("arm", "leg", "lane", "cell", "wave", "grader", "rung"):
        assert not re.search(r"(?<![A-Za-z0-9_])" + word + r"(?![A-Za-z0-9_])", text, re.IGNORECASE), word
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["status", "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["multi_source_counts"]["refused"] == 1 and any(r["state"] == "refused" for r in js["multi_source"])
    assert cli.main(["status"]) == 0
    printed = capsys.readouterr().out
    assert "multi-source readings: refused 1  stopgap 1  folded 2" in printed and "REFUSED qwen35_4b_it/arditi_allpos/multi_turn" in printed


# --------------------------------------------------------------------------------------------
# the REAL secondary: the all-positions adapter's multi-turn rows through the runner (B3)
# --------------------------------------------------------------------------------------------

def _allpos_dims():
    """The dimension rows the all-positions adapter's runs resolve against (the seeds' ids; synthetic contents): the served model, every
    eval the ladder and its multi-turn block file, their item sets. The adapter mints its own instrument / judge / object / computation rows."""
    from sisteer.registry.adapters import declared_common as DC
    t = synth.t
    unv = {c: {"confidence": "unverified", "reason": "synthetic fixture: never read"} for c in ("hf_revision", "chat_template_sha256")}
    # the adapter's dispositions (the held best-of-N) name the paper round `r2`; a constituent blocked on a missing parent holds the whole revision
    out = [synth.rec("paper_round", {"round_id": DC.ROUND_ID, "opened_utc": t(39), "description": "synthetic round under the registry's round id"}, t(39)),
           synth.rec("model", {"model_id": "qwen35_4b_it", "family": "qwen35", "size_label": "4b", "checkpoint_kind": "instruct", "hf_repo": "synthetic/qwen35-4b-it",
                               "hf_revision": "unverified", "chat_template_sha256": "unverified", "generation_config_present": 0, "thinking_channel": "available",
                               "paper_status": "in_paper", "field_confidence_json": unv}, t(40))]
    # the multi-turn instruments name the attacker model (instrument.attacker_model_id -> model): the seeds' row, synthetic contents
    out.append(synth.rec("model", {"model_id": "qwen36_27b_abliterated_attacker", "family": "qwen36", "size_label": "27B", "checkpoint_kind": "instruct",
                                   "hf_repo": "synthetic/qwen36-27b-abliterated", "hf_revision": "unverified", "chat_template_sha256": "unverified",
                                   "generation_config_present": 0, "thinking_channel": "available", "paper_status": "reference", "field_confidence_json": unv}, t(40)))
    evals = {"pasttense": ("static_attack", "asr", "lower_is_better", 0), "codeattack": ("static_attack", "asr", "lower_is_better", 0),
             "renellm": ("static_attack", "asr", "lower_is_better", 0), "grok420": ("static_attack", "asr", "lower_is_better", 0),
             "xstest": ("over_refusal", "xst_refused", "lower_is_better", 0), "mmlu": ("capability", "acc", "higher_is_better", 0),
             "strongreject_bon": ("best_of_n", "asr", "lower_is_better", 0),
             "crescendo": ("multi_turn", "asr", "lower_is_better", 1), "pair": ("multi_turn", "asr", "lower_is_better", 1), "actorattack": ("multi_turn", "asr", "lower_is_better", 1)}
    seeds = DC.load_seeds()
    for i, (eid, (cat, metric, direction, attacker)) in enumerate(evals.items()):
        out.append(synth.rec("eval", {"eval_id": eid, "eval_category": cat, "headline_metric_id": metric, "metric_direction": direction,
                                      "attacker_required": attacker, "paper_status": "in_paper"}, t(41 + i)))
    seen = set()
    for i, eid in enumerate(evals):
        iset = seeds.item_set(eid)
        if iset in seen:
            continue
        seen.add(iset)
        out.append(synth.rec("item_set", {"item_set_id": iset, "eval_id": eid, "dataset_version": f"synthetic/{eid}@v1", "n_items": 259, "items_sha256": "unverified",
                                          "source_uri": f"synthetic://items/{iset}.jsonl", "item_id_scheme": "g<index>", "subsample_rule": "full",
                                          "field_confidence_json": {"items_sha256": {"confidence": "unverified", "reason": "synthetic fixture: never read"}}}, t(60 + i)))
    return out


def _owner_rows_for(root: str, mt_rows, i0: int = 3000):
    """r2_export-shaped OWNER rows for the given secondary rows (one run per attack, `asr` at the three thresholds with the SAME k / n):
    one instrument per eval (the two readers mint different instruments in production), lane_source_legacy r2_export, the declared served
    operating point, the ladder's object under the declared label."""
    import test_registry_adapter_arditi_allpos as AAT
    from sisteer.registry.adapters import declared_common as DC
    seeds = DC.load_seeds()
    recs, by_eval = [_export_computation(i0 - 1)], {}       # the export's build computation: the owner side is its own computation rows only
    for r in mt_rows:
        by_eval.setdefault(r["eval_id"], []).append(r)
    campaign = "synth_r2_owner"
    for j, (eid, rows) in enumerate(sorted(by_eval.items())):
        inst = synth.instrument_fields(eval_id=eid, item_set_id=seeds.item_set(eid), token_cap_tokens=4096, token_cap_ladder_json="[4096]", max_turns=10,
                                       harness_config_hash=("o" * 7 + str(j)))
        recs.append(synth.rec("instrument", inst, synth.t(i0 + 100 * j)))
        key = f"owner_store_{eid}"
        obj_key = f"{LABEL}@{OP_ID}"
        rid = I.run_id(campaign, key, obj_key, eid, str(inst["instrument_id"]))
        k0 = i0 + 100 * j + 1
        p = synth.run_payload(key, obj_key, AAT.SHA_A, k0)
        p.update(run_id=rid, campaign=campaign, model_id=MODEL_ID, eval_id=eid, item_set_id=seeds.item_set(eid), instrument_id=str(inst["instrument_id"]),
                 transfer_kind="instruct_direct", multi_turn_steering_scope="every_turn_prefill_and_decode", token_cap_tokens=4096,
                 lane_source_legacy=OWNER_SRC, operating_point_id_legacy=OP_ID, strength_coefficient=0.3, strength_nominal=1.0,
                 strength_nominal_source="derived", is_operating_point=1, n_items_planned=259)
        jid = I.judgment_id(rid, synth.JUDGE_ID)
        t, host = synth.t, synth.TESTHOST
        # the same lifecycle the module's `_run` gives a synthetic owner run: queued -> running -> generated -> judged, each state with its evidence
        recs += [synth.rec("run", p, t(k0)),
                 synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(k0), "evidence_kind": "submit_wrapper"}, t(k0)),
                 synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(k0 + 1), "evidence_kind": "job_log", "scheduler_state": "RUNNING"}, t(k0 + 1)),
                 synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": host, "cluster_job_id": "4242", "gpu_count": 1},
                                         "verify": {"serving_path": "lora_native"}}, t(k0 + 2)),
                 synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(k0 + 3), "evidence_kind": "completion_artifact",
                                         "evidence_uri": f"synthetic:///store/{key}/results.json"}, t(k0 + 3)),
                 synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 259, "store_completion_artifact_uri": f"synthetic:///store/{key}/results.json",
                                                                   "store_completion_sha256": synth.SHA256_B}}, t(k0 + 4)),
                 synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": t(k0 + 5), "finished_utc": t(k0 + 6),
                                        "n_items_judged": 259, "status": "complete", "api_pool": "lowprio"}, t(k0 + 6)),
                 synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(k0 + 7), "evidence_kind": "completion_artifact"}, t(k0 + 7)),
                 synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 259, "judge_id": synth.JUDGE_ID}}, t(k0 + 8))]
        for m, r in enumerate(rows):
            if r["metric"] != "asr":
                continue                                                   # the owner carries the asr grid only: the held rows stay stopgaps
            recs.append(_metric(rid, jid, "asr", float(r["threshold"]), int(r["k"]), int(r["n"]), i0 + 100 * j + 10 + m, computation_id=EXPORT_COMP,
                                frac_at_token_cap=OWNER_AT_CAP, extra={"loop_share_measure_legacy": "rep_ratio_loop", "status_legacy": "graded", "note": "owner note"}))
    return recs


def test_the_allpos_adapters_multi_turn_rows_are_stopgaps_until_equal_owner_rows_fold_them(tmp_path):
    """END TO END with the REAL secondary: the all-positions adapter's multi-turn rows (B3) run through the runner and the harvester
    into a registry root. With no r2 owner row, `v_multi_source_check` reads STOPGAP for every one of the 15 identities (3 attacks x asr at
    0.25 / 0.5 / 0.75 + asr_held_in / asr_held_out at 0.5) and `v_paper_long` carries each once, from the secondary; add OWNER rows with
    EXACTLY the secondary's k / n for the nine asr identities and those nine fold (the owner row stands, the sidecar's degenerate_share /
    loop_share travel onto it, at_cap_share never does) while the six held rows stay stopgaps."""
    import test_registry_adapter_arditi_allpos as AAT
    from sisteer.registry.adapters import arditi_allpos as AA
    from sisteer.registry.adapters import runner as RUN
    store = AAT.build_store(str(tmp_path / "allpos"), with_mt=True)
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + _allpos_dims())
    ing0 = H.ingest_once(root, workers=0, min_free=0)
    assert ing0.refused == 0 and ing0.blocked == 0, ing0
    ad = AA.ArditiAllposAdapter(root=store, n_boot=200)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = RUN.run_pass(root, [ad], conn=conn, now="2026-09-18T00:00:00.000Z")
    finally:
        conn.close()
    assert rep.groups_refused == 0 and rep.records_written > 0, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_held == 0, ing
    mt_rows = [r for r in ad.long_rows()[0] if r.get("component") == AA.MT_COMPONENT and r["status"] == "graded"]
    assert len(mt_rows) == 15
    identities = {(a, "asr", t) for a in AA.MT_ATTACKS for t in AA.MT_THRESHOLDS} | {(a, f"asr_{g}", AA.MT_HELD_THRESHOLD) for a in AA.MT_ATTACKS for g in AA.MT_HELD_GROUPS}
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = _check(conn)
    assert set(chk) == identities, (sorted(set(chk) ^ identities))
    assert {r["state"] for r in chk.values()} == {"stopgap"} and all(r["owner_run_id"] is None and OWNER_SRC in r["reason"] for r in chk.values())
    fix = AAT.MT_FIX["crescendo"]
    d = AAT._det_counts(fix["excl"])
    c5 = chk[("crescendo", "asr", 0.5)]
    assert (c5["k_secondary"], c5["n_secondary"]) == (fix["k"][1], AA.MT_GOALS)
    assert c5["at_cap_share_secondary"] == pytest.approx(d["at_cap"] / fix["rows"]) and c5["degenerate_share_secondary"] == pytest.approx(d["union"] / fix["rows"])
    assert c5["loop_share_secondary"] == pytest.approx(d["rep_ratio_loop"] / fix["rows"]) and c5["loop_share_measure_secondary"] == "rep_ratio_loop"
    # the frame's scope rows for the declaration: the ladder's own 30 steered rows (static / over-refusal / capability) sit beside them under
    # the same label and are outside the multi-turn declaration
    long_rows = [r for r in _long(conn) if r["eval_category"] == DECL.eval_category]
    assert len(long_rows) == 15 and all(r["multi_source_state"] == "stopgap" and r["lane_source_legacy"] == SECOND_SRC and r["value"] is not None for r in long_rows)
    assert all(r["multi_source_state"] is None for r in _long(conn) if r["eval_category"] != DECL.eval_category)
    conn.close()
    df = _frame(root)
    scope = df[(df.model == "qwen35_4b") & (df.arm == LABEL) & (df.component == "multi-turn")]
    assert len(scope) == 15 and all("STOPGAP ROW" in str(n) for n in scope["note"])
    row = scope[(scope.eval_id == "crescendo") & (scope.metric == "asr") & (scope.threshold == 0.5)].iloc[0]
    assert row["at_cap_share"] == pytest.approx(d["at_cap"] / fix["rows"]) and row["loop_share"] == pytest.approx(d["rep_ratio_loop"] / fix["rows"])
    assert row["degenerate_share"] == pytest.approx(d["union"] / fix["rows"]) and row["delta"] is not None and not _null(row["delta"])   # the parked contrast reaches the frame
    assert row["delta_method"] == AA.FORMED_CONTRAST_METHOD and row["arm_operating_point_id"] == OP_ID
    # ---- the owner arrives with EQUAL k / n for the nine asr identities: those fold, the held rows stay stopgaps
    _write(root, "owner", "000003-owner.jsonl", _owner_rows_for(root, mt_rows))
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.blocked == 0, ing2
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk2 = _check(conn)
    assert {k: v["state"] for k, v in chk2.items()} == {k: ("folded" if k[1] == "asr" else "stopgap") for k in identities}
    f5 = chk2[("crescendo", "asr", 0.5)]
    assert (f5["k_owner"], f5["n_owner"]) == (f5["k_secondary"], f5["n_secondary"]) == (fix["k"][1], AA.MT_GOALS)
    assert json.loads(f5["carried_json"]) == ["degenerate_share", "loop_share"]
    assert f5["at_cap_share_owner"] == pytest.approx(OWNER_AT_CAP) and f5["at_cap_share_secondary"] == pytest.approx(d["at_cap"] / fix["rows"])
    rows2 = [r for r in _long(conn) if r["eval_category"] == DECL.eval_category]
    assert len(rows2) == 15
    asr_rows = [r for r in rows2 if r["metric_id"] == "asr"]
    assert len(asr_rows) == 9 and all(r["lane_source_legacy"] == OWNER_SRC and r["multi_source_state"] == "folded" for r in asr_rows)
    held_rows = [r for r in rows2 if r["metric_id"] != "asr"]
    assert len(held_rows) == 6 and all(r["lane_source_legacy"] == SECOND_SRC and r["multi_source_state"] == "stopgap" for r in held_rows)
    conn.close()
    df2 = _frame(root)
    scope2 = df2[(df2.model == "qwen35_4b") & (df2.arm == LABEL) & (df2.component == "multi-turn")]
    fold = scope2[(scope2.eval_id == "crescendo") & (scope2.metric == "asr") & (scope2.threshold == 0.5)].iloc[0]
    assert fold["source"] == OWNER_SRC and fold["multi_source_state"] == "folded"
    assert fold["degenerate_share"] == pytest.approx(d["union"] / fix["rows"]) and fold["loop_share"] == pytest.approx(d["rep_ratio_loop"] / fix["rows"])   # carried
    assert fold["at_cap_share"] == pytest.approx(OWNER_AT_CAP)                                                                                          # never the sidecar's


# --------------------------------------------------------------------------------------------
# run pairing (migration 0005): a declaration per transfer kind, paired by producer run key, both sides' own computation rows only,
# an eval id excluded by name; the model's own reader OWNS the Llama Petri scopes (paperlib 2026-09-14)
# --------------------------------------------------------------------------------------------

LLAMA_LABELS = ("constitution_lora", "control_lora", "difference_lora")
LLAMA_KINDS = ("base_to_instruct", "instruct_direct")
LLAMA_IDS = tuple(f"llama8b_it/{label}/petri/{kind}" for label in LLAMA_LABELS for kind in LLAMA_KINDS)


def test_llama_petri_declarations_are_per_transfer_kind_and_pair_by_producer_key():
    assert tuple(d.declaration_id for d in OW.declarations()) == (DECL.declaration_id,) + LLAMA_IDS + tuple(d.declaration_id for d in OW.TWIN_OWNERSHIP)   # + the two twin scopes (migration 0026)
    assert OW.LLAMA_PETRI_SERVED_OPERATING_POINT == {"base_to_instruct": "base__constv2-mix-full__lora_r4_ce", "instruct_direct": "itdoc__constv2-mix-full__lora_r4_ce"}
    assert OW.LLAMA_PETRI_EXCLUDED_EVALS == ("petri_pooled",)
    for label in LLAMA_LABELS:
        for kind in LLAMA_KINDS:
            d = OW.BY_ID[f"llama8b_it/{label}/petri/{kind}"]
            assert d.model_ids == ("llama8b_it",) and d.paper_label == label and d.eval_category == "petri" and d.transfer_kind == kind
            assert (d.owner_source, d.secondary_source) == ("llama8b_lane", "r2_export"), "the model's own reader owns; the export is the secondary"
            assert d.served_operating_point_id == OW.LLAMA_PETRI_SERVED_OPERATING_POINT[kind] and d.strength_rule == "served_operating_point"
            assert d.pairing_rule == "producer_run_key" and d.pairing_key_sql() == "r.producer_run_key_legacy"
            assert d.exclude_eval_ids == ("petri_pooled",)
            assert d.since == "2026-09-21"
            for phrase in ("2026-09-14", "load_r2 drops", "2026-09-21 01:30Z", "scenarios PLANNED", "scenarios SCORED", "59",
                           "n 9 in the reader against 10 in the export", "REFUSED per identity", "petri_pooled", "not_in_roster"):
                assert phrase in d.ruling, phrase
            sql = d.scope_sql()
            assert f"r.transfer_kind = '{kind}'" in sql and f"o.paper_label = '{label}'" in sql and "e.eval_category = 'petri'" in sql
            assert "r.eval_id NOT IN ('petri_pooled')" in sql
            assert f"r.operating_point_id_legacy = '{d.served_operating_point_id}'" in sql and "r.strength_nominal = 1.0" in sql
            assert "r.lane_source_legacy IN ('llama8b_lane', 'r2_export')" in sql
    # rendered: one block per declaration; BOTH sides restricted to their own computation rows in EVERY block; the pairing key per rule;
    # the excluded eval in the six Llama blocks only
    rows_sql = S.multi_source_rows_sql()
    for did in LLAMA_IDS:
        assert f"'{did}' AS declaration_id" in rows_sql
    assert rows_sql.count("\n  AND rm.computation_id <> ''\n") == len(OW.declarations())
    assert "lane_source_legacy <> " not in rows_sql
    assert rows_sql.count("r.producer_run_key_legacy AS pairing_key") == len(LLAMA_IDS) and rows_sql.count("'' AS pairing_key") == 1
    assert rows_sql.count("r.eval_id NOT IN ('petri_pooled')") == len(LLAMA_IDS)
    chk_sql = S.multi_source_check_sql()
    assert "GROUP BY declaration_id, pairing_key, eval_id, metric_id, threshold" in chk_sql and "ow.pairing_key = s.pairing_key" in chk_sql
    paper_long = {n: sql for n, sql, _ in S.render_views()}["v_paper_long"]
    assert "msc.pairing_key = msr.pairing_key" in paper_long


def test_adapter_priorities_for_the_llama_declarations():
    from sisteer.registry.adapters import llama8b_wave as LW
    rows = {r["declaration_id"]: r for r in OW.check_priorities(ADP.load_all())}
    assert set(rows) == {d.declaration_id for d in OW.declarations()}
    for did in LLAMA_IDS:
        assert rows[did]["owner_priorities"] == {"llama8b_wave": IC.PRIORITY_OWNING_READER}
        assert rows[did]["secondary_priorities"] == {"r2_export": IC.PRIORITY_PAPER_SOURCE}
    assert LW.Llama8bWaveAdapter.priority == IC.PRIORITY_OWNING_READER > IC.PRIORITY_PAPER_SOURCE
    # the export, secondary here and owner of the Qwen scope, is outranked by one and outranks the other: a registry where the Llama reader
    # sat at the export's priority is refused by name
    class Flat:
        name = "llama8b_wave"
        covers_sources = ("llama8b_lane",)
        priority = IC.PRIORITY_PAPER_SOURCE
    with pytest.raises(ValueError, match="outrank"):
        OW.check_priorities(dict(ADP.load_all(), llama8b_wave=lambda: Flat()))


def test_the_qwen_scope_pairs_on_the_identity_alone_whatever_the_producer_keys(tmp_path):
    """The Qwen fixture's two runs carry DIFFERENT producer run keys (as that scope's two readers do in production); under `identity_only`
    the pairing key is '' on both sides and the fold is exactly the one before the pairing key existed: one check row per
    (eval, metric, threshold), the same four states. Both sides are their computation rows (the export's build, the reader's stats)."""
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    keys = {r[0]: r[1] for r in conn.execute("SELECT run_id, producer_run_key_legacy FROM run WHERE run_id IN (?, ?)", (ids["owner_run"], ids["secondary_run"]))}
    assert keys[ids["owner_run"]] != keys[ids["secondary_run"]]
    rows = [dict(r) for r in conn.execute("SELECT * FROM v_multi_source_check WHERE declaration_id = ?", (DECL.declaration_id,))]
    assert len(rows) == 4 and all(r["pairing_key"] == "" and r["transfer_kind"] == "instruct_direct" for r in rows)
    assert {(r["eval_id"], r["metric_id"], r["threshold"]): r["state"] for r in rows} == {
        (EVAL_ID, "asr", 0.25): "stopgap", (EVAL_ID, "asr", 0.5): "folded", (EVAL_ID, "asr", 0.75): "refused", (EVAL_ID, "asr_held_in", -1.0): "folded"}
    fold = next(r for r in rows if r["state"] == "folded" and r["metric_id"] == "asr")
    assert (fold["owner_producer_run_key_legacy"], fold["secondary_producer_run_key_legacy"]) == (keys[ids["owner_run"]], keys[ids["secondary_run"]])
    assert (fold["owner_computation_id"], fold["secondary_computation_id"]) == (EXPORT_COMP, READER_COMP)
    scoped = [dict(r) for r in conn.execute("SELECT role, computation_id FROM v_multi_source_rows WHERE declaration_id = ?", (DECL.declaration_id,))]
    assert sorted((r["role"], r["computation_id"]) for r in scoped) == [("owner", EXPORT_COMP)] * 3 + [("secondary", READER_COMP)] * 4
    conn.close()


def test_strength_rules_admit_real_runs_and_blank_rows_stay_out_on_both_sides(tmp_path, monkeypatch):
    """Rendered against a real corpus: the `unfitted` predicate admits the synth corpus's `none` run (no operating point, no nominal) and
    not the fitted one; `served_operating_point` the reverse. Then a whole `unfitted` declaration over the `none` object renders to NO
    scoped row: the corpus's own rows are producer-filed blanks, and the own-computation rule holds on both sides."""
    root, ids = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    cat = conn.execute("SELECT eval_category FROM eval WHERE eval_id = ?", (synth.EVAL_ID,)).fetchone()[0]
    none_run = conn.execute("SELECT run_id, lane_source_legacy, transfer_kind, strength_nominal, operating_point_id_legacy FROM run "
                            "WHERE object_id = 'none' AND model_id = ?", (synth.MODEL_ID,)).fetchone()
    assert none_run[2] == "none" and none_run[3] is None and none_run[4] is None
    n_blank = conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND computation_id = ''", (none_run[0],)).fetchone()[0]
    assert n_blank > 0
    kw = dict(model_ids=(synth.MODEL_ID,), paper_label="none", eval_category=cat, transfer_kind="none", owner_source="synthetic_other",
              secondary_source=none_run[1], since="2026-09-21", ruling="synthetic", pairing_rule="identity_only")
    unfitted = OW.Ownership("synthetic/none/unfitted", served_operating_point_id=None, strength_rule="unfitted", **kw)
    served = OW.Ownership("synthetic/none/served", served_operating_point_id=OP_ID, **kw)
    both = (none_run[0], ids["owner_run"])
    admitted = lambda d: sorted(r[0] for r in conn.execute("SELECT r.run_id FROM run r WHERE r.run_id IN (?, ?) AND " + d.strength_sql("r"), both))  # noqa: E731
    assert admitted(unfitted) == [none_run[0]] and admitted(served) == [ids["owner_run"]]
    monkeypatch.setattr(OW, "MULTI_SOURCE_OWNERSHIP", (unfitted,))
    conn.executescript(S.multi_source_rows_sql().replace("CREATE VIEW v_multi_source_rows AS", "CREATE TEMP VIEW t_rows AS"))
    assert conn.execute("SELECT COUNT(*) FROM t_rows").fetchone()[0] == 0, "a producer-filed blank row is never in a scope, on either side"
    conn.close()


# ---- a scope shaped like the Llama-3.1-8B Petri measurements: one label, two fitted objects under two transfer kinds, the model's own
# reader owning, the export secondary with the stats reader's blank twin on every one of its runs, and the pooled eval left out of scope
P_MODEL, P_LABEL, P_KINDS = "llama8b_it", "control_lora", LLAMA_KINDS
P_EVALS = ("synth_petri_a", "synth_petri_b")
P_POOLED = "petri_pooled"                                      # the excluded id, by its spelling (the declarations exclude it by name)
P_ALL_EVALS = P_EVALS + (P_POOLED,)
P_ITEM_SET = {eid: f"{eid}_10" for eid in P_ALL_EVALS}
P_SHA = {"base_to_instruct": "e" * 64, "instruct_direct": "f" * 64}
P_KEY = {"base_to_instruct": "synth_petri_key_base", "instruct_direct": "synth_petri_key_it"}
P_DECL = {k: OW.BY_ID[f"llama8b_it/{P_LABEL}/petri/{k}"] for k in P_KINDS}
P_READER_SRC, P_EXPORT_SRC = P_DECL["base_to_instruct"].owner_source, P_DECL["base_to_instruct"].secondary_source
P_READER_CAMPAIGN, P_EXPORT_CAMPAIGN = "synth_petri_reader", "synth_petri_export"
P_SCORE = {("base_to_instruct", "synth_petri_a"): 7.333333333333333, ("base_to_instruct", "synth_petri_b"): 6.1, ("base_to_instruct", P_POOLED): 6.6,
           ("instruct_direct", "synth_petri_a"): 6.4, ("instruct_direct", "synth_petri_b"): 5.8, ("instruct_direct", P_POOLED): 6.28}
P_DISAGREE = ("base_to_instruct", "synth_petri_a")            # the reader scored 9 scenarios where the export counted the 10 planned


def _petri_dims():
    t = synth.t
    model = dict(synth.seed_dimensions()[2]["payload"])
    model["model_id"] = P_MODEL
    out = [synth.rec("model", model, t(40)), _export_computation(41), _reader_computation(42)]
    insts = {}
    for j, eid in enumerate(P_ALL_EVALS):
        out.append(synth.rec("eval", {"eval_id": eid, "eval_category": "petri", "headline_metric_id": "petri_score", "metric_direction": "lower_is_better",
                                      "attacker_required": 1, "paper_status": "in_paper"}, t(43 + j)))
        out.append(synth.rec("item_set", {"item_set_id": P_ITEM_SET[eid], "eval_id": eid, "dataset_version": f"synthetic/{eid}@v1", "n_items": 10,
                                          "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{eid}.jsonl", "item_id_scheme": "s<index>",
                                          "subsample_rule": "full"}, t(46 + j)))
        for tag in ("x", "y"):
            inst = synth.instrument_fields(eval_id=eid, item_set_id=P_ITEM_SET[eid], token_cap_tokens=1024, token_cap_ladder_json="[1024]",
                                           harness_config_hash=(tag * 7 + str(j)))
            insts[(tag, eid)] = inst
            out.append(synth.rec("instrument", inst, t(50 + 2 * j + (tag == "y"))))
    for kind in P_KINDS:
        out.append(synth.rec("object", synth.object_lora(P_SHA[kind], paper_label=P_LABEL, role="control"), t(60 + (kind == "instruct_direct"))))
    return out, insts


def _petri_run(campaign: str, source: str, kind: str, eid: str, inst, i0: int):
    """One judged Petri run of the scope: the object fit under `kind`, no operating point filed, nominal 1.0 (as both Llama sources file)."""
    obj_key = f"{P_LABEL}@{kind}"
    rid = I.run_id(campaign, P_KEY[kind], obj_key, eid, str(inst["instrument_id"]))
    p = synth.run_payload(P_KEY[kind], obj_key, P_SHA[kind], i0)
    p.update(run_id=rid, campaign=campaign, model_id=P_MODEL, eval_id=eid, item_set_id=P_ITEM_SET[eid], instrument_id=str(inst["instrument_id"]),
             transfer_kind=kind, token_cap_tokens=1024, lane_source_legacy=source, strength_nominal=1.0, strength_nominal_source="derived",
             is_operating_point=1, n_items_planned=10)
    assert "operating_point_id_legacy" not in p
    t, host = synth.t, synth.TESTHOST
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs = [
        synth.rec("run", p, t(i0)),
        synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0)),
        synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log", "scheduler_state": "RUNNING"}, t(i0 + 1)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": host, "cluster_job_id": "4242", "gpu_count": 1},
                                "verify": {"serving_path": "lora_native"}}, t(i0 + 2)),
        synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                "evidence_uri": f"synthetic:///store/{P_KEY[kind]}/{eid}.json"}, t(i0 + 3)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 10, "store_completion_artifact_uri": f"synthetic:///store/{P_KEY[kind]}/{eid}.json",
                                                          "store_completion_sha256": synth.SHA256_B}}, t(i0 + 4)),
        synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": t(i0 + 5), "finished_utc": t(i0 + 6),
                               "n_items_judged": 10, "status": "complete", "api_pool": "lowprio"}, t(i0 + 6)),
        synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"}, t(i0 + 7)),
        synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 10, "judge_id": synth.JUDGE_ID}}, t(i0 + 8)),
    ]
    return rid, jid, recs


def _score(rid, jid, value, n, i, **cols):
    """A `petri_score` row: a mean, no k."""
    p = {"run_id": rid, "judgment_id": jid, "metric_id": "petri_score", "threshold": -1.0, "value": value, "n": n, "computed_by": "producer"}
    p.update(cols)
    return synth.rec("run_metric", p, synth.t(i))


def _petri_corpus(tmp_path):
    """Both sources read every (kind, eval) once under the SAME producer key per object — the pooled eval included; every EXPORT run also
    carries the stats reader's blank-computation twin of its row; the reader scored n 9 at P_DISAGREE where the export counted 10."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=4)
    dims, insts = _petri_dims()
    _write(root, "dimensions", "000002-petri.jsonl", dims)
    ids, i0 = {}, 5000
    for kind in P_KINDS:
        for eid in P_ALL_EVALS:
            for source, campaign, tag in ((P_READER_SRC, P_READER_CAMPAIGN, "y"), (P_EXPORT_SRC, P_EXPORT_CAMPAIGN, "x")):
                rid, jid, recs = _petri_run(campaign, source, kind, eid, insts[(tag, eid)], i0)
                n = 9 if (source == P_READER_SRC and (kind, eid) == P_DISAGREE) else 10
                value = P_SCORE[(kind, eid)]
                if source == P_EXPORT_SRC:
                    metrics = [_score(rid, jid, value, n, i0 + 10, computation_id=EXPORT_COMP, extra={"status_legacy": "graded"}),
                               _score(rid, jid, value, n, i0 + 11)]           # the stats reader's twin: same numbers, blank computation
                else:
                    metrics = [_score(rid, jid, value, n, i0 + 10, computation_id=READER_COMP)]
                synth.write_run_dir(root, campaign, rid, [(str(r["record_type"]), [r]) for r in recs] + [("run_metric", metrics)])
                ids[(source, kind, eid)] = rid
                i0 += 20
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked)
    return root, ids


def test_two_fitted_objects_of_one_label_pair_per_transfer_kind_the_reader_owns_and_the_twin_never_counts(tmp_path):
    root, ids = _petri_corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    chk = {(r["declaration_id"], r["pairing_key"], r["eval_id"]): dict(r) for r in conn.execute("SELECT * FROM v_multi_source_check WHERE declaration_id LIKE 'llama8b_it/%'")}
    # the two sub-evals per kind; the pooled eval is out of scope and has no identity
    assert set(chk) == {(P_DECL[k].declaration_id, P_KEY[k], eid) for k in P_KINDS for eid in P_EVALS}
    for (did, key, eid), r in chk.items():
        kind = did.rsplit("/", 1)[1]
        # one run per side per kind: the export's twin is not counted, the other kind's object is another declaration
        assert r["transfer_kind"] == kind and r["owner_rows"] == 1 and r["secondary_rows"] == 1
        assert r["owner_producer_run_key_legacy"] == r["secondary_producer_run_key_legacy"] == key
        assert (r["owner_source"], r["secondary_source"]) == (P_READER_SRC, P_EXPORT_SRC)
        assert r["owner_run_id"] == ids[(P_READER_SRC, kind, eid)] and r["secondary_run_id"] == ids[(P_EXPORT_SRC, kind, eid)]
        assert r["owner_computation_id"] == READER_COMP and r["secondary_computation_id"] == EXPORT_COMP
        assert r["value_owner"] == pytest.approx(P_SCORE[(kind, eid)]) and r["value_secondary"] == pytest.approx(P_SCORE[(kind, eid)])
        assert r["k_owner"] is None and r["k_secondary"] is None and r["carried_json"] == "[]"
    ref = chk[(P_DECL[P_DISAGREE[0]].declaration_id, P_KEY[P_DISAGREE[0]], P_DISAGREE[1])]
    assert ref["state"] == "refused" and (ref["n_owner"], ref["n_secondary"]) == (9, 10)
    assert "owner k=null n=9" in ref["reason"] and "secondary k=null n=10" in ref["reason"] and "never averaged" in ref["reason"]
    assert sum(r["state"] == "folded" for r in chk.values()) == 3 and {r["state"] for r in chk.values()} == {"folded", "refused"}
    # the scope holds the reader's rows and the export's computed rows only, never a blank row on either side, never the pooled eval
    scoped = [dict(r) for r in conn.execute("SELECT role, computation_id, run_id, eval_id FROM v_multi_source_rows WHERE declaration_id LIKE 'llama8b_it/%'")]
    assert len(scoped) == 8 and all(r["computation_id"] != "" and r["eval_id"] != P_POOLED for r in scoped)
    assert sorted((r["role"], r["computation_id"]) for r in scoped) == [("owner", READER_COMP)] * 4 + [("secondary", EXPORT_COMP)] * 4
    # by label alone the owner would carry two runs at every eval — the transfer kind is what tells the two objects apart
    both = conn.execute("SELECT eval_id, COUNT(DISTINCT run_id) FROM v_multi_source_rows WHERE role = 'owner' AND paper_label = ? GROUP BY eval_id ORDER BY 1", (P_LABEL,)).fetchall()
    assert [tuple(r) for r in both] == [(P_EVALS[0], 2), (P_EVALS[1], 2)]
    # v_paper_long, the sub-evals: per (kind, eval) the READER's row stands (folded, or refused with its numbers NULL); the export's computed
    # row leaves; the export's blank twin stays as metric-role evidence OUTSIDE the scope
    rows = [dict(r) for r in conn.execute("SELECT * FROM v_paper_long WHERE model_id = ? AND paper_label = ? ORDER BY transfer_kind, eval_id, computation_id", (P_MODEL, P_LABEL))]
    sub = [r for r in rows if r["eval_id"] != P_POOLED]
    assert len(sub) == 8
    own = [r for r in sub if r["lane_source_legacy"] == P_READER_SRC]
    twins = [r for r in sub if r["lane_source_legacy"] == P_EXPORT_SRC]
    assert len(own) == 4 and all(r["computation_id"] == READER_COMP and r["multi_source_role"] == "owner" and r["metric_role"] == "owner"
                                 and r["multi_source_declaration_id"] == P_DECL[r["transfer_kind"]].declaration_id for r in own)
    assert len(twins) == 4 and all(r["computation_id"] == "" and r["multi_source_role"] is None and r["multi_source_state"] is None
                                   and r["metric_role"] == "secondary" and r["value"] is not None for r in twins)
    refused = [r for r in own if r["multi_source_state"] == "refused"]
    assert len(refused) == 1 and (refused[0]["transfer_kind"], refused[0]["eval_id"]) == P_DISAGREE
    assert refused[0]["value"] is None and refused[0]["n"] is None and "n=9" in refused[0]["multi_source_reason"] and "n=10" in refused[0]["multi_source_reason"]
    folded = [r for r in own if r["multi_source_state"] == "folded"]
    assert len(folded) == 3 and all(r["value"] == pytest.approx(P_SCORE[(r["transfer_kind"], r["eval_id"])]) and r["n"] == 10 for r in folded)
    assert all(r["multi_source_secondary_run_id"] == ids[(P_EXPORT_SRC, r["transfer_kind"], r["eval_id"])] for r in own)
    # v_paper_long, the pooled eval: untouched — both sources' rows stand with no role, no state, no fold and no refusal (the export's
    # computed row, its twin as metric-role evidence, and the reader's row), exactly as before the declarations
    pooled = [r for r in rows if r["eval_id"] == P_POOLED]
    assert len(pooled) == 6 and all(r["multi_source_role"] is None and r["multi_source_state"] is None and r["value"] is not None for r in pooled)
    assert sorted((r["lane_source_legacy"], r["computation_id"], r["metric_role"]) for r in pooled) == sorted(
        [(P_EXPORT_SRC, EXPORT_COMP, "owner"), (P_EXPORT_SRC, "", "secondary"), (P_READER_SRC, READER_COMP, "owner")] * 2)
    conn.close()
    # surfaced: the status page counts the refusal under its declaration, transfer kind and pairing key; the declarations name the exclusion
    ctx = status.status_json(root)
    assert ctx["multi_source_counts"] == {"refused": 1, "stopgap": 0, "folded": 3}
    r0 = ctx["multi_source_refused"][0]
    assert (r0["declaration_id"], r0["transfer_kind"], r0["pairing_key"], r0["eval_id"]) == (P_DECL[P_DISAGREE[0]].declaration_id, P_DISAGREE[0], P_KEY[P_DISAGREE[0]], P_DISAGREE[1])
    assert (r0["value_owner"], r0["value_secondary"], r0["n_owner"], r0["n_secondary"]) == (pytest.approx(P_SCORE[P_DISAGREE]), pytest.approx(P_SCORE[P_DISAGREE]), 9, 10)
    decls = {d["declaration_id"]: d for d in ctx["multi_source_declarations"]}
    assert len(decls) == len(OW.declarations()) == 9                                      # the Qwen scope, six Llama Petri scopes, two twin scopes (migration 0026)
    for did in LLAMA_IDS:
        assert (decls[did]["owner"], decls[did]["secondary"], decls[did]["pairing_rule"], decls[did]["excluded_eval_ids"]) == (P_READER_SRC, P_EXPORT_SRC, "producer_run_key", P_POOLED)
    assert decls[DECL.declaration_id]["pairing_rule"] == "identity_only" and decls[DECL.declaration_id]["excluded_eval_ids"] == ""
    assert decls[P_DECL["instruct_direct"].declaration_id]["transfer_kind"] == "instruct_direct" and decls[DECL.declaration_id]["strength_rule"] == "served_operating_point"
    assert decls[P_DECL["base_to_instruct"].declaration_id]["served_operating_point_id"] == "base__constv2-mix-full__lora_r4_ce"


# --------------------------------------------------------------------------------------------
# 0010: the owner's OWN degeneration reading (a run_degeneration row from any reader) stands in the carry columns
# --------------------------------------------------------------------------------------------

def test_an_owners_own_degeneration_row_fills_its_carry_column_and_only_the_missing_carry_travels(tmp_path):
    """The owner (r2_export) filed no `frac_degenerate_union5` mirror, but a MEASURED `union5` row exists on its run under its judgment
    (a second reader's, the by_run_id batch's, the export's own union rows): since 0010 `degenerate_share_owner` reads it, the fold
    stands when it equals the secondary's, and only `loop_share` -- which the owner still lacks -- is carried."""
    root, ids = _corpus(tmp_path)
    _write(root, "second_reader", "000001-owner-degeneration.jsonl",
           [_degeneration(ids["owner_run"], ids["owner_judgment"], "union5", SECOND_DEGEN, 4000, 1030)])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    fold = _check(conn)[(EVAL_ID, "asr", 0.5)]
    assert fold["state"] == "folded" and fold["degenerate_share_owner"] == pytest.approx(SECOND_DEGEN) and fold["loop_share_owner"] is None
    assert json.loads(fold["carried_json"]) == ["loop_share"] and "carried onto the owner where it had none: loop_share" in fold["reason"]
    rows = {(r["metric_id"], r["threshold"]): r for r in _long(conn) if r["lane_source_legacy"] == OWNER_SRC}
    owner = rows[("asr", 0.5)]
    assert owner["frac_degenerate_union5"] == pytest.approx(SECOND_DEGEN) and json.loads(owner["multi_source_carried_json"]) == ["loop_share"]
    conn.close()
    df = _frame(root)
    scope = df[(df.model == "qwen35_4b") & (df.arm == LABEL)]
    row = scope[(scope.metric == "asr") & (scope.threshold == 0.5)].iloc[0]
    assert row["degenerate_share"] == pytest.approx(SECOND_DEGEN) and row["loop_share"] == pytest.approx(SECOND_LOOP)


def test_an_owners_own_degeneration_row_that_differs_from_the_secondary_refuses_the_fold(tmp_path):
    """The same reading at another value: the owner already carries a DIFFERENT `degenerate_share` (its own run's measured row now
    counts as one), and a disagreement is never overwritten -- neither value is admitted, exactly as a differing mirror refuses."""
    root, ids = _corpus(tmp_path)
    _write(root, "second_reader", "000001-owner-degeneration.jsonl",
           [_degeneration(ids["owner_run"], ids["owner_judgment"], "union5", 0.25, 4000, 1030)])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    fold = _check(conn)[(EVAL_ID, "asr", 0.5)]
    assert fold["state"] == "refused" and "degenerate_share=0.25" in fold["reason"] and "never overwritten" in fold["reason"]
    assert fold["carried_json"] == "[]"
    rows = {(r["metric_id"], r["threshold"]): r for r in _long(conn) if r["lane_source_legacy"] == OWNER_SRC}
    assert rows[("asr", 0.5)]["multi_source_state"] == "refused" and rows[("asr", 0.5)]["value"] is None
    conn.close()
