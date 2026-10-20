"""Catch-up of `sisteer.registry.paper` with the paper past the parity pin (catch-up B1).

`paperlib.load.COLUMNS` at paper-figures f33dca5d (vendored as a fixture with the file's sha) gained eight columns after the
40997227 pin: the degeneracy decomposition and the second at-cap estimand (`degen_truncation_only_share`,
`degen_loop_without_cap_share`, `at_cap_share_finish_reason`, `degeneracy_basis`), the serving engine and the cross-engine
flag (`serving_engine`, `cross_engine_contrast`) and the judge-censored bound (`judge_blocked`, `lower_bound`).
`paper_long_frame(columns=load.COLUMNS)` REFUSED them (`LongColumnUnsupplied`). Each is now supplied from the registry's own
data model — `run_degeneration` measures (the pin stamps the export's degeneracy METRIC ROWS onto the cell's headline rows;
the registry reads the measures the union adapter and the r2 adapter filed), `run.serving_identity`, the contrast's
producer-attested flag, `judgment.n_judge_refusals` — or NULL with a documented reason (`paper.LONG_COLUMN_NULL_REASONS`),
never a refusal.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import vocab as V

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "registry", "paper_load_columns_f33dca5d.json")
#: the pinned paper checkout, a linked checkout of the main one (declared_common.MAIN_CHECKOUT), where present
PIN_DIR = os.path.join(DC.MAIN_CHECKOUT, "worktrees", "paper-figures-pin-f33dca5d", "paper", "paperlib", "load.py")

ENGINE = "proxy_vllm+lens+interleaved"


def pinned_columns():
    with open(FIXTURE, encoding="utf-8") as fh:
        fx = json.load(fh)
    return fx["columns"], fx


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _roster_entry(i, object_id=None, object_selector=None, comparison="fig_synth"):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = object_selector or {"paper_label": "constitution_lora"}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _degeneration(run_id, jid, measure, rate, n=200, scope="all", aggregation="single_detector", i=300, pooling=None, notes=None):
    k = int(round(rate * n))
    return synth.rec("run_degeneration", {
        "run_id": run_id, "judgment_id": jid, "measure": measure, "scope": scope, "measurement_state": "measured",
        "numerator": k, "denominator": n, "denominator_unit": "replies", "pooling_json": pooling or {"rows_are": "target replies over every turn"},
        "aggregation": aggregation, "rate": k / n, "definition_version": "degeneracy_union_mt_cap4096/1@synthetic",
        "threshold_params_json": {}, "computed_by": "producer", "computed_utc": synth.t(i), "source_uri": "synthetic://union", "notes": notes}, synth.t(i))


def _corpus(tmp_path, *, degeneration=True, engine=True, refusals=True, contrast=True):
    """Two judged runs (none + constitution_lora) on a roster; the lora run carries the post-pin facts the eight columns read."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    none_run, lora_run = ids["run_ids"]
    jid = ids["judgment_ids"][1]
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1), _roster_entry(2, object_id="none", comparison="baseline")])
    extra = []
    if degeneration:
        degs = [_degeneration(lora_run, jid, "at_token_cap_retokenized", 0.05, i=300),
                _degeneration(lora_run, jid, "at_token_cap_finish_reason", 0.03, i=301),
                _degeneration(lora_run, jid, "truncation_only", 0.01, i=302),
                _degeneration(lora_run, jid, "loop_without_cap", 0.02, i=303),
                _degeneration(lora_run, jid, "rep_ratio_loop", 0.02, i=304),
                _degeneration(lora_run, jid, "union5", 0.30, aggregation="union", i=305,
                              pooling={"rows_are": "target replies over every turn of every conversation", "denominator": "target replies (every target turn in the store)"})]
        synth.write_run_dir(root, synth.CAMPAIGN, lora_run, [("run_degeneration", degs)])
        # the producer's own at-cap value on the headline metric row (the export's `at_cap_share` column → frac_at_token_cap)
        extra.append(synth.rec("run_metric", {"run_id": lora_run, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 1, "n": 4,
                                              "frac_at_token_cap": 0.2, "computed_by": "producer"}, synth.t(310)))
    if engine:
        extra.append(synth.rec("run_patch", {"run_id": lora_run, "fields": {"serving_identity": ENGINE, "serving_engine_version": "vllm 0.26.0"}}, synth.t(311)))
    if refusals:
        extra.append(synth.rec("judgment", {"judgment_id": jid, "run_id": lora_run, "judge_id": synth.JUDGE_ID, "started_utc": synth.t(125),
                                            "finished_utc": synth.t(126), "n_items_judged": 4, "n_judge_refusals": 28, "status": "complete",
                                            "api_pool": "lowprio"}, synth.t(312)))
    if contrast:
        extra.append(synth.rec("run_contrast", {"run_id": lora_run, "anchor_run_id": none_run, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5,
                                                "anchor_kind": "same_standup_none", "delta": -10.0, "delta_method": "exact McNemar", "n_pairs": 4,
                                                "computed_by": "producer", "extra": {"cross_engine_contrast": True, "anchor_engine": "proxy_vllm+passthrough+interleaved"}},
                               synth.t(313)))
    if extra:
        _write(root, "catchup", "000001-facts.jsonl", extra)
    H.ingest_once(root, workers=0, min_free=0)
    return root, none_run, lora_run


def _null(v) -> bool:
    """A missing value in the frame: None on an object column, NaN on a numeric one (the paper's own spelling of a missing numeric)."""
    return v is None or (isinstance(v, float) and v != v)


def _frame(root, **kw):
    cols, _ = pinned_columns()
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return paper.paper_long_frame(conn, "r_synth", columns=cols, **kw)
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# the fixture is the pin's list
# --------------------------------------------------------------------------------------------

def test_fixture_is_the_pinned_load_columns():
    cols, fx = pinned_columns()
    assert len(cols) == 68 and cols[0] == "source" and cols[-1] == "petri_served_object_sha256"
    for c in ("degen_truncation_only_share", "degen_loop_without_cap_share", "at_cap_share_finish_reason", "degeneracy_basis",
              "serving_engine", "cross_engine_contrast", "judge_blocked", "lower_bound"):
        assert c in cols
    if os.path.exists(PIN_DIR):   # the pin worktree, where present, is the fixture's source: the sha must still match
        with open(PIN_DIR, "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == fx["load_py_sha256"], "the pin's load.py moved under the fixture"


def test_every_pinned_column_is_supplied_and_the_frame_builds(tmp_path):
    cols, _ = pinned_columns()
    assert not [c for c in cols if c not in paper.LONG_COLUMN_SOURCES]
    root, none_run, lora_run = _corpus(tmp_path)
    df = _frame(root)
    assert list(df.columns)[:len(cols)] == cols and len(df) == 2
    # every NULL-by-design column names its reason
    for c, why in paper.LONG_COLUMN_NULL_REASONS.items():
        assert c in paper.LONG_COLUMN_SOURCES and why


# --------------------------------------------------------------------------------------------
# the degeneracy decomposition and the second at-cap estimand, from run_degeneration
# --------------------------------------------------------------------------------------------

def test_headline_row_reads_the_canonical_at_cap_and_the_decomposition_from_run_degeneration(tmp_path):
    root, none_run, lora_run = _corpus(tmp_path)
    df = _frame(root)
    lora = df[df.run_id == lora_run].iloc[0]
    # the CANONICAL re-tokenized reading takes at_cap_share; the producer's finish-reason reading rides beside it
    assert lora["at_cap_share"] == pytest.approx(0.05) and lora["at_cap_share_finish_reason"] == pytest.approx(0.03)
    assert lora["degen_truncation_only_share"] == pytest.approx(0.01) and lora["degen_loop_without_cap_share"] == pytest.approx(0.02)
    assert lora["degenerate_share"] == pytest.approx(0.30)
    basis = str(lora["degeneracy_basis"])
    assert "at_cap_share = at_token_cap_retokenized" in basis and "degenerate_share = union5" in basis
    assert "union over n = 200" in basis and "target replies (every target turn in the store)" in basis
    assert "CANONICAL re-tokenized estimand" in basis
    # the unsteered run has no degeneration row: the export's value (none here) stands and the basis says so
    none = df[df.run_id == none_run].iloc[0]
    assert _null(none["at_cap_share"]) and _null(none["at_cap_share_finish_reason"]) and _null(none["degen_truncation_only_share"])
    assert none["degeneracy_basis"] == "no degeneracy reading and no at-cap reading for this cell"


def test_without_a_retokenized_row_the_producer_value_stands_and_is_labelled(tmp_path):
    root, none_run, lora_run = _corpus(tmp_path, degeneration=False)
    # only the producer's at-cap on the headline row
    jid = I.judgment_id(lora_run, synth.JUDGE_ID)
    _write(root, "catchup2", "000001-frac.jsonl", [synth.rec("run_metric", {"run_id": lora_run, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5,
                                                                             "value": 0.25, "k": 1, "n": 4, "frac_at_token_cap": 0.2, "computed_by": "producer"}, synth.t(320))])
    H.ingest_once(root, workers=0, min_free=0)
    df = _frame(root)
    lora = df[df.run_id == lora_run].iloc[0]
    assert lora["at_cap_share"] == pytest.approx(0.2) and _null(lora["at_cap_share_finish_reason"])
    assert lora["degeneracy_basis"] == "at_cap_share = producer finish_reason == length (no re-tokenized reading for this cell); no degeneration reading"


def test_a_non_headline_metric_row_is_not_stamped(tmp_path):
    root, none_run, lora_run = _corpus(tmp_path)
    jid = I.judgment_id(lora_run, synth.JUDGE_ID)
    _write(root, "catchup3", "000001-held.jsonl", [synth.rec("run_metric", {"run_id": lora_run, "judgment_id": jid, "metric_id": "asr_held_in", "threshold": 0.5,
                                                                             "value": 0.5, "k": 2, "n": 4, "frac_at_token_cap": 0.2, "computed_by": "producer"}, synth.t(321))])
    H.ingest_once(root, workers=0, min_free=0)
    df = _frame(root)
    held = df[(df.run_id == lora_run) & (df.metric == "asr_held_in")].iloc[0]
    # the pin stamps HEADLINE rows only (config.HEADLINE per component); the other rows keep what the producer filed
    assert held["at_cap_share"] == pytest.approx(0.2) and _null(held["at_cap_share_finish_reason"])
    assert _null(held["degen_truncation_only_share"]) and held["degeneracy_basis"] is None


# --------------------------------------------------------------------------------------------
# the serving engine, the cross-engine flag, the judge-censored bound
# --------------------------------------------------------------------------------------------

def test_serving_engine_judge_blocked_lower_bound_cross_engine(tmp_path):
    root, none_run, lora_run = _corpus(tmp_path)
    df = _frame(root)
    lora = df[df.run_id == lora_run].iloc[0]
    none = df[df.run_id == none_run].iloc[0]
    assert lora["serving_engine"] == ENGINE and none["serving_engine"] is None
    assert lora["judge_blocked"] == 28 and lora["lower_bound"] is True
    assert _null(none["judge_blocked"]) and none["lower_bound"] is None      # the judgment counted no refusals: unknown, never False
    assert lora["cross_engine_contrast"] is True and none["cross_engine_contrast"] is None


def test_zero_refusals_is_a_point_not_a_bound(tmp_path):
    root, none_run, lora_run = _corpus(tmp_path, refusals=False)
    jid = I.judgment_id(lora_run, synth.JUDGE_ID)
    _write(root, "catchup4", "000001-j.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": lora_run, "judge_id": synth.JUDGE_ID, "finished_utc": synth.t(126),
                                                                       "n_items_judged": 4, "n_judge_refusals": 0, "status": "complete"}, synth.t(330))])
    H.ingest_once(root, workers=0, min_free=0)
    df = _frame(root)
    lora = df[df.run_id == lora_run].iloc[0]
    assert lora["judge_blocked"] == 0 and lora["lower_bound"] is False


def test_an_adapter_filed_spelling_wins_over_the_registry_derivation(tmp_path):
    """Precedence rule of LONG_COLUMN_SOURCES: the extra_json key the producer's adapter filed under the paper's spelling wins."""
    root, none_run, lora_run = _corpus(tmp_path)
    jid = I.judgment_id(lora_run, synth.JUDGE_ID)
    _write(root, "catchup5", "000001-m.jsonl", [synth.rec("run_metric", {"run_id": lora_run, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.25,
                                                                          "k": 1, "n": 4, "computed_by": "producer",
                                                                          "extra": {"serving_engine": "proxy_vllm+lora+interleaved", "judge_blocked": 3,
                                                                                    "degeneracy_basis": "filed by the adapter"}}, synth.t(340))])
    H.ingest_once(root, workers=0, min_free=0)
    df = _frame(root)
    lora = df[df.run_id == lora_run].iloc[0]
    assert lora["serving_engine"] == "proxy_vllm+lora+interleaved" and lora["judge_blocked"] == 3 and lora["degeneracy_basis"] == "filed by the adapter"


def test_the_two_decomposition_measures_are_in_the_vocabulary():
    assert {"truncation_only", "loop_without_cap"} <= set(V.DEGENERATION_MEASURES)


# --------------------------------------------------------------------------------------------
# `loop_share`: the writer names the detector its legacy column IS; the reader honours it
# --------------------------------------------------------------------------------------------
# The paper's `loop_share` is one column for two definitions: the r2 export's is the published loop rule (word 4-gram
# repetition ratio > 0.5, the `rep_ratio_loop` measure; the export's own note says so), the declared readers' (arditi,
# alphasteer read `classifier_share`; gsm8k passes its count under `classifier`) is the pinned loop CLASSIFIER's share
# (`classifier`). Both writers file the right per-definition mirror column; the reader read `frac_loop_first512`, a third
# statistic neither files as loop_share, so every r2 row read the first-512 share and every declared row NULL.


def _r2_registry(tmp_path, **kw):
    import _b1_fixtures as B1
    from sisteer.registry.adapters import r2_export as X
    paths = B1.synthetic_r2_export(tmp_path, **kw)
    seeds = B1.synthetic_seeds()
    ad = X.R2ExportAdapter(data_dir=paths["data"], rs=paths["rs"], seeds=seeds, objects=B1.resolver(seeds, tmp_path))
    objs = [ad.objects.resolve(B1.VIEW, c) for c in (B1.LORA, B1.RELU, B1.SUB, B1.PROMPTED)]
    root = B1.registry_root(tmp_path, seeds, objs)
    # two passes: the first over a fresh registry parks every contrast (the export's own unsteered runs are not folded yet; r2_export 11),
    # the fold makes them current and the second files them -- the steady state these tests read
    for i in (1, 2):
        rep, ing, ing2 = B1.run_and_ingest(root, ad, now=f"2026-10-03T0{i}:00:00.000Z")
        assert rep.groups_refreshed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    return root, ad, paths, B1


def _r2_frame(root):
    cols, _ = pinned_columns()
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return paper.paper_long_frame(conn, None, current_only=False, columns=cols)
    finally:
        conn.close()


def _row(df, **where):
    g = df
    for k, v in where.items():
        g = g[g[k] == v]
    assert len(g) == 1, (where, len(g))
    return g.iloc[0]


def test_r2_loop_share_is_the_exports_rep_ratio_value(tmp_path):
    root, ad, paths, B1 = _r2_registry(tmp_path)
    df = _r2_frame(root)
    lora = _row(df, view=B1.VIEW, cell_id=B1.LORA, eval_id="pasttense", metric="asr", threshold=0.5, decode_cap_tokens=4096)
    # the fixture files loop_share=0.1 and loop_share_first512=0.05 on this row: the paper's column is the former
    assert lora["loop_share"] == pytest.approx(0.1)
    none = _row(df, view=B1.VIEW, cell_id="none", eval_id="pasttense", metric="asr", threshold=0.5, decode_cap_tokens=4096)
    assert none["loop_share"] == pytest.approx(0.0)
    # the writer states which measure its legacy column names
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        hints = {json.loads(r[0]).get("loop_share_measure_legacy") for r in conn.execute("SELECT extra_json FROM run_metric").fetchall()}
    finally:
        conn.close()
    assert hints == {"rep_ratio_loop"}


def test_declared_writer_names_the_classifier_and_the_reader_reads_that_mirror(tmp_path):
    import registry_declared_fixtures as DF
    from test_registry_adapter_declared import _read_all
    from sisteer.registry.adapters import gsm8k_p200 as G8
    fx = DF.build_gsm8k_report(str(tmp_path / "g8k"))
    ad = G8.Gsm8kP200Adapter(report=fx["path"], later_reports=(), expect_pin=G8.STORE_PIN)
    recs, rep = _read_all(ad)
    metrics = [r["payload"] for r in recs if r["record_type"] == "run_metric"]
    assert metrics and all(m["extra"].get("loop_share_measure_legacy") == "classifier" for m in metrics)
    assert all(m.get("frac_degenerate_rep_ratio_loop") is None for m in metrics)
    # the reader, on rows of exactly that shape: the classifier mirror is the paper's loop_share, and a filled rep_ratio mirror
    # beside it does not displace it; a row with NO hint reads the paper's published definition (rep_ratio_loop)
    root, none_run, lora_run = _corpus(tmp_path, degeneration=False)
    jid = I.judgment_id(lora_run, synth.JUDGE_ID)
    jid0 = I.judgment_id(none_run, synth.JUDGE_ID)
    _write(root, "loop", "000001-loop.jsonl", [
        synth.rec("run_metric", {"run_id": lora_run, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 1, "n": 4,
                                 "frac_degenerate_classifier": 0.0, "frac_degenerate_rep_ratio_loop": 0.1, "computed_by": "producer",
                                 "extra": {"loop_share_measure_legacy": "classifier"}}, synth.t(400)),
        synth.rec("run_metric", {"run_id": none_run, "judgment_id": jid0, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 1, "n": 4,
                                 "frac_degenerate_classifier": 0.0, "frac_degenerate_rep_ratio_loop": 0.1, "computed_by": "producer"}, synth.t(401))])
    H.ingest_once(root, workers=0, min_free=0)
    df = _frame(root)
    assert df[df.run_id == lora_run].iloc[0]["loop_share"] == pytest.approx(0.0)
    assert df[df.run_id == none_run].iloc[0]["loop_share"] == pytest.approx(0.1)
    assert "loop_share = classifier" in str(df[df.run_id == lora_run].iloc[0]["degeneracy_basis"]) or \
        "loop_share = the writer" in str(df[df.run_id == lora_run].iloc[0]["degeneracy_basis"])


# --------------------------------------------------------------------------------------------
# the `_sampled4096` legacy spelling: instrument identity in the registry, the pin's eval id on the frame
# --------------------------------------------------------------------------------------------
# The paper spells the sampled re-run as a second EVAL id (`pasttense_sampled4096`) and `select_sampled4096` keys on the suffix.
# The registry's eval is the clean benchmark; the cap / sampling is INSTRUMENT identity (a distinct instrument_id with its own
# token_cap_tokens, decode_source, decode_seed_rule, n_trials). The r2 adapter kept the paper's spelling only in
# `extra.eval_id_legacy`, so the frame's `eval_id` (derived from the registry id) collided with the greedy rows' and the
# paper's selector found nothing.


def test_sampled4096_rows_are_a_distinct_instrument_and_keep_the_legacy_eval_spelling(tmp_path):
    root, ad, paths, B1 = _r2_registry(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        runs = conn.execute("SELECT r.eval_id, r.instrument_id, r.token_cap_tokens, i.decode_source, i.decode_seed_rule, i.n_trials, r.extra_json "
                            "FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id "
                            "WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = 'pasttense' ORDER BY r.token_cap_tokens", (B1.LORA,)).fetchall()
        legacy = {r[0] for r in conn.execute("SELECT eval_id_legacy FROM v_paper_long WHERE campaign_cell_id_legacy = ? AND eval_id = 'pasttense'", (B1.LORA,))}
    finally:
        conn.close()
    assert len(runs) == 2 and runs[0][1] != runs[1][1], "the greedy and the sampled reading of one cell are two instruments"
    greedy, sampled = runs
    assert (greedy[2], greedy[3], greedy[4], greedy[5]) == (4096, "greedy", "none", 1)
    assert (sampled[2], sampled[3], sampled[4], sampled[5]) == (8192, "vendor_recommendation", "per_item", 2)
    assert json.loads(sampled[6])["instrument_seed_id"] == "static_sampled_vendor_preset_4096_8192"
    # the view carries the paper's spelling as a column; the registry id stays the clean eval
    assert legacy == {"pasttense", "pasttense_sampled4096"}
    df = _r2_frame(root)
    cell = df[(df.view == B1.VIEW) & (df.cell_id == B1.LORA) & (df.metric == "asr") & (df.threshold == 0.5) & df.eval_id.astype(str).str.startswith("pasttense")]
    assert set(cell.eval_id) == {"pasttense", "pasttense_sampled4096"}
    # `select_sampled4096`-style: the suffixed ids at the metric / threshold, one record per cell at its rung
    picked = cell[cell.eval_id.astype(str).str.endswith("_sampled4096")]
    assert len(picked) == 1 and picked.iloc[0]["decode_cap_tokens"] == 8192 and picked.iloc[0]["value"] == pytest.approx(0.3)
    assert set(paper.LONG_REGISTRY_COLUMNS) >= {"eval_id_legacy"}


def test_eval_id_legacy_is_derived_from_the_sampled_instrument_when_no_writer_spelled_it():
    """A row of another adapter on a sampled instrument seed carries no `eval_id_legacy`; the paper's spelling is the legacy eval id
    plus the suffix, derived from the instrument seed the run names — never from a string match on the id."""
    row = {"eval_id": "or_bench_hard", "eval_id_legacy": None}
    extras = [{}, {"instrument_seed_id": "overrefusal_orbench_sampled_vendor_preset_ladder"}]
    assert paper._derive("eval_id", row, extras, {}) == "or_bench_hard_1k_sampled4096"
    assert paper._derive("eval_id", {"eval_id": "or_bench_hard", "eval_id_legacy": None}, [{}, {"instrument_seed_id": "overrefusal_greedy_512_controls"}], {}) == "or_bench_hard_1k"
    assert set(paper.SAMPLED4096_INSTRUMENT_SEEDS) == {"static_sampled_vendor_preset_4096_8192", "overrefusal_xstest_sampled_vendor_preset_ladder",
                                                       "overrefusal_orbench_sampled_vendor_preset_ladder"}


# --------------------------------------------------------------------------------------------
# r2's degeneration statistics are `run_degeneration` measures, never metric rows
# --------------------------------------------------------------------------------------------
# The export files its degeneration statistics as metric rows per cell (`at_cap_share`, `loop_share`, `degenerate_share`,
# `degen_<detector>_share`, the union instrument's rows) and the adapter passed the spellings through as `run_metric.metric_id`.
# The schema's door for a degeneration statistic is `run_degeneration` (measure x scope, counts, measurement_state), reaching
# `run_metric` only through the `frac_*` mirrors on the headline row; the metric-id CHECK did not literally name the `_share`
# spellings, so it refused none of them — it now does, and the adapter routes them.

ROUTED = ("at_cap_share", "loop_share", "degenerate_share", "degen_classifier_share", "degen_contentless_share", "degen_rep_ratio_loop_share",
          "degen_zlib_collapse_share", "degenerate_share_union5", "at_cap_share_retokenized", "at_cap_share_finish_reason",
          "degen_truncation_only_share", "degen_loop_without_cap_share", "defended_degenerate_share")


def test_the_check_now_refuses_the_exports_degeneration_spellings_and_spares_qualified_ids():
    banned = set(V.banned_metric_ids())
    assert set(ROUTED) <= banned, set(ROUTED) - banned
    # qualified statistics the export also carries stay metric rows (no measure names them): never banned
    assert not (banned & {"residual_at_cap_share_bon_4096", "at_cap_share_all", "at_cap_share_compliance", "goal_producer_capped_share",
                          "defended_collapse_share_lower", "audit_any_loop_share", "recovered_at_cap_share", "asr", "acc", "petri_score"})


def test_r2_degeneration_metric_rows_are_routed_to_run_degeneration_and_mirrored(tmp_path):
    root, ad, paths, B1 = _r2_registry(tmp_path, degeneration_rows=True)
    assert len(ad.refusals) == 0, ad.refusals.items
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        ids = {r[0] for r in conn.execute("SELECT DISTINCT metric_id FROM run_metric")}
        assert not (ids & set(ROUTED)), ids & set(ROUTED)
        run = conn.execute("SELECT run_id FROM run WHERE campaign_cell_id_legacy = ? AND eval_id = 'crescendo' AND token_cap_tokens = 512", (B1.LORA,)).fetchone()[0]
        deg = {(r[0], r[1]): r for r in conn.execute("SELECT measure, scope, rate, numerator, denominator, denominator_unit, measurement_state, extra_json, "
                                                      "definition_version FROM run_degeneration WHERE run_id = ?", (run,))}
        got = {k: (v[2], v[3], v[4], v[5], v[6]) for k, v in deg.items()}
        assert got[("at_token_cap", "all")] == (pytest.approx(0.6), 60, 100, "replies", "measured")
        assert got[("rep_ratio_loop", "all")] == (pytest.approx(0.02), 2, 100, "replies", "measured")
        assert got[("loop_or_at_cap", "all")] == (pytest.approx(0.6), 60, 100, "replies", "measured")
        assert got[("classifier", "all")][0] == pytest.approx(0.01) and got[("contentless", "all")][0] == 0.0 and got[("zlib_collapse", "all")][0] == 0.0
        assert got[("union5", "all")] == (pytest.approx(0.62), 62, 100, "replies", "measured")
        assert got[("at_token_cap_retokenized", "all")][0] == pytest.approx(0.58) and got[("at_token_cap_finish_reason", "all")][0] == pytest.approx(0.6)
        assert got[("truncation_only", "all")][0] == pytest.approx(0.5) and got[("loop_without_cap", "all")][0] == pytest.approx(0.01)
        assert got[("defended_degenerate", "defended_only")] == (pytest.approx(0.5), 1, 2, "items", "measured")
        # the plain `loop_share` row and the union instrument's `degen_rep_ratio_loop_share` are ONE measure: both spellings are kept
        rr = json.loads(deg[("rep_ratio_loop", "all")][7])
        assert {p["producer_metric_spelling"] for p in rr["r2_rows"]} == {"loop_share", "degen_rep_ratio_loop_share"}
        assert "98ae9114672c" in deg[("union5", "all")][8]
        # the headline row carries the union's mirrors; the producer's own columns stand where they were filed
        head = conn.execute("SELECT frac_at_token_cap, frac_degenerate_rep_ratio_loop, frac_degenerate_union5, frac_degenerate_classifier, "
                            "frac_degenerate_contentless, frac_degenerate_zlib_collapse, frac_defended_degenerate, extra_json FROM run_metric "
                            "WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", (run,)).fetchone()
        assert head[0] == pytest.approx(0.6) and head[1] == pytest.approx(0.02) and head[2] == pytest.approx(0.62) and head[3] == pytest.approx(0.01)
        assert head[4] == 0.0 and head[5] == 0.0 and head[6] == pytest.approx(0.5)
        assert json.loads(head[7])["degenerate_share"] == pytest.approx(0.6)
    finally:
        conn.close()
    assert ad.stats["degeneration"] >= 12 and ad.stats["degeneration_routed_rows"] == 13


def test_paper_frame_reproduces_the_pins_degeneracy_columns_from_the_routed_rows(tmp_path):
    root, ad, paths, B1 = _r2_registry(tmp_path, degeneration_rows=True)
    df = _r2_frame(root)
    head = _row(df, view=B1.VIEW, cell_id=B1.LORA, eval_id="crescendo", metric="asr", threshold=0.5, decode_cap_tokens=512)
    # what `select_degeneracy_rows` stamps at the pin: the canonical at-cap, the finish-reason estimand beside it, the published
    # loop rule, the five-detector union, the decomposition, and a basis naming each
    assert head["at_cap_share"] == pytest.approx(0.58) and head["at_cap_share_finish_reason"] == pytest.approx(0.6)
    assert head["loop_share"] == pytest.approx(0.02) and head["degenerate_share"] == pytest.approx(0.62)
    assert head["degen_truncation_only_share"] == pytest.approx(0.5) and head["degen_loop_without_cap_share"] == pytest.approx(0.01)
    basis = str(head["degeneracy_basis"])
    assert "at_cap_share = at_token_cap_retokenized" in basis and "degenerate_share = union5" in basis and "loop_share = rep_ratio_loop" in basis
    assert "union over n = 100 (target replies (every target turn in the store), n = 100)" in basis    # the pin's _denominator_rule reading of the note
    # the cell's unsteered anchor has no union rows: the export's own columns stand (the two-detector union is the row's degenerate_share)
    none = _row(df, view=B1.VIEW, cell_id="none", eval_id="crescendo", metric="asr", threshold=0.5, decode_cap_tokens=512)
    assert none["at_cap_share"] == pytest.approx(0.7) and none["loop_share"] == pytest.approx(0.01) and none["degenerate_share"] == pytest.approx(0.7)
    assert _null(none["at_cap_share_finish_reason"]) and "no re-tokenized reading" in str(none["degeneracy_basis"])
    # the static lora cell (no degeneration metric rows at all): the export's columns
    st = _row(df, view=B1.VIEW, cell_id=B1.LORA, eval_id="pasttense", metric="asr", threshold=0.5, decode_cap_tokens=4096)
    assert st["at_cap_share"] == pytest.approx(0.2) and st["loop_share"] == pytest.approx(0.1) and st["degenerate_share"] == pytest.approx(0.2)
    # the degeneration statistics are no longer rows of the paper frame
    assert not (set(df.metric.astype(str)) & set(ROUTED))


def test_r2compat_round_trips_the_routed_rows(tmp_path):
    import pandas as pd
    from sisteer.registry import r2compat as RC
    root, ad, paths, B1 = _r2_registry(tmp_path, degeneration_rows=True)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        df = RC.render_frame(conn)
    finally:
        conn.close()
    ref = pd.read_parquet(paths["parquet"])
    rep = RC.compare(df, ref)
    assert len(df) == len(ref) == 17 + len(ROUTED)
    assert rep["only_rendered"] == 0 and rep["only_reference"] == 0, rep
    assert rep["columns_differing"] == {}, json.dumps(rep["columns_differing"], indent=1)
