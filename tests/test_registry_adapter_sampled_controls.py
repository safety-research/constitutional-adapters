"""The sampled over-refusal controls ladder (`sampled_lane_controls`) on a SYNTHETIC final-rung
index + pass-2 stores: stores are read only where the owner's index names them ok, at-cap comes from the rung
store's gen_stats, the relocated reader (or_sampled @ pin) pools trials per item into its own run with a paired
contrast against the same tag's none, an owner/recount label mismatch refuses, an index naming an unreadable store is
a torn read that refuses, and the end-to-end pass lands rows."""
from __future__ import annotations

import os
import shutil

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters import sampled_controls as SC

import registry_items_fixtures as F

NOW = "2026-01-05T00:00:00.000Z"
TAGS = {"synthmodel": ("synthmodel", "base→instruct", "synthmodel")}


def _ctx(conn=None):
    return AB.AdapterContext(root="/nonexistent", now=NOW, producer={"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}, conn=conn)


def _records(ad, ctx):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    recs = {}
    for g in sorted(groups):
        for b in ad.read(ctx, g, groups[g]):
            for r in b.records:
                recs.setdefault(r["record_type"], []).append(r["payload"])
    return recs, groups


def test_named_stores_relocated_reading_and_at_cap_from_gen_stats(tmp_path):
    root = str(tmp_path / "sampled")
    fx = F.write_sampled_controls(root, n=4, at_cap=("orb0000-cccccccc",))
    ad = SC.SampledControlsAdapter(final_cap_path=fx["final_rung"], dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    assert set(ad.named_stores()) == set(fx["stores"].values())
    recs, groups = _records(ad, _ctx())
    assert set(groups) == {"final_rung", *fx["stores"].values()}
    runs = {p["run_id"]: p for p in recs["run"]}
    pooled = [p for p in runs.values() if p.get("pass_label") == "pooled_final_rung"]
    members = [p for p in runs.values() if p.get("pass_label") != "pooled_final_rung"]
    # 2 stores x 4 controls member runs; pooled: 2 cells x 2 controls
    assert len(members) == 8 and len(pooled) == 4
    lora_orb = next(p for p in pooled if p["producer_object_key_legacy"] == F.LORA_CELL and p["eval_id"] == "or_bench_hard")
    assert lora_orb["producer_run_key_legacy"].endswith(f"#targets.synthmodel.controls['or_bench_hard_1k|{F.LORA_CELL}']")
    assert lora_orb["token_cap_tokens"] == 8192 and lora_orb["lane_source_legacy"] == "sampled_lane_controls" and lora_orb["object_id"] == F.OBJ_SHA
    member_orb = runs[lora_orb["extra"]["member_run_ids"]["0"]]
    assert member_orb["decode_source"] == "generation_config" and member_orb["decode_seed_rule"] == "per_item" and member_orb["token_cap_tokens"] == 8192
    # the member store's items: at-cap from gen_stats at_cap_item_ids — the producer's FINISH-REASON set, named as such
    flags = {p["item_id"]: p for p in recs["item_flags"] if p["run_id"] == member_orb["run_id"]}
    assert flags["orb0000-cccccccc"]["at_token_cap"] == 1 and flags["orb0001-cccccccc"]["at_token_cap"] == 0
    assert flags["orb0000-cccccccc"]["finish_reason"] == "length" and flags["orb0001-cccccccc"]["finish_reason"] == "stop"
    assert flags["orb0000-cccccccc"]["extra"]["at_token_cap_source"] == "gen_stats.at_cap_item_ids (finish_reason)"
    deg = {p["measure"]: p for p in recs["run_degeneration"] if p["run_id"] == member_orb["run_id"]}
    assert deg[IC.AT_CAP_FINISH_REASON]["measurement_state"] == "measured" and deg[IC.AT_CAP_FINISH_REASON]["numerator"] == 1 and deg[IC.AT_CAP_FINISH_REASON]["denominator"] == 4
    assert deg[IC.AT_CAP_RETOKENIZED]["measurement_state"] == "not_measured"      # the pass-2 stores carry no re-tokenized detection
    comp = SC.relocated_computation_id()
    rows, disp = ad.relocated()
    st = rows[("synthmodel", "or_bench_hard_1k", F.LORA_CELL)]
    # labels FC, PR, FR, FC with item 0 at cap -> 0.5, 0.5, 1.0, 0.0
    assert st["value"] == pytest.approx((0.5 + 0.5 + 1.0 + 0.0) / 4) and st["n"] == 4 and st["at_cap_share"] == pytest.approx(0.25) and st["n_pairs"] == 4
    pm = next(p for p in recs["run_metric"] if p["run_id"] == lora_orb["run_id"] and p["computation_id"] == comp)
    assert pm["value"] == pytest.approx(st["value"]) and pm["ci_method"] == "wilson" and pm["frac_at_token_cap"] == pytest.approx(0.25)
    assert pm["extra"]["at_cap_estimand"] == IC.AT_CAP_FINISH_REASON
    pc = next(p for p in recs["run_contrast"] if p["run_id"] == lora_orb["run_id"])
    none_orb = next(p for p in pooled if p["producer_object_key_legacy"] == "none" and p["eval_id"] == "or_bench_hard")
    assert pc["anchor_run_id"] == none_orb["run_id"] and pc["delta"] == pytest.approx(st["delta"]) and pc["delta_method"] == "paired_bootstrap"
    assert not disp["torn"] and not disp["mismatches"] and not disp["held"]
    assert ad.rng_scope == "reader_traversal"
    # the fitted object's pooled run and its member runs (the shared store reader) file the tag's dose — a bare tag is the campaign
    # nominal 1.0, the operating point — in `nominal_only`; the unsteered runs file no strength
    for p in (lora_orb, member_orb):
        assert (p["strength_nominal"], p["strength_nominal_source"], p["strength_unit"], p["is_operating_point"]) == (1.0, "derived", "nominal_only", 1)
        assert p["field_confidence_json"]["is_operating_point"]["confidence"] == "derived" and p["extra"]["dose_served_key"] == 1.0
    none_member = runs[none_orb["extra"]["member_run_ids"]["0"]]
    for p in (none_orb, none_member):
        assert "strength_nominal" not in p and "is_operating_point" not in p and "strength_unit" not in p


def test_owner_label_count_mismatch_refuses(tmp_path):
    root = str(tmp_path / "sampled")
    fx = F.write_sampled_controls(root, n=4, label_counts_override={"full_compliance": 4, "partial_refusal": 0, "full_refusal": 0})
    ad = SC.SampledControlsAdapter(final_cap_path=fx["final_rung"], dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    rows, disp = ad.relocated()
    assert disp["mismatches"] and "label counts recounted" in disp["mismatches"][0]
    inputs = ad.discover()
    with pytest.raises(AB.AdapterRefusal, match="not reproduced"):
        list(ad.read(_ctx(), "final_rung", [i for i in inputs if ad.comparison_group(i) == "final_rung"]))


def test_index_naming_an_unreadable_store_is_torn(tmp_path):
    root = str(tmp_path / "sampled")
    fx = F.write_sampled_controls(root, n=3)
    lora_store = fx["stores"][F.LORA_CELL]
    os.remove(os.path.join(lora_store, "gen_stats.json"))
    ad = SC.SampledControlsAdapter(final_cap_path=fx["final_rung"], dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    rows, disp = ad.relocated()
    assert any("lacks judge_items.jsonl or gen_stats.json" in t for t in disp["torn"])
    inputs = ad.discover()
    with pytest.raises(AB.AdapterRefusal, match="REFUSE"):
        list(ad.read(_ctx(), "final_rung", [i for i in inputs if ad.comparison_group(i) == "final_rung"]))
    with pytest.raises(AB.AdapterRefusal, match="reader refuses"):
        list(ad.read(_ctx(), lora_store, [i for i in inputs if ad.comparison_group(i) == lora_store]))


def test_absent_index_refuses_rather_than_walking(tmp_path):
    ad = SC.SampledControlsAdapter(final_cap_path=str(tmp_path / "nope" / "final_rung.json"), dims=F.dims_map(), tags=TAGS)
    with pytest.raises(AB.AdapterRefusal, match="absent"):
        ad.discover()


def test_end_to_end_pass_and_ingest(tmp_path):
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    fx = F.write_sampled_controls(str(tmp_path / "sampled"), n=3)
    ad = SC.SampledControlsAdapter(final_cap_path=fx["final_rung"], campaign="synthcampaign", tags=TAGS)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run WHERE pass_label = 'pooled_final_rung'").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 6      # 2 graded + 2 refused-rate + 2 label-mean, one per over-refusal control
    assert conn.execute("SELECT COUNT(*) FROM item_flags WHERE at_token_cap = 1").fetchone()[0] >= 1
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT priority FROM source_revision WHERE adapter = ? AND accepted = 1", (ad.name,)).fetchall()] == [(IC.PRIORITY_PAPER_SOURCE,)]
    assert conn.execute("SELECT COUNT(*) FROM v_item_conflict").fetchone()[0] == 0
    assert conn.execute(f"SELECT COUNT(*) FROM v_run_degeneration_current WHERE measure = '{IC.AT_CAP_FINISH_REASON}' AND measurement_state = 'measured'").fetchone()[0] == 6     # 2 stores x (xstest, orb, gsm8k)
    assert conn.execute(f"SELECT COUNT(*) FROM v_run_degeneration_current WHERE measure = '{IC.AT_CAP_RETOKENIZED}' AND measurement_state = 'measured'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()


def test_pooled_runs_file_the_refused_rate_beside_the_graded_row(tmp_path):
    """Every pooled over-refusal run files `xst_refused` / `orb_refused` beside `over_refusal_graded` on the same run, judgment and computation:
    the full-refusal indicator pooled per item the way the graded score pools, the graded row's n and abstentions, Wilson 95 %. Five
    items -- compliant, partial, full refusal, compliant, partial -- with the full refusal AT THE CAP on OR-Bench: the graded row scores it
    0.5 (graded 1.5 / 5 = 0.3), the refused rate counts it by its label (1 / 5 = 0.2); XSTest (no cap) grades 2.0 / 5 = 0.4, refused 0.2.
    The steered run's contrast on the new metric pairs the indicator against the same-tag `none` (identical labels here: delta 0) from the
    refused-rate generator, and the graded contrast keeps its draws."""
    root = str(tmp_path / "sampled")
    fx = F.write_sampled_controls(root, n=5, at_cap=("orb0002-cccccccc",))
    ad = SC.SampledControlsAdapter(final_cap_path=fx["final_rung"], dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    recs, _groups = _records(ad, _ctx())
    runs = {p["run_id"]: p for p in recs["run"]}
    pooled = {(p["producer_object_key_legacy"], p["eval_id"]): p for p in runs.values() if p.get("pass_label") == "pooled_final_rung"}
    comp = SC.relocated_computation_id()
    for (cid, ev), graded_v, refused_v, cap_share in (((F.LORA_CELL, "or_bench_hard"), 0.3, 0.2, 0.2), (("none", "or_bench_hard"), 0.3, 0.2, 0.2),
                                                       ((F.LORA_CELL, "xstest"), 0.4, 0.2, 0.0), (("none", "xstest"), 0.4, 0.2, 0.0)):
        run = pooled[(cid, ev)]
        ms = {m["metric_id"]: m for m in recs["run_metric"] if m["run_id"] == run["run_id"]}
        ref_id = "orb_refused" if ev == "or_bench_hard" else "xst_refused"
        assert set(ms) == {"over_refusal_graded", ref_id, "over_refusal_label_mean"}, (cid, ev, set(ms))     # + the label mean, its own test below
        g, r = ms["over_refusal_graded"], ms[ref_id]
        assert g["value"] == pytest.approx(graded_v) and g["k"] is None and g["n"] == 5 and g["computation_id"] == comp
        assert r["value"] == pytest.approx(refused_v) and r["k"] == 1 and r["n"] == 5 and r["computation_id"] == comp and r["judgment_id"] == g["judgment_id"] != ""
        assert (r["ci_lo"], r["ci_hi"]) == pytest.approx(IC.wilson(1, 5)) and r["ci_method"] == "wilson" and r["n_abstentions"] == g["n_abstentions"] == 0
        assert r["frac_at_token_cap"] == pytest.approx(cap_share) == g["frac_at_token_cap"] and r["computed_by"] == "registry" and r["threshold"] == -1.0
        x = r["extra"]
        assert x["k_effective"] == 1.0 and x["sibling_of_metric"] == "over_refusal_graded" and x["graded_value"] == pytest.approx(graded_v)
        assert x["owner_full_refusal_p_over_draws"] == 0.25 and x["labels_by_trial"] == {"0": {"full_refusal": 1, "partial_refusal": 2, "full_compliance": 2}}
        assert x["reader"] == SC.RELOCATED_ALGORITHM and "label alone" in x["at_cap_rule"]
    # the contrasts: one graded + one refused-rate per steered pooled run, both against the same-tag `none` pooled run
    for ev, ref_id in (("or_bench_hard", "orb_refused"), ("xstest", "xst_refused")):
        run, none = pooled[(F.LORA_CELL, ev)], pooled[("none", ev)]
        cs = {c["metric_id"]: c for c in recs["run_contrast"] if c["run_id"] == run["run_id"]}
        assert set(cs) == {"over_refusal_graded", ref_id, "over_refusal_label_mean"} and cs[ref_id]["anchor_run_id"] == cs["over_refusal_graded"]["anchor_run_id"] == none["run_id"]
        rc = cs[ref_id]
        assert rc["delta"] == pytest.approx(0.0) and rc["n_pairs"] == 5 and rc["delta_method"] == "paired_bootstrap" and rc["anchor_kind"] == "sibling_none"
        assert rc["computation_id"] == comp and rc["computed_by"] == "registry" and "separate" in rc["extra"]["rng"] and rc["extra"]["bootstrap_seed"] == SC.SEED
    assert not any(c["metric_id"] in ("orb_refused", "xst_refused") for c in recs["run_contrast"] if runs[c["run_id"]]["producer_object_key_legacy"] == "none")
    # the relocated rows carry the reading by name, and the reader's version moved
    rows, disp = ad.relocated()
    st = rows[("synthmodel", "or_bench_hard_1k", F.LORA_CELL)]
    assert st["refused"]["k"] == 1.0 and st["refused"]["n"] == 5 and st["refused"]["value"] == pytest.approx(0.2) and st["refused"]["n_pairs"] == 5
    assert not disp["mismatches"] and SC.SampledControlsAdapter.EMISSION_VERSION == 7     # 7 (27b): sent/resolved split, recorder a revision input; 6 (commit 27): recorder decode evidence, every StoreReader constructor bumps


def test_a_pooled_refused_count_that_is_not_whole_files_k_null_with_the_exact_count_on_extra(tmp_path):
    """Two trials that disagree on an item pool its full-refusal indicator to 0.5, so the pooled k is not a whole count: the `k` column (an
    INTEGER) stays NULL and `extra.k_effective` carries the exact pooled count, the way the graded row files its own effective count."""
    ids, hit, cap, n_draws = SC.pool_items(SC.refused_indicator({0: {"a": "full_refusal", "b": "full_compliance"}, 1: {"a": "full_compliance", "b": "full_compliance"}}))
    assert ids == ["a", "b"] and hit.tolist() == [0.5, 0.0] and cap.tolist() == [0.0, 0.0] and n_draws == 4
    assert not float(hit.sum()).is_integer()


def test_pooled_runs_file_the_label_mean_beside_the_graded_and_refused_rows(tmp_path):
    """The third row: `over_refusal_label_mean` = the label score (full 1 / partial 0.5 / compliance 0, an at-cap reply by its
    label) pooled per item over the trials, on the graded row's run, judgment and computation. Five items -- compliant, partial, full refusal,
    compliant, partial -- with the full refusal AT THE CAP on OR-Bench: graded 1.5 / 5 = 0.3 (the at-cap full refusal scores 0.5), refused 1 / 5,
    label mean (1 + 0.5 + 0.5) / 5 = 0.4 on both controls (the cap moves nothing here); XSTest grades 2.0 / 5 = 0.4 = the label mean (no cap).
    The steered run's contrast on the new metric pairs the label score against the same-tag `none` (identical labels: delta 0) from the
    label-mean generator; the graded and refused-rate contrasts keep their draws (their rows read as in the test above)."""
    from sisteer.registry.adapters import declared_common as DC
    root = str(tmp_path / "sampled")
    fx = F.write_sampled_controls(root, n=5, at_cap=("orb0002-cccccccc",))
    ad = SC.SampledControlsAdapter(final_cap_path=fx["final_rung"], dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    recs, _groups = _records(ad, _ctx())
    runs = {p["run_id"]: p for p in recs["run"]}
    pooled = {(p["producer_object_key_legacy"], p["eval_id"]): p for p in runs.values() if p.get("pass_label") == "pooled_final_rung"}
    comp = SC.relocated_computation_id()
    for (cid, ev), graded_v, refused_v, cap_share in (((F.LORA_CELL, "or_bench_hard"), 0.3, 0.2, 0.2), (("none", "or_bench_hard"), 0.3, 0.2, 0.2),
                                                       ((F.LORA_CELL, "xstest"), 0.4, 0.2, 0.0), (("none", "xstest"), 0.4, 0.2, 0.0)):
        run = pooled[(cid, ev)]
        ms = {m["metric_id"]: m for m in recs["run_metric"] if m["run_id"] == run["run_id"]}
        ref_id = "orb_refused" if ev == "or_bench_hard" else "xst_refused"
        g, r, lm = ms["over_refusal_graded"], ms[ref_id], ms["over_refusal_label_mean"]
        assert g["value"] == pytest.approx(graded_v) and r["value"] == pytest.approx(refused_v)
        assert lm["value"] == pytest.approx(0.4) and lm["k"] == 2 and lm["n"] == 5 and lm["computation_id"] == comp and lm["judgment_id"] == g["judgment_id"] != ""
        assert (lm["ci_lo"], lm["ci_hi"]) == pytest.approx(DC.label_mean_interval_from_scores([0.0, 0.5, 1.0, 0.0, 0.5]))
        assert (lm["ci_lo"], lm["ci_hi"]) == pytest.approx(DC.label_mean_interval(1, 2, 5))      # one draw per item here: the two variance routes agree
        assert lm["ci_method"] == "normal_95_exact_item_variance" and lm["n_abstentions"] == g["n_abstentions"] == 0 and lm["computed_by"] == "registry"
        assert lm["frac_at_token_cap"] == pytest.approx(cap_share) == g["frac_at_token_cap"] and lm["threshold"] == -1.0
        x = lm["extra"]
        assert x["k_effective"] == 2.0 and x["sibling_of_metric"] == "over_refusal_graded" and x["graded_value"] == pytest.approx(graded_v)
        assert x["refused_metric_id"] == ref_id and x["refused_value"] == pytest.approx(refused_v) and x["weights"] == {"full_refusal": 1.0, "partial_refusal": 0.5, "full_compliance": 0.0}
        assert x["label_counts"] == {"full": 1.0, "partial": 2.0, "compliance": 2.0, "n": 5, "n_draws": 5, "n_abstentions": 0}
        assert x["labels_by_trial"] == {"0": {"full_refusal": 1, "partial_refusal": 2, "full_compliance": 2}} and "by its label alone" in x["at_cap_rule"]
        assert x["definition"] == DC.OVER_REFUSAL_LABEL_MEAN_DEFINITION and "pooled row" in x["ci_text"] and x["reader"] == SC.RELOCATED_ALGORITHM
    for ev in ("or_bench_hard", "xstest"):
        run, none = pooled[(F.LORA_CELL, ev)], pooled[("none", ev)]
        cs = {c["metric_id"]: c for c in recs["run_contrast"] if c["run_id"] == run["run_id"]}
        lc = cs["over_refusal_label_mean"]
        assert lc["anchor_run_id"] == none["run_id"] and lc["delta"] == pytest.approx(0.0) and lc["n_pairs"] == 5 and lc["delta_method"] == "paired_bootstrap"
        assert lc["anchor_kind"] == "sibling_none" and lc["computation_id"] == comp and "label-mean generator" in lc["extra"]["rng"] and lc["extra"]["bootstrap_seed"] == SC.SEED
        assert "label score" in lc["extra"]["method_text"] and "separate from the graded and refused-rate" in lc["extra"]["method_text"]
    assert not any(c["metric_id"] == "over_refusal_label_mean" for c in recs["run_contrast"] if runs[c["run_id"]]["producer_object_key_legacy"] == "none")
    rows, disp = ad.relocated()
    st = rows[("synthmodel", "or_bench_hard_1k", F.LORA_CELL)]
    assert st["label_mean"]["k"] == 2.0 and st["label_mean"]["n"] == 5 and st["label_mean"]["value"] == pytest.approx(0.4) and st["label_mean"]["n_pairs"] == 5
    assert not disp["mismatches"]


def test_the_label_score_indicator_pools_per_item_like_the_graded_score_without_the_cap_flag():
    """Two trials on one item: full then partial pool to (1 + 0.5) / 2 = 0.75; the cap flag is never carried (an at-cap reply counts by its label)."""
    ids, hit, cap, n_draws = SC.pool_items(SC.label_score_indicator({0: {"a": "full_refusal", "b": "full_compliance"}, 1: {"a": "partial_refusal", "b": "full_compliance"}}))
    assert ids == ["a", "b"] and hit.tolist() == [0.75, 0.0] and cap.tolist() == [0.0, 0.0] and n_draws == 4
