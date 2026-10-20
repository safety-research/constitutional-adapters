"""The sampled static lane (`sampled_lane`) on a SYNTHETIC sampled root: per-trial runs keyed on the
grades dir + cell with the object resolved from the standup record, per-goal item rows under the Haiku judgment with
at-cap flags and the aggregate pointer, producer asr per threshold, the relocated pooled reading (paperlib.load
.load_sampled @ pin) as its own run with contrasts, refusals when the standup names no object, and the end-to-end pass.

Since emission 3 every pooled contrast is decided by the pass's anchor currency
(`sampled_static.POOLED_CONTRAST_RULE`): the in-memory reads here run on a frame PRIMED from the reader's own emissions
(`_anchor_frame_fixtures`, the steady state after the fold); the end-to-end pass is two passes (the first parks every pooled
comparison, the fold makes the pooled `none` current, the second files); a tag without a `none` aggregate parks by name instead
of skipping in silence; a re-picked anchor refuses by name; a read without a connection refuses the group and the smoke says so. The
`$.row_refusals` ledger is per group and per revision (`lineage_common.ROW_REFUSALS_LEDGER_RULE`): keyed on the
grades-dir group, never on its first file, and a re-activated revision carries its own ledger, never the adapter's last read's."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters import sampled_static as SS

import _anchor_frame_fixtures as AF
import _reanchor_fixtures as RX
import registry_items_fixtures as F

NOW = "2026-01-05T00:00:00.000Z"
TAGS = {"synthmodel": ("synthmodel", "base→instruct", "synthmodel")}
PRODUCER = {"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}


def _ctx(conn=None):
    return AB.AdapterContext(root="/nonexistent", now=NOW, producer=dict(PRODUCER), conn=conn)


def _primed(ad, frame=None):
    """The context of the steady state: bound to a frame primed from the adapter's own emissions over every group (or to `frame`)."""
    return AF.primed_ctx(ad, now=NOW, producer=PRODUCER, root="/nonexistent", frame=frame)


def _records(ad, ctx=None):
    """Every group read on `ctx` -- a primed context when none is given (a read without a connection refuses a group that intends a pooled contrast)."""
    ctx = ctx if ctx is not None else _primed(ad)
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


def test_per_trial_runs_items_and_relocated_pooled_reading(tmp_path):
    root = str(tmp_path / "sampled")
    paths = F.write_sampled_root(root)
    ad = SS.SampledStaticAdapter(root=root, dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    recs, groups = _records(ad)
    inst = F.dims_map().instrument(IC.STATIC_INSTRUMENT_SEED, "pasttense", F.MODEL_IT)[0]
    runs = {p["run_id"]: p for p in recs["run"]}
    # 2 trials x 2 cells per-trial runs + 2 pooled runs
    per_trial = [p for p in runs.values() if p["pass_label"] != "pooled_t0_t1"]
    pooled = [p for p in runs.values() if p["pass_label"] == "pooled_t0_t1"]
    assert len(per_trial) == 4 and len(pooled) == 2
    t0 = next(p for p in per_trial if p["producer_run_key_legacy"] == os.path.basename(paths["t0"]) and p["producer_object_key_legacy"] == F.LORA_CELL)
    assert t0["run_id"] == I.run_id("synthcampaign", os.path.basename(paths["t0"]), F.LORA_CELL, "pasttense", inst)
    assert t0["object_id"] == F.OBJ_SHA and t0["model_id"] == F.MODEL_IT and t0["model_revision"] == "c" * 40 and t0["served_max_num_seqs"] == 8
    assert t0["token_cap_tokens"] == 8192 and t0["token_cap_rung_index"] == 1 and t0["decode_source"] == "generation_config" and t0["transfer_kind"] == "base_to_instruct"
    assert t0["field_confidence_json"]["object_id"]["source"].startswith("serving_records")
    # items: one strongreject_best row per goal under the Haiku judgment; flags carry capped + the aggregate pointer
    hj = F.judge_ids()["strongreject_v1_haiku45"]
    items = [p for p in recs["item_result"] if p["run_id"] == t0["run_id"]]
    assert len(items) == 5 and all(p["metric_id"] == "strongreject_best" and p["judgment_id"] == I.judgment_id(t0["run_id"], hj) and p["api_pool"] == "lowprio" for p in items)
    flags = {p["item_id"]: p for p in recs["item_flags"] if p["run_id"] == t0["run_id"]}
    agg_sha = AB.check_input(os.path.join(paths["t0"], "two_si_pasttense_aggregate.json")).sha256
    assert flags["goal_000"]["generation_artifact_id"] == agg_sha and flags["goal_000"]["generation_offset"] == 0
    # per_goal.capped is the attack's FORWARD-cap flag, not a token-cap reading: no at_token_cap per goal, the flag parked in extra
    assert all(p["at_token_cap"] is None and p["finish_reason"] is None for p in flags.values())
    assert flags["goal_004"]["extra"]["forward_capped"] is False and "forward_capped" in flags["goal_000"]["extra"]
    # producer asr rows at 3 thresholds (frac_at_token_cap NULL: n_capped counts the forward-cap flag); degeneration rows per declared measure
    asr = [p for p in recs["run_metric"] if p["run_id"] == t0["run_id"] and p["computation_id"] == ""]
    assert sorted(p["threshold"] for p in asr) == [0.25, 0.5, 0.75] and all(p["computed_by"] == "producer" and p["frac_at_token_cap"] is None for p in asr)
    assert all(p["extra"]["n_forward_capped_goals"] == 0 for p in asr)
    deg = {p["measure"]: p for p in recs["run_degeneration"] if p["run_id"] == t0["run_id"]}
    assert set(deg) == set(IC.DECLARED_MEASURES)
    # the finish-reason count comes from completion.json's per-(cell, family) block over the response rows; re-tokenized is elsewhere
    fin = deg[IC.AT_CAP_FINISH_REASON]
    assert fin["measurement_state"] == "measured" and fin["numerator"] == 0 and fin["denominator"] == 15 and fin["denominator_unit"] == "replies"
    assert fin["definition_version"].startswith("completion.json:cells[cell][family].n_at_cap / n_responses") and fin["extra"]["escalation"]["cap_to"] == 8192
    none_t0 = next(p for p in per_trial if p["producer_run_key_legacy"] == os.path.basename(paths["t0"]) and p["producer_object_key_legacy"] == "none")
    fin_none = next(p for p in recs["run_degeneration"] if p["run_id"] == none_t0["run_id"] and p["measure"] == IC.AT_CAP_FINISH_REASON)
    assert fin_none["numerator"] == 1 and fin_none["rate"] == pytest.approx(1 / 15)
    assert deg[IC.AT_CAP_RETOKENIZED]["measurement_state"] == "not_measured" and "static_degeneracy_union.json" in deg[IC.AT_CAP_RETOKENIZED]["notes"]
    # the pooled reading equals the relocated reader on the same aggregates
    comp = SS.relocated_computation_id()
    stats = SS.relocated_sampled(ad.aggregate_of, tags=TAGS)
    pool_lora = next(p for p in pooled if p["producer_object_key_legacy"] == F.LORA_CELL)
    assert pool_lora["producer_run_key_legacy"].endswith("sdfbase_full22_synthmodel_sampled_t*_cap8192")
    assert set(pool_lora["extra"]["member_run_ids"]) == {p["run_id"] for p in per_trial if p["producer_object_key_legacy"] == F.LORA_CELL}
    pm = {p["threshold"]: p for p in recs["run_metric"] if p["run_id"] == pool_lora["run_id"] and p["computation_id"] == comp}
    assert set(pm) == {0.5, 0.75}
    assert pm[0.5]["value"] == pytest.approx(stats[("synthmodel", F.LORA_CELL, "pasttense", 0.5)]["value"]) and pm[0.5]["ci_method"] == "wilson"
    # the reader's at_cap_share (per_goal.capped, the forward-cap flag) is quoted in extra, never as an at-cap fraction
    assert pm[0.5]["frac_at_token_cap"] is None and pm[0.5]["extra"]["reader_at_cap_share"] == pytest.approx(stats[("synthmodel", F.LORA_CELL, "pasttense", 0.5)]["at_cap_share"])
    pc = [p for p in recs["run_contrast"] if p["run_id"] == pool_lora["run_id"]]
    assert len(pc) == 2 and all(p["delta_method"] == "paired_bootstrap" and p["effect_kind"] == "cohens_h" and p["n_pairs"] == 5 for p in pc)
    pool_none = next(p for p in pooled if p["producer_object_key_legacy"] == "none")
    assert pc[0]["anchor_run_id"] == pool_none["run_id"]
    # emission 3: the same tag's pooled `none` is the OWN anchor, current on the primed frame -> the producer's contrast plus the selection block, keyed on
    # the pooled judgment the frame holds; the pooled steered metric rows state delta_block: null, the pooled `none` rows carry no block (no comparison)
    hj = F.judge_ids()["strongreject_v1_haiku45"]
    for p in pc:
        sel = p["extra"]["anchor_selection"]
        assert sel["state"] == "own_current" and sel["own_reason"] == "current" and sel["own_anchor_run_id"] == pool_none["run_id"] and sel["rule"] == AC.RULE
        assert sel["key"] == {"token_cap_tokens": 8192, "trial_index": None, "pass_label": "pooled_t0_t1", "decode_policy_id": None} and sel["reading"]["judgment_id"] == I.judgment_id(pool_none["run_id"], hj)
        assert p["anchor_judgment_id"] == I.judgment_id(pool_none["run_id"], hj) and p["computed_by"] == "producer"
    assert all(m["extra"]["delta_block"] is None for m in pm.values())
    none_metrics = [p for p in recs["run_metric"] if p["run_id"] == pool_none["run_id"] and p["computation_id"] == comp]
    assert len(none_metrics) == 2 and all("delta_block" not in m["extra"] for m in none_metrics)
    gd = next(g for g in groups if g.endswith("_t0_cap8192"))
    facts = ad.status_extra(gd)
    assert facts["pooled_contrasts"] == 2 and facts["pooled_contrasts_parked"] == 0 and facts["pooled_anchor_states"] == {"own_current": 2} and facts["row_refusals"] == []
    assert set(facts["pooled_comparisons"]) == {f"synthmodel|{F.LORA_CELL}|pasttense|asr@0.5", f"synthmodel|{F.LORA_CELL}|pasttense|asr@0.75"}
    assert ad.status_extra(next(g for g in groups if g.endswith("_t1_cap8192"))) == {}, "a group without a pooled comparison reports no contrast facts"
    # relocated reader: the two-trial pooling over goals and determinism
    st = stats[("synthmodel", F.LORA_CELL, "pasttense", 0.5)]
    assert st["n"] == 5 and st["n_draws"] == 10 and st["trials"] == [0, 1]
    again = SS.relocated_sampled(ad.aggregate_of, tags=TAGS)
    assert again[("synthmodel", F.LORA_CELL, "pasttense", 0.5)]["delta_lo"] == st["delta_lo"]
    assert ad.rng_scope == "reader_traversal"


def test_missing_standup_object_parks_the_cell_by_name_and_publishes_the_others(tmp_path):
    """With no standup record naming the fitted cell's served object, the fitted cell is PARKED in this read's ledger (`<tag>/<cell>|cell`,
    the Unresolved reason) and the group's `none` cell publishes -- its run, judgment, metrics and items -- where until now the refusal
    propagated out of read() and the runner held the whole revision on the one cell. The `$.row_refusals` state row of the group carries the
    park (and clears on a later read once the record lands); the status facts name it."""
    root = str(tmp_path / "sampled")
    F.write_sampled_root(root, with_standup=False)
    d = F.dims_map()
    ad = SS.SampledStaticAdapter(root=root, dims=d, campaign="synthcampaign", tags=TAGS)
    inputs = ad.discover()
    g = next(ad.comparison_group(i) for i in inputs if i.uri.endswith("_aggregate.json"))
    gi = [i for i in inputs if ad.comparison_group(i) == g]
    recs = {}
    for b in ad.read(_ctx(), g, gi):
        for r in b.records:
            recs.setdefault(r["record_type"], []).append(r["payload"])
    assert [r["object_id"] for r in recs["run"]] == ["none", "none"] and len(recs["judgment"]) == 2 and recs["run_metric"]   # the per-trial none and the pooled none
    assert [it["key"] for it in ad.refusals.items] == [f"synthmodel/{F.LORA_CELL}|cell", f"synthmodel/{F.LORA_CELL}|pooled"]   # the cell and its pooled reading, once each
    assert all(it["why"].startswith("parked: ") and "no standup record on this machine names this cell's served object sha256" in it["why"] for it in ad.refusals.items)
    assert ad.status_extra(g)["row_refusals"] == [f"{it['key']} -- {it['why']}" for it in ad.refusals.items]
    rows = ad.source_states(_ctx(), gi)
    ledger = [r for r in rows if r.json_path == LC.ROW_REFUSALS_PATH]
    assert len(ledger) == 1 and ledger[0].current_members == ad.status_extra(g)["row_refusals"]
    assert SS.SampledStaticAdapter.EMISSION_VERSION == 4


def test_standup_conflict_refuses(tmp_path):
    root = str(tmp_path / "sampled")
    paths = F.write_sampled_root(root)
    import json
    other = os.path.join(root, "serving_records", "static_base_t1_synthmodel_22222", "static")
    os.makedirs(other)
    json.dump({"bank_objects": {F.LORA_CELL: {"object_sha256": "9" * 64}}}, open(os.path.join(other, "cells_served.json"), "w"))
    ad = SS.SampledStaticAdapter(root=root, dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    inputs = ad.discover()
    g = paths["t0"]
    with pytest.raises(AB.AdapterRefusal, match="disagree"):
        list(ad.read(_ctx(), g, [i for i in inputs if ad.comparison_group(i) == g]))


def _pass(root, ad, now):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = RUN.run_pass(root, [ad], conn=conn, now=now)
    finally:
        conn.close()
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    return rep


def test_end_to_end_pass_and_ingest(tmp_path):
    """Two passes through the runner and the harvester (emission 3; no provisional anchors): the first files every run and PARKS both
    pooled comparisons (the pooled `none` is emitted by that very read, the frame does not know it: `not_in_frame`, named in `row_refusals`), the
    fold makes it current and the anchor decision input moves, the second re-reads (`reader_traversal`: every group) under new revision ids and
    files the two contrasts `own_current`; a third pass is quiet."""
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    sroot = str(tmp_path / "sampled")
    paths = F.write_sampled_root(sroot)
    ad = SS.SampledStaticAdapter(root=sroot, campaign="synthcampaign", tags=TAGS)
    rep1 = _pass(root, ad, "2026-01-04T00:00:00.000Z")
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run WHERE pass_label = 'pooled_t0_t1'").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    parks = conn.execute("SELECT json_extract(rm.extra_json, '$.delta_block.anchor.own_reason'), json_extract(rm.extra_json, '$.delta_block.delta') IS NOT NULL FROM run_metric rm "
                         "JOIN run r ON r.run_id = rm.run_id WHERE r.pass_label = 'pooled_t0_t1' AND r.object_id <> 'none'").fetchall()
    assert sorted(tuple(r) for r in parks) == [("not_in_frame", 1), ("not_in_frame", 1)], parks           # the producer's delta parks with the comparison
    # every grades dir read carries its `$.row_refusals` row: the t0 dir's names the two parks, the t1 dir (no pooled comparison) reads []
    refusals = {r[0]: r[1] for r in conn.execute("SELECT source_id, current_members_json FROM source_state WHERE json_path = '$.row_refusals'").fetchall()}
    assert set(refusals) == {f"{ad.name}:{paths['t0']}", f"{ad.name}:{paths['t1']}"} and refusals[f"{ad.name}:{paths['t1']}"] == "[]", refusals
    t0 = refusals[f"{ad.name}:{paths['t0']}"]
    assert "contrast has no current anchor" in t0 and "not_in_frame" in t0 and len(json.loads(t0)) == 2, t0
    conn.close()
    rep = _pass(root, ad, NOW)
    assert rep.anchor_decisions_changed >= 1 and rep.groups_refreshed >= 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run WHERE campaign = 'synthcampaign'").fetchone()[0] == 6
    assert conn.execute("SELECT COUNT(*) FROM run WHERE pass_label = 'pooled_t0_t1'").fetchone()[0] == 2
    # (the second pass re-read the t0 group under a new revision id, so its per-item rows exist under two revisions; the CURRENT view holds them once)
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current WHERE metric_id = 'strongreject_best'").fetchone()[0] == 20
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 2
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.anchor_selection.state'), anchor_kind, computed_by FROM run_contrast")] == [("own_current", "same_standup_none", "producer")]
    assert conn.execute("SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.pass_label = 'pooled_t0_t1' AND r.object_id <> 'none' "
                        "AND json_type(rm.extra_json, '$.delta_block') = 'null'").fetchone()[0] == 2, "a filed contrast states delta_block: null"
    # the second pass RE-WROTE the grades-dir group's `$.row_refusals` row EMPTY: `source_state` is keyed by (source_id, input_uri,
    # json_path) and says what the source names NOW, so the park that cleared is seen to clear -- the same key, `[]`, observed by this pass -- instead of the
    # first pass's list standing as the last value written
    # (the t1 dir's content identity did not move -- no comparison, no decision input -- so the runner re-stamped its watermarks alone and its [] row keeps
    # the first pass's observation; the t0 row is the one re-written)
    rows2 = conn.execute("SELECT source_id, current_members_json, observed_utc FROM source_state WHERE json_path = '$.row_refusals'").fetchall()
    assert sorted(tuple(r) for r in rows2) == sorted([(f"{ad.name}:{paths['t0']}", "[]", NOW), (f"{ad.name}:{paths['t1']}", "[]", "2026-01-04T00:00:00.000Z")]), rows2
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM artifact WHERE kind = 'judge_items'").fetchone()[0] == 2
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT priority FROM source_revision WHERE adapter = ? AND accepted = 1", (ad.name,)).fetchall()] == [(IC.PRIORITY_OWNING_READER,)]   # the owner of the export twins' duplicate identities since migration 0026 (ownership.TWIN_OWNERSHIP)
    assert conn.execute("SELECT COUNT(*) FROM v_item_conflict").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM item_flags WHERE at_token_cap IS NOT NULL").fetchone()[0] == 0
    assert conn.execute(f"SELECT COUNT(*) FROM v_run_degeneration_current WHERE measure = '{IC.AT_CAP_FINISH_REASON}' AND measurement_state = 'measured'").fetchone()[0] == 4
    conn.close()
    rep3 = _pass(root, ad, "2026-01-06T00:00:00.000Z")
    assert rep3.anchor_decisions_changed == 0 and rep3.groups_refreshed == 0


def test_without_a_registry_connection_a_pooled_contrast_read_refuses_the_group_and_the_smoke_names_it(tmp_path):
    """The pooled steered row intends a contrast, so its group asks the pass's anchor currency: with no connection the GROUP refuses by name (never a
    contrast-less revision that would withdraw the contrasts); the smoke counts it per group; a group with no pooled comparison (the t1 dir) reads."""
    root = str(tmp_path / "sampled")
    paths = F.write_sampled_root(root)
    ad = SS.SampledStaticAdapter(root=root, dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    inputs = ad.discover()
    g0, g1 = paths["t0"], paths["t1"]
    with pytest.raises(AC.AnchorCurrencyUnavailable, match="no registry connection"):
        list(ad.read(_ctx(), g0, [i for i in inputs if ad.comparison_group(i) == g0]))
    recs = [r for b in ad.read(_ctx(), g1, [i for i in inputs if ad.comparison_group(i) == g1]) for r in b.records]
    assert {r["record_type"] for r in recs} >= {"run", "run_metric"} and not any(r["record_type"] == "run_contrast" for r in recs)
    out = DC.smoke(ad)
    assert out["anchor_unavailable_groups"] == 1 and any(x.startswith(f"{g0}: anchor currency unavailable") for x in out["refused_groups"]), out["refused_groups"]


def test_a_tag_without_a_none_aggregate_parks_the_pooled_comparison_by_name_instead_of_skipping_in_silence(tmp_path):
    """The pin's traversal forms a delta only for a tag whose aggregates carry a `none` entry; a tag serving the steered object alone used to skip
    the contrast in silence. Now the comparison is MADE: the composed same-tag `none` is the own anchor (`not_in_frame`: never pooled), nothing
    else stands at the exact key, the pooled steered rows park on `extra.delta_block` (no producer delta to park: the block says so), the park is a
    `row_refusals` line and a `source_state` row, and the facts tally it."""
    root = str(tmp_path / "sampled")
    paths = F.write_sampled_root(root, cells=(F.LORA_CELL,))
    ad = SS.SampledStaticAdapter(root=root, dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    recs, groups = _records(ad)
    assert "run_contrast" not in recs and len([p for p in recs["run"] if p["pass_label"] == "pooled_t0_t1"]) == 1
    comp = SS.relocated_computation_id()
    pooled = next(p for p in recs["run"] if p["pass_label"] == "pooled_t0_t1")
    pm = {p["threshold"]: p for p in recs["run_metric"] if p["run_id"] == pooled["run_id"] and p["computation_id"] == comp}
    assert set(pm) == {0.5, 0.75}
    for m in pm.values():
        blk = m["extra"]["delta_block"]
        assert blk["anchor"]["state"] == "no_current_anchor" and blk["anchor"]["own_reason"] == "not_in_frame" and blk["anchor"]["own_anchor_kind"] == "same_standup_none"
        assert blk["delta"] is None and blk["n_pairs"] == 0 and blk["delta_method"].startswith("none: the pin's traversal formed no paired bootstrap") and blk["rule"] == AC.RULE
        assert blk["anchor_selection"]["state"] == "no_current_anchor" and blk["anchor_selection"]["n_candidates"] == 0 and blk["producer_block"] == SS.POOLED_CONTRAST_RULE
        assert "no unsteered run" in blk["why_not_a_contrast_row"] or "not_in_frame" in blk["why_not_a_contrast_row"] or "folded" in blk["why_not_a_contrast_row"], blk["why_not_a_contrast_row"]
    gd = paths["t0"]
    facts = ad.status_extra(gd)
    assert facts["pooled_contrasts"] == 0 and facts["pooled_contrasts_parked"] == 2 and facts["pooled_anchor_states"] == {"no_current_anchor": 2}
    assert len(facts["row_refusals"]) == 2 and all("contrast has no current anchor" in x and "not_in_frame" in x for x in facts["row_refusals"]), facts["row_refusals"]
    # the source-state rows follow the group's read (the runner collects them right after it): the parks ride `$.row_refusals` of the grades-dir group
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == gd]
    ctx = _primed(ad)
    list(ad.read(ctx, gd, inputs))
    states = ad.source_states(ctx, inputs)
    rows = [st for st in states if st.json_path == "$.row_refusals"]
    assert len(rows) == 1 and len(rows[0].current_members) == 2 and rows[0].source_id == f"{ad.name}:{gd}" and all("contrast has no current anchor" in m for m in rows[0].current_members)


def test_a_re_picked_anchor_refuses_by_name_and_parks_the_producers_delta(tmp_path):
    """The same tag's pooled `none` retired on the frame while another tag's pooled `none` of this source stands current at the exact key: the
    currency re-picks it (`sibling_same_source`) and the family rule REFUSES by name -- the pooled contrast is the pin's paired bootstrap over this
    tag's goals and k is NULL on both sides -- so the producer's delta parks on the metric row with the refusal, and no registry-derived contrast
    stands in."""
    root = str(tmp_path / "sampled")
    paths = F.write_sampled_root(root)
    ad = SS.SampledStaticAdapter(root=root, dims=F.dims_map(), campaign="synthcampaign", tags=TAGS)
    frame = AF.primed_frame(ad, now=NOW, producer=PRODUCER, root="/nonexistent")
    recs0, groups = _records(ad, _primed(ad, frame))
    pooled_none = next(p for p in recs0["run"] if p["pass_label"] == "pooled_t0_t1" and p["producer_object_key_legacy"] == "none")
    pooled_lora = next(p for p in recs0["run"] if p["pass_label"] == "pooled_t0_t1" and p["producer_object_key_legacy"] == F.LORA_CELL)
    assert len(recs0["run_contrast"]) == 2
    AF.retire(frame, pooled_none["run_id"], "selection_state excluded (decisive ruling exclude r_synth_exclude_pooled_none)")
    hj = F.judge_ids()["strongreject_v1_haiku45"]
    comp = SS.relocated_computation_id()
    sib = frame.run("synthcampaign:pasttense:sibling-pooled-none", model=pooled_lora["model_id"], eval_id="pasttense", inst=pooled_lora["instrument_id"], cap=8192, trial=None,
                    pass_label="pooled_t0_t1", source="sampled_lane", standup="othertag", key="othertag/pooled", in_paper=0, roster=False, reading=False)
    sjid = frame.judgment(sib, judge=hj)
    for thr in (0.5, 0.75):
        frame.metric(sib, sjid, metric="asr", thr=thr, computation=comp, k=None, n=5, value=0.4)
    recs, _ = _records(ad, _primed(ad, frame))
    assert "run_contrast" not in recs
    pm = {p["threshold"]: p for p in recs["run_metric"] if p["run_id"] == pooled_lora["run_id"] and p["computation_id"] == comp}
    for thr, m in pm.items():
        blk = m["extra"]["delta_block"]
        assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["own_reason"] == "excluded" and blk["anchor"]["anchor_run_id"] == sib and blk["anchor"]["anchor_kind"] == "sibling_none"
        assert blk["delta"] is not None and blk["n_pairs"] == 5 and blk["delta_method"] == "paired_bootstrap" and blk["computed_by"] == "producer"
        assert "refused by name" in blk["why_not_a_contrast_row"] and "k is NULL on both sides" in blk["why_not_a_contrast_row"] and blk["anchor_selection"]["candidates"] == [sib]
    facts = ad.status_extra(paths["t0"])
    assert facts["pooled_anchor_states"] == {"sibling_same_source": 2} and facts["pooled_contrasts_parked"] == 2 and facts["pooled_contrasts"] == 0
    assert all("refused by name" in x for x in facts["row_refusals"]) and len(facts["row_refusals"]) == 2


def test_fitted_runs_carry_the_served_key_dose_and_the_operating_point(tmp_path):
    """Every fitted run of this reader had `strength_nominal` / `is_operating_point` NULL, so a roster entry's `operating_point` rule could
    never admit it (2026-09-20). A fitted object's per-trial and pooled runs now file the dose the standup tag spells — a bare tag the
    campaign nominal 1.0 (the operating point), `_d0p5` 0.5 (not the operating point) — as a `derived` nominal in
    `nominal_only`; the unsteered cell files no strength. The served object's dose sidecar (the object row's base_scale) is the second
    authority: agreement is recorded, a disagreement refuses the grades directory by name rather than filing either dose."""
    # one root per standup tag: the fixture serves ONE synthetic object, and the same object cannot be both the shipped fit (a bare tag)
    # and a 0.5 copy (`_d0p5`) once the sidecar is consulted below
    tags = {"synthmodel": ("synthmodel", "base→instruct", "synthmodel"), "synthmodel_d0p5": ("synthmodel", "base→instruct", "synthmodel_d0p5")}
    roots = {tag: str(tmp_path / f"sampled_{tag}") for tag in tags}
    recs_by_view = {}
    for tag, root in roots.items():
        F.write_sampled_root(root, tag=tag)
        ad = SS.SampledStaticAdapter(root=root, dims=F.dims_map(), campaign="synthcampaign", tags={tag: tags[tag]})
        recs_by_view[tag], _ = _records(ad)
    for view, nominal, is_op in (("synthmodel", 1.0, 1), ("synthmodel_d0p5", 0.5, 0)):
        recs = recs_by_view[view]
        lora = [p for p in recs["run"] if p["view_legacy"] == view and p["producer_object_key_legacy"] == F.LORA_CELL]
        assert {p["pass_label"] for p in lora} == {"ladder_rung", "pooled_t0_t1"} and len(lora) == 3
        for p in lora:
            assert (p["strength_nominal"], p["strength_nominal_source"], p["strength_unit"], p["is_operating_point"]) == (nominal, "derived", "nominal_only", is_op), p["pass_label"]
            fc = p["field_confidence_json"]
            assert fc["strength_nominal"]["confidence"] == "derived" and "standup tag" in fc["strength_nominal"]["source"]
            assert fc["is_operating_point"]["confidence"] == "derived" and fc["strength_unit"]["confidence"] == "unverified"
            assert p["extra"]["dose_served_key"] == nominal and p["extra"]["dose_sidecar_derivation"] is None      # no object row here: not consulted
        none = [p for p in recs["run"] if p["view_legacy"] == view and p["producer_object_key_legacy"] == "none"]
        assert len(none) == 3 and all("strength_nominal" not in p and "is_operating_point" not in p and "strength_unit" not in p for p in none)
    # the sidecar agrees (a plain fit at the campaign nominal under the bare tag; a 0.5 dose-scaled copy under `_d0p5`): recorded on the row
    dims = F.dims_map()
    dims.object_scales[F.OBJ_SHA] = (None, "none")
    ad = SS.SampledStaticAdapter(root=roots["synthmodel"], dims=dims, campaign="synthcampaign", tags={"synthmodel": tags["synthmodel"]})
    recs, _ = _records(ad)
    lora = [p for p in recs["run"] if p["producer_object_key_legacy"] == F.LORA_CELL]
    assert lora and all(p["strength_nominal"] == 1.0 and "agrees with" in p["field_confidence_json"]["strength_nominal"]["source"] and p["extra"]["dose_sidecar_derivation"] == "none" for p in lora)
    dims.object_scales[F.OBJ_SHA] = (0.5, "dose_scale")
    ad = SS.SampledStaticAdapter(root=roots["synthmodel_d0p5"], dims=dims, campaign="synthcampaign", tags={"synthmodel_d0p5": tags["synthmodel_d0p5"]})
    recs, _ = _records(ad)
    lora = [p for p in recs["run"] if p["producer_object_key_legacy"] == F.LORA_CELL]
    assert lora and all(p["strength_nominal"] == 0.5 and p["is_operating_point"] == 0 and p["extra"]["dose_sidecar_base_scale"] == 0.5 for p in lora)
    # the sidecar disagrees (the served object is a 0.5 copy but the bare tag spells 1.0): the grades directory is refused by name
    ad = SS.SampledStaticAdapter(root=roots["synthmodel"], dims=dims, campaign="synthcampaign", tags={"synthmodel": tags["synthmodel"]})
    with pytest.raises(AB.AdapterRefusal, match="dose sidecar"):
        _records(ad)


def test_seeded_dose_family_tag_files_its_middle_dose_against_the_sidecar(tmp_path):
    """The seeded difference-object standups spell the dose in the MIDDLE of the tag (`qwen35_4b_d1p25_loradiff64s1`); the trailing grammar
    read them as the bare 1.0, so the sidecar (a 1.25 dose-scaled copy) disagreed and all eleven seeded groups were refused by name
    (window F2b, 2026-09-24). The seeded spelling now reads its token: with the sidecar at 1.25 the fitted runs file `strength_nominal`
    1.25, `derived`, `nominal_only`, not the operating point; a REAL disagreement (a `_d0p75_` tag over a 1.25 copy) is still refused."""
    tags = {"synthmodel_d1p25_loradiff64s1": ("synthmodel", "base→instruct", "synthmodel_d1p25_loradiff64s1"),
            "synthmodel_d0p75_loradiff64s1": ("synthmodel", "base→instruct", "synthmodel_d0p75_loradiff64s1")}
    roots = {tag: str(tmp_path / f"sampled_{tag}") for tag in tags}
    for tag, root in roots.items():
        F.write_sampled_root(root, tag=tag)
    dims = F.dims_map()
    dims.object_scales[F.OBJ_SHA] = (1.25, "dose_scale")
    tag = "synthmodel_d1p25_loradiff64s1"
    ad = SS.SampledStaticAdapter(root=roots[tag], dims=dims, campaign="synthcampaign", tags={tag: tags[tag]})
    recs, _ = _records(ad)
    lora = [p for p in recs["run"] if p["view_legacy"] == tag and p["producer_object_key_legacy"] == F.LORA_CELL]
    assert {p["pass_label"] for p in lora} == {"ladder_rung", "pooled_t0_t1"} and len(lora) == 3
    for p in lora:
        assert (p["strength_nominal"], p["strength_nominal_source"], p["strength_unit"], p["is_operating_point"]) == (1.25, "derived", "nominal_only", 0), p["pass_label"]
        assert p["extra"]["dose_served_key"] == 1.25 and p["extra"]["dose_sidecar_base_scale"] == 1.25 and p["extra"]["dose_sidecar_derivation"] == "dose_scale"
        assert "agrees with" in p["field_confidence_json"]["strength_nominal"]["source"]
    none = [p for p in recs["run"] if p["view_legacy"] == tag and p["producer_object_key_legacy"] == "none"]
    assert len(none) == 3 and all("strength_nominal" not in p for p in none)
    # the key spells 0.75 but the served object is the 1.25 copy: two authorities disagree, the grades directory is refused by name
    tag = "synthmodel_d0p75_loradiff64s1"
    ad = SS.SampledStaticAdapter(root=roots[tag], dims=dims, campaign="synthcampaign", tags={tag: tags[tag]})
    with pytest.raises(AB.AdapterRefusal, match=r"spells dose 0\.75 but the served object .* says 1\.25"):
        _records(ad)


def test_control_family_tag_at_minus_one_files_its_middle_dose_against_the_sidecar(tmp_path):
    """The control-negative battery (2026-09-25): the static grades of `qwen35_4b_dm1_ctllora64s<k>` / `_ctlrelu64s<k>` land in this
    reader's layout under those tags, under the mirror root read by `CtlNeg1StaticAdapter` (source `ctl_neg1`). The family alternation has one home (`derive.SEEDED_DOSE_FAMILIES`), so the tag strips to the model key
    and its middle token reads -1.0; the served object is a -1 dose-scaled copy, so the sidecar agrees and the fitted runs file
    `strength_nominal` -1.0 (`derived`, `nominal_only`, not the operating point). Before the one home, the trailing grammar read the tag as
    the bare 1.0 and the sidecar cross-check would have refused every control static by name -- the seeded difference statics' 2026-09-24
    failure again."""
    for fam_seg, cell in (("ctllora64s2", F.LORA_CELL), ("ctlrelu64s3", F.LORA_CELL)):
        tag = f"synthmodel_dm1_{fam_seg}"
        assert SS.tag_model_key(tag) == ("synthmodel", tag) and SS.tag_model_key(f"{tag}_t0") == ("synthmodel", f"{tag}_t0")
        root = str(tmp_path / f"sampled_{tag}")
        F.write_sampled_root(root, tag=tag, cells=(cell,))          # the steered object only: Scale serves no `none` (the seed standup's is the anchor)
        dims = F.dims_map()
        dims.object_scales[F.OBJ_SHA] = (-1.0, "dose_scale")
        ad = SS.CtlNeg1StaticAdapter(root=root, dims=dims, campaign="synthcampaign", tags={tag: ("synthmodel", "base→instruct", tag)})
        assert (ad.name, ad.source, ad.owner, ad.covers_sources) == ("ctl_neg1_static", "ctl_neg1", "Scale", ("ctl_neg1",))
        assert SS.CtlNeg1StaticAdapter().root == SS.CTL_NEG1_ROOT and SS.SampledStaticAdapter().root == SS.SAMPLED_ROOT and SS.SampledStaticAdapter.source == "sampled_lane"
        recs, _ = _records(ad, _ctx())          # no pooled comparison on this reader (POOLED_CONTRASTS False): no frame is asked
        assert ad.POOLED_CONTRASTS is False and SS.SampledStaticAdapter.POOLED_CONTRASTS is True
        steered = [p for p in recs["run"] if p["view_legacy"] == tag and p["producer_object_key_legacy"] == cell]
        assert steered and {p["pass_label"] for p in steered} == {"ladder_rung", "pooled_t0_t1"}, [(p["pass_label"], p["view_legacy"]) for p in recs["run"]]
        seed = fam_seg[-1]
        for p in steered:
            assert (p["strength_nominal"], p["strength_nominal_source"], p["strength_unit"], p["is_operating_point"]) == (-1.0, "derived", "nominal_only", 0), p["pass_label"]
            assert p["extra"]["dose_served_key"] == -1.0 and p["extra"]["dose_sidecar_base_scale"] == -1.0 and p["extra"]["dose_sidecar_derivation"] == "dose_scale"
            assert "agrees with" in p["field_confidence_json"]["strength_nominal"]["source"]
            # the battery's rows: its own source, and the declared seed-standup anchor on every steered row (the standup serves no `none`)
            assert p["lane_source_legacy"] == "ctl_neg1"
            assert (p["extra"]["anchor_standup_legacy"], p["extra"]["anchor_kind_declared"]) == (f"synthmodel_s{seed}", "sibling_none")
            assert "standing anchor ruling" in p["extra"]["anchor_rule"] and tag in p["extra"]["anchor_source"]
        assert not [p for p in recs["run"] if p["producer_object_key_legacy"] == "none"]
        # the paper's sampled reader over the same root: `sampled_lane`, no anchor stamp -- its bytes are its own
        base = SS.SampledStaticAdapter(root=root, dims=dims, campaign="synthcampaign", tags={tag: ("synthmodel", "base→instruct", tag)})
        brecs, _ = _records(base)          # the sampled reader makes the comparison: on the primed frame no `none` stands at the key, the pooled row parks by name
        for p in brecs["run"]:
            assert p["lane_source_legacy"] == "sampled_lane" and "anchor_standup_legacy" not in p["extra"] and "anchor_kind_declared" not in p["extra"]
        # an empty mirror root (the grades not yet mirrored) discovers nothing and refuses nothing
        assert SS.CtlNeg1StaticAdapter(root=str(tmp_path / "empty-mirror-root"), dims=dims, campaign="synthcampaign").discover() == []
        # a real disagreement is still refused: the same tag over a +1 copy
        dims.object_scales[F.OBJ_SHA] = (1.0, "dose_scale")
        ad = SS.CtlNeg1StaticAdapter(root=root, dims=dims, campaign="synthcampaign", tags={tag: ("synthmodel", "base→instruct", tag)})
        with pytest.raises(AB.AdapterRefusal, match=r"spells dose -1 but the served object .* says 1"):
            _records(ad, _ctx())


def test_load_standups_reads_modal_and_slurm_named_records(tmp_path):
    """A Modal-served standup is named ``static_base_t<T>_<tag>_modal-<app id>``; a Slurm-served one ends in the job number.
    Both carry ``cells_served.json:bank_objects.<id>.object_sha256`` the same way, so both must load (the live run's LoRA
    objects at the Modal-served doses were refused as unnamed before the name pattern admitted the Modal spelling)."""
    import json
    root = str(tmp_path / "sampled")
    F.write_sampled_root(root)                       # the Slurm-named record: static_base_t0_synthmodel_11111
    modal = os.path.join(root, "serving_records", "static_base_t1_synthmodel_d1_modal-01m2f1vemdb70wdc", "static")
    os.makedirs(modal)
    modal_sha = "a" * 64
    json.dump({"bank_objects": {F.LORA_CELL: {"object_sha256": modal_sha}}}, open(os.path.join(modal, "cells_served.json"), "w"))
    json.dump({"model": F.HF_IT, "model_key": "synthmodel", "model_revision": "b" * 40, "attested_max_num_seqs": 8, "attested_max_model_len": 4096,
               "reduction_mode": "stacked", "cells": ["none", F.LORA_CELL], "written_at": "2026-01-01T00:02:00+00:00"}, open(os.path.join(modal, "endpoint.json"), "w"))
    st = SS.load_standups(root)
    assert st.objects[("synthmodel", F.LORA_CELL)] == F.OBJ_SHA
    assert st.objects[("synthmodel_d1", F.LORA_CELL)] == modal_sha
    assert not st.conflicts
    assert st.endpoints["synthmodel_d1"]["model"] == F.HF_IT
    assert sum(1 for f in st.files if "_modal-" in os.path.basename(os.path.dirname(os.path.dirname(f)))) == 2   # the record dir, not the tmp path
    # the name pattern on the live spellings: the tag keeps its underscores, the job part is digits or a Modal app id
    m = SS._STANDUP_RE.match("static_base_t0_qwen35_4b_it_d1_modal-01m2f1vemdb70wdc")
    assert (m.group("trial"), m.group("tag"), m.group("job")) == ("0", "qwen35_4b_it_d1", "modal-01m2f1vemdb70wdc")
    m = SS._STANDUP_RE.match("static_base_t1_llama8b_it_d1p5_100037")
    assert (m.group("trial"), m.group("tag"), m.group("job")) == ("1", "llama8b_it_d1p5", "100037")
    assert SS._STANDUP_RE.match("static_esc_t0_qwen35_4b_it_d1_modal-01m2f84t01vqp5f6") is None       # an escalation record, not a base standup
    assert SS._STANDUP_RE.match("static_base_t0_qwen35_4b_it_d1_modal-01M2F1VEMDB70WDC") is None      # Modal app ids are lowercase


def test_seed_replicate_tag_resolves_its_served_model_through_the_legacy_map(tmp_path):
    """A seed-replicate standup tag (`qwen35_4b_s2`; grades dir `sdfbase_full22_qwen35_4b_s2_sampled_t0`) carries no dose / instruct
    suffix for `tag_model_key` to strip, so its model key IS the tag; with no standup record on this machine the served model comes
    from the seeds' legacy map (legacy_spellings.yaml model_keys, 2026-09-20) as `derived` — never the hardcoded seed-less fallback."""
    from sisteer.registry.adapters import dimensions as D
    assert SS.tag_model_key("qwen35_4b_s2") == ("qwen35_4b_s2", "qwen35_4b_s2")
    assert SS.parse_grades_dir("sdfbase_full22_qwen35_4b_s2_sampled_t0") == ("qwen35_4b_s2", 0, "")
    seeds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry")
    dims = IC.Dims(models=IC.load_legacy_model_map(D.adapter_legacy_spellings(seeds_dir)))
    ad = SS.SampledStaticAdapter(root=str(tmp_path), dims=dims, campaign="synthcampaign")
    for s in (1, 2, 3):
        conf = IC.Confidence()
        assert ad._served_model(dims, f"qwen35_4b_s{s}", conf) == ("qwen35_4b_it", None)
        assert conf.entries["model_id"]["confidence"] == "derived" and "explicit map" in conf.entries["model_id"]["source"]
    with pytest.raises(IC.Unresolved):
        ad._served_model(dims, "qwen35_4b_s4", IC.Confidence())


def test_load_standups_reads_the_recorded_endpoint_and_the_seeded_tag_resolves(tmp_path):
    """The native serving-record writer names the endpoint identity ``endpoint.recorded.json`` (the seeded difference-LoRA
    standups of 2026-09-23 carried no ``endpoint.json``, and the 2026-09-24 pass refused all eleven groups as "no standup record
    names the served model and the tag matches no known model key"). The recorded file speaks when the classic name is absent, the
    row remembers which file spoke, and the seeded dose-family tag strips to its model key."""
    import json
    root = str(tmp_path / "sampled")
    F.write_sampled_root(root)
    rec = os.path.join(root, "serving_records", "static_base_t0_synthmodel_d1p25_loradiff64s1_128747", "static")
    os.makedirs(rec)
    json.dump({"bank_objects": {F.LORA_CELL: {"object_sha256": "c" * 64}}}, open(os.path.join(rec, "cells_served.json"), "w"))
    json.dump({"model": F.HF_IT, "model_key": "synthmodel", "model_revision": "d" * 40, "reduction_mode": "not_applicable",
               "serving_path": "lora_native", "engine": "lora_native", "attested_max_num_seqs": 8, "attested_max_model_len": 4096,
               "written_at": "2026-09-23T20:00:00+00:00"}, open(os.path.join(rec, "endpoint.recorded.json"), "w"))
    st = SS.load_standups(root)
    ep = st.endpoints["synthmodel_d1p25_loradiff64s1"]
    assert ep["model"] == F.HF_IT and ep["model_revision"] == "d" * 40 and ep["endpoint_record_file"] == "endpoint.recorded.json"
    assert ep["serving_path"] == "lora_native"
    assert st.objects[("synthmodel_d1p25_loradiff64s1", F.LORA_CELL)] == "c" * 64
    assert SS.tag_model_key("qwen35_4b_d1p25_loradiff64s1") == ("qwen35_4b", "qwen35_4b_d1p25_loradiff64s1")
    assert SS.tag_model_key("qwen35_4b_dm0p5_loradiff64s3_t0") == ("qwen35_4b", "qwen35_4b_dm0p5_loradiff64s3_t0")
    assert SS.tag_model_key("qwen35_4b_it_d0p5") == ("qwen35_4b", "qwen35_4b_it_d0p5")


def test_a_lifted_hold_out_reactivates_the_filed_revision_with_its_own_empty_ledger_never_the_parked_reads(tmp_path):
    """For the sampled reader: pass 1 parks, pass 2 files (the t0 group's revision rev2: `$.row_refusals`
    `[]`, keyed on the GROUP). A `hold_out` ruling on the pooled `none` retires the own anchor: pass 3 re-reads the traversal, the two pooled
    comparisons park and rev3's ledger names them. The lift (the same ruling id re-filed as `reference`) makes the decision set rev2's again, so
    rev2 is RE-ACTIVATED -- and its ledger must read `[]` at the pass-4 stamp, restored from rev2's own state rows: the same adapter instance read
    rev3 last (every group, the traversal), so before the fix the t0 row either kept rev3's parks or took the adapter's list."""
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    sroot = str(tmp_path / "sampled")
    paths = F.write_sampled_root(sroot)
    ad = SS.SampledStaticAdapter(root=sroot, campaign="synthcampaign", tags=TAGS)
    t0 = paths["t0"]
    sid = f"{ad.name}:{t0}"
    key = AB.row_refusals_uri(sid)

    def ledger():
        with DB.reading(DB.db_path(root)) as conn:
            return [tuple(r) for r in conn.execute("SELECT input_uri, found, current_members_json, observed_utc FROM source_state WHERE source_id = ? AND json_path = ?",
                                                   (sid, AB.ROW_REFUSALS_PATH)).fetchall()]

    def head():
        return RX.head_of(root, sid)[0]

    _pass(root, ad, "2026-01-04T00:00:00.000Z")
    (row1,) = ledger()
    assert row1[0] == key and row1[1] == 1 and len(json.loads(row1[2])) == 2 and "no current anchor" in row1[2], row1
    _pass(root, ad, NOW)
    rev2 = head()
    assert ledger() == [(key, 1, "[]", NOW)] and RX.q(root, "SELECT COUNT(*) FROM run_contrast") == [(2,)]
    with DB.reading(DB.db_path(root)) as conn:
        inputs = [i for i in ad.discover() if ad.comparison_group(i) == t0]
        (sha,) = conn.execute("SELECT input_sha256 FROM source_state WHERE source_id = ? AND json_path = ?", (sid, AB.ROW_REFUSALS_PATH)).fetchone()
        assert sha == AB.inputs_fingerprint(inputs)
        assert conn.execute("SELECT COUNT(*) FROM source_state WHERE json_path = ? AND input_uri NOT LIKE ?", (AB.ROW_REFUSALS_PATH, AB.ROW_REFUSALS_URI_PREFIX + "%")).fetchone()[0] == 0
    ((none_run,),) = RX.q(root, "SELECT run_id FROM run WHERE pass_label = 'pooled_t0_t1' AND object_id = 'none'")
    # the pooled none held out: the own anchor is not current, no other pooled none stands at the key -> both comparisons park -> a new revision
    RX.file_ruling(root, "r-hold-pooled-none", "hold_out", none_run, 2600)
    assert RX.q(root, "SELECT is_current, held_out FROM v_run_current WHERE run_id = ?", none_run) == [(0, 1)]
    rep3 = _pass(root, ad, "2026-01-06T00:00:00.000Z")
    assert rep3.anchor_decisions_changed == 1 and rep3.groups_refreshed >= 1
    rev3 = head()
    (row3,) = ledger()
    assert rev3 != rev2 and row3[0] == key and row3[1] == 1 and row3[3] == "2026-01-06T00:00:00.000Z" and len(json.loads(row3[2])) == 2, row3
    assert all("no current anchor" in m and "held_out" in m for m in json.loads(row3[2])), row3[2]
    assert RX.current_contrast_keys(root, sid) == []
    # the lift: the decision set is rev2's again -> rev2 re-activated, its own (empty) ledger restored at the pass-4 stamp
    RX.file_ruling(root, "r-hold-pooled-none", "reference", none_run, 2900)
    assert RX.q(root, "SELECT is_current, held_out FROM v_run_current WHERE run_id = ?", none_run) == [(1, 0)]
    rep4 = _pass(root, ad, "2026-01-07T00:00:00.000Z")
    (res4,) = [r for r in rep4.results if r.group == t0]
    assert res4.accepted and res4.reactivated and res4.revision_id == rev2 and f"{os.sep}_activations{os.sep}" in res4.files[0], res4
    assert head() == rev2 and ledger() == [(key, 1, "[]", "2026-01-07T00:00:00.000Z")], ledger()
    assert len(RX.current_contrast_keys(root, sid)) == 2
    assert ad.refusals.read_of[1] != "2026-01-07T00:00:00.000Z", "no read of the traversal ran in pass 4 (a re-stamp and a re-activation): the adapter was not asked"
    # quiet after
    rep5 = _pass(root, ad, "2026-01-08T00:00:00.000Z")
    assert rep5.anchor_decisions_changed == 0 and rep5.groups_refreshed == 0 and ledger() == [(key, 1, "[]", "2026-01-07T00:00:00.000Z")]
