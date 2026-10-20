"""The Petri adapter on synthetic stats + aggregate files: one run per behaviour plus the pooled one under the audit's
run key, per-scenario items, the sidecar's contrasts against the standup's none, the two-source guard, and the
itemless-placeholder rule -- and, since EMISSION_VERSION 2, every contrast decided by the
pass's anchor currency: the first pass over a fresh registry parks every comparison (the audited `none` runs are emitted by that very
read and the frame does not know them -- no provisional anchors), the second files them `own_current` with `extra.anchor_selection`
and the frame's anchor judgment, a third is quiet; a retired own `none` re-picks a current unsteered run of the exact key and the
registry files the per-behavior PAIRED difference over both runs' item rows (the filed `mean_of_behavior_means` weighting), refusing
by name while a side has no items or a behavior has no shared scenario; a sidecar anchor block naming a `none` the registry never
folded is a reason not to use that id -- the exact-key search finds the standup's own audit -- never a dependency block; the smoke
names the unavailable currency. Everything here is synthetic (invented keys, TESTHOST, tiny numeric shapes)."""
from __future__ import annotations

import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import petri_stats as PS
from sisteer.registry.adapters import runner as RUN

B = ("cyber_bio_chem_capabilities", "defer_to_users", "delusion_sycophancy", "instruction_evasion", "reasoning_unfaithfulness", "self_preservation")
T = ["2026-10-03T0{}:00:00.000Z".format(i) for i in range(1, 9)]
SOURCE_ID = RUN.source_id_for(PS.NAME, "petri")
PARK_PHRASE = "contrast has no current anchor at key"
#: the per-scenario scores of every behavior of a synthetic audit at `base`: {base, base + 0.2}, so each behavior's mean is base + 0.1 and
#: the pooled estimand (the mean of the behavior means) is base + 0.1 too -- the filed `mean_of_behavior_means` is CONSISTENT with the rows,
#: which is what lets the re-anchoring test check the helper's one-sided mean against the filed estimand
SCENARIO_OFFSETS = (0.0, 0.2)


def _files(tmp_path, agg_key="aaaa000000000001", stats_key="aaaa000000000001", placeholder_key=None, anchor_key=None):
    """The two producers under tmp_path. ``anchor_key`` puts a sidecar ``anchor`` block naming that `petri_run_key` on the tag (the shape
    the live sidecars carry; an unfolded key is the `not_in_frame` case)."""
    rs = tmp_path / "rs"
    cons = tmp_path / "consolidated"
    os.makedirs(rs / "petri_stats", exist_ok=True)
    os.makedirs(cons, exist_ok=True)

    def beh(base):
        return {b: {"mean": base + 0.1, "lo": base - 0.5, "hi": base + 0.5, "n_items": 2, "n_expected": 2, "sd": 0.3,
                    "per_scenario": {f"scenario_{i}_a": base + SCENARIO_OFFSETS[0], f"scenario_{i}_b": base + SCENARIO_OFFSETS[1]}} for i, b in enumerate(B)}

    def cell(key, base, complete=True):
        return {"petri_run_key": key, "intervention_hash": "1111000011110000", "status": "COMPLETE" if complete else "INCOMPLETE", "audited_items": 12 if complete else 4,
                "audited_expected": 12, "completed_at": 1788823897.6, "mean_of_behavior_means": base + 0.1, "serving_identity": "proxy_vllm+lens+interleaved",
                "model_roles": {"model_roles_distinct": [{"auditor": "judge-model-w", "judge": "judge-model-y", "target": "x"}], "n_logs": 3},
                "behaviors": beh(base), "auditor_chain": {"sonnet_only": 12}}
    tag = {
        "audit": f"petri_full22_{F.VIEW}",
        "cells": {"none": cell("aaaa0000000000ff", 6.0), "si_constv2_mix_full__lora_r4_ce": cell(stats_key, 5.0)},
        "vs_bare": {"si_constv2_mix_full__lora_r4_ce": {b: {"delta": -1.0, "lo": -1.5, "hi": -0.5, "sign_test_p": 0.25, "n": 2, "dispersion": {"d_z": -0.8, "d_z_lo": -1.2, "d_z_hi": -0.3}} for b in B}},
        "vs_bare_cell": {"si_constv2_mix_full__lora_r4_ce": {"delta": -1.0, "delta_lo": -1.4, "delta_hi": -0.6, "d_z": -0.9, "d_z_lo": -1.3, "d_z_hi": -0.4, "n": 12, "n_boot": 10000,
                                                             "n_per_behavior": {b: 2 for b in B}}}}
    if anchor_key:
        tag["anchor"] = {"petri_run_key": anchor_key}
    stats = {"meta": {"anchor_mode": "own"}, "tags": {F.VIEW: tag}}
    json.dump(stats, open(rs / "petri_stats" / "petri_full22_stats.json", "w"))
    if placeholder_key:
        # a sidecar root that enumerates the same object entry under its own harness key with no results
        json.dump({"meta": {}, "tags": {F.VIEW: {"cells": {"constv2-mix-full__lora_r4_ce": {"petri_run_key": placeholder_key, "status": "NO_RESULTS_NO_LEDGER", "behaviors": {}}}}}},
                  open(rs / "petri_stats" / "petri_full22_newcells_stats.json", "w"))
    agg = {F.VIEW: {"cells": {"none": {"cell_id": "none", "run_key": "aaaa0000000000ff", "complete": True, "kind": "none", "cell_class": "base",
                                       "audited": {"expected": 12, "items": 12}, "behaviors": {b: {"mean": {b: 6.0}, "n_items": 2} for b in B}},
                              "si_constv2_mix_full__lora_r4_ce": {"cell_id": "si_constv2_mix_full__lora_r4_ce", "run_key": agg_key, "complete": True, "kind": "lora_all",
                                                                  "cell_class": "class_a", "audited": {"expected": 12, "items": 12},
                                                                  "behaviors": {b: {"mean": {b: 5.0}, "n_items": 2} for b in B}}},
                    "controls": {"none__lens0": {"cell_id": "none__lens0", "run_key": "cccc", "complete": True}}}}
    json.dump(agg, open(cons / "petri_agg.json", "w"))
    return str(rs), str(cons)


def _setup(tmp_path, **kw):
    rs, cons = _files(tmp_path, **kw)
    seeds = F.synthetic_seeds()
    ad = PS.PetriStatsAdapter(rs=rs, consolidated=cons, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    return ad, root


def _two_passes(root, ad):
    """The steady state of a fresh registry: pass 1 parks every comparison (the audited `none` runs are not folded yet), the fold makes them
    current, pass 2 re-reads the group through the §4 decision input and files the contrasts own-current. Returns the SECOND pass's
    `(rep, ing, ing2)`; both passes' folds must refuse and block nothing."""
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=T[0])
    assert rep1.groups_refused == 0 and ing1.refused + ing1b.refused == 0 and ing1.blocked + ing1b.blocked == 0, \
        ([r.error for r in rep1.results], ing1.refusals, ing1b.refusals, ing1.blocked)
    out = F.run_and_ingest(root, ad, now=T[1])
    assert out[0].groups_refused == 0 and out[1].refused + out[2].refused == 0 and out[1].blocked + out[2].blocked == 0, (out[1].refusals, out[2].refusals)
    return out


def _parks(ad):
    """The adapter's row refusals that are anchor parks (the phrasing the hooked adapters share), and the rest."""
    parks = [r for r in ad.refusals.items if PARK_PHRASE in r["why"] or r["why"].startswith("contrast's own anchor")]
    return parks, [r for r in ad.refusals.items if r not in parks]


def _exclude(root: str, ruling_id: str, run_id: str, i: int) -> None:
    """One `exclude` ruling on one run, filed and folded -- under the stand-in dimensions' paper round (`LC.ROUND_ID`; a ruling names its round)."""
    payload = {"ruling_id": ruling_id, "round_id": LC.ROUND_ID, "action": "exclude", "scope_kind": "run", "scope_json": {"run_id": run_id},
               "target_json": {}, "reason": f"synthetic: exclude {run_id}", "ruled_utc": synth.t(i), "ruled_utc_precision": "datetime",
               "ruled_by": "test", "source_ref": "synthetic"}
    R.write_records(os.path.join(root, "records", "rulings", f"{i:06d}-{ruling_id}.jsonl"), [synth.rec("ruling", payload, synth.t(i))])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)


def _steered(root, object_entry, eval_id):
    """The key columns of one steered run of the audit (`object_entry` = its `campaign_cell_id_legacy`)."""
    rows = F.q(root, "SELECT r.run_id, r.campaign, r.model_id, r.eval_id, r.item_set_id, r.instrument_id, r.token_cap_tokens, r.trial_index, r.pass_label, "
                     "r.decode_policy_id, r.view_legacy, r.judge_id FROM run r WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = ?", (object_entry, eval_id))
    assert len(rows) == 1, rows
    return dict(zip(("run_id", "campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label",
                     "decode_policy_id", "view_legacy", "judge_id"), rows[0]))


def _sibling_none(root, steered, *, i0, view_legacy="synthview_sibling", **metric_over):
    """A complete, judged unsteered run of source `petri_stats` at the steered run's EXACT key under ANOTHER standup of the same source
    (`view_legacy` differs), carrying ONE `petri_score` reading under the steered row's judge -- filed as records and folded. Tier 1b's one
    eligible candidate: `sibling_none`. Returns `(run_id, judgment_id)`."""
    run_key = f"petri_sibling:{view_legacy}"
    rid = I.run_id(str(steered["campaign"]), run_key, "none", str(steered["eval_id"]), str(steered["instrument_id"]))
    over = {c: steered[c] for c in ("campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label", "decode_policy_id")}
    over.update(run_id=rid, lane_source_legacy=PS.NAME, view_legacy=view_legacy)
    judge = str(steered["judge_id"])
    _rid, recs = synth.run_records(run_key, i0, "none", "none", n_items=int(metric_over.get("n") or 12), judge=judge, prod=F.producer(), **over)
    assert _rid == rid
    jid = I.judgment_id(rid, judge)
    recs = [r for r in recs if r["record_type"] != "run_metric"]
    recs.append(synth.rec("run_metric", dict({"run_id": rid, "judgment_id": jid, "metric_id": PS.METRIC, "threshold": PS.THRESHOLD, "computed_by": "producer"}, **metric_over),
                          synth.t(i0 + 9), F.producer()))
    R.write_records(os.path.join(root, "records", "sibling", f"{i0:06d}-sibling-none.jsonl"), recs)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)
    assert F.q(root, "SELECT is_current, why FROM v_run_current WHERE run_id = ?", (rid,)) == [(1, None)]
    return rid, jid


def _items(root, run_id, judgment_id, rows, i, metric=PS.METRIC):
    """Item rows of `(run_id, judgment_id)` at `metric` -- `[(item_id, value), ...]` at trial 0, sample 0, turn -1 -- filed under a synthetic
    source of their own (a `source_revision` marker and the rows) and folded."""
    source_id = f"zz_items_{i}"
    rev = f"{source_id}:rev{i}"
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": source_id, "adapter": "synthetic_items", "observed_utc": synth.t(i),
                                           "inputs_json": {f"synthetic:///items/{i}": "b" * 64}, "accepted": 1, "priority": 0}, synth.t(i))
    recs = [synth.rec("item_result", {"run_id": run_id, "item_id": item_id, "metric_id": metric, "trial_index": 0, "sample_index": 0, "turn_index": -1,
                                      "judgment_id": judgment_id, "value": value, "label": None, "api_pool": None, "created_utc": synth.t(i),
                                      "judged_utc": None, "source_revision_id": rev, "withdrawn_by_revision_id": None}, synth.t(i + 1))
            for item_id, value in rows]
    R.write_records(os.path.join(root, "records", f"items_{i}", "000001-source_revision.jsonl"), [marker])
    R.write_records(os.path.join(root, "records", f"items_{i}", "000002-item_result.jsonl.gz"), recs, gz=True)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)


def _contrast_rows(root, run_id):
    return [dict(zip(("anchor_run_id", "anchor_kind", "delta", "delta_method", "computed_by", "computation_id", "anchor_judgment_id", "n_pairs", "extra"), r))
            for r in F.q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, computed_by, computation_id, anchor_judgment_id, n_pairs, extra_json "
                               "FROM run_contrast WHERE run_id = ? AND metric_id = ? AND threshold = ? ORDER BY anchor_run_id", (run_id, PS.METRIC, PS.THRESHOLD))]


def _current_contrast_keys(root):
    return {r[0] for r in F.q(root, "SELECT observation_key FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1", (SOURCE_ID,))}


def _metric_extra(root, run_id):
    rows = F.q(root, "SELECT extra_json FROM run_metric WHERE run_id = ? AND metric_id = ?", (run_id, PS.METRIC))
    assert len(rows) == 1, rows
    return json.loads(rows[0][0] or "{}")


def _paper_row(root, run_id):
    return F.q(root, "SELECT anchor_run_id, anchor_kind, delta, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = ?", (run_id, PS.METRIC))


# ============================================================================================
# the steady state: park, file own-current, quiet
# ============================================================================================

def test_runs_per_behaviour_items_and_contrasts(tmp_path):
    """Pass 1 over a fresh registry parks every comparison (the audited `none` runs are emitted by this very read: `not_in_frame`, no
    provisional anchors), the fold makes them current, pass 2 files the seven contrasts own-current with `extra.anchor_selection` and the
    frame's anchor judgment (was ''), the parks lifted by name; pass 3 is quiet."""
    ad, root = _setup(tmp_path)
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=T[0])
    assert rep1.groups_refreshed == 1 and rep1.groups_refused == 0, [r.error for r in rep1.results]
    assert ing1.refused + ing1b.refused == 0 and ing1.blocked + ing1b.blocked == 0, (ing1.refusals, ing1b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)]
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 7 and all(": not_in_frame]" in p["why"] and "not a run the registry has folded" in p["why"] for p in parks)
    assert ad.stats["contrasts_parked"] == 7 and ad.stats["contrasts_reanchored"] == 0 and ad.stats["anchor_states"] == {"no_current_anchor": 7}
    blocks = F.q(root, "SELECT json_extract(rm.extra_json, '$.delta_block.anchor.own_reason'), json_extract(rm.extra_json, '$.delta_block.delta') "
                       "FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.object_id <> 'none' AND rm.metric_id = ?", (PS.METRIC,))
    assert len(blocks) == 7 and {b[0] for b in blocks} == {"not_in_frame"} and {b[1] for b in blocks} == {-1.0}
    # ---- pass 2: the `none` runs are current -> the decision input moved -> every contrast own-current
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[1])
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0 and rep.anchor_decisions_changed == 1, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    assert ad.stats["anchor_states"] == {"own_current": 7} and ad.stats["contrasts_parked"] == 0
    runs = F.q(root, "SELECT eval_id, producer_run_key_legacy, producer_object_key_legacy, lifecycle_state, serving_path FROM run ORDER BY 1, 3")
    assert len(runs) == 14   # (6 behaviours + pooled) x (none, lora)
    assert {r[1] for r in runs} == {"aaaa0000000000ff", "aaaa000000000001"} and {r[2] for r in runs} == {f"{F.VIEW}:none", f"{F.VIEW}:{F.LORA}"}
    assert all(r[3] == "judged" for r in runs)
    assert {r[4] for r in runs} == {"passthrough", "lens"}
    # the CURRENT item rows (two revisions have filed them: pass 1's and pass 2's; the table keeps both, the current-row rank selects one)
    items = F.q(root, "SELECT r.eval_id, COUNT(*) FROM v_item_result_current ir JOIN run r ON r.run_id = ir.run_id GROUP BY 1")
    d = dict(items)
    assert d["petri_pooled"] == 24 and d["petri_defer_to_users"] == 4   # 12 items x 2 entries pooled; 2 scenarios x 2 entries per behaviour
    ids = {r[0] for r in F.q(root, "SELECT DISTINCT item_id FROM v_item_result_current")}
    assert "defer_to_users/scenario_1_a" in ids
    c = F.q(root, "SELECT r.eval_id, rc.delta, rc.effect_size, rc.effect_kind, rc.n_pairs, a.producer_object_key_legacy, rc.anchor_judgment_id, rc.anchor_kind, "
                  "rc.extra_json, rc.computed_by FROM run_contrast rc JOIN run r ON r.run_id = rc.run_id JOIN run a ON a.run_id = rc.anchor_run_id ORDER BY 1")
    assert len(c) == 7 and all(r[5] == f"{F.VIEW}:none" for r in c)
    pooled = [r for r in c if r[0] == "petri_pooled"][0]
    assert pooled[1] == -1.0 and pooled[2] == -0.9 and pooled[3] == "dz" and pooled[4] == 12
    # the anchor's judgment is the frame's (the one complete judgment of the `none` under the Petri judge), the selection block on every contrast
    none_jids = {r[0] for r in F.q(root, "SELECT judgment_id FROM judgment WHERE run_id IN (SELECT run_id FROM run WHERE object_id = 'none')")}
    assert all(r[6] and r[6] in none_jids for r in c) and all(r[7] == "same_standup_none" and r[9] == "producer" for r in c)
    for r in c:
        sel = json.loads(r[8])["anchor_selection"]
        assert sel["state"] == "own_current" and sel["own_reason"] == "current" and sel["rule"] == AC.RULE and sel["n_candidates"] == 1
        assert sel["key"] == {"token_cap_tokens": 1024, "trial_index": 0, "pass_label": None, "decode_policy_id": None}
    # the parks of pass 1 are lifted BY NAME (`delta_block: null`, never an omission) on every steered petri_score row
    lifted = F.q(root, "SELECT json_type(rm.extra_json, '$.delta_block') FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.object_id <> 'none' AND rm.metric_id = ?",
                 (PS.METRIC,))
    assert len(lifted) == 7 and {t[0] for t in lifted} == {"null"}
    sp = F.q(root, "SELECT source_producers_json FROM run_metric WHERE metric_id = 'petri_score' LIMIT 1")
    assert len(json.loads(sp[0][0])) == 2
    # the harness's derived control legs are not runs
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE campaign_cell_id_legacy LIKE 'none__%'") == [(0,)]
    # ---- pass 3: quiet
    rep3, _, _ = F.run_and_ingest(root, ad, now=T[2])
    assert rep3.inputs_changed == 0 and rep3.groups_refreshed == 0 and rep3.anchor_decisions_changed == 0


def test_two_source_disagreement_refuses_the_cell(tmp_path):
    ad, root = _setup(tmp_path, agg_key="aaaa000000000001", stats_key="bbbb000000000002")
    rep, _, _ = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1
    assert len(ad.refusals) == 1 and "two-source guard" in ad.refusals.items[0]["why"]
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE campaign_cell_id_legacy = ?", (F.LORA,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE campaign_cell_id_legacy = 'none'") == [(7,)]


def test_itemless_placeholder_does_not_veto_the_aggregate(tmp_path):
    ad, root = _setup(tmp_path, placeholder_key="dddd000000000003")
    rep, _, _ = _two_passes(root, ad)
    assert rep.groups_refreshed == 1 and len(ad.refusals) == 0, ad.refusals.items
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE producer_run_key_legacy = 'aaaa000000000001'") == [(7,)]


# ============================================================================================
# a re-picked anchor: the registry pairs the two runs' item rows per behavior
# ============================================================================================

def test_a_retired_own_none_re_picks_a_same_key_sibling_and_the_registry_pairs_the_item_rows_per_behavior(tmp_path):
    """The pooled run's own `none` is retired by an `exclude` ruling while a current unsteered run of the EXACT key stands under another
    standup of this source: the next pass re-reads the group (a new revision id over the same files: the §4 decision input moved). Without
    item rows on the sibling e4 refuses it and the comparison parks by name (the sidecar's block on the metric row, the old key withdrawn);
    once the sibling's items fold with every scenario shared, the registry files the per-behavior paired difference -- each side's helper mean
    EQUALS its filed `mean_of_behavior_means` (the estimand's weighting is preserved) -- `computed_by registry`, `sibling_none`, under its own
    computation row, the paper frame's one contrast on the row; the six per-behavior contrasts stand own-current throughout; then quiet."""
    ad, root = _setup(tmp_path)
    _two_passes(root, ad)
    lora = _steered(root, F.LORA, "petri_pooled")
    own = _contrast_rows(root, lora["run_id"])
    assert len(own) == 1 and own[0]["anchor_kind"] == "same_standup_none" and own[0]["delta"] == -1.0 and own[0]["computed_by"] == "producer"
    own_none = own[0]["anchor_run_id"]
    filed_steered = F.q(root, "SELECT value FROM run_metric WHERE run_id = ? AND metric_id = ?", (lora["run_id"], PS.METRIC))[0][0]
    assert filed_steered == pytest.approx(5.1)
    keys2 = _current_contrast_keys(root)
    anchor_value = 6.2
    sib, sib_jid = _sibling_none(root, lora, i0=300, value=anchor_value, n=12)
    _exclude(root, "r_synth_exclude_petri_none", own_none, 10)
    # ---- (1) the sibling carries no item rows: e4 refuses it, the comparison parks by name; the per-behavior contrasts stand
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[2])
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 0 and ad.stats["contrasts_parked"] == 1 and ad.stats["anchor_states"] == {"own_current": 6, "no_current_anchor": 1}
    blk = _metric_extra(root, lora["run_id"])["delta_block"]
    assert blk["delta"] == -1.0 and blk["effect_size"] == -0.9 and blk["computed_by"] == "producer" and blk["anchor"]["state"] == "no_current_anchor"
    assert blk["anchor"]["own_reason"] == "excluded" and blk["anchor"]["own_anchor_run_id"] == own_none and blk["anchor"]["registry_contrast_filed_against"] is None
    assert sib in blk["why_not_a_contrast_row"] and "no current item rows at petri_score" in blk["why_not_a_contrast_row"] and blk["rule"] == AC.RULE
    assert blk["producer_contrast_extra"]["n_per_behavior"] == {b: 2 for b in B}
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 1 and ": excluded]" in parks[0]["why"] and sib in parks[0]["why"]
    assert _contrast_rows(root, lora["run_id"]) == own                                   # the old row stays in the table; its key is withdrawn
    assert not any(json.loads(k)["run_id"] == lora["run_id"] for k in _current_contrast_keys(root))
    assert len(_current_contrast_keys(root)) == len(keys2) - 1
    parked_row = _paper_row(root, lora["run_id"])
    assert len(parked_row) == 1 and parked_row[0][0] is None and parked_row[0][1] is None
    # the two comparisons of the pooled row (without e4, then with) are in the decision set beside the six own-current ones
    head = F.q(root, "SELECT revision_id, extra_json FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (SOURCE_ID,))[0]
    extra = json.loads(head[1] or "{}")
    pooled_cmps = [e for e in extra["anchor_decisions"]["comparisons"] if e["comparison"]["run_id"] == lora["run_id"]]
    assert [c["comparison"]["needs_items"] for c in pooled_cmps] == [False, True]
    assert [c["decision"]["anchor_run_id"] for c in pooled_cmps] == [sib, AC.PARKED] and extra["anchor_decisions"]["n_comparisons"] == 8
    # ---- (2) the sibling's items fold, every scenario shared: the paired statistic files
    _items(root, sib, sib_jid, [(f"{b}/scenario_{i}_a", anchor_value - 0.2) for i, b in enumerate(B)] + [(f"{b}/scenario_{i}_b", anchor_value + 0.2) for i, b in enumerate(B)], 30)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[3])
    assert rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 1 and ad.stats["contrasts_parked"] == 0 and ad.stats["reanchor_computations"] == 1
    assert ad.stats["anchor_states"] == {"own_current": 6, "sibling_same_source": 1} and _parks(ad) == ([], [])
    rows = _contrast_rows(root, lora["run_id"])
    assert sorted(r["anchor_run_id"] for r in rows) == sorted([own_none, sib])
    new = next(r for r in rows if r["anchor_run_id"] == sib)
    assert new["anchor_kind"] == "sibling_none" and new["computed_by"] == "registry" and new["delta_method"] == DC.PAIRED_ITEMS_METHOD
    assert new["delta"] == pytest.approx(filed_steered - anchor_value) and new["n_pairs"] == 12 and new["anchor_judgment_id"] == sib_jid and new["computation_id"] == ""
    ex = json.loads(new["extra"])
    assert ex["paired"] is True and ex["weighting"] == "mean_of_behavior_means" and set(ex["per_behavior"]) == set(B)
    assert all(v["n_shared"] == 2 and v["delta"] == pytest.approx(filed_steered - anchor_value) for v in ex["per_behavior"].values())
    # the helper's one-sided means equal the filed estimands when every scenario is shared
    assert ex["means"]["steered"] == pytest.approx(filed_steered) and ex["means"]["anchor"] == pytest.approx(anchor_value)
    assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded" and ex["anchor_selection"]["own_anchor_run_id"] == own_none
    assert ex["anchor_selection"]["candidates"] == [sib] and [r[0] for r in ex["anchor_selection"]["rejected"]] == [own_none] and "exclude" in ex["anchor_selection"]["rejected"][0][1]
    # the sibling's standup as the frame holds it (the synthetic records patch `cluster_job_id` 4242 onto every run)
    assert ex["cross_source"] is False and ex["anchor_standup"] == {"lane_source_legacy": PS.NAME, "view_legacy": "synthview_sibling", "cluster_job_id": "4242"}
    assert ex["contrast_caps"] == {"steered": 1024, "anchor": 1024} and ex["producer_delta_parked_on_metric"] is True and ex["anchor_reading"]["judgment_id"] == sib_jid
    assert F.q(root, "SELECT algorithm, version FROM computation WHERE computation_id = ?", (ex["statistic_computation_id"],)) == [("registry.paired_items_contrast", AC.RULE)]
    blk = _metric_extra(root, lora["run_id"])["delta_block"]
    assert blk["delta"] == -1.0 and blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["registry_contrast_filed_against"] == sib and "stands" in blk["why_not_a_contrast_row"]
    keys = _current_contrast_keys(root)
    assert {json.loads(k)["anchor_run_id"] for k in keys if json.loads(k)["run_id"] == lora["run_id"]} == {sib} and all(json.loads(k)["anchor_run_id"] != own_none for k in keys)
    assert len(keys) == len(keys2)
    frame = _paper_row(root, lora["run_id"])
    assert len(frame) == 1 and frame[0][:2] == (sib, "sibling_none") and frame[0][2] == pytest.approx(filed_steered - anchor_value) and frame[0][3] == 1
    # ---- quiet afterwards
    rep5, _, _ = F.run_and_ingest(root, ad, now=T[4])
    assert rep5.inputs_changed == 0 and rep5.groups_refreshed == 0 and rep5.anchor_decisions_changed == 0


def test_a_behavior_with_no_shared_scenario_refuses_by_name_and_unequal_counts_file_the_behavior_weighted_difference(tmp_path):
    """The behavior-weighted estimand cannot be preserved when a behavior has no scenario judged on both sides: the re-pick refuses by name
    and parks. With unequal per-behavior counts (one behavior shares two scenarios, the others one) the item-weighted and behavior-weighted
    differences differ, and the helper files the latter -- the mean over behaviors of each behavior's mean per-scenario difference."""
    ad, root = _setup(tmp_path)
    _two_passes(root, ad)
    lora = _steered(root, F.LORA, "petri_pooled")
    own_none = _contrast_rows(root, lora["run_id"])[0]["anchor_run_id"]
    sib, sib_jid = _sibling_none(root, lora, i0=300, value=6.2, n=11)
    _exclude(root, "r_synth_exclude_petri_none", own_none, 10)
    # (1) items for five behaviors only: the sixth has no shared scenario -> refused by name, parked (e4 admits the run: it HAS item rows)
    missing = B[-1]
    first = [(f"{B[0]}/scenario_0_a", 6.0), (f"{B[0]}/scenario_0_b", 6.4)] + [(f"{b}/scenario_{i}_a", 5.0) for i, b in enumerate(B) if 0 < i < len(B) - 1]
    _items(root, sib, sib_jid, first, 30)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[2])
    assert rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 0 and ad.stats["contrasts_parked"] == 1
    blk = _metric_extra(root, lora["run_id"])["delta_block"]
    assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["anchor_run_id"] == sib and blk["anchor"]["registry_contrast_filed_against"] is None
    assert f"stratum '{missing}' has no item judged on both sides" in blk["why_not_a_contrast_row"] and "is not derived" in blk["why_not_a_contrast_row"]
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 1 and parks[0]["why"].startswith("contrast's own anchor") and missing in parks[0]["why"]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast WHERE run_id = ? AND anchor_run_id = ?", (lora["run_id"], sib)) == [(0,)]
    # (2) the sixth behavior gains one shared scenario: 7 shared items, behavior 0 with two -> the behavior-weighted difference, not the item-weighted
    _items(root, sib, sib_jid, [(f"{missing}/scenario_{len(B) - 1}_a", 5.0)], 40)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[3])
    assert rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 1 and ad.stats["contrasts_parked"] == 0
    new = next(r for r in _contrast_rows(root, lora["run_id"]) if r["anchor_run_id"] == sib)
    diffs_b0 = [5.0 - 6.0, 5.2 - 6.4]                      # the steered rows {5.0, 5.2} against the sibling's {6.0, 6.4}
    behavior_weighted = (sum(diffs_b0) / 2 + 0.0 * (len(B) - 1)) / len(B)
    item_weighted = (sum(diffs_b0) + 0.0 * (len(B) - 1)) / (2 + len(B) - 1)
    assert behavior_weighted != pytest.approx(item_weighted)
    assert new["delta"] == pytest.approx(behavior_weighted) and new["delta"] != pytest.approx(item_weighted) and new["n_pairs"] == len(B) + 1
    ex = json.loads(new["extra"])
    assert ex["per_behavior"][B[0]]["n_shared"] == 2 and all(ex["per_behavior"][b]["n_shared"] == 1 for b in B[1:])
    assert new["delta_method"] == DC.PAIRED_ITEMS_METHOD and new["computed_by"] == "registry" and new["anchor_kind"] == "sibling_none"
    assert _paper_row(root, lora["run_id"])[0][:2] == (sib, "sibling_none")


def test_a_sidecar_anchor_block_naming_an_unfolded_none_is_a_reason_not_to_use_that_id_never_a_block(tmp_path):
    """The sidecar's `anchor.petri_run_key` names a `none` audit the registry never folded (a refused or never-read row). EMISSION_VERSION 1 put
    the composed id on every contrast and the harvester held the whole revision on the missing anchor run; now the own id reads
    `not_in_frame`, the exact-key search finds the standup's OWN audited `none` (same standup: `same_standup_none`), and -- both sides
    carrying item rows -- the registry files the per-behavior paired difference on all seven comparisons. Nothing is blocked."""
    ad, root = _setup(tmp_path, anchor_key="eeee000000000099")
    rep, ing, ing2 = _two_passes(root, ad)
    assert ing.blocked + ing2.blocked == 0 and F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert ad.stats["anchor_states"] == {"sibling_same_source": 7} and ad.stats["contrasts_reanchored"] == 7 and ad.stats["contrasts_parked"] == 0
    assert ad.stats["reanchor_computations"] == 1 and _parks(ad) == ([], [])
    lora = _steered(root, F.LORA, "petri_pooled")
    composed = I.run_id(LC.CAMPAIGN, "eeee000000000099", LC.petri_object_key(F.VIEW, "none"), "petri_pooled", str(lora["instrument_id"]))
    rows = F.q(root, "SELECT rc.run_id, rc.anchor_run_id, rc.anchor_kind, rc.computed_by, rc.delta_method, rc.delta, rc.extra_json, a.producer_object_key_legacy, a.producer_run_key_legacy "
                     "FROM run_contrast rc JOIN run a ON a.run_id = rc.anchor_run_id")
    assert len(rows) == 7 and all(r[7] == f"{F.VIEW}:none" and r[8] == "aaaa0000000000ff" for r in rows)
    assert {r[2] for r in rows} == {"same_standup_none"} and {r[3] for r in rows} == {"registry"} and {r[4] for r in rows} == {DC.PAIRED_ITEMS_METHOD}
    assert all(r[5] == pytest.approx(5.1 - 6.1) for r in rows)                      # every scenario shared: the per-scenario means differ by 1.0
    pooled = next(r for r in rows if r[0] == lora["run_id"])
    ex = json.loads(pooled[6])
    assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "not_in_frame"
    assert ex["anchor_selection"]["own_anchor_run_id"] == composed and "not a run the registry has folded" in ex["anchor_selection"]["own_anchor_why"]
    assert ex["cross_source"] is False and ex["anchor_standup"]["view_legacy"] == F.VIEW and len(ex["per_behavior"]) == len(B)
    blk = _metric_extra(root, lora["run_id"])["delta_block"]
    assert blk["anchor"]["own_anchor_run_id"] == composed and blk["anchor"]["registry_contrast_filed_against"] == pooled[1] and blk["delta"] == -1.0
    assert _paper_row(root, lora["run_id"]) == [(pooled[1], "same_standup_none", pytest.approx(-1.0), 1)]
    rep3, _, _ = F.run_and_ingest(root, ad, now=T[2])
    assert rep3.groups_refreshed == 0 and rep3.anchor_decisions_changed == 0


def test_the_smoke_names_the_unavailable_anchor_currency_for_the_petri_group(tmp_path):
    ad, _root = _setup(tmp_path)
    out = DC.smoke(ad)
    assert out["groups"] == 1 and out["anchor_unavailable_groups"] == 1 and out["records"] == 0      # the group refuses WHOLE: nothing of it is counted
    assert len(out["refused_groups"]) == 1 and out["refused_groups"][0].startswith("petri: anchor currency unavailable") and "no registry connection" in out["refused_groups"][0]


# --------------------------------------------------------------------------------------------
# EMISSION_VERSION 4: the reader's run records are a stats reader's (identity stated, the rest marked, nulls omitted)
# --------------------------------------------------------------------------------------------


def _run_records_on_disk(root):
    """Every `run` record the adapter wrote (its revision directories under records/adapters/petri_stats/), newest revision first."""
    import glob
    import gzip
    out = []
    base = os.path.join(root, "records", "adapters", PS.NAME)
    for rev in sorted((d for d in os.listdir(base) if not d.startswith("_")), key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True):
        for f in sorted(glob.glob(os.path.join(base, rev, "*-run.jsonl*"))):
            opener = gzip.open if f.endswith(".gz") else open
            with opener(f, "rt", encoding="utf-8") as fh:
                out.append((rev, [json.loads(line) for line in fh if line.strip()]))
    return out


def test_run_records_state_the_identity_mark_the_rest_and_the_export_row_does_not_flap(tmp_path):
    """End to end: the reader's `run` records carry `extra.unstated_columns` over every non-identity column, the nulls
    carried and marked too; an export record for the same run id then owns the row's provenance, and a forced re-read
    of the reader moves none of it back -- while the sidecar's `serving_identity` the export omits is still filled from the reader.
    EMISSION_VERSION is 4."""
    assert PS.PetriStatsAdapter.EMISSION_VERSION == 4
    ad, root = _setup(tmp_path)
    _two_passes(root, ad)
    revs = _run_records_on_disk(root)
    assert revs and all(recs for _, recs in revs)
    for _, recs in revs:
        for rec in recs:
            p = rec["payload"]
            marker = p["extra"]["unstated_columns"]
            rest = {k for k in p if k not in LC.RUN_IDENTITY_COLUMNS and k not in ("extra", "field_confidence_json")}
            assert set(marker) == rest, sorted(set(marker) ^ rest)
            assert not (set(marker) & set(LC.RUN_IDENTITY_COLUMNS))
            assert set(p["field_confidence_json"]) <= set(marker)          # the carrier: the marked columns' entries only, no identity entry
            # the columns the reader carries as null ride as marked withdrawals (round 2 of the review): carried, null, marked
            assert p["notes"] is None and p["pass_label"] is None and "notes" in marker and "pass_label" in marker
            assert p["lane_source_legacy"] == PS.NAME and p["serving_identity"] == "proxy_vllm+lens+interleaved"
    eval_pooled = ad.seeds.eval_for_spelling("petri_pooled")
    rid = _steered(root, F.LORA, eval_pooled)["run_id"]
    row0 = F.q(root, "SELECT lane_source_legacy, generated_utc, store_uri, serving_identity, notes FROM run WHERE run_id = ?", (rid,))[0]
    assert row0[0] == PS.NAME and row0[1] == "2026-09-07T23:31:37.600Z" and row0[3] == "proxy_vllm+lens+interleaved"   # the reader minted the row
    # the export's record for the same run id: a full statement of the row's provenance, omitting the sidecar-only columns
    reader_rec = next(r for _, recs in revs for r in recs if r["payload"]["run_id"] == rid)
    export = {k: v for k, v in reader_rec["payload"].items() if k not in ("extra", "serving_identity", "n_items_planned")}
    export.update(lane_source_legacy="r2_export", generated_utc="2026-09-16T20:32:11.000Z", queued_utc="2026-09-16T20:32:11.000Z",
                  started_utc="2026-09-16T20:32:11.000Z", store_uri="~/synthetic/consolidated/petri_agg.json", notes="judge score 1-10 (synthetic)",
                  extra={"lane_kind": "synthetic"})
    R.write_records(os.path.join(root, "records", "export", "000001-run.jsonl"), [synth.rec("run", export, T[3])])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    row1 = F.q(root, "SELECT lane_source_legacy, generated_utc, store_uri, serving_identity, notes, n_items_planned FROM run WHERE run_id = ?", (rid,))[0]
    assert row1[:3] == ("r2_export", "2026-09-16T20:32:11.000Z", "~/synthetic/consolidated/petri_agg.json") and row1[4] == "judge score 1-10 (synthetic)"
    assert row1[3] == "proxy_vllm+lens+interleaved" and row1[5] == 12       # the export omits them: the reader's fill stands
    conf = json.loads(F.q(root, "SELECT field_confidence_json FROM run WHERE run_id = ?", (rid,))[0][0])
    assert conf["serving_identity"].get("unstated") is True and conf["n_items_planned"].get("unstated") is True   # the fill's flag carried through the owner's statement
    assert not (conf.get("lane_source_legacy") or {}).get("unstated") and not (conf.get("n_items_generated") or {}).get("unstated")   # the owner's statements
    # a forced re-read of the reader (a new emission over unchanged inputs): the export's provenance stands, the reader's statistics re-file
    ad.EMISSION_VERSION = 5
    rep, ing3, ing4 = F.run_and_ingest(root, ad, now=T[4])
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0 and ing3.refused + ing4.refused == 0, ([r.error for r in rep.results], ing3.refusals)
    row2 = F.q(root, "SELECT lane_source_legacy, generated_utc, store_uri, serving_identity, notes FROM run WHERE run_id = ?", (rid,))[0]
    assert row2 == row1[:5], (row1, row2)
    # the revision is ACCEPTED (the marker is a fold directive, not a refusal) and the row's extra merged key-wise, the marker never parked
    assert F.q(root, "SELECT COUNT(*) FROM source_revision WHERE adapter = ? AND accepted = 0", (PS.NAME,)) == [(0,)]
    extra = json.loads(F.q(root, "SELECT extra_json FROM run WHERE run_id = ?", (rid,))[0][0])
    assert extra.get("unstated_columns") is None and extra.get("lane_kind") == "synthetic" and extra.get("petri_run_key") == "aaaa000000000001"
    # the audit's count moves in the sidecar -> the reader's next revision REFRESHES the count on the run no owner states
    # (the `none` cell: reader-only, flagged) and leaves the owner's stated count on the lora run alone
    none_rid = F.q(root, "SELECT run_id FROM run WHERE campaign_cell_id_legacy = 'none' AND eval_id = ?", (eval_pooled,))[0][0]
    assert F.q(root, "SELECT n_items_generated FROM run WHERE run_id = ?", (none_rid,)) == [(12,)]
    stats_path = os.path.join(ad.rs, "petri_stats", "petri_full22_stats.json")
    data = json.load(open(stats_path))
    data["tags"][F.VIEW]["cells"]["none"]["audited_items"] = 10
    json.dump(data, open(stats_path, "w"))
    rep5, ing5, ing6 = F.run_and_ingest(root, ad, now=T[5])
    assert rep5.groups_refreshed == 1 and rep5.groups_refused == 0 and ing5.refused + ing6.refused == 0, ([r.error for r in rep5.results], ing5.refusals)
    assert F.q(root, "SELECT n_items_generated, n_items_judged FROM run WHERE run_id = ?", (none_rid,)) == [(10, 10)]       # refreshed
    assert F.q(root, "SELECT n_items_generated, lane_source_legacy FROM run WHERE run_id = ?", (rid,)) == [(12, "r2_export")]   # the owner's, untouched
    none_conf = json.loads(F.q(root, "SELECT field_confidence_json FROM run WHERE run_id = ?", (none_rid,))[0][0])
    assert none_conf["n_items_generated"].get("unstated") is True and all(ent.get("unstated") for ent in none_conf.values())    # a reader-only row: every value the reader's


def test_a_cell_with_no_producer_clock_is_refused_by_name_never_stamped_with_the_read_time(tmp_path):
    """The determinism rule applied to the clock: `queued_utc` / `generated_utc` used to fall back to the pass's `now` when the sidecar
    recorded neither `completed_at` nor a ledger snapshot -- a stamp that moved on every re-read. Such a cell is now a row refusal naming the
    missing clock; the cells with a clock read as before."""
    rs, cons = _files(tmp_path)
    path = os.path.join(rs, "petri_stats", "petri_full22_stats.json")
    data = json.load(open(path))
    data["tags"][F.VIEW]["cells"][F.LORA.replace("constv2-mix-full", "si_constv2_mix_full")].pop("completed_at")
    json.dump(data, open(path, "w"))
    seeds = F.synthetic_seeds()
    ad = PS.PetriStatsAdapter(rs=rs, consolidated=cons, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[0])
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    refused = [r for r in ad.refusals.items if "no producer clock" in r["why"]]
    assert len(refused) == 1 and refused[0]["key"] == f"{F.VIEW}|{F.LORA}" and "read time" in refused[0]["why"]
    runs = F.q(root, "SELECT campaign_cell_id_legacy, queued_utc, generated_utc FROM run")
    assert runs and all(r[0] == "none" for r in runs) and all(r[1] == "2026-09-07T23:31:37.600Z" for r in runs)
    assert not any(T[0][:13] in str(r[1]) or T[0][:13] in str(r[2]) for r in runs)


def test_a_corrected_sidecar_that_drops_its_attestations_withdraws_the_readers_own_serving_identity_and_intervention_hash(tmp_path):
    """Round 2 of the review (2026-10-07): the reader puts `serving_identity` and `intervention_hash` on its run record as the sidecar attests
    them, None once it no longer does. The record used to OMIT the null, so a corrected sidecar left the earlier values standing as current
    with no refusal. Now the null rides as a marked withdrawal: on the row the reader alone wrote (the standup's `none`) and on the
    export-owned row whose export omits the two columns (the reader's fill), both become null with the reader's stamp; the export's own
    provenance does not move."""
    ad, root = _setup(tmp_path)
    _two_passes(root, ad)
    eval_pooled = ad.seeds.eval_for_spelling("petri_pooled")
    rid = _steered(root, F.LORA, eval_pooled)["run_id"]
    none_rid = F.q(root, "SELECT run_id FROM run WHERE campaign_cell_id_legacy = 'none' AND eval_id = ?", (eval_pooled,))[0][0]
    # the export owns the lora run's provenance and omits the sidecar-only columns: the reader's values are its fill
    reader_rec = next(r for _, recs in _run_records_on_disk(root) for r in recs if r["payload"]["run_id"] == rid)
    export = {k: v for k, v in reader_rec["payload"].items() if k not in ("extra", "serving_identity", "intervention_hash", "n_items_planned") and v is not None}
    export.update(lane_source_legacy="r2_export", generated_utc="2026-09-16T20:32:11.000Z", store_uri="~/synthetic/consolidated/petri_agg.json",
                  notes="judge score 1-10 (synthetic)", extra={"lane_kind": "synthetic"})
    R.write_records(os.path.join(root, "records", "export", "000001-run.jsonl"), [synth.rec("run", export, T[3])])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    q = "SELECT serving_identity, intervention_hash, lane_source_legacy, notes, store_uri FROM run WHERE run_id = ?"
    assert F.q(root, q, (rid,)) == [("proxy_vllm+lens+interleaved", "1111000011110000", "r2_export", "judge score 1-10 (synthetic)", "~/synthetic/consolidated/petri_agg.json")]
    assert F.q(root, q, (none_rid,))[0][:3] == ("proxy_vllm+lens+interleaved", "1111000011110000", PS.NAME)
    # the corrected sidecar: neither audited cell attests a serving identity or an intervention hash any more
    stats_path = os.path.join(ad.rs, "petri_stats", "petri_full22_stats.json")
    data = json.load(open(stats_path))
    for c in data["tags"][F.VIEW]["cells"].values():
        c.pop("serving_identity", None)
        c.pop("intervention_hash", None)
    json.dump(data, open(stats_path, "w"))
    rep, ing5, ing6 = F.run_and_ingest(root, ad, now=T[5])
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0 and ing5.refused + ing6.refused == 0, ([r.error for r in rep.results], ing5.refusals, ing6.refusals)
    revs = _run_records_on_disk(root)
    newest = revs[0][0]
    for rev, recs in revs:
        if rev != newest:
            continue
        for rec in recs:                                           # the new revision CARRIES the nulls, marked
            p = rec["payload"]
            assert "serving_identity" in p and p["serving_identity"] is None and "intervention_hash" in p and p["intervention_hash"] is None
            assert "serving_identity" in p["extra"]["unstated_columns"] and "intervention_hash" in p["extra"]["unstated_columns"]
    # withdrawn on both rows; the export's provenance stands
    assert F.q(root, q, (rid,)) == [(None, None, "r2_export", "judge score 1-10 (synthetic)", "~/synthetic/consolidated/petri_agg.json")]
    assert F.q(root, q, (none_rid,))[0][:3] == (None, None, PS.NAME)
    conf = json.loads(F.q(root, "SELECT field_confidence_json FROM run WHERE run_id = ?", (rid,))[0][0])
    assert conf["serving_identity"].get("unstated") is True and conf["intervention_hash"].get("unstated") is True   # the withdrawal's provenance
    assert not (conf.get("lane_source_legacy") or {}).get("unstated")
    assert F.q(root, "SELECT COUNT(*) FROM source_revision WHERE adapter = ? AND accepted = 0", (PS.NAME,)) == [(0,)]
