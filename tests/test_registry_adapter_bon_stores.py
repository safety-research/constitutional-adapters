"""The best-of-N stores adapter on synthetic statistics: two runs per campaign store keyed with the r2 row key, the 512 store's
counts as degeneration rows over draws, the counterfactual metric with its self-contrast parked, the vs-unsteered delta as a
contrast against the view's none; and the REGISTERED points of another source (relu_paired's paired writes): the instrument family
chosen from the store's RECORDED sampler (the vendor-preset seeds), the 512 reading filed on NO run -- its counts parked on the climb
run's extra with relu_paired's run id as a cross-reference, the drop counted (version 4; version 2 attached it to that id and the
harvester held the revision when relu_paired had filed no such run; a second 512 run row was never the contract), the climb run as the
paper-facing row, the same-standup contrast read off the statistics file, the degeneration rows per run, the emitted-run self-check that
refuses a batch naming a run it does not emit, a point with no recorded sampler REFUSED by name, and the best-of-N CURVE (version 5):
seven `asr_at_n<N>` rows per climb run in seedrep_cluster's shape, none on a 512 run, a missing grid refused by name. Every store,
manifest and object here is invented."""
from __future__ import annotations

import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import identity as I
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import paper as P
from sisteer.registry import vocab as V
from sisteer.registry.adapters import bon_native as BN
from sisteer.registry.adapters import bon_stores as BS
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import relu_paired as RP
from sisteer.registry.adapters import seedrep_cluster as SC

EVAL = "strongreject_bon"
POINT = "paired_d1_s1"                       # the registration's point spelling (relu_paired's view_legacy)
PAIRED_KEY = f"{F.VIEW}_paired_d1_s1"        # the statistics file's group spelling of the point
PAIRED_KEY_NO_MANIFEST = f"{F.VIEW}_paired_d1_s2"   # registered, but no store record on this host
PAIRED_KEY_UNREGISTERED = f"{F.VIEW}_paired_d1_s3"  # in the statistics file only
ROOT_REL = f"relu_paired/transcripts_paired_d1_s1/sdfbase_full22_{F.VIEW}_paired_d1_s1"


BON_GRID_NS = (1, 2, 4, 8, 16, 32, 64, 75)


def _grid(p75: float) -> dict:
    """The statistics file's per-N grid of a climb (`asr_at_n_4096_cf_t<t>`): the producer's p per N as a STRING key, rising to the headline's p
    at 75 -- over 10 goals, so k = p x 10 is an integer at every N."""
    steps = (0.2, 0.2, 0.4, 0.4, 0.6, 0.8, 1.0, 1.0)
    return {str(N): round(round(p75 * s * 10) / 10, 2) for N, s in zip(BON_GRID_NS, steps)}


def _cell(group, cid, asr, n_at_cap=200, n_loop=5):
    return {"lane": group, "cell_id": cid, "method": "m", "n_rows_512": 750, "n_at_cap_512": n_at_cap, "n_loop_512": n_loop, "n_loop_rep_ratio_512": n_loop, "n_secondary_flag_only_512": 2,
            "n_coherent_512": n_at_cap - n_loop, "status": "graded", "asr_512_t0.5": {"k": int(asr * 10), "n": 10, "p": asr, "lo": asr - 0.2, "hi": asr + 0.2},
            "n_climbed": n_at_cap - n_loop, "n_4096_terminated": n_at_cap - n_loop - 5, "n_4096_residual_at_cap": 5, "n_4096_residual_loop": 4,
            "residual_at_cap_share_of_climbed": 5 / (n_at_cap - n_loop), "finish_reasons_4096": {"stop": n_at_cap - n_loop - 5, "length": 5},
            "asr_4096_cf_t0.5": {"k": int(asr * 10) - 1, "n": 10, "p": asr - 0.1, "lo": asr - 0.3, "hi": asr + 0.1,
                                 "paired_vs_512": {"delta_pp": -10.0, "lo_pp": -20.0, "hi_pp": 0.0, "n_goals": 10, "mcnemar_exact_p": 0.5}},
            "asr_4096_cf_t0.75": {"k": 3, "n": 10, "p": 0.3, "lo": 0.1, "hi": 0.6, "paired_vs_512": {"delta_pp": 5.0}},
            "asr_at_n_4096_cf_t0.5": _grid(round(asr - 0.1, 2)), "asr_at_n_4096_cf_t0.75": _grid(0.3)}


def _files(tmp_path, paired: bool = False, half_dose: bool = False):
    rs = tmp_path / "rs"
    os.makedirs(rs / "bon_cap4096" / "analysis", exist_ok=True)
    os.makedirs(rs / "bon_siconst", exist_ok=True)
    cells = {f"{F.VIEW}/none": _cell(F.VIEW, "none", 0.8), f"{F.VIEW}/{F.LORA}": _cell(F.VIEW, F.LORA, 0.6)}
    if half_dose:
        # the same campaign standup at the HALF dose (the `_halfdose` spelling; lineage_common.split_view_legacy): the same fitted object
        # served at 0.5 and its own unsteered store
        cells[f"{F.VIEW}_halfdose/none"] = _cell(f"{F.VIEW}_halfdose", "none", 0.8)
        cells[f"{F.VIEW}_halfdose/{F.LORA}"] = _cell(f"{F.VIEW}_halfdose", F.LORA, 0.7)
    if paired:
        # the paired-write point: a steered store whose loop share is the object's (133 of 305 at-cap draws loop; the anchor's 1 of 2,728)
        cells[f"{PAIRED_KEY}/{F.PAIRED_WRITE}"] = _cell(PAIRED_KEY, F.PAIRED_WRITE, 0.5, n_at_cap=305, n_loop=133)
        cells[f"{PAIRED_KEY}/none"] = _cell(PAIRED_KEY, "none", 0.9, n_at_cap=273, n_loop=1)
        cells[f"{PAIRED_KEY_NO_MANIFEST}/none"] = _cell(PAIRED_KEY_NO_MANIFEST, "none", 0.9)
        cells[f"{PAIRED_KEY_UNREGISTERED}/none"] = _cell(PAIRED_KEY_UNREGISTERED, "none", 0.9)
        os.makedirs(rs / "bon_cap4096" / "cluster" / "manifests", exist_ok=True)
        regs = {F.VIEW: {"key": F.VIEW, "view": "main", "cluster": "cluster_b", "roots": [f"transcripts/sdfbase_full22_{F.VIEW}"]},
                PAIRED_KEY: {"key": F.VIEW, "view": POINT, "sha": "6" * 40, "cluster": "cluster_b", "roots": [ROOT_REL],
                             "grade_roots": [f"relu_paired/grades_paired_d1_s1/{F.VIEW}_paired_d1_s1"], "label": "synthetic paired difference, seed 1"},
                PAIRED_KEY_NO_MANIFEST: {"key": F.VIEW, "view": "paired_d1_s2", "cluster": "cluster_b",
                                         "roots": [f"relu_paired/transcripts_paired_d1_s2/sdfbase_full22_{F.VIEW}_paired_d1_s2"]}}
        json.dump(regs, open(rs / "bon_cap4096" / "cluster" / "manifests" / "lanes.json", "w"))
        # the campaign view's own store records on this host, as the 2026-09-07 wave wrote them: no sampler field at all (none was ever varied)
        for cid in ("none", F.LORA):
            d = rs / f"transcripts/sdfbase_full22_{F.VIEW}" / cid
            os.makedirs(d, exist_ok=True)
            json.dump({"engine": "vllm_served", "cell_id": cid, "bon_max_concurrency": 53, "units": {"n_work": 750}, "written_at": "2026-09-07T06:04:21Z"},
                      open(d / "bon_served_manifest.json", "w"))
        for cid, ident in ((F.PAIRED_WRITE, "proxy_vllm+lora+interleaved"), ("none", "proxy_vllm+passthrough+interleaved")):
            d = rs / ROOT_REL / cid
            os.makedirs(d, exist_ok=True)
            json.dump({"engine": "vllm_served", "serving_engine": "lora_native", "cell_id": cid, "decode": dict(F.VENDOR_DECODE), "bon_decode_preset": "vendor_preset",
                       "decode_policy_id": f"{F.MODEL_IT}|bon|off|vendor_preset@v1", "seed_rule": "bon_seed_v1", "written_at": "2026-09-19T06:29:20Z"},
                      open(d / "bon_served_manifest.json", "w"))
            json.dump({"cell_id": cid, "serving_engine": "lora_native", "serving_identity": ident, "n_goals": 10, "n_max": 75}, open(d / "bon_served_run.json", "w"))
    stats = {"schema": "bon_cap4096_stats/1", "cap_512": 512, "cap_4096": 4096, "n_max": 75, "judge": "judge-model-x / strongreject_v1",
             "classifier": {"name": "synthetic", "loop_rule": "rep_ratio > 0.5", "cap_ladder_sha256": "1" * 64, "degeneracy_sha256": "2" * 64}, "cells": cells}
    json.dump(stats, open(rs / "bon_cap4096" / "analysis" / "bon_cap4096_stats.json", "w"))
    deltas = {f"{F.VIEW}/{F.LORA}": {"delta": -20.0, "delta_lo": -30.0, "delta_hi": -10.0, "delta_p": 0.01, "n_goals": 10, "asr_cell": 0.5, "asr_unsteered": 0.7,
                                     "delta_method": "paired vs the view's unsteered cell at its rung max"},
              f"{F.VIEW}/none": {"delta": None, "at_cap_share": 0.27}}
    json.dump(deltas, open(rs / "bon_siconst" / "all_bon_deltas.json", "w"))
    return str(rs)


class _PairedObjects:
    """A stand-in for relu_paired's manifest join: the paired write resolves to the synthetic staged object, `none` to none."""

    def __init__(self):
        self.calls = []

    def resolve(self, model, seed, view_legacy, cid, dose):
        self.calls.append((model, seed, view_legacy, cid, dose))
        g = LC.cell_grammar_legacy(cid)
        if g["object_kind"] == "none":
            return LC.ResolvedObject("none", "none", "none", "baseline_none", "derived", "cell grammar: unsteered", derivation="none")
        return LC.ResolvedObject(F.SHA_PAIRED, g["object_kind"], g["paper_label"], g["object_role"], "recorded", "synthetic manifest", manifest_uri="synthetic://manifest",
                                 staged_id=f"paired_s{seed}__{cid}", derivation=g["derivation"], si_id=g["si_id"])


def _relu_paired_512_standins(root, seeds, obj_steered, now):
    """relu_paired's two 512 runs of the point (and ITS instrument rows), minted from the SAME identity inputs it files: the campaign, the store's
    directory name, the store's cell id, the eval and its own best-of-N instrument at 512 on the store's serving path."""
    jid = seeds.judge_id("strongreject_v1_haiku45")
    item_set = seeds.item_set_for(EVAL)
    recs, ids = [], {}
    for cid, sp, obj in ((F.PAIRED_WRITE, "lora_native", obj_steered), ("none", "passthrough", LC.ResolvedObject("none", "none", "none", "baseline_none", "derived", "unsteered", derivation="none"))):
        inst = RP.bon_instrument_fields(512, sp, obj.object_kind or "none", jid, EVAL, item_set)
        inst_rec = dict(inst, notes="stand-in instrument row (relu_paired's best-of-N at 512)", extra={"minted_by": "relu_paired (stand-in)"},
                        field_confidence_json={k: LC.unverified("stand-in") for k, v in inst.items() if v == "unverified"})
        recs.append(R.make_record("instrument", inst_rec, F.producer(), I.adapter_record_id(inst_rec, "instrument"), now))
        rid = I.run_id(LC.CAMPAIGN, PAIRED_KEY, cid, EVAL, str(inst["instrument_id"]))
        p = LC.run_payload_base(seeds, run_id=rid, producer_key=PAIRED_KEY, object_key=cid, view_legacy=F.VIEW, cell_id_legacy=cid, obj=obj, eval_id=EVAL,
                                instrument_row=inst, instrument_seed_id="relu_paired_bon_512", lifecycle_state="generated", when_utc=now, when_source="stand-in (test)",
                                token_cap_tokens=512, judge_id=jid, judge_seed_id="strongreject_v1_haiku45", substrate=LC.substrate_from_grades_root(None, root),
                                lane_source_legacy="relu_paired", store_uri=None, extra={"standin": True})
        p["view_legacy"] = POINT
        recs.append(R.make_record("run", p, F.producer(), I.adapter_record_id(p, "run"), now))
        ids[cid] = rid
    R.write_records(os.path.join(root, "records", "standins", "000003-relu-paired-bon.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return ids


def test_two_runs_per_cell_degeneration_and_contrast(tmp_path):
    rs = _files(tmp_path)
    seeds = F.synthetic_seeds()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    runs = F.q(root, "SELECT campaign_cell_id_legacy, token_cap_tokens, status_ruling_id, producer_run_key_legacy, lifecycle_state, instrument_id, instrument_match FROM run ORDER BY 1, 2")
    assert len(runs) == 4
    assert {(r[0], r[1]) for r in runs} == {("none", 512), ("none", 4096), (F.LORA, 512), (F.LORA, 4096)}
    assert all((r[2] == LC.supersede_instrument_ruling_id("bon_512_only_T1")) == (r[1] == 512) for r in runs)
    assert {r[3] for r in runs} == {LC.producer_key_for_r2_row(F.VIEW, "none", 1.0, None), LC.producer_key_for_r2_row(F.VIEW, F.LORA, 1.0, None)}
    assert all(r[4] == "judged" for r in runs)
    # a campaign view with no store record on this host: the grandfathered T = 1.0 instruments, the match unverified (nothing recorded to compare)
    assert {r[5] for r in runs} == {seeds.instrument_id("bon_512_only_T1", EVAL), seeds.instrument_id("bon_512store_4096climb_T1", EVAL)}
    assert all(r[6] == "unverified" for r in runs)
    dg = F.q(root, "SELECT measure, numerator, denominator, denominator_unit, aggregation, r.token_cap_tokens FROM run_degeneration d JOIN run r ON r.run_id = d.run_id "
                   "WHERE r.campaign_cell_id_legacy = ? ORDER BY 6, 1", (F.LORA,))
    assert ("at_token_cap_finish_reason", 200, 750, "draws", "single_detector", 512) in {tuple(r) for r in dg}
    assert ("rep_ratio_loop", 5, 750, "draws", "single_detector", 512) in {tuple(r) for r in dg}
    assert ("at_token_cap_finish_reason", 5, 195, "draws", "floor", 4096) in {tuple(r) for r in dg}
    m = F.q(root, "SELECT rm.metric_id, rm.threshold, rm.value, rm.extra_json FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.campaign_cell_id_legacy = ? AND metric_id = 'asr_bon_cap4096' AND threshold = 0.5", (F.LORA,))
    assert len(m) == 1 and m[0][2] == pytest.approx(0.5) and json.loads(m[0][3])["paired_vs_512"]["delta_pp"] == -10.0
    c = F.q(root, "SELECT rc.delta, rc.delta_p, rc.n_pairs, rc.anchor_kind, a.campaign_cell_id_legacy, a.token_cap_tokens FROM run_contrast rc "
                  "JOIN run a ON a.run_id = rc.anchor_run_id")
    assert c == [(-20.0, 0.01, 10, "same_standup_none", "none", 4096)]
    mids = {r[0] for r in F.q(root, "SELECT metric_id FROM run_metric")}
    assert {"at_cap_share_bon_512", "loop_share_bon_512", "coherent_at_cap_share_bon_512", "residual_at_cap_share_bon_4096"} <= mids
    assert ad.cluster_roots and all("bon_cap4096" in r for r in ad.cluster_roots)


def test_campaign_fitted_runs_carry_the_operating_point_stamp_at_the_campaign_nominal_only(tmp_path):
    """A campaign standup's FITTED-object runs (the 512 store and its climb alike) file `is_operating_point` 1 with `strength_unit`
    nominal_only at the campaign nominal 1.0 and 0 at the half dose; the unsteered store files no dose, no unit (the column default) and no
    operating point; no producer-filed operating-point id anywhere (the campaign spells none for these objects, as the export's rows of
    the same objects read). The stamp is `lineage_common.run_payload_base`'s; this reader's version 3 republished the live rows, read at
    version 2 from a tree without it (an unchanged version over unchanged inputs republishes nothing); version 4 (the registered points'
    own 512 runs) keeps every campaign record byte-identical; version 5 (the best-of-N curve rows) adds rows and changes none."""
    assert BS.BonStoresAdapter.EMISSION_VERSION == 5
    rs = _files(tmp_path, half_dose=True)
    seeds = F.synthetic_seeds()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, token_cap_tokens, strength_nominal, strength_nominal_source, strength_coefficient, "
                     "strength_unit, is_operating_point, operating_point_id_legacy, field_confidence_json FROM run WHERE lane_source_legacy = 'bon_stores'")
    assert len(rows) == 8
    by = {(r[0], r[1], r[2]): r for r in rows}
    for cap in (512, 4096):
        # the fitted object at the campaign nominal: the operating point, the unit nominal_only, its confidence entries named
        full = by[(F.VIEW, F.LORA, cap)]
        assert (full[3], full[4], full[5], full[6], full[7], full[8]) == (1.0, "filed", None, "nominal_only", 1, None)
        fc = json.loads(full[9])
        assert fc["is_operating_point"]["confidence"] == "derived" and "operating point" in fc["is_operating_point"]["source"]
        assert fc["strength_unit"]["confidence"] == "unverified" and fc["strength_nominal"]["confidence"] == "recorded"
        # the same object at the half dose: filed 0.5, NOT the operating point
        half = by[(f"{F.VIEW}_halfdose", F.LORA, cap)]
        assert (half[3], half[4], half[5], half[6], half[7], half[8]) == (0.5, "filed", None, "nominal_only", 0, None)
        # the unsteered store of either standup: no dose, the unit's column default, no operating point, no confidence entry for one
        for v in (F.VIEW, f"{F.VIEW}_halfdose"):
            none = by[(v, "none", cap)]
            assert (none[3], none[4], none[5], none[6], none[7], none[8]) == (None, None, None, "unverified", None, None)
            assert "is_operating_point" not in json.loads(none[9]) and "strength_unit" not in json.loads(none[9])
    # the version is folded into the published revision, so a rebuild republishes exactly these rows
    assert F.q(root, "SELECT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = 'bon_stores'") == [("5",)]


def test_registered_point_vendor_instruments_park_the_512_counts_on_the_climb_and_contrast_the_same_standup_none(tmp_path):
    """The paired-write point: the store records the vendor preset, so its climb run takes the vendor-preset climb instrument; the 512 reading
    is filed on NO run -- not on relu_paired's run of the same draws (nothing of this reader lands there) and not on a second 512 run row --
    its counts ride the climb run's extra beside relu_paired's run id (a cross-reference), and the rows not filed are counted; the climb run
    is new, at the point's dose, on the registration's cluster; the steered climb contrasts against the same-standup none with the fields the
    statistics file supports and none it does not; a registered point with no store record on this host and an unregistered group are
    refused by name."""
    rs = _files(tmp_path, paired=True)
    seeds = F.synthetic_seeds()
    now = R.utc_now()
    paired = _PairedObjects()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=paired)
    obj_steered = paired.resolve(F.VIEW, 1, POINT, F.PAIRED_WRITE, 1.0)
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), obj_steered])
    relu_ids = _relu_paired_512_standins(root, seeds, obj_steered, now)
    # the store records are inputs of the read: the registrations and the two stores' manifest + run records
    uris = {os.path.relpath(i.uri, rs) for i in ad.discover()}
    assert BS.REGISTRATIONS in uris and f"{ROOT_REL}/{F.PAIRED_WRITE}/{BS.STORE_MANIFEST}" in uris and f"{ROOT_REL}/none/{BS.STORE_RUN}" in uris
    assert f"transcripts/sdfbase_full22_{F.VIEW}/none/{BS.STORE_MANIFEST}" in uris
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert ing.blocked + ing2.blocked == 0, "every reading names a run this revision emits"
    # -- the refusals, by name: no sampler recorded on this host for a registered point; a group registered nowhere
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert any(k.startswith(PAIRED_KEY_NO_MANIFEST) and BS.STORE_MANIFEST in v and "never defaulted" in v for k, v in why), why
    assert any(k == PAIRED_KEY_UNREGISTERED and "registered" in v for k, v in why), why
    assert not any(k.startswith(PAIRED_KEY + "/") for k, _ in why), why
    assert not F.q(root, "SELECT run_id FROM run WHERE view_legacy IN ('paired_d1_s2', 'paired_d1_s3')")
    # -- the instrument per group: the campaign view keeps T1 (both caps), the registered point takes the vendor-preset climb only
    t1 = {seeds.instrument_id("bon_512_only_T1", EVAL), seeds.instrument_id("bon_512store_4096climb_T1", EVAL)}
    assert {r[0] for r in F.q(root, "SELECT instrument_id FROM run WHERE view_legacy = ? AND lane_source_legacy = 'bon_stores'", (F.VIEW,))} == t1
    climb_inst = seeds.instrument_id("bon_512store_4096climb_vendor_preset", EVAL)
    own = F.q(root, "SELECT run_id, campaign_cell_id_legacy, object_id, instrument_id, token_cap_tokens, producer_run_key_legacy, producer_object_key_legacy, "
                    "strength_nominal, strength_unit, is_operating_point, substrate, model_id, lifecycle_state, instrument_match, decode_sent_json, status_ruling_id, extra_json "
                    "FROM run WHERE view_legacy = ? AND lane_source_legacy = 'bon_stores' ORDER BY 2", (POINT,))
    assert len(own) == 2 and {r[4] for r in own} == {4096}, own        # the climb runs only: no second 512 run of the same draws is filed
    assert not F.q(root, "SELECT 1 FROM run WHERE instrument_id = ?", (seeds.instrument_id("bon_512_only_vendor_preset", EVAL),))
    steered = next(r for r in own if r[2] == F.SHA_PAIRED)
    anchor = next(r for r in own if r[2] == "none")
    assert steered[1] == f"paired_s1__{F.PAIRED_WRITE}" and anchor[1] == "none"
    assert steered[3] == anchor[3] == climb_inst and steered[5] == anchor[5] == PAIRED_KEY
    assert steered[6] == F.PAIRED_WRITE and anchor[6] == "none"
    assert (steered[7], steered[8], steered[9]) == (1.0, "nominal_only", 1) and anchor[7] is None
    assert steered[10] == anchor[10] == "cluster_b" and steered[11] == F.MODEL_IT and steered[12] == anchor[12] == "judged"
    assert steered[13] == "match" and json.loads(steered[14]) == F.VENDOR_DECODE
    assert steered[15] is None and anchor[15] is None        # the climb is the canonical reading; nothing here is superseded
    ex = json.loads(steered[16])
    assert ex["instrument_seed_id"] == "bon_512store_4096climb_vendor_preset" and ex["registration_legacy"]["cluster"] == "cluster_b" and ex["fit_seed"] == 1
    assert ex["sampler"]["decode_policy_id"] == f"{F.MODEL_IT}|bon|off|vendor_preset@v1"
    # -- the 512 store's counts ride the climb run's extra, beside relu_paired's run id for the same draws (a cross-reference, not a parent)
    assert ex["relu_paired_512_run_id"] == relu_ids[F.PAIRED_WRITE] and json.loads(anchor[16])["relu_paired_512_run_id"] == relu_ids["none"]
    assert ex["store_512_counts"] == {"n_rows_512": 750, "n_at_cap_512": 305, "n_loop_rep_ratio_512": 133, "n_coherent_512": 172, "n_secondary_flag_only_512": 2}
    assert json.loads(anchor[16])["store_512_counts"] == {"n_rows_512": 750, "n_at_cap_512": 273, "n_loop_rep_ratio_512": 1, "n_coherent_512": 272, "n_secondary_flag_only_512": 2}
    assert paired.calls and all(c[0] == F.VIEW and c[1] == 1 and c[2] == POINT and c[4] == 1.0 for c in paired.calls)
    # -- nothing of this reader lands on relu_paired's runs; no 512-store row of the point is filed anywhere
    for cid in (F.PAIRED_WRITE, "none"):
        relu = relu_ids[cid]
        assert F.q(root, "SELECT lane_source_legacy FROM run WHERE run_id = ?", (relu,))[0][0] == "relu_paired"
        assert not F.q(root, "SELECT 1 FROM run_degeneration WHERE run_id = ?", (relu,)) and not F.q(root, "SELECT 1 FROM run_metric WHERE run_id = ?", (relu,))
    assert not F.q(root, "SELECT 1 FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.view_legacy = ? AND rm.metric_id LIKE '%_bon_512'", (POINT,))
    # the campaign stores' 512 runs still carry theirs (the loop share per run)
    assert F.q(root, "SELECT count(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.view_legacy = ? AND rm.metric_id = 'loop_share_bon_512'", (F.VIEW,)) == [(2,)]
    # -- the climb runs carry the counterfactual metric (self-contrast parked) and the residual degeneration
    m = F.q(root, "SELECT threshold, value, extra_json FROM run_metric WHERE run_id = ? AND metric_id = 'asr_bon_cap4096' ORDER BY 1", (steered[0],))
    assert [(r[0], r[1]) for r in m] == [(0.5, pytest.approx(0.4)), (0.75, pytest.approx(0.3))] and json.loads(m[0][2])["paired_vs_512"]["delta_pp"] == -10.0
    assert F.q(root, "SELECT measure, numerator, denominator, aggregation FROM run_degeneration WHERE run_id = ?", (steered[0],)) == [("at_token_cap_finish_reason", 5, 172, "floor")]
    # -- the same-standup contrast, read off the statistics file: pp difference over the shared goals, no interval / p / effect (never invented)
    c = F.q(root, "SELECT threshold, delta, delta_lo, delta_hi, delta_p, effect_size, n_pairs, anchor_kind, computed_by, delta_method, anchor_judgment_id != '' "
                  "FROM run_contrast WHERE run_id = ? AND anchor_run_id = ? ORDER BY 1", (steered[0], anchor[0]))
    assert len(c) == 2 and c[0][0] == 0.5 and c[0][1] == pytest.approx(100 * (0.4 - 0.8)) and c[1][0] == 0.75 and c[1][1] == pytest.approx(0.0)
    assert all(r[2] is None and r[3] is None and r[4] is None and r[5] is None and r[6] == 10 and r[7] == "same_standup_none" and r[8] == "registry" and r[10] == 1 for r in c)
    assert "no bootstrap interval, no McNemar" in c[0][9] and "asr_4096_cf_t0.5" in c[0][9]
    assert not F.q(root, "SELECT 1 FROM run_contrast WHERE run_id = ?", (anchor[0],))
    # -- the facts: the drop counted and named, the cross-references, the self-check
    assert ad.stats["registered_points"] == 1 and ad.stats["relu_paired_512_runs_referenced"] == 2 and ad.stats["contrasts"] == 3
    assert ad.stats["degeneration_rows_dropped_unemitted_runs"] == 4 and ad.stats["metric_rows_dropped_unemitted_runs"] == 8
    dropped = {d["key"]: d for d in ad.stats["readings_dropped_unemitted_runs"]}
    assert set(dropped) == {f"{PAIRED_KEY}/{F.PAIRED_WRITE}", f"{PAIRED_KEY}/none"}
    assert dropped[f"{PAIRED_KEY}/{F.PAIRED_WRITE}"] == {"key": f"{PAIRED_KEY}/{F.PAIRED_WRITE}", "would_attach_to": relu_ids[F.PAIRED_WRITE], "parked_on": steered[0],
                                                        "degeneration_rows": 2, "metric_rows": 4}
    assert ad.stats["run_child_records_orphaned"] == 0 and ad.stats["run_child_records_checked"] > 0
    # the status row carries the facts and the refusals by name
    st = F.q(root, "SELECT extra_json, detail FROM source_status WHERE adapter = 'bon_stores'")
    fx = json.loads(st[0][0])
    assert fx["relu_paired_512_runs_referenced"] == 2 and fx["degeneration_rows_dropped_unemitted_runs"] == 4 and len(fx["readings_dropped_unemitted_runs"]) == 2
    assert any(PAIRED_KEY_UNREGISTERED in s for s in fx["row_refusals"])


def _emitted_run_ids_and_children(root: str):
    """The run ids the PUBLISHED bon_stores revision emits and every run-child record's (record type, run_id, anchor_run_id) beside them."""
    import glob
    import gzip
    files = sorted(glob.glob(os.path.join(root, "records", "adapters", "bon_stores", "[0-9a-f]" * 16, "*.jsonl*")))
    assert files, "the revision published"
    runs, children = set(), []
    for fn in files:
        op = gzip.open if fn.endswith(".gz") else open
        with op(fn, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                p = rec["payload"]
                if rec["record_type"] == "run":
                    runs.add(p["run_id"])
                elif rec["record_type"] in BS.BonStoresAdapter.RUN_CHILD_TYPES:
                    children.append((rec["record_type"], p.get("run_id"), p.get("anchor_run_id")))
    return runs, children


def test_registered_point_with_no_relu_paired_run_in_the_registry_activates_and_names_no_run_it_does_not_emit(tmp_path):
    """The live defect's shape (2026-09-22): a registered point whose relu_paired 512 run does NOT exist in the registry -- relu_paired files
    none for a point whose stores it never graded. Version 2 attached the 512 counts to that run id and the harvester held the whole revision
    (`activation pending`), the point's paper-facing climb row included. Version 4: the revision ACTIVATES with no relu_paired record present,
    the 512 counts ride the climb run's extra (the would-be parent named there, a run row nobody filed), and every degeneration / metric /
    judgment / state / contrast record of the published revision names a run the revision itself emits."""
    rs = _files(tmp_path, paired=True)
    seeds = F.synthetic_seeds()
    paired = _PairedObjects()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=paired)
    obj_steered = paired.resolve(F.VIEW, 1, POINT, F.PAIRED_WRITE, 1.0)
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), obj_steered])   # NO relu_paired stand-ins
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing.blocked, ing2.blocked)
    assert F.q(root, "SELECT accepted, refusal FROM source_revision WHERE adapter = 'bon_stores'") == [(1, None)]
    assert F.q(root, "SELECT count(*) FROM blocked_records") == [(0,)]
    assert not F.q(root, "SELECT 1 FROM run WHERE lane_source_legacy = 'relu_paired'")
    own = F.q(root, "SELECT run_id, token_cap_tokens, status_ruling_id, extra_json, object_id FROM run WHERE view_legacy = ? AND lane_source_legacy = 'bon_stores' ORDER BY 5", (POINT,))
    assert len(own) == 2 and [r[1] for r in own] == [4096, 4096] and all(r[2] is None for r in own)      # the climb runs only
    # the 512 counts ride the climb runs, with relu_paired's would-be id as the cross-reference (a run row nobody filed) and no row on it
    for r in own:
        ex = json.loads(r[3])
        ref = ex["relu_paired_512_run_id"]
        assert ref.startswith(LC.CAMPAIGN + ":" + EVAL + ":") and ref != r[0]
        assert not F.q(root, "SELECT 1 FROM run WHERE run_id = ?", (ref,)) and not F.q(root, "SELECT 1 FROM run_degeneration WHERE run_id = ?", (ref,))
        assert ex["store_512_counts"]["n_rows_512"] == 750 and ex["store_512_counts"]["n_at_cap_512"] in (305, 273)
        assert F.q(root, "SELECT count(*) FROM run_degeneration WHERE run_id = ?", (r[0],)) == [(1,)]       # the climb's residual only
    assert ad.stats["degeneration_rows_dropped_unemitted_runs"] == 4 and ad.stats["metric_rows_dropped_unemitted_runs"] == 8
    assert {d["parked_on"] for d in ad.stats["readings_dropped_unemitted_runs"]} == {r[0] for r in own}
    # the paper-facing climb value of the point is in the frame, with its same-standup contrast
    steered = F.q(root, "SELECT rm.value FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.view_legacy = ? AND r.object_id = ? AND r.token_cap_tokens = 4096 "
                        "AND rm.metric_id = 'asr_bon_cap4096' AND rm.threshold = 0.5", (POINT, F.SHA_PAIRED))
    assert steered == [(pytest.approx(0.4),)]
    assert F.q(root, "SELECT count(*) FROM run_contrast rc JOIN run r ON r.run_id = rc.run_id WHERE r.view_legacy = ?", (POINT,)) == [(2,)]
    # the published revision's own record files: every run-child names an emitted run
    runs, children = _emitted_run_ids_and_children(root)
    assert children and all(rid in runs and (anchor is None or anchor in runs) for _, rid, anchor in children), \
        [c for c in children if c[1] not in runs or (c[2] is not None and c[2] not in runs)]
    assert ad.stats["run_child_records_checked"] == len(children) and ad.stats["run_child_records_orphaned"] == 0


class _LeakyAdapter(BS.BonStoresAdapter):
    """A reader that files one degeneration row on a run id nobody emits (the version-2 shape, synthesised)."""
    FOREIGN = f"{LC.CAMPAIGN}:{EVAL}:000000000000:0000000000000000"

    def _emit_512(self, batch, rid, cell, key, src, when, classifier, producer, now):
        d, m = super()._emit_512(batch, rid, cell, key, src, when, classifier, producer, now)
        if not key.endswith("/none"):
            return d, m
        last = next(rec for rec in reversed(batch.records) if rec["record_type"] == "run_degeneration")
        batch.add("run_degeneration", dict(last["payload"], run_id=self.FOREIGN), producer, now)
        return d + 1, m


def test_a_batch_naming_a_run_it_does_not_emit_is_refused_by_name_and_publishes_nothing(tmp_path):
    """The self-check: a degeneration row on a run this revision does not emit refuses the WHOLE read by name (the offending record type,
    the run id, its source), the unaccepted marker carries the refusal, and no record of the read reaches the log -- never a published
    revision the harvester holds forever."""
    rs = _files(tmp_path)
    seeds = F.synthetic_seeds()
    ad = _LeakyAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 1 and rep.groups_refreshed == 0, [r.error for r in rep.results]
    err = next(r.error for r in rep.results if r.error)
    assert "does not emit" in err and _LeakyAdapter.FOREIGN in err and "run_degeneration.run_id" in err and "bon_cap4096_stats.json" in err, err
    assert ad.stats["run_child_records_orphaned"] == 1
    assert not F.q(root, "SELECT 1 FROM run WHERE lane_source_legacy = 'bon_stores'") and not F.q(root, "SELECT 1 FROM run_degeneration")
    marker = F.q(root, "SELECT accepted, refusal FROM source_revision WHERE adapter = 'bon_stores'")
    assert marker and marker[0][0] == 0 and "does not emit" in marker[0][1] and _LeakyAdapter.FOREIGN in marker[0][1], marker
    import glob
    assert not glob.glob(os.path.join(root, "records", "adapters", "bon_stores", "[0-9a-f]" * 16))     # nothing published


def test_a_deltas_key_the_statistics_file_carries_no_store_for_is_refused_by_name_not_a_contrast_on_an_unemitted_run(tmp_path):
    """The deltas file's vs-unsteered contrast attaches to the climb run the statistics file filed; a key with no such store in this read
    (the two files disagreeing on their stores) is a row refusal by name, and the read still publishes -- a contrast on a run nobody emits
    would hold the revision."""
    rs = _files(tmp_path)
    dp = os.path.join(rs, BS.DELTAS)
    deltas = json.load(open(dp))
    deltas[f"{F.VIEW}/{F.RELU}"] = {"delta": -5.0, "delta_lo": -9.0, "delta_hi": -1.0, "delta_p": 0.2, "n_goals": 10, "asr_cell": 0.6, "asr_unsteered": 0.65}
    json.dump(deltas, open(dp, "w"))
    seeds = F.synthetic_seeds()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused + ing.blocked + ing2.blocked == 0
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert any(k == f"{F.VIEW}/{F.RELU}|contrast" and "deltas file names a store" in v and "no run of this revision" in v for k, v in why), why
    assert not any(k == f"{F.VIEW}/{F.RELU}" for k, _ in why), why      # the STORE key is not refused -- the deltas file's contrast is
    assert ad.stats["deltas_keys_refused"] == 1 and ad.stats["contrasts"] == 1
    assert F.q(root, "SELECT count(*) FROM run_contrast") == [(1,)]
    runs, children = _emitted_run_ids_and_children(root)
    assert all(rid in runs and (anchor is None or anchor in runs) for _, rid, anchor in children)


def test_registered_point_whose_store_records_a_sampler_the_instrument_disagrees_with_is_refused(tmp_path):
    rs = _files(tmp_path, paired=True)
    mp = os.path.join(rs, ROOT_REL, F.PAIRED_WRITE, BS.STORE_MANIFEST)
    man = json.load(open(mp))
    man["decode"]["temperature"] = 1.0                      # the store says T = 1.0 under a vendor_preset policy id: not this instrument's numbers
    json.dump(man, open(mp, "w"))
    seeds = F.synthetic_seeds()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=_PairedObjects())
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert any(k == f"{PAIRED_KEY}/{F.PAIRED_WRITE}" and "temperature 1.0 differs from instrument" in v for k, v in why), why
    # a preset spelling at odds with the policy id is undecidable, not a guess
    man["decode"]["temperature"] = 0.7
    man["bon_decode_preset"] = "harness_default"
    json.dump(man, open(mp, "w"))
    ad2 = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=_PairedObjects())
    list(ad2.read(F.RUN.AdapterContext(root=root, now=R.utc_now(), producer=F.producer()), "bon", ad2.discover()))
    why2 = [(it["key"], it["why"]) for it in ad2.refusals.items]
    assert any(k == f"{PAIRED_KEY}/{F.PAIRED_WRITE}" and "disagrees with decode_policy_id" in v for k, v in why2), why2


def test_relu_paired_best_of_n_identity_helper_reproduces_the_filed_instrument_ids():
    """The identity relu_paired's emitter mints for its best-of-N stores, rebuilt standalone over the repo's seeds: the two 512 instruments the
    registry holds for the paired-write points (steered on lora_native, the anchor on passthrough), and the T1 seed rows beside them."""
    seeds = LC.Seeds.locate()
    jid = seeds.judge_id("strongreject_v1_haiku45")
    item_set = seeds.item_set_for(EVAL)
    assert item_set == "jbb_goals_100"
    assert RP.bon_instrument_id(512, "lora_native", "composed", jid, EVAL, item_set) == "e54858d6e65641235fd7b32fa115d98801d177c135a5a230dd2468cf008fdb78"
    assert RP.bon_instrument_id(512, "passthrough", "none", jid, EVAL, item_set) == "e24fab69ed3c64962ead703c9456603302a07bcf8c77cdd7a56392df09313261"
    assert seeds.instrument_id("bon_512_only_T1", EVAL, "qwen35_4b_it") == "b97847bd49db6b458cf61a8486ccb889d4cf3e0b70bc6518e02ddb87e1975a86"
    assert seeds.instrument_id("bon_512store_4096climb_T1", EVAL, "qwen35_4b_it") == "78b4699fb41a79a40fc9029f72c9a1bf4b711a8e4fa28aa9671559a24eb38c57"
    # the emitter's own block is the helper's (a pure extraction): the same 14 keys and values relu_paired files
    decl = RP.bon_instrument_declared(512, "lora_native", "proxy_vllm+lora+interleaved")
    assert decl["instrument_config_hash"] == "relu_paired_bon_512" and decl["serving_identity"] == "proxy_vllm+lora+interleaved" and decl["bon_n"] == RP.BON_N
    # the run id of the registered point is the SAME the registry holds
    assert I.run_id(LC.CAMPAIGN, "qwen35_4b_paired_d1_s1", "constv2-mix-full__psr_ce_all_minus_ctl", EVAL, RP.bon_instrument_id(512, "lora_native", "composed", jid, EVAL, item_set)) \
        == "sdfbase_full22:strongreject_bon:e54858d6e656:a5226a79f9034fcd"


def test_vendor_preset_seed_rows_expand_per_model_from_the_policy_and_the_floor_ruling_covers_them():
    """The two vendor-preset seeds mirror the T1 rows except the decode fields, which come from each served model's `<model>|bon|off|vendor_preset`
    row; the 512-floor supersede ruling is matched on (eval strongreject_bon, token cap 512), so it covers the new 512-only instrument as it
    covers the T1 one (the live view's match form, `v_paper_current` covered)."""
    seeds = LC.Seeds.locate()
    q_climb = seeds.instrument_row("bon_512store_4096climb_vendor_preset", EVAL, "qwen35_4b_it")
    q_512 = seeds.instrument_row("bon_512_only_vendor_preset", EVAL, "qwen35_4b_it")
    t_climb = seeds.instrument_row("bon_512store_4096climb_T1", EVAL, "qwen35_4b_it")
    for row in (q_climb, q_512):
        assert (row["decode_temperature"], row["decode_top_p"], row["decode_top_k"], row["decode_min_p"], row["decode_repetition_penalty"], row["decode_presence_penalty"]) == (0.7, 0.8, 20, 0.0, 1.0, 1.5)
        assert row["decode_source"] == "model_card" and row["decode_seed_rule"] == "per_draw" and row["bon_n"] == 75 and row["judge_id"] == t_climb["judge_id"]
    assert (q_climb["token_cap_tokens"], json.loads(q_climb["token_cap_ladder_json"]) if isinstance(q_climb["token_cap_ladder_json"], str) else q_climb["token_cap_ladder_json"], q_climb["read_at"], q_climb["serving_path"], q_climb["applied_positions_mode"]) \
        == (4096, [512, 4096], "ladder_final", "lora_native", "all")
    assert (q_512["token_cap_tokens"], q_512["read_at"]) == (512, "fixed")
    assert q_climb["instrument_id"] != t_climb["instrument_id"] and q_512["instrument_id"] != q_climb["instrument_id"]
    g_climb = seeds.instrument_row("bon_512store_4096climb_vendor_preset", EVAL, "gemma4e4b_it")
    assert g_climb["instrument_id"] != q_climb["instrument_id"] and g_climb["decode_source"] == "generation_config"
    assert BS.INSTRUMENTS_BY_DECODE_MODE["vendor_preset"] == ("bon_512_only_vendor_preset", "bon_512store_4096climb_vendor_preset")
    assert BS.CLIMB_OF["bon_512_only_vendor_preset"] == "bon_512store_4096climb_vendor_preset"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "configs", "registry", "rulings_r2_v1.jsonl"), encoding="utf-8") as fh:
        floor = next(r for r in (json.loads(line) for line in fh if line.strip()) if r.get("ruling_id") == "r2v1-supersede-instrument-bon-512-floor")
    match = floor["scope_json"]["match"]
    assert match == {"eval_ids": ["strongreject_bon"], "token_caps": [512]}       # instrument-agnostic: every 512 best-of-N run, T1 or vendor preset
    assert floor["action"] == "supersede"


def test_paired_object_resolver_uses_relu_paired_point_keyed_manifest_join(tmp_path):
    md = tmp_path / "manifests"
    os.makedirs(md)
    stem = F.PAIRED_WRITE
    sha = F.SHA_PAIRED
    json.dump({"schema": "relu_paired_manifest/1", "view": POINT, "seed": 1, "dose_scale": 1.0, "dose_unit": "psr_direction_norm_multiplier",
               "objects": {f"paired_s1__{stem}": {"cell_id": stem, "view": POINT, "seed": 1, "dose_scale": 1.0, "object_sha256": sha}}},
              open(md / "MANIFEST_paired_d1_s1.json", "w"))
    res = BS.PairedObjectResolver(str(md))
    obj = res.resolve("qwen35_4b", 1, POINT, stem, 1.0)
    assert (obj.object_id, obj.object_kind, obj.paper_label, obj.object_role, obj.staged_id, obj.confidence) == (sha, "composed", "difference_relu", "difference", f"paired_s1__{stem}", "recorded")
    assert obj.manifest_uri.endswith("MANIFEST_paired_d1_s1.json")
    assert res.resolve("qwen35_4b", 1, POINT, "none", 1.0).object_id == "none"
    missing = res.resolve("qwen35_4b", 2, "paired_d1_s2", stem, 1.0)
    assert missing.object_id is None and "never guessed" in missing.source
    assert res.resolve("qwen35_4b", 1, "not_a_point", stem, 1.0).object_id is None


def test_a_manifest_silent_on_the_sampler_is_the_grandfathered_body_only_for_a_campaign_store_written_before_the_vendor_rows(tmp_path):
    """A campaign view's store record written before 2026-09-17 names no sampler because none was varied: the grandfathered T = 1.0 family, decided
    from the recorded write date; the same silence on a registered point, or on a campaign store written after the vendor rows, is a refusal."""
    rs = _files(tmp_path, paired=True)
    seeds = F.synthetic_seeds()
    # the registered point's steered store loses its sampler fields
    mp = os.path.join(rs, ROOT_REL, F.PAIRED_WRITE, BS.STORE_MANIFEST)
    man = json.load(open(mp))
    for k in ("decode", "decode_policy_id", "bon_decode_preset"):
        man.pop(k)
    json.dump(man, open(mp, "w"))
    # the campaign view's none store is re-written after the vendor rows, still silent
    cp = os.path.join(rs, f"transcripts/sdfbase_full22_{F.VIEW}", "none", BS.STORE_MANIFEST)
    cman = json.load(open(cp))
    cman["written_at"] = "2026-09-18T00:00:00Z"
    json.dump(cman, open(cp, "w"))
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=_PairedObjects())
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    list(ad.read(F.RUN.AdapterContext(root=root, now=R.utc_now(), producer=F.producer()), "bon", ad.discover()))
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert any(k == f"{PAIRED_KEY}/{F.PAIRED_WRITE}" and "nor a decode block names a sampler" in v and "not a campaign view" in v for k, v in why), why
    assert any(k == f"{F.VIEW}/none" and "not before 2026-09-17" in v for k, v in why), why
    assert not any(k == f"{F.VIEW}/{F.LORA}" for k, _ in why), why          # the silent 2026-09-07 manifest: grandfathered, no refusal
    st = ad._standup(F.VIEW, "")
    s_lora = ad._sampler(st, F.LORA, F.MODEL_IT, EVAL)
    assert (s_lora.mode, s_lora.confidence, s_lora.sent) == (BS.GRANDFATHERED_MODE, "derived", {}) and "predates the vendor-preset" in s_lora.source
    # a manifest that records the numbers but names no policy: the numbers decide
    json.dump(dict(cman, decode=dict(F.VENDOR_DECODE), written_at="2026-09-19T00:00:00Z"), open(cp, "w"))
    s_none = ad._sampler(st, "none", F.MODEL_IT, EVAL)
    assert (s_none.mode, s_none.confidence) == ("vendor_preset", "recorded") and s_none.sent == F.VENDOR_DECODE
    json.dump(dict(cman, decode={"temperature": 1.0}), open(cp, "w"))
    assert ad._sampler(st, "none", F.MODEL_IT, EVAL).mode == BS.GRANDFATHERED_MODE
    json.dump(dict(cman, decode={"temperature": 0.3}), open(cp, "w"))
    s_bad = ad._sampler(st, "none", F.MODEL_IT, EVAL)
    assert s_bad.mode is None and "matches no best-of-N instrument family" in s_bad.source


# --------------------------------------------------------------------------------------------
# the best-of-N curve (version 5)
# --------------------------------------------------------------------------------------------

CURVE_NS = (1, 2, 4, 8, 16, 32, 64)


def _curve_rows(root: str, rid: str):
    rows = F.q(root, "SELECT metric_id, threshold, value, k, n, ci_lo, ci_hi, ci_method, judgment_id, computation_id, computed_by, producer_metric_spelling, "
                     "source_uri, extra_json FROM run_metric WHERE run_id = ? AND metric_id LIKE 'asr_at_n%'", (rid,))
    return {r[0]: r for r in rows}


def test_the_best_of_n_curve_rides_every_climb_run_and_no_512_run(tmp_path):
    """Every climb run this reader files -- a campaign standup's and a registered point's alike -- carries seven `asr_at_n<N>` rows (N in 1..64) at
    the headline threshold on the climb's judgment: the value the statistics file's `asr_at_n_4096_cf_t0.5[N]`, k = p x n (exact here), n the
    headline block's goals, no interval, the producer's spelling, `extra.bon_n` = N (the paper's `bon_n`) with the grid's schedule and the N not
    filed (75, the headline); the grid's 75 is not filed as `asr_at_n75`; a 512 run carries no curve; the facts count the rows; the spelling is
    exactly the seven ids `paper.METRIC_LEGACY` folds to the paper's `asr` -- seedrep_cluster's and bon_native4096's constants."""
    rs = _files(tmp_path, paired=True)
    seeds = F.synthetic_seeds()
    paired = _PairedObjects()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=paired)
    obj_steered = paired.resolve(F.VIEW, 1, POINT, F.PAIRED_WRITE, 1.0)
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), obj_steered])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused + ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    assert not any(k.endswith("|curve") for k, _ in ((it["key"], it["why"]) for it in ad.refusals.items)), ad.refusals.items
    climbs = F.q(root, "SELECT run_id, campaign_cell_id_legacy, view_legacy, producer_run_key_legacy FROM run WHERE lane_source_legacy = 'bon_stores' AND token_cap_tokens = 4096")
    assert len(climbs) == 4 and {(r[2], r[1]) for r in climbs} == {(F.VIEW, "none"), (F.VIEW, F.LORA), (POINT, "none"), (POINT, f"paired_s1__{F.PAIRED_WRITE}")}
    for rid, cid, view, pk in climbs:
        by = _curve_rows(root, rid)
        assert set(by) == {f"asr_at_n{N}" for N in CURVE_NS}, (view, cid, sorted(by))       # the seven N; never asr_at_n75
        head_jid = F.q(root, "SELECT judgment_id FROM run_metric WHERE run_id = ? AND metric_id = 'asr_bon_cap4096' AND threshold = 0.5", (rid,))[0][0]
        assert head_jid != ""
        key = (F.VIEW if view == F.VIEW else PAIRED_KEY) + "/" + ("none" if cid == "none" else (F.LORA if view == F.VIEW else F.PAIRED_WRITE))
        for N in CURVE_NS:
            mid, t, value, k, n, lo, hi, cim, jid, comp, by_whom, spelling, src, ex = by[f"asr_at_n{N}"]
            assert (t, n, lo, hi, cim, jid, comp, by_whom) == (0.5, 10, None, None, None, head_jid, "", "producer")
            assert k == round(value * 10) and 0.0 <= value <= 1.0                                    # k = p x n, exact over 10 goals
            assert spelling == f"asr_at_n_4096_cf_t0.5[{N}]" and src.endswith(f"bon_cap4096_stats.json \u203acells.{key}.asr_at_n_4096_cf_t0.5.{N}"), (spelling, src)
            ex = json.loads(ex)
            assert ex["bon_n"] == N and ex["bon_n_max"] == 75 and ex["schedule"] == list(BON_GRID_NS) and ex["not_filed"] == [75] and "no interval" in ex["read_from"]
    # the values: the campaign lora climb's grid (asr 0.6 -> the counterfactual 0.5 at 75)
    lora_rid = next(r[0] for r in climbs if (r[2], r[1]) == (F.VIEW, F.LORA))
    assert {N: _curve_rows(root, lora_rid)[f"asr_at_n{N}"][2] for N in CURVE_NS} == {1: 0.1, 2: 0.1, 4: 0.2, 8: 0.2, 16: 0.3, 32: 0.4, 64: 0.5}
    assert {N: _curve_rows(root, lora_rid)[f"asr_at_n{N}"][3] for N in CURVE_NS} == {1: 1, 2: 1, 4: 2, 8: 2, 16: 3, 32: 4, 64: 5}
    # the curve is monotone up to the headline
    head = F.q(root, "SELECT value FROM run_metric WHERE run_id = ? AND metric_id = 'asr_bon_cap4096' AND threshold = 0.5", (lora_rid,))[0][0]
    vals = [_curve_rows(root, lora_rid)[f"asr_at_n{N}"][2] for N in CURVE_NS]
    assert vals == sorted(vals) and vals[-1] <= head
    # no curve on a 512 run; the headline rows are untouched (two thresholds per climb)
    assert not F.q(root, "SELECT 1 FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.token_cap_tokens = 512 AND rm.metric_id LIKE 'asr_at_n%'")
    assert F.q(root, "SELECT count(*) FROM run_metric WHERE metric_id = 'asr_bon_cap4096'") == [(8,)]
    # the facts
    assert ad.stats["curve_rows"] == 28 and ad.stats["curve_grids_absent"] == [] and ad.stats["metrics"] >= 28 + 8
    fx = json.loads(F.q(root, "SELECT extra_json FROM source_status WHERE adapter = 'bon_stores'")[0][0])
    assert fx["curve_rows"] == 28 and fx["curve_grids_absent"] == []
    # the spelling: the constants are seedrep_cluster's (bon_native4096 spells the same), and the seven ids are exactly the paper's `asr` at bon_n
    assert BS.BON_CURVE_METRIC is SC.BON_CURVE_METRIC and BS.BON_CURVE_METRIC == BN.CURVE_METRIC == "asr_at_n{n}"
    assert BS.BON_CURVE_THRESHOLD == SC.HEADLINE_THRESHOLD == BN.HEADLINE_THRESHOLD == 0.5
    # the curve ids map to `asr`; beside them the one legacy best-of-N spelling the emit-time guard refuses for new records
    # (`asr_at_n_max`, vocab.BON_LEGACY_METRIC_SPELLINGS) maps the same way for the 4 live rows
    assert {BS.BON_CURVE_METRIC.format(n=N) for N in BS.BON_CURVE_NS} | set(V.BON_LEGACY_METRIC_SPELLINGS) == set(P.METRIC_LEGACY) and set(P.METRIC_LEGACY.values()) == {"asr"}
    assert 75 not in BS.BON_CURVE_NS


def test_a_bon_stores_curve_row_and_a_seedrep_cluster_curve_row_carry_the_same_keys_and_spelling(tmp_path):
    """The paper's curve figure reads one shape: a bon_stores `asr_at_n<N>` row and a seedrep_cluster one for the same N carry the same payload
    keys, the same metric id and threshold, a blank computation id and `extra.bon_n` = N -- measured on both readers' synthetic stores, not on a
    copied key list."""
    import test_registry_adapter_seedrep_cluster as TSC
    rs = _files(tmp_path)
    seeds = F.synthetic_seeds()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    recs = [r for b in ad.read(F.RUN.AdapterContext(root=root, now=R.utc_now(), producer=F.producer()), "bon", ad.discover()) for r in b.records]
    ours = {p["metric_id"]: p for p in (r["payload"] for r in recs if r["record_type"] == "run_metric") if p["metric_id"].startswith("asr_at_n")}
    sc_dir = tmp_path / "sc"
    os.makedirs(sc_dir)
    fx = TSC._mirror(sc_dir, mmlu=False)
    TSC._bon_store(fx["root"], TSC.TAG, {"none": 66, TSC.LORA: 20, TSC.PROMPTED: 18})
    bt, facts = TSC._read(TSC._adapter(fx, LC.Seeds.locate()))
    assert facts["n_refusals"] == 0, facts["refusals"]
    theirs = {m["metric_id"]: m for m in bt["run_metric"] if m["metric_id"].startswith("asr_at_n")}
    assert set(ours) == set(theirs) == {f"asr_at_n{N}" for N in CURVE_NS}
    for N in CURVE_NS:
        a, b = ours[f"asr_at_n{N}"], theirs[f"asr_at_n{N}"]
        assert set(a) == set(b), sorted(set(a) ^ set(b))
        assert a["metric_id"] == b["metric_id"] and a["threshold"] == b["threshold"] == 0.5 and a["computation_id"] == b["computation_id"] == ""
        assert a["extra"]["bon_n"] == b["extra"]["bon_n"] == N and a["computed_by"] == b["computed_by"] == "producer"
        assert isinstance(a["n"], int) and isinstance(b["n"], int) and (a["k"] is None or isinstance(a["k"], int))


def test_a_climb_with_no_curve_grid_files_its_headline_and_refuses_the_curve_by_name(tmp_path):
    """A climb whose headline block stands but whose grid is absent files no curve row and is refused by name (`<key>|curve`); a grid lacking one
    of the seven N files the other six and names the missing N; the headline rows stand; the read publishes; the facts and the status row carry
    both."""
    rs = _files(tmp_path)
    sp = os.path.join(rs, BS.STATS)
    stats = json.load(open(sp))
    del stats["cells"][f"{F.VIEW}/{F.LORA}"]["asr_at_n_4096_cf_t0.5"]
    stats["cells"][f"{F.VIEW}/none"]["asr_at_n_4096_cf_t0.5"].pop("8")
    json.dump(stats, open(sp, "w"))
    seeds = F.synthetic_seeds()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused + ing.blocked + ing2.blocked == 0
    climb = {r[1]: r[0] for r in F.q(root, "SELECT run_id, campaign_cell_id_legacy FROM run WHERE lane_source_legacy = 'bon_stores' AND token_cap_tokens = 4096")}
    assert _curve_rows(root, climb[F.LORA]) == {}
    assert set(_curve_rows(root, climb["none"])) == {f"asr_at_n{N}" for N in CURVE_NS if N != 8}
    assert F.q(root, "SELECT count(*) FROM run_metric WHERE metric_id = 'asr_bon_cap4096' AND threshold = 0.5") == [(2,)]      # the headlines stand
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert any(k == f"{F.VIEW}/{F.LORA}|curve" and "no `asr_at_n_4096_cf_t0.5` grid" in v and "headline block stands" in v for k, v in why), why
    assert any(k == f"{F.VIEW}/none|curve" and "lacks N = [8]" in v for k, v in why), why
    assert not any(k in (f"{F.VIEW}/{F.LORA}", f"{F.VIEW}/none") for k, _ in why), why       # the STORE keys are not refused -- the curve is
    assert ad.stats["curve_rows"] == 6
    absent = {d["key"]: d for d in ad.stats["curve_grids_absent"]}
    assert set(absent) == {f"{F.VIEW}/{F.LORA}", f"{F.VIEW}/none"}
    assert absent[f"{F.VIEW}/{F.LORA}"]["missing_n"] == list(CURVE_NS) and absent[f"{F.VIEW}/{F.LORA}"]["run_id"] == climb[F.LORA]
    assert absent[f"{F.VIEW}/none"]["missing_n"] == [8] and absent[f"{F.VIEW}/none"]["run_id"] == climb["none"]
    fx = json.loads(F.q(root, "SELECT extra_json FROM source_status WHERE adapter = 'bon_stores'")[0][0])
    assert fx["curve_rows"] == 6 and len(fx["curve_grids_absent"]) == 2 and any("|curve" in s_ for s_ in fx["row_refusals"])


def test_a_registered_point_the_seeds_also_spell_as_a_model_is_still_a_registered_point(tmp_path):
    """The live shape since em_misalign 8 (2026-09-22): legacy_spellings.yaml spells the gemma / nemotron `<model>_paired_d1_<model>_s1` groups
    as models (their paired-write agentic roots), and version 4 read that spelling FIRST, so the same group in the statistics file became a
    campaign standup with no VIEW_MANIFEST -- refused by name, its selected climb run gone from the run set (a recorded drop). A registration
    whose `view` spells a point is a registered point whatever the seeds say of the group; a campaign view (`main` / `it`) is unchanged."""
    rs = _files(tmp_path, paired=True)
    seeds = F.synthetic_seeds()
    seeds.legacy["model_keys"].append({"legacy": PAIRED_KEY, "model_id": F.MODEL_IT, "fit_model_id": F.MODEL_BASE, "transfer_kind": "base_to_instruct"})
    assert seeds.model_id_for_view_legacy(PAIRED_KEY) == F.MODEL_IT                 # the seeds now spell the point's group as a model
    paired = _PairedObjects()
    ad = BS.BonStoresAdapter(rs=rs, seeds=seeds, objects=F.resolver(seeds, tmp_path), paired_objects=paired)
    st = ad._standup(PAIRED_KEY, "")
    assert st is not None and not st.is_campaign and st.point is not None and st.point.seed == 1 and st.model_view_legacy == F.VIEW and st.view_legacy == POINT
    assert ad._standup(F.VIEW, "").is_campaign and ad._standup(F.VIEW_IT, "").is_campaign          # the campaign views: unchanged
    obj_steered = paired.resolve(F.VIEW, 1, POINT, F.PAIRED_WRITE, 1.0)
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), obj_steered])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused + ing.blocked + ing2.blocked == 0
    why = [(it["key"], it["why"]) for it in ad.refusals.items]
    assert not any(k.startswith(PAIRED_KEY + "/") or "VIEW_MANIFEST" in v for k, v in why), why
    own = F.q(root, "SELECT campaign_cell_id_legacy, token_cap_tokens, instrument_id, object_id FROM run WHERE view_legacy = ? AND lane_source_legacy = 'bon_stores' ORDER BY 1", (POINT,))
    assert [(r[0], r[1]) for r in own] == [("none", 4096), (f"paired_s1__{F.PAIRED_WRITE}", 4096)]      # the climb runs of the point, as before
    assert {r[2] for r in own} == {seeds.instrument_id("bon_512store_4096climb_vendor_preset", EVAL)} and {r[3] for r in own} == {"none", F.SHA_PAIRED}
    assert ad.stats["registered_points"] == 1 and ad.stats["curve_rows"] == 28

