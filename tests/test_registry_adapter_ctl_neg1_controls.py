"""The over-refusal controls of the control-negative battery (`cellstore.CtlNeg1ControlsAdapter`; 2026-09-25) on a synthetic mirror in
EXACTLY the producer's shape (hostname TESTHOST, invented shas, tiny stores):

* `controls_sampled/controls_cap{512,2048,8192}/<standup>_t0/cells/<store>/`, ONE store per standup and cap whose manifest declares BOTH
  controls, each control judged ONLY at its own final rung (xstest at 2,048; or_bench_hard_1k at 8,192), the 512 root never judged, the
  ladder record `ladders/<standup>_t0/ladder.json` the exit authority; a `flip_pass/` sibling that is never read;
* NO `none__controls__*` store anywhere (the standups serve only the steered object): every steered run files with NO contrast and the
  DECLARED seed anchor (`anchor_standup_legacy` = `<model>_s<k>`, `anchor_kind_declared` = `sibling_none`, the rule) on its extra;
* the dose -1.0 from the served-key grammar's one home (`derive.SEEDED_DOSE_FAMILIES`), cross-checked against the object's dose sidecar,
  `strength_unit nominal_only`, not the operating point; `lane_source_legacy ctl_neg1`; `view_legacy` the standup (no `_t0`); the object by
  the manifest's sha prefix; the ladder's exit in `extra.escalation_state`;
* a target directory the producer SET ASIDE (`<standup>_t0.failed_<job>_<task>`: an empty steered draw the liveness gate refused) is never a
  store root: its files are the facts-only group `_failed_roots`, whose facts name the standup's MISSING over-refusal row and the reason --
  never a refusal of the battery;
* a pending store (generated, no judge pass) is skipped, never filed `generated`; an empty or absent mirror discovers nothing and refuses
  nothing; every record validates; the reader's name / source / owner / priority are the battery's, distinct from the statics reader's.
"""
from __future__ import annotations

import json
import os

import pytest

import _anchor_frame_fixtures as AF
import registry_items_fixtures as F
from sisteer.registry import vocab as V
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import cellstore as CS
from sisteer.registry.adapters import sampled_static as SS
from sisteer.registry.adapters import strength_mtdiff as SM

from test_registry_adapter_orxs_seedrep import _ctx, _payloads, _validate_all, write_ladder, write_store  # noqa: E402

CELL_LORA = "constv2ctl2-full__lora_r4_ce"
CELL_RELU = "constv2ctl2-full__psr_ce_all"
LORA_S2 = "synthmodel_dm1_ctllora64s2_t0"
RELU_S3 = "synthmodel_dm1_ctlrelu64s3_t0"
FAILED_RELU_S1 = "synthmodel_dm1_ctlrelu64s1_t0.failed_140946_3"
CAPS = (512, 2048, 8192)
#: the pairs' exits, as the ladder records spell them: xstest left at 2,048, or_bench_hard_1k at 8,192 (the LoRA s1 / s3 shape)
EXITS_LORA = {CELL_LORA: {"xstest": 2048, "or_bench_hard_1k": 8192}}
EXITS_RELU = {CELL_RELU: {"xstest": 2048, "or_bench_hard_1k": 2048}}


def _mirror(tmp_path, *, with_failed=True):
    root = str(tmp_path / "ctl_neg1" / "sampled")
    caps_parent = os.path.join(root, "controls_sampled")
    ladders = os.path.join(root, "ladders")
    stores = {}
    # the LoRA s2 standup: 512 generated only; 2,048 judged on xstest only; 8,192 judged on or_bench_hard_1k only
    for cap, judged in ((512, ()), (2048, ("xstest",)), (8192, ("or_bench_hard_1k",))):
        stores[(LORA_S2, cap)] = write_store(caps_parent, cap, CELL_LORA, judged_controls=judged, target=LORA_S2, job="140982",
                                             escalated_from=({512: None, 2048: 512, 8192: 2048}[cap]),
                                             labels={"xstest": ["full_compliance", "full_compliance", "full_refusal", "partial_refusal"], "or_bench_hard_1k": ["full_compliance", "full_refusal"]},
                                             at_cap={"xstest": ["xst0001-cccccccc"]} if cap == 512 else {})
    # the ReLU s3 standup: both controls left at 2,048, judged there
    for cap, judged in ((512, ()), (2048, ("xstest", "or_bench_hard_1k"))):
        stores[(RELU_S3, cap)] = write_store(caps_parent, cap, CELL_RELU, judged_controls=judged, target=RELU_S3, job="140946",
                                             escalated_from=({512: None, 2048: 512}[cap]), labels={"xstest": ["full_refusal"], "or_bench_hard_1k": ["full_compliance"]}, at_cap={})
    write_ladder(ladders, LORA_S2, EXITS_LORA)
    write_ladder(ladders, RELU_S3, EXITS_RELU)
    # a flip_pass sibling: never a rung
    flip = os.path.join(caps_parent, "flip_pass", "cap2048", LORA_S2, "cells")
    os.makedirs(flip, exist_ok=True)
    if with_failed:
        # the producer's set-aside root for the ReLU s1 standup: the manifest and the evidence, no generation record, no judge pass
        failed = os.path.join(caps_parent, "controls_cap512", FAILED_RELU_S1)
        store = os.path.join(failed, "cells", f"{CELL_RELU.replace('__', '--')}__controls__t0__a0__00ad62c4__9f27ffa7")
        os.makedirs(store, exist_ok=True)
        json.dump({"cell_id": CELL_RELU, "controls": ["xstest", "or_bench_hard_1k"], "fit_provenance": {"object_sha": "54df102e494780f4"}}, open(os.path.join(store, "manifest.json"), "w"))
        json.dump({"cell_id": CELL_RELU, "driver": "served_capability_run_v2", "engine_stamp": {"object_sha": "54df102e494780f4"}}, open(os.path.join(store, "evidence.json"), "w"))
        json.dump({"cell_id": CELL_RELU, "driver": "served_capability_run_v2", "engine": "vllm"}, open(os.path.join(failed, "evidence.json"), "w"))
    roots = CS.ctl_neg1_controls_roots(parents=(caps_parent,), declared=CAPS, ladders=ladders)
    return {"root": root, "caps_parent": caps_parent, "ladders": ladders, "stores": stores, "roots": roots}


def _adapter(mirror):
    dims = F.dims_map()
    dims.object_scales[F.OBJ_SHA] = (-1.0, "dose_scale")        # the served object is the -1 dose-scaled copy (objects_bank: base_scale -1.0)
    return CS.CtlNeg1ControlsAdapter(roots=mirror["roots"], dims=dims)


def _groups(ad):
    out = {}
    for i in ad.discover():
        out.setdefault(ad.comparison_group(i), []).append(i)
    return out


def _read(ad, group, groups):
    """One group's read on a frame PRIMED from the adapter's own emissions (the steady state after an ingest; every comparison a steered store
    intends is decided by the pass's anchor currency since the shared reader's 8 -- the battery holds no `none`, so each parks by name)."""
    frame = getattr(ad, "_test_frame", None)
    if frame is None:
        frame = AF.primed_frame(ad, now=_ctx().now, producer=dict(_ctx().producer), root="/nonexistent")
        ad._test_frame = frame
    batches = list(ad.read(_ctx(frame.con), group, groups[group]))
    assert len(batches) == 1
    return batches[0].records


def test_the_reader_is_the_batterys_own_and_its_roots_are_the_mirrors_caps_by_name(tmp_path):
    m = _mirror(tmp_path)
    ad = _adapter(m)
    assert (ad.name, ad.covers_sources, ad.owner, ad.priority) == ("ctl_neg1_controls", ("ctl_neg1",), "Scale", IC.PRIORITY_GENERIC_READER)
    assert ad.status_sources("x") == ("ctl_neg1",) and CS.CTL_NEG1_SOURCE == SS.CTL_NEG1_SOURCE == "ctl_neg1"
    by_cap = {int(r.path.rsplit("controls_cap", 1)[1]): r for r in m["roots"]}
    assert list(by_cap) == [512, 2048, 8192]
    for r in m["roots"]:
        assert (r.run_key_scope, r.at_cap_from_gen_stats, r.source_legacy, r.over_refusal_headline, r.cap_provenance, r.file_unjudged) == ("target_dir", True, "ctl_neg1", True, True, False)
        assert (r.object_resolution, r.ladder_parent, r.strip_trial_suffix, r.substrate_from_routing_table) == ("sha_prefix", m["ladders"], True, True)
    # the live constants: the battery's mirror root, its caps, the failed-root mark
    assert CS.CTL_NEG1_ROOT == os.path.join(IC.RUN_STATE, "ctl_neg1", "sampled") == SS.CTL_NEG1_ROOT
    assert CS.CTL_NEG1_CONTROLS_PARENTS == (os.path.join(CS.CTL_NEG1_ROOT, "controls_sampled"),) and CS.CTL_NEG1_LADDERS == os.path.join(CS.CTL_NEG1_ROOT, "ladders")
    assert CS.CTL_NEG1_DECLARED_CAPS == (512, 2048, 8192) and CS.CTL_NEG1_FAILED_MARK == ".failed_"
    # the statics reader files the same source at another priority (the coverage rule), so both are the battery's
    assert SS.CtlNeg1StaticAdapter.priority != ad.priority and SS.CtlNeg1StaticAdapter.owner == ad.owner == "Scale"


def test_only_judged_stores_are_groups_the_pending_roots_are_skipped_and_the_failed_root_is_the_facts_group(tmp_path):
    m = _mirror(tmp_path)
    ad = _adapter(m)
    groups = _groups(ad)
    stores = {g for g in groups if g != CS.CTL_NEG1_FAILED_GROUP}
    assert stores == {m["stores"][(LORA_S2, 2048)], m["stores"][(LORA_S2, 8192)], m["stores"][(RELU_S3, 2048)]}, stores
    assert m["stores"][(LORA_S2, 512)] not in groups and m["stores"][(RELU_S3, 512)] not in groups          # pending, never filed
    assert not any("flip_pass" in g for g in groups)
    failed = groups[CS.CTL_NEG1_FAILED_GROUP]
    assert {os.path.basename(i.uri) for i in failed} == {"evidence.json", "manifest.json"} and all(FAILED_RELU_S1 in i.uri for i in failed)
    assert _read(ad, CS.CTL_NEG1_FAILED_GROUP, groups) == []
    facts = ad.status_extra(CS.CTL_NEG1_FAILED_GROUP)
    assert facts["missing_over_refusal_rows"] == ["synthmodel_dm1_ctlrelu64s1"]
    (fr,) = facts["failed_roots"]
    assert (fr["standup"], fr["cap"], fr["producer_suffix"]) == ("synthmodel_dm1_ctlrelu64s1", 512, "140946_3")
    assert fr["failed_root"] == f"controls_cap512/{FAILED_RELU_S1}" and "text-liveness gate" in fr["reason_declared"] and "NO over-refusal row" in fr["reason_declared"]
    assert "evidence.json" in fr["files"] and any(f.endswith("manifest.json") for f in fr["files"]) and "no over-refusal run" in fr["missing_row"]
    assert len(ad.refusals) == 0 if hasattr(ad, "refusals") else True
    # without the set-aside root there is no facts group
    m2 = _mirror(tmp_path / "clean", with_failed=False)
    assert CS.CTL_NEG1_FAILED_GROUP not in _groups(_adapter(m2))


def test_each_judged_store_files_one_run_per_judged_control_at_its_cap_with_the_dose_the_source_and_the_declared_anchor(tmp_path):
    m = _mirror(tmp_path)
    ad = _adapter(m)
    groups = _groups(ad)
    want = {(LORA_S2, 2048): {"xstest"}, (LORA_S2, 8192): {"or_bench_hard"}, (RELU_S3, 2048): {"xstest", "or_bench_hard"}}
    seed_of = {LORA_S2: "s2", RELU_S3: "s3"}
    for (target, cap), evals in want.items():
        recs = _read(ad, m["stores"][(target, cap)], groups)
        assert _validate_all(recs) == []
        runs = _payloads(recs, "run")
        assert {p["eval_id"] for p in runs} == evals, [(p["eval_id"], p["token_cap_tokens"]) for p in runs]
        assert not _payloads(recs, "run_contrast")                                           # no `none` store anywhere, nothing current at the key
        # the shared reader's 8: each over-refusal comparison is decided (`none_declared`, no unsteered run at the exact key on the frame) and parks
        # BY NAME on its steered row -- one refusal line per comparison, the park on the row, the facts counting them; never in silence
        for p in runs:
            parked = [mr for mr in _payloads(recs, "run_metric") if mr["run_id"] == p["run_id"] and "delta_block" in mr.get("extra", {})]
            assert {mr["metric_id"] for mr in parked} == {("xst_refused" if p["eval_id"] == "xstest" else "orb_refused"), "over_refusal_graded", "over_refusal_label_mean"}
            for mr in parked:
                blk = mr["extra"]["delta_block"]
                assert (blk["anchor"]["state"], blk["anchor"]["own_reason"], blk["anchor"]["own_anchor_run_id"], blk["delta"]) == ("no_current_anchor", "none_declared", None, None)
                assert "no unsteered run of" in blk["why_not_a_contrast_row"] and blk["anchor_selection"]["n_candidates"] == 0
        facts = ad.status_extra(m["stores"][(target, cap)])
        assert facts["contrasts_parked"] == 3 * len(runs) == len(ad.refusals) and facts["anchor_states"] == {"no_current_anchor": 3 * len(runs)} and facts["contrasts"] == 0
        assert all(it["key"].endswith("|contrast") and "none_declared" in it["why"] for it in ad.refusals.items), ad.refusals.items
        for p in runs:
            assert p["lane_source_legacy"] == "ctl_neg1" and p["view_legacy"] == target[:-3] and p["token_cap_tokens"] == cap
            assert p["producer_run_key_legacy"].startswith(f"{target}/") and p["object_id"] == F.OBJ_SHA and p["model_id"] == F.MODEL_IT
            assert (p["strength_nominal"], p["strength_nominal_source"], p["strength_unit"], p["is_operating_point"]) == (-1.0, "derived", "nominal_only", 0)
            assert p["extra"]["dose_served_key"] == -1.0 and p["extra"]["dose_sidecar_base_scale"] == -1.0
            assert (p["extra"]["anchor_standup_legacy"], p["extra"]["anchor_kind_declared"]) == (f"synthmodel_{seed_of[target]}", "sibling_none")
            assert p["extra"]["anchor_kind_declared"] in V.ANCHOR_KINDS and p["extra"]["anchor_rule"] == SM.ANCHOR_RULE and target in p["extra"]["anchor_source"]
            assert p["token_cap_rung_index"] == [512, 2048, 8192].index(cap) and p["extra"]["escalation_state"]
        # the shared reader's convention: the run row is `generated`, judging is the `judged` run_state event beside its judgment -- one per run
        judged_states = {s["run_id"] for s in _payloads(recs, "run_state") if s["state"] == "judged"}
        assert judged_states == {p["run_id"] for p in runs} == {j["run_id"] for j in _payloads(recs, "judgment")}
        # the headline rows the shared reader files on a judged over-refusal store
        metric_ids = {mtr["metric_id"] for mtr in _payloads(recs, "run_metric")}
        assert {"over_refusal_graded", "over_refusal_label_mean"} <= metric_ids and ({"xst_refused", "orb_refused"} & metric_ids)
    # the two standups' run ids are their own
    ids = {t: {p["run_id"] for p in _payloads(_read(ad, m["stores"][(t, 2048)], groups), "run")} for t in (LORA_S2, RELU_S3)}
    assert ids[LORA_S2].isdisjoint(ids[RELU_S3])


def test_the_anchor_stamp_reads_the_seed_from_the_target_directory_and_stamps_only_steered_rows():
    stamp = CS.ctl_neg1_anchor_stamp("qwen35_4b_dm1_ctlrelu64s3_t0", "relu_steer")
    assert (stamp["anchor_standup_legacy"], stamp["anchor_kind_declared"]) == ("qwen35_4b_s3", "sibling_none") and stamp["anchor_rule"] == SM.ANCHOR_RULE
    assert CS.ctl_neg1_anchor_stamp("qwen35_4b_dm1_ctllora64s1", "lora")["anchor_standup_legacy"] == "qwen35_4b_s1"
    assert CS.ctl_neg1_anchor_stamp("qwen35_4b_dm1_ctllora64s1_t0", "none") == {} and CS.ctl_neg1_anchor_stamp("qwen35_4b_dm1_ctllora64s1_t0", "system_instruction") == {}
    assert CS.ctl_neg1_anchor_stamp("qwen35_4b_s2_t0", "lora") == {} and CS.ctl_neg1_anchor_stamp("qwen35_4b_d0p5_loradiff64s2_t0", "lora_subtraction")["anchor_standup_legacy"] == "qwen35_4b_s2"


def test_an_absent_or_empty_mirror_discovers_nothing_and_refuses_nothing(tmp_path):
    roots = CS.ctl_neg1_controls_roots(parents=(str(tmp_path / "nowhere" / "controls_sampled"),), declared=CAPS, ladders=str(tmp_path / "nowhere" / "ladders"))
    ad = CS.CtlNeg1ControlsAdapter(roots=roots, dims=F.dims_map())
    assert ad.discover() == [] and len(roots) == 3
