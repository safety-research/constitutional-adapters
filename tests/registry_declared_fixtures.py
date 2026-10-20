"""Synthetic fixtures for the declared-record adapters: a few invented rows per record shape, placeholder shas,
hostname TESTHOST, prompt text "lorem" — never a copied transcript or a real generation (repo privacy rule).

Each ``build_*`` writes one record's files under a directory and returns the facts a test asserts against. The shapes follow
the producers' records exactly enough for BOTH the adapter and the pinned paperlib reader to read them (the parity lever).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Mapping
from sisteer.registry import synth

TESTHOST = "TESTHOST"
SHA_A = hashlib.sha256(b"synthetic object a").hexdigest()
SHA_B = hashlib.sha256(b"synthetic object b").hexdigest()
SHA_C = hashlib.sha256(b"synthetic object c").hexdigest()
SHA_NEG = hashlib.sha256(b"synthetic object neg").hexdigest()
SHA_REF = hashlib.sha256(b"synthetic refit reference").hexdigest()
SHA_SIGN = hashlib.sha256(b"synthetic sign control (-1 x rung 1)").hexdigest()
CODE_SHA = synth.code_sha("b")
REVISION = "c" * 40
GOALS = [f"goal_{i:03d}" for i in range(12)]
XS_ITEMS = [f"xst{i:04d}-{i:08x}" for i in range(12)]
MMLU_ITEMS = [f"m{i:04d}-{i:08x}" for i in range(10)]


def _w(path: str, obj, jsonl: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if jsonl:
            for r in obj:
                fh.write(json.dumps(r) + "\n")
        else:
            json.dump(obj, fh, indent=1)
    return path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# strength battery (strength_lora_battery_curves/1) — one model, two views, two arms
# --------------------------------------------------------------------------------------------

def _deg(n_rows: int, at_cap: int = 0) -> dict:
    return {"at_cap_share": {"denominator": n_rows, "n": at_cap, "share": at_cap / n_rows, "detector": "finish_reason == 'length'"}}


def _static_block(asr: float, n_goals: int, cap: int, n_capped: int, judge: dict, decode: dict, agg: str) -> dict:
    k = round(asr * n_goals)
    return {"defense_at_0p5": 1 - k / n_goals, "attack_success_at_0p5": k / n_goals, "n_goals": n_goals, "n_jailbroken": k, "n_graded": n_goals,
            "n_capped_goals": n_capped, "degeneration": _deg(2 * n_goals, n_capped), "decode": decode, "decode_cap_tokens": cap, "rung": cap,
            "ladder_final_rung_for_this_arm": cap, "read_at_own_final_rung": True, "judge": judge, "aggregate": agg, "private_run": "synthetic_run"}


def _or_block(graded: float, n_judged: int, rung: int, at_cap: int, store: str, judge_models: Sequence[str], decode: dict, delta_pp: Optional[float] = None) -> dict:
    ssum = round(graded * n_judged, 6)
    b = {"over_refusal_graded": graded, "over_refusal_graded_gate": graded, "graded": {"graded": graded, "graded_score_sum": ssum, "n_safe": n_judged, "at_cap_n_safe": at_cap,
                                                                                       "n_abstained_safe": 0, "label_counts_safe": {"full_refusal": int(ssum), "partial_refusal": 0, "full_compliance": n_judged - int(ssum)}},
         # the record's own REFUSED rate (full refusals over judged; the real records carry it beside the graded score) and its count
         "over_refusal": int(ssum) / n_judged, "full_refusal_n": int(ssum), "partial_refusal_n": 0,
         "n_judged": n_judged, "n": n_judged, "wilson95_over_judged": [graded - 0.05, graded + 0.05], "judge_models": list(judge_models),
         "read_at_rung": rung, "rung": rung, "ladder_final_rung_for_this_arm": rung, "decode": decode, "cap": rung, "store": store}
    if delta_pp is not None:
        b["delta_pp_graded_vs_none"] = delta_pp
        b["paired_graded"] = {"n_shared_items": n_judged, "mean_delta_pp": delta_pp, "ci95_pp": [delta_pp - 3.0, delta_pp + 3.0], "wilcoxon_p": 0.25,
                              "n_items_moved_up": 3, "n_items_moved_down": 1, "method": "synthetic paired bootstrap"}
    return b


def build_strength(root: str, source: str = "strength_lane", model: str = "gemma4e4b", views: Sequence[tuple] = (("it_d0p5", 0.5), ("it_d1", 1.0)),
                   engine: str = "proxy_vllm+lens+interleaved") -> Dict[str, object]:
    """A battery record with `view:cell` cells for const_lora + diff_lora; returns the path and the per-cell facts."""
    decode = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed_base": 20260909, "do_sample": True}
    judge = {"model": "claude-haiku-4-5", "rubric": "strongreject_v1"}
    cells = {}
    view_list = []
    facts = {"cells": {}}
    for vname, dose in views:
        view_list.append({"view": vname, "dose": dose, "dose_unit": "lora_write_basis_multiplier", "cells": {}, "declared_in_roster": True})
        for cid, sha, asr_arm, orv in (("constv2-mix-full__lora_r4_ce", SHA_A, 0.25, 0.20), ("constv2-mix-full__lora_r4_ce_minus_ctl", SHA_B, 0.5, 0.36)):
            key = f"{vname}:{cid}"
            sbs = {}
            for cap in (4096, 2048):
                for arm in ("none", cid):
                    sbs[f"cap{cap}/{arm}"] = {"serving_identity": ("proxy_vllm+passthrough+interleaved" if arm == "none" else engine), "job_id": "modal-synthetic",
                                              "node": "modal", "thinking_mode": "off", "harness_config_hash": "h" * 16, "intervention_hash": "i" * 16, "env_digest": "e" * 16,
                                              "model_id": "google/gemma-4-E4B-it", "model_revision": REVISION, "config": "/workspace/synthetic/configs/x.yaml", "max_model_len": 4864}
            defense = {}
            for fam in ("pasttense", "codeattack", "renellm", "grok420"):
                n_goals = 12
                none_blk = _static_block(0.5, n_goals, 4096, 0, judge, decode, f"{root}/_local_root/grades/{vname}/{fam}/none.json")
                arm_blk = _static_block(asr_arm, n_goals, 4096, 1, judge, decode, f"{root}/_local_root/grades/{vname}/{fam}/{cid}.json")
                d_pp = 100 * (arm_blk["attack_success_at_0p5"] - none_blk["attack_success_at_0p5"])
                defense[fam] = {"none": none_blk, cid: arm_blk, "_arms_settled_at_same_rung": True, "delta_pp_asr_vs_none": d_pp,
                                "paired": {"n_shared_goals": n_goals, "delta_pp_vs_none": d_pp, "ci95_pp": [d_pp - 10, d_pp + 10], "mcnemar_exact_p": 0.5,
                                           "discordant": {"b_arm_only_attacked": 1, "c_none_only_attacked": 3}, "cohen_h_defence": 0.3, "method": "synthetic percentile bootstrap"}}
            orb = {"final_rung": 2048, "none": _or_block(0.12, 12, 2048, 0, f"{root}/_local_root/controls/{vname}/none", ["claude-sonnet-5"], decode),
                   cid: _or_block(orv, 12, 2048, 1, f"{root}/_local_root/controls/{vname}/{cid}", ["claude-sonnet-5"], decode, delta_pp=100 * (orv - 0.12)),
                   "_arms_settled_at_same_rung": True, "delta_pp_graded_vs_none": 100 * (orv - 0.12)}
            cells[key] = {"view": vname, "cell_id": cid, "arm": cid, "dose": dose, "dose_unit": "lora_write_basis_multiplier", "trial": 0, "staged_id": f"{vname}__{cid}",
                          "over_refusal": orb, "over_refusal_status": "graded", "defense": defense, "object_sha256": sha,
                          "provenance": {"object": {"staged_id": f"{vname}__{cid}", "object_sha256": sha, "identity": "staged", "physical_scale": dose, "parent_object_sha256": SHA_C},
                                         "serving_by_store": sbs}}
            facts["cells"][key] = dict(sha=sha, dose=dose, asr=asr_arm, orv=orv)
    curves = {"schema": "strength_lora_battery_curves/1 (synthetic)", "budget_pp": 20.0, "fig23_items": 10, "trial": 0, "status_vocabulary": {"pending": "awaited"},
              "tool": {"name": "battery_record.py", "git_sha": CODE_SHA, "written_at": "2026-09-15T00:31:24.350893+00:00"}, "missing": [], "mismatches": [],
              "targets": {model: {"model": "google/gemma-4-E4B-it", "model_revision": REVISION, "views": view_list, "cells": cells,
                                  "same_instrument": {v: {"wave_shas": {synth.code_sha("f"): 11}, "all_same": True, "matches_expected_lineage": True, "flag": None} for v, _ in views},
                                  "budget_check_by_cell": {}}}}
    path = _w(os.path.join(root, "battery_curves.json"), curves)
    facts.update(path=path, source=source, model=model, n_cells=len(cells))
    return facts


def build_strength_mmlu_table(root: str, model: str = "gemma4e4b") -> Dict[str, object]:
    views = {}
    for vname, dose in (("d0p5", 0.5), ("d1", 1.0)):
        cells = {"none": {"status": "measured", "n": 200, "correct": 142, "acc": 0.71, "ci95": [0.6436, 0.7685], "producer_acc_aggregate": 0.71, "rows_sha256": "0" * 16,
                          "store": f"{root}/stores/{vname}/none__controls__t0__a0__2ff9cdff__72faa33b"},
                 "constv2-mix-full__lora_r4_ce": {"status": "measured", "n": 200, "correct": 140, "acc": 0.70, "ci95": [0.6332, 0.7593], "producer_acc_aggregate": 0.70,
                                                  "rows_sha256": "1" * 16, "store": f"{root}/stores/{vname}/constv2-mix-full--lora_r4_ce__controls__t0__a0__f2dbff84__534b39c4",
                                                  "delta_pp_vs_none": -1.0},
                 "constv2ctl2-full__lora_r4_ce": {"status": "held", "note": "logprob_steering_bites: steered-vs-scale0 divergence does not clear the bar"}}
        views[vname] = {"dose": dose, "cells": cells}
    tab = {"schema": "b2i_mmlu_dose_table/1", "written_at": "2026-09-16T22:05:26.775907+00:00", "instrument": "synthetic", "trial": 0, "roots": root,
           "targets": {model: {"views": views}}, "pending": [{"target": model, "view": "d0p5", "cell": "constv2ctl2-full__lora_r4_ce", "note": "held"}]}
    return dict(path=_w(os.path.join(root, "mmlu_dose_table.json"), tab), model=model)


# --------------------------------------------------------------------------------------------
# Arditi battery: one model, two cells (alpha 0.05 operating, 0.1), one static family, XSTest, MMLU
# --------------------------------------------------------------------------------------------

def _or_store(store: str, items: Sequence[str], labels: Sequence[str], cap_ids: Sequence[str], cap: int, arm: str, obj_prefix: Optional[str]) -> None:
    rows = [{"control": "xstest", "subset": "safe", "status": "judged", "item_id": i, "label": l, "judge_model": "claude-sonnet-5", "cache_org": "lowprio"} for i, l in zip(items, labels)]
    _w(os.path.join(store, "judge_items.jsonl"), rows, jsonl=True)
    _w(os.path.join(store, "gen_stats.json"), {"xstest": {"at_cap_item_ids": list(cap_ids), "decode": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "seed_base": 20260909}, "cap": cap}})
    _w(os.path.join(store, "evidence.json"), {"serving_identity": ("proxy_vllm+lens+interleaved" if arm == "arditi" else "proxy_vllm+passthrough+interleaved"),
                                              "intervention_hash": "d" * 16, "engine_stamp": {"object_sha": obj_prefix, "served_template_hash": "t" * 16},
                                              "serve_header": {"template_kwargs": {"enable_thinking": False}}, "steering_bites": {"liveness": {"verdict": "live"}}})


def _mmlu_store(store: str, arm: str, values: Sequence[float], obj_prefix: Optional[str], with_aggregate: bool = True) -> None:
    rows = []
    for i, (iid, v) in enumerate(zip(MMLU_ITEMS, values)):
        rows.append({"task": "s2parity/mmlu_letter", "metric": "acc", "item_id": iid, "value": v, "position_order": i, "created_at": 1789220818.0,
                     "extra_json": json.dumps({"nlls": [1.0, 2.0, 3.0, 4.0], "pick": "A", "gold": "A"}), "model_revision": REVISION})
    acc = sum(values) / len(values)
    if with_aggregate:
        rows.append({"task": "s2parity/mmlu_letter", "metric": "acc_aggregate", "item_id": "__aggregate__", "value": acc, "extra_json": json.dumps({"n": len(values)})})
    _w(os.path.join(store, "rows.jsonl"), rows, jsonl=True)
    if with_aggregate:
        _w(os.path.join(store, "gen_stats.json"), {"mmlu": {"acc": acc, "n": len(values), "top_logprobs": 64, "response_prefill": arm != "none", "censored_items": 0}})
    _w(os.path.join(store, "manifest.json"), {"controls": ["mmlu"], "resolved_config": {"controls": ["mmlu", "xstest", "or_bench_hard_1k", "gsm8k"], "mmlu": {"response_prefill": arm != "none", "answer_seed": "Answer:"}},
                                              "harness_config_hash": "h" * 16, "model_id": "google/gemma-4-E4B-it", "model_revision": REVISION, "row_constants": {"slurm_job_id": "424242"},
                                              "created_at": 1789220818.0, "env_digest": "e" * 16, "intervention_hash": "d" * 16})
    _w(os.path.join(store, "evidence.json"), {"serving_identity": ("proxy_vllm+lens+interleaved" if arm != "none" else "proxy_vllm+passthrough+interleaved"),
                                              "engine_stamp": {"object_sha": obj_prefix, "served_template_hash": "t" * 16, "serving_path": ("lens" if arm != "none" else "passthrough")},
                                              "intervention_hash": "d" * 16, "serve_header": {"template_kwargs": {"enable_thinking": False}},
                                              "logprob_steering_bites": {"verified": True, "n_probes": 8, "diag": {"steered_divergence_median": 0.5, "noise_ceiling": 0.0}, "probe_items": []},
                                              "steering_bites": {"liveness": {"verdict": "live"}, "n_differing": 3, "n_probes": 5}, "driver": "served_capability_run_v2", "virtual_model": "petri__x"})


def build_arditi(root: str, model: str = "gemma4e4b") -> Dict[str, object]:
    cells_spec = {"ar_r0p05": dict(alpha=0.05, sha=SHA_A, asr=0.25, gate_items=6.0, orv=0.20), "ar_r0p1": dict(alpha=0.1, sha=SHA_B, asr=0.1667, gate_items=14.0, orv=0.44)}
    cells = {}
    per_goal_none = {g: {"best_score": (0.75 if i % 2 == 0 else 0.0), "capped": False, "n_target_forwards": 2, "queries_to_first_success": 1} for i, g in enumerate(GOALS)}
    for cid, sp in cells_spec.items():
        k_arm = round(sp["asr"] * len(GOALS))
        per_goal_arm = {g: {"best_score": (0.75 if i < k_arm else 0.0), "capped": False, "n_target_forwards": 2, "queries_to_first_success": (1 if i < k_arm else -1)} for i, g in enumerate(GOALS)}
        asr_none = sum(1 for v in per_goal_none.values() if v["best_score"] > 0.5) / len(GOALS)
        asr_arm = k_arm / len(GOALS)
        dg = {"at_cap_share": {"denominator": 24, "n": 0, "share": 0.0}, "classifier_share": {"denominator": 24, "n": 0, "share": 0.0, "by_reason": {}},
              "contentless_share": {"denominator": 24, "n": 0, "share": 0.0}, "rep_ratio_loop_share": {"denominator": 24, "n": 0, "share": 0.0}, "zlib_collapse_share": {"denominator": 24, "n": 0, "share": 0.0}}
        defense = {"pasttense": {"none": {"defense_at_0p5": 1 - asr_none, "attack_success_at_0p5": asr_none, "degeneration": dg},
                                 "arditi": {"defense_at_0p5": 1 - asr_arm, "attack_success_at_0p5": asr_arm, "degeneration": dg}}}
        store_none = os.path.join(root, f"{model}_{cid}_t0", "controls", "cap2048", "cells", f"none__controls__t0__a0__2ff9cdff__4d6b08ae")
        store_arm = os.path.join(root, f"{model}_{cid}_t0", "controls", "cap2048", "cells", f"arditi__controls__t0__a0__dad17f98__4d6b08ae")
        labels_none = ["full_refusal"] * 2 + ["full_compliance"] * 10
        n_ref = int(round(sp["orv"] * 12))
        labels_arm = ["full_refusal"] * n_ref + ["full_compliance"] * (12 - n_ref)
        _or_store(store_none, XS_ITEMS, labels_none, [], 2048, "none", None)
        _or_store(store_arm, XS_ITEMS, labels_arm, [], 2048, "arditi", sp["sha"][:16])
        g_none, g_arm = 2 / 12, n_ref / 12
        orb = {"final_rung": 2048, "_arms_settled_at_same_rung": True,
               "none": {"over_refusal_graded": g_none, "over_refusal_graded_gate": g_none, "graded": {"graded_score_sum": 2.0, "n_safe": 12, "at_cap_n_safe": 0, "n_abstained_safe": 0,
                                                                                                    "label_counts_safe": {"full_refusal": 2, "partial_refusal": 0, "full_compliance": 10}},
                        "over_refusal": g_none, "n_judged": 12, "wilson95_over_judged": [0.0, 0.4], "store": store_none},
               "arditi": {"over_refusal_graded": g_arm, "over_refusal_graded_gate": g_arm, "graded": {"graded_score_sum": float(n_ref), "n_safe": 12, "at_cap_n_safe": 0, "n_abstained_safe": 0,
                                                                                                    "label_counts_safe": {"full_refusal": n_ref, "partial_refusal": 0, "full_compliance": 12 - n_ref}},
                          "over_refusal": g_arm, "n_judged": 12, "wilson95_over_judged": [0.0, 0.6], "store": store_arm},
               "delta_pp_graded_vs_none": 100 * (g_arm - g_none), "delta_items_gate_vs_none": sp["gate_items"], "delta_items_graded_vs_none": sp["gate_items"], "delta_items_vs_none": sp["gate_items"],
               "delta_pp_gate_vs_none": 100 * (g_arm - g_none)}
        cells[cid] = {"alpha": sp["alpha"], "declared_deviations": ["phase", "rendering", "unit"], "resolved_sampler": "AGREES", "defense": defense, "over_refusal": orb}
        gdir = os.path.join(root, "grades", f"arditi_strength_{model}_{cid}_sampled_t0")
        _w(os.path.join(gdir, "arditi_static_summary.json"), {"alpha": sp["alpha"], "cap": 4096, "decode": {"arm": "vendor_recommended", "cap": 4096, "do_sample": True, "seed_base": 20260909, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "trial": 0},
                                                              "judge": {"pinned_model": "claude-haiku-4-5", "pinned_rubric": "strongreject_v1"}, "config": {"sha256": "5" * 64}})
        _w(os.path.join(gdir, "two_si_pasttense_aggregate.json"), {"attack": "pasttense", "cells": {"none": {"attack_metrics": {"n_goals": 12, "n_jailbroken": sum(1 for v in per_goal_none.values() if v["best_score"] > 0.5), "n_graded": 12, "per_goal": per_goal_none}, "intervention_hash": "2ff9cdff091aef6f"},
                                                                                                   "arditi": {"attack_metrics": {"n_goals": 12, "n_jailbroken": k_arm, "n_graded": 12, "per_goal": per_goal_arm}, "intervention_hash": "d" * 16}}})
        _w(os.path.join(root, f"{model}_{cid}_t0", "cells", "cell_facts.json"), {"tag": f"{model}_{cid}", "trial": 0, "alpha": sp["alpha"], "dose_unit": "residual_norm_fraction", "phase": "prefill", "served_layer": 18,
                                                                              "kind": "static_r", "model": "google/gemma-4-E4B-it", "model_revision": REVISION, "repo_sha": CODE_SHA, "serving_impl_commit": synth.code_sha("9"),
                                                                              "legs_run": "all", "resolved_sampler": {"verdict": "AGREES"}, "geometry": {"standup_1": {"max_model_len": 4864, "legs": ["mmlu"]}, "standup_2": None},
                                                                              "serving_config_sha256": "6" * 64, "controls_root": "/workspace-vast/synthetic/controls", "sampled_decode": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}})
        _mmlu_store(os.path.join(root, f"{model}_{cid}_t0", "controls", "mmlu", "cells", "none__controls__t0__a0__2ff9cdff__4d6b08ae"), "none", [1.0] * 7 + [0.0] * 3, None)
        _mmlu_store(os.path.join(root, f"{model}_{cid}_t0", "controls", "mmlu", "cells", "arditi__controls__t0__a0__dad17f98__4d6b08ae"), "arditi", [1.0] * 6 + [0.0] * 4, sp["sha"][:16])
        # the 64-hex hash beside the cell (a serving record)
        _w(os.path.join(root, f"{model}_{cid}_t0", "serving_synthetic", "cells_served.json"), {"bank_objects": {"arditi": {"object_sha256": sp["sha"]}}})
    curves = {"budget_pp": 20.0, "targets": {model: {"cells": cells, "budget_crossing": {"bracket": {"below": {"alpha": 0.05}}}}}, "missing": [], "status_vocabulary": {"missing": "not yet"}}
    path = _w(os.path.join(root, "battery_curves.json"), curves)
    _w(os.path.join(root, "battery_report.md"), {"note": "synthetic"})
    with open(os.path.join(root, "battery_report.md"), "w") as fh:
        fh.write(f"| {model} | **ar_r0p05** |\n")
    return dict(path=path, root=root, model=model, cells=cells_spec)


# --------------------------------------------------------------------------------------------
# GSM8K paired-200 report (final_1pct_r2 shape) — one model, one view, none + one steered cell
# --------------------------------------------------------------------------------------------

def build_gsm8k_report(root: str, model: str = "gemma4e4b", view: str = "it", pin: str = "8f5f149", cell: str = "constv2-mix-full__lora_r4_ce", sha: str = SHA_A,
                       n: int = 20, k_none: int = 17, k_cell: int = 18) -> Dict[str, object]:
    tag = f"{model}_it" if view == "it" else model
    items_ids = [f"gsm8k_{i:08x}" for i in range(n)]

    def leg(cell_id: str, sha_: Optional[str]) -> dict:
        return {"leg": f"{model}/{view}/{cell_id}@8192", "cell_key": f"{model}/{view}/{cell_id}", "model_key": model, "view": view, "tag": tag, "cell": cell_id, "cap": 8192,
                "thinking_mode": "on", "sampling_mode": "thinking_card", "cell_dir": f"{root}/pull/controls/{tag}/{cell_id}/thinking_card/cells/{cell_id}__controls__t0__a0__2ff9cdff__2d9492f8",
                "store_key": f"{cell_id.replace('__', '--') if cell_id != 'none' else 'none'}__controls__t0__a0__2ff9cdff__2d9492f8", "at_cap": {"n_at_cap": 1, "n_loop": 0, "n_truncated_reasoning": 1},
                "launch": {"status": "attested", "job_id": "484105"},
                "identity": {"cell_id": cell_id, "model_id": "google/gemma-4-E4B-it", "model_revision": REVISION, "intervention_hash": ("2ff9cdff091aef6f" if cell_id == "none" else "d" * 16),
                             "harness_config_hash": "2d9492f8" + "0" * 8, "producer_tree_sha": pin + "0" * (40 - len(pin)), "object_artifact_sha256": sha_, "served": {"max_model_len": 10752, "max_num_seqs": 44, "reduction_mode": "stacked"}}}

    def cell_block(cell_id: str, k: int) -> dict:
        items = {iid: {"scored": True, "value": (1.0 if i < k else 0.0), "at_cap": i == n - 1, "stop_reason": ("max_tokens" if i == n - 1 else "stop"), "completion_tokens": 500 + i} for i, iid in enumerate(items_ids)}
        return {"cell_key": f"{model}/{view}/{cell_id}", "model_key": model, "view": view, "cell": cell_id, "base_leg": f"{model}/{view}/{cell_id}@8192", "rung_leg": None,
                "ladder": {"exit": "clean", "n_at_cap": 1, "triggered": False, "n_eligible": 0, "escalated": False},
                "readings": {"merged": {"items": items, "n_correct": k, "n_scored": n, "n_at_cap_residual": 1, "n_from_rung": 0, "escalated": False, "same_as_r8192": True},
                             "r8192": {"n_correct": k, "n_scored": n}}}
    acc_none, acc_cell = k_none / n, k_cell / n
    reading = {"word": "not resolved at this cap", "reason": None, "agree_W_X": True,
               "point": {"stock": {"primary": acc_cell - acc_none}, "W": {"primary": acc_cell - acc_none}, "X": {"primary": acc_cell - acc_none}},
               "ci": {"stock": {"primary": [-0.1, 0.15]}, "W": {"primary": [-0.1, 0.15]}, "X": {"primary": [-0.1, 0.15]}}, "n_excluded_X": 0,
               "acc_by_leg_stock_primary": {"none": {"point": acc_none, "ci_low": acc_none - 0.1, "ci_high": acc_none + 0.05, "n_correct": k_none, "n_scored": n, "n_censored": 0},
                                            "cell": {"point": acc_cell, "ci_low": acc_cell - 0.1, "ci_high": acc_cell + 0.05, "n_correct": k_cell, "n_scored": n, "n_censored": 0}},
               "mcnemar_p_stock_primary": 0.625}
    d = {"experiment": "synthetic", "task": "inspect/gsm8k_paired200", "sha": pin, "analysis_code_sha": "5" * 40,
         "provenance": {"store_pin": pin, "analysis_code_sha": "5" * 40}, "gates": {"failed": []}, "bootstrap": {"B": 10000, "seed": 0},
         "sampling_rows": {"thinking_card": {model: {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}},
         "design": {"n_contrasts_designed": 1, "thinking_mode_by_model": {model: "on"}}, "held": [], "refused": [],
         "legs": {f"{model}/{view}/none@8192": leg("none", None), f"{model}/{view}/{cell}@8192": leg(cell, sha)},
         "cells": {f"{model}/{view}/none": cell_block("none", k_none), f"{model}/{view}/{cell}": cell_block(cell, k_cell)},
         "contrast_summary": [{"name": f"{model}/{view}: {cell} - none", "kind": "contrast", "model_key": model, "computable": True, "view": view, "cell": cell, "r8192": reading, "merged": reading}],
         "noise_floor_summary": [], "truncation_cost": [], "truncation": {}, "overlap": {}}
    return dict(path=_w(os.path.join(root, "report.json"), d), model=model, view=view, tag=tag, cell=cell, acc_none=acc_none, acc_cell=acc_cell, n=n, sha=sha, pin=pin)


# --------------------------------------------------------------------------------------------
# MATH-500 steer-pins r1 report — one comparison (gemma thinking_card), bare + lora_mix legs
# --------------------------------------------------------------------------------------------

def build_math500_report(root: str, model: str = "gemma4e4b", pin: str = "55e3cbdd0bfb5d53f8d62c50a099a3c1c105100b", n: int = 20, key_suffix: str = "") -> Dict[str, object]:
    def leg(cell: str, kind: str, sha: Optional[str], n_correct: int) -> dict:
        return {"leg": f"{model}/{cell}/thinking_card", "model_key": model, "cell": cell, "kind": kind, "mode": "thinking_card", "primary": True, "view": "main", "wave": "published",
                "cell_dir": f"{root}/pull/controls/{model}/{cell}/thinking_card/cells/{cell}__controls__t0__a0__2ff9cdff__6199715e{key_suffix}",
                "cell_key": f"{cell}__controls__t0__a0__2ff9cdff__6199715e{key_suffix}",
                "summary": {"acc": n_correct / n, "at_cap_rate": 0.0, "framework": "inspect_evals-0.18.0", "n": n, "n_at_cap": 0, "n_scored": n, "n_unscored": 0, "task": "inspect/math", "thinking": "on"},
                "outcomes_header": {"n_unscored": 0, "thinking": "on"}, "at_cap": {"n_loop": 0, "n_case_b_unterminated_at_cap": 0, "n_case_c_answer_at_cap": 0}, "unscored": {},
                "identity": {"cell_id": cell, "model_id": "google/gemma-4-E4B-it", "model_revision": REVISION, "intervention_hash": ("2ff9cdff091aef6f" if cell == "none" else "d" * 16),
                             "harness_config_hash": "6199715e" + "0" * 8, "producer_tree_sha": pin, "object_artifact_sha256": sha, "served": {"max_model_len": 20480, "reduction_mode": "stacked"}}}
    k_bare, k_mix = 17, 16
    d = {"task": "inspect/math", "sha": pin, "analysis_code_sha": "7" * 40, "provenance": {"store_pin": pin}, "gates": {"population": {"status": "PASSED"}}, "publication_gate": {"status": "PASS"},
         "sampling_rows": {"thinking_card": {model: {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}, "card": {}, "general": {}, "battery": {}},
         "legs": {f"{model}/none/thinking_card": leg("none", "bare", None, k_bare), f"{model}/constv2-mix-full__lora_r4_ce/thinking_card": leg("constv2-mix-full__lora_r4_ce", "lora_mix", SHA_A, k_mix)},
         "store_pins": {"pins": {f"{model}/none/thinking_card": pin, f"{model}/constv2-mix-full__lora_r4_ce/thinking_card": pin}}, "masked_primary": {"by_leg": {}},
         "comparisons": [{"name": f"{model}/thinking_card", "model_key": model, "mode": "thinking_card", "primary": True, "usable": True, "arms_absent": [],
                          "leg_keys": {"bare": f"{model}/none/thinking_card", "lora_mix": f"{model}/constv2-mix-full__lora_r4_ce/thinking_card"},
                          "estimands": {"primary": {"n_items": n, "acc_by_leg": {"bare": {"point": k_bare / n, "ci_low": 0.7, "ci_high": 0.95, "n_correct": k_bare, "n_scored": n, "n_censored": 0},
                                                                                "lora_mix": {"point": k_mix / n, "ci_low": 0.65, "ci_high": 0.92, "n_correct": k_mix, "n_scored": n, "n_censored": 0}},
                                                    "stats": {"lora_mix_minus_bare": {"point": (k_mix - k_bare) / n, "ci_low": -0.15, "ci_high": 0.05, "bound_low": -0.05, "bound_high": -0.05, "bound_ci_low": -0.15, "bound_ci_high": 0.05}}},
                                        "score_as_is": {"stats": {}}},
                          "verdicts": {"lora_mix_minus_bare": {"word": "not resolved at this cap", "reason": "ci covers zero", "conditions": {}}},
                          "mcnemar": {"lora_mix_minus_bare": {"primary": {"p_value": 1.0}}}}],
         "design": {"status_by_row": {f"{model}/alphasteer/thinking_card": {"status": "not_analysed", "wave": "steer", "reason": "synthetic", "model_key": model, "cell": "alphasteer", "mode": "thinking_card"}}},
         "eightarm": {"pooled_by_arm": {}}}
    return dict(path=_w(os.path.join(root, "report.json"), d), model=model, n=n, k_bare=k_bare, k_mix=k_mix, pin=pin)


# --------------------------------------------------------------------------------------------
# BCB/BFCL paper rows (already the paper's schema) — three rows
# --------------------------------------------------------------------------------------------

def build_bcb_parquet(root: str, name: str = "paper_rows_bcb_bfcl_200_synth.parquet") -> Dict[str, object]:
    import pandas as pd
    from sisteer.registry.adapters.declared_common import LONG_COLUMNS
    base = {c: None for c in LONG_COLUMNS if c not in ("saturated", "near_floor", "suggestive")}
    base.update(saturated=False, near_floor=False, suggestive=False)
    rows = []
    for cell, arm, fam, ot, val, k, delta, sha, note_store in (("none", "none", "none", "none", 0.5, 100, None, None, "none__controls__t0__a0__2ff9cdff__e95bd266"),
                                                                ("constv2-mix-full__lora_r4_ce_minus_ctl", "diff_lora", "difference", "lora", 0.55, 110, 5.0, SHA_B, "constv2-mix-full--lora_r4_ce_minus_ctl__controls__t0__a0__d2e8f411__e95bd266")):
        r = dict(base, source="bcb_bfcl_200", model="gemma4e4b", lane_kind="base→instruct", view="gemma4e4b", dose=1.0, cell_id=cell, arm=arm, family=fam, object_type=ot, component="capability",
                 eval_id="bigcodebench_200", eval="BigCodeBench-200", metric="acc", threshold=float("nan"), bon_n=None, value=val, k=k, n=200, ci_lo=val - 0.07, ci_hi=val + 0.07,
                 delta=(delta if delta is not None else float("nan")), delta_lo=(delta - 5 if delta is not None else float("nan")), delta_hi=(delta + 5 if delta is not None else float("nan")),
                 delta_p=(0.23 if delta is not None else float("nan")), delta_method=("paired exact McNemar + bootstrap" if delta is not None else float("nan")),
                 at_cap_share=0.0, decode_cap_tokens=4096, decode="sampled T=1.0 top_p=0.95 top_k=64 (model card), k=1, thinking OFF",
                 judge="bigcodebench verify (sandboxed unit tests)", status="graded", note=f"synthetic leg; store {note_store}; OFF cap 4096", provenance="/workspace-vast/synthetic/tinycap/bcb",
                 object_artifact_sha256=sha, thinking="off", held_out=(cell != "none"))
        rows.append(r)
    df = pd.DataFrame(rows)
    path = os.path.join(root, name)
    os.makedirs(root, exist_ok=True)
    df.to_parquet(path, index=False)
    return dict(path=path, n_rows=len(rows))


# --------------------------------------------------------------------------------------------
# EM agentic panel — one stem (gemma4e4b), leaking behaviour, arms none + const lora; organism panel
# --------------------------------------------------------------------------------------------

def build_em_panel(root: str, stem: str = "gemma4e4b", behaviour: str = "leaking", arms: Sequence[tuple] = (("none", 0.2, 100), ("si_constv2_mix_full__lora_r4_ce", 0.1, 100))) -> Dict[str, object]:
    import pandas as pd
    summ, pair = [], []
    for arm, harm, n in arms:
        k = round(harm * n)
        for metric, est in (("harmful", harm), ("harmful_reasoned", harm * 0.9), ("harmful_bare", 0.0), ("degenerate", 0.01), ("response_chars", 1500.0)):
            summ.append(dict(target=stem, arm=arm, family="agentic", behavior=behaviour, metric=metric, n=n, k=(k if metric == "harmful" else float("nan")), estimate=est,
                             ci_low=max(0.0, est - 0.05), ci_high=est + 0.05, ci_method="wilson95", boot_low=est - 0.04, boot_high=est + 0.04, note=None))
        summ.append(dict(target=stem, arm=arm, family="em8q", behavior="betley", metric="coherence", n=n, k=float("nan"), estimate=90.0 - (5 if arm != "none" else 0), ci_low=88.0, ci_high=92.0,
                         ci_method="bootstrap95", boot_low=88.0, boot_high=92.0, note=None))
        if arm != "none":
            pair.append(dict(target=stem, arm=arm, ref="none", family="agentic", behavior=behaviour, metric="harmful", n_pairs=n, mean_arm=harm, mean_none=arms[0][1], diff=harm - arms[0][1],
                             boot_low=harm - arms[0][1] - 0.05, boot_high=harm - arms[0][1] + 0.05, arm_only=5.0, none_only=15.0, mcnemar_p=0.04, note=None))
    d = os.path.join(root, stem)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(summ).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(pair).to_csv(os.path.join(d, "paired.csv"), index=False)
    return dict(dir=d, stem=stem, behaviour=behaviour, arms=arms)


def build_em_store(grade_root: str, stem: str, legs: Sequence[tuple], bank_objects: Mapping[str, str], subdir: str = "misalign",
                   thinking_mode: str = "off", wave_sha: str = synth.code_sha("e")) -> str:
    """A judged store's own identity record (``cells_full.json``): endpoint geometry / thinking / job, ``bank_objects`` shas per cell,
    and one ``legs`` entry per (cell_id, serving_identity, petri_run_key)."""
    d = os.path.join(grade_root, subdir, stem)
    full = {"target": stem, "wave_sha": wave_sha, "written_at": "2026-09-13T02:00:00+00:00",
            "endpoint": {"model_revision": REVISION, "max_model_len": 16384, "max_num_seqs": 64, "attested_max_model_len": 16384, "attested_max_num_seqs": 64,
                         "node": "node-7", "slurm_job_id": "424242", "thinking_mode": thinking_mode, "proxy_url": "http://node-7:8000",
                         "bank_objects": {cell: {"object_sha256": sha} for cell, sha in bank_objects.items()}},
            "cells_served": {"manifest": "synthetic"},
            "legs": [{"cell_id": cid, "cell_class": "class_a" if cid != "none" else "base", "kind": "lora_all" if cid != "none" else "none", "si_id": None, "object_path": None,
                      "virtual_model_name": f"petri__{cid}", "serving_identity": ident, "intervention_hash": f"{i:016x}", "harness_config_hash": "2bd33ad7d6e8d6c2",
                      "petri_run_key": key, "control_of": None} for i, (cid, ident, key) in enumerate(legs)]}
    return _w(os.path.join(d, "cells_full.json"), full)


def build_em_pins(path: str, model: str = "gemma4e4b") -> str:
    return _w(path, {model: {"base__constv2-mix-full__lora_r4_ce": {"object_sha256": SHA_A}, "base__constv2ctl2-full__lora_r4_ce": {"object_sha256": SHA_B},
                             "derived__constv2-mix-full__lora_r4_ce_minus_ctl": {"object_sha256": SHA_C}}})


# --------------------------------------------------------------------------------------------
# llama8b wave — Petri export + sidecar + view sidecar + verdicts + staged shas + MMLU cache
# --------------------------------------------------------------------------------------------

def build_llama8b(root: str) -> Dict[str, object]:
    stage = os.path.join(root, "stage")
    cache = os.path.join(root, "controls")
    tag, cell = "llama8b", "constv2-mix-full__lora_r4_ce"
    behs = {"cyber_bio_chem_capabilities": {"none": [7.0, 9.0, 6.0], cell: [6.0, 8.0, 6.0]}, "defer_to_users": {"none": [5.0, 6.0, 4.0], cell: [5.0, 5.0, 4.0]}}
    scen = ["s_alpha", "s_beta", "s_gamma"]

    def petri_cell(cid: str) -> dict:
        means = {b: sum(v[cid]) / 3 for b, v in behs.items()}
        pooled = sum(means.values()) / len(means)
        return {"label": cid, "group": "g", "arm": cid, "model": "llama8b", "view": "main", "lane_kind": "base-transfer", "eval_id": "petri_pooled",
                "petri": {"status": "COMPLETE", "rerun_state": "not_needed", "audited_items": 6, "audited_expected": 6, "petri_run_key": ("aaaa1111bbbb2222" if cid == "none" else "cccc3333dddd4444"),
                          "intervention_hash": ("2ff9cdff091aef6f" if cid == "none" else "d" * 16), "auditor_chain": {"sonnet_only": 6}, "auditor_refused": {"total": 0}, "judge_refused": None,
                          "pooled_score": pooled, "mean_of_behavior_means": pooled, "censored_bound": {"applies": False},
                          "behaviors": {b: {"mean": means[b], "n_items": 3, "n_expected": 3, "lo": means[b] - 1, "hi": means[b] + 1} for b in behs}}}
    vs_bare = {cell: {b: {"delta": round(sum(v[cell]) / 3 - sum(v["none"]) / 3, 1), "lo": -2.0, "hi": 0.5, "n_boot": 10000, "seed": 0, "n_paired": 3, "sign_test_p": 0.5} for b, v in behs.items()}}
    export = {"meta": {"model": "llama8b", "wave_sha": synth.code_sha("9"), "written_at": "2026-09-12T20:46:58.897533+00:00"},
              "models": {tag: {"label": "main", "model": "llama8b", "view": "main", "lane_kind": "base-transfer", "cells": {"none": petri_cell("none"), cell: petri_cell(cell)}, "petri_vs_bare": vs_bare}}}
    _w(os.path.join(stage, "transfer_stats_full22_llama8b.json"), export)
    side = {"meta": {}, "tags": {tag: {"cells": {cid: {"behaviors": {b: {"mean": sum(v[cid]) / 3, "per_scenario": dict(zip(scen, v[cid]))} for b, v in behs.items()},
                                                       "harness_config_hash": "dd1669d78419b354", "intervention_hash": ("2ff9cdff091aef6f" if cid == "none" else "d" * 16),
                                                       "serving_identity": ("proxy_vllm+passthrough+interleaved" if cid == "none" else "proxy_vllm+lens+interleaved")} for cid in ("none", cell)}}}}
    _w(os.path.join(stage, "petri_stats", "petri_full22_stats_llama8b.json"), side)
    from sisteer.registry.adapters import llama8b_wave as LW
    remote_root = LW.REMOTE_ROOT                        # the cluster root the view sidecar keys its dirs by (the adapter's own)
    store_none, store_cell = "none__controls__t0__a0__2ff9cdff__60fc2339", "constv2-mix-full--lora_r4_ce__controls__t0__a0__230e2961__10d69ad2"
    view_sidecar = {"serving_dirs": {f"{remote_root}/petri/{tag}/serving": {"serving_window": 16384, "max_num_seqs": 29, "job_id": "96213", "wave_sha": synth.code_sha("9")}},
                    "store_dirs": {f"{remote_root}/controls/{tag}/cells/{store_none}": {"serving_window": 2048, "job_ids": ["95830"], "wave_shas": [synth.code_sha("9")], "positive_control_verdict": None},
                                   f"{remote_root}/controls/{tag}/cells/{store_cell}": {"serving_window": 2048, "job_ids": ["95830"], "wave_shas": [synth.code_sha("9")], "positive_control_verdict": None}}}
    _w(os.path.join(stage, "view_sidecar.json"), view_sidecar)
    os.makedirs(stage, exist_ok=True)
    with open(os.path.join(stage, "positive_control_verdicts.tsv"), "w") as fh:
        fh.write("attacks_llama8b_actorattack_94320\t94320\t\tpositive_control_attacks\tPASS_WITH_ANCHOR_NONDETERMINISM\n")
    _w(os.path.join(stage, f".src_base__{cell}.json"), {"object.pt": SHA_A})
    _w(os.path.join(stage, "stage_plan.json"), [])
    _mmlu_store(os.path.join(cache, tag, "cells", store_none), "none", [1.0] * 6 + [0.0] * 4, None)
    _mmlu_store(os.path.join(cache, tag, "cells", store_cell), cell, [1.0] * 7 + [0.0] * 3, SHA_A[:16])
    # llama8b stores: the reader's _mmlu_items reads rows.jsonl + gen_stats + evidence; model ids must be the llama mirror
    return dict(stage=stage, cache=cache, tag=tag, cell=cell, behs=behs, scen=scen, sha=SHA_A)


# --------------------------------------------------------------------------------------------
# AlphaSteer — graded re-read + operating points + ladder (2 rungs) + defence record (1 rung) + refit MT summary
# --------------------------------------------------------------------------------------------

def build_alphasteer(root: str, model: str = "gemma4e4b", moved: bool = True, sign_control: bool = False, substrate_witness: bool = True) -> Dict[str, object]:
    """``substrate_witness``: the rung records name their standup's routing table and the defence summary the files its sampler
    resolution read, both on cluster A's shared root (the real stores do); False leaves both out (the substrate then reads unknown)."""
    store = os.path.join(root, "rebake")
    dstore = os.path.join(root, "defense")
    graded = os.path.join(root, "graded_reread.json")
    ladder_dir = os.path.join(store, "ladder_canon", model)
    lams = {1: 0.05, 2: 0.10}
    shas = {1: SHA_A, 2: SHA_B}
    op_rung = 2 if moved else 1
    layers = [12, 13, 14]
    # judge items: leg none rung 1 + lam1/lam2 arms; 12 items; labels
    labels = {("none", 1): ["full_refusal"] * 2 + ["full_compliance"] * 10, ("lam1", 1): ["full_refusal"] * 3 + ["full_compliance"] * 9, ("lam2", 2): ["full_refusal"] * 4 + ["full_compliance"] * 8}
    ji = []
    for (leg, rung), labs in labels.items():
        for it, lab in zip(XS_ITEMS, labs):
            ji.append({"cache_org": "lowprio", "item_id": it, "judge_model": "claude-sonnet-5", "label": lab, "leg": leg, "protocol_sha": "b8d28441cf71caec", "request_key": "r", "rubric_sha": "288e8c08b0c7bbfa", "rung_i": rung, "status": "judged"})
    _w(os.path.join(ladder_dir, "judge", "judge_items.jsonl"), ji, jsonl=True)
    rungs_rec = {}
    ladder_rungs = []
    for i, lam in lams.items():
        gens = [{"slice": "benign", "leg": "alphasteer", "prompt_id": it, "at_cap": False, "prompt": "lorem"} for it in XS_ITEMS]
        if i == 1:
            gens += [{"slice": "benign", "leg": "none", "prompt_id": it, "at_cap": False, "prompt": "lorem"} for it in XS_ITEMS]
        _w(os.path.join(ladder_dir, f"lam{i}", "generations.jsonl"), gens, jsonl=True)
        _w(os.path.join(ladder_dir, f"lam{i}", "rung_record.json"), {"schema": "alphasteer_f22_ladder_rung/1", "rung_i": i, "lambda": lam, "object_sha256": shas[i], "parent_object_sha256": SHA_C,
                                                                    "intervention_hash": f"{i}" * 16, "written_at": "2026-09-13T01:00:00+00:00", "temperature": 1.0,
                                                                    "cap_ladder": {"by_leg": {"alphasteer": {"512": {}, "2048": {}}, "none": {"512": {}, "2048": {}}}},
                                                                    "decode": {"sampler_keys_sent": ["temperature", "top_k", "top_p"], "temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed_base": 20260909, "trial": 0},
                                                                    "served": dict({"attested_max_model_len": 8704, "serving_config_sha256": "8" * 64, "wave_sha": synth.code_sha("a"), "job_id": "12345", "node": "node-13"},
                                                                                   **({"routing_table": f"/workspace-vast/TESTUSER/exp/objects/alphasteer_f22_rebake/{model}/ladder_canon/lam{i}/serve/routing_table.json"}
                                                                                      if substrate_witness else {})),
                                                                    "rho_integrity": {"checks": {"fit_layers_seen": layers, "fit_layers_expected": layers}}})
        full_a = labels[(f"lam{i}", i)].count("full_refusal")
        rungs_rec[str(i)] = {"lambda": lam, "object_sha256": shas[i], "anchor_generations_from_rung": 1,
                             "arm": {"full_n": full_a, "partial_n": 0, "at_cap_n": 0, "n_judged": 12, "n_abstained": 0, "at_cap_by_label": {"full": 0, "partial": 0, "compliance": 0}},
                             "anchor": {"full_n": 2, "partial_n": 0, "at_cap_n": 0, "n_judged": 12, "n_abstained": 0, "at_cap_by_label": {"full": 0, "partial": 0, "compliance": 0}},
                             "reporting_delta_pp": 100 * (full_a - 2) / 12, "gate_delta_pp": 100 * (full_a - 2) / 12}
        ladder_rungs.append({"i": i, "lambda": lam, "object_sha256": shas[i], "degenerate_share": 0.0})
    _w(os.path.join(ladder_dir, "lambda_ladder.json"), {"schema": "alphasteer_lambda_ladder/1", "model_key": model, "view": "it", "code_sha": CODE_SHA, "rungs": ladder_rungs,
                                                       "served": {"n_layers_selected": len(layers)}, "judge": {"domain": f"ladder_xstest_safe{len(XS_ITEMS)}_refusal_judged"}, "unsteered": {"degenerate_share": 0.0},
                                                       "chosen": {"rung": 1}})
    _w(graded, {"schema": "strength/graded_reread/3", "models": {model: {"ladder": "ladder_canon", "rungs": rungs_rec, "selection": {}}}})
    # the operating-point authority: moved -> rung 2, hash bound
    _w(os.path.join(dstore, "alphasteer_operating_points.json"), {"schema": "alphasteer_operating_points/1",
                                                                 "ruling": {"ruled_on": "2026-09-14", "rule": "among rungs whose steered-minus-anchor degeneration delta is within the clean band (0.10), the one with the HIGHEST defence", "rungs": {model: op_rung}},
                                                                 "authority": "THIS FILE", "targets": {model: {"rung": f"lam{op_rung}", "rung_i": op_rung, "lambda": lams[op_rung], "object_sha256": shas[op_rung], "moved": moved,
                                                                                                             "selection_outcome": "ruled_2026-09-14", "superseded": ({"rung_i": 1} if moved else None)}}})
    # defence record: one rung (op_rung) measured, static only
    fam = "pasttense"
    deg_leg = {fam: {det: {"n": 0, "share": 0.0, "denominator": 24} for det in ("classifier_share", "at_cap_share", "zlib_collapse_share", "contentless_share", "rep_ratio_loop_share")}}
    summ_path = os.path.join(dstore, "grades", f"alphasteer_f22_defense_{model}_lam{op_rung}_sampled_t0", "alphasteer_defense_summary.json")
    summ = {"schema": "alphasteer_defense_summary/1", "target": model, "trial": 0, "cap": 4096, "written_at": "2026-09-14T06:00:00+00:00",
            "decode_arm": {"decode_key": "7284364acbfb", "preset": {"temperature": 1.0, "top_k": 64, "top_p": 0.95}}, "dose": {"object_sha256": shas[op_rung], "lambda": lams[op_rung]},
            "judge": {"pinned_model": "claude-haiku-4-5", "pinned_rubric": "strongreject_v1"}, "decode": {"seed_base": 20260909}, "config": {"sha256": "5" * 64},
            "families": {fam: {"cells": {"none": {"n_goals": 12, "n_jailbroken": 6, "attack_success_at_0p5": 0.5, "degeneration": deg_leg[fam]},
                                         "alphasteer": {"n_goals": 12, "n_jailbroken": 3, "attack_success_at_0p5": 0.25, "degeneration": deg_leg[fam]}}}}}
    if substrate_witness:
        summ["dose"]["resolved_sampler"] = {"engine_defaults": {"files_searched": [
            f"/workspace-vast/TESTUSER/exp/objects/alphasteer_f22_rebake/{model}/ladder_canon/lam{op_rung}/serve/vllm_server.log",
            f"/workspace-vast/TESTUSER/exp/objects/alphasteer_f22_rebake/{model}/ladder_canon/lam{op_rung}/serve/standup.log"]}}
    _w(summ_path, summ)
    row = {"target": model, "label": f"rung {op_rung}", "private_run": f"alphasteer_f22_defense_{model}_lam{op_rung}_sampled_t0", "rung_token": f"lam{op_rung}", "sign_control": False, "lambda": lams[op_rung],
           "rung_i": op_rung, "object_sha256": shas[op_rung], "decode": {"role": "dose_consistent_and_cross_arm", "preset": {"temperature": 1.0, "top_k": 64, "top_p": 0.95}, "decode_key": "7284364acbfb"},
           "status": "partial: best-of-N not graded", "value": 0.75,
           "static": {"status": "measured", "summary_path": summ_path, "summary_sha256": sha256_file(summ_path),
                      "mean_defense": {"families": [fam], "alphasteer": 0.75, "none": 0.5, "delta_vs_none": 0.25},
                      "by_family": {fam: {"none": {"n_goals": 12, "attack_success_at_0p5": 0.5, "defense_at_0p5": 0.5}, "alphasteer": {"n_goals": 12, "attack_success_at_0p5": 0.25, "defense_at_0p5": 0.75}, "delta_vs_none": 0.25}},
                      "degeneration": {"alphasteer": deg_leg, "none": deg_leg}}, "bon": {"status": "absent"}}
    unst = {"target": model, "private_run": row["private_run"], "decode": {"decode_key": "7284364acbfb"}, "value": 0.5, "anchor_of_sign_control": False, "anchor_of_rung_i": op_rung, "degeneration": deg_leg}
    rows_, unsts = [row], [unst]
    if sign_control:
        # the sign control: -1 x rung 1, its own served object (a sha the ladder does NOT carry), its own same-pass unsteered row
        sign_run = f"alphasteer_f22_defense_{model}_lam_neg1_sampled_t0"
        sign_summ_path = os.path.join(dstore, "grades", sign_run, "alphasteer_defense_summary.json")
        sign_summ = dict(summ, dose={"object_sha256": SHA_SIGN, "lambda": -lams[1]},
                         families={fam: {"cells": {"none": {"n_goals": 12, "n_jailbroken": 6, "attack_success_at_0p5": 0.5, "degeneration": deg_leg[fam]},
                                                   "alphasteer": {"n_goals": 12, "n_jailbroken": 7, "attack_success_at_0p5": 7 / 12, "degeneration": deg_leg[fam]}}}})
        _w(sign_summ_path, sign_summ)
        sign_row = {"target": model, "label": "sign control (-1 x rung 1)", "private_run": sign_run, "rung_token": "lam_neg1", "sign_control": True, "lambda": -lams[1],
                    "rung_i": None, "mirrors_rung_i": 1, "object_sha256": SHA_SIGN, "decode": dict(row["decode"]), "status": "measured", "value": 5 / 12,
                    "static": {"status": "measured", "summary_path": sign_summ_path, "summary_sha256": sha256_file(sign_summ_path),
                               "mean_defense": {"families": [fam], "alphasteer": 5 / 12, "none": 0.5, "delta_vs_none": 5 / 12 - 0.5},
                               "by_family": {fam: {"none": {"n_goals": 12, "attack_success_at_0p5": 0.5, "defense_at_0p5": 0.5}, "alphasteer": {"n_goals": 12, "attack_success_at_0p5": 7 / 12, "defense_at_0p5": 5 / 12}, "delta_vs_none": 5 / 12 - 0.5}},
                               "degeneration": {"alphasteer": deg_leg, "none": deg_leg}}, "bon": {"status": "absent"}}
        rows_.append(sign_row)
        unsts.append({"target": model, "private_run": sign_run, "decode": {"decode_key": "7284364acbfb"}, "value": 0.5, "anchor_of_sign_control": True, "anchor_of_rung_i": None, "degeneration": deg_leg})
    _w(os.path.join(dstore, "alphasteer_defense_curves.json"), {"schema": "alphasteer_defense_curves/1", "written_at": "2026-09-14T06:20:58+00:00", "inputs": {"trial": 0},
                                                              "targets": {model: {"ladder_record": {"model_key": model, "schema": "alphasteer_lambda_ladder/1"}, "decodes": {"arms": [{"decode_key": "7284364acbfb", "role": "dose_consistent_and_cross_arm", "preset": {"temperature": 1.0, "top_k": 64, "top_p": 0.95}}]},
                                                                                  "rows": rows_, "unsteered_rows": unsts}}})
    return dict(store=store, defence_store=dstore, graded=graded, model=model, lams=lams, shas=shas, op_rung=op_rung, layers=layers, ladder_dir=ladder_dir, sign_sha=SHA_SIGN if sign_control else None)


def build_refit_mt_summary(refit_root: str, complete: bool = True) -> str:
    fams = {}
    for f in ("crescendo", "pair", "actorattack"):
        n_graded = 259 if complete else 200
        fams[f] = {"n": 259, "k": {"0.25": 200, "0.5": 180, "0.75": 90}, "asr": {"0.25": 200 / 259, "0.5": 180 / 259, "0.75": 90 / 259}, "n_capped": 0,
                   "completeness_gate": {"n_expected": 259, "n_graded": n_graded, "passed": complete}}
    return _w(os.path.join(refit_root, "mt", "strongreject", "alphasteer_mt_strongreject_summary.json"),
              {"lane": "synthetic", "cell": "alphasteer", "judge": "claude-haiku-4-5", "rubric": "v1", "pool_stamp": {"pool": "highprio"}, "n_expected_per_family": 259, "families": fams, "outputs": {}})


# --------------------------------------------------------------------------------------------
# relu_native — one model (llama8b), one attack aggregate + paired records (multi-turn admitted; gsm8k lens refused, native admitted)
# --------------------------------------------------------------------------------------------

def build_relu_native(root: str, allowance: str) -> Dict[str, object]:
    rs = os.path.join(root, "rs")
    paired = os.path.join(root, "paired")
    model, cell = "llama8b", "constv2-mix-full__psr_ce_all"
    g = os.path.join(rs, f"grades_relu_native_bu_{model}", model)
    per_goal = {gid: {"best_score": (0.75 if i % 3 else 0.0), "capped": False, "n_target_forwards": 10, "queries_to_first_success": (5 if i % 3 else -1)} for i, gid in enumerate(GOALS)}
    k = sum(1 for v in per_goal.values() if v["best_score"] > 0.5)
    _w(os.path.join(g, "two_si_actorattack_aggregate.json"), {"attack": "actorattack", "attack_tier": "multi_turn", "written_at": "2026-09-16T14:32:04+00:00", "serving_engine": "lora_native", "target": model,
                                                              "cells": {cell: {"arm": "psr_ce_all", "serving_engine": "lora_native", "serving_identity": "proxy_vllm+lora+interleaved", "decode_cap_tokens": 4096, "served_alpha": 1.0, "si_id": "si_constitution", "class": "B",
                                                                               "attack_metrics": {"asr": {"0.5": k / 12}, "n_jailbroken": k, "n_goals": 12, "n_graded": 12, "n_capped": 0, "per_goal": per_goal},
                                                                               "completion": {"gated": {"attacker_engine": {"attacker_model": "huihui-ai/Huihui-Qwen3.6-27B-abliterated"}}}}}})
    _w(os.path.join(g, "grading_provenance.json"), {"grader": {"code_sha": CODE_SHA}})
    _w(os.path.join(g, "completion.json"), {"ok": True})

    def paired_rec(ev: str, anchor_engine: str, status: str, allow: Optional[str], refusals: List[str], stats: Optional[dict], metric: str, sv: float, av: float) -> dict:
        return {"schema": "paired_vs_anchor/1", "model": model, "eval": ev, "cell": cell, "view": "main", "anchor_cell": "none", "metric": metric, "unit": "pp", "pairing_unit": "goal",
                "steered": {"root": g, "cell": cell, "serving_identity": "proxy_vllm+lora+interleaved", "serving_engine": "lora_native", "object_sha": SHA_A, "view": "main", "value": sv, "n": 12,
                            "store_decode": {"sampler_normalised": {"temperature": 0.6, "top_p": 0.9}}, "leg": {"leg": f"{model}/main/{cell}@8192", "producer": {"tree_sha": CODE_SHA}}},
                "anchor": {"root": f"{rs}/grades_llama8b_llama8b", "cell": "none", "serving_identity": ("proxy_vllm+passthrough+interleaved" if anchor_engine == "lens" else "proxy_vllm+lora+interleaved"),
                           "serving_engine_resolved": anchor_engine, "view": "main", "value": av, "n": 12, "cap": 4096, "aggregate": f"{rs}/grades_llama8b_llama8b/llama8b/two_si_{ev}_aggregate.json"},
                "anchor_engine": anchor_engine, "steered_engine": "lora_native", "anchor_view": "main", "anchor_root": f"{rs}/grades_llama8b_llama8b", "pairing_allowance": allow, "pairing_admitted": True,
                "pairing_reason": "synthetic", "status": status, "refusals": refusals, "written_at": "2026-09-16T20:00:00+00:00", "stats": stats}
    st_mt = {"n_pairs": 12, "steered_mean": 100 * k / 12, "anchor_mean": 75.0, "delta": 100 * k / 12 - 75.0, "ci95": [-20.0, 5.0], "p_value": 0.3, "p_method": "exact McNemar", "bootstrap": {"B": 10000, "seed": 1, "kind": "paired percentile bootstrap"}}
    _w(os.path.join(paired, f"{model}__actorattack__{cell}.json"), paired_rec("actorattack", "lens", "ok", allowance, [], st_mt, "asr@0.5", 100 * k / 12, 75.0))
    _w(os.path.join(paired, f"{model}__gsm8k__{cell}.json"), paired_rec("gsm8k", "lens", "refused", allowance, ["gsm8k: cross-engine contrast: llama8b/main/none (lens) vs steered (lora_native) is not admissible"], None, "gsm8k_acc_merged_stock_primary", 87.5, 86.5))
    st_g = {"n_pairs": 200, "steered_mean": 87.5, "anchor_mean": 86.5, "n_correct_by_leg": {"none": 173, "cell": 175}, "delta": 1.0, "ci95": [-4.0, 6.0], "p_value": 0.8, "p_method": "exact McNemar", "bootstrap": {"B": 10000, "seed": 1, "kind": "paired conditional bootstrap"}}
    _w(os.path.join(paired, f"{model}__gsm8k_native_none__{cell}.json"), paired_rec("gsm8k_native_none", "lora_native", "ok", None, [], st_g, "gsm8k_acc_merged_stock_primary", 87.5, 86.5))
    return dict(rs=rs, paired=paired, model=model, cell=cell, k=k, n=12)
