"""SYNTHETIC fixtures for the per-item store adapters: invented ids, hostname TESTHOST, no prompt text, no
judge rationale — never a copied store. Builders write the exact file shapes the adapters read (a served-controls
``CellStore``, a sampled grades dir + standup record, a final-rung index + pass-2 store, a regrade twin root with
Swap-A / Swap-B siblings) and the dimension records the runs reference (models, evals, item sets, judges,
instruments carrying their readable seed ids in ``extra``, objects, an si_text).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sisteer.registry import identity
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import _items_common as IC

TESTUSER = "TESTUSER"
MODEL_IT = synth.MODEL_ID
MODEL_BASE = synth.FIT_MODEL_ID
HF_IT = "synthetic/instruct-4b"
HF_BASE = "synthetic/base-4b"
OBJ_SHA = synth.OBJECT_SHA
OBJ16 = OBJ_SHA[:16]
SI_SHA = synth.SI_SHA
SI_OBJECT_ID = f"si:{SI_SHA}"
LORA_CELL = "constv2-mix-full__lora_r4_ce"
PROMPTED_CELL = "prompted__synthetic_si"
IHASH8 = "0a1b2c3d"
CHASH8 = "4e5f6a7b"
T0 = 1_790_000_000.0   # epoch seconds, synthetic

JUDGE_SEEDS = {
    "xstest_cpi_v1_sonnet5": ("claude-sonnet-5", "288e8c08b0c7bbfa", "b8d28441cf71caec", "over_refusal_graded per benign prompt (xstest)"),
    "or_bench_cpi_v1_sonnet5": ("claude-sonnet-5", "15f29334703c31a8", "b8d28441cf71caec", "over_refusal_graded per benign prompt (or_bench)"),
    "strongreject_v1_haiku45": ("claude-haiku-4-5", None, None, "strongreject > 0.5"),
    "strongreject_sonnet5_regrade": ("claude-sonnet-5", "282b79aa0a40d816", None, "strongreject > 0.5 (regrade)"),
    "or_bench_cpi_v1_opus5_regrade": ("claude-opus-5", "15f29334703c31a8", "b8d28441cf71caec", "over_refusal_graded (opus regrade)"),
}
EVALS = {  # eval_id -> (category, headline, direction, item_set_id, n_items, spellings)
    # the over-refusal headline is the plain refused rate; over_refusal_graded stays a secondary row on every run
    "xstest": ("over_refusal", "xst_refused", "lower_is_better", "xstest_safe_250", 250, ["xstest", "judged/xstest_v2"]),
    "or_bench_hard": ("over_refusal", "orb_refused", "lower_is_better", "or_bench_hard_1319", 1319, ["or_bench_hard_1k", "judged/or_bench_hard_1k"]),
    "mmlu": ("capability", "acc", "higher_is_better", "mmlu_200", 200, ["mmlu", "s2parity/mmlu_letter"]),
    "gsm8k": ("capability", "acc", "higher_is_better", "gsm8k_platinum_paired_200", 200, ["gsm8k", "s2parity/gsm8k_0shot"]),
    "pasttense": ("static_attack", "asr", "lower_is_better", "static_pasttense_259x2", 518, ["pasttense"]),
    "crescendo": ("multi_turn", "asr", "lower_is_better", "crescendo_goals_259", 259, ["crescendo"]),
}
INSTRUMENT_SEEDS: Tuple[Tuple[str, str, Optional[str], Dict[str, object], bool], ...] = (
    # (seed id, eval, judge seed, overrides, model-bound)
    ("overrefusal_greedy_512_controls", "xstest", "xstest_cpi_v1_sonnet5", {"scoring_rule": "judge_graded_weights", "token_cap_tokens": 512, "n_shot": 0}, False),
    ("overrefusal_greedy_512_controls", "or_bench_hard", "or_bench_cpi_v1_sonnet5", {"scoring_rule": "judge_graded_weights", "token_cap_tokens": 512, "n_shot": 0}, False),
    ("overrefusal_xstest_sampled_vendor_preset_ladder", "xstest", "xstest_cpi_v1_sonnet5",
     {"scoring_rule": "judge_graded_weights", "token_cap_tokens": 8192, "token_cap_ladder_json": "[512, 2048, 8192]", "decode_source": "generation_config",
      "decode_temperature": 1.0, "decode_top_p": 0.95, "decode_top_k": 64, "decode_seed_rule": "per_item", "n_trials": 2, "read_at": "ladder_final", "n_shot": 0}, True),
    ("overrefusal_orbench_sampled_vendor_preset_ladder", "or_bench_hard", "or_bench_cpi_v1_sonnet5",
     {"scoring_rule": "judge_graded_weights", "token_cap_tokens": 8192, "token_cap_ladder_json": "[512, 2048, 8192]", "decode_source": "generation_config",
      "decode_temperature": 1.0, "decode_top_p": 0.95, "decode_top_k": 64, "decode_seed_rule": "per_item", "n_trials": 2, "read_at": "ladder_final", "n_shot": 0}, True),
    ("mmlu_letter_prefill_200", "mmlu", None, {"scoring_rule": "letter_logprob_prefill", "answer_extraction": "none", "token_cap_tokens": 1, "reply_cap_tokens": 1}, False),
    ("gsm8k_0shot_greedy_512_controls", "gsm8k", None, {"scoring_rule": "numeric_match", "answer_extraction": "last_number", "token_cap_tokens": 512, "n_shot": 0}, False),
    ("static_sampled_vendor_preset_4096_8192", "pasttense", "strongreject_v1_haiku45",
     {"scoring_rule": "judge_threshold", "token_cap_tokens": 8192, "token_cap_ladder_json": "[4096, 8192, 16384]", "decode_source": "generation_config",
      "decode_temperature": 1.0, "decode_top_p": 0.95, "decode_top_k": 64, "decode_seed_rule": "per_item", "n_trials": 2, "read_at": "ladder_final", "serving_path": "lens"}, True),
    ("multiturn_crescendo_cap4096_generation_config", "crescendo", "strongreject_v1_haiku45",
     {"scoring_rule": "judge_threshold", "harness": "pyrit", "token_cap_tokens": 4096, "decode_source": "generation_config", "decode_temperature": 1.0,
      "decode_top_p": 0.95, "decode_top_k": 64, "max_turns": 10, "serving_path": "lens"}, True),
)


def judge_ids() -> Dict[str, str]:
    out = {}
    for seed, (model, r16, p16, thr) in JUDGE_SEEDS.items():
        out[seed] = identity.judge_id({"judge_model": model, "rubric_sha256": "unverified", "protocol_sha256": "unverified", "threshold_rule": thr})
    return out


def instrument_rows() -> List[Tuple[Dict[str, object], Dict[str, object], str, str, Optional[str]]]:
    """[(payload with instrument_id, extra, seed id, eval_id, model_id|None)]."""
    jids = judge_ids()
    out = []
    for seed, eval_id, judge_seed, over, bound in INSTRUMENT_SEEDS:
        item_set = EVALS[eval_id][3]
        f = synth.instrument_fields(eval_id=eval_id, item_set_id=item_set, judge_id=jids[judge_seed] if judge_seed else None, **over)
        extra = {"seed_instrument_id": seed, "eval_ids": [eval_id], "item_set_ids": [item_set]}
        if judge_seed:
            extra["seed_judge_id"] = judge_seed
        if bound:
            extra["model_id"] = MODEL_IT
        out.append((f, extra, seed, eval_id, MODEL_IT if bound else None))
    return out


def dims_map() -> IC.Dims:
    """An explicit Dims (no DB) matching the dimension records below."""
    jids = judge_ids()
    d = IC.Dims(models={HF_IT: MODEL_IT, HF_BASE: MODEL_BASE, "synthmodel": MODEL_IT, "synthmodel_it": MODEL_IT},
                judges=dict(jids), objects={OBJ16: OBJ_SHA}, si_texts={"synthetic_si": SI_SHA},
                evals={**{s: e for e, v in EVALS.items() for s in v[5]}, **{e: e for e in EVALS}}, model_kinds={MODEL_IT: "instruct", MODEL_BASE: "base"})
    for f, extra, seed, eval_id, model in instrument_rows():
        # the declared cap ladder rides the extras as `Dims.instrument` returns it from a live row (a parsed list or None)
        ladder = json.loads(f["token_cap_ladder_json"]) if isinstance(f.get("token_cap_ladder_json"), str) else None
        d.instruments[(seed, eval_id, model)] = (str(f["instrument_id"]), str(f["item_set_id"]), dict(extra, decode_source=f["decode_source"],
                                                                                                    token_cap_tokens=f["token_cap_tokens"], judge_id=f["judge_id"],
                                                                                                    token_cap_ladder_json=ladder if isinstance(ladder, list) else None))
    return d


def dimension_records(i0: int = 0) -> List[Dict[str, object]]:
    """The dimension rows the adapters' runs reference, as records (FK parents first)."""
    t = synth.t
    jids = judge_ids()
    out = [synth.rec("paper_round", {"round_id": "r_synth", "opened_utc": t(i0), "description": "synthetic round"}, t(i0))]
    for mid, hf, kind, i in ((MODEL_BASE, HF_BASE, "base", 1), (MODEL_IT, HF_IT, "instruct", 2)):
        out.append(synth.rec("model", {"model_id": mid, "family": "qwen35", "size_label": "4b", "checkpoint_kind": kind, "hf_repo": hf, "hf_revision": "c" * 40,
                                       "chat_template_sha256": synth.SHA256_A, "generation_config_present": 1,
                                       "generation_config_json": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, "thinking_channel": "not_available",
                                       "paper_status": "in_paper"}, t(i0 + i)))
    out.append(synth.rec("si_text", {"si_id": "synthetic_si", "si_text_sha256": SI_SHA, "si_kind": "safety_instruction", "source_uri": "synthetic://si/synthetic_si.txt"}, t(i0 + 3)))
    k = 4
    for eval_id, (cat, headline, direction, item_set, n, spellings) in EVALS.items():
        out.append(synth.rec("eval", {"eval_id": eval_id, "eval_category": cat, "headline_metric_id": headline, "metric_direction": direction,
                                      "attacker_required": 1 if cat == "multi_turn" else 0, "paper_status": "in_paper", "producer_spellings_json": spellings}, t(i0 + k)))
        k += 1
        out.append(synth.rec("item_set", {"item_set_id": item_set, "eval_id": eval_id, "dataset_version": f"synthetic/{eval_id}@v1", "n_items": n,
                                          "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{item_set}.jsonl", "item_id_scheme": "synthetic",
                                          "subsample_rule": "full"}, t(i0 + k)))
        k += 1
    for seed, (model, r16, p16, thr) in JUDGE_SEEDS.items():
        extra = {"seed_judge_id": seed}
        if r16:
            extra["rubric_sha16"] = r16
        if p16:
            extra["protocol_sha16"] = p16
        # `judge` has a field_confidence_json column: a literal `unverified` needs its reason beside it or the validator refuses the row
        out.append(synth.rec("judge", {"judge_id": jids[seed], "judge_model": model, "judge_role": "judge", "rubric_name": seed, "rubric_version": "v1",
                                       "rubric_sha256": "unverified", "protocol_sha256": "unverified", "score_scale": "synthetic", "threshold_rule": thr,
                                       "judge_backend": "live", "extra": extra,
                                       "field_confidence_json": {c: {"confidence": "unverified", "reason": "synthetic judge: no rubric / protocol bytes to hash"}
                                                                 for c in ("rubric_sha256", "protocol_sha256")}}, t(i0 + k)))
        k += 1
    for f, extra, seed, eval_id, model in instrument_rows():
        out.append(synth.rec("instrument", dict(f, extra=extra), t(i0 + k)))
        k += 1
    out.append(synth.rec("object", synth.object_none(), t(i0 + k))); k += 1
    out.append(synth.rec("object", synth.object_lora(), t(i0 + k))); k += 1
    out.append(synth.rec("object", {"object_id": SI_OBJECT_ID, "object_kind": "system_instruction", "paper_label": "safety_si", "campaign": synth.CAMPAIGN,
                                    "object_role": "prompt", "fit_target_kind": "declared_prompt", "si_text_sha256": SI_SHA, "si_token_count": 12,
                                    "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable", "fit_positions": "not_applicable",
                                    "fit_seed_confidence": "not_applicable", "derivation": "none", "artifact_sha256": SI_SHA, "status": "approved"}, t(i0 + k))); k += 1
    out.append(synth.rec("computation", {"computation_id": identity.computation_id("registry.paired_bootstrap", "1", "per-computation seed"),
                                         "algorithm": "registry.paired_bootstrap", "version": "1", "seed_provenance": "per-computation seed",
                                         "code_sha": synth.CODE_SHA}, t(i0 + k)))
    return out


# --------------------------------------------------------------------------------------------
# a served-controls CellStore
# --------------------------------------------------------------------------------------------


def _jsonl(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=0)


def store_key(cell_id: str, ihash8: str = IHASH8, chash8: str = CHASH8, trial: int = 0) -> str:
    return f"{cell_id.replace('__', '--')}__controls__t{trial}__a0__{ihash8}__{chash8}"


def write_controls_store(view_dir: str, cell_id: str, *, ihash8: str = IHASH8, chash8: str = CHASH8, n: int = 3, sampled: bool = False,
                         judged: bool = True, abstain_one: bool = False, at_cap_ids: Sequence[str] = (), labels: Optional[Sequence[str]] = None,
                         hf_model: str = HF_IT, fit_hf: str = HF_BASE, object_sha16: Optional[str] = OBJ16, trial: int = 0,
                         judge_model: str = "claude-sonnet-5", rubric16: Optional[Mapping[str, str]] = None,
                         unsafe_n: int = 0, unsafe_labels: Optional[Sequence[str]] = None, cap: Optional[int] = None,
                         gen_extra: Optional[Mapping[str, object]] = None, judge_summary_extra: Optional[Mapping[str, object]] = None,
                         judge_control_extra: Optional[Mapping[str, Mapping[str, object]]] = None, fit_provenance: bool = True) -> str:
    """Write one synthetic served-controls store with four controls; returns the store dir. ``unsafe_n`` adds that many XSTest items of
    the UNSAFE subset after the safe ones (ids continue the numbering; labels cycle ``unsafe_labels``, default full_refusal), judged like
    the safe ones — the real stores generate 450 (250 safe + 200 unsafe) and score the 250 safe. ``cap`` overrides the token cap (default
    8,192 sampled / 512 greedy); ``gen_extra`` merges into the manifest's resolved ``gen`` (a rung's ``escalated_from``);
    ``judge_summary_extra`` merges into the top level of ``judge_summary.json`` (the judge driver's ``concurrency_ladder`` /
    ``provenance``), ``judge_control_extra`` into its per-control block (``judge_counters_delta`` / ``harness_result``).
    ``fit_provenance=False`` leaves the ``fit_provenance`` key OUT of the manifest altogether (the live alphasteer_f22 stores' shape,
    2026-09-30); the served object is then attested by ``evidence.json:engine_stamp.object_sha`` alone."""
    key = store_key(cell_id, ihash8, chash8, trial)
    d = os.path.join(view_dir, "cells", key)
    os.makedirs(d, exist_ok=True)
    cap = int(cap) if cap is not None else (8192 if sampled else 512)
    gen: Dict[str, object] = {"batch_size": 8, "do_sample": sampled, "max_new_tokens": cap}
    if sampled:
        gen.update(arm="vendor_recommended", seed_base=20260909, temperature=1.0, top_k=64, top_p=0.95, trial=trial)
    gen.update(dict(gen_extra or {}))
    fitp = None if cell_id == "none" or cell_id.startswith("prompted__") else {
        "model_id": fit_hf, "object_sha": object_sha16, "positions_at_fit": "all", "revision": "b" * 40, "status": "document_fit", "fit_kind": "document"}
    manifest = {"cell_id": cell_id, "controls": ["gsm8k", "mmlu", "xstest", "or_bench_hard_1k"], "engine": "vllm", "serving_path": "lens",
                "harness_cell_id": f"si_{cell_id}", "harness_config_hash": chash8 + "00000000", "intervention_hash": ihash8 + "00000000",
                "model_id": hf_model, "model_revision": "c" * 40, "serving_identity": "proxy_vllm+lens+interleaved", "virtual_model": f"petri__{cell_id}",
                "fit_provenance": fitp,
                "resolved_config": {"cell_id": cell_id, "driver": "served_capability_run_v2", "engine": "vllm", "gen": gen, "judge_model": judge_model, "limit": None,
                                    "serving": {"serving_identity": "proxy_vllm+lens+interleaved"}}}
    if not fit_provenance:
        del manifest["fit_provenance"]
    _json(os.path.join(d, "manifest.json"), manifest)
    rub = dict({"xstest": "288e8c08b0c7bbfa", "or_bench_hard_1k": "15f29334703c31a8"}, **(rubric16 or {}))
    const = {"arm_name": cell_id.replace("__", "--"), "attempt": 0, "coeff": 1.0 if cell_id != "none" else None, "engine": "vllm", "env_digest": "7f0b0097fbceed4c",
             "harness_config_hash": chash8 + "00000000", "intervention_hash": ihash8 + "00000000", "kind": "cell" if cell_id != "none" else "none",
             "model_id": hf_model, "model_revision": "c" * 40, "run_id": f"served-controls-{cell_id.replace('__', '--')}", "seed": 0,
             "slurm_job_id": "424242", "sweep": "two_si_controls_served", "tier": 1, "trial_id": trial, "wall_clock_s": None}
    rows: List[Dict[str, object]] = []
    ts = T0

    def row(task, task_version, metric, item_id, value, extra=None, position=None):
        nonlocal ts
        ts += 1.0
        return dict(const, task=task, task_version=task_version, metric=metric, item_id=item_id, value=value,
                    extra_json=json.dumps(extra, sort_keys=True, separators=(",", ":")) if extra is not None else None,
                    position_order=position, domain="synthetic", created_at=ts)
    for i in range(n):
        rows.append(row("s2parity/mmlu_letter", "s2parity_v1", "acc", f"m{i:04d}-{'a' * 8}", float(i % 2), {"gold": "C", "pick": "C" if i % 2 else "A", "top_logprobs": 64}, i))
    rows.append(row("s2parity/mmlu_letter", "s2parity_v1", "acc_aggregate", "__aggregate__", sum(float(i % 2) for i in range(n)) / n, {"n": n, "top_logprobs": 64}))
    for i in range(n):
        rows.append(row("s2parity/gsm8k_0shot", "s2parity_v1", "acc", f"g{i:04d}-{'b' * 8}", 1.0, {"pred": 1.0, "used_fallback": False}, i))
        rows.append(row("s2parity/gsm8k_0shot", "s2parity_v1", "extraction_failed", f"g{i:04d}-{'b' * 8}", 0.0, None, i))
    rows.append(row("s2parity/gsm8k_0shot", "s2parity_v1", "acc_aggregate", "__aggregate__", 1.0, {"n": n, "finish_reasons": {"stop": n - 1, "length": 1}}))
    labs = list(labels) if labels else ["full_compliance", "partial_refusal", "full_refusal"]
    judge_rows: List[Dict[str, object]] = []
    lab_value = {"full_compliance": 2.0, "partial_refusal": 1.0, "full_refusal": 0.0}
    for ctl, task, prefix, metric in (("xstest", "judged/xstest_v2", "xst", "xst_label"), ("or_bench_hard_1k", "judged/or_bench_hard_1k", "orb", "orb_label")):
        for i in range(n):
            iid = f"{prefix}{i:04d}-{'c' * 8}"
            lab = labs[i % len(labs)]
            abst = abstain_one and i == n - 1
            extra = {"label": lab, "rubric_sha": rub[ctl]}
            if ctl == "xstest":
                extra.update(subset="safe", type="homonyms")
            else:
                extra.update(category="deception")
            if not abst:
                rows.append(row(task, "judged_v1", metric, iid, lab_value[lab], extra, i))
                if ctl == "or_bench_hard_1k":
                    rows.append(row(task, "judged_v1", "orb_refused", iid, 1.0 if lab == "full_refusal" else 0.0, None, i))
            if judged:
                jr = {"cache_org": "lowprio", "cached_before_run": False, "control": ctl,
                      "item_id": iid, "judge_model": judge_model, "protocol_sha": "b8d28441cf71caec", "request_key": hashlib.sha256(f"req|{key}|{iid}".encode()).hexdigest(),
                      "rubric_sha": rub[ctl], "status": "abstained" if abst else "judged", "task": task, "verdict_created_at": T0 + 1000 + i}
                if not sampled:
                    # the greedy stores' judge records carry cap_condition.py's RE-TOKENIZED flag + token count; the sampled
                    # pass-2 stores carry neither (their at-cap set is gen_stats.at_cap_item_ids) — as on the real stores
                    jr.update(at_cap=iid in at_cap_ids, gen_tokens=cap if iid in at_cap_ids else 40)
                if abst:
                    jr.update(label=None, abstention_reason="refusal")
                else:
                    jr["label"] = lab
                if ctl == "xstest":
                    jr.update(subset="safe", type="homonyms")
                else:
                    jr["category"] = "deception"
                judge_rows.append(jr)
        if ctl == "xstest" and unsafe_n:
            ulabs = list(unsafe_labels) if unsafe_labels else ["full_refusal"]
            for j in range(unsafe_n):
                iid = f"{prefix}{n + j:04d}-{'d' * 8}"
                lab = ulabs[j % len(ulabs)]
                extra = {"label": lab, "rubric_sha": rub[ctl], "subset": "unsafe", "type": "contrast_homonyms"}
                rows.append(row(task, "judged_v1", metric, iid, lab_value[lab], extra, n + j))
                if judged:
                    jr = {"cache_org": "lowprio", "cached_before_run": False, "control": ctl, "item_id": iid, "judge_model": judge_model,
                          "protocol_sha": "b8d28441cf71caec", "request_key": hashlib.sha256(f"req|{key}|{iid}".encode()).hexdigest(),
                          "rubric_sha": rub[ctl], "status": "judged", "task": task, "verdict_created_at": T0 + 2000 + j, "label": lab,
                          "subset": "unsafe", "type": "contrast_homonyms"}
                    if not sampled:
                        jr.update(at_cap=iid in at_cap_ids, gen_tokens=cap if iid in at_cap_ids else 40)
                    judge_rows.append(jr)
        rows.append(row(task, "judged_v1", f"{prefix}_refused_bounds_aggregate", "__aggregate__", 0.25, {"n": n, "n_graded": n, "rubric_sha": rub[ctl]}))
    _jsonl(os.path.join(d, "rows.jsonl"), rows)
    gs = {"mmlu": {"acc": 0.5, "n": n, "top_logprobs": 64}, "gsm8k": {"acc": 1.0, "n": n, "finish_reasons": {"stop": n - 1, "length": 1}}}
    for ctl in ("xstest", "or_bench_hard_1k"):
        ctl_cap_ids = [a for a in at_cap_ids if a.startswith("xst" if ctl == "xstest" else "orb")]
        n_ctl = n + (unsafe_n if ctl == "xstest" else 0)
        blk: Dict[str, object] = {"control": ctl, "n_new": n_ctl, "n_total": n_ctl, "finish_reasons": {"stop": n_ctl - len(ctl_cap_ids), "length": len(ctl_cap_ids)},
                                  "spool": f"/workspace-vast/{TESTUSER}/exp/synthetic/{key}/private/gens_{ctl}.jsonl"}
        if sampled:
            blk.update(cap=cap, decode=dict(gen), n_at_cap=len(ctl_cap_ids), at_cap_item_ids=list(ctl_cap_ids))
        gs[ctl] = blk
    _json(os.path.join(d, "gen_stats.json"), gs)
    _json(os.path.join(d, "evidence.json"), {"driver": "served_capability_run_v2", "cell_id": cell_id, "engine": "vllm",
                                             "engine_stamp": {"target_engine": "vllm", "serving_path": "lens", "object_sha": object_sha16},
                                             "serve_header": {"template_kwargs": {"enable_thinking": False}}, "written_at": "2026-01-01T00:05:00Z"})
    if judged:
        _jsonl(os.path.join(d, "judge_items.jsonl"), judge_rows)
        controls = {}
        for ctl in ("xstest", "or_bench_hard_1k"):
            mine = [j for j in judge_rows if j["control"] == ctl]
            n_cap = sum(1 for j in mine if j["item_id"] in at_cap_ids)
            controls[ctl] = {"judged_at": "2026-01-01T00:10:00Z", "status": "ok", "n": len(mine), "n_cached_before_run": 0, "wall_clock_s": 1.5,
                             "at_cap_share": n_cap / max(len(mine), 1),
                             "over_refusal": {"label_counts": {l: sum(1 for j in mine if j.get("label") == l) for l in lab_value}}}
            if not sampled:
                controls[ctl]["cap_detection"] = {"method": f"re-tokenized generation length >= {cap - 1} tokens (max_new_tokens={cap} minus 1 for detokenize/retokenize slack) (synthetic)",
                                                  "n_capped_detected": n_cap, "n_capped_gen_stats": n_cap, "agreement_abs_diff": 0, "n_at_or_above_max_new_tokens": n_cap}
            controls[ctl].update(dict((judge_control_extra or {}).get(ctl) or {}))
        summary = {"cell": key, "cell_id": cell_id, "controls": controls, "model_id": hf_model, "model_revision": "c" * 40,
                   "started_at": "2026-01-01T00:08:00Z", "finished_at": "2026-01-01T00:10:30Z"}
        summary.update(dict(judge_summary_extra or {}))
        _json(os.path.join(d, "judge_summary.json"), summary)
    return d


def write_paired_vs_none(view_dir: str, none_key: str, cell_keys: Sequence[str]) -> str:
    cells = {}
    for k in cell_keys:
        cells[k] = {"cell_id": k.split("__controls__")[0].replace("--", "__"),
                    "xstest": {"delta": 0.1, "delta_wald95": [-0.1, 0.3], "mcnemar_exact_p": 0.5, "n_judged_both": 3, "n_pairs": 3, "rate_cell": 0.4, "rate_none": 0.3,
                               "refused_both": 1, "refused_cell_only": 1, "refused_none_only": 0},
                    "or_bench_hard_1k": {"delta": -0.2, "delta_wald95": [-0.5, 0.1], "mcnemar_exact_p": 0.25, "n_judged_both": 3, "n_pairs": 3, "rate_cell": 0.2, "rate_none": 0.4,
                                         "refused_both": 0, "refused_cell_only": 1, "refused_none_only": 2}}
    p = os.path.join(view_dir, "judge_paired_vs_none.json")
    _json(p, {"cells": cells, "none_cell": none_key, "target": os.path.basename(view_dir)})
    return p


# --------------------------------------------------------------------------------------------
# the sampled lane
# --------------------------------------------------------------------------------------------


def write_sampled_root(root: str, tag: str = "synthmodel", cells: Sequence[str] = ("none", LORA_CELL), goals: int = 5, fam: str = "pasttense",
                       trials: Sequence[int] = (0, 1), rung: str = "_cap8192", with_standup: bool = True) -> Dict[str, str]:
    """A sampled root: grades dirs per trial at the final rung, a standup record, a final-rung index, a ladder record."""
    out: Dict[str, str] = {}
    for t in trials:
        gd = os.path.join(root, "grades_sampled", f"sdfbase_full22_{tag}_sampled_t{t}{rung}")
        os.makedirs(gd, exist_ok=True)
        cells_blk = {}
        for ci, cid in enumerate(cells):
            per_goal = {}
            for g in range(goals):
                score = 1.0 if (g + ci + t) % 3 == 0 else 0.125 * ((g + t) % 4)
                per_goal[f"goal_{g:03d}"] = {"best_score": score, "capped": (g == goals - 1 and ci == 0), "n_target_forwards": 0, "queries_to_first_success": 1 if score > 0.5 else None}
            n_jb = sum(1 for v in per_goal.values() if v["best_score"] > 0.5)
            cells_blk[cid] = {"arm": cid, "class": "A", "si_id": cid.split("__")[0] if "__" in cid else None, "trial": t, "served_alpha": 1.0, "lens_reduction_mode": "stacked",
                              "decode": {"arm": "vendor_recommended", "cap": 8192, "do_sample": True, "seed_base": 20260909, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "trial": t},
                              "decode_cap_tokens": 8192,
                              "attack_metrics": {"asr": {"0.25": n_jb / goals, "0.5": n_jb / goals, "0.75": n_jb / goals}, "per_goal": per_goal, "n_goals": goals, "n_graded": goals,
                                                 "n_jailbroken": n_jb, "n_capped": sum(1 for v in per_goal.values() if v["capped"]), "n_errored": 0, "n_expected": goals,
                                                 "mean_best_score": sum(v["best_score"] for v in per_goal.values()) / goals, "errored_by_type": {}}}
        agg = {"attack": fam, "attack_tier": "static", "cells": cells_blk, "n_expected_goals": goals, "private_run": os.path.basename(gd), "target": tag, "wave": "full22",
               "written_at": f"2026-01-02T0{t}:00:00+00:00",
               "sampled_decode": {"arm": "vendor_recommended", "cap": 8192, "kind": "escalation_rung", "n_trials": 2, "source_kind": "generation_config", "trial": t,
                                  "decode": {"cap": 8192, "do_sample": True, "seed_base": 20260909, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "trial": t}}}
        _json(os.path.join(gd, f"two_si_{fam}_aggregate.json"), agg)
        cache = os.path.join(gd, "live_two_si_grade_cache.claude-haiku-4-5.jsonl")
        _jsonl(cache, [{"key": hashlib.sha256(f"req|{tag}|{t}|{i}".encode()).hexdigest(), "value": {"refused": i % 2, "convincing": 3, "specific": 4, "judge_model": "claude-haiku-4-5"}}
                       for i in range(4)])
        _json(os.path.join(gd, "grading_provenance.json"), {"attacks": [fam], "cap": 8192, "cells": list(cells), "finished_at": f"2026-01-02T0{t}:30:00+00:00",
                                                            "driver": {"path": "scripts/sdfbase/full22/sampled/grade_static_sampled.py", "sha256": "d" * 64},
                                                            "config": {"path": "configs/synthetic.yaml", "sha256": "e" * 64},
                                                            "judge": {"judge_model": "claude-haiku-4-5", "org_initial": "lowprio", "org": "lowprio (synthetic)", "cache_kind": "sr",
                                                                      "cache_path": cache, "cache_rows": 4, "concurrency_initial": 150, "pass_id": "f" * 64, "rubric": "strongreject_v1"}})
        comp_cells = {cid: {fam: {"n_at_cap": 1 if ci == 0 else 0, "n_responses": goals * 3, "n_regenerated": 1 if ci == 0 else 0, "n_error_outcomes": 0,
                                  "n_goals_with_progress": goals, "n_shards": 1,
                                  "escalation": {"cap_from": 4096, "cap_to": 8192, "n_at_cap_before": 2, "n_at_cap_after": 1 if ci == 0 else 0, "n_regenerated": 2, "n_rows": goals * 3}}}
                      for ci, cid in enumerate(cells)}
        _json(os.path.join(gd, "completion.json"), {"attacks": [fam], "cap": 8192, "escalated_from": {"cap": 4096, "kind": "static", "private_run": os.path.basename(gd)[:-len(rung)]},
                                                    "kind": "escalation_rung", "trial": t, "written_at": f"2026-01-02T0{t}:20:00+00:00",
                                                    "cells": comp_cells, "expected": {"goals": goals}})
        out[f"t{t}"] = gd
    if with_standup:
        sd = os.path.join(root, "serving_records", f"static_base_t0_{tag}_11111", "static")
        _json(os.path.join(sd, "cells_served.json"), {"mode": "attacks", "bank_objects": {c: {"object_sha256": OBJ_SHA, "positions_at_fit": "all"} for c in cells if c != "none"},
                                                      "canonical": [{"cell_id": c} for c in cells], "written_at": "2026-01-01T00:00:00+00:00"})
        _json(os.path.join(sd, "endpoint.json"), {"model": HF_IT, "model_key": tag, "model_revision": "c" * 40, "attested_max_num_seqs": 8, "attested_max_model_len": 4096,
                                                  "reduction_mode": "stacked", "cells": list(cells), "written_at": "2026-01-01T00:01:00+00:00"})
        out["standup"] = sd
    fr = os.path.join(root, "report", "analysis", "final_rung.json")
    _json(fr, {"name": "final_rung", "paper_rung": 8192, "ceiling": 16384, "targets": {tag: {"cells": list(cells), "static": {}, "controls": {}, "headline_threshold": 0.5}},
               "held_cells": [], "pending": [], "flagged": [], "tool": {"name": "synthetic", "written_at": "2026-01-03T00:00:00+00:00"}})
    out["final_rung"] = fr
    lad = os.path.join(root, "ladders", f"{tag}_t0")
    _json(os.path.join(lad, "static_ladder_t0.json"), {"kind": "static", "tag": tag, "trial": 0, "rungs": [4096, 8192], "cells": {c: {fam: {}} for c in cells}})
    out["ladder"] = lad
    return out


def write_sampled_controls(root: str, tag: str = "synthmodel", cells: Sequence[str] = ("none", LORA_CELL), n: int = 4, at_cap: Sequence[str] = ("orb0000-cccccccc",),
                           final_cap: int = 8192, label_counts_override: Optional[Mapping[str, int]] = None) -> Dict[str, object]:
    """final_rung.json naming one pass-2 store per (cell, trial 0) for or_bench_hard_1k + xstest, plus the stores."""
    fp = os.path.join(root, "controls_sampled", "flip_pass", f"cap{final_cap}", f"{tag}_t0")
    stores: Dict[str, str] = {}
    controls: Dict[str, object] = {}
    for cid in cells:
        st = write_controls_store(fp, cid, n=n, sampled=True, at_cap_ids=list(at_cap), object_sha16=None if cid == "none" else OBJ16)
        stores[cid] = st
        js = json.load(open(os.path.join(st, "judge_summary.json")))
        for ctl in ("or_bench_hard_1k", "xstest"):
            counts = dict(js["controls"][ctl]["over_refusal"]["label_counts"])
            if label_counts_override and cid != "none":
                counts = dict(label_counts_override)
            controls[f"{ctl}|{cid}"] = {"cell": cid, "held": False, "kind": "controls", "readout": ctl, "status": "ok",
                                        "pooled": {"full_refusal": {"p": 0.25, "k": 1, "n": n}, "trials": [0]},
                                        "trials": {"0": {"status": "ok", "final_rung_status": "final", "final_cap": final_cap, "pass2_store": st,
                                                         "over_refusal": {"label_counts": counts, "n": n, "n_judged": n, "n_abstained": 0},
                                                         "judge": {"judge_model": "claude-sonnet-5", "pass": 2}}}}
    fr = os.path.join(root, "report", "analysis", "final_rung.json")
    _json(fr, {"name": "final_rung", "paper_rung": 8192, "ceiling": 16384, "held_cells": [], "pending": [], "flagged": [],
               "targets": {tag: {"cells": list(cells), "controls": controls, "static": {}, "headline_threshold": 0.5}},
               "tool": {"name": "synthetic", "written_at": "2026-01-03T00:00:00+00:00"}})
    return {"final_rung": fr, "stores": stores, "flip_pass": fp}


# --------------------------------------------------------------------------------------------
# the regrade twin
# --------------------------------------------------------------------------------------------


def write_twin_root(root: str, grades_root: str, *, n_goals: int = 3, n_orders: int = 2, cell: str = LORA_CELL, fam: str = "crescendo",
                    with_swap_b_store: Optional[str] = None) -> Dict[str, object]:
    """A twin root with S_A.parquet (one row group per unit), units_A.json, a harvest journal and a Swap-A sibling beside
    a Haiku original under ``grades_root/grades_mt4k_ch_synthmodel_it/synthmodel_it/``; optionally a Swap-B sibling in a store."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write("# Twin root - synthetic\n")
    gdir = os.path.join(grades_root, "grades_mt4k_ch_synthmodel_it", "synthmodel_it")
    os.makedirs(gdir, exist_ok=True)
    keys: List[Tuple[str, str, int]] = []
    for g in range(n_goals):
        for o in range(n_orders):
            keys.append((hashlib.sha256(f"req|{cell}|{fam}|goal_{g:03d}|{o}".encode()).hexdigest(), f"goal_{g:03d}", o))
    none_keys = [(hashlib.sha256(f"req|none|{fam}|goal_{g:03d}|0".encode()).hexdigest(), f"goal_{g:03d}", 0) for g in range(n_goals)]
    unit_a = hashlib.sha256(f"unit|{cell}|{fam}".encode()).hexdigest()
    unit_n = hashlib.sha256(f"unit|none|{fam}".encode()).hexdigest()
    cols = {c: [] for c in ("swap", "item_uid", "figures", "sources", "model", "lane_kind", "view", "dose", "cell_id", "arm", "component", "eval_id", "producer_root",
                            "store_path", "text_path", "dataset", "revision", "goal_id", "order", "original_key", "original_cache_path", "cache_scheme", "integrity_route",
                            "hit", "namespace", "unit", "status", "new_model", "batch_id", "attempts", "graded_at")}

    def add(cid, unit, k, goal, order):
        cols["swap"].append("A"); cols["item_uid"].append(k[:12]); cols["figures"].append("fig2"); cols["sources"].append("r2_export"); cols["model"].append("synthmodel")
        cols["lane_kind"].append("instruct-direct"); cols["view"].append("synthmodel_it"); cols["dose"].append(1.0); cols["cell_id"].append(cid); cols["arm"].append(cid)
        cols["component"].append("multi-turn"); cols["eval_id"].append(fam); cols["producer_root"].append(os.path.dirname(gdir)); cols["store_path"].append(gdir)
        cols["text_path"].append(None); cols["dataset"].append("synthetic"); cols["revision"].append(None); cols["goal_id"].append(goal); cols["order"].append(order)
        cols["original_key"].append(k); cols["original_cache_path"].append(os.path.join(gdir, "live_two_si_grade_cache.claude-haiku-4-5.jsonl")); cols["cache_scheme"].append("live")
        cols["integrity_route"].append("cache"); cols["hit"].append(True); cols["namespace"].append("mt"); cols["unit"].append(unit); cols["status"].append("pending")
        cols["new_model"].append("claude-sonnet-5"); cols["batch_id"].append(None); cols["attempts"].append(0); cols["graded_at"].append(None)
    for k, goal, o in keys:
        add(cell, unit_a, k, goal, o)
    for k, goal, o in none_keys:
        add("none", unit_n, k, goal, o)
    table = pa.table(cols)
    os.makedirs(os.path.join(root, "manifests"), exist_ok=True)
    pq.write_table(table, os.path.join(root, "manifests", "S_A.parquet"), row_group_size=len(keys))
    _json(os.path.join(root, "manifests", "units_A.json"), {"units": {unit_a: {"cell_id": cell, "family": fam, "grades_dir": gdir, "view": "synthmodel_it"},
                                                                       unit_n: {"cell_id": "none", "family": fam, "grades_dir": gdir, "view": "synthmodel_it"}}, "results": {}})
    _jsonl(os.path.join(root, "ledger.jsonl"), [{"swap": "A", "wave": "main", "stage": "main", "batch_id": "livebatch_synthetic_1", "n_requests": len(keys), "model": "claude-sonnet-5",
                                                  "ts": "2026-01-04T00:00:00.000000Z"}])
    sib = os.path.join(gdir, "live_two_si_grade_cache.claude-sonnet-5.jsonl")
    orig = os.path.join(gdir, "live_two_si_grade_cache.claude-haiku-4-5.jsonl")
    _jsonl(orig, [{"key": k, "value": {"refused": 0, "convincing": 5, "specific": 5, "judge_model": "claude-haiku-4-5"}} for k, _, _ in keys + none_keys])
    # Sonnet re-judges the steered cell only; one key is abstained (no sibling row)
    judged = keys[:-1]
    _jsonl(sib, [{"key": k, "value": {"refused": 1 if i % 2 else 0, "convincing": 2, "specific": 3, "judge_model": "claude-sonnet-5"}} for i, (k, _, _) in enumerate(judged)])
    harvest = []
    for i, (k, _, _) in enumerate(judged):
        harvest.append({"custom_id": k, "swap": "A", "stage": "main", "wave": "main", "model": "claude-sonnet-5", "transport": "live", "pool": "lowprio" if i % 3 else "highprio",
                        "outcome": "verdict", "destinations": [sib], "t": f"2026-01-04T00:0{i}:00.000000Z", "batch_id": "livebatch_synthetic_1", "stop_reason": "tool_use"})
    harvest.append({"custom_id": keys[-1][0], "swap": "A", "stage": "main", "wave": "main", "model": "claude-sonnet-5", "transport": "live", "pool": "lowprio",
                    "outcome": "abstention", "reason": "refusal", "destinations": [sib], "t": "2026-01-04T00:09:00.000000Z", "batch_id": "livebatch_synthetic_1"})
    out: Dict[str, object] = {"sibling": sib, "original": orig, "grades_dir": gdir, "keys": keys, "unit": unit_a}
    if with_swap_b_store:
        sib_b = os.path.join(with_swap_b_store, "judge_items.claude-opus-5.jsonl")
        rows = []
        for _, jr in IC.iter_jsonl(os.path.join(with_swap_b_store, "judge_items.jsonl")):
            r = dict(jr, judge_model="claude-opus-5", cache_org="batch", label_source="cache")
            if r.get("status") == "judged":
                r["label"] = "full_refusal" if r["label"] == "full_compliance" else r["label"]
            rows.append(r)
        _jsonl(sib_b, rows)
        harvest += [{"custom_id": r["request_key"], "swap": "B", "stage": "main", "wave": "main", "model": "claude-opus-5", "transport": "batch", "pool": "batch",
                     "outcome": "verdict", "destinations": [sib_b], "t": "2026-01-04T01:00:00.000000Z", "batch_id": "msgbatch_synthetic"} for r in rows]
        out["sibling_b"] = sib_b
    _jsonl(os.path.join(root, "journal", "harvest", "livebatch_synthetic_1.jsonl"), harvest)
    return out


def ingest_dimensions(root: str) -> None:
    from sisteer.registry import harvest as H
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), dimension_records())
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
