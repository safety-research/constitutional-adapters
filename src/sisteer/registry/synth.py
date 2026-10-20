"""Synthetic record corpora for tests and the sizing gates (invented ids, hostname TESTHOST, never a
real transcript, prompt or generation).

``seed_dimensions()`` returns the dimension records a run needs (paper_round, model, si_text, eval,
item_set, judge, instrument, object ``none``, one fitted object, computation); ``run_records()``
returns a run's lifecycle (run, states, judgment, metrics); ``item_records()`` its per-item rows;
``write_corpus()`` lays a whole corpus out under a registry root's ``records/`` directory.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import identity
from . import records as R

TESTHOST = "TESTHOST"


def code_sha(tag: str) -> str:
    """A synthetic 40-hex code sha, distinct per ``tag`` and never a placeholder: a ``run`` record whose ``code_sha`` is forty identical
    hex digits is refused by name (``validate.run_code_sha_refusal``), so the corpus states derived shas where it used to state
    ``"a" * 40`` -- ``code_sha("a")`` where a test wrote ``"a" * 40``, ``code_sha("b")`` for its second sha, the old letter kept as the tag
    so a fixture reads as it did. Deterministic (a digest over the tag), never a commit of any checkout."""
    out = hashlib.sha256(f"synthetic code sha {tag} (tests; never a commit)".encode("utf-8")).hexdigest()[:40]
    assert len(set(out)) > 1, out
    return out


CODE_SHA = code_sha("a")
SHA256_A = hashlib.sha256(b"synthetic-a").hexdigest()
SHA256_B = hashlib.sha256(b"synthetic-b").hexdigest()
SI_SHA = hashlib.sha256(b"synthetic system instruction text").hexdigest()
JUDGE_ID = identity.judge_id({"judge_model": "judge-model-x", "rubric_sha256": SHA256_A, "protocol_sha256": SHA256_B,
                              "threshold_rule": "score > 0.5"})
JUDGE_ID_2 = identity.judge_id({"judge_model": "judge-model-y", "rubric_sha256": SHA256_A, "protocol_sha256": SHA256_B,
                                "threshold_rule": "score > 0.5"})
OBJECT_SHA = hashlib.sha256(b"synthetic lora object bytes").hexdigest()
MODEL_ID = "synthmodel_4b_it"
FIT_MODEL_ID = "synthmodel_4b_base"
EVAL_ID = "syntheval"
ITEM_SET_ID = "syntheval_200"
CAMPAIGN = "synthcampaign"


def producer(script: str = "tests.synthetic", substrate: str = "grading_host", session: Optional[str] = None) -> Dict[str, object]:
    return {"script": script, "code_sha": CODE_SHA, "session": session, "host": TESTHOST, "substrate": substrate}


def t(i: int, base: str = "2026-01-01T00:00:00.000Z") -> str:
    """Monotone synthetic timestamps: base + i seconds (i may exceed 59; carries into minutes/hours)."""
    ms = R.utc_to_epoch_ms(base) + i * 1000
    return R.epoch_ms_to_utc(ms)


def instrument_fields(**over) -> Dict[str, object]:
    f: Dict[str, object] = {
        "eval_id": EVAL_ID, "item_set_id": ITEM_SET_ID, "harness": "bespoke_sisteer", "harness_version": "0.1",
        "harness_config_hash": "h" * 8, "n_shot": None, "prompt_template_sha256": SHA256_A, "cot_prompting": None,
        "scoring_rule": "judge_threshold", "answer_extraction": "judge", "token_cap_tokens": 512,
        "token_cap_ladder_json": "[512]", "ladder_promotion_rule": None, "read_at": "fixed", "reply_cap_tokens": None,
        "context_window_tokens": 8192, "decode_source": "greedy", "decode_temperature": 0.0, "decode_top_p": None,
        "decode_top_k": None, "decode_min_p": None, "decode_repetition_penalty": None, "decode_presence_penalty": None,
        "decode_seed_rule": "none", "thinking_mode": "off", "n_trials": 1, "bon_n": None, "max_turns": None,
        "serving_path": "lora_native", "applied_positions_mode": "all", "attacker_model_id": None,
        "attacker_decode_json": None, "judge_id": JUDGE_ID, "prefill_scheme": None,
    }
    f.update(over)
    f["instrument_id"] = identity.instrument_id(f)
    return f


INSTRUMENT = instrument_fields()
INSTRUMENT_ID = str(INSTRUMENT["instrument_id"])


# ---- a RULE-SCORED instrument (a capability eval: no judge; the headline metric is filed at generation) --------------
CAP_EVAL_ID = "synthcap"
CAP_ITEM_SET_ID = "synthcap_200"
RULE_INSTRUMENT = instrument_fields(eval_id=CAP_EVAL_ID, item_set_id=CAP_ITEM_SET_ID, harness="lm_eval", scoring_rule="exact_match",
                                    answer_extraction="regex", judge_id=None)
RULE_INSTRUMENT_ID = str(RULE_INSTRUMENT["instrument_id"])


def rec(record_type: str, payload: Mapping[str, object], written_utc: str, prod: Optional[Mapping[str, object]] = None,
        deterministic: bool = True) -> Dict[str, object]:
    rid = identity.adapter_record_id(payload, record_type) if deterministic else identity.producer_record_id()
    return R.make_record(record_type, payload, prod or producer(), rid, written_utc)


def seed_dimensions(i0: int = 0, with_fitted_object: bool = True) -> List[Dict[str, object]]:
    out = [
        rec("paper_round", {"round_id": "r_synth", "opened_utc": t(i0), "description": "synthetic round"}, t(i0)),
        rec("model", {"model_id": FIT_MODEL_ID, "family": "qwen35", "size_label": "4b", "checkpoint_kind": "base",
                      "hf_repo": "synthetic/base-4b", "hf_revision": "b" * 40, "chat_template_sha256": SHA256_A,
                      "generation_config_present": 1, "generation_config_json": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                      "thinking_channel": "not_available", "paper_status": "in_paper"}, t(i0 + 1)),
        rec("model", {"model_id": MODEL_ID, "family": "qwen35", "size_label": "4b", "checkpoint_kind": "instruct",
                      "hf_repo": "synthetic/instruct-4b", "hf_revision": "c" * 40, "chat_template_sha256": SHA256_A,
                      "generation_config_present": 1, "generation_config_json": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
                      "thinking_channel": "available", "paper_status": "in_paper"}, t(i0 + 2)),
        rec("si_text", {"si_id": "synthetic_si", "si_text_sha256": SI_SHA, "si_kind": "constitution_short",
                        "source_uri": "synthetic://si/synthetic_si.txt"}, t(i0 + 3)),
        rec("eval", {"eval_id": EVAL_ID, "eval_category": "static_attack", "headline_metric_id": "asr",
                     "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "in_paper"}, t(i0 + 4)),
        rec("item_set", {"item_set_id": ITEM_SET_ID, "eval_id": EVAL_ID, "dataset_version": "synthetic/syntheval@v1",
                         "n_items": 200, "items_sha256": SHA256_B, "source_uri": "synthetic://items/syntheval_200.jsonl",
                         "item_id_scheme": "q<index>", "subsample_rule": "full"}, t(i0 + 5)),
        rec("judge", {"judge_id": JUDGE_ID, "judge_model": "judge-model-x", "judge_role": "judge", "rubric_name": "synthetic_rubric",
                      "rubric_version": "1", "rubric_sha256": SHA256_A, "protocol_sha256": SHA256_B, "score_scale": "0-1",
                      "threshold_rule": "score > 0.5", "judge_backend": "live"}, t(i0 + 6)),
        rec("judge", {"judge_id": JUDGE_ID_2, "judge_model": "judge-model-y", "judge_role": "judge", "rubric_name": "synthetic_rubric",
                      "rubric_version": "1", "rubric_sha256": SHA256_A, "protocol_sha256": SHA256_B, "score_scale": "0-1",
                      "threshold_rule": "score > 0.5", "judge_backend": "batch"}, t(i0 + 7)),
        rec("instrument", INSTRUMENT, t(i0 + 8)),
        rec("object", object_none(), t(i0 + 9)),
        rec("computation", {"computation_id": identity.computation_id("registry.paired_bootstrap", "1", "per-computation seed"),
                            "algorithm": "registry.paired_bootstrap", "version": "1", "seed_provenance": "per-computation seed",
                            "code_sha": CODE_SHA}, t(i0 + 10)),
    ]
    if with_fitted_object:
        out.append(rec("object", object_lora(), t(i0 + 11)))
    return out


def seed_rule_scored_dimensions(i0: int = 20) -> List[Dict[str, object]]:
    """The capability eval, its item set and a rule-scored instrument (no judge) — written beside :func:`seed_dimensions`."""
    return [
        rec("eval", {"eval_id": CAP_EVAL_ID, "eval_category": "capability", "headline_metric_id": "acc",
                     "metric_direction": "higher_is_better", "attacker_required": 0, "paper_status": "in_paper"}, t(i0)),
        rec("item_set", {"item_set_id": CAP_ITEM_SET_ID, "eval_id": CAP_EVAL_ID, "dataset_version": "synthetic/synthcap@v1",
                         "n_items": 200, "items_sha256": SHA256_B, "source_uri": "synthetic://items/synthcap_200.jsonl",
                         "item_id_scheme": "q<index>", "subsample_rule": "full"}, t(i0 + 1)),
        rec("instrument", RULE_INSTRUMENT, t(i0 + 2)),
    ]


def object_none() -> Dict[str, object]:
    return {"object_id": "none", "object_kind": "none", "paper_label": "none", "campaign": CAMPAIGN, "object_role": "baseline_none",
            "fit_target_kind": "none", "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable",
            "fit_positions": "not_applicable", "fit_seed_confidence": "not_applicable", "derivation": "none",
            "artifact_sha256": "none", "status": "approved"}


def object_si(si_sha: str = SI_SHA, paper_label: str = "constitution_si") -> Dict[str, object]:
    """The prompted object of the synthetic system instruction (:data:`SI_SHA`): ``object_id = si:<sha>`` (identity.object_id's rule),
    a declared prompt with no fit and no artifact beyond the text's sha -- the shape the seeds loader mints for every si_text."""
    return {"object_id": identity.object_id("system_instruction", si_text_sha256=si_sha), "object_kind": "system_instruction",
            "paper_label": paper_label, "campaign": CAMPAIGN, "object_role": "prompt", "fit_target_kind": "declared_prompt",
            "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable", "fit_positions": "not_applicable",
            "fit_seed_confidence": "not_applicable", "derivation": "none", "si_text_sha256": si_sha, "artifact_sha256": si_sha,
            "status": "approved"}


def object_lora(sha: str = OBJECT_SHA, paper_label: str = "constitution_lora", role: str = "constitution") -> Dict[str, object]:
    return {"object_id": sha, "object_kind": "lora", "paper_label": paper_label, "campaign": CAMPAIGN, "object_role": role,
            "fit_target_kind": "corpus_document_fit", "fit_corpus_id": "synthetic-corpus", "fit_corpus_sha256": SHA256_B,
            "fit_model_id": FIT_MODEL_ID, "fit_thinking_render": "off", "fit_objective": "ce", "fit_positions": "all",
            "fit_layers_json": [4, 5, 6], "fit_layer_index_convention": "zero_based_decoder_block", "fit_layer_count": 3,
            "fit_layer_selection_rule": "listed_in_config", "fit_rank": 4, "fit_alpha": 8.0, "fit_seed": 0,
            "fit_seed_confidence": "recorded", "fit_lr": 1e-3, "fit_lr_selected_by": "search", "fit_code_sha": CODE_SHA,
            "derivation": "none", "artifact_sha256": sha, "artifact_bytes": 1234, "status": "approved",
            "fit_hyperparameters_json": {"lr": 1e-3, "epochs": 3}}


def run_payload(run_key: str, object_key: str = "none", object_id: str = "none", i: int = 0, **over) -> Dict[str, object]:
    rid = identity.run_id(CAMPAIGN, run_key, object_key, EVAL_ID, INSTRUMENT_ID)
    p: Dict[str, object] = {
        "run_id": rid, "campaign": CAMPAIGN, "producer_run_key_legacy": run_key, "producer_object_key_legacy": object_key,
        "lifecycle_state": "queued", "queued_utc": t(i), "model_id": MODEL_ID, "object_id": object_id,
        "transfer_kind": "none" if object_id == "none" else "base_to_instruct", "eval_id": EVAL_ID, "item_set_id": ITEM_SET_ID,
        "instrument_id": INSTRUMENT_ID, "instrument_match": "match", "harness": "bespoke_sisteer", "harness_version": "0.1",
        "scoring_rule": "judge_threshold", "prompt_template_sha256": SHA256_A, "decode_seed_rule": "none",
        "decode_source": "greedy", "token_cap_tokens": 512, "thinking_mode": "off", "serving_path": "lora_native",
        "applied_positions_mode": "all" if object_id != "none" else "none",
        "multi_turn_steering_scope": "single_turn" if object_id != "none" else "not_applicable",
        "code_sha": CODE_SHA, "substrate": "cluster_a", "substrate_source": "recorded",
        "n_items_planned": 200, "store_uri": f"synthetic:///store/{run_key}",
        # the synthetic corpus stands for runs a paper reader files: it carries a source, as every reader's run does (a source-less run
        # is registry-only and never reaches the paper frame — paper.paper_frame_exclusion)
        "lane_source_legacy": "synth_source",
    }
    if object_id != "none":
        # a fitted object served at its operating point (the roster's `operating_point` rule needs the FACT filed)
        p.update(strength_coefficient=1.0, strength_nominal=1.0, strength_nominal_source="filed", is_operating_point=1)
    p.update(over)
    return p


def run_records(run_key: str, i0: int, object_key: str = "none", object_id: str = "none", n_items: int = 200,
                judge: str = JUDGE_ID, prod: Optional[Mapping[str, object]] = None, through: str = "judged",
                deterministic: bool = True, **over) -> Tuple[str, List[Dict[str, object]]]:
    """A run's lifecycle records through ``through`` ∈ {queued, running, generated, judged}; ``over`` overrides (or, with a value of
    None, REMOVES) fields of the run payload -- a fitted run without its `is_operating_point` stamp, a strength at another nominal."""
    p = run_payload(run_key, object_key, object_id, i0, **over)
    for k, v in over.items():
        if v is None:
            p.pop(k, None)
    rid = str(p["run_id"])
    out = [rec("run", p, t(i0), prod, deterministic),
           rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0), prod, deterministic)]
    if through == "queued":
        return rid, out
    out.append(rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log",
                                 "scheduler_state": "RUNNING"}, t(i0 + 1), prod, deterministic))
    out.append(rec("run_patch", {"run_id": rid, "fields": {"node_hostname": TESTHOST, "cluster_job_id": "4242", "gpu_count": 1},
                                 "verify": {"serving_path": "lora_native"}}, t(i0 + 2), prod, deterministic))
    if through == "running":
        return rid, out
    out.append(rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                 "evidence_uri": f"synthetic:///store/{run_key}/results.json"}, t(i0 + 3), prod, deterministic))
    out.append(rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": n_items, "store_completion_artifact_uri":
                                 f"synthetic:///store/{run_key}/results.json", "store_completion_sha256": SHA256_B}}, t(i0 + 4), prod, deterministic))
    if through == "generated":
        return rid, out
    jid = identity.judgment_id(rid, judge)
    out.append(rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": judge, "started_utc": t(i0 + 5), "finished_utc": t(i0 + 6),
                                "n_items_judged": n_items, "status": "complete", "api_pool": "lowprio"}, t(i0 + 6), prod, deterministic))
    out.append(rec("run_state", {"run_id": rid, "state": "judged", "event_utc": t(i0 + 7), "evidence_kind": "completion_artifact"},
                   t(i0 + 7), prod, deterministic))
    out.append(rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": n_items, "judge_id": judge}}, t(i0 + 8), prod, deterministic))
    out.append(rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": n_items // 4,
                                  "n": n_items, "computed_by": "producer"}, t(i0 + 9), prod, deterministic))
    return rid, out


def rule_scored_run_records(run_key: str, i0: int, object_key: str = "none", object_id: str = "none", n_items: int = 200,
                            n_scored: Optional[int] = None, computed_utc: bool = True, with_artifact: bool = False,
                            through: str = "generated", prod: Optional[Mapping[str, object]] = None,
                            deterministic: bool = True) -> Tuple[str, List[Dict[str, object]]]:
    """A rule-scored run's records (instrument :data:`RULE_INSTRUMENT`, no judge) through ``through`` ∈ {queued, running,
    generated}. A ``generated`` run carries its headline metric row — metric ``acc``, no judgment, ``n = n_scored``
    (default ``n_items``) — the completion evidence of a run nobody judges; ``computed_utc`` files the scoring time on
    it; ``with_artifact`` also records the store's completion artifact, which most live readers did not."""
    rid = identity.run_id(CAMPAIGN, run_key, object_key, CAP_EVAL_ID, RULE_INSTRUMENT_ID)
    p = run_payload(run_key, object_key, object_id, i0, run_id=rid, eval_id=CAP_EVAL_ID, item_set_id=CAP_ITEM_SET_ID,
                    instrument_id=RULE_INSTRUMENT_ID, harness="lm_eval", scoring_rule="exact_match")
    out = [rec("run", p, t(i0), prod, deterministic),
           rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(i0), "evidence_kind": "submit_wrapper"}, t(i0), prod, deterministic)]
    if through == "queued":
        return rid, out
    out.append(rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(i0 + 1), "evidence_kind": "job_log",
                                 "scheduler_state": "RUNNING"}, t(i0 + 1), prod, deterministic))
    if through == "running":
        return rid, out
    out.append(rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(i0 + 3), "evidence_kind": "completion_artifact",
                                 "evidence_uri": f"synthetic:///store/{run_key}/results.json"}, t(i0 + 3), prod, deterministic))
    fields: Dict[str, object] = {"n_items_generated": n_items}
    if with_artifact:
        fields.update(store_completion_artifact_uri=f"synthetic:///store/{run_key}/results.json", store_completion_sha256=SHA256_B)
    out.append(rec("run_patch", {"run_id": rid, "fields": fields}, t(i0 + 4), prod, deterministic))
    n = n_items if n_scored is None else n_scored
    metric: Dict[str, object] = {"run_id": rid, "metric_id": "acc", "value": 0.5, "k": n // 2, "n": n, "computed_by": "producer"}
    if computed_utc:
        metric["computed_utc"] = t(i0 + 5)
    out.append(rec("run_metric", metric, t(i0 + 5), prod, deterministic))
    return rid, out


def item_records(run_id: str, judgment_id: str, i0: int, n_items: int = 200, seed: int = 0, metric: str = "asr",
                 prod: Optional[Mapping[str, object]] = None, source_revision_id: str = "",
                 deterministic: bool = True) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    out = []
    for k in range(n_items):
        payload = {"run_id": run_id, "item_id": f"q{k:05d}", "metric_id": metric, "trial_index": 0, "sample_index": 0,
                   "turn_index": -1, "judgment_id": judgment_id, "value": round(rng.random(), 6), "label": None,
                   "api_pool": "lowprio", "created_utc": t(i0), "judged_utc": t(i0 + 1), "source_revision_id": source_revision_id,
                   "withdrawn_by_revision_id": None}
        # written AFTER the judgment record the rows cite (a grader claims, then writes its rows)
        out.append(rec("item_result", payload, t(i0 + 2), prod, deterministic))
    return out


def write_run_dir(root: str, campaign: str, run_id: str, records_by_file: Sequence[Tuple[str, Sequence[Mapping[str, object]]]],
                  gz_items: bool = True) -> List[str]:
    """Write ``[(record_type, records)]`` as ``<seq>-<type>.jsonl`` files under records/<campaign>/<run_dir>/."""
    d = os.path.join(root, "records", campaign, identity.run_dir_name(run_id))
    paths = []
    for seq, (rt, recs) in enumerate(records_by_file, start=1):
        gz = gz_items and rt in R.ITEM_SCALE_RECORD_TYPES
        paths.append(R.write_records(os.path.join(d, R.record_file_name(seq, rt, gz=gz)), list(recs), gz=gz))
    return paths


def write_corpus(root: str, n_runs: int = 3, n_items: int = 50, with_items: bool = True, i0: int = 0,
                 gz_items: bool = True) -> Dict[str, object]:
    """A complete synthetic corpus: dimensions + n_runs judged runs (+ items). Returns ids."""
    dims = seed_dimensions(i0)
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    run_ids = []
    judgment_ids = []
    for k in range(n_runs):
        obj_key = "none" if k % 2 == 0 else "constitution_lora@deadbeef"
        obj_id = "none" if k % 2 == 0 else OBJECT_SHA
        rid, recs = run_records(f"synthstore_{k:03d}", i0 + 100 + 20 * k, obj_key, obj_id, n_items)
        jid = identity.judgment_id(rid, JUDGE_ID)
        files: List[Tuple[str, Sequence[Mapping[str, object]]]] = []
        # one file per record (mirrors the emitter), items in one gz file
        for r in recs:
            files.append((str(r["record_type"]), [r]))
        if with_items:
            files.append(("item_result", item_records(rid, jid, i0 + 100 + 20 * k + 5, n_items, seed=k)))
        write_run_dir(root, CAMPAIGN, rid, files, gz_items=gz_items)
        run_ids.append(rid)
        judgment_ids.append(jid)
    return {"run_ids": run_ids, "judgment_ids": judgment_ids, "campaign": CAMPAIGN, "n_items": n_items}
