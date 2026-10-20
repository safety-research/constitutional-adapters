"""The `math500_seedrep` reader on a SYNTHETIC mirror shaped like the seed-replicate MATH-500 stores (invented item ids and shas,
`$SHARED/...` paths with a stand-in user, no generation text): one seed tag with its unsteered anchor, a fitted ReLU-steer object, a prompted key and
the Arditi all-positions object, plus the paired tag's composed difference_relu object; the producer's store pins, three-number record,
final-state record and the launch set written from the same synthetic truth; a harness roster / pins and an all-positions roster / pins
naming the synthetic objects.

What it pins:

* ONE GROUP PER SEED: the paired tag reads in the seed tag's batch (the composed row pairs across standups with `s<N>/none`);
* THE THREE NUMBERS on every row: the VALUE (`acc` = correct / scored, the recount computation's non-blank id, `n_items_judged` = scored), the
  BOUND (`acc` = correct / 500, a blank computation id, the interval `[correct, correct + unresolved] / 500`), the at-cap SHARE as
  `run_degeneration at_token_cap_finish_reason` over every at-cap item, the producer's loop count as `rep_ratio_loop` (computed_by producer),
  the unresolved non-loops as `truncation_only`;
* THE THREE ITEM STATES: a scored value; an at-cap item without an answer is `value NULL` + `at_cap_unresolved` with its class (`loop` by the
  producer's rule, else `ceiling`); an item unscored below the cap is `value NULL` + `unscored`; a scored-at-cap item's value counts with
  `at_token_cap 1`;
* THE INSTRUMENT: every row on the canonical MATH-500 instrument bound to qwen35_4b_it — thinking ON, cap 16,384, the one-step ladder, the
  thinking-card decode row matched to the registered policy;
* THE ANCHORS: every steered / prompted row of the seed tag against its `none`; the composed row across standups against the same `none`
  (`extra.cross_standup`) while the paired tag has no `none` of its own; the Arditi rows carrying the declared role `arditi_unpaired_value` on the run
  and the contrast;
* THE PAIRED TAG'S OWN `none`: launch-set class `anchor`, a run of the paired tag on the same
  instrument (a decode-seed trial at the seed index, `standup_replicate_of_view`), no contrast of its own; the composed row then pairs against IT
  (`same_standup_none`, `cross_standup` false, `anchor_view_legacy` the paired tag) and the seed tag's rows never do — even when the seed tag's `none`
  is absent; a paired `none` the launch set names but that did not file (not mirrored, refused, or filed under another class) leaves the composed row
  on the cross-standup fallback with `own_none_state` naming why;
* REFUSED BY NAME (the row, never the group): a store attesting another sha than the pins; a launch set expecting another sha than the pins; a
  store pin that is not the manifest's tree; a missing pin; a three-number row disagreeing with the re-derivation; a final-state entry
  disagreeing; a replicate seed base that is not the tag's index; a launch-set row with no mirrored store; a row the pins' `_meta.refused` names;
* end to end: the runs reach `ingested` through the harvester's rule-scored completion with no state event from the reader and nothing blocked.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

import _anchor_frame_fixtures as AF
import _b1_fixtures as F
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import math500_seedrep as M
from sisteer.registry.adapters.base import AdapterContext

SEED_N = 2
TAG = f"qwen35_4b_s{SEED_N}"
TAG_P = f"{TAG}p"
SEL, SEL_P = f"s{SEED_N}", f"s{SEED_N}p"
NONE = "none"
RELU = "constv2-mix-full__psr_ce_all"
PROMPTED = "prompted__safety_refusal"
ARDITI = "arditi_allpos"
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"
STAGED = f"base64s{SEED_N}__{RELU}"
STAGED_P = f"paired_eb64s{SEED_N}__{COMPOSED}"
STAGED_A = "ar_c0p3__arditi_allpos"
OBJ = hashlib.sha256(b"synthetic math500 seedrep relu s2").hexdigest()
OBJ_P = hashlib.sha256(b"synthetic math500 seedrep composed s2p").hexdigest()
OBJ_A = hashlib.sha256(b"synthetic math500 seedrep arditi allpos").hexdigest()
TREE = hashlib.sha1(b"synthetic math500 seedrep tree").hexdigest()
READER_TREE = hashlib.sha1(b"synthetic math500 identity reader tree").hexdigest()
MODEL_HF = "Qwen/Qwen3.5-4B"
NOW = "2026-09-23T08:00:00.000Z"
ITEM_IDS = [f"math_{i:08x}" for i in range(M.N_ITEMS)]
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
SEED_RULE = "sha256(f'{seed_base}|inspect-replicate|{item_id}|{sample_idx}')[:4] big-endian mod 2^31-1; seed_base = <N>"


def _seed_rule(n: int) -> str:
    return SEED_RULE.replace("<N>", str(n))
PRODUCER_ROOT = "/workspace/TESTUSER/exp/math500_thinking_on_full22"
OOS = {RELU: "math500_thinking_on_full22_qwen35_4b_native", COMPOSED: "math500_thinking_on_full22_qwen35_4b_native_paired", ARDITI: "math500_thinking_on_full22_qwen35_4b_steer_allpos"}
LOOP_RULE = "rep_ratio(text) > 0.5 or rep_ratio(last 200 whitespace tokens) > 0.5; rep_ratio = 1 - unique/total whitespace-token 4-grams (< 8 words -> 0)"


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _ctx(conn=None) -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=F.producer(), conn=conn)


def _write(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return path


# --------------------------------------------------------------------------------------------
# the synthetic truth of one (tag, object key) and the files written from it
# --------------------------------------------------------------------------------------------

def _spec(*, n_correct: int, n_wrong: int, loops: int, eligible: int, scored_at_cap: int = 0, scored_loops: int = 0, below_cap: int = 0) -> dict:
    """`states[item] = (state, value)`: scored | loop (at cap, unscored, degenerate) | eligible (at cap, unscored, not degenerate) | scored_at_cap
    (at cap, scored 0, not degenerate) | scored_loop (at cap, scored 0, degenerate) | below_cap (unscored below the cap)."""
    states = {}
    ids = iter(ITEM_IDS)
    for _ in range(n_correct):
        states[next(ids)] = ("scored", 1.0)
    for _ in range(n_wrong):
        states[next(ids)] = ("scored", 0.0)
    for _ in range(loops):
        states[next(ids)] = ("loop", None)
    for _ in range(eligible):
        states[next(ids)] = ("eligible", None)
    for _ in range(scored_at_cap):
        states[next(ids)] = ("scored_at_cap", 0.0)
    for _ in range(scored_loops):
        states[next(ids)] = ("scored_loop", 0.0)
    for _ in range(below_cap):
        states[next(ids)] = ("below_cap", None)
    assert len(states) == M.N_ITEMS
    return {"states": states}


AT_CAP_STATES = ("loop", "eligible", "scored_at_cap", "scored_loop")
LOOP_STATES = ("loop", "scored_loop")
SCORED_STATES = ("scored", "scored_at_cap", "scored_loop")


def _counts(spec: dict) -> dict:
    st = spec["states"]
    scored = {i: v for i, (s, v) in st.items() if s in SCORED_STATES}
    at_cap = [i for i, (s, _v) in st.items() if s in AT_CAP_STATES]
    loops = [i for i, (s, _v) in st.items() if s in LOOP_STATES]
    eligible = [i for i in at_cap if i not in loops]
    unresolved = [i for i, (s, _v) in st.items() if s in ("loop", "eligible")]
    below = [i for i, (s, _v) in st.items() if s == "below_cap"]
    correct = int(sum(scored.values()))
    return {"n_scored": len(scored), "correct": correct, "acc": correct / len(scored), "at_cap": at_cap, "loops": loops, "eligible": eligible, "unresolved": unresolved,
            "below": below, "scored_at_cap": [i for i, (s, _v) in st.items() if s in ("scored_at_cap", "scored_loop")],
            "loops_unresolved": [i for i, (s, _v) in st.items() if s == "loop"]}


def _outcome(iid: str, st: str, v, seed: int) -> dict:
    at_cap = st in AT_CAP_STATES
    scored = st in SCORED_STATES
    return {"item_id": iid, "at_cap": at_cap, "scored": scored, "value": (v if scored else None), "stop_reason": ("max_tokens" if at_cap else "stop"),
            "completion_tokens": (M.CAP if at_cap else 4200), "seed": seed, "sample_idx": 0, "think_block_unterminated": (st in ("loop", "eligible", "below_cap")),
            "unparseable": None, "attempt": 1, "prompt_tokens": 170, "sampling_ref": "872ee1a9a40d", "content_empty_with_reasoning": False, "stray_open_tag": False}


def _job_dir(slug: str, job: str) -> str:
    return f"m5t_qwen35_4b_{slug}_thinking_card_{job}"


def _class_of(okey: str, paired: bool) -> str:
    """The launch set's class grammar (the reader's): the Arditi key `arditi`; `none` `anchor` on EITHER tag (the paired tag's own anchor); a
    prompted key `prompted`; a fitted key `composed` under the paired tag, else `arm`."""
    if okey == ARDITI:
        return "arditi"
    if okey == NONE:
        return "anchor"
    if okey.startswith("prompted__"):
        return "prompted"
    return "composed" if paired else "arm"


def _store_files(root: str, tag: str, okey: str, slug: str, spec: dict, *, obj_sha, staged: str, tree: str, job: str, ihash8: str, chash8: str, seed_n: int, bank: str,
                 node: str = "gpu-9", max_lora_rank: int = 8, ended: str = "2026-09-23T04:16:05+00:00", started: str = "2026-09-23T03:58:12Z", failed_first: bool = False,
                 attest_sha: str = None, stamp_sha: str = None, with_summary: bool = True, with_serving: bool = True) -> str:
    """One store directory and its standup records: the named files the reader opens. `attest_sha` makes the store attest another object than the
    pins (the manifest, the rows aggregate, the launch stamp); `stamp_sha` alters the launch stamp alone."""
    d = os.path.join(root, M.STORES_HEAD, tag, slug, M.STORE_DIR)
    store_key = f"{slug}__controls__t0__a0__{ihash8}__{chash8}"
    kd = os.path.join(d, "cells", store_key)
    steered = okey not in (NONE, PROMPTED)
    a_sha = (attest_sha or obj_sha) if steered else None
    s_sha = (stamp_sha or a_sha) if steered else None
    states = spec["states"]
    outcomes = [_outcome(iid, st, v, 1000 + k) for k, (iid, (st, v)) in enumerate(sorted(states.items()))]
    c = _counts(spec)
    n_scored, n_at_cap, correct, acc = c["n_scored"], len(c["at_cap"]), c["correct"], c["acc"]
    _write(os.path.join(kd, "outcomes", M.OUTCOMES_FILE), {"task": M.TASK, "item_set": None, "items": outcomes, "max_tokens": M.CAP, "n_attempted_items": len(outcomes),
                                                           "n_attempted_samples": len(outcomes), "n_scored": n_scored, "n_unscored": len(outcomes) - n_scored, "n_at_cap": n_at_cap,
                                                           "producer_tree_sha": tree, "samples_per_item": 1, "thinking": "on", "think_split_mode": "implicit_open",
                                                           "unterminated_policy": "unscored"})
    harness_id = okey if okey in (NONE, ARDITI) else ("prompted__constitution" if okey == "prompted__si_constitution" else ("prompted__safety" if okey == PROMPTED else "si_" + okey.replace("-", "_")))
    route_kind = "lens" if steered else ("prompted" if okey.startswith("prompted__") else "base")
    lines = []
    for k, o in enumerate(outcomes):
        ex = {"answer": None, "completion_tokens": o["completion_tokens"], "stop_reason": o["stop_reason"], "seed": o["seed"], "sample_idx": 0, "max_tokens": M.CAP,
              "object_artifact_sha256": a_sha, "view": (bank if okey == ARDITI else tag.rsplit("_", 1)[1]), "route_kind": route_kind, "content": "synthetic"}
        lines.append(json.dumps({"task": M.TASK, "metric": ("acc" if o["scored"] else "unscored"), "item_id": o["item_id"], "value": (o["value"] if o["scored"] else 1.0),
                                 "created_at": 1790134559.0 + k, "slurm_job_id": job, "env_digest": "7bad6067725d533c", "model_id": MODEL_HF, "model_revision": _REVISION,
                                 "harness_config_hash": chash8 + "21d9039f", "intervention_hash": ihash8 + "21480189", "kind": "cell", "extra_json": json.dumps(ex)}))
    agg = {"object_artifact_sha256": a_sha, "producer_tree_sha": tree, "route_kind": route_kind, "view": (bank if okey == ARDITI else tag.rsplit("_", 1)[1]), "n": M.N_ITEMS, "n_scored": n_scored,
           "n_at_cap": n_at_cap}
    lines.append(json.dumps({"task": M.TASK, "metric": "acc_aggregate", "item_id": "__aggregate__", "value": acc, "kind": "cell", "extra_json": json.dumps(agg)}))
    lines.append(json.dumps({"task": M.TASK, "metric": "n", "item_id": "__aggregate__", "value": float(M.N_ITEMS), "kind": "cell", "extra_json": json.dumps(agg)}))
    _write(os.path.join(kd, "rows.jsonl"), "\n".join(lines) + "\n")
    gs = dict(SAMPLING, acc=acc, n=M.N_ITEMS, n_scored=n_scored, n_unscored=M.N_ITEMS - n_scored, n_at_cap=n_at_cap, max_tokens=M.CAP, thinking="on", task=M.TASK,
              framework="inspect_evals-0.18.0", engine="vllm", samples_per_item=1, sampling_ref="872ee1a9a40d", at_cap_rate=n_at_cap / M.N_ITEMS, unterminated_policy="unscored")
    _write(os.path.join(kd, "gen_stats.json"), {M.TASK: gs})
    _write(os.path.join(kd, "summary", M.SUMMARY_FILE), dict(gs))
    routing = f"{PRODUCER_ROOT}/{M.SERVING_HEAD}/{_job_dir(slug, job)}/math500/routing_table.json"
    ident = M.STEERED_IDENTITY if steered else M.PASSTHROUGH_IDENTITY
    man = {"cell_id": okey, "harness_cell_id": harness_id, "model_id": MODEL_HF, "model_revision": _REVISION, "serving_identity": ident, "serving_path": ("lora" if steered else "passthrough"),
           "serving_engine": "lora_native", "steer_positions": "all",
           "fit_provenance": {"object_sha": (a_sha[:16] if steered else None), "model_id": ((MODEL_HF if okey == ARDITI else "Qwen/Qwen3.5-4B-Base") if steered else None),
                              "transfer": (steered and okey != ARDITI), "fit_kind": ("class_b" if okey == ARDITI else ("document" if steered else None)), "served_model_id": MODEL_HF,
                              "served_revision": _REVISION},
           "producer": {"driver": "served_capability_run", "tree_sha": tree}, "row_constants": {"slurm_job_id": job, "env_digest": "7bad6067725d533c", "arm_name": slug},
           "routing_table": routing, "harness_config_hash": chash8 + "21d9039f", "intervention_hash": ihash8 + "21480189",
           "geometry": {"max_model_len": (24064 if okey == "prompted__si_constitution" else 17664), "max_num_seqs": 29, "source": "endpoint.json", "eval": "capability:gen_long_math"},
           "seeds": {"inspect": {"replicate": {"derivation": "contracts.inspect_replicate_seed", "policy": "replicate", "seed_base": seed_n}, "rule": _seed_rule(seed_n),
                                 "samples_per_item": {M.TASK: 1}}},
           "frameworks": {"inspect_ai": "0.3.260", "inspect_evals": "0.18.0", "client_lock_sha256": hashlib.sha256(b"synthetic client lock").hexdigest()},
           "sampling": dict(SAMPLING, mode="thinking_card", model_key="qwen35_4b"),
           "resolved_config": {"cell_id": okey, "system_sha": (_SI_SHA16 if okey == PROMPTED else None), "serving": {"serving_engine": "lora_native"}},
           "provenance": {"artifact_sha256": a_sha, "cell_id": okey, "class": ("B" if okey == ARDITI else ("A" if steered else "baseline"))},
           "route": {"bank_view": bank, "object_artifact_sha256": (a_sha if okey == ARDITI else None), "object_artifact_sha256_by_cell": ({harness_id: a_sha} if steered else None),
                     "serving_identity": ident, "serving_engine": "lora_native"},
           "oos_experiment": OOS.get(okey), "thinking": {"mode": "on", "enable_thinking": True, "render": "flag_only"},
           "tinycap": {"max_tokens_by_task": {M.TASK: M.CAP}, "math500_max_tokens": M.CAP, "math500_retry_max_tokens": 0, "samples": 1},
           "resume_attempts": ([{"attempt": 1, "completed": False, "tree_sha": tree, "ts": "2026-09-23T03:40:00Z"}] if failed_first else []) + [{"attempt": (2 if failed_first else 1), "completed": True, "tree_sha": tree, "ts": started}]}
    _write(os.path.join(kd, "manifest.json"), man)
    _write(os.path.join(kd, "evidence.json"), {"written_at": ended, "serving_identity": ident, "engine_stamp": {"object_sha": (a_sha[:16] if steered else None)}})
    _write(os.path.join(d, "evidence.json"), {"written_at": ended, "serving_identity": ident})
    _write(os.path.join(d, "resolved_sampler.json"), {"verdict": "UNCLASSIFIED_REQUESTS", "classes": {"client": {"n": M.N_ITEMS, "verdict": "AGREES", "distinct_resolved": [{"sampler": SAMPLING, "n": M.N_ITEMS}]}}})
    _write(os.path.join(d, "request_evidence.jsonl"), json.dumps({"kind": "task", "task": M.TASK, "item_id": ITEM_IDS[0], "seed": 1000, "max_tokens": M.CAP, **SAMPLING}) + "\n")
    _write(os.path.join(d, "usage_ledger.jsonl"), json.dumps({"task": M.TASK, "finish_reason": "stop", "max_tokens_requested": M.CAP}) + "\n")
    _write(os.path.join(d, "oos_prompt_budget.json"), {"arm": "on", "cap": M.CAP, "cap_task": M.TASK, "lengths": [170] * 3})
    sel = tag.rsplit("_", 1)[1]
    job_name = f"m5t-qwen35_4b-{sel}-{slug}"
    result_lines = []
    if failed_first:
        result_lines.append(f"2026-09-23T03:41:00+00:00 job={int(job) - 100} array=78 row=78 {job_name} m5t qwen35_4b {okey} thinking_card fitted=1 rc=2 driver_rc=1 result=DRIVER_FAILED wave=qwen_seedrep thinking=on geometry=capability:gen_long_math logreq=1")
    result_lines.append(f"{ended} job={job} array=62 row=62 {job_name} m5t qwen35_4b {okey} thinking_card fitted={1 if steered else 0} rc=0 driver_rc=0 result=COMPLETE wave=qwen_seedrep thinking=on geometry=capability:gen_long_math logreq=1")
    _write(os.path.join(d, "RESULT"), "\n".join(result_lines) + "\n")
    _write(os.path.join(d, "row_identity.json"), {"schema": M.ROW_IDENTITY_SCHEMA, "complete": (not steered), "problems": ([] if not steered else ["object_artifact_sha256_agrees: the driver's manifest disagrees with the row"]),
                                                  "checks": {"cell_agrees": True, "object_artifact_sha256_agrees": (not steered), "producer_tree_agrees": True, "thinking_mode_agrees": True},
                                                  "identity": {"client_lock_sha256": man["frameworks"]["client_lock_sha256"], "object_artifact_sha256": s_sha, "pin": tree, "producer_tree_sha": tree,
                                                               "route_kind": route_kind, "sampling_mode": "thinking_card", "sampling_ref": "872ee1a9a40d", "thinking": "on", "view": sel},
                                                  "row": {"budget_class": (okey if okey.startswith("prompted__") else "none"), "cell": okey, "harness_cell": harness_id, "index": 62, "job_id": job_name,
                                                          "model_key": "qwen35_4b", "out_root": f"{M.STORES_HEAD}/{tag}/{slug}/{M.STORE_DIR}", "sampling_mode": "thinking_card",
                                                          "si_sha256": (_SI_SHA if okey == PROMPTED else None), "staged_id": (staged if steered else None), "status": "launch", "tuple_name": OOS.get(okey), "wave": "qwen_seedrep"}})
    if with_summary:
        _write(os.path.join(d, slug, M.CONTROLS_SUMMARY_FILE),
               {"cell_id": okey, "provenance": {"artifact_sha256": a_sha, "cell_id": okey},
                "sources": {M.TASK: {"cell_key": store_key, "producer_tree_sha": tree, "harness_config_hash": man["harness_config_hash"]}},
                "controls": {M.TASK: {"acc": acc, "n": M.N_ITEMS, "n_scored": n_scored, "n_at_cap": n_at_cap, "max_tokens": M.CAP, "thinking": "on", "task": M.TASK, "engine": "vllm"}}})
    if with_serving:
        sd = os.path.join(root, M.SERVING_HEAD, _job_dir(slug, job), "math500")
        _write(os.path.join(sd, M.ATTESTATION_FILE), {"kind": "lora_native", "hostname": node, "model": MODEL_HF, "lora_config": {"max_lora_rank": max_lora_rank, "max_loras": 2, "target_modules": ["o_proj"]},
                                                      "expected": {"max_model_len": man["geometry"]["max_model_len"], "max_num_seqs": 29, "lora_loaded_count": 2, "lora_max_lora_rank": max_lora_rank},
                                                      "adapters": {"none__lora0": {"r": 1}}})
        _write(os.path.join(sd, M.STANDUP_JOBS_FILE), json.dumps({"job_id": job, "node": node, "key": "qwen35_4b", "view": sel, "sha": tree, "engine": "lora_native", "cluster": "cluster_b"}) + "\n")
    return store_key


_REVISION = None
_SI_SHA = None
_SI_SHA16 = None


def _bind_seeds(seeds):
    global _REVISION, _SI_SHA, _SI_SHA16
    _REVISION = str(seeds.models["qwen35_4b_it"]["hf_revision"])
    _SI_SHA = str(seeds.si_object_id("safety_refusal")).split(":", 1)[1]
    _SI_SHA16 = _SI_SHA[:16]


def _three_row(tag: str, sel: str, okey: str, slug: str, spec: dict, *, obj_sha, tree: str, job: str, bank: str, seed_n: int, max_lora_rank: int, anchor_spec: dict = None) -> dict:
    """The producer's three-number row for one (tag, object key) from the synthetic truth (m5t_three_numbers_seedrep/1)."""
    c = _counts(spec)
    steered = okey not in (NONE, PROMPTED)
    by_item = {i: {"rep_ratio": (0.74 if i in c["loops"] else 0.12), "tail_rep_ratio": (0.93 if i in c["loops"] else 0.2), "degenerate": (i in c["loops"])} for i in c["at_cap"]}
    row = {"tag": tag, "view": sel, "cell_slug": slug, "cell_id": okey, "arm": LC.cell_grammar_legacy(okey)["paper_label"], "sampling": "thinking_card",
           "out_root": f"{M.STORES_HEAD}/{tag}/{slug}/{M.STORE_DIR}", "status": "complete", "n_attempted": M.N_ITEMS,
           "value_correct_over_scored": round(c["acc"], 4), "n_scored_final": c["n_scored"], "correct_final": c["correct"], "bound_correct_over_attempted": round(c["correct"] / M.N_ITEMS, 4),
           "third_state": {"n_at_cap": len(c["at_cap"]), "share_at_cap": round(len(c["at_cap"]) / M.N_ITEMS, 4), "n_unresolved_at_cap": len(c["unresolved"]),
                           "share_unresolved": round(len(c["unresolved"]) / M.N_ITEMS, 4), "n_scored_at_cap": len(c["scored_at_cap"]),
                           "loop_eligible_split": {"loops": len(c["loops"]), "eligible": len(c["eligible"]), "loop_ids": sorted(c["loops"]), "eligible_ids": sorted(c["eligible"]), "rule": LOOP_RULE,
                                                   "informational": True, "note": "no rung exists on MATH-500; the split is informational (D73 (e))"},
                           "n_unscored_below_cap": len(c["below"]), "unscored_below_cap_ids": sorted(c["below"])},
           "off_task": {"rule": {"candidates": "items NOT CORRECT in the final state (scorer-anchored)"}, "counts": {"loop": len(c["loops"]), "other_incorrect": 3, "other_unscored": len(c["eligible"])},
                        "share_off_task_structural": 0.004, "ids": {}, "n_texts_read": M.N_ITEMS},
           "degeneracy_by_item": by_item,
           "store": {"producer_tree_sha": tree, "serving_engine": "lora_native", "serving_identity": (M.STEERED_IDENTITY if steered else M.PASSTHROUGH_IDENTITY),
                     "serving_path": ("lora" if steered else "passthrough"), "bank_view": bank, "object_artifact_sha256": (obj_sha if okey == ARDITI else None), "cell_id": okey,
                     "harness_cell_id": (okey if okey in (NONE, ARDITI) else ("prompted__safety" if okey == PROMPTED else "si_" + okey.replace("-", "_"))), "oos_experiment": OOS.get(okey),
                     "thinking_mode": "on", "sampling_mode": "thinking_card", "sampling": dict(SAMPLING), "seed_policy": "replicate", "seed_base": seed_n, "seed_rule": _seed_rule(seed_n),
                     "model_id": MODEL_HF, "model_revision": _REVISION, "engine": "vllm", "result": "COMPLETE", "slurm_job_id": job},
           "standup": {"max_lora_rank": max_lora_rank, "lora_target_modules": ["o_proj"], "standup_job_id": job, "source": "serving_attestation.json+vllm_server.log"},
           "outcome_doc": {"n_attempted_items": M.N_ITEMS, "n_scored": c["n_scored"], "n_at_cap": len(c["at_cap"]), "max_tokens": M.CAP, "producer_tree_sha": tree, "task": M.TASK},
           "design": {"view": sel, "bank_view": bank, "inspect_seed_base": seed_n, "tuple_name": OOS.get(okey), "object_sha256": (obj_sha if steered else None)},
           "contrast": None}
    if anchor_spec is not None:
        ac = _counts(anchor_spec)
        row["contrast"] = {"anchor": f"{M.STORES_HEAD}/qwen35_4b_s{seed_n}/none/{M.STORE_DIR}", "anchor_view": f"s{seed_n}", "delta_value": round(c["acc"] - ac["acc"], 4),
                           "max_lora_rank": {"row": max_lora_rank, "anchor": 8}}
        if okey == ARDITI:
            row["contrast"]["comparability_caveat"] = "the Arditi engine is single-object (max_lora_rank 1 by construction) — the VALUE is filed regardless (GSM8K C4 precedent, D71)"
    return row


def _final_state_row(tag: str, sel: str, spec: dict, *, obj_sha, tree: str, seed_n: int) -> dict:
    c = _counts(spec)
    items = {}
    for k, (iid, (st, v)) in enumerate(sorted(spec["states"].items())):
        at_cap = st in AT_CAP_STATES
        scored = st in SCORED_STATES
        items[iid] = {"scored": scored, "correct": bool(scored and v == 1.0), "at_cap": at_cap, "completion_tokens": (M.CAP if at_cap else 4200), "stop_reason": ("max_tokens" if at_cap else "stop"),
                      "think_block_unterminated": (st in ("loop", "eligible", "below_cap")), "seed": 1000 + k}
    return {"tag": tag, "view": sel, "status": "complete", "n_items": M.N_ITEMS, "n_scored": c["n_scored"], "n_correct": c["correct"], "acc_over_scored": round(c["acc"], 4),
            "acc_over_attempted": round(c["correct"] / M.N_ITEMS, 4), "n_at_cap_unresolved": len(c["unresolved"]), "n_unscored_below_cap": len(c["below"]), "n_scored_at_cap": len(c["scored_at_cap"]),
            "n_at_cap_total": len(c["at_cap"]), "rung": None, "note": "MATH-500 has no rung: the final state IS the base state (D73 (e))", "producer_tree_sha": tree, "seed_base": seed_n,
            "seed_policy": "replicate", "row_identity_complete": True, "row_identity_manifest_object_sha256": obj_sha, "row_identity_reader_tree_sha": READER_TREE, "items": items}


def _launch_row(sel: str, okey: str, slug: str, tag: str, seed_n: int, *, pin_key, expected, paired: bool, index: int) -> dict:
    cls = _class_of(okey, paired)
    return {"tuple_id": f"{sel}/{okey}", "view": sel, "cell": okey, "paper_label": LC.cell_grammar_legacy(okey)["paper_label"], "class": cls,
            "block": ("paired" if paired else ("arditi" if cls == "arditi" else "seed")), "fitted": (okey not in (NONE, PROMPTED)), "seed_base": seed_n, "pin_key": pin_key, "tag": tag,
            "cell_slug": slug, "table_index": index, "object_sha256_expected": expected, "census_status": "run", "m5t_index": 54 + index, "m5t_out_root": f"{M.STORES_HEAD}/{tag}/{slug}/{M.STORE_DIR}"}


NONE_SPEC = dict(n_correct=440, n_wrong=30, loops=8, eligible=15, scored_at_cap=5, below_cap=2)
RELU_SPEC = dict(n_correct=420, n_wrong=45, loops=10, eligible=19, scored_at_cap=3, scored_loops=1, below_cap=2)
PROMPTED_SPEC = dict(n_correct=450, n_wrong=25, loops=5, eligible=10, scored_at_cap=6, below_cap=4)
ARDITI_SPEC = dict(n_correct=445, n_wrong=30, loops=6, eligible=12, scored_at_cap=4, below_cap=3)
COMPOSED_SPEC = dict(n_correct=300, n_wrong=40, loops=120, eligible=10, scored_at_cap=20, below_cap=10)

#: (object key, paired?, staged id, object sha, bank selection, job id, ihash8, chash8, max_lora_rank)
ROWS = [(NONE, False, None, None, SEL, "128777", "2ff9cdff", "ff80b842", 8),
        (RELU, False, STAGED, OBJ, SEL, "128778", "d6dc3d5f", "1b85e6e8", 8),
        (PROMPTED, False, None, None, SEL, "128832", "faa4c8cf", "fa3fc829", 8),
        (ARDITI, False, STAGED_A, OBJ_A, "ar_c0p3", "128989", "92b66dce", "79448d8c", 1),
        (COMPOSED, True, STAGED_P, OBJ_P, SEL_P, "128984", "76c03d28", "86573405", 8)]
SPECS = {NONE: NONE_SPEC, RELU: RELU_SPEC, PROMPTED: PROMPTED_SPEC, ARDITI: ARDITI_SPEC, COMPOSED: COMPOSED_SPEC}
#: the PAIRED tag's own `none` (`_mirror(paired_none=True)`): its store REUSES the seed tag's `none` store key byte for byte (the GSM8K reality; the
#: tag-prefixed producer key keeps the ids apart) under its own job; fewer correct than the seed tag's `none` so an anchor's accuracy tells which `none` paired
NONE_P_SPEC = dict(n_correct=300, n_wrong=170, loops=8, eligible=15, scored_at_cap=5, below_cap=2)
NONE_P_ROW = (NONE, True, None, None, SEL_P, "129051", "2ff9cdff", "ff80b842", 8)


def _mirror(tmp_path, seeds, *, root: str = None, drop: tuple = (), pins_override: dict = None, three_override=None, fs_override=None, launch_override=None,
            attest_sha: dict = None, seed_base_override: int = None, meta_override=None, without_final_state: bool = False, pins_sha: dict = None, extra_launch: list = None,
            paired_none=False) -> dict:
    """A synthetic mirror root under tmp_path: the seed tag's four stores and the paired tag's composed store, the producer's records written from
    the same truth, the launch set, and the harness / all-positions roster and pins naming the synthetic objects. ``paired_none`` adds the paired tag's
    OWN `none` (`True`: its store and records; `"not_mirrored"`: the launch-set row alone, the way an early launch set names a row no mirror carries yet)."""
    _bind_seeds(seeds)
    root = root or str(tmp_path / "math500_seedrep")
    pins = {"_meta": {"schema": M.PINS_SCHEMA, "label": "synthetic", "refused": {}, "partial": {}, "identity_refused": {}}}
    three = {"schema": M.THREE_NUMBERS_SCHEMA, "written": NOW, "rows": {}}
    fstate = {"schema": M.FINAL_STATE_SCHEMA, "reader_tree_sha": READER_TREE, "stores": {}}
    launch = []
    keys = {}
    specs = {k: _spec(**v) for k, v in SPECS.items()}
    for i, (okey, paired, staged, sha, bank, job, ihash8, chash8, mlr) in enumerate(ROWS):
        tag = TAG_P if paired else TAG
        sel = SEL_P if paired else SEL
        slug = okey.replace("__", "--", 1)
        sb = seed_base_override if seed_base_override is not None else SEED_N
        launch.append(_launch_row(sel, okey, slug, tag, SEED_N, pin_key=staged, expected=sha, paired=paired, index=i))
        if okey in drop:
            continue
        keys[okey] = _store_files(root, tag, okey, slug, specs[okey], obj_sha=sha, staged=staged, tree=TREE, job=job, ihash8=ihash8, chash8=chash8, seed_n=sb, bank=bank,
                                  node=f"gpu-{i}", max_lora_rank=mlr, failed_first=paired, attest_sha=(attest_sha or {}).get(okey))
        pins[M.pin_key(sel, okey)] = TREE
        three["rows"][f"{tag}/{slug}"] = _three_row(tag, sel, okey, slug, specs[okey], obj_sha=sha, tree=TREE, job=job, bank=bank, seed_n=SEED_N, max_lora_rank=mlr,
                                                    anchor_spec=(specs[NONE] if okey != NONE else None))
        fstate["stores"][f"{tag}/{slug}/{M.STORE_DIR}"] = _final_state_row(tag, sel, specs[okey], obj_sha=sha, tree=TREE, seed_n=SEED_N)
    fx_paired_none = None
    if paired_none:
        okey, _p, _staged, _sha, bank, job, ihash8, chash8, mlr = NONE_P_ROW
        spec_p = _spec(**NONE_P_SPEC)
        launch.append(_launch_row(SEL_P, okey, okey, TAG_P, SEED_N, pin_key=None, expected=None, paired=True, index=len(ROWS)))
        fx_paired_none = {"key": None, "spec": spec_p}
        if paired_none != "not_mirrored":
            fx_paired_none["key"] = _store_files(root, TAG_P, okey, okey, spec_p, obj_sha=None, staged=None, tree=TREE, job=job, ihash8=ihash8, chash8=chash8, seed_n=SEED_N, bank=bank,
                                                 node=f"gpu-{len(ROWS)}", max_lora_rank=mlr)
            pins[M.pin_key(SEL_P, okey)] = TREE
            three["rows"][f"{TAG_P}/{okey}"] = _three_row(TAG_P, SEL_P, okey, okey, spec_p, obj_sha=None, tree=TREE, job=job, bank=bank, seed_n=SEED_N, max_lora_rank=mlr)
            fstate["stores"][f"{TAG_P}/{okey}/{M.STORE_DIR}"] = _final_state_row(TAG_P, SEL_P, spec_p, obj_sha=None, tree=TREE, seed_n=SEED_N)
    if extra_launch:
        launch += extra_launch
    if pins_override:
        pins.update(pins_override)
    if meta_override:
        meta_override(pins["_meta"])
    if three_override:
        three_override(three)
    if fs_override:
        fs_override(fstate)
    launch_doc = {"schema": M.LAUNCH_SET_SCHEMA, "derived_at": NOW, "evals": {M.EVAL_ID: {"own_source": M.SOURCE, "launch": launch,
                                                                                          "canonical_decode": {"policy": "qwen35_4b_it|math500|on|thinking_card@v1", "thinking": "on", "cap": M.CAP}}}}
    if launch_override:
        launch_override(launch_doc)
    _write(os.path.join(root, M.PINS_FILE), pins)
    _write(os.path.join(root, M.THREE_NUMBERS_FILE), three)
    if not without_final_state:
        _write(os.path.join(root, M.FINAL_STATE_FILE), fstate)
    _write(os.path.join(root, M.RECORDS_DIR, M.LAUNCH_SET_FILE), launch_doc)
    import yaml
    hp = dict(pins_sha or {})
    roster = {"targets": {"qwen35_4b": {"views": {SEL: {"config": f"configs/sdfbase_full22_{TAG}.yaml", "cells": {RELU: STAGED, PROMPTED: None, COMPOSED: None}},
                                                  SEL_P: {"config": f"configs/sdfbase_full22_{TAG_P}.yaml", "paired_of": SEL, "cells": {COMPOSED: STAGED_P}}}}}}
    roster_path = _write(str(tmp_path / "roster.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": hp.get(RELU, OBJ), "source": "synthetic"}, STAGED_P: {"object_sha256": hp.get(COMPOSED, OBJ_P), "source": "synthetic"}}})
    a_roster = {"targets": {"qwen35_4b": {"views": {"ar_c0p3": {"c": 0.3, "dose_unit": "raw_coefficient", "phase": "both", "cells": {ARDITI: STAGED_A}}}}}}
    a_roster_path = _write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump(a_roster))
    a_pins_path = _write(str(tmp_path / "arditi_pins.json"), {"qwen35_4b": {STAGED_A: {"staged_id": STAGED_A, "object_sha256": hp.get(ARDITI, OBJ_A), "kind": "vector_all", "served_layer": 12, "c": 0.3,
                                                                                         "dose_unit": "raw_coefficient", "phase": "both", "positions": "all", "view": "ar_c0p3"}}})
    return {"root": root, "roster": roster_path, "pins": pins_path, "arditi_roster": a_roster_path, "arditi_pins": a_pins_path, "keys": keys, "specs": specs, "paired_none": fx_paired_none}


def _adapter(fx: dict, seeds) -> M.Math500SeedrepAdapter:
    return M.Math500SeedrepAdapter(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], arditi_roster_path=fx["arditi_roster"], arditi_pins_path=fx["arditi_pins"])


def _primed_frame(ad: M.Math500SeedrepAdapter):
    """The steady-state frame of an in-memory read (`_anchor_frame_fixtures`): since emission 2 every contrast of the reader is decided by the pass's
    anchor currency, and on an EMPTY frame every comparison parks (no provisional anchors). The in-memory tests describe the pass AFTER the fold, so
    they read on a frame primed from the adapter's own emissions (every run current, complete, off-roster)."""
    frame = AF.Frame()
    AF.load_records(frame, AF.priming_read(ad, _ctx(frame.con)))
    return frame


def _primed_conn(ad: M.Math500SeedrepAdapter, conn=None):
    return conn if conn is not None else _primed_frame(ad).con


def _read(ad: M.Math500SeedrepAdapter, conn=None, group: str = TAG):
    """One group's read on `conn`, or on a frame primed from the adapter's own emissions (:func:`_primed_frame`) when none is given; every record validated."""
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert sorted(groups) == [group], list(groups)
    recs = []
    for batch in ad.read(_ctx(_primed_conn(ad, conn)), group, groups[group]):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group)


def _run(bt, okey: str) -> dict:
    hits = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == okey]
    assert len(hits) == 1, (okey, len(hits))
    return hits[0]


def _metrics(bt, rid: str):
    rows = [m for m in bt["run_metric"] if m["run_id"] == rid]
    value = [m for m in rows if m["computation_id"] == M.recount_computation_id()]
    bound = [m for m in rows if m["computation_id"] == ""]
    assert len(rows) == 2 and len(value) == 1 and len(bound) == 1, rows
    return value[0], bound[0]


def _refused(facts, tag: str, okey: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{tag}|{okey} -- ")]
    assert len(hits) == 1, facts["refusals"]
    return hits[0]


# --------------------------------------------------------------------------------------------
# layout and identity
# --------------------------------------------------------------------------------------------

def test_discover_groups_the_seed_tag_with_its_paired_tag_and_aliases_the_records_and_standups(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = sorted(os.path.basename(i.uri) for i in inputs)
    for shared in M.SHARED_FILES:
        assert f"{shared}#{TAG}" in names, shared
    assert names.count("manifest.json") == 5 and names.count(M.OUTCOMES_FILE) == 10 and names.count("rows.jsonl") == 5 and names.count("RESULT") == 5 and names.count("row_identity.json") == 5
    assert names.count(f"{M.ATTESTATION_FILE}#{TAG}") == 5 and names.count(f"{M.STANDUP_JOBS_FILE}#{TAG}") == 5 and names.count(M.CONTROLS_SUMMARY_FILE) == 5
    assert {ad.comparison_group(i) for i in inputs} == {TAG}, "the paired tag reads in its seed tag's group (GROUP_RULE)"
    roles = {os.path.basename(i.uri): i.role for i in inputs}
    assert roles["rows.jsonl"] == "rows" and roles["gen_stats.json"] == "completion" and roles["RESULT"] == "log"
    assert M.seed_tag_of(TAG_P) == TAG and M.seed_tag_of("qwen35_4b_s2pp") is None and M.tag_parts(TAG_P) == (SEED_N, True)


def test_run_ids_key_on_the_tag_prefixed_store_name_not_the_root_path(tmp_path, seeds):
    fx1 = _mirror(tmp_path, seeds, root=str(tmp_path / "a" / "math500_seedrep"))
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "b" / "elsewhere"))
    bt1, f1 = _read(_adapter(fx1, seeds))
    bt2, _f2 = _read(_adapter(fx2, seeds))
    assert f1["n_refusals"] == 0, f1["refusals"]
    ids1 = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r["run_id"] for r in bt1["run"]}
    ids2 = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r["run_id"] for r in bt2["run"]}
    assert ids1 == ids2 and len(ids1) == 5
    inst = str(seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"])
    key = fx1["keys"][RELU]
    assert ids1[(TAG, RELU)] == I.run_id(LC.CAMPAIGN, f"{TAG}/{key}", RELU, M.EVAL_ID, inst) == M.run_id_for(TAG, key, RELU, inst)
    assert ids1[(TAG, RELU)].startswith(f"{LC.CAMPAIGN}:{M.EVAL_ID}:{inst[:12]}:")
    r = _run(bt1, RELU)
    assert r["producer_run_key_legacy"] == f"{TAG}/{key}" and r["producer_object_key_legacy"] == RELU and r["instrument_id"] == inst
    assert _run(bt1, COMPOSED)["producer_run_key_legacy"] == f"{TAG_P}/{fx1['keys'][COMPOSED]}" and _run(bt1, COMPOSED)["view_legacy"] == TAG_P


# --------------------------------------------------------------------------------------------
# the three numbers, the three item states, the instrument
# --------------------------------------------------------------------------------------------

def test_the_three_numbers_are_re_derived_from_the_outcome_record_and_the_items_carry_their_states(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    r = _run(bt, RELU)
    c = _counts(fx["specs"][RELU])
    # 420 + 45 scored + 3 scored at cap + 1 scored loop = 469 scored / 420 correct; at cap = 10 + 19 + 3 + 1 = 33; unresolved 29 (10 loops + 19 at the ceiling); 2 unscored below the cap
    assert (c["n_scored"], c["correct"], len(c["at_cap"]), len(c["unresolved"]), len(c["loops"]), len(c["below"])) == (469, 420, 33, 29, 11, 2)
    assert r["lifecycle_state"] == "generated" and r["token_cap_tokens"] == M.CAP and r["token_cap_rung_index"] == 0 and r["token_cap_ladder_json"] == [M.CAP]
    assert r["n_items_planned"] == 500 and r["n_items_generated"] == 500 and r["n_items_judged"] == 469 and r["extra"]["escalation_state"] == "final_at_base"
    tn = r["extra"]["three_numbers"]
    assert (tn["n_scored_final"], tn["correct_final"], tn["n_unresolved_neither_hit_nor_miss"], tn["n_at_cap_final"], tn["n_loops"], tn["n_loops_unresolved"], tn["n_at_cap_ceiling"],
            tn["n_unscored_below_cap"], tn["n_scored_at_cap_final"], tn["n_eligible_informational"]) == (469, 420, 29, 33, 11, 10, 19, 2, 4, 22)
    assert tn["producer_record_agrees"] is True and tn["final_state_record_agrees"] is True and tn["final_state_reader_tree_sha"] == READER_TREE and tn["loop_split_informational"] is True
    assert tn["producer_off_task"] == {"loop": 11, "other_incorrect": 3, "other_unscored": 22} and tn["producer_off_task_share_structural"] == 0.004
    value, bound = _metrics(bt, r["run_id"])
    assert value["metric_id"] == "acc" and value["value"] == 420 / 469 and (value["k"], value["n"]) == (420, 469) and value["computed_by"] == "registry" and value["ci_method"] == "wilson_95"
    assert value["producer_metric_spelling"] == "value_correct_over_scored" and value["extra"]["producer_value_correct_over_scored"] == round(420 / 469, 4)
    assert bound["metric_id"] == "acc" and bound["value"] == 420 / 500 and (bound["k"], bound["n"]) == (420, 500) and bound["computed_by"] == "producer"
    assert bound["producer_metric_spelling"] == M.BOUND_SPELLING and bound["ci_method"] == M.BOUND_CI_METHOD and (bound["ci_lo"], bound["ci_hi"]) == (420 / 500, 449 / 500)
    assert "secondary" in bound["extra"]["metric_role_expected"] and bound["n_incomplete"] == 29
    comps = {x["computation_id"] for x in bt["computation"]}
    assert value["computation_id"] in comps and value["computation_id"] == I.computation_id(M.RECOUNT_ALGORITHM, M.RECOUNT_VERSION, M.RECOUNT_SEED_PROVENANCE)
    # items: 500 rows, the three states
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == r["run_id"]}
    assert len(items) == 500 and all(i["judgment_id"] == "" and i["metric_id"] == "acc" and i["trial_index"] == 0 for i in items.values())
    labels = {}
    for i in items.values():
        k = (i["label"], i["extra"].get("at_cap_class"), i["extra"].get("unscored_reason"))
        labels[k] = labels.get(k, 0) + 1
    assert labels == {(None, None, None): 469, (IC.ITEM_LABEL_AT_CAP_UNRESOLVED, "loop", None): 10, (IC.ITEM_LABEL_AT_CAP_UNRESOLVED, "ceiling", None): 19,
                      (M.ITEM_LABEL_UNSCORED, None, M.UNSCORED_BELOW_CAP_CLASS): 2}, labels
    assert all(i["value"] is None for i in items.values() if i["label"] is not None) and all(i["value"] in (0.0, 1.0) for i in items.values() if i["label"] is None)
    assert sum(1 for i in items.values() if i["extra"]["scored_at_cap"]) == 4
    flags = {f["item_id"]: f for f in bt["item_flags"] if f["run_id"] == r["run_id"]}
    assert sum(f["at_token_cap"] for f in flags.values()) == 33 and sum(1 for f in flags.values() if f["degenerate_rep_ratio_loop"] == 1) == 11
    assert sum(1 for f in flags.values() if f["degenerate_rep_ratio_loop"] == 0) == 22 and sum(1 for f in flags.values() if f["degenerate_rep_ratio_loop"] is None) == 467
    assert all(f["generation_artifact_id"] is not None and f["generation_offset"] is not None for f in flags.values()), "every item has a rows.jsonl row (scored or unscored)"
    assert all(f["rep_ratio"] is not None for f in flags.values() if f["at_token_cap"] == 1) and all(f["rep_ratio"] is None for f in flags.values() if f["at_token_cap"] == 0)
    # the run-level shares over the 500 attempted items
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"], deg[M.RAW_AT_CAP_MEASURE]["denominator_unit"], deg[M.RAW_AT_CAP_MEASURE]["computed_by"]) == (33, 500, "items", "registry")
    assert (deg[M.LOOP_MEASURE]["numerator"], deg[M.LOOP_MEASURE]["computed_by"], deg[M.COHERENT_MEASURE]["numerator"]) == (11, "producer", 19)
    assert deg["classifier"]["measurement_state"] == "not_measured" and deg[IC.AT_CAP_RETOKENIZED]["measurement_state"] == "not_measured"
    assert set(deg) == set(IC.DECLARED_MEASURES) | {M.COHERENT_MEASURE}
    # the unsteered anchor: a decode-seed trial at the seed index; its value equals gen_stats' acc
    n = _run(bt, NONE)
    nc = _counts(fx["specs"][NONE])
    v_n, b_n = _metrics(bt, n["run_id"])
    assert (v_n["k"], v_n["n"]) == (nc["correct"], nc["n_scored"]) and v_n["extra"]["equals_gen_stats_acc"] is True and n["trial_index"] == SEED_N and n["seed_base"] == SEED_N
    assert n["extra"]["decode_replicate"]["twin_tags_legacy"] == ["qwen35_4b_s1", "qwen35_4b_s3"] and n["object_id"] == "none" and n["serving_path"] == "passthrough"
    assert facts["rows"][f"{TAG}|{NONE}"]["max_lora_rank_recorded"] == 8 and facts["rows"][f"{TAG}|{ARDITI}"]["max_lora_rank_recorded"] == 1


def test_the_run_row_carries_the_attested_identity_and_its_own_instrument(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds))
    r = _run(bt, RELU)
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")
    assert r["model_id"] == "qwen35_4b_it" and r["object_id"] == OBJ and r["view_legacy"] == TAG and r["lane_source_legacy"] == M.SOURCE and r["campaign"] == LC.CAMPAIGN
    assert r["instrument_id"] == inst["instrument_id"] and r["eval_id"] == M.EVAL_ID and r["item_set_id"] == "math500_500" and r["harness"] == "inspect_ai" and r["scoring_rule"] == "sympy_equiv"
    assert r["thinking_mode"] == "on" and r["token_cap_tokens"] == 16384 and r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all" and r["transfer_kind"] == "base_to_instruct"
    assert r["decode_sent_json"] == SAMPLING and r["decode_temperature_sent"] == 0.6 and r["decode_top_p_sent"] == 0.95 and r["decode_top_k_sent"] == 20 and r["decode_source"] == "model_card"
    assert r["instrument_match"] == "match" and r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|math500|on|thinking_card@v1"
    assert r["seed_base"] == SEED_N and r["decode_seed_rule"] == "per_item" and r["trial_index"] == 0 and r["extra"]["decode_seed_policy"] == "replicate" and r["extra"]["fit_seed"] == SEED_N
    assert r["extra"]["decode_seed_rule_text"] == _seed_rule(SEED_N)
    assert r["code_sha"] == TREE and r["model_revision"] == _REVISION and r["cluster_job_id"] == "128778" and r["job_name"] == f"m5t-qwen35_4b-{SEL}-{RELU.replace('__', '--', 1)}"
    assert r["substrate"] == "cluster_b" and r["substrate_source"] == "recorded" and r["node_hostname"] == "gpu-1" and r["context_window_tokens"] == 17664 and r["served_max_num_seqs"] == 29
    assert r["producer_session"] == "Overflow" and "0.3.260" in r["harness_version"] and r["is_operating_point"] == 1 and r["strength_nominal"] == 1.0 and r["judge_id"] is None
    assert r["extra"]["store_pin"] == {"key": M.pin_key(SEL, RELU), "tree": TREE} and r["extra"]["launch_set"]["pin_key"] == STAGED and r["extra"]["launch_set"]["object_sha256_expected"] == OBJ
    assert r["extra"]["launch_set"]["class"] == "arm" and r["extra"]["standup"]["max_lora_rank"] == 8 and r["extra"]["standup"]["cluster"] == "cluster_b"
    assert r["extra"]["launch_stamp"]["complete"] is False and r["extra"]["launch_stamp"]["problems"], "the stamp's own verdict is carried, not trusted"
    assert r["generated_utc"] == "2026-09-23T04:16:05.000Z" and r["started_utc"] == "2026-09-23T03:58:12.000Z" and r["queued_utc"] == r["started_utc"]
    assert r["store_completion_artifact_uri"].endswith(os.path.join(M.STORE_DIR, "cells", fx["keys"][RELU], "gen_stats.json"))
    fc = r["field_confidence_json"]
    assert fc["code_sha"]["confidence"] == "recorded" and fc["seed_base"]["confidence"] == "recorded" and fc["n_items_judged"]["confidence"] == "derived" and fc["substrate"]["confidence"] == "recorded"
    assert "run_state" not in bt
    # the prompted key resolves through si_texts and its system_sha; the composed row is the paired tag's, trial 0, paired_of s2
    p = _run(bt, PROMPTED)
    assert p["object_id"] == str(seeds.si_object_id("safety_refusal")) and p["serving_path"] == "passthrough" and p["transfer_kind"] == "prompted_only" and p["trial_index"] == SEED_N
    cp = _run(bt, COMPOSED)
    assert cp["object_id"] == OBJ_P and cp["view_legacy"] == TAG_P and cp["trial_index"] == 0 and cp["extra"]["paired_of"] == SEL and cp["extra"]["roster_paired_of"] == SEL
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and cp["extra"]["oos_tuple"] == OOS[COMPOSED] and len(cp["extra"]["attempts_before_accepted"]) == 1
    assert cp["extra"]["attempts_before_accepted"][0]["result"] == "DRIVER_FAILED" and cp["started_utc"] == "2026-09-23T03:58:12.000Z"
    # the all-positions object: ONE fit at the three seed bases -> a decode-seed trial with the operating point's strength columns and the declared role
    a = _run(bt, ARDITI)
    assert a["object_id"] == OBJ_A and a["trial_index"] == SEED_N and a["seed_base"] == SEED_N and a["extra"]["fit_seed"] is None and a["extra"]["declared_role"] == M.ARDITI_ROLE
    assert (a["strength_coefficient"], a["strength_unit"], a["strength_nominal"], a["is_operating_point"], a["operating_point_id_legacy"]) == (0.3, "raw_coefficient", 1.0, 1, STAGED_A)
    assert a["transfer_kind"] == "instruct_direct" and a["extra"]["single_object"]["operating_point"] == "ar_c0p3" and a["extra"]["standup"]["max_lora_rank"] == 1
    assert a["extra"]["decode_replicate"]["single_object"]["object_id"] == OBJ_A and M.SOURCE in a["extra"]["decode_replicate"]["coexist_ruling_needed"]


def test_every_row_pairs_with_the_seed_tags_none_the_composed_row_across_standups_and_the_arditi_row_declared_unpaired(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds))
    none_id = _run(bt, NONE)["run_id"]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {_run(bt, k)["run_id"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)}
    assert all(c["anchor_run_id"] == none_id and c["metric_id"] == "acc" and c["delta_method"] == "exact_mcnemar" for c in contrasts.values())
    assert {k: contrasts[_run(bt, k)["run_id"]]["anchor_kind"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)} == \
        {RELU: "same_standup_none", PROMPTED: "same_standup_none", ARDITI: "same_standup_none", COMPOSED: "sibling_none"}, "the D65 fallback is another standup's none"
    assert all(c["computation_id"] == M.recount_computation_id() and c["computed_by"] == "registry" and 0.0 <= c["delta_p"] <= 1.0 for c in contrasts.values())
    c_relu = contrasts[_run(bt, RELU)["run_id"]]
    # pairs = the items scored on both sides: the fixture lays states out in item order, so the first 465 items of RELU are scored (420 + 45) and the first 470 of NONE
    assert c_relu["n_pairs"] == 465 + 4 - sum(1 for i, (s, _v) in fx["specs"][RELU]["states"].items() if s in SCORED_STATES and fx["specs"][NONE]["states"][i][0] not in SCORED_STATES)
    assert c_relu["extra"]["cross_standup"] is False and c_relu["extra"].get("declared_role") is None and c_relu["extra"]["anchor_view_legacy"] == TAG
    c_comp = contrasts[_run(bt, COMPOSED)["run_id"]]
    assert c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["row_tag_legacy"] == TAG_P and "cross_standup_note" in c_comp["extra"]
    c_ard = contrasts[_run(bt, ARDITI)["run_id"]]
    assert c_ard["extra"]["declared_role"] == M.ARDITI_ROLE and "anchor_standup_note" in c_ard["extra"]
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["cross_standup"] is True and facts["contrasts"][f"{TAG}|{ARDITI}"]["declared_role"] == M.ARDITI_ROLE
    # no paired `none` in the launch set at all: the standing anchor spelling, nothing of the own-anchor vocabulary (the live mirrors' records stay byte-identical)
    cp = _run(bt, COMPOSED)
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"] and "own_none_state" not in cp["extra"] and cp["extra"]["anchor_rule"].startswith(M.ANCHOR_RULE)
    assert "own_none_state" not in c_comp["extra"] and c_comp["extra"]["anchor_view_legacy"] == TAG and c_comp["extra"]["acc_anchor_over_pairs"] == 1.0, "the seed none is correct on every item the composed row scored"
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is False and facts["contrasts"][f"{TAG}|{RELU}"]["own_standup"] is True


# --------------------------------------------------------------------------------------------
# refusals by name (the row, never the group)
# --------------------------------------------------------------------------------------------

def test_a_store_attesting_another_sha_than_the_pins_is_refused_and_the_siblings_file(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = _mirror(tmp_path, seeds, attest_sha={RELU: other})
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, TAG, RELU)
    assert "fit_provenance.object_sha" in why and "no prefix of the pinned object" in why and len(bt["run"]) == 4
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, PROMPTED, ARDITI, COMPOSED}


def test_a_launch_set_expecting_another_sha_than_the_pins_is_refused_by_name(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()

    def bump(doc):
        for row in doc["evals"][M.EVAL_ID]["launch"]:
            if row["cell"] == COMPOSED:
                row["object_sha256_expected"] = other
    fx = _mirror(tmp_path, seeds, launch_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, TAG_P, COMPOSED)
    assert "the launch set expects object" in why and other[:16] in why and len(bt["run"]) == 4


def test_a_launch_set_pin_key_that_is_not_the_rosters_staged_id_is_refused(tmp_path, seeds):
    def bump(doc):
        for row in doc["evals"][M.EVAL_ID]["launch"]:
            if row["cell"] == ARDITI:
                row["pin_key"] = "ar_c0p5__arditi_allpos"
    fx = _mirror(tmp_path, seeds, launch_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    assert "pin key 'ar_c0p5__arditi_allpos' is not the roster's staged id" in _refused(facts, TAG, ARDITI) and len(bt["run"]) == 4


def test_a_store_pin_that_is_not_the_manifests_tree_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pins_override={M.pin_key(SEL, RELU): "f" * 40})
    bt, facts = _read(_adapter(fx, seeds))
    assert "store pin" in _refused(facts, TAG, RELU) and len(bt["run"]) == 4


def test_a_row_without_a_store_pin_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    p = os.path.join(fx["root"], M.PINS_FILE)
    pins = json.load(open(p, encoding="utf-8"))
    pins.pop(M.pin_key(SEL, PROMPTED))
    _write(p, pins)
    bt, facts = _read(_adapter(fx, seeds))
    assert "no store pin" in _refused(facts, TAG, PROMPTED) and len(bt["run"]) == 4


def test_a_row_the_pins_meta_refused_names_never_files(tmp_path, seeds):
    def bump(meta):
        meta["refused"][M.pin_key(SEL, RELU)] = {"result": "DRIVER_FAILED", "line": "synthetic"}
    fx = _mirror(tmp_path, seeds, meta_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    assert "_meta.refused names this row" in _refused(facts, TAG, RELU) and len(bt["run"]) == 4


def test_a_three_number_row_that_disagrees_refuses_the_row_naming_the_number(tmp_path, seeds):
    def bump(three):
        row = three["rows"][f"{TAG}/{RELU.replace('__', '--', 1)}"]
        row["n_scored_final"] -= 1
        row["third_state"]["n_at_cap"] += 2
    fx = _mirror(tmp_path, seeds, three_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, TAG, RELU)
    assert "n_scored_final: producer 468, re-derived 469" in why and "third_state.n_at_cap: producer 35, re-derived 33" in why and len(bt["run"]) == 4


def test_a_three_number_row_whose_loop_split_does_not_partition_the_at_cap_items_is_refused(tmp_path, seeds):
    def bump(three):
        row = three["rows"][f"{TAG}/none"]
        row["third_state"]["loop_eligible_split"]["loop_ids"].append(ITEM_IDS[0])
        row["third_state"]["loop_eligible_split"]["loops"] += 1
    fx = _mirror(tmp_path, seeds, three_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    assert "loop / eligible split disagrees" in _refused(facts, TAG, NONE) and len(bt["run"]) == 4
    assert all(c["anchor"] is None for c in facts["contrasts"].values()), "no `none`, no contrast — the rows still file"


def test_a_final_state_entry_that_disagrees_refuses_the_row_naming_the_count(tmp_path, seeds):
    def bump(fs):
        row = fs["stores"][f"{TAG}/{PROMPTED.replace('__', '--', 1)}/{M.STORE_DIR}"]
        row["n_correct"] += 1
        iid = next(iter(row["items"]))
        row["items"][iid]["correct"] = not row["items"][iid]["correct"]
    fx = _mirror(tmp_path, seeds, fs_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, TAG, PROMPTED)
    assert "n_correct: producer 451, re-derived 450" in why and "1 item(s) disagree" in why and len(bt["run"]) == 4


def test_a_launch_set_row_with_no_mirrored_store_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, drop=(ARDITI,))
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, TAG, ARDITI)
    assert "no store is mirrored at" in why and f"{M.STORES_HEAD}/{TAG}/{ARDITI}/{M.STORE_DIR}" in why and len(bt["run"]) == 4


def test_a_store_the_launch_set_does_not_declare_is_refused_by_name(tmp_path, seeds):
    def bump(doc):
        doc["evals"][M.EVAL_ID]["launch"] = [r for r in doc["evals"][M.EVAL_ID]["launch"] if r["cell"] != PROMPTED]
    fx = _mirror(tmp_path, seeds, launch_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    assert "the launch set names no row" in _refused(facts, TAG, PROMPTED) and len(bt["run"]) == 4


def test_a_replicate_seed_base_that_is_not_the_tags_index_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, seed_base_override=3)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 5 and all("seed_base 3" in x and f"replicate at {SEED_N}" in x for x in facts["refusals"]) and "run" not in bt


def test_a_launch_set_absent_from_the_mirror_refuses_every_row_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    os.remove(os.path.join(fx["root"], M.RECORDS_DIR, M.LAUNCH_SET_FILE))
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 5 and all(M.LAUNCH_SET_FILE in x for x in facts["refusals"]) and "run" not in bt


def test_the_reader_never_consults_the_registry_for_the_object(tmp_path, seeds):
    """An object row absent from a connected registry is NOT a refusal here: the run is emitted on the attested identity and blocked_records holds it.
    (The connection is a primed frame -- the anchor currency reads `v_run_current` from it -- with an EMPTY `object` table beside it.)"""
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    frame = _primed_frame(ad)
    frame.con.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    bt, facts = _read(ad, conn=frame.con)
    assert facts["n_refusals"] == 0 and _run(bt, RELU)["object_id"] == OBJ and _run(bt, ARDITI)["object_id"] == OBJ_A
    assert frame.con.execute("SELECT COUNT(*) FROM object").fetchone()[0] == 0 and len(bt["run_contrast"]) == 4


def test_a_missing_final_state_record_is_not_a_refusal_and_is_named_on_the_run(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, without_final_state=True)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["final_state"] is None
    tn = _run(bt, RELU)["extra"]["three_numbers"]
    assert tn["final_state_record"] is None and tn["final_state_record_agrees"] is None and tn["producer_record_agrees"] is True


# --------------------------------------------------------------------------------------------
# end to end: the harvester's rule-scored completion
# --------------------------------------------------------------------------------------------

def _registry_root(tmp_path, seeds, objects) -> str:
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in objects:
        r = LC.standin_object_record(o, seeds, F.producer(), now, "qwen35_4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def test_runs_reach_ingested_through_the_rule_scored_completion_with_the_value_row_the_owner(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    objects = [ad.resolve_object(RELU, SEL)[0], ad.resolve_object(COMPOSED, SEL_P)[0], ad.resolve_object(ARDITI, SEL, bank="ar_c0p3")[0]]
    root = _registry_root(tmp_path, seeds, objects)
    # TWO passes (emission 2, the anchor currency; no provisional anchors): the first pass over the fresh registry files every run and PARKS every
    # comparison; the fold makes the seed `none` current and the anchor decision input moves, so the second pass re-reads the group under a new revision id
    rep0, ing0, ing0b = F.run_and_ingest(root, ad, now="2026-09-23T07:00:00.000Z")       # strictly before NOW: two revisions with one observed_utc have no order
    assert rep0.groups_refused == 0 and ing0.refused + ing0b.refused == 0, ([r.error for r in rep0.results], ing0.refusals, ing0b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [(5,)]
    parked = F.q(root, "SELECT json_extract(rm.extra_json, '$.delta_block.anchor.own_reason') FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                       "WHERE r.lane_source_legacy = ? AND r.object_id <> 'none' AND rm.computation_id <> ''", (M.SOURCE,))
    assert parked == [("not_in_frame",)] * 4, parked
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, lifecycle_state, n_items_planned, n_items_generated, n_items_judged, token_cap_tokens, trial_index, seed_base, "
                     "judged_utc IS NOT NULL, ingested_utc IS NOT NULL FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (M.SOURCE,))
    c = {k: _counts(v) for k, v in fx["specs"].items()}
    assert rows == [(TAG, ARDITI, "ingested", 500, 500, c[ARDITI]["n_scored"], M.CAP, SEED_N, SEED_N, 1, 1),
                    (TAG, RELU, "ingested", 500, 500, c[RELU]["n_scored"], M.CAP, 0, SEED_N, 1, 1),
                    (TAG, NONE, "ingested", 500, 500, c[NONE]["n_scored"], M.CAP, SEED_N, SEED_N, 1, 1),
                    (TAG, PROMPTED, "ingested", 500, 500, c[PROMPTED]["n_scored"], M.CAP, SEED_N, SEED_N, 1, 1),
                    (TAG_P, COMPOSED, "ingested", 500, 500, c[COMPOSED]["n_scored"], M.CAP, 0, SEED_N, 1, 1)], rows
    assert F.q(root, "SELECT COUNT(*) FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(2500,)]
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ? AND i.value IS NULL AND i.label = ?",
               (M.SOURCE, IC.ITEM_LABEL_AT_CAP_UNRESOLVED)) == [(sum(len(v["unresolved"]) for v in c.values()),)]
    m = F.q(root, "SELECT r.campaign_cell_id_legacy, rm.computation_id <> '', rm.value, rm.k, rm.n FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                  "WHERE r.lane_source_legacy = ? AND rm.metric_id = 'acc' ORDER BY 1, 2 DESC", (M.SOURCE,))
    assert [x for x in m if x[0] == RELU] == [(RELU, 1, 420 / 469, 420, 469), (RELU, 0, 420 / 500, 420, 500)]
    dg = F.q(root, "SELECT d.measure, d.measurement_state, d.numerator, d.denominator, d.rate FROM v_run_degeneration_current d JOIN run r ON r.run_id = d.run_id "
                   "WHERE r.campaign_cell_id_legacy = ? AND d.measure IN (?, ?, ?) ORDER BY 1", (RELU, M.RAW_AT_CAP_MEASURE, M.LOOP_MEASURE, M.COHERENT_MEASURE))
    assert dg == [(M.RAW_AT_CAP_MEASURE, "measured", 33, 500, 33 / 500), (M.LOOP_MEASURE, "measured", 11, 500, 11 / 500), (M.COHERENT_MEASURE, "measured", 19, 500, 19 / 500)], dg
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(4,)]
    assert sorted(F.q(root, "SELECT DISTINCT json_extract(extra_json, '$.anchor_selection.state'), anchor_kind FROM run_contrast")) == [("own_current", "same_standup_none"), ("own_current", "sibling_none")]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.lane_source_legacy = ? AND r.object_id <> 'none' AND rm.computation_id <> '' "
                     "AND json_extract(rm.extra_json, '$.delta_block') IS NULL AND json_type(rm.extra_json, '$.delta_block') = 'null'", (M.SOURCE,)) == [(4,)], \
        "a filed contrast states delta_block: null"
    # a second pass over unchanged inputs publishes nothing and the rows stay ingested
    rep2, _ing3, _ing4 = F.run_and_ingest(root, _adapter(fx, seeds), now="2026-09-23T10:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.Math500SeedrepAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "math500_seedrep"
    assert AB.emission_version_of(M.Math500SeedrepAdapter) == "2" and inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,)
    assert M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "math500_seedrep")) and M.LADDER == (16384,) and M.CAMPAIGN == "sdfbase_full22" and M.N_ITEMS == 500
    assert M.EVAL_ID == "math500" and M.INSTRUMENT_SEED == "math500_inspect_thinking_on_16384_onepass" and M.TASK == "inspect/math" and M.DECODE_SEED_RULE == "per_item"
    assert M.ITEM_LABEL_AT_CAP_UNRESOLVED == IC.ITEM_LABEL_AT_CAP_UNRESOLVED == "at_cap_unresolved" and set(M.AT_CAP_CLASSES) == {"loop", "ceiling"}
    assert M.pin_key("s1", "none") == "qwen35_4b/none/thinking_card@s1" and M.producer_key_for("qwen35_4b_s1p", "k") == "qwen35_4b_s1p/k"
    assert M.serving_dir_rel("/workspace/TESTUSER/exp/x/serving/m5t_qwen35_4b_none_thinking_card_1/math500/routing_table.json") == "serving/m5t_qwen35_4b_none_thinking_card_1/math500"
    assert M.serving_dir_rel("/nowhere/routing_table.json") is None and M.slug_to_object_key("constv2-mix-full--psr_ce_all") == RELU
    line = "2026-09-23T03:36:11+00:00 job=128075 array=54 row=54 m5t-qwen35_4b-s1-none m5t qwen35_4b none thinking_card fitted=0 rc=0 driver_rc=0 result=COMPLETE wave=qwen_seedrep thinking=on geometry=capability:gen_long_math logreq=1"
    parsed = M.parse_result_line(line)
    assert (parsed["job"], parsed["okey"], parsed["mode"], parsed["result"], parsed["job_name"], parsed["thinking"]) == ("128075", "none", "thinking_card", "COMPLETE", "m5t-qwen35_4b-s1-none", "on")


# --------------------------------------------------------------------------------------------
# the paired tag's OWN `none`
# --------------------------------------------------------------------------------------------

def test_the_paired_tags_own_none_files_as_its_anchor_and_the_composed_row_pairs_against_it_never_the_seed_rows(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none=True)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6, facts["refusals"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    none_s, none_p = runs[(TAG, NONE)], runs[(TAG_P, NONE)]
    # the paired `none`: launch-set class anchor, a run of the paired tag on the same instrument, a decode-seed trial at the seed index, its own tag-prefixed id
    assert none_p["extra"]["launch_set"]["class"] == "anchor" and none_p["object_id"] == "none" and none_p["instrument_id"] == none_s["instrument_id"] and none_p["serving_path"] == "passthrough"
    assert none_p["producer_run_key_legacy"] == f"{TAG_P}/{fx['paired_none']['key']}" == f"{TAG_P}/{fx['keys'][NONE]}", "the store key is the seed tag's, byte for byte; the tag prefix keeps the ids apart"
    assert none_p["run_id"] != none_s["run_id"] and none_p["trial_index"] == SEED_N and none_p["seed_base"] == SEED_N and none_p["view_legacy"] == TAG_P
    assert none_p["extra"]["paired_of"] == SEL and none_p["extra"]["roster_paired_of"] == SEL and none_p["extra"]["standup_replicate_of_view"] == TAG
    assert none_p["extra"]["anchor_rule"] == M.PAIRED_ANCHOR_RULE == M.PAIRED_ANCHOR_RULE and f"{TAG}/none" in none_p["extra"]["anchor_role"]
    assert none_p["extra"]["decode_replicate"]["standup_replicate_of_view"] == TAG and "paired-view-none-anchors" in none_p["extra"]["decode_replicate"]["coexist_ruling_needed"]
    assert "cross_standup_anchor" not in none_p["extra"] and "own_anchor" not in none_p["extra"] and "anchor" not in none_p["extra"]
    v_p, _b = _metrics(bt, none_p["run_id"])
    v_s, _b = _metrics(bt, none_s["run_id"])
    assert (v_p["k"], v_p["n"]) == (300, 475) and (v_s["k"], v_s["n"]) == (440, 475), "two different `none` readings at one seed base (300 + 170 + 5 scored at the cap)"
    # the contrasts: every seed row against the SEED `none`, the composed row against ITS OWN tag's `none`; no `none` is a contrast's row
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {_run(bt, k)["run_id"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)} and none_p["run_id"] not in contrasts and none_s["run_id"] not in contrasts
    for k in (RELU, PROMPTED, ARDITI):
        c = contrasts[_run(bt, k)["run_id"]]
        assert c["anchor_run_id"] == none_s["run_id"] and c["extra"]["anchor_view_legacy"] == TAG and c["extra"]["cross_standup"] is False and "own_standup_note" not in c["extra"]
        assert c["extra"]["anchor_rule"] == M.ANCHOR_RULE and facts["contrasts"][f"{TAG}|{k}"]["own_standup"] is True and facts["contrasts"][f"{TAG}|{k}"]["anchor_view_legacy"] == TAG
    c_relu = contrasts[_run(bt, RELU)["run_id"]]
    assert c_relu["n_pairs"] == 469 and abs(c_relu["extra"]["acc_anchor_over_pairs"] - 440 / 469) < 1e-12, "the seed none over the seed row's 469 shared items"
    c_comp = contrasts[_run(bt, COMPOSED)["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["anchor_view_legacy"] == TAG_P and c_comp["extra"]["row_tag_legacy"] == TAG_P
    assert c_comp["extra"]["cross_standup"] is False and "cross_standup_note" not in c_comp["extra"] and "own_none_state" not in c_comp["extra"]
    assert c_comp["extra"]["anchor_rule"] == M.PAIRED_ANCHOR_RULE and TAG_P in c_comp["extra"]["own_standup_note"] and f"{TAG}/none" in c_comp["extra"]["own_standup_note"]
    assert c_comp["n_pairs"] == 340 and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 300 / 340) < 1e-12, "the PAIRED none over the composed row's 340 shared items, not the seed none's 1.0"
    assert c_comp["computation_id"] == M.recount_computation_id() and c_comp["computed_by"] == "registry"
    # the composed run spells the anchor its contrast pairs against (the `none` rows filed first)
    cp = _run(bt, COMPOSED)
    assert cp["extra"]["own_anchor"] == f"{TAG_P}/none" and "cross_standup_anchor" not in cp["extra"] and "own_none_state" not in cp["extra"]
    assert cp["extra"]["standup_replicate_of_view"] == TAG and cp["extra"]["anchor_rule"].startswith(M.PAIRED_ANCHOR_RULE) and cp["extra"]["paired_of"] == SEL
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"] == {**facts["contrasts"][f"{TAG_P}|{COMPOSED}"], "anchor": none_p["run_id"], "cross_standup": False, "own_standup": True, "anchor_view_legacy": TAG_P}
    assert facts["rows"][f"{TAG_P}|{NONE}"]["paired"] is True and facts["rows"][f"{TAG_P}|{NONE}"]["steered"] is False and facts["rows"][f"{TAG_P}|{NONE}"]["trial_index"] == SEED_N


def test_the_paired_none_anchors_the_composed_row_even_when_the_seed_tags_none_is_absent_and_the_seed_rows_re_anchor_on_it_as_a_sibling(tmp_path, seeds):
    """The paired tag's own `none` anchors the composed row (own_current). The seed rows' OWN pairing names the seed tag's `none`, whose store is not
    mirrored (the adapter's rule never names the paired `none` for them) -- and since emission 2 the anchor currency then searches the exact key and
    finds the paired tag's `none`: the same source, another standup, a current run of the frame -> `sibling_same_source`, anchor_kind `sibling_none`,
    cross_standup true, paired per item because that `none` is a row of this very batch (the bcbw readers' precedent). When the
    seed `none` lands and folds, the decision input moves and the pairing returns to it."""
    fx = _mirror(tmp_path, seeds, paired_none=True, drop=(NONE,))
    bt, facts = _read(_adapter(fx, seeds))
    assert "no store is mirrored" in _refused(facts, TAG, NONE) and facts["runs"] == 5
    none_p = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}[(TAG_P, NONE)]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    # the paired `none` is a decode-seed trial at the seed index (trial_index = the seed base): the prompted and Arditi rows share that key and re-anchor on it;
    # the relu row is trial 0 (the one fitted object on every seed tag) and has NO unsteered run at its exact key -- it parks by name (the exact key never
    # widens to another trial); the composed row's own anchor IS the paired none and stands whatever the trials
    assert set(contrasts) == {_run(bt, k)["run_id"] for k in (COMPOSED, PROMPTED, ARDITI)}
    assert facts["anchor_states"] == {"no_current_anchor": 1, "own_current": 1, "sibling_same_source": 2} and facts["contrasts_reanchored"] == 2 and facts["contrasts_own"] == 1
    assert facts["contrasts_parked"] == 1 and facts["contrasts"][f"{TAG}|{RELU}"]["state"] == M.OWN_NONE_NOT_YET_RUN and facts["contrasts"][f"{TAG}|{RELU}"]["currency_state"] == "no_current_anchor"
    assert "has not run yet" in facts["contrasts"][f"{TAG}|{RELU}"]["why"] and _run(bt, RELU)["extra"]["anchor"]["state"] == M.OWN_NONE_NOT_YET_RUN
    assert _metrics(bt, _run(bt, RELU)["run_id"])[0]["extra"]["delta_block"]["anchor_selection"]["n_candidates"] == 0
    c_comp = contrasts[_run(bt, COMPOSED)["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["extra"]["cross_standup"] is False and c_comp["extra"]["anchor_view_legacy"] == TAG_P
    assert c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["anchor_selection"]["state"] == "own_current"
    for k in (PROMPTED, ARDITI):
        c = contrasts[_run(bt, k)["run_id"]]
        assert c["anchor_run_id"] == none_p["run_id"] and c["anchor_kind"] == "sibling_none" and c["extra"]["cross_standup"] is True and c["extra"]["anchor_view_legacy"] == TAG_P
        assert c["extra"]["anchor_selection"]["state"] == "sibling_same_source" and c["extra"]["anchor_selection"]["own_reason"] == "none_declared" and c["extra"]["cross_source"] is False
        assert c["n_pairs"] > 0 and c["computed_by"] == "registry" and c["delta_method"] == M.CONTRAST_METHOD and "a row of this batch" in c["extra"]["pairing"]
        assert "has not run yet" in c["extra"]["own_pairing"]["why"] and c["extra"]["own_pairing"]["state"] == M.OWN_NONE_NOT_YET_RUN and c["extra"]["anchor_rule"] == M.CURRENCY_RULE
        blk = _run(bt, k)["extra"]["anchor"]
        assert blk["state"] == M.ANCHOR_CONTRASTED and blk["anchor_run_id"] == none_p["run_id"] and blk["anchor_view_legacy"] == TAG_P and blk["reanchored"] is True and blk["cross_standup"] is True
        assert facts["contrasts"][f"{TAG}|{k}"]["reanchored"] is True and facts["contrasts"][f"{TAG}|{k}"]["own_standup"] is False and facts["contrasts"][f"{TAG}|{k}"]["reanchor_tier"] == "sibling_current"
    assert "anchor" not in none_p["extra"]

def test_a_paired_none_the_launch_set_names_but_no_mirror_carries_leaves_the_composed_row_on_the_fallback_and_names_it(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none="not_mirrored")
    bt, facts = _read(_adapter(fx, seeds))
    assert "no store is mirrored" in _refused(facts, TAG_P, NONE) and facts["runs"] == 5, facts["refusals"]
    cp = _run(bt, COMPOSED)
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"] and cp["extra"]["anchor_rule"].startswith(M.ANCHOR_RULE)
    assert cp["extra"]["own_none_state"] == M.OWN_NONE_NOT_YET_RUN == "anchor_not_yet_run" and f"{TAG_P}/none" in cp["extra"]["own_none_note"] and f"{TAG}/none" in cp["extra"]["own_none_note"]
    c_comp = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert c_comp["anchor_run_id"] == _run(bt, NONE)["run_id"] and c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["anchor_view_legacy"] == TAG and c_comp["extra"]["acc_anchor_over_pairs"] == 1.0
    assert c_comp["anchor_kind"] == "sibling_none", "the fallback while the own none is pending is another standup's none"
    assert c_comp["extra"]["own_none_state"] == "anchor_not_yet_run" and "did not file" in c_comp["extra"]["cross_standup_note"] and "own_standup_note" not in c_comp["extra"]
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["cross_standup"] is True and facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is False
    assert "own_none_state" not in _run(bt, RELU)["extra"] and "own_none_state" not in {c["run_id"]: c for c in bt["run_contrast"]}[_run(bt, RELU)["run_id"]]["extra"]


def test_a_paired_none_that_refuses_leaves_the_composed_row_on_the_fallback_naming_the_refusal(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none=True, meta_override=lambda meta: meta["refused"].__setitem__(M.pin_key(SEL_P, NONE), {"why": "synthetic: the producer refused the row"}))
    bt, facts = _read(_adapter(fx, seeds))
    assert "a refused row never files" in _refused(facts, TAG_P, NONE) and facts["runs"] == 5
    cp = _run(bt, COMPOSED)
    assert cp["extra"]["own_none_state"] == M.OWN_NONE_REFUSED == "anchor_refused" and cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"]
    c_comp = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert c_comp["anchor_run_id"] == _run(bt, NONE)["run_id"] and c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["own_none_state"] == "anchor_refused"
    assert c_comp["anchor_kind"] == "sibling_none", "the fallback while the own none refused is another standup's none"
    assert "anchor_refused" in c_comp["extra"]["cross_standup_note"] and c_comp["extra"]["anchor_view_legacy"] == TAG


def test_a_launch_set_filing_the_paired_none_as_composed_is_refused_by_the_grammar(tmp_path, seeds):
    def relabel(doc):
        for r in doc["evals"][M.EVAL_ID]["launch"]:
            if r["view"] == SEL_P and r["cell"] == NONE:
                r["class"] = "composed"
    fx = _mirror(tmp_path, seeds, paired_none=True, launch_override=relabel)
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, TAG_P, NONE)
    assert "class 'composed'" in why and "the key's grammar says 'anchor'" in why and facts["runs"] == 5
    cp = _run(bt, COMPOSED)
    assert cp["extra"]["own_none_state"] == "anchor_refused" and {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]["anchor_run_id"] == _run(bt, NONE)["run_id"]


# --------------------------------------------------------------------------------------------
# the launch set's ROW TABLE (the census of 2026-09-23): launch + carried
# --------------------------------------------------------------------------------------------

def test_the_launch_sets_row_table_is_launch_plus_carried_and_the_first_shape_re_reads_unchanged(tmp_path, seeds):
    def carry(doc):
        blk = doc["evals"][M.EVAL_ID]
        rows = blk["launch"]
        own = [r for r in rows if r["view"] == SEL_P and r["cell"] == NONE]
        blk["launch"] = [dict(r, anchor_tuple=None, anchor_fallback=None) for r in own]
        blk["carried"] = [dict(r, census_status="carry", anchor_tuple=f"{r['view']}/none", anchor_fallback=(f"{SEL}/none" if r["view"] == SEL_P else None)) for r in rows if r not in own]
        blk["carried_late"] = []
        blk["counts"] = {"table": len(rows), "launch": len(own), "carried": len(blk["carried"])}
    fx = _mirror(tmp_path, seeds, paired_none=True, launch_override=carry)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6, facts["refusals"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    assert _run(bt, RELU)["extra"]["launch_set"]["census_status"] == "carry" and runs[(TAG_P, NONE)]["extra"]["launch_set"]["census_status"] == "run"
    cp = _run(bt, COMPOSED)
    assert cp["extra"]["launch_set"]["anchor_tuple"] == f"{SEL_P}/none" and cp["extra"]["launch_set"]["anchor_fallback"] == f"{SEL}/none"
    assert {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]["anchor_run_id"] == runs[(TAG_P, NONE)]["run_id"]
    # the first launch set's shape (every row under `launch`, census_status run, no carried list) spells none of the census keys: its rows re-read byte for byte
    fx0 = _mirror(tmp_path, seeds, root=str(tmp_path / "first_shape"))
    bt0, _f0 = _read(_adapter(fx0, seeds))
    blk = _run(bt0, RELU)["extra"]["launch_set"]
    assert "census_status" not in blk and "anchor_tuple" not in blk and blk["class"] == "arm"
    ad0 = _adapter(fx0, seeds)
    ls0 = [r for r in ad0.source_states(_ctx(), ad0.discover()) if r.input_uri.startswith(os.path.join(fx0["root"], M.RECORDS_DIR, M.LAUNCH_SET_FILE))]
    assert ls0[0].json_path == f"$.evals.{M.EVAL_ID}.launch" and len(ls0[0].current_members) == 5


# --------------------------------------------------------------------------------------------
# the anchor currency decides every contrast
# --------------------------------------------------------------------------------------------

def test_on_an_empty_frame_every_comparison_parks_on_the_value_row_and_names_the_reason(tmp_path, seeds):
    """The first pass over a fresh registry: no own anchor is in the frame and no exact-key candidate exists, so every comparison PARKS by name -- a
    `row_refusals` line and a `delta_block` on the VALUE row (never the bound row); every steered run carries `extra.anchor` with the plan's spelling
    and the currency's reason; the `none` rows carry none; no contrast is written."""
    from sisteer.registry.adapters import anchor_current as AC
    fx = _mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds), conn=AF.Frame().con)
    assert "run_contrast" not in bt and facts["runs"] == 5 and facts["contrasts_parked"] == 4 and facts["contrasts_filed"] == 0 and facts["anchor_states"] == {"no_current_anchor": 4}
    assert facts["n_refusals"] == 4 and all(x.split(" -- ")[0].endswith("|contrast") and "contrast has no current anchor at key" in x and "not_in_frame" in x for x in facts["refusals"]), facts["refusals"]
    for okey in (RELU, PROMPTED, ARDITI, COMPOSED):
        blk = _run(bt, okey)["extra"]["anchor"]
        assert blk["state"] == "no_current_anchor" and blk["anchor_run_id"] is None and blk["currency_own_reason"] == "not_in_frame" and blk["anchor_selection"]["state"] == "no_current_anchor"
        value, bound = _metrics(bt, _run(bt, okey)["run_id"])
        blk2 = value["extra"]["delta_block"]
        assert blk2["anchor"]["state"] == "no_current_anchor" and blk2["anchor"]["own_reason"] == "not_in_frame" and blk2["delta"] is None and blk2["n_pairs"] == 0 and blk2["rule"] == AC.RULE
        assert "delta_block" not in bound["extra"] and blk2["producer_block"]["producer_record"] is not None
    assert "anchor" not in _run(bt, NONE)["extra"]
    bt2, facts2 = _read(_adapter(fx, seeds))
    assert facts2["contrasts_filed"] == 4 and facts2["contrasts_parked"] == 0 and facts2["anchor_states"] == {"own_current": 4} and facts2["reanchor_computations"] == 0
    assert all(_metrics(bt2, _run(bt2, k)["run_id"])[0]["extra"]["delta_block"] is None for k in (RELU, PROMPTED, ARDITI, COMPOSED))
    assert all(c["extra"]["anchor_selection"]["state"] == "own_current" and c["extra"]["currency_rule"] == M.CURRENCY_RULE for c in bt2["run_contrast"])
    assert all(_run(bt2, k)["extra"]["anchor"]["state"] == M.ANCHOR_CONTRASTED for k in (RELU, PROMPTED, ARDITI, COMPOSED))


def test_a_retired_seed_none_re_anchors_the_rows_at_its_key_on_a_current_sibling_of_the_frame_with_the_registrys_unpaired_rate_difference(tmp_path, seeds):
    """The seed tag's `none` (a decode-seed trial at the seed index) is superseded on the frame and the frame holds a CURRENT unsteered run of the EXACT key
    under this source from another standup that is NOT a row of this batch: the currency re-anchors the rows at that key -- the prompted and Arditi rows, trial
    = the seed index -- as `sibling_same_source` / `sibling_none` and, with no items of it in hand, files the registry's UNPAIRED rate difference from the two
    readings' k / n (`computed_by registry`, its own computation row). The relu and composed rows are trial 0 (the one fitted object on every seed tag) and
    have no unsteered run at their exact key once their own `none` is retired: they park by name with `own_reason superseded` and no candidate."""
    from sisteer.registry.adapters import anchor_current as AC
    from sisteer.registry.adapters import declared_common as DC
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    frame = _primed_frame(ad)
    none_s = {(r["payload"]["view_legacy"], r["payload"]["campaign_cell_id_legacy"]): r for r in AF.priming_read(ad, _ctx(frame.con)) if r["record_type"] == "run"}[(TAG, NONE)]["payload"]
    sib = dict(none_s, run_id="rid-sibling-none-s9", view_legacy=f"{M.MODEL_KEY}_s9", producer_run_key_legacy=f"{M.MODEL_KEY}_s9/none-store")
    jid = "jid-sibling-none-s9"
    AF.load_records(frame, [{"record_type": "run", "payload": sib},
                            {"record_type": "judgment", "payload": {"judgment_id": jid, "run_id": sib["run_id"], "judge_id": "rule", "status": "complete", "n_items_judged": 500}},
                            {"record_type": "run_metric", "payload": {"run_id": sib["run_id"], "judgment_id": "", "metric_id": M.HEADLINE_METRIC, "threshold": -1.0,
                                                                      "computation_id": M.recount_computation_id(), "value": 300 / 480, "k": 300, "n": 480, "computed_by": "registry"}}],
                    items=False)
    frame.con.execute("UPDATE v_run_current SET is_current = 0, why = 'selection_state superseded (ruling r-test)', selection_state = 'superseded' WHERE run_id = ?", (none_s["run_id"],))
    frame.con.commit()
    bt, facts = _read(ad, conn=frame.con)
    assert facts["anchor_states"] == {"no_current_anchor": 2, "sibling_same_source": 2} and facts["contrasts_filed"] == 2 and facts["contrasts_reanchored"] == 2, facts["anchor_states"]
    assert facts["contrasts_parked"] == 2 and facts["n_refusals"] == 2 and facts["reanchor_computations"] == 1, facts["refusals"]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    comp_rows = {c["computation_id"]: c for c in bt["computation"]}
    assert set(contrasts) == {_run(bt, k)["run_id"] for k in (PROMPTED, ARDITI)}
    for k in (PROMPTED, ARDITI):
        c = contrasts[_run(bt, k)["run_id"]]
        assert c["anchor_run_id"] == sib["run_id"] and c["anchor_kind"] == "sibling_none" and c["anchor_judgment_id"] == "" and c["n_pairs"] == 0 and c["computed_by"] == "registry"
        assert c["delta_method"] == DC.UNPAIRED_RATE_METHOD and c["extra"]["anchor_selection"]["state"] == "sibling_same_source" and c["extra"]["anchor_selection"]["own_reason"] == "superseded"
        assert c["extra"]["cross_standup"] is True and c["extra"]["cross_source"] is False and c["extra"]["reanchor_tier"] == "sibling_current" and c["extra"]["anchor_view_legacy"] == f"{M.MODEL_KEY}_s9"
        assert c["extra"]["own_pairing"]["own_anchor_run_id"] == none_s["run_id"] and "unpaired" in c["extra"]["pairing"] and c["extra"]["anchor_rule"] == M.CURRENCY_RULE
        assert c["extra"]["statistic_computation_id"] in comp_rows and comp_rows[c["extra"]["statistic_computation_id"]]["version"] == AC.RULE
        value, _b = _metrics(bt, c["run_id"])
        assert abs(c["delta"] - 100.0 * (value["k"] / value["n"] - 300 / 480)) < 1e-9 and value["extra"]["delta_block"] is None
        blk = _run(bt, k)["extra"]["anchor"]
        assert blk["reanchored"] is True and blk["reanchor_tier"] == "sibling_current" and blk["own_anchor_run_id"] == none_s["run_id"] and blk["anchor_run_id"] == sib["run_id"]
    for k in (RELU, COMPOSED):
        blk2 = _metrics(bt, _run(bt, k)["run_id"])[0]["extra"]["delta_block"]
        assert blk2["anchor"]["state"] == "no_current_anchor" and blk2["anchor"]["own_reason"] == "superseded" and blk2["anchor_selection"]["n_candidates"] == 0 and blk2["anchor_selection"]["key"]["trial_index"] == 0
        assert _run(bt, k)["extra"]["anchor"]["currency_own_reason"] == "superseded" and _run(bt, k)["extra"]["anchor"]["anchor_run_id"] is None
