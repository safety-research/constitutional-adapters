"""A SYNTHETIC mirror of the `ifeval_seedrep` reader's inputs (invented item ids, shas and Modal ids, `TESTHOST`, no generation text): one seed tag
with its unsteered anchor, a fitted ReLU-steer object, a fitted LoRA object, a prompted key and the Arditi all-positions object, plus the paired
tag's composed difference_relu object. The rows differ in how far the cap ladder took them, so the completion-cap merge is exercised on every
shape: a base-only row, a row escalated to 4,096 and stopped there, rows escalated to 8,192. Every store, subset, dispatcher record, ladder plan
and the producer's per-store three-number record are written from ONE synthetic truth (`truth()`), so a test asserts against numbers it can
compute by hand from the spec below.

THE SPEC, per row (items `"0"` .. `"540"`, trials 0 / 1 -> 1,082 draws; a draw "passes" when its strict value is 1):
  `base_pass`: the first N draws (row order: item 0 t0, item 0 t1, item 1 t0, ...) pass at the base; every other base draw fails
  `at_cap`: the item ids whose BOTH draws end at the cap at the base (value 0 unless named in `pass_at_cap`); `loops`: the at-cap items the
  ladder classifies loops (never re-drawn); the rest are eligible and re-drawn at the next cap when the ladder escalates
  `steps[cap]`: what the re-drawn draws did at that cap — `pass`: items whose both draws pass; `fail`: items whose both draws terminate failing;
  `at_cap`: items whose both draws end at the cap again; `loops`: the at-cap items classified loops there; the remainder are eligible for the next cap
  `pass_draws` / `fail_draws` / `at_cap_draws` / `loop_draws` name single draws for the finer cases.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import _b1_fixtures as F
from sisteer.registry.adapters import ifeval_seedrep as M
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters.base import AdapterContext

SEED_N = 2
TAG = f"qwen35_4b_s{SEED_N}"
TAG_P = f"{TAG}p"
SEL, SEL_P = f"s{SEED_N}", f"s{SEED_N}p"
NONE = "none"
RELU = "constv2-mix-full__psr_ce_all"
LORA = "constv2-mix-full__lora_r4_ce"
PROMPTED = "prompted__safety_refusal"
ARDITI = "arditi_allpos"
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"
STAGED = f"base64s{SEED_N}__{RELU}"
STAGED_L = f"base64s{SEED_N}__{LORA}"
STAGED_P = f"paired_eb64s{SEED_N}__{COMPOSED}"
STAGED_A = "ar_c0p3__arditi_allpos"
OBJ = hashlib.sha256(b"synthetic ifeval seedrep relu s2").hexdigest()
OBJ_L = hashlib.sha256(b"synthetic ifeval seedrep lora s2").hexdigest()
OBJ_P = hashlib.sha256(b"synthetic ifeval seedrep composed s2p").hexdigest()
OBJ_A = hashlib.sha256(b"synthetic ifeval seedrep arditi allpos").hexdigest()
TREE = hashlib.sha1(b"synthetic ifeval seedrep tree").hexdigest()
SHA7 = TREE[:7]
MODEL_HF = "Qwen/Qwen3.5-4B"
NOW = "2026-09-23T12:00:00.000Z"
SEED_RULE = "sha256(f'{20260909}|{trial}|ifeval|{item_key}')[:4] big-endian (no modulus); trial = sample_idx; item_key = the Inspect sample id"
DEFAULT_RULE = "sha256(item_id NUL sample_idx)[:4] big-endian mod 2^31-1"
ITEMS = [str(i) for i in range(M.N_ITEMS)]
DRAWS: List[Tuple[str, int]] = [(iid, t) for iid in ITEMS for t in range(M.K_TRIALS)]
VOLUME = f"{M.VOLUME_HEAD}{SHA7}"
_REVISION: Optional[str] = None
_SI_SHA: Optional[str] = None
DECODE = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
AT = {2048: "2026-09-23T04:48:02Z", 4096: "2026-09-23T07:07:01Z", 8192: "2026-09-23T08:31:24Z"}
STARTED = {2048: "2026-09-23T04:45:12Z", 4096: "2026-09-23T07:05:30Z", 8192: "2026-09-23T08:30:05Z"}


def bind_seeds(seeds: LC.Seeds) -> None:
    global _REVISION, _SI_SHA
    _REVISION = str(seeds.models["qwen35_4b_it"]["hf_revision"])
    _SI_SHA = str(seeds.si_object_id("safety_refusal")).split(":", 1)[1]


def ctx(conn=None) -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=F.producer(), conn=conn)


def write(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return path


def items_draws(items: Sequence[str]) -> List[Tuple[str, int]]:
    return [(str(i), t) for i in items for t in range(M.K_TRIALS)]


# --------------------------------------------------------------------------------------------
# the synthetic truth: per row, the draws at every cap of its chain
# --------------------------------------------------------------------------------------------

#: per row: the base and the cap steps (module docstring, THE SPEC)
SPECS: Dict[str, dict] = {
    NONE: {"base_pass": 800, "at_cap": [], "at_cap_draws": [("538", 0), ("539", 0), ("539", 1), ("540", 0), ("540", 1)], "loops": [], "steps": {}},
    RELU: {"base_pass": 500, "at_cap": [str(i) for i in range(500, 515)], "loops": ["500", "501"], "pass_at_cap": [("514", 0)],
           "steps": {4096: {"pass": [str(i) for i in range(502, 512)], "fail": ["512"], "at_cap": ["513", "514"], "loops": ["513"]}}},
    LORA: {"base_pass": 300, "at_cap": [str(i) for i in range(500, 520)], "loops": [str(i) for i in range(500, 505)],
           "steps": {4096: {"pass": [str(i) for i in range(505, 515)], "fail": ["515", "516"], "at_cap": ["517", "518", "519"], "loops": ["517"]},
                     8192: {"pass": ["518"], "fail_draws": [("519", 0)], "at_cap_draws": [("519", 1)]}}},
    PROMPTED: {"base_pass": 700, "at_cap": [], "loops": [], "steps": {}},
    ARDITI: {"base_pass": 750, "at_cap": ["540"], "at_cap_draws": [("539", 0)], "loops": [], "loop_draws": [("540", 0)], "steps": {}},
    COMPOSED: {"base_pass": 600, "at_cap": [str(i) for i in range(500, 515)], "loops": ["500", "501", "502"],
               "steps": {4096: {"pass": [str(i) for i in range(503, 511)], "fail": ["511"], "at_cap": ["512", "513", "514"], "loops": ["512"]},
                         8192: {"pass": ["513"], "pass_draws": [("514", 0)], "at_cap_draws": [("514", 1)]}}},
}
#: the PAIRED tag's own `none` (`mirror(paired_none=True)`): base-only, one draw at the cap, 700 passes against the seed tag's 800 so an anchor's
#: pass rate over the pairs tells which `none` a contrast used; the seed tag's `none` store key byte for byte (the tag-prefixed producer key keeps the ids apart)
SPECS_P: Dict[str, dict] = {NONE: {"base_pass": 700, "at_cap": [], "at_cap_draws": [("540", 1)], "loops": [], "steps": {}}}
ROW_P = (NONE, True, None, None, "2ff9cdff", {2048: "a4579cb7", 4096: "0b6e2a19", 8192: "7d3f11aa"})
#: (object key, paired?, staged id, object sha, ihash8, the per-cap chash8)
ROWS = [(NONE, False, None, None, "2ff9cdff", {2048: "a4579cb7"}),
        (RELU, False, STAGED, OBJ, "d6dc3d5f", {2048: "aee3abab", 4096: "3318f4e0"}),
        (LORA, False, STAGED_L, OBJ_L, "18984b43", {2048: "295b3dbc", 4096: "5d4141bb", 8192: "e83f5618"}),
        (PROMPTED, False, None, None, "faa4c8cf", {2048: "6baef1b7"}),
        (ARDITI, False, STAGED_A, OBJ_A, "92b66dce", {2048: "62c3f48f"}),
        (COMPOSED, True, STAGED_P, OBJ_P, "76c03d28", {2048: "c0f39180", 4096: "dce54718", 8192: "f5fabca9"})]


def truth(okey: str, spec: Optional[dict] = None) -> Dict[str, Any]:
    """The row's draws at every cap of its chain: ``caps[cap][draw] = {value, at_cap, loop}`` (a step's draws are the ones it re-drew), ``chain``, the
    eligible / loop draws per cap (the ladder's classification), and the merged FINAL draws with their source cap. ``spec`` overrides ``SPECS[okey]``
    (the paired tag's own ``none``, ``SPECS_P``)."""
    sp = spec if spec is not None else SPECS[okey]
    base: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for n, k in enumerate(DRAWS):
        base[k] = {"value": 1.0 if n < sp["base_pass"] else 0.0, "at_cap": False, "loop": False}
    at_cap = set(items_draws(sp.get("at_cap", []))) | set(sp.get("at_cap_draws", []))
    loops = set(items_draws(sp.get("loops", []))) | set(sp.get("loop_draws", []))
    for k in at_cap:
        base[k].update(at_cap=True, value=(1.0 if k in set(sp.get("pass_at_cap", [])) else 0.0), loop=(k in loops))
    caps = {M.CAP_BASE: base}
    eligible_by_cap = {M.CAP_BASE: sorted(at_cap - loops)}
    loops_by_cap = {M.CAP_BASE: sorted(loops)}
    chain = [M.CAP_BASE]
    prev_eligible = at_cap - loops
    for cap in M.LADDER[1:]:
        st = sp["steps"].get(cap)
        if st is None:
            break
        draws: Dict[Tuple[str, int], Dict[str, Any]] = {}
        passes = set(items_draws(st.get("pass", []))) | set(st.get("pass_draws", []))
        fails = set(items_draws(st.get("fail", []))) | set(st.get("fail_draws", []))
        capped = set(items_draws(st.get("at_cap", []))) | set(st.get("at_cap_draws", []))
        lp = set(items_draws(st.get("loops", []))) | set(st.get("loop_draws", []))
        assert passes | fails | capped == prev_eligible and not (passes & fails) and not (passes & capped) and not (fails & capped), (okey, cap)
        for k in prev_eligible:
            draws[k] = {"value": 1.0 if k in passes else 0.0, "at_cap": k in capped, "loop": k in lp}
        caps[cap] = draws
        chain.append(cap)
        eligible_by_cap[cap] = sorted(capped - lp)
        loops_by_cap[cap] = sorted(lp)
        prev_eligible = capped - lp
    final: Dict[Tuple[str, int], Dict[str, Any]] = {k: dict(v, source_cap=M.CAP_BASE) for k, v in base.items()}
    for cap in chain[1:]:
        for k, v in caps[cap].items():
            final[k] = dict(v, source_cap=cap)
    return {"caps": caps, "chain": chain, "eligible": eligible_by_cap, "loops": loops_by_cap, "final": final}


def merged_at_cap_before(tr: Dict[str, Any], cap: int) -> int:
    """The at-cap draws of the merged state BEFORE the step into ``cap`` (the plan's decision reads it)."""
    state = dict(tr["caps"][M.CAP_BASE])
    for c in tr["chain"][1:]:
        if c >= cap:
            break
        state.update(tr["caps"][c])
    return sum(1 for d in state.values() if d["at_cap"])


def expected(okey: str) -> Dict[str, Any]:
    """The hand-checkable numbers of a row: the merged strict passes over the 1,082 draws, the final at-cap draws by class, the chain."""
    tr = truth(okey)
    final = tr["final"]
    final_cap = tr["chain"][-1]
    passes = int(sum(d["value"] for d in final.values()))
    at_cap = {k for k, d in final.items() if d["at_cap"]}
    loops = {k for k, d in final.items() if d["at_cap"] and d["loop"]}
    ceiling = {k for k in at_cap - loops if final[k]["source_cap"] == M.CAP_CEILING}
    not_esc = at_cap - loops - ceiling
    base_passes = int(sum(d["value"] for d in tr["caps"][M.CAP_BASE].values()))
    return {"chain": tr["chain"], "final_cap": final_cap, "passes": passes, "value": passes / M.N_DRAWS, "base_value": base_passes / M.N_DRAWS, "n_at_cap": len(at_cap),
            "n_loops": len(loops), "n_ceiling": len(ceiling), "n_not_escalated": len(not_esc), "base_at_cap": sum(1 for d in tr["caps"][M.CAP_BASE].values() if d["at_cap"]),
            "draws_from_cap": {c: sum(1 for d in final.values() if d["source_cap"] == c) for c in tr["chain"]}}


# --------------------------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------------------------

def slug_of(okey: str) -> str:
    return okey.replace("__", "--", 1)


def class_of(okey: str, paired: bool) -> str:
    """The launch set's class grammar (the reader's): the Arditi key `arditi`; `none` `anchor` on EITHER tag (the paired tag's own anchor); a
    prompted key `prompted`; a fitted key `composed` under the paired tag, else `arm`."""
    if okey == ARDITI:
        return "arditi"
    if okey == NONE:
        return "anchor"
    if okey.startswith("prompted__"):
        return "prompted"
    return "composed" if paired else "arm"


def harness_id(okey: str) -> str:
    if okey in (NONE, ARDITI):
        return okey
    if okey == PROMPTED:
        return "prompted__safety"
    return "si_" + okey.replace("-", "_")


def store_key_of(okey: str, ihash8: str, chash8: str) -> str:
    return f"{slug_of(okey)}__controls__t0__a0__{ihash8}__{chash8}"


def _trial_means(draws: Dict[Tuple[str, int], Dict[str, Any]], loose: Dict[Tuple[str, int], float]) -> Dict[str, list]:
    out = {"prompt_strict_mean_by_trial": [], "prompt_loose_mean_by_trial": [], "inst_strict_mean_by_trial": [], "inst_loose_mean_by_trial": []}
    for t in range(M.K_TRIALS):
        ks = [k for k in draws if k[1] == t]
        if not ks:
            for f in out:
                out[f].append(None)
            continue
        out["prompt_strict_mean_by_trial"].append(sum(draws[k]["value"] for k in ks) / len(ks))
        out["prompt_loose_mean_by_trial"].append(sum(loose[k] for k in ks) / len(ks))
        n_inst = sum(2 for _ in ks)
        out["inst_strict_mean_by_trial"].append(sum(2 * draws[k]["value"] for k in ks) / n_inst)
        out["inst_loose_mean_by_trial"].append(sum(2 * loose[k] for k in ks) / n_inst)
    return out


def store_reading_of(draws: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    """The producer's own per-store aggregate: acc = the mean over the store's items of each item's mean over the store's draws."""
    by_item: Dict[str, List[float]] = {}
    for (iid, _t), d in draws.items():
        by_item.setdefault(iid, []).append(d["value"])
    acc = sum(sum(v) / len(v) for v in by_item.values()) / len(by_item)
    loose = {k: (1.0 if (d["value"] == 1.0 or (k[1] == 1 and not d["at_cap"] and int(k[0]) % 7 == 0)) else 0.0) for k, d in draws.items()}
    n_at_cap = sum(1 for d in draws.values() if d["at_cap"])
    return {"acc": acc, "n_items": len(by_item), "n_draws": len(draws), "n_at_cap": n_at_cap, "loose": loose, **_trial_means(draws, loose)}


def store_files(root: str, tag: str, okey: str, cap: int, tr: Dict[str, Any], *, obj_sha, ihash8: str, chash8: str, prev_chash8: Optional[str], obj_sha_attested=None,
                result_complete: bool = True, replicate_block: bool = False, max_lora_rank: int = 8) -> str:
    """One store directory (the base or a cap step) and the named files the reader opens; returns the store key. ``obj_sha_attested`` makes the manifest
    attest another object than the pins."""
    slug = slug_of(okey)
    d = os.path.join(root, M.rel_store(tag, slug, cap))
    store_key = store_key_of(okey, ihash8, chash8)
    kd = os.path.join(d, "cells", store_key)
    steered = okey not in (NONE, PROMPTED)
    a_sha = (obj_sha_attested or obj_sha) if steered else None
    ident = M.STEERED_IDENTITY if steered else M.PASSTHROUGH_IDENTITY
    hid = harness_id(okey)
    draws = tr["caps"][cap]
    rd = store_reading_of(draws)
    outcomes = []
    rows: List[str] = []
    for n, (k, dd) in enumerate(sorted(draws.items(), key=lambda kv: (int(kv[0][0]), kv[0][1]))):
        iid, t = k
        at_cap = dd["at_cap"]
        loose_v = rd["loose"][k]
        outcomes.append({"at_cap": at_cap, "attempt": 1, "completion_tokens": (cap if at_cap else 300), "content_empty_with_reasoning": False, "inst_level_loose": int(2 * loose_v),
                         "inst_level_strict": int(2 * dd["value"]), "instruction_id_list": ["synthetic:a", "synthetic:b"], "item_id": iid, "num_instructions": 2,
                         "prompt_level_loose": bool(loose_v), "prompt_level_strict": bool(dd["value"]), "prompt_tokens": 50, "sample_idx": t, "sampling_ref": "21cf301f757f",
                         "scored": True, "seed": 1000 + 2 * int(iid) + t, "stop_reason": ("max_tokens" if at_cap else "stop"), "stray_open_tag": False, "think_block_present": False,
                         "think_block_unterminated": False, "unparseable": False, "value": dd["value"]})
        rows.append(json.dumps({"arm_name": slug, "attempt": 0, "coeff": 1.0, "created_at": 1790138800.0 + n, "domain": "inspect_gen", "engine": "vllm", "env_digest": "108112cfb52ae5df",
                                "extra_json": json.dumps({"answer": "synthetic", "stop_reason": ("max_tokens" if at_cap else "stop")}), "harness_config_hash": chash8 + "85257254",
                                "intervention_hash": ihash8 + "2e209076", "item_id": iid, "kind": "cell", "metric": f"{M.HEADLINE_ROW_METRIC}@{t}", "model_id": MODEL_HF, "model_revision": _REVISION,
                                "run_id": f"served-controls-{slug}", "seed": 0, "slurm_job_id": None, "task": M.TASK, "task_version": "inspect_evals-0.18.0", "trial_id": 0, "value": dd["value"]}))
    if cap == M.CAP_BASE:
        by_item: Dict[str, List[float]] = {}
        for (iid, _t), dd in draws.items():
            by_item.setdefault(iid, []).append(dd["value"])
        for iid in sorted(by_item, key=int):
            rows.append(json.dumps({"arm_name": slug, "item_id": iid, "kind": "cell", "metric": M.HEADLINE_ROW_METRIC, "task": M.TASK, "value": sum(by_item[iid]) / len(by_item[iid]),
                                    "created_at": 1790138869.0, "harness_config_hash": chash8 + "85257254", "intervention_hash": ihash8 + "2e209076", "model_id": MODEL_HF, "model_revision": _REVISION,
                                    "env_digest": "108112cfb52ae5df", "extra_json": None}))
    agg = {"object_artifact_sha256": a_sha, "producer_tree_sha": TREE, "route_kind": ("lens" if steered else ("prompted" if okey == PROMPTED else "base")), "view": SEL_P if tag == TAG_P else SEL,
           "samples_per_item": M.K_TRIALS, "n": rd["n_items"], "n_at_cap": rd["n_at_cap"]}
    rows.append(json.dumps({"item_id": "__aggregate__", "kind": "cell", "metric": "acc_aggregate", "task": M.TASK, "value": rd["acc"], "extra_json": json.dumps(agg)}))
    rows.append(json.dumps({"item_id": "__aggregate__", "kind": "cell", "metric": "n", "task": M.TASK, "value": float(rd["n_items"]), "extra_json": None}))
    write(os.path.join(kd, "rows.jsonl"), "\n".join(rows) + "\n")
    write(os.path.join(kd, "outcomes", M.OUTCOMES_FILE), {"item_set": "ifeval541", "items": outcomes, "max_tokens": cap, "n_at_cap": rd["n_at_cap"], "n_attempted_items": rd["n_items"],
                                                          "n_attempted_samples": rd["n_draws"], "n_completion_tokens": rd["n_draws"], "n_scored": rd["n_items"], "n_scored_samples": rd["n_draws"],
                                                          "n_stop_reason": rd["n_draws"], "n_think_block_present": 0, "n_think_block_unterminated": 0, "n_unscored": 0, "n_unscored_samples": 0,
                                                          "n_unterminated_not_at_cap": 0, "producer_tree_sha": TREE, "samples_per_item": M.K_TRIALS, "task": M.TASK, "think_split_mode": "record_only",
                                                          "thinking": "off", "unterminated_policy": "unscored"})
    gs = {"acc": rd["acc"], "n": rd["n_items"], "n_requests": rd["n_draws"], "n_rejected": 0, "samples_per_item": M.K_TRIALS, "task": M.TASK, "item_set": "ifeval541", "framework": "inspect_evals-0.18.0",
          "engine": "vllm", "route": f"petri__{hid}", "thinking": "off", "render": "native", "n_scored": rd["n_items"], "n_unscored": 0, "n_scored_samples": rd["n_draws"], "n_unscored_samples": 0,
          "n_at_cap": rd["n_at_cap"], "at_cap_rate": rd["n_at_cap"] / rd["n_draws"], "max_tokens": cap, **DECODE, "sampling_ref": "21cf301f757f", "ifeval_task_version": "2-A", "langdetect_seed": 0,
          "unterminated_policy": "unscored", "wall_clock_s": 143.3, **{f: rd[f] for f in M.TRIAL_MEAN_FIELDS}}
    write(os.path.join(kd, "gen_stats.json"), {M.TASK: gs})
    write(os.path.join(kd, "summary", M.OUTCOMES_FILE), dict(gs))
    write(os.path.join(kd, "done", M.OUTCOMES_FILE), {"finished_at": 1790138869.5, "n_draws": rd["n_draws"], "n_expected_draws": rd["n_draws"], "n_items": rd["n_items"], "n_planned": rd["n_items"],
                                                      "n_rejected": 0, "n_requests": rd["n_draws"], "n_rows": len(rows) - 2, "task": M.TASK, "wall_clock_s": 143.3})
    write(os.path.join(kd, "evidence.json"), {"serving_identity": ident, "cell_id": okey})
    seeds_block: Dict[str, Any] = {"inspect": {"ifeval": {"derivation": "inspect_served.ifeval_sample_seed", "readout": "ifeval", "rule": SEED_RULE, "seed_base": M.IFEVAL_SEED_BASE}, "rule": DEFAULT_RULE,
                                               "samples_per_item": {M.TASK: M.K_TRIALS}}}
    if replicate_block:
        seeds_block["inspect"]["replicate"] = {"policy": "replicate", "seed_base": SEED_N}
    man: Dict[str, Any] = {"accepted_generation_attempt": "a1", "cell_id": okey, "harness_cell_id": hid, "model_id": MODEL_HF, "model_revision": _REVISION, "serving_identity": ident,
                           "serving_path": ("lora" if steered else "passthrough"), "serving_engine": "lora_native", "steer_positions": "all", "engine": "vllm", "virtual_model": f"petri__{hid}",
                           "fit_provenance": {"object_sha": (a_sha[:16] if steered else None), "model_id": ((MODEL_HF if okey == ARDITI else "Qwen/Qwen3.5-4B-Base") if steered else None),
                                              "transfer": (steered and okey != ARDITI), "fit_kind": ("class_b" if okey == ARDITI else ("document" if steered else None)),
                                              "status": ("verified" if okey == ARDITI else ("document_fit" if steered else None)), "served_model_id": MODEL_HF, "served_revision": _REVISION},
                           "producer": {"driver": "served_capability_run", "tree_sha": TREE}, "row_constants": {"slurm_job_id": None, "env_digest": "108112cfb52ae5df", "arm_name": slug, "seed": 0},
                           "harness_config_hash": chash8 + "85257254", "intervention_hash": ihash8 + "2e209076",
                           "geometry": {"max_model_len": cap + 480, "max_num_seqs": 207, "source": "endpoint.json+served_geometry.json", "eval": f"capability:ifeval_{cap}"},
                           "seeds": seeds_block, "frameworks": {"inspect_ai": "0.3.260", "inspect_evals": "0.18.0", "client_lock_sha256": hashlib.sha256(b"synthetic client lock").hexdigest()},
                           "sampling": {"mode": "vendor_preset", "model_key": "qwen35_4b", **DECODE, "source": "synthetic vendor preset", "seed_rule": {"inspect": DEFAULT_RULE, M.TASK: SEED_RULE}},
                           "resolved_config": {"cell_id": okey, "system_sha": (_SI_SHA[:16] if okey == PROMPTED else None), "serving": {"serving_engine": "lora_native", "serving_identity": ident}},
                           "provenance": {"artifact_sha256": a_sha, "cell_id": okey, "class": ("B" if okey == ARDITI else ("A" if steered else "baseline")), "serving": {"serving_identity": ident}},
                           "route": {"bank_view": (SEL_P if tag == TAG_P else SEL), "object_artifact_sha256": a_sha, "object_artifact_sha256_by_cell": ({hid: a_sha} if steered else None),
                                     "serving_identity": ident, "serving_engine": "lora_native", "tool_calls": False},
                           "thinking": {"mode": "off", "enable_thinking": False, "render": "native", "by_task": {M.TASK: "off"}},
                           "tinycap": {"max_tokens_by_task": {M.TASK: cap}, "samples_by_task": {M.TASK: M.K_TRIALS}, "served": {"serving_engine": "lora_native", "max_model_len": cap + 480, "max_num_seqs": 207},
                                       "thinking": {"mode": "off", "render": "native"}},
                           "resume_attempts": [{"attempt": 1, "attempt_id": "a1", "completed": True, "ended": AT[cap], "generation_calls": rd["n_draws"], "kind": "generation", "started": STARTED[cap],
                                                "terminal": "completed", "tree_sha": TREE, "ts": STARTED[cap]}],
                           "usage_ledger": f"{VOLUME}/{M.rel_store(tag, slug, cap)}/usage_ledger.jsonl", "oos_experiment": None}
    if cap != M.CAP_BASE:
        prev = M.LADDER[M.LADDER.index(cap) - 1]
        redrawn = sorted(draws, key=lambda k: (int(k[0]), k[1]))
        man["item_subset"] = {"cap": cap, "draws": [list(k) for k in redrawn], "ids": sorted({k[0] for k in redrawn}, key=int), "n": len({k[0] for k in redrawn}), "n_draws": len(redrawn),
                              "n_items": len({k[0] for k in redrawn}), "path": f"{VOLUME}/{M.rel_subset(tag, slug, cap)}", "schema": M.SUBSET_SCHEMA, "source_cap": prev,
                              "source_identity_hash": prev_chash8 + "85257254", "source_out_root": f"{VOLUME}/{M.rel_store(tag, slug, prev)}", "subset_sha16": hashlib.sha256(json.dumps(redrawn).encode()).hexdigest()[:16],
                              "task": M.TASK}
        write(os.path.join(d, M.LADDER_STATE_FILE), {"accepted_generation_attempt": None, "attempts": [], "schema": M.LADDER_STATE_SCHEMA,
                                                     "rung": {"cap": cap, "n_draws": len(redrawn), "n_items": len({k[0] for k in redrawn}), "source_identity_hash": prev_chash8 + "85257254",
                                                              "source_out_root": f"{VOLUME}/{M.rel_store(tag, slug, prev)}", "state": "planned", "subset_path": f"{VOLUME}/{M.rel_subset(tag, slug, cap)}",
                                                              "subset_sha16": man["item_subset"]["subset_sha16"], "trigger_provenance": "1 % raw at-cap of draws, per the IFEval ruling"}})
    write(os.path.join(kd, "manifest.json"), man)
    lines = []
    if cap != M.CAP_BASE:
        lines.append(f"{M.RESULT_HEAD} name=tinycap-synthetic-{slug} rc=1 result=INCOMPLETE status=INCOMPLETE sha={TREE} at=2026-09-23T06:24:11Z")
    lines.append(f"{M.RESULT_HEAD} name=tinycap-synthetic-{slug} rc=0 result=COMPLETE status=OK sha={TREE} at={AT[cap]}" if result_complete
                 else f"{M.RESULT_HEAD} name=tinycap-synthetic-{slug} rc=1 result=INCOMPLETE status=INCOMPLETE sha={TREE} at={AT[cap]}")
    write(os.path.join(d, "RESULT"), "\n".join(lines) + "\n")
    write(os.path.join(d, M.GEOMETRY_FILE), {"serving_path": "lora_native", "serving_engine": "lora_native", "standup_argv": ["python", "capctrl_serving_up.py"],
                                              "geometry": {"eval": f"capability:ifeval_{cap}", "max_model_len": cap + 480, "max_num_seqs": 207},
                                              "endpoint": {"engine": "lora_native", "node": None, "lora_config": {"max_lora_rank": max_lora_rank, "max_loras": 2, "target_modules": ["o_proj"]}}})
    write(os.path.join(d, "evidence.json"), {"serving_identity": ident, "cell_id": okey})
    write(os.path.join(d, "resolved_sampler.json"), {"verdict": "AGREES"})
    write(os.path.join(d, "usage_ledger.jsonl"), json.dumps({"task": M.TASK, "finish_reason": "stop"}) + "\n")
    # the ladder's classification of THIS store (written when the ladder promoted from it)
    nxt = M.LADDER[M.LADDER.index(cap) + 1] if cap != M.CAP_CEILING else None
    if nxt is not None and nxt in tr["chain"]:
        eligible = [list(k) for k in tr["eligible"][cap]]
        loops = [list(k) for k in tr["loops"][cap]]
        write(os.path.join(d, M.CAP_LADDER_FILE), {"schema": M.CAP_LADDER_SCHEMA, "cell_id": store_key, "cap": cap, "ceiling": M.CAP_CEILING, "ladder": list(M.LADDER[M.LADDER.index(cap):]),
                                                   "eligible_draws": eligible, "eligible_ids": sorted({k[0] for k in eligible}, key=int), "loop_draws": loops, "non_terminating_ids": sorted({k[0] for k in loops}, key=int),
                                                   "n": rd["n_items"], "n_draws": rd["n_draws"], "n_at_cap": rd["n_at_cap"], "n_loop": len(loops), "n_eligible": len(eligible), "n_truncated": len(eligible),
                                                   "share_at_cap": rd["n_at_cap"] / rd["n_draws"], "exit": M.EXIT_ESCALATE, "escalate": True, "next_cap": nxt, "threshold": 0.01,
                                                   "rules": {"at_cap": "outcome.stop_reason == 'max_tokens'", "degenerate": "rep_ratio(text) > 0.5 or rep_ratio(last 200 whitespace tokens) > 0.5",
                                                             "trigger": "trigger iff n_at_cap_draws / n_draws > 0.01"},
                                                   "source": {"identity_hash": chash8 + "85257254", "is_rung": cap != M.CAP_BASE, "out_root": f"{VOLUME}/{M.rel_store(tag, slug, cap)}", "task": M.TASK},
                                                   "samples_per_item": M.K_TRIALS, "engine": "vllm", "instrument": "ifeval/cap_ladder"})
    if cap == M.CAP_BASE:
        write(os.path.join(d, okey, M.CONTROLS_SUMMARY_FILE), {"cell_id": okey, "provenance": {"artifact_sha256": a_sha, "cell_id": okey},
                                                                "controls": {M.TASK: {"acc": rd["acc"], "n": rd["n_items"], "n_scored": rd["n_items"], "n_at_cap": rd["n_at_cap"], "max_tokens": cap, "thinking": "off", "task": M.TASK}},
                                                                "sources": {M.TASK: {"cell_key": store_key, "producer_tree_sha": TREE, "harness_config_hash": chash8 + "85257254"}}})
    return store_key


def subset_file(root: str, tag: str, okey: str, cap: int, tr: Dict[str, Any], prev_chash8: str) -> None:
    slug = slug_of(okey)
    prev = M.LADDER[M.LADDER.index(cap) - 1]
    redrawn = sorted(tr["caps"][cap], key=lambda k: (int(k[0]), k[1]))
    prev_rd = store_reading_of(tr["caps"][prev])
    write(os.path.join(root, M.rel_subset(tag, slug, cap)), {"cap": cap, "draws": [list(k) for k in redrawn], "ids": sorted({k[0] for k in redrawn}, key=int), "n_at_cap": prev_rd["n_at_cap"],
                                                             "n_boundary": 0, "n_draws": len(redrawn), "n_items": len({k[0] for k in redrawn}), "n_loop": len(tr["loops"][prev]), "n_truncated": len(redrawn),
                                                             "rule": "trigger iff n_at_cap_draws / n_draws > 0.01", "schema": M.SUBSET_SCHEMA, "source_cap": prev, "source_identity_hash": prev_chash8 + "85257254",
                                                             "source_out_root": f"{VOLUME}/{M.rel_store(tag, slug, prev)}", "task": M.TASK, "threshold": 0.01})


def dispatch_record(tag: str, okey: str, cap: int, *, store_key: str, obj_sha, state: str = "done", status: str = "OK", attested: Optional[str] = "lora_native", error=None) -> dict:
    slug = slug_of(okey)
    cid = M.configuration_id(tag, slug, cap)
    return {"app_id": f"ap-synthetic{cap}", "app_name": f"ifeval-seedrep-{SHA7}", "cap": cap, "cell": okey, "configuration_id": cid, "error": error, "fc_id": f"fc-01SYNTHETIC{cap:05d}{abs(hash(cid)) % 10**8:08d}",
            "function": "run_configuration", "geometry_eval": f"capability:ifeval_{cap}", "image": "default", "item_subset": (None if cap == M.CAP_BASE else f"{VOLUME}/{M.rel_subset(tag, slug, cap)}"),
            "journal_fc_ids": None, "launch_key": f"{SHA7}-synthetic/{cid}/deadbeef0000", "name": f"tinycap-synthetic-{slug}", "object_sha256": obj_sha, "out_dir": f"{VOLUME}/{M.rel_store(tag, slug, cap)}",
            "rec": {"attempt_id": "a1", "status": status, "serving_path": "lora_native", "serving_engine": "lora_native", "object_artifact_sha256": obj_sha, "tree_sha": TREE, "sha": SHA7, "driver_rc": 0},
            "requested_serving_path": "lora_native", "reserved_at": 1790137798.9, "run_id": f"{SHA7}-synthetic", "schema": M.DISPATCH_SCHEMA, "serving_path_attested": attested,
            "serving_path_attested_source": (f"/tinycap/ifeval_seedrep/{SHA7}/{M.rel_store(tag, slug, cap)}/cells/{store_key}/manifest.json ['serving_engine'] = 'lora_native'" if attested else None),
            "slug": f"qwen35_4b-sdfbase-{SEL_P if tag == TAG_P else SEL}", "spawned_at": 1790137799.0, "state": state, "status": status, "status_unknown_error": None, "status_unknown_polls": None,
            "status_unknown_since": None, "t": 1790138921.0, "view": (SEL_P if tag == TAG_P else SEL)}


def three_row(tag: str, okey: str, cap: int, tr: Dict[str, Any], *, obj_sha, store_key: str, steered: bool, anchor_tr: dict = None) -> dict:
    slug = slug_of(okey)
    rd = store_reading_of(tr["caps"][cap])
    sel = SEL_P if tag == TAG_P else SEL
    row = {"key": (f"{tag}/{slug}" if cap == M.CAP_BASE else f"{tag}/{slug}/rung{cap}"), "tag": tag, "view": sel, "seed": SEED_N, "paired": tag == TAG_P, "cell_id": okey, "cell_slug": slug,
           "arm": LC.cell_grammar_legacy(okey)["paper_label"], "status": "complete", "cap": cap, "is_rung": cap != M.CAP_BASE, "merged": False, "store": M.rel_store(tag, slug, cap),
           "result_line": f"{M.RESULT_HEAD} name=tinycap-synthetic-{slug} rc=0 result=COMPLETE status=OK sha={TREE} at={AT[cap]}", "engine": "lora_native",
           "serving_identity": (M.STEERED_IDENTITY if steered else M.PASSTHROUGH_IDENTITY), "producer_tree": TREE, "bank_view": sel, "n_attempted": rd["n_items"], "n_scored": rd["n_items"], "correct": None,
           "value": rd["acc"], "bound": rd["acc"], "at_cap_n": rd["n_at_cap"], "at_cap_share": rd["n_at_cap"] / rd["n_draws"], "n_unscored": 0, "delta_value_vs_none": None, "note": None,
           "object_sha256": (obj_sha if steered else None), "object_sha256_source": ("route" if steered else None), "draws": rd["n_draws"], "samples_per_item": M.K_TRIALS, "max_tokens": cap,
           "summary": f"cells/{store_key}/summary/{M.OUTCOMES_FILE}", "ifeval": {**{f: rd[f] for f in M.TRIAL_MEAN_FIELDS}, "ifeval_task_version": "2-A", "item_set": "ifeval541"}}
    if anchor_tr is not None and cap == M.CAP_BASE:
        row["delta_value_vs_none"] = round(rd["acc"] - store_reading_of(anchor_tr["caps"][M.CAP_BASE])["acc"], 4)
    return row


def launch_row(sel: str, okey: str, tag: str, *, pin_key, expected_sha, paired: bool, index: int) -> dict:
    slug = slug_of(okey)
    cls = class_of(okey, paired)
    return {"tuple_id": f"{sel}/{okey}", "view": sel, "cell": okey, "paper_label": LC.cell_grammar_legacy(okey)["paper_label"], "class": cls, "block": ("composed" if paired else ("arditi" if cls == "arditi" else "seed")),
            "fitted": okey not in (NONE, PROMPTED), "seed_base": SEED_N, "pin_key": pin_key, "tag": tag, "cell_slug": slug, "table_index": index, "object_sha256_expected": expected_sha, "census_status": "run",
            "configuration_id": M.configuration_id(tag, slug, M.CAP_BASE), "configuration_ids": {str(c): M.configuration_id(tag, slug, c) for c in M.LADDER}, "out_root": M.rel_store(tag, slug, M.CAP_BASE)}


def plan_records(root: str, cap: int, rows: Sequence[Tuple[str, str, Dict[str, Any], str]], override=None) -> None:
    """``records/ladder_<cap>/plan.json`` + ``promoted.json`` for the step INTO ``cap``: ``rows`` = (tag, okey, truth, job id)."""
    prev = M.LADDER[M.LADDER.index(cap) - 1]
    planned, promoted, not_promoted = [], [], []
    for idx, (tag, okey, tr, job) in enumerate(rows):
        sel = SEL_P if tag == TAG_P else SEL
        slug = slug_of(okey)
        key = f"qwen35_4b/{sel}/{okey}"
        if prev not in tr["chain"]:
            planned.append({"cell_key": key, "job_id": job, "index": idx, "from_cap": prev, "exit": None, "status": "pending", "note": "store not complete: not classified", "decision": None,
                            "n_draws": None, "next_cap": None, "share_at_cap": None, "store_exit": None, "subset_path": None, "out_root": f"{VOLUME}/{M.rel_store(tag, slug, prev)}"})
            not_promoted.append({"job_id": job, "exit": None, "status": "pending", "note": "store not complete: not classified"})
            continue
        n_at_cap = merged_at_cap_before(tr, cap)
        eligible = [list(k) for k in tr["eligible"][prev]]
        escalate = cap in tr["chain"]
        exit_ = M.EXIT_ESCALATE if escalate else ("triggered_no_eligible" if (n_at_cap / M.N_DRAWS > 0.01 and not eligible) else "below_threshold")
        dec = {"chain": [c for c in tr["chain"] if c < cap], "decided_by": {"rule": "THE ladder decision at every rung is over the MERGED reading", "script": "scripts/tinycap_ifeval_ladder_plan.py", "sha": TREE},
               "eligible_draws": eligible, "escalate": escalate, "exit": exit_, "n_at_cap": n_at_cap, "n_draws": M.N_DRAWS, "n_eligible": len(eligible), "n_rung": 0, "next_cap": (cap if escalate else None),
               "reading": ("r2048" if prev == M.CAP_BASE else "merged"), "rule": "trigger iff n_at_cap_draws / n_draws > 0.01", "share_at_cap": round(n_at_cap / M.N_DRAWS, 6), "threshold": 0.01,
               "transport": None, "trigger_provenance": "1 % raw at-cap of draws, per the IFEval ruling"}
        planned.append({"cell_key": key, "job_id": job, "index": idx, "from_cap": prev, "exit": exit_, "status": "complete", "note": None if escalate else f"merged reading: {exit_} ({n_at_cap} of {M.N_DRAWS} at cap); no rung",
                        "decision": dec, "n_draws": (len(eligible) if escalate else None), "next_cap": (cap if escalate else None), "share_at_cap": dec["share_at_cap"], "store_exit": exit_,
                        "subset_path": (f"{VOLUME}/{M.rel_subset(tag, slug, cap)}" if escalate else None), "out_root": f"{VOLUME}/{M.rel_store(tag, slug, prev)}"})
        if escalate:
            promoted.append({"configuration_id": M.configuration_id(tag, slug, cap), "index": idx, "job_id": job, "n_draws": len(eligible), "subset_rel": M.rel_subset(tag, slug, cap),
                             "ladder_state_rel": f"{M.rel_store(tag, slug, cap)}/{M.LADDER_STATE_FILE}", "cap_ladder_rel": f"{M.rel_store(tag, slug, prev)}/{M.CAP_LADDER_FILE}", "prediction": None})
        else:
            not_promoted.append({"job_id": job, "exit": exit_, "status": "complete", "note": planned[-1]["note"]})
    plan = {"from_cap": prev, "lines": [], "planned": planned, "sha": TREE}
    prom = {"schema": M.PLAN_SCHEMA, "sha": TREE, "from_cap": prev, "cap": cap, "pull_root": "/synthetic/pull", "n_planned": len(rows), "n_promoted": len(promoted), "promoted": promoted, "not_promoted": not_promoted}
    if override:
        override(cap, plan, prom)
    write(os.path.join(root, M.RECORDS_DIR, M.plan_head(cap), M.PLAN_FILE), plan)
    write(os.path.join(root, M.RECORDS_DIR, M.plan_head(cap), M.PROMOTED_FILE), prom)


def census_row_table(rows: Sequence[dict], sel: str, sel_p: str) -> Dict[str, Any]:
    """The census's launch + carried shape (2026-09-23 14:19Z): the paired tag's `none` rows are this pass's `launch` (census_status run), every other
    row is `carried` (census_status carry); each row carries the census's anchor declaration (its own tag's `none`; the cross-standup fallback on a
    composed row) and `counts` sums the table."""
    def decorate(r: dict) -> dict:
        own_none = r["cell"] == NONE
        return dict(r, anchor_tuple=(None if own_none else f"{r['view']}/{NONE}"), anchor_fallback=(f"{sel}/{NONE}" if (r["view"] == sel_p and not own_none) else None))
    run_rows = [decorate(r) for r in rows if r["view"] == sel_p and r["cell"] == NONE]
    carried = [dict(decorate(r), census_status="carry") for r in rows if not (r["view"] == sel_p and r["cell"] == NONE)]
    return {"launch": run_rows, "carried": carried, "carried_late": [], "counts": {"table": len(rows), "launch": len(run_rows), "carried": len(carried), "duplicates_for_user": 0}}


def mirror(tmp_path, seeds: LC.Seeds, *, root: str = None, not_run: Sequence[str] = (), drop_store: Sequence[Tuple[str, int]] = (), drop_dispatch: Sequence[Tuple[str, int]] = (),
           rate_limited: Sequence[Tuple[str, int]] = (), dispatch_override=None, three_override=None, launch_override=None, plan_override=None, attest_sha: dict = None, pins_sha: dict = None,
           mirrored_pins_sha: dict = None, replicate_block: bool = False, result_incomplete: Sequence[Tuple[str, int]] = (), failed_attempt_roots: bool = True, matrix: bool = True,
           paired_none=False, launch_layout: str = "launch", paired_none_dispatch_root: str = "dispatch", paired_none_planned: bool = True, paired_none_spec: Optional[dict] = None,
           incomplete_attempt_root: bool = False) -> dict:
    """A synthetic mirror root: the seed tag's five rows and the paired tag's composed row (every cap of each row's chain), the subsets, the dispatcher
    records and states, the ladder plans of both cap steps, the launch set, the pins, the matrix, the producer's per-store three-number record, and the
    harness / all-positions roster and pins naming the synthetic objects. ``not_run`` names keys whose launch-set row gets no store; ``drop_store`` removes a
    cap's store while the plan still promotes it; ``rate_limited`` turns a dispatcher record into the rate-limited attestation read; ``failed_attempt_roots``
    writes the attempt-1 `.failed-D205` root beside the fresh dispatcher (never an input); ``paired_none`` adds the paired tag's OWN `none` (`True`: its
    base store, dispatcher record and records; `"not_run"`: the launch-set, matrix and plan rows alone, the way a row that has not run appears).
    The mirror of 2026-09-23 15:24Z: ``launch_layout="carried"`` writes the census's launch + carried shape (the paired `none` rows under `launch` with
    census_status run, every other row under `carried` with census_status carry, the census's anchor declaration on each, `counts`, the top-level
    `row_table`); ``paired_none_dispatch_root`` `"dispatch"` (pass-1), `"paired"` (`records/dispatch_paired_none/` with its own dispatch_state.json) or
    `"both"`; ``paired_none_planned=False`` leaves the paired `none` out of the ladder plans (dispatched outside the ladder matrix); ``paired_none_spec``
    overrides SPECS_P[NONE]; ``incomplete_attempt_root`` parks a `dispatch_rung8192.incomplete-D999` root beside the live ones (never an input)."""
    bind_seeds(seeds)
    root = root or str(tmp_path / "ifeval_seedrep")
    recs = os.path.join(root, M.RECORDS_DIR)
    hp = dict(pins_sha or {})
    mp = dict(mirrored_pins_sha or {})
    truths = {okey: truth(okey) for okey, *_ in ROWS}
    truth_p = truth(NONE, paired_none_spec or SPECS_P[NONE])
    rows = list(ROWS) + ([ROW_P] if paired_none else [])
    paired_none_key: Optional[str] = None
    paired_roots = {"dispatch": [M.DISPATCH_HEAD_BASE], "paired": [M.DISPATCH_HEAD_PAIRED_NONE], "both": [M.DISPATCH_HEAD_BASE, M.DISPATCH_HEAD_PAIRED_NONE]}[paired_none_dispatch_root]
    dispatch_state_p = {"schema": M.DISPATCH_STATE_SCHEMA, "sha": TREE, "run_id": f"{SHA7}-synthetic-paired", "cap": M.CAP_BASE, "done": {}, "failed": {}, "held": {}, "refused": {}, "queued": [], "active": {}}
    three: Dict[str, Any] = {"schema": M.THREE_NUMBERS_SCHEMA, "eval": M.EVAL_ID, "source": M.SOURCE, "written": NOW, "contract": "synthetic: value = acc over scored; bound = acc x n_scored / n; third state = n_at_cap over draws",
                             "seed_reading": M.SEED_AXIS_RULE, "rows": {}}
    launch: List[dict] = []
    matrix_rows: List[dict] = []
    keys: Dict[Tuple[str, int], str] = {}
    dispatch_states: Dict[int, dict] = {cap: {"schema": M.DISPATCH_STATE_SCHEMA, "sha": TREE, "run_id": f"{SHA7}-synthetic", "cap": cap, "done": {}, "failed": {}, "held": {}, "refused": {}, "queued": [], "active": {}} for cap in M.LADDER}
    plan_rows: List[Tuple[str, str, Dict[str, Any], str]] = []
    for i, (okey, paired, staged, sha, ihash8, chash) in enumerate(rows):
        tag = TAG_P if paired else TAG
        sel = SEL_P if paired else SEL
        steered = staged is not None
        is_paired_none = paired and okey == NONE
        tr = truth_p if is_paired_none else truths[okey]
        job = f"ife-qwen35_4b-{sel}-{okey[:12]}"
        launch.append(launch_row(sel, okey, tag, pin_key=staged, expected_sha=(mp.get(okey, sha) if staged else None), paired=paired, index=i))
        matrix_rows.append({"index": i, "job_id": job, "model_key": "qwen35_4b", "view": sel, "tag": tag, "cell": okey, "cell_slug": slug_of(okey), "staged_id": staged, "object_sha256": sha, "fitted": steered,
                            "thinking": "off", "samples_per_item": M.K_TRIALS, "cap": M.CAP_BASE, "caps": list(M.LADDER), "cap_ceiling": M.CAP_CEILING, "serving_path": "lora_native", "sampling_mode": "vendor_preset",
                            "out_root": M.rel_store(tag, slug_of(okey), M.CAP_BASE), "rung_out_roots": {str(c): M.rel_store(tag, slug_of(okey), c) for c in M.LADDER[1:]}, "status": "launched", "route_kind": ("lens" if steered else "base")})
        if not (is_paired_none and not paired_none_planned):
            plan_rows.append((tag, okey, tr, job))
        if (okey in not_run and not is_paired_none) or (is_paired_none and paired_none == "not_run"):
            continue
        prev_chash = None
        for cap in tr["chain"]:
            if (okey, cap) in drop_store:
                prev_chash = chash[cap]
                continue
            key = store_files(root, tag, okey, cap, tr, obj_sha=sha, ihash8=ihash8, chash8=chash[cap], prev_chash8=prev_chash, obj_sha_attested=(attest_sha or {}).get(okey),
                              replicate_block=replicate_block, result_complete=((okey, cap) not in result_incomplete), max_lora_rank=(1 if okey == ARDITI else 8))
            if is_paired_none:
                paired_none_key = key
            else:
                keys[(okey, cap)] = key
            if cap != M.CAP_BASE:
                subset_file(root, tag, okey, cap, tr, prev_chash)
            if (okey, cap) not in drop_dispatch:
                a_sha = (attest_sha or {}).get(okey, sha) if steered else None
                rec = dispatch_record(tag, okey, cap, store_key=key, obj_sha=a_sha)
                if (okey, cap) in rate_limited:
                    rec.update(state="failed", status=M.DISPATCH_RATE_LIMITED_STATUS, serving_path_attested=None, serving_path_attested_source=None,
                               error=f"SERVING_PATH_MISMATCH: requested 'lora_native'; the store at {rec['out_dir']} attests nothing readable — VolumeListFiles rate limit exceeded. Please wait and retry.")
                    rec["rec"]["status"] = M.DISPATCH_RATE_LIMITED_STATUS
                if dispatch_override:
                    dispatch_override(okey, cap, rec)
                bucket = "done" if rec["state"] == "done" else "failed"
                heads = paired_roots if (is_paired_none and cap == M.CAP_BASE) else [M.dispatch_head(cap)]
                for head in heads:
                    write(os.path.join(recs, head, M.DISPATCH_SUB, tag, slug_of(okey), M.dispatch_file(cap)), rec)
                    st_doc = dispatch_state_p if head == M.DISPATCH_HEAD_PAIRED_NONE else dispatch_states[cap]
                    st_doc[bucket][rec["configuration_id"]] = {"fc_id": rec["fc_id"], "name": rec["name"], "cell": okey}
            three["rows"][three_row(tag, okey, cap, tr, obj_sha=((attest_sha or {}).get(okey, sha) if steered else None), store_key=key, steered=steered, anchor_tr=(truths[NONE] if okey != NONE else None))["key"]] = \
                three_row(tag, okey, cap, tr, obj_sha=((attest_sha or {}).get(okey, sha) if steered else None), store_key=key, steered=steered, anchor_tr=(truths[NONE] if okey != NONE else None))
            prev_chash = chash[cap]
    for cap in M.LADDER:
        write(os.path.join(recs, M.dispatch_head(cap), M.DISPATCH_STATE_FILE), dispatch_states[cap])
    if paired_none and M.DISPATCH_HEAD_PAIRED_NONE in paired_roots:
        write(os.path.join(recs, M.DISPATCH_HEAD_PAIRED_NONE, M.DISPATCH_STATE_FILE), dispatch_state_p)
        write(os.path.join(recs, M.DISPATCH_HEAD_PAIRED_NONE, "spawned.jsonl"), json.dumps({"configuration_id": M.configuration_id(TAG_P, NONE, M.CAP_BASE), "cell": NONE}) + "\n")
    if incomplete_attempt_root:
        # a parked `*.incomplete-*` root (attempt history the mirror never deletes): its record would refuse the LoRA row if it were read
        write(os.path.join(recs, "dispatch_rung8192.incomplete-D999", M.DISPATCH_SUB, TAG, slug_of(LORA), M.dispatch_file(8192)),
              dict(dispatch_record(TAG, LORA, 8192, store_key=keys.get((LORA, 8192), "x"), obj_sha=OBJ_L, state="active", status=None, attested=None), rec=None))
        write(os.path.join(recs, "dispatch_rung8192.incomplete-D999", M.DISPATCH_STATE_FILE), {"schema": M.DISPATCH_STATE_SCHEMA, "active": {M.configuration_id(TAG, slug_of(LORA), 8192): {}}, "done": {}, "failed": {}})
    if failed_attempt_roots:
        # attempt-1 history of a failed dispatch, carried beside the fresh root: NEVER an input (its records read `active` / INCOMPLETE)
        write(os.path.join(recs, "dispatch_rung4096.failed-D205", M.DISPATCH_SUB, TAG, slug_of(LORA), M.dispatch_file(4096)),
              dict(dispatch_record(TAG, LORA, 4096, store_key=keys.get((LORA, 4096), "x"), obj_sha=OBJ_L, state="active", status=None, attested=None), rec=None))
        write(os.path.join(recs, "dispatch_rung4096.failed-D205", M.DISPATCH_STATE_FILE), {"schema": M.DISPATCH_STATE_SCHEMA, "active": {M.configuration_id(TAG, slug_of(LORA), 4096): {}}, "done": {}, "failed": {}})
    for cap in M.LADDER[1:]:
        plan_records(root, cap, plan_rows, override=plan_override)
    if three_override:
        three_override(three)
    three["n_rows"] = len(three["rows"])
    three["n_complete"] = len(three["rows"])
    write(os.path.join(recs, M.THREE_NUMBERS_FILE), three)
    launch_doc = {"schema": M.LAUNCH_SET_SCHEMA, "derived_at": NOW, "evals": {M.EVAL_ID: {"own_source": M.SOURCE, "launch": launch,
                                                                                            "canonical_decode": {"policy": "qwen35_4b_it|ifeval|off|vendor_preset@v1", "thinking": "off", **{k: DECODE[k] for k in ("temperature", "top_p", "top_k")},
                                                                                                                 "ladder": list(M.LADDER)}}}}
    if launch_layout == "carried":
        launch_doc["evals"][M.EVAL_ID].update(census_row_table(launch, SEL, SEL_P))
        launch_doc["row_table"] = list(launch)
    elif launch_layout != "launch":
        raise ValueError(launch_layout)
    if launch_override:
        launch_override(launch_doc)
    write(os.path.join(recs, M.LAUNCH_SET_FILE), launch_doc)
    write(os.path.join(recs, M.PINS_FILE), {"qwen35_4b": {STAGED: {"object_sha256": mp.get(RELU, OBJ)}, STAGED_L: {"object_sha256": mp.get(LORA, OBJ_L)}, STAGED_P: {"object_sha256": mp.get(COMPOSED, OBJ_P)},
                                                          STAGED_A: {"object_sha256": mp.get(ARDITI, OBJ_A), "kind": "vector_all", "staged_id": STAGED_A}}})
    if matrix:
        write(os.path.join(recs, M.MATRIX_FILE), {"schema": M.MATRIX_SCHEMA, "task": M.TASK, "n_items": M.N_ITEMS, "samples_per_item": M.K_TRIALS, "thinking": "off", "sampling_mode": "vendor_preset",
                                                  "caps": list(M.LADDER), "serving_path": "lora_native", "rows": matrix_rows})
    import yaml
    h_roster = {"targets": {"qwen35_4b": {"views": {SEL: {"config": f"configs/sdfbase_full22_{TAG}.yaml", "cells": {RELU: STAGED, LORA: STAGED_L, PROMPTED: None, "prompted__si_constitution": None}},
                                                    SEL_P: {"config": f"configs/sdfbase_full22_{TAG_P}.yaml", "paired_of": SEL, "cells": {COMPOSED: STAGED_P}}}}}}
    h_roster_path = write(str(tmp_path / "roster.yaml"), yaml.safe_dump(h_roster))
    h_pins_path = write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": hp.get(RELU, OBJ)}, STAGED_L: {"object_sha256": hp.get(LORA, OBJ_L)}, STAGED_P: {"object_sha256": hp.get(COMPOSED, OBJ_P)}}})
    a_roster = {"targets": {"qwen35_4b": {"views": {"ar_c0p3": {"c": 0.3, "dose_unit": "raw_coefficient", "phase": "both", "cells": {ARDITI: STAGED_A}}}}}}
    a_roster_path = write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump(a_roster))
    a_pins_path = write(str(tmp_path / "arditi_pins.json"), {"qwen35_4b": {STAGED_A: {"staged_id": STAGED_A, "object_sha256": hp.get(ARDITI, OBJ_A), "kind": "vector_all", "served_layer": 12, "c": 0.3,
                                                                                       "dose_unit": "raw_coefficient", "phase": "both", "positions": "all", "view": "ar_c0p3"}}})
    return {"root": root, "roster": h_roster_path, "pins": h_pins_path, "arditi_roster": a_roster_path, "arditi_pins": a_pins_path, "keys": keys, "truths": truths,
            "paired_none_key": paired_none_key, "paired_none_truth": truth_p}


def adapter(fx: dict, seeds: LC.Seeds) -> M.IfevalSeedrepAdapter:
    return M.IfevalSeedrepAdapter(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], arditi_roster_path=fx["arditi_roster"], arditi_pins_path=fx["arditi_pins"])


def primed_frame(ad: M.IfevalSeedrepAdapter):
    """The steady-state frame of an in-memory read (`_anchor_frame_fixtures`): since emission 3 every contrast of the reader is decided by the pass's
    anchor currency, and on an EMPTY frame every comparison parks (no provisional anchors: the first pass over a fresh registry files no contrast). The
    in-memory tests describe the pass AFTER the fold, so they read on a frame primed from the adapter's own emissions (every run current, complete,
    off-roster)."""
    import _anchor_frame_fixtures as AF
    frame = AF.Frame()
    AF.load_records(frame, AF.priming_read(ad, ctx(frame.con)))
    return frame


def primed_conn(ad: M.IfevalSeedrepAdapter, conn=None):
    """`conn` when the test gave one; else the connection of a frame primed from the adapter's own emissions (:func:`primed_frame`)."""
    return conn if conn is not None else primed_frame(ad).con


def read(ad: M.IfevalSeedrepAdapter, conn=None, group: str = TAG):
    """One group's read on `conn`, or on a frame primed from the adapter's own emissions (:func:`primed_frame`) when none is given; every record validated."""
    from sisteer.registry import validate as VAL
    inputs = ad.discover()
    groups: Dict[str, list] = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert sorted(groups) == [group], list(groups)
    recs = []
    for batch in ad.read(ctx(primed_conn(ad, conn)), group, groups[group]):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt: Dict[str, list] = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group)


def run_of(bt, okey: str) -> dict:
    hits = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == okey]
    assert len(hits) == 1, (okey, len(hits))
    return hits[0]


def metrics_of(bt, rid: str) -> Tuple[dict, dict, dict]:
    """(the value row, the bound row, the secondary loose row) of a run."""
    rows = [m for m in bt["run_metric"] if m["run_id"] == rid]
    value = [m for m in rows if m["metric_id"] == M.HEADLINE_METRIC and m["computation_id"] == M.merge_computation_id()]
    bound = [m for m in rows if m["metric_id"] == M.HEADLINE_METRIC and m["computation_id"] == ""]
    loose = [m for m in rows if m["metric_id"] == M.SECONDARY_METRIC]
    assert len(rows) == 3 and len(value) == 1 and len(bound) == 1 and len(loose) == 1, rows
    return value[0], bound[0], loose[0]


def refused(facts, tag: str, okey: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{tag}|{okey} -- ")]
    assert len(hits) == 1, facts["refusals"]
    return hits[0]
