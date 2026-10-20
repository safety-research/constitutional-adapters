"""The `seedrep_controls` legacy-store reader on a SYNTHETIC battery root shaped like the seed-1 served-controls store (invented shas,
`fc-SYNTH…` call ids, placeholder item ids, no generation text; the three keys `none`, one staged LoRA, one declared prompt).

What it pins:

* RUN IDENTITY: every run files under EXACTLY the id the premint queued -- `identity.run_id(seedrep_battery, seedrep_battery:<tree>:<axis>:s<N>,
  <the roster's staged object key>, <eval>, <the battery's minted instrument>)` with the eval's own axis (`or_bench` for `or_bench_hard`) --
  and the premint-shaped plan, the producing tree's plan behind `registry_runs.json` and the re-derivation agree; a plan whose producer key
  differs REFUSES that (eval, key) by name and names the field. One LIVE queued id (captured from the registry 2026-09-21) is re-derived from
  the battery's declared inputs by pure identity arithmetic, and bound to the declaration on disk when the battery root is on this host;
* the 1,319-vs-200 rule: OR-Bench's FULL list at the premint's 200-item instrument is by design -- `n_items_planned 200`, `n_items_generated
  1,319`, exactly 200 item rows `extra.in_subset` at the contract's positions (the stdlib mirror equals `contracts.or_bench_subset_positions`);
  a store SHORT of the full list is refused as incomplete;
* MMLU is rule-scored: `run_metric acc` recounted from the 200 item rows and equal to the producer's aggregate, `n_items_judged 200`, one
  `generated` state event that moves the queued row (the harvester's rule-scored completion then stamps `ingested`), the declared degeneration
  measures `not_applicable`;
* XSTest / OR-Bench file `generated`: one `generated` item row + pointer flags per item, the at-cap share measured from the store, ONE `pending`
  disposition per (key, eval), `escalation_pending` for a key with at-cap items and `final_at_base` for one without, cap 512 at index 0;
* the DURABLE carried copy is read before the volatile peek holding the same bytes, its files bind to the carry entry's evidence_inventory (a
  changed byte refuses the root), the decode pin is the certifying tree's re-derived sampler record, `code_sha` is the producing tree;
* a root whose `COMPLETE.json` is absent is REFUSED by name (no run, no state, no disposition);
* the HARVESTER folds the reader's records onto the queued rows: `queued` -> `generated` (XSTest / OR-Bench) and -> `ingested` (MMLU),
  campaign / view / source unchanged, the premint's `queued_utc` kept;
* the JUDGED path (the judge driver's files landed on the store keys): the judge resolved to the seeded row and bound by its rubric /
  protocol pins, the OR-Bench judgment RESTRICTED to the subset (n <= 200) with the full-1,319 reading on extra, abstentions counted as
  abstentions (never verdicts) in the judgment and the metrics, every id unchanged across the ungraded -> judged read, the digest agreeing (a
  disagreeing digest, a half-landed pass and a foreign rubric each refuse by name), and the harvester moving the judged rows to `ingested` with the
  pending dispositions withdrawn while MMLU is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid

import pytest

import _b1_fixtures as F
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import relu_paired as RP
from sisteer.registry.adapters import seedrep_cluster as SC
from sisteer.registry.adapters import seedrep_controls as SCT
from sisteer.registry.adapters import seedrep_statics as SS
from sisteer.registry.adapters.base import AdapterContext

SEED = 1
W7 = "abc1234"
TREE = W7 + "5" * 33
FROM_W7 = "def5678"
FROM_TREE = FROM_W7 + "6" * 33
UUID = "11111111-2222-3333-4444-555555555555"
FROM_UUID = "22222222-3333-4444-5555-666666666666"
ATTEMPT = "01SYNTHETICCONTROLS00000AA-0123456789ab"
KEY = f"{W7}|{UUID}|{SCT.STANDUP}|s{SEED}|base|0"
OWN_KEY = f"{FROM_W7}|{FROM_UUID}|{SCT.STANDUP}|s{SEED}|base|0"
STANDUP_ID = f"smoke-S5__{SCT.STANDUP}__s{SEED}__base__shard00"
LORA = "constv2-mix-full__lora_r4_ce"
PROMPTED = "prompted__safety_refusal"
STAGED = f"base64s{SEED}__{LORA}"
OBJ = hashlib.sha256(b"synthetic seedrep controls lora s1").hexdigest()
HARNESS_KEY = {LORA: "si_constv2_mix_full__lora_r4_ce", "none": "none", PROMPTED: "prompted__safety"}
IHASH = {LORA: "f5a30100", "none": "2ff9cdff", PROMPTED: "faa4c8cf"}
CHASH = {LORA: "1c374725", "none": "9c11adc3", PROMPTED: "2fb9695d"}
STORE_KEY = {k: f"{k.replace('__', '--')}__{SCT.STANDUP}__t0__a0__{IHASH[k]}__{CHASH[k]}" for k in (LORA, "none", PROMPTED)}
MODEL_ID = "qwen35_4b_it"
HF_MODEL, HF_REV = "Qwen/Qwen3.5-4B", "8" * 40
STORE = f"controls_cap512__s{SEED}"
SAMPLER = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
NOW = "2026-09-21T12:00:00.000Z"
COMPLETED = "2026-09-20T06:46:25Z"
#: at-cap items per (key, eval): the LoRA and the prompt hit the cap, `none` does not on XSTest (its `final_at_base` reading)
AT_CAP = {(LORA, "xstest"): 3, (LORA, "or_bench_hard"): 5, ("none", "xstest"): 0, ("none", "or_bench_hard"): 2, (PROMPTED, "xstest"): 1, (PROMPTED, "or_bench_hard"): 0}
MMLU_CORRECT = {LORA: 130, "none": 142, PROMPTED: 143}


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=F.producer(), conn=None)


def _write(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return path


def _h8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _instrument(seeds, eval_id: str, w7: str = W7) -> dict:
    """The battery's minted instrument for one controls eval: the seeded canonical row with the battery's changes, minted through
    identity.instrument_id (the harness_version names the tree, so two trees mint two ids -- the producing tree's premint is another set)."""
    seed_inst, seed_judge = SCT.SEEDS_BY_EVAL[eval_id]
    row = seeds.instrument_row(seed_inst, eval_id, model_id=MODEL_ID)
    fields = {name: row.get(name) for name, _t in I.INSTRUMENT_ID_FIELDS}
    fields.update(context_window_tokens=7424, harness_version=f"synthetic controls harness ({eval_id}) @ seedrep_battery W {w7}", n_trials=1)
    if eval_id in SCT.SAMPLED_EVALS:
        fields.update(token_cap_tokens=16384, token_cap_ladder_json=[512, 2048, 8192, 16384], read_at="ladder_final",
                      ladder_promotion_rule="synthetic controls ladder 512 -> 2,048 -> 8,192 -> 16,384; 1 % item-filtered trigger")
        if eval_id == "or_bench_hard":
            fields["item_set_id"] = "or_bench_hard_200"
    else:
        fields.update(decode_seed_rule="fixed", ladder_promotion_rule="n.a. (1-token letter-logprob prefill read)")
    iid = I.instrument_id(fields)
    return {"id": iid, "fields": fields, "seeded_id": str(row["instrument_id"]), "judge_id": fields.get("judge_id"),
            "declaration": {"fields": fields, "legs": [{"eval_id": eval_id, "leg": SCT.AXIS_BY_EVAL[eval_id]}],
                            "provenance": {"derived_from_instrument_id": str(row["instrument_id"]), "derived_from_seed_instrument": seed_inst, "instrument_id": iid,
                                           "judge_seed_id": seed_judge, "w7": w7}}}


def _plan_block(w7: str, eval_id: str, key: str, pokey: str, iid: str, object_id: str, judge_id) -> dict:
    """One premint plan block, shaped like the live `_stores/controls/<key>__s1.json[<task>]` (also the queued `run` payload of the fold test)."""
    kind = LC.cell_grammar_legacy(key)["object_kind"]
    sampled = eval_id in SCT.SAMPLED_EVALS
    rid = SCT.run_id_for(w7, eval_id, SEED, pokey, iid)
    blk = {"run_id": rid, "campaign": SCT.CAMPAIGN, "producer_run_key_legacy": SCT.producer_key(w7, SCT.AXIS_BY_EVAL[eval_id], SEED), "producer_object_key_legacy": pokey,
           "lifecycle_state": "queued", "model_id": MODEL_ID, "object_id": object_id, "transfer_kind": {"lora": "base_to_instruct", "none": "none"}.get(kind, "prompted_only"),
           "eval_id": eval_id, "item_set_id": {"xstest": "xstest_safe_250", "or_bench_hard": "or_bench_hard_200", "mmlu": "mmlu_200"}[eval_id], "instrument_id": iid,
           "instrument_match": "match", "harness": "bespoke_sisteer", "harness_version": f"synthetic controls harness ({eval_id}) @ seedrep_battery W {w7}",
           "scoring_rule": ("judge_graded_weights" if sampled else "letter_logprob_prefill"), "prompt_template_sha256": "unverified", "judge_id": judge_id,
           "decode_seed_rule": ("per_item" if sampled else "fixed"), "decode_source": ("model_card" if sampled else "greedy"),
           "decode_policy_id": ("qwen35_4b_it|over_refusal|off|vendor_preset@v1" if sampled else "qwen35_4b_it|mmlu|off|greedy@v1"),
           "token_cap_tokens": (512 if sampled else 1), "token_cap_rung_index": 0, "token_cap_ladder_json": ([512, 2048, 8192, 16384] if sampled else [1]),
           "reply_cap_tokens": (None if sampled else 1), "context_window_tokens": 7424, "thinking_mode": "off", "serving_path": "lora_native",
           "applied_positions_mode": ("all" if kind == "lora" else "none"), "multi_turn_steering_scope": ("single_turn" if kind == "lora" else "not_applicable"),
           "code_sha": (TREE if w7 == W7 else FROM_TREE), "substrate": "modal", "substrate_source": "recorded",
           "n_items_planned": {"xstest": 450, "or_bench_hard": 200, "mmlu": 200}[eval_id], "seed_base": SEED, "trial_index": 0, "view_legacy": f"s{SEED}",
           "lane_source_legacy": SCT.SOURCE, "campaign_cell_id_legacy": key, "extra": {"w7": w7, "view": f"s{SEED}", "served_key": key, "store_task": SCT.TASK_BY_EVAL[eval_id]},
           "field_confidence_json": {"prompt_template_sha256": LC.unverified("the served path renders the chat template server-side (synthetic premint)")}}
    if kind == "lora":
        blk.update(strength_nominal=1.0, strength_nominal_source="filed", strength_unit="nominal_only", is_operating_point=1)
    return blk


def _events(key: str, n_at_cap: dict, n_items: dict) -> str:
    """The store's per-item draw ledger: one row per (control, item), the at-cap items first, no text."""
    lines, seq = [], 0
    for eval_id in SCT.SAMPLED_EVALS:
        control, prefix = SCT.CONTROL_BY_EVAL[eval_id], ("xst" if eval_id == "xstest" else "orb")
        for i in range(n_items[eval_id]):
            seq += 1
            at_cap = i < n_at_cap[eval_id]
            lines.append(json.dumps({"v": 1, "kind": "draw", "seq": seq, "ts_utc": f"2026-09-20T06:{(seq // 60) % 60:02d}:{seq % 60:02d}.102035Z", "cell_id": key, "control": control,
                                     "cap": 512, "escalated_from_cap": None, "item_id": f"{prefix}{i:04d}-{_h8(f'{prefix}{i}')}", "sampling_seed": 1000 + i,
                                     "finish_reason": ("length" if at_cap else "stop"), "at_cap": at_cap, "completion_tokens": (512 if at_cap else 100 + i % 300), "prompt_tokens": 20,
                                     "text_sha256": hashlib.sha256(f"{key}:{control}:{i}".encode()).hexdigest(), "request_id": None, "completion_id": f"chatcmpl-{seq:016x}", "error": None}))
    return "\n".join(lines) + "\n"


def _mmlu_rows(key: str, n_correct: int) -> str:
    """200 letter-read acc rows + the producer's aggregate, the shape relu_paired.read_mmlu_items / seedrep_cluster._mmlu_letter_reads read."""
    lines = []
    base = 1789884421.0
    for i in range(200):
        v = 1.0 if i < n_correct else 0.0
        lines.append(json.dumps({"arm_name": key.replace("__", "--"), "attempt": 0, "coeff": 1.0, "created_at": base + i * 0.05, "domain": "knowledge_s2parity", "engine": "vllm",
                                 "env_digest": "5cebc145d1f882a5", "harness_config_hash": CHASH[key] + "75a1b112", "intervention_hash": IHASH[key] + "2e209076",
                                 "item_id": f"m{i:04d}-{_h8(f'm{i}')}", "kind": "cell", "metric": "acc", "model_id": HF_MODEL, "model_revision": HF_REV, "position_order": i,
                                 "run_id": f"served-controls-{key}", "seed": SEED, "sweep": "two_si_controls_served", "task": RP.MMLU_TASK, "task_version": "s2parity_v1",
                                 "tier": 1, "trial_id": 0, "value": v,
                                 "extra_json": json.dumps({"censored_letters": [], "gold": "C", "letter_read": "all_present", "n_censored": 0, "pick": ("C" if v else "D"), "top_logprobs": 64})}))
    lines.append(json.dumps({"arm_name": key.replace("__", "--"), "created_at": base + 200 * 0.05, "item_id": "__aggregate__", "metric": "acc_aggregate", "task": RP.MMLU_TASK,
                             "value": n_correct / 200, "extra_json": json.dumps({"n": 200, "response_prefill": False, "top_logprobs": 64, "censored_items": 0})}))
    return "\n".join(lines) + "\n"


def _manifest(key: str, tree: str) -> dict:
    steered = key == LORA
    return {"cell_id": key, "harness_cell_id": HARNESS_KEY[key], "arm_name": key.replace("__", "--"), "attempt": 0, "controls": ["xstest", "or_bench_hard_1k", "mmlu"], "engine": "vllm",
            "env_digest": "5cebc145d1f882a5", "harness_config_hash": CHASH[key] + "75a1b112", "intervention_hash": IHASH[key] + "2e209076", "kind": "cell", "model_id": HF_MODEL,
            "model_revision": HF_REV, "producer": {"driver": "served_capability_run", "tree_sha": tree}, "serving_engine": "lora_native",
            "serving_identity": (SC.STEERED_IDENTITY if steered else SC.PASSTHROUGH_IDENTITY), "serving_path": ("lora" if steered else "passthrough"), "steer_positions": "all",
            "fit_provenance": ({"model_id": "Qwen/Qwen3.5-4B-Base", "object_sha": OBJ[:16], "positions_at_fit": "all", "served_model_id": HF_MODEL, "served_revision": HF_REV} if steered else {}),
            "resolved_config": {"cell_id": key, "gen": dict(SAMPLER, do_sample=True, max_new_tokens=512, arm="vendor_recommended", seed_base=SEED, trial=0, batch_size=8),
                                "mmlu": {"answer_seed": "Answer:", "response_prefill": False, "top_logprobs": 64}, "model_id": HF_MODEL, "model_revision": HF_REV, "steer_positions": "all"},
            "routing_table": "/workspace/attempt/exp/serve/routing_table.json", "trial_id": 0, "sweep": "two_si_controls_served"}


def _gen_stats(key: str, n_at_cap: dict, n_items: dict, n_correct: int) -> dict:
    out = {"mmlu": {"acc": n_correct / 200, "n": 200, "top_logprobs": 64, "response_prefill": False, "censored_items": 0, "censored_letter_counts": {}, "sampled_decode_applies": False,
                    "sampled_decode_note": "teacher-forced letter-logprob argmax: one greedy token at temperature 0"}}
    for eval_id in SCT.SAMPLED_EVALS:
        control, prefix = SCT.CONTROL_BY_EVAL[eval_id], ("xst" if eval_id == "xstest" else "orb")
        n, a = n_items[eval_id], n_at_cap[eval_id]
        out[control] = {"control": control, "n_new": n, "n_total": n, "finish_reasons": {"length": a, "stop": n - a}, "cap": 512,
                        "decode": dict(SAMPLER, do_sample=True, cap=512, arm="vendor_recommended", seed_base=SEED, trial=0), "n_at_cap": a,
                        "at_cap_item_ids": [f"{prefix}{i:04d}-{_h8(f'{prefix}{i}')}" for i in range(a)], "spool": f"/workspace/attempt/exp/sampled/{STORE}/cells/{STORE_KEY[key]}/private/gens_{control}.jsonl"}
    return out


def _root(tmp_path, seeds, *, carried: bool = True, n_items: dict = None, at_cap: dict = None) -> dict:
    """A synthetic battery root: accepted.json, carry_over.json (a carried root by default, as the live store is), the plans of the certifying
    tree, the producing tree's plans under prior/, the three instrument declarations, one accepted base store with three keys, the certifying
    tree's re-derived sampler record."""
    n_items = dict({"xstest": 450, "or_bench_hard": 1319}, **(n_items or {}))
    at_cap = dict(AT_CAP, **(at_cap or {}))
    rs = str(tmp_path / "seedrep_battery")
    insts = {e: _instrument(seeds, e) for e in SCT.EVALS}
    from_insts = {e: _instrument(seeds, e, w7=FROM_W7) for e in SCT.EVALS}
    keys = {LORA: (STAGED, OBJ), "none": ("none", "none"), PROMPTED: (PROMPTED, seeds.si_object_id("safety_refusal"))}
    rids = {(k, e): SCT.run_id_for(W7, e, SEED, pokey, insts[e]["id"]) for k, (pokey, _o) in keys.items() for e in SCT.EVALS}
    from_rids = {(k, e): SCT.run_id_for(FROM_W7, e, SEED, pokey, from_insts[e]["id"]) for k, (pokey, _o) in keys.items() for e in SCT.EVALS}
    _write(os.path.join(rs, SCT.ACCEPTED_FILE), {KEY: ATTEMPT})
    _write(os.path.join(rs, *SCT.PLANS_REL, SCT.PLANS_INDEX), {"campaign": SCT.CAMPAIGN, "w7": W7, "instruments": {insts[e]["id"]: {"derived_from_seed_instrument": SCT.SEEDS_BY_EVAL[e][0],
                                                                                                                          "legs": [{"eval_id": e, "leg": SCT.AXIS_BY_EVAL[e]}]} for e in SCT.EVALS}})
    for e in SCT.EVALS:
        _write(os.path.join(rs, *SCT.PLANS_REL, SCT.PLANS_INSTRUMENTS_SUB, f"{insts[e]['id']}.json"), insts[e]["declaration"])
    prior = os.path.join(rs, "prior", FROM_W7)
    for k, (pokey, oid) in keys.items():
        _write(os.path.join(rs, *SCT.PLANS_REL, SCT.PLANS_STORES_SUB, SCT.STANDUP, f"{k}__s{SEED}.json"),
               {SCT.TASK_BY_EVAL[e]: _plan_block(W7, e, k, pokey, insts[e]["id"], oid, insts[e]["judge_id"]) for e in SCT.EVALS})
        if carried:
            _write(os.path.join(prior, SCT.PRIOR_PLANS_SUB, SCT.PLANS_STORES_SUB, SCT.STANDUP, f"{k}__s{SEED}.json"),
                   {SCT.TASK_BY_EVAL[e]: _plan_block(FROM_W7, e, k, pokey, from_insts[e]["id"], oid, from_insts[e]["judge_id"]) for e in SCT.EVALS})
    if carried:
        root = os.path.join(rs, SS.CARRIED_SUB, FROM_W7, SS.STORES_SUB, f"S5_{SCT.STANDUP}", ATTEMPT, ATTEMPT)
        own_key, tree, map_rids = OWN_KEY, FROM_TREE, from_rids
    else:
        root = os.path.join(rs, SS.STORES_SUB, W7, SCT.STANDUP, ATTEMPT)
        own_key, tree, map_rids = KEY, TREE, rids
    n_children = 3
    _write(os.path.join(root, "COMPLETE.json"), {"S": 70, "attempt_id": ATTEMPT, "children": n_children, "completed_utc": COMPLETED, "control_key": own_key, "elapsed_s": 1.0})
    _write(os.path.join(root, "RESULT"), f"{COMPLETED} status=DONE rc=0 attempt={ATTEMPT} standup={STANDUP_ID} children={n_children} failed=0 S=70 elapsed_s=1.0\n")
    _write(os.path.join(root, "ATTEMPT.json"), {"W": tree, "accepted": True, "accepted_key": own_key, "attempt_id": ATTEMPT, "call_id": "fc-SYNTH0000000000000002",
                                                "launch_plan_row": STANDUP_ID, "launch_plan_sha": "0" * 16, "schema": "seedrep_attempt/1"})
    _write(os.path.join(root, SCT.EVIDENCE_FILE), {"attempt_id": ATTEMPT, "hardware": "modal", "attested_max_num_seqs": 70, "readiness": {"attested_max_model_len": 7424, "attested_max_num_seqs": 70},
                                                  "children": [{"arm": k, "eval_leg": SCT.STANDUP, "rc": 0, "s": SEED} for k in keys], "modal": {"app_name": f"seedrep-served-{tree[:7]}", "gpu": "H200"},
                                                  "determinism": {c: {"absent": True, "rc": 2} for c in ("xstest", "or_bench_hard_1k", "mmlu")}, "schema": "seedrep_served/modal_evidence/1"})
    streams, entries = [], []
    for k in keys:
        n_rows = sum(n_items.values())
        n_cap = sum(at_cap[(k, e)] for e in SCT.SAMPLED_EVALS)
        streams.append({"at_cap": n_cap, "bad_lines": 0, "enqueued": n_cap, "rows_read": n_rows, "path": f"/workspace/attempt/exp/sampled/{STORE}/cells/{STORE_KEY[k]}/{SCT.EVENTS_FILE}"})
        for e in SCT.SAMPLED_EVALS:
            control, prefix = SCT.CONTROL_BY_EVAL[e], ("xst" if e == "xstest" else "orb")
            for i in range(at_cap[(k, e)]):
                iid = f"{prefix}{i:04d}-{_h8(f'{prefix}{i}')}"
                entries.append({"arm_s": f"{k}__s{SEED}", "item_key": f"{SCT.STANDUP}__{control}__{iid}__0", "path": f"synthetic/ladder/{SCT.STANDUP}/s{SEED}/2048/queue/{k}__s{SEED}/{iid}/{ATTEMPT}.json",
                                "sha256": "0" * 64, "slot": f"{k}__s{SEED}/{SCT.STANDUP}__{control}__{iid}__0"})
    _write(os.path.join(root, "ladder_enqueued.json"), {"accepted_key": own_key, "attempt_id": ATTEMPT, "complete": True, "control_key": own_key.replace("|", "/"),
                                                        "ladder": {"cap_next": 2048, "feeder_shards": ["0"], "kind": SCT.LADDER_KIND, "producer_of": "2048", "shard": "0", "shards": ["0"]},
                                                        "n": len(entries), "entries": entries, "problems": [], "streams": streams, "schema": "seedrep_ladder_enqueued/1", "written_utc": COMPLETED})
    n_sampled = 3 * sum(n_items.values())
    tasks = {SCT.TASK_BY_EVAL[e]: {"caps": ["512"], "n": n_sampled, "verdict": "AGREES", "distinct_resolved": [{"sampler": SAMPLER, "n": n_sampled}],
                                   "distinct_max_tokens": [{"max_tokens": "512", "n": n_sampled}]} for e in SCT.SAMPLED_EVALS}
    tasks[RP.MMLU_TASK] = {"caps": ["1"], "n": 600, "verdict": "AGREES", "distinct_resolved": [{"sampler": GREEDY, "n": 600}], "distinct_max_tokens": [{"max_tokens": "1", "n": 600}]}
    sampler_rec = {"verdict": "AGREES", "classes": {"client": {"n": n_sampled + 600, "verdict": "AGREES", "tasks": tasks,
                                                                "distinct_resolved": [{"sampler": SAMPLER, "n": n_sampled}, {"sampler": GREEDY, "n": 600}]}},
                   "client_task_verdicts": {t: "AGREES" for t in tasks}}
    _write(os.path.join(root, "resolved_sampler.json"), (dict(sampler_rec, verdict="UNCOMPARED") if carried else sampler_rec))
    _write(os.path.join(root, *SCT.SERVE_REL, "serving_config.json"), {"model": HF_MODEL, "model_revision": HF_REV, "serving_config_sha256": "3" * 64, "model_key": "qwen35_4b"})
    _write(os.path.join(root, *SCT.SERVE_REL, "cells_served.json"), {"object_artifact_sha256_by_cell": {HARNESS_KEY[LORA]: OBJ}, "attested_max_num_seqs": 70})
    _write(os.path.join(root, *SCT.SERVE_REL, "cells.json"), {"cells": [{"cell_id": HARNESS_KEY[k], "intervention_hash": IHASH[k] + "2e209076"} for k in keys]})
    _write(os.path.join(root, *SCT.SERVE_REL, "registry_runs.json"), {HARNESS_KEY[k]: [{"campaign": SCT.CAMPAIGN, "expected": {"serving_path": "lora_native"}, "run_id": map_rids[(k, e)]} for e in SCT.EVALS] for k in keys})
    for k in keys:
        d = os.path.join(root, *SCT.STORES_REL, STORE, SCT.KEYS_SUB, STORE_KEY[k])
        _write(os.path.join(d, "manifest.json"), _manifest(k, tree))
        _write(os.path.join(d, "gen_stats.json"), _gen_stats(k, {e: at_cap[(k, e)] for e in SCT.SAMPLED_EVALS}, n_items, MMLU_CORRECT[k]))
        _write(os.path.join(d, SCT.EVENTS_FILE), _events(k, {e: at_cap[(k, e)] for e in SCT.SAMPLED_EVALS}, n_items))
        _write(os.path.join(d, "rows.jsonl"), _mmlu_rows(k, MMLU_CORRECT[k]))
        _write(os.path.join(d, "evidence.json"), {"cell_id": k, "serving_identity": _manifest(k, tree)["serving_identity"], "serve_header": {"template_kwargs": {"enable_thinking": False}}})
        _write(os.path.join(d, "cap_ladder.json"), {"cap": 512, "ceiling": 16384, "ladder": [512, 1024, 2048, 4096, 8192, 16384], "controls": {
            SCT.CONTROL_BY_EVAL[e]: {"exit": "escalate", "escalate": True, "next_cap": 1024, "n": n_items[e], "n_at_cap": at_cap[(k, e)], "n_truncated": at_cap[(k, e)]} for e in SCT.SAMPLED_EVALS}})
    fx = {"rs": rs, "root": root, "insts": insts, "from_insts": from_insts, "rids": rids, "from_rids": from_rids, "keys": keys, "tree": tree}
    if carried:
        ev = os.path.join(rs, "carry_over", "reaudit", f"{SCT.STANDUP}__s{SEED}__base__shard00", ATTEMPT, "evidence")
        p = _write(os.path.join(ev, "resolved_sampler.json"), sampler_rec)
        inv = {f"/vol/seedrep/{FROM_W7}/exp/seedrep_battery_modal_smoke/S5/{SCT.STANDUP}/s{SEED}/shard00/attempts/{ATTEMPT}/{fn}": F.sha_file(os.path.join(root, fn))
               for fn in SCT.INVENTORY_BOUND_FILES if os.path.isfile(os.path.join(root, fn))}
        inv[f"/vol/seedrep/{W7}/{UUID}/carried/{SCT.STANDUP}__s{SEED}__base__shard00__{ATTEMPT}/evidence/resolved_sampler.json"] = F.sha_file(p)
        _write(os.path.join(rs, SCT.CARRY_OVER_FILE), {"W": TREE, "W7": W7, "carried": {f"{SCT.STANDUP}__s{SEED}__base__shard00": {
            "accepted_key": KEY, "attempt_id": ATTEMPT, "reaudited_by": TREE,
            "carried_from": {"W": FROM_TREE, "W7": FROM_W7, "accepted_key": OWN_KEY, "attempt_id": ATTEMPT, "standup_id": STANDUP_ID, "source": "s5_smoke", "rs": prior},
            "criterion3_waiver": {"at": "2026-09-20T12:57:26Z", "by": "D180 (synthetic)", "decision": "ADMIT the absence of the producer-time determinism spot-check artefact"},
            "evidence_inventory": inv, "reaudit": {"local_evidence_root": ev, "evidence_root": f"/vol/seedrep/{W7}/{UUID}/carried/{SCT.STANDUP}__s{SEED}__base__shard00__{ATTEMPT}/evidence",
                                                   "reaudited_by": TREE}}}})
        fx["reaudit_dir"] = ev
    return fx


RUBRIC = {"xstest": "288e8c08b0c7bbfa", "or_bench_hard_1k": "15f29334703c31a8"}
PROTOCOL = "b8d28441cf71caec"
HARNESS_SHA = "8" * 39 + "a"
#: label cycle per control (the judge's labels over the generated items, in item order) and the abstentions the pass records
LABEL_CYCLE = ("full_refusal", "full_refusal", "partial_refusal", "full_compliance")
#: (key, control) -> {item id: abstention reason}; the LoRA's OR-Bench pass abstains on one SUBSET item (a refusal) and one item outside it (an
#: error), `none` on one XSTest UNSAFE item, so the subset arithmetic and the all-items arithmetic differ
ABSTAIN = {(LORA, "or_bench_hard_1k"): {"orb0002-" + _h8("orb2"): "refusal", "orb0000-" + _h8("orb0"): "error"}, ("none", "xstest"): {"xst0300-" + _h8("xst300"): "refusal"}}


def _judge(fx: dict, *, abstain: dict = None, rubric_override: dict = None, drop: Sequence[str] = ()) -> None:
    """The judge driver's files ADDED to every store key of the fixture root (the shape its pass lands in): judge_items.jsonl (one verdict or
    abstention per generated item; XSTest items carry their safe / unsafe subset), judge_summary.json with the driver's provenance and per-control
    counters, and the label rows + aggregates appended to rows.jsonl. `drop` removes named driver files after writing (a half-landed pass)."""
    abstain = ABSTAIN if abstain is None else abstain
    rubric = dict(RUBRIC, **(rubric_override or {}))
    positions = set(SCT.or_bench_subset_positions())
    for k in fx["keys"]:
        d = os.path.join(fx["root"], *SCT.STORES_REL, STORE, SCT.KEYS_SUB, STORE_KEY[k])
        events = [json.loads(l) for l in open(os.path.join(d, SCT.EVENTS_FILE), encoding="utf-8") if l.strip()]
        judge_rows, label_lines, summary = [], [], {}
        ts = 1790001000.0
        for eval_id in SCT.SAMPLED_EVALS:
            control, task = SCT.CONTROL_BY_EVAL[eval_id], SCT.TASK_BY_EVAL[eval_id]
            metric = SCT.LABEL_METRIC[control]
            items = [e for e in events if e["control"] == control]
            abst = abstain.get((k, control), {})
            n_judged = n_abst = 0
            full_refusal_scored = judged_scored = 0
            for i, e in enumerate(items):
                iid = e["item_id"]
                ts += 0.5
                jr = {"cache_org": "lowprio", "cached_before_run": False, "control": control, "item_id": iid, "judge_model": "claude-sonnet-5", "protocol_sha": PROTOCOL,
                      "request_key": hashlib.sha256(f"{k}|{iid}".encode()).hexdigest(), "rubric_sha": rubric[control], "task": task, "verdict_created_at": ts}
                if eval_id == "xstest":
                    subset = "safe" if i < 250 else "unsafe"
                    jr.update(subset=subset, type="homonyms")
                    scored = subset == "safe"
                else:
                    jr["category"] = "deception"
                    scored = SCT.or_bench_position(iid) in positions
                if iid in abst:
                    jr.update(status="abstained", abstention_reason=abst[iid]); n_abst += 1
                else:
                    lab = LABEL_CYCLE[i % len(LABEL_CYCLE)]
                    jr.update(status="judged", label=lab, label_source="store"); n_judged += 1
                    if scored:
                        judged_scored += 1
                        full_refusal_scored += lab == "full_refusal"
                    ex = {"label": lab, "rubric_sha": rubric[control], **({"subset": subset, "type": "homonyms"} if eval_id == "xstest" else {"category": "deception"})}
                    label_lines.append(json.dumps({"arm_name": k.replace("__", "--"), "created_at": ts, "item_id": iid, "metric": metric, "task": task, "task_version": "judged_v1",
                                                   "value": {"full_refusal": 0.0, "partial_refusal": 1.0, "full_compliance": 2.0}[lab], "extra_json": json.dumps(ex), "position_order": i}))
                    if eval_id == "or_bench_hard":
                        label_lines.append(json.dumps({"arm_name": k.replace("__", "--"), "created_at": ts, "item_id": iid, "metric": SCT.ORB_REFUSED_METRIC, "task": task,
                                                       "task_version": "judged_v1", "value": (1.0 if lab == "full_refusal" else 0.0), "extra_json": None, "position_order": i}))
                judge_rows.append(jr)
            agg = "xst_safe_refused_bounds_aggregate" if eval_id == "xstest" else "orb_refused_bounds_aggregate"
            label_lines.append(json.dumps({"arm_name": k.replace("__", "--"), "created_at": ts + 1, "item_id": "__aggregate__", "metric": agg, "task": task, "task_version": "judged_v1",
                                           "value": (full_refusal_scored / judged_scored if judged_scored else None), "position_order": None,
                                           "extra_json": json.dumps({"n": len(items), "n_graded": n_judged, "n_abstained": n_abst, "rubric_sha": rubric[control]})}))
            summary[control] = {"judged_at": ("2026-09-21T14:30:50Z" if eval_id == "xstest" else "2026-09-21T14:36:32Z"), "status": "ok", "n": len(items), "task": task,
                                "n_cached_before_run": 0, "wall_clock_s": 100.0, "filed_by_harness": True, "n_store_cache_disagree": 0,
                                "harness_result": {"n": len(items), "n_graded": n_judged, "n_abstained": n_abst, "rubric_sha": rubric[control],
                                                   "abstained_by_reason": {r: sum(1 for v in abst.values() if v == r) for r in set(abst.values())},
                                                   "judge": {"api_calls": len(items), "backend": "anthropic", "cache_hits": 0, "calls_by_org": {"highprio": 0, "lowprio": len(items)}, "transport_refusals": 0}},
                                "judge_counters_delta": {"abstained": n_abst, "api_calls": len(items), "gate_live_ceiling": 100, "gate_rate_limited_total": 0}}
        with open(os.path.join(d, "rows.jsonl"), "a", encoding="utf-8") as fh:
            fh.write("\n".join(label_lines) + "\n")
        _write(os.path.join(d, "judge_items.jsonl"), "\n".join(json.dumps(r) for r in judge_rows) + "\n")
        _write(os.path.join(d, "judge_summary.json"), {
            "cell": STORE_KEY[k], "cell_id": k, "controls": summary, "model_id": HF_MODEL, "model_revision": HF_REV, "started_at": "2026-09-21T14:29:03Z", "finished_at": "2026-09-21T14:36:32Z",
            "target": STORE, "harness_cell_id": HARNESS_KEY[k],
            "provenance": {"harness_sha": HARNESS_SHA, "harness_branch": "main", "harness_dirty_paths": [], "driver": "served_capability_judge_v1", "driver_path": "scripts/served_capability_judge.py",
                           "judge_model": "claude-sonnet-5", "judge_org": "lowprio", "judge_backend": "anthropic",
                           "judge_cache_note": "delegated judge pass over a peer store (cap 512 base): seeded private copy of the shared cache, merged back after the run; judged by the orxs_seedrep lane (synthetic)",
                           "protocol_sha": {c: PROTOCOL for c in SCT.CONTROL_BY_EVAL.values() if c != "mmlu"},
                           "rubrics": {c: {"name": ("xstest_cpi" if c == "xstest" else "or_bench_cpi"), "sha": rubric[c], "version": "v1"} for c in SCT.CONTROL_BY_EVAL.values() if c != "mmlu"},
                           "sdk_max_retries": 2, "anthropic_sdk": "0.120.2", "concurrency_total": 100, "cell_parallel": 6}})
        for fn in drop:
            os.remove(os.path.join(d, fn))


def _expected_subset(fx: dict, key: str, abstain: dict = None) -> dict:
    """The OR-Bench subset arithmetic the fixture's judge files imply for one key: label counts, abstentions, n_judged, the full-only rate."""
    abstain = ABSTAIN if abstain is None else abstain
    positions = set(SCT.or_bench_subset_positions())
    d = os.path.join(fx["root"], *SCT.STORES_REL, STORE, SCT.KEYS_SUB, STORE_KEY[key])
    items = [json.loads(l) for l in open(os.path.join(d, SCT.EVENTS_FILE), encoding="utf-8") if l.strip() and json.loads(l)["control"] == "or_bench_hard_1k"]
    abst = abstain.get((key, "or_bench_hard_1k"), {})
    out = {"full_refusal": 0, "partial_refusal": 0, "full_compliance": 0, "n_abstained": 0, "n_judged": 0, "n_subset_present": 0}
    for i, e in enumerate(items):
        if SCT.or_bench_position(e["item_id"]) not in positions:
            continue
        out["n_subset_present"] += 1
        if e["item_id"] in abst:
            out["n_abstained"] += 1
        else:
            out[LABEL_CYCLE[i % len(LABEL_CYCLE)]] += 1
            out["n_judged"] += 1
    out["rate"] = out["full_refusal"] / out["n_judged"]
    return out


def _digest(tmp_path, fx: dict, *, override: dict = None) -> dict:
    """The digest of the subset readings for the fixture's keys at cap 512 (+ its positions file), as `orxs_seedrep` writes them."""
    rows = {f"s{SEED}|{k}|{SCT.DIGEST_ITEM_SET}|512": _expected_subset(fx, k) for k in fx["keys"]}
    for kk, patch in (override or {}).items():
        rows[f"s{SEED}|{kk}|{SCT.DIGEST_ITEM_SET}|512"].update(patch)
    return {"digest": _write(str(tmp_path / "results_or_bench_200_subset.json"), {"declaration": "synthetic", "key": "seed|cell|item_set|cap", "positions_file": "or_bench_hard_200_positions.json",
                                                                                   "rate": "full_refusal / judged over the 200-item subset (abstentions excluded)", "rows": rows}),
            "positions": _write(str(tmp_path / "or_bench_hard_200_positions.json"), {"item_set_id": SCT.DIGEST_ITEM_SET, "rule": "or_bench_subset_positions(n=200, seed=20260906, of=1319)",
                                                                                    "positions": list(SCT.or_bench_subset_positions()), "item_id_form": "orb<pos:04d>-<hash>"})}


def _roster(tmp_path) -> dict:
    import yaml
    roster = {"targets": {"qwen35_4b": {"views": {f"s{SEED}": {"cells": {LORA: STAGED, PROMPTED: None}}}}}}
    return {"roster": _write(str(tmp_path / "roster.yaml"), yaml.safe_dump(roster)), "pins": _write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": OBJ}}})}


def _adapter(fx, rp, seeds, volatile_root=None, digest=None) -> SCT.SeedrepControlsAdapter:
    """The reader over the fixture root; the digest is the fixture's own when given, else none (never the host's live one)."""
    return SCT.SeedrepControlsAdapter(root=fx["rs"], volatile_root=volatile_root, seeds=seeds, roster_path=rp["roster"], pins_path=rp["pins"],
                                      digest_path=(digest["digest"] if digest else None), digest_positions_path=(digest["positions"] if digest else None))


def _read(ad: SCT.SeedrepControlsAdapter, group: str = f"{SCT.STANDUP}__s{SEED}"):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert list(groups) == [group], list(groups)
    recs = []
    for batch in ad.read(_ctx(), group, groups[group]):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group)


def _by_key_eval(runs):
    return {(r["campaign_cell_id_legacy"], r["eval_id"]): r for r in runs}


# --------------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------------

def test_run_ids_are_the_premints_and_equal_identity_run_id_over_the_re_derived_inputs(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert len(bt["run"]) == 9 and facts["instrument_ids"] == {e: fx["insts"][e]["id"] for e in SCT.EVALS}
    for r in bt["run"]:
        key, eval_id = r["campaign_cell_id_legacy"], r["eval_id"]
        pokey, oid = fx["keys"][key]
        iid = fx["insts"][eval_id]["id"]
        assert r["run_id"] == fx["rids"][(key, eval_id)] == I.run_id(SCT.CAMPAIGN, SCT.producer_key(W7, SCT.AXIS_BY_EVAL[eval_id], SEED), pokey, eval_id, iid) \
            == SCT.run_id_for(W7, eval_id, SEED, pokey, iid)
        assert (r["campaign"], r["producer_run_key_legacy"], r["producer_object_key_legacy"], r["instrument_id"], r["object_id"]) == \
               (SCT.CAMPAIGN, f"{SCT.CAMPAIGN}:{W7}:{SCT.AXIS_BY_EVAL[eval_id]}:s{SEED}", pokey, iid, oid)
        assert r["view_legacy"] == f"s{SEED}" and r["lane_source_legacy"] == SCT.SOURCE == SS.SOURCE == "seedrep_battery" and r["model_id"] == MODEL_ID
        assert r["lifecycle_state"] == "generated" and r["generated_utc"].startswith("2026-09-20T06:46:25") and r["code_sha"] == FROM_TREE
        assert r["extra"]["spool_landing"] == SCT.SPOOL_LANDING and r["extra"]["certifying_tree_sha"] == TREE and r["extra"]["w7"] == W7 and r["extra"]["carried_from"]["W7"] == FROM_W7
        assert r["extra"]["identity_plan"] == "matches" and r["extra"]["registry_runs_map"].startswith("matches the producing tree")
        assert r["extra"]["producing_tree_run_id"] == fx["from_rids"][(key, eval_id)] != r["run_id"]
        assert r["substrate"] == "modal" and r["substrate_source"] == "recorded" and r["producer_session"] == "Reseed" and r["trial_index"] == 0 and r["seed_base"] == SEED
        assert r["served_max_num_seqs"] == 70 and r["served_max_num_seqs_source"] == "attested" and r["context_window_tokens"] == 7424 and r["thinking_mode"] == "off"
        assert r["model_revision"] == HF_REV and r["serving_config_sha256"] == "3" * 64 and r["modal_call_id"] == "fc-SYNTH0000000000000002"
        assert r["store_completion_sha256"] and r["store_uri"].endswith(STORE_KEY[key])
    by = _by_key_eval(bt["run"])
    lora = by[(LORA, "or_bench_hard")]
    assert lora["object_id"] == OBJ and lora["transfer_kind"] == "base_to_instruct" and lora["serving_path"] == "lora_native" and lora["applied_positions_mode"] == "all"
    assert lora["strength_nominal"] == 1.0 and lora["is_operating_point"] == 1 and lora["serving_identity"] == SC.STEERED_IDENTITY
    none = by[("none", "xstest")]
    assert none["object_id"] == "none" and none["transfer_kind"] == "none" and none["serving_path"] == "passthrough" and none["serving_identity"] == SC.PASSTHROUGH_IDENTITY
    prompted = by[(PROMPTED, "mmlu")]
    assert prompted["object_id"] == seeds.si_object_id("safety_refusal") and prompted["transfer_kind"] == "prompted_only"
    assert "judgment" not in bt and "ruling" not in bt and all(r["superseded_by_run_id"] is None for r in bt["run"])


#: ONE live queued row of the registry (2026-09-21): constv2-mix-full__lora_r4_ce x or_bench_hard under the certifying tree e526edf. The
#: producer key comes from the battery's accepted.json (tree e526edf, axis or_bench), the staged key from the roster's s1 selection, the
#: instrument id from the minted declaration `registry/plans/_instruments/3881d895….json` -- pure identity arithmetic reproduces the id
LIVE_W7, LIVE_INSTRUMENT = "e526edf", "3881d895b4796c9b2dbb18f2689feaa1acd1a3fcbafa21d35ad359e4d3cf37e1"
LIVE_RUN_ID = "seedrep_battery:or_bench_hard:3881d895b479:5fcf5972edb77220"
LIVE_MMLU_INSTRUMENT, LIVE_MMLU_RUN_ID = "9d3fc308d63760e0c793743d69d616a0e5ebc020baa35e0ae4e0b0bb2c7678e5", "seedrep_battery:mmlu:9d3fc308d637:d810342262a1e17f"
LIVE_XSTEST_INSTRUMENT, LIVE_XSTEST_RUN_ID = "27625722c20728490d7b7b37e09f130de47162fac701d34f4e6f69c2484b0e05", "seedrep_battery:xstest:27625722c207:5687a3a2f1df7743"


def test_a_live_queued_id_re_derives_from_the_batterys_declared_inputs():
    """The premint's own queued ids (read off the live registry 2026-09-21) are exactly what the reader's identity arithmetic mints from the
    accepted key's tree, the eval's axis, the roster's staged key and the minted instrument -- the fold lands on those rows and no other."""
    assert SCT.run_id_for(LIVE_W7, "or_bench_hard", 1, "base64s1__constv2-mix-full__lora_r4_ce", LIVE_INSTRUMENT) == LIVE_RUN_ID
    assert SCT.run_id_for(LIVE_W7, "mmlu", 1, "base64s1__constv2-mix-full__lora_r4_ce", LIVE_MMLU_INSTRUMENT) == LIVE_MMLU_RUN_ID
    assert SCT.run_id_for(LIVE_W7, "xstest", 1, "none", LIVE_XSTEST_INSTRUMENT) == LIVE_XSTEST_RUN_ID
    assert SCT.producer_key(LIVE_W7, SCT.AXIS_BY_EVAL["or_bench_hard"], 1) == "seedrep_battery:e526edf:or_bench:s1"
    # the declaration on disk, when the battery root is on this host, re-hashes to the id the constant names (never trusted by name alone)
    p = os.path.join(SCT.RS_DEFAULT, *SCT.PLANS_REL, SCT.PLANS_INSTRUMENTS_SUB, f"{LIVE_INSTRUMENT}.json")
    if not os.path.isfile(p):
        pytest.skip("the battery's run-state root is not on this host")
    with open(p, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    assert I.instrument_id(d["fields"]) == LIVE_INSTRUMENT and d["fields"]["eval_id"] == "or_bench_hard" and d["fields"]["item_set_id"] == "or_bench_hard_200"
    assert d["fields"]["token_cap_ladder_json"] == [512, 2048, 8192, 16384]


def test_the_minted_instrument_is_verified_not_trusted(tmp_path, seeds):
    """A declaration whose fields do not re-hash to its id refuses that EVAL by name; the other two evals still file."""
    fx = _root(tmp_path, seeds)
    p = os.path.join(fx["rs"], *SCT.PLANS_REL, SCT.PLANS_INSTRUMENTS_SUB, f"{fx['insts']['xstest']['id']}.json")
    d = json.load(open(p, encoding="utf-8"))
    d["fields"]["context_window_tokens"] = 11520
    _write(p, d)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert {r["eval_id"] for r in bt["run"]} == {"or_bench_hard", "mmlu"} and len(bt["run"]) == 6
    assert facts["n_refusals"] == 1 and "re-hashes to" in facts["refusals"][0] and facts["refusals"][0].startswith(f"{SCT.STANDUP}__s{SEED}|xstest"), facts["refusals"]


def test_a_plan_whose_producer_key_differs_refuses_that_eval_and_key_and_names_the_field(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    p = os.path.join(fx["rs"], *SCT.PLANS_REL, SCT.PLANS_STORES_SUB, SCT.STANDUP, f"none__s{SEED}.json")
    d = json.load(open(p, encoding="utf-8"))
    d[SCT.TASK_BY_EVAL["xstest"]]["producer_run_key_legacy"] = f"{SCT.CAMPAIGN}:0000000:xstest:s{SEED}"
    _write(p, d)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert len(bt["run"]) == 8 and ("none", "xstest") not in _by_key_eval(bt["run"]) and ("none", "or_bench_hard") in _by_key_eval(bt["run"])
    assert facts["n_refusals"] == 1 and "identity REFUSED" in facts["refusals"][0] and "producer_run_key_legacy: plan" in facts["refusals"][0], facts["refusals"]


def test_a_served_object_that_is_not_the_pinned_one_is_refused(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    p = os.path.join(fx["root"], *SCT.SERVE_REL, "cells_served.json")
    d = json.load(open(p, encoding="utf-8"))
    d["object_artifact_sha256_by_cell"][HARNESS_KEY[LORA]] = "9" * 64
    _write(p, d)
    ad = _adapter(fx, _roster(tmp_path), seeds)
    # the carried binding does not pin cells_served.json in this fixture's inventory beyond its sha, so re-pin it to the changed bytes
    co = json.load(open(os.path.join(fx["rs"], SCT.CARRY_OVER_FILE), encoding="utf-8"))
    for k in list(co["carried"][f"{SCT.STANDUP}__s{SEED}__base__shard00"]["evidence_inventory"]):
        if k.endswith("/exp/serve/cells_served.json"):
            co["carried"][f"{SCT.STANDUP}__s{SEED}__base__shard00"]["evidence_inventory"][k] = F.sha_file(p)
    _write(os.path.join(fx["rs"], SCT.CARRY_OVER_FILE), co)
    bt, facts = _read(ad)
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {"none", PROMPTED} and len(bt["run"]) == 6
    assert facts["n_refusals"] == 1 and "the served object is not the pinned one" in facts["refusals"][0], facts["refusals"]


# --------------------------------------------------------------------------------------------
# the 1,319-vs-200 rule, MMLU rule-scored, the sampled evals generated
# --------------------------------------------------------------------------------------------

def test_the_subset_mirror_equals_the_contract():
    from sisteer.capeval.tinycap import contracts as C
    assert SCT.or_bench_subset_positions() == C.or_bench_subset_positions() and len(SCT.or_bench_subset_positions()) == 200
    assert (SCT.OR_BENCH_SUBSET_N, SCT.OR_BENCH_SUBSET_SEED, SCT.OR_BENCH_FULL_N) == (C.OR_BENCH_SUBSET_N, C.OR_BENCH_SUBSET_SEED, C.OR_BENCH_FULL_N)
    assert SCT.or_bench_position("orb0035-b791e972") == 35 and SCT.or_bench_position("xst0035-b791e972") is None


def test_or_bench_files_the_full_list_at_the_subset_instrument_and_marks_the_subset(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]
    for key in fx["keys"]:
        r = _by_key_eval(bt["run"])[(key, "or_bench_hard")]
        assert r["n_items_planned"] == 200 and r["n_items_generated"] == 1319 and r["n_items_judged"] is None and r["item_set_id"] == "or_bench_hard_200"
        assert r["extra"]["n_items_generated_full"] == 1319 and r["extra"]["n_items_in_subset"] == 200 and "by design" in r["extra"]["item_set_note"]
        assert r["extra"]["subset"] == {"n": 200, "seed": 20260906, "of": 1319, "rule": r["extra"]["subset"]["rule"]}
        items = [i for i in bt["item_result"] if i["run_id"] == r["run_id"]]
        assert len(items) == 1319 and all(i["metric_id"] == SCT.GENERATED_METRIC and i["value"] is None and i["label"] is None and i["judgment_id"] == "" for i in items)
        marked = sorted(i["extra"]["subset_position"] for i in items if i["extra"]["in_subset"])
        assert marked == sorted(SCT.or_bench_subset_positions()) and len(marked) == 200
        assert all(i["extra"]["subset_position"] == int(i["item_id"][3:7]) for i in items)
    # a store SHORT of the full list is an incomplete generation, refused by name; the other evals of the key still file
    fx2 = _root(tmp_path / "short", seeds, n_items={"or_bench_hard": 1318})
    bt2, facts2 = _read(_adapter(fx2, _roster(tmp_path / "short"), seeds))
    assert {r["eval_id"] for r in bt2["run"]} == {"xstest", "mmlu"} and facts2["n_refusals"] == 3
    assert all("1318 or_bench_hard_1k items" in w and "incomplete store" in w for w in facts2["refusals"]), facts2["refusals"]


def test_mmlu_is_rule_scored_moves_the_queued_row_and_reproduces_the_producers_aggregate(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]
    mmlu_ids = {r["run_id"] for r in bt["run"] if r["eval_id"] == "mmlu"}
    mmlu_states = [s for s in bt["run_state"] if s["run_id"] in mmlu_ids]
    assert len(mmlu_ids) == 3 and len(mmlu_states) == 3 and all(s["state"] == "generated" and "rule-scored" in s["note"] for s in mmlu_states)
    for key in fx["keys"]:
        r = _by_key_eval(bt["run"])[(key, "mmlu")]
        assert (r["n_items_planned"], r["n_items_generated"], r["n_items_judged"]) == (200, 200, 200) and r["judge_id"] is None and r["token_cap_tokens"] == 1
        assert r["token_cap_ladder_json"] == [1] and r["reply_cap_tokens"] == 1 and r["pass_label"] is None and r["instrument_match"] == "match"
        assert r["decode_policy_id"] == "qwen35_4b_it|mmlu|off|greedy@v1" and r["decode_source"] == "greedy" and r["decode_sent_json"] == {"temperature": 0.0}
        assert r["extra"]["generation_state"].startswith("generated; rule-scored") and "escalation_state" not in r["extra"]
        m = [x for x in bt["run_metric"] if x["run_id"] == r["run_id"]]
        assert len(m) == 1 and m[0]["metric_id"] == "acc" and m[0]["judgment_id"] == "" and m[0]["k"] == MMLU_CORRECT[key] and m[0]["n"] == 200
        assert abs(m[0]["value"] - MMLU_CORRECT[key] / 200) < 1e-12 and m[0]["ci_method"] == "wilson_95"
        items = [i for i in bt["item_result"] if i["run_id"] == r["run_id"]]
        assert len(items) == 200 and sum(i["value"] for i in items) == MMLU_CORRECT[key] and all(i["metric_id"] == "acc" and i["label"] in ("C", "D") for i in items)
        deg = [d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]]
        assert len(deg) == len(IC.DECLARED_MEASURES) and all(d["measurement_state"] == "not_applicable" for d in deg)
        assert facts["by_eval"]["mmlu"][key]["acc"] == MMLU_CORRECT[key] / 200
    assert not any(d["eval_id"] == "mmlu" for d in bt["disposition"])
    # the recount is checked against the producer's aggregate: a store whose gen_stats disagrees is refused, nothing filed for it
    fx2 = _root(tmp_path / "bad", seeds)
    p = os.path.join(fx2["root"], *SCT.STORES_REL, STORE, SCT.KEYS_SUB, STORE_KEY["none"], "gen_stats.json")
    d = json.load(open(p, encoding="utf-8"))
    d["mmlu"]["acc"] = 0.5
    _write(p, d)
    bt2, facts2 = _read(_adapter(fx2, _roster(tmp_path / "bad"), seeds))
    assert ("none", "mmlu") not in _by_key_eval(bt2["run"]) and len(bt2["run"]) == 8 and any("does not reproduce the producer's aggregate" in w for w in facts2["refusals"])


def test_xstest_and_or_bench_file_generated_items_pending_dispositions_and_the_at_cap_measure(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    by = _by_key_eval(bt["run"])
    assert set(SCT.ESCALATION_STATES) <= set(RP.ESCALATION_STATES)
    for (key, eval_id), n_cap in AT_CAP.items():
        r = by[(key, eval_id)]
        n = 450 if eval_id == "xstest" else 1319
        assert r["lifecycle_state"] == "generated" and r["token_cap_tokens"] == 512 and r["pass_label"] == "base_cap" and r["token_cap_rung_index"] == 0
        assert r["token_cap_ladder_json"] == [512, 2048, 8192, 16384] and r["decode_sent_json"] == SAMPLER and r["decode_policy_id"] == "qwen35_4b_it|over_refusal|off|vendor_preset@v1"
        assert r["instrument_match"] == "match" and r["judge_id"] == seeds.judge_id(SCT.SEEDS_BY_EVAL[eval_id][1]) and r["n_items_generated"] == n and r["n_items_judged"] is None
        assert r["extra"]["escalation_state"] == (SCT.STATE_FINAL_AT_BASE if n_cap == 0 else SCT.STATE_PENDING)
        esc = r["extra"]["escalation"]
        assert esc["n_at_cap"] == n_cap == esc["n_enqueued_to_next_cap"] and esc["next_cap"] == 2048 and esc["base_cap"] == 512
        assert (esc["next_cap_state"].startswith("not_needed") and esc["note"] is None) if n_cap == 0 else (esc["next_cap_state"] == "pending" and "D216" in esc["note"])
        assert esc["producer_claim"]["next_cap"] == 1024 and esc["producer_claim"]["file"] == "cap_ladder.json"
        assert abs(esc["at_cap_share"] - n_cap / n) < 1e-12 and r["extra"]["finish_reasons"] == {"length": n_cap, "stop": n - n_cap}
        items = [i for i in bt["item_result"] if i["run_id"] == r["run_id"]]
        flags = [f for f in bt["item_flags"] if f["run_id"] == r["run_id"]]
        assert len(items) == len(flags) == n and sum(f["at_token_cap"] for f in flags) == n_cap and all(f["generation_artifact_id"] and f["generation_offset"] >= 1 for f in flags)
        assert sum(1 for i in items if i["extra"]["at_cap"]) == n_cap and all(i["created_utc"].startswith("2026-09-20T06:") for i in items)
        deg = [d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]]
        at_cap = next(d for d in deg if d["measure"] == IC.AT_CAP_FINISH_REASON)
        assert at_cap["measurement_state"] == "measured" and (at_cap["numerator"], at_cap["denominator"], at_cap["denominator_unit"]) == (n_cap, n, "items")
        assert all(d["measurement_state"] == "not_measured" for d in deg if d["measure"] != IC.AT_CAP_FINISH_REASON)
        assert all("D216" in d["notes"] for d in deg if d["measure"] not in (IC.AT_CAP_FINISH_REASON, IC.AT_CAP_RETOKENIZED))
        assert "no re-tokenized cap detection" in next(d for d in deg if d["measure"] == IC.AT_CAP_RETOKENIZED)["notes"]
        states = [s for s in bt["run_state"] if s["run_id"] == r["run_id"]]
        assert len(states) == 1 and states[0]["state"] == "generated" and states[0]["evidence_uri"].endswith("COMPLETE.json")
    assert len(bt["disposition"]) == 6 and all(d["state"] == "pending" and d["reason"] == SCT.PENDING_REASON and d["eval_id"] in SCT.SAMPLED_EVALS for d in bt["disposition"])
    assert {(d["extra"]["campaign_cell_id_legacy"], d["eval_id"]): d["extra"]["escalation_state"]["512"] for d in bt["disposition"]} == \
           {k: (SCT.STATE_FINAL_AT_BASE if v == 0 else SCT.STATE_PENDING) for k, v in AT_CAP.items()}
    # never a final ladder verdict: no run above the base cap, no ruling, nothing superseded
    assert {r["token_cap_tokens"] for r in bt["run"] if r["eval_id"] in SCT.SAMPLED_EVALS} == {512} and "ruling" not in bt
    xst = by[("none", "xstest")]
    assert xst["n_items_planned"] == 450 and xst["item_set_id"] == "xstest_safe_250" and xst["extra"]["n_items_scored_planned"] == 250


# --------------------------------------------------------------------------------------------
# roots: the durable copy first, the carried binding, the refusal without the completion marker
# --------------------------------------------------------------------------------------------

def test_the_durable_carried_copy_is_read_before_the_volatile_peek_and_binds_to_the_carry_entry(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    peek = str(tmp_path / "peek")
    shutil.copytree(fx["root"], os.path.join(peek, FROM_W7, SS.STORES_SUB, f"S5_{SCT.STANDUP}", ATTEMPT, ATTEMPT))
    ad = _adapter(fx, _roster(tmp_path), seeds, volatile_root=peek)
    bt, facts = _read(ad)
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert facts["root"].endswith(os.path.join(SS.CARRIED_SUB, FROM_W7, SS.STORES_SUB, f"S5_{SCT.STANDUP}", ATTEMPT, ATTEMPT)) and len(facts["shadows"]) == 1 and "peek" in facts["shadows"][0]
    assert facts["decode_pin"].startswith("the certifying tree's re-derived sampler record") and "UNCOMPARED" in facts["decode_pin"]
    assert facts["determinism"]["criterion"] == SCT.DETERMINISM_WAIVED and facts["determinism"]["artefacts_absent"] == ["mmlu", "or_bench_hard_1k", "xstest"]
    assert len(bt["run"]) == 9 and all(r["code_sha"] == FROM_TREE and r["extra"]["certifying_tree_sha"] == TREE and r["extra"]["decode_pin"] == facts["decode_pin"] for r in bt["run"])
    # the durable copy's bytes are bound to the carry entry: a byte changed in the copy refuses the root
    _write(os.path.join(fx["root"], "RESULT"), open(os.path.join(fx["root"], "RESULT"), encoding="utf-8").read().replace("elapsed_s=1.0", "elapsed_s=2.0"))
    bt2, facts2 = _read(_adapter(fx, _roster(tmp_path), seeds, volatile_root=peek))
    assert bt2 == {} and any("evidence_inventory pins" in w for w in facts2["refusals"]), facts2["refusals"]


def test_a_carried_root_without_the_re_derived_sampler_record_is_refused(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    shutil.rmtree(fx["reaudit_dir"])
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert bt == {} and any("re-derived sampler record" in w and "not on this host" in w for w in facts["refusals"]), facts["refusals"]


def test_a_root_landed_under_the_certifying_tree_reads_without_a_carry_entry(tmp_path, seeds):
    fx = _root(tmp_path, seeds, carried=False)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert facts["root"].endswith(os.path.join(SS.STORES_SUB, W7, SCT.STANDUP, ATTEMPT)) and facts["decode_pin"] == "the root's own resolved_sampler.json"
    assert len(bt["run"]) == 9 and all(r["code_sha"] == TREE and r["extra"]["carried_from"] is None and r["extra"]["producing_tree_run_id"] is None
                                       and r["extra"]["registry_runs_map"] == "matches" for r in bt["run"])
    assert facts["determinism"]["criterion"] == "absent"


def test_a_root_without_complete_json_is_refused_by_name(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    os.remove(os.path.join(fx["root"], "COMPLETE.json"))
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert bt == {} and facts["n_refusals"] == 1 and "COMPLETE.json is absent" in facts["refusals"][0], facts["refusals"]


# --------------------------------------------------------------------------------------------
# the harvester folds the read onto the queued rows
# --------------------------------------------------------------------------------------------

def _registry_root(tmp_path, seeds, objects) -> str:
    """A registry root holding the REAL seeds' dimension rows as the registry's own loader mints them plus a stand-in object row for the
    synthetic seed-1 object (the statics reader's fold test does the same)."""
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


def _queued_records(fx, seeds, now: str) -> list:
    """The premint's queued rows (run + run_state per (key, eval)), premint-shaped, plus the battery's three minted instrument rows."""
    recs = []
    for e in SCT.EVALS:
        inst = fx["insts"][e]
        fc_inst = {k: LC.unverified("synthetic premint: the value is unverified") for k, v in inst["fields"].items() if v == "unverified"}
        recs.append(R.make_record("instrument", dict(inst["fields"], instrument_id=inst["id"], notes="synthetic minted instrument", field_confidence_json=fc_inst,
                                                     extra={"seed_instrument_id": SCT.SEEDS_BY_EVAL[e][0]}), F.producer(), str(uuid.uuid4()), now))
    for key, (pokey, oid) in fx["keys"].items():
        for e in SCT.EVALS:
            blk = _plan_block(W7, e, key, pokey, fx["insts"][e]["id"], oid, fx["insts"][e]["judge_id"])
            blk["queued_utc"] = now
            recs.append(R.make_record("run", blk, F.producer(), str(uuid.uuid4()), now))
            recs.append(R.make_record("run_state", {"run_id": blk["run_id"], "state": "queued", "event_utc": now, "evidence_kind": "submit_wrapper", "evidence_uri": "",
                                                    "scheduler_state": "SPAWNED"}, F.producer(), str(uuid.uuid4()), now))
    return recs


def test_the_harvester_folds_the_read_onto_the_queued_runs_and_stamps_the_rule_scored_mmlu_ingested(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    rp = _roster(tmp_path)
    ad = _adapter(fx, rp, seeds)
    obj, _staged = ad.resolve_object(LORA, f"s{SEED}")
    root = _registry_root(tmp_path, seeds, [obj])
    queued_utc = "2026-09-21T06:31:43.507Z"
    R.write_records(os.path.join(root, "records", "standins", "000002-premint.jsonl"), _queued_records(fx, seeds, queued_utc))
    rep0 = H.ingest_once(root, workers=0, min_free=0)
    assert rep0.refused == 0, rep0.refusals
    before = F.q(root, "SELECT run_id, lifecycle_state FROM run WHERE campaign = ?", (SCT.CAMPAIGN,))
    assert len(before) == 9 and {b[1] for b in before} == {"queued"} and F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    after = F.q(root, "SELECT run_id, eval_id, lifecycle_state, campaign, view_legacy, lane_source_legacy, queued_utc, generated_utc, code_sha, n_items_planned, n_items_generated, "
                      "n_items_judged, json_extract(extra_json, '$.spool_landing'), json_extract(extra_json, '$.w7'), json_extract(extra_json, '$.escalation_state'), "
                      "json_extract(extra_json, '$.producing_tree_run_id') FROM run WHERE campaign = ? ORDER BY 1", (SCT.CAMPAIGN,))
    assert {a[0] for a in after} == set(fx["rids"].values())
    for a in after:
        rid, eval_id, state = a[0], a[1], a[2]
        assert a[3] == SCT.CAMPAIGN and a[4] == f"s{SEED}" and a[5] == SCT.SOURCE and a[6] == queued_utc, "the premint's queued stamp is event-owned and stands"
        assert a[7] and a[7].startswith("2026-09-20T06:46:25") and a[8] == FROM_TREE and a[12] == SCT.SPOOL_LANDING and a[13] == W7
        key = next(k for (k, e), r in fx["rids"].items() if r == rid)
        assert a[15] == fx["from_rids"][(key, eval_id)]
        if eval_id == "mmlu":
            assert state == "ingested" and (a[9], a[10], a[11]) == (200, 200, 200) and a[14] is None
        else:
            assert state == "generated" and a[11] is None and a[14] in SCT.ESCALATION_STATES
            assert (a[9], a[10]) == ((200, 1319) if eval_id == "or_bench_hard" else (450, 450))
    assert F.q(root, "SELECT COUNT(*) FROM disposition WHERE state = 'pending'") == [(6,)]
    assert F.q(root, "SELECT COUNT(*) FROM judgment") == [(0,)]
    assert F.q(root, "SELECT eval_id, COUNT(*) FROM item_result i JOIN run r USING (run_id) WHERE r.campaign = ? GROUP BY eval_id ORDER BY 1", (SCT.CAMPAIGN,)) == \
           [("mmlu", 600), ("or_bench_hard", 3 * 1319), ("xstest", 3 * 450)]
    assert F.q(root, "SELECT COUNT(*) FROM item_result WHERE json_extract(extra_json, '$.in_subset') = 1") == [(600,)]
    mm = F.q(root, "SELECT r.campaign_cell_id_legacy, m.value, m.n FROM run_metric m JOIN run r USING (run_id) WHERE r.eval_id = 'mmlu' AND r.campaign = ? ORDER BY 1", (SCT.CAMPAIGN,))
    assert {k: (round(v, 6), n) for k, v, n in mm} == {k: (round(c / 200, 6), 200) for k, c in MMLU_CORRECT.items()}
    # a second pass over unchanged inputs republishes nothing and leaves the rows where they are
    rep2, _ing3, _ing4 = F.run_and_ingest(root, _adapter(fx, rp, seeds), now="2026-09-21T13:00:00.000Z")
    assert rep2.groups_refreshed == 0
    assert F.q(root, "SELECT eval_id, lifecycle_state FROM run WHERE campaign = ? GROUP BY 1, 2 ORDER BY 1", (SCT.CAMPAIGN,)) == \
           [("mmlu", "ingested"), ("or_bench_hard", "generated"), ("xstest", "generated")]


# --------------------------------------------------------------------------------------------
# the judged path (the judge driver's pass landed on the store keys)
# --------------------------------------------------------------------------------------------

def test_the_judged_read_keeps_every_id_restricts_or_bench_to_the_subset_and_carries_the_full_reading(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    rp = _roster(tmp_path)
    bt0, facts0 = _read(_adapter(fx, rp, seeds))
    ids0 = {(r["campaign_cell_id_legacy"], r["eval_id"]): r["run_id"] for r in bt0["run"]}
    assert facts0["n_refusals"] == 0 and len(bt0["disposition"]) == 6 and "judgment" not in bt0
    _judge(fx)
    dg = _digest(tmp_path, fx)
    bt, facts = _read(_adapter(fx, rp, seeds, digest=dg))
    assert facts["n_refusals"] == 0, facts["refusals"]
    # every id unchanged: nothing in the identity depends on the judge
    assert {(r["campaign_cell_id_legacy"], r["eval_id"]): r["run_id"] for r in bt["run"]} == ids0
    assert all(r["lifecycle_state"] == "generated" for r in bt["run"]), "a run record carries the state its store proves; `judged` moves through the event"
    by = _by_key_eval(bt["run"])
    judged_ids = {r["run_id"] for r in bt["run"] if r["eval_id"] in SCT.SAMPLED_EVALS}
    states = {}
    for st in bt["run_state"]:
        states.setdefault(st["run_id"], set()).add(st["state"])
    assert all(states[rid] == {"generated", "judged"} for rid in judged_ids) and all(states[r["run_id"]] == {"generated"} for r in bt["run"] if r["eval_id"] == "mmlu")
    assert "disposition" not in bt and len(bt["judgment"]) == 6 and facts["judged_runs"] == 6 and facts["digest"]["state"] == "present"
    jm = {j["run_id"]: j for j in bt["judgment"]}
    for key in fx["keys"]:
        # OR-Bench: the judgment is the SUBSET's (n <= 200), the full list rides extra, the digest agrees
        r = by[(key, "or_bench_hard")]
        exp = _expected_subset(fx, key)
        j = jm[r["run_id"]]
        assert j["judge_id"] == r["judge_id"] == seeds.judge_id("or_bench_cpi_v1_sonnet5") and j["judgment_id"] == I.judgment_id(r["run_id"], j["judge_id"])
        assert j["n_items_judged"] == r["n_items_judged"] == exp["n_judged"] <= 200 and j["n_judge_refusals"] + j["n_errors"] == exp["n_abstained"]
        assert j["status"] == "complete" and j["api_pool"] == "lowprio" and j["grading_code_sha"] == HARNESS_SHA and j["producer_session"] == "OR"
        assert j["extra"]["judge_seed_id"] == "or_bench_cpi_v1_sonnet5" and j["extra"]["rubric_sha16"] == RUBRIC["or_bench_hard_1k"] and "orxs_seedrep" in j["extra"]["judge_cache_note"]
        assert r["extra"]["generation_state"] == "judged" and r["extra"]["ungraded_reason"] is None and r["n_items_planned"] == 200 and r["n_items_generated"] == 1319
        full = r["extra"]["or_bench_full_1319"]
        n_abst_all = len(ABSTAIN.get((key, "or_bench_hard_1k"), {}))
        assert full["n_items"] == 1319 and full["n_judged"] == 1319 - n_abst_all and full["abstentions"] == n_abst_all and 0 <= full["rate"] <= 1 and "SECONDARY" in full["note"]
        assert j["extra"]["or_bench_full_1319"]["n_judged"] == full["n_judged"] and r["extra"]["digest_cross_check"]["state"] == "agrees"
        ms = {m["metric_id"]: m for m in bt["run_metric"] if m["run_id"] == r["run_id"]}
        assert ms["orb_refused"]["k"] == exp["full_refusal"] and ms["orb_refused"]["n"] == exp["n_judged"] and ms["orb_refused"]["n_abstentions"] == exp["n_abstained"]
        assert abs(ms["orb_refused"]["value"] - exp["rate"]) < 1e-12 and ms["orb_refused"]["judgment_id"] == j["judgment_id"] and ms["orb_refused"]["computed_by"] == "registry"
        assert ms["over_refusal_graded"]["n"] == exp["n_judged"] and 0 <= ms["over_refusal_graded"]["value"] <= 1 and ms["over_refusal_graded"]["extra"]["weights"] == SCT.GRADED_WEIGHTS
        assert ms["orb_refused_bounds"]["computed_by"] == "producer" and "FULL 1,319" in ms["orb_refused_bounds"]["extra"]["item_set"]
        items = [i for i in bt["item_result"] if i["run_id"] == r["run_id"]]
        labels = [i for i in items if i["metric_id"] == "orb_label"]
        assert len(labels) == 1319 and sum(1 for i in labels if i["extra"]["in_subset"]) == 200 and all(i["judgment_id"] == j["judgment_id"] for i in items)
        assert sum(1 for i in labels if i["extra"].get("status") == "abstained") == n_abst_all and all(i["label"] in SCT.LABELS for i in labels if i["extra"].get("status") != "abstained")
        assert sum(1 for i in items if i["metric_id"] == "orb_refused") == 1319 - n_abst_all and not any(i["metric_id"] == SCT.GENERATED_METRIC for i in items)
        flags = [f for f in bt["item_flags"] if f["run_id"] == r["run_id"]]
        assert len(flags) == 1319 and all(f["rationale_artifact_id"] and f["rationale_offset"] >= 1 and f["judgment_id"] == j["judgment_id"] for f in flags)
        # XSTest: the 250 safe prompts scored, every one of the 450 items keeps its label row
        x = by[(key, "xstest")]
        jx = jm[x["run_id"]]
        n_abst_x = len(ABSTAIN.get((key, "xstest"), {}))
        assert jx["judge_id"] == seeds.judge_id("xstest_cpi_v1_sonnet5") and jx["n_items_judged"] == x["n_items_judged"] == 250 and jx["n_judge_refusals"] + jx["n_errors"] == 0
        assert x["extra"]["scored_subset"]["n_judge_refusals_all_subsets"] + x["extra"]["scored_subset"]["n_errors_all_subsets"] == n_abst_x
        xitems = [i for i in bt["item_result"] if i["run_id"] == x["run_id"]]
        assert len(xitems) == 450 and all(i["metric_id"] == "xst_label" for i in xitems) and sum(1 for i in xitems if i["extra"].get("subset") == "safe") == 250
        mx = {m["metric_id"]: m for m in bt["run_metric"] if m["run_id"] == x["run_id"]}
        assert mx["xst_refused"]["n"] == 250 and mx["xst_refused"]["n_abstentions"] == 0 and "xst_safe_refused_bounds" in mx
        # MMLU unchanged by the pass
        m = by[(key, "mmlu")]
        assert m["n_items_judged"] == 200 and m["judge_id"] is None and any(x["metric_id"] == "acc" for x in bt["run_metric"] if x["run_id"] == m["run_id"])
    # the ONE steered key (the LoRA; a declared prompt is not steered) pairs against the same-standup `none` on the three headline metrics (the
    # label mean among them), per sampled eval
    cs = bt["run_contrast"]
    assert len(cs) == 1 * 2 * 3 and all(c["run_id"] == by[(LORA, c["run_id"].split(":")[1])]["run_id"] for c in cs)
    assert all(c["anchor_run_id"] == by[("none", c["run_id"].split(":")[1])]["run_id"] and c["anchor_kind"] == "same_standup_none" for c in cs)
    assert {c["metric_id"] for c in cs} == {"xst_refused", "orb_refused", "over_refusal_graded", "over_refusal_label_mean"} and all(c["n_pairs"] > 0 for c in cs)


def test_abstention_arithmetic_counts_abstentions_as_abstentions_never_verdicts(tmp_path, seeds):
    """The LoRA's OR-Bench pass abstains on one subset item (a refusal) and one item outside it (an error): the judgment counts 199 judged + 1
    refusal + 0 errors over the subset, the full reading 1,317 judged + 2 abstentions, the headline's denominator drops the abstention, and the
    abstained items file with no label and their reason."""
    fx = _root(tmp_path, seeds)
    _judge(fx)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds, digest=_digest(tmp_path, fx)))
    assert facts["n_refusals"] == 0, facts["refusals"]
    r = _by_key_eval(bt["run"])[(LORA, "or_bench_hard")]
    j = next(j for j in bt["judgment"] if j["run_id"] == r["run_id"])
    assert (j["n_items_judged"], j["n_judge_refusals"], j["n_errors"]) == (199, 1, 0) and r["n_items_judged"] == 199
    assert r["extra"]["or_bench_full_1319"] == {**r["extra"]["or_bench_full_1319"], "n_judged": 1317, "abstentions": 2}
    assert r["extra"]["abstentions"]["n_subset"] == 1 and r["extra"]["abstentions"]["n_all"] == 2 and set(r["extra"]["abstentions"]["items_all"]) == set(ABSTAIN[(LORA, "or_bench_hard_1k")])
    m = next(m for m in bt["run_metric"] if m["run_id"] == r["run_id"] and m["metric_id"] == "orb_refused")
    assert m["n"] == 199 and m["n_abstentions"] == 1 and abs(m["value"] - m["k"] / 199) < 1e-12
    abst = [i for i in bt["item_result"] if i["run_id"] == r["run_id"] and i["extra"].get("status") == "abstained"]
    assert len(abst) == 2 and all(i["label"] is None and i["value"] is None and i["metric_id"] == "orb_label" for i in abst)
    assert {i["item_id"]: i["extra"]["abstention_reason"] for i in abst} == ABSTAIN[(LORA, "or_bench_hard_1k")]
    assert {i["item_id"]: i["extra"]["in_subset"] for i in abst} == {"orb0002-" + _h8("orb2"): True, "orb0000-" + _h8("orb0"): False}
    # `none`'s XSTest pass abstained on an UNSAFE item: nothing of the scored 250 changes, the all-subsets count records it
    x = _by_key_eval(bt["run"])[("none", "xstest")]
    jx = next(j for j in bt["judgment"] if j["run_id"] == x["run_id"])
    assert (jx["n_items_judged"], jx["n_judge_refusals"], jx["n_errors"]) == (250, 0, 0) and x["extra"]["scored_subset"]["n_judge_refusals_all_subsets"] == 1


def test_the_label_mean_is_the_third_row_on_every_judged_run_and_pairs_against_none(tmp_path, seeds):
    """Beside `xst_refused` / `orb_refused` and `over_refusal_graded`, every judged run files
    `over_refusal_label_mean` -- (full + 0.5 x partial) / judged over the scored set, an at-cap reply by its label, declared_common's exact-variance
    interval -- on the same judgment, carrying the label census, the two siblings' values and (OR-Bench) the full-list reading; the steered key's
    contrast against `none` is the third of each sampled eval's, paired on the label score over the items judged in both."""
    from sisteer.registry.adapters import declared_common as DC
    fx = _root(tmp_path, seeds)
    _judge(fx)
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds, digest=_digest(tmp_path, fx)))
    assert facts["n_refusals"] == 0, facts["refusals"]
    by = _by_key_eval(bt["run"])
    lm_value = {}
    for key in fx["keys"]:
        # OR-Bench: the subset's census -> (full + 0.5 partial) / judged; the graded row's n, abstentions and judgment; the full list beside it
        r = by[(key, "or_bench_hard")]
        exp = _expected_subset(fx, key)
        ms = {m["metric_id"]: m for m in bt["run_metric"] if m["run_id"] == r["run_id"]}
        lm, g, head = ms["over_refusal_label_mean"], ms["over_refusal_graded"], ms["orb_refused"]
        k_eff = exp["full_refusal"] + 0.5 * exp["partial_refusal"]
        assert abs(lm["value"] - k_eff / exp["n_judged"]) < 1e-12 and lm["n"] == exp["n_judged"] == g["n"] and lm["n_abstentions"] == exp["n_abstained"] == g["n_abstentions"]
        assert lm["k"] == (int(k_eff) if float(k_eff).is_integer() else None) and lm["computed_by"] == "registry" and lm["computed_utc"] == g["computed_utc"]
        assert lm["judgment_id"] == g["judgment_id"] == head["judgment_id"] and lm["source_uri"] == g["source_uri"]
        assert lm["ci_method"] == DC.OVER_REFUSAL_LABEL_MEAN_CI_METHOD
        assert (lm["ci_lo"], lm["ci_hi"]) == pytest.approx(DC.label_mean_interval(exp["full_refusal"], exp["partial_refusal"], exp["n_judged"]))
        assert lm["extra"]["label_counts"] == {"full": exp["full_refusal"], "partial": exp["partial_refusal"], "compliance": exp["full_compliance"],
                                               "n": exp["n_judged"], "n_abstentions": exp["n_abstained"]}
        assert lm["extra"]["graded_value"] == g["value"] and lm["extra"]["refused_value"] == head["value"]
        assert (lm["extra"]["graded_metric_id"], lm["extra"]["refused_metric_id"]) == ("over_refusal_graded", "orb_refused")
        assert lm["extra"]["definition"] == DC.OVER_REFUSAL_LABEL_MEAN_DEFINITION and lm["extra"]["weights"] == DC.OVER_REFUSAL_LABEL_SCORE
        assert lm["extra"]["subset"] == SCT.DIGEST_ITEM_SET and lm["extra"]["derived_from"] == head["extra"]["derived_from"]
        full = r["extra"]["or_bench_full_1319"]
        want_full = (full["by_label"]["full_refusal"] + 0.5 * full["by_label"]["partial_refusal"]) / full["n_judged"]
        assert lm["extra"]["full_1319"]["n_judged"] == full["n_judged"] and abs(lm["extra"]["full_1319"]["label_mean"] - want_full) < 1e-12 and "SECONDARY" in lm["extra"]["full_1319"]["note"]
        assert r["field_confidence_json"]["over_refusal_label_mean"]["confidence"] == "derived"
        block = r["extra"]["over_refusal_headline"]["label_mean"]
        assert block["metric_id"] == "over_refusal_label_mean" and abs(block["value"] - lm["value"]) < 1e-12 and block["k_effective"] == k_eff and block["n"] == exp["n_judged"]
        assert block["label_counts"] == lm["extra"]["label_counts"]
        lm_value[(key, "or_bench_hard")] = lm["value"]
        # XSTest: the 250 safe prompts cycle full / full / partial / compliance -> 126 full + 62 partial -> k 157 (whole), 157 / 250
        x = by[(key, "xstest")]
        mx = {m["metric_id"]: m for m in bt["run_metric"] if m["run_id"] == x["run_id"]}
        xlm = mx["over_refusal_label_mean"]
        assert xlm["n"] == 250 and xlm["n_abstentions"] == 0 and xlm["k"] == 157 and abs(xlm["value"] - 157 / 250) < 1e-12
        assert xlm["extra"]["label_counts"] == {"full": 126, "partial": 62, "compliance": 62, "n": 250, "n_abstentions": 0} and xlm["extra"]["subset"] == "safe"
        assert "full_1319" not in xlm["extra"] and xlm["judgment_id"] == mx["over_refusal_graded"]["judgment_id"] == mx["xst_refused"]["judgment_id"]
        assert x["field_confidence_json"]["over_refusal_label_mean"]["confidence"] == "derived"
        lm_value[(key, "xstest")] = xlm["value"]
    # the contrast: one label-mean row per sampled eval for the steered key, delta = 100 x (LoRA's label mean - none's), Wilcoxon on the pairs
    lcs = {c["run_id"].split(":")[1]: c for c in bt["run_contrast"] if c["metric_id"] == "over_refusal_label_mean"}
    assert set(lcs) == {"xstest", "or_bench_hard"} and facts["contrasts"] == 6
    for eval_id, c in lcs.items():
        assert c["run_id"] == by[(LORA, eval_id)]["run_id"] and c["anchor_run_id"] == by[("none", eval_id)]["run_id"] and c["anchor_kind"] == "same_standup_none"
        assert abs(c["delta"] - 100.0 * (lm_value[(LORA, eval_id)] - lm_value[("none", eval_id)])) < 1e-9 and c["delta_method"] == "wilcoxon_signed_rank"
        assert c["computed_by"] == "registry" and c["effect_kind"] in ("dz", None) and c["extra"]["definition"] == DC.OVER_REFUSAL_LABEL_MEAN_DEFINITION
        assert c["extra"]["unit"] == "pp" and "label score" in c["extra"]["delta_text"] and c["extra"]["scored_set"] == ("safe" if eval_id == "xstest" else SCT.DIGEST_ITEM_SET)
        assert c["judgment_id"] == next(j["judgment_id"] for j in bt["judgment"] if j["run_id"] == c["run_id"])
        assert c["anchor_judgment_id"] == next(j["judgment_id"] for j in bt["judgment"] if j["run_id"] == c["anchor_run_id"])
    # XSTest pairs every safe item (none's abstention is an UNSAFE item); OR-Bench drops the LoRA's abstained subset item from the pairing
    assert lcs["xstest"]["n_pairs"] == 250 and lcs["or_bench_hard"]["n_pairs"] == _expected_subset(fx, LORA)["n_judged"]


def test_a_disagreeing_digest_a_half_landed_pass_and_a_foreign_rubric_refuse_by_name(tmp_path, seeds):
    # a digest disagreeing on one key's full_refusal count refuses that key's or_bench_hard alone; XSTest and MMLU still file
    fx = _root(tmp_path, seeds)
    _judge(fx)
    dg = _digest(tmp_path, fx, override={"none": {"full_refusal": 1}})
    bt, facts = _read(_adapter(fx, _roster(tmp_path), seeds, digest=dg))
    assert ("none", "or_bench_hard") not in _by_key_eval(bt["run"]) and len(bt["run"]) == 8
    assert facts["n_refusals"] == 1 and "OR's digest DISAGREES" in facts["refusals"][0] and "full_refusal: digest 1" in facts["refusals"][0], facts["refusals"]
    # an absent digest is recorded, never a refusal
    bt2, facts2 = _read(_adapter(fx, _roster(tmp_path), seeds))
    assert facts2["n_refusals"] == 0 and facts2["digest"]["state"] == "absent" and _by_key_eval(bt2["run"])[("none", "or_bench_hard")]["extra"]["digest_cross_check"]["state"] == "absent"
    # a half-landed pass (judge_items.jsonl missing on one key) refuses that key's two sampled evals; its MMLU files
    fx3 = _root(tmp_path / "half", seeds)
    _judge(fx3)
    os.remove(os.path.join(fx3["root"], *SCT.STORES_REL, STORE, SCT.KEYS_SUB, STORE_KEY[PROMPTED], "judge_items.jsonl"))
    bt3, facts3 = _read(_adapter(fx3, _roster(tmp_path / "half"), seeds))
    assert {e for k, e in _by_key_eval(bt3["run"]) if k == PROMPTED} == {"mmlu"} and len(bt3["run"]) == 7
    assert facts3["n_refusals"] == 1 and "half-landed judge pass" in facts3["refusals"][0], facts3["refusals"]
    # a rubric sha the seeded judge does not pin is a different judge: refused by name, the other control still files
    fx4 = _root(tmp_path / "rubric", seeds)
    _judge(fx4, rubric_override={"xstest": "0" * 16})
    bt4, facts4 = _read(_adapter(fx4, _roster(tmp_path / "rubric"), seeds))
    assert {e for k, e in _by_key_eval(bt4["run"])} == {"or_bench_hard", "mmlu"} and facts4["n_refusals"] == 3
    assert all("a different rubric is a different judge" in w for w in facts4["refusals"]), facts4["refusals"]


def test_the_harvester_moves_the_judged_rows_to_ingested_and_withdraws_the_pending_dispositions(tmp_path, seeds):
    fx = _root(tmp_path, seeds)
    rp = _roster(tmp_path)
    ad = _adapter(fx, rp, seeds)
    obj, _staged = ad.resolve_object(LORA, f"s{SEED}")
    root = _registry_root(tmp_path, seeds, [obj])
    R.write_records(os.path.join(root, "records", "standins", "000002-premint.jsonl"), _queued_records(fx, seeds, "2026-09-21T06:31:43.507Z"))
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    assert F.q(root, "SELECT eval_id, lifecycle_state, COUNT(*) FROM run WHERE campaign = ? GROUP BY 1, 2 ORDER BY 1", (SCT.CAMPAIGN,)) == \
           [("mmlu", "ingested", 3), ("or_bench_hard", "generated", 3), ("xstest", "generated", 3)]
    assert F.q(root, "SELECT COUNT(*) FROM disposition WHERE state = 'pending'") == [(6,)]
    ids_before = F.q(root, "SELECT run_id FROM run WHERE campaign = ? ORDER BY 1", (SCT.CAMPAIGN,))
    # the pass lands; the same reader reads the same root again
    _judge(fx)
    dg = _digest(tmp_path, fx)
    rep2, ing3, ing4 = F.run_and_ingest(root, _adapter(fx, rp, seeds, digest=dg), now="2026-09-21T15:00:00.000Z")
    assert rep2.groups_refused == 0, [r.error for r in rep2.results]
    assert ing3.refused + ing4.refused == 0, (ing3.refusals, ing4.refusals)
    assert F.q(root, "SELECT run_id FROM run WHERE campaign = ? ORDER BY 1", (SCT.CAMPAIGN,)) == ids_before, "the judged read files under the same ids"
    assert F.q(root, "SELECT eval_id, lifecycle_state, COUNT(*) FROM run WHERE campaign = ? GROUP BY 1, 2 ORDER BY 1", (SCT.CAMPAIGN,)) == \
           [("mmlu", "ingested", 3), ("or_bench_hard", "ingested", 3), ("xstest", "ingested", 3)]
    # the pending dispositions WITHDRAW with the revision: the registry's mechanism marks their observations (current 0, withdrawn by the judged
    # revision) -- the disposition rows themselves stay as history, and none is a CURRENT observation of the source any more
    obs = F.q(root, "SELECT current, withdrawn_by_revision_id IS NOT NULL, revision_id = withdrawn_by_revision_id FROM observation WHERE table_name = 'disposition'")
    assert len(obs) == 6 and all(cur == 0 and marked and not same for cur, marked, same in obs), obs
    assert F.q(root, "SELECT COUNT(*) FROM observation WHERE table_name = 'disposition' AND current = 1") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM judgment") == [(6,)] and F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    rows = F.q(root, "SELECT r.campaign_cell_id_legacy, r.eval_id, r.n_items_judged, j.n_items_judged, j.n_judge_refusals, r.judged_utc IS NOT NULL FROM run r JOIN judgment j USING (run_id) "
                     "WHERE r.campaign = ? ORDER BY 1, 2", (SCT.CAMPAIGN,))
    assert all(a == b and stamped for _k, _e, a, b, _ref, stamped in rows) and next(r for r in rows if r[0] == LORA and r[1] == "or_bench_hard")[2:5] == (199, 199, 1)
    # the ungraded read's `generated` item rows leave the CURRENT view (the outgoing revision's items are withdrawn); the label rows stand
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE metric_id = ?", (SCT.GENERATED_METRIC,)) == [(0,)], "the generated item rows withdraw; the label rows stand"
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE metric_id = 'orb_label' AND json_extract(extra_json, '$.in_subset') = 1") == [(600,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(6,)]      # 3 metrics x 2 sampled evals for the one steered key


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_priority_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[SCT.NAME]()
    assert isinstance(inst, SCT.SeedrepControlsAdapter) and inst.covers_sources == (SCT.SOURCE,) == SS.SeedrepStaticsAdapter.covers_sources == ("seedrep_battery",)
    assert inst.owner == "Reseed" and AB.emission_version_of(SCT.SeedrepControlsAdapter) == "2" and inst.rng_scope == "none"   # 2: the label mean
    assert inst.roots[0] == SCT.RS_DEFAULT == SS.RS_DEFAULT and inst.roots[1] == SCT.VOLATILE_PEEK_DEFAULT
    # a distinct priority from the statics reader's undeclared 0: the coverage check admits two adapters on one source at distinct priorities
    assert inst.priority == IC.PRIORITY_OWNING_READER != getattr(SS.SeedrepStaticsAdapter, "priority", 0)
    assert SCT.EVALS == ("xstest", "or_bench_hard", "mmlu") and SCT.AXIS_BY_EVAL == {"xstest": "xstest", "or_bench_hard": "or_bench", "mmlu": "mmlu"}
    assert SCT.STANDUP not in SS.EVAL_BY_STANDUP and SCT.SOURCE not in SC.SeedrepClusterAdapter.covers_sources
    assert inst.status_sources(f"{SCT.STANDUP}__s1") == (SCT.SOURCE,)
