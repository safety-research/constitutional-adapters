"""The `gsm8k_paired_allmodels` reader on a SYNTHETIC mirror shaped like the paired_allmodels GSM8K document (invented item ids, object shas
and trees, `$SHARED/...` paths with a stand-in user, no generation text): the seed-replicate document's store shape under the document's layout —
`<root>[/controls_paired_allmodels]/<tag>/<object dir>/<thinking_card|card>[__rung16384]/` — for the three standups (gemma4e4b_it, llama8b_it,
nemotron3nano4b) and the four (tag, key) pairs the document declares, read against an in-memory registry carrying the objects' rows.

What it pins:

* THE LAYOUT AND THE GRAIN: one group per tag (the `controls_paired_allmodels/` head accepted under the root), one run per (tag, key) keyed
  `<tag>/<base store key>` at the ladder's final state, the ceiling store's items replacing the base's item-wise; trial 0, no seed base, the default
  Inspect seed rule; the role `paired_allmodels_value` on the run and both metric rows;
* PER MODEL: the thinking spelling (gemma / nemotron `on` under `thinking_card`, llama `not_available` under `card`), the model's own decode policy row
  (`gemma4e4b_it|gsm8k|on|thinking_card`, `llama8b_it|gsm8k|not_available|card`, `nemotron3nano4b|gsm8k|on|thinking_card`), the per-model seed
  instrument MINTED with the record's facts (distinct ids per model), the transfer DERIVED from the fit provenance (`base_to_instruct` for the
  paired writes fit on the base checkpoints, `chat_checkpoint` for nemotron) — never the `it` spelling's `instruct_direct`;
* THE OBJECT: attested three ways by the store and bound to the registry's `object` row (an absent row, a kind that is not the key's, no connection:
  refused by name), to the roster / pins when the panel stages the key (nemotron's constitution ReLU; a disagreeing pin refuses), and to ReLU's
  staging manifest when present (a disagreeing sha refuses);
* THE ANCHORS: no `run_contrast`; every metric row parks `delta_block.anchor.anchor_state = pooled_in_frame_anchors` naming the in-frame unsteered
  GSM8K runs of the model derived by identity from the registry's `run` rows and the frame flag (`v_paper_current`), `unresolved` without the flag,
  mirrored on the run;
* REFUSED BY NAME (the pair, never the tag): a `none` object dir, an undeclared key, an undeclared tag, a store under the other model's sampling
  directory, a replicate seed block, an outcome record that is not the item set's 200 (unless the base manifest declares an `item_subset`), a
  provenance sha that is not the rows' aggregate, a declaration record that disagrees;
* THE PRODUCER'S RECORDS at the mirror root (store pins `g8k_store_pins/2` keyed `<model key>/<panel>/<key>@<cap>`, the three-number record
  `g8k_three_numbers_offtask/1` keyed `<tag>/<object dir>`, the final-state record) bind when present -- every count cross-checked, a disagreement
  or a pin that is not the manifest's tree refuses the pair by name, a malformed record refuses the tag;
* A LEG THE DRIVER REFUSED (RESULT `DRIVER_FAILED`, no accepted attempt, the render probe's ON draws all `unclosed_block`) files as a REFUSAL STATE:
  one `not_served` disposition spelling `thinking_unservable_predicate_a` (never null, never an absent row) with the RESULT line, the failed attempt,
  the evidence path and the pins' refused entry; a DRIVER_FAILED leg without that evidence spells `driver_failed_unclassified`;
* end to end: the runs reach `ingested` through the harvester's rule-scored completion with nothing blocked, the disposition lands beside them, and
  the seed-replicate reader's own tests stay green (its helpers only gained defaults).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest

import _b1_fixtures as F
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import gsm8k_paired_allmodels as P
from sisteer.registry.adapters import gsm8k_seedrep as G
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters.base import AdapterContext

NOW = "2026-09-22T06:00:00.000Z"
ITEM_IDS = [f"gsm8k_{i:08x}" for i in range(G.N_ITEMS)]
COMPOSED, CONSTITUTION = P.COMPOSED_KEY, P.CONSTITUTION_KEY
GEMMA, LLAMA, NEMO = "gemma4e4b_it", "llama8b_it", "nemotron3nano4b"
TREE = hashlib.sha1(b"synthetic paired allmodels tree").hexdigest()
DEFAULT_SEED_RULE = "sha256(item_id NUL sample_idx)[:4] big-endian mod 2^31-1"
ROUTING = "/workspace-vast/TESTUSER/exp/gsm8k_paired200_paper_objects_paired_allmodels/serving/g8k_{tag}_{slug}_{cap}_{job}/gsm8k/routing_table.json"
#: the four synthetic objects of the document (never the live shas)
OBJ = {(GEMMA, COMPOSED): hashlib.sha256(b"synthetic paired allmodels gemma composed").hexdigest(),
       (LLAMA, COMPOSED): hashlib.sha256(b"synthetic paired allmodels llama composed").hexdigest(),
       (NEMO, COMPOSED): hashlib.sha256(b"synthetic paired allmodels nemotron composed").hexdigest(),
       (NEMO, CONSTITUTION): hashlib.sha256(b"synthetic paired allmodels nemotron constitution").hexdigest()}
STAGED = {(GEMMA, COMPOSED): "paired_eb8__constv2-mix-full__psr_ce_all_minus_ctl", (LLAMA, COMPOSED): "paired_eb64__constv2-mix-full__psr_ce_all_minus_ctl",
          (NEMO, COMPOSED): "paired_eb8__constv2-mix-full__psr_ce_all_minus_ctl", (NEMO, CONSTITUTION): "base__constv2-mix-full__psr_ce_all"}
#: the models the tags spell: the served / fit checkpoints, the shipped decode row, the thinking spelling and store directory
MODELS = {GEMMA: dict(model_id="gemma4e4b_it", model_key="gemma4e4b", panel="it", served_hf="google/gemma-4-E4B-it", fit_hf="google/gemma-4-E4B", transfer=True, thinking="on",
                      sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64}, policy="gemma4e4b_it|gsm8k|on|thinking_card@v1", transfer_kind="base_to_instruct", fit_model_id="gemma4e4b_base"),
          LLAMA: dict(model_id="llama8b_it", model_key="llama8b", panel="it", served_hf="NousResearch/Meta-Llama-3.1-8B-Instruct", fit_hf="NousResearch/Meta-Llama-3.1-8B", transfer=True,
                      thinking="not_available", sampling={"temperature": 0.6, "top_p": 0.9}, policy="llama8b_it|gsm8k|not_available|card@v1", transfer_kind="base_to_instruct", fit_model_id="llama8b_base"),
          NEMO: dict(model_id="nemotron3nano4b", model_key="nemotron3nano4b", panel="main", served_hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", fit_hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                     transfer=False, thinking="on", sampling={"temperature": 1.0, "top_p": 0.95}, policy="nemotron3nano4b|gsm8k|on|thinking_card@v1", transfer_kind="chat_checkpoint",
                     fit_model_id="nemotron3nano4b")}
REVISION = {}


@pytest.fixture(scope="module")
def seeds():
    s = LC.Seeds.locate()
    for tag, m in MODELS.items():
        REVISION[tag] = str(s.models[m["model_id"]]["hf_revision"])
    return s


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
# the synthetic truth of one (tag, object key) pair and the files written from it
# --------------------------------------------------------------------------------------------

def _spec(*, n_correct: int, n_wrong: int, loops: int, truncated: int, below_cap: int, scored_at_cap: int = 0, rung=None, n_items: int = G.N_ITEMS) -> dict:
    """The per-item truth (the seed-replicate test's shape): `states[item] = (state, base value)`; `rung` maps an eligible item to (rung state, rung value)."""
    states = {}
    ids = iter(ITEM_IDS[:n_items])
    for _ in range(n_correct):
        states[next(ids)] = ("scored", 1.0)
    for _ in range(n_wrong):
        states[next(ids)] = ("scored", 0.0)
    for _ in range(loops):
        states[next(ids)] = ("loop", None)
    for _ in range(truncated):
        states[next(ids)] = ("truncated", None)
    for _ in range(scored_at_cap):
        states[next(ids)] = ("scored_at_cap", 0.0)
    for _ in range(below_cap):
        states[next(ids)] = ("below_cap", None)
    assert len(states) == n_items
    eligible = [i for i, (st, _v) in states.items() if st in ("truncated", "scored_at_cap")]
    if rung is not None:
        rung = {eligible[k]: v for k, v in enumerate(rung)}
        assert len(rung) == len(eligible)
    spec = {"states": states, "eligible": eligible, "rung": rung, "n_items": n_items}
    spec["exit"] = G.LADDER_EXIT_ESCALATE if rung else ("below_threshold" if eligible else ("triggered_no_eligible" if loops else "clean"))
    return spec


def _outcome(iid: str, st: str, v, cap: int, seed: int) -> dict:
    at_cap = st in ("loop", "truncated", "scored_at_cap", "ceiling")
    scored = st in ("scored", "scored_at_cap")
    return {"item_id": iid, "at_cap": at_cap, "scored": scored, "value": (v if scored else None), "stop_reason": ("max_tokens" if at_cap else "stop"),
            "completion_tokens": (cap if at_cap else 1500), "seed": seed, "sample_idx": 0, "think_block_unterminated": (st in ("loop", "truncated", "below_cap", "ceiling")),
            "unparseable": False, "attempt": 1, "prompt_tokens": 2250}


def _store_files(d: str, tag: str, okey: str, slug: str, cap: int, items: dict, *, obj_sha: str, tree: str, job: str, ihash8: str, chash8: str, spec: dict, subset=None,
                 source_identity_hash: str = None, started="2026-09-22T03:12:50Z", ended="2026-09-22T03:40:02Z", oos: str = None, prov_sha: str = None, seed_block=None,
                 node: str = "gpu-4", agg_sha: str = None, fit_hf: str = None, transfer=None, thinking: str = None, sampling: dict = None, item_subset_base=None) -> str:
    """One store directory (base or ceiling) of a document pair: the named files the reader opens, the model's served / fit identity, thinking and
    sampling from MODELS[tag]. `seed_block` (a replicate block) makes the store a seed-replicate one; `prov_sha` / `agg_sha` let the two 64-hex
    attestations disagree; `item_subset_base` puts an `item_subset` on a BASE manifest (a declared count)."""
    m = MODELS[tag]
    thinking = thinking or m["thinking"]
    sampling = dict(sampling or m["sampling"])
    fit_hf = fit_hf or m["fit_hf"]
    transfer = m["transfer"] if transfer is None else transfer
    store_key = f"{slug}__controls__t0__a0__{ihash8}__{chash8}"
    kd = os.path.join(d, "cells", store_key)
    prov_sha = prov_sha if prov_sha is not None else obj_sha
    agg_sha = agg_sha if agg_sha is not None else obj_sha
    outcomes = [_outcome(iid, st, v, cap, 1000 + k) for k, (iid, (st, v)) in enumerate(sorted(items.items()))]
    n_scored = sum(1 for o in outcomes if o["scored"])
    n_at_cap = sum(1 for o in outcomes if o["at_cap"])
    correct = sum(o["value"] for o in outcomes if o["scored"])
    acc = (correct / n_scored) if n_scored else None
    _write(os.path.join(kd, "outcomes", G.OUTCOMES_FILE), {"task": G.TASK, "item_set": "paired200", "items": outcomes, "max_tokens": cap, "n_attempted_items": len(items),
                                                          "n_scored": n_scored, "n_unscored": len(items) - n_scored, "n_at_cap": n_at_cap, "producer_tree_sha": tree})
    lines = []
    for k, o in enumerate(outcomes):
        if not o["scored"]:
            continue
        ex = {"answer": "42", "completion_tokens": o["completion_tokens"], "stop_reason": o["stop_reason"], "seed": o["seed"], "sample_idx": 0, "max_tokens": cap,
              "object_artifact_sha256": agg_sha, "view": m["panel"], "route_kind": "lens"}
        lines.append(json.dumps({"task": G.TASK, "metric": "acc", "item_id": o["item_id"], "value": o["value"], "created_at": 1790046000.0 + k, "slurm_job_id": job,
                                 "env_digest": "5a1c0e2b9d7f3311", "model_id": m["served_hf"], "harness_config_hash": chash8 + "870d1a0b", "intervention_hash": ihash8 + "2e209076",
                                 "extra_json": json.dumps(ex)}))
    agg = {"object_artifact_sha256": agg_sha, "producer_tree_sha": tree, "route_kind": "lens", "sampling_mode": P.SAMPLING_DIR_BY_THINKING[thinking], "view": m["panel"]}
    lines.append(json.dumps({"task": G.TASK, "metric": "acc_aggregate", "item_id": "__aggregate__", "value": acc, "extra_json": json.dumps(agg)}))
    lines.append(json.dumps({"task": G.TASK, "metric": "n", "item_id": "__aggregate__", "value": float(len(items)), "extra_json": json.dumps(agg)}))
    _write(os.path.join(kd, "rows.jsonl"), "\n".join(lines) + "\n")
    gs = dict(sampling, acc=acc, n=len(items), n_scored=n_scored, n_unscored=len(items) - n_scored, n_at_cap=n_at_cap, max_tokens=cap, thinking=thinking, task=G.TASK,
              framework="inspect_evals-0.18.0", engine="vllm", samples_per_item=1)
    _write(os.path.join(kd, "gen_stats.json"), {G.TASK: gs})
    inspect_seeds = {"rule": DEFAULT_SEED_RULE, "samples_per_item": {G.TASK: 1}}
    if seed_block is not None:
        inspect_seeds["replicate"] = seed_block
    man = {"cell_id": okey, "harness_cell_id": okey, "model_id": m["served_hf"], "model_revision": REVISION[tag], "serving_identity": G.STEERED_IDENTITY, "serving_path": "lora",
           "serving_engine": "lora_native", "steer_positions": "all",
           "fit_provenance": {"object_sha": (obj_sha[:16] if obj_sha else None), "model_id": fit_hf, "transfer": transfer, "served_model_id": m["served_hf"], "served_revision": REVISION[tag]},
           "producer": {"driver": "served_capability_run", "tree_sha": tree}, "row_constants": {"slurm_job_id": job, "env_digest": "5a1c0e2b9d7f3311"},
           "routing_table": ROUTING.format(tag=tag, slug=slug, cap=cap, job=job), "harness_config_hash": chash8 + "870d1a0b", "intervention_hash": ihash8 + "2e209076",
           "geometry": {"max_model_len": (16640 if cap == G.CAP else 24832), "max_num_seqs": (31 if cap == G.CAP else 21), "source": "endpoint.json"},
           "seeds": {"inspect": inspect_seeds}, "frameworks": {"inspect_ai": "0.3.260", "inspect_evals": "0.18.0"}, "sampling": dict(sampling, mode=P.SAMPLING_DIR_BY_THINKING[thinking]),
           "resolved_config": {"cell_id": okey, "system_sha": None, "serving": {"serving_engine": "lora_native"}},
           "provenance": {"artifact_sha256": prov_sha, "cell_id": okey}, "oos_experiment": oos}
    if subset is not None:
        man["item_subset"] = {"cap": cap, "ids": sorted(subset), "n": len(subset), "source_cap": G.CAP, "task": G.TASK, "source_identity_hash": source_identity_hash}
    if item_subset_base is not None:
        man["item_subset"] = dict(item_subset_base)
    _write(os.path.join(kd, "manifest.json"), man)
    _write(os.path.join(kd, "evidence.json"), {"written_at": ended, "serving_identity": man["serving_identity"]})
    _write(os.path.join(d, "evidence.json"), {"written_at": ended, "serving_identity": man["serving_identity"]})
    _write(os.path.join(d, "resolved_sampler.json"), {"verdict": "AGREES", "classes": {"client": {"n": len(items), "verdict": "AGREES", "distinct_resolved": [{"sampler": sampling, "n": len(items)}]}}})
    _write(os.path.join(d, "request_evidence.jsonl"), json.dumps({"kind": "task", "task": G.TASK, "item_id": ITEM_IDS[0], "seed": 1000, "max_tokens": cap, **sampling}) + "\n")
    _write(os.path.join(d, "usage_ledger.jsonl"), json.dumps({"task": G.TASK, "finish_reason": "stop", "max_tokens_requested": cap}) + "\n")
    _write(os.path.join(d, "ladder_state.json"), {"schema": G.LADDER_STATE_SCHEMA, "accepted_generation_attempt": "a1",
                                                  "attempts": [{"attempt_id": "a1", "job_id": job, "job_name": f"g8k-{tag}-{slug}", "node": node, "cluster": "cluster_a",
                                                                "result": "COMPLETE", "rc": 0, "started": started, "ended": ended, "rung": str(cap), "tag": tag, "sha": tree}]})
    _write(os.path.join(d, "RESULT"), f"{ended} job={job} row=0 g8k-{tag}-{slug} g8k {m['model_key']} {m['panel']} {okey} rung={cap} attempt=a1 rc=0 driver_rc=0 result=COMPLETE\n")
    if cap == G.CAP:
        states = spec["states"]
        loops = sorted(i for i, (st, _v) in states.items() if st == "loop")
        facts = []
        for o in outcomes:
            st = states[o["item_id"]][0]
            is_loop = st == "loop"
            facts.append({"item_id": o["item_id"], "at_cap": o["at_cap"], "scored": o["scored"], "value": o["value"], "degenerate": is_loop,
                          "kind": ("non_terminating" if is_loop else ("truncated" if o["at_cap"] else None)), "eligible": (o["item_id"] in spec["eligible"]),
                          "rep_ratio": (0.8 if is_loop else 0.05), "tail_rep_ratio": (0.9 if is_loop else 0.03), "text_sha256": hashlib.sha256(o["item_id"].encode()).hexdigest(),
                          "periodic_block": {"loop": is_loop}, "stop_reason": o["stop_reason"], "completion_tokens": o["completion_tokens"]})
        _write(os.path.join(d, "cap_ladder.json"), {"schema": G.CAP_LADDER_SCHEMA, "cell_id": store_key, "source": {"is_rung": False, "task": G.TASK}, "cap": G.CAP, "ceiling": G.CEILING,
                                                    "n": len(items), "n_at_cap": n_at_cap, "n_eligible": len(spec["eligible"]), "n_non_terminating": len(loops), "exit": spec["exit"],
                                                    "eligible_ids": sorted(spec["eligible"]), "non_terminating_ids": loops, "item_facts": facts,
                                                    "rules": {"at_cap": "outcome.stop_reason == 'max_tokens'", "degenerate": "rep_ratio(text) > 0.5 or rep_ratio(last 200 whitespace tokens) > 0.5",
                                                              "trigger": "trigger iff n_at_cap / n > 0.01", "periodic_block": "non-deciding"}})
    return store_key


#: the pairs' shapes: gemma's composed object escalates (6 eligible, 1 still at cap), llama's exits below threshold, nemotron's composed is all loops,
#: nemotron's constitution ReLU escalates 3 items (all re-drawn correct)
SPECS = {(GEMMA, COMPOSED): dict(n_correct=160, n_wrong=10, loops=22, truncated=5, scored_at_cap=1, below_cap=2,
                                 rung=[("scored", 1.0), ("scored", 1.0), ("scored", 1.0), ("scored", 0.0), ("ceiling", None), ("scored", 1.0)]),
         (LLAMA, COMPOSED): dict(n_correct=166, n_wrong=31, loops=2, truncated=1, below_cap=0, rung=None),
         (NEMO, COMPOSED): dict(n_correct=171, n_wrong=15, loops=14, truncated=0, below_cap=0, rung=None),
         (NEMO, CONSTITUTION): dict(n_correct=184, n_wrong=12, loops=1, truncated=3, below_cap=0, rung=[("scored", 1.0), ("scored", 1.0), ("scored", 1.0)])}
OOS = {GEMMA: "gsm8k_paired200_gemma4e4b_native_paired_{cap}", NEMO: "gsm8k_paired200_nemotron3nano4b_native_paired_{cap}"}


def _three_row(tag: str, okey: str, spec: dict, obj_sha: str) -> dict:
    """The producer's `g8k_three_numbers_offtask/1` row for one pair from the synthetic truth (the value, the bound, the third state, the readings)."""
    states, rung = spec["states"], spec["rung"]
    scored_b = {i: v for i, (st, v) in states.items() if st in ("scored", "scored_at_cap")}
    n_at_cap = sum(1 for st, _v in states.values() if st in ("loop", "truncated", "scored_at_cap"))
    loops = sum(1 for st, _v in states.values() if st == "loop")
    final = dict(scored_b)
    still = r_scored = r_correct = 0
    if rung:
        for iid, (rst, rv) in rung.items():
            final.pop(iid, None)
            if rst == "scored":
                final[iid] = rv; r_scored += 1; r_correct += int(rv)
            elif rst == "ceiling":
                still += 1
    not_esc = 0 if rung else sum(1 for st, _v in states.values() if st == "truncated")
    below = sum(1 for st, _v in states.values() if st == "below_cap")
    correct_f, n = int(sum(final.values())), spec["n_items"]
    return {"tag": tag, "cell_slug": okey.replace("__", "--", 1), "cell_id": okey, "arm": LC.cell_grammar_legacy(okey)["paper_label"], "sampling": P.SAMPLING_DIR_BY_THINKING[MODELS[tag]["thinking"]],
            "n_attempted": n, "value_correct_over_scored": round(correct_f / len(final), 4), "n_scored_final": len(final), "correct_final": correct_f,
            "bound_correct_over_attempted": round(correct_f / n, 4),
            "third_state": {"n_unresolved_at_cap": loops + still + not_esc, "share": (loops + still + not_esc) / n, "loops": loops, "still_at_ceiling_after_rung": still,
                            "eligible_not_escalated": not_esc, "n_unscored_below_cap": below, "unscored_below_cap_ids": []},
            "base_8192": {"n_scored": len(scored_b), "correct": int(sum(scored_b.values())), "n_at_cap": n_at_cap, "ladder_exit": spec["exit"], "n_eligible": len(spec["eligible"])},
            "rung_16384": ({"n_drawn": len(rung), "n_scored": r_scored, "correct": r_correct, "still_at_cap": still} if rung else None),
            "off_task": {"counts": {"other_incorrect": len(final) - correct_f, "other_unscored": below}, "share_off_task_structural": 0.0}, "record_role": P.ROLE, "object_sha256": obj_sha}


def _fstate_row(tag: str, okey: str, spec: dict) -> dict:
    states, rung = spec["states"], spec["rung"]
    scored_b = {i: v for i, (st, v) in states.items() if st in ("scored", "scored_at_cap")}
    final = dict(scored_b)
    still = 0
    if rung:
        for iid, (rst, rv) in rung.items():
            final.pop(iid, None)
            if rst == "scored":
                final[iid] = rv
            elif rst == "ceiling":
                still += 1
    unresolved = sum(1 for st, _v in states.values() if st == "loop") + still + (0 if rung else sum(1 for st, _v in states.values() if st == "truncated"))
    n = spec["n_items"]
    return {"tag": tag, "store_cell": okey.replace("__", "--", 1), "n_items": n, "n_scored_8192": len(scored_b), "correct_8192": int(sum(scored_b.values())), "rung_present": bool(rung),
            "n_rung_drawn": len(rung or {}), "n_rung_drawn_that_were_scored_at_base": len(set(rung or {}) & set(scored_b)), "n_scored_final": len(final), "correct_final": int(sum(final.values())),
            "acc_over_scored_final": round(sum(final.values()) / len(final), 4), "n_unresolved_final": unresolved, "n_unscored_below_cap_final": sum(1 for st, _v in states.values() if st == "below_cap"),
            "n_scored_at_cap_final": (0 if rung else sum(1 for st, _v in states.values() if st == "scored_at_cap")), "acc_over_attempted_final": round(sum(final.values()) / n, 4),
            "rung_items_not_in_base": 0, "files": []}


REFUSED_RESULT = ("2026-09-22T05:28:09+00:00 job=596511 array=1 row=1 g8k-nemotron3nano4b-main-mix-psr g8k nemotron3nano4b main constv2-mix-full__psr_ce_all rung=8192 fitted=1 declared=0 "
                  "attempt=a1 rc=2 driver_rc=1 result=DRIVER_FAILED")


def _refused_store(d: str, tag: str, okey: str, slug: str, obj_sha: str, *, tree: str = TREE, probe_classes=("unclosed_block", "unclosed_block", "unclosed_block"), accepted=None) -> str:
    """A leg the driver refused (THINKING_UNSERVABLE predicate A): RESULT ending DRIVER_FAILED, ladder_state.json accepting no attempt, the manifest and the
    render-probe evidence in the cell dir, no gen_stats / rows / outcomes / cap_ladder."""
    m = MODELS[tag]
    store_key = f"{slug}__controls__t0__a0__6c9b469b__69e571f0"
    kd = os.path.join(d, "cells", store_key)
    man = {"cell_id": okey, "harness_cell_id": okey, "model_id": m["served_hf"], "model_revision": REVISION[tag], "serving_identity": G.STEERED_IDENTITY, "serving_engine": "lora_native",
           "steer_positions": "all", "fit_provenance": {"object_sha": obj_sha[:16], "model_id": m["fit_hf"], "chat_template_bypassed": True}, "producer": {"driver": "served_capability_run", "tree_sha": tree},
           "row_constants": {"slurm_job_id": "596511"}, "routing_table": ROUTING.format(tag=tag, slug=slug, cap=G.CAP, job="596511"), "harness_config_hash": "69e571f09dea1012",
           "provenance": {"artifact_sha256": obj_sha, "cell_id": okey}, "oos_experiment": "gsm8k_paired200_nemotron3nano4b_native_8192", "seeds": {"inspect": {"rule": DEFAULT_SEED_RULE}}}
    _write(os.path.join(kd, "manifest.json"), man)
    ev = {"cell_id": okey, "serving_identity": G.STEERED_IDENTITY, "written_at": "2026-09-22T05:27:49Z",
          "render_certified": {"family": "nemotron_h", "think_shape_certified": False, "render_regime_certified": False, "header_certified": False, "render_certified": False,
                               "split_mode": "implicit_open", "evidence_status": "certified", "standup_thinking_mode": "on",
                               "certification_problems": ["the ON probe completion does not carry the family's think shape"]},
          "render_probe": {"off": {"content": "43", "finish_reason": "stop"}, "on": {"content": "...", "finish_reason": "stop", "reasoning": None},
                           "on_attempts": [{"attempt": i + 1, "prompt_id": "arithmetic", "seed": i, "max_tokens": 512, "finish_reason": "stop", "reasoning_chars": 0, "content_chars": 100, "class": c}
                                           for i, c in enumerate(probe_classes)]}}
    _write(os.path.join(kd, "evidence.json"), ev)
    _write(os.path.join(d, "evidence.json"), ev)
    _write(os.path.join(d, "ladder_state.json"), {"schema": G.LADDER_STATE_SCHEMA, "accepted_generation_attempt": accepted,
                                                  "attempts": [{"attempt_id": "a1", "job_id": "596511", "job_name": "g8k-fitted", "node": "gpu-20", "cluster": "cluster_a", "result": "DRIVER_FAILED",
                                                                "rc": 2, "started": "2026-09-22T05:23:08Z", "ended": "2026-09-22T05:28:09Z", "rung": "8192", "tag": tag, "sha": tree}]})
    _write(os.path.join(d, "RESULT"), REFUSED_RESULT + "\n")
    _write(os.path.join(d, "request_evidence.jsonl"), "")
    _write(os.path.join(d, "usage_ledger.jsonl"), "")
    _write(os.path.join(d, "resolved_sampler.json"), {"verdict": "NO_REQUESTS"})
    return store_key


def _mirror(tmp_path, seeds, *, root: str = None, pairs=None, head: bool = False, specs: dict = None, store_over: dict = None, dir_over: dict = None, extra_dirs=(),
            records: bool = False, refused=(), pins_over=None, three_over=None, fstate_over=None) -> dict:
    """A synthetic mirror root: the document's four pairs (or `pairs`), each a base store and (per spec) a ceiling store, plus `extra_dirs` (a list of
    (tag, slug, store dir, okey, obj sha) stores outside the document); a synthetic roster / pins staging nemotron's constitution key. `records` writes
    the producer's three records at the root from the same truth (`*_over` edit them); `refused` lists (tag, okey) legs written as DRIVER_FAILED stores."""
    root = root or str(tmp_path / "gsm8k_paired_allmodels")
    troot = os.path.join(root, P.CONTROLS_HEAD) if head else root
    pairs = list(pairs if pairs is not None else SPECS)
    specs = dict(specs or {})
    keys = {}
    pins = {"_meta": {"schema": P.PINS_SCHEMA, "label": "synthetic", "refused": {}}}
    three = {"schema": P.THREE_NUMBERS_SCHEMA, "written": NOW, "root": "/shared/TESTUSER/exp/gsm8k_paired200_paper_objects_paired_allmodels", "prefix": P.CONTROLS_HEAD, "rows": {}}
    fstate = {}
    for tag, okey in refused:
        m = MODELS[tag]
        slug = okey.replace("__", "--", 1)
        base_dir, _c = P.store_dirs_for(m["thinking"])
        keys[(tag, okey, G.CAP)] = _refused_store(os.path.join(troot, tag, slug, base_dir), tag, okey, slug, OBJ[(tag, okey)])
        pins["_meta"]["refused"][P.Gsm8kPairedAllmodelsAdapter._pin_key(m["model_key"], m["panel"], okey, G.CAP)] = {"RESULT": REFUSED_RESULT, "tree": TREE}
    for k, (tag, okey) in enumerate(pairs):
        spec = specs.get((tag, okey)) or _spec(**SPECS[(tag, okey)])
        slug = okey.replace("__", "--", 1)
        base_dir, ceil_dir = P.store_dirs_for(MODELS[tag]["thinking"])
        base_dir = (dir_over or {}).get((tag, okey), base_dir)
        ihash8, job = f"{k:08x}", str(600100 + k)
        over = dict((store_over or {}).get((tag, okey), {}))
        oos = OOS.get(tag)
        keys[(tag, okey, G.CAP)] = _store_files(os.path.join(troot, tag, slug, base_dir), tag, okey, slug, G.CAP, dict(spec["states"]), obj_sha=OBJ[(tag, okey)], tree=TREE, job=job,
                                               ihash8=ihash8, chash8=f"aaaa{k:04x}", spec=spec, oos=(oos.format(cap=G.CAP) if oos else None), **over)
        if spec["rung"]:
            rung_items = {iid: (rst, rv) for iid, (rst, rv) in spec["rung"].items()}
            keys[(tag, okey, G.CEILING)] = _store_files(os.path.join(troot, tag, slug, f"{base_dir}__rung{G.CEILING}"), tag, okey, slug, G.CEILING, rung_items, obj_sha=OBJ[(tag, okey)],
                                                       tree=TREE, job=str(600200 + k), ihash8=ihash8, chash8=f"bbbb{k:04x}", spec=spec, subset=list(spec["rung"]),
                                                       source_identity_hash=f"aaaa{k:04x}870d1a0b", started="2026-09-22T04:05:00Z", ended="2026-09-22T04:12:00Z",
                                                       oos=(oos.format(cap=G.CEILING) if oos else None), **{kk: vv for kk, vv in over.items() if kk in ("thinking", "sampling", "fit_hf", "transfer")})
        _write(os.path.join(troot, tag, slug, G.TARGET_PROVENANCE_FILE), "# synthetic mirror of the paired_allmodels document\n")
        m = MODELS[tag]
        pins[P.Gsm8kPairedAllmodelsAdapter._pin_key(m["model_key"], m["panel"], okey, G.CAP)] = TREE
        if spec["rung"]:
            pins[P.Gsm8kPairedAllmodelsAdapter._pin_key(m["model_key"], m["panel"], okey, G.CEILING)] = TREE
        three["rows"][f"{tag}/{slug}"] = _three_row(tag, okey, spec, OBJ[(tag, okey)])
        fstate[f"{tag}/{slug}"] = _fstate_row(tag, okey, spec)
    if records:
        if pins_over:
            pins_over(pins)
        if three_over:
            three_over(three)
        if fstate_over:
            fstate_over(fstate)
        _write(os.path.join(root, P.PINS_FILE), pins)
        _write(os.path.join(root, P.THREE_NUMBERS_FILE), three)
        _write(os.path.join(root, P.FINAL_STATE_FILE), fstate)
    for tag, slug, store_dir, okey, sha in extra_dirs:
        spec = _spec(**SPECS[(NEMO, COMPOSED)])
        _store_files(os.path.join(troot, tag, slug, store_dir), tag, okey, slug, G.CAP, dict(spec["states"]), obj_sha=sha, tree=TREE, job="600900", ihash8="ffffffff", chash8="cccc0000", spec=spec)
    import yaml
    roster = {"targets": {"nemotron3nano4b": {"views": {"main": {"config": "configs/sdfbase_full22_nemotron3nano4b.yaml", "cells": {CONSTITUTION: STAGED[(NEMO, CONSTITUTION)]}}}},
                          "gemma4e4b": {"views": {"it": {"cells": {"constv2-mix-full__lora_r4_ce": "it__constv2-mix-full__lora_r4_ce"}}}}}}
    roster_path = _write(str(tmp_path / "roster.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / "pins.json"), {"nemotron3nano4b": {STAGED[(NEMO, CONSTITUTION)]: {"object_sha256": OBJ[(NEMO, CONSTITUTION)], "source": "synthetic"}},
                                                     "gemma4e4b": {"it__constv2-mix-full__lora_r4_ce": {"object_sha256": hashlib.sha256(b"synthetic gemma lora").hexdigest()}}})
    return {"root": root, "roster": roster_path, "pins": pins_path, "keys": keys}


def _conn(objects=None, *, frame=None, runs=None) -> sqlite3.Connection:
    """An in-memory registry with the objects' rows (kind / label from the key's grammar), and — when `runs` is given — a `run` table plus a
    `v_paper_current` flag view over `frame` {run_id: (in_current_paper, selection_state)}."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, object_kind TEXT, paper_label TEXT, object_role TEXT, fit_model_id TEXT, derivation TEXT, status TEXT, campaign_cell_id_legacy TEXT)")
    for (tag, okey), sha in (objects if objects is not None else OBJ).items():
        g = LC.cell_grammar_legacy(okey)
        conn.execute("INSERT INTO object VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (sha, g["object_kind"], g["paper_label"], g["object_role"], MODELS[tag]["fit_model_id"], g["derivation"], "approved", STAGED[(tag, okey)]))
    if runs is not None:
        conn.execute("CREATE TABLE run (run_id TEXT PRIMARY KEY, model_id TEXT, object_id TEXT, eval_id TEXT, view_legacy TEXT, lane_source_legacy TEXT, instrument_id TEXT, token_cap_tokens INTEGER, "
                     "thinking_mode TEXT, lifecycle_state TEXT)")
        conn.executemany("INSERT INTO run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", runs)
        conn.execute("CREATE TABLE paper_flag (run_id TEXT PRIMARY KEY, in_current_paper INTEGER, selection_state TEXT)")
        conn.executemany("INSERT INTO paper_flag VALUES (?, ?, ?)", [(rid, f[0], f[1]) for rid, f in (frame or {}).items()])
        conn.execute("CREATE VIEW v_paper_current AS SELECT run_id, in_current_paper, selection_state FROM paper_flag")
    conn.commit()
    return conn


def _adapter(fx: dict, seeds, **kw) -> P.Gsm8kPairedAllmodelsAdapter:
    kw.setdefault("staging_dir", None)
    kw.setdefault("declaration_path", str(os.path.join(fx["root"], "..", "no_such_declaration.json")))
    return P.Gsm8kPairedAllmodelsAdapter(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], **kw)


def _read(ad: P.Gsm8kPairedAllmodelsAdapter, group: str, conn=None):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert group in groups, (group, sorted(groups))
    recs = []
    for batch in ad.read(_ctx(conn), group, groups[group]):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group), sorted(groups)


def _run(bt, okey: str) -> dict:
    hits = [r for r in bt.get("run", []) if r["campaign_cell_id_legacy"] == okey]
    assert len(hits) == 1, (okey, len(hits))
    return hits[0]


def _metrics(bt, rid: str):
    rows = [m for m in bt["run_metric"] if m["run_id"] == rid]
    value = [m for m in rows if m["computation_id"] == G.merge_computation_id()]
    bound = [m for m in rows if m["computation_id"] == ""]
    assert len(rows) == 2 and len(value) == 1 and len(bound) == 1, rows
    return value[0], bound[0]


def _refused(facts, tag: str, okey: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{tag}|{okey} -- ")]
    assert len(hits) == 1, facts["refusals"]
    return hits[0]


# --------------------------------------------------------------------------------------------
# layout, identity, the per-model facts
# --------------------------------------------------------------------------------------------

def test_discover_groups_the_stores_by_tag_under_the_root_or_its_controls_head(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, head=True)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = sorted(os.path.basename(i.uri) for i in inputs)
    assert names.count("cap_ladder.json") == 4 and names.count(G.OUTCOMES_FILE) == 6 and names.count("rows.jsonl") == 6 and names.count(G.TARGET_PROVENANCE_FILE) == 4
    assert {ad.comparison_group(i) for i in inputs} == {GEMMA, LLAMA, NEMO}
    # the same tree without the head groups the same way
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "flat"))
    assert {_adapter(fx2, seeds).comparison_group(i) for i in _adapter(fx2, seeds).discover()} == {GEMMA, LLAMA, NEMO}


def test_each_pair_files_one_run_at_the_final_state_with_the_models_own_thinking_decode_transfer_and_minted_instrument(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    conn = _conn()
    runs = {}
    for tag in (GEMMA, LLAMA, NEMO):
        bt, facts, groups = _read(ad, tag, conn=conn)
        assert groups == [GEMMA, LLAMA, NEMO] and facts["n_refusals"] == 0, facts["refusals"]
        m = MODELS[tag]
        for okey in P.DOCUMENT[tag][2]:
            r = _run(bt, okey)
            runs[(tag, okey)] = r
            base_key = fx["keys"][(tag, okey, G.CAP)]
            assert r["producer_run_key_legacy"] == f"{tag}/{base_key}" and r["run_id"] == P.run_id_for(tag, base_key, okey, r["instrument_id"])
            assert r["run_id"] == I.run_id(LC.CAMPAIGN, f"{tag}/{base_key}", okey, "gsm8k", r["instrument_id"])
            assert r["model_id"] == m["model_id"] and r["view_legacy"] == tag and r["lane_source_legacy"] == P.SOURCE and r["campaign"] == LC.CAMPAIGN and r["producer_session"] == "Overflow"
            assert r["object_id"] == OBJ[(tag, okey)] and r["transfer_kind"] == m["transfer_kind"] and r["extra"]["fit_model_id"] == m["fit_model_id"]
            assert r["thinking_mode"] == m["thinking"] and r["extra"]["thinking_spelling"] == m["thinking"] and r["extra"]["sampling_dir"] == P.SAMPLING_DIR_BY_THINKING[m["thinking"]]
            assert r["decode_sent_json"] == m["sampling"] and r["extra"]["decode_policy_ref_legacy"] == m["policy"] and r["decode_source"] == "generation_config" and r["instrument_match"] == "match"
            assert r["trial_index"] == 0 and r["seed_base"] is None and r["decode_seed_rule"] == "per_item" and r["extra"]["decode_seed_policy"] == "default" and r["extra"]["fit_seed"] is None
            assert r["extra"]["declared_role"] == P.ROLE and r["extra"]["document"] == "paired_allmodels" and r["extra"]["trial_rule"] == G.DEFAULT_SEED_TRIAL_RULE
            assert r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all" and r["code_sha"] == TREE and r["model_revision"] == REVISION[tag]
            assert r["substrate"] == "cluster_a" and r["substrate_source"] == "derived" and r["extra"]["store_pins"] is None and r["extra"]["tree_attestation"] == G.TREE_ATTESTATION_TARGET + "; no store pins beside this leg"
            assert r["extra"]["anchor_state"] == P.ANCHOR_STATE and r["extra"]["in_frame_unsteered_reference"]["anchor_state"] == P.ANCHOR_STATE
            assert r["is_operating_point"] == 1 and r["strength_nominal"] == 1.0 and r["n_shot"] == 10 and r["scoring_rule"] == "numeric_match"
            # the appendix column's per-model instrument: the canonical seed row bound to the model (the per-model expansion), nothing minted
            seed_inst = seeds.instrument_row(G.INSTRUMENT_SEED, "gsm8k", model_id=m["model_id"])
            assert r["instrument_id"] == str(seed_inst["instrument_id"]) and r["extra"]["instrument_minted"] is None and r["extra"]["instrument_rule"] == P.INSTRUMENT_RULE
            assert seed_inst["thinking_mode"] == m["thinking"] and seed_inst["decode_temperature"] == m["sampling"]["temperature"] and "instrument" not in bt
            fc = r["field_confidence_json"]
            assert fc["transfer_kind"]["confidence"] == "derived" and "fit_provenance" in fc["transfer_kind"]["source"] and fc["seed_base"]["confidence"] == "derived"
            assert fc["thinking_mode"]["confidence"] == "recorded" and fc["object_id"]["source"].startswith("attested by the store")
            # no contrast, no state event: the anchors are parked on both metric rows
            assert "run_contrast" not in bt and "run_state" not in bt
            value, bound = _metrics(bt, r["run_id"])
            for row in (value, bound):
                blk = row["extra"]["delta_block"]
                assert blk["anchor"]["anchor_state"] == P.ANCHOR_STATE and blk["anchor"]["anchor_run_id"] is None and blk["delta"] is None and blk["paired"] is False
                assert blk["anchor"]["state"] == "unresolved" and "the connected registry" in blk["anchor"]["why"] and row["extra"]["declared_role"] == P.ROLE
            assert value["computed_by"] == "registry" and bound["computed_by"] == "producer" and bound["producer_metric_spelling"] is None and bound["computation_id"] == ""
    # the instruments differ per model (thinking, decode) and the run ids are distinct
    assert len({r["instrument_id"] for r in runs.values()}) == 3 and len({r["run_id"] for r in runs.values()}) == 4
    assert runs[(NEMO, COMPOSED)]["instrument_id"] == runs[(NEMO, CONSTITUTION)]["instrument_id"]
    # the transfer is the object's, never the standup spelling's (`it` reads instruct_direct in legacy_spellings.yaml)
    assert seeds.model_for_view_legacy(GEMMA)["transfer_kind"] == "instruct_direct" and runs[(GEMMA, COMPOSED)]["transfer_kind"] == "base_to_instruct"
    # gemma's composed object: the three numbers item-wise (the seed-replicate test's shape: 175 scored / 164 correct after the ceiling, 23 unresolved)
    r = runs[(GEMMA, COMPOSED)]
    tn = r["extra"]["three_numbers"]
    assert r["token_cap_tokens"] == G.CEILING and r["token_cap_rung_index"] == 1 and r["extra"]["escalation_state"] == "final_at_ceiling" and r["n_items_judged"] == 175
    assert (tn["n_scored_final"], tn["correct_final"], tn["n_unresolved_neither_hit_nor_miss"], tn["n_loops"], tn["n_at_cap_ceiling"], tn["n_unscored_below_cap"]) == (175, 164, 23, 22, 1, 2)
    assert tn["producer_record"] is None and tn["producer_record_version"] == G.RECORD_VERSION_NONE and tn["n_attempted"] == 200
    # llama's: no ceiling store, 1 truncated item never escalated (below threshold) + 2 loops unresolved
    r = runs[(LLAMA, COMPOSED)]
    assert r["token_cap_tokens"] == G.CAP and r["extra"]["escalation_state"] == "final_at_base" and r["extra"]["at_cap_classes"] == {"loop": 2, "ceiling": 0, "not_escalated": 1}
    assert r["extra"]["oos_tuple"] == {"8192": None, "16384": None}       # the no-switch fitted row carries no tuple
    # nemotron's constitution ReLU: resolved through the roster / pins too
    r = runs[(NEMO, CONSTITUTION)]
    assert r["extra"]["object_attestation"]["roster_staging"] == {"staged_id": STAGED[(NEMO, CONSTITUTION)], "selection": "main", "pins": LC.rel_home(fx["pins"])}
    assert r["extra"]["object_staged_id"] == STAGED[(NEMO, CONSTITUTION)] and runs[(NEMO, COMPOSED)]["extra"]["object_attestation"]["roster_staging"] is None
    assert runs[(NEMO, COMPOSED)]["extra"]["object_staged_id"] == STAGED[(NEMO, COMPOSED)]      # from the registry's row (campaign_cell_id_legacy)
    assert r["extra"]["oos_tuple"] == {"8192": OOS[NEMO].format(cap=G.CAP), "16384": OOS[NEMO].format(cap=G.CEILING)}
    assert all(r["extra"]["coexist_ruling"]["ruling_id"] == P.COEXIST_RULING_ID and r["extra"]["coexist_ruling"]["matches_object_prefix"] is False for r in runs.values())


def test_the_same_store_under_another_root_mints_the_same_ids(tmp_path, seeds):
    fx1 = _mirror(tmp_path, seeds, root=str(tmp_path / "a" / "gsm8k_paired_allmodels"))
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "b" / "elsewhere"), head=True)
    conn = _conn()
    bt1, _f1, _g1 = _read(_adapter(fx1, seeds), NEMO, conn=conn)
    bt2, _f2, _g2 = _read(_adapter(fx2, seeds), NEMO, conn=conn)
    ids1 = {r["campaign_cell_id_legacy"]: (r["run_id"], r["instrument_id"]) for r in bt1["run"]}
    ids2 = {r["campaign_cell_id_legacy"]: (r["run_id"], r["instrument_id"]) for r in bt2["run"]}
    assert ids1 == ids2 and set(ids1) == {COMPOSED, CONSTITUTION}


def test_the_in_frame_unsteered_reference_is_derived_by_identity_from_the_registry_rows(tmp_path, seeds):
    """The model's unsteered GSM8K runs (run.model_id / eval_id gsm8k / object_id none) flagged in the frame are named on every metric's parked block and
    on the run; a steered run, another model's run and an unsteered run outside the frame are not; the query runs once per pass."""
    fx = _mirror(tmp_path, seeds, pairs=[(GEMMA, COMPOSED), (NEMO, COMPOSED)])
    inst = "0" * 64
    runs = [("sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000001", "gemma4e4b_it", "none", "gsm8k", "gemma4e4b", "evalcheck/gsm8k_paired200", inst, 8192, "on", "ingested"),
            ("sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000002", "gemma4e4b_it", "none", "gsm8k", "gemma4e4b_it", "evalcheck/gsm8k_paired200", inst, 8192, "on", "ingested"),
            ("sdfbase_full22:gsm8k:bbbbbbbbbbbb:0000000000000003", "gemma4e4b_it", "none", "gsm8k", "gemma4e4b_it", "evalcheck/gsm8k_paired200", inst, 8192, "on", "superseded"),
            ("sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000004", "gemma4e4b_it", OBJ[(GEMMA, COMPOSED)], "gsm8k", "gemma4e4b_it", "other", inst, 8192, "on", "ingested"),
            ("sdfbase_full22:mmlu:aaaaaaaaaaaa:0000000000000005", "gemma4e4b_it", "none", "mmlu", "gemma4e4b_it", "other", inst, 1, "on", "ingested"),
            ("sdfbase_full22:gsm8k:cccccccccccc:0000000000000006", "nemotron3nano4b", "none", "gsm8k", "nemotron3nano4b", "evalcheck/gsm8k_paired200", inst, 8192, "on", "ingested")]
    frame = {"sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000001": (1, "selected"), "sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000002": (1, "selected"),
             "sdfbase_full22:gsm8k:bbbbbbbbbbbb:0000000000000003": (0, "superseded"), "sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000004": (1, "selected"),
             "sdfbase_full22:gsm8k:cccccccccccc:0000000000000006": (0, "not_in_roster")}
    conn = _conn(frame=frame, runs=runs)
    ad = _adapter(fx, seeds)
    bt, facts, _ = _read(ad, GEMMA, conn=conn)
    r = _run(bt, COMPOSED)
    ref = r["extra"]["in_frame_unsteered_reference"]
    assert ref["state"] == "derived" and ref["in_frame_unsteered_run_ids"] == ["sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000001", "sdfbase_full22:gsm8k:aaaaaaaaaaaa:0000000000000002"]
    assert ref["n_candidates"] == 3 and ref["why"] is None and ref["anchor_run_id"] is None and {a["selection_state"] for a in ref["anchors"]} == {"selected"}
    assert {a["view_legacy"] for a in ref["anchors"]} == {"gemma4e4b", "gemma4e4b_it"} and ref["derived_at"] == NOW
    value, bound = _metrics(bt, r["run_id"])
    assert value["extra"]["delta_block"]["anchor"]["in_frame_unsteered_run_ids"] == ref["in_frame_unsteered_run_ids"] == bound["extra"]["delta_block"]["anchor"]["in_frame_unsteered_run_ids"]
    assert value["extra"]["delta_block"]["delta_method"] == f"not computed: {P.ANCHOR_STATE}" and "run_contrast" not in bt
    assert facts["in_frame_unsteered_reference"]["in_frame_unsteered_run_ids"] == ref["in_frame_unsteered_run_ids"]
    # nemotron's only unsteered run is outside the frame: derived, empty, naming why
    bt2, facts2, _ = _read(ad, NEMO, conn=conn)
    ref2 = _run(bt2, COMPOSED)["extra"]["in_frame_unsteered_reference"]
    assert ref2["state"] == "derived" and ref2["in_frame_unsteered_run_ids"] == [] and ref2["n_candidates"] == 1 and "no in-frame unsteered GSM8K run" in ref2["why"]
    # one pass, one derivation: the cache holds every document model
    assert set(ad.in_frame_unsteered(conn, NOW)) == {"gemma4e4b_it", "llama8b_it", "nemotron3nano4b"}
    # a registry without the frame flag (the object rows alone) parks the reference unresolved, never a refusal
    bt3, facts3, _ = _read(_adapter(fx, seeds), GEMMA, conn=_conn())
    ref3 = _run(bt3, COMPOSED)["extra"]["in_frame_unsteered_reference"]
    assert ref3["state"] == "unresolved" and ref3["in_frame_unsteered_run_ids"] == [] and facts3["n_refusals"] == 0


# --------------------------------------------------------------------------------------------
# refusals by name (the pair, never the tag)
# --------------------------------------------------------------------------------------------

def test_a_none_object_dir_and_an_undeclared_key_are_refused_by_name_and_the_declared_pairs_file(tmp_path, seeds):
    other = hashlib.sha256(b"synthetic gemma lora").hexdigest()
    fx = _mirror(tmp_path, seeds, pairs=[(GEMMA, COMPOSED)], extra_dirs=[(GEMMA, "none", "thinking_card", "none", None), (GEMMA, "constv2-mix-full--lora_r4_ce", "thinking_card", "constv2-mix-full__lora_r4_ce", other)])
    bt, facts, _ = _read(_adapter(fx, seeds), GEMMA, conn=_conn())
    assert "declares no unsteered run" in _refused(facts, GEMMA, "none") and "collapse key" in _refused(facts, GEMMA, "none")
    assert "is not declared on gemma4e4b_it" in _refused(facts, GEMMA, "constv2-mix-full__lora_r4_ce")
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {COMPOSED} and facts["n_refusals"] == 2


def test_an_undeclared_tag_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pairs=[(NEMO, COMPOSED)])
    # a fourth standup under the root: listed, read, refused by name (never silently skipped)
    spec = _spec(**SPECS[(NEMO, COMPOSED)])
    _store_files(os.path.join(fx["root"], "qwen35_4b_it", COMPOSED.replace("__", "--", 1), "thinking_card"), NEMO, COMPOSED, COMPOSED.replace("__", "--", 1), G.CAP, dict(spec["states"]),
                 obj_sha=OBJ[(NEMO, COMPOSED)], tree=TREE, job="600950", ihash8="eeeeeeee", chash8="dddd0000", spec=spec)
    ad = _adapter(fx, seeds)
    bt, facts, groups = _read(ad, "qwen35_4b_it", conn=_conn())
    assert groups == [NEMO, "qwen35_4b_it"] and "run" not in bt and facts["n_refusals"] == 1 and "not one of the document's standups" in facts["refusals"][0]


def test_a_store_under_the_other_models_sampling_directory_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pairs=[(LLAMA, COMPOSED), (NEMO, COMPOSED)], dir_over={(LLAMA, COMPOSED): "thinking_card"})
    bt, facts, _ = _read(_adapter(fx, seeds), LLAMA, conn=_conn())
    why = _refused(facts, LLAMA, COMPOSED)
    assert "['thinking_card']" in why and "card / card__rung16384" in why and "run" not in bt


def test_an_object_absent_from_the_registry_or_of_another_kind_or_without_a_connection_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pairs=[(NEMO, COMPOSED), (NEMO, CONSTITUTION)])
    ad = _adapter(fx, seeds)
    # the composed object's row is absent: the pair refuses, the constitution pair files
    bt, facts, _ = _read(ad, NEMO, conn=_conn(objects={(NEMO, CONSTITUTION): OBJ[(NEMO, CONSTITUTION)]}))
    assert "is not in the registry's object table" in _refused(facts, NEMO, COMPOSED) and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {CONSTITUTION}
    # the row is there under another key's kind
    conn = _conn(objects={(NEMO, CONSTITUTION): OBJ[(NEMO, CONSTITUTION)]})
    conn.execute("INSERT INTO object VALUES (?, 'relu_steer', 'constitution_relu', 'constitution', 'nemotron3nano4b', 'none', 'approved', 'x')", (OBJ[(NEMO, COMPOSED)],))
    bt2, facts2, _ = _read(_adapter(fx, seeds), NEMO, conn=conn)
    assert "kind 'relu_steer'" in _refused(facts2, NEMO, COMPOSED) and "spells 'composed'" in _refused(facts2, NEMO, COMPOSED) and len(bt2["run"]) == 1
    # no connection at all: nothing can be verified, every pair refuses by name
    bt3, facts3, _ = _read(_adapter(fx, seeds), NEMO, conn=None)
    assert facts3["n_refusals"] == 2 and all("no registry connection" in x for x in facts3["refusals"]) and "run" not in bt3


def test_an_attestation_that_disagrees_or_a_roster_pin_that_disagrees_is_refused_by_name(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = _mirror(tmp_path, seeds, pairs=[(GEMMA, COMPOSED), (NEMO, CONSTITUTION)], store_over={(GEMMA, COMPOSED): {"prov_sha": other}})
    bt, facts, _ = _read(_adapter(fx, seeds), GEMMA, conn=_conn())
    assert "name two objects" in _refused(facts, GEMMA, COMPOSED) and "run" not in bt
    # the fit provenance's 16-hex prefix names another object
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "m2"), pairs=[(GEMMA, COMPOSED)], store_over={(GEMMA, COMPOSED): {"prov_sha": other, "agg_sha": other}})
    _bt2, facts2, _ = _read(_adapter(fx2, seeds), GEMMA, conn=_conn())
    assert "fit_provenance.object_sha" in _refused(facts2, GEMMA, COMPOSED) and "no prefix of the attested object" in _refused(facts2, GEMMA, COMPOSED)
    # the roster stages nemotron's constitution key but the pin names another sha
    _write(fx["pins"], {"nemotron3nano4b": {STAGED[(NEMO, CONSTITUTION)]: {"object_sha256": other}}})
    _bt3, facts3, _ = _read(_adapter(fx, seeds), NEMO, conn=_conn())
    assert "the roster / pins name object" in _refused(facts3, NEMO, CONSTITUTION) and other[:16] in _refused(facts3, NEMO, CONSTITUTION)


def test_a_replicate_seed_block_a_short_outcome_record_and_a_fit_provenance_off_the_seeds_are_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pairs=[(NEMO, COMPOSED)], store_over={(NEMO, COMPOSED): {"seed_block": {"policy": "replicate", "seed_base": 1}}})
    _bt, facts, _ = _read(_adapter(fx, seeds), NEMO, conn=_conn())
    assert "replicate policy" in _refused(facts, NEMO, COMPOSED) and "seed-replicate document" in _refused(facts, NEMO, COMPOSED)
    short = _spec(**dict(SPECS[(NEMO, COMPOSED)], n_correct=170), n_items=199)
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "m2"), pairs=[(NEMO, COMPOSED)], specs={(NEMO, COMPOSED): short})
    _bt2, facts2, _ = _read(_adapter(fx2, seeds), NEMO, conn=_conn())
    assert "holds 199 items, not 200" in _refused(facts2, NEMO, COMPOSED)
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "m3"), pairs=[(NEMO, COMPOSED)], store_over={(NEMO, COMPOSED): {"fit_hf": "acme/no-such-checkpoint"}})
    _bt3, facts3, _ = _read(_adapter(fx3, seeds), NEMO, conn=_conn())
    assert "fit_provenance.model_id 'acme/no-such-checkpoint' is no model of models.yaml" in _refused(facts3, NEMO, COMPOSED)
    fx4 = _mirror(tmp_path, seeds, root=str(tmp_path / "m4"), pairs=[(NEMO, COMPOSED)], store_over={(NEMO, COMPOSED): {"transfer": True}})
    _bt4, facts4, _ = _read(_adapter(fx4, seeds), NEMO, conn=_conn())
    assert "fit_provenance.transfer True disagrees" in _refused(facts4, NEMO, COMPOSED)


def test_a_base_manifest_that_declares_an_item_subset_sets_the_count(tmp_path, seeds):
    n = 150
    spec = _spec(n_correct=120, n_wrong=20, loops=7, truncated=3, below_cap=0, rung=None, n_items=n)
    subset = {"cap": G.CAP, "ids": sorted(spec["states"]), "n": n, "source_cap": G.CAP, "task": G.TASK}
    fx = _mirror(tmp_path, seeds, pairs=[(NEMO, COMPOSED)], specs={(NEMO, COMPOSED): spec}, store_over={(NEMO, COMPOSED): {"item_subset_base": subset}})
    bt, facts, _ = _read(_adapter(fx, seeds), NEMO, conn=_conn())
    assert facts["n_refusals"] == 0, facts["refusals"]
    r = _run(bt, COMPOSED)
    assert r["n_items_planned"] == n and r["extra"]["n_items_declared_by_manifest"] == n and r["extra"]["three_numbers"]["n_attempted"] == n
    value, bound = _metrics(bt, r["run_id"])
    assert (value["k"], value["n"]) == (120, 140) and (bound["k"], bound["n"], bound["ci_lo"], bound["ci_hi"]) == (120, n, 120 / n, 130 / n)
    assert len([i for i in bt["item_result"] if i["run_id"] == r["run_id"]]) == n


# --------------------------------------------------------------------------------------------
# the optional bindings: ReLU's staging manifest, the producer's declaration record
# --------------------------------------------------------------------------------------------

def _staging(tmp_path, tag: str, sha: str, *, kind: str = P.STAGING_KIND, fit_hf: str = None) -> str:
    m = MODELS[tag]
    d = str(tmp_path / "staging")
    _write(os.path.join(d, P.STAGING_MANIFEST_NAME.format(model_key=m["model_key"])),
           {"_schema": P.STAGING_SCHEMA, "_what": "synthetic", "_written_at": "2026-09-19T20:50:00+00:00",
            "object": {"staged_id": STAGED[(tag, COMPOSED)], "bank": "relu_paired", "view": f"paired_d1_{m['model_key']}_s1", "seed": 1, "cell_id": COMPOSED, "kind": kind,
                       "object_sha256": sha, "dose": 1.0, "fit_model": {"id": fit_hf or m["fit_hf"], "revision": "1" * 40}, "served_model": {"id": m["served_hf"], "revision": REVISION[tag]},
                       "parents": {"pos": {"object_sha256": hashlib.sha256(b"pos").hexdigest()}, "neg": {"object_sha256": hashlib.sha256(b"neg").hexdigest()}}}})
    return d


def test_relu_staging_manifest_binds_when_present_and_a_disagreement_refuses(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pairs=[(GEMMA, COMPOSED)])
    stg = _staging(tmp_path, GEMMA, OBJ[(GEMMA, COMPOSED)])
    bt, facts, _ = _read(_adapter(fx, seeds, staging_dir=stg), GEMMA, conn=_conn())
    assert facts["n_refusals"] == 0, facts["refusals"]
    r = _run(bt, COMPOSED)
    att = r["extra"]["object_attestation"]
    assert att["staging_manifest"]["staged_id"] == STAGED[(GEMMA, COMPOSED)] and att["staging_manifest"]["kind"] == P.STAGING_KIND and att["staging_manifest"]["fit_model"]["id"] == MODELS[GEMMA]["fit_hf"]
    assert set(att["staging_manifest"]["parents"]) == {"pos", "neg"} and "staging manifest" in r["field_confidence_json"]["object_id"]["source"] and facts["pairs"][COMPOSED]["staging_manifest_bound"] is True
    # the manifest stages another object
    stg2 = _staging(tmp_path / "s2", GEMMA, hashlib.sha256(b"another composed").hexdigest())
    _bt2, facts2, _ = _read(_adapter(fx, seeds, staging_dir=stg2), GEMMA, conn=_conn())
    assert "staging manifest" in _refused(facts2, GEMMA, COMPOSED) and "the store attests" in _refused(facts2, GEMMA, COMPOSED)
    # the manifest fits on another checkpoint than the store's fit provenance
    stg3 = _staging(tmp_path / "s3", GEMMA, OBJ[(GEMMA, COMPOSED)], fit_hf="google/gemma-4-E4B-it")
    _bt3, facts3, _ = _read(_adapter(fx, seeds, staging_dir=stg3), GEMMA, conn=_conn())
    assert "fits on 'google/gemma-4-E4B-it'" in _refused(facts3, GEMMA, COMPOSED)
    # absent: recorded, never a refusal
    bt4, facts4, _ = _read(_adapter(fx, seeds, staging_dir=str(tmp_path / "nowhere")), GEMMA, conn=_conn())
    assert facts4["n_refusals"] == 0 and _run(bt4, COMPOSED)["extra"]["object_attestation"]["staging_manifest"] is None


def _declaration(tmp_path, rows, *, schema: str = P.DECLARATION_SCHEMA, role: str = P.ROLE) -> str:
    return _write(str(tmp_path / "paired_allmodels_rows.json"), {"schema": schema, "generated_by": "synthetic", "wave": "paired_allmodels", "role": role, "rows": rows})


def _decl_row(tag: str, okey: str, index: int = 0, **over) -> dict:
    m = MODELS[tag]
    slug = okey.replace("__", "--", 1)
    base_dir, ceil_dir = P.store_dirs_for(m["thinking"])
    row = {"index": index, "job_id": f"g8k-{m['model_key']}-{m['panel']}-{okey}", "model_key": m["model_key"], "view": m["panel"], "panel_view": m["panel"], "tag": tag, "cell": okey,
           "cell_slug": slug, "harness_cell": okey, "fitted": True, "route_kind": "lens", "thinking_mode": m["thinking"], "sampling_mode": base_dir, "sampling": dict(m["sampling"], source="synthetic"),
           "out_root": f"{P.CONTROLS_HEAD}/{tag}/{slug}/{base_dir}", "rung_out_root": f"{P.CONTROLS_HEAD}/{tag}/{slug}/{ceil_dir}", "object_sha256": OBJ[(tag, okey)], "staged_id": STAGED[(tag, okey)],
           "tuple_names": ({str(G.CAP): OOS[tag].format(cap=G.CAP), str(G.CEILING): OOS[tag].format(cap=G.CEILING)} if tag in OOS else {})}
    row.update(expected_refusal=False, expected_refusal_reason=None, held_reason=None)
    row.update(over)
    return row


def test_the_declaration_record_binds_when_present_and_a_disagreement_or_a_missing_row_refuses(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pairs=[(NEMO, COMPOSED), (NEMO, CONSTITUTION)])
    decl = _declaration(tmp_path, [_decl_row(NEMO, COMPOSED, 0), _decl_row(NEMO, CONSTITUTION, 1)])
    bt, facts, _ = _read(_adapter(fx, seeds, declaration_path=decl), NEMO, conn=_conn())
    assert facts["n_refusals"] == 0 and facts["declaration_state"] == "present", facts["refusals"]
    r = _run(bt, CONSTITUTION)
    assert r["extra"]["declaration"]["index"] == 1 and r["extra"]["declaration"]["_schema"] == P.DECLARATION_SCHEMA and r["extra"]["declaration_state"] == "present"
    # a row whose out root is another store's
    decl2 = _declaration(tmp_path / "d2", [_decl_row(NEMO, COMPOSED, 0, out_root="controls_paired_allmodels/nemotron3nano4b/other/thinking_card"), _decl_row(NEMO, CONSTITUTION, 1)])
    bt2, facts2, _ = _read(_adapter(fx, seeds, declaration_path=decl2), NEMO, conn=_conn())
    assert "out_root" in _refused(facts2, NEMO, COMPOSED) and {x["campaign_cell_id_legacy"] for x in bt2["run"]} == {CONSTITUTION}
    # a present record that names no row for a store
    decl3 = _declaration(tmp_path / "d3", [_decl_row(NEMO, COMPOSED, 0)])
    _bt3, facts3, _ = _read(_adapter(fx, seeds, declaration_path=decl3), NEMO, conn=_conn())
    assert "names 0 row(s) for tag nemotron3nano4b" in _refused(facts3, NEMO, CONSTITUTION)
    # another role: every pair refuses naming it
    decl4 = _declaration(tmp_path / "d4", [_decl_row(NEMO, COMPOSED, 0), _decl_row(NEMO, CONSTITUTION, 1)], role="arditi_unpaired_value")
    _bt4, facts4, _ = _read(_adapter(fx, seeds, declaration_path=decl4), NEMO, conn=_conn())
    assert facts4["n_refusals"] == 2 and all("declares role 'arditi_unpaired_value'" in x for x in facts4["refusals"])
    # absent: the runs say so
    bt5, facts5, _ = _read(_adapter(fx, seeds), NEMO, conn=_conn())
    assert facts5["declaration_state"].startswith("absent") and _run(bt5, COMPOSED)["extra"]["declaration"] is None


# --------------------------------------------------------------------------------------------
# the producer's records at the mirror root: store pins, the three-number record, the final-state record
# --------------------------------------------------------------------------------------------

def test_the_producers_records_bind_when_present_and_a_disagreement_refuses_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, head=True, records=True)
    ad = _adapter(fx, seeds)
    conn = _conn()
    inputs = ad.discover()
    names = sorted(os.path.basename(i.uri) for i in inputs)
    assert names.count(f"{P.PINS_FILE}#{NEMO}") == 1 and names.count(f"{P.THREE_NUMBERS_FILE}#{GEMMA}") == 1 and names.count(f"{P.FINAL_STATE_FILE}#{LLAMA}") == 1
    assert {ad.comparison_group(i) for i in inputs} == {GEMMA, LLAMA, NEMO}
    for tag in (GEMMA, LLAMA, NEMO):
        bt, facts, _ = _read(ad, tag, conn=conn)
        assert facts["n_refusals"] == 0, facts["refusals"]
        assert facts["producer_records"]["store_pins"].endswith(P.PINS_FILE) and facts["producer_records"]["absent_or_malformed"] == {}
        for okey in P.DOCUMENT[tag][2]:
            r = _run(bt, okey)
            tn = r["extra"]["three_numbers"]
            assert tn["producer_record_version"] == P.RECORD_VERSION_PRODUCER and tn["producer_record"].endswith(P.THREE_NUMBERS_FILE) and tn["producer_record_agrees"] is True
            assert r["extra"]["store_pins"] == {"8192": TREE, "16384": (TREE if r["extra"]["store_key_ceiling"] else None)} and r["extra"]["store_pins_record"].endswith(P.PINS_FILE)
            assert "the store pins agree" in r["extra"]["tree_attestation"] and tn["producer_off_task"] is not None
            _value, bound = _metrics(bt, r["run_id"])
            assert bound["producer_metric_spelling"] == "bound_correct_over_attempted" and bound["source_uri"].endswith(P.THREE_NUMBERS_FILE)
            assert bound["extra"]["producer_spellings"] == {"bound_correct_over_attempted": round(bound["value"], 4)}
            assert facts["pairs"][okey]["store_pins_bound"] and facts["pairs"][okey]["three_numbers_bound"] and facts["pairs"][okey]["final_state_bound"]
    # the source_state rows name the pins' members per tag
    rows = ad.source_states(_ctx(conn), [i for i in inputs if ad.comparison_group(i) == NEMO])
    pins_rows = [r for r in rows if r.json_path == "$.pins"]
    assert len(pins_rows) == 1 and sorted(pins_rows[0].current_members) == sorted([f"nemotron3nano4b/main/{COMPOSED}@8192", f"nemotron3nano4b/main/{CONSTITUTION}@16384", f"nemotron3nano4b/main/{CONSTITUTION}@8192"])
    # a three-number row that disagrees refuses the pair naming the number; the tag's other pair files
    def bump(three):
        three["rows"][f"{NEMO}/{COMPOSED.replace('__', '--', 1)}"]["third_state"]["loops"] += 1
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "m2"), records=True, three_over=bump)
    bt2, facts2, _ = _read(_adapter(fx2, seeds), NEMO, conn=conn)
    assert "third_state.loops: producer 15, re-derived 14" in _refused(facts2, NEMO, COMPOSED) and {x["campaign_cell_id_legacy"] for x in bt2["run"]} == {CONSTITUTION}
    # a store pin that is not the manifest's tree refuses; a ceiling store present but unpinned refuses (gemma's synthetic pair is the one with a ceiling store)
    def wrong_pin(pins):
        pins[P.Gsm8kPairedAllmodelsAdapter._pin_key("gemma4e4b", "it", COMPOSED, G.CAP)] = "f" * 40
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "m3"), records=True, pins_over=wrong_pin)
    _bt3, facts3, _ = _read(_adapter(fx3, seeds), GEMMA, conn=conn)
    assert "store pin" in _refused(facts3, GEMMA, COMPOSED) and "ffffffffffff" in _refused(facts3, GEMMA, COMPOSED)
    def unpinned_ceiling(pins):
        del pins[P.Gsm8kPairedAllmodelsAdapter._pin_key("gemma4e4b", "it", COMPOSED, G.CEILING)]
    fx4 = _mirror(tmp_path, seeds, root=str(tmp_path / "m4"), records=True, pins_over=unpinned_ceiling)
    _bt4, facts4, _ = _read(_adapter(fx4, seeds), GEMMA, conn=conn)
    assert "ceiling store is present but its store pin" in _refused(facts4, GEMMA, COMPOSED)
    # a final-state row that disagrees refuses naming the count
    def bump_fs(fs):
        fs[f"{GEMMA}/{COMPOSED.replace('__', '--', 1)}"]["n_unresolved_final"] += 1
    fx5 = _mirror(tmp_path, seeds, root=str(tmp_path / "m5"), records=True, fstate_over=bump_fs)
    _bt5, facts5, _ = _read(_adapter(fx5, seeds), GEMMA, conn=conn)
    assert P.FINAL_STATE_FILE in _refused(facts5, GEMMA, COMPOSED) and "n_unresolved_final: producer 24, re-derived 23" in _refused(facts5, GEMMA, COMPOSED)
    # a present record with another schema refuses the whole tag by name (nothing can be cross-checked)
    def wrong_schema(three):
        three["schema"] = "g8k_three_numbers/1"
    fx6 = _mirror(tmp_path, seeds, root=str(tmp_path / "m6"), pairs=[(GEMMA, COMPOSED)], records=True, three_over=wrong_schema)
    bt6, facts6, _ = _read(_adapter(fx6, seeds), GEMMA, conn=conn)
    assert any(x.startswith(f"{GEMMA}|* -- ") and P.THREE_NUMBERS_SCHEMA in x for x in facts6["refusals"]) and len(bt6.get("run", [])) == 1


# --------------------------------------------------------------------------------------------
# a leg the driver refused files as a REFUSAL STATE (a not_served disposition), never as nothing
# --------------------------------------------------------------------------------------------

def test_a_driver_failed_leg_files_a_not_served_disposition_spelling_thinking_unservable_predicate_a(tmp_path, seeds):
    """The live shape (nemotron3nano4b x the constitution ReLU): RESULT `rc=2 driver_rc=1 result=DRIVER_FAILED`, ladder_state.json accepting no
    attempt, the render probe's ON draws all `unclosed_block`. No run, no metric, no item; ONE disposition (state not_served) whose reason and
    extra.refusal_state spell `thinking_unservable_predicate_a`, carrying the RESULT line, the failed attempt, the evidence path and every draw's class,
    the store pins' refused entry and the declaration row when present; the sibling pair files."""
    fx = _mirror(tmp_path, seeds, head=True, pairs=[(NEMO, COMPOSED)], refused=[(NEMO, CONSTITUTION)], records=True)
    decl = _declaration(tmp_path, [_decl_row(NEMO, COMPOSED, 0), _decl_row(NEMO, CONSTITUTION, 1, expected_refusal=False, held_reason=None)])
    conn = _conn()
    bt, facts, _ = _read(_adapter(fx, seeds, declaration_path=decl), NEMO, conn=conn)
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {COMPOSED} and len(bt["disposition"]) == 1
    d = bt["disposition"][0]
    assert d["state"] == "not_served" and d["reason"].startswith(f"{P.REFUSAL_STATE_THINKING_UNSERVABLE_A}: ") and REFUSED_RESULT in d["reason"]
    assert d["model_id"] == "nemotron3nano4b" and d["object_id"] == OBJ[(NEMO, CONSTITUTION)] and d["eval_id"] == "gsm8k" and d["round_id"] == "r2" and d["roster_entry_id"] is None
    assert d["comparison_id"] == f"{P.SOURCE}/{NEMO}/{CONSTITUTION}" and d["source_ref"].endswith("RESULT") and d["ruled_utc"] == "2026-09-22T05:28:09.000Z"
    x = d["extra"]
    assert x["refusal_state"] == P.REFUSAL_STATE_THINKING_UNSERVABLE_A == "thinking_unservable_predicate_a" and x["result_line"] == REFUSED_RESULT
    assert x["failed_attempt"]["job_id"] == "596511" and x["failed_attempt"]["node"] == "gpu-20" and x["failed_attempt"]["result"] == "DRIVER_FAILED" and x["tree_sha"] == TREE
    assert x["evidence_uri"].endswith("evidence.json") and x["render_probe_classes"] == ["unclosed_block"] * 3 and len(x["render_probe_on_attempts"]) == 3
    assert x["render_certified"]["think_shape_certified"] is False and x["render_certified"]["certification_problems"]
    assert x["store_pins_refused"]["key"] == f"nemotron3nano4b/main/{CONSTITUTION}@8192" and x["store_pins_refused"]["RESULT"] == REFUSED_RESULT
    assert x["declaration"]["index"] == 1 and x["declaration"]["expected_refusal"] is False and x["object_kind"] == "relu_steer" and x["paper_label"] == "constitution_relu"
    assert x["view_legacy"] == NEMO and x["campaign_cell_id_legacy"] == CONSTITUTION and x["lane_source_legacy"] == P.SOURCE and x["declared_role"] == P.ROLE
    assert "predicate A" in x["refusal_rule"] and x["coexist_ruling"]["matches_object_prefix"] is False
    assert facts["refused_legs"][CONSTITUTION]["refusal_state"] == P.REFUSAL_STATE_THINKING_UNSERVABLE_A and facts["refused_legs"][CONSTITUTION]["pins_refused_bound"] is True
    # the same store under another root mints the same disposition id (keyed on the source, model, tag, key and state -- never the path)
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "elsewhere" / "mirror"), pairs=[(NEMO, COMPOSED)], refused=[(NEMO, CONSTITUTION)])
    bt2, _f2, _ = _read(_adapter(fx2, seeds), NEMO, conn=conn)
    assert bt2["disposition"][0]["disposition_id"] == d["disposition_id"] and bt2["disposition"][0]["extra"]["store_pins_refused"] is None
    # a DRIVER_FAILED leg whose probe names no unclosed block is still a refusal state -- spelled driver_failed_unclassified, never null
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "m3"), pairs=[(NEMO, COMPOSED)])
    _refused_store(os.path.join(fx3["root"], NEMO, CONSTITUTION.replace("__", "--", 1), "thinking_card"), NEMO, CONSTITUTION, CONSTITUTION.replace("__", "--", 1), OBJ[(NEMO, CONSTITUTION)],
                   probe_classes=("ok", "unclosed_block", "ok"))
    bt3, facts3, _ = _read(_adapter(fx3, seeds), NEMO, conn=conn)
    assert facts3["n_refusals"] == 0 and bt3["disposition"][0]["extra"]["refusal_state"] == P.REFUSAL_STATE_DRIVER_FAILED and bt3["disposition"][0]["state"] == "not_served"
    # without a registry connection the disposition still files, naming no object and saying why
    bt4, facts4, _ = _read(_adapter(fx2, seeds), NEMO, conn=None)
    d4 = [x for x in bt4["disposition"]]
    assert len(d4) == 1 and d4[0]["object_id"] is None and "no registry connection" in d4[0]["extra"]["object_resolution"] and "run" not in bt4      # the composed pair refuses without a connection
    # a store pin that names the refused leg as COMPLETED, or a refused record whose RESULT is not the store's, refuses by name
    def completed_pin(pins):
        pins[P.Gsm8kPairedAllmodelsAdapter._pin_key("nemotron3nano4b", "main", CONSTITUTION, G.CAP)] = TREE
    fx5 = _mirror(tmp_path, seeds, root=str(tmp_path / "m5"), pairs=[(NEMO, COMPOSED)], refused=[(NEMO, CONSTITUTION)], records=True, pins_over=completed_pin)
    bt5, facts5, _ = _read(_adapter(fx5, seeds), NEMO, conn=conn)
    assert "carry a completed pin" in _refused(facts5, NEMO, CONSTITUTION) and "disposition" not in bt5
    # a ladder_state.json that ACCEPTS an attempt beside a DRIVER_FAILED RESULT is a disagreement, refused by name
    fx6 = _mirror(tmp_path, seeds, root=str(tmp_path / "m6"), pairs=[(NEMO, COMPOSED)])
    _refused_store(os.path.join(fx6["root"], NEMO, CONSTITUTION.replace("__", "--", 1), "thinking_card"), NEMO, CONSTITUTION, CONSTITUTION.replace("__", "--", 1), OBJ[(NEMO, CONSTITUTION)], accepted="a1")
    _bt6, facts6, _ = _read(_adapter(fx6, seeds), NEMO, conn=conn)
    assert "accepts attempt 'a1'" in _refused(facts6, NEMO, CONSTITUTION)


# --------------------------------------------------------------------------------------------
# end to end: the harvester's rule-scored completion
# --------------------------------------------------------------------------------------------

def _registry_root(tmp_path, seeds) -> str:
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for (tag, okey), sha in OBJ.items():
        g = LC.cell_grammar_legacy(okey)
        obj = LC.ResolvedObject(sha, g["object_kind"], g["paper_label"], g["object_role"], "recorded", "synthetic", staged_id=STAGED[(tag, okey)], derivation=g["derivation"], si_id=g["si_id"])
        r = LC.standin_object_record(obj, seeds, F.producer(), now, MODELS[tag]["fit_model_id"])
        assert r is not None, (tag, okey)
        recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def test_runs_reach_ingested_through_the_rule_scored_completion_with_nothing_blocked_and_no_contrast(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, head=True, pairs=[(GEMMA, COMPOSED), (LLAMA, COMPOSED), (NEMO, COMPOSED)], refused=[(NEMO, CONSTITUTION)], records=True)
    ad = _adapter(fx, seeds)
    root = _registry_root(tmp_path, seeds)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    # the refused leg landed as a not_served disposition beside the three runs; no run for it, nothing blocked
    disp = F.q(root, "SELECT state, reason, model_id, object_id, eval_id, json_extract(extra_json, '$.refusal_state') FROM disposition WHERE comparison_id = ?", (f"{P.SOURCE}/{NEMO}/{CONSTITUTION}",))
    assert len(disp) == 1 and disp[0][0] == "not_served" and disp[0][1].startswith("thinking_unservable_predicate_a: ") and disp[0][2:5] == ("nemotron3nano4b", OBJ[(NEMO, CONSTITUTION)], "gsm8k")
    assert disp[0][5] == "thinking_unservable_predicate_a"
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ? AND campaign_cell_id_legacy = ?", (P.SOURCE, CONSTITUTION)) == [(0,)]
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, lifecycle_state, trial_index, seed_base, n_items_judged, token_cap_tokens, thinking_mode, transfer_kind, judged_utc IS NOT NULL, "
                     "ingested_utc IS NOT NULL FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (P.SOURCE,))
    assert rows == [(GEMMA, COMPOSED, "ingested", 0, None, 175, G.CEILING, "on", "base_to_instruct", 1, 1),
                    (LLAMA, COMPOSED, "ingested", 0, None, 197, G.CAP, "not_available", "base_to_instruct", 1, 1),
                    (NEMO, COMPOSED, "ingested", 0, None, 186, G.CAP, "on", "chat_checkpoint", 1, 1)], rows
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE r.lane_source_legacy = ?", (P.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (P.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM item_result i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (P.SOURCE,)) == [(600,)]
    assert F.q(root, "SELECT COUNT(DISTINCT r.instrument_id) FROM run r WHERE r.lane_source_legacy = ?", (P.SOURCE,)) == [(3,)]
    m = F.q(root, "SELECT r.campaign_cell_id_legacy, rm.computation_id <> '', rm.value, rm.k, rm.n, json_extract(rm.extra_json, '$.delta_block.anchor.anchor_state') FROM run_metric rm "
                  "JOIN run r ON r.run_id = rm.run_id WHERE r.lane_source_legacy = ? AND r.view_legacy = ? AND rm.metric_id = 'acc' ORDER BY 2 DESC", (P.SOURCE, GEMMA))
    assert m == [(COMPOSED, 1, 164 / 175, 164, 175, P.ANCHOR_STATE), (COMPOSED, 0, 164 / 200, 164, 200, P.ANCHOR_STATE)], m
    dg = F.q(root, "SELECT d.measure, d.numerator, d.denominator FROM run_degeneration d JOIN run r ON r.run_id = d.run_id WHERE r.view_legacy = ? AND d.measure IN (?, ?, ?) ORDER BY 1",
             (GEMMA, G.RAW_AT_CAP_MEASURE, G.LOOP_MEASURE, G.COHERENT_MEASURE))
    assert dg == [(G.RAW_AT_CAP_MEASURE, 23, 200), (G.LOOP_MEASURE, 22, 200), (G.COHERENT_MEASURE, 1, 200)], dg
    # the in-frame reference read the registry's own frame flag (no unsteered GSM8K run in the scratch registry: derived, empty)
    ref = json.loads(F.q(root, "SELECT extra_json FROM run WHERE lane_source_legacy = ? AND view_legacy = ?", (P.SOURCE, GEMMA))[0][0])["in_frame_unsteered_reference"]
    assert ref["state"] == "derived" and ref["in_frame_unsteered_run_ids"] == [] and ref["n_candidates"] == 0
    # a second pass over unchanged inputs publishes nothing and the rows stay ingested
    rep2, _i3, _i4 = F.run_and_ingest(root, _adapter(fx, seeds), now="2026-09-22T07:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (P.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration and the document's constants
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_as_the_gsm8k_readers_second_source_with_its_own_root_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[P.NAME]()
    assert isinstance(inst, P.Gsm8kPairedAllmodelsAdapter) and isinstance(inst, G.Gsm8kSeedrepAdapter) and inst.covers_sources == (P.SOURCE,) and inst.owner == "Overflow"
    assert P.SOURCE == "gsm8k_paired_allmodels" and P.ROLE == "paired_allmodels_value" and P.ANCHOR_STATE == "pooled_in_frame_anchors"
    assert AB.emission_version_of(P.Gsm8kPairedAllmodelsAdapter) == "2" and AB.emission_version_of(G.Gsm8kSeedrepAdapter) == "4" and inst.rng_scope == "none"
    assert inst.roots == (P.ROOT_DEFAULT,) and P.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "gsm8k_paired_allmodels")) and P.ROOT_DEFAULT != G.ROOT_DEFAULT
    assert set(P.DOCUMENT) == {GEMMA, LLAMA, NEMO} and P.DOCUMENT[NEMO][2] == (COMPOSED, CONSTITUTION) and P.DOCUMENT[GEMMA][2] == P.DOCUMENT[LLAMA][2] == (COMPOSED,)
    assert P.INSTRUMENT_RULE.startswith("one instrument per MODEL") and P.INSTRUMENT_SEED == G.INSTRUMENT_SEED
    assert P.store_dirs_for("on") == ("thinking_card", "thinking_card__rung16384") and P.store_dirs_for("not_available") == ("card", "card__rung16384")
    assert P.COEXIST_OBJECT_PREFIXES == ("87863240", "5066333f", "ad3e1922", "81321803") and P.COEXIST_RULING_ID.startswith("r2v4-coexist-gsm8k-paired-allmodels")
    assert P.producer_key_for("gemma4e4b_it", "k") == "gemma4e4b_it/k" and G.Gsm8kSeedrepAdapter.THINKING_SPELLING == "on"
    assert P.DECLARATION_REL == ("configs", "sdfbase", "gsm8k", "paired_allmodels_rows.json") and P.DECLARATION_SCHEMA == "g8k_paired_allmodels_rows/1"
    assert P.SHARED_FILES == ("store_pins_paired_allmodels.json", "three_numbers_paired_allmodels.json", "final_state_paired_allmodels.json") and P.PINS_SCHEMA == G.PINS_SCHEMA
    assert P.THREE_NUMBERS_SCHEMA == "g8k_three_numbers_offtask/1" and P.REFUSAL_DISPOSITION_STATE == "not_served" and P.REFUSAL_STATE_THINKING_UNSERVABLE_A == "thinking_unservable_predicate_a"
    from sisteer.registry import vocab as V
    assert P.REFUSAL_DISPOSITION_STATE in V.DISPOSITIONS


def test_the_document_models_resolve_through_the_seeds_with_their_thinking_channel(seeds):
    for tag, m in MODELS.items():
        mv = seeds.model_for_view_legacy(tag)
        assert (mv["model_id"], mv["base"], mv["view_key"]) == (m["model_id"], m["model_key"], m["panel"])
        assert P.thinking_for(seeds, m["model_id"]) == m["thinking"]
        assert seeds.model_for_hf_id(m["served_hf"]) == m["model_id"] and seeds.model_for_hf_id(m["fit_hf"]) == m["fit_model_id"]
