"""The `gsm8k_seedrep` reader on a SYNTHETIC mirror shaped like the three-seed family's GSM8K stores (invented item ids and shas,
`$SHARED/...` paths with a stand-in user, no generation text): a base store at 8,192 and its rung store at 16,384, the producer's store pins and
three-number record written from the same synthetic truth, and a roster / pins naming the synthetic object.

What it pins:

* THE ITEM-WISE MERGE: the rung's items REPLACE the base's (a base item scored 0 at the cap re-drawn correct at the rung counts correct once), one
  run per (tag, object key) at the ladder's final state (`token_cap_tokens 16384`, rung index 1, `escalation_state final_at_ceiling`,
  `extra.rung_item_ids`), keyed by the BASE store's NAME (the same fixture under two roots mints one id);
* THE THREE STATES: a scored item's value; an item at the cap with no answer is `value NULL` + `at_cap_unresolved` with its class (`loop` never
  re-drawn, `ceiling` still at cap at 16,384, `not_escalated` truncated at 8,192 without a rung) — ALL THREE unresolved, neither hit nor miss (the
  producer's widened third state); an item the instrument left unscored below the cap is `value NULL` + `unscored` (a miss in the bound); a
  scored-at-cap item's value counts with `at_token_cap 1`;
* THE THREE NUMBERS: the VALUE row (`acc` = correct / scored, the merge computation's non-blank id, `n_items_judged` = scored), the BOUND row (`acc`
  = correct / 200, a blank computation id so the frame's metric role ranks it second, the interval `[correct, correct + unresolved] / 200` in
  ci_lo / ci_hi), and the run-level shares as run_degeneration measures (`at_token_cap_finish_reason` = the at-cap items at the final state,
  `rep_ratio_loop` = the loops, `truncation_only` = ceiling + not escalated);
* THE RECORD'S VERSIONS: the producer's three-number record is checked under its own definitions — the item-wise write (final-state keys,
  `n_unresolved_ladder_only` beside the widened count) and the mirror root's first write (base reading, unresolved = loops + ceiling) both
  pass, the reader filing the widened definition either way; a wrong `n_scored_final` on the new write refuses the pair naming it;
* REFUSED BY NAME (the pair, never the tag): a store pin that is not the manifest's tree; a three-number record disagreeing with the re-derivation;
  a ladder that declared `escalate` without a rung store; a replicate seed base that is not the tag's index; a rung subset that is not the base's
  eligible set; an attested object sha that is not the roster / pins';
* the unfitted keys file as decode-seed trials (`trial_index` = the seed index), the fitted key at trial 0; the prompted key resolves through
  si_texts and its `system_sha`;
* end to end: the run reaches `ingested` through the harvester's rule-scored completion (the bound row's n = 200 covers the item set) with no
  state event from the reader and nothing blocked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import gsm8k_seedrep as G
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import relu_paired as RP
from sisteer.registry.adapters import seedrep_cluster as SC
from sisteer.registry.adapters.base import AdapterContext

TAG = "qwen35_4b_s2"
SEED_N = 2
LORA = "constv2-mix-full__lora_r4_ce"
NONE = "none"
PROMPTED = "prompted__safety_refusal"
STAGED = f"base64s2__{LORA}"
OBJ = hashlib.sha256(b"synthetic gsm8k seedrep lora s2").hexdigest()
TREE = hashlib.sha1(b"synthetic gsm8k seedrep tree").hexdigest()
MODEL_HF = "Qwen/Qwen3.5-4B"
NOW = "2026-09-21T09:00:00.000Z"
ITEM_IDS = [f"gsm8k_{i:08x}" for i in range(G.N_ITEMS)]
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
SEED_RULE = "sha256(f'{seed_base}|inspect-replicate|{item_id}|{sample_idx}')[:4] big-endian mod 2^31-1; seed_base = 2"
DEFAULT_SEED_RULE = "sha256(item_id NUL sample_idx)[:4] big-endian mod 2^31-1"
ROUTING = "/workspace-vast/TESTUSER/exp/gsm8k_seedrep/serving/g8k_{tag}_{slug}_{cap}_{job}/gsm8k/routing_table.json"


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
# the synthetic truth of one (tag, object key) pair and the files written from it
# --------------------------------------------------------------------------------------------

def _spec(*, n_correct: int, n_wrong: int, loops: int, truncated: int, below_cap: int, scored_at_cap: int = 0, scored_loops: int = 0, rung=None) -> dict:
    """The per-item truth: `states[item] = (state, base value)` with state in scored | loop | truncated | scored_at_cap | below_cap; `rung` maps an
    eligible item to (rung state, rung value) with state in scored | ceiling | below_cap — None means no ceiling store (the exit must then not be
    `escalate`)."""
    states = {}
    ids = iter(ITEM_IDS)
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
    for _ in range(scored_loops):
        states[next(ids)] = ("scored_loop", 0.0)
    for _ in range(below_cap):
        states[next(ids)] = ("below_cap", None)
    assert len(states) == G.N_ITEMS
    eligible = [i for i, (st, _v) in states.items() if st in ("truncated", "scored_at_cap")]
    if rung is not None:
        rung = {eligible[k]: v for k, v in enumerate(rung)}
        assert len(rung) == len(eligible)
    return {"states": states, "eligible": eligible, "rung": rung}


def _outcome(iid: str, st: str, v, cap: int, seed: int) -> dict:
    at_cap = st in ("loop", "truncated", "scored_at_cap", "scored_loop", "ceiling")
    scored = st in ("scored", "scored_at_cap", "scored_loop")
    return {"item_id": iid, "at_cap": at_cap, "scored": scored, "value": (v if scored else None), "stop_reason": ("max_tokens" if at_cap else "stop"),
            "completion_tokens": (cap if at_cap else 1500), "seed": seed, "sample_idx": 0, "think_block_unterminated": (st in ("loop", "truncated", "below_cap", "ceiling")),
            "unparseable": False, "attempt": 1, "prompt_tokens": 2250}


def _store_files(d: str, tag: str, okey: str, slug: str, cap: int, items: dict, *, obj_sha, tree: str, job: str, ihash8: str, chash8: str, seed_n: int,
                 spec: dict, subset=None, started="2026-09-21T06:12:50Z", ended="2026-09-21T06:20:02Z", oos: str = None, prov_sha: str = None,
                 seed_policy: str = "replicate", fit_transfer: bool = None, attempts_before: tuple = (), node: str = "gpu-1") -> str:
    """One store directory (base or rung): the named files the reader opens. `items` maps item -> (state, value) at THIS cap. `seed_policy`
    `replicate` (the seed views: a replicate block at seed base `seed_n`) or `default` (the non-seed target: the rule alone, no replicate block);
    `attempts_before` names earlier FAILED attempts (`(attempt_id, result, rc)`) written to RESULT and ladder_state.json before the accepted one."""
    store_key = f"{slug}__controls__t0__a0__{ihash8}__{chash8}"
    kd = os.path.join(d, "cells", store_key)
    steered = okey not in (NONE, PROMPTED)
    fit_transfer = steered if fit_transfer is None else fit_transfer
    accepted = f"a{len(attempts_before) + 1}"
    prov_sha = prov_sha if prov_sha is not None else (obj_sha if steered else None)
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
              "object_artifact_sha256": (obj_sha if steered else None), "view": f"s{seed_n}", "route_kind": ("lens" if steered else "base")}
        lines.append(json.dumps({"task": G.TASK, "metric": "acc", "item_id": o["item_id"], "value": o["value"], "created_at": 1789971587.0 + k, "slurm_job_id": job,
                                 "env_digest": "117faa9c973067f5", "model_id": MODEL_HF, "harness_config_hash": chash8 + "870d1a0b", "intervention_hash": ihash8 + "2e209076",
                                 "extra_json": json.dumps(ex)}))
    agg = {"object_artifact_sha256": (obj_sha if steered else None), "producer_tree_sha": tree, "route_kind": ("lens" if steered else "base"), "sampling_mode": "thinking_card", "view": f"s{seed_n}"}
    lines.append(json.dumps({"task": G.TASK, "metric": "acc_aggregate", "item_id": "__aggregate__", "value": acc, "extra_json": json.dumps(agg)}))
    lines.append(json.dumps({"task": G.TASK, "metric": "n", "item_id": "__aggregate__", "value": float(len(items)), "extra_json": json.dumps(agg)}))
    _write(os.path.join(kd, "rows.jsonl"), "\n".join(lines) + "\n")
    gs = dict(SAMPLING, acc=acc, n=len(items), n_scored=n_scored, n_unscored=len(items) - n_scored, n_at_cap=n_at_cap, max_tokens=cap, thinking="on", task=G.TASK,
              framework="inspect_evals-0.18.0", engine="vllm", samples_per_item=1)
    _write(os.path.join(kd, "gen_stats.json"), {G.TASK: gs})
    man = {"cell_id": okey, "harness_cell_id": okey, "model_id": MODEL_HF, "model_revision": _REVISION, "serving_identity": (G.STEERED_IDENTITY if steered else G.PASSTHROUGH_IDENTITY),
           "serving_path": ("lora" if steered else "passthrough"), "serving_engine": "lora_native", "steer_positions": ("all" if steered else None),
           "fit_provenance": {"object_sha": (obj_sha[:16] if steered else None), "model_id": (("Qwen/Qwen3.5-4B-Base" if fit_transfer else MODEL_HF) if steered else None),
                              "transfer": fit_transfer, "served_model_id": MODEL_HF, "served_revision": _REVISION},
           "producer": {"driver": "served_capability_run", "tree_sha": tree}, "row_constants": {"slurm_job_id": job, "env_digest": "117faa9c973067f5"},
           "routing_table": ROUTING.format(tag=tag, slug=slug, cap=cap, job=job), "harness_config_hash": chash8 + "870d1a0b", "intervention_hash": ihash8 + "2e209076",
           "geometry": {"max_model_len": (16640 if cap == G.CAP else 24832), "max_num_seqs": (31 if cap == G.CAP else 21), "source": "endpoint.json"},
           "seeds": {"inspect": ({"replicate": {"derivation": "contracts.inspect_replicate_seed", "policy": "replicate", "seed_base": seed_n}, "rule": SEED_RULE.replace("= 2", f"= {seed_n}"),
                                  "samples_per_item": {G.TASK: 1}} if seed_policy == "replicate" else
                                 {"rule": DEFAULT_SEED_RULE, "samples_per_item": {G.TASK: 1}})},
           "frameworks": {"inspect_ai": "0.3.260", "inspect_evals": "0.18.0"}, "sampling": dict(SAMPLING, mode="thinking_card"),
           "resolved_config": {"cell_id": okey, "system_sha": (_SI_SHA16 if okey == PROMPTED else None), "serving": {"serving_engine": "lora_native"}},
           "provenance": {"artifact_sha256": prov_sha, "cell_id": okey}, "oos_experiment": oos}
    if subset is not None:
        man["item_subset"] = {"cap": cap, "ids": sorted(subset), "n": len(subset), "source_cap": G.CAP, "task": G.TASK}
    _write(os.path.join(kd, "manifest.json"), man)
    _write(os.path.join(kd, "evidence.json"), {"written_at": ended, "serving_identity": man["serving_identity"]})
    _write(os.path.join(d, "evidence.json"), {"written_at": ended, "serving_identity": man["serving_identity"]})
    _write(os.path.join(d, "resolved_sampler.json"), {"verdict": "AGREES", "classes": {"client": {"n": len(items), "verdict": "AGREES", "distinct_resolved": [{"sampler": SAMPLING, "n": len(items)}]}}})
    _write(os.path.join(d, "request_evidence.jsonl"), json.dumps({"kind": "task", "task": G.TASK, "item_id": ITEM_IDS[0], "seed": 1000, "max_tokens": cap, **SAMPLING}) + "\n")
    _write(os.path.join(d, "usage_ledger.jsonl"), json.dumps({"task": G.TASK, "finish_reason": "stop", "max_tokens_requested": cap}) + "\n")
    earlier = [{"attempt_id": aid, "job_id": str(int(job) - 10 * (k + 1)), "job_name": f"g8k-{tag}-{slug}", "node": node, "cluster": "cluster_a", "result": res, "rc": rc,
                "started": started, "ended": started, "rung": str(cap), "tag": tag, "sha": (hashlib.sha1(aid.encode()).hexdigest() if res == "DRIVER_FAILED" else tree)}
               for k, (aid, res, rc) in enumerate(attempts_before)]
    _write(os.path.join(d, "ladder_state.json"), {"schema": G.LADDER_STATE_SCHEMA, "accepted_generation_attempt": accepted,
                                                  "attempts": earlier + [{"attempt_id": accepted, "job_id": job, "job_name": f"g8k-{tag}-{slug}", "node": node, "cluster": "cluster_a",
                                                                          "result": "COMPLETE", "rc": 0, "started": started, "ended": ended, "rung": str(cap), "tag": tag, "sha": tree}]})
    result_lines = [f"{started} job={a['job_id']} row=0 g8k-{tag}-{slug} g8k qwen35_4b {tag.rsplit('_', 1)[1]} {okey} rung={cap} attempt={a['attempt_id']} rc={a['rc']} driver_rc=none result={a['result']}"
                    for a in earlier]
    result_lines.append(f"{ended} job={job} row=0 g8k-{tag}-{slug} g8k qwen35_4b {tag.rsplit('_', 1)[1]} {okey} rung={cap} attempt={accepted} rc=0 driver_rc=0 result=COMPLETE")
    _write(os.path.join(d, "RESULT"), "\n".join(result_lines) + "\n")
    if cap == G.CAP:
        states = spec["states"]
        loops = sorted(i for i, (st, _v) in states.items() if st in ("loop", "scored_loop"))
        facts = []
        for o in outcomes:
            st = states[o["item_id"]][0]
            is_loop = st in ("loop", "scored_loop")
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


_REVISION = None
_SI_SHA16 = None


def _bind_seeds(seeds):
    global _REVISION, _SI_SHA16
    _REVISION = str(seeds.models["qwen35_4b_it"]["hf_revision"])
    _SI_SHA16 = str(seeds.si_object_id("safety_refusal")).split(":", 1)[1][:16]


def _three_row(seed_n: int, okey: str, spec: dict, legacy: bool = False) -> dict:
    """The producer's record for one pair: the item-wise write (default; `n_scored_final` / `correct_final` / `acc_over_scored_final` / `headline`,
    unresolved counting the not-escalated items with `n_unresolved_ladder_only` beside it) or, `legacy`, the mirror root's first write (base
    reading only; unresolved = loops + ceiling; a not-escalated item a miss in its bound)."""
    states, rung = spec["states"], spec["rung"]
    scored_b = {i: v for i, (st, v) in states.items() if st in ("scored", "scored_at_cap", "scored_loop")}
    n_at_cap = sum(1 for st, _v in states.values() if st in ("loop", "truncated", "scored_at_cap", "scored_loop"))
    loops = sum(1 for st, _v in states.values() if st in ("loop", "scored_loop"))          # the classifier's count
    loops_unresolved = sum(1 for st, _v in states.values() if st == "loop")
    correct_b = int(sum(scored_b.values()))
    final = dict(scored_b)
    still = 0
    r_scored = r_correct = 0
    if rung:
        for iid, (rst, rv) in rung.items():
            final.pop(iid, None)
            if rst == "scored":
                final[iid] = rv
                r_scored += 1
                r_correct += int(rv)
            elif rst == "ceiling":
                still += 1
    correct_f = int(sum(final.values()))
    ladder_only = loops + still
    not_escalated = 0 if rung else sum(1 for st, _v in states.values() if st == "truncated")
    below = sum(1 for st, _v in states.values() if st == "below_cap") + sum(1 for rst, _v in (rung or {}).values() if rst == "below_cap")
    unresolved = loops_unresolved + still + not_escalated
    row = {"view": f"s{seed_n}", "cell": LC.cell_grammar_legacy(okey)["paper_label"], "cell_id": okey, "n_attempted": G.N_ITEMS, "n_scored_8192": len(scored_b), "correct_8192": correct_b,
           "acc_over_scored": round(correct_b / len(scored_b), 4), "n_at_cap_8192": n_at_cap, "at_cap_share": n_at_cap / G.N_ITEMS, "n_loops": loops, "loop_share": loops / G.N_ITEMS,
           "n_eligible_for_rung": len(spec["eligible"]), "ladder_exit": spec["exit"],
           "rung_16384": ({"n": len(rung), "n_scored": r_scored, "correct": r_correct, "still_at_cap": still} if rung else None),
           "correct_after_rung": correct_f, "acc_over_attempted_after_rung": round(correct_f / G.N_ITEMS, 4), "acc_over_attempted_lower_bound": round(correct_f / G.N_ITEMS, 4)}
    if legacy:
        row.update(n_unresolved_neither_hit_nor_miss=ladder_only, acc_over_attempted_upper_bound=round((correct_f + ladder_only) / G.N_ITEMS, 4))
        return row
    row.update(n_unresolved_neither_hit_nor_miss=unresolved, n_unresolved_ladder_only=ladder_only, acc_over_attempted_upper_bound=round((correct_f + unresolved) / G.N_ITEMS, 4),
               n_scored_final=len(final), correct_final=correct_f, acc_over_scored_final=round(correct_f / len(final), 4), acc_over_scored_8192=round(correct_b / len(scored_b), 4),
               headline="acc_over_scored_final", n_unscored_below_cap_final=below, n_rung_drawn=len(rung or {}), n_rung_drawn_that_were_scored_at_base=len(set(rung or {}) & set(scored_b)),
               acc_over_attempted_final=round(correct_f / G.N_ITEMS, 4))
    return row


LORA_SPEC = dict(n_correct=160, n_wrong=10, loops=22, truncated=5, scored_at_cap=1, below_cap=2, rung=[("scored", 1.0), ("scored", 1.0), ("scored", 1.0), ("scored", 0.0), ("ceiling", None), ("scored", 1.0)])
NONE_SPEC = dict(n_correct=188, n_wrong=10, loops=1, truncated=1, below_cap=0, rung=None)
PROMPTED_SPEC = dict(n_correct=186, n_wrong=9, loops=2, truncated=3, below_cap=0, rung=[("scored", 1.0), ("scored", 0.0), ("scored", 1.0)])


def _mirror(tmp_path, seeds, *, root: str = None, specs: dict = None, pins_override: dict = None, three_override=None, drop_rung_for=(), seed_base_override=None,
            legacy_record: bool = False) -> dict:
    """A synthetic mirror root under tmp_path: the three pairs (fitted LoRA with a rung, unsteered without, prompted with), the store pins and the
    three-number record written from the same truth, and a roster / pins naming the synthetic object."""
    _bind_seeds(seeds)
    root = root or str(tmp_path / "gsm8k_seedrep")
    specs = specs or {LORA: LORA_SPEC, NONE: NONE_SPEC, PROMPTED: PROMPTED_SPEC}
    pins = {"_meta": {"schema": G.PINS_SCHEMA, "label": "synthetic"}}
    three = {"schema": G.THREE_NUMBERS_SCHEMA, "written": NOW, "cells": {}}
    keys = {}
    for k, (okey, sp) in enumerate(sorted(specs.items())):
        spec = _spec(**sp)
        spec["exit"] = G.LADDER_EXIT_ESCALATE if spec["rung"] else ("below_threshold" if spec["eligible"] else ("triggered_no_eligible" if any(st == "loop" for st, _ in spec["states"].values()) else "clean"))
        slug = okey.replace("__", "--", 1)
        ihash8, job = f"{k:08x}", str(585900 + k)
        base_items = dict(spec["states"])
        base_dir = os.path.join(root, TAG, slug, G.BASE_DIR)
        sb = seed_base_override if seed_base_override is not None else SEED_N
        keys[(okey, G.CAP)] = _store_files(base_dir, TAG, okey, slug, G.CAP, base_items, obj_sha=OBJ, tree=TREE, job=job, ihash8=ihash8, chash8="aaaa" + f"{k:04x}", seed_n=sb, spec=spec)
        pins[G.pin_key(G.selection_key(SEED_N), okey, G.CAP)] = TREE
        if spec["rung"] and okey not in drop_rung_for:
            rung_items = {iid: (("scored" if rst == "scored" else rst), rv) for iid, (rst, rv) in spec["rung"].items()}
            rung_dir = os.path.join(root, TAG, slug, G.CEILING_DIR)
            keys[(okey, G.CEILING)] = _store_files(rung_dir, TAG, okey, slug, G.CEILING, rung_items, obj_sha=OBJ, tree=TREE, job=str(585990 + k), ihash8=ihash8, chash8="bbbb" + f"{k:04x}",
                                                   seed_n=sb, spec=spec, subset=list(spec["rung"]), started="2026-09-21T06:45:35Z", ended="2026-09-21T06:51:11Z")
            pins[G.pin_key(G.selection_key(SEED_N), okey, G.CEILING)] = TREE
        three["cells"][f"s{SEED_N}/{LC.cell_grammar_legacy(okey)['paper_label']}"] = _three_row(SEED_N, okey, spec, legacy=legacy_record)
    if pins_override:
        pins.update(pins_override)
    if three_override:
        three_override(three)
    _write(os.path.join(root, G.PINS_FILE), pins)
    _write(os.path.join(root, G.THREE_NUMBERS_FILE), three)
    roster = {"targets": {"qwen35_4b": {"views": {f"s{SEED_N}": {"config": f"configs/sdfbase_full22_{TAG}.yaml", "cells": {LORA: STAGED, PROMPTED: None}}}}}}
    import yaml
    roster_path = _write(str(tmp_path / "roster.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": OBJ, "source": "synthetic"}}})
    return {"root": root, "roster": roster_path, "pins": pins_path, "keys": keys}


def _adapter(fx: dict, seeds) -> G.Gsm8kSeedrepAdapter:
    return G.Gsm8kSeedrepAdapter(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"])


def _read(ad: G.Gsm8kSeedrepAdapter, conn=None, group: str = TAG):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert sorted(groups) == [group], list(groups)
    recs = []
    for batch in ad.read(_ctx(conn), group, groups[group]):
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
    value = [m for m in rows if m["computation_id"] == G.merge_computation_id()]
    bound = [m for m in rows if m["computation_id"] == ""]
    assert len(rows) == 2 and len(value) == 1 and len(bound) == 1, rows
    return value[0], bound[0]


# --------------------------------------------------------------------------------------------
# layout and identity
# --------------------------------------------------------------------------------------------

def test_discover_names_the_store_files_and_groups_them_by_tag_with_the_shared_records_aliased(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = sorted(os.path.basename(i.uri) for i in inputs)
    assert f"{G.PINS_FILE}#{TAG}" in names and f"{G.THREE_NUMBERS_FILE}#{TAG}" in names
    assert names.count("cap_ladder.json") == 3 and names.count(G.OUTCOMES_FILE) == 5 and names.count("rows.jsonl") == 5 and names.count("manifest.json") == 5
    assert {ad.comparison_group(i) for i in inputs} == {TAG}
    roles = {os.path.basename(i.uri): i.role for i in inputs}
    assert roles["rows.jsonl"] == "rows" and roles["gen_stats.json"] == "completion" and roles["RESULT"] == "log"


def test_run_ids_key_on_the_base_store_name_not_the_root_path(tmp_path, seeds):
    fx1 = _mirror(tmp_path, seeds, root=str(tmp_path / "a" / "gsm8k_seedrep"))
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "b" / "elsewhere"))
    bt1, _ = _read(_adapter(fx1, seeds))
    bt2, _ = _read(_adapter(fx2, seeds))
    ids1 = {r["campaign_cell_id_legacy"]: r["run_id"] for r in bt1["run"]}
    ids2 = {r["campaign_cell_id_legacy"]: r["run_id"] for r in bt2["run"]}
    assert ids1 == ids2 and len(ids1) == 3
    inst = str(seeds.instrument_row(G.INSTRUMENT_SEED, G.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"])
    base_key = fx1["keys"][(LORA, G.CAP)]
    assert ids1[LORA] == I.run_id(LC.CAMPAIGN, base_key, LORA, G.EVAL_ID, inst) == G.merged_run_id(base_key, LORA, inst)
    assert ids1[LORA].startswith(f"{LC.CAMPAIGN}:{G.EVAL_ID}:{inst[:12]}:")
    r = _run(bt1, LORA)
    assert r["producer_run_key_legacy"] == base_key and r["producer_object_key_legacy"] == LORA and r["instrument_id"] == inst


# --------------------------------------------------------------------------------------------
# the item-wise merge, the three states, the three numbers
# --------------------------------------------------------------------------------------------

def test_the_rung_items_replace_the_base_items_item_wise_and_the_three_numbers_are_re_derived(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]
    r = _run(bt, LORA)
    # base: 170 scored (160 correct) + 1 scored-at-cap (0) = 171 scored / 160 correct; rung: 6 eligible -> 5 scored (4 correct), 1 still at cap;
    # final: 175 scored, 164 correct; unresolved (loops + ceiling) 23; 2 unscored below the cap
    assert r["lifecycle_state"] == "generated" and r["token_cap_tokens"] == G.CEILING and r["token_cap_rung_index"] == 1 and r["token_cap_ladder_json"] == [G.CAP, G.CEILING]
    assert r["n_items_planned"] == 200 and r["n_items_generated"] == 200 and r["n_items_judged"] == 175
    assert r["extra"]["escalation_state"] == "final_at_ceiling" and len(r["extra"]["ceiling_item_ids"]) == 6 and r["extra"]["n_ceiling_items_merged"] == 6
    tn = r["extra"]["three_numbers"]
    assert (tn["n_scored_final"], tn["correct_final"], tn["n_unresolved_neither_hit_nor_miss"], tn["n_loops"], tn["n_at_cap_ceiling"], tn["n_at_cap_not_escalated"],
            tn["n_unscored_below_cap"], tn["n_scored_at_cap_final"]) == (175, 164, 23, 22, 1, 0, 2, 0)
    assert tn["base_reading"] == {"acc": 160 / 171, "correct": 160, "n_scored": 171} and tn["ceiling_reading"] == {"n": 6, "n_scored": 5, "correct": 4, "still_at_cap": 1}
    assert tn["producer_record_agrees"] is True and tn["ladder_exit"] == "escalate" and tn["n_unresolved_ladder_only"] == 23 and tn["n_rung_drawn_that_were_scored_at_base"] == 1
    assert tn["producer_record_version"] == G.RECORD_VERSION_ITEMWISE
    value, bound = _metrics(bt, r["run_id"])
    assert value["metric_id"] == "acc" and value["value"] == 164 / 175 and (value["k"], value["n"]) == (164, 175) and value["computed_by"] == "registry" and value["ci_method"] == "wilson_95"
    assert bound["metric_id"] == "acc" and bound["value"] == 164 / 200 and (bound["k"], bound["n"]) == (164, 200) and bound["computed_by"] == "producer"
    assert bound["producer_metric_spelling"] == G.BOUND_SPELLING and bound["ci_method"] == G.BOUND_CI_METHOD and (bound["ci_lo"], bound["ci_hi"]) == (164 / 200, 187 / 200)
    assert "secondary" in bound["extra"]["metric_role_expected"]
    # the computation the value row names is filed
    comps = {c["computation_id"] for c in bt["computation"]}
    assert value["computation_id"] in comps and value["computation_id"] == I.computation_id(G.MERGE_ALGORITHM, G.MERGE_VERSION, G.MERGE_SEED_PROVENANCE)
    # items: 200 rows, the three states
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == r["run_id"]}
    assert len(items) == 200 and all(i["judgment_id"] == "" and i["metric_id"] == "acc" and i["trial_index"] == 0 for i in items.values())
    labels = {}
    for i in items.values():
        labels[(i["label"], i["extra"].get("at_cap_class"), i["extra"].get("unscored_reason"))] = labels.get((i["label"], i["extra"].get("at_cap_class"), i["extra"].get("unscored_reason")), 0) + 1
    assert labels == {(None, None, None): 175, (IC.ITEM_LABEL_AT_CAP_UNRESOLVED, "loop", None): 22, (IC.ITEM_LABEL_AT_CAP_UNRESOLVED, "ceiling", None): 1,
                      ("unscored", None, G.UNSCORED_BELOW_CAP_CLASS): 2}, labels
    assert all(i["value"] is None for i in items.values() if i["label"] is not None) and all(i["value"] in (0.0, 1.0) for i in items.values() if i["label"] is None)
    rung_items = [i for i in items.values() if i["extra"]["source_store"] == "ceiling"]
    assert len(rung_items) == 6 and all(i["extra"]["final_cap"] == G.CEILING for i in rung_items)
    replaced = [i for i in rung_items if i["extra"]["base_value_replaced"] == 0.0]
    assert len(replaced) == 1 and replaced[0]["extra"]["base_at_cap"] is True        # the scored-at-cap base item, re-drawn at the rung
    flags = {f["item_id"]: f for f in bt["item_flags"] if f["run_id"] == r["run_id"]}
    assert sum(f["at_token_cap"] for f in flags.values()) == 23 and sum(1 for f in flags.values() if f["degenerate_rep_ratio_loop"] == 1) == 22
    assert all(f["generation_artifact_id"] is not None for f in flags.values() if items[f["item_id"]]["value"] is not None)
    assert all(f["generation_artifact_id"] is None for f in flags.values() if items[f["item_id"]]["value"] is None)
    # the run-level shares over the 200 attempted items
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[G.RAW_AT_CAP_MEASURE]["numerator"], deg[G.RAW_AT_CAP_MEASURE]["denominator"], deg[G.RAW_AT_CAP_MEASURE]["denominator_unit"]) == (23, 200, "items")
    assert (deg[G.LOOP_MEASURE]["numerator"], deg[G.COHERENT_MEASURE]["numerator"]) == (22, 1)
    assert deg["classifier"]["measurement_state"] == "not_measured" and deg[IC.AT_CAP_RETOKENIZED]["measurement_state"] == "not_measured"
    assert set(deg) == set(IC.DECLARED_MEASURES) | {G.COHERENT_MEASURE}


def test_the_run_row_carries_the_attested_identity(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds))
    r = _run(bt, LORA)
    assert r["model_id"] == "qwen35_4b_it" and r["object_id"] == OBJ and r["view_legacy"] == TAG and r["lane_source_legacy"] == G.SOURCE and r["campaign"] == LC.CAMPAIGN
    assert r["transfer_kind"] == "base_to_instruct" and r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all" and r["thinking_mode"] == "on"
    assert r["code_sha"] == TREE and r["model_revision"] == _REVISION and r["cluster_job_id"] == "585900" and r["substrate"] == "cluster_a" and r["substrate_source"] == "recorded"
    assert r["seed_base"] == SEED_N and r["decode_seed_rule"] == "per_item" and r["trial_index"] == 0 and r["extra"]["decode_seed_policy"] == "replicate"
    assert r["decode_sent_json"] == SAMPLING and r["decode_temperature_sent"] == 0.6 and r["decode_top_k_sent"] == 20 and r["decode_source"] == "model_card"
    assert r["instrument_match"] == "match" and r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|gsm8k|on|thinking_card@v1"
    assert r["producer_session"] == "Overflow" and r["n_shot"] == 10 and r["scoring_rule"] == "numeric_match" and r["harness"] == "inspect_ai" and "0.3.260" in r["harness_version"]
    assert r["is_operating_point"] == 1 and r["strength_nominal"] == 1.0 and r["judge_id"] is None
    assert r["extra"]["store_pins"] == {str(G.CAP): TREE, str(G.CEILING): TREE} and r["extra"]["cluster_job_ids"][str(G.CEILING)] == "585990"
    assert r["store_completion_artifact_uri"].endswith(os.path.join(G.CEILING_DIR, "cells", fx["keys"][(LORA, G.CEILING)], "gen_stats.json"))
    fc = r["field_confidence_json"]
    assert fc["code_sha"]["confidence"] == "recorded" and fc["seed_base"]["confidence"] == "recorded" and fc["n_items_judged"]["confidence"] == "derived"
    # NO state event: the harvester's rule-scored completion stamps the run
    assert "run_state" not in bt
    # every steered run is contrasted against the view's own `none`, item-wise over the items scored on both sides
    none_id = _run(bt, NONE)["run_id"]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {r["run_id"], _run(bt, PROMPTED)["run_id"]} and all(c["anchor_run_id"] == none_id and c["anchor_kind"] == "same_standup_none" for c in contrasts.values())
    c = contrasts[r["run_id"]]
    assert c["n_pairs"] == 175 - 0 - 0 and c["metric_id"] == "acc" and c["delta_method"] == "exact_mcnemar" and c["effect_kind"] == "cohens_h" and c["computed_by"] == "registry"
    assert c["computation_id"] == G.merge_computation_id() and 0.0 <= c["delta_p"] <= 1.0 and facts["contrasts"][LORA]["anchor"] == none_id
    assert facts["pairs"][LORA]["ceiling_items_merged"] == 6 and facts["ceiling_merged"] == 2 and facts["runs"] == 3


def test_a_pair_without_a_rung_files_final_at_base_with_the_not_escalated_item_and_the_unfitted_keys_are_trials(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, _ = _read(_adapter(fx, seeds))
    r = _run(bt, NONE)
    assert r["token_cap_tokens"] == G.CAP and r["token_cap_rung_index"] == 0 and r["extra"]["escalation_state"] == "final_at_base" and r["extra"]["ceiling_item_ids"] == []
    assert r["object_id"] == "none" and r["serving_path"] == "passthrough" and r["transfer_kind"] == "none" and r.get("strength_nominal") is None
    assert r["trial_index"] == SEED_N and r["seed_base"] == SEED_N and r["extra"]["decode_replicate"]["trial_index"] == SEED_N
    assert r["extra"]["decode_replicate"]["twin_tags_legacy"] == ["qwen35_4b_s1", "qwen35_4b_s3"]
    value, bound = _metrics(bt, r["run_id"])
    assert (value["k"], value["n"]) == (188, 198) and value["extra"]["equals_gen_stats_acc"] is True
    assert (bound["k"], bound["n"], bound["ci_lo"], bound["ci_hi"]) == (188, 200, 188 / 200, 190 / 200)      # 1 loop + 1 not-escalated item unresolved: both widen the bound
    assert bound["extra"]["n_unresolved_neither_hit_nor_miss"] == 2 and bound["extra"]["n_unresolved_ladder_only"] == 1 and bound["extra"]["n_at_cap_not_escalated"] == 1
    assert r["extra"]["three_numbers"]["n_unresolved_neither_hit_nor_miss"] == 2 and r["extra"]["three_numbers"]["n_unresolved_ladder_only"] == 1
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == r["run_id"]}
    classes = sorted(i["extra"]["at_cap_class"] for i in items.values() if i["label"] == IC.ITEM_LABEL_AT_CAP_UNRESOLVED)
    assert classes == ["loop", "not_escalated"] and all(i["trial_index"] == SEED_N for i in items.values())
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[G.RAW_AT_CAP_MEASURE]["numerator"], deg[G.LOOP_MEASURE]["numerator"], deg[G.COHERENT_MEASURE]["numerator"]) == (2, 1, 1)
    p = _run(bt, PROMPTED)
    assert p["object_id"] == str(seeds.si_object_id("safety_refusal")) and p["serving_path"] == "passthrough" and p["transfer_kind"] == "prompted_only" and p["trial_index"] == SEED_N
    assert p["extra"]["escalation_state"] == "final_at_ceiling" and len(p["extra"]["ceiling_item_ids"]) == 3


# --------------------------------------------------------------------------------------------
# refusals by name (the pair, never the tag)
# --------------------------------------------------------------------------------------------

def _refused(facts, okey: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{TAG}|{okey} -- ")]
    assert len(hits) == 1, facts["refusals"]
    return hits[0]


def test_a_store_pin_that_is_not_the_manifests_tree_refuses_the_pair_and_the_siblings_file(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, pins_override={G.pin_key(G.selection_key(SEED_N), LORA, G.CAP): "f" * 40})
    bt, facts = _read(_adapter(fx, seeds))
    assert "store pin" in _refused(facts, LORA) and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, PROMPTED}


def test_a_three_number_record_that_disagrees_refuses_the_pair_naming_the_number(tmp_path, seeds):
    def bump(three):
        three["cells"][f"s{SEED_N}/constitution_lora"]["n_loops"] += 1
        three["cells"][f"s{SEED_N}/constitution_lora"]["n_unresolved_neither_hit_nor_miss"] += 1
    fx = _mirror(tmp_path, seeds, three_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, LORA)
    assert "n_loops: producer 23, re-derived 22" in why and "n_unresolved_neither_hit_nor_miss" in why and len(bt["run"]) == 2


def test_a_ladder_that_declared_escalate_without_a_rung_store_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, drop_rung_for=(LORA,))
    # the pin for the rung is then absent too; the base's cap_ladder still says escalate
    bt, facts = _read(_adapter(fx, seeds))
    assert "escalate" in _refused(facts, LORA) and "no ceiling store" in _refused(facts, LORA) and len(bt["run"]) == 2


def test_a_replicate_seed_base_that_is_not_the_tags_index_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, seed_base_override=3)
    bt, facts = _read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 3 and all("seed_base 3" in x and f"replicate at {SEED_N}" in x for x in facts["refusals"]) and "run" not in bt


def test_a_rung_subset_that_is_not_the_bases_eligible_set_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    p = os.path.join(fx["root"], TAG, LORA.replace("__", "--", 1), G.CEILING_DIR, "cells", fx["keys"][(LORA, G.CEILING)], "manifest.json")
    man = json.load(open(p, encoding="utf-8"))
    man["item_subset"]["ids"] = man["item_subset"]["ids"][:-1] + [ITEM_IDS[0]]
    _write(p, man)
    bt, facts = _read(_adapter(fx, seeds))
    assert "item_subset.ids" in _refused(facts, LORA) and len(bt["run"]) == 2


def test_an_attested_object_sha_that_is_not_the_roster_pins_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    other = hashlib.sha256(b"another object").hexdigest()
    _write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": other, "source": "synthetic"}}})
    bt, facts = _read(_adapter(fx, seeds))
    assert "fit_provenance.object_sha" in _refused(facts, LORA) and "no prefix of the pinned object" in _refused(facts, LORA) and len(bt["run"]) == 2


def test_the_reader_never_consults_the_registry_for_the_object(tmp_path, seeds):
    """An object row absent from a connected registry is NOT a refusal here: the run is emitted on the attested identity and blocked_records holds it."""
    import sqlite3
    fx = _mirror(tmp_path, seeds)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    bt, facts = _read(_adapter(fx, seeds), conn=conn)
    assert facts["n_refusals"] == 0 and _run(bt, LORA)["object_id"] == OBJ


def test_the_records_first_write_is_checked_under_its_own_definition_and_the_widened_bound_is_filed_anyway(tmp_path, seeds):
    """The mirror root's first three-number write (no final-state keys; unresolved = loops + ceiling; a not-escalated item a miss in its
    bound) passes the cross-check as the producer wrote it, and the rows filed are the SAME as under the item-wise record: the value row, the
    widened bound, the item labels."""
    fx_new = _mirror(tmp_path, seeds, root=str(tmp_path / "new" / "gsm8k_seedrep"))
    fx_old = _mirror(tmp_path, seeds, root=str(tmp_path / "old" / "gsm8k_seedrep"), legacy_record=True)
    bt_new, facts_new = _read(_adapter(fx_new, seeds))
    bt_old, facts_old = _read(_adapter(fx_old, seeds))
    assert facts_new["n_refusals"] == 0 and facts_old["n_refusals"] == 0, (facts_new["refusals"], facts_old["refusals"])
    assert facts_old["pairs"][NONE]["producer_record_version"] == G.RECORD_VERSION_BASE and facts_new["pairs"][NONE]["producer_record_version"] == G.RECORD_VERSION_ITEMWISE
    for okey in (LORA, NONE, PROMPTED):
        rn, ro = _run(bt_new, okey), _run(bt_old, okey)
        assert rn["run_id"] == ro["run_id"]
        vn, bn = _metrics(bt_new, rn["run_id"]); vo, bo = _metrics(bt_old, ro["run_id"])
        assert (vn["value"], vn["k"], vn["n"]) == (vo["value"], vo["k"], vo["n"]) and (bn["ci_lo"], bn["ci_hi"], bn["n_incomplete"]) == (bo["ci_lo"], bo["ci_hi"], bo["n_incomplete"])
        tn, to = rn["extra"]["three_numbers"], ro["extra"]["three_numbers"]
        assert {k: v for k, v in tn.items() if k not in ("producer_record_version", "producer_record")} == {k: v for k, v in to.items() if k not in ("producer_record_version", "producer_record")}
    labels_new = sorted((i["run_id"], i["item_id"], i["label"] or "") for i in bt_new["item_result"])
    labels_old = sorted((i["run_id"], i["item_id"], i["label"] or "") for i in bt_old["item_result"])
    assert labels_new == labels_old and len(labels_new) == 600


def test_a_final_state_key_that_disagrees_refuses_the_pair_naming_it(tmp_path, seeds):
    def bump(three):
        three["cells"][f"s{SEED_N}/none"]["n_scored_final"] -= 1
        three["cells"][f"s{SEED_N}/none"]["acc_over_scored_final"] = round(188 / 197, 4)
    fx = _mirror(tmp_path, seeds, three_override=bump)
    bt, facts = _read(_adapter(fx, seeds))
    why = _refused(facts, NONE)
    assert "n_scored_final: producer 197, re-derived 198" in why and "acc_over_scored_final" in why and len(bt["run"]) == 2


# --------------------------------------------------------------------------------------------
# the paired views: the view's own `none` anchor and the composed difference_relu object
# --------------------------------------------------------------------------------------------

TAG_P = "qwen35_4b_s2p"
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"
STAGED_P = f"paired_eb64s2__{COMPOSED}"
OBJ_P = hashlib.sha256(b"synthetic gsm8k seedrep composed s2p").hexdigest()
OOS_P = "gsm8k_paired200_qwen35_4b_native_paired_8192"
NONE_P_SPEC = dict(n_correct=187, n_wrong=10, loops=0, truncated=3, below_cap=0, rung=[("scored", 1.0), ("scored", 1.0), ("scored", 0.0)])
COMPOSED_SPEC = dict(n_correct=171, n_wrong=15, loops=12, truncated=0, below_cap=0, scored_loops=2, rung=None)
#: the seed view's `none` store key hashes, which the paired view's `none` store REUSES
NONE_HASHES = ("2ff9cdff", "aaaa0001")


def _final_state_row(selection: str, slug: str, spec: dict) -> dict:
    """The producer's per-pair item-wise record (g8k_final_state.py) from the synthetic truth."""
    states, rung = spec["states"], spec["rung"]
    scored_b = {i: v for i, (st, v) in states.items() if st in ("scored", "scored_at_cap", "scored_loop")}
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
    scored_at_cap = 0 if rung else sum(1 for st, _v in states.values() if st in ("scored_at_cap", "scored_loop"))
    return {"view": selection, "store_cell": slug, "n_items": G.N_ITEMS, "n_scored_8192": len(scored_b), "correct_8192": int(sum(scored_b.values())), "rung_present": bool(rung),
            "n_rung_drawn": len(rung or {}), "n_rung_drawn_that_were_scored_at_base": len(set(rung or {}) & set(scored_b)), "n_scored_final": len(final),
            "correct_final": int(sum(final.values())), "acc_over_scored_final": round(sum(final.values()) / len(final), 4), "n_unresolved_final": unresolved,
            "n_unscored_below_cap_final": sum(1 for st, _v in states.values() if st == "below_cap"), "n_scored_at_cap_final": scored_at_cap,
            "acc_over_attempted_final": round(sum(final.values()) / G.N_ITEMS, 4), "rung_items_not_in_base": 0, "files": []}


def _paired_mirror(tmp_path, seeds, *, root: str = None, none_base_pin: bool = True, composed_sha_pinned: str = None, fs_override=None) -> dict:
    """A synthetic paired view `qwen35_4b_s2p`: its own `none` (a ceiling store re-drawing 3 items) reusing the seed view's `none` store key, and the
    composed difference_relu object (no ceiling store; 14 loops of which 2 the scorer scored at the cap). The producer's records: store pins, an
    EMPTY three-number record (the paired views are not in it) and the per-pair final-state record."""
    _bind_seeds(seeds)
    root = root or str(tmp_path / "gsm8k_seedrep")
    pins = {"_meta": {"schema": G.PINS_SCHEMA, "label": "synthetic"}}
    fstate = {}
    keys = {}
    none_spec = _spec(**NONE_P_SPEC); none_spec["exit"] = G.LADDER_EXIT_ESCALATE
    comp_spec = _spec(**COMPOSED_SPEC); comp_spec["exit"] = "triggered_no_eligible"
    keys[(NONE, G.CAP)] = _store_files(os.path.join(root, TAG_P, "none", G.BASE_DIR), TAG_P, NONE, "none", G.CAP, dict(none_spec["states"]), obj_sha=None, tree=TREE, job="589634",
                                       ihash8=NONE_HASHES[0], chash8=NONE_HASHES[1], seed_n=SEED_N, spec=none_spec)
    keys[(NONE, G.CEILING)] = _store_files(os.path.join(root, TAG_P, "none", G.CEILING_DIR), TAG_P, NONE, "none", G.CEILING,
                                           {iid: (rst, rv) for iid, (rst, rv) in none_spec["rung"].items()}, obj_sha=None, tree=TREE, job="591251", ihash8=NONE_HASHES[0], chash8="71d27393",
                                           seed_n=SEED_N, spec=none_spec, subset=list(none_spec["rung"]), started="2026-09-21T14:05:00Z", ended="2026-09-21T14:12:00Z")
    slug = COMPOSED.replace("__", "--", 1)
    keys[(COMPOSED, G.CAP)] = _store_files(os.path.join(root, TAG_P, slug, G.BASE_DIR), TAG_P, COMPOSED, slug, G.CAP, dict(comp_spec["states"]), obj_sha=OBJ_P, tree=TREE, job="590983",
                                           ihash8="76c03d28", chash8="fe4a3ad6", seed_n=SEED_N, spec=comp_spec, oos=OOS_P)
    if none_base_pin:
        pins[G.pin_key("s2p", NONE, G.CAP)] = TREE
    pins[G.pin_key("s2p", NONE, G.CEILING)] = TREE
    pins[G.pin_key("s2p", COMPOSED, G.CAP)] = TREE
    fstate[f"s2p/none"] = _final_state_row("s2p", "none", none_spec)
    fstate[f"s2p/{slug}"] = _final_state_row("s2p", slug, comp_spec)
    if fs_override:
        fs_override(fstate)
    _write(os.path.join(root, G.PINS_FILE), pins)
    _write(os.path.join(root, G.THREE_NUMBERS_FILE), {"schema": G.THREE_NUMBERS_SCHEMA, "written": NOW, "cells": {}})
    _write(os.path.join(root, G.FINAL_STATE_FILE), fstate)
    roster = {"targets": {"qwen35_4b": {"views": {"s2": {"cells": {LORA: STAGED, PROMPTED: None}},
                                                  "s2p": {"config": f"configs/sdfbase_full22_{TAG_P}.yaml", "cells": {COMPOSED: STAGED_P, PROMPTED: None}}}}}}
    import yaml
    roster_path = _write(str(tmp_path / "roster_p.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / "pins_p.json"), {"qwen35_4b": {STAGED: {"object_sha256": OBJ}, STAGED_P: {"object_sha256": composed_sha_pinned or OBJ_P}}})
    return {"root": root, "roster": roster_path, "pins": pins_path, "keys": keys}


def test_a_paired_view_files_its_own_none_and_the_composed_object_paired_against_it(tmp_path, seeds):
    fx = _paired_mirror(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds), group=TAG_P)
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert facts["paired"] is True and facts["paired_of"] == "s2" and facts["runs"] == 2
    none_r, comp_r = _run(bt, NONE), _run(bt, COMPOSED)
    inst = none_r["instrument_id"]
    # the paired view's stores reuse the seed view's store key: the producer key is tag-prefixed and the id differs from the seed view's
    assert none_r["producer_run_key_legacy"] == f"{TAG_P}/{fx['keys'][(NONE, G.CAP)]}" and none_r["view_legacy"] == TAG_P
    assert none_r["run_id"] == G.merged_run_id(fx["keys"][(NONE, G.CAP)], NONE, inst, tag=TAG_P, paired=True) != G.merged_run_id(fx["keys"][(NONE, G.CAP)], NONE, inst)
    assert comp_r["producer_run_key_legacy"] == f"{TAG_P}/{fx['keys'][(COMPOSED, G.CAP)]}" and comp_r["object_id"] == OBJ_P and comp_r["transfer_kind"] == "base_to_instruct"
    for r in (none_r, comp_r):
        assert r["extra"]["paired_of"] == "s2" and r["extra"]["standup_replicate_of_view"] == "qwen35_4b_s2" and r["extra"]["producer_key_rule"] == G.PAIRED_KEY_RULE
        assert r["extra"]["three_numbers"]["producer_record_version"] == G.RECORD_VERSION_FINAL_STATE_FILE and r["seed_base"] == SEED_N
    assert comp_r["extra"]["oos_tuple"] == OOS_P and none_r["extra"]["oos_tuple"] is None
    assert none_r["trial_index"] == SEED_N and none_r["extra"]["decode_replicate"]["standup_replicate_of_view"] == "qwen35_4b_s2" and comp_r["trial_index"] == 0
    # the paired `none`: 3 items re-drawn at the ceiling, all scored -> 200 scored, 189 correct, nothing unresolved
    v, b = _metrics(bt, none_r["run_id"])
    assert (v["k"], v["n"]) == (189, 200) and (b["ci_lo"], b["ci_hi"]) == (189 / 200, 189 / 200) and none_r["extra"]["escalation_state"] == "final_at_ceiling"
    # the composed object: no ceiling store, every at-cap item a loop; 2 loops the scorer scored keep their value, 12 are unresolved
    v, b = _metrics(bt, comp_r["run_id"])
    tn = comp_r["extra"]["three_numbers"]
    assert (v["k"], v["n"]) == (171, 188) and (tn["n_loops"], tn["n_unresolved_neither_hit_nor_miss"], tn["n_unresolved_ladder_only"], tn["n_scored_at_cap_final"]) == (14, 12, 14, 2)
    assert comp_r["extra"]["at_cap_classes"] == {"loop": 12, "ceiling": 0, "not_escalated": 0} and tn["ladder_exit"] == "triggered_no_eligible" and comp_r["extra"]["escalation_state"] == "final_at_base"
    assert (b["ci_lo"], b["ci_hi"]) == (171 / 200, 183 / 200)
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == comp_r["run_id"]}
    assert (deg[G.LOOP_MEASURE]["numerator"], deg[G.RAW_AT_CAP_MEASURE]["numerator"], deg[G.COHERENT_MEASURE]["numerator"]) == (14, 14, 0)
    flags = {f["item_id"]: f for f in bt["item_flags"] if f["run_id"] == comp_r["run_id"]}
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == comp_r["run_id"]}
    scored_loops = [i for i in items.values() if i["value"] is not None and i["extra"]["scored_at_cap"]]
    assert len(scored_loops) == 2 and all(flags[i["item_id"]]["degenerate_rep_ratio_loop"] == 1 and flags[i["item_id"]]["at_token_cap"] == 1 for i in scored_loops)
    # the contrast: the composed object against ITS view's `none` (never the seed view's), item-wise over the items scored on both sides
    assert len(bt["run_contrast"]) == 1
    c = bt["run_contrast"][0]
    assert c["run_id"] == comp_r["run_id"] and c["anchor_run_id"] == none_r["run_id"] and c["anchor_kind"] == "same_standup_none" and c["n_pairs"] == 188
    assert c["extra"]["anchor_view_legacy"] == TAG_P and c["delta_method"] == "exact_mcnemar" and c["effect_kind"] == "cohens_h" and c["computed_by"] == "registry"
    assert facts["contrasts"][COMPOSED]["anchor"] == none_r["run_id"] and facts["contrasts"][COMPOSED]["n_pairs"] == 188


def test_a_paired_composed_object_whose_served_sha_is_not_the_pinned_one_is_refused(tmp_path, seeds):
    fx = _paired_mirror(tmp_path, seeds, composed_sha_pinned=hashlib.sha256(b"another composed object").hexdigest())
    bt, facts = _read(_adapter(fx, seeds), group=TAG_P)
    why = [x for x in facts["refusals"] if x.startswith(f"{TAG_P}|{COMPOSED} -- ")]
    assert len(why) == 1 and "no prefix of the pinned object" in why[0] and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE} and "run_contrast" not in bt


def test_a_paired_none_without_its_base_pin_is_refused_and_the_composed_object_files_without_a_contrast(tmp_path, seeds):
    fx = _paired_mirror(tmp_path, seeds, none_base_pin=False)
    bt, facts = _read(_adapter(fx, seeds), group=TAG_P)
    why = [x for x in facts["refusals"] if x.startswith(f"{TAG_P}|none -- ")]
    assert len(why) == 1 and "store pin <absent> (qwen35_4b/s2p/none@8192)" in why[0]
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {COMPOSED} and "run_contrast" not in bt
    assert facts["contrasts"][COMPOSED]["anchor"] is None and "did not file" in facts["contrasts"][COMPOSED]["why"]


def test_a_final_state_record_that_disagrees_refuses_the_pair_naming_the_count(tmp_path, seeds):
    def bump(fs):
        fs["s2p/none"]["n_unresolved_final"] += 1
    fx = _paired_mirror(tmp_path, seeds, fs_override=bump)
    bt, facts = _read(_adapter(fx, seeds), group=TAG_P)
    why = [x for x in facts["refusals"] if x.startswith(f"{TAG_P}|none -- ")]
    assert len(why) == 1 and G.FINAL_STATE_FILE in why[0] and "n_unresolved_final: producer 1, re-derived 0" in why[0]


# --------------------------------------------------------------------------------------------
# the all-positions object (v4): ONE fit on every seed view = three decode-seed trials; its own default-seed row a non-seed target
# --------------------------------------------------------------------------------------------

ARDITI_KEY = G.ARDITI_OBJECT_KEY
ARDITI_STAGED = "ar_c0p3__arditi_allpos"
ARDITI_OBJ = hashlib.sha256(b"synthetic gsm8k seedrep arditi allpos c0p3").hexdigest()          # ONE object on every seed view (never the live sha)
ARDITI_TREE = hashlib.sha1(b"synthetic gsm8k arditi allpos tree").hexdigest()
ARDITI_TUPLES = {"8192": "gsm8k_paired200_qwen35_4b_steer_allpos_8192", "16384": "gsm8k_paired200_qwen35_4b_steer_allpos_16384"}
TAG_IT = "qwen35_4b_it"
TARGET_GROUP = f"{G.TARGET_MIRROR_DIR}/{TAG_IT}"
#: the shape of the live default-seed store: 196 scored / 190 correct at 8,192 (acc 0.9694), 4 at the cap of which 3 loops and 1 eligible, re-drawn correct
ARDITI_SPEC = dict(n_correct=190, n_wrong=6, loops=3, truncated=1, below_cap=0, rung=[("scored", 1.0)])
ARDITI_SPEC_NO_RUNG = dict(n_correct=190, n_wrong=7, loops=3, truncated=0, below_cap=0, rung=None)
ARDITI_IHASH = "92b66dce"


def _arditi_row(tag: str, selection: str, out_head: str, index: int = 0) -> dict:
    """One declaration row (the producer's eligibility record's / the matrix's shape) for the all-positions object on a standup."""
    return {"index": index, "job_id": f"g8k-qwen35_4b-{selection}-{ARDITI_KEY}", "model_key": "qwen35_4b", "view": selection, "tag": tag, "cell": ARDITI_KEY, "cell_slug": ARDITI_KEY,
            "harness_cell": ARDITI_KEY, "staged_id": ARDITI_STAGED, "object_sha256": ARDITI_OBJ, "fitted": True, "route_kind": "lens", "served_positions_expected": "all",
            "declared": False, "thinking_mode": "on", "tuple_name": ARDITI_TUPLES["8192"], "tuple_names": dict(ARDITI_TUPLES), "sampling_mode": "thinking_card",
            "sampling": dict(SAMPLING, source="synthetic"), "out_root": f"{out_head}/{tag}/{ARDITI_KEY}/{G.BASE_DIR}", "rung_out_root": f"{out_head}/{tag}/{ARDITI_KEY}/{G.CEILING_DIR}",
            "bank_view": "ar_c0p3", "served_layer": 12, "status": "launch", "object_source": "external"}


def _arditi_declarations(tmp_path, *, seed_tags=(1, 2, 3), edit_rows=None, edit_matrix=None) -> dict:
    """The all-positions roster / pins the key resolves through (synthetic: one rung, one staged object), the seed rows' eligibility record and the
    matrix's own row, as files under tmp_path."""
    import yaml
    roster = {"targets": {"qwen35_4b": {"views": {"ar_c0p3": {"c": 0.3, "dose_unit": "raw_coefficient", "phase": "both", "cells": {ARDITI_KEY: ARDITI_STAGED}}}}}}
    pins = {"qwen35_4b": {ARDITI_STAGED: {"staged_id": ARDITI_STAGED, "object_sha256": ARDITI_OBJ, "kind": "vector_all", "served_layer": 12, "c": 0.3, "dose_unit": "raw_coefficient",
                                          "phase": "both", "positions": "all", "view": "ar_c0p3"}}}
    rows_doc = {"schema": G.ARDITI_SEED_ROWS_SCHEMA, "generated_by": "synthetic", "wave": "qwen_seedrep", "role": G.ARDITI_ROLE,
                "rows": [_arditi_row(f"qwen35_4b_s{k}", f"s{k}", "controls_qwen_seedrep", index=i) for i, k in enumerate(seed_tags)]}
    if edit_rows:
        edit_rows(rows_doc)
    matrix_doc = {"schema": "g8k_matrix/2", "wave": "allpos", "rows": [_arditi_row(TAG_IT, "it", "controls_allpos")]}
    if edit_matrix:
        edit_matrix(matrix_doc)
    return {"arditi_roster_path": _write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump(roster)),
            "arditi_pins_path": _write(str(tmp_path / "arditi_pins.json"), pins),
            "arditi_seed_rows_path": _write(str(tmp_path / "arditi_rows_qwen_seedrep.json"), rows_doc),
            "arditi_matrix_path": _write(str(tmp_path / "matrix_arditi_allpos.json"), matrix_doc)}


def _arditi_seed_mirror(tmp_path, seeds, *, seed_tags=(1, 2, 3), root: str = None, store_sha: str = None, decls: dict = None) -> dict:
    """A synthetic mirror of the seed views' all-positions rows: per tag its `none` (no rung) and `arditi_allpos` (a rung on s1 / s3, none on s2),
    the store pins and the three-number record written from the same truth. `store_sha` makes the arditi stores attest another object than the pin."""
    _bind_seeds(seeds)
    root = root or str(tmp_path / "gsm8k_seedrep")
    pins = {"_meta": {"schema": G.PINS_SCHEMA, "label": "synthetic"}}
    three = {"schema": G.THREE_NUMBERS_SCHEMA, "written": NOW, "cells": {}}
    keys = {}
    for k in seed_tags:
        tag, sel = f"qwen35_4b_s{k}", f"s{k}"
        none_spec = _spec(**NONE_SPEC); none_spec["exit"] = "below_threshold"
        keys[(tag, NONE, G.CAP)] = _store_files(os.path.join(root, tag, "none", G.BASE_DIR), tag, NONE, "none", G.CAP, dict(none_spec["states"]), obj_sha=None, tree=TREE,
                                                job=str(585900 + k), ihash8=NONE_HASHES[0], chash8=f"d02ab8{k:02d}", seed_n=k, spec=none_spec)
        pins[G.pin_key(sel, NONE, G.CAP)] = TREE
        three["cells"][f"{sel}/none"] = _three_row(k, NONE, none_spec)
        a_spec = _spec(**(ARDITI_SPEC if k != 2 else ARDITI_SPEC_NO_RUNG))
        a_spec["exit"] = G.LADDER_EXIT_ESCALATE if a_spec["rung"] else "triggered_no_eligible"
        sha = store_sha or ARDITI_OBJ
        keys[(tag, ARDITI_KEY, G.CAP)] = _store_files(os.path.join(root, tag, ARDITI_KEY, G.BASE_DIR), tag, ARDITI_KEY, ARDITI_KEY, G.CAP, dict(a_spec["states"]), obj_sha=sha,
                                                      tree=ARDITI_TREE, job=str(586100 + k), ihash8=ARDITI_IHASH, chash8=f"aa41f9{k:02d}", seed_n=k, spec=a_spec,
                                                      oos=ARDITI_TUPLES["8192"], fit_transfer=False)
        pins[G.pin_key(sel, ARDITI_KEY, G.CAP)] = ARDITI_TREE
        if a_spec["rung"]:
            keys[(tag, ARDITI_KEY, G.CEILING)] = _store_files(os.path.join(root, tag, ARDITI_KEY, G.CEILING_DIR), tag, ARDITI_KEY, ARDITI_KEY, G.CEILING,
                                                              {iid: (rst, rv) for iid, (rst, rv) in a_spec["rung"].items()}, obj_sha=sha, tree=ARDITI_TREE, job=str(586200 + k),
                                                              ihash8=ARDITI_IHASH, chash8=f"fab9ff{k:02d}", seed_n=k, spec=a_spec, subset=list(a_spec["rung"]),
                                                              started="2026-09-21T20:05:00Z", ended="2026-09-21T20:12:00Z", oos=ARDITI_TUPLES["16384"], fit_transfer=False)
            pins[G.pin_key(sel, ARDITI_KEY, G.CEILING)] = ARDITI_TREE
        three["cells"][f"{sel}/{ARDITI_KEY}"] = _three_row(k, ARDITI_KEY, a_spec)
    _write(os.path.join(root, G.PINS_FILE), pins)
    _write(os.path.join(root, G.THREE_NUMBERS_FILE), three)
    import yaml
    roster_path = _write(str(tmp_path / "roster.yaml"), yaml.safe_dump({"targets": {"qwen35_4b": {"views": {f"s{k}": {"cells": {LORA: STAGED}} for k in seed_tags}}}}))
    pins_path = _write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": OBJ}}})
    return {"root": root, "roster": roster_path, "pins": pins_path, "keys": keys, **(decls or _arditi_declarations(tmp_path, seed_tags=seed_tags))}


def _target_mirror(tmp_path, seeds, *, root: str = None, seed_policy: str = "default", summary_acc=None, slug: str = ARDITI_KEY, decls: dict = None, with_summary: bool = True) -> dict:
    """A synthetic mirror of the all-positions matrix's own default-seed row under `arditi_allpos_paper_objects/controls_allpos/qwen35_4b_it/`: a base
    store whose RESULT / ladder_state.json list two FAILED attempts before the accepted one, its rung store, the producer's controls_summary.json
    beside the base store and a PROVENANCE.md; no pins, no count record, no `none`."""
    _bind_seeds(seeds)
    root = root or str(tmp_path / "gsm8k_seedrep")
    spec = _spec(**ARDITI_SPEC); spec["exit"] = G.LADDER_EXIT_ESCALATE
    od = os.path.join(root, G.TARGET_MIRROR_DIR, G.TARGET_CONTROLS_DIR, TAG_IT, slug)
    okey = G.slug_to_object_key(slug)
    base_key = _store_files(os.path.join(od, G.BASE_DIR), TAG_IT, okey, slug, G.CAP, dict(spec["states"]), obj_sha=ARDITI_OBJ, tree=ARDITI_TREE, job="544615", ihash8=ARDITI_IHASH,
                            chash8="aa41f940", seed_n=0, spec=spec, oos=ARDITI_TUPLES["8192"], fit_transfer=False, seed_policy=seed_policy,
                            attempts_before=(("a1", "INCOMPLETE", 1), ("a2", "DRIVER_FAILED", 2)), started="2026-09-17T17:37:00Z", ended="2026-09-17T17:45:40Z", node="gpu-8")
    rung_key = _store_files(os.path.join(od, G.CEILING_DIR), TAG_IT, okey, slug, G.CEILING, {iid: (rst, rv) for iid, (rst, rv) in spec["rung"].items()}, obj_sha=ARDITI_OBJ,
                            tree=ARDITI_TREE, job="544730", ihash8=ARDITI_IHASH, chash8="fab9ffc1", seed_n=0, spec=spec, subset=list(spec["rung"]), oos=ARDITI_TUPLES["16384"],
                            fit_transfer=False, seed_policy=seed_policy, started="2026-09-17T17:51:16Z", ended="2026-09-17T17:58:23Z", node="gpu-23")
    # the rung manifest's source identity is the base store's configuration hash (the reader binds the two)
    rman_path = os.path.join(od, G.CEILING_DIR, "cells", rung_key, "manifest.json")
    with open(rman_path, "r", encoding="utf-8") as fh:
        rman = json.load(fh)
    rman["item_subset"]["source_identity_hash"] = "aa41f940870d1a0b"
    _write(rman_path, rman)
    if with_summary:
        n_scored = sum(1 for st, _v in spec["states"].values() if st in ("scored", "scored_at_cap", "scored_loop"))
        correct = sum(v for st, v in spec["states"].values() if st in ("scored", "scored_at_cap", "scored_loop"))
        _write(os.path.join(od, G.BASE_DIR, slug, G.TARGET_SUMMARY_FILE),
               {"cell_id": okey, "provenance": {"artifact_sha256": ARDITI_OBJ, "cell_id": okey, "class": "B"},
                "sources": {"gsm8k": {"cell_key": base_key, "producer_tree_sha": ARDITI_TREE, "harness_config_hash": "aa41f940870d1a0b"},
                            G.TASK: {"cell_key": base_key, "producer_tree_sha": ARDITI_TREE, "harness_config_hash": "aa41f940870d1a0b"}},
                "controls": {"gsm8k": {"acc": (summary_acc if summary_acc is not None else correct / n_scored), "n": 200, "max_tokens": G.CAP, "thinking": "on", "task": G.TASK,
                                       "engine": "vllm", "framework": "inspect_evals-0.18.0", "item_set": "paired200", "serving_engine": "lora_native", "temperature": 0.6}}})
    _write(os.path.join(od, G.TARGET_PROVENANCE_FILE), "# synthetic mirror of the all-positions matrix's own GSM8K row (default Inspect seed)\n")
    import yaml
    roster_path = _write(str(tmp_path / "roster_t.yaml"), yaml.safe_dump({"targets": {"qwen35_4b": {"views": {"s1": {"cells": {LORA: STAGED}}}}}}))
    pins_path = _write(str(tmp_path / "pins_t.json"), {"qwen35_4b": {STAGED: {"object_sha256": OBJ}}})
    return {"root": root, "roster": roster_path, "pins": pins_path, "keys": {(okey, G.CAP): base_key, (okey, G.CEILING): rung_key}, **(decls or _arditi_declarations(tmp_path))}


def _arditi_adapter(fx: dict, seeds) -> G.Gsm8kSeedrepAdapter:
    return G.Gsm8kSeedrepAdapter(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], arditi_roster_path=fx["arditi_roster_path"],
                                 arditi_pins_path=fx["arditi_pins_path"], arditi_seed_rows_path=fx["arditi_seed_rows_path"], arditi_matrix_path=fx["arditi_matrix_path"])


def _read_group(ad: G.Gsm8kSeedrepAdapter, group: str, conn=None):
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


def test_the_all_positions_seed_rows_file_as_decode_seed_trials_of_one_object(tmp_path, seeds):
    """Three seed views of ONE fit: three runs on one object id, trial_index = seed_base = the seed index, fit_seed unset, the operating point's strength
    columns and transfer_kind, the canonical seed instrument, the three numbers as everywhere, a contrast against the view's own `none`."""
    fx = _arditi_seed_mirror(tmp_path, seeds)
    ad = _arditi_adapter(fx, seeds)
    inst = str(seeds.instrument_row(G.INSTRUMENT_SEED, G.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"])
    runs, facts_by = {}, {}
    for k in (1, 2, 3):
        tag = f"qwen35_4b_s{k}"
        bt, facts, groups = _read_group(ad, tag)
        assert groups == ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"] and facts["n_refusals"] == 0, facts["refusals"]
        r = _run(bt, ARDITI_KEY)
        runs[k], facts_by[k] = r, facts
        assert r["object_id"] == ARDITI_OBJ and r["view_legacy"] == tag and r["trial_index"] == k and r["seed_base"] == k and r["extra"]["fit_seed"] is None
        assert r["producer_object_key_legacy"] == ARDITI_KEY and r["campaign_cell_id_legacy"] == ARDITI_KEY and r["producer_run_key_legacy"] == fx["keys"][(tag, ARDITI_KEY, G.CAP)]
        assert r["instrument_id"] == inst and r["instrument_match"] == "match" and r["campaign"] == LC.CAMPAIGN and r["lane_source_legacy"] == G.SOURCE
        assert (r["strength_coefficient"], r["strength_unit"], r["strength_nominal"], r["strength_nominal_source"], r["is_operating_point"], r["operating_point_id_legacy"]) == \
            (0.3, "raw_coefficient", 1.0, "derived", 1, ARDITI_STAGED)
        assert r["transfer_kind"] == "instruct_direct" and r["applied_positions_mode"] == "all" and r["serving_path"] == "lora_native" and r["object_id"] == ARDITI_OBJ
        assert r["extra"]["decode_replicate"]["rule"] == G.DECODE_TRIAL_RULE == SC.DECODE_TRIAL_RULE and r["extra"]["decode_replicate"]["trial_index"] == k
        assert r["extra"]["decode_replicate"]["twin_tags_legacy"] == [f"qwen35_4b_s{j}" for j in (1, 2, 3) if j != k]
        assert r["extra"]["decode_replicate"]["single_object"]["object_id"] == ARDITI_OBJ and r["extra"]["single_object"]["operating_point"] == "ar_c0p3"
        assert r["extra"]["declared_role"] == G.ARDITI_ROLE and r["extra"]["single_object"]["staged_id"] == ARDITI_STAGED and r["extra"]["object_staged_id"] == ARDITI_STAGED
        assert r["extra"]["oos_tuple"] == {"8192": ARDITI_TUPLES["8192"], "16384": (ARDITI_TUPLES["16384"] if k != 2 else None)}
        assert r["extra"]["paper_dose"] == 0.3 and r["extra"]["rung_legacy"] == "ar_c0p3" and r["extra"]["decode_seed_policy"] == "replicate" and r["extra"]["inspect_seed_base"] == k
        assert inst in r["extra"]["decode_replicate"]["coexist_ruling_needed"] and ARDITI_OBJ in r["extra"]["decode_replicate"]["coexist_ruling_needed"]
        items = [i for i in bt["item_result"] if i["run_id"] == r["run_id"]]
        assert len(items) == 200 and all(i["trial_index"] == k for i in items)
        fc = r["field_confidence_json"]
        assert fc["trial_index"]["confidence"] == "derived" and "DECODE-SEED TRIAL" in fc["trial_index"]["source"] and fc["strength_coefficient"]["confidence"] == "recorded"
        assert fc["transfer_kind"]["confidence"] == "recorded" and "instruct checkpoint" in fc["transfer_kind"]["source"]
        # the view's own `none` anchors it, the declared role riding on the contrast
        none_id = _run(bt, NONE)["run_id"]
        c = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]]
        assert len(c) == 1 and c[0]["anchor_run_id"] == none_id and c[0]["anchor_kind"] == "same_standup_none" and c[0]["extra"]["declared_role"] == G.ARDITI_ROLE
        assert facts["pairs"][ARDITI_KEY]["trial_rule"] == "DECODE_TRIAL_RULE" and facts["pairs"][ARDITI_KEY]["fit_seed"] is None and facts["pairs"][ARDITI_KEY]["seed_base"] == k
        assert "run_state" not in bt and "instrument" not in bt
    assert len({r["run_id"] for r in runs.values()}) == 3 and len({r["object_id"] for r in runs.values()}) == 1
    # the three numbers on s1 (a rung): base 196 / 190 (acc 0.9694), 4 at cap; final 197 / 191, 3 loops unresolved, bound [191, 194] / 200
    r1 = runs[1]
    tn = r1["extra"]["three_numbers"]
    assert (tn["n_scored_final"], tn["correct_final"], tn["n_unresolved_neither_hit_nor_miss"], tn["n_loops"], tn["n_at_cap_8192"], tn["n_eligible_for_rung"]) == (197, 191, 3, 3, 4, 1)
    assert tn["base_reading"] == {"acc": 190 / 196, "correct": 190, "n_scored": 196} and round(tn["base_reading"]["acc"], 4) == 0.9694 and tn["producer_record_version"] == G.RECORD_VERSION_ITEMWISE
    assert r1["token_cap_tokens"] == G.CEILING and r1["extra"]["escalation_state"] == "final_at_ceiling" and runs[2]["token_cap_tokens"] == G.CAP
    value, bound = _metrics(_read_group(ad, "qwen35_4b_s1")[0], r1["run_id"])
    assert (value["k"], value["n"], value["value"]) == (191, 197, 191 / 197) and (bound["k"], bound["n"], bound["ci_lo"], bound["ci_hi"]) == (191, 200, 191 / 200, 194 / 200)


def test_an_all_positions_seed_row_whose_store_attests_another_sha_is_refused_by_name_and_the_none_files(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = _arditi_seed_mirror(tmp_path, seeds, seed_tags=(2,), store_sha=other)
    bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), "qwen35_4b_s2")
    why = [x for x in facts["refusals"] if x.startswith(f"qwen35_4b_s2|{ARDITI_KEY} -- ")]
    assert len(why) == 1 and other[:16] in why[0] and ARDITI_OBJ[:16] in why[0] and "fit_provenance.object_sha" in why[0], facts["refusals"]
    assert _run(bt, NONE)["trial_index"] == 2 and len(bt["run"]) == 1


def test_an_all_positions_seed_row_with_no_declaration_row_is_refused_by_name(tmp_path, seeds):
    decls = _arditi_declarations(tmp_path, seed_tags=(1, 3))          # the record names s1 and s3 only
    fx = _arditi_seed_mirror(tmp_path, seeds, seed_tags=(2,), decls=decls)
    bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), "qwen35_4b_s2")
    why = [x for x in facts["refusals"] if x.startswith(f"qwen35_4b_s2|{ARDITI_KEY} -- ")]
    assert len(why) == 1 and "names 0 row(s) for tag qwen35_4b_s2" in why[0], facts["refusals"]
    assert len(bt["run"]) == 1 and _run(bt, NONE)["object_id"] == "none"


def test_an_all_positions_declaration_that_disagrees_with_the_store_refuses_the_pair_naming_the_fact(tmp_path, seeds):
    def bump(doc):
        doc["rows"][0]["tuple_names"]["8192"] = "gsm8k_paired200_qwen35_4b_steer_allpos_4096"
    fx = _arditi_seed_mirror(tmp_path, seeds, seed_tags=(1,), decls=_arditi_declarations(tmp_path, seed_tags=(1,), edit_rows=bump))
    _bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), "qwen35_4b_s1")
    why = [x for x in facts["refusals"] if x.startswith(f"qwen35_4b_s1|{ARDITI_KEY} -- ")]
    assert len(why) == 1 and "oos_experiment" in why[0] and "steer_allpos_4096" in why[0], facts["refusals"]
    # a declaration whose object is not the pins' is refused before any store is read
    def other_sha(doc):
        doc["rows"][0]["object_sha256"] = hashlib.sha256(b"not the pin").hexdigest()
    fx2 = _arditi_seed_mirror(tmp_path, seeds, seed_tags=(1,), root=str(tmp_path / "m2"), decls=_arditi_declarations(tmp_path / "d2", seed_tags=(1,), edit_rows=other_sha))
    _bt2, facts2, _ = _read_group(_arditi_adapter(fx2, seeds), "qwen35_4b_s1")
    why2 = [x for x in facts2["refusals"] if x.startswith(f"qwen35_4b_s1|{ARDITI_KEY} -- ")]
    assert len(why2) == 1 and "one object, two spellings" in why2[0], facts2["refusals"]


def test_the_reader_never_consults_the_registry_for_the_all_positions_object(tmp_path, seeds):
    """The object row is the arditi_allpos reader's to mint: its absence from a connected registry is never a refusal here (blocked_records holds the run)."""
    import sqlite3
    fx = _arditi_seed_mirror(tmp_path, seeds, seed_tags=(1,))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), "qwen35_4b_s1", conn=conn)
    assert facts["n_refusals"] == 0 and _run(bt, ARDITI_KEY)["object_id"] == ARDITI_OBJ


def test_the_default_seed_target_files_as_a_single_object_non_seed_point_on_its_own_instrument(tmp_path, seeds):
    """The matrix's own row: trial 0, no seed base, the default seed rule, the strength columns, a minted instrument (the record's harness configuration
    hash), the item-wise final state across 8,192 + the 16,384 rung, RESULT's earlier failed attempts ignored, no `none` and no contrast."""
    fx = _target_mirror(tmp_path, seeds)
    ad = _arditi_adapter(fx, seeds)
    bt, facts, groups = _read_group(ad, TARGET_GROUP)
    assert groups == [TARGET_GROUP] and facts["n_refusals"] == 0, facts["refusals"]
    r = _run(bt, ARDITI_KEY)
    seed_inst = seeds.instrument_row(G.INSTRUMENT_SEED, G.EVAL_ID, model_id="qwen35_4b_it")
    assert r["view_legacy"] == TAG_IT and r["trial_index"] == 0 and r["seed_base"] is None and r["decode_seed_rule"] == "per_item" and r["extra"]["fit_seed"] is None
    assert r["extra"]["decode_seed_policy"] == G.INSPECT_SEED_POLICY_DEFAULT and r["extra"]["inspect_seed_base"] is None and r["extra"]["seed_replicate_label"] is None
    assert r["extra"]["decode_seed_rule_text"] == DEFAULT_SEED_RULE and "decode_replicate" not in r["extra"] and r["extra"]["trial_rule"] == G.DEFAULT_SEED_TRIAL_RULE
    assert r["object_id"] == ARDITI_OBJ and r["producer_object_key_legacy"] == ARDITI_KEY and r["producer_run_key_legacy"] == fx["keys"][(ARDITI_KEY, G.CAP)]
    assert (r["strength_coefficient"], r["strength_unit"], r["strength_nominal"], r["is_operating_point"], r["operating_point_id_legacy"]) == (0.3, "raw_coefficient", 1.0, 1, ARDITI_STAGED)
    assert r["transfer_kind"] == "instruct_direct" and r["applied_positions_mode"] == "all" and r["serving_path"] == "lora_native" and r["model_id"] == "qwen35_4b_it"
    # its OWN instrument, minted from the record: the store's harness configuration hash, versions, window and ladder rule move it off the seed instrument
    assert r["instrument_id"] != str(seed_inst["instrument_id"]) and r["instrument_match"] == "match"
    insts = [i for i in bt["instrument"] if i["instrument_id"] == r["instrument_id"]]
    assert len(insts) == 1 and insts[0]["harness_config_hash"] == "aa41f940870d1a0b" and insts[0]["context_window_tokens"] == 16640 and insts[0]["decode_seed_rule"] == "per_item"
    assert insts[0]["eval_id"] == G.EVAL_ID and insts[0]["item_set_id"] == seed_inst["item_set_id"] and insts[0]["token_cap_ladder_json"] == [G.CAP, G.CEILING] and insts[0]["serving_path"] == "lora_native"
    assert insts[0]["instrument_id"] == I.instrument_id(insts[0]) and insts[0]["extra"]["derived_from_instrument_id"] == str(seed_inst["instrument_id"]) and insts[0]["extra"]["minted_by"] == G.NAME
    m = r["extra"]["instrument_minted"]
    assert m["instrument_id"] == r["instrument_id"] and m["seed_instrument_decode_match"] == "match" and set(m["changed_fields"]) >= {"harness_config_hash", "harness_version", "context_window_tokens"}
    assert m["decode_seed_policy"] == "default" and "replicate" in m["why"]
    assert r["run_id"] == I.run_id(LC.CAMPAIGN, fx["keys"][(ARDITI_KEY, G.CAP)], ARDITI_KEY, G.EVAL_ID, r["instrument_id"])
    # the final state item-wise: base 196 scored / 190 correct (0.9694), 4 at cap; the rung re-draws the one eligible item correct -> 197 / 191, 3 loops unresolved
    tn = r["extra"]["three_numbers"]
    assert tn["base_reading"] == {"acc": 190 / 196, "correct": 190, "n_scored": 196} and round(tn["base_reading"]["acc"], 4) == 0.9694 and tn["n_at_cap_8192"] == 4
    assert (tn["n_scored_final"], tn["correct_final"], tn["n_unresolved_neither_hit_nor_miss"], tn["n_loops"], tn["n_at_cap_ceiling"], tn["n_at_cap_not_escalated"]) == (197, 191, 3, 3, 0, 0)
    assert tn["producer_record"] is None and tn["producer_record_version"] == G.RECORD_VERSION_NONE and tn["ceiling_reading"] == {"n": 1, "n_scored": 1, "correct": 1, "still_at_cap": 0}
    assert r["token_cap_tokens"] == G.CEILING and r["token_cap_rung_index"] == 1 and r["n_items_judged"] == 197 and r["extra"]["escalation_state"] == "final_at_ceiling"
    value, bound = _metrics(bt, r["run_id"])
    assert (value["k"], value["n"], value["value"]) == (191, 197, 191 / 197) and value["computed_by"] == "registry"
    assert (bound["k"], bound["n"], bound["ci_lo"], bound["ci_hi"]) == (191, 200, 191 / 200, 194 / 200) and bound["producer_metric_spelling"] is None
    assert bound["computation_id"] == "" and bound["source_uri"].endswith("gen_stats.json") and bound["extra"]["producer_spellings"] is None and "producer_counts" in bound["extra"]
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == r["run_id"]}
    assert len(items) == 200 and sum(1 for i in items.values() if i["label"] == IC.ITEM_LABEL_AT_CAP_UNRESOLVED) == 3 and all(i["trial_index"] == 0 for i in items.values())
    assert sum(1 for i in items.values() if i["extra"]["source_store"] == "ceiling") == 1
    # the accepted attempt, never the first RESULT line; no pins, the tree attested by ladder_state.json
    assert r["cluster_job_id"] == "544615" and r["node_hostname"] == "gpu-8" and r["extra"]["cluster_job_ids"] == {"8192": "544615", "16384": "544730"}
    assert r["extra"]["accepted_attempt"] == {"8192": "a3", "16384": "a1"} and r["extra"]["attempts_before_accepted"] == {"8192": ["a1", "a2"]}
    assert r["extra"]["store_pins"] is None and r["extra"]["store_pins_record"] is None and r["extra"]["tree_attestation"] == G.TREE_ATTESTATION_TARGET and r["code_sha"] == ARDITI_TREE
    assert r["extra"]["producer_summary"].endswith(G.TARGET_SUMMARY_FILE) and r["extra"]["mirror_provenance"].endswith(G.TARGET_PROVENANCE_FILE)
    assert r["extra"]["oos_tuple"] == ARDITI_TUPLES and r["extra"]["single_object"]["operating_point"] == "ar_c0p3" and r["extra"]["declared_role"] is None
    assert r["substrate"] == "cluster_a" and r["started_utc"] == "2026-09-17T17:37:00.000Z" and r["generated_utc"] == "2026-09-17T17:58:23.000Z"
    fc = r["field_confidence_json"]
    assert fc["seed_base"]["confidence"] == "derived" and "no seed base" in fc["seed_base"]["source"] and "minted" in fc["instrument_id"]["source"] and fc["trial_index"]["confidence"] == "derived"
    # no `none` beside it: no contrast, none invented; no state event
    assert "run_contrast" not in bt and "run_state" not in bt and facts["contrasts"][ARDITI_KEY]["anchor"] is None and "roster pairs it" in facts["contrasts"][ARDITI_KEY]["why"]
    assert facts["pairs"][ARDITI_KEY]["trial_rule"] == "DEFAULT_SEED_TRIAL_RULE" and facts["pairs"][ARDITI_KEY]["instrument_minted"] is True and facts["instrument_minted"]["instrument_id"] == r["instrument_id"]
    assert facts["view_legacy"] == TAG_IT and "tag" not in facts and facts["runs"] == 1 and facts["ceiling_merged"] == 1
    # the same store under another root mints the same ids (the store's NAME keys the run, the record's facts the instrument)
    fx2 = _target_mirror(tmp_path, seeds, root=str(tmp_path / "elsewhere" / "mirror"), decls={k: fx[k] for k in fx if k.startswith("arditi_")})
    bt2, _f2, _g2 = _read_group(_arditi_adapter(fx2, seeds), TARGET_GROUP)
    assert _run(bt2, ARDITI_KEY)["run_id"] == r["run_id"] and _run(bt2, ARDITI_KEY)["instrument_id"] == r["instrument_id"]


def test_a_target_store_with_a_replicate_seed_block_is_refused_and_belongs_under_its_tag(tmp_path, seeds):
    fx = _target_mirror(tmp_path, seeds, seed_policy="replicate")
    _bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), TARGET_GROUP)
    why = [x for x in facts["refusals"] if x.startswith(f"{TARGET_GROUP}|{ARDITI_KEY} -- ")]
    assert len(why) == 1 and "replicate policy" in why[0] and "belongs under its tag" in why[0], facts["refusals"]
    assert "run" not in _bt and "instrument" not in _bt


def test_a_target_summary_that_disagrees_with_the_store_refuses_the_pair_naming_the_fact(tmp_path, seeds):
    fx = _target_mirror(tmp_path, seeds, summary_acc=0.95)
    _bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), TARGET_GROUP)
    why = [x for x in facts["refusals"] if x.startswith(f"{TARGET_GROUP}|{ARDITI_KEY} -- ")]
    assert len(why) == 1 and G.TARGET_SUMMARY_FILE in why[0] and "controls.gsm8k.acc 0.95" in why[0], facts["refusals"]
    # without the summary the target files on gen_stats.json alone
    fx2 = _target_mirror(tmp_path, seeds, root=str(tmp_path / "m2"), with_summary=False, decls={k: fx[k] for k in fx if k.startswith("arditi_")})
    bt2, facts2, _ = _read_group(_arditi_adapter(fx2, seeds), TARGET_GROUP)
    assert facts2["n_refusals"] == 0 and _run(bt2, ARDITI_KEY)["extra"]["producer_summary"] is None


def test_another_object_dir_under_the_target_mirror_is_refused_by_name(tmp_path, seeds):
    fx = _target_mirror(tmp_path, seeds, slug="constv2-mix-full--lora_r4_ce")
    _bt, facts, _ = _read_group(_arditi_adapter(fx, seeds), TARGET_GROUP)
    assert facts["n_refusals"] == 1 and "has no declaration here" in facts["refusals"][0] and LORA in facts["refusals"][0] and "run" not in _bt


def test_the_all_positions_rows_reach_ingested_end_to_end_with_the_minted_instrument_landed(tmp_path, seeds):
    fx = _arditi_seed_mirror(tmp_path, seeds, seed_tags=(1,))
    _target_mirror(tmp_path / "t", seeds, root=fx["root"], decls={k: fx[k] for k in fx if k.startswith("arditi_")})
    ad = _arditi_adapter(fx, seeds)
    decl = ad._arditi_declaration("qwen35_4b_s1", "s1", False)
    obj, _op = ad.resolve_arditi_object(decl)
    root = _registry_root(tmp_path, seeds, [obj])
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, lifecycle_state, trial_index, seed_base, n_items_judged, token_cap_tokens, strength_coefficient, is_operating_point, "
                     "operating_point_id_legacy, transfer_kind FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (G.SOURCE,))
    assert rows == [(TAG_IT, ARDITI_KEY, "ingested", 0, None, 197, G.CEILING, 0.3, 1, ARDITI_STAGED, "instruct_direct"),
                    ("qwen35_4b_s1", ARDITI_KEY, "ingested", 1, 1, 197, G.CEILING, 0.3, 1, ARDITI_STAGED, "instruct_direct"),
                    ("qwen35_4b_s1", NONE, "ingested", 1, 1, 198, G.CAP, None, None, None, "none")], rows
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    minted = F.q(root, "SELECT i.instrument_id, i.harness_config_hash FROM instrument i JOIN run r ON r.instrument_id = i.instrument_id WHERE r.view_legacy = ?", (TAG_IT,))
    assert len(minted) == 1 and minted[0][1] == "aa41f940870d1a0b"
    assert F.q(root, "SELECT COUNT(DISTINCT object_id) FROM run WHERE campaign_cell_id_legacy = ?", (ARDITI_KEY,)) == [(1,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (G.SOURCE,)) == [(1,)]


def test_the_checkouts_declarations_name_one_object_on_the_three_seed_views_and_the_matrix_row():
    """The REAL declarations (the eligibility record, the matrix, the all-positions roster / pins): one object id on every seed view and on the
    matrix's own row, staged at the operating point the reader files; the record's out roots are the mirror's directories."""
    ad = G.Gsm8kSeedrepAdapter(root="/nonexistent/gsm8k_seedrep")
    if not all(os.path.isfile(p) for p in (ad.arditi_seed_rows_path, ad.arditi_matrix_path, ad.arditi_roster_path, ad.arditi_pins_path)):
        pytest.skip("the checkout carries no all-positions declarations")
    ids = set()
    for k in (1, 2, 3):
        decl = ad._arditi_declaration(f"qwen35_4b_s{k}", f"s{k}", False)
        obj, op = ad.resolve_arditi_object(decl)
        ids.add(obj.object_id)
        assert (op.rung, op.coefficient, op.unit, obj.staged_id, decl["_role"]) == ("ar_c0p3", 0.3, "raw_coefficient", ARDITI_STAGED, G.ARDITI_ROLE)
        assert decl["out_root"] == f"controls_qwen_seedrep/qwen35_4b_s{k}/{ARDITI_KEY}/{G.BASE_DIR}" and decl["rung_out_root"] == decl["out_root"] + f"__rung{G.CEILING}"
        assert decl["tuple_names"] == ARDITI_TUPLES
    mdecl = ad._arditi_declaration(TAG_IT, "it", True)
    mobj, mop = ad.resolve_arditi_object(mdecl)
    ids.add(mobj.object_id)
    assert len(ids) == 1 and mop.rung == "ar_c0p3" and mdecl["out_root"] == f"{G.TARGET_CONTROLS_DIR}/{TAG_IT}/{ARDITI_KEY}/{G.BASE_DIR}" and mdecl["_role"] is None
    assert G.ARDITI_ROSTER_REL == SC.SINGLE_OBJECT_ROSTER_REL and G.ARDITI_PINS_REL == SC.SINGLE_OBJECT_PINS_REL and G.ARDITI_OBJECT_KEY == "arditi_allpos"


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
    root = _registry_root(tmp_path, seeds, [ad.resolve_object(LORA, f"s{SEED_N}")])
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT campaign_cell_id_legacy, lifecycle_state, n_items_planned, n_items_generated, n_items_judged, token_cap_tokens, trial_index, judged_utc IS NOT NULL, "
                     "ingested_utc IS NOT NULL FROM run WHERE lane_source_legacy = ? ORDER BY 1", (G.SOURCE,))
    assert rows == [(LORA, "ingested", 200, 200, 175, G.CEILING, 0, 1, 1), (NONE, "ingested", 200, 200, 198, G.CAP, SEED_N, 1, 1), (PROMPTED, "ingested", 200, 200, 198, G.CEILING, SEED_N, 1, 1)], rows
    assert F.q(root, "SELECT COUNT(*) FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE r.lane_source_legacy = ?", (G.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM item_result i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (G.SOURCE,)) == [(600,)]
    assert F.q(root, "SELECT COUNT(*) FROM item_result i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ? AND i.value IS NULL AND i.label = ?",
               (G.SOURCE, IC.ITEM_LABEL_AT_CAP_UNRESOLVED)) == [(23 + 2 + 2,)]      # LoRA 22 loops + 1 ceiling; none 1 loop + 1 not escalated; prompted 2 loops
    m = F.q(root, "SELECT r.campaign_cell_id_legacy, rm.computation_id <> '', rm.value, rm.k, rm.n FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                  "WHERE r.lane_source_legacy = ? AND rm.metric_id = 'acc' ORDER BY 1, 2 DESC", (G.SOURCE,))
    assert [x for x in m if x[0] == LORA] == [(LORA, 1, 164 / 175, 164, 175), (LORA, 0, 164 / 200, 164, 200)]
    # the run-level shares landed as run_degeneration rows over the 200 attempted items
    dg = F.q(root, "SELECT d.measure, d.measurement_state, d.numerator, d.denominator, d.rate FROM run_degeneration d JOIN run r ON r.run_id = d.run_id "
                   "WHERE r.campaign_cell_id_legacy = ? AND d.measure IN (?, ?, ?) ORDER BY 1", (LORA, G.RAW_AT_CAP_MEASURE, G.LOOP_MEASURE, G.COHERENT_MEASURE))
    assert dg == [(G.RAW_AT_CAP_MEASURE, "measured", 23, 200, 23 / 200), (G.LOOP_MEASURE, "measured", 22, 200, 22 / 200), (G.COHERENT_MEASURE, "measured", 1, 200, 1 / 200)], dg
    # a second pass over unchanged inputs publishes nothing and the rows stay ingested
    rep2, _ing3, _ing4 = F.run_and_ingest(root, _adapter(fx, seeds), now="2026-09-21T10:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (G.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[G.NAME]()
    assert isinstance(inst, G.Gsm8kSeedrepAdapter) and inst.covers_sources == (G.SOURCE,) and inst.owner == "Overflow" and G.SOURCE == "gsm8k_seedrep"
    assert AB.emission_version_of(G.Gsm8kSeedrepAdapter) == "4" and inst.rng_scope == "none" and inst.roots == (G.ROOT_DEFAULT,)
    assert G.tag_parts("qwen35_4b_s2") == (2, False) and G.tag_parts("qwen35_4b_s2p") == (2, True) and G.tag_parts("qwen35_4b_s2pp") is None
    assert G.producer_key_for("k", "qwen35_4b_s2", False) == "k" and G.producer_key_for("k", "qwen35_4b_s2p", True) == "qwen35_4b_s2p/k"
    assert G.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "gsm8k_seedrep")) and G.LADDER == (8192, 16384) and G.CAMPAIGN == "sdfbase_full22"
    assert set(G.ESCALATION_STATES) <= set(RP.ESCALATION_STATES) and G.ITEM_LABEL_AT_CAP_UNRESOLVED == IC.ITEM_LABEL_AT_CAP_UNRESOLVED == "at_cap_unresolved"
    assert G.INSTRUMENT_SEED == "gsm8k_paired200_inspect_10shot_thinking_on_8192" and G.DECODE_SEED_RULE == "per_item" and G.HEADLINE_METRIC == "acc"
    assert G.slug_to_object_key("constv2-mix-full--lora_r4_ce") == LORA and G.slug_to_object_key("none") == "none" and G.slug_to_object_key("prompted--safety_refusal") == PROMPTED
