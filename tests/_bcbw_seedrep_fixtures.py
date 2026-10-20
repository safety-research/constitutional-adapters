"""A SYNTHETIC mirror shaped like the seed-replicate BigCodeBench-200 / BFCL-200 stores (invented item ids and shas, `TESTHOST`, no
generation text), shared by the two readers' tests (`test_registry_adapter_bigcodebench_seedrep.py`, `test_registry_adapter_bfcl_seedrep.py`):
one seed tag with its unsteered anchor, a fitted ReLU-steer object, a prompted key and the Arditi all-positions object, plus the paired tag's
composed difference_relu object; every store carrying BOTH tasks the way the real ones do; the producer's records (the gate attestation per
row, the launch set, the launched pins and roster, the three-number record) written from the same synthetic truth; a harness roster / pins
and an all-positions roster / pins naming the synthetic objects. `mirror(..., not_run=(...))` leaves a launch-set row without a store and
writes its FAIL gate record, the way the real mirror carries the five rows that had not run."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import _anchor_frame_fixtures as AF
import _b1_fixtures as F
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters.base import AdapterContext

SEED_N = 2
TAG = f"qwen35_4b_s{SEED_N}"
TAG_P = f"{TAG}p"
SEL, SEL_P = f"s{SEED_N}", f"s{SEED_N}p"
ROW_DIR, ROW_DIR_P = f"qwen35_4b-f22{SEL}", f"qwen35_4b-f22{SEL_P}"
NONE = "none"
RELU = "constv2-mix-full__psr_ce_all"
PROMPTED = "prompted__safety_refusal"
ARDITI = "arditi_allpos"
COMPOSED = "constv2-mix-full__psr_ce_all_minus_ctl"
STAGED = f"base64s{SEED_N}__{RELU}"
STAGED_P = f"paired_eb64s{SEED_N}__{COMPOSED}"
STAGED_A = "ar_c0p3__arditi_allpos"
OBJ = hashlib.sha256(b"synthetic bcbw seedrep relu s2").hexdigest()
OBJ_P = hashlib.sha256(b"synthetic bcbw seedrep composed s2p").hexdigest()
OBJ_A = hashlib.sha256(b"synthetic bcbw seedrep arditi allpos").hexdigest()
TREE = hashlib.sha1(b"synthetic bcbw seedrep tree").hexdigest()
SHA7 = TREE[:7]
MODEL_HF = "Qwen/Qwen3.5-4B"
NOW = "2026-09-23T09:00:00.000Z"
SEED_RULE = "sha256(item_id NUL sample_idx)[:4] big-endian mod 2^31-1"
#: the two tasks every store carries: (task, record spelling, outcomes file, cap, protocol T, clamp override?, item set key, item set of)
TASKS = {"bigcodebench": ("inspect/bigcodebench_200", "bigcodebench_200", "inspect__bigcodebench_200.json", 4096, 0.0, False, "bcb_item_set", 992),
         "bfcl": ("inspect/bfcl_200", "bfcl_200", "inspect__bfcl_200.json", 1024, 0.001, True, "bfcl_item_set", 3981)}
ITEM_IDS = {"bigcodebench": [f"{i}" for i in range(B.N_ITEMS)], "bfcl": [f"exec_{i:04d}" for i in range(B.N_ITEMS)]}


@dataclasses.dataclass(frozen=True)
class Shape:
    """The task table a synthetic store is written with: ``tasks`` (per eval: task, record task, outcomes file, cap, protocol T, clamp?, item set key, of),
    the item count, the item ids per eval, the LABEL the harness stamps (`pinned200` / `full`) and the manifest's ``tinycap`` item-set blocks. The default
    :data:`SHAPE_200` is the 200-item stores' (every writer below keeps its bytes); `_bcbw_full_fixtures` binds the full-suite shape."""
    tasks: Dict[str, tuple]
    n_items: int
    item_ids: Dict[str, List[str]]
    item_set_label: str
    item_set_blocks: Dict[str, dict]


SHAPE_200 = Shape(TASKS, B.N_ITEMS, ITEM_IDS, "pinned200",
                  {"bcb_item_set": {"ids_sha256": hashlib.sha256(b"bcb ids").hexdigest(), "n": B.N_ITEMS, "of": 992, "seed": 0},
                   "bfcl_item_set": {"ids_sha256": hashlib.sha256(b"bfcl ids").hexdigest(), "n": B.N_ITEMS, "of": 3981, "seed": 0}})
_REVISION: Optional[str] = None
_SI_SHA: Optional[str] = None
#: the declared prompts' text shas by si id (si_texts.yaml), for every prompted key the grammar knows (`lineage_common.PROMPTED_SI`)
_SI_SHAS: Dict[str, str] = {}
#: "keep the module default" sentinel for the keyword parameters below (None is a value a caller may want to write)
KEEP = object()


def bind_seeds(seeds: LC.Seeds) -> None:
    global _REVISION, _SI_SHA
    _REVISION = str(seeds.models["qwen35_4b_it"]["hf_revision"])
    _SI_SHA = str(seeds.si_object_id("safety_refusal")).split(":", 1)[1]
    for si in sorted(set(LC.PROMPTED_SI.values())):
        oid = seeds.si_object_id(si)
        if oid is not None:
            _SI_SHAS[si] = str(oid).split(":", 1)[1]


def system_sha_of(okey: str) -> Optional[str]:
    """The 16-hex prefix a prompted store's manifest attests (`resolved_config.system_sha`): the declared prompt's text sha; None for any other key."""
    si = LC.PROMPTED_SI.get(okey)
    return _SI_SHAS[si][:16] if si is not None and si in _SI_SHAS else None


def ctx(conn=None) -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=F.producer(), conn=conn)


def primed_frame(ad) -> AF.Frame:
    """The steady-state frame of an in-memory read (`_anchor_frame_fixtures`): since emission 3 every contrast of the six capability readers is decided by
    the pass's anchor currency, and on an EMPTY frame every comparison parks (no provisional anchors: the first pass over a fresh registry files no
    contrast). The in-memory tests describe the pass AFTER the fold, so they read on a frame primed from the adapter's own emissions over every group
    (every run current, complete, off-roster) -- and, for a SHARED reader, with every solo run the shared rows name (`extra.solo_run_id`) made current by
    its currency row alone (another source's run: checked for currency, never a candidate), the way the solo source's fold would hold it."""
    frame = AF.Frame()
    records = AF.priming_read(ad, ctx(frame.con))
    AF.load_records(frame, records)
    solo_ids = {str(r["payload"]["extra"]["solo_run_id"]) for r in records if r.get("record_type") == "run" and (r["payload"].get("extra") or {}).get("solo_run_id")}
    AF.make_current(frame, sorted(solo_ids))
    return frame


def primed_conn(ad, conn=None):
    """`conn` when the test gave one; else the connection of a frame primed from the adapter's own emissions (:func:`primed_frame`)."""
    return conn if conn is not None else primed_frame(ad).con


def write(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return path


# --------------------------------------------------------------------------------------------
# the synthetic truth of one (tag, object key, task) and the files written from it
# --------------------------------------------------------------------------------------------

def spec(*, n_correct: int, n_wrong: int, at_cap_correct: int = 0, at_cap_wrong: int = 0, at_cap_unscored: int = 0, below_cap_unscored: int = 0, eval_id: str = "bigcodebench",
         shape: Shape = SHAPE_200) -> dict:
    """`states[item] = (state, value)`: scored | at_cap_scored (at the cap, graded) | at_cap_unscored (at the cap, ungraded) | below_cap_unscored; `n_items` the shape's."""
    states = {}
    ids = iter(shape.item_ids[eval_id])
    for _ in range(n_correct):
        states[next(ids)] = ("scored", 1.0)
    for _ in range(n_wrong):
        states[next(ids)] = ("scored", 0.0)
    for _ in range(at_cap_correct):
        states[next(ids)] = ("at_cap_scored", 1.0)
    for _ in range(at_cap_wrong):
        states[next(ids)] = ("at_cap_scored", 0.0)
    for _ in range(at_cap_unscored):
        states[next(ids)] = ("at_cap_unscored", None)
    for _ in range(below_cap_unscored):
        states[next(ids)] = ("below_cap_unscored", None)
    assert len(states) == shape.n_items, len(states)
    return {"states": states, "n_items": shape.n_items}


def counts(sp: dict) -> dict:
    st = sp["states"]
    n_items = int(sp.get("n_items", B.N_ITEMS))
    scored = {i: v for i, (s, v) in st.items() if s in ("scored", "at_cap_scored")}
    at_cap = [i for i, (s, _v) in st.items() if s.startswith("at_cap")]
    correct = int(sum(scored.values()))
    return {"n_scored": len(scored), "correct": correct, "acc": correct / len(scored), "at_cap": at_cap, "n_at_cap": len(at_cap),
            "at_cap_unscored": [i for i, (s, _v) in st.items() if s == "at_cap_unscored"], "at_cap_scored": [i for i, (s, _v) in st.items() if s == "at_cap_scored"],
            "below": [i for i, (s, _v) in st.items() if s == "below_cap_unscored"], "n_unscored": n_items - len(scored)}


def _outcome(iid: str, st: str, v, seed: int, cap: int, eval_id: str) -> dict:
    at_cap = st.startswith("at_cap")
    scored = st in ("scored", "at_cap_scored")
    stop = "max_tokens" if at_cap else ("tool_calls" if eval_id == "bfcl" else "stop")
    return {"at_cap": at_cap, "attempt": 1, "completion_tokens": (cap if at_cap else 300), "content_empty_with_reasoning": False, "item_id": iid, "prompt_tokens": 305, "sample_idx": 0,
            "sampling_ref": "676cc39ce315", "scored": scored, "seed": seed, "stop_reason": stop, "stray_open_tag": False, "think_block_present": False, "think_block_unterminated": False,
            "unparseable": False, "value": (v if scored else None)}


def class_of(okey: str, paired: bool) -> str:
    """The launch set's class grammar (the readers'): the Arditi key `arditi`; `none` `anchor` on EITHER tag (the paired tag's own anchor); a
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
    if okey == "prompted__si_constitution":
        return "prompted__constitution"
    if okey == PROMPTED:
        return "prompted__safety"
    return "si_" + okey.replace("-", "_")


def suffix_of(tag: str, okey: str, launch_tag: Optional[str] = None, tree: Optional[str] = None, suffix_extra: str = "") -> str:
    """`bcbw-<launch tag>-<sha7>`; `suffix_extra` appends a relaunch tag (`-r2`, the addendum of 2026-09-24)."""
    return f"bcbw-{launch_tag or B.launch_tag_of(tag, okey)}-{(tree or TREE)[:7]}{suffix_extra}"


def store_files(root: str, tag: str, okey: str, specs: Dict[str, dict], *, obj_sha, staged, tree: str, ihash8: str, chash8: str, seed_n: int, roster_sha256: str,
                max_lora_rank: int = 8, call_id: str = "fc-01SYNTHETIC0000000000000000", ended: str = "2026-09-23T04:17:11Z", launched: str = "2026-09-23T03:55:56+00:00",
                attest_sha: str = None, lane_sha: str = None, result_complete: bool = True, replicate_block: bool = False,
                model_hf: Optional[str] = None, revision: Optional[str] = None, model_key: str = "qwen35_4b", fit_model_hf: str = "Qwen/Qwen3.5-4B-Base",
                row_dir: Optional[str] = None, launch_tag: Optional[str] = None, tool_calls: bool = True, shape: Shape = SHAPE_200, suffix_extra: str = "",
                fit_transfer: Optional[bool] = None, max_model_len: int = 16384, max_num_seqs: int = 31) -> str:
    """One store directory (both tasks) and the named files the readers open. `attest_sha` makes the manifest attest another object than the pins;
    `lane_sha` alters `lane_roster.json:object_sha256` alone. The model parameters default to the three-seed Qwen family's; an appendix family's
    fixture (`_bcbw_allmodels_fixtures`) passes its own served id / revision / model key, and the `main` canary its own row dir and launch tag."""
    model_hf = model_hf or MODEL_HF
    revision = revision or _REVISION
    row_dir = row_dir or B.row_dir_of(tag)
    launch_tag = launch_tag or B.launch_tag_of(tag, okey)
    suffix = suffix_of(tag, okey, launch_tag, tree, suffix_extra)
    d = os.path.join(root, B.STORES_HEAD, row_dir, okey, B.THINKING_DIR, suffix)
    slug = okey.replace("__", "--", 1)
    store_key = f"{slug}__controls__t0__a0__{ihash8}__{chash8}"
    kd = os.path.join(d, "cells", store_key)
    steered = okey not in (NONE, PROMPTED, "prompted__si_constitution")
    a_sha = (attest_sha or obj_sha) if steered else None
    l_sha = (lane_sha or a_sha) if steered else None
    ident = B.STEERED_IDENTITY if steered else B.PASSTHROUGH_IDENTITY
    hid = harness_id(okey)
    rows: List[str] = []
    gen_stats: Dict[str, dict] = {}
    by_task_sampling: Dict[str, dict] = {}
    n_items, label = shape.n_items, shape.item_set_label
    for eval_id, (task, record_task, ofile, cap, temp, _clamp, item_key, of) in shape.tasks.items():
        sp = specs[eval_id]
        c = counts(sp)
        outcomes = [_outcome(iid, st, v, 1000 + k, cap, eval_id) for k, (iid, (st, v)) in enumerate(sp["states"].items())]
        write(os.path.join(kd, "outcomes", ofile), {"item_set": label, "items": outcomes, "max_tokens": cap, "n_at_cap": c["n_at_cap"], "n_attempted_items": n_items,
                                                     "n_attempted_samples": n_items, "n_completion_tokens": n_items, "n_scored": c["n_scored"], "n_scored_samples": c["n_scored"],
                                                     "n_stop_reason": n_items, "n_think_block_present": 0, "n_think_block_unterminated": 0, "n_unscored": c["n_unscored"],
                                                     "n_unscored_samples": c["n_unscored"], "n_unterminated_not_at_cap": 0, "producer_tree_sha": tree, "samples_per_item": 1, "task": task,
                                                     "think_split_mode": "record_only", "thinking": "off", "unterminated_policy": "unscored"})
        gs = {"acc": c["acc"], "n": n_items, "n_requests": n_items, "samples_per_item": 1, "task": task, "item_set": label, "framework": "inspect_evals-0.18.0", "engine": "vllm",
              "route": f"petri__{hid}", "thinking": "off", "render": "native", "n_scored": c["n_scored"], "n_unscored": c["n_unscored"], "at_cap_rate": c["n_at_cap"] / n_items,
              "n_at_cap": c["n_at_cap"], "max_tokens": cap, "temperature": temp, "top_p": 1.0, "top_k": 0, "protocol_temperature": temp, "sampling_ref": "676cc39ce315",
              "unterminated_policy": "unscored", "wall_clock_s": 85.1}
        gen_stats[task] = gs
        write(os.path.join(kd, "summary", ofile), dict(gs))
        write(os.path.join(kd, "done", ofile), {"finished_at": 1790137006.659015, "n_items": n_items, "n_planned": n_items, "n_rejected": 0, "n_requests": n_items, "n_rows": n_items,
                                                 "task": task, "wall_clock_s": 85.1})
        for k, o in enumerate(outcomes):
            if not o["scored"]:
                continue
            rows.append(json.dumps({"arm_name": slug, "attempt": 0, "coeff": 1.0, "created_at": 1790137006.0 + k, "domain": "inspect_gen", "engine": "vllm", "env_digest": "108112cfb52ae5df",
                                    "extra_json": json.dumps({"answer": "synthetic", "stop_reason": o["stop_reason"], "seed": o["seed"]}), "harness_config_hash": chash8 + "5c84d9c5",
                                    "intervention_hash": ihash8 + "091aef6f", "item_id": o["item_id"], "kind": "cell", "metric": "acc", "model_id": model_hf, "model_revision": revision,
                                    "run_id": f"served-controls-{okey}", "seed": 0, "slurm_job_id": None, "task": task, "task_version": "inspect_evals-0.18.0", "trial_id": 0, "value": o["value"]}))
        rows.append(json.dumps({"item_id": "__aggregate__", "kind": "cell", "metric": "acc_aggregate", "task": task, "value": c["acc"], "extra_json": None}))
        rows.append(json.dumps({"item_id": "__aggregate__", "kind": "cell", "metric": "n", "task": task, "value": float(n_items), "extra_json": None}))
        by_task_sampling[task] = {"effective": {"frequency_penalty": 0.0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "temperature": temp, "top_k": 0, "top_p": 1.0},
                                  "executed": {"temperature": temp, "top_k": 0, "top_p": 1.0}, "executed_reason": ("equal (clamp override applied)" if eval_id == "bfcl" else "equal"),
                                  "protocol_temperature": temp}
    write(os.path.join(kd, "rows.jsonl"), "\n".join(rows) + "\n")
    write(os.path.join(kd, "gen_stats.json"), gen_stats)
    write(os.path.join(kd, "evidence.json"), {"serving_identity": ident, "engine_stamp": {"object_sha": (a_sha[:16] if steered else None)}})
    ledger = f"{B.VOLUME_HEAD}{B.rel_store(row_dir, okey, suffix)}/usage_ledger.jsonl"
    seeds_block = {"inspect": {"rule": SEED_RULE, "samples_per_item": {t[0]: 1 for t in shape.tasks.values()}}}
    if replicate_block:
        seeds_block["inspect"]["replicate"] = {"policy": "replicate", "seed_base": seed_n}
    man = {"cell_id": okey, "harness_cell_id": hid, "model_id": model_hf, "model_revision": revision, "serving_identity": ident, "serving_path": ("lora" if steered else "passthrough"),
           "serving_engine": "lora_native", "steer_positions": "all", "engine": "vllm", "virtual_model": f"petri__{hid}",
           "fit_provenance": {"object_sha": (a_sha[:16] if steered else None), "model_id": ((model_hf if okey == ARDITI else fit_model_hf) if steered else None),
                              "transfer": ((steered and okey != ARDITI) if fit_transfer is None else fit_transfer), "fit_kind": ("class_b" if okey == ARDITI else ("document" if steered else None)),
                              "status": ("verified" if okey == ARDITI else ("document_fit" if steered else None)), "served_model_id": model_hf, "served_revision": revision},
           "producer": {"driver": "served_capability_run", "tree_sha": tree}, "row_constants": {"slurm_job_id": None, "env_digest": "108112cfb52ae5df", "arm_name": slug, "seed": 0},
           "harness_config_hash": chash8 + "5c84d9c5", "intervention_hash": ihash8 + "091aef6f",
           "geometry": {"max_model_len": max_model_len, "max_num_seqs": max_num_seqs, "source": "endpoint.json+served_geometry.json"},
           "seeds": seeds_block, "frameworks": {"inspect_ai": "0.3.260", "inspect_evals": "0.18.0", "client_lock_sha256": hashlib.sha256(b"synthetic client lock").hexdigest()},
           "sampling": {"mode": "protocol", "model_key": model_key, "by_task": by_task_sampling, "source": "synthetic protocol", "seed_rule": {"inspect": SEED_RULE}},
           "resolved_config": {"cell_id": okey, "system_sha": system_sha_of(okey), "serving": {"serving_engine": "lora_native", "serving_identity": ident}},
           "provenance": {"artifact_sha256": a_sha, "cell_id": okey, "class": ("B" if okey == ARDITI else ("A" if steered else "baseline")), "serving": {"serving_identity": ident}},
           "route": {"bank_view": None, "object_artifact_sha256": None, "object_artifact_sha256_by_cell": None, "serving_identity": ident, "serving_engine": "lora_native", "tool_calls": tool_calls,
                     "bcb_sandbox": {"provider": "modal_owned", "app_name": "inspect_modal_sandbox", "image": "ghcr.io/synthetic/sandbox@sha256:" + "0" * 64,
                                     "owner": {"cell": okey, "function_call_id": call_id, "run_id": f"{SHA7}-synthetic"}}},
           "thinking": {"mode": "off", "enable_thinking": False, "render": "native", "by_task": {t[0]: "off" for t in shape.tasks.values()}},
           "tinycap": {**{k: dict(v) for k, v in shape.item_set_blocks.items()},
                       "served": {"agent_loop_entry": True, "serving_engine": "lora_native", "tool_calls": tool_calls, "max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
                                  "sampling_clamp_override": {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"}},
                       "thinking": {"mode": "off", "render": "native"}},
           "usage_ledger": ledger}
    write(os.path.join(kd, "manifest.json"), man)
    write(os.path.join(d, "RESULT"), (f"{B.RESULT_HEAD} name=tinycap-synthetic-{slug} rc=0 result=COMPLETE status=OK sha={tree} at={ended}\n" if result_complete
                                      else f"{B.RESULT_HEAD} name=tinycap-synthetic-{slug} rc=1 result=INCOMPLETE status=INCOMPLETE sha={tree} at={ended}\n"))
    write(os.path.join(d, B.LAUNCHED_ENTRY_FILE), {"cell_id": okey, "entry_id": entry_id_of(okey), "launched_at": launched, "object_generation": ("batch64" if steered else None),
                                                "object_sha256": l_sha, "roster_sha256": roster_sha256, "snapshot_sha256": hashlib.sha256(b"snapshot").hexdigest(),
                                                "superseded_by_key": None, "wave": launch_tag})
    write(os.path.join(d, B.GEOMETRY_FILE), {"serving_path": "lora_native", "serving_engine": "lora_native", "standup_argv": ["python", "capctrl_serving_up.py", "--tool-calls"],
                                             "endpoint": {"engine": "lora_native", "zero_adapter_only": (not steered), "adapter_bank": f"/vol/tinycap/multilora_adapters/{model_key}",
                                                          "attested_max_model_len": max_model_len, "attested_max_num_seqs": max_num_seqs, "node": None, "tool_calls": True, "reasoning_parser": None,
                                                          "lora_config": {"max_lora_rank": max_lora_rank, "max_loras": 2, "target_modules": ["o_proj"]},
                                                          "sampling_clamp_override": {"from": 0.01, "to": 0.001}}})
    write(os.path.join(d, "evidence.json"), {"serving_identity": ident, "cell_id": okey})
    write(os.path.join(d, "protocol_check.json"), {"ok": True, "tasks": list(t[0] for t in shape.tasks.values())})
    write(os.path.join(d, "resolved_sampler.json"), {"verdict": "AGREES"})
    write(os.path.join(d, "usage_ledger.jsonl"), json.dumps({"task": next(iter(shape.tasks.values()))[0], "finish_reason": "stop"}) + "\n")     # a one-task shape may hold BFCL alone
    summary = {"cell_id": okey, "provenance": {"artifact_sha256": a_sha, "cell_id": okey}, "controls": {}}
    for eval_id, (task, _rt, _of, cap, _t, _c, _ik, _o) in shape.tasks.items():
        c = counts(specs[eval_id])
        summary["controls"][task] = {"acc": c["acc"], "n": n_items, "n_scored": c["n_scored"], "n_at_cap": c["n_at_cap"], "max_tokens": cap, "thinking": "off", "task": task, "engine": "vllm"}
    write(os.path.join(d, okey, B.CONTROLS_SUMMARY_FILE), summary)
    return store_key


def entry_id_of(okey: str) -> str:
    return {NONE: "none", RELU: "const_psr", PROMPTED: "si_safety", "prompted__si_constitution": "const_si", ARDITI: "arditi_allpos", COMPOSED: "diff_psr"}.get(okey, okey)


def gate_record(tag: str, okey: str, *, tree: str, store_key: str, obj_sha, steered: bool, max_lora_rank: int, verdict: str = "PASS", completed: bool = True, engine: str = "lora_native",
                row_dir: Optional[str] = None, launch_tag: Optional[str] = None, model_key: str = "qwen35_4b", shape: Shape = SHAPE_200, suffix_extra: str = "",
                compared_against: Optional[dict] = None, cross_tree: Optional[bool] = None) -> dict:
    launch_tag = launch_tag or B.launch_tag_of(tag, okey)
    suffix = suffix_of(tag, okey, launch_tag, tree, suffix_extra)
    ident = B.STEERED_IDENTITY if steered else B.PASSTHROUGH_IDENTITY
    tasks = [t[1] for t in shape.tasks.values()]
    ofiles = [t[2] for t in shape.tasks.values()]
    problems = [] if verdict == "PASS" else [f"leg {t}: no done/{o} — the leg has not COMPLETED (a standup manifest + endpoint alone is not PASS)" for t, o in zip(tasks, ofiles)]
    return {"wave": launch_tag, "cell": okey, "suffix": suffix, "row": (row_dir or B.row_dir_of(tag)), "sha": tree, "legs": tasks, "serving_path_requested": "lora_native",
            "read_at": "2026-09-23T04:51:30Z", "completed_legs": (tasks if completed else []), "requested_from": "the roster header (a REQUEST, never a fact)",
            "attested_from": [f"{suffix}/cells/{store_key}/manifest.json", f"{suffix}/{B.GEOMETRY_FILE}#endpoint", f"{suffix}/{B.LAUNCHED_ENTRY_FILE}", f"{suffix}/RESULT"],
            "facts": [{"manifest": f"{suffix}/cells/{store_key}/manifest.json", "route.serving_engine": engine, "serving_path": ("lora" if steered else "passthrough"), "route.bank_view": None,
                       "provenance.artifact_sha256": (obj_sha if steered else None), "route.object_artifact_sha256_by_cell": None, "producer.tree_sha": tree, "serving_identity": ident,
                       "launch_flags": {"serving_engine": engine, "tool_calls": True, "reduction_mode": "not_applicable", "enable_lora": True, "max_model_len": 16384, "max_num_seqs": 31},
                       "recorded_flags": {"max_lora_rank": max_lora_rank, "lora_target_modules": ["o_proj"], "zero_adapter_only": (not steered), "adapter_bank": f"/vol/tinycap/multilora_adapters/{model_key}"},
                       "flags_source": f"manifest.route + {suffix}/{B.GEOMETRY_FILE}#endpoint"}],
            "records": {}, "done_markers": ([f"{suffix}/cells/{store_key}/done/{o}" for o in ofiles] if completed else []), "verdict": verdict, "problems": problems,
            **({"compared_against": dict(compared_against)} if compared_against is not None else {}), **({"cross_tree": cross_tree} if cross_tree is not None else {})}


def three_row(tag: str, sel: str, okey: str, sp: dict, *, eval_id: str, obj_sha, tree: str, store_key: str, steered: bool, anchor_spec: dict = None,
              seed_n=KEEP, row_dir: Optional[str] = None, launch_tag: Optional[str] = None, tag_spelling: Optional[str] = None, cell_slug: Optional[str] = None,
              launched_as: Optional[dict] = None, shape: Shape = SHAPE_200, suffix_extra: str = "") -> dict:
    """The producer's three-number row. `tag_spelling` / `sel` / `seed_n` let the `main` canary's row carry the producer's own spelling (its `main` tag and
    selection, no seed) beside the store's real row dir and launch tag."""
    task, _rt, ofile, cap, _t, _c, _ik, _o = shape.tasks[eval_id]
    c = counts(sp)
    n_items = shape.n_items
    row_dir = row_dir or B.row_dir_of(tag)
    suffix = suffix_of(tag, okey, launch_tag, tree, suffix_extra)
    spelled = tag_spelling or tag
    row = {"key": f"{spelled}/{okey}", "tag": spelled, "view": sel, "seed": (SEED_N if seed_n is KEEP else seed_n), "paired": tag.endswith("p"), "cell_id": okey, "cell_slug": (cell_slug or okey), "arm": LC.cell_grammar_legacy(okey)["paper_label"],
           "status": "complete", "cap": cap, "is_rung": False, "merged": False, "store": B.rel_store(row_dir, okey, suffix),
           "result_line": f"{B.RESULT_HEAD} name=tinycap-synthetic-{okey.replace('__', '--', 1)} rc=0 result=COMPLETE status=OK sha={tree} at=2026-09-23T04:17:11Z",
           "engine": "lora_native", "serving_identity": (B.STEERED_IDENTITY if steered else B.PASSTHROUGH_IDENTITY), "producer_tree": tree, "bank_view": None,
           "n_attempted": n_items, "n_scored": c["n_scored"], "correct": c["correct"], "value": round(c["acc"], 4), "bound": round(c["correct"] / n_items, 4),
           "at_cap_n": c["n_at_cap"], "at_cap_share": round(c["n_at_cap"] / n_items, 4), "n_unscored": c["n_unscored"], "delta_value_vs_none": None, "note": None,
           "object_sha256": (obj_sha if steered else None), "object_sha256_source": ("provenance.artifact_sha256" if steered else None), "draws": n_items, "samples_per_item": 1,
           "max_tokens": cap, "summary": f"cells/{store_key}/summary/{ofile}"}
    if anchor_spec is not None:
        row["delta_value_vs_none"] = round(c["acc"] - counts(anchor_spec)["acc"], 4)
    if launched_as is not None:
        row["launched_as"] = dict(launched_as)
    return row


def launch_row(sel: str, okey: str, tag: str, *, pin_key, expected, paired: bool, index: int, seed_base=KEEP, launch_tag: Optional[str] = None,
               launched_as: Optional[dict] = None) -> dict:
    """A launch-set row. `sel` / `tag` / `seed_base` are the row's own spelling (the `main` canary may spell `main` / `<model key>_main` / null);
    `launch_tag` the store's launch tag when it differs from the tag's (the canary's `<model key>_main`)."""
    cls = class_of(okey, paired)
    row = {"tuple_id": f"{sel}/{okey}", "view": sel, "cell": okey, "paper_label": LC.cell_grammar_legacy(okey)["paper_label"], "class": cls,
           "block": ("composed" if paired else ("arditi" if cls == "arditi" else "seed")), "fitted": (okey not in (NONE, PROMPTED, "prompted__si_constitution")),
           "seed_base": (SEED_N if seed_base is KEEP else seed_base),
           "pin_key": pin_key, "tag": tag, "cell_slug": okey.replace("__", "--", 1), "table_index": index, "object_sha256_expected": expected, "census_status": "run",
           "bcbw_wave": (launch_tag or B.launch_tag_of(tag, okey)), "bcbw_cell": okey}
    if launched_as is not None:
        row["launched_as"] = dict(launched_as)
    return row


#: per eval: the synthetic truth of every row (values differ per task the way the real stores' do)
SPECS = {
    "bigcodebench": {NONE: dict(n_correct=95, n_wrong=102, at_cap_correct=1, at_cap_wrong=2), RELU: dict(n_correct=80, n_wrong=53, at_cap_correct=2, at_cap_wrong=60, at_cap_unscored=5),
                     PROMPTED: dict(n_correct=88, n_wrong=110, at_cap_wrong=2), ARDITI: dict(n_correct=88, n_wrong=112), COMPOSED: dict(n_correct=79, n_wrong=115, at_cap_wrong=4, below_cap_unscored=2)},
    "bfcl": {NONE: dict(n_correct=166, n_wrong=34), RELU: dict(n_correct=99, n_wrong=36, at_cap_correct=2, at_cap_wrong=63), PROMPTED: dict(n_correct=159, n_wrong=41),
             ARDITI: dict(n_correct=153, n_wrong=46, at_cap_wrong=1), COMPOSED: dict(n_correct=119, n_wrong=75, at_cap_wrong=6)},
}
#: (object key, paired?, staged id, object sha, ihash8, chash8, max_lora_rank)
ROWS = [(NONE, False, None, None, "2ff9cdff", "ed8a6ada", 1),
        (RELU, False, STAGED, OBJ, "d69810e6", "466f1aaf", 1),
        (PROMPTED, False, None, None, "aa7fa55b", "27ad6f72", 1),
        (ARDITI, False, STAGED_A, OBJ_A, "92b66dce", "de23b220", 1),
        (COMPOSED, True, STAGED_P, OBJ_P, "f0bb156d", "01671a70", 1)]
#: the PAIRED tag's own `none` (`mirror(paired_none=True)`): the seed tag's `none` store key byte for byte under the paired row dir (the tag-prefixed
#: producer key keeps the ids apart); fewer correct than the seed tag's `none`, so an anchor's accuracy over the pairs tells which `none` a contrast used
ROW_P = (NONE, True, None, None, "2ff9cdff", "ed8a6ada", 1)
SPECS_P = {"bigcodebench": dict(n_correct=60, n_wrong=137, at_cap_correct=1, at_cap_wrong=2), "bfcl": dict(n_correct=120, n_wrong=80)}


def census_row_table(rows: Sequence[dict], sel: str, sel_p: str) -> Dict[str, object]:
    """The census's launch + carried shape (2026-09-23 14:19Z): the paired tag's `none` rows are this pass's `launch` (census_status run), every other row
    is `carried` (census_status carry); each row carries the census's anchor declaration; `counts` sums the table."""
    def decorate(r: dict) -> dict:
        own_none = r["cell"] == NONE
        return dict(r, anchor_tuple=(None if own_none else f"{r['view']}/{NONE}"), anchor_fallback=(f"{sel}/{NONE}" if (r["view"] == sel_p and not own_none) else None))
    run_rows = [decorate(r) for r in rows if r["view"] == sel_p and r["cell"] == NONE]
    carried = [dict(decorate(r), census_status="carry") for r in rows if not (r["view"] == sel_p and r["cell"] == NONE)]
    return {"launch": run_rows, "carried": carried, "carried_late": [], "counts": {"table": len(rows), "launch": len(run_rows), "carried": len(carried), "duplicates_for_user": 0}}


def mirror(tmp_path, seeds: LC.Seeds, *, eval_id: str, root: str = None, not_run: Sequence[str] = (), drop_gate: Sequence[str] = (), gate_override=None, three_override=None,
           launch_override=None, attest_sha: dict = None, pins_sha: dict = None, launched_pins_sha: dict = None, replicate_block: bool = False, result_incomplete: Sequence[str] = (),
           paired_none=False, launch_layout: str = "launch") -> dict:
    """A synthetic mirror root for one reader: the seed tag's four stores and the paired tag's composed store (both tasks each), the gate records, the launch
    set, the launched pins and roster, the three-number records of BOTH evals, and the harness / all-positions roster and pins naming the synthetic objects.
    `not_run` names SEED-tag object keys whose launch-set row gets no store and a FAIL gate record; `drop_gate` leaves a present store without a gate record;
    `paired_none` adds the paired tag's OWN `none` (`True`: its store, gate and records; `"not_run"`: the launch-set row and a FAIL gate record alone);
    `launch_layout="carried"` writes the census's launch + carried shape of 2026-09-23 (the paired `none` rows under `launch`, census_status run; every
    other row under `carried`, census_status carry; the census's anchor declaration on each; `counts`)."""
    bind_seeds(seeds)
    root = root or str(tmp_path / f"{eval_id}_seedrep")
    recs = os.path.join(root, B.RECORDS_DIR)
    import yaml
    hp = dict(pins_sha or {})
    lp = dict(launched_pins_sha or {})
    # the launched roster (its sha256 is what every lane_roster.json names)
    launched_roster = {"lane": "bcb-bfcl-200-widen", "legs": [t[0] for t in TASKS.values()], "sampling": "protocol", "thinking": "off", "pins": "configs/sdfbase_full22_pins.json",
                       "serving_path": "lora_native", "waves": {}}
    rows = list(ROWS) + ([ROW_P] if paired_none else [])
    for okey, paired, staged, sha, _i, _c, _m in rows:
        tag = TAG_P if paired else TAG
        launch_tag = B.launch_tag_of(tag, okey)
        blk = launched_roster["waves"].setdefault(launch_tag, {"family": "qwen35_4b", "view": tag, "legs": {t[1]: "runnable" for t in TASKS.values()}, "entries": {}})
        entry = {"cell_id": okey, "status": "runnable", "paper_arm": entry_id_of(okey)}
        if staged:
            entry["object"] = {"staged_id": staged, "sha256": lp.get(okey, sha), "kind": "synthetic"}
        blk["entries"][entry_id_of(okey)] = entry
    roster_path = write(os.path.join(recs, B.ROSTER_FILE), yaml.safe_dump(launched_roster, sort_keys=False))
    roster_sha256 = hashlib.sha256(open(roster_path, "rb").read()).hexdigest()
    three = {ev: {"schema": B.THREE_NUMBERS_SCHEMA, "eval": ev, "source": f"{ev}_seedrep", "written": NOW, "contract": "synthetic", "seed_reading": B.SEED_AXIS_RULE, "rows": {}} for ev in TASKS}
    launch: Dict[str, List[dict]] = {ev: [] for ev in TASKS}
    keys: Dict[str, str] = {}
    specs = {ev: {k: spec(eval_id=ev, **v) for k, v in SPECS[ev].items()} for ev in TASKS}
    specs_p = {ev: spec(eval_id=ev, **SPECS_P[ev]) for ev in TASKS}
    paired_none_key = None
    for i, (okey, paired, staged, sha, ihash8, chash8, mlr) in enumerate(rows):
        tag = TAG_P if paired else TAG
        sel = SEL_P if paired else SEL
        steered = staged is not None
        is_paired_none = paired and okey == NONE
        row_specs = {ev: (specs_p[ev] if is_paired_none else specs[ev][okey]) for ev in TASKS}
        for ev in TASKS:
            launch[ev].append(launch_row(sel, okey, tag, pin_key=staged, expected=(lp.get(okey, sha) if staged else None), paired=paired, index=i))
        store_key = f"{okey.replace('__', '--', 1)}__controls__t0__a0__{ihash8}__{chash8}"
        if (okey in not_run and not is_paired_none) or (is_paired_none and paired_none == "not_run"):
            gate = gate_record(tag, okey, tree=TREE, store_key=store_key, obj_sha=sha, steered=steered, max_lora_rank=mlr, verdict="FAIL", completed=False)
            write(os.path.join(recs, f"gate_{B.launch_tag_of(tag, okey)}_{okey}_{suffix_of(tag, okey)}.json"), gate)
            continue
        key = store_files(root, tag, okey, row_specs, obj_sha=sha, staged=staged, tree=TREE, ihash8=ihash8, chash8=chash8, seed_n=SEED_N,
                          roster_sha256=roster_sha256, max_lora_rank=mlr, attest_sha=(attest_sha or {}).get(okey), replicate_block=replicate_block,
                          result_complete=(okey not in result_incomplete))
        assert key == store_key
        if is_paired_none:
            paired_none_key = key
        else:
            keys[okey] = key
        if okey not in drop_gate:
            gate = gate_record(tag, okey, tree=TREE, store_key=store_key, obj_sha=(attest_sha or {}).get(okey, sha), steered=steered, max_lora_rank=mlr)
            if gate_override:
                gate_override(okey, gate)
            write(os.path.join(recs, f"gate_{B.launch_tag_of(tag, okey)}_{okey}_{suffix_of(tag, okey)}.json"), gate)
        for ev in TASKS:
            three[ev]["rows"][f"{tag}/{okey}"] = three_row(tag, sel, okey, row_specs[ev], eval_id=ev, obj_sha=(attest_sha or {}).get(okey, sha), tree=TREE, store_key=store_key, steered=steered,
                                                           anchor_spec=(specs[ev][NONE] if okey != NONE else None))
    for ev in TASKS:
        if three_override and ev == eval_id:
            three_override(three[ev])
        three[ev]["n_rows"] = len(launch[ev])
        three[ev]["n_complete"] = len(three[ev]["rows"])
        write(os.path.join(recs, f"three_numbers_{ev}_qwen_seedrep.json"), three[ev])
    launch_doc = {"schema": B.LAUNCH_SET_SCHEMA, "derived_at": NOW, "evals": {ev: {"own_source": f"{ev}_seedrep", "launch": launch[ev],
                                                                                     "canonical_decode": {"policy": f"qwen35_4b_it|{ev}|off|protocol@v1", "thinking": "off", "temperature": TASKS[ev][4], "cap": TASKS[ev][3]}}
                                                                                for ev in TASKS}}
    if launch_layout == "carried":
        for ev in TASKS:
            launch_doc["evals"][ev].update(census_row_table(launch[ev], SEL, SEL_P))
        launch_doc["row_table"] = list(launch[eval_id])
    elif launch_layout != "launch":
        raise ValueError(launch_layout)
    if launch_override:
        launch_override(launch_doc)
    write(os.path.join(recs, B.LAUNCH_SET_FILE), launch_doc)
    write(os.path.join(recs, B.PINS_FILE), {"qwen35_4b": {STAGED: {"object_sha256": lp.get(RELU, OBJ), "source": "synthetic"}, STAGED_P: {"object_sha256": lp.get(COMPOSED, OBJ_P), "source": "synthetic"},
                                                          STAGED_A: {"object_sha256": lp.get(ARDITI, OBJ_A), "source": "synthetic", "kind": "vector_all", "staged_id": STAGED_A}}})
    write(os.path.join(recs, B.ROWS_LAUNCH_FILE), "launched_at\tsha\twave\trow\tcells\tstage\n")
    write(os.path.join(recs, B.SCHEDULE_FILE), {"written": NOW, "per_cell_h": 0.25, "planned": []})
    h_roster = {"targets": {"qwen35_4b": {"views": {SEL: {"config": f"configs/sdfbase_full22_{TAG}.yaml", "cells": {RELU: STAGED, PROMPTED: None, "prompted__si_constitution": None}},
                                                    SEL_P: {"config": f"configs/sdfbase_full22_{TAG_P}.yaml", "paired_of": SEL, "cells": {COMPOSED: STAGED_P}}}}}}
    h_roster_path = write(str(tmp_path / "roster.yaml"), yaml.safe_dump(h_roster))
    h_pins_path = write(str(tmp_path / "pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": hp.get(RELU, OBJ), "source": "synthetic"}, STAGED_P: {"object_sha256": hp.get(COMPOSED, OBJ_P), "source": "synthetic"}}})
    a_roster = {"targets": {"qwen35_4b": {"views": {"ar_c0p3": {"c": 0.3, "dose_unit": "raw_coefficient", "phase": "both", "cells": {ARDITI: STAGED_A}}}}}}
    a_roster_path = write(str(tmp_path / "arditi_roster.yaml"), yaml.safe_dump(a_roster))
    a_pins_path = write(str(tmp_path / "arditi_pins.json"), {"qwen35_4b": {STAGED_A: {"staged_id": STAGED_A, "object_sha256": hp.get(ARDITI, OBJ_A), "kind": "vector_all", "served_layer": 12, "c": 0.3,
                                                                                       "dose_unit": "raw_coefficient", "phase": "both", "positions": "all", "view": "ar_c0p3"}}})
    return {"root": root, "roster": h_roster_path, "pins": h_pins_path, "arditi_roster": a_roster_path, "arditi_pins": a_pins_path, "keys": keys, "specs": specs[eval_id], "all_specs": specs,
            "roster_sha256": roster_sha256, "paired_none_key": paired_none_key, "paired_none_spec": specs_p[eval_id]}


def adapter(cls, fx: dict, seeds: LC.Seeds):
    return cls(root=fx["root"], seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], arditi_roster_path=fx["arditi_roster"], arditi_pins_path=fx["arditi_pins"])


def read(ad, conn=None, group: str = TAG):
    """One group's read on `conn`, or on a frame primed from the adapter's own emissions (:func:`primed_frame`) when none is given; every record validated."""
    from sisteer.registry import validate as VAL
    inputs = ad.discover()
    groups = {}
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


def metrics_of(bt, rid: str, comp_id: str) -> Tuple[dict, dict]:
    rows = [m for m in bt["run_metric"] if m["run_id"] == rid]
    value = [m for m in rows if m["computation_id"] == comp_id]
    bound = [m for m in rows if m["computation_id"] == ""]
    assert len(rows) == 2 and len(value) == 1 and len(bound) == 1, rows
    return value[0], bound[0]


def refused(facts, tag: str, okey: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{tag}|{okey} -- ")]
    assert len(hits) == 1, facts["refusals"]
    return hits[0]
