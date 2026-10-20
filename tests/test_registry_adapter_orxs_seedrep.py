"""Seed-replicate over-refusal controls (source `orxs_seedrep`; `cellstore.OrxsSeedrepAdapter` 3 over `cellstore.ORXS_SEEDREP_ROOTS`) on a
SYNTHETIC mirror shaped like the durable copy (`controls_sampled/controls_cap<N>/qwen35_4b_s2_t0/cells/<store key>/`, the two result files at
the root, the ladder record; invented shas, `$SHARED/...` paths, a handful of item rows; no generation text). What it pins:

* THE COMPLETION-CAP MERGE (`cellstore.COMPLETION_CAP_RULE`): a pair that escalated to 8,192 files its 8,192 row as the
  scored merge (`completion_cap_merged`, `final_rung`, `n_at_cap_by_rung`, the residual still capped as the at-cap share) and its 2,048 and 512
  rows as PROVENANCE ONLY — the run row with its generation facts and at-cap measure, no judgment / metric / item / state row however judged
  the roots are; a pair that ended at 2,048 is scored there with its 512 row provenance only; a pair that never left 512 is scored at 512.
* the merged map is cross-checked against the stores (final cap = the ladder's exit; n_at_cap_by_rung = gen_stats; rate / n_judged = the
  final root's labels; the by-cap file agrees) and a disagreement refuses the store by name; a missing result file refuses every group.
* contrasts pair merged value against merged value — the steered pair's final-cap row against the unsteered pair's final-cap row of the same
  seed, at different caps when the ladders exited differently (`extra.contrast_caps`), never a lower cap's `none`.
* a final-cap store whose judge pass has not landed files `generated` and the read that finds the judge files attaches the judgment to the SAME
  run ids; run ids never depend on the judge or on the merge.
* `view_legacy` `qwen35_4b_s2`, the producer key `qwen35_4b_s2_t0/<store key>`, the seed-selection objects (refused by name), the routing-table
  substrate, `token_cap_rung_index`, the ladder's exit vocabulary, `flip_pass/` never read, the result files and the ladder declared as inputs
  of every group.
* the end-to-end pass + ingest: scored rows reach `ingested`, provenance-only rows stay `generated`, a later judged read moves the final-cap
  rows without minting runs; the default reader is unchanged.
* THE SEED-1 TREE (a regression pin of version 2, which read it unchanged when the re-run of 2026-09-21 — array 592103, judged at each pair's
  FINAL cap only — reached the mirror at 17:44Z: `enumerate_stores` lists every target directory under a cap root, so the seed-1 directory was
  read the moment it existed): a second synthetic target directory `qwen35_4b_s1_t0` in the REAL seed-1 shape — eight stores at 512 with NO
  judge files, eight at 2,048 judged only on the controls that exited there (two of them unjudged altogether), five at 8,192 whose manifests
  name only the controls that climbed — beside the s2 and an s3 directory, the result files naming all three seeds. Sixteen scored rows (nine
  at 2,048, seven at 8,192), twenty-three provenance-only rows (sixteen at 512, seven at 2,048) whose ABSENCE of judge files is neither pending
  nor torn nor a refusal (v2's lower-cap rows on s2 / s3 were judged-and-withheld; the seed-1 tree is the first where nothing was judged there),
  the seed-1 fitted objects under their own ids, `none` / the prompted objects the same object on every seed view.
* THE NON-SEED TARGET (`OrxsSeedrepAdapter` 3; the `none` + `arditi_allpos` OR-Bench controls at the Arditi all-positions point c = 0.3,
  `qwen35_4b_ar_c0p3_t0`, cluster A 592686, seed base 20260909, judged at each pair's final cap only): the target directory's GRAMMAR picks
  the object resolver (`target_grammar`; `cellstore.orxs_target_object`) — `arditi_allpos` is the Arditi arm's object of that rung with the
  rung's strength columns as every reader of the tag files them, `none` is `none`, any other store id refuses; `view_legacy` `qwen35_4b_ar_c0p3`;
  a target still climbing (512 in flight: no ladder record, or one with no exit) files its stores as `generated` rows with
  `final_cap_undeclared` — no refusal, no disposition, no judgment — and the read that finds the exit files the merge under the same ids;
  the complete target scores the final cap, files 512 as provenance only and contrasts `arditi_allpos` against the same standup's `none`,
  with the result files cross-checked under the rung's key when they carry a row and the ladder alone (`derived`) when they do not.
* THE ARDITI ARM'S OWN OR-BENCH STORES (`cellstore.ARDITI_ALLPOS_ORB_ROOTS` on the generic adapter's DEFAULT_ROOTS, source `arditi_allpos`;
  served 2026-09-17, job 546174): the alphasteer OR-Bench roots' `cells` layout under `<tag>_t0/orb_cap<N>/`, so the root declares what the
  directory does not spell — the served tag (the c = 0.3 strength columns), the source's `view_legacy` `qwen35_4b_it`, campaign
  `arditi_allpos` — and reads the arm's graded digest `orbench_graded.json` as the pair's final-cap record: the 2,048 store is the merge
  and must reproduce the digest's full-refusal rate / n_judged / n_abstained to the digit, the unjudged 512 store is `superseded_by_rung`
  naming the 2,048 run, a digest of another schema / arm / value refuses by name. The generic adapter's version stays 5.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Sequence

import pytest

from sisteer.registry import db as DB
from sisteer import site as SITE
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import cellstore as CS
from sisteer.registry.adapters import dose_grammar as DG
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN

import _anchor_frame_fixtures as AF
import registry_items_fixtures as F

NOW = "2026-09-21T10:00:00.000Z"
SEED_TAG = "qwen35_4b_s2"
TARGET = f"{SEED_TAG}_t0"
NONE = "none"
LORA = F.LORA_CELL                                   # constv2-mix-full__lora_r4_ce
PROMPTED = "prompted__safety_refusal"
UNKNOWN = "constv2-mix-full__lora_r4_ce_minus_ctl_distilled"     # a store id the synthetic roster does not name at s2
#: the seed-1 target directory and the s3 one beside it: the eight store ids OR served, as the harness spells them
S1_TAG = "qwen35_4b_s1"
S1_TARGET = f"{S1_TAG}_t0"
S3_TAG = "qwen35_4b_s3"
S3_TARGET = f"{S3_TAG}_t0"
RELU = "constv2-mix-full__psr_ce_all"                    # the constitution ReLU-steer object
CTL_LORA = "constv2ctl2-full__lora_r4_ce"                # the control LoRA
CTL_RELU = "constv2ctl2-full__psr_ce_all"                # the control ReLU-steer object
DIFF_LORA = "constv2-mix-full__lora_r4_ce_minus_ctl"     # the LoRA difference
SI_CONST = "prompted__si_constitution"
S1_FITTED = (LORA, RELU, CTL_LORA, CTL_RELU, DIFF_LORA)
S1_STORE_IDS = (NONE, PROMPTED, SI_CONST) + S1_FITTED
IHASH = {NONE: "2ff9cdff", LORA: "18984b43", PROMPTED: "faa4c8cf", UNKNOWN: "0badf00d",
         RELU: "d69810e6", CTL_LORA: "960ba2cf", CTL_RELU: "29d675e3", DIFF_LORA: "461a8da0", SI_CONST: "98eaea2d"}
#: the seed-1 fitted objects' synthetic ids (their own, distinct from s2's F.OBJ_SHA and s3's), the s3 fitted object, the second prompt
S1_OBJECTS = {c: hashlib.sha256(f"seed1|{c}".encode()).hexdigest() for c in S1_FITTED}
S3_LORA_OBJECT = hashlib.sha256(b"seed3|constv2-mix-full__lora_r4_ce").hexdigest()
SI_CONST_SHA = hashlib.sha256(b"synthetic constitution instruction").hexdigest()
SI_CONST_OBJECT_ID = f"si:{SI_CONST_SHA}"
#: the synthetic roster: (seed tag, store id) -> object id; the fitted one is the fixtures' object (its sha16 is what the manifests carry)
SEED_OBJECTS = {(SEED_TAG, LORA): F.OBJ_SHA, (SEED_TAG, PROMPTED): F.SI_OBJECT_ID, (SEED_TAG, SI_CONST): SI_CONST_OBJECT_ID,
                (S3_TAG, LORA): S3_LORA_OBJECT, (S3_TAG, PROMPTED): F.SI_OBJECT_ID, (S3_TAG, SI_CONST): SI_CONST_OBJECT_ID,
                (S1_TAG, PROMPTED): F.SI_OBJECT_ID, (S1_TAG, SI_CONST): SI_CONST_OBJECT_ID, **{(S1_TAG, c): S1_OBJECTS[c] for c in S1_FITTED}}
#: the seed-1 pairs' exits, as the real ladder record declares them: `none` and the two prompts left at 2,048; XSTest and OR-Bench parted ways on
#: three fitted objects; the control ReLU-steer object and the LoRA difference climbed to 8,192 on both
S1_EXITS = {NONE: {"xstest": 2048, "or_bench_hard_1k": 2048}, PROMPTED: {"xstest": 2048, "or_bench_hard_1k": 2048},
            SI_CONST: {"xstest": 2048, "or_bench_hard_1k": 2048}, RELU: {"xstest": 2048, "or_bench_hard_1k": 8192},
            LORA: {"xstest": 8192, "or_bench_hard_1k": 2048}, CTL_LORA: {"xstest": 2048, "or_bench_hard_1k": 8192},
            CTL_RELU: {"xstest": 8192, "or_bench_hard_1k": 8192}, DIFF_LORA: {"xstest": 8192, "or_bench_hard_1k": 8192}}
S1_JOB = "592103"
#: the non-seed target: the Arditi all-positions point c = 0.3 (cluster A 592686), `none` + `arditi_allpos`, OR-Bench only, seed base 20260909
AR_TAG = "qwen35_4b_ar_c0p3"
AR_TARGET = f"{AR_TAG}_t0"
ARDITI = "arditi_allpos"
AR_JOB = "592686"
AR_SEED_BASE = 20260909
ARDITI_OBJECT = hashlib.sha256(b"arditi all-positions direction at c = 0.3").hexdigest()
IHASH[ARDITI] = "92b66dce"
SEED_OBJECTS[(AR_TAG, ARDITI)] = ARDITI_OBJECT
AR_CTL = ("or_bench_hard_1k",)
RUBRIC = {"xstest": "288e8c08b0c7bbfa", "or_bench_hard_1k": "15f29334703c31a8"}
TREE = "bcb099a674291e3f1383eef5a57f9cfe3a3fa03c"
HARNESS_SHA = "f5ecf345cb90ba9b5c392fc2aca4cba58a2f7094"
T0 = 1_789_979_000.0
CTLS = ("xstest", "or_bench_hard_1k")
EVAL_OF = {"xstest": "xstest", "or_bench_hard_1k": "or_bench_hard"}
DEFAULT_LABELS = ["full_compliance", "partial_refusal", "full_refusal"]
#: the pairs' ladder exits: the fitted object escalated to 8,192, `none` ended at 2,048, the prompt never left 512
EXITS = {NONE: {"xstest": 2048, "or_bench_hard_1k": 2048}, LORA: {"xstest": 8192, "or_bench_hard_1k": 8192}, PROMPTED: {"xstest": 512, "or_bench_hard_1k": 512}}
#: per store: the judge labels (cycled over the items) and the at-cap item ids the producer recorded
SPEC = {
    (512, NONE): {"labels": {"xstest": ["full_compliance", "full_compliance", "full_refusal"], "or_bench_hard_1k": ["full_compliance", "full_refusal"]},
                  "at_cap": {"xstest": ["xst0001-cccccccc"]}, "job": "586141"},
    (2048, NONE): {"labels": {"xstest": ["full_compliance"], "or_bench_hard_1k": ["full_compliance", "partial_refusal"]}, "at_cap": {}, "job": "586141"},
    (512, LORA): {"labels": {"xstest": ["full_refusal", "partial_refusal", "full_compliance"], "or_bench_hard_1k": ["full_refusal"]},
                  "at_cap": {"or_bench_hard_1k": ["orb0000-eeeeeeee"]}, "job": "586143"},
    (2048, LORA): {"labels": {"xstest": ["full_refusal", "partial_refusal"], "or_bench_hard_1k": ["partial_refusal"]},
                   "at_cap": {"xstest": ["xst0000-cccccccc", "xst0002-cccccccc"]}, "job": "586143"},
    (8192, LORA): {"labels": {"xstest": ["full_refusal", "full_compliance"], "or_bench_hard_1k": ["full_refusal", "full_compliance", "full_compliance"]},
                   "at_cap": {"xstest": ["xst0001-cccccccc"]}, "job": "586143"},
    (512, PROMPTED): {"labels": {}, "at_cap": {}, "job": "586154"},
}
CAPS = (512, 2048, 8192)


def _synthetic_seed_selection(cell_id_legacy: str, seed_tag: str, object_sha16: Optional[str], conn) -> tuple:
    """The target-grammar resolver over a synthetic roster: a seed tag names `none`, the declared prompts and the seed's fitted objects; the
    Arditi point names `none` and its one all-positions object (refusing any other store id, as the real resolver does); anything else refuses."""
    tag = CS.seed_tag_of(seed_tag)
    if not CS._SEED_TAG_RE.match(tag):
        spec = DG.parse_arditi_allpos_tag(seed_tag)
        if spec is None:
            raise AB.AdapterRefusal(seed_tag, "spells neither a seed selection nor an Arditi all-positions point")
        if cell_id_legacy == NONE:
            return "none", "synthetic: the unsteered object key"
        if cell_id_legacy != ARDITI:
            raise AB.AdapterRefusal(cell_id_legacy, f"the Arditi all-positions point serves `none` and `{ARDITI}` only")
        oid = SEED_OBJECTS[(spec.standup_tag, ARDITI)]
        if not object_sha16 or not oid.startswith(object_sha16):
            raise AB.AdapterRefusal(cell_id_legacy, f"the store's object_sha {object_sha16} is no prefix of the synthetic Arditi object")
        return oid, f"synthetic Arditi lane {spec.staged_id}"
    if cell_id_legacy == NONE:
        return "none", "synthetic: the unsteered object key"
    oid = SEED_OBJECTS.get((tag, cell_id_legacy))
    if oid is None:
        raise IC.Unresolved(cell_id_legacy, f"the synthetic roster's {tag} selection names no object for it")
    if not oid.startswith("si:") and (not object_sha16 or not oid.startswith(object_sha16)):
        raise AB.AdapterRefusal(cell_id_legacy, f"the store's object_sha {object_sha16} is no prefix of the object the synthetic roster names")
    return oid, f"synthetic roster {tag}"


class _Adapter(CS.OrxsSeedrepAdapter):
    object_resolvers = {"seed_selection": _synthetic_seed_selection, "target_grammar": _synthetic_seed_selection}


def _ctx(conn=None):
    return AB.AdapterContext(root="/nonexistent", now=NOW, producer={"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}, conn=conn)


def _json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def _jsonl(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")


def _key(cell_id: str, cap: int) -> str:
    """The store key: the harness hash differs per cap, as on the real stores."""
    chash8 = hashlib.sha256(f"{cell_id}|{cap}".encode()).hexdigest()[:8]
    return f"{cell_id.replace('__', '--', 1)}__controls__t0__a0__{IHASH[cell_id]}__{chash8}"


def _item_ids(ctl: str, n_safe: int, n_unsafe: int, n_orb: int) -> List[tuple]:
    if ctl == "xstest":
        return [(f"xst{i:04d}-{'c' * 8}", "safe") for i in range(n_safe)] + [(f"xst{n_safe + j:04d}-{'d' * 8}", "unsafe") for j in range(n_unsafe)]
    return [(f"orb{i:04d}-{'e' * 8}", "all") for i in range(n_orb)]


def _rate(labs: Sequence[str], n_scored: int) -> float:
    """full_refusal / judged over the SCORED items (the first ``n_scored`` items, labelled by cycling ``labs`` the way judge_store does)."""
    labs = list(labs) or DEFAULT_LABELS
    return sum(1 for k in range(n_scored) if labs[k % len(labs)] == "full_refusal") / n_scored


def write_store(caps_parent: str, cap: int, cell_id: str, *, judged_controls: Sequence[str] = (), labels: Optional[Dict[str, Sequence[str]]] = None,
                at_cap: Dict[str, Sequence[str]] = None, job: str = "586142", seed_base: int = 2, escalated_from: Optional[int] = None,
                object_sha16: Optional[str] = F.OBJ16, target: str = TARGET, n_safe: int = 4, n_unsafe: int = 2, n_orb: int = 4,
                controls: Optional[Sequence[str]] = None, fit_hf: str = F.HF_BASE, cells_parent: Optional[str] = None) -> str:
    """One store in the REAL shape: manifest / gen_stats / evidence always (the producer's), rows / judge_items / judge_summary only for the
    controls the judge has reached (the judge driver's). ``controls`` (default both) are the controls the store GENERATED — the manifest's
    list and the gen_stats blocks: a store above the base cap names only the controls that climbed to it. Returns the store directory."""
    controls = list(controls) if controls is not None else list(CTLS)
    seed_tag = CS.seed_tag_of(target)
    key = _key(cell_id, cap)
    d = os.path.join(cells_parent, "cells", key) if cells_parent else os.path.join(caps_parent, f"controls_cap{cap}", target, "cells", key)
    os.makedirs(os.path.join(d, "private"), exist_ok=True)
    fitted = cell_id not in (NONE,) and not cell_id.startswith("prompted__")
    ihash16, chash16 = IHASH[cell_id] + "283114a3", key.rsplit("__", 1)[1] + "3f253f9c"
    gen = {"arm": "vendor_recommended", "batch_size": 8, "do_sample": True, "max_new_tokens": cap, "min_p": 0.0, "presence_penalty": 1.5,
           "repetition_penalty": 1.0, "seed_base": seed_base, "temperature": 0.7, "top_k": 20, "top_p": 0.8, "trial": 0}
    if escalated_from is not None:
        gen["escalated_from"] = {"cap": escalated_from, "root": f"/workspace-vast/{F.TESTUSER}/exp/sdfbase_full22_native/sampled/controls_cap{escalated_from}/{target}"}
        gen["floor_sample"] = {"k": 200, "rule": "synthetic"}
    fitp = ({"chat_template_bypassed": True, "fit_kind": "document", "model_id": fit_hf, "object_sha": object_sha16, "positions_at_fit": "all",
             "revision": "b" * 40, "served_model_id": F.HF_IT, "served_revision": "c" * 40, "status": "document_fit", "transfer": True} if fitted
            else {"model_id": None, "object_sha": None, "positions_at_fit": None, "served_model_id": F.HF_IT, "served_revision": "c" * 40, "transfer": False})
    row_constants = {"arm_name": cell_id.replace("__", "--", 1), "attempt": 0, "coeff": 1.0 if cell_id != NONE else None, "engine": "vllm",
                     "env_digest": "7f0b0097fbceed4c", "harness_config_hash": chash16, "intervention_hash": ihash16, "kind": "cell" if cell_id != NONE else "none",
                     "model_id": F.HF_IT, "model_revision": "c" * 40, "run_id": f"served-controls-{cell_id.replace('__', '--', 1)}", "seed": 0,
                     "slurm_job_id": job, "sweep": "two_si_controls_served", "trial_id": 0}
    identity_text = "proxy_vllm+lora+interleaved" if fitted else "proxy_vllm+passthrough+interleaved"
    manifest = {"cell_id": cell_id, "contract_version": 3, "controls": controls, "engine": "vllm", "fit_provenance": fitp,
                "harness_cell_id": f"si_{cell_id}", "harness_config_hash": chash16, "intervention_hash": ihash16, "model_id": F.HF_IT, "model_revision": "c" * 40,
                "producer": {"driver": "served_capability_run", "tree_sha": TREE},
                "resolved_config": {"cell_id": cell_id, "controls": ["gsm8k", "mmlu", "or_bench_hard_1k", "xstest"], "driver": "served_capability_run_v2", "engine": "vllm",
                                    "gen": gen, "judge_model": "claude-sonnet-5", "limit": None, "model_id": F.HF_IT, "model_revision": "c" * 40,
                                    "serving": {"serving_engine": "lora_native", "serving_identity": identity_text, "virtual_model": f"petri__{cell_id}"},
                                    "steer_positions": "all"},
                "routing_table": f"/workspace-vast/{F.TESTUSER}/exp/sdfbase_full22_native/sampled/serving/controls_t0_{seed_tag}_{job}/controls_{cap}/routing_table.json",
                "row_constants": row_constants, "serving_engine": "lora_native", "serving_identity": identity_text,
                "serving_path": "lora" if fitted else "passthrough", "steer_positions": "all", "virtual_model": f"petri__{cell_id}"}
    _json(os.path.join(d, "manifest.json"), manifest)
    at_cap = at_cap or {}
    gs: Dict[str, object] = {}
    for ctl in controls:
        ids = [i for i, _ in _item_ids(ctl, n_safe, n_unsafe, n_orb)]
        capped = [i for i in ids if i in set(at_cap.get(ctl, ()))]
        gs[ctl] = {"control": ctl, "n_new": len(ids), "n_total": len(ids), "finish_reasons": {"length": len(capped), "stop": len(ids) - len(capped)},
                   "spool": f"/workspace-vast/{F.TESTUSER}/exp/sdfbase_full22_native/sampled/controls_cap{cap}/{target}/cells/{key}/private/gens_{ctl}.jsonl",
                   "cap": cap, "decode": dict(gen, cap=cap), "n_at_cap": len(capped), "at_cap_item_ids": capped, "n_regenerated": len(ids), "n_copied": 0}
    gs["decode_resolution"] = {"decode_default": "config_block", "served_model": F.HF_IT, "served_revision": "c" * 40}
    _json(os.path.join(d, "gen_stats.json"), gs)
    _json(os.path.join(d, "evidence.json"), {"driver": "served_capability_run_v2", "cell_id": cell_id, "engine": "vllm", "serving_identity": identity_text,
                                             "engine_stamp": {"target_engine": "vllm", "serving_path": "lora" if fitted else "passthrough", "object_sha": object_sha16 if fitted else None},
                                             "fit_provenance": fitp, "serve_header": {"template_kwargs": {"enable_thinking": False}}, "written_at": "2026-09-21T07:48:18Z"})
    _json(os.path.join(d, "cap_ladder.json"), {"cap": cap, "cell_id": cell_id, "ladder": [512, 1024, 2048, 4096, 8192, 16384], "schema": "cap_ladder/1"})
    if judged_controls:
        judge_store(d, cell_id, key, list(judged_controls), labels or {}, at_cap, n_safe, n_unsafe, n_orb, target=target)
    return d


def judge_store(d: str, cell_id: str, key: str, judged_controls: List[str], labels: Dict[str, Sequence[str]], at_cap: Dict[str, Sequence[str]],
                n_safe: int = 4, n_unsafe: int = 2, n_orb: int = 4, dirty: Sequence[str] = (".claude/", "constitutional-adapters"), target: str = TARGET) -> None:
    """The judge driver's files for `judged_controls`, ADDED to a generated store (the shape grading lands in): rows.jsonl (labels + aggregates),
    judge_items.jsonl, judge_summary.json with the driver's provenance and per-control counters. Re-judging a store with more controls rewrites
    all three with the union, as the driver does."""
    existing = json.load(open(os.path.join(d, "judge_summary.json"))) if os.path.isfile(os.path.join(d, "judge_summary.json")) else {}
    controls = sorted(set(existing.get("controls", {}).keys()) | set(judged_controls))
    const = json.load(open(os.path.join(d, "manifest.json")))["row_constants"]
    rows: List[Dict] = []
    judge_rows: List[Dict] = []
    summary_controls: Dict[str, object] = {}
    ts = T0
    value_of = {"full_compliance": 2.0, "partial_refusal": 1.0, "full_refusal": 0.0}
    for ctl in controls:
        task = "judged/xstest_v2" if ctl == "xstest" else "judged/or_bench_hard_1k"
        labs = list(labels.get(ctl) or DEFAULT_LABELS)
        capped = set(at_cap.get(ctl, ()))
        n_lab = 0
        for k, (iid, subset) in enumerate(_item_ids(ctl, n_safe, n_unsafe, n_orb)):
            lab = labs[k % len(labs)]
            ts += 1.0
            extra = {"label": lab, "rubric_sha": RUBRIC[ctl]}
            jr = {"cache_org": "lowprio", "cached_before_run": False, "control": ctl, "item_id": iid, "judge_model": "claude-sonnet-5", "label": lab,
                  "label_source": "store", "protocol_sha": "b8d28441cf71caec", "request_key": hashlib.sha256(f"{key}|{iid}".encode()).hexdigest(),
                  "rubric_sha": RUBRIC[ctl], "status": "judged", "task": task, "verdict_created_at": ts - 100.0}
            if ctl == "xstest":
                extra.update(subset=subset, type="homonyms")
                jr.update(subset=subset, type="homonyms")
            else:
                extra.update(category="deception")
                jr["category"] = "deception"
            metric = "xst_label" if ctl == "xstest" else "orb_label"
            rows.append(dict(const, task=task, task_version="judged_v1", metric=metric, item_id=iid, value=value_of[lab], created_at=ts,
                             extra_json=json.dumps(extra, sort_keys=True, separators=(",", ":")), position_order=k, domain="refusal_calibration_judged"))
            if ctl != "xstest":
                rows.append(dict(const, task=task, task_version="judged_v1", metric="orb_refused", item_id=iid, value=1.0 if lab == "full_refusal" else 0.0,
                                 created_at=ts, extra_json=None, position_order=k, domain="refusal_calibration_judged"))
            judge_rows.append(jr)
            n_lab += 1
        agg = "xst_safe_refused_bounds_aggregate" if ctl == "xstest" else "orb_refused_bounds_aggregate"
        rows.append(dict(const, task=task, task_version="judged_v1", metric=agg, item_id="__aggregate__", value=0.25, created_at=ts + 1,
                         extra_json=json.dumps({"n": n_lab, "n_graded": n_lab, "rubric_sha": RUBRIC[ctl]}), position_order=None, domain="refusal_calibration_judged"))
        summary_controls[ctl] = {"judged_at": "2026-09-21T08:27:27Z", "status": "ok", "n": n_lab, "n_cached_before_run": 0, "wall_clock_s": 118.0,
                                 "at_cap_share": len(capped) / max(n_lab, 1), "task": task, "filed_by_harness": True,
                                 "harness_result": {"n": n_lab, "n_graded": n_lab, "n_abstained": 0, "rubric_sha": RUBRIC[ctl],
                                                    "judge": {"api_calls": n_lab, "backend": "anthropic", "cache_hits": 0, "calls_by_org": {"highprio": 0, "lowprio": n_lab}}},
                                 "judge_counters_delta": {"abstained": 0, "api_calls": n_lab, "cache_hits": 0, "gate_live_ceiling": 100, "gate_rate_limited_total": 0}}
    _jsonl(os.path.join(d, "rows.jsonl"), rows)
    _jsonl(os.path.join(d, "judge_items.jsonl"), judge_rows)
    _json(os.path.join(d, "judge_summary.json"), {
        "cell": key, "cell_id": cell_id, "controls": summary_controls, "model_id": F.HF_IT, "model_revision": "c" * 40,
        "started_at": "2026-09-21T08:25:29Z", "finished_at": "2026-09-21T08:41:47Z", "target": target,
        "provenance": {"harness_sha": HARNESS_SHA, "harness_branch": "main", "harness_dirty_paths": list(dirty), "driver": "served_capability_judge_v1",
                       "driver_path": "scripts/served_capability_judge.py", "judge_model": "claude-sonnet-5", "judge_org": "lowprio",
                       "judge_cache_note": "pass-1 (primary labels): synthetic", "sdk_max_retries": 2, "anthropic_sdk": "0.0.0", "concurrency_total": 100, "cell_parallel": 6,
                       "rubrics": {c: {"name": f"{c}_cpi", "sha": RUBRIC[c], "version": "v1"} for c in controls}}})


def write_ladder(ladders: str, target: str, exits: Dict[str, Dict[str, int]], caps: Sequence[int] = CAPS) -> str:
    """The producer's exit record for a target directory: per (store id, control) the caps attempted up to the exit, the gate's `escalate`
    at every cap below it and the terminal `exit_le_1pct` at the exit (`left_ladder_at`), as ladder.json spells it."""
    cells: Dict[str, object] = {}
    for cell_id, per_ctl in exits.items():
        cells[cell_id] = {}
        for ctl, exit_cap in per_ctl.items():
            attempted = [c for c in caps if c <= exit_cap]
            decisions = {str(c): {"decision": ("exit_le_1pct" if c == exit_cap else "escalate"), "n_truncated": 0 if c == exit_cap else 9, "n_at_cap": 0 if c == exit_cap else 9,
                                  "next_cap": (None if c == exit_cap else caps[caps.index(c) + 1]), "at": "2026-09-21T07:40:34+00:00"} for c in attempted}
            cells[cell_id][ctl] = {"n_at_cap": {str(c): (0 if c == exit_cap else 9) for c in attempted}, "rungs_attempted": attempted, "decisions": decisions,
                                   "roots": {str(c): f"/workspace-vast/{F.TESTUSER}/exp/sdfbase_full22_native/sampled/controls_cap{c}/{target}" for c in attempted},
                                   "left_ladder_at": {"status": "left", "rung": exit_cap, "exit": "exit_le_1pct", "n_truncated": 0, "n_loop": 0, "unresolved_truncation": False}}
    path = os.path.join(ladders, target, "ladder.json")
    _json(path, {"kind": "controls", "tag": CS.seed_tag_of(target), "trial": 0, "rungs": list(caps), "extend_kind": "controls",
                 "exit_rule": "controls: a (cell, control) climbs a cap ONLY on the gate's ESCALATE (synthetic)", "written_at": "2026-09-21T08:07:14+00:00",
                 "count_only_ceiling": 8192, "cells": cells, "summary": {"n_pairs": sum(len(v) for v in exits.values())}})
    return path


def result_map_rows(stores: Dict[tuple, str], exits: Dict[str, Dict[str, int]], spec: Dict[tuple, dict], *, seed: str, target: str,
                    n_safe: int = 4, n_unsafe: int = 2, n_orb: int = 4) -> tuple:
    """One seed's rows of the two result files as the stores REALISE them: `results_final_merged.json` rows per (seed, store id, control) —
    final_rung = the ladder's exit, n_at_cap_by_rung from every cap root's gen_stats, the residual = the final cap's at-cap count, rate /
    n_judged = the final root's labels over the scored subset — and `results_by_rung.json` main_mirror rows per (seed, store id, control, cap)
    for every control a store GENERATED (a store above the base cap generated only the controls that climbed to it)."""
    merged: Dict[str, object] = {}
    by_cap: Dict[str, object] = {}
    for (cap, cell_id), store in stores.items():
        gs = json.load(open(os.path.join(store, "gen_stats.json")))
        for ctl in CTLS:
            if not isinstance(gs.get(ctl), dict):
                continue
            n_scored = n_safe if ctl == "xstest" else n_orb
            labs = spec[(cap, cell_id)]["labels"].get(ctl) or DEFAULT_LABELS
            rate = _rate(labs, n_scored)
            by_cap[f"{seed}|{cell_id}|{ctl}|{cap}"] = {"rate": rate, "n_judged": n_scored, "n": gs[ctl]["n_total"], "n_abstained": 0,
                                                       "at_cap_share": gs[ctl]["n_at_cap"] / gs[ctl]["n_total"], "ci95_over_judged": [rate, rate], "status": "ok"}
            final_cap = exits[cell_id][ctl]
            if cap == final_cap:
                n_by = {}
                for c in CAPS:
                    if (c, cell_id) in stores and c <= final_cap:
                        lower = json.load(open(os.path.join(stores[(c, cell_id)], "gen_stats.json")))
                        if isinstance(lower.get(ctl), dict):
                            n_by[str(c)] = lower[ctl]["n_at_cap"]
                merged[f"{seed}|{cell_id}|{ctl}"] = {"final_rung": final_cap, "rate": rate, "n_judged": n_scored, "n_abstained": 0, "ci95_over_judged": [rate, rate],
                                                     "n_at_cap_by_rung": n_by, "residual_capped_at_final": gs[ctl]["n_at_cap"], "rungs_attempted": sorted(int(c) for c in n_by),
                                                     "store": f"controls_sampled/controls_cap{cap}/{target}/cells/<{cell_id}>__controls__t0__a0__*/judge_items.jsonl", "subset200_rate": None}
    return merged, by_cap


def write_result_maps_seeds(root: str, parts: Sequence[dict]) -> Dict[str, str]:
    """The two result files over several seeds' target directories (`parts`: the keyword arguments of :func:`result_map_rows` per seed, with
    `stores` / `exits` / `spec`), the way the real files name s1 / s2 / s3 in one `rows` mapping."""
    merged: Dict[str, object] = {}
    by_cap: Dict[str, object] = {}
    seeds: Dict[str, str] = {}
    for part in parts:
        m, b = result_map_rows(**part)
        merged.update(m)
        by_cap.update(b)
        seeds[part["seed"]] = f"synthetic {part['target']}"
    merged_path = os.path.join(root, "results_final_merged.json")
    by_cap_path = os.path.join(root, "results_by_rung.json")
    _json(merged_path, {"key": "seed|cell|control", "rule": CS.COMPLETION_CAP_RULE, "how_the_stores_realise_it": "synthetic", "seeds": seeds, "rows": merged})
    _json(by_cap_path, {"key": "seed|cell|control|cap", "headline_rung": 2048, "tree": TREE, "main_mirror": by_cap, "flip_pass_live_reask": {}})
    return {"merged": merged_path, "by_rung": by_cap_path}


def write_result_maps(root: str, stores: Dict[tuple, str], exits: Dict[str, Dict[str, int]], spec: Dict[tuple, dict], *, n_safe: int = 4, n_unsafe: int = 2,
                      n_orb: int = 4, seed: str = "s2", target: str = TARGET) -> Dict[str, str]:
    """The two result files for ONE seed's target directory (:func:`result_map_rows`)."""
    return write_result_maps_seeds(root, [dict(stores=stores, exits=exits, spec=spec, seed=seed, target=target, n_safe=n_safe, n_unsafe=n_unsafe, n_orb=n_orb)])


def _build_mirror(tmp_path, n_unsafe: int = 2, judged_final_fitted: bool = True) -> dict:
    """A synthetic durable mirror: caps 512 / 2,048 / 8,192 under `controls_sampled/`; `none` at 512 and 2,048 (exit 2,048), the fitted object at
    all three (exit 8,192; its 8,192 store unjudged when ``judged_final_fitted`` is False), the prompt at 512 only (exit 512); every judged
    store judged on both controls; a `flip_pass/` copy that must never be read; the ladder record; the two result files computed from the
    stores (the fitted 8,192 row from the labels the judge WILL write). ``n_unsafe`` XSTest items of the unsafe subset ride beside the four
    safe ones (the real stores: 200 beside 250)."""
    root = str(tmp_path / "orxs_seedrep")
    caps_parent = os.path.join(root, "controls_sampled")
    ladders = os.path.join(root, "ladders")
    stores: Dict[tuple, str] = {}
    for (cap, cell_id), s in SPEC.items():
        judged = CTLS if not (cell_id == LORA and cap == 8192 and not judged_final_fitted) else ()
        lower = [c for c in CAPS if c < cap]
        stores[(cap, cell_id)] = write_store(caps_parent, cap, cell_id, judged_controls=judged, labels=s["labels"], at_cap=s["at_cap"], job=s["job"],
                                             escalated_from=(lower[-1] if lower else None), object_sha16=(F.OBJ16 if cell_id == LORA else None), n_unsafe=n_unsafe)
    flip = os.path.join(caps_parent, "flip_pass", "cap512", TARGET, "cells")
    os.makedirs(flip, exist_ok=True)
    shutil.copytree(stores[(512, NONE)], os.path.join(flip, os.path.basename(stores[(512, NONE)])))
    ladder = write_ladder(ladders, TARGET, EXITS)
    maps = write_result_maps(root, stores, EXITS, SPEC, n_unsafe=n_unsafe)
    roots = CS.orxs_seedrep_roots(parents=(caps_parent,), declared=(512, 2048, 8192), ladders=ladders)
    return {"root": root, "caps_parent": caps_parent, "ladders": ladders, "ladder": ladder, "stores": stores, "roots": roots, "n_unsafe": n_unsafe,
            "merged": maps["merged"], "by_rung": maps["by_rung"]}


@pytest.fixture
def mirror(tmp_path):
    return _build_mirror(tmp_path)


def _adapter(mirror, dims=None) -> _Adapter:
    # the companion group's directory is the synthetic mirror's own `allmodels_paired/` (absent: nothing discovered), never a real tree on disk
    return _Adapter(roots=mirror["roots"], dims=dims or F.dims_map(), merged_path=mirror["merged"], by_cap_path=mirror["by_rung"],
                    role_collapse_dir=os.path.join(os.path.dirname(mirror["merged"]), ORC.DIR_NAME))


def _primed_conn(ad):
    """The connection of a frame PRIMED from the adapter's own emissions over every group (`_anchor_frame_fixtures`), built once per adapter
    and cached on it: since the shared reader's 8 every merged contrast is decided by the pass's anchor currency, so an in-memory read
    describes the steady state after an ingest -- every own `none` current -- rather than the first pass over an empty registry."""
    frame = getattr(ad, "_test_frame", None)
    if frame is None:
        frame = AF.primed_frame(ad, now=NOW, producer={"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"},
                                root="/nonexistent")
        ad._test_frame = frame
    return frame.con


def _read_group(ad: _Adapter, store: str, conn=None):
    inputs = ad.discover()
    mine = [i for i in inputs if ad.comparison_group(i) == store]
    batches = list(ad.read(_ctx(conn if conn is not None else _primed_conn(ad)), store, mine))
    assert len(batches) == 1
    return batches[0].records


def _payloads(records, record_type: str) -> List[Dict]:
    return [r["payload"] for r in records if r["record_type"] == record_type]


def _runs(records) -> Dict[str, Dict]:
    return {p["eval_id"]: p for p in _payloads(records, "run")}


def _validate_all(records) -> list:
    from sisteer.registry import records as R
    bad = []
    for r in records:
        rt, p = r["record_type"], dict(r["payload"])
        if rt in R.ITEM_SCALE_RECORD_TYPES:
            p.setdefault("source_revision_id", "rev1"); p.setdefault("withdrawn_by_revision_id", None)
        rec = R.make_record(rt, p, {"script": "t", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}, I.adapter_record_id(p, rt), NOW)
        out = VAL.validate_record(rec, "mem")
        if not isinstance(out, VAL.Validated):
            bad.append((rt, out.field, out.reason))
    return bad


# --------------------------------------------------------------------------------------------
# roots, discovery, declared inputs
# --------------------------------------------------------------------------------------------

def test_roots_discover_caps_by_name_declare_the_result_files_and_never_read_flip_pass(mirror):
    roots = mirror["roots"]
    by_cap = {int(r.path.rsplit("controls_cap", 1)[1]): r for r in roots}
    assert list(by_cap) == [512, 2048, 8192]
    for r in roots:
        assert r.run_key_scope == "target_dir" and r.at_cap_from_gen_stats and r.source_legacy == CS.ORXS_SEEDREP_SOURCE == "orxs_seedrep"
        assert r.over_refusal_headline and r.cap_provenance and r.file_unjudged and r.strip_trial_suffix and r.substrate_from_routing_table
        assert r.object_resolution == "target_grammar" and r.ladder_parent == mirror["ladders"]
    ad = _adapter(mirror)
    inputs = ad.discover()
    stores = {os.path.dirname(i.uri) for i in inputs if "#" not in i.uri}
    assert stores == set(mirror["stores"].values()) and not any("flip_pass" in s for s in stores)
    # the two result files and the ladder record are inputs of EVERY group (aliased per group; the same sha behind each alias)
    for store in stores:
        mine = [i for i in inputs if ad.comparison_group(i) == store]
        side = {i.uri.split("#", 1)[0] for i in mine if "#" in i.uri}
        assert side == {mirror["merged"], mirror["by_rung"], mirror["ladder"]}
        assert all(i.path in side and i.sha256 == AB.check_input(i.path).sha256 for i in mine if "#" in i.uri)
        assert {os.path.dirname(i.uri) for i in mine if "#" not in i.uri} == {store}
    aliases = [i for i in inputs if "#" in i.uri]
    assert len(aliases) == 3 * len(stores) and len({i.uri for i in aliases}) == len(aliases)
    # the generic reader's default is unchanged: a root without the opt-in skips an unjudged store, and reading it directly refuses
    unjudged = _build_mirror(pathlib.Path(mirror["root"]).parent / "second", judged_final_fitted=False)
    plain = CS.StoreRoot("full22", unjudged["roots"][2].path, "targets")
    assert CS.enumerate_stores(plain) == []
    with pytest.raises(AB.AdapterRefusal, match="no rows.jsonl"):
        CS.StoreReader(F.dims_map(), "full22", NOW).read(unjudged["stores"][(8192, LORA)], CS.store_inputs(unjudged["stores"][(8192, LORA)]), view=TARGET)


def test_the_module_roots_name_the_durable_mirror_and_the_adapter_is_registered_under_the_source():
    assert set(CS.ORXS_SEEDREP_DECLARED_CAPS) == {512, 2048, 8192} <= set(CS.ORXS_SEEDREP_ROOTS_BY_CAP)
    for cap, candidates in CS.ORXS_SEEDREP_ROOTS_BY_CAP.items():
        assert candidates == (os.path.join(IC.RUN_STATE, "orxs_seedrep", "controls_sampled", f"controls_cap{cap}"),)
    assert [r.path for r in CS.ORXS_SEEDREP_ROOTS] == [c[0] for _, c in sorted(CS.ORXS_SEEDREP_ROOTS_BY_CAP.items())]
    assert all(r.ladder_parent == os.path.join(IC.RUN_STATE, "orxs_seedrep", "ladders") and r.file_unjudged for r in CS.ORXS_SEEDREP_ROOTS)
    assert not ({r.path for r in CS.ORXS_SEEDREP_ROOTS} & {r.path for r in CS.DEFAULT_ROOTS})
    assert CS.ORXS_SEEDREP_MERGED_FILE == os.path.join(IC.RUN_STATE, "orxs_seedrep", "results_final_merged.json")
    assert CS.ORXS_SEEDREP_BY_CAP_FILE == os.path.join(IC.RUN_STATE, "orxs_seedrep", "results_by_rung.json")
    from sisteer.registry.adapters import load_all
    reg = load_all()
    ad = reg["orxs_seedrep"]()
    assert isinstance(ad, CS.OrxsSeedrepAdapter) and ad.name == "orxs_seedrep" and ad.owner == "OR" and ad.covers_sources == ("orxs_seedrep",)
    # 4 / 6: the label-mean row and contrast beside the refused rate and the graded mean
    assert ad.status_sources("anything") == ("orxs_seedrep",) and AB.emission_version_of(ad) == "5" and CS.CellStoreAdapter.EMISSION_VERSION == 10
    assert ad.merged_path == CS.ORXS_SEEDREP_MERGED_FILE and ad.by_cap_path == CS.ORXS_SEEDREP_BY_CAP_FILE
    assert ad.object_resolvers["seed_selection"] is CS.seed_selection_object and ad.object_resolvers["target_grammar"] is CS.orxs_target_object
    # + `mmlu_full_grammar` (2026-09-21): the full-width MMLU roots' resolver (tests/test_registry_adapter_mmlu_full.py); these roots never name it
    assert set(CS.OBJECT_RESOLUTIONS) == {"sha_prefix", "seed_selection", "target_grammar", "mmlu_full_grammar"}
    assert all(r.object_resolution == "target_grammar" for r in CS.ORXS_SEEDREP_ROOTS)
    assert CS.seed_tag_of("qwen35_4b_s2_t0") == "qwen35_4b_s2" and CS.seed_tag_of("qwen35_4b_s3") == "qwen35_4b_s3"
    assert set(CS.PROVENANCE_ONLY_WITHHELD) == {"judgment", "run_state", "run_metric", "item_result", "item_flags"}


# --------------------------------------------------------------------------------------------
# the completion-cap merge
# --------------------------------------------------------------------------------------------

def test_final_cap_row_is_the_scored_merge_and_lower_cap_rows_are_provenance_only(mirror):
    ad = _adapter(mirror)
    recs = {k: _read_group(ad, s) for k, s in mirror["stores"].items()}
    runs = {k: _runs(r) for k, r in recs.items()}
    # the fitted pair escalated to 8,192: the 8,192 row is THE scored row
    for ev, ctl, n_scored, n_total in (("xstest", "xstest", 4, 6), ("or_bench_hard", "or_bench_hard_1k", 4, 4)):
        p = runs[(8192, LORA)][ev]
        assert p["judge_id"] and p["n_items_judged"] == n_scored and p["store_completion_artifact_uri"].endswith("judge_summary.json")
        ex = p["extra"]
        assert ex["completion_cap_merged"] is True and ex["final_rung"] == 8192 and ex["escalation_state"] == "final_at_rung" and "provenance_only_lower_rung" not in ex
        expected_by_cap = {"512": (0 if ctl == "xstest" else 1), "2048": (2 if ctl == "xstest" else 0), "8192": (1 if ctl == "xstest" else 0)}
        assert ex["n_at_cap_by_rung"] == expected_by_cap
        assert ex["residual_capped_at_final"] == expected_by_cap["8192"] and ex["at_cap_share_residual"] == pytest.approx(expected_by_cap["8192"] / n_total)
        assert ex["merged_map"]["uri"] == mirror["merged"] and ex["merged_map"]["sha256"] == AB.check_input(mirror["merged"]).sha256
        assert ex["merged_map"]["key"] == f"s2|{LORA}|{ctl}" and ex["merged_map"]["rate"] == pytest.approx(0.5) and ex["by_rung_map"]["key"] == f"s2|{LORA}|{ctl}|8192"
        assert ex["completion_cap_rule"] == CS.COMPLETION_CAP_RULE and p["field_confidence_json"]["completion_cap_merged"]["confidence"] == "recorded"
        # its residual IS the at-cap share the run files: the finish-reason measure of the final root
        deg = {q["measure"]: q for q in _payloads(recs[(8192, LORA)], "run_degeneration") if q["run_id"] == p["run_id"]}
        assert deg[IC.AT_CAP_FINISH_REASON]["numerator"] == expected_by_cap["8192"] and deg[IC.AT_CAP_FINISH_REASON]["denominator"] == n_total
        headline = {q["metric_id"]: q for q in _payloads(recs[(8192, LORA)], "run_metric") if q["run_id"] == p["run_id"] and q["computed_by"] == "registry"}
        assert headline[CS.OVER_REFUSAL_HEADLINE_METRIC[ctl]]["value"] == pytest.approx(0.5) and headline[CS.OVER_REFUSAL_HEADLINE_METRIC[ctl]]["extra"]["completion_cap_merged"] is True
        assert headline[CS.OVER_REFUSAL_GRADED_METRIC]["extra"]["final_rung"] == 8192
    assert len(_payloads(recs[(8192, LORA)], "judgment")) == 2 and len(_payloads(recs[(8192, LORA)], "run_state")) == 2 and _payloads(recs[(8192, LORA)], "item_result")
    # its 2,048 and 512 rows are PROVENANCE ONLY, although both roots carry judge files
    for cap in (512, 2048):
        assert CS._judged(mirror["stores"][(cap, LORA)])
        kinds = {r["record_type"] for r in recs[(cap, LORA)]}
        assert kinds == {"run", "run_degeneration"}, kinds
        for ev in ("xstest", "or_bench_hard"):
            p = runs[(cap, LORA)][ev]
            ex = p["extra"]
            assert ex["provenance_only_lower_rung"] is True and ex["final_rung"] == 8192 and "completion_cap_merged" not in ex
            assert ex["superseded_by_final_rung_run_id"] == runs[(8192, LORA)][ev]["run_id"] and ex["superseded_by_final_rung_store"] == mirror["stores"][(8192, LORA)]
            assert ex["escalation_state"] == "superseded_by_rung" and ex["superseded_by_run_id"] == runs[(2048 if cap == 512 else 8192, LORA)][ev]["run_id"]
            assert p["lifecycle_state"] == "generated" and p["judge_id"] is None and p["n_items_judged"] is None
            assert p["store_completion_artifact_uri"].endswith("gen_stats.json") and p["n_items_generated"] == (6 if ev == "xstest" else 4)
            assert ex["judged_in_store_withheld"]["n_judged_in_store"] == 4 and "never scored and never filed" in ex["judged_in_store_withheld"]["reason"]
            assert not any(k in p["field_confidence_json"] for k in ("xst_refused", "orb_refused", CS.OVER_REFUSAL_GRADED_METRIC, "n_items_judged"))
            assert p["field_confidence_json"]["provenance_only_lower_rung"]["confidence"] == "recorded"
            assert not any(k in ex for k in ("judge_summary", "store_judge", "over_refusal_headline", "scored_subset"))
            # the generation facts and the at-cap count stay: the finish-reason measure, no judgment behind it
            deg = {q["measure"]: q for q in _payloads(recs[(cap, LORA)], "run_degeneration") if q["run_id"] == p["run_id"]}
            assert deg[IC.AT_CAP_FINISH_REASON]["measurement_state"] == "measured" and all(q["judgment_id"] == "" for q in deg.values())
            assert ex["n_at_cap_by_rung"] == runs[(8192, LORA)][ev]["extra"]["n_at_cap_by_rung"]
    # `none` ended at 2,048: scored there, provenance only at 512
    for ev in ("xstest", "or_bench_hard"):
        assert runs[(2048, NONE)][ev]["extra"]["completion_cap_merged"] is True and runs[(2048, NONE)][ev]["extra"]["final_rung"] == 2048
        assert runs[(2048, NONE)][ev]["extra"]["residual_capped_at_final"] == 0 and runs[(2048, NONE)][ev]["judge_id"]
        assert runs[(512, NONE)][ev]["extra"]["provenance_only_lower_rung"] is True and runs[(512, NONE)][ev]["judge_id"] is None
        assert runs[(512, NONE)][ev]["extra"]["superseded_by_final_rung_run_id"] == runs[(2048, NONE)][ev]["run_id"]
    assert {r["record_type"] for r in recs[(512, NONE)]} == {"run", "run_degeneration"} and len(_payloads(recs[(2048, NONE)], "judgment")) == 2
    # the prompt never left 512: scored at 512 (the ladder's base)
    for ev in ("xstest", "or_bench_hard"):
        p = runs[(512, PROMPTED)][ev]
        assert p["extra"]["completion_cap_merged"] is True and p["extra"]["final_rung"] == 512 and p["extra"]["escalation_state"] == "final_at_base" and p["judge_id"]
    assert len(_payloads(recs[(512, PROMPTED)], "judgment")) == 2
    for r in recs.values():
        assert not _validate_all(r)


def test_contrasts_pair_merged_value_against_the_unsteered_pairs_final_cap_whatever_the_caps(mirror):
    ad = _adapter(mirror)
    none_final = _runs(_read_group(ad, mirror["stores"][(2048, NONE)]))
    none_low = _runs(_read_group(ad, mirror["stores"][(512, NONE)]))
    # the fitted pair's scored row (8,192) against `none`'s scored row (2,048): different caps, expected and recorded
    contrasts = _payloads(_read_group(ad, mirror["stores"][(8192, LORA)]), "run_contrast")
    assert len(contrasts) == 6          # 3 per steered scored row: refused rate, graded mean, label mean
    assert {p["anchor_run_id"] for p in contrasts} == {none_final["xstest"]["run_id"], none_final["or_bench_hard"]["run_id"]}
    assert not ({p["anchor_run_id"] for p in contrasts} & {none_low["xstest"]["run_id"], none_low["or_bench_hard"]["run_id"]})
    for p in contrasts:
        assert p["extra"]["contrast_caps"] == {"steered": 8192, "anchor": 2048} and p["extra"]["completion_cap_merged"] is True
        assert p["extra"]["anchor_store"] == mirror["stores"][(2048, NONE)] and p["anchor_kind"] == "sibling_none" and "cap_matched 0" in p["extra"]["cap_matched_note"]
    xst = next(p for p in contrasts if p["metric_id"] == "xst_refused")
    assert xst["delta"] == pytest.approx(100.0 * (0.5 - 0.0)) and xst["n_pairs"] == 4
    # the prompt's scored row (512) against the same 2,048 anchor
    pr = _payloads(_read_group(ad, mirror["stores"][(512, PROMPTED)]), "run_contrast")
    assert len(pr) == 6 and {p["anchor_run_id"] for p in pr} == {none_final["xstest"]["run_id"], none_final["or_bench_hard"]["run_id"]}   # 3 metrics x 2 controls
    assert all(p["extra"]["contrast_caps"] == {"steered": 512, "anchor": 2048} for p in pr)
    # provenance-only rows contrast nothing; `none` anchors nothing of its own
    for k in ((512, LORA), (2048, LORA), (512, NONE), (2048, NONE)):
        assert _payloads(_read_group(ad, mirror["stores"][k]), "run_contrast") == []


def test_unjudged_final_cap_store_files_generated_and_the_judged_read_keeps_the_ids(tmp_path):
    mirror = _build_mirror(tmp_path, judged_final_fitted=False)
    ad = _adapter(mirror)
    store = mirror["stores"][(8192, LORA)]
    assert not CS._judged(store)
    recs = _read_group(ad, store)
    runs = _runs(recs)
    assert set(runs) == {"xstest", "or_bench_hard"} and {r["record_type"] for r in recs} == {"run", "run_degeneration"}
    for ev, p in runs.items():
        assert p["lifecycle_state"] == "generated" and p["judge_id"] is None and p["n_items_judged"] is None
        assert p["extra"]["completion_cap_merged"] is True and p["extra"]["final_rung"] == 8192 and "provenance_only_lower_rung" not in p["extra"]
        assert p["n_items_generated"] == (6 if ev == "xstest" else 4) and p["store_completion_artifact_uri"].endswith("gen_stats.json")
        assert p["generated_utc"] == "2026-09-21T07:48:18.000Z" and "no rows.jsonl yet" in p["field_confidence_json"]["queued_utc"]["source"]
    assert _payloads(recs, "run_contrast") == [] and not _validate_all(recs)
    ids0 = {ev: p["run_id"] for ev, p in runs.items()}
    # the lower caps are provenance only whatever the state of the final root
    low = _runs(_read_group(ad, mirror["stores"][(2048, LORA)]))
    assert all(p["extra"]["provenance_only_lower_rung"] is True and p["extra"]["superseded_by_final_rung_run_id"] == ids0[ev] for ev, p in low.items())
    # the judge lands (the labels the merged map was written from): the same ids, judged, cross-checked, contrasted
    judge_store(store, LORA, os.path.basename(store), list(CTLS), SPEC[(8192, LORA)]["labels"], SPEC[(8192, LORA)]["at_cap"], n_unsafe=mirror["n_unsafe"])
    recs1 = _read_group(ad, store)
    runs1 = _runs(recs1)
    assert {ev: p["run_id"] for ev, p in runs1.items()} == ids0
    assert {p["run_id"] for p in _payloads(recs1, "judgment")} == set(ids0.values()) and len(_payloads(recs1, "run_contrast")) == 6   # 3 per steered row
    assert all(p["n_items_judged"] == 4 and p["extra"]["completion_cap_merged"] is True and p["extra"]["merged_map"]["rate"] == pytest.approx(0.5) for p in runs1.values())
    assert not _validate_all(recs1)


def test_merged_map_cross_checks_refuse_the_store_by_name(mirror):
    ad = _adapter(mirror)
    store = mirror["stores"][(8192, LORA)]
    key = f"s2|{LORA}|xstest"
    original = json.load(open(mirror["merged"]))

    def tamper(fn):
        d = json.loads(json.dumps(original))
        fn(d["rows"])
        _json(mirror["merged"], d)

    tamper(lambda rows: rows[key].__setitem__("rate", 0.75))
    with pytest.raises(AB.AdapterRefusal, match="not what the final-cap root reproduces"):
        _read_group(ad, store)
    tamper(lambda rows: rows[key]["n_at_cap_by_rung"].__setitem__("8192", 3))
    with pytest.raises(AB.AdapterRefusal, match="n_at_cap_by_rung"):
        _read_group(ad, store)
    tamper(lambda rows: rows[key].__setitem__("final_rung", 2048))
    with pytest.raises(AB.AdapterRefusal, match="two authorities disagree"):
        _read_group(ad, store)
    tamper(lambda rows: rows.pop(key))
    with pytest.raises(AB.AdapterRefusal, match="no row in"):
        _read_group(ad, store)
    tamper(lambda rows: rows[key].__setitem__("residual_capped_at_final", 0))
    with pytest.raises(AB.AdapterRefusal, match="residual_capped_at_final"):
        _read_group(ad, store)
    # a lower-cap root is cross-checked on its own at-cap count too
    tamper(lambda rows: rows[key]["n_at_cap_by_rung"].__setitem__("2048", 9))
    with pytest.raises(AB.AdapterRefusal, match="n_at_cap_by_rung\\[2048\\]"):
        _read_group(ad, mirror["stores"][(2048, LORA)])
    _json(mirror["merged"], original)
    # the by-cap file disagreeing on the final cap's rate refuses too
    by_cap = json.load(open(mirror["by_rung"]))
    by_cap["main_mirror"][f"{key}|8192"]["rate"] = 0.25
    _json(mirror["by_rung"], by_cap)
    with pytest.raises(AB.AdapterRefusal, match="results_by_rung.json says rate"):
        _read_group(ad, store)
    del by_cap["main_mirror"][f"{key}|8192"]
    _json(mirror["by_rung"], by_cap)
    with pytest.raises(AB.AdapterRefusal, match="no row in .*results_by_rung.json"):
        _read_group(ad, store)
    # with the files restored the store reads again; without a result file no group reads at all
    write_result_maps(mirror["root"], mirror["stores"], EXITS, SPEC, n_unsafe=mirror["n_unsafe"])
    assert len(_payloads(_read_group(ad, store), "judgment")) == 2
    os.remove(mirror["merged"])
    with pytest.raises(AB.AdapterRefusal, match="not among the group's inputs"):
        _read_group(ad, store)


# --------------------------------------------------------------------------------------------
# spellings, objects, substrate, cap index
# --------------------------------------------------------------------------------------------

def test_view_spelling_producer_key_objects_substrate_and_cap_index(mirror):
    ad = _adapter(mirror)
    for (cap, cell_id), store in sorted(mirror["stores"].items()):
        runs = _payloads(_read_group(ad, store), "run")
        assert len(runs) == 2
        for p in runs:
            assert p["view_legacy"] == SEED_TAG and p["extra"]["view_legacy_dir"] == TARGET
            assert p["producer_run_key_legacy"] == f"{TARGET}/{os.path.basename(store)}"
            assert p["run_id"] == I.run_id("full22", p["producer_run_key_legacy"], p["producer_object_key_legacy"], p["eval_id"], p["instrument_id"])
            assert p["lane_source_legacy"] == "orxs_seedrep" and p["model_id"] == F.MODEL_IT and p["seed_base"] == 2 and p["trial_index"] == 0
            assert p["thinking_mode"] == "off" and p["decode_seed_rule"] == "per_item" and p["token_cap_tokens"] == cap
            assert p["token_cap_rung_index"] == {512: 0, 2048: 1, 8192: 2}[cap] and p["token_cap_ladder_json"] == [512, 2048, 8192]
            assert p["substrate"] == "cluster_a" and p["substrate_source"] == "derived"
            assert p["field_confidence_json"]["substrate"]["source"].startswith("manifest.json:routing_table (/workspace-vast/")
            assert p["extra"]["ladder_exit"]["ladder_sha256"] == AB.check_input(mirror["ladder"]).sha256
            if cell_id == NONE:
                assert p["object_id"] == "none" and p["transfer_kind"] == "none" and p["serving_path"] == "passthrough" and "strength_nominal" not in p
            elif cell_id == PROMPTED:
                assert p["object_id"] == F.SI_OBJECT_ID and p["transfer_kind"] == "prompted_only" and "strength_nominal" not in p
            else:
                assert p["object_id"] == F.OBJ_SHA and p["transfer_kind"] == "base_to_instruct" and p["serving_path"] == "lora_native"
                assert p["strength_nominal"] == 1.0 and p["is_operating_point"] == 1 and p["applied_positions_mode"] == "all"
            assert p["field_confidence_json"]["object_id"] == {"confidence": "recorded", "source": f"synthetic roster {SEED_TAG}"} or cell_id == NONE


# --------------------------------------------------------------------------------------------
# the ladder's exit
# --------------------------------------------------------------------------------------------

def test_ladder_exit_is_recorded_per_store_and_control_in_the_seed_replicate_vocabulary(mirror):
    from sisteer.registry.adapters import relu_paired as RP
    ad = _adapter(mirror)
    runs = {k: _runs(_read_group(ad, store)) for k, store in mirror["stores"].items()}
    for ev in ("xstest", "or_bench_hard"):
        # the fitted pair left at 8,192: 512 names the 2,048 run as the cap it was re-drawn into, 2,048 names 8,192, 8,192 is final
        assert runs[(512, LORA)][ev]["extra"]["superseded_by_run_id"] == runs[(2048, LORA)][ev]["run_id"]
        assert runs[(2048, LORA)][ev]["extra"]["superseded_by_run_id"] == runs[(8192, LORA)][ev]["run_id"]
        assert runs[(8192, LORA)][ev]["extra"]["escalation_state"] == "final_at_rung" and runs[(8192, LORA)][ev]["extra"]["superseded_by_run_id"] is None
        for cap in (512, 2048):
            le = runs[(cap, LORA)][ev]["extra"]["ladder_exit"]
            assert runs[(cap, LORA)][ev]["extra"]["escalation_state"] == "superseded_by_rung" and le["basis"] == "declared_exit" and le["exit"] == "exit_le_1pct" and le["exit_cap"] == 8192
            assert le["caps_attempted"] == [512, 2048, 8192] and le["decision_at_cap"] == "escalate"
        # `none` left at 2,048; the prompt never left 512
        assert runs[(512, NONE)][ev]["extra"]["escalation_state"] == "superseded_by_rung" and runs[(512, NONE)][ev]["extra"]["superseded_by_run_id"] == runs[(2048, NONE)][ev]["run_id"]
        assert runs[(2048, NONE)][ev]["extra"]["escalation_state"] == "final_at_rung"
        assert runs[(512, PROMPTED)][ev]["extra"]["escalation_state"] == "final_at_base" and runs[(512, PROMPTED)][ev]["extra"]["ladder_exit"]["exit_cap"] == 512
    for per in runs.values():
        for p in per.values():
            assert p["extra"]["escalation_state"] in RP.ESCALATION_STATES and p["field_confidence_json"]["escalation_state"]["confidence"] == "recorded"
            assert p["extra"]["ladder_exit"]["ladder_uri"] == mirror["ladder"] and p["extra"]["ladder_exit"]["ladder_caps"] == [512, 2048, 8192]
            assert "superseded_by_run_id" not in p and "status_ruling_id" not in p       # recorded in extra only: no run column, no ruling
    for store in mirror["stores"].values():
        recs = _read_group(ad, store)
        assert not [r for r in recs if r["record_type"] == "ruling"] and not [p for p in _payloads(recs, "run_state") if p["state"] != "judged"]
    # the pure function over shapes the record can take
    assert CS.ladder_exit_state(None, 512, 512)["escalation_state"] is None
    pending = {"rungs_attempted": [512, 2048], "n_at_cap": {"512": 9, "2048": 3}, "decisions": {"512": {"decision": "escalate"}}, "left_ladder_at": {"status": "undecided"}}
    assert CS.ladder_exit_state(pending, 2048, 512)["escalation_state"] == "escalation_pending"
    assert CS.ladder_exit_state(pending, 512, 512) == dict(CS.ladder_exit_state(pending, 512, 512), escalation_state="superseded_by_rung", superseded_by_cap=2048, basis="decision")
    above = {"rungs_attempted": [512], "left_ladder_at": {"status": "left", "rung": 512, "exit": "exit_le_1pct"}}
    assert CS.ladder_exit_state(above, 512, 512)["escalation_state"] == "final_at_base"
    out = CS.ladder_exit_state(above, 2048, 512)
    assert out["escalation_state"] is None and "above the ladder's declared exit" in out["note"]
    # a pair the ladder record does not name has no final cap: under the completion-cap merge the store refuses by name
    ladder = json.load(open(mirror["ladder"]))
    del ladder["cells"][PROMPTED]
    _json(mirror["ladder"], ladder)
    with pytest.raises(AB.AdapterRefusal, match="declares no exit"):
        _read_group(ad, mirror["stores"][(512, PROMPTED)])


# --------------------------------------------------------------------------------------------
# objects: refused by name, never minted
# --------------------------------------------------------------------------------------------

def test_a_store_whose_object_the_roster_does_not_name_or_whose_sha_disagrees_is_refused(mirror):
    ad = _adapter(mirror)
    unknown = write_store(mirror["caps_parent"], 512, UNKNOWN, judged_controls=("xstest",), object_sha16="0123456789abcdef")
    with pytest.raises(IC.Unresolved, match="names no object"):
        _read_group(ad, unknown)
    shutil.rmtree(unknown)
    mismatch = write_store(mirror["caps_parent"], 512, LORA, judged_controls=("xstest",), object_sha16="ffffffffffffffff", target="qwen35_4b_s2_t1")
    with pytest.raises(AB.AdapterRefusal, match="no prefix"):
        _read_group(ad, mismatch)
    shutil.rmtree(os.path.dirname(os.path.dirname(mismatch)))
    assert len(_payloads(_read_group(ad, mirror["stores"][(8192, LORA)]), "run")) == 2


def test_the_real_seed_selection_resolver_reads_the_checkout_roster_and_pins_and_refuses_by_name():
    """`seed_selection_object` over THIS checkout's roster / pins / si_texts (the declaration itself): the s2 / s3 selections name the ten
    seed objects whose sha16 the real manifests carry; a sha16 of another object refuses; a store id the selection does not name is
    unresolved; a target directory spelling no seed refuses; a connected registry without the object row is unresolved."""
    import yaml
    from sisteer.registry.adapters import declared_common as DC
    roster = yaml.safe_load(open(DC.repo_path("configs", "sdfbase_full22_roster.yaml")))
    pins = json.load(open(DC.repo_path("configs", "sdfbase_full22_pins.json")))
    for sel in ("s2", "s3"):
        for key, staged in roster["targets"]["qwen35_4b"]["views"][sel]["cells"].items():
            if not staged:
                continue
            sha = pins["qwen35_4b"][staged]["object_sha256"]
            oid, src = CS.seed_selection_object(key, f"qwen35_4b_{sel}_t0", sha[:16], None)
            assert oid == sha and staged in src and f"seed selection {sel}" in src
    assert CS.seed_selection_object("none", "qwen35_4b_s2_t0", None, None)[0] == "none"
    si = LC.Seeds.locate().si_object_id("safety_refusal")
    assert si and CS.seed_selection_object("prompted__safety_refusal", "qwen35_4b_s3_t0", None, None)[0] == si
    with pytest.raises(AB.AdapterRefusal, match="no prefix"):
        CS.seed_selection_object("constv2-mix-full__psr_ce_all", "qwen35_4b_s2_t0", "ffffffffffffffff", None)
    with pytest.raises(AB.AdapterRefusal, match="names no object_sha"):
        CS.seed_selection_object("constv2-mix-full__psr_ce_all", "qwen35_4b_s2_t0", None, None)
    with pytest.raises(IC.Unresolved):
        CS.seed_selection_object(UNKNOWN, "qwen35_4b_s2_t0", "0123456789abcdef", None)
    with pytest.raises(AB.AdapterRefusal, match="spells no seed selection"):
        CS.seed_selection_object("none", "qwen35_4b_t0", None, None)
    conn = DB.connect_memory()
    from sisteer.registry.migrations import apply_migrations
    apply_migrations(conn)
    with pytest.raises(IC.Unresolved, match="no object row"):
        CS.seed_selection_object("constv2-mix-full__psr_ce_all", "qwen35_4b_s2_t0", "2d43a58e55e674d3", conn)


# --------------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------------

def test_adapter_pass_and_ingest_land_scored_rows_keep_provenance_rows_generated_and_a_judged_read_moves_states(tmp_path):
    """The harvester's completion rule is `min(n_items_generated, item_set.n_items)` judged (schema_def.required_coverage_sql): on the real
    stores min(450, 250) = 250, met by the 250 scored safe prompts. The synthetic item set still declares 250 while a store here generates
    a handful, so this mirror carries no unsafe items. The fitted pair's final root is unjudged at the first pass."""
    mirror = _build_mirror(tmp_path, n_unsafe=0, judged_final_fitted=False)
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    ad = _adapter(mirror)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 6, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    # the first pass over an empty registry parks every merged comparison (its own `none` is not folded yet: no provisional anchors);
    # the pass after the fold re-reads the steered group on the moved anchor-decision input and files the contrasts against the now-current `none`
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    rep1b = RUN.run_pass(root, [ad], conn=conn, now="2026-09-21T10:30:00.000Z")
    conn.close()
    assert rep1b.groups_refused == 0 and rep1b.groups_refreshed == 1 and rep1b.anchor_decisions_changed == 1, [r.error for r in rep1b.results]
    ing1b = H.ingest_once(root, workers=0, min_free=0)
    assert ing1b.refused == 0, ing1b.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = conn.execute("SELECT run_id, eval_id, lifecycle_state, view_legacy, lane_source_legacy, token_cap_tokens, campaign_cell_id_legacy, "
                        "json_extract(extra_json, '$.provenance_only_lower_rung'), json_extract(extra_json, '$.completion_cap_merged') FROM run ORDER BY run_id").fetchall()
    assert len(rows) == 12 and all(r[3] == SEED_TAG and r[4] == "orxs_seedrep" for r in rows)
    states = {(r[6], r[5], r[1]): (r[2], r[7], r[8]) for r in rows}
    scored_judged = {(NONE, 2048), (PROMPTED, 512)}
    for (cell_id, cap, ev), (state, prov, merged) in states.items():
        if (cell_id, cap) in scored_judged:
            assert (state, prov, merged) == ("ingested", None, 1), (cell_id, cap, ev, state)
        elif (cell_id, cap) == (LORA, 8192):
            assert (state, prov, merged) == ("generated", None, 1), (cell_id, cap, ev, state)      # scored row, judge not landed yet
        else:
            assert (state, prov, merged) == ("generated", 1, None), (cell_id, cap, ev, state)      # provenance only: never judged, never complete
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 6              # the prompt's scored row against `none`'s (2,048): 3 metrics x 2 controls
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.contrast_caps.steered'), json_extract(extra_json, '$.contrast_caps.anchor') FROM run_contrast")] == [(512, 2048)]
    assert conn.execute("SELECT COUNT(*) FROM item_result ir JOIN run r ON r.run_id = ir.run_id WHERE json_extract(r.extra_json, '$.provenance_only_lower_rung') = 1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE json_extract(r.extra_json, '$.provenance_only_lower_rung') = 1").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM ruling").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run WHERE superseded_by_run_id IS NOT NULL OR status_ruling_id IS NOT NULL").fetchone()[0] == 0
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT lane_source_legacy, owner, adapter FROM source_status").fetchall()] == [("orxs_seedrep", "OR", "orxs_seedrep")]
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = ?", (ad.name,)).fetchall()] == [("5",)]   # the anchor currency (adapter re-anchoring)
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.anchor_selection.state'), anchor_kind FROM run_contrast").fetchall()] == [("own_current", "sibling_none")]
    ids_before = {r[0] for r in rows}
    conn.close()
    # the judge lands on the fitted pair's final root: one group republishes, its two rows reach ingested, four contrasts join, nothing new is minted
    store = mirror["stores"][(8192, LORA)]
    judge_store(store, LORA, os.path.basename(store), list(CTLS), SPEC[(8192, LORA)]["labels"], SPEC[(8192, LORA)]["at_cap"], n_unsafe=mirror["n_unsafe"])
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep2 = RUN.run_pass(root, [ad], conn=conn, now="2026-09-21T11:00:00.000Z")
    conn.close()
    assert rep2.groups_refused == 0 and rep2.groups_refreshed == 1, [r.error for r in rep2.results]
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0, ing2.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows2 = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT run_id, lifecycle_state, json_extract(extra_json, '$.provenance_only_lower_rung') FROM run").fetchall()}
    assert set(rows2) == ids_before
    assert all(state == ("generated" if prov == 1 else "ingested") for state, prov in rows2.values())
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 6
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 12           # 3 per steered scored row (including the label mean)
    assert sorted(tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.contrast_caps.steered'), json_extract(extra_json, '$.contrast_caps.anchor') FROM run_contrast")) == [(512, 2048), (8192, 2048)]
    conn.close()


# --------------------------------------------------------------------------------------------
# the seed-1 tree: judged at each pair's final cap only (a regression pin; the reader read it unchanged)
# --------------------------------------------------------------------------------------------

#: per seed-1 store id: the judge labels cycled over the items (the final root's labels are what the merged map is written from) and the
#: at-cap items the producer recorded at each cap — many capped at 512, a few at 2,048, one or none at the ladder's end, as on the real tree
def _s1_spec(cap: int, store_id: str) -> dict:
    k = S1_STORE_IDS.index(store_id)
    labels = {"xstest": DEFAULT_LABELS[k % 3:] + DEFAULT_LABELS[:k % 3], "or_bench_hard_1k": DEFAULT_LABELS[(k + 1) % 3:] + DEFAULT_LABELS[:(k + 1) % 3]}
    if cap == 512:
        at_cap = {"xstest": ["xst0000-cccccccc", "xst0002-cccccccc"], "or_bench_hard_1k": ["orb0000-eeeeeeee", "orb0001-eeeeeeee", "orb0003-eeeeeeee"]}
    elif cap == 2048:
        at_cap = {"xstest": ["xst0001-cccccccc"] if S1_EXITS[store_id]["xstest"] > 2048 else [],
                  "or_bench_hard_1k": ["orb0002-eeeeeeee"] if S1_EXITS[store_id]["or_bench_hard_1k"] > 2048 else []}
    else:
        at_cap = {"xstest": ["xst0003-cccccccc"] if store_id == DIFF_LORA else [], "or_bench_hard_1k": []}
    return {"labels": labels, "at_cap": at_cap}


def _build_seed1_mirror(tmp_path, n_unsafe: int = 2) -> dict:
    """A synthetic durable mirror with THREE target directories under one `controls_sampled/`: the seed-1 directory in the REAL seed-1 shape
    (a re-run judged at each pair's final cap only) beside the s2 directory of :func:`_build_mirror` and an s3 copy of it, the two result
    files naming all three seeds, one ladder record per directory.

    Seed 1: the eight store ids at 512 — every one generated on both controls and NONE judged (the base cap is never a final cap on this tree);
    the eight at 2,048 — both controls generated, judge files ONLY for the controls whose ladder exited there (`none` and the two prompts on
    both; the constitution ReLU-steer object and the control LoRA on XSTest alone; the constitution LoRA on OR-Bench alone; the control
    ReLU-steer object and the LoRA difference on neither: two stores with no judge files at all); five at 8,192 whose manifests name only the
    controls that climbed, judged on those. Sixteen (store id, control) pairs, sixteen final caps, 21 stores. ``stores`` is keyed
    (seed, cap, store id)."""
    root = str(tmp_path / "orxs_seedrep")
    caps_parent = os.path.join(root, "controls_sampled")
    ladders = os.path.join(root, "ladders")
    stores: Dict[tuple, str] = {}
    spec: Dict[tuple, dict] = {}
    for store_id in S1_STORE_IDS:
        exits = S1_EXITS[store_id]
        sha16 = S1_OBJECTS[store_id][:16] if store_id in S1_FITTED else None
        for cap in CAPS:
            generated = [c for c in CTLS if exits[c] >= cap]
            if not generated:
                continue
            judged = [c for c in generated if exits[c] == cap]
            sp = _s1_spec(cap, store_id)
            spec[(cap, store_id)] = sp
            lower = [c for c in CAPS if c < cap]
            stores[("s1", cap, store_id)] = write_store(caps_parent, cap, store_id, judged_controls=judged, labels=sp["labels"], at_cap=sp["at_cap"],
                                                        job=S1_JOB, seed_base=1, escalated_from=(lower[-1] if lower else None), object_sha16=sha16,
                                                        target=S1_TARGET, n_unsafe=n_unsafe, controls=generated)
    s1_stores = {(cap, sid): d for (seed, cap, sid), d in stores.items() if seed == "s1"}
    ladder_s1 = write_ladder(ladders, S1_TARGET, S1_EXITS)
    # the s2 directory of `_build_mirror` and an s3 copy of it (seed base 3, its own fitted object, its own job)
    per_seed = {"s2": (TARGET, 2, F.OBJ16, {}), "s3": (S3_TARGET, 3, S3_LORA_OBJECT[:16], {"job": "586140"})}
    ladders_by_seed = {"s1": ladder_s1}
    for seed, (target, seed_base, sha16, over) in per_seed.items():
        for (cap, store_id), sp in SPEC.items():
            lower = [c for c in CAPS if c < cap]
            stores[(seed, cap, store_id)] = write_store(caps_parent, cap, store_id, judged_controls=CTLS, labels=sp["labels"], at_cap=sp["at_cap"],
                                                        job=over.get("job", sp["job"]), seed_base=seed_base, escalated_from=(lower[-1] if lower else None),
                                                        object_sha16=(sha16 if store_id == LORA else None), target=target, n_unsafe=n_unsafe)
        ladders_by_seed[seed] = write_ladder(ladders, target, EXITS)
    parts = [dict(stores=s1_stores, exits=S1_EXITS, spec=spec, seed="s1", target=S1_TARGET, n_unsafe=n_unsafe)]
    for seed, (target, _sb, _sha, _over) in per_seed.items():
        parts.append(dict(stores={(cap, sid): d for (sd, cap, sid), d in stores.items() if sd == seed}, exits=EXITS, spec=SPEC, seed=seed, target=target, n_unsafe=n_unsafe))
    maps = write_result_maps_seeds(root, parts)
    roots = CS.orxs_seedrep_roots(parents=(caps_parent,), declared=(512, 2048, 8192), ladders=ladders)
    return {"root": root, "caps_parent": caps_parent, "ladders": ladders, "ladder": ladder_s1, "ladders_by_seed": ladders_by_seed, "stores": stores, "spec": spec,
            "roots": roots, "n_unsafe": n_unsafe, "merged": maps["merged"], "by_rung": maps["by_rung"]}


#: the seed-1 (store id, control, cap) rows the completion-cap merge SCORES — nine at 2,048, seven at 8,192 — and the ones it files as
#: provenance only — every pair at 512, the seven climbed pairs at 2,048
S1_SCORED = {(sid, ctl, cap) for sid, per in S1_EXITS.items() for ctl, cap in per.items()}
S1_PROVENANCE = {(sid, ctl, 512) for sid, per in S1_EXITS.items() for ctl in per} | {(sid, ctl, 2048) for sid, per in S1_EXITS.items() for ctl, cap in per.items() if cap == 8192}


def test_seed1_scores_sixteen_final_cap_rows_and_files_every_unjudged_lower_cap_as_provenance_only(tmp_path):
    """The seed-1 shape end to end at the reader: the 512 root and the climbed pairs' 2,048 stores carry NO judge files (the first tree where a
    lower-cap root is unjudged on disk, not judged-and-withheld) and file as the same provenance-only row v2 files for s2 / s3 — generated, no
    judge, no judgment / metric / item / state row, `judged_in_store_withheld.n_judged_in_store` None, the ladder's `superseded_by_rung`, no
    pending or undetermined state, no disposition, no refusal; the final-cap rows are the sixteen scored merges."""
    m = _build_seed1_mirror(tmp_path)
    ad = _adapter(m)
    recs = {k: _read_group(ad, d) for k, d in m["stores"].items() if k[0] == "s1"}
    runs: Dict[tuple, Dict] = {}
    for (_seed, cap, sid), r in recs.items():
        for p in _payloads(r, "run"):
            runs[(sid, p["extra"]["control"], cap)] = p
    assert len(recs) == 21 and set(runs) == S1_SCORED | S1_PROVENANCE and len(S1_SCORED) == 16 and len(S1_PROVENANCE) == 23
    assert sum(1 for (_s, _c, cap) in S1_SCORED if cap == 2048) == 9 and sum(1 for (_s, _c, cap) in S1_SCORED if cap == 8192) == 7
    assert sum(1 for (_s, _c, cap) in S1_PROVENANCE if cap == 512) == 16 and sum(1 for (_s, _c, cap) in S1_PROVENANCE if cap == 2048) == 7
    # the base cap: eight stores, no judge files anywhere, enumerated as generated
    for sid in S1_STORE_IDS:
        store = m["stores"][("s1", 512, sid)]
        assert not CS._judged(store) and CS._generated(store) and store in {os.path.dirname(i.uri) for i in ad.discover() if "#" not in i.uri}
        assert {r["record_type"] for r in recs[("s1", 512, sid)]} == {"run", "run_degeneration"}
    # the two 2,048 stores of pairs that climbed on both controls carry no judge files either; the three that parted ways carry them for one control
    for sid in (CTL_RELU, DIFF_LORA):
        assert not CS._judged(m["stores"][("s1", 2048, sid)]) and {r["record_type"] for r in recs[("s1", 2048, sid)]} == {"run", "run_degeneration"}
    for sid, judged_ctl in ((RELU, "xstest"), (CTL_LORA, "xstest"), (LORA, "or_bench_hard_1k")):
        store = m["stores"][("s1", 2048, sid)]
        assert CS._judged(store)
        assert {json.loads(l)["control"] for l in open(os.path.join(store, "judge_items.jsonl"))} == {judged_ctl}
        assert {p["run_id"] for p in _payloads(recs[("s1", 2048, sid)], "judgment")} == {runs[(sid, judged_ctl, 2048)]["run_id"]}
    # the 8,192 stores generated only the controls that climbed: one run per generated control, all scored
    for sid, per in S1_EXITS.items():
        climbed = [c for c in CTLS if per[c] == 8192]
        if climbed:
            assert sorted(p["extra"]["control"] for p in _payloads(recs[("s1", 8192, sid)], "run")) == sorted(climbed)
        else:
            assert ("s1", 8192, sid) not in m["stores"]
    for (sid, ctl, cap), p in runs.items():
        ex = p["extra"]
        assert p["view_legacy"] == S1_TAG and ex["view_legacy_dir"] == S1_TARGET and p["seed_base"] == 1 and p["lane_source_legacy"] == "orxs_seedrep"
        assert p["producer_run_key_legacy"].startswith(f"{S1_TARGET}/") and p["cluster_job_id"] == S1_JOB and p["substrate"] == "cluster_a"
        assert ex["escalation_state"] not in ("escalation_pending", "undetermined") and p["token_cap_tokens"] == cap
        assert ex["merged_map"]["key"] == f"s1|{sid}|{ctl}"
        if (sid, ctl, cap) in S1_SCORED:
            assert ex["completion_cap_merged"] is True and ex["final_rung"] == cap and "provenance_only_lower_rung" not in ex
            assert p["judge_id"] and p["lifecycle_state"] == "generated" and p["store_completion_artifact_uri"].endswith("judge_summary.json")
            assert ex["escalation_state"] == "final_at_rung" and p["n_items_judged"] == 4
        else:
            assert ex["provenance_only_lower_rung"] is True and "completion_cap_merged" not in ex and ex["final_rung"] == S1_EXITS[sid][ctl]
            assert p["judge_id"] is None and p["n_items_judged"] is None and p["lifecycle_state"] == "generated"
            assert p["store_completion_artifact_uri"].endswith("gen_stats.json") and ex["escalation_state"] == "superseded_by_rung"
            # NO judge files behind this row: nothing was judged in the store, so nothing is withheld — the row says so, and is not pending
            assert ex["judged_in_store_withheld"] == {"n_judged_in_store": None, "reason": CS.PROVENANCE_ONLY_REASON}
            assert ex["superseded_by_final_rung_run_id"] == runs[(sid, ctl, S1_EXITS[sid][ctl])]["run_id"]
            assert not any(k in ex for k in ("judge_summary", "store_judge", "over_refusal_headline", "scored_subset"))
    # per store: the record kinds a provenance-only run leaves behind, no disposition anywhere, every payload valid
    for k, r in recs.items():
        assert not [x for x in r if x["record_type"] in ("disposition", "ruling")]
        prov_ids = {p["run_id"] for p in _payloads(r, "run") if p["extra"].get("provenance_only_lower_rung")}
        for x in r:
            if x["payload"].get("run_id") in prov_ids:
                assert x["record_type"] in ("run", "run_degeneration"), (k, x["record_type"])
        assert not _validate_all(r), k
    assert sum(len(_payloads(r, "judgment")) for r in recs.values()) == 16
    # objects: the five fitted seed-1 objects under their OWN ids (the roster's s1 selection), `none` and the two prompts the same object as on s2
    s2_runs = {p["extra"]["control"]: p for p in _payloads(_read_group(ad, m["stores"][("s2", 2048, NONE)]), "run")}
    assert {runs[(sid, "xstest", 512)]["object_id"] for sid in S1_FITTED} == set(S1_OBJECTS.values()) and F.OBJ_SHA not in set(S1_OBJECTS.values())
    assert runs[(NONE, "xstest", 2048)]["object_id"] == s2_runs["xstest"]["object_id"] == "none"
    assert runs[(PROMPTED, "xstest", 2048)]["object_id"] == F.SI_OBJECT_ID and runs[(SI_CONST, "xstest", 2048)]["object_id"] == SI_CONST_OBJECT_ID
    assert runs[(NONE, "xstest", 2048)]["run_id"] != s2_runs["xstest"]["run_id"]        # one object, two seed views: two runs (the coexist ruling's case)
    for sid in S1_FITTED:
        p = runs[(sid, "xstest", 512)]
        assert p["field_confidence_json"]["object_id"] == {"confidence": "recorded", "source": f"synthetic roster {S1_TAG}"}
        assert p["strength_nominal"] == 1.0 and p["is_operating_point"] == 1 and p["transfer_kind"] == "base_to_instruct"
    # contrasts: every scored steered control against the seed-1 `none` at ITS final cap (2,048), never s2's; 14 pairs x 2 metrics
    none_final = {ctl: runs[(NONE, ctl, 2048)]["run_id"] for ctl in CTLS}
    contrasts = [c for r in recs.values() for c in _payloads(r, "run_contrast")]
    assert len(contrasts) == 42 and {c["anchor_run_id"] for c in contrasts} == set(none_final.values())    # 14 steered rows x 3 metrics
    assert sorted({(c["extra"]["contrast_caps"]["steered"], c["extra"]["contrast_caps"]["anchor"]) for c in contrasts}) == [(2048, 2048), (8192, 2048)]
    prov_run_ids = {p["run_id"] for (sid, ctl, cap), p in runs.items() if (sid, ctl, cap) in S1_PROVENANCE}
    assert not [c for c in contrasts if c["run_id"] in prov_run_ids or c["anchor_run_id"] in prov_run_ids]


def test_seed1_result_files_name_three_seeds_and_the_other_seeds_read_unchanged(tmp_path):
    m = _build_seed1_mirror(tmp_path)
    merged = json.load(open(m["merged"]))
    assert set(merged["seeds"]) == {"s1", "s2", "s3"} and {k.split("|", 1)[0] for k in merged["rows"]} == {"s1", "s2", "s3"}
    assert sum(1 for k in merged["rows"] if k.startswith("s1|")) == 16 and {merged["rows"][k]["final_rung"] for k in merged["rows"] if k.startswith("s1|")} == {2048, 8192}
    ad = _adapter(m)
    for seed, target, tag, obj, seed_base in (("s2", TARGET, SEED_TAG, F.OBJ_SHA, 2), ("s3", S3_TARGET, S3_TAG, S3_LORA_OBJECT, 3)):
        p = _runs(_read_group(ad, m["stores"][(seed, 8192, LORA)]))["xstest"]
        assert p["view_legacy"] == tag and p["object_id"] == obj and p["seed_base"] == seed_base and p["extra"]["merged_map"]["key"] == f"{seed}|{LORA}|xstest"
        assert p["extra"]["completion_cap_merged"] is True and p["extra"]["final_rung"] == 8192 and p["producer_run_key_legacy"] == f"{target}/{os.path.basename(m['stores'][(seed, 8192, LORA)])}"
        low = _runs(_read_group(ad, m["stores"][(seed, 512, LORA)]))["xstest"]
        assert low["extra"]["provenance_only_lower_rung"] is True and low["extra"]["judged_in_store_withheld"]["n_judged_in_store"] == 4    # judged in the store, withheld
    # the same fitted store id names three objects across the three seed views
    assert len({_runs(_read_group(ad, m["stores"][(seed, 512, LORA)]))["xstest"]["object_id"] for seed in ("s1", "s2", "s3")}) == 3


def _seed_object_records(root: str) -> None:
    """Object rows for the synthetic seed-1 / seed-3 fitted objects and the second prompt with its si_text parent (the live registry carries them
    from the objects bank and the seeds; the fixtures' dimension records carry only s2's). Every fitted one is written as a LoRA row: the run's
    object KIND is spelled from the store id, never read off the row, and the ingest needs the parent row alone."""
    from sisteer.registry import records as R
    from sisteer.registry import synth
    labels = {LORA: ("constitution_lora", "constitution"), RELU: ("constitution_relu", "constitution"), CTL_LORA: ("control_lora", "control"),
              CTL_RELU: ("control_relu", "control"), DIFF_LORA: ("difference_lora", "difference")}
    out = [synth.rec("object", synth.object_lora(S1_OBJECTS[c], *labels[c]), synth.t(200 + i)) for i, c in enumerate(S1_FITTED)]
    out.append(synth.rec("object", synth.object_lora(S3_LORA_OBJECT, "constitution_lora", "constitution"), synth.t(210)))
    out.append(synth.rec("si_text", {"si_id": "synthetic_constitution", "si_text_sha256": SI_CONST_SHA, "si_kind": "constitution_short",
                                     "source_uri": "synthetic://si/synthetic_constitution.txt"}, synth.t(209)))
    out.append(synth.rec("object", {"object_id": SI_CONST_OBJECT_ID, "object_kind": "system_instruction", "paper_label": "constitution_si", "campaign": synth.CAMPAIGN,
                                    "object_role": "prompt", "fit_target_kind": "declared_prompt", "si_text_sha256": SI_CONST_SHA, "si_token_count": 40,
                                    "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable", "fit_positions": "not_applicable",
                                    "fit_seed_confidence": "not_applicable", "derivation": "none", "artifact_sha256": SI_CONST_SHA, "status": "approved"}, synth.t(211)))
    R.write_records(os.path.join(root, "records", "dims", "000002-seed-objects.jsonl"), out)


def test_seed1_pass_and_ingest_land_sixteen_scored_rows_and_keep_twenty_three_provenance_rows_generated(tmp_path):
    """The three-seed mirror through the pass and the harvester: every seed-1 final-cap row reaches `ingested`, every unjudged lower-cap row
    stays `generated` with no judgment / item / metric behind it, nothing is held or blocked, the s2 / s3 groups land as before. As in the other
    end-to-end test the mirror carries no unsafe XSTest items (the synthetic item set declares 250)."""
    m = _build_seed1_mirror(tmp_path, n_unsafe=0)
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    _seed_object_records(root)
    ad = _adapter(m)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_held == 0 and rep.groups_refreshed == 21 + 6 + 6, [r.error for r in rep.results if r.error]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    # the first pass parks every merged comparison (the own `none` is not folded yet: no provisional anchors); the pass after the
    # fold re-reads each steered scored group on its moved anchor-decision input and files its contrasts against the now-current `none`
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    rep2 = RUN.run_pass(root, [ad], conn=conn, now="2026-09-21T10:30:00.000Z")
    conn.close()
    assert rep2.groups_refused == 0 and rep2.groups_held == 0 and rep2.anchor_decisions_changed == rep2.groups_refreshed > 0, [r.error for r in rep2.results if r.error]
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0, ing2.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = conn.execute("SELECT run_id, eval_id, lifecycle_state, view_legacy, token_cap_tokens, campaign_cell_id_legacy, judge_id, n_items_judged, "
                        "json_extract(extra_json, '$.provenance_only_lower_rung'), json_extract(extra_json, '$.completion_cap_merged'), object_id, seed_base "
                        "FROM run WHERE view_legacy = ? ORDER BY run_id", (S1_TAG,)).fetchall()
    assert len(rows) == 39 and all(r[11] == 1 for r in rows)
    by_key = {(r[5], EVAL_OF_INV[r[1]], r[4]): r for r in rows}
    assert set(by_key) == S1_SCORED | S1_PROVENANCE
    for key, r in by_key.items():
        if key in S1_SCORED:
            assert (r[2], r[8], r[9]) == ("ingested", None, 1) and r[6] and r[7] == 4, key
        else:
            assert (r[2], r[8], r[9]) == ("generated", 1, None) and r[6] is None and r[7] is None, key
    assert {r[10] for r in rows if r[5] in S1_FITTED} == set(S1_OBJECTS.values())
    s1_ids = tuple(r[0] for r in rows)
    q = ",".join("?" * len(s1_ids))
    assert conn.execute(f"SELECT COUNT(*) FROM judgment WHERE run_id IN ({q})", s1_ids).fetchone()[0] == 16
    assert conn.execute(f"SELECT COUNT(*) FROM run_contrast WHERE run_id IN ({q})", s1_ids).fetchone()[0] == 42    # 14 x 3 (including the label mean)
    prov_ids = tuple(r[0] for r in rows if r[8] == 1)
    qp = ",".join("?" * len(prov_ids))
    for table in ("judgment", "item_result", "item_flags", "run_metric", "run_state_event", "run_contrast"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id IN ({qp})", prov_ids).fetchone()[0] == 0, table
    # the CURRENT degeneration rows (a scored store's group re-read on the second pass re-files its lower-cap rows beside the first read's)
    assert conn.execute(f"SELECT COUNT(*) FROM v_run_degeneration_current WHERE run_id IN ({qp}) AND measurement_state = 'measured'", prov_ids).fetchone()[0] == 23
    # the other two seeds landed beside: the fitted pair scored at 8,192 (judged here), `none` at 2,048, the prompt at 512
    for tag in (SEED_TAG, S3_TAG):
        other = conn.execute("SELECT lifecycle_state, COUNT(*) FROM run WHERE view_legacy = ? GROUP BY 1 ORDER BY 1", (tag,)).fetchall()
        assert [tuple(r) for r in other] == [("generated", 6), ("ingested", 6)], (tag, other)
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 39 + 12 + 12
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 16 + 6 + 6 and conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 42 + 12 + 12   # 3 per steered row
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0 and conn.execute("SELECT COUNT(*) FROM ruling").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM disposition").fetchone()[0] == 0
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = ?", (ad.name,)).fetchall()] == [("5",)]   # the anchor currency (adapter re-anchoring)
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT lane_source_legacy, owner, adapter FROM source_status").fetchall()] == [("orxs_seedrep", "OR", "orxs_seedrep")]
    conn.close()


EVAL_OF_INV = {v: k for k, v in EVAL_OF.items()}


# --------------------------------------------------------------------------------------------
# the non-seed target: the Arditi all-positions point c = 0.3 (OrxsSeedrepAdapter 3)
# --------------------------------------------------------------------------------------------

AR_EXITS = {NONE: {"or_bench_hard_1k": 2048}, ARDITI: {"or_bench_hard_1k": 2048}}
AR_LABELS = {NONE: {"or_bench_hard_1k": ["full_compliance", "full_compliance", "full_refusal", "partial_refusal"]},
             ARDITI: {"or_bench_hard_1k": ["full_refusal", "full_compliance", "full_refusal", "full_compliance"]}}


def write_ladder_open(ladders: str, target: str, cells: Sequence[str], attempted: Sequence[int] = (512,)) -> str:
    """A ladder record of a target still climbing: every pair attempted the caps so far, no gate decision yet, `left_ladder_at` undecided."""
    body = {c: {"or_bench_hard_1k": {"n_at_cap": {str(a): 3 for a in attempted}, "rungs_attempted": list(attempted), "decisions": {},
                                     "roots": {str(a): f"/workspace-vast/{F.TESTUSER}/exp/sdfbase_full22_native/sampled/controls_cap{a}/{target}" for a in attempted},
                                     "left_ladder_at": {"status": "undecided"}}} for c in cells}
    path = os.path.join(ladders, target, "ladder.json")
    _json(path, {"kind": "controls", "tag": CS.seed_tag_of(target), "trial": 0, "rungs": [512, 2048, 8192, 16384], "extend_kind": "controls",
                 "exit_rule": "synthetic", "written_at": "2026-09-21T19:40:00+00:00", "count_only_ceiling": 16384, "cells": body, "summary": {"n_pairs": len(cells)}})
    return path


def _build_arditi_mirror(tmp_path, state: str, n_unsafe: int = 2, merged_rows_for_point: bool = True) -> dict:
    """The s2 directory of :func:`_build_mirror` plus the Arditi point `qwen35_4b_ar_c0p3_t0` in one of three states: ``partial`` — the 512
    stores of `none` and `arditi_allpos` generated on OR-Bench only, unjudged, NO ladder record yet; ``partial_open`` — the same with a ladder
    record that attempted 512 and declared nothing; ``complete`` — 2,048 stores judged (both pairs exit there), 512 unjudged, the ladder
    declaring the exits, the result files carrying the point's rows under the rung's key (`ar_c0p3|<store id>|or_bench_hard_1k`) unless
    ``merged_rows_for_point`` is False."""
    assert state in ("partial", "partial_open", "complete")
    root = str(tmp_path / "orxs_seedrep")
    caps_parent = os.path.join(root, "controls_sampled")
    ladders = os.path.join(root, "ladders")
    stores: Dict[tuple, str] = {}
    for (cap, cell_id), sp in SPEC.items():
        lower = [c for c in CAPS if c < cap]
        stores[("s2", cap, cell_id)] = write_store(caps_parent, cap, cell_id, judged_controls=CTLS, labels=sp["labels"], at_cap=sp["at_cap"], job=sp["job"],
                                                   escalated_from=(lower[-1] if lower else None), object_sha16=(F.OBJ16 if cell_id == LORA else None), n_unsafe=n_unsafe)
    write_ladder(ladders, TARGET, EXITS)
    parts = [dict(stores={(cap, sid): d for (sd, cap, sid), d in stores.items() if sd == "s2"}, exits=EXITS, spec=SPEC, seed="s2", target=TARGET, n_unsafe=n_unsafe)]
    ar_spec: Dict[tuple, dict] = {}
    for cell_id in (NONE, ARDITI):
        sha16 = ARDITI_OBJECT[:16] if cell_id == ARDITI else None
        caps = (512,) if state != "complete" else (512, 2048)
        for cap in caps:
            sp = {"labels": AR_LABELS[cell_id], "at_cap": {"or_bench_hard_1k": (["orb0001-eeeeeeee", "orb0002-eeeeeeee"] if cap == 512 else [])}}
            ar_spec[(cap, cell_id)] = sp
            stores[("ar", cap, cell_id)] = write_store(caps_parent, cap, cell_id, judged_controls=(AR_CTL if cap == 2048 else ()), labels=sp["labels"], at_cap=sp["at_cap"],
                                                       job=AR_JOB, seed_base=AR_SEED_BASE, escalated_from=(512 if cap == 2048 else None), object_sha16=sha16,
                                                       target=AR_TARGET, n_unsafe=n_unsafe, controls=AR_CTL, fit_hf=F.HF_IT)
    ladder_ar = None
    if state == "partial_open":
        ladder_ar = write_ladder_open(ladders, AR_TARGET, (NONE, ARDITI))
    elif state == "complete":
        ladder_ar = write_ladder(ladders, AR_TARGET, AR_EXITS)
        if merged_rows_for_point:
            parts.append(dict(stores={(cap, sid): d for (sd, cap, sid), d in stores.items() if sd == "ar"}, exits=AR_EXITS, spec=ar_spec, seed="ar_c0p3", target=AR_TARGET, n_unsafe=n_unsafe))
    maps = write_result_maps_seeds(root, parts)
    roots = CS.orxs_seedrep_roots(parents=(caps_parent,), declared=(512, 2048, 8192), ladders=ladders)
    return {"root": root, "caps_parent": caps_parent, "ladders": ladders, "ladder": ladder_ar, "stores": stores, "roots": roots, "n_unsafe": n_unsafe,
            "merged": maps["merged"], "by_rung": maps["by_rung"], "state": state}


def _assert_arditi_point_row(p: Dict, cap: int) -> None:
    """What every `arditi_allpos` run of the point files, whatever its cap: the Arditi arm's object, the rung's strength columns spelled as the
    `arditi_allpos` source's own rows spell them, the store's decode seed and trial, the directory's spelling, the cluster."""
    assert p["object_id"] == ARDITI_OBJECT and p["view_legacy"] == AR_TAG and p["extra"]["view_legacy_dir"] == AR_TARGET and p["token_cap_tokens"] == cap
    assert p["strength_nominal"] == 1.0 and p["is_operating_point"] == 1 and p["operating_point_id_legacy"] == "ar_c0p3__arditi_allpos"
    assert p["strength_unit"] == "raw_coefficient" and p["strength_coefficient"] == pytest.approx(0.3) and p["strength_nominal_source"] == "derived"
    assert p["extra"]["paper_dose"] == pytest.approx(0.3) and p["extra"]["c"] == pytest.approx(0.3) and p["extra"]["staged_id"] == "ar_c0p3__arditi_allpos"
    assert p["extra"]["object_role"] == "baseline_published" and p["transfer_kind"] == "instruct_direct"
    assert p["seed_base"] == AR_SEED_BASE and p["trial_index"] == 0 and p["decode_seed_rule"] == "per_item" and p["cluster_job_id"] == AR_JOB
    assert p["substrate"] == "cluster_a" and p["substrate_source"] == "derived" and p["lane_source_legacy"] == "orxs_seedrep"
    assert p["eval_id"] == "or_bench_hard" and p["producer_run_key_legacy"].startswith(f"{AR_TARGET}/") and p["field_confidence_json"]["object_id"]["confidence"] == "recorded"


@pytest.mark.parametrize("state", ["partial", "partial_open"])
def test_arditi_point_still_climbing_files_generated_rows_with_no_final_cap_and_refuses_nothing(tmp_path, state):
    m = _build_arditi_mirror(tmp_path, state)
    ad = _adapter(m)
    inputs = ad.discover()
    for cell_id in (NONE, ARDITI):
        store = m["stores"][("ar", 512, cell_id)]
        assert not CS._judged(store) and store in {os.path.dirname(i.uri) for i in inputs if "#" not in i.uri}
        recs = _read_group(ad, store)
        assert {r["record_type"] for r in recs} == {"run", "run_degeneration"} and not _validate_all(recs)
        runs = _payloads(recs, "run")
        assert len(runs) == 1 and runs[0]["extra"]["control"] == "or_bench_hard_1k"           # OR-Bench only: the manifest names one control
        p = runs[0]
        assert p["lifecycle_state"] == "generated" and p["judge_id"] is None and p["n_items_judged"] is None and p["n_items_generated"] == 4
        assert p["extra"]["final_cap_undeclared"] is True and "completion_cap_merged" not in p["extra"] and "provenance_only_lower_rung" not in p["extra"]
        assert p["extra"]["merged_map"]["status"] == "absent" and p["extra"]["merged_map"]["key"] == f"ar_c0p3|{cell_id}|or_bench_hard_1k"
        assert p["field_confidence_json"]["completion_cap_merged"]["confidence"] == "unverified"
        assert p["extra"]["escalation_state"] == ("escalation_pending" if state == "partial_open" else None)
        if state == "partial":
            assert "no ladder record" in p["extra"]["ladder_exit"]["note"]
        assert p["view_legacy"] == AR_TAG and p["seed_base"] == AR_SEED_BASE and p["substrate"] == "cluster_a"
        if cell_id == ARDITI:
            _assert_arditi_point_row(p, 512)
        else:
            assert p["object_id"] == "none" and p["transfer_kind"] == "none" and "strength_nominal" not in p and "operating_point_id_legacy" not in p
        assert _payloads(recs, "run_contrast") == [] and not [r for r in recs if r["record_type"] in ("disposition", "ruling", "judgment", "run_state")]
    # the seed directory beside it reads as before
    assert len(_payloads(_read_group(ad, m["stores"][("s2", 2048, NONE)]), "judgment")) == 2


def test_arditi_point_complete_scores_the_final_cap_and_contrasts_against_its_own_none(tmp_path):
    m = _build_arditi_mirror(tmp_path, "complete")
    ad = _adapter(m)
    recs = {k: _read_group(ad, d) for k, d in m["stores"].items() if k[0] == "ar"}
    runs = {k: _payloads(r, "run")[0] for k, r in recs.items()}
    for cell_id in (NONE, ARDITI):
        top, low = runs[("ar", 2048, cell_id)], runs[("ar", 512, cell_id)]
        assert top["extra"]["completion_cap_merged"] is True and top["extra"]["final_rung"] == 2048 and top["judge_id"] and top["n_items_judged"] == 4
        assert top["extra"]["escalation_state"] == "final_at_rung" and top["extra"]["merged_map"]["key"] == f"ar_c0p3|{cell_id}|or_bench_hard_1k"
        assert "status" not in top["extra"]["merged_map"] and top["field_confidence_json"]["completion_cap_merged"]["confidence"] == "recorded"
        assert top["extra"]["by_rung_map"]["key"] == f"ar_c0p3|{cell_id}|or_bench_hard_1k|2048" and top["extra"]["n_at_cap_by_rung"] == {"512": 2, "2048": 0}
        assert low["extra"]["provenance_only_lower_rung"] is True and low["judge_id"] is None and low["extra"]["superseded_by_final_rung_run_id"] == top["run_id"]
        assert low["extra"]["judged_in_store_withheld"]["n_judged_in_store"] is None and low["extra"]["escalation_state"] == "superseded_by_rung"
        assert {r["record_type"] for r in recs[("ar", 512, cell_id)]} == {"run", "run_degeneration"}
        for r in (recs[("ar", 512, cell_id)], recs[("ar", 2048, cell_id)]):
            assert not _validate_all(r) and not [x for x in r if x["record_type"] == "disposition"]
    _assert_arditi_point_row(runs[("ar", 2048, ARDITI)], 2048)
    _assert_arditi_point_row(runs[("ar", 512, ARDITI)], 512)
    # the contrast: arditi_allpos at its final cap against THIS target directory's none at its final cap, the same standup
    contrasts = _payloads(recs[("ar", 2048, ARDITI)], "run_contrast")
    assert len(contrasts) == 3 and {c["metric_id"] for c in contrasts} == {"orb_refused", CS.OVER_REFUSAL_GRADED_METRIC, "over_refusal_label_mean"}   # + the label mean
    for c in contrasts:
        assert c["anchor_run_id"] == runs[("ar", 2048, NONE)]["run_id"] and c["anchor_kind"] == "same_standup_none" and c["extra"]["contrast_caps"] == {"steered": 2048, "anchor": 2048}
        assert c["extra"]["anchor_store"] == m["stores"][("ar", 2048, NONE)] and c["extra"]["standup_job_ids"] == [AR_JOB, AR_JOB]
    orb = next(c for c in contrasts if c["metric_id"] == "orb_refused")
    assert orb["delta"] == pytest.approx(100.0 * (0.5 - 0.25)) and orb["n_pairs"] == 4
    assert _payloads(recs[("ar", 512, ARDITI)], "run_contrast") == [] and _payloads(recs[("ar", 2048, NONE)], "run_contrast") == []
    # the s2 directory's anchor is its own none, never the point's
    s2 = _payloads(_read_group(ad, m["stores"][("s2", 8192, LORA)]), "run_contrast")
    assert s2 and {c["anchor_run_id"] for c in s2} == {p["run_id"] for p in _payloads(_read_group(ad, m["stores"][("s2", 2048, NONE)]), "run")}


def test_arditi_point_without_rows_in_ors_result_files_scores_on_the_ladder_alone_and_says_so(tmp_path):
    m = _build_arditi_mirror(tmp_path, "complete", merged_rows_for_point=False)
    ad = _adapter(m)
    assert not any(k.startswith("ar_c0p3|") or k.startswith(f"{AR_TAG}|") for k in json.load(open(m["merged"]))["rows"])
    top = _payloads(_read_group(ad, m["stores"][("ar", 2048, ARDITI)]), "run")[0]
    assert top["extra"]["completion_cap_merged"] is True and top["extra"]["final_rung"] == 2048 and top["judge_id"]
    assert top["extra"]["merged_map"]["status"] == "absent" and "ar_c0p3|arditi_allpos|or_bench_hard_1k" in top["extra"]["merged_map"]["note"]
    assert top["extra"]["n_at_cap_by_rung"] is None and top["extra"]["residual_capped_at_final"] == 0 and top["extra"]["by_rung_map"] is None
    assert top["field_confidence_json"]["completion_cap_merged"]["confidence"] == "derived"
    low = _payloads(_read_group(ad, m["stores"][("ar", 512, ARDITI)]), "run")[0]
    assert low["extra"]["provenance_only_lower_rung"] is True and low["extra"]["merged_map"]["status"] == "absent" and low["extra"]["n_at_cap_by_rung"] is None
    contrasts = _payloads(_read_group(ad, m["stores"][("ar", 2048, ARDITI)]), "run_contrast")
    assert len(contrasts) == 3 and all(c["anchor_kind"] == "same_standup_none" for c in contrasts)     # refused rate, graded mean, label mean
    # a seed target's absent row still refuses
    merged = json.load(open(m["merged"]))
    merged["rows"].pop(f"s2|{NONE}|xstest")
    _json(m["merged"], merged)
    with pytest.raises(AB.AdapterRefusal, match="no row in"):
        _read_group(ad, m["stores"][("s2", 2048, NONE)])


def test_arditi_point_ids_are_stable_from_climbing_to_complete_and_torn_states_refuse(tmp_path):
    m = _build_arditi_mirror(tmp_path, "partial")
    ad = _adapter(m)
    ids0 = {c: _payloads(_read_group(ad, m["stores"][("ar", 512, c)]), "run")[0]["run_id"] for c in (NONE, ARDITI)}
    # a store judged while its pair is still open is torn: refused by name, never scored
    judge_store(m["stores"][("ar", 512, ARDITI)], ARDITI, os.path.basename(m["stores"][("ar", 512, ARDITI)]), list(AR_CTL), AR_LABELS[ARDITI], {}, n_unsafe=m["n_unsafe"], target=AR_TARGET)
    with pytest.raises(AB.AdapterRefusal, match="declares no exit for the pair"):
        _read_group(ad, m["stores"][("ar", 512, ARDITI)])
    for fn in ("rows.jsonl", "judge_items.jsonl", "judge_summary.json"):
        os.remove(os.path.join(m["stores"][("ar", 512, ARDITI)], fn))
    # the ladder closes at 2,048 and the 2,048 stores land judged: the 512 rows keep their ids and become provenance only
    m2 = _build_arditi_mirror(tmp_path / "later", "complete")
    for c in (NONE, ARDITI):
        assert os.path.basename(m2["stores"][("ar", 512, c)]) == os.path.basename(m["stores"][("ar", 512, c)])
    ad2 = _adapter(m2)
    ids1 = {c: _payloads(_read_group(ad2, m2["stores"][("ar", 512, c)]), "run")[0]["run_id"] for c in (NONE, ARDITI)}
    assert ids1 == ids0
    # a store id the point does not serve, and a target directory of neither grammar, refuse by name
    stray = write_store(m2["caps_parent"], 512, LORA, target=AR_TARGET, controls=AR_CTL, seed_base=AR_SEED_BASE, object_sha16=F.OBJ16)
    with pytest.raises(AB.AdapterRefusal, match="serves `none` and `arditi_allpos` only"):
        _read_group(ad2, stray)
    shutil.rmtree(stray)
    neither = write_store(m2["caps_parent"], 512, NONE, target="qwen35_4b_foo_t0", object_sha16=None)
    with pytest.raises(AB.AdapterRefusal, match="spells neither a seed selection"):
        _read_group(ad2, neither)


class _FakeConn:
    """A connection returning fixed object rows to the target-grammar resolver's one query."""
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params):
        assert "FROM object" in sql
        return self

    def fetchall(self):
        return self.rows


def test_the_real_target_grammar_resolver_names_the_arditi_lanes_object_of_the_rung_and_refuses_by_name():
    """`orxs_target_object` over the real seed roster for a seed tag (delegation), and over the object table for the Arditi point: exactly one
    row must prefix the manifest sha16, of kind vector_all, staged as the rung the directory spells; `none` needs no table; anything else refuses."""
    sha = json.load(open(__import__("sisteer.registry.adapters.declared_common", fromlist=["repo_path"]).repo_path("configs", "sdfbase_full22_pins.json")))["qwen35_4b"]["base64s2__constv2-mix-full__lora_r4_ce"]["object_sha256"]
    assert CS.orxs_target_object("constv2-mix-full__lora_r4_ce", "qwen35_4b_s2_t0", sha[:16], None) == CS.seed_selection_object("constv2-mix-full__lora_r4_ce", "qwen35_4b_s2_t0", sha[:16], None)
    assert CS.orxs_target_object("none", AR_TARGET, None, None)[0] == "none" and "c = 0.3" in CS.orxs_target_object("none", AR_TARGET, None, None)[1]
    with pytest.raises(AB.AdapterRefusal, match="spells neither a seed selection"):
        CS.orxs_target_object("none", "qwen35_4b_foo_t0", None, None)
    with pytest.raises(AB.AdapterRefusal, match="serves `none` and `arditi_allpos` only"):
        CS.orxs_target_object("constv2-mix-full__lora_r4_ce", AR_TARGET, "0123456789abcdef", None)
    with pytest.raises(AB.AdapterRefusal, match="names no object_sha"):
        CS.orxs_target_object(ARDITI, AR_TARGET, None, None)
    with pytest.raises(AB.AdapterRefusal, match="no connected registry"):
        CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", None)
    good = ("ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2", "vector_all", "ar_c0p3__arditi_allpos")
    oid, src = CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", _FakeConn([good]))
    assert oid == good[0] and "ar_c0p3__arditi_allpos" in src and "vector_all" in src
    with pytest.raises(IC.Unresolved, match="matches 0 object rows"):
        CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", _FakeConn([]))
    with pytest.raises(IC.Unresolved, match="matches 2 object rows"):
        CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", _FakeConn([good, good]))
    with pytest.raises(AB.AdapterRefusal, match="not the all-positions"):
        CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", _FakeConn([(good[0], "lora", good[2])]))
    with pytest.raises(AB.AdapterRefusal, match="the served object and the tag disagree"):
        CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", _FakeConn([(good[0], "vector_all", "ar_c0p5__arditi_allpos")]))
    with pytest.raises(AB.AdapterRefusal, match="the served object and the tag disagree"):
        CS.orxs_target_object(ARDITI, "qwen35_4b_ar_c0p5_t0", "ba2212f0c2f9ffef", _FakeConn([good]))
    conn = DB.connect_memory()
    from sisteer.registry.migrations import apply_migrations
    apply_migrations(conn)
    with pytest.raises(IC.Unresolved, match="matches 0 object rows"):
        CS.orxs_target_object(ARDITI, AR_TARGET, "ba2212f0c2f9ffef", conn)


def _arditi_object_record(root: str) -> None:
    """The synthetic all-positions object row (the shape the `arditi_allpos` reader mints), so the point's runs have their parent at ingest."""
    from sisteer.registry import records as R
    from sisteer.registry import synth
    payload = {"object_id": ARDITI_OBJECT, "object_kind": "vector_all", "paper_label": "arditi_allpos", "campaign": synth.CAMPAIGN,
               "campaign_cell_id_legacy": "ar_c0p3__arditi_allpos", "object_role": "baseline_published", "fit_target_kind": "contrastive_direction",
               "fit_model_id": F.MODEL_IT, "fit_contrast_sets_json": {"unverified": "synthetic"}, "fit_thinking_render": "unverified",
               "fit_objective": "arditi_search", "fit_positions": "all", "fit_layers_json": [10], "fit_layer_index_convention": "zero_based_decoder_block",
               "fit_layer_count": 1, "fit_layer_selection_rule": "separability_search", "fit_seed_confidence": "unverified", "derivation": "none",
               "base_scale": 0.3, "artifact_sha256": ARDITI_OBJECT, "status": "approved",
               "field_confidence_json": {k: {"confidence": "unverified", "source": "synthetic", "reason": "no fit record beside the synthetic point"}
                                         for k in ("fit_contrast_sets_json", "fit_seed", "fit_seed_confidence", "fit_thinking_render")}}
    R.write_records(os.path.join(root, "records", "dims", "000003-arditi-object.jsonl"), [synth.rec("object", payload, synth.t(220))])


def test_arditi_point_pass_and_ingest_land_the_scored_pair_and_keep_the_base_cap_generated(tmp_path):
    m = _build_arditi_mirror(tmp_path, "complete", n_unsafe=0)
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    _arditi_object_record(root)
    ad = _adapter(m)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_held == 0 and rep.groups_refreshed == 6 + 4, [r.error for r in rep.results if r.error]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    # the first pass parks every merged comparison (the own `none` is not folded yet: no provisional anchors); the pass after the
    # fold re-reads each steered scored group on its moved anchor-decision input and files its contrasts against the now-current `none`
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    rep2 = RUN.run_pass(root, [ad], conn=conn, now="2026-09-21T10:30:00.000Z")
    conn.close()
    assert rep2.groups_refused == 0 and rep2.groups_held == 0 and rep2.anchor_decisions_changed == rep2.groups_refreshed > 0, [r.error for r in rep2.results if r.error]
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0, ing2.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = [tuple(r) for r in conn.execute("SELECT campaign_cell_id_legacy, token_cap_tokens, lifecycle_state, object_id, strength_nominal, is_operating_point, operating_point_id_legacy, "
                                           "strength_unit, seed_base, trial_index, json_extract(extra_json, '$.provenance_only_lower_rung'), json_extract(extra_json, '$.completion_cap_merged') "
                                           "FROM run WHERE view_legacy = ? ORDER BY 2, 1", (AR_TAG,)).fetchall()]
    assert rows == [(ARDITI, 512, "generated", ARDITI_OBJECT, 1.0, 1, "ar_c0p3__arditi_allpos", "raw_coefficient", AR_SEED_BASE, 0, 1, None),
                    (NONE, 512, "generated", "none", None, None, None, "unverified", AR_SEED_BASE, 0, 1, None),     # an unsteered run: no dose, the unit column's default
                    (ARDITI, 2048, "ingested", ARDITI_OBJECT, 1.0, 1, "ar_c0p3__arditi_allpos", "raw_coefficient", AR_SEED_BASE, 0, None, 1),
                    (NONE, 2048, "ingested", "none", None, None, None, "unverified", AR_SEED_BASE, 0, None, 1)]
    assert conn.execute("SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.view_legacy = ?", (AR_TAG,)).fetchone()[0] == 3   # 3 metrics
    assert conn.execute("SELECT COUNT(*) FROM judgment j JOIN run r ON r.run_id = j.run_id WHERE r.view_legacy = ?", (AR_TAG,)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0 and conn.execute("SELECT COUNT(*) FROM disposition").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run WHERE view_legacy = ?", (SEED_TAG,)).fetchone()[0] == 12
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = ?", (ad.name,)).fetchall()] == [("5",)]   # the anchor currency (adapter re-anchoring)
    conn.close()


# --------------------------------------------------------------------------------------------
# the Arditi arm's own OR-Bench stores at c = 0.3 (cellstore DEFAULT_ROOTS, source arditi_allpos)
# --------------------------------------------------------------------------------------------

ORB_JOB = "546174"
ORB_LABELS = {NONE: ["full_compliance", "full_refusal", "partial_refusal", "full_refusal"], ARDITI: ["full_refusal", "full_refusal", "full_refusal", "partial_refusal"]}


def write_orb_digest(path: str, stores_2048: Dict[str, str], labels: Dict[str, Sequence[str]], rung: int = 2048, n_orb: int = 4, schema: str = "arditi_orbench_graded/3") -> str:
    """The lane's `orbench_graded.json` as the stores realise it: `rung`, `control`, and per arm the final-cap store key and the graded block
    (full-refusal rate over judged, n_judged, n_abstained, label counts) computed from the labels the judge wrote."""
    arms = {}
    for arm, store in stores_2048.items():
        labs = list(labels[arm])
        counts = {lab: sum(1 for k in range(n_orb) if labs[k % len(labs)] == lab) for lab in DEFAULT_LABELS}
        arms[arm] = {"cell": os.path.basename(store), "store": store, "cap": rung, "n_items": n_orb, "cell_id": arm, "standup": {"slurm_job_id": ORB_JOB},
                     "graded": {"full_refusal_rate_judged": counts["full_refusal"] / n_orb, "n_judged": n_orb, "n_abstained": 0, "n_items": n_orb,
                                "label_counts_judged": {k: v for k, v in counts.items() if v}}}
    _json(path, {"schema": schema, "tag": AR_TAG, "control": "or_bench_hard_1k", "task": "judged/or_bench_hard_1k", "cluster": "cluster_a", "rung": rung,
                 "rung_record": {"cap": rung, "rungs_present": {"512": sorted(arms), str(rung): sorted(arms)}}, "n_items": n_orb, "arms": arms, "warnings": []})
    return path


def _build_orb_mirror(tmp_path, n_unsafe: int = 0) -> dict:
    """`<tmp>/orb_private_allpos/qwen35_4b_ar_c0p3_t0/orb_cap{512,2048}/cells/<key>/`: `none` and `arditi_allpos` on OR-Bench only, seed base
    20260909, one standup (job 546174); the 2,048 stores judged, the 512 stores generated only; the digest beside under `orb_grades/`."""
    parent = str(tmp_path / "orb_private_allpos" / AR_TARGET)
    stores: Dict[tuple, str] = {}
    for cell_id in (NONE, ARDITI):
        for cap in (512, 2048):
            stores[(cap, cell_id)] = write_store("", cap, cell_id, judged_controls=(AR_CTL if cap == 2048 else ()), labels={"or_bench_hard_1k": ORB_LABELS[cell_id]},
                                                 at_cap={"or_bench_hard_1k": (["orb0000-eeeeeeee"] if cap == 512 else [])}, job=ORB_JOB, seed_base=AR_SEED_BASE,
                                                 escalated_from=(512 if cap == 2048 else None), object_sha16=(ARDITI_OBJECT[:16] if cell_id == ARDITI else None),
                                                 target=AR_TARGET, n_unsafe=n_unsafe, controls=AR_CTL, fit_hf=F.HF_IT, cells_parent=os.path.join(parent, f"orb_cap{cap}"))
    digest = write_orb_digest(str(tmp_path / "orb_grades" / AR_TARGET / "orbench_graded.json"), {c: stores[(2048, c)] for c in (NONE, ARDITI)},
                              {c: ORB_LABELS[c] for c in (NONE, ARDITI)})
    roots = CS.arditi_allpos_orb_roots(parent=parent, digest=digest)
    return {"parent": parent, "digest": digest, "stores": stores, "roots": roots}


def _orb_dims() -> IC.Dims:
    d = F.dims_map()
    d.objects[ARDITI_OBJECT[:16]] = ARDITI_OBJECT
    return d


def _read_generic(ad: CS.CellStoreAdapter, store: str, conn=None):
    inputs = ad.discover()
    mine = [i for i in inputs if ad.comparison_group(i) == store]
    batches = list(ad.read(_ctx(conn if conn is not None else _primed_conn(ad)), store, mine))
    assert len(batches) == 1
    return batches[0].records


def test_arditi_orb_roots_are_declared_on_the_generic_adapter_with_the_sources_own_spellings():
    roots = CS.ARDITI_ALLPOS_ORB_ROOTS
    assert [r.path for r in roots] == [os.path.join(CS.ARDITI_ALLPOS_ORB_PARENT, f"orb_cap{c}") for c in (512, 2048)]
    assert CS.ARDITI_ALLPOS_ORB_PARENT == os.path.join(str(SITE.expand_path("$GRADE/arditi_allpos/orb_private_allpos")), "qwen35_4b_ar_c0p3_t0")
    assert CS.ARDITI_ALLPOS_ORB_DIGEST == os.path.join(str(SITE.expand_path("$GRADE/arditi_allpos/orb_grades")), "qwen35_4b_ar_c0p3_t0", "orbench_graded.json")
    for r in roots:
        assert r.campaign == "arditi_allpos" and r.layout == "cells" and r.run_key_scope == "store_key" and r.source_legacy == "arditi_allpos"
        assert r.at_cap_from_gen_stats and r.over_refusal_headline and r.cap_provenance and r.file_unjudged and not r.ladder_parent
        assert r.served_tag == "qwen35_4b_ar_c0p3_t0" and r.view_spelling_legacy == "qwen35_4b_it" and r.final_cap_digest == CS.ARDITI_ALLPOS_ORB_DIGEST
        assert r.object_resolution == "sha_prefix" and not r.strip_trial_suffix and not r.substrate_from_routing_table
    assert set(roots) <= set(CS.DEFAULT_ROOTS) and CS.CellStoreAdapter.EMISSION_VERSION == 10      # 10: above both parents -- 9 (27b) + 8: the anchor currency; 7: transfer_kind from the object row when the manifest names no fit model; 6: the label-mean row
    assert not ({r.path for r in roots} & {r.path for r in CS.ORXS_SEEDREP_ROOTS}) and not ({r.path for r in roots} & {r.path for r in CS.ARDITI_VENDOR_CONTROLS_ROOTS})
    # the other default roots declare none of the three new opt-ins: their bytes do not move
    assert all(r.served_tag is None and r.view_spelling_legacy is None and r.final_cap_digest is None for r in CS.DEFAULT_ROOTS if r not in roots)


def test_arditi_orb_stores_file_the_point_under_the_sources_spellings_with_the_digest_as_the_final_cap_record(tmp_path):
    m = _build_orb_mirror(tmp_path)
    ad = CS.CellStoreAdapter(roots=m["roots"], dims=_orb_dims())
    recs = {k: _read_generic(ad, d) for k, d in m["stores"].items()}
    runs = {k: _payloads(r, "run")[0] for k, r in recs.items()}
    assert all(len(_payloads(r, "run")) == 1 for r in recs.values())
    for (cap, cell_id), p in runs.items():
        assert p["campaign"] == "arditi_allpos" and p["lane_source_legacy"] == "arditi_allpos" and p["view_legacy"] == "qwen35_4b_it"
        assert p["extra"]["view_legacy_dir"] == AR_TARGET and p["producer_run_key_legacy"] == os.path.basename(m["stores"][(cap, cell_id)])
        assert p["run_id"] == I.run_id("arditi_allpos", p["producer_run_key_legacy"], p["producer_object_key_legacy"], "or_bench_hard", p["instrument_id"])
        assert p["eval_id"] == "or_bench_hard" and p["token_cap_tokens"] == cap and p["token_cap_rung_index"] == {512: 0, 2048: 1}[cap]
        assert p["seed_base"] == AR_SEED_BASE and p["trial_index"] == 0 and p["cluster_job_id"] == ORB_JOB and p["substrate"] == "cluster_a"
        assert p["extra"]["final_cap_digest"]["final_cap"] == 2048 and p["extra"]["final_cap_digest"]["sha256"] == AB.check_input(m["digest"]).sha256
        if cell_id == ARDITI:
            assert p["object_id"] == ARDITI_OBJECT and p["transfer_kind"] == "instruct_direct" and p["extra"]["object_role"] == "baseline_published"
            assert p["strength_nominal"] == 1.0 and p["is_operating_point"] == 1 and p["operating_point_id_legacy"] == "ar_c0p3__arditi_allpos"
            assert p["strength_unit"] == "raw_coefficient" and p["strength_coefficient"] == pytest.approx(0.3) and p["extra"]["paper_dose"] == pytest.approx(0.3)
        else:
            assert p["object_id"] == "none" and "strength_nominal" not in p
        if cap == 2048:
            assert p["judge_id"] and p["n_items_judged"] == 4 and p["extra"]["escalation_state"] == "final_at_rung" and p["extra"]["superseded_by_run_id"] is None
            assert p["field_confidence_json"]["final_cap_digest"]["confidence"] == "recorded" and "reproduce" in p["field_confidence_json"]["final_cap_digest"]["source"]
        else:
            assert p["judge_id"] is None and p["lifecycle_state"] == "generated" and p["extra"]["escalation_state"] == "superseded_by_rung"
            assert p["extra"]["superseded_by_run_id"] == runs[(2048, cell_id)]["run_id"] and p["extra"]["final_cap_digest"]["superseded_by_store"] == m["stores"][(2048, cell_id)]
            assert {r["record_type"] for r in recs[(cap, cell_id)]} == {"run", "run_degeneration"}
    for r in recs.values():
        assert not _validate_all(r)
    # the headline reproduces the digest's full-refusal rate; the contrast pairs the point against its own none, the same standup
    headline = next(q for q in _payloads(recs[(2048, ARDITI)], "run_metric") if q["metric_id"] == "orb_refused" and q["computed_by"] == "registry")
    assert headline["value"] == pytest.approx(0.75) and headline["n"] == 4
    contrasts = _payloads(recs[(2048, ARDITI)], "run_contrast")
    assert len(contrasts) == 3 and all(c["anchor_run_id"] == runs[(2048, NONE)]["run_id"] and c["anchor_kind"] == "same_standup_none" for c in contrasts)   # 3 metrics
    assert next(c for c in contrasts if c["metric_id"] == "orb_refused")["delta"] == pytest.approx(100.0 * (0.75 - 0.5))
    assert _payloads(recs[(512, ARDITI)], "run_contrast") == [] and _payloads(recs[(2048, NONE)], "run_contrast") == []


def test_arditi_orb_digest_disagreements_refuse_the_store_by_name(tmp_path):
    m = _build_orb_mirror(tmp_path)
    ad = CS.CellStoreAdapter(roots=m["roots"], dims=_orb_dims())
    original = json.load(open(m["digest"]))

    def tamper(fn):
        d = json.loads(json.dumps(original))
        fn(d)
        _json(m["digest"], d)

    tamper(lambda d: d["arms"][ARDITI]["graded"].__setitem__("full_refusal_rate_judged", 0.5))
    with pytest.raises(AB.AdapterRefusal, match="not what the store reproduces"):
        _read_generic(ad, m["stores"][(2048, ARDITI)])
    tamper(lambda d: d["arms"][ARDITI]["graded"].__setitem__("n_abstained", 1))
    with pytest.raises(AB.AdapterRefusal, match="not what the store reproduces"):
        _read_generic(ad, m["stores"][(2048, ARDITI)])
    tamper(lambda d: d["arms"].pop(ARDITI))
    with pytest.raises(AB.AdapterRefusal, match="names no arm"):
        _read_generic(ad, m["stores"][(2048, ARDITI)])
    tamper(lambda d: d.__setitem__("schema", "something_else/1"))
    with pytest.raises(AB.AdapterRefusal, match="schema"):
        _read_generic(ad, m["stores"][(512, NONE)])
    tamper(lambda d: d.__setitem__("rung", 512))
    with pytest.raises(AB.AdapterRefusal, match="above the digest's final cap"):
        _read_generic(ad, m["stores"][(2048, NONE)])
    tamper(lambda d: d["arms"][NONE].__setitem__("cell", "none__controls__t0__a0__2ff9cdff__deadbeef"))
    with pytest.raises(AB.AdapterRefusal, match="names the store"):
        _read_generic(ad, m["stores"][(2048, NONE)])
    _json(m["digest"], original)
    # a judged store below the final cap is never filed
    judge_store(m["stores"][(512, ARDITI)], ARDITI, os.path.basename(m["stores"][(512, ARDITI)]), list(AR_CTL), {"or_bench_hard_1k": ORB_LABELS[ARDITI]}, {}, n_unsafe=0, target=AR_TARGET)
    with pytest.raises(AB.AdapterRefusal, match="below the final cap"):
        _read_generic(ad, m["stores"][(512, ARDITI)])
    assert len(_payloads(_read_generic(ad, m["stores"][(2048, ARDITI)]), "judgment")) == 1


def test_arditi_orb_pass_and_ingest_land_the_final_cap_and_keep_the_base_cap_generated(tmp_path):
    m = _build_orb_mirror(tmp_path)
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    _arditi_object_record(root)
    ad = CS.CellStoreAdapter(roots=m["roots"], dims=_orb_dims())
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_held == 0 and rep.groups_refreshed == 4, [r.error for r in rep.results if r.error]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    # the first pass parks every merged comparison (the own `none` is not folded yet: no provisional anchors); the pass after the
    # fold re-reads each steered scored group on its moved anchor-decision input and files its contrasts against the now-current `none`
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    rep2 = RUN.run_pass(root, [ad], conn=conn, now="2026-09-21T10:30:00.000Z")
    conn.close()
    assert rep2.groups_refused == 0 and rep2.groups_held == 0 and rep2.anchor_decisions_changed == rep2.groups_refreshed > 0, [r.error for r in rep2.results if r.error]
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0, ing2.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = [tuple(r) for r in conn.execute("SELECT campaign_cell_id_legacy, token_cap_tokens, lifecycle_state, object_id, view_legacy, campaign, lane_source_legacy, "
                                           "strength_nominal, operating_point_id_legacy, json_extract(extra_json, '$.escalation_state') FROM run ORDER BY 2, 1").fetchall()]
    assert rows == [(ARDITI, 512, "generated", ARDITI_OBJECT, "qwen35_4b_it", "arditi_allpos", "arditi_allpos", 1.0, "ar_c0p3__arditi_allpos", "superseded_by_rung"),
                    (NONE, 512, "generated", "none", "qwen35_4b_it", "arditi_allpos", "arditi_allpos", None, None, "superseded_by_rung"),
                    (ARDITI, 2048, "ingested", ARDITI_OBJECT, "qwen35_4b_it", "arditi_allpos", "arditi_allpos", 1.0, "ar_c0p3__arditi_allpos", "final_at_rung"),
                    (NONE, 2048, "ingested", "none", "qwen35_4b_it", "arditi_allpos", "arditi_allpos", None, None, "final_at_rung")]
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 2 and conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 3   # 3 metrics
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = 'cellstore'").fetchall()] == [("10",)]   # 10: above both parents -- 9 (27b) + 8: the anchor currency; 7: transfer_kind from the object row (2026-09-30); 6: the label-mean addendum
    conn.close()


# --------------------------------------------------------------------------------------------
# the role-collapse companion group (`orxs_seedrep:allmodels_paired/role_collapse_v1`; 2026-09-22)
# --------------------------------------------------------------------------------------------
#
# The structural third state over ReLU's paired stores and the sampled controls' unsteered anchors files ONE `run_metric` row per current
# run its keys name -- `over_refusal_graded_noncollapsed` beside the owner's `over_refusal_graded` on the same judgment / threshold /
# computation, k = the full refusals among judged non-collapsed items, n = the judged non-collapsed items, the trials pooled, every count in
# `extra_json` under exactly the keys the downstream consumer reads (the agreed spelling) -- mapped by identity in the connected registry (the
# model, the control's eval, the store label, the store's target directory + key among the run's stores, the trial's final cap, a CURRENT run
# with a current owner row), refused by name otherwise. Synthetic world: stand-in pooled runs shaped like relu_paired's controls row (items
# on the run, per trial) and sampled_controls' pooled anchor (items on member runs), invented counts, `/home/TESTUSER/...` store paths.

from sisteer.registry import records as R
from sisteer.registry import synth as SY
from sisteer.registry.adapters import orxs_role_collapse as ORC
from sisteer.registry.migrations import apply_migrations

RC_MODEL = F.MODEL_IT
RC_CAMPAIGN = "synth_rc_campaign"
RC_PAIRED_TAG = "synthmodel_paired_d1_synthmodel_s1"        # the target directory without the `_t<T>` suffix
RC_ANCHOR_TAG = "synthmodel"
RC_STEERED = F.LORA_CELL
RC_UNKNOWN = "constv2-mix-full__lora_r4_ce_minus_ctl"
RC_RS = "/home/TESTUSER/rs/full22"
RC_JUDGE_SEED = {"xstest": "xstest_cpi_v1_sonnet5", "or_bench_hard_1k": "or_bench_cpi_v1_sonnet5"}
RC_COMP = "aaaa1111bbbb2222"
RC_COMPONENTS_ZERO = {k: 0.0 for k in ORC.COMPONENT_KEYS}


def _rc_sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _rc_store_key(store_id: str, trial: int) -> str:
    return f"{store_id.replace('__', '--', 1)}__controls__t0__a0__{IHASH.get(store_id, '0badf00d')}__{hashlib.sha256(f'{store_id}|{trial}'.encode()).hexdigest()[:8]}"


def _rc_or_store(tag: str, trial: int, cap: int, store_id: str) -> str:
    """The generator's spelling of a store: the GENERATION root (`controls_cap<C>/`), not the pass-2 copy the registry's runs name."""
    return f"{RC_RS}/relu_paired/sampled/controls_sampled/controls_cap{cap}/{tag}_t{trial}/cells/{_rc_store_key(store_id, trial)}/"


def _rc_registry_store(tag: str, trial: int, cap: int, store_id: str) -> str:
    return f"{RC_RS}/relu_paired/sampled/controls_sampled/flip_pass/cap{cap}/{tag}_t{trial}/cells/{_rc_store_key(store_id, trial)}"


def _rc_labels(n_judged: int, k_full: int, n_items: int) -> List[Optional[str]]:
    """`n_items` labels: `k_full` full refusals, the rest of the judged ones compliance, the unjudged tail None."""
    return ["full_refusal"] * k_full + ["full_compliance"] * (n_judged - k_full) + [None] * (n_items - n_judged)


def _rc_standin(root: str, *, tag: str, store_id: str, object_id: str, ctl: str, caps: Dict[int, int], labels: Dict[int, List[Optional[str]]],
                items_on_members: bool, value: float, k: Optional[int], n: int, ci_method: str, salt: str = "", i0: int = 100,
                owner_full_refusal_p: Optional[float] = None) -> Dict[str, object]:
    """A pooled over-refusal run the way the controls readers file it, ingested as stand-in records: the run (pass_label pooled_final_rung,
    `extra.stores` the pass-2 stores, `extra.final_caps` per trial, `extra.owner_full_refusal_p`), its judgment, the shared computation, the
    owner `over_refusal_graded` metric row and the per-trial item labels -- on the run itself (relu_paired's shape) or on member runs named
    by `extra.member_run_ids` (sampled_controls' pooled anchor). `salt` mints a second run of the same identity (a second producer key)."""
    dims = F.dims_map()
    eval_id = IC.CONTROL_EVALS[ctl][0]
    instrument_id, item_set_id, _x = dims.instrument(CS.SAMPLED_INSTRUMENTS[ctl], eval_id, RC_MODEL)
    judge_id = F.judge_ids()[RC_JUDGE_SEED[ctl]]
    producer_key = f"{tag}_controls_final_rung" + (f"#{salt}" if salt else "")
    run_id = I.run_id(RC_CAMPAIGN, producer_key, store_id, eval_id, instrument_id)
    jid = I.judgment_id(run_id, judge_id)
    stores = [_rc_registry_store(tag, t, cap, store_id) for t, cap in sorted(caps.items())]
    n_items = {t: len(labs) for t, labs in labels.items()}
    n_judged = {t: sum(1 for lab in labs if lab) for t, labs in labels.items()}
    k_full = {t: sum(1 for lab in labs if lab == "full_refusal") for t, labs in labels.items()}
    p = owner_full_refusal_p if owner_full_refusal_p is not None else (sum(k_full.values()) / sum(n_judged.values()) if sum(n_judged.values()) else None)
    now = SY.t(i0)
    recs: List[Dict[str, object]] = []
    members: Dict[str, str] = {}
    if items_on_members:
        for t, cap in sorted(caps.items()):
            mid = I.run_id(RC_CAMPAIGN, f"{tag}_t{t}/{_rc_store_key(store_id, t)}", store_id, eval_id, instrument_id)
            mjid = I.judgment_id(mid, judge_id)
            members[str(t)] = mid
            recs.append(SY.rec("run", _rc_run_payload(mid, f"{tag}_t{t}/{_rc_store_key(store_id, t)}", store_id, object_id, eval_id, item_set_id, instrument_id,
                                                      judge_id, cap, tag, n_judged[t], now, store_uri=stores[t], trial_index=t, pass_label=None), now))
            recs.append(SY.rec("judgment", {"judgment_id": mjid, "run_id": mid, "judge_id": judge_id, "status": "complete", "n_items_judged": n_judged[t]}, now))
            recs.append(SY.rec("run_state", {"run_id": mid, "state": "judged", "event_utc": now, "evidence_kind": "harvester", "evidence_uri": stores[t]}, now))
            recs += _rc_item_records(mid, mjid, t, labels[t], "xst_label", now)
    extra = {"stores": stores, "final_caps": {str(t): cap for t, cap in caps.items()}, "owner_full_refusal_p": p, "n_draws": sum(n_items.values())}
    if members:
        extra["member_run_ids"] = members
    recs.insert(0, SY.rec("run", _rc_run_payload(run_id, producer_key, store_id, object_id, eval_id, item_set_id, instrument_id, judge_id, max(caps.values()), tag,
                                                 sum(n_judged.values()), now, store_uri=f"{RC_RS}/relu_paired/sampled/report/analysis/final_rung.json#s1",
                                                 pass_label="pooled_final_rung", extra=extra), now))
    recs.append(SY.rec("judgment", {"judgment_id": jid, "run_id": run_id, "judge_id": judge_id, "status": "complete", "n_items_judged": sum(n_judged.values())}, now))
    recs.append(SY.rec("run_state", {"run_id": run_id, "state": "judged", "event_utc": now, "evidence_kind": "harvester", "evidence_uri": stores[0]}, now))
    recs.append(SY.rec("computation", {"computation_id": RC_COMP, "algorithm": "synthetic pooled controls reader", "version": "1", "seed_provenance": "none",
                                       "code_sha": "unverified"}, now))
    recs.append(SY.rec("run_metric", {"run_id": run_id, "judgment_id": jid, "metric_id": ORC.OWNER_METRIC_ID, "threshold": -1.0, "computation_id": RC_COMP,
                                      "producer_metric_spelling": ORC.OWNER_METRIC_ID, "value": value, "k": k, "n": n, "ci_lo": max(0.0, value - 0.05),
                                      "ci_hi": min(1.0, value + 0.05), "ci_method": ci_method, "computed_by": "producer", "computed_utc": now,
                                      "source_uri": stores[0], "extra": {"k_effective": (k if k is not None else value * n), "reader": "synthetic"}}, now))
    if not items_on_members:
        for t in sorted(labels):
            recs += _rc_item_records(run_id, jid, t, labels[t], ORC.OWNER_METRIC_ID, now)
    fn = os.path.join(root, "records", "standins", f"{i0:06d}-rc-{hashlib.sha256(run_id.encode()).hexdigest()[:8]}.jsonl")
    R.write_records(fn, recs)
    return {"run_id": run_id, "judgment_id": jid, "eval_id": eval_id, "instrument_id": instrument_id, "stores": stores, "members": members, "caps": dict(caps),
            "labels": labels, "owner_full_refusal_p": p}


def _rc_run_payload(run_id, producer_key, store_id, object_id, eval_id, item_set_id, instrument_id, judge_id, cap, tag, n_judged, now, *, store_uri, pass_label,
                    trial_index=None, extra=None) -> Dict[str, object]:
    p = {"run_id": run_id, "campaign": RC_CAMPAIGN, "producer_run_key_legacy": producer_key, "producer_object_key_legacy": store_id, "lifecycle_state": "generated",
         "queued_utc": now, "started_utc": now, "generated_utc": now, "model_id": RC_MODEL, "object_id": object_id,
         "transfer_kind": ("none" if object_id == "none" else "base_to_instruct"), "eval_id": eval_id, "item_set_id": item_set_id, "instrument_id": instrument_id,
         "instrument_match": "unverified", "harness": "bespoke_sisteer", "harness_version": "synthetic", "scoring_rule": "judge_graded_weights",
         "prompt_template_sha256": "unverified", "judge_id": judge_id, "decode_seed_rule": "per_item", "decode_source": "unverified", "token_cap_tokens": cap,
         "token_cap_ladder_json": [512, 2048, 8192], "thinking_mode": "off", "serving_path": "lens", "applied_positions_mode": ("none" if object_id == "none" else "all"),
         "multi_turn_steering_scope": ("not_applicable" if object_id == "none" else "single_turn"), "code_sha": "unverified", "substrate": "cluster_a",
         "substrate_source": "recorded", "n_items_planned": n_judged, "n_items_generated": n_judged, "n_items_judged": n_judged, "store_uri": store_uri,
         "view_legacy": tag, "lane_source_legacy": "synth_controls", "campaign_cell_id_legacy": store_id, "extra": dict(extra or {})}
    conf = IC.Confidence()
    for col in ("code_sha", "prompt_template_sha256", "decode_source", "instrument_match"):
        conf.unverified(col, "synthetic stand-in of a pooled controls reading")
    p["field_confidence_json"] = conf.as_json()
    if pass_label:
        p["pass_label"] = pass_label
    if trial_index is not None:
        p["trial_index"] = trial_index
    return p


def _rc_item_records(run_id: str, jid: str, trial: int, labels: Sequence[Optional[str]], metric_id: str, now: str) -> List[Dict[str, object]]:
    out = []
    for i, lab in enumerate(labels):
        p = {"run_id": run_id, "item_id": f"xst{i:04d}-{'c' * 8}", "metric_id": metric_id, "trial_index": trial, "sample_index": 0, "turn_index": -1,
             "judgment_id": jid, "created_utc": now, "judged_utc": now, "api_pool": None, "source_revision_id": "", "withdrawn_by_revision_id": None,
             "label": lab, "value": (1.0 if lab == "full_refusal" else 0.0 if lab else None)}
        out.append(SY.rec("item_result", p, now))
    return out


def _rc_entry(model: str, label: str, trial: int, ctl: str, tag: str, store_id: str, cap: int, counts: Dict[str, int], components: Optional[Dict[str, float]] = None,
              labels_source: str = "pass-2 flip copy at the final rung") -> Tuple[str, Dict[str, object]]:
    """One self-consistent summary entry (the producer's v2 shape): the shares and rates derived from the counts the way the producer writes
    them, the graded estimand (full + 0.5 partial) over the non-collapsed and over every judged item beside the full-only ones."""
    c = dict(counts)
    key = f"{model}|{label}|t{trial}|{ctl}"
    n_nc = c["n_non_collapsed_judged"]
    entry = {"model": model, "cell": store_id, "control": ctl, "tag": f"{tag}_t{trial}", "store": _rc_or_store(tag, trial, cap, store_id), "final_rung": cap,
             "counts": c, "components": dict(components or RC_COMPONENTS_ZERO), "collapse_share": c["n_collapsed"] / c["n_items"],
             "n_items": c["n_items"], "n_judged": c["n_judged"], "n_full_refusal": c["k_full_refusal_all_judged"], "n_non_collapsed_judged": n_nc,
             "rate_over_non_collapsed": (c["k_full_refusal_non_collapsed"] / n_nc) if n_nc else None,
             "graded_over_non_collapsed": ((c["k_full_refusal_non_collapsed"] + 0.5 * c["k_partial_refusal_non_collapsed"]) / n_nc) if n_nc else None,
             "filed_full_refusal_rate": (c["k_full_refusal_all_judged"] / c["n_judged"]) if c["n_judged"] else None,
             "graded_all_judged": ((c["k_full_refusal_all_judged"] + 0.5 * c["k_partial_refusal_all_judged"]) / c["n_judged"]) if c["n_judged"] else None,
             "full_refusal_with_refusal_phrase": (c["k_full_refusal_with_refusal_phrase"] / c["k_full_refusal_all_judged"]) if c["k_full_refusal_all_judged"] else None,
             "labels_source": labels_source, "collapse_by_label": {}}
    return key, entry


def _rc_counts(n_items, n_judged, n_collapsed, n_collapsed_judged, k_fr_nc, k_pr_nc, k_fr_all, k_fr_phrase, k_pr_all: Optional[int] = None) -> Dict[str, int]:
    return {"n_items": n_items, "n_judged": n_judged, "n_collapsed": n_collapsed, "n_collapsed_judged": n_collapsed_judged,
            "n_non_collapsed_judged": n_judged - n_collapsed_judged, "k_full_refusal_non_collapsed": k_fr_nc, "k_partial_refusal_non_collapsed": k_pr_nc,
            "k_full_refusal_all_judged": k_fr_all, "k_partial_refusal_all_judged": (k_pr_nc if k_pr_all is None else k_pr_all),
            "k_full_refusal_with_refusal_phrase": k_fr_phrase}


def _rc_items_for(key: str, counts: Dict[str, int]) -> List[Dict[str, object]]:
    """Item rows reproducing a key's counts: the first `n_collapsed` rows collapsed (the first `n_collapsed_judged` of them judged), the judged
    rows' labels spelling `k_full_refusal_all_judged` full refusals."""
    c = counts
    rows = []
    judged_flags = [True] * c["n_collapsed_judged"] + [False] * (c["n_collapsed"] - c["n_collapsed_judged"]) + [True] * c["n_non_collapsed_judged"]
    judged_flags += [False] * (c["n_items"] - len(judged_flags))
    full_left = c["k_full_refusal_all_judged"]
    for i, judged in enumerate(judged_flags):
        lab = None
        if judged:
            lab = "full_refusal" if full_left > 0 else "full_compliance"
            full_left -= int(lab == "full_refusal")
        rows.append({"key": key, "item_id": f"xst{i:04d}-{'c' * 8}", "off_task_role_collapse": i < c["n_collapsed"], "label": lab, "loop": False, "at_cap": False,
                     "refusal_phrase_400": False, "step_template": False, "code_scaffold": False, "echo": False, "first_person_short": False,
                     "refusal_pattern_ids": [], "n_words": 12, "distinct_ratio": 0.9})
    return rows


#: the good world's counts: the steered pair's two trials on each control; the anchor's t0 with labels (the producer's v2 reads them off
#: the sampled controls' pass-2 flip copies): XSTest 1 full refusal of 10 judged, 1 collapsed, none partial -> graded_all_judged 0.1
#: against an owner value of 0.11
RC_PAIRED_XST = {0: _rc_counts(10, 10, 4, 4, 2, 1, 4, 1, k_pr_all=2), 1: _rc_counts(10, 9, 3, 3, 3, 0, 5, 2, k_pr_all=1)}
RC_PAIRED_ORB = {0: _rc_counts(12, 12, 6, 6, 1, 0, 3, 0, k_pr_all=2), 1: _rc_counts(12, 12, 6, 6, 2, 1, 4, 1, k_pr_all=1)}
RC_ANCHOR_XST = {0: _rc_counts(10, 10, 1, 1, 1, 0, 1, 1)}
RC_ANCHOR_ORB = {0: _rc_counts(12, 12, 2, 2, 5, 0, 5, 5)}
#: the anchor's t0 with NO judge labels (the producer's v1 shape, still a valid key): k 0 / n 0 / value NULL
RC_ANCHOR_XST_UNLABELLED = {0: _rc_counts(10, 0, 1, 0, 0, 0, 0, 0)}


def _rc_world(tmp_path, *, paired_labels_xst: Optional[Dict[int, List[Optional[str]]]] = None) -> Dict[str, object]:
    """A registry root with the dimension seeds and four stand-in pooled runs: the steered pair on XSTest and OR-Bench (items on the run),
    the unsteered anchor on XSTest (items on member runs) and TWO current anchors on OR-Bench of one identity (the >1-match refusal)."""
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    xst_labels = paired_labels_xst or {t: _rc_labels(c["n_judged"], c["k_full_refusal_all_judged"], c["n_items"]) for t, c in RC_PAIRED_XST.items()}
    w: Dict[str, object] = {"root": root}
    w["paired_xst"] = _rc_standin(root, tag=RC_PAIRED_TAG, store_id=RC_STEERED, object_id=F.OBJ_SHA, ctl="xstest", caps={0: 2048, 1: 2048}, labels=xst_labels,
                                  items_on_members=False, value=0.45, k=5, n=10, ci_method="wilson_95", i0=100)
    w["paired_orb"] = _rc_standin(root, tag=RC_PAIRED_TAG, store_id=RC_STEERED, object_id=F.OBJ_SHA, ctl="or_bench_hard_1k", caps={0: 512, 1: 2048},
                                  labels={t: _rc_labels(c["n_judged"], c["k_full_refusal_all_judged"], c["n_items"]) for t, c in RC_PAIRED_ORB.items()},
                                  items_on_members=False, value=0.3, k=4, n=12, ci_method="wilson_95", i0=110)
    w["anchor_xst"] = _rc_standin(root, tag=RC_ANCHOR_TAG, store_id=NONE, object_id="none", ctl="xstest", caps={0: 2048, 1: 2048},
                                  labels={0: _rc_labels(10, 1, 10), 1: _rc_labels(10, 2, 10)}, items_on_members=True, value=0.11, k=None, n=10, ci_method="wilson", i0=120)
    w["anchor_orb_a"] = _rc_standin(root, tag=RC_ANCHOR_TAG, store_id=NONE, object_id="none", ctl="or_bench_hard_1k", caps={0: 2048, 1: 2048},
                                    labels={0: _rc_labels(12, 5, 12), 1: _rc_labels(12, 6, 12)}, items_on_members=True, value=0.4, k=None, n=12, ci_method="wilson", i0=130)
    w["anchor_orb_b"] = _rc_standin(root, tag=RC_ANCHOR_TAG, store_id=NONE, object_id="none", ctl="or_bench_hard_1k", caps={0: 2048, 1: 2048},
                                    labels={0: _rc_labels(12, 5, 12), 1: _rc_labels(12, 6, 12)}, items_on_members=True, value=0.4, k=None, n=12, ci_method="wilson",
                                    salt="second instrument", i0=140)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refusals, rep.blocked)
    return w


def _rc_entries(w: Mapping[str, object], anchor_xst: Optional[Dict[int, Dict[str, int]]] = None) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for t, c in RC_PAIRED_XST.items():
        k, e = _rc_entry(RC_MODEL, "difference_relu_paired", t, "xstest", RC_PAIRED_TAG, RC_STEERED, 2048, c); out[k] = e
    for t, c in RC_PAIRED_ORB.items():
        k, e = _rc_entry(RC_MODEL, "difference_relu_paired", t, "or_bench_hard_1k", RC_PAIRED_TAG, RC_STEERED, w["paired_orb"]["caps"][t], c); out[k] = e
    src = "sampled lane pass-2 flip copy at the final rung"
    k, e = _rc_entry(RC_MODEL, "none_sampled_anchor", 0, "xstest", RC_ANCHOR_TAG, NONE, 2048, (anchor_xst or RC_ANCHOR_XST)[0], labels_source=src); out[k] = e
    k, e = _rc_entry(RC_MODEL, "none_sampled_anchor", 0, "or_bench_hard_1k", RC_ANCHOR_TAG, NONE, 2048, RC_ANCHOR_ORB[0], labels_source=src); out[k] = e
    return out


def _rc_write(tmp_path, entries: Mapping[str, Mapping[str, object]], items: Optional[Sequence[Dict[str, object]]] = None, *, rule: str = ORC.RULE_ID,
              with_items: bool = True, with_module: bool = True, top: Optional[Dict[str, object]] = None, version: int = 1,
              items_version: Optional[int] = None) -> str:
    """The producer's files at ``version`` (a bare name for 1, ``.v<N>`` beyond); ``items_version`` writes the items file under another version."""
    d = str(tmp_path / "rs" / "orxs_seedrep" / ORC.DIR_NAME)
    os.makedirs(d, exist_ok=True)
    doc: Dict[str, object] = {"cells": dict(entries), "definition": "role_collapse_v1 -- synthetic copy of OR's rule text", "rule": rule,
                              "version": f"v{version}: synthetic", "metric_note": "synthetic copy of OR's note"}
    doc.update(top or {})
    _json(os.path.join(d, ORC.versioned_name(ORC.SUMMARY_STEM, version)), doc)
    if with_items:
        rows = list(items) if items is not None else [r for k, e in entries.items() for r in _rc_items_for(k, e["counts"])]
        _jsonl(os.path.join(d, ORC.versioned_name(ORC.ITEMS_STEM, items_version if items_version is not None else version)), rows)
    if with_module:
        with open(os.path.join(d, ORC.MODULE_NAME), "w", encoding="utf-8") as fh:
            fh.write("RULE_ID = 'role_collapse_v1'\n")
    return d


def _rc_adapter(d: str) -> CS.OrxsSeedrepAdapter:
    return CS.OrxsSeedrepAdapter(roots=(), role_collapse_dir=d)


def _rc_read(ad: CS.OrxsSeedrepAdapter, root: str):
    inputs = ad.discover()
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        batches = list(ad.read(_ctx(conn), ORC.GROUP, inputs))
    finally:
        conn.close()
    assert len(batches) == 1 and batches[0].source_id == "orxs_seedrep:allmodels_paired/role_collapse_v1"
    return batches[0], ad.status_extra(ORC.GROUP)


def _rc_q(root: str, sql: str, params=()):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _rc_owner_rows(root: str):
    return _rc_q(root, f"SELECT * FROM run_metric WHERE metric_id = '{ORC.OWNER_METRIC_ID}' ORDER BY run_id, judgment_id, threshold, computation_id")


def test_role_collapse_group_is_discovered_under_the_source_string_with_the_spelling_pinned(tmp_path):
    w = _rc_world(tmp_path)
    d = _rc_write(tmp_path, _rc_entries(w))
    ad = _rc_adapter(d)
    inputs = ad.discover()
    assert sorted(os.path.basename(i.uri) for i in inputs) == sorted([ORC.SUMMARY_NAME, ORC.ITEMS_NAME, ORC.MODULE_NAME])
    assert {ad.comparison_group(i) for i in inputs} == {ORC.GROUP} and ORC.GROUP == "allmodels_paired/role_collapse_v1"
    assert f"{ad.name}:{ORC.GROUP}" == "orxs_seedrep:allmodels_paired/role_collapse_v1" and ad.status_sources(ORC.GROUP) == ("orxs_seedrep",)
    assert {i.role for i in inputs} == {"report", "rows"} and ad.EMISSION_VERSION == 5     # 5: the anchor currency; 4: the label-mean row on the store groups
    # the companion metric passes the metric-naming CHECK (no migration), sits beside `over_refusal_graded`, and the extra keys are the spelling
    from sisteer.registry import vocab as V
    assert ORC.METRIC_ID == "over_refusal_graded_noncollapsed" and ORC.METRIC_ID not in V.banned_metric_ids() and ORC.OWNER_METRIC_ID == "over_refusal_graded"
    assert ORC.EXTRA_KEYS == ("role_collapse_share", "role_collapse_n", "n_items", "n_collapsed", "n_collapsed_judged", "noncollapsed_k", "noncollapsed_n",
                              "k_full_refusal_all_judged", "k_partial_refusal_non_collapsed", "k_full_refusal_with_refusal_phrase", "final_rung", "components",
                              "filed_rate_matches", "source_file", "source_sha256", "definition")
    assert not set(ORC.EXTRA_KEYS) & set(ORC.EXTRA_KEYS_BESIDE) and "metric_role" in ORC.EXTRA_KEYS_BESIDE
    assert {"rate_over_non_collapsed_full", "graded_all_judged", "k_effective", "source_version", "inputs_retired"} <= set(ORC.EXTRA_KEYS_BESIDE)
    assert ORC.PARTIAL_WEIGHT == 0.5 and ORC.PRODUCER_METRIC_SPELLING == "graded_over_non_collapsed" and ORC.ANCHOR_GRADED_TOLERANCE == 0.02
    assert "k_partial_refusal_all_judged" in ORC.COUNT_KEYS
    # the module roots name the producer's directory on the mirror; an absent root discovers nothing and refuses nothing
    assert ORC.ROOT == os.path.join(IC.RUN_STATE, "orxs_seedrep", "allmodels_paired")
    assert ORC.discover_inputs(str(tmp_path / "nowhere")) == [] and _rc_adapter(str(tmp_path / "nowhere")).discover() == []
    assert ORC.retired_inputs(str(tmp_path / "nowhere")) == [] and _rc_adapter(str(tmp_path / "nowhere")).inputs_retired(ORC.GROUP) == []
    # the version grammar: a bare name is 1, `.v<N>` is N, another stem / extension / a superseded copy is nothing
    assert ORC.version_of("role_collapse_v1_summary.json", ORC.SUMMARY_STEM) == 1 and ORC.version_of("role_collapse_v1_summary.v2.json", ORC.SUMMARY_STEM) == 2
    assert ORC.version_of("role_collapse_v1_items.v13.jsonl", ORC.ITEMS_STEM) == 13 and ORC.version_of("role_collapse_v1_items.jsonl", ORC.ITEMS_STEM) == 1
    assert ORC.version_of("role_collapse_v1_summary.v2.json", ORC.ITEMS_STEM) is None and ORC.version_of("role_collapse_v1_summary.v2.jsonl", ORC.SUMMARY_STEM) is None
    assert ORC.version_of("role_collapse_v1_summary.json.superseded-by-v2", ORC.SUMMARY_STEM) is None and ORC.version_of("role_collapse_v1.py", ORC.SUMMARY_STEM) is None
    assert ORC.versioned_name(ORC.SUMMARY_STEM, 1) == "role_collapse_v1_summary.json" and ORC.versioned_name(ORC.ITEMS_STEM, 3) == "role_collapse_v1_items.v3.jsonl"


def test_role_collapse_maps_keys_to_the_pooled_runs_and_files_one_companion_row_per_run(tmp_path):
    w = _rc_world(tmp_path)
    d = _rc_write(tmp_path, _rc_entries(w))
    batch, facts = _rc_read(_rc_adapter(d), w["root"])
    rows = {r["payload"]["run_id"]: r["payload"] for r in batch.records}
    assert all(r["record_type"] == "run_metric" for r in batch.records) and len(rows) == 3
    assert facts["n_keys"] == 6 and facts["keys_filed"] == 5 and facts["runs_filed"] == 3 and facts["keys_refused"] == 1
    # the two trial keys of a steered pair land as ONE row on the pooled run, on the owner's judgment / threshold / computation
    px = rows[w["paired_xst"]["run_id"]]
    assert (px["judgment_id"], px["threshold"], px["computation_id"], px["metric_id"]) == (w["paired_xst"]["judgment_id"], -1.0, RC_COMP, ORC.METRIC_ID)
    # the GRADED estimand over the non-collapsed items: t0 2 full + 0.5 x 1 partial of 6, t1 3 full of 6 -> k 5.5 (the INTEGER column rounds as the
    # owner's does, the exact k beside it), n 12, value 5.5 / 12 = the producer's graded_over_non_collapsed pooled
    assert (px["k"], px["n"]) == (6, 6 + 6) and px["value"] == pytest.approx(5.5 / 12) and px["ci_method"] == "wilson_95" and px["computed_by"] == "registry"
    lo, hi = IC.wilson(5.5, 12)
    assert (px["ci_lo"], px["ci_hi"]) == (lo, hi) and px["n_abstentions"] == (10 - 10) + (10 - 9) and px["producer_metric_spelling"] == "graded_over_non_collapsed"
    ex = px["extra"]
    assert set(ex) == set(ORC.EXTRA_KEYS) | set(ORC.EXTRA_KEYS_BESIDE)
    assert (ex["noncollapsed_k"], ex["k_effective"], ex["noncollapsed_n"], ex["n_items"], ex["n_collapsed"], ex["n_collapsed_judged"], ex["role_collapse_n"]) == (5.5, 5.5, 12, 20, 7, 7, 19)
    assert ex["k_full_refusal_non_collapsed"] == 5 and ex["rate_over_non_collapsed_full"] == pytest.approx(5 / 12)
    assert ex["graded_all_judged"] == pytest.approx((4 + 0.5 * 2 + 5 + 0.5 * 1) / 19) and ex["k_partial_refusal_all_judged"] == 3
    assert ex["role_collapse_share"] == pytest.approx(7 / 20) and ex["k_full_refusal_all_judged"] == 9 and ex["k_partial_refusal_non_collapsed"] == 1
    assert ex["k_full_refusal_with_refusal_phrase"] == 3 and ex["final_rung"] == 2048 and ex["definition"] == "role_collapse_v1" and ex["metric_role"] == "secondary"
    assert ex["trials_measured"] == [0, 1] and ex["trial_coverage"] == "complete" and sorted(ex["by_trial"]) == ["0", "1"] and ex["rate_state"] == "measured"
    assert ex["by_trial"]["1"]["noncollapsed_k"] == 3 and ex["by_trial"]["1"]["graded_over_non_collapsed"] == pytest.approx(0.5) and ex["by_trial"]["1"]["final_rung"] == 2048
    assert ex["by_trial"]["0"]["noncollapsed_k"] == 2.5 and ex["by_trial"]["0"]["rate_over_non_collapsed_full"] == pytest.approx(2 / 6)
    assert ex["by_trial"]["0"]["graded_all_judged"] == pytest.approx(0.5) and ex["by_trial"]["0"]["k_partial_refusal_all_judged"] == 2
    assert ex["components"] == RC_COMPONENTS_ZERO and ex["source_sha256"] == _rc_sha(os.path.join(d, ORC.SUMMARY_NAME))
    assert ex["items_sha256"] == _rc_sha(os.path.join(d, ORC.ITEMS_NAME)) and ex["definition_module_sha256"] == _rc_sha(os.path.join(d, ORC.MODULE_NAME))
    assert ex["source_file"].endswith(f"orxs_seedrep/allmodels_paired/{ORC.SUMMARY_NAME}") and px["source_uri"].startswith(ex["source_file"] + " ›cells[")
    assert ex["source_version"] == {"file_version": 1, "version_note": "v1: synthetic", "metric_note": "synthetic copy of OR's note"} and ex["inputs_retired"] == []
    # filed_rate_matches (steered): the producer's full refusals / judged per trial against the registry's filed labels, the pooled rate
    # against owner_full_refusal_p; graded_vs_owner recorded, not judged (the owner value 0.45 is a stand-in, far from the producer's
    # graded 0.5 x ... = 0.4737)
    assert ex["filed_rate_matches"] is True
    chk = ex["filed_rate_check"]
    assert chk["shape"] == "steered" and chk["by_trial"]["0"] == {"compared": True, "or_k_full_refusal": 4, "or_n_judged": 10, "registry_k_full_refusal": 4,
                                                                  "registry_n_judged": 10, "registry_where": f"run {w['paired_xst']['run_id']} items at trial 0", "matches": True}
    assert chk["pooled"] == {"compared": True, "or_pooled_full_refusal_rate": 9 / 19, "registry_owner_full_refusal_p": 9 / 19, "matches": True}
    assert chk["graded_vs_owner"]["compared"] is True and chk["graded_vs_owner"]["owner_value"] == 0.45 and chk["graded_vs_owner"]["trials"] == "every pooled trial"
    assert chk["graded_vs_owner"]["or_graded_all_judged"] == pytest.approx(10.5 / 19) and chk["graded_vs_owner"]["within_tolerance"] is False
    # a pair whose trials exited at different caps files final_rung per trial
    po = rows[w["paired_orb"]["run_id"]]
    assert po["extra"]["final_rung"] == {"0": 512, "1": 2048} and (po["k"], po["n"]) == (4, 12) and po["extra"]["noncollapsed_k"] == 3.5 and po["extra"]["filed_rate_matches"] is True
    # the anchor: t0 alone against a run pooling t0 / t1; the producer's v2 read its labels -> a real k / n / value, and the check is its
    # graded_all_judged against the owner value within the tolerance (0.1 vs 0.11: within)
    ax = rows[w["anchor_xst"]["run_id"]]
    assert (ax["value"], ax["k"], ax["n"], ax["ci_method"], ax["n_abstentions"]) == (pytest.approx(1 / 9), 1, 9, "wilson", 0)
    assert ax["extra"]["rate_state"] == "measured" and ax["extra"]["role_collapse_share"] == pytest.approx(0.1) and ax["extra"]["filed_rate_matches"] is True
    assert ax["extra"]["trials_measured"] == [0] and ax["extra"]["trial_coverage"] == "partial: trials [0] of the run's [0, 1]"
    achk = ax["extra"]["filed_rate_check"]
    assert achk["shape"] == "anchor" and "by_trial" not in achk and achk["graded_vs_owner"] == {
        "compared": True, "or_graded_all_judged": pytest.approx(0.1), "owner_value": 0.11, "difference": pytest.approx(-0.01), "tolerance": 0.02,
        "within_tolerance": True, "trials": "OR read trials ['0'] of the run's ['0', '1']"}
    assert ax["judgment_id"] == w["anchor_xst"]["judgment_id"]
    # the OR-Bench anchor names TWO current runs of one identity: refused by name, nothing filed for it
    (why,) = [r for r in facts["refused"] if r.startswith(f"{RC_MODEL}|none_sampled_anchor|t0|or_bench_hard_1k")]
    assert "matches 2 current runs" in why and w["anchor_orb_a"]["run_id"] in why and w["anchor_orb_b"]["run_id"] in why
    assert w["anchor_orb_a"]["run_id"] not in rows and w["anchor_orb_b"]["run_id"] not in rows
    assert facts["filed"][w["paired_xst"]["run_id"]]["or_keys"] == [f"{RC_MODEL}|difference_relu_paired|t0|xstest", f"{RC_MODEL}|difference_relu_paired|t1|xstest"]
    assert facts["filed_rate_mismatches"] == [] and facts["rule"] == "role_collapse_v1" and facts["metric_id"] == ORC.METRIC_ID


def test_role_collapse_refuses_by_name_and_files_the_rest(tmp_path):
    w = _rc_world(tmp_path)
    entries = _rc_entries(w)
    base_items = [r for k, e in entries.items() for r in _rc_items_for(k, e["counts"])]
    orb0 = f"{RC_MODEL}|difference_relu_paired|t0|or_bench_hard_1k"
    orb1 = f"{RC_MODEL}|difference_relu_paired|t1|or_bench_hard_1k"
    xst1 = f"{RC_MODEL}|difference_relu_paired|t1|xstest"
    # (a) a store label the registry files no run for; (b) a missing component; (c) collapse_share disagreeing with the counts;
    # (d) final_rung disagreeing with the run's final cap of the trial; (e) a key the items file does not reproduce; (f) a key spelling no trial;
    # (g) a control the registry does not file; (h) a store path spelling another target directory than `tag`
    ka, ea = _rc_entry(RC_MODEL, "difference_lora_paired", 0, "xstest", RC_PAIRED_TAG, RC_UNKNOWN, 2048, RC_PAIRED_XST[0]); entries[ka] = ea
    del entries[orb0]["components"]["loop"]
    entries[orb1]["collapse_share"] = 0.99
    entries[xst1]["final_rung"] = 512
    entries[xst1]["store"] = entries[xst1]["store"].replace("controls_cap2048", "controls_cap512")
    kg, eg = _rc_entry(RC_MODEL, "difference_relu_paired", 0, "mmlu", RC_PAIRED_TAG, RC_STEERED, 2048, RC_PAIRED_XST[0]); entries[kg] = eg
    entries["nonsense-key"] = dict(entries[f"{RC_MODEL}|difference_relu_paired|t0|xstest"])
    kh, eh = _rc_entry(RC_MODEL, "none_sampled_anchor", 0, "xstest", RC_ANCHOR_TAG, NONE, 2048, RC_ANCHOR_XST[0]); eh["tag"] = "othermodel_t0"
    entries.pop(f"{RC_MODEL}|none_sampled_anchor|t0|xstest"); entries[kh] = eh
    # (i) a graded rate OR states that the counts do not give; (j) a counts block without the partial count the graded estimand needs
    # (their keys spell another store label: the key format `<model>|<label>|t<T>|<control>` carries no tag, so a label shared with the good keys would overwrite them)
    ki, ei = _rc_entry(RC_MODEL, "difference_relu_paired_b", 0, "xstest", RC_PAIRED_TAG, RC_STEERED, 2048, RC_PAIRED_XST[0]); ei["graded_over_non_collapsed"] = 0.9
    entries[ki] = ei
    kj, ej = _rc_entry(RC_MODEL, "difference_relu_paired_b", 1, "xstest", RC_PAIRED_TAG, RC_STEERED, 2048, RC_PAIRED_XST[1]); del ej["counts"]["k_partial_refusal_all_judged"]
    entries[kj] = ej
    items = [r for r in base_items if r["key"] != f"{RC_MODEL}|difference_relu_paired|t0|xstest"] + _rc_items_for(f"{RC_MODEL}|difference_relu_paired|t0|xstest", RC_PAIRED_XST[0])[:-1]
    items += _rc_items_for(ka, RC_PAIRED_XST[0]) + _rc_items_for(kg, RC_PAIRED_XST[0]) + _rc_items_for(kh, RC_ANCHOR_XST[0])
    items += _rc_items_for(ki, RC_PAIRED_XST[0]) + _rc_items_for(kj, RC_PAIRED_XST[1])
    d = _rc_write(tmp_path, entries, items)
    batch, facts = _rc_read(_rc_adapter(d), w["root"])
    refused = {r.split(" -- ", 1)[0]: r.split(" -- ", 1)[1] for r in facts["refused"]}
    assert "the key matches no run" in refused[ka] and RC_UNKNOWN in refused[ka]
    assert refused[orb0].startswith("components.loop None is not a number")
    assert "collapse_share reads 0.99" in refused[orb1] and "the file disagrees with itself" in refused[orb1]
    assert "final_rung 512 is not the run's final cap of trial t1 (2048)" in refused[xst1]
    assert refused[f"{RC_MODEL}|difference_relu_paired|t0|xstest"] == "summary and items disagree on n_items: summary 10 vs items 9; n_judged: summary 10 vs items 9"
    assert "is not an over-refusal control the registry judges (xstest, or_bench_hard_1k)" in refused[kg] and "'mmlu'" in refused[kg]
    assert "does not spell <model>|<label>|t<T>|<control>" in refused["nonsense-key"]
    assert "tag 'othermodel_t0' is not the store path's target directory" in refused[kh]
    assert "matches 2 current runs" in refused[f"{RC_MODEL}|none_sampled_anchor|t0|or_bench_hard_1k"]
    assert "graded_over_non_collapsed reads 0.9" in refused[ki] and "the file disagrees with itself" in refused[ki]
    assert refused[kj].startswith("counts.k_partial_refusal_all_judged None is not a count")
    assert len(refused) == 11 and batch.records == [] and facts["runs_filed"] == 0 and facts["keys_filed"] == 0
    # every refusal is durable through the group's source_state rows
    conn = DB.connect(DB.db_path(w["root"]), readonly=True)
    try:
        ad = _rc_adapter(d)
        inputs = ad.discover()
        list(ad.read(_ctx(conn), ORC.GROUP, inputs))
        states = ad.source_states(_ctx(conn), inputs)
    finally:
        conn.close()
    paths = {s.json_path: s for s in states}
    assert set(paths) == {"$.cells", "$.row_refusals"} and len(paths["$.row_refusals"].current_members) == 11 and set(paths["$.cells"].current_members) == set(entries)
    assert all(s.source_id == "orxs_seedrep:allmodels_paired/role_collapse_v1" for s in states)


def test_role_collapse_filed_rate_mismatch_is_recorded_never_a_refusal(tmp_path):
    # the registry's XSTest labels at t0 carry 3 full refusals where OR counts 4: the row files, filed_rate_matches reads False, the facts name the run;
    # the XSTest anchor's OR labels give graded 0.4 against the owner's 0.11: outside the tolerance, recorded the same way
    labels = {0: _rc_labels(10, 3, 10), 1: _rc_labels(9, 5, 10)}
    w = _rc_world(tmp_path, paired_labels_xst=labels)
    d = _rc_write(tmp_path, _rc_entries(w, anchor_xst={0: _rc_counts(10, 10, 1, 1, 4, 0, 4, 4)}))
    batch, facts = _rc_read(_rc_adapter(d), w["root"])
    px = next(r["payload"] for r in batch.records if r["payload"]["run_id"] == w["paired_xst"]["run_id"])
    assert (px["k"], px["n"]) == (6, 12) and px["extra"]["filed_rate_matches"] is False
    chk = px["extra"]["filed_rate_check"]
    assert chk["by_trial"]["0"]["matches"] is False and chk["by_trial"]["0"]["registry_k_full_refusal"] == 3 and chk["by_trial"]["1"]["matches"] is True
    assert chk["pooled"]["matches"] is False and chk["pooled"]["registry_owner_full_refusal_p"] == pytest.approx(8 / 19)
    ax = next(r["payload"] for r in batch.records if r["payload"]["run_id"] == w["anchor_xst"]["run_id"])
    assert (ax["k"], ax["n"], ax["value"]) == (4, 9, pytest.approx(4 / 9)) and ax["extra"]["filed_rate_matches"] is False
    assert ax["extra"]["filed_rate_check"]["graded_vs_owner"]["difference"] == pytest.approx(0.29) and ax["extra"]["filed_rate_check"]["graded_vs_owner"]["within_tolerance"] is False
    assert sorted(facts["filed_rate_mismatches"]) == sorted([w["paired_xst"]["run_id"], w["anchor_xst"]["run_id"]]) and facts["keys_refused"] == 1   # the OR-Bench anchor's two-run refusal only
    po = next(r["payload"] for r in batch.records if r["payload"]["run_id"] == w["paired_orb"]["run_id"])
    assert po["extra"]["filed_rate_matches"] is True
    # an anchor OR read without labels (the v1 shape) still files: k 0 / n 0 / value NULL, the collapse share beside it, no check
    d2 = _rc_write(tmp_path / "unlabelled", _rc_entries(w, anchor_xst=RC_ANCHOR_XST_UNLABELLED))
    batch2, _facts2 = _rc_read(_rc_adapter(d2), w["root"])
    ax2 = next(r["payload"] for r in batch2.records if r["payload"]["run_id"] == w["anchor_xst"]["run_id"])
    assert (ax2["value"], ax2["k"], ax2["n"], ax2["ci_lo"], ax2["ci_method"], ax2["n_abstentions"]) == (None, 0, 0, None, None, None)
    assert ax2["extra"]["rate_state"].startswith("not_measured") and ax2["extra"]["filed_rate_matches"] is None and ax2["extra"]["role_collapse_share"] == pytest.approx(0.1)
    assert ax2["extra"]["filed_rate_check"]["graded_vs_owner"] == {"compared": False, "why": "OR read no labels", "tolerance": 0.02}


def test_role_collapse_whole_group_refusals(tmp_path):
    w = _rc_world(tmp_path)
    entries = _rc_entries(w)
    # no registry connection
    d = _rc_write(tmp_path, entries)
    ad = _rc_adapter(d)
    with pytest.raises(AB.AdapterRefusal, match="no registry connection"):
        list(ad.read(_ctx(None), ORC.GROUP, ad.discover()))
    conn = DB.connect(DB.db_path(w["root"]), readonly=True)
    try:
        # the items file absent
        d2 = _rc_write(tmp_path / "b", entries, with_items=False)
        ad2 = _rc_adapter(d2)
        with pytest.raises(AB.AdapterRefusal, match="is beside the summary"):
            list(ad2.read(_ctx(conn), ORC.GROUP, ad2.discover()))
        # the items file of another version than the summary's: refused by name, naming both
        d6 = _rc_write(tmp_path / "f", entries, version=2, items_version=1)
        ad6 = _rc_adapter(d6)
        assert sorted(os.path.basename(i.uri) for i in ad6.discover()) == ["role_collapse_v1.py", "role_collapse_v1_items.jsonl", "role_collapse_v1_summary.v2.json"]
        with pytest.raises(AB.AdapterRefusal, match="the items file is version 1 but the summary role_collapse_v1_summary.v2.json is version 2"):
            list(ad6.read(_ctx(conn), ORC.GROUP, ad6.discover()))
        # another rule
        d3 = _rc_write(tmp_path / "c", entries, rule="role_collapse_v2")
        ad3 = _rc_adapter(d3)
        with pytest.raises(AB.AdapterRefusal, match="this reader knows one rule"):
            list(ad3.read(_ctx(conn), ORC.GROUP, ad3.discover()))
        # no `cells` object
        d4 = str(tmp_path / "d" / "rs" / "orxs_seedrep" / ORC.DIR_NAME)
        _json(os.path.join(d4, ORC.SUMMARY_NAME), {"rule": ORC.RULE_ID, "cells": []})
        _jsonl(os.path.join(d4, ORC.ITEMS_NAME), [])
        ad4 = _rc_adapter(d4)
        with pytest.raises(AB.AdapterRefusal, match="`cells` object"):
            list(ad4.read(_ctx(conn), ORC.GROUP, ad4.discover()))
        # an items line that is not a row
        d5 = _rc_write(tmp_path / "e", entries, items=[{"no_key": 1}])
        ad5 = _rc_adapter(d5)
        with pytest.raises(AB.AdapterRefusal, match="line 1 is not an object with a string `key`"):
            list(ad5.read(_ctx(conn), ORC.GROUP, ad5.discover()))
    finally:
        conn.close()


def test_role_collapse_current_run_reads_the_observation_table():
    """A run is current when an observation of table `run` carries it with current = 1, or when no revision carries it at all (a stand-in,
    a natively emitted run); a run every revision has withdrawn is not."""
    conn = DB.connect_memory()
    apply_migrations(conn)
    conn.execute("PRAGMA foreign_keys = OFF")        # the observation rows alone, without the revisions they would cite: the lookup is what is under test
    key = lambda rid: I.canonical_json({"run_id": rid})
    rows = [("src:a", "rev1", key("run-withdrawn"), "run", 0), ("src:a", "rev2", key("run-current"), "run", 0), ("src:a", "rev3", key("run-current"), "run", 1)]
    conn.executemany("INSERT INTO observation(source_id, revision_id, observation_key, table_name, current, field_versions_json, extra_json, record_uri, ingest_seq) "
                     "VALUES (?, ?, ?, ?, ?, '{}', '{}', 'synthetic', 1)", rows)
    assert ORC.is_current_run(conn, "run-withdrawn") is False and ORC.is_current_run(conn, "run-current") is True and ORC.is_current_run(conn, "run-never-carried") is True
    conn.close()


def test_role_collapse_pass_and_ingest_leave_every_owner_row_byte_identical_and_file_the_companions(tmp_path):
    w = _rc_world(tmp_path)
    root = w["root"]
    d = _rc_write(tmp_path, _rc_entries(w))
    before = _rc_owner_rows(root)
    assert len(before) == 5
    ad = _rc_adapter(d)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_held == 0 and rep.groups_refreshed == 1, [r.error for r in rep.results if r.error]
    assert [r.group for r in rep.results] == [ORC.GROUP]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    assert _rc_owner_rows(root) == before                                             # beside, never instead: the owner rows' bytes did not move
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = conn.execute(f"SELECT run_id, judgment_id, threshold, computation_id, value, k, n, ci_method, computed_by, extra_json FROM run_metric WHERE metric_id = '{ORC.METRIC_ID}' ORDER BY run_id").fetchall()
    assert len(rows) == 3 and {r[0] for r in rows} == {w["paired_xst"]["run_id"], w["paired_orb"]["run_id"], w["anchor_xst"]["run_id"]}
    for r in rows:
        assert r[3] == RC_COMP and r[8] == "registry" and json.loads(r[9])["definition"] == "role_collapse_v1"
    px = next(r for r in rows if r[0] == w["paired_xst"]["run_id"])
    assert (px[1], px[4], px[5], px[6], px[7]) == (w["paired_xst"]["judgment_id"], pytest.approx(5.5 / 12), 6, 12, "wilson_95")
    assert json.loads(px[9])["noncollapsed_k"] == 5.5 and json.loads(px[9])["k_effective"] == 5.5
    ax = next(r for r in rows if r[0] == w["anchor_xst"]["run_id"])
    assert (ax[4], ax[5], ax[6], ax[7]) == (pytest.approx(1 / 9), 1, 9, "wilson")
    # one owner row and one companion row per (run, judgment): the paper view's per-metric-id rank leaves the owner's `over_refusal_graded` where it was
    assert conn.execute(f"SELECT COUNT(*) FROM run_metric WHERE metric_id = '{ORC.OWNER_METRIC_ID}'").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    # the status row under the source string, ok, with the facts; the refusal durable in source_state
    st = conn.execute("SELECT lane_source_legacy, adapter, source_id, owner, ok, extra_json FROM source_status WHERE source_id = ?", (f"orxs_seedrep:{ORC.GROUP}",)).fetchall()
    assert len(st) == 1 and st[0][:5] == ("orxs_seedrep", "orxs_seedrep", f"orxs_seedrep:{ORC.GROUP}", "OR", 1)
    facts = json.loads(st[0][5])
    assert facts["runs_filed"] == 3 and facts["keys_filed"] == 5 and len(facts["refused"]) == 1 and facts["record_counts"] == {"run_metric": 3}
    assert facts["filed"][w["paired_xst"]["run_id"]]["value"] == pytest.approx(5.5 / 12) and facts["sha256"] == _rc_sha(os.path.join(d, ORC.SUMMARY_NAME))
    assert facts["source_version"]["file_version"] == 1 and facts["inputs_retired"] == []
    members = conn.execute("SELECT json_path, current_members_json FROM source_state WHERE source_id = ? ORDER BY json_path", (f"orxs_seedrep:{ORC.GROUP}",)).fetchall()
    assert [m[0] for m in members] == ["$.cells", "$.row_refusals"] and len(json.loads(members[0][1])) == 6 and "matches 2 current runs" in json.loads(members[1][1])[0]
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = 'orxs_seedrep'").fetchall()] == [("5",)]   # 5: the anchor currency
    conn.close()
    # a second pass over unchanged inputs republishes nothing (the revision id is the inputs' shas + the version)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep2 = RUN.run_pass(root, [_rc_adapter(d)], conn=conn, now="2026-09-22T06:00:00.000Z")
    conn.close()
    assert rep2.groups_refreshed == 0, [r.error for r in rep2.results if r.error]
    assert _rc_owner_rows(root) == before


def test_role_collapse_newest_version_is_read_and_the_older_ones_retire_without_tripping_the_drop_guard(tmp_path):
    """`.v2` ships beside the v1 pair: the v2 files are the inputs, the v1 pair is declared retired, and the runner's inputs-dropped
    guard (2 of the previous revision's 3 inputs absent = 67 % > 10 %) reads the move as deliberate and publishes the v2 revision."""
    w = _rc_world(tmp_path)
    root = w["root"]
    entries = _rc_entries(w)
    d = _rc_write(tmp_path, entries)                                   # v1
    ad = _rc_adapter(d)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep1 = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep1.groups_refreshed == 1 and rep1.groups_refused == 0, [r.error for r in rep1.results if r.error]
    ing1 = H.ingest_once(root, workers=0, min_free=0)
    assert ing1.refused == 0, ing1.refusals
    v1_rows = _rc_q(root, f"SELECT run_id, value FROM run_metric WHERE metric_id = '{ORC.METRIC_ID}' ORDER BY run_id")
    assert len(v1_rows) == 3
    # v2 beside v1: the anchor's labels differ (k 2 of 10) so the v2 row carries another value
    entries2 = _rc_entries(w, anchor_xst={0: _rc_counts(10, 10, 1, 1, 2, 0, 2, 2)})
    d2 = _rc_write(tmp_path, entries2, version=2)
    assert d2 == d and sorted(os.listdir(d)) == ["role_collapse_v1.py", "role_collapse_v1_items.jsonl", "role_collapse_v1_items.v2.jsonl",
                                                  "role_collapse_v1_summary.json", "role_collapse_v1_summary.v2.json"]
    ad2 = _rc_adapter(d)
    inputs = ad2.discover()
    assert sorted(os.path.basename(i.uri) for i in inputs) == ["role_collapse_v1.py", "role_collapse_v1_items.v2.jsonl", "role_collapse_v1_summary.v2.json"]
    assert ad2.inputs_retired(ORC.GROUP) == sorted([os.path.join(d, "role_collapse_v1_items.jsonl"), os.path.join(d, "role_collapse_v1_summary.json")])
    assert AB.inputs_retired_of(ad2, ORC.GROUP) == tuple(ad2.inputs_retired(ORC.GROUP)) and AB.inputs_retired_of(ad2, "some/store") == ()
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep2 = RUN.run_pass(root, [ad2], conn=conn, now="2026-09-22T07:00:00.000Z")
    conn.close()
    assert rep2.groups_refreshed == 1 and rep2.groups_refused == 0 and getattr(rep2, "groups_inputs_dropped", 0) == 0, [r.error for r in rep2.results if r.error]
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0, ing2.refusals
    revs = _rc_q(root, "SELECT revision_id, accepted, superseded_by_revision_id, extra_json FROM source_revision WHERE source_id = ? ORDER BY observed_utc",
                 (f"orxs_seedrep:{ORC.GROUP}",))
    assert len(revs) == 2 and [r[1] for r in revs] == [1, 1] and revs[0][2] == revs[1][0] and revs[1][2] is None
    cmp = json.loads(revs[1][3]).get("inputs_dropped_comparison") or {k: v for k, v in json.loads(revs[1][3]).items() if k.startswith("inputs_")}
    assert cmp.get("inputs_retired") == 2 and cmp.get("inputs_dropped") == 0 and cmp.get("inputs_added") == 2, revs[1][3]
    v2_rows = _rc_q(root, f"SELECT run_id, value, json_extract(extra_json, '$.source_version.file_version'), json_extract(extra_json, '$.inputs_retired') "
                          f"FROM run_metric WHERE metric_id = '{ORC.METRIC_ID}' ORDER BY run_id")
    assert [r[0] for r in v2_rows] == [r[0] for r in v1_rows] and all(r[2] == 2 for r in v2_rows)
    assert all(sorted(json.loads(r[3])) == sorted(LC.rel_home(u) for u in ad2.inputs_retired(ORC.GROUP)) for r in v2_rows)
    ax1 = next(r[1] for r in v1_rows if r[0] == w["anchor_xst"]["run_id"])
    ax2 = next(r[1] for r in v2_rows if r[0] == w["anchor_xst"]["run_id"])
    assert (ax1, ax2) == (pytest.approx(1 / 9), pytest.approx(2 / 9))
    # the v1 rows left the frame with their revision: one current companion row per run
    cur = _rc_q(root, "SELECT COUNT(*) FROM observation WHERE table_name = 'run_metric' AND current = 1 AND source_id = ?", (f"orxs_seedrep:{ORC.GROUP}",))
    assert cur == [(3,)]
    assert _rc_owner_rows(root) == _rc_owner_rows(root)      # still there, untouched


#: the producer's files as read on 2026-09-22 (a changed file republishes under its new sha -- the pin is this test's, never the reader's)
RC_REAL_SUMMARY_SHA256 = "0fdc1cf49d0998f01623cc2de122ad700fdeb334eedb7ca89d1d2938f1d7db57"
RC_REAL_ITEMS_SHA256 = "c0406533e393e5ec20c31138aaf8bc8550d17fe64491c6c1c2e3ebabbbf22bea"
RC_REAL_COUNTS = {  # (model, control, trial) -> (k_full_refusal_non_collapsed, n_non_collapsed_judged, n_collapsed, n_items, final rung)
    ("llama8b_it", "xstest", 0): (16, 114, 136, 250, 512), ("llama8b_it", "xstest", 1): (20, 112, 138, 250, 512),
    ("llama8b_it", "or_bench_hard_1k", 0): (81, 372, 947, 1319, 2048), ("llama8b_it", "or_bench_hard_1k", 1): (74, 371, 947, 1319, 2048),
    ("gemma4e4b_it", "xstest", 0): (63, 242, 8, 250, 2048), ("gemma4e4b_it", "xstest", 1): (59, 243, 7, 250, 2048),
    ("gemma4e4b_it", "or_bench_hard_1k", 0): (816, 1299, 19, 1319, 2048), ("gemma4e4b_it", "or_bench_hard_1k", 1): (810, 1298, 19, 1319, 2048),
}


#: v2 (06:41Z): anchor labels read from the sampled controls' pass-2 flip copies, `k_partial_refusal_all_judged` and the graded rates beside
RC_REAL_V2_SUMMARY_SHA256 = "9108dc7ce1a6e2ea1e577b408c970ec4a1a506159fa9a19974dbabc82b778004"
RC_REAL_V2_ITEMS_SHA256 = "1ffa9020663c78d03b15de6ebf257e0fba59754e5744312b4403f4245451f522"
RC_REAL_V2_GRADED = {  # (model, store label, control, trial) -> (graded_over_non_collapsed, k_full_refusal_all_judged, n_judged, k_full_refusal_non_collapsed, n_non_collapsed_judged)
    ("llama8b_it", "difference_relu_paired", "xstest", 0): (0.15789473684210525, 59, 250, 16, 114), ("llama8b_it", "difference_relu_paired", "xstest", 1): (0.18303571428571427, 60, 250, 20, 112),
    ("llama8b_it", "difference_relu_paired", "or_bench_hard_1k", 0): (0.25, 180, 1317, 81, 372), ("llama8b_it", "difference_relu_paired", "or_bench_hard_1k", 1): (0.23315363881401618, 160, 1315, 74, 371),
    ("gemma4e4b_it", "difference_relu_paired", "xstest", 0): (0.27479338842975204, 65, 250, 63, 242), ("gemma4e4b_it", "difference_relu_paired", "xstest", 1): (0.2716049382716049, 60, 250, 59, 243),
    ("gemma4e4b_it", "difference_relu_paired", "or_bench_hard_1k", 0): (0.7159353348729792, 823, 1318, 816, 1299), ("gemma4e4b_it", "difference_relu_paired", "or_bench_hard_1k", 1): (0.7230354391371341, 816, 1317, 810, 1298),
    ("gemma4e4b_it", "none_sampled_anchor", "xstest", 0): (0.10975609756097561, 26, 250, 25, 246), ("gemma4e4b_it", "none_sampled_anchor", "or_bench_hard_1k", 0): (0.6020092735703245, 613, 1318, 612, 1294),
    ("llama8b_it", "none_sampled_anchor", "xstest", 0): (0.05668016194331984, 14, 250, 14, 247), ("llama8b_it", "none_sampled_anchor", "or_bench_hard_1k", 0): (0.4515246286161063, 567, 1318, 566, 1279),
}


def test_role_collapse_real_handoff_v1_is_pinned_as_a_retired_input():
    """The v1 pair stays on the mirror as a RETIRED input beside v2: its shas hold, the reader declares it retired, and its counts (no partial count,
    no anchor labels) are the v1 shape this test recorded before v2 shipped."""
    summary = os.path.join(ORC.ROOT, ORC.SUMMARY_NAME)
    items = os.path.join(ORC.ROOT, ORC.ITEMS_NAME)
    if not (os.path.isfile(summary) and os.path.isfile(items)):
        pytest.skip("OR's v1 handoff is not on this machine")
    assert _rc_sha(summary) == RC_REAL_SUMMARY_SHA256 and _rc_sha(items) == RC_REAL_ITEMS_SHA256, "OR's v1 handoff changed: re-pin after reading it"
    if max(ORC.newest_versions(ORC.ROOT)[ORC.SUMMARY_STEM]) > 1:
        assert {summary, items} <= set(ORC.retired_inputs(ORC.ROOT))
    data = IC.read_json(summary)
    assert data["rule"] == ORC.RULE_ID and len(data["cells"]) == 12
    tallies = ORC.read_items(items)
    seen = {}
    for key, entry in data["cells"].items():
        c = entry["counts"]
        assert ORC.items_disagreement(dict(c, k_partial_refusal_all_judged=0), tallies.get(key)) is None, key
        model, label, trial, control = ORC.parse_key(key)
        if label == "difference_relu_paired":
            seen[(model, control, trial)] = (c["k_full_refusal_non_collapsed"], c["n_non_collapsed_judged"], c["n_collapsed"], c["n_items"], entry["final_rung"])
        else:
            assert label == "none_sampled_anchor" and entry["cell"] == "none" and c["n_judged"] == 0 and 0.01 <= entry["collapse_share"] <= 0.03
    assert seen == RC_REAL_COUNTS


def test_role_collapse_real_handoff_v2_sha_is_pinned_and_its_graded_counts_reproduce_ors_numbers():
    found = ORC.newest_versions(ORC.ROOT)
    if 2 not in found[ORC.SUMMARY_STEM] or 2 not in found[ORC.ITEMS_STEM]:
        pytest.skip("OR's v2 handoff is not on this machine")
    summary, items = found[ORC.SUMMARY_STEM][2], found[ORC.ITEMS_STEM][2]
    assert _rc_sha(summary) == RC_REAL_V2_SUMMARY_SHA256 and _rc_sha(items) == RC_REAL_V2_ITEMS_SHA256, "OR's v2 handoff changed: re-pin after reading it"
    if max(found[ORC.SUMMARY_STEM]) == 2:
        assert [os.path.basename(i.uri) for i in ORC.discover_inputs(ORC.ROOT)] == ["role_collapse_v1_summary.v2.json", "role_collapse_v1_items.v2.jsonl", "role_collapse_v1.py"]
    data = IC.read_json(summary)
    assert data["rule"] == ORC.RULE_ID and len(data["cells"]) == 12 and str(data["version"]).startswith("v2")
    tallies = ORC.read_items(items)
    seen = {}
    for key, entry in data["cells"].items():
        e = ORC.entry_facts(key, entry)
        assert ORC.items_disagreement(e["counts"], tallies.get(key)) is None, key
        c = e["counts"]
        assert c["n_judged"] > 0 and e["graded_over_non_collapsed"] is not None and e["graded_all_judged"] is not None
        seen[(e["model"], e["label"], e["control"], e["trial"])] = (e["graded_over_non_collapsed"], c["k_full_refusal_all_judged"], c["n_judged"],
                                                                    c["k_full_refusal_non_collapsed"], c["n_non_collapsed_judged"])
    assert {k: (pytest.approx(v[0]),) + v[1:] for k, v in RC_REAL_V2_GRADED.items()} == seen
    assert sorted(os.path.basename(u) for u in ORC.retired_inputs(ORC.ROOT)) == ["role_collapse_v1_items.jsonl", "role_collapse_v1_summary.json"]
