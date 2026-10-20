"""Qwen MMLU at the FULL 1,531-item width (source `mmlu_full`; `cellstore.MmluFullAdapter` 1 over `cellstore.MMLU_FULL_ROOTS`) on SYNTHETIC
stores shaped like the durable mirror (`<root>/<tag>_t0/cells/<store key>/{rows.jsonl, manifest.json, gen_stats.json, evidence.json}` and the
Arditi point's `cells` root; invented item ids and shas, `$SHARED/...` paths with a stand-in user, no generation text). What it pins:

* IDENTITY BY ATTESTATION, never by n (`cellstore.mmlu_item_set_attestation`): a store whose `resolved_config.mmlu.item_set` is the declared
  full spec files on the NEW instrument (`mmlu_letter_prefill_full_1531`, item set `mmlu_1531`) with 1,531 `acc` item rows, the producer's `acc`
  metric, the ids' sha256 checked against the declared item set, the head's sha16 parked; a 200-item store beside it (no `item_set` key) files
  on `mmlu_letter_prefill_200` byte for byte as a reader that knows nothing of the full width files it, and a full-width-only root refuses it
  by name; a foreign spec, a torn attestation (block / ids / gen_stats / rows disagreeing) or a declaration the store does not reproduce refuses
  by name;
* the target directory's GRAMMAR resolves the objects (`cellstore.mmlu_full_target_object`): the roster's `it` / `s<N>` / `s<N>p` selections over
  the REAL roster and pins (the manifest sha16 must prefix the pinned sha; a paired tag's LoRA objects are the seed tag's own), the Arditi point
  through the Arditi branch, anything else refused by name;
* the accuracy contrast against the same tag's `none` (paired per item over the items scored in both, McNemar's exact p, Cohen's h,
  `same_standup_none` when the two stores name one serving job);
* the end-to-end pass + ingest: every run `ingested` with its 1,531 items, `view_legacy` as the tags spell them, tag-prefixed producer keys,
  `lane_source_legacy` mmlu_full, run ids disjoint from the 200-item instrument's, the same-object rows across tags sharing the collapse key;
* the registration (`mmlu_full`, owner Summary), the seeds (item set + instrument, nested on the 200), and that the generic `cellstore` adapter's
  and OR's roots declare none of the new opt-ins (their versions stay 5 and 3; the byte-identity of their re-read was proved by a dry run).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import cellstore as CS
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import dimensions as D
from sisteer.registry.adapters import dose_grammar as DG
from sisteer.registry.adapters import runner as RUN

import _anchor_frame_fixtures as AF
import registry_items_fixtures as F

NOW = "2026-09-21T22:00:00.000Z"
N = int(CS.MMLU_FULL_SPEC["n"])
IDS: List[str] = [f"m{i:04d}-{hashlib.sha256(f'synthetic mmlu item {i}'.encode()).hexdigest()[:8]}" for i in range(N)]
IDS_SHA = CS.ids_sha256(IDS)
FULL_SEED, FULL_SET = CS.MMLU_FULL_INSTRUMENT_SEED, CS.MMLU_FULL_ITEM_SET_SEED
TREE = "c326f847e" + "a" * 31
FULL_INSTRUMENT = synth.instrument_fields(eval_id="mmlu", item_set_id=FULL_SET, harness_version="synthetic full-width letter read", scoring_rule="letter_logprob_prefill",
                                          answer_extraction="none", token_cap_tokens=1, token_cap_ladder_json="[1]", reply_cap_tokens=1, judge_id=None)
FULL_ID = str(FULL_INSTRUMENT["instrument_id"])
ARDITI = DG.OBJECT_KEY_LEGACY
AR_TAG = "qwen35_4b_ar_c0p3_t0"
ARDITI_OBJECT = hashlib.sha256(b"synthetic arditi all-positions direction at c = 0.3").hexdigest()
PROD = {"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}


def _dims() -> IC.Dims:
    """The fixtures' explicit Dims plus the full-width instrument and its item set (n + the synthetic ids' sha)."""
    d = F.dims_map()
    d.instruments[(FULL_SEED, "mmlu", None)] = (FULL_ID, FULL_SET, {"seed_instrument_id": FULL_SEED, "eval_ids": ["mmlu"], "item_set_ids": [FULL_SET],
                                                                  "decode_source": "greedy", "token_cap_tokens": 1, "judge_id": None, "token_cap_ladder_json": [1]})
    d.item_sets[FULL_SET] = (N, IDS_SHA)
    return d


def _id200() -> str:
    return F.dims_map().instruments[("mmlu_letter_prefill_200", "mmlu", None)][0]


def _ctx(conn=None):
    return AB.AdapterContext(root="/nonexistent", now=NOW, producer=PROD, conn=conn)


def _primed(ad) -> AB.AdapterContext:
    """A context on a frame PRIMED from the adapter's own emissions (every own `none` current): since this reader's 2 the accuracy contrast is
    decided by the pass's anchor currency, so an in-memory read describes the steady state after an ingest."""
    return AF.primed_ctx(ad, now=NOW, producer=PROD, root="/nonexistent")


def _validate_all(records) -> list:
    bad = []
    for rt, p in records:
        p = dict(p)
        if rt in R.ITEM_SCALE_RECORD_TYPES:
            p.setdefault("source_revision_id", "rev1"); p.setdefault("withdrawn_by_revision_id", None)
        rec = R.make_record(rt, p, PROD, I.adapter_record_id(p, rt), NOW)
        out = VAL.validate_record(rec, "mem")
        if not isinstance(out, VAL.Validated):
            bad.append((rt, out.field, out.reason))
    return bad


def _json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=0)


# --------------------------------------------------------------------------------------------
# the synthetic full-width store
# --------------------------------------------------------------------------------------------

def scores_none() -> List[float]:
    """The unsteered pattern: an item is correct unless its index is a multiple of 4 (1,148 of 1,531 correct)."""
    return [0.0 if i % 4 == 0 else 1.0 for i in range(N)]


def scores_steered(steered_only: int = 10, anchor_only: int = 20) -> List[float]:
    """The steered pattern: `steered_only` items the anchor missed are correct here (indices 0, 4, ... below 4 * steered_only), and `anchor_only`
    items the anchor had are missed here (indices 1, 5, ... below 4 * anchor_only): the discordant pairs are known by construction."""
    s = scores_none()
    for i in range(0, 4 * steered_only, 4):
        s[i] = 1.0
    for i in range(1, 4 * anchor_only, 4):
        s[i] = 0.0
    return s


def write_full_store(tag_dir: str, cell_id: str, *, scores: Sequence[float], ihash8: str, chash8: str, object_sha16: Optional[str] = None,
                     hf_model: str = F.HF_IT, fit_hf: str = F.HF_BASE, job: str = "593403", spec: Optional[Dict[str, object]] = None,
                     block_spec: Optional[Dict[str, object]] = None, ids: Optional[Sequence[str]] = None, gs_spec: Optional[Dict[str, object]] = None,
                     drop_row_ids: Sequence[str] = (), tree_sha: Optional[str] = TREE, omit_block: bool = False, omit_item_set_key: bool = False) -> str:
    """One synthetic full-width MMLU store in the served driver's layout; returns the store dir. The knobs tear the attestation apart for the
    refusal tests: `spec` (resolved_config.mmlu.item_set), `block_spec` / `ids` / `omit_block` (manifest.mmlu_item_set), `gs_spec`
    (gen_stats.mmlu.item_set), `drop_row_ids` (rows.jsonl), `omit_item_set_key` (a 200-shaped manifest carrying the full ids block)."""
    spec = dict(CS.MMLU_FULL_SPEC) if spec is None else dict(spec)
    ids = list(IDS) if ids is None else list(ids)
    key = F.store_key(cell_id, ihash8, chash8)
    d = os.path.join(tag_dir, "cells", key)
    os.makedirs(d, exist_ok=True)
    steered = cell_id != "none" and not cell_id.startswith("prompted__")
    ident = "proxy_vllm+lora+interleaved" if steered else "proxy_vllm+passthrough+interleaved"
    fitp = ({"model_id": fit_hf, "object_sha": object_sha16, "positions_at_fit": "all", "revision": "b" * 40, "status": "document_fit", "fit_kind": "document",
             "served_model_id": hf_model, "transfer": fit_hf != hf_model} if steered
            else {"model_id": None, "object_sha": None, "positions_at_fit": None, "served_model_id": hf_model, "transfer": False, "template_check": "baseline"})
    mcfg: Dict[str, object] = {"answer_seed": "Answer:", "response_prefill": False, "top_logprobs": 64}
    if not omit_item_set_key:
        mcfg[CS.MMLU_ITEM_SET_KEY] = spec
    const = {"arm_name": cell_id.replace("__", "--"), "attempt": 0, "coeff": 1.0, "engine": "vllm", "env_digest": "7f0b0097fbceed4c",
             "harness_config_hash": chash8 + "72d6408c", "intervention_hash": ihash8 + "091aef6f", "kind": "cell", "model_id": hf_model, "model_revision": "c" * 40,
             "run_id": f"served-controls-{cell_id.replace('__', '--')}", "seed": 0, "slurm_job_id": job, "sweep": "two_si_controls_served", "trial_id": 0}
    manifest: Dict[str, object] = {
        "cell_id": cell_id, "contract_version": 3, "controls": ["mmlu"], "engine": "vllm", "serving_path": "lora" if steered else "passthrough",
        "serving_engine": "lora_native", "serving_identity": ident, "virtual_model": f"petri__{cell_id}", "harness_cell_id": cell_id,
        "harness_config_hash": chash8 + "72d6408c", "intervention_hash": ihash8 + "091aef6f", "model_id": hf_model, "model_revision": "c" * 40,
        "steer_positions": "all", "fit_provenance": fitp,
        "routing_table": f"/workspace-vast/{F.TESTUSER}/exp/synthetic/serving/{os.path.basename(tag_dir)}_{job}/routing_table.json",
        "row_constants": const,
        "resolved_config": {"cell_id": cell_id, "controls": ["gsm8k", "mmlu", "or_bench_hard_1k", "xstest"], "driver": "served_capability_run_v2", "engine": "vllm",
                            "gen": {"batch_size": 8, "do_sample": False, "max_new_tokens": 512}, "judge_model": "claude-sonnet-5", "limit": None, "mmlu": mcfg,
                            "model_id": hf_model, "model_revision": "c" * 40, "order_seed": 0, "scale": 1.0,
                            "serving": {"serving_engine": "lora_native", "serving_identity": ident, "virtual_model": f"petri__{cell_id}"}},
    }
    if tree_sha is not None:
        manifest["producer"] = {"driver": "served_capability_run", "tree_sha": tree_sha}
    if not omit_block:
        manifest[CS.MMLU_ITEM_SET_BLOCK] = dict(block_spec if block_spec is not None else spec, item_ids=list(ids))
    _json(os.path.join(d, "manifest.json"), manifest)
    rows = []
    t0 = 1_790_022_939.0
    for i, (iid, v) in enumerate(zip(ids, scores)):
        if iid in drop_row_ids:
            continue
        rows.append(dict(const, task="s2parity/mmlu_letter", task_version="s2parity_v1", metric="acc", item_id=iid, value=float(v), position_order=i, domain="knowledge_s2parity",
                         tier=1, created_at=t0 + i * 0.01, wall_clock_s=None,
                         extra_json=json.dumps({"censored_letters": [], "gold": "A", "letter_read": "all_present", "n_censored": 0, "pick": "A" if v else "B", "top_logprobs": 64},
                                               sort_keys=True, separators=(",", ":"))))
    acc = sum(scores) / len(scores)
    rows.append(dict(const, task="s2parity/mmlu_letter", task_version="s2parity_v1", metric="acc_aggregate", item_id="__aggregate__", value=acc, position_order=None,
                     domain="knowledge_s2parity", tier=1, created_at=t0 + N * 0.01 + 1, wall_clock_s=24.8,
                     extra_json=json.dumps({"censored_items": 0, "censored_letter_counts": {}, "n": len(ids), "response_prefill": False, "top_logprobs": 64, "wall_clock_s": 24.8},
                                           sort_keys=True, separators=(",", ":"))))
    with open(os.path.join(d, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    gs: Dict[str, object] = {"acc": acc, "n": len(ids), "top_logprobs": 64, "response_prefill": False, "censored_items": 0, "censored_letter_counts": {}, "wall_clock_s": 24.8,
                             CS.MMLU_ITEM_SET_KEY: (dict(gs_spec) if gs_spec is not None else spec)}
    _json(os.path.join(d, "gen_stats.json"), {"mmlu": gs})
    _json(os.path.join(d, "evidence.json"), {"driver": "served_capability_run_v2", "cell_id": cell_id, "engine": "vllm", "serving_identity": ident,
                                             "engine_stamp": {"target_engine": "vllm", "serving_path": "lora" if steered else "passthrough", "object_sha": object_sha16},
                                             "fit_provenance": fitp, "serve_header": {"template_kwargs": {"enable_thinking": False}}, "written_at": "2026-09-21T20:35:27Z"})
    return d


def _read(store: str, dims: Optional[IC.Dims] = None, **over) -> CS.StoreRecords:
    """The full-width roots' reader configuration (mmlu_full_roots' opt-ins), over an explicit Dims (the default `sha_prefix` object resolution)."""
    kw = dict(run_key_scope="target_dir", source_by_control_legacy={"mmlu": CS.MMLU_FULL_SOURCE}, substrate_from_routing_table=True,
              require_mmlu_full=True, producer_tree_sha=True)
    kw.update(over)
    reader = CS.StoreReader(dims or _dims(), "full22", NOW, **kw)
    tag_dir = os.path.basename(os.path.dirname(os.path.dirname(store)))
    return reader.read(store, CS.store_inputs(store), view=tag_dir, view_legacy=CS.seed_tag_of(tag_dir))


# --------------------------------------------------------------------------------------------
# identity by attestation
# --------------------------------------------------------------------------------------------

def test_a_full_store_files_the_full_instrument_every_item_and_its_attestation(tmp_path):
    tag_dir = str(tmp_path / "mirror" / "qwen35_4b_it_t0")
    scores = scores_none()
    store = write_full_store(tag_dir, "none", scores=scores, ihash8="2ff9cdff", chash8="f24f745b")
    sr = _read(store)
    assert set(sr.controls) == {"mmlu"} and sr.object_id == "none"
    runs = [p for rt, p in sr.records if rt == "run"]
    assert len(runs) == 1
    run = runs[0]
    key = F.store_key("none", "2ff9cdff", "f24f745b")
    assert run["instrument_id"] == FULL_ID and run["item_set_id"] == FULL_SET and run["eval_id"] == "mmlu"
    assert run["run_id"] == I.run_id("full22", f"qwen35_4b_it_t0/{key}", "none@2ff9cdff", "mmlu", FULL_ID)
    assert run["run_id"] != I.run_id("full22", f"qwen35_4b_it_t0/{key}", "none@2ff9cdff", "mmlu", _id200())       # a new instrument: new ids
    assert run["producer_run_key_legacy"] == f"qwen35_4b_it_t0/{key}" and run["view_legacy"] == "qwen35_4b_it" and run["extra"]["view_legacy_dir"] == "qwen35_4b_it_t0"
    assert run["lane_source_legacy"] == "mmlu_full" and run["campaign"] == "full22" and run["lifecycle_state"] == "generated"
    assert run["n_items_planned"] == run["n_items_generated"] == N and run["n_items_judged"] is None and run["judge_id"] is None
    fc = run["field_confidence_json"]
    assert run["code_sha"] == TREE and fc["code_sha"]["confidence"] == "recorded" and run["harness_version"].endswith(f"@ {TREE}")
    assert run["substrate"] == "cluster_a" and run["substrate_source"] == "derived" and "routing_table" in fc["substrate"]["source"]
    assert run["token_cap_tokens"] == 1 and run["decode_source"] == "greedy" and run["thinking_mode"] == "off" and run["cluster_job_id"] == "593403"
    assert run["serving_path"] == "passthrough" and run["transfer_kind"] == "none" and run["trial_index"] == 0 and run["scoring_rule"] == "letter_logprob_prefill"
    assert fc["item_set_id"]["confidence"] == "recorded" and fc["instrument_id"]["confidence"] == "recorded" and fc["n_items_planned"]["confidence"] == "derived"
    att = run["extra"]["mmlu_item_set"]
    assert att["spec"] == CS.MMLU_FULL_SPEC and att["n_ids"] == N and att["ids_sha256"] == IDS_SHA and att["first_ids"] == IDS[:3]
    assert att["head_n"] == CS.MMLU_HEAD_N == 200 and att["head_sha16"] == CS.ids_sha256(IDS[:200])[:16] and att["producer"]["tree_sha"] == TREE
    assert "never pool" in att["nesting"] and att["attestation"] == CS.MMLU_ITEM_SET_ATTESTATION
    items = [p for rt, p in sr.records if rt == "item_result"]
    assert len(items) == N and {p["item_id"] for p in items} == set(IDS)
    assert all(p["metric_id"] == "acc" and p["judgment_id"] == "" and p["api_pool"] is None and p["run_id"] == run["run_id"] for p in items)
    by_id = {p["item_id"]: p["value"] for p in items}
    assert all(by_id[iid] == v for iid, v in zip(IDS, scores))
    flags = [p for rt, p in sr.records if rt == "item_flags"]
    rows_sha = AB.check_input(os.path.join(store, "rows.jsonl")).sha256
    assert len(flags) == N and all(p["generation_artifact_id"] == rows_sha and isinstance(p["generation_offset"], int) and p["at_token_cap"] is None for p in flags)
    metrics = [p for rt, p in sr.records if rt == "run_metric"]
    assert len(metrics) == 1 and metrics[0]["metric_id"] == "acc" and metrics[0]["value"] == pytest.approx(sum(scores) / N) and metrics[0]["n"] == N
    assert metrics[0]["computed_by"] == "producer" and metrics[0]["producer_metric_spelling"] == "acc_aggregate"
    deg = [p for rt, p in sr.records if rt == "run_degeneration"]
    assert len(deg) == len(IC.DECLARED_MEASURES) and {p["measurement_state"] for p in deg} == {"not_applicable"}
    assert not [p for rt, p in sr.records if rt in ("run_state", "judgment", "run_contrast")]
    assert sr.controls["mmlu"].scores == dict(zip(IDS, scores))                                              # kept for the accuracy contrast
    assert not _validate_all(sr.records)
    # a store with NO producer block on an opted-in root: code_sha stays `unverified` with the reason (no refusal, no guess)
    bare = write_full_store(str(tmp_path / "mirror2" / "qwen35_4b_it_t0"), "none", scores=scores, ihash8="2ff9cdff", chash8="f24f745b", tree_sha=None)
    run = next(p for rt, p in _read(bare).records if rt == "run")
    assert run["code_sha"] == "unverified" and run["field_confidence_json"]["code_sha"]["confidence"] == "unverified" and run["extra"]["mmlu_item_set"]["producer"] is None


def test_the_attestation_picks_the_instrument_even_without_the_opt_ins_and_a_fitted_store_files_its_dose(tmp_path):
    """The default reader (no opt-ins) files a full-width store on the full instrument too — the attestation decides, never the root — while the
    opt-in facts (source, tree sha, planned n, routing-table substrate) stay unfiled; a fitted object's run files the tag's dose 1.0 (the operating
    point) and the sha-prefix object, as every fitted run of the shared reader does."""
    tag_dir = str(tmp_path / "mirror" / "qwen35_4b_s1_t0")
    store = write_full_store(tag_dir, F.LORA_CELL, scores=scores_steered(), ihash8="cac79126", chash8="3fe9e1aa", object_sha16=F.OBJ16, job="593404")
    plain = CS.StoreReader(_dims(), "full22", NOW).read(store, CS.store_inputs(store), view="qwen35_4b_s1_t0")
    run = next(p for rt, p in plain.records if rt == "run")
    assert run["instrument_id"] == FULL_ID and run["item_set_id"] == FULL_SET and run["extra"]["mmlu_item_set"]["ids_sha256"] == IDS_SHA
    assert run["lane_source_legacy"] is None and run["code_sha"] == "unverified" and run["n_items_planned"] == N       # planned n is the attestation's, not the opt-in's
    assert run["producer_run_key_legacy"] == F.store_key(F.LORA_CELL, "cac79126", "3fe9e1aa") and run["view_legacy"] == "qwen35_4b_s1_t0"
    assert run["substrate_source"] == "unverified"          # no routing-table opt-in: a tmp store path names no root, and nothing is guessed
    full = _read(store)
    run = next(p for rt, p in full.records if rt == "run")
    assert run["object_id"] == F.OBJ_SHA and run["transfer_kind"] == "base_to_instruct" and run["serving_path"] == "lora_native" and run["applied_positions_mode"] == "all"
    assert (run["strength_nominal"], run["is_operating_point"], run["strength_unit"], run["strength_nominal_source"]) == (1.0, 1, "nominal_only", "derived")
    assert run["view_legacy"] == "qwen35_4b_s1" and run["substrate"] == "cluster_a" and run["lane_source_legacy"] == "mmlu_full" and run["cluster_job_id"] == "593404"
    assert not _validate_all(full.records)


def test_a_200_item_store_files_as_today_beside_a_full_one_and_a_full_width_only_root_refuses_it(tmp_path):
    """The 200-item shape (no `item_set` key under resolved_config.mmlu, no `mmlu_item_set` block) is the read every existing store gets: the
    200-item instrument, item set mmlu_200, nothing parked — byte for byte what a reader that knows NOTHING of the full width files (the new
    dims rows are never consulted). The full-width roots refuse it by name instead of filing it under their source."""
    tag_dir = str(tmp_path / "mirror" / "qwen35_4b_it_t0")
    store200 = F.write_controls_store(tag_dir, "none", ihash8="2ff9cdff", chash8="60fc2339", object_sha16=None, n=3)
    write_full_store(tag_dir, "none", scores=scores_none(), ihash8="2ff9cdff", chash8="f24f745b")
    knowing = CS.StoreReader(_dims(), "synthcampaign", NOW).read(store200, CS.store_inputs(store200), view="qwen35_4b_it_t0")
    unknowing = CS.StoreReader(F.dims_map(), "synthcampaign", NOW).read(store200, CS.store_inputs(store200), view="qwen35_4b_it_t0")
    assert knowing.records == unknowing.records
    mm = next(p for rt, p in knowing.records if rt == "run" and p["eval_id"] == "mmlu")
    assert mm["instrument_id"] == _id200() and mm["item_set_id"] == "mmlu_200" and "mmlu_item_set" not in mm["extra"]
    assert mm["n_items_planned"] is None and mm["code_sha"] == "unverified" and mm["lane_source_legacy"] is None and mm["n_items_generated"] == 3
    assert CS.mmlu_item_set_attestation(json.load(open(os.path.join(store200, "manifest.json"))), json.load(open(os.path.join(store200, "gen_stats.json"))), "x") is None
    with pytest.raises(AB.AdapterRefusal, match="FULL-width MMLU read only"):
        _read(store200)
    # the opted-in reader with `require_mmlu_full` off files the 200-item store as today too (the other opt-ins do not depend on it)
    sr = _read(store200, require_mmlu_full=False)
    mm = next(p for rt, p in sr.records if rt == "run" and p["eval_id"] == "mmlu")
    assert mm["instrument_id"] == _id200() and mm["lane_source_legacy"] == "mmlu_full" and mm["producer_run_key_legacy"].startswith("qwen35_4b_it_t0/")


@pytest.mark.parametrize("knobs, exc, match", [
    ({"spec": {"item_set": "full", "n": 1000, "seed": 0, "of": 1531}}, AB.AdapterRefusal, "not the one declared full width"),
    ({"spec": {"item_set": "subset", "n": 200, "seed": 0, "of": 1531}}, AB.AdapterRefusal, "not the one declared full width"),
    ({"spec": {"item_set": "full", "n": 1531, "seed": 1, "of": 1531}}, AB.AdapterRefusal, "not the one declared full width"),
    ({"omit_block": True}, AB.AdapterRefusal, "no `mmlu_item_set` block"),
    ({"block_spec": {"item_set": "full", "n": 1531, "seed": 1, "of": 1531}}, AB.AdapterRefusal, "two authorities disagree"),
    ({"ids": IDS[:1530]}, AB.AdapterRefusal, "lists 1530 ids"),
    ({"ids": IDS[:1530] + [IDS[0]]}, AB.AdapterRefusal, "1530 distinct"),
    ({"gs_spec": {"item_set": "subset", "n": 200, "seed": 0, "of": 1531}}, AB.AdapterRefusal, "gen_stats.mmlu.item_set"),
    ({"drop_row_ids": (IDS[5],)}, AB.AdapterRefusal, "torn store"),
    ({"omit_item_set_key": True}, AB.AdapterRefusal, "FULL-width MMLU read only"),      # a 200-shaped manifest carrying an ids block: no attestation
])
def test_a_foreign_spec_or_a_torn_attestation_refuses_by_name(tmp_path, knobs, exc, match):
    tag_dir = str(tmp_path / "mirror" / "qwen35_4b_it_t0")
    store = write_full_store(tag_dir, "none", scores=scores_none(), ihash8="2ff9cdff", chash8="f24f745b", **knobs)
    with pytest.raises(exc, match=match):
        _read(store)
    if not knobs.get("omit_item_set_key"):
        # the attestation refuses on the default reader too: a present spec is checked whatever the root
        with pytest.raises(exc, match=match):
            CS.StoreReader(_dims(), "full22", NOW).read(store, CS.store_inputs(store), view="qwen35_4b_it_t0")


def test_a_declaration_the_store_does_not_reproduce_or_an_undeclared_item_set_refuses(tmp_path):
    tag_dir = str(tmp_path / "mirror" / "qwen35_4b_it_t0")
    store = write_full_store(tag_dir, "none", scores=scores_none(), ihash8="2ff9cdff", chash8="f24f745b")
    d = _dims()
    d.item_sets[FULL_SET] = (N, "0" * 64)
    with pytest.raises(AB.AdapterRefusal, match="not the declared item set"):
        _read(store, dims=d)
    d = _dims()
    d.item_sets[FULL_SET] = (1530, IDS_SHA)
    with pytest.raises(AB.AdapterRefusal, match="not the declared item set"):
        _read(store, dims=d)
    d = _dims()
    del d.item_sets[FULL_SET]
    with pytest.raises(IC.Unresolved, match="no item_set row"):
        _read(store, dims=d)
    d = _dims()
    del d.instruments[(FULL_SEED, "mmlu", None)]
    with pytest.raises(IC.Unresolved, match="no instrument row"):
        _read(store, dims=d)
    # the live DB path of Dims.item_set: an empty registry names no row; a seeded one names the declared n + sha
    conn = DB.connect_memory()
    from sisteer.registry.migrations import apply_migrations
    apply_migrations(conn)
    with pytest.raises(IC.Unresolved, match="no item_set row"):
        IC.Dims(conn=conn).item_set(FULL_SET)


# --------------------------------------------------------------------------------------------
# the target directory's grammar (the real roster and pins)
# --------------------------------------------------------------------------------------------

class _Conn:
    """A connection answering the two object queries the resolvers make: `SELECT 1 FROM object WHERE object_id = ?` (the roster resolution's
    row check) and the sha16 `LIKE` query (the Arditi branch), over a fixed {object_id: (kind, staged id)} table."""
    def __init__(self, objects: Dict[str, tuple]):
        self.objects = objects
        self._params: tuple = ()

    def execute(self, sql, params):
        assert "FROM object" in sql
        self._params = tuple(params)
        return self

    def fetchone(self):
        return (1,) if self._params and self._params[0] in self.objects else None

    def fetchall(self):
        prefix = str(self._params[0])
        return [(oid, kind, staged) for oid, (kind, staged) in self.objects.items() if oid.startswith(prefix)]


def _roster_and_pins():
    import yaml
    with open(DC.repo_path("configs", "sdfbase_full22_roster.yaml"), "r", encoding="utf-8") as fh:
        roster = yaml.safe_load(fh)["targets"]["qwen35_4b"]["views"]
    with open(DC.repo_path("configs", "sdfbase_full22_pins.json"), "r", encoding="utf-8") as fh:
        pins = json.load(fh)["qwen35_4b"]
    return roster, pins


def test_the_grammar_resolves_every_roster_selection_and_the_arditi_point_over_the_real_roster_and_refuses_by_name():
    roster, pins = _roster_and_pins()
    cases = [("qwen35_4b_it_t0", "constv2-mix-full__psr_ce_all", "it"), ("qwen35_4b_it_t0", "constv2-mix-full__lora_r4_ce_minus_ctl", "it"),
             ("qwen35_4b_s1_t0", "constv2ctl2-full__psr_ce_all", "s1"), ("qwen35_4b_s1p_t0", "constv2-mix-full__psr_ce_all_minus_ctl", "s1p"),
             ("qwen35_4b_s2p_t0", "constv2-mix-full__lora_r4_ce", "s2p"), ("qwen35_4b_s3", "constv2-mix-full__lora_r4_ce_minus_ctl", "s3")]
    for tag, cell_id, sel in cases:
        staged = roster[sel]["cells"][cell_id]
        sha = pins[staged]["object_sha256"]
        oid, src = CS.mmlu_full_target_object(cell_id, tag, sha[:16], _Conn({sha: ("lora", staged)}))
        assert oid == sha and f"views.{sel}.cells" in src and staged in src and f"the roster selection {sel} of the target directory {tag!r}" in src
        with pytest.raises(AB.AdapterRefusal, match="no prefix"):
            CS.mmlu_full_target_object(cell_id, tag, "ffffffffffffffff", _Conn({sha: ("lora", staged)}))
        with pytest.raises(AB.AdapterRefusal, match="names no object_sha"):
            CS.mmlu_full_target_object(cell_id, tag, None, _Conn({sha: ("lora", staged)}))
        with pytest.raises(IC.Unresolved, match="no object row"):
            CS.mmlu_full_target_object(cell_id, tag, sha[:16], _Conn({}))
    # a paired tag's LoRA objects ARE the seed tag's own (one pinned sha), and the seed tag delegates exactly as the seed-replicate resolver does
    lora = "constv2-mix-full__lora_r4_ce"
    sha_s2 = pins[roster["s2"]["cells"][lora]]["object_sha256"]
    assert pins[roster["s2p"]["cells"][lora]]["object_sha256"] == sha_s2
    via_grammar = CS.mmlu_full_target_object(lora, "qwen35_4b_s2_t0", sha_s2[:16], None)
    via_seed = CS.seed_selection_object(lora, "qwen35_4b_s2_t0", sha_s2[:16], None)
    assert via_grammar[0] == via_seed[0] == sha_s2 and via_grammar[1].split(" (the roster selection")[0] == via_seed[1].split(" (seed selection")[0]
    assert "(seed selection s2 of the target directory" in via_seed[1]           # the seed-replicate wording is unchanged: OR's bytes do not move
    assert CS.mmlu_full_target_object(lora, "qwen35_4b_s2p_t0", sha_s2[:16], None)[0] == sha_s2
    # `none` and the prompted objects: the same object on every selection, no table needed
    for tag in ("qwen35_4b_it_t0", "qwen35_4b_s3_t0", "qwen35_4b_s3p_t0", AR_TAG):
        assert CS.mmlu_full_target_object("none", tag, None, None)[0] == "none"
    si = {CS.mmlu_full_target_object("prompted__safety_refusal", tag, None, None)[0] for tag in ("qwen35_4b_it_t0", "qwen35_4b_s1_t0", "qwen35_4b_s2_t0")}
    assert len(si) == 1 and next(iter(si)).startswith("si:")
    # the Arditi point: the Arditi branch (kind vector_all, the point's staged id), `none` and `arditi_allpos` only
    good = ("ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2", "vector_all", "ar_c0p3__arditi_allpos")
    oid, src = CS.mmlu_full_target_object(ARDITI, AR_TAG, "ba2212f0c2f9ffef", _Conn({good[0]: good[1:]}))
    assert oid == good[0] and "ar_c0p3__arditi_allpos" in src and "vector_all" in src
    with pytest.raises(AB.AdapterRefusal, match="serves `none` and `arditi_allpos` only"):
        CS.mmlu_full_target_object(lora, AR_TAG, sha_s2[:16], _Conn({}))
    with pytest.raises(AB.AdapterRefusal, match="not the all-positions"):
        CS.mmlu_full_target_object(ARDITI, AR_TAG, "ba2212f0c2f9ffef", _Conn({good[0]: ("lora", good[2])}))
    # a directory spelling neither grammar, another model key, or a store id the selection maps nothing to
    with pytest.raises(AB.AdapterRefusal, match="none of the roster's selections"):
        CS.mmlu_full_target_object("none", "qwen35_4b_halfdose_t0", None, None)
    with pytest.raises(AB.AdapterRefusal, match="none of the roster's selections"):
        CS.mmlu_full_target_object("none", "qwen35_4b_t0", None, None)
    with pytest.raises(AB.AdapterRefusal, match="declares its selections for 'qwen35_4b' only"):
        CS.mmlu_full_target_object("none", "gemma4e4b_s1_t0", None, None)
    with pytest.raises(IC.Unresolved):
        CS.mmlu_full_target_object("constv2ctl2-full__psr_ce_all", "qwen35_4b_it_t0", "0123456789abcdef", None)     # not a store of the `it` selection
    assert CS.CellStoreAdapter.object_resolvers["mmlu_full_grammar"] is CS.mmlu_full_target_object and "mmlu_full_grammar" in CS.OBJECT_RESOLUTIONS
    assert CS.CellStoreAdapter.object_resolvers["seed_selection"] is CS.seed_selection_object and CS.CellStoreAdapter.object_resolvers["target_grammar"] is CS.orxs_target_object


# --------------------------------------------------------------------------------------------
# the adapter: the accuracy contrast and the end-to-end pass
# --------------------------------------------------------------------------------------------

def _synthetic_resolver(cell_id_legacy: str, target_dir: str, object_sha16: Optional[str], conn) -> tuple:
    """A synthetic roster: the one fitted object on every selection, the Arditi point's own object, `none` and the prompt as themselves."""
    if cell_id_legacy == "none":
        return "none", "synthetic: the unsteered object key"
    if cell_id_legacy.startswith("prompted__"):
        return F.SI_OBJECT_ID, "synthetic prompt"
    if DG.parse_arditi_allpos_tag(target_dir) is not None:
        if cell_id_legacy != ARDITI:
            raise AB.AdapterRefusal(cell_id_legacy, f"the Arditi point serves `none` and `{ARDITI}` only")
        return ARDITI_OBJECT, "synthetic Arditi object"
    if not object_sha16 or not F.OBJ_SHA.startswith(object_sha16):
        raise AB.AdapterRefusal(cell_id_legacy, "the store's object_sha is no prefix of the synthetic object")
    return F.OBJ_SHA, f"synthetic roster {target_dir}"


class _Adapter(CS.MmluFullAdapter):
    object_resolvers = dict(CS.CellStoreAdapter.object_resolvers, mmlu_full_grammar=_synthetic_resolver)


def _adapter(mirror: str, arditi: str, dims: Optional[IC.Dims] = None) -> _Adapter:
    return _Adapter(roots=CS.mmlu_full_roots(mirror, arditi), dims=dims)


def _arditi_object_record() -> Dict[str, object]:
    """The synthetic all-positions object row (the shape the `arditi_allpos` reader mints), so the point's runs have their parent at ingest."""
    return {"object_id": ARDITI_OBJECT, "object_kind": "vector_all", "paper_label": "arditi_allpos", "campaign": synth.CAMPAIGN,
            "campaign_cell_id_legacy": "ar_c0p3__arditi_allpos", "object_role": "baseline_published", "fit_target_kind": "contrastive_direction",
            "fit_model_id": F.MODEL_IT, "fit_contrast_sets_json": {"unverified": "synthetic"}, "fit_thinking_render": "unverified",
            "fit_objective": "arditi_search", "fit_positions": "all", "fit_layers_json": [10], "fit_layer_index_convention": "zero_based_decoder_block",
            "fit_layer_count": 1, "fit_layer_selection_rule": "separability_search", "fit_seed_confidence": "unverified", "derivation": "none",
            "base_scale": 0.3, "artifact_sha256": ARDITI_OBJECT, "status": "approved",
            "field_confidence_json": {k: {"confidence": "unverified", "source": "synthetic", "reason": "no fit record beside the synthetic point"}
                                      for k in ("fit_contrast_sets_json", "fit_seed", "fit_seed_confidence", "fit_thinking_render")}}


def _mirror(tmp_path, tags=("qwen35_4b_it_t0", "qwen35_4b_s1p_t0"), none_job_by_tag: Optional[Dict[str, str]] = None):
    mirror = str(tmp_path / "mirror")
    arditi = str(tmp_path / "arditi" / AR_TAG / "mmlu_full")
    jobs = {"qwen35_4b_it_t0": "593403", "qwen35_4b_s1p_t0": "593406", "qwen35_4b_s1_t0": "593404"}
    for tag in tags:
        job = jobs[tag]
        write_full_store(os.path.join(mirror, tag), "none", scores=scores_none(), ihash8="2ff9cdff", chash8="f24f745b", job=(none_job_by_tag or {}).get(tag, job))
        write_full_store(os.path.join(mirror, tag), F.LORA_CELL, scores=scores_steered(), ihash8="cac79126", chash8="3fe9e1aa", object_sha16=F.OBJ16, job=job)
    write_full_store(arditi, "none", scores=scores_none(), ihash8="2ff9cdff", chash8="9a379809", job="593621")
    write_full_store(arditi, ARDITI, scores=scores_steered(3, 7), ihash8="92b66dce", chash8="013fbfba", object_sha16=ARDITI_OBJECT[:16], hf_model=F.HF_IT, fit_hf=F.HF_IT, job="593621")
    return mirror, arditi


def test_the_accuracy_contrast_pairs_the_steered_store_with_the_same_tags_none(tmp_path):
    mirror, arditi = _mirror(tmp_path, tags=("qwen35_4b_s1_t0",))
    ad = _adapter(mirror, arditi, dims=_dims())
    inputs = ad.discover()
    groups = sorted({os.path.dirname(i.uri) for i in inputs})
    assert len(groups) == 4
    steered = next(g for g in groups if "qwen35_4b_s1_t0" in g and F.LORA_CELL.replace("__", "--") in g)
    none_group = next(g for g in groups if "qwen35_4b_s1_t0" in g and "/none__" in g)
    ctx = _primed(ad)
    batch = next(ad.read(ctx, steered, [i for i in inputs if os.path.dirname(i.uri) == steered]))
    contrasts = [r["payload"] for r in batch.records if r["record_type"] == "run_contrast"]
    run = next(r["payload"] for r in batch.records if r["record_type"] == "run")
    assert len(contrasts) == 1
    c = contrasts[0]
    p_s, p_a = sum(scores_steered()) / N, sum(scores_none()) / N
    key_none = F.store_key("none", "2ff9cdff", "f24f745b")
    assert c["run_id"] == run["run_id"] and c["anchor_run_id"] == I.run_id("full22", f"qwen35_4b_s1_t0/{key_none}", "none@2ff9cdff", "mmlu", FULL_ID)
    assert c["metric_id"] == "acc" and c["anchor_kind"] == "same_standup_none" and c["judgment_id"] == "" and c["anchor_judgment_id"] == "" and c["n_pairs"] == N
    assert c["delta"] == pytest.approx(100.0 * (p_s - p_a)) and c["delta_method"] == "exact_mcnemar" and c["delta_p"] == pytest.approx(DC.mcnemar_exact(10, 20))
    assert c["effect_size"] == pytest.approx(DC.cohen_h(p_s, p_a)) and c["effect_kind"] == "cohens_h" and c["computed_by"] == "registry"
    ex = c["extra"]
    assert (ex["discordant_steered_only"], ex["discordant_anchor_only"]) == (10, 20) and ex["acc_steered_over_pairs"] == pytest.approx(p_s) and ex["acc_anchor_over_pairs"] == pytest.approx(p_a)
    assert ex["standup_job_ids"] == ["593404", "593404"] and ex["anchor_store"] == none_group and ex["unit"] == "pp" and ex["n_steered_items"] == ex["n_anchor_items"] == N
    # the anchor currency's decision rides the contrast (the shared reader's 8): the own pairing stands; the steered `acc` row states the lifted park
    assert (ex["anchor_selection"]["state"], ex["anchor_selection"]["own_reason"], ex["anchor_selection"]["own_anchor_run_id"]) == ("own_current", "current", c["anchor_run_id"])
    acc_row = next(r["payload"] for r in batch.records if r["record_type"] == "run_metric" and r["payload"]["metric_id"] == "acc" and r["payload"]["run_id"] == run["run_id"])
    assert acc_row["judgment_id"] == "" and acc_row["extra"]["delta_block"] is None
    # the unsteered store files no contrast; the Arditi point's steered store contrasts against its own `none` under the `cells` root
    batch = next(ad.read(ctx, none_group, [i for i in inputs if os.path.dirname(i.uri) == none_group]))
    assert not [r for r in batch.records if r["record_type"] == "run_contrast"]
    ar_group = next(g for g in groups if g.startswith(arditi) and "arditi_allpos__" in g)
    batch = next(ad.read(ctx, ar_group, [i for i in inputs if os.path.dirname(i.uri) == ar_group]))
    c = next(r["payload"] for r in batch.records if r["record_type"] == "run_contrast")
    run = next(r["payload"] for r in batch.records if r["record_type"] == "run")
    assert (c["extra"]["discordant_steered_only"], c["extra"]["discordant_anchor_only"]) == (3, 7) and c["anchor_kind"] == "same_standup_none"
    assert run["view_legacy"] == "qwen35_4b_ar_c0p3" and run["producer_run_key_legacy"].startswith(f"{AR_TAG}/") and run["object_id"] == ARDITI_OBJECT
    assert (run["strength_coefficient"], run["strength_unit"], run["strength_nominal"], run["is_operating_point"], run["operating_point_id_legacy"]) == (0.3, "raw_coefficient", 1.0, 1, "ar_c0p3__arditi_allpos")
    # a `none` served by another job is a `sibling_none` anchor, never a borrowed one
    mirror2, arditi2 = _mirror(tmp_path / "two", tags=("qwen35_4b_s1_t0",), none_job_by_tag={"qwen35_4b_s1_t0": "999999"})
    ad2 = _adapter(mirror2, arditi2, dims=_dims())
    inputs2 = ad2.discover()
    steered2 = next(os.path.dirname(i.uri) for i in inputs2 if "qwen35_4b_s1_t0" in i.uri and F.LORA_CELL.replace("__", "--") in i.uri)
    batch = next(ad2.read(_primed(ad2), steered2, [i for i in inputs2 if os.path.dirname(i.uri) == steered2]))
    assert [r["payload"]["anchor_kind"] for r in batch.records if r["record_type"] == "run_contrast"] == ["sibling_none"]
    # without a registry connection a steered store's read refuses by name (the runner turns it into a refusal of the group)
    with pytest.raises(AC.AnchorCurrencyUnavailable, match="no registry connection"):
        list(ad2.read(_ctx(), steered2, [i for i in inputs2 if os.path.dirname(i.uri) == steered2]))


def test_adapter_pass_and_ingest_land_every_run_ingested_with_its_items_under_the_tags_spellings(tmp_path):
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    t = synth.t
    extra_dims = [
        synth.rec("item_set", {"item_set_id": FULL_SET, "eval_id": "mmlu", "dataset_version": "synthetic/mmlu@full", "n_items": N, "items_sha256": IDS_SHA,
                               "source_uri": "synthetic://items/mmlu_1531.jsonl", "item_id_scheme": "synthetic m<index>-<hash>", "subsample_rule": "full",
                               "held_out_of_fit": 1, "extra": {"head_of": {"item_set_id": "mmlu_200", "n_head": 200}}}, t(900)),
        synth.rec("instrument", dict(FULL_INSTRUMENT, extra={"seed_instrument_id": FULL_SEED, "eval_ids": ["mmlu"], "item_set_ids": [FULL_SET]}), t(901)),
        synth.rec("object", _arditi_object_record(), t(902)),
    ]
    R.write_records(os.path.join(root, "records", "dims", "000002-mmlu-full.jsonl"), extra_dims)
    rep0 = H.ingest_once(root, workers=0, min_free=0)
    assert rep0.refused == 0 and rep0.blocked == 0, rep0.refusals
    mirror, arditi = _mirror(tmp_path)
    ad = _adapter(mirror, arditi)                     # dims from the registry (ctx.conn): the DB path of Dims.item_set / Dims.instrument
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 6, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, ing.refusals
    # the first pass parks every accuracy comparison (its own `none` is not folded yet: no provisional anchors); the pass after the
    # fold re-reads the three steered groups on their moved anchor-decision inputs and files the contrasts against the now-current `none`s
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0] == 0
    rep2 = RUN.run_pass(root, [ad], conn=conn, now="2026-09-21T22:30:00.000Z")
    conn.close()
    assert rep2.groups_refused == 0 and rep2.groups_refreshed == 3 and rep2.anchor_decisions_changed == 3, [r.error for r in rep2.results]
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert ing2.refused == 0 and ing2.blocked == 0, ing2.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    runs = conn.execute("SELECT run_id, view_legacy, producer_run_key_legacy, campaign_cell_id_legacy, lifecycle_state, lane_source_legacy, instrument_id, item_set_id, "
                        "object_id, n_items_generated, n_items_planned, judged_utc FROM run WHERE campaign = 'full22' ORDER BY view_legacy, campaign_cell_id_legacy").fetchall()
    assert len(runs) == 6 and {r[4] for r in runs} == {"ingested"} and {r[5] for r in runs} == {"mmlu_full"}
    assert {r[6] for r in runs} == {FULL_ID} and {r[7] for r in runs} == {FULL_SET} and {r[9] for r in runs} == {N} == {r[10] for r in runs}
    assert all(r[11] for r in runs)                                                          # the rule-scored completion stamp
    assert sorted({r[1] for r in runs}) == ["qwen35_4b_ar_c0p3", "qwen35_4b_it", "qwen35_4b_s1p"]
    assert all(r[2].split("/", 1)[0] in {"qwen35_4b_it_t0", "qwen35_4b_s1p_t0", AR_TAG} for r in runs)
    # every item row once in the CURRENT relation (`v_item_result_current`); the three steered groups re-read once (the second pass), so their item rows are on the record twice
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 9 * N == conn.execute("SELECT COUNT(*) FROM item_flags").fetchone()[0]
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0] == 6 * N and conn.execute("SELECT COUNT(*) FROM v_item_conflict").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE metric_id = 'acc'").fetchone()[0] == 6
    assert tuple(conn.execute("SELECT COUNT(*), SUM(anchor_kind = 'same_standup_none') FROM run_contrast").fetchone()) == (3, 3)
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0 and conn.execute("SELECT COUNT(*) FROM v_pending_judgment").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_in_flight").fetchone()[0] == 0 and conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    deg = dict(conn.execute("SELECT measurement_state, COUNT(*) FROM run_degeneration GROUP BY 1").fetchall())
    assert deg == {"not_applicable": 9 * len(IC.DECLARED_MEASURES)}          # the three steered groups' rows twice on the record (the second pass)
    assert conn.execute("SELECT COUNT(*) FROM v_run_degeneration_current").fetchone()[0] == 6 * len(IC.DECLARED_MEASURES)
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT priority FROM source_revision WHERE adapter = 'mmlu_full' AND accepted = 1").fetchall()] == [(IC.PRIORITY_GENERIC_READER,)]
    # the same-object rows across tags share the collapse key (model, object, eval, instrument, judge, strength) with distinct run ids: the
    # `none` on three tags, the one fitted object on two -- the coexist / include / supersede terms are filed elsewhere, never by the reader
    groups = conn.execute("SELECT object_id, COUNT(*), COUNT(DISTINCT run_id), group_concat(view_legacy, ' ') FROM run WHERE campaign = 'full22' "
                          "GROUP BY model_id, object_id, eval_id, instrument_id, judge_id, strength_nominal, strength_coefficient, operating_point_id_legacy "
                          "HAVING COUNT(*) > 1 ORDER BY 2 DESC").fetchall()
    assert [(g[0], g[1], g[2]) for g in groups] == [("none", 3, 3), (F.OBJ_SHA, 2, 2)]
    # nothing collides with the 200-item instrument's ids
    for r in runs:
        assert r[0] != I.run_id("full22", r[2], r[2].split("/", 1)[1].split("__controls__")[0] + "@" + r[2].split("__a0__")[1][:8], "mmlu", _id200())
    # the status rows name the source; the declared inputs are the four store files per group
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT lane_source_legacy, adapter, owner FROM source_status").fetchall()] == [("mmlu_full", "mmlu_full", "Summary")]
    conn.close()
    assert ad.status_sources("any") == (CS.MMLU_FULL_SOURCE,)


# --------------------------------------------------------------------------------------------
# registration, roots, seeds
# --------------------------------------------------------------------------------------------

def test_registration_owner_roots_and_the_generic_and_or_adapters_declare_none_of_the_new_opt_ins():
    from sisteer.registry import adapters as ADP
    reg = ADP.load_all()
    assert "mmlu_full" in reg
    ad = reg["mmlu_full"]()
    assert isinstance(ad, CS.MmluFullAdapter) and AB.owner_of(ad) == "Summary" and ad.covers_sources == ("mmlu_full",) and AB.status_sources_of(ad, "x") == ("mmlu_full",)
    # cellstore 6 / orxs_seedrep 4: the opted-in roots' label-mean row; the MMLU-full reader files no over-refusal and stays at 1
    assert CS.MmluFullAdapter.EMISSION_VERSION == 2 and CS.CellStoreAdapter.EMISSION_VERSION == 10 and CS.OrxsSeedrepAdapter.EMISSION_VERSION == 5   # the anchor currency
    assert RUN.adapter_priority(ad) == IC.PRIORITY_GENERIC_READER
    roots = CS.MMLU_FULL_ROOTS
    assert len(roots) == 2 and [r.layout for r in roots] == ["targets", "cells"] and [r.path for r in roots] == [CS.MMLU_FULL_ROOT, CS.MMLU_FULL_ARDITI_ROOT]
    assert CS.MMLU_FULL_ROOT.endswith(os.path.join("run-state", "full22", "mmlu_full"))
    assert CS.MMLU_FULL_ARDITI_ROOT.endswith(os.path.join("full22", "arditi_allpos", "controls", AR_TAG, "mmlu_full"))
    for r in roots:
        assert (r.campaign, r.run_key_scope, r.source_legacy, r.source_controls, r.object_resolution) == ("full22", "target_dir", "mmlu_full", ("mmlu",), "mmlu_full_grammar")
        assert r.substrate_from_routing_table and r.require_mmlu_full and r.accuracy_contrast and r.producer_tree_sha
        assert not (r.over_refusal_headline or r.cap_provenance or r.file_unjudged or r.at_cap_from_gen_stats or r.ladder_parent or r.final_cap_digest)
    assert roots[0].strip_trial_suffix and roots[0].served_tag is None and roots[0].view_spelling_legacy is None
    assert roots[1].served_tag == AR_TAG and roots[1].view_spelling_legacy == "qwen35_4b_ar_c0p3" and not roots[1].strip_trial_suffix
    assert CS.MMLU_FULL_TAGS == ("qwen35_4b_it", "qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3", "qwen35_4b_s1p", "qwen35_4b_s2p", "qwen35_4b_s3p")
    # the generic adapter's and OR's roots declare NONE of the new opt-ins and none lies under the new roots: their bytes do not move
    for r in CS.DEFAULT_ROOTS + CS.ORXS_SEEDREP_ROOTS:
        assert r.source_controls == CS.OR_CTLS and not r.require_mmlu_full and not r.accuracy_contrast and not r.producer_tree_sha
        assert r.object_resolution != "mmlu_full_grammar" and r.path not in {x.path for x in roots}
    assert CS.MMLU_FULL_SOURCE not in CS.CellStoreAdapter().covers_sources and CS.MMLU_FULL_SOURCE not in CS.OrxsSeedrepAdapter().covers_sources
    # the defaults of the new reader arguments are the pre-existing behaviour
    reader = CS.StoreReader(F.dims_map(), "synthcampaign", NOW)
    assert not reader.require_mmlu_full and not reader.producer_tree_sha and reader.source_by_control_legacy == {}
    assert CS.StoreRoot("c", "/p").source_controls == CS.OR_CTLS == ("xstest", "or_bench_hard_1k")


def test_the_seeds_declare_the_item_set_and_the_instrument_nested_on_the_200_item_head():
    s = D.load_seeds()
    row = next(r for r in s.item_sets if r["item_set_id"] == FULL_SET)
    assert row["eval_id"] == "mmlu" and row["n_items"] == N == 1531 and row["subsample_rule"] == "full" and row["held_out_of_fit"] == 1
    assert row["items_sha256"] == "5b1153eae340d07f94c48f6d00ef2ac386842d53db1b70f2bf5fa68453b0fe11"
    assert row["head_of"]["item_set_id"] == "mmlu_200" == IC.CONTROL_EVALS["mmlu"][1] and row["head_of"]["n_head"] == CS.MMLU_HEAD_N == 200
    assert row["head_of"]["head_ids_sha16"] == "373f1258bf4fcafc" and "never pool" in row["head_of"]["rule"]
    assert "cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe" in row["dataset_version"]
    head = next(r for r in s.item_sets if r["item_set_id"] == "mmlu_200")
    assert head["n_items"] == 200 and head["dataset_version"].split(" ")[0] == row["dataset_version"].split(" ")[0]        # one pinned corpus
    inst = next(r for r in s.instruments if r["instrument_id"] == FULL_SEED)
    assert inst["eval_ids"] == ["mmlu"] and inst["item_set_ids"] == [FULL_SET] and inst["token_cap_tokens"] == 1 and inst["reply_cap_tokens"] == 1
    assert inst["decode_source"] == "greedy" and inst["judge_id"] is None
    # the paper's wildcard row first, then the pipeline-test models' EXPLICIT canonical rows (qwen35_35b_a3b and gemma4_31b
    # follow the paper's canonical instrument, each with its own `<model>|mmlu|off|greedy` policy row in the seeds; the pin was
    # closed on the wildcard alone)
    # + the worked example's explicit row: the recipe runs MMLU for every key and registers a
    # front-door key under this instrument, and no wildcard reaches a `reference` key -- named here ONLY, never by the 200-item head,
    # so the demo key resolves one MMLU instrument (tests/test_registry_seeds_demo_instruments.py)
    assert inst["decode_policy_refs"] == ["*|mmlu|off|greedy", "qwen35_35b_a3b|mmlu|off|greedy", "gemma4_31b|mmlu|off|greedy", "qwen35_4b_demo|mmlu|off|greedy"]
    assert inst["scoring_rule"] == "letter_logprob_prefill" and inst["thinking_mode"] == "off" and inst["serving_path"] == "lora_native"
    head_inst = next(r for r in s.instruments if r["instrument_id"] == "mmlu_letter_prefill_200")
    for k in ("scoring_rule", "answer_extraction", "token_cap_tokens", "decode_source", "thinking_mode", "serving_path", "applied_positions_mode", "n_shot", "cot_prompting"):
        assert inst[k] == head_inst[k], k                                                     # the same read; the width is the item set
    # the same decode policy too: the 200-item head is the paper's read alone (the wildcard row), which the full read carries FIRST
    # and then widens by the test models' explicit rows
    assert head_inst["decode_policy_refs"] == ["*|mmlu|off|greedy"] == inst["decode_policy_refs"][:1]
    bundle = D.build_records(s, PROD, verify_si=False)
    full_ids, head_ids = bundle.instrument_ids[FULL_SEED], bundle.instrument_ids["mmlu_letter_prefill_200"]
    assert len(full_ids) == 1 and len(head_ids) == 1 and full_ids != head_ids
    iset = next(r for r in bundle.all_records() if r["record_type"] == "item_set" and r["payload"]["item_set_id"] == FULL_SET)
    assert iset["payload"]["items_sha256"] == row["items_sha256"] and iset["payload"]["extra"]["head_of"]["item_set_id"] == "mmlu_200"
    assert CS.MMLU_FULL_SPEC == {"item_set": "full", "n": row["n_items"], "seed": 0, "of": row["n_items"]}
    # the attestation's ids-hash rule is the seed's rule: newline-joined, in order, no trailing newline
    assert CS.ids_sha256(["a", "b"]) == hashlib.sha256(b"a\nb").hexdigest()
