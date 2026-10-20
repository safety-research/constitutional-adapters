"""Synthetic fixtures for the lineage adapter tests (invented model ids, hostname TESTHOST, tiny numeric shapes;
never a real prompt, transcript or generation). Cell ids, eval spellings and instrument SEED ids use the campaign's
public identifiers because the adapters' grammars key on them; every model / object / run identity is synthetic.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN
from sisteer.registry import synth

TESTHOST = "TESTHOST"
SHA_A = hashlib.sha256(b"synthetic lora object a").hexdigest()
SHA_B = hashlib.sha256(b"synthetic relu object b").hexdigest()
SHA_SUB = hashlib.sha256(b"synthetic subtraction object").hexdigest()
SI_SHA = hashlib.sha256(b"synthetic safety instruction text").hexdigest()
VIEW = "synthview"
VIEW_IT = "synthview_it"
MODEL_IT = "synthmodel_4b_it"
MODEL_BASE = "synthmodel_4b_base"
LORA = "constv2-mix-full__lora_r4_ce"
RELU = "constv2ctl2-full__psr_ce_all"
SUB = "constv2-mix-full__lora_r4_ce_minus_ctl"
PROMPTED = "prompted__safety_refusal"
PAIRED_WRITE = "constv2-mix-full__psr_ce_all_minus_ctl"      # the undistilled paired write: object_kind composed, paper label difference_relu
SHA_PAIRED = hashlib.sha256(b"synthetic paired write object").hexdigest()
#: the synthetic served model's vendor preset, as a store manifest records it and as the vendor-preset instrument seeds carry it
VENDOR_DECODE = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
VENDOR_DECODE_SEED = {f"decode_{k}": v for k, v in VENDOR_DECODE.items()}


def producer() -> Dict[str, Any]:
    return {"script": "tests.b1", "code_sha": synth.code_sha("a"), "session": None, "host": TESTHOST, "substrate": "grading_host"}


def _inst(seed_id: str, eval_ids: List[str], item_sets: List[str], cap: int, **over) -> Dict[str, Any]:
    d = {"instrument_id": seed_id, "eval_ids": eval_ids, "item_set_ids": item_sets, "harness": "bespoke_sisteer", "harness_version": "synthetic 0.1",
         "harness_config_hash": "unverified", "n_shot": None, "prompt_template_sha256": "unverified", "cot_prompting": "none", "scoring_rule": "judge_threshold",
         "answer_extraction": "judge", "token_cap_tokens": cap, "token_cap_ladder_json": [cap], "ladder_promotion_rule": "none", "read_at": "fixed",
         "decode_source": "greedy", "decode_seed_rule": "none", "thinking_mode": "off", "n_trials": 1, "serving_path": "lens",
         "applied_positions_mode": "unverified", "judge_id": "strongreject_v1_haiku45", "paper_status": "superseded"}
    d.update(over)
    return d


def synthetic_seeds() -> LC.Seeds:
    s = LC.Seeds(source_dir=None)
    s.models = {
        MODEL_IT: {"model_id": MODEL_IT, "family": "gemma4", "size_label": "4B", "checkpoint_kind": "instruct", "hf_repo": "synthetic/instruct-4b",
                   "hf_revision": "c" * 40, "chat_template_sha256": "unverified", "generation_config_present": 1, "generation_config_json": {"temperature": 1.0},
                   "thinking_channel": "available", "paper_status": "in_paper", "producer_spellings": [VIEW, VIEW_IT]},
        MODEL_BASE: {"model_id": MODEL_BASE, "family": "gemma4", "size_label": "4B", "checkpoint_kind": "base", "hf_repo": "synthetic/base-4b",
                     "hf_revision": "b" * 40, "chat_template_sha256": "unverified", "generation_config_present": 0, "thinking_channel": "not_available",
                     "paper_status": "in_paper"},
        "glm46vflash": {"model_id": "glm46vflash", "family": "glm46v", "size_label": "flash", "checkpoint_kind": "chat", "hf_repo": "synthetic/glm",
                        "hf_revision": "d" * 40, "chat_template_sha256": "unverified", "generation_config_present": 1, "thinking_channel": "available",
                        "paper_status": "descoped", "paper_status_ruling_id": "synthetic ruling: descoped"},
        "synthattacker": {"model_id": "synthattacker", "family": "qwen35", "size_label": "27B", "checkpoint_kind": "instruct", "hf_repo": "synthetic/attacker",
                          "hf_revision": "e" * 40, "chat_template_sha256": "unverified", "generation_config_present": 1, "thinking_channel": "available",
                          "paper_status": "reference"},
    }
    evals = [("pasttense", "static", ["pasttense"]), ("codeattack", "static", ["codeattack"]), ("crescendo", "multi_turn", ["crescendo"]),
             ("pair", "multi_turn", ["pair"]), ("actorattack", "multi_turn", ["actorattack"]), ("strongreject_bon", "bon", ["bon"]),
             ("xstest", "over_refusal", ["xstest"]), ("or_bench_hard", "over_refusal", ["or_bench_hard_1k"]), ("mmlu", "capability", ["mmlu"]),
             ("gsm8k", "capability", ["gsm8k_controls512"]), ("petri_pooled", "petri", ["petri_pooled"])]
    for b in ("cyber_bio_chem_capabilities", "defer_to_users", "delusion_sycophancy", "instruction_evasion", "reasoning_unfaithfulness", "self_preservation"):
        evals.append((f"petri_{b}", "petri", [b]))
    s.evals = {}
    s.item_sets = {}
    for eid, cat, spell in evals:
        iset = f"{eid}_items"
        # the over-refusal headline is the plain refused rate (`xst_refused` / `orb_refused`); over_refusal_graded stays a secondary row
        s.evals[eid] = {"eval_id": eid, "eval_category": cat, "headline_metric_id": "asr" if cat in ("static", "multi_turn", "bon") else "petri_score" if cat == "petri" else ("xst_refused" if eid == "xstest" else "orb_refused") if cat == "over_refusal" else "acc",
                        "metric_direction": "lower_is_better", "default_item_set_id": iset, "attacker_required": 1 if cat in ("multi_turn", "petri") else 0,
                        "producer_spellings": spell, "paper_status": "in_paper"}
        s.item_sets[iset] = {"item_set_id": iset, "eval_id": eid, "dataset_version": "synthetic@v1", "n_items": 10, "items_sha256": "unverified",
                             "source_uri": "synthetic://items", "item_id_scheme": "q<i>", "subsample_rule": "full"}
    s.judges = {
        "strongreject_v1_haiku45": {"judge_id": "strongreject_v1_haiku45", "judge_model": "judge-model-x", "judge_role": "judge", "rubric_name": "strongreject",
                                    "rubric_version": "v1", "rubric_sha256": "unverified", "protocol_sha256": "unverified", "score_scale": "[0,1]",
                                    "threshold_rule": "strongreject > 0.5", "judge_backend": ["batch", "live"]},
        "petri_judge_opus5": {"judge_id": "petri_judge_opus5", "judge_model": "judge-model-y", "judge_role": "judge", "rubric_name": "petri_bloom_judge",
                              "rubric_version": "1", "rubric_sha256": "unverified", "protocol_sha256": "unverified", "score_scale": "1-10", "threshold_rule": None,
                              "judge_backend": "live"},
        "xstest_cpi_v1_sonnet5": {"judge_id": "xstest_cpi_v1_sonnet5", "judge_model": "judge-model-z", "judge_role": "judge", "rubric_name": "xstest_cpi",
                                  "rubric_version": "v1", "rubric_sha256": "unverified", "protocol_sha256": "unverified", "score_scale": "label", "threshold_rule": "cpi",
                                  "judge_backend": "live"},
        "or_bench_cpi_v1_sonnet5": {"judge_id": "or_bench_cpi_v1_sonnet5", "judge_model": "judge-model-z", "judge_role": "judge", "rubric_name": "or_bench_cpi",
                                    "rubric_version": "v1", "rubric_sha256": "unverified", "protocol_sha256": "unverified", "score_scale": "label",
                                    "threshold_rule": "cpi", "judge_backend": "live"},
    }
    static = ["pasttense", "codeattack"]
    s.instruments = {i["instrument_id"]: i for i in [
        _inst("static_greedy_4096_full22", static, [f"{e}_items" for e in static], 4096),
        _inst("static_sampled_vendor_preset_4096_8192", static, [f"{e}_items" for e in static], 8192, decode_source="vendor_preset_per_model",
              decode_seed_rule="per_item", read_at="final_rung", serving_path="lora_native", applied_positions_mode="all", n_trials=2, paper_status="canonical"),
        _inst("multiturn_greedy_512_full22", ["crescendo", "pair", "actorattack"], ["crescendo_items", "pair_items", "actorattack_items"], 512, harness="pyrit",
              attacker_model_id="synthattacker", max_turns=10),
        _inst("multiturn_crescendo_cap4096_generation_config", ["crescendo"], ["crescendo_items"], 4096, harness="pyrit", decode_source="generation_config",
              serving_path="lora_native", applied_positions_mode="all", attacker_model_id="synthattacker", max_turns=10, paper_status="canonical"),
        _inst("multiturn_pair_cap4096_generation_config", ["pair"], ["pair_items"], 4096, harness="pyrit", decode_source="generation_config",
              serving_path="lora_native", applied_positions_mode="all", attacker_model_id="synthattacker", max_turns=3, paper_status="canonical"),
        _inst("multiturn_actorattack_cap4096_generation_config", ["actorattack"], ["actorattack_items"], 4096, harness="actorattack",
              decode_source="generation_config", serving_path="lora_native", applied_positions_mode="all", attacker_model_id="synthattacker", paper_status="canonical"),
        _inst("bon_512_only_T1", ["strongreject_bon"], ["strongreject_bon_items"], 512, decode_source="grandfathered_T1", decode_seed_rule="per_draw", bon_n=75),
        # the climb row carries the grandfathered body's one number (temperature 1.0), as the loader's row does from the `|grandfathered_T1` policy
        _inst("bon_512store_4096climb_T1", ["strongreject_bon"], ["strongreject_bon_items"], 4096, decode_source="grandfathered_T1", decode_seed_rule="per_draw",
              bon_n=75, read_at="final_rung", serving_path="lora_native", applied_positions_mode="all", paper_status="canonical", decode_temperature=1.0),
        # the vendor-preset best-of-N family (the paired-write stores): the same shape, the decode numbers on the row (no policy expansion here)
        _inst("bon_512_only_vendor_preset", ["strongreject_bon"], ["strongreject_bon_items"], 512, decode_source="vendor_preset", decode_seed_rule="per_draw", bon_n=75,
              serving_path="lora_native", **VENDOR_DECODE_SEED),
        _inst("bon_512store_4096climb_vendor_preset", ["strongreject_bon"], ["strongreject_bon_items"], 4096, decode_source="vendor_preset", decode_seed_rule="per_draw",
              bon_n=75, read_at="final_rung", serving_path="lora_native", applied_positions_mode="all", paper_status="canonical", **VENDOR_DECODE_SEED),
        # the NATIVE 4,096 vendor-preset best-of-N (no 512 store, no climb): adapters/bon_native.py's instrument (2026-09-22)
        _inst("bon_native4096_vendor_preset", ["strongreject_bon"], ["strongreject_bon_items"], 4096, decode_source="vendor_preset", decode_seed_rule="per_draw",
              bon_n=75, read_at="fixed", serving_path="lora_native", applied_positions_mode="all", paper_status="sensitivity",
              ladder_promotion_rule="none: a NATIVE best-of-75 pass at a fixed 4,096 cap on every draw (no 512 store, no climb)", **VENDOR_DECODE_SEED),
        _inst("petri_full22_as_run", ["petri_pooled"] + [f"petri_{b}" for b in ("cyber_bio_chem_capabilities", "defer_to_users", "delusion_sycophancy",
                                                                                  "instruction_evasion", "reasoning_unfaithfulness", "self_preservation")],
              ["petri_pooled_items"] + [f"petri_{b}_items" for b in ("cyber_bio_chem_capabilities", "defer_to_users", "delusion_sycophancy", "instruction_evasion",
                                                                     "reasoning_unfaithfulness", "self_preservation")],
              1024, harness="petri", scoring_rule="judge_graded_weights", decode_source="generation_config", serving_path="lora_native",
              applied_positions_mode="all", judge_id="petri_judge_opus5", max_turns=15, paper_status="canonical"),
        _inst("overrefusal_greedy_512_controls", ["xstest", "or_bench_hard"], ["xstest_items", "or_bench_hard_items"], 512, scoring_rule="judge_graded_weights",
              judge_id="xstest_cpi_v1_sonnet5"),
        # the sampled over-refusal re-run's ladder instruments (the export's `_sampled4096` over-refusal rows select them): two seeded
        # trials read at the pair's final ladder step and pooled per item
        _inst("overrefusal_xstest_sampled_vendor_preset_ladder", ["xstest"], ["xstest_items"], 2048, scoring_rule="judge_graded_weights",
              judge_id="xstest_cpi_v1_sonnet5", decode_source="vendor_preset_per_model", decode_seed_rule="per_item", read_at="final_rung",
              serving_path="lora_native", applied_positions_mode="all", n_trials=2, paper_status="canonical"),
        _inst("overrefusal_orbench_sampled_vendor_preset_ladder", ["or_bench_hard"], ["or_bench_hard_items"], 2048, scoring_rule="judge_graded_weights",
              judge_id="or_bench_cpi_v1_sonnet5", decode_source="vendor_preset_per_model", decode_seed_rule="per_item", read_at="final_rung",
              serving_path="lora_native", applied_positions_mode="all", n_trials=2, paper_status="canonical"),
        _inst("mmlu_letter_prefill_200", ["mmlu"], ["mmlu_items"], 1, scoring_rule="letter_logprob_prefill", answer_extraction="none", judge_id=None),
        _inst("gsm8k_0shot_greedy_512_controls", ["gsm8k"], ["gsm8k_items"], 512, scoring_rule="numeric_match", answer_extraction="last_number", judge_id=None),
    ]}
    s.si_texts = {"safety_refusal": {"si_id": "safety_refusal", "si_kind": "safety_instruction", "si_text_sha256": SI_SHA, "source_uri": "synthetic://si"}}
    s.legacy = {
        "model_keys": [{"legacy": VIEW, "model_id": MODEL_IT, "fit_model_id": MODEL_BASE, "transfer_kind": "base_to_instruct"},
                       {"legacy": VIEW_IT, "model_id": MODEL_IT, "fit_model_id": MODEL_IT, "transfer_kind": "instruct_direct"},
                       {"legacy": "glm46vflash | glm", "model_id": "glm46vflash"}],
        "judge_strings": [{"legacy": "judge-model-x / strongreject_v1 | strongreject_v1", "judge_id": "strongreject_v1_haiku45"},
                          {"legacy": "judge judge-model-y (Petri)", "judge_id": "petri_judge_opus5"},
                          {"legacy": "judge-model-z (over-refusal judge)", "judge_id": "xstest_cpi_v1_sonnet5"}],
    }
    return s


def write_view_manifest(objects_root: str, view_key: str = "main", cells: Optional[Dict[str, str]] = None) -> str:
    d = os.path.join(objects_root, VIEW, "cells_view", view_key)
    os.makedirs(d, exist_ok=True)
    cells = cells or {LORA: SHA_A, RELU: SHA_B, SUB: SHA_SUB}
    vm = {"wave": "synthetic", "target": VIEW, "view": view_key, "bank": objects_root,
          "entries": [{"cell_id": c, "staged_id": f"base__{c}", "object_sha256": sha, "identity": "staged", "mode": "symlink"} for c, sha in cells.items()],
          "written_at": "2026-01-01T00:00:00Z"}
    p = os.path.join(d, "VIEW_MANIFEST.json")
    with open(p, "w") as fh:
        json.dump(vm, fh)
    return p


def resolver(seeds: LC.Seeds, tmp_path) -> LC.ObjectResolver:
    objects_root = str(tmp_path / "objects")
    write_view_manifest(objects_root, "main")
    write_view_manifest(objects_root, "it")
    write_view_manifest(objects_root, "halfdose")
    return LC.ObjectResolver(seeds, objects_root=objects_root, petri_manifests_root=str(tmp_path / "arm_views"))


def registry_root(tmp_path, seeds: LC.Seeds, objects: Optional[List[LC.ResolvedObject]] = None) -> str:
    """A registry root with stand-in dimension rows (and stand-in object rows for the given resolved objects) ingested."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = LC.standin_dimension_records(seeds, producer(), now)
    for o in objects or []:
        r = LC.standin_object_record(o, seeds, producer(), now, MODEL_BASE)
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def standin_sampled_trial_runs(root: str, seeds: LC.Seeds, obj: LC.ResolvedObject, trials, cell: str, eval_id: str, rung: int = 8192) -> List[str]:
    """Stand-in `run` rows for the sampled re-run's per-trial runs (the `sampled_static` adapter owns the real ones;
    the union adapter attaches its per-trial readings to them by the same id derivation). Tests only."""
    now = R.utc_now()
    inst_seed = "static_sampled_vendor_preset_4096_8192"
    inst = seeds.instrument_row(inst_seed, eval_id)
    recs, ids = [], []
    for t in trials:
        gd = f"sdfbase_full22_{VIEW}_sampled_t{t}_cap{rung}"
        rid = I.run_id(LC.CAMPAIGN, gd, cell, eval_id, str(inst["instrument_id"]))
        p = LC.run_payload_base(seeds, run_id=rid, producer_key=gd, object_key=cell, view_legacy=VIEW, cell_id_legacy=cell, obj=obj, eval_id=eval_id,
                                instrument_row=inst, instrument_seed_id=inst_seed, lifecycle_state="generated", when_utc=now,
                                when_source="stand-in (test)", token_cap_tokens=rung, judge_id=None, judge_seed_id=None,
                                substrate=LC.substrate_from_grades_root(None, root), lane_source_legacy="standin", store_uri=None, extra={"trial": t})
        recs.append(R.make_record("run", p, producer(), now))
        ids.append(rid)
    # a NEW record file per call (the next free number in the stand-ins directory): an ingested record file is immutable, so a
    # second batch of stand-in runs for another cell or trial set is never appended to the first file the harvester folded
    standins = os.path.join(root, "records", "standins")
    heads = [int(f.split("-", 1)[0]) for f in (os.listdir(standins) if os.path.isdir(standins) else []) if f.split("-", 1)[0].isdigit()]
    R.write_records(os.path.join(standins, f"{max(heads + [1]) + 1:06d}-sampled-runs.jsonl"), recs, exclusive=True)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return ids


def run_and_ingest(root: str, adapter, now: Optional[str] = None):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = RUN.run_pass(root, [adapter], conn=conn, now=now or R.utc_now(), producer=producer())
    finally:
        conn.close()
    ing = H.ingest_once(root, workers=0, min_free=0)
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    return rep, ing, ing2


def sha_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def q(root: str, sql: str, params=()):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# a synthetic r2 export (parquet + companions + the two Petri producers) under tmp_path
# --------------------------------------------------------------------------------------------

def _src_entry(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    from datetime import datetime, timezone
    return {"mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"), "size": st.st_size, "sha256": sha_file(path)}


def synthetic_r2_export(tmp_path, petri_stats_key: str = "aaaa000000000001", petri_agg_key: str = "aaaa000000000001",
                        extra_column: Optional[str] = None, missing_source: bool = False, degeneration_rows: bool = False,
                        petri_degeneration_first: bool = False, moved_source: bool = False, overrefusal_count_rows: bool = False,
                        halfdose_rows: bool = False, overrefusal_contrast_rows: bool = False) -> Dict[str, str]:
    """Writes data/, rs/ (stats files) and consolidated/ (petri_agg.json); returns their paths. ``moved_source`` adds a builder-pinned
    file whose bytes are rewritten AFTER the pin is taken, and one best-of-N climb row naming it (the live 2026-09-20 shape).
    ``overrefusal_count_rows`` adds the three over-refusal shapes whose judged count the reader had left NULL (the live 2026-09-22 audit:
    59 per over-refusal eval) plus a per-behaviour Petri row: a sampled re-run's pooled rows for the lora cell (`over_refusal_rate_full*`
    over two seeded trials, the item count in the pinned `final_rung.json`), a controls store's at-cap decomposition rows for the relu
    cell under the store's own producer key (`at_cap_share_all` = the replies judged), and a store whose only judged-metric row is
    still pending beside a measured at-cap statistic (no count anywhere). ``halfdose_rows`` adds the lora object's HALF-DOSE reading (the
    `<tag>_halfdose` standup, dose 0.5) at past tense: its producer delta against an unsteered row the export does NOT carry under that standup,
    and its `delta_vs_full_dose*` against the same object's 1x reading (the main standup's lora row) -- the export's second contrast kind.
    ``overrefusal_contrast_rows`` gives the lora `over_refusal_rate` row (xstest, cap 512) the producer's paired delta and adds this standup's
    unsteered xstest row at the same cap -- the export's own anchor of that comparison AND of its refused-rate twin's (`xst_refused`), which
    r2_export 12 decides as a comparison of its own at the twin's id."""
    import pandas as pd
    from sisteer.registry.adapters.r2_export import COLUMN_MAP
    data = tmp_path / "data"
    rs = tmp_path / "rs"
    cons = tmp_path / "consolidated"
    for d in (data, rs / "petri_stats", cons):
        os.makedirs(d, exist_ok=True)
    ts_path = rs / "transfer_stats_full22_all.json"
    with open(ts_path, "w") as fh:
        json.dump({"models": {VIEW: {"cells": {}}}, "meta": {}}, fh)
    stats_path = rs / "petri_stats" / "petri_full22_stats.json"
    with open(stats_path, "w") as fh:
        json.dump({"meta": {}, "tags": {VIEW: {"cells": {
            "none": {"petri_run_key": "aaaa0000000000ff", "intervention_hash": "ffff0000", "behaviors": {"defer_to_users": {"per_scenario": {"s1": 5.0, "s2": 6.0}}}},
            LORA: {"petri_run_key": petri_stats_key, "intervention_hash": "1111000011110000", "behaviors": {"defer_to_users": {"per_scenario": {"s1": 4.0, "s2": 5.0}}}}}}}}, fh)
    agg_path = cons / "petri_agg.json"
    with open(agg_path, "w") as fh:
        json.dump({VIEW: {"cells": {"none": {"run_key": "aaaa0000000000ff", "complete": True}, "si_constv2_mix_full__lora_r4_ce": {"run_key": petri_agg_key, "complete": True}}}}, fh)
    prov_ts = f"{ts_path} ›models.{VIEW}.cells"
    nan = float("nan")
    base = {c: nan for c in COLUMN_MAP}
    rows = []

    def row(**kw):
        r = dict(base)
        r.update(model=VIEW.replace("_halfdose", ""), model_family="Synth", served_model="synthetic/instruct-4b", lane_kind="base→instruct", lane_label="Synth",
                 view=VIEW, dose=1.0, method="m", method_family="constitution", arm="lora", in_artifact_roster=True, relayed=False, judge="judge-model-x / strongreject_v1",
                 status="graded", note="synthetic", provenance=f"{prov_ts}.{kw.get('cell_id', 'none')}.{kw.get('eval_id', 'pasttense')}")
        r.update(kw)
        rows.append(r)

    # static: none + lora at 4096 (asr 0.5 with delta; 0.25 threshold view; bon curve)
    row(cell_id="none", component="static", eval_id="pasttense", eval="past tense", metric="asr", threshold=0.5, value=0.5, k=5, n=10, ci_lo=0.2, ci_hi=0.8,
        ci_method="Wilson 95% (join)", decode_cap_tokens=4096, at_cap_share=0.1, loop_share=0.0, degenerate_share=0.1, defended_degenerate_k=1, defended_n=5,
        method="unsteered", method_family="none", arm="none")
    # the unsteered row at the 0.25 threshold too: the live export files the none at every threshold its steered rows carry, and the
    # steered 0.25 row's effect size was paired against it (r2_export 11 anchors each threshold's comparison on the anchor's reading AT that threshold)
    row(cell_id="none", component="static", eval_id="pasttense", eval="past tense", metric="asr", threshold=0.25, value=0.6, k=6, n=10, ci_lo=0.3, ci_hi=0.85,
        ci_method="Wilson 95% (recomputed from asr_grid × n)", decode_cap_tokens=4096, at_cap_share=0.1, loop_share=0.0, degenerate_share=0.1,
        defended_degenerate_k=1, defended_n=5, method="unsteered", method_family="none", arm="none")
    row(cell_id=LORA, component="static", eval_id="pasttense", eval="past tense", metric="asr", threshold=0.5, value=0.3, k=3, n=10, ci_lo=0.1, ci_hi=0.6,
        ci_method="Wilson 95% (join)", decode_cap_tokens=4096, delta=-20.0, delta_lo=-40.0, delta_hi=0.0, delta_p=0.25,
        delta_method="paired bootstrap over goals (10,000 draws, seed 0) + exact McNemar; join", effect_size=0.41, effect_kind="cohen_h defence (1−ASR) vs own unsteered",
        at_cap_share=0.2, loop_share=0.1, degenerate_share=0.2, defended_degenerate_k=2, defended_n=7, loop_share_first512=0.05,
        steer_live_ratio=float("inf"), steer_live_d=0.3, steer_live_noise=0.0)   # the live export's d / zero-noise ratio
    row(cell_id=LORA, component="static", eval_id="pasttense", eval="past tense", metric="asr", threshold=0.25, value=0.4, k=4, n=10, ci_lo=0.15, ci_hi=0.7,
        ci_method="Wilson 95% (recomputed from asr_grid × n)", decode_cap_tokens=4096, effect_size=0.2, effect_kind="cohen_h defence (1−ASR) vs own unsteered",
        at_cap_share=0.2, loop_share=0.1, degenerate_share=0.2, defended_degenerate_k=2, defended_n=7, loop_share_first512=0.05)
    # bon curve points (asr at N=64 and N=75) at 512
    row(cell_id=LORA, component="bon", eval_id="bon", eval="best-of-75", metric="asr", threshold=0.5, bon_n=64, value=0.6, k=6, n=10, decode_cap_tokens=512,
        judge="judge-model-x / strongreject_v1", provenance=f"{prov_ts}.{LORA}.bon")
    row(cell_id=LORA, component="bon", eval_id="bon", eval="best-of-75", metric="asr", threshold=0.5, bon_n=75, value=0.7, k=7, n=10, decode_cap_tokens=512,
        provenance=f"{prov_ts}.{LORA}.bon")
    if petri_degeneration_first:
        # the live export sorts a cell's rows by metric name, so the Petri unsteered cell's `at_cap_share` (a degeneration statistic over
        # its 725 replies, from degeneracy_petri.json) is listed BEFORE its `petri_score` row: the same run, no judged count on this row
        row(cell_id="none", component="petri", eval_id="petri_pooled", eval="Petri (6 behaviours pooled)", metric="at_cap_share", value=0.15, k=3, n=20,
            judge="auditor judge-model-w, judge judge-model-y (Petri)", petri_run_key="aaaa0000000000ff", petri_intervention_hash="ffff0000",
            note="share of target replies decoded to the token cap", provenance=f"{agg_path} ›{VIEW}.cells.none.degeneracy",
            method="unsteered", method_family="none", arm="none")
    # petri pooled row for the lora cell (two producers) and its none anchor
    row(cell_id="none", component="petri", eval_id="petri_pooled", eval="Petri (6 behaviours pooled)", metric="petri_score", value=5.5, n=2,
        judge="auditor judge-model-w, judge judge-model-y (Petri)", petri_run_key="aaaa0000000000ff", petri_intervention_hash="ffff0000",
        provenance=f"{agg_path} ›{VIEW}.cells.none", method="unsteered", method_family="none", arm="none")
    row(cell_id=LORA, component="petri", eval_id="petri_pooled", eval="Petri (6 behaviours pooled)", metric="petri_score", value=4.5, n=2, delta=-1.0, delta_lo=-2.0,
        delta_hi=0.0, delta_method="paired mean difference over 60 shared (behaviour, scenario) items, percentile bootstrap (join petri_vs_bare_cell)",
        effect_size=0.9, effect_kind="d_z paired (per cell, 60 items pooled), sign flipped", effect_lo=0.1, effect_hi=1.5,
        judge="auditor judge-model-w, judge judge-model-y (Petri)", petri_run_key=petri_stats_key, petri_intervention_hash="1111000011110000",
        provenance=f"{agg_path} ›{VIEW}.cells.si_constv2_mix_full__lora_r4_ce")
    # the filed greedy 512 multi-turn pair (none anchor + the lora cell with its paired contrast)
    row(cell_id="none", component="multi-turn", eval_id="crescendo", eval="Crescendo", metric="asr", threshold=0.5, value=0.9, k=9, n=10, decode_cap_tokens=512,
        at_cap_share=0.7, loop_share=0.01, degenerate_share=0.7, method="unsteered", method_family="none", arm="none",
        provenance=f"{prov_ts}.none.crescendo")
    row(cell_id=LORA, component="multi-turn", eval_id="crescendo", eval="Crescendo", metric="asr", threshold=0.5, value=0.8, k=8, n=10, decode_cap_tokens=512,
        delta=-10.0, delta_lo=-30.0, delta_hi=10.0, delta_p=1.0, delta_method="paired bootstrap over goals (10,000 draws, seed 0) + exact McNemar; join",
        effect_size=0.3, effect_kind="cohen_h defence (1−ASR) vs own unsteered", at_cap_share=0.6, loop_share=0.02, degenerate_share=0.6,
        defended_degenerate_k=1, defended_n=2, provenance=f"{prov_ts}.{LORA}.crescendo")
    if missing_source:
        # a source the builder read that is gone from disk (renamed / in flight); the one row naming it is the refusal's whole scope
        row(cell_id=LORA, component="multi-turn", eval_id="crescendo", eval="Crescendo", metric="degen_classifier_share", value=0.03, n=100,
            decode_cap_tokens=512, provenance=f"{rs / 'vanished_union.json'} ›cells[{VIEW}|crescendo|{LORA}].degen_classifier_share")
    moved_path = rs / "bon_cap4096_stats.json"
    if moved_source:
        # a source the builder read and pinned, then rewritten by its producer (the live 2026-09-20 case: `bon_cap4096/analysis/
        # bon_cap4096_stats.json`, 378 best-of-N climb rows over 42 cells); the one row naming it is the refusal's whole scope
        with open(moved_path, "w") as fh:
            json.dump({"cells": {f"{VIEW}|{LORA}": {"asr_bon_cap4096": 0.8}}}, fh)
        row(cell_id=LORA, component="bon", eval_id="bon", eval="best-of-75", metric="asr_bon_cap4096", threshold=0.5, value=0.8, k=8, n=10,
            decode_cap_tokens=4096, provenance=f"{moved_path} ›cells[{VIEW}|{LORA}].asr_bon_cap4096")
    if degeneration_rows:
        # the export's DEGENERATION STATISTICS filed as metric rows per cell (the live parquet: at_cap_share x1066, loop_share x752,
        # degenerate_share x650, degen_<detector>_share x368 each, the union instrument's rows): the crescendo lora cell at 512
        # carries the producer's three, the union classifier's five detectors + union, the two at-cap estimands, the decomposition
        # and the defended bracket; every value is k / n exactly, as on the live export
        union = f"{rs / 'degeneracy_union' / 'degeneracy_full22_union.json'} ›cells[{VIEW}|crescendo|{LORA}] (DEFINITION_SHA 98ae9114672c)"
        for metric, k, n, note, prov in (
                ("at_cap_share", 60, 100, "share of responses decoded to the token cap", f"{prov_ts}.degeneracy"),
                ("loop_share", 2, 100, "share of responses whose word 4-gram repetition ratio > 0.5", f"{prov_ts}.degeneracy"),
                ("degenerate_share", 60, 100, "share of responses that looped OR ran to the cap (union)", f"{prov_ts}.degeneracy"),
                ("degen_classifier_share", 1, 100, "share of responses the pinned loop classifier flags (tail loop or line loop) -- one of the five detectors", union),
                ("degen_contentless_share", 0, 100, "share of contentless responses -- one of the five detectors", union),
                ("degen_rep_ratio_loop_share", 2, 100, "share of responses whose word 4-gram repetition ratio > 0.5 (the published loop rule)", union),
                ("degen_zlib_collapse_share", 0, 100, "share of zlib-collapsed responses -- one of the five detectors", union),
                ("degenerate_share_union5", 62, 100, "share flagged by ANY of five detectors -- loop classifier, word 4-gram repetition > 0.5, zlib collapse, contentless, at-cap. Denominator: target replies (every target turn in the store), n = 100. union-of-five degeneracy module (DEFINITION_SHA 98ae9114672c)", union),
                ("at_cap_share_retokenized", 58, 100, "re-tokenized length >= 0.98 x cap -- THE at-cap reading (CANONICAL)", union),
                ("at_cap_share_finish_reason", 60, 100, "finish_reason == 'length' -- the FINISH-REASON estimand", union),
                ("degen_truncation_only_share", 50, 100, "share of target replies at cap (canonical) with NO other detector firing", union),
                ("degen_loop_without_cap_share", 1, 100, "share of target replies flagged by a loop detector and NOT at cap", union),
                ("defended_degenerate_share", 1, 2, "defended goals with a degenerate response", f"{prov_ts}.degeneracy")):
            row(cell_id=LORA, component="multi-turn", eval_id="crescendo", eval="Crescendo", metric=metric, value=k / n, k=k, n=n, decode_cap_tokens=512,
                at_cap_share=0.6, loop_share=0.02, degenerate_share=0.6, defended_degenerate_k=1, defended_n=2, note=note, provenance=prov)
    # the sampled re-run's POOLED reading of the lora cell at its exit rung (the union's pooled sampled rows attach to this run)
    row(cell_id=LORA, component="static", eval_id="pasttense_sampled4096", eval="past tense (sampled)", metric="asr", threshold=0.5, value=0.3, k=3, n=10,
        ci_lo=0.1, ci_hi=0.6, decode_cap_tokens=8192, decode="vendor preset (T=1.0)", at_cap_share=0.0, loop_share=0.05, degenerate_share=0.05,
        provenance=f"{prov_ts}.{LORA}.pasttense_sampled4096")
    # an over-refusal row with no judge string (the eval implies its CPI judge) and a checker-scored capability row (no judge at all)
    over_refusal_contrast = (dict(delta=-20.0, delta_lo=-40.0, delta_hi=0.0, delta_p=0.25, effect_size=0.45, effect_kind="cohen_h defence (1−rate) vs own unsteered",
                                  delta_method="paired bootstrap over items (10,000 draws, seed 0) + exact McNemar; join")
                             if overrefusal_contrast_rows else {})
    row(cell_id=LORA, component="over-refusal", eval_id="xstest", eval="XSTest", metric="over_refusal_rate", value=0.1, k=1, n=10, decode_cap_tokens=512,
        judge=None, provenance=f"{prov_ts}.{LORA}.xstest", **over_refusal_contrast)
    if overrefusal_contrast_rows:
        # this standup's unsteered xstest row at the same cap: the export's own anchor of the lora row's over-refusal comparison (and, through
        # the twin the adapter files on it too, of the refused-rate twin's comparison)
        row(cell_id="none", component="over-refusal", eval_id="xstest", eval="XSTest", metric="over_refusal_rate", value=0.3, k=3, n=10, decode_cap_tokens=512,
            judge=None, method="unsteered", method_family="none", arm="none", provenance=f"{prov_ts}.none.xstest")
    row(cell_id=LORA, component="capability", eval_id="mmlu", eval="MMLU", metric="acc", value=0.6, k=6, n=10, decode_cap_tokens=None,
        judge=None, provenance=f"{prov_ts}.{LORA}.mmlu")
    # a pending row (queued run) and a not-applicable design-grid row (disposition)
    row(cell_id=RELU, component="static", eval_id="codeattack", eval="CodeAttack", metric="asr", threshold=0.5, status="pending", note="pending: not yet graded",
        method_family="control", arm="gated")
    row(cell_id=SUB, component="multi-turn", eval_id="crescendo", eval="Crescendo", metric="asr", status="not applicable", note="no such object for this lane",
        method_family="derived", arm="lora")
    # a relu_native-style cap-4096 multi-turn row with a grades root and a cross-engine anchor stamp
    row(cell_id=RELU, component="multi-turn", eval_id="crescendo", eval="Crescendo", metric="asr", threshold=0.5, value=0.9, k=9, n=10, decode_cap_tokens=4096,
        decode="shipped sampling (T=1.0)", grades_root="grades_synth_bu_synthview", delta=-5.0, delta_lo=-10.0, delta_hi=0.0, delta_p=0.5,
        delta_method="paired bootstrap over goals (10,000 draws, seed 0) + exact McNemar; lane file", effect_size=0.1,
        effect_kind="cohen_h defence (1−ASR) vs own unsteered", pairing_allowance="passthrough-anchor-engine-agnostic-2026-09-15", anchor_engine="lens",
        anchor_view=VIEW_IT, anchor_root="grades_anchor_synthview_it", attacker_provenance_class="attested-from-elsewhere", attacker_identity_attested="attested",
        method_family="control", arm="gated", pass_="relu_native")
    rows[-1]["pass"] = rows[-1].pop("pass_")
    row(view=VIEW_IT, cell_id="none", component="multi-turn", eval_id="crescendo", eval="Crescendo", metric="asr", threshold=0.5, value=0.95, k=95, n=100,
        decode_cap_tokens=4096, decode="shipped sampling (T=1.0)", grades_root="grades_anchor_synthview_it", method="unsteered", method_family="none", arm="none",
        model=VIEW_IT)
    if halfdose_rows:
        # the lora object at HALF dose (the `<tag>_halfdose` standup; dose 0.5): the export carries no unsteered row under this standup (the
        # producer's delta was paired against the half-dose standup's own bare run, absent from the export), and `delta_vs_full_dose*` -- the
        # half-dose join's paired record against the same object's 1x reading (the main standup's lora row), stamped on the headline threshold's row
        row(view=f"{VIEW}_halfdose", dose=0.5, cell_id=LORA, component="static", eval_id="pasttense", eval="past tense", metric="asr", threshold=0.5,
            value=0.4, k=4, n=10, ci_lo=0.15, ci_hi=0.7, ci_method="Wilson 95% (join)", decode_cap_tokens=4096,
            delta=-12.0, delta_lo=-30.0, delta_hi=6.0, delta_p=0.4, delta_method="paired bootstrap over goals (10,000 draws, seed 0) + exact McNemar; join",
            effect_size=0.25, effect_kind="cohen_h defence (1−ASR) vs own unsteered",
            delta_vs_full_dose=10.0, delta_vs_full_dose_lo=-5.0, delta_vs_full_dose_hi=25.0,
            at_cap_share=0.1, loop_share=0.0, degenerate_share=0.1, provenance=f"{prov_ts}.{LORA}.pasttense_halfdose")
    final_rung_path = rs / "sampled" / "report" / "analysis" / "final_rung.json"
    if overrefusal_count_rows:
        # (1) the sampled re-run's POOLED over-refusal reading of the lora cell at its exit cap: `n` counts the judged LABELS over two
        # seeded trials (19 of 2 x 10 items), the judged ITEM count (10) lives in the producer's pooled block at the row's provenance path
        os.makedirs(final_rung_path.parent, exist_ok=True)
        pooled = {"trials": [0, 1], "final_caps": [2048],
                  "at_cap_at_final": {"k": 1, "n": 20, "p": 0.05},
                  "full_refusal": {"k": 3, "n": 19, "p": 3 / 19, "n_items": 10, "bootstrap95": {"n_clusters": 10}},
                  "full_plus_partial_refusal": {"k": 5, "n": 19, "p": 5 / 19, "n_items": 10, "bootstrap95": {"n_clusters": 10}}}
        with open(final_rung_path, "w") as fh:
            json.dump({"name": "final_rung", "targets": {VIEW: {"controls": {
                f"xstest|{LORA}": {"cell": LORA, "kind": "controls", "readout": "xstest", "status": "ok", "pooled": pooled,
                                   "trials": {"0": {"over_refusal": {"n_judged": 10, "n_abstained": 0, "n": 10}}, "1": {"over_refusal": {"n_judged": 9, "n_abstained": 1, "n": 10}}}},
                # a block the producer filed WITHOUT n_items: the count stays NULL with the reason (never derived from the labels)
                f"or_bench_hard_1k|{LORA}": {"cell": LORA, "kind": "controls", "readout": "or_bench_hard_1k", "status": "ok",
                                             "pooled": {"trials": [0, 1], "full_refusal": {"k": 4, "n": 20, "p": 0.2}}}}}}}, fh)
        fr = f"{final_rung_path} ›targets.{VIEW}.controls.xstest|{LORA}.pooled"
        row(cell_id=LORA, component="over-refusal", eval_id="xstest_sampled4096", eval="XSTest (sampled re-run, final rung)", metric="at_cap_share", value=0.05, k=1, n=20,
            decode_cap_tokens=2048, decode="vendor preset (T=1.0)", judge=None, note="share of the benign replies still at the cap AT THE FINAL RUNG (1 of 20)",
            provenance=f"{fr}.at_cap_at_final")
        row(cell_id=LORA, component="over-refusal", eval_id="xstest_sampled4096", eval="XSTest (sampled re-run, final rung)", metric="over_refusal_rate_full", value=3 / 19, k=3, n=19,
            decode_cap_tokens=2048, decode="vendor preset (T=1.0)", judge="judge-model-z (over-refusal judge, pass 2)", at_cap_share=0.05,
            note="share of the judged labels that are labelled FULL refusal: 3 of 19 (both trials pooled)", provenance=f"{fr}.full_refusal")
        row(cell_id=LORA, component="over-refusal", eval_id="xstest_sampled4096", eval="XSTest (sampled re-run, final rung)", metric="over_refusal_rate_full_plus_partial", value=5 / 19, k=5, n=19,
            decode_cap_tokens=2048, decode="vendor preset (T=1.0)", judge="judge-model-z (over-refusal judge, pass 2)", at_cap_share=0.05,
            note="share of the judged labels that are labelled full OR partial refusal: 5 of 19 (both trials pooled)", provenance=f"{fr}.full_plus_partial_refusal")
        fr2 = f"{final_rung_path} ›targets.{VIEW}.controls.or_bench_hard_1k|{LORA}.pooled"
        row(cell_id=LORA, component="over-refusal", eval_id="or_bench_hard_1k_sampled4096", eval="OR-Bench-Hard (sampled re-run, final rung)", metric="over_refusal_rate_full",
            value=0.2, k=4, n=20, decode_cap_tokens=2048, decode="vendor preset (T=1.0)", judge="judge-model-z (over-refusal judge, pass 2)", at_cap_share=0.05,
            note="share of the judged labels that are labelled FULL refusal: 4 of 20 (both trials pooled)", provenance=f"{fr2}.full_refusal")
        # (2) a controls store's at-cap decomposition for the relu cell, filed under the STORE's producer key (the `/cells/<key>/` path of
        # its judge_items.jsonl): `at_cap_share_all` says "of the 10 replies judged (every label)" -- the run's only count
        store_prov = f"{rs / 'grades' / 'cells' / 'relu_store_x' / 'judge_items.jsonl'} (status == judged; subset == safe)"
        for metric, k, n, note in (("at_cap_share_all", 2, 10, "of the 10 replies judged (every label) (10 judged; XSTest's safe prompts), the share whose generation ran to the 512-token cap"),
                                   ("at_cap_share_compliance", 1, 8, "of the 8 replies judged compliant, the share at the cap"),
                                   ("at_cap_share_refusal", 1, 2, "of the 2 replies judged a refusal, the share at the cap")):
            row(cell_id=RELU, component="over-refusal", eval_id="xstest", eval="XSTest", metric=metric, value=k / n, k=k, n=n, decode_cap_tokens=512, judge=None,
                note=note, provenance=store_prov, method_family="control", arm="gated")
        # (3) a store the export files as generated but not yet judged: a measured at-cap statistic beside a PENDING judged-metric row --
        # nothing on the run counts judged items
        row(cell_id=SUB, component="over-refusal", eval_id="or_bench_hard_1k", eval="OR-Bench-Hard", metric="at_cap_share", value=0.3, k=3, n=10, decode_cap_tokens=512,
            judge=None, note="share of the 10 benign replies decoded to the 512-token cap (all prompts)", provenance=f"{prov_ts}.{SUB}.or_bench_hard_1k",
            method_family="derived", arm="lora")
        row(cell_id=SUB, component="over-refusal", eval_id="or_bench_hard_1k", eval="OR-Bench-Hard", metric="over_refusal_rate_terminated", status="pending",
            judge=None, note="generated; no paired judge record for this cell yet", provenance=f"{prov_ts}.{SUB}.or_bench_hard_1k", method_family="derived", arm="lora")
        # (4) a per-behaviour Petri row for the lora cell (its count is the row's own n, as for the pooled score)
        row(cell_id=LORA, component="petri", eval_id="defer_to_users", eval="Petri: defer to users", metric="petri_score", value=4.0, n=10,
            judge="auditor judge-model-w, judge judge-model-y (Petri)", petri_run_key=petri_stats_key, petri_intervention_hash="1111000011110000",
            provenance=f"{agg_path} ›{VIEW}.cells.si_constv2_mix_full__lora_r4_ce.behaviors.defer_to_users")
    df = pd.DataFrame(rows, columns=list(COLUMN_MAP))
    df["row_id"] = range(len(df))
    if extra_column:
        df[extra_column] = 1
    pq = data / "sdf_base_steering_r2_results.parquet"
    df.to_parquet(pq, index=False)
    with open(data / "sdf_base_steering_r2_schema.json", "w") as fh:
        json.dump({c: "synthetic" for c in COLUMN_MAP}, fh)
    sources = {"built_utc": "2026-01-02T00:00:00+00:00", "n_rows": len(df), "n_measured_rows": int(df["value"].notna().sum()),
               "sources": {str(ts_path): _src_entry(str(ts_path)), str(stats_path): _src_entry(str(stats_path)), str(agg_path): _src_entry(str(agg_path)),
                           str(rs / "grades_synth_bu_synthview") + "/ (cap-4,096 multi-turn grade root)": {"view": VIEW, "aggregates": {}}}}
    if missing_source:
        sources["sources"][str(rs / "vanished_union.json")] = {"mtime_utc": "2026-01-01T00:00:00+00:00", "size": 123, "sha256": "0" * 64}
    if moved_source:
        sources["sources"][str(moved_path)] = _src_entry(str(moved_path))      # the pin, taken over the bytes the builder read ...
        with open(moved_path, "w") as fh:
            json.dump({"cells": {f"{VIEW}|{LORA}": {"asr_bon_cap4096": 0.85}}, "rewritten": True}, fh)    # ... which a producer then rewrote
    if overrefusal_count_rows:
        sources["sources"][str(final_rung_path)] = _src_entry(str(final_rung_path))   # sha-pinned, as the live builder pins it
    with open(data / "sdf_base_steering_r2_sources.json", "w") as fh:
        json.dump(sources, fh)
    return {"data": str(data), "rs": str(rs), "consolidated": str(cons), "parquet": str(pq), "stats": str(stats_path), "agg": str(agg_path), "ts": str(ts_path),
            "moved": str(moved_path), "final_rung": str(final_rung_path)}
