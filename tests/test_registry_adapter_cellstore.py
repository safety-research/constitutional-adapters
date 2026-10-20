"""The generic CellStore reader on a SYNTHETIC served-controls store: one run per (cell key,
control) with the object discriminator, one item_result per (item, metric) exactly as stored, item_flags with the
pointer pair and at-cap, the judgment's counts / refusals / api_pool, producer aggregates as run_metric, a
run_degeneration row per declared measure, every record validating; refusals on a missing manifest, a
symlinked input, an unresolved model spelling and a rubric that differs from the seed judge; the end-to-end
pass + ingest lands rows in the registry."""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import cellstore as CS
from sisteer.registry.adapters import runner as RUN

import registry_items_fixtures as F

NOW = "2026-01-05T00:00:00.000Z"


def _ctx(conn=None):
    return AB.AdapterContext(root="/nonexistent", now=NOW, producer={"script": "tests", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}, conn=conn)


def _read(store, dims=None, **kw):
    reader = CS.StoreReader(dims or F.dims_map(), "synthcampaign", NOW, **kw)
    return reader.read(store, CS.store_inputs(store), view=os.path.basename(os.path.dirname(os.path.dirname(store))))


def _validate_all(sr):
    from sisteer.registry import records as R
    bad = []
    for rt, p in sr.records:
        p = dict(p)
        if rt in R.ITEM_SCALE_RECORD_TYPES:
            p.setdefault("source_revision_id", "rev1"); p.setdefault("withdrawn_by_revision_id", None)
        rec = R.make_record(rt, p, {"script": "t", "code_sha": "unverified", "session": None, "host": "TESTHOST", "substrate": "grading_host"}, I.adapter_record_id(p, rt), NOW)
        out = VAL.validate_record(rec, "mem")
        if not isinstance(out, VAL.Validated):
            bad.append((rt, out.field, out.reason))
    return bad


def test_run_key_scope_target_dir_prefixes_the_producer_key_and_the_default_keeps_the_store_key(tmp_path):
    """`StoreReader(run_key_scope=...)` (RUN_KEY_SCOPES): the default keys a store's runs by the store key alone — what every
    constructing adapter emitted before, so their EMISSION_VERSIONs stand; `target_dir` keys them `<target dir>/<store key>`, the
    opt-in the over-refusal source takes because its unsteered store key recurs across a model's target dirs (b2, 2026-09-19).
    A `target_dir` read that names no target directory refuses the store; an unknown scope is a construction error."""
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, "none", ihash8="2ff9cdff", object_sha16=None, n=3)
    key = F.store_key("none", "2ff9cdff")
    default = _read(store)
    scoped = _read(store, run_key_scope="target_dir")
    d_runs = {p["eval_id"]: p for rt, p in default.records if rt == "run"}
    s_runs = {p["eval_id"]: p for rt, p in scoped.records if rt == "run"}
    assert set(d_runs) == set(s_runs) == {"xstest", "or_bench_hard", "mmlu", "gsm8k"}
    for ev in d_runs:
        assert d_runs[ev]["producer_run_key_legacy"] == key
        assert s_runs[ev]["producer_run_key_legacy"] == f"synthmodel/{key}"
        assert d_runs[ev]["producer_object_key_legacy"] == s_runs[ev]["producer_object_key_legacy"] == "none@2ff9cdff"
        assert d_runs[ev]["run_id"] != s_runs[ev]["run_id"]
        assert d_runs[ev]["run_id"] == I.run_id("synthcampaign", key, "none@2ff9cdff", ev, d_runs[ev]["instrument_id"])
        assert s_runs[ev]["run_id"] == I.run_id("synthcampaign", f"synthmodel/{key}", "none@2ff9cdff", ev, s_runs[ev]["instrument_id"])
        # the same store under a second target dir: a second run under `target_dir`, the SAME run by default (the collision)
        assert I.run_id("synthcampaign", f"synthmodel_halfdose/{key}", "none@2ff9cdff", ev, s_runs[ev]["instrument_id"]) != s_runs[ev]["run_id"]
    # every dependent record follows the run id (judgments, items, flags, metrics, degeneration)
    for rt, p in scoped.records:
        if "run_id" in p:
            assert p["run_id"] in {r["run_id"] for r in s_runs.values()}, rt
    assert not _validate_all(scoped)
    reader = CS.StoreReader(F.dims_map(), "synthcampaign", NOW, run_key_scope="target_dir")
    with pytest.raises(AB.AdapterRefusal, match="target directory"):
        reader.read(store, CS.store_inputs(store), view="")
    with pytest.raises(ValueError, match="run_key_scope"):
        CS.StoreReader(F.dims_map(), "synthcampaign", NOW, run_key_scope="cells")


def test_store_yields_one_run_per_control_with_object_discriminator(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=3)
    sr = _read(store)
    assert set(sr.controls) == {"xstest", "or_bench_hard_1k", "mmlu", "gsm8k"}
    runs = {p["eval_id"]: p for rt, p in sr.records if rt == "run"}
    assert set(runs) == {"xstest", "or_bench_hard", "mmlu", "gsm8k"}
    key = F.store_key(F.LORA_CELL)
    for p in runs.values():
        assert p["producer_run_key_legacy"] == key
        assert p["producer_object_key_legacy"] == f"{F.LORA_CELL.replace('__', '--')}@{F.IHASH8}"
        assert p["object_id"] == F.OBJ_SHA and p["model_id"] == F.MODEL_IT and p["transfer_kind"] == "base_to_instruct"
        assert p["lifecycle_state"] == "generated" and p["queued_utc"] and p["started_utc"] and p["generated_utc"]
        assert p["serving_path"] == "lens" and p["applied_positions_mode"] == "all" and p["cluster_job_id"] == "424242"
        assert p["code_sha"] == "unverified" and p["field_confidence_json"]["code_sha"]["confidence"] == "unverified"
        assert I.parse_run_id(p["run_id"])["eval_id"] == p["eval_id"]
    assert runs["xstest"]["token_cap_tokens"] == 512 and runs["mmlu"]["token_cap_tokens"] == 1 and runs["xstest"]["decode_source"] == "greedy"
    assert len({p["run_id"] for p in runs.values()}) == 4
    assert not _validate_all(sr)


# --------------------------------------------------------------------------------------------
# transfer_kind (2026-09-30): the manifest's fit model OR the ingested object row's; never the `unverified` sentinel in the column
# --------------------------------------------------------------------------------------------

ALPHASTEER_HF = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def _alphasteer_dims(fit_model_id=F.MODEL_IT, object_row: bool = True):
    """The fixture dims with the served model resolving to the instruct checkpoint and, by default, an ingested object row for the served
    object (the alphasteer_rebake objects bank's row: `object_kind alphasteer`, `fit_model_id llama8b_it`)."""
    d = F.dims_map()
    d.models[ALPHASTEER_HF] = F.MODEL_IT
    if object_row:
        d.object_rows[F.OBJ_SHA] = {"object_kind": "alphasteer", "paper_label": "alphasteer_refit_reference", "object_role": "baseline_published",
                                    "campaign": "synthcampaign", "campaign_cell_id_legacy": "alphasteer", "fit_model_id": fit_model_id, "fit_positions": "all",
                                    "fit_replicate_label": None, "base_scale": None, "derivation": None, "status": "document_fit", "extra": {}}
    return d


def test_a_store_without_fit_provenance_takes_the_fit_model_from_the_ingested_object_row(tmp_path):
    """The live alphasteer_f22 shape: `manifest.json` carries NO `fit_provenance` key, the served object is attested by evidence.json's
    engine stamp, and the registry already holds the object row (alphasteer_rebake's bank) naming `fit_model_id llama8b_it`. The run
    files `instruct_direct` -- the value alphasteer_rebake and r2_export file for the same objects -- under a derived confidence whose
    source names the object row and the missing manifest key; every record validates (no `is not in TRANSFER_KINDS` refusal)."""
    view = str(tmp_path / "root" / "llama8b_lam1_t0")
    store = F.write_controls_store(view, "alphasteer", hf_model=ALPHASTEER_HF, fit_provenance=False, n=2)
    manifest = json.load(open(os.path.join(store, "manifest.json")))
    assert "fit_provenance" not in manifest
    sr = _read(store, dims=_alphasteer_dims())
    runs = [p for rt, p in sr.records if rt == "run"]
    assert runs and sr.object_kind == "alphasteer" and sr.object_id == F.OBJ_SHA
    for p in runs:
        assert p["transfer_kind"] == "instruct_direct" and p["model_id"] == F.MODEL_IT
        fc = p["field_confidence_json"]["transfer_kind"]
        assert fc["confidence"] == "derived" and "object.fit_model_id" in fc["source"] and "no fit_provenance" in fc["source"]
    assert not _validate_all(sr)
    # the object row's fit model on the BASE checkpoint: base_to_instruct, from the row alone
    sr_base = _read(store, dims=_alphasteer_dims(fit_model_id=F.MODEL_BASE))
    assert {p["transfer_kind"] for rt, p in sr_base.records if rt == "run"} == {"base_to_instruct"}
    assert not _validate_all(sr_base)


def test_a_fitted_store_with_neither_attestation_is_refused_by_name_never_the_sentinel(tmp_path):
    view = str(tmp_path / "root" / "llama8b_lam1_t0")
    store = F.write_controls_store(view, "alphasteer", hf_model=ALPHASTEER_HF, fit_provenance=False, n=2)
    with pytest.raises(AB.AdapterRefusal) as ei:
        _read(store, dims=_alphasteer_dims(object_row=False))                  # no object row -> dims.object_facts is None
    msg = str(ei.value)
    assert "manifest.json" in msg and "transfer_kind unresolved" in msg and "object.fit_model_id missing" in msg and F.OBJ16 in msg
    # a manifest that NAMES a fit model the registry does not know: refused, not a silent None (which minted the sentinel)
    store2 = F.write_controls_store(str(tmp_path / "root2" / "synthmodel"), F.LORA_CELL, fit_hf="synthetic/unknown-fit-9b", n=2)
    with pytest.raises(AB.AdapterRefusal, match="fit_provenance.model_id 'synthetic/unknown-fit-9b' is no model of the registry"):
        _read(store2)


def test_a_manifest_fit_model_that_disagrees_with_the_object_row_is_refused_and_an_agreeing_one_files_as_before(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=2)                     # fit_provenance.model_id = the BASE checkpoint
    dims = F.dims_map()
    dims.object_rows[F.OBJ_SHA] = {"object_kind": "lora", "paper_label": "constitution_lora", "object_role": "constitution", "campaign": "synthcampaign",
                                   "campaign_cell_id_legacy": F.LORA_CELL, "fit_model_id": F.MODEL_IT, "fit_positions": "all", "fit_replicate_label": None,
                                   "base_scale": None, "derivation": None, "status": "document_fit", "extra": {}}
    with pytest.raises(AB.AdapterRefusal, match="the two attestations disagree"):
        _read(store, dims=dims)
    dims.object_rows[F.OBJ_SHA]["fit_model_id"] = F.MODEL_BASE
    sr = _read(store, dims=dims)
    runs = [p for rt, p in sr.records if rt == "run"]
    # the manifest's attestation wins the spelling: the same value and the SAME source text as before this rule (the bytes of every store
    # carrying fit_provenance do not move)
    assert {p["transfer_kind"] for p in runs} == {"base_to_instruct"}
    assert {p["field_confidence_json"]["transfer_kind"]["source"] for p in runs} == {"fit on base checkpoint, served on instruct checkpoint"}
    assert not _validate_all(sr)


def test_no_cellstore_run_ever_carries_the_unverified_transfer_kind_and_the_reader_family_bumped_together(tmp_path):
    """Over every fixture shape the reader files: the value is a member of TRANSFER_KINDS (the sentinel is a confidence, never a value).
    The derivation is a default change of the shared reader, so its three constructors bumped together."""
    from sisteer.registry import vocab as V
    from sisteer.registry.adapters import overrefusal_items as OI
    from sisteer.registry.adapters import sampled_controls as SC
    view = str(tmp_path / "root" / "synthmodel")
    stores = [F.write_controls_store(view, F.LORA_CELL, n=2), F.write_controls_store(view, "none", ihash8="2ff9cdff", object_sha16=None, n=2),
              F.write_controls_store(view, F.PROMPTED_CELL, ihash8="faa4c8cf", object_sha16=None, n=2)]
    for store in stores:
        for rt, p in _read(store).records:
            if rt == "run":
                assert p["transfer_kind"] in V.TRANSFER_KINDS, (store, p["transfer_kind"])
    alpha = F.write_controls_store(str(tmp_path / "root" / "llama8b_lam1_t0"), "alphasteer", hf_model=ALPHASTEER_HF, fit_provenance=False, n=2)
    for rt, p in _read(alpha, dims=_alphasteer_dims()).records:
        if rt == "run":
            assert p["transfer_kind"] in V.TRANSFER_KINDS
    assert "unverified" not in V.TRANSFER_KINDS
    # 7 / 5 / 5 bumped together for the transfer_kind rule; the shared store reader's 8 is its own: the two
    # other constructors file their contrasts elsewhere (`overrefusal_items` files none; `sampled_controls` through its own pooled pairing)
    assert CS.CellStoreAdapter.EMISSION_VERSION == 10 and OI.OverrefusalItemsAdapter.EMISSION_VERSION == 7 and SC.SampledControlsAdapter.EMISSION_VERSION == 7   # 10 = above both parents (cellstore.py); 7 / 7 = the merged files (27b)


def test_run_source_is_the_adapter_s_word_per_control(tmp_path):
    """A store read by no paper reader files NO `lane_source_legacy` (the generic adapter: cache stores, drills — registry-only rows,
    paper.paper_frame_exclusion); an adapter that IS a paper reader names its source per control it files, and the store's other
    controls stay unattributed."""
    store = F.write_controls_store(str(tmp_path / "root" / "synthmodel"), F.LORA_CELL, n=2)
    plain = {p["eval_id"]: p["lane_source_legacy"] for rt, p in _read(store).records if rt == "run"}
    assert plain and set(plain.values()) == {None}
    named = {p["eval_id"]: p["lane_source_legacy"] for rt, p in _read(store, source_by_control_legacy={"xstest": "overrefusal_items"}).records if rt == "run"}
    assert named["xstest"] == "overrefusal_items" and all(v is None for e, v in named.items() if e != "xstest")


def test_items_are_one_row_per_item_metric_with_pointers_and_pool(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=3, at_cap_ids=["orb0001-cccccccc"])
    sr = _read(store)
    items = [p for rt, p in sr.records if rt == "item_result"]
    flags = [p for rt, p in sr.records if rt == "item_flags"]
    orb = sr.controls["or_bench_hard_1k"]
    orb_items = [p for p in items if p["run_id"] == orb.run_id]
    # orb: label + refused per item, judged metrics under the judgment with the pool and the verdict time
    assert sorted({p["metric_id"] for p in orb_items}) == ["orb_label", "orb_refused"]
    assert len(orb_items) == 6
    assert all(p["judgment_id"] == orb.judgment_id and p["api_pool"] == "lowprio" and p["judged_utc"] for p in orb_items)
    labels = {p["item_id"]: p["label"] for p in orb_items if p["metric_id"] == "orb_label"}
    assert labels["orb0000-cccccccc"] == "full_compliance" and labels["orb0002-cccccccc"] == "full_refusal"
    # mmlu rows: generation-time metric, no judgment, value as stored, extra parked
    mm = sr.controls["mmlu"]
    mm_items = [p for p in items if p["run_id"] == mm.run_id]
    assert len(mm_items) == 3 and all(p["judgment_id"] == "" and p["api_pool"] is None for p in mm_items)
    assert mm_items[0]["extra"]["gold"] == "C"
    # flags: pointer at rows.jsonl (sha + line) and judge_items (sha + line); at_cap from the judge record
    rows_sha = AB.check_input(os.path.join(store, "rows.jsonl")).sha256
    ji_sha = AB.check_input(os.path.join(store, "judge_items.jsonl")).sha256
    orb_flags = {p["item_id"]: p for p in flags if p["run_id"] == orb.run_id}
    assert len(orb_flags) == 3
    f1 = orb_flags["orb0001-cccccccc"]
    assert f1["generation_artifact_id"] == rows_sha and isinstance(f1["generation_offset"], int) and f1["generation_offset"] > 0
    # the greedy store's per-item cap reading is the judge driver's RE-TOKENIZED flag; the producer's finish reason is not
    # recorded per item there (gen_stats holds counts only), so finish_reason stays NULL and the source is named
    assert f1["rationale_artifact_id"] == ji_sha and f1["at_token_cap"] == 1 and f1["finish_reason"] is None and f1["response_tokens"] == 512
    assert f1["extra"]["at_token_cap_source"].startswith("judge_items.at_cap (re-tokenized") and f1["extra"]["at_cap_retokenized"] == 1 and f1["extra"]["at_cap_finish_reason"] is None
    assert orb_flags["orb0000-cccccccc"]["at_token_cap"] == 0
    # the rows.jsonl line the pointer names is that item's first row
    with open(os.path.join(store, "rows.jsonl")) as fh:
        line = fh.readlines()[f1["generation_offset"] - 1]
    assert json.loads(line)["item_id"] == "orb0001-cccccccc"
    # graded scores follow OR_GRADE with the at-cap item at the intermediate grade
    assert orb.items["orb0001-cccccccc"] == (0.5, True) and orb.items["orb0002-cccccccc"] == (1.0, False)
    assert not _validate_all(sr)


def test_judgment_counts_refusals_and_abstention_row(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=4, abstain_one=True)
    sr = _read(store)
    js = {p["run_id"]: p for rt, p in sr.records if rt == "judgment"}
    xs = sr.controls["xstest"]
    j = js[xs.run_id]
    assert j["n_items_judged"] == 3 and j["n_judge_refusals"] == 1 and j["n_errors"] == 0 and j["api_pool"] == "lowprio" and j["status"] == "complete"
    assert j["store_uri"].endswith("judge_items.jsonl") and j["extra"]["rubric_sha16"] == "288e8c08b0c7bbfa"
    abst = [p for rt, p in sr.records if rt == "item_result" and p["run_id"] == xs.run_id and p["item_id"] == "xst0003-cccccccc"]
    assert len(abst) == 1 and abst[0]["label"] is None and abst[0]["value"] is None and abst[0]["extra"]["abstention_reason"] == "refusal"
    states = [p for rt, p in sr.records if rt == "run_state"]
    assert {p["state"] for p in states} == {"judged"} and len(states) == 2
    # no judgment for the rule-scored controls
    assert sr.controls["mmlu"].judge_id is None and sr.controls["gsm8k"].judgment_id == ""


def test_degeneration_rows_one_per_declared_measure(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=3, at_cap_ids=["orb0001-cccccccc", "xst0000-cccccccc"])
    sr = _read(store)
    deg = [p for rt, p in sr.records if rt == "run_degeneration"]
    by_run = {}
    for p in deg:
        by_run.setdefault(p["run_id"], {})[p["measure"]] = p
    for cr in sr.controls.values():
        assert set(by_run[cr.run_id]) == set(IC.DECLARED_MEASURES)
    # the two at-cap estimands are two measures; the single ambiguous `at_token_cap` is never written
    assert "at_token_cap" not in IC.DECLARED_MEASURES and {IC.AT_CAP_RETOKENIZED, IC.AT_CAP_FINISH_REASON} <= set(IC.DECLARED_MEASURES)
    orb = by_run[sr.controls["or_bench_hard_1k"].run_id]
    retok, fin = orb[IC.AT_CAP_RETOKENIZED], orb[IC.AT_CAP_FINISH_REASON]
    # the greedy store carries BOTH: the judge driver's re-tokenized flag (rule from cap_detection.method) and gen_stats' finish reasons
    assert retok["measurement_state"] == "measured" and retok["numerator"] == 1 and retok["denominator"] == 3 and retok["rate"] == pytest.approx(1 / 3)
    assert retok["definition_version"].startswith("re-tokenized generation length >= 511") and retok["denominator_unit"] == "items"
    assert retok["extra"]["n_capped_detected"] == 1 and retok["extra"]["n_capped_gen_stats"] == 1 and retok["threshold_params_json"] == {"cap": 512}
    assert fin["measurement_state"] == "measured" and fin["numerator"] == 1 and fin["denominator"] == 3 and fin["denominator_unit"] == "items"
    assert fin["definition_version"].startswith("gen_stats.finish_reasons[length|max_tokens] / n_total") and fin["extra"]["n_judged"] == 3
    assert orb["union5"]["measurement_state"] == "not_measured" and orb["union5"]["numerator"] is None
    mm = by_run[sr.controls["mmlu"].run_id]
    assert all(p["measurement_state"] == "not_applicable" for p in mm.values())
    gs = by_run[sr.controls["gsm8k"].run_id]
    # GSM8K carries the producer's finish reasons only; no re-tokenized detection exists for the rule-scored control
    assert gs[IC.AT_CAP_FINISH_REASON]["measurement_state"] == "measured" and gs[IC.AT_CAP_FINISH_REASON]["numerator"] == 1 and gs[IC.AT_CAP_FINISH_REASON]["denominator"] == 3
    assert gs[IC.AT_CAP_RETOKENIZED]["measurement_state"] == "not_measured" and "no re-tokenized cap detection" in gs[IC.AT_CAP_RETOKENIZED]["notes"]
    metrics = [p for rt, p in sr.records if rt == "run_metric"]
    assert {p["producer_metric_spelling"] for p in metrics} >= {"acc_aggregate", "xst_refused_bounds_aggregate", "orb_refused_bounds_aggregate"}
    acc = next(p for p in metrics if p["run_id"] == sr.controls["mmlu"].run_id)
    assert acc["metric_id"] == "acc" and acc["computed_by"] == "producer" and acc["n"] == 3
    assert not _validate_all(sr)


def test_sampled_store_carries_the_finish_reason_reading_only(tmp_path):
    """A sampled pass-2 store (as on disk: gen_stats.at_cap_item_ids + finish_reasons, no judge at_cap, no cap_detection):
    the per-item flag and the run measure are the producer's finish reason; the re-tokenized row is not_measured and says so;
    a store with gen_stats ids but no judge records (the Llama mirror) still measures the finish-reason row."""
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=3, sampled=True, at_cap_ids=["orb0001-cccccccc"])
    for at_cap_from_gs in (False, True):
        sr = _read(store, at_cap_from_gen_stats=at_cap_from_gs)
        orb = sr.controls["or_bench_hard_1k"]
        flags = {p["item_id"]: p for rt, p in sr.records if rt == "item_flags" and p["run_id"] == orb.run_id}
        f1 = flags["orb0001-cccccccc"]
        assert f1["at_token_cap"] == 1 and f1["finish_reason"] == "length" and f1["extra"]["at_token_cap_source"] == "gen_stats.at_cap_item_ids (finish_reason)"
        assert f1["extra"]["at_cap_retokenized"] is None and f1["extra"]["at_cap_finish_reason"] == 1
        assert flags["orb0000-cccccccc"]["at_token_cap"] == 0 and flags["orb0000-cccccccc"]["finish_reason"] == "stop"
        deg = {p["measure"]: p for rt, p in sr.records if rt == "run_degeneration" and p["run_id"] == orb.run_id}
        assert deg[IC.AT_CAP_FINISH_REASON]["measurement_state"] == "measured" and deg[IC.AT_CAP_FINISH_REASON]["numerator"] == 1 and deg[IC.AT_CAP_FINISH_REASON]["denominator"] == 3
        assert deg[IC.AT_CAP_FINISH_REASON]["extra"]["n_at_cap_item_ids"] == 1 and deg[IC.AT_CAP_FINISH_REASON]["extra"]["n_at_cap_among_judged"] == 1
        assert deg[IC.AT_CAP_RETOKENIZED]["measurement_state"] == "not_measured" and "no re-tokenized cap detection" in deg[IC.AT_CAP_RETOKENIZED]["notes"]
        # the graded score follows the pinned reader's rule: gen_stats ids for the ladder reader, the (absent) judge flag otherwise
        assert orb.items["orb0001-cccccccc"] == (0.5, at_cap_from_gs)      # partial_refusal grades 0.5 either way; at-cap only under the ladder reader's rule
        assert not _validate_all(sr)
    # a mirror store: gen_stats with the finish-reason set, no judge files at all
    os.remove(os.path.join(store, "judge_items.jsonl")); os.remove(os.path.join(store, "judge_summary.json"))
    sr = _read(store)
    orb = sr.controls["or_bench_hard_1k"]
    assert orb.judge_id is None and not orb.items
    deg = {p["measure"]: p for rt, p in sr.records if rt == "run_degeneration" and p["run_id"] == orb.run_id}
    assert deg[IC.AT_CAP_FINISH_REASON]["measurement_state"] == "measured" and deg[IC.AT_CAP_FINISH_REASON]["numerator"] == 1 and deg[IC.AT_CAP_FINISH_REASON]["judgment_id"] == ""
    flags = {p["item_id"]: p for rt, p in sr.records if rt == "item_flags" and p["run_id"] == orb.run_id}
    assert flags["orb0001-cccccccc"]["at_token_cap"] == 1 and flags["orb0001-cccccccc"]["judgment_id"] == ""
    assert not _validate_all(sr)


def test_none_and_prompted_cells_resolve_objects(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    none_store = F.write_controls_store(view, "none", ihash8="2ff9cdff", object_sha16=None)
    prompted = F.write_controls_store(view, F.PROMPTED_CELL, ihash8="faa4c8cf", object_sha16=None)
    a = _read(none_store)
    b = _read(prompted)
    assert a.object_id == "none" and a.object_kind == "none"
    ra = next(p for rt, p in a.records if rt == "run")
    assert ra["transfer_kind"] == "none" and ra["applied_positions_mode"] == "none" and ra["multi_turn_steering_scope"] == "not_applicable"
    assert b.object_id == F.SI_OBJECT_ID and b.object_kind == "system_instruction"
    rb = next(p for rt, p in b.records if rt == "run")
    assert rb["transfer_kind"] == "prompted_only"


def test_refusals_missing_manifest_symlink_unresolved_model_and_rubric_mismatch(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=2)
    # missing manifest
    os.rename(os.path.join(store, "manifest.json"), os.path.join(store, "manifest.json.bak"))
    with pytest.raises(AB.AdapterRefusal, match="manifest"):
        _read(store)
    os.rename(os.path.join(store, "manifest.json.bak"), os.path.join(store, "manifest.json"))
    # symlinked input
    real = os.path.join(store, "rows.jsonl")
    os.rename(real, real + ".real")
    os.symlink(real + ".real", real)
    with pytest.raises(AB.InputRefusal):
        CS.store_inputs(store)
    os.remove(real)
    os.rename(real + ".real", real)
    # unresolved model spelling: never defaulted
    d = F.dims_map(); d.models.pop(F.HF_IT)
    with pytest.raises(IC.Unresolved, match="hf_repo"):
        _read(store, dims=d)
    # a store judged under a different rubric than the seed judge pins is a different judge -> refusal
    store2 = F.write_controls_store(str(tmp_path / "root2" / "synthmodel"), F.LORA_CELL, n=2, rubric16={"xstest": "0000000000000000"})
    d2 = F.dims_map()
    conn = DB.connect_memory()
    from sisteer.registry.migrations import apply_migrations
    apply_migrations(conn)
    jid = F.judge_ids()["xstest_cpi_v1_sonnet5"]
    conn.execute("INSERT INTO record_ingest(file_uri, record_file_id, file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq) VALUES ('f', 1, ?, '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z', 1, 1)", ("0" * 64,))
    conn.execute("INSERT INTO judge(judge_id, judge_model, judge_role, rubric_name, rubric_version, rubric_sha256, protocol_sha256, score_scale, threshold_rule, judge_backend, extra_json, record_uri, ingest_seq) "
                 "VALUES (?, 'claude-sonnet-5', 'judge', 'x', 'v1', 'unverified', 'unverified', 's', 't', 'live', ?, 'f', 1)",
                 (jid, json.dumps({"seed_judge_id": "xstest_cpi_v1_sonnet5", "rubric_sha16": "288e8c08b0c7bbfa"})))
    d2.conn = conn
    with pytest.raises(AB.AdapterRefusal, match="different rubric"):
        _read(store2, dims=d2)


def test_unmapped_object_sha_is_a_refusal_not_a_default(tmp_path):
    view = str(tmp_path / "root" / "synthmodel")
    store = F.write_controls_store(view, F.LORA_CELL, n=2, object_sha16="ffffffffffffffff")
    with pytest.raises(IC.Unresolved, match="object"):
        _read(store)


def test_adapter_pass_and_ingest_land_rows(tmp_path):
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    view = str(tmp_path / "stores" / "synthmodel")
    F.write_controls_store(view, "none", ihash8="2ff9cdff", object_sha16=None, n=3)
    F.write_controls_store(view, F.LORA_CELL, n=3, at_cap_ids=["orb0001-cccccccc"])
    ad = CS.CellStoreAdapter(roots=[CS.StoreRoot("synthcampaign", str(tmp_path / "stores"), "targets")])
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = RUN.run_pass(root, [ad], conn=conn, now=NOW)
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 2, [r.error for r in rep.results]
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    n_runs = conn.execute("SELECT COUNT(*) FROM run WHERE campaign = 'synthcampaign'").fetchone()[0]
    assert n_runs == 8
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM item_flags WHERE generation_artifact_id IS NOT NULL").fetchone()[0] > 0
    states = dict(conn.execute("SELECT lifecycle_state, COUNT(*) FROM run GROUP BY lifecycle_state").fetchall())
    # every run completes: the four over-refusal runs through the judged path (a complete judgment by the instrument's judge +
    # the completion artifact), the four capability runs (mmlu, gsm8k x 2 stores) as rule-scored runs — their instrument names
    # no judge, so the generation-time headline metric with full coverage is the completion evidence and the harvester stamps
    # them ingested with judged_utc = the metric's computed_utc (else generated_utc) and no judgment row
    assert states == {"ingested": 8}
    judged_path = conn.execute("SELECT COUNT(*) FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id WHERE i.judge_id IS NOT NULL "
                               "AND EXISTS (SELECT 1 FROM judgment j WHERE j.run_id = r.run_id) AND r.judged_utc IS NOT NULL").fetchone()[0]
    assert judged_path == 4
    rule_scored = conn.execute(
        "SELECT r.eval_id, e.eval_category, r.judged_utc = COALESCE((SELECT MAX(rm.computed_utc) FROM run_metric rm WHERE rm.run_id = r.run_id "
        "AND rm.metric_id = e.headline_metric_id AND rm.judgment_id = ''), r.generated_utc), "
        "(SELECT COUNT(*) FROM judgment j WHERE j.run_id = r.run_id) "
        "FROM run r JOIN instrument i ON i.instrument_id = r.instrument_id JOIN eval e ON e.eval_id = r.eval_id "
        "WHERE i.judge_id IS NULL ORDER BY r.eval_id").fetchall()
    assert [tuple(r) for r in rule_scored] == [("gsm8k", "capability", 1, 0)] * 2 + [("mmlu", "capability", 1, 0)] * 2
    assert conn.execute("SELECT COUNT(*) FROM v_in_flight").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id IN (SELECT run_id FROM run WHERE eval_id IN ('mmlu', 'gsm8k'))").fetchone()[0] == 0
    deg = conn.execute("SELECT measurement_state, COUNT(*) FROM run_degeneration GROUP BY measurement_state").fetchall()
    assert dict(deg).get("not_measured", 0) > 0 and dict(deg).get("measured", 0) > 0
    # both at-cap measures land on core vocabulary and read back through the current view
    both = conn.execute("SELECT measure, COUNT(*) FROM v_run_degeneration_current WHERE measurement_state = 'measured' GROUP BY measure").fetchall()
    assert dict(both) == {IC.AT_CAP_RETOKENIZED: 4, IC.AT_CAP_FINISH_REASON: 6}     # 2 stores x (xstest, orb) re-tokenized; + gsm8k finish reasons
    assert conn.execute("SELECT COUNT(*) FROM run_degeneration WHERE measure = 'at_token_cap'").fetchone()[0] == 0
    wide = conn.execute(f"SELECT {IC.AT_CAP_RETOKENIZED}__all_state, {IC.AT_CAP_FINISH_REASON}__all_state FROM v_run_degeneration_wide "
                        "WHERE run_id IN (SELECT run_id FROM run WHERE eval_id = 'or_bench_hard')").fetchall()
    assert wide and all(tuple(r) == ("measured", "measured") for r in wide)
    # the adapter's declared priority rides every accepted revision; one current row per key, no conflicts
    assert [tuple(r) for r in conn.execute("SELECT DISTINCT priority FROM source_revision WHERE adapter = ? AND accepted = 1", (ad.name,)).fetchall()] == [(IC.PRIORITY_GENERIC_READER,)]
    assert conn.execute("SELECT COUNT(*) FROM v_item_conflict").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0] == conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0]
    ptr = conn.execute("SELECT generation_uri, generation_offset FROM v_item_pointer WHERE generation_uri IS NOT NULL LIMIT 1").fetchone()
    assert ptr is not None and ptr[0].endswith("rows.jsonl")
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()


def test_fitted_run_files_the_target_directory_s_dose_and_the_operating_point(tmp_path):
    """A fitted object's runs file the dose the store's target directory spells — `<model>` the campaign nominal 1.0 (the operating point,
    the rule 2026-09-20), `<model>_halfdose` 0.5 (not) — as a `derived` nominal in `nominal_only`; an unsteered store files no
    strength. The served object's dose sidecar (the object row's base_scale) is the second authority: a disagreement refuses the store by
    name, and a `cells`-layout store (no target directory) takes the sidecar's dose alone — with neither, nothing is filed and the row says why."""
    base = F.write_controls_store(str(tmp_path / "root" / "synthmodel"), F.LORA_CELL, n=2)
    half = F.write_controls_store(str(tmp_path / "root" / "synthmodel_halfdose"), F.LORA_CELL, n=2)
    none = F.write_controls_store(str(tmp_path / "root" / "synthmodel"), "none", ihash8="2ff9cdff", object_sha16=None, n=2)
    for store, nominal, is_op in ((base, 1.0, 1), (half, 0.5, 0)):
        sr = _read(store)
        runs = [p for rt, p in sr.records if rt == "run"]
        assert len(runs) == 4
        for p in runs:
            assert (p["strength_nominal"], p["strength_nominal_source"], p["strength_unit"], p["is_operating_point"]) == (nominal, "derived", "nominal_only", is_op)
            assert p["extra"]["dose_served_key"] == nominal and p["field_confidence_json"]["is_operating_point"]["confidence"] == "derived"
            assert p["field_confidence_json"]["strength_unit"]["confidence"] == "unverified"
        assert not _validate_all(sr)
    runs = [p for rt, p in _read(none).records if rt == "run"]
    assert runs and all("strength_nominal" not in p and "is_operating_point" not in p and "strength_unit" not in p for p in runs)
    # the sidecar: a 0.5 dose-scaled copy behind the bare target directory is a disagreement (refused); behind `_halfdose` it agrees
    dims = F.dims_map()
    dims.object_scales[F.OBJ_SHA] = (0.5, "dose_scale")
    with pytest.raises(AB.AdapterRefusal, match="dose sidecar"):
        _read(base, dims=dims)
    runs = [p for rt, p in _read(half, dims=dims).records if rt == "run"]
    assert runs and all(p["strength_nominal"] == 0.5 and p["extra"]["dose_sidecar_base_scale"] == 0.5 and p["extra"]["dose_sidecar_derivation"] == "dose_scale"
                        and "agrees with" in p["field_confidence_json"]["strength_nominal"]["source"] for p in runs)
    # no target directory (a `cells`-layout store): the sidecar alone spells the dose; with neither, nothing is filed and the reason rides the row
    runs = [p for rt, p in CS.StoreReader(dims, "synthcampaign", NOW).read(half, CS.store_inputs(half), view="").records if rt == "run"]
    assert runs and all(p["strength_nominal"] == 0.5 and p["is_operating_point"] == 0 and p["extra"]["dose_served_key"] is None for p in runs)
    bare = CS.StoreReader(F.dims_map(), "synthcampaign", NOW).read(half, CS.store_inputs(half), view="")
    runs = [p for rt, p in bare.records if rt == "run"]
    assert runs and all("strength_nominal" not in p and p["field_confidence_json"]["strength_nominal"]["confidence"] == "unverified" for p in runs)
    assert not _validate_all(bare)


# --------------------------------------------------------------------------------------------
# the anchor currency (EMISSION_VERSION 8): every contrast of the shared store reader is decided by the pass's frame
# --------------------------------------------------------------------------------------------

import _anchor_frame_fixtures as AF  # noqa: E402
from sisteer.registry.adapters import anchor_current as AC  # noqa: E402
from sisteer.registry.adapters import declared_common as DC  # noqa: E402

SRC = "synth_store_source"
HEADLINE = {"xstest": "xst_refused", "or_bench_hard_1k": "orb_refused"}
NONE_LABELS = ["full_compliance", "full_compliance", "full_compliance", "full_refusal"]


def _root(tmp_path, accuracy: bool = False) -> CS.StoreRoot:
    return CS.StoreRoot("synthcampaign", str(tmp_path / "stores"), "targets", run_key_scope="target_dir", at_cap_from_gen_stats=True, source_legacy=SRC,
                        over_refusal_headline=True, accuracy_contrast=accuracy, source_controls=("xstest", "or_bench_hard_1k") + (("mmlu",) if accuracy else ()))


def _stores(tmp_path, *, with_none=("synthmodel",), steered=("synthmodel",), jobs=None, n=4, none_at_cap=(), steered_at_cap=()):
    """A `targets` root: per target directory a steered store (labels compliance / partial / full / abstained) and, where named, the unsteered
    store, every store of one directory naming that directory's serving job. `none_at_cap` / `steered_at_cap` name the unsteered / steered
    stores' items that ran to the token cap (the sampled stores' `gen_stats.at_cap_item_ids`: the per-item cap reading the graded mean scores by)."""
    jobs = dict(jobs or {})
    out = {}
    for tag in sorted(set(with_none) | set(steered)):
        job = jobs.get(tag, "424242")
        d = str(tmp_path / "stores" / tag)
        if tag in steered:
            out[(tag, "steered")] = _with_job(F.write_controls_store(d, F.LORA_CELL, n=n, sampled=True, abstain_one=True, at_cap_ids=list(steered_at_cap)), job)
        if tag in with_none:
            out[(tag, "none")] = _with_job(F.write_controls_store(d, "none", n=n, sampled=True, object_sha16=None, ihash8="2ff9cdff", labels=NONE_LABELS,
                                                                  at_cap_ids=list(none_at_cap)), job)
    return out


def _with_job(store: str, job: str) -> str:
    path = os.path.join(store, "rows.jsonl")
    lines = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    with open(path, "w", encoding="utf-8") as fh:
        for row in lines:
            fh.write(json.dumps(dict(row, slurm_job_id=job), sort_keys=True) + "\n")
    return store


def _adapter(root):
    return CS.CellStoreAdapter(roots=(root,), dims=F.dims_map())


def _read_on(ad, store, frame=None):
    """One store's read on `frame` (or on a frame primed from the adapter's own emissions over every group); the records by type."""
    fr = frame if frame is not None else AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == store]
    recs = [r for b in ad.read(_ctx(fr.con), store, inputs) for r in b.records]
    by = {}
    for r in recs:
        by.setdefault(r["record_type"], []).append(r["payload"])
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    return by, fr


def _contrasts(by, run_id):
    return {c["metric_id"]: c for c in by.get("run_contrast", []) if c["run_id"] == run_id}


def _metrics(by, run_id):
    return {m["metric_id"]: m for m in by["run_metric"] if m["run_id"] == run_id}


def test_the_own_unsteered_store_of_the_directory_is_kept_while_current_and_a_read_without_a_frame_refuses_by_name(tmp_path):
    """The steady state: the same directory's `none` is the own anchor, current on the frame -> the paired per-item records as before, plus
    `extra.anchor_selection` (`own_current`), the steered headline rows stating `delta_block: null`, the group's facts counting the decisions.
    Without a registry connection the read raises `AnchorCurrencyUnavailable` (the runner turns it into a refusal of the GROUP)."""
    stores = _stores(tmp_path)
    ad = _adapter(_root(tmp_path))
    steered = stores[("synthmodel", "steered")]
    with pytest.raises(AC.AnchorCurrencyUnavailable, match="no registry connection"):
        list(ad.read(_ctx(), steered, [i for i in ad.discover() if ad.comparison_group(i) == steered]))
    by, fr = _read_on(ad, steered)
    runs = {p["eval_id"]: p for p in by["run"]}
    none_by, _ = _read_on(ad, stores[("synthmodel", "none")], frame=fr)
    none_runs = {p["eval_id"]: p for p in none_by["run"]}
    assert "run_contrast" not in none_by
    for ctl, ev in (("xstest", "xstest"), ("or_bench_hard_1k", "or_bench_hard")):
        c = _contrasts(by, runs[ev]["run_id"])
        assert set(c) == {HEADLINE[ctl], "over_refusal_graded", "over_refusal_label_mean"}
        for metric_id, p in c.items():
            sel = p["extra"]["anchor_selection"]
            assert p["anchor_run_id"] == none_runs[ev]["run_id"] and p["anchor_kind"] == "same_standup_none" and p["computed_by"] == "registry"
            assert (sel["state"], sel["own_reason"], sel["own_anchor_run_id"], sel["rule"]) == ("own_current", "current", none_runs[ev]["run_id"], AC.RULE)
            assert sel["candidates"] == [none_runs[ev]["run_id"]] and sel["reading"]["judgment_id"] == p["anchor_judgment_id"]
            assert p["extra"]["anchor_store"] == stores[("synthmodel", "none")] and p["extra"]["standup_job_ids"] == ["424242", "424242"]
            assert _metrics(by, runs[ev]["run_id"])[metric_id]["extra"]["delta_block"] is None
        assert c[HEADLINE[ctl]]["delta_method"] == "exact_mcnemar" and c[HEADLINE[ctl]]["delta"] == pytest.approx(100 * (1 / 3 - 1 / 4))
    facts = ad.status_extra(steered)
    assert facts["anchor_states"] == {"own_current": 6} and (facts["contrasts"], facts["contrasts_own"], facts["contrasts_parked"], facts["contrasts_reanchored"]) == (6, 6, 0, 0)
    assert set(facts["contrast_anchors"]) == {"xstest", "or_bench_hard_1k"} and facts["row_refusals"] == [] and len(ad.refusals) == 0


def test_without_an_unsteered_store_the_comparisons_are_decided_a_same_source_store_under_another_directory_is_re_read_and_paired_or_they_park_by_name(tmp_path):
    """A target directory with no `none` store (until now: no contrast, in silence). The comparison is still decided -- `none_declared`, the exact
    key searched: a current unsteered run of this SOURCE under another target directory (another serving job) is the anchor (`sibling_none`), its
    store resolved under the root through the producer key, re-read and paired per item (`extra.paired_from`); retired on the frame, every
    comparison parks BY NAME -- one `row_refusals` line each (the source-state rows carry them), the park on the steered row."""
    stores = _stores(tmp_path, with_none=("synthmodel",), steered=("synthmodel", "synthmodel_b"), jobs={"synthmodel": "111111", "synthmodel_b": "222222"})
    ad = _adapter(_root(tmp_path))
    steered_b = stores[("synthmodel_b", "steered")]
    by, fr = _read_on(ad, steered_b)
    runs = {p["eval_id"]: p for p in by["run"]}
    none_runs = {r[1]: r[0] for r in fr.con.execute("SELECT run_id, eval_id FROM run WHERE object_id = 'none'")}
    assert set(none_runs) >= {"xstest", "or_bench_hard"}
    for ctl, ev in (("xstest", "xstest"), ("or_bench_hard_1k", "or_bench_hard")):
        c = _contrasts(by, runs[ev]["run_id"])
        assert set(c) == {HEADLINE[ctl], "over_refusal_graded", "over_refusal_label_mean"}
        for p in c.values():
            sel = p["extra"]["anchor_selection"]
            assert p["anchor_run_id"] == none_runs[ev] and p["anchor_kind"] == "sibling_none" and (sel["state"], sel["own_reason"], sel["own_anchor_run_id"]) == ("sibling_same_source", "none_declared", None)
            assert p["extra"]["anchor_store"] == stores[("synthmodel", "none")] and stores[("synthmodel", "none")] in p["extra"]["paired_from"]
            assert p["extra"]["cross_source"] is False and p["extra"]["anchor_standup"] == {"lane_source_legacy": SRC, "view_legacy": "synthmodel", "cluster_job_id": "111111"}
            assert p["extra"]["standup_job_ids"] == ["222222", "111111"] and p["extra"]["producer_delta_parked_on_metric"] is False
            assert _metrics(by, runs[ev]["run_id"])[p["metric_id"]]["extra"]["delta_block"] is None
        assert c[HEADLINE[ctl]]["delta_method"] == "exact_mcnemar" and c[HEADLINE[ctl]]["n_pairs"] == 3
    facts = ad.status_extra(steered_b)
    assert facts["anchor_states"] == {"sibling_same_source": 6} and (facts["contrasts"], facts["contrasts_reanchored"], facts["contrasts_parked"]) == (6, 6, 0)
    assert facts["contrast_anchors"]["xstest"]["own_anchor_run_id"] is None and facts["contrast_anchors"]["xstest"]["by_metric"]["xst_refused"]["anchor_run_id"] == none_runs["xstest"]
    # ---- the candidates retired: nothing current at the exact key, every comparison parks by name
    for rid in none_runs.values():
        AF.retire(fr, rid)
    by2, _ = _read_on(ad, steered_b, frame=fr)
    assert "run_contrast" not in by2
    assert len(ad.refusals) == 6 and all(it["key"].endswith("|contrast") and "none_declared" in it["why"] and "contrast has no current anchor at key" in it["why"] for it in ad.refusals.items)
    for ev in ("xstest", "or_bench_hard"):
        for metric_id, m in _metrics(by2, runs[ev]["run_id"]).items():
            if metric_id in (HEADLINE["xstest"], HEADLINE["or_bench_hard_1k"], "over_refusal_graded", "over_refusal_label_mean"):
                blk = m["extra"]["delta_block"]
                assert (blk["anchor"]["state"], blk["anchor"]["own_reason"], blk["anchor"]["own_anchor_run_id"], blk["delta"], blk["n_pairs"]) == ("no_current_anchor", "none_declared", None, None, 0)
                assert "no own pairing" in blk["delta_method"] and blk["rule"] == AC.RULE and [r[0] for r in blk["anchor_selection"]["rejected"]] == [none_runs[ev]]
    facts2 = ad.status_extra(steered_b)
    assert facts2["anchor_states"] == {"no_current_anchor": 6} and facts2["contrasts_parked"] == 6 and len(facts2["row_refusals"]) == 6
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == steered_b]
    rows = [st for st in ad.source_states(_ctx(fr.con), inputs) if st.json_path == "$.row_refusals"]
    assert len(rows) == 1 and len(rows[0].current_members) == 6 and rows[0].source_id == f"{ad.name}:{steered_b}"


def test_a_roster_matched_run_of_another_source_re_anchors_the_headline_unpaired_and_the_item_paired_contrasts_park(tmp_path):
    """The own `none` retired and the paper's roster-matched unsteered run of ANOTHER source at the exact key: the refused-rate headline files
    the registry's unpaired rate difference from the two rows' k / n (`borrowed`, `cross_source`, its own computation row), the label-mean and
    graded contrasts park by name (paired per item over two stores' rows, which another source's run does not carry here), the own pairing's
    numbers park beside the filed headline; the rule-scored accuracy contrast parks on the same re-pick (no count on the row)."""
    stores = _stores(tmp_path)
    ad = _adapter(_root(tmp_path, accuracy=True))
    steered = stores[("synthmodel", "steered")]
    by, fr = _read_on(ad, steered)
    runs = {p["eval_id"]: p for p in by["run"]}
    own = {ev: _contrasts(by, runs[ev]["run_id"]) for ev in ("xstest", "mmlu")}
    assert set(own["mmlu"]) == {"acc"} and own["mmlu"]["acc"]["anchor_kind"] == "same_standup_none" and own["mmlu"]["acc"]["extra"]["anchor_selection"]["state"] == "own_current"
    assert _metrics(by, runs["mmlu"]["run_id"])["acc"]["extra"]["delta_block"] is None
    none_runs = {r[1]: r[0] for r in fr.con.execute("SELECT run_id, eval_id FROM run WHERE object_id = 'none'")}
    for rid in none_runs.values():
        AF.retire(fr, rid)
    others = {}
    for ev, k_a, n_a in (("xstest", 1, 5), ("mmlu", None, 4)):
        run = runs[ev]
        other = f"other:{ev}"
        fr.run(other, model=run["model_id"], eval_id=run["eval_id"], inst=run["instrument_id"], cap=run["token_cap_tokens"], trial=run["trial_index"],
               pass_label=run.get("pass_label"), decode=run.get("decode_policy_id"), source="other_source", standup="othertag", job="999999",
               key="othertag/none__x", in_paper=1, roster=True, judge=run["judge_id"], reading=False)
        jid = fr.judgment(other, judge=run["judge_id"]) if run["judge_id"] else AC.NO_JUDGMENT
        if ev == "xstest":
            fr.metric(other, jid, metric="xst_refused", thr=-1.0, k=k_a, n=n_a, value=k_a / n_a)
            fr.metric(other, jid, metric="over_refusal_graded", thr=-1.0, k=None, n=n_a, value=0.4)
            fr.metric(other, jid, metric="over_refusal_label_mean", thr=-1.0, k=None, n=n_a, value=0.3)
        else:
            fr.metric(other, jid, metric="acc", thr=-1.0, k=None, n=n_a, value=0.5)
        others[ev] = (other, jid)
    by2, _ = _read_on(ad, steered, frame=fr)
    c = _contrasts(by2, runs["xstest"]["run_id"])
    assert set(c) == {"xst_refused"}
    p = c["xst_refused"]
    other, jid = others["xstest"]
    assert p["anchor_run_id"] == other and p["anchor_kind"] == "borrowed" and p["anchor_judgment_id"] == jid and p["computed_by"] == "registry"
    assert p["delta_method"] == DC.UNPAIRED_RATE_METHOD and p["n_pairs"] == 0 and p["delta"] == pytest.approx(100 * (1 / 3 - 1 / 5)) and p["computation_id"] == ""
    ex = p["extra"]
    assert ex["cross_source"] is True and ex["anchor_selection"]["state"] == "borrowed_reference" and ex["anchor_selection"]["own_reason"] == "superseded"
    assert (ex["k_steered"], ex["n_steered"], ex["k_anchor"], ex["n_anchor"]) == (1, 3, 1, 5) and ex["producer_delta_parked_on_metric"] is True
    assert ex["anchor_standup"] == {"lane_source_legacy": "other_source", "view_legacy": "othertag", "cluster_job_id": "999999"} and ex["pairing"].startswith("unpaired: another source")
    comps = {q["computation_id"]: q for q in by2["computation"]}
    assert comps[ex["statistic_computation_id"]]["algorithm"] == "registry.unpaired_rate_contrast"
    mets = _metrics(by2, runs["xstest"]["run_id"])
    assert mets["xst_refused"]["extra"]["delta_block"]["anchor"]["registry_contrast_filed_against"] == other and mets["xst_refused"]["extra"]["delta_block"]["delta"] == pytest.approx(own["xstest"]["xst_refused"]["delta"])
    for metric_id in ("over_refusal_graded", "over_refusal_label_mean"):
        blk = mets[metric_id]["extra"]["delta_block"]
        assert blk["anchor"]["state"] == "borrowed_reference" and blk["anchor"]["registry_contrast_filed_against"] is None and "another source" in blk["why_not_a_contrast_row"]
        assert blk["delta"] == pytest.approx(own["xstest"][metric_id]["delta"])
    # the rule-scored accuracy contrast: a re-pick of another source parks -- the row carries a value and n, never a count
    assert not _contrasts(by2, runs["mmlu"]["run_id"])
    blk = _metrics(by2, runs["mmlu"]["run_id"])["acc"]["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "borrowed_reference" and blk["anchor"]["anchor_run_id"] == others["mmlu"][0] and "derived count is not a count" in blk["why_not_a_contrast_row"]
    assert blk["delta"] == pytest.approx(own["mmlu"]["acc"]["delta"]) and blk["delta_method"] == "exact_mcnemar"
    facts = ad.status_extra(steered)
    assert facts["anchor_states"]["borrowed_reference"] == 4 and facts["contrasts_reanchored"] == 1 and facts["contrasts_parked"] == 3 + facts["anchor_states"].get("no_current_anchor", 0)
    keys = {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items}
    assert {"over_refusal_graded", "over_refusal_label_mean", "acc"} <= keys


# --------------------------------------------------------------------------------------------
# the paired evidence is part of the decision: a label swap in the anchor's store
# --------------------------------------------------------------------------------------------

LABEL_METRIC = {"xstest": "xst_label", "or_bench_hard_1k": "orb_label"}


def _swap_labels(store: str, ctl: str, i: int, j: int) -> None:
    """Re-judge two items of one control in a store, exchanging their labels -- every count and mean of the store stands, the discordant pairs
    move: rows.jsonl's label-metric rows (value + extra_json) and `orb_refused` rows (value) of the two items are exchanged, as are the two
    judge_items.jsonl records' labels."""
    prefix = "xst" if ctl == "xstest" else "orb"
    a, b = f"{prefix}{i:04d}-{'c' * 8}", f"{prefix}{j:04d}-{'c' * 8}"
    path = os.path.join(store, "rows.jsonl")
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    at = {}
    for idx, r in enumerate(rows):
        if r.get("item_id") in (a, b) and r.get("metric") in ("xst_label", "orb_label", "orb_refused"):
            at.setdefault(r["metric"], {})[r["item_id"]] = idx
    assert at and all(set(ids) == {a, b} for ids in at.values()), at
    for ids in at.values():
        ia, ib = ids[a], ids[b]
        for k in ("value", "extra_json"):
            rows[ia][k], rows[ib][k] = rows[ib][k], rows[ia][k]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    jpath = os.path.join(store, "judge_items.jsonl")
    jrows = [json.loads(ln) for ln in open(jpath, encoding="utf-8") if ln.strip()]
    ja = next(r for r in jrows if r["item_id"] == a and r["control"] == ctl)
    jb = next(r for r in jrows if r["item_id"] == b and r["control"] == ctl)
    assert ja["label"] != jb["label"], (ja["label"], jb["label"])
    ja["label"], jb["label"] = jb["label"], ja["label"]
    with open(jpath, "w", encoding="utf-8") as fh:
        for r in jrows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _swap_cap_flags(store: str, ctl: str, i: int, j: int) -> None:
    """Exchange the token-cap flag of two items of one control in a sampled store: item `i` leaves `gen_stats.at_cap_item_ids` and item `j`
    joins it. Every label stands (rows.jsonl and judge_items.jsonl are untouched; a sampled store's judge records carry no `at_cap`), so the
    store's label rows, its label digest and -- when the two items are equally labeled -- its aggregate graded value all stand, while the
    graded mean's paired per-item inputs move: the one change the label rows cannot carry."""
    prefix = "xst" if ctl == "xstest" else "orb"
    a, b = f"{prefix}{i:04d}-{'c' * 8}", f"{prefix}{j:04d}-{'c' * 8}"
    path = os.path.join(store, "gen_stats.json")
    with open(path, encoding="utf-8") as fh:
        gs = json.load(fh)
    ids = list(gs[ctl]["at_cap_item_ids"])
    assert a in ids and b not in ids, (a, b, ids)
    gs[ctl]["at_cap_item_ids"] = [b if x == a else x for x in ids]
    assert gs[ctl]["n_at_cap"] == len(gs[ctl]["at_cap_item_ids"]) and gs[ctl]["finish_reasons"]["length"] == gs[ctl]["n_at_cap"]   # the counts stand
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gs, fh, sort_keys=True, indent=1)
    jpath = os.path.join(store, "judge_items.jsonl")
    assert all("at_cap" not in json.loads(ln) for ln in open(jpath, encoding="utf-8") if ln.strip()), "a sampled store's judge records carry no cap flag"


def _read_logged(ad, store, frame):
    """One store's read on `frame` with the group's decision log captured: `(records by type, the decision log)`."""
    ctx = _ctx(frame.con)
    ctx.begin_group(store)
    inputs = [i for i in ad.discover() if ad.comparison_group(i) == store]
    by = {}
    for b in ad.read(ctx, store, inputs):
        for r in b.records:
            by.setdefault(r["record_type"], []).append(r["payload"])
    return by, list(ctx.decision_log)


def _comparisons(log):
    return [e["comparison"] for e in log]


def test_the_paired_evidence_is_part_of_the_decision_a_label_swap_in_the_anchors_store_parks_until_the_fold_then_refreshes_the_contrast(tmp_path):
    """Every store is its own comparison group, so the unsteered store a paired statistic reads -- the directory's own `none`
    (the OWN anchor) as much as a re-picked same-source store under another directory -- is another group's input. A re-judge of it that
    exchanges two items' labels keeps its headline counts and means (the decision's reading) and changes the discordant pairs; with only the
    reading in the decision the steered group replayed quiet and kept a wrong contrast. Now an own or same-source pairing is decided once
    more with e4 asked at the control's LABEL metric, so the decision carries both stores' current label rows, and the store's rows are
    checked against that digest before the pairing files: on the frame the fold has not caught up with, the own pairing PARKS by name and the
    re-picked one files the headline unpaired from the accepted reading; once the fold holds the re-judged rows the recorded decision set
    replays to another fingerprint (the anchor's items digest moved, its reading did not) and the re-read files the refreshed pairing."""
    stores = _stores(tmp_path, with_none=("synthmodel",), steered=("synthmodel", "synthmodel_b"), jobs={"synthmodel": "111111", "synthmodel_b": "222222"})
    ad = _adapter(_root(tmp_path))
    steered, steered_b, none = stores[("synthmodel", "steered")], stores[("synthmodel_b", "steered")], stores[("synthmodel", "none")]
    fr = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    assert fr.load().items_known
    by0, log0 = _read_logged(ad, steered, fr)
    byb0, logb0 = _read_logged(ad, steered_b, fr)
    xst = {p["eval_id"]: p["run_id"] for p in by0["run"]}["xstest"]
    xst_b = {p["eval_id"]: p["run_id"] for p in byb0["run"]}["xstest"]
    # the decision set: two comparisons per metric row (six rows), the second with e4 asked at the label metric, both sides' digests set
    for log, state in ((log0, "own_current"), (logb0, "sibling_same_source")):
        assert len(log) == 12 and all(e["decision"]["state"] == state for e in log)
        e4 = [e for e in log if e["comparison"]["needs_items"]]
        assert len(e4) == 6 and {e["comparison"]["items_metric_id"] for e in e4} == set(LABEL_METRIC.values())
        assert all(len(e["decision"]["items_digest"]) == 64 and len(e["decision"]["steered_items_digest"]) == 64 and e["decision"]["steered_item_conflicts"] == 0 for e in e4)
        assert all(e["decision"]["items_digest"] is None and e["comparison"]["items_metric_id"] is None for e in log if not e["comparison"]["needs_items"])
    c0, cb0 = _contrasts(by0, xst), _contrasts(byb0, xst_b)
    assert set(c0) == set(cb0) == {"xst_refused", "over_refusal_graded", "over_refusal_label_mean"}
    assert (c0["xst_refused"]["extra"]["discordant_arm_only"], c0["xst_refused"]["extra"]["k_anchor_shared"]) == (1, 0)
    assert cb0["xst_refused"]["delta_method"] == "exact_mcnemar" and none in cb0["xst_refused"]["extra"]["paired_from"]
    fp0, fpb0 = AC.decision_fingerprint(log0), AC.decision_fingerprint(logb0)
    # ---- the unsteered store re-judged: two items' labels exchanged per control (every count and mean of the store unchanged)
    for ctl in ("xstest", "or_bench_hard_1k"):
        _swap_labels(none, ctl, 2, 3)
    # on the frame primed BEFORE the swap (the fold has not caught up): the own pairing parks by name, the pairing's numbers beside it ...
    by1, log1 = _read_logged(ad, steered, fr)
    assert "run_contrast" not in by1 and len(log1) == 12
    assert len(ad.refusals) == 6 and all("not the registry's current rows" in it["why"] and "waits for the fold" in it["why"] for it in ad.refusals.items)
    blk = _metrics(by1, xst)["xst_refused"]["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "own_current" and "not the registry's current rows" in blk["why_not_a_contrast_row"]
    assert blk["delta"] == pytest.approx(c0["xst_refused"]["delta"]) and blk["anchor"]["anchor_run_id"] == c0["xst_refused"]["anchor_run_id"]
    assert ad.status_extra(steered)["anchor_states"] == {"own_current": 6} and ad.status_extra(steered)["contrasts_parked"] == 6
    # ... the recorded decision set replays unchanged on it (nothing folded moved), so no re-read is planned until the fold ...
    assert AC.decision_fingerprint(fr.load().decisions_of(_comparisons(log0))) == fp0
    # ... and the re-picked pairing files the headline UNPAIRED from the registry's accepted reading, the item-paired contrasts parked by name
    byb1, _ = _read_logged(ad, steered_b, fr)
    cb1 = _contrasts(byb1, xst_b)
    assert set(cb1) == {"xst_refused"} and cb1["xst_refused"]["delta_method"] == DC.UNPAIRED_RATE_METHOD and cb1["xst_refused"]["n_pairs"] == 0
    assert "not the registry's current rows" in cb1["xst_refused"]["extra"]["pairing"] and cb1["xst_refused"]["delta"] == pytest.approx(cb0["xst_refused"]["delta"])
    assert {m for m, _ in ((it["key"].rsplit("|", 2)[1], it) for it in ad.refusals.items)} == {"over_refusal_graded", "over_refusal_label_mean"}
    # ---- the fold catches up (a frame primed from the re-judged stores): the recorded set replays to ANOTHER fingerprint -- the anchor's
    # items digest moved while its reading (k / n / value) and the steered side did not -- so the steered groups re-read on the decision input alone
    fr2 = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    replay = fr2.load().decisions_of(_comparisons(log0))
    assert AC.decision_fingerprint(replay) != fp0
    before = {I.canonical_json(e["comparison"]): e["decision"] for e in log0}
    for e in replay:
        d0 = before[I.canonical_json(e["comparison"])]
        assert e["decision"]["reading"] == d0["reading"] and e["decision"]["anchor_run_id"] == d0["anchor_run_id"] and e["decision"]["state"] == "own_current"
        assert e["decision"]["steered_items_digest"] == d0["steered_items_digest"]
        assert (e["decision"]["items_digest"] != d0["items_digest"]) is bool(e["comparison"]["needs_items"])
    assert AC.decision_fingerprint(fr2.load().decisions_of(_comparisons(logb0))) != fpb0
    # the re-reads file the refreshed pairings: the same anchor, the same delta, other discordances and paired means
    by2, log2 = _read_logged(ad, steered, fr2)
    c2 = _contrasts(by2, xst)
    assert set(c2) == set(c0) and c2["xst_refused"]["anchor_run_id"] == c0["xst_refused"]["anchor_run_id"] and not ad.refusals.items
    assert c2["xst_refused"]["delta"] == pytest.approx(c0["xst_refused"]["delta"]) and (c2["xst_refused"]["extra"]["discordant_arm_only"], c2["xst_refused"]["extra"]["k_anchor_shared"]) == (0, 1)
    for m in ("over_refusal_graded", "over_refusal_label_mean"):
        assert c2[m]["delta"] == pytest.approx(c0[m]["delta"]) and c2[m]["extra"]["paired_mean_delta_pp"] != pytest.approx(c0[m]["extra"]["paired_mean_delta_pp"])
    byb2, _ = _read_logged(ad, steered_b, fr2)
    cb2 = _contrasts(byb2, xst_b)
    assert set(cb2) == set(cb0) and cb2["xst_refused"]["delta_method"] == "exact_mcnemar" and none in cb2["xst_refused"]["extra"]["paired_from"]
    assert (cb2["xst_refused"]["extra"]["discordant_arm_only"], cb2["xst_refused"]["extra"]["k_anchor_shared"]) == (0, 1) and not ad.refusals.items
    # quiet after: the new set replays to itself
    assert AC.decision_fingerprint(fr2.load().decisions_of(_comparisons(log2))) == AC.decision_fingerprint(log2)


def _pass(root, ad, now):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = RUN.run_pass(root, [ad], conn=conn, now=now)
    finally:
        conn.close()
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    ing, ing2 = H.ingest_once(root, workers=0, min_free=0), H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    return rep


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def test_a_label_swap_in_the_unsteered_store_re_reads_the_steered_group_on_the_moved_decision_input_alone_end_to_end(tmp_path):
    """Through the runner and the harvester on a temp registry: the two-pass steady state files the directory's own contrasts, the head's
    decision set carrying the e4 comparisons at the label metrics; a re-judge of the unsteered store that exchanges two labels re-reads ITS
    group on its files (the steered group stays quiet in that pass: nothing folded has moved), and the pass after the fold re-reads the
    steered group on the moved decision input alone (no file of its own changed, no bump) and restates the contrast key with the refreshed
    discordances under a new revision; the paper frame shows it once; quiet after."""
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    stores = _stores(tmp_path)
    ad = _adapter(_root(tmp_path))
    steered, none = stores[("synthmodel", "steered")], stores[("synthmodel", "none")]
    stamps = [f"2026-01-05T00:0{i}:00.000Z" for i in range(1, 8)]
    rep1 = _pass(root, ad, stamps[0])
    assert rep1.groups_refreshed == 2 and rep1.anchor_decisions_changed == 0
    assert _q(root, "SELECT COUNT(*) FROM run_contrast")[0][0] == 0                 # no provisional anchors: the first pass parks
    rep2 = _pass(root, ad, stamps[1])                                               # the fold holds the own anchor: the steered group re-reads on the decision input
    assert (rep2.groups_refreshed, rep2.anchor_decisions_changed, rep2.inputs_changed) == (1, 1, 1)
    xst = _q(root, "SELECT run_id FROM run WHERE eval_id = 'xstest' AND object_id <> 'none'")[0][0]
    rows0 = _q(root, "SELECT anchor_run_id, delta, extra_json, anchor_kind FROM run_contrast WHERE run_id = ? AND metric_id = 'xst_refused'", xst)
    assert len(rows0) == 1 and rows0[0][3] == "same_standup_none"
    ex0 = json.loads(rows0[0][2])
    assert ex0["anchor_selection"]["state"] == "own_current" and (ex0["discordant_arm_only"], ex0["k_anchor_shared"]) == (1, 0)
    source_id = RUN.source_id_for(ad.name, steered)
    def head():
        rid, extra = _q(root, "SELECT revision_id, extra_json FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", source_id)[0]
        return rid, json.loads(extra or "{}")["anchor_decisions"]
    rev0, block0 = head()
    assert (block0["n_comparisons"], block0["n_parked"], block0["rule"]) == (12, 0, AC.DECISION_RULE)
    e4 = [c for c in block0["comparisons"] if c["comparison"]["needs_items"]]
    assert len(e4) == 6 and {c["comparison"]["items_metric_id"] for c in e4} == set(LABEL_METRIC.values()) and all(len(c["decision"]["items_digest"]) == 64 for c in e4)
    rep3 = _pass(root, ad, stamps[2])
    assert (rep3.inputs_changed, rep3.groups_refreshed, rep3.anchor_decisions_changed) == (0, 0, 0)
    # ---- the unsteered store re-judged with two labels exchanged: its own group re-reads on its files; the steered group is quiet in that pass
    for ctl in ("xstest", "or_bench_hard_1k"):
        _swap_labels(none, ctl, 2, 3)
    rep4 = _pass(root, ad, stamps[3])
    assert rep4.groups_refreshed == 1 and rep4.anchor_decisions_changed == 0 and rep4.inputs_changed >= 1
    assert head()[0] == rev0
    none_id = rows0[0][0]
    reading = _q(root, "SELECT k, n, value FROM run_metric WHERE run_id = ? AND metric_id = 'xst_refused'", none_id)
    assert len(reading) == 1 and reading[0][0] == 1 and reading[0][1] == 4                  # the anchor's reading stands ...
    labels = dict(_q(root, "SELECT item_id, label FROM v_item_result_current WHERE run_id = ? AND metric_id = 'xst_label'", none_id))
    assert labels[f"xst0002-{'c' * 8}"] == "full_refusal" and labels[f"xst0003-{'c' * 8}"] == "full_compliance"     # ... its item rows moved
    # ---- the pass after the fold: the steered group re-reads on the moved decision input alone and restates the key with the refreshed pairing
    rep5 = _pass(root, ad, stamps[4])
    assert (rep5.groups_refreshed, rep5.anchor_decisions_changed, rep5.inputs_changed) == (1, 1, 1)
    rev1, block1 = head()
    assert rev1 != rev0 and block1["pre_read_fingerprint"] == block1["fingerprint"] != block0["fingerprint"]
    rows1 = _q(root, "SELECT anchor_run_id, delta, extra_json FROM run_contrast WHERE run_id = ? AND metric_id = 'xst_refused'", xst)
    assert len(rows1) == 1 and rows1[0][0] == none_id and rows1[0][1] == pytest.approx(rows0[0][1])
    ex1 = json.loads(rows1[0][2])
    assert (ex1["discordant_arm_only"], ex1["k_anchor_shared"]) == (0, 1) and ex1["anchor_selection"]["state"] == "own_current"
    paper = _q(root, "SELECT anchor_run_id, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'xst_refused'", xst)
    assert paper == [(none_id, 1)]
    keys = _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1", source_id)
    assert keys == [(6,)]
    rep6 = _pass(root, ad, stamps[5])
    assert (rep6.inputs_changed, rep6.groups_refreshed, rep6.anchor_decisions_changed) == (0, 0, 0)


# --------------------------------------------------------------------------------------------
# the cap evidence is part of the decision too: a cap-flag swap in the anchor's store
# --------------------------------------------------------------------------------------------

#: the unsteered store's item 1 of each control ran to the token cap; items 0, 1 and 2 are equally labeled (NONE_LABELS), so moving the flag
#: from item 1 to item 2 keeps every label, every label row and the aggregate graded value, and moves the graded mean's paired inputs alone
NONE_AT_CAP = (f"xst0001-{'c' * 8}", f"orb0001-{'c' * 8}")
XST1, XST2 = f"xst0001-{'c' * 8}", f"xst0002-{'c' * 8}"


def test_the_cap_evidence_is_part_of_the_decision_a_cap_flag_swap_in_the_anchors_store_parks_the_graded_pairing_until_the_fold_then_refreshes_it(tmp_path):
    """The graded mean scores an item by its label AND by whether it ran to the token cap (`or_item_score`: at the cap 0.5
    whatever the label), and the cap reading is filed per item as `item_flags.at_token_cap`, never on a label row -- so a cap flag exchanged
    between two equally labeled items of the unsteered store keeps its aggregate graded value AND its label-row digest while the paired
    Wilcoxon inputs move; with the label rows alone in the decision the steered group replayed quiet and kept a wrong graded contrast. Now the
    graded metric's e4 comparison asks the cap evidence too (both stores' current flags in the decision) and the store's flags are checked
    against it before the graded pairing files: on the frame the fold has not caught up with, the own graded pairing PARKS by name while the
    refused-rate and label-mean contrasts (labels alone) file as before, and a re-picked graded pairing parks the same way; the recorded set
    still replays equal (nothing folded moved); once the fold holds the swapped flags the set replays to another fingerprint -- the graded
    comparisons' anchor flags digest moved, every label digest and every reading did not -- and the re-read files the refreshed graded
    contrast: the same delta (the aggregates stand), another effect size (the pairs moved). Quiet after."""
    stores = _stores(tmp_path, with_none=("synthmodel",), steered=("synthmodel", "synthmodel_b"), jobs={"synthmodel": "111111", "synthmodel_b": "222222"},
                     none_at_cap=NONE_AT_CAP)
    ad = _adapter(_root(tmp_path))
    steered, steered_b, none = stores[("synthmodel", "steered")], stores[("synthmodel_b", "steered")], stores[("synthmodel", "none")]
    fr = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    assert fr.load().items_known and fr.load().flags_known
    by0, log0 = _read_logged(ad, steered, fr)
    byb0, logb0 = _read_logged(ad, steered_b, fr)
    xst = {p["eval_id"]: p["run_id"] for p in by0["run"]}["xstest"]
    xst_b = {p["eval_id"]: p["run_id"] for p in byb0["run"]}["xstest"]
    # the decision set: the cap evidence is asked on the graded metric's e4 comparison alone (one per control), both sides' flags digested there
    for log, state in ((log0, "own_current"), (logb0, "sibling_same_source")):
        assert len(log) == 12 and all(e["decision"]["state"] == state for e in log)
        flagged = [e for e in log if e["comparison"]["needs_flags"]]
        assert len(flagged) == 2 and {e["comparison"]["metric_id"] for e in flagged} == {"over_refusal_graded"} and all(e["comparison"]["needs_items"] for e in flagged)
        assert all(len(e["decision"]["flags_digest"]) == 64 and len(e["decision"]["steered_flags_digest"]) == 64 for e in flagged)
        assert all(e["decision"]["flags_digest"] is None and e["decision"]["steered_flags_digest"] is None for e in log if not e["comparison"]["needs_flags"])
    c0, cb0 = _contrasts(by0, xst), _contrasts(byb0, xst_b)
    assert set(c0) == set(cb0) == {"xst_refused", "over_refusal_graded", "over_refusal_label_mean"}
    g0 = c0["over_refusal_graded"]
    # steered scores 0 / 0.5 / 1 (item 3 abstained); the unsteered 0 / 0.5 (at the cap) / 0 / 1: d = [0, 0, 1], dz = (1/3) / sd = 0.577
    assert g0["effect_size"] == pytest.approx(0.577, abs=1e-3) and g0["extra"]["paired_mean_delta_pp"] == pytest.approx(100.0 / 3) and g0["delta"] == pytest.approx(12.5)
    fp0, fpb0 = AC.decision_fingerprint(log0), AC.decision_fingerprint(logb0)
    none_id = g0["anchor_run_id"]

    def none_graded(frame):
        return frame.con.execute("SELECT value FROM run_metric WHERE run_id = ? AND metric_id = 'over_refusal_graded'", (none_id,)).fetchone()[0]

    def none_flags(frame):
        return dict(frame.con.execute("SELECT item_id, at_token_cap FROM v_item_flags_current WHERE run_id = ? AND judgment_id <> ''", (none_id,)).fetchall())

    assert none_graded(fr) == pytest.approx(0.375) and (none_flags(fr)[XST1], none_flags(fr)[XST2]) == (1, 0)
    # ---- the unsteered store's cap flag moved from item 1 to item 2 (equally labeled), per control: labels, label rows and aggregates unchanged
    for ctl in ("xstest", "or_bench_hard_1k"):
        _swap_cap_flags(none, ctl, 1, 2)
    # on the frame primed BEFORE the swap (the fold has not caught up): the own graded pairing parks by name, its numbers beside it; the refused
    # rate and the label mean (labels alone) file as before ...
    by1, log1 = _read_logged(ad, steered, fr)
    c1 = _contrasts(by1, xst)
    assert set(c1) == {"xst_refused", "over_refusal_label_mean"} and len(log1) == 12
    assert len(ad.refusals) == 2 and all("not the registry's current flags" in it["why"] and "graded pairing" in it["why"] and "waits for the fold" in it["why"]
                                         for it in ad.refusals.items)
    assert {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items} == {"over_refusal_graded"}
    blk = _metrics(by1, xst)["over_refusal_graded"]["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "own_current" and "item_flags.at_token_cap" in blk["why_not_a_contrast_row"] and blk["anchor"]["anchor_run_id"] == none_id
    assert blk["delta"] == pytest.approx(g0["delta"]) and blk["effect_size"] == pytest.approx(1.155, abs=1e-3)
    assert _metrics(by1, xst)["xst_refused"]["extra"]["delta_block"] is None and _metrics(by1, xst)["over_refusal_label_mean"]["extra"]["delta_block"] is None
    assert ad.status_extra(steered)["anchor_states"] == {"own_current": 6} and ad.status_extra(steered)["contrasts_parked"] == 2
    # ... the recorded decision set replays unchanged on it (nothing folded moved), so no re-read is planned until the fold ...
    assert AC.decision_fingerprint(fr.load().decisions_of(_comparisons(log0))) == fp0
    # ... and a re-picked graded pairing parks the same way while the refused rate and the label mean pair over the candidate's store as before
    byb1, _ = _read_logged(ad, steered_b, fr)
    cb1 = _contrasts(byb1, xst_b)
    assert set(cb1) == {"xst_refused", "over_refusal_label_mean"} and cb1["xst_refused"]["delta_method"] == "exact_mcnemar" and none in cb1["xst_refused"]["extra"]["paired_from"]
    assert {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items} == {"over_refusal_graded"} and all("not the registry's current flags" in it["why"] for it in ad.refusals.items)
    # ---- the fold catches up (a frame primed from the swapped store): the recorded set replays to ANOTHER fingerprint -- the graded comparisons'
    # anchor flags digest moved while every label digest, every reading and the steered side did not; the anchor's aggregate graded value stands
    fr2 = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    assert none_graded(fr2) == pytest.approx(none_graded(fr)) and (none_flags(fr2)[XST1], none_flags(fr2)[XST2]) == (0, 1)
    replay = fr2.load().decisions_of(_comparisons(log0))
    assert AC.decision_fingerprint(replay) != fp0
    before = {I.canonical_json(e["comparison"]): e["decision"] for e in log0}
    for e in replay:
        d0 = before[I.canonical_json(e["comparison"])]
        assert e["decision"]["reading"] == d0["reading"] and e["decision"]["anchor_run_id"] == d0["anchor_run_id"] and e["decision"]["state"] == "own_current"
        assert e["decision"]["items_digest"] == d0["items_digest"] and e["decision"]["steered_items_digest"] == d0["steered_items_digest"]
        assert e["decision"]["steered_flags_digest"] == d0["steered_flags_digest"]
        assert (e["decision"]["flags_digest"] != d0["flags_digest"]) is bool(e["comparison"]["needs_flags"])
    assert AC.decision_fingerprint(fr2.load().decisions_of(_comparisons(logb0))) != fpb0
    # the re-reads file the refreshed graded pairing: the same anchor, the same delta and paired mean, another effect size; the other two as before
    by2, log2 = _read_logged(ad, steered, fr2)
    c2 = _contrasts(by2, xst)
    assert set(c2) == set(c0) and not ad.refusals.items
    g2 = c2["over_refusal_graded"]
    assert g2["anchor_run_id"] == none_id and g2["delta"] == pytest.approx(g0["delta"]) and g2["extra"]["paired_mean_delta_pp"] == pytest.approx(g0["extra"]["paired_mean_delta_pp"])
    assert g2["effect_size"] == pytest.approx(1.155, abs=1e-3)
    for m in ("xst_refused", "over_refusal_label_mean"):
        assert c2[m]["delta"] == pytest.approx(c0[m]["delta"]) and c2[m]["effect_size"] == pytest.approx(c0[m]["effect_size"])
    byb2, _ = _read_logged(ad, steered_b, fr2)
    cb2 = _contrasts(byb2, xst_b)
    assert set(cb2) == set(cb0) and cb2["over_refusal_graded"]["effect_size"] == pytest.approx(g2["effect_size"]) and not ad.refusals.items
    assert none in cb2["over_refusal_graded"]["extra"]["paired_from"]
    # quiet after: the new set replays to itself
    assert AC.decision_fingerprint(fr2.load().decisions_of(_comparisons(log2))) == AC.decision_fingerprint(log2)


def test_a_cap_flag_swap_in_the_unsteered_store_re_reads_the_steered_group_on_the_moved_decision_input_alone_end_to_end(tmp_path):
    """Through the runner and the harvester on a temp registry: the steady state files the directory's own contrasts, the graded comparison's
    decision carrying both stores' flags digests; a cap flag of the unsteered store exchanged between two equally labeled items re-reads ITS
    group on its files (the steered group stays quiet in that pass: nothing folded has moved) and leaves its label rows and its aggregate
    graded value exactly as they were; the pass after the fold re-reads the steered group on the moved decision input alone (no file of its
    own changed, no bump) and restates the graded contrast with the same delta and another effect size under a new revision; quiet after."""
    root = str(tmp_path / "reg")
    F.ingest_dimensions(root)
    stores = _stores(tmp_path, none_at_cap=NONE_AT_CAP)
    ad = _adapter(_root(tmp_path))
    steered, none = stores[("synthmodel", "steered")], stores[("synthmodel", "none")]
    stamps = [f"2026-01-05T00:0{i}:00.000Z" for i in range(1, 8)]
    rep1 = _pass(root, ad, stamps[0])
    assert rep1.groups_refreshed == 2 and rep1.anchor_decisions_changed == 0
    rep2 = _pass(root, ad, stamps[1])                                               # the fold holds the own anchor: the steered group re-reads on the decision input
    assert (rep2.groups_refreshed, rep2.anchor_decisions_changed, rep2.inputs_changed) == (1, 1, 1)
    xst = _q(root, "SELECT run_id FROM run WHERE eval_id = 'xstest' AND object_id <> 'none'")[0][0]
    rows0 = _q(root, "SELECT anchor_run_id, delta, effect_size, extra_json FROM run_contrast WHERE run_id = ? AND metric_id = 'over_refusal_graded'", xst)
    assert len(rows0) == 1 and rows0[0][1] == pytest.approx(12.5) and rows0[0][2] == pytest.approx(0.577, abs=1e-3)
    none_id = rows0[0][0]
    source_id = RUN.source_id_for(ad.name, steered)

    def head():
        rid, extra = _q(root, "SELECT revision_id, extra_json FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", source_id)[0]
        return rid, json.loads(extra or "{}")["anchor_decisions"]

    rev0, block0 = head()
    assert (block0["n_comparisons"], block0["n_parked"], block0["rule"]) == (12, 0, AC.DECISION_RULE)
    flagged = [c for c in block0["comparisons"] if c["comparison"]["needs_flags"]]
    assert len(flagged) == 2 and {c["comparison"]["metric_id"] for c in flagged} == {"over_refusal_graded"}
    assert all(len(c["decision"]["flags_digest"]) == 64 and len(c["decision"]["steered_flags_digest"]) == 64 for c in flagged)
    rep3 = _pass(root, ad, stamps[2])
    assert (rep3.inputs_changed, rep3.groups_refreshed, rep3.anchor_decisions_changed) == (0, 0, 0)
    # ---- the unsteered store's cap flag moved between two equally labeled items: its own group re-reads on its files; the steered group is quiet
    graded0 = _q(root, "SELECT value, n FROM run_metric WHERE run_id = ? AND metric_id = 'over_refusal_graded'", none_id)
    labels0 = _q(root, "SELECT item_id, label FROM v_item_result_current WHERE run_id = ? AND metric_id = 'xst_label' ORDER BY item_id", none_id)
    assert graded0 == [(pytest.approx(0.375), 4)] and len(labels0) == 4
    for ctl in ("xstest", "or_bench_hard_1k"):
        _swap_cap_flags(none, ctl, 1, 2)
    rep4 = _pass(root, ad, stamps[3])
    assert rep4.groups_refreshed == 1 and rep4.anchor_decisions_changed == 0 and rep4.inputs_changed >= 1
    assert head()[0] == rev0
    assert _q(root, "SELECT value, n FROM run_metric WHERE run_id = ? AND metric_id = 'over_refusal_graded'", none_id) == graded0          # the aggregate stands ...
    assert _q(root, "SELECT item_id, label FROM v_item_result_current WHERE run_id = ? AND metric_id = 'xst_label' ORDER BY item_id", none_id) == labels0   # ... and the label rows ...
    flags = dict(_q(root, "SELECT item_id, at_token_cap FROM v_item_flags_current WHERE run_id = ? AND judgment_id <> ''", none_id))
    assert (flags[XST1], flags[XST2]) == (0, 1)                                                                                              # ... the cap flags moved
    # ---- the pass after the fold: the steered group re-reads on the moved decision input alone and restates the graded contrast refreshed
    rep5 = _pass(root, ad, stamps[4])
    assert (rep5.groups_refreshed, rep5.anchor_decisions_changed, rep5.inputs_changed) == (1, 1, 1)
    rev1, block1 = head()
    assert rev1 != rev0 and block1["pre_read_fingerprint"] == block1["fingerprint"] != block0["fingerprint"]
    rows1 = _q(root, "SELECT anchor_run_id, delta, effect_size, extra_json FROM run_contrast WHERE run_id = ? AND metric_id = 'over_refusal_graded'", xst)
    assert len(rows1) == 1 and rows1[0][0] == none_id and rows1[0][1] == pytest.approx(rows0[0][1]) and rows1[0][2] == pytest.approx(1.155, abs=1e-3)
    assert json.loads(rows1[0][3])["anchor_selection"]["state"] == "own_current"
    paper = _q(root, "SELECT anchor_run_id, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'over_refusal_graded'", xst)
    assert paper == [(none_id, 1)]
    keys = _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1", source_id)
    assert keys == [(6,)]
    rep6 = _pass(root, ad, stamps[5])
    assert (rep6.inputs_changed, rep6.groups_refreshed, rep6.anchor_decisions_changed) == (0, 0, 0)


# --------------------------------------------------------------------------------------------
# the STEERED side's evidence is checked too: a database-side correction of the steered run's items
# --------------------------------------------------------------------------------------------

#: with five items the steered store's items 0 and 3 are equally labeled (full_compliance; the labels cycle compliance / partial / full) and
#: item 4 abstains; its item 0 ran to the cap, so moving the flag to item 3 keeps every label and the aggregate graded value
STEERED_AT_CAP = (f"xst0000-{'c' * 8}", f"orb0000-{'c' * 8}")
STEERED_TEXT = "the steered store's per-item evidence is not the registry's current evidence of it (an accepted native correction outranks the reader's rows)"
#: a control's eval id (the run payloads' `eval_id`): the controls' spellings differ from the evals' for OR-Bench
EVAL_OF = {ctl: IC.CONTROL_EVALS[ctl][0] for ctl in ("xstest", "or_bench_hard_1k")}


def _ids(prefix: str, *idx: int):
    return [f"{prefix}{i:04d}-{'c' * 8}" for i in idx]


def _db_swap_flags(frame, run_id: str, ctl: str, i: int, j: int) -> None:
    """A database-side correction of the steered run's cap flags (what an accepted NATIVE item_flags record does in the fold, where it
    outranks the reader's rows): item `i` leaves the cap, item `j` joins it; the store's files are untouched."""
    a, b = _ids("xst" if ctl == "xstest" else "orb", i, j)
    assert frame.con.execute("SELECT at_token_cap FROM item_flags WHERE run_id = ? AND item_id = ?", (run_id, a)).fetchone()[0] == 1
    frame.con.execute("UPDATE item_flags SET at_token_cap = 0 WHERE run_id = ? AND item_id = ?", (run_id, a))
    frame.con.execute("UPDATE item_flags SET at_token_cap = 1 WHERE run_id = ? AND item_id = ?", (run_id, b))
    frame.con.commit()


def _db_swap_labels(frame, run_id: str, ctl: str, i: int, j: int) -> None:
    """A database-side correction of the steered run's LABEL rows (an accepted native item_result record): the two items' value and label at
    the control's label metric exchanged; every count, mean and file of the store stands."""
    a, b = _ids("xst" if ctl == "xstest" else "orb", i, j)
    metric = LABEL_METRIC[ctl]
    q = "SELECT value, label FROM item_result WHERE run_id = ? AND metric_id = ? AND item_id = ?"
    ra, rb = frame.con.execute(q, (run_id, metric, a)).fetchone(), frame.con.execute(q, (run_id, metric, b)).fetchone()
    assert ra[1] != rb[1], (ra, rb)
    u = "UPDATE item_result SET value = ?, label = ? WHERE run_id = ? AND metric_id = ? AND item_id = ?"
    frame.con.execute(u, (rb[0], rb[1], run_id, metric, a))
    frame.con.execute(u, (ra[0], ra[1], run_id, metric, b))
    frame.con.commit()


def test_a_database_side_cap_flag_correction_of_the_steered_run_parks_the_graded_pairing_until_the_files_catch_up_or_it_is_withdrawn(tmp_path):
    """The paired statistic reads the STEERED store's rows too (`cr.items`), and an accepted native correction of the steered
    run's items -- native provenance outranks the reader's rows in the fold -- moves the decision's steered digest and re-reads the group while
    the files are unchanged, so the re-read would compute the stale pairing, record the corrected digest and replay quiet. Now the steered
    store's flags are checked against the frame's accepted steered flags digest before the graded pairing files (own and re-picked alike): a
    database-side flag swap between two equally labeled items of the steered run (aggregate unchanged, files unchanged) parks the graded
    pairing by name while the refused rate and the label mean file; the decision records the frame's digests; once the files catch up the
    pairing files with the refreshed effect size, and once the correction is withdrawn it files as before. A frame that holds no steered
    rows yet (the group's first pass after the anchor folded) is not checked."""
    stores = _stores(tmp_path, with_none=("synthmodel",), steered=("synthmodel", "synthmodel_b"), jobs={"synthmodel": "111111", "synthmodel_b": "222222"},
                     n=5, steered_at_cap=STEERED_AT_CAP)
    ad = _adapter(_root(tmp_path))
    steered, steered_b, none = stores[("synthmodel", "steered")], stores[("synthmodel_b", "steered")], stores[("synthmodel", "none")]
    fr = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    by0, log0 = _read_logged(ad, steered, fr)
    byb0, logb0 = _read_logged(ad, steered_b, fr)
    runs0 = {p["eval_id"]: p["run_id"] for p in by0["run"]}
    runs_b0 = {p["eval_id"]: p["run_id"] for p in byb0["run"]}
    xst, xst_b = runs0["xstest"], runs_b0["xstest"]
    jx = {p["run_id"]: p["judgment_id"] for p in by0["judgment"]}[xst]
    c0, cb0 = _contrasts(by0, xst), _contrasts(byb0, xst_b)
    assert set(c0) == set(cb0) == {"xst_refused", "over_refusal_graded", "over_refusal_label_mean"} and not ad.refusals.items
    g0 = c0["over_refusal_graded"]
    # steered 0.5 (cap) / 0.5 / 1 / 0, unsteered 0 / 0 / 0 / 1 over the four items judged in both: d = [0.5, 0.5, 1, -1]
    assert g0["delta"] == pytest.approx(30.0) and g0["extra"]["paired_mean_delta_pp"] == pytest.approx(25.0) and g0["effect_size"] == pytest.approx(0.289, abs=1e-3)
    flagged0 = [e for e in log0 if e["comparison"]["needs_flags"] and e["comparison"]["run_id"] == xst]
    assert len(flagged0) == 1 and flagged0[0]["decision"]["steered_flags_digest"] == fr.load().pair_flags_digest(xst, jx)
    # ---- the correction: the steered run's cap flag moved from item 0 to item 3 (equally labeled) in the REGISTRY alone, for both stores
    for ctl in ("xstest", "or_bench_hard_1k"):
        _db_swap_flags(fr, runs0[EVAL_OF[ctl]], ctl, 0, 3)
        _db_swap_flags(fr, runs_b0[EVAL_OF[ctl]], ctl, 0, 3)
    corrected = fr.load().pair_flags_digest(xst, jx)
    assert corrected != flagged0[0]["decision"]["steered_flags_digest"]
    # the own pairing: the graded contrast parks by name (the steered store's flags are not the registry's), the other two file; the decision
    # records the FRAME's digest, not the store's
    by1, log1 = _read_logged(ad, steered, fr)
    c1 = _contrasts(by1, xst)
    assert set(c1) == {"xst_refused", "over_refusal_label_mean"} and c1["xst_refused"]["delta"] == pytest.approx(c0["xst_refused"]["delta"])
    assert len(ad.refusals) == 2 and {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items} == {"over_refusal_graded"}
    assert all(STEERED_TEXT in it["why"] and "steered store's per-item cap flags" in it["why"] and "graded pairing waits" in it["why"] for it in ad.refusals.items)
    blk = _metrics(by1, xst)["over_refusal_graded"]["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "own_current" and STEERED_TEXT in blk["why_not_a_contrast_row"] and blk["effect_size"] == pytest.approx(g0["effect_size"])
    flagged1 = [e for e in log1 if e["comparison"]["needs_flags"] and e["comparison"]["run_id"] == xst]
    assert flagged1[0]["decision"]["steered_flags_digest"] == corrected and flagged1[0]["decision"]["items_digest"] == flagged0[0]["decision"]["items_digest"]
    assert ad.status_extra(steered)["contrasts_parked"] == 2 and ad.status_extra(steered)["anchor_states"] == {"own_current": 6}
    # the re-picked pairing (another directory's steered store, the same correction): the graded contrast parks the same way, the refused
    # rate and the label mean pair over the candidate's store as before
    byb1, _ = _read_logged(ad, steered_b, fr)
    cb1 = _contrasts(byb1, xst_b)
    assert set(cb1) == {"xst_refused", "over_refusal_label_mean"} and cb1["xst_refused"]["delta_method"] == "exact_mcnemar" and none in cb1["xst_refused"]["extra"]["paired_from"]
    assert {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items} == {"over_refusal_graded"} and all(STEERED_TEXT in it["why"] for it in ad.refusals.items)
    assert all("against the re-picked " in it["why"] for it in ad.refusals.items)
    assert any(f"re-picked {cb1['xst_refused']['anchor_run_id']} (" in it["why"] for it in ad.refusals.items)     # the xstest one names the xstest anchor
    # ---- the files catch up (the steered store re-flagged the same way): the graded pairing files, refreshed -- the same delta and paired
    # mean (the aggregates stand), another effect size (d = [0, 0.5, 1, -0.5])
    for ctl in ("xstest", "or_bench_hard_1k"):
        _swap_cap_flags(steered, ctl, 0, 3)
    by2, log2 = _read_logged(ad, steered, fr)
    c2 = _contrasts(by2, xst)
    assert set(c2) == set(c0) and not ad.refusals.items
    g2 = c2["over_refusal_graded"]
    assert g2["delta"] == pytest.approx(g0["delta"]) and g2["extra"]["paired_mean_delta_pp"] == pytest.approx(25.0) and g2["effect_size"] == pytest.approx(0.387, abs=1e-3)
    assert [e for e in log2 if e["comparison"]["needs_flags"] and e["comparison"]["run_id"] == xst][0]["decision"]["steered_flags_digest"] == corrected
    # ---- or the correction is withdrawn (the other store): the frame's digest returns to the store's and the pairing files as before
    for ctl in ("xstest", "or_bench_hard_1k"):
        _db_swap_flags(fr, runs_b0[EVAL_OF[ctl]], ctl, 3, 0)
    byb2, _ = _read_logged(ad, steered_b, fr)
    cb2 = _contrasts(byb2, xst_b)
    assert set(cb2) == set(cb0) and not ad.refusals.items and cb2["over_refusal_graded"]["effect_size"] == pytest.approx(cb0["over_refusal_graded"]["effect_size"])
    # ---- the first-pass allowance: a frame holding the anchor but no steered rows yet carries no steered digests and checks nothing
    groups = AF.groups_of(ad)
    fr_none = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent", groups={none: groups[none]})
    by3, log3 = _read_logged(ad, steered, fr_none)
    assert set(_contrasts(by3, xst)) == set(c0) and not ad.refusals.items
    assert all(e["decision"]["steered_items_digest"] is None and e["decision"]["steered_flags_digest"] is None for e in log3 if e["comparison"]["needs_items"])


def test_a_database_side_label_correction_of_the_steered_run_parks_every_pairing_until_the_files_catch_up(tmp_path):
    """The same gap for the LABEL rows: an accepted native correction exchanging two labels of the steered run keeps every count and mean
    (and every file) while the discordant pairs move. The steered store's label rows are checked against the frame's `steered_items_digest`
    before any pairing files: the own refused-rate, label-mean and graded pairings all park by name, the re-picked headline files unpaired
    from the accepted readings while its item-paired contrasts park, and once the files catch up every pairing files with the refreshed
    discordances."""
    stores = _stores(tmp_path, with_none=("synthmodel",), steered=("synthmodel", "synthmodel_b"), jobs={"synthmodel": "111111", "synthmodel_b": "222222"}, n=5)
    ad = _adapter(_root(tmp_path))
    steered, steered_b, none = stores[("synthmodel", "steered")], stores[("synthmodel_b", "steered")], stores[("synthmodel", "none")]
    fr = AF.primed_frame(ad, now=NOW, producer=dict(_ctx().producer), root="/nonexistent")
    by0, log0 = _read_logged(ad, steered, fr)
    byb0, logb0 = _read_logged(ad, steered_b, fr)
    runs0 = {p["eval_id"]: p["run_id"] for p in by0["run"]}
    runs_b0 = {p["eval_id"]: p["run_id"] for p in byb0["run"]}
    xst, xst_b = runs0["xstest"], runs_b0["xstest"]
    jx = {p["run_id"]: p["judgment_id"] for p in by0["judgment"]}[xst]
    c0, cb0 = _contrasts(by0, xst), _contrasts(byb0, xst_b)
    assert set(c0) == set(cb0) == {"xst_refused", "over_refusal_graded", "over_refusal_label_mean"} and not ad.refusals.items
    e4_0 = [e for e in log0 if e["comparison"]["needs_items"] and e["comparison"]["run_id"] == xst]
    assert len(e4_0) == 3 and all(e["decision"]["steered_items_digest"] == fr.load().pair_items_digest("xst_label", xst, jx) for e in e4_0)
    # ---- the correction: items 2 (full refusal) and 3 (compliance) of the steered run exchange their labels in the REGISTRY alone, both stores
    # (the unsteered store's item 3 is its one refusal, so the discordances move while every count stands)
    for ctl in ("xstest", "or_bench_hard_1k"):
        _db_swap_labels(fr, runs0[EVAL_OF[ctl]], ctl, 2, 3)
        _db_swap_labels(fr, runs_b0[EVAL_OF[ctl]], ctl, 2, 3)
    corrected = fr.load().pair_items_digest("xst_label", xst, jx)
    assert corrected != e4_0[0]["decision"]["steered_items_digest"]
    # the own pairings all park by name (every statistic reads the labels); the decisions record the frame's digest
    by1, log1 = _read_logged(ad, steered, fr)
    assert "run_contrast" not in by1 and len(ad.refusals) == 6
    assert all(STEERED_TEXT in it["why"] and "steered store's item rows" in it["why"] and "the pairing waits" in it["why"] for it in ad.refusals.items)
    assert {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items} == {"xst_refused", "orb_refused", "over_refusal_graded", "over_refusal_label_mean"}
    blk = _metrics(by1, xst)["xst_refused"]["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "own_current" and STEERED_TEXT in blk["why_not_a_contrast_row"] and blk["delta"] == pytest.approx(c0["xst_refused"]["delta"])
    assert all(e["decision"]["steered_items_digest"] == corrected for e in log1 if e["comparison"]["needs_items"] and e["comparison"]["run_id"] == xst)
    # the re-picked pairing: the headline files UNPAIRED from the accepted readings (the plan's allowance), the item-paired contrasts park
    byb1, _ = _read_logged(ad, steered_b, fr)
    cb1 = _contrasts(byb1, xst_b)
    assert set(cb1) == {"xst_refused"} and cb1["xst_refused"]["delta_method"] == DC.UNPAIRED_RATE_METHOD and cb1["xst_refused"]["n_pairs"] == 0
    assert STEERED_TEXT in cb1["xst_refused"]["extra"]["pairing"] and cb1["xst_refused"]["delta"] == pytest.approx(cb0["xst_refused"]["delta"])
    assert {it["key"].rsplit("|", 2)[1] for it in ad.refusals.items} == {"over_refusal_graded", "over_refusal_label_mean"} and all(STEERED_TEXT in it["why"] for it in ad.refusals.items)
    # ---- the files catch up (the steered stores re-judged the same way): every pairing files, the same deltas, other discordances
    for ctl in ("xstest", "or_bench_hard_1k"):
        _swap_labels(steered, ctl, 2, 3)
        _swap_labels(steered_b, ctl, 2, 3)
    by2, log2 = _read_logged(ad, steered, fr)
    c2 = _contrasts(by2, xst)
    assert set(c2) == set(c0) and not ad.refusals.items
    assert c2["xst_refused"]["delta"] == pytest.approx(c0["xst_refused"]["delta"]) and c2["xst_refused"]["extra"]["discordant_arm_only"] != c0["xst_refused"]["extra"]["discordant_arm_only"]
    assert all(e["decision"]["steered_items_digest"] == corrected for e in log2 if e["comparison"]["needs_items"] and e["comparison"]["run_id"] == xst)
    byb2, _ = _read_logged(ad, steered_b, fr)
    cb2 = _contrasts(byb2, xst_b)
    assert set(cb2) == set(cb0) and cb2["xst_refused"]["delta_method"] == "exact_mcnemar" and not ad.refusals.items
