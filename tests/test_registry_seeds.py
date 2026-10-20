"""The dimension seeds (`adapters/dimensions.py`, decode policy, B0/G rows): every seed parses and
loads with zero refusals; a seed row missing a required field is refused NAMING the field (the loader never fills);
every id the seeds reference exists (dimension closure); exactly one default per decode-policy key with the known
no-default key pinned; the committed export equals a fresh registry's export and `check` catches drift; the CLI path.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sqlite3

import pytest
import yaml

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import decode_policy as DP
from sisteer.registry import harvest as H
from sisteer.registry import identity
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
EXPORT = os.path.join(ROOT, "configs", "registry", "decode_policy.json")
PROD = synth.producer("tests.registry_seeds")


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def bundle(seeds):
    return D.build_records(seeds, PROD, now="2026-09-16T12:00:00.000Z")


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """A fresh registry with the seeds ingested through the harvester (the real path)."""
    root = str(tmp_path_factory.mktemp("reg") / "root")
    H.ensure_layout(root)
    bundle, paths = D.write_seed_records(root, SEEDS, producer=PROD, now="2026-09-16T12:00:00.000Z")
    assert paths, "nothing written"
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    assert rep.blocked == 0
    return root, bundle


# --------------------------------------------------------------------------------------------
# the non-template variants of the two multi-turn instruments (qwen35_35b_a3b)
# --------------------------------------------------------------------------------------------

#: variant -> the 35B's canonical every-token seed: since the 2026-10-14 split by decode route, the vendor-preset seed (per_goal_turn), which
#: is where the 35B's `|multi_turn|off|vendor_preset` row lives; the generation_config seed keeps the paper's sends-nothing route alone
NTPL_SEEDS = {"multiturn_actorattack_cap4096_generation_config_non_template": "multiturn_actorattack_cap4096_vendor_preset",
              "multiturn_crescendo_cap4096_generation_config_non_template": "multiturn_crescendo_cap4096_vendor_preset"}


def test_the_non_template_seeds_mint_exactly_two_instrument_rows_differing_from_the_canonical_35b_rows_in_the_mode_alone(bundle, loaded):
    """`dimensions load` mints exactly two new instrument rows from the two seeds: each for qwen35_35b_a3b alone (its one decode policy
    ref), `applied_positions_mode non_template`, `paper_status sensitivity` (never canonical), `variant_of` the canonical seed; its
    identity is the canonical 35B row's with the mode swapped and nothing else (`applied_positions_mode` is an identity field)."""
    rows = {r["payload"]["extra"]["seed_instrument_id"]: r["payload"] for r in bundle.records["instrument"]
            if r["payload"]["extra"]["seed_instrument_id"] in NTPL_SEEDS}
    assert set(rows) == set(NTPL_SEEDS)
    assert sum(1 for r in bundle.records["instrument"] if r["payload"]["extra"]["seed_instrument_id"] in NTPL_SEEDS) == 2
    canonical = {(r["payload"]["eval_id"], r["payload"]["extra"].get("model_id")): r["payload"] for r in bundle.records["instrument"]
                 if r["payload"]["extra"]["seed_instrument_id"] in NTPL_SEEDS.values()}
    for seed_id, variant in rows.items():
        extra = variant["extra"]
        assert extra["model_id"] == "qwen35_35b_a3b" and extra["paper_status"] == "sensitivity" and extra["variant_of"] == NTPL_SEEDS[seed_id]
        assert variant["applied_positions_mode"] == "non_template" and variant["serving_path"] == "lora_native"
        base = canonical[(variant["eval_id"], "qwen35_35b_a3b")]
        assert base["applied_positions_mode"] == "all" and base["extra"]["paper_status"] == "canonical"
        differing = {k for k in set(variant) | set(base) if k not in ("instrument_id", "extra", "notes") and variant.get(k) != base.get(k)}
        assert differing == {"applied_positions_mode"}, differing
        swapped = identity.instrument_identity_fields(base)
        swapped["applied_positions_mode"] = "non_template"
        assert identity.instrument_id(swapped) == variant["instrument_id"] != base["instrument_id"]
        assert "not chat-template scaffold" in variant["notes"].lower() and "same rule on the realized ids" in variant["notes"].lower()
    # loaded through the harvester: both rows in the table with the mode, beside the canonical rows
    root, _ = loaded
    con = sqlite3.connect(f"file:{DB.db_path(root)}?mode=ro", uri=True)
    try:
        got = {r[0]: r[1] for r in con.execute("SELECT json_extract(extra_json, '$.seed_instrument_id'), applied_positions_mode FROM instrument "
                                               "WHERE json_extract(extra_json, '$.seed_instrument_id') IN (?, ?)", tuple(NTPL_SEEDS)).fetchall()}
    finally:
        con.close()
    assert got == {s: "non_template" for s in NTPL_SEEDS}


# --------------------------------------------------------------------------------------------
# parse + counts + strictness
# --------------------------------------------------------------------------------------------


def test_every_seed_parses_and_loads_without_refusal(bundle):
    counts = bundle.counts()
    # 34 drafted evals + `em8q` (the Betley 8-question benchmark as ITS OWN eval, 2026-09-17: the Qwen3.5-4B EM organisms are model rows and
    # the benchmark is the eval, so one eval serves every organism; the Llama-era `em_<panel>` evals stay)
    # 15 drafted models + the three Qwen3.5-4B EM organisms (2026-09-17: checkpoint_kind organism, parent qwen35_4b_it)
    # 22 drafted item sets + `mmlu_1531` (2026-09-21: the full-width MMLU read's whole pinned pool, the one item set whose ids sha is measured;
    # mmlu_200 is its first 200 items, item for item -- two item sets, two instruments, never pooled)
    # item_set 23 -> 25: `bigcodebench_full_1140` + `bfcl_full_3981`, seeded on BOTH sides and merged 2026-09-24: the
    # FULL-suite BigCodeBench / BFCL passes, the whole population of each 200 sibling's pin, no seed; their ids digests contracts.ids_sha256 of the pin
    # files at 8ba0b48cf -- two more item sets, two more instruments, never pooled with the 200 rows
    # model 18 -> 22 (2026-09-29): the paper-companion-repo PIPELINE-TEST checkpoints qwen35_35b_a3b_base / qwen35_35b_a3b and gemma4_31b_base /
    # gemma4_31b (paper_status reference; the served gemma id carries no _it suffix) -- tests/test_registry_seeds_test_models.py
    # item_set 25 -> 31 (2026-10-01): the six per-behaviour Petri sets `petri_scenarios_<behaviour>` (10 each), the partition of
    # petri_scenarios_6x10 by behaviour -- a per-behaviour Petri run's planned count is its ten scenarios, not the pooled sixty
    # model 22 -> 24 (2026-10-08): the front-door WORKED EXAMPLE's key qwen35_4b_demo / qwen35_4b_demo_base -- the SAME Qwen3.5-4B
    # pins as qwen35_4b_it / qwen35_4b_base under a key of their own (paper_status reference), so the recipe's phase names s1 / d0p75 / d0p75_relu
    # (one steered run and its ladder) are never spelled like the paper's qwen35_4b_s<s> replicates -- tests/test_registry_seeds_demo_model.py
    assert counts["model"] == 24 and counts["eval"] == 35 and counts["item_set"] == 31
    # 12 drafted judges + the Swap-B Opus over-refusal regrade judge
    assert counts["judge"] == 13
    # 4 paper texts + 1 reference text (si_caa) + the 3 candidate-SI teachers; the HHH reference
    # text left with the unlicensed HHH prompt
    assert counts["si_text"] == 8
    # the objects no producer owns: `none` + one prompted `si:<sha>` per si_text row
    assert counts["object"] == 1 + counts["si_text"]
    # 70 drafted + 3 loader-pass rows + 2 Qwen vendor-preset TARGET rows (multi-turn, Petri; 2026-09-17) + 5 `<model>|bon|off|vendor_preset`
    # rows (2026-09-17, decode-bon-vendor-preset; Qwen's is ONE row for two producers) + 2 `qwen35_4b_it|em8q|off|*` rows (2026-09-17, the EM
    # organisms panel: the vendor_preset default and the T = 1.0 record rung; ruling decode-vendor-preset-default-2026-09-17) + 16 Llama
    # `not_available` twins (2026-09-18, ruling decode-llama-thinking-not-available-2026-09-18: one per llama8b_it `off` row, the 9 explicit
    # and the 7 wildcard-expanded; tests/test_registry_decode_llama_thinking_na.py) + 1 `nemotron3nano4b|multi_turn|off|vendor_preset@v1`
    # (2026-09-20: the TARGET row of the paired ReLU multi-turn runs SENT on Nemotron -- the yaml's vendor row 1.0 / 1.0, verified from all 144
    # shards; non-default, the key's default stays the sends-nothing generation_config row), plus 7 wildcard rows over the 5 served models
    # + 7 rows of the ruling of 2026-09-21 (decode-vendor-default-all-evals-2026-09-21: the vendor preset is the default of EVERY
    # family): `<model>|agentic|off|vendor` for gemma4e4b_it / glm46vflash / nemotron3nano4b / llama8b_it, llama8b_it|em_organism|off|vendor,
    # and the two Llama not_available twins; the runner_forced_T1 rows stay, non-default
    # + 20 rows of 2026-09-29 (ruling decode-qwen35-35b-a3b-shipped-generation-config-2026-09-29): TEN per pipeline-test model (qwen35_35b_a3b, gemma4_31b),
    # each from ITS shipped generation_config.json -- static / over_refusal / multi_turn / petri vendor_preset, bon vendor_preset, agentic vendor, gsm8k ON
    # vendor_preset, and EXPLICIT mmlu greedy + bigcodebench / bfcl protocol rows (the `*` wildcards expand over in_paper / descoped served models only)
    # + 27 rows of 2026-10-09 (ruling decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09): the worked example
    # qwen35_4b_demo's twin of EVERY explicit qwen35_4b_it row, identical in every parameter (same_values_as the 4B's row; the `*` wildcards
    # still reach served models only) -- tests/test_registry_seeds_demo_instruments.py
    # + 1 (2026-10-09, the same ruling widened): the worked example's explicit `qwen35_4b_demo|mmlu|off|greedy` row --
    # the recipe runs MMLU for every key and registers a front-door key under mmlu_letter_prefill_full_1531, which names the row (never the 200-item one)
    assert counts["decode_policy"] == 83 + 16 + 7 + 7 * 5 + 20 + 2 + 27 + 1     # + 2: the test models' canonical-T1 BoN rows
    # 24 -> 26 (ruling decode-thinking-card-mapping-pipeline-test-models-2026-09-30): the two pipeline-test models'
    # (model_key, thinking_card) -> `<model>|gsm8k|on|vendor_preset` entries (tests/test_gsm8k_thinking_card_row_test_models.py)
    # 26 -> 31 (2026-10-09, the same ruling): the worked example's (qwen35_4b_demo, mode) twins of the five (qwen35_4b, mode) entries
    assert counts["sampling_mode_policy"] == 31
    assert counts["instrument"] > 27, "every seed row spanning several evals / models expands"


def test_ingested_row_counts_equal_records(loaded):
    root, bundle = loaded
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        for table, n in bundle.counts().items():
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == n, table
    finally:
        conn.close()


def test_missing_required_field_is_refused_naming_the_field(seeds):
    s = copy.deepcopy(seeds)
    del s.models[0]["family"]
    with pytest.raises(D.SeedRefusal) as ei:
        D.build_records(s, PROD)
    assert ei.value.field == "family" and ei.value.seed_file == "models.yaml" and ei.value.row == s.models[0]["model_id"]
    assert "required field 'family' missing" in ei.value.why


def test_out_of_vocabulary_value_is_refused_naming_the_column(seeds):
    s = copy.deepcopy(seeds)
    s.evals[0]["eval_category"] = "not-a-category"
    with pytest.raises(D.SeedRefusal) as ei:
        D.build_records(s, PROD)
    assert ei.value.field == "eval_category" and "EVAL_CATEGORIES" in ei.value.why


def test_an_instrument_naming_an_eval_without_its_item_set_is_refused(seeds):
    s = copy.deepcopy(seeds)
    row = next(r for r in s.instruments if r["instrument_id"] == "mmlu_letter_prefill_200")
    row["item_set_ids"] = []
    with pytest.raises(D.SeedRefusal) as ei:
        D.build_records(s, PROD)
    assert ei.value.field == "item_set_ids" and "never named without its version" in ei.value.why


def test_unverified_on_a_column_that_cannot_spell_it_is_refused(seeds):
    s = copy.deepcopy(seeds)
    row = next(r for r in s.instruments if r["instrument_id"] == "mmlu_letter_prefill_200")
    row["thinking_mode"] = "unverified"
    with pytest.raises(D.SeedRefusal) as ei:
        D.build_records(s, PROD)
    assert ei.value.field == "thinking_mode" and "cannot spell 'unverified'" in ei.value.why


def test_loader_never_fills_a_value(seeds):
    """Every payload column is a seed value or a mapping of one: no key appears in a record that the seed row
    (or the loader's documented derivations: hashed ids, date normalisation, category spellings) did not supply."""
    b = D.build_records(seeds, PROD)
    derived = {"model": set(), "eval": {"producer_spellings_json"}, "item_set": set(),
               "judge": {"judge_id", "label_vocabulary_json"}, "si_text": {"si_token_count_by_model_json"}}
    for table, extra_ok in derived.items():
        seed_rows = {"model": seeds.models, "eval": seeds.evals, "item_set": seeds.item_sets, "judge": seeds.judges,
                     "si_text": seeds.si_texts}[table]
        seed_keys = set().union(*(set(r) for r in seed_rows)) | extra_ok | {"field_confidence_json", "extra"}
        for rec in b.records[table]:
            unexplained = set(rec["payload"]) - seed_keys
            assert not unexplained, (table, unexplained)


def test_field_confidence_carries_inferred_and_unverified(bundle):
    by_id = {r["payload"]["model_id"]: r["payload"] for r in bundle.records["model"]}
    fc = json.loads(by_id["llama3b_it"]["field_confidence_json"]) if isinstance(by_id["llama3b_it"]["field_confidence_json"], str) \
        else by_id["llama3b_it"]["field_confidence_json"]
    assert fc["family"]["confidence"] == "inferred" and "vocabulary addition" in fc["family"]["reason"]
    fcb3 = by_id["llama3b_base"]["field_confidence_json"]
    fcb3 = fcb3 if isinstance(fcb3, dict) else json.loads(fcb3)
    assert fcb3["hf_revision"]["confidence"] == "unverified" and "README.md" in fcb3["hf_revision"]["reason"]
    assert fcb3["row_existence"]["confidence"] == "inferred" and "no pin found" in fcb3["row_existence"]["reason"]
    # an INTEGER column's `unverified` is NULL with the reason kept
    base = by_id["gemma4e4b_base"]
    assert base["generation_config_present"] is None
    fcb = base["field_confidence_json"] if isinstance(base["field_confidence_json"], dict) else json.loads(base["field_confidence_json"])
    assert fcb["generation_config_present"]["confidence"] == "unverified"
    # a ruling FK stays NULL until the rulings are seeded; the text is kept
    llama = by_id["llama8b_it"]
    assert llama["paper_status_ruling_id"] is None and "Llama added when GLM dropped" in llama["extra"]["paper_status_ruling_id_note"]


def test_dates_are_normalised_to_utc_ms_with_their_precision(bundle):
    gemma = next(r["payload"] for r in bundle.records["model"] if r["payload"]["model_id"] == "gemma4e4b_it")
    assert gemma["generation_config_source_utc"] == "2026-09-11T00:00:00.000Z"
    assert gemma["extra"]["generation_config_source_utc_confidence"] == "date_only"
    si = next(r["payload"] for r in bundle.records["si_text"] if r["payload"]["si_id"] == "si_constitution")
    assert si["first_used_utc"] == "2026-08-18T00:00:00.000Z"


def test_records_are_deterministic(seeds):
    a = D.build_records(seeds, PROD, now="2026-09-16T12:00:00.000Z")
    b = D.build_records(seeds, PROD, now="2026-09-16T12:00:00.000Z")
    assert [r["record_id"] for r in a.all_records()] == [r["record_id"] for r in b.all_records()]
    assert a.revision_id == b.revision_id == seeds.revision_id()


# --------------------------------------------------------------------------------------------
# closure
# --------------------------------------------------------------------------------------------


def _pipes(text: str):
    return [t.strip() for t in str(text).split(" | ")]


def test_dimension_closure(seeds, bundle):
    models = {m["model_id"] for m in seeds.models}
    evals = {e["eval_id"] for e in seeds.evals}
    sets = {s["item_set_id"] for s in seeds.item_sets}
    judges = {j["judge_id"] for j in seeds.judges}
    idx = bundle.policies
    # decode policies -> models / evals
    for p in seeds.decode["policies"]:
        assert p["model_id"] == "*" or p["model_id"] in models, p["policy_ref"]
        if p.get("eval_id"):
            assert p["eval_id"] in evals, p["policy_ref"]
    # every mode mapping's model key resolves to a seeded model
    mmap = D.legacy_model_map(seeds)
    for ent in seeds.decode["sampling_mode_policy"]:
        if ent.get("model_key") and ent["model_key"] != "*":
            assert ent["model_key"] in mmap, ent
    # instruments -> evals, item sets, judges, attacker models, policy refs
    served = D.served_model_ids(seeds.models)
    for r in seeds.instruments:
        assert set(r["eval_ids"]) <= evals, r["instrument_id"]
        assert set(r["item_set_ids"]) <= sets, r["instrument_id"]
        if r.get("judge_id"):
            assert r["judge_id"] in judges, r["instrument_id"]
        if r.get("attacker_model_id"):
            assert r["attacker_model_id"] in models, r["instrument_id"]
        for ref in r.get("decode_policy_refs") or []:
            targets = [idx.resolve_ref(ref, m) for m in served] if str(ref).startswith("*|") else [idx.resolve_ref(ref)]
            assert all(targets), (r["instrument_id"], ref)
    # item sets / evals point at each other
    for s in seeds.item_sets:
        assert s.get("eval_id") in evals or s.get("eval_id") is None, s["item_set_id"]
    for e in seeds.evals:
        assert e.get("default_item_set_id") in sets or e.get("default_item_set_id") is None, e["eval_id"]
    # legacy spellings: every model_id / fit_model_id / eval_id / judge_id they name exists (context prose excepted)
    for ent in seeds.legacy.get("model_keys") or []:
        for k in ("model_id", "fit_model_id"):
            v = ent.get(k)
            if isinstance(v, str) and " " not in v and ":" not in v:
                assert v in models, ent
    for ent in seeds.legacy.get("eval_ids") or []:
        v = ent.get("eval_id")
        if isinstance(v, str) and " " not in v and "|" not in v and "<" not in v:
            assert v in evals, ent
    for ent in seeds.legacy.get("judge_strings") or []:
        v = ent.get("judge_id")
        if isinstance(v, str) and " " not in v:
            assert v in judges, ent
    # SI token counts name seeded models
    for row in seeds.si_texts:
        counts = row.get("si_token_count_by_model")
        if isinstance(counts, dict):
            assert set(counts) <= models, row["si_id"]


def test_served_models_are_the_instruct_and_chat_paper_checkpoints(seeds):
    assert D.served_model_ids(seeds.models) == ["gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"]


# --------------------------------------------------------------------------------------------
# decode policy: uniqueness, references, contradictions kept
# --------------------------------------------------------------------------------------------


def test_at_most_one_default_per_key_and_the_known_no_default_key(bundle):
    idx = bundle.policies
    groups = {}
    for pid, (p, e) in idx.rows.items():
        if p.get("superseded_by_policy_id") or e.get("superseded_by_unresolved"):
            continue
        key = (p["model_id"], p.get("eval_id") or f"category:{p.get('eval_category')}", p["thinking_mode"])
        groups.setdefault(key, []).append(int(p.get("is_default") or 0))
    assert all(sum(v) <= 1 for v in groups.values())
    # no key is without a default: the one such key (qwen35_9b_it|category:agentic|off) got one on 2026-09-21, when the rule
    # decode-vendor-default-all-evals-2026-09-21 made every `<model>|agentic|<thinking>|vendor` row its key's default
    assert idx.no_default_keys == []


def test_two_defaults_for_one_key_are_refused(seeds):
    s = copy.deepcopy(seeds)
    row = next(p for p in s.decode["policies"] if p["policy_ref"] == "qwen35_4b_it|capability|on|general")
    row["is_default_for_lookup"] = 1   # thinking_card is already the default for (qwen35_4b_it, capability, on)
    with pytest.raises(D.SeedRefusal) as ei:
        D.build_records(s, PROD)
    assert ei.value.field == "is_default_for_lookup" and "at most one default" in ei.value.why


def test_same_values_as_mismatch_is_refused(seeds):
    s = copy.deepcopy(seeds)
    row = next(p for p in s.decode["policies"] if p["policy_ref"] == "gemma4e4b_it|gsm8k|on|thinking_card")
    row["top_k"] = 63
    with pytest.raises(D.SeedRefusal) as ei:
        D.build_records(s, PROD)
    assert ei.value.field == "same_values_as"


def test_unresolvable_superseded_by_is_null_with_a_note_and_exactly_the_known_two(bundle):
    idx = bundle.policies
    assert sorted(pid for pid, _, _ in idx.unresolved) == ["glm46vflash|gsm8k|off|greedy@v1", "qwen35_4b_it|multi_turn|off|greedy@v1"]
    for pid, col, text in idx.unresolved:
        payload, extra = idx.rows[pid]
        assert payload["superseded_by_policy_id"] is None and extra["superseded_by_unresolved"] == text


def test_wildcard_rows_expand_over_served_models_only(bundle):
    idx = bundle.policies
    greedy = [pid for pid in idx.rows if pid.endswith("|static|off|greedy@v1")]
    assert sorted(greedy) == sorted(f"{m}|static|off|greedy@v1" for m in ["gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"])
    assert all(idx.rows[pid][1]["wildcard_expanded_from"] == "*|static|off|greedy" for pid in greedy)


def test_contradictions_keep_both_values_with_the_ruled_default(bundle):
    """The 14 contradictions keep both values; the drafter's default flag decides which `resolve()` serves."""
    idx = bundle.policies
    # #1 Qwen card row (0.7/0.95/20, matches no vendor row): kept as the legacy identity, labelled as such
    card = idx.rows["qwen35_4b_it|capability|off|card@v1"][0]
    assert (card["temperature"], card["top_p"], card["top_k"]) == (0.7, 0.95, 20) and card["decode_source"] == "legacy_contracts_row"
    assert idx.rows["qwen35_4b_it|capability|off|card@v1"][1]["seed_decode_source"] == "model_card"
    # #2 two vendor thinking rows for Qwen GSM8K: precise is the default, general kept non-default
    assert idx.rows["qwen35_4b_it|gsm8k|on|thinking_card@v1"][0]["is_default"] == 1
    assert idx.rows["qwen35_4b_it|gsm8k|on|general@v1"][0]["is_default"] == 0
    # #4 Nemotron top_p: since 2026-09-17 the SHIPPED 0.95 (v2) is the default for new work (ruling
    # decode-nemotron-vendor-preset-0.95-2026-09-17, applied under the "a row is the default for NEW work"
    # rule); 1.0 (v1, the sampled panels' grandfathered instrument) is kept live and non-default, pinned @v1 by the
    # instrument refs; the omit-params 0.95 superseded
    for cat in ("static", "over_refusal"):
        assert idx.rows[f"nemotron3nano4b|{cat}|off|vendor_preset@v1"][0]["is_default"] == 0
        assert idx.rows[f"nemotron3nano4b|{cat}|off|vendor_preset@v2"][0]["is_default"] == 1
        # live, readable, NOT superseded: since 2026-10-14 the loader STATES the column on every row (an explicit None where the seed
        # names no successor), so a pointer a seed drops folds the live row back to NULL -- before that a live row carried no key at all
        # and a dropped pointer changed nothing in the database (tests/test_registry_decode_history.py pins the fold)
        assert "superseded_by_policy_id" in idx.rows[f"nemotron3nano4b|{cat}|off|vendor_preset@v1"][0]
        assert idx.rows[f"nemotron3nano4b|{cat}|off|vendor_preset@v1"][0]["superseded_by_policy_id"] is None
    assert idx.numbers("nemotron3nano4b|static|off|vendor_preset@v2")["top_p"] == 0.95
    # the instrument that RAN keeps @v1: the flip re-describes no run
    nemo = 0
    for iid in bundle.instrument_ids["static_sampled_vendor_preset_4096_8192"]:
        rec = next(r["payload"] for r in bundle.records["instrument"] if r["payload"]["instrument_id"] == iid)
        if rec["extra"].get("model_id") == "nemotron3nano4b":     # .get: the gemma4e4b_it rows are MERGED with gemma4_31b's (model_ids only)
            assert rec["extra"]["decode_policy_id"] == "nemotron3nano4b|static|off|vendor_preset@v1" and rec["decode_top_p"] == 1.0
            nemo += 1
    assert nemo == 4     # one row per static eval
    assert idx.rows["nemotron3nano4b|static|off|generation_config@v1"][0]["superseded_by_policy_id"] == "nemotron3nano4b|static|off|vendor_preset@v1"
    # the earlier Qwen multi-turn route is kept, superseded by the server-default route
    assert idx.rows["qwen35_4b_it|multi_turn|off|vllm_default@v1"][0]["superseded_by_policy_id"] == "qwen35_4b_it|multi_turn|off|server_default@v1"
    # best-of-N (2026-09-17): the grandfathered T=1 rows are kept, non-default, beside the per-model vendor_preset rows (the default),
    # whose numbers are the static vendor rows' (same_values_as; nemotron: the shipped @v2). Superseded by the vendor rows from 2026-09-17
    # to 2026-10-14; LIVE since the rule of 2026-10-14 (decode-bon-canonical-T1-live-non-default-2026-10-14: canonical best-of-N
    # stays T = 1, so the row its draws send is no longer superseded -- tests/test_registry_decode_history.py)
    for m in ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"):
        old_, new_ = idx.rows[f"{m}|bon|off|grandfathered_T1@v1"], idx.rows[f"{m}|bon|off|vendor_preset@v1"]
        assert old_[0]["is_default"] == 0 and old_[0]["superseded_by_policy_id"] is None
        assert "decode-bon-canonical-T1-live-non-default-2026-10-14" in str(old_[0].get("notes"))
        assert new_[0]["is_default"] == 1 and new_[0]["eval_id"] == "strongreject_bon" and new_[1]["mode"] == "vendor_preset"
        static = "nemotron3nano4b|static|off|vendor_preset@v2" if m == "nemotron3nano4b" else f"{m}|static|off|vendor_preset@v1"
        assert new_[1]["same_values_as"] == static and idx.numbers(f"{m}|bon|off|vendor_preset@v1") == idx.numbers(static)
    assert idx.numbers("nemotron3nano4b|bon|off|vendor_preset@v1")["top_p"] == 0.95
    assert idx.numbers("qwen35_4b_it|bon|off|vendor_preset@v1")["presence_penalty"] == 1.5


def test_ruling_fks_are_null_and_the_ruling_text_is_kept(bundle):
    idx = bundle.policies
    bon = idx.rows["gemma4e4b_it|bon|off|grandfathered_T1@v1"]
    assert bon[0]["ruling_id"] is None
    assert bon[1]["decode_ruling_ref"] == "decode-bon-T1-grandfathered-2026-09-11" and "grandfathered" in bon[1]["decode_ruling_text"]


def test_effective_from_is_utc_ms_or_unverified(bundle):
    for pid, (p, e) in bundle.policies.rows.items():
        v = p["effective_from_utc"]
        assert v == "unverified" or re.match(r"^\d{4}-\d{2}-\d{2}T00:00:00\.000Z$", v), (pid, v)
        if v != "unverified":
            assert e.get("effective_from_confidence") == "date_only", pid


# --------------------------------------------------------------------------------------------
# judges and instruments: the spec's hashes
# --------------------------------------------------------------------------------------------


def test_judge_ids_are_the_spec_hash_and_readable_ids_are_kept(bundle):
    for rec in bundle.records["judge"]:
        p = rec["payload"]
        assert p["judge_id"] == identity.judge_id({k: p.get(k) for k in identity.JUDGE_ID_FIELDS})
        assert p["extra"]["seed_judge_id"] in bundle.judge_ids
    assert bundle.judge_ids["strongreject_v1_haiku45"] != bundle.judge_ids["strongreject_v2_haiku45"]


def test_instruments_expand_per_eval_and_decode_policy_and_hash_canonically(bundle):
    recs = {r["payload"]["instrument_id"]: r["payload"] for r in bundle.records["instrument"]}
    for iid, p in recs.items():
        assert iid == identity.instrument_id(p)
    # the static sampled instrument: 4 evals x 5 served models (decode numbers differ per model)
    # 20 -> 24 (2026-09-29): + 4 for qwen35_35b_a3b (1.0 / 0.95 / 20 matches no served model); gemma4_31b's rows MERGE into gemma4e4b_it's (the same
    # shipped numbers -> the same instrument identity; extra.model_ids lists both)
    assert len(bundle.instrument_ids["static_sampled_vendor_preset_4096_8192"]) == 24
    # the greedy static instrument: the same numbers on every model -> ONE row per eval, every policy listed
    greedy = bundle.instrument_ids["static_greedy_4096_full22"]
    assert len(greedy) == 4
    assert len(recs[greedy[0]]["extra"]["decode_policy_ids"]) == 5
    # the GSM8K paired instrument: llama's row carries the policy's `not_available` thinking arm
    g = [recs[i] for i in bundle.instrument_ids["gsm8k_paired200_inspect_10shot_thinking_on_8192"]]
    llama = next(p for p in g if p["extra"].get("model_id") == "llama8b_it")     # .get: gemma's row is merged with gemma4_31b's
    assert llama["thinking_mode"] == "not_available" and llama["extra"]["thinking_mode_declared"] == "on"
    assert llama["decode_temperature"] == 0.6 and llama["decode_top_k"] is None
    # read_at spelled with the controlled value
    assert {p["read_at"] for p in recs.values()} <= set(V.READ_AT)
    # a judge reference is the hash
    assert recs[bundle.instrument_ids["bon_512store_4096climb_T1"][0]]["judge_id"] == bundle.judge_ids["strongreject_v1_haiku45"]


# --------------------------------------------------------------------------------------------
# export / check round trip, CLI
# --------------------------------------------------------------------------------------------


def test_committed_export_equals_a_fresh_registry_export(loaded):
    root, _ = loaded
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert DP.export_diff(conn, EXPORT) == [], DP.export_diff(conn, EXPORT)
        assert DP.check_export(conn, EXPORT)
        assert DP.check_all(conn) == [], DP.check_all(conn)        # both committed copies, and equal to each other
    finally:
        conn.close()


def test_committed_export_is_hash_pinned_and_stamped():
    with open(EXPORT, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["export_version"] == DP.EXPORT_VERSION
    assert data["header"]["policy_set_sha256"] == DP.policy_set_sha256(data)
    assert isinstance(data["header"]["ingest_seq"], int) and data["header"]["ingest_seq"] > 0
    # 133 + the nemotron3nano4b|multi_turn|off|vendor_preset@v1 row + the seven `vendor` rows of the rule of 2026-09-21
    # (decode-vendor-default-all-evals-2026-09-21: gemma4e4b_it / glm46vflash / nemotron3nano4b / llama8b_it agentic, llama8b_it em_organism,
    # and the two Llama not_available twins)
    # 141 -> 161 (2026-09-29): the twenty pipeline-test rows (ten per model; tests/test_registry_seeds_test_models.py)
    # 161 -> 163: + the two `<test model>|bon|off|grandfathered_T1` rows of the canonical T1 climb
    # 163 -> 190 (2026-10-09, ruling decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09): the worked example's 27 twins of the qwen35_4b_it rows
    # (tests/test_registry_seeds_demo_instruments.py)
    # 190 -> 191 (2026-10-09): + the worked example's explicit MMLU greedy row (tests/test_registry_seeds_demo_instruments.py)
    assert data["header"]["n_policies"] == len(data["policies"]) == 191
    # 24 legacy entries + the two onboarded pipeline-test models' thinking_card mappings (2026-09-30;
    # tests/test_gsm8k_thinking_card_row_test_models.py)
    # 26 -> 31 (2026-10-09): the worked example's five (qwen35_4b_demo, mode) twins
    assert data["header"]["n_sampling_modes"] == len(data["sampling_mode_policy"]) == 31
    text = open(EXPORT, "r", encoding="utf-8").read()
    assert text == json.dumps(data, indent=1, sort_keys=True) + "\n", "the committed export is not in canonical form"


SERVED = ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash")


def test_committed_export_bon_default_is_the_vendor_preset_and_the_grandfathered_row_is_one_named_row():
    """2026-09-17: ruling decode-bon-vendor-preset-2026-09-17 makes `<model>|bon|off|vendor_preset` THE default for strongreject_bon on
    every served model, and `<model>|bon|off|grandfathered_T1` exactly ONE non-default row still readable by its bare ref (every store
    drawn before that day names it) -- superseded by the vendor row until 2026-10-14, LIVE since the ruling of that day
    (decode-bon-canonical-T1-live-non-default-2026-10-14: canonical best-of-N stays the T = 1 instrument, so the row its draws send stands
    beside the default; tests/test_registry_decode_history.py). Qwen's vendor row was seeded by two branches (the decode-pins branch as the default,
    the rulings branch as the row the seed-replicate battery draws under) and merged into ONE row bound once, by the BoN
    ruling; the multi-turn / Petri target rows stay bound by decode-qwen-vendor-preset-bon-multiturn-petri-2026-09-17 and are THE
    defaults of their keys, the filed stores' rows superseded and kept by name."""
    pf = DP.PolicyFile.load(EXPORT)
    for m in SERVED:
        p = pf.resolve(m, eval_id="strongreject_bon", thinking_mode="off")
        assert p.policy_id == f"{m}|bon|off|vendor_preset@v1" and p.mode == "vendor_preset", m
        assert p.is_default == 1 and p.superseded_by_policy_id is None and p.ruling_ref == "decode-bon-vendor-preset-2026-09-17", m
        g = pf.by_ref(f"{m}|bon|off|grandfathered_T1")
        assert g.policy_id == f"{m}|bon|off|grandfathered_T1@v1" and g.is_default == 0 and g.superseded_by_policy_id is None, m
        assert g in pf.find(m, eval_id="strongreject_bon") and p.policy_id != g.policy_id, m     # live (non-default) beside the default
        assert [x["policy_id"] for x in pf.policies if x["policy_ref"] == f"{m}|bon|off|grandfathered_T1"] == [g.policy_id], m
        assert [x["policy_id"] for x in pf.policies if x["policy_ref"] == f"{m}|bon|off|vendor_preset"] == [p.policy_id], m
    q = pf.resolve("qwen35_4b_it", eval_id="strongreject_bon", thinking_mode="off")
    assert q.numbers() == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
    assert q.request_params() == q.numbers() and "presence_penalty" in q.params_sent    # Qwen's 1.5 rides the request, never a server default
    assert pf.by_ref("qwen35_4b_it|bon|off|vendor_preset").policy_id == q.policy_id
    # the rows the ruling still binds: the two TARGET rows, THE defaults of their keys;
    # the filed stores' rows are superseded by them, non-default, and still exactly one row each by bare ref
    for ref, old_ref in (("qwen35_4b_it|multi_turn|off|vendor_preset", "qwen35_4b_it|multi_turn|off|server_default"),
                         ("qwen35_4b_it|petri|off|vendor_preset", "qwen35_4b_it|petri|off|vllm_default")):
        t = pf.by_ref(ref)
        assert t.is_default == 1 and t.superseded_by_policy_id is None, ref
        assert t.ruling_ref == "decode-qwen-vendor-preset-bon-multiturn-petri-2026-09-17" and t.numbers() == q.numbers(), ref
        assert "presence_penalty" in t.params_sent and t.request_params()["presence_penalty"] == 1.5, ref
        assert pf.resolve("qwen35_4b_it", eval_category=t.eval_category, thinking_mode="off").policy_id == t.policy_id, ref
        o = pf.by_ref(old_ref)
        assert o.policy_id == f"{old_ref}@v1" and o.is_default == 0 and o.superseded_by_policy_id == t.policy_id, old_ref
        assert [x["policy_id"] for x in pf.policies if x["policy_ref"] == old_ref] == [o.policy_id], old_ref
        assert o.request_params() == {} and o.eval_category == t.eval_category, old_ref     # the filed stores' row sends nothing
    # the earlier multi-turn row keeps its own pointer (a two-step chain: vllm_default -> server_default -> vendor_preset)
    assert pf.by_ref("qwen35_4b_it|multi_turn|off|vllm_default").superseded_by_policy_id == "qwen35_4b_it|multi_turn|off|server_default@v1"
    assert not [x for x in pf.policies if x["ruling_ref"] == "decode-qwen-vendor-preset-bon-multiturn-petri-2026-09-17" and x.get("eval_id") == "strongreject_bon"]
    # the harness default every existing store was drawn under still reads the superseded rows, model-independently (bon_served)
    assert pf.model_independent(eval_id="strongreject_bon", thinking_mode="off", mode="grandfathered_T1").request_params() == {"temperature": 1.0}


EM8Q_VENDOR = "qwen35_4b_it|em8q|off|vendor_preset"
EM8Q_T1 = "qwen35_4b_it|em8q|off|t1_record"
VENDOR_DEFAULT_RULING = "decode-vendor-preset-default-2026-09-17"


def test_em8q_eval_and_the_qwen_organism_decode_rows_encode_the_vendor_preset_ruling(seeds, bundle):
    """2026-09-17: the EM MODEL ORGANISMS panel `qwen35_4b_org` runs the Betley 8-question
    family `em8q` (8 questions x 50 samples, 600-token cap) on three Qwen3.5-4B organisms x the campaign's objects. `em8q` is seeded as
    ITS OWN eval (the benchmark; the organism is the model), and two decode rows key on it: `qwen35_4b_it|em8q|off|vendor_preset` (the
    six vendor values + a per-response seed; THE default, bound by the rule "always use the vendor presets unless specifically
    instructed otherwise") and `qwen35_4b_it|em8q|off|t1_record` (temperature 1.0 alone -- the RECORD rung every organism cell was first
    drawn under; superseded, non-default, readable by name because `EM_ORG_SAMPLER=t1` reproduces it and its store reads it)."""
    ev = next(e for e in seeds.evals if e["eval_id"] == "em8q")
    assert ev["eval_category"] == "em_organism" and ev["default_item_set_id"] == "em8q_questions" and ev["attacker_required"] == 0
    assert "em8q" in ev["producer_spellings"]
    # exactly one eval owns the spelling `em8q` (the export drops a spelling two evals claim; the Llama-era panel evals keep their own)
    assert [e["eval_id"] for e in seeds.evals if "em8q" in (e.get("producer_spellings") or [])] == ["em8q"]
    assert VENDOR_DEFAULT_RULING in seeds.decode["rulings"] and "unless specifically instructed otherwise" in seeds.decode["rulings"][VENDOR_DEFAULT_RULING]
    idx = bundle.policies
    v_p, v_e = idx.rows[f"{EM8Q_VENDOR}@v1"]
    t_p, t_e = idx.rows[f"{EM8Q_T1}@v1"]
    # the default: the Qwen instruct row, presence_penalty on the request, a per-response seed, same numbers as the static vendor row
    assert v_p["eval_id"] == "em8q" and v_p.get("eval_category") is None and v_p["thinking_mode"] == "off" and v_p["is_default"] == 1
    assert v_e["mode"] == "vendor_preset" and v_p["superseded_by_policy_id"] is None     # stated null on every live row since 2026-10-14
    assert idx.numbers(f"{EM8Q_VENDOR}@v1") == idx.numbers("qwen35_4b_it|static|off|vendor_preset@v1") == \
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
    assert v_e["same_values_as"] == "qwen35_4b_it|static|off|vendor_preset@v1"
    assert v_p["seed_rule"] == "per_response" and "seed" in v_e["params_sent"] and "presence_penalty" in v_e["params_sent"]
    assert "600" in v_p["max_new_tokens_rule"]
    assert v_e["decode_ruling_ref"] == VENDOR_DEFAULT_RULING and "vendor preset" in v_e["decode_ruling_text"]
    assert v_p["effective_from_utc"] == "2026-09-17T00:00:00.000Z"
    # the record rung: T = 1.0 alone, superseded by the vendor row, non-default, kept by name
    assert t_p["is_default"] == 0 and t_p["superseded_by_policy_id"] == v_p["policy_id"] and t_e["mode"] == "t1_record"
    assert t_p["decode_source"] == "runner_forced_T1" and t_p["temperature"] == 1.0 and t_p["top_p"] is None and t_p["presence_penalty"] is None
    assert t_p["seed_rule"] == "none" and t_e["params_sent"] == ["temperature", "max_tokens"] and "600" in t_p["max_new_tokens_rule"]
    assert t_p["eval_id"] == "em8q"
    # the instruments: one row per rung (the decode numbers are instrument identity), both on em8q x em8q_questions, judge em8q_judge
    recs = {r["payload"]["instrument_id"]: r["payload"] for r in bundle.records["instrument"]}
    by_seed = {sid: [recs[i] for i in ids] for sid, ids in bundle.instrument_ids.items() if sid.startswith("em8q_qwen35_4b_org")}
    assert set(by_seed) == {"em8q_qwen35_4b_org_vendor_preset_600", "em8q_qwen35_4b_org_t1_record_600"}
    vend, rec = by_seed["em8q_qwen35_4b_org_vendor_preset_600"], by_seed["em8q_qwen35_4b_org_t1_record_600"]
    assert len(vend) == 1 and len(rec) == 1 and vend[0]["instrument_id"] != rec[0]["instrument_id"]
    for p in vend + rec:
        assert p["eval_id"] == "em8q" and p["item_set_id"] == "em8q_questions" and p["harness"] == "em_organism" and p["token_cap_tokens"] == 600
        # the worked example's twin ref (2026-10-09) states the same numbers, so its record merges into this one: model_ids lists both keys, the
        # singular model_id / decode_policy_id give way to the lists (tests/test_registry_seeds_demo_instruments.py)
        assert p["judge_id"] == bundle.judge_ids["em8q_judge"] and p["thinking_mode"] == "off" and p["extra"]["model_ids"] == ["qwen35_4b_it", "qwen35_4b_demo"]
    assert vend[0]["extra"]["decode_policy_ids"] == [f"{EM8Q_VENDOR}@v1", f"{EM8Q_VENDOR}@v1".replace("qwen35_4b_it|", "qwen35_4b_demo|")]
    assert vend[0]["decode_presence_penalty"] == 1.5 and vend[0]["decode_seed_rule"] == "per_response"
    assert rec[0]["extra"]["decode_policy_ids"] == [f"{EM8Q_T1}@v1", f"{EM8Q_T1}@v1".replace("qwen35_4b_it|", "qwen35_4b_demo|")]
    assert rec[0]["decode_temperature"] == 1.0 and rec[0]["decode_top_p"] is None
    assert vend[0]["extra"]["paper_status"] == "canonical" and rec[0]["extra"]["paper_status"] == "superseded"   # not a column: rides extra_json
    # the committed export resolves the eval row: the vendor default, and the record rung by name
    pf = DP.PolicyFile.load(EXPORT)
    p = pf.resolve("qwen35_4b_it", eval_id="em8q", thinking_mode="off")
    assert p.policy_id == f"{EM8Q_VENDOR}@v1" and p.mode == "vendor_preset" and p.ruling_ref == VENDOR_DEFAULT_RULING
    assert p.request_params() == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
    assert pf.eval_category_of("em8q") == "em_organism" and pf.eval_id("em8q") == "em8q"
    g = pf.by_ref(EM8Q_T1)
    assert g.policy_id == f"{EM8Q_T1}@v1" and g.is_default == 0 and g.superseded_by_policy_id == p.policy_id and g.request_params() == {"temperature": 1.0}
    assert [x["policy_id"] for x in pf.policies if x["policy_ref"] == EM8Q_T1] == [g.policy_id]
    # no Qwen `em_organism` CATEGORY row exists (the organisms' eval row is the authority); the Llama-era category key resolves to its VENDOR row
    # since the rule of 2026-09-21 (decode-vendor-default-all-evals-2026-09-21), its T1 row kept by name, non-default, not superseded
    with pytest.raises(DP.NoPolicy):
        pf.resolve("qwen35_4b_it", eval_category="em_organism", thinking_mode="off")
    assert pf.resolve("llama8b_it", eval_category="em_organism", thinking_mode="off").policy_id == "llama8b_it|em_organism|off|vendor@v1"
    t1 = pf.by_ref("llama8b_it|em_organism|off|runner_forced_T1")
    assert t1.is_default == 0 and t1.superseded_by_policy_id is None and t1.request_params() == {"temperature": 1.0}


ORGANISMS = {"qwen35_4b_org_financial": "financial_r32", "qwen35_4b_org_medical": "medical_r32", "qwen35_4b_org_sports": "sports_r32"}


def test_the_qwen_em_organisms_are_model_rows_of_kind_organism_with_their_parent_and_unverified_shas(seeds, bundle):
    """2026-09-17: three Qwen3.5-4B EM organisms (financial / medical / sports, LoRA r32 fine-tunes of qwen35_4b_it on
    Betley datasets) are MODEL rows -- the organism is the model, em8q the eval -- of the new checkpoint kind `organism`, each naming its
    parent (`extra_json.parent_model_id`: the schema has no parent column) and carrying `unverified` revision / sha fields with the reason
    "shas not yet received" (the loader's field-confidence rule). They are NOT served checkpoints for the wildcard expansion (not instruct / chat)."""
    assert "organism" in V.CHECKPOINT_KINDS and "organism" in V.EVIDENCE["CHECKPOINT_KINDS"]
    assert "LoRA" in V.EVIDENCE["CHECKPOINT_KINDS"]["organism"] and "parent_model_id" in V.EVIDENCE["CHECKPOINT_KINDS"]["organism"]
    by_id = {r["payload"]["model_id"]: r["payload"] for r in bundle.records["model"]}
    for mid, tag in ORGANISMS.items():
        p = by_id[mid]
        assert p["checkpoint_kind"] == "organism" and p["family"] == "qwen35" and p["size_label"] == "4B"
        assert p["extra"]["parent_model_id"] == "qwen35_4b_it" and p["extra"]["lora_rank"] == 32
        assert f"em-{tag}" in p["extra"]["producer_spellings"] and tag in p["extra"]["producer_spellings"]
        assert p["hf_revision"] == "unverified" and p["chat_template_sha256"] == "unverified" and p["tokenizer_sha256"] == "unverified"
        fc = p["field_confidence_json"] if isinstance(p["field_confidence_json"], dict) else json.loads(p["field_confidence_json"])
        for col in ("hf_revision", "adapter_sha256", "chat_template_sha256"):
            assert fc[col]["confidence"] == "unverified" and "shas not yet received" in fc[col]["reason"], (mid, col)
        assert p["paper_status"] == "in_paper" and fc["row"]["confidence"] == "inferred" and "Qwen counterpart of the paper's organism panel" in fc["row"]["reason"]
        assert p["generation_config_present"] is None and "adapter" in fc["generation_config_present"]["reason"]
    # the parent exists and is a served checkpoint; the organisms are not served checkpoints for the wildcard rows
    assert "qwen35_4b_it" in by_id and by_id["qwen35_4b_it"]["checkpoint_kind"] == "instruct"
    assert not (set(ORGANISMS) & set(D.served_model_ids(seeds.models)))
    assert len([pid for pid in bundle.policies.rows if pid.endswith("|static|off|greedy@v1")]) == 5
    # every organism spelling resolves to its own row (no clash) and the export carries the rows
    mmap = D.legacy_model_map(seeds)
    for mid, tag in ORGANISMS.items():
        assert mmap[f"em-{tag}"] == mid and mmap[tag] == mid and mmap[mid] == mid
    pf = DP.PolicyFile.load(EXPORT)
    for mid in ORGANISMS:
        assert pf.models[mid]["checkpoint_kind"] == "organism" and pf.model_id(f"em-{ORGANISMS[mid]}") == mid
    # the organisms decode under the PARENT's em8q row (they are LoRA modules on the served qwen35_4b_it standup): no row keys on an organism
    assert not [x for x in pf.policies if x["model_id"] in ORGANISMS]


RESEED_TARGET_SEED_RULES = {
    # policy_ref -> the label naming the closure the decode seed is derived from (base = the literal training seed s)
    "qwen35_4b_it|agentic|off|vendor": "per_response",             # the misalignment runner: (seed_base, family / behaviour, item, sample)
    "qwen35_4b_it|petri|off|vendor_preset": "per_scenario_turn",   # Petri's target turns: (seed_base, behaviour, scenario_id, turn)
    "qwen35_4b_it|multi_turn|off|vendor_preset": "per_goal_turn",  # the multi-turn target: (seed_base, goal, turn)
}


def test_committed_export_reseed_target_rows_carry_the_seed_rule_naming_their_closure():
    """The seed-replicate battery sends a
    per-request seed on every TARGET request, derived from the literal training seed and a closure that differs per
    producer. The three rows the battery draws under carry the label naming that closure -- the strictness rule:
    a label naming the closure beats `per_item` plus a free-text `seed_rule_detail`, and `none` is wrong for a row whose
    request carries `seed`. The seed yaml carries no `per_item` row with a detail left over from before the labels existed."""
    pf = DP.PolicyFile.load(EXPORT)
    for ref, rule in RESEED_TARGET_SEED_RULES.items():
        p = pf.by_ref(ref)
        assert p.seed_rule == rule, (ref, p.seed_rule)
        assert p.params_sent is not None and "seed" in p.params_sent, ref     # a seed rule other than `none` means the request carries one
    assert {x["seed_rule"] for x in pf.policies} <= set(V.DECODE_SEED_RULES)
    with open(os.path.join(SEEDS, "decode_policy_seed.yaml"), "r", encoding="utf-8") as fh:
        seed = yaml.safe_load(fh)
    by_ref = {str(r["policy_ref"]): r for r in seed["policies"]}
    for ref, rule in RESEED_TARGET_SEED_RULES.items():
        assert by_ref[ref]["seed_rule"] == rule, ref
    leftover = [r["policy_ref"] for r in seed["policies"] if r.get("seed_rule") == "per_item" and r.get("seed_rule_detail")]
    assert leftover == [], leftover
    # every row whose request carries `seed` names a rule other than `none`, and vice versa
    for r in seed["policies"]:
        sent = r.get("params_sent")
        if isinstance(sent, list):
            assert ("seed" in sent) == (r["seed_rule"] != "none"), (r["policy_ref"], r["seed_rule"], sent)


def test_check_detects_drift(loaded, tmp_path):
    root, _ = loaded
    with open(EXPORT, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["policies"][0]["temperature"] = 0.99
    bad = tmp_path / "drift.json"
    bad.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        diff = DP.export_diff(conn, str(bad))
        assert "section 'policies' differs" in diff and "header.policy_set_sha256 does not pin the committed body" in diff
        assert not DP.check_export(conn, str(bad))
    finally:
        conn.close()


def test_cli_dimensions_load_then_export_check(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    assert cli.main(["dimensions", "load", "--seeds", SEEDS]) == 0
    rep = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    # 18 model + 9 si_text + 35 eval + 22 item_set + 13 judge + 105 instrument + 141 decode_policy + 24 sampling_mode_policy + 10 object
    # + the 9 seed files' file_arrival facts, folded in the same cycle. Instruments: 95 + the two vendor-preset best-of-N seeds
    # (bon_512_only_vendor_preset / bon_512store_4096climb_vendor_preset, 2026-09-20), each expanded over the five served models' rows.
    # decode_policy 133 -> 134: the nemotron3nano4b|multi_turn|off|vendor_preset@v1 row (2026-09-20, the TARGET row the paired multi-turn runs sent);
    # 134 -> 141: the seven `<model>|agentic|<thinking>|vendor` / llama8b_it|em_organism|<thinking>|vendor rows of the 2026-09-21 ruling
    # 377 -> 379 (2026-09-21): the full-width MMLU read's item set `mmlu_1531` + its instrument `mmlu_letter_prefill_full_1531` (one shared row:
    # the greedy letter read's decode numbers agree across the served models)
    # 379 -> 384 (2026-09-22): the NATIVE 4,096 vendor-preset best-of-N seed `bon_native4096_vendor_preset` (adapters/bon_native.py: the Llama
    # paired-write native pass, no 512 store, no climb), expanded over the five served models' vendor rows like the two vendor-preset seeds above
    # 384 -> 388: the FULL-suite BigCodeBench / BFCL item sets `bigcodebench_full_1140` / `bfcl_full_3981` + their instruments
    # `bigcodebench_full_protocol_greedy_4096` / `bfcl_full_protocol_T0001_1024` (one shared row each: the protocol rows agree across the served models,
    # as the 200 siblings' do); seeded on BOTH sides, 2026-09-23 and 2026-09-24, and merged 2026-09-24 with the later side's blocks kept
    # 388 -> 444 (2026-09-29, the paper-companion-repo pipeline-test seeds): + 4 model rows, + 20 decode_policy rows, + 32 instrument rows (static + 4,
    # bon_native + 1, the three multi-turn + 2 each, the two over-refusal + 1 each, petri 7 evals x 2 models, the vendor agentic sibling 2 x 2, gsm8k + 1;
    # gemma4_31b's vendor numbers equal gemma4e4b_it's so its static / over-refusal / bon / gsm8k rows merge; the mmlu / bcb / bfcl explicit refs merge
    # into the model-independent rows) -- tests/test_registry_seeds_test_models.py pins the per-model shape
    # 444 -> 446 (2026-09-29): + 2 decode_policy rows (the test models' bon|off|grandfathered_T1); the T1 instrument row they join already exists
    # 446 -> 448 (2026-09-30): + 2 sampling_mode_policy records, the test models' thinking_card mappings (no policy row added)
    # 448 -> 454 (2026-10-01): + 6 item_set rows, the per-behaviour Petri sets `petri_scenarios_<behaviour>` (the partition of
    # petri_scenarios_6x10 by behaviour; no instrument, eval or policy row added -- the instruments still name the pooled set)
    # 454 -> 456 (2026-10-08): + 2 model rows, the front-door WORKED EXAMPLE's key qwen35_4b_demo / qwen35_4b_demo_base (the same
    # Qwen3.5-4B pins as qwen35_4b_it / qwen35_4b_base under a key of their own, paper_status reference; no decode, instrument or si_text row added)
    # 456 -> 488 (2026-10-09, ruling decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09): + 27 decode_policy rows (the worked example's twins of every
    # explicit qwen35_4b_it row) + 5 sampling_mode_policy records (its mode twins); NO instrument row added -- a twin states the 4B's sent numbers, so
    # its instrument record hashes to the 4B's identity and merges into it (extra.model_ids lists both keys) -- tests/test_registry_seeds_demo_instruments.py
    # 488 -> 489 (2026-10-09): + 1 decode_policy row, the worked example's explicit MMLU greedy row (its instrument record merges into the full-1531 one)
    # 489 -> 492 (2026-10-09): + 3 instrument records, the worked example's own crescendo / pair / actorattack records (its cap-4096 multi-turn
    # instruments bind the key's six-key vendor_preset twin, whose sent numbers differ from the 4B's as-filed server_default row, so nothing merges)
    # 492 -> 494: + 2 instrument records, the non-template variants of the 35B's actorattack / crescendo instruments (applied_positions_mode
    # non_template is identity, so each mints its own row beside the canonical every-token one; one decode ref each, so nothing merges)
    # 494 -> 492: - the HHH reference text and its declared-prompt object (the unlicensed HHH prompt was removed)
    assert {k: rep["ingest"][k] for k in ("ok", "refused", "blocked")} == {"ok": 492 + 9, "refused": 0, "blocked": 0} and rep["files_written"] == 9
    # the nine files fold in ONE transaction (tests/test_registry_dimensions_atomic.py pins the reader's side); the report carries its length
    assert rep["ingest"]["dimension_transactions"] == 1 and 0 < rep["ingest"]["dimension_write_s"] <= rep["lock_hold_s"]
    out = str(tmp_path / "export.json")
    assert cli.main(["decode-policy", "export", "--out", out]) == 0
    assert cli.main(["decode-policy", "check", "--out", out]) == 0
    with open(out) as fh, open(EXPORT) as gh:
        a, b = json.load(fh), json.load(gh)
    for k in DP._BODY_SECTIONS:
        assert a[k] == b[k], k
    # a second load of the same seed revision writes nothing
    assert cli.main(["dimensions", "load", "--seeds", SEEDS, "--no-ingest"]) == 0
    assert "already on the record log" in capsys.readouterr().out


def test_cli_load_refuses_a_broken_seed_naming_the_field(tmp_path, monkeypatch, capsys):
    import shutil
    seeds = tmp_path / "seeds"
    shutil.copytree(SEEDS, seeds)
    path = seeds / "models.yaml"
    data = yaml.safe_load(path.read_text())
    del data["models"][0]["hf_repo"]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    assert cli.main(["dimensions", "load", "--seeds", str(seeds)]) == 1
    err = capsys.readouterr().err
    assert "models.yaml" in err and "hf_repo" in err


# --------------------------------------------------------------------------------------------
# landing pass (2026-09-17)
# --------------------------------------------------------------------------------------------


def _strongreject_prompt_sha256(name: str) -> str:
    """sha256 of the prompt constant's text in scripts/judge_batch.py (the rubric identity), read without importing
    the script (it may import the Anthropic client)."""
    import hashlib
    src = open(os.path.join(ROOT, "scripts", "judge_batch.py"), encoding="utf-8").read()
    m = re.search(rf"^{name}\s*=\s*(\"\"\"|''')(.*?)\1", src, re.S | re.M)
    assert m, name
    return hashlib.sha256(m.group(2).encode("utf-8")).hexdigest()


def test_regrade_judges_are_filled_from_d31_and_share_the_rubric_sha(seeds, bundle):
    rows = {r["extra"]["seed_judge_id"]: r for r in (rec["payload"] for rec in bundle.records["judge"])}
    v1 = _strongreject_prompt_sha256("STRONGREJECT_PROMPT")
    assert v1.startswith("282b79aa0a40d816"), "D31's sha16 is the prefix of the prompt text's sha256"
    haiku, sonnet = rows["strongreject_v1_haiku45"], rows["strongreject_sonnet5_regrade"]
    assert haiku["rubric_sha256"] == sonnet["rubric_sha256"] == v1
    assert rows["strongreject_v2_haiku45"]["rubric_sha256"] == _strongreject_prompt_sha256("STRONGREJECT_PROMPT_V2")
    # two judges of ONE rubric: same rubric sha, different judge ids
    assert haiku["judge_id"] != sonnet["judge_id"]
    assert sonnet["judge_model"] == "claude-sonnet-5" and sonnet["threshold_rule"].startswith("strongreject > 0.5")
    assert sonnet["judge_backend"] == "batch_or_live" and sonnet["extra"]["judge_backends_seeded"] == ["batch", "live"]
    assert sonnet["extra"]["fallback_chain"] == ["claude-sonnet-4-6"]
    assert "row" not in sonnet.get("field_confidence_json", {}), "no longer `confidence: inferred`"
    opus = rows["or_bench_cpi_v1_opus5_regrade"]
    assert opus["judge_model"] == "claude-opus-5" and opus["rubric_name"] == "or_bench_cpi" and opus["judge_backend"] == "live"
    assert opus["extra"]["rubric_sha16"] == "15f29334703c31a8" and opus["extra"]["protocol_sha16"] == "b8d28441cf71caec"
    assert opus["extra"]["fallback_chain"] == ["claude-opus-4-8"]
    # the readable ids the peer adapters resolve through extra_json.seed_judge_id
    assert {"strongreject_sonnet5_regrade", "or_bench_cpi_v1_opus5_regrade"} <= set(bundle.judge_ids)
    # every judge id is minted by identity.judge_id from the normalised payload
    for p in rows.values():
        assert p["judge_id"] == identity.judge_id(p)


def test_object_rows_are_minted_for_none_and_every_si_text(seeds, bundle):
    objs = {r["payload"]["object_id"]: r["payload"] for r in bundle.records["object"]}
    none = objs["none"]
    assert none["object_kind"] == "none" and none["paper_label"] == "none" and none["object_role"] == "baseline_none"
    assert none["fit_target_kind"] == "none" and none["status"] == "approved" and none["artifact_sha256"] == "none"
    si_rows = {r["payload"]["si_id"]: r["payload"] for r in bundle.records["si_text"]}
    prompted = D.prompted_paper_labels(seeds)
    assert prompted == {"safety_refusal": "safety_si", "si_constitution": "constitution_si", "si_constitution_full": "constitution_si_full"}
    for si_id, row in si_rows.items():
        p = objs[identity.object_id("system_instruction", si_text_sha256=row["si_text_sha256"])]
        assert p["object_kind"] == "system_instruction" and p["object_role"] == "prompt" and p["fit_target_kind"] == "declared_prompt"
        assert p["si_text_sha256"] == p["artifact_sha256"] == row["si_text_sha256"]
        assert p["fit_objective"] == "not_applicable" and p["derivation"] == "none"
        assert p["extra"]["si_id"] == si_id and p["campaign"] == D.OBJECT_SEED_CAMPAIGN and "campaign_note" in p["extra"]
        if si_id in prompted:
            assert p["paper_label"] == prompted[si_id] and p["status"] == "approved"
        else:
            assert p["paper_label"] == "other" and p["status"] == "reference"
    assert len(objs) == 1 + len(si_rows)


def test_object_rows_ingest_and_satisfy_the_pairing_check(loaded):
    root, bundle = loaded
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        by_status = dict(conn.execute("SELECT status, COUNT(*) FROM object GROUP BY status").fetchall())
        assert by_status == {"approved": 4, "reference": 5}
        # the declared prompts point at ingested si_text rows (the FK the peers' runs were blocked on)
        n = conn.execute("SELECT COUNT(*) FROM object o JOIN si_text s ON s.si_text_sha256 = o.si_text_sha256 "
                         "WHERE o.object_kind = 'system_instruction'").fetchone()[0]
        assert n == 8
    finally:
        conn.close()


def test_instrument_extra_spells_the_model_binding_one_way(bundle):
    """`model_ids` is ALWAYS a list beside a singleton `model_id`; a seed entry whose rows state identical
    decode numbers on several models is ONE row listing them all (identical identity fields -> one instrument_id)."""
    recs = [r["payload"] for r in bundle.records["instrument"]]
    for p in recs:
        ex = p["extra"]
        if "model_id" in ex:
            assert ex["model_ids"] == [ex["model_id"]], ex.get("seed_instrument_id")
        if "decode_policy_id" in ex:
            assert ex["decode_policy_ids"] == [ex["decode_policy_id"]]
        if "model_ids" in ex:
            assert isinstance(ex["model_ids"], list) and len(ex["model_ids"]) == len(set(ex["model_ids"]))
            assert len(ex["model_ids"]) == 1 or "model_id" not in ex


def test_multiturn_instruments_resolve_to_one_row_per_eval_and_served_model(seeds, bundle):
    """The resolver rule (`model_id == m or m in model_ids`) selects exactly one row per (seed, eval, served
    model): the four sends-nothing generation_config policies share one row, Qwen's server_default row stands alone."""
    recs = [r["payload"] for r in bundle.records["instrument"]]
    served = D.served_model_ids(seeds.models)
    seeds_mt = ("multiturn_crescendo_cap4096_generation_config", "multiturn_pair_cap4096_generation_config",
                "multiturn_actorattack_cap4096_generation_config", "multiturn_greedy_512_full22")
    for seed_id in seeds_mt:
        rows = [p for p in recs if p["extra"]["seed_instrument_id"] == seed_id]
        for eval_id in {p["eval_id"] for p in rows}:
            for m in served:
                hit = [p for p in rows if p["eval_id"] == eval_id and (p["extra"].get("model_id") == m or m in p["extra"].get("model_ids", []))]
                assert len(hit) == 1, (seed_id, eval_id, m, len(hit))
    cres = [p for p in recs if p["extra"]["seed_instrument_id"] == "multiturn_crescendo_cap4096_generation_config"]
    # 2 -> 4 (2026-09-29): + the two pipeline-test models' `|multi_turn|off|vendor_preset` rows (qwen35_35b_a3b, gemma4_31b; each sends its
    # shipped numbers, so neither merges with the sends-nothing row nor with each other); the served models' rows are unchanged
    # 4 -> 5 (2026-10-09): + the worked example's OWN record -- its three cap-4096 multi-turn instruments bind the key's `|multi_turn|off|vendor_preset`
    # twin (the six-key vendor row a fresh run sends) rather than the 4B's as-filed server_default row, so the record merges with nobody's
    # 5 -> 2 (2026-10-14, the registry owner's ruling): the three vendor_preset refs moved to multiturn_crescendo_cap4096_vendor_preset, the
    # per_goal_turn seed of the vendor route (those rows require per_goal_turn and every run of the route was seeded per (goal, turn); the seed
    # rule is instrument identity) -- tests/test_registry_decode_history.py. The paper's two records here are byte-identical to before.
    assert len(cres) == 2
    shared = next(p for p in cres if "gemma4e4b_it" in p["extra"]["model_ids"])
    assert sorted(shared["extra"]["model_ids"]) == ["gemma4e4b_it", "glm46vflash", "llama8b_it", "nemotron3nano4b"]
    assert shared["decode_source"] == "generation_config" and shared["decode_temperature"] is None and shared["decode_seed_rule"] == "none"
    # Qwen's server_default row stands alone among the served models (the worked example's crescendo record is its own: the vendor_preset twin's
    # sent numbers differ from the as-filed server_default's -- tests/test_registry_seeds_demo_instruments.py)
    qwen = next(p for p in cres if p["extra"].get("model_id") == "qwen35_4b_it")
    assert qwen["extra"]["model_ids"] == ["qwen35_4b_it"] and qwen["decode_source"] == "vendor_recommendation"
    vend = [p for p in recs if p["extra"]["seed_instrument_id"] == "multiturn_crescendo_cap4096_vendor_preset"]
    assert sorted(p["extra"]["model_id"] for p in vend) == ["gemma4_31b", "qwen35_35b_a3b", "qwen35_4b_demo"]
    assert all(p["decode_seed_rule"] == "per_goal_turn" and p["extra"]["model_ids"] == [p["extra"]["model_id"]] for p in vend)
    demo = next(p for p in vend if p["extra"].get("model_id") == "qwen35_4b_demo")
    assert demo["extra"]["decode_policy_ref"] == "qwen35_4b_demo|multi_turn|off|vendor_preset" and demo["decode_source"] == "model_card"
    assert demo["decode_presence_penalty"] == 1.5 and demo["instrument_id"] != qwen["instrument_id"]


def test_legacy_spellings_carry_d37_overrides(seeds):
    legacy = seeds.legacy
    si_ids = {r["si_id"] for r in seeds.si_texts}
    over = {ent["fit_teacher_si_id"]: ent for ent in legacy["fit_target_overrides"]}
    assert set(over) == {"A10", "B3", "C16"} <= si_ids
    for sid, ent in over.items():
        assert ent["fit_target_kind"] in V.FIT_TARGET_KINDS and ent["object_role"] in V.OBJECT_ROLES and ent["paper_label"] in V.PAPER_LABELS
        assert sid in ent["legacy"] and ent.get("citation")
    kinds = {sp.strip(): ent for ent in legacy["object_kinds"] for sp in str(ent["legacy"]).split("|")}
    assert kinds["psr_paired_write"]["object_kind"] == "composed" and kinds["psr_paired_write"]["derivation"] == "composition"
    oos = legacy["out_of_scope_object_kinds"]
    spelled = {sp.strip() for ent in oos for sp in str(ent["legacy"]).split("|")}
    assert {"lofit", "alphakl_soft", "token_faithful", "diffmean_random", "rand_", "bidpo"} <= spelled
    for ent in oos:
        assert ent["status"] == "withdrawn" and ent["object_role"] == "other" and ent.get("reason") and ent.get("source")
        for sp in str(ent["legacy"]).split("|"):
            assert sp.strip().split(" ")[0] not in V.OBJECT_KINDS, sp
    labels = {sp.strip().split(" (")[0]: ent for ent in legacy["paper_labels"] for sp in str(ent["legacy"]).split("|")}
    assert labels["si_constitution__psr_kl_all"]["paper_label"] == "constitution_relu"
    assert labels["safety_refusal__lora_r1_all"]["paper_label"] == "safety_lora"
    judges = {j["judge_id"] for j in seeds.judges}
    strings = {sp.strip(): ent["judge_id"] for ent in legacy["judge_strings"] for sp in str(ent["legacy"]).split("|")}
    assert strings["judge_items.claude-opus-5.jsonl"] == "or_bench_cpi_v1_opus5_regrade" in judges
    assert strings["live_two_si_grade_cache.claude-sonnet-5.jsonl"] == "strongreject_sonnet5_regrade" in judges


def test_paper_labels_spell_every_fitted_stem_of_the_paper_s_arm_table():
    """The `paper_labels` table is keyed by campaign STEM and must agree with the paper's arm table (declared_common.ARMS, the readers'
    contract) on every fitted stem's label and role: the objects adapter reads the yaml first and ARMS second, so a disagreement would
    make the label depend on which table answered. The fit-sidecars loader must resolve every fitted stem, prefixed or not."""
    from sisteer.registry.adapters import declared_common as DC
    from sisteer.registry.adapters import fit_sidecars as FS
    fitted = {stem: a for stem, a in DC.ARMS.items() if a.object_kind not in ("none", "system_instruction", "static_r_arditi", "vector_all", "alphasteer")}
    assert {a.paper_label for a in fitted.values()} == {"constitution_relu", "constitution_lora", "control_relu", "control_lora",
                                                        "difference_relu_distilled", "difference_relu", "difference_lora", "safety_relu", "safety_lora"}
    fs = FS.load_seeds(SEEDS)
    for stem, arm in fitted.items():
        ent = fs.paper_label_for(stem)
        assert ent is not None, f"legacy_spellings.yaml paper_labels has no entry for the paper's fitted stem {stem!r}"
        assert (ent["paper_label"], ent["object_role"]) == (arm.paper_label, arm.object_role), stem
        for prefix in ("base", "itdoc", "itdoc64", "base64s2", "d0p5_eb64s1", "derived_itdoc64", "halfdoseit", "si12"):
            got, src, got_stem, got_prefix = FS.paper_label_entry_for(arm.object_kind, (f"{prefix}__{stem}",), fs)
            assert got is not None and got["paper_label"] == arm.paper_label and got_stem == stem and got_prefix == prefix, (prefix, stem)
    # the candidate-instruction distillations are ruled `other` by fit_target_overrides, never by a paper arm
    assert FS.paper_label_entry_for("lora", ("candsi__A10__lora_r4",), fs)[0] is None


def test_adapter_legacy_spellings_come_from_the_seeds(seeds):
    ls = D.adapter_legacy_spellings(seeds=seeds)
    assert set(ls) == {"model_ids", "si_texts"}
    assert ls["model_ids"] == D.legacy_model_map(seeds)
    assert ls["model_ids"]["google/gemma-4-E4B-it"] == "gemma4e4b_it" and ls["model_ids"]["gemma4e4b"] == "gemma4e4b_it"
    assert ls["si_texts"] == {r["si_id"]: r["si_text_sha256"] for r in seeds.si_texts}
    assert D.adapter_legacy_spellings(SEEDS) == ls


def test_cli_adapters_run_hands_the_seeds_legacy_spellings_to_the_pass(tmp_path, monkeypatch, capsys):
    """`registry adapters run` populates AdapterContext.legacy_spellings (model_ids, si_texts) from the seeds."""
    from sisteer.registry.adapters import base as B, runner as RUN
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    seen = {}

    def fake_run_pass(root_, adapters, conn=None, full=False, legacy_spellings=None, **kw):
        seen["legacy"] = legacy_spellings
        return RUN.PassReport()

    class _Dummy:
        name = "dummy"

    monkeypatch.setattr(RUN, "run_pass", fake_run_pass)
    monkeypatch.setattr(B, "registered", lambda: {"dummy": _Dummy})
    monkeypatch.setattr(H := __import__("sisteer.registry.harvest", fromlist=["x"]), "ingest_under_lock",
                        lambda root_, workers=0: type("R", (), {"ok": 0, "refused": 0, "blocked": 0})())
    assert cli.main(["adapters", "run", "--seeds", SEEDS]) == 0
    assert seen["legacy"] == D.adapter_legacy_spellings(SEEDS)
    assert seen["legacy"]["model_ids"]["Qwen/Qwen3.5-4B"] == "qwen35_4b_it" and "A10" in seen["legacy"]["si_texts"]


def test_seed_replicate_model_keys_spell_the_served_qwen_checkpoint(seeds):
    """The seed-replicate battery registers under `qwen35_4b_s<s>` (2026-09-20: refused as not a legacy spelling; 24 cluster B
    static jobs ran unregistered). The three keys map to the served instruct checkpoint with the base-checkpoint fit — in the
    adapters' legacy map (what `sampled_static._served_model` and the controls readers consult through `dims.models`) and in the
    lineage helper the harvester's readers use — and each row names its fit seed; the seed is on the object, never a model."""
    from sisteer.registry.adapters import lineage_common as LC
    keys = [f"qwen35_4b_s{s}" for s in (1, 2, 3)]
    mmap = D.legacy_model_map(seeds)
    assert [mmap[k] for k in keys] == ["qwen35_4b_it"] * 3
    assert [D.adapter_legacy_spellings(seeds=seeds)["model_ids"][k] for k in keys] == ["qwen35_4b_it"] * 3
    rows = {r["legacy"]: r for r in seeds.legacy["model_keys"] if r.get("legacy") in keys}
    assert [rows[k]["fit_replicate_label"] for k in keys] == ["s1", "s2", "s3"]
    assert {(rows[k]["fit_model_id"], rows[k]["transfer_kind"]) for k in keys} == {("qwen35_4b_base", "base_to_instruct")}
    for k in keys:
        m = LC.Seeds.from_dir(SEEDS).model_for_view_legacy(k)
        assert (m["model_id"], m["fit_model_id"], m["transfer_kind"], m["confidence"]) == ("qwen35_4b_it", "qwen35_4b_base", "base_to_instruct", "recorded"), k
    assert "qwen35_4b_s4" not in mmap and LC.Seeds.from_dir(SEEDS).model_for_view_legacy("qwen35_4b_s4")["model_id"] is None


def test_control_negative_standups_spell_the_served_qwen_checkpoint_with_their_seed(seeds):
    """The control-negative battery (2026-09-25): the six standups `qwen35_4b_dm1_ctl{lora,relu}64s<k>` map to the served instruct
    checkpoint with the base-checkpoint fit (the same triple as the seed-replicate keys and the source's spelling `qwen35_4b`, which the
    strength_mtdiff reader cross-checks them against) and each row names its fit seed; the controls twins `…_t0` are not listed (they
    resolve through the standup)."""
    from sisteer.registry.adapters import lineage_common as LC
    keys = [f"qwen35_4b_dm1_{fam}s{k}" for k in (1, 2, 3) for fam in ("ctllora64", "ctlrelu64")]
    mmap = D.legacy_model_map(seeds)
    assert [mmap[k] for k in keys] == ["qwen35_4b_it"] * 6
    rows = {}
    for r in seeds.legacy["model_keys"]:
        for spelling in LC._split_legacy(r.get("legacy", "")):
            if spelling in keys:
                rows[spelling] = r
    assert set(rows) == set(keys)
    assert [rows[k]["fit_replicate_label"] for k in keys] == ["s1", "s1", "s2", "s2", "s3", "s3"]
    assert {(rows[k]["fit_model_id"], rows[k]["transfer_kind"]) for k in keys} == {("qwen35_4b_base", "base_to_instruct")}
    src = LC.Seeds.from_dir(SEEDS).model_for_view_legacy("qwen35_4b")
    for k in keys:
        m = LC.Seeds.from_dir(SEEDS).model_for_view_legacy(k)
        assert (m["model_id"], m["fit_model_id"], m["transfer_kind"], m["confidence"]) == ("qwen35_4b_it", "qwen35_4b_base", "base_to_instruct", "recorded"), k
        assert (m["model_id"], m["fit_model_id"], m["transfer_kind"]) == (src["model_id"], src["fit_model_id"], src["transfer_kind"])
    assert "qwen35_4b_dm1_ctllora64s1_t0" not in mmap and "qwen35_4b_dm1_ctllora64s4" not in mmap


# --------------------------------------------------------------------------------------------
# accepted headline spellings (2026-09-21): strongreject_bon lists `asr` then `asr_bon_cap4096`
# --------------------------------------------------------------------------------------------


def test_strongreject_bon_seed_lists_its_accepted_headline_ids(seeds, bundle, loaded):
    """The eval row carries `extra_json.headline_metric_ids` = [asr, asr_bon_cap4096] beside the declared `headline_metric_id` = asr
    (the 512-store + 4,096-climb reading at bon_n 75; the native 4,096 pass is the same per-goal any-hit unit, threshold 0.5 both),
    from the seed through the record to the ingested row; `paper.headline_metric_ids` reads it. The two over-refusal evals accept
    `over_refusal` after their declared refused rate; every other eval accepts its declared id alone."""
    from sisteer.registry import paper as P
    row = next(e for e in seeds.evals if e["eval_id"] == "strongreject_bon")
    assert row["headline_metric_id"] == "asr" and row["headline_metric_ids"] == ["asr", "asr_bon_cap4096"]
    assert "asr_bon_cap4096" in row["headline_metric_ids_note"] and "per-goal any-hit unit" in row["headline_metric_ids_note"]
    r = next(r for r in bundle.records["eval"] if r["payload"]["eval_id"] == "strongreject_bon")
    assert r["payload"]["headline_metric_id"] == "asr" and r["payload"]["extra"]["headline_metric_ids"] == ["asr", "asr_bon_cap4096"]
    root, _ = loaded
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        ev = conn.execute("SELECT eval_id, headline_metric_id, extra_json FROM eval WHERE eval_id = 'strongreject_bon'").fetchone()
        assert ev["headline_metric_id"] == "asr"
        assert json.loads(ev["extra_json"])["headline_metric_ids"] == ["asr", "asr_bon_cap4096"]
        assert P.headline_metric_ids(dict(ev)) == ["asr", "asr_bon_cap4096"]
        assert not (set(P.headline_metric_ids(dict(ev))) & set(V.banned_metric_ids()))
        # the two over-refusal evals ACCEPT `over_refusal` (the native judge's spelling) after their declared refused rate
        for eval_id, head in (("xstest", "xst_refused"), ("or_bench_hard", "orb_refused")):
            ev = conn.execute("SELECT eval_id, headline_metric_id, extra_json FROM eval WHERE eval_id = ?", (eval_id,)).fetchone()
            assert ev["headline_metric_id"] == head and P.headline_metric_ids(dict(ev)) == [head, "over_refusal"], eval_id
        others = conn.execute("SELECT eval_id, headline_metric_id, extra_json FROM eval WHERE eval_id NOT IN ('strongreject_bon', 'xstest', 'or_bench_hard')").fetchall()
        assert others and all(P.headline_metric_ids(dict(e)) == [e["headline_metric_id"]] for e in others)
    finally:
        conn.close()


def test_the_over_refusal_headline_is_the_refused_rate(seeds, loaded):
    """The 2026-09-24 ruling r2v4-reference-over-refusal-headline-is-the-refused-rate: the headline of both
    over-refusal evals is the plain refused rate -- `xst_refused` for xstest, `orb_refused` for or_bench_hard: FULL refusals over the JUDGED
    items of the scored benign subset. `over_refusal_graded` (full 1 / partial 0.5 / a reply at the cap 0.5 whatever its label / compliance 0)
    is cap-dependent and stays a secondary row on every run. Pinned in the parsed seeds and in the loaded `eval` table."""
    heads = {e["eval_id"]: e["headline_metric_id"] for e in seeds.evals if e["eval_category"] == "over_refusal"}
    assert heads == {"xstest": "xst_refused", "or_bench_hard": "orb_refused"}
    assert all(e["metric_direction"] == "lower_is_better" and e.get("headline_threshold") is None for e in seeds.evals if e["eval_category"] == "over_refusal")
    root, _bundle = loaded
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows = conn.execute("SELECT eval_id, headline_metric_id FROM eval WHERE eval_category = 'over_refusal' ORDER BY eval_id").fetchall()
    finally:
        conn.close()
    assert [tuple(r) for r in rows] == [("or_bench_hard", "orb_refused"), ("xstest", "xst_refused")]


def test_load_seeds_exposes_the_d37_out_of_scope_table(seeds):
    """`fit_sidecars.load_seeds` reads the `out_of_scope_object_kinds` table (exact spellings, `rand_` as a prefix) so the objects bank
    parks these kinds by name; the table's spellings are the ones the live refusals named (lofit x4, alphakl_soft, token_appended, bidpo x2,
    the 40 `_controls/*` directions)."""
    from sisteer.registry.adapters import declared_common as DC
    from sisteer.registry.adapters import fit_sidecars as FS
    s = FS.load_seeds(DC.SEEDS_DIR)
    assert {"lofit", "alphakl_soft", "token_faithful", "token_appended", "diffmean_random", "bidpo", "did", "cast", "circuit_breaker"} <= set(s.out_of_scope_by_spelling)
    assert s.out_of_scope_prefixes == ("rand_",) and "rand_" in s.out_of_scope_by_spelling
    for sp, ent in s.out_of_scope_by_spelling.items():
        assert ent["status"] == "withdrawn" and ent["object_role"] == "other" and ent["reason"] and ent["source"], sp
    assert s.out_of_scope_for("safety_refusal__lofit")[0] == "lofit" and s.out_of_scope_for("rand_deepnat_s31")[0] == "rand_"
    assert s.out_of_scope_for("si_constitution__rand_matched_s33")[0] == "rand_" and s.out_of_scope_for("lora_r4", None) is None


def test_the_soft_alphakl_row_spells_the_blobs_own_kind(seeds):
    """Review of registry-hardening-2b (2026-10-07): a bundle that declares its kind parks only when the table spells THAT kind, so the
    row for soft alphakl objects carries `alphakl` -- the live `<si>__alphakl_soft` blob's own `kind` -- beside the directory spelling `alphakl_soft`,
    as the same row; the `object_kinds` OUT OF SCOPE listing agrees."""
    from sisteer.registry.adapters import declared_common as DC
    from sisteer.registry.adapters import fit_sidecars as FS
    s = FS.load_seeds(DC.SEEDS_DIR)
    assert s.out_of_scope_for("alphakl") == ("alphakl", s.out_of_scope_by_spelling["alphakl_soft"])
    assert "alphakl" in str(s.out_of_scope_by_spelling["alphakl"]["note"]) and "object.pt kind" in str(s.out_of_scope_by_spelling["alphakl"]["note"])
    rows = [ent for ent in seeds.legacy["object_kinds"] if str(ent.get("object_kind", "")).startswith("OUT OF SCOPE")]
    assert len(rows) == 1 and {"alphakl", "alphakl_soft"} <= {sp.strip() for sp in str(rows[0]["legacy"]).split("|")}
