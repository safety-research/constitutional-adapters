"""The `bigcodebench_allmodels` reader on the SYNTHETIC appendix mirror of `_bcbw_allmodels_fixtures`: the gemma4e4b family's one seed tag whose
conditions — the `none`, a fitted ReLU-steer object, a prompted key — ran under the `main` CANARY ROW, plus the paired tag's composed difference_relu
object and its own `none`; every store carrying both tasks; the gate records, the launch set, the launched pins and roster and the three-number record
written from the same truth.

What it pins beyond the shared reading (tested in full on the seedrep side):

* THE MODEL-KEY TABLE: `_bcbw_common.MODEL_KEYS` names the three-seed Qwen family and the three appendix families with the served checkpoint and
  revision the producer stamps (= models.yaml), the seeds each runs, whether the roster stages its selections and whether its `main` canary aliases;
  the tag grammar (`<model key>_s<N>[p]`, row dir `<model key>-f22s<N>[p]`, the `_main` canary) reads every family, longest key first;
* THE `main` CANARY ROW ALIAS: every store under `native/gemma4e4b-f22/<key>/thinking_off/bcbw-gemma4e4b_main-<sha7>/` files as `gemma4e4b_s1/<key>` —
  the producer key `gemma4e4b_s1/<store key>`, `extra.tag_legacy_original gemma4e4b_main`, the `none` the seed rows' anchor — whether the launch set and
  the three-number record spell the rows `s1` / `gemma4e4b_s1` / seed 1 or `main` / `gemma4e4b_main` / no seed; a fitted key on the canary row resolves
  through the harness roster's `main` selection (staged id -> pins; the launch set's pin key must be that staged id, `extra.object_route`); a key the
  `s1` row also carries refuses the canary row's store as superseded; a launch set naming another launch tag for a canary-row store refuses it;
* OBJECT IDENTITY BY PIN KEY on the `s1` / `s1p` rows: the harness roster stages no `s1` / `s1p` selection for the family, so a fitted key's staged id
  is the launch set's pin key and its sha the harness pins' entry (`extra.object_route`); a pin key the harness pins do not carry refuses the row by
  name (the siblings file), a roster `main` selection staging another id refuses too, and `resolve_object` without a pin key is unresolved;
* THE PAIRED TAG'S OWN `none` anchors the composed row with `paired_of` from the tag grammar (`extra.paired_of_source`, no roster selection);
* THE MODEL SET: a qwen store under this root refuses its group by name; the seedrep reader refuses the gemma group likewise; a second appendix
  family (llama8b) reads in its own group with its own model, and files `thinking_mode not_available`;
* THE DECODE POLICY: the registered protocol rows exist for all three appendix models (the wildcard expansion), and end to end the harvester's
  DERIVED match stamps `exact` with the gemma row (and llama's `not_available` twin); `decode_seed_rule none` on every row;
* NOT RUN: a launch-set row with no store (llama8b_s1/none) files nothing;
* the six appendix tags resolve through legacy_spellings.yaml; registration, source, owner, version.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

import _b1_fixtures as F
import _bcbw_allmodels_fixtures as Y
import _bcbw_seedrep_fixtures as X
from sisteer.registry import decode_policy as DP
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import bigcodebench_allmodels as M
from sisteer.registry.adapters import bigcodebench_seedrep as SR
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import seedrep_cluster as SC

EVAL = "bigcodebench"
MK, TAG, TAG_P, TAG_MAIN, SEL, SEL_P = Y.MK, Y.TAG, Y.TAG_P, Y.TAG_MAIN, Y.SEL, Y.SEL_P
NONE, RELU, PROMPTED, COMPOSED = Y.NONE, Y.RELU, Y.PROMPTED, Y.COMPOSED
APPENDIX_MODEL_IDS = ("gemma4e4b_it", "llama8b_it", "nemotron3nano4b")


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds, **kw):
    return Y.mirror(tmp_path, seeds, eval_id=EVAL, **kw)


def _adapter(fx, seeds) -> M.BigcodebenchAllmodelsAdapter:
    return Y.adapter(M.BigcodebenchAllmodelsAdapter, fx, seeds)


def _metrics(bt, rid):
    return Y.metrics_of(bt, rid, M.recount_computation_id())


# --------------------------------------------------------------------------------------------
# the model-key table and the tag grammar
# --------------------------------------------------------------------------------------------

def test_the_model_key_table_names_the_five_families_and_agrees_with_models_yaml(seeds):
    # the four of 2026-09-23 plus qwen35_35b_a3b (2026-10-04: the pipeline test's served checkpoint; bound by the FULL readers alone, tests/test_registry_adapter_bcbw_qwen35_35b_a3b.py)
    assert set(B.MODEL_KEYS) == {"qwen35_4b", "gemma4e4b", "llama8b", "nemotron3nano4b", "qwen35_35b_a3b"} and B.MODEL_KEY == SC.MODEL_KEY == "qwen35_4b"
    assert B.SEEDREP_MODEL_KEYS == ("qwen35_4b",) and B.APPENDIX_MODEL_KEYS == ("gemma4e4b", "llama8b", "nemotron3nano4b") == M.MODEL_KEYS == M.SPEC.models
    assert SR.SPEC.models == ("qwen35_4b",), "the seedrep twin keeps the three-seed family alone"
    served = {"gemma4e4b": ("gemma4e4b_it", "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
              "llama8b": ("llama8b_it", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
              "nemotron3nano4b": ("nemotron3nano4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"),
              "qwen35_4b": ("qwen35_4b_it", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
              "qwen35_35b_a3b": ("qwen35_35b_a3b", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754")}
    for mk, (mid, hf, rev) in served.items():
        row = B.MODEL_KEYS[mk]
        assert (row.key, row.model_id, row.served_hf_id, row.served_revision) == (mk, mid, hf, rev), mk
        assert seeds.models[mid]["hf_repo"] == hf and seeds.models[mid]["hf_revision"] == rev, "the table is models.yaml's pin, never a second answer"
        assert seeds.model_for_hf_id(hf) == mid
    assert B.MODEL_KEYS["qwen35_4b"].seeds == (1, 2, 3) and B.MODEL_KEYS["qwen35_4b"].roster_selections is True and B.MODEL_KEYS["qwen35_4b"].legacy_main_alias is False
    for mk in B.APPENDIX_MODEL_KEYS:
        assert B.MODEL_KEYS[mk].seeds == (1,) and B.MODEL_KEYS[mk].roster_selections is False and B.MODEL_KEYS[mk].legacy_main_alias is True, mk
    # the 35B: one seed row, the appendix object route, no `main` canary, no instruct-direct tag; outside BOTH 200-item tuples (no 200-item 35B store exists)
    assert B.MODEL_KEYS["qwen35_35b_a3b"] == B.ModelKey("qwen35_35b_a3b", "qwen35_35b_a3b", "Qwen/Qwen3.5-35B-A3B", "59d61f3ce65a6d9863b86d2e96597125219dc754", (1,), False, False, False, unsteered_row=True,
                                                        dose_relu_row=True, dose_lora_row=True, lecun_relu_row=True)
    assert all(not B.MODEL_KEYS[k].unsteered_row for k in B.MODEL_KEYS if k != "qwen35_35b_a3b"), "the unsteered row is the 35B's alone (UNSTEERED_ROW_RULE)"
    # the dose-0.75 random-init ReLU row (DOSE_RELU_ROW_RULE, 2026-10-05) is the 35B's alone too; the field exists on every row of the table, False elsewhere
    assert B.MODEL_KEYS["qwen35_35b_a3b"].dose_relu_row is True and all(B.MODEL_KEYS[k].dose_relu_row is False for k in B.MODEL_KEYS if k != "qwen35_35b_a3b")
    # the dose-1.25 difference-LoRA row (DOSE_LORA_ROW_RULE, 2026-10-05) likewise: the 35B's alone
    assert B.MODEL_KEYS["qwen35_35b_a3b"].dose_lora_row is True and all(B.MODEL_KEYS[k].dose_lora_row is False for k in B.MODEL_KEYS if k != "qwen35_35b_a3b")
    # the LeCun-init ReLU row (LECUN_RELU_ROW_RULE, 2026-10-07) likewise: the 35B's alone
    assert B.MODEL_KEYS["qwen35_35b_a3b"].lecun_relu_row is True and all(B.MODEL_KEYS[k].lecun_relu_row is False for k in B.MODEL_KEYS if k != "qwen35_35b_a3b")
    assert "qwen35_35b_a3b" not in B.SEEDREP_MODEL_KEYS and "qwen35_35b_a3b" not in B.APPENDIX_MODEL_KEYS and "qwen35_35b_a3b" not in M.SPEC.models
    assert B.tag_of_row_dir("qwen35_35b_a3b-f22s1") == "qwen35_35b_a3b_s1" and B.row_dir_of("qwen35_35b_a3b_s1") == "qwen35_35b_a3b-f22s1" and B.seed_tag_of("qwen35_35b_a3b_s1p") == "qwen35_35b_a3b_s1"
    # the u0 row is GRAMMAR since 2026-10-05 (UNSTEERED_ROW_RULE): its tag reads in the family's s1 group; the FULL readers admit it (TaskSpec.unsteered_row), this 200-item reader does not
    assert B.tag_of_row_dir("qwen35_35b_a3b-f22u0") == "qwen35_35b_a3b_u0" and B.seed_tag_of("qwen35_35b_a3b_u0") == "qwen35_35b_a3b_s1" and "u0" not in B.group_tags("qwen35_35b_a3b_s1", M.SPEC.instruct_direct, M.SPEC.unsteered_row)
    assert B.tag_of_row_dir("qwen35_35b_a3b-f22") is None, "a bare -f22 row of the 35B is undeclared (no main canary): reported by name, never read (UNDECLARED_ROW_DIR_RULE)"
    # the grammar reads every family (longest key first) and the `main` canary of an aliased family alone
    assert B.tag_parts("nemotron3nano4b_s1p") == (1, True) and B.model_key_of("nemotron3nano4b_s1p") == "nemotron3nano4b" and B.seed_tag_of("nemotron3nano4b_s1p") == "nemotron3nano4b_s1"
    assert B.tag_of_row_dir("gemma4e4b-f22s1p") == "gemma4e4b_s1p" and B.row_dir_of("llama8b_s1") == "llama8b-f22s1" and B.row_dir_of("qwen35_4b_s3p") == "qwen35_4b-f22s3p"
    assert B.tag_of_row_dir("gemma4e4b-f22") == "gemma4e4b_s1" and B.legacy_main_tag_of_row_dir("gemma4e4b-f22") == "gemma4e4b_main" and B.seed_tag_of("gemma4e4b_main") == "gemma4e4b_s1"
    assert B.model_key_of("gemma4e4b_main") == "gemma4e4b" and B.tag_parts("gemma4e4b_main") is None and B.legacy_main_alias_tag("llama8b_main") == "llama8b_s1"
    assert B.tag_of_row_dir("qwen35_4b-f22") is None and B.seed_tag_of("qwen35_4b_main") is None and B.legacy_main_tag_of_row_dir("qwen35_4b-f22") is None, "the three-seed family has no alias"
    assert B.tag_of_row_dir("gemma4e4b-f22s1x") is None and B.seed_tag_of("glm46vflash_s1") is None
    # the instruct-direct tag (2026-09-25) is GRAMMAR here -- it reads in the s1 group -- but this 200-item reader admits it for no selection (TaskSpec.instruct_direct false)
    assert B.seed_tag_of("gemma4e4b_it") == "gemma4e4b_s1" and B.tag_parts("gemma4e4b_it") is None and B.INSTRUCT_SELECTION not in B.group_tags(TAG, M.SPEC.instruct_direct) and M.SPEC.instruct_direct is False
    assert B.launch_tag_of(TAG, NONE, "gemma4e4b_main") == "gemma4e4b_main" and B.launch_tag_of(TAG, NONE) == TAG and B.launch_tag_of("qwen35_4b_s1", "arditi_allpos") == "qwen35_4b_s1_arditi"
    assert B.SUFFIX_RE.match("bcbw-gemma4e4b_main-8365992").group("launch_tag") == "gemma4e4b_main" and B.SUFFIX_RE.match("bcbw-nemotron3nano4b_s1p-8365992").group("launch_tag") == "nemotron3nano4b_s1p"
    assert B.SUFFIX_RE.match("bcbw-qwen35_4b_s1_arditi-8365992").group("launch_tag") == "qwen35_4b_s1_arditi"
    # the instruct-direct suffix (2026-09-25) is grammar -- it matches, its launch tag the `_it` tag -- and read() refuses such a store under this reader by name (TaskSpec.instruct_direct false)
    assert B.SUFFIX_RE.match("bcbw-gemma4e4b_it-8365992").group("launch_tag") == "gemma4e4b_it"
    assert B.gate_name_parts("gate_gemma4e4b_main_none_bcbw-gemma4e4b_main-8365992.json") == {"tag": "gemma4e4b_main", "arditi": None, "okey": "none", "suffix": "bcbw-gemma4e4b_main-8365992"}
    assert B.gate_name_parts("gate_llama8b_s1p_constv2-mix-full__psr_ce_all_minus_ctl_bcbw-llama8b_s1p-8365992.json")["tag"] == "llama8b_s1p"
    assert B.gate_name_parts("gate_qwen35_4b_s1_arditi_arditi_allpos_bcbw-qwen35_4b_s1_arditi-8365992.json")["tag"] == "qwen35_4b_s1"


def test_the_appendix_tags_resolve_through_legacy_spellings_to_the_tables_models(seeds):
    want = {"gemma4e4b": ("gemma4e4b_it", "gemma4e4b_base", "base_to_instruct"), "llama8b": ("llama8b_it", "llama8b_base", "base_to_instruct"),
            "nemotron3nano4b": ("nemotron3nano4b", "nemotron3nano4b", "chat_checkpoint")}
    for mk, (mid, fit, tk) in want.items():
        for tag in (f"{mk}_s1", f"{mk}_s1p"):
            mv = seeds.model_for_view_legacy(tag)
            assert (mv["model_id"], mv["fit_model_id"], mv["transfer_kind"], mv["confidence"]) == (mid, fit, tk, "recorded"), tag
            assert mv["model_id"] == B.MODEL_KEYS[mk].model_id
    assert seeds.model_for_view_legacy("gemma4e4b_main")["model_id"] is None, "the canary tag is an alias inside the reader, no spelling of its own"
    assert seeds.model_for_view_legacy("gemma4e4b_s2")["model_id"] is None, "one seed per appendix family"


# --------------------------------------------------------------------------------------------
# discovery, the canary alias, the anchors
# --------------------------------------------------------------------------------------------

def test_discover_groups_the_gemma_s1_tag_with_its_paired_tag_and_its_main_canary_row(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = [os.path.basename(i.uri) for i in inputs]
    assert {ad.comparison_group(i) for i in inputs} == {TAG}, "the canary row's stores and the paired tag's read in ONE group"
    assert names.count("manifest.json") == 5 and names.count(M.OUTCOMES_FILE) == 15 and not any(n.startswith("inspect__bfcl") for n in names)
    assert f"{M.THREE_NUMBERS_FILE}#{TAG}" in names and M.THREE_NUMBERS_FILE == "three_numbers_bigcodebench_allmodels.json"
    gates = sorted(n for n in names if n.startswith(B.GATE_PREFIX))
    assert {f"gate_{TAG_MAIN}_{k}_bcbw-{TAG_MAIN}-{Y.SHA7}.json#{TAG}" for k in (NONE, RELU, PROMPTED)} <= set(gates) and len(gates) == 5
    for k in (NONE, RELU, PROMPTED):
        d = os.path.join(fx["root"], B.STORES_HEAD, Y.ROW_DIR_MAIN, k, B.THINKING_DIR, f"bcbw-{TAG_MAIN}-{Y.SHA7}")
        assert any(i.uri.startswith(d) for i in inputs), k
    assert not os.path.isdir(os.path.join(fx["root"], B.STORES_HEAD, Y.ROW_DIR)), "the whole seed-tag row sits under the canary row (the producer's layout for gemma / llama)"
    assert os.path.isdir(os.path.join(fx["root"], B.STORES_HEAD, Y.ROW_DIR_P))


def test_the_canary_rows_stores_file_under_the_s1_tag_and_its_none_anchors_the_seed_rows(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 0, facts["refusals"]
    n = Y.run_at(bt, TAG, NONE)
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="gemma4e4b_it")
    assert n["view_legacy"] == TAG and n["producer_run_key_legacy"] == f"{TAG}/{fx['keys'][NONE]}" and n["run_id"] == M.run_id_for(TAG, fx["keys"][NONE], NONE, inst["instrument_id"])
    assert n["model_id"] == "gemma4e4b_it" and n["object_id"] == "none" and n["lane_source_legacy"] == M.SOURCE == "bigcodebench_allmodels" and n["instrument_id"] == inst["instrument_id"]
    assert n["extra"]["tag_legacy_original"] == TAG_MAIN and n["extra"]["producer_tag_legacy"] == TAG_MAIN and n["extra"]["producer_row_dir_legacy"] == Y.ROW_DIR_MAIN
    alias = n["extra"]["legacy_main_alias"]
    launched_as = {"wave": TAG_MAIN, "row_dir": Y.ROW_DIR_MAIN, "suffix": f"bcbw-{TAG_MAIN}-{Y.SHA7}", "sha": Y.TREE, "legs": ["bigcodebench_200", "bfcl_200"]}
    assert alias == {"tag_legacy_original": TAG_MAIN, "alias_tag": TAG, "row_dir_legacy_original": Y.ROW_DIR_MAIN, "launch_tag_legacy": TAG_MAIN, "launch_set_tag_spelling": TAG,
                     "launch_set_selection_spelling": SEL, "launch_set_seed_base": 1, "launch_set_launch_tag_spelling": TAG, "launch_set_launched_as": launched_as,
                     "three_numbers_key": f"{TAG}/{NONE}", "rule": B.LEGACY_MAIN_ALIAS_RULE}, "the census spells the row by the tag it files under and keeps the launch spelling on launched_as"
    assert n["extra"]["store_suffix_legacy"] == f"bcbw-{TAG_MAIN}-{Y.SHA7}" and n["extra"]["launch_set"]["launch_tag_legacy"] == TAG and n["extra"]["launched_roster"]["launch_tag_legacy"] == TAG_MAIN
    assert n["extra"]["store_uri"].endswith(f"{Y.ROW_DIR_MAIN}/{NONE}/{B.THINKING_DIR}/bcbw-{TAG_MAIN}-{Y.SHA7}") and n["extra"]["modal"]["usage_ledger"].startswith(B.VOLUME_HEAD + f"native/{Y.ROW_DIR_MAIN}/")
    assert "same_object_rows" not in n["extra"], "one seed per appendix family: no twin tag, no coexist note"
    assert f"{TAG_MAIN}" in n["notes"] and f"filed as {TAG}/{NONE}" in n["notes"]
    assert facts["rows"][f"{TAG}|{NONE}"]["tag_legacy_original"] == TAG_MAIN
    assert facts["legacy_main_alias_rows"] == [{"tag": TAG, "tag_legacy_original": TAG_MAIN, "row_dir": Y.ROW_DIR_MAIN, "object_key": k} for k in (RELU, NONE, PROMPTED)]
    # the canary row's steered and prompted stores file under the s1 tag too, their objects the roster's main selection's / si_texts
    r = Y.run_of(bt, RELU)
    assert r["view_legacy"] == TAG and r["extra"]["tag_legacy_original"] == TAG_MAIN and r["extra"]["producer_row_dir_legacy"] == Y.ROW_DIR_MAIN and r["object_id"] == Y.OBJ
    assert r["extra"]["object_route"]["route"] == "harness_roster_main_selection" and r["extra"]["object_route"]["harness_roster_selection"] == "main"
    p = Y.run_of(bt, PROMPTED)
    assert p["extra"]["tag_legacy_original"] == TAG_MAIN and p["extra"]["object_route"]["route"] == "si_texts" and p["object_id"].startswith("si:")
    assert facts["rows"][f"{TAG}|{RELU}"]["tag_legacy_original"] == TAG_MAIN and "tag_legacy_original" not in facts["rows"][f"{TAG_P}|{COMPOSED}"]
    # the value: correct / scored recounted from the canary's own outcome record
    c = Y.counts(fx["specs"][NONE])
    v, b = _metrics(bt, n["run_id"])
    assert (v["k"], v["n"]) == (c["correct"], c["n_scored"]) and b["producer_metric_spelling"] == "bound"
    # the seed rows anchor on it; the composed row on the paired tag's own none
    contrasts = {cn["run_id"]: cn for cn in bt["run_contrast"]}
    for k in (RELU, PROMPTED):
        assert contrasts[Y.run_of(bt, k)["run_id"]]["anchor_run_id"] == n["run_id"] and contrasts[Y.run_of(bt, k)["run_id"]]["extra"]["cross_standup"] is False
        assert Y.run_of(bt, k)["extra"]["anchor"]["state"] == B.ANCHOR_CONTRASTED
    assert not any("run_state" == k for k in bt)


@pytest.mark.parametrize("spelling", ["s1", "main"])
def test_the_launch_set_and_three_number_record_may_spell_the_canary_either_way(tmp_path, seeds, spelling):
    fx = _mirror(tmp_path, seeds, canary_spelling=spelling)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    n = Y.run_at(bt, TAG, NONE)
    alias = n["extra"]["legacy_main_alias"]
    if spelling == "main":
        assert (alias["launch_set_tag_spelling"], alias["launch_set_selection_spelling"], alias["launch_set_seed_base"], alias["three_numbers_key"]) == (TAG_MAIN, "main", None, f"{TAG_MAIN}/{NONE}")
        assert n["extra"]["launch_set"]["tuple_id"] == f"main/{NONE}" and alias["launch_set_launch_tag_spelling"] == TAG_MAIN and alias["launch_set_launched_as"] is None
    else:
        assert (alias["launch_set_tag_spelling"], alias["launch_set_selection_spelling"], alias["launch_set_seed_base"], alias["three_numbers_key"]) == (TAG, SEL, 1, f"{TAG}/{NONE}")
        assert alias["launch_set_launch_tag_spelling"] == TAG and alias["launch_set_launched_as"]["wave"] == TAG_MAIN
    assert n["view_legacy"] == TAG and n["extra"]["tag_legacy_original"] == TAG_MAIN
    # the source state names the canary's launch row and three-number row as members of the group
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    states = {os.path.basename(s.input_uri).rsplit("#", 1)[0]: s for s in ad.source_states(X.ctx(), inputs)}
    assert any(m.endswith(f"/{NONE}") and m.startswith(("main/", "s1/")) for m in states[B.LAUNCH_SET_FILE].current_members) and len(states[B.LAUNCH_SET_FILE].current_members) == 5
    assert alias["three_numbers_key"] in states[M.THREE_NUMBERS_FILE].current_members and len(states[M.THREE_NUMBERS_FILE].current_members) == 5


def test_stores_under_the_s1_row_file_without_the_alias(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, canary_layout="s1")
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    for r in bt["run"]:
        assert "tag_legacy_original" not in r["extra"] and "legacy_main_alias" not in r["extra"], r["campaign_cell_id_legacy"]
    n = Y.run_at(bt, TAG, NONE)
    assert n["extra"]["producer_tag_legacy"] == TAG and n["extra"]["producer_row_dir_legacy"] == Y.ROW_DIR and n["extra"]["store_suffix_legacy"] == f"bcbw-{TAG}-{Y.SHA7}"
    assert "legacy_main_alias_rows" not in facts
    # the `none` alone on the canary row, the rest under the s1 row: the alias rides the none and nothing else
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "none_only"), canary_layout="none")
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    assert facts2["n_refusals"] == 0 and facts2["runs"] == 5 and [x["object_key"] for x in facts2["legacy_main_alias_rows"]] == [NONE]
    assert Y.run_at(bt2, TAG, NONE)["extra"]["tag_legacy_original"] == TAG_MAIN and "tag_legacy_original" not in Y.run_of(bt2, RELU)["extra"]
    assert Y.run_of(bt2, RELU)["extra"]["object_route"]["route"] == "launch_set_pin_key"


def test_a_canary_row_store_launched_under_an_earlier_roster_revision_files_with_the_revision_recorded(tmp_path, seeds):
    """The real canary stores (tree 8365992) predate the roster's s1 blocks: their `lane_roster.json:roster_sha256` is an earlier revision's, while the mirrored
    roster is the current one. For a canary-row store the mismatch is RECORDED (the mirrored revision's `<model key>_main` block must still agree with the
    store's entry); for any other store it refuses by name, as before."""
    fx = _mirror(tmp_path, seeds, canary_roster_stale=True)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    for k in (NONE, RELU, PROMPTED):
        lr = Y.run_at(bt, TAG, k)["extra"]["launched_roster"]
        assert lr["roster_sha256"] != fx["roster_sha256"] == lr["roster_sha256_mirrored"] and "earlier revision" in lr["roster_revision_note"] and TAG_MAIN in lr["roster_revision_note"]
        assert lr["roster_entry"]["cell_id"] == k
    assert "roster_revision_note" not in Y.run_at(bt, TAG_P, NONE)["extra"]["launched_roster"]
    assert "roster_file" not in lr and "roster_set" not in facts, "no rosters.json: the plain roster is the set and nothing is recorded about a set"
    # a NON-canary store under an earlier revision still refuses
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "s1stale"), canary_layout="s1")
    import json
    p = os.path.join(fx2["root"], B.STORES_HEAD, Y.ROW_DIR, RELU, B.THINKING_DIR, f"bcbw-{TAG}-{Y.SHA7}", B.LAUNCHED_ENTRY_FILE)
    d = json.load(open(p)); d["roster_sha256"] = hashlib.sha256(b"another roster").hexdigest(); X.write(p, d)
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    assert "roster_sha256" in Y.refused(facts2, TAG, RELU) and "is not the mirrored" in Y.refused(facts2, TAG, RELU) and facts2["runs"] == 4


def test_a_key_the_s1_row_also_carries_refuses_the_canary_rows_store_as_superseded(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, canary_layout="s1", steered_under_main=True)
    bt, facts = Y.read(_adapter(fx, seeds))
    why = Y.refused(facts, TAG_MAIN, RELU)
    assert "superseded" in why and Y.ROW_DIR_MAIN in why and Y.ROW_DIR in why and "D565" in why and "never a second run of the key" in why
    assert facts["runs"] == 5 and Y.run_of(bt, RELU)["extra"]["producer_row_dir_legacy"] == Y.ROW_DIR and "tag_legacy_original" not in Y.run_of(bt, RELU)["extra"]
    assert sum(1 for r in bt["run"] if r["campaign_cell_id_legacy"] == RELU) == 1


def test_a_launch_set_naming_another_launch_tag_or_launch_record_for_a_canary_store_refuses_it(tmp_path, seeds):
    """The canary row's launch-set row may spell its launch tag as the store's (`gemma4e4b_main`) or as the tag it files under (`gemma4e4b_s1`, the census's
    shape); any other spelling, or a `launched_as` record naming another launch tag / row dir / suffix, refuses the row by name."""
    def relabel(doc):
        for r in doc["evals"][EVAL]["launch"]:
            if r["cell"] == NONE and r["tag"] == TAG:
                r["bcbw_wave"] = TAG_P
    fx = _mirror(tmp_path, seeds, launch_override=relabel)
    bt, facts = Y.read(_adapter(fx, seeds))
    why = Y.refused(facts, TAG, NONE)
    assert f"launch tag / key ('{TAG_P}'" in why and TAG_MAIN in why and TAG in why and facts["runs"] == 4
    def relaunch(doc):
        for r in doc["evals"][EVAL]["launch"]:
            if r["cell"] == RELU and r["tag"] == TAG:
                r["launched_as"]["row_dir"] = Y.ROW_DIR
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "relaunch"), launch_override=relaunch)
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    assert "the launch set says the row was launched as" in Y.refused(facts2, TAG, RELU) and facts2["runs"] == 4
    def remodel(doc):
        for r in doc["evals"][EVAL]["launch"]:
            if r["cell"] == PROMPTED and r["tag"] == TAG:
                r["model_key"] = "llama8b"
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "remodel"), launch_override=remodel)
    bt3, facts3 = Y.read(_adapter(fx3, seeds))
    assert "names model key 'llama8b', not this store's gemma4e4b" in Y.refused(facts3, TAG, PROMPTED) and facts3["runs"] == 4
    # the canary `none` refused: the seed rows' OWN pairing is gone (`anchor_refused`), and the anchor currency re-anchors them on the paired tag's `none`
    # -- the same source, another standup, current on the frame (sibling_none, cross_standup) -- naming the refused own anchor beside it
    for k in (RELU, PROMPTED):
        blk = Y.run_of(bt, k)["extra"]["anchor"]
        assert blk["state"] == "contrasted" and blk["reanchored"] is True and blk["own_anchor_state"] == B.ANCHOR_REFUSED and blk["anchor_view_legacy"] == TAG_P, blk
        assert blk["currency_own_reason"] == "none_declared" and blk["cross_standup"] is True


# --------------------------------------------------------------------------------------------
# object identity by pin key; the paired tag
# --------------------------------------------------------------------------------------------

def test_a_canary_rows_fitted_key_resolves_through_the_rosters_main_selection(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    bt, facts = Y.read(ad)
    r = Y.run_of(bt, RELU)
    assert r["object_id"] == Y.OBJ and r["extra"]["launch_set"]["pin_key"] == Y.STAGED and r["extra"]["tag_legacy_original"] == TAG_MAIN
    res = r["extra"]["object_route"]
    assert (res["route"], res["pin_key"], res["staged_id"], res["harness_roster_selection"]) == ("harness_roster_main_selection", Y.STAGED, Y.STAGED, "main")
    assert f"roster targets.{MK}.views.main.cells[{RELU}] -> {Y.STAGED}" in res["source"] and "canary row's objects are the roster's main selection's" in res["source"]
    assert "main selection's for the family" in res["rule"] and r["extra"]["object_resolution"] == res["source"]
    obj, op = ad.resolve_object(RELU, SEL, pin_key=Y.STAGED, model_key=MK, roster_selection="main")
    assert (obj.object_id, obj.staged_id, obj.object_kind) == (Y.OBJ, Y.STAGED, "relu_steer") and op is None
    # the launch set's pin key must be the main selection's staged id
    def repin(doc):
        for row in doc["evals"][EVAL]["launch"]:
            if row["cell"] == RELU:
                row["pin_key"] = f"base64s1__{RELU}"
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "repin"), launch_override=repin)
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    why = Y.refused(facts2, TAG, RELU)
    assert f"pin key 'base64s1__{RELU}' is not the roster's staged id '{Y.STAGED}' at main" in why and facts2["runs"] == 4


def test_an_s1_rows_fitted_key_resolves_through_the_launch_sets_pin_key_and_the_harness_pins(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, canary_layout="s1")
    ad = _adapter(fx, seeds)
    bt, facts = Y.read(ad)
    r = Y.run_of(bt, RELU)
    assert r["object_id"] == Y.OBJ and r["extra"]["launch_set"]["pin_key"] == Y.STAGED and r["extra"]["launched_roster"]["roster_entry"]["object"]["staged_id"] == Y.STAGED
    res = r["extra"]["object_route"]
    assert (res["route"], res["pin_key"], res["staged_id"], res["harness_roster_selection"]) == ("launch_set_pin_key", Y.STAGED, Y.STAGED, None)
    assert "stages the same id" in res["source"] and "D565" in res["rule"] and "no s1 / s1p selection" in res["rule"]
    fam = r["extra"]["appendix_family"]
    assert fam == {"model_key": MK, "model_id": "gemma4e4b_it", "served_hf_id": Y.MODEL_HF, "served_revision": Y.revision_of("gemma4e4b_it"), "seed_tags": [TAG], "rule": B.APPENDIX_FAMILY_RULE}
    assert r["transfer_kind"] == "base_to_instruct" and r["model_revision"] == Y.revision_of("gemma4e4b_it") and r["extra"]["served_model_id_hf"] == Y.MODEL_HF and r["extra"]["fit_model_id_hf"] == Y.FIT_HF
    assert Y.run_at(bt, TAG, NONE)["extra"]["object_route"]["route"] == "none" and Y.run_of(bt, PROMPTED)["extra"]["object_route"]["route"] == "si_texts"
    # the resolution itself: the pin key is the staged id; without one nothing can be named
    obj, op = ad.resolve_object(RELU, SEL, pin_key=Y.STAGED, model_key=MK)
    assert (obj.object_id, obj.staged_id, obj.object_kind, obj.paper_label) == (Y.OBJ, Y.STAGED, "relu_steer", "constitution_relu") and op is None
    with pytest.raises(IC.Unresolved) as exc:
        ad.resolve_object(RELU, SEL, model_key=MK)
    assert "names no pin key" in exc.value.why and "no 's1' selection" in exc.value.why
    with pytest.raises(IC.Unresolved) as exc:
        ad.resolve_object(RELU, SEL, pin_key=f"base64s1__{RELU}", model_key=MK)
    assert "carry no object_sha256 under gemma4e4b.base64s1__" in exc.value.why
    assert r["extra"]["three_numbers"]["producer_record_agrees"] is True, "the record's cell_slug spells the store key's `--` token; the reader accepts it beside the object key"
    # the three-seed family still resolves through its roster selection (model_key defaults to qwen35_4b): the fixture's qwen roster stages s2 but its
    # pins are empty, and it has no s1 selection at all -- both surface as the roster / pins refusal, never the pin-key route
    with pytest.raises(IC.Unresolved) as exc:
        ad.resolve_object(RELU, "s1")
    assert "the roster's 's1' selection / the pins name no object" in exc.value.why
    with pytest.raises(IC.Unresolved) as exc:
        ad.resolve_object(RELU, "s2", model_key="qwen35_4b")
    assert "pins name no object_sha256 for staged id 'base64s2__" in exc.value.why


@pytest.mark.parametrize("label", ["eb8", "eb64"])
def test_a_composed_key_the_harness_pins_do_not_carry_resolves_through_the_stores_tree_pins_by_object_sha256(tmp_path, seeds, label):
    """The composed s1p objects ARE in the mirrored pins, keyed `paired_eb<batch>__<cell>` (gemma / nemotron eb8, llama eb64) while the
    repo's harness pins do not carry them. A composed store whose lane_roster.json names an object_sha256 resolves through ITS TREE'S pins by that sha (any key
    under the model's block); the key found rides extra.launched_pins_key with pins_key_rule, the route is named, and the launch set's pin key must be that key.
    A fitted `arm` key the harness pins lack still refuses by name; a composed sha in no pins refuses by name; a bare-name hit stays as it was."""
    staged_p = f"paired_{label}__{COMPOSED}"
    fx = _mirror(tmp_path, seeds, composed_batch_label=label, harness_pins_without=(staged_p,))
    assert fx["staged_p"] == staged_p and staged_p not in json.load(open(fx["pins"]))[MK] and staged_p in json.load(open(os.path.join(fx["root"], B.RECORDS_DIR, B.PINS_FILE)))[MK]
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    r = Y.run_of(bt, COMPOSED)
    assert r["object_id"] == Y.OBJ_P and r["extra"]["launched_pins_key"] == staged_p and r["extra"]["pins_key_rule"] == B.PINS_KEY_RULE
    assert B.PINS_KEY_RULE == "matched by object_sha256; the pins spell a composed cell paired_eb<batch>__<cell>"
    route = r["extra"]["object_route"]
    assert (route["route"], route["pin_key"], route["staged_id"]) == (B.OBJECT_ROUTE_LAUNCHED_PINS_BY_SHA, staged_p, staged_p) and "lane_roster.json:object_sha256 ->" in route["source"] and B.PINS_FILE in route["source"]
    assert r["extra"]["launch_set"]["pin_key"] == staged_p and r["extra"]["launched_roster"]["roster_entry"]["object"]["staged_id"] == staged_p
    for k in (RELU,):
        assert "launched_pins_key" not in Y.run_of(bt, k)["extra"] and Y.run_of(bt, k)["extra"]["object_route"]["route"] == "harness_roster_main_selection", "a bare-name hit stays as it was"
    # the sha in NO pins (the harness pins lack the key, the tree pins lack the sha): refused by name, the siblings file
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "nowhere"), composed_batch_label=label, harness_pins_without=(staged_p,), launched_pins_without=(staged_p,))
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    why = Y.refused(facts2, TAG_P, COMPOSED)
    assert f"carry no object_sha256 under {MK}.{staged_p}" in why and f"records/{B.PINS_FILE} names no entry under {MK} with the launched object_sha256 {Y.OBJ_P[:16]}" in why and facts2["runs"] == 4
    assert {r["campaign_cell_id_legacy"] for r in bt2["run"]} == {NONE, RELU, PROMPTED} and sum(1 for r in bt2["run"] if r["view_legacy"] == TAG_P) == 1
    # a fitted `arm` key the harness pins lack: refused as before (the by-sha route is the composed objects')
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "arm"), canary_layout="s1", harness_pins_without=(Y.STAGED,))
    bt3, facts3 = Y.read(_adapter(fx3, seeds))
    assert f"carry no object_sha256 under {MK}.{Y.STAGED}" in Y.refused(facts3, TAG, RELU) and "names no entry" not in Y.refused(facts3, TAG, RELU) and facts3["runs"] == 4
    # the launch set's pin key must be the key the tree pins spell
    def other_pin_key(doc):
        for ev in doc["evals"].values():
            for row in ev["launch"]:
                if row["cell"] == COMPOSED:
                    row["pin_key"] = f"paired_eb99__{COMPOSED}"
    fx4 = _mirror(tmp_path, seeds, root=str(tmp_path / "pinkey"), composed_batch_label=label, harness_pins_without=(staged_p,), launch_override=other_pin_key)
    bt4, facts4 = Y.read(_adapter(fx4, seeds))
    assert f"the launch set's pin key 'paired_eb99__{COMPOSED}' is not the roster's staged id '{staged_p}'" in Y.refused(facts4, TAG_P, COMPOSED) and facts4["runs"] == 4


@pytest.mark.parametrize("layout", ["all", "s1"])
def test_a_roster_main_selection_staging_another_id_than_the_pin_key_refuses_the_row(tmp_path, seeds, layout):
    """On the canary row the roster's main selection IS the resolution (its staged id is not pinned, and the pin key is not it); on the s1 row it is the
    cross-check of the pin-key route. Either way the row refuses by name and its siblings file."""
    fx = _mirror(tmp_path, seeds, canary_layout=layout, roster_main_override={RELU: f"itdoc__{RELU}", Y.CONTROL_RELU: Y.STAGED_CTL})
    bt, facts = Y.read(_adapter(fx, seeds))
    why = Y.refused(facts, TAG, RELU)
    if layout == "all":
        assert f"pins name no object_sha256 for staged id 'itdoc__{RELU}'" in why and "'main' selection" in why
    else:
        assert "`main` selection stages" in why and f"'itdoc__{RELU}'" in why and f"'{Y.STAGED}'" in why
    assert facts["runs"] == 4 and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, PROMPTED, COMPOSED}


def test_the_paired_tags_own_none_anchors_the_composed_row_with_paired_of_from_the_tag_grammar(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = Y.read(_adapter(fx, seeds))
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    none_s, none_p, cp = runs[(TAG, NONE)], runs[(TAG_P, NONE)], runs[(TAG_P, COMPOSED)]
    assert none_p["extra"]["launch_set"]["class"] == "anchor" and none_p["extra"]["standup_replicate_of_view"] == TAG and none_p["run_id"] != none_s["run_id"]
    assert none_p["extra"]["paired_of"] == "s1" and none_p["extra"]["roster_paired_of"] is None and "tag grammar" in none_p["extra"]["paired_of_source"] and "no s1p selection" in none_p["extra"]["paired_of_source"]
    assert "tag_legacy_original" not in none_p["extra"], "the paired tag's none is no canary"
    c = {cn["run_id"]: cn for cn in bt["run_contrast"]}[cp["run_id"]]
    assert c["anchor_run_id"] == none_p["run_id"] and c["anchor_kind"] == "same_standup_none" and c["extra"]["cross_standup"] is False and c["extra"]["anchor_view_legacy"] == TAG_P
    assert cp["object_id"] == Y.OBJ_P and cp["extra"]["object_route"]["pin_key"] == Y.STAGED_P and "does not stage" in cp["extra"]["object_route"]["source"]
    assert cp["extra"]["own_anchor"] == f"{TAG_P}/none" and cp["transfer_kind"] == "base_to_instruct"
    v_p, _b = _metrics(bt, none_p["run_id"])
    assert (v_p["k"], v_p["n"]) == (Y.counts(fx["paired_none_spec"])["correct"], 200)


def test_without_the_paired_none_the_composed_row_pairs_across_standups_with_the_canary(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none=False)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 4
    cp = Y.run_of(bt, COMPOSED)
    c = {cn["run_id"]: cn for cn in bt["run_contrast"]}[cp["run_id"]]
    assert c["anchor_run_id"] == Y.run_at(bt, TAG, NONE)["run_id"] and c["extra"]["cross_standup"] is True and cp["extra"]["cross_standup_anchor"] == f"{TAG}/none"


# --------------------------------------------------------------------------------------------
# the roster set: one launched roster per producer tree, matched by hash (ROSTER_SET_RULE)
# --------------------------------------------------------------------------------------------

def test_a_roster_set_matches_each_store_by_hash_and_records_the_file_tree_and_pins_copy(tmp_path, seeds):
    """Two producer trees, two launched rosters: the canary row's stores (tree TREE) cite `bcb_bfcl_200_widen_roster.<sha7>.yaml` and the per-tree pins copy
    (the plain pins lack their ReLU object), the s1p stores (tree TREE_S1) cite the plain (latest) roster; `records/rosters.json` maps both hashes. Every
    store files against ITS roster and pins, the file and tree recorded; nothing is a "revision mismatch" any more."""
    fx = _mirror(tmp_path, seeds, roster_set=True)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = [os.path.basename(i.uri) for i in inputs]
    assert f"{B.ROSTERS_INDEX_FILE}#{TAG}" in names and f"{Y.ROSTER_CANARY_FILE}#{TAG}" in names and f"{Y.PINS_CANARY_FILE}#{TAG}" in names and f"{B.ROSTER_FILE}#{TAG}" in names
    bt, facts = Y.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    assert [x["file"] for x in facts["roster_set"]] and {x["roster_sha256"]: (x["file"], x["tree_sha"]) for x in facts["roster_set"]} == \
        {fx["canary_roster_sha256"]: (Y.ROSTER_CANARY_FILE, Y.TREE), fx["roster_sha256"]: (B.ROSTER_FILE, Y.TREE_S1)}
    assert facts["launched_pins_per_tree"] == [Y.PINS_CANARY_FILE] and facts["roster_set_rule"] == B.ROSTER_SET_RULE
    for k in (NONE, RELU, PROMPTED):
        r = Y.run_at(bt, TAG, k)
        lr = r["extra"]["launched_roster"]
        assert (lr["roster_file"], lr["tree_sha"], lr["roster_sha256"]) == (Y.ROSTER_CANARY_FILE, Y.TREE, fx["canary_roster_sha256"]) and lr["roster_set_rule"] == B.ROSTER_SET_RULE
        assert "roster_revision_note" not in lr and lr["roster_record"].endswith(Y.ROSTER_CANARY_FILE) and r["code_sha"] == Y.TREE
        assert r["extra"]["launched_pins_record"].endswith(Y.PINS_CANARY_FILE) and r["extra"]["launched_pins_tree_sha"] == Y.TREE, k
    for k in (NONE, COMPOSED):
        r = Y.run_at(bt, TAG_P, k)
        lr = r["extra"]["launched_roster"]
        assert (lr["roster_file"], lr["tree_sha"], lr["roster_sha256"]) == (B.ROSTER_FILE, Y.TREE_S1, fx["roster_sha256"]) and r["code_sha"] == Y.TREE_S1
        assert r["extra"]["launched_pins_record"].endswith(B.PINS_FILE) and "launched_pins_tree_sha" not in r["extra"], "the plain pins for a tree with no copy"
    assert Y.run_at(bt, TAG, RELU)["object_id"] == Y.OBJ and Y.run_at(bt, TAG, RELU)["extra"]["store_suffix_legacy"].endswith(Y.TREE[:7]) and Y.run_at(bt, TAG_P, COMPOSED)["extra"]["store_suffix_legacy"].endswith(Y.TREE_S1[:7])


def test_a_store_citing_a_roster_hash_the_set_lacks_refuses_by_name_and_the_siblings_file(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, roster_set=True, roster_missing_for=(RELU,))
    bt, facts = Y.read(_adapter(fx, seeds))
    why = Y.refused(facts, TAG, RELU)
    assert "matches no mirrored roster" in why and f"{B.ROSTERS_INDEX_FILE} names 2" in why and Y.ROSTER_CANARY_FILE in why and B.ROSTER_FILE in why and facts["runs"] == 4
    assert {r["campaign_cell_id_legacy"] for r in bt["run"] if r["view_legacy"] == TAG} == {NONE, PROMPTED}


def test_a_roster_index_naming_a_file_whose_sha_differs_or_that_is_not_mirrored_refuses_the_stores_citing_it(tmp_path, seeds):
    def point_at_plain(index):
        index["rosters"][fx_sha["canary"]]["file"] = B.ROSTER_FILE
    fx_sha = {}
    fx0 = _mirror(tmp_path, seeds, roster_set=True, root=str(tmp_path / "probe"))
    fx_sha["canary"] = fx0["canary_roster_sha256"]
    fx = _mirror(tmp_path, seeds, roster_set=True, root=str(tmp_path / "sha"), rosters_index_override=point_at_plain)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["runs"] == 2 and {r["view_legacy"] for r in bt["run"]} == {TAG_P}
    for k in (NONE, RELU, PROMPTED):
        why = Y.refused(facts, TAG, k)
        assert "is unusable" in why and f"but the file's sha256 is {fx['roster_sha256'][:12]}" in why, k
    def rename(index):
        index["rosters"][fx_sha["canary"]]["file"] = "bcb_bfcl_200_widen_roster.deadbee.yaml"
    fx2 = _mirror(tmp_path, seeds, roster_set=True, root=str(tmp_path / "missing"), rosters_index_override=rename)
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    assert "which is not mirrored under records/" in Y.refused(facts2, TAG, NONE) and facts2["runs"] == 2
    def empty(index):
        index["rosters"] = {}
    fx3 = _mirror(tmp_path, seeds, roster_set=True, root=str(tmp_path / "empty"), rosters_index_override=empty)
    bt3, facts3 = Y.read(_adapter(fx3, seeds))
    assert facts3["runs"] == 0 and facts3["n_refusals"] == 5 and all("names no roster" in x for x in facts3["refusals"])


# --------------------------------------------------------------------------------------------
# the model set; a second family; not run
# --------------------------------------------------------------------------------------------

def test_a_store_of_another_family_refuses_its_group_in_both_directions(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, qwen_store=True)
    ad = _adapter(fx, seeds)
    bt, facts, groups = Y.read_group(ad, "qwen35_4b_s2")
    assert groups == [TAG, "qwen35_4b_s2"] and "run" not in bt
    assert len(facts["refusals"]) == 1 and "model key 'qwen35_4b' is not one of this source's families (gemma4e4b, llama8b, nemotron3nano4b)" in facts["refusals"][0]
    bt_g, facts_g, _g = Y.read_group(ad, TAG)
    assert facts_g["n_refusals"] == 0 and facts_g["runs"] == 5
    # the seedrep reader over the appendix mirror: the gemma group refuses, naming its own family
    sr = X.adapter(SR.BigcodebenchSeedrepAdapter, fx, seeds)
    bt_s, facts_s, groups_s = Y.read_group(sr, TAG)
    assert groups_s == [TAG, "qwen35_4b_s2"] and "run" not in bt_s and "model key 'gemma4e4b' is not one of this source's families (qwen35_4b)" in facts_s["refusals"][0]


def test_a_second_appendix_family_reads_in_its_own_group_with_its_own_model_and_thinking_state(tmp_path, seeds):
    """llama8b was served WITHOUT the tool-call route (no BFCL row): the BigCodeBench reader RECORDS the route (`extra.tool_calls` false) and files the
    row; the BFCL reader requires it (its task is tool calls) and refuses such a store by name."""
    fx = _mirror(tmp_path, seeds, llama_none=True, llama_tool_calls=False)
    ad = _adapter(fx, seeds)
    bt, facts, groups = Y.read_group(ad, Y.LLAMA_TAG)
    assert groups == [TAG, Y.LLAMA_TAG] and facts["n_refusals"] == 0 and facts["runs"] == 1, facts["refusals"]
    r = bt["run"][0]
    assert r["extra"]["tool_calls"] is False and M.SPEC.tool_calls_required is False and SR.SPEC.tool_calls_required is False
    assert all(x["extra"]["tool_calls"] is True for x in Y.read_group(ad, TAG)[0]["run"]), "gemma's stores attest the route; the flag is the store's, never a constant"
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="llama8b_it")
    assert r["model_id"] == "llama8b_it" and r["view_legacy"] == Y.LLAMA_TAG and r["instrument_id"] == inst["instrument_id"] and r["model_revision"] == Y.revision_of("llama8b_it")
    assert r["thinking_mode"] == "not_available", "the no-switch family (models.yaml thinking_channel not_available; the user's ruling of 2026-09-18)"
    assert r["extra"]["appendix_family"]["model_key"] == "llama8b" and r["extra"]["served_model_id_hf"] == Y.LLAMA_HF and "tag_legacy_original" not in r["extra"]
    assert r["extra"]["decode_policy_ref_legacy"] == "llama8b_it|bigcodebench|off|protocol@v1", "the reader's own resolution reads the store's `off`; the derived match reads the run's state"
    assert r["producer_run_key_legacy"] == f"{Y.LLAMA_TAG}/{fx['keys'][f'{Y.LLAMA_TAG}/{NONE}']}" and "run_contrast" not in bt
    assert facts["tags"] == [Y.LLAMA_TAG, f"{Y.LLAMA_TAG}p"] and facts["model_id"] == "llama8b_it"


def test_a_launch_set_row_with_no_store_files_nothing(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, llama_none="not_run")
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    groups = {ad.comparison_group(i) for i in inputs}
    assert groups == {TAG}, "no llama store: no llama group -- the gate record alone names no group that holds a store"
    bt, facts = Y.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 0, "the llama row is another group's; this group's table is complete"
    # the seed tag's `none` not run: the rows' OWN pairing waits (`anchor_not_yet_run`, the same rule as the three-seed family's) -- on the s1 row and on the
    # canary row -- and the anchor currency re-anchors them on the paired tag's `none` (the same source, another standup; sibling_none) until it lands
    for layout, sub in (("s1", "nr"), ("all", "nr_canary")):
        fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / sub), not_run=(NONE,), canary_layout=layout)
        bt2, facts2 = Y.read(_adapter(fx2, seeds))
        assert facts2["n_refusals"] == 0 and facts2["runs"] == 4 and facts2["n_not_run"] == 1 and facts2["not_run"][0]["tuple_id"] == f"{SEL}/{NONE}", (layout, facts2["refusals"])
        assert facts2["not_run"][0]["gate_verdict"] == "FAIL", "the FAIL record is found under the canary row too"
        blk = Y.run_of(bt2, RELU)["extra"]["anchor"]
        assert blk["own_anchor_state"] == B.ANCHOR_NOT_YET_RUN and blk["state"] == "contrasted" and blk["reanchored"] is True and blk["anchor_view_legacy"] == TAG_P
        assert facts2["anchor_states"] == {"own_current": 1, "sibling_same_source": 2} and facts2["contrasts"][f"{TAG}|{RELU}"]["state"] == "contrasted"


def test_the_launch_sets_row_table_is_launch_plus_carried_for_the_appendix_too(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, launch_layout="carried")
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    assert Y.run_at(bt, TAG, NONE)["extra"]["launch_set"]["census_status"] == "carry" and Y.run_at(bt, TAG, NONE)["extra"]["tag_legacy_original"] == TAG_MAIN


# --------------------------------------------------------------------------------------------
# the decode policy: the registered rows, and end to end the harvester's derived match
# --------------------------------------------------------------------------------------------

def test_the_protocol_rows_exist_for_every_appendix_model_and_the_reader_names_them(tmp_path, seeds):
    pf = DP.load(None)
    qwen = {ev: pf.by_id[f"qwen35_4b_it|{ev}|off|protocol@v1"] for ev in ("bigcodebench", "bfcl")}
    for mid in APPENDIX_MODEL_IDS:
        for ev, (t, cap) in (("bigcodebench", (0.0, "4096")), ("bfcl", (0.001, "1024"))):
            pid = f"{mid}|{ev}|off|protocol@v1"
            p = pf.by_id[pid]
            assert (p["model_id"], p["eval_id"], p["thinking_mode"], p["mode"], p["is_default"]) == (mid, ev, "off", "protocol", 1), pid
            assert (p["temperature"], p["top_p"], p["top_k"], p["seed_rule"], p["max_new_tokens_rule"]) == (t, 1.0, 0, "none", cap), pid
            assert p["source_citation"] == qwen[ev]["source_citation"] and p["ruling_ref"] == qwen[ev]["ruling_ref"] == "decode-protocol-rows-2026-09-11"
            assert p["wildcard_expanded_from"] == f"*|{ev}|off|protocol" == qwen[ev]["wildcard_expanded_from"], "one wildcard row of the seed, expanded onto every served model"
            resolved = pf.resolve(mid, eval_id=ev, thinking_mode="off")
            assert resolved.policy_id == pid and resolved.numbers() == pf.resolve("qwen35_4b_it", eval_id=ev, thinking_mode="off").numbers() and resolved.decode_source == "greedy"
    assert pf.by_id["llama8b_it|bigcodebench|not_available|protocol@v1"]["same_values_as"] == "llama8b_it|bigcodebench|off|protocol@v1"
    fx = _mirror(tmp_path, seeds)
    bt, facts = Y.read(_adapter(fx, seeds))
    for tag, okey in ((TAG, NONE), (TAG, RELU), (TAG, PROMPTED), (TAG_P, COMPOSED), (TAG_P, NONE)):
        r = Y.run_at(bt, tag, okey)
        assert r["decode_policy_match"] == "no_policy" and "decode_policy_id" not in r and r["decode_seed_rule"] == "none" == M.DECODE_SEED_RULE
        assert r["extra"]["decode_policy_ref_legacy"] == "gemma4e4b_it|bigcodebench|off|protocol@v1" and r["extra"]["decode_policy_reader_match"] == "exact by the stated sampling numbers"
        assert r["decode_source"] == "greedy" and r["instrument_match"] == "match" and r["decode_sent_json"] == {"temperature": 0.0, "top_p": 1.0, "top_k": 0}


def _registry_root(tmp_path, seeds, objects) -> str:
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in objects:
        r = LC.standin_object_record(o, seeds, F.producer(), now, "gemma4e4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def test_runs_reach_ingested_and_the_derived_decode_policy_match_names_each_models_protocol_row(tmp_path, seeds):
    """Two passes (emission 2, the anchor currency): the first files every run and parks every comparison (the `none` runs are not folded
    yet), the fold makes them current and the decision input moves, the second re-reads the gemma group and files the three contrasts."""
    fx = _mirror(tmp_path, seeds, llama_none=True)
    ad = _adapter(fx, seeds)
    objects = [ad.resolve_object(RELU, SEL, pin_key=Y.STAGED, model_key=MK, roster_selection="main")[0], ad.resolve_object(COMPOSED, SEL_P, pin_key=Y.STAGED_P, model_key=MK)[0]]
    root = _registry_root(tmp_path, seeds, objects)
    rep0, ing0, ing0b = F.run_and_ingest(root, ad, now="2026-09-23T08:00:00.000Z")
    assert rep0.groups_refused == 0 and ing0.refused + ing0b.refused == 0, ([r.error for r in rep0.results], ing0.refusals, ing0b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [(6,)]
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, model_id, lifecycle_state, thinking_mode, trial_index, seed_base, json_extract(extra_json, '$.tag_legacy_original') "
                     "FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (M.SOURCE,))
    assert rows == [(TAG, RELU, "gemma4e4b_it", "ingested", "off", 0, None, TAG_MAIN),
                    (TAG, NONE, "gemma4e4b_it", "ingested", "off", 0, None, TAG_MAIN),
                    (TAG, PROMPTED, "gemma4e4b_it", "ingested", "off", 0, None, TAG_MAIN),
                    (TAG_P, COMPOSED, "gemma4e4b_it", "ingested", "off", 0, None, None),
                    (TAG_P, NONE, "gemma4e4b_it", "ingested", "off", 0, None, None),
                    (Y.LLAMA_TAG, NONE, "llama8b_it", "ingested", "not_available", 0, None, None)], rows
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(3,)]
    # the DERIVED decode-policy match (decode_match.stamp_run_rows): gemma's rows name the gemma protocol row, llama's the `not_available` twin (its run state)
    assert F.q(root, "SELECT model_id, decode_seed_rule, decode_policy_match, decode_policy_id, COUNT(*) FROM run WHERE lane_source_legacy = ? GROUP BY 1, 2, 3, 4 ORDER BY 1", (M.SOURCE,)) == \
        [("gemma4e4b_it", "none", "exact", "gemma4e4b_it|bigcodebench|off|protocol@v1", 5), ("llama8b_it", "none", "exact", "llama8b_it|bigcodebench|not_available|protocol@v1", 1)]
    m = F.q(root, "SELECT r.view_legacy, r.campaign_cell_id_legacy, rm.computation_id <> '', rm.value FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                  "WHERE r.lane_source_legacy = ? AND r.campaign_cell_id_legacy = ? AND rm.metric_id = 'acc' ORDER BY 1, 3 DESC", (M.SOURCE, NONE))
    c_s, c_p, c_l = Y.counts(fx["specs"][NONE]), Y.counts(fx["paired_none_spec"]), Y.counts(fx["llama_spec"])
    assert m == [(TAG, NONE, 1, c_s["acc"]), (TAG, NONE, 0, c_s["correct"] / 200), (TAG_P, NONE, 1, c_p["acc"]), (TAG_P, NONE, 0, c_p["correct"] / 200),
                 (Y.LLAMA_TAG, NONE, 1, c_l["acc"]), (Y.LLAMA_TAG, NONE, 0, c_l["correct"] / 200)]
    rep2, _i3, _i4 = F.run_and_ingest(root, _adapter(fx, seeds), now="2026-09-23T22:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_version_and_model_set():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.BigcodebenchAllmodelsAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "bigcodebench_allmodels"
    assert AB.emission_version_of(M.BigcodebenchAllmodelsAdapter) == "2" and AB.emission_version_of(SR.BigcodebenchSeedrepAdapter) == "3" and inst.rng_scope == "none"
    assert inst.roots == (M.ROOT_DEFAULT,) and M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bigcodebench_allmodels")) and M.ROOT_DEFAULT != SR.ROOT_DEFAULT
    assert (M.EVAL_ID, M.TASK, M.INSTRUMENT_SEED, M.CAP, M.LADDER, M.PROTOCOL_TEMPERATURE) == (SR.EVAL_ID, SR.TASK, SR.INSTRUMENT_SEED, SR.CAP, SR.LADDER, SR.PROTOCOL_TEMPERATURE)
    assert M.SPEC.record_task == "bigcodebench_200" and M.SPEC.clamp_override is False and M.SPEC.source == M.SOURCE and M.SPEC.name == M.NAME
    assert M.RECOUNT_ALGORITHM == "sisteer.registry.adapters.bigcodebench_allmodels.recount" and M.recount_computation_id() != SR.recount_computation_id()
    assert M.run_id_for(TAG, "k", NONE, "i") == I.run_id(LC.CAMPAIGN, f"{TAG}/k", NONE, "bigcodebench", "i") == SR.run_id_for(TAG, "k", NONE, "i"), "the same producer key, eval and instrument mint the same id whichever reader files it"
    assert not isinstance(getattr(B.BcbwSeedrepReader, "name", None), str)
